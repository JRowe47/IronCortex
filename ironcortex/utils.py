from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EPS_LOG


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ==========================================================


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm with learnable scale (weight).

    x: [..., d] -> output: [..., d]
    Keeps magnitudes comparable, which is important because FF goodness
    is defined on post-norm activations (mean(x^2)).
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / s)


def KWTA(
    x: torch.Tensor, k: int, *, soft: bool = False, temp: float = 1.0
) -> torch.Tensor:
    """k-Winners-Take-All along the last dimension.

    Keeps the top-k (by |value|) units and zeros the rest. When ``soft`` is
    True, a sigmoid mask with temperature ``temp`` is used instead of a hard
    cutoff during warmup to preserve gradients.
    """
    d = x.shape[-1]
    if k <= 0:
        return torch.zeros_like(x)
    k = max(1, min(k, d))
    absx = x.abs()
    # threshold at the kth largest absolute value (ties keep extra units)
    thresh = torch.topk(absx, k, dim=-1).values[..., -1:].expand_as(absx)
    if soft:
        gate = torch.sigmoid((absx - thresh) / temp)
        return x * gate
    mask = absx >= thresh
    return x * mask


def masked_mean(Y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Masked mean over time: Y:[B,T,d], M:[B,T] (bool) -> [B,d]"""
    M = M.to(Y.dtype)
    s = (Y * M.unsqueeze(-1)).sum(dim=1)
    z = M.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return s / z


def extract_spans(mask_1d_bool: torch.Tensor) -> List[Tuple[int, int]]:
    """Turn a 1-D boolean mask into [(start, end)] half-open spans."""
    spans, in_span, start = [], False, 0
    seq = mask_1d_bool.tolist() + [False]  # sentinel
    for i, m in enumerate(seq):
        if m and not in_span:
            in_span, start = True, i
        if in_span and not m:
            spans.append((start, i))
            in_span = False
    return spans


def pad_batch(X_list, C_list, UM_list):
    """Right-pad variable-length sequences for batching.

    X_list: List[Tensor[T_i,d]]
    C_list: List[Tensor[T_i,coord_dim]]
    UM_list: List[BoolTensor[T_i]]
    """
    B = len(X_list)
    Tm = max(x.shape[0] for x in X_list)
    d = X_list[0].shape[1]
    cdim = C_list[0].shape[1]
    device = X_list[0].device
    X = torch.zeros(B, Tm, d, device=device)
    C = torch.zeros(B, Tm, cdim, device=device)
    UM = torch.zeros(B, Tm, dtype=torch.bool, device=device)
    for b, (x, c, um) in enumerate(zip(X_list, C_list, UM_list)):
        T = x.shape[0]
        X[b, :T] = x
        C[b, :T] = c
        UM[b, :T] = um
    return X, C, UM


def nms_topk(scores: torch.Tensor, k: int, neighbors: List[List[int]]) -> List[int]:
    """Greedy neighbor-suppressed top-k (hex-NMS style).

    scores: [R] tensor
    neighbors: adjacency list; neighbors[i] gives indices of i's neighbors.
    """
    order = torch.argsort(scores, descending=True).tolist()
    selected, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        selected.append(i)
        if len(selected) >= k:
            break
        suppressed.add(i)
        for nb in neighbors[i]:
            suppressed.add(nb)
    return selected


def uncertainty_from_logits(logits: torch.Tensor) -> float:
    """Scalar uncertainty proxy from logits: 1 - mean(max probability)."""
    p = F.softmax(logits, dim=-1)
    return float((1.0 - p.max(dim=-1).values.mean()).item())


def context_logprob(logits: torch.Tensor) -> torch.Tensor:
    """Heuristic branch bonus: negative entropy of the logits."""
    p = F.softmax(logits, dim=-1)
    ent = -(p * (p + EPS_LOG).log()).sum(dim=-1)
    return -ent.mean()  # higher is better


def schedule_burst(u_mean: float, v_hat: float, R: int) -> int:
    """Simple burst schedule: more compute when uncertainty or value is high."""
    if (u_mean > 0.30) or (v_hat > 0.0):
        return max(1, R // 8)
    return 0


def should_halt(
    branch_scores: List[torch.Tensor], u_mean: float, k: int, K_inner: int
) -> bool:
    """Early halting when confident and enough micro-steps done."""
    return (u_mean < 0.05) and (k >= max(1, K_inner // 2))


# ==========================================================
# 3) Goodness & Region FF State
# ==========================================================


def goodness(h_stack: torch.Tensor) -> torch.Tensor:
    """Scalar FF goodness on a stack of post-norm activations: mean of squared."""
    h = h_stack.float()
    return (h.pow(2).mean(dim=-1)).mean()


class RegionFFState(nn.Module):
    """Per-region FF threshold τ (EMA)."""

    def __init__(self, init_tau: float = 0.0):
        super().__init__()
        self.register_buffer("tau", torch.tensor(init_tau, dtype=torch.float32))

    def update_tau(
        self,
        g_pos_mean: torch.Tensor,
        mean_prec: torch.Tensor | None = None,
        *,
        alpha: float = 0.01,
        kappa: float = 0.0,
        target_prec: float = 1.0,
    ) -> None:
        tau_target = g_pos_mean.to(self.tau.dtype)
        if mean_prec is not None:
            tau_target = tau_target + kappa * (
                mean_prec.to(self.tau.dtype) - target_prec
            )
        self.tau = (1 - alpha) * self.tau + alpha * tau_target
