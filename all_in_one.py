"""
IRON-CORTEX: A reasoning-native, always-on RWKV+Diffusion cortex with Iron RoPE

This single-file reference implements the end-to-end architecture you specified:
- Sparse modular **regions** arranged on a hex-like graph
- **RWKV**-style region cells with Δt skip accounting (EMA fast weights, per-channel decay)
- **RMSNorm** everywhere + **KWTA** inside regions for sharper FF margins
- **Diffusion / mask-predict** inner steps with localized pseudo tokens (anchors)
- **Forward-Forward (FF)** goodness learning (push g_pos > τ and g_neg < τ per region)
- **RTD head** (replaced-token detection) for hard negatives
- Tiny **Planner → Critic → Verifier** trio and a **Workspace** (scratchpad) region
- **Gate** that allocates compute by predicted Δgoodness (usefulness) & uncertainty
- **Iron RoPE** (Fourier-enhanced RoPE) in three places:
    (1) token geometry for the local mixer (bidirectional, mask-focused)
    (2) inner-step "time" conditioning (rotate RWKV v-channels)
    (3) relative Fourier bias on the hex router edges

The file includes:
- All modules (norms, sparsity, Iron utilities, attention, router, RWKV regions, heads)
- A CortexReasoner container with a value-biased gate and the inner reasoning loop
- A basic Forward-Forward training step that mixes FF, RTD, and denoising losses
- A simple mask-predict generator
- Extensive docstrings and comments for clarity and ablation

NOTE
----
This is a *reference & research* implementation designed for readability and ablation.
It runs with PyTorch and standard libraries. Many elements are lightweight / simplified
(e.g., token head, router biasing, verifier targets, etc.) to keep the single-file scope.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 0) Utility Layers & Helpers
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


def KWTA(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-Winners-Take-All along the last dimension.

    Keeps the top-k (by |value|) units and zeros the rest. Gradient flows
    through the surviving units only. Stable and simple sparsity primitive.
    """
    d = x.shape[-1]
    if k <= 0:
        return torch.zeros_like(x)
    k = max(1, min(k, d))
    absx = x.abs()
    # threshold at the kth largest absolute value (ties keep extra units)
    thresh = torch.topk(absx, k, dim=-1).values[..., -1:].expand_as(absx)
    mask = (absx >= thresh)
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
    ent = -(p * (p + 1e-9).log()).sum(dim=-1)
    return -ent.mean()  # higher is better


def schedule_burst(u_mean: float, v_hat: float, R: int) -> int:
    """Simple burst schedule: more compute when uncertainty or value is high."""
    if (u_mean > 0.30) or (v_hat > 0.0):
        return max(1, R // 8)
    return 0


def should_halt(branch_scores: List[torch.Tensor], u_mean: float, k: int, K_inner: int) -> bool:
    """Early halting when confident and enough micro-steps done."""
    return (u_mean < 0.05) and (k >= max(1, K_inner // 2))


# ==========================================================
# 1) Iron RoPE: Fourier banks, rotary rotation, relative bias
# ==========================================================

def make_freq_bank(m: int, d_coord: int, kind: str = "log",
                   base: float = 10000.0, sigma: float = 1.0,
                   device=None, dtype=None) -> torch.Tensor:
    """Construct a frequency bank W in R^{m x d_coord}."""
    if kind == "log":
        steps = torch.logspace(math.log(1.0 / base), 0.0, steps=m, base=math.e,
                               device=device, dtype=dtype)
        W = torch.zeros(m, d_coord, device=device, dtype=dtype)
        for j in range(d_coord):
            W[:, j] = steps
        return W
    elif kind == "gaussian":
        return torch.randn(m, d_coord, device=device, dtype=dtype) * sigma
    else:
        raise ValueError(f"Unknown freq kind: {kind}")


def sincos(W: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Fourier features: [sin(Wp), cos(Wp)].

    W: [m, d], p: [..., d] -> [..., 2m]
    """
    Θ = p @ W.T
    return torch.cat([torch.sin(Θ), torch.cos(Θ)], dim=-1)


def rope_rotate_pairs(x: torch.Tensor, cos_th: torch.Tensor, sin_th: torch.Tensor, m_pairs: int) -> torch.Tensor:
    """Rotate the first 2*m_pairs channels in (pairs) by angle arrays.

    x: [..., 2*m_pairs + rest]
    cos_th/sin_th: arrays broadcastable to [..., m_pairs]
    """
    if m_pairs <= 0:
        return x
    *prefix, D = x.shape
    rot = x[..., :2 * m_pairs].reshape(*prefix, m_pairs, 2)
    x0, x1 = rot[..., 0], rot[..., 1]
    c, s = cos_th, sin_th
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    y = torch.stack((y0, y1), dim=-1).reshape(*prefix, 2 * m_pairs)
    return torch.cat([y, x[..., 2 * m_pairs:]], dim=-1)


def relative_fourier_bias(p_q: torch.Tensor, p_k: torch.Tensor,
                          W: torch.Tensor,
                          beta_cos: torch.Tensor,
                          beta_sin: torch.Tensor,
                          scale: float = 1.0) -> torch.Tensor:
    """Relative bias b(Δp) via Fourier features.

    p_q: [B,Tq,d], p_k: [B,Tk,d]
    W: [m,d], beta_cos/sin: [m] or [H,m]
    Returns:
      [B,1,Tq,Tk] if shared across heads, else [B,H,Tq,Tk]
    """
    B, Tq, d = p_q.shape
    Tk = p_k.shape[1]
    Δ = p_q.unsqueeze(2) - p_k.unsqueeze(1)         # [B,Tq,Tk,d]
    S = torch.einsum("bqkd,md->bqkm", Δ, W)         # [B,Tq,Tk,m]
    c, s = torch.cos(S), torch.sin(S)
    if beta_cos.dim() == 2:  # headwise [H,m]
        b = (c.unsqueeze(1) * beta_cos[None, None, None, :] +
             s.unsqueeze(1) * beta_sin[None, None, None, :]).sum(-1)  # [B,H,Tq,Tk]
    else:
        b = (c * beta_cos + s * beta_sin).sum(-1).unsqueeze(1)         # [B,1,Tq,Tk]
    return scale * b


class IronRoPESelfAttention(nn.Module):
    """Bidirectional (default) self-attention with Iron RoPE on q/k and relative Fourier bias.

    Designed for diffusion/mask-predict:
    - `update_mask`: only those token positions are *updated*; others pass-through
    - `coords`: per-token coordinates (e.g., [pos, pos/T])
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 use_iron_rope: bool = True, rope_m: int = 64, rope_coord_dim: int = 1,
                 rope_kind: str = "log", rope_base: float = 10000.0, rope_sigma: float = 1.0,
                 use_fourier_bias: bool = True, fb_m: int = 32, fb_coord_dim: int = 1,
                 fb_kind: str = "gaussian", fb_base: float = 10000.0, fb_sigma: float = 1.0,
                 fb_headwise: bool = False, attn_pdrop: float = 0.0, resid_pdrop: float = 0.0,
                 causal: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.causal = causal

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.adrop = nn.Dropout(attn_pdrop)
        self.rdrop = nn.Dropout(resid_pdrop)

        # Attention mask
        if causal:
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        else:
            mask = torch.ones(block_size, block_size).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask, persistent=False)

        # RoPE
        self.use_rope = use_iron_rope
        self.rope_m = min(self.head_dim // 2, rope_m) if use_iron_rope else 0
        if self.rope_m > 0:
            self.W_rope = make_freq_bank(self.rope_m, rope_coord_dim, rope_kind, rope_base, rope_sigma)

        # Relative Fourier bias
        self.use_fb = use_fourier_bias
        self.fb_m = fb_m if use_fourier_bias else 0
        if self.fb_m > 0:
            self.W_fb = make_freq_bank(self.fb_m, fb_coord_dim, fb_kind, fb_base, fb_sigma)
            if fb_headwise:
                self.beta_cos = nn.Parameter(torch.zeros(self.n_head, self.fb_m))
                self.beta_sin = nn.Parameter(torch.zeros(self.n_head, self.fb_m))
            else:
                self.beta_cos = nn.Parameter(torch.zeros(self.fb_m))
                self.beta_sin = nn.Parameter(torch.zeros(self.fb_m))
            self.fb_scale = 1.0 / math.sqrt(self.fb_m)

    def forward(self,
                x: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                update_mask: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x:[B,T,C]; coords:[B,T,d_coord] or None; update_mask/context_mask:[B,T]"""
        B, T, C = x.shape
        assert T <= self.block_size
        device, dtype = x.device, x.dtype

        if coords is None:
            coords = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1).expand(B, T, 1)
        elif coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(B, -1, -1)

        qkv = self.c_attn(x)  # [B,T,3C]
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Iron RoPE rotation on q/k
        if self.rope_m > 0:
            Θ = coords @ self.W_rope.T  # [B,T,m]
            c = torch.cos(Θ).unsqueeze(1)  # [B,1,T,m]
            s = torch.sin(Θ).unsqueeze(1)
            q = rope_rotate_pairs(q, c, s, self.rope_m)
            k = rope_rotate_pairs(k, c, s, self.rope_m)

        # Attention logits
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B,H,T,T]

        # Relative Fourier bias
        if self.fb_m > 0:
            b = relative_fourier_bias(coords, coords, self.W_fb, self.beta_cos, self.beta_sin, self.fb_scale)
            att = att + b

        # Masks
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        if context_mask is not None:
            cm = context_mask[:, None, None, :].to(torch.bool)  # [B,1,1,T]
            att = att.masked_fill(~cm, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.adrop(att)

        # Weighted sum
        y = att @ v  # [B,H,T,hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.rdrop(self.c_proj(y))

        if update_mask is not None:
            um = update_mask.unsqueeze(-1).to(y.dtype)  # [B,T,1]
            y = x + y * um  # update only masked positions
        else:
            y = x + y
        return y


class LocalTokenMixer(nn.Module):
    """A tiny, bidirectional, Iron-RoPE attention block for denoising/mask-predict.

    - Builds a few **anchor** pseudo tokens that summarize context at multiple scales.
    - Uses Iron RoPE so attention depends on relative positions, helping mask repair.
    - Only updates **focused** tokens (the corrupted / low-confidence ones).

    Returns a pooled vector to feed into the sensor/motor regions.
    """
    def __init__(self, d: int, n_head: int = 4, block_size: int = 8192, m_tok: int = 64):
        super().__init__()
        self.attn = IronRoPESelfAttention(
            n_embd=d, n_head=n_head, block_size=block_size,
            use_iron_rope=True, rope_m=m_tok, rope_coord_dim=2,
            use_fourier_bias=True, fb_m=32, fb_coord_dim=1,
            causal=False
        )
        self.anch_embed = nn.Parameter(torch.randn(3, d))  # center / span_start / span_end

    @torch.no_grad()
    def build_anchors(self, T: int, focus_mask: torch.Tensor, strides=(128, 32)):
        anchors = []
        for S in strides:
            centers = list(range(S // 2, T, S))
            anchors += [(i, 0) for i in centers]  # kind 0
        # span boundary anchors around focused spans
        if focus_mask is not None and focus_mask.any():
            for l, r in extract_spans(focus_mask):
                anchors.append((l, 1))
                anchors.append((max(l, r - 1), 2))
        return anchors

    def forward(self,
                tok_emb: torch.Tensor,      # [B,T,d]
                pos_coords: torch.Tensor,   # [B,T,2]  (e.g., [i, i/T])
                focus_mask: torch.Tensor,   # [B,T]    (True = update this pos)
                ws_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, d = tok_emb.shape
        device = tok_emb.device
        X_list, C_list, UM_list = [], [], []
        for b in range(B):
            x = tok_emb[b]          # [T,d]
            c = pos_coords[b]       # [T,2]
            fm = focus_mask[b]
            anchors = self.build_anchors(T, fm)
            if len(anchors) > 0:
                a_idx, a_kind = zip(*anchors)
                a_idx = torch.tensor(a_idx, dtype=torch.long, device=device)
                a_kind = torch.tensor(a_kind, dtype=torch.long, device=device)
                a_emb = self.anch_embed[a_kind]  # [Na,d]
                a_coords = torch.stack([a_idx.to(torch.float32),
                                        a_idx.to(torch.float32) / (T + 1e-9)], dim=-1)
                x = torch.cat([x, a_emb], dim=0)
                c = torch.cat([c, a_coords], dim=0)
                um = torch.cat([fm, torch.zeros(len(a_idx), device=device, dtype=torch.bool)], dim=0)
            else:
                um = fm
            X_list.append(x)
            C_list.append(c)
            UM_list.append(um)
        X, C, UM = pad_batch(X_list, C_list, UM_list)  # [B,Taug,d], [B,Taug,2], [B,Taug]
        Y = self.attn(X, coords=C, update_mask=UM, context_mask=None)
        Y_main = Y[:, :T, :]
        pooled = masked_mean(Y_main, focus_mask)  # [B,d]
        return pooled


# ==========================================================
# 2) Workspace + Heads (Planner, Critic, Verifier, RTD, Token Head)
# ==========================================================

class Workspace(nn.Module):
    """Tiny scratchpad connected to all regions."""
    def __init__(self, d: int, N_slots: int = 8):
        super().__init__()
        self.N = N_slots
        self.norm = RMSNorm(d)
        self.register_buffer("slots", torch.zeros(N_slots, d), persistent=False)

    def read(self, keys: torch.Tensor) -> torch.Tensor:
        """keys:[K,d] -> values:[K,d] via dot-prod attn over slots."""
        s = keys @ self.slots.T  # [K,N]
        att = F.softmax(s, dim=-1)
        return att @ self.slots  # [K,d]

    def write(self, delta_slots: torch.Tensor):
        """Additive write, then sparsify (KWTA) to reduce interference."""
        assert delta_slots.shape == self.slots.shape
        self.slots = KWTA(self.slots + delta_slots, k=max(1, self.slots.shape[-1] // 4))


class MLP(nn.Module):
    """2-layer MLP with GELU."""
    def __init__(self, d_in: int, d_out: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(d_in, d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x): return self.net(x)


class PlannerHead(nn.Module):
    """Proposes subgoals from workspace/motor state (tiny)."""
    def __init__(self, d: int):
        super().__init__()
        self.mlp_p = MLP(d, d)
        self.mlp_g = MLP(2 * d, d)

    def forward(self, ws_state: torch.Tensor, motor_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.mlp_p(ws_state)
        g = self.mlp_g(torch.cat([p, ws_state], dim=-1))
        return p, g


class CriticHead(nn.Module):
    """Predicts expected Δgoodness if we allocate more compute."""
    def __init__(self, d: int):
        super().__init__()
        self.mlp = MLP(2 * d, 1)

    def forward(self, ws_state: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        v = self.mlp(torch.cat([ws_state, g], dim=-1)).squeeze(-1)
        return v  # scalar (per-sequence)


class VerifierHead(nn.Module):
    """Lightweight constraint checker on the motor state."""
    def __init__(self, d: int):
        super().__init__()
        self.mlp = MLP(d, 1)

    def forward(self, motor_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(motor_state)).squeeze(-1)  # 0..1


class RTDHead(nn.Module):
    """Replaced-Token Detection: binary classifier (real vs replaced)."""
    def __init__(self, d: int):
        super().__init__()
        self.clf = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf(x)  # logits [*,2]


class TokenHead_MFS(nn.Module):
    """Multi-Facet Softmax (Mixture-of-Facets) token head.

    A tiny, robust variant: facet gate over Kf lightweight linear heads.
    Returns:
      logits: [V], aux: dict with 'facet_balance' loss term
    """
    def __init__(self, d: int, V: int, Kf: int = 8):
        super().__init__()
        self.V = V
        self.Kf = Kf
        self.gate = nn.Linear(d, Kf)
        self.facets = nn.ModuleList([nn.Linear(d, V) for _ in range(Kf)])

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # x: [d] (motor state)
        g = F.softmax(self.gate(x), dim=-1)         # [Kf]
        logits_list = [head(x) for head in self.facets]  # each [V]
        logits = torch.stack(logits_list, dim=0)    # [Kf,V]
        out = (g.unsqueeze(-1) * logits).sum(dim=0) # [V]

        # facet balance loss: encourage uniform average gate usage
        with torch.no_grad():
            # If used in batches, you'd track EMA. Here, per-call proxy only.
            pass
        aux = {}
        g_mean = g.mean()
        uni = 1.0 / self.Kf
        # Simple L2 deviation from uniform (KL would need averaging over batches)
        aux['facet_balance'] = (g_mean - uni).pow(2)
        return out, aux


# ==========================================================
# 3) Goodness & Region FF State
# ==========================================================

def goodness(h_stack: torch.Tensor) -> torch.Tensor:
    """Scalar FF goodness on a stack of post-norm activations: mean of squared."""
    return (h_stack.pow(2).mean(dim=-1)).mean()  # average across stack and channels


class RegionFFState:
    """Per-region FF threshold τ (EMA)."""
    def __init__(self, init_tau: float = 0.0):
        self.tau = float(init_tau)

    def update_tau(self, g_pos_mean: float, alpha: float = 0.01):
        self.tau = (1.0 - alpha) * self.tau + alpha * float(g_pos_mean)


# ==========================================================
# 4) Gate (compute allocation) & Router (message passing with Fourier bias)
# ==========================================================

class Gate(nn.Module):
    """Discrete compute allocator over regions.

    - `gain_ema[r]` tracks usefulness (Δgoodness EMA)
    - `homeo[r]` implements firing-rate homeostasis
    - scores combine content (|H|), usefulness, focus proximity, homeostasis, and critic value bias
    """
    def __init__(self, R: int, neighbor_indices: List[List[int]], io_idxs: Dict[str, int]):
        super().__init__()
        self.R = R
        self.neighbors = neighbor_indices
        self.io_idxs = io_idxs
        self.register_buffer("gain_ema", torch.zeros(R))
        self.register_buffer("homeo", torch.zeros(R))
        # hyper-weights (safe small defaults)
        self.w1 = 1.0
        self.w2 = 0.3
        self.w3 = 0.5
        self.w4 = 0.2
        self.w5 = 0.5
        self.value_bias = 0.0  # set externally each step

    def overlap_with_focus(self, focus_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Crude focus proximity: boost sensor/motor when any focus exists."""
        if focus_mask is None or (focus_mask.numel() > 0 and not bool(focus_mask.any())):
            s = torch.zeros(self.R, device=self.gain_ema.device)
        else:
            s = torch.zeros(self.R, device=self.gain_ema.device)
            s[self.io_idxs['sensor']] = 1.0
            s[self.io_idxs['motor']] = 1.0
        return s

    def score_regions(self, H_hint: torch.Tensor, focus_mask: Optional[torch.Tensor], u_mean: float) -> torch.Tensor:
        """H_hint:[R,d] -> scores:[R]"""
        content = H_hint.abs().mean(dim=-1)                    # [R]
        focus_boost = self.overlap_with_focus(focus_mask)      # [R]
        scores = (self.w1 * content
                  + self.w2 * self.gain_ema
                  + self.w3 * focus_boost
                  - self.w4 * self.homeo
                  + self.w5 * float(self.value_bias))
        return scores

    def select_k(self, scores: torch.Tensor, k_active: int, burst_extra_k: int,
                 io_force_on: bool = True) -> torch.Tensor:
        """Neighbor-suppressed top-k + force sensor/motor on."""
        k_total = max(0, k_active + burst_extra_k)
        mask = torch.zeros(self.R, dtype=torch.bool, device=scores.device)
        forced = []
        if io_force_on:
            forced = [self.io_idxs['sensor'], self.io_idxs['motor']]
            mask[forced] = True
            k_total = max(0, k_total - len(forced))
        idx = nms_topk(scores, k_total, self.neighbors)
        mask[idx] = True
        return mask

    def update_gain(self, r: int, goodness_gain: float, beta: float = 0.9):
        self.gain_ema[r] = beta * self.gain_ema[r] + (1.0 - beta) * float(goodness_gain)

    def update_homeo(self, reg_mask: torch.Tensor, eta: float = 1e-3, target: float = 0.1):
        """Homeostatic drift toward target firing rate."""
        self.homeo += eta * (reg_mask.float() - target)


class Router(nn.Module):
    """Hex-graph message router with optional learned edge transforms and Fourier relative bias.

    Messages are constrained to neighbors (short paths). A small relative Fourier bias
    in the transform helps the router "feel" the hex geometry.
    """
    def __init__(self, neighbor_indices: List[List[int]], d: int, R: int):
        super().__init__()
        self.R = R
        self.d = d
        self.neighbors = neighbor_indices

        # Edge transforms
        edges = {}
        for r in range(R):
            for s in self.neighbors[r]:
                edges[f"{s}->{r}"] = nn.Linear(d, d, bias=False)
        self.W_edge = nn.ModuleDict(edges)

        # Fourier relative bias over region coordinates (2-D axial)
        self.fb_alpha = 0.1
        m_reg = 8
        self.W_reg = make_freq_bank(m_reg, 2, kind="gaussian", sigma=1.0)
        self.beta_cos = nn.Parameter(torch.zeros(m_reg))
        self.beta_sin = nn.Parameter(torch.zeros(m_reg))
        self.fb_scale = 1.0 / math.sqrt(m_reg)

    def messages(self, H: torch.Tensor, reg_mask_prev: torch.Tensor, reg_coords: torch.Tensor) -> torch.Tensor:
        """Aggregate messages from previously active neighbors.

        H: [R,d], reg_mask_prev: [R] bool, reg_coords: [R,2]
        Returns M: [R,d]
        """
        device = H.device
        M = torch.zeros(self.R, self.d, device=device)
        B = 1  # per-sequence call
        # Prepare coordinate tensors for bias
        P = reg_coords.to(H.dtype).to(device).unsqueeze(0)  # [1,R,2]
        for r in range(self.R):
            acc = torch.zeros(self.d, device=device)
            for s in self.neighbors[r]:
                if not bool(reg_mask_prev[s]):
                    continue
                msg = self.W_edge[f"{s}->{r}"](H[s])
                # Relative Fourier bias b(Δcoords) (scalar per edge)
                b = relative_fourier_bias(P[:, r:r+1, :], P[:, s:s+1, :],
                                          self.W_reg, self.beta_cos, self.beta_sin,
                                          self.fb_scale)[0, 0, 0, 0]
                acc = acc + (1.0 + self.fb_alpha * b) * msg
            M[r] = acc
        return M


# ==========================================================
# 5) RWKV Region Cell (with Δt skip + time rotation on v)
# ==========================================================

class RWKVRegionCell(nn.Module):
    """RWKV-like region processor with per-channel decay and Δt skip handling.

    State:
      - state_num/state_den: EMA numerators/denominators (fast weights)
      - dt: number of skipped micro-steps (for exact fast-forward decay)

    We add a small Iron time rotation (inner-step) to **v** only to encode
    diffusion time without breaking w=exp(k) positivity.
    """
    def __init__(self, d: int, m_time_pairs: int = 16):
        super().__init__()
        self.d = d
        self.norm = RMSNorm(d)
        self.r_lin = nn.Linear(d, d, bias=False)
        self.k_lin = nn.Linear(d, d, bias=False)
        self.v_lin = nn.Linear(d, d, bias=False)
        self.o_lin = nn.Linear(d, d, bias=False)
        self.decay_param = nn.Parameter(torch.zeros(d))
        self.register_buffer("state_num", torch.zeros(d), persistent=False)
        self.register_buffer("state_den", torch.zeros(d), persistent=False)
        self.dt = 0
        # Iron time rotation pairs for v
        self.m_time = max(0, min(d // 8 // 2, m_time_pairs))
        if self.m_time > 0:
            self.W_time = make_freq_bank(self.m_time, 1, kind="log", base=10000.0)

    def decay_vec(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_param).pow(2.0)  # (0,1)

    def fast_forward(self):
        if self.dt == 0:
            return
        lam = self.decay_vec()  # [d]
        mul = lam.pow(self.dt)
        self.state_num = self.state_num * mul
        self.state_den = self.state_den * mul
        self.dt = 0

    def skip(self):
        self.dt += 1

    def step(self, x_in: torch.Tensor, step_pos_scalar: float) -> torch.Tensor:
        """One RWKV region update.

        x_in: [d] input vector, step_pos_scalar ∈ [0,1] inner-step position
        """
        x = self.norm(x_in)
        self.fast_forward()
        r = torch.sigmoid(self.r_lin(x))
        k = self.k_lin(x)                    # keep unrotated (exp(k) >= 0)
        v = self.v_lin(x)

        # Iron time rotation on v (encode inner-step time)
        if self.m_time > 0:
            Θ = step_pos_scalar * self.W_time.squeeze(-1)  # [m_time]
            c, s = torch.cos(Θ), torch.sin(Θ)
            v = rope_rotate_pairs(v, c, s, self.m_time)

        lam = self.decay_vec()
        w = torch.exp(k)  # positive
        self.state_num = self.state_num * lam + w * v
        self.state_den = self.state_den * lam + w
        y = r * (self.state_num / (self.state_den + 1e-9))  # [d]
        h = x + self.o_lin(y)
        h = KWTA(h, k=max(1, self.d // 8))
        return h


# ==========================================================
# 6) Cortex Reasoner (container + inner loop)
# ==========================================================

@dataclass
class CortexConfig:
    R: int = 32
    d: int = 128
    V: int = 256           # vocab size (e.g., bytes)
    K_inner: int = 8
    B_br: int = 2
    k_active: int = 8
    max_T: int = 8192


class CortexReasoner(nn.Module):
    """The top-level container: regions, gate, router, heads, workspace, and inner loop."""
    def __init__(self,
                 neighbor_indices: List[List[int]],
                 reg_coords: torch.Tensor,
                 io_idxs: Dict[str, int],
                 cfg: CortexConfig):
        super().__init__()
        self.cfg = cfg
        self.R = cfg.R
        self.d = cfg.d
        self.V = cfg.V
        self.io_idxs = io_idxs
        self.neighbors = neighbor_indices
        self.register_buffer("reg_coords", reg_coords.to(torch.float32))

        # Tokens
        self.embed = nn.Embedding(self.V, self.d)

        # Regions
        self.regions = nn.ModuleList([RWKVRegionCell(self.d) for _ in range(self.R)])

        # Gate & Router
        self.gate = Gate(self.R, self.neighbors, io_idxs)
        self.router = Router(self.neighbors, self.d, self.R)

        # Heads & workspace
        self.rtdd = RTDHead(self.d)
        self.lm_head = TokenHead_MFS(self.d, self.V, Kf=8)
        self.work = Workspace(self.d, N_slots=8)
        self.plan = PlannerHead(self.d)
        self.critic = CriticHead(self.d)
        self.verify = VerifierHead(self.d)

        # Local token mixer (Iron RoPE) and input norm
        self.local_mix = LocalTokenMixer(self.d, n_head=4, block_size=cfg.max_T, m_tok=64)
        self.norm_in = RMSNorm(self.d)

        # Per-region FF threshold τ
        self.reg_ff = [RegionFFState() for _ in range(self.R)]

    def region_input(self, r: int, x_sensor: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        x = msg
        if r in (self.io_idxs['sensor'], self.io_idxs['motor']):
            x = x + x_sensor
        return self.norm_in(x)

    def forward_inner_step(self,
                           tokens: torch.Tensor,       # [T] Long
                           step_k: int,
                           focus_map: torch.Tensor,    # [T] Bool
                           H_prev: torch.Tensor,       # [R,d]
                           reg_mask_prev: torch.Tensor # [R] Bool
                           ):
        """One inner micro-step: gate -> route -> active region updates -> heads."""
        device = H_prev.device
        T = tokens.shape[0]

        # --- 0) Build a sensor vector via local Iron mixer (update only focus) ---
        tok_emb = self.embed(tokens).unsqueeze(0)  # [1,T,d]
        pos = torch.stack([torch.arange(T, device=device, dtype=torch.float32),
                           torch.arange(T, device=device, dtype=torch.float32) / (T + 1e-9)], dim=-1).unsqueeze(0)
        focus_b = focus_map.unsqueeze(0)
        sensor_vec = self.local_mix(tok_emb, pos, focus_b, ws_slots=None)[0]  # [d]

        # --- 1) Gate selection ---
        H_hint = H_prev
        scores = self.gate.score_regions(H_hint, focus_map, u_mean=0.0)
        reg_mask = self.gate.select_k(scores, k_active=self.cfg.k_active, burst_extra_k=0, io_force_on=True)

        # --- 2) Router messages from previously active regions ---
        M = self.router.messages(H_prev, reg_mask_prev, self.reg_coords)  # [R,d]

        # --- 3) Region updates ---
        H_cur = torch.zeros_like(H_prev)
        for r in range(self.R):
            if not bool(reg_mask[r]):
                self.regions[r].skip()
                continue
            sensor_or_zero = sensor_vec if r in (self.io_idxs['sensor'], self.io_idxs['motor']) else torch.zeros(self.d, device=device)
            x_r = self.region_input(r, sensor_or_zero, M[r])
            step_pos = float(step_k) / float(max(1, self.cfg.K_inner - 1))
            H_cur[r] = self.regions[r].step(x_r, step_pos_scalar=step_pos)

        # --- 4) Heads on motor & workspace ---
        motor_state = H_cur[self.io_idxs['motor']]
        # workspace summary: mean over active regions (simple choice)
        active = reg_mask.nonzero(as_tuple=False).squeeze(-1)
        if active.numel() > 0:
            ws_state = H_cur[active].mean(dim=0)
        else:
            ws_state = torch.zeros(self.d, device=device)

        logits, aux = self.lm_head(motor_state)
        rtd_logits = self.rtdd(motor_state)

        return H_cur, reg_mask, logits, rtd_logits, ws_state, motor_state

    @torch.no_grad()
    def zeros_state(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.R, self.d, device=device), torch.zeros(self.R, dtype=torch.bool, device=device)

    def reasoning_loop(self,
                       tokens: torch.Tensor,      # [T] Long
                       K_inner: int,
                       focus_map: torch.Tensor,   # [T] Bool
                       reg_mask_prev: torch.Tensor,
                       H_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Always-on latent reasoning loop (plan→critic→gate; branch; verify; halt)."""
        # For simplicity, we keep B_br branches as alternative snapshots of region states.
        branches = [(self.work.slots.clone(), reg_mask_prev.clone(), H_prev.clone()) for _ in range(self.cfg.B_br)]
        branch_scores = [torch.tensor(-1e9, device=H_prev.device) for _ in range(self.cfg.B_br)]
        traces: List[torch.Tensor] = []

        last_logits = torch.zeros(self.V, device=H_prev.device)

        for k in range(K_inner):
            # Uncertainty from last logits (scalar)
            u_mean = uncertainty_from_logits(last_logits) if k > 0 else 0.0

            # Plan & value
            # (Simple: ws_state = mean of previous active states if any)
            active = reg_mask_prev.nonzero(as_tuple=False).squeeze(-1)
            if active.numel() > 0:
                ws_state_prev = H_prev[active].mean(dim=0)
            else:
                ws_state_prev = torch.zeros(self.d, device=H_prev.device)
            p, g = self.plan(ws_state_prev, H_prev[self.io_idxs['motor']])
            v_hat = float(self.critic(ws_state_prev, g).item())

            # Gate bias by value-of-compute:
            self.gate.value_bias = v_hat
            burst_extra_k = schedule_burst(u_mean, v_hat, self.R)

            # Try each branch
            for b in range(self.cfg.B_br):
                self.work.slots = branches[b][0].clone()
                H_prev_b, reg_mask_prev_b = branches[b][2].clone(), branches[b][1].clone()

                H_cur, reg_mask, logits, rtd_logits, ws_state, motor_state = \
                    self.forward_inner_step(tokens, step_k=k,
                                            focus_map=focus_map,
                                            H_prev=H_prev_b, reg_mask_prev=reg_mask_prev_b)

                verify_score = self.verify(motor_state)                     # scalar in [0,1]
                # Estimate Δgoodness (local): g(H_cur) - g(H_prev)
                delta_g = goodness(H_cur) - goodness(H_prev_b)
                score = 0.6 * delta_g + 0.2 * verify_score + 0.2 * context_logprob(logits)

                branches[b] = (self.work.slots.clone(), reg_mask.clone(), H_cur.clone())
                branch_scores[b] = score
                last_logits = logits

            # Keep top-2 branches
            top_idx = torch.topk(torch.stack(branch_scores), k=min(self.cfg.B_br, 2)).indices.tolist()
            branches = [branches[i] for i in top_idx]
            branch_scores = [branch_scores[i] for i in top_idx]

            # Gather trace
            traces.append(branches[0][2].clone())

            if should_halt(branch_scores, u_mean, k, K_inner):
                break

            # Prepare next prev
            self.work.slots, reg_mask_prev, H_prev = branches[0]

        # Emit best branch
        best = int(torch.stack(branch_scores).argmax().item())
        self.work.slots, reg_mask, H_cur = branches[best]
        return H_cur, reg_mask, last_logits, traces


# ==========================================================
# 7) Corruptions for FF training (RTD / SPAN / BLOCK)
# ==========================================================

def corrupt(tokens: torch.Tensor, V: int, mode: str):
    """Return negative stream inputs and metadata.

    tokens: [T] Long
    mode in {'RTD', 'SPAN', 'BLOCK'}
    Returns:
      x_neg:[T] Long, is_real:[T] Long in {0,1}, focus:[T] Bool,
      denoise_targets:[T] Long, denoise_mask:[T] Bool
    """
    T = tokens.shape[0]
    device = tokens.device

    if mode == 'RTD':
        p = 0.15
        mask = torch.rand(T, device=device) < p
        repl = torch.randint(low=0, high=V, size=(T,), device=device)
        x_neg = torch.where(mask, repl, tokens)
        is_real = (x_neg == tokens).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    elif mode == 'SPAN':
        rate = 0.3
        avg_len = 5
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        t = 0
        while t < T:
            if random.random() < rate:
                L = max(1, int(random.expovariate(1.0 / avg_len)))
                mask[t: min(T, t + L)] = True
                t += L
            else:
                t += 1
        x_neg = tokens.clone()
        MASK_ID = V - 1  # reserve last id as [MASK] if desired
        x_neg[mask] = MASK_ID
        is_real = (~mask).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    elif mode == 'BLOCK':
        # Shuffle a random block
        block_len = max(2, T // 8)
        start = random.randint(0, max(0, T - block_len))
        perm = torch.randperm(block_len, device=device)
        x_neg = tokens.clone()
        x_neg[start:start + block_len] = x_neg[start:start + block_len][perm]
        is_real = (x_neg == tokens).long()
        focus = torch.zeros(T, dtype=torch.bool, device=device)
        focus[start:start + block_len] = True
        denoise_targets = tokens
        denoise_mask = focus

    else:
        raise ValueError("unknown mode")

    return x_neg, is_real, focus, denoise_targets, denoise_mask


# ==========================================================
# 8) FF Training Step (basic, single-sequence version)
# ==========================================================

@dataclass
class LossWeights:
    ff: float = 1.0
    rtd: float = 0.5
    denoise: float = 1.0
    critic: float = 0.25
    verify: float = 0.25
    facet: float = 1e-3


def train_step(model: CortexReasoner,
               optimizer: torch.optim.Optimizer,
               clean_tokens: torch.Tensor,  # [B,T] Long
               λ: LossWeights,
               device: torch.device) -> Dict[str, float]:
    """One training step mixing FF goodness, RTD, denoising, and tiny auxiliaries.

    NOTE: For clarity, this is a *single-step* trainer over a batch of independent
    sequences. It loops per sample (B small), running the full inner reasoning loop
    for positive and negative streams, then aggregates losses.

    Returns metrics dict with component losses.
    """
    model.train()
    B, T = clean_tokens.shape
    total_ff = 0.0
    total_rtd = 0.0
    total_denoise = 0.0
    total_critic = 0.0
    total_verify = 0.0

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        tokens = clean_tokens[b]

        # Sample corruption mode
        mode = random.choices(['RTD', 'SPAN', 'BLOCK'], weights=[0.5, 0.3, 0.2])[0]
        x_neg, is_real, focus_map, denoise_targets, denoise_mask = corrupt(tokens, model.V, mode)

        # --- Positive stream ---
        H_prev, reg_mask_prev = model.zeros_state(device)
        focus_zero = torch.zeros_like(focus_map, dtype=torch.bool)
        H_pos, reg_mask_p, logits_pos, traces_pos = model.reasoning_loop(tokens, model.cfg.K_inner,
                                                                         focus_zero, reg_mask_prev, H_prev)

        # --- Negative stream ---
        H_prev, reg_mask_prev = model.zeros_state(device)
        H_neg, reg_mask_n, logits_neg, traces_neg = model.reasoning_loop(x_neg, model.cfg.K_inner,
                                                                         focus_map, reg_mask_prev, H_prev)

        # -------- FF per-region losses --------
        ff_loss = torch.tensor(0.0, device=device)
        # stack traces per region: list of [R,d] -> [K,R,d]
        if len(traces_pos) > 0:
            Hpos_stack = torch.stack(traces_pos, dim=0)  # [K,R,d]
        else:
            Hpos_stack = torch.zeros(0, model.R, model.d, device=device)

        if len(traces_neg) > 0:
            Hneg_stack = torch.stack(traces_neg, dim=0)  # [K,R,d]
        else:
            Hneg_stack = torch.zeros(0, model.R, model.d, device=device)

        for r in range(model.R):
            hpos = Hpos_stack[:, r, :] if Hpos_stack.numel() > 0 else None
            hneg = Hneg_stack[:, r, :] if Hneg_stack.numel() > 0 else None

            gpos = goodness(hpos) if hpos is not None and hpos.numel() > 0 else torch.tensor(0.0, device=device)
            gneg = goodness(hneg) if hneg is not None and hneg.numel() > 0 else torch.tensor(0.0, device=device)
            τ = model.reg_ff[r].tau
            L_pos = F.softplus(-(gpos - τ))
            L_neg = F.softplus(+(gneg - τ))
            ff_loss = ff_loss + (L_pos + L_neg)
            # Update τ (EMA) and gate usefulness (detach)
            if hpos is not None and hpos.numel() > 0:
                model.reg_ff[r].update_tau(float(gpos.detach().item()))
            gain = float((gpos - gneg).detach().item())
            model.gate.update_gain(r, gain)

        # -------- RTD loss (on negative stream motor state) --------
        motor_neg = H_neg[model.io_idxs['motor']]  # [d]
        rtd_logits = model.rtdd(motor_neg).unsqueeze(0)  # [1,2]
        # Build a simple target: if any replacement happened -> 0 else 1 (crude)
        rtd_target = torch.tensor([[1, 0]], device=device) if is_real.all() else torch.tensor([[0, 1]], device=device)
        rtd_loss = F.cross_entropy(rtd_logits, rtd_target.argmax(dim=-1))

        # -------- Denoising loss (mask-predict) on masked positions --------
        token_logits, aux = model.lm_head(motor_neg)  # [V]
        if bool(denoise_mask.any()):
            # For simplicity, compare against the *first* masked target
            first_idx = int(denoise_mask.nonzero(as_tuple=False)[0].item())
            target_id = denoise_targets[first_idx].unsqueeze(0)  # [1]
            ce = F.cross_entropy(token_logits.unsqueeze(0), target_id)
            denoise_loss = ce + aux.get('facet_balance', torch.tensor(0.0, device=device)) * λ.facet
        else:
            denoise_loss = torch.tensor(0.0, device=device)

        # -------- Critic regression (predict realized Δgoodness) --------
        realized_delta_g = (goodness(H_pos) - goodness(H_neg)).detach()
        # ws_state (neg stream) proxy: mean over final active
        active_n = reg_mask_n.nonzero(as_tuple=False).squeeze(-1)
        ws_state_neg = H_neg[active_n].mean(dim=0) if active_n.numel() > 0 else torch.zeros(model.d, device=device)
        # Use plan->critic on that
        p, g = model.plan(ws_state_neg, H_neg[model.io_idxs['motor']])
        v_hat = model.critic(ws_state_neg, g)
        critic_loss = F.mse_loss(v_hat, realized_delta_g)

        # -------- Verifier auxiliary --------
        ver_score = model.verify(motor_neg)
        # Structural target: encourage "OK" (1.0) when focus existed (we attempted a repair)
        ver_target = torch.tensor(1.0 if bool(focus_map.any()) else 0.5, device=device)
        verifier_loss = F.binary_cross_entropy(ver_score, ver_target)

        # -------- Total --------
        total = (λ.ff * ff_loss
                 + λ.rtd * rtd_loss
                 + λ.denoise * denoise_loss
                 + λ.critic * critic_loss
                 + λ.verify * verifier_loss)

        total_loss = total_loss + total

        # Metrics (detach)
        total_ff += float(ff_loss.detach().item())
        total_rtd += float(rtd_loss.detach().item())
        total_denoise += float(denoise_loss.detach().item())
        total_critic += float(critic_loss.detach().item())
        total_verify += float(verifier_loss.detach().item())

        # Homeostasis update (drifts slowly)
        model.gate.update_homeo(reg_mask_n)

    # Backprop once per batch
    total_loss.backward()
    optimizer.step()

    return {
        "ff": total_ff / B,
        "rtd": total_rtd / B,
        "denoise": total_denoise / B,
        "critic": total_critic / B,
        "verify": total_verify / B,
        "total": float(total_loss.detach().item())
    }


# ==========================================================
# 9) Generation (mask-predict style, non-autoregressive outer loop)
# ==========================================================

@torch.no_grad()
def generate(model: CortexReasoner,
             prompt_tokens: torch.Tensor,  # [T0] Long
             T_total: int,
             max_outer_iters: int = 8,
             conf_threshold: float = 0.9) -> torch.Tensor:
    """Mask-predict generator with always-on inner loop."""
    device = next(model.parameters()).device
    T0 = prompt_tokens.shape[0]
    tokens = torch.cat([prompt_tokens, torch.full((T_total - T0,), model.V - 1, device=device, dtype=torch.long)], dim=0)  # fill with [MASK]
    conf = torch.zeros(T_total, device=device)
    H_prev, reg_mask_prev = model.zeros_state(device)

    for _ in range(max_outer_iters):
        focus_map = (tokens == (model.V - 1)) | (conf < conf_threshold)
        H_prev, reg_mask_prev, logits, traces = model.reasoning_loop(tokens, model.cfg.K_inner,
                                                                     focus_map, reg_mask_prev, H_prev)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        maxp = probs.max(dim=-1).values
        conf[:] = maxp
        tokens = torch.where(focus_map, pred, tokens)
        if bool((conf > conf_threshold).all()) and bool((tokens != (model.V - 1)).all()):
            break
    return tokens


# ==========================================================
# 10) Minimal wiring helpers (hex-like neighbor graph)
# ==========================================================

def hex_neighbors_grid(R: int, side: int) -> List[List[int]]:
    """Build a simple 2D grid neighborhood (4-neighbors) as a hex proxy.

    For research convenience; replace with true hex axial coords if you have them.
    Assumes R == side * side. Returns adjacency list.
    """
    assert R == side * side
    neighbors = [[] for _ in range(R)]
    def idx(x, y): return x * side + y
    for i in range(side):
        for j in range(side):
            r = idx(i, j)
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neigh
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    neighbors[r].append(idx(ni, nj))
    return neighbors


def hex_axial_coords_from_grid(R: int, side: int) -> torch.Tensor:
    """Produce 2-D coordinates for regions laid out on a square grid (proxy for hex)."""
    coords = []
    for i in range(side):
        for j in range(side):
            coords.append([float(i), float(j)])
    return torch.tensor(coords, dtype=torch.float32)


# ==========================================================
# 11) Example (smoke test)
# ==========================================================

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Config
    cfg = CortexConfig(R=16, d=128, V=258, K_inner=8, B_br=2, k_active=8, max_T=2048)
    side = int(math.sqrt(cfg.R))
    neighbors = hex_neighbors_grid(cfg.R, side=side)
    reg_coords = hex_axial_coords_from_grid(cfg.R, side=side)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    # Dummy batch of byte tokens (B=2, T=64)
    B, T = 2, 64
    clean_tokens = torch.randint(low=0, high=cfg.V - 1, size=(B, T), device=device)

    # Loss weights
    λ = LossWeights(ff=1.0, rtd=0.5, denoise=1.0, critic=0.25, verify=0.25, facet=1e-3)

    # One training step
    metrics = train_step(model, optimizer, clean_tokens, λ, device)
    print({k: float(v) for k, v in metrics.items()})

    # Simple generation from a prompt of length 8 tokens
    prompt = torch.randint(low=0, high=cfg.V - 1, size=(8,), device=device)
    out = generate(model, prompt, T_total=32, max_outer_iters=4, conf_threshold=0.85)
    print("Generated tokens:", out.tolist())
