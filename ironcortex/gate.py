import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import nms_topk
from .iron_rope import make_freq_bank, relative_fourier_bias
from .constants import EPS_DIV, EPS_LOG

# 4) Gate (compute allocation) & Router (message passing with Fourier bias)
# ==========================================================


class Gate(nn.Module):
    """Discrete compute allocator over regions.

    - `gain_ema[r]` tracks usefulness (Δgoodness EMA)
    - `homeo[r]` implements firing-rate homeostasis
    - scores combine content (|H|), usefulness, focus proximity, homeostasis, and critic value bias
    """

    def __init__(
        self, R: int, neighbor_indices: List[List[int]], io_idxs: Dict[str, int]
    ):
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
        self.w6 = 0.3
        self.value_bias = 0.0  # set externally each step

    def overlap_with_focus(self, focus_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Crude focus proximity: boost sensor/motor when any focus exists."""
        if focus_mask is None or (
            focus_mask.numel() > 0 and not bool(focus_mask.any())
        ):
            s = torch.zeros(self.R, device=self.gain_ema.device)
        else:
            s = torch.zeros(self.R, device=self.gain_ema.device)
            s[self.io_idxs["sensor"]] = 1.0
            s[self.io_idxs["motor"]] = 1.0
        return s

    def score_regions(
        self, H_hint: torch.Tensor, focus_mask: Optional[torch.Tensor], u_mean: float
    ) -> torch.Tensor:
        """H_hint:[R,d] -> scores:[R]"""
        content = H_hint.abs().mean(dim=-1)  # [R]
        focus_boost = self.overlap_with_focus(focus_mask)  # [R]
        scores = (
            self.w1 * content
            + self.w2 * self.gain_ema
            + self.w3 * focus_boost
            - self.w4 * self.homeo
            + self.w5 * float(self.value_bias)
            + self.w6 * float(u_mean)
        )
        return scores

    def select_k(
        self,
        scores: torch.Tensor,
        k_active: int,
        burst_extra_k: int,
        io_force_on: bool = True,
    ) -> torch.Tensor:
        """Neighbor-suppressed top-k + force sensor/motor on."""
        k_total = max(0, k_active + burst_extra_k)
        mask = torch.zeros(self.R, dtype=torch.bool, device=scores.device)
        forced = []
        if io_force_on:
            forced = [self.io_idxs["sensor"], self.io_idxs["motor"]]
            mask[forced] = True
            k_total = max(0, k_total - len(forced))
        idx = nms_topk(scores, k_total, self.neighbors)
        mask[idx] = True
        return mask

    def update_gain(self, r: int, goodness_gain: float, beta: float = 0.9):
        self.gain_ema[r] = beta * self.gain_ema[r] + (1.0 - beta) * float(goodness_gain)

    def update_homeo(
        self, reg_mask: torch.Tensor, eta: float = 1e-3, target: float = 0.1
    ):
        """Homeostatic drift toward target firing rate."""
        self.homeo += eta * (reg_mask.float() - target)


class Router(nn.Module):
    """Hex-graph message router with optional learned edge transforms and Fourier relative bias.

    Messages are constrained to neighbors (short paths). A small relative Fourier bias
    in the transform helps the router "feel" the hex geometry.
    """

    def __init__(
        self,
        neighbor_indices: List[List[int]],
        d: int,
        R: int,
        *,
        enable_precision_routed_messages: bool = False,
    ):
        super().__init__()
        self.R = R
        self.d = d
        self.neighbors = neighbor_indices
        self.enable_precision_routed_messages = enable_precision_routed_messages

        # Edge transforms
        edges = {}
        for r in range(R):
            for s in self.neighbors[r]:
                edges[f"{s}->{r}"] = nn.Linear(d, d, bias=False)
        self.W_edge = nn.ModuleDict(edges)

        # Content scoring projections
        self.query_lin = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(R)])
        self.key_lin = nn.ModuleDict(
            {k: nn.Linear(d, d, bias=False) for k in self.W_edge.keys()}
        )

        # Edge precision parameters (diag)
        self.raw_P_edge = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d)) for k in self.W_edge.keys()}
        )

        # Storage for last routing weights (for interpretability)
        self.last_weights: Dict[str, float] = {}
        self.last_weight_mean: float = 0.0
        self.last_weight_entropy: float = 0.0

        # Fourier relative bias over region coordinates (2-D axial)
        self.fb_alpha = 0.1
        m_reg = 8
        self.register_buffer(
            "W_reg", make_freq_bank(m_reg, 2, kind="gaussian", sigma=1.0)
        )
        self.beta_cos = nn.Parameter(torch.zeros(m_reg))
        self.beta_sin = nn.Parameter(torch.zeros(m_reg))
        nn.init.normal_(self.beta_cos, std=0.01)
        nn.init.normal_(self.beta_sin, std=0.01)
        self.fb_scale = 1.0 / math.sqrt(m_reg)

    def messages(
        self, H: torch.Tensor, reg_mask_prev: torch.Tensor, reg_coords: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate messages from previously active neighbors.

        H: [R,d], reg_mask_prev: [R] bool, reg_coords: [R,2]
        Returns M: [R,d]
        """
        device = H.device
        M = torch.zeros(self.R, self.d, device=device)
        self.last_weights = {}
        # Prepare coordinate tensors for bias
        P = reg_coords.to(H.dtype).to(device).unsqueeze(0)  # [1,R,2]
        for r in range(self.R):
            acc = torch.zeros(self.d, device=device)
            msgs = []
            scores = []
            robust = []
            edge_keys = []
            q_r = None
            if self.enable_precision_routed_messages:
                q_r = self.query_lin[r](H[r])
            for s in self.neighbors[r]:
                if not bool(reg_mask_prev[s]):
                    continue
                edge_key = f"{s}->{r}"
                msg = self.W_edge[edge_key](H[s])
                # Relative Fourier bias b(Δcoords) (scalar per edge)
                b = relative_fourier_bias(
                    P[:, r : r + 1, :],
                    P[:, s : s + 1, :],
                    self.W_reg,
                    self.beta_cos,
                    self.beta_sin,
                    self.fb_scale,
                )[0, 0, 0, 0]
                msg = (1.0 + self.fb_alpha * b) * msg
                if self.enable_precision_routed_messages:
                    k_sr = self.key_lin[edge_key](H[s])
                    score = (q_r * k_sr).sum() / math.sqrt(self.d)
                    resid = msg - H[r]
                    P_edge = F.softplus(self.raw_P_edge[edge_key])
                    mah = (resid.pow(2) * P_edge).sum()
                    w = torch.exp(-0.5 * mah)
                    msgs.append(msg)
                    scores.append(score)
                    robust.append(w)
                    edge_keys.append(edge_key)
                else:
                    acc = acc + msg
            if self.enable_precision_routed_messages:
                if len(msgs) > 0:
                    scores_t = torch.stack(scores)
                    content_w = torch.softmax(scores_t, dim=0)
                    robust_w = torch.stack(robust)
                    w = content_w * robust_w
                    Z = w.sum()
                    if float(Z) > 0:
                        w = w / Z
                    for wi, msgi, key in zip(w, msgs, edge_keys):
                        acc = acc + wi * msgi
                        self.last_weights[key] = float(wi.detach())
            M[r] = acc
        if self.enable_precision_routed_messages and self.last_weights:
            w_t = torch.tensor(list(self.last_weights.values()), device=device)
            self.last_weight_mean = float(w_t.mean().item())
            Z = w_t.sum().clamp_min(EPS_DIV)
            p = w_t / Z
            self.last_weight_entropy = float((-(p + EPS_LOG).log() * p).sum().item())
        else:
            self.last_weight_mean = 0.0
            self.last_weight_entropy = 0.0
        return M
