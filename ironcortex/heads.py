from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import RMSNorm, KWTA

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


