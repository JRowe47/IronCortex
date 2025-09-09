"""Energy-based verification head and inner-loop optimisation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyVerifierHead(nn.Module):
    """Tiny MLP producing a scalar energy given context and prediction."""

    def __init__(
        self, ctx_dim: int, y_dim: int, hidden: int, *, use_attn_energy: bool = False
    ):
        super().__init__()
        self.use_attn_energy = use_attn_energy
        in_dim = ctx_dim + y_dim + (1 if use_attn_energy else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        ctx: torch.Tensor,
        y_hat: torch.Tensor,
        attn_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_attn_energy:
            if attn_energy is None:
                if ctx.dim() == 1:
                    attn_energy = torch.zeros(1, device=ctx.device, dtype=ctx.dtype)
                else:
                    attn_energy = torch.zeros(
                        ctx.shape[0], device=ctx.device, dtype=ctx.dtype
                    )
            if attn_energy.dim() == 0:
                attn_energy = attn_energy.expand(1)
            if ctx.dim() == 1:
                z = torch.cat([ctx, y_hat, attn_energy.reshape(1)], dim=-1)
            else:
                z = torch.cat([ctx, y_hat, attn_energy.unsqueeze(-1)], dim=-1)
        else:
            z = torch.cat([ctx, y_hat], dim=-1)
        return self.net(z).squeeze(-1)


def energy_descent_step(
    verifier: EnergyVerifierHead,
    ctx: torch.Tensor,
    y_hat: torch.Tensor,
    alpha: float,
    sigma: float = 0.0,
    clamp: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform one gradient descent step on ``y_hat`` to reduce energy."""

    y_hat.requires_grad_(True)
    energy = verifier(ctx, y_hat).sum()
    grad = torch.autograd.grad(energy, y_hat, create_graph=False)[0]
    noise = sigma * torch.randn_like(y_hat) if sigma > 0 else 0.0
    y_next = y_hat - alpha * grad + noise
    if clamp is not None:
        y_next = y_next.clamp(*clamp)
    return y_next.detach(), energy.detach()
