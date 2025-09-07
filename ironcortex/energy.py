"""Energy-based verification head and inner-loop optimisation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyVerifierHead(nn.Module):
    """Tiny MLP producing a scalar energy given context and prediction."""

    def __init__(self, ctx_dim: int, y_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(ctx_dim + y_dim),
            nn.Linear(ctx_dim + y_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
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
