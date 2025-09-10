"""Energy-based verification head and inner-loop optimisation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyVerifierHead(nn.Module):
    """Tiny MLP producing a scalar energy given context and prediction."""

    def __init__(self, ctx_dim: int, y_dim: int, hidden: int, *, aux_dim: int = 0):
        super().__init__()
        self.aux_dim = aux_dim
        in_dim = ctx_dim + y_dim + aux_dim
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
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.aux_dim > 0:
            if aux is None:
                if ctx.dim() == 1:
                    aux = torch.zeros(self.aux_dim, device=ctx.device, dtype=ctx.dtype)
                else:
                    aux = torch.zeros(
                        ctx.shape[0], self.aux_dim, device=ctx.device, dtype=ctx.dtype
                    )
            if ctx.dim() == 1:
                z = torch.cat([ctx, y_hat, aux.view(-1)], dim=-1)
            else:
                if aux.dim() == 1:
                    aux = aux.unsqueeze(0).expand(ctx.shape[0], -1)
                z = torch.cat([ctx, y_hat, aux], dim=-1)
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
