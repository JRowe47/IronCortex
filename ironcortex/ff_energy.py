"""Forward-Forward helpers using energy as goodness."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .energy import EnergyVerifierHead


def ff_energy_loss(
    E_pos: torch.Tensor, E_neg: torch.Tensor, tau: float = 0.0
) -> torch.Tensor:
    return F.softplus(E_pos - tau).mean() + F.softplus(-(E_neg - tau)).mean()


def ff_step(
    verifier: EnergyVerifierHead,
    ctx_pos: torch.Tensor,
    y_pos: torch.Tensor,
    ctx_neg: torch.Tensor,
    y_neg: torch.Tensor,
    tau: float = 0.0,
) -> dict[str, float]:
    E_pos = verifier(ctx_pos.detach(), y_pos.detach())
    E_neg = verifier(ctx_neg.detach(), y_neg.detach())
    loss = ff_energy_loss(E_pos, E_neg, tau=tau)
    loss.backward()
    return {
        "loss": float(loss.detach().item()),
        "E_pos": float(E_pos.detach().mean().item()),
        "E_neg": float(E_neg.detach().mean().item()),
    }
