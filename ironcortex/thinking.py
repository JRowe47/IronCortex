"""Inference-time optimisation helpers for the energy-based verifier."""

from __future__ import annotations

import torch

from .energy import EnergyVerifierHead, energy_descent_step


class Thinker:
    def __init__(
        self,
        verifier: EnergyVerifierHead,
        max_steps: int = 3,
        alpha: float | tuple[float, float] = (2e-2, 5e-2),
        sigma: float = 0.0,
        restarts: int = 1,
    ):
        self.verifier = verifier
        self.max_steps = max_steps
        self.alpha = alpha if isinstance(alpha, tuple) else (alpha, alpha)
        self.sigma = sigma
        self.restarts = restarts

    def optimize(
        self, ctx: torch.Tensor, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        best = None
        for _ in range(self.restarts):
            y_hat = y0 + 0.01 * torch.randn_like(y0)
            energies: list[torch.Tensor] = []
            for _ in range(self.max_steps):
                a = torch.empty(1).uniform_(*self.alpha).item()
                y_hat, E = energy_descent_step(
                    self.verifier, ctx, y_hat, alpha=a, sigma=self.sigma
                )
                energies.append(E)
                if len(energies) > 1 and abs(energies[-1] - energies[-2]) < 1e-4:
                    break
            final_E = self.verifier(ctx, y_hat).detach()
            if best is None or final_E.mean() < best[0].mean():
                best = (final_E, y_hat.detach(), energies)
        assert best is not None
        return best
