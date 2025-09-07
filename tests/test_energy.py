"""Basic tests for energy-based utilities."""

import torch

from ironcortex import EnergyVerifierHead, energy_descent_step


def test_energy_descent_reduces_energy():
    torch.manual_seed(0)
    ctx = torch.randn(4, 8)
    y = torch.randn(4, 6)
    verifier = EnergyVerifierHead(ctx_dim=8, y_dim=6, hidden=16)
    y_next, E_before = energy_descent_step(verifier, ctx, y, alpha=0.1)
    _, E_after = energy_descent_step(verifier, ctx, y_next, alpha=0.1)
    assert E_after.item() <= E_before.item()
