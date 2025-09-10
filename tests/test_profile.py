import torch

from ironcortex import CortexConfig, CortexReasoner
from ironcortex.wiring import hex_neighbors, hex_axial_coords


def test_profile_accumulates_and_resets():
    cfg = CortexConfig(R=3, d=8, V=32, K_inner=2, profile=True, profile_every_n_steps=1)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    model = CortexReasoner(
        neighbors, reg_coords, {"sensor": 0, "motor": cfg.R - 1}, cfg
    )
    tokens = torch.zeros(1, 2, dtype=torch.long)
    focus = torch.zeros(1, 2, dtype=torch.bool)
    model.reasoning_loop_batch(tokens, cfg.K_inner, focus)
    assert model._profile_times["routing"] > 0
    model.report_profile()
    assert model._profile_times["routing"] == 0.0
