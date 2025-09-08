import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    evaluate_perplexity,
    hex_axial_coords,
    hex_neighbors,
)


def test_evaluate_perplexity_runs():
    cfg = CortexConfig(R=2, d=8, V=16, K_inner=2, B_br=1, k_active=1, max_T=8)
    neighbors = hex_neighbors(cfg.R)
    coords = hex_axial_coords(cfg.R)
    io = {"sensor": 0, "motor": cfg.R - 1}
    model = CortexReasoner(neighbors, coords, io, cfg)
    tokens = torch.randint(0, cfg.V, (1, 4), dtype=torch.long)
    metrics = evaluate_perplexity(model, tokens, device=torch.device("cpu"))
    assert metrics["cross_entropy"] > 0
    assert metrics["perplexity"] > 0
