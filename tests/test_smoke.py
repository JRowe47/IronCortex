import random
import torch

from ironcortex import CortexConfig, CortexReasoner, LossWeights, train_step, generate, hex_neighbors_grid, hex_axial_coords_from_grid


def test_smoke():
    torch.manual_seed(0)
    random.seed(0)
    cfg = CortexConfig(R=4, d=32, V=20, K_inner=2, B_br=1, k_active=2, max_T=64)
    side = int(cfg.R ** 0.5)
    neighbors = hex_neighbors_grid(cfg.R, side)
    reg_coords = hex_axial_coords_from_grid(cfg.R, side)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    device = torch.device("cpu")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B, T = 1, 8
    clean_tokens = torch.randint(low=0, high=cfg.V - 1, size=(B, T), device=device)
    lambda_weights = LossWeights()
    metrics = train_step(model, optimizer, clean_tokens, lambda_weights, device)
    assert "total" in metrics
    prompt = torch.randint(low=0, high=cfg.V - 1, size=(4,), device=device)
    out = generate(model, prompt, T_total=8, max_outer_iters=2, conf_threshold=0.8)
    assert out.shape[0] == 8
