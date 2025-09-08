import random

import torch
import pytest

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    DiffusionConfig,
    LossWeights,
    diffusion_generate,
    generate,
    hex_axial_coords,
    hex_neighbors,
    train_step,
)


def test_smoke():
    torch.manual_seed(0)
    random.seed(0)
    cfg = CortexConfig(R=4, d=32, V=20, K_inner=2, B_br=1, k_active=2, max_T=64)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
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
    diff_out = diffusion_generate(
        model, prompt, T_total=8, diff_cfg=DiffusionConfig(steps=2)
    )
    assert diff_out.shape[0] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_router_fourier_bias_device():
    """Ensure Router's Fourier bias frequency bank moves with the module's device."""
    torch.manual_seed(0)
    random.seed(0)
    cfg = CortexConfig(R=4, d=32, V=20, K_inner=2, B_br=1, k_active=2, max_T=64)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    device = torch.device("cuda")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    router = model.router
    assert router.W_reg.device == device
    H = torch.randn(cfg.R, cfg.d, device=device)
    reg_mask = torch.ones(cfg.R, dtype=torch.bool, device=device)
    # Should not raise a device mismatch
    router.messages(H, reg_mask, reg_coords.to(device))
