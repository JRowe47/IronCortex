import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    LossWeights,
    hex_axial_coords,
    hex_neighbors,
    train_step,
)
from ironcortex.gate import Router
from ironcortex.attention.adaptive_filter_attention import AdaptiveFilterAttention
from ironcortex.evaluation import runtime_memory_benchmark, noise_rejection_benchmark


def test_no_nans_grad_flow_and_telemetry():
    torch.manual_seed(0)
    cfg = CortexConfig(
        R=3,
        d=8,
        V=16,
        K_inner=1,
        B_br=1,
        k_active=2,
        max_T=16,
        enable_adaptive_filter_dynamics=True,
        enable_precision_routed_messages=True,
        enable_radial_tangential_updates=True,
        enable_afa_attention=True,
        enable_ff_energy_alignment=True,
        surprise_lambda=1.0,
        surprise_lambda_schedule=10,
    )
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    device = torch.device("cpu")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, cfg.V - 1, (1, 4), device=device)
    weights = LossWeights()
    metrics = train_step(model, optimizer, tokens, weights, device)
    for v in metrics.values():
        assert torch.isfinite(torch.tensor(v))
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    telem = model.telemetry()
    assert "regions" in telem and len(telem["regions"]) == cfg.R
    assert "routing_weight_mean" in telem
    assert metrics["lambda_s"] > 0
    assert model.verify_state is not None and model.verify_state.tau.item() >= 0
    assert getattr(model.verify, "aux_dim", 0) == 3


def test_edge_cases_router_and_attention():
    torch.manual_seed(0)
    router = Router([[]], d=4, R=1, enable_precision_routed_messages=True)
    H = torch.randn(1, 4)
    mask = torch.zeros(1, dtype=torch.bool)
    coords = torch.zeros(1, 2)
    out = router.messages(H, mask, coords)
    assert torch.isfinite(out).all()
    attn = AdaptiveFilterAttention(d_model=8, n_head=2)
    x = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4, 4)
    y = attn(x, mask=mask)
    assert torch.isfinite(y).all()


def test_benchmarks_run():
    torch.manual_seed(0)
    cfg = CortexConfig(
        R=2,
        d=8,
        V=16,
        K_inner=1,
        B_br=1,
        k_active=1,
        max_T=16,
        enable_afa_attention=True,
    )
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": 1}
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg)
    bench = runtime_memory_benchmark(model, seq_len=4, device=torch.device("cpu"))
    assert "runtime_s" in bench and "memory_bytes" in bench
    quality = noise_rejection_benchmark(seq_len=8, d=8)
    assert quality["mse_std"] >= 0 and quality["mse_afa"] >= 0
