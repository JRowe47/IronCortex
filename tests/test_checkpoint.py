import random

import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    LossWeights,
    hex_axial_coords,
    hex_neighbors,
    train_step,
    save_checkpoint,
    load_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    random.seed(0)
    cfg = CortexConfig(R=4, d=32, V=20, K_inner=2, B_br=1, k_active=2, max_T=64)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    device = torch.device("cpu")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    clean_tokens = torch.randint(low=0, high=cfg.V - 1, size=(1, 8), device=device)
    lamb = LossWeights()
    train_step(model, optimizer, clean_tokens, lamb, device)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, 1, ckpt)

    model2 = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step = load_checkpoint(ckpt, model2, optimizer2, map_location=device)
    assert step == 1

    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k])

    state1 = optimizer.state_dict()
    state2 = optimizer2.state_dict()
    assert state1.keys() == state2.keys()
    assert state1["state"].keys() == state2["state"].keys()
    for key in state1["state"]:
        sub1 = state1["state"][key]
        sub2 = state2["state"][key]
        for sk in sub1:
            v1, v2 = sub1[sk], sub2[sk]
            if isinstance(v1, torch.Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2
