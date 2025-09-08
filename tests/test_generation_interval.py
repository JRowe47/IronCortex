import random

import torch

import train_tiny_shakespeare as tts
from ironcortex import CortexConfig, CortexReasoner, hex_axial_coords, hex_neighbors


def test_generation_interval(monkeypatch):
    torch.manual_seed(0)
    random.seed(0)
    cfg = CortexConfig(R=4, d=32, V=20, K_inner=2, B_br=1, k_active=2, max_T=64)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    device = torch.device("cpu")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    data = torch.randint(0, cfg.V - 1, size=(20, 8))
    loader = torch.utils.data.DataLoader(data, batch_size=1)

    calls = []

    def fake_generate(
        model, prompt, T_total, max_outer_iters=None, conf_threshold=None
    ):
        calls.append(T_total)
        return torch.zeros(T_total, dtype=torch.long)

    monkeypatch.setattr(tts, "generate", fake_generate)
    hparams = tts.TrainHyperParams(
        epochs=1,
        max_steps=10,
        log_interval=2,
        gen_interval=5,
        gen_tokens=8,
        batch_size=1,
        seq_len=8,
        visualize=False,
    )
    tts.train(model, loader, hparams, device)
    assert len(calls) == 2
