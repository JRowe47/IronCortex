import random
import torch
import train_tiny_shakespeare as tts
from ironcortex import CortexConfig, CortexReasoner, hex_axial_coords, hex_neighbors


def test_train_hex_visualizer(monkeypatch):
    torch.manual_seed(0)
    random.seed(0)
    cfg = CortexConfig(R=4, d=32, V=20, K_inner=2, B_br=1, k_active=2, max_T=64)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    device = torch.device("cpu")
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    data = torch.randint(0, cfg.V - 1, size=(10, 8))
    loader = torch.utils.data.DataLoader(data, batch_size=1)

    instances = []

    class DummyVis:
        def __init__(self, R):
            self.R = R
            self.updates: list[list[float]] = []
            instances.append(self)

        def update(self, states):
            self.updates.append(list(states))

    monkeypatch.setattr(tts, "HexStateVisualizer", DummyVis)

    hparams = tts.TrainHyperParams(
        epochs=1,
        max_steps=1,
        log_interval=10,
        gen_interval=0,
        batch_size=1,
        seq_len=8,
        visualize=False,
        hex_visualize=True,
    )
    tts.train(model, loader, hparams, device)
    assert instances and instances[0].updates
    assert len(instances[0].updates[0]) == cfg.R
