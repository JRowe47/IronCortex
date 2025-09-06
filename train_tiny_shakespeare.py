import random
import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    LossWeights,
    train_step,
    load_tiny_shakespeare,
    hex_neighbors_grid,
    hex_axial_coords_from_grid,
)


def build_model(device: torch.device) -> CortexReasoner:
    # The wiring helpers assume the region count forms a perfect square so that
    # regions can be arranged on a 2D grid. Using a non-square value (e.g. 32)
    # caused `hex_neighbors_grid` to raise an assertion error. Set ``R`` to a
    # square number (36 = 6x6 grid) to make the demo run.
    cfg = CortexConfig(R=36, d=256, V=256, K_inner=8, B_br=2, k_active=8, max_T=512)
    side = int(cfg.R**0.5)
    neighbors = hex_neighbors_grid(cfg.R, side)
    reg_coords = hex_axial_coords_from_grid(cfg.R, side)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    return model


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = load_tiny_shakespeare("data")
    seq_len = 256
    n_seq = len(tokens) // seq_len
    tokens = tokens[: n_seq * seq_len].view(n_seq, seq_len)
    loader = torch.utils.data.DataLoader(tokens, batch_size=8, shuffle=True)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lamb = LossWeights()

    header = "step,ff,rtd,denoise,critic,verify,ce,total,gain_mean,tau_mean"
    print(header)
    for step, batch in enumerate(loader, 1):
        batch = batch.to(device)
        metrics = train_step(model, optimizer, batch, lamb, device)
        gain_mean = float(model.gate.gain_ema.mean().item())
        tau_mean = float(torch.stack([r.tau for r in model.reg_ff]).mean().item())
        line = (
            f"{step},{metrics['ff']:.4f},{metrics['rtd']:.4f},{metrics['denoise']:.4f},"
            f"{metrics['critic']:.4f},{metrics['verify']:.4f},{metrics['ce']:.4f},"
            f"{metrics['total']:.4f},{gain_mean:.4f},{tau_mean:.4f}"
        )
        print(line)
        if step >= 50:
            break


if __name__ == "__main__":
    main()
