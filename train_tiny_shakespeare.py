import random
from dataclasses import dataclass

import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    DiffusionConfig,
    LossWeights,
    diffusion_generate,
    hex_axial_coords_from_grid,
    hex_neighbors_grid,
    load_tiny_shakespeare,
    train_step,
)


@dataclass
class TrainHyperParams:
    """Knobs controlling the Tiny Shakespeare demo."""

    epochs: int = 1
    max_steps: int = 50
    log_interval: int = 10
    batch_size: int = 8
    seq_len: int = 256
    gen_tokens: int = 128
    diffusion_steps: int = 4


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


def train(
    model: CortexReasoner, loader, hparams: TrainHyperParams, device: torch.device
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lamb = LossWeights()
    header = "step,ff,rtd,denoise,critic,verify,ce,total,gain_mean,tau_mean"
    print(header)
    step = 0
    for epoch in range(1, hparams.epochs + 1):
        for batch in loader:
            step += 1
            batch = batch.to(device)
            metrics = train_step(model, optimizer, batch, lamb, device)
            gain_mean = float(model.gate.gain_ema.mean().item())
            tau_mean = float(torch.stack([r.tau for r in model.reg_ff]).mean().item())
            if step % hparams.log_interval == 0:
                line = (
                    f"{step},{metrics['ff']:.4f},{metrics['rtd']:.4f},{metrics['denoise']:.4f},"
                    f"{metrics['critic']:.4f},{metrics['verify']:.4f},{metrics['ce']:.4f},"
                    f"{metrics['total']:.4f},{gain_mean:.4f},{tau_mean:.4f}"
                )
                print(line)
            if hparams.max_steps and step >= hparams.max_steps:
                return


def run_generation(
    model: CortexReasoner, prompt_text: str, hparams: TrainHyperParams
) -> None:
    device = next(model.parameters()).device
    prompt = torch.tensor(
        list(prompt_text.encode("utf-8")), dtype=torch.long, device=device
    )
    diff_cfg = DiffusionConfig(steps=hparams.diffusion_steps)
    out = diffusion_generate(
        model, prompt, T_total=hparams.gen_tokens, diff_cfg=diff_cfg
    )
    text = bytes(out.tolist()).decode("utf-8", errors="ignore")
    print(text)


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = TrainHyperParams()
    tokens = load_tiny_shakespeare("data")
    n_seq = len(tokens) // hparams.seq_len
    tokens = tokens[: n_seq * hparams.seq_len].view(n_seq, hparams.seq_len)
    loader = torch.utils.data.DataLoader(
        tokens, batch_size=hparams.batch_size, shuffle=True
    )

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    train(model, loader, hparams, device)
    run_generation(model, "ROMEO:", hparams)


if __name__ == "__main__":
    main()
