import argparse
import random
from dataclasses import dataclass

import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    DiffusionConfig,
    LossWeights,
    generate,
    diffusion_generate,
    hex_axial_coords,
    hex_neighbors,
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
    gen_tokens: int = 100
    diffusion_steps: int = 4
    seed: int | None = None


def build_model(device: torch.device) -> CortexReasoner:
    cfg = CortexConfig(R=36, d=256, V=256, K_inner=8, B_br=2, k_active=8, max_T=512)
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    return model


def train(
    model: CortexReasoner, loader, hparams: TrainHyperParams, device: torch.device
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lamb = LossWeights()
    header = "step,ff,rtd,denoise,critic,verify,E_pos,E_neg,ce,total,gain_mean,tau_mean"
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
                    f"{metrics['critic']:.4f},{metrics['verify']:.4f},{metrics['E_pos']:.4f},"
                    f"{metrics['E_neg']:.4f},{metrics['ce']:.4f},{metrics['total']:.4f},"
                    f"{gain_mean:.4f},{tau_mean:.4f}"
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
    raw = diffusion_generate(
        model, prompt, T_total=hparams.gen_tokens, diff_cfg=diff_cfg
    )
    raw_text = bytes(raw.tolist()).decode("utf-8", errors="ignore")
    print("=== Raw Diffusion Output ===")
    print(raw_text)

    think = generate(model, prompt, T_total=hparams.gen_tokens)
    think_text = bytes(think.tolist()).decode("utf-8", errors="ignore")
    print("=== Energy-Based Thinking Output ===")
    print(think_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = TrainHyperParams(seed=seed)
    tokens = load_tiny_shakespeare("data")
    n_seq = len(tokens) // hparams.seq_len
    tokens = tokens[: n_seq * hparams.seq_len].view(n_seq, hparams.seq_len)
    loader = torch.utils.data.DataLoader(
        tokens, batch_size=hparams.batch_size, shuffle=True
    )

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    print(f"using seed {seed}")
    train(model, loader, hparams, device)
    run_generation(model, "ROMEO:", hparams)


if __name__ == "__main__":
    main()
