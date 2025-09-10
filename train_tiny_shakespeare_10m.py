"""
Tiny Shakespeare demo with ~10M parameter model and all experimental features enabled.
"""

import argparse
import random
from dataclasses import dataclass

import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    hex_axial_coords,
    hex_neighbors,
    load_tiny_shakespeare,
)
from train_tiny_shakespeare import TrainHyperParams, train, run_generation


@dataclass
class _Args:
    seed: int | None = None
    visualize: bool = False
    hex_visualize: bool = False
    ckpt_path: str | None = None
    ckpt_interval: int = 0


def build_small_model(device: torch.device) -> CortexReasoner:
    """Builds a ~10M parameter model with all experimental flags enabled."""
    cfg = CortexConfig(
        R=20,
        d=256,
        V=256,
        K_inner=8,
        B_br=2,
        k_active=8,
        max_T=512,
        enable_adaptive_filter_dynamics=True,
        enable_precision_routed_messages=True,
        enable_radial_tangential_updates=True,
        enable_afa_attention=True,
        enable_ff_energy_alignment=True,
        enable_energy_verifier=True,
        enable_forward_forward=True,
    )
    neighbors = hex_neighbors(cfg.R)
    reg_coords = hex_axial_coords(cfg.R)
    io_idxs = {"sensor": 0, "motor": cfg.R - 1}
    model = CortexReasoner(neighbors, reg_coords, io_idxs, cfg).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--visualize", action="store_true", help="Enable training metrics visualizer"
    )
    parser.add_argument(
        "--hex-vis",
        action="store_true",
        dest="hex_visualize",
        help="Enable hex region visualizer",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to save/load training checkpoints",
    )
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=0,
        help="Steps between checkpoint saves (0 disables)",
    )
    args = parser.parse_args(namespace=_Args())

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hparams = TrainHyperParams(
        seed=seed,
        visualize=args.visualize,
        hex_visualize=args.hex_visualize,
        ckpt_path=args.ckpt_path,
        ckpt_interval=args.ckpt_interval,
    )

    tokens = load_tiny_shakespeare("data")
    n_seq = len(tokens) // hparams.seq_len
    tokens = tokens[: n_seq * hparams.seq_len].view(n_seq, hparams.seq_len)
    loader = torch.utils.data.DataLoader(
        tokens, batch_size=hparams.batch_size, shuffle=True
    )

    model = build_small_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    print(f"using seed {seed}")
    train(model, loader, hparams, device)
    run_generation(model, hparams.gen_prompt, hparams)


if __name__ == "__main__":
    main()
