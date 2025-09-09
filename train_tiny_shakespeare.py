import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from ironcortex import (
    CortexConfig,
    CortexReasoner,
    DiffusionConfig,
    LossWeights,
    evaluate_perplexity,
    generate,
    diffusion_generate,
    hex_axial_coords,
    hex_neighbors,
    load_tiny_shakespeare,
    train_step,
    TrainVisualizer,
    HexStateVisualizer,
    save_checkpoint,
    load_checkpoint,
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
    gen_interval: int = 20
    gen_prompt: str = "ROMEO:"
    visualize: bool = False
    hex_visualize: bool = False
    ckpt_path: str | None = None
    ckpt_interval: int = 0


def build_model(
    device: torch.device,
    *,
    enable_adaptive_filter_dynamics: bool = False,
    enable_precision_routed_messages: bool = False,
    enable_radial_tangential_updates: bool = False,
    enable_afa_attention: bool = False,
    enable_ff_energy_alignment: bool = False,
    enable_energy_verifier: bool = True,
    enable_forward_forward: bool = True,
    debug_metrics_every_n_steps: int = 0,
) -> CortexReasoner:
    cfg = CortexConfig(
        R=36,
        d=256,
        V=256,
        K_inner=8,
        B_br=2,
        k_active=8,
        max_T=512,
        enable_adaptive_filter_dynamics=enable_adaptive_filter_dynamics,
        enable_precision_routed_messages=enable_precision_routed_messages,
        enable_radial_tangential_updates=enable_radial_tangential_updates,
        enable_afa_attention=enable_afa_attention,
        enable_ff_energy_alignment=enable_ff_energy_alignment,
        enable_energy_verifier=enable_energy_verifier,
        enable_forward_forward=enable_forward_forward,
        debug_metrics_every_n_steps=debug_metrics_every_n_steps,
    )
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
    header = "step,ff,rtd,denoise,critic,verify,E_pos,E_neg,xent,ppl,total,gain_mean,tau_mean"
    print(header)
    step = 0
    if hparams.ckpt_path and Path(hparams.ckpt_path).exists():
        step = load_checkpoint(hparams.ckpt_path, model, optimizer, map_location=device)
        print(f"loaded checkpoint from {hparams.ckpt_path} at step {step}")
    vis = None
    if hparams.visualize:
        try:
            vis = TrainVisualizer()
        except Exception as e:  # pragma: no cover - visualization optional
            print(f"visualization disabled: {e}")
    hex_vis = None
    if hparams.hex_visualize:
        try:
            hex_vis = HexStateVisualizer(R=model.cfg.R)
            gains = torch.sigmoid(model.gate.gain_ema.detach()).tolist()
            hex_vis.update(gains)
        except Exception as e:  # pragma: no cover - visualization optional
            print(f"hex visualization disabled: {e}")
    prompt_ids = torch.tensor(
        list(hparams.gen_prompt.encode("utf-8")), dtype=torch.long, device=device
    )
    for epoch in range(1, hparams.epochs + 1):
        for batch in loader:
            step += 1
            batch = batch.to(device)
            metrics = train_step(model, optimizer, batch, lamb, device)
            if (
                model.cfg.debug_metrics_every_n_steps > 0
                and step % model.cfg.debug_metrics_every_n_steps == 0
            ):
                telem = model.telemetry()
                dbg = (
                    f"dbg step {step}: var={telem['state_var_mean']:.4f}"
                    f" prec={telem['state_prec_mean']:.4f}"
                    f" surprise={telem['surprise_ema']:.4f}"
                    f" router_H={telem['router_weight_entropy']:.4f}"
                    f" attn_H={telem['attn_entropy_mean']:.4f}"
                )
                print(dbg)
            gain_mean = float(model.gate.gain_ema.mean().item())
            tau_mean = float(torch.stack([r.tau for r in model.reg_ff]).mean().item())
            if step % hparams.log_interval == 0:
                eval_metrics = evaluate_perplexity(model, batch, device)
                line = (
                    f"{step},{metrics['ff']:.4f},{metrics['rtd']:.4f},{metrics['denoise']:.4f},"
                    f"{metrics['critic']:.4f},{metrics['verify']:.4f},{metrics['E_pos']:.4f},"
                    f"{metrics['E_neg']:.4f},{eval_metrics['cross_entropy']:.4f},"
                    f"{eval_metrics['perplexity']:.2f},{metrics['total']:.4f},"
                    f"{gain_mean:.4f},{tau_mean:.4f}"
                )
                print(line)
                if vis is not None:
                    vis.update(
                        step,
                        {**metrics, "gain_mean": gain_mean, "tau_mean": tau_mean},
                        eval_metrics,
                    )
                if hex_vis is not None:
                    gains = torch.sigmoid(model.gate.gain_ema.detach()).tolist()
                    hex_vis.update(gains)
            if hparams.gen_interval and step % hparams.gen_interval == 0:
                sample = generate(model, prompt_ids, T_total=hparams.gen_tokens)
                txt = bytes(sample.tolist()).decode("utf-8", errors="ignore")
                print("=== Sample Generation ===")
                print(txt)
            if (
                hparams.ckpt_path
                and hparams.ckpt_interval
                and step % hparams.ckpt_interval == 0
            ):
                save_checkpoint(model, optimizer, step, hparams.ckpt_path)
            if hparams.max_steps and step >= hparams.max_steps:
                if hparams.ckpt_path:
                    save_checkpoint(model, optimizer, step, hparams.ckpt_path)
                return
    if hparams.ckpt_path:
        save_checkpoint(model, optimizer, step, hparams.ckpt_path)


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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable training metrics visualizer",
    )
    parser.add_argument(
        "--hex-vis",
        action="store_true",
        dest="hex_visualize",
        help="Enable hex region state visualizer",
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
        help="Steps between checkpoint saves (0 disables periodic saving)",
    )
    parser.add_argument(
        "--enable-adaptive-filter-dynamics",
        action="store_true",
        help="Enable adaptive filter dynamics in regions",
    )
    parser.add_argument(
        "--enable-precision-routed-messages",
        action="store_true",
        help="Enable precision-weighted routing",
    )
    parser.add_argument(
        "--enable-radial-tangential-updates",
        action="store_true",
        help="Enable radial-tangential state updates",
    )
    parser.add_argument(
        "--enable-afa-attention",
        action="store_true",
        help="Enable Adaptive Filter Attention",
    )
    parser.add_argument(
        "--enable-ff-energy-alignment",
        action="store_true",
        help="Enable forward-forward energy alignment",
    )
    parser.add_argument(
        "--disable-energy-verifier",
        action="store_true",
        help="Disable energy verifier head",
    )
    parser.add_argument(
        "--disable-forward-forward",
        action="store_true",
        help="Disable forward-forward training loss",
    )
    parser.add_argument(
        "--debug-metrics-every-n-steps",
        type=int,
        default=0,
        help="Print telemetry metrics every N steps (0 disables)",
    )
    args = parser.parse_args()
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

    model = build_model(
        device,
        enable_adaptive_filter_dynamics=args.enable_adaptive_filter_dynamics,
        enable_precision_routed_messages=args.enable_precision_routed_messages,
        enable_radial_tangential_updates=args.enable_radial_tangential_updates,
        enable_afa_attention=args.enable_afa_attention,
        enable_ff_energy_alignment=args.enable_ff_energy_alignment,
        enable_energy_verifier=not args.disable_energy_verifier,
        enable_forward_forward=not args.disable_forward_forward,
        debug_metrics_every_n_steps=args.debug_metrics_every_n_steps,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    print(f"using seed {seed}")
    train(model, loader, hparams, device)
    run_generation(model, hparams.gen_prompt, hparams)


if __name__ == "__main__":
    main()
