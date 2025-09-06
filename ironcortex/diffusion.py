from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch

from .model import CortexReasoner


@dataclass
class DiffusionConfig:
    """Hyperparameters for diffusion generation."""

    steps: int = 4
    noise_start: float = 1.0
    noise_end: float = 0.0


@torch.no_grad()
def diffusion_generate(
    model: CortexReasoner,
    prompt_tokens: torch.Tensor,
    T_total: int,
    diff_cfg: Optional[DiffusionConfig] = None,
    region_generators: Optional[
        Iterable[Callable[[torch.Tensor], torch.Tensor]]
    ] = None,
) -> torch.Tensor:
    """Generate tokens using a simple diffusion-style denoising loop.

    Parameters
    ----------
    model:
        The ``CortexReasoner`` used to denoise sequences.
    prompt_tokens:
        Conditioning prompt of shape ``[T0]`` (dtype ``torch.long``).
    T_total:
        Total length of sequence to generate.
    diff_cfg:
        Configuration controlling diffusion steps and noise schedule.
    region_generators:
        Optional hooks for region-specific generators. Each callable receives
        the current token sequence and returns an updated version. These hooks
        can later be implemented asynchronously on separate nodes.
    """

    if diff_cfg is None:
        diff_cfg = DiffusionConfig()

    device = next(model.parameters()).device
    T0 = prompt_tokens.shape[0]
    tokens = torch.randint(low=0, high=model.V - 1, size=(T_total,), device=device)
    tokens[:T0] = prompt_tokens.to(device)
    H_prev, reg_mask_prev = model.zeros_state(device)

    for step in range(diff_cfg.steps, 0, -1):
        noise_p = (
            diff_cfg.noise_end
            + (diff_cfg.noise_start - diff_cfg.noise_end) * step / diff_cfg.steps
        )
        mask = torch.rand(T_total, device=device) < noise_p
        random_tokens = torch.randint(0, model.V - 1, (T_total,), device=device)
        noisy = torch.where(mask, random_tokens, tokens)
        focus_map = torch.ones(T_total, dtype=torch.bool, device=device)
        H_prev, reg_mask_prev, logits, traces = model.reasoning_loop(
            noisy, model.cfg.K_inner, focus_map, reg_mask_prev, H_prev
        )
        tokens = logits.argmax(dim=-1)
        if region_generators:
            for fn in region_generators:
                tokens = fn(tokens)

    return tokens
