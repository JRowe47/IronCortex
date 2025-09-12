"""Evaluation utilities for cross entropy, benchmarking, and quality checks."""

import math
import time
from typing import Dict

import torch
import torch.nn.functional as F

from .model import CortexReasoner
from .attention.adaptive_filter_attention import AdaptiveFilterAttention


@torch.no_grad()
def evaluate_perplexity(
    model: CortexReasoner, tokens: torch.Tensor, device: torch.device
) -> Dict[str, float]:
    """Compute cross entropy and perplexity for sequences with teacher forcing.

    Args:
        model: The ``CortexReasoner`` to evaluate.
        tokens: Tensor of shape ``[B, T]`` containing token ids.
        device: Device on which to run the computation.

    Returns:
        Dictionary with ``cross_entropy`` and ``perplexity``.
    """
    model.eval()
    tokens = tokens.to(device)
    B, T = tokens.shape
    if T < 2:
        raise ValueError("tokens must have length >= 2 for evaluation")

    # Evaluate cross-entropy across the entire sequence for a more faithful
    # perplexity estimate. This loops over each time step with teacher forcing,
    # trading runtime for accuracy compared to the previous single-step
    # approximation.
    ce_sum = 0.0
    n_pred = 0
    for t in range(1, T):
        ctx = tokens[:, :t]
        targets = tokens[:, t]
        focus = torch.zeros(B, t, dtype=torch.bool, device=device)
        _, _, logits, _, _, _ = model.reasoning_loop_batch(
            ctx, model.cfg.K_inner, focus
        )
        log_probs = F.log_softmax(logits, dim=-1)
        ce_sum += F.nll_loss(log_probs, targets, reduction="sum").item()
        n_pred += B
    ce = ce_sum / max(1, n_pred)
    ppl = float(math.exp(ce))
    return {"cross_entropy": ce, "perplexity": ppl}


@torch.no_grad()
def runtime_memory_benchmark(
    model: CortexReasoner, seq_len: int, device: torch.device
) -> Dict[str, float]:
    """Measure runtime and memory for a synthetic sequence."""
    model.eval()
    tokens = torch.randint(0, model.cfg.V - 1, (1, seq_len), device=device)
    focus = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    start_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    start = time.perf_counter()
    model.reasoning_loop_batch(tokens, model.cfg.K_inner, focus)
    runtime = time.perf_counter() - start
    end_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    return {"runtime_s": runtime, "memory_bytes": float(end_mem - start_mem)}


@torch.no_grad()
def noise_rejection_benchmark(seq_len: int = 32, d: int = 16) -> Dict[str, float]:
    """Compare denoising with and without Adaptive Filter Attention."""
    attn_std = AdaptiveFilterAttention(d_model=d, n_head=2, alpha=0.0)
    attn_afa = AdaptiveFilterAttention(d_model=d, n_head=2, alpha=0.1)
    x = torch.randn(1, seq_len, d)
    noise = 0.1 * torch.randn_like(x)
    target = x
    noisy = x + noise
    out_std = attn_std(noisy)
    out_afa = attn_afa(noisy)
    mse_std = F.mse_loss(out_std, target).item()
    mse_afa = F.mse_loss(out_afa, target).item()
    return {"mse_std": mse_std, "mse_afa": mse_afa}
