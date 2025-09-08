"""Evaluation utilities for cross entropy and perplexity."""

import math
from typing import Dict

import torch
import torch.nn.functional as F

from .model import CortexReasoner


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
    focus = torch.zeros(B, T, dtype=torch.bool, device=device)
    total_nll = 0.0
    total_tokens = 0

    for t in range(T - 1):
        ctx = tokens[:, : t + 1]
        targets = tokens[:, t + 1]
        _, _, logits, _ = model.reasoning_loop_batch(
            ctx, model.cfg.K_inner, focus[:, : t + 1]
        )
        log_probs = F.log_softmax(logits, dim=-1)
        total_nll += F.nll_loss(log_probs, targets, reduction="sum").item()
        total_tokens += targets.numel()

    cross_entropy = total_nll / max(1, total_tokens)
    perplexity = float(math.exp(cross_entropy))
    return {"cross_entropy": cross_entropy, "perplexity": perplexity}
