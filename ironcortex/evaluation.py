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
    if T < 2:
        raise ValueError("tokens must have length >= 2 for evaluation")

    # Evaluate only the final next-token prediction to avoid O(T^2) work.
    # This provides a lightweight approximation sufficient for logging.
    ctx = tokens[:, :-1]
    targets = tokens[:, -1]
    focus = torch.zeros(B, T - 1, dtype=torch.bool, device=device)
    _, _, logits, _, _ = model.reasoning_loop_batch(ctx, model.cfg.K_inner, focus)
    log_probs = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(log_probs, targets, reduction="mean").item()
    ppl = float(math.exp(ce))
    return {"cross_entropy": ce, "perplexity": ppl}
