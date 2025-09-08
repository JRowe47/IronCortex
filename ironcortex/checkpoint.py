"""Checkpoint utilities for saving and loading models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .model import CortexReasoner


def save_checkpoint(
    model: CortexReasoner,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    path: str | Path,
) -> None:
    """Save model/optimizer state and step to *path*.

    Args:
        model: Model to serialize.
        optimizer: Optional optimizer whose state will also be saved.
        step: Training step number to record.
        path: Destination file path.
    """

    ckpt_path = Path(path)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
    }
    torch.save(ckpt, ckpt_path)


def load_checkpoint(
    path: str | Path,
    model: Optional[CortexReasoner] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device | None = None,
) -> int:
    """Load a checkpoint from *path* into *model* and *optimizer*.

    Returns the training step stored in the checkpoint.
    """

    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))
