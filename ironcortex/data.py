"""Data loading utilities for text datasets and diffusion experiments.

This module currently provides helpers for downloading and preparing the
Tiny Shakespeare corpus as well as a simple text diffusion dataset that
emits (noisy, clean, t) triplets for denoising objectives.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tiny_shakespeare(root: str) -> str:
    """Ensure the Tiny Shakespeare dataset exists under ``root``.

    Parameters
    ----------
    root: str
        Directory where the dataset should live.

    Returns
    -------
    str
        Path to the downloaded text file.
    """

    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, "tiny_shakespeare.txt")
    if not os.path.exists(out_path):
        try:
            with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as resp:
                data = resp.read()
            with open(out_path, "wb") as f:
                f.write(data)
        except Exception as e:  # pragma: no cover - network failures
            raise RuntimeError("failed to download Tiny Shakespeare dataset") from e
    return out_path


def load_tiny_shakespeare(root: str) -> torch.Tensor:
    """Return the Tiny Shakespeare corpus tokenized as bytes.

    The dataset is downloaded on demand if missing.
    """

    path = download_tiny_shakespeare(root)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    tokens = torch.tensor(list(data.encode("utf-8")), dtype=torch.long)
    return tokens


@dataclass
class TextDiffusionSample:
    noisy: torch.Tensor
    clean: torch.Tensor
    t: float


class TextDiffusionDataset(Dataset):
    """Simple text diffusion dataset.

    Each item is a tuple ``(noisy, clean, t)`` where ``noisy`` is produced by
    randomly replacing tokens with probability ``t``.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int, vocab_size: int = 256):
        if tokens.dim() != 1:
            raise ValueError("tokens must be a 1-D tensor")
        self.tokens = tokens
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_seqs = len(tokens) // seq_len

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_seqs

    def __getitem__(self, idx: int) -> TextDiffusionSample:
        start = idx * self.seq_len
        clean = self.tokens[start : start + self.seq_len]
        t = torch.rand(()).item()
        mask = torch.rand(self.seq_len) < t
        noisy = clean.clone()
        if mask.any():
            noisy[mask] = torch.randint(
                0, self.vocab_size, (int(mask.sum()),), dtype=torch.long
            )
        return TextDiffusionSample(noisy=noisy, clean=clean, t=t)
