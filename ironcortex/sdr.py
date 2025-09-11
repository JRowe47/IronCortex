import torch
import torch.nn as nn
from torch import Tensor


class SparseTokenEncoder(nn.Module):
    """Fixed sparse distributed token encodings.

    Each vocabulary item is assigned a random binary vector with ``k`` active
    bits out of ``d`` dimensions. The codes are non-trainable and stored as a
    buffer to mirror HTM-style spatial pooler encoders.
    """

    def __init__(self, V: int, d: int, k: int) -> None:
        super().__init__()
        if k > d:
            raise ValueError("k must be <= d")
        self.V = V
        self.d = d
        self.k = k
        # Pre-generate random sparse binary codes
        codes = torch.zeros(V, d)
        for i in range(V):
            idx = torch.randperm(d)[:k]
            codes[i, idx] = 1.0
        self.register_buffer("codes", codes)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.codes[tokens]
