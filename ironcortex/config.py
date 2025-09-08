from dataclasses import dataclass


@dataclass
class CortexConfig:
    R: int = 32
    d: int = 128
    V: int = 256  # vocab size (e.g., bytes)
    sdr_k: int = 16  # active bits in sparse token encoding
    K_inner: int = 8
    B_br: int = 2
    k_active: int = 8
    max_T: int = 8192
    # Optional dropout in goodness path (disabled by default)
    ff_dropout: bool = False
