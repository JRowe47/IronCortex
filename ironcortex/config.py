from dataclasses import dataclass


@dataclass
class CortexConfig:
    R: int = 32
    d: int = 128
    V: int = 256  # vocab size (e.g., bytes)
    K_inner: int = 8
    B_br: int = 2
    k_active: int = 8
    max_T: int = 8192
    use_dropout: bool = False
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
