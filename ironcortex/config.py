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
    init_decay: float = 0.25
    # Optional dropout in goodness path (disabled by default)
    ff_dropout: bool = False
    enable_adaptive_filter_dynamics: bool = False
    enable_precision_routed_messages: bool = False
    enable_radial_tangential_updates: bool = False
    afd_noise_mode: str = "scalar"
    use_predictive_trace: bool = True
    enable_afa_attention: bool = False
    enable_ff_energy_alignment: bool = False
    router_vectorized: bool = True
    edge_transform_mode: str = "per_edge"
    enable_energy_verifier: bool = True
    enable_forward_forward: bool = True
    debug_metrics_every_n_steps: int = 0
    surprise_lambda: float = 0.0
    tau_kappa: float = 0.0
    tau_target_prec: float = 1.0
    # Attention debugging: force exact path or threshold for kernel path
    debug_exact: bool = False
    afa_exact_threshold: int = 64
