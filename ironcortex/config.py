from dataclasses import dataclass


@dataclass
class CortexConfig:
    R: int = 32
    d: int = 128
    V: int = 256  # vocab size (e.g., bytes)
    sdr_k: int = 16  # active bits in sparse token encoding
    K_inner: int = 8
    B_br: int = 2
    train_deterministic_inner_loop: bool = False
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
    profile: bool = False
    profile_every_n_steps: int = 0
    surprise_lambda: float = 0.0
    surprise_lambda_schedule: int = 0
    tau_kappa: float = 0.0
    tau_target_prec: float = 1.0
    # KWTA sparsity controls
    kwta_k: int = 0  # 0 -> use d//8
    kwta_k_start: int = 0  # 0 -> same as kwta_k
    kwta_k_schedule: int = 0
    kwta_soft_mode: bool = False
    kwta_soft_temp: float = 1.0
    disable_kwta_during_gating: bool = False
    # Attention debugging: force exact path or threshold for kernel path
    debug_exact: bool = False
    afa_exact_threshold: int = 64

    def __post_init__(self) -> None:
        if self.sdr_k <= 0 or self.sdr_k > self.d:
            self.sdr_k = max(1, self.d // 8)
