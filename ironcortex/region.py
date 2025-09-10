import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import RMSNorm, KWTA, init_weights
from .iron_rope import rope_rotate_pairs, make_freq_bank
from .constants import EPS_DIV, MAX_EXP, MAX_NOISE

# 5) RWKV Region Cell (with Δt skip + time rotation on v)
# ==========================================================


class RWKVRegionCell(nn.Module):
    """RWKV-like region processor with per-channel decay and Δt skip handling.

    State:
      - state_num/state_den: EMA numerators/denominators (fast weights)
      - dt: number of skipped micro-steps (for exact fast-forward decay)

    We add a small Iron time rotation (inner-step) to **v** only to encode
    diffusion time without breaking w=exp(k) positivity.
    """

    def __init__(
        self,
        d: int,
        m_time_pairs: int = 16,
        init_decay: float = 0.25,
        *,
        enable_adaptive_filter_dynamics: bool = False,
        enable_radial_tangential_updates: bool = False,
        afd_noise_mode: str = "scalar",
        use_predictive_trace: bool = True,
    ):
        super().__init__()
        self.d = d
        self.enable_adaptive_filter_dynamics = enable_adaptive_filter_dynamics
        self.enable_radial_tangential_updates = enable_radial_tangential_updates
        self.afd_noise_mode = afd_noise_mode
        self.use_predictive_trace = use_predictive_trace
        self.norm = RMSNorm(d)
        self.r_lin = nn.Linear(d, d, bias=False)
        self.k_lin = nn.Linear(d, d, bias=False)
        self.v_lin = nn.Linear(d, d, bias=False)
        self.o_lin = nn.Linear(d, d, bias=False)
        decay = math.sqrt(init_decay)
        raw_init = math.log((1 / decay) - 1)
        self.raw_decay = nn.Parameter(torch.full((d,), raw_init))
        if afd_noise_mode == "scalar":
            self.process_noise_param = nn.Parameter(torch.tensor(-4.0))
            self.obs_noise_param = nn.Parameter(torch.tensor(0.0))
        else:
            self.process_noise_param = nn.Parameter(torch.full((d,), -4.0))
            self.obs_noise_param = nn.Parameter(torch.zeros(d))
        self.register_buffer("state_num", torch.zeros(d), persistent=False)
        self.register_buffer("state_den", torch.zeros(d), persistent=False)
        self.register_buffer("state_var", torch.zeros(d), persistent=False)
        self.register_buffer("surprise_ema", torch.tensor(0.0), persistent=False)
        self.surprise_beta = 0.99
        # Radial–tangential buffers
        self.radius_beta = 0.9
        self.register_buffer("radius", torch.zeros(1), persistent=False)
        self.register_buffer("radius_var", torch.zeros(1), persistent=False)
        self.radius_process_noise_param = nn.Parameter(torch.tensor(-4.0))
        self.radius_obs_noise_param = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("last_dir", torch.zeros(d), persistent=False)
        self.register_buffer("last_norm", torch.zeros(1), persistent=False)
        # Predictive trace for HTM-style temporal memory
        self.register_buffer("pred", torch.zeros(d), persistent=False)
        self.dt = 0
        # Iron time rotation pairs for v
        self.m_time = max(0, min(d // 8 // 2, m_time_pairs))
        if self.m_time > 0:
            self.register_buffer(
                "W_time", make_freq_bank(self.m_time, 1, kind="log", base=10000.0)
            )
        self.apply(init_weights)

    def decay_rate(self) -> torch.Tensor:
        return -F.softplus(self.raw_decay).clamp_min(-MAX_EXP)

    def decay(self, dt: int = 1) -> torch.Tensor:
        return torch.exp(self.decay_rate() * dt).clamp_min(EPS_DIV)

    def process_noise(self) -> torch.Tensor:
        pn = (
            F.softplus(self.process_noise_param).clamp_min(EPS_DIV).clamp_max(MAX_NOISE)
        )
        if pn.numel() == 1:
            pn = pn.expand(self.d)
        return pn

    def obs_noise(self) -> torch.Tensor:
        on = F.softplus(self.obs_noise_param).clamp_min(EPS_DIV).clamp_max(MAX_NOISE)
        if on.numel() == 1:
            on = on.expand(self.d)
        return on

    def radius_process_noise(self) -> torch.Tensor:
        return (
            F.softplus(self.radius_process_noise_param)
            .clamp_min(EPS_DIV)
            .clamp_max(MAX_NOISE)
        )

    def radius_obs_noise(self) -> torch.Tensor:
        return (
            F.softplus(self.radius_obs_noise_param)
            .clamp_min(EPS_DIV)
            .clamp_max(MAX_NOISE)
        )

    def fast_forward(self, dt: int):
        if dt <= 0:
            return
        lam = self.decay(dt=dt)
        self.state_num = self.state_num * lam
        self.state_den = self.state_den * lam
        if self.enable_adaptive_filter_dynamics:
            self.state_var = self.state_var * lam.pow(2) + self.process_noise() * dt

    def skip(self):
        self.dt += 1

    def predict(self, msg: torch.Tensor, alpha: float = 0.95):
        """Update predictive trace from router messages when inactive.

        alpha controls decay of the previous trace."""
        if not self.use_predictive_trace:
            return
        self.pred = alpha * self.pred + (1.0 - alpha) * msg

    def detach_state(self):
        """Detach fast-weight buffers to prevent graph ties across steps."""
        self.state_num = self.state_num.detach()
        self.state_den = self.state_den.detach()
        if self.use_predictive_trace:
            self.pred = self.pred.detach()
        self.state_var = self.state_var.detach()
        self.radius = self.radius.detach()
        self.radius_var = self.radius_var.detach()
        self.surprise_ema = self.surprise_ema.detach()

    def telemetry(self) -> dict:
        """Return lightweight telemetry for logging."""
        prec = (self.state_var + EPS_DIV).reciprocal()
        return {
            "state_var": float(self.state_var.mean().detach()),
            "state_prec": float(prec.mean().detach()),
            "surprise_ema": float(self.surprise_ema.detach()),
        }

    def step(self, x_in: torch.Tensor, step_pos_scalar: float) -> torch.Tensor:
        """One RWKV region update.

        x_in: [d] input vector, step_pos_scalar ∈ [0,1] inner-step position
        """
        if self.use_predictive_trace:
            x = self.norm(x_in + self.pred)
            self.pred.zero_()
        else:
            x = self.norm(x_in)
        self.fast_forward(self.dt + 1)
        self.dt = 0
        r = torch.sigmoid(self.r_lin(x))
        k = self.k_lin(x)  # keep unrotated (exp(k) >= 0)
        v = self.v_lin(x)

        # Iron time rotation on v (encode inner-step time)
        if self.m_time > 0:
            Θ = step_pos_scalar * self.W_time.squeeze(-1)  # [m_time]
            c, s = torch.cos(Θ), torch.sin(Θ)
            v = rope_rotate_pairs(v, c, s, self.m_time)

        w = torch.exp(k.clamp_max(MAX_EXP))  # positive

        if self.enable_adaptive_filter_dynamics:
            prior_num = self.state_num
            prior_den = self.state_den
            prior_var = self.state_var
            pred = prior_num / (prior_den + EPS_DIV)
            msg = w * v
            resid = msg - pred
            obs_var = torch.exp((-k).clamp_max(MAX_EXP)) * self.obs_noise() + EPS_DIV
            gain = prior_var / (prior_var + obs_var)
            gain = gain.clamp(EPS_DIV, 1 - EPS_DIV)
            new_state = pred + gain * resid
            self.state_num = new_state * (prior_den + w)
            self.state_den = prior_den + w
            self.state_var = (1 - gain) * prior_var
            state_prec = (prior_var + obs_var).reciprocal()
            surprise = (resid.pow(2) * state_prec).sum()
            self.surprise_ema = (
                self.surprise_beta * self.surprise_ema
                + (1 - self.surprise_beta) * surprise.detach()
            )
            y = r * new_state
        else:
            self.state_num = self.state_num + w * v
            self.state_den = self.state_den + w
            y = r * (self.state_num / (self.state_den + EPS_DIV))  # [d]

        if self.enable_radial_tangential_updates:
            norm = y.norm(2, dim=-1, keepdim=True).clamp_min(EPS_DIV)
            dir = y / norm
            self.last_dir = dir.detach()
            self.last_norm = norm.detach()
            h_dir = self.o_lin(dir)
            prior_var = self.radius_var + self.radius_process_noise()
            resid = norm - self.radius
            gain = prior_var / (prior_var + self.radius_obs_noise() + EPS_DIV)
            radius_upd = self.radius + gain * resid
            self.radius_var = (1 - gain) * prior_var
            self.radius = (
                self.radius_beta * self.radius + (1 - self.radius_beta) * radius_upd
            )
            h = x + self.radius * h_dir
        else:
            h = x + self.o_lin(y)
        h = KWTA(h, k=max(1, self.d // 8))
        return h
