import torch
import torch.nn as nn

from .utils import RMSNorm, KWTA
from .iron_rope import rope_rotate_pairs, make_freq_bank

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
    def __init__(self, d: int, m_time_pairs: int = 16):
        super().__init__()
        self.d = d
        self.norm = RMSNorm(d)
        self.r_lin = nn.Linear(d, d, bias=False)
        self.k_lin = nn.Linear(d, d, bias=False)
        self.v_lin = nn.Linear(d, d, bias=False)
        self.o_lin = nn.Linear(d, d, bias=False)
        self.decay_param = nn.Parameter(torch.zeros(d))
        self.register_buffer("state_num", torch.zeros(d), persistent=False)
        self.register_buffer("state_den", torch.zeros(d), persistent=False)
        self.dt = 0
        # Iron time rotation pairs for v
        self.m_time = max(0, min(d // 8 // 2, m_time_pairs))
        if self.m_time > 0:
            self.W_time = make_freq_bank(self.m_time, 1, kind="log", base=10000.0)

    def decay_vec(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_param).pow(2.0)  # (0,1)

    def fast_forward(self):
        if self.dt == 0:
            return
        lam = self.decay_vec()  # [d]
        mul = lam.pow(self.dt)
        self.state_num = self.state_num * mul
        self.state_den = self.state_den * mul
        self.dt = 0

    def skip(self):
        self.dt += 1

    def step(self, x_in: torch.Tensor, step_pos_scalar: float) -> torch.Tensor:
        """One RWKV region update.

        x_in: [d] input vector, step_pos_scalar ∈ [0,1] inner-step position
        """
        x = self.norm(x_in)
        self.fast_forward()
        r = torch.sigmoid(self.r_lin(x))
        k = self.k_lin(x)                    # keep unrotated (exp(k) >= 0)
        v = self.v_lin(x)

        # Iron time rotation on v (encode inner-step time)
        if self.m_time > 0:
            Θ = step_pos_scalar * self.W_time.squeeze(-1)  # [m_time]
            c, s = torch.cos(Θ), torch.sin(Θ)
            v = rope_rotate_pairs(v, c, s, self.m_time)

        lam = self.decay_vec()
        w = torch.exp(k)  # positive
        self.state_num = self.state_num * lam + w * v
        self.state_den = self.state_den * lam + w
        y = r * (self.state_num / (self.state_den + 1e-9))  # [d]
        h = x + self.o_lin(y)
        h = KWTA(h, k=max(1, self.d // 8))
        return h


