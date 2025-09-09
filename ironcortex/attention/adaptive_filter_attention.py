import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFilterAttention(nn.Module):
    """Simplified Adaptive Filter Attention.

    This module approximates the paper's formulation with a
    time-decay kernel. When ``alpha == 0`` and noise terms approach
    zero it reduces to standard dot-product attention.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dt: float = 1.0,
        alpha: float = 0.0,
        sigma_proc: float = 0.0,
        eta_obs: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim**-0.5
        self.dt = dt
        # Parameters controlling temporal decay / precision.
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.sigma_proc = nn.Parameter(torch.tensor(sigma_proc))
        self.eta_obs = nn.Parameter(torch.tensor(eta_obs))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer("last_attn_energy", torch.tensor(0.0), persistent=False)

    def build_time_kernels(self, T: int) -> torch.Tensor:
        """Return a kernel ``k[t] = exp(-alpha * t * dt)`` for ``t`` in ``[0, T)``."""
        device = self.alpha.device
        tau = torch.arange(T, device=device, dtype=torch.float32)
        return torch.exp(-self.alpha * tau * self.dt)

    def pairwise_precision(self, lags: torch.Tensor) -> torch.Tensor:
        """Simple exponential precision falloff with distance."""
        return torch.exp(-self.eta_obs * lags * self.dt) / (self.sigma_proc + 1e-9)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]

        # apply temporal decay based on |i-j|
        kernels = self.build_time_kernels(T)  # [T]
        idx = torch.arange(T, device=x.device)
        lag = (idx.view(1, 1, T, 1) - idx.view(1, 1, 1, T)).abs()
        decay = kernels[lag]
        scores = scores * decay

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.last_attn_energy = (-(attn + 1e-9).log().mean()).detach()
        out = torch.matmul(attn, v)  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
