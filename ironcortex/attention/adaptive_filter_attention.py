import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import EPS_DIV, EPS_LOG, MAX_EXP


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
        *,
        debug_exact: bool = False,
        exact_threshold: int = 64,
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
        self.debug_exact = debug_exact
        self.exact_threshold = exact_threshold

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer("last_attn_energy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_kernel_norm", torch.tensor(0.0), persistent=False)
        # Cache for time kernels keyed by (T, alpha, dt)
        self._kernel_cache: dict[tuple[int, float, float], torch.Tensor] = {}

    def build_time_kernels(self, T: int) -> torch.Tensor:
        """Return a cached decay kernel ``k[t] = exp(-alpha * t * dt)``."""
        key = (T, float(self.alpha.item()), float(self.dt))
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            device = self.alpha.device
            tau = torch.arange(T, device=device, dtype=torch.float32)
            kernel = torch.exp(-self.alpha * tau * self.dt)
            self._kernel_cache[key] = kernel
        return kernel

    def pairwise_precision(self, lags: torch.Tensor) -> torch.Tensor:
        """Simple exponential precision falloff with distance."""
        return torch.exp((-self.eta_obs * lags * self.dt).clamp_max(MAX_EXP)) / (
            self.sigma_proc + EPS_DIV
        )

    def _exact_path(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        T = q.size(-2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        kernels = self.build_time_kernels(T)
        idx = torch.arange(T, device=q.device)
        lag = (idx.view(1, 1, T, 1) - idx.view(1, 1, 1, T)).abs()
        decay = kernels[lag] * self.pairwise_precision(lag)
        self.last_kernel_norm = decay.norm().detach()
        scores = scores * decay
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -MAX_EXP)
        attn = F.softmax(scores, dim=-1)
        if mask is not None:
            attn = attn * mask
            z = attn.sum(dim=-1, keepdim=True).clamp_min(EPS_DIV)
            attn = attn / z
        self.last_attn_energy = (-(attn + EPS_LOG).log().mean()).detach()
        self.last_attn_entropy = (
            (-(attn + EPS_LOG).log() * attn).sum(dim=-1).mean().detach()
        )
        out = torch.matmul(attn, v)
        return out

    def _kernel_path(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Placeholder kernelized path using exact computation."""
        # TODO: implement true linear-time kernelization via depthwise convolution
        return self._exact_path(q, k, v, mask)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.debug_exact or T <= self.exact_threshold:
            out = self._exact_path(q, k, v, mask)
        else:
            out = self._kernel_path(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    def telemetry(self) -> dict:
        """Return telemetry metrics for logging."""
        return {
            "alpha": float(self.alpha.detach()),
            "sigma_proc": float(self.sigma_proc.detach()),
            "eta_obs": float(self.eta_obs.detach()),
            "kernel_norm": float(self.last_kernel_norm),
            "attn_entropy_mean": float(self.last_attn_entropy),
        }
