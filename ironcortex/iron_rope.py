import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pad_batch, masked_mean, extract_spans

# 1) Iron RoPE: Fourier banks, rotary rotation, relative bias
# ==========================================================


def make_freq_bank(
    m: int,
    d_coord: int,
    kind: str = "log",
    base: float = 10000.0,
    sigma: float = 1.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Construct a frequency bank W in R^{m x d_coord}."""
    if kind == "log":
        steps = torch.logspace(
            math.log(1.0 / base), 0.0, steps=m, base=math.e, device=device, dtype=dtype
        )
        W = torch.zeros(m, d_coord, device=device, dtype=dtype)
        for j in range(d_coord):
            W[:, j] = steps
        return W
    elif kind == "gaussian":
        return torch.randn(m, d_coord, device=device, dtype=dtype) * sigma
    else:
        raise ValueError(f"Unknown freq kind: {kind}")


def sincos(W: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Fourier features: [sin(Wp), cos(Wp)].

    W: [m, d], p: [..., d] -> [..., 2m]
    """
    Θ = p @ W.T
    return torch.cat([torch.sin(Θ), torch.cos(Θ)], dim=-1)


def rope_rotate_pairs(
    x: torch.Tensor, cos_th: torch.Tensor, sin_th: torch.Tensor, m_pairs: int
) -> torch.Tensor:
    """Rotate the first 2*m_pairs channels in (pairs) by angle arrays.

    x: [..., 2*m_pairs + rest]
    cos_th/sin_th: arrays broadcastable to [..., m_pairs]
    """
    if m_pairs <= 0:
        return x
    *prefix, D = x.shape
    rot = x[..., : 2 * m_pairs].reshape(*prefix, m_pairs, 2)
    x0, x1 = rot[..., 0], rot[..., 1]
    c, s = cos_th, sin_th
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    y = torch.stack((y0, y1), dim=-1).reshape(*prefix, 2 * m_pairs)
    return torch.cat([y, x[..., 2 * m_pairs :]], dim=-1)


def relative_fourier_bias(
    p_q: torch.Tensor,
    p_k: torch.Tensor,
    W: torch.Tensor,
    beta_cos: torch.Tensor,
    beta_sin: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Relative bias b(Δp) via Fourier features.

    p_q: [B,Tq,d], p_k: [B,Tk,d]
    W: [m,d], beta_cos/sin: [m] or [H,m]
    Returns:
      [B,1,Tq,Tk] if shared across heads, else [B,H,Tq,Tk]
    """
    B, Tq, d = p_q.shape
    Δ = p_q.unsqueeze(2) - p_k.unsqueeze(1)  # [B,Tq,Tk,d]
    S = torch.einsum("bqkd,md->bqkm", Δ, W)  # [B,Tq,Tk,m]
    c, s = torch.cos(S), torch.sin(S)
    if beta_cos.dim() == 2:  # headwise [H,m]
        b = (
            c.unsqueeze(1) * beta_cos[None, None, None, :]
            + s.unsqueeze(1) * beta_sin[None, None, None, :]
        ).sum(
            -1
        )  # [B,H,Tq,Tk]
    else:
        b = (c * beta_cos + s * beta_sin).sum(-1).unsqueeze(1)  # [B,1,Tq,Tk]
    return scale * b


class IronRoPESelfAttention(nn.Module):
    """Bidirectional (default) self-attention with Iron RoPE on q/k and relative Fourier bias.

    Designed for diffusion/mask-predict:
    - `update_mask`: only those token positions are *updated*; others pass-through
    - `coords`: per-token coordinates (e.g., [pos, pos/T])
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        use_iron_rope: bool = True,
        rope_m: int = 64,
        rope_coord_dim: int = 1,
        rope_kind: str = "log",
        rope_base: float = 10000.0,
        rope_sigma: float = 1.0,
        use_fourier_bias: bool = True,
        fb_m: int = 32,
        fb_coord_dim: int = 1,
        fb_kind: str = "gaussian",
        fb_base: float = 10000.0,
        fb_sigma: float = 1.0,
        fb_headwise: bool = False,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.causal = causal

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.adrop = nn.Dropout(attn_pdrop)
        self.rdrop = nn.Dropout(resid_pdrop)

        # Attention mask
        if causal:
            mask = torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            )
        else:
            mask = torch.ones(block_size, block_size).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask, persistent=False)

        # RoPE
        self.use_rope = use_iron_rope
        self.rope_m = min(self.head_dim // 2, rope_m) if use_iron_rope else 0
        if self.rope_m > 0:
            self.register_buffer(
                "W_rope",
                make_freq_bank(
                    self.rope_m, rope_coord_dim, rope_kind, rope_base, rope_sigma
                ),
            )

        # Relative Fourier bias
        self.use_fb = use_fourier_bias
        self.fb_m = fb_m if use_fourier_bias else 0
        if self.fb_m > 0:
            self.register_buffer(
                "W_fb",
                make_freq_bank(self.fb_m, fb_coord_dim, fb_kind, fb_base, fb_sigma),
            )
            if fb_headwise:
                self.beta_cos = nn.Parameter(torch.zeros(self.n_head, self.fb_m))
                self.beta_sin = nn.Parameter(torch.zeros(self.n_head, self.fb_m))
            else:
                self.beta_cos = nn.Parameter(torch.zeros(self.fb_m))
                self.beta_sin = nn.Parameter(torch.zeros(self.fb_m))
            self.fb_scale = 1.0 / math.sqrt(self.fb_m)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        dropout: bool = True,
    ) -> torch.Tensor:
        """x:[B,T,C]; coords:[B,T,d_coord] or None; update_mask/context_mask:[B,T]"""
        B, T, C = x.shape
        assert T <= self.block_size
        device = x.device

        if coords is None:
            coords = (
                torch.arange(T, device=device, dtype=torch.float32)
                .view(1, T, 1)
                .expand(B, T, 1)
            )
        elif coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(B, -1, -1)

        qkv = self.c_attn(x)  # [B,T,3C]
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Iron RoPE rotation on q/k
        if self.rope_m > 0:
            Θ = coords @ self.W_rope.T  # [B,T,m]
            c = torch.cos(Θ).unsqueeze(1)  # [B,1,T,m]
            s = torch.sin(Θ).unsqueeze(1)
            q = rope_rotate_pairs(q, c, s, self.rope_m)
            k = rope_rotate_pairs(k, c, s, self.rope_m)

        # Attention logits
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B,H,T,T]

        # Relative Fourier bias
        if self.fb_m > 0:
            b = relative_fourier_bias(
                coords, coords, self.W_fb, self.beta_cos, self.beta_sin, self.fb_scale
            )
            att = att + b

        # Masks
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        if context_mask is not None:
            cm = context_mask[:, None, None, :].to(torch.bool)  # [B,1,1,T]
            att = att.masked_fill(~cm, float("-inf"))

        # Softmax and (optional) dropout
        att = F.softmax(att, dim=-1)
        if dropout:
            att = self.adrop(att)

        # Weighted sum
        y = att @ v  # [B,H,T,hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if dropout:
            y = self.rdrop(y)

        if update_mask is not None:
            um = update_mask.unsqueeze(-1).to(y.dtype)  # [B,T,1]
            y = x + y * um  # update only masked positions
        else:
            y = x + y
        return y


class LocalTokenMixer(nn.Module):
    """A tiny, bidirectional, Iron-RoPE attention block for denoising/mask-predict.

    - Builds a few **anchor** pseudo tokens that summarize context at multiple scales.
    - Uses Iron RoPE so attention depends on relative positions, helping mask repair.
    - Only updates **focused** tokens (the corrupted / low-confidence ones).

    Returns a pooled vector to feed into the sensor/motor regions.
    """

    def __init__(
        self, d: int, n_head: int = 4, block_size: int = 8192, m_tok: int = 64
    ):
        super().__init__()
        self.attn = IronRoPESelfAttention(
            n_embd=d,
            n_head=n_head,
            block_size=block_size,
            use_iron_rope=True,
            rope_m=m_tok,
            rope_coord_dim=2,
            use_fourier_bias=True,
            fb_m=32,
            fb_coord_dim=1,
            causal=False,
        )
        self.anch_embed = nn.Parameter(
            torch.randn(3, d) * 0.02
        )  # center / span_start / span_end

    @torch.no_grad()
    def build_anchors(self, T: int, focus_mask: torch.Tensor, strides=(128, 32)):
        anchors = []
        for S in strides:
            centers = list(range(S // 2, T, S))
            anchors += [(i, 0) for i in centers]  # kind 0
        # span boundary anchors around focused spans
        if focus_mask is not None and focus_mask.any():
            for start, end in extract_spans(focus_mask):
                anchors.append((start, 1))
                anchors.append((max(start, end - 1), 2))
        return anchors

    def forward(
        self,
        tok_emb: torch.Tensor,  # [B,T,d]
        pos_coords: torch.Tensor,  # [B,T,2]  (e.g., [i, i/T])
        focus_mask: torch.Tensor,  # [B,T]    (True = update this pos)
        ws_slots: Optional[torch.Tensor] = None,
        use_dropout: bool = True,
    ) -> torch.Tensor:
        B, T, d = tok_emb.shape
        device = tok_emb.device
        X_list, C_list, UM_list = [], [], []
        for b in range(B):
            x = tok_emb[b]  # [T,d]
            c = pos_coords[b]  # [T,2]
            fm = focus_mask[b]
            anchors = self.build_anchors(T, fm)
            if len(anchors) > 0:
                a_idx, a_kind = zip(*anchors)
                a_idx = torch.tensor(a_idx, dtype=torch.long, device=device)
                a_kind = torch.tensor(a_kind, dtype=torch.long, device=device)
                a_emb = self.anch_embed[a_kind]  # [Na,d]
                a_coords = torch.stack(
                    [a_idx.to(torch.float32), a_idx.to(torch.float32) / (T + 1e-9)],
                    dim=-1,
                )
                x = torch.cat([x, a_emb], dim=0)
                c = torch.cat([c, a_coords], dim=0)
                um = torch.cat(
                    [fm, torch.zeros(len(a_idx), device=device, dtype=torch.bool)],
                    dim=0,
                )
            else:
                um = fm
            X_list.append(x)
            C_list.append(c)
            UM_list.append(um)
        X, C, UM = pad_batch(
            X_list, C_list, UM_list
        )  # [B,Taug,d], [B,Taug,2], [B,Taug]
        Y = self.attn(
            X, coords=C, update_mask=UM, context_mask=None, dropout=use_dropout
        )
        Y_main = Y[:, :T, :]
        pooled = masked_mean(Y_main, focus_mask)  # [B,d]
        return pooled
