import math

import torch

from ironcortex.attention.adaptive_filter_attention import AdaptiveFilterAttention


def test_trivial_reduces_to_dot_product():
    B, T, D = 1, 4, 8
    x = torch.randn(B, T, D)
    afa = AdaptiveFilterAttention(
        d_model=D,
        n_head=1,
        alpha=0.0,
        sigma_proc=0.0,
        eta_obs=0.0,
        exact_threshold=0,
    )
    with torch.no_grad():
        eye = torch.eye(D)
        afa.q_proj.weight.copy_(eye)
        afa.k_proj.weight.copy_(eye)
        afa.v_proj.weight.copy_(eye)
        afa.out_proj.weight.copy_(eye)
        afa.q_proj.bias.zero_()
        afa.k_proj.bias.zero_()
        afa.v_proj.bias.zero_()
        afa.out_proj.bias.zero_()
    out = afa(x)
    q = k = v = x
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    ref = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-5)
