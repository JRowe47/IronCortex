import torch

# 7) Corruptions for FF training (RTD / SPAN / BLOCK)
# ==========================================================


def corrupt(tokens: torch.Tensor, V: int, mode: str):
    """Return negative stream inputs and metadata.

    GPU-friendly implementation without host-side loops.

    tokens: [T] Long
    mode in {'RTD', 'SPAN', 'BLOCK'}
    Returns:
      x_neg:[T] Long, is_real:[T] Long in {0,1}, focus:[T] Bool,
      denoise_targets:[T] Long, denoise_mask:[T] Bool
    """
    T = tokens.shape[0]
    device = tokens.device

    if mode == "RTD":
        p = 0.15
        mask = torch.rand(T, device=device) < p
        repl = torch.randint(0, V, (T,), device=device)
        x_neg = torch.where(mask, repl, tokens)
        is_real = (x_neg == tokens).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    elif mode == "SPAN":
        avg_len = 5.0
        start = torch.randint(0, T, (1,), device=device, dtype=torch.float32)
        length = torch.poisson(torch.tensor([avg_len], device=device)).clamp(min=1)
        end = torch.clamp(start + length, max=float(T))
        positions = torch.arange(T, device=device, dtype=torch.float32)
        mask = (positions >= start) & (positions < end)
        MASK_ID = V - 1
        x_neg = torch.where(mask, torch.full_like(tokens, MASK_ID), tokens)
        is_real = (~mask).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    elif mode == "BLOCK":
        block_len = max(2, T // 8)
        start = torch.randint(0, T - block_len + 1, (1,), device=device)
        idx = torch.arange(T, device=device)
        mask = (idx >= start) & (idx < start + block_len)
        block_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        perm = block_idx[torch.randperm(block_len, device=device)]
        x_neg = tokens.clone()
        x_neg[block_idx] = x_neg[perm]
        is_real = (x_neg == tokens).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    else:
        raise ValueError("unknown mode")

    return x_neg, is_real, focus, denoise_targets, denoise_mask
