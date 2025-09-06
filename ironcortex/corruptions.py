import torch

# 7) Corruptions for FF training (RTD / SPAN / BLOCK)
# ==========================================================


def corrupt(tokens: torch.Tensor, V: int, mode: str):
    """Return negative stream inputs and metadata.

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
        repl = torch.randint(low=0, high=V, size=(T,), device=device)
        x_neg = torch.where(mask, repl, tokens)
        is_real = (x_neg == tokens).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    elif mode == "SPAN":
        rate = 0.3
        avg_len = 5
        mask = torch.zeros(T, dtype=torch.bool, device=device)
        t = 0
        while t < T:
            if torch.rand((), device=device).item() < rate:
                L = torch.distributions.Exponential(1.0 / avg_len).sample().to(device)
                L = int(max(1, torch.round(L).item()))
                mask[t : min(T, t + L)] = True
                t += L
            else:
                t += 1
        x_neg = tokens.clone()
        MASK_ID = V - 1  # reserve last id as [MASK] if desired
        x_neg[mask] = MASK_ID
        is_real = (~mask).long()
        focus = mask
        denoise_targets = tokens
        denoise_mask = mask

    elif mode == "BLOCK":
        # Shuffle a random block
        block_len = max(2, T // 8)
        start = int(
            torch.randint(0, max(1, T - block_len + 1), (1,), device=device).item()
        )
        perm = torch.randperm(block_len, device=device)
        x_neg = tokens.clone()
        x_neg[start : start + block_len] = x_neg[start : start + block_len][perm]
        is_real = (x_neg == tokens).long()
        focus = torch.zeros(T, dtype=torch.bool, device=device)
        focus[start : start + block_len] = True
        denoise_targets = tokens
        denoise_mask = focus

    else:
        raise ValueError("unknown mode")

    return x_neg, is_real, focus, denoise_targets, denoise_mask
