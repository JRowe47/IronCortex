import torch
import torch.nn.functional as F

from .model import CortexReasoner

# 9) Generation (mask-predict style, non-autoregressive outer loop)
# ==========================================================

@torch.no_grad()
def generate(model: CortexReasoner,
             prompt_tokens: torch.Tensor,  # [T0] Long
             T_total: int,
             max_outer_iters: int = 8,
             conf_threshold: float = 0.9) -> torch.Tensor:
    """Mask-predict generator with always-on inner loop."""
    device = next(model.parameters()).device
    T0 = prompt_tokens.shape[0]
    tokens = torch.cat([prompt_tokens, torch.full((T_total - T0,), model.V - 1, device=device, dtype=torch.long)], dim=0)  # fill with [MASK]
    conf = torch.zeros(T_total, device=device)
    H_prev, reg_mask_prev = model.zeros_state(device)

    for _ in range(max_outer_iters):
        focus_map = (tokens == (model.V - 1)) | (conf < conf_threshold)
        H_prev, reg_mask_prev, logits, traces = model.reasoning_loop(tokens, model.cfg.K_inner,
                                                                     focus_map, reg_mask_prev, H_prev)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        maxp = probs.max(dim=-1).values
        conf[:] = maxp
        tokens = torch.where(focus_map, pred, tokens)
        if bool((conf > conf_threshold).all()) and bool((tokens != (model.V - 1)).all()):
            break
    return tokens

