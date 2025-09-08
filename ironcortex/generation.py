import torch
import torch.nn.functional as F

from .model import CortexReasoner
from .thinking import Thinker

# 9) Generation (mask-predict style, non-autoregressive outer loop)
# ==========================================================


@torch.no_grad()
def generate(
    model: CortexReasoner,
    prompt_tokens: torch.Tensor,  # [T0] Long
    T_total: int,
    max_outer_iters: int = 8,
    conf_threshold: float = 0.9,
    n_samples: int = 1,
) -> torch.Tensor:
    """Mask-predict generator with optional best-of-N sampling and energy refinement."""
    device = next(model.parameters()).device
    thinker = Thinker(model.verify, max_steps=3, alpha=(2e-2, 5e-2), sigma=0.01)
    best_tokens = None
    best_energy = None

    for _ in range(n_samples):
        T0 = prompt_tokens.shape[0]
        tokens = torch.cat(
            [
                prompt_tokens,
                torch.full(
                    (T_total - T0,),
                    model.V - 1,
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        conf = torch.zeros(T_total, device=device)
        H_prev, reg_mask_prev = model.zeros_state(device)

        for _ in range(max_outer_iters):
            focus_map = (tokens == (model.V - 1)) | (conf < conf_threshold)
            H_prev, reg_mask_prev, logits, traces = model.reasoning_loop(
                tokens, model.cfg.K_inner, focus_map, reg_mask_prev, H_prev
            )
            motor_state = H_prev[model.io_idxs["motor"]]
            probs = F.softmax(logits, dim=-1)
            with torch.enable_grad():
                _, refined, _ = thinker.optimize(motor_state, probs)
            probs = refined.softmax(dim=-1)
            # avoid predicting the padding token (V-1)
            probs[:, -1] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            pred = torch.multinomial(probs, num_samples=1).squeeze(-1)
            maxp = probs.max(dim=-1).values
            conf[:] = maxp
            tokens = torch.where(focus_map, pred, tokens)
            if bool((conf > conf_threshold).all()) and bool(
                (tokens != (model.V - 1)).all()
            ):
                break

        energy = model.verify(motor_state, probs)
        if (best_energy is None) or (energy < best_energy):
            best_energy = energy
            best_tokens = tokens.clone()

    return best_tokens
