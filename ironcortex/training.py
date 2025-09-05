import random
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .model import CortexReasoner
from .corruptions import corrupt
from .utils import goodness


def _detach_model_state(model: CortexReasoner) -> None:
    """Detach stateful tensors to avoid cross-step autograd graphs."""
    if hasattr(model, "work"):
        model.work.slots = model.work.slots.detach()
    for region in getattr(model, "regions", []):
        region.detach_state()


# 8) FF Training Step (basic, single-sequence version)
# ==========================================================


@dataclass
class LossWeights:
    ff: float = 1.0
    rtd: float = 0.5
    denoise: float = 1.0
    critic: float = 0.25
    verify: float = 0.25
    facet: float = 1e-3


def train_step(
    model: CortexReasoner,
    optimizer: torch.optim.Optimizer,
    clean_tokens: torch.Tensor,  # [B,T] Long
    λ: LossWeights,
    device: torch.device,
) -> Dict[str, float]:
    """One training step mixing FF goodness, RTD, denoising, and tiny auxiliaries.

    NOTE: For clarity, this is a *single-step* trainer over a batch of independent
    sequences. It loops per sample (B small), running the full inner reasoning loop
    for positive and negative streams, then aggregates losses.

    Returns metrics dict with component losses and an approximate cross entropy.
    """
    model.train()
    _detach_model_state(model)

    B, T = clean_tokens.shape
    total_ff = 0.0
    total_rtd = 0.0
    total_denoise = 0.0
    total_critic = 0.0
    total_verify = 0.0
    total_ce = 0.0

    optimizer.zero_grad()
    total_loss_val = 0.0

    for b in range(B):
        tokens = clean_tokens[b]

        # Sample corruption mode
        mode = random.choices(["RTD", "SPAN", "BLOCK"], weights=[0.5, 0.3, 0.2])[0]
        x_neg, is_real, focus_map, denoise_targets, denoise_mask = corrupt(
            tokens, model.V, mode
        )

        # --- Positive stream ---
        _detach_model_state(model)
        H_prev, reg_mask_prev = model.zeros_state(device)
        focus_zero = torch.zeros_like(focus_map, dtype=torch.bool)
        H_pos, reg_mask_p, logits_pos, traces_pos = model.reasoning_loop(
            tokens, model.cfg.K_inner, focus_zero, reg_mask_prev, H_prev
        )
        ce_loss = F.cross_entropy(logits_pos.unsqueeze(0), tokens[-1].unsqueeze(0))

        # --- Negative stream ---
        _detach_model_state(model)
        H_prev, reg_mask_prev = model.zeros_state(device)
        H_neg, reg_mask_n, logits_neg, traces_neg = model.reasoning_loop(
            x_neg, model.cfg.K_inner, focus_map, reg_mask_prev, H_prev
        )

        # -------- FF per-region losses --------
        ff_loss = torch.tensor(0.0, device=device)
        # stack traces per region: list of [R,d] -> [K,R,d]
        if len(traces_pos) > 0:
            Hpos_stack = torch.stack(traces_pos, dim=0)  # [K,R,d]
        else:
            Hpos_stack = torch.zeros(0, model.R, model.d, device=device)

        if len(traces_neg) > 0:
            Hneg_stack = torch.stack(traces_neg, dim=0)  # [K,R,d]
        else:
            Hneg_stack = torch.zeros(0, model.R, model.d, device=device)

        for r in range(model.R):
            hpos = Hpos_stack[:, r, :] if Hpos_stack.numel() > 0 else None
            hneg = Hneg_stack[:, r, :] if Hneg_stack.numel() > 0 else None

            gpos = (
                goodness(hpos)
                if hpos is not None and hpos.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            gneg = (
                goodness(hneg)
                if hneg is not None and hneg.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            τ = model.reg_ff[r].tau
            L_pos = F.softplus(-(gpos - τ))
            L_neg = F.softplus(+(gneg - τ))
            ff_loss = ff_loss + (L_pos + L_neg)
            # Update τ (EMA) and gate usefulness (detach)
            if hpos is not None and hpos.numel() > 0:
                model.reg_ff[r].update_tau(float(gpos.detach().item()))
            gain = float((gpos - gneg).detach().item())
            model.gate.update_gain(r, gain)

        # -------- RTD loss (on negative stream motor state) --------
        motor_neg = H_neg[model.io_idxs["motor"]]  # [d]
        rtd_logits = model.rtdd(motor_neg).unsqueeze(0)  # [1,2]
        # Build a simple target: if any replacement happened -> 0 else 1 (crude)
        rtd_target = (
            torch.tensor([[1, 0]], device=device)
            if is_real.all()
            else torch.tensor([[0, 1]], device=device)
        )
        rtd_loss = F.cross_entropy(rtd_logits, rtd_target.argmax(dim=-1))

        # -------- Denoising loss (mask-predict) on masked positions --------
        token_logits, aux = model.lm_head(motor_neg)  # [V]
        if bool(denoise_mask.any()):
            # For simplicity, compare against the *first* masked target
            first_idx = int(denoise_mask.nonzero(as_tuple=False)[0].item())
            target_id = denoise_targets[first_idx].unsqueeze(0)  # [1]
            ce = F.cross_entropy(token_logits.unsqueeze(0), target_id)
            denoise_loss = (
                ce
                + aux.get("facet_balance", torch.tensor(0.0, device=device)) * λ.facet
            )
        else:
            denoise_loss = torch.tensor(0.0, device=device)

        # -------- Critic regression (predict realized Δgoodness) --------
        realized_delta_g = (goodness(H_pos) - goodness(H_neg)).detach()
        # ws_state (neg stream) proxy: mean over final active
        active_n = reg_mask_n.nonzero(as_tuple=False).squeeze(-1)
        ws_state_neg = (
            H_neg[active_n].mean(dim=0)
            if active_n.numel() > 0
            else torch.zeros(model.d, device=device)
        )
        # Use plan->critic on that
        p, g = model.plan(ws_state_neg, H_neg[model.io_idxs["motor"]])
        v_hat = model.critic(ws_state_neg, g)
        critic_loss = F.mse_loss(v_hat, realized_delta_g)

        # -------- Verifier auxiliary --------
        ver_score = model.verify(motor_neg)
        # Structural target: encourage "OK" (1.0) when focus existed (we attempted a repair)
        ver_target = torch.tensor(1.0 if bool(focus_map.any()) else 0.5, device=device)
        verifier_loss = F.binary_cross_entropy(ver_score, ver_target)

        # -------- Total --------
        total = (
            λ.ff * ff_loss
            + λ.rtd * rtd_loss
            + λ.denoise * denoise_loss
            + λ.critic * critic_loss
            + λ.verify * verifier_loss
        )

        # Backprop immediately to avoid building large graphs and double backward errors
        total.backward()
        total_loss_val += float(total.detach().item())

        # Metrics (detach)
        total_ff += float(ff_loss.detach().item())
        total_rtd += float(rtd_loss.detach().item())
        total_denoise += float(denoise_loss.detach().item())
        total_critic += float(critic_loss.detach().item())
        total_verify += float(verifier_loss.detach().item())
        total_ce += float(ce_loss.detach().item())

        # Homeostasis update (drifts slowly)
        model.gate.update_homeo(reg_mask_n)

    optimizer.step()

    return {
        "ff": total_ff / B,
        "rtd": total_rtd / B,
        "denoise": total_denoise / B,
        "critic": total_critic / B,
        "verify": total_verify / B,
        "ce": total_ce / B,
        "total": total_loss_val / B,
    }
