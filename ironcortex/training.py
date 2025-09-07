import random
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .model import CortexReasoner
from .corruptions import corrupt
from .utils import goodness
from .ff_energy import ff_energy_loss


def _detach_model_state(model: CortexReasoner) -> None:
    model.detach_state()


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

        # Batch positive and negative together
        focus_zero = torch.zeros_like(focus_map, dtype=torch.bool)
        x_all = torch.stack([tokens, x_neg], dim=0)
        focus_all = torch.stack([focus_zero, focus_map], dim=0)
        H_all, reg_all, logits_all, traces_all = model.reasoning_loop_batch(
            x_all, model.cfg.K_inner, focus_all
        )
        H_pos, H_neg = H_all
        reg_mask_p, reg_mask_n = reg_all
        logits_pos, logits_neg = logits_all
        traces_pos, traces_neg = traces_all

        ce_loss = F.cross_entropy(logits_pos.unsqueeze(0), tokens[-1].unsqueeze(0))

        # -------- FF per-region losses --------
        ff_loss = torch.tensor(0.0, device=device)
        Hpos_stack = (
            torch.stack(traces_pos, dim=0)
            if len(traces_pos) > 0
            else torch.zeros(0, model.R, model.d, device=device)
        )
        Hneg_stack = (
            torch.stack(traces_neg, dim=0)
            if len(traces_neg) > 0
            else torch.zeros(0, model.R, model.d, device=device)
        )

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
            if hpos is not None and hpos.numel() > 0:
                model.reg_ff[r].update_tau(gpos.detach())
            gain = float((gpos - gneg).detach().item())
            model.gate.update_gain(r, gain)

        # -------- RTD loss (on negative stream motor state) --------
        motor_pos = H_pos[model.io_idxs["motor"]]
        motor_neg = H_neg[model.io_idxs["motor"]]  # [d]
        rtd_logits = model.rtdd(motor_neg).unsqueeze(0)  # [1,2]
        rtd_target = (
            torch.tensor([[1, 0]], device=device)
            if is_real.all()
            else torch.tensor([[0, 1]], device=device)
        )
        rtd_loss = F.cross_entropy(rtd_logits, rtd_target.argmax(dim=-1))

        # -------- Denoising loss (mask-predict) on masked positions --------
        token_logits, aux = model.lm_head(motor_neg)  # [V]
        if bool(denoise_mask.any()):
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
        active_n = reg_mask_n.nonzero(as_tuple=False).squeeze(-1)
        ws_state_neg = (
            H_neg[active_n].mean(dim=0)
            if active_n.numel() > 0
            else torch.zeros(model.d, device=device)
        )
        p, g = model.plan(ws_state_neg, H_neg[model.io_idxs["motor"]])
        v_hat = model.critic(ws_state_neg, g)
        critic_loss = F.mse_loss(v_hat, realized_delta_g)

        # -------- Verifier energy FF loss --------
        y_pos = F.one_hot(tokens[-1], num_classes=model.V).float()
        y_neg = F.softmax(logits_neg.detach(), dim=-1)
        E_pos = model.verify(motor_pos.detach(), y_pos)
        E_neg = model.verify(motor_neg.detach(), y_neg)
        verifier_loss = ff_energy_loss(E_pos, E_neg, tau=0.0)

        # -------- Total --------
        total = (
            λ.ff * ff_loss
            + λ.rtd * rtd_loss
            + λ.denoise * denoise_loss
            + λ.critic * critic_loss
            + λ.verify * verifier_loss
        )

        total.backward()
        total_loss_val += float(total.detach().item())

        total_ff += float(ff_loss.detach().item())
        total_rtd += float(rtd_loss.detach().item())
        total_denoise += float(denoise_loss.detach().item())
        total_critic += float(critic_loss.detach().item())
        total_verify += float(verifier_loss.detach().item())
        total_ce += float(ce_loss.detach().item())

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
