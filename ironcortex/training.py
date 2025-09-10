import random
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .model import CortexReasoner
from .corruptions import corrupt
from .ff_energy import ff_energy_loss
from .constants import EPS_DIV


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
    sequences. The heavy reasoning loop now runs in parallel over the batch to
    improve throughput. Per‑sample bookkeeping (e.g. corruption targets and
    denoising masks) remains in lightweight Python loops.

    Returns metrics dict with component losses and an approximate cross entropy.
    """
    model.train()
    _detach_model_state(model)

    B, T = clean_tokens.shape

    optimizer.zero_grad()

    step = getattr(model, "_train_step", 0) + 1
    model._train_step = step
    if model.cfg.enable_ff_energy_alignment:
        if model.cfg.surprise_lambda_schedule > 0:
            warm = min(1.0, step / model.cfg.surprise_lambda_schedule)
        else:
            warm = 1.0
        lam_s = model.cfg.surprise_lambda * warm
    else:
        lam_s = 0.0
    model._lambda_s = lam_s

    # Prepare negative/corrupted streams for the whole batch.
    neg_tokens = []
    focus_maps = []
    denoise_targets = []
    denoise_masks = []
    is_real_flags = []
    for b in range(B):
        tokens = clean_tokens[b]
        mode = random.choices(["RTD", "SPAN", "BLOCK"], weights=[0.5, 0.3, 0.2])[0]
        x_neg, is_real, focus_map, d_targets, d_mask = corrupt(tokens, model.V, mode)
        neg_tokens.append(x_neg)
        focus_maps.append(focus_map)
        denoise_targets.append(d_targets)
        denoise_masks.append(d_mask)
        is_real_flags.append(bool(is_real.all()))

    neg_tokens = torch.stack(neg_tokens, dim=0)
    focus_neg = torch.stack(focus_maps, dim=0)
    focus_pos = torch.zeros_like(focus_neg, dtype=torch.bool)
    tokens_all = torch.cat([clean_tokens, neg_tokens], dim=0)
    focus_all = torch.cat([focus_pos, focus_neg], dim=0)

    # Run the reasoning loop once over all sequences (pos & neg).
    (
        H_all,
        reg_all,
        logits_all,
        _traces,
        energy_all,
        entropy_all,
    ) = model.reasoning_loop_batch(tokens_all, model.cfg.K_inner, focus_all)

    H_pos, H_neg = H_all[:B], H_all[B:]
    reg_mask_p, reg_mask_n = reg_all[:B], reg_all[B:]
    logits_pos, logits_neg = logits_all[:B], logits_all[B:]
    attn_energy_pos, attn_energy_neg = energy_all[:B], energy_all[B:]
    attn_entropy_pos, attn_entropy_neg = entropy_all[:B], entropy_all[B:]

    # Cross-entropy on final token (monitoring only).
    ce_loss = F.cross_entropy(logits_pos, clean_tokens[:, -1])

    mean_surprise = (
        torch.stack([r.surprise_ema for r in model.regions]).mean().to(device)
    )
    if model.cfg.enable_ff_energy_alignment:
        ms_vec = mean_surprise.expand(B)
        aux_pos = torch.stack([attn_energy_pos, ms_vec, attn_entropy_pos], dim=-1)
        aux_neg = torch.stack([attn_energy_neg, ms_vec, attn_entropy_neg], dim=-1)
    else:
        aux_pos = aux_neg = None

    # -------- FF per-region losses (using final states) --------
    ff_loss = torch.tensor(0.0, device=device)
    hebbian_updates = []
    if model.cfg.enable_forward_forward:
        for r in range(model.R):
            hpos = H_pos[:, r, :]
            hneg = H_neg[:, r, :]
            gpos = hpos.pow(2).mean(dim=-1)
            gneg = hneg.pow(2).mean(dim=-1)
            surprise = model.regions[r].surprise_ema.to(device)
            gpos_eff = gpos - lam_s * surprise
            gneg_eff = gneg - lam_s * surprise
            tau = model.reg_ff[r].tau
            L_pos = F.softplus(-(gpos_eff - tau))
            L_neg = F.softplus(+(gneg_eff - tau))
            ff_loss = ff_loss + (L_pos + L_neg).mean()

            mean_prec = (model.regions[r].state_var + EPS_DIV).reciprocal().mean()
            model.reg_ff[r].update_tau(
                gpos_eff.detach().mean(),
                mean_prec.detach(),
                kappa=model.cfg.tau_kappa,
                target_prec=model.cfg.tau_target_prec,
            )

            gain = float((gpos_eff - gneg_eff).detach().mean().item())
            model.gate.update_gain(r, gain)
            if gain > 0:
                for b in range(B):
                    if not bool(reg_mask_p[b, r]):
                        continue
                    post = H_pos[b, r].detach()
                    for s in model.neighbors[r]:
                        if not bool(reg_mask_p[b, s]):
                            continue
                        pre = H_pos[b, s].detach()
                        hebbian_updates.append((r, s, gain, post, pre))

    # -------- RTD loss (on negative stream motor state) --------
    motor_pos = H_pos[:, model.io_idxs["motor"], :]
    motor_neg = H_neg[:, model.io_idxs["motor"], :]
    rtd_logits = model.rtdd(motor_neg)
    rtd_target = (~torch.tensor(is_real_flags, device=device)).long()
    rtd_loss = F.cross_entropy(rtd_logits, rtd_target)

    # -------- Denoising loss (mask-predict) --------
    denoise_loss = torch.tensor(0.0, device=device)
    for b in range(B):
        logits_b, aux_b = model.lm_head(motor_neg[b])
        if bool(denoise_masks[b].any()):
            idx = int(denoise_masks[b].nonzero(as_tuple=False)[0].item())
            target_id = denoise_targets[b][idx].unsqueeze(0).to(device)
            ce = F.cross_entropy(logits_b.unsqueeze(0), target_id)
            denoise_loss = (
                denoise_loss
                + ce
                + aux_b.get("facet_balance", torch.tensor(0.0, device=device)) * λ.facet
            )
    denoise_loss = denoise_loss / max(1, B)

    # -------- Critic regression --------
    critic_loss = torch.tensor(0.0, device=device)
    if not model.cfg.train_deterministic_inner_loop:
        realized_delta_g = (
            H_pos.pow(2).mean(dim=-1).mean(dim=-1)
            - H_neg.pow(2).mean(dim=-1).mean(dim=-1)
        ).detach()
        ws_state_neg = torch.zeros(B, model.d, device=device)
        for b in range(B):
            active = reg_mask_n[b].nonzero(as_tuple=False).squeeze(-1)
            if active.numel() > 0:
                ws_state_neg[b] = H_neg[b, active].mean(dim=0)
        p, g = model.plan(ws_state_neg, motor_neg)
        v_hat = model.critic(ws_state_neg, g)
        critic_loss = F.mse_loss(v_hat, realized_delta_g)

    # -------- Verifier energy FF loss --------
    if model.verify is not None and model.cfg.enable_energy_verifier:
        y_pos = F.one_hot(clean_tokens[:, -1], num_classes=model.V).float()
        y_neg = F.softmax(logits_neg.detach(), dim=-1)
        tau_v = 0.0
        if model.cfg.enable_ff_energy_alignment and model.verify_state is not None:
            tau_v = model.verify_state.tau
        E_pos = model.verify(motor_pos.detach(), y_pos, aux=aux_pos)
        E_neg = model.verify(motor_neg.detach(), y_neg, aux=aux_neg)
        verifier_loss = ff_energy_loss(E_pos, E_neg, tau=tau_v)
        if model.cfg.enable_ff_energy_alignment and model.verify_state is not None:
            Ep = E_pos.mean().detach()
            En = E_neg.mean().detach()
            model.E_pos_mean = 0.99 * model.E_pos_mean + 0.01 * Ep
            model.E_neg_mean = 0.99 * model.E_neg_mean + 0.01 * En
            mid = 0.5 * (model.E_pos_mean + model.E_neg_mean)
            model.verify_state.update_tau(mid)
    else:
        E_pos = torch.zeros_like(attn_energy_pos)
        E_neg = torch.zeros_like(attn_energy_neg)
        verifier_loss = torch.tensor(0.0, device=device)

    total = (
        λ.ff * ff_loss
        + λ.rtd * rtd_loss
        + λ.denoise * denoise_loss
        + λ.critic * critic_loss
        + λ.verify * verifier_loss
    )

    total.backward()
    with torch.no_grad():
        for r, s, gain, post, pre in hebbian_updates:
            W = model.router.W_edge[f"{s}->{r}"].weight
            W.add_(1e-3 * gain * torch.outer(post, pre))

    model.gate.update_homeo(reg_mask_n.any(dim=0))

    optimizer.step()

    return {
        "ff": float(ff_loss.detach().item()),
        "rtd": float(rtd_loss.detach().item()),
        "denoise": float(denoise_loss.detach().item()),
        "critic": float(critic_loss.detach().item()),
        "verify": float(verifier_loss.detach().item()),
        "E_pos": float(E_pos.detach().mean().item()),
        "E_neg": float(E_neg.detach().mean().item()),
        "lambda_s": float(lam_s),
        "ce_last": float(ce_loss.detach().item()),
        "total": float(total.detach().item()),
    }
