from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .config import CortexConfig
from .utils import (
    uncertainty_from_logits,
    should_halt,
    goodness,
    context_logprob,
    RMSNorm,
    RegionFFState,
)
from .iron_rope import LocalTokenMixer
from .heads import (
    Workspace,
    PlannerHead,
    CriticHead,
    VerifierHead,
    RTDHead,
    TokenHead_MFS,
)
from .gate import Gate, Router
from .region import RWKVRegionCell


class CortexReasoner(nn.Module):
    """The top-level container: regions, gate, router, heads, workspace, and inner loop."""

    def __init__(
        self,
        neighbor_indices: List[List[int]],
        reg_coords: torch.Tensor,
        io_idxs: Dict[str, int],
        cfg: CortexConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.R = cfg.R
        self.d = cfg.d
        self.V = cfg.V
        self.io_idxs = io_idxs
        self.neighbors = neighbor_indices
        self.register_buffer("reg_coords", reg_coords.to(torch.float32))

        # Tokens
        self.embed = nn.Embedding(self.V, self.d)

        # Regions
        self.regions = nn.ModuleList([RWKVRegionCell(self.d) for _ in range(self.R)])

        # Gate & Router
        self.gate = Gate(self.R, self.neighbors, io_idxs)
        self.router = Router(self.neighbors, self.d, self.R)

        # Heads & workspace
        self.rtdd = RTDHead(self.d)
        self.lm_head = TokenHead_MFS(self.d, self.V, Kf=8)
        self.work = Workspace(self.d, N_slots=8)
        self.plan = PlannerHead(self.d)
        self.critic = CriticHead(self.d)
        self.verify = VerifierHead(self.d)

        # Local token mixer (Iron RoPE) and input norm
        self.local_mix = LocalTokenMixer(
            self.d,
            n_head=4,
            block_size=cfg.max_T,
            m_tok=64,
            attn_pdrop=cfg.attn_pdrop,
            resid_pdrop=cfg.resid_pdrop,
            use_dropout=cfg.use_dropout,
        )
        self.norm_in = RMSNorm(self.d)

        # Per-region FF threshold τ
        self.reg_ff = nn.ModuleList([RegionFFState() for _ in range(self.R)])

    def region_input(
        self, r: int, x_sensor: torch.Tensor, msg: torch.Tensor
    ) -> torch.Tensor:
        x = msg
        if r in (self.io_idxs["sensor"], self.io_idxs["motor"]):
            x = x + x_sensor
        return self.norm_in(x)

    def forward_inner_step(
        self,
        tokens: torch.Tensor,  # [T] Long
        step_k: int,
        focus_map: torch.Tensor,  # [T] Bool
        H_prev: torch.Tensor,  # [R,d]
        reg_mask_prev: torch.Tensor,  # [R] Bool
    ):
        """One inner micro-step: gate -> route -> active region updates -> heads."""
        device = H_prev.device
        T = tokens.shape[0]

        # --- 0) Build a sensor vector via local Iron mixer (update only focus) ---
        tok_emb = self.embed(tokens).unsqueeze(0)  # [1,T,d]
        pos = torch.stack(
            [
                torch.arange(T, device=device, dtype=torch.float32),
                torch.arange(T, device=device, dtype=torch.float32) / (T + 1e-9),
            ],
            dim=-1,
        ).unsqueeze(0)
        focus_b = focus_map.unsqueeze(0)
        sensor_vec = self.local_mix(tok_emb, pos, focus_b, ws_slots=None)[0]  # [d]

        # --- 1) Gate selection ---
        H_hint = H_prev
        scores = self.gate.score_regions(H_hint, focus_map, u_mean=0.0)
        reg_mask = self.gate.select_k(
            scores, k_active=self.cfg.k_active, burst_extra_k=0, io_force_on=True
        )

        # --- 2) Router messages from previously active regions ---
        M = self.router.messages(H_prev, reg_mask_prev, self.reg_coords)  # [R,d]

        # --- 3) Region updates ---
        H_cur = torch.zeros_like(H_prev)
        for r in range(self.R):
            if not bool(reg_mask[r]):
                self.regions[r].skip()
                continue
            sensor_or_zero = (
                sensor_vec
                if r in (self.io_idxs["sensor"], self.io_idxs["motor"])
                else torch.zeros(self.d, device=device)
            )
            x_r = self.region_input(r, sensor_or_zero, M[r])
            step_pos = float(step_k) / float(max(1, self.cfg.K_inner - 1))
            H_cur[r] = self.regions[r].step(x_r, step_pos_scalar=step_pos)

        # --- 4) Heads on motor & workspace ---
        motor_state = H_cur[self.io_idxs["motor"]]
        # workspace summary: mean over active regions (simple choice)
        active = reg_mask.nonzero(as_tuple=False).squeeze(-1)
        if active.numel() > 0:
            ws_state = H_cur[active].mean(dim=0)
        else:
            ws_state = torch.zeros(self.d, device=device)

        logits, aux = self.lm_head(motor_state)
        rtd_logits = self.rtdd(motor_state)

        return H_cur, reg_mask, logits, rtd_logits, ws_state, motor_state

    @torch.no_grad()
    def zeros_state(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.R, self.d, device=device), torch.zeros(
            self.R, dtype=torch.bool, device=device
        )

    def reasoning_loop(
        self,
        tokens: torch.Tensor,  # [T] Long
        K_inner: int,
        focus_map: torch.Tensor,  # [T] Bool
        reg_mask_prev: torch.Tensor,
        H_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Always-on latent reasoning loop (plan→critic→gate; branch; verify; halt)."""
        # For simplicity, we keep B_br branches as alternative snapshots of region states.
        branches = [
            (self.work.slots.clone(), reg_mask_prev.clone(), H_prev.clone())
            for _ in range(self.cfg.B_br)
        ]
        branch_scores = [
            torch.tensor(-1e9, device=H_prev.device) for _ in range(self.cfg.B_br)
        ]
        traces: List[torch.Tensor] = []

        last_logits = torch.zeros(self.V, device=H_prev.device)

        for k in range(K_inner):
            # Uncertainty from last logits (scalar)
            u_mean = uncertainty_from_logits(last_logits) if k > 0 else 0.0

            # Plan & value
            # (Simple: ws_state = mean of previous active states if any)
            active = reg_mask_prev.nonzero(as_tuple=False).squeeze(-1)
            if active.numel() > 0:
                ws_state_prev = H_prev[active].mean(dim=0)
            else:
                ws_state_prev = torch.zeros(self.d, device=H_prev.device)
            p, g = self.plan(ws_state_prev, H_prev[self.io_idxs["motor"]])
            v_hat = float(self.critic(ws_state_prev, g).item())

            # Gate bias by value-of-compute:
            self.gate.value_bias = v_hat

            # Try each branch
            for b in range(self.cfg.B_br):
                self.work.slots = branches[b][0].clone()
                H_prev_b, reg_mask_prev_b = (
                    branches[b][2].clone(),
                    branches[b][1].clone(),
                )

                H_cur, reg_mask, logits, rtd_logits, ws_state, motor_state = (
                    self.forward_inner_step(
                        tokens,
                        step_k=k,
                        focus_map=focus_map,
                        H_prev=H_prev_b,
                        reg_mask_prev=reg_mask_prev_b,
                    )
                )

                verify_score = self.verify(motor_state)  # scalar in [0,1]
                # Estimate Δgoodness (local): g(H_cur) - g(H_prev)
                delta_g = goodness(H_cur) - goodness(H_prev_b)
                score = (
                    0.6 * delta_g + 0.2 * verify_score + 0.2 * context_logprob(logits)
                )

                branches[b] = (self.work.slots.clone(), reg_mask.clone(), H_cur.clone())
                branch_scores[b] = score
                last_logits = logits

            # Keep top-2 branches
            top_idx = torch.topk(
                torch.stack(branch_scores), k=min(self.cfg.B_br, 2)
            ).indices.tolist()
            branches = [branches[i] for i in top_idx]
            branch_scores = [branch_scores[i] for i in top_idx]

            # Gather trace
            traces.append(branches[0][2].clone())

            if should_halt(branch_scores, u_mean, k, K_inner):
                break

            # Prepare next prev
            self.work.slots, reg_mask_prev, H_prev = branches[0]

        # Emit best branch
        best = int(torch.stack(branch_scores).argmax().item())
        self.work.slots, reg_mask, H_cur = branches[best]
        return H_cur, reg_mask, last_logits, traces
