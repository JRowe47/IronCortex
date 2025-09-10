from typing import Dict, List, Tuple

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CortexConfig
from .sdr import SparseTokenEncoder
from .utils import (
    uncertainty_from_logits,
    should_halt,
    goodness,
    context_logprob,
    schedule_burst,
    RMSNorm,
    RegionFFState,
    init_weights,
)
from .iron_rope import LocalTokenMixer
from .heads import (
    Workspace,
    PlannerHead,
    CriticHead,
    RTDHead,
    TokenHead_MFS,
)
from .energy import EnergyVerifierHead
from .gate import Gate, Router
from .region import RWKVRegionCell
from .constants import EPS_DIV


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
        self.embed = SparseTokenEncoder(self.V, self.d, cfg.sdr_k)

        # Regions
        self.regions = nn.ModuleList(
            [
                RWKVRegionCell(
                    self.d,
                    init_decay=cfg.init_decay,
                    enable_adaptive_filter_dynamics=cfg.enable_adaptive_filter_dynamics,
                    enable_radial_tangential_updates=cfg.enable_radial_tangential_updates,
                    afd_noise_mode=cfg.afd_noise_mode,
                    use_predictive_trace=cfg.use_predictive_trace,
                    kwta_k=cfg.kwta_k,
                    kwta_k_start=cfg.kwta_k_start,
                    kwta_k_schedule=cfg.kwta_k_schedule,
                    kwta_soft_mode=cfg.kwta_soft_mode,
                    kwta_soft_temp=cfg.kwta_soft_temp,
                    disable_kwta=cfg.disable_kwta_during_gating,
                )
                for _ in range(self.R)
            ]
        )

        # Gate & Router
        self.gate = Gate(self.R, self.neighbors, io_idxs)
        self.router = Router(
            self.neighbors,
            self.d,
            self.R,
            enable_precision_routed_messages=cfg.enable_precision_routed_messages,
            vectorized=cfg.router_vectorized,
            edge_transform_mode=cfg.edge_transform_mode,
        )

        # Heads & workspace
        self.rtdd = RTDHead(self.d)
        self.lm_head = TokenHead_MFS(self.d, self.V, Kf=8)
        self.region_heads = nn.ModuleList(
            [nn.Linear(self.d, self.V) for _ in range(self.R)]
        )
        self.work = Workspace(self.d, N_slots=8)
        if cfg.train_deterministic_inner_loop:
            self.plan = None
            self.critic = None
        else:
            self.plan = PlannerHead(self.d)
            self.critic = CriticHead(self.d)
        self.verify = None
        self.verify_state = None
        if cfg.enable_energy_verifier:
            aux_dim = 3 if cfg.enable_ff_energy_alignment else 0
            self.verify = EnergyVerifierHead(
                self.d,
                self.V,
                hidden=self.d,
                aux_dim=aux_dim,
            )
            if cfg.enable_ff_energy_alignment:
                self.verify_state = RegionFFState()
                self.register_buffer("E_pos_mean", torch.tensor(0.0), persistent=False)
                self.register_buffer("E_neg_mean", torch.tensor(0.0), persistent=False)

        # Local token mixer (Iron RoPE or Adaptive Filter Attention)
        self.local_mix = LocalTokenMixer(
            self.d,
            n_head=4,
            block_size=cfg.max_T,
            m_tok=64,
            use_afa=cfg.enable_afa_attention,
        )
        self.norm_in = RMSNorm(self.d)

        # Per-region FF threshold τ
        self.reg_ff = nn.ModuleList([RegionFFState() for _ in range(self.R)])
        self.apply(init_weights)
        self._profile_times = {
            "attention": 0.0,
            "routing": 0.0,
            "region": 0.0,
            "total": 0.0,
        }
        self._profile_mem = {"attention": 0, "routing": 0, "region": 0}

    def detach_state(self) -> None:
        """Detach stateful tensors to avoid cross-step autograd graphs."""
        if hasattr(self, "work"):
            self.work.slots = self.work.slots.detach()
        for region in getattr(self, "regions", []):
            region.detach_state()

    def telemetry(self) -> dict:
        """Collect telemetry from regions, router, and attention modules."""
        reg_stats = [r.telemetry() for r in self.regions]
        state_var_mean = float(
            torch.tensor([r["state_var"] for r in reg_stats]).mean().item()
        )
        state_prec_mean = float(
            torch.tensor([r["state_prec"] for r in reg_stats]).mean().item()
        )
        surprise_mean = float(
            torch.tensor([r["surprise_ema"] for r in reg_stats]).mean().item()
        )
        attn_entropy = 0.0
        attn = getattr(self.local_mix, "attn", None)
        afa_telem = None
        if attn is not None and hasattr(attn, "telemetry"):
            afa_telem = attn.telemetry()
            attn_entropy = afa_telem.get("attn_entropy_mean", 0.0)
        metrics = {
            "regions": reg_stats,
            "routing_weight_mean": self.router.last_weight_mean,
            "router_weight_entropy": self.router.last_weight_entropy,
            "state_var_mean": state_var_mean,
            "state_prec_mean": state_prec_mean,
            "surprise_ema": surprise_mean,
            "attn_entropy_mean": attn_entropy,
        }
        if self.verify_state is not None:
            metrics["E_pos_mean"] = float(self.E_pos_mean.detach())
            metrics["E_neg_mean"] = float(self.E_neg_mean.detach())
            metrics["verifier_tau"] = float(self.verify_state.tau.detach())
        if afa_telem is not None:
            metrics["afa"] = afa_telem
        return metrics

    def report_profile(self) -> None:
        total = self._profile_times.get("total", 0.0)
        if total <= 0:
            return
        a = self._profile_times["attention"] / total * 100.0
        r = self._profile_times["routing"] / total * 100.0
        u = self._profile_times["region"] / total * 100.0
        am = self._profile_mem["attention"] / 1e6
        rm = self._profile_mem["routing"] / 1e6
        um = self._profile_mem["region"] / 1e6
        msg = (
            f"profile time%%: attn={a:.1f} rout={r:.1f} reg={u:.1f}; "
            f"memMB: attn={am:.1f} rout={rm:.1f} reg={um:.1f}"
        )
        print(msg)
        for k in self._profile_times:
            self._profile_times[k] = 0.0
        for k in self._profile_mem:
            self._profile_mem[k] = 0

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
        u_mean: float,
        burst_extra_k: int,
    ):
        """One inner micro-step: gate -> route -> active region updates -> heads."""
        device = H_prev.device
        T = tokens.shape[0]

        def _mem():
            return (
                torch.cuda.max_memory_allocated(device)
                if torch.cuda.is_available()
                else 0
            )

        if self.cfg.profile:
            total_start = time.perf_counter()

        # --- 0) Build a sensor vector via local Iron mixer (update only focus) ---
        tok_emb = self.embed(tokens).unsqueeze(0)  # [1,T,d]
        pos = torch.stack(
            [
                torch.arange(T, device=device, dtype=torch.float32),
                torch.arange(T, device=device, dtype=torch.float32) / (T + EPS_DIV),
            ],
            dim=-1,
        ).unsqueeze(0)
        focus_b = focus_map.unsqueeze(0)
        if self.cfg.profile:
            mem0 = _mem()
            t0 = time.perf_counter()
        sensor_vec = self.local_mix(
            tok_emb, pos, focus_b, ws_slots=None, use_dropout=self.cfg.ff_dropout
        )[
            0
        ]  # [d]
        if self.cfg.profile:
            self._profile_times["attention"] += time.perf_counter() - t0
            self._profile_mem["attention"] += max(0, _mem() - mem0)

        # --- 1) Gate selection ---
        H_hint = H_prev
        scores = self.gate.score_regions(H_hint, focus_map, u_mean=u_mean)
        reg_mask = self.gate.select_k(
            scores,
            k_active=self.cfg.k_active,
            burst_extra_k=burst_extra_k,
            io_force_on=True,
        )

        # --- 2) Router messages from previously active regions ---
        if self.cfg.profile:
            mem0 = _mem()
            t0 = time.perf_counter()
        M = self.router.messages(H_prev, reg_mask_prev, self.reg_coords)  # [R,d]
        if self.cfg.profile:
            self._profile_times["routing"] += time.perf_counter() - t0
            self._profile_mem["routing"] += max(0, _mem() - mem0)

        # --- 3) Region updates ---
        if self.cfg.profile:
            mem0 = _mem()
            t0 = time.perf_counter()
        H_cur = torch.zeros_like(H_prev)
        for r in range(self.R):
            if not bool(reg_mask[r]):
                # Update predictive trace from messages but skip actual update
                if self.cfg.use_predictive_trace:
                    self.regions[r].predict(M[r])
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
        if self.cfg.profile:
            self._profile_times["region"] += time.perf_counter() - t0
            self._profile_mem["region"] += max(0, _mem() - mem0)
            self._profile_times["total"] += time.perf_counter() - total_start

        # --- 4) Heads on motor & workspace ---
        motor_state = H_cur[self.io_idxs["motor"]]
        active = reg_mask.nonzero(as_tuple=False).squeeze(-1)

        # Active regions write their states to the workspace
        delta_slots = torch.zeros_like(self.work.slots)
        n_write = min(active.numel(), delta_slots.shape[0])
        if n_write > 0:
            delta_slots[:n_write] = H_cur[active[:n_write]]
            self.work.write(delta_slots)
            ws_state = self.work.slots.mean(dim=0)
        else:
            ws_state = torch.zeros(self.d, device=device)

        # Token predictions from motor and other active regions
        logits_list = []
        motor_logits, _ = self.lm_head(motor_state)
        logits_list.append(motor_logits)
        for r in active.tolist():
            if r == self.io_idxs["motor"]:
                continue
            logits_list.append(self.region_heads[r](H_cur[r]))
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
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
        if self.cfg.train_deterministic_inner_loop:
            traces: List[torch.Tensor] = []
            last_logits = torch.zeros(self.V, device=H_prev.device)
            for k in range(K_inner):
                u_mean = uncertainty_from_logits(last_logits) if k > 0 else 0.0
                self.gate.value_bias = 0.0
                H_cur, reg_mask, logits, _rtd, _ws, _motor = self.forward_inner_step(
                    tokens,
                    step_k=k,
                    focus_map=focus_map,
                    H_prev=H_prev,
                    reg_mask_prev=reg_mask_prev,
                    u_mean=u_mean,
                    burst_extra_k=0,
                )
                traces.append(H_cur.clone())
                H_prev, reg_mask_prev = H_cur, reg_mask
                last_logits = logits
            return H_cur, reg_mask, last_logits, traces

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
            burst_extra = schedule_burst(u_mean, v_hat, self.R)

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
                        u_mean=u_mean,
                        burst_extra_k=burst_extra,
                    )
                )

                ver_energy = self.verify(
                    motor_state,
                    F.softmax(logits.detach(), dim=-1),
                    attn_energy=(
                        self.local_mix.last_energy
                        if self.cfg.enable_ff_energy_alignment
                        else None
                    ),
                )
                verify_score = -ver_energy
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

    def reasoning_loop_batch(
        self,
        tokens_batch: torch.Tensor,
        K_inner: int,
        focus_batch: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[List[torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run reasoning_loop over a batch of token sequences.

        Returns stacked final states, masks, logits, and per-sample traces.
        """
        B = tokens_batch.shape[0]
        H_list, M_list, L_list, T_list, E_list, A_list = [], [], [], [], [], []
        device = tokens_batch.device
        for b in range(B):
            self.detach_state()
            H_prev, reg_mask_prev = self.zeros_state(device)
            H_cur, reg_mask, logits, traces = self.reasoning_loop(
                tokens_batch[b], K_inner, focus_batch[b], reg_mask_prev, H_prev
            )
            H_list.append(H_cur)
            M_list.append(reg_mask)
            L_list.append(logits)
            T_list.append(traces)
            E_list.append(self.local_mix.last_energy.detach())
            attn = getattr(self.local_mix, "attn", None)
            ent = getattr(attn, "last_attn_entropy", torch.tensor(0.0))
            A_list.append(ent.detach())
        return (
            torch.stack(H_list),
            torch.stack(M_list),
            torch.stack(L_list),
            T_list,
            torch.stack(E_list),
            torch.stack(A_list),
        )
