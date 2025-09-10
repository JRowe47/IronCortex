TODO: read AGENTS.md completely

# Project TODOs

## Milestone 0 — Scaffolding, Flags & Safety Rails ✅

Objective: Ensure all features are toggleable, safe, and observable.

### Tasks
- [x] Config flags plumbing
  - RG: enable_adaptive_filter_dynamics|enable_afa_attention|EnergyVerifier|Forward-Forward
  - Add flags to CFG + CLI parser + default config file. Ensure accessible in model/region/router/attention constructors.
- [x] Safety clamps & epsilons
  - Clamp decay (decay_vec = -softplus(raw_decay)), noise scales (process_noise, obs_noise) with clamp_min(ε) and clamp_max(max_noise).
  - Centralize constants: EPS_DIV=1e-8, EPS_LOG=1e-12, MAX_EXP=40.0.
- [x] Telemetry hooks
  - Add per‑step logs: state_var_mean, state_prec_mean, surprise_ema, router_weight_entropy, attn_entropy_mean.
  - Gate logs by CFG.debug_metrics_every_n_steps.

### Acceptance
- With all flags off, unit/integration tests pass unchanged.
- No NaNs/Infs under random stress test (100 steps).

### Rollback
- Keep prior config defaults; guard all new paths behind flags.

## Milestone 1 — Vectorize Forward‑Forward Training Loop ✅

Objective: Remove per-sample Python loops; add batched inner loop.

### Tasks
- [x] Batch reasoning loop
  - RG: def reasoning_loop\(.*\): and/or for sample in batch, train_step
  - Add reasoning_loop_batch(inputs, targets, ...) that handles [B, T, ...] in one call, reusing the same micro‑step logic in parallel.
  - Keep original reasoning_loop as a thin wrapper that calls the batch version with B=1.
- [x] Refactor trainer
  - Replace per-sample iteration with one batched call; ensure FF goodness and energy losses compute per sample then reduce.
- Optional: micro‑step weighting
  - Add CFG.ff_goodness_aggregation ∈ {'final','mean','ema'}.
  - Implement 'final' (default for simplicity) and keep 'mean' for compatibility.

### Acceptance
- Throughput ↑ (≥1.3×) on a medium batch vs. baseline.
- Loss parity (±3%) with flags off.

### Rollback
- Keep old path behind use_batched_reasoning=False.

## Milestone 2 — Router Refactor: Vectorized, Robust, Lean ✅

Objective: Replace nested loops with batched message passing; fuse robust weights; reduce transform params.

### Tasks
- [x] Vectorized aggregation interface
  - RG: class Router, def messages|aggregate
  - Create Router.aggregate(H_prev, active_mask) returning M (messages per region).
  - Build tensors of all active edges:
    - src_idx, dst_idx (shape [E])
  - Gather H_prev[src_idx] → [E, d].
- [x] Query/Key precompute
  - Compute Q[dst] once; K[src] once (shared or per edge‑type).
  - scores = (Q[dst_idx] * K[src_idx]).sum(-1) / sqrt(d) → [E].
- [x] Robust weighting (fused)
  - Compute messages: msg = W_edge[src->dst](H_prev[src]) + FourierBias.
  - Residual: resid = msg - H_prev[dst_idx].
  - Mahalanobis: mah = (resid.pow(2) * P_edge[src->dst]).sum(-1).clamp_max(MAX_EXP/0.5).
  - Fused exponent: logw = scores - 0.5 * mah; normalize with scatter_logsumexp per destination; w = exp(logw - logZ[dst_idx]).
- [x] Scatter-add aggregation
  - M = zeros([R,d]); scatter_add(M[dst_idx], w.unsqueeze(-1)*msg).
  - Handle empty destinations: if no incoming edge, keep zeros.
- [x] Parameter reduction modes
  - Add CFG.edge_transform_mode:
    - per_edge (current)
    - by_direction: share W_edge for each of 6 hex directions; build a direction map upfront.
    - factorized: W = U @ diag(w_{edge}) @ V with global U,V and per‑edge small vector.

### Acceptance
- Functional parity with flags off.
- With robust routing on: synthetic test where one neighbor is corrupted → weight ↓≥80% vs clean neighbors.
- Vectorized path ≥1.2× faster on dense neighbor graphs.

### Rollback
- Keep old loop under CFG.router_vectorized=False.

## Milestone 3 — Adaptive Filter Dynamics: Streamline & Stabilize ✅

Objective: Simplify Kalman‑like updates; support dt>1 fast‑forward; limit parameter explosion.

### Tasks
- [x] Fast‑forward prior
  - RG: RWKVRegionCell, state_num/state_den, dt
  - Implement fast_forward(dt) that updates mean & variance with decay = exp(decay_vec*dt); reuses same function for dt=1.
- [x] Scalar vs. vector noise options
  - Add CFG.afd_noise_mode ∈ {'scalar','vector'}; default 'scalar' for stability.
  - If scalar: process_noise and obs_noise are scalars per region or per head.
- [x] Gain computation
  - Keep obs_var = exp(-k)*obs_noise + EPS_DIV.
  - Clamp gain to [0, 1]; avoid exact 0 or 1.
- [x] Predictive trace simplification
  - RG: pred or predictive buffer in region cell
  - Ablation switch: CFG.use_predictive_trace. If False, remove predictive accumulation and rely on prior variance growth when inactive.

### Acceptance
- No NaNs in long runs (≥10k micro‑steps) on random input.
- With afd_noise_mode='scalar', fewer params; equal or better stability vs. vector.

### Rollback
- Keep old update under enable_adaptive_filter_dynamics=False.

## Milestone 4 — Radial–Tangential Update: Remove Double Smoothing ✅

Objective: Reduce redundancy; keep one smoothing mechanism on the radius.

### Tasks
- [x] Single‑stage radius update
  - RG: radial_tangential, radius_var, radius_beta
  - Remove secondary EMA on radius; set radius = radius_upd.
  - If needed, keep small momentum: radius = (1-μ)*radius + μ*radius_upd with μ small (e.g., 0.2).
- [x] Ensure unit direction
  - Normalize dir = y / (||y|| + EPS_DIV); assert abs(||dir|| - 1) < 1e-3 in debug.

### Acceptance
- Same or faster adaptation of scale on step changes.
- Fewer state vars; code shorter.

### Rollback
- Reintroduce EMA via flag CFG.radius_double_ema=True if instability observed.

## Milestone 5 — AFA Attention: Linear‑Time Kernelization

Objective: Replace O(T²) attention with convolutional/state‑space AFA.

### Tasks
- [x] Kernel builder & cache
  - RG: AdaptiveFilterAttention, pairwise_precision, forward
  - Implement build_time_kernels(T) → returns lag kernels for e^{AΔtτ}; cache by (T, α, ω).
- [ ] Depthwise convolution path
  - Convolve K or V along time per head with the kernel (FFT or causal 1D convolution).
  - Replace explicit T×T weight matrix with kernelized propagation + local normalization.
- [x] Robust weighting
  - Optionally compute residual‑based scalars per lag (precomputed pairwise_precision(τ)), multiply into kernel before convolution.
- [ ] Fallback & parity
  - If T <= T_small or CFG.debug_exact, fall back to exact dot‑product path.
  - Unit test: with α=0, σ=0 → outputs match dot‑product attention (±1e‑4).

### Acceptance
- Peak memory sub‑quadratic for large T (e.g., T=4096).
- Speedup vs. baseline attention on long sequences.

### Rollback
- Flag‑guarded; keep current attention as fallback.

## Milestone 6 — FF Energy Alignment: Adaptive Thresholds & Signals

Objective: Stabilize and make energy signals more informative.

### Tasks
- Adaptive verifier threshold
  - Track running means E_pos_mean, E_neg_mean; set verifier τ midway or use region τ ensemble.
  - Make τ learnable with EMA target.
- Surprise scaling schedule
  - Add CFG.surprise_lambda_schedule: start at 0; linear warmup over N steps to target λ.
- Aux features
  - Feed mean_region_surprise, attn_entropy_mean (or per‑token stats) into verifier head (gated by feature flag).

### Acceptance
- Energy separation (E_neg − E_pos) increases over training.
- Goodness − surprise remains positive and stable (no domination by λ).

### Rollback
- Keep fixed τ and no surprise coupling if regressions appear.

## Milestone 7 — Deterministic Training Inner Loop

Objective: Simplify training path; reduce branching overhead.

### Tasks
- Bypass branching
  - RG: reasoning_loop, B_br, should_halt
  - If CFG.train_deterministic_inner_loop: force B_br=1, skip should_halt, run fixed K_inner.
- Disable planner/critic (optional)
  - Gate planner/critic with same flag to reduce compute during LM training.

### Acceptance
- Faster training epoch time; no accuracy regression on baseline tasks.
- Identical results across repeated runs with same seed (determinism).

### Rollback
- Keep branching available for inference and advanced tasks.

## Milestone 8 — Unify/Trim Sparsity Mechanisms

Objective: Reduce redundancy between region gating and KWTA.

### Tasks
- KWTA schedule & soft mode
  - Add kwta_k_schedule: start with larger k (less sparse), anneal to target k.
  - kwta_soft_mode=True uses a soft-threshold approximation during warmup (e.g., top‑k mask replaced by sigmoid gate with temperature anneal).
- Ablate dual sparsity
  - Experiment flag CFG.disable_kwta_during_gating to rely on gating only; or reduce region gating aggressiveness when KWTA is strict.
- Vectorized KWTA
  - Implement row‑wise top‑k on [R,d] in one call instead of loop per region.

### Acceptance
- No gradient starvation; similar or improved convergence speed.
- Reduced complexity / better clarity in activation pathway.

### Rollback
- Revert to current KWTA and gating defaults.

## Milestone 9 — Profiling, Telemetry & Tests

Objective: Make performance visible; ensure robustness.

### Tasks
- Profiling
  - Add timers (torch.cuda.Event / time.perf_counter) around: routing, region update, attention.
  - Report % time per component every N steps when CFG.profile=True.
- Memory checks
  - Log torch.cuda.max_memory_allocated() deltas per block under CFG.profile=True.
- Unit tests
  - Add tests for: vectorized router correctness, linear‑time AFA parity, dt>1 fast‑forward, robust weighting downweights noise, RT update unit norm.

### Acceptance
- CI green; profiling report shows expected reductions post‑refactors.

### Rollback
- Telemetry gated; no effect when disabled.

## Milestone 10 — Decomplexity & Docs Cleanup

Objective: Remove or quarantine unused paths; align docs.

### Tasks
- Prune unused code
  - RG: pairwise_precision (unused), legacy anchors when AFA on, duplicate loops.
  - Delete or @deprecated mark and move to legacy/.
- Docs
  - Update README / design docs to reflect flags and recommended defaults.
  - Add a Feature Matrix table: what each flag does and when to enable.

### Acceptance
- Reduced LOC; no references to removed symbols remain.
- Docs describe current, not historical, behavior.

### Rollback
- Stash in legacy/ to restore if needed.

## Milestone 11 — A/B Harness & Benchmarks

Objective: Validate improvements quantitatively.

### Tasks
- Ablation grid script
  - scripts/ablation_grid.py: run combinations of flags across a small task; save metrics.jsonl (loss, E_pos/E_neg gap, runtime, memory).
- Compare
  - Add notebooks/ or simple Python script to plot before/after metrics.

### Acceptance
- Clear wins on at least one dimension (speed, memory, stability, energy separation) without quality loss.

### Rollback
- Use harness to identify and disable harmful combos.

