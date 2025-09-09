# Adaptive Filter Attention (AFA) Integration Plan

This document outlines the milestones and tasks for integrating Adaptive Filter Attention based on the paper "Attention as an Adaptive Filter" (2509.04154v1.pdf). Consult the PDF for theoretical background and mathematical derivations when implementing any portion of this plan.

---

## Milestone 0 — Scaffolding & Feature Flags

* [x] **Add feature flags** to your config/system:

  * `enable_adaptive_filter_dynamics` (default: false)
  * `enable_precision_routed_messages` (default: false)
  * `enable_radial_tangential_updates` (default: false)
  * `enable_afa_attention` (default: false)
  * `enable_ff_energy_alignment` (default: false)
* [x] **Centralize flags**: expose these in the model constructor(s) and CLI.
* [x] **Guard code paths** with flags to allow safe incremental rollouts.

---

## Milestone 1 — Adaptive Filter Dynamics in RWKV‑like Regions

**Goal:** Replace/augment single‑λ decay with per‑dim, learnable SDE‑like dynamics and uncertainty tracking.

### 1.1 State & Parameters

* [x] **Create precision-aware state buffers** in your RWKV region cell (e.g., `iron_cortex/regions/rwkv_region.py`):

  * `state_var: Tensor` (or `state_prec`) same shape as state.
  * `process_noise: Parameter` (per-dim, initialized small).
  * `obs_noise: Parameter` (per-dim or scalar).
* [x] **Extend decay to vector form**:

  * `decay_vec: Parameter` (negative real parts; clamp to ≤ 0).
  * (Optional) `freq_vec: Parameter` for paired dims to enable rotations (imag parts).

### 1.2 Propagation Step (Prior)

* [x] **Propagate mean & variance** each micro‑step:

  * Prior mean via `exp(A Δt)`; begin with elementwise `decay = exp(decay_vec * dt)`.
  * Prior variance via `state_var = state_var * decay.pow(2) + process_noise`.

### 1.3 Update Step (Robust/Kalman‑like)

* [x] **Compute observation residual** after forming the new contribution (e.g., `w * v` or message injection):

  * `pred = state_num / (state_den + eps)` (your existing fast-weight average).
  * `resid = (w * v) - pred`.
* [x] **Kalman gain** per dim:

  * `obs_var = f(k) * obs_noise` (e.g., `obs_var = torch.exp(-k)*obs_noise` to encode “confidence” in strong keys).
  * `gain = prior_var / (prior_var + obs_var)`.
* [x] **State update**:

  * `new_state = pred + gain * resid`.
  * Write back `state_num`, `state_var = (1 - gain) * prior_var`.
* [x] **Gate with feature flag** to fall back to original `state_num = state_num*lam + w*v`.

### 1.4 Optional: Complex/Rotational Dynamics

* [ ] **Pair channels** and apply rotation by `freq_vec * dt`:

  * Rotate both `state_num` and `state_den` pairs (2×2 rotation).

### Tests/Acceptance

* [x] Unit tests: forward parity vs. original (flag off).
* [x] Numerical sanity: `state_var` non-negative, decreasing with small `gain`.
* [x] Gradients stable on a random mini‑run.

---

## Milestone 2 — Precision‑Weighted, Robust Routing

**Goal:** Replace unweighted neighbor sum with content‑aware keys/queries and residual/precision reweighting.

### 2.1 Content Scoring for Neighbors

* [x] In `iron_cortex/routing/router.py` (or equivalent):

  * Add `query_lin: ModuleDict` per target region (or shared).
  * Add `key_lin: ModuleDict` per edge `(s→r)` (or shared typed by edge class).
* [x] At routing time:

  * `q_r = query_lin[r](H_prev[r])` (or learned static).
  * `k_sr = key_lin[s,r](H_prev[s])`.
  * `score[s] = (q_r · k_sr) / sqrt(d)`.

### 2.2 Robust Reweighting via Residuals

* [x] Form neighbor message `msg_sr = W_edge[s→r](H[s]) + fourier_bias`.
* [x] Compute residual to r’s **prior**: `resid_sr = msg_sr - H_prev[r]`.
* [x] Maintain/learn **edge precision** `P_edge[s,r]` (diag or scalar).
* [x] **Robust weight**:

  * `mah = (resid_sr.pow(2) * P_edge[s,r]).sum()`
  * `w_sr = exp(-0.5 * mah)`
* [x] **Aggregate**:

  * `M[r] = sum_s (w_sr * msg_sr)`
  * Normalize by `Z = sum_s w_sr` if `Z>0`.
* [x] Log/emit weights for interpretability.

### Tests/Acceptance

* [x] Routing produces identical results to old method with flag off.
* [x] With flag on: influence decreases for inconsistent neighbors (synthetic test).
* [ ] Stress tests: graph with many neighbors, ensure no OOM.

---

## Milestone 3 — Radial–Tangential State Updates

**Goal:** Split vector updates into **magnitude** (radial) and **direction** (tangential) for stability & interpretability.

### 3.1 Directional Processing

* [x] After computing region output vector `y`:

  * `norm = y.norm(2, dim=-1, keepdim=True) + eps`
  * `dir = y / norm`
* [x] Apply output transform to **direction only**:

  * `h_dir = o_lin(dir)`
  * Compose with magnitude:

    * Option A: `h = x + norm * h_dir`
    * [x] Option B: EMA magnitude `radius = ρ*radius + (1-ρ)*norm` and `h = x + radius * h_dir`.

### 3.2 Precision on Magnitude (Optional)

* [x] Track `radius_var` (1‑D Kalman on norm) and update radial component with small gain.

### Tests/Acceptance

* [x] Unit test: norm of `dir` \~ 1, finite grads.
* [x] Radial EMA reduces spikes on synthetic bursts.

---

## Milestone 4 — AFA‑Style Attention Layer (Sequence)

**Goal:** Provide a drop‑in **Adaptive Filter Attention** alternative to local self‑attention / anchor logic.

### 4.1 New Module

* [ ] Create `iron_cortex/attention/adaptive_filter_attention.py`:

  * Class `AdaptiveFilterAttention(heads, d_model, dt, ...)`.
  * Params: `alpha (decay)`, `sigma_proc`, `eta_obs`, optional `omega` per head.
  * Methods:

    * `build_time_kernels(T)` → precompute `e^{AΔtτ}` for τ (Toeplitz or FFT conv).
    * `pairwise_precision(|i-j|)` → closed‑form scalar precision per lag.
    * `forward(q, k, v, mask)`:

      1. Propagate `k`/`v` to query times using kernels (convolutional application).
      2. Compute residual‑based robust weights (optional).
      3. Row‑normalize → attention.
      4. Aggregate values.

### 4.2 Integration

* [ ] Wire flag: replace `IronRoPESelfAttention` (or equivalent) with `AdaptiveFilterAttention` when enabled.
* [ ] Remove/reduce anchor tokens logic behind the same flag.

### Tests/Acceptance

* [ ] Parity test: with trivial dynamics (α=0, σ→0), behaves like standard attention on short sequences.
* [ ] Speed/memory checks for long T (linear-ish memory).

---

## Milestone 5 — Forward‑Forward & Energy Alignment

**Goal:** Inject probabilistic “energy/surprise” signals and precision cues into FF goodness and verifier.

### 5.1 Surprise/Energy per Region

* [ ] After update, compute `surprise = (resid.pow(2) * state_prec).sum()` per region.
* [ ] Maintain `surprise_ema = β * surprise_ema + (1-β) * surprise`.
* [ ] Expose `surprise_ema` for logging and FF integration.

### 5.2 Goodness Adjustment

* [ ] In FF goodness (positive phase), incorporate **low energy == high goodness**:

  * `region_goodness = base_goodness - λ * surprise_ema`
  * λ via config.

### 5.3 Verifier Features

* [ ] Aggregate **attention energy** per token (e.g., `-mean(log w_i)`).
* [ ] Feed as auxiliary input to `EnergyVerifierHead` (or regularizer term).

### 5.4 Adaptive τ (Threshold)

* [ ] In region FF state, modulate `τ` by mean precision:

  * `τ_target = g_pos_mean + κ * (mean(state_prec) - target_prec)`
  * `τ = (1-α)*τ + α*τ_target`.

### Tests/Acceptance

* [ ] FF training still converges on small task with flags off.
* [ ] With flags on: decreasing energy over epochs; τ tracks precision trends.

---

## Milestone 6 — Telemetry, Testing, Benchmarks

### 6.1 Telemetry

* [ ] Log per‑region: `state_var/prec`, `surprise_ema`, routing weights stats (mean, entropy).
* [ ] Log AFA: `alpha`, `sigma_proc`, `eta_obs`, kernel norms.

### 6.2 Unit Tests

* [ ] RNG‑seeded checks for:

  * No NaNs/Inf under stress.
  * Grad flow through all new branches.
  * Edge cases: empty neighbor set, masked tokens.

### 6.3 Benchmarks

* [ ] Micro‑benchmarks: before/after runtime & memory on synthetic long sequences.
* [ ] Quality proxy: noise‑rejection toy task (AFA on/off).

---

## Milestone 7 — Rollout & Risk Controls

* [ ] **Runtime flags** to toggle each feature independently at init and mid‑run (via registry or env var).
* [ ] **A/B harness**: run baseline vs. each feature combo on a held‑out suite.
* [ ] **Safe fallback** on divergence: auto‑disable feature if loss/energy explodes (thresholded watchdog).

---

## Implementation Hints (for the coding agent)

* **Numerical stability**

  * Clamp decays `decay_vec = -softplus(raw_decay)` so real parts ≤ 0.
  * Add small `eps` to denominators (`state_den`, `norm`) and to variances.
* **Complex/rotation pairs**

  * Enforce even number of dims if enabling `freq_vec`; otherwise keep rotations off by default.
* **Precision/variance choice**

  * Store **variance** internally (non‑negative), derive precision lazily as `1/(var+eps)` when needed.
* **Vectorization**

  * Prefer element‑wise ops and batched tensor math over Python loops in routing and AFA.
* **Compute budget**

  * Gate robust residual weighting (exp/mahalanobis) behind config if it becomes hot.

---

## Acceptance Criteria (overall)

* [ ] With all flags **off**, behavior and metrics match current baseline within tolerance.
* [ ] With **adaptive filter dynamics** enabled, synthetic noisy‑sequence tests show improved denoising (lower MSE) and stable variance traces.
* [ ] With **precision‑weighted routing** enabled, ablations show reduced error when some neighbors are adversarial/noisy.
* [ ] With **radial–tangential** enabled, fewer activation spikes and improved training stability (fewer gradient explosions).
* [ ] With **AFA attention** enabled on long sequences, memory growth is sub‑quadratic and accuracy does not regress.
* [ ] With **FF energy alignment** enabled, training curves show decreasing surprise/energy and improved goodness separation.

---

### Nice‑to‑Have (later)

* [ ] Shared low‑rank parameterization for decays/frequencies across regions to cut params.
* [ ] Learned mapping from token time to Δt (non‑uniform time steps).
* [ ] Visualization tool: per‑step routing graphs colored by robust weights.

