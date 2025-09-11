TODO: read AGENTS.md completely
# Iron Cortex

Iron Cortex is a research project exploring a reasoning-native neural architecture. Sparse RWKV regions are arranged on a hex-like graph and a gate allocates compute based on predicted payoff and uncertainty. An inner micro-loop (plan → route → update → verify) runs every token. Training uses a forward-forward routine with reconstruction, replaced-token detection, denoising auxiliaries, and an energy-based verifier. Generation is mask-predict with energy refinement and optional diffusion-style loops.

## Recent Changes

- Added Numenta HTM integration notes under `docs/numenta_htm_integration.md`.
- Energy-based verifier head and forward-forward energy loss utilities.
- Diffusion-style generation and gradient-based token refinement helpers.
- Dataset loaders for Tiny Shakespeare and text diffusion samples.

## Code Layout

```text
ironcortex/
├── __init__.py          # Public API exports
├── config.py            # Model and training hyperparameters
├── data.py              # Tiny Shakespeare downloader and diffusion dataset
├── model.py             # CortexReasoner module and reasoning loop
├── gate.py              # Gate (compute allocator) and Router
├── region.py            # RWKV region cell with fast-weight updates
├── iron_rope.py         # Local token mixer & positional encoding helpers
├── heads.py             # Workspace plus Planner/Critic/Verifier/Token heads
├── energy.py            # Energy-based verifier and gradient descent step
├── ff_energy.py         # Forward-Forward energy loss helpers
├── sdr.py               # Sparse token encoder
├── training.py          # Loss weights and train_step routine
├── generation.py        # Mask-predict generator with energy descent
├── diffusion.py         # Diffusion-style token generator
├── evaluation.py        # Evaluation helpers
├── visualization.py     # Plotting utilities
├── corruptions.py       # Negative/denoising transformations
├── thinking.py          # Gradient-based refinement helper
├── utils.py             # RMSNorm, KWTA, batching, and other utilities
└── wiring.py            # Hex grid neighborhood utilities
tests/
└── test_smoke.py        # Basic integration test
train_tiny_shakespeare.py  # Example training script
```

## Feature Flags

| Flag | Default | Purpose | When to enable |
| --- | --- | --- | --- |
| `enable_adaptive_filter_dynamics` | `False` | Kalman‑style state updates in regions | Exploring adaptive dynamics |
| `enable_precision_routed_messages` | `False` | Robust, precision‑weighted routing | Studying noisy or adversarial inputs |
| `enable_radial_tangential_updates` | `False` | Split magnitude and direction updates | Improving norm stability |
| `enable_afa_attention` | `False` | Adaptive Filter Attention replaces anchor tokens | Long sequences or experimenting with AFA |
| `enable_ff_energy_alignment` | `False` | Couples energy signals with Forward‑Forward goodness | When using energy verifier |
| `train_deterministic_inner_loop` | `False` | Fixed inner loop for reproducible training | Deterministic experiments |

All flags live in `CortexConfig` and default to the values above. Unless you're researching the corresponding feature, the defaults are recommended for stability.

## Model Architecture

IronCortex is a GAN‑MoE, energy‑based, diffusion‑augmented RWKV‑Transformer that learns online with Forward‑Forward over a six‑stage cortical routing stack. Its computation is built from Numenta‑inspired columnar nodes (minicolumns) that cluster into topographic regions; each node is an RWKV state cell with persistent fast‑weights and a predictive trace, and regions communicate over a cortical graph. A sparse‑distributed representation (SDR) is enforced by k‑Winners‑Take‑All and homeostatic gating: at every micro‑step a mixture‑of‑experts router selects a small top‑k set of regions to activate using neighbor‑suppressed competition, usefulness priors (gain EMA), uncertainty cues, and value bias from a learned critic. Inter‑region messages use IronRoPE geometric routing—a capsule‑style, routing‑by‑agreement mechanism that combines relative/rotational position coding with learned edge transforms—providing six “layers/hops” of iterative plan → route → update → verify → act → halt processing within each token’s inner loop (with optional branch/beam maintenance and early‑exit on confidence). A lightweight workspace/blackboard lets active regions write and read shared slots for latent, native reasoning, while motor/sensor heads interface with token prediction. Inputs use byte‑patch tokenization (byte‑level units with local patching) and a compact local attention/RWKV fusion for context ingestion. Training is multi‑objective: (1) Forward‑Forward goodness per region with adaptive thresholds τ separates positive vs corrupted sequences; (2) denoising / masked‑token reconstruction and replaced‑token detection (RTD) teach linguistic structure; (3) a value critic predicts expected goodness‑gain to guide compute allocation; (4) an energy‑based verifier learns low E_pos / high E_neg consistency for predictions and also scores branches; (5) a lightweight adversarial (GAN) discriminator/critic regularizes generative outputs; and (6) standard cross‑entropy / perplexity monitor end‑to‑end language modeling. Generation couples the token head with a diffusion refiner that iteratively denoises logits/latents under energy guidance, yielding proposals that are re‑scored by the verifier and the cortical planner. Core telemetry (ff, rtd, denoise, critic, verify, E_pos/E_neg, gain_mean, tau_mean, cross_entropy, perplexity) tracks region specialization, compute efficiency, and model certainty as the system self‑organizes a thousand‑brains–style ensemble: many columnar specialists forming consensus via recurrent, sparse, geometrically routed communication.

## Documentation

- [Numenta HTM integration ideas](docs/numenta_htm_integration.md)
- [Project to-do list](TODO.md)

## Installation

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Install dependencies and run the smoke test:

   ```bash
   pip install -r requirements.txt
   pytest
   ```

2. Train on Tiny Shakespeare:

   ```bash
   python train_tiny_shakespeare.py --seed 1234  # optional
   ```

   The `--seed` flag controls reproducibility. If omitted, a random seed is chosen each run.
   Add `--visualize` to enable realtime training plots, or `--hex-vis` to display
   region states on a hex grid.

The repository is under active development. See [TODO.md](TODO.md) for planned improvements such as hex axial wiring, new token heads, and stronger verifier targets.
