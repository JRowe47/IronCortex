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

## Model Architecture

`CortexReasoner` embeds tokens with a sparse SDR encoder and mixes local context with an Iron RoPE mixer to form a "sensor" vector. A population of RWKVRegionCell modules is arranged on a hex-like graph. A Gate scores each region by content magnitude, usefulness EMA, focus proximity, homeostatic firing rate, critic value bias, and token uncertainty, then activates a sparse subset. A Router sends messages between previously active neighbors using learned edge transforms with Fourier relative positional bias. Active regions update their fast weights and emit new states; inactive ones fast-forward their decays while updating a predictive trace. Region states can write to a small shared Workspace. Heads operating on the motor state and workspace include:

- `PlannerHead` for proposing subgoals.
- `CriticHead` to estimate the value of allocating more compute.
- `TokenHead_MFS` for next-token prediction via multi-facet softmax.
- `RTDHead` for replaced-token detection.
- `EnergyVerifierHead` for consistency checking.

## Forward Pass and Learning Mechanics

1. **Sensor construction** – tokens are embedded and mixed locally; the focus map chooses which positions feed the sensor vector.
2. **Gate selection** – the Gate ranks regions using gain EMA, homeostasis, uncertainty, focus overlap, and critic bias, forcing sensor and motor regions on and selecting additional experts.
3. **Routing** – the Router aggregates messages from previously active neighbors with learned edge transforms modulated by a Fourier relative positional bias.
4. **Region updates** – active RWKVRegionCell modules combine sensor input and routed messages, apply Iron time rotation, update fast weights, and sparsify outputs; inactive regions update predictive traces and skip state advances.
5. **Workspace and heads** – active region states write to the Workspace; Planner proposes subgoals, the Critic estimates expected Δgoodness, the Token head predicts tokens, RTD classifies replacements, and the Energy verifier scores motor states.
6. **Branching loop** – the reasoning loop keeps multiple branches of region states, scores them using Δgoodness, verifier energy, and context log-prob, retains top candidates, and halts when uncertainty and scores stabilize.
7. **Training losses** – `train_step` mixes forward-forward per-region goodness thresholds, replaced-token detection, denoising cross-entropy, critic regression to realized Δgoodness, verifier energy FF loss, and Hebbian router updates alongside optimizer steps and gate homeostasis.

## Documentation

- [Numenta HTM integration ideas](docs/numenta_htm_integration.md)
- [Project to-do list](TODO.md)

## Getting Started

1. Install dependencies and run the smoke test:

   ```bash
   pytest
   ```

2. Train on Tiny Shakespeare:

   ```bash
   python train_tiny_shakespeare.py --seed 1234  # optional
   ```

   The `--seed` flag controls reproducibility. If omitted, a random seed is chosen each run.

The repository is under active development. See [TODO.md](TODO.md) for planned improvements such as hex axial wiring, new token heads, and stronger verifier targets.
