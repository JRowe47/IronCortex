# Iron Cortex

Iron Cortex is a research project exploring a reasoning-native neural architecture. Sparse RWKV regions are arranged on a hex-like graph and a gate allocates compute based on predicted payoff and uncertainty. An inner micro-loop (plan → route → update → verify) runs every token. Training uses a forward-forward routine with reconstruction, replaced-token detection, denoising auxiliaries, and an energy-based verifier. Generation is mask-predict with energy refinement and optional diffusion-style loops.

## Recent Changes

- Added Numenta HTM integration notes under `docs/numenta_htm_integration.md`.
- Energy-based verifier head and forward-forward energy loss utilities.
- Diffusion-style generation and gradient-based token refinement helpers.
- Dataset loaders for Tiny Shakespeare and text diffusion samples.

## Architecture

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

   The `--seed` flag controls reproducibility. If omitted, a random seed is
   chosen each run.

The repository is under active development. See [TODO.md](TODO.md) for planned improvements such as hex axial wiring, new token heads, and stronger verifier targets.
