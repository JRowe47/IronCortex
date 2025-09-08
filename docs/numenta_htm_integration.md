TODO: read AGENTS.md completely

# Integrating Thousand Brains Theory and HTM Principles into Iron Cortex

Iron Cortex already employs several biologically inspired components. This document outlines additional opportunities for
bringing ideas from Numenta's Thousand Brains Theory and Hierarchical Temporal Memory (HTM) into the codebase.

## 1. Strengthen Sparse Representations
- **Spatial pooler style token encodings** – Implemented via a fixed sparse distributed representation (`SparseTokenEncoder`)
  replacing the dense embedding layer[model-embed].
- **Diverse region receptive fields** – Introduce sparsity masks on router edge weights so regions specialize to different
  feature subsets. At present every neighbor connection owns a full linear transform[router-edges].
- **Tune KWTA and homeostasis** – Experiment with smaller `k` in KWTA and lower `k_active` for the gate. The region cell
  applies KWTA after each update[region-kwta] and the gate updates homeostatic firing rates[gate-homeo].

## 2. Incorporate Temporal Memory for Sequence Prediction
- **Predictive activation** – Implemented via a per-region `predict` trace that accumulates router
  messages while inactive so regions respond faster when later selected, analogous to HTM distal
  dendrites.
- **Hebbian fast weights for transitions** – Maintain a light-weight association matrix of region-to-region transitions
  and bias the gate toward frequently observed sequences. Regions already store fast weights via `state_num` and
  `state_den` buffers[region-fastweights].
- **Bursting on novelty** – When no region strongly fits the current input, temporarily increase the number of active
  regions to encourage exploration, similar to HTM column bursting.

## 3. Enable Multi‑Region Consensus
- **Distributed token prediction** – Give multiple active regions lightweight prediction heads and combine their logits
  instead of relying solely on the motor region's output. Currently, the motor state alone feeds the language
  model head[model-lmhead].
- **Workspace for voting** – Use the global `Workspace` as a blackboard where active regions write summaries that a
  decoder can read to produce a consensus prediction.

## 4. Enhance Online Plasticity
- **Local Hebbian updates** – Augment gradient-based learning with small Hebbian nudges when regions succeed, echoing HTM's
  synapse permanence adjustments. The training loop already tracks per-region goodness differences and updates gate
  gains accordingly[training-ff].
- **Learning-rate modulation by surprise** – Use verifier or RTD losses as anomaly signals to momentarily boost learning
  rates when the model encounters unexpected inputs.

These steps aim to make Iron Cortex sparser, more predictive, and more adaptive while staying within the existing deep
learning framework.

<!-- References -->
[model-embed]: ../ironcortex/model.py#L49-L51
[router-edges]: ../ironcortex/gate.py#L110-L114
[region-kwta]: ../ironcortex/region.py#L84-L85
[gate-homeo]: ../ironcortex/gate.py#L89-L93
[region-fastweights]: ../ironcortex/region.py#L31-L32
[model-lmhead]: ../ironcortex/model.py#L148-L158
[training-ff]: ../ironcortex/training.py#L100-L121
