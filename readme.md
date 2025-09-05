A fully wired reasoning-native cortex with sparse RWKV regions on a hex-like graph, a gate that allocates compute by predicted payoff (critic) and uncertainty, and an inner micro-loop (plan → route → update → verify) that runs every token.

A Forward-Forward training routine that pushes per-region goodness on clean (positive) inputs above a learned threshold and below it for corrupted (negative) inputs, plus RTD (hard negatives) and denoising (mask‑predict) auxiliaries.

Iron RoPE integrated in three spots:

A LocalTokenMixer (bidirectional attention with RoPE + relative Fourier bias) that only updates masked/uncertain locations and reads from localized pseudo tokens (anchors).

A small time rotation on RWKV v‑channels so the fast‑weight EMA can represent inner‑step (diffusion) “time” without breaking the w=exp(k) positivity.

A relative Fourier bias on router edges so message passing respects the hex geometry.

A Workspace scratchpad and minimal Planner → Critic → Verifier heads to steer compute and gate bias by value.

A mask‑predict generator that repeatedly focuses low‑confidence positions, runs the inner loop, and fills tokens.

How to extend

Replace the grid neighbors/coords with true hex axial layouts.

Swap the simple TokenHead_MFS for your preferred multi‑facet or MoE head.

Strengthen Verifier targets using structural constraints specific to your domain.

If you want strict FF (no cross-step backprop), wrap step outputs with detach() along cross‑time paths.
