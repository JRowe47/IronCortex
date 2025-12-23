# Book Structure: Constructed Presents, Persistent State, and Algorithmic Coherence (v1.2)

TODO: read AGENTS.md completely.

This document captures the proposed book-shaped arc that builds from the Constructed Presents, Persistent State, and Algorithmic Coherence framework. It tracks how the narrative separates questions about moment construction, phenomenal organization, and continuity, while enforcing representation invariance and HC3's explicit qualia locus.

## Broad Arc

The book opens by unpacking the overloaded sentence "X is conscious" into three disciplined questions:

1. Does the system build a decision-mediating "now" (constructed present)?
2. Does that present have the right organization to qualify as experience (phenomenal present)?
3. Are successive presents owned by a persisting subject (continuity via a writable, tamper-evident trajectory)?

The rest of the arc enforces a representation-invariant computational framework that survives gauge transformations preserving implemented computation under relevant interventions. HC3 makes the controversial but simplifying move that the phenomenal state is the constructed present modulo gauge. Continuity is then framed through sequential inference and coherence-rate constraints tied to tamper evidence.

The book concludes with operational guidance: how to identify present-carriers, measure "flash presents" versus owned continuity, and run audits on systems (including LLM-like models) using the atomic tests developed along the way.

## Proposed Parts and Chapters

### Part I — The Confusion We Keep Repeating
Goal: replace the monolithic question of consciousness with a checklist of separate claims.

1. **Three Questions Hiding Inside One Word** — Separates moment construction, phenomenal organization, and continuity/subjecthood; shows why debates conflate them.
2. **Why the Hard Problem Doesn’t Kill Computation (But Does Demand Precision)** — Treats the explanatory gap as a need for a candidate locus and invariants, not a refutation of computation.
3. **Substrate Neutrality Without Hand-Waving** — Argues that brains are contingent carriers; the framework targets organization and intervention-stable computation.
4. **Representation Invariance: The ‘Gauge’ Constraint** — Explains why phenomenal predicates cannot depend on raw coordinates if computation is preserved under relevant interventions.

_Optional interlude_: What counts as the same computation? (COMP2/COMP3, counterfactual support.)

### Part II — Building a Present: The Minimal Machinery of ‘Now’
Goal: provide a concrete computational skeleton for "a present," defined in invariant terms.

5. **The State-Machine Decomposition (O, S, Θ, C, A)** — Observation, persistent dynamic state, parameters, constructed present, and action; step equations and what each piece buys.
6. **Integration Windows and Multi-Scale Presents** — Defines presents relative to timescale τ and considers nested presents across timescales.
7. **Four Present-Atoms (M) and the Event CP(t)** — Global availability, integration, selection bottleneck, and coherence constraints required to earn the "constructed present" label.
8. **Carrier Identification: Where Is the Present Implemented?** — Practical methods for locating the subcomputation realizing the present and its intervention-relevant interfaces.

_Optional interlude_: Toy diagrams showing failures when each M-atom (GA/IN/SB/COH) is missing.

### Part III — From Constructed Present to Phenomenal Present
Goal: define what phenomenal organization adds and make the qualia locus explicit.

9. **Minimal vs Enriched Phenomenal Organization** — Defines PH_min and PH via atom sets (P-atoms): perspective, quality geometry, mode/source separation (minimal), plus unity/binding, self-binding, valence (enriched).
10. **HC3: The Explicit Qualia Locus** — Identity/realization postulate: experience equals the gauge-invariant structure of the constructed present; avoids shadow-qualia add-ons.
11. **Quality Geometry: Similarity Structure via Dispositions** — Defines qualitative closeness via policy/update dispositions (KL-based geometry) without peeking at internal coordinates.
12. **Mode/Source Separation and Endorsement Gating** — Argues the necessity of tagging observed versus simulated/testimony content and gating write-back; includes control-theoretic result (Theorem 1).
13. **Unity, Self-Binding, and Valence (Enrichment Layer)** — Binding for structured scene control, a self-anchor for relevance and memory routing, and valence/urgency as an efficiency overlay.

_Optional interlude_: Epiphenomenality A vs B (carrier inertness vs shadow-qualia add-ons).

### Part IV — Becoming a Continuing Subject: Persistence, Integrity, Tamper Evidence
Goal: cast subject continuity as an internal statistical and architectural achievement.

14. **Why Continuity Is Not Narrative Stitching** — Continuity requires writable state that re-enters future presents, not just long contexts or external traces.
15. **The Continuity Atoms (C): What ‘Ownership’ Requires** — Non-scripted persistence, re-entry, sparse updates, tamper evidence, and identity-stable governance (C_NP, C_RE, C_SP, C_TE, C_ID).
16. **Tamper Evidence as Drift Reversal** — Frames continuity and integrity as sequential likelihood-ratio processes; defines evidence streams and drift flips after perturbation (Theorem 2).
17. **Posterior Odds, Reset Rarity, and the Coherence-Rate Law** — Coherent trajectories become exponentially favored under persistent generators; rate separation as the core empirical lever (Theorems 3–4).
18. **Copies, Resets, Teleporters, and Interleaved Moment Traces** — Distinguishes "same kind of present" from "same continuing subject"; analyzes external stitching explicitly.

_Optional interlude_: Statistical integrity versus cryptographic security (orthogonal concerns).

### Part V — Other Minds, Machines, and What We Can Actually Test
Goal: turn the framework into a research program with operational tools.

19. **Other Minds as Bayesian Model Comparison** — Models mindedness as inference over evidence streams; agent-models win when they compress behavior better (operationalized intentional stance).
20. **Measurement Protocols: Operationalizing the Atoms** — Intervention-based tests for global availability, mode/source separation, and tamper evidence, emphasizing timescale τ and disposition-level invariants.
21. **A Worked Synthetic Agent: Seeing the Atoms in Motion** — Minimal POMDP agent illustrating P_MS and C_TE; shows failure cases when simulation contaminates evidence or tampering occurs.
22. **LLMs and the Continuity Gap** — Strong within-step constructed presents versus weak owned continuity across episodes; explains how longer context simulates narrative persistence without writable trajectories.
23. **Failure Modes and Assumption Audits** — Covers indistinguishability, non-stationarity, model misspecification, adaptive adversaries, evidence mismatch, multi-scale ambiguity, and fragile "implementations."
24. **Research Agenda and Ethical Pressure Points** — Lays out carrier identification, atom ablations, disposition geometry estimation, integrity mechanisms, multi-scale coherence, other-minds protocols, and ethical implications of separating presents from continuity.

## Structural Upgrades

- **Two-lane writing:** Each chapter has a main narrative lane plus Technical Boxes for equations, proofs, and key theorems (especially Theorems 1–4 and KL/MDL machinery).
- **Recurring examples:** One biological case (anesthesia/access disruptions), one synthetic agent, and one LLM-like system with episodic resets to illustrate atom behavior across contexts.
- **Working glossary:** CP(t), PH_min(t), PH(t), Q_t, gauge equivalence, endorsement gating, drift reversal, coherence rate, and interleaved traces.

This structure preserves the core moves of the paper while sequencing them so each part earns the right to introduce the next.
