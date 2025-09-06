TODO: read AGENTS.md completely

# AGENTS.md

This file provides guidance to OpenAI Codex when working with code in this repository.

**⚠️ MAINTENANCE REQUIRED: This is a living document. When you update any section, you MUST update the line numbers in the index below.**

**🚨 MANDATORY: Every compaction or summarization must have a TODO inserted to read AGENTS.md completely.**

## Index
- IMPORTANT: READ THIS ENTIRE FILE (lines 26-52)
- CRITICAL RULES FOR SYSTEMATIC WORK (lines 53-129)
- CORE ENGINEERING PRINCIPLES (lines 130-188)
- APPLYING THESE PRINCIPLES (lines 189-197)
- UPDATING THIS DOCUMENT (lines 198-214)
- PROJECT-SPECIFIC GUIDANCE (lines 216-293)
  - Project Overview (lines 218-224)
  - Project Status (lines 225-229)
  - Architecture (lines 230-249)
  - Prompts and Tools (lines 250-254)
  - Common Tasks (lines 255-274)
  - Testing Commands (lines 275-284)
    - Next Development Steps (lines 286-293)

## IMPORTANT: READ THIS ENTIRE FILE

**DO NOT just read the first 50 lines.** This document contains:
- Critical rules for systematic work (lines 53-129)
- Core engineering principles for strategic decisions (lines 130-188)
- Applying these principles (lines 189-197)
- Updating this document (lines 198-214)
- Project-specific guidance (lines 216-293)

**CRITICAL: When updating AGENTS.md, you MUST update these line numbers!**
- After ANY edit, recalculate the line ranges for each section
- Use your editor's line numbers or search for section headers
- Accurate line numbers are essential for document navigation
- Outdated line numbers defeat the purpose of this index

Each section contains essential information. Skipping sections leads to:
- Missing tests or leaving debugging artifacts
- Not understanding the project's architecture
- Making poor strategic decisions about implementation approaches
- Violating established patterns and principles

**When starting work**: Read the ENTIRE file first
**When working on specific areas**: Re-read relevant sections
**When stuck**: Check if this document already has guidance
**When making design decisions**: Consult the engineering principles section
**When editing this document**: UPDATE THE LINE NUMBERS IN THE INDEX ABOVE

## CRITICAL RULES FOR SYSTEMATIC WORK

When performing systematic transformations (like grammar conversions, refactoring, or any complex multi-part task):

### 0. LISTEN TO DEFINITIVE STATEMENTS
- When the user makes an absolute statement like "X is NEVER Y" or "X should ALWAYS be Z", treat it as an axiom
- Do NOT revisit or contradict these statements later
- Write them down and refer back to them
- If you find yourself doing something that contradicts a definitive statement, STOP

### 0.5. ANSWER THE QUESTION ASKED
- When the user asks "Why is X doing Y?", investigate THAT SPECIFIC ISSUE
- Don't explore alternative approaches until you've answered the direct question
- If they point out something suspicious (like "Why are we allowing X?"), that's usually the key insight
- Start with the simplest interpretation of their question

### 1. UNDERSTAND BEFORE TRANSFORMING
- NEVER do mechanical search-and-replace without understanding what the code actually does
- For each rule/function/component: trace through what it ACTUALLY matches/computes
- Write out concrete examples of inputs and outputs

### 2. MODEL THE FULL PROBLEM SPACE
- When dealing with transformations, explicitly write out:
  - What the input looks like (with concrete examples)
  - What the output looks like (with concrete examples)
  - How EACH piece maps from input to output
- Don't hand-wave with "it produces X" - show EXACTLY what X looks like

### 3. WORK INCREMENTALLY
- Pick ONE small piece
- Transform it completely and correctly
- Test it
- Only then move to the next piece
- RESIST the urge to do everything at once

### 4. PRESERVE KNOWLEDGE
- Existing code encodes hard-won knowledge about edge cases
- Every special case exists for a reason
- When transforming, preserve ALL the original logic, just in new form

### 5. ADMIT WHEN YOU DON'T UNDERSTAND
- If you find yourself pattern-matching or guessing, STOP
- Either figure out what the code actually does, or admit you need help
- Half-understanding leads to cascading failures

### 6. NO PREMATURE DECLARATIONS OF VICTORY
- Don't declare "Done!" until you've:
  - Tested the complete system
  - Verified all edge cases work
  - Checked that nothing was lost in transformation

### 7. TESTS OVER TEMPORARY FILES
**STOP! Before creating ANY file with these patterns:**
- `test_*.py` in random locations
- `debug_*` anywhere
- `tmp_*` or `temp_*` anywhere
- Any one-off `main()` or `if __name__ == "__main__":` snippets for testing

**INSTEAD, you MUST:**
1. Identify which module this tests
2. Find or create the appropriate test file under `tests/`
3. Write a proper test function using `pytest`
4. Use table-driven tests for multiple cases

**RED FLAGS that you're violating this rule:**
- Thinking "let me quickly test this..."
- Creating a file with ad-hoc `main` blocks
- Writing print statements for debugging
- Using `exit()` or `fatal()` in test code

**Remember:** If you need to debug something, that's a sign of missing test coverage. The temporary file you're about to create should be a permanent test case.

### 8. ACTIVELY CONSULT THIS DOCUMENT
- When starting complex work, re-read AGENTS.md
- When stuck or unsure, check if AGENTS.md has relevant guidance
- This document exists to prevent repeated mistakes - USE IT

## CORE ENGINEERING PRINCIPLES

These principles guide strategic decision-making and should inform all work on this project:

### PRINCIPLE 1: Determine the actual problem being solved for that has the most impact

**Make sure that the problem we are solving is the actual root problem, or at least addresses part of it.**

- When approached with "we need Solution X", dig deeper to uncover Problem A
- Often users/customers describe solutions, not problems
- Transform solution requests into problem statements, then derive requirements
- From requirements, evaluate multiple solutions (Y, Z) and their tradeoffs

**Key questions to ask:**
- How much calendar time do we have?
- How much are we willing to spend (time/resources) to solve the problem?
- If we spent more, would it give us an advantage in reducing calendar time?
- What do the individual solutions solve of the requirements?
- Which requirement is actually important?

This approach leads to better outcomes for everyone involved by ensuring we're solving the right problem.

### PRINCIPLE 2: Be cognizant of the cost of an approach or solution in terms of time or complexity

**When making tactical or strategic decisions that might incur technical debt or only solve part of the requirements, consider the full cost implications.**

Evaluate three scenarios for every decision:
1. **Cost of doing it now**
   - Does it incur up-front complexity?
   - Does it simplify the problem surface?
   - What is the immediate impact?

2. **Cost of deferring it**
   - Does it impact other architectural decisions downstream?
   - What happens to dependencies?
   - Will the cost increase over time?

3. **Cost of not doing it at all**
   - Does it need to be done?
   - Does it add unnecessary complexity?
   - How does it solve the requirements in the end?

Remember: Every feature or code pathway adds to project complexity and increases maintenance and cognitive burden.

### PRINCIPLE 3: Determine the 'good enough' point, and recognize that perfection is the enemy of good

**It is often better to make tactical decisions that solve 80% of the immediate problem, as long as it's the most important 80% that in turn solves 80% of the strategic problem.**

- Avoid the trap of eternally polishing the solution
- Many junior (and senior!) engineers fall into this perfectionism trap
- Be mindful of the entire project goal when working on components

**Before diving deep into any component, ask:**
- If I spend X time on this component, what percentage of the entire project timeline will it consume?
- Will the impact on the final result be commensurate with the time spent?
- What is the minimum viable solution that delivers the most value?

This principle builds upon and ties together Principles #1 and #2 - understand the real problem, evaluate the costs, and then determine what "good enough" looks like for this specific context.

## APPLYING THESE PRINCIPLES

When starting any new work:
1. First apply the CRITICAL RULES for systematic, correct implementation
2. Then step back and apply the ENGINEERING PRINCIPLES for strategic decision-making
3. Balance tactical correctness with strategic efficiency

The rules ensure you do things right; the principles ensure you do the right things.

## UPDATING THIS DOCUMENT

When modifying AGENTS.md:

1. **Make your content changes** - Add, modify, or remove sections as needed
2. **Update the line number index** at the top:
   - Find each section header in the document
   - Note its starting line number
   - Calculate the line range (approximate end line)
   - Update the corresponding entry in the index
3. **Verify the index** - Spot-check that line numbers actually lead to the correct sections
4. **Consider the impact** - If adding new guidance, consider where it fits:
   - Is it a tactical rule? Add to CRITICAL RULES
   - Is it a strategic principle? Add to ENGINEERING PRINCIPLES
   - Is it project-specific? Add it below this section with a clear header

Remember: An outdated index is worse than no index. Keep it current!

## PROJECT-SPECIFIC GUIDANCE

### Project Overview

Iron Cortex is a research project exploring a reasoning-native neural architecture.
It combines sparse RWKV regions arranged on a hex-like graph, a gate that allocates compute by predicted payoff and uncertainty,
and an inner micro-loop (plan → route → update → verify) that runs every token.
Training uses a forward-forward routine with reconstruction, RTD, and denoising auxiliaries.

### Project Status (June 2025)

**Current State**: Core model, training step, and generation loop implemented with a basic smoke test.
Areas like dataset loaders, larger model configurations, and evaluation tooling remain open for development.

### Architecture

```
ironcortex/
├── __init__.py        # Public API exports
├── config.py          # Model and training hyperparameters
├── model.py           # CortexReasoner module and reasoning loop
├── training.py        # Loss weights and train_step routine
├── generation.py      # Autoregressive generator with uncertainty masking
├── gate.py            # Gate and Router components
├── region.py          # RWKV region cell
├── heads.py           # Workspace and head definitions
├── iron_rope.py       # Local token mixer and positional encoding
├── corruptions.py     # Negative/denoising transformations
├── utils.py           # Helper functions
└── wiring.py          # Hex grid utilities
tests/
└── test_smoke.py      # Basic integration test
```

### Prompts and Tools

The project may later integrate prompt templates or external tools (e.g., for data fetching or analysis).
If such components are added, document their usage and locations here.

### Common Tasks

#### Training
- Define a `CortexConfig` and instantiate `CortexReasoner`.
- Prepare token batches and call `train_step(model, optimizer, tokens, weights, device)`.
- Update or extend `LossWeights` when adding new loss components.

#### Generation
- Use `generate(model, prompt, T_total, max_outer_iters, conf_threshold)` for sampling.
- Modify `generation.py` to experiment with new confidence strategies or routing heuristics.

#### Wiring and Geometry
- Use `hex_neighbors_grid()` and `hex_axial_coords_from_grid()` from `wiring.py` to build region graphs.
- For true hex axial layouts or alternative topologies, add new helpers here.

#### Adding New Modules
- Place core logic under `ironcortex/`.
- Expose public APIs in `__init__.py`.
- Add targeted tests under `tests/`.

### Testing Commands

```bash
# Run the test suite
pytest

# Optional formatting and linting
black .        # code formatting
ruff .         # static analysis
```

### Next Development Steps

1. Replace grid neighbors/coords with true hex axial layouts.
2. Explore alternative token heads or mixture-of-experts approaches.
3. Strengthen verifier targets with domain-specific constraints.
4. Investigate strict forward-forward training by detaching cross-time paths.
5. Expand tests to cover training stability, generation quality, and wiring utilities.
6. Begin exploring text generation using a diffusion paradigm and other non-standard model features.
