# v13 — Scenario-Mediated Reasoning (SMR) Architecture Proposal

Written 2026-04-22 as the phase2 scaffold approach hits its ceiling. This document lays out the theoretical framework and two MVP implementations.

## Motivation — why 1-pass scaffold caps out

Our phase2 trajectory (see `phase2_final_state_report.md`) shows that 1-pass prompt scaffolding hits a ceiling around 54% vs baseline on 3-flash. More content in the prompt → more tokens the LLM acknowledges but doesn't apply. Meta-knowledge (Russell, sunk cost, confirmation bias) is especially prone to being **read-but-not-used** in 1-pass.

### User's key insight (the cognitive mechanism)

Humans don't apply abstract meta-principles directly. We:

1. **Parse the principle** — take "已有的后必再有" (Ecclesiastes) abstractly
2. **Instantiate to the current problem structure** — generate 2-3 specific hypothetical scenarios of how the principle might manifest in this situation
3. **Project the actual unfolding onto scenarios** — pattern-match our real-time observation to those pre-rehearsed cases
4. **Interpolate continuously** — our actual state usually sits BETWEEN scenarios; dogmatic match to one is a failure mode ("书上没教就傻眼")
5. **Branch-evaluate in real time** — as more signal comes in, re-weight which scenario is materializing

1-pass LLM execution short-circuits this. The LLM sees the principle, barrels through to the answer, and either over-applies it (dogma) or ignores it (decorative prologue).

### Why this matters now

With 3-flash's latent world model being strong enough that bare-prompt baseline_long beats our scaffold by 20pp, we're not getting value from **content injection**. We need to get value from **inference structure** — forcing the LLM to do the parse/instantiate/project loop explicitly.

---

## Core architecture — Scenario-Mediated Reasoning

### Formal structure

```
Input:  problem P, meta-principles M (priors + wisdom)

Stage 1 — Scenario Construction (parse layer):
  Generate 3 hypothetical scenarios S1, S2, S3 such that:
    - Each S_i is P's problem structure instantiated through one meta-principle
    - S_1, S_2, S_3 span DIFFERENT possible trajectories (not 3 near-copies)
    - Each is specific enough to pattern-match against, not so specific it misses

Stage 2 — Signal Projection (branch evaluation):
  For each S_i, assess w_i ∈ [0, 1] from P's explicit signals
    (w = how much of S_i's trajectory is actually unfolding here)

Stage 3 — Continuous Synthesis (answer):
  Produce answer conditioned on weighted scenarios.
  KEY: do NOT rigidly pick max(w_i) as "the scenario". Reason in the convex
  combination of scenarios, continuous-valued rather than discrete argmax.
```

### Distinction from existing methods

| Method | What it is | Difference from SMR |
|---|---|---|
| Chain-of-Thought | Linear reasoning trace | No explicit branching or scenario enumeration |
| Self-Refine | draft → critique → revise | Refines existing answer; doesn't pre-generate scenarios |
| Tree of Thoughts | Tree search over reasoning steps | Discrete path exploration; SMR is continuous between branches |
| ReAct | action → observation → reason loop | Agent-loop; SMR is reasoning-only |
| Case-Based Reasoning | Retrieve pre-stored cases | Static case library; SMR dynamically generates cases from P's structure |
| **SMR (this proposal)** | Generate cases from P × meta-principle, continuous-project | Synthesizes case generation + interpolation |

### Relation to Kahneman System 2

System 2 is characterized by:
- Effortful mental simulation
- Explicit consideration of alternatives
- Checking against heuristics before committing

SMR operationalizes this for LLMs — we explicitly instruct the model to do what System 2 does, rather than hoping the model does it implicitly during 1-pass generation.

---

## Two MVP implementations (already coded)

### MVP 1 — v13-reflect (simpler, 2-turn)

File: `phase one/scripts/validation/phase2_v13_reflect_framework.py`

```
Turn 1 (draft): 
  priors + triggers + wisdom + problem → draft answer (same as v11)

Turn 2 (self-audit + revise):
  "Your draft: D.
   For each prior/wisdom W_i, classify:
     A. W_i actually shaped D (some analysis directly traces to it), OR
     B. W_i was only cited or not used at all
   Find 1-2 type-B items that SHOULD have been applied.
   Produce revised answer integrating those 1-2 items."
```

**Purpose**: baseline for SMR claim. If explicit reflection alone (without scenario construction) helps, the mechanism might just be "forced second pass". If SMR (v13-scenario) beats v13-reflect, the scenario mechanism is the active ingredient.

Math/science bypass to v12c hygiene (no reflection needed for pure computation).

### MVP 2 — v13-scenario (full SMR, 2-turn compressed)

File: `phase one/scripts/validation/phase2_v13_scenario_framework.py`

```
Turn 1 (scenario generation + self-weighted):
  "Given P and meta-principles, generate 3 hypothetical scenarios S1/S2/S3
   that instantiate the principles on P's structure. For each, give
   signal_hit ∈ [0, 10] based on how much P's details hint at it."
  → JSON output with scenarios + weights

Turn 2 (synthesize):
  "Scenarios (not fixed answers, just reference paths):
     S1 [weight=7/10]: ...
     S2 [weight=3/10]: ...
     S3 [weight=5/10]: ...
   Use weights to guide focus, but do NOT dogmatically match one scenario.
   If P partially sits between S1 and S2, your answer should reflect the
   intermediate state. Continuous interpolation, not discrete match."
```

Math/science bypass to v12c hygiene (weak world-model for proof "scenarios").

Cost: 2x gen per problem (~17-20 min per 100 problems on 3-flash).

---

## Full architecture (v14, not yet implemented)

If v13-scenario shows signal, the natural extension is to separate state modeling from scenario generation:

```
Turn 1 (state snapshot):
  "Extract from P: (a) known state variables, (b) uncertain variables, (c) constraints"

Turn 2 (scenario branching):
  "State: {T1}. Meta-principles: M.
   Generate 3 futures based on different values for uncertain variables."

Turn 3 (signal evaluation):
  "State: {T1}. Scenarios: {T2}.
   For each scenario, score 0-10 how much P's explicit signals support it being the actual path."

Turn 4 (answer):
  "Weighted synthesis, continuous interpolation, final recommendation."
```

Cost: 4x gen (~60-80 min per 100 problems). Justify only if v13-scenario shows +5pp over baseline_long.

This is close to **agent-style explicit planning** applied to reasoning rather than action (cf. ReAct, Voyager subgoal planning).

---

## Expected domain performance

Based on our phase2 domain variance analysis and the world-model argument:

| Domain | v13 prediction | Reasoning |
|---|---|---|
| business, daily_life | **+8-10pp** vs baseline_long | LLM has rich scenario generators; meta-principles gain teeth via branching |
| engineering | **+5-8pp** | Same but more technical constraint, less open-ended |
| software_engineering | **+3-5pp** | LLM has strong native sw_eng knowledge; SMR may be redundant |
| science | **neutral to +3pp** | Partially covered by v12c hygiene |
| mathematics | **-2 to 0pp** | No useful scenarios for abstract proofs; keep v12c hygiene route |

Expected total: 58-62% vs baseline_long (vs current v12c 49%).

---

## Known risks

1. **Scenario quality is LLM-bound**: if 3-flash generates plausible-sounding but off-target scenarios (hallucinated rehearsals), Turn 2 inherits the error. Mitigation: human-audit 30 problems' Turn 1 output to calibrate.

2. **Weight self-estimation is noisy**: LLM giving itself signal_hit scores may anchor rather than evaluate. Mitigation: future v14 splits state-modeling from scenario generation into separate turns.

3. **Continuous interpolation is hard for LLM**: "don't dogmatically match one scenario" is a soft instruction LLM may ignore. Mitigation: include example of continuous reasoning in prompt.

4. **Cost**: 2x inference time. Only justified if it beats single-pass by ≥+5pp at same generator.

---

## Test plan

When v13-reflect and v13-scenario gens finish (both running in background):

1. **Judge v13-scenario vs baseline_long** — key test (does SMR beat bare + budget?)
2. **Judge v13-scenario vs v12c** — does SMR improve over math/sci-routed scaffold?
3. **Judge v13-reflect vs baseline_long** — does generic reflection help, or is scenario-specific the mechanism?
4. **Judge v13-scenario vs v13-reflect** — isolates the scenario-mechanism vs generic reflection

Decision rules:
- v13-scenario > baseline_long by ≥+5pp AND > v13-reflect by ≥+3pp → **SMR works, proceed to v14**
- v13-scenario ~ v13-reflect, both > baseline_long → reflection generic matters; drop scenarios
- v13-scenario ≤ baseline_long → SMR MVP fails; reconsider architecture (see `world_model_thinking_layer.md`)
