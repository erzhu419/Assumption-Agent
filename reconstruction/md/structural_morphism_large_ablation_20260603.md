# Structural Morphism Large Ablation - 2026-06-03

## Objective

Validate whether the Structural Morphism layer improves downstream answer quality on a larger live sample, instead of only passing offline structural tests.

This run tests a **gold-routed structural benchmark**:

- `coverage_tags` / reference strategy tags route each problem to a structural pattern.
- The model still answers the original problem normally.
- A `gpt-5.5` pairwise judge compares `STRUCTURAL` against `BASE` and `PLACEBO`.

Important limitation: this validates whether structural context is useful once routed correctly. It does **not** prove natural structural retrieval recall is already solved.

## Implementation Changes

- Added `assumption_os/structural_live_ablation.py`.
- Added `retrieval | coverage | hybrid` selection modes.
- Added strategy-tag to structural-pattern routing with optimal-strategy priority over acceptable-strategy priority.
- Added problem-facing structural transfer checklists for `S01`-`S27`.
- Added deterministic placebo pattern selection that excludes patterns implied by the current problem's reference tags.
- Added direct `requests` judge transport with explicit timeout to avoid OpenAI SDK long-tail hangs during large runs.
- Added relaxed judge JSON parser so malformed JSON/escaping cannot crash the full experiment.
- Updated model router defaults to current env-configurable models:
  - solver: `gpt-5.4-mini`
  - judge: `gpt-5.5`
  - Gemini aliases preserved but not used for this run.

## Runs

### Dry Run

Command:

```bash
python3 -m assumption_os.structural_live_ablation \
  --eval-id structural_live_hybrid_dryrun2_20260603 \
  --selection-mode hybrid \
  --max-cases 100 \
  --dry-run
```

Result:

- selected cases: `100/100`
- route source: `coverage_gold=100`
- planned solver cells: `300`
- planned judge pairs: `200`

### Initial Smoke

`structural_live_hybrid_smoke5_20260603`

- structural vs base: `1 win / 4 loss`, utility `0.20`
- structural vs placebo: `1 win / 4 loss`, utility `0.20`

Diagnosis:

- context was too abstract
- route selection incorrectly let high-confidence acceptable tags override optimal tags
- structural hints sometimes caused over-large trials or generic A/B framing

Fix:

- optimal tags now outrank acceptable tags
- context includes concrete strategy-specific transfer checklists

### Fixed Smoke

`structural_live_hybrid_smoke5_v2_20260603`

- structural vs base: `4 win / 1 loss`, utility `0.80`
- structural vs placebo: `3 win / 1 loss / 1 tie`, utility `0.70`

## Large Run

Eval id:

`structural_live_hybrid100_v2_gpt54mini_gpt55_20260603`

Command:

```bash
MODEL_ROUTER_TIMEOUT=25 python3 -m assumption_os.structural_live_ablation \
  --eval-id structural_live_hybrid100_v2_gpt54mini_gpt55_20260603 \
  --selection-mode hybrid \
  --max-cases 100 \
  --solver-model gpt_mini \
  --judge-model gpt55 \
  --judge-transport requests \
  --solve-workers 2 \
  --judge-workers 2
```

Artifacts:

- `phase four/assumption_graph/structural_live_ablation_20260603/structural_live_hybrid100_v2_gpt54mini_gpt55_20260603_summary.json`
- `phase four/assumption_graph/structural_live_ablation_20260603/structural_live_hybrid100_v2_gpt54mini_gpt55_20260603_judgments.json`

## Result

Pass gate: `true`

Overall:

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 57 | 38 | 5 | 0.595 | 0.57 | 0.38 |
| structural vs placebo | 100 | 54 | 34 | 12 | 0.600 | 0.54 | 0.34 |

Domain split, structural vs base:

| Domain | n | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|
| business | 15 | 0.8333 | 0.8000 | 0.1333 |
| daily_life | 15 | 0.5333 | 0.5333 | 0.4667 |
| engineering | 15 | 0.5333 | 0.5333 | 0.4667 |
| mathematics | 15 | 0.4000 | 0.3333 | 0.5333 |
| science | 15 | 0.6667 | 0.6000 | 0.2667 |
| software_engineering | 25 | 0.6000 | 0.6000 | 0.4000 |

Pattern split, structural vs base:

| Pattern | n | Utility |
|---|---:|---:|
| pat_controlled_intervention | 21 | 0.7143 |
| pat_counterexample_refinement | 20 | 0.6250 |
| pat_bottleneck_capacity | 17 | 0.6176 |
| pat_residual_correction | 9 | 0.6667 |
| pat_monotone_progress | 8 | 0.6250 |
| pat_signal_nuisance_separation | 4 | 0.7500 |
| pat_decomposition_composition | 14 | 0.5000 |
| pat_negative_feedback | 4 | 0.1250 |
| pat_incremental_replacement | 2 | 0.0000 |
| pat_conservation_balance | 1 | 0.0000 |

## Interpretation

The Morphism layer is now behavior-positive under gold routing:

- It beats baseline and placebo on the 100-case sample.
- It provides a measurable benefit beyond "no structural context."
- The best-supported patterns are controlled intervention, counterexample refinement, bottleneck capacity, residual correction, and signal/noise separation.

Main remaining gaps:

- Natural retrieval recall is still not solved; the large pass used `coverage_gold` routing.
- Mathematics regressed overall against base.
- `negative_feedback`, `incremental_replacement`, and `conservation_balance` have weak or too-small evidence.
- Decomposition/composition is borderline and needs stronger problem-specific realization guidance.

Next work:

- Improve natural structural extraction/routing so `coverage_gold` is not required.
- Add pattern-specific repair for math and decomposition/composition losses.
- Expand first-party traces and let failed morphism cases feed recursive hypothesis generation.
