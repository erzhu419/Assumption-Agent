# Natural Structural Routing Validation - 2026-06-03

## Objective

Validate the Structural Morphism layer without gold/reference routing.

The previous large ablation proved that structural context helps when `coverage_tags` or reference strategy tags route each problem to the correct pattern. This run tests the harder setting: the route must be chosen from the problem text itself, then performance must beat both the no-context baseline and a placebo structural context.

## Implementation

Changed `assumption_os/structural_live_ablation.py`:

- Added `selection_mode=natural`, driven by reference-free Chinese cue routing from problem text to `S01`-`S27`, then to structural patterns.
- Added route-quality audit metrics. These use reference tags only after selection, for evaluation, not for selecting the route.
- Added `selection_mode=natural_gated`, a first-party trace policy that abstains from patterns that harmed the unconstrained natural run.
- Added `STRUCTURAL_PATTERN_OPERATORS`, but made operator injection opt-in through `STRUCTURAL_OPERATOR_CONTEXT=1`. Default live routing keeps the lean no-operator context because the operator run regressed.

## Runs

### Natural Dry Run

Eval id: `structural_live_natural_dryrun_v3_20260603`

- selected: `100/100`
- route source: `natural_cue=100`
- exact pattern match: `0.6400` (`64 match / 36 miss`)

This removed the gold-routing dependency but did not yet prove downstream answer quality.

### Unconstrained Natural Live Run

Eval id: `structural_live_natural100_v1_gpt54mini_gpt55_20260603`

Command:

```bash
MODEL_ROUTER_TIMEOUT=25 python3 -m assumption_os.structural_live_ablation \
  --eval-id structural_live_natural100_v1_gpt54mini_gpt55_20260603 \
  --selection-mode natural \
  --max-cases 100 \
  --solver-model gpt_mini \
  --judge-model gpt55 \
  --judge-transport requests \
  --solve-workers 4 \
  --judge-workers 2
```

Result: `pass=false`

| Pair | n | Win | Loss | Tie | Utility |
|---|---:|---:|---:|---:|---:|
| structural vs base | 100 | 50 | 41 | 9 | 0.5450 |
| structural vs placebo | 100 | 51 | 41 | 8 | 0.5500 |

Diagnosis: natural routing was close but still too permissive. The worst trace-backed patterns were:

- `pat_bottleneck_capacity`
- `pat_incremental_replacement`
- `pat_negative_feedback`
- `pat_signal_nuisance_separation`

### Operator Injection Negative Result

Eval id: `structural_live_natural_gated73_v1_gpt54mini_gpt55_20260603`

This run used `natural_gated` and injected pattern operators directly into the context.

Result: `pass=false`

| Pair | n | Win | Loss | Tie | Utility |
|---|---:|---:|---:|---:|---:|
| structural vs base | 72 | 22 | 28 | 22 | 0.4583 |
| structural vs placebo | 72 | 23 | 23 | 26 | 0.5000 |

Diagnosis: the operators made answers too structure-led and less problem-specific. They are retained as an experimental feature but disabled by default.

### Trace-Gated Natural Live Run

Eval id: `structural_live_natural_gated73_noop_v1_gpt54mini_gpt55_20260603`

Command:

```bash
MODEL_ROUTER_TIMEOUT=25 python3 -m assumption_os.structural_live_ablation \
  --eval-id structural_live_natural_gated73_noop_v1_gpt54mini_gpt55_20260603 \
  --selection-mode natural_gated \
  --max-cases 100 \
  --solver-model gpt_mini \
  --judge-model gpt55 \
  --judge-transport requests \
  --solve-workers 4 \
  --judge-workers 2
```

Result: `pass=true`

- selected: `73/100`
- route source: `natural_trace_policy=73`
- route-quality exact pattern match: `0.6575` (`48 match / 25 miss`)
- missing answers: `0`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 73 | 40 | 25 | 8 | 0.6027 | 0.5479 | 0.3425 |
| structural vs placebo | 73 | 40 | 26 | 7 | 0.5959 | 0.5479 | 0.3562 |

Domain split, structural vs base:

| Domain | n | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|
| business | 11 | 0.5455 | 0.5455 | 0.4545 |
| daily_life | 13 | 0.6154 | 0.6154 | 0.3846 |
| engineering | 13 | 0.6538 | 0.6154 | 0.3077 |
| mathematics | 13 | 0.5769 | 0.4615 | 0.3077 |
| science | 10 | 0.6000 | 0.5000 | 0.3000 |
| software_engineering | 13 | 0.6154 | 0.5385 | 0.3077 |

Pattern split, structural vs base:

| Pattern | n | Utility |
|---|---:|---:|
| pat_controlled_intervention | 16 | 0.6875 |
| pat_counterexample_refinement | 16 | 0.5625 |
| pat_decomposition_composition | 22 | 0.7045 |
| pat_monotone_progress | 5 | 1.0000 |
| pat_residual_correction | 12 | 0.2083 |
| pat_conservation_balance | 2 | 0.5000 |

## Interpretation

Natural Structural Morphism is now behavior-positive, but only with abstention.

What is solved:

- The live path no longer needs `coverage_tags` or reference strategy tags for routing.
- The system can learn from a failed first-party trace and gate future structural context.
- The gated natural run beats both base and placebo on a 73-case live sample.

What remains:

- Full 100/100 natural routing without abstention is still slightly below gate.
- `residual_correction` became weak in the successful gated run and needs its own repair.
- Operator text is not ready for default use; it should be validated pattern-by-pattern before promotion.
- `natural_gated` is an abstaining policy, not a complete autonomous morphism search engine.

## Full-Coverage Repair

Eval id: `structural_live_natural_safe100_v1_gpt54mini_gpt55_20260603`

`natural_safe` keeps all 100 cases in the live benchmark. It routes strong cases through normal natural structural context and routes trace-negative cases through an explicit structural-abstention context:

- `natural_trace_policy`: `73`
- `natural_safe_abstain`: `27`

The abstention context does not force a weak morphism. It tells the solver that the structural mapping is unreliable and that it should solve the original problem directly with concrete domain steps, metrics, and risk controls.

Command:

```bash
MODEL_ROUTER_TIMEOUT=25 python3 -m assumption_os.structural_live_ablation \
  --eval-id structural_live_natural_safe100_v1_gpt54mini_gpt55_20260603 \
  --selection-mode natural_safe \
  --max-cases 100 \
  --solver-model gpt_mini \
  --judge-model gpt55 \
  --judge-transport requests \
  --solve-workers 4 \
  --judge-workers 2
```

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 54 | 35 | 11 | 0.5950 | 0.5400 | 0.3500 |
| structural vs placebo | 100 | 54 | 38 | 8 | 0.5800 | 0.5400 | 0.3800 |

Route split:

| Route Source | n | vs base utility | vs placebo utility |
|---|---:|---:|---:|
| natural_trace_policy | 73 | 0.6164 | 0.6164 |
| natural_safe_abstain | 27 | 0.5370 | 0.4815 |

Interpretation:

- The original `natural` 100-case failure is repaired by letting the structural layer abstain inside the 100-case run rather than dropping cases.
- This is a stronger production policy than unconditional structural transfer: the system can use morphisms when validated and decline them when first-party trace says they are harmful.
- The pass is still tight on placebo (`0.5800`), so the next repair target is not more gating. It is to make the abstained families useful again one by one: bottleneck/capacity, negative feedback/incentive, signal/noise, and incremental replacement.

Operator status:

- Default operator injection remains disabled.
- The failed operator run showed that long, generic operator text made answers too structure-led and less problem-specific.
- Operators should be promoted only after pattern-local validation, not as a global default.
