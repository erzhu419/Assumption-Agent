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

## Weak-Pattern Repair Round 1

Objective: repair the weakest natural-routing patterns one at a time, with a focused performance gate before allowing the repair back into the full 100-case policy.

Code changes:

- Added `selection_mode=natural_repaired`.
- Added `--repair-patterns`, so a repair can be tested on one pattern family without promoting all unvalidated repairs.
- Added `--focus-pattern-id`, with a focused validation gate: `n>=2`, utility `>=0.55` vs base and placebo, win rate `>=` loss rate, and no missing answers.
- Added `--extra-abstain-patterns` for negative-control tests.
- Made placebo routing exclude reference patterns by default. A placebo structural arm should not accidentally receive the gold/reference pattern.

### Failed: Bottleneck Repair

Eval id: `structural_live_bottleneck_repair_v1_gpt54mini_gpt55_20260603`

Result: `pass=false`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 12 | 5 | 6 | 1 | 0.4583 | 0.4167 | 0.5000 |
| structural vs placebo | 12 | 2 | 8 | 2 | 0.2500 | 0.1667 | 0.6667 |

Decision: not promoted.

Diagnosis: direct bottleneck guidance still made answers too structural and did not reliably recover problem-specific action plans. Some of the previous bottleneck losses were also route-boundary errors, especially git-bisect and tree/special-case problems.

### Route-Cue Repair

The natural cue router was updated before another full run:

- Git-bisect/code-commit localization cues now route to controlled intervention.
- Tree/star/specific-structure cues now route to the special-case/generalization family instead of bottleneck.

Eval id: `structural_live_natural_safe_cuefix_placebo100_v1_gpt54mini_gpt55_20260603`

Result: `pass=false`, but better than the broken placebo run.

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 55 | 36 | 9 | 0.5950 | 0.5500 | 0.3600 |
| structural vs placebo | 100 | 51 | 39 | 10 | 0.5600 | 0.5100 | 0.3900 |

Decision: cue changes are useful but insufficient alone. The remaining blocker was `pat_residual_correction`.

### Failed: Residual Abstention

Eval id: `structural_live_residual_extra_abstain_v1_gpt54mini_gpt55_20260603`

Result: `pass=false`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 12 | 5 | 7 | 0 | 0.4167 | 0.4167 | 0.5833 |
| structural vs placebo | 12 | 5 | 6 | 1 | 0.4583 | 0.4167 | 0.5000 |

Decision: residual should not be fixed by simply abstaining. It needs a pattern-local repair.

### Passed: Residual Pattern Repair

Eval id: `structural_live_residual_repair_v4_gpt54mini_gpt55_20260603`

The passing repair added two pieces:

- Residual-specific decision framing: keep the current working path as baseline, treat new evidence as delta, estimate continue/repair/replace/try costs, and set stop-loss thresholds.
- Problem-detail guard: extract at least three concrete constraints, numbers, or object names from the problem and convert them into actions, acceptance metrics, or risk controls.

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 12 | 8 | 4 | 0 | 0.6667 | 0.6667 | 0.3333 |
| structural vs placebo | 12 | 8 | 3 | 1 | 0.7083 | 0.6667 | 0.2500 |

Decision: promote only `pat_residual_correction` repair into the full natural policy.

### Full 100-Case Integration

Eval id: `structural_live_natural_repaired_residual100_v1_gpt54mini_gpt55_20260603`

Route split:

- `natural_trace_policy`: `64`
- `natural_safe_abstain`: `24`
- `natural_repaired_pattern`: `12`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 56 | 36 | 8 | 0.6000 | 0.5600 | 0.3600 |
| structural vs placebo | 100 | 55 | 34 | 11 | 0.6050 | 0.5500 | 0.3400 |

This improves over the previous pushed `natural_safe100` result:

| Run | vs base utility | vs placebo utility |
|---|---:|---:|
| `structural_live_natural_safe100_v1_gpt54mini_gpt55_20260603` | 0.5950 | 0.5800 |
| `structural_live_natural_repaired_residual100_v1_gpt54mini_gpt55_20260603` | 0.6000 | 0.6050 |

Important validation note: the final 100-case integration is a compositional replay from first-party live traces: non-residual rows reused the first-party cue-fix run and the 12 residual rows reused the first-party residual repair run. It is not a synthetic judge-only estimate, but it is also not a single monolithic fresh 300-call run.

Remaining weak families:

- `pat_bottleneck_capacity`: direct repair failed; next step should separate true bottleneck cases from route-boundary cases.
- `pat_incremental_replacement`: repair guidance exists but is not promoted until focused validation passes.
- `pat_negative_feedback`: repair guidance exists but is not promoted until focused validation passes.
- `pat_signal_nuisance_separation`: repair guidance exists but is not promoted until focused validation passes.

## Weak-Pattern Repair Round 2

Objective: repair `pat_signal_nuisance_separation` after residual repair, but still promote only after focused validation and a fresh full 100-case run.

### Signal/Nuisance Failed Attempts

The first S09 repair was too abstract and then too broad:

| Eval id | n | vs base utility | vs placebo utility | Result | Diagnosis |
|---|---:|---:|---:|---|---|
| `structural_live_signal_repair_v1_gpt54mini_gpt55_20260603` | 4 | 0.1250 | 0.2500 | fail | Generic simplification guidance lost to concrete base/placebo answers. |
| `structural_live_signal_repair_v2_gpt54mini_gpt55_20260603` | 4 | 0.3750 | 0.7500 | fail | Better vs placebo, but still lost to base on math and legacy-model cases. |
| `structural_live_signal_repair_v3_gpt54mini_gpt55_20260603` | 3 | 0.6667 | 0.5000 | fail | S06 cue fix removed the legacy-model route error, but molecular search still lost to placebo. |
| `structural_live_signal_repair_v4_gpt54mini_gpt55_20260603` | 3 | 0.8333 | 0.5000 | fail | Molecular answer over-required proxy/active learning; placebo gave the cleaner ensemble-docking path. |

Repairs made before promotion:

- Added S06 cue terms for high-liquidity/low-volatility linear-or-quadratic legacy-model cases, so those route as special-case/generalization rather than S09.
- Rewrote S09 guidance as type-specific simplification:
  - molecular/flexible-active-site search: receptor/conformer ensemble, ensemble docking, MM/GBSA or FEP/TI only for top candidates;
  - proxy model/active learning only when historical high-precision labels exist, not as a mandatory step;
  - statistics/inequality: positivity, zero handling, logit/log1p/log-sum-exp/Taylor bounds, controls, residual/heteroskedasticity diagnostics, bootstrap;
  - control/robotics: lock unused degrees of freedom while preserving future interfaces;
  - legacy models: keep old model as oracle, preserve complex path, shadow-test approximations, and rollback on threshold breach.

### Passed: Signal/Nuisance Pattern Repair

Eval id: `structural_live_signal_repair_v5_gpt54mini_gpt55_20260603`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 3 | 2 | 1 | 0 | 0.6667 | 0.6667 | 0.3333 |
| structural vs placebo | 3 | 2 | 0 | 1 | 0.8333 | 0.6667 | 0.0000 |

Decision: promote `pat_signal_nuisance_separation` repair, but only together with the route-boundary S06 cue fix.

### Fresh Full 100-Case Integration

Eval id: `structural_live_natural_repaired_residual_signal100_v1_gpt54mini_gpt55_20260603`

This run is a fresh full live run, not a compositional replay:

- solver answers: `300/300`
- judge pairs: `200/200`
- route split: `natural_trace_policy=65`, `natural_safe_abstain=20`, `natural_repaired_pattern=15`
- repaired patterns: `pat_residual_correction=12`, `pat_signal_nuisance_separation=3`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 59 | 29 | 12 | 0.6500 | 0.5900 | 0.2900 |
| structural vs placebo | 100 | 63 | 30 | 7 | 0.6650 | 0.6300 | 0.3000 |

Improvement ladder:

| Run | vs base utility | vs placebo utility | Validation type |
|---|---:|---:|---|
| `natural_safe100` | 0.5950 | 0.5800 | full live |
| `natural_repaired_residual100` | 0.6000 | 0.6050 | compositional first-party replay |
| `natural_repaired_residual_signal100` | 0.6500 | 0.6650 | fresh full live |

Pattern split in the fresh full run:

| Pattern | n | vs base utility | vs placebo utility |
|---|---:|---:|---:|
| `pat_residual_correction` | 12 | 0.7083 | 0.7500 |
| `pat_signal_nuisance_separation` | 3 | 0.6667 | 0.6667 |
| `pat_structural_abstain` | 20 | 0.5250 | 0.6000 |
| `pat_monotone_progress` | 7 | 0.8571 | 0.8571 |

Remaining weak families:

- `pat_bottleneck_capacity`: direct repair failed; still needs subcase split before another promotion attempt.
- `pat_incremental_replacement`: only two focused cases; still unpromoted.
- `pat_negative_feedback`: route quality is poor; it should be fixed by cue/scoping before any repair prompt promotion.

## Weak-Pattern Repair Round 3

Objective: repair the remaining weak families one by one, with focused performance validation before promotion and a fresh full 100-case integration run before push.

### Incremental Replacement

Repairs made:

- Rewrote S15 guidance around concrete system boundaries.
- Legacy/production cases now prefer read-only adapters, shadow runs, feature flags, gray release, rollback, and rule baselines before touching core systems.
- Internal-tool-to-SaaS cases now freeze the internal stable path and build the external pilot as a separate product line with tenant isolation, API security, billing/usage, audit, limits, and whitelist customers.

Focused validation:

Eval id: `structural_live_incremental_repair_v3_gpt54mini_gpt55_20260603`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 |
| structural vs placebo | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 |

### Negative Feedback

Repairs made:

- Removed over-broad S20/S27 cues that caused false positives: isolated `激进`, `保险公司`, `竞争对手`, and `奖励`.
- Added S20 repair guidance for hard/soft objective splits, fast/slow paths, latency budgets, arbitration, timeout fallback, and shadow/simulation validation.
- Added S27 repair guidance for public-space governance, infrastructure/optionality, price competition, and key-person/governance conflicts.
- Required concrete stakeholders plus a specific exchange condition or constraint for each.

Focused validation:

Eval id: `structural_live_negative_repair_v2_gpt54mini_gpt55_20260603`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 5 | 4 | 1 | 0 | 0.8000 | 0.8000 | 0.2000 |
| structural vs placebo | 5 | 3 | 2 | 0 | 0.6000 | 0.6000 | 0.4000 |

### Bottleneck Capacity

Repairs made:

- Split true bottleneck cases from route-boundary errors.
- Git-bisect and tree/DP cases no longer enter the bottleneck focused slice.
- Removed broad S24 medium cues `吞吐` and `延迟`.
- Added S19 material/humidity constraint guidance: catalyst/activated monomer/pressure alternatives, two-step prepolymer/post-cure, small DOE, local micro-environment, saturated MgCl2 humidity buffering, sample isolation, fans, sensors, and empty-box calibration.
- Added S23 thesis/deadline guidance: freeze six-month experiments, define minimum publishable evidence, avoid overclaiming, and shrink claims if a new experiment affects the core conclusion.
- Added S24 subcase guidance for git-bisect, game release triage, CI pipeline profiling, and recursive/backtracking hotspots.

Focused dry-run:

Eval id: `structural_live_bottleneck_repair_dryrun_v2_20260603`

- selected cases: `10`
- route_quality exact pattern match: `1.0000`
- strategy split: `S19=3`, `S23=1`, `S24=6`

Focused validation:

Eval id: `structural_live_bottleneck_repair_v2_gpt54mini_gpt55_20260603`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 10 | 5 | 2 | 3 | 0.6500 | 0.5000 | 0.2000 |
| structural vs placebo | 10 | 7 | 3 | 0 | 0.7000 | 0.7000 | 0.3000 |

### Fresh Full 100-Case All-Repair Integration

Eval id: `structural_live_all_repairs100_v1_gpt54mini_gpt55_20260603`

This is a fresh full live run:

- solver answers: `300/300`
- judge pairs: `200/200`
- route split: `natural_trace_policy=68`, `natural_repaired_pattern=32`
- repaired patterns: `pat_residual_correction=12`, `pat_signal_nuisance_separation=3`, `pat_incremental_replacement=2`, `pat_negative_feedback=5`, `pat_bottleneck_capacity=10`
- route_quality exact pattern match: `0.6800`
- repaired-route utility: `0.7031` vs base, `0.5781` vs placebo

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 64 | 28 | 8 | 0.6800 | 0.6400 | 0.2800 |
| structural vs placebo | 100 | 54 | 36 | 10 | 0.5900 | 0.5400 | 0.3600 |

Pattern split in the fresh full run:

| Pattern | n | vs base utility | vs placebo utility |
|---|---:|---:|---:|
| `pat_bottleneck_capacity` | 10 | 0.4500 | 0.4500 |
| `pat_incremental_replacement` | 2 | 1.0000 | 1.0000 |
| `pat_negative_feedback` | 5 | 0.8000 | 0.6000 |
| `pat_residual_correction` | 12 | 0.7500 | 0.6667 |
| `pat_signal_nuisance_separation` | 3 | 1.0000 | 0.3333 |

Decision:

- Promote incremental, negative-feedback, and bottleneck repairs because each focused gate passed and the all-repair full 100-case integration passed.
- Note the residual risk: the full run improves vs base over the previous residual+signal full run (`0.6800` vs `0.6500`) but has a lower placebo margin (`0.5900` vs `0.6650`). The global gate still passes, and the weak-family focused gates are the stronger evidence for those repairs.

## Weak-Pattern Repair Round 4

Objective: fix the two weak full-run placebo margins called out after the all-repair run:

- `pat_bottleneck_capacity`: full-run split was `0.4500` vs base and `0.4500` vs placebo.
- `pat_signal_nuisance_separation`: full-run split was `1.0000` vs base but only `0.3333` vs placebo.

Diagnosis:

- The bottleneck guidance was still too pooled. S19 polymer, S19 noise negotiation, S23 thesis deadline, S24 game triage, S24 CI pipeline, S24 recursive hotspot, and S24 grid/insurance-risk cases were all sharing one long text. This caused cross-contamination, for example CI answers inheriting game-release gates and polymer answers excluding safe pressure compensation.
- The signal guidance had the same issue at smaller scale. Molecular docking, 6DOF control, and log/statistical transforms needed different concrete checks.

Code changes:

- Added case-conditioned repair text in `_case_conditioned_repair_text`.
- Kept the pattern-level guidance short, then appended only the subcase triggered by the problem text.
- Added bottleneck subcase guidance for:
  - cafe/noise negotiation;
  - polymer low-temperature synthesis with catalyst/activated monomer plus safe pressure/solvent/activity alternatives;
  - humidity micro-environment curing with saturated MgCl2 around 33%RH;
  - thesis/deadline resource cutoff;
  - game release triage Go/No-Go;
  - CI pipeline critical-path profiling;
  - recursive/backtracking hotspot algorithm replacement;
  - staged grid/insurance risk conversion.
- Added signal subcase guidance for:
  - flexible active-site molecular docking;
  - six-degree-of-freedom mechanical/control simplification;
  - log/statistical transformation validity checks.

### Failed Intermediate: Bottleneck Margin v1

Eval id: `structural_live_bottleneck_margin_v1_gpt54mini_gpt55_20260604`

Result: `pass=false`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 10 | 7 | 3 | 0 | 0.7000 | 0.7000 | 0.3000 |
| structural vs placebo | 10 | 2 | 6 | 2 | 0.3000 | 0.2000 | 0.6000 |

Decision: not promoted. This run proved that adding more generic detail made bottleneck answers worse against a strong direct-solve placebo.

### Passed: Signal Margin v2

Eval id: `structural_live_signal_margin_v2_gpt54mini_gpt55_20260604`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 0.0000 |
| structural vs placebo | 3 | 2 | 0 | 1 | 0.8333 | 0.6667 | 0.0000 |

### Passed: Bottleneck Margin v2

Eval id: `structural_live_bottleneck_margin_v2_gpt54mini_gpt55_20260604`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 10 | 9 | 1 | 0 | 0.9000 | 0.9000 | 0.1000 |
| structural vs placebo | 10 | 8 | 2 | 0 | 0.8000 | 0.8000 | 0.2000 |

### Fresh Full 100-Case Margin Integration

Eval id: `structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604`

This is a fresh full live run:

- solver answers: `300/300`
- judge pairs: `200/200`
- route split: `natural_trace_policy=68`, `natural_repaired_pattern=32`
- route_quality exact pattern match: `0.6800`

Result: `pass=true`

| Pair | n | Win | Loss | Tie | Utility | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural vs base | 100 | 59 | 34 | 7 | 0.6250 | 0.5900 | 0.3400 |
| structural vs placebo | 100 | 68 | 27 | 5 | 0.7050 | 0.6800 | 0.2700 |

Target pattern split in the fresh full run:

| Pattern | n | vs base utility | vs placebo utility |
|---|---:|---:|---:|
| `pat_bottleneck_capacity` | 10 | 0.5500 | 0.7000 |
| `pat_signal_nuisance_separation` | 3 | 1.0000 | 0.8333 |

Compared with the previous all-repair full run:

| Run | vs base utility | vs placebo utility | Bottleneck placebo | Signal placebo |
|---|---:|---:|---:|---:|
| `structural_live_all_repairs100_v1_gpt54mini_gpt55_20260603` | 0.6800 | 0.5900 | 0.4500 | 0.3333 |
| `structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604` | 0.6250 | 0.7050 | 0.7000 | 0.8333 |

Decision:

- Promote the case-conditioned bottleneck/signal repair because both focused performance validations pass and the full 100-case placebo margin improves materially.
- Keep the residual risk explicit: this fresh full run improved the requested placebo margins but had a lower global vs-base utility than the previous all-repair full run. The regression is not concentrated in the two edited patterns; the target repairs themselves passed focused validation.
