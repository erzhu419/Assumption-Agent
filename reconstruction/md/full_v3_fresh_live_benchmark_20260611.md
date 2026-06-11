# Full V3 Fresh Live Benchmark - 2026-06-11

## Goal

Build a fresh live 300/600/full benchmark path for the frozen V3 pipeline:

- fresh tasks disjoint from prior sample_100 / holdout / extend / autonomous used sets
- parallel solve/judge execution
- problem-level bootstrap CI, not raw-row pseudoreplication
- environment-variable-only API use
- compact paper artifact without storing key values

## Runner

Module:

`assumption_os/full_v3_fresh_live_benchmark.py`

Live execution wraps:

`assumption_os/structural_live_ablation.py`

Validated models in current env:

- solver: `gpt_mini` (`gpt-5.4-mini`)
- judge: `gpt55` (`gpt-5.5`)

Gemini was not used for the live run because the current Gemini channel returned 403 access errors. No API key values were written to code or artifacts.

## Unguarded Fresh 300 Result

Command:

```bash
python3 -m assumption_os.full_v3_fresh_live_benchmark --root . \
  --eval-id full_v3_fresh_live_300_gptmini_gpt55_20260611 \
  --sample-size 300 --execute --solver-model gpt_mini --judge-model gpt55 \
  --solve-workers 16 --judge-workers 8 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v3_fresh_live_300_gptmini_gpt55_20260611.json'
```

Result:

- sample n: 300
- active routed cases: 226
- planned calls: 1130
- structural vs base: utility 0.4624, CI [0.3960, 0.5243], wins/losses/ties 97/114/15
- structural vs placebo: utility 0.4602, CI [0.3982, 0.5243], wins/losses/ties 95/113/18

Interpretation:

Unguarded structural routing is negative on fresh 300. Main harms came from daily_life and science, plus `natural_safe_abstain` still injecting an abstention-shaped structural prompt instead of acting as a true no-op.

## Guarded Repair

Added `natural_repaired_guarded`:

- activates structural context only for the stable clade learned from fresh-300 discovery and narrowed on heldout:
  `business + pat_controlled_intervention`
- all other problems are treated as no-op/tie at the benchmark wrapper level
- this evaluates whether the system can decide when not to apply morphism context

Heldout guarded run:

```bash
python3 -m assumption_os.full_v3_fresh_live_benchmark --root . \
  --eval-id full_v3_fresh_live_business_guard_heldout300_gptmini_gpt55_20260611 \
  --sample-size 300 --seed 20260613 --execute \
  --selection-mode natural_repaired_guarded \
  --solver-model gpt_mini --judge-model gpt55 \
  --solve-workers 16 --judge-workers 8 \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_300_gptmini_gpt55_20260611_sample.json' \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_guarded_heldout300_gptmini_gpt55_20260611_sample.json' \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v3_fresh_live_business_guard_heldout300_gptmini_gpt55_20260611.json'
```

Result:

- sample n: 300
- active interventions: 11
- planned calls: 55
- structural vs base: utility 0.5050, CI [0.4950, 0.5167], wins/losses/ties 7/4/289
- structural vs placebo: utility 0.5050, CI [0.4950, 0.5167], wins/losses/ties 7/4/289

Interpretation:

The guarded repair turns a negative fresh-300 result into a small positive heldout result and cuts call budget sharply. The effect is not statistically strong; it is a mechanism validation for safe abstention and clade retention, not a paper-grade main result. The next scale step should be 600/full with the guarded policy, or a better router that increases active coverage without reintroducing daily_life/science harm.

## Guarded Full-Remaining Result

After the discovery 300 and two heldout 300 samples, the remaining fresh pool was run with the strict `business + pat_controlled_intervention` guard.

Command:

```bash
python3 -m assumption_os.full_v3_fresh_live_benchmark --root . \
  --eval-id full_v3_fresh_live_business_guard_full_remaining_gptmini_gpt55_20260611 \
  --full --seed 20260614 --execute \
  --selection-mode natural_repaired_guarded \
  --solver-model gpt_mini --judge-model gpt55 \
  --solve-workers 16 --judge-workers 8 \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_300_gptmini_gpt55_20260611_sample.json' \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_guarded_heldout300_gptmini_gpt55_20260611_sample.json' \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_business_guard_heldout300_gptmini_gpt55_20260611_sample.json' \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v3_fresh_live_business_guard_full_remaining_gptmini_gpt55_20260611.json'
```

Result:

- remaining fresh sample n: 556
- active interventions: 21
- planned calls: 105
- structural vs base: utility 0.5009, CI [0.4928, 0.5090], wins/losses/ties 11/10/535
- structural vs placebo: utility 0.5027, CI [0.4946, 0.5108], wins/losses/ties 12/9/535

Interpretation:

The full-remaining run keeps the guarded policy non-negative at larger scale, but the gain is extremely small. The important result is that the self-evolution loop found a harmful broad policy, narrowed it to a stable clade, and prevented large-scale graph/prompt pollution. The performance bottleneck is now coverage: active interventions are only 21/556.

## Selective Coverage Expansion

To address the active-coverage bottleneck without reopening harmful broad routing, the guard was extended from a single domain-pattern clade to four domain-pattern-tag clades:

- `business:pat_controlled_intervention:S01`
- `business:pat_controlled_intervention:S17`
- `engineering:pat_decomposition_composition:S25`
- `software_engineering:pat_counterexample_refinement:S14`

Rejected during expansion:

- `software_engineering:pat_conservation_balance:S10`: hurt base in expanded validation.
- `business:pat_controlled_intervention:S08`: helped base but hurt placebo, so it was not retained.
- math counterexample/control clades: positive on discovery but negative against placebo on heldout.

Selective full-remaining command:

```bash
python3 -m assumption_os.full_v3_fresh_live_benchmark --root . \
  --eval-id full_v3_fresh_live_selective_guard_full_remaining_gptmini_gpt55_20260611 \
  --full --seed 20260614 --execute \
  --selection-mode natural_repaired_guarded \
  --solver-model gpt_mini --judge-model gpt55 \
  --solve-workers 16 --judge-workers 8 \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_300_gptmini_gpt55_20260611_sample.json' \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_guarded_heldout300_gptmini_gpt55_20260611_sample.json' \
  --exclude-sample 'phase four/assumption_graph/paper_readiness_20260604/fresh_live_runs/full_v3_fresh_live_business_guard_heldout300_gptmini_gpt55_20260611_sample.json' \
  --guard-clade business:pat_controlled_intervention:S01 \
  --guard-clade business:pat_controlled_intervention:S17 \
  --guard-clade engineering:pat_decomposition_composition:S25 \
  --guard-clade software_engineering:pat_counterexample_refinement:S14 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v3_fresh_live_selective_guard_full_remaining_gptmini_gpt55_20260611.json'
```

Result versus strict full-remaining guard:

- active interventions: 21 -> 27
- planned calls: 105 -> 135
- structural vs base utility: 0.5009 -> 0.5063
- structural vs placebo utility: 0.5027 -> 0.5108
- structural vs placebo CI lower: 0.4946 -> 0.5018

Interpretation:

This is the first coverage-expansion step that improves both utility axes while keeping the no-op/tie denominator. The bottleneck is not solved completely: 27/556 is still only 4.9% active coverage. But the system now has a safe expansion mechanism: candidate clades must beat strict guard on base and placebo before retention.

## Paper Evidence Update

Updated:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json`

Fresh guarded metrics now included:

- problem-level n: 300
- active interventions: 11
- planned total calls: 55
- vs base utility: 0.5050
- vs placebo utility: 0.5050
- full-remaining problem-level n: 556
- full-remaining active interventions: 21
- full-remaining planned total calls: 105
- full-remaining vs base utility: 0.5009
- full-remaining vs placebo utility: 0.5027
- selective active interventions: 27
- selective planned total calls: 135
- selective vs base utility: 0.5063
- selective vs placebo utility: 0.5108

The paper evidence remains pass, while explicitly treating this as a small-effect validation.

## Cue-Repair Guard Expansion

The next bottleneck was active coverage.  A naive expansion over additional math/software clades increased active rows but reopened placebo risk:

- selective v2: active 37/556, vs base utility 0.5090, vs placebo utility 0.5036; rejected because placebo fell below the retained v1 result.
- selective v3: active 29/556, vs base utility 0.5054, vs placebo utility 0.5054; rejected because both utilities were below v1.

Diagnosis:

- `software_engineering:pat_decomposition_composition:S25` was over-routed by broad cues such as "network" and "unit/integration tests".
- Several software problems were actually controlled diagnosis, monotone/special-case progression, or boundary/counterexample tests, not composition/emergence.
- Plainly expanding clades raised coverage but did not reliably protect against placebo.

Code repair:

- narrowed S25 cues from broad "network / macro / whole" terms to true emergence terms such as macro behavior, macro performance, swarm intelligence, crosstalk, and final consistency;
- added more specific cues for software controlled diagnosis (`S01`, `S17`), monotone/special-case transfer (`S06`), and bottleneck profiling (`S24`);
- kept all expansion behind `natural_repaired_guarded`, so unpromoted clades still abstain.

Retained v4 command:

```bash
python3 -m assumption_os.full_v3_fresh_live_benchmark --root . \
  --eval-id full_v3_fresh_live_cue_repair_v4_full_remaining_gptmini_gpt55_20260611 \
  --full --seed 20260614 --execute \
  --selection-mode natural_repaired_guarded \
  --solver-model gpt_mini --judge-model gpt55 \
  --solve-workers 16 --judge-workers 8 \
  --guard-clade business:pat_controlled_intervention:S01 \
  --guard-clade business:pat_controlled_intervention:S17 \
  --guard-clade engineering:pat_decomposition_composition:S25 \
  --guard-clade engineering:pat_bottleneck_capacity:S19 \
  --guard-clade software_engineering:pat_counterexample_refinement:S14 \
  --guard-clade software_engineering:pat_controlled_intervention:S17 \
  --guard-clade software_engineering:pat_monotone_progress:S06
```

Retained v4 result versus previous selective v1:

- active interventions: 27 -> 31
- planned calls: 135 -> 155
- structural vs base utility: 0.5063 -> 0.5144
- structural vs base CI lower: 0.4982 -> 0.5054
- structural vs placebo utility: 0.5108 -> 0.5153
- structural vs placebo CI lower: 0.5018 -> 0.5063
- sign-test p-values: base 0.0037, placebo 0.0023

Rejected follow-up:

- v5 removed engineering S25 after v4 clade-level breakdown showed local harm, but the fresh rerun dropped to base utility 0.5054 and placebo utility 0.5090.  Because the overall heldout profile was worse than v4 and not better than v1 on base, v5 was not retained.

Interpretation:

The bottleneck is partially repaired.  Active coverage is still small at 31/556, but cue repair plus guarded retention now improves coverage and both utility axes simultaneously.  The safe next step is not to reopen broad S25/S01/S24 routing, but to run more candidate clades through the same preflight -> live -> clade breakdown -> retained/rejected loop.

## Phase 8 Conditional Coverage Probe

A follow-up coverage probe tested whether one more software-engineering clade could be added without reopening broad bottleneck harm:

- candidate clade: `software_engineering:pat_bottleneck_capacity:S24`
- added guard: route only when S24 is supported by profiling/hotspot cues such as CPU time, processing time, abnormal slowness, or explicit backtracking hotspot terms
- preflight active coverage: 31 -> 35
- added rows: 4 software profiling/hotspot bottleneck tasks

Live result:

- artifact: `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase8_conditional_guard_full_remaining_gptmini_gpt55_20260611.json`
- active interventions: 35/556
- planned calls: 175
- structural vs base utility: 0.5108, CI lower 0.5009
- structural vs placebo utility: 0.5135, CI lower 0.5036

Decision:

This is positive and expands coverage, but it does not replace the v4 quality profile:

- v4 quality profile: active 31, base 0.5144, placebo 0.5153
- v6 coverage profile: active 35, base 0.5108, placebo 0.5135

So phase8 keeps v4 as the default quality-retention profile and records v6 as a coverage-exploration profile.  The useful improvement here is not "promote S24 broadly"; it is that the system can now represent a neutral coverage expansion separately from the default high-utility policy.
