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

The paper evidence remains pass, while explicitly treating this as a small-effect validation.
