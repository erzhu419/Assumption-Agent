# Full V3 Phase9 V1 Regression Validation - 2026-06-11

## Purpose

Phase9 tests the hard question that the frozen V1 comparison could not answer:
on the same fresh active slice, does the current V3 assumption/morphism pipeline
beat the older V1/V20-style critical-frame kernel?

All live calls used `gpt-5.4-mini` through environment variables only. No API
keys are stored in code or artifacts.

## Original Same-Batch Result

Artifact:
`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase9_v1_live_regression_20260611.json`

- Active fresh cases: 31
- V3 full vs V1 case-reflection utility: 0.5484
- Margin over tie: 0.0484
- Wins/losses: 17 / 14
- V3 full vs no-morphism utility: 0.7419

Interpretation: morphism context is useful, but the original V3 prompt did not
clear the hard V1 regression gate. The main weakness was not structural
retrieval; it was missing V1's critical-reframe behavior, especially in business
controlled-intervention problems.

## Failed Broad Repair

Artifact:
`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase9_frame_morphism_repair_20260611.json`

The broad "explicit frame + morphism" repair did not help:

- Repair vs V1 utility: 0.5484
- Repair vs original V3 utility: 0.2903

Interpretation: forcing the model to explicitly reason about frame/morphism made
answers too structured and harmed the original V3 behavior.

## Passing Compact Guard

Artifact:
`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase9_compact_frame_guard_20260611.json`

The compact guard keeps the reframe step implicit and asks for a direct final
answer:

- Compact guard vs V1 utility: 0.6774
- Compact guard vs V1 margin: 0.1774
- Gain over original V3 vs V1 margin: +0.1290
- Compact guard vs original V3 utility: 0.4839
- Calls: 31 answer calls + 62 judge calls

Interpretation: this clears the V1 hard regression gate, but it is retained as a
V1-regression profile rather than an unconditional default replacement, because
it is slightly below original V3 on direct pairwise comparison by roughly one
case.

## Current Policy

- Keep the original V3 quality profile as the main default.
- Retain compact frame guard as a gated V1-regression profile.
- Do not promote the broad explicit frame+morphism repair.
- Future work: learn a selector that activates compact frame guard only where
critical-reframe risk is high, rather than applying it to every routed problem.
