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

## Heldout Selector Attempts

The first selector used compact guard on all S14/S19 heldout cases.

Artifact:
`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase9_selective_compact_guard_heldout_20260611.json`

- Heldout active cases: 54
- Selected compact cases: 17
- Policy vs V1 utility: 0.6111
- Policy vs V1 margin: 0.1111
- Lift over original V3-vs-V1: +0.0185
- Policy vs original V3 utility: 0.4722
- Result: failed `policy_improves_over_v3_heldout` and
  `policy_noninferior_to_original_v3`

Interpretation: compact guard is strong against V1 on selected cases, but tag
level routing is too coarse and still over-structures some answers.

The second selector tested a V3-preserving micro guard on the same S14/S19
heldout cases.

Artifact:
`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase9_micro_guard_heldout_20260611.json`

- Heldout active cases: 54
- Selected micro cases: 17
- Micro vs V1 on selected cases: 0.6471
- Micro vs original V3 on selected cases: 0.5882
- Policy vs V1 utility: 0.5926
- Policy vs original V3 utility: 0.5278
- Result: fixed the V3 regression, but did not improve V1 margin

Interpretation: micro guard is safer than compact guard, but too weak to solve
the V1 regression by itself.

## Passing Hybrid Guard

Artifact:
`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase9_hybrid_guard_heldout_20260611.json`

The retained Phase9 profile is a cue-level hybrid selector:

- keep original V3 for non-S14/S19 and for formal-proof/high-risk unmatched
  cases;
- use micro guard for common-cause, hidden-dependency deletion, and hard
  ecological/ethical-constraint cases;
- use compact guard for urgent triage, infinite-loop/termination, medical
  safety robustness, staged global scaling, explicit multi-objective balancing,
  and latency/resource tradeoff cases.

Performance:

- Heldout active cases: 54
- Candidate S14/S19 cases: 17
- Chosen arms: 40 original V3, 6 micro guard, 8 compact guard
- Original V3 vs V1 utility: 0.5926
- Hybrid vs V1 utility: 0.6481
- Hybrid vs V1 margin: 0.1481
- Lift over original V3-vs-V1: +0.0555
- Hybrid vs original V3 utility: 0.6111
- Result: all heldout gates passed

Interpretation: the final retained policy is not "use a longer prompt."  It is
selective retention over three arms after two rejected repairs.  The useful
behavior is exactly the recursive loop: failure residual -> candidate repair ->
heldout ablation -> rejection -> narrower repair -> selective retention.

## Current Policy

- Keep the original V3 quality profile as the main default.
- Retain the Phase9 hybrid guard as the gated V1-regression profile.
- Keep compact and micro guard as internal arms of that selector, not defaults.
- Do not promote the broad explicit frame+morphism repair.
- Future work: replace the cue rules with a learned world-model selector and
  revalidate on a new fresh active slice.
