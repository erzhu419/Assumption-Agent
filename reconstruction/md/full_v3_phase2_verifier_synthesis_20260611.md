# Full V3 Phase 2 Verifier Synthesis - 2026-06-11

## Scope

This phase implements the Phase 2 v3 verifier-synthesis requirement from `reconstruction_v2_full.md`.

For each candidate assumption, the system synthesizes:

- positive tests
- negative controls
- placebo controls
- regression tests
- minimal falsification cases
- scope-boundary tests
- fresh-distribution tests

The synthesized verifier contract then decides whether to accept, reject, defer, or route to execution repair.

## Artifact

- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase2_verifier_synthesis_20260611.json`

## Performance Validation

Result: pass.

Metrics:

- candidate_count: 5
- test_count: 35
- test_type_coverage: 1.0
- contract_completeness: 1.0
- decision_accuracy: 1.0
- false_positive_rate_of_acceptance: 0.0
- regression_detection_recall: 1.0
- placebo_sensitivity: 1.0
- fresh_split_generalization: 1.0
- falsification_power: 1.0
- execution_lapse_new_hypothesis_count: 0

## Interpretation

Phase 2 v3 now verifies that candidates carry synthesized falsification contracts rather than manually specified judge checks.  The module accepts only the in-scope good candidate, rejects overbroad/placebo/regressive candidates, and routes execution lapses to repair instead of inventing a new assumption.

