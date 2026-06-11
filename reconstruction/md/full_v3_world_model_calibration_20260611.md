# Full V3 World Model Calibration / Leave-Domain-Out Audit

Date: 2026-06-11

## Purpose

This closes the remaining world-model calibration gap in the V3 reconstruction line.

The important distinction is:

- A world model can be useful as a cheap verifier / exploration gate.
- It should not be promoted as a production simulator unless it is calibrated and survives leave-domain-out checks.

## What Changed

Added `assumption_os/full_v3_world_model_calibration.py`.

The audit reads committed, redacted artifacts only:

- `full_v3_phase8_creativity_world_coverage_20260611.json`
- `full_v3_phase9_v1_live_regression_20260611.json`
- `full_v3_phase10_discrete_world_model_selector_20260611.json`
- `full_v3_phase5_contextual_bandit_scheduler_20260611.json`

It creates four calibration surfaces:

1. `phase8_quality_profile_world_model`
2. `phase9_leave_domain_out_nonregression`
3. `phase10_discrete_graph_action_world_model`
4. `phase5_scheduler_promotion_gate`

## Performance Validation

Generated artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_world_model_calibration_20260611.json`

Key metrics:

- `source_artifact_count`: 4
- `calibration_surface_count`: 4
- `calibrated_surface_count`: 2
- `leave_domain_out_surface_count`: 1
- `uncalibrated_promotion_count`: 0
- `phase8_quality_brier_improvement`: 0.1066
- `phase9_leave_domain_out_domain_count`: 3
- `phase9_leave_domain_out_nonnegative_domain_count`: 2
- `phase9_leave_domain_out_max_calibration_error`: 0.3611
- `phase10_all_lift_over_v3`: 0.0185
- `phase10_calibration_beats_base_rate`: false
- `phase10_selected_arm_mae_minus_base_rate`: 0.0098
- `phase5_keeps_phase10_candidate`: true

Interpretation:

- Phase8 profile-level world model is calibrated enough for scheduler gating.
- Phase9 leave-domain-out is available and records a real boundary: business-domain negative transfer.
- Phase10 discrete graph-action world model has positive utility lift, but scalar reward calibration is worse than base-rate prediction.
- Therefore Phase10 remains an exploration candidate and is blocked from production promotion.

## Integrated Gates

Phase11 capability audit now includes:

- `phase10_world_model_calibration`
- status: `calibration_audit_blocks_uncalibrated_world_model`

Paper-scale evidence now includes:

- required artifact count: 28
- v3 mechanism count: 12
- world-model calibration surface metrics
- explicit gates blocking uncalibrated Phase10 promotion

## Validation Commands

```bash
python3 -m assumption_os.full_v3_world_model_calibration --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_world_model_calibration_20260611.json"
python3 -m assumption_os.full_v3_phase11_capability_audit --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase11_capability_audit_20260611.json"
python3 -m assumption_os.full_v3_paper_scale_evidence --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json"
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_full_v3_world_model_calibration_blocks_uncalibrated_promotion tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase11_capability_audit_separates_fixture_from_production tests.test_assumption_os.AssumptionOSTest.test_full_v3_paper_scale_evidence_aggregates_live_and_mechanism_artifacts tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase10_discrete_world_model_selector_beats_original_v3
```

Targeted test result: 4 tests OK.

## Current Claim

The world model is no longer only an unqualified positive artifact. It is now a calibrated promotion system:

- calibrated surfaces can support gating;
- uncalibrated but positive surfaces remain candidates;
- leave-domain-out failures become explicit boundaries;
- production promotion is blocked until calibration and domain robustness improve.
