# Full V3 World Model Calibration / Leave-Domain-Out Audit

Date: 2026-06-11

## Purpose

This closes the remaining world-model calibration gap in the V3 reconstruction line and records the follow-up repair.

The important distinction is:

- A world model can be useful as a cheap verifier / exploration gate.
- A raw predictor should not be promoted as a production simulator unless it is calibrated and survives leave-domain-out checks.
- A bounded guarded policy can be promoted separately if it beats the retained policy without promoting the raw predictor.

## What Changed

Added `assumption_os/full_v3_world_model_calibration.py`.

Follow-up repair:

- `assumption_os/full_v3_phase10_discrete_world_model_selector.py` now emits a `calibrated_residual_guard`.
- `assumption_os/full_v3_phase5_contextual_bandit_scheduler.py` can select that guarded policy as production.
- The raw Phase10 predictor remains an exploration candidate because its scalar reward calibration still does not beat base rate.

The audit reads committed, redacted artifacts only:

- `full_v3_phase8_creativity_world_coverage_20260611.json`
- `full_v3_phase9_v1_live_regression_20260611.json`
- `full_v3_phase10_discrete_world_model_selector_20260611.json`
- `full_v3_phase5_contextual_bandit_scheduler_20260611.json`

It creates five calibration surfaces:

1. `phase8_quality_profile_world_model`
2. `phase9_leave_domain_out_nonregression`
3. `phase10_discrete_graph_action_world_model`
4. `phase10_calibrated_residual_guard`
5. `phase5_scheduler_promotion_gate`

## Performance Validation

Generated artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_world_model_calibration_20260611.json`

Key metrics:

- `source_artifact_count`: 4
- `calibration_surface_count`: 5
- `calibrated_surface_count`: 3
- `leave_domain_out_surface_count`: 1
- `uncalibrated_promotion_count`: 0
- `phase8_quality_brier_improvement`: 0.1066
- `phase9_leave_domain_out_domain_count`: 3
- `phase9_leave_domain_out_nonnegative_domain_count`: 2
- `phase9_leave_domain_out_max_calibration_error`: 0.3611
- `phase10_all_lift_over_v3`: 0.0185
- `phase10_calibration_beats_base_rate`: false
- `phase10_selected_arm_mae_minus_base_rate`: 0.0098
- `phase10_calibrated_policy_vs_v1_utility`: 0.6667
- `phase10_calibrated_policy_vs_original_v3_utility`: 0.6204
- `phase10_calibrated_policy_lift_over_v3`: 0.0741
- `phase10_calibrated_policy_lift_over_raw_world_model`: 0.0556
- `phase10_calibrated_policy_lift_over_retained_hybrid`: 0.0186
- `phase10_calibrated_policy_vs_original_v3_lift_over_hybrid`: 0.0093
- `phase10_calibrated_policy_harm_vs_hybrid_count`: 0
- `phase10_calibrated_policy_override_count`: 9
- `phase5_selected_production_profile`: `phase10_calibrated_residual_guard`
- `phase5_keeps_phase10_candidate`: true

Interpretation:

- Phase8 profile-level world model is calibrated enough for scheduler gating.
- Phase9 leave-domain-out is available and records a real boundary: business-domain negative transfer.
- Raw Phase10 discrete graph-action prediction has positive utility lift, but scalar reward calibration is worse than base-rate prediction.
- The calibrated residual guard repairs the raw selector's mistakes and beats the retained Phase9 hybrid on the same heldout slice.
- Therefore the raw Phase10 predictor remains an exploration candidate, while the bounded calibrated residual guard is promoted as the production profile.

## Integrated Gates

Phase11 capability audit now includes:

- `phase10_world_model_calibration`
- status: `calibration_audit_promotes_guard_blocks_raw_predictor`

Paper-scale evidence now includes:

- required artifact count: 28
- v3 mechanism count: 12
- world-model calibration surface metrics
- explicit gates blocking raw uncalibrated Phase10 promotion
- explicit gates promoting only the calibrated residual guard

## Validation Commands

```bash
python3 -m assumption_os.full_v3_world_model_calibration --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_world_model_calibration_20260611.json"
python3 -m assumption_os.full_v3_phase5_contextual_bandit_scheduler --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase5_contextual_bandit_scheduler_20260611.json"
python3 -m assumption_os.full_v3_phase11_capability_audit --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase11_capability_audit_20260611.json"
python3 -m assumption_os.full_v3_paper_scale_evidence --root . --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json"
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase5_contextual_bandit_scheduler_learns_policy_bundle tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase10_discrete_world_model_selector_beats_original_v3 tests.test_assumption_os.AssumptionOSTest.test_full_v3_world_model_calibration_blocks_uncalibrated_promotion tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase11_capability_audit_separates_fixture_from_production tests.test_assumption_os.AssumptionOSTest.test_full_v3_paper_scale_evidence_aggregates_live_and_mechanism_artifacts
```

Targeted test result: 5 tests OK.

## Current Claim

The world model is no longer only an unqualified positive artifact. It is now a calibrated promotion system:

- calibrated surfaces can support gating;
- uncalibrated but positive raw predictors remain candidates;
- guarded policies can be promoted only when they beat the retained policy without harm;
- leave-domain-out failures become explicit boundaries;
- production promotion is blocked for the raw predictor until calibration and domain robustness improve.
