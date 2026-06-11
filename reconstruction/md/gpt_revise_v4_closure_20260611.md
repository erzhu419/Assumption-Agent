# GPT Revise V4 Closure - 2026-06-11

## Scope

This note records the five GPT_revise_v4 repair items implemented on `reconstruction-v2`.

The goal was not to add another report-only phase. The repaired path is now:

```text
residual / proposal generation
  -> proposal contract
  -> preflight / acceptance
  -> gated shadow apply
  -> Phase10 calibration readback
  -> Phase5 scheduler / Phase11 claim guard
  -> paper-scale evidence aggregation
```

## 1. Proposal Contract Enforced In Evolution Cycle

`evolution_cycle.py` now builds a proposal contract immediately after merged proposal generation and filters proposals before novelty, formal alignment, preflight, and policy planning.

Key behavior:

- Candidate overlays must pass the production contract before downstream gates.
- Manifest-only proposals remain visible for traceability but do not bypass contract admission.
- Draft/invalid proposals are quarantined and reported in `policy_update_plan`.
- The cycle payload now includes both `raw_proposals` and filtered `proposals`, plus `proposal_contract`.

Validation:

- `test_evolution_cycle_plans_loop_without_mutating_by_default`: pass
- `test_evolution_cycle_autonomous_apply_writes_only_gated_acceptance`: pass

## 2. Phase10 Leave-Pattern / Leave-Route-Tag Calibration

Phase10 now evaluates the raw discrete graph-action world-model under leave-pattern-out and leave-route-tag-out splits.

Latest artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase10_discrete_world_model_selector_20260611.json`

Key metrics:

- `leave_pattern_out_group_count`: 2
- `leave_pattern_out_raw_vs_v3_lift`: -0.0588
- `leave_pattern_out_raw_harm_count`: 1
- `leave_pattern_out_guard_vs_v3_lift`: 0.1176
- `leave_pattern_out_guard_harm_count`: 0
- `leave_route_tag_out_raw_vs_v3_lift`: -0.0588
- `leave_route_tag_out_guard_vs_v3_lift`: 0.1176
- `leave_route_tag_out_guard_harm_count`: 0

Interpretation:

The raw predictor remains useful but unsafe as a production default. The calibrated residual guard catches the cross-pattern/route boundary and removes observed harm, so raw Phase10 remains an exploration candidate while the bounded guard is the promoted policy.

## 3. Guard Rules Materialized As Graph Assumption Nodes

Phase10 no longer treats calibrated residual guard rules as invisible if/else logic only. Each guard rule is emitted as a machine-readable assumption node with trigger, expected effect, risk, support rows, negative controls, and validation payload.

Key metrics:

- `guard_assumption_node_count`: 7
- `guard_assumption_active_count`: 6
- `guard_assumption_candidate_count`: 1
- `calibrated_rows_with_guard_assumption_rate`: 1.0

Interpretation:

The current policy is still not a fully learned simulator, but the hand-calibrated pieces are now assumption graph objects that can be audited, revised, demoted, or learned over later.

## 4. Residual Multi-Generation Live Mini-Loop

Added `full_v3_residual_live_mini_loop.py`.

It takes retained descendants from the residual multigeneration loop, builds contract-checked candidate proposals, creates preflight-ready trigger/control rows, runs the real acceptance path on replayed scoped judgments, applies accepted candidates to a graph copy, and reruns Phase10 readback.

Latest artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_residual_live_mini_loop_20260611.json`

Key metrics:

- `source_generation_count`: 3
- `source_retained_count`: 35
- `selected_candidate_count`: 3
- `contract_ready_count`: 3
- `preflight_ready_count`: 3
- `accepted_count`: 3
- `applied_count`: 3
- `graph_copy_node_delta`: 3
- `main_graph_mutation_count`: 0
- `new_api_call_count`: 0
- `phase10_readback_pass`: true
- `phase10_leave_pattern_guard_harm_count`: 0
- `phase10_leave_route_guard_harm_count`: 0

Interpretation:

This is now a real gated mini-loop over the project primitives, not only a dry-run expected-utility loop. It still uses replayed redacted judgments rather than new API calls, so it should be described as live-derived / replayed validation rather than a fresh live experiment.

## 5. Main Graph Memory Consolidation Shadow Pass

Added `full_v3_main_graph_memory_shadow.py`.

It runs the production memory consolidation job over the committed main graph in dry-run mode, copies the graph to a temporary store, applies the consolidation plan only to that copy, and measures retrieval before/after on a frozen family-query suite.

Latest artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_main_graph_memory_shadow_20260611.json`

Key metrics:

- `main_graph_node_count`: 403
- `dry_run_group_count`: 8
- `dry_run_planned_archive_count`: 40
- `dry_run_planned_consolidated_node_count`: 8
- `main_graph_mutated`: false
- `shadow_applied_archived_node_count`: 40
- `shadow_applied_consolidated_node_count`: 8
- `precision_before`: 0.0750
- `precision_after`: 0.2562
- `precision_delta`: 0.1812
- `archive_exposure_after`: 0
- `memory_hit_delta`: 13
- `context_efficiency_delta`: 0.0472

Interpretation:

Main-graph sleep/consolidation is now validated as a shadow pass on the real graph. It should still require explicit apply before mutating the committed graph.

## Paper-Scale Evidence Update

`full_v3_paper_scale_evidence_20260611.json` now includes the two new artifacts and the Phase10 guard-assumption / leave-group metrics.

Key aggregate metrics:

- `required_artifact_count`: 35
- `required_artifact_pass_rate`: 1.0
- `v3_mechanism_count`: 19
- `v3_mechanism_pass_rate`: 1.0
- `residual_live_mini_accepted_count`: 3
- `phase10_leave_pattern_guard_harm_count`: 0
- `phase10_leave_route_tag_guard_harm_count`: 0
- `phase10_guard_assumption_node_count`: 7
- `main_graph_memory_shadow_precision_delta`: 0.1812
- `main_graph_memory_shadow_main_graph_mutated`: false

## Validation

Targeted validation passed:

```text
python3 -m unittest \
  tests.test_assumption_os.AssumptionOSTest.test_evolution_cycle_plans_loop_without_mutating_by_default \
  tests.test_assumption_os.AssumptionOSTest.test_evolution_cycle_autonomous_apply_writes_only_gated_acceptance \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase10_discrete_world_model_selector_beats_original_v3 \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase5_contextual_bandit_scheduler_learns_policy_bundle \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_world_model_calibration_blocks_uncalibrated_promotion \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_residual_live_mini_loop_applies_accepted_descendants_to_graph_copy \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_main_graph_memory_shadow_improves_shadow_retrieval \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_paper_scale_evidence_aggregates_live_and_mechanism_artifacts
```

Result:

```text
Ran 8 tests in 1.902s
OK
```

Full unittest passed:

```text
python3 -m unittest tests.test_assumption_os
Ran 156 tests in 198.632s
OK
```

Project-level performance validation passed:

```text
python3 -m assumption_os.performance_validation
overall_pass: true
assumption_bench.overall_score: 0.9968
assumption_bench.world_model_quality: 0.9716
reconstruction_progress.structure_percent: 86.4
reconstruction_progress.behavior_percent: 78.1
```

## Remaining Claim Boundary

The repaired system is now stronger than the previous V3.5 state on the five review points, but it is still not a fully autonomous hypothesis OS:

- Raw Phase10 is still an exploration candidate, not a production simulator.
- The live mini-loop uses replayed redacted judgments, not new fresh API calls.
- Main graph memory consolidation is a shadow pass, not committed apply.
- Daemon execution/apply gates remain opt-in by design.
