# GPT Revise V4 Claim-Gap Closure - 2026-06-11

This note records the second V4 closure pass after the explicit claim gaps were listed:

1. raw Phase10 world model not calibrated enough for production simulation;
2. guard rules were graph nodes but not learned policy objects;
3. residual multigeneration loop was replayed, not fresh-live capable;
4. main-graph memory consolidation had only a shadow pass;
5. daemon was bounded/gated, not long-horizon schedulable.

## 1. Phase10 Reliability Calibration

Added:

`assumption_os/full_v3_phase10_reliability_calibration.py`

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_phase10_reliability_calibration_20260611.json`

Key metrics:

- `observed_arm_record_count`: 51
- `raw_mae`: 0.3129
- `base_rate_mae`: 0.3062
- `calibrated_mae`: 0.1839
- `calibrated_mae_lift_over_base_rate`: 0.1223
- `calibrated_brier_lift_over_base_rate`: 0.0539
- `raw_ece`: 0.1957
- `calibrated_ece`: 0.1839
- `calibrated_ece_lift_over_raw`: 0.0118

Interpretation:

The uncalibrated raw Phase10 predictor still fails the base-rate test, but an out-of-fold reliability-bin calibrator now beats base-rate on MAE, Brier, and ECE. This promotes the calibrated raw predictor to a budget/search-control candidate, not to a graph-mutation simulator.

## 2. Learned Guard Policy

Added:

`assumption_os/full_v3_guard_policy_learning.py`

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_guard_policy_learning_20260611.json`

Key metrics:

- `learned_guard_update_count`: 7
- `supported_guard_count`: 7
- `guard_weight_range`: 0.3333
- `promote_weight_count`: 1
- `learned_policy_lift_over_hybrid`: 0.0185
- `learned_policy_harm_vs_hybrid_count`: 0
- `raw_world_model_status`: candidate

Interpretation:

Guard assumptions are no longer only static rules. Each guard now receives posterior weight/confidence updates from heldout and leave-group outcomes. Raw world-model selection remains candidate-only.

## 3. Fresh-Live-Capable Residual Loop

Added:

`assumption_os/full_v3_residual_fresh_live_loop.py`

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_residual_fresh_live_loop_20260611.json`

Original dry-run metrics:

- `execution_mode`: dry_run
- `selected_candidate_count`: 3
- `contract_ready_count`: 3
- `preflight_ready_count`: 3
- `fresh_live_path_present`: true
- `planned_fresh_api_call_count`: 18
- `fresh_api_call_count`: 0
- `accepted_count`: 3
- `applied_count`: 3
- `main_graph_mutation_count`: 0

Interpretation:

The fresh API execution path is implemented and preflighted. The committed validation run is dry-run because the local GPT keyfile currently returns an invalid-token response when probed. Therefore this closes the engineering path, but it is not counted as fresh API evidence.

Fresh API update:

After receiving a valid GPT API key, the same artifact was rerun in `execute_live` mode with `gpt-5.4-mini`.

```text
fresh_api_call_count: 18
planned_fresh_api_call_count: 18
selected_candidate_count: 3
accepted_count: 1
acceptance_decision_counts: {reject_harm: 1, accept: 1, reject_benefit: 1}
applied_count: 1
graph_copy_node_delta: 1
main_graph_mutation_count: 0
secret_value_exposed: false
failed_gates: []
```

Interpretation:

This now counts as fresh API evidence for the residual multi-generation loop. The live path generated trigger/control judgments, rejected one harmful candidate, rejected one low-benefit candidate, accepted one candidate, and applied it only to a graph copy. Main graph mutation remains gated.

## 4. Main-Graph Memory Controlled Apply

Added:

`assumption_os/full_v3_main_graph_memory_controlled_apply.py`

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_main_graph_memory_controlled_apply_20260611.json`

Key metrics:

- `planned_archive_count`: 40
- `rollback_entry_count`: 40
- `applied_archived_node_count`: 40
- `applied_consolidated_node_count`: 8
- `precision_delta`: 0.1812
- `archive_exposure_after`: 0
- `context_efficiency_delta`: 0.0472
- `main_graph_mutated`: false

Interpretation:

The main-graph consolidation path now has a rollback manifest and controlled apply/readback. The default validation still applies to a controlled copy. Mutating the committed graph requires explicit `--apply-main`.

## 5. Continuous Daemon Scheduler

Added:

`assumption_os/full_v3_continuous_daemon_scheduler.py`

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_continuous_daemon_scheduler_20260611.json`

Key metrics:

- `scheduled_cycle_count`: 12
- `checkpoint_pair_count`: 12
- `rate_limit_violation_count`: 0
- `recovery_action_count`: 2
- `fresh_loop_queue_integrated`: true
- `memory_apply_queue_integrated`: true
- `daemon_readback_queue_integrated`: true
- `ungated_graph_mutation_count`: 0
- `continuous_background_ready`: true
- `background_process_started`: false

Interpretation:

The daemon is now long-horizon schedulable with queues, budgets, rate limits, checkpoint/recovery, and gated mutation. Validation deliberately does not spawn an uncontrolled background worker.

## Paper-Scale Update

`full_v3_paper_scale_evidence_20260611.json` now includes 40 required artifacts and 24 V3 mechanism artifacts.

Key new aggregate metrics:

- `required_artifact_count`: 40
- `required_artifact_pass_rate`: 1.0
- `v3_mechanism_count`: 24
- `v3_mechanism_pass_rate`: 1.0
- `phase10_reliability_calibrated_mae_lift_over_base`: 0.1223
- `guard_policy_learned_update_count`: 7
- `residual_fresh_planned_api_call_count`: 18
- `main_graph_memory_controlled_apply_rollback_entry_count`: 40
- `continuous_daemon_scheduled_cycle_count`: 12

## Validation

Targeted validation passed:

```text
python3 -m unittest \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase10_reliability_calibration_beats_base_rate \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_guard_policy_learning_updates_guard_assumptions \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_residual_fresh_live_loop_preflights_api_capable_path \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_main_graph_memory_controlled_apply_has_rollback_and_readback \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_continuous_daemon_scheduler_integrates_long_horizon_queues \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_paper_scale_evidence_aggregates_live_and_mechanism_artifacts

Ran 6 tests in 1.194s
OK
```

Full unit validation passed:

```text
python3 -m unittest tests.test_assumption_os

Ran 161 tests in 195.969s
OK
```

Performance validation passed:

```text
python3 -m assumption_os.performance_validation

overall_pass: true
assumption_bench_score: 0.9968
world_model_quality: 0.9716
reconstruction_structure: 86.4
reconstruction_behavior: 78.1
failed_sections: []
```

## Remaining Boundary

The main remaining boundary is no longer fresh API execution for the residual loop. It is controlled autonomy:

- Main graph memory apply is controlled and rollback-ready, but the committed graph has not been mutated in validation.
- The daemon is long-horizon schedulable, but validation does not leave a persistent background process running.
