# Last Three Part Execution Plan

Source plan: `reconstruction/md/last_three_part.md`.

## Claim Ladder

- L1 bounded mechanism: fixed inputs, fixed budget, fixed queue, fixed validation protocol.
- L2 robust bounded system: repeated cycles, replay, checkpoint, recovery, negative controls, fault injection.
- L3 production candidate: default-enabled only in restricted real flows with gated apply.
- L4 unbounded/general claim: 24/7 autonomy, broad cross-domain replacement of validation, or full theorem proving.

Current target is L2 -> L3, not L4.

## Track Order

1. Track A: bounded autonomy envelope -> supervised production autonomy.
2. Track B: calibrated simulator candidate -> graph-action simulator for triage/routing.
3. Track C: finite category proof engine -> bounded formal reasoning stack.
4. Integrated mini-loop: connect A/B/C into replayable recursive self-evolution episodes.

## Immediate Tickets

| Ticket | Module | Status | Validation |
| --- | --- | --- | --- |
| A1 | `assumption_os/autonomy_journal.py` | implemented | `tests/test_autonomy_journal.py`, `autonomy_journal_replay_20260612.json` |
| A2 | `assumption_os/autonomy_queue.py` | implemented | `tests/test_autonomy_queue.py`, `autonomy_queue_lease_20260612.json` |
| A3 | `assumption_os/autonomy_recovery_hardening.py` | implemented | `tests/test_autonomy_recovery_hardening.py`, `autonomy_recovery_hardening_20260612.json` |
| B1 | `assumption_os/simulator_transition_schema.py` | implemented | `tests/test_simulator_transition_schema.py`, `simulator_transition_schema_validation_20260612.json` |
| B2 | `assumption_os/simulator_eval_splits.py` | implemented | `tests/test_simulator_eval_splits.py`, `simulator_eval_splits_20260612.json` |
| B3 | `assumption_os/simulator_uncertainty.py` | implemented | `tests/test_simulator_uncertainty.py`, `simulator_uncertainty_20260612.json` |
| B4 | `assumption_os/simulator_counterfactual_policy_eval.py` | repaired | `tests/test_simulator_counterfactual_policy_eval.py`, `simulator_counterfactual_policy_eval_20260612.json` |
| B5/B6 | `assumption_os/simulator_gate_calibration_loop.py` | implemented | `tests/test_simulator_gate_calibration_loop.py`, `simulator_gate_calibration_loop_20260612.json` |
| C1 | `assumption_os/finite_category_certificate.py` | implemented | `tests/test_finite_category_certificate.py`, `finite_category_certificate_20260612.json` |
| C2 | `assumption_os/finite_category_lean_export.py` | implemented | `tests/test_finite_category_lean_export.py`, `finite_category_lean_export_20260612.json` |
| I1 | `assumption_os/integrated_recursive_episode.py` | implemented | `tests/test_integrated_recursive_episode.py`, `integrated_recursive_episode_20260612.json` |
| I2 | `assumption_os/integrated_recursive_episode_b3_c2.py` | implemented | `tests/test_integrated_recursive_episode_b3_c2.py`, `integrated_recursive_episode_b3_c2_20260612.json` |

## A1 Completion Snapshot

- Append-only JSONL journal.
- `event_id`, `cycle_id`, `idempotency_key`, graph before/after hashes.
- Deterministic replay.
- Duplicate event no-op.
- Idempotency conflict blocked.
- Crash-mid-cycle recovery replay.
- Graph hash divergence detection.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/autonomy_journal_replay_20260612.json`

Metrics:

- `replay_same_journal_same_state=true`
- `duplicate_event_no_double_apply=true`
- `crash_mid_cycle_recoverable=true`
- `graph_hash_divergence_detected=true`

## Next Step

Next step is to collect more true same-state multi-arm rows, then rerun B4 until coverage is high enough for production promotion consideration.

## A2 Completion Snapshot

- Lease-based checkpoint queue.
- Task states: `pending`, `leased`, `completed`, `failed`, `deferred`, `blocked`, `expired`.
- Worker lease ownership and TTL.
- Retry-bounded requeue after crash/timeout.
- Terminal expiry after retry budget is exhausted.
- Completed task idempotency.
- Blocked task isolation: timeout processing cannot auto-unblock it.
- Atomic JSON checkpoint reload.
- Optional A1 journal writeback for every mutating queue operation.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/autonomy_queue_lease_20260612.json`

Metrics:

- `double_lease_blocked_for_original_task=true`
- `worker_crash_releases_lease=true`
- `expired_task_requeues=true`
- `same_task_not_executed_twice=true`
- `retry_limit_expires_terminal=true`
- `blocked_task_not_auto_unblocked=true`
- `checkpoint_reload_same_state=true`
- `journal_replay_divergence_detected=false`

## A3 Completion Snapshot

- Fault-injects the bounded autonomy queue/journal substrate.
- Faults covered:
  - `kill_after_queue_read`
  - `kill_after_candidate_preflight`
  - `kill_after_acceptance`
  - `kill_during_apply`
  - `corrupt_one_artifact`
  - `missing_judgment_bundle`
  - `world_model_returns_nan`
- Allowed resolutions:
  - `recover`
  - `defer`
  - `rollback`
  - `manual_review_required`
- Validates that failure handling does not create ungated graph mutation, orphan manifests, dangling candidates, or replay divergence.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/autonomy_recovery_hardening_20260612.json`

Metrics:

- `fault_count=7`
- `resolved_fault_count=7`
- `recover_count=2`
- `defer_count=1`
- `rollback_count=2`
- `manual_review_required_count=2`
- `rollback_success_rate=1.0`
- `ungated_mutation_count=0`
- `orphan_manifest_count=0`
- `dangling_candidate_count=0`
- `replay_divergence_count=0`
- `allowed_resolution_coverage=1.0`

## B1 Completion Snapshot

- Frozen simulator transition schema `simulator_transition_schema_v0`.
- Materialized the current first-party transition-like rows into JSONL while preserving the Phase13 345-row floor.
- Sources:
  - Phase9 V1 same-batch toggle multi-arm rows: 155
  - Phase9 compact-frame same-batch arm rows: 31
  - Phase10 reliability observed-arm rows: 51
  - residual fresh live judgments: 18
  - live multigeneration transition-like rows: 36
  - blinded recursive live judgments: 240
- Required row sections: `state`, `action`, `prediction`, `outcome`, `provenance`.
- Split labels: `train`, `validation`, `test`.
- Provenance hash on every row.
- Redaction check blocks prompt/answer/secret payloads.
- Invalid rows are written to quarantine.

Artifacts:

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_schema_v0.json`

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_dataset_v0.jsonl`

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_quarantine_v0.jsonl`

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_schema_validation_20260612.json`

Metrics:

- `expected_transition_row_count=345`
- `raw_row_count=531`
- `added_transition_row_count_over_phase13=186`
- `valid_row_count=531`
- `invalid_row_count=0`
- `quarantine_row_count=0`
- `redacted_row_count=531`
- `split_counts={"train":393,"validation":60,"test":78}`
- `provenance_hash_unique=true`
- `secret_or_prompt_payload_detected=false`

## B2 Completion Snapshot

- Evaluates the frozen 531-row transition dataset under:
  - leave-one-out
  - leave-domain-out
  - leave-pattern-out
  - leave-artifact-out
  - leave-residual-family-out
- Reports Brier, ECE, abstention rate, true-positive block rate, and false-positive block rate.
- Baselines:
  - feature-similarity simulator candidate
  - base-rate per arm
  - current cheap heuristic world model
  - handwritten hybrid guard
  - random-with-abstain
  - always-original-v3
  - always-run-ablation
- Decision-derived features are explicitly excluded from the feature model.
- Promotion rule blocks raw/current heuristic promotion if heldout false-positive block or ECE is unsafe.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/simulator_eval_splits_20260612.json`

Metrics:

- `row_count=531`
- `leave_one_out_group_count=531`
- `leave_domain_out_group_count=9`
- `leave_pattern_out_group_count=10`
- `leave_artifact_out_group_count=6`
- `leave_residual_family_out_group_count=17`
- `feature_model_loo_brier=0.2007`
- `base_rate_loo_brier=0.2162`
- `current_heuristic_loo_brier=0.142`
- `feature_model_leave_pattern_brier=0.201`
- `base_rate_leave_pattern_brier=0.2186`
- `raw_predictor_promotion_allowed=false`
- `feature_model_promotion_allowed=true`
- `production_simulator_replacement_allowed=false`

## B3 Completion Snapshot

- Adds uncertainty and abstention routing on top of the B2 feature simulator.
- Simulator outputs now include:
  - prediction
  - confidence interval
  - calibration bin
  - abstain reason
  - required verifier tier
- Allowed recommendations:
  - `recommend_run_ablation`
  - `recommend_collect_more_evidence`
  - `recommend_repair_scope`
  - `recommend_reject_low_value`
  - `abstain_to_live_validation`
- Forbidden recommendations are structurally blocked:
  - `auto_accept_without_live`
  - `auto_apply_policy_change`
  - `replace_judge`
- Low-support stress probe abstains to live validation.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/simulator_uncertainty_20260612.json`

Metrics:

- `row_count=531`
- `decision_count=531`
- `leave_pattern_base_rate_brier_with_abstain_as_half=0.2186`
- `leave_pattern_uncertainty_brier_with_abstain_as_half=0.2066`
- `leave_pattern_uncertainty_ece=0.1152`
- `leave_pattern_abstention_rate=0.1695`
- `accepted_candidate_block_rate=0.0`
- `forbidden_action_recommended_count=0`
- `allowed_action_coverage=1.0`
- `low_support_probe_abstained=true`
- `production_simulator_replacement_allowed=false`

## B4 Completion Snapshot / Repair

- Adds matched counterfactual policy evaluation over current same-state multi-arm transition rows.
- Initial coarse grouping by domain/pattern/residual gave only 2 groups and a weak selector.
- Repaired grouping now uses problem-level state IDs from `provenance.source_row_id` before `::`.
- Requires at least 3 observed arms per matched group.
- Reports:
  - empirical best-arm utility
  - B3 selected-arm utility
  - always-v3-full utility
  - leave-one-replicate MAE
  - global baseline MAE
  - leave-state-out feature-conditioned selector utility
  - promotion block reasons
- Result after repair:
  - exploration selector is positive versus always-v3-full
  - same-state multi-arm coverage is now above the production coverage floor
  - production counterfactual promotion is still blocked by estimator and B3 best-arm-selection quality

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/simulator_counterfactual_policy_eval_20260612.json`

Metrics:

- `matched_counterfactual_group_count=48`
- `matched_counterfactual_row_count=237`
- `matched_action_coverage=0.4463`
- `min_arm_count_per_matched_group=3`
- `max_arm_count_per_matched_group=6`
- `leave_one_replicate_mae=0.3949`
- `global_baseline_mae=0.3949`
- `counterfactual_mae_beats_global_baseline=false`
- `leave_state_out_feature_policy_coverage=0.875`
- `leave_state_out_feature_policy_mean_utility=0.5589`
- `leave_state_out_feature_policy_v3_full_utility=0.5286`
- `leave_state_out_feature_policy_lift_over_v3=0.0302`
- `leave_state_out_feature_policy_best_arm_agreement=0.2292`
- `b3_best_arm_agreement_rate=0.1875`
- `empirical_best_policy_mean_utility=0.9062`
- `b3_selected_policy_mean_utility=0.5286`
- `always_v3_full_policy_mean_utility=0.5286`
- `production_counterfactual_gate_allowed=false`
- `exploration_counterfactual_audit_passed=true`

## B5/B6 Completion Snapshot

- Limits simulator output to gate/router roles:
  - `S1_budget_triage`
  - `S2_verifier_routing`
  - `S3_policy_selection`
- Explicitly blocks oracle roles:
  - `S4_replace_fresh_ablation`
  - `S5_replace_judge`
  - `S6_simulate_arbitrary_real_world_outcome`
- Consumes I2 fresh readback and writes calibration rows.
- Emits `SIMULATOR_DEFECT` residuals for high-confidence wrong routing predictions.
- Promotion event:
  - raw simulator remains unpromoted
  - gate/router profile can be promoted within S1/S2 scope
  - no main graph mutation

Artifacts:

`phase four/assumption_graph/paper_readiness_20260604/simulator_gate_calibration_loop_20260612.json`

`phase four/assumption_graph/paper_readiness_20260604/simulator_gate_calibration_writeback_20260612.jsonl`

Metrics:

- `routing_policy_count=5`
- `allowed_routing_level_count=2`
- `forbidden_oracle_level_count=0`
- `forbidden_action_count=0`
- `writeback_row_count=8`
- `fresh_writeback_row_count=6`
- `deferred_writeback_row_count=2`
- `accepted_writeback_row_count=4`
- `rejected_writeback_row_count=2`
- `high_confidence_wrong_count=2`
- `simulator_defect_residual_count=2`
- `raw_simulator_promoted=false`
- `gate_router_promoted=true`
- `main_graph_mutation_count=0`

## C1 Completion Snapshot

- Converts 16 proof-lite formal mappings into finite category certificates.
- Accepted mappings output `allow`; rejected mappings output `block_unsafe_mapping`.
- Each certificate records:
  - objects
  - morphisms
  - explicit composition table
  - functor object/morphism maps
  - naturality square
  - proof obligations
  - negative controls
  - scope conditions
  - not-claimed boundaries
- Validated obligations:
  - identity
  - composition closure
  - associativity
  - functor preserves identity
  - functor preserves composition
  - naturality square
  - diagram commutativity
  - negative-control rejection

Artifacts:

`phase four/assumption_graph/paper_readiness_20260604/finite_category_certificate_20260612.json`

`phase four/assumption_graph/paper_readiness_20260604/finite_category_proof_engine_v0.json`

Metrics:

- `certificate_count=16`
- `valid_certificate_count=16`
- `accepted_certificate_count=9`
- `blocked_certificate_count=7`
- `proof_obligation_pass_rate=1.0`
- `identity_law_pass_rate=1.0`
- `composition_closure_pass_rate=1.0`
- `associativity_pass_rate=1.0`
- `functor_identity_pass_rate=1.0`
- `functor_composition_pass_rate=1.0`
- `naturality_square_pass_rate=1.0`
- `negative_control_pass_rate=1.0`
- `unbounded_theorem_prover_claim_allowed=false`

## C2 Completion Snapshot

- Exports C1 finite category certificates into a Lean-readable text artifact.
- Keeps the formal layer gate-only:
  - `allow`
  - `repair_before_promotion`
  - `block_unsafe_mapping`
  - `not_applicable`
- Explicitly excludes generator / production mutation actions:
  - `generate_new_hypothesis`
  - `synthesize_philosophical_rule`
  - `auto_accept_without_live`
  - `auto_apply_policy_change`
  - `replace_judge`
- Includes expected proof obligations and not-claimed boundaries in the Lean text.
- External Lean syntax check is available and passed in the current environment.

Artifacts:

`phase four/assumption_graph/paper_readiness_20260604/finite_category_certificate_20260612.lean`

`phase four/assumption_graph/paper_readiness_20260604/finite_category_lean_export_20260612.json`

Metrics:

- `certificate_count=16`
- `lean_definition_count=16`
- `proof_obligation_name_count=9`
- `supported_gate_output_count=4`
- `forbidden_generator_output_count=0`
- `not_claimed_boundary_count=6`
- `lean_text_line_count=632`
- `external_lean_available=true`
- `external_lean_check_passed=true`
- `full_theorem_prover_claim_allowed=false`

## I1 Completion Snapshot

- Bounded 10-cycle integrated recursive episode.
- Inputs:
  - frozen 531-row simulator transition dataset
  - B2 simulator split report
  - C1 finite category certificates
  - A1 append-only journal
  - A2 lease-based autonomy queue
- Episode shape:
  - 3 residual clusters
  - 9 candidate proposals
  - top 3 simulator-selected candidates
  - finite formal gate allow/block/not-applicable handling
  - fresh judgment readback from frozen evidence
  - accepted/rejected retention split
  - accepted candidate recheck
  - queue lease execution
  - journal replay
  - simulator calibration row-count update
- Graph mutation remains copy-only; main graph is not mutated.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/integrated_recursive_episode_20260612.json`

Metrics:

- `residual_cluster_count=3`
- `candidate_proposal_count=9`
- `contract_invalid_admitted_count=0`
- `simulator_selected_count=3`
- `simulator_true_positive_block_count=0`
- `fresh_ablation_accept_count=2`
- `fresh_ablation_reject_count=1`
- `accepted_candidate_survival_on_recheck=true`
- `queue_cycle_count=10`
- `autonomy_replay_exact=true`
- `graph_copy_mutation_count=2`
- `main_graph_mutation_count=0`
- `world_model_calibration_row_count_delta=4`

## I2 Completion Snapshot

- Integrates B3 uncertainty/abstention routing and C2 Lean-checkable formal certificates into a bounded recursive episode.
- Uses the real B3 artifact rather than hand-written route decisions.
- Uses the real C2 artifact and requires external Lean check pass in this environment.
- Episode branches:
  - B3 `recommend_run_ablation`
  - B3 `abstain_to_live_validation`
  - C2 checked `allow`
  - C2 checked `block_unsafe_mapping`
  - `not_applicable`
  - fresh readback accept/reject
  - accepted candidate recheck
- Abstained candidates are deferred to live validation and are not auto-executed.
- Retention remains copy-only; main graph is not mutated.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/integrated_recursive_episode_b3_c2_20260612.json`

Metrics:

- `candidate_count=9`
- `b3_pass=true`
- `b3_allowed_action_coverage=1.0`
- `b3_forbidden_action_recommended_count=0`
- `b3_uncertainty_brier_beats_base_rate=true`
- `b3_run_ablation_selected_count=7`
- `b3_abstain_selected_count=2`
- `abstained_candidate_auto_execute_count=0`
- `c2_pass=true`
- `c2_external_lean_check_passed=true`
- `c2_forbidden_generator_output_count=0`
- `formal_gate_block_count=1`
- `formal_gate_lean_checked_count=6`
- `fresh_ablation_accept_count=4`
- `fresh_ablation_reject_count=2`
- `accepted_candidate_survival_on_recheck=true`
- `queue_cycle_count=10`
- `autonomy_replay_exact=true`
- `graph_copy_mutation_count=4`
- `main_graph_mutation_count=0`
- `calibration_row_count_delta=7`

## B4 Same-State Multi-Arm Coverage Repair Snapshot

- Added first-party Phase9 same-state multi-arm rows to the frozen simulator transition dataset.
- Source rows:
  - Phase9 V1 live regression toggle-off same-batch judgments.
  - Phase9 compact frame guard same-batch judgments.
- Preserved the Phase13 345-row floor while expanding the current redacted transition dataset.
- B3 now applies a light out-of-fold bin-calibrated score instead of only reporting calibration error.
- B4 feature policy now excludes arm-specific features from state features and only switches away from `v3_full` when training evidence clears a conservative arm-switch margin.

Artifacts:

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_schema_validation_20260612.json`

`phase four/assumption_graph/paper_readiness_20260604/simulator_eval_splits_20260612.json`

`phase four/assumption_graph/paper_readiness_20260604/simulator_uncertainty_20260612.json`

`phase four/assumption_graph/paper_readiness_20260604/simulator_counterfactual_policy_eval_20260612.json`

Metrics:

- `raw_row_count=531`
- `added_transition_row_count_over_phase13=186`
- `valid_row_count=531`
- `matched_counterfactual_group_count=48`
- `matched_counterfactual_row_count=237`
- `matched_action_coverage=0.4463`
- `leave_pattern_uncertainty_ece=0.1152`
- `leave_pattern_uncertainty_brier_with_abstain_as_half=0.2066`
- `leave_pattern_base_rate_brier_with_abstain_as_half=0.2186`
- `leave_state_out_feature_policy_mean_utility=0.5589`
- `leave_state_out_feature_policy_v3_full_utility=0.5286`
- `leave_state_out_feature_policy_lift_over_v3=0.0302`
- `production_counterfactual_gate_allowed=false`

Remaining production blockers:

- `leave_one_replicate_mae_does_not_beat_global_baseline`
- `b3_selector_does_not_agree_with_empirical_best_arm`
