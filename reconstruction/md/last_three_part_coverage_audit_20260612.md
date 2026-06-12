# last_three_part.md Coverage Audit

- pass: `True`
- engineering tickets: `33/33`
- open engineering gaps: `0`
- blocked claim boundaries: `5/5`
- source artifact pass rate: `1.0`

## Ticket Coverage

| Ticket | Status | Evidence | Key metrics |
| --- | --- | --- | --- |
| `0_claim_ladder` | `pass` | last_three_part_final_closure_20260612.json | allowed_claim_count=7; blocked_strong_claim_count=14 |
| `A0_phase13_freeze` | `pass` | full_v3_phase13_general_autonomy_lift_20260612.json | cycle_count=96; ungated_mutation_count=0 |
| `A1_autonomy_journal` | `pass` | autonomy_journal_replay_20260612.json | replay_event_count=4; duplicate_noop_count=1 |
| `A2_lease_queue` | `pass` | autonomy_queue_lease_20260612.json | task_count=4; journal_event_count=12 |
| `A3_recovery_rollback` | `pass` | autonomy_recovery_hardening_20260612.json | fault_count=7; rollback_success_rate=1.0 |
| `A4_shadow_service` | `pass` | autonomy_shadow_service_20260612.json | shadow_day_count=7; recommendation_manifest_count=24 |
| `A5_low_risk_auto_apply` | `pass` | autonomy_shadow_service_20260612.json | low_risk_auto_apply_count=15; forbidden_policy_change_auto_apply_count=0 |
| `A6_supervised_production_candidate` | `pass` | autonomy_supervised_production_run_20260612.json | supervised_day_count=30; auto_apply_count=625 |
| `B0_transition_schema` | `pass` | simulator_transition_schema_validation_20260612.json | valid_row_count=531; split_counts={'test': 78, 'train': 393, 'validation': 60} |
| `B1_split_discipline` | `pass` | simulator_eval_splits_20260612.json | split_eval_count=5; leave_pattern_groups=10 |
| `B2_simulator_baselines` | `pass` | simulator_eval_splits_20260612.json | leave_pattern_brier=0.201; base_rate_leave_pattern_brier=0.2186 |
| `B3_uncertainty_abstain` | `pass` | simulator_uncertainty_20260612.json | abstention_rate=0.1695; ece=0.1152 |
| `B4_counterfactual_policy` | `pass` | simulator_counterfactual_policy_eval_20260612.json, simulator_production_evidence_20260612.json | matched_group_count=48; production_counterfactual_mae=0.004; global_baseline_mae=0.0877; no_leakage_pass=True |
| `B4b_no_leakage_audit` | `pass` | simulator_no_leakage_audit_20260612.json, simulator_production_evidence_20260612.json | state_leaks=0; prediction_identity=0; best_arm_agreement=0.9611 |
| `B5_simulator_as_gate` | `pass` | simulator_gate_calibration_loop_20260612.json | routing_policy_count=5; allowed_routing_level_count=2 |
| `B6_closed_loop_calibration` | `pass` | simulator_gate_calibration_loop_20260612.json | writeback_row_count=8; simulator_defect_residual_count=2 |
| `B7_production_simulator_candidate` | `pass` | simulator_production_gate_20260612.json, simulator_production_evidence_20260612.json | transition_row_count=2160; pattern_count=24 |
| `C0_finite_engine_freeze` | `pass` | finite_category_certificate_20260612.json, finite_category_proof_engine_v0.json | certificate_count=16; not_claimed_count=6 |
| `C1_certificate_schema` | `pass` | finite_category_certificate_20260612.json | proof_obligation_count=144; negative_control_blocked_count=7 |
| `C2_lean_export` | `pass` | finite_category_lean_export_20260612.json, finite_theorem_lean_verifier_20260612.json | lean_definition_count=16; lean_theorem_count=36 |
| `C3_finite_category_dsl` | `pass` | finite_formal_reasoning_stack_20260612.json | dsl_object_count=4; dsl_morphism_count=10 |
| `C4_proof_assistant_check` | `pass` | finite_theorem_lean_verifier_20260612.json, finite_formal_reasoning_stack_20260612.json | external_lean_theorem_count=36; advanced_constructions=True |
| `C5_markov_kernel_extension` | `pass` | finite_formal_reasoning_stack_20260612.json | markov_kernel_count=5; row_stochastic_pass_count=4 |
| `C6_information_geometry_plugin` | `pass` | finite_formal_reasoning_stack_20260612.json | metric_count=6; not_truth_oracle=True |
| `C7_formal_transfer_benchmark` | `pass` | finite_formal_reasoning_stack_20260612.json | pairwise_auc=1.0; overreach_residual_count=1 |
| `C8_claim_gate` | `pass` | finite_formal_reasoning_stack_20260612.json, nl_to_diagram_scale_benchmark_20260612.json | bounded_formal_stack_claim_allowed=True; full_theorem_prover_claim_allowed=False |
| `I1_integrated_recursive_episode` | `pass` | integrated_recursive_episode_20260612.json | residual_cluster_count=3; fresh_ablation_accept_count=2 |
| `I2_integrated_b3_c2_slice` | `pass` | integrated_recursive_episode_b3_c2_20260612.json | b3_abstain_selected_count=2; formal_gate_block_count=1 |
| `P1_paper_main_line` | `pass` | paper_frozen_main_experiment_v2_20260612.json | problem_count=1768; baseline_count=8; margin=0.0417 |
| `P2_creative_generator` | `pass` | creative_hypothesis_trajectory_search_20260612.json | candidate_count=372; retained_count=201 |
| `P3_main_graph_controlled_apply` | `pass` | main_graph_controlled_apply_monitor_20260612.json | monitor_day_count=30; min_precision_delta_vs_before=0.1695 |
| `P4_fresh_frozen_rerun_protocol` | `pass` | paper_fresh_frozen_rerun_protocol_20260612.json, full_v3_blinded_recursive_live_line_20260612.json | target_calls=720; pilot_calls=240; dry_run_calls=720 |
| `P5_fresh_live_720_selective_retention_result` | `pass` | paper_fresh_rerun_result_integration_20260612.json, paper_fresh_frozen_rerun_live_720_20260612.json | fresh_calls=720; accepted_count=4; accepted_trigger_ci95=[0.6667, 0.9583]; accepted_control_ci95=[0.0, 0.0] |

## Blocked Claim Boundaries

| Claim | Blocked | Reason |
| --- | --- | --- |
| `unbounded_24_7_general_autonomous_os` | `True` | Evidence supports supervised bounded autonomy, not unrestricted 24/7 general OS. |
| `raw_world_simulator_replaces_live_validation` | `True` | Simulator is promoted only for triage/routing; raw replacement remains blocked. |
| `complete_category_theory_theorem_prover` | `True` | Lean-verified finite fragment is allowed; arbitrary theorem proving is not. |
| `brand_new_live_api_main_paper_experiment` | `True` | The fresh rerun is now completed, but the raw unfiltered generator did not clear the broad all-candidate trigger gate.  The allowed claim is selective-retention fresh support, not an unqualified live main-paper win. |
| `ungated_default_policy_or_main_graph_mutation` | `True` | Graph and policy mutations remain gated or canary-scoped. |
