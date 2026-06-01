# Reconstruction Gap Performance Validation: reconstruction_gap_perf_20260601_expanded

Overall: PASS

## Summary

| Gap | Result | Key Metric |
| --- | --- | --- |
| world_model | PASS | labels=16, pre_auc=1.0, brier=0.0081 |
| trajectory_search | PASS | multi_path=0.8, hit=1.0 |
| verifier_stack | PASS | accepted=2, rejected=14, protocols=27/33 |
| recursive_daemon | PASS | applied=2/2 |
| recursive_audit | PASS | score=1.0, issues=0/0 |
| manifest_logger | PASS | events=112, real_logs=12, leak=False |
| harness_observer | PASS | artifacts=4, backfill=0/19, covered=True |
| residual_clusterer | PASS | clusters=6, proposals=2 |
| formal_metrics | PASS | mappings=9, warnings=0 |
| evolution_context | PASS | decision=ready_for_manual_apply->gated_apply_allowed, resp={'pass': 9} |
| memory_surfaces | PASS | types=11->11, edges=11->11 |
| assumption_bench | PASS | score=0.9968, passed=9/9 |

## Details

### world_model

- `label_counts`: {"accept": 2, "reject": 14}
- `matched_label_count`: 16
- `unmatched_label_count`: 0
- `raw_pre_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.7984, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.3051, "auc": 1.0, "labeled_count": 16, "rejected_count": 14, "rejected_mean_probability": 0.4933, "top_ranked": [{"label": "accept", "probability": 0.803, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.7937, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.5108, "proposal_id": "prop_69d3d6dd67c7"}, {"label": "reject", "probability": 0.5089, "proposal_id": "prop_1382d47d213b"}, {"label": "reject", "probability": 0.5089, "proposal_id": "prop_2892408c37de"}]}
- `pre_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.7984, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.7359, "auc": 1.0, "labeled_count": 16, "rejected_count": 14, "rejected_mean_probability": 0.0625, "top_ranked": [{"label": "accept", "probability": 0.803, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.7937, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.0625, "proposal_id": "prop_2ec0255facee"}, {"label": "reject", "probability": 0.0625, "proposal_id": "prop_69d3d6dd67c7"}, {"label": "reject", "probability": 0.0625, "proposal_id": "prop_54db59587ab9"}]}
- `post_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.8333, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.7765, "auc": 1.0, "labeled_count": 16, "rejected_count": 14, "rejected_mean_probability": 0.0568, "top_ranked": [{"label": "accept", "probability": 0.8333, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.8333, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.1667, "proposal_id": "prop_dfa8c5b146f9"}, {"label": "reject", "probability": 0.1667, "proposal_id": "prop_66a126a35878"}, {"label": "reject", "probability": 0.0385, "proposal_id": "prop_2ec0255facee"}]}
- `post_calibration`: {"brier_score": 0.0081, "labeled_predictions": 16, "mean_absolute_error": 0.0705}
- `trained_calibration`: {"calibrated_metrics": {"brier_score": 0.0085, "labeled_count": 16, "mean_absolute_error": 0.0799}, "decision_probabilities": {"accept": 0.8333, "insufficient_judgments": 0.5, "reject_benefit": 0.0385, "reject_harm": 0.1667}, "eval_id": "perf_world_calibration", "high_priority_accept_floor": 0.75, "label_counts": {"accept": 2, "reject": 14}, "labeled_count": 16, "leave_one_out_calibrated_metrics": {"brier_score": 0.009, "labeled_count": 16, "mean_absolute_error": 0.0836}, "low_priority_probability_cap": 0.0625, "matched_label_count": 16, "priority_boundary": 1.485, "raw_metrics": {"brier_score": 0.2182, "labeled_count": 16, "mean_absolute_error": 0.4568}, "source_acceptance_eval_id": "perf_combined_acceptance", "source_prediction_eval_id": "perf_world_pre_acceptance_raw", "status": "trained", "unmatched_label_count": 0}
- `prediction_count`: 33
- note: pre_acceptance excludes candidate acceptance labels to avoid leakage
- note: post_acceptance validates calibration/logging after real judgments are attached

### trajectory_search

- `frontier_actions`: 10
- `trajectory_count`: 26
- `proposal_count`: 10
- `multi_path_rate`: 0.8
- `top_path_label_hit_rate`: 1.0
- `path_type_counts`: {"evidence_first": 8, "promote_after_verification": 2, "reject_and_synthesize": 8, "repair_then_retest": 8}
- `selected_path_types`: {"evidence_first": 8, "promote_after_verification": 2, "reject_and_synthesize": 8, "repair_then_retest": 2}

### verifier_stack

- `proposal_count`: 33
- `verdict_counts`: {"accepted_for_gated_apply": 2, "collect_more_evidence": 11, "needs_preflight_repair": 6, "rejected_control_harm": 2, "rejected_weak_benefit": 12}
- `confidence_counts`: {"high": 16, "low": 11, "medium": 6}
- `next_action_counts`: {"apply_accepted_candidate_if_requested": 2, "collect_more_evidence": 11, "collect_more_trigger_rows": 6, "reject_or_narrow_scope": 2, "reject_or_revise_candidate": 12}
- `accepted_count`: 2
- `rejected_count`: 14
- `stage_status_counts`: {"V0:defer": 6, "V0:pass": 21, "V0:repair": 6, "V1:defer": 31, "V1:pass": 2, "V2:not_applicable": 33, "V3:block": 6, "V3:defer": 6, "V3:fail": 14, "V3:pass": 7, "V4:fail": 14, "V4:missing": 17, "V4:pass": 2}
- `falsification_experiment_count`: 135
- `falsification_protocol_candidate_count`: 27
- `falsification_experiment_status_counts`: {"blocked": 24, "failed": 34, "passed": 41, "planned": 36}
- `falsification_experiment_name_counts`: {"control_harm_sequential": 27, "fresh_cross_judge_replay": 27, "placebo_context_control": 27, "route_power_and_scope_probe": 27, "trigger_benefit_sequential": 27}
- `accepted_protocol_ok`: true
- `rejected_protocol_ok`: true

### recursive_daemon

- `case_count`: 2
- `accepted_apply_count`: 2
- `results`: [{"accepted_counts": {"accept": 1}, "applied_candidate_node_ids": ["cand_e61e596b7f98"], "applied_nodes_present": true, "case": "ms_bridge", "dry_applied_count": 0, "dry_mutated": false, "manifest_count": 2}, {"accepted_counts": {"accept": 1}, "applied_candidate_node_ids": ["cand_50e44c655f61"], "applied_nodes_present": true, "case": "se_hard_policy", "dry_applied_count": 0, "dry_mutated": false, "manifest_count": 2}]

### recursive_audit

- `case_count`: 2
- `frame_count`: 12
- `actionable_count`: 5
- `critical_issue_count`: 0
- `warning_issue_count`: 0
- `min_closure_score`: 1.0
- `case_summaries`: [{"actionable_count": 4, "closure_score": 1.0, "eval_id": "perf_recursive_audit_dry", "frame_count": 9, "issue_counts": {}, "pass": true}, {"actionable_count": 1, "closure_score": 1.0, "eval_id": "perf_recursive_audit_accepted", "frame_count": 3, "issue_counts": {}, "pass": true}]

### manifest_logger

- `event_count`: 112
- `synthetic_event_count`: 100
- `real_log_event_count`: 12
- `real_log_paths`: ["phase four/assumption_graph/recursive_scoped_judge_run_gpt55_21_50.log", "phase four/assumption_graph/recursive_scoped_ablation_run_gpt55_21_50.log", "phase four/assumption_graph/candidate_ablation_run_phase2_v20_gpt54mini_21_50.log", "phase four/assumption_graph/candidate_ablation_run_phase2_v20_gpt55_21_50.log", "phase six/autonomous/exp80_run.log"]
- `written_trials`: 112
- `secret_leak_detected`: false
- `throughput_events_per_sec`: 1475.63
- `event_counts`: {"judge_call": 28, "llm_call": 21, "retrieval": 20, "simulator_rollout": 20, "tool_use": 23}

### harness_observer

- `artifact_file_count`: 4
- `discovered_event_count`: 19
- `backfilled_event_count`: 0
- `skipped_covered_event_count`: 19
- `event_counts`: {}
- `discovered_event_counts`: {"judge_call": 17, "llm_call": 2}
- `artifact_kind_counts`: {"answer_meta_json": 1, "judgment_json": 9, "run_log": 9}
- `post_covered_file_count`: 4
- `uncovered_after_writeback`: []
- `full_coverage_after_writeback`: true
- `secret_leak_detected`: false
- `artifact_paths`: ["phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ms_bridge_gpt55_21_50.json", "phase two/analysis/cache/answers/phase2_v20_ms_bridge_gpt55_21_50_meta.json", "phase four/assumption_graph/recursive_scoped_judge_run_gpt55_21_50.log", "phase six/autonomous/exp80_run.log"]

### residual_clusterer

- `record_count`: 109
- `cluster_count`: 6
- `proposal_count`: 2
- `residual_type_counts`: {"memory_defect": 8, "optimization": 40, "unknown": 61}
- `proposal_parent_ids`: ["strategy_S08", "strategy_S21"]
- `validation_plans_complete`: true

### formal_metrics

- `mapping_count`: 9
- `complete_count`: 9
- `same_shape_count`: 9
- `warning_count`: 0
- `metric_summary`: {"mean_blackwell_dominance_proxy": 1.0, "mean_frobenius_distance": 0.424409, "mean_total_variation": 0.142899}

### evolution_context

- `responsibility_count`: 9
- `responsibility_status_counts`: {"pass": 9}
- `dry_policy_decision`: "ready_for_manual_apply"
- `apply_policy_decision`: "gated_apply_allowed"
- `blocked_policy_decision`: "blocked_by_permissions"
- `blocked_violation_count`: 2
- `accepted_candidate_count`: 2
- `actionable_frontier_count`: 5
- `procedure_update_count`: 4
- `procedure_update_ids`: ["require_verifier_stack_before_apply", "require_recursive_audit_before_daemon_apply", "require_manifest_and_harness_coverage", "manual_apply_available"]

### memory_surfaces

- `surface_count`: 10
- `edge_count`: 16
- `new_node_count`: 0
- `new_edge_count`: 0
- `before_node_type_count`: 11
- `after_node_type_count`: 11
- `before_edge_type_count`: 11
- `after_edge_type_count`: 11
- `node_type_counts`: {"alignment": 1, "case": 150, "evaluator": 2, "harness": 49, "memory": 1, "method": 111, "residual": 85, "retrieval": 1, "self_modification": 1, "verifier": 1, "world_model": 1}
- `edge_type_counts`: {"depends_on": 3, "derived_from": 46, "failed_because": 192, "generalizes": 1, "has_case": 150, "has_residual": 61, "has_verifier": 5, "is_formal_isomorphism_of": 1, "specializes": 3, "supports": 3, "uses_evaluator": 1}

### assumption_bench

- `overall_score`: 0.9968
- `min_score`: 0.9716
- `capability_count`: 9
- `passed_capability_count`: 9
- `failed_capabilities`: []
- `score_by_capability`: {"assumption_explicitness": 1.0, "context_selection": 1.0, "execution_fidelity": 1.0, "harness_governance": 1.0, "memory_transfer": 1.0, "metaproductivity": 1.0, "residual_attribution": 1.0, "verifier_reliability": 1.0, "world_model_quality": 0.9716}
