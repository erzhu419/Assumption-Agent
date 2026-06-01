# Reconstruction Gap Performance Validation: reconstruction_gap_perf_20260601_expanded

Overall: PASS

## Summary

| Gap | Result | Key Metric |
| --- | --- | --- |
| world_model | PASS | labels=16, pre_auc=1.0, brier=0.0081 |
| trajectory_search | PASS | multi_path=0.8, hit=1.0 |
| recursive_daemon | PASS | applied=2/2 |
| manifest_logger | PASS | events=112, real_logs=12, leak=False |
| residual_clusterer | PASS | clusters=5, proposals=2 |
| formal_metrics | PASS | mappings=9, warnings=0 |

## Details

### world_model

- `label_counts`: {"accept": 2, "reject": 14}
- `matched_label_count`: 16
- `unmatched_label_count`: 0
- `raw_pre_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.8577, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.0814, "auc": 1.0, "labeled_count": 16, "rejected_count": 14, "rejected_mean_probability": 0.7762, "top_ranked": [{"label": "accept", "probability": 0.8612, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.8541, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.8196, "proposal_id": "prop_69d3d6dd67c7"}, {"label": "reject", "probability": 0.8184, "proposal_id": "prop_1382d47d213b"}, {"label": "reject", "probability": 0.8184, "proposal_id": "prop_2892408c37de"}]}
- `pre_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.8577, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.7952, "auc": 1.0, "labeled_count": 16, "rejected_count": 14, "rejected_mean_probability": 0.0625, "top_ranked": [{"label": "accept", "probability": 0.8612, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.8541, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.0625, "proposal_id": "prop_2ec0255facee"}, {"label": "reject", "probability": 0.0625, "proposal_id": "prop_69d3d6dd67c7"}, {"label": "reject", "probability": 0.0625, "proposal_id": "prop_54db59587ab9"}]}
- `post_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.8333, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.7765, "auc": 1.0, "labeled_count": 16, "rejected_count": 14, "rejected_mean_probability": 0.0568, "top_ranked": [{"label": "accept", "probability": 0.8333, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.8333, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.1667, "proposal_id": "prop_dfa8c5b146f9"}, {"label": "reject", "probability": 0.1667, "proposal_id": "prop_66a126a35878"}, {"label": "reject", "probability": 0.0385, "proposal_id": "prop_2ec0255facee"}]}
- `post_calibration`: {"brier_score": 0.0081, "labeled_predictions": 16, "mean_absolute_error": 0.0705}
- `trained_calibration`: {"calibrated_metrics": {"brier_score": 0.006, "labeled_count": 16, "mean_absolute_error": 0.0725}, "decision_probabilities": {"accept": 0.8333, "insufficient_judgments": 0.5, "reject_benefit": 0.0385, "reject_harm": 0.1667}, "eval_id": "perf_world_calibration", "high_priority_accept_floor": 0.75, "label_counts": {"accept": 2, "reject": 14}, "labeled_count": 16, "leave_one_out_calibrated_metrics": {"brier_score": 0.0064, "labeled_count": 16, "mean_absolute_error": 0.0762}, "low_priority_probability_cap": 0.0625, "matched_label_count": 16, "priority_boundary": 1.485, "raw_metrics": {"brier_score": 0.5316, "labeled_count": 16, "mean_absolute_error": 0.697}, "source_acceptance_eval_id": "perf_combined_acceptance", "source_prediction_eval_id": "perf_world_pre_acceptance_raw", "status": "trained", "unmatched_label_count": 0}
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

### recursive_daemon

- `case_count`: 2
- `accepted_apply_count`: 2
- `results`: [{"accepted_counts": {"accept": 1}, "applied_candidate_node_ids": ["cand_e61e596b7f98"], "applied_nodes_present": true, "case": "ms_bridge", "dry_applied_count": 0, "dry_mutated": false, "manifest_count": 2}, {"accepted_counts": {"accept": 1}, "applied_candidate_node_ids": ["cand_50e44c655f61"], "applied_nodes_present": true, "case": "se_hard_policy", "dry_applied_count": 0, "dry_mutated": false, "manifest_count": 2}]

### manifest_logger

- `event_count`: 112
- `synthetic_event_count`: 100
- `real_log_event_count`: 12
- `real_log_paths`: ["phase four/assumption_graph/recursive_scoped_judge_run_gpt55_21_50.log", "phase four/assumption_graph/recursive_scoped_ablation_run_gpt55_21_50.log", "phase four/assumption_graph/candidate_ablation_run_phase2_v20_gpt54mini_21_50.log", "phase four/assumption_graph/candidate_ablation_run_phase2_v20_gpt55_21_50.log", "phase six/autonomous/exp80_run.log"]
- `written_trials`: 112
- `secret_leak_detected`: false
- `throughput_events_per_sec`: 1043.8
- `event_counts`: {"judge_call": 28, "llm_call": 21, "retrieval": 20, "simulator_rollout": 20, "tool_use": 23}

### residual_clusterer

- `record_count`: 109
- `cluster_count`: 5
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
