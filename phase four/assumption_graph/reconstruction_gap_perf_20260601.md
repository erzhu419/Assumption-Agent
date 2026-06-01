# Reconstruction Gap Performance Validation: reconstruction_gap_perf_20260601

Overall: PASS

## Summary

| Gap | Result | Key Metric |
| --- | --- | --- |
| world_model | PASS | pre_auc=1.0, brier=0.0359 |
| trajectory_search | PASS | multi_path=0.8, hit=1.0 |
| recursive_daemon | PASS | applied=2/2 |
| manifest_logger | PASS | throughput=1386.96/s, leak=False |
| residual_clusterer | PASS | clusters=5, proposals=2 |
| formal_metrics | PASS | mappings=9, warnings=0 |

## Details

### world_model

- `label_counts`: {"accept": 2, "reject": 14}
- `pre_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.8577, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.0413, "auc": 1.0, "labeled_count": 10, "rejected_count": 8, "rejected_mean_probability": 0.8163, "top_ranked": [{"label": "accept", "probability": 0.8612, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.8541, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.8196, "proposal_id": "prop_69d3d6dd67c7"}, {"label": "reject", "probability": 0.8184, "proposal_id": "prop_1382d47d213b"}, {"label": "reject", "probability": 0.8184, "proposal_id": "prop_2892408c37de"}]}
- `post_acceptance`: {"accepted_count": 2, "accepted_mean_probability": 0.9451, "accepted_recall_at_k": 1.0, "accepted_rejected_margin": 0.7376, "auc": 1.0, "labeled_count": 10, "rejected_count": 8, "rejected_mean_probability": 0.2075, "top_ranked": [{"label": "accept", "probability": 0.9466, "proposal_id": "prop_e61e596b7f98"}, {"label": "accept", "probability": 0.9436, "proposal_id": "prop_50e44c655f61"}, {"label": "reject", "probability": 0.22, "proposal_id": "prop_2ec0255facee"}, {"label": "reject", "probability": 0.22, "proposal_id": "prop_69d3d6dd67c7"}, {"label": "reject", "probability": 0.22, "proposal_id": "prop_54db59587ab9"}]}
- `post_calibration`: {"brier_score": 0.0359, "labeled_predictions": 10, "mean_absolute_error": 0.177}
- `prediction_count`: 24
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

- `event_count`: 100
- `written_trials`: 100
- `secret_leak_detected`: false
- `throughput_events_per_sec`: 1386.96
- `event_counts`: {"judge_call": 20, "llm_call": 20, "retrieval": 20, "simulator_rollout": 20, "tool_use": 20}

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
