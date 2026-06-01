# Reconstruction Gap Performance Validation: reconstruction_gap_perf_20260601_recursive_trace

Overall: PASS

## Summary

| Gap | Result | Key Metric |
| --- | --- | --- |
| world_model | PASS | labels=16, pre_auc=1.0, brier=0.0081 |
| trajectory_search | PASS | multi_path=0.8, hit=1.0 |
| metaproductivity_benchmark | PASS | queries=4, acp_meta=0.6687, imm_meta=0.2225 |
| verifier_stack | PASS | accepted=2, rejected=14, protocols=27/33 |
| recursive_daemon | PASS | applied=2/2 |
| recursive_audit | PASS | score=1.0, issues=0/0 |
| manifest_logger | PASS | events=112, real_logs=12, leak=False |
| runtime_trace | PASS | events=3, written=3, leak=False |
| trace_dataset | PASS | rows=141/161, coverage=1.0, leak=False |
| trace_outcome_model | PASS | rows=141, brier=0.1671, updates=5 |
| trace_policy_proposals | PASS | proposals=5, repair=4, parent=surface_6e7d9d238212 |
| trace_policy_preflight | PASS | ready=5/5, missed=0, outside=0 |
| harness_observer | PASS | artifacts=4, backfill=0/19, covered=True |
| residual_clusterer | PASS | clusters=7, proposals=2 |
| formal_metrics | PASS | mappings=9, warnings=0, dedup=0, transfer_auc=1.0 |
| surface_hypothesis_generator | PASS | proposals=4, world=2, evaluator=2 |
| evolution_context | PASS | decision=ready_for_manual_apply->gated_apply_allowed, resp={'pass': 9} |
| memory_surfaces | PASS | types=11->11, edges=11->11 |
| assumption_bench | PASS | score=0.9968, passed=9/9 |
| reconstruction_progress | PASS | structure=83.2%, behavior=74.2%, weighted=78.3% |

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

### metaproductivity_benchmark

- `query_count`: 4
- `positive_control`: {"acp_top_clade_metaproductivity": 0.6975, "acp_top_id": "productive_parent", "acp_top_immediate_utility": 0.23, "acp_top_score": 1.2442, "expected_acp_top_id": "productive_parent", "expected_immediate_top_id": "quick_win", "immediate_top_clade_metaproductivity": 0.0, "immediate_top_id": "quick_win", "immediate_top_immediate_utility": 0.65, "immediate_top_score": 1.3675, "pass": true, "query": "risk rollback guardrail", "same_top": false}
- `mean_acp_top_clade_metaproductivity`: 0.6687
- `mean_immediate_top_clade_metaproductivity`: 0.2225
- `distinct_acp_top_count`: 2
- `distinct_immediate_top_count`: 4
- `live_probe`: {"distinct_acp_top_count": 2, "distinct_immediate_top_count": 4, "mean_acp_top_clade_metaproductivity": 0.6687, "mean_immediate_top_clade_metaproductivity": 0.2225, "queries": [{"acp_top_clade_metaproductivity": 0.825, "acp_top_id": "surface_f3aeb19c2ffe", "acp_top_immediate_utility": 0.18, "acp_top_score": 0.7164, "immediate_top_clade_metaproductivity": 0.14, "immediate_top_id": "hyp_4f9fb9e6877c", "immediate_top_immediate_utility": 0.7, "immediate_top_score": 0.9533, "query": "risk rollback guardrail", "same_top": false}, {"acp_top_clade_metaproductivity": 0.825, "acp_top_id": "surface_f3aeb19c2ffe", "acp_top_immediate_utility": 0.18, "acp_top_score": 0.7411, "immediate_top_clade_metaproductivity": 0.2, "immediate_top_id": "surface_8db9bb44a936", "immediate_top_immediate_utility": 0.18, "immediate_top_score": 0.7553, "query": "formal mapping verifier transfer", "same_top": false}, {"acp_top_clade_metaproductivity": 0.2, "acp_top_id": "surface_ef3d98589e5c", "acp_top_immediate_utility": 0.18, "acp_top_score": 0.7717, "immediate_top_clade_metaproductivity": 0.2, "immediate_top_id": "surface_ef3d98589e5c", "immediate_top_immediate_utility": 0.18, "immediate_top_score": 0.7923, "query": "world model trace calibration", "same_top": true}, {"acp_top_clade_metaproductivity": 0.825, "acp_top_id": "surface_f3aeb19c2ffe", "acp_top_immediate_utility": 0.18, "acp_top_score": 0.7022, "immediate_top_clade_metaproductivity": 0.35, "immediate_top_id": "surface_8d569750fe58", "immediate_top_immediate_utility": 0.18, "immediate_top_score": 0.4681, "query": "residual cluster evaluator policy", "same_top": false}]}

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
- `throughput_events_per_sec`: 1525.89
- `event_counts`: {"judge_call": 28, "llm_call": 21, "retrieval": 20, "simulator_rollout": 20, "tool_use": 23}

### runtime_trace

- `event_count`: 3
- `event_counts`: {"llm_call": 1, "retrieval": 1, "tool_use": 1}
- `written_trials`: 3
- `events_out_written`: true
- `summary_out_written`: true
- `secret_leak_detected`: false

### trace_dataset

- `row_count`: 161
- `trainable_row_count`: 141
- `weighted_trainable_row_count`: 75.0
- `first_party_trace_count`: 9
- `first_party_trainable_row_count`: 9
- `artifact_replay_count`: 152
- `artifact_replay_trainable_row_count`: 132
- `missing_trace_count`: 0
- `traced_outcome_coverage`: 1.0
- `assumption_id_coverage`: 0.5
- `outcome_counts`: {"loss": 35, "tie": 20, "win": 106}
- `residual_type_counts`: {"no_residual": 106, "optimization": 4, "unknown": 51}
- `event_counts`: {"tool_use": 161}
- `source_eval_ids`: ["trace_dataset_ms_bridge_20260601", "trace_dataset_ms_bridge_ms100_20260601", "trace_dataset_ms_bridge_ms100_vs_v20_20260601", "trace_dataset_recursive_proposal_1382d47d213b_20260601", "trace_dataset_recursive_proposal_1c34e615945c_20260601", "trace_dataset_recursive_proposal_2892408c37de_20260601", "trace_dataset_recursive_proposal_2ec0255facee_20260601", "trace_dataset_recursive_proposal_54db59587ab9_20260601", "trace_dataset_recursive_proposal_69d3d6dd67c7_20260601", "trace_dataset_recursive_proposal_dfa8c5b146f9_20260601", "trace_dataset_recursive_proposal_fb5fc39a8090_20260601"]
- `positive_control`: {"first_party_trace_count": 2, "pass": true, "row_count": 2, "secret_leak_detected": false, "trainable_row_count": 2}
- `secret_leak_detected`: false

### trace_outcome_model

- `trace_dataset_path`: "phase four/assumption_graph/trace_dataset_collection_ms_bridge_20260601.json"
- `collection_mode`: true
- `trainable_row_count`: 141
- `weighted_trainable_row_count`: 75.0
- `trace_source_counts`: {"artifact_replay": 132, "first_party_runtime": 9}
- `trace_source_weighted_counts`: {"artifact_replay": 66.0, "first_party_runtime": 9.0}
- `route_group_count`: 5
- `component_group_count`: 2
- `residual_group_count`: 4
- `policy_update_count`: 5
- `loss_policy_update_count`: 4
- `leave_one_out_metrics`: {"accuracy_at_half": 0.7518, "brier_score": 0.1671, "mean_absolute_error": 0.3586, "prediction_count": 141, "prediction_level_counts": {"route_component": 141}, "weighted_accuracy_at_half": 0.76, "weighted_brier_score": 0.1667, "weighted_mean_absolute_error": 0.3585, "weighted_prediction_count": 75.0}
- `feature_leave_one_out_metrics`: {"accuracy_at_half": 0.9716, "brier_score": 0.0701, "mean_absolute_error": 0.2425, "prediction_count": 141, "prediction_level_counts": {"feature_blend": 141}, "weighted_accuracy_at_half": 0.9667, "weighted_brier_score": 0.0711, "weighted_mean_absolute_error": 0.2411, "weighted_prediction_count": 75.0}
- `best_brier_score`: 0.0711
- `feature_schema`: {"feature_count": 47, "feature_family_counts": {"active_assumption_count": 1, "active_assumption_count_bucket": 1, "baseline_variant": 1, "bypass_route": 4, "component": 2, "component_count": 2, "difficulty": 3, "domain": 6, "event": 1, "frame": 3, "gold_hit": 1, "intervention_variant": 8, "judgment_pair": 8, "residual_type": 3, "trace_event_count": 1, "trace_source": 2}, "top_features": [{"count": 141, "feature": "trace_event_count=1"}, {"count": 141, "feature": "active_assumption_count_bucket=0"}, {"count": 141, "feature": "active_assumption_count=0"}, {"count": 141, "feature": "event=tool_use"}, {"count": 141, "feature": "gold_hit=False"}, {"count": 132, "feature": "trace_source=artifact_replay"}, {"count": 132, "feature": "component_count:artifact_replay_answer_meta=1"}, {"count": 132, "feature": "component=artifact_replay_answer_meta"}, {"count": 106, "feature": "residual_type=no_residual"}, {"count": 84, "feature": "difficulty=medium"}, {"count": 74, "feature": "baseline_variant=phase2_v20_gpt55"}, {"count": 67, "feature": "frame=hybrid"}, {"count": 40, "feature": "frame=object"}, {"count": 36, "feature": "domain=science"}, {"count": 34, "feature": "difficulty=hard"}, {"count": 33, "feature": "domain=mathematics"}, {"count": 32, "feature": "frame=paradigm"}, {"count": 31, "feature": "residual_type=unknown"}, {"count": 30, "feature": "domain=engineering"}, {"count": 26, "feature": "bypass_route=science_mechanism"}]}
- `route_stats`: [{"count": 74, "key": "route=no_route", "loss_count": 31, "mean_score_delta": 0.1486, "problem_ids": ["business_0199", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0057", "software_engineering_0086", "business_0199", "daily_life_0018", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0057", "engineering_0160", "engineering_0197", "business_0199", "business_0218", "daily_life_0018", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0057", "engineering_0160", "business_0199", "business_0218", "daily_life_0133", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0160", "engineering_0197", "business_0199", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0173", "engineering_0056", "engineering_0057", "engineering_0197", "science_0204", "business_0199", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0057", "engineering_0160", "engineering_0197", "software_engineering_0355", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0160", "engineering_0197", "science_0273", "business_0199", "business_0218", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0160"], "residual_type_counts": {"no_residual": 43, "unknown": 31}, "smoothed_win_probability": 0.5789, "trace_source_counts": {"artifact_replay": 74}, "trace_source_weighted_counts": {"artifact_replay": 37.0}, "weighted_count": 37.0, "weighted_loss_count": 15.5, "weighted_mean_score_delta": 0.1486, "weighted_smoothed_win_probability": 0.5769, "weighted_win_count": 21.5, "weighted_win_rate": 0.5811, "win_count": 43, "win_rate": 0.5811}, {"count": 26, "key": "route=science_mechanism", "loss_count": 1, "mean_score_delta": 1.0769, "problem_ids": ["science_0083", "science_0097", "science_0180", "science_0022", "science_0083", "science_0085", "science_0097", "science_0109", "science_0130", "science_0135", "science_0154", "science_0175", "science_0180", "science_0188", "science_0215", "science_0022", "science_0083", "science_0085", "science_0097", "science_0109", "science_0130", "science_0135", "science_0175", "science_0180", "science_0188", "science_0215"], "residual_type_counts": {"no_residual": 25, "optimization": 1}, "smoothed_win_probability": 0.9286, "trace_source_counts": {"artifact_replay": 23, "first_party_runtime": 3}, "trace_source_weighted_counts": {"artifact_replay": 11.5, "first_party_runtime": 3.0}, "weighted_count": 14.5, "weighted_loss_count": 1.0, "weighted_mean_score_delta": 1.0, "weighted_smoothed_win_probability": 0.8788, "weighted_win_count": 13.5, "weighted_win_rate": 0.931, "win_count": 25, "win_rate": 0.9615}, {"count": 19, "key": "route=math_research_bridge", "loss_count": 1, "mean_score_delta": 2.1579, "problem_ids": ["mathematics_0192", "mathematics_0243", "mathematics_0251", "mathematics_0257", "mathematics_0018", "mathematics_0092", "mathematics_0111", "mathematics_0192", "mathematics_0243", "mathematics_0251", "mathematics_0257", "mathematics_0018", "mathematics_0030", "mathematics_0092", "mathematics_0111", "mathematics_0192", "mathematics_0243", "mathematics_0251", "mathematics_0257"], "residual_type_counts": {"no_residual": 18, "optimization": 1}, "smoothed_win_probability": 0.9048, "trace_source_counts": {"artifact_replay": 15, "first_party_runtime": 4}, "trace_source_weighted_counts": {"artifact_replay": 7.5, "first_party_runtime": 4.0}, "weighted_count": 11.5, "weighted_loss_count": 0.5, "weighted_mean_score_delta": 2.0, "weighted_smoothed_win_probability": 0.8889, "weighted_win_count": 11.0, "weighted_win_rate": 0.9565, "win_count": 18, "win_rate": 0.9474}, {"count": 14, "key": "route=math_formal", "loss_count": 2, "mean_score_delta": 1.1429, "problem_ids": ["mathematics_0081", "mathematics_0082", "mathematics_0126", "mathematics_0163", "mathematics_0212", "mathematics_0236", "mathematics_0239", "mathematics_0081", "mathematics_0082", "mathematics_0126", "mathematics_0163", "mathematics_0212", "mathematics_0236", "mathematics_0239"], "residual_type_counts": {"no_residual": 12, "optimization": 2}, "smoothed_win_probability": 0.8125, "trace_source_counts": {"artifact_replay": 14}, "trace_source_weighted_counts": {"artifact_replay": 7.0}, "weighted_count": 7.0, "weighted_loss_count": 1.0, "weighted_mean_score_delta": 1.1429, "weighted_smoothed_win_probability": 0.7778, "weighted_win_count": 6.0, "weighted_win_rate": 0.8571, "win_count": 12, "win_rate": 0.8571}, {"count": 8, "key": "route=science_decision", "loss_count": 0, "mean_score_delta": 1.625, "problem_ids": ["science_0204", "science_0273", "science_0197", "science_0204", "science_0273", "science_0197", "science_0204", "science_0273"], "residual_type_counts": {"no_residual": 8}, "smoothed_win_probability": 0.9, "trace_source_counts": {"artifact_replay": 6, "first_party_runtime": 2}, "trace_source_weighted_counts": {"artifact_replay": 3.0, "first_party_runtime": 2.0}, "weighted_count": 5.0, "weighted_loss_count": 0, "weighted_mean_score_delta": 1.5, "weighted_smoothed_win_probability": 0.8571, "weighted_win_count": 5.0, "weighted_win_rate": 1.0, "win_count": 8, "win_rate": 1.0}]
- `policy_updates`: [{"decision": "keep_with_targeted_repair", "expected_effect": "Reduce route-specific losses while preserving the observed win rate.", "policy_update_id": "trace_policy_15faf31ae514", "residual_groups": [{"bypass_route": "no_route", "components": ["artifact_replay_answer_meta"], "count": 31, "key": "residual=unknown|route=no_route|components=artifact_replay_answer_meta", "problem_ids": ["daily_life_0161", "engineering_0057", "software_engineering_0086", "daily_life_0018", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0057", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "daily_life_0133", "daily_life_0173", "engineering_0004", "engineering_0160", "daily_life_0018", "engineering_0056", "engineering_0057", "engineering_0197", "science_0204", "business_0199", "engineering_0057", "engineering_0197", "daily_life_0018", "engineering_0004", "engineering_0056", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004"], "residual_previews": ["No active assumptions were available for attribution; collect first-party trace before mutating graph. Judge: B在评分框架、硬约束识别和验证实验上更系统，决策可操作性略强...", "No active assumptions were available for attribution; collect first-party trace before mutating graph. Judge: A分层诊断更细，证据采集、隔离测试和闭环验证更具可执行性；B...", "No active assumptions were available for attribution; collect first-party trace before mutating graph. Judge: B更完整覆盖复现环境、good/bad基线、二分细节、干扰控...", "No active assumptions were available for attribution; collect first-party trace before mutating graph. Judge: B对不适合原因、未来收益、替代方案和维权路径覆盖更全，结构清...", "No active assumptions were available for attribution; collect first-party trace before mutating graph. Judge: 两者都优秀；B评分框架、验证实验和判断条件更清晰，决策可操作..."], "residual_type": "unknown", "weighted_count": 15.5}], "residual_type_counts": {"no_residual": 43, "unknown": 31}, "scope": "route=no_route", "trigger_problem_ids": ["business_0199", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0057", "software_engineering_0086", "business_0199", "daily_life_0018", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0057", "engineering_0160", "engineering_0197", "business_0199", "business_0218", "daily_life_0018", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0057", "engineering_0160", "business_0199", "business_0218", "daily_life_0133", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0160", "engineering_0197", "business_0199", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0173", "engineering_0056", "engineering_0057", "engineering_0197", "science_0204", "business_0199", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0057", "engineering_0160", "engineering_0197", "software_engineering_0355", "business_0218", "daily_life_0018", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0160", "engineering_0197", "science_0273", "business_0199", "business_0218", "daily_life_0133", "daily_life_0161", "daily_life_0173", "engineering_0004", "engineering_0056", "engineering_0160"], "verification_plan": ["rerun heldout trigger rows for the route", "include outside-control rows from other routes", "reject if repair lowers route win rate or increases control losses"]}, {"decision": "keep_with_targeted_repair", "expected_effect": "Reduce route-specific losses while preserving the observed win rate.", "policy_update_id": "trace_policy_a0241c19156c", "residual_groups": [{"bypass_route": "science_mechanism", "components": ["phase2_cache_hit"], "count": 1, "key": "residual=optimization|route=science_mechanism|components=phase2_cache_hit", "problem_ids": ["science_0097"], "residual_previews": ["No graph ids fired, but bypass/cache route science_mechanism lost; optimize the bypass bridge. Judge: A层级清晰，实验条件、因果干预与可证伪预测更完整；B机制细节好但实验展开略少..."], "residual_type": "optimization", "weighted_count": 1.0}], "residual_type_counts": {"no_residual": 25, "optimization": 1}, "scope": "route=science_mechanism", "trigger_problem_ids": ["science_0083", "science_0097", "science_0180", "science_0022", "science_0083", "science_0085", "science_0097", "science_0109", "science_0130", "science_0135", "science_0154", "science_0175", "science_0180", "science_0188", "science_0215", "science_0022", "science_0083", "science_0085", "science_0097", "science_0109", "science_0130", "science_0135", "science_0175", "science_0180", "science_0188", "science_0215"], "verification_plan": ["rerun heldout trigger rows for the route", "include outside-control rows from other routes", "reject if repair lowers route win rate or increases control losses"]}, {"decision": "keep_with_targeted_repair", "expected_effect": "Reduce route-specific losses while preserving the observed win rate.", "policy_update_id": "trace_policy_10324dc40550", "residual_groups": [{"bypass_route": "math_research_bridge", "components": ["artifact_replay_answer_meta"], "count": 1, "key": "residual=optimization|route=math_research_bridge|components=artifact_replay_answer_meta", "problem_ids": ["mathematics_0030"], "residual_previews": ["No graph ids fired, but bypass/cache route math_research_bridge lost; optimize the bypass bridge. Judge: 两者均正确严谨；B结构更清晰，含边界验证与组合解释，实用性略强。"], "residual_type": "optimization", "weighted_count": 0.5}], "residual_type_counts": {"no_residual": 18, "optimization": 1}, "scope": "route=math_research_bridge", "trigger_problem_ids": ["mathematics_0192", "mathematics_0243", "mathematics_0251", "mathematics_0257", "mathematics_0018", "mathematics_0092", "mathematics_0111", "mathematics_0192", "mathematics_0243", "mathematics_0251", "mathematics_0257", "mathematics_0018", "mathematics_0030", "mathematics_0092", "mathematics_0111", "mathematics_0192", "mathematics_0243", "mathematics_0251", "mathematics_0257"], "verification_plan": ["rerun heldout trigger rows for the route", "include outside-control rows from other routes", "reject if repair lowers route win rate or increases control losses"]}, {"decision": "keep_with_targeted_repair", "expected_effect": "Reduce route-specific losses while preserving the observed win rate.", "policy_update_id": "trace_policy_eb6f1b46ec8a", "residual_groups": [{"bypass_route": "math_formal", "components": ["artifact_replay_answer_meta"], "count": 2, "key": "residual=optimization|route=math_formal|components=artifact_replay_answer_meta", "problem_ids": ["mathematics_0163", "mathematics_0163"], "residual_previews": ["No graph ids fired, but bypass/cache route math_formal lost; optimize the bypass bridge. Judge: A覆盖矩阵性质、预处理、存储与并行化，工程实用性更强；B算法细节充分但方案面略窄。", "No graph ids fired, but bypass/cache route math_formal lost; optimize the bypass bridge. Judge: 两者均正确，A在模型性质、复杂度、预条件与存储优化上更深入，实用建议更充分。"], "residual_type": "optimization", "weighted_count": 1.0}], "residual_type_counts": {"no_residual": 12, "optimization": 2}, "scope": "route=math_formal", "trigger_problem_ids": ["mathematics_0081", "mathematics_0082", "mathematics_0126", "mathematics_0163", "mathematics_0212", "mathematics_0236", "mathematics_0239", "mathematics_0081", "mathematics_0082", "mathematics_0126", "mathematics_0163", "mathematics_0212", "mathematics_0236", "mathematics_0239"], "verification_plan": ["rerun heldout trigger rows for the route", "include outside-control rows from other routes", "reject if repair lowers route win rate or increases control losses"]}, {"decision": "reinforce_route_prior", "expected_effect": "Preserve a route with clean observed wins as a prior for similar cached traces.", "policy_update_id": "trace_policy_0294837d133f", "residual_groups": [], "residual_type_counts": {"no_residual": 8}, "scope": "route=science_decision", "trigger_problem_ids": ["science_0204", "science_0273", "science_0197", "science_0204", "science_0273", "science_0197", "science_0204", "science_0273"], "verification_plan": ["continue sampling outside controls", "demote if new losses cluster under the same route"]}]
- `secret_leak_detected`: false

### trace_policy_proposals

- `trace_outcome_path`: "phase four/assumption_graph/trace_outcome_model_collection_ms_bridge_20260601.json"
- `collection_mode`: true
- `parent_node_id`: "surface_6e7d9d238212"
- `proposal_count`: 5
- `proposal_counts`: {"assumption_revision": 5}
- `decision_counts`: {"keep_with_targeted_repair": 4, "reinforce_route_prior": 1}
- `repair_policy_count`: 4
- `candidate_count`: 5
- `heldout_verifier_count`: 5
- `secret_leak_detected`: false

### trace_policy_preflight

- `preflight_path`: "phase four/assumption_graph/trace_policy_preflight_collection_ms_bridge_20260601.json"
- `collection_mode`: true
- `proposal_count`: 5
- `readiness_counts`: {"ready_for_fresh_ablation": 5}
- `ready_count`: 5
- `missed_trigger_count`: 0
- `outside_active_count`: 0
- `command_hint_count`: 5

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
- `cluster_count`: 7
- `proposal_count`: 2
- `residual_type_counts`: {"memory_defect": 8, "optimization": 40, "unknown": 61}
- `proposal_parent_ids`: ["strategy_S08", "strategy_S08"]
- `validation_plans_complete`: true

### formal_metrics

- `mapping_count`: 9
- `complete_count`: 9
- `same_shape_count`: 9
- `warning_count`: 0
- `metric_summary`: {"mean_blackwell_dominance_proxy": 1.0, "mean_frobenius_distance": 0.424409, "mean_total_variation": 0.142899}
- `dedup_pass`: true
- `dedup_complete_mapping_count`: 9
- `dedup_unique_signature_count`: 9
- `dedup_duplicate_cluster_count`: 0
- `dedup_merge_recommendation_count`: 0
- `dedup_incomplete_mapping_excluded_count`: 0
- `dedup_positive_control`: {"duplicate_cluster_count": 1, "incomplete_mapping_excluded_count": 1, "merge_recommendation_count": 1}
- `transfer_eval_pass`: true
- `transfer_search_eval_path`: "phase four/assumption_graph/formal_mapping_search_eval_expanded_phase2_graph.json"
- `transfer_search_query_count`: 9
- `transfer_search_top1_hit_rate`: 1.0
- `transfer_search_negative_application_count`: 72
- `transfer_query_count`: 9
- `transfer_application_count`: 81
- `transfer_top1_hit_rate`: 1.0
- `transfer_pairwise_auc`: 1.0
- `transfer_positive_mean_score`: 8.00862
- `transfer_negative_mean_score`: 0.033369

### surface_hypothesis_generator

- `proposal_count`: 4
- `proposal_counts`: {"failure_hypothesis": 4}
- `surface_counts`: {"evaluator_policy": 2, "world_model_screen": 2}
- `world_model_proposal_count`: 2
- `evaluator_proposal_count`: 2
- `candidate_count`: 4
- `manifest_count`: 4
- `verifier_count`: 4
- `secret_leak_detected`: false
- `proposals`: [{"candidate_node": {"claim": "Route-policy candidates should be prioritized by feature-blend trace predictions when feature leave-one-out calibration beats route-only calibration.", "confidence": 0.44, "context_conditions": ["surface_key=world_model_screen", "issue_key=feature_beats_route_calibration"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_0b1958b45f06", "kind": "claim", "metaproductivity": 0.1, "payload": {"issue_key": "feature_beats_route_calibration", "source": "surface_hypothesis_generator", "source_metrics": {"feature_brier": 0.0711, "feature_count": 47, "route_brier": 0.1667}, "surface_key": "world_model_screen", "validation_plan": {"acceptance": "fresh ablation priority improves without increasing control losses", "current_feature_brier": 0.0711, "current_route_brier": 0.1667, "metric": "feature_leave_one_out_weighted_brier"}}, "predicted_effects": ["surface underpowered route repairs before spending fresh judge calls", "prefer proposal tests where trace features and route priors disagree"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "world_model_screen", "feature_beats_route_calibration"], "type": "world_model", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["trace_feature_disagreement_ablation", "outside_control_harm_check", "candidate_acceptance_gate"]}, "edges": [{"created_at": "2026-06-01T11:12:09Z", "evidence": null, "payload": {"issue_key": "feature_beats_route_calibration", "source": "surface_hypothesis_generator", "surface_key": "world_model_screen"}, "source": "surface_ef3d98589e5c", "target": "cand_0b1958b45f06", "type": "generated_from_residual", "weight": 0.62}], "manifest": {"action_type": "surface_hypothesis_synthesis", "artifacts": {"candidate_node": {"claim": "Route-policy candidates should be prioritized by feature-blend trace predictions when feature leave-one-out calibration beats route-only calibration.", "confidence": 0.44, "context_conditions": ["surface_key=world_model_screen", "issue_key=feature_beats_route_calibration"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_0b1958b45f06", "kind": "claim", "metaproductivity": 0.1, "payload": {"issue_key": "feature_beats_route_calibration", "source": "surface_hypothesis_generator", "source_metrics": {"feature_brier": 0.0711, "feature_count": 47, "route_brier": 0.1667}, "surface_key": "world_model_screen", "validation_plan": {"acceptance": "fresh ablation priority improves without increasing control losses", "current_feature_brier": 0.0711, "current_route_brier": 0.1667, "metric": "feature_leave_one_out_weighted_brier"}}, "predicted_effects": ["surface underpowered route repairs before spending fresh judge calls", "prefer proposal tests where trace features and route priors disagree"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "world_model_screen", "feature_beats_route_calibration"], "type": "world_model", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["trace_feature_disagreement_ablation", "outside_control_harm_check", "candidate_acceptance_gate"]}, "validation_plan": {"acceptance": "fresh ablation priority improves without increasing control losses", "current_feature_brier": 0.0711, "current_route_brier": 0.1667, "metric": "feature_leave_one_out_weighted_brier"}}, "assumption": "Route-policy candidates should be prioritized by feature-blend trace predictions when feature leave-one-out calibration beats route-only calibration.", "assumption_ids": ["surface_ef3d98589e5c", "cand_0b1958b45f06"], "component": "surface_hypothesis_generator", "cost": 0.0, "expected_effect": "Convert system-level evaluator/world-model residuals into falsifiable candidate work.", "metadata": {"eval_id": "perf_surface_hypotheses", "issue_key": "feature_beats_route_calibration", "surface_key": "world_model_screen"}, "observed_effect": null, "parent_trial_id": null, "predicted_regressions": [], "problem_id": "surface_hypothesis::feature_beats_route_calibration", "residual": null, "residual_type": null, "rollback_condition": "Reject if heldout trigger/control checks fail or evaluator uncertainty increases.", "status": "pending", "timestamp": "2026-06-01T11:12:09Z", "trial_id": "trial_0b1958b45f06", "verification_plan": "{\"acceptance\": \"fresh ablation priority improves without increasing control losses\", \"current_feature_brier\": 0.0711, \"current_route_brier\": 0.1667, \"metric\": \"feature_leave_one_out_weighted_brier\"}", "verifier": "trace_feature_disagreement_ablation", "why_selected": "Performance validation exposed feature_beats_route_calibration on world_model_screen."}, "parent_node_id": "surface_ef3d98589e5c", "priority": 0.74, "proposal_id": "prop_bfed841fed9d", "proposal_type": "failure_hypothesis", "rationale": "Generated from world_model_screen residual signal feature_beats_route_calibration.", "source_action": {"action_type": "surface_hypothesis", "feature_brier": 0.0711, "feature_count": 47, "issue_key": "feature_beats_route_calibration", "route_brier": 0.1667, "surface_key": "world_model_screen"}}, {"candidate_node": {"claim": "Before promoting replay-heavy route repairs, require source-stratified calibration with a first-party runtime quota and replay evidence capped below live trace evidence.", "confidence": 0.44, "context_conditions": ["surface_key=world_model_screen", "issue_key=artifact_replay_dominates_trace_calibration"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_a56537408cc8", "kind": "claim", "metaproductivity": 0.1, "payload": {"issue_key": "artifact_replay_dominates_trace_calibration", "source": "surface_hypothesis_generator", "source_metrics": {"artifact_replay_trainable_rows": 132, "first_party_trainable_rows": 9}, "surface_key": "world_model_screen", "validation_plan": {"acceptance": "first-party calibration remains within the replay-weighted confidence band", "artifact_replay_trainable_rows": 132, "first_party_trainable_rows": 9}}, "predicted_effects": ["prevent artifact replay from overpowering first-party runtime evidence", "focus new data collection on routes with replay-only losses"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "world_model_screen", "artifact_replay_dominates_trace_calibration"], "type": "world_model", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["source_stratified_trace_calibration", "outside_control_harm_check", "candidate_acceptance_gate"]}, "edges": [{"created_at": "2026-06-01T11:12:09Z", "evidence": null, "payload": {"issue_key": "artifact_replay_dominates_trace_calibration", "source": "surface_hypothesis_generator", "surface_key": "world_model_screen"}, "source": "surface_ef3d98589e5c", "target": "cand_a56537408cc8", "type": "generated_from_residual", "weight": 0.62}], "manifest": {"action_type": "surface_hypothesis_synthesis", "artifacts": {"candidate_node": {"claim": "Before promoting replay-heavy route repairs, require source-stratified calibration with a first-party runtime quota and replay evidence capped below live trace evidence.", "confidence": 0.44, "context_conditions": ["surface_key=world_model_screen", "issue_key=artifact_replay_dominates_trace_calibration"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_a56537408cc8", "kind": "claim", "metaproductivity": 0.1, "payload": {"issue_key": "artifact_replay_dominates_trace_calibration", "source": "surface_hypothesis_generator", "source_metrics": {"artifact_replay_trainable_rows": 132, "first_party_trainable_rows": 9}, "surface_key": "world_model_screen", "validation_plan": {"acceptance": "first-party calibration remains within the replay-weighted confidence band", "artifact_replay_trainable_rows": 132, "first_party_trainable_rows": 9}}, "predicted_effects": ["prevent artifact replay from overpowering first-party runtime evidence", "focus new data collection on routes with replay-only losses"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "world_model_screen", "artifact_replay_dominates_trace_calibration"], "type": "world_model", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["source_stratified_trace_calibration", "outside_control_harm_check", "candidate_acceptance_gate"]}, "validation_plan": {"acceptance": "first-party calibration remains within the replay-weighted confidence band", "artifact_replay_trainable_rows": 132, "first_party_trainable_rows": 9}}, "assumption": "Before promoting replay-heavy route repairs, require source-stratified calibration with a first-party runtime quota and replay evidence capped below live trace evidence.", "assumption_ids": ["surface_ef3d98589e5c", "cand_a56537408cc8"], "component": "surface_hypothesis_generator", "cost": 0.0, "expected_effect": "Convert system-level evaluator/world-model residuals into falsifiable candidate work.", "metadata": {"eval_id": "perf_surface_hypotheses", "issue_key": "artifact_replay_dominates_trace_calibration", "surface_key": "world_model_screen"}, "observed_effect": null, "parent_trial_id": null, "predicted_regressions": [], "problem_id": "surface_hypothesis::artifact_replay_dominates_trace_calibration", "residual": null, "residual_type": null, "rollback_condition": "Reject if heldout trigger/control checks fail or evaluator uncertainty increases.", "status": "pending", "timestamp": "2026-06-01T11:12:09Z", "trial_id": "trial_a56537408cc8", "verification_plan": "{\"acceptance\": \"first-party calibration remains within the replay-weighted confidence band\", \"artifact_replay_trainable_rows\": 132, \"first_party_trainable_rows\": 9}", "verifier": "source_stratified_trace_calibration", "why_selected": "Performance validation exposed artifact_replay_dominates_trace_calibration on world_model_screen."}, "parent_node_id": "surface_ef3d98589e5c", "priority": 0.7, "proposal_id": "prop_d6f5832b52bf", "proposal_type": "failure_hypothesis", "rationale": "Generated from world_model_screen residual signal artifact_replay_dominates_trace_calibration.", "source_action": {"action_type": "surface_hypothesis", "artifact_replay_trainable_rows": 132, "first_party_trainable_rows": 9, "issue_key": "artifact_replay_dominates_trace_calibration", "surface_key": "world_model_screen"}}, {"candidate_node": {"claim": "Candidates with missing V4 acceptance evidence should spawn evaluator-calibration tests before they can become promotion candidates.", "confidence": 0.44, "context_conditions": ["surface_key=evaluator_policy", "issue_key=acceptance_missing_evaluator_evidence"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_15a570b22f99", "kind": "evaluator_policy", "metaproductivity": 0.1, "payload": {"issue_key": "acceptance_missing_evaluator_evidence", "source": "surface_hypothesis_generator", "source_metrics": {"stage_status_counts": {"V0:defer": 6, "V0:pass": 21, "V0:repair": 6, "V1:defer": 31, "V1:pass": 2, "V2:not_applicable": 33, "V3:block": 6, "V3:defer": 6, "V3:fail": 14, "V3:pass": 7, "V4:fail": 14, "V4:missing": 17, "V4:pass": 2}}, "surface_key": "evaluator_policy", "validation_plan": {"acceptance": "missing V4 cases receive scoped trigger/control judgments before promotion", "v4_fail": 14, "v4_missing": 17}}, "predicted_effects": ["turn missing acceptance states into explicit judge/evaluator work items", "separate weak benefit from evaluator uncertainty before graph mutation"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "evaluator_policy", "acceptance_missing_evaluator_evidence"], "type": "evaluator", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["cross_judge_acceptance_completion", "outside_control_harm_check", "candidate_acceptance_gate"]}, "edges": [{"created_at": "2026-06-01T11:12:09Z", "evidence": null, "payload": {"issue_key": "acceptance_missing_evaluator_evidence", "source": "surface_hypothesis_generator", "surface_key": "evaluator_policy"}, "source": "surface_d339783fea78", "target": "cand_15a570b22f99", "type": "generated_from_residual", "weight": 0.62}], "manifest": {"action_type": "surface_hypothesis_synthesis", "artifacts": {"candidate_node": {"claim": "Candidates with missing V4 acceptance evidence should spawn evaluator-calibration tests before they can become promotion candidates.", "confidence": 0.44, "context_conditions": ["surface_key=evaluator_policy", "issue_key=acceptance_missing_evaluator_evidence"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_15a570b22f99", "kind": "evaluator_policy", "metaproductivity": 0.1, "payload": {"issue_key": "acceptance_missing_evaluator_evidence", "source": "surface_hypothesis_generator", "source_metrics": {"stage_status_counts": {"V0:defer": 6, "V0:pass": 21, "V0:repair": 6, "V1:defer": 31, "V1:pass": 2, "V2:not_applicable": 33, "V3:block": 6, "V3:defer": 6, "V3:fail": 14, "V3:pass": 7, "V4:fail": 14, "V4:missing": 17, "V4:pass": 2}}, "surface_key": "evaluator_policy", "validation_plan": {"acceptance": "missing V4 cases receive scoped trigger/control judgments before promotion", "v4_fail": 14, "v4_missing": 17}}, "predicted_effects": ["turn missing acceptance states into explicit judge/evaluator work items", "separate weak benefit from evaluator uncertainty before graph mutation"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "evaluator_policy", "acceptance_missing_evaluator_evidence"], "type": "evaluator", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["cross_judge_acceptance_completion", "outside_control_harm_check", "candidate_acceptance_gate"]}, "validation_plan": {"acceptance": "missing V4 cases receive scoped trigger/control judgments before promotion", "v4_fail": 14, "v4_missing": 17}}, "assumption": "Candidates with missing V4 acceptance evidence should spawn evaluator-calibration tests before they can become promotion candidates.", "assumption_ids": ["surface_d339783fea78", "cand_15a570b22f99"], "component": "surface_hypothesis_generator", "cost": 0.0, "expected_effect": "Convert system-level evaluator/world-model residuals into falsifiable candidate work.", "metadata": {"eval_id": "perf_surface_hypotheses", "issue_key": "acceptance_missing_evaluator_evidence", "surface_key": "evaluator_policy"}, "observed_effect": null, "parent_trial_id": null, "predicted_regressions": [], "problem_id": "surface_hypothesis::acceptance_missing_evaluator_evidence", "residual": null, "residual_type": null, "rollback_condition": "Reject if heldout trigger/control checks fail or evaluator uncertainty increases.", "status": "pending", "timestamp": "2026-06-01T11:12:09Z", "trial_id": "trial_15a570b22f99", "verification_plan": "{\"acceptance\": \"missing V4 cases receive scoped trigger/control judgments before promotion\", \"v4_fail\": 14, \"v4_missing\": 17}", "verifier": "cross_judge_acceptance_completion", "why_selected": "Performance validation exposed acceptance_missing_evaluator_evidence on evaluator_policy."}, "parent_node_id": "surface_d339783fea78", "priority": 0.68, "proposal_id": "prop_54bd770812c8", "proposal_type": "failure_hypothesis", "rationale": "Generated from evaluator_policy residual signal acceptance_missing_evaluator_evidence.", "source_action": {"action_type": "surface_hypothesis", "issue_key": "acceptance_missing_evaluator_evidence", "stage_status_counts": {"V0:defer": 6, "V0:pass": 21, "V0:repair": 6, "V1:defer": 31, "V1:pass": 2, "V2:not_applicable": 33, "V3:block": 6, "V3:defer": 6, "V3:fail": 14, "V3:pass": 7, "V4:fail": 14, "V4:missing": 17, "V4:pass": 2}, "surface_key": "evaluator_policy"}}, {"candidate_node": {"claim": "Formal-transfer labels that are generated from trigger schemas should be followed by independent downstream judge probes before being treated as evaluator ground truth.", "confidence": 0.44, "context_conditions": ["surface_key=evaluator_policy", "issue_key=formal_transfer_labels_are_trigger_derived"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_54a719dad304", "kind": "evaluator_policy", "metaproductivity": 0.1, "payload": {"issue_key": "formal_transfer_labels_are_trigger_derived", "source": "surface_hypothesis_generator", "source_metrics": {"transfer_search_negative_application_count": 72, "transfer_search_query_count": 9}, "surface_key": "evaluator_policy", "validation_plan": {"acceptance": "independent downstream probes preserve the same top-1 mapping decisions", "negative_application_count": 72, "trigger_derived_query_count": 9}}, "predicted_effects": ["keep formal alignment useful without mistaking trigger-derived labels for independent validation", "create heldout judge probes for formal-transfer candidates"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "evaluator_policy", "formal_transfer_labels_are_trigger_derived"], "type": "evaluator", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["independent_formal_transfer_judge_probe", "outside_control_harm_check", "candidate_acceptance_gate"]}, "edges": [{"created_at": "2026-06-01T11:12:09Z", "evidence": null, "payload": {"issue_key": "formal_transfer_labels_are_trigger_derived", "source": "surface_hypothesis_generator", "surface_key": "evaluator_policy"}, "source": "surface_d339783fea78", "target": "cand_54a719dad304", "type": "generated_from_residual", "weight": 0.62}], "manifest": {"action_type": "surface_hypothesis_synthesis", "artifacts": {"candidate_node": {"claim": "Formal-transfer labels that are generated from trigger schemas should be followed by independent downstream judge probes before being treated as evaluator ground truth.", "confidence": 0.44, "context_conditions": ["surface_key=evaluator_policy", "issue_key=formal_transfer_labels_are_trigger_derived"], "created_at": "2026-06-01T11:12:09Z", "evidence_ids": [], "formal_form": null, "id": "cand_54a719dad304", "kind": "evaluator_policy", "metaproductivity": 0.1, "payload": {"issue_key": "formal_transfer_labels_are_trigger_derived", "source": "surface_hypothesis_generator", "source_metrics": {"transfer_search_negative_application_count": 72, "transfer_search_query_count": 9}, "surface_key": "evaluator_policy", "validation_plan": {"acceptance": "independent downstream probes preserve the same top-1 mapping decisions", "negative_application_count": 72, "trigger_derived_query_count": 9}}, "predicted_effects": ["keep formal alignment useful without mistaking trigger-derived labels for independent validation", "create heldout judge probes for formal-transfer candidates"], "residual_ids": [], "risk_predictions": ["may overfit current performance-validation artifacts", "must pass heldout trigger/control checks before graph mutation"], "source_refs": [], "status": "candidate", "tags": ["candidate", "surface_hypothesis", "evaluator_policy", "formal_transfer_labels_are_trigger_derived"], "type": "evaluator", "updated_at": "2026-06-01T11:12:09Z", "verifiers": ["independent_formal_transfer_judge_probe", "outside_control_harm_check", "candidate_acceptance_gate"]}, "validation_plan": {"acceptance": "independent downstream probes preserve the same top-1 mapping decisions", "negative_application_count": 72, "trigger_derived_query_count": 9}}, "assumption": "Formal-transfer labels that are generated from trigger schemas should be followed by independent downstream judge probes before being treated as evaluator ground truth.", "assumption_ids": ["surface_d339783fea78", "cand_54a719dad304"], "component": "surface_hypothesis_generator", "cost": 0.0, "expected_effect": "Convert system-level evaluator/world-model residuals into falsifiable candidate work.", "metadata": {"eval_id": "perf_surface_hypotheses", "issue_key": "formal_transfer_labels_are_trigger_derived", "surface_key": "evaluator_policy"}, "observed_effect": null, "parent_trial_id": null, "predicted_regressions": [], "problem_id": "surface_hypothesis::formal_transfer_labels_are_trigger_derived", "residual": null, "residual_type": null, "rollback_condition": "Reject if heldout trigger/control checks fail or evaluator uncertainty increases.", "status": "pending", "timestamp": "2026-06-01T11:12:09Z", "trial_id": "trial_54a719dad304", "verification_plan": "{\"acceptance\": \"independent downstream probes preserve the same top-1 mapping decisions\", \"negative_application_count\": 72, \"trigger_derived_query_count\": 9}", "verifier": "independent_formal_transfer_judge_probe", "why_selected": "Performance validation exposed formal_transfer_labels_are_trigger_derived on evaluator_policy."}, "parent_node_id": "surface_d339783fea78", "priority": 0.62, "proposal_id": "prop_35804d4d7168", "proposal_type": "failure_hypothesis", "rationale": "Generated from evaluator_policy residual signal formal_transfer_labels_are_trigger_derived.", "source_action": {"action_type": "surface_hypothesis", "issue_key": "formal_transfer_labels_are_trigger_derived", "surface_key": "evaluator_policy", "transfer_search_negative_application_count": 72, "transfer_search_query_count": 9}}]

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

### reconstruction_progress

- `structure_percent`: 83.2
- `behavior_percent`: 74.2
- `weighted_percent`: 78.3
- `completed_item_count`: 3
- `item_count`: 9
- `status_counts`: {"operational": 8, "partial": 1}
- `lowest_behavior_items`: [{"behavior_score": 0.68, "key": "G_formal_alignment_layer", "status": "partial", "structure_score": 0.78}, {"behavior_score": 0.7, "key": "C_world_model_simulator", "status": "operational", "structure_score": 0.82}, {"behavior_score": 0.73, "key": "B_hypothesis_generator", "status": "operational", "structure_score": 0.82}]
- `top_next_actions`: [{"action": "Expand formal-transfer labels beyond the current five-query audit.", "item": "G_formal_alignment_layer", "priority": 0.275}, {"action": "Use dedup recommendations to merge complete formal equivalents after verifier approval.", "item": "G_formal_alignment_layer", "priority": 0.275}, {"action": "Accumulate a larger trace dataset from real first-party runs.", "item": "C_world_model_simulator", "priority": 0.246}, {"action": "Train/calibrate a cheap predictor over problem + activated assumptions + trace features + residual label.", "item": "C_world_model_simulator", "priority": 0.246}, {"action": "Add a generator pass that turns evaluator/world-model residuals into candidate proposals.", "item": "B_hypothesis_generator", "priority": 0.2295}]
