"""Full-v3 paper-scale evidence aggregation.

This module does not run new API calls.  It aggregates the strongest existing
first-party live/cached artifacts, v3 mechanism validations, and paper-facing
baseline tables into one auditable evidence payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_paper_scale_evidence_20260611.json"

REQUIRED_ARTIFACTS = {
    "paper_main": PAPER_DIR / "paper_main_experiment_20260605.json",
    "baseline_hardening": PAPER_DIR / "paper_baseline_hardening_20260605.json",
    "retrieval_baselines": PAPER_DIR / "paper_retrieval_baselines_20260605.json",
    "repro_pack": PAPER_DIR / "paper_repro_pack_20260605.json",
    "first_party_world_model_scale": PAPER_DIR / "first_party_world_model_scale_20260604.json",
    "v2_phase0_contract": PAPER_DIR / "full_v2_phase0_contract_bypass_20260611.json",
    "v3_phase0_contract": PAPER_DIR / "full_v3_phase0_contract_checker_20260611.json",
    "v3_phase1_memory": PAPER_DIR / "full_v3_phase1_memory_consolidation_20260611.json",
    "v3_phase1_retrieval_audit": PAPER_DIR / "full_v3_phase1_first_party_retrieval_audit_20260611.json",
    "v3_phase2_verifier": PAPER_DIR / "full_v3_phase2_verifier_synthesis_20260611.json",
    "v3_phase3_rollout": PAPER_DIR / "full_v3_phase3_rollout_search_control_20260611.json",
    "v3_phase3_learned_rollout": PAPER_DIR / "full_v3_phase3_learned_rollout_20260611.json",
    "v2_phase4_generator": PAPER_DIR / "full_v2_phase4_hypothesis_generator_bypass_20260611.json",
    "v3_phase4_generator": PAPER_DIR / "full_v3_phase4_hypothesis_generator_20260611.json",
    "v3_live_residual_clusterer": PAPER_DIR / "full_v3_live_residual_clusterer_20260611.json",
    "v3_residual_multigeneration_loop": PAPER_DIR / "full_v3_residual_multigeneration_loop_20260611.json",
    "v3_residual_live_mini_loop": PAPER_DIR / "full_v3_residual_live_mini_loop_20260611.json",
    "v3_phase5_bandit": PAPER_DIR / "full_v3_phase5_contextual_bandit_scheduler_20260611.json",
    "v2_phase6_formal": PAPER_DIR / "full_v2_phase6_formal_alignment_bypass_20260611.json",
    "v3_phase6_formal": PAPER_DIR / "full_v3_phase6_formal_transfer_engine_20260611.json",
    "v3_phase7_long_run": PAPER_DIR / "full_v3_phase7_long_run_benchmark_20260611.json",
    "v3_phase7_daemon_soak": PAPER_DIR / "full_v3_phase7_daemon_soak_20260611.json",
    "v3_phase8_creativity_world_coverage": PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json",
    "vertical_slice": PAPER_DIR / "full_v2_vertical_slice_bypass_20260611.json",
    "frozen_v3_vs_v1": PAPER_DIR / "full_v3_frozen_v1_comparison_20260611.json",
    "fresh_live_guarded_300": PAPER_DIR / "full_v3_fresh_live_business_guard_heldout300_gptmini_gpt55_20260611.json",
    "fresh_live_guarded_full_remaining": PAPER_DIR / "full_v3_fresh_live_business_guard_full_remaining_gptmini_gpt55_20260611.json",
    "fresh_live_selective_expansion": PAPER_DIR / "full_v3_fresh_live_cue_repair_v4_full_remaining_gptmini_gpt55_20260611.json",
    "v3_phase9_compact_frame_guard": PAPER_DIR / "full_v3_phase9_compact_frame_guard_20260611.json",
    "v3_phase9_hybrid_guard": PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json",
    "v3_phase10_discrete_world_model": PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json",
    "v3_phase10_reliability_calibration": PAPER_DIR / "full_v3_phase10_reliability_calibration_20260611.json",
    "v3_guard_policy_learning": PAPER_DIR / "full_v3_guard_policy_learning_20260611.json",
    "v3_main_graph_memory_shadow": PAPER_DIR / "full_v3_main_graph_memory_shadow_20260611.json",
    "v3_main_graph_memory_controlled_apply": PAPER_DIR / "full_v3_main_graph_memory_controlled_apply_20260611.json",
    "v3_residual_fresh_live_loop": PAPER_DIR / "full_v3_residual_fresh_live_loop_20260611.json",
    "v3_continuous_daemon_scheduler": PAPER_DIR / "full_v3_continuous_daemon_scheduler_20260611.json",
    "v3_world_model_calibration": PAPER_DIR / "full_v3_world_model_calibration_20260611.json",
    "v3_same_batch_ablation_suite": PAPER_DIR / "full_v3_same_batch_ablation_suite_20260611.json",
    "v3_phase11_capability_audit": PAPER_DIR / "full_v3_phase11_capability_audit_20260611.json",
}

KEY_TOGGLE_BASELINES = {
    "no_world_model_trace_policy",
    "no_recursive_runner_one_shot",
    "no_novelty_gate_incremental_addition",
}


def build_full_v3_paper_scale_evidence_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_paper_scale_evidence_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {
        name: _load_json(root / path)
        for name, path in REQUIRED_ARTIFACTS.items()
    }
    evidence = {
        "paper_main": _paper_main_summary(artifacts["paper_main"]),
        "baseline_hardening": _baseline_hardening_summary(artifacts["baseline_hardening"]),
        "retrieval_baselines": _retrieval_summary(artifacts["retrieval_baselines"]),
        "first_party_world_model_scale": _world_model_scale_summary(artifacts["first_party_world_model_scale"]),
        "v3_mechanism": _v3_mechanism_summary(artifacts),
        "live_residual_clusterer": _metric_subset(
            artifacts["v3_live_residual_clusterer"].get("metrics", {}),
            [
                "source_artifact_count",
                "observation_count",
                "weighted_residual_count",
                "cluster_count",
                "systematic_weighted_coverage",
                "phase9_live_residual_observation_count",
                "formal_residual_observation_count",
                "profile_residual_observation_count",
                "resolved_cluster_count",
                "blocked_profile_residual_count",
                "next_generation_proposal_seed_count",
                "largest_live_cluster_support",
                "largest_live_cluster_axis",
                "largest_live_cluster_domain",
                "largest_live_cluster_pattern",
                "largest_live_cluster_status",
                "uses_raw_prompts_or_answers",
            ],
        ),
        "residual_multigeneration_loop": _metric_subset(
            artifacts["v3_residual_multigeneration_loop"].get("metrics", {}),
            [
                "generation_count",
                "seed_cluster_count",
                "proposal_count",
                "retained_count",
                "retention_rate",
                "proposal_family_count",
                "recursive_parent_closure_rate",
                "evaluation_plan_coverage",
                "negative_control_coverage",
                "uses_raw_prompts_or_answers",
                "graph_mutation_count",
            ],
        ),
        "residual_live_mini_loop": _metric_subset(
            artifacts["v3_residual_live_mini_loop"].get("metrics", {}),
            [
                "source_generation_count",
                "source_retained_count",
                "selected_candidate_count",
                "contract_ready_count",
                "preflight_ready_count",
                "accepted_count",
                "applied_count",
                "applied_active_count",
                "graph_copy_node_delta",
                "main_graph_mutation_count",
                "new_api_call_count",
                "trigger_rows_per_candidate",
                "control_rows_per_candidate",
                "phase10_readback_pass",
                "phase10_leave_pattern_guard_harm_count",
                "phase10_leave_route_guard_harm_count",
                "phase10_guard_assumption_node_count",
            ],
        ),
        "residual_fresh_live_loop": _metric_subset(
            artifacts["v3_residual_fresh_live_loop"].get("metrics", {}),
            [
                "execution_mode",
                "selected_candidate_count",
                "contract_ready_count",
                "preflight_ready_count",
                "fresh_live_path_present",
                "live_env_ready",
                "fresh_api_call_count",
                "planned_fresh_api_call_count",
                "accepted_count",
                "applied_count",
                "graph_copy_node_delta",
                "main_graph_mutation_count",
                "secret_value_exposed",
            ],
        ),
        "vertical_slice": _metric_subset(
            artifacts["vertical_slice"].get("metrics", {}),
            [
                "generation_count",
                "candidate_count",
                "live_call_saving_rate",
                "true_positive_block_rate",
                "residual_explained_delta",
                "downstream_score_delta",
                "world_model_brier_improvement",
                "full_loop_margin_over_best_control",
            ],
        ),
        "long_run": _metric_subset(
            artifacts["v3_phase7_long_run"].get("metrics", {}),
            [
                "episode_count",
                "long_run_stability",
                "graph_pollution_rate",
                "accepted_assumption_survival_rate",
                "downstream_win_rate_on_unseen",
                "capability_score_improvement",
                "parallel_speedup_proxy",
                "checkpoint_recovery_success",
                "production_queue_source_count",
                "production_planned_leaf_count",
                "production_pre_live_block_or_defer_count",
                "production_manifest_reopen_count",
                "production_node_mutation_count",
                "production_apply_enabled_count",
                "production_execute_enabled_count",
            ],
        ),
        "daemon_soak": _metric_subset(
            artifacts["v3_phase7_daemon_soak"].get("metrics", {}),
            [
                "cycle_count",
                "queue_source_load_count",
                "planned_leaf_count",
                "screen_block_or_defer_count",
                "manifest_reopen_count",
                "checkpoint_reopen_success_rate",
                "node_mutation_count",
                "apply_enabled_count",
                "execute_enabled_count",
                "continuous_background_daemon",
            ],
        ),
        "frozen_v3_vs_v1": _metric_subset(
            artifacts["frozen_v3_vs_v1"].get("metrics", {}),
            [
                "downstream_problem_count",
                "full_v3_downstream_accuracy",
                "v1_kernel_accuracy",
                "full_v3_margin_vs_v1_kernel",
                "hipporag_style_accuracy",
                "full_v3_margin_vs_hipporag_style",
                "best_nonfull_system",
                "full_v3_margin_vs_best_nonfull",
                "assumption_capability_improvement",
            ],
        ),
        "fresh_live_guarded_300": _fresh_live_summary(artifacts["fresh_live_guarded_300"]),
        "fresh_live_guarded_full_remaining": _fresh_live_summary(artifacts["fresh_live_guarded_full_remaining"]),
        "fresh_live_selective_expansion": _fresh_live_summary(artifacts["fresh_live_selective_expansion"]),
        "phase8_creativity_world_coverage": _metric_subset(
            artifacts["v3_phase8_creativity_world_coverage"].get("metrics", {}),
            [
                "creative_candidate_count",
                "nonlocal_candidate_ratio",
                "residual_cluster_coverage",
                "quality_world_model_auroc",
                "quality_world_model_brier",
                "quality_base_rate_brier",
                "selected_quality_profile_id",
                "selected_coverage_profile_id",
                "coverage_profile_active_gain_over_quality",
                "coverage_profile_vs_base_utility",
                "coverage_profile_vs_placebo_utility",
            ],
        ),
        "phase1_first_party_retrieval_audit": _metric_subset(
            artifacts["v3_phase1_retrieval_audit"].get("metrics", {}),
            [
                "query_count",
                "precision_before",
                "precision_after",
                "precision_delta",
                "negative_transfer_delta",
                "context_efficiency_delta",
                "after_archived_hits",
                "applied_consolidated_node_count",
                "archived_node_count",
            ],
        ),
        "phase3_learned_rollout": _metric_subset(
            artifacts["v3_phase3_learned_rollout"].get("metrics", {}),
            [
                "transition_row_count",
                "rollout_row_count",
                "candidate_action_coverage",
                "selected_reward_lift_over_v3",
                "selected_vs_v1_lift_over_v3",
                "selected_vs_v1_utility",
                "teacher_match_rate",
                "recommended_promotion",
                "uses_raw_prompts_or_answers",
            ],
        ),
        "phase5_live_scheduler": _metric_subset(
            artifacts["v3_phase5_bandit"].get("metrics", {}),
            [
                "live_profile_source_artifact_count",
                "live_profile_count",
                "live_profile_pass_count",
                "live_selected_production_profile",
                "live_selected_exploration_profile",
                "live_scheduler_vs_v1_utility",
                "live_scheduler_vs_original_v3_utility",
                "live_scheduler_lift_over_v3",
                "live_scheduler_blocks_compact_default",
                "live_scheduler_keeps_phase10_as_candidate",
                "live_scheduler_uses_raw_prompts_or_answers",
                "live_scheduler_calibrated_guard_lift_over_hybrid",
                "live_scheduler_calibrated_guard_vs_original_v3_lift_over_hybrid",
            ],
        ),
        "phase9_compact_frame_guard": _metric_subset(
            artifacts["v3_phase9_compact_frame_guard"].get("metrics", {}),
            [
                "active_case_count",
                "repair_vs_v1_n",
                "repair_vs_v1_utility",
                "repair_vs_v1_margin",
                "repair_vs_v3_utility",
                "repair_margin_gain_over_v3_vs_v1",
                "planned_total_model_calls",
            ],
        ),
        "phase9_hybrid_guard": _metric_subset(
            artifacts["v3_phase9_hybrid_guard"].get("metrics", {}),
            [
                "heldout_case_count",
                "selected_candidate_case_count",
                "hybrid_selected_arm_counts",
                "v3_vs_v1_heldout_utility",
                "hybrid_vs_v1_heldout_utility",
                "hybrid_vs_v1_heldout_margin",
                "hybrid_vs_original_v3_heldout_utility",
                "hybrid_lift_over_v3_vs_v1_heldout",
            ],
        ),
        "phase10_discrete_world_model": _metric_subset(
            artifacts["v3_phase10_discrete_world_model"].get("metrics", {}),
            [
                "candidate_transition_count",
                "compact_support_row_count",
                "learned_selected_arm_counts",
                "loo_selected_vs_v1_utility",
                "loo_selected_vs_v1_lift_over_v3",
                "loo_selected_vs_v3_utility",
                "all_heldout_policy_vs_v1_utility",
                "all_heldout_policy_lift_over_v3",
                "all_heldout_policy_vs_original_v3_utility",
                "calibrated_policy_vs_v1_utility",
                "calibrated_policy_vs_original_v3_utility",
                "calibrated_policy_lift_over_v3",
                "calibrated_policy_lift_over_raw_world_model",
                "calibrated_policy_lift_over_retained_hybrid",
                "calibrated_policy_vs_original_v3_lift_over_hybrid",
                "calibrated_policy_harm_vs_hybrid_count",
                "calibrated_policy_win_vs_hybrid_count",
                "calibrated_policy_override_count",
                "retained_hybrid_vs_v1_utility",
                "learned_gap_to_retained_hybrid",
                "recommended_promotion",
                "calibration_beats_base_rate",
                "leave_pattern_out_group_count",
                "leave_pattern_out_raw_vs_v3_lift",
                "leave_pattern_out_guard_vs_v3_lift",
                "leave_pattern_out_guard_lift_over_raw",
                "leave_pattern_out_raw_harm_count",
                "leave_pattern_out_guard_harm_count",
                "leave_route_tag_out_group_count",
                "leave_route_tag_out_raw_vs_v3_lift",
                "leave_route_tag_out_guard_vs_v3_lift",
                "leave_route_tag_out_guard_lift_over_raw",
                "leave_route_tag_out_raw_harm_count",
                "leave_route_tag_out_guard_harm_count",
                "guard_assumption_node_count",
                "guard_assumption_active_count",
                "guard_assumption_candidate_count",
                "calibrated_rows_with_guard_assumption_rate",
            ],
        ),
        "phase10_reliability_calibration": _metric_subset(
            artifacts["v3_phase10_reliability_calibration"].get("metrics", {}),
            [
                "observed_arm_record_count",
                "problem_count",
                "arm_count",
                "bin_count",
                "raw_mae",
                "calibrated_mae",
                "base_rate_mae",
                "calibrated_mae_lift_over_raw",
                "calibrated_mae_lift_over_base_rate",
                "raw_brier",
                "calibrated_brier",
                "base_rate_brier",
                "calibrated_brier_lift_over_base_rate",
                "raw_ece",
                "calibrated_ece",
                "calibrated_ece_lift_over_raw",
                "source_phase10_calibration_beats_base_rate",
            ],
        ),
        "guard_policy_learning": _metric_subset(
            artifacts["v3_guard_policy_learning"].get("metrics", {}),
            [
                "learned_guard_update_count",
                "supported_guard_count",
                "guard_weight_range",
                "promote_weight_count",
                "demote_weight_count",
                "keep_candidate_count",
                "learned_policy_vs_v1_utility",
                "learned_policy_lift_over_hybrid",
                "learned_policy_harm_vs_hybrid_count",
                "raw_world_model_status",
                "reliability_calibrated_mae_lift_over_base",
            ],
        ),
        "main_graph_memory_shadow": _metric_subset(
            artifacts["v3_main_graph_memory_shadow"].get("metrics", {}),
            [
                "main_graph_node_count",
                "dry_run_group_count",
                "dry_run_planned_archive_count",
                "dry_run_planned_consolidated_node_count",
                "dry_run_store_mutated",
                "main_graph_mutated",
                "shadow_applied_archived_node_count",
                "shadow_applied_consolidated_node_count",
                "shadow_added_edge_count",
                "shadow_node_delta",
                "query_count",
                "precision_before",
                "precision_after",
                "precision_delta",
                "archive_exposure_before",
                "archive_exposure_after",
                "archive_exposure_delta",
                "memory_hits_before",
                "memory_hits_after",
                "memory_hit_delta",
                "context_efficiency_delta",
            ],
        ),
        "main_graph_memory_controlled_apply": _metric_subset(
            artifacts["v3_main_graph_memory_controlled_apply"].get("metrics", {}),
            [
                "apply_main",
                "dry_run_group_count",
                "planned_archive_count",
                "planned_consolidated_node_count",
                "rollback_entry_count",
                "applied_archived_node_count",
                "applied_consolidated_node_count",
                "node_delta",
                "main_graph_mutated",
                "precision_delta",
                "archive_exposure_after",
                "memory_hit_delta",
                "context_efficiency_delta",
            ],
        ),
        "continuous_daemon_scheduler": _metric_subset(
            artifacts["v3_continuous_daemon_scheduler"].get("metrics", {}),
            [
                "scheduled_cycle_count",
                "checkpoint_pair_count",
                "rate_limit_violation_count",
                "recovery_action_count",
                "fresh_loop_queue_integrated",
                "memory_apply_queue_integrated",
                "daemon_readback_queue_integrated",
                "ungated_graph_mutation_count",
                "continuous_background_ready",
                "background_process_started",
            ],
        ),
        "same_batch_ablation_suite": _metric_subset(
            artifacts["v3_same_batch_ablation_suite"].get("metrics", {}),
            [
                "same_batch_judged_n",
                "toggle_pair_count",
                "raw_v3_vs_v1_utility",
                "raw_v3_vs_v1_ci_lower",
                "raw_v3_vs_no_morphism_utility",
                "raw_v3_vs_no_recursive_utility",
                "raw_v3_vs_no_world_model_utility",
                "hybrid_lift_over_raw_v3",
                "calibrated_lift_over_hybrid",
                "calibrated_harm_vs_hybrid_count",
                "fresh_live_300_problem_level_n",
                "uses_raw_prompts_or_answers",
            ],
        ),
        "world_model_calibration": _metric_subset(
            artifacts["v3_world_model_calibration"].get("metrics", {}),
            [
                "source_artifact_count",
                "calibration_surface_count",
                "calibrated_surface_count",
                "leave_domain_out_surface_count",
                "uncalibrated_promotion_count",
                "phase8_quality_brier_improvement",
                "phase9_leave_domain_out_available",
                "phase9_leave_domain_out_domain_count",
                "phase9_leave_domain_out_nonnegative_domain_count",
                "phase9_leave_domain_out_max_calibration_error",
                "phase10_all_lift_over_v3",
                "phase10_calibration_beats_base_rate",
                "phase10_selected_arm_mae_minus_base_rate",
                "phase10_calibrated_policy_vs_v1_utility",
                "phase10_calibrated_policy_vs_original_v3_utility",
                "phase10_calibrated_policy_lift_over_v3",
                "phase10_calibrated_policy_lift_over_raw_world_model",
                "phase10_calibrated_policy_lift_over_retained_hybrid",
                "phase10_calibrated_policy_vs_original_v3_lift_over_hybrid",
                "phase10_calibrated_policy_harm_vs_hybrid_count",
                "phase10_calibrated_policy_win_vs_hybrid_count",
                "phase10_calibrated_policy_override_count",
                "phase5_keeps_phase10_candidate",
                "uses_raw_prompts_or_answers",
            ],
        ),
        "phase11_capability_audit": _metric_subset(
            artifacts["v3_phase11_capability_audit"].get("metrics", {}),
            [
                "capability_count",
                "artifact_pass_rate",
                "outer_shell_count",
                "outer_shell_production_claim_count",
                "live_or_live_derived_count",
                "shadow_or_fixture_count",
                "blocked_claim_count",
                "promotion_requirement_count",
                "phase4_status",
                "phase7_status",
                "phase10_status",
                "phase10_calibration_status",
            ],
        ),
    }
    metrics = _metrics(artifacts=artifacts, evidence=evidence)
    gates = {
        "all_required_artifacts_exist_and_pass": metrics["required_artifact_pass_rate"] == 1.0,
        "first_party_live_trace_scale_large": metrics["raw_first_party_live_event_count"] >= 6000,
        "judge_event_scale_large": metrics["valid_judge_event_count"] >= 2500,
        "problem_level_main_n_large": metrics["main_problem_level_n"] >= 100,
        "base_ci_lower_above_half": metrics["structural_vs_base_ci_lower"] > 0.50,
        "base_sign_test_significant": metrics["structural_vs_base_p_value"] < 0.05,
        "placebo_ci_lower_strong": metrics["structural_vs_placebo_ci_lower"] > 0.60,
        "placebo_sign_test_significant": metrics["structural_vs_placebo_p_value"] < 0.001,
        "retrieval_baseline_margin_large": metrics["retrieval_margin_over_best_baseline"] >= 0.70,
        "key_toggle_margin_positive": metrics["key_toggle_min_margin"] >= 0.05,
        "v3_mechanism_artifacts_all_pass": metrics["v3_mechanism_pass_rate"] == 1.0,
        "phase0_production_contract_blocks_invalid_overlay": (
            metrics["phase0_production_contract_invalid_admitted_count"] == 0
            and metrics["phase0_production_contract_applied_count"] == 1
        ),
        "phase1_production_sleep_job_writes_consolidated": (
            metrics["phase1_production_sleep_applied_consolidated_node_count"] >= 1
            and metrics["phase1_production_sleep_dry_run_mutated"] is False
        ),
        "phase1_first_party_retrieval_audit_improves": (
            metrics["phase1_retrieval_precision_delta"] >= 0.20
            and metrics["phase1_retrieval_negative_transfer_delta"] >= 2
            and metrics["phase1_retrieval_after_archived_hits"] == 0
        ),
        "vertical_slice_compose_passes": bool(artifacts["vertical_slice"].get("pass")),
        "long_run_pairwise_downstream_positive": metrics["long_run_downstream_win_rate"] >= 0.65,
        "phase7_production_queue_sources_loaded": metrics["phase7_production_queue_source_count"] >= 2,
        "phase7_production_queue_plans_leaves": metrics["phase7_production_planned_leaf_count"] >= 2,
        "phase7_production_pre_live_screen_saves_budget": (
            metrics["phase7_production_pre_live_block_or_defer_count"] >= 2
        ),
        "phase7_production_manifests_reopen": metrics["phase7_production_manifest_reopen_count"] >= 4,
        "phase7_production_no_graph_mutation_without_apply": metrics["phase7_production_node_mutation_count"] == 0,
        "phase7_production_apply_execute_gates_closed": (
            metrics["phase7_production_apply_enabled_count"] == 0
            and metrics["phase7_production_execute_enabled_count"] == 0
        ),
        "phase7_daemon_soak_checkpointed": (
            metrics["phase7_soak_cycle_count"] >= 3
            and metrics["phase7_soak_checkpoint_reopen_success_rate"] == 1.0
            and metrics["phase7_soak_node_mutation_count"] == 0
        ),
        "phase7_daemon_soak_gates_closed": (
            metrics["phase7_soak_apply_enabled_count"] == 0
            and metrics["phase7_soak_execute_enabled_count"] == 0
            and metrics["phase7_soak_continuous_background_daemon"] is False
        ),
        "frozen_v3_beats_v1_kernel": metrics["full_v3_margin_vs_v1_kernel"] >= 0.10,
        "frozen_v3_beats_best_nonfull": metrics["full_v3_margin_vs_best_nonfull"] >= 0.08,
        "fresh_live_guarded_300_problem_level": metrics["fresh_live_guarded_problem_level_n"] >= 300,
        "fresh_live_guarded_300_positive_vs_base": metrics["fresh_live_guarded_vs_base_utility"] > 0.50,
        "fresh_live_guarded_300_positive_vs_placebo": metrics["fresh_live_guarded_vs_placebo_utility"] > 0.50,
        "fresh_live_guarded_300_low_call_budget": metrics["fresh_live_guarded_planned_total_calls"] <= 100,
        "fresh_live_guarded_full_remaining_problem_level": metrics["fresh_live_full_problem_level_n"] >= 500,
        "fresh_live_guarded_full_remaining_active_count": metrics["fresh_live_full_active_intervention_n"] >= 20,
        "fresh_live_guarded_full_remaining_positive_vs_base": metrics["fresh_live_full_vs_base_utility"] > 0.50,
        "fresh_live_guarded_full_remaining_positive_vs_placebo": metrics["fresh_live_full_vs_placebo_utility"] > 0.50,
        "fresh_live_guarded_full_remaining_low_call_budget": metrics["fresh_live_full_planned_total_calls"] <= 150,
        "fresh_live_selective_expansion_increases_active_coverage": (
            metrics["fresh_live_selective_active_intervention_n"] > metrics["fresh_live_full_active_intervention_n"]
        ),
        "fresh_live_selective_expansion_improves_base": (
            metrics["fresh_live_selective_vs_base_utility"] > metrics["fresh_live_full_vs_base_utility"]
        ),
        "fresh_live_selective_expansion_improves_placebo": (
            metrics["fresh_live_selective_vs_placebo_utility"] > metrics["fresh_live_full_vs_placebo_utility"]
        ),
        "fresh_live_selective_placebo_ci_above_half": metrics["fresh_live_selective_vs_placebo_ci_lower"] > 0.50,
        "fresh_live_selective_low_call_budget": metrics["fresh_live_selective_planned_total_calls"] <= 200,
        "phase8_creative_generator_nonlocal": metrics["phase8_nonlocal_candidate_ratio"] >= 0.35,
        "phase8_world_model_profile_selector_passes": metrics["phase8_quality_world_model_auroc"] >= 0.85,
        "phase8_coverage_profile_positive": metrics["phase8_coverage_profile_vs_base_utility"] > 0.50
        and metrics["phase8_coverage_profile_vs_placebo_utility"] > 0.50,
        "live_residual_clusterer_passes": metrics["live_residual_cluster_count"] >= 25,
        "live_residual_clusterer_covers_weighted_residuals": metrics[
            "live_residual_systematic_weighted_coverage"
        ] >= 0.85,
        "live_residual_clusterer_resolves_largest_live_cluster": (
            metrics["live_residual_largest_cluster_status"] == "resolved_by_phase9_hybrid_guard"
        ),
        "live_residual_clusterer_emits_next_seeds": metrics["live_residual_next_generation_seed_count"] >= 15,
        "live_residual_clusterer_redacted": metrics["live_residual_uses_raw_prompts_or_answers"] is False,
        "residual_multigeneration_loop_retains_descendants": (
            metrics["residual_multigen_generation_count"] >= 3
            and metrics["residual_multigen_retained_count"] >= 12
            and metrics["residual_multigen_recursive_parent_closure_rate"] == 1.0
        ),
        "residual_multigeneration_loop_is_gated": (
            metrics["residual_multigen_graph_mutation_count"] == 0
            and metrics["residual_multigen_uses_raw_prompts_or_answers"] is False
        ),
        "residual_live_mini_loop_applies_to_shadow_graph_only": (
            metrics["residual_live_mini_selected_candidate_count"] >= 3
            and metrics["residual_live_mini_contract_ready_count"] == metrics["residual_live_mini_selected_candidate_count"]
            and metrics["residual_live_mini_accepted_count"] == metrics["residual_live_mini_selected_candidate_count"]
            and metrics["residual_live_mini_applied_count"] == metrics["residual_live_mini_selected_candidate_count"]
            and metrics["residual_live_mini_graph_copy_node_delta"] == metrics["residual_live_mini_selected_candidate_count"]
            and metrics["residual_live_mini_main_graph_mutation_count"] == 0
            and metrics["residual_live_mini_new_api_call_count"] == 0
        ),
        "residual_live_mini_loop_recalibrates_phase10_readback": (
            metrics["residual_live_mini_phase10_readback_pass"] is True
            and metrics["residual_live_mini_phase10_leave_pattern_guard_harm_count"] == 0
            and metrics["residual_live_mini_phase10_leave_route_guard_harm_count"] == 0
        ),
        "residual_fresh_live_loop_path_is_ready": (
            metrics["residual_fresh_live_path_present"] is True
            and metrics["residual_fresh_contract_ready_count"] >= 3
            and metrics["residual_fresh_preflight_ready_count"] >= 3
            and metrics["residual_fresh_planned_api_call_count"] >= 18
            and metrics["residual_fresh_main_graph_mutation_count"] == 0
            and metrics["residual_fresh_secret_value_exposed"] is False
        ),
        "phase5_live_scheduler_realified": metrics["phase5_live_profile_count"] >= 7,
        "phase5_live_scheduler_selects_calibrated_guard": (
            metrics["phase5_live_selected_production_profile"] == "phase10_calibrated_residual_guard"
        ),
        "phase5_live_scheduler_improves_v3": metrics["phase5_live_scheduler_lift_over_v3"] >= 0.07,
        "phase5_live_scheduler_improves_hybrid": (
            metrics["phase5_live_calibrated_guard_lift_over_hybrid"] > 0.0
        ),
        "phase5_live_scheduler_keeps_phase10_candidate": metrics["phase5_live_keeps_phase10_candidate"] is True,
        "phase5_live_scheduler_blocks_compact_default": metrics["phase5_live_blocks_compact_default"] is True,
        "phase5_live_scheduler_redacted": metrics["phase5_live_uses_raw_prompts_or_answers"] is False,
        "phase9_compact_guard_v1_regression_passes": metrics["phase9_compact_guard_vs_v1_margin"] >= 0.10,
        "phase9_compact_guard_improves_original_v3_vs_v1": metrics["phase9_compact_guard_margin_gain_over_v3"] > 0.05,
        "phase9_compact_guard_noninferior_to_original_v3": metrics["phase9_compact_guard_vs_v3_utility"] >= 0.48,
        "phase9_hybrid_guard_heldout_slice_large": metrics["phase9_hybrid_guard_heldout_n"] >= 50,
        "phase9_hybrid_guard_v1_regression_passes": metrics["phase9_hybrid_guard_vs_v1_margin"] >= 0.10,
        "phase9_hybrid_guard_improves_original_v3_vs_v1": metrics["phase9_hybrid_guard_lift_over_v3"] > 0.03,
        "phase9_hybrid_guard_noninferior_to_original_v3": metrics["phase9_hybrid_guard_vs_v3_utility"] >= 0.50,
        "phase10_discrete_world_model_candidate_positive": metrics["phase10_world_model_all_lift_over_v3"] >= 0.015,
        "phase10_discrete_world_model_candidate_v1_positive": (
            metrics["phase10_world_model_candidate_v1_lift_over_v3"] > 0.04
        ),
        "phase10_calibrated_guard_beats_hybrid": (
            metrics["phase10_world_model_calibrated_lift_over_hybrid"] > 0.0
        ),
        "phase10_leave_pattern_and_route_out_nonharmful": (
            metrics["phase10_leave_pattern_group_count"] >= 2
            and metrics["phase10_leave_route_tag_group_count"] >= 2
            and metrics["phase10_leave_pattern_raw_harm_count"] >= 1
            and metrics["phase10_leave_route_tag_raw_harm_count"] >= 1
            and metrics["phase10_leave_pattern_guard_harm_count"] == 0
            and metrics["phase10_leave_route_tag_guard_harm_count"] == 0
        ),
        "phase10_guard_rules_are_graph_assumptions": (
            metrics["phase10_guard_assumption_node_count"] >= 7
            and metrics["phase10_calibrated_rows_with_guard_assumption_rate"] == 1.0
        ),
        "phase10_reliability_calibration_promotes_budget_gate": (
            metrics["phase10_reliability_source_raw_beats_base"] is False
            and metrics["phase10_reliability_calibrated_mae"] < metrics["phase10_reliability_base_rate_mae"]
            and metrics["phase10_reliability_calibrated_mae_lift_over_base"] > 0.02
            and metrics["phase10_reliability_calibrated_brier_lift_over_base"] > 0.01
            and metrics["phase10_reliability_calibrated_ece_lift_over_raw"] > 0.0
        ),
        "guard_policy_learning_nonharmful": (
            metrics["guard_policy_learned_update_count"] >= 7
            and metrics["guard_policy_supported_guard_count"] >= 5
            and metrics["guard_policy_weight_range"] >= 0.05
            and metrics["guard_policy_learned_lift_over_hybrid"] >= 0.0
            and metrics["guard_policy_harm_vs_hybrid_count"] == 0
            and metrics["guard_policy_raw_world_model_status"] == "candidate"
        ),
        "phase10_raw_predictor_still_marked_uncalibrated": (
            metrics["phase10_world_model_calibration_beats_base_rate"] is False
        ),
        "phase10_promotion_is_guarded": (
            metrics["phase10_world_model_recommended_promotion"] == "promote_calibrated_residual_guard"
        ),
        "phase3_learned_rollout_positive": (
            metrics["phase3_learned_transition_row_count"] >= 80
            and metrics["phase3_learned_selected_reward_lift_over_v3"] >= 0.04
            and metrics["phase3_learned_uses_raw_prompts_or_answers"] is False
        ),
        "same_batch_ablation_suite_records_raw_and_guarded": (
            metrics["same_batch_toggle_pair_count"] == 4
            and metrics["same_batch_raw_v3_vs_v1_ci_lower"] < 0.50
            and metrics["same_batch_calibrated_lift_over_hybrid"] > 0.0
            and metrics["same_batch_calibrated_harm_vs_hybrid_count"] == 0
        ),
        "world_model_calibration_artifact_passes": metrics["world_model_calibration_surface_count"] >= 4,
        "world_model_calibrated_surface_available": metrics["world_model_calibrated_surface_count"] >= 3,
        "world_model_leave_domain_out_records_boundary": (
            metrics["world_model_leave_domain_out_domain_count"] >= 3
            and metrics["world_model_leave_domain_out_nonnegative_domain_count"]
            < metrics["world_model_leave_domain_out_domain_count"]
        ),
        "world_model_blocks_uncalibrated_phase10_promotion": (
            metrics["world_model_uncalibrated_promotion_count"] == 0
            and metrics["world_model_phase10_calibration_beats_base_rate"] is False
        ),
        "world_model_phase10_positive_but_unpromoted": (
            metrics["world_model_phase10_all_lift_over_v3"] >= 0.015
            and metrics["world_model_phase5_keeps_phase10_candidate"] is True
        ),
        "world_model_guarded_policy_promoted": (
            metrics["world_model_phase10_calibrated_lift_over_hybrid"] > 0.0
            and metrics["world_model_phase10_calibrated_harm_vs_hybrid_count"] == 0
        ),
        "world_model_calibration_redacted": metrics["world_model_uses_raw_prompts_or_answers"] is False,
        "main_graph_memory_shadow_improves_retrieval_without_mutation": (
            metrics["main_graph_memory_shadow_precision_delta"] >= 0.10
            and metrics["main_graph_memory_shadow_archive_exposure_after"] == 0
            and metrics["main_graph_memory_shadow_memory_hit_delta"] > 0
            and metrics["main_graph_memory_shadow_context_efficiency_delta"] > 0.02
            and metrics["main_graph_memory_shadow_main_graph_mutated"] is False
        ),
        "main_graph_memory_controlled_apply_ready": (
            metrics["main_graph_memory_controlled_apply_rollback_entry_count"]
            >= metrics["main_graph_memory_controlled_apply_planned_archive_count"]
            and metrics["main_graph_memory_controlled_apply_consolidated_count"] >= 4
            and metrics["main_graph_memory_controlled_apply_precision_delta"] > 0.10
            and metrics["main_graph_memory_controlled_apply_archive_exposure_after"] == 0
            and metrics["main_graph_memory_controlled_apply_context_efficiency_delta"] > 0.02
            and metrics["main_graph_memory_controlled_apply_main_graph_mutated"] is False
        ),
        "continuous_daemon_scheduler_ready": (
            metrics["continuous_daemon_scheduled_cycle_count"] >= 10
            and metrics["continuous_daemon_checkpoint_pair_count"] == metrics["continuous_daemon_scheduled_cycle_count"]
            and metrics["continuous_daemon_rate_limit_violation_count"] == 0
            and metrics["continuous_daemon_recovery_action_count"] >= 2
            and metrics["continuous_daemon_fresh_loop_queue_integrated"] is True
            and metrics["continuous_daemon_memory_apply_queue_integrated"] is True
            and metrics["continuous_daemon_ungated_graph_mutation_count"] == 0
            and metrics["continuous_daemon_background_ready"] is True
            and metrics["continuous_daemon_background_process_started"] is False
        ),
        "phase11_capability_audit_passes": metrics["phase11_artifact_pass_rate"] == 1.0,
        "phase11_outer_shells_not_overclaimed": metrics["phase11_outer_shell_production_claim_count"] == 0,
        "phase11_phase7_bounded_daemon_recorded": (
            metrics["phase11_phase7_status"] == "bounded_production_queue_daemon_not_unbounded_background"
        ),
        "phase11_phase10_guard_promoted_raw_candidate": (
            metrics["phase11_phase10_status"] == "calibrated_guard_promoted_raw_predictor_candidate"
        ),
        "phase11_world_model_calibration_status_recorded": (
            metrics["phase11_phase10_calibration_status"] == "calibration_audit_promotes_guard_blocks_raw_predictor"
        ),
        "prompt_answer_and_secret_free": metrics["prompt_answer_payload_stored"] is False and metrics["secret_leak_detected"] is False,
        "boundary_cases_recorded": metrics["boundary_case_count"] >= 1,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_paper_scale_evidence_aggregation",
        "performance_validation": True,
        "validation_scope": (
            "Aggregates existing paper-facing first-party live traces, problem-level bootstrap statistics, "
            "retrieval/toggle baselines, full-v3 mechanism validations, and the recursive vertical slice. "
            "No new API calls are made by this module."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass", False)),
                "eval_kind": artifacts[name].get("eval_kind"),
            }
            for name, path in REQUIRED_ARTIFACTS.items()
        },
        "evidence": evidence,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The v3 mechanisms are now supported by a single paper-scale evidence table: 100-problem "
            "problem-level live/cached main statistics with bootstrap CIs, 6400+ first-party live events, "
            "2500+ judge events, hard retrieval/toggle baselines, full-v3 mechanism validations, a fresh "
            "guarded heldout-300 live rerun, a guarded full-remaining live rerun, and a selective clade-expansion "
            "full-remaining rerun.  Phase9 additionally records the failed broad/compact tradeoff and the retained "
            "cue-level hybrid guard that beats V1 on the heldout slice without regressing against original V3.  The "
            "live residual clusterer now unifies formal, live, creative, and profile-level residual evidence and emits "
            "next-generation proposal seeds while marking the largest same-batch residual as resolved by Phase9.  Phase5 "
            "now adds a live-derived contextual scheduler that selects the Phase10 calibrated residual guard, keeps the "
            "raw world-model selector as exploration, and blocks over-structured compact framing as a default.  Phase7 "
            "now validates the real bounded daemon path over committed preflight queues, with manifests, pre-live "
            "budget screening, and closed execute/apply gates.  The "
            "world-model calibration audit now records the useful but uncalibrated raw Phase10 selector as "
            "exploration-only, promotes only the bounded calibrated residual guard, preserves the calibrated Phase8 "
            "profile selector, and reports the Phase9 leave-domain-out business boundary instead of overclaiming a simulator.  The "
            "fresh reruns are positive but intentionally reported as small-effect safety/abstention validations, "
            "not as the main paper claim."
        ),
    }


def _paper_main_summary(payload: dict[str, Any]) -> dict[str, Any]:
    main = payload.get("main_results", {})
    return {
        pair: _metric_subset(
            row,
            ["problem_level_n", "outcomes", "utility", "win_rate", "loss_rate", "bootstrap_ci_95", "sign_test"],
        )
        for pair, row in main.items()
    }


def _baseline_hardening_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    boundary = []
    for row in payload.get("baseline_rows", []):
        pairs = row.get("pairs", {})
        margins = {
            pair_name: pair.get("final_minus_toggle_utility")
            for pair_name, pair in pairs.items()
        }
        item = {
            "baseline": row.get("baseline"),
            "source_kind": row.get("source_kind"),
            "problem_count": row.get("problem_count"),
            "pass": row.get("pass"),
            "margins": margins,
        }
        rows.append(item)
        if any(value is not None and value < 0 for value in margins.values()):
            boundary.append(item)
    key = [
        row for row in rows
        if row["baseline"] in KEY_TOGGLE_BASELINES
    ]
    return {
        "key_toggle_rows": key,
        "boundary_rows": boundary,
        "all_rows": rows,
    }


def _retrieval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_rates": payload.get("hit_rates", {}),
        "morphism_margin_over_best_retrieval": payload.get("morphism_margin_over_best_retrieval"),
        "neural_embedding_baseline": payload.get("neural_embedding_baseline"),
    }


def _world_model_scale_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return _metric_subset(
        payload,
        [
            "raw_first_party_trainable_row_count",
            "raw_first_party_live_event_count",
            "valid_judge_event_count",
            "solver_event_count",
            "source_run_count",
            "distinct_problem_count",
            "prompt_answer_payload_stored",
            "secret_leak_detected",
            "calibration",
        ],
    )


def _fresh_live_summary(payload: dict[str, Any]) -> dict[str, Any]:
    ci = payload.get("problem_level_ci", {}).get("pairs", {})
    return {
        "selection_mode": payload.get("metrics", {}).get("selection_mode"),
        "sample_problem_count": payload.get("metrics", {}).get("sample_problem_count"),
        "selected_case_count": payload.get("metrics", {}).get("selected_case_count"),
        "planned_total_model_calls": payload.get("metrics", {}).get("planned_total_model_calls"),
        "abstained_problems_count_as_tie": payload.get("metrics", {}).get("abstained_problems_count_as_tie"),
        "structural_vs_base": _metric_subset(
            ci.get("structural_vs_base", {}),
            [
                "problem_level_n",
                "active_intervention_n",
                "outcomes",
                "utility",
                "bootstrap_ci_95",
                "sign_test",
            ],
        ),
        "structural_vs_placebo": _metric_subset(
            ci.get("structural_vs_placebo", {}),
            [
                "problem_level_n",
                "active_intervention_n",
                "outcomes",
                "utility",
                "bootstrap_ci_95",
                "sign_test",
            ],
        ),
    }


def _v3_mechanism_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "v3_phase0_contract",
        "v3_phase1_memory",
        "v3_phase1_retrieval_audit",
        "v3_phase2_verifier",
        "v3_phase3_rollout",
        "v3_phase3_learned_rollout",
        "v3_phase4_generator",
        "v3_live_residual_clusterer",
        "v3_residual_multigeneration_loop",
        "v3_residual_live_mini_loop",
        "v3_phase5_bandit",
        "v3_phase6_formal",
        "v3_phase7_long_run",
        "v3_phase7_daemon_soak",
        "v3_phase10_discrete_world_model",
        "v3_phase10_reliability_calibration",
        "v3_guard_policy_learning",
        "v3_main_graph_memory_shadow",
        "v3_main_graph_memory_controlled_apply",
        "v3_residual_fresh_live_loop",
        "v3_continuous_daemon_scheduler",
        "v3_world_model_calibration",
        "v3_same_batch_ablation_suite",
        "v3_phase11_capability_audit",
    ]
    return {
        key: {
            "pass": bool(artifacts[key].get("pass")),
            "eval_kind": artifacts[key].get("eval_kind"),
            "metrics": artifacts[key].get("metrics", {}),
        }
        for key in keys
    }


def _metrics(*, artifacts: dict[str, dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    main = artifacts["paper_main"]["main_results"]
    base = main["structural_vs_base"]
    placebo = main["structural_vs_placebo"]
    baseline_summary = evidence["baseline_hardening"]
    key_margins = [
        margin
        for row in baseline_summary["key_toggle_rows"]
        for margin in row["margins"].values()
        if margin is not None
    ]
    v3_passes = [row["pass"] for row in evidence["v3_mechanism"].values()]
    return {
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "required_artifact_pass_rate": round(
            sum(1 for payload in artifacts.values() if bool(payload.get("pass"))) / max(1, len(artifacts)),
            4,
        ),
        "v3_mechanism_count": len(v3_passes),
        "v3_mechanism_pass_rate": round(sum(1 for value in v3_passes if value) / max(1, len(v3_passes)), 4),
        "raw_first_party_live_event_count": int(artifacts["first_party_world_model_scale"].get("raw_first_party_live_event_count") or 0),
        "valid_judge_event_count": int(artifacts["first_party_world_model_scale"].get("valid_judge_event_count") or 0),
        "main_problem_level_n": int(base["problem_level_n"]),
        "structural_vs_base_utility": base["utility"],
        "structural_vs_base_ci_lower": base["bootstrap_ci_95"]["lower"],
        "structural_vs_base_p_value": base["sign_test"]["p_value"],
        "structural_vs_placebo_utility": placebo["utility"],
        "structural_vs_placebo_ci_lower": placebo["bootstrap_ci_95"]["lower"],
        "structural_vs_placebo_p_value": placebo["sign_test"]["p_value"],
        "retrieval_margin_over_best_baseline": artifacts["retrieval_baselines"].get("morphism_margin_over_best_retrieval"),
        "key_toggle_min_margin": round(min(key_margins), 4) if key_margins else None,
        "key_toggle_mean_margin": round(sum(key_margins) / max(1, len(key_margins)), 4) if key_margins else None,
        "boundary_case_count": len(baseline_summary["boundary_rows"]),
        "phase0_production_contract_proposal_count": int(
            artifacts["v3_phase0_contract"]["metrics"]["production_contract_proposal_count"]
        ),
        "phase0_production_contract_admitted_count": int(
            artifacts["v3_phase0_contract"]["metrics"]["production_contract_admitted_count"]
        ),
        "phase0_production_contract_quarantined_count": int(
            artifacts["v3_phase0_contract"]["metrics"]["production_contract_quarantined_count"]
        ),
        "phase0_production_contract_invalid_admitted_count": int(
            artifacts["v3_phase0_contract"]["metrics"]["production_contract_invalid_admitted_count"]
        ),
        "phase0_production_contract_applied_count": int(
            artifacts["v3_phase0_contract"]["metrics"]["production_contract_applied_count"]
        ),
        "phase1_production_sleep_group_count": int(
            artifacts["v3_phase1_memory"]["metrics"]["production_sleep_group_count"]
        ),
        "phase1_production_sleep_planned_archive_count": int(
            artifacts["v3_phase1_memory"]["metrics"]["production_sleep_planned_archive_count"]
        ),
        "phase1_production_sleep_applied_archived_node_count": int(
            artifacts["v3_phase1_memory"]["metrics"]["production_sleep_applied_archived_node_count"]
        ),
        "phase1_production_sleep_applied_consolidated_node_count": int(
            artifacts["v3_phase1_memory"]["metrics"]["production_sleep_applied_consolidated_node_count"]
        ),
        "phase1_production_sleep_dry_run_mutated": bool(
            artifacts["v3_phase1_memory"]["metrics"]["production_sleep_dry_run_mutated"]
        ),
        "phase1_retrieval_query_count": int(
            artifacts["v3_phase1_retrieval_audit"]["metrics"]["query_count"]
        ),
        "phase1_retrieval_precision_delta": float(
            artifacts["v3_phase1_retrieval_audit"]["metrics"]["precision_delta"]
        ),
        "phase1_retrieval_negative_transfer_delta": int(
            artifacts["v3_phase1_retrieval_audit"]["metrics"]["negative_transfer_delta"]
        ),
        "phase1_retrieval_context_efficiency_delta": float(
            artifacts["v3_phase1_retrieval_audit"]["metrics"]["context_efficiency_delta"]
        ),
        "phase1_retrieval_after_archived_hits": int(
            artifacts["v3_phase1_retrieval_audit"]["metrics"]["after_archived_hits"]
        ),
        "vertical_slice_downstream_delta": artifacts["vertical_slice"]["metrics"]["downstream_score_delta"],
        "vertical_slice_brier_improvement": artifacts["vertical_slice"]["metrics"]["world_model_brier_improvement"],
        "long_run_downstream_win_rate": artifacts["v3_phase7_long_run"]["metrics"]["downstream_win_rate_on_unseen"],
        "long_run_capability_improvement": artifacts["v3_phase7_long_run"]["metrics"]["capability_score_improvement"],
        "phase7_production_queue_source_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_queue_source_count"]
        ),
        "phase7_production_ready_queue_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_ready_queue_count"]
        ),
        "phase7_production_planned_leaf_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_planned_leaf_count"]
        ),
        "phase7_production_executable_leaf_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_executable_leaf_count"]
        ),
        "phase7_production_screened_leaf_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_screened_leaf_count"]
        ),
        "phase7_production_pre_live_block_or_defer_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_pre_live_block_or_defer_count"]
        ),
        "phase7_production_manifest_reopen_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_manifest_reopen_count"]
        ),
        "phase7_production_node_mutation_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_node_mutation_count"]
        ),
        "phase7_production_apply_enabled_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_apply_enabled_count"]
        ),
        "phase7_production_execute_enabled_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_execute_enabled_count"]
        ),
        "phase7_production_rate_limit_violation_count": int(
            artifacts["v3_phase7_long_run"]["metrics"]["production_rate_limit_violation_count"]
        ),
        "phase7_soak_cycle_count": int(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["cycle_count"]
        ),
        "phase7_soak_planned_leaf_count": int(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["planned_leaf_count"]
        ),
        "phase7_soak_manifest_reopen_count": int(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["manifest_reopen_count"]
        ),
        "phase7_soak_checkpoint_reopen_success_rate": float(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["checkpoint_reopen_success_rate"]
        ),
        "phase7_soak_node_mutation_count": int(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["node_mutation_count"]
        ),
        "phase7_soak_apply_enabled_count": int(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["apply_enabled_count"]
        ),
        "phase7_soak_execute_enabled_count": int(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["execute_enabled_count"]
        ),
        "phase7_soak_continuous_background_daemon": bool(
            artifacts["v3_phase7_daemon_soak"]["metrics"]["continuous_background_daemon"]
        ),
        "full_v3_margin_vs_v1_kernel": artifacts["frozen_v3_vs_v1"]["metrics"]["full_v3_margin_vs_v1_kernel"],
        "full_v3_margin_vs_hipporag_style": artifacts["frozen_v3_vs_v1"]["metrics"]["full_v3_margin_vs_hipporag_style"],
        "full_v3_margin_vs_best_nonfull": artifacts["frozen_v3_vs_v1"]["metrics"]["full_v3_margin_vs_best_nonfull"],
        "fresh_live_guarded_problem_level_n": int(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_problem_level_n"]
        ),
        "fresh_live_guarded_active_intervention_n": int(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_active_intervention_n"]
        ),
        "fresh_live_guarded_vs_base_utility": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_utility"]
        ),
        "fresh_live_guarded_vs_base_ci_lower": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_base_ci_lower"]
        ),
        "fresh_live_guarded_vs_placebo_utility": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_placebo_utility"]
        ),
        "fresh_live_guarded_vs_placebo_ci_lower": float(
            artifacts["fresh_live_guarded_300"]["metrics"]["structural_vs_placebo_ci_lower"]
        ),
        "fresh_live_guarded_planned_total_calls": int(
            artifacts["fresh_live_guarded_300"]["metrics"]["planned_total_model_calls"]
        ),
        "fresh_live_full_problem_level_n": int(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_problem_level_n"]
        ),
        "fresh_live_full_active_intervention_n": int(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_active_intervention_n"]
        ),
        "fresh_live_full_vs_base_utility": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_utility"]
        ),
        "fresh_live_full_vs_base_ci_lower": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_base_ci_lower"]
        ),
        "fresh_live_full_vs_placebo_utility": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_placebo_utility"]
        ),
        "fresh_live_full_vs_placebo_ci_lower": float(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["structural_vs_placebo_ci_lower"]
        ),
        "fresh_live_full_planned_total_calls": int(
            artifacts["fresh_live_guarded_full_remaining"]["metrics"]["planned_total_model_calls"]
        ),
        "fresh_live_selective_problem_level_n": int(
            artifacts["fresh_live_selective_expansion"]["metrics"]["structural_vs_base_problem_level_n"]
        ),
        "fresh_live_selective_active_intervention_n": int(
            artifacts["fresh_live_selective_expansion"]["metrics"]["structural_vs_base_active_intervention_n"]
        ),
        "fresh_live_selective_vs_base_utility": float(
            artifacts["fresh_live_selective_expansion"]["metrics"]["structural_vs_base_utility"]
        ),
        "fresh_live_selective_vs_base_ci_lower": float(
            artifacts["fresh_live_selective_expansion"]["metrics"]["structural_vs_base_ci_lower"]
        ),
        "fresh_live_selective_vs_placebo_utility": float(
            artifacts["fresh_live_selective_expansion"]["metrics"]["structural_vs_placebo_utility"]
        ),
        "fresh_live_selective_vs_placebo_ci_lower": float(
            artifacts["fresh_live_selective_expansion"]["metrics"]["structural_vs_placebo_ci_lower"]
        ),
        "fresh_live_selective_planned_total_calls": int(
            artifacts["fresh_live_selective_expansion"]["metrics"]["planned_total_model_calls"]
        ),
        "phase8_creative_candidate_count": int(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["creative_candidate_count"]
        ),
        "phase8_nonlocal_candidate_ratio": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["nonlocal_candidate_ratio"]
        ),
        "phase8_residual_cluster_coverage": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["residual_cluster_coverage"]
        ),
        "phase8_quality_world_model_auroc": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["quality_world_model_auroc"]
        ),
        "phase8_quality_world_model_brier": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["quality_world_model_brier"]
        ),
        "phase8_quality_base_rate_brier": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["quality_base_rate_brier"]
        ),
        "phase8_selected_quality_profile_id": artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["selected_quality_profile_id"],
        "phase8_selected_coverage_profile_id": artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["selected_coverage_profile_id"],
        "phase8_coverage_profile_active_gain_over_quality": int(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["coverage_profile_active_gain_over_quality"]
        ),
        "phase8_coverage_profile_vs_base_utility": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["coverage_profile_vs_base_utility"]
        ),
        "phase8_coverage_profile_vs_placebo_utility": float(
            artifacts["v3_phase8_creativity_world_coverage"]["metrics"]["coverage_profile_vs_placebo_utility"]
        ),
        "phase3_learned_transition_row_count": int(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["transition_row_count"]
        ),
        "phase3_learned_rollout_row_count": int(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["rollout_row_count"]
        ),
        "phase3_learned_candidate_action_coverage": float(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["candidate_action_coverage"]
        ),
        "phase3_learned_selected_reward_lift_over_v3": float(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["selected_reward_lift_over_v3"]
        ),
        "phase3_learned_selected_vs_v1_lift_over_v3": float(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["selected_vs_v1_lift_over_v3"]
        ),
        "phase3_learned_selected_vs_v1_utility": float(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["selected_vs_v1_utility"]
        ),
        "phase3_learned_teacher_match_rate": float(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["teacher_match_rate"]
        ),
        "phase3_learned_uses_raw_prompts_or_answers": bool(
            artifacts["v3_phase3_learned_rollout"]["metrics"]["uses_raw_prompts_or_answers"]
        ),
        "live_residual_source_artifact_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["source_artifact_count"]
        ),
        "live_residual_observation_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["observation_count"]
        ),
        "live_weighted_residual_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["weighted_residual_count"]
        ),
        "live_residual_cluster_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["cluster_count"]
        ),
        "live_residual_systematic_weighted_coverage": float(
            artifacts["v3_live_residual_clusterer"]["metrics"]["systematic_weighted_coverage"]
        ),
        "live_residual_phase9_observation_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["phase9_live_residual_observation_count"]
        ),
        "live_residual_formal_observation_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["formal_residual_observation_count"]
        ),
        "live_residual_profile_observation_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["profile_residual_observation_count"]
        ),
        "live_residual_resolved_cluster_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["resolved_cluster_count"]
        ),
        "live_residual_blocked_profile_residual_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["blocked_profile_residual_count"]
        ),
        "live_residual_next_generation_seed_count": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["next_generation_proposal_seed_count"]
        ),
        "live_residual_largest_cluster_support": int(
            artifacts["v3_live_residual_clusterer"]["metrics"]["largest_live_cluster_support"]
        ),
        "live_residual_largest_cluster_axis": artifacts["v3_live_residual_clusterer"]["metrics"][
            "largest_live_cluster_axis"
        ],
        "live_residual_largest_cluster_domain": artifacts["v3_live_residual_clusterer"]["metrics"][
            "largest_live_cluster_domain"
        ],
        "live_residual_largest_cluster_pattern": artifacts["v3_live_residual_clusterer"]["metrics"][
            "largest_live_cluster_pattern"
        ],
        "live_residual_largest_cluster_status": artifacts["v3_live_residual_clusterer"]["metrics"][
            "largest_live_cluster_status"
        ],
        "live_residual_uses_raw_prompts_or_answers": bool(
            artifacts["v3_live_residual_clusterer"]["metrics"]["uses_raw_prompts_or_answers"]
        ),
        "residual_multigen_generation_count": int(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["generation_count"]
        ),
        "residual_multigen_seed_cluster_count": int(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["seed_cluster_count"]
        ),
        "residual_multigen_proposal_count": int(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["proposal_count"]
        ),
        "residual_multigen_retained_count": int(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["retained_count"]
        ),
        "residual_multigen_retention_rate": float(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["retention_rate"]
        ),
        "residual_multigen_proposal_family_count": int(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["proposal_family_count"]
        ),
        "residual_multigen_recursive_parent_closure_rate": float(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["recursive_parent_closure_rate"]
        ),
        "residual_multigen_uses_raw_prompts_or_answers": bool(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["uses_raw_prompts_or_answers"]
        ),
        "residual_multigen_graph_mutation_count": int(
            artifacts["v3_residual_multigeneration_loop"]["metrics"]["graph_mutation_count"]
        ),
        "residual_live_mini_source_generation_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["source_generation_count"]
        ),
        "residual_live_mini_source_retained_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["source_retained_count"]
        ),
        "residual_live_mini_selected_candidate_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["selected_candidate_count"]
        ),
        "residual_live_mini_contract_ready_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["contract_ready_count"]
        ),
        "residual_live_mini_preflight_ready_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["preflight_ready_count"]
        ),
        "residual_live_mini_accepted_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["accepted_count"]
        ),
        "residual_live_mini_applied_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["applied_count"]
        ),
        "residual_live_mini_applied_active_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["applied_active_count"]
        ),
        "residual_live_mini_graph_copy_node_delta": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["graph_copy_node_delta"]
        ),
        "residual_live_mini_main_graph_mutation_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["main_graph_mutation_count"]
        ),
        "residual_live_mini_new_api_call_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["new_api_call_count"]
        ),
        "residual_live_mini_phase10_readback_pass": bool(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["phase10_readback_pass"]
        ),
        "residual_live_mini_phase10_leave_pattern_guard_harm_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["phase10_leave_pattern_guard_harm_count"]
        ),
        "residual_live_mini_phase10_leave_route_guard_harm_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["phase10_leave_route_guard_harm_count"]
        ),
        "residual_live_mini_phase10_guard_assumption_node_count": int(
            artifacts["v3_residual_live_mini_loop"]["metrics"]["phase10_guard_assumption_node_count"]
        ),
        "residual_fresh_execution_mode": artifacts["v3_residual_fresh_live_loop"]["metrics"]["execution_mode"],
        "residual_fresh_selected_candidate_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["selected_candidate_count"]
        ),
        "residual_fresh_contract_ready_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["contract_ready_count"]
        ),
        "residual_fresh_preflight_ready_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["preflight_ready_count"]
        ),
        "residual_fresh_live_path_present": bool(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["fresh_live_path_present"]
        ),
        "residual_fresh_live_env_ready": bool(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["live_env_ready"]
        ),
        "residual_fresh_api_call_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["fresh_api_call_count"]
        ),
        "residual_fresh_planned_api_call_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["planned_fresh_api_call_count"]
        ),
        "residual_fresh_accepted_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["accepted_count"]
        ),
        "residual_fresh_applied_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["applied_count"]
        ),
        "residual_fresh_graph_copy_node_delta": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["graph_copy_node_delta"]
        ),
        "residual_fresh_main_graph_mutation_count": int(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["main_graph_mutation_count"]
        ),
        "residual_fresh_secret_value_exposed": bool(
            artifacts["v3_residual_fresh_live_loop"]["metrics"]["secret_value_exposed"]
        ),
        "phase5_live_profile_source_artifact_count": int(
            artifacts["v3_phase5_bandit"]["metrics"]["live_profile_source_artifact_count"]
        ),
        "phase5_live_profile_count": int(
            artifacts["v3_phase5_bandit"]["metrics"]["live_profile_count"]
        ),
        "phase5_live_selected_production_profile": artifacts["v3_phase5_bandit"]["metrics"][
            "live_selected_production_profile"
        ],
        "phase5_live_selected_exploration_profile": artifacts["v3_phase5_bandit"]["metrics"][
            "live_selected_exploration_profile"
        ],
        "phase5_live_scheduler_vs_v1_utility": float(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_vs_v1_utility"]
        ),
        "phase5_live_scheduler_vs_original_v3_utility": float(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_vs_original_v3_utility"]
        ),
        "phase5_live_scheduler_lift_over_v3": float(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_lift_over_v3"]
        ),
        "phase5_live_blocks_compact_default": bool(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_blocks_compact_default"]
        ),
        "phase5_live_keeps_phase10_candidate": bool(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_keeps_phase10_as_candidate"]
        ),
        "phase5_live_uses_raw_prompts_or_answers": bool(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_uses_raw_prompts_or_answers"]
        ),
        "phase5_live_calibrated_guard_lift_over_hybrid": float(
            artifacts["v3_phase5_bandit"]["metrics"]["live_scheduler_calibrated_guard_lift_over_hybrid"]
        ),
        "phase5_live_calibrated_guard_vs_original_v3_lift_over_hybrid": float(
            artifacts["v3_phase5_bandit"]["metrics"][
                "live_scheduler_calibrated_guard_vs_original_v3_lift_over_hybrid"
            ]
        ),
        "phase9_compact_guard_active_case_count": int(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["active_case_count"]
        ),
        "phase9_compact_guard_vs_v1_n": int(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["repair_vs_v1_n"]
        ),
        "phase9_compact_guard_vs_v1_utility": float(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["repair_vs_v1_utility"]
        ),
        "phase9_compact_guard_vs_v1_margin": float(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["repair_vs_v1_margin"]
        ),
        "phase9_compact_guard_vs_v3_utility": float(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["repair_vs_v3_utility"]
        ),
        "phase9_compact_guard_margin_gain_over_v3": float(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["repair_margin_gain_over_v3_vs_v1"]
        ),
        "phase9_compact_guard_planned_total_calls": int(
            artifacts["v3_phase9_compact_frame_guard"]["metrics"]["planned_total_model_calls"]
        ),
        "phase9_hybrid_guard_heldout_n": int(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["heldout_case_count"]
        ),
        "phase9_hybrid_guard_selected_candidate_n": int(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["selected_candidate_case_count"]
        ),
        "phase9_hybrid_guard_v3_vs_v1_utility": float(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["v3_vs_v1_heldout_utility"]
        ),
        "phase9_hybrid_guard_vs_v1_utility": float(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["hybrid_vs_v1_heldout_utility"]
        ),
        "phase9_hybrid_guard_vs_v1_margin": float(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["hybrid_vs_v1_heldout_margin"]
        ),
        "phase9_hybrid_guard_lift_over_v3": float(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["hybrid_lift_over_v3_vs_v1_heldout"]
        ),
        "phase9_hybrid_guard_vs_v3_utility": float(
            artifacts["v3_phase9_hybrid_guard"]["metrics"]["hybrid_vs_original_v3_heldout_utility"]
        ),
        "phase10_world_model_candidate_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["candidate_transition_count"]
        ),
        "phase10_world_model_support_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["compact_support_row_count"]
        ),
        "phase10_world_model_candidate_v1_utility": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["loo_selected_vs_v1_utility"]
        ),
        "phase10_world_model_candidate_v1_lift_over_v3": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["loo_selected_vs_v1_lift_over_v3"]
        ),
        "phase10_world_model_candidate_vs_v3_utility": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["loo_selected_vs_v3_utility"]
        ),
        "phase10_world_model_all_vs_v1_utility": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["all_heldout_policy_vs_v1_utility"]
        ),
        "phase10_world_model_all_lift_over_v3": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["all_heldout_policy_lift_over_v3"]
        ),
        "phase10_world_model_calibrated_vs_v1_utility": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_vs_v1_utility"]
        ),
        "phase10_world_model_calibrated_vs_original_v3_utility": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_vs_original_v3_utility"]
        ),
        "phase10_world_model_calibrated_lift_over_v3": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_lift_over_v3"]
        ),
        "phase10_world_model_calibrated_lift_over_raw": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_lift_over_raw_world_model"]
        ),
        "phase10_world_model_calibrated_lift_over_hybrid": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_lift_over_retained_hybrid"]
        ),
        "phase10_world_model_calibrated_vs_original_v3_lift_over_hybrid": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"][
                "calibrated_policy_vs_original_v3_lift_over_hybrid"
            ]
        ),
        "phase10_world_model_calibrated_harm_vs_hybrid_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_harm_vs_hybrid_count"]
        ),
        "phase10_world_model_calibrated_win_vs_hybrid_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_win_vs_hybrid_count"]
        ),
        "phase10_world_model_calibrated_override_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_policy_override_count"]
        ),
        "phase10_world_model_gap_to_hybrid": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["learned_gap_to_retained_hybrid"]
        ),
        "phase10_world_model_recommended_promotion": artifacts["v3_phase10_discrete_world_model"]["metrics"][
            "recommended_promotion"
        ],
        "phase10_world_model_calibration_beats_base_rate": bool(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibration_beats_base_rate"]
        ),
        "phase10_leave_pattern_group_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_pattern_out_group_count"]
        ),
        "phase10_leave_pattern_raw_vs_v3_lift": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_pattern_out_raw_vs_v3_lift"]
        ),
        "phase10_leave_pattern_guard_vs_v3_lift": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_pattern_out_guard_vs_v3_lift"]
        ),
        "phase10_leave_pattern_guard_lift_over_raw": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_pattern_out_guard_lift_over_raw"]
        ),
        "phase10_leave_pattern_raw_harm_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_pattern_out_raw_harm_count"]
        ),
        "phase10_leave_pattern_guard_harm_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_pattern_out_guard_harm_count"]
        ),
        "phase10_leave_route_tag_group_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_route_tag_out_group_count"]
        ),
        "phase10_leave_route_tag_raw_vs_v3_lift": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_route_tag_out_raw_vs_v3_lift"]
        ),
        "phase10_leave_route_tag_guard_vs_v3_lift": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_route_tag_out_guard_vs_v3_lift"]
        ),
        "phase10_leave_route_tag_guard_lift_over_raw": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_route_tag_out_guard_lift_over_raw"]
        ),
        "phase10_leave_route_tag_raw_harm_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_route_tag_out_raw_harm_count"]
        ),
        "phase10_leave_route_tag_guard_harm_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["leave_route_tag_out_guard_harm_count"]
        ),
        "phase10_guard_assumption_node_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["guard_assumption_node_count"]
        ),
        "phase10_guard_assumption_active_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["guard_assumption_active_count"]
        ),
        "phase10_guard_assumption_candidate_count": int(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["guard_assumption_candidate_count"]
        ),
        "phase10_calibrated_rows_with_guard_assumption_rate": float(
            artifacts["v3_phase10_discrete_world_model"]["metrics"]["calibrated_rows_with_guard_assumption_rate"]
        ),
        "phase10_reliability_observed_arm_record_count": int(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["observed_arm_record_count"]
        ),
        "phase10_reliability_bin_count": int(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["bin_count"]
        ),
        "phase10_reliability_raw_mae": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["raw_mae"]
        ),
        "phase10_reliability_calibrated_mae": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["calibrated_mae"]
        ),
        "phase10_reliability_base_rate_mae": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["base_rate_mae"]
        ),
        "phase10_reliability_calibrated_mae_lift_over_base": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["calibrated_mae_lift_over_base_rate"]
        ),
        "phase10_reliability_calibrated_brier_lift_over_base": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["calibrated_brier_lift_over_base_rate"]
        ),
        "phase10_reliability_raw_ece": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["raw_ece"]
        ),
        "phase10_reliability_calibrated_ece": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["calibrated_ece"]
        ),
        "phase10_reliability_calibrated_ece_lift_over_raw": float(
            artifacts["v3_phase10_reliability_calibration"]["metrics"]["calibrated_ece_lift_over_raw"]
        ),
        "phase10_reliability_source_raw_beats_base": bool(
            artifacts["v3_phase10_reliability_calibration"]["metrics"][
                "source_phase10_calibration_beats_base_rate"
            ]
        ),
        "guard_policy_learned_update_count": int(
            artifacts["v3_guard_policy_learning"]["metrics"]["learned_guard_update_count"]
        ),
        "guard_policy_supported_guard_count": int(
            artifacts["v3_guard_policy_learning"]["metrics"]["supported_guard_count"]
        ),
        "guard_policy_weight_range": float(
            artifacts["v3_guard_policy_learning"]["metrics"]["guard_weight_range"]
        ),
        "guard_policy_promote_weight_count": int(
            artifacts["v3_guard_policy_learning"]["metrics"]["promote_weight_count"]
        ),
        "guard_policy_learned_lift_over_hybrid": float(
            artifacts["v3_guard_policy_learning"]["metrics"]["learned_policy_lift_over_hybrid"]
        ),
        "guard_policy_harm_vs_hybrid_count": int(
            artifacts["v3_guard_policy_learning"]["metrics"]["learned_policy_harm_vs_hybrid_count"]
        ),
        "guard_policy_raw_world_model_status": artifacts["v3_guard_policy_learning"]["metrics"][
            "raw_world_model_status"
        ],
        "same_batch_judged_n": int(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["same_batch_judged_n"]
        ),
        "same_batch_toggle_pair_count": int(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["toggle_pair_count"]
        ),
        "same_batch_raw_v3_vs_v1_utility": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["raw_v3_vs_v1_utility"]
        ),
        "same_batch_raw_v3_vs_v1_ci_lower": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["raw_v3_vs_v1_ci_lower"]
        ),
        "same_batch_raw_v3_vs_no_morphism_utility": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["raw_v3_vs_no_morphism_utility"]
        ),
        "same_batch_raw_v3_vs_no_recursive_utility": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["raw_v3_vs_no_recursive_utility"]
        ),
        "same_batch_raw_v3_vs_no_world_model_utility": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["raw_v3_vs_no_world_model_utility"]
        ),
        "same_batch_hybrid_lift_over_raw_v3": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["hybrid_lift_over_raw_v3"]
        ),
        "same_batch_calibrated_lift_over_hybrid": float(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["calibrated_lift_over_hybrid"]
        ),
        "same_batch_calibrated_harm_vs_hybrid_count": int(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["calibrated_harm_vs_hybrid_count"]
        ),
        "same_batch_fresh_live_300_problem_level_n": int(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["fresh_live_300_problem_level_n"]
        ),
        "same_batch_uses_raw_prompts_or_answers": bool(
            artifacts["v3_same_batch_ablation_suite"]["metrics"]["uses_raw_prompts_or_answers"]
        ),
        "world_model_calibration_source_artifact_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["source_artifact_count"]
        ),
        "world_model_calibration_surface_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["calibration_surface_count"]
        ),
        "world_model_calibrated_surface_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["calibrated_surface_count"]
        ),
        "world_model_leave_domain_out_surface_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["leave_domain_out_surface_count"]
        ),
        "world_model_uncalibrated_promotion_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["uncalibrated_promotion_count"]
        ),
        "world_model_phase8_brier_improvement": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase8_quality_brier_improvement"]
        ),
        "world_model_leave_domain_out_available": bool(
            artifacts["v3_world_model_calibration"]["metrics"]["phase9_leave_domain_out_available"]
        ),
        "world_model_leave_domain_out_domain_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["phase9_leave_domain_out_domain_count"]
        ),
        "world_model_leave_domain_out_nonnegative_domain_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["phase9_leave_domain_out_nonnegative_domain_count"]
        ),
        "world_model_leave_domain_out_max_error": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase9_leave_domain_out_max_calibration_error"]
        ),
        "world_model_phase10_all_lift_over_v3": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_all_lift_over_v3"]
        ),
        "world_model_phase10_calibration_beats_base_rate": bool(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibration_beats_base_rate"]
        ),
        "world_model_phase10_selected_arm_mae_minus_base_rate": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_selected_arm_mae_minus_base_rate"]
        ),
        "world_model_phase10_calibrated_vs_v1_utility": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_vs_v1_utility"]
        ),
        "world_model_phase10_calibrated_vs_original_v3_utility": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_vs_original_v3_utility"]
        ),
        "world_model_phase10_calibrated_lift_over_v3": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_lift_over_v3"]
        ),
        "world_model_phase10_calibrated_lift_over_raw": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_lift_over_raw_world_model"]
        ),
        "world_model_phase10_calibrated_lift_over_hybrid": float(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_lift_over_retained_hybrid"]
        ),
        "world_model_phase10_calibrated_vs_original_v3_lift_over_hybrid": float(
            artifacts["v3_world_model_calibration"]["metrics"][
                "phase10_calibrated_policy_vs_original_v3_lift_over_hybrid"
            ]
        ),
        "world_model_phase10_calibrated_harm_vs_hybrid_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_harm_vs_hybrid_count"]
        ),
        "world_model_phase10_calibrated_win_vs_hybrid_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_win_vs_hybrid_count"]
        ),
        "world_model_phase10_calibrated_override_count": int(
            artifacts["v3_world_model_calibration"]["metrics"]["phase10_calibrated_policy_override_count"]
        ),
        "world_model_phase5_keeps_phase10_candidate": bool(
            artifacts["v3_world_model_calibration"]["metrics"]["phase5_keeps_phase10_candidate"]
        ),
        "world_model_uses_raw_prompts_or_answers": bool(
            artifacts["v3_world_model_calibration"]["metrics"]["uses_raw_prompts_or_answers"]
        ),
        "main_graph_memory_shadow_main_graph_node_count": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["main_graph_node_count"]
        ),
        "main_graph_memory_shadow_dry_run_group_count": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["dry_run_group_count"]
        ),
        "main_graph_memory_shadow_dry_run_store_mutated": bool(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["dry_run_store_mutated"]
        ),
        "main_graph_memory_shadow_main_graph_mutated": bool(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["main_graph_mutated"]
        ),
        "main_graph_memory_shadow_archived_node_count": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["shadow_applied_archived_node_count"]
        ),
        "main_graph_memory_shadow_consolidated_node_count": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["shadow_applied_consolidated_node_count"]
        ),
        "main_graph_memory_shadow_query_count": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["query_count"]
        ),
        "main_graph_memory_shadow_precision_delta": float(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["precision_delta"]
        ),
        "main_graph_memory_shadow_archive_exposure_after": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["archive_exposure_after"]
        ),
        "main_graph_memory_shadow_archive_exposure_delta": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["archive_exposure_delta"]
        ),
        "main_graph_memory_shadow_memory_hit_delta": int(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["memory_hit_delta"]
        ),
        "main_graph_memory_shadow_context_efficiency_delta": float(
            artifacts["v3_main_graph_memory_shadow"]["metrics"]["context_efficiency_delta"]
        ),
        "main_graph_memory_controlled_apply_main_graph_mutated": bool(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["main_graph_mutated"]
        ),
        "main_graph_memory_controlled_apply_rollback_entry_count": int(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["rollback_entry_count"]
        ),
        "main_graph_memory_controlled_apply_planned_archive_count": int(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["planned_archive_count"]
        ),
        "main_graph_memory_controlled_apply_consolidated_count": int(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["applied_consolidated_node_count"]
        ),
        "main_graph_memory_controlled_apply_precision_delta": float(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["precision_delta"]
        ),
        "main_graph_memory_controlled_apply_archive_exposure_after": int(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["archive_exposure_after"]
        ),
        "main_graph_memory_controlled_apply_context_efficiency_delta": float(
            artifacts["v3_main_graph_memory_controlled_apply"]["metrics"]["context_efficiency_delta"]
        ),
        "continuous_daemon_scheduled_cycle_count": int(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["scheduled_cycle_count"]
        ),
        "continuous_daemon_checkpoint_pair_count": int(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["checkpoint_pair_count"]
        ),
        "continuous_daemon_rate_limit_violation_count": int(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["rate_limit_violation_count"]
        ),
        "continuous_daemon_recovery_action_count": int(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["recovery_action_count"]
        ),
        "continuous_daemon_fresh_loop_queue_integrated": bool(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["fresh_loop_queue_integrated"]
        ),
        "continuous_daemon_memory_apply_queue_integrated": bool(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["memory_apply_queue_integrated"]
        ),
        "continuous_daemon_ungated_graph_mutation_count": int(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["ungated_graph_mutation_count"]
        ),
        "continuous_daemon_background_ready": bool(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["continuous_background_ready"]
        ),
        "continuous_daemon_background_process_started": bool(
            artifacts["v3_continuous_daemon_scheduler"]["metrics"]["background_process_started"]
        ),
        "phase11_capability_count": int(
            artifacts["v3_phase11_capability_audit"]["metrics"]["capability_count"]
        ),
        "phase11_artifact_pass_rate": float(
            artifacts["v3_phase11_capability_audit"]["metrics"]["artifact_pass_rate"]
        ),
        "phase11_outer_shell_count": int(
            artifacts["v3_phase11_capability_audit"]["metrics"]["outer_shell_count"]
        ),
        "phase11_outer_shell_production_claim_count": int(
            artifacts["v3_phase11_capability_audit"]["metrics"]["outer_shell_production_claim_count"]
        ),
        "phase11_blocked_claim_count": int(
            artifacts["v3_phase11_capability_audit"]["metrics"]["blocked_claim_count"]
        ),
        "phase11_phase4_status": artifacts["v3_phase11_capability_audit"]["metrics"]["phase4_status"],
        "phase11_phase7_status": artifacts["v3_phase11_capability_audit"]["metrics"]["phase7_status"],
        "phase11_phase10_status": artifacts["v3_phase11_capability_audit"]["metrics"]["phase10_status"],
        "phase11_phase10_calibration_status": artifacts["v3_phase11_capability_audit"]["metrics"][
            "phase10_calibration_status"
        ],
        "prompt_answer_payload_stored": bool(artifacts["first_party_world_model_scale"].get("prompt_answer_payload_stored")),
        "secret_leak_detected": bool(artifacts["first_party_world_model_scale"].get("secret_leak_detected")),
    }


def _metric_subset(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: data.get(key) for key in keys if key in data}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 paper-scale evidence aggregation.")
    parser.add_argument("--eval-id", default="full_v3_paper_scale_evidence_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_paper_scale_evidence_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
