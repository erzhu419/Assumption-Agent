"""Comprehensive Pytest Suite for Hegel V18.0 (`M5` Final Culmination):
Generalized Multi-Scale Temporal Folding (`P_tau / N_tau / F_tau,H` + `Pi_fold` memory kernels)
& Consistent-Binding `SharedBaseSubtreePrior` + Calibrated Uncertainty Governance.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree import (
    run_full_v18_0_m5_study,
    run_m5_anonymous_multiseed_benchmark,
    run_m5_contract_and_counterexample_suite,
    run_m5_semantic_twin_showcase,
)


def test_m5_all_12_mathematical_and_redteam_controls() -> None:
    res = run_m5_contract_and_counterexample_suite()
    assert res["all_12_controls_passed"] is True
    c = res["controls"]
    assert c["C01_causal_cutoff_and_irregular_dt"]["centered_window_blocked"] is True
    assert c["C01_causal_cutoff_and_irregular_dt"]["post_t0_sample_blocked"] is True
    assert c["C02_normalized_fast_transfer_vs_ephemeral_spike"]["ephemeral_spike_rejected"] is True
    assert c["C02_normalized_fast_transfer_vs_ephemeral_spike"]["cross_timescale_10000x_binding_valid"] is True
    assert c["C03_pi_fold_memory_kernel_vs_constant"]["memory_kernel_accurate"] is True
    assert c["C04_short_window_long_reversal_uncertainty"]["long_goal_buys_slow_horizon_probe"] is True
    assert c["C04_short_window_long_reversal_uncertainty"]["short_goal_stops_via_hec_without_wasting_wait"] is True
    assert c["C05_order_invariant_and_lineage_dedup_diversity"]["strictly_order_invariant"] is True
    assert c["C05_order_invariant_and_lineage_dedup_diversity"]["renamed_clones_d_eff_equals_one"] is True
    assert c["C06_soft_warm_start_and_heterogeneity_fallback"]["compatible_warm_nfev_lt_cold"] is True
    assert c["C06_soft_warm_start_and_heterogeneity_fallback"]["soft_warm_start_escapes_to_wide_fallback"] is True
    assert c["C06_soft_warm_start_and_heterogeneity_fallback"]["soft_warm_start_solves_hetero_exact"] is True
    assert c["C07_probing_task_risk_voi_vs_raw_variance"]["task_risk_voi_picks_relevant"] is True
    assert c["C08_consistent_joint_binding_and_dedup"]["disconnected_edge_stitch_rejected"] is True
    assert c["C08_consistent_joint_binding_and_dedup"]["reencoded_duplicate_evidence_not_double_counted"] is True
    assert c["C09_attribution_gated_upstream_revision"]["subtree_revised_upstream_to_v2"] is True
    assert c["C09_attribution_gated_upstream_revision"]["original_source_node_untouched_in_scope"] is True
    assert c["C10_gray_zone_blocks_s02_180_and_bisects"]["b_180_blocked_as_gray_zone"] is True
    assert c["C10_gray_zone_blocks_s02_180_and_bisects"]["bisected_bound_truly_safe"] is True
    assert c["C11_anti_unify_subtree_extraction"]["valid_extraction"] is True
    assert c["C12_unanimous_wrong_candidates_clean_exit"]["mandatory_target_confirm_catches_joint_miss"] is True


def test_m5_track1_anonymous_benchmark_and_joint_stream_cold_reload(tmp_path: Path) -> None:
    bench = run_m5_anonymous_multiseed_benchmark(tmp_path / "bench", seeds=(101, 202))
    arms = bench["arm_summaries"]

    # 1. Zero coarse-channel authorization confound between Forest and EqualAuth_Flat!
    assert bench["coarse_channel_measurement_parity_on_adiabatic"]["zero_coarse_auth_confound"] is True

    # 2. Hegel M5 achieves 100% in-scope pass, 100% OOS scoped_unknown, 0 false confirmations!
    hegel = arms["Hegel_M5_Full_TemporalSubtree_Forest"]
    assert hegel["in_scope_strict_pass"] == hegel["in_scope_count"]
    assert hegel["oos_honest_unknown"] == hegel["oos_count"]
    assert hegel["false_confirmations"] == 0

    # 3. Family I End-to-End: Hegel strictly beats Strong_Dynamic_StateSpace_AllHistory even including formation cost!
    strong_dyn = arms["Strong_Dynamic_StateSpace_AllHistory"]
    assert strong_dyn["in_scope_strict_pass"] == strong_dyn["in_scope_count"]
    assert hegel["lifecycle_composite_cost_with_formation"] < strong_dyn["lifecycle_composite_cost_with_formation"]
    assert hegel["total_scipy_nfev"] < strong_dyn["total_scipy_nfev"]

    # 4. Capability & Variance Ablations demonstrate why temporal state & confirmation-gated warm-start are essential:
    static_ab = arms["Ab_Memoryless_Static_Baseline"]
    blind_ab = arms["Ab_Blind_Parameter_Copy_Transfer"]
    assert static_ab["in_scope_pass_fraction"] < 0.40  # Fails qualitatively on temporal transition & memory!
    assert blind_ab["false_confirmations"] > 0  # Unchecked parameter copy produces real false confirmations!

    # 5. Family II Same-Certificate Organization Ablations:
    # Hegel_M5_Forest_Subtree_PPR strictly beats SingleNode_Wormhole_Only, EqualAuth_Dynamic_Flat_BM25, & RawVariance_Probe!
    hegel_org = arms["Hegel_M5_Forest_Subtree_PPR"]
    single_wh = arms["EqualAuth_SingleNode_Wormhole_Only"]
    eq_flat = arms["EqualAuth_Dynamic_Flat_BM25"]
    raw_var = arms["EqualAuth_RawVariance_Probe_Ablation"]
    assert hegel_org["target_composite_cost_total"] < single_wh["target_composite_cost_total"]
    assert hegel_org["target_composite_cost_total"] < eq_flat["target_composite_cost_total"]
    assert hegel_org["target_composite_cost_total"] < raw_var["target_composite_cost_total"]

    # 6. Flagship Joint Continual Stream `J` + Fresh Subprocess Cold Continuation (`J17..J20`):
    j_res = bench["flagship_joint_stream_J"]
    j_arms = j_res["arms"]
    assert j_arms["Hegel_M5_Full_TemporalSubtree_Forest"]["false_confirmations"] == 0
    assert j_arms["Hegel_M5_Full_TemporalSubtree_Forest"]["in_scope_pass_rate"] == "14/14"
    assert j_arms["Hegel_M5_Full_TemporalSubtree_Forest"]["oos_scoped_unknown_rate"] == "2/2"
    assert (
        j_arms["Hegel_M5_Full_TemporalSubtree_Forest"]["total_composite_cost"]
        < j_arms["EqualAuth_SingleNode_Wormhole_Only"]["total_composite_cost"]
    )
    assert (
        j_arms["Hegel_M5_Full_TemporalSubtree_Forest"]["total_composite_cost"]
        < j_arms["EqualAuth_Dynamic_Flat_BM25"]["total_composite_cost"]
    )
    strong_j_key = (
        "Strong_Dynamic_StateSpace_NoSubtree"
        if "Strong_Dynamic_StateSpace_NoSubtree" in j_arms
        else "Strong_Dynamic_StateSpace_AllHistory"
    )
    assert j_arms["Hegel_M5_Full_TemporalSubtree_Forest"]["total_composite_cost"] < j_arms[strong_j_key]["total_composite_cost"]

    cold = j_res["cold_subprocess_continuation_J17_J20"]
    assert cold["digest_check"]["semantic_digest_mismatch_count"] == 0
    assert cold["digest_check"]["subtree_digest_mismatch_count"] == 0
    assert cold["siphon_subtree_version"] == 2
    assert cold["cold_steps_passed"] == 4
    assert cold["cold_false_confirmations"] == 0


def test_m5_track2_semantic_twin_showcase_and_full_evidence_bundle(tmp_path: Path) -> None:
    twins = run_m5_semantic_twin_showcase()
    assert twins["twin_1_lenz_lechatelier_memory"]["normalized_t_over_tau_match"] is True
    assert twins["twin_2_short_gain_long_reversal"]["voi_selected_probe"] == "mid_term_gaming_indicator"
    assert twins["twin_3_changsha_nanchang_base_subtree"]["joint_binding_valid"] is True
    assert twins["twin_3_changsha_nanchang_base_subtree"]["voi_selected_discriminating_probe"] == "Q_local_anchor_industry_retention"

    ev_dir = tmp_path / "ev_bundle"
    manifest = run_full_v18_0_m5_study(ev_dir)
    assert manifest["part1_all_12_controls_passed"] is True
    assert (ev_dir / "SHA256SUMS.txt").is_file()
