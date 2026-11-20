"""V17.6 (M4.4) Regression & Integration Suite: Full `Hegel_M4_3_Audit` Invariant Closure
(Instance Anchor, Role-Map Lock, Self-Verifying Archive Digests, Exact 2-Mode Eigen-Basis,
Legacy Compatibility & Dynamic Index Refresh) + Stateful Online Write-Back Stream (`M5` Bridge).
"""

from __future__ import annotations

from pathlib import Path

from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
    build_m4_3_continual_forest,
)
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    run_m4_3_audit_closure_verification_suite,
    run_m4_4_online_writeback_and_multicue_stream,
)


def test_v17_6_all_m4_3_audit_p1_and_p2_counterfactual_probes_closed(tmp_path: Path) -> None:
    """Locks in 100% closure across all P1 & P2 findings from `Hegel_M4_3_Audit`."""
    rep = run_m4_3_audit_closure_verification_suite(tmp_path / "m43_audit_closure")

    # P1-1: Node-to-instance anchor + genuine same-AST calibrated instance swap rejected
    p1_inst = rep["p1_instance_anchor_and_same_ast_swap"]
    assert p1_inst["post_stage2_artifact_node_count"] == 14
    assert p1_inst["missing_expected_instance_count"] == 0
    assert p1_inst["same_ast_skeleton"] is True
    assert p1_inst["same_model_source_sha256"] is True
    assert p1_inst["distinct_confirmed_instance_digest"] is True
    assert p1_inst["baseline_status"] == "dispatched_ok"
    assert abs(p1_inst["baseline_prediction"] - (6.0 * 1.6 / 3.6)) < 1e-6
    assert p1_inst["baseline_has_explicit_source_params_and_instance_hashes"] is True
    assert p1_inst["swapped_same_ast_rejected"] is True
    assert p1_inst["missing_digest_rejected"] is True
    assert p1_inst["restored_original_status"] == "dispatched_ok"

    # P1-2: Unified coordinate role-map lock + fresh confirm box intersection + boundary safety
    p1_role = rep["p1_coordinate_role_lock_and_fresh_confirm_box"]
    assert p1_role["native_T21_channel"] == "coarse_macro"
    assert p1_role["native_T21_strict_pass"] is True
    assert p1_role["native_T21_false_confirmation"] is False
    assert p1_role["canonical_axis_swap_box_gate_accepted"] is False
    assert p1_role["canonical_axis_swap_selected_channel"] == "micro_fine"
    assert p1_role["canonical_axis_swap_strict_pass"] is True
    assert p1_role["canonical_axis_swap_false_confirmation"] is False
    assert p1_role["canonical_axis_swap_heldout_rmse"] < 1e-5
    assert p1_role["canonical_axis_swap_no_fine_channel"] == "micro_fine"
    assert p1_role["canonical_axis_swap_no_fine_selected_node"] != "ode_meso_schur_pi_thermal"
    assert p1_role["canonical_axis_swap_no_fine_false_confirmation"] is False
    assert p1_role["anon_role_locked_map"] == {"x0": "T_horizon", "x1": "u_drive"}
    assert p1_role["anon_role_oob_heldout_status"] == "out_of_verified_scale_box"
    assert p1_role["anon_role_oob_heldout_false_confirmation"] is False
    assert len(p1_role["fresh_confirm_coarse_points"]) == 1
    assert p1_role["all_coarse_acquisitions_inside_verified_box"] is True
    assert p1_role["fresh_confirm_strict_pass"] is True
    assert p1_role["fresh_confirm_false_confirmation"] is False
    assert p1_role["boundary_zone_selected_channel"] == "micro_fine"
    assert p1_role["boundary_zone_strict_pass"] is True
    assert p1_role["boundary_zone_false_confirmation"] is False

    # P2-S3 & S5: Cross-instance certificate rejected atomically + self-verifying relocatable digests
    p2_cert = rep["p2_certificate_identity_and_self_verifying_archive"]
    assert p2_cert["cross_instance_certificate_accepted"] is False
    assert p2_cert["cross_instance_certificate_reason"] == "certificate_instance_digest_mismatch"
    assert p2_cert["failed_revalidation_left_forest_unmutated"] is True
    for snap_chk in p2_cert["archive_digest_verification"].values():
        assert snap_chk["disk_json"]["semantic_digest_mismatch_count"] == 0
        assert snap_chk["disk_json"]["certificate_digest_mismatch_count"] == 0
        assert len(snap_chk["disk_json"]["missing_expected_instance_nodes"]) == 0
        assert snap_chk["reloaded_memory"]["semantic_digest_mismatch_count"] == 0
    assert p2_cert["cold_reloaded_r3_all_routes_exact"] is True
    assert p2_cert["cold_reloaded_r3_python_fit_calls"] == 0
    assert p2_cert["cold_reloaded_r3_strictly_readonly"] is True

    # P2-S4 & Cost Completeness: Exact `A12 == g_sf` matrix + zero discarded prefit + metered local RK4
    p2_mat = rep["p2_exact_2mode_matrix_and_event_accounting"]
    assert p2_mat["quench_T23_strict_pass"] is True
    assert p2_mat["quench_T23_heldout_rmse"] < 1e-5
    assert p2_mat["discarded_generic_prefit_count"] == 0
    assert p2_mat["evaluated_candidate_count"] == 1
    assert p2_mat["local_rk4_compute_metered"] is True
    assert p2_mat["artifact_integrity_compute_metered"] is True

    # P2-Legacy & Dynamic Index Invalidation
    p2_leg = rep["p2_legacy_entrypoints_and_dynamic_index_invalidation"]
    assert p2_leg["legacy_unbundled_rejected_under_strict_flags"] is True
    assert p2_leg["legacy_unbundled_refit_calls_under_strict_flags"] == 0
    assert p2_leg["legacy_unbundled_permissive_mode"] == "legacy_isolated_refit"
    assert p2_leg["legacy_scale_blind_t07_strict_pass"] is True
    assert p2_leg["legacy_scale_blind_t07_duplicate_micro_confirm_points"] == 0
    assert p2_leg["dynamic_w4_version_before"] == 1
    assert p2_leg["dynamic_ppr_version_after_live_revision"] == 2
    assert p2_leg["dynamic_ppr_refreshed_automatically"] is True
    assert p2_leg["dynamic_ppr_has_new_precondition"] is True
    assert p2_leg["dynamic_cuetable_version_after_live_revision"] == 2
    assert p2_leg["dynamic_cuetable_recompiled_automatically"] is True


def test_v17_6_stateful_online_writeback_and_multicue_continual_stream(tmp_path: Path) -> None:
    """Verifies genuine stateful mid-stream write-back, live index refresh, & backward retention (`M5` Bridge)."""
    built = build_m4_3_continual_forest(tmp_path / "base_life")
    stream = run_m4_4_online_writeback_and_multicue_stream(tmp_path / "online_stream", built)

    arms = stream["arms"]
    ppr_on = arms["Hegel_LiveWriteBack_PPR_PoolOn"]
    ppr_off = arms["Hegel_LiveWriteBack_PPR_PoolOff"]
    cue_live = arms["Compiled_LiveWriteBack_CueTable"]
    frozen = arms["Frozen_PostStage2_OpenRediscovery"]

    assert ppr_on["strict_pass_rate"] == 1.0
    assert ppr_on["false_confirmation_rate"] == 0.0
    assert ppr_on["max_prefix_confirm_overlap"] == 0
    assert ppr_on["downstream_reuse_recall_count"] == 12
    assert frozen["downstream_reuse_recall_count"] == 0
    assert ppr_on["index_auto_refresh_count"] >= 3
    assert cue_live["index_auto_refresh_count"] >= 3

    # Backward retention (`10/10` historical `R3`) + new `W6`/`W7` zero-`fit()` dispatch + self-verifying digests
    assert ppr_on["post_stream_r3_all_routes_exact"] is True
    assert ppr_on["post_stream_r3_any_fit_called"] is False
    assert ppr_on["post_stream_semantic_digest_mismatches"] == 0
    assert ppr_on["post_stream_missing_instance_digests"] == 0
    assert ppr_on["new_W6_zero_fit_dispatched"] is True
    assert ppr_on["new_W7_zero_fit_dispatched"] is True

    # Online write-back beats Frozen Post-Stage-2, AND PPR_PoolOn beats Compiled_CueTable under online mutations!
    assert stream["net_online_writeback_savings_vs_frozen_post_stage2"] > 80.0
    assert ppr_on["total_stream_composite_cost"] < cue_live["total_stream_composite_cost"]
    assert ppr_on["total_stream_composite_cost"] <= ppr_off["total_stream_composite_cost"]
