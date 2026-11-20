"""V17.5 (M4.3) Regression & Integration Suite: Full `Hegel_M4_2_Audit` Closure & Channel-Bound Continual Study."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
    run_m4_2_audit_closure_verification_suite,
    run_m4_3_channel_bound_continual_study,
)
from hegel_machine.cognition.m4_compositional_discovery import load_verified_authored_method


def test_v17_5_committed_v17_4_w1_param_mutation_rejected(tmp_path: Path) -> None:
    """Reproduces GPT-6's exact `probe_m42_contract_controls.py` `p0 * 3` mutation on a copy of committed W1."""
    repo_root = Path(__file__).resolve().parents[1]
    w1_committed = (
        repo_root
        / "docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120"
        / "persisted_authored_workspaces/persisted_stage1_authored_workspaces/stage1_W1_haldane_shunt"
    )
    copy_ws = tmp_path / "w1_copy"
    shutil.copytree(w1_committed, copy_ws)
    art_p = copy_ws / "authored_method.json"
    orig = json.loads(art_p.read_text(encoding="utf-8"))

    good = load_verified_authored_method(
        art_p,
        expected_ast_digest=orig["ast_skeleton_digest"],
        require_external_manifest=True,
    )
    assert good is not None

    bad = deepcopy(orig)
    bad["fitted_params"]["p0"] *= 3.0
    art_p.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    badload = load_verified_authored_method(
        art_p,
        expected_ast_digest=orig["ast_skeleton_digest"],
        require_external_manifest=True,
    )
    assert badload is None


def test_v17_5_all_m4_2_audit_counterfactual_probes_closed(tmp_path: Path) -> None:
    """Locks in 100% closure across all 6 `Hegel_M4_2_Audit` counterfactual probe families (`C1`–`C6`)."""
    rep = run_m4_2_audit_closure_verification_suite(tmp_path / "m42_audit_probes")

    c1 = rep["c1_confirmed_instance_binding"]
    assert c1["baseline_accepted"] is True
    assert c1["mutated_artifact_param_accepted"] is False
    assert c1["mutated_both_param_files_accepted"] is False
    assert c1["cross_node_swap_rejected"] is True

    c2 = rep["c2_or_branch_and_program_order"]
    assert c2["permission_merge_accepted"] is True
    assert c2["scope_A_bounded_trial_granted"] is True
    assert c2["scope_B_bounded_trial_granted"] is False
    assert c2["scope_B_capped_tier"] == "retrieve_candidate"
    assert c2["heterogeneous_law_merge_accepted"] is False
    assert c2["canonical_program_valid"] is True
    assert c2["reversed_program_rejected"] is True

    c3 = rep["c3_fresh_confirmation_invariants"]
    assert c3["requested_3_acquired_confirm_points"] == 3
    assert c3["requested_3_overlap_max"] == 0
    assert c3["parent_exhaustion_child_confirm_points"] == 2
    assert c3["parent_exhaustion_overlap_max"] == 0
    assert c3["dev_t13_prefix_confirm_overlap"] == 0

    c4 = rep["c4_trajectory_and_certificate_governance"]
    assert c4["baseline_certified"] is True
    assert c4["nonterminal_window_shift_certified"] is False
    assert c4["endpoint_fast_shift_certified"] is False
    assert c4["narrow_cert_widened_box_accepted"] is False
    assert c4["node_and_edge_boxes_synced"] is True
    assert c4["stage3_oob_heldout_status"] == "out_of_verified_scale_box"
    assert c4["stage3_oob_heldout_false_confirmation"] is False
    assert c4["named_oob_confirm_via_recall"] is False
    assert c4["renamed_oob_confirm_via_recall"] is False
    assert c4["renamed_inbox_confirm_via_recall"] is True
    assert c4["raw_unmapped_point_in_box"] is False

    c5 = rep["c5_readonly_dispatcher_and_edge_escalation"]
    assert c5["optical_rejected_on_raw_post_stage2_forest"] is True
    assert c5["dispatcher_stream_strictly_readonly"] is True
    assert c5["any_fit_called_including_cold_loader"] is False
    assert c5["deleting_coarse_edges_blocks_quench_escalation"] is True

    c6 = rep["c6_channel_pricing_stage0_and_open_rediscovery"]
    assert c6["complete_forest_pi_channel"] == "coarse_macro"
    assert c6["unrelated_hydro_edge_pi_channel"] == "micro_fine"
    assert c6["no_coarse_edge_pi_channel"] == "micro_fine"
    assert c6["unrelated_edge_vs_no_edge_meas_cost_equal"] is True
    assert c6["unrelated_edge_vs_no_edge_composite_cost_equal"] is True
    assert c6["true_stage0_node_count"] == 8
    assert c6["true_stage0_coarse_edge_count"] == 2
    assert c6["true_stage0_wormhole_edge_count"] == 1
    assert c6["open_rediscovery_ran_distinct_acquisitions"] is True
    assert c6["open_rediscovery_strict_pass"] is True


def test_v17_5_m4_3_channel_bound_single_seed_continual_benchmark(tmp_path: Path) -> None:
    """Verifies the 11-arm `M4.3` channel-bound continual benchmark, zero confirm overlap, & maintenance deltas."""
    rep = run_m4_3_channel_bound_continual_study(tmp_path / "m43_1seed", seeds=(20_301_120,))
    arms = rep["stage3_arms"]
    hegel = arms["Hegel_Unified_Continual_Forest"]

    assert hegel["strict_pass_rate"] == 1.0
    assert hegel["false_confirmation_rate"] == 0.0
    assert hegel["oos_scoped_unknown_rate"] == 1.0
    assert hegel["oos_false_confirmation_rate"] == 0.0

    # Universal fresh confirmation invariant: ZERO training-prefix overlap across ALL 11 arms!
    for arm_name, st in arms.items():
        assert st["max_prefix_confirm_overlap"] == 0, f"{arm_name} had training-confirm overlap!"
        assert st["oos_false_confirmation_rate"] == 0.0

    for comp_arm in (
        "Hegel_NoMaintenance_SameHistory",
        "Forest_Alpha0_PoolOn_SameHistory",
        "Forest_Alpha0_PoolOff_SameHistory",
        "Ordinary_AllHistory_StratifiedDevOrder",
        "Ordinary_AllHistory_BestPilotRMSE",
        "Flat_BM25_SameHistory",
        "Frozen_After_Stage0_OpenRediscovery",
    ):
        assert hegel["mean_composite_cost"] < arms[comp_arm]["mean_composite_cost"], (
            f"Expected Hegel ({hegel['mean_composite_cost']}) < {comp_arm} ({arms[comp_arm]['mean_composite_cost']})"
        )

    # True Stage-0 Frozen without discovery cannot solve newly discovered Stage-1 mechanisms (`W1..W5`)!
    froz_nodisc = arms["Frozen_After_Stage0_NoDiscovery"]
    assert froz_nodisc["strict_pass_rate"] < 0.35
    assert froz_nodisc["false_confirmation_rate"] == 0.0

    # Stage-2 maintenance improves BOTH non-Pi (`HaldaneShunt_2D` via Minton/scope) AND `PiS_MultiScale_Mixed_2D`!
    m_by_fam = rep["maintenance_by_family"]
    assert m_by_fam["HaldaneShunt_2D"]["stage3_total_delta_saved_by_maintenance"] > 0.0
    assert m_by_fam["PiS_MultiScale_Mixed_2D"]["stage3_total_delta_saved_by_maintenance"] > 0.0
