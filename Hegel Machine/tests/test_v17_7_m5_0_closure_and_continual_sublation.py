"""V17.7 (`M4.5` Closure & `M5.0` Continual Sublation) Regression & Integration Suite:

1. `test_v17_7_all_m4_4_audit_p2_packages_closed`: Locks in 100% closure across all 5 `Hegel_M44_Audit` P2 packages;
2. `test_v17_7_m5_0_counterexample_sublation_pooling_and_cross_process_cold_resume`: Verifies the 24-step
   outcome-driven `M5.0` continual lifecycle across 5 arms (`PoolOn < PoolOff < Compiled < Dynamic_Flat < Frozen`),
   empirical `W4` counterexample scope sublation, genuine two-domain `W6` & `W7` wormhole promotion + negative
   `Domain C` refutation, full persisted bundle, AND fresh OS subprocess cold-reload continuation (`S25..S27`)!
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
    build_m4_3_continual_forest,
)
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    serialize_location_independent_forest,
    verify_forest_snapshot_digests,
)
from hegel_machine.cognition.m5_0_closure_and_continual_sublation import (
    persist_m5_0_online_state_to_bundle,
    run_m4_4_audit_closure_verification_suite,
    run_m5_0_counterexample_sublation_and_persistent_stream,
)


def test_v17_7_all_m4_4_audit_p2_packages_closed(tmp_path: Path) -> None:
    """Verifies 100% closure of `add_coarse_graining_edge` identity, OR `selected_action`, role lock, & S01 guards."""
    rep = run_m4_4_audit_closure_verification_suite(tmp_path / "m44_audit_closure")

    p1 = rep["p2_1_certificate_edge_node_relational_identity"]
    assert p1["add_cross_instance_edge_raised"] is True
    assert "certificate_instance_digest_mismatch" in p1["add_cross_instance_error"]
    assert p1["entire_forest_state_unchanged_on_rejected_add"] is True
    assert p1["tampered_edge_live_children"] == []
    assert p1["tampered_edge_verifier_mismatch_count"] == 1
    assert p1["no_edge_T21_channel"] == "micro_fine"
    assert p1["tampered_edge_T21_channel"] == "micro_fine"
    assert p1["tampered_edge_T21_measurement_cost"] == 70.0

    p2 = rep["p2_2_selected_action_and_canonical_role_unification"]
    assert p2["or_intact_loaded_only_selected_branch_B"] is True
    assert p2["or_intact_m43_strict_pass"] is True
    assert p2["or_absent_A_r3_status"] == "dispatched_ok"
    assert p2["or_absent_A_r3_within_tolerance"] is True
    assert p2["or_absent_A_m43_status"] == "completed"
    assert p2["or_absent_A_m43_confirmed_via_recall"] is True
    assert p2["or_absent_A_m43_strict_pass"] is True
    assert p2["or_absent_A_m43_heldout_rmse"] < 1e-6
    assert p2["r3_normal_canonical_status"] == "dispatched_ok"
    assert p2["r3_normal_canonical_abs_error"] < 1e-6
    assert p2["r3_contradictory_canonical_map_status"] == "invalid_contradictory_role_map"
    assert p2["r3_contradictory_canonical_map_prediction"] is None
    assert p2["r3_valid_anonymous_map_status"] == "dispatched_ok"
    assert p2["r3_valid_anonymous_map_abs_error"] < 1e-6
    assert p2["partial_canonical_selected_channel"] == "micro_fine"
    assert p2["partial_canonical_wasted_coarse_acquisitions"] == 0
    assert p2["partial_canonical_measurement_cost"] == 70.0
    assert p2["partial_canonical_strict_pass"] is True
    assert p2["partial_canonical_false_confirmation"] is False

    p3 = rep["p2_3_online_scope_semantics_and_failure_guard"]
    assert p3["version_incremented_on_failed_s01"] is False
    assert p3["w4_unchanged_on_failed_s01"] is True
    assert p3["counterexample_w4_false_confirmation"] is False
    assert p3["w4_version_after_counterexample"] == 2
    assert p3["w4_in_scope_b_1_45_gate_granted"] is True
    assert p3["w4_in_scope_b_1_45_r3_status"] == "dispatched_ok"
    assert p3["w4_out_of_scope_b_2_10_gate_granted"] is False
    assert p3["w4_out_of_scope_b_2_10_r3_status"] == "scope_unauthorized"


def test_v17_7_m5_0_counterexample_sublation_pooling_and_cross_process_cold_resume(tmp_path: Path) -> None:
    """Verifies outcome-driven `M5.0` continual sublation, end-to-end Pooling win, & fresh subprocess cold resume!"""
    transient_build = tmp_path / "transient_build"
    built = build_m4_3_continual_forest(transient_build / "base_life")
    m5_out = run_m5_0_counterexample_sublation_and_persistent_stream(transient_build / "m5_stream", built)

    arms = m5_out["arms"]
    ppr_on = arms["Hegel_M5_LiveSublation_PPR_PoolOn"]
    ppr_off = arms["Hegel_M5_LiveSublation_PPR_PoolOff"]
    cue_live = arms["Compiled_M5_LiveSublation_CueTable"]
    flat_live = arms["Dynamic_Flat_StructureMemory_LiveWriteBack"]
    frozen = arms["Frozen_PostStage2_OpenRediscovery"]

    assert ppr_on["in_scope_strict_pass_rate"] == 1.0
    assert ppr_on["in_scope_false_confirmation_rate"] == 0.0
    assert ppr_on["out_of_scope_false_confirmation_rate"] == 0.0
    assert ppr_on["out_of_scope_scoped_unknown_rate"] == 1.0
    assert ppr_on["max_prefix_confirm_overlap"] == 0
    assert ppr_on["downstream_reuse_recall_count"] == 16
    assert frozen["downstream_reuse_recall_count"] == 0
    # After S02 empirical sublation (`b <= theta_emp`), S04 (`b >= 1.95`) skips `W4` at gate (`0` wasted fits on W4!)
    assert ppr_on["s04_high_bleed_evaluated_candidates"] < frozen["s04_high_bleed_evaluated_candidates"]

    # End-to-end paid-task Equivalence Pooling win (`PoolOn` evaluates 8 candidates across 8 multi-cue W7 tasks vs 24 on `PoolOff`!)
    assert ppr_on["w7_multicue_evaluated_candidates_total"] == 8
    assert ppr_off["w7_multicue_evaluated_candidates_total"] == 24
    assert ppr_on["total_stream_composite_cost"] < ppr_off["total_stream_composite_cost"]
    assert ppr_on["total_stream_composite_cost"] < cue_live["total_stream_composite_cost"]
    assert ppr_on["total_stream_composite_cost"] < flat_live["total_stream_composite_cost"]
    assert flat_live["total_stream_composite_cost"] < frozen["total_stream_composite_cost"]

    # Archive bundle & DELETE `transient_build` completely, then verify cold-reload continuation in a NEW OS subprocess!
    archive_bundle = tmp_path / "archived_v17_7_bundle"
    persisted_dst = archive_bundle / "persisted_authored_workspaces"
    persisted_dst.mkdir(parents=True, exist_ok=True)
    life_src = Path(str(built["forest"].artifact_root))
    for sub_n in ("stage0_metered_sources", "persisted_stage1_authored_workspaces", "stage2_lifecycle"):
        if (life_src / sub_n).exists():
            shutil.copytree(life_src / sub_n, persisted_dst / sub_n, dirs_exist_ok=True)

    primary_state = m5_out["primary_persisted_state"]
    wb_meta = persist_m5_0_online_state_to_bundle(primary_state, out_bundle_dir=archive_bundle)
    assert set(wb_meta["online_writeback_node_dirs"]) == {
        "online_W6_reuptake_domain_b",
        "online_W6_tubule_domain_a",
        "online_W7_allosteric_domain_b",
        "online_W7_photobio_domain_a",
    }

    # Destroy transient build directory so ANY dangling `/tmp` reference MUST fail!
    shutil.rmtree(transient_build)

    snap_json = json.loads(
        (archive_bundle / "relocatable_forest_snapshots" / "post_m5_online_stream_forest_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    disk_dig = verify_forest_snapshot_digests(snap_json)
    assert disk_dig["semantic_digest_mismatch_count"] == 0
    assert disk_dig["history_semantic_digest_mismatch_count"] == 0
    assert disk_dig["certificate_digest_mismatch_count"] == 0
    assert disk_dig["edge_certificate_instance_mismatch_count"] == 0
    assert len(disk_dig["missing_expected_instance_nodes"]) == 0

    repo_root = Path(__file__).resolve().parents[1]
    out_json_p = tmp_path / "subproc_cold_resume.json"
    cmd_code = (
        "import json, sys; from pathlib import Path; "
        "from hegel_machine.cognition.m5_0_closure_and_continual_sublation import "
        "resume_m5_0_continual_checkpoint_cold_continuation; "
        "res = resume_m5_0_continual_checkpoint_cold_continuation("
        "Path(sys.argv[1]), continuation_work_dir=Path(sys.argv[2])); "
        "Path(sys.argv[3]).write_text(json.dumps(res), encoding='utf-8')"
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str((repo_root / "src").resolve())
    proc = subprocess.run(
        [sys.executable, "-c", cmd_code, str(archive_bundle), str(tmp_path / "cold_work"), str(out_json_p)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    cold_res = json.loads(out_json_p.read_text(encoding="utf-8"))
    assert cold_res["resumed_from_cursor"] == 28
    assert cold_res["final_cursor"] == 31
    assert cold_res["all_continuation_strict_pass"] is True
    assert cold_res["all_continuation_confirmed_via_recall"] is True
    assert cold_res["any_source_rediscovery_triggered"] is False
    assert cold_res["s30_w7_multicue_evaluated_candidates"] == 1

    r3_c = cold_res["restored_cold_comprehensive_r3"]
    assert r3_c["historical_r3_full"]["all_routes_exact"] is True
    assert r3_c["historical_r3_full"]["authorized_pass_rate"] == 1.0
    assert r3_c["historical_r3_full"]["unauthorized_cleanly_blocked"] is True
    assert r3_c["historical_r3_full"]["any_fit_called"] is False
    assert r3_c["w4_in_scope_r3"]["status"] == "dispatched_ok"
    assert r3_c["w4_above_empirical_bound_r3"]["status"] == "scope_unauthorized"
    assert r3_c["distinct_domain_instances_verified"] is True
    assert r3_c["w6_domain_a_r3"]["within_tolerance"] is True
    assert r3_c["w6_domain_b_r3"]["within_tolerance"] is True
    assert r3_c["w6_negative_domain_c_r3"]["status"] == "scope_unauthorized"
    assert r3_c["w7_domain_a_r3"]["within_tolerance"] is True
    assert r3_c["w7_domain_b_r3"]["within_tolerance"] is True
