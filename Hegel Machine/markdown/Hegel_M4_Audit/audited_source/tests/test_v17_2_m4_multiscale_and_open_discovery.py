"""Tests for Hegel V17.2 (`M4`): Open Compositional Discovery, Active Multi-Scale Renormalization & K=64 Continual Lifecycle."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from hegel_machine.cognition.m4_compositional_discovery import run_m4_compositional_discovery_suite
from hegel_machine.cognition.m4_multiscale_renormalization import (
    MultiScaleTargetSpec,
    MultiScaleWorkspaceTaskLab,
    build_m4_cluster_targets,
    build_m4_multiscale_forest,
    evaluate_m4_target_under_arm,
    extract_multiscale_pilot_signatures,
    run_m4_continual_lifecycle_stream,
)


def test_v17_2_compositional_counterfactual_witnesses_and_pure_ast(tmp_path: Path) -> None:
    res = run_m4_compositional_discovery_suite(tmp_path / "comp_suite")
    cf = res["counterfactual_summary"]
    assert cf["all_5_witnesses_verified_pure_ast"] is True
    assert cf["all_5_witnesses_strict_pass"] is True
    assert cf["monod_3_witnesses_distinct_pure_ast_count"] == 3
    assert cf["time_2_witnesses_distinct_pure_ast_count"] == 2
    assert cf["all_monod_witnesses_preserve_s1_scope"] is True

    act = res["artifact_causality_and_transfer"]
    assert act["Hegel_Intact_Artifact"]["strict_pass"] is True
    assert act["Hegel_Deleted_Artifact"]["strict_pass"] is True
    assert act["Hegel_Restored_Artifact"]["strict_pass"] is True
    assert act["Ordinary_AllHistory_EqualPilot"]["strict_pass"] is True
    assert act["deletion_changes_behavior_and_cost"] is True
    assert act["restoration_recovers_exact_behavior"] is True
    assert act["hegel_beats_equal_pilot_all_history_compute"] is True


def test_v17_2_scale_indexed_wormholes_and_coarse_graining_edges() -> None:
    forest, _ = build_m4_multiscale_forest(cluster_seed=42, extra_distractor_count=6)
    assert len(forest.coarse_graining_edges) == 6

    # Same aqueous tree has distinct wormhole neighbors across micro, meso, and macro scales!
    micro_peers = {
        (wh["node_b_id"] if wh["node_a_id"] == "aq_micro_polar" else wh["node_a_id"])
        for wh in forest.horizontal_wormhole_edges.values()
        if "aq_micro_polar" in (wh["node_a_id"], wh["node_b_id"])
    }
    meso_peers = {
        (wh["node_b_id"] if wh["node_a_id"] == "aq_meso_hex_cell" else wh["node_a_id"])
        for wh in forest.horizontal_wormhole_edges.values()
        if "aq_meso_hex_cell" in (wh["node_a_id"], wh["node_b_id"])
    }
    macro_peers = {
        (wh["node_b_id"] if wh["node_a_id"] == "aq_macro_river_tree" else wh["node_a_id"])
        for wh in forest.horizontal_wormhole_edges.values()
        if "aq_macro_river_tree" in (wh["node_a_id"], wh["node_b_id"])
    }
    assert micro_peers == {"chem_ammonia_micro", "sys_queue_micro"}
    assert "bio_wax_hydrophobic_micro" not in micro_peers
    assert meso_peers == {"bio_honeycomb_meso", "sys_shard_meso"}
    assert macro_peers == {"bio_vascular_macro"}

    # Verify coarse-graining vertical traversal (`micro <-> meso <-> macro`)
    assert [n["node_id"] for n in forest.fine_scale_children_of("aq_meso_hex_cell")] == ["aq_micro_polar"]
    assert [n["node_id"] for n in forest.coarse_scale_parents_of("aq_meso_hex_cell")] == ["aq_macro_river_tree"]


def test_v17_2_active_scale_closure_probe_and_multiscale_2hop_walk(tmp_path: Path) -> None:
    forest, _ = build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=12)
    targets, oos_targets = build_m4_cluster_targets(cluster_idx=0)

    idx_walk = HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
    idx_alpha0 = HippoRAGWormholeIndex(forest, damping_alpha=0.0, wormhole_equivalence_pooling=True, scale_aware=True)
    idx_blind = HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=False)
    idx_flat = FlatEqualMetadataIndex(forest, mode="bm25", wormhole_equivalence_pooling=True)
    idx_vert = HippoRAGWormholeIndex(
        forest, allow_horizontal_wormholes=False, wormhole_equivalence_pooling=False, damping_alpha=0.65, scale_aware=True
    )

    hegel_runs = [
        evaluate_m4_target_under_arm(t, arm_name="Hegel_M4_MultiScale_HippoWalk", forest=forest, index=idx_walk, stage1_gate=None, work_dir=tmp_path / f"h_{t.target_id}")
        for t in targets
    ]
    alpha0_runs = [
        evaluate_m4_target_under_arm(t, arm_name="HippoPPR_Alpha0_MultiScalePool", forest=forest, index=idx_alpha0, stage1_gate=None, work_dir=tmp_path / f"a0_{t.target_id}")
        for t in targets
    ]
    blind_runs = [
        evaluate_m4_target_under_arm(t, arm_name="HippoPPR_ScaleBlind_Walk", forest=forest, index=idx_blind, stage1_gate=None, work_dir=tmp_path / f"sb_{t.target_id}")
        for t in targets
    ]
    flat_runs = [
        evaluate_m4_target_under_arm(t, arm_name="Flat_BM25_WormholePool_EqualContent", forest=forest, index=idx_flat, stage1_gate=None, work_dir=tmp_path / f"fl_{t.target_id}")
        for t in targets
    ]
    vert_runs = [
        evaluate_m4_target_under_arm(t, arm_name="Forest_VerticalOnly_NoWormholes", forest=forest, index=idx_vert, stage1_gate=None, work_dir=tmp_path / f"vt_{t.target_id}")
        for t in targets
    ]
    micro_only_runs = [
        evaluate_m4_target_under_arm(t, arm_name="MicroOnly_AlwaysFine_Forest", forest=forest, index=idx_walk, stage1_gate=None, work_dir=tmp_path / f"mo_{t.target_id}")
        for t in targets
    ]
    oos_runs = [
        evaluate_m4_target_under_arm(t, arm_name="Hegel_M4_MultiScale_HippoWalk", forest=forest, index=idx_walk, stage1_gate=None, work_dir=tmp_path / f"oos_{t.target_id}")
        for t in oos_targets
    ]

    assert all(r["strict_pass"] and not r["false_confirmation"] for r in hegel_runs)
    assert all(r["status"] == "scoped_unknown" and not r["false_confirmation"] for r in oos_runs)

    # 1. Adiabatic coarse targets save >45% measurement cost over MicroOnly
    h_adiab_m = sum(r["cost"]["measurement_cost"] for r in hegel_runs if r["category"] == "T_Adiabatic_Coarse")
    mo_adiab_m = sum(r["cost"]["measurement_cost"] for r in micro_only_runs if r["category"] == "T_Adiabatic_Coarse")
    assert h_adiab_m < 0.58 * mo_adiab_m

    # 2. Closure breakdown targets trigger escalation (`escalated_to_micro == True`) and beat ScaleBlind
    h_brk = [r for r in hegel_runs if r["category"] == "T_Closure_Breakdown"]
    sb_brk = [r for r in blind_runs if r["category"] == "T_Closure_Breakdown"]
    assert all(r["escalated_to_micro"] is True for r in h_brk)
    assert sum(r["cost"]["composite_cost"] for r in h_brk) < sum(r["cost"]["composite_cost"] for r in sb_brk)

    # 3. Safe-pilot MultiScale Walk targets succeed via 2-hop walk ONLY when alpha > 0 and horizontal wormholes exist!
    h_walk_wins = sum(1 for r in hegel_runs if r["category"] == "T_MultiScale_Walk" and r["non_direct_walk_win"])
    a0_walk_wins = sum(1 for r in alpha0_runs if r["category"] == "T_MultiScale_Walk" and r["non_direct_walk_win"])
    fl_walk_wins = sum(1 for r in flat_runs if r["category"] == "T_MultiScale_Walk" and r["non_direct_walk_win"])
    vt_walk_wins = sum(1 for r in vert_runs if r["category"] == "T_MultiScale_Walk" and r["non_direct_walk_win"])
    assert h_walk_wins >= 3
    assert a0_walk_wins == 0
    assert fl_walk_wins == 0
    assert vt_walk_wins == 0


def test_v17_2_s2_coordinate_permutation_invariance(tmp_path: Path) -> None:
    targets, _ = build_m4_cluster_targets(cluster_idx=1)
    perm_t = next(t for t in targets if t.category == "T_DualSensor_Permuted")
    rows_a = perm_t.lab.acquire_channel(perm_t.pilot_points, channel="coarse_macro")
    sigs_a, _ = extract_multiscale_pilot_signatures(
        rows_a, feature_names=("plate_aspect", "plate_area"), scale_probe={"closure_verified": True}, domain_cue="cue_shard_boundary"
    )
    sigs_b, _ = extract_multiscale_pilot_signatures(
        rows_a, feature_names=("plate_area", "plate_aspect"), scale_probe={"closure_verified": True}, domain_cue="cue_shard_boundary"
    )
    assert sigs_a == sigs_b
    assert "sig:variational_aspect_penalty" in sigs_a
    assert "sig:sensor_neg" in sigs_a


def test_v17_2_continual_k64_fidelity_compression_and_zero_forgetting(tmp_path: Path) -> None:
    res = run_m4_continual_lifecycle_stream(tmp_path / "k64_stream")
    assert res["zero_backward_forgetting_verified"] is True
    assert res["zero_stale_wormholes_left_unrevalidated"] is True
    assert res["final_k64_active_nodes_full_lifecycle"] <= 20
    assert res["final_k64_active_nodes_unpruned"] >= 58

    k64_arms = res["stages"][-1]["arms"]
    hegel_c = k64_arms["Hegel_M4_FullLifecycle"]["mean_composite_cost"]
    assert hegel_c < k64_arms["Forest_NoCompression_NoPruning"]["mean_composite_cost"]
    assert hegel_c < k64_arms["Flat_BM25_WormholePool_Continual"]["mean_composite_cost"]
    assert hegel_c < k64_arms["Continuing_AllHistory_CodeAuthor"]["mean_composite_cost"]
    assert hegel_c < k64_arms["Frozen_Stage1_Memory"]["mean_composite_cost"]
