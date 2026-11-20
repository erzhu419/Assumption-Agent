"""V17.3 (M4.1) Regression & Integration Suite: Audit Boundary Closures, Executable Pi_s, & Unified Chain."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.cross_domain_clusters import CostMeter
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m3_1_walk_factorization import (
    M3_1FactorizedResearcher,
    build_m3_1_high_load_confirmation_point,
    build_m3_1_pilot_points,
    extract_m3_1_observable_signatures,
    make_m3_1_factorized_walk_suite,
)
import hegel_machine.cognition.m3_scale_universe as m3_mod
from hegel_machine.cognition.m4_1_unified_discovery_and_renormalization import (
    run_executable_pi_s_renormalization_suite,
    run_m4_1_unified_continual_discovery_chain,
)
from hegel_machine.cognition.m4_compositional_discovery import (
    apply_structural_combinator,
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    load_verified_authored_method,
    make_atomic_operator_spec,
)
from hegel_machine.cognition.m4_multiscale_renormalization import (
    build_m4_cluster_targets,
    build_m4_multiscale_forest,
    evaluate_m4_target_under_arm,
)
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.open_workspace_labs import OpenWorkspaceTaskLab
from hegel_machine.cognition.open_workspace_wormhole import (
    canonicalize_python_model_ast,
    validate_workspace_model_on_rows,
)
from hegel_machine.cognition.wormhole_forest import ScaleMap, TwoStageWormholeGate, authorize_abstraction_use


def test_v17_3_m31_and_author_audit_boundary_closures(tmp_path: Path) -> None:
    """Locks in every AST purity, receipt, absolute-path, 4-param combinator, and artifact-integrity fix."""
    # 1. Duplicate `def predict` & `math.exp = ...` attribute shadowing MUST be rejected as impure with distinct digests
    dup_a = canonicalize_python_model_ast("def predict(row):\n    return 4.0\ndef predict(row):\n    return 4.0\n")
    dup_b = canonicalize_python_model_ast("def predict(row):\n    return 4.0\ndef predict(row):\n    return 8.0\n")
    assert dup_a["skeleton_status"] == "UNSUPPORTED_IMPURE_PYTHON"
    assert dup_b["skeleton_status"] == "UNSUPPORTED_IMPURE_PYTHON"
    assert dup_a["ast_skeleton_digest"] != dup_b["ast_skeleton_digest"]

    shad_a = canonicalize_python_model_ast("import math\nmath.exp = lambda x: 4.0\ndef predict(row):\n    return math.exp(1.0)\n")
    shad_b = canonicalize_python_model_ast("import math\nmath.exp = lambda x: 8.0\ndef predict(row):\n    return math.exp(1.0)\n")
    assert shad_a["skeleton_status"] == "UNSUPPORTED_IMPURE_PYTHON"
    assert shad_b["skeleton_status"] == "UNSUPPORTED_IMPURE_PYTHON"
    assert shad_a["ast_skeleton_digest"] != shad_b["ast_skeleton_digest"]

    # 2. Boolean `heldout_confirmed=True` WITHOUT `evidence_receipt` & WITHOUT `target_id` MUST be denied!
    auth_no_rec = authorize_abstraction_use(
        requested_tier="local_solver_guidance",
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.05},
        target_features={},
        claim_contract={
            "claim_id": "c1",
            "family_plugin": "m3_open_workspace_ast",
            "preconditions": [],
            "conclusion_or_action": {"schema_tag": "s1"},
            "protocol_spec": {},
            "tolerance": 0.05,
            "permission_tier": "local_solver_guidance",
        },
        heldout_confirmed=True,
        evidence_receipt=None,
    )
    assert auth_no_rec["granted"] is False

    # 3. Absolute path inside workspace matches relative path; path escaping workspace raises ValueError
    ws = tmp_path / "abs_ws"
    ws.mkdir(parents=True)
    model_file = ws / "model.py"
    model_file.write_text("def predict(row):\n    return 2.0 * row['features']['x']\n", encoding="utf-8")
    lab = OpenWorkspaceTaskLab(
        task_id="abs_test",
        surface_domain="phys",
        feature_names=("x",),
        feature_bounds={"x": (0.0, 5.0)},
        cost_per_obs=1.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.05},
        meter=CostMeter(obs_budget=16, meas_budget=100.0, compute_budget=500.0),
        private_oracle_fn=lambda p: 2.0 * p["x"],
    )
    obs_rows = [Observation(features={"x": 3.0}, target=15.0, history=())]
    r_rel = validate_workspace_model_on_rows(ws, "model.py", obs_rows, lab, training_prefix=[])
    r_abs = validate_workspace_model_on_rows(ws, str(model_file.resolve()), obs_rows, lab, training_prefix=[])
    assert r_rel["rmse"] == pytest.approx(9.0)
    assert r_abs["rmse"] == pytest.approx(9.0)

    ev_abs = evaluate_isolated_python_model(
        ws,
        str(model_file.resolve()),
        [{"features": {"x": 3.0}, "target": 15.0}],
        isolated_prefix_rows=[],
    )
    assert ev_abs["rmse"] == pytest.approx(9.0)
    with pytest.raises(ValueError):
        evaluate_isolated_python_model(ws, str((tmp_path / "outside.py").resolve()), [], isolated_prefix_rows=[])

    # 4. 4-Parameter `exp_relax -> oscillatory_damping -> additive_secondary_shunt` compiles with p0..p3 cleanly!
    osc = apply_structural_combinator(make_atomic_operator_spec("exp_relax_2d"), "oscillatory_damping_modulation")
    osc_shunt = apply_structural_combinator(osc, "additive_secondary_shunt", secondary_role=1)
    compiled_4p = compile_operator_spec_to_python(
        osc_shunt, role_names=["u", "t"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0}
    )
    assert "'p0'" in compiled_4p and "'p1'" in compiled_4p and "'p2'" in compiled_4p and "'p3'" in compiled_4p
    ast_4p = canonicalize_python_model_ast(compiled_4p)
    assert ast_4p["skeleton_status"] == "VERIFIED_PURE_AST"

    # 5. `load_verified_authored_method` rejects metadata-only and tampered operator_spec artifacts
    meta_only_file = tmp_path / "meta_only.json"
    meta_only_file.write_text(json.dumps({"operator_spec": osc_shunt, "confirmed_ok": False}), encoding="utf-8")
    assert load_verified_authored_method(meta_only_file) is None

    valid_art_file = tmp_path / "valid_art.json"
    ns_4p: dict[str, object] = {}
    exec(compile(compiled_4p, "<test_4p>", "exec"), ns_4p)
    p_rows_4p = [
        {"features": {"u": u_v, "t": t_v}, "target": float(ns_4p["predict"]({"features": {"u": u_v, "t": t_v}}))}  # type: ignore[index,operator]
        for u_v, t_v in ((1.0, 0.2), (1.5, 0.8), (2.0, 1.6), (1.8, 2.8))
    ]
    c_rows_4p = [
        {"features": {"u": 1.6, "t": 1.2}, "target": float(ns_4p["predict"]({"features": {"u": 1.6, "t": 1.2}}))}  # type: ignore[index,operator]
    ]
    c_sha_4p = json.dumps(c_rows_4p, sort_keys=True).encode("utf-8")
    p_sha_4p = json.dumps(p_rows_4p, sort_keys=True).encode("utf-8")
    import hashlib as _hl
    valid_payload = {
        "operator_spec": osc_shunt,
        "role_names": ["u", "t"],
        "ast_skeleton_digest": ast_4p["ast_skeleton_digest"],
        "skeleton_status": "VERIFIED_PURE_AST",
        "model_sha256": ast_4p["model_sha256"],
        "authored_model_source": compiled_4p,
        "confirmation_digest": _hl.sha256(c_sha_4p).hexdigest()[:16],
        "host_confirmation_bundle": {
            "task_id": "test_4p",
            "model_sha256": ast_4p["model_sha256"],
            "ast_skeleton_digest": ast_4p["ast_skeleton_digest"],
            "training_prefix_rows": p_rows_4p,
            "training_prefix_sha256": _hl.sha256(p_sha_4p).hexdigest(),
            "confirm_rows": c_rows_4p,
            "confirm_rows_sha256": _hl.sha256(c_sha_4p).hexdigest(),
            "observed_confirm_rmse": 0.0,
            "tolerance": 0.05,
            "confirmed_ok": True,
        },
        "confirmed_ok": True,
    }
    valid_art_file.write_text(json.dumps(valid_payload), encoding="utf-8")
    assert load_verified_authored_method(valid_art_file) is not None

    tampered = deepcopy(valid_payload)
    tampered["operator_spec"] = make_atomic_operator_spec("linear_1d")
    valid_art_file.write_text(json.dumps(tampered), encoding="utf-8")
    assert load_verified_authored_method(valid_art_file) is None


def test_v17_3_multiscale_and_fidelity_audit_closures(tmp_path: Path) -> None:
    """Locks in program_steps interpretation, child Tier-3 authorization, FidelityCompression commit, and coarse-edge liveness."""
    forest, _ = build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    targets, _ = build_m4_cluster_targets(cluster_idx=0)

    # 1. Invalid `program_steps` opcode rejects recall
    bad_prog_f = deepcopy(forest)
    for nd in bad_prog_f.nodes.values():
        nd["program_steps"] = [{"step": "NOT_AN_IMPLEMENTED_OPCODE"}]
    r_intact = evaluate_m4_target_under_arm(
        targets[0],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=forest,
        index=HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=tmp_path / "prog_intact",
    )
    r_bad = evaluate_m4_target_under_arm(
        targets[0],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=bad_prog_f,
        index=HippoRAGWormholeIndex(bad_prog_f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=tmp_path / "prog_bad",
    )
    assert r_intact["confirmed_via_recall"] is True
    assert r_bad["confirmed_via_recall"] is False
    assert r_bad["cost"]["composite_cost"] > r_intact["cost"]["composite_cost"]

    # 2. Child path (`bio_vascular_macro_capped`) enforces Tier-3 permission & blocks UNKNOWN preconditions
    cap_tier_f = deepcopy(forest)
    cap_tier_f.nodes["bio_vascular_macro_capped"]["claim_contract"]["permission_tier"] = "bounded_trial"
    r_cap_tier = evaluate_m4_target_under_arm(
        targets[10],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=cap_tier_f,
        index=HippoRAGWormholeIndex(cap_tier_f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=tmp_path / "child_tier_cap",
    )
    assert r_cap_tier["confirmed_via_recall"] is False

    unk_scope_f = deepcopy(forest)
    unk_scope_f.nodes["bio_vascular_macro_capped"]["claim_contract"]["preconditions"].append(
        {"field": "audit_boundary_measured", "op": "==", "value": True}
    )
    r_unk_scope = evaluate_m4_target_under_arm(
        targets[10],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=unk_scope_f,
        index=HippoRAGWormholeIndex(unk_scope_f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=tmp_path / "child_unk_scope",
    )
    assert r_unk_scope["confirmed_via_recall"] is False

    # 3. `compress_compatible_nodes` rejects wrong-law overlap AND commits validated disjunctive branches onto keeper
    wrong_f = deepcopy(forest)
    wrong_rep = wrong_f.compress_compatible_nodes(
        keeper_node_id="aq_macro_river_tree",
        redundant_node_id="dist_node_00",
        evaluation_queries=[{"active_arity": 1}],
        evaluator_fn=lambda nd, q: "ACC" if int(nd["observation_map"]["parameter_semantics"]["active_arity"]) == q["active_arity"] else "UNK",
    )
    assert wrong_rep["accepted"] is False

    disj_f = deepcopy(forest)
    disj_f.nodes["aq_macro_river_tree"]["claim_contract"]["audit_case"] = "A"
    disj_f.nodes["dist_node_00"]["claim_contract"]["audit_case"] = "B"
    disj_f.nodes["dist_node_00"]["claim_contract"]["conclusion_or_action"] = deepcopy(
        disj_f.nodes["aq_macro_river_tree"]["claim_contract"]["conclusion_or_action"]
    )
    branch_ev = lambda nd, q: (
        "ACC"
        if any(b.get("audit_case") == q["case"] for b in nd["claim_contract"].get("disjunctive_branches", [nd["claim_contract"]]))
        else "UNK"
    )
    disj_rep = disj_f.compress_compatible_nodes(
        keeper_node_id="aq_macro_river_tree",
        redundant_node_id="dist_node_00",
        evaluation_queries=[{"case": "A"}, {"case": "B"}],
        evaluator_fn=branch_ev,
    )
    assert disj_rep["accepted"] is True
    assert branch_ev(disj_f.nodes["aq_macro_river_tree"], {"case": "B"}) == "ACC"

    # 4. Coarse-graining edge liveness: preserved on protocol update, marked stale on scale_map change, restored on revalidation
    rev_f = deepcopy(forest)
    assert HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 6
    cc_p = deepcopy(rev_f.nodes["aq_meso_hex_cell"]["claim_contract"])
    cc_p["protocol_spec"]["rev_note"] = 2
    rev_f.empirical_revise_node("aq_meso_hex_cell", witness_episode_id="ep_p", cause="protocol_or_scope_mismatch", updated_claim_contract=cc_p)
    assert HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 6

    new_sm = ScaleMap(
        coarse_graining_op="pi_changed_window",
        window_or_partition={"scale_regime": "meso"},
        preserved_queries=["open_workspace_mechanism_prediction"],
        closure_scope="not_closed",
        proposed_discarded_variables=["fast_micro_mode"],
        verified_discarded_in_scope=[],
        error_budget=0.25,
    ).to_dict()
    rev_f.empirical_revise_node("aq_meso_hex_cell", witness_episode_id="ep_sm", cause="abstraction_closure_loss", updated_scale_map=new_sm)
    assert HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 4
    rev_f.revalidate_vertical_edges_after_revision("aq_meso_hex_cell", witness_episode_id="ep_reval", reason="reverified_pi")
    assert HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 6


def test_v17_3_m3_1_single_flag_update_and_complete_2x2_factorization(tmp_path: Path) -> None:
    """Verifies the 4th cell (`hipporag_alpha0_nopool`) of the 2x2 retrieval factorization and single-flag update."""
    orig_p, orig_c, orig_e = m3_mod.build_m3_pilot_points, m3_mod.build_m3_confirmation_point, m3_mod.extract_m3_observable_signatures
    m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
    m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
    m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures
    try:
        suite = make_m3_1_factorized_walk_suite(8_028_101, target_k=36)
        base = M3_1FactorizedResearcher(
            gate=TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=16),
            retrieval_mode="hipporag_ppr",
            damping_alpha=0.65,
        )
        base.form_and_revise_episodes("m3_1_2x2_c1", suite, tmp_path / "src_2x2")

        modes = (
            "hipporag_ppr",           # Cell (Walk=1, Pool=1)
            "hipporag_alpha0_pool",   # Cell (Walk=0, Pool=1)
            "hipporag_walk_nopool",   # Cell (Walk=1, Pool=0)
            "hipporag_alpha0_nopool", # Cell (Walk=0, Pool=0) -- the 4th cell!
        )
        arms = {m: base.clone_with_mode(m) for m in modes}
        walk_wins = {m: 0 for m in modes}
        for _cat, tname, (tfac, tev) in suite["walk_targets"]:
            for m in modes:
                lb = tfac()
                ws = tmp_path / m / tname
                sub = arms[m].solve_target(tname, lb, ws)
                ev = tev(ws, sub, lb.observations)
                assert ev["strict_pass"] is True
                if sub.get("selected_reached_only_by_walk"):
                    walk_wins[m] += 1

        assert walk_wins["hipporag_ppr"] >= 3
        assert walk_wins["hipporag_alpha0_pool"] == 0
        assert walk_wins["hipporag_alpha0_nopool"] == 0
    finally:
        m3_mod.build_m3_pilot_points = orig_p
        m3_mod.build_m3_confirmation_point = orig_c
        m3_mod.extract_m3_observable_signatures = orig_e


def test_v17_3_executable_pi_s_fast_slow_renormalization_and_coarse_edge_causality(tmp_path: Path) -> None:
    """Verifies RK4 fast-slow ODE + executable Pi_s + regional box closure + causal coarse-graining edge utility."""
    res = run_executable_pi_s_renormalization_suite(tmp_path / "pi_s_suite")

    traj = res["trajectory_projection_verification"]
    assert traj["adiabatic_t4_fast_slaving_defect"] < 0.015
    assert traj["quench_t008_fast_slaving_defect"] > 0.40
    assert traj["in_box_query_accepted"] is True
    assert traj["out_of_box_query_rejected"] is True

    cg = res["adiabatic_and_coarse_edge_causality"]
    assert cg["coarse_edge_causally_active"] is True
    assert cg["With_CoarseGraining_Edges"]["strict_pass"] is True
    assert cg["With_CoarseGraining_Edges"]["walked_coarse_graining_edge"] is True

    assert res["short_time_quench_escalation"]["strict_pass"] is True
    assert res["short_time_quench_escalation"]["selected_channel"] == "micro_fine"

    bump = res["interior_micro_bump_regional_vs_single_point"]
    assert bump["single_point_misses_interior_bump"] is True
    assert bump["regional_box_catches_bump_zero_false_confirm"] is True
    assert bump["Regional_5Pt_Box_Probe"]["false_confirmation"] is False
    assert bump["Regional_5Pt_Box_Probe"]["status"] == "scoped_unknown"


def test_v17_3_unified_continual_discovery_forest_chain_beats_both_orders_and_frozen(tmp_path: Path) -> None:
    """Verifies the full unbroken M4.1 chain: Stage-0 identical start -> CodeAgent compositional discovery ->
    PhilosophicalForest registration -> 3-tier R3 -> empirical DEV lure archival -> beating Forward, Reversed,
    Flat, Alpha0, and fair Frozen_After_Stage0!
    """
    res = run_m4_1_unified_continual_discovery_chain(tmp_path / "unified_chain")

    for w_key in ("W1_haldane_shunt_2d", "W2_oscillatory_shunt_4p", "W3_murray_capped_1d"):
        assert res["stage1_discovered_witnesses"][w_key]["strict_pass"] is True

    r3 = res["r3_three_tier_old_scope_preservation"]
    assert r3["tier1_exact_frozen_s1_pass"] is True
    assert r3["tier2_composed_family_refit_pass"] is True
    assert r3["tier3_scope_routed_both_old_and_new_no_refit_pass"] is True

    gov = res["stage2_lifecycle_governance"]
    assert gov["compressed_compatible_duplicates"] == 3
    assert gov["wrong_law_merge_rejected"] is True
    assert gov["both_lures_empirically_archived"] is True

    r1 = res["r1_forest_artifact_integrity_controls"]
    assert r1["metadata_only_rejected_properly"] is True
    assert r1["deletion_increases_cost_and_restoration_recovers"] is True

    inv = res["key_invariants"]
    assert inv["hegel_strictly_beats_both_forward_and_reversed_ordinary"] is True
    assert inv["hegel_strictly_beats_flat_and_alpha0_via_walk"] is True
    assert inv["hegel_strictly_beats_fair_frozen_stage0_plasticity"] is True

    h_sum = res["stage3_arm_summaries"]["Hegel_M4_1_UnifiedForest"]
    assert h_sum["strict_pass_rate_12"] == 1.0
    assert h_sum["recall_rate_12"] == 1.0
    assert h_sum["false_confirmations_14"] == 0
    assert h_sum["oos_scoped_unknown_rate"] == 1.0
