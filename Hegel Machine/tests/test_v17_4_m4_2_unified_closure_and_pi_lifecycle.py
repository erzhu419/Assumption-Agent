"""V17.4 (M4.2) Regression & Integration Suite: Audit Closures, Executable Pi_s, Zero-Fit Dispatcher & Unified Chain."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m4_1_unified_discovery_and_renormalization import (
    _load_node_verified_operator_spec,
    run_executable_pi_s_renormalization_suite,
)
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import (
    build_and_evolve_m4_2_cross_pillar_forest,
    certify_regional_pi_s_closure_with_trajectory_projection,
    run_deduplicated_m3_1_2x2_factorization,
    run_interleaved_zero_fit_dispatcher_stream,
    run_m4_2_unified_cross_pillar_study,
)
from hegel_machine.cognition.m4_compositional_discovery import (
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    load_verified_authored_method,
    make_atomic_operator_spec,
    predict_with_frozen_instance,
)
import hegel_machine.cognition.m4_multiscale_renormalization as m4_ms
from hegel_machine.cognition.open_workspace_wormhole import canonicalize_python_model_ast
from hegel_machine.cognition.wormhole_forest import ScaleMap, authorize_abstraction_use


def test_v17_4_m4_1_audit_closure_witnesses(tmp_path: Path) -> None:
    """Locks in 100% closure of all counterfactual probes from `Hegel_M4_1_Audit/closure/`."""
    forest, _ = m4_ms.build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    targets, _ = m4_ms.build_m4_cluster_targets(cluster_idx=0)

    # 1. `program_steps` `fresh_points` parameter genuinely changes confirmation acquisitions (`1` vs `2`)!
    def _run_with_fresh(fp: int) -> tuple[dict, int]:
        f_clone = deepcopy(forest)
        for nd in f_clone.nodes.values():
            for st in nd.get("program_steps", []):
                if st.get("step") == "subprocess_isolated_confirm":
                    st["fresh_points"] = fp
        acq_counts: list[int] = []
        orig_acq = m4_ms.MultiScaleWorkspaceTaskLab.acquire_channel

        def _rec_acq(self, points, *, channel="coarse_macro", phase="pilot"):
            rows = orig_acq(self, points, channel=channel, phase=phase)
            if phase.startswith("confirm"):
                acq_counts.append(len(points))
            return rows

        with patch.object(m4_ms.MultiScaleWorkspaceTaskLab, "acquire_channel", _rec_acq):
            res = m4_ms.evaluate_m4_target_under_arm(
                targets[0],
                arm_name="Hegel_M4_MultiScale_HippoWalk",
                forest=f_clone,
                index=HippoRAGWormholeIndex(f_clone, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                stage1_gate=None,
                work_dir=tmp_path / f"fp_{fp}",
            )
        return res, sum(acq_counts)

    r1, c1 = _run_with_fresh(1)
    r2, c2 = _run_with_fresh(2)
    assert c1 == 1 and c2 == 2
    assert r2["cost"]["measurement_cost"] > r1["cost"]["measurement_cost"]

    # 2. Disjoint-scope `FidelityCompression` OR-branches are genuinely consumed by `authorize_abstraction_use`!
    f_comp = deepcopy(forest)
    a_id, b_id = "aq_macro_river_tree", "dist_node_00"
    for nid, sc in ((a_id, "A"), (b_id, "B")):
        nd = f_comp.nodes[nid]
        nd["observation_map"] = deepcopy(f_comp.nodes[a_id]["observation_map"])
        nd["scale_map"] = deepcopy(f_comp.nodes[a_id]["scale_map"])
        nd["claim_contract"]["conclusion_or_action"] = deepcopy(
            f_comp.nodes[a_id]["claim_contract"]["conclusion_or_action"]
        )
        nd["claim_contract"]["preconditions"] = [{"field": "audit_scope", "op": "==", "value": sc}]

    rep = f_comp.compress_compatible_nodes(
        keeper_node_id=a_id,
        redundant_node_id=b_id,
        evaluation_queries=[{"audit_scope": "A"}, {"audit_scope": "B"}, {"audit_scope": "C"}],
        evaluator_fn=lambda nd, q: (
            "ACC"
            if any(
                all(q.get(p["field"]) == p["value"] for p in br["preconditions"])
                for br in (nd["claim_contract"].get("disjunctive_branches") or [nd["claim_contract"]])
            )
            else "UNK"
        ),
    )
    assert rep["accepted"] is True
    merged_nd = f_comp.nodes[a_id]
    assert merged_nd["claim_contract"]["preconditions"] == []
    assert len(merged_nd["claim_contract"]["disjunctive_branches"]) == 2
    for sc, expected in (("A", True), ("B", True), ("C", False)):
        auth = authorize_abstraction_use(
            requested_tier="bounded_trial",
            task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.1},
            target_features={"audit_scope": sc},
            claim_contract=merged_nd["claim_contract"],
            scale_map=merged_nd["scale_map"],
        )
        assert auth["granted"] is expected

    # 3. `load_verified_authored_method` rejects self-consistent JSON without confirmation provenance
    #    AND `_load_node_verified_operator_spec` rejects cross-node artifact swap (`W2` into `W1`)!
    sp_mm = make_atomic_operator_spec("michaelis_1d")
    code_mm = compile_operator_spec_to_python(sp_mm, role_names=["x"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0})
    ast_mm = canonicalize_python_model_ast(code_mm)
    unconfirmed_json = tmp_path / "self_consistent_no_confirm.json"
    unconfirmed_json.write_text(
        json.dumps({
            "operator_spec": sp_mm,
            "role_names": ["x"],
            "ast_skeleton_digest": ast_mm["ast_skeleton_digest"],
            "skeleton_status": ast_mm["skeleton_status"],
            "model_sha256": hashlib.sha256(code_mm.encode("utf-8")).hexdigest(),
            "authored_model_source": code_mm,
            "confirmation_digest": "no_confirmation_was_run",
            "confirmed_ok": True,
        }),
        encoding="utf-8",
    )
    assert load_verified_authored_method(unconfirmed_json) is None

    # Build two genuine host-confirmed V17.4 artifacts (`art_a` and `art_b`) and swap `art_b` into `node_a`
    def _make_confirmed_art(sp_obj: dict, val_fn) -> dict:
        c_src = compile_operator_spec_to_python(sp_obj, role_names=["x"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0})
        a_inf = canonicalize_python_model_ast(c_src)
        p_r = [{"features": {"x": x}, "target": float(val_fn(x))} for x in (0.5, 1.5, 3.0, 5.0)]
        c_r = [{"features": {"x": 2.5}, "target": float(val_fn(2.5))}]
        c_sha = hashlib.sha256(json.dumps(c_r, sort_keys=True).encode("utf-8")).hexdigest()
        p_sha = hashlib.sha256(json.dumps(p_r, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "operator_spec": sp_obj,
            "role_names": ["x"],
            "ast_skeleton_digest": a_inf["ast_skeleton_digest"],
            "skeleton_status": "VERIFIED_PURE_AST",
            "model_sha256": a_inf["model_sha256"],
            "authored_model_source": c_src,
            "confirmation_digest": c_sha[:16],
            "host_confirmation_bundle": {
                "task_id": "t_swap",
                "model_sha256": a_inf["model_sha256"],
                "ast_skeleton_digest": a_inf["ast_skeleton_digest"],
                "training_prefix_rows": p_r,
                "training_prefix_sha256": p_sha,
                "confirm_rows": c_r,
                "confirm_rows_sha256": c_sha,
                "observed_confirm_rmse": 0.0,
                "tolerance": 0.05,
                "confirmed_ok": True,
            },
            "confirmed_ok": True,
        }

    art_a = _make_confirmed_art(sp_mm, lambda x: 6.0 * x / (2.0 + x))
    art_b = _make_confirmed_art(make_atomic_operator_spec("linear_1d"), lambda x: 2.0 * x)
    swap_file = tmp_path / "swap_slot.json"
    swap_file.write_text(json.dumps(art_a), encoding="utf-8")
    node_a = {
        "claim_contract": {
            "conclusion_or_action": {
                "operator_tag": "op_mm",
                "operator_spec": deepcopy(sp_mm),
                "ast_skeleton_digest": art_a["ast_skeleton_digest"],
                "authored_artifact_path": str(swap_file),
            }
        }
    }
    assert _load_node_verified_operator_spec(node_a) == sp_mm
    swap_file.write_text(json.dumps(art_b), encoding="utf-8")
    assert _load_node_verified_operator_spec(node_a) is None

    # 4. Relative parent-path escape (`../outside.py`) raises `ValueError`!
    ws_sub = tmp_path / "sandbox_a" / "sandbox_b"
    ws_sub.mkdir(parents=True)
    outside_py = tmp_path / "outside.py"
    outside_py.write_text("def predict(row):\n    return 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes workspace boundary"):
        evaluate_isolated_python_model(ws_sub, "../../outside.py", [], isolated_prefix_rows=[])

    # 5. `revalidate_vertical_edges_with_certificate` blocks unclosed / uncertified rebinds!
    vf = deepcopy(forest)
    sm_nc = ScaleMap(
        coarse_graining_op="pi_changed",
        window_or_partition={"scale_regime": "meso"},
        preserved_queries=["open_workspace_mechanism_prediction"],
        closure_scope="not_closed",
        proposed_discarded_variables=["fast_micro_mode"],
        verified_discarded_in_scope=[],
        error_budget=0.25,
    ).to_dict()
    vf.empirical_revise_node("aq_meso_hex_cell", witness_episode_id="ep_nc", cause="abstraction_closure_loss", updated_scale_map=sm_nc)
    assert HippoRAGWormholeIndex(vf, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 4
    rej = vf.revalidate_vertical_edges_with_certificate(
        "aq_meso_hex_cell",
        witness_episode_id="ep_bad_cert",
        closure_certificate={"certified": True, "max_closure_defect": 0.01, "tolerance": 0.05},
    )
    assert rej["accepted"] is False
    assert HippoRAGWormholeIndex(vf, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 4

    sm_closed = deepcopy(sm_nc)
    sm_closed["closure_scope"] = "regional_box_verified"
    sm_closed["verified_discarded_in_scope"] = ["fast_micro_mode"]
    ok_reval = vf.revalidate_vertical_edges_with_certificate(
        "aq_meso_hex_cell",
        witness_episode_id="ep_good_cert",
        closure_certificate={"certified": True, "max_closure_defect": 0.012, "tolerance": 0.05},
        updated_scale_map=sm_closed,
    )
    assert ok_reval["accepted"] is True
    assert HippoRAGWormholeIndex(vf, damping_alpha=0.65).edge_counts["coarse_graining_edges"] == 6


def test_v17_4_executable_pi_s_box_gating_trajectory_slaving_and_zero_dup_fallback(tmp_path: Path) -> None:
    """Locks in all `pi/probe_pi_paths.py` fixes: trajectory projection, out-of-box fallback/export guard, & 0 dup rows."""
    cert = certify_regional_pi_s_closure_with_trajectory_projection(tmp_path / "pi_cert")
    assert cert["certified"] is True
    assert cert["projections_with_fine_trajectory"] == 6
    assert cert["max_closure_defect"] <= 0.045
    assert cert["max_trajectory_slaving_defect"] <= 0.045

    suite = run_executable_pi_s_renormalization_suite(tmp_path / "pi_suite")
    cf = suite["out_of_box_confirm_and_heldout_counterfactuals"]

    # Out-of-box confirmation point (`horizon_t = 1.2 < 2.2`) MUST escalate to `micro_fine` with 0 false confirmations!
    res_oob_cf = cf["outside_confirm_fallback"]
    assert res_oob_cf["selected_channel"] == "micro_fine"
    assert res_oob_cf["strict_pass"] is True
    assert res_oob_cf["false_confirmation"] is False

    # Out-of-box exported heldout query (`horizon_t = 0.12`) after valid in-box confirmation MUST be blocked by scope contract!
    res_oob_ho = cf["outside_heldout_query"]
    assert res_oob_ho["status"] == "out_of_verified_scale_box"
    assert res_oob_ho["out_of_box_query_blocked"] is True
    assert res_oob_ho["false_confirmation"] is False


def test_v17_4_deduplicated_m3_1_2x2_factorization_single_seed(tmp_path: Path) -> None:
    """Verifies clean deduplicated `27`-target/seed M3.1 2x2 factorization (zero walk-target double counting)."""
    rep = run_deduplicated_m3_1_2x2_factorization(tmp_path / "m31_dedup", seeds=(8_028_101,))
    assert rep["duplicate_walk_targets_included"] is False
    assert rep["unique_runs_per_cell"] == 27
    assert rep["total_unique_runs"] == 108
    cells = rep["cells"]
    for c_st in cells.values():
        assert c_st["strict_pass_rate"] == 1.0
        assert c_st["false_confirmation_rate"] == 0.0
    assert cells["hipporag_ppr"]["unique_non_direct_walk_wins"] >= 3
    assert cells["hipporag_alpha0_pool"]["unique_non_direct_walk_wins"] == 0
    assert cells["hipporag_alpha0_nopool"]["unique_non_direct_walk_wins"] == 0
    assert cells["hipporag_ppr"]["mean_composite_cost"] < cells["hipporag_alpha0_pool"]["mean_composite_cost"]
    assert cells["hipporag_walk_nopool"]["mean_composite_cost"] < cells["hipporag_alpha0_nopool"]["mean_composite_cost"]


def test_v17_4_zero_fit_forest_scope_dispatcher_and_unified_cross_pillar_chain(tmp_path: Path) -> None:
    """Verifies true zero-`fit()` `ForestScopeDispatcher` (`R3`) AND full unified cross-pillar dominance."""
    rep = run_m4_2_unified_cross_pillar_study(tmp_path / "m42_full_1seed", seeds=(20_261_120,))

    # 1. Stage 0, Stage 1, Stage 2 cross-pillar invariants
    assert rep["stage0"]["composite_cost"] > 0.0
    assert rep["stage0"]["pi_s_initial_certificate"]["projections_with_fine_trajectory"] == 6
    assert len(rep["stage1"]["discoveries"]) == 5
    assert all(d["strict_pass"] for d in rep["stage1"]["discoveries"].values())
    s2 = rep["stage2"]
    assert s2["compression_report"]["accepted"] is True
    assert s2["compression_report"]["real_authorize_abstraction_use_by_modality"] == {
        "thermal": True,
        "hydraulic": True,
        "optical": False,
    }
    assert len({ep["surface_domain"] for ep in s2["dev_lure_episodes"]}) == 3
    assert s2["lure_archived_by_minton"] is True
    assert s2["quench_revision"]["rejected_unclosed_revalidation"]["accepted"] is False
    assert s2["quench_revision"]["certified_revalidation"]["accepted"] is True
    edges = s2["edge_counts_post_stage2"]
    assert edges["coarse_graining_edges"] >= 2
    assert edges["horizontal_wormhole_edges"] >= 2
    assert edges["vertical_tree_edges"] >= 4

    # 2. Zero-`fit()` `ForestScopeDispatcher` (`R3` instance-level revisit without `fit()`!)
    disp = rep["dispatcher_zero_fit_r3"]
    assert disp["query_count"] == 10
    assert disp["any_fit_called"] is False
    assert disp["all_routes_exact"] is True
    assert disp["authorized_pass_rate"] == 1.0
    assert disp["unauthorized_cleanly_blocked"] is True

    # 3. Stage 3 unseen transfer dominance over DEV-frozen static order, Forward, Reversed, Alpha0, Flat, & Frozen!
    arms = rep["stage3_arms"]
    hegel = arms["Hegel_Unified_Continual_Forest"]
    assert hegel["strict_pass_rate"] == 1.0
    assert hegel["false_confirmation_rate"] == 0.0
    assert hegel["oos_scoped_unknown_rate"] == 1.0
    assert hegel["oos_false_confirmation_rate"] == 0.0
    assert hegel["non_direct_walk_wins"] >= 10

    for baseline in (
        "Hegel_NoMaintenance_SameHistory",
        "Ordinary_AllHistory_DevFrozenOrder",
        "Ordinary_AllHistory_Forward",
        "Ordinary_AllHistory_Reversed",
        "Forest_Alpha0_SameHistory",
        "Flat_BM25_SameHistory",
        "Frozen_After_Stage0_OpenRediscovery",
        "Frozen_After_Stage0_PoolFallback",
    ):
        assert arms[baseline]["strict_pass_rate"] == 1.0
        assert arms[baseline]["false_confirmation_rate"] == 0.0
        assert arms[baseline]["oos_false_confirmation_rate"] == 0.0
        assert hegel["mean_composite_cost"] < arms[baseline]["mean_composite_cost"], (
            f"Expected Hegel ({hegel['mean_composite_cost']}) < {baseline} ({arms[baseline]['mean_composite_cost']})"
        )
    assert hegel["mean_measurement_cost"] < arms["Ordinary_AllHistory_DevFrozenOrder"]["mean_measurement_cost"]
    assert hegel["mean_measurement_cost"] < arms["Ordinary_AllHistory_Forward"]["mean_measurement_cost"]
    assert hegel["mean_measurement_cost"] < arms["Ordinary_AllHistory_Reversed"]["mean_measurement_cost"]
