"""Execute Preregistered Hegel V17.3 (M4.1) Audit Closure, Complete 2x2 Factorization,
Executable Dynamical Pi_s Renormalization, and Unified Continual Discovery-to-Forest Study.

Writes full static evidence bundle exclusively to:
  `docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/`
without modifying ANY prior V16.8, V16.9, V17.0, V17.1, or V17.2 evidence or audit directories!
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Any

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
    build_unified_12_target_suite,
    evaluate_unified_target_under_arm,
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


REPO_ROOT = Path(__file__).resolve().parents[1]
EVID_DIR = REPO_ROOT / "docs" / "evidence" / "v17_3_m4_1_unified_discovery_and_renormalization_20261120"

M3_1_EVAL_SEEDS = [
    ("m3_1_eval_01", 8_028_101),
    ("m3_1_eval_02", 8_028_202),
    ("m3_1_eval_03", 8_028_303),
    ("m3_1_eval_04", 8_028_404),
]

UNIFIED_EVAL_SEEDS = [20261120, 20261221, 20261322, 20261423]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_part1_audit_closures(work_dir: Path) -> dict[str, Any]:
    """Verify all boundary, author, and lifecycle closures from Hegel_M4_Audit."""
    work_dir.mkdir(parents=True, exist_ok=True)

    dup_a = canonicalize_python_model_ast("def predict(row):\n    return 4.0\ndef predict(row):\n    return 4.0\n")
    dup_b = canonicalize_python_model_ast("def predict(row):\n    return 4.0\ndef predict(row):\n    return 8.0\n")
    shad_a = canonicalize_python_model_ast("import math\nmath.exp = lambda x: 4.0\ndef predict(row):\n    return math.exp(1.0)\n")
    shad_b = canonicalize_python_model_ast("import math\nmath.exp = lambda x: 8.0\ndef predict(row):\n    return math.exp(1.0)\n")

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

    ws_abs = work_dir / "abs_check"
    ws_abs.mkdir(parents=True, exist_ok=True)
    mfile = ws_abs / "model.py"
    mfile.write_text("def predict(row):\n    return 2.0 * row['features']['x']\n", encoding="utf-8")
    lab = OpenWorkspaceTaskLab(
        task_id="abs_check",
        surface_domain="phys",
        feature_names=("x",),
        feature_bounds={"x": (0.0, 5.0)},
        cost_per_obs=1.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.05},
        meter=CostMeter(obs_budget=16, meas_budget=100.0, compute_budget=500.0),
        private_oracle_fn=lambda p: 2.0 * p["x"],
    )
    obs_rows = [Observation(features={"x": 3.0}, target=15.0, history=())]
    r_rel = validate_workspace_model_on_rows(ws_abs, "model.py", obs_rows, lab, training_prefix=[])
    r_abs = validate_workspace_model_on_rows(ws_abs, str(mfile.resolve()), obs_rows, lab, training_prefix=[])

    osc = apply_structural_combinator(make_atomic_operator_spec("exp_relax_2d"), "oscillatory_damping_modulation")
    osc_shunt = apply_structural_combinator(osc, "additive_secondary_shunt", secondary_role=1)
    compiled_4p = compile_operator_spec_to_python(
        osc_shunt, role_names=["u", "t"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0}
    )
    ast_4p = canonicalize_python_model_ast(compiled_4p)

    forest, _ = build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    targets, _ = build_m4_cluster_targets(cluster_idx=0)

    bad_prog_f = deepcopy(forest)
    for nd in bad_prog_f.nodes.values():
        nd["program_steps"] = [{"step": "NOT_AN_IMPLEMENTED_OPCODE"}]
    r_prog_intact = evaluate_m4_target_under_arm(
        targets[0],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=forest,
        index=HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=work_dir / "prog_intact",
    )
    r_prog_bad = evaluate_m4_target_under_arm(
        targets[0],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=bad_prog_f,
        index=HippoRAGWormholeIndex(bad_prog_f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=work_dir / "prog_bad",
    )

    cap_tier_f = deepcopy(forest)
    cap_tier_f.nodes["bio_vascular_macro_capped"]["claim_contract"]["permission_tier"] = "bounded_trial"
    r_child_cap = evaluate_m4_target_under_arm(
        targets[10],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=cap_tier_f,
        index=HippoRAGWormholeIndex(cap_tier_f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=work_dir / "child_cap",
    )

    unk_scope_f = deepcopy(forest)
    unk_scope_f.nodes["bio_vascular_macro_capped"]["claim_contract"]["preconditions"].append(
        {"field": "audit_boundary_measured", "op": "==", "value": True}
    )
    r_child_unk = evaluate_m4_target_under_arm(
        targets[10],
        arm_name="Hegel_M4_MultiScale_HippoWalk",
        forest=unk_scope_f,
        index=HippoRAGWormholeIndex(unk_scope_f, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
        stage1_gate=None,
        work_dir=work_dir / "child_unk",
    )

    wrong_f = deepcopy(forest)
    wrong_merge = wrong_f.compress_compatible_nodes(
        keeper_node_id="aq_macro_river_tree",
        redundant_node_id="dist_node_00",
        evaluation_queries=[{"active_arity": 1}],
        evaluator_fn=lambda nd, q: "ACC" if int(nd["observation_map"]["parameter_semantics"]["active_arity"]) == q["active_arity"] else "UNK",
    )

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
    disj_merge = disj_f.compress_compatible_nodes(
        keeper_node_id="aq_macro_river_tree",
        redundant_node_id="dist_node_00",
        evaluation_queries=[{"case": "A"}, {"case": "B"}],
        evaluator_fn=branch_ev,
    )

    rev_f = deepcopy(forest)
    cc_p = deepcopy(rev_f.nodes["aq_meso_hex_cell"]["claim_contract"])
    cc_p["protocol_spec"]["rev_note"] = 2
    rev_f.empirical_revise_node("aq_meso_hex_cell", witness_episode_id="ep_p", cause="protocol_or_scope_mismatch", updated_claim_contract=cc_p)
    cg_after_protocol = HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"]

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
    cg_after_scale_change = HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"]
    rev_f.revalidate_vertical_edges_after_revision("aq_meso_hex_cell", witness_episode_id="ep_reval", reason="reverified_pi")
    cg_after_reval = HippoRAGWormholeIndex(rev_f, damping_alpha=0.65).edge_counts["coarse_graining_edges"]

    return {
        "duplicate_predict_rejected": bool(
            dup_a["skeleton_status"] == "UNSUPPORTED_IMPURE_PYTHON"
            and dup_a["ast_skeleton_digest"] != dup_b["ast_skeleton_digest"]
        ),
        "module_attribute_shadowing_rejected": bool(
            shad_a["skeleton_status"] == "UNSUPPORTED_IMPURE_PYTHON"
            and shad_a["ast_skeleton_digest"] != shad_b["ast_skeleton_digest"]
        ),
        "no_receipt_no_target_denied": (not auth_no_rec["granted"]),
        "absolute_path_confirmation_parity": bool(abs(r_rel["rmse"] - r_abs["rmse"]) < 1e-12 and r_abs["rmse"] == 9.0),
        "four_param_oscillatory_shunt_verified": bool(
            ast_4p["skeleton_status"] == "VERIFIED_PURE_AST" and "'p3'" in compiled_4p
        ),
        "nonsense_program_steps_rejected": bool(
            r_prog_intact["confirmed_via_recall"] is True and r_prog_bad["confirmed_via_recall"] is False
        ),
        "child_tier3_cap_denied": (r_child_cap["confirmed_via_recall"] is False),
        "child_unknown_precondition_denied": (r_child_unk["confirmed_via_recall"] is False),
        "fidelity_wrong_law_overlap_rejected": (wrong_merge["accepted"] is False),
        "fidelity_committed_keeper_matches_validated": bool(
            disj_merge["accepted"] is True and branch_ev(disj_f.nodes["aq_macro_river_tree"], {"case": "B"}) == "ACC"
        ),
        "coarse_graining_edge_counts": {
            "after_protocol_only_revision": cg_after_protocol,
            "after_scale_map_revision_before_reval": cg_after_scale_change,
            "after_explicit_vertical_revalidation": cg_after_reval,
        },
    }


def run_part2_complete_m3_1_2x2_factorization(work_dir: Path) -> dict[str, Any]:
    """Run all 4 cells of {Walk (alpha=0.65), NoWalk (alpha=0.0)} x {Pool, NoPool} across 4 M3.1 clusters (`448` runs)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    orig_p, orig_c, orig_e = m3_mod.build_m3_pilot_points, m3_mod.build_m3_confirmation_point, m3_mod.extract_m3_observable_signatures
    m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
    m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
    m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures

    cells = [
        ("Walk_Pool", "hipporag_ppr"),
        ("Alpha0_Pool", "hipporag_alpha0_pool"),
        ("Walk_NoPool", "hipporag_walk_nopool"),
        ("Alpha0_NoPool", "hipporag_alpha0_nopool"),
    ]
    agg: dict[str, list[dict[str, Any]]] = {label: [] for label, _ in cells}

    try:
        for cid, seed in M3_1_EVAL_SEEDS:
            suite = make_m3_1_factorized_walk_suite(seed, target_k=36)
            base = M3_1FactorizedResearcher(
                gate=TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=16),
                retrieval_mode="hipporag_ppr",
                damping_alpha=0.65,
            )
            base.form_and_revise_episodes(cid, suite, work_dir / cid / "src")
            arms = {label: base.clone_with_mode(mode) for label, mode in cells}

            all_targets = [*suite["targets"], *suite["walk_targets"]]
            for cat, tname, (tfac, tev) in all_targets:
                for label, _ in cells:
                    lb = tfac()
                    ws = work_dir / cid / label / tname
                    sub = arms[label].solve_target(tname, lb, ws)
                    ev = tev(ws, sub, lb.observations)
                    snap = lb.meter.snapshot()
                    agg[label].append({
                        "cluster_id": cid,
                        "category": cat,
                        "target": tname,
                        "strict_pass": bool(ev["strict_pass"]),
                        "walk_win": bool(sub.get("selected_reached_only_by_walk", False)),
                        "measurement_cost": snap["measurement_cost"],
                        "compute_cost": snap["compute_cost"],
                        "composite_cost": snap["composite_cost"],
                    })
    finally:
        m3_mod.build_m3_pilot_points = orig_p
        m3_mod.build_m3_confirmation_point = orig_c
        m3_mod.extract_m3_observable_signatures = orig_e

    summary_cells: dict[str, Any] = {}
    for label, _ in cells:
        rows = agg[label]
        t01_06 = [r for r in rows if r["target"].startswith(("T01_", "T02_", "T03_", "T04_", "T05_", "T06_"))]
        t25_28 = [r for r in rows if r["target"].startswith(("T25_", "T26_", "T27_", "T28_"))]
        summary_cells[label] = {
            "total_runs": len(rows),
            "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in rows), 4),
            "overall_mean_measurement_cost": round(mean(r["measurement_cost"] for r in rows), 4),
            "overall_mean_compute_cost": round(mean(r["compute_cost"] for r in rows), 4),
            "overall_mean_composite_cost": round(mean(r["composite_cost"] for r in rows), 4),
            "T01_T06_cross_domain_mean_composite_cost": round(mean(r["composite_cost"] for r in t01_06), 4),
            "T25_T28_safe_pilot_walk_wins_of_16": sum(1 for r in t25_28 if r["walk_win"]),
            "T25_T28_safe_pilot_mean_composite_cost": round(mean(r["composite_cost"] for r in t25_28), 4),
        }
    return {"cells_2x2": summary_cells}


def main() -> None:
    t0 = time.perf_counter()
    EVID_DIR.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="hegel_v17_3_study_") as tmp_str:
        tmp_root = Path(tmp_str)

        print("[1/4] Verifying M3.1 + M4 Audit Closures...")
        part1 = run_part1_audit_closures(tmp_root / "p1_closures")
        (EVID_DIR / "part1_audit_boundary_and_lifecycle_closures.json").write_text(
            json.dumps(part1, indent=2, sort_keys=True), encoding="utf-8"
        )

        print("[2/4] Running Complete 2x2 M3.1 Retrieval Factorization (448 runs)...")
        part2 = run_part2_complete_m3_1_2x2_factorization(tmp_root / "p2_2x2")
        (EVID_DIR / "part2_m3_1_complete_2x2_retrieval_factorization.json").write_text(
            json.dumps(part2, indent=2, sort_keys=True), encoding="utf-8"
        )

        print("[3/4] Running Executable Dynamical Pi_s Trajectory Renormalization Suite...")
        part3 = run_executable_pi_s_renormalization_suite(tmp_root / "p3_pi_s")
        (EVID_DIR / "part3_executable_pi_s_fast_slow_renormalization.json").write_text(
            json.dumps(part3, indent=2, sort_keys=True), encoding="utf-8"
        )

        print("[4/4] Running Multi-Seed Unified Continual Discovery-to-Forest Chain...")
        primary_chain = run_m4_1_unified_continual_discovery_chain(tmp_root / "p4_seed_0")
        # Copy primary Stage-1 persisted authored workspaces into evidence bundle for independent R1/R2/R3 inspection!
        art_out = EVID_DIR / "persisted_stage1_authored_workspaces"
        if art_out.exists():
            shutil.rmtree(art_out)
        art_out.mkdir(parents=True, exist_ok=True)
        for ws_name in (
            "stage1_S1_monod_parent",
            "stage1_W1_haldane_shunt",
            "stage1_W2_osc_shunt_4p",
            "stage1_W3_murray_capped",
        ):
            shutil.copytree(tmp_root / "p4_seed_0" / ws_name, art_out / ws_name)

        # Also evaluate 3 additional unseen target cluster seeds (`4 x 14 x 6 = 336` target runs total!)
        multi_seed_runs: dict[str, list[dict[str, Any]]] = {
            arm: list(recs) for arm, recs in primary_chain["per_target_records"].items()
        }
        for s_idx, extra_seed in enumerate(UNIFIED_EVAL_SEEDS[1:], start=1):
            extra_targets, extra_oos = build_unified_12_target_suite(seed=extra_seed)
            f_u = tmp_root / f"p4_seed_{s_idx}_custom"
            from hegel_machine.cognition.m4_1_unified_discovery_and_renormalization import (
                FlatEqualMetadataIndex,
                _register_unified_forest_node,
                build_stage0_atomic_forest,
            )
            w1_v = load_verified_authored_method(art_out / "stage1_W1_haldane_shunt" / "authored_method.json")
            w2_v = load_verified_authored_method(art_out / "stage1_W2_osc_shunt_4p" / "authored_method.json")
            w3_v = load_verified_authored_method(art_out / "stage1_W3_murray_capped" / "authored_method.json")
            assert w1_v and w2_v and w3_v
            f_s0, s0_specs = build_stage0_atomic_forest()
            f_unif = deepcopy(f_s0)
            _register_unified_forest_node(
                f_unif, node_id="s1_child_haldane_shunt", tree_id="tree_biochem",
                surface_domain="biochem_high_substrate_witness", scale_regime="meso",
                operator_tag="op_composed_haldane_shunt_2d", operator_spec=w1_v["operator_spec"],
                role_names=["s", "b"], probe_signatures=["sig:witness_haldane_shunt_2d"],
                preconditions=[{"field": "haldane_turnover_shunt_regime", "op": "==", "value": True}],
                authored_artifact_path=str(art_out / "stage1_W1_haldane_shunt" / "authored_method.json"),
            )
            f_unif.add_local_implication_edge(
                stronger_node_id="s1_child_haldane_shunt", weaker_node_id="s0_biochem_monod",
                preserved_contract={"low_substrate_limit": "monod"}, scope={"s_le": 2.4},
            )
            f_unif.refine_wormhole_with_near_miss(
                "wh_s0_monod", near_miss_episode_id="w1",
                exclusion_predicate={"field": "haldane_turnover_shunt_regime", "op": "==", "value": True},
            )
            for pid in ("s0_biochem_monod", "s0_reactor_monod"):
                f_unif.nodes[pid]["claim_contract"]["preconditions"].append(
                    {"field": "haldane_turnover_shunt_regime", "op": "!=", "value": True}
                )

            _register_unified_forest_node(
                f_unif, node_id="s1_child_osc_shunt_4p", tree_id="tree_piezo",
                surface_domain="piezo_ringing_secular_drift", scale_regime="micro",
                operator_tag="op_composed_osc_shunt_4p_2d", operator_spec=w2_v["operator_spec"],
                role_names=["u", "t"], probe_signatures=["sig:witness_oscillatory_shunt_4p"],
                preconditions=[{"field": "oscillatory_secular_shunt_regime", "op": "==", "value": True}],
                authored_artifact_path=str(art_out / "stage1_W2_osc_shunt_4p" / "authored_method.json"),
            )
            f_unif.add_local_implication_edge(
                stronger_node_id="s1_child_osc_shunt_4p", weaker_node_id="s0_piezo_relax",
                preserved_contract={"overdamped_limit": "exp_relax"}, scope={"omega_eq": 0.0},
            )
            f_unif.refine_wormhole_with_near_miss(
                "wh_s0_relax", near_miss_episode_id="w2",
                exclusion_predicate={"field": "oscillatory_secular_shunt_regime", "op": "==", "value": True},
            )
            for pid in ("s0_piezo_relax", "s0_servo_relax"):
                f_unif.nodes[pid]["claim_contract"]["preconditions"].append(
                    {"field": "oscillatory_secular_shunt_regime", "op": "!=", "value": True}
                )

            _register_unified_forest_node(
                f_unif, node_id="s1_child_murray_capped", tree_id="tree_vascular",
                surface_domain="vascular_saturated_ceiling", scale_regime="macro",
                operator_tag="op_composed_murray_capped_1d", operator_spec=w3_v["operator_spec"],
                role_names=["flow"], probe_signatures=["sig:witness_macro_flow_ceiling"],
                preconditions=[{"field": "high_flow_ceiling_regime", "op": "==", "value": True}],
                authored_artifact_path=str(art_out / "stage1_W3_murray_capped" / "authored_method.json"),
            )
            f_unif.add_local_implication_edge(
                stronger_node_id="s1_child_murray_capped", weaker_node_id="s0_bio_vascular_murray",
                preserved_contract={"low_flow_limit": "murray_2_3"}, scope={"flow_le": 3.2},
            )
            f_unif.refine_wormhole_with_near_miss(
                "wh_s0_murray", near_miss_episode_id="w3",
                exclusion_predicate={"field": "high_flow_ceiling_regime", "op": "==", "value": True},
            )
            for pid in ("s0_bio_vascular_murray", "s0_sys_cdn_murray"):
                f_unif.nodes[pid]["claim_contract"]["preconditions"].append(
                    {"field": "high_flow_ceiling_regime", "op": "!=", "value": True}
                )

            idx_h = HippoRAGWormholeIndex(f_unif, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
            idx_a0 = HippoRAGWormholeIndex(f_unif, damping_alpha=0.0, wormhole_equivalence_pooling=True, scale_aware=True)
            idx_fl = FlatEqualMetadataIndex(f_unif, mode="bm25", wormhole_equivalence_pooling=True)
            idx_frz = HippoRAGWormholeIndex(f_s0, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)

            ord_fwd = [
                *s0_specs,
                ("op_inter_ring_3p", apply_structural_combinator(make_atomic_operator_spec("exp_relax_2d"), "oscillatory_damping_modulation")),
                ("op_inter_monod_shunt", apply_structural_combinator(make_atomic_operator_spec("michaelis_1d", arity=2), "additive_secondary_shunt", secondary_role=1)),
                ("op_composed_haldane_shunt_2d", deepcopy(w1_v["operator_spec"])),
                ("op_composed_osc_shunt_4p_2d", deepcopy(w2_v["operator_spec"])),
                ("op_composed_murray_capped_1d", deepcopy(w3_v["operator_spec"])),
            ]
            ord_rev = list(reversed(ord_fwd))

            for arm_label, fo, io, ospecs in [
                ("Hegel_M4_1_UnifiedForest", f_unif, idx_h, None),
                ("Forest_Alpha0_SameHistory", f_unif, idx_a0, None),
                ("Flat_BM25_SameHistory", f_unif, idx_fl, None),
                ("Ordinary_AllHistory_Forward", None, None, ord_fwd),
                ("Ordinary_AllHistory_Reversed", None, None, ord_rev),
                ("Frozen_After_Stage0", f_s0, idx_frz, None),
            ]:
                for t_obj in [*extra_targets, *extra_oos]:
                    rec = evaluate_unified_target_under_arm(
                        t_obj,
                        arm_name=arm_label,
                        forest=fo,
                        index=io,
                        ordinary_history_specs=ospecs,
                        fallback_universe_specs=ord_fwd,
                        ws_dir=f_u / arm_label / f"s{extra_seed}_{t_obj.target_id}",
                    )
                    rec["seed"] = extra_seed
                    multi_seed_runs[arm_label].append(rec)

        # Write full 336-run JSONL receipt log
        jsonl_lines: list[str] = []
        multi_seed_summary: dict[str, Any] = {}
        for arm_label, rows in multi_seed_runs.items():
            for r in rows:
                jsonl_lines.append(json.dumps(r, sort_keys=True))
            in_scope = [r for r in rows if r["mechanism_family"] != "OutOfScope"]
            oos = [r for r in rows if r["mechanism_family"] == "OutOfScope"]
            multi_seed_summary[arm_label] = {
                "in_scope_target_runs": len(in_scope),
                "oos_runs": len(oos),
                "strict_pass_rate_48": round(mean(1.0 if r["strict_pass"] else 0.0 for r in in_scope), 4),
                "recall_rate_48": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in in_scope), 4),
                "walk_wins_of_16_safe_pilot": sum(1 for r in in_scope if r["non_direct_walk_win"]),
                "false_confirmations_56": sum(1 for r in rows if r["false_confirmation"]),
                "oos_scoped_unknown_rate_8": round(mean(1.0 if r["status"] == "scoped_unknown" else 0.0 for r in oos), 4),
                "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in in_scope), 4),
                "mean_measurement_cost": round(mean(r["cost"]["measurement_cost"] for r in in_scope), 4),
                "mean_compute_cost": round(mean(r["cost"]["compute_cost"] for r in in_scope), 4),
                "mean_composite_cost": round(mean(r["cost"]["composite_cost"] for r in in_scope), 4),
            }

        (EVID_DIR / "per_target_receipts_336_runs.jsonl").write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")

        part4_out = {
            k: v for k, v in primary_chain.items() if k != "per_target_records"
        }
        part4_out["four_seed_48_target_arm_summaries"] = multi_seed_summary
        (EVID_DIR / "part4_unified_continual_discovery_forest_chain.json").write_text(
            json.dumps(part4_out, indent=2, sort_keys=True), encoding="utf-8"
        )

    elapsed = round(time.perf_counter() - t0, 3)
    master_summary = {
        "study": "Hegel V17.3 (M4.1) — Audit Closure, Executable Pi_s Renormalization & Unified Continual Chain",
        "date": "2026-11-20",
        "elapsed_wall_seconds": elapsed,
        "part1_audit_closures": part1,
        "part2_m3_1_complete_2x2_factorization": part2["cells_2x2"],
        "part3_executable_pi_s_highlights": {
            "coarse_edge_causally_active": part3["adiabatic_and_coarse_edge_causality"]["coarse_edge_causally_active"],
            "with_cg_composite_cost": part3["adiabatic_and_coarse_edge_causality"]["With_CoarseGraining_Edges"]["composite_cost"],
            "without_cg_composite_cost": part3["adiabatic_and_coarse_edge_causality"]["Without_CoarseGraining_Edges"]["composite_cost"],
            "micro_only_composite_cost": part3["adiabatic_and_coarse_edge_causality"]["MicroOnly_AlwaysFine"]["composite_cost"],
            "regional_box_catches_interior_bump": part3["interior_micro_bump_regional_vs_single_point"]["regional_box_catches_bump_zero_false_confirm"],
            "out_of_box_query_rejected": part3["trajectory_projection_verification"]["out_of_box_query_rejected"],
        },
        "part4_unified_chain_4_seed_summary": part4_out["four_seed_48_target_arm_summaries"],
        "part4_r3_three_tier_old_scope_preservation": part4_out["r3_three_tier_old_scope_preservation"],
        "part4_r1_forest_artifact_integrity": {
            "metadata_only_rejected_properly": part4_out["r1_forest_artifact_integrity_controls"]["metadata_only_rejected_properly"],
            "deletion_increases_cost_and_restoration_recovers": part4_out["r1_forest_artifact_integrity_controls"]["deletion_increases_cost_and_restoration_recovers"],
        },
        "part4_key_invariants": part4_out["key_invariants"],
        "unified_full_lifecycle_ledger": part4_out["unified_full_lifecycle_ledger"],
    }
    (EVID_DIR / "summary.json").write_text(json.dumps(master_summary, indent=2, sort_keys=True), encoding="utf-8")

    sha_lines: list[str] = []
    for fpath in sorted(EVID_DIR.rglob("*")):
        if fpath.is_file() and fpath.name != "SHA256SUMS.txt":
            rel = fpath.relative_to(EVID_DIR).as_posix()
            sha_lines.append(f"{_sha256_file(fpath)}  {rel}")
    (EVID_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"Completed V17.3 (M4.1) study in {elapsed:.2f}s -> {EVID_DIR}")


if __name__ == "__main__":
    main()
