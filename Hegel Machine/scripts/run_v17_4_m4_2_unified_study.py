"""Execute Preregistered Hegel V17.4 (M4.2) Audit Closure, Deduplicated M3.1 2x2 Factorization,
Executable Trajectory-Projected Pi_s, Zero-Fit Scope Dispatcher, and Cross-Pillar Continual Study.

Writes full static evidence bundle by default to:
  `docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120/`
and supports `--out-dir <path>` for clean read-only external audit replays without touching
any committed `docs/evidence/*` or `markdown/*Audit*/` directories!
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m4_1_unified_discovery_and_renormalization import (
    _load_node_verified_operator_spec,
    run_executable_pi_s_renormalization_suite,
)
from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import (
    certify_regional_pi_s_closure_with_trajectory_projection,
    run_deduplicated_m3_1_2x2_factorization,
    run_m4_2_unified_cross_pillar_study,
)
from hegel_machine.cognition.m4_compositional_discovery import (
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    load_verified_authored_method,
    make_atomic_operator_spec,
)
import hegel_machine.cognition.m4_multiscale_renormalization as m4_ms
from hegel_machine.cognition.open_workspace_wormhole import canonicalize_python_model_ast
from hegel_machine.cognition.wormhole_forest import ScaleMap, authorize_abstraction_use


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVID_DIR = REPO_ROOT / "docs" / "evidence" / "v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120"

M3_1_SEEDS = (8_028_101, 8_028_202, 8_028_303, 8_028_404)
M4_2_SEEDS = (20_261_120, 20_271_120, 20_281_120, 20_291_120)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_part1_m4_1_audit_closures(work_dir: Path) -> dict[str, Any]:
    """Replay and verify all closure witnesses from `Hegel_M4_1_Audit`."""
    work_dir.mkdir(parents=True, exist_ok=True)
    forest, _ = m4_ms.build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    targets, _ = m4_ms.build_m4_cluster_targets(cluster_idx=0)

    # 1. `program_steps` `fresh_points` consumption (`1` vs `2`)
    def _run_fp(fp: int) -> dict[str, Any]:
        fc = deepcopy(forest)
        for nd in fc.nodes.values():
            for st in nd.get("program_steps", []):
                if st.get("step") == "subprocess_isolated_confirm":
                    st["fresh_points"] = fp
        acqs: list[dict[str, Any]] = []
        orig = m4_ms.MultiScaleWorkspaceTaskLab.acquire_channel

        def _rec(self, points, *, channel="coarse_macro", phase="pilot"):
            rows = orig(self, points, channel=channel, phase=phase)
            acqs.append({"phase": phase, "channel": channel, "count": len(points)})
            return rows

        with patch.object(m4_ms.MultiScaleWorkspaceTaskLab, "acquire_channel", _rec):
            r = m4_ms.evaluate_m4_target_under_arm(
                targets[0],
                arm_name="Hegel_M4_MultiScale_HippoWalk",
                forest=fc,
                index=HippoRAGWormholeIndex(fc, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                stage1_gate=None,
                work_dir=work_dir / f"fp_{fp}",
            )
        r["cost"].pop("wall_clock_seconds", None)
        return {
            "declared_fresh_points": fp,
            "actual_confirm_count": sum(x["count"] for x in acqs if x["phase"].startswith("confirm")),
            "measurement_cost": r["cost"]["measurement_cost"],
            "composite_cost": r["cost"]["composite_cost"],
        }

    fp1 = _run_fp(1)
    fp2 = _run_fp(2)

    # 2. Disjoint-scope `FidelityCompression` OR-branch consumption in `authorize_abstraction_use`
    fc_or = deepcopy(forest)
    a_id, b_id = "aq_macro_river_tree", "dist_node_00"
    for nid, sc in ((a_id, "A"), (b_id, "B")):
        nd = fc_or.nodes[nid]
        nd["observation_map"] = deepcopy(fc_or.nodes[a_id]["observation_map"])
        nd["scale_map"] = deepcopy(fc_or.nodes[a_id]["scale_map"])
        nd["claim_contract"]["conclusion_or_action"] = deepcopy(fc_or.nodes[a_id]["claim_contract"]["conclusion_or_action"])
        nd["claim_contract"]["preconditions"] = [{"field": "audit_scope", "op": "==", "value": sc}]

    comp_rep = fc_or.compress_compatible_nodes(
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
    merged_nd = fc_or.nodes[a_id]
    or_auth = {
        sc: authorize_abstraction_use(
            requested_tier="bounded_trial",
            task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.1},
            target_features={"audit_scope": sc},
            claim_contract=merged_nd["claim_contract"],
            scale_map=merged_nd["scale_map"],
        )["granted"]
        for sc in ("A", "B", "C")
    }

    # 3. `load_verified_authored_method` confirmation provenance & cross-node swap block
    sp_mm = make_atomic_operator_spec("michaelis_1d")
    code_mm = compile_operator_spec_to_python(sp_mm, role_names=["x"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0})
    ast_mm = canonicalize_python_model_ast(code_mm)
    no_conf_file = work_dir / "self_consistent_no_confirm.json"
    no_conf_file.write_text(
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
    unconfirmed_accepted = load_verified_authored_method(no_conf_file) is not None

    def _mk_confirmed(sp_obj: dict, fn) -> dict:
        c_src = compile_operator_spec_to_python(sp_obj, role_names=["x"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0})
        a_inf = canonicalize_python_model_ast(c_src)
        p_r = [{"features": {"x": x}, "target": float(fn(x))} for x in (0.5, 1.5, 3.0, 5.0)]
        c_r = [{"features": {"x": 2.5}, "target": float(fn(2.5))}]
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
                "task_id": "swap_check",
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

    art_w1 = _mk_confirmed(sp_mm, lambda x: 6.0 * x / (2.0 + x))
    art_w2 = _mk_confirmed(make_atomic_operator_spec("linear_1d"), lambda x: 2.0 * x)
    slot_path = work_dir / "swap_slot.json"
    slot_path.write_text(json.dumps(art_w1), encoding="utf-8")
    nd_slot = {
        "claim_contract": {
            "conclusion_or_action": {
                "operator_tag": "op_mm",
                "operator_spec": deepcopy(sp_mm),
                "ast_skeleton_digest": art_w1["ast_skeleton_digest"],
                "authored_artifact_path": str(slot_path),
            }
        }
    }
    before_swap_ok = _load_node_verified_operator_spec(nd_slot) == sp_mm
    slot_path.write_text(json.dumps(art_w2), encoding="utf-8")
    after_swap_accepted = _load_node_verified_operator_spec(nd_slot) is not None

    # 4. Relative parent-path escape blocked
    sub_ws = work_dir / "nest_a" / "nest_b"
    sub_ws.mkdir(parents=True, exist_ok=True)
    (work_dir / "outside.py").write_text("def predict(row):\n    return 1.0\n", encoding="utf-8")
    rel_escape_blocked = False
    try:
        evaluate_isolated_python_model(sub_ws, "../../outside.py", [], isolated_prefix_rows=[])
    except ValueError:
        rel_escape_blocked = True

    # 5. Certified vertical revalidation gate
    vf = deepcopy(forest)
    sm_nc = ScaleMap(
        coarse_graining_op="pi_nc",
        window_or_partition={"scale_regime": "meso"},
        preserved_queries=["open_workspace_mechanism_prediction"],
        closure_scope="not_closed",
        proposed_discarded_variables=["fast_micro_mode"],
        verified_discarded_in_scope=[],
        error_budget=0.25,
    ).to_dict()
    vf.empirical_revise_node("aq_meso_hex_cell", witness_episode_id="ep_nc", cause="abstraction_closure_loss", updated_scale_map=sm_nc)
    rej_unclosed = vf.revalidate_vertical_edges_with_certificate(
        "aq_meso_hex_cell",
        witness_episode_id="ep_rej",
        closure_certificate={"certified": True, "max_closure_defect": 0.01, "tolerance": 0.05},
    )
    sm_closed = deepcopy(sm_nc)
    sm_closed["closure_scope"] = "regional_box_verified"
    sm_closed["verified_discarded_in_scope"] = ["fast_micro_mode"]
    acc_closed = vf.revalidate_vertical_edges_with_certificate(
        "aq_meso_hex_cell",
        witness_episode_id="ep_acc",
        closure_certificate={"certified": True, "max_closure_defect": 0.01, "tolerance": 0.05},
        updated_scale_map=sm_closed,
    )

    return {
        "program_steps_fresh_points": {
            "fresh_1": fp1,
            "fresh_2": fp2,
            "behavior_differs_as_declared": fp1["actual_confirm_count"] == 1 and fp2["actual_confirm_count"] == 2,
        },
        "fidelity_or_branch_real_authorization": {
            "compression_accepted": comp_rep["accepted"],
            "installed_top_level_preconditions": merged_nd["claim_contract"]["preconditions"],
            "disjunctive_branch_count": len(merged_nd["claim_contract"]["disjunctive_branches"]),
            "authorize_abstraction_use_grants": or_auth,
        },
        "authored_artifact_provenance_and_node_binding": {
            "unconfirmed_self_consistent_accepted": unconfirmed_accepted,
            "registered_w1_accepted_before_swap": before_swap_ok,
            "swapped_w2_accepted_for_w1_node": after_swap_accepted,
        },
        "relative_parent_path_escape_blocked": rel_escape_blocked,
        "certified_vertical_revalidation": {
            "unclosed_rejected": rej_unclosed["accepted"] is False,
            "certified_closed_accepted": acc_closed["accepted"] is True,
            "restored_coarse_graining_edges": HippoRAGWormholeIndex(vf, damping_alpha=0.65).edge_counts["coarse_graining_edges"],
        },
    }


def run_part3_pi_s_suite_and_counterfactuals(work_dir: Path) -> dict[str, Any]:
    """Run executable `Pi_s` suite + out-of-box fallback/export counterfactuals + trajectory projection check."""
    work_dir.mkdir(parents=True, exist_ok=True)
    suite_rep = run_executable_pi_s_renormalization_suite(work_dir / "pi_suite")
    traj_cert = certify_regional_pi_s_closure_with_trajectory_projection(work_dir / "pi_traj_cert")
    cf = suite_rep["out_of_box_confirm_and_heldout_counterfactuals"]
    oob_confirm = cf["outside_confirm_fallback"]
    oob_heldout = cf["outside_heldout_query"]
    return {
        "pi_s_suite": suite_rep,
        "trajectory_projection_certificate": traj_cert,
        "counterfactual_guards": {
            "outside_box_confirm_escalates_to_micro": {
                "selected_channel": oob_confirm["selected_channel"],
                "strict_pass": oob_confirm["strict_pass"],
                "false_confirmation": oob_confirm["false_confirmation"],
            },
            "outside_box_exported_query_blocked": {
                "status": oob_heldout["status"],
                "out_of_box_query_blocked": oob_heldout["out_of_box_query_blocked"],
                "false_confirmation": oob_heldout["false_confirmation"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel V17.4 (M4.2) Unified Closure & Pi_s Lifecycle Study")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_EVID_DIR,
        help="Output directory for evidence bundle (use /tmp/... for read-only audit replay)",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with TemporaryDirectory(prefix="hegel-v17-4-run-") as td:
        tmp_work = Path(td)
        part1 = run_part1_m4_1_audit_closures(tmp_work / "part1")
        part2 = run_deduplicated_m3_1_2x2_factorization(tmp_work / "part2", seeds=M3_1_SEEDS)
        part3 = run_part3_pi_s_suite_and_counterfactuals(tmp_work / "part3")
        part4 = run_m4_2_unified_cross_pillar_study(tmp_work / "part4", seeds=M4_2_SEEDS)

        # Copy persisted Stage-0 & Stage-1 workspaces into `out_dir/persisted_authored_workspaces/`
        persisted_dst = out_dir / "persisted_authored_workspaces"
        if persisted_dst.exists():
            shutil.rmtree(persisted_dst)
        persisted_dst.mkdir(parents=True, exist_ok=True)
        for sub_src in (
            tmp_work / "part4" / "continual_lifecycle" / "stage0_metered_sources",
            tmp_work / "part4" / "continual_lifecycle" / "persisted_stage1_authored_workspaces",
        ):
            if sub_src.exists():
                shutil.copytree(sub_src, persisted_dst / sub_src.name)

    m31_records = part2.pop("per_run_records")
    m42_receipts = part4.pop("execution_receipts")

    (out_dir / "part1_m4_1_audit_closures.json").write_text(
        json.dumps(part1, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_deduplicated_m3_1_2x2_factorization.json").write_text(
        json.dumps(part2, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m3_1_unique_432_runs.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in m31_records) + "\n", encoding="utf-8"
    )
    (out_dir / "part3_executable_pi_s_and_guards.json").write_text(
        json.dumps(part3, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part4_unified_cross_pillar_lifecycle_and_breakeven.json").write_text(
        json.dumps(part4, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part4_unified_execution_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in m42_receipts) + "\n", encoding="utf-8"
    )

    elapsed = round(time.perf_counter() - t0, 2)
    manifest = {
        "version": "V17.4_M4.2",
        "elapsed_wall_seconds": elapsed,
        "m3_1_unique_runs_per_cell": part2["unique_runs_per_cell"],
        "m3_1_total_unique_runs": part2["total_unique_runs"],
        "m4_2_seeds": list(M4_2_SEEDS),
        "dispatcher_zero_fit_all_exact": part4["dispatcher_zero_fit_r3"]["all_routes_exact"],
        "dispatcher_any_fit_called": part4["dispatcher_zero_fit_r3"]["any_fit_called"],
        "hegel_stage3_mean_composite_cost": part4["stage3_arms"]["Hegel_Unified_Continual_Forest"]["mean_composite_cost"],
        "dev_frozen_static_mean_composite_cost": part4["stage3_arms"]["Ordinary_AllHistory_DevFrozenOrder"]["mean_composite_cost"],
        "observed_breakeven_vs_frozen_open_rediscovery": part4["lifecycle_breakeven"]["observed_breakeven_target_vs_frozen_open_rediscovery"],
        "observed_breakeven_vs_frozen_pool_fallback": part4["lifecycle_breakeven"]["observed_breakeven_target_vs_frozen_pool_fallback"],
    }
    (out_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Write SHA256SUMS.txt covering every generated file in `out_dir`
    sha_lines: list[str] = []
    for fpath in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        rel = fpath.relative_to(out_dir).as_posix()
        sha_lines.append(f"{_sha256_file(fpath)}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
