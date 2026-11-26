"""Generate Sealed M8.9.1 Deterministic Audit-Closure, HS01 Regrade, and True LOTO Router Receipts (`R1`, `R6`, `R7`).

Strictly non-destructive: NEVER modifies `docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/`
or `docs/44_*`. Writes deterministic receipts to `docs/evidence/m8_9_1_audit_closure_and_regrade_20261126/`.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hegel_machine.cognition import m6_4_executable_wormhole_library as lib
from hegel_machine.cognition import m6_frontier_real_benchmarks as front
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import build_holdout_ood_12_tasks
from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import build_unseen_postfreeze_4_tasks
from hegel_machine.cognition.m8_9_1_box_l0_certificate_kernel import (
    ExactAllSupportBoxQPReference,
    eval_exact_box_l0_objective,
    verify_box_l0_subcube_certificate,
)
from hegel_machine.cognition.m8_9_hard_stress_benchmark import HARD_STRESS_M89_TASKS
from hegel_machine.cognition import m8_offline_container_planner as rtr


M89_EVIDENCE_DIR = PROJECT_ROOT / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126"
OUT_DIR = PROJECT_ROOT / "docs/evidence/m8_9_1_audit_closure_and_regrade_20261126"

HS01_PRIV_SPECS = [
    (99101, 46, 14, 0.06, 0.30, 1.2),
    (99102, 50, 14, 0.03, 0.26, 1.15),
    (99103, 52, 15, 0.07, 0.34, 1.25),
]


def _make_hs01_private_case(seed: int, n: int, p: int, ridge: float, l0_pen: float, box: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z_lat = rng.normal(size=(n, 4))
    X = np.zeros((n, p), dtype=float)
    for j in range(p):
        X[:, j] = 0.74 * z_lat[:, j % 4] + 0.67 * rng.normal(size=n)
    X[:, 4] = 0.53 * X[:, 0] + 0.51 * X[:, 2] + 0.13 * rng.normal(size=n)
    X /= np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-8) / np.sqrt(n)
    w_true = np.zeros(p, dtype=float)
    w_true[0] = 0.80 * box
    w_true[2] = 0.72 * box
    w_true[7] = -0.66 * box
    w_true[11] = 0.58 * box
    y = X @ w_true + rng.normal(0.0, 0.15, size=n)
    return {
        "X": X,
        "y": y,
        "ridge": float(ridge),
        "l0_penalty": float(l0_pen),
        "box": float(box),
        "gap_tol": 1e-6,
        "max_leaves": 512,
    }


def _run_sealed_candidate_subprocess(cand_path: Path, cases_json: str) -> dict[str, Any]:
    runner_code = (
        "import importlib.util, json, sys, time, numpy as np\n"
        "def _def(x):\n"
        "    if isinstance(x, np.ndarray): return x.tolist()\n"
        "    if isinstance(x, np.generic): return x.item()\n"
        "    raise TypeError(type(x).__name__)\n"
        "spec = importlib.util.spec_from_file_location('sealed_cand', sys.argv[1])\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "cases = json.load(sys.stdin)\n"
        "out = []; t0 = time.monotonic()\n"
        "for c in cases:\n"
        "    c['X'] = np.asarray(c['X'], dtype=float); c['y'] = np.asarray(c['y'], dtype=float)\n"
        "    ts = time.monotonic(); r = mod.solve_certified_box_l0(c)\n"
        "    out.append({'result': r, 'elapsed_sec': time.monotonic() - ts})\n"
        "print(json.dumps({'cases': out, 'total_elapsed_sec': time.monotonic() - t0}, default=_def, allow_nan=False))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner_code, str(cand_path)],
        input=cases_json,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(proc.stdout)


def build_hs01_regrade_receipt() -> dict[str, Any]:
    hs01_task = next(t for t in HARD_STRESS_M89_TASKS if t.task_id == "hs01__correlated_box_l0_subcube_certificate")
    new_verifier_sha = hashlib.sha256(hs01_task.private_verifier_py.encode("utf-8")).hexdigest()

    cases = [_make_hs01_private_case(*s) for s in HS01_PRIV_SPECS]
    refs = [ExactAllSupportBoxQPReference(c["X"], c["y"], c["ridge"], c["l0_penalty"], c["box"]) for c in cases]
    for r_obj in refs:
        r_obj.interval()

    cases_json = json.dumps(
        [{**c, "X": c["X"].tolist(), "y": c["y"].tolist()} for c in cases]
    )

    # Discover all sealed hs01 candidate files across primary/thesis/policy snapshots in M8.9
    by_sha: dict[str, list[str]] = {}
    for py_fp in sorted((M89_EVIDENCE_DIR / "ablation_episodes").rglob("certified_box_l0.py")):
        rel = py_fp.relative_to(PROJECT_ROOT).as_posix()
        sha = hashlib.sha256(py_fp.read_bytes()).hexdigest()
        by_sha.setdefault(sha, []).append(rel)

    artifact_rows: list[dict[str, Any]] = []
    for sha, rel_paths in sorted(by_sha.items()):
        cand_fp = PROJECT_ROOT / rel_paths[0]
        exec_out = _run_sealed_candidate_subprocess(cand_fp, cases_json)
        case_verdicts: list[dict[str, Any]] = []
        for (seed_i, _, p_i, _, _, _), case_d, run_c, ref_i in zip(HS01_PRIV_SPECS, cases, exec_out["cases"], refs):
            verd = verify_box_l0_subcube_certificate(case_d, run_c["result"], reference=ref_i)
            case_verdicts.append({"seed": seed_i, "p": p_i, **verd})
        all_pass = bool(
            exec_out["total_elapsed_sec"] < 20.0 and all(cv["strict_pass"] for cv in case_verdicts)
        )
        artifact_rows.append(
            {
                "artifact_sha256": sha,
                "sealed_relative_paths": rel_paths,
                "total_elapsed_sec": round(float(exec_out["total_elapsed_sec"]), 4),
                "regraded_strict_pass": all_pass,
                "cases": case_verdicts,
            }
        )

    # Also record arm-level side-by-side original vs regraded labels (`0` modification to sealed M8.9 files!)
    arm_comparison: dict[str, Any] = {}
    master_m89 = json.loads((M89_EVIDENCE_DIR / "m8_9_ablation_master_summary.json").read_text(encoding="utf-8"))
    sha_to_pass = {r["artifact_sha256"]: r["regraded_strict_pass"] for r in artifact_rows}

    for arm_nm in ("dedup_full_l1_l2", "progressive_all_l1_l2", "progressive_net_value_gated"):
        ep_dir = M89_EVIDENCE_DIR / "ablation_episodes" / arm_nm / "hs01__correlated_box_l0_subcube_certificate"
        pol_fp = ep_dir / "04_submitted_artifacts_snapshot/certified_box_l0.py"
        pri_fps = list(ep_dir.glob("_branches_work/*/branch_2_antithesis_wormhole/04_submitted_artifacts_snapshot/certified_box_l0.py"))
        pol_sha = hashlib.sha256(pol_fp.read_bytes()).hexdigest()
        pri_sha = hashlib.sha256(pri_fps[0].read_bytes()).hexdigest()
        orig_agg = master_m89["arm_aggregates"][arm_nm]["all_8_tasks"]
        arm_comparison[arm_nm] = {
            "original_m89_primary_1x_pass_count": orig_agg["primary_1x_strict_pass_count"],
            "original_m89_policy_pass_count": orig_agg["policy_strict_pass_count"],
            "hs01_primary_1x_artifact_sha256": pri_sha,
            "hs01_policy_artifact_sha256": pol_sha,
            "hs01_regraded_primary_1x_pass": sha_to_pass[pri_sha],
            "hs01_regraded_policy_pass": sha_to_pass[pol_sha],
            "regraded_all_8_primary_1x_pass_count": orig_agg["primary_1x_strict_pass_count"] + (1 if sha_to_pass[pri_sha] else 0),
            "regraded_all_8_policy_pass_count": (
                orig_agg["policy_strict_pass_count"]
                if arm_nm == "dedup_full_l1_l2"
                else orig_agg["policy_strict_pass_count"] + (1 if sha_to_pass[pol_sha] else 0)
            ),
        }

    # Negative control verification receipt
    small_ctrl = {"X": np.eye(3), "y": np.array([0.0, 1.0, 1.0]), "ridge": 0.0, "l0_penalty": 0.1, "box": 2.0, "gap_tol": 1e-6}
    fake_small_ans = {
        "weights": [0.0, 1.0, 0.0],
        "objective": 0.6,
        "certified_lower_bound": 0.59999999,
        "certificate_leaves": [
            {"fixed_zero": [0], "fixed_nonzero": [], "leaf_lower_bound": 0.59999999},
            {"fixed_zero": [], "fixed_nonzero": [0], "leaf_lower_bound": 0.59999999},
        ],
    }
    neg_verd = verify_box_l0_subcube_certificate(small_ctrl, fake_small_ans)

    return {
        "schema": "hegel_m8_9_1_hs01_independent_regrade_receipt_v1",
        "scope": "Finite numerical recheck of 6 unique SHA-pinned M8.9 HS01 submissions across original 3 private instances (p=14,14,15; 65,536 total supports). Original M8.9 labels remain preserved and unmodified.",
        "m891_hs01_private_verifier_sha256": new_verifier_sha,
        "unique_artifact_count": len(artifact_rows),
        "total_support_count_verified": sum(1 << r.p for r in refs),
        "all_18_artifact_case_pairs_passed": all(r["regraded_strict_pass"] for r in artifact_rows),
        "negative_control_rejected": (neg_verd["strict_pass"] is False and neg_verd["leaf_bounds_valid"] is False),
        "negative_control_verdict": neg_verd,
        "arm_side_by_side_original_vs_regraded": arm_comparison,
        "artifacts": artifact_rows,
    }


def build_router_loto_receipt() -> dict[str, Any]:
    ev = PROJECT_ROOT / "docs/evidence"
    m74 = rtr.load_historical_suite_episode_records(ev / "m7_4_exp1_frozen_20task_and_ablations_20261124")
    m75 = rtr.load_historical_suite_episode_records(ev / "m7_5_exp2_holdout_ood_12task_blind_20261124")

    def _run_loto(records: list[dict[str, Any]], tag: str) -> dict[str, Any]:
        per_task = []
        for idx, rec in enumerate(records):
            train = [r for j, r in enumerate(records) if j != idx]
            pl = rtr.M8OfflineContainerPlanner.fit_from_historical_records(
                train, suite_ids=[f"{tag}_LOTO_excluding_{rec['task_id']}"]
            )
            ev_one = rtr.evaluate_planner_on_records(pl, [rec])["per_task"][0]
            per_task.append(ev_one)
        return {
            "evaluation_protocol": "leave_one_task_out_cross_validation",
            "total_tasks": len(per_task),
            "m8_selective_passes": sum(1 for x in per_task if x["m8_selective_pass"]),
            "spawn_second_branch_count": sum(1 for x in per_task if x["spawn_second_branch"]),
            "effective_turns_sum": sum(int(x["effective_turns"]) for x in per_task),
            "per_task": per_task,
        }

    orig_fn = rtr.build_pre_action_feature_vector

    def _relaxed_partial(s: dict[str, Any]) -> np.ndarray:
        row = orig_fn(s).copy()
        if (
            s["public_verification_state"] == "SUPPORTED_IN_PARTIAL_SCOPE"
            and not bool(s.get("identifiability_warning") or s.get("unresolved_target_identifiability_nogood"))
            and bool(s.get("ended_by_finish"))
            and not bool(s.get("pullback_gate_triggered"))
            and (int(s["skill_activation_count"]) >= 1 or int(s["nogood_count"]) >= 1 or bool(s.get("contract_accepted")))
        ):
            row[-1] = 1.0
        return row

    m74_loto_cur = _run_loto(m74, "m74_current")
    m75_loto_cur = _run_loto(m75, "m75_current")
    try:
        rtr.build_pre_action_feature_vector = _relaxed_partial
        m74_loto_relaxed = _run_loto(m74, "m74_relax_partial_096")
        m75_loto_relaxed = _run_loto(m75, "m75_relax_partial_096")
    finally:
        rtr.build_pre_action_feature_vector = orig_fn

    pl_full = rtr.M8OfflineContainerPlanner.fit_from_historical_records(m74, suite_ids=["m74_full"])
    m74_resub = rtr.evaluate_planner_on_records(pl_full, m74)
    m74_to_m75 = rtr.evaluate_planner_on_records(pl_full, m75)

    return {
        "schema": "hegel_m8_9_1_router_loto_and_schema_receipt_v1",
        "feature_schema_version": rtr.ROUTER_FEATURE_SCHEMA_VERSION,
        "feature_names": ["intercept", *rtr.CONTAINER_STATE_FEATURE_NAMES],
        "feature_builder_sha256": pl_full._feature_builder_sha256(),
        "m74_true_loto_current": m74_loto_cur,
        "m74_true_loto_relax_partial_096": m74_loto_relaxed,
        "m75_true_loto_current": m75_loto_cur,
        "m75_true_loto_relax_partial_096": m75_loto_relaxed,
        "m74_fit_m75_blind_transfer": m74_to_m75,
        "m74_in_sample_resubstitution_reference": {
            "evaluation_protocol": "in_sample_resubstitution_not_loto",
            "total_tasks": m74_resub["total_tasks"],
            "m8_selective_passes": m74_resub["m8_variational_selective_passes"],
            "spawn_second_branch_count": m74_resub["spawn_second_branch_count"],
        },
    }


def build_m89_ledger_errata_audit() -> dict[str, Any]:
    """Recompute exact M8.9 max_turns, card lengths, and recommended vs mounted counts for docs/44 errata."""
    all_tasks = {
        t.task_id: t
        for t in build_holdout_ood_12_tasks() + build_unseen_postfreeze_4_tasks() + list(HARD_STRESS_M89_TASKS)
    }
    max_turns_seen = set()
    card_lengths_by_arm: dict[str, list[int]] = {}
    recovered_entrypoints: dict[str, list[str]] = {}

    for p in sorted((M89_EVIDENCE_DIR / "ablation_episodes").rglob("00_episode_summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if "disclosure_mode" not in s or "Antithesis" not in str(s.get("arm", "")):
            continue
        arm_nm = p.relative_to(M89_EVIDENCE_DIR).parts[1]
        max_turns_seen.add(int(s.get("max_turns", 0)))
        for mod_e in s.get("staged_wormhole_modules") or []:
            for c_str in mod_e.get("index_cards") or []:
                card_lengths_by_arm.setdefault(arm_nm, []).append(len(c_str))
        t_obj = all_tasks[s["task_id"]]
        recovered_entrypoints[s["task_id"]] = list(front._extract_declared_python_entrypoints(t_obj.instruction_text, task=t_obj))

    card_stats = {}
    for arm_nm, lens in sorted(card_lengths_by_arm.items()):
        sl = sorted(lens)
        card_stats[arm_nm] = {
            "card_count": len(sl),
            "min_chars": sl[0],
            "median_chars": float(sl[len(sl) // 2]),
            "mean_chars": round(sum(sl) / len(sl), 2),
            "max_chars": sl[-1],
        }
    return {
        "schema": "hegel_m8_9_1_ledger_errata_recomputation_v1",
        "actual_primary_max_turns_across_all_24_episodes": sorted(max_turns_seen),
        "m89_archived_function_card_char_stats": card_stats,
        "m891_recovered_entrypoints_all_8_tasks": recovered_entrypoints,
        "all_8_tasks_have_nonempty_entrypoints": all(bool(v) for v in recovered_entrypoints.values()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hs01_rec = build_hs01_regrade_receipt()
    loto_rec = build_router_loto_receipt()
    errata_rec = build_m89_ledger_errata_audit()

    (OUT_DIR / "m8_9_hs01_independent_regrade_receipt.json").write_text(
        json.dumps(hs01_rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "m8_9_1_router_true_loto_receipt.json").write_text(
        json.dumps(loto_rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "m8_9_1_ledger_errata_audit.json").write_text(
        json.dumps(errata_rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "hs01_all_18_passed": hs01_rec["all_18_artifact_case_pairs_passed"],
                "hs01_negative_control_rejected": hs01_rec["negative_control_rejected"],
                "arm_regraded": hs01_rec["arm_side_by_side_original_vs_regraded"],
                "m74_loto_current": loto_rec["m74_true_loto_current"]["m8_selective_passes"],
                "m74_loto_relax_096": loto_rec["m74_true_loto_relax_partial_096"]["m8_selective_passes"],
                "actual_max_turns": errata_rec["actual_primary_max_turns_across_all_24_episodes"],
                "card_stats": errata_rec["m89_archived_function_card_char_stats"],
                "recovered_entrypoints": errata_rec["m891_recovered_entrypoints_all_8_tasks"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
