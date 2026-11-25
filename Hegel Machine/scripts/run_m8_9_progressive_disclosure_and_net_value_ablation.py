#!/usr/bin/env python3
"""Hegel M8.9 Factor-Isolated 3-Arm x 8-Task = 24-Episode Parallel Experiment Runner (`<= 32` `gpt-6.2-sol` CLI).

Executes on a single frozen commit:
1. Clean-container re-verification of `pretrained_autogenous_forest_v4_contracts.json` ->
   `pretrained_autogenous_forest_v5_m89.json` (`6/6` `L1` primitives bound to fresh SHA-256 `unit_test_receipt`s,
   including the `M8.9` `G - diag(d) >= 0` PSD minorant guard on `autogenous_prim_box_ridge_bounds.py`).
2. Hardened `NaN`/`Inf`-proof `gap <= 1e-5` `uf04` re-check receipt on unchanged sealed `M8.8` deliverables
   (`l2_only` and `l1_l2`, plus synthetic `NaN` negative control) -> `m8_8_uf04_hardened_verifier_recheck_receipt.json`.
3. `M8OfflineContainerPlanner` calibration (`feature_schema_version = "m8_9_prefix_clean_v2"`) on sealed `m7_4` (`Frontier-20`).
4. **Factor-Isolated 3-Arm x 8-Task = 24-Episode Parallel Controlled Evaluation** (`--max-episode-workers 24 --max-codex-concurrency 32`):
   - **Arm 1 (`dedup_full_l1_l2`)**: `disclosure_mode="dedup_full"`, `staging_policy="all_compatible"` (`L1+L2` lossless deduplicated contracts).
   - **Arm 2 (`progressive_all_l1_l2`)**: `disclosure_mode="progressive"`, `staging_policy="all_compatible"` (exact same staged modules as Arm 1 -> isolates pure progressive contract disclosure!).
   - **Arm 3 (`progressive_net_value_gated`)**: `disclosure_mode="progressive"`, `staging_policy="net_value_gated"` (exact same progressive disclosure as Arm 2 -> isolates pure expected-net-value staging gate!).
   - **Tasks (`8` total)**:
     - Longitudinal `M8.8` Regression Subset (`4` tasks): `ho01`, `ho06`, `uf03`, `uf04`.
     - New `Hegel-M89-HardStress-4` (`4` tasks): `hs01`, `hs02`, `hs03`, `hs04`.
5. Computes complete separated primary-branch (`1x`) vs whole-policy (`1x + Repair + Selective Challenger`) statistics
   and writes `SHA256SUMS.txt` over all generated artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_4_executable_wormhole_library as m64
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import run_m6_9_a_two_branch_episode
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    ensure_clean_scipy_sandbox_image,
    holdout_tasks_as_frontier_waves,
)
from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import (
    run_unseen_postfreeze_private_verifier_by_id,
    unseen_postfreeze_tasks_as_frontier_list,
)
from hegel_machine.cognition.m8_9_hard_stress_benchmark import get_hard_stress_m89_frontier_tasks
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    evaluate_planner_on_records,
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph

LONGITUDINAL_M88_HOLD_PREFIXES: tuple[str, ...] = ("ho01__", "ho06__")
LONGITUDINAL_M88_UF_PREFIXES: tuple[str, ...] = ("uf03__", "uf04__")

ARMS_CONFIG: tuple[tuple[str, str, str], ...] = (
    ("dedup_full_l1_l2", "dedup_full", "all_compatible"),
    ("progressive_all_l1_l2", "progressive", "all_compatible"),
    ("progressive_net_value_gated", "progressive", "net_value_gated"),
)


def _install_codex_concurrency_limiter(max_codex_concurrency: int) -> dict[str, Any]:
    sem = threading.BoundedSemaphore(max(1, int(max_codex_concurrency)))
    lock = threading.Lock()
    stats = {"active": 0, "peak_concurrent_codex_calls": 0, "total_codex_calls": 0}
    orig_call = m6f.FrontierCodexClient.call_structured

    def _throttled_call(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with sem:
            with lock:
                stats["active"] += 1
                stats["total_codex_calls"] += 1
                if stats["active"] > stats["peak_concurrent_codex_calls"]:
                    stats["peak_concurrent_codex_calls"] = stats["active"]
            try:
                return orig_call(self, *args, **kwargs)
            finally:
                with lock:
                    stats["active"] -= 1

    m6f.FrontierCodexClient.call_structured = _throttled_call  # type: ignore[method-assign]
    return stats


def _git_state() -> tuple[str, list[str]]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    status_lines = [
        ln
        for ln in subprocess.check_output(["git", "status", "--porcelain", "."], cwd=str(REPO_ROOT), text=True).splitlines()
        if ln.strip() and "docs/evidence/m8_9_" not in ln
    ]
    return head, status_lines


def _recheck_sealed_m88_uf04(evidence_dir: Path) -> dict[str, Any]:
    m88_ev_root = REPO_ROOT / "docs/evidence/m8_8_parallel_paired_ablation_and_unseen_ood_20261125"
    uf04_snaps = [
        p
        for p in sorted(m88_ev_root.glob("**/uf04__certified_box_l0_sensor_selection/04_submitted_artifacts_snapshot/certified_l0_solver.py"))
        if "_branches_work" not in p.as_posix()
    ]
    assert len(uf04_snaps) == 2, f"Expected 2 sealed M8.8 uf04 snapshots, found {uf04_snaps}"
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="m89_uf04_recheck_") as td:
        td_p = Path(td)
        for src_py in uf04_snaps:
            arm_nm = "l1_l2" if "l1_l2" in src_py.as_posix() else "l2_only"
            code_bytes = src_py.read_bytes()
            tar_p = td_p / f"uf04_{arm_nm}.tar"
            rel = "holdout_task/certified_l0_solver.py"
            tmp_f = td_p / rel
            tmp_f.parent.mkdir(parents=True, exist_ok=True)
            tmp_f.write_bytes(code_bytes)
            with tarfile.open(tar_p, "w") as tf:
                tf.add(str(tmp_f), arcname=rel)
            ver = run_unseen_postfreeze_private_verifier_by_id("uf04__certified_box_l0_sensor_selection", tar_p)
            results[arm_nm] = {
                "sealed_artifact_path": str(src_py.relative_to(REPO_ROOT)),
                "sealed_artifact_sha256": hashlib.sha256(code_bytes).hexdigest(),
                "strict_pass": bool(ver["strict_pass"]),
                "verifier_metrics": ver.get("verifier_metrics"),
                "failure_taxonomy": ver.get("failure_taxonomy"),
            }

        # Synthetic NaN negative control
        nan_py = (
            b"def solve_certified_box_l0(instance):\n"
            b"    p = len(instance['X'][0])\n"
            b"    return {'weights': [float('nan')]*p, 'objective': float('nan'), 'certified_lower_bound': float('nan'), 'nodes_explored': 1}\n"
        )
        nan_tar = td_p / "uf04_nan_ctrl.tar"
        tmp_nan = td_p / "nan_case/holdout_task/certified_l0_solver.py"
        tmp_nan.parent.mkdir(parents=True, exist_ok=True)
        tmp_nan.write_bytes(nan_py)
        with tarfile.open(nan_tar, "w") as tf:
            tf.add(str(tmp_nan), arcname="holdout_task/certified_l0_solver.py")
        ver_nan = run_unseen_postfreeze_private_verifier_by_id("uf04__certified_box_l0_sensor_selection", nan_tar)
        results["synthetic_nan_negative_control"] = {
            "strict_pass": bool(ver_nan["strict_pass"]),
            "verifier_metrics": ver_nan.get("verifier_metrics"),
            "failure_taxonomy": ver_nan.get("failure_taxonomy"),
        }

    receipt = {
        "schema": "hegel_m8_9_uf04_hardened_verifier_recheck_v1",
        "hardened_verifier_rules": {
            "non_finite_rejected": True,
            "bool_scalars_rejected": True,
            "json_allow_nan": False,
            "gap_threshold": 1e-5,
            "objective_consistency_tol": 1e-6,
        },
        "results": results,
    }
    assert results["l2_only"]["strict_pass"] is True
    assert results["l1_l2"]["strict_pass"] is True
    assert results["synthetic_nan_negative_control"]["strict_pass"] is False
    out_fp = evidence_dir / "m8_8_uf04_hardened_verifier_recheck_receipt.json"
    out_fp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def _extract_primary_branch_metrics(ep_dir: Path, ep_sum: dict[str, Any]) -> dict[str, Any]:
    """Read isolated `branch_2_antithesis_wormhole/00_episode_summary.json` so Primary (`1x`) tokens/turns are
    NEVER conflated with whole-policy (`1x + Repair + Selective Challenger`) metrics (`Hegel_M8_8_Audit` §3.2 & §9)!
    """
    work_root = ep_dir / "_branches_work"
    anti_sums = sorted(work_root.glob("**/branch_2_antithesis_wormhole/00_episode_summary.json")) if work_root.is_dir() else []
    if anti_sums:
        a_s = json.loads(anti_sums[0].read_text(encoding="utf-8"))
        init_manifest_chars = len(json.dumps(a_s.get("staged_wormhole_modules", []), ensure_ascii=False))
        return {
            "primary_turns_used": int(a_s.get("turns_used", 0)),
            "primary_ended_by_finish": bool(a_s.get("ended_by_finish", False)),
            "primary_strict_pass_postlock": bool((ep_sum.get("postlock_branch_outcomes") or {}).get("antithesis_wormhole_strict_pass", False)),
            "primary_prompt_tokens": int(a_s.get("prompt_tokens", 0)),
            "primary_cached_tokens": int(a_s.get("cached_tokens", 0)),
            "primary_completion_tokens": int(a_s.get("completion_tokens", 0)),
            "primary_total_tokens": int(a_s.get("total_tokens", 0)),
            "primary_effective_billed_tokens": float(a_s.get("effective_billed_tokens", 0.0)),
            "primary_wall_ms": float(a_s.get("wall_ms", 0.0)),
            "turn1_staged_manifest_json_chars": init_manifest_chars,
            "staged_module_imports": [m.get("module_import") for m in a_s.get("staged_wormhole_modules", [])],
            "describe_skill_inspection_count": int((a_s.get("skill_adoption_funnel") or {}).get("describe_skill_inspection_count", 0)),
            "describe_skill_queries": list((a_s.get("skill_adoption_funnel") or {}).get("describe_skill_queries", [])),
        }
    return {
        "primary_turns_used": int(ep_sum.get("turns_used", 0)),
        "primary_ended_by_finish": bool(ep_sum.get("ended_by_finish", False)),
        "primary_strict_pass_postlock": bool(ep_sum.get("strict_pass", False)),
        "primary_prompt_tokens": int(ep_sum.get("prompt_tokens", 0)),
        "primary_cached_tokens": int(ep_sum.get("cached_tokens", 0)),
        "primary_completion_tokens": int(ep_sum.get("completion_tokens", 0)),
        "primary_total_tokens": int(ep_sum.get("total_tokens", 0)),
        "primary_effective_billed_tokens": float(ep_sum.get("effective_billed_tokens", 0.0)),
        "primary_wall_ms": float(ep_sum.get("wall_ms", 0.0)),
        "turn1_staged_manifest_json_chars": len(json.dumps(ep_sum.get("staged_wormhole_modules", []), ensure_ascii=False)),
        "staged_module_imports": [m.get("module_import") for m in ep_sum.get("staged_wormhole_modules", [])],
        "describe_skill_inspection_count": int((ep_sum.get("skill_adoption_funnel") or {}).get("describe_skill_inspection_count", 0)),
        "describe_skill_queries": list((ep_sum.get("skill_adoption_funnel") or {}).get("describe_skill_queries", [])),
    }


def _write_sha256sums(evidence_dir: Path) -> int:
    sha_fp = evidence_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for p in sorted(evidence_dir.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS.txt":
            continue
        rel = p.relative_to(evidence_dir).as_posix()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    sha_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel M8.9 3-Arm x 8-Task Parallel Controlled Experiment")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--max-turns", type=int, default=28)
    parser.add_argument("--max-episode-workers", type=int, default=24)
    parser.add_argument("--max-codex-concurrency", type=int, default=32)
    args = parser.parse_args()

    t_global0 = monotonic()
    ev_dir: Path = args.evidence_dir.resolve()
    ev_dir.mkdir(parents=True, exist_ok=True)

    git_commit, dirty_lines = _git_state()
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt", f"Official ChatGPT CLI required: {codex_info}"
    ensure_clean_scipy_sandbox_image()
    codex_stats = _install_codex_concurrency_limiter(args.max_codex_concurrency)

    print(f"[M8.9] Frozen Git Commit: {git_commit} (dirty={len(dirty_lines)}) | Model={args.model} | MaxCodex={args.max_codex_concurrency}", flush=True)

    # 1. Re-verify `v4 -> v5` in a clean container so `autogenous_prim_box_ridge_bounds` gets a fresh SHA-bound receipt!
    v4_fp = (
        REPO_ROOT
        / "docs/evidence/m8_8_parallel_paired_ablation_and_unseen_ood_20261125/pretrained_autogenous_forest_v4_m88.json"
    )
    base_forest, base_archive, meta_dag, _ = m7a.load_autogenous_forest_bundle(v4_fp)
    enr_v5 = m7a.enrich_and_harden_l1_primitive_contracts(base_archive, reverify_in_clean_container=True)
    v5_fp = ev_dir / "pretrained_autogenous_forest_v5_m89.json"
    m7a.serialize_autogenous_forest_bundle(
        forest=base_forest,
        archive=base_archive,
        meta_dag=meta_dag,
        out_path=v5_fp,
        extra_metadata={"m8_9_contract_and_minorant_reverification": enr_v5},
    )
    print(f"[M8.9] Saved v5 forest bundle: {v5_fp.name} | reverified={enr_v5['reverified_modules']}", flush=True)

    # 2. Re-check sealed `M8.8` `uf04` deliverables against hardened `NaN`-proof `1e-5` verifier
    uf04_recheck = _recheck_sealed_m88_uf04(ev_dir)
    print(
        f"[M8.9] Sealed M8.8 uf04 re-check: l2_only={uf04_recheck['results']['l2_only']['strict_pass']} "
        f"l1_l2={uf04_recheck['results']['l1_l2']['strict_pass']} nan_ctrl={uf04_recheck['results']['synthetic_nan_negative_control']['strict_pass']}",
        flush=True,
    )

    # 3. Calibrate `M8OfflineContainerPlanner` (`m8_9_prefix_clean_v2`) on sealed `m7_4` (`Frontier-20`)
    m74_suite_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    m74_records = load_historical_suite_episode_records(m74_suite_dir)
    cal_planner = M8OfflineContainerPlanner.fit_from_historical_records(
        m74_records,
        suite_ids=[m74_suite_dir.name],
    )
    cal_summary = evaluate_planner_on_records(cal_planner, m74_records)
    (ev_dir / "m8_9_container_router_calibration.json").write_text(
        json.dumps({"calibration_summary": cal_summary, "planner": cal_planner.to_json_dict()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 4. Assemble the 8 evaluation tasks (`4` longitudinal M8.8 + `4` new HardStress-4)
    ho_waves = holdout_tasks_as_frontier_waves()
    all_ho_tasks = ho_waves["wave_1"] + ho_waves["wave_2"]
    all_uf_tasks = unseen_postfreeze_tasks_as_frontier_list()
    eval_tasks: list[m6f.FrontierBenchmarkTask] = (
        [t for t in all_ho_tasks if t.task_id.startswith(LONGITUDINAL_M88_HOLD_PREFIXES)]
        + [t for t in all_uf_tasks if t.task_id.startswith(LONGITUDINAL_M88_UF_PREFIXES)]
        + get_hard_stress_m89_frontier_tasks()
    )
    assert len(eval_tasks) == 8, f"Expected 8 eval_tasks, got {[t.task_id for t in eval_tasks]}"

    # Record pre-episode net-value gate table across all 8 tasks for full transparency
    net_value_matrix: dict[str, Any] = {}
    for t_obj in eval_tasks:
        _, _, diag_t = m64.collect_recalled_executable_wormhole_sources(
            [],
            include_foundational_skills=False,
            allow_l1_primitives=True,
            full_forest_archive=base_archive,
            task_query_text=t_obj.instruction_text,
            disclosure_mode="progressive",
            staging_policy="net_value_gated",
            return_diagnostics=True,
        )
        net_value_matrix[t_obj.task_id] = {
            "selected_l1_modules": diag_t["selected_l1_modules"],
            "excluded_l1_candidates": diag_t["excluded_l1_candidates"],
            "net_value_evaluations": diag_t["net_value_evaluations"],
        }
    (ev_dir / "m8_9_pre_run_net_value_gate_matrix.json").write_text(
        json.dumps(net_value_matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 5. Launch ALL `3 x 8 = 24` episodes concurrently with per-episode isolated deep-copied graphs & archives!
    episodes_root = ev_dir / "ablation_episodes"
    jobs: list[tuple[str, str, str, m6f.FrontierBenchmarkTask]] = [
        (arm_id, disc_mode, stg_pol, t_obj)
        for arm_id, disc_mode, stg_pol in ARMS_CONFIG
        for t_obj in eval_tasks
    ]
    print(f"[M8.9] Launching {len(jobs)} episodes across {args.max_episode_workers} workers...", flush=True)

    def _run_single_job(job: tuple[str, str, str, m6f.FrontierBenchmarkTask]) -> tuple[str, str, dict[str, Any]]:
        arm_id, disc_mode, stg_pol, task_obj = job
        ep_dir = episodes_root / arm_id / task_obj.task_id
        sum_fp = ep_dir / "00_episode_summary.json"
        if sum_fp.is_file():
            loaded = json.loads(sum_fp.read_text(encoding="utf-8"))
            return arm_id, task_obj.task_id, loaded

        ep_forest = deepcopy(base_forest)
        ep_archive = deepcopy(base_archive)
        ep_archive["__autogenous_blank_seed__"] = {"enabled": True}
        ep_hippo = HippoRAGWormholeIndex(ep_forest)
        ep_graph = PersistentMechanismGraph(forest=ep_forest)
        ep_planner = M8OfflineContainerPlanner.from_json_dict(cal_planner.to_json_dict())

        res = run_m6_9_a_two_branch_episode(
            task_obj,
            model=args.model,
            codex_info=codex_info,
            forest=ep_forest,
            forest_archive=ep_archive,
            hippo_idx=ep_hippo,
            episode_out_dir=ep_dir,
            branches_work_root=ep_dir / "_branches_work",
            run_id=f"m8_9_{arm_id}",
            max_turns_per_branch=args.max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=args.reasoning_effort,
            selective_second_branch=True,
            m8_container_planner=ep_planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=ep_graph,
            m8_graph_checkpoint_path=ep_dir / "m8_graph_checkpoint.json",
            episode_nominal_budget=2.5,
            allow_l1_primitives=True,
            disclosure_mode=disc_mode,
            staging_policy=stg_pol,
        )
        return arm_id, task_obj.task_id, res

    results_by_arm_task: dict[str, dict[str, dict[str, Any]]] = {a[0]: {} for a in ARMS_CONFIG}
    job_errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_episode_workers))) as pool:
        fut_to_job = {pool.submit(_run_single_job, jb): jb for jb in jobs}
        done_cnt = 0
        for fut in as_completed(fut_to_job):
            done_cnt += 1
            arm_id, _, _, task_obj = fut_to_job[fut]
            try:
                r_arm, r_tid, r_sum = fut.result()
                results_by_arm_task[r_arm][r_tid] = r_sum
                prim = _extract_primary_branch_metrics(episodes_root / r_arm / r_tid, r_sum)
                print(
                    f"[{done_cnt:02d}/{len(jobs):02d}] {r_arm} | {r_tid} -> "
                    f"strict_pass={r_sum['strict_pass']} | prim_turns={prim['primary_turns_used']} | "
                    f"policy_units={r_sum.get('episode_nominal_units_total')} | "
                    f"prim_tok={prim['primary_total_tokens']:,} | policy_tok={r_sum['total_tokens']:,}",
                    flush=True,
                )
            except Exception as exc:
                err_rec = {"arm": arm_id, "task_id": task_obj.task_id, "error": f"{type(exc).__name__}:{exc}"}
                job_errors.append(err_rec)
                print(f"[ERROR] {err_rec}", flush=True)

    if job_errors:
        (ev_dir / "m8_9_job_errors.json").write_text(json.dumps(job_errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"M8.9 ablation encountered {len(job_errors)} episode error(s): {job_errors}")

    # 6. Build calibrated aggregate + per-suite + per-task comparison tables (`Primary 1x` strictly separated from `Full Policy`!)
    arm_aggregates: dict[str, Any] = {}
    per_task_comparison: list[dict[str, Any]] = []

    for t_obj in eval_tasks:
        tid = t_obj.task_id
        row_arms: dict[str, Any] = {}
        for arm_id, disc_mode, stg_pol in ARMS_CONFIG:
            s = results_by_arm_task[arm_id][tid]
            ep_dir = episodes_root / arm_id / tid
            prim = _extract_primary_branch_metrics(ep_dir, s)
            funnel = s.get("skill_adoption_funnel") or {}
            ret = funnel.get("final_selected_artifact_retention") or {}
            row_arms[arm_id] = {
                "disclosure_mode": disc_mode,
                "staging_policy": stg_pol,
                "policy_strict_pass": bool(s["strict_pass"]),
                "primary_1x_strict_pass": prim["primary_strict_pass_postlock"],
                "primary_1x_turns_used": prim["primary_turns_used"],
                "policy_turns_used": int(s["turns_used"]),
                "policy_nominal_units": float(s.get("episode_nominal_units_total", 1.0)),
                "second_branch_spawned": bool((s.get("postlock_branch_outcomes") or {}).get("thesis_spawned", False)),
                "local_repair_attempted": bool((s.get("local_container_repair_summary") or {}).get("attempted", False)),
                "local_repair_verified": bool((s.get("local_container_repair_summary") or {}).get("repaired_verified", False)),
                "selected_branch": (s.get("synthesis_decision") or {}).get("selected_branch"),
                "turn1_staged_manifest_json_chars": prim["turn1_staged_manifest_json_chars"],
                "staged_module_count": int(funnel.get("staged_module_count", 0)),
                "staged_l1_primitive_count": int(funnel.get("staged_l1_primitive_count", 0)),
                "staged_module_imports": prim["staged_module_imports"],
                "describe_skill_inspection_count": prim["describe_skill_inspection_count"],
                "describe_skill_queries": prim["describe_skill_queries"],
                "dynamic_skill_calls": int(funnel.get("dynamic_skill_activation_count", 0)),
                "dynamic_l1_calls": int(funnel.get("dynamic_l1_primitive_activation_count", 0)),
                "dynamic_unique_l1_functions": list(funnel.get("dynamic_unique_l1_functions_called", [])),
                "final_retained_l1_modules": list(ret.get("final_retained_l1_modules", [])),
                "entrypoint_reachable_l1_functions": list(ret.get("entrypoint_reachable_l1_functions", [])),
                "all_retained_ast_match_source": bool(ret.get("all_retained_ast_match_source", True)),
                "primary_1x_prompt_tokens": prim["primary_prompt_tokens"],
                "primary_1x_cached_tokens": prim["primary_cached_tokens"],
                "primary_1x_completion_tokens": prim["primary_completion_tokens"],
                "primary_1x_total_tokens": prim["primary_total_tokens"],
                "primary_1x_effective_billed_tokens": round(prim["primary_effective_billed_tokens"], 2),
                "primary_1x_wall_sec": round(prim["primary_wall_ms"] / 1000.0, 2),
                "policy_total_tokens": int(s["total_tokens"]),
                "policy_effective_billed_tokens": round(float(s["effective_billed_tokens"]), 2),
                "policy_wall_sec": round(float(s["wall_ms"]) / 1000.0, 2),
            }
        per_task_comparison.append(
            {
                "task_id": tid,
                "benchmark": t_obj.benchmark,
                "scientific_domain": t_obj.scientific_domain,
                "arms": row_arms,
            }
        )

    for arm_id, disc_mode, stg_pol in ARMS_CONFIG:
        rows_a = [r["arms"][arm_id] for r in per_task_comparison]
        long_rows = [r["arms"][arm_id] for r in per_task_comparison if r["benchmark"] != "Hegel-M89-HardStress-4"]
        hs_rows = [r["arms"][arm_id] for r in per_task_comparison if r["benchmark"] == "Hegel-M89-HardStress-4"]

        def _agg_subset(sub: list[dict[str, Any]]) -> dict[str, Any]:
            n_s = max(len(sub), 1)
            return {
                "episode_count": len(sub),
                "policy_strict_pass_count": sum(1 for x in sub if x["policy_strict_pass"]),
                "policy_strict_pass_rate": round(sum(1 for x in sub if x["policy_strict_pass"]) / n_s, 4),
                "primary_1x_strict_pass_count": sum(1 for x in sub if x["primary_1x_strict_pass"]),
                "primary_1x_strict_pass_rate": round(sum(1 for x in sub if x["primary_1x_strict_pass"]) / n_s, 4),
                "mean_primary_1x_turns": round(sum(x["primary_1x_turns_used"] for x in sub) / n_s, 2),
                "mean_policy_turns": round(sum(x["policy_turns_used"] for x in sub) / n_s, 2),
                "mean_policy_nominal_units": round(sum(x["policy_nominal_units"] for x in sub) / n_s, 3),
                "second_branch_spawn_count": sum(1 for x in sub if x["second_branch_spawned"]),
                "local_repair_attempt_count": sum(1 for x in sub if x["local_repair_attempted"]),
                "local_repair_verified_count": sum(1 for x in sub if x["local_repair_verified"]),
                "mean_turn1_manifest_json_chars": round(sum(x["turn1_staged_manifest_json_chars"] for x in sub) / n_s, 1),
                "mean_staged_l1_count": round(sum(x["staged_l1_primitive_count"] for x in sub) / n_s, 2),
                "episodes_with_describe_skill_cli": sum(1 for x in sub if x["describe_skill_inspection_count"] > 0),
                "total_describe_skill_inspections": sum(x["describe_skill_inspection_count"] for x in sub),
                "episodes_with_dynamic_l1_calls": sum(1 for x in sub if x["dynamic_l1_calls"] > 0),
                "total_dynamic_l1_calls": sum(x["dynamic_l1_calls"] for x in sub),
                "episodes_with_exact_retained_l1_deliverable": sum(1 for x in sub if len(x["final_retained_l1_modules"]) > 0),
                "episodes_with_entrypoint_reachable_l1": sum(1 for x in sub if len(x["entrypoint_reachable_l1_functions"]) > 0),
                "primary_1x_total_tokens_sum": sum(x["primary_1x_total_tokens"] for x in sub),
                "primary_1x_effective_billed_tokens_sum": round(sum(x["primary_1x_effective_billed_tokens"] for x in sub), 2),
                "mean_primary_1x_wall_sec": round(sum(x["primary_1x_wall_sec"] for x in sub) / n_s, 2),
                "policy_total_tokens_sum": sum(x["policy_total_tokens"] for x in sub),
                "policy_effective_billed_tokens_sum": round(sum(x["policy_effective_billed_tokens"] for x in sub), 2),
                "mean_policy_wall_sec": round(sum(x["policy_wall_sec"] for x in sub) / n_s, 2),
            }

        arm_aggregates[arm_id] = {
            "disclosure_mode": disc_mode,
            "staging_policy": stg_pol,
            "all_8_tasks": _agg_subset(rows_a),
            "longitudinal_m88_4_tasks": _agg_subset(long_rows),
            "hard_stress_m89_4_tasks": _agg_subset(hs_rows),
        }

    # Factor-isolated contrasts:
    # Contrast A (`Progressive Disclosure Effect`): `progressive_all_l1_l2` minus `dedup_full_l1_l2` (identical staged module sets!)
    # Contrast B (`Expected-Net-Value Gate Effect`): `progressive_net_value_gated` minus `progressive_all_l1_l2` (identical disclosure format!)
    a1 = arm_aggregates["dedup_full_l1_l2"]["all_8_tasks"]
    a2 = arm_aggregates["progressive_all_l1_l2"]["all_8_tasks"]
    a3 = arm_aggregates["progressive_net_value_gated"]["all_8_tasks"]

    factor_contrasts = {
        "contrast_A_progressive_disclosure_effect_arm2_vs_arm1": {
            "holds_constant": "staging_policy='all_compatible' (exact same staged L1+L2 modules)",
            "turn1_manifest_chars_delta": round(a2["mean_turn1_manifest_json_chars"] - a1["mean_turn1_manifest_json_chars"], 1),
            "turn1_manifest_chars_pct_change": round(
                100.0 * (a2["mean_turn1_manifest_json_chars"] - a1["mean_turn1_manifest_json_chars"]) / max(a1["mean_turn1_manifest_json_chars"], 1.0),
                2,
            ),
            "policy_strict_pass_delta": a2["policy_strict_pass_count"] - a1["policy_strict_pass_count"],
            "primary_1x_strict_pass_delta": a2["primary_1x_strict_pass_count"] - a1["primary_1x_strict_pass_count"],
            "mean_primary_1x_turns_delta": round(a2["mean_primary_1x_turns"] - a1["mean_primary_1x_turns"], 2),
            "primary_1x_total_tokens_pct_change": round(
                100.0 * (a2["primary_1x_total_tokens_sum"] - a1["primary_1x_total_tokens_sum"]) / max(a1["primary_1x_total_tokens_sum"], 1),
                2,
            ),
            "policy_total_tokens_pct_change": round(
                100.0 * (a2["policy_total_tokens_sum"] - a1["policy_total_tokens_sum"]) / max(a1["policy_total_tokens_sum"], 1),
                2,
            ),
        },
        "contrast_B_net_value_staging_gate_effect_arm3_vs_arm2": {
            "holds_constant": "disclosure_mode='progressive' (exact same compact index cards + --describe CLI)",
            "mean_staged_l1_count_delta": round(a3["mean_staged_l1_count"] - a2["mean_staged_l1_count"], 2),
            "policy_strict_pass_delta": a3["policy_strict_pass_count"] - a2["policy_strict_pass_count"],
            "primary_1x_strict_pass_delta": a3["primary_1x_strict_pass_count"] - a2["primary_1x_strict_pass_count"],
            "mean_primary_1x_turns_delta": round(a3["mean_primary_1x_turns"] - a2["mean_primary_1x_turns"], 2),
            "primary_1x_total_tokens_pct_change": round(
                100.0 * (a3["primary_1x_total_tokens_sum"] - a2["primary_1x_total_tokens_sum"]) / max(a2["primary_1x_total_tokens_sum"], 1),
                2,
            ),
            "policy_total_tokens_pct_change": round(
                100.0 * (a3["policy_total_tokens_sum"] - a2["policy_total_tokens_sum"]) / max(a2["policy_total_tokens_sum"], 1),
                2,
            ),
        },
    }

    master_summary = {
        "experiment_id": "hegel_m8_9_progressive_disclosure_and_net_value_ablation_20261126",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit,
        "git_dirty_at_start": len(dirty_lines) > 0,
        "git_dirty_files": dirty_lines,
        "single_clean_invocation": len(dirty_lines) == 0,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "auth_mode": codex_info["auth_mode"],
        "provider_backend": codex_info["provider_backend"],
        "concurrency_telemetry": {
            "configured_max_episode_workers": args.max_episode_workers,
            "configured_max_codex_concurrency": args.max_codex_concurrency,
            "peak_concurrent_codex_calls": codex_stats["peak_concurrent_codex_calls"],
            "total_codex_calls": codex_stats["total_codex_calls"],
            "total_wall_clock_sec": round(monotonic() - t_global0, 2),
        },
        "v5_forest_bundle": {
            "filename": v5_fp.name,
            "sha256": hashlib.sha256(v5_fp.read_bytes()).hexdigest(),
            "enrichment_and_reverification": enr_v5,
        },
        "sealed_m8_8_uf04_recheck": uf04_recheck["results"],
        "router_calibration": cal_summary,
        "arm_aggregates": arm_aggregates,
        "factor_isolated_contrasts": factor_contrasts,
        "per_task_comparison": per_task_comparison,
    }
    (ev_dir / "m8_9_ablation_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_hashed = _write_sha256sums(ev_dir)
    print(f"[M8.9] COMPLETE in {round(monotonic() - t_global0, 2)}s | Hashed {n_hashed} files into SHA256SUMS.txt", flush=True)


if __name__ == "__main__":
    main()
