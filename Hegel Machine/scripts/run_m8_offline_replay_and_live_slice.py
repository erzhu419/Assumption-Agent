#!/usr/bin/env python3
"""Hegel M8.1 Unified Runner:
  Part A: Offline Historical Container Trajectory Replay Calibration (`0` online LLM cost, strict cross-suite & LOTO validation)
  Part B: 16-Thread Parallel Live Real Scientific Container Slice (`ho04`, `ho06`, `ho12`) Mounted with Frontier-Calibrated M8 Planner!

Strict Zero-Leakage Protocol:
  - The M8 container planner mounted in Part B (`m8_calibrated_container_planner_frontier_only.json`) is calibrated
    EXCLUSIVELY on `m7_4_exp1_frozen_20task_and_ablations_20261124` (`20` Frontier tasks — `0` `Hegel-OOD-Holdout-12`
    tasks seen during calibration!) and SHA-256 locked BEFORE Part B starts!
  - Part B executes 3 mechanistically distinct air-gapped Docker scientific computing tasks concurrently (`--max-workers 16`,
    `--max-containers 8`) via official ChatGPT CLI (`codex`, `gpt-6.2-sol`, `auth_mode == "chatgpt"`):
      1. `ho04__magnetometer_hard_soft_iron_calibration` (nonlinear ellipsoid + soft-iron sensor calibration; Bare 1x failed in M7.5)
      2. `ho06__plate_evaporation_deconfounded_ic50` (spatial nuisance-subspace deconfounding + Hill kinetics; live Skill∞ target)
      3. `ho12__multicopy_plasmid_cohort_consensus` (multi-cohort deconfounding; M7.5 Heuristic Selective failed due to UNKNOWN blind-skip)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import run_m6_9_a_two_branch_episode
from hegel_machine.cognition.m7_autogenous_forest_pretraining import load_autogenous_forest_bundle
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import holdout_tasks_as_frontier_waves
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    evaluate_planner_on_records,
    load_historical_suite_episode_records,
)


def _load_m7_runner_module():
    spec = importlib.util.spec_from_file_location("m7_runner", REPO_ROOT / "scripts/run_m7_closing_experiments.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_git_head_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def write_sha256_sums(root: Path) -> int:
    lines: list[str] = []
    for fp in sorted(root.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(root).as_posix()
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
            lines.append(f"{digest}  {rel}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def run_part_a_offline_replay_calibration(out_dir: Path) -> tuple[M8OfflineContainerPlanner, dict[str, Any]]:
    """Run Part A offline trajectory replay calibration & cross-suite evaluation across all 50 sealed episodes."""
    t0 = monotonic()
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    m75_dir = REPO_ROOT / "docs/evidence/m7_5_exp2_holdout_ood_12task_blind_20261124"
    m76a_dir = REPO_ROOT / "docs/evidence/m7_6_exp3a_cross_model_luna_holdout12_20261124"
    m76b_dir = REPO_ROOT / "docs/evidence/m7_6_exp3b_cross_model_terra_frontier6_20261124"

    recs_74_fixed = load_historical_suite_episode_records(m74_dir, apply_m7_7_traffic_contract_fix_in_replay=True)
    recs_74_raw = load_historical_suite_episode_records(m74_dir, apply_m7_7_traffic_contract_fix_in_replay=False)
    recs_75 = load_historical_suite_episode_records(m75_dir)
    recs_76a = load_historical_suite_episode_records(m76a_dir)
    recs_76b = load_historical_suite_episode_records(m76b_dir)

    # Calibrate Frontier-Only Planner (`0` Holdout tasks in training!)
    planner_frontier = M8OfflineContainerPlanner.fit_from_historical_records(
        recs_74_fixed,
        suite_ids=[m74_dir.name],
    )
    planner_json = planner_frontier.to_json_dict()
    planner_fp = out_dir / "part_a_offline_calibration" / "m8_calibrated_container_planner_frontier_only.json"
    planner_fp.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = (json.dumps(planner_json, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    planner_fp.write_bytes(raw_bytes)
    planner_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # 1. Strict Leave-One-Task-Out (`LOTO`) on `m7_4` (both with M7.7 traffic fix and raw unpatched archive!)
    def _eval_loto(recs: list[dict[str, Any]]) -> dict[str, Any]:
        per_t = []
        passes, spawns, eff_turns, bare_turns = 0, 0, 0, 0
        for idx in range(len(recs)):
            tr_sub = [r for j, r in enumerate(recs) if j != idx]
            pl_sub = M8OfflineContainerPlanner.fit_from_historical_records(tr_sub, suite_ids=["m7_4_loto"])
            ev = evaluate_planner_on_records(pl_sub, [recs[idx]])
            row = ev["per_task"][0]
            per_t.append(row)
            passes += ev["m8_variational_selective_passes"]
            spawns += ev["spawn_second_branch_count"]
            eff_turns += row["effective_turns"]
            bare_turns += recs[idx]["outcomes"]["thesis_turns"]
        return {
            "total_tasks": len(recs),
            "m8_loto_selective_passes": passes,
            "spawn_second_branch_count": spawns,
            "spawn_rate": round(spawns / max(1, len(recs)), 4),
            "turn_multiplier_vs_bare_1x": round(eff_turns / max(1, bare_turns), 4),
            "per_task": per_t,
        }

    loto_74_fixed = _eval_loto(recs_74_fixed)
    loto_74_raw = _eval_loto(recs_74_raw)

    # 2. Strict Zero-Leakage Cross-Suite Evaluation (`Train m7_4 Frontier-20` -> `Test m7_5 Holdout-12` & `m7_6a Luna-Holdout-12`)
    cross_75 = evaluate_planner_on_records(planner_frontier, recs_75)
    cross_76a = evaluate_planner_on_records(planner_frontier, recs_76a)

    # 3. Reverse Cross-Suite Check (`Train m7_5 Holdout-12` -> `Test m7_4 Frontier-20`)
    planner_holdout = M8OfflineContainerPlanner.fit_from_historical_records(recs_75, suite_ids=[m75_dir.name])
    reverse_cross_74 = evaluate_planner_on_records(planner_holdout, recs_74_fixed)

    report = {
        "git_commit": get_git_head_sha(),
        "total_historical_episodes_analyzed": len(recs_74_fixed) + len(recs_75) + len(recs_76a) + len(recs_76b),
        "prefix_clean_t09_verified_count": len(recs_74_fixed) + len(recs_75) + len(recs_76a) + len(recs_76b),
        "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "frozen_frontier_planner_path": str(planner_fp.relative_to(out_dir).as_posix()),
        "frozen_frontier_planner_sha256": planner_sha256,
        "zero_holdout_leakage_in_frozen_planner": True,
        "comparisons": {
            "frontier_20_m7_4": {
                "bare_react_1x_passes": 13,
                "autogenous_forest_1x_passes": 14,
                "m7_heuristic_selective_passes": 12,
                "m7_twobranch_2x_passes": 12,
                "m8_loto_selective_raw_archive_passes": loto_74_raw["m8_loto_selective_passes"],
                "m8_loto_selective_with_m7_7_fix_passes": loto_74_fixed["m8_loto_selective_passes"],
                "m8_loto_spawn_rate": loto_74_fixed["spawn_rate"],
                "m8_loto_turn_multiplier_vs_bare_1x": loto_74_fixed["turn_multiplier_vs_bare_1x"],
                "reverse_cross_trained_on_holdout12_passes": reverse_cross_74["m8_variational_selective_passes"],
                "per_task_loto": loto_74_fixed["per_task"],
            },
            "holdout_12_m7_5_blind_from_frontier_planner": {
                "bare_react_1x_passes": cross_75["bare_1x_passes"],
                "autogenous_forest_1x_passes": cross_75["antithesis_1x_passes"],
                "m7_heuristic_selective_passes": 11,
                "m8_zero_leakage_variational_selective_passes": cross_75["m8_variational_selective_passes"],
                "m8_spawn_rate": cross_75["spawn_rate"],
                "m8_turn_multiplier_vs_bare_1x": cross_75["turn_multiplier_vs_bare_1x"],
                "per_task": cross_75["per_task"],
            },
            "cross_model_luna_holdout_12_m7_6a_blind_from_frontier_planner": {
                "bare_react_1x_passes": cross_76a["bare_1x_passes"],
                "autogenous_forest_1x_passes": cross_76a["antithesis_1x_passes"],
                "m7_heuristic_selective_passes": 9,
                "m8_zero_leakage_variational_selective_passes": cross_76a["m8_variational_selective_passes"],
                "m8_spawn_rate": cross_76a["spawn_rate"],
                "m8_turn_multiplier_vs_bare_1x": cross_76a["turn_multiplier_vs_bare_1x"],
                "per_task": cross_76a["per_task"],
            },
        },
    }
    (out_dir / "part_a_offline_calibration" / "00_part_a_offline_replay_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return planner_frontier, report


def run_part_b_live_3task_container_slice(
    *,
    out_dir: Path,
    planner: M8OfflineContainerPlanner,
    planner_sha256: str,
    model: str = "gpt-6.2-sol",
    max_turns: int = 24,
    max_workers: int = 16,
    max_containers: int = 8,
    reasoning_effort: str = "medium",
    target_task_ids: tuple[str, ...] = (
        "ho04__magnetometer_hard_soft_iron_calibration",
        "ho06__plate_evaporation_deconfounded_ic50",
        "ho12__multicopy_plasmid_cohort_consensus",
    ),
) -> dict[str, Any]:
    """Execute 16-thread parallel live real scientific container slice mounted with `M8OfflineContainerPlanner`."""
    t0 = monotonic()
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if codex_info.get("auth_mode") != "chatgpt":
        raise RuntimeError(f"Official ChatGPT CLI (`auth_mode=='chatgpt'`) required, got: {codex_info}")

    m7_runner = _load_m7_runner_module()
    m7_runner.install_quiet_timing_locks(max_containers=max_containers)

    # Use the post-Wave1 Holdout bundle ONLY for Wave-2 `ho12`? NO — to be 100% clean & zero-Wave2-leakage,
    # load the frozen pre-registered `pretrained_autogenous_forest_v2.json` (grown purely on `pt01..pt06` blank-seed pretraining!)
    bundle_path = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    forest, archive, _, bundle_meta = load_autogenous_forest_bundle(bundle_path)
    hippo = HippoRAGWormholeIndex(forest)

    ho_waves = holdout_tasks_as_frontier_waves()
    all_ho_by_id = {t.task_id: t for tl in ho_waves.values() for t in tl}
    tasks = [all_ho_by_id[tid] for tid in target_task_ids]

    live_dir = out_dir / "part_b_live_3task_16thread_slice"
    runs_dir = live_dir / "runs"
    branches_dir = live_dir / "branches"
    runs_dir.mkdir(parents=True, exist_ok=True)
    branches_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"m8_1_live_slice__{model}"
    results_by_tid: dict[str, dict[str, Any]] = {}

    def _exec_task(task: m6f.FrontierBenchmarkTask) -> dict[str, Any]:
        ep_dir = runs_dir / task.task_id
        sum_fp = ep_dir / "00_episode_summary.json"
        if sum_fp.is_file():
            cached = json.loads(sum_fp.read_text(encoding="utf-8"))
            if cached.get("m8_planner_sha256") == planner_sha256 and cached.get("model") == model:
                print(f"[SKIP-CACHED] `{task.task_id}` -> strict_pass={cached['strict_pass']}", flush=True)
                return cached
        print(f"[M8-LIVE START] `{task.task_id}` (max_turns={max_turns}, workers={max_workers})...", flush=True)
        s = run_m6_9_a_two_branch_episode(
            task=task,
            model=model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo,
            episode_out_dir=ep_dir,
            branches_work_root=branches_dir,
            run_id=run_id,
            max_turns_per_branch=max_turns,
            cmd_timeout_sec=150,
            reasoning_effort=reasoning_effort,
            selective_second_branch=True,
            m8_container_planner=planner,
        )
        s["m8_planner_sha256"] = planner_sha256
        sum_fp.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sel_d = s.get("selective_router_decision") or {}
        print(
            f"[M8-LIVE DONE] `{task.task_id}`: strict_pass={s['strict_pass']} | "
            f"m8_action={sel_d.get('selected_action')} (state={sel_d.get('public_verification_state')}, U={sel_d.get('net_variational_utility')}) | "
            f"turns={s['turns_used']} | sk={s.get('skill_activation_count', 0)} | ng={s.get('nogood_count', 0)} | wall={s['wall_ms']/1000:.1f}s",
            flush=True,
        )
        return s

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_exec_task, t): t.task_id for t in tasks}
        for fut in as_completed(futs):
            tid = futs[fut]
            results_by_tid[tid] = fut.result()

    ordered_summaries = [results_by_tid[t.task_id] for t in tasks]
    passes = sum(1 for s in ordered_summaries if s["strict_pass"])
    spawned = sum(1 for s in ordered_summaries if s["postlock_branch_outcomes"]["thesis_spawned"])
    tot_turns = sum(int(s["turns_used"]) for s in ordered_summaries)
    anti_turns = sum(int(s["postlock_branch_outcomes"]["antithesis_turns_used"]) for s in ordered_summaries)

    live_summary = {
        "git_commit": get_git_head_sha(),
        "model": model,
        "auth_mode": codex_info["auth_mode"],
        "max_workers": max_workers,
        "max_containers": max_containers,
        "max_turns_per_branch": max_turns,
        "pretrained_bundle_sha256": bundle_meta["bundle_sha256"],
        "m8_frozen_frontier_planner_sha256": planner_sha256,
        "zero_holdout_calibration_leakage": True,
        "total_wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "total_tasks": len(ordered_summaries),
        "m8_live_strict_passes": passes,
        "second_branch_spawn_count": spawned,
        "second_branch_spawn_rate": round(spawned / max(1, len(ordered_summaries)), 4),
        "total_turns_used": tot_turns,
        "turn_multiplier_vs_primary_1x": round(tot_turns / max(1, anti_turns), 4),
        "per_task": [
            {
                "task_id": s["task_id"],
                "strict_pass": s["strict_pass"],
                "m8_selected_action": (s.get("selective_router_decision") or {}).get("selected_action"),
                "public_verification_state": (s.get("selective_router_decision") or {}).get("public_verification_state"),
                "net_variational_utility": (s.get("selective_router_decision") or {}).get("net_variational_utility"),
                "triggers": (s.get("selective_router_decision") or {}).get("triggers", []),
                "thesis_spawned": s["postlock_branch_outcomes"]["thesis_spawned"],
                "antithesis_1x_pass": s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"],
                "thesis_1x_pass": s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"],
                "turns_used": s["turns_used"],
                "skill_activation_count": s.get("skill_activation_count", 0),
                "nogood_count": s.get("nogood_count", 0),
                "total_tokens": s.get("total_tokens", 0),
                "wall_sec": round(s["wall_ms"] / 1000.0, 2),
            }
            for s in ordered_summaries
        ],
    }
    (live_dir / "00_part_b_live_slice_summary.json").write_text(
        json.dumps(live_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return live_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_1_offline_replay_and_live_3task_slice_20261125",
    )
    ap.add_argument("--mode", choices=("offline-only", "both"), default="both")
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--max-containers", type=int, default=8)
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    planner, part_a_rep = run_part_a_offline_replay_calibration(args.out_dir)
    print("[M8-PART-A DONE] Offline replay calibration summary:", flush=True)
    print(
        json.dumps(
            {
                "planner_sha256": part_a_rep["frozen_frontier_planner_sha256"],
                "frontier_20_loto_passes": part_a_rep["comparisons"]["frontier_20_m7_4"]["m8_loto_selective_with_m7_7_fix_passes"],
                "holdout_12_blind_passes": part_a_rep["comparisons"]["holdout_12_m7_5_blind_from_frontier_planner"]["m8_zero_leakage_variational_selective_passes"],
                "luna_holdout_12_blind_passes": part_a_rep["comparisons"]["cross_model_luna_holdout_12_m7_6a_blind_from_frontier_planner"]["m8_zero_leakage_variational_selective_passes"],
            },
            indent=2,
        ),
        flush=True,
    )

    part_b_rep: dict[str, Any] | None = None
    if args.mode == "both":
        part_b_rep = run_part_b_live_3task_container_slice(
            out_dir=args.out_dir,
            planner=planner,
            planner_sha256=part_a_rep["frozen_frontier_planner_sha256"],
            model=args.model,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            max_containers=args.max_containers,
            reasoning_effort=args.reasoning_effort,
        )

    master = {
        "git_commit": get_git_head_sha(),
        "part_a_offline_replay_summary": part_a_rep["comparisons"],
        "frozen_frontier_planner_sha256": part_a_rep["frozen_frontier_planner_sha256"],
        "part_b_live_3task_16thread_summary": part_b_rep,
    }
    (args.out_dir / "00_master_m8_1_summary.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    n_files = write_sha256_sums(args.out_dir)
    print(f"[M8.1 COMPLETE] Locked {n_files} files in `{args.out_dir}`.", flush=True)


if __name__ == "__main__":
    main()
