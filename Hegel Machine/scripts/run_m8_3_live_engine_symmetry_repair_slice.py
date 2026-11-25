#!/usr/bin/env python3
"""Hegel M8.3 Live Main-Engine 3-Tier Symmetry/Invariance Escalation Slice (`16` Workers).

Mounts:
  - Frontier-only calibrated `M8OfflineContainerPlanner` (`0` Holdout exposure),
  - Persisted `PersistentMechanismGraph` seeded from M8.2 (`checkpoint_after_p2.json`), and
  - Generalized Multi-Domain Public Symmetry & Contract Invariance Probes + Tier-1 `LocalContainerRepair`
    directly inside `run_m6_9_a_two_branch_episode`!

Executes a fast, concurrent (`--max-workers 16`) 3-task Wave-2 live scientific container slice via official
ChatGPT CLI (`codex`, `auth_mode == "chatgpt"`, `gpt-6.2-sol`):
  1. `ho08__prony_viscoelastic_memory_inversion` (Prospective Wave-2 Rheology / Mori-Zwanzig Prony memory +
     affine load-cell thermal-drift gauge invariance)
  2. `ho10__haldane_bioreactor_runaway_control` (Prospective Wave-2 Exothermic Haldane runaway safety control;
     was blind `UNKNOWN` in M7.5/M8.1!)
  3. `ho12__multicopy_plasmid_cohort_consensus` (Wave-2 Multi-isolate genomics; required `2.0x` second-branch
     spawning in M8.1 — tests whether M8.3 resolves at `1.0x Commit` or `1.1x LocalContainerRepair`!)
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
import threading
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
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph


def _load_m7_runner_module():
    spec = importlib.util.spec_from_file_location("m7_runner", REPO_ROOT / "scripts/run_m7_closing_experiments.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_execution_identity() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()
    h = hashlib.sha256()
    for sub in ("src", "scripts"):
        for fp in sorted((REPO_ROOT / sub).rglob("*.py")):
            rel = fp.relative_to(REPO_ROOT).as_posix()
            h.update(rel.encode("utf-8") + b"\0" + fp.read_bytes() + b"\0")
    return {
        "git_head_commit": head,
        "working_tree_dirty": bool(st),
        "dirty_file_count": len([ln for ln in st.splitlines() if ln.strip()]),
        "code_tree_sha256": h.hexdigest(),
    }


def write_sha256_sums(root: Path) -> int:
    lines: list[str] = []
    for fp in sorted(root.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(root).as_posix()
            lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_3_live_engine_symmetry_repair_slice_20261125",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--max-containers", type=int, default=8)
    args = ap.parse_args()

    t0 = monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if codex_info.get("auth_mode") != "chatgpt":
        raise RuntimeError(f"Official ChatGPT CLI (`auth_mode=='chatgpt'`) required, got {codex_info}")

    m7_runner = _load_m7_runner_module()
    m7_runner.install_quiet_timing_locks(max_containers=args.max_containers)

    # 1. Calibrate Frontier-Only Planner (`0` Holdout exposure!)
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    recs_74 = load_historical_suite_episode_records(m74_dir)
    planner = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    planner_bytes = (json.dumps(planner.to_json_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    planner_fp = args.out_dir / "m8_3_frontier_calibrated_planner.json"
    planner_fp.write_bytes(planner_bytes)
    planner_sha = hashlib.sha256(planner_bytes).hexdigest()

    # 2. Seed `PersistentMechanismGraph` from sealed M8.2 checkpoint (`checkpoint_after_p2.json`)!
    seed_ckpt = (
        REPO_ROOT
        / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125"
        / "checkpoints/checkpoint_after_p2.json"
    )
    shared_graph = PersistentMechanismGraph.load_checkpoint(seed_ckpt) if seed_ckpt.is_file() else PersistentMechanismGraph()
    graph_lock = threading.Lock()

    bundle_path = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    forest, archive, _, bundle_meta = load_autogenous_forest_bundle(bundle_path)
    hippo = HippoRAGWormholeIndex(forest)

    target_ids = (
        "ho08__prony_viscoelastic_memory_inversion",
        "ho10__haldane_bioreactor_runaway_control",
        "ho12__multicopy_plasmid_cohort_consensus",
    )
    ho_waves = holdout_tasks_as_frontier_waves()
    all_ho = {t.task_id: t for tl in ho_waves.values() for t in tl}
    tasks = [all_ho[tid] for tid in target_ids]

    runs_dir = args.out_dir / "runs"
    branches_dir = args.out_dir / "branches"
    runs_dir.mkdir(parents=True, exist_ok=True)
    branches_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"m8_3_live_engine__{args.model}"

    def _run_task(task: m6f.FrontierBenchmarkTask) -> dict[str, Any]:
        # Give each worker an isolated clone of `shared_graph` and merge newly learned guards under `graph_lock`!
        worker_graph = PersistentMechanismGraph.load_checkpoint(seed_ckpt) if seed_ckpt.is_file() else PersistentMechanismGraph()
        ep_dir = runs_dir / task.task_id
        print(f"[M8.3-LIVE START] `{task.task_id}` (max_turns={args.max_turns})...", flush=True)
        s = run_m6_9_a_two_branch_episode(
            task=task,
            model=args.model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo,
            episode_out_dir=ep_dir,
            branches_work_root=branches_dir,
            run_id=run_id,
            max_turns_per_branch=args.max_turns,
            cmd_timeout_sec=150,
            reasoning_effort="medium",
            selective_second_branch=True,
            m8_container_planner=planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=worker_graph,
        )
        with graph_lock:
            for mid, mech in worker_graph.mechanisms.items():
                if mid not in shared_graph.mechanisms:
                    shared_graph.register_mechanism(mech, tree_id="tree_container_invariance")
        s["m8_planner_sha256"] = planner_sha
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sel = s.get("selective_router_decision") or {}
        print(
            f"[M8.3-LIVE DONE] `{task.task_id}`: strict_pass={s['strict_pass']} | "
            f"action={sel.get('selected_action')} (state={sel.get('public_verification_state')}, sym={sel.get('primary_public_metrics', {}).get('symmetry_probe_family')}) | "
            f"thesis_spawned={s['postlock_branch_outcomes']['thesis_spawned']} | turns={s['turns_used']} | wall={s['wall_ms']/1000:.1f}s",
            flush=True,
        )
        return s

    results_by_tid: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(_run_task, t): t.task_id for t in tasks}
        for fut in as_completed(futs):
            results_by_tid[futs[fut]] = fut.result()

    out_ckpt = args.out_dir / "checkpoints" / "m8_3_post_slice_checkpoint.json"
    ckpt_meta = shared_graph.save_checkpoint(out_ckpt)

    ordered = [results_by_tid[t.task_id] for t in tasks]
    passes = sum(1 for s in ordered if s["strict_pass"])
    thesis_spawns = sum(1 for s in ordered if s["postlock_branch_outcomes"]["thesis_spawned"])
    local_repairs = sum(1 for s in ordered if (s.get("local_container_repair_summary") or {}).get("repaired_verified"))
    tot_turns = sum(int(s["turns_used"]) for s in ordered)
    anti_turns = sum(int(s["postlock_branch_outcomes"]["antithesis_turns_used"]) for s in ordered)

    summary = {
        "experiment_id": "m8_3_live_engine_symmetry_repair_slice_20261125",
        "execution_identity": compute_execution_identity(),
        "model": args.model,
        "auth_mode": codex_info["auth_mode"],
        "max_workers": args.max_workers,
        "max_containers": args.max_containers,
        "pretrained_bundle_sha256": bundle_meta["bundle_sha256"],
        "m8_planner_sha256": planner_sha,
        "seed_checkpoint_from_m8_2": str(seed_ckpt.relative_to(REPO_ROOT).as_posix()) if seed_ckpt.is_file() else None,
        "post_slice_checkpoint": ckpt_meta,
        "total_wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "total_tasks": len(ordered),
        "strict_passes": passes,
        "local_container_repairs_verified": local_repairs,
        "second_branch_thesis_spawn_count": thesis_spawns,
        "total_turns_used": tot_turns,
        "turn_multiplier_vs_primary_1x": round(tot_turns / max(1, anti_turns), 4),
        "per_task": [
            {
                "task_id": s["task_id"],
                "strict_pass": s["strict_pass"],
                "declared_budget_multiplier": s["declared_budget_multiplier"],
                "m8_selected_action": (s.get("selective_router_decision") or {}).get("selected_action"),
                "public_verification_state": (s.get("selective_router_decision") or {}).get("public_verification_state"),
                "symmetry_probe_family": ((s.get("selective_router_decision") or {}).get("primary_public_metrics") or {}).get("symmetry_probe_family"),
                "local_repair_attempted": bool((s.get("local_container_repair_summary") or {}).get("attempted")),
                "local_repair_verified": bool((s.get("local_container_repair_summary") or {}).get("repaired_verified")),
                "thesis_spawned": s["postlock_branch_outcomes"]["thesis_spawned"],
                "raw_antithesis_1x_pass": s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"],
                "turns_used": s["turns_used"],
                "skill_activation_count": s.get("skill_activation_count", 0),
                "nogood_count": s.get("nogood_count", 0),
                "total_tokens": s.get("total_tokens", 0),
                "wall_sec": round(s["wall_ms"] / 1000.0, 2),
            }
            for s in ordered
        ],
    }
    (args.out_dir / "00_m8_3_live_engine_slice_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    n_files = write_sha256_sums(args.out_dir)
    print(
        f"[M8.3 COMPLETE] {passes}/{len(ordered)} strict_pass | "
        f"local_repairs={local_repairs} | thesis_spawns={thesis_spawns} | "
        f"turn_mult={summary['turn_multiplier_vs_primary_1x']}x | {n_files} files SHA-256 locked!",
        flush=True,
    )


if __name__ == "__main__":
    main()
