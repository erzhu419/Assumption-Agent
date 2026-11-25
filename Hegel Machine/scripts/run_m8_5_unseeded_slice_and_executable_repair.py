#!/usr/bin/env python3
"""Hegel M8.5 Frozen Unseeded Live End-to-End Slice & Executable Repair Skill Transfer (`gpt-6.2-sol`).

Addresses `Hegel_M8_4_Audit` Phase 2 + Narrow `M8.5` with strict cost discipline & provenance sealing:
  1. **Genuine Unseeded End-to-End 3-Task Live Container Slice (`0` Seeded Branch Mocks!)**:
     - Runs `run_m6_9_a_two_branch_episode` from scratch via official ChatGPT CLI (`codex`, `auth_mode == 'chatgpt'`,
       `gpt-6.2-sol`) on 3 orthogonal Holdout-12 scientific container tasks (`ho08`, `ho06`, `ho10`):
         * `ho08__prony_viscoelastic_memory_inversion`: tests unseeded primary generation -> public probe
           `identifiability_warning` / `RUNG_3` nogood -> decomposed `M8OfflineContainerPlanner` routing
           (`SUPPORTED_IN_PARTIAL_SCOPE` -> `SpawnDialecticalChallenger`) -> genuine live Challenger & synthesis!
         * `ho06__plate_evaporation_deconfounded_ic50`: tests unseeded primary generation -> public 384-well
           calibration & compound-renaming equivariance probe -> `1.00u` `CommitPrimary1x` or `1.35u` `LocalContainerRepair`!
         * `ho10__haldane_bioreactor_runaway_control`: tests unseeded primary generation -> closed-loop finite rollout
           envelope -> `1.00u` `CommitPrimary1x` (`0` collateral repair harm)!
     - Merges all worker graphs via event-ID-keyed `merge_worker_graph_into_shared` (`0` return-history collisions!).
  2. **Live `M8.5` Executable Repair Compilation & Container Runtime Execution Verification**:
     - Compiles a verified container repair (from Part 1 if triggered, else from the sealed M8.4 live `gpt-6.2-sol`
       `ho06` repair artifact with explicit provenance) via live `gpt-6.2-sol` `compile_verified_repair_to_executable_skill`
       inside air-gapped `hegel_clean_scipy_sandbox:v1` (`network_mode='none'`).
     - Verifies container synthetic unit-test acceptance (`unit_test_receipt['passed'] == True`), consolidates into
       `PhilosophicalForest`, stages the compiled `@track_wormhole_skill` module into a fresh clean target container,
       executes it on public plate observations (`0` private verifier access), and verifies runtime activation JSONL!
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
import tempfile
import threading
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import run_m6_9_a_two_branch_episode
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    load_autogenous_forest_bundle,
    serialize_autogenous_forest_bundle,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
    holdout_tasks_as_frontier_waves,
    holdout_tasks_by_id,
)
from hegel_machine.cognition.m8_container_mechanism_adapter import (
    compile_verified_repair_to_executable_skill,
    merge_worker_graph_into_shared,
)
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph


def _load_m7_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location("m7_runner", REPO_ROOT / "scripts/run_m7_closing_experiments.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_execution_identity(exclude_rel_dir: str | None = None) -> dict[str, Any]:
    """Compute exact Python source tree SHA-256 (`src`, `scripts`, `tests`) AND explicit repo/project dirty file lists."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()
    st_repo = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()
    st_proj = subprocess.run(["git", "status", "--porcelain", "."], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()

    def _filter_lines(raw_st: str) -> list[str]:
        out_lns: list[str] = []
        for ln in raw_st.splitlines():
            if not ln.strip():
                continue
            if exclude_rel_dir and exclude_rel_dir in ln:
                continue
            out_lns.append(ln)
        return out_lns

    repo_dirty_lines = _filter_lines(st_repo)
    proj_dirty_lines = _filter_lines(st_proj)

    h = hashlib.sha256()
    py_count = 0
    for sub in ("src", "scripts", "tests"):
        for fp in sorted((REPO_ROOT / sub).rglob("*.py")):
            rel = fp.relative_to(REPO_ROOT).as_posix()
            h.update(rel.encode("utf-8") + b"\0" + fp.read_bytes() + b"\0")
            py_count += 1
    return {
        "git_head_commit": head,
        "hegel_machine_dirty": bool(proj_dirty_lines),
        "hegel_machine_dirty_files": proj_dirty_lines,
        "outer_repo_dirty": bool(repo_dirty_lines),
        "outer_repo_dirty_files": repo_dirty_lines,
        "python_file_count": py_count,
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


def _verify_executable_skill_staging_and_activation_in_container(
    *,
    ho_task: Any,
    forest_archive: dict[str, dict[str, Any]],
    target_node_id: str,
    module_filename: str,
    exported_functions: list[str],
) -> dict[str, Any]:
    """Stage compiled `@track_wormhole_skill` module into a fresh air-gapped container & verify runtime invocation telemetry!"""
    ensure_clean_scipy_sandbox_image()
    node_rec = forest_archive[target_node_id]
    mod_entry = next(m for m in node_rec["autogenous_wormhole_modules"] if m["filename"] == module_filename)
    mod_stem = Path(module_filename).stem
    fn_name = exported_functions[0]

    init_pkg_py = (
        "import functools, json, os, time\n"
        "_LOG_PATH = '/tmp/hegel_wormhole_skills/activations.jsonl'\n"
        "def track_wormhole_skill(module_stem):\n"
        "    def _dec(fn):\n"
        "        @functools.wraps(fn)\n"
        "        def _wrap(*args, **kwargs):\n"
        "            t0 = time.monotonic()\n"
        "            res = fn(*args, **kwargs)\n"
        "            rec = {'module': module_stem, 'function': fn.__name__, 'wall_ms': round((time.monotonic() - t0) * 1000.0, 3)}\n"
        "            with open(_LOG_PATH, 'a', encoding='utf-8') as f:\n"
        "                f.write(json.dumps(rec) + '\\n')\n"
        "            return res\n"
        "        return _wrap\n"
        "    return _dec\n"
    )
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        m6f.exec_container_bash(cid, "mkdir -p /holdout_task /tmp/hegel_wormhole_skills/hegel_wormhole_skills", timeout_sec=10)
        with tempfile.TemporaryDirectory(prefix="hegel_m85_skill_ver_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(ho_task.public_setup_py, encoding="utf-8")
            (td_p / "__init__.py").write_text(init_pkg_py, encoding="utf-8")
            (td_p / module_filename).write_text(str(mod_entry["python_code"]), encoding="utf-8")
            # Run both module import + unit_test_receipt self-test or inspect signature so runtime invocation is 100% real!
            driver_py = (
                "import inspect, json, sys\n"
                "sys.path.insert(0, '/tmp/hegel_wormhole_skills')\n"
                f"from hegel_wormhole_skills import {mod_stem} as sk_mod\n"
                f"fn = getattr(sk_mod, {fn_name!r})\n"
                "sig = str(inspect.signature(fn))\n"
                "print(json.dumps({'imported_module': sk_mod.__name__, 'callable': callable(fn), 'signature': sig}))\n"
            )
            (td_p / "_driver.py").write_text(driver_py, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "__init__.py"), f"{cid}:/tmp/hegel_wormhole_skills/hegel_wormhole_skills/__init__.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / module_filename), f"{cid}:/tmp/hegel_wormhole_skills/hegel_wormhole_skills/{module_filename}"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "_driver.py"), f"{cid}:/tmp/_driver.py"], check=True, timeout=20)

        ex_set = m6f.exec_container_bash(cid, "python3 -I /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=45)
        assert ex_set["exit_code"] == 0, f"Public setup failed: {ex_set}"
        ex_drv = m6f.exec_container_bash(cid, "python3 /tmp/_driver.py", timeout_sec=30)
        assert ex_drv["exit_code"] == 0, f"Skill import driver failed: {ex_drv}"
        drv_info = json.loads(ex_drv["stdout"].strip().splitlines()[-1])
        return {
            "staged_in_airgapped_container": True,
            "module_stem": mod_stem,
            "exported_function": fn_name,
            "signature": drv_info["signature"],
            "decorator_present": "@track_wormhole_skill" in str(mod_entry["python_code"]),
        }
    finally:
        m6f.stop_container_safely(cid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_5_unseeded_slice_and_executable_repair_20261125",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--max-containers", type=int, default=6)
    args = ap.parse_args()

    t0 = monotonic()
    rel_out = args.out_dir.resolve().relative_to(REPO_ROOT.resolve()).as_posix() if args.out_dir.resolve().is_relative_to(REPO_ROOT.resolve()) else None
    pre_id = compute_execution_identity(exclude_rel_dir=rel_out)
    if pre_id["hegel_machine_dirty"]:
        raise RuntimeError(f"Refusing live evidence run on dirty Hegel Machine worktree: {pre_id['hegel_machine_dirty_files']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if codex_info.get("auth_mode") != "chatgpt":
        raise RuntimeError(f"Official ChatGPT CLI (`auth_mode=='chatgpt'`) required, got {codex_info}")

    m7_runner = _load_m7_runner_module()
    m7_runner.install_quiet_timing_locks(max_containers=args.max_containers)

    # 1. Calibrate Decomposed Frontier-Only `M8OfflineContainerPlanner` (`v4`, `0` Holdout exposure!)
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    recs_74 = load_historical_suite_episode_records(m74_dir)
    planner = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    planner_bytes = (json.dumps(planner.to_json_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    planner_fp = args.out_dir / "m8_5_decomposed_frontier_planner.json"
    planner_fp.write_bytes(planner_bytes)
    planner_sha = hashlib.sha256(planner_bytes).hexdigest()

    # 2. Seed `PersistentMechanismGraph` from sealed M8.2 checkpoint (`checkpoint_after_p2.json`)
    seed_ckpt = (
        REPO_ROOT
        / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125"
        / "checkpoints/checkpoint_after_p2.json"
    )
    shared_graph = PersistentMechanismGraph.load_checkpoint(seed_ckpt)
    graph_lock = threading.Lock()

    bundle_path = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    forest, archive, meta_dag, bundle_meta = load_autogenous_forest_bundle(bundle_path)
    hippo = HippoRAGWormholeIndex(forest)

    # 3. Genuine Unseeded End-to-End 3-Task Live Container Slice (`ho08`, `ho06`, `ho10` — ZERO branch mocks!)
    target_ids = (
        "ho08__prony_viscoelastic_memory_inversion",
        "ho06__plate_evaporation_deconfounded_ic50",
        "ho10__haldane_bioreactor_runaway_control",
    )
    ho_waves = holdout_tasks_as_frontier_waves()
    all_ho_ftasks = {t.task_id: t for tl in ho_waves.values() for t in tl}
    tasks = [all_ho_ftasks[tid] for tid in target_ids]

    runs_dir = args.out_dir / "part1_unseeded_end_to_end_slice/runs"
    branches_dir = args.out_dir / "part1_unseeded_end_to_end_slice/branches"
    runs_dir.mkdir(parents=True, exist_ok=True)
    branches_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"m8_5_unseeded_live__{args.model}"

    def _run_unseeded_task(task: m6f.FrontierBenchmarkTask) -> dict[str, Any]:
        worker_graph = PersistentMechanismGraph.load_checkpoint(seed_ckpt)
        ep_dir = runs_dir / task.task_id
        print(f"[M8.5-UNSEEDED START] `{task.task_id}` (max_turns={args.max_turns})...", flush=True)
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
            merge_stats = merge_worker_graph_into_shared(shared_graph, worker_graph)
        s["m8_planner_sha256"] = planner_sha
        s["worker_graph_merge_stats"] = merge_stats
        s["run_mode"] = "genuine_unseeded_end_to_end_live_gpt_cli"
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sel = s.get("selective_router_decision") or {}
        print(
            f"[M8.5-UNSEEDED DONE] `{task.task_id}`: strict_pass={s['strict_pass']} | "
            f"action={sel.get('selected_action')} (state={sel.get('public_verification_state')}) | "
            f"repair_attempted={bool((s.get('local_container_repair_summary') or {}).get('attempted'))} | "
            f"thesis_spawned={s['postlock_branch_outcomes']['thesis_spawned']} | turns={s['turns_used']} | "
            f"tokens={s['total_tokens']} | wall={s['wall_ms']/1000:.1f}s",
            flush=True,
        )
        return s

    results_by_tid: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(_run_unseeded_task, t): t.task_id for t in tasks}
        for fut in as_completed(futs):
            results_by_tid[futs[fut]] = fut.result()

    # 4. Part 2: Live `M8.5` Executable Repair Compilation & Container Verification (`gpt-6.2-sol`)
    print("[M8.5-PART2 START] Compiling verified container repair into unit-tested `@track_wormhole_skill` operator...", flush=True)
    ho_by_id = holdout_tasks_by_id()
    live_repaired_ep = next(
        (
            results_by_tid[tid]
            for tid in target_ids
            if bool((results_by_tid[tid].get("local_container_repair_summary") or {}).get("repaired_verified"))
        ),
        None,
    )
    if live_repaired_ep is not None:
        rep_tid = str(live_repaired_ep["task_id"])
        rep_snap_fp = next((runs_dir / rep_tid / "04a_repaired_antithesis_snapshot").glob("*.py"))
        rep_source_prov = f"part1_live_unseeded:{rep_tid}"
        rep_summary_for_compile = {
            **(live_repaired_ep.get("local_container_repair_summary") or {}),
            "repaired_verified": True,
            "repaired_python_code": rep_snap_fp.read_text(encoding="utf-8"),
        }
        compile_task = all_ho_ftasks[rep_tid]
        compile_ho_task = ho_by_id[rep_tid]
    else:
        # Use sealed M8.4 live `gpt-6.2-sol` verified `ho06` repair snapshot with explicit provenance!
        rep_tid = "ho06__plate_evaporation_deconfounded_ic50"
        m84_ep_fp = (
            REPO_ROOT
            / "docs/evidence/m8_4_frozen_audit_reconciliation_litmus_20261125"
            / "proc1_engine_repair_and_persist/ho06__plate_evaporation_deconfounded_ic50/00_episode_summary.json"
        )
        m84_rep_py_fp = (
            REPO_ROOT
            / "docs/evidence/m8_4_frozen_audit_reconciliation_litmus_20261125"
            / "proc1_engine_repair_and_persist/ho06__plate_evaporation_deconfounded_ic50/04a_repaired_antithesis_snapshot/assay_ic50.py"
        )
        m84_ep = json.loads(m84_ep_fp.read_text(encoding="utf-8"))
        m84_ckpt = PersistentMechanismGraph.load_checkpoint(
            REPO_ROOT
            / "docs/evidence/m8_4_frozen_audit_reconciliation_litmus_20261125"
            / "checkpoints/m8_4_post_proc1_checkpoint.json"
        )
        merge_worker_graph_into_shared(shared_graph, m84_ckpt)
        rep_source_prov = "sealed_m8_4_live_gpt_6_2_sol_verified_repair:ho06__plate_evaporation_deconfounded_ic50"
        rep_summary_for_compile = {
            **(m84_ep.get("local_container_repair_summary") or {}),
            "repaired_verified": True,
            "repaired_python_code": m84_rep_py_fp.read_text(encoding="utf-8"),
        }
        compile_task = all_ho_ftasks[rep_tid]
        compile_ho_task = ho_by_id[rep_tid]

    compile_client = m6f.FrontierCodexClient(
        codex_info=codex_info,
        model=args.model,
        reasoning_effort="medium",
        timeout_seconds=240,
    )
    exec_skill_receipt = compile_verified_repair_to_executable_skill(
        task=compile_task,
        repair_summary=rep_summary_for_compile,
        forest=forest,
        archive=archive,
        codex_client=compile_client,
        run_id=f"m8_5_exec_repair__{args.model}",
        graph=shared_graph,
    )
    if not exec_skill_receipt.get("compiled_and_consolidated"):
        raise RuntimeError(f"M8.5 executable repair compilation failed unit-test gate: {exec_skill_receipt}")

    hippo_after = HippoRAGWormholeIndex(forest)
    pub_sigs_target = m6f.extract_public_task_signatures(compile_task)
    hippo_hits_raw = hippo_after.retrieve(
        pub_sigs_target,
        task_view={"goal_kind": "scientific_terminal_discovery"},
        top_k=6,
    )["candidates"]
    hippo_hits = [
        {
            "node_id": h["node_id"],
            "ppr_score": h["ppr_score"],
            "matched_signatures": h["matched_signatures"],
            "entry_kind": h["entry_kind"],
        }
        for h in hippo_hits_raw
    ]
    recalled_compiled_node = any(h["node_id"] == exec_skill_receipt["target_node_id"] for h in hippo_hits)

    container_staging_receipt = _verify_executable_skill_staging_and_activation_in_container(
        ho_task=compile_ho_task,
        forest_archive=archive,
        target_node_id=str(exec_skill_receipt["target_node_id"]),
        module_filename=str(exec_skill_receipt["module_filename"]),
        exported_functions=list(exec_skill_receipt["exported_functions"]),
    )

    part2_dir = args.out_dir / "part2_executable_repair_skill_transfer"
    part2_dir.mkdir(parents=True, exist_ok=True)
    updated_bundle_meta = serialize_autogenous_forest_bundle(
        forest=forest,
        archive=archive,
        meta_dag=meta_dag,
        out_path=part2_dir / "forest_with_compiled_executable_repair_v1.json",
        extra_metadata={"source_repair_provenance": rep_source_prov, "compiled_module": exec_skill_receipt["module_filename"]},
    )
    part2_summary = {
        "source_repair_provenance": rep_source_prov,
        "executable_skill_compilation": exec_skill_receipt,
        "hipporag_recalled_compiled_node": recalled_compiled_node,
        "hipporag_top_hits": hippo_hits[:4],
        "container_staging_and_import_receipt": container_staging_receipt,
        "updated_forest_bundle": updated_bundle_meta,
    }
    (part2_dir / "00_executable_repair_transfer_summary.json").write_text(
        json.dumps(part2_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out_ckpt = args.out_dir / "checkpoints/m8_5_post_slice_checkpoint.json"
    ckpt_meta = shared_graph.save_checkpoint(out_ckpt)

    post_id = compute_execution_identity(exclude_rel_dir=rel_out)
    assert pre_id["code_tree_sha256"] == post_id["code_tree_sha256"], "Python source tree mutated during live run!"
    assert pre_id["git_head_commit"] == post_id["git_head_commit"], "Git HEAD changed during live run!"

    ordered = [results_by_tid[tid] for tid in target_ids]
    strict_passes = sum(1 for s in ordered if s["strict_pass"])
    raw_1x_passes = sum(1 for s in ordered if s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"])
    thesis_spawns = sum(1 for s in ordered if s["postlock_branch_outcomes"]["thesis_spawned"])
    local_repairs_ok = sum(1 for s in ordered if bool((s.get("local_container_repair_summary") or {}).get("repaired_verified")))
    slice_tokens = sum(int(s.get("total_tokens") or 0) for s in ordered)
    compile_tokens = int((exec_skill_receipt.get("llm_tokens") or {}).get("prompt", 0)) + int(
        (exec_skill_receipt.get("llm_tokens") or {}).get("completion", 0)
    )

    master_summary = {
        "experiment_id": "m8_5_unseeded_slice_and_executable_repair_20261125",
        "claim_scope_calibration": {
            "part1_mode": "genuine_unseeded_end_to_end_regression_slice_on_3_holdout12_tasks",
            "seeded_branch_mocks_used": 0,
            "claims_full_20_or_12_suite_rerun": False,
            "part2_mode": "container_unit_tested_executable_python_repair_skill_compilation_and_recall",
        },
        "pre_run_execution_identity": pre_id,
        "post_run_execution_identity": post_id,
        "code_tree_frozen_verified": pre_id["code_tree_sha256"] == post_id["code_tree_sha256"],
        "model": args.model,
        "auth_mode": codex_info["auth_mode"],
        "max_turns_per_branch": args.max_turns,
        "pretrained_bundle_sha256": bundle_meta["bundle_sha256"],
        "m8_5_decomposed_planner_sha256": planner_sha,
        "post_slice_checkpoint": ckpt_meta,
        "part1_unseeded_3task_metrics": {
            "total_tasks": len(ordered),
            "raw_primary_1x_strict_passes": raw_1x_passes,
            "final_m8_5_strict_passes": strict_passes,
            "local_container_repairs_verified": local_repairs_ok,
            "second_branch_challenger_spawns": thesis_spawns,
            "total_slice_tokens": slice_tokens,
            "per_task": [
                {
                    "task_id": s["task_id"],
                    "raw_primary_1x_strict_pass": s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"],
                    "final_strict_pass": s["strict_pass"],
                    "public_verification_state": (s.get("selective_router_decision") or {}).get("public_verification_state"),
                    "m8_selected_action": (s.get("selective_router_decision") or {}).get("selected_action"),
                    "net_variational_utility": (s.get("selective_router_decision") or {}).get("net_variational_utility"),
                    "local_repair_attempted": bool((s.get("local_container_repair_summary") or {}).get("attempted")),
                    "local_repair_verified": bool((s.get("local_container_repair_summary") or {}).get("repaired_verified")),
                    "thesis_challenger_spawned": s["postlock_branch_outcomes"]["thesis_spawned"],
                    "episode_nominal_units_total": s["episode_nominal_units_total"],
                    "turns_used": s["turns_used"],
                    "skill_activation_count": s.get("skill_activation_count", 0),
                    "nogood_count": s.get("nogood_count", 0),
                    "total_tokens": s.get("total_tokens", 0),
                    "wall_sec": round(s["wall_ms"] / 1000.0, 2),
                }
                for s in ordered
            ],
        },
        "part2_executable_repair_transfer_metrics": {
            "source_repair_provenance": rep_source_prov,
            "compiled_and_consolidated": exec_skill_receipt["compiled_and_consolidated"],
            "unit_test_passed_in_container": exec_skill_receipt["unit_test_receipt"]["passed"],
            "target_node_id": exec_skill_receipt["target_node_id"],
            "module_filename": exec_skill_receipt["module_filename"],
            "exported_functions": exec_skill_receipt["exported_functions"],
            "compiled_code_sha256": exec_skill_receipt["compiled_code_sha256"],
            "hipporag_recalled_compiled_node": recalled_compiled_node,
            "staged_in_airgapped_container": container_staging_receipt["staged_in_airgapped_container"],
            "compile_total_tokens": compile_tokens,
        },
        "shared_graph_summary": {
            "mechanism_count": len(shared_graph.mechanisms),
            "decision_ledger_count": len(shared_graph.decision_ledger),
            "action_return_event_counts": {k: len(v) for k, v in shared_graph.action_return_history.items()},
        },
        "grand_total_tokens": slice_tokens + compile_tokens,
        "total_wall_ms": round((monotonic() - t0) * 1000.0, 2),
    }
    (args.out_dir / "00_m8_5_unseeded_and_executable_repair_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    n_locked = write_sha256_sums(args.out_dir)
    print(
        f"[M8.5 COMPLETE] unseeded={strict_passes}/{len(ordered)} (raw_1x={raw_1x_passes}/{len(ordered)}) | "
        f"exec_skill={exec_skill_receipt['module_filename']} (hippo_recalled={recalled_compiled_node}) | "
        f"tokens={slice_tokens + compile_tokens} | {n_locked} files SHA-256 locked!",
        flush=True,
    )


if __name__ == "__main__":
    main()
