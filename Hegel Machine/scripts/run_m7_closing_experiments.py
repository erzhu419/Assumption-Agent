#!/usr/bin/env python3
"""Hegel M7.4–M7.6 Unified Runner for the 3 Pre-Registered Paper-Closing Experiments.

All 3 experiments execute on a SINGLE FROZEN GIT COMMIT with `pretrained_autogenous_forest_v2.json`
(`100%` blank-seed machine-grown, `0` hand-crafted `SKILL_01..07`, `0` `node_F01..F12`, official
OpenAI ChatGPT OAuth `auth_mode == "chatgpt"`):

  1. `exp1_frontier_20task`:
     Single-commit 20-task `Terminal-Bench-Science` + `BioMysteryBench` evaluation & causal ablation matrix:
     - `Bare_ReAct_1x` (`Thesis` 1x + independent `Bare_Run2_1x` -> `Bare_BestOf2_2x` & `Public_Synthesis_Bare2x`),
     - `Autogenous_Forest_1x` (`Antithesis` 1x, `0` human skills),
     - `Autogenous_Forest_Selective_1.3x` & `Autogenous_Forest_TwoBranch_2x` (with blind Wave-1 -> Wave-2 `AutonomousAufhebungCompiler`),
     - `Autogenous_Forest_NoWormhole_1x` (`__disable_wormholes__ = True` horizontal wormhole ablation).
  2. `exp2_holdout_ood_12task`:
     One-shot blind evaluation (`zero` post-hoc edits/tuning!) on pre-registered `Hegel-OOD-Holdout-12`
     (`ho01..ho06` Wave-1 -> blind online `AutonomousAufhebungCompiler` -> `ho07..ho12` Wave-2).
  3. `exp3_cross_model`:
     Zero-shot cross-model transfer mounting `pretrained_autogenous_forest_v2.json` (grown by `gpt-6.2-sol`)
     onto a distinct second base model (`--model <second_model>`) head-to-head against `Bare_ReAct_1x (<second_model>)`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
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
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    HeavyTimingRWLock,
    verify_candidate_artifact_contract_in_container,
)
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import (
    _init_holdout_public_env_if_needed,
    _run_public_container_candidate_benchmark,
    reconcile_thesis_and_antithesis_publicly,
    run_m6_9_a_two_branch_episode,
    should_spawn_dialectical_second_branch,
)
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    consolidate_wave1_via_aufhebung_compiler,
    load_autogenous_forest_bundle,
    serialize_autogenous_forest_bundle,
)
from hegel_machine.cognition import m7_holdout_ood_blind_benchmark as m7_ho
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    compute_holdout_suite_preregistration_sha256,
    holdout_tasks_as_frontier_waves,
)


def install_quiet_timing_locks(*, max_containers: int = 8) -> None:
    """Protect `amr-poisson-optimize` (`verifier_memory_mb >= 10000`) from CPU noise AND cap concurrent containers.

    Decouples high LLM worker concurrency (`--max-workers 16`) from host RAM container pressure (`--max-containers`)
    so 16-thread runs do not OOM a 43 GiB host with 32 simultaneous heavy Docker containers!
    Wraps BOTH `run_tb_separate_verifier` AND `run_holdout_private_verifier` with `in_verifier_tls.active = True`
    so nested verifier containers never deadlock against `container_sem` (`M7.7` `holdout_nested_container` fix).
    """
    rw_lock = HeavyTimingRWLock()
    cap_c = max(1, int(max_containers))
    verifier_sem = threading.Semaphore(min(4, max(1, (cap_c + 1) // 2)))
    container_sem = threading.Semaphore(cap_c)
    in_verifier_tls = threading.local()
    active_cids: set[str] = set()
    active_cids_lock = threading.Lock()
    orig_exec_bash = m6f.exec_container_bash
    orig_tb_verifier = m6f.run_tb_separate_verifier
    orig_ho_verifier = m7_ho.run_holdout_private_verifier
    orig_start_container = m6f.start_detached_container
    orig_stop_container = m6f.stop_container_safely

    def _capped_start_container(*args: Any, **kwargs: Any) -> str:
        # Deadlock prevention: `run_tb_separate_verifier` & `run_holdout_private_verifier` boot a short-lived
        # `verifier_cid` WHILE the caller already holds `agent_cid`! Bypass `container_sem` inside verifiers.
        if getattr(in_verifier_tls, "active", False):
            return orig_start_container(*args, **kwargs)
        container_sem.acquire()
        try:
            cid = orig_start_container(*args, **kwargs)
        except Exception:
            container_sem.release()
            raise
        with active_cids_lock:
            active_cids.add(cid)
        return cid

    def _capped_stop_container(cid: str) -> None:
        try:
            orig_stop_container(cid)
        finally:
            release = False
            with active_cids_lock:
                if cid in active_cids:
                    active_cids.remove(cid)
                    release = True
            if release:
                container_sem.release()

    def _quiet_exec_bash(cid: str, command: str, *, timeout_sec: int = 120) -> dict[str, Any]:
        rw_lock.acquire_read()
        try:
            return orig_exec_bash(cid, command, timeout_sec=timeout_sec)
        finally:
            rw_lock.release_read()

    def _quiet_tb_verifier(task: m6f.FrontierBenchmarkTask, agent_cid: str) -> dict[str, Any]:
        in_verifier_tls.active = True
        try:
            if task.verifier_memory_mb >= 10000:
                rw_lock.acquire_write()
                try:
                    return orig_tb_verifier(task, agent_cid)
                finally:
                    rw_lock.release_write()
            with verifier_sem:
                rw_lock.acquire_read()
                try:
                    return orig_tb_verifier(task, agent_cid)
                finally:
                    rw_lock.release_read()
        finally:
            in_verifier_tls.active = False

    def _quiet_ho_verifier(task: m7_ho.HoldoutOODTask, submitted_tar_path: Path) -> dict[str, Any]:
        in_verifier_tls.active = True
        try:
            with verifier_sem:
                rw_lock.acquire_read()
                try:
                    return orig_ho_verifier(task, submitted_tar_path)
                finally:
                    rw_lock.release_read()
        finally:
            in_verifier_tls.active = False

    m6f.start_detached_container = _capped_start_container  # type: ignore[assignment]
    m6f.stop_container_safely = _capped_stop_container  # type: ignore[assignment]
    m6f.exec_container_bash = _quiet_exec_bash  # type: ignore[assignment]
    m6f.run_tb_separate_verifier = _quiet_tb_verifier  # type: ignore[assignment]
    m7_ho.run_holdout_private_verifier = _quiet_ho_verifier  # type: ignore[assignment]


def get_git_head_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def episode_summary_matches_suite_cache(
    s: dict[str, Any],
    *,
    expected_model: str,
    suite_cfg_fp: str,
    selective_online: bool,
) -> bool:
    """Strict fingerprint & selective-mode cache match (`M8.2` Fix: rejects un-fingerprinted legacy summaries!)."""
    return bool(
        s.get("model") == expected_model
        and isinstance(s.get("suite_config_fingerprint"), str)
        and s.get("suite_config_fingerprint") == suite_cfg_fp
        and bool(
            s["selective_second_branch"]
            if "selective_second_branch" in s
            else (s.get("selective_router_decision") is not None)
        )
        == bool(selective_online)
    )


def write_sha256_manifest(evidence_root: Path) -> int:
    lines: list[str] = []
    for fp in sorted(evidence_root.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(evidence_root).as_posix()
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
            lines.append(f"{digest}  {rel}")
    (evidence_root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def evaluate_selective_router_from_two_branch_summary(
    task: m6f.FrontierBenchmarkTask,
    ep_sum: dict[str, Any],
    branches_root: Path,
    model: str,
) -> dict[str, Any]:
    """Compute exact `Selective (~1.3x)` outcome (`Antithesis` if router says 1-branch, `Synthesis` if router spawns 2nd branch)."""
    anti_dir = branches_root / f"{task.wave}__{model}__{task.task_id}" / "branch_2_antithesis_wormhole"
    anti_sum_fp = anti_dir / "00_episode_summary.json"
    if not anti_sum_fp.is_file():
        return {"spawn_second_branch": True, "selective_strict_pass": bool(ep_sum["strict_pass"]), "effective_turns": ep_sum["turns_used"]}
    anti_sum = json.loads(anti_sum_fp.read_text(encoding="utf-8"))
    # Extract public metrics already computed in `D3_DIALECTICAL_SYNTHESIS_COMMIT` if available
    pub_a: dict[str, Any] = {}
    dag_fp = (branches_root.parent / "runs" / task.task_id / "01_decision_provenance_dag.json")
    if dag_fp.is_file():
        dag = json.loads(dag_fp.read_text(encoding="utf-8"))
        for rec in dag.get("records", []):
            if rec.get("decision_type") == "D3_DIALECTICAL_SYNTHESIS_COMMIT":
                for cand in rec.get("candidates_evaluated", []):
                    if cand.get("branch") == "antithesis" and isinstance(cand.get("public_metrics"), dict):
                        pub_a = cand["public_metrics"]
    chk_ok = bool(anti_sum.get("final_candidate_sha256")) or task.benchmark == "BioMysteryBench-Preview"
    dec = should_spawn_dialectical_second_branch(
        task=task,
        primary_summary=anti_sum,
        contract_check={"accepted": chk_ok},
        public_metrics=pub_a,
    )
    spawn = bool(dec["spawn_second_branch"])
    anti_pass = bool(ep_sum["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"])
    sel_pass = bool(ep_sum["strict_pass"]) if spawn else anti_pass
    eff_turns = int(ep_sum["turns_used"]) if spawn else int(ep_sum["postlock_branch_outcomes"]["antithesis_turns_used"])
    return {
        "spawn_second_branch": spawn,
        "triggers": dec["triggers"],
        "selective_strict_pass": sel_pass,
        "effective_turns": eff_turns,
    }


def run_two_wave_autogenous_suite(
    *,
    waves: dict[str, list[m6f.FrontierBenchmarkTask]],
    suite_dir: Path,
    pretrained_bundle_path: Path,
    model: str,
    codex_info: dict[str, Any],
    max_turns: int,
    max_workers: int,
    cmd_timeout_sec: int,
    reasoning_effort: str,
    online_wave1_aufhebung: bool = True,
    selective_online: bool = False,
    task_filter: set[str] | None = None,
) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = suite_dir / "runs"
    branches_root = suite_dir / "branches"
    runs_dir.mkdir(parents=True, exist_ok=True)
    branches_root.mkdir(parents=True, exist_ok=True)

    forest, archive, meta_dag, bundle_meta = load_autogenous_forest_bundle(pretrained_bundle_path)
    run_id = f"{suite_dir.name}__{model}"
    code_tree_hasher = hashlib.sha256()
    for py_fp in sorted((REPO_ROOT / "src/hegel_machine/cognition").glob("*.py")):
        code_tree_hasher.update(py_fp.name.encode("utf-8"))
        code_tree_hasher.update(py_fp.read_bytes())
    code_tree_sha16 = code_tree_hasher.hexdigest()[:16]
    suite_cfg_fp = hashlib.sha256(
        json.dumps(
            {
                "code_tree_sha16": code_tree_sha16,
                "model": model,
                "bundle_sha256": bundle_meta["bundle_sha256"],
                "max_turns": int(max_turns),
                "cmd_timeout_sec": int(cmd_timeout_sec),
                "reasoning_effort": str(reasoning_effort),
                "online_wave1_aufhebung": bool(online_wave1_aufhebung),
                "selective_online": bool(selective_online),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]

    def _run_wave(wave_name: str, cur_forest: Any, cur_archive: dict[str, Any]) -> list[dict[str, Any]]:
        w_tasks = [t for t in waves[wave_name] if (not task_filter or t.task_id in task_filter)]
        if not w_tasks:
            return []
        hippo = HippoRAGWormholeIndex(cur_forest)
        results_by_tid: dict[str, dict[str, Any]] = {}

        def _do_one(task: m6f.FrontierBenchmarkTask) -> dict[str, Any]:
            ep_dir = runs_dir / task.task_id
            sum_fp = ep_dir / "00_episode_summary.json"
            if sum_fp.is_file():
                s = json.loads(sum_fp.read_text(encoding="utf-8"))
                cfg_matches = episode_summary_matches_suite_cache(
                    s,
                    expected_model=model,
                    suite_cfg_fp=suite_cfg_fp,
                    selective_online=selective_online,
                )
                if cfg_matches:
                    print(f"[SKIP-DONE] `{task.task_id}`: SYNTH={s['strict_pass']} | {s['postlock_branch_outcomes']}", flush=True)
                    return s
                print(f"[RE-RUN CONFIG-MISMATCH] `{task.task_id}` (cached fingerprint != `{suite_cfg_fp}`)...", flush=True)
            print(f"[START {wave_name}] `{task.task_id}` ({model})...", flush=True)
            s = run_m6_9_a_two_branch_episode(
                task=task,
                model=model,
                codex_info=codex_info,
                forest=cur_forest,
                forest_archive=cur_archive,
                hippo_idx=hippo,
                episode_out_dir=ep_dir,
                branches_work_root=branches_root,
                run_id=run_id,
                max_turns_per_branch=max_turns,
                cmd_timeout_sec=cmd_timeout_sec,
                reasoning_effort=reasoning_effort,
                selective_second_branch=selective_online,
            )
            s["suite_config_fingerprint"] = suite_cfg_fp
            sum_fp.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                f"[DONE {wave_name}] `{task.task_id}`: SYNTH={s['strict_pass']} | "
                f"Thesis(Bare1x)={s['postlock_branch_outcomes']['thesis_bare_run2_strict_pass']} | "
                f"Anti(Forest1x)={s['postlock_branch_outcomes']['antithesis_wormhole_strict_pass']} | "
                f"SkillCalls={s.get('skill_activation_count', 0)}",
                flush=True,
            )
            return s

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_do_one, t): t.task_id for t in w_tasks}
            for fut in as_completed(futs):
                tid = futs[fut]
                results_by_tid[tid] = fut.result()
        return [results_by_tid[t.task_id] for t in w_tasks]

    w1_summaries = _run_wave("wave_1", forest, archive)

    w1_cons: dict[str, Any] = {"skipped": True}
    post_w1_fp = suite_dir / "post_wave1_autogenous_forest_bundle.json"
    w1_cons_fp = suite_dir / "wave1_aufhebung_consolidation.json"
    if online_wave1_aufhebung and w1_summaries:
        if post_w1_fp.is_file() and w1_cons_fp.is_file():
            w1_cons = json.loads(w1_cons_fp.read_text(encoding="utf-8"))
            forest, archive, meta_dag, _ = load_autogenous_forest_bundle(post_w1_fp)
        else:
            w1_anti_records: list[tuple[m6f.FrontierBenchmarkTask, dict[str, Any], Path]] = []
            w1_by_id = {t.task_id: t for t in waves["wave_1"]}
            for s in w1_summaries:
                t_obj = w1_by_id[s["task_id"]]
                anti_d = branches_root / f"{t_obj.wave}__{model}__{t_obj.task_id}" / "branch_2_antithesis_wormhole"
                anti_sfp = anti_d / "00_episode_summary.json"
                if anti_sfp.is_file():
                    w1_anti_records.append((t_obj, json.loads(anti_sfp.read_text(encoding="utf-8")), anti_d))
            print(f"[AUFHEBUNG] Compiling blind Wave-1 -> Wave-2 sublations across {len(w1_anti_records)} episodes...", flush=True)
            w1_cons = consolidate_wave1_via_aufhebung_compiler(
                wave1_antithesis_episodes=w1_anti_records,
                forest=forest,
                archive=archive,
                codex_info=codex_info,
                model=model,
                run_id=run_id,
                reasoning_effort=reasoning_effort,
            )
            w1_cons_fp.write_text(json.dumps(w1_cons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            serialize_autogenous_forest_bundle(
                forest=forest,
                archive=archive,
                meta_dag=meta_dag,
                out_path=post_w1_fp,
                extra_metadata={"stage": "post_wave1_online_aufhebung", "git_commit": get_git_head_sha()},
            )

    w2_summaries = _run_wave("wave_2", forest, archive)
    all_summaries = [*w1_summaries, *w2_summaries]
    all_tasks_by_id = {t.task_id: t for tl in waves.values() for t in tl}

    selective_rows: list[dict[str, Any]] = []
    for s in all_summaries:
        t_obj = all_tasks_by_id[s["task_id"]]
        sel = evaluate_selective_router_from_two_branch_summary(t_obj, s, branches_root, model)
        selective_rows.append({"task_id": s["task_id"], **sel})

    n_tot = len(all_summaries)
    any_bare_not_run = any(
        s["postlock_branch_outcomes"].get("thesis_bare_run2_strict_pass") is None for s in all_summaries
    )
    bare_1x_passes = (
        None
        if any_bare_not_run
        else sum(1 for s in all_summaries if s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"])
    )
    anti_1x_passes = sum(1 for s in all_summaries if s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"])
    sel_passes = sum(1 for r in selective_rows if r["selective_strict_pass"])
    twobranch_passes = sum(1 for s in all_summaries if s["strict_pass"])
    bare_turns = sum(int(s["postlock_branch_outcomes"]["thesis_turns_used"]) for s in all_summaries)
    sel_turns = sum(int(r["effective_turns"]) for r in selective_rows)

    agg = {
        "git_commit": get_git_head_sha(),
        "model": model,
        "auth_mode": codex_info["auth_mode"],
        "pretrained_bundle_sha256": bundle_meta["bundle_sha256"],
        "suite_config_fingerprint": suite_cfg_fp,
        "handcrafted_domain_skills_mounted": 0,
        "total_tasks": n_tot,
        "bare_react_1x_status": "NOT_EVALUATED_IN_SELECTIVE_ONLINE_MODE" if any_bare_not_run else "EVALUATED",
        "bare_react_1x_passes": bare_1x_passes,
        "autogenous_forest_1x_passes": anti_1x_passes,
        "autogenous_forest_selective_passes": sel_passes,
        "autogenous_forest_twobranch_2x_passes": twobranch_passes,
        "selective_second_branch_spawn_rate": round(
            sum(1 for r in selective_rows if r["spawn_second_branch"]) / max(n_tot, 1), 4
        ),
        "selective_turn_multiplier_vs_bare_1x": (
            None if (any_bare_not_run or bare_turns <= 0) else round(sel_turns / bare_turns, 4)
        ),
        "total_autogenous_skill_calls": sum(int(s.get("skill_activation_count", 0)) for s in all_summaries),
        "wave1_aufhebung_consolidation": w1_cons,
        "per_task": [
            {
                "task_id": s["task_id"],
                "wave": s["wave"],
                "bare_1x_pass": s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"],
                "autogenous_1x_pass": s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"],
                "selective_pass": sel["selective_strict_pass"],
                "selective_spawned_2nd": sel["spawn_second_branch"],
                "twobranch_2x_pass": s["strict_pass"],
                "selected_branch": s["synthesis_decision"]["selected_branch"],
                "selection_mechanism": s["synthesis_decision"]["selection_mechanism"],
                "skill_calls": s.get("skill_activation_count", 0),
                "staged_modules": [m["module_import"] for m in s.get("staged_wormhole_modules", [])],
            }
            for s, sel in zip(all_summaries, selective_rows)
        ],
    }
    (suite_dir / "00_suite_summary.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return agg


def run_wormhole_ablation_and_bare_bestof2(
    *,
    waves: dict[str, list[m6f.FrontierBenchmarkTask]],
    suite_dir: Path,
    initial_bundle_path: Path,
    model: str,
    codex_info: dict[str, Any],
    max_turns: int,
    max_workers: int,
    cmd_timeout_sec: int,
    reasoning_effort: str,
    run_second_bare: bool = True,
    wormhole_off_mode: str = "frozen_snapshot",
    wormhole_ablation_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run unconfounded `Wormhole OFF` 1x ablation (`frozen_snapshot` or `independent_continual`) + `Bare_Run2_1x`."""
    abl_dir = suite_dir / "ablations"
    abl_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = [t for w in ("wave_1", "wave_2") for t in waves[w]]
    tasks_by_id = {t.task_id: t for t in all_tasks}

    # A. Independent Bare_Run2_1x across all 20 tasks -> clean single-commit `Bare Best-of-2 (2x)`!
    bare2_results: dict[str, dict[str, Any]] = {}
    if run_second_bare:
        b2_root = abl_dir / "bare_run2_1x"
        b2_root.mkdir(parents=True, exist_ok=True)
        empty_f, empty_a, _ = m6f.build_m6_frontier_general_forest()
        empty_h = HippoRAGWormholeIndex(empty_f)

        def _do_bare2(task: m6f.FrontierBenchmarkTask) -> dict[str, Any]:
            ep_d = b2_root / task.task_id
            sfp = ep_d / "00_episode_summary.json"
            if sfp.is_file():
                return json.loads(sfp.read_text(encoding="utf-8"))
            print(f"[BARE-RUN2] Starting `{task.task_id}`...", flush=True)
            s = m6f.run_frontier_episode(
                task,
                "Bare_ReAct_LiveGPT",
                model=model,
                codex_info=codex_info,
                forest=empty_f,
                forest_archive=empty_a,
                hippo_idx=empty_h,
                episode_out_dir=ep_d,
                run_id=f"{suite_dir.name}_bare_run2",
                max_turns=max_turns,
                cmd_timeout_sec=cmd_timeout_sec,
                reasoning_effort=reasoning_effort,
            )
            print(f"[BARE-RUN2] Finished `{task.task_id}`: pass={s['strict_pass']}", flush=True)
            return s

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_do_bare2, t): t.task_id for t in all_tasks}
            for fut in as_completed(futs):
                bare2_results[futs[fut]] = fut.result()

    # B. Unconfounded Horizontal Wormhole OFF (`allow_horizontal_wormholes=False, wormhole_equivalence_pooling=False`)
    #   - `frozen_snapshot`: All target tasks run on the EXACT same frozen `initial_bundle_path` (`0` post-Wave-1 contamination).
    #   - `independent_continual`: Clones `initial_bundle_path`, runs Wave-1 OFF -> vertical-only Aufhebung (`0` horizontal edges) -> Wave-2 OFF!
    forest_nw, archive_nw, meta_dag_nw, _ = load_autogenous_forest_bundle(initial_bundle_path)
    archive_nw["__disable_wormholes__"] = {"enabled": True}
    target_set = set(wormhole_ablation_task_ids) if wormhole_ablation_task_ids else {
        t.task_id for t in all_tasks if t.benchmark != "BioMysteryBench-Preview"
    }
    target_ids = [t.task_id for t in all_tasks if t.task_id in target_set]
    nw_root = abl_dir / f"wormhole_off_{wormhole_off_mode}_1x"
    nw_root.mkdir(parents=True, exist_ok=True)
    nw_results: dict[str, dict[str, Any]] = {}

    def _run_nw_wave(wave_tids: list[str], cur_f: Any, cur_a: dict[str, Any]) -> None:
        if not wave_tids:
            return
        cur_h = HippoRAGWormholeIndex(
            cur_f,
            allow_horizontal_wormholes=False,
            wormhole_equivalence_pooling=False,
        )
        assert cur_h.edge_counts.get("horizontal_wormhole_edges", 0) == 0, (
            f"Wormhole OFF violation: expected 0 horizontal edges, got {cur_h.edge_counts}"
        )

        def _do_nw(tid: str) -> dict[str, Any]:
            task = tasks_by_id[tid]
            ep_d = nw_root / tid
            sfp = ep_d / "00_episode_summary.json"
            if sfp.is_file():
                return json.loads(sfp.read_text(encoding="utf-8"))
            print(f"[WORMHOLE-OFF:{wormhole_off_mode}] Starting `{tid}`...", flush=True)
            s = m6f.run_frontier_episode(
                task,
                "Hegel_M6_9_Antithesis_LiveGPT",
                model=model,
                codex_info=codex_info,
                forest=cur_f,
                forest_archive=cur_a,
                hippo_idx=cur_h,
                episode_out_dir=ep_d,
                run_id=f"{suite_dir.name}_wormhole_off_{wormhole_off_mode}",
                max_turns=max_turns,
                cmd_timeout_sec=cmd_timeout_sec,
                reasoning_effort=reasoning_effort,
            )
            print(
                f"[WORMHOLE-OFF:{wormhole_off_mode}] Finished `{tid}`: pass={s['strict_pass']} | recalled={s.get('recalled_node_ids')} | calls={s.get('skill_activation_count', 0)}",
                flush=True,
            )
            return s

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_do_nw, tid): tid for tid in wave_tids}
            for fut in as_completed(futs):
                nw_results[futs[fut]] = fut.result()

    w1_tids = [t.task_id for t in waves["wave_1"] if t.task_id in target_set]
    w2_tids = [t.task_id for t in waves["wave_2"] if t.task_id in target_set]
    nw_w1_cons: dict[str, Any] = {"skipped": True}
    if wormhole_off_mode == "independent_continual":
        _run_nw_wave(w1_tids, forest_nw, archive_nw)
        w1_recs = [
            (tasks_by_id[tid], nw_results[tid], nw_root / tid)
            for tid in w1_tids
            if tid in nw_results
        ]
        if w1_recs:
            nw_w1_cons = consolidate_wave1_via_aufhebung_compiler(
                wave1_antithesis_episodes=w1_recs,
                forest=forest_nw,
                archive=archive_nw,
                codex_info=codex_info,
                model=model,
                run_id=f"{suite_dir.name}_wormhole_off_continual",
                reasoning_effort=reasoning_effort,
                allow_horizontal_wormholes=False,
            )
            serialize_autogenous_forest_bundle(
                forest=forest_nw,
                archive=archive_nw,
                meta_dag=meta_dag_nw,
                out_path=nw_root / "post_wave1_wormhole_off_bundle.json",
                extra_metadata={"stage": "post_wave1_wormhole_off_independent", "git_commit": get_git_head_sha()},
            )
        _run_nw_wave(w2_tids, forest_nw, archive_nw)
    else:
        _run_nw_wave(target_ids, forest_nw, archive_nw)

    out = {
        "git_commit": get_git_head_sha(),
        "wormhole_off_mode": wormhole_off_mode,
        "wormhole_off_wave1_consolidation": nw_w1_cons,
        "bare_run2_count": len(bare2_results),
        "bare_run2_passes": sum(1 for s in bare2_results.values() if s["strict_pass"]),
        "wormhole_off_count": len(nw_results),
        "wormhole_off_passes": sum(1 for s in nw_results.values() if s["strict_pass"]),
        "wormhole_off_per_task": {
            tid: {
                "strict_pass": nw_results[tid]["strict_pass"],
                "recalled_node_ids": nw_results[tid].get("recalled_node_ids", []),
                "staged_modules": [m["module_import"] for m in nw_results[tid].get("staged_wormhole_modules", [])],
                "skill_calls": nw_results[tid].get("skill_activation_count", 0),
            }
            for tid in target_ids
        },
        "bare_run2_per_task": {tid: bool(s["strict_pass"]) for tid, s in bare2_results.items()},
    }
    (abl_dir / f"00_ablation_{wormhole_off_mode}_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        choices=("exp1_frontier_20task", "exp1_ablations", "exp2_holdout_ood_12task", "exp3_cross_model"),
        required=True,
    )
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument(
        "--pretrained-bundle",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--max-turns", type=int, default=28)
    ap.add_argument("--max-workers", type=int, default=16, help="Task/LLM worker concurrency (decoupled from container cap).")
    ap.add_argument("--max-containers", type=int, default=8, help="Hard cap on simultaneous live Docker containers to prevent host OOM.")
    ap.add_argument("--cmd-timeout", type=int, default=180)
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--skip-bare-run2", action="store_true")
    ap.add_argument("--wormhole-off-mode", choices=("frozen_snapshot", "independent_continual"), default="frozen_snapshot")
    ap.add_argument("--selective-online", action="store_true", help="Execute true online selective 2nd-branch routing instead of full 2-branch.")
    ap.add_argument("--freeze-cross-model", action="store_true", help="Disable online Wave-1 Aufhebung during cross-model evaluation.")
    args = ap.parse_args()

    install_quiet_timing_locks(max_containers=args.max_containers)
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("auth_mode") == "chatgpt", f"Official ChatGPT OAuth required: {codex_info}"

    ev_root = args.evidence_dir.resolve()
    ev_root.mkdir(parents=True, exist_ok=True)
    t_filter = set(args.tasks) if args.tasks else None

    if args.exp == "exp1_frontier_20task":
        waves = m6f.load_frontier_scout_20_tasks()
        rep = run_two_wave_autogenous_suite(
            waves=waves,
            suite_dir=ev_root,
            pretrained_bundle_path=args.pretrained_bundle.resolve(),
            model=args.model,
            codex_info=codex_info,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            cmd_timeout_sec=args.cmd_timeout,
            reasoning_effort=args.reasoning_effort,
            online_wave1_aufhebung=True,
            selective_online=args.selective_online,
            task_filter=t_filter,
        )
        print("EXP1_MAIN_DONE:", json.dumps({k: v for k, v in rep.items() if k != "per_task"}, ensure_ascii=False), flush=True)

    elif args.exp == "exp1_ablations":
        waves = m6f.load_frontier_scout_20_tasks()
        # M7-Audit Fix: NEVER pass post_wave1_autogenous_forest_bundle.json to Wave-1 ablation tasks (eliminates self-episode contamination)!
        bundle_use = args.pretrained_bundle.resolve()
        abl = run_wormhole_ablation_and_bare_bestof2(
            waves=waves,
            suite_dir=ev_root,
            initial_bundle_path=bundle_use,
            model=args.model,
            codex_info=codex_info,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            cmd_timeout_sec=args.cmd_timeout,
            reasoning_effort=args.reasoning_effort,
            run_second_bare=(not args.skip_bare_run2),
            wormhole_off_mode=args.wormhole_off_mode,
            wormhole_ablation_task_ids=list(t_filter) if t_filter else None,
        )
        print("EXP1_ABLATIONS_DONE:", json.dumps(abl, ensure_ascii=False), flush=True)

    elif args.exp == "exp2_holdout_ood_12task":
        prereg_sha = compute_holdout_suite_preregistration_sha256()
        (ev_root / "preregistration_lock.json").write_text(
            json.dumps(
                {
                    "benchmark": "Hegel-OOD-Holdout-12",
                    "holdout_suite_sha256": prereg_sha,
                    "frozen_git_commit": get_git_head_sha(),
                    "pretrained_bundle_path": str(args.pretrained_bundle),
                    "pretrained_bundle_file_sha256": hashlib.sha256(args.pretrained_bundle.read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        waves = holdout_tasks_as_frontier_waves()
        rep = run_two_wave_autogenous_suite(
            waves=waves,
            suite_dir=ev_root,
            pretrained_bundle_path=args.pretrained_bundle.resolve(),
            model=args.model,
            codex_info=codex_info,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            cmd_timeout_sec=args.cmd_timeout,
            reasoning_effort=args.reasoning_effort,
            online_wave1_aufhebung=True,
            selective_online=args.selective_online,
            task_filter=t_filter,
        )
        print("EXP2_HOLDOUT_DONE:", json.dumps({k: v for k, v in rep.items() if k != "per_task"}, ensure_ascii=False), flush=True)

    elif args.exp == "exp3_cross_model":
        # Support evaluating cross-model transfer on either Holdout-OOD-12 or Frontier-20, frozen or continual!
        use_holdout = bool(t_filter and all(tid.startswith("ho") for tid in t_filter))
        waves = holdout_tasks_as_frontier_waves() if use_holdout else m6f.load_frontier_scout_20_tasks()
        rep = run_two_wave_autogenous_suite(
            waves=waves,
            suite_dir=ev_root,
            pretrained_bundle_path=args.pretrained_bundle.resolve(),
            model=args.model,
            codex_info=codex_info,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            cmd_timeout_sec=args.cmd_timeout,
            reasoning_effort=args.reasoning_effort,
            online_wave1_aufhebung=(not args.freeze_cross_model),
            selective_online=args.selective_online,
            task_filter=t_filter,
        )
        print("EXP3_CROSS_MODEL_DONE:", json.dumps({k: v for k, v in rep.items() if k != "per_task"}, ensure_ascii=False), flush=True)

    cnt = write_sha256_manifest(ev_root)
    print(f"LOCKED_SHA256SUMS: {cnt} files in {ev_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
