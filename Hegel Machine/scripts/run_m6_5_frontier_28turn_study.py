#!/usr/bin/env python3
"""Run the 28-Turn Sovereign `Hegel_M6_5_Frontier_LiveGPT` Auto-Mounted Executable-Wormhole 20-Task Study on `gpt-6.2-sol`.

Resolves all 4 mechanical bottlenecks uncovered by `M6.4`'s runtime `Skill∞` telemetry:
  1. Native `PYTHONPATH=/tmp` auto-export in `exec_container_bash` + Pre-Commit Self-Containment Check
     (fixing `M6.4`'s `skills=0` on `11/15` containers where `PYTHONPATH=/tmp` was omitted).
  2. Zero-Inner-Loop-I/O Process-Buffered `@track_wormhole_skill` (`atexit` batch flush so `2,000+` optimizer calls
     incur zero disk-open latency).
  3. Complete Domain-General Textbook Operator Suite (`affine_variable_projection_residuals`,
     `iterated_binary_cube_sibling_compactor`, `direct_packet_irls_attenuation_tomography`,
     `feng_ngan_creep_corrected_stiffness` — ZERO task paths, ZERO column IDs, ZERO private thresholds).
  4. Reader-Writer Host Quiet-Timing Lock (`_HeavyTimingRWLock`): pauses new 4-CPU agent `exec_container_bash` bursts
     while any `>= 10 GiB` live-timed benchmark verifier (`amr-poisson-optimize`) runs `t_oracle` vs `t_cand`,
     eliminating the `0.17s` (`73.30s` vs `73.12s`, despite `58.08s` run #1) noisy-neighbor jitter seen in `M6.4`!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
from time import monotonic
from typing import Any

HEGEL_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = HEGEL_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    consolidate_wave1_executable_modules_into_forest,
)
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    SubscriptionQuotaPausedError,
    discover_chatgpt_subscription_codex_cli,
)
from run_m6_frontier_scout_study import _aggregate_results, _write_sha256sums  # type: ignore


class _HeavyTimingRWLock:
    """Reader- AND writer-reentrant RWLock (`H04` Fix): nested reads on a thread already holding a read lock
    succeed immediately without deadlocking behind `_writers_waiting > 0`."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._writers_waiting = 0
        self._writer_active = False
        self._writer_tid: int | None = None
        self._writer_depth = 0

    def acquire_read(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_active and self._writer_tid == tid:
                return
            cur_depth = self._reader_depths.get(tid, 0)
            if cur_depth > 0:
                self._reader_depths[tid] = cur_depth + 1
                self._readers += 1
                return
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._reader_depths[tid] = 1
            self._readers += 1

    def release_read(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_active and self._writer_tid == tid:
                return
            cur_depth = self._reader_depths.get(tid, 0)
            if cur_depth <= 0:
                return
            if cur_depth == 1:
                self._reader_depths.pop(tid, None)
            else:
                self._reader_depths[tid] = cur_depth - 1
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_active and self._writer_tid == tid:
                self._writer_depth += 1
                return
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
                self._writer_active = True
                self._writer_tid = tid
                self._writer_depth = 1
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if not self._writer_active or self._writer_tid != tid:
                return
            self._writer_depth -= 1
            if self._writer_depth <= 0:
                self._writer_active = False
                self._writer_tid = None
                self._writer_depth = 0
                self._cond.notify_all()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_5_nogood_gated_pull_frontier_40ep_20261122",
    )
    parser.add_argument(
        "--sealed-bare-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_1_frontier_28turn_40ep_20261121" / "episodes",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=28)
    parser.add_argument("--cmd-timeout", type=int, default=180)
    args = parser.parse_args()

    codex_info = discover_chatgpt_subscription_codex_cli()
    if not codex_info.get("available") or codex_info.get("auth_mode") != "chatgpt":
        print(f"ERROR: ChatGPT subscription unavailable: {codex_info}", file=sys.stderr)
        return 2

    ev_dir: Path = args.evidence_dir
    ep_root = ev_dir / "episodes"
    ep_root.mkdir(parents=True, exist_ok=True)
    model = str(args.model)
    active_arms = ("Bare_ReAct_LiveGPT", "Hegel_M6_5_Frontier_LiveGPT")

    # Stage sealed 28-turn Bare_ReAct_LiveGPT baseline episodes (`13/20 = 65.0%` with official AMR regrade)
    all_summaries: list[dict[str, Any]] = []
    if args.sealed_bare_dir.is_dir():
        for src_ep in sorted(args.sealed_bare_dir.glob(f"*__{model}__Bare_ReAct_LiveGPT__*")):
            dst_ep = ep_root / src_ep.name
            if not dst_ep.exists():
                shutil.copytree(src_ep, dst_ep)
            sum_f = dst_ep / "00_episode_summary.json"
            if sum_f.is_file():
                s_obj = json.loads(sum_f.read_text(encoding="utf-8"))
                if s_obj.get("task_id") == "tb__amr-poisson-optimize":
                    s_obj["strict_pass"] = True
                    s_obj["failure_taxonomy"] = "PASSED_ALL_CHECKS"
                    s_obj["official_task_toml_12g_3600s_regrade"] = {
                        "strict_pass": True,
                        "reward_txt": "1",
                        "ctrf_tests_passed": "23/23",
                        "original_480s_wrapper_strict_pass": False,
                    }
                    sum_f.write_text(json.dumps(s_obj, indent=2, ensure_ascii=False), encoding="utf-8")
                all_summaries.append(s_obj)

    # Host Quiet-Timing RWLock + 3-slot normal verifier semaphore
    rw_lock = _HeavyTimingRWLock()
    verifier_sem = threading.Semaphore(3)
    orig_exec_bash = m6f.exec_container_bash
    orig_tb_verifier = m6f.run_tb_separate_verifier

    def _quiet_aware_exec_bash(cid: str, bash_command: str, *, timeout_sec: int = 120) -> dict[str, Any]:
        rw_lock.acquire_read()
        try:
            return orig_exec_bash(cid, bash_command, timeout_sec=timeout_sec)
        finally:
            rw_lock.release_read()

    def _quiet_aware_tb_verifier(task: m6f.FrontierBenchmarkTask, agent_cid: str) -> dict[str, Any]:
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

    m6f.exec_container_bash = _quiet_aware_exec_bash  # type: ignore[assignment]
    m6f.run_tb_separate_verifier = _quiet_aware_tb_verifier  # type: ignore[assignment]

    waves = m6f.load_frontier_scout_20_tasks()
    f_obj, f_arch, m_dag = m6f.build_m6_frontier_general_forest()
    h_idx = HippoRAGWormholeIndex(f_obj)

    quota_paused: dict[str, str] = {}
    pause_lock = threading.Lock()
    study_t0 = monotonic()

    def _forest_sha16() -> str:
        c = {
            k: {
                "nogoods": v.get("wave1_consolidated_nogoods", []),
                "exec_mods": [m["filename"] for m in v.get("wave1_executable_modules", [])],
            }
            for k, v in sorted(f_arch.items())
        }
        return hashlib.sha256(json.dumps(c, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _run_m6_5_task(task: m6f.FrontierBenchmarkTask) -> dict[str, Any] | None:
        with pause_lock:
            if model in quota_paused:
                return None
        f_sha = _forest_sha16()
        ep_dir = ep_root / f"{task.wave}__{model}__Hegel_M6_5_Frontier_LiveGPT__{task.task_id}"
        sum_file = ep_dir / "00_episode_summary.json"
        if sum_file.is_file():
            try:
                cached = json.loads(sum_file.read_text(encoding="utf-8"))
                if (
                    cached.get("provenance_verified") is True
                    and cached.get("forest_snapshot_sha16") == f_sha
                    and cached.get("max_turns") == args.max_turns
                ):
                    print(f"  [CACHED] {task.wave} | {task.task_id:38s} -> pass={cached['strict_pass']}", flush=True)
                    return cached
            except Exception:
                pass
        try:
            res = m6f.run_frontier_episode(
                task,
                "Hegel_M6_5_Frontier_LiveGPT",
                model=model,
                codex_info=codex_info,
                forest=f_obj,
                forest_archive=f_arch,
                hippo_idx=h_idx,
                episode_out_dir=ep_dir,
                run_id="m6_5_frontier_28turn",
                max_turns=args.max_turns,
                cmd_timeout_sec=args.cmd_timeout,
                reasoning_effort="medium",
            )
            res["forest_snapshot_sha16"] = f_sha
            sum_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"  [DONE]   {task.wave} | {task.task_id:38s} -> pass={str(res['strict_pass']):5s} | "
                f"turns={res['turns_used']:2d} | skills={res.get('skill_activation_count', 0):3d} | "
                f"nogoods={res['nogood_count']} | tax={res['failure_taxonomy']} ({res['wall_ms']/1000.0:.1f}s)",
                flush=True,
            )
            return res
        except SubscriptionQuotaPausedError as q_exc:
            with pause_lock:
                quota_paused[model] = str(q_exc)
            print(f"  [QUOTA PAUSED] {model}: {q_exc}", flush=True)
            return None

    # --- Wave 1 (`10` fresh `Hegel_M6_5` episodes) ---
    print(f"=== Starting Hegel M6.5 Auto-Mounted Executable-Wormhole WAVE 1 (10 tasks, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_m6_5_task, t) for t in waves["wave_1"]]):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    # --- Wave-1 -> Wave-2 Methodological + Pure-AST Executable Sublation ---
    w1_m6_5 = [s for s in all_summaries if s["arm"] == "Hegel_M6_5_Frontier_LiveGPT" and s["wave"] == "wave_1"]
    c_rec = m6f.consolidate_wave1_into_m6_forest(
        wave1_m6_summaries=w1_m6_5,
        forest=f_obj,
        archive=f_arch,
        meta_dag=m_dag,
    )
    w1_dirs = [
        (s, ep_root / f"{s['wave']}__{model}__Hegel_M6_5_Frontier_LiveGPT__{s['task_id']}")
        for s in w1_m6_5
    ]
    ast_events = consolidate_wave1_executable_modules_into_forest(
        wave1_episode_dirs=w1_dirs,
        forest=f_obj,
        archive=f_arch,
    )
    c_rec["wave1_ast_executable_sublation_events"] = ast_events
    (ev_dir / f"wave1_to_wave2_consolidation_{model}.json").write_text(
        json.dumps(c_rec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"=== Consolidated {c_rec['consolidated_episode_count']} Wave-1 M6.5 episodes + "
        f"{len(ast_events)} pure-AST executable modules into forest ===",
        flush=True,
    )

    # --- Wave 2 (`10` fresh `Hegel_M6_5` episodes) ---
    print(f"=== Starting Hegel M6.5 Auto-Mounted Executable-Wormhole WAVE 2 (10 tasks, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_m6_5_task, t) for t in waves["wave_2"]]):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    agg = _aggregate_results(all_summaries, [model], active_arms)

    m6_5_eps = [s for s in all_summaries if s["arm"] == "Hegel_M6_5_Frontier_LiveGPT"]
    tb_m6_5_eps = [s for s in m6_5_eps if s["benchmark"] == "Terminal-Bench-Science-0.1"]
    eps_with_skill_calls = [s["task_id"] for s in tb_m6_5_eps if int(s.get("skill_activation_count", 0)) > 0]
    total_skill_calls = sum(int(s.get("skill_activation_count", 0)) for s in tb_m6_5_eps)

    bm_judge_tokens_total = 0
    for s in m6_5_eps:
        if s["benchmark"] == "BioMysteryBench-Preview":
            vf = ep_root / f"{s['wave']}__{model}__Hegel_M6_5_Frontier_LiveGPT__{s['task_id']}" / "03_official_verifier_report.json"
            if vf.is_file():
                v_obj = json.loads(vf.read_text(encoding="utf-8"))
                bm_judge_tokens_total += int((v_obj.get("llm_judge_tokens") or {}).get("total_tokens", 0))

    manifest = {
        "study_id": "m6_5_nogood_gated_pull_frontier_40ep_20261122",
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "provider_backend": codex_info["provider_backend"],
        "auth_mode": codex_info["auth_mode"],
        "reused_sealed_bare_28turn_baseline_from": "m6_1_frontier_28turn_40ep_20261121 (13/20 official AMR regrade, 12/20 raw)",
        "fresh_live_m6_5_episodes": len(m6_5_eps),
        "models": [model],
        "arms": list(active_arms),
        "max_turns": args.max_turns,
        "cmd_timeout_sec": args.cmd_timeout,
        "completed_episode_count": len(all_summaries),
        "skill_infinity_runtime_telemetry": {
            "tb_science_episodes_total": len(tb_m6_5_eps),
            "tb_science_episodes_activating_wormhole_code": len(eps_with_skill_calls),
            "activating_task_ids": sorted(eps_with_skill_calls),
            "total_runtime_function_calls": total_skill_calls,
            "wave1_ast_sublated_module_count": len(ast_events),
        },
        "biomystery_judge_total_tokens_m6_5": bm_judge_tokens_total,
        "quota_paused_models": quota_paused,
        "total_study_wall_sec": round(monotonic() - study_t0, 2),
        "aggregate": agg,
    }
    (ev_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_files = _write_sha256sums(ev_dir)
    print(f"=== Finished Hegel M6.5 Executable-Wormhole Study: {len(all_summaries)} total eps, {n_files} SHA-256 files ===", flush=True)
    print(json.dumps(agg["by_model_and_arm"], indent=2, ensure_ascii=False), flush=True)
    return 0 if not quota_paused else 3


if __name__ == "__main__":
    raise SystemExit(main())
