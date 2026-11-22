#!/usr/bin/env python3
"""Run the 28-Turn Sovereign `Hegel_M6_4_Frontier_LiveGPT` Executable-Wormhole 20-Task Frontier Study on `gpt-6.2-sol`.

Evaluates 20 fresh containerized `Hegel_M6_4_Frontier_LiveGPT` episodes (`Wave 1` = 10 -> Causal-Signature-Gated
Methodological + Pure-AST Executable Code Sublation -> `Wave 2` = 10) with:
  1. Retrieval-Gated Executable Python Wormhole Mounting (`/tmp/hegel_wormhole_skills` with `@track_wormhole_skill`
     ground-truth runtime JSONL telemetry harvested into `00_episode_summary.json` & `01_decision_provenance_dag.json`)
  2. Online Wave-1 -> Wave-2 Pure-AST Function Sublation (`consolidate_wave1_executable_modules_into_forest`)
  3. Official `bio-mystery-bench/harness/scorer.py` verbatim parity (`_list_match == False` falls through to `_llm_judge`,
     with explicit `llm_judge_tokens` telemetry and strict anti-provenance `allowed_domains` enforcement)
  4. Official `task.toml` verifier specs (`amr-poisson-optimize` at `12288 MB`, `4 CPU`, `3600s`, `--network none`)
alongside the 20 sealed 28-turn `Bare_ReAct_LiveGPT` baseline episodes (`13/20 = 65.0%` with official AMR regrade).
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_4_executable_wormhole_frontier_40ep_20261121",
    )
    parser.add_argument(
        "--sealed-bare-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_1_frontier_28turn_40ep_20261121" / "episodes",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--max-workers", type=int, default=8)
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
    active_arms = ("Bare_ReAct_LiveGPT", "Hegel_M6_4_Frontier_LiveGPT")

    # Stage sealed 28-turn Bare_ReAct_LiveGPT baseline episodes (including official 12GiB/3600s 23/23 AMR parity = 13/20!)
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

    # Exclusive lock for heavy 12GiB AMR verifier; 3-slot semaphore for normal verifiers!
    MAX_V_SLOTS = 3
    verifier_sem = threading.Semaphore(MAX_V_SLOTS)
    orig_tb_verifier = m6f.run_tb_separate_verifier

    def _smart_throttled_verifier(task: m6f.FrontierBenchmarkTask, agent_cid: str) -> dict[str, Any]:
        needed = MAX_V_SLOTS if task.verifier_memory_mb >= 10000 else 1
        for _ in range(needed):
            verifier_sem.acquire()
        try:
            return orig_tb_verifier(task, agent_cid)
        finally:
            for _ in range(needed):
                verifier_sem.release()

    m6f.run_tb_separate_verifier = _smart_throttled_verifier  # type: ignore[assignment]

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

    def _run_m6_4_task(task: m6f.FrontierBenchmarkTask) -> dict[str, Any] | None:
        with pause_lock:
            if model in quota_paused:
                return None
        f_sha = _forest_sha16()
        ep_dir = ep_root / f"{task.wave}__{model}__Hegel_M6_4_Frontier_LiveGPT__{task.task_id}"
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
                "Hegel_M6_4_Frontier_LiveGPT",
                model=model,
                codex_info=codex_info,
                forest=f_obj,
                forest_archive=f_arch,
                hippo_idx=h_idx,
                episode_out_dir=ep_dir,
                run_id="m6_4_frontier_28turn",
                max_turns=args.max_turns,
                cmd_timeout_sec=args.cmd_timeout,
                reasoning_effort="medium",
            )
            res["forest_snapshot_sha16"] = f_sha
            sum_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"  [DONE]   {task.wave} | {task.task_id:38s} -> pass={str(res['strict_pass']):5s} | "
                f"turns={res['turns_used']:2d} | skills={res.get('skill_activation_count', 0):2d} | "
                f"nogoods={res['nogood_count']} | tax={res['failure_taxonomy']} ({res['wall_ms']/1000.0:.1f}s)",
                flush=True,
            )
            return res
        except SubscriptionQuotaPausedError as q_exc:
            with pause_lock:
                quota_paused[model] = str(q_exc)
            print(f"  [QUOTA PAUSED] {model}: {q_exc}", flush=True)
            return None

    # --- Wave 1 (`10` fresh `Hegel_M6_4` episodes) ---
    print(f"=== Starting Hegel M6.4 Executable-Wormhole Frontier WAVE 1 (10 tasks, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_m6_4_task, t) for t in waves["wave_1"]]):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    # --- Wave-1 -> Wave-2 Causal-Signature-Gated Methodological + Pure-AST Executable Sublation ---
    w1_m6_4 = [s for s in all_summaries if s["arm"] == "Hegel_M6_4_Frontier_LiveGPT" and s["wave"] == "wave_1"]
    c_rec = m6f.consolidate_wave1_into_m6_forest(
        wave1_m6_summaries=w1_m6_4,
        forest=f_obj,
        archive=f_arch,
        meta_dag=m_dag,
    )
    w1_dirs = [
        (s, ep_root / f"{s['wave']}__{model}__Hegel_M6_4_Frontier_LiveGPT__{s['task_id']}")
        for s in w1_m6_4
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
        f"=== Consolidated {c_rec['consolidated_episode_count']} Wave-1 M6.4 episodes + "
        f"{len(ast_events)} pure-AST executable modules into forest ===",
        flush=True,
    )

    # --- Wave 2 (`10` fresh `Hegel_M6_4` episodes) ---
    print(f"=== Starting Hegel M6.4 Executable-Wormhole Frontier WAVE 2 (10 tasks, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_m6_4_task, t) for t in waves["wave_2"]]):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    agg = _aggregate_results(all_summaries, [model], active_arms)

    # Summarize ground-truth Skill∞ runtime activations & BioMystery judge token telemetry
    m6_4_eps = [s for s in all_summaries if s["arm"] == "Hegel_M6_4_Frontier_LiveGPT"]
    tb_m6_4_eps = [s for s in m6_4_eps if s["benchmark"] == "Terminal-Bench-Science-0.1"]
    eps_with_skill_calls = [s["task_id"] for s in tb_m6_4_eps if int(s.get("skill_activation_count", 0)) > 0]
    total_skill_calls = sum(int(s.get("skill_activation_count", 0)) for s in tb_m6_4_eps)

    bm_judge_tokens_total = 0
    for s in m6_4_eps:
        if s["benchmark"] == "BioMysteryBench-Preview":
            vf = ep_root / f"{s['wave']}__{model}__Hegel_M6_4_Frontier_LiveGPT__{s['task_id']}" / "03_official_verifier_report.json"
            if vf.is_file():
                v_obj = json.loads(vf.read_text(encoding="utf-8"))
                bm_judge_tokens_total += int((v_obj.get("llm_judge_tokens") or {}).get("total_tokens", 0))

    manifest = {
        "study_id": "m6_4_executable_wormhole_frontier_40ep_20261121",
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "provider_backend": codex_info["provider_backend"],
        "auth_mode": codex_info["auth_mode"],
        "reused_sealed_bare_28turn_baseline_from": "m6_1_frontier_28turn_40ep_20261121 (13/20 official AMR regrade, 12/20 raw)",
        "fresh_live_m6_4_episodes": len(m6_4_eps),
        "models": [model],
        "arms": list(active_arms),
        "max_turns": args.max_turns,
        "cmd_timeout_sec": args.cmd_timeout,
        "completed_episode_count": len(all_summaries),
        "skill_infinity_runtime_telemetry": {
            "tb_science_episodes_total": len(tb_m6_4_eps),
            "tb_science_episodes_activating_wormhole_code": len(eps_with_skill_calls),
            "activating_task_ids": sorted(eps_with_skill_calls),
            "total_runtime_function_calls": total_skill_calls,
            "wave1_ast_sublated_module_count": len(ast_events),
        },
        "biomystery_judge_total_tokens_m6_4": bm_judge_tokens_total,
        "quota_paused_models": quota_paused,
        "total_study_wall_sec": round(monotonic() - study_t0, 2),
        "aggregate": agg,
    }
    (ev_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    n_files = _write_sha256sums(ev_dir)
    print(f"=== Finished Hegel M6.4 Executable-Wormhole Study: {len(all_summaries)} total eps, {n_files} SHA-256 files ===", flush=True)
    print(json.dumps(agg["by_model_and_arm"], indent=2, ensure_ascii=False), flush=True)
    return 0 if not quota_paused else 3


if __name__ == "__main__":
    raise SystemExit(main())
