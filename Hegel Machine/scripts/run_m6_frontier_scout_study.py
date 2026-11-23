#!/usr/bin/env python3
"""Execute the 20-Task Frontier Real Scientific Benchmark Scout Study (`TB-Science v0.1.0` + `BioMysteryBench`).

Runs `Bare_ReAct_LiveGPT` vs `Hegel_M6_Frontier_LiveGPT` across `gpt-6-luna` and `gpt-6.2-sol`
(`20 tasks x 2 arms x 2 models = 80 containerized multi-step episodes`):
  - Wave 1 (`10` tasks: `8` TB-Science + `2` BioMystery)
  - Online Wave-1 -> Wave-2 Causal Sublation (`consolidate_wave1_into_m6_forest`, strictly blind to `/tests`)
  - Wave 2 (`10` tasks: `7` TB-Science + `3` BioMystery)
  - Full forensic per-episode folders (`00_episode_summary.json`, `01_decision_provenance_dag.json`,
    `02_human_readable_trajectory.md`, `03_official_verifier_report.json`, `04_submitted_artifacts_snapshot/`)
  - Cryptographic `SHA256SUMS.txt` over every evidence artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
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
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    SubscriptionQuotaPausedError,
    discover_chatgpt_subscription_codex_cli,
)


ARMS = ("Bare_ReAct_LiveGPT", "Hegel_M6_Frontier_LiveGPT")


def _write_sha256sums(evidence_dir: Path) -> int:
    lines: list[str] = []
    files = sorted(
        p for p in evidence_dir.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    )
    for fp in files:
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        rel = fp.relative_to(evidence_dir).as_posix()
        lines.append(f"{h}  {rel}")
    (evidence_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def _aggregate_results(all_summaries: list[dict[str, Any]], models: list[str], arms: tuple[str, ...]) -> dict[str, Any]:
    by_model_arm: dict[str, dict[str, Any]] = {}
    head_to_head: dict[str, list[dict[str, Any]]] = {}
    hegel_arm = arms[1]

    for m in models:
        by_model_arm[m] = {}
        for arm in arms:
            sub = [s for s in all_summaries if s["model"] == m and s["arm"] == arm]
            if not sub:
                continue
            tb_sub = [s for s in sub if s["benchmark"] == "Terminal-Bench-Science-0.1"]
            bm_sub = [s for s in sub if s["benchmark"] == "BioMysteryBench-Preview"]
            w1_sub = [s for s in sub if s["wave"] == "wave_1"]
            w2_sub = [s for s in sub if s["wave"] == "wave_2"]

            tax_counts: dict[str, int] = {}
            for s in sub:
                tx = str(s["failure_taxonomy"])
                tax_counts[tx] = tax_counts.get(tx, 0) + 1

            by_model_arm[m][arm] = {
                "episode_count": len(sub),
                "overall_pass_count": sum(1 for s in sub if s["strict_pass"]),
                "overall_pass_rate": round(sum(1 for s in sub if s["strict_pass"]) / len(sub), 4),
                "tb_science_pass": f"{sum(1 for s in tb_sub if s['strict_pass'])}/{len(tb_sub)}",
                "tb_science_pass_rate": round(sum(1 for s in tb_sub if s["strict_pass"]) / max(1, len(tb_sub)), 4),
                "biomystery_pass": f"{sum(1 for s in bm_sub if s['strict_pass'])}/{len(bm_sub)}",
                "biomystery_pass_rate": round(sum(1 for s in bm_sub if s["strict_pass"]) / max(1, len(bm_sub)), 4),
                "wave_1_pass": f"{sum(1 for s in w1_sub if s['strict_pass'])}/{len(w1_sub)}",
                "wave_1_pass_rate": round(sum(1 for s in w1_sub if s["strict_pass"]) / max(1, len(w1_sub)), 4),
                "wave_2_pass": f"{sum(1 for s in w2_sub if s['strict_pass'])}/{len(w2_sub)}",
                "wave_2_pass_rate": round(sum(1 for s in w2_sub if s["strict_pass"]) / max(1, len(w2_sub)), 4),
                "mean_turns_used": round(sum(int(s["turns_used"]) for s in sub) / len(sub), 2),
                "total_nogoods_recorded": sum(int(s.get("nogood_count", 0)) for s in sub),
                "incumbent_checkpoint_alerts_fired": sum(int(s.get("incumbent_checkpoint_alerts_fired", 0)) for s in sub),
                "stress_challenge_gates_triggered": sum(1 for s in sub if s.get("stress_challenge_gate_triggered")),
                "pullback_gates_triggered": sum(1 for s in sub if s.get("pullback_gate_triggered")),
                "total_tokens": sum(int(s["total_tokens"]) for s in sub),
                "effective_billed_tokens": sum(int(s["effective_billed_tokens"]) for s in sub),
                "mean_wall_sec": round(sum(float(s["wall_ms"]) for s in sub) / (1000.0 * len(sub)), 2),
                "failure_taxonomy_breakdown": tax_counts,
            }

        # Task-by-task head-to-head table for model `m`
        m_bare = {s["task_id"]: s for s in all_summaries if s["model"] == m and s["arm"] == "Bare_ReAct_LiveGPT"}
        m_hegel = {s["task_id"]: s for s in all_summaries if s["model"] == m and s["arm"] == hegel_arm}
        rows_h2h: list[dict[str, Any]] = []
        for tid in sorted(set(m_bare.keys()) | set(m_hegel.keys())):
            b = m_bare.get(tid)
            h = m_hegel.get(tid)
            if not b or not h:
                continue
            if h["strict_pass"] and not b["strict_pass"]:
                delta_tag = "HEGEL_M6_WIN"
            elif b["strict_pass"] and not h["strict_pass"]:
                delta_tag = "BARE_REACT_WIN"
            elif h["strict_pass"] and b["strict_pass"]:
                delta_tag = "BOTH_PASS"
            else:
                delta_tag = "BOTH_FAIL"
            rows_h2h.append(
                {
                    "task_id": tid,
                    "wave": h["wave"],
                    "benchmark": h["benchmark"],
                    "domain": h["scientific_domain"],
                    "outcome": delta_tag,
                    "bare_pass": b["strict_pass"],
                    "hegel_m6_pass": h["strict_pass"],
                    "bare_taxonomy": b["failure_taxonomy"],
                    "hegel_m6_taxonomy": h["failure_taxonomy"],
                    "bare_turns": b["turns_used"],
                    "hegel_m6_turns": h["turns_used"],
                    "hegel_nogoods": h["nogood_count"],
                    "hegel_pullback_gate": h["pullback_gate_triggered"],
                }
            )
        head_to_head[m] = rows_h2h

    return {
        "by_model_and_arm": by_model_arm,
        "head_to_head_by_model": head_to_head,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_frontier_scout_80ep_20261121",
    )
    parser.add_argument("--models", nargs="+", default=["gpt-6-luna", "gpt-6.2-sol"])
    parser.add_argument("--hegel-arm", type=str, default="Hegel_M6_Frontier_LiveGPT")
    parser.add_argument("--study-id", type=str, default="m6_frontier_scout_80ep_20261121")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument("--cmd-timeout", type=int, default=120)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    args = parser.parse_args()
    active_arms = ("Bare_ReAct_LiveGPT", str(args.hegel_arm))

    codex_info = discover_chatgpt_subscription_codex_cli()
    if not codex_info.get("available") or codex_info.get("auth_mode") != "chatgpt":
        print(f"ERROR: Official ChatGPT subscription unavailable: {codex_info}", file=sys.stderr)
        return 2

    ev_dir: Path = args.evidence_dir
    ep_root = ev_dir / "episodes"
    ep_root.mkdir(parents=True, exist_ok=True)

    # Throttle concurrent heavy Harbor verifier runs to 3 so wall-clock/CPU-sensitive verifiers never starve!
    verifier_sem = threading.Semaphore(3)
    orig_tb_verifier = m6f.run_tb_separate_verifier

    def _throttled_tb_verifier(task: m6f.FrontierBenchmarkTask, agent_cid: str) -> dict[str, Any]:
        with verifier_sem:
            return orig_tb_verifier(task, agent_cid)

    m6f.run_tb_separate_verifier = _throttled_tb_verifier  # type: ignore[assignment]

    waves = m6f.load_frontier_scout_20_tasks()
    models = list(args.models)

    # Initialize independent M6 forest per model so each model's Wave-1 -> Wave-2 consolidation is 100% self-contained!
    model_forests: dict[str, tuple[Any, dict[str, dict[str, Any]], list[tuple[str, str]], HippoRAGWormholeIndex]] = {}
    for m in models:
        f_obj, f_arch, m_dag = m6f.build_m6_frontier_general_forest()
        model_forests[m] = (f_obj, f_arch, m_dag, HippoRAGWormholeIndex(f_obj))

    all_summaries: list[dict[str, Any]] = []
    quota_paused: dict[str, str] = {}
    pause_lock = threading.Lock()
    study_t0 = monotonic()

    def _run_one_job(model: str, arm: str, task: m6f.FrontierBenchmarkTask) -> dict[str, Any] | None:
        with pause_lock:
            if model in quota_paused:
                return None
        ep_dir = ep_root / f"{task.wave}__{model}__{arm}__{task.task_id}"
        sum_file = ep_dir / "00_episode_summary.json"
        if sum_file.is_file():
            try:
                cached = json.loads(sum_file.read_text(encoding="utf-8"))
                if cached.get("provenance_verified") is True:
                    print(
                        f"  [CACHED] {task.wave} | {model:12s} | {arm:25s} | {task.task_id:38s} -> pass={cached['strict_pass']}",
                        flush=True,
                    )
                    return cached
            except Exception:
                pass

        f_obj, f_arch, _, h_idx = model_forests[model]
        try:
            res = m6f.run_frontier_episode(
                task,
                arm,
                model=model,
                codex_info=codex_info,
                forest=f_obj,
                forest_archive=f_arch,
                hippo_idx=h_idx,
                episode_out_dir=ep_dir,
                run_id=str(args.study_id),
                max_turns=args.max_turns,
                cmd_timeout_sec=args.cmd_timeout,
                reasoning_effort=args.reasoning_effort,
            )
            print(
                f"  [DONE]   {task.wave} | {model:12s} | {arm:25s} | {task.task_id:38s} -> "
                f"pass={str(res['strict_pass']):5s} | turns={res['turns_used']:2d} | nogoods={res['nogood_count']} | "
                f"tax={res['failure_taxonomy']} ({res['wall_ms']/1000.0:.1f}s)",
                flush=True,
            )
            return res
        except SubscriptionQuotaPausedError as q_exc:
            with pause_lock:
                quota_paused[model] = str(q_exc)
            print(f"  [QUOTA PAUSED] {model}: {q_exc}", flush=True)
            return None

    # --- Execute Wave 1 ---
    w1_jobs = [(m, arm, t) for t in waves["wave_1"] for m in models for arm in active_arms]
    print(f"=== Starting Frontier Study WAVE 1 ({len(w1_jobs)} containerized episodes, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(_run_one_job, m, arm, t) for (m, arm, t) in w1_jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    # --- Online Wave-1 -> Wave-2 Causal Sublation per Model ---
    consolidation_receipts: dict[str, Any] = {}
    for m in models:
        w1_m6 = [s for s in all_summaries if s["model"] == m and s["arm"] == str(args.hegel_arm) and s["wave"] == "wave_1"]
        f_obj, f_arch, m_dag, _ = model_forests[m]
        c_rec = m6f.consolidate_wave1_into_m6_forest(
            wave1_m6_summaries=w1_m6,
            forest=f_obj,
            archive=f_arch,
            meta_dag=m_dag,
        )
        consolidation_receipts[m] = c_rec
        (ev_dir / f"wave1_to_wave2_consolidation_{m}.json").write_text(
            json.dumps(c_rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"=== Consolidated {c_rec['consolidated_episode_count']} Wave-1 M6 episodes into `{m}` forest ===", flush=True)

    # --- Execute Wave 2 ---
    w2_jobs = [(m, arm, t) for t in waves["wave_2"] for m in models for arm in active_arms]
    print(f"=== Starting Frontier Study WAVE 2 ({len(w2_jobs)} containerized episodes, workers={args.max_workers}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(_run_one_job, m, arm, t) for (m, arm, t) in w2_jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    agg = _aggregate_results(all_summaries, models, active_arms)
    manifest = {
        "study_id": str(args.study_id),
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "provider_backend": codex_info["provider_backend"],
        "auth_mode": codex_info["auth_mode"],
        "zero_ruoli_proxy_verified": True,
        "models": models,
        "arms": list(active_arms),
        "max_turns": args.max_turns,
        "cmd_timeout_sec": args.cmd_timeout,
        "completed_episode_count": len(all_summaries),
        "expected_episode_count": len(w1_jobs) + len(w2_jobs),
        "quota_paused_models": quota_paused,
        "total_study_wall_sec": round(monotonic() - study_t0, 2),
        "aggregate": agg,
        "wave1_consolidation_summary": {
            m: {
                "consolidated_episode_count": r["consolidated_episode_count"],
                "bayes_ball_mechanism_channel_active": r["bayes_ball_mechanism_channel_active"],
                "bayes_ball_collider_channel_blocked": r["bayes_ball_collider_channel_blocked"],
            }
            for m, r in consolidation_receipts.items()
        },
    }
    (ev_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ev_dir / "all_episode_summaries.json").write_text(
        json.dumps(sorted(all_summaries, key=lambda x: (x["model"], x["wave"], x["task_id"], x["arm"])), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    hashed_cnt = _write_sha256sums(ev_dir)
    print(f"=== Frontier Scout Study Finished: {len(all_summaries)}/{len(w1_jobs)+len(w2_jobs)} episodes, {hashed_cnt} SHA-256 sealed files ===", flush=True)
    return 0 if not quota_paused else 3


if __name__ == "__main__":
    raise SystemExit(main())
