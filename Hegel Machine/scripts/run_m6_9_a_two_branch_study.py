#!/usr/bin/env python3
"""Hegel M6.9-A Stage-2 Two-Branch Dialectical Synthesis (`Thesis + Antithesis -> Public Synthesis`) Live Study.

Executes the declared `2x`-budget (`28 + 28` turns + blind public synthesis) Frontier evaluation with
full `1x` and `2x` matched-budget controls (`02_plan_and_implementation_changes.md` §1 Track `A`):
  1. `Bare_ReAct_Run1_1x` (sealed `m6_1`, `12/20` native / `13/20` w/ hash-bound `05_regrade_receipt.json`).
  2. `Bare_ReAct_Run2_1x` (fresh live `Branch 1 / Thesis`, graded strictly post-lock).
  3. `Oracle_BestOf2_Bare_2x` (`Bare_Run1 U Bare_Run2`, matched `2x`-budget clairvoyant Bare upper bound!).
  4. `Hegel_M6_9_Antithesis_1x` (fresh live `Branch 2 / Antithesis`, graded strictly post-lock).
  5. `Hegel_M6_9_A_TwoBranch_LiveGPT` (`Blind Public Synthesis(Thesis, Antithesis)`, locked & graded by
     official air-gapped verifiers BEFORE any branch verifier executes!).
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
SCRIPTS_DIR = HEGEL_ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    SubscriptionQuotaPausedError,
    discover_chatgpt_subscription_codex_cli,
)
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY,
    NODE_TO_EXECUTABLE_WORMHOLE_MODULE,
    consolidate_wave1_executable_modules_into_forest,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    HeavyTimingRWLock,
    SKILL_07_MULTISCALE_RECONCILIATION,
    consolidate_wave1_skill_manifests_with_reconciliation,
)
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import (
    PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY,
    run_m6_9_a_two_branch_episode,
)
from run_m6_frontier_scout_study import _aggregate_results, _write_sha256sums  # type: ignore


def _compute_snapshot_dir_sha256(snap_dir: Path) -> str:
    h = hashlib.sha256()
    if not snap_dir.is_dir():
        return ""
    for fp in sorted(p for p in snap_dir.rglob("*") if p.is_file()):
        h.update(fp.name.encode("utf-8") + b"\0" + hashlib.sha256(fp.read_bytes()).digest())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_9_a_two_branch_frontier_40ep_20261122",
    )
    parser.add_argument(
        "--sealed-bare-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_1_frontier_28turn_40ep_20261121" / "episodes",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-turns-per-branch", type=int, default=28)
    parser.add_argument("--cmd-timeout", type=int, default=180)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument(
        "--wave1-consolidation-source",
        choices=("antithesis_only", "synthesized"),
        default="antithesis_only",
        help="Consolidate Wave-1 lessons/producers strictly from Antithesis 1x branch (`antithesis_only`, pure 1x lineage) or full `synthesized` episode.",
    )
    args = parser.parse_args()

    codex_info = discover_chatgpt_subscription_codex_cli()
    if not codex_info.get("available") or codex_info.get("auth_mode") != "chatgpt":
        print(f"ERROR: ChatGPT subscription unavailable: {codex_info}", file=sys.stderr)
        return 2

    ev_dir: Path = args.evidence_dir
    ep_root = ev_dir / "episodes"
    branches_root = ev_dir / "branch_trajectories"
    ep_root.mkdir(parents=True, exist_ok=True)
    branches_root.mkdir(parents=True, exist_ok=True)
    model = str(args.model)
    hegel_arm = "Hegel_M6_9_A_TwoBranch_LiveGPT"
    active_arms = ("Bare_ReAct_LiveGPT", hegel_arm)

    # 1. Stage sealed 28-turn Bare_ReAct_Run1 (`12/20` native — ZERO synthetic regrade fabrication, ZERO mutation of `00_episode_summary.json`, `A03`/`P0a` Fix!)
    all_summaries: list[dict[str, Any]] = []
    bare_run1_by_task: dict[str, bool] = {}
    if args.sealed_bare_dir.is_dir():
        for src_ep in sorted(args.sealed_bare_dir.glob(f"*__{model}__Bare_ReAct_LiveGPT__*")):
            dst_ep = ep_root / src_ep.name
            if not dst_ep.exists():
                shutil.copytree(src_ep, dst_ep)
            sum_f = dst_ep / "00_episode_summary.json"
            ver_f = dst_ep / "03_official_verifier_report.json"
            if sum_f.is_file() and ver_f.is_file():
                s_obj = json.loads(sum_f.read_text(encoding="utf-8"))
                v_obj = json.loads(ver_f.read_text(encoding="utf-8"))
                bare_run1_by_task[str(s_obj["task_id"])] = bool(v_obj.get("strict_pass"))
                all_summaries.append({**s_obj, "strict_pass": bool(v_obj.get("strict_pass"))})

    # 2. Reader+Writer Reentrant `HeavyTimingRWLock` + 3-slot normal verifier semaphore
    rw_lock = HeavyTimingRWLock()
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

    lib_code_sha = hashlib.sha256(
        json.dumps(
            {
                "foundational_skills": {k: v[1] for k, v in sorted(NODE_TO_EXECUTABLE_WORMHOLE_MODULE.items())},
                "skill_07": SKILL_07_MULTISCALE_RECONCILIATION,
                "host_gate": HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY,
                "public_synth_bench": PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY,
                "cmd_timeout": args.cmd_timeout,
                "reasoning_effort": args.reasoning_effort,
                "wave1_consolidation_source": args.wave1_consolidation_source,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    def _forest_sha16() -> str:
        c = {
            "lib_code_sha": lib_code_sha,
            "nodes": {
                k: {
                    "nogoods": v.get("wave1_consolidated_nogoods", []),
                    "exec_mods": [
                        {"filename": m["filename"], "code_sha": hashlib.sha256(str(m.get("python_code", "")).encode("utf-8")).hexdigest()}
                        for m in v.get("wave1_executable_modules", [])
                    ],
                }
                for k, v in sorted(f_arch.items())
            },
            "subtrees": {
                sk: sv.to_dict() if hasattr(sv, "to_dict") else str(sv)
                for sk, sv in sorted(getattr(f_obj, "shared_base_subtrees", {}).items())
            },
        }
        return hashlib.sha256(json.dumps(c, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _run_m6_9_task(task: m6f.FrontierBenchmarkTask) -> dict[str, Any] | None:
        with pause_lock:
            if model in quota_paused:
                return None
        f_sha = _forest_sha16()
        ep_dir = ep_root / f"{task.wave}__{model}__{hegel_arm}__{task.task_id}"
        sum_file = ep_dir / "00_episode_summary.json"
        if sum_file.is_file():
            try:
                cached = json.loads(sum_file.read_text(encoding="utf-8"))
                if (
                    cached.get("provenance_verified") is True
                    and cached.get("forest_snapshot_sha16") == f_sha
                    and cached.get("max_turns") == 2 * args.max_turns_per_branch
                ):
                    print(f"  [CACHED] {task.wave} | {task.task_id:38s} -> pass={cached['strict_pass']}", flush=True)
                    return cached
            except Exception:
                pass
        try:
            res = run_m6_9_a_two_branch_episode(
                task,
                model=model,
                codex_info=codex_info,
                forest=f_obj,
                forest_archive=f_arch,
                hippo_idx=h_idx,
                episode_out_dir=ep_dir,
                branches_work_root=branches_root,
                run_id=ev_dir.name,
                max_turns_per_branch=args.max_turns_per_branch,
                cmd_timeout_sec=args.cmd_timeout,
                reasoning_effort=args.reasoning_effort,
            )
            res["forest_snapshot_sha16"] = f_sha
            sum_file.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            pb = res["postlock_branch_outcomes"]
            sd = res["synthesis_decision"]
            print(
                f"  [DONE]   {task.wave} | {task.task_id:38s} -> SYNTH={str(res['strict_pass']):5s} "
                f"(T={str(pb['thesis_bare_run2_strict_pass']):5s}, A={str(pb['antithesis_wormhole_strict_pass']):5s}, "
                f"pick={sd['selected_branch']} via {sd['selection_mechanism'][:28]}) ({res['wall_ms']/1000.0:.1f}s)",
                flush=True,
            )
            return res
        except SubscriptionQuotaPausedError as q_exc:
            with pause_lock:
                quota_paused[model] = str(q_exc)
            print(f"  [QUOTA PAUSED] {model}: {q_exc}", flush=True)
            return None

    # --- Wave 1 (`10` Two-Branch Dialectical Synthesis tasks) ---
    print(f"=== Starting Hegel M6.9-A Two-Branch Synthesis WAVE 1 (10 tasks, task_workers={args.max_workers}, 2 branches/task) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_m6_9_task, t) for t in waves["wave_1"]]):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    if quota_paused:
        print(f"Stopping before Wave 2 due to quota pause: {quota_paused}", flush=True)
        return 3

    # --- Wave-1 -> Wave-2 Dialectical Sublation (`R01` Fix: `antithesis_only` preserves pure 1x Antithesis lineage!) ---
    w1_m6_9 = [s for s in all_summaries if s["arm"] == hegel_arm and s["wave"] == "wave_1"]
    if args.wave1_consolidation_source == "antithesis_only":
        w1_source_pairs: list[tuple[dict[str, Any], Path]] = []
        for s in w1_m6_9:
            anti_d = branches_root / f"{s['wave']}__{model}__{s['task_id']}" / "branch_2_antithesis_wormhole"
            anti_sf = anti_d / "00_episode_summary.json"
            anti_s = json.loads(anti_sf.read_text(encoding="utf-8")) if anti_sf.is_file() else dict(s)
            anti_s["strict_pass"] = bool((s.get("postlock_branch_outcomes") or {}).get("antithesis_wormhole_strict_pass", s["strict_pass"]))
            w1_source_pairs.append((anti_s, anti_d if anti_d.is_dir() else (ep_root / f"{s['wave']}__{model}__{hegel_arm}__{s['task_id']}")))
        w1_cons_summaries = [p[0] for p in w1_source_pairs]
        w1_dirs = w1_source_pairs
    else:
        w1_cons_summaries = w1_m6_9
        w1_dirs = [(s, ep_root / f"{s['wave']}__{model}__{hegel_arm}__{s['task_id']}") for s in w1_m6_9]

    c_rec = m6f.consolidate_wave1_into_m6_forest(
        wave1_m6_summaries=w1_cons_summaries,
        forest=f_obj,
        archive=f_arch,
        meta_dag=m_dag,
    )
    ast_events = consolidate_wave1_executable_modules_into_forest(
        wave1_episode_dirs=w1_dirs,
        forest=f_obj,
        archive=f_arch,
    )
    manifest_reconcile = consolidate_wave1_skill_manifests_with_reconciliation(
        wave1_episode_dirs=w1_dirs,
        forest=f_obj,
        archive=f_arch,
    )
    c_rec["wave1_consolidation_source"] = args.wave1_consolidation_source
    c_rec["wave1_ast_executable_sublation_events"] = ast_events
    c_rec["wave1_full_manifest_and_m5_reconciliation"] = manifest_reconcile
    (ev_dir / f"wave1_to_wave2_consolidation_{model}.json").write_text(
        json.dumps(c_rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"=== Consolidated {c_rec['consolidated_episode_count']} Wave-1 M6.9-A episodes + "
        f"{len(manifest_reconcile['wave1_manifest_events'])} full producer manifests + "
        f"{len(manifest_reconcile['registered_shared_base_subtrees'])} SharedBaseSubtrees ===",
        flush=True,
    )

    # --- Wave 2 (`10` Two-Branch Dialectical Synthesis tasks) ---
    print(f"=== Starting Hegel M6.9-A Two-Branch Synthesis WAVE 2 (10 tasks, task_workers={args.max_workers}, 2 branches/task) ===", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_m6_9_task, t) for t in waves["wave_2"]]):
            r = fut.result()
            if r is not None:
                all_summaries.append(r)

    native_agg = _aggregate_results(all_summaries, [model], active_arms)

    regrade_adjusted_summaries: list[dict[str, Any]] = []
    regrade_receipts_applied: list[str] = []
    for s in all_summaries:
        s_copy = dict(s)
        ep_d = ep_root / f"{s['wave']}__{model}__{s['arm']}__{s['task_id']}"
        rec_f = ep_d / "05_regrade_receipt.json"
        if rec_f.is_file():
            r_obj = json.loads(rec_f.read_text(encoding="utf-8"))
            if r_obj.get("candidate_hash_match_verified") and r_obj.get("regrade_strict_pass"):
                s_copy["strict_pass"] = True
                regrade_receipts_applied.append(ep_d.name)
        regrade_adjusted_summaries.append(s_copy)
    regrade_agg = _aggregate_results(regrade_adjusted_summaries, [model], active_arms)

    # Compute complete `1x` AND `2x` matched-budget control scoreboard (`02_plan_and_implementation_changes.md`)
    m6_9_eps = sorted([s for s in all_summaries if s["arm"] == hegel_arm], key=lambda x: str(x["task_id"]))
    bare_r1_cnt = sum(1 for v in bare_run1_by_task.values() if v)
    bare_r2_cnt = sum(1 for s in m6_9_eps if s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"])
    anti_1x_cnt = sum(1 for s in m6_9_eps if s["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"])
    synth_2x_cnt = sum(1 for s in m6_9_eps if s["strict_pass"])
    oracle_bare_2x_cnt = sum(
        1
        for s in m6_9_eps
        if bare_run1_by_task.get(str(s["task_id"]), False) or s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"]
    )
    oracle_hegel_2x_cnt = sum(1 for s in m6_9_eps if s["postlock_branch_outcomes"]["oracle_either_branch_pass"])
    false_promos = sum(
        1
        for s in m6_9_eps
        if s["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"] and (not s["strict_pass"])
    )

    tb_m6_9_eps = [s for s in m6_9_eps if s["benchmark"] == "Terminal-Bench-Science-0.1"]
    eps_with_skill_calls = [s["task_id"] for s in tb_m6_9_eps if int(s.get("skill_activation_count", 0)) > 0]
    total_skill_calls = sum(int(s.get("skill_activation_count", 0)) for s in tb_m6_9_eps)

    bm_judge_tokens_total = 0
    solving_tokens_thesis = 0
    solving_tokens_antithesis = 0
    synthesis_selection_tokens = 0
    for s in m6_9_eps:
        ep_f = ep_root / f"{s['wave']}__{model}__{hegel_arm}__{s['task_id']}"
        dag_f = ep_f / "01_decision_provenance_dag.json"
        if dag_f.is_file():
            for r_d in json.loads(dag_f.read_text(encoding="utf-8")).get("records", []):
                dt_s = str(r_d.get("decision_type", ""))
                tok_d = int(r_d.get("prompt_tokens_delta", 0)) + int(r_d.get("completion_tokens_delta", 0))
                if dt_s == "D1_THESIS_BRANCH_COMPLETED":
                    solving_tokens_thesis += tok_d
                elif dt_s == "D2_ANTITHESIS_BRANCH_COMPLETED":
                    solving_tokens_antithesis += tok_d
                elif dt_s == "D3_DIALECTICAL_SYNTHESIS_COMMIT":
                    synthesis_selection_tokens += tok_d
        if s["benchmark"] == "BioMysteryBench-Preview":
            vf = ep_f / "03_official_verifier_report.json"
            if vf.is_file():
                v_obj = json.loads(vf.read_text(encoding="utf-8"))
                bm_judge_tokens_total += int((v_obj.get("llm_judge_tokens") or {}).get("total_tokens", 0))

    manifest = {
        "study_id": ev_dir.name,
        "architecture_track": "A_two_branch_dialectical_synthesis_declared_2x_budget",
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "provider_backend": codex_info["provider_backend"],
        "auth_mode": codex_info["auth_mode"],
        "declared_budget": "2x (28T Thesis + 28T Antithesis + Blind Public Synthesis)",
        "bare_run1_provenance_study": str(args.sealed_bare_dir),
        "wave1_consolidation_source": args.wave1_consolidation_source,
        "models": [model],
        "arms": list(active_arms),
        "max_turns_per_branch": args.max_turns_per_branch,
        "cmd_timeout_sec": args.cmd_timeout,
        "reasoning_effort": args.reasoning_effort,
        "completed_episode_count": len(all_summaries),
        "four_column_token_accounting": {
            "formation_consolidation_llm_tokens": 0,
            "solving_thesis_1x_total_tokens": solving_tokens_thesis,
            "solving_antithesis_1x_total_tokens": solving_tokens_antithesis,
            "solving_two_branch_2x_total_tokens": solving_tokens_thesis + solving_tokens_antithesis,
            "synthesis_selection_llm_tokens": synthesis_selection_tokens,
            "official_verifier_biomystery_judge_tokens": bm_judge_tokens_total,
        },
        "matched_1x_and_2x_budget_scoreboard": {
            "Bare_ReAct_Run1_1x_native": f"{bare_r1_cnt}/20 ({bare_r1_cnt/20:.1%})",
            "Bare_ReAct_Run2_1x_native": f"{bare_r2_cnt}/20 ({bare_r2_cnt/20:.1%})",
            "Mean_Bare_1x_native": round(0.5 * (bare_r1_cnt + bare_r2_cnt), 2),
            "Oracle_BestOf2_Bare_2x_native": f"{oracle_bare_2x_cnt}/20 ({oracle_bare_2x_cnt/20:.1%})",
            "Hegel_M6_9_Antithesis_1x_native": f"{anti_1x_cnt}/20 ({anti_1x_cnt/20:.1%})",
            "Hegel_M6_9_A_Synthesized_2x_native": f"{synth_2x_cnt}/20 ({synth_2x_cnt/20:.1%})",
            "Oracle_Thesis_Union_Antithesis_2x_native": f"{oracle_hegel_2x_cnt}/20 ({oracle_hegel_2x_cnt/20:.1%})",
            "synthesis_selection_regret_vs_branch_oracle": oracle_hegel_2x_cnt - synth_2x_cnt,
            "false_promotions_vs_thesis": false_promos,
        },
        "rw_lock_metering": {
            "total_read_wait_ms": rw_lock.total_read_wait_ms,
            "total_write_wait_ms": rw_lock.total_write_wait_ms,
        },
        "skill_infinity_runtime_telemetry": {
            "tb_science_episodes_total": len(tb_m6_9_eps),
            "tb_science_episodes_activating_wormhole_code": len(eps_with_skill_calls),
            "activating_task_ids": sorted(eps_with_skill_calls),
            "total_runtime_function_calls": total_skill_calls,
            "wave1_ast_sublated_module_count": len(ast_events),
        },
        "biomystery_judge_total_tokens_m6_8": bm_judge_tokens_total,
        "regrade_receipts_applied": regrade_receipts_applied,
        "quota_paused_models": quota_paused,
        "total_study_wall_sec": round(monotonic() - study_t0, 2),
        "native_verifier_aggregate": native_agg,
        "declared_regrade_receipt_aggregate": regrade_agg,
        "aggregate": native_agg,
    }
    (ev_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_sha256sums(ev_dir)
    print(f"=== Hegel M6.9-A Two-Branch Study Complete in {manifest['total_study_wall_sec']}s ===", flush=True)
    print(json.dumps(manifest["matched_1x_and_2x_budget_scoreboard"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
