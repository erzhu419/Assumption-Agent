#!/usr/bin/env python3
"""Hegel M6.9.2 Targeted Live Verification on `tb__virtual-baseline-localization` (`19/20` Frontier Test).

Protocol guarantees:
  1. Official OpenAI Codex CLI (`auth_mode == 'chatgpt'`, `gpt-6.2-sol`, zero third-party proxy).
  2. Pure `Antithesis-only` (`1x`) Wave-1 sublation read-only from sealed `m6_9_1_a` Wave-1 `branch_2_antithesis_wormhole`
     (zero mutation of historical `m6_9_a` or `m6_9_1_a` evidence bundles).
  3. Full `28 + 28` turn Two-Branch Dialectical Synthesis (`Thesis + Antithesis -> Blind Public Synthesis`) with
     official air-gapped verifier deferred strictly until AFTER synthesis locks its candidate (`P0c` / `E04`).
  4. Reader-Writer `HeavyTimingRWLock` enforced + cryptographic `SHA256SUMS.txt` (excluding `.pyc`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
from hegel_machine.cognition.m5_live_gpt_and_provenance import discover_chatgpt_subscription_codex_cli
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
from run_m6_frontier_scout_study import _write_sha256sums  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_9_2_virtual_baseline_targeted_20261123",
    )
    parser.add_argument(
        "--sealed-wave1-root",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_9_1_a_two_branch_frontier_40ep_20261123",
    )
    parser.add_argument("--task-id", type=str, default="tb__virtual-baseline-localization")
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--max-turns-per-branch", type=int, default=28)
    parser.add_argument("--cmd-timeout", type=int, default=180)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
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
    hegel_arm = "Hegel_M6_9_2_TwoBranch_LiveGPT"

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
    all_tasks = {t.task_id: t for w in waves.values() for t in w}
    if args.task_id not in all_tasks:
        print(f"ERROR: Unknown task_id {args.task_id}", file=sys.stderr)
        return 2
    target_task = all_tasks[args.task_id]

    f_obj, f_arch, m_dag = m6f.build_m6_frontier_general_forest()
    h_idx = HippoRAGWormholeIndex(f_obj)

    # Pure `Antithesis-only` (`1x`) Wave-1 consolidation from sealed `m6_9_1_a` (read-only!)
    sealed_ep_root = args.sealed_wave1_root / "episodes"
    sealed_br_root = args.sealed_wave1_root / "branch_trajectories"
    w1_source_pairs: list[tuple[dict[str, Any], Path]] = []
    for w1_task in waves["wave_1"]:
        top_sum_f = sealed_ep_root / f"wave_1__{model}__Hegel_M6_9_A_TwoBranch_LiveGPT__{w1_task.task_id}" / "00_episode_summary.json"
        anti_d = sealed_br_root / f"wave_1__{model}__{w1_task.task_id}" / "branch_2_antithesis_wormhole"
        anti_sf = anti_d / "00_episode_summary.json"
        if not top_sum_f.is_file() or not anti_sf.is_file():
            raise FileNotFoundError(f"Missing sealed Wave-1 receipt for {w1_task.task_id}: {anti_sf}")
        top_s = json.loads(top_sum_f.read_text(encoding="utf-8"))
        anti_s = json.loads(anti_sf.read_text(encoding="utf-8"))
        anti_s["strict_pass"] = bool((top_s.get("postlock_branch_outcomes") or {}).get("antithesis_wormhole_strict_pass", False))
        w1_source_pairs.append((anti_s, anti_d))

    c_rec = m6f.consolidate_wave1_into_m6_forest(
        wave1_m6_summaries=[p[0] for p in w1_source_pairs],
        forest=f_obj,
        archive=f_arch,
        meta_dag=m_dag,
    )
    ast_events = consolidate_wave1_executable_modules_into_forest(
        wave1_episode_dirs=w1_source_pairs,
        forest=f_obj,
        archive=f_arch,
    )
    manifest_reconcile = consolidate_wave1_skill_manifests_with_reconciliation(
        wave1_episode_dirs=w1_source_pairs,
        forest=f_obj,
        archive=f_arch,
    )
    c_rec["wave1_consolidation_source"] = "antithesis_only"
    c_rec["wave1_ast_executable_sublation_events"] = ast_events
    c_rec["wave1_full_manifest_and_m5_reconciliation"] = manifest_reconcile
    (ev_dir / f"wave1_antithesis_only_consolidation_{model}.json").write_text(
        json.dumps(c_rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"=== Consolidated {c_rec['consolidated_episode_count']} Wave-1 Antithesis-only (`1x`) episodes + "
        f"{len(manifest_reconcile['wave1_manifest_events'])} producer manifests + "
        f"{len(manifest_reconcile['registered_shared_base_subtrees'])} SharedBaseSubtrees ===",
        flush=True,
    )

    lib_code_sha = hashlib.sha256(
        json.dumps(
            {
                "foundational_skills": {k: v[1] for k, v in sorted(NODE_TO_EXECUTABLE_WORMHOLE_MODULE.items())},
                "skill_07": SKILL_07_MULTISCALE_RECONCILIATION,
                "host_gate": HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY,
                "public_synth_bench": PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY,
                "cmd_timeout": args.cmd_timeout,
                "reasoning_effort": args.reasoning_effort,
                "wave1_consolidation_source": "antithesis_only",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    t0 = monotonic()
    ep_dir = ep_root / f"{target_task.wave}__{model}__{hegel_arm}__{target_task.task_id}"
    print(f"=== Running Hegel M6.9.2 Live Verification on `{target_task.task_id}` (max_turns={args.max_turns_per_branch}/branch) ===", flush=True)
    res = run_m6_9_a_two_branch_episode(
        target_task,
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
    res["arm"] = hegel_arm
    res["lib_code_sha256"] = lib_code_sha
    (ep_dir / "00_episode_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pb = res["postlock_branch_outcomes"]
    sd = res["synthesis_decision"]
    summary_payload = {
        "study_id": ev_dir.name,
        "model": model,
        "codex_cli_info": codex_info,
        "task_id": target_task.task_id,
        "wave1_consolidation_source": "antithesis_only",
        "synthesis_strict_pass": bool(res["strict_pass"]),
        "thesis_bare_run2_strict_pass": bool(pb["thesis_bare_run2_strict_pass"]),
        "antithesis_wormhole_strict_pass": bool(pb["antithesis_wormhole_strict_pass"]),
        "selected_branch": sd["selected_branch"],
        "selection_mechanism": sd["selection_mechanism"],
        "thesis_public_metrics": sd.get("thesis_public_metrics"),
        "antithesis_public_metrics": sd.get("antithesis_public_metrics"),
        "skill_activation_count": res.get("skill_activation_count", 0),
        "staged_wormhole_modules": res.get("staged_wormhole_modules", []),
        "token_accounting": {
            "prompt_tokens_fresh": int(res["prompt_tokens"]) - int(res["cached_tokens"]),
            "cached_tokens": int(res["cached_tokens"]),
            "completion_tokens": int(res["completion_tokens"]),
            "total_context_processed": int(res["prompt_tokens"]) + int(res["completion_tokens"]),
        },
        "elapsed_wall_sec": round(monotonic() - t0, 2),
    }
    (ev_dir / "targeted_verification_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    sha_count = _write_sha256sums(ev_dir)
    print(
        f"=== COMPLETED `{target_task.task_id}` -> SYNTH={res['strict_pass']} "
        f"(Thesis={pb['thesis_bare_run2_strict_pass']}, Antithesis={pb['antithesis_wormhole_strict_pass']}, "
        f"pick={sd['selected_branch']} via {sd['selection_mechanism']}) | sha_files={sha_count} | elapsed={summary_payload['elapsed_wall_sec']}s ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
