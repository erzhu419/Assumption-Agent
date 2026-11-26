#!/usr/bin/env python3
"""Hegel M9.2 Equal-Information On-Demand Reuse Runner (`3 Arms [B, K, E] x 4 Tasks x 2 Replicates = 24 Single-Branch Episodes`).

Executes on a SINGLE frozen clean commit with `<= 32` concurrent `gpt-6.2-sol` Codex CLI workers:
- Single-Branch Locked (`direct 1x run_frontier_episode`, `selective_second_branch` disabled, `max_turns = 18` symmetric across all 3 arms!).
- Equal-Information Deliverable Schema Parity (`/holdout_task/validate_public_schema.py` mounted inside EVERY arm's container!).
- Arms:
  1. `bare_no_library` (`B`): `0` skills mounted, `0` Turn-1 manifest bytes, shared public schema validator.
  2. `on_demand_knowledge_only` (`K`): `0` Turn-1 manifest bytes, `/tmp/hegel_knowledge_ref` (`describe_ref.py --search/--describe` + `.reference.txt`), `0` importable Python package.
  3. `on_demand_executable_library` (`E`): `0` Turn-1 manifest bytes, matched Turn-1 system prompt length vs `K`, `/tmp/hegel_wormhole_skills` (`v6` 4-tuple verified `L1` library) + static AST dependency closure packager!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
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
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m9_2_equal_info_reuse_benchmark import get_equal_info_m92_frontier_tasks
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

ARMS_CONFIG: tuple[tuple[str, str, str, bool], ...] = (
    # (arm_id, disclosure_mode, staging_policy, enable_auto_skill_packaging)
    ("bare_no_library", "none", "none", False),
    ("on_demand_knowledge_only", "on_demand_knowledge_only", "net_value_gated", False),
    ("on_demand_executable_library", "on_demand_executable", "net_value_gated", True),
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


def _git_and_source_tree_state(ev_rel_prefix: str) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    status_lines = [
        ln
        for ln in subprocess.check_output(["git", "status", "--porcelain", "."], cwd=str(REPO_ROOT), text=True).splitlines()
        if ln.strip() and ev_rel_prefix not in ln
    ]
    h = hashlib.sha256()
    for sub in ("src", "scripts", "tests"):
        root_s = REPO_ROOT / sub
        for p in sorted(root_s.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            h.update(f"{rel}\0".encode("utf-8"))
            h.update(p.read_bytes())
            h.update(b"\0")
    return {
        "git_head": head,
        "dirty_lines": status_lines,
        "source_tree_py_sha256": h.hexdigest(),
    }


def _extract_turn1_tokens(ep_dir: Path) -> dict[str, int]:
    dag_fp = ep_dir / "01_decision_provenance_dag.json"
    if not dag_fp.is_file():
        return {"turn1_prompt_tokens": 0, "turn1_cached_tokens": 0, "turn1_completion_tokens": 0}
    dag = json.loads(dag_fp.read_text(encoding="utf-8"))
    t1 = next((r for r in dag.get("records", []) if str(r.get("decision_type", "")).startswith(("D_TURN_01", "D2_TURN_01"))), None)
    if not t1:
        return {"turn1_prompt_tokens": 0, "turn1_cached_tokens": 0, "turn1_completion_tokens": 0}
    cd = t1.get("cost_ledger_delta") or t1
    return {
        "turn1_prompt_tokens": int(cd.get("prompt_tokens_delta", 0)),
        "turn1_cached_tokens": int(cd.get("cached_tokens_delta", 0)),
        "turn1_completion_tokens": int(cd.get("completion_tokens_delta", 0)),
    }


def _extract_verifier_metrics(ep_dir: Path) -> dict[str, Any]:
    ver_fp = ep_dir / "03_official_verifier_report.json"
    if not ver_fp.is_file():
        return {}
    ver = json.loads(ver_fp.read_text(encoding="utf-8"))
    parsed = ver.get("verifier_parsed_json")
    if isinstance(parsed, dict):
        return parsed
    out_s = str(ver.get("verifier_stdout") or "")
    for ln in reversed(out_s.splitlines()):
        s = ln.strip()
        if s.startswith("{") and "strict_pass" in s:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    return {}


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
    parser = argparse.ArgumentParser(description="Run Hegel M9.2 Equal-Information On-Demand Reuse Experiment")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m9_2_equal_information_on_demand_reuse_20261126",
    )
    parser.add_argument("--model", type=str, default="gpt-6.2-sol")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--num-replicates", type=int, default=2)
    parser.add_argument("--max-episode-workers", type=int, default=24)
    parser.add_argument("--max-codex-concurrency", type=int, default=32)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    t_global0 = monotonic()
    ev_dir: Path = args.evidence_dir.resolve()
    ev_rel = ev_dir.relative_to(REPO_ROOT).as_posix() if ev_dir.is_relative_to(REPO_ROOT) else ev_dir.name
    start_state = _git_and_source_tree_state(ev_rel)
    if start_state["dirty_lines"] and not args.allow_dirty:
        raise RuntimeError(f"Refusing to start M9.2 on dirty working tree: {start_state['dirty_lines']}")

    ev_dir.mkdir(parents=True, exist_ok=True)
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt", f"Official ChatGPT CLI required: {codex_info}"
    ensure_clean_scipy_sandbox_image()
    codex_stats = _install_codex_concurrency_limiter(args.max_codex_concurrency)

    v6_src_fp = (
        REPO_ROOT
        / "docs/evidence/m9_1_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v6_m911.json"
    )
    v6_sha = hashlib.sha256(v6_src_fp.read_bytes()).hexdigest()
    v6_forest, v6_archive, _, _ = m7a.load_autogenous_forest_bundle(v6_src_fp)

    # Fail-closed 4-tuple qualification pre-flight check on `v6`
    verified_v6_stems: list[str] = []
    for _, rec in m7a.iter_live_archive_nodes(v6_archive):
        for m in rec.get("autogenous_wormhole_modules") or []:
            fn = str(m.get("filename") or "")
            if fn.startswith("autogenous_prim_") and fn not in verified_v6_stems:
                ok_q, rsn_q, _ = m64.verify_l1_module_qualification_record(m, require_4tuple_protocol=True)
                assert ok_q, f"v6 4-tuple check failed for {fn}: {rsn_q}"
                verified_v6_stems.append(fn)
    assert len(verified_v6_stems) == 6

    print(
        f"[M9.2] Frozen Commit: {start_state['git_head']} | TreeSHA={start_state['source_tree_py_sha256'][:12]} | "
        f"v6SHA={v6_sha[:12]} | Model={args.model} | MaxTurns={args.max_turns} (Single-Branch Locked) | MaxCodex={args.max_codex_concurrency}",
        flush=True,
    )

    eval_tasks = get_equal_info_m92_frontier_tasks()
    assert len(eval_tasks) == 4

    pre_run_matrix: dict[str, Any] = {}
    for t_obj in eval_tasks:
        _, man_k, diag_k = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v6_archive,
            task_query_text=t_obj.instruction_text, disclosure_mode="on_demand_knowledge_only", staging_policy="net_value_gated", return_diagnostics=True,
        )
        _, man_e, diag_e = m64.collect_recalled_executable_wormhole_sources(
            [], include_foundational_skills=False, allow_l1_primitives=True, full_forest_archive=v6_archive,
            task_query_text=t_obj.instruction_text, disclosure_mode="on_demand_executable", staging_policy="net_value_gated", return_diagnostics=True,
        )
        sys_b, pay_b = m6f._build_system_and_initial_brief(
            t_obj, "Hegel_M6_9_Antithesis_LiveGPT", env_boot_info="linux", recalled_m6_notes=[],
            staged_wormhole_manifest=[], max_turns=args.max_turns, cmd_timeout_sec=180, disclosure_mode="dedup_full", has_dormant_on_demand_skills=False,
        )
        sys_k, pay_k = m6f._build_system_and_initial_brief(
            t_obj, "Hegel_M6_9_Antithesis_LiveGPT", env_boot_info="linux", recalled_m6_notes=[{"node_id": "n1", "autogenous_only": True, "methodology_principle": "p"}],
            staged_wormhole_manifest=[], max_turns=args.max_turns, cmd_timeout_sec=180, disclosure_mode="on_demand_knowledge_only", has_dormant_on_demand_skills=True,
        )
        sys_e, pay_e = m6f._build_system_and_initial_brief(
            t_obj, "Hegel_M6_9_Antithesis_LiveGPT", env_boot_info="linux", recalled_m6_notes=[{"node_id": "n1", "autogenous_only": True, "methodology_principle": "p"}],
            staged_wormhole_manifest=[], max_turns=args.max_turns, cmd_timeout_sec=180, disclosure_mode="on_demand_executable", has_dormant_on_demand_skills=True,
        )
        assert man_k == [] and man_e == []
        assert json.dumps(pay_b) == json.dumps(pay_k) == json.dumps(pay_e)
        pre_run_matrix[t_obj.task_id] = {
            "bare_system_chars": len(sys_b),
            "knowledge_system_chars": len(sys_k),
            "executable_system_chars": len(sys_e),
            "init_payload_chars_identical_all_arms": len(json.dumps(pay_b)),
            "recommended_l1_count": diag_e["demand_triggered_recommended_l1_count"],
            "recommended_l1_modules": diag_e["selected_l1_modules"],
            "recommended_cards": diag_e["demand_triggered_recommended_cards"],
        }
    (ev_dir / "m9_2_pre_run_equal_information_matrix.json").write_text(
        json.dumps(pre_run_matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    replicates = [f"rep_{r_idx:02d}" for r_idx in range(1, int(args.num_replicates) + 1)]
    jobs: list[tuple[str, m6f.FrontierBenchmarkTask, str, str, str, bool]] = [
        (rep_id, t_obj, arm_id, disc_mode, stg_pol, pkg_flag)
        for rep_id in replicates
        for t_obj in eval_tasks
        for arm_id, disc_mode, stg_pol, pkg_flag in ARMS_CONFIG
    ]
    episodes_root = ev_dir / "episodes"
    resumed_lock = threading.Lock()
    run_counters = {"resumed_episode_count": 0, "fresh_executed_episode_count": 0}

    print(f"[M9.2] Launching {len(jobs)} interleaved single-branch episodes across {args.max_episode_workers} workers...", flush=True)

    def _run_single_job(
        job: tuple[str, m6f.FrontierBenchmarkTask, str, str, str, bool],
    ) -> tuple[str, str, str, dict[str, Any]]:
        rep_id, task_obj, arm_id, disc_mode, stg_pol, pkg_flag = job
        ep_dir = episodes_root / rep_id / arm_id / task_obj.task_id
        sum_fp = ep_dir / "00_episode_summary.json"
        if sum_fp.is_file():
            with resumed_lock:
                run_counters["resumed_episode_count"] += 1
            return rep_id, arm_id, task_obj.task_id, json.loads(sum_fp.read_text(encoding="utf-8"))

        if arm_id == "bare_no_library":
            ep_forest = PhilosophicalForest()
            ep_archive: dict[str, dict[str, Any]] = {}
            allow_l1 = False
            eff_disc = "dedup_full"
            eff_stg = "all_compatible"
        else:
            ep_forest = deepcopy(v6_forest)
            ep_archive = deepcopy(v6_archive)
            ep_archive["__autogenous_blank_seed__"] = {"enabled": True}
            allow_l1 = True
            eff_disc = disc_mode
            eff_stg = stg_pol

        ep_hippo = HippoRAGWormholeIndex(ep_forest)
        res = m6f.run_frontier_episode(
            task_obj,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=args.model,
            codex_info=codex_info,
            forest=ep_forest,
            forest_archive=ep_archive,
            hippo_idx=ep_hippo,
            episode_out_dir=ep_dir,
            run_id=f"m9_2_{rep_id}_{arm_id}",
            max_turns=args.max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=args.reasoning_effort,
            defer_official_verifier=False,
            enable_auto_skill_packaging=pkg_flag,
            allow_l1_primitives=allow_l1,
            disclosure_mode=eff_disc,
            staging_policy=eff_stg,
        )
        with resumed_lock:
            run_counters["fresh_executed_episode_count"] += 1
        print(
            f"  [DONE] {rep_id} | {arm_id:30s} | {task_obj.short_name:42s} | "
            f"pass={res['strict_pass']} | turns={res['turns_used']} | tok={res['total_tokens']}",
            flush=True,
        )
        return rep_id, arm_id, task_obj.task_id, res

    results_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_episode_workers))) as pool:
        futs = [pool.submit(_run_single_job, jb) for jb in jobs]
        for fut in as_completed(futs):
            rep_id, arm_id, tid, res_d = fut.result()
            results_by_key[(rep_id, arm_id, tid)] = res_d

    episode_rows: list[dict[str, Any]] = []
    for rep_id in replicates:
        for t_obj in eval_tasks:
            for arm_id, disc_mode, stg_pol, _ in ARMS_CONFIG:
                ep_dir = episodes_root / rep_id / arm_id / t_obj.task_id
                s = results_by_key[(rep_id, arm_id, t_obj.task_id)]
                t1_tok = _extract_turn1_tokens(ep_dir)
                ver_m = _extract_verifier_metrics(ep_dir)
                af = s.get("skill_adoption_funnel") or {}
                ret = af.get("final_selected_artifact_retention") or {}
                p_tok = int(s.get("prompt_tokens", 0))
                c_tok = int(s.get("cached_tokens", 0))
                o_tok = int(s.get("completion_tokens", 0))
                proxy_tok = float((p_tok - c_tok) + 0.25 * c_tok + o_tok)
                if "inspected_knowledge_ref_in_bash" not in af:
                    raise KeyError(f"Missing required `inspected_knowledge_ref_in_bash` in `skill_adoption_funnel` for {ep_dir}")
                pa = s.get("provider_attempt_accounting") or {}
                psr = s.get("public_setup_receipt") or {}
                episode_rows.append(
                    {
                        "replicate": rep_id,
                        "arm": arm_id,
                        "disclosure_mode": disc_mode,
                        "staging_policy": stg_pol,
                        "task_id": t_obj.task_id,
                        "short_name": t_obj.short_name,
                        "is_negative_control": t_obj.task_id.startswith("eq04"),
                        "strict_pass": bool(s.get("strict_pass", False)),
                        "turns_used": int(s.get("turns_used", 0)),
                        "max_turns": int(s.get("max_turns", args.max_turns)),
                        "ended_by_finish": bool(s.get("ended_by_finish", False)),
                        "public_setup_executed": bool(psr.get("executed", False)),
                        "shared_public_schema_validator_present": bool(psr.get("shared_public_schema_validator_present", False)),
                        "turn1_staged_manifest_json_chars": len(json.dumps(s.get("staged_wormhole_modules", []), ensure_ascii=False)),
                        **t1_tok,
                        "raw_token_triple": {
                            "input_tokens": p_tok,
                            "cached_input_tokens": c_tok,
                            "output_tokens": o_tok,
                        },
                        "total_tokens": int(s.get("total_tokens", 0)),
                        "token_cost_proxy_w025": round(proxy_tok, 2),
                        "provider_attempt_count": int(pa.get("total_provider_attempts", s.get("turns_used", 0))),
                        "usage_unknown_attempt_count": int(pa.get("usage_unknown_attempt_count", 0)),
                        "cost_accounting_status": str(pa.get("cost_accounting_status") or "complete"),
                        "episode_wall_ms": float(s.get("wall_ms", 0.0)),
                        "verifier_window_wall_sec": ver_m.get("student_elapsed_sec"),
                        "candidate_timing_breakdown": ver_m.get("candidate_timing_breakdown"),
                        "demand_triggered_recommended_l1_count": int(af.get("demand_triggered_recommended_l1_count", 0)),
                        "demand_skill_hint_emitted": bool(af.get("demand_skill_hint_emitted", False)),
                        "demand_skill_hint_turn": af.get("demand_skill_hint_turn"),
                        "search_skill_query_count": int(af.get("search_skill_query_count", 0)),
                        "describe_skill_inspection_count": int(af.get("describe_skill_inspection_count", 0)),
                        "describe_skill_queries": list(af.get("describe_skill_queries", [])),
                        "inspected_knowledge_ref_in_bash": bool(af["inspected_knowledge_ref_in_bash"]),
                        "dynamic_l1_calls": int(af.get("dynamic_l1_primitive_activation_count", 0)),
                        "dynamic_unique_l1_modules": list(af.get("dynamic_unique_l1_modules_called", [])),
                        "dynamic_unique_l1_functions": list(af.get("dynamic_unique_l1_functions_called", [])),
                        "closure_exact_present_l1_functions": list(ret.get("closure_exact_present_l1_functions", [])),
                        "entrypoint_reachable_l1_functions": list(ret.get("entrypoint_reachable_l1_functions", [])),
                        "verifier_metrics": ver_m,
                    }
                )

    def _agg_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(1, len(rows))
        rep_rates = [
            sum(1 for r in rows if r["replicate"] == rp and r["strict_pass"]) / max(1, sum(1 for r in rows if r["replicate"] == rp))
            for rp in replicates
        ]
        in_tok = sum(int(r["raw_token_triple"]["input_tokens"]) for r in rows)
        ca_tok = sum(int(r["raw_token_triple"]["cached_input_tokens"]) for r in rows)
        ou_tok = sum(int(r["raw_token_triple"]["output_tokens"]) for r in rows)
        stud_secs = [float(r["verifier_window_wall_sec"]) for r in rows if r.get("verifier_window_wall_sec") is not None]
        cand_cpus = [
            float((r.get("candidate_timing_breakdown") or {}).get("candidate_call_cpu_sec"))
            for r in rows
            if (r.get("candidate_timing_breakdown") or {}).get("candidate_call_cpu_sec") is not None
        ]
        unk_cnt = sum(int(r["usage_unknown_attempt_count"]) for r in rows)
        return {
            "episode_count": len(rows),
            "strict_pass_count": sum(1 for r in rows if r["strict_pass"]),
            "strict_pass_rate": round(sum(1 for r in rows if r["strict_pass"]) / n, 4),
            "replicate_pass_rates": [round(x, 4) for x in rep_rates],
            "replicate_sd": round(statistics.stdev(rep_rates), 4) if len(rep_rates) > 1 else 0.0,
            "mean_turns_used": round(sum(r["turns_used"] for r in rows) / n, 2),
            "mean_turn1_prompt_tokens": round(sum(r["turn1_prompt_tokens"] for r in rows) / n, 1),
            "raw_token_triple_sum": {"input_tokens": in_tok, "cached_input_tokens": ca_tok, "output_tokens": ou_tok},
            "total_tokens_sum": sum(r["total_tokens"] for r in rows),
            "mean_total_tokens": round(sum(r["total_tokens"] for r in rows) / n, 1),
            "token_cost_proxy_w025_sum": round(sum(r["token_cost_proxy_w025"] for r in rows), 1),
            "total_provider_attempts": sum(int(r["provider_attempt_count"]) for r in rows),
            "usage_unknown_attempt_count": unk_cnt,
            "cost_accounting_status": ("partial_unknown_usage" if unk_cnt > 0 else "complete"),
            "mean_episode_wall_sec": round(sum(r["episode_wall_ms"] for r in rows) / (1000.0 * n), 2),
            "mean_verifier_window_wall_sec": round(sum(stud_secs) / len(stud_secs), 4) if stud_secs else None,
            "mean_candidate_call_cpu_sec": round(sum(cand_cpus) / len(cand_cpus), 4) if cand_cpus else None,
            "episodes_with_search_or_describe": sum(
                1 for r in rows if (r["search_skill_query_count"] + r["describe_skill_inspection_count"]) > 0 or r["inspected_knowledge_ref_in_bash"]
            ),
            "total_search_queries": sum(r["search_skill_query_count"] for r in rows),
            "total_describe_inspections": sum(r["describe_skill_inspection_count"] for r in rows),
            "episodes_reading_knowledge_ref_txt": sum(1 for r in rows if r["inspected_knowledge_ref_in_bash"]),
            "episodes_with_dynamic_l1_calls": sum(1 for r in rows if r["dynamic_l1_calls"] > 0),
            "total_dynamic_l1_calls": sum(r["dynamic_l1_calls"] for r in rows),
            "episodes_with_entrypoint_reachable_l1": sum(1 for r in rows if len(r["entrypoint_reachable_l1_functions"]) > 0),
        }

    arm_aggregates: dict[str, Any] = {}
    for arm_id, disc_mode, stg_pol, pkg_flag in ARMS_CONFIG:
        a_rows = [r for r in episode_rows if r["arm"] == arm_id]
        pos_rows = [r for r in a_rows if not r["is_negative_control"]]
        neg_rows = [r for r in a_rows if r["is_negative_control"]]
        eq01_rows = [r for r in a_rows if r["task_id"].startswith("eq01")]
        arm_aggregates[arm_id] = {
            "disclosure_mode": disc_mode,
            "staging_policy": stg_pol,
            "enable_auto_skill_packaging": pkg_flag,
            "all_8_episodes": _agg_subset(a_rows),
            "reuse_eligible_6_episodes_eq01_to_eq03": _agg_subset(pos_rows),
            "negative_control_2_episodes_eq04": _agg_subset(neg_rows),
            "eq01_3d_continuation_2_episodes": _agg_subset(eq01_rows),
        }

    end_state = _git_and_source_tree_state(ev_rel)
    total_wall_sec = round(monotonic() - t_global0, 2)
    master_summary = {
        "experiment_id": "hegel_m9_2_equal_information_on_demand_reuse_20261126",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "code_commit_at_start": start_state["git_head"],
            "code_commit_at_end": end_state["git_head"],
            "same_frozen_commit_throughout": start_state["git_head"] == end_state["git_head"],
            "source_tree_py_sha256_at_start": start_state["source_tree_py_sha256"],
            "source_tree_py_sha256_at_end": end_state["source_tree_py_sha256"],
            "zero_source_mutation_during_run": start_state["source_tree_py_sha256"] == end_state["source_tree_py_sha256"],
            "git_dirty_at_start": bool(start_state["dirty_lines"]),
            "git_dirty_at_end": bool(end_state["dirty_lines"]),
            "resumed_episode_count": run_counters["resumed_episode_count"],
            "fresh_executed_episode_count": run_counters["fresh_executed_episode_count"],
            "single_clean_unresumed_invocation": run_counters["resumed_episode_count"] == 0 and run_counters["fresh_executed_episode_count"] == len(jobs),
            "frozen_v6_bundle_sha256": v6_sha,
            "qualification_protocol": "hegel_l1_qual_4tuple_v2",
            "single_branch_locked": True,
            "shared_public_schema_validator_all_arms": bool(
                episode_rows and all(r.get("shared_public_schema_validator_present") for r in episode_rows)
            ),
            "public_setup_executed_all_episodes": bool(
                episode_rows and all(r.get("public_setup_executed") for r in episode_rows)
            ),
        },
        "protocol": {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "auth_mode": str(codex_info["auth_mode"]),
            "provider_backend": str(codex_info["provider_backend"]),
            "max_turns": args.max_turns,
            "num_replicates": args.num_replicates,
            "interleaved_submission_order": True,
            "zero_artificial_solver_bans": True,
        },
        "concurrency_telemetry": {
            "configured_max_episode_workers": args.max_episode_workers,
            "configured_max_codex_concurrency": args.max_codex_concurrency,
            "peak_concurrent_codex_calls": codex_stats["peak_concurrent_codex_calls"],
            "total_codex_calls": codex_stats["total_codex_calls"],
            "total_wall_clock_sec": total_wall_sec,
        },
        "arm_aggregates": arm_aggregates,
        "episode_rows": episode_rows,
    }
    (ev_dir / "m9_2_equal_info_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_hashed = _write_sha256sums(ev_dir)
    print(
        f"[M9.2 COMPLETE] Wall={total_wall_sec}s | PeakCodex={codex_stats['peak_concurrent_codex_calls']} | HashedFiles={n_hashed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
