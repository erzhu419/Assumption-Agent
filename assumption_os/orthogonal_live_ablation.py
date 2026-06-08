"""Live downstream ablation for orthogonal multi-cluster proposals.

This module intentionally reads credentials only from the process environment.
It does not serialize keys into commands, artifacts, or source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .orthogonal_multi_cluster import DEFAULT_OUT as DEFAULT_QUEUE_PATH
from .orthogonal_multi_cluster import DEFAULT_GRAPH_DIR
from .recursive_daemon import build_preflight_queue_daemon_payload
from .recursive_executor import JudgmentSet
from .schema import EdgeType
from .structural_live_ablation import (
    PAIRWISE_JUDGE_PROMPT,
    _call_with_retry,
    _make_judge_client,
    _parse_judge_json,
)


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
CACHE_DIR = Path("phase two/analysis/cache")
ANSWERS_DIR = CACHE_DIR / "answers"
JUDGMENTS_DIR = CACHE_DIR / "judgments"
SAMPLES_DIR = CACHE_DIR / "proposal_samples"
DEFAULT_OUT = PAPER_DIR / "orthogonal_multi_cluster_live_ablation_20260608.json"
BASELINE_VARIANT = "phase2_v20_gpt54mini_prop_union"


def build_orthogonal_live_ablation_payload(
    *,
    root: Path,
    queue_path: Path | None = None,
    graph_dir: Path | None = None,
    eval_id: str | None = None,
    execute_answers: bool = False,
    run_judge: bool = False,
    answer_workers: int = 3,
    judge_workers: int = 3,
    judge_model: str = "gpt55",
    baseline_variant: str = BASELINE_VARIANT,
    route_scoped_noop_controls: bool = False,
    apply_accepted: bool = False,
) -> dict[str, Any]:
    """Run or plan a live answer-quality ablation for the multi-cluster queue."""

    root = root.resolve()
    eval_id = eval_id or "orthogonal_multi_cluster_live_ablation_20260608"
    queue_path = _resolve(root, queue_path or DEFAULT_QUEUE_PATH)
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    queue = _load_json(queue_path)
    proposals = queue["proposal_payload"]["proposals"]
    preflight = queue["preflight_payload"]
    evolution = queue["daemon_validation"]["evolution_payload"]
    sample_by_pid = _sample_by_pid(root)
    proposal_ids = [p["proposal_id"] for p in proposals]
    sample_specs = _write_live_sample_files(
        root=root,
        eval_id=eval_id,
        preflight=preflight,
        sample_by_pid=sample_by_pid,
        proposal_ids=proposal_ids,
        answer_triggers_only=route_scoped_noop_controls,
    )
    env = _env_status()
    answer_results = _run_answer_generation(
        root=root,
        queue=queue,
        sample_specs=sample_specs,
        execute=execute_answers,
        max_workers=answer_workers,
    )
    judgment_results = _run_pairwise_judges(
        root=root,
        eval_id=eval_id,
        preflight=preflight,
        sample_by_pid=sample_by_pid,
        proposal_ids=proposal_ids,
        run_judge=run_judge,
        judge_workers=judge_workers,
        judge_model=judge_model,
        baseline_variant=baseline_variant,
        route_scoped_noop_controls=route_scoped_noop_controls,
    )
    readback = _run_readback(
        root=root,
        graph_dir=graph_dir,
        preflight=preflight,
        evolution=evolution,
        judgment_results=judgment_results,
        proposal_ids=proposal_ids,
        eval_id=eval_id,
        baseline_variant=baseline_variant,
        apply_accepted=apply_accepted,
    )
    metrics = _metrics(
        proposal_ids=proposal_ids,
        sample_specs=sample_specs,
        answer_results=answer_results,
        judgment_results=judgment_results,
        readback=readback,
        env=env,
    )
    gates = {
        "live_env_ready_for_gpt": metrics["live_env_ready"],
        "answer_generation_completed_or_planned": (
            not execute_answers or metrics["answer_success_count"] == metrics["proposal_count"]
        ),
        "judge_completed_or_planned": (
            not run_judge or metrics["judged_problem_count"] == metrics["expected_judged_problem_count"]
        ),
        "readback_has_acceptance_when_judged": (
            not run_judge or sum(metrics["candidate_acceptance_counts"].values()) == metrics["proposal_count"]
        ),
        "readback_without_apply_does_not_mutate_graph": (
            not run_judge or (metrics["readback_applied_count"] == 0 and not metrics["node_mutation_without_apply"])
        ),
        "commands_are_secret_free": _commands_are_secret_free(answer_results),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_multi_cluster_live_downstream_ablation",
        "performance_validation": bool(execute_answers and run_judge),
        "validation_scope": (
            "live candidate answer generation plus GPT-5.5 pairwise judging over trigger/control rows; "
            "graph mutation remains gated and disabled unless --apply-accepted is explicit"
            + (
                "; route-scoped controls are recorded as no-op ties because the candidate route is inactive there"
                if route_scoped_noop_controls
                else ""
            )
        ),
        "status": _status(execute_answers=execute_answers, run_judge=run_judge, metrics=metrics),
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "queue_path": _display_path(root, queue_path),
            "graph_dir": _display_path(root, graph_dir),
            "baseline_variant": baseline_variant,
        },
        "mode": {
            "execute_answers": execute_answers,
            "run_judge": run_judge,
            "answer_workers": answer_workers,
            "judge_workers": judge_workers,
            "judge_model": judge_model,
            "baseline_variant": baseline_variant,
            "route_scoped_noop_controls": route_scoped_noop_controls,
            "apply_accepted": apply_accepted,
        },
        "env_status": env,
        "sample_specs": sample_specs,
        "answer_results": answer_results,
        "judgment_results": _compact_judgment_results(judgment_results),
        "readback": readback,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "interpretation": _interpretation(metrics),
    }


def _write_live_sample_files(
    *,
    root: Path,
    eval_id: str,
    preflight: dict[str, Any],
    sample_by_pid: dict[str, dict[str, Any]],
    proposal_ids: list[str],
    answer_triggers_only: bool = False,
) -> list[dict[str, Any]]:
    summary_by_id = {row["proposal_id"]: row for row in preflight.get("summaries", [])}
    out = []
    sample_dir = root / SAMPLES_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    for proposal_id in proposal_ids:
        summary = summary_by_id[proposal_id]
        trigger_pids = summary.get("trigger_problem_ids", [])
        control_pids = summary.get("control_problem_ids", [])
        judgment_pids = _unique(trigger_pids + control_pids)
        answer_pids = _unique(trigger_pids if answer_triggers_only else judgment_pids)
        rows = [sample_by_pid[pid] for pid in answer_pids if pid in sample_by_pid]
        sample_path = sample_dir / f"{eval_id}_{proposal_id}_sample.json"
        sample_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out.append({
            "proposal_id": proposal_id,
            "candidate_variant": _candidate_variant(proposal_id),
            "sample_path": _display_path(root, sample_path),
            "sample_arg": _display_path(root, sample_path.relative_to(root / CACHE_DIR) if sample_path.is_relative_to(root / CACHE_DIR) else sample_path),
            "problem_ids": judgment_pids,
            "answer_problem_ids": answer_pids,
            "trigger_problem_ids": trigger_pids,
            "control_problem_ids": control_pids,
            "n": len(rows),
            "expected_judgment_n": len(judgment_pids),
            "answer_triggers_only": answer_triggers_only,
        })
    return out


def _run_answer_generation(
    *,
    root: Path,
    queue: dict[str, Any],
    sample_specs: list[dict[str, Any]],
    execute: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        (spec, _answer_command(root=root, queue=queue, spec=spec))
        for spec in sample_specs
    ]
    if not execute:
        return [
            {
                "proposal_id": spec["proposal_id"],
                "candidate_variant": spec["candidate_variant"],
                "status": "planned",
                "command": _redacted_command(cmd),
                "returncode": None,
            }
            for spec, cmd in jobs
        ]
    env = _framework_env()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futures = {
            ex.submit(_run_answer_job, root, spec, cmd, env): spec["proposal_id"]
            for spec, cmd in jobs
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    return sorted(results, key=lambda row: row["proposal_id"])


def _run_answer_job(root: Path, spec: dict[str, Any], cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    variant = spec["candidate_variant"]
    answer_path = root / ANSWERS_DIR / f"{variant}_answers.json"
    answers = _load_json(answer_path) if answer_path.exists() else {}
    return {
        "proposal_id": spec["proposal_id"],
        "candidate_variant": variant,
        "status": "succeeded" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - t0, 3),
        "answer_path": _display_path(root, answer_path),
        "answer_count": len(answers),
        "sample_n": spec["n"],
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "command": _redacted_command(cmd),
    }


def _answer_command(*, root: Path, queue: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    source = queue.get("source", {})
    return [
        sys.executable,
        "phase one/scripts/validation/phase2_v20_framework.py",
        "--variant",
        spec["candidate_variant"],
        "--sample",
        spec["sample_arg"],
        "--n",
        str(spec["n"]),
        "--assumption-graph",
        source.get("graph_dir", "phase four/assumption_graph"),
        "--assumption-graph-skip-domains",
        "",
        "--assumption-proposals",
        source.get("proposals_out", "phase four/assumption_graph/paper_readiness_20260604/orthogonal_multi_cluster_proposals_20260608.json"),
        "--assumption-proposal-ids",
        spec["proposal_id"],
        "--assumption-force-proposal-route",
        "--assumption-route-scope-proposals",
    ]


def _run_pairwise_judges(
    *,
    root: Path,
    eval_id: str,
    preflight: dict[str, Any],
    sample_by_pid: dict[str, dict[str, Any]],
    proposal_ids: list[str],
    run_judge: bool,
    judge_workers: int,
    judge_model: str,
    baseline_variant: str,
    route_scoped_noop_controls: bool,
) -> list[dict[str, Any]]:
    baseline_answers = _load_json(root / ANSWERS_DIR / f"{baseline_variant}_answers.json")
    summary_by_id = {row["proposal_id"]: row for row in preflight.get("summaries", [])}
    results = []
    for proposal_id in proposal_ids:
        candidate_variant = _candidate_variant(proposal_id)
        candidate_answers_path = root / ANSWERS_DIR / f"{candidate_variant}_answers.json"
        candidate_answers = _load_json(candidate_answers_path) if candidate_answers_path.exists() else {}
        trigger_ids = summary_by_id[proposal_id].get("trigger_problem_ids", [])
        control_ids = summary_by_id[proposal_id].get("control_problem_ids", [])
        pids = _unique(trigger_ids + control_ids)
        judge_pids = trigger_ids if route_scoped_noop_controls else pids
        planned = {
            "proposal_id": proposal_id,
            "candidate_variant": candidate_variant,
            "baseline_variant": baseline_variant,
            "judgment_path": _display_path(root, _judgment_path(root, eval_id, proposal_id, candidate_variant, baseline_variant)),
            "expected_problem_count": len(pids),
            "judged_problem_count": 0,
            "live_judged_pair_count": 0,
            "route_scoped_noop_control_count": 0,
            "control_mode": "route_scoped_noop" if route_scoped_noop_controls else "live_pairwise",
            "winner_counts": {},
            "status": "planned",
        }
        if not run_judge:
            results.append(planned)
            continue
        missing = [
            pid for pid in judge_pids
            if not candidate_answers.get(pid) or not baseline_answers.get(pid)
        ]
        if missing:
            planned.update({
                "status": "missing_answers",
                "missing_problem_ids": missing,
            })
            results.append(planned)
            continue
        judge = _make_judge_client(judge_model, transport="requests")
        rows: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, judge_workers)) as ex:
            futures = [
                ex.submit(
                    _judge_one_pair,
                    judge,
                    judge_model,
                    sample_by_pid[pid],
                    proposal_id,
                    candidate_variant,
                    baseline_variant,
                    candidate_answers[pid],
                    baseline_answers[pid],
                )
                for pid in judge_pids
            ]
            for fut in as_completed(futures):
                pid, row = fut.result()
                row["is_trigger"] = pid in set(trigger_ids)
                row["control_mode"] = planned["control_mode"]
                rows[pid] = row
        if route_scoped_noop_controls:
            for pid in control_ids:
                rows[pid] = {
                    "winner": "tie",
                    "raw_winner": "tie",
                    "a_was": candidate_variant,
                    "b_was": baseline_variant,
                    "reasoning": (
                        "Route-scoped no-op control: candidate assumption did not route to this problem, "
                        "so production behavior should remain baseline-equivalent."
                    ),
                    "model_alias": "route_scoped_noop",
                    "model": "deterministic_noop_control",
                    "domain": sample_by_pid.get(pid, {}).get("domain", ""),
                    "difficulty": sample_by_pid.get(pid, {}).get("difficulty", ""),
                    "is_trigger": False,
                    "control_mode": "route_scoped_noop",
                }
        path = _judgment_path(root, eval_id, proposal_id, candidate_variant, baseline_variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        planned.update({
            "status": "judged",
            "judged_problem_count": len(rows),
            "live_judged_pair_count": len(judge_pids),
            "route_scoped_noop_control_count": len(control_ids) if route_scoped_noop_controls else 0,
            "winner_counts": dict(Counter(row.get("winner", "tie") for row in rows.values())),
            "judgment_path": _display_path(root, path),
        })
        results.append(planned)
    return sorted(results, key=lambda row: row["proposal_id"])


def _judge_one_pair(
    judge,
    judge_model: str,
    problem: dict[str, Any],
    proposal_id: str,
    candidate_variant: str,
    baseline_variant: str,
    candidate_answer: str,
    baseline_answer: str,
) -> tuple[str, dict[str, Any]]:
    pid = problem["problem_id"]
    swap = int(hashlib.sha1(f"{pid}:{proposal_id}:orthogonal-live".encode()).hexdigest(), 16) % 2 == 1
    a_variant, b_variant = (baseline_variant, candidate_variant) if swap else (candidate_variant, baseline_variant)
    answer_a, answer_b = (baseline_answer, candidate_answer) if swap else (candidate_answer, baseline_answer)
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        problem=problem.get("description", "")[:3000],
        reference=json.dumps(problem.get("reference_answer", {}), ensure_ascii=False)[:3000],
        answer_a=answer_a[:3500],
        answer_b=answer_b[:3500],
    )
    response = _call_with_retry(judge, prompt, max_tokens=260, temperature=0.0)
    parsed = _parse_judge_json(response.get("text", "").strip())
    raw_winner = parsed.get("winner", "tie")
    if raw_winner == "A":
        winner = a_variant
    elif raw_winner == "B":
        winner = b_variant
    else:
        winner = "tie"
    return pid, {
        "winner": winner,
        "raw_winner": raw_winner,
        "a_was": a_variant,
        "b_was": b_variant,
        "reasoning": parsed.get("reason", ""),
        "model_alias": judge_model,
        "model": response.get("model", ""),
        "domain": problem.get("domain", ""),
        "difficulty": problem.get("difficulty", ""),
    }


def _run_readback(
    *,
    root: Path,
    graph_dir: Path,
    preflight: dict[str, Any],
    evolution: dict[str, Any],
    judgment_results: list[dict[str, Any]],
    proposal_ids: list[str],
    eval_id: str,
    baseline_variant: str,
    apply_accepted: bool,
) -> dict[str, Any]:
    judged = [row for row in judgment_results if row.get("status") == "judged"]
    if not judged:
        return {
            "status": "planned",
            "candidate_acceptance_counts": {},
            "applied_candidate_node_ids": [],
            "node_mutation_without_apply": False,
        }
    judgment_sets = [
        JudgmentSet(
            candidate_variant=row["candidate_variant"],
            baseline_variant=baseline_variant,
            judgment_paths=[root / row["judgment_path"]],
            proposal_ids=[row["proposal_id"]],
        )
        for row in judged
    ]
    with tempfile.TemporaryDirectory() as td:
        temp_graph = Path(td) / "graph"
        shutil.copytree(graph_dir, temp_graph)
        before_nodes = set(JsonlGraphStore(temp_graph).nodes)
        readback = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_readback",
            queue_name="orthogonal_multi_cluster_live",
            command_limit=len(proposal_ids),
            execute=False,
            apply_accepted=False,
            writeback_manifests=True,
        )
        after_readback_nodes = set(JsonlGraphStore(temp_graph).nodes)
        applied = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight,
            evolution_payload=evolution,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_apply",
            queue_name="orthogonal_multi_cluster_live",
            command_limit=len(proposal_ids),
            execute=False,
            apply_accepted=apply_accepted,
            writeback_manifests=True,
        )
        applied_store = JsonlGraphStore(temp_graph)
        orthogonal_edges = sum(1 for edge in applied_store.edges if edge.type == EdgeType.ORTHOGONAL_TO)
    return {
        "status": "readback_complete",
        "candidate_acceptance_counts": readback.get("candidate_acceptance_counts", {}),
        "accepted_proposal_ids": readback.get("accepted_proposal_ids", []),
        "applied_candidate_node_ids": readback.get("applied_candidate_node_ids", []),
        "readback_applied_count": len(readback.get("applied_candidate_node_ids", [])),
        "node_mutation_without_apply": before_nodes != after_readback_nodes,
        "temp_apply_candidate_acceptance_counts": applied.get("candidate_acceptance_counts", {}),
        "temp_apply_accepted_proposal_ids": applied.get("accepted_proposal_ids", []),
        "temp_apply_applied_candidate_node_ids": applied.get("applied_candidate_node_ids", []),
        "temp_apply_orthogonal_edge_count": orthogonal_edges,
    }


def _metrics(
    *,
    proposal_ids: list[str],
    sample_specs: list[dict[str, Any]],
    answer_results: list[dict[str, Any]],
    judgment_results: list[dict[str, Any]],
    readback: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    acceptance_counts = readback.get("candidate_acceptance_counts", {})
    return {
        "proposal_count": len(proposal_ids),
        "expected_judged_problem_count": sum(
            int(spec.get("expected_judgment_n", spec["n"]))
            for spec in sample_specs
        ),
        "answer_success_count": sum(1 for row in answer_results if row.get("status") == "succeeded"),
        "answer_planned_count": sum(1 for row in answer_results if row.get("status") == "planned"),
        "judged_proposal_count": sum(1 for row in judgment_results if row.get("status") == "judged"),
        "judged_problem_count": sum(int(row.get("judged_problem_count") or 0) for row in judgment_results),
        "live_judged_pair_count": sum(int(row.get("live_judged_pair_count") or 0) for row in judgment_results),
        "route_scoped_noop_control_count": sum(
            int(row.get("route_scoped_noop_control_count") or 0)
            for row in judgment_results
        ),
        "candidate_acceptance_counts": acceptance_counts,
        "accepted_count": int(acceptance_counts.get("accept", 0)),
        "rejected_benefit_count": int(acceptance_counts.get("reject_benefit", 0)),
        "rejected_harm_count": int(acceptance_counts.get("reject_harm", 0)),
        "readback_applied_count": int(readback.get("readback_applied_count", 0)),
        "node_mutation_without_apply": bool(readback.get("node_mutation_without_apply")),
        "temp_apply_applied_count": len(readback.get("temp_apply_applied_candidate_node_ids", [])),
        "live_env_ready": bool(env["gpt_solver_ready"] and env["gpt_judge_ready"]),
    }


def _status(*, execute_answers: bool, run_judge: bool, metrics: dict[str, Any]) -> str:
    if not execute_answers and not run_judge:
        return "planned"
    if not metrics["live_env_ready"]:
        return "live_env_missing"
    if metrics["judged_problem_count"] < metrics["expected_judged_problem_count"]:
        return "incomplete_live_run"
    if metrics["accepted_count"]:
        return "live_positive_acceptance"
    return "live_completed_no_acceptance"


def _interpretation(metrics: dict[str, Any]) -> str:
    if not metrics["live_env_ready"]:
        return "Live run requires GPT solver/judge environment variables; no key values are written to artifacts."
    if metrics["accepted_count"]:
        return "At least one orthogonal new-family proposal passed live trigger/control acceptance."
    if metrics["judged_problem_count"] == metrics["expected_judged_problem_count"]:
        return "Live answer/judge pipeline completed, but no orthogonal proposal passed the acceptance gate."
    return "Live run is incomplete; inspect answer and judgment result rows."


def _framework_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("LLM_PROVIDER", "gpt")
    if not env.get("GPT5_API_KEY") and env.get("RUOLI_GPT_KEY"):
        env["GPT5_API_KEY"] = env["RUOLI_GPT_KEY"]
    if not env.get("GPT5_BASE_URL") and env.get("RUOLI_BASE_URL"):
        env["GPT5_BASE_URL"] = env["RUOLI_BASE_URL"].rstrip("/") + "/v1"
    env.setdefault("GPT5_MODEL", env.get("GPT_MINI_MODEL", "gpt-5.4-mini"))
    env.setdefault("GPT55_MODEL", "gpt-5.5")
    env.setdefault("OPENAI_TIMEOUT", "180")
    return env


def _env_status() -> dict[str, Any]:
    env = _framework_env()
    gpt_key_ready = bool(env.get("GPT5_API_KEY") or env.get("RUOLI_GPT_KEY"))
    gpt_base_ready = bool(env.get("GPT5_BASE_URL") or env.get("RUOLI_BASE_URL"))
    return {
        "gpt_solver_ready": gpt_key_ready and gpt_base_ready,
        "gpt_judge_ready": gpt_key_ready and gpt_base_ready,
        "set_names": [
            name for name in [
                "LLM_PROVIDER",
                "GPT5_API_KEY",
                "GPT5_BASE_URL",
                "RUOLI_GPT_KEY",
                "RUOLI_BASE_URL",
                "GPT5_MODEL",
                "GPT_MINI_MODEL",
                "GPT55_MODEL",
            ]
            if bool(env.get(name))
        ],
        "required_names": ["GPT5_API_KEY or RUOLI_GPT_KEY", "GPT5_BASE_URL or RUOLI_BASE_URL"],
    }


def _compact_judgment_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": row.get("proposal_id"),
            "candidate_variant": row.get("candidate_variant"),
            "baseline_variant": row.get("baseline_variant"),
            "judgment_path": row.get("judgment_path"),
            "expected_problem_count": row.get("expected_problem_count"),
            "judged_problem_count": row.get("judged_problem_count"),
            "live_judged_pair_count": row.get("live_judged_pair_count"),
            "route_scoped_noop_control_count": row.get("route_scoped_noop_control_count"),
            "control_mode": row.get("control_mode"),
            "winner_counts": row.get("winner_counts"),
            "status": row.get("status"),
            "missing_problem_ids": row.get("missing_problem_ids", []),
        }
        for row in rows
    ]


def _commands_are_secret_free(answer_results: list[dict[str, Any]]) -> bool:
    text = json.dumps([row.get("command", []) for row in answer_results], ensure_ascii=False)
    return "sk-" not in text and "newapi_channel_conn" not in text


def _redacted_command(cmd: list[str]) -> list[str]:
    return list(cmd)


def _judgment_path(
    root: Path,
    eval_id: str,
    proposal_id: str,
    candidate_variant: str,
    baseline_variant: str,
) -> Path:
    return root / JUDGMENTS_DIR / f"{candidate_variant}_vs_{baseline_variant}_{eval_id}_{proposal_id}.json"


def _candidate_variant(proposal_id: str) -> str:
    return f"proposal_{proposal_id.replace('prop_', '')}"


def _sample_by_pid(root: Path) -> dict[str, dict[str, Any]]:
    rows = _load_json(root / CACHE_DIR / "sample_100.json")
    return {row["problem_id"]: row for row in rows}


def _unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live downstream ablation for orthogonal multi-cluster proposals.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE_PATH))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--eval-id", default="orthogonal_multi_cluster_live_ablation_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--execute-answers", action="store_true")
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--answer-workers", type=int, default=3)
    parser.add_argument("--judge-workers", type=int, default=3)
    parser.add_argument("--judge-model", default="gpt55")
    parser.add_argument("--baseline-variant", default=BASELINE_VARIANT)
    parser.add_argument("--route-scoped-noop-controls", action="store_true")
    parser.add_argument("--apply-accepted", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_orthogonal_live_ablation_payload(
        root=root,
        queue_path=Path(args.queue),
        graph_dir=Path(args.graph_dir),
        eval_id=args.eval_id,
        execute_answers=args.execute_answers,
        run_judge=args.run_judge,
        answer_workers=args.answer_workers,
        judge_workers=args.judge_workers,
        judge_model=args.judge_model,
        baseline_variant=args.baseline_variant,
        route_scoped_noop_controls=args.route_scoped_noop_controls,
        apply_accepted=args.apply_accepted,
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "status": payload["status"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
