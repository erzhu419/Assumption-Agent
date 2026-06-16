"""Parallel shard runner for HLE smoke evaluations.

This module orchestrates multiple ``hle_smoke_eval`` subprocesses.  It does
not change the underlying scoring path; it only adds bounded parallelism,
heartbeat files, soft timeouts, and error-stratified aggregate reports.
Artifacts intentionally store hashes, counts, and metadata only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .hle_smoke_eval import (
    DATASET_NAME,
    HLE_OFFICIAL_SOURCES,
    _aggregate_rows,
    _component_efficacy_summary,
    _control_comparison,
    _expected_but_missing_modules,
    _module_activation_summary,
)


DEFAULT_RUN_DIR = PAPER_DIR / "hle_parallel_runs"
DEFAULT_MD_DIR = Path("reconstruction/md")
ERROR_EVENT_NAMES = {
    "call_error",
    "recursive_child_error",
    "recursive_child_timeout",
    "candidate_claim_verifier_error",
    "counter_assumption_verifier_error",
    "source_grounded_mc_verifier_error",
    "option_evidence_arbitrator_error",
    "domain_rule_mc_verifier_error",
    "critic_synthesis_child_error",
    "math_tool_child_error",
}


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    eval_id: str
    sample_size: int
    seed_offset: int
    out: Path
    md_out: Path
    log_out: Path
    stdout_out: Path


@dataclass
class ShardRunState:
    spec: ShardSpec
    command: list[str]
    process: subprocess.Popen[Any] | None = None
    started_monotonic: float | None = None
    finished_monotonic: float | None = None
    returncode: int | None = None
    status: str = "pending"
    soft_timeout_sent: bool = False
    hard_kill_sent: bool = False
    error: str | None = None
    _stdout_handle: Any = field(default=None, repr=False, compare=False)

    def elapsed_sec(self, now: float | None = None) -> float | None:
        if self.started_monotonic is None:
            return None
        end = self.finished_monotonic if self.finished_monotonic is not None else (now or time.monotonic())
        return round(max(0.0, end - self.started_monotonic), 4)


def build_shard_specs(
    *,
    eval_id: str,
    total_sample_size: int,
    shard_size: int,
    seed_offset: int,
    seed_stride: int,
    run_dir: Path,
    md_dir: Path,
) -> list[ShardSpec]:
    if total_sample_size <= 0:
        raise ValueError("total_sample_size must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    shard_count = math.ceil(total_sample_size / shard_size)
    specs: list[ShardSpec] = []
    for shard_index in range(shard_count):
        current_size = min(shard_size, total_sample_size - shard_index * shard_size)
        shard_eval_id = f"{eval_id}_shard_{shard_index:03d}"
        specs.append(
            ShardSpec(
                shard_index=shard_index,
                eval_id=shard_eval_id,
                sample_size=current_size,
                seed_offset=seed_offset + shard_index * seed_stride,
                out=run_dir / f"{shard_eval_id}.json",
                md_out=md_dir / f"{shard_eval_id}.md",
                log_out=run_dir / f"{shard_eval_id}.jsonl",
                stdout_out=run_dir / f"{shard_eval_id}.stdout.log",
            )
        )
    return specs


def build_shard_command(
    spec: ShardSpec,
    *,
    root: Path,
    max_scan: int,
    models: str,
    variants: str,
    execute_live: bool,
    call_timeout: float | None,
    max_tokens: int,
    graph_dir: Path,
    agent_top_k: int,
    agent_context_max_chars: int,
    agent_child_mode: str,
    agent_child_timeout: float | None,
    evidence_bridge_enabled: bool,
    exclude_existing_hle_artifacts: bool,
    exclude_artifact_glob: str,
    sample_answer_type: str,
    sample_subject_contains: str,
) -> list[str]:
    effective_max_scan = max_scan + max(0, spec.seed_offset)
    cmd = [
        sys.executable,
        "-m",
        "assumption_os.hle_smoke_eval",
        "--root",
        str(root),
        "--eval-id",
        spec.eval_id,
        "--sample-size",
        str(spec.sample_size),
        "--max-scan",
        str(effective_max_scan),
        "--seed-offset",
        str(spec.seed_offset),
        "--models",
        models,
        "--variants",
        variants,
        "--max-tokens",
        str(max_tokens),
        "--log-out",
        str(spec.log_out),
        "--graph-dir",
        str(graph_dir),
        "--agent-top-k",
        str(agent_top_k),
        "--agent-context-max-chars",
        str(agent_context_max_chars),
        "--agent-child-mode",
        agent_child_mode,
        "--out",
        str(spec.out),
        "--md-out",
        str(spec.md_out),
        "--hard-exit-after-write",
    ]
    if execute_live:
        cmd.append("--execute-live")
    if call_timeout is not None:
        cmd.extend(["--call-timeout", str(call_timeout)])
    if agent_child_timeout is not None:
        cmd.extend(["--agent-child-timeout", str(agent_child_timeout)])
    if not evidence_bridge_enabled:
        cmd.append("--disable-evidence-bridge")
    if exclude_existing_hle_artifacts:
        cmd.append("--exclude-existing-hle-artifacts")
    if exclude_artifact_glob:
        cmd.extend(["--exclude-artifact-glob", exclude_artifact_glob])
    if sample_answer_type:
        cmd.extend(["--sample-answer-type", sample_answer_type])
    if sample_subject_contains:
        cmd.extend(["--sample-subject-contains", sample_subject_contains])
    return cmd


def build_runner_env(
    *,
    model_router_attempts: int | None,
    model_router_timeout: float | None,
    model_router_per_attempt_timeout: float | None = None,
    model_router_backoff_base_sec: float | None = None,
    model_router_global_concurrency: int | None = None,
    model_router_global_concurrency_dir: str | None = None,
    model_router_global_slot_ttl_sec: float | None = None,
    model_router_global_slot_wait_sec: float | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if model_router_attempts is not None:
        env["MODEL_ROUTER_ATTEMPTS"] = str(model_router_attempts)
    if model_router_timeout is not None:
        env["MODEL_ROUTER_TIMEOUT"] = str(model_router_timeout)
    if model_router_per_attempt_timeout is not None:
        env["MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"] = str(model_router_per_attempt_timeout)
    if model_router_backoff_base_sec is not None:
        env["MODEL_ROUTER_BACKOFF_BASE_SEC"] = str(model_router_backoff_base_sec)
    if model_router_global_concurrency is not None:
        env["MODEL_ROUTER_GLOBAL_CONCURRENCY"] = str(model_router_global_concurrency)
    if model_router_global_concurrency_dir:
        env["MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR"] = model_router_global_concurrency_dir
    if model_router_global_slot_ttl_sec is not None:
        env["MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC"] = str(model_router_global_slot_ttl_sec)
    if model_router_global_slot_wait_sec is not None:
        env["MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC"] = str(model_router_global_slot_wait_sec)
    return env


def run_parallel_shards(
    *,
    root: Path,
    shard_states: list[ShardRunState],
    parallel_workers: int,
    heartbeat_path: Path,
    poll_interval_sec: float,
    heartbeat_interval_sec: float,
    soft_timeout_sec: float | None,
    terminate_grace_sec: float,
    env: dict[str, str],
) -> list[ShardRunState]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    pending = list(shard_states)
    running: list[ShardRunState] = []
    completed: list[ShardRunState] = []
    last_heartbeat = 0.0
    while pending or running:
        now = time.monotonic()
        while pending and len(running) < parallel_workers:
            state = pending.pop(0)
            state.spec.out.parent.mkdir(parents=True, exist_ok=True)
            state.spec.md_out.parent.mkdir(parents=True, exist_ok=True)
            state.spec.log_out.parent.mkdir(parents=True, exist_ok=True)
            state._stdout_handle = state.spec.stdout_out.open("w", encoding="utf-8")
            try:
                state.process = subprocess.Popen(
                    state.command,
                    cwd=str(root),
                    env=env,
                    stdout=state._stdout_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                state.started_monotonic = time.monotonic()
                state.status = "running"
                running.append(state)
            except Exception as exc:  # pragma: no cover - defensive subprocess path.
                state.error = f"{type(exc).__name__}: {exc}"
                state.status = "spawn_failed"
                state.finished_monotonic = time.monotonic()
                _close_stdout(state)
                completed.append(state)
        still_running: list[ShardRunState] = []
        for state in running:
            process = state.process
            if process is None:
                state.status = "spawn_failed"
                state.finished_monotonic = time.monotonic()
                completed.append(state)
                continue
            returncode = process.poll()
            elapsed = state.elapsed_sec(now)
            if returncode is None and soft_timeout_sec is not None and elapsed is not None:
                if elapsed > soft_timeout_sec and not state.soft_timeout_sent:
                    state.soft_timeout_sent = True
                    state.status = "soft_timed_out"
                    process.terminate()
                elif (
                    state.soft_timeout_sent
                    and elapsed > soft_timeout_sec + terminate_grace_sec
                    and not state.hard_kill_sent
                ):
                    state.hard_kill_sent = True
                    state.status = "hard_killed"
                    process.kill()
            returncode = process.poll()
            if returncode is None:
                still_running.append(state)
                continue
            state.returncode = int(returncode)
            state.finished_monotonic = time.monotonic()
            if state.status in {"soft_timed_out", "hard_killed"}:
                pass
            elif returncode == 0:
                state.status = "completed"
            else:
                state.status = "failed"
            _close_stdout(state)
            completed.append(state)
        running = still_running
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_interval_sec:
            heartbeat_path.write_text(
                json.dumps(build_heartbeat(shard_states), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            last_heartbeat = now
        if pending or running:
            time.sleep(max(0.1, poll_interval_sec))
    heartbeat_path.write_text(
        json.dumps(build_heartbeat(shard_states), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return shard_states


def _close_stdout(state: ShardRunState) -> None:
    handle = state._stdout_handle
    if handle is not None:
        handle.flush()
        handle.close()
        state._stdout_handle = None


def build_heartbeat(states: list[ShardRunState]) -> dict[str, Any]:
    now = time.monotonic()
    status_counts = Counter(state.status for state in states)
    shard_rows = []
    for state in states:
        latest_event = _read_latest_jsonl_event(state.spec.log_out)
        shard_rows.append(
            {
                "shard_index": state.spec.shard_index,
                "eval_id": state.spec.eval_id,
                "status": state.status,
                "returncode": state.returncode,
                "elapsed_sec": state.elapsed_sec(now),
                "sample_size": state.spec.sample_size,
                "seed_offset": state.spec.seed_offset,
                "out_exists": state.spec.out.exists(),
                "log_out_exists": state.spec.log_out.exists(),
                "soft_timeout_sent": state.soft_timeout_sent,
                "hard_kill_sent": state.hard_kill_sent,
                "latest_event": latest_event,
                "error": state.error,
            }
        )
    return {
        "heartbeat_kind": "hle_parallel_shard_runner",
        "status_counts": dict(sorted(status_counts.items())),
        "shards": shard_rows,
        "raw_content_persisted": False,
    }


def _read_latest_jsonl_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest = {
                    "event": event.get("event"),
                    "model": event.get("model"),
                    "variant": event.get("variant"),
                    "problem_id_hash": event.get("problem_id_hash"),
                    "error_type": event.get("error_type"),
                    "stage": event.get("stage"),
                }
    except OSError:
        return None
    return latest


def load_shard_payloads(specs: list[ShardSpec]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.out.exists():
            continue
        try:
            payloads.append(json.loads(spec.out.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def aggregate_parallel_payload(
    *,
    eval_id: str,
    specs: list[ShardSpec],
    states: list[ShardRunState],
    shard_payloads: list[dict[str, Any]],
    execute_live: bool,
    models: str,
    variants: str,
    total_sample_size: int,
    shard_size: int,
    parallel_workers: int,
    soft_timeout_sec: float | None,
) -> dict[str, Any]:
    run_rows = _merged_run_rows(shard_payloads)
    metrics = _parallel_metrics(run_rows=run_rows, shard_payloads=shard_payloads)
    error_stratification = build_error_stratification(
        rows=run_rows,
        specs=specs,
        states=states,
    )
    pollution_audit = build_pollution_audit(
        rows=run_rows,
        shard_payloads=shard_payloads,
        metrics=metrics,
        error_stratification=error_stratification,
        execute_live=execute_live,
    )
    gates = {
        "all_shards_finished_without_process_failure": all(
            state.status == "completed" for state in states
        ),
        "all_available_payloads_preserve_raw_content": all(
            (payload.get("metrics") or {}).get("raw_content_persisted") is False
            for payload in shard_payloads
        ),
        "sample_rows_loaded": metrics["sample_count"] >= min(total_sample_size, 1),
        "live_rows_resolved_if_requested": (
            not execute_live
            or metrics["resolved_live_model_calls"] == metrics["planned_live_model_calls"]
        ),
    }
    paper_clean_gates = dict(gates)
    paper_clean_gates["zero_top_level_live_errors"] = error_stratification["top_level_error_count"] == 0
    paper_clean_gates["zero_process_timeouts"] = error_stratification["process_timeout_count"] == 0
    paper_clean_gates["no_duplicate_sample_problems"] = metrics["duplicate_sample_problem_count"] == 0
    pollution_gates = pollution_audit["gates"]
    return {
        "eval_id": eval_id,
        "eval_kind": "hle_parallel_shard_runner",
        "dataset": DATASET_NAME,
        "official_sources": HLE_OFFICIAL_SOURCES,
        "performance_validation": True,
        "validation_scope": (
            "Runs HLE smoke-eval shards through a bounded parallel subprocess runner. "
            "The artifact stores only hashes, counts, process states, and error types."
        ),
        "sampling": {
            "requested_total_sample_size": total_sample_size,
            "shard_size": shard_size,
            "planned_shard_count": len(specs),
            "parallel_workers": parallel_workers,
            "models": [item.strip() for item in models.split(",") if item.strip()],
            "variants": [item.strip() for item in variants.split(",") if item.strip()],
        },
        "runtime_policy": {
            "execute_live": execute_live,
            "soft_timeout_sec": soft_timeout_sec,
            "raw_content_persisted": False,
        },
        "shards": [_shard_summary(state) for state in states],
        "loaded_shard_payload_count": len(shard_payloads),
        "metrics": metrics,
        "error_stratification": error_stratification,
        "pollution_audit": pollution_audit,
        "pass": all(gates.values()),
        "paper_clean_pass": all(paper_clean_gates.values()),
        "pollution_pass": all(pollution_gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "paper_clean_failed_gates": [name for name, passed in paper_clean_gates.items() if not passed],
        "pollution_failed_gates": [name for name, passed in pollution_gates.items() if not passed],
        "raw_content_persisted": False,
    }


def _shard_summary(state: ShardRunState) -> dict[str, Any]:
    return {
        "shard_index": state.spec.shard_index,
        "eval_id": state.spec.eval_id,
        "status": state.status,
        "returncode": state.returncode,
        "elapsed_sec": state.elapsed_sec(),
        "sample_size": state.spec.sample_size,
        "seed_offset": state.spec.seed_offset,
        "out": str(state.spec.out),
        "log_out": str(state.spec.log_out),
        "stdout_out": str(state.spec.stdout_out),
        "soft_timeout_sent": state.soft_timeout_sent,
        "hard_kill_sent": state.hard_kill_sent,
        "error": state.error,
    }


def _merged_run_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("rows", []) or payload.get("run_rows", []) or [])
    return rows


def _parallel_metrics(*, run_rows: list[dict[str, Any]], shard_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_key[f"{row.get('model')}::{row.get('variant')}"].append(row)
    sample_count = sum(int((payload.get("metrics") or {}).get("sample_count") or 0) for payload in shard_payloads)
    planned = sum(
        int((payload.get("metrics") or {}).get("planned_live_model_calls") or 0)
        for payload in shard_payloads
    )
    live_executed = sum(
        int((payload.get("metrics") or {}).get("live_model_calls_executed") or 0)
        for payload in shard_payloads
    )
    underlying = sum(
        int((payload.get("metrics") or {}).get("underlying_model_calls_executed") or 0)
        for payload in shard_payloads
    )
    resolved = sum(
        int((payload.get("metrics") or {}).get("resolved_live_model_calls") or 0)
        for payload in shard_payloads
    )
    sample_problem_hashes = _merged_sample_problem_hashes(shard_payloads)
    row_problem_hashes = [str(row.get("problem_id_hash")) for row in run_rows if row.get("problem_id_hash")]
    distinct_sample_problem_hashes = set(sample_problem_hashes)
    distinct_row_problem_hashes = set(row_problem_hashes)
    return {
        "sample_count": sample_count,
        "distinct_sample_problem_count": len(distinct_sample_problem_hashes),
        "duplicate_sample_problem_count": max(0, len(sample_problem_hashes) - len(distinct_sample_problem_hashes)),
        "distinct_scored_problem_count": len(distinct_row_problem_hashes),
        "duplicate_scored_problem_count": max(0, len(row_problem_hashes) - len(distinct_row_problem_hashes)),
        "planned_live_model_calls": planned,
        "live_model_calls_executed": live_executed,
        "underlying_model_calls_executed": underlying,
        "resolved_live_model_calls": resolved,
        "scored_row_count": len(run_rows),
        "overall_accuracy": _accuracy(run_rows),
        "by_model_variant": {key: _aggregate_rows(rows) for key, rows in sorted(by_key.items())},
        "control_comparison": _control_comparison(run_rows),
        "module_activation_summary": _module_activation_summary(run_rows),
        "expected_but_missing_modules": _expected_but_missing_modules(run_rows),
        "component_efficacy_summary": _component_efficacy_summary(run_rows),
        "clean_shared_subset": _clean_shared_subset(run_rows),
        "raw_content_persisted": False,
    }


def _merged_sample_problem_hashes(payloads: list[dict[str, Any]]) -> list[str]:
    hashes: list[str] = []
    for payload in payloads:
        sampling = payload.get("sampling") or {}
        for value in sampling.get("sample_problem_hashes", []) or []:
            hashes.append(str(value))
    return hashes


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get("correct")) / len(rows), 4)


def _clean_shared_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_model[str(row.get("model"))][str(row.get("variant"))][str(row.get("problem_id_hash"))] = row
    out: dict[str, Any] = {}
    for model, by_variant in sorted(by_model.items()):
        if not by_variant:
            continue
        shared_ids: set[str] | None = None
        for variant_rows in by_variant.values():
            clean_ids = {pid for pid, row in variant_rows.items() if not row.get("error")}
            shared_ids = clean_ids if shared_ids is None else shared_ids & clean_ids
        shared_ids = shared_ids or set()
        variant_metrics = {
            variant: {
                "n": len(shared_ids),
                "accuracy": _accuracy([variant_rows[pid] for pid in sorted(shared_ids)]),
            }
            for variant, variant_rows in sorted(by_variant.items())
        }
        out[model] = {
            "shared_clean_problem_count": len(shared_ids),
            "by_variant": variant_metrics,
        }
    return out


def build_error_stratification(
    *,
    rows: list[dict[str, Any]],
    specs: list[ShardSpec],
    states: list[ShardRunState],
) -> dict[str, Any]:
    top_level_by_variant: Counter[str] = Counter()
    top_level_by_type: Counter[str] = Counter()
    top_level_by_variant_type: Counter[str] = Counter()
    for row in rows:
        error = row.get("error") or {}
        if not error:
            continue
        variant = str(row.get("variant"))
        error_type = str(error.get("type") or "unknown")
        top_level_by_variant[variant] += 1
        top_level_by_type[error_type] += 1
        top_level_by_variant_type[f"{variant}::{error_type}"] += 1

    jsonl_events = _jsonl_error_events(specs)
    process_status_counts = Counter(state.status for state in states)
    process_timeout_count = sum(1 for state in states if state.soft_timeout_sent or state.hard_kill_sent)
    return {
        "top_level_error_count": sum(top_level_by_variant.values()),
        "top_level_errors_by_variant": dict(sorted(top_level_by_variant.items())),
        "top_level_errors_by_type": dict(sorted(top_level_by_type.items())),
        "top_level_errors_by_variant_type": dict(sorted(top_level_by_variant_type.items())),
        "jsonl_error_event_count": sum(jsonl_events["by_event"].values()),
        "jsonl_error_events_by_event": dict(sorted(jsonl_events["by_event"].items())),
        "jsonl_error_events_by_variant": dict(sorted(jsonl_events["by_variant"].items())),
        "jsonl_error_events_by_type": dict(sorted(jsonl_events["by_error_type"].items())),
        "process_status_counts": dict(sorted(process_status_counts.items())),
        "process_timeout_count": process_timeout_count,
        "raw_content_persisted": False,
    }


def build_pollution_audit(
    *,
    rows: list[dict[str, Any]],
    shard_payloads: list[dict[str, Any]],
    metrics: dict[str, Any],
    error_stratification: dict[str, Any],
    execute_live: bool,
) -> dict[str, Any]:
    sample_hashes = _merged_sample_problem_hashes(shard_payloads)
    api_summaries = [payload.get("api_summary") or {} for payload in shard_payloads]
    excluded_existing_problem_count = sum(
        int(summary.get("excluded_existing_problem_count") or 0)
        for summary in api_summaries
    )
    exclude_existing_enabled_count = sum(
        1 for summary in api_summaries if bool(summary.get("exclude_existing_hle_artifacts"))
    )
    context_by_variant = _context_pollution_by_variant(rows)
    selection_credit = _selection_credit(rows)
    clean_shared = metrics.get("clean_shared_subset") or {}
    agent_advantage = _clean_shared_agent_advantage(clean_shared)
    top_level_errors = int(error_stratification.get("top_level_error_count") or 0)
    process_timeouts = int(error_stratification.get("process_timeout_count") or 0)
    clean_shared_problem_count = max(
        [int(row.get("shared_clean_problem_count") or 0) for row in clean_shared.values()] or [0]
    )
    claim_scope = {
        "paper_clean_claim_allowed": top_level_errors == 0 and process_timeouts == 0,
        "selective_agent_advantage_claim_allowed": bool(agent_advantage.get("agent_beats_all_controls")),
        "recommended_hle_claim_scope": (
            "full_resolved_rows"
            if top_level_errors == 0 and process_timeouts == 0
            else "clean_shared_subset_due_to_endpoint_noise"
        ),
        "agent_advantage": agent_advantage,
    }
    gates = {
        "raw_content_not_persisted": metrics.get("raw_content_persisted") is False,
        "fresh_problem_hashes_accounted": bool(sample_hashes) or not execute_live,
        "no_duplicate_problem_hashes": int(metrics.get("duplicate_sample_problem_count") or 0) == 0,
        "cache_live_separation_accounted": True,
        "endpoint_errors_separated": "top_level_errors_by_variant" in error_stratification,
        "clean_shared_subset_available_if_endpoint_errors": top_level_errors == 0 or clean_shared_problem_count > 0,
        "context_pollution_accounted": isinstance(context_by_variant, dict),
        "selection_credit_accounted": bool(selection_credit.get("by_selection_method")) or not rows,
        "claim_scope_downgraded_when_endpoint_errors": (
            top_level_errors == 0
            or claim_scope["recommended_hle_claim_scope"] == "clean_shared_subset_due_to_endpoint_noise"
        ),
    }
    return {
        "audit_kind": "hle_anti_pollution_audit",
        "fresh_problem_hash_exclusion": {
            "sample_problem_hash_count": len(sample_hashes),
            "distinct_sample_problem_hash_count": len(set(sample_hashes)),
            "duplicate_sample_problem_hash_count": max(0, len(sample_hashes) - len(set(sample_hashes))),
            "exclude_existing_enabled_shard_count": exclude_existing_enabled_count,
            "excluded_existing_problem_count": excluded_existing_problem_count,
        },
        "cache_live_separation": {
            "execute_live": execute_live,
            "planned_live_model_calls": metrics.get("planned_live_model_calls"),
            "resolved_live_model_calls": metrics.get("resolved_live_model_calls"),
            "live_model_calls_executed": metrics.get("live_model_calls_executed"),
            "underlying_model_calls_executed": metrics.get("underlying_model_calls_executed"),
            "top_level_error_count": top_level_errors,
            "process_timeout_count": process_timeouts,
            "top_level_errors_by_variant": error_stratification.get("top_level_errors_by_variant") or {},
        },
        "context_pollution": {
            "by_variant": context_by_variant,
            "summary": _context_pollution_summary(context_by_variant),
        },
        "module_credit_assignment": selection_credit,
        "claim_guard": claim_scope,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "raw_content_persisted": False,
    }


def _context_pollution_by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        variant = str(row.get("variant") or "unknown")
        ce = row.get("component_efficacy") if isinstance(row.get("component_efficacy"), dict) else {}
        flags = ce.get("flags") if isinstance(ce.get("flags"), dict) else {}
        correct = bool(row.get("correct"))
        has_error = bool(row.get("error"))
        outcome_bucket = "error" if has_error else "correct" if correct else "wrong"
        graph = ce.get("graph") if isinstance(ce.get("graph"), dict) else {}
        evidence = ce.get("evidence") if isinstance(ce.get("evidence"), dict) else {}
        hipporag = ce.get("agent_hipporag") if isinstance(ce.get("agent_hipporag"), dict) else {}
        morphism = ce.get("morphism") if isinstance(ce.get("morphism"), dict) else {}
        if _flag_true(flags, "graph_context_discarded"):
            by_variant[variant]["graph_context_discarded"] += 1
        if _flag_true(flags, "generic_graph_context_only"):
            by_variant[variant]["generic_graph_context_only"] += 1
        if graph.get("status") in {"activated", "used"}:
            by_variant[variant]["graph_retrieval_activated"] += 1
            if _is_generic_harness_graph_context(graph):
                by_variant[variant]["graph_generic_harness_retrieved"] += 1
        if _flag_true(flags, "graph_context_injected"):
            by_variant[variant]["graph_context_used"] += 1
            by_variant[variant][f"graph_context_{outcome_bucket}"] += 1
            if _is_generic_harness_graph_context(graph):
                by_variant[variant]["graph_generic_harness_context"] += 1
        if _flag_true(flags, "evidence_bridge_activated") or evidence.get("status") in {"activated", "used"}:
            by_variant[variant]["evidence_context_used"] += 1
            by_variant[variant][f"evidence_context_{outcome_bucket}"] += 1
        if evidence.get("status") in {"no_results", "empty"}:
            by_variant[variant]["evidence_no_results"] += 1
        if _flag_true(flags, "agent_hipporag_context_activated") or hipporag.get("status") in {"activated", "used"}:
            by_variant[variant]["hipporag_context_used"] += 1
            by_variant[variant][f"hipporag_context_{outcome_bucket}"] += 1
        if hipporag.get("status") in {"no_results", "empty"}:
            by_variant[variant]["hipporag_no_results"] += 1
        if _flag_true(flags, "morphism_hit") or int(morphism.get("formal_hit_count") or 0) > 0 or int(morphism.get("structural_hit_count") or 0) > 0:
            by_variant[variant]["morphism_hit"] += 1
            by_variant[variant][f"morphism_{outcome_bucket}"] += 1
        if _flag_true(flags, "strong_morphism_hit") or int(morphism.get("strong_hit_count") or 0) > 0:
            by_variant[variant]["strong_morphism_hit"] += 1
            by_variant[variant][f"strong_morphism_{outcome_bucket}"] += 1
    return {variant: dict(sorted(counter.items())) for variant, counter in sorted(by_variant.items())}


def _context_pollution_summary(context_by_variant: dict[str, dict[str, int]]) -> dict[str, int]:
    summary: Counter[str] = Counter()
    for counts in context_by_variant.values():
        for key, value in counts.items():
            summary[key] += int(value)
    return dict(sorted(summary.items()))


def _flag_true(flags: dict[str, Any], key: str) -> bool:
    return bool(flags.get(key))


def _is_generic_harness_graph_context(graph: dict[str, Any]) -> bool:
    counts = graph.get("top_node_type_counts")
    if not isinstance(counts, dict) or not counts:
        return False
    total = sum(int(value or 0) for value in counts.values())
    harness = int(counts.get("harness") or counts.get("generic_harness") or 0)
    return total > 0 and harness >= total


def _selection_credit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    by_variant_method: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        method = _selection_method(row)
        variant = str(row.get("variant") or "unknown")
        correct = bool(row.get("correct"))
        has_error = bool(row.get("error"))
        by_method[method]["n"] += 1
        by_variant_method[f"{variant}::{method}"]["n"] += 1
        if correct:
            by_method[method]["correct"] += 1
            by_variant_method[f"{variant}::{method}"]["correct"] += 1
        if has_error:
            by_method[method]["error"] += 1
            by_variant_method[f"{variant}::{method}"]["error"] += 1
    return {
        "by_selection_method": _credit_counter_rows(by_method),
        "by_variant_selection_method": _credit_counter_rows(by_variant_method),
    }


def _credit_counter_rows(counters: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, counter in sorted(counters.items()):
        n = int(counter.get("n") or 0)
        out[key] = {
            "n": n,
            "correct": int(counter.get("correct") or 0),
            "error": int(counter.get("error") or 0),
            "accuracy": round(int(counter.get("correct") or 0) / n, 4) if n else None,
        }
    return out


def _selection_method(row: dict[str, Any]) -> str:
    ce = row.get("component_efficacy") if isinstance(row.get("component_efficacy"), dict) else {}
    selection = ce.get("selection") if isinstance(ce.get("selection"), dict) else {}
    stages = ce.get("stages") if isinstance(ce.get("stages"), dict) else {}
    multi = stages.get("multi_candidate_self_verifier") if isinstance(stages.get("multi_candidate_self_verifier"), dict) else {}
    metadata = row.get("call_metadata") if isinstance(row.get("call_metadata"), dict) else {}
    return str(
        selection.get("method")
        or selection.get("selection_method")
        or ce.get("selection_method")
        or multi.get("selection_method")
        or metadata.get("selection_method")
        or "unknown"
    )


def _clean_shared_agent_advantage(clean_shared: dict[str, Any]) -> dict[str, Any]:
    best_payload: dict[str, Any] = {
        "agent_beats_all_controls": False,
        "model": None,
        "agent_variant": None,
        "agent_accuracy": None,
        "best_control_accuracy": None,
        "margin": None,
    }
    for model, row in sorted(clean_shared.items()):
        by_variant = row.get("by_variant") or {}
        agent_items = [
            (variant, variant_row)
            for variant, variant_row in by_variant.items()
            if str(variant).startswith("assumption_agent")
        ]
        control_items = [
            (variant, variant_row)
            for variant, variant_row in by_variant.items()
            if not str(variant).startswith("assumption_agent")
        ]
        for agent_variant, agent_row in agent_items:
            agent_acc = agent_row.get("accuracy")
            control_accs = [
                control_row.get("accuracy")
                for _, control_row in control_items
                if control_row.get("accuracy") is not None
            ]
            if agent_acc is None or not control_accs:
                continue
            best_control = max(float(value) for value in control_accs)
            margin = round(float(agent_acc) - best_control, 4)
            if best_payload["margin"] is None or margin > float(best_payload["margin"]):
                best_payload = {
                    "agent_beats_all_controls": margin > 0,
                    "model": model,
                    "agent_variant": agent_variant,
                    "agent_accuracy": agent_acc,
                    "best_control_accuracy": best_control,
                    "margin": margin,
                }
    return best_payload


def _jsonl_error_events(specs: list[ShardSpec]) -> dict[str, Counter[str]]:
    by_event: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()
    by_error_type: Counter[str] = Counter()
    for spec in specs:
        if not spec.log_out.exists():
            continue
        try:
            with spec.log_out.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = str(event.get("event") or "")
                    if name not in ERROR_EVENT_NAMES:
                        continue
                    by_event[name] += 1
                    by_variant[str(event.get("variant") or "unknown")] += 1
                    by_error_type[str(event.get("error_type") or "unknown")] += 1
        except OSError:
            continue
    return {"by_event": by_event, "by_variant": by_variant, "by_error_type": by_error_type}


def format_parallel_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    errors = payload["error_stratification"]
    pollution = payload.get("pollution_audit") or {}
    claim_guard = pollution.get("claim_guard") or {}
    lines = [
        "# HLE Parallel Shard Evaluation",
        "",
        f"- pass: `{payload['pass']}`",
        f"- paper clean pass: `{payload['paper_clean_pass']}`",
        f"- pollution pass: `{payload.get('pollution_pass')}`",
        f"- loaded shard payloads: `{payload['loaded_shard_payload_count']}/{payload['sampling']['planned_shard_count']}`",
        f"- sample count: `{metrics['sample_count']}`",
        f"- distinct sample problems: `{metrics['distinct_sample_problem_count']}`",
        f"- duplicate sample problems: `{metrics['duplicate_sample_problem_count']}`",
        f"- live attempts resolved: `{metrics['resolved_live_model_calls']}/{metrics['planned_live_model_calls']}`",
        f"- scored rows: `{metrics['scored_row_count']}`",
        f"- overall accuracy: `{metrics['overall_accuracy']}`",
        f"- top-level live errors: `{errors['top_level_error_count']}`",
        f"- process timeouts: `{errors['process_timeout_count']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        f"- paper-clean failed gates: `{payload['paper_clean_failed_gates']}`",
        f"- pollution failed gates: `{payload.get('pollution_failed_gates')}`",
        f"- recommended HLE claim scope: `{claim_guard.get('recommended_hle_claim_scope')}`",
        "",
        "## By Variant",
        "",
        "| model | variant | n | accuracy | error count | MCQ accuracy | exact accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in sorted(metrics["by_model_variant"].items()):
        model, variant = key.split("::", 1)
        lines.append(
            f"| `{model}` | `{variant}` | `{row['n']}` | `{row['accuracy']}` | "
            f"`{row['error_count']}` | `{row['multiple_choice_accuracy']}` | "
            f"`{row['exact_match_accuracy']}` |"
        )
    lines.extend([
        "",
        "## Clean Shared Subset",
        "",
        "| model | variant | clean shared n | accuracy |",
        "| --- | --- | ---: | ---: |",
    ])
    for model, row in sorted(metrics.get("clean_shared_subset", {}).items()):
        for variant, variant_row in sorted(row.get("by_variant", {}).items()):
            lines.append(
                f"| `{model}` | `{variant}` | `{variant_row['n']}` | `{variant_row['accuracy']}` |"
            )
    lines.extend([
        "",
        "## Error Stratification",
        "",
        "| bucket | key | count |",
        "| --- | --- | ---: |",
    ])
    for bucket in (
        "top_level_errors_by_variant",
        "top_level_errors_by_type",
        "top_level_errors_by_variant_type",
        "jsonl_error_events_by_event",
        "jsonl_error_events_by_variant",
        "jsonl_error_events_by_type",
        "process_status_counts",
    ):
        for key, count in sorted((errors.get(bucket) or {}).items()):
            lines.append(f"| `{bucket}` | `{key}` | `{count}` |")
    context_summary = (pollution.get("context_pollution") or {}).get("summary") or {}
    lines.extend([
        "",
        "## Pollution Audit",
        "",
        "| bucket | key | value |",
        "| --- | --- | ---: |",
    ])
    for key, value in sorted((pollution.get("fresh_problem_hash_exclusion") or {}).items()):
        lines.append(f"| `fresh_problem_hash_exclusion` | `{key}` | `{value}` |")
    for key, value in sorted((pollution.get("cache_live_separation") or {}).items()):
        if isinstance(value, dict):
            continue
        lines.append(f"| `cache_live_separation` | `{key}` | `{value}` |")
    for key, value in sorted(context_summary.items()):
        lines.append(f"| `context_pollution_summary` | `{key}` | `{value}` |")
    for key, value in sorted((pollution.get("gates") or {}).items()):
        lines.append(f"| `pollution_gate` | `{key}` | `{value}` |")
    lines.extend([
        "",
        "## Selection Credit",
        "",
        "| method | n | correct | error | accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method, row in sorted(((pollution.get("module_credit_assignment") or {}).get("by_selection_method") or {}).items()):
        lines.append(
            f"| `{method}` | `{row['n']}` | `{row['correct']}` | `{row['error']}` | `{row['accuracy']}` |"
        )
    lines.extend([
        "",
        "## Shards",
        "",
        "| shard | status | returncode | elapsed sec | sample size | seed offset | timeout |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for shard in sorted(payload.get("shards", []), key=lambda item: item["shard_index"]):
        timeout = "soft" if shard.get("soft_timeout_sent") else "hard" if shard.get("hard_kill_sent") else "none"
        lines.append(
            f"| `{shard['shard_index']}` | `{shard['status']}` | `{shard['returncode']}` | "
            f"`{shard['elapsed_sec']}` | `{shard['sample_size']}` | `{shard['seed_offset']}` | `{timeout}` |"
        )
    lines.extend([
        "",
        "Raw HLE questions, answers, rationales, canaries, and prediction text are not persisted.",
    ])
    return "\n".join(lines) + "\n"


def build_payload_without_execution(args: argparse.Namespace) -> tuple[list[ShardSpec], list[ShardRunState]]:
    root = Path(args.root).resolve()
    run_dir = _path_arg(args.run_dir, root=root)
    md_dir = _path_arg(args.md_dir, root=root)
    specs = build_shard_specs(
        eval_id=args.eval_id,
        total_sample_size=args.total_sample_size,
        shard_size=args.shard_size,
        seed_offset=args.seed_offset,
        seed_stride=args.seed_stride,
        run_dir=run_dir,
        md_dir=md_dir,
    )
    graph_dir = _path_arg(args.graph_dir, root=root)
    states = [
        ShardRunState(
            spec=spec,
            command=build_shard_command(
                spec,
                root=root,
                max_scan=args.max_scan,
                models=args.models,
                variants=args.variants,
                execute_live=args.execute_live,
                call_timeout=args.call_timeout,
                max_tokens=args.max_tokens,
                graph_dir=graph_dir,
                agent_top_k=args.agent_top_k,
                agent_context_max_chars=args.agent_context_max_chars,
                agent_child_mode=args.agent_child_mode,
                agent_child_timeout=args.agent_child_timeout,
                evidence_bridge_enabled=not args.disable_evidence_bridge,
                exclude_existing_hle_artifacts=args.exclude_existing_hle_artifacts,
                exclude_artifact_glob=args.exclude_artifact_glob,
                sample_answer_type=args.sample_answer_type,
                sample_subject_contains=args.sample_subject_contains,
            ),
        )
        for spec in specs
    ]
    return specs, states


def _path_arg(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HLE smoke eval through parallel shards.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_parallel_shard_eval_20260616")
    parser.add_argument("--total-sample-size", type=int, default=30)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--seed-offset", type=int, default=3000)
    parser.add_argument("--seed-stride", type=int, default=400)
    parser.add_argument("--sample-answer-type", default="")
    parser.add_argument("--sample-subject-contains", default="")
    parser.add_argument("--models", default="gpt-5.4-mini")
    parser.add_argument("--variants", default="raw,assumption_agent_recursive_verify,hipporag_baseline")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--call-timeout", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--graph-dir", default=str(Path("phase four/assumption_graph")))
    parser.add_argument("--agent-top-k", type=int, default=5)
    parser.add_argument("--agent-context-max-chars", type=int, default=2800)
    parser.add_argument("--agent-child-mode", choices=["serial", "parallel_quorum"], default=os.environ.get("HLE_AGENT_CHILD_MODE", "parallel_quorum"))
    parser.add_argument("--agent-child-timeout", type=float, default=None)
    parser.add_argument("--disable-evidence-bridge", action="store_true")
    parser.add_argument("--exclude-existing-hle-artifacts", action="store_true")
    parser.add_argument(
        "--exclude-artifact-glob",
        default="phase four/assumption_graph/paper_readiness_20260604/hle*.json*",
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--md-dir", default=str(DEFAULT_MD_DIR))
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument("--heartbeat-out", default="")
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--heartbeat-interval-sec", type=float, default=10.0)
    parser.add_argument("--soft-timeout-sec", type=float, default=None)
    parser.add_argument("--terminate-grace-sec", type=float, default=30.0)
    parser.add_argument("--model-router-attempts", type=int, default=None)
    parser.add_argument("--model-router-timeout", type=float, default=None)
    parser.add_argument("--model-router-per-attempt-timeout", type=float, default=None)
    parser.add_argument("--model-router-backoff-base-sec", type=float, default=None)
    parser.add_argument("--model-router-global-concurrency", type=int, default=None)
    parser.add_argument("--model-router-global-concurrency-dir", default="")
    parser.add_argument("--model-router-global-slot-ttl-sec", type=float, default=None)
    parser.add_argument("--model-router-global-slot-wait-sec", type=float, default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    specs, states = build_payload_without_execution(args)
    run_dir = _path_arg(args.run_dir, root=root)
    out = _path_arg(args.out, root=root) if args.out else run_dir / f"{args.eval_id}.json"
    md_out = _path_arg(args.md_out, root=root) if args.md_out else _path_arg(args.md_dir, root=root) / f"{args.eval_id}.md"
    heartbeat_path = (
        _path_arg(args.heartbeat_out, root=root)
        if args.heartbeat_out
        else run_dir / f"{args.eval_id}.heartbeat.json"
    )
    env = build_runner_env(
        model_router_attempts=args.model_router_attempts,
        model_router_timeout=args.model_router_timeout,
        model_router_per_attempt_timeout=args.model_router_per_attempt_timeout,
        model_router_backoff_base_sec=args.model_router_backoff_base_sec,
        model_router_global_concurrency=args.model_router_global_concurrency,
        model_router_global_concurrency_dir=args.model_router_global_concurrency_dir,
        model_router_global_slot_ttl_sec=args.model_router_global_slot_ttl_sec,
        model_router_global_slot_wait_sec=args.model_router_global_slot_wait_sec,
    )
    run_parallel_shards(
        root=root,
        shard_states=states,
        parallel_workers=args.parallel_workers,
        heartbeat_path=heartbeat_path,
        poll_interval_sec=args.poll_interval_sec,
        heartbeat_interval_sec=args.heartbeat_interval_sec,
        soft_timeout_sec=args.soft_timeout_sec,
        terminate_grace_sec=args.terminate_grace_sec,
        env=env,
    )
    payloads = load_shard_payloads(specs)
    payload = aggregate_parallel_payload(
        eval_id=args.eval_id,
        specs=specs,
        states=states,
        shard_payloads=payloads,
        execute_live=args.execute_live,
        models=args.models,
        variants=args.variants,
        total_sample_size=args.total_sample_size,
        shard_size=args.shard_size,
        parallel_workers=args.parallel_workers,
        soft_timeout_sec=args.soft_timeout_sec,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_parallel_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "paper_clean_pass": payload["paper_clean_pass"],
        "pollution_pass": payload["pollution_pass"],
        "metrics": {
            "sample_count": payload["metrics"]["sample_count"],
            "distinct_sample_problem_count": payload["metrics"]["distinct_sample_problem_count"],
            "duplicate_sample_problem_count": payload["metrics"]["duplicate_sample_problem_count"],
            "scored_row_count": payload["metrics"]["scored_row_count"],
            "overall_accuracy": payload["metrics"]["overall_accuracy"],
            "resolved_live_model_calls": payload["metrics"]["resolved_live_model_calls"],
            "planned_live_model_calls": payload["metrics"]["planned_live_model_calls"],
        },
        "error_stratification": {
            "top_level_error_count": payload["error_stratification"]["top_level_error_count"],
            "process_timeout_count": payload["error_stratification"]["process_timeout_count"],
        },
        "pollution_audit": {
            "recommended_hle_claim_scope": payload["pollution_audit"]["claim_guard"]["recommended_hle_claim_scope"],
            "failed_gates": payload["pollution_failed_gates"],
        },
        "failed_gates": payload["failed_gates"],
        "paper_clean_failed_gates": payload["paper_clean_failed_gates"],
        "out": str(out),
        "heartbeat_out": str(heartbeat_path),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
