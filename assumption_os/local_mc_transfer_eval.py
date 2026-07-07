"""Local multiple-choice transfer evaluation for the HLE answer-time harness.

This runner is intentionally dataset-agnostic: it consumes a local JSONL file
with multiple-choice questions and reuses the same raw/HippoRAG/agent variants
used by the HLE smoke eval.  Artifacts store hashes and aggregate metadata, not
raw questions, options, predictions, or gold answers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .autonomy_journal import stable_hash
from .hle_smoke_eval import (
    DEFAULT_GRAPH_DIR,
    _HleProblem,
    _JsonlLogger,
    _attach_same_run_baseline_cache,
    _build_assumption_agent_plan,
    _build_hipporag_baseline_plan,
    _call_budget_matched_control_answer,
    _call_model,
    _call_recursive_verified_answer,
    _component_efficacy_from_plan,
    _default_call_timeout,
    _error_row,
    _is_budget_matched_control_variant,
    _metrics,
    _module_trace,
    _prompt_for,
    _score_prediction,
    _update_same_run_baseline_cache,
    _variant_execution_watchdog,
    _variant_watchdog_mark_error,
    _variant_watchdog_model_call_budget_from_env,
    _variant_watchdog_recursive_selection_reserved_budget_from_env,
    _variant_watchdog_summary,
    _variant_watchdog_timeout_from_env,
    apply_hle_offline_defaults_to_environ,
)


DEFAULT_OUT = Path("phase four/assumption_graph/local_mc_transfer_eval_20260630.json")
DEFAULT_MD_OUT = Path("reconstruction/md/local_mc_transfer_eval_20260630.md")
DEFAULT_LOG_OUT = Path("phase four/assumption_graph/local_mc_transfer_eval_20260630.jsonl")


def build_local_mc_transfer_eval_payload(
    *,
    root: Path,
    input_jsonl: Path,
    eval_id: str = "local_mc_transfer_eval_20260630",
    sample_size: int = 24,
    seed_offset: int = 0,
    models: list[str] | None = None,
    variants: list[str] | None = None,
    execute_live: bool = False,
    call_timeout: float | None = None,
    max_tokens: int = 512,
    log_out: Path | None = None,
    graph_dir: Path | None = None,
    agent_top_k: int = 5,
    agent_context_max_chars: int = 2800,
    agent_child_mode: str = "parallel_quorum",
    agent_child_timeout: float | None = None,
    evidence_bridge_enabled: bool = True,
    variant_total_timeout_sec: float | None = None,
    variant_total_model_call_budget: int | None = None,
    force_cache_only_sources: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    input_jsonl = input_jsonl if input_jsonl.is_absolute() else root / input_jsonl
    graph_dir = graph_dir or (root / DEFAULT_GRAPH_DIR)
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    models = models or ["gpt-5.4-mini"]
    variants = variants or [
        "raw",
        "hipporag_baseline",
        "raw_budget_matched",
        "hipporag_budget_matched",
        "assumption_agent_recursive_verify",
    ]
    if force_cache_only_sources:
        _apply_transfer_cache_only_source_policy(os.environ)
    apply_hle_offline_defaults_to_environ(os.environ)
    transfer_runtime_defaults = (
        _apply_local_transfer_live_runtime_defaults(os.environ)
        if execute_live
        else _local_transfer_live_runtime_defaults_summary(activated=False)
    )
    agent_model_call_timeout = call_timeout
    if execute_live and agent_model_call_timeout is None:
        agent_model_call_timeout = _local_transfer_default_agent_model_call_timeout(os.environ)
        transfer_runtime_defaults["agent_model_call_timeout_default_applied"] = agent_model_call_timeout is not None
        transfer_runtime_defaults["agent_model_call_timeout_sec"] = agent_model_call_timeout
    if execute_live and agent_child_timeout is None:
        agent_child_timeout = _local_transfer_default_agent_child_timeout(
            os.environ,
            agent_model_call_timeout=agent_model_call_timeout,
        )
        transfer_runtime_defaults["agent_child_timeout_default_applied"] = agent_child_timeout is not None
        transfer_runtime_defaults["agent_child_timeout_sec"] = agent_child_timeout
    logger = _JsonlLogger(log_out) if log_out else None
    sample_rows = _load_local_mc_sample(
        input_jsonl=input_jsonl,
        sample_size=sample_size,
        seed_offset=seed_offset,
    )
    variant_total_timeout_sec = (
        variant_total_timeout_sec
        if variant_total_timeout_sec is not None
        else _variant_watchdog_timeout_from_env()
    )
    if variant_total_timeout_sec is None and execute_live:
        variant_total_timeout_sec = _local_transfer_default_variant_timeout()
    if variant_total_model_call_budget is None:
        variant_total_model_call_budget = _variant_watchdog_model_call_budget_from_env()
    if variant_total_model_call_budget is None and execute_live:
        variant_total_model_call_budget = _local_transfer_default_model_call_budget()
    variant_recursive_selection_reserved_model_call_budget = (
        _variant_watchdog_recursive_selection_reserved_budget_from_env(
            variant_total_model_call_budget
        )
        if variant_total_model_call_budget is not None
        else 0
    )
    api_summary = {
        "execute_live_requested": execute_live,
        "planned_live_model_calls": len(sample_rows) * len(models) * len(variants) if execute_live else 0,
        "live_model_calls_executed": 0,
        "live_model_call_errors": [],
        "call_timeout_sec": call_timeout if call_timeout is not None else _default_call_timeout(),
        "max_tokens": max_tokens,
        "diagnostic_log_out": str(log_out) if log_out else None,
        "graph_dir": str(graph_dir),
        "agent_top_k": agent_top_k,
        "agent_context_max_chars": agent_context_max_chars,
        "agent_child_mode": agent_child_mode,
        "agent_child_timeout_sec": agent_child_timeout if agent_child_timeout is not None else call_timeout,
        "agent_model_call_timeout_sec": agent_model_call_timeout,
        "evidence_bridge_enabled": evidence_bridge_enabled,
        "underlying_model_calls_executed": 0,
        "variant_total_timeout_sec": variant_total_timeout_sec,
        "variant_total_model_call_budget": variant_total_model_call_budget,
        "variant_recursive_selection_reserved_model_call_budget": (
            variant_recursive_selection_reserved_model_call_budget
        ),
        "variant_watchdog_enabled": bool(
            variant_total_timeout_sec is not None or variant_total_model_call_budget is not None
        ),
        "force_cache_only_sources": bool(force_cache_only_sources),
        "subprocess_model_calls_enabled": _env_truthy(os.environ, "MODEL_ROUTER_SUBPROCESS_CALLS"),
        "subprocess_no_byte_timeout_sec": _optional_float_env(
            os.environ,
            "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC",
            "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT",
            "MODEL_ROUTER_NO_BYTE_TIMEOUT_SEC",
        ),
        "recursive_child_batch_max_wait_sec": _optional_float_env(
            os.environ,
            "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC",
            "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC",
        ),
        "recursive_late_child_model_call_budget": _optional_int_env(
            os.environ,
            "HLE_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET",
            "HLE_RECURSIVE_CHILD_TOTAL_MODEL_CALL_BUDGET",
        ),
        "recursive_selection_model_call_budget": _optional_int_env(
            os.environ,
            "HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET",
            "HLE_RECURSIVE_SELECTION_ADJUDICATOR_MODEL_CALL_BUDGET",
        ),
        "recursive_selection_wallclock_budget_sec": _optional_float_env(
            os.environ,
            "HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC",
            "HLE_RECURSIVE_SELECTION_TOTAL_WALLCLOCK_SEC",
            "HLE_RECURSIVE_SELECTION_TOTAL_TIMEOUT_SEC",
        ),
        "agent_parallel_child_max_workers": _optional_int_env(
            os.environ,
            "HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS",
        ),
        "timeout_recovery_timeout_sec": _optional_float_env(
            os.environ,
            "HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC",
        ),
        "timeout_recovery_max_tokens": _optional_int_env(
            os.environ,
            "HLE_TIMEOUT_RECOVERY_MAX_TOKENS",
        ),
        "local_transfer_runtime_defaults": transfer_runtime_defaults,
    }
    run_rows: list[dict[str, Any]] = []
    if execute_live and sample_rows:
        same_run_baseline_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for problem in sample_rows:
            for model in models:
                for variant in variants:
                    call_id = stable_hash({
                        "eval_id": eval_id,
                        "problem_id_hash": problem["id_hash"],
                        "model": model,
                        "variant": variant,
                    })
                    agent_plan = None
                    variant_plan = None
                    if variant.startswith("assumption_agent"):
                        agent_plan = _build_assumption_agent_plan(
                            root=root,
                            graph_dir=graph_dir,
                            problem=problem,
                            eval_id=eval_id,
                            call_id=call_id,
                            model=model,
                            agent_variant=variant,
                            logger=logger,
                            top_k=agent_top_k,
                            context_max_chars=agent_context_max_chars,
                        )
                        variant_plan = agent_plan
                        _attach_same_run_baseline_cache(
                            agent_plan=agent_plan,
                            cache=same_run_baseline_cache,
                            problem=problem,
                            model=model,
                        )
                    elif variant.startswith("hipporag"):
                        variant_plan = _build_hipporag_baseline_plan(
                            problem=problem,
                            eval_id=eval_id,
                            call_id=call_id,
                            model=model,
                            logger=logger,
                            context_max_chars=agent_context_max_chars,
                        )
                    module_trace = _module_trace(problem, variant=variant, agent_plan=variant_plan)
                    watchdog_manager = _variant_execution_watchdog(
                        eval_id=eval_id,
                        call_id=call_id,
                        problem=problem,
                        model=model,
                        variant=variant,
                        logger=logger,
                        timeout_sec=variant_total_timeout_sec,
                        model_call_budget=variant_total_model_call_budget,
                    )
                    variant_watchdog = watchdog_manager.__enter__()
                    _write_transfer_event(
                        logger,
                        {
                            "event": "transfer_call_start",
                            "eval_id": eval_id,
                            "call_id": call_id,
                            "problem_id_hash": problem["id_hash"],
                            "question_hash": problem["question_hash"],
                            "model": model,
                            "variant": variant,
                            "answer_type": problem["answer_type"],
                            "module_trace": module_trace,
                            "variant_watchdog": _variant_watchdog_summary(variant_watchdog),
                        },
                    )
                    started = time.monotonic()
                    try:
                        if variant == "assumption_agent_recursive_verify":
                            solved = _call_recursive_verified_answer(
                                problem=problem,
                                model=model,
                                agent_plan=agent_plan or {},
                                eval_id=eval_id,
                                call_id=call_id,
                                logger=logger,
                                timeout=agent_model_call_timeout,
                                child_mode=agent_child_mode,
                                child_timeout=agent_child_timeout,
                                max_tokens=max_tokens,
                                evidence_bridge_enabled=evidence_bridge_enabled,
                            )
                            answer_text = solved["answer_text"]
                            api_summary["underlying_model_calls_executed"] += solved["underlying_model_calls"]
                            module_trace = _module_trace(problem, variant=variant, agent_plan=agent_plan)
                        elif _is_budget_matched_control_variant(variant):
                            variant_plan = variant_plan or {"stages": {}}
                            solved = _call_budget_matched_control_answer(
                                problem=problem,
                                model=model,
                                variant=variant,
                                variant_plan=variant_plan,
                                eval_id=eval_id,
                                call_id=call_id,
                                logger=logger,
                                timeout=call_timeout,
                                max_tokens=max_tokens,
                            )
                            answer_text = solved["answer_text"]
                            api_summary["underlying_model_calls_executed"] += solved["underlying_model_calls"]
                            module_trace = _module_trace(problem, variant=variant, agent_plan=variant_plan)
                        else:
                            answer_text = _call_model(
                                model=model,
                                prompt=_prompt_for(problem, variant=variant, agent_plan=variant_plan),
                                timeout=call_timeout,
                                max_tokens=max_tokens,
                            )
                            api_summary["underlying_model_calls_executed"] += 1
                        latency = round(time.monotonic() - started, 4)
                        if variant_watchdog.get("status") == "running":
                            variant_watchdog["status"] = "completed"
                        api_summary["live_model_calls_executed"] += 1
                        row = _score_prediction(
                            problem=problem,
                            model=model,
                            variant=variant,
                            prediction=answer_text,
                            module_trace=module_trace,
                            call_metadata={
                                "call_id": call_id,
                                "latency_sec": latency,
                                "timeout_sec": (
                                    agent_model_call_timeout
                                    if variant == "assumption_agent_recursive_verify"
                                    else api_summary["call_timeout_sec"]
                                ),
                                "max_tokens": max_tokens,
                                "agent_plan_hash": stable_hash(variant_plan or {}),
                                "variant_watchdog": _variant_watchdog_summary(variant_watchdog),
                            },
                        )
                        row["component_efficacy"] = _component_efficacy_from_plan(
                            problem=problem,
                            variant=variant,
                            plan=variant_plan or {},
                            correct=bool(row["correct"]),
                            error=None,
                        )
                        _update_same_run_baseline_cache(
                            cache=same_run_baseline_cache,
                            problem=problem,
                            model=model,
                            variant=variant,
                            prediction=answer_text,
                            plan=variant_plan or {},
                        )
                        run_rows.append(row)
                        _write_transfer_event(
                            logger,
                            {
                                "event": "transfer_call_end",
                                "eval_id": eval_id,
                                "call_id": call_id,
                                "problem_id_hash": problem["id_hash"],
                                "model": model,
                                "variant": variant,
                                "latency_sec": latency,
                                "correct": row["correct"],
                                "prediction_hash": row["prediction_hash"],
                                "module_trace": module_trace,
                                "component_efficacy": row["component_efficacy"],
                                "variant_watchdog": _variant_watchdog_summary(variant_watchdog),
                            },
                        )
                    except Exception as exc:  # pragma: no cover - live API path.
                        _variant_watchdog_mark_error(variant_watchdog, exc=exc)
                        latency = round(time.monotonic() - started, 4)
                        api_summary["live_model_call_errors"].append({
                            "problem_id_hash": problem["id_hash"],
                            "model": model,
                            "variant": variant,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:300],
                            "latency_sec": latency,
                        })
                        error_row = _error_row(
                            problem=problem,
                            model=model,
                            variant=variant,
                            exc=exc,
                            module_trace=module_trace,
                            call_metadata={
                                "call_id": call_id,
                                "latency_sec": latency,
                                "timeout_sec": (
                                    agent_model_call_timeout
                                    if variant == "assumption_agent_recursive_verify"
                                    else api_summary["call_timeout_sec"]
                                ),
                                "max_tokens": max_tokens,
                                "agent_plan_hash": stable_hash(variant_plan or {}),
                                "variant_watchdog": _variant_watchdog_summary(variant_watchdog),
                            },
                        )
                        error_row["component_efficacy"] = _component_efficacy_from_plan(
                            problem=problem,
                            variant=variant,
                            plan=variant_plan or {},
                            correct=False,
                            error={"type": type(exc).__name__},
                        )
                        run_rows.append(error_row)
                        _write_transfer_event(
                            logger,
                            {
                                "event": "transfer_call_error",
                                "eval_id": eval_id,
                                "call_id": call_id,
                                "problem_id_hash": problem["id_hash"],
                                "model": model,
                                "variant": variant,
                                "latency_sec": latency,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:300],
                                "module_trace": module_trace,
                                "component_efficacy": error_row["component_efficacy"],
                                "variant_watchdog": _variant_watchdog_summary(variant_watchdog),
                            },
                        )
                    watchdog_manager.__exit__(None, None, None)
    metrics = _metrics(sample_rows=sample_rows, run_rows=run_rows, api_summary=api_summary)
    gates = {
        "local_dataset_loaded": bool(sample_rows),
        "multiple_choice_only": all(row.get("answer_type") == "multipleChoice" for row in sample_rows),
        "no_raw_content_persisted": metrics["raw_content_persisted"] is False,
        "live_attempts_resolved_if_requested": (
            not execute_live
            or metrics["resolved_live_model_calls"] == metrics["planned_live_model_calls"]
        ),
        "score_rows_complete_if_live": (
            not execute_live
            or metrics["scored_row_count"] == metrics["planned_live_model_calls"]
        ),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "local_mc_transfer_eval",
        "dataset": "local_jsonl",
        "dataset_path_hash": stable_hash({"input_jsonl": str(input_jsonl)}),
        "performance_validation": bool(execute_live),
        "validation_scope": (
            "Runs a local non-HLE multiple-choice transfer set through the HLE answer-time harness. "
            "Artifacts store only hashes, counts, and metadata."
        ),
        "sampling": {
            "requested_sample_size": sample_size,
            "seed_offset": seed_offset,
            "sample_count": len(sample_rows),
            "sample_problem_hashes": [row["id_hash"] for row in sample_rows],
            "raw_content_persisted": False,
        },
        "models": models,
        "variants": variants,
        "runtime_policy": {
            "execute_live": execute_live,
            "force_cache_only_sources": bool(force_cache_only_sources),
            "variant_watchdog": {
                "enabled": bool(
                    variant_total_timeout_sec is not None
                    or variant_total_model_call_budget is not None
                ),
                "total_timeout_sec": variant_total_timeout_sec,
                "total_model_call_budget": variant_total_model_call_budget,
                "recursive_selection_reserved_model_call_budget": (
                    variant_recursive_selection_reserved_model_call_budget
                ),
                "raw_content_persisted": False,
            },
            "local_transfer_runtime_defaults": transfer_runtime_defaults,
            "subprocess_model_calls_enabled": _env_truthy(os.environ, "MODEL_ROUTER_SUBPROCESS_CALLS"),
            "subprocess_no_byte_timeout_sec": _optional_float_env(
                os.environ,
                "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC",
                "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT",
                "MODEL_ROUTER_NO_BYTE_TIMEOUT_SEC",
            ),
            "recursive_child_batch_max_wait_sec": _optional_float_env(
                os.environ,
                "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC",
                "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC",
            ),
            "recursive_late_child_model_call_budget": _optional_int_env(
                os.environ,
                "HLE_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET",
                "HLE_RECURSIVE_CHILD_TOTAL_MODEL_CALL_BUDGET",
            ),
            "recursive_selection_model_call_budget": _optional_int_env(
                os.environ,
                "HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET",
                "HLE_RECURSIVE_SELECTION_ADJUDICATOR_MODEL_CALL_BUDGET",
            ),
            "recursive_selection_wallclock_budget_sec": _optional_float_env(
                os.environ,
                "HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC",
                "HLE_RECURSIVE_SELECTION_TOTAL_WALLCLOCK_SEC",
                "HLE_RECURSIVE_SELECTION_TOTAL_TIMEOUT_SEC",
            ),
            "agent_parallel_child_max_workers": _optional_int_env(
                os.environ,
                "HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS",
            ),
            "timeout_recovery_timeout_sec": _optional_float_env(
                os.environ,
                "HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC",
            ),
            "timeout_recovery_max_tokens": _optional_int_env(
                os.environ,
                "HLE_TIMEOUT_RECOVERY_MAX_TOKENS",
            ),
            "raw_content_persisted": False,
        },
        "api_summary": api_summary,
        "rows": run_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "raw_content_persisted": False,
    }


def _load_local_mc_sample(*, input_jsonl: Path, sample_size: int, seed_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for scanned, line in enumerate(handle, start=1):
            if scanned <= seed_offset:
                continue
            if len(rows) >= sample_size:
                break
            if not line.strip():
                continue
            raw = json.loads(line)
            problem = _local_mc_problem_from_row(raw, scanned_index=scanned)
            if problem is not None:
                rows.append(problem)
    return rows


def _local_mc_problem_from_row(raw: dict[str, Any], *, scanned_index: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    answer = str(raw.get("answer") or raw.get("label") or raw.get("gold") or "").strip()
    if not answer:
        return None
    question = str(raw.get("question") or raw.get("prompt") or "").strip()
    choices = raw.get("choices") or raw.get("options")
    if isinstance(choices, dict):
        normalized_choices = {str(k).strip().upper()[:1]: str(v).strip() for k, v in choices.items()}
    elif isinstance(choices, list):
        normalized_choices = {
            chr(ord("A") + index): str(value).strip()
            for index, value in enumerate(choices)
        }
    else:
        normalized_choices = {}
    if normalized_choices and not _question_contains_labeled_options(question):
        option_lines = [f"{label}. {text}" for label, text in sorted(normalized_choices.items())]
        question = question + "\n" + "\n".join(option_lines)
    if not question or not normalized_choices:
        return None
    problem_id = str(raw.get("id") or raw.get("uid") or scanned_index)
    category = str(raw.get("category") or raw.get("domain") or "local_mc_transfer")
    subject = str(raw.get("subject") or raw.get("raw_subject") or category)
    source = str(raw.get("source") or raw.get("dataset") or "local_jsonl")
    question_hash = stable_hash({"local_mc_question": question})
    answer_hash = stable_hash({"local_mc_answer": answer})
    return _HleProblem({
        "id_hash": stable_hash({"local_mc_id": problem_id, "question_hash": question_hash}),
        "question_hash": question_hash,
        "answer_hash": answer_hash,
        "_question": question,
        "_answer": answer,
        "answer_type": "multipleChoice",
        "category": category,
        "raw_subject": subject,
        "source": source,
        "scanned_index": scanned_index,
        "raw_question_persisted": False,
        "gold_answer_persisted": False,
    })


def _question_contains_labeled_options(question: str) -> bool:
    return bool(re.search(r"(?m)^\s*[A-D][\).\:]\s+\S+", question or ""))


def _apply_transfer_cache_only_source_policy(env: dict[str, str] | os._Environ[str]) -> None:
    env["HLE_EVIDENCE_SOURCE_CACHE_ONLY"] = "1"
    env["HLE_SOURCE_SEARCH_CACHE_ONLY"] = "1"
    env["HLE_DISABLE_LIVE_SOURCE_SEARCH"] = "1"
    env["HLE_ALLOW_LIVE_SOURCE_SEARCH"] = "0"
    env.pop("SEMANTIC_SCHOLAR_API_KEY", None)
    env.pop("OPENALEX_API_KEY", None)
    env.pop("HLE_SEMANTIC_SCHOLAR_API_KEY", None)
    env.pop("HLE_OPENALEX_API_KEY", None)


def _env_has_nonempty(env: dict[str, str] | os._Environ[str], *names: str) -> bool:
    return any(str(env.get(name, "")).strip() for name in names)


def _env_truthy(env: dict[str, str] | os._Environ[str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _optional_float_env(
    env: dict[str, str] | os._Environ[str],
    *names: str,
) -> float | None:
    for name in names:
        raw = str(env.get(name, "")).strip()
        if not raw:
            continue
        if raw.lower() in {"none", "null", "off", "false", "no", "0"}:
            return None
        try:
            return max(1.0, float(raw))
        except ValueError:
            return None
    return None


def _optional_int_env(
    env: dict[str, str] | os._Environ[str],
    *names: str,
) -> int | None:
    for name in names:
        raw = str(env.get(name, "")).strip()
        if not raw:
            continue
        if raw.lower() in {"none", "null", "off", "false", "no", "0"}:
            return None
        try:
            return max(1, int(raw))
        except ValueError:
            return None
    return None


def _local_transfer_live_runtime_defaults_summary(*, activated: bool) -> dict[str, Any]:
    return {
        "status": "activated" if activated else "not_required",
        "policy": "local_mc_transfer_live_budget_reserve_subprocess_batch_cap_defaults_v3",
        "variant_recursive_selection_reserve_applied": False,
        "weak_source_structural_audit_skip_applied": False,
        "subprocess_model_calls_applied": False,
        "subprocess_no_byte_timeout_applied": False,
        "subprocess_no_byte_timeout_sec": None,
        "recursive_child_batch_max_wait_applied": False,
        "recursive_child_batch_max_wait_sec": None,
        "recursive_late_child_model_call_budget_applied": False,
        "recursive_late_child_model_call_budget": None,
        "recursive_selection_model_call_budget_applied": False,
        "recursive_selection_model_call_budget": None,
        "recursive_selection_wallclock_budget_applied": False,
        "recursive_selection_wallclock_budget_sec": None,
        "agent_parallel_child_max_workers_applied": False,
        "agent_parallel_child_max_workers": None,
        "timeout_recovery_timeout_applied": False,
        "timeout_recovery_timeout_sec": None,
        "timeout_recovery_max_tokens_applied": False,
        "timeout_recovery_max_tokens": None,
        "agent_model_call_timeout_default_applied": False,
        "agent_model_call_timeout_sec": None,
        "agent_child_timeout_default_applied": False,
        "agent_child_timeout_sec": None,
        "raw_content_persisted": False,
    }


def _apply_local_transfer_live_runtime_defaults(
    env: dict[str, str] | os._Environ[str],
) -> dict[str, Any]:
    summary = _local_transfer_live_runtime_defaults_summary(activated=True)
    reserve_env_names = (
        "HLE_VARIANT_RECURSIVE_SELECTION_RESERVED_MODEL_CALL_BUDGET",
        "HLE_VARIANT_SELECTION_RESERVED_MODEL_CALL_BUDGET",
        "HLE_VARIANT_RESERVED_SELECTION_MODEL_CALLS",
    )
    if not _env_has_nonempty(env, *reserve_env_names):
        reserve = _local_transfer_default_recursive_selection_reserved_model_calls()
        if reserve > 0:
            env["HLE_VARIANT_RECURSIVE_SELECTION_RESERVED_MODEL_CALL_BUDGET"] = str(reserve)
            summary["variant_recursive_selection_reserve_applied"] = True
            summary["variant_recursive_selection_reserved_model_call_budget"] = reserve
    if not _env_has_nonempty(env, "HLE_WEAK_SOURCE_FALLBACK_CASCADE_SKIP_STRUCTURAL_AUDIT"):
        if _local_transfer_default_skip_structural_audit_on_weak_source():
            env["HLE_WEAK_SOURCE_FALLBACK_CASCADE_SKIP_STRUCTURAL_AUDIT"] = "1"
            summary["weak_source_structural_audit_skip_applied"] = True
    if not _env_has_nonempty(env, "MODEL_ROUTER_SUBPROCESS_CALLS"):
        env["MODEL_ROUTER_SUBPROCESS_CALLS"] = "1"
        summary["subprocess_model_calls_applied"] = True
    no_byte_env_names = (
        "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC",
        "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT",
        "MODEL_ROUTER_NO_BYTE_TIMEOUT_SEC",
    )
    if not _env_has_nonempty(env, *no_byte_env_names):
        no_byte_timeout = _local_transfer_default_subprocess_no_byte_timeout()
        if no_byte_timeout is not None:
            env["MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC"] = str(no_byte_timeout)
            summary["subprocess_no_byte_timeout_applied"] = True
            summary["subprocess_no_byte_timeout_sec"] = no_byte_timeout
    else:
        summary["subprocess_no_byte_timeout_sec"] = _optional_float_env(env, *no_byte_env_names)
    batch_wait_env_names = (
        "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC",
        "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC",
    )
    if not _env_has_nonempty(env, *batch_wait_env_names):
        batch_wait_sec = _local_transfer_default_recursive_child_batch_max_wait()
        if batch_wait_sec is not None:
            env["HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC"] = str(batch_wait_sec)
            summary["recursive_child_batch_max_wait_applied"] = True
            summary["recursive_child_batch_max_wait_sec"] = batch_wait_sec
    else:
        summary["recursive_child_batch_max_wait_sec"] = _optional_float_env(env, *batch_wait_env_names)
    late_child_budget_env_names = (
        "HLE_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET",
        "HLE_RECURSIVE_CHILD_TOTAL_MODEL_CALL_BUDGET",
    )
    if not _env_has_nonempty(env, *late_child_budget_env_names):
        late_child_budget = _local_transfer_default_recursive_late_child_model_call_budget()
        if late_child_budget is not None:
            env["HLE_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET"] = str(late_child_budget)
            summary["recursive_late_child_model_call_budget_applied"] = True
            summary["recursive_late_child_model_call_budget"] = late_child_budget
    else:
        summary["recursive_late_child_model_call_budget"] = _optional_int_env(
            env,
            *late_child_budget_env_names,
        )
    selection_budget_env_names = (
        "HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET",
        "HLE_RECURSIVE_SELECTION_ADJUDICATOR_MODEL_CALL_BUDGET",
    )
    if not _env_has_nonempty(env, *selection_budget_env_names):
        selection_budget = _local_transfer_default_recursive_selection_model_call_budget()
        if selection_budget is not None:
            env["HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET"] = str(selection_budget)
            summary["recursive_selection_model_call_budget_applied"] = True
            summary["recursive_selection_model_call_budget"] = selection_budget
    else:
        summary["recursive_selection_model_call_budget"] = _optional_int_env(
            env,
            *selection_budget_env_names,
        )
    selection_wallclock_env_names = (
        "HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC",
        "HLE_RECURSIVE_SELECTION_TOTAL_WALLCLOCK_SEC",
        "HLE_RECURSIVE_SELECTION_TOTAL_TIMEOUT_SEC",
    )
    if not _env_has_nonempty(env, *selection_wallclock_env_names):
        selection_wallclock = _local_transfer_default_recursive_selection_wallclock_budget()
        if selection_wallclock is not None:
            env["HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC"] = str(selection_wallclock)
            summary["recursive_selection_wallclock_budget_applied"] = True
            summary["recursive_selection_wallclock_budget_sec"] = selection_wallclock
    else:
        summary["recursive_selection_wallclock_budget_sec"] = _optional_float_env(
            env,
            *selection_wallclock_env_names,
        )
    if not _env_has_nonempty(env, "HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS"):
        child_workers = _local_transfer_default_agent_parallel_child_max_workers()
        if child_workers is not None:
            env["HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS"] = str(child_workers)
            summary["agent_parallel_child_max_workers_applied"] = True
            summary["agent_parallel_child_max_workers"] = child_workers
    else:
        summary["agent_parallel_child_max_workers"] = _optional_int_env(
            env,
            "HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS",
        )
    if not _env_has_nonempty(env, "HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC"):
        timeout_recovery_timeout = _local_transfer_default_timeout_recovery_timeout()
        if timeout_recovery_timeout is not None:
            env["HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC"] = str(timeout_recovery_timeout)
            summary["timeout_recovery_timeout_applied"] = True
            summary["timeout_recovery_timeout_sec"] = timeout_recovery_timeout
    else:
        summary["timeout_recovery_timeout_sec"] = _optional_float_env(
            env,
            "HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC",
        )
    if not _env_has_nonempty(env, "HLE_TIMEOUT_RECOVERY_MAX_TOKENS"):
        timeout_recovery_max_tokens = _local_transfer_default_timeout_recovery_max_tokens()
        if timeout_recovery_max_tokens is not None:
            env["HLE_TIMEOUT_RECOVERY_MAX_TOKENS"] = str(timeout_recovery_max_tokens)
            summary["timeout_recovery_max_tokens_applied"] = True
            summary["timeout_recovery_max_tokens"] = timeout_recovery_max_tokens
    else:
        summary["timeout_recovery_max_tokens"] = _optional_int_env(
            env,
            "HLE_TIMEOUT_RECOVERY_MAX_TOKENS",
        )
    return summary


def _local_transfer_default_variant_timeout() -> float:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_VARIANT_TOTAL_TIMEOUT_SEC")
        or os.environ.get("LOCAL_MC_TRANSFER_PER_VARIANT_TIMEOUT_SEC")
        or ""
    ).strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 900.0


def _local_transfer_default_model_call_budget() -> int:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_VARIANT_TOTAL_MODEL_CALL_BUDGET")
        or os.environ.get("LOCAL_MC_TRANSFER_PER_VARIANT_MODEL_CALL_BUDGET")
        or ""
    ).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 12


def _local_transfer_default_recursive_selection_reserved_model_calls() -> int:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_RECURSIVE_SELECTION_RESERVED_MODEL_CALL_BUDGET")
        or os.environ.get("LOCAL_MC_TRANSFER_SELECTION_RESERVED_MODEL_CALLS")
        or ""
    ).strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 1


def _local_transfer_default_skip_structural_audit_on_weak_source() -> bool:
    raw = os.environ.get("LOCAL_MC_TRANSFER_SKIP_STRUCTURAL_AUDIT_ON_WEAK_SOURCE", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _local_transfer_default_subprocess_no_byte_timeout() -> float | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_MODEL_SUBPROCESS_NO_BYTE_TIMEOUT_SEC")
        or os.environ.get("LOCAL_MC_TRANSFER_MODEL_NO_BYTE_TIMEOUT_SEC")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0"}:
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 180.0


def _local_transfer_default_recursive_child_batch_max_wait() -> float | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC")
        or os.environ.get("LOCAL_MC_TRANSFER_CHILD_BATCH_MAX_WAIT_SEC")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 120.0


def _local_transfer_default_recursive_late_child_model_call_budget() -> int | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET")
        or os.environ.get("LOCAL_MC_TRANSFER_LATE_CHILD_MODEL_CALL_BUDGET")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 7


def _local_transfer_default_recursive_selection_model_call_budget() -> int | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_RECURSIVE_SELECTION_MODEL_CALL_BUDGET")
        or os.environ.get("LOCAL_MC_TRANSFER_SELECTION_MODEL_CALL_BUDGET")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 1


def _local_transfer_default_recursive_selection_wallclock_budget() -> float | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC")
        or os.environ.get("LOCAL_MC_TRANSFER_SELECTION_WALLCLOCK_BUDGET_SEC")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 120.0


def _local_transfer_default_agent_parallel_child_max_workers() -> int | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_AGENT_PARALLEL_CHILD_MAX_WORKERS")
        or os.environ.get("LOCAL_MC_TRANSFER_CHILD_MAX_WORKERS")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 2


def _local_transfer_default_timeout_recovery_timeout() -> float | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_TIMEOUT_RECOVERY_TIMEOUT_SEC")
        or os.environ.get("LOCAL_MC_TRANSFER_TIMEOUT_RECOVERY_CALL_TIMEOUT_SEC")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 60.0


def _local_transfer_default_timeout_recovery_max_tokens() -> int | None:
    raw = (
        os.environ.get("LOCAL_MC_TRANSFER_TIMEOUT_RECOVERY_MAX_TOKENS")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(16, int(raw))
        except ValueError:
            pass
    return 64


def _local_transfer_default_agent_model_call_timeout(
    env: dict[str, str] | os._Environ[str],
) -> float | None:
    raw = (
        env.get("LOCAL_MC_TRANSFER_AGENT_MODEL_CALL_TIMEOUT_SEC")
        or env.get("LOCAL_MC_TRANSFER_AGENT_CALL_TIMEOUT_SEC")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return _optional_float_env(
        env,
        "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC",
        "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC",
    )


def _local_transfer_default_agent_child_timeout(
    env: dict[str, str] | os._Environ[str],
    *,
    agent_model_call_timeout: float | None,
) -> float | None:
    raw = (
        env.get("LOCAL_MC_TRANSFER_AGENT_CHILD_TIMEOUT_SEC")
        or env.get("LOCAL_MC_TRANSFER_CHILD_TIMEOUT_SEC")
        or ""
    ).strip()
    if raw.lower() in {"none", "null", "off", "false", "no", "0", "unlimited"}:
        return None
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    if agent_model_call_timeout is not None:
        return agent_model_call_timeout
    return _optional_float_env(
        env,
        "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC",
        "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC",
    )


def _write_transfer_event(logger: _JsonlLogger | None, event: dict[str, Any]) -> None:
    if logger is None:
        return
    event = dict(event)
    event.setdefault("raw_content_persisted", False)
    logger.write(event)


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Local MC Transfer Evaluation",
        "",
        f"- pass: `{payload['pass']}`",
        f"- sample count: `{metrics['sample_count']}`",
        f"- live calls returned: `{metrics['live_model_calls_executed']}/{metrics['planned_live_model_calls']}`",
        f"- live attempts resolved: `{metrics['resolved_live_model_calls']}/{metrics['planned_live_model_calls']}`",
        f"- live call errors: `{metrics['live_model_call_error_count']}`",
        f"- overall accuracy: `{metrics['overall_accuracy']}`",
        f"- variant watchdog: `{metrics.get('variant_watchdog_summary', {})}`",
        f"- raw content persisted: `{metrics['raw_content_persisted']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        "",
        "## By Variant",
        "",
        "| model | variant | n | accuracy | error count |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, row in sorted(metrics["by_model_variant"].items()):
        model, variant = key.split("::", 1)
        lines.append(
            f"| `{model}` | `{variant}` | `{row['n']}` | `{row['accuracy']}` | `{row['error_count']}` |"
        )
    lines.extend([
        "",
        "Raw questions, options, predictions, and gold answers are not persisted.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local non-HLE multiple-choice transfer evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--eval-id", default="local_mc_transfer_eval_20260630")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--models", default="gpt-5.4-mini")
    parser.add_argument(
        "--variants",
        default="raw,hipporag_baseline,raw_budget_matched,hipporag_budget_matched,assumption_agent_recursive_verify",
    )
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--call-timeout", type=float, default=None)
    parser.add_argument("--variant-total-timeout-sec", type=float, default=None)
    parser.add_argument("--variant-total-model-call-budget", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log-out", default=str(DEFAULT_LOG_OUT))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--agent-top-k", type=int, default=5)
    parser.add_argument("--agent-context-max-chars", type=int, default=2800)
    parser.add_argument("--agent-child-mode", choices=["serial", "parallel_quorum"], default="parallel_quorum")
    parser.add_argument("--agent-child-timeout", type=float, default=None)
    parser.add_argument("--disable-evidence-bridge", action="store_true")
    parser.add_argument("--allow-live-source-search", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    log_out = Path(args.log_out)
    log_out = log_out if log_out.is_absolute() else root / log_out
    graph_dir = Path(args.graph_dir)
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    payload = build_local_mc_transfer_eval_payload(
        root=root,
        input_jsonl=Path(args.input_jsonl),
        eval_id=args.eval_id,
        sample_size=args.sample_size,
        seed_offset=args.seed_offset,
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        variants=[item.strip() for item in args.variants.split(",") if item.strip()],
        execute_live=args.execute_live,
        call_timeout=args.call_timeout,
        max_tokens=args.max_tokens,
        log_out=log_out if args.execute_live else None,
        graph_dir=graph_dir,
        agent_top_k=args.agent_top_k,
        agent_context_max_chars=args.agent_context_max_chars,
        agent_child_mode=args.agent_child_mode,
        agent_child_timeout=args.agent_child_timeout,
        evidence_bridge_enabled=not args.disable_evidence_bridge,
        variant_total_timeout_sec=args.variant_total_timeout_sec,
        variant_total_model_call_budget=args.variant_total_model_call_budget,
        force_cache_only_sources=not args.allow_live_source_search,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
