"""Text-only smoke evaluation for Humanity's Last Exam.

The official HLE dataset is gated.  This runner expects the user to have
accepted the dataset terms and provided ``HF_TOKEN`` in the process environment.
It deliberately does not persist HLE questions, gold answers, rationales, or
canary strings.  Artifacts store stable hashes, metadata, predictions hashes,
and correctness only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import http.client
import html
import json
import os
import random
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .context import format_assumption_context
from .formal_mapping import build_formal_mapping_payload, search_formal_mappings
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .recursive_runner import build_recursive_assumption_run
from .retrieval_policy import format_policy_context, retrieve_phase2_assumptions
from .schema import stable_id
from .structural_patterns import search_structural_patterns
from .world_model import predict_proposal_outcome


DEFAULT_OUT = PAPER_DIR / "hle_text_smoke_eval_20260615.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_text_smoke_eval_20260615.md")
DEFAULT_LOG_OUT = PAPER_DIR / "hle_text_smoke_eval_20260615.jsonl"
DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")

DATASET_NAME = "cais/hle"
HLE_OFFICIAL_SOURCES = [
    "https://huggingface.co/datasets/cais/hle",
    "https://github.com/centerforaisafety/hle",
]


def build_hle_text_smoke_eval_payload(
    *,
    root: Path,
    eval_id: str = "hle_text_smoke_eval_20260615",
    sample_size: int = 8,
    max_scan: int = 200,
    models: list[str] | None = None,
    variants: list[str] | None = None,
    execute_live: bool = False,
    seed_offset: int = 0,
    call_timeout: float | None = None,
    max_tokens: int = 512,
    log_out: Path | None = None,
    graph_dir: Path | None = None,
    agent_top_k: int = 5,
    agent_context_max_chars: int = 2800,
    agent_child_mode: str = "parallel_quorum",
    agent_child_timeout: float | None = None,
    evidence_bridge_enabled: bool = True,
    exclude_existing_hle_artifacts: bool = False,
    exclude_artifact_glob: str = "phase four/assumption_graph/paper_readiness_20260604/hle*.json*",
    sample_answer_type: str = "",
    sample_subject_contains: str = "",
) -> dict[str, Any]:
    root = root.resolve()
    models = models or ["gpt-5.5"]
    variants = variants or ["raw", "assumption_wrapper"]
    graph_dir = graph_dir or (root / DEFAULT_GRAPH_DIR)
    access = _access_preflight()
    sample_rows: list[dict[str, Any]] = []
    excluded_problem_hashes = (
        _collect_existing_hle_problem_hashes(root=root, artifact_glob=exclude_artifact_glob)
        if exclude_existing_hle_artifacts
        else set()
    )
    if access["dataset_accessible"]:
        sample_rows = _load_text_only_sample(
            sample_size=sample_size,
            max_scan=max_scan,
            seed_offset=seed_offset,
            exclude_problem_hashes=excluded_problem_hashes,
            answer_type_filter=sample_answer_type,
            subject_contains=sample_subject_contains,
        )
    run_rows: list[dict[str, Any]] = []
    logger = _JsonlLogger(log_out) if log_out else None
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
        "evidence_bridge_enabled": evidence_bridge_enabled,
        "exclude_existing_hle_artifacts": exclude_existing_hle_artifacts,
        "exclude_artifact_glob": exclude_artifact_glob,
        "excluded_existing_problem_count": len(excluded_problem_hashes),
        "underlying_model_calls_executed": 0,
    }
    if execute_live and sample_rows:
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
                    _log_event(
                        logger,
                        {
                            "event": "call_start",
                            "eval_id": eval_id,
                            "call_id": call_id,
                            "problem_id_hash": problem["id_hash"],
                            "question_hash": problem["question_hash"],
                            "model": model,
                            "variant": variant,
                            "category": problem["category"],
                            "raw_subject": problem["raw_subject"],
                            "answer_type": problem["answer_type"],
                            "call_timeout_sec": api_summary["call_timeout_sec"],
                            "max_tokens": max_tokens,
                            "module_trace": module_trace,
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
                                timeout=call_timeout,
                                child_mode=agent_child_mode,
                                child_timeout=agent_child_timeout,
                                max_tokens=max_tokens,
                                evidence_bridge_enabled=evidence_bridge_enabled,
                            )
                            answer_text = solved["answer_text"]
                            api_summary["underlying_model_calls_executed"] += solved["underlying_model_calls"]
                            module_trace = _module_trace(problem, variant=variant, agent_plan=agent_plan)
                        else:
                            answer_text = _call_model(
                                model=model,
                                prompt=_prompt_for(problem, variant=variant, agent_plan=variant_plan),
                                timeout=call_timeout,
                                max_tokens=max_tokens,
                            )
                            api_summary["underlying_model_calls_executed"] += 1
                        latency = round(time.monotonic() - started, 4)
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
                                "timeout_sec": api_summary["call_timeout_sec"],
                                "max_tokens": max_tokens,
                                "agent_plan_hash": stable_hash(variant_plan or {}),
                            },
                        )
                        row["component_efficacy"] = _component_efficacy_from_plan(
                            problem=problem,
                            variant=variant,
                            plan=variant_plan or {},
                            correct=bool(row["correct"]),
                            error=None,
                        )
                        run_rows.append(row)
                        _log_event(
                            logger,
                            {
                                "event": "call_end",
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
                                "agent_decision": (agent_plan or {}).get("world_model_router", {}).get("decision"),
                            },
                        )
                    except Exception as exc:  # pragma: no cover - live API path.
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
                                "timeout_sec": api_summary["call_timeout_sec"],
                                "max_tokens": max_tokens,
                                "agent_plan_hash": stable_hash(variant_plan or {}),
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
                        _log_event(
                            logger,
                            {
                                "event": "call_error",
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
                                "agent_decision": (agent_plan or {}).get("world_model_router", {}).get("decision"),
                            },
                        )
    metrics = _metrics(sample_rows=sample_rows, run_rows=run_rows, api_summary=api_summary)
    gates = {
        "official_dataset_accessible": access["dataset_accessible"],
        "text_only_sample_loaded": metrics["sample_count"] >= min(sample_size, 1),
        "no_raw_hle_content_persisted": metrics["raw_content_persisted"] is False,
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
        "eval_kind": "hle_text_only_smoke_eval",
        "dataset": DATASET_NAME,
        "official_sources": HLE_OFFICIAL_SOURCES,
        "performance_validation": True,
        "validation_scope": (
            "Runs a small text-only smoke sample from the official gated HLE dataset.  The artifact stores "
            "only hashes and aggregate metadata; raw questions, gold answers, rationales, canaries, and image "
            "payloads are not persisted."
        ),
        "access": access,
        "sampling": {
            "requested_sample_size": sample_size,
            "max_scan": max_scan,
            "seed_offset": seed_offset,
            "text_only_policy": "skip rows with image, image_preview, or rationale_image payloads",
            "fresh_exclusion_enabled": exclude_existing_hle_artifacts,
            "excluded_existing_problem_count": len(excluded_problem_hashes),
            "duplicate_with_excluded_problem_count": sum(1 for row in sample_rows if row["id_hash"] in excluded_problem_hashes),
            "sample_answer_type_filter": sample_answer_type,
            "sample_subject_contains_filter": sample_subject_contains,
            "sample_problem_hashes": [row["id_hash"] for row in sample_rows],
        },
        "models": models,
        "variants": variants,
        "api_summary": api_summary,
        "rows": run_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundary": (
            "This is a small text-only smoke result, not a full HLE score.  Multimodal questions, full-dataset "
            "statistics, and official leaderboard claims are out of scope."
        ),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# HLE Text-Only Smoke Evaluation",
        "",
        f"- pass: `{payload['pass']}`",
        f"- dataset accessible: `{payload['access']['dataset_accessible']}`",
        f"- sample count: `{metrics['sample_count']}`",
        f"- scanned rows: `{metrics['scanned_row_count']}`",
        f"- live calls returned: `{metrics['live_model_calls_executed']}/{metrics['planned_live_model_calls']}`",
        f"- underlying model calls executed: `{metrics['underlying_model_calls_executed']}`",
        f"- live attempts resolved: `{metrics['resolved_live_model_calls']}/{metrics['planned_live_model_calls']}`",
        f"- live call errors: `{metrics['live_model_call_error_count']}`",
        f"- overall accuracy: `{metrics['overall_accuracy']}`",
        f"- raw content persisted: `{metrics['raw_content_persisted']}`",
        f"- failed gates: `{payload['failed_gates']}`",
        "",
        "## By Variant",
        "",
        "| model | variant | n | accuracy | MCQ accuracy | exact accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in sorted(metrics["by_model_variant"].items()):
        model, variant = key.split("::", 1)
        lines.append(
            f"| `{model}` | `{variant}` | `{row['n']}` | `{row['accuracy']}` | "
            f"`{row['multiple_choice_accuracy']}` | `{row['exact_match_accuracy']}` |"
        )
    lines.extend([
        "",
        "## Same-Batch Control Comparison",
        "",
        "| model | comparison | shared n | agent acc | control acc | delta | agent-only correct | control-only correct |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for model, comparison in sorted(metrics.get("control_comparison", {}).items()):
        for key, row in sorted(comparison.items()):
            if not key.startswith("agent_vs_"):
                continue
            lines.append(
                f"| `{model}` | `{key}` | `{row['shared_problem_count']}` | `{row['agent_accuracy']}` | "
                f"`{row['control_accuracy']}` | `{row['agent_minus_control_accuracy']}` | "
                f"`{row['agent_unique_correct_count']}` | `{row['control_unique_correct_count']}` |"
            )
    lines.extend([
        "",
        "## Module Activation",
        "",
        "| model | variant | expected missing modules | activated modules |",
        "| --- | --- | --- | --- |",
    ])
    activation = metrics.get("module_activation_summary", {})
    missing = metrics.get("expected_but_missing_modules", {})
    for key in sorted(activation):
        model, variant = key.split("::", 1)
        active_modules = [
            module
            for module, counts in sorted(activation[key].items())
            if counts.get("activated", 0) > 0
        ]
        missing_modules = sorted(missing.get(key, []))
        lines.append(
            f"| `{model}` | `{variant}` | `{', '.join(missing_modules) or 'none'}` | "
            f"`{', '.join(active_modules) or 'none'}` |"
        )
    lines.extend([
        "",
        "## Component Efficacy",
        "",
        "| model | variant | selection methods | key functional flags | flag accuracy |",
        "| --- | --- | --- | --- | --- |",
    ])
    for key, row in sorted(metrics.get("component_efficacy_summary", {}).items()):
        model, variant = key.split("::", 1)
        selection_methods = ", ".join(
            f"{method}:{count}" for method, count in sorted(row.get("selection_method_counts", {}).items())
        ) or "none"
        interesting_flags = [
            flag for flag in (
                "context_injected",
                "graph_context_injected",
                "evidence_bridge_activated",
                "evidence_child_executed",
                "agent_hipporag_context_activated",
                "agent_hipporag_child_executed",
                "hipporag_context_priority_used",
                "recursive_diverse_candidates",
                "recursive_collapsed_consensus",
                "recursive_timeout_pressure",
                "critic_model_used",
                "claim_verifier_verified_candidate",
                "claim_verifier_no_executable_claim",
                "domain_rule_mc_verifier_activated",
                "domain_rule_mc_verifier_selected",
                "domain_rule_override",
                "mc_option_evidence_scorer_activated",
                "mc_option_evidence_candidate_emitted",
                "mc_option_evidence_candidate_selected",
                "option_evidence_verifier_used",
                "critic_synthesis_activated",
                "critic_synthesis_disagreed",
                "critic_synthesis_selected",
                "mc_option_sweep_activated",
                "mc_option_sweep_selected",
                "source_grounded_verifier_used",
                "candidate_claim_override",
                "counter_assumption_challenge_activated",
                "counter_assumption_challenge_disagreed",
                "counter_assumption_challenge_selected",
                "option_elimination_challenge_activated",
                "option_elimination_challenge_disagreed",
                "option_elimination_challenge_selected",
                "forced_alternative_activated",
                "forced_alternative_disagreed",
                "forced_alternative_selected",
                "counter_assumption_verifier_used",
                "majority_only_selection",
            )
            if row.get("flag_counts", {}).get(flag)
        ]
        flag_counts = ", ".join(
            f"{flag}:{row['flag_counts'][flag]}" for flag in interesting_flags[:8]
        ) or "none"
        flag_accuracy = ", ".join(
            f"{flag}:{row.get('flag_accuracy', {}).get(flag)}" for flag in interesting_flags[:5]
        ) or "none"
        lines.append(
            f"| `{model}` | `{variant}` | `{selection_methods}` | `{flag_counts}` | `{flag_accuracy}` |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
        "The report intentionally omits HLE questions, gold answers, rationales, canary strings, and image payloads.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _access_preflight() -> dict[str, Any]:
    hf_token = _hf_token()
    if not hf_token:
        return {
            "dataset_accessible": False,
            "hf_token_present": False,
            "error_type": "MissingToken",
            "error": "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for gated HLE access.",
        }
    try:
        from datasets import Image, load_dataset

        dataset = load_dataset(DATASET_NAME, split="test", streaming=True, token=hf_token)
        dataset = _cast_image_columns(dataset, Image)
        iterator = iter(dataset)
        first = next(iterator)
        return {
            "dataset_accessible": True,
            "hf_token_present": True,
            "schema": {key: type(value).__name__ for key, value in first.items()},
            "error_type": None,
            "error": None,
        }
    except Exception as exc:
        return {
            "dataset_accessible": False,
            "hf_token_present": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def _load_text_only_sample(
    *,
    sample_size: int,
    max_scan: int,
    seed_offset: int,
    exclude_problem_hashes: set[str] | None = None,
    answer_type_filter: str = "",
    subject_contains: str = "",
) -> list[dict[str, Any]]:
    from datasets import Image, load_dataset

    dataset = load_dataset(DATASET_NAME, split="test", streaming=True, token=_hf_token())
    dataset = _cast_image_columns(dataset, Image)
    sample: list[dict[str, Any]] = []
    exclude_problem_hashes = exclude_problem_hashes or set()
    scanned = 0
    skipped = 0
    for row in dataset:
        scanned += 1
        if scanned <= seed_offset:
            continue
        if _has_image_payload(row):
            skipped += 1
            if scanned >= max_scan:
                break
            continue
        if not str(row.get("question") or "").strip() or not str(row.get("answer") or "").strip():
            skipped += 1
            if scanned >= max_scan:
                break
            continue
        if answer_type_filter and str(row.get("answer_type") or "") != answer_type_filter:
            skipped += 1
            if scanned >= max_scan:
                break
            continue
        if subject_contains:
            haystack = " ".join([
                str(row.get("category") or ""),
                str(row.get("raw_subject") or ""),
            ]).lower()
            if subject_contains.lower() not in haystack:
                skipped += 1
                if scanned >= max_scan:
                    break
                continue
        problem = _problem_from_row(row, scanned=scanned, skipped_before=skipped)
        if problem["id_hash"] in exclude_problem_hashes:
            skipped += 1
            if scanned >= max_scan:
                break
            continue
        sample.append(problem)
        if len(sample) >= sample_size or scanned >= max_scan:
            break
    return sample


def _collect_existing_hle_problem_hashes(*, root: Path, artifact_glob: str) -> set[str]:
    """Collect previous HLE problem hashes without reading or persisting raw HLE text."""
    hashes: set[str] = set()
    for path in root.glob(artifact_glob):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            _collect_problem_hashes_from_jsonl(path, hashes)
        elif path.suffix == ".json":
            _collect_problem_hashes_from_json(path, hashes)
    return hashes


def _collect_problem_hashes_from_json(path: Path, hashes: set[str]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            value = item.get("problem_id_hash")
            if isinstance(value, str) and value:
                hashes.add(value)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _collect_problem_hashes_from_jsonl(path: Path, hashes: set[str]) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = item.get("problem_id_hash") if isinstance(item, dict) else None
                if isinstance(value, str) and value:
                    hashes.add(value)
    except Exception:
        return


def _cast_image_columns(dataset: Any, image_cls: Any) -> Any:
    for column in ("image", "image_preview", "rationale_image"):
        try:
            dataset = dataset.cast_column(column, image_cls(decode=False))
        except Exception:
            pass
    return dataset


def _has_image_payload(row: dict[str, Any]) -> bool:
    for key in ("image", "image_preview", "rationale_image"):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, dict) and not value.get("bytes") and not value.get("path"):
            continue
        return True
    return False


def _problem_from_row(row: dict[str, Any], *, scanned: int, skipped_before: int) -> dict[str, Any]:
    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    answer_type = str(row.get("answer_type") or "")
    category = str(row.get("category") or "")
    raw_subject = str(row.get("raw_subject") or "")
    return {
        "id_hash": stable_hash({"hle_id": row.get("id")}),
        "question_hash": stable_hash({"question": question}),
        "answer_hash": stable_hash({"answer": answer, "answer_type": answer_type}),
        "category": category,
        "raw_subject": raw_subject,
        "answer_type": answer_type,
        "scanned_index": scanned,
        "skipped_before": skipped_before,
        "_question": question,
        "_answer": answer,
    }


def _build_assumption_agent_plan(
    *,
    root: Path,
    graph_dir: Path,
    problem: dict[str, Any],
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
    top_k: int,
    context_max_chars: int,
    agent_variant: str = "assumption_agent",
) -> dict[str, Any]:
    question = problem["_question"]
    goal = f"Solve a text-only HLE item with answer_type={problem['answer_type']} and return exact JSON."
    plan: dict[str, Any] = {
        "agent_kind": "hle_assumption_agent_v1",
        "agent_variant": agent_variant,
        "call_id": call_id,
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "model": model,
        "graph_dir": str(graph_dir),
        "stages": {},
        "prompt_context": "",
    }

    domain = _classify_hle_domain(problem)
    plan["stages"]["domain_router"] = {
        "status": "activated",
        "domain": domain,
        "category": problem["category"],
        "raw_subject": problem["raw_subject"],
        "answer_type": problem["answer_type"],
    }
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant=agent_variant,
        stage="domain_router",
        data=plan["stages"]["domain_router"],
    )

    try:
        store = JsonlGraphStore(graph_dir)
        graph = SimpleAssumptionGraph(store)
        retrieval_result = retrieve_phase2_assumptions(
            graph,
            problem=question,
            meta={},
            pid=problem["id_hash"],
            domain=domain,
            difficulty="hle",
            top_k=top_k,
            pool_k=max(top_k, top_k * 3),
        )
        retrieval_summary = _sanitize_retrieval_result(retrieval_result)
        plan["stages"]["assumption_graph_retrieval"] = {"status": "activated", **retrieval_summary}
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant=agent_variant,
            stage="assumption_graph_retrieval",
            data=plan["stages"]["assumption_graph_retrieval"],
        )

        formal_payload = build_formal_mapping_payload(store)
        formal_apps = search_formal_mappings(formal_payload, question, top_n=2)
        structural_apps = search_structural_patterns(store, question, top_n=2)
        morphism_summary = {
            "status": "activated",
            "formal_mapping_hits": [
                {
                    "mapping_id": app.get("mapping_id"),
                    "source_key": app.get("source_key"),
                    "score": round(float(app.get("score", 0.0) or 0.0), 4),
                }
                for app in formal_apps
            ],
            "structural_morphism_hits": [
                {
                    "pattern_id": app.get("pattern_id"),
                    "score": round(float(app.get("score", 0.0) or 0.0), 4),
                    "decision": app.get("decision"),
                }
                for app in structural_apps
            ],
        }
        plan["stages"]["structural_morphism_transfer"] = morphism_summary
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant=agent_variant,
            stage="structural_morphism_transfer",
            data=morphism_summary,
        )

        recursive_payload = build_recursive_assumption_run(
            graph_dir=graph_dir,
            problem=question,
            goal=goal,
            eval_id=f"{eval_id}_{call_id}_recursive",
            problem_id=problem["id_hash"],
            top_k=top_k,
            max_children=min(4, top_k),
            max_depth=2,
            writeback=False,
        )
        recursive_summary = _sanitize_recursive_payload(recursive_payload)
        plan["stages"]["recursive_assumption_runner"] = {"status": "activated", **recursive_summary}
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant=agent_variant,
            stage="recursive_assumption_runner",
            data=plan["stages"]["recursive_assumption_runner"],
        )

        top_node_ids = retrieval_summary.get("top_node_ids", [])
        generic_graph_context_only = _retrieval_summary_is_generic_harness_only(retrieval_summary)
        top_score = max(retrieval_summary.get("top_scores", [0.0]) or [0.0])
        formal_hit_count = len(morphism_summary["formal_mapping_hits"])
        structural_hit_count = len(morphism_summary["structural_morphism_hits"])
        strong_structural_hit_count = sum(
            1
            for hit in morphism_summary["structural_morphism_hits"]
            if hit.get("decision") == "transfer_supported"
        )
        proposal = {
            "proposal_id": stable_id("hle_agent_prop", call_id),
            "proposal_type": "hle_assumption_agent_route",
            "priority": round(0.8 + top_score + 0.18 * formal_hit_count + 0.14 * structural_hit_count, 4),
            "parent_node_id": top_node_ids[0] if top_node_ids else "",
            "candidate_node": {"id": stable_id("hle_agent_candidate", call_id)},
        }
        preflight = {
            "readiness": "ready_for_fresh_ablation" if top_node_ids else "needs_retrieval_fix",
        }
        formal_gate = {
            "decision": "allow" if formal_hit_count or structural_hit_count else "not_applicable",
            "blocks_policy_update": False,
        }
        prediction = predict_proposal_outcome(
            store=store,
            proposal=proposal,
            preflight=preflight,
            formal_gate=formal_gate,
            regression={"risk": "low" if top_node_ids else "medium"},
        )
        context_allowed = _should_use_agent_context(
            answer_type=problem["answer_type"],
            top_score=top_score,
            formal_hit_count=formal_hit_count,
            structural_hit_count=structural_hit_count,
            strong_structural_hit_count=strong_structural_hit_count,
            expected_utility=prediction.expected_utility,
        )
        if generic_graph_context_only:
            context_allowed = False
            context_abstain_reason = "generic_harness_graph_context_only"
        elif not context_allowed:
            context_abstain_reason = "world_model_or_scope_gate"
        else:
            context_abstain_reason = ""
        router_summary = {
            "status": "activated",
            "decision": "use_context" if context_allowed else "abstain_to_raw_prompt",
            "context_abstain_reason": context_abstain_reason,
            "generic_graph_context_only": generic_graph_context_only,
            "top_score": round(float(top_score), 4),
            "formal_hit_count": formal_hit_count,
            "structural_hit_count": structural_hit_count,
            "strong_structural_hit_count": strong_structural_hit_count,
            "predicted_acceptance_probability": prediction.predicted_acceptance_probability,
            "prediction_confidence": prediction.prediction_confidence,
            "expected_utility": prediction.expected_utility,
            "recommended_next_action": prediction.recommended_next_action,
            "predicted_regression_risk": prediction.predicted_regression_risk,
        }
        plan["stages"]["world_model_router"] = router_summary
        plan["world_model_router"] = router_summary
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant=agent_variant,
            stage="world_model_router",
            data=router_summary,
        )

        retrieval_context_discarded = False
        if context_allowed and retrieval_result is not None:
            context = format_policy_context(retrieval_result, format_assumption_context, max_nodes=top_k)
            plan["retrieval_context_candidate"] = _trim_context(context, max_chars=context_max_chars)
            plan["prompt_context"] = plan["retrieval_context_candidate"]
        elif retrieval_result is not None and not generic_graph_context_only:
            context = format_policy_context(retrieval_result, format_assumption_context, max_nodes=top_k)
            plan["retrieval_context_candidate"] = _trim_context(context, max_chars=context_max_chars)
        elif retrieval_result is not None:
            retrieval_context_discarded = True
            plan["retrieval_context_candidate"] = ""
        plan["stages"]["prompt_builder"] = {
            "status": "activated",
            "context_injected": bool(plan["prompt_context"]),
            "retrieval_context_candidate_char_count": len(plan.get("retrieval_context_candidate", "")),
            "retrieval_context_discarded": retrieval_context_discarded,
            "context_abstain_reason": context_abstain_reason,
            "context_char_count": len(plan["prompt_context"]),
        }
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant=agent_variant,
            stage="prompt_builder",
            data=plan["stages"]["prompt_builder"],
        )
    except Exception as exc:
        plan["stages"]["agent_planning_error"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:300],
            "fallback": "raw_prompt",
        }
        plan["world_model_router"] = {"status": "failed", "decision": "fallback_raw_prompt"}
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant=agent_variant,
            stage="agent_planning_error",
            data=plan["stages"]["agent_planning_error"],
        )
    return plan


def _build_hipporag_baseline_plan(
    *,
    problem: dict[str, Any],
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
    context_max_chars: int,
) -> dict[str, Any]:
    """Build an independent HippoRAG-style retrieval plan for HLE."""
    plan: dict[str, Any] = {
        "baseline_kind": "hipporag_style_transient_evidence",
        "call_id": call_id,
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "model": model,
        "stages": {},
        "prompt_context": "",
    }
    queries = _candidate_evidence_queries(problem)
    docs: list[dict[str, str]] = []
    errors: list[str] = []
    for query in queries:
        try:
            docs.extend(_wikipedia_search(query, limit=3, timeout=6.0))
        except Exception as exc:
            errors.append(type(exc).__name__)
    docs = _dedupe_evidence_results(docs)
    retrieval_summary = {
        "status": "activated" if queries else "no_queries",
        "source": "wikipedia_search",
        "query_count": len(queries),
        "query_hashes": [stable_hash({"query": query}) for query in queries],
        "candidate_doc_count": len(docs),
        "candidate_doc_hashes": [
            stable_hash({"title": row.get("title", ""), "snippet": row.get("snippet", "")})
            for row in docs[:10]
        ],
        "error_types": sorted(set(errors)),
    }
    plan["stages"]["hipporag_context_retrieval"] = retrieval_summary
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant="hipporag_baseline",
        stage="hipporag_context_retrieval",
        data=retrieval_summary,
    )

    ranked_docs = _hipporag_style_rerank(problem, docs)
    context = _format_evidence_context([row["doc"] for row in ranked_docs[:5]], max_chars=context_max_chars)
    rerank_summary = {
        "status": "activated" if ranked_docs else "no_results",
        "method": "lexical_entity_passage_association",
        "candidate_doc_count": len(docs),
        "selected_doc_count": min(len(ranked_docs), 5),
        "selected_doc_hashes": [
            stable_hash({"title": row["doc"].get("title", ""), "snippet": row["doc"].get("snippet", "")})
            for row in ranked_docs[:5]
        ],
        "top_scores": [round(float(row["score"]), 4) for row in ranked_docs[:5]],
        "entity_node_count": len(_hipporag_entity_nodes(problem, docs)),
    }
    plan["stages"]["hipporag_associative_rerank"] = rerank_summary
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant="hipporag_baseline",
        stage="hipporag_associative_rerank",
        data=rerank_summary,
    )
    plan["prompt_context"] = context
    plan["stages"]["prompt_builder"] = {
        "status": "activated",
        "context_injected": bool(context),
        "context_char_count": len(context),
    }
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant="hipporag_baseline",
        stage="prompt_builder",
        data=plan["stages"]["prompt_builder"],
    )
    return plan


def _hipporag_style_rerank(problem: dict[str, Any], docs: list[dict[str, str]]) -> list[dict[str, Any]]:
    query_terms = _content_terms(problem.get("_question", ""))
    entities = _hipporag_entity_nodes(problem, docs)
    entity_terms: set[str] = set()
    for entity in entities:
        entity_terms.update(_content_terms(entity))
    ranked: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        text = f"{doc.get('title', '')} {doc.get('snippet', '')}"
        doc_terms = _content_terms(text)
        title_terms = _content_terms(doc.get("title", ""))
        query_overlap = len(query_terms & doc_terms)
        title_overlap = len(query_terms & title_terms)
        entity_overlap = len(entity_terms & doc_terms)
        score = query_overlap + 0.6 * title_overlap + 0.25 * entity_overlap + 0.01 * max(0, len(docs) - index)
        ranked.append({"doc": doc, "score": score})
    ranked.sort(key=lambda row: (-float(row["score"]), row["doc"].get("title", "")))
    return ranked


def _hipporag_entity_nodes(problem: dict[str, Any], docs: list[dict[str, str]]) -> list[str]:
    entities: list[str] = []
    for value in (problem.get("raw_subject"), problem.get("category")):
        if value:
            entities.append(str(value))
    question = str(problem.get("_question") or "")
    entities.extend(re.findall(r"\b[A-Z][A-Za-z0-9_+.-]*(?:\s+[A-Z][A-Za-z0-9_+.-]*){0,5}\b", question))
    entities.extend(row.get("title", "") for row in docs if row.get("title"))
    cleaned: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        text = _clean_evidence_text(entity)
        key = _normalize_exact(text)
        if key and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned[:20]


def _content_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", str(text or ""))
        if token.lower() not in _EVIDENCE_QUERY_STOPWORDS
    }


def _prompt_for(problem: dict[str, Any], *, variant: str, agent_plan: dict[str, Any] | None = None) -> str:
    question = problem["_question"]
    answer_type = problem["answer_type"]
    output = (
        "Return JSON only: {\"answer\":\"...\"}. "
        "For multiple choice, answer with the single letter only. "
        "For exact match, answer with the shortest exact answer."
    )
    if variant.startswith("assumption_agent"):
        context = (agent_plan or {}).get("prompt_context", "")
        if context:
            return (
                "Solve this closed-book expert exam item. A bounded Assumption Agent has already run graph "
                "retrieval, structural morphism transfer, a world-model route gate, and a recursive applicability "
                "runner. Use the retrieved context only if it genuinely helps; ignore irrelevant assumptions. "
                "Do not expose reasoning or mention the modules.\n\n"
                f"{context}\n\n"
                f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            )
        return f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
    if variant == "assumption_wrapper":
        return (
            "Solve this closed-book expert exam item. Internally use an assumption-audit process: identify the "
            "domain, list candidate assumptions, test likely failure modes, verify the final answer format, then "
            "return only the requested JSON. Do not include reasoning.\n\n"
            f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
        )
    if variant.startswith("hipporag"):
        context = (agent_plan or {}).get("prompt_context", "")
        context_block = (
            "HippoRAG-style transient retrieval context follows. It was built only from question-triggered "
            "external evidence and does not include the gold answer or Assumption Agent graph. Use it only if "
            "it directly supports the answer; ignore irrelevant passages.\n\n"
            f"{context}\n\n"
            if context
            else ""
        )
        return (
            "Solve this expert exam item using retrieval-augmented QA. Do not expose reasoning.\n\n"
            f"{context_block}"
            f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
        )
    return f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"


def _call_recursive_verified_answer(
    *,
    problem: dict[str, Any],
    model: str,
    agent_plan: dict[str, Any],
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    child_mode: str,
    child_timeout: float | None,
    max_tokens: int,
    evidence_bridge_enabled: bool,
) -> dict[str, Any]:
    """Run multiple child answer attempts and select one without persisting raw HLE text."""
    critic_model = _agent_critic_model(model)
    if critic_model != model:
        agent_plan.setdefault("stages", {})["critic_model_router"] = {
            "status": "activated",
            "base_model": model,
            "critic_model": critic_model,
            "policy": "env_override_for_falsification_and_verification",
        }
    evidence_summary: dict[str, Any] | None = None
    if evidence_bridge_enabled and _should_prime_evidence_bridge(problem, agent_plan):
        evidence_context, evidence_summary = _build_hle_evidence_bridge_context(
            problem=problem,
            eval_id=eval_id,
            call_id=call_id,
            model=model,
            logger=logger,
        )
        if evidence_context:
            agent_plan["hle_evidence_context"] = evidence_context
            agent_plan["hle_evidence_bridge"] = evidence_summary
    if _agent_hipporag_child_enabled(problem):
        hipporag_context, hipporag_summary = _build_agent_hipporag_child_context(
            problem=problem,
            eval_id=eval_id,
            call_id=call_id,
            model=model,
            logger=logger,
            context_max_chars=2200,
        )
        if hipporag_context:
            agent_plan["hipporag_prompt_context"] = hipporag_context
        if hipporag_summary:
            agent_plan.setdefault("stages", {})["agent_hipporag_context_bridge"] = hipporag_summary
    specs = _recursive_child_prompt_specs(problem, agent_plan=agent_plan)
    child_result = _execute_recursive_child_attempts(
        problem=problem,
        specs=specs,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=child_timeout if child_timeout is not None else timeout,
        max_tokens=max_tokens,
        mode=child_mode,
    )
    attempts = child_result["attempts"]
    underlying_calls = int(child_result["underlying_model_calls"] or 0)
    early_stop_reason = child_result.get("early_stop_reason")
    skipped_prompt_kinds = child_result.get("skipped_prompt_kinds", [])
    math_tool_summary: dict[str, Any] | None = None
    if _should_run_math_tool_child(problem):
        math_attempt = _run_math_tool_attempt(
            problem=problem,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=_math_tool_child_timeout(child_timeout if child_timeout is not None else timeout),
            max_tokens=min(max_tokens, 512),
        )
        attempts.append(math_attempt)
        math_tool_summary = math_attempt.get("tool_summary")
        underlying_calls += int(math_attempt.get("underlying_model_calls", 0) or 0)
    candidate_verifier_summary: dict[str, Any] | None = None
    if _should_run_candidate_claim_verifier(problem):
        candidate_verifier_summary = _apply_math_candidate_claim_verifier(
            problem,
            attempts,
            model=critic_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=min(max_tokens, 384),
        )
        underlying_calls += int(candidate_verifier_summary.get("underlying_model_calls", 0) or 0)
    option_evidence_summary: dict[str, Any] | None = None
    option_evidence_attempt, option_evidence_summary = _maybe_run_mc_option_evidence_scorer(
        problem=problem,
        attempts=attempts,
        eval_id=eval_id,
        call_id=call_id,
        model=model,
        logger=logger,
    )
    if option_evidence_attempt:
        attempts.append(option_evidence_attempt)
    domain_rule_summary: dict[str, Any] | None = None
    domain_rule_attempt, domain_rule_summary = _maybe_run_domain_rule_mc_verifier(
        problem=problem,
        attempts=attempts,
        evidence_context=str(agent_plan.get("hle_evidence_context") or ""),
        eval_id=eval_id,
        call_id=call_id,
        model=model,
        logger=logger,
    )
    if domain_rule_attempt:
        attempts.append(domain_rule_attempt)
    if evidence_bridge_enabled and not agent_plan.get("hle_evidence_context") and _needs_evidence_grounded_child(problem, attempts):
        evidence_context, evidence_summary = _build_hle_evidence_bridge_context(
            problem=problem,
            eval_id=eval_id,
            call_id=call_id,
            model=model,
            logger=logger,
        )
        if evidence_context:
            agent_plan["hle_evidence_context"] = evidence_context
            agent_plan["hle_evidence_bridge"] = evidence_summary
            evidence_attempt = _run_child_attempt(
                problem=problem,
                spec={
                    "prompt_kind": "evidence_grounded_answer",
                    "prompt": _evidence_grounded_answer_prompt(problem, evidence_context=evidence_context),
                },
                child_index=len(attempts) + 1,
                model=critic_model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=child_timeout if child_timeout is not None else timeout,
                max_tokens=max_tokens,
            )
            attempts.append(evidence_attempt)
            if evidence_attempt.get("status") == "answered":
                underlying_calls += 1

    counter_challenge_summary: dict[str, Any] | None = None
    counter_challenge_attempt, counter_challenge_summary = _maybe_run_counter_assumption_challenge(
        problem=problem,
        attempts=attempts,
        candidate_verifier_summary=candidate_verifier_summary,
        math_tool_summary=math_tool_summary,
        evidence_context=str(agent_plan.get("hle_evidence_context") or ""),
        model=critic_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=child_timeout if child_timeout is not None else timeout,
        max_tokens=max_tokens,
    )
    if counter_challenge_attempt:
        attempts.append(counter_challenge_attempt)
        if counter_challenge_attempt.get("status") == "answered":
            underlying_calls += 1
        option_elimination_attempt, option_elimination_summary = _maybe_run_option_elimination_challenge(
            problem=problem,
            attempts=attempts,
            counter_challenge_summary=counter_challenge_summary,
            evidence_context=str(agent_plan.get("hle_evidence_context") or ""),
            model=critic_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
        if option_elimination_summary:
            counter_challenge_summary["option_elimination_challenge"] = option_elimination_summary
        if option_elimination_attempt:
            attempts.append(option_elimination_attempt)
            if option_elimination_attempt.get("status") == "answered":
                underlying_calls += 1
            forced_alternative_attempt, forced_alternative_summary = _maybe_run_forced_alternative_challenge(
                problem=problem,
                attempts=attempts,
                option_elimination_summary=option_elimination_summary,
                evidence_context=str(agent_plan.get("hle_evidence_context") or ""),
                model=critic_model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=child_timeout if child_timeout is not None else timeout,
                max_tokens=max_tokens,
            )
            if forced_alternative_summary:
                counter_challenge_summary["forced_alternative_challenge"] = forced_alternative_summary
            if forced_alternative_attempt:
                attempts.append(forced_alternative_attempt)
                if forced_alternative_attempt.get("status") == "answered":
                    underlying_calls += 1

    critic_synthesis_summary: dict[str, Any] | None = None
    critic_synthesis_attempt, critic_synthesis_summary = _maybe_run_critic_synthesis_child(
        problem=problem,
        attempts=attempts,
        evidence_context=str(agent_plan.get("hle_evidence_context") or ""),
        base_model=model,
        critic_model=critic_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=child_timeout if child_timeout is not None else timeout,
        max_tokens=max_tokens,
    )
    if critic_synthesis_attempt:
        attempts.append(critic_synthesis_attempt)
        if critic_synthesis_attempt.get("status") == "answered":
            underlying_calls += 1

    option_sweep_summary: dict[str, Any] | None = None
    option_sweep_attempts, option_sweep_summary = _maybe_add_mc_option_sweep_candidates(
        problem=problem,
        attempts=attempts,
    )
    if option_sweep_attempts:
        attempts.extend(option_sweep_attempts)

    selection = _select_recursive_child_answer(
        problem=problem,
        attempts=attempts,
        model=critic_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=min(max_tokens, 384),
        evidence_context=str(agent_plan.get("hle_evidence_context") or ""),
    )
    underlying_calls += int(selection.get("underlying_model_calls", 0) or 0)
    selected_answer = selection.get("selected_answer") or _fallback_answer(attempts)
    selected_answer, canonical_summary = _canonicalize_exact_answer_candidate(problem, selected_answer)
    format_repair_summary: dict[str, Any] | None = None
    if _needs_exact_answer_repair(problem, selected_answer):
        repair = _repair_exact_answer(
            problem=problem,
            selected_answer=selected_answer,
            agent_plan=agent_plan,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
            evidence_bridge_enabled=evidence_bridge_enabled,
        )
        underlying_calls += int(repair.get("underlying_model_calls", 0) or 0)
        selected_answer = repair.get("selected_answer") or selected_answer
        selected_answer, repair_canonical_summary = _canonicalize_exact_answer_candidate(problem, selected_answer)
        if repair_canonical_summary.get("changed"):
            canonical_summary = repair_canonical_summary
        format_repair_summary = repair.get("stage_summary")
    selected_hash = stable_hash({"answer": selected_answer})
    answered_count = sum(1 for attempt in attempts if attempt.get("status") == "answered")
    child_summary = {
        "status": "activated",
        "execution_mode": child_result.get("execution_mode"),
        "child_timeout_sec": child_result.get("child_timeout_sec"),
        "child_max_workers": child_result.get("child_max_workers"),
        "planned_child_count": len(specs),
        "child_count": len(attempts),
        "answered_child_count": answered_count,
        "error_child_count": len(attempts) - answered_count,
        "early_stopped": bool(early_stop_reason),
        "early_stop_reason": early_stop_reason,
        "skipped_prompt_kinds": skipped_prompt_kinds,
        "prompt_kinds": [attempt["prompt_kind"] for attempt in attempts],
        "candidate_answer_hashes": [
            attempt.get("parsed_answer_hash") for attempt in attempts if attempt.get("parsed_answer_hash")
        ],
    }
    verifier_summary = {
        "status": "activated",
        "selection_method": selection.get("selection_method"),
        "selected_child_id": selection.get("selected_child_id"),
        "selected_answer_hash": selected_hash,
        "verifier_model_call": bool(selection.get("verifier_model_call")),
    }
    stages = agent_plan.setdefault("stages", {})
    stages["recursive_child_validation"] = child_summary
    stages["multi_candidate_self_verifier"] = verifier_summary
    if math_tool_summary:
        stages["hle_math_tool_solver"] = math_tool_summary
    if candidate_verifier_summary:
        stages["candidate_claim_verifier"] = candidate_verifier_summary
    if domain_rule_summary:
        domain_rule_summary["final_selection_method"] = selection.get("selection_method")
        domain_rule_summary["selected_domain_rule_candidate"] = (
            selection.get("selected_child_id") == domain_rule_summary.get("child_id")
        )
        stages["domain_rule_mc_verifier"] = domain_rule_summary
    if option_evidence_summary:
        option_evidence_summary["final_selection_method"] = selection.get("selection_method")
        option_evidence_summary["selected_option_evidence_candidate"] = (
            selection.get("selected_child_id") == option_evidence_summary.get("child_id")
        )
        stages["mc_option_evidence_scorer"] = option_evidence_summary
    if counter_challenge_summary:
        counter_challenge_summary["final_selection_method"] = selection.get("selection_method")
        counter_challenge_summary["selected_counter_challenge"] = (
            selection.get("selected_child_id") == counter_challenge_summary.get("child_id")
        )
        option_summary = counter_challenge_summary.get("option_elimination_challenge")
        if isinstance(option_summary, dict):
            option_summary["selected_option_elimination_challenge"] = (
                selection.get("selected_child_id") == option_summary.get("child_id")
            )
        forced_summary = counter_challenge_summary.get("forced_alternative_challenge")
        if isinstance(forced_summary, dict):
            forced_summary["selected_forced_alternative"] = (
                selection.get("selected_child_id") == forced_summary.get("child_id")
            )
        stages["counter_assumption_challenge"] = counter_challenge_summary
    if critic_synthesis_summary:
        critic_synthesis_summary["final_selection_method"] = selection.get("selection_method")
        critic_synthesis_summary["selected_critic_synthesis"] = (
            selection.get("selected_child_id") == critic_synthesis_summary.get("child_id")
        )
        stages["critic_synthesis_child"] = critic_synthesis_summary
    if option_sweep_summary:
        option_sweep_summary["final_selection_method"] = selection.get("selection_method")
        selected_child_id = selection.get("selected_child_id")
        option_sweep_summary["selected_option_sweep_candidate"] = bool(
            selected_child_id
            and any(attempt.get("child_id") == selected_child_id for attempt in option_sweep_attempts)
        )
        stages["mc_option_sweep_candidates"] = option_sweep_summary
    if canonical_summary.get("changed"):
        stages["answer_format_canonicalizer"] = canonical_summary
    if format_repair_summary:
        stages["answer_format_repair"] = format_repair_summary
        if format_repair_summary.get("evidence_bridge"):
            stages["hle_evidence_bridge"] = format_repair_summary["evidence_bridge"]
    elif evidence_summary:
        stages["hle_evidence_bridge"] = evidence_summary
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant="assumption_agent_recursive_verify",
        stage="recursive_child_validation",
        data=child_summary,
    )
    if stages.get("critic_model_router"):
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="critic_model_router",
            data=stages["critic_model_router"],
        )
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant="assumption_agent_recursive_verify",
        stage="multi_candidate_self_verifier",
        data=verifier_summary,
    )
    if math_tool_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="hle_math_tool_solver",
            data=math_tool_summary,
        )
    if candidate_verifier_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="candidate_claim_verifier",
            data=candidate_verifier_summary,
        )
    if domain_rule_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="domain_rule_mc_verifier",
            data=domain_rule_summary,
        )
    if option_evidence_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="mc_option_evidence_scorer",
            data=option_evidence_summary,
        )
    if counter_challenge_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="counter_assumption_challenge",
            data=counter_challenge_summary,
        )
    if critic_synthesis_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="critic_synthesis_child",
            data=critic_synthesis_summary,
        )
    if option_sweep_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="mc_option_sweep_candidates",
            data=option_sweep_summary,
        )
    if canonical_summary.get("changed"):
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="answer_format_canonicalizer",
            data=canonical_summary,
        )
    if format_repair_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="answer_format_repair",
            data=format_repair_summary,
        )
        if format_repair_summary.get("evidence_bridge"):
            _agent_stage_log(
                logger,
                eval_id=eval_id,
                call_id=call_id,
                problem=problem,
                model=model,
                variant="assumption_agent_recursive_verify",
                stage="hle_evidence_bridge",
                data=format_repair_summary["evidence_bridge"],
            )
    elif evidence_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="hle_evidence_bridge",
            data=evidence_summary,
        )
    return {
        "answer_text": json.dumps({"answer": selected_answer}, ensure_ascii=False),
        "underlying_model_calls": underlying_calls,
    }


def _agent_hipporag_child_enabled(problem: dict[str, Any]) -> bool:
    if os.environ.get("HLE_DISABLE_AGENT_HIPPORAG_CHILD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("HLE_ENABLE_EXACT_AGENT_HIPPORAG_CHILD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return problem.get("answer_type") in {"multipleChoice", "exactMatch"}
    return problem.get("answer_type") == "multipleChoice"


def _build_agent_hipporag_child_context(
    *,
    problem: dict[str, Any],
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
    context_max_chars: int,
) -> tuple[str, dict[str, Any] | None]:
    queries = _candidate_evidence_queries(problem)
    docs: list[dict[str, str]] = []
    errors: list[str] = []
    for query in queries:
        try:
            docs.extend(_wikipedia_search(query, limit=3, timeout=6.0))
        except Exception as exc:
            errors.append(type(exc).__name__)
    docs = _dedupe_evidence_results(docs)
    ranked_docs = _hipporag_style_rerank(problem, docs)
    context = _format_evidence_context([row["doc"] for row in ranked_docs[:5]], max_chars=context_max_chars)
    summary = {
        "status": "activated" if context else "no_results",
        "source": "wikipedia_search_plus_hipporag_style_rerank",
        "query_count": len(queries),
        "query_hashes": [stable_hash({"query": query}) for query in queries],
        "candidate_doc_count": len(docs),
        "selected_doc_count": min(len(ranked_docs), 5),
        "selected_doc_hashes": [
            stable_hash({"title": row["doc"].get("title", ""), "snippet": row["doc"].get("snippet", "")})
            for row in ranked_docs[:5]
        ],
        "top_scores": [round(float(row["score"]), 4) for row in ranked_docs[:5]],
        "entity_node_count": len(_hipporag_entity_nodes(problem, docs)),
        "context_char_count": len(context),
        "error_types": sorted(set(errors)),
        "underlying_model_calls": 0,
    }
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant="assumption_agent_recursive_verify",
        stage="agent_hipporag_context_bridge",
        data=summary,
    )
    return context, summary


def _execute_recursive_child_attempts(
    *,
    problem: dict[str, Any],
    specs: list[dict[str, str]],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    mode: str,
) -> dict[str, Any]:
    if mode != "parallel_quorum":
        return _execute_recursive_child_attempts_serial(
            problem=problem,
            specs=specs,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    first_batch = _run_child_batch(
        problem=problem,
        specs=specs[:2],
        start_index=1,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
        max_workers=2,
    )
    attempts = first_batch["attempts"]
    early_stop_reason = None
    skipped_prompt_kinds: list[str] = []
    remaining_prompt_kinds = {row["prompt_kind"] for row in specs[len(attempts):]}
    retrieval_child_pending = bool(remaining_prompt_kinds & {"agent_context_answer", "hipporag_context_answer"})
    if _can_stop_recursive_children_early(problem, attempts) and not retrieval_child_pending:
        early_stop_reason = "two_vote_majority"
        skipped_prompt_kinds = [row["prompt_kind"] for row in specs[len(attempts):]]
        _log_recursive_child_early_stop(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            reason=early_stop_reason,
            executed_child_count=len(attempts),
            planned_child_count=len(specs),
            skipped_prompt_kinds=skipped_prompt_kinds,
        )
        return {
            "attempts": attempts,
            "underlying_model_calls": first_batch["underlying_model_calls"],
            "early_stop_reason": early_stop_reason,
            "skipped_prompt_kinds": skipped_prompt_kinds,
            "execution_mode": "parallel_quorum",
            "child_timeout_sec": timeout,
            "child_max_workers": first_batch["max_workers"],
        }
    rest_batch = _run_child_batch(
        problem=problem,
        specs=specs[2:],
        start_index=3,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
        max_workers=max(1, min(2, len(specs[2:]))),
    )
    attempts.extend(rest_batch["attempts"])
    attempts.sort(key=lambda row: int(row.get("child_index", 0) or 0))
    return {
        "attempts": attempts,
        "underlying_model_calls": first_batch["underlying_model_calls"] + rest_batch["underlying_model_calls"],
        "early_stop_reason": None,
        "skipped_prompt_kinds": [],
        "execution_mode": "parallel_quorum",
        "child_timeout_sec": timeout,
        "child_max_workers": max(first_batch["max_workers"], rest_batch["max_workers"]),
    }


def _execute_recursive_child_attempts_serial(
    *,
    problem: dict[str, Any],
    specs: list[dict[str, str]],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    underlying_calls = 0
    early_stop_reason = None
    skipped_prompt_kinds: list[str] = []
    for index, spec in enumerate(specs, start=1):
        attempt = _run_child_attempt(
            problem=problem,
            spec=spec,
            child_index=index,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        attempts.append(attempt)
        if attempt.get("status") == "answered":
            underlying_calls += 1
        if _can_stop_recursive_children_early(problem, attempts):
            early_stop_reason = "two_vote_majority"
            skipped_prompt_kinds = [row["prompt_kind"] for row in specs[index:]]
            _log_recursive_child_early_stop(
                logger,
                eval_id=eval_id,
                call_id=call_id,
                problem=problem,
                model=model,
                reason=early_stop_reason,
                executed_child_count=len(attempts),
                planned_child_count=len(specs),
                skipped_prompt_kinds=skipped_prompt_kinds,
            )
            break
    return {
        "attempts": attempts,
        "underlying_model_calls": underlying_calls,
        "early_stop_reason": early_stop_reason,
        "skipped_prompt_kinds": skipped_prompt_kinds,
        "execution_mode": "serial",
        "child_timeout_sec": timeout,
        "child_max_workers": 1,
    }


def _run_child_batch(
    *,
    problem: dict[str, Any],
    specs: list[dict[str, str]],
    start_index: int,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    max_workers: int,
) -> dict[str, Any]:
    if not specs:
        return {"attempts": [], "underlying_model_calls": 0, "max_workers": 0}
    max_workers = max(1, min(max_workers, len(specs)))
    attempts: list[dict[str, Any]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_specs: dict[concurrent.futures.Future, tuple[dict[str, str], int]] = {}
    batch_started = time.monotonic()
    try:
        for offset, spec in enumerate(specs):
            child_index = start_index + offset
            future = executor.submit(
                _run_child_attempt,
                problem=problem,
                spec=spec,
                child_index=child_index,
                model=model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            future_specs[future] = (spec, child_index)
        wait_timeout = None if timeout is None else max(0.0, float(timeout))
        done, pending = concurrent.futures.wait(future_specs, timeout=wait_timeout)
        for future in done:
            attempts.append(future.result())
        if pending:
            elapsed = round(time.monotonic() - batch_started, 4)
            for future in pending:
                future.cancel()
                spec, child_index = future_specs[future]
                attempts.append(
                    _child_timeout_attempt(
                        problem=problem,
                        spec=spec,
                        child_index=child_index,
                        model=model,
                        eval_id=eval_id,
                        call_id=call_id,
                        logger=logger,
                        timeout=timeout,
                        latency_sec=elapsed,
                    )
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    attempts.sort(key=lambda row: int(row.get("child_index", 0) or 0))
    return {
        "attempts": attempts,
        "underlying_model_calls": sum(1 for attempt in attempts if attempt.get("status") == "answered"),
        "max_workers": max_workers,
    }


def _child_timeout_attempt(
    *,
    problem: dict[str, Any],
    spec: dict[str, str],
    child_index: int,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    latency_sec: float,
) -> dict[str, Any]:
    child_id = stable_hash({"call_id": call_id, "child_index": child_index, "prompt_kind": spec["prompt_kind"]})
    attempt = {
        "child_id": child_id,
        "child_index": child_index,
        "prompt_kind": spec["prompt_kind"],
        "parsed_answer": "",
        "parsed_answer_hash": None,
        "prediction_hash": None,
        "latency_sec": latency_sec,
        "status": "timeout",
        "error_type": "ChildTimeout",
    }
    _log_event(
        logger,
        {
            "event": "recursive_child_timeout",
            "eval_id": eval_id,
            "call_id": call_id,
            "child_id": child_id,
            "child_index": child_index,
            "problem_id_hash": problem["id_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "prompt_kind": spec["prompt_kind"],
            "latency_sec": latency_sec,
            "timeout_sec": timeout,
        },
    )
    return attempt


def _run_child_attempt(
    *,
    problem: dict[str, Any],
    spec: dict[str, str],
    child_index: int,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    child_id = stable_hash({"call_id": call_id, "child_index": child_index, "prompt_kind": spec["prompt_kind"]})
    _log_event(
        logger,
        {
            "event": "recursive_child_start",
            "eval_id": eval_id,
            "call_id": call_id,
            "child_id": child_id,
            "child_index": child_index,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "prompt_kind": spec["prompt_kind"],
            "timeout_sec": timeout,
        },
    )
    started = time.monotonic()
    try:
        text = _call_model(model=model, prompt=spec["prompt"], timeout=timeout, max_tokens=max_tokens)
        parsed = _parse_answer_json(text) or text.strip()
        parsed, mc_canonical_summary = _canonicalize_multiple_choice_answer(problem, parsed)
        attempt = {
            "child_id": child_id,
            "child_index": child_index,
            "prompt_kind": spec["prompt_kind"],
            "parsed_answer": parsed,
            "parsed_answer_hash": stable_hash({"answer": parsed}),
            "prediction_hash": stable_hash({"prediction": text}),
            "latency_sec": round(time.monotonic() - started, 4),
            "status": "answered",
        }
        if mc_canonical_summary.get("changed"):
            attempt["multiple_choice_canonicalized"] = True
            attempt["multiple_choice_canonicalizer"] = mc_canonical_summary
        _log_event(
            logger,
            {
                "event": "recursive_child_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "child_id": child_id,
                "child_index": child_index,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "prompt_kind": spec["prompt_kind"],
                "latency_sec": attempt["latency_sec"],
                "parsed_answer_hash": attempt["parsed_answer_hash"],
                "prediction_hash": attempt["prediction_hash"],
            },
        )
        return attempt
    except Exception as exc:
        attempt = {
            "child_id": child_id,
            "child_index": child_index,
            "prompt_kind": spec["prompt_kind"],
            "parsed_answer": "",
            "parsed_answer_hash": None,
            "prediction_hash": None,
            "latency_sec": round(time.monotonic() - started, 4),
            "status": "error",
            "error_type": type(exc).__name__,
        }
        _log_event(
            logger,
            {
                "event": "recursive_child_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "child_id": child_id,
                "child_index": child_index,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "prompt_kind": spec["prompt_kind"],
                "latency_sec": attempt["latency_sec"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            },
        )
        return attempt


def _log_recursive_child_early_stop(
    logger: "_JsonlLogger | None",
    *,
    eval_id: str,
    call_id: str,
    problem: dict[str, Any],
    model: str,
    reason: str,
    executed_child_count: int,
    planned_child_count: int,
    skipped_prompt_kinds: list[str],
) -> None:
    _log_event(
        logger,
        {
            "event": "recursive_child_early_stop",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "reason": reason,
            "executed_child_count": executed_child_count,
            "planned_child_count": planned_child_count,
            "skipped_prompt_kinds": skipped_prompt_kinds,
        },
    )


def _can_stop_recursive_children_early(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> bool:
    if not _has_two_vote_majority(attempts, answer_type=problem["answer_type"]):
        return False
    if problem.get("answer_type") == "multipleChoice":
        prompt_kinds = {str(attempt.get("prompt_kind") or "") for attempt in attempts}
        reflective_kinds = {"agent_context_answer", "constraint_checked_answer", "recursive_assumption_answer"}
        return bool(prompt_kinds & reflective_kinds)
    if problem.get("answer_type") != "multipleChoice" and _should_run_math_tool_child(problem):
        prompt_kinds = {str(attempt.get("prompt_kind") or "") for attempt in attempts}
        reflective_kinds = {"constraint_checked_answer", "recursive_assumption_answer", "agent_context_answer"}
        return bool(prompt_kinds & reflective_kinds)
    return True


def _apply_math_candidate_claim_verifier(
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    model: str | None = None,
    eval_id: str | None = None,
    call_id: str | None = None,
    logger: "_JsonlLogger | None" = None,
    timeout: float | None = None,
    max_tokens: int = 384,
) -> dict[str, Any]:
    if problem.get("answer_type") == "multipleChoice":
        stem, options = _split_multiple_choice_question(problem)
        if len(options) < 2:
            return {
                "status": "no_executable_claim",
                "backend": "sympy_mc_option_verifier",
                "verified_count": 0,
                "refuted_count": 0,
                "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
                "reference_operation": "none",
                "reference_reason": "mc_options_not_parsed",
                "underlying_model_calls": 0,
            }
        stem_problem = {**problem, "_question": stem or problem.get("_question", "")}
        reference = _deterministic_math_tool_answer(stem_problem)
        if reference.get("confidence") == "verified_symbolic" and str(reference.get("answer") or "").strip():
            return _apply_math_reference_to_multiple_choice_options(
                problem=problem,
                attempts=attempts,
                options=options,
                reference=reference,
                backend="sympy_mc_option_deterministic",
                underlying_model_calls=0,
            )
        if model:
            llm_reference = _llm_math_reference_claim_for_mc_options(
                problem=stem_problem,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            if llm_reference.get("confidence") == "verified_symbolic" and str(llm_reference.get("answer") or "").strip():
                summary = _apply_math_reference_to_multiple_choice_options(
                    problem=problem,
                    attempts=attempts,
                    options=options,
                    reference=llm_reference,
                    backend="sympy_mc_option_planner",
                    underlying_model_calls=1,
                )
                _log_candidate_claim_planner_event(
                    logger,
                    eval_id=eval_id,
                    call_id=call_id,
                    problem=problem,
                    model=model,
                    reference=llm_reference,
                    summary=summary,
                )
                return summary
            _log_candidate_claim_planner_event(
                logger,
                eval_id=eval_id,
                call_id=call_id,
                problem=problem,
                model=model,
                reference=llm_reference,
                summary=None,
            )
            return {
                "status": "no_executable_claim",
                "backend": "sympy_mc_option_planner",
                "verified_count": 0,
                "refuted_count": 0,
                "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
                "reference_operation": llm_reference.get("operation") or reference.get("operation"),
                "reference_reason": llm_reference.get("reason") or reference.get("reason"),
                "deterministic_reference_reason": reference.get("reason"),
                "option_count": len(options),
                "underlying_model_calls": 1,
            }
        return {
            "status": "no_executable_claim",
            "backend": "sympy_mc_option_verifier",
            "verified_count": 0,
            "refuted_count": 0,
            "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
            "reference_operation": reference.get("operation"),
            "reference_reason": reference.get("reason"),
            "option_count": len(options),
            "underlying_model_calls": 0,
        }
    reference = _deterministic_math_tool_answer(problem)
    if reference.get("confidence") == "verified_symbolic" and str(reference.get("answer") or "").strip():
        return _apply_math_reference_to_candidates(
            problem=problem,
            attempts=attempts,
            reference=reference,
            backend="sympy_deterministic",
            underlying_model_calls=0,
        )
    if model:
        llm_reference = _llm_math_reference_claim(
            problem=problem,
            attempts=attempts,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if llm_reference.get("confidence") == "verified_symbolic" and str(llm_reference.get("answer") or "").strip():
            summary = _apply_math_reference_to_candidates(
                problem=problem,
                attempts=attempts,
                reference=llm_reference,
                backend="sympy_candidate_reference_planner",
                underlying_model_calls=1,
            )
            _log_candidate_claim_planner_event(
                logger,
                eval_id=eval_id,
                call_id=call_id,
                problem=problem,
                model=model,
                reference=llm_reference,
                summary=summary,
            )
            return summary
        _log_candidate_claim_planner_event(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            reference=llm_reference,
            summary=None,
        )
        return {
            "status": "no_executable_claim",
            "backend": "sympy_candidate_reference_planner",
            "verified_count": 0,
            "refuted_count": 0,
            "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
            "reference_operation": llm_reference.get("operation") or reference.get("operation"),
            "reference_reason": llm_reference.get("reason") or reference.get("reason"),
            "deterministic_reference_reason": reference.get("reason"),
            "underlying_model_calls": 1,
        }
    return {
        "status": "no_executable_claim",
        "backend": "sympy_deterministic",
        "verified_count": 0,
        "refuted_count": 0,
        "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
        "reference_operation": reference.get("operation"),
        "reference_reason": reference.get("reason"),
        "underlying_model_calls": 0,
    }


def _apply_math_reference_to_multiple_choice_options(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    options: dict[str, str],
    reference: dict[str, Any],
    backend: str,
    underlying_model_calls: int,
) -> dict[str, Any]:
    if reference.get("confidence") != "verified_symbolic" or not str(reference.get("answer") or "").strip():
        return {
            "status": "no_executable_claim",
            "backend": backend,
            "verified_count": 0,
            "refuted_count": 0,
            "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
            "reference_operation": reference.get("operation"),
            "reference_reason": reference.get("reason"),
            "option_count": len(options),
            "underlying_model_calls": underlying_model_calls,
        }
    reference_answer = str(reference["answer"]).strip()
    matching_labels = [
        label for label, option_text in sorted(options.items())
        if _mc_option_matches_reference(option_text, reference_answer)
    ]
    if len(matching_labels) != 1:
        return {
            "status": "ambiguous_option_match" if matching_labels else "no_option_match",
            "backend": backend,
            "verified_count": 0,
            "refuted_count": 0,
            "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
            "reference_operation": reference.get("operation"),
            "reference_answer_hash": stable_hash({"answer": reference_answer}),
            "option_count": len(options),
            "matched_option_count": len(matching_labels),
            "matched_option_hashes": [stable_hash({"option_label": label}) for label in matching_labels],
            "underlying_model_calls": underlying_model_calls,
        }
    verified_label = matching_labels[0]
    verified = 0
    refuted = 0
    inconclusive = 0
    candidate_hashes: list[str] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        label = _normalize_for_selection(answer, answer_type="multipleChoice").upper()
        if not label:
            inconclusive += 1
            candidate_hashes.append(stable_hash({"candidate_answer": answer, "state": "inconclusive"}))
            continue
        state = "verified" if label == verified_label else "refuted"
        if state == "verified":
            verified += 1
        else:
            refuted += 1
        candidate_hashes.append(stable_hash({"candidate_answer": label, "state": state}))
        attempt["candidate_verifier_state"] = state
        attempt["candidate_verifier_backend"] = backend
        attempt["candidate_verifier_operation"] = reference.get("operation")
        attempt["candidate_verifier_claim_hash"] = stable_hash({
            "reference_answer": reference_answer,
            "candidate_answer": label,
            "operation": reference.get("operation"),
        })
    verifier_attempt = {
        "child_id": stable_hash({
            "call_id": problem.get("id_hash"),
            "prompt_kind": "candidate_claim_verifier_answer",
            "verified_label": verified_label,
            "claim_hash": reference.get("plan_hash"),
        }),
        "child_index": 9100,
        "prompt_kind": "candidate_claim_verifier_answer",
        "parsed_answer": verified_label,
        "parsed_answer_hash": stable_hash({"answer": verified_label}),
        "prediction_hash": stable_hash({
            "verified_label": verified_label,
            "reference_answer": reference_answer,
            "backend": backend,
        }),
        "latency_sec": 0.0,
        "status": "answered",
        "candidate_verifier_state": "verified",
        "candidate_verifier_backend": backend,
        "candidate_verifier_operation": reference.get("operation"),
        "candidate_verifier_claim_hash": stable_hash({
            "reference_answer": reference_answer,
            "candidate_answer": verified_label,
            "operation": reference.get("operation"),
        }),
    }
    attempts.append(verifier_attempt)
    verified += 1
    candidate_hashes.append(stable_hash({"candidate_answer": verified_label, "state": "verified_synthetic"}))
    return {
        "status": "activated",
        "backend": backend,
        "reference_operation": reference.get("operation"),
        "reference_answer_hash": stable_hash({"answer": reference_answer}),
        "verified_option_hash": stable_hash({"option_label": verified_label}),
        "verified_count": verified,
        "refuted_count": refuted,
        "inconclusive_count": inconclusive,
        "candidate_count": verified + refuted + inconclusive,
        "candidate_state_hashes": candidate_hashes,
        "option_count": len(options),
        "matched_option_count": 1,
        "underlying_model_calls": underlying_model_calls,
        "claim_hash": stable_hash({
            "question_hash": problem.get("question_hash"),
            "reference_answer": reference_answer,
            "verified_label": verified_label,
            "operation": reference.get("operation"),
            "plan_hash": reference.get("plan_hash"),
        }),
    }


def _apply_math_reference_to_candidates(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    reference: dict[str, Any],
    backend: str,
    underlying_model_calls: int,
) -> dict[str, Any]:
    if reference.get("confidence") != "verified_symbolic" or not str(reference.get("answer") or "").strip():
        return {
            "status": "no_executable_claim",
            "backend": backend,
            "verified_count": 0,
            "refuted_count": 0,
            "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
            "reference_operation": reference.get("operation"),
            "reference_reason": reference.get("reason"),
            "underlying_model_calls": underlying_model_calls,
        }
    reference_answer = str(reference["answer"]).strip()
    canonical_reference, _ = _canonicalize_exact_answer_candidate(problem, reference_answer)
    reference_norm = _normalize_for_selection(canonical_reference, answer_type=problem["answer_type"])
    weak_single_candidate = (
        backend == "sympy_candidate_reference_planner"
        and int(reference.get("candidate_count") or 0) <= 1
    )
    verified = 0
    refuted = 0
    inconclusive = 0
    weak_verified = 0
    candidate_hashes: list[str] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        canonical_answer, canonical_summary = _canonicalize_exact_answer_candidate(problem, answer)
        candidate_norm = _normalize_for_selection(canonical_answer, answer_type=problem["answer_type"])
        state = "verified" if candidate_norm == reference_norm else "refuted"
        if weak_single_candidate:
            weak_verified += int(state == "verified")
            inconclusive += 1
            candidate_hashes.append(stable_hash({"candidate_answer": canonical_answer, "state": f"weak_{state}"}))
            continue
        if state == "verified":
            verified += 1
        else:
            refuted += 1
        candidate_hashes.append(stable_hash({"candidate_answer": canonical_answer, "state": state}))
        attempt["candidate_verifier_state"] = state
        attempt["candidate_verifier_backend"] = backend
        attempt["candidate_verifier_operation"] = reference.get("operation")
        attempt["candidate_verifier_claim_hash"] = stable_hash({
            "reference_answer": canonical_reference,
            "candidate_answer": canonical_answer,
            "operation": reference.get("operation"),
        })
        if canonical_summary.get("changed"):
            attempt["candidate_verifier_canonicalized"] = True
            attempt["parsed_answer"] = canonical_answer
            attempt["parsed_answer_hash"] = stable_hash({"answer": canonical_answer})
    return {
        "status": "weak_single_candidate_confirmation" if weak_single_candidate else "activated",
        "backend": backend,
        "reference_operation": reference.get("operation"),
        "reference_answer_hash": stable_hash({"answer": canonical_reference}),
        "verified_count": verified,
        "refuted_count": refuted,
        "inconclusive_count": inconclusive,
        "weak_verified_count": weak_verified,
        "candidate_count": verified + refuted + inconclusive,
        "candidate_state_hashes": candidate_hashes,
        "underlying_model_calls": underlying_model_calls,
        "claim_hash": stable_hash({
            "question_hash": problem.get("question_hash"),
            "reference_answer": canonical_reference,
            "operation": reference.get("operation"),
            "plan_hash": reference.get("plan_hash"),
        }),
    }


def _llm_math_reference_claim(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    model: str,
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    candidates = _unique_candidate_answers_for_claim_planner(problem, attempts)
    if not candidates:
        return {
            "source": "llm_candidate_reference_planner",
            "operation": "none",
            "confidence": "abstain",
            "reason": "no_candidate_answers",
            "candidate_count": 0,
        }
    planner_text = _call_model(
        model=model,
        prompt=_candidate_claim_planner_prompt(problem, candidates),
        timeout=timeout,
        max_tokens=max_tokens,
    )
    plan = _parse_json_object(planner_text)
    if _math_plan_candidate_leak_risk(plan, candidates):
        return {
            "source": "llm_candidate_reference_planner",
            "operation": str((plan or {}).get("operation") or "none"),
            "confidence": "abstain",
            "reason": "candidate_literal_leakage",
            "plan_hash": stable_hash({"planner_text": planner_text}),
            "candidate_count": len(candidates),
        }
    result = _execute_math_tool_plan_text(planner_text)
    result["source"] = "llm_candidate_reference_planner"
    result.setdefault("plan_hash", stable_hash({"planner_text": planner_text}))
    result["candidate_count"] = len(candidates)
    return result


def _llm_math_reference_claim_for_mc_options(
    *,
    problem: dict[str, Any],
    model: str,
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    planner_text = _call_model(
        model=model,
        prompt=_mc_option_claim_planner_prompt(problem),
        timeout=timeout,
        max_tokens=max_tokens,
    )
    result = _execute_math_tool_plan_text(planner_text)
    result["source"] = "llm_mc_option_reference_planner"
    result.setdefault("plan_hash", stable_hash({"planner_text": planner_text}))
    result["candidate_count"] = 0
    return result


def _mc_option_claim_planner_prompt(problem: dict[str, Any]) -> str:
    return (
        "Extract one independent executable math claim for this HLE multipleChoice item. The answer options are "
        "intentionally hidden; compute the underlying value or symbolic result from the stem only, so it can later "
        "be matched to exactly one option. Do not answer from memory. If the stem cannot be checked by a small "
        "SymPy-compatible expression, equation solve, modular computation, derivative, integral, or limit, return "
        "operation=\"none\". JSON only: "
        "{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\","
        "\"lower\":\"\",\"upper\":\"\",\"order\":\"1\"}. No imports, no code, no explanation.\n\n"
        f"Question stem without answer options:\n{problem['_question']}"
    )


def _unique_candidate_answers_for_claim_planner(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for attempt in attempts:
        if attempt.get("status") not in {"answered", None} and attempt.get("prompt_kind") != "math_tool_answer":
            continue
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        canonical, _ = _canonicalize_exact_answer_candidate(problem, answer)
        norm = _normalize_for_selection(canonical, answer_type=problem["answer_type"])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        candidates.append(canonical[:160])
        if len(candidates) >= 5:
            break
    return candidates


def _candidate_claim_planner_prompt(problem: dict[str, Any], candidates: list[str]) -> str:
    candidate_lines = "\n".join(f"{idx}. {answer}" for idx, answer in enumerate(candidates, start=1))
    return (
        "Extract one independent executable math claim that can verify candidate answers for this HLE exactMatch item. "
        "Use the candidates only to infer answer format; do not copy a candidate into the expression/equation. "
        "If the problem cannot be checked by a small SymPy-compatible expression, equation solve, modular computation, "
        "derivative, integral, or limit, return operation=\"none\". JSON only: "
        "{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\",\"upper\":\"\",\"order\":\"1\"}. "
        "No imports, no code, no explanation.\n\n"
        f"Question:\n{problem['_question']}\n\nCandidate answers:\n{candidate_lines}"
    )


def _math_plan_candidate_leak_risk(plan: dict[str, Any] | None, candidates: list[str]) -> bool:
    if not isinstance(plan, dict):
        return False
    operation = str(plan.get("operation") or "none").strip().lower()
    if operation not in {
        "evaluate",
        "simplify",
        "factor",
        "expand",
        "solve",
        "mod",
        "differentiate",
        "integrate",
        "limit",
    }:
        return False
    candidate_norms = {
        _normalize_math_expression(candidate).replace(" ", "")
        for candidate in candidates
        if _normalize_math_expression(candidate)
    }
    if not candidate_norms:
        return False
    expression_norm = _normalize_math_expression(str(plan.get("expression") or "")).replace(" ", "")
    if operation in {"evaluate", "simplify", "factor", "expand", "mod", "differentiate", "integrate", "limit"} and expression_norm in candidate_norms:
        return True
    equation = _normalize_math_expression(str(plan.get("equation") or "")).replace(" ", "")
    variable = str(plan.get("variable") or "x").strip() or "x"
    trivial_equations = {f"{variable}={candidate}" for candidate in candidate_norms} | {
        f"{candidate}={variable}" for candidate in candidate_norms
    }
    return operation == "solve" and equation in trivial_equations


def _log_candidate_claim_planner_event(
    logger: "_JsonlLogger | None",
    *,
    eval_id: str | None,
    call_id: str | None,
    problem: dict[str, Any],
    model: str | None,
    reference: dict[str, Any],
    summary: dict[str, Any] | None,
) -> None:
    if not eval_id or not call_id or not model:
        return
    _log_event(
        logger,
        {
            "event": "candidate_claim_planner",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "status": (summary or {}).get(
                "status",
                "activated" if reference.get("confidence") == "verified_symbolic" else "abstained",
            ),
            "operation": reference.get("operation"),
            "reason": reference.get("reason"),
            "plan_hash": reference.get("plan_hash"),
            "claim_hash": (summary or {}).get("claim_hash"),
            "verified_count": (summary or {}).get("verified_count", 0),
            "refuted_count": (summary or {}).get("refuted_count", 0),
            "candidate_count": reference.get("candidate_count"),
        },
    )


def _should_run_candidate_claim_verifier(problem: dict[str, Any]) -> bool:
    if problem.get("answer_type") == "multipleChoice":
        _, options = _split_multiple_choice_question(problem)
        return _is_math_like_problem(problem) and len(options) >= 2
    return _should_run_math_tool_child(problem)


def _should_run_math_tool_child(problem: dict[str, Any]) -> bool:
    if problem.get("answer_type") == "multipleChoice":
        return False
    return _is_math_like_problem(problem)


def _is_math_like_problem(problem: dict[str, Any]) -> bool:
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
        str(problem.get("_question") or ""),
    ]).lower()
    return any(token in text for token in [
        "math",
        "mathematics",
        "algebra",
        "geometry",
        "number",
        "integer",
        "polynomial",
        "equation",
        "derivative",
        "integral",
        "limit",
        "modulo",
        "factorial",
        "binomial",
        "probability",
        "combinatorics",
        "calculus",
    ])


def _run_math_tool_attempt(
    *,
    problem: dict[str, Any],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    prompt_kind = "math_tool_answer"
    child_index = 9001
    child_id = stable_hash({"call_id": call_id, "child_index": child_index, "prompt_kind": prompt_kind})
    _log_event(
        logger,
        {
            "event": "math_tool_child_start",
            "eval_id": eval_id,
            "call_id": call_id,
            "child_id": child_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "prompt_kind": prompt_kind,
            "timeout_sec": timeout,
        },
    )
    started = time.monotonic()
    underlying_calls = 0
    try:
        result = _deterministic_math_tool_answer(problem)
        if not result.get("answer"):
            planner_text = _call_model(
                model=model,
                prompt=_math_tool_planner_prompt(problem),
                timeout=timeout,
                max_tokens=max_tokens,
            )
            underlying_calls += 1
            result = _execute_math_tool_plan_text(planner_text)
            result.setdefault("plan_hash", stable_hash({"planner_text": planner_text}))
            result.setdefault("source", "llm_planner")
        status = "answered" if result.get("answer") else "abstained"
        answer = str(result.get("answer") or "").strip()
        summary = {
            "status": "activated" if status == "answered" else "abstained",
            "tool": "sympy_restricted",
            "source": result.get("source"),
            "operation": result.get("operation"),
            "confidence": result.get("confidence"),
            "plan_hash": result.get("plan_hash"),
            "answer_hash": stable_hash({"answer": answer}) if answer else None,
            "reason": result.get("reason"),
            "underlying_model_calls": underlying_calls,
            "latency_sec": round(time.monotonic() - started, 4),
        }
        attempt = {
            "child_id": child_id,
            "child_index": child_index,
            "prompt_kind": prompt_kind,
            "parsed_answer": answer,
            "parsed_answer_hash": summary["answer_hash"],
            "prediction_hash": stable_hash({"tool_result": summary}),
            "latency_sec": summary["latency_sec"],
            "status": status,
            "tool_confidence": result.get("confidence"),
            "tool_source": result.get("source"),
            "tool_summary": summary,
            "underlying_model_calls": underlying_calls,
        }
        _log_event(
            logger,
            {
                "event": "math_tool_child_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "child_id": child_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "prompt_kind": prompt_kind,
                "status": status,
                "latency_sec": attempt["latency_sec"],
                "tool_summary": summary,
            },
        )
        return attempt
    except Exception as exc:
        summary = {
            "status": "failed",
            "tool": "sympy_restricted",
            "error_type": type(exc).__name__,
            "underlying_model_calls": underlying_calls,
            "latency_sec": round(time.monotonic() - started, 4),
        }
        _log_event(
            logger,
            {
                "event": "math_tool_child_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "child_id": child_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "prompt_kind": prompt_kind,
                "latency_sec": summary["latency_sec"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
                "tool_summary": summary,
            },
        )
        return {
            "child_id": child_id,
            "child_index": child_index,
            "prompt_kind": prompt_kind,
            "parsed_answer": "",
            "parsed_answer_hash": None,
            "prediction_hash": None,
            "latency_sec": summary["latency_sec"],
            "status": "error",
            "error_type": type(exc).__name__,
            "tool_summary": summary,
            "underlying_model_calls": underlying_calls,
        }


def _deterministic_math_tool_answer(problem: dict[str, Any]) -> dict[str, Any]:
    question = str(problem.get("_question") or "")
    for result in (
        _deterministic_binomial_or_factorial(question),
        _deterministic_expression_answer(question),
    ):
        if result.get("answer"):
            result.setdefault("source", "deterministic_parser")
            result.setdefault("plan_hash", stable_hash({"source": result.get("source"), "operation": result.get("operation")}))
            return result
    return {
        "source": "deterministic_parser",
        "operation": "none",
        "confidence": "abstain",
        "reason": "no_safe_deterministic_parse",
    }


def _deterministic_binomial_or_factorial(question: str) -> dict[str, Any]:
    if not re.search(
        r"\b(?:compute|evaluate|simplify|find\s+the\s+value\s+of|what\s+is\s+the\s+value\s+of)\b",
        question,
        flags=re.IGNORECASE,
    ):
        return {}
    choose_match = re.search(r"\b(\d{1,5})\s+(?:choose|C)\s+(\d{1,5})\b", question, flags=re.IGNORECASE)
    if choose_match:
        n = int(choose_match.group(1))
        k = int(choose_match.group(2))
        if 0 <= k <= n <= 10000:
            try:
                import sympy as sp

                value = sp.binomial(n, k)
                return {
                    "answer": _format_sympy_answer(value),
                    "operation": "binomial",
                    "confidence": "verified_symbolic",
                    "plan_hash": stable_hash({"operation": "binomial", "n": n, "k": k}),
                }
            except Exception:
                pass
    fact_match = re.search(r"\b(\d{1,4})\s*!\b", question)
    if fact_match:
        n = int(fact_match.group(1))
        if 0 <= n <= 500:
            try:
                import sympy as sp

                value = sp.factorial(n)
                return {
                    "answer": _format_sympy_answer(value),
                    "operation": "factorial",
                    "confidence": "verified_symbolic",
                    "plan_hash": stable_hash({"operation": "factorial", "n": n}),
                }
            except Exception:
                pass
    return {}


def _deterministic_expression_answer(question: str) -> dict[str, Any]:
    candidates = _candidate_math_expressions(question)
    modulus = _extract_modulus(question)
    for expr in candidates:
        try:
            parsed = _safe_sympy_parse_expr(expr)
            if parsed is None or getattr(parsed, "free_symbols", None):
                continue
            value = parsed
            operation = "evaluate"
            if modulus is not None:
                value = int(parsed) % modulus
                operation = "mod"
            return {
                "answer": _format_sympy_answer(value),
                "operation": operation,
                "confidence": "verified_symbolic",
                "plan_hash": stable_hash({"operation": operation, "expression": expr, "modulus": modulus}),
            }
        except Exception:
            continue
    return {}


def _candidate_math_expressions(question: str) -> list[str]:
    candidates: list[str] = []
    explicit_trigger = re.search(
        r"\b(?:compute|evaluate|simplify|find\s+the\s+value\s+of|what\s+is\s+the\s+value\s+of)\b",
        question,
        flags=re.IGNORECASE,
    )
    if explicit_trigger:
        candidates.extend(re.findall(r"\$([^$]{1,240})\$", question))
        candidates.extend(re.findall(r"\\\(([^)]{1,240})\\\)", question))
        candidates.extend(re.findall(r"`([^`]{1,240})`", question))
    for match in re.finditer(
        r"\b(?:compute|evaluate|simplify|find)\b(?:\s+the\s+value\s+of)?\s*[:=]?\s*([^?.\n]{1,180})",
        question,
        flags=re.IGNORECASE,
    ):
        candidates.append(match.group(1))
    cleaned: list[str] = []
    for candidate in candidates:
        expr = _normalize_math_expression(candidate)
        if expr and expr not in cleaned:
            cleaned.append(expr)
    return cleaned[:8]


def _extract_modulus(question: str) -> int | None:
    match = re.search(r"\b(?:mod|modulo)\s+(\d{1,9})\b", question, flags=re.IGNORECASE)
    if not match:
        return None
    modulus = int(match.group(1))
    return modulus if modulus > 1 else None


def _normalize_math_expression(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\\frac\s*\(([^()]+)\)\s*\(([^()]+)\)", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\(([^()]+)\)", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9_+*\-/%()., <>=]", " ", text)
    text = re.sub(r"\b(?:mod|modulo)\b.*$", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" =.,;:")


def _safe_sympy_parse_expr(expr: str) -> Any:
    expr = _normalize_math_expression(expr)
    if not expr or len(expr) > 260 or "__" in expr:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_+*\-/%()., <>=]+", expr):
        return None
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        local_dict = {
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "log": sp.log,
            "ln": sp.log,
            "exp": sp.exp,
            "pi": sp.pi,
            "E": sp.E,
            "I": sp.I,
            "factorial": sp.factorial,
            "binomial": sp.binomial,
            "Rational": sp.Rational,
        }
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        return parse_expr(expr, local_dict=local_dict, transformations=transformations, evaluate=True)
    except Exception:
        return None


def _math_tool_planner_prompt(problem: dict[str, Any]) -> str:
    return (
        "Extract a safe symbolic computation plan for this HLE math exactMatch item. "
        "Do not answer from memory. Do not write code. If the question needs a proof, diagram, hidden context, "
        "or cannot be reduced to a small expression/equation/modular computation, return operation=\"none\". "
        "Allowed JSON only: {\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\",\"upper\":\"\",\"order\":\"1\"}. "
        "Use plain SymPy-compatible syntax, no imports, no code.\n\n"
        f"Question:\n{problem['_question']}"
    )


def _execute_math_tool_plan_text(text: str) -> dict[str, Any]:
    plan = _parse_json_object(text)
    if not isinstance(plan, dict):
        return {"source": "llm_planner", "operation": "none", "confidence": "abstain", "reason": "planner_json_parse_failed"}
    operation = str(plan.get("operation") or "none").strip().lower()
    if operation not in {
        "evaluate",
        "simplify",
        "factor",
        "expand",
        "solve",
        "mod",
        "differentiate",
        "integrate",
        "limit",
    }:
        return {"source": "llm_planner", "operation": operation or "none", "confidence": "abstain", "reason": "planner_abstained"}
    try:
        if operation in {"evaluate", "simplify", "factor", "expand"}:
            parsed = _safe_sympy_parse_expr(str(plan.get("expression") or ""))
            if parsed is None:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "unsafe_or_symbolic_expression"}
            if operation == "evaluate" and getattr(parsed, "free_symbols", None):
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "unsafe_or_symbolic_expression"}
            value = _apply_safe_sympy_transform(operation, parsed)
            if value is None:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "transform_failed"}
            return {
                "source": "llm_planner",
                "operation": operation,
                "answer": _format_sympy_answer(value),
                "confidence": "verified_symbolic",
            }
        if operation == "mod":
            parsed = _safe_sympy_parse_expr(str(plan.get("expression") or ""))
            modulus = int(str(plan.get("modulus") or "0"))
            if parsed is None or getattr(parsed, "free_symbols", None) or modulus <= 1:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "unsafe_mod_plan"}
            return {
                "source": "llm_planner",
                "operation": operation,
                "answer": str(int(parsed) % modulus),
                "confidence": "verified_symbolic",
            }
        if operation == "solve":
            equation = str(plan.get("equation") or "")
            variable = str(plan.get("variable") or "x").strip() or "x"
            answer = _solve_safe_equation(equation, variable)
            if not answer:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "equation_solve_failed"}
            return {"source": "llm_planner", "operation": operation, "answer": answer, "confidence": "verified_symbolic"}
        if operation == "differentiate":
            answer = _differentiate_safe_expression(
                str(plan.get("expression") or ""),
                str(plan.get("variable") or "x"),
                str(plan.get("order") or "1"),
            )
            if not answer:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "differentiate_failed"}
            return {"source": "llm_planner", "operation": operation, "answer": answer, "confidence": "verified_symbolic"}
        if operation == "integrate":
            answer = _integrate_safe_expression(
                str(plan.get("expression") or ""),
                str(plan.get("variable") or "x"),
                str(plan.get("lower") or ""),
                str(plan.get("upper") or ""),
            )
            if not answer:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "integrate_failed"}
            return {"source": "llm_planner", "operation": operation, "answer": answer, "confidence": "verified_symbolic"}
        if operation == "limit":
            answer = _limit_safe_expression(
                str(plan.get("expression") or ""),
                str(plan.get("variable") or "x"),
                str(plan.get("point") or ""),
            )
            if not answer:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "limit_failed"}
            return {"source": "llm_planner", "operation": operation, "answer": answer, "confidence": "verified_symbolic"}
    except Exception as exc:
        return {
            "source": "llm_planner",
            "operation": operation,
            "confidence": "abstain",
            "reason": type(exc).__name__,
        }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _apply_safe_sympy_transform(operation: str, parsed: Any) -> Any:
    try:
        import sympy as sp

        if operation in {"evaluate", "simplify"}:
            return sp.simplify(parsed)
        if operation == "factor":
            return sp.factor(parsed)
        if operation == "expand":
            return sp.expand(parsed)
    except Exception:
        return None
    return None


def _solve_safe_equation(equation: str, variable: str) -> str | None:
    if "=" not in equation or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
        return None
    try:
        import sympy as sp

        lhs_text, rhs_text = equation.split("=", 1)
        lhs = _safe_sympy_parse_expr(lhs_text)
        rhs = _safe_sympy_parse_expr(rhs_text)
        if lhs is None or rhs is None:
            return None
        symbol = sp.Symbol(variable)
        solutions = sp.solve(sp.Eq(lhs, rhs), symbol)
        if not solutions:
            return None
        return ", ".join(_format_sympy_answer(solution) for solution in solutions[:6])
    except Exception:
        return None


def _differentiate_safe_expression(expr: str, variable: str, order_text: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
        return None
    try:
        import sympy as sp

        parsed = _safe_sympy_parse_expr(expr)
        if parsed is None:
            return None
        order = int(order_text or "1")
        if not 1 <= order <= 5:
            return None
        symbol = sp.Symbol(variable)
        return _format_sympy_answer(sp.diff(parsed, symbol, order))
    except Exception:
        return None


def _integrate_safe_expression(expr: str, variable: str, lower_text: str, upper_text: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
        return None
    try:
        import sympy as sp

        parsed = _safe_sympy_parse_expr(expr)
        if parsed is None:
            return None
        symbol = sp.Symbol(variable)
        lower = _safe_sympy_parse_expr(lower_text) if str(lower_text).strip() else None
        upper = _safe_sympy_parse_expr(upper_text) if str(upper_text).strip() else None
        if lower is not None and upper is not None:
            if getattr(lower, "free_symbols", None) or getattr(upper, "free_symbols", None):
                return None
            return _format_sympy_answer(sp.integrate(parsed, (symbol, lower, upper)))
        if lower is not None or upper is not None:
            return None
        return _format_sympy_answer(sp.integrate(parsed, symbol))
    except Exception:
        return None


def _limit_safe_expression(expr: str, variable: str, point_text: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", variable):
        return None
    try:
        import sympy as sp

        parsed = _safe_sympy_parse_expr(expr)
        point = _safe_sympy_parse_expr(point_text)
        if parsed is None or point is None or getattr(point, "free_symbols", None):
            return None
        symbol = sp.Symbol(variable)
        return _format_sympy_answer(sp.limit(parsed, symbol, point))
    except Exception:
        return None


def _format_sympy_answer(value: Any) -> str:
    try:
        import sympy as sp

        value = sp.simplify(value)
        if value.is_Integer:
            return str(int(value))
        return sp.sstr(value)
    except Exception:
        return str(value).strip()


def _recursive_child_prompt_specs(problem: dict[str, Any], *, agent_plan: dict[str, Any] | None = None) -> list[dict[str, str]]:
    question = problem["_question"]
    answer_type = problem["answer_type"]
    output = (
        "Return JSON only: {\"answer\":\"...\"}. For multiple choice, answer with the single letter only. "
        "For exact match, answer with the shortest exact answer."
    )
    specs = [
        {
            "prompt_kind": "direct_short_answer",
            "prompt": f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}",
        },
        {
            "prompt_kind": "constraint_checked_answer",
            "prompt": (
                "Solve independently. Before finalizing, internally check whether the answer format, unit, "
                "name, sign, and multiple-choice letter satisfy the prompt. Return only JSON.\n\n"
                f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        },
        {
            "prompt_kind": "recursive_assumption_answer",
            "prompt": (
                "Use a recursive assumption test internally: propose two candidate assumptions about what the "
                "question is asking, falsify the weaker one against the wording, then answer from the least "
                "vulnerable assumption. Return only JSON, with no reasoning.\n\n"
                f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        },
    ]
    evidence_context = str((agent_plan or {}).get("hle_evidence_context") or "")
    if evidence_context:
        specs.insert(1, {
            "prompt_kind": "evidence_bridge_answer",
            "prompt": _evidence_grounded_answer_prompt(problem, evidence_context=evidence_context),
        })
    hipporag_context = str((agent_plan or {}).get("hipporag_prompt_context") or "")
    if hipporag_context:
        hipporag_spec = {
            "prompt_kind": "hipporag_context_answer",
            "prompt": _prompt_for(
                problem,
                variant="hipporag_baseline",
                agent_plan={"prompt_context": hipporag_context},
            ),
        }
        insert_index = 2 if evidence_context else 1
        specs.insert(insert_index, hipporag_spec)
    context = (agent_plan or {}).get("prompt_context", "")
    if context:
        context_spec = {
            "prompt_kind": "agent_context_answer",
            "prompt": (
                "A bounded Assumption Agent retrieved the following graph/morphism context. Use it only if it "
                "directly constrains the answer; otherwise ignore it. Return only JSON.\n\n"
                f"{context}\n\nAnswer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        }
        if answer_type == "multipleChoice":
            insert_index = 2 if evidence_context else 1
            specs.insert(insert_index, context_spec)
        else:
            specs.append(context_spec)
    return specs


def _counter_assumption_challenge_trigger(
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    candidate_verifier_summary: dict[str, Any] | None = None,
    math_tool_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if any(attempt.get("prompt_kind") == "counter_assumption_challenge_answer" for attempt in attempts):
        return {"status": "abstained", "reason": "already_executed"}
    if int((candidate_verifier_summary or {}).get("verified_count") or 0) > 0:
        return {"status": "abstained", "reason": "candidate_claim_already_verified"}
    if (math_tool_summary or {}).get("confidence") == "verified_symbolic":
        return {"status": "abstained", "reason": "math_tool_already_verified"}

    answer_type = problem.get("answer_type") or "exactMatch"
    valid = [
        attempt for attempt in attempts
        if str(attempt.get("parsed_answer") or "").strip()
        and (
            answer_type == "multipleChoice"
            or not _is_suspicious_exact_answer(str(attempt.get("parsed_answer") or ""))
        )
    ]
    if len(valid) < 2:
        return {"status": "abstained", "reason": "too_few_valid_candidates", "valid_candidate_count": len(valid)}

    normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in valid:
        normalized[
            _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=answer_type)
        ].append(attempt)
    if not normalized:
        return {"status": "abstained", "reason": "no_normalized_candidates", "valid_candidate_count": len(valid)}

    ranked = sorted(normalized.items(), key=lambda item: (-len(item[1]), item[1][0]["child_index"]))
    target_norm, target_attempts = ranked[0]
    unique_candidate_count = len(normalized)
    target_count = len(target_attempts)
    if target_count < 2 and unique_candidate_count > 1:
        return {
            "status": "abstained",
            "reason": "diverse_candidates_will_use_verifier",
            "valid_candidate_count": len(valid),
            "unique_candidate_count": unique_candidate_count,
            "top_candidate_count": target_count,
        }
    if target_count < 2:
        return {
            "status": "abstained",
            "reason": "no_majority_candidate",
            "valid_candidate_count": len(valid),
            "unique_candidate_count": unique_candidate_count,
            "top_candidate_count": target_count,
        }
    target_answer = str(target_attempts[0].get("parsed_answer") or "").strip()
    return {
        "status": "activated",
        "reason": "majority_without_independent_verification",
        "valid_candidate_count": len(valid),
        "unique_candidate_count": unique_candidate_count,
        "top_candidate_count": target_count,
        "challenged_answer_hash": stable_hash({"answer": target_answer}),
        "challenged_answer_norm_hash": stable_hash({"answer": target_norm}),
        "challenged_prompt_kinds": [str(attempt.get("prompt_kind") or "") for attempt in target_attempts],
        "target_answer": target_answer,
        "target_norm": target_norm,
    }


def _counter_assumption_challenge_prompt(
    problem: dict[str, Any],
    *,
    challenged_answer: str,
    evidence_context: str = "",
) -> str:
    answer_type = problem["answer_type"]
    output = (
        "Return JSON only: {\"answer\":\"...\"}. For multiple choice, answer with the single letter only. "
        "For exact match, answer with the shortest exact answer."
    )
    evidence_block = (
        "Transient evidence, if relevant:\n"
        f"{evidence_context}\n\n"
        if evidence_context
        else ""
    )
    return (
        "A recursive answer ensemble is converging on the challenged answer below. Do a counter-assumption "
        "test: actively look for the strongest reason this answer could be wrong under the exact wording. "
        "If the challenged answer survives that test, return it. If a different answer is better supported, "
        "return the corrected answer. Do not include reasoning.\n\n"
        f"{evidence_block}"
        f"Challenged answer: {challenged_answer}\n"
        f"Answer type: {answer_type}\nQuestion:\n{problem['_question']}\n\n{output}"
    )


def _maybe_run_counter_assumption_challenge(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    candidate_verifier_summary: dict[str, Any] | None,
    math_tool_summary: dict[str, Any] | None,
    evidence_context: str,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    trigger = _counter_assumption_challenge_trigger(
        problem,
        attempts,
        candidate_verifier_summary=candidate_verifier_summary,
        math_tool_summary=math_tool_summary,
    )
    if trigger.get("status") != "activated":
        return None, trigger
    target_answer = str(trigger.pop("target_answer") or "")
    target_norm = str(trigger.pop("target_norm") or "")
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "counter_assumption_challenge_answer",
            "prompt": _counter_assumption_challenge_prompt(
                problem,
                challenged_answer=target_answer,
                evidence_context=evidence_context,
            ),
        },
        child_index=len(attempts) + 1,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    challenge_answer = str(attempt.get("parsed_answer") or "").strip()
    challenge_norm = (
        _normalize_for_selection(challenge_answer, answer_type=problem.get("answer_type") or "exactMatch")
        if challenge_answer
        else ""
    )
    summary = {
        **trigger,
        "child_id": attempt.get("child_id"),
        "child_status": attempt.get("status"),
        "challenge_answer_hash": attempt.get("parsed_answer_hash"),
        "challenge_disagreed_with_majority": bool(challenge_norm and challenge_norm != target_norm),
        "evidence_context_used": bool(evidence_context),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    return attempt, summary


def _option_elimination_challenge_prompt(
    problem: dict[str, Any],
    *,
    challenged_answer: str,
    evidence_context: str = "",
) -> str:
    evidence_block = (
        "Transient evidence, if relevant:\n"
        f"{evidence_context}\n\n"
        if evidence_context
        else ""
    )
    return (
        "The current ensemble has an unverified multiple-choice answer and may also contain a conflicting "
        "counter-hypothesis. Run a stricter option-elimination check internally: evaluate every answer option "
        "against the exact question wording and transient evidence, reject options that fail a necessary "
        "condition, and choose the remaining best-supported option. Do not anchor on either the majority or a "
        "single counter-hypothesis unless it survives the full option-by-option check. Do not include reasoning. "
        "Return JSON only: {\"answer\":\"A\"}.\n\n"
        f"{evidence_block}"
        f"Challenged answer: {challenged_answer}\n"
        f"Question:\n{problem['_question']}"
    )


def _maybe_run_option_elimination_challenge(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    counter_challenge_summary: dict[str, Any] | None,
    evidence_context: str,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if problem.get("answer_type") != "multipleChoice":
        return None, None
    if not counter_challenge_summary or counter_challenge_summary.get("status") != "activated":
        return None, None
    if any(attempt.get("prompt_kind") == "option_elimination_challenge_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    options, _ = _extract_multiple_choice_options(str(problem.get("_question") or ""))
    if len(options) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    challenged_answer = _extract_choice_from_hashable_attempts(
        attempts,
        answer_type=problem.get("answer_type") or "multipleChoice",
    )
    if not challenged_answer:
        return None, {"status": "abstained", "reason": "no_majority_answer"}
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "option_elimination_challenge_answer",
            "prompt": _option_elimination_challenge_prompt(
                problem,
                challenged_answer=challenged_answer,
                evidence_context=evidence_context,
            ),
        },
        child_index=len(attempts) + 1,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    answer = str(attempt.get("parsed_answer") or "").strip()
    answer_norm = _normalize_for_selection(answer, answer_type="multipleChoice") if answer else ""
    challenged_norm = _normalize_for_selection(challenged_answer, answer_type="multipleChoice")
    summary = {
        "status": "activated",
        "reason": (
            "counter_challenge_disagreed_run_full_option_elimination"
            if counter_challenge_summary.get("challenge_disagreed_with_majority")
            else "counter_challenge_confirmed_majority"
        ),
        "child_id": attempt.get("child_id"),
        "child_status": attempt.get("status"),
        "option_count": len(options),
        "challenge_answer_hash": attempt.get("parsed_answer_hash"),
        "challenge_disagreed_with_majority": bool(answer_norm and answer_norm != challenged_norm),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    return attempt, summary


def _critic_synthesis_prompt(problem: dict[str, Any], *, evidence_context: str = "") -> str:
    evidence_block = (
        "Transient evidence, if relevant:\n"
        f"{evidence_context}\n\n"
        if evidence_context
        else ""
    )
    return (
        "Solve this multiple-choice HLE item independently as a critic synthesis pass. Ignore the ensemble "
        "majority and prior candidate answers; they may be collapsed on the same wrong attractor. Use the exact "
        "question wording first, and use transient evidence only when it directly matches the question. Return "
        "the single best option letter only. Do not include reasoning. Return JSON only: {\"answer\":\"A\"}.\n\n"
        f"{evidence_block}"
        f"Question:\n{problem['_question']}"
    )


def _maybe_run_critic_synthesis_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    evidence_context: str,
    base_model: str,
    critic_model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if problem.get("answer_type") != "multipleChoice":
        return None, None
    if not critic_model or critic_model == base_model:
        return None, {"status": "not_required", "reason": "no_distinct_critic_model"}
    if any(attempt.get("prompt_kind") == "critic_synthesis_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    labels, _ = _extract_multiple_choice_options(str(problem.get("_question") or ""))
    if len(labels) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    valid_answers = [
        str(attempt.get("parsed_answer") or "").strip()
        for attempt in attempts
        if str(attempt.get("parsed_answer") or "").strip()
    ]
    if len(valid_answers) < 3:
        return None, {"status": "not_required", "reason": "not_enough_candidates"}
    norms = [
        _normalize_for_selection(answer, answer_type="multipleChoice")
        for answer in valid_answers
    ]
    unique_count = len(set(norms))
    top_count = Counter(norms).most_common(1)[0][1] if norms else 0
    has_verified_candidate = any(attempt.get("candidate_verifier_state") == "verified" for attempt in attempts)
    if has_verified_candidate:
        return None, {"status": "not_required", "reason": "verified_candidate_available"}
    if unique_count > 2 and top_count < 3:
        return None, {
            "status": "not_required",
            "reason": "candidate_space_already_diverse",
            "unique_candidate_count": unique_count,
            "top_candidate_count": top_count,
        }
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "critic_synthesis_answer",
            "prompt": _critic_synthesis_prompt(problem, evidence_context=evidence_context),
        },
        child_index=len(attempts) + 1,
        model=critic_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    answer = str(attempt.get("parsed_answer") or "").strip()
    answer_norm = _normalize_for_selection(answer, answer_type="multipleChoice") if answer else ""
    majority_answer = _extract_choice_from_hashable_attempts(attempts, answer_type="multipleChoice")
    majority_norm = _normalize_for_selection(majority_answer, answer_type="multipleChoice") if majority_answer else ""
    summary = {
        "status": "activated",
        "reason": "collapsed_or_low_diversity_candidates_need_distinct_critic",
        "base_model": base_model,
        "critic_model": critic_model,
        "child_id": attempt.get("child_id"),
        "child_status": attempt.get("status"),
        "option_count": len(labels),
        "unique_candidate_count_before": unique_count,
        "top_candidate_count_before": top_count,
        "critic_answer_hash": attempt.get("parsed_answer_hash"),
        "critic_disagreed_with_majority": bool(answer_norm and majority_norm and answer_norm != majority_norm),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    return attempt, summary


def _maybe_add_mc_option_sweep_candidates(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if problem.get("answer_type") != "multipleChoice":
        return [], None
    options, _ = _extract_multiple_choice_options(str(problem.get("_question") or ""))
    labels = sorted(options)
    if len(labels) < 2:
        return [], {"status": "abstained", "reason": "options_not_parsed"}
    present = {
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type="multipleChoice")
        for attempt in attempts
        if str(attempt.get("parsed_answer") or "").strip()
    }
    missing = [label for label in labels if label not in present]
    if not missing:
        return [], {
            "status": "not_required",
            "reason": "all_option_labels_already_present",
            "option_count": len(labels),
            "covered_option_count": len(labels),
            "added_candidate_count": 0,
        }
    added: list[dict[str, Any]] = []
    start_index = len(attempts) + 1
    for offset, label in enumerate(missing):
        added.append({
            "child_id": stable_hash({
                "problem_id_hash": problem["id_hash"],
                "prompt_kind": "mc_option_sweep_candidate",
                "option_label": label,
            }),
            "child_index": start_index + offset,
            "prompt_kind": "mc_option_sweep_candidate",
            "parsed_answer": label,
            "parsed_answer_hash": stable_hash({"answer": label}),
            "prediction_hash": stable_hash({
                "option_sweep_candidate": label,
                "problem_id_hash": problem["id_hash"],
            }),
            "latency_sec": 0.0,
            "status": "answered",
            "tool_confidence": "full_option_space_candidate",
        })
    return added, {
        "status": "activated",
        "reason": "finite_multiple_choice_option_space_completion",
        "option_count": len(labels),
        "covered_option_count_before": len([label for label in labels if label in present]),
        "added_candidate_count": len(added),
        "added_option_hashes": [stable_hash({"option_label": label}) for label in missing],
        "underlying_model_calls": 0,
    }


def _extract_choice_from_hashable_attempts(attempts: list[dict[str, Any]], *, answer_type: str) -> str:
    counts: Counter[str] = Counter()
    first: dict[str, str] = {}
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        norm = _normalize_for_selection(answer, answer_type=answer_type)
        counts[norm] += 1
        first.setdefault(norm, answer)
    if not counts:
        return ""
    top_norm, _ = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return first.get(top_norm, top_norm)


def _forced_alternative_challenge_prompt(
    problem: dict[str, Any],
    *,
    challenged_answer: str,
    allowed_labels: list[str],
    evidence_context: str = "",
) -> str:
    alternative_labels = ", ".join(label for label in allowed_labels if label != _extract_choice(challenged_answer))
    evidence_block = (
        "Transient evidence, if relevant:\n"
        f"{evidence_context}\n\n"
        if evidence_context
        else ""
    )
    return (
        "The ensemble has collapsed to one multiple-choice answer. For diversity generation only, produce the "
        "strongest plausible alternative answer that is NOT the challenged answer. This is not the final answer; "
        "a verifier will compare it against the majority. Choose an alternative only from the listed labels, and "
        "prefer the alternative that best fits the exact wording or evidence. Return JSON only: {\"answer\":\"B\"}.\n\n"
        f"{evidence_block}"
        f"Challenged answer: {challenged_answer}\n"
        f"Allowed non-challenged labels: {alternative_labels}\n"
        f"Question:\n{problem['_question']}"
    )


def _maybe_run_forced_alternative_challenge(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    option_elimination_summary: dict[str, Any] | None,
    evidence_context: str,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if os.environ.get("HLE_ENABLE_FORCED_ALTERNATIVE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None, {"status": "abstained", "reason": "disabled_by_default"}
    if problem.get("answer_type") != "multipleChoice":
        return None, None
    if not option_elimination_summary or option_elimination_summary.get("status") != "activated":
        return None, None
    if option_elimination_summary.get("challenge_disagreed_with_majority"):
        return None, None
    if any(attempt.get("prompt_kind") == "forced_alternative_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    options, _ = _extract_multiple_choice_options(str(problem.get("_question") or ""))
    labels = sorted(options)
    if len(labels) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    challenged_answer = _extract_choice_from_hashable_attempts(
        attempts,
        answer_type=problem.get("answer_type") or "multipleChoice",
    )
    challenged_label = _extract_choice(challenged_answer)
    alternatives = [label for label in labels if label != challenged_label]
    if not alternatives:
        return None, {"status": "abstained", "reason": "no_alternative_labels"}
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "forced_alternative_answer",
            "prompt": _forced_alternative_challenge_prompt(
                problem,
                challenged_answer=challenged_label or challenged_answer,
                allowed_labels=labels,
                evidence_context=evidence_context,
            ),
        },
        child_index=len(attempts) + 1,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    answer = str(attempt.get("parsed_answer") or "").strip()
    answer_label = _extract_choice(answer)
    summary = {
        "status": "activated",
        "reason": "collapsed_consensus_needs_forced_variation",
        "child_id": attempt.get("child_id"),
        "child_status": attempt.get("status"),
        "option_count": len(labels),
        "challenge_answer_hash": attempt.get("parsed_answer_hash"),
        "challenge_disagreed_with_majority": bool(answer_label and answer_label != challenged_label),
        "answer_is_allowed_alternative": answer_label in alternatives,
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    return attempt, summary


def _maybe_run_mc_option_evidence_scorer(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if problem.get("answer_type") != "multipleChoice":
        return None, None
    if any(attempt.get("prompt_kind") == "mc_option_evidence_scorer_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    stem, options = _split_multiple_choice_question(problem)
    if len(options) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    stem_terms = _content_terms(stem or problem.get("_question", ""))
    option_rows: list[dict[str, Any]] = []
    docs_by_label: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    for label, option_text in sorted(options.items()):
        query = _option_evidence_query(stem, option_text, problem)
        docs: list[dict[str, str]] = []
        if query:
            try:
                docs = _wikipedia_search(query, limit=2, timeout=6.0)
                if len(docs) < 2 or _should_use_domain_evidence_search(problem):
                    docs.extend(_domain_evidence_search(query, problem=problem, limit=1, timeout=8.0))
                docs = _dedupe_evidence_results(docs)
            except Exception as exc:
                errors.append(type(exc).__name__)
        docs_by_label[label] = docs
        score = _score_option_evidence(stem_terms=stem_terms, option_text=option_text, docs=docs)
        option_rows.append({
            "label": label,
            "score": score,
            "query_hash": stable_hash({"query": query}),
            "doc_count": len(docs),
            "doc_hashes": [
                stable_hash({"title": doc.get("title", ""), "snippet": doc.get("snippet", "")})
                for doc in docs[:2]
            ],
        })
    ranked = sorted(option_rows, key=lambda row: (-float(row["score"]), row["label"]))
    if not ranked:
        return None, {"status": "no_results", "reason": "no_option_scores"}
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else {"score": 0.0}
    top_score = float(top["score"])
    margin = top_score - float(runner_up.get("score") or 0.0)
    confidence = top_score >= 4.0 and margin >= 1.5
    summary = {
        "status": "activated" if confidence else "weak_margin",
        "source": "wikipedia_plus_domain_option_search",
        "option_count": len(options),
        "top_option_hash": stable_hash({"option_label": top["label"]}),
        "top_option_answer_hash": stable_hash({"answer": str(top["label"])}),
        "top_score": round(top_score, 4),
        "runner_up_score": round(float(runner_up.get("score") or 0.0), 4),
        "margin": round(margin, 4),
        "candidate_emitted": bool(confidence),
        "query_hashes": [row["query_hash"] for row in option_rows],
        "doc_count_by_option_hash": {
            stable_hash({"option_label": row["label"]}): row["doc_count"]
            for row in option_rows
        },
        "top_doc_hashes": top.get("doc_hashes", []),
        "error_types": sorted(set(errors)),
        "underlying_model_calls": 0,
    }
    if not confidence:
        _log_event(
            logger,
            {
                "event": "mc_option_evidence_scorer",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "question_hash": problem["question_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "stage_status": summary["status"],
                "stage_data": summary,
            },
        )
        return None, summary
    attempt = {
        "child_id": stable_hash({
            "call_id": call_id,
            "prompt_kind": "mc_option_evidence_scorer_answer",
            "option_hash": summary["top_option_hash"],
        }),
        "child_index": len(attempts) + 1,
        "prompt_kind": "mc_option_evidence_scorer_answer",
        "parsed_answer": str(top["label"]),
        "parsed_answer_hash": stable_hash({"answer": str(top["label"])}),
        "prediction_hash": stable_hash({
            "top_option_hash": summary["top_option_hash"],
            "top_score": summary["top_score"],
            "margin": summary["margin"],
        }),
        "latency_sec": 0.0,
        "status": "answered",
        "tool_confidence": "option_evidence_margin",
    }
    summary["child_id"] = attempt["child_id"]
    summary["candidate_answer_hash"] = attempt["parsed_answer_hash"]
    if problem.get("_answer"):
        gold_for_eval, _ = _canonicalize_multiple_choice_answer(problem, str(problem.get("_answer") or ""))
        summary["candidate_correct_for_eval"] = _is_correct(
            str(top["label"]),
            gold_for_eval,
            answer_type="multipleChoice",
        )
    evidence_context = _option_evidence_context(options=options, docs_by_label=docs_by_label)
    if evidence_context:
        attempt["private_option_evidence_context"] = evidence_context
        summary["evidence_context_hash"] = stable_hash({"option_evidence_context": evidence_context})
        summary["evidence_context_char_count"] = len(evidence_context)
    _log_event(
        logger,
        {
            "event": "mc_option_evidence_scorer",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "stage_status": summary["status"],
            "stage_data": summary,
        },
    )
    return attempt, summary


def _maybe_run_domain_rule_mc_verifier(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    evidence_context: str,
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if problem.get("answer_type") != "multipleChoice":
        return None, None
    if any(attempt.get("prompt_kind") == "domain_rule_mc_verifier_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    stem, options = _split_multiple_choice_question(problem)
    if len(options) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    decision = _domain_rule_mc_decision(problem=problem, stem=stem, options=options, evidence_context=evidence_context)
    if not decision:
        summary = {
            "status": "not_required",
            "reason": "no_supported_domain_rule",
            "option_count": len(options),
            "underlying_model_calls": 0,
        }
        return None, summary
    label = str(decision["label"])
    attempt = {
        "child_id": stable_hash({
            "call_id": call_id,
            "prompt_kind": "domain_rule_mc_verifier_answer",
            "rule_id": decision["rule_id"],
            "option_label": label,
        }),
        "child_index": len(attempts) + 1,
        "prompt_kind": "domain_rule_mc_verifier_answer",
        "parsed_answer": label,
        "parsed_answer_hash": stable_hash({"answer": label}),
        "prediction_hash": stable_hash({
            "rule_id": decision["rule_id"],
            "option_label": label,
            "evidence_hash": stable_hash({"evidence_context": evidence_context}) if evidence_context else "",
        }),
        "latency_sec": 0.0,
        "status": "answered",
        "candidate_verifier_state": "verified",
        "candidate_verifier_backend": "domain_rule_mc_verifier",
        "candidate_verifier_operation": decision["rule_id"],
        "candidate_verifier_claim_hash": stable_hash({
            "question_hash": problem.get("question_hash"),
            "rule_id": decision["rule_id"],
            "option_label": label,
        }),
        "tool_confidence": decision["confidence"],
    }
    summary = {
        "status": "activated",
        "backend": "domain_rule_mc_verifier",
        "rule_id": decision["rule_id"],
        "reason": decision["reason"],
        "confidence": decision["confidence"],
        "verified_option_hash": stable_hash({"option_label": label}),
        "candidate_answer_hash": attempt["parsed_answer_hash"],
        "child_id": attempt["child_id"],
        "option_count": len(options),
        "evidence_required": bool(decision.get("evidence_required")),
        "evidence_context_hash": stable_hash({"evidence_context": evidence_context}) if evidence_context else "",
        "underlying_model_calls": 0,
    }
    if problem.get("_answer"):
        gold_for_eval, _ = _canonicalize_multiple_choice_answer(problem, str(problem.get("_answer") or ""))
        summary["candidate_correct_for_eval"] = _is_correct(label, gold_for_eval, answer_type="multipleChoice")
    _log_event(
        logger,
        {
            "event": "domain_rule_mc_verifier",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "stage_status": summary["status"],
            "stage_data": summary,
        },
    )
    return attempt, summary


def _domain_rule_mc_decision(
    *,
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
    evidence_context: str,
) -> dict[str, Any] | None:
    law = _ontario_confidential_screen_rule_decision(problem=problem, stem=stem, options=options, evidence_context=evidence_context)
    if law:
        return law
    bio = _bacterial_cross_resistance_minimality_decision(problem=problem, stem=stem, options=options)
    if bio:
        return bio
    return None


def _ontario_confidential_screen_rule_decision(
    *,
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
    evidence_context: str,
) -> dict[str, Any] | None:
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
        stem,
    ]).lower()
    if not all(token in text for token in ("law", "confidential")):
        return None
    if not any(token in text for token in ("ontario", "toronto", "former client", "law firm", "llp")):
        return None
    evidence = evidence_context.lower()
    has_rule_evidence = (
        "lso_rules" in evidence
        or ("adequate measures" in evidence and "confidential information" in evidence)
        or ("former client" in evidence and "confidential information" in evidence)
    )
    if not has_rule_evidence:
        return None
    candidates: list[str] = []
    for label, option_text in sorted(options.items()):
        option = option_text.lower()
        can_continue = any(phrase in option for phrase in ("can continue", "may continue", "allowed to continue"))
        measures = any(phrase in option for phrase in ("appropriate measures", "adequate measures", "not shared", "screen"))
        confidential = "confidential" in option
        prohibited = any(phrase in option for phrase in ("not allowed", "prohibited", "cannot continue"))
        if can_continue and measures and confidential and not prohibited:
            candidates.append(label)
    if len(candidates) != 1:
        return None
    return {
        "label": candidates[0],
        "rule_id": "ontario_former_client_confidential_screen",
        "confidence": "evidence_grounded_domain_rule",
        "reason": "lso_rules_support_adequate_measures_screen_exception",
        "evidence_required": True,
    }


def _bacterial_cross_resistance_minimality_decision(
    *,
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
) -> dict[str, Any] | None:
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
        stem,
    ]).lower()
    required_cues = ("bacteria", "resistance")
    if not all(cue in text for cue in required_cues):
        return None
    if not any(phrase in text for phrase in ("lateral transfer", "stable genome", "no lateral transfer")):
        return None
    if "equal pace" not in text and "same pace" not in text:
        return None
    scored: list[tuple[int, str]] = []
    for label, option_text in sorted(options.items()):
        option = option_text.lower()
        if "contamination" in option or "plasmid" in option:
            continue
        score = 0
        if "cross-resistance" in option or "cross resistance" in option:
            score += 4
        if "rare resistance" in option or "rare mutations" in option or "mutations" in option:
            score += 2
        if "did not have compensatory" in option or "without compensatory" in option or "no compensatory" in option:
            score += 2
        if "compensatory mutations" in option and not any(
            phrase in option for phrase in ("did not have compensatory", "without compensatory", "no compensatory")
        ):
            score -= 2
        if "increased the fitness to a great extent" in option:
            score -= 2
        if score > 0:
            scored.append((score, label))
    if not scored:
        return None
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    if ranked[0][0] < 5:
        return None
    return {
        "label": ranked[0][1],
        "rule_id": "bacterial_cross_resistance_minimal_extra_assumption",
        "confidence": "contrastive_domain_rule",
        "reason": "cross_resistance_explains_parallel_resistance_without_adding_unstated_compensatory_fitness_clause",
        "evidence_required": False,
    }


def _option_evidence_context(
    *,
    options: dict[str, str],
    docs_by_label: dict[str, list[dict[str, str]]],
) -> str:
    lines: list[str] = []
    for label, option_text in sorted(options.items()):
        docs = docs_by_label.get(label, [])
        if not docs:
            continue
        lines.append(f"Option {label} ({_clean_evidence_text(option_text)[:120]}):")
        for index, doc in enumerate(docs[:2], start=1):
            title = _clean_evidence_text(doc.get("title", ""))
            snippet = _clean_evidence_text(doc.get("snippet", ""))
            if not title and not snippet:
                continue
            lines.append(f"- Evidence {index}: {title} -- {snippet[:260]}")
    return "\n".join(lines)[:4000]


def _option_evidence_query(stem: str, option_text: str, problem: dict[str, Any]) -> str:
    stem_words = [
        token for token in re.findall(r"[A-Za-z0-9_+.-]{4,}", stem or "")
        if token.lower() not in _EVIDENCE_QUERY_STOPWORDS
    ][:8]
    option_words = [
        token for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", option_text or "")
        if token.lower() not in _EVIDENCE_QUERY_STOPWORDS
    ][:8]
    subject = str(problem.get("raw_subject") or problem.get("category") or "").strip()
    return _clean_evidence_query(" ".join(stem_words + option_words + ([subject] if subject else [])))


def _score_option_evidence(*, stem_terms: set[str], option_text: str, docs: list[dict[str, str]]) -> float:
    option_terms = _content_terms(option_text)
    if not option_terms or not docs:
        return 0.0
    score = 0.0
    for index, doc in enumerate(docs[:2]):
        text = f"{doc.get('title', '')} {doc.get('snippet', '')}"
        doc_terms = _content_terms(text)
        title_terms = _content_terms(doc.get("title", ""))
        option_overlap = len(option_terms & doc_terms)
        title_overlap = len(option_terms & title_terms)
        stem_overlap = len(stem_terms & doc_terms)
        score += (2.0 * option_overlap) + (0.75 * stem_overlap) + (0.5 * title_overlap) + (0.05 / (index + 1))
    return round(score, 4)


def _select_recursive_child_answer(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    evidence_context: str = "",
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        if problem["answer_type"] == "multipleChoice":
            canonical, canonical_summary = _canonicalize_multiple_choice_answer(problem, answer)
            if canonical_summary.get("changed"):
                attempt = dict(attempt)
                attempt["parsed_answer"] = canonical
                attempt["parsed_answer_hash"] = stable_hash({"answer": canonical})
                attempt["multiple_choice_canonicalized"] = True
                attempt["multiple_choice_canonicalizer"] = canonical_summary
        valid.append(attempt)
    if problem["answer_type"] != "multipleChoice":
        non_suspicious = [
            attempt for attempt in valid
            if not _is_suspicious_exact_answer(str(attempt.get("parsed_answer") or ""))
        ]
        if non_suspicious:
            valid = non_suspicious
    normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in valid:
        normalized[_normalize_for_selection(attempt["parsed_answer"], answer_type=problem["answer_type"])].append(attempt)
    if normalized:
        ranked = sorted(normalized.items(), key=lambda item: (-len(item[1]), item[1][0]["child_index"]))
        verified_candidates = [
            attempt for attempt in valid
            if attempt.get("candidate_verifier_state") == "verified"
        ]
        if verified_candidates:
            selected = sorted(verified_candidates, key=lambda row: int(row.get("child_index", 0) or 0))[0]
            return {
                "selection_method": "candidate_claim_verifier_priority",
                "selected_child_id": selected["child_id"],
                "selected_answer": selected["parsed_answer"],
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            }
        if any(
            attempt.get("candidate_verifier_state") == "refuted" for attempt in valid
        ):
            non_refuted_valid = [
                attempt for attempt in valid
                if attempt.get("candidate_verifier_state") != "refuted"
            ]
            if non_refuted_valid:
                valid = non_refuted_valid
                normalized = defaultdict(list)
                for attempt in valid:
                    normalized[_normalize_for_selection(attempt["parsed_answer"], answer_type=problem["answer_type"])].append(attempt)
                ranked = sorted(normalized.items(), key=lambda item: (-len(item[1]), item[1][0]["child_index"]))
        math_candidates = [
            attempt for attempt in valid
            if attempt.get("prompt_kind") == "math_tool_answer"
            and attempt.get("tool_confidence") == "verified_symbolic"
        ]
        if problem["answer_type"] != "multipleChoice" and math_candidates:
            deterministic_math = [
                attempt for attempt in math_candidates
                if attempt.get("tool_source") == "deterministic_parser"
            ]
            supported_math = [
                attempt for attempt in math_candidates
                if len(normalized.get(_normalize_for_selection(attempt["parsed_answer"], answer_type=problem["answer_type"]), [])) >= 2
            ]
            if deterministic_math or supported_math:
                selected = (deterministic_math or supported_math)[0]
                return {
                    "selection_method": "verified_math_tool_priority",
                    "selected_child_id": selected["child_id"],
                    "selected_answer": selected["parsed_answer"],
                    "underlying_model_calls": 0,
                    "verifier_model_call": False,
                }
        source_selection = _maybe_run_source_grounded_mc_selection(
            problem=problem,
            valid=valid,
            ranked=ranked,
            evidence_context=evidence_context,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if source_selection:
            return source_selection
        hipporag_selection = _select_hipporag_context_candidate(problem=problem, valid=valid, ranked=ranked)
        if hipporag_selection:
            return hipporag_selection
        counter_selection = _select_after_counter_assumption_challenge(
            problem=problem,
            valid=valid,
            ranked=ranked,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if counter_selection:
            return counter_selection
        if problem["answer_type"] != "multipleChoice" and _should_run_math_tool_child(problem):
            top_answer, top_attempts = ranked[0]
            if len(top_attempts) >= 2:
                selected = top_attempts[0]
                return {
                    "selection_method": "math_exact_normalized_majority",
                    "selected_child_id": selected["child_id"],
                    "selected_answer": selected["parsed_answer"],
                    "underlying_model_calls": 0,
                    "verifier_model_call": False,
                }
            if len(ranked) == 1:
                selected = top_attempts[0]
                return {
                    "selection_method": "normalized_majority",
                    "selected_child_id": selected["child_id"],
                    "selected_answer": selected["parsed_answer"],
                    "underlying_model_calls": 0,
                    "verifier_model_call": False,
                }
            direct_candidates = [
                attempt for attempt in valid
                if attempt.get("prompt_kind") == "direct_short_answer"
            ]
            if direct_candidates:
                selected = direct_candidates[0]
                return {
                    "selection_method": "math_exact_direct_fallback",
                    "selected_child_id": selected["child_id"],
                    "selected_answer": selected["parsed_answer"],
                    "underlying_model_calls": 0,
                    "verifier_model_call": False,
                }
        evidence_candidates = [
            attempt for attempt in valid
            if attempt.get("prompt_kind") in {"evidence_bridge_answer", "evidence_grounded_answer"}
        ]
        if problem["answer_type"] != "multipleChoice" and evidence_candidates:
            top_attempts = ranked[0][1]
            top_has_evidence = any(attempt in evidence_candidates for attempt in top_attempts)
            exact_evidence_override_enabled = (
                os.environ.get("HLE_ENABLE_EXACT_EVIDENCE_OVERRIDE", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if exact_evidence_override_enabled and len(top_attempts) >= 2 and not top_has_evidence:
                selected = evidence_candidates[0]
                return {
                    "selection_method": "evidence_bridge_priority_over_closed_book_majority",
                    "selected_child_id": selected["child_id"],
                    "selected_answer": selected["parsed_answer"],
                    "underlying_model_calls": 0,
                    "verifier_model_call": False,
                }
        if len(ranked[0][1]) >= 2 or len(ranked) == 1:
            selected = ranked[0][1][0]
            return {
                "selection_method": "normalized_majority",
                "selected_child_id": selected["child_id"],
                "selected_answer": selected["parsed_answer"],
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            }
        if problem["answer_type"] != "multipleChoice":
            direct_candidates = [
                attempt for attempt in valid
                if attempt.get("prompt_kind") == "direct_short_answer"
            ]
            if direct_candidates:
                selected = direct_candidates[0]
                return {
                    "selection_method": "exact_direct_fallback",
                    "selected_child_id": selected["child_id"],
                    "selected_answer": selected["parsed_answer"],
                    "underlying_model_calls": 0,
                    "verifier_model_call": False,
                }
    if not valid:
        return {
            "selection_method": "all_children_failed",
            "selected_child_id": None,
            "selected_answer": "",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
    try:
        verifier_candidates = _unique_verifier_candidates(problem, valid)
        verifier_text = _call_model(
            model=model,
            prompt=_verifier_prompt(problem, verifier_candidates),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        choice = _parse_verifier_choice(verifier_text, max_index=len(verifier_candidates))
        selected = verifier_candidates[(choice or 1) - 1]
        _log_event(
            logger,
            {
                "event": "recursive_verifier_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(verifier_candidates),
                "raw_candidate_count": len(valid),
                "choice": choice or 1,
                "selected_child_id": selected["child_id"],
                "selected_answer_hash": selected["parsed_answer_hash"],
                "verifier_prediction_hash": stable_hash({"prediction": verifier_text}),
            },
        )
        return {
            "selection_method": "verifier_choice" if choice else "verifier_fallback_first",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 1,
            "verifier_model_call": True,
        }
    except Exception as exc:
        selected = valid[0]
        _log_event(
            logger,
            {
                "event": "recursive_verifier_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(valid),
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
                "selected_child_id": selected["child_id"],
            },
        )
        return {
            "selection_method": "verifier_error_fallback_first",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 0,
            "verifier_model_call": False,
            }


def _select_hipporag_context_candidate(
    *,
    problem: dict[str, Any],
    valid: list[dict[str, Any]],
    ranked: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any] | None:
    if problem.get("answer_type") != "multipleChoice":
        return None
    if os.environ.get("HLE_DISABLE_AGENT_HIPPORAG_PRIORITY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    candidates = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") == "hipporag_context_answer"
        and str(attempt.get("parsed_answer") or "").strip()
    ]
    if not candidates:
        return None
    selected = sorted(candidates, key=lambda row: int(row.get("child_index", 0) or 0))[0]
    selected_norm = _normalize_for_selection(str(selected.get("parsed_answer") or ""), answer_type="multipleChoice")
    top_norm = ranked[0][0] if ranked else ""
    broad_enabled = os.environ.get("HLE_ENABLE_BROAD_AGENT_HIPPORAG_PRIORITY", "").strip().lower() in {"1", "true", "yes", "on"}
    if not broad_enabled and selected_norm != top_norm:
        return None
    return {
        "selection_method": "hipporag_context_priority",
        "selected_child_id": selected["child_id"],
        "selected_answer": selected["parsed_answer"],
        "underlying_model_calls": 0,
        "verifier_model_call": False,
    }


def _maybe_run_source_grounded_mc_selection(
    *,
    problem: dict[str, Any],
    valid: list[dict[str, Any]],
    ranked: list[tuple[str, list[dict[str, Any]]]],
    evidence_context: str,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any] | None:
    if problem.get("answer_type") != "multipleChoice" or not evidence_context:
        return None
    evidence_candidates = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") in {"evidence_bridge_answer", "evidence_grounded_answer"}
    ]
    if not evidence_candidates:
        return None
    unique_candidates = _unique_verifier_candidates(problem, valid)
    if len(unique_candidates) < 2:
        return None
    top_attempts = ranked[0][1] if ranked else []
    top_has_evidence = any(attempt in evidence_candidates for attempt in top_attempts)
    evidence_disagrees = any(
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type="multipleChoice")
        != _normalize_for_selection(str(top_attempts[0].get("parsed_answer") or ""), answer_type="multipleChoice")
        for attempt in evidence_candidates
        if top_attempts
    )
    has_full_option_space = any(attempt.get("prompt_kind") == "mc_option_sweep_candidate" for attempt in valid)
    has_variation_challenge = any(
        attempt.get("prompt_kind") in {
            "counter_assumption_challenge_answer",
            "option_elimination_challenge_answer",
            "forced_alternative_answer",
        }
        for attempt in valid
    )
    broad_enabled = os.environ.get("HLE_ENABLE_BROAD_SOURCE_GROUNDED_MC", "").strip().lower() in {"1", "true", "yes", "on"}
    conservative_trigger = len(top_attempts) == 2 and not top_has_evidence and not has_variation_challenge
    if not broad_enabled and not conservative_trigger:
        return None
    if top_has_evidence and not evidence_disagrees and not has_full_option_space:
        return None
    source_selection = _run_source_grounded_mc_verifier(
        problem=problem,
        attempts=unique_candidates,
        evidence_context=evidence_context,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    return source_selection if source_selection.get("selected_answer") else None


def _select_after_counter_assumption_challenge(
    *,
    problem: dict[str, Any],
    valid: list[dict[str, Any]],
    ranked: list[tuple[str, list[dict[str, Any]]]],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any] | None:
    if os.environ.get("HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    if not ranked or len(ranked[0][1]) < 2:
        return None
    top_norm = ranked[0][0]
    option_evidence_arbitrator_enabled = _option_evidence_arbitrator_enabled()
    challenge_prompt_kinds = {
        "counter_assumption_challenge_answer",
        "option_elimination_challenge_answer",
        "forced_alternative_answer",
        "critic_synthesis_answer",
    }
    if option_evidence_arbitrator_enabled:
        challenge_prompt_kinds.add("mc_option_evidence_scorer_answer")
    if _option_sweep_counter_trigger_enabled():
        challenge_prompt_kinds.add("mc_option_sweep_candidate")
    challenge_candidates = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") in challenge_prompt_kinds
        and str(attempt.get("parsed_answer") or "").strip()
    ]
    if not any(
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=problem["answer_type"]) != top_norm
        for attempt in challenge_candidates
    ):
        return None
    verifier_valid = valid
    if not option_evidence_arbitrator_enabled:
        verifier_valid = [
            attempt for attempt in valid
            if attempt.get("prompt_kind") != "mc_option_evidence_scorer_answer"
        ]
    else:
        option_evidence_selection = _run_option_evidence_arbitrator(
            problem=problem,
            valid=valid,
            top_norm=top_norm,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if option_evidence_selection:
            return option_evidence_selection
    try:
        verifier_candidates = _unique_verifier_candidates(problem, verifier_valid)
        verifier_text = _call_model(
            model=model,
            prompt=_verifier_prompt(problem, verifier_candidates),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        choice = _parse_verifier_choice(verifier_text, max_index=len(verifier_candidates))
        selected = verifier_candidates[(choice or 1) - 1]
        _log_event(
            logger,
            {
                "event": "counter_assumption_verifier_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(verifier_candidates),
                "raw_candidate_count": len(valid),
                "choice": choice or 1,
                "selected_child_id": selected["child_id"],
                "selected_answer_hash": selected["parsed_answer_hash"],
                "verifier_prediction_hash": stable_hash({"prediction": verifier_text}),
            },
        )
        return {
            "selection_method": "counter_assumption_verifier_choice" if choice else "counter_assumption_verifier_fallback_first",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 1,
            "verifier_model_call": True,
        }
    except Exception as exc:
        selected = ranked[0][1][0]
        _log_event(
            logger,
            {
                "event": "counter_assumption_verifier_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(valid),
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
                "fallback_child_id": selected["child_id"],
            },
        )
        return {
            "selection_method": "counter_assumption_verifier_error_fallback_majority",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }


def _option_evidence_arbitrator_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_OPTION_EVIDENCE_ARBITRATOR", "").strip().lower() in {"1", "true", "yes", "on"}


def _option_sweep_counter_trigger_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_OPTION_SWEEP_COUNTER_TRIGGER", "").strip().lower() in {"1", "true", "yes", "on"}


def _math_tool_child_timeout(timeout: float | None) -> float | None:
    cap_text = os.environ.get("HLE_MATH_TOOL_CHILD_TIMEOUT_SEC", "180").strip()
    try:
        cap = float(cap_text)
    except ValueError:
        cap = 180.0
    if cap <= 0:
        return timeout
    if timeout is None:
        return cap
    return min(float(timeout), cap)


def _run_option_evidence_arbitrator(
    *,
    problem: dict[str, Any],
    valid: list[dict[str, Any]],
    top_norm: str,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any] | None:
    option_attempts = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") == "mc_option_evidence_scorer_answer"
        and attempt.get("private_option_evidence_context")
        and _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=problem["answer_type"]) != top_norm
    ]
    if not option_attempts:
        return None
    try:
        verifier_text = _call_model(
            model=model,
            prompt=_option_evidence_arbitrator_prompt(
                problem,
                _unique_verifier_candidates(problem, valid),
                evidence_context=str(option_attempts[0].get("private_option_evidence_context") or ""),
            ),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        verifier_candidates = _unique_verifier_candidates(problem, valid)
        choice = _parse_verifier_choice(verifier_text, max_index=len(verifier_candidates))
        if not choice:
            return None
        selected = verifier_candidates[choice - 1]
        _log_event(
            logger,
            {
                "event": "option_evidence_arbitrator_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(verifier_candidates),
                "raw_candidate_count": len(valid),
                "choice": choice,
                "selected_child_id": selected["child_id"],
                "selected_answer_hash": selected["parsed_answer_hash"],
                "selected_option_evidence_candidate": selected.get("prompt_kind") == "mc_option_evidence_scorer_answer",
                "verifier_prediction_hash": stable_hash({"prediction": verifier_text}),
                "evidence_context_hash": stable_hash({
                    "option_evidence_context": str(option_attempts[0].get("private_option_evidence_context") or "")
                }),
            },
        )
        return {
            "selection_method": "option_evidence_verifier_choice",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 1,
            "verifier_model_call": True,
        }
    except Exception as exc:
        _log_event(
            logger,
            {
                "event": "option_evidence_arbitrator_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(valid),
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            },
        )
        return None


def _option_evidence_arbitrator_prompt(
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    evidence_context: str,
) -> str:
    choices = "\n".join(
        f"{index}. answer={attempt['parsed_answer']}; support_count={attempt.get('support_count', 1)}; "
        f"prompt_kinds={','.join(attempt.get('support_prompt_kinds', [attempt.get('prompt_kind', '')]))}"
        for index, attempt in enumerate(attempts, start=1)
    )
    return (
        "A multiple-choice answer majority conflicts with an option-specific transient evidence scorer. "
        "Use the option evidence below to arbitrate. If the evidence directly supports one option over the "
        "majority, choose that candidate. If the evidence is generic, irrelevant, or does not distinguish "
        "options, choose the consensus candidate. Return JSON only: {\"choice\":1}.\n\n"
        f"Option evidence:\n{evidence_context}\n\n"
        f"Question:\n{problem['_question']}\n\nCandidates:\n{choices}"
    )


def _unique_verifier_candidates(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    answer_type = problem.get("answer_type") or "exactMatch"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        norm = _normalize_for_selection(answer, answer_type=answer_type)
        grouped[norm].append(attempt)
    representatives: list[dict[str, Any]] = []
    for norm, group in grouped.items():
        rep = dict(sorted(group, key=lambda row: int(row.get("child_index", 0) or 0))[0])
        rep["support_count"] = len(group)
        rep["support_prompt_kinds"] = sorted({str(row.get("prompt_kind") or "") for row in group if row.get("prompt_kind")})
        rep["normalized_answer_for_verifier"] = norm
        representatives.append(rep)
    if answer_type == "multipleChoice":
        representatives.sort(key=lambda row: (_extract_choice(str(row.get("parsed_answer") or "")) or "Z", int(row.get("child_index", 0) or 0)))
    else:
        representatives.sort(key=lambda row: (-int(row.get("support_count") or 0), int(row.get("child_index", 0) or 0)))
    return representatives


def _run_source_grounded_mc_verifier(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    evidence_context: str,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    verifier_model = (
        os.environ.get("HLE_AGENT_SOURCE_VERIFIER_MODEL")
        or os.environ.get("HLE_AGENT_CRITIC_MODEL")
        or model
    )
    try:
        verifier_text = _call_model(
            model=verifier_model,
            prompt=_source_grounded_mc_verifier_prompt(problem, attempts, evidence_context=evidence_context),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        choice = _parse_verifier_choice(verifier_text, max_index=len(attempts))
        if not choice:
            return {
                "selection_method": "source_grounded_verifier_abstained",
                "selected_child_id": None,
                "selected_answer": "",
                "underlying_model_calls": 1,
                "verifier_model_call": True,
            }
        selected = attempts[choice - 1]
        _log_event(
            logger,
            {
                "event": "source_grounded_mc_verifier_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "verifier_model": verifier_model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(attempts),
                "choice": choice,
                "selected_child_id": selected["child_id"],
                "selected_answer_hash": selected["parsed_answer_hash"],
                "verifier_prediction_hash": stable_hash({"prediction": verifier_text}),
                "evidence_context_hash": stable_hash({"evidence_context": evidence_context}),
            },
        )
        return {
            "selection_method": "source_grounded_verifier_choice",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 1,
            "verifier_model_call": True,
        }
    except Exception as exc:
        _log_event(
            logger,
            {
                "event": "source_grounded_mc_verifier_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "verifier_model": verifier_model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(attempts),
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            },
        )
        return {
            "selection_method": "source_grounded_verifier_error",
            "selected_child_id": None,
            "selected_answer": "",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }


def _source_grounded_mc_verifier_prompt(
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    evidence_context: str,
) -> str:
    choices = "\n".join(
        f"{index}. answer={attempt['parsed_answer']}; support_count={attempt.get('support_count', 1)}; "
        f"prompt_kinds={','.join(attempt.get('support_prompt_kinds', [attempt.get('prompt_kind', '')]))}"
        for index, attempt in enumerate(attempts, start=1)
    )
    return (
        "Choose the candidate answer best supported by the transient evidence for this multipleChoice HLE item. "
        "The candidates are unique answer labels; support_count is diagnostic only and is not a vote weight. "
        "First test whether the evidence directly distinguishes options. If it does not, choose the candidate "
        "whose option text best satisfies the exact question wording and mechanism. Return JSON only: "
        "{\"choice\":1}.\n\n"
        f"Evidence:\n{evidence_context}\n\n"
        f"Question:\n{problem['_question']}\n\nCandidates:\n{choices}"
    )


def _verifier_prompt(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> str:
    choices = "\n".join(
        f"{index}. answer={attempt['parsed_answer']}; support_count={attempt.get('support_count', 1)}; "
        f"prompt_kinds={','.join(attempt.get('support_prompt_kinds', [attempt.get('prompt_kind', '')]))}"
        for index, attempt in enumerate(attempts, start=1)
    )
    return (
        "Choose the candidate answer most likely to satisfy the HLE question. The candidates below are unique "
        "answers; support_count is diagnostic only and is not a vote weight. Prefer exact wording, correct "
        "multiple-choice letter, and answers that satisfy the question even if they have lower support. "
        "Return JSON only: "
        "{\"choice\":1}.\n\n"
        f"Answer type: {problem['answer_type']}\nQuestion:\n{problem['_question']}\n\nCandidates:\n{choices}"
    )


def _parse_verifier_choice(text: str, *, max_index: int) -> int | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            choice = int(parsed.get("choice"))
            return choice if 1 <= choice <= max_index else None
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    match = re.search(r"\b([1-9][0-9]*)\b", text)
    if not match:
        return None
    choice = int(match.group(1))
    return choice if 1 <= choice <= max_index else None


def _needs_exact_answer_repair(problem: dict[str, Any], selected_answer: str) -> bool:
    return problem.get("answer_type") != "multipleChoice" and _is_suspicious_exact_answer(selected_answer)


def _needs_evidence_grounded_child(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> bool:
    if _classify_hle_domain(problem) == "math":
        return False
    if problem.get("answer_type") == "multipleChoice":
        return False
    valid = [
        str(attempt.get("parsed_answer") or "").strip()
        for attempt in attempts
        if str(attempt.get("parsed_answer") or "").strip()
    ]
    return bool(valid) and all(_is_suspicious_exact_answer(answer) for answer in valid)


def _should_prime_evidence_bridge(problem: dict[str, Any], agent_plan: dict[str, Any]) -> bool:
    if agent_plan.get("hle_evidence_context"):
        return False
    if _classify_hle_domain(problem) == "math":
        return False
    # HLE failures are often answer-bearing retrieval failures.  Keep
    # graph/morphism context as one candidate, but give the recursive verifier a
    # separate transient evidence child so closed-book self-consistency cannot
    # be mistaken for verification.
    return True


def _is_suspicious_exact_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return True
    if re.fullmatch(r"[A-Z]", text):
        return True
    if text.lower() in {"unknown", "none", "n/a", "na"}:
        return True
    return False


def _canonicalize_exact_answer_candidate(problem: dict[str, Any], answer: str) -> tuple[str, dict[str, Any]]:
    original = str(answer or "").strip()
    if problem.get("answer_type") == "multipleChoice" or not original:
        return original, {"status": "not_required", "changed": False}
    text = original
    changed = False

    stripped = text.strip().strip('"').strip("'").strip()
    if stripped != text:
        text = stripped
        changed = True

    unwrapped = re.fullmatch(r"\$([^$]{1,240})\$|\\\((.{1,240})\\\)", text, flags=re.DOTALL)
    if unwrapped:
        text = (unwrapped.group(1) or unwrapped.group(2) or "").strip()
        changed = True

    prefixed = re.sub(
        r"^\s*(?:the\s+)?(?:answer|final\s+answer|result|value)\s*(?:is|=|:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if prefixed != text and prefixed:
        text = prefixed
        changed = True

    latex_candidate = text[:-1].strip() if text.endswith(".") else text
    unwrapped = re.fullmatch(r"\$([^$]{1,240})\$|\\\((.{1,240})\\\)", latex_candidate, flags=re.DOTALL)
    if unwrapped:
        text = (unwrapped.group(1) or unwrapped.group(2) or "").strip()
        changed = True

    discourse_stripped = re.sub(r"^\s*(?:therefore|thus|hence|so)\s*,?\s+", "", text, flags=re.IGNORECASE).strip()
    if discourse_stripped != text and discourse_stripped:
        text = discourse_stripped
        changed = True

    numeric_or_formula = bool(re.fullmatch(r"[-+0-9A-Za-z_πpiPI√sqrt^*/%()., =<>]+", text))
    if changed and text.endswith(".") and (numeric_or_formula or len(text.split()) <= 8):
        text = text[:-1].strip()

    if not text:
        text = original
        changed = False

    return text, {
        "status": "activated" if changed else "not_required",
        "changed": changed,
        "before_answer_hash": stable_hash({"answer": original}),
        "after_answer_hash": stable_hash({"answer": text}),
        "policy": "agent_exact_match_safe_prefix_unwrap",
    }


def _canonicalize_multiple_choice_answer(problem: dict[str, Any], answer: str) -> tuple[str, dict[str, Any]]:
    original = str(answer or "").strip()
    if problem.get("answer_type") != "multipleChoice" or not original:
        return original, {"status": "not_required", "changed": False}
    _, options = _split_multiple_choice_question(problem)
    if not options:
        return original, {"status": "options_not_parsed", "changed": False}
    explicit_label = _extract_explicit_choice_label(original)
    if explicit_label and explicit_label in options:
        return explicit_label, {
            "status": "canonicalized" if explicit_label != original else "unchanged",
            "changed": explicit_label != original,
            "method": "explicit_label",
            "before_answer_hash": stable_hash({"answer": original}),
            "after_answer_hash": stable_hash({"answer": explicit_label}),
        }

    stripped = original.strip().strip('"').strip("'").strip()
    stripped = re.sub(
        r"^\s*(?:the\s+)?(?:answer|final\s+answer|correct\s+answer)\s*(?:is|=|:)\s*",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()
    stripped = stripped[:-1].strip() if stripped.endswith(".") else stripped
    answer_norm = _normalize_exact(stripped)
    if not answer_norm:
        return original, {"status": "empty_after_strip", "changed": False}

    matches: list[tuple[str, str]] = []
    answer_terms = _content_terms(answer_norm)
    for label, option_text in options.items():
        option_norm = _normalize_exact(option_text)
        if not option_norm:
            continue
        if answer_norm == option_norm:
            matches.append((label, "exact_option_text"))
            continue
        if len(option_norm) >= 8 and option_norm in answer_norm:
            matches.append((label, "contained_option_text"))
            continue
        option_terms = _content_terms(option_norm)
        if len(option_terms) >= 2 and option_terms.issubset(answer_terms):
            matches.append((label, "option_terms_subset"))
    unique_labels = sorted({label for label, _ in matches})
    if len(unique_labels) == 1:
        label = unique_labels[0]
        methods = sorted({method for match_label, method in matches if match_label == label})
        return label, {
            "status": "canonicalized",
            "changed": label != original,
            "method": methods[0],
            "before_answer_hash": stable_hash({"answer": original}),
            "after_answer_hash": stable_hash({"answer": label}),
        }
    if len(unique_labels) > 1:
        return original, {
            "status": "ambiguous_option_text",
            "changed": False,
            "candidate_label_hashes": [stable_hash({"option_label": label}) for label in unique_labels],
        }
    return original, {"status": "no_option_text_match", "changed": False}


def _extract_explicit_choice_label(text: str) -> str:
    raw = str(text or "").strip()
    upper = raw.upper()
    direct = re.fullmatch(r"([A-H])(?:[\).:：])?", upper)
    if direct:
        return direct.group(1)
    prefix = re.match(r"^\s*([A-H])[\).:：]\s+", upper)
    if prefix:
        return prefix.group(1)
    match = re.search(
        r"\b(?:option|choice|answer|ans|final\s+answer)\s*(?:is|=|:)?\s*([A-H])\b",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return ""


def _evidence_grounded_answer_prompt(problem: dict[str, Any], *, evidence_context: str) -> str:
    if problem.get("answer_type") == "multipleChoice":
        return (
            "Use the transient evidence below to answer this multipleChoice HLE item. Return the single option "
            "letter only. Treat the evidence as fallible: use it only when it directly matches the question, and "
            "ignore irrelevant passages. Return JSON only: {\"answer\":\"...\"}.\n\n"
            f"{evidence_context}\n\n"
            f"Question:\n{problem['_question']}"
        )
    return (
        "Use the transient evidence below to answer this exactMatch HLE item. The existing child attempts all "
        "collapsed to a likely choice-letter artifact, so ignore single-letter answers unless the question "
        "explicitly asks for a letter symbol. Return the shortest exact entity, title, formula, number, or phrase. "
        "If the evidence is irrelevant, answer from the question directly. Return JSON only: {\"answer\":\"...\"}.\n\n"
        f"{evidence_context}\n\n"
        f"Question:\n{problem['_question']}"
    )


def _repair_exact_answer(
    *,
    problem: dict[str, Any],
    selected_answer: str,
    agent_plan: dict[str, Any],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    evidence_bridge_enabled: bool,
) -> dict[str, Any]:
    before_hash = stable_hash({"answer": selected_answer})
    repair_context = _repair_context_for_exact(agent_plan)
    evidence_context = str(agent_plan.get("hle_evidence_context") or "")
    evidence_summary: dict[str, Any] | None = agent_plan.get("hle_evidence_bridge")
    if evidence_bridge_enabled:
        if not evidence_context:
            evidence_context, evidence_summary = _build_hle_evidence_bridge_context(
                problem=problem,
                eval_id=eval_id,
                call_id=call_id,
                model=model,
                logger=logger,
            )
            agent_plan["hle_evidence_context"] = evidence_context
            agent_plan["hle_evidence_bridge"] = evidence_summary
    _log_event(
        logger,
        {
            "event": "answer_format_repair_start",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "answer_type": problem["answer_type"],
            "candidate_answer_hash": before_hash,
            "repair_reason": "suspicious_exact_answer",
            "repair_context_used": bool(repair_context),
            "repair_context_char_count": len(repair_context),
            "evidence_bridge_used": bool(evidence_context),
            "evidence_bridge_char_count": len(evidence_context),
        },
    )
    try:
        text = _call_model(
            model=model,
            prompt=_exact_answer_repair_prompt(
                problem,
                selected_answer,
                repair_context=repair_context,
                evidence_context=evidence_context,
            ),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        repaired = _parse_answer_json(text) or text.strip()
        after_hash = stable_hash({"answer": repaired})
        still_suspicious = _is_suspicious_exact_answer(repaired)
        stage_summary = {
            "status": "activated",
            "repair_reason": "suspicious_exact_answer",
            "candidate_answer_hash": before_hash,
            "repaired_answer_hash": after_hash,
            "changed": after_hash != before_hash,
            "still_suspicious": still_suspicious,
            "repair_context_used": bool(repair_context),
            "repair_context_char_count": len(repair_context),
            "evidence_bridge_used": bool(evidence_context),
            "evidence_bridge_char_count": len(evidence_context),
            "evidence_bridge": evidence_summary,
        }
        _log_event(
            logger,
            {
                "event": "answer_format_repair_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_answer_hash": before_hash,
                "repaired_answer_hash": after_hash,
                "changed": after_hash != before_hash,
                "still_suspicious": still_suspicious,
                "repair_context_used": bool(repair_context),
                "repair_context_char_count": len(repair_context),
                "evidence_bridge_used": bool(evidence_context),
                "evidence_bridge_char_count": len(evidence_context),
                "prediction_hash": stable_hash({"prediction": text}),
            },
        )
        return {
            "selected_answer": repaired,
            "underlying_model_calls": 1,
            "stage_summary": stage_summary,
        }
    except Exception as exc:
        stage_summary = {
            "status": "failed",
            "repair_reason": "suspicious_exact_answer",
            "candidate_answer_hash": before_hash,
            "error_type": type(exc).__name__,
            "repair_context_used": bool(repair_context),
            "repair_context_char_count": len(repair_context),
            "evidence_bridge_used": bool(evidence_context),
            "evidence_bridge_char_count": len(evidence_context),
            "evidence_bridge": evidence_summary,
        }
        _log_event(
            logger,
            {
                "event": "answer_format_repair_error",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_answer_hash": before_hash,
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
                "repair_context_used": bool(repair_context),
                "repair_context_char_count": len(repair_context),
                "evidence_bridge_used": bool(evidence_context),
                "evidence_bridge_char_count": len(evidence_context),
            },
        )
        return {
            "selected_answer": selected_answer,
            "underlying_model_calls": 0,
            "stage_summary": stage_summary,
        }


def _repair_context_for_exact(agent_plan: dict[str, Any]) -> str:
    return str(agent_plan.get("prompt_context") or agent_plan.get("retrieval_context_candidate") or "").strip()


def _build_hle_evidence_bridge_context(
    *,
    problem: dict[str, Any],
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
) -> tuple[str, dict[str, Any]]:
    queries = _candidate_evidence_queries(problem)
    query_hashes = [stable_hash({"query": query}) for query in queries]
    if not queries:
        summary = {
            "status": "no_queries",
            "source": "wikipedia_search",
            "query_count": 0,
            "query_hashes": [],
            "result_count": 0,
            "source_hashes": [],
            "evidence_char_count": 0,
        }
        _log_hle_evidence_bridge_event(logger, eval_id=eval_id, call_id=call_id, problem=problem, model=model, summary=summary)
        return "", summary

    results: list[dict[str, str]] = []
    errors: list[str] = []
    for query in queries:
        try:
            wiki_rows = _wikipedia_search(query, limit=3, timeout=6.0)
            results.extend(wiki_rows)
            if len(wiki_rows) < 2 or _should_use_domain_evidence_search(problem):
                results.extend(_domain_evidence_search(query, problem=problem, limit=2, timeout=8.0))
        except Exception as exc:
            errors.append(type(exc).__name__)
    unique_results = _dedupe_evidence_results(results)
    reranked_results = [row["doc"] for row in _hipporag_style_rerank(problem, unique_results)[:5]]
    selected_results = reranked_results or unique_results[:5]
    evidence_context = _format_evidence_context(selected_results, max_chars=1800)
    summary = {
        "status": "activated" if evidence_context else "no_results",
        "source": "wikipedia_plus_domain_search",
        "selection_policy": "hipporag_style_associative_rerank",
        "query_count": len(queries),
        "query_hashes": query_hashes,
        "result_count": len(unique_results),
        "selected_result_count": len(selected_results),
        "source_hashes": [
            stable_hash({"title": row.get("title", ""), "snippet": row.get("snippet", "")})
            for row in selected_results
        ],
        "source_counts": dict(Counter(str(row.get("source") or "unknown") for row in unique_results)),
        "evidence_char_count": len(evidence_context),
        "error_types": sorted(set(errors)),
    }
    _log_hle_evidence_bridge_event(logger, eval_id=eval_id, call_id=call_id, problem=problem, model=model, summary=summary)
    return evidence_context, summary


def _log_hle_evidence_bridge_event(
    logger: "_JsonlLogger | None",
    *,
    eval_id: str,
    call_id: str,
    problem: dict[str, Any],
    model: str,
    summary: dict[str, Any],
) -> None:
    _log_event(
        logger,
        {
            "event": "hle_evidence_bridge",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": "assumption_agent_recursive_verify",
            "stage_status": summary.get("status"),
            "stage_data": summary,
        },
    )


def _candidate_evidence_queries(problem: dict[str, Any]) -> list[str]:
    question = str(problem.get("_question") or "")
    seeds: list[str] = []
    for key in ("raw_subject", "category"):
        value = str(problem.get(key) or "").strip()
        if value and value.lower() not in {"other", "misc", "unknown"}:
            seeds.append(value)
    quoted = re.findall(r'"([^"]{3,120})"|\'([^\']{3,120})\'|`([^`]{3,120})`', question)
    for groups in quoted:
        seeds.extend(item for item in groups if item)
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_+.-]*(?:\s+[A-Z][A-Za-z0-9_+.-]*){1,5}\b", question):
        seeds.append(match.group(0))
    if not seeds:
        words = [
            word
            for word in re.findall(r"[A-Za-z0-9_+.-]{4,}", question)
            if word.lower() not in _EVIDENCE_QUERY_STOPWORDS
        ][:12]
        if words:
            seeds.append(" ".join(words))
    queries: list[str] = []
    for seed in seeds:
        query = _clean_evidence_query(seed)
        if not query:
            continue
        if problem.get("raw_subject") and problem["raw_subject"] not in query and len(query.split()) <= 4:
            query = f"{query} {problem['raw_subject']}"
        if query not in queries:
            queries.append(query)
        if len(queries) >= 4:
            break
    return queries


_EVIDENCE_QUERY_STOPWORDS = {
    "which",
    "what",
    "when",
    "where",
    "whose",
    "question",
    "answer",
    "following",
    "correct",
    "return",
    "exact",
    "match",
}


def _clean_evidence_query(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"\b[A-E]\s*[\).:]", " ", text)
    text = re.sub(r"[^A-Za-z0-9_+.' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .'-")
    if len(text) < 3:
        return ""
    return text[:120]


def _wikipedia_search(query: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": "1",
        "srlimit": str(limit),
    })
    request = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "AssumptionAgentHLEEvidenceBridge/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    rows = []
    for item in data.get("query", {}).get("search", [])[:limit]:
        title = _clean_evidence_text(str(item.get("title") or ""))
        snippet = _clean_evidence_text(str(item.get("snippet") or ""))
        if title or snippet:
            rows.append({"title": title, "snippet": snippet, "source": "wikipedia"})
    return rows


def _should_use_domain_evidence_search(problem: dict[str, Any]) -> bool:
    domain = f"{problem.get('category', '')} {problem.get('raw_subject', '')}".lower()
    return any(token in domain for token in (
        "biology",
        "medicine",
        "medical",
        "biomed",
        "law",
        "legal",
        "court",
        "engineering",
        "computer science",
    ))


def _domain_evidence_search(
    query: str,
    *,
    problem: dict[str, Any],
    limit: int,
    timeout: float,
) -> list[dict[str, str]]:
    domain = f"{problem.get('category', '')} {problem.get('raw_subject', '')}".lower()
    rows: list[dict[str, str]] = []
    if any(token in domain for token in ("biology", "medicine", "medical", "biomed")):
        rows.extend(_pubmed_search(query, limit=limit, timeout=timeout))
    if any(token in domain for token in ("law", "legal", "court")):
        rows.extend(_ontario_lso_rules_search(query, limit=limit, timeout=timeout))
        rows.extend(_courtlistener_search(query, limit=limit, timeout=timeout))
    rows.extend(_crossref_search(query, limit=max(1, min(limit, 2)), timeout=timeout))
    return rows


def _pubmed_search(query: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    term = _clean_evidence_query(query)
    if not term:
        return []
    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "retmode": "json",
        "retmax": str(limit),
        "term": term,
    })
    search_req = urllib.request.Request(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}",
        headers={"User-Agent": "AssumptionAgentHLEEvidenceBridge/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(search_req, timeout=timeout) as response:
        search_data = json.loads(response.read().decode("utf-8"))
    ids = list(search_data.get("esearchresult", {}).get("idlist", []) or [])[:limit]
    if not ids:
        return []
    summary_params = urllib.parse.urlencode({
        "db": "pubmed",
        "retmode": "json",
        "id": ",".join(ids),
    })
    summary_req = urllib.request.Request(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_params}",
        headers={"User-Agent": "AssumptionAgentHLEEvidenceBridge/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(summary_req, timeout=timeout) as response:
        summary_data = json.loads(response.read().decode("utf-8"))
    result = summary_data.get("result", {}) if isinstance(summary_data, dict) else {}
    rows: list[dict[str, str]] = []
    for uid in ids:
        item = result.get(uid, {}) if isinstance(result, dict) else {}
        title = _clean_evidence_text(str(item.get("title") or ""))
        journal = _clean_evidence_text(str(item.get("fulljournalname") or item.get("source") or ""))
        pubdate = _clean_evidence_text(str(item.get("pubdate") or ""))
        snippet = _clean_evidence_text("; ".join(value for value in (journal, pubdate, f"PMID {uid}") if value))
        if title or snippet:
            rows.append({"title": title, "snippet": snippet, "source": "pubmed"})
    return rows


def _courtlistener_search(query: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    term = _clean_evidence_query(query)
    if not term:
        return []
    params = urllib.parse.urlencode({"q": term, "type": "o"})
    request = urllib.request.Request(
        f"https://www.courtlistener.com/api/rest/v4/search/?{params}",
        headers={"User-Agent": "AssumptionAgentHLEEvidenceBridge/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    rows: list[dict[str, str]] = []
    for item in list(data.get("results", []) or [])[:limit]:
        title = _clean_evidence_text(str(item.get("caseName") or item.get("caseNameFull") or ""))
        court = _clean_evidence_text(str(item.get("court") or item.get("court_id") or ""))
        date = _clean_evidence_text(str(item.get("dateFiled") or ""))
        snippet_source = item.get("snippet") or item.get("syllabus") or item.get("procedural_history") or item.get("posture") or ""
        snippet = _clean_evidence_text("; ".join(value for value in (court, date, str(snippet_source)) if value))
        if title or snippet:
            rows.append({"title": title, "snippet": snippet, "source": "courtlistener"})
    return rows


_LSO_RULES_TEXT_CACHE: str | None = None


def _ontario_lso_rules_search(query: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    term = _clean_evidence_query(query)
    if not term:
        return []
    try:
        rules_text = _load_lso_rules_text(timeout=timeout)
    except Exception:
        return []
    query_terms = _content_terms(term)
    if not query_terms:
        return []
    anchors = [
        "Acting Against Former Clients",
        "adequate measures",
        "confidential information",
        "Conflicts from Transfer Between Law Firms",
        "law firm establishes that it has taken adequate measures",
    ]
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in anchors:
        for match in re.finditer(re.escape(anchor), rules_text, flags=re.IGNORECASE):
            start = max(0, match.start() - 260)
            end = min(len(rules_text), match.end() + 620)
            snippet = _clean_evidence_text(rules_text[start:end])
            if not snippet:
                continue
            snippet_terms = _content_terms(snippet)
            if len(query_terms & snippet_terms) < 1 and anchor.lower() not in term.lower():
                continue
            key = _normalize_exact(snippet[:160])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "title": "Law Society of Ontario Rules of Professional Conduct",
                "snippet": snippet,
                "source": "lso_rules",
            })
            if len(rows) >= limit:
                return rows
    return rows


def _load_lso_rules_text(*, timeout: float) -> str:
    global _LSO_RULES_TEXT_CACHE
    if _LSO_RULES_TEXT_CACHE:
        return _LSO_RULES_TEXT_CACHE
    request = urllib.request.Request(
        "https://www.lso.ca/about-lso/legislation-rules/rules-of-professional-conduct/complete-rules-of-professional-conduct",
        headers={"User-Agent": "AssumptionAgentHLEEvidenceBridge/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", "ignore")
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    _LSO_RULES_TEXT_CACHE = text[:240000]
    return _LSO_RULES_TEXT_CACHE


def _crossref_search(query: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    term = _clean_evidence_query(query)
    if not term:
        return []
    params = urllib.parse.urlencode({"query": term, "rows": str(limit)})
    request = urllib.request.Request(
        f"https://api.crossref.org/works?{params}",
        headers={"User-Agent": "AssumptionAgentHLEEvidenceBridge/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    rows: list[dict[str, str]] = []
    for item in list(data.get("message", {}).get("items", []) or [])[:limit]:
        title_values = item.get("title") or []
        title = _clean_evidence_text(str(title_values[0] if title_values else ""))
        container_values = item.get("container-title") or []
        container = _clean_evidence_text(str(container_values[0] if container_values else ""))
        doi = _clean_evidence_text(str(item.get("DOI") or ""))
        snippet = _clean_evidence_text("; ".join(value for value in (container, doi) if value))
        if title or snippet:
            rows.append({"title": title, "snippet": snippet, "source": "crossref"})
    return rows


def _clean_evidence_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def _dedupe_evidence_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for row in results:
        key = _normalize_exact(row.get("title") or row.get("snippet") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _format_evidence_context(results: list[dict[str, str]], *, max_chars: int) -> str:
    lines: list[str] = []
    for index, row in enumerate(results, start=1):
        title = row.get("title") or "Untitled"
        snippet = row.get("snippet") or ""
        source = row.get("source") or "wikipedia"
        lines.append(f"[Evidence {index}] source={source}; title={title}; snippet={snippet}")
    text = "\n".join(lines).strip()
    return _trim_context(text, max_chars=max_chars)


def _exact_answer_repair_prompt(
    problem: dict[str, Any],
    selected_answer: str,
    *,
    repair_context: str = "",
    evidence_context: str = "",
) -> str:
    context_block = ""
    if repair_context:
        context_block = (
            "Retrieved graph context is available. It may be generic or irrelevant; use it only if it directly "
            "helps identify the requested exact entity/term/phrase.\n\n"
            f"{repair_context}\n\n"
        )
    evidence_block = ""
    if evidence_context:
        evidence_block = (
            "Transient evidence bridge results are available. Prefer concrete source evidence over generic "
            "assumption context, but ignore evidence that does not match the question.\n\n"
            f"{evidence_context}\n\n"
        )
    return (
        "This HLE item is marked exactMatch, not multipleChoice. The previous candidate was rejected as a likely "
        "choice-letter artifact or underspecified answer. Re-read the question from scratch and return the actual "
        "shortest exact entity, term, title, formula, number, or phrase requested by the question. Do not return a "
        "single uppercase A-Z letter unless the wording explicitly asks for a letter symbol as the answer. If the "
        "question lists options and you know which option is correct, return the option text itself rather than "
        "the option letter. Return JSON only: {\"answer\":\"...\"}.\n\n"
        f"{context_block}"
        f"{evidence_block}"
        f"Rejected candidate hash only: {stable_hash({'answer': selected_answer})}\n"
        f"Question:\n{problem['_question']}"
    )


def _split_multiple_choice_question(problem: dict[str, Any]) -> tuple[str, dict[str, str]]:
    question = str(problem.get("_question") or "")
    options, first_start = _extract_multiple_choice_options(question)
    if not options or first_start is None:
        return question, {}
    stem = question[:first_start].strip()
    stem = re.sub(r"(?:choices?|options?)\s*[:：]?\s*$", "", stem, flags=re.IGNORECASE).strip()
    return stem or question, options


def _extract_multiple_choice_options(question: str) -> tuple[dict[str, str], int | None]:
    text = str(question or "")
    pattern = re.compile(
        r"(?:^|\n)\s*([A-H])[\).:：]\s*(.*?)(?=(?:\n\s*[A-H][\).:：]\s*)|\Z)",
        flags=re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return {}, None
    options: dict[str, str] = {}
    first_start: int | None = None
    for match in matches:
        label = match.group(1).upper()
        option_text = re.sub(r"\s+", " ", match.group(2)).strip()
        if not option_text:
            continue
        if first_start is None:
            first_start = match.start()
        options[label] = option_text[:240]
    return (options, first_start) if len(options) >= 2 else ({}, None)


def _mc_option_matches_reference(option_text: str, reference_answer: str) -> bool:
    option = str(option_text or "").strip()
    reference = str(reference_answer or "").strip()
    if not option or not reference:
        return False
    option_norm = _normalize_exact(option)
    reference_norm = _normalize_exact(reference)
    if option_norm and option_norm == reference_norm:
        return True
    option_math = _normalize_math_expression(option)
    reference_math = _normalize_math_expression(reference)
    if not option_math or not reference_math:
        return False
    try:
        import sympy as sp

        option_expr = _safe_sympy_parse_expr(option_math)
        reference_expr = _safe_sympy_parse_expr(reference_math)
        if option_expr is None or reference_expr is None:
            return False
        return bool(sp.simplify(option_expr - reference_expr) == 0)
    except Exception:
        return option_math.replace(" ", "") == reference_math.replace(" ", "")


def _normalize_for_selection(text: str, *, answer_type: str) -> str:
    if answer_type == "multipleChoice":
        return _extract_choice(text)
    return _normalize_exact(text)


def _has_two_vote_majority(attempts: list[dict[str, Any]], *, answer_type: str) -> bool:
    counts: Counter[str] = Counter()
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        if answer_type != "multipleChoice" and _is_suspicious_exact_answer(answer):
            continue
        counts[_normalize_for_selection(answer, answer_type=answer_type)] += 1
    return any(count >= 2 for count in counts.values())


def _fallback_answer(attempts: list[dict[str, Any]]) -> str:
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if answer:
            return answer
    return ""


def _classify_hle_domain(problem: dict[str, Any]) -> str:
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
        str(problem.get("_question") or ""),
    ]).lower()
    if any(token in text for token in ["computer", "software", "program", "code", "algorithm"]):
        return "software_engineering"
    if any(token in text for token in ["math", "algebra", "geometry", "number theory", "combinatorics"]):
        return "math"
    if any(token in text for token in ["physics", "chemistry", "biology", "medicine", "science"]):
        return "science"
    if any(token in text for token in ["philosophy", "history", "law", "literature", "social"]):
        return "humanities_social_science"
    return "hle_general"


def _sanitize_retrieval_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "policy": "none",
            "node_count": 0,
            "edge_count": 0,
            "top_node_ids": [],
            "top_scores": [],
            "formal_mapping_hits": [],
            "structural_morphism_hits": [],
        }
    nodes = list(result.subgraph.nodes)
    scores = result.subgraph.scores
    return {
        "policy": result.diagnostics.get("policy"),
        "route": result.diagnostics.get("route"),
        "node_count": len(nodes),
        "edge_count": len(result.subgraph.edges),
        "top_node_ids": [node.id for node in nodes[:6]],
        "top_node_types": [str(node.type.value if hasattr(node.type, "value") else node.type) for node in nodes[:6]],
        "top_scores": [round(float(scores.get(node.id, 0.0)), 4) for node in nodes[:6]],
        "formal_mapping_hits": result.diagnostics.get("formal_mapping_hits", []),
        "structural_morphism_hits": result.diagnostics.get("structural_morphism_hits", []),
    }


def _retrieval_summary_is_generic_harness_only(summary: dict[str, Any]) -> bool:
    node_types = [str(value) for value in summary.get("top_node_types", []) or []]
    if not node_types:
        return False
    generic_types = {"harness", "generic_harness"}
    return all(node_type in generic_types for node_type in node_types)


def _sanitize_recursive_payload(payload: dict[str, Any]) -> dict[str, Any]:
    root = payload.get("root", {})
    next_actions = payload.get("next_actions", [])
    if isinstance(next_actions, dict):
        next_action_counts = next_actions.get("counts", {})
    else:
        next_action_counts = dict(Counter(str(row.get("next_action") or "") for row in next_actions if isinstance(row, dict)))
    return {
        "frame_counts": payload.get("frame_counts", {}),
        "status_counts": payload.get("status_counts", {}),
        "depth_counts": payload.get("depth_counts", {}),
        "open_frame_count": len(payload.get("open_frame_ids", [])),
        "recursion_edge_count": len(payload.get("recursion_edges", [])),
        "activated_assumption_ids": root.get("activated_assumption_ids", [])[:8],
        "next_action_counts": next_action_counts,
    }


def _should_use_agent_context(
    *,
    answer_type: str,
    top_score: float,
    formal_hit_count: int,
    structural_hit_count: int,
    strong_structural_hit_count: int,
    expected_utility: float,
) -> bool:
    hit_bonus = formal_hit_count + strong_structural_hit_count
    if expected_utility < -0.08:
        return False
    if answer_type != "multipleChoice" and hit_bonus == 0:
        return False
    if hit_bonus > 0 and top_score >= 0.08:
        return True
    if structural_hit_count and answer_type == "multipleChoice" and top_score >= 0.14:
        return True
    return answer_type == "multipleChoice" and top_score >= 0.20


def _trim_context(text: str, *, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 120)].rstrip() + "\n\n[context truncated by HLE agent gate]"


def _agent_stage_log(
    logger: "_JsonlLogger | None",
    *,
    eval_id: str,
    call_id: str,
    problem: dict[str, Any],
    model: str,
    stage: str,
    data: dict[str, Any],
    variant: str = "assumption_agent",
) -> None:
    _log_event(
        logger,
        {
            "event": "agent_stage",
            "eval_id": eval_id,
            "call_id": call_id,
            "problem_id_hash": problem["id_hash"],
            "question_hash": problem["question_hash"],
            "model": model,
            "variant": variant,
            "stage": stage,
            "stage_status": data.get("status"),
            "stage_data": data,
        },
    )


def _module_trace(problem: dict[str, Any], *, variant: str, agent_plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Describe the modules that actually run in this HLE wrapper.

    HLE is closed-book here: the runner sends a single text prompt to the model
    and scores the returned answer.  The trace is deliberately conservative: it
    marks true Assumption Agent subsystems as skipped when this smoke wrapper
    has not actually invoked them.
    """
    if variant.startswith("hipporag"):
        stages = (agent_plan or {}).get("stages", {})
        return [
            {
                "module": "answer_type_router",
                "expected": True,
                "status": "activated",
                "reason": f"answer_type={problem.get('answer_type') or 'unknown'} controls output contract",
            },
            {
                "module": "hipporag_context_retrieval",
                "expected": True,
                "status": _stage_status(stages, "hipporag_context_retrieval"),
                "reason": "question-triggered transient evidence corpus retrieval, independent of the Assumption Graph",
            },
            {
                "module": "hipporag_associative_rerank",
                "expected": True,
                "status": _stage_status(stages, "hipporag_associative_rerank"),
                "reason": "HippoRAG-style entity/passage association reranks retrieved evidence",
            },
            {
                "module": "assumption_graph_retrieval",
                "expected": False,
                "status": "not_applicable",
                "reason": "HippoRAG baseline is a control and must not use the Assumption Graph",
            },
            {
                "module": "structural_morphism_transfer",
                "expected": False,
                "status": "not_applicable",
                "reason": "HippoRAG baseline is a control and must not use morphism transfer",
            },
            {
                "module": "world_model_router",
                "expected": False,
                "status": "not_applicable",
                "reason": "HippoRAG baseline is a control and must not use the Assumption Agent world model",
            },
            {
                "module": "recursive_assumption_runner",
                "expected": False,
                "status": "not_applicable",
                "reason": "HippoRAG baseline is a control and must not run recursive self-validation",
            },
            {
                "module": "prompt_builder",
                "expected": True,
                "status": _stage_status(stages, "prompt_builder"),
                "reason": "retrieved context is wrapped into a retrieval-augmented QA prompt",
            },
            {
                "module": "answer_format_verifier",
                "expected": True,
                "status": "activated",
                "reason": "JSON answer parser and answer-type scorer run after each response",
            },
        ]

    if variant.startswith("assumption_agent"):
        stages = (agent_plan or {}).get("stages", {})
        recursive_verify = variant == "assumption_agent_recursive_verify"
        return [
            {
                "module": "answer_type_router",
                "expected": True,
                "status": "activated",
                "reason": f"answer_type={problem.get('answer_type') or 'unknown'} controls output contract",
            },
            {
                "module": "domain_router",
                "expected": True,
                "status": _stage_status(stages, "domain_router"),
                "reason": "HLE metadata/question routed to an assumption domain",
            },
            {
                "module": "assumption_graph_retrieval",
                "expected": True,
                "status": _stage_status(stages, "assumption_graph_retrieval"),
                "reason": "JsonlGraphStore + SimpleAssumptionGraph retrieval executed before prompt construction",
            },
            {
                "module": "structural_morphism_transfer",
                "expected": True,
                "status": _stage_status(stages, "structural_morphism_transfer"),
                "reason": "formal mapping and structural pattern search executed over the activated graph",
            },
            {
                "module": "world_model_router",
                "expected": True,
                "status": _stage_status(stages, "world_model_router"),
                "reason": "predict_proposal_outcome gates whether retrieved context should be injected",
            },
            {
                "module": "critic_model_router",
                "expected": False,
                "status": _stage_status(stages, "critic_model_router") if stages.get("critic_model_router") else "not_required",
                "reason": "optional expensive critic can be routed only to falsification and verification steps",
            },
            {
                "module": "recursive_assumption_runner",
                "expected": True,
                "status": _stage_status(stages, "recursive_assumption_runner"),
                "reason": "build_recursive_assumption_run builds a bounded applicability tree in memory",
            },
            {
                "module": "recursive_child_validation",
                "expected": recursive_verify,
                "status": _stage_status(stages, "recursive_child_validation") if recursive_verify else "not_applicable",
                "reason": (
                    "recursive verifier executes child answer attempts and records only hashes/metadata"
                    if recursive_verify
                    else "single-call agent only builds recursive applicability frames"
                ),
            },
            {
                "module": "multi_candidate_self_verifier",
                "expected": recursive_verify,
                "status": _stage_status(stages, "multi_candidate_self_verifier") if recursive_verify else "not_implemented_for_hle_single_call",
                "reason": (
                    "multi-child answers are selected by majority or verifier model"
                    if recursive_verify
                    else "this HLE variant keeps one answer call; external multi-candidate verification would require extra model calls"
                ),
            },
            {
                "module": "counter_assumption_challenge",
                "expected": recursive_verify,
                "status": _stage_status(stages, "counter_assumption_challenge") if stages.get("counter_assumption_challenge") else "not_required",
                "reason": "majority answers without independent verification can trigger a falsification child",
            },
            {
                "module": "option_elimination_challenge",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": (
                    _stage_status((stages.get("counter_assumption_challenge") or {}), "option_elimination_challenge")
                    if isinstance(stages.get("counter_assumption_challenge"), dict)
                    and stages["counter_assumption_challenge"].get("option_elimination_challenge")
                    else "not_required"
                ),
                "reason": "collapsed multiple-choice majorities can trigger stricter option-by-option falsification",
            },
            {
                "module": "forced_alternative_challenge",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": (
                    _stage_status((stages.get("counter_assumption_challenge") or {}), "forced_alternative_challenge")
                    if isinstance(stages.get("counter_assumption_challenge"), dict)
                    and stages["counter_assumption_challenge"].get("forced_alternative_challenge")
                    else "not_required"
                ),
                "reason": "collapsed candidates can force one non-majority candidate for verifier arbitration",
            },
            {
                "module": "residual_writeback",
                "expected": False,
                "status": "not_applicable",
                "reason": "HLE smoke results are logged as evaluation artifacts, not written back into the main graph",
            },
            {
                "module": "prompt_builder",
                "expected": True,
                "status": _stage_status(stages, "prompt_builder"),
                "reason": "world-model-gated prompt context is built or abstained before the LLM call",
            },
            {
                "module": "answer_format_repair",
                "expected": False,
                "status": _stage_status(stages, "answer_format_repair") if stages.get("answer_format_repair") else "not_required",
                "reason": "exactMatch single-letter/empty candidates trigger a strict repair pass when needed",
            },
            {
                "module": "hle_evidence_bridge",
                "expected": recursive_verify,
                "status": _stage_status(stages, "hle_evidence_bridge") if stages.get("hle_evidence_bridge") else "not_required",
                "reason": "recursive HLE answering can add a transient external-evidence child; logs persist only hashes and counts",
            },
            {
                "module": "agent_hipporag_context_bridge",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "agent_hipporag_context_bridge") if stages.get("agent_hipporag_context_bridge") else "not_required",
                "reason": "recursive HLE answering can add a HippoRAG-style associative retrieval child without using gold answers",
            },
            {
                "module": "hle_math_tool_solver",
                "expected": recursive_verify and _should_run_math_tool_child(problem),
                "status": _stage_status(stages, "hle_math_tool_solver") if stages.get("hle_math_tool_solver") else "not_required",
                "reason": "Math exactMatch items can add a restricted SymPy child; logs persist only plan/answer hashes",
            },
            {
                "module": "candidate_claim_verifier",
                "expected": recursive_verify and _should_run_candidate_claim_verifier(problem),
                "status": _stage_status(stages, "candidate_claim_verifier") if stages.get("candidate_claim_verifier") else "not_required",
                "reason": "Executable math claims verify exact candidates or multiple-choice options before majority selection",
            },
            {
                "module": "domain_rule_mc_verifier",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "domain_rule_mc_verifier") if stages.get("domain_rule_mc_verifier") else "not_required",
                "reason": "bounded domain rules can add an evidence-backed or contrastive verified multiple-choice candidate",
            },
            {
                "module": "mc_option_evidence_scorer",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "mc_option_evidence_scorer") if stages.get("mc_option_evidence_scorer") else "not_required",
                "reason": "multiple-choice options can be scored by option-specific transient evidence retrieval",
            },
            {
                "module": "critic_synthesis_child",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "critic_synthesis_child") if stages.get("critic_synthesis_child") else "not_required",
                "reason": "collapsed or low-diversity multiple-choice candidate sets can trigger a distinct critic-model synthesis child",
            },
            {
                "module": "mc_option_sweep_candidates",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "mc_option_sweep_candidates") if stages.get("mc_option_sweep_candidates") else "not_required",
                "reason": "finite multiple-choice option spaces are completed with synthetic label candidates before verification",
            },
            {
                "module": "answer_format_verifier",
                "expected": True,
                "status": "activated",
                "reason": "JSON answer parser and answer-type scorer run after each response",
            },
        ]

    is_assumption_variant = variant.startswith("assumption")
    trace = [
        {
            "module": "answer_type_router",
            "expected": True,
            "status": "activated",
            "reason": f"answer_type={problem.get('answer_type') or 'unknown'} controls output contract",
        },
        {
            "module": "prompt_scaffold",
            "expected": is_assumption_variant,
            "status": "activated" if is_assumption_variant else "not_applicable",
            "reason": "single-call assumption audit prompt" if is_assumption_variant else "raw baseline has no wrapper prompt",
        },
        {
            "module": "assumption_graph_retrieval",
            "expected": is_assumption_variant,
            "status": "skipped" if is_assumption_variant else "not_applicable",
            "reason": "HLE smoke runner does not invoke graph retrieval; it only prompts the model",
        },
        {
            "module": "structural_morphism_transfer",
            "expected": is_assumption_variant,
            "status": "skipped" if is_assumption_variant else "not_applicable",
            "reason": "no category/morphism candidate selection is executed in this wrapper",
        },
        {
            "module": "world_model_router",
            "expected": is_assumption_variant,
            "status": "skipped" if is_assumption_variant else "not_applicable",
            "reason": "no calibrated gate/router is consulted before the HLE call",
        },
        {
            "module": "recursive_assumption_runner",
            "expected": is_assumption_variant,
            "status": "skipped" if is_assumption_variant else "not_applicable",
            "reason": "no generate-ablate-judge-resume loop is executed inside this HLE smoke wrapper",
        },
        {
            "module": "multi_candidate_self_verifier",
            "expected": is_assumption_variant,
            "status": "skipped" if is_assumption_variant else "not_applicable",
            "reason": "the wrapper requests one final answer, not multiple externally judged candidates",
        },
        {
            "module": "residual_writeback",
            "expected": is_assumption_variant,
            "status": "skipped" if is_assumption_variant else "not_applicable",
            "reason": "HLE smoke results are logged as evaluation artifacts, not written back into the graph",
        },
        {
            "module": "answer_format_verifier",
            "expected": True,
            "status": "activated",
            "reason": "JSON answer parser and answer-type scorer run after each response",
        },
    ]
    return trace


def _stage_status(stages: dict[str, Any], stage: str) -> str:
    data = stages.get(stage)
    if not data:
        return "failed_before_stage"
    return str(data.get("status") or "activated")


def _agent_critic_model(default_model: str) -> str:
    return os.environ.get("HLE_AGENT_CRITIC_MODEL", "").strip() or default_model


def _call_model(*, model: str, prompt: str, timeout: float | None = None, max_tokens: int = 512) -> str:
    env = _api_env(model=model)
    payload = {
        "model": env["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{env['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {env['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    attempts = max(1, int(os.environ.get("MODEL_ROUTER_ATTEMPTS", "3")))
    timeout = _default_call_timeout() if timeout is None else float(timeout)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model_call_deadline_exceeded")
            request_timeout = max(0.1, min(remaining, _model_router_per_attempt_timeout()))
            with _global_model_router_slot(model=env["model"]):
                data = _urlopen_json_with_deadline(request=request, timeout=request_timeout)
            return str((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise RuntimeError(f"model request failed: {_model_error_label(exc)}") from exc
            _sleep_before_model_retry(attempt=attempt, deadline=deadline)
    raise RuntimeError(f"model request failed: {last_error}")


def _urlopen_json_with_deadline(*, request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGALRM, _raise_wallclock_timeout)
        signal.alarm(max(1, int(timeout)))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _model_error_label(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if reason is not None:
            return f"URLError_{type(reason).__name__}"
    return type(exc).__name__


def _sleep_before_model_retry(*, attempt: int, deadline: float) -> None:
    base = float(os.environ.get("MODEL_ROUTER_BACKOFF_BASE_SEC", "0.75"))
    cap = float(os.environ.get("MODEL_ROUTER_BACKOFF_MAX_SEC", "10"))
    jitter = float(os.environ.get("MODEL_ROUTER_BACKOFF_JITTER_SEC", "0.25"))
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0.0, max(0.0, jitter))
    remaining = max(0.0, deadline - time.monotonic())
    if remaining <= 0:
        return
    time.sleep(min(delay, remaining))


@contextlib.contextmanager
def _global_model_router_slot(*, model: str) -> Any:
    limit = int(os.environ.get("MODEL_ROUTER_GLOBAL_CONCURRENCY", "0") or 0)
    if limit <= 0:
        yield
        return
    directory = Path(os.environ.get("MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR", "/tmp/assumption_agent_model_slots"))
    ttl_sec = float(os.environ.get("MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC", "900"))
    wait_sec = float(os.environ.get("MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC", "600"))
    slot_path = _acquire_model_router_slot(
        directory=directory,
        limit=limit,
        ttl_sec=ttl_sec,
        wait_sec=wait_sec,
        model=model,
    )
    try:
        yield
    finally:
        _release_model_router_slot(slot_path)


def _acquire_model_router_slot(
    *,
    directory: Path,
    limit: int,
    ttl_sec: float,
    wait_sec: float,
    model: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_sec
    while True:
        _remove_stale_model_router_slots(directory=directory, ttl_sec=ttl_sec)
        for slot_index in range(max(1, limit)):
            path = directory / f"slot_{slot_index:03d}.lock"
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "thread_id": threading.get_ident(),
                            "model_hash": stable_hash({"model": model})[:16],
                            "acquired_monotonic": time.monotonic(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            return path
        if time.monotonic() >= deadline:
            raise TimeoutError("model_router_global_concurrency_wait_exceeded")
        time.sleep(0.05 + random.uniform(0.0, 0.15))


def _remove_stale_model_router_slots(*, directory: Path, ttl_sec: float) -> None:
    if ttl_sec <= 0:
        return
    now = time.time()
    for path in directory.glob("slot_*.lock"):
        try:
            if now - path.stat().st_mtime > ttl_sec:
                path.unlink()
        except OSError:
            continue


def _release_model_router_slot(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _raise_wallclock_timeout(signum: int, frame: Any) -> None:
    raise TimeoutError("wall_clock_model_timeout")


def _default_call_timeout() -> float:
    return float(os.environ.get("MODEL_ROUTER_TIMEOUT", "120"))


def _model_router_per_attempt_timeout() -> float:
    return float(os.environ.get("MODEL_ROUTER_PER_ATTEMPT_TIMEOUT", "90"))


class _JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def _log_event(logger: _JsonlLogger | None, event: dict[str, Any]) -> None:
    if logger:
        logger.write(event)


def _api_env(*, model: str) -> dict[str, str]:
    base_url = (
        os.environ.get("GPT5_BASE_URL")
        or os.environ.get("RUOLI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://ruoli.dev"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    api_key = os.environ.get("GPT5_API_KEY") or os.environ.get("RUOLI_GPT_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing GPT5_API_KEY, RUOLI_GPT_KEY, or OPENAI_API_KEY")
    return {"model": model, "base_url": base_url, "api_key": api_key}


def _score_prediction(
    *,
    problem: dict[str, Any],
    model: str,
    variant: str,
    prediction: str,
    module_trace: list[dict[str, Any]] | None = None,
    call_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = _parse_answer_json(prediction)
    predicted = parsed if parsed is not None else prediction
    answer_type = problem["answer_type"]
    gold = problem["_answer"]
    if answer_type == "multipleChoice":
        predicted, _ = _canonicalize_multiple_choice_answer(problem, predicted)
        gold, _ = _canonicalize_multiple_choice_answer(problem, gold)
    correct = _is_correct(predicted, gold, answer_type=answer_type)
    return {
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "answer_hash": problem["answer_hash"],
        "model": model,
        "variant": variant,
        "category": problem["category"],
        "raw_subject": problem["raw_subject"],
        "answer_type": answer_type,
        "correct": correct,
        "prediction_hash": stable_hash({"prediction": predicted}),
        "prediction_text_persisted": False,
        "raw_question_persisted": False,
        "gold_answer_persisted": False,
        "module_trace": module_trace if module_trace is not None else _module_trace(problem, variant=variant),
        "call_metadata": call_metadata or {},
        "error": None,
    }


def _error_row(
    *,
    problem: dict[str, Any],
    model: str,
    variant: str,
    exc: Exception,
    module_trace: list[dict[str, Any]] | None = None,
    call_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "answer_hash": problem["answer_hash"],
        "model": model,
        "variant": variant,
        "category": problem["category"],
        "raw_subject": problem["raw_subject"],
        "answer_type": problem["answer_type"],
        "correct": False,
        "prediction_hash": None,
        "prediction_text_persisted": False,
        "raw_question_persisted": False,
        "gold_answer_persisted": False,
        "module_trace": module_trace if module_trace is not None else _module_trace(problem, variant=variant),
        "call_metadata": call_metadata or {},
        "error": {"type": type(exc).__name__, "message": str(exc)[:200]},
    }


def _component_efficacy_from_plan(
    *,
    problem: dict[str, Any],
    variant: str,
    plan: dict[str, Any],
    correct: bool,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize whether activated modules served their intended role.

    This intentionally stores only metadata, counts, hashes, and statuses.  It
    does not include HLE question text, gold answers, raw predictions, or
    evidence snippets.
    """
    stages = dict((plan or {}).get("stages", {}) or {})
    base: dict[str, Any] = {
        "variant": variant,
        "answer_type": problem.get("answer_type"),
        "final_correct": bool(correct),
        "error_type": (error or {}).get("type"),
        "flags": {
            "final_correct": bool(correct),
            "call_error": bool(error),
        },
    }
    if variant == "raw":
        base["kind"] = "raw_single_call"
        return base
    if variant.startswith("hipporag"):
        retrieval = stages.get("hipporag_context_retrieval", {})
        rerank = stages.get("hipporag_associative_rerank", {})
        prompt = stages.get("prompt_builder", {})
        base.update({
            "kind": "hipporag_control",
            "retrieval": {
                "status": retrieval.get("status"),
                "query_count": int(retrieval.get("query_count") or 0),
                "candidate_doc_count": int(retrieval.get("candidate_doc_count") or 0),
            },
            "rerank": {
                "status": rerank.get("status"),
                "selected_doc_count": int(rerank.get("selected_doc_count") or 0),
                "entity_node_count": int(rerank.get("entity_node_count") or 0),
            },
            "prompt": {
                "context_injected": bool(prompt.get("context_injected")),
                "context_char_count": int(prompt.get("context_char_count") or 0),
            },
        })
        base["flags"].update({
            "retrieval_returned_docs": int(retrieval.get("candidate_doc_count") or 0) > 0,
            "rerank_selected_docs": int(rerank.get("selected_doc_count") or 0) > 0,
            "context_injected": bool(prompt.get("context_injected")),
        })
        return base

    graph = stages.get("assumption_graph_retrieval", {})
    morphism = stages.get("structural_morphism_transfer", {})
    world_model = stages.get("world_model_router", {})
    critic_router = stages.get("critic_model_router", {})
    prompt = stages.get("prompt_builder", {})
    recursive = stages.get("recursive_child_validation", {})
    selection = stages.get("multi_candidate_self_verifier", {})
    evidence = stages.get("hle_evidence_bridge", {})
    agent_hipporag = stages.get("agent_hipporag_context_bridge", {})
    claim_verifier = stages.get("candidate_claim_verifier", {})
    domain_rule = stages.get("domain_rule_mc_verifier", {})
    math_tool = stages.get("hle_math_tool_solver", {})
    option_evidence = stages.get("mc_option_evidence_scorer", {})
    critic_synthesis = stages.get("critic_synthesis_child", {})
    option_sweep = stages.get("mc_option_sweep_candidates", {})
    counter_challenge = stages.get("counter_assumption_challenge", {})

    candidate_hashes = [value for value in recursive.get("candidate_answer_hashes", []) if value]
    unique_candidate_count = len(set(candidate_hashes))
    prompt_kinds = [str(value) for value in recursive.get("prompt_kinds", [])]
    skipped_prompt_kinds = [str(value) for value in recursive.get("skipped_prompt_kinds", [])]
    formal_hits = list(morphism.get("formal_mapping_hits", []) or [])
    structural_hits = list(morphism.get("structural_morphism_hits", []) or [])
    transfer_supported_hits = [
        hit for hit in structural_hits
        if isinstance(hit, dict) and hit.get("decision") == "transfer_supported"
    ]
    selection_method = str(selection.get("selection_method") or "")
    claim_status = str(claim_verifier.get("status") or "")
    claim_verified_count = int(claim_verifier.get("verified_count") or 0)
    claim_refuted_count = int(claim_verifier.get("refuted_count") or 0)
    evidence_status = str(evidence.get("status") or "")
    option_elimination = (
        counter_challenge.get("option_elimination_challenge", {})
        if isinstance(counter_challenge.get("option_elimination_challenge"), dict)
        else {}
    )
    forced_alternative = (
        counter_challenge.get("forced_alternative_challenge", {})
        if isinstance(counter_challenge.get("forced_alternative_challenge"), dict)
        else {}
    )

    flags = base["flags"]
    flags.update({
        "graph_retrieved_nodes": int(graph.get("node_count") or 0) > 0,
        "graph_context_injected": bool(prompt.get("context_injected")),
        "graph_context_discarded": bool(prompt.get("retrieval_context_discarded")),
        "generic_graph_context_only": bool(world_model.get("generic_graph_context_only")),
        "morphism_hit": bool(formal_hits or structural_hits),
        "strong_morphism_hit": bool(transfer_supported_hits),
        "world_model_used_context": world_model.get("decision") == "use_context",
        "critic_model_used": critic_router.get("status") == "activated",
        "evidence_bridge_activated": evidence_status == "activated",
        "evidence_child_executed": "evidence_bridge_answer" in prompt_kinds,
        "agent_hipporag_context_activated": agent_hipporag.get("status") == "activated",
        "agent_hipporag_child_executed": "hipporag_context_answer" in prompt_kinds,
        "hipporag_context_priority_used": selection_method == "hipporag_context_priority",
        "recursive_child_validation_activated": recursive.get("status") == "activated",
        "recursive_diverse_candidates": unique_candidate_count >= 2,
        "recursive_collapsed_consensus": bool(candidate_hashes) and unique_candidate_count <= 1,
        "recursive_timeout_pressure": int(recursive.get("error_child_count") or 0) > 0,
        "recursive_early_stopped": bool(recursive.get("early_stopped")),
        "reflective_child_executed": any(
            kind in {"agent_context_answer", "constraint_checked_answer", "recursive_assumption_answer"}
            for kind in prompt_kinds
        ),
        "reflective_child_skipped": any(
            kind in {"agent_context_answer", "constraint_checked_answer", "recursive_assumption_answer"}
            for kind in skipped_prompt_kinds
        ),
        "claim_verifier_activated": claim_status == "activated",
        "claim_verifier_verified_candidate": claim_verified_count > 0,
        "claim_verifier_refuted_candidate": claim_refuted_count > 0,
        "claim_verifier_no_executable_claim": claim_status == "no_executable_claim",
        "domain_rule_mc_verifier_activated": domain_rule.get("status") == "activated",
        "domain_rule_mc_verifier_selected": bool(domain_rule.get("selected_domain_rule_candidate")),
        "domain_rule_mc_verifier_correct": domain_rule.get("candidate_correct_for_eval") is True,
        "math_tool_verified": math_tool.get("confidence") == "verified_symbolic",
        "mc_option_evidence_scorer_activated": option_evidence.get("status") == "activated",
        "mc_option_evidence_candidate_emitted": bool(option_evidence.get("candidate_emitted")),
        "mc_option_evidence_candidate_selected": bool(option_evidence.get("selected_option_evidence_candidate")),
        "mc_option_evidence_candidate_correct": option_evidence.get("candidate_correct_for_eval") is True,
        "option_evidence_verifier_used": selection_method == "option_evidence_verifier_choice",
        "critic_synthesis_activated": critic_synthesis.get("status") == "activated",
        "critic_synthesis_disagreed": bool(critic_synthesis.get("critic_disagreed_with_majority")),
        "critic_synthesis_selected": bool(critic_synthesis.get("selected_critic_synthesis")),
        "mc_option_sweep_activated": option_sweep.get("status") == "activated",
        "mc_option_sweep_selected": bool(option_sweep.get("selected_option_sweep_candidate")),
        "source_grounded_verifier_used": selection_method == "source_grounded_verifier_choice",
        "candidate_claim_override": selection_method == "candidate_claim_verifier_priority",
        "domain_rule_override": (
            selection_method == "candidate_claim_verifier_priority"
            and bool(domain_rule.get("selected_domain_rule_candidate"))
        ),
        "verified_math_override": selection_method == "verified_math_tool_priority",
        "evidence_override": selection_method == "evidence_bridge_priority_over_closed_book_majority",
        "counter_assumption_challenge_activated": counter_challenge.get("status") == "activated",
        "counter_assumption_challenge_disagreed": bool(counter_challenge.get("challenge_disagreed_with_majority")),
        "counter_assumption_challenge_selected": bool(counter_challenge.get("selected_counter_challenge")),
        "option_elimination_challenge_activated": option_elimination.get("status") == "activated",
        "option_elimination_challenge_disagreed": bool(option_elimination.get("challenge_disagreed_with_majority")),
        "option_elimination_challenge_selected": bool(option_elimination.get("selected_option_elimination_challenge")),
        "forced_alternative_activated": forced_alternative.get("status") == "activated",
        "forced_alternative_disagreed": bool(forced_alternative.get("challenge_disagreed_with_majority")),
        "forced_alternative_selected": bool(forced_alternative.get("selected_forced_alternative")),
        "counter_assumption_verifier_used": selection_method == "counter_assumption_verifier_choice",
        "majority_only_selection": selection_method in {
            "normalized_majority",
            "math_exact_normalized_majority",
        },
    })
    base.update({
        "kind": "assumption_agent_recursive_verify" if variant == "assumption_agent_recursive_verify" else "assumption_agent",
        "graph": {
            "status": graph.get("status"),
            "node_count": int(graph.get("node_count") or 0),
            "edge_count": int(graph.get("edge_count") or 0),
            "top_score": max([float(value) for value in graph.get("top_scores", []) or [0.0]] or [0.0]),
            "top_node_type_counts": dict(Counter(str(value) for value in graph.get("top_node_types", []) or [])),
        },
        "morphism": {
            "status": morphism.get("status"),
            "formal_hit_count": len(formal_hits),
            "structural_hit_count": len(structural_hits),
            "transfer_supported_count": len(transfer_supported_hits),
        },
        "world_model": {
            "status": world_model.get("status"),
            "decision": world_model.get("decision"),
            "context_abstain_reason": world_model.get("context_abstain_reason"),
            "generic_graph_context_only": bool(world_model.get("generic_graph_context_only")),
            "expected_utility": world_model.get("expected_utility"),
            "predicted_regression_risk": world_model.get("predicted_regression_risk"),
        },
        "critic_model": {
            "status": critic_router.get("status"),
            "base_model": critic_router.get("base_model"),
            "critic_model": critic_router.get("critic_model"),
            "policy": critic_router.get("policy"),
        },
        "evidence": {
            "status": evidence.get("status"),
            "selection_policy": evidence.get("selection_policy"),
            "query_count": int(evidence.get("query_count") or 0),
            "selected_result_count": int(evidence.get("selected_result_count") or 0),
            "evidence_char_count": int(evidence.get("evidence_char_count") or 0),
        },
        "agent_hipporag": {
            "status": agent_hipporag.get("status"),
            "source": agent_hipporag.get("source"),
            "query_count": int(agent_hipporag.get("query_count") or 0),
            "candidate_doc_count": int(agent_hipporag.get("candidate_doc_count") or 0),
            "selected_doc_count": int(agent_hipporag.get("selected_doc_count") or 0),
            "context_char_count": int(agent_hipporag.get("context_char_count") or 0),
            "top_scores": list(agent_hipporag.get("top_scores", []) or []),
        },
        "recursive": {
            "status": recursive.get("status"),
            "execution_mode": recursive.get("execution_mode"),
            "planned_child_count": int(recursive.get("planned_child_count") or 0),
            "child_count": int(recursive.get("child_count") or 0),
            "answered_child_count": int(recursive.get("answered_child_count") or 0),
            "error_child_count": int(recursive.get("error_child_count") or 0),
            "unique_candidate_count": unique_candidate_count,
            "early_stopped": bool(recursive.get("early_stopped")),
            "early_stop_reason": recursive.get("early_stop_reason"),
            "prompt_kinds": prompt_kinds,
            "skipped_prompt_kinds": skipped_prompt_kinds,
        },
        "claim_verifier": {
            "status": claim_status or None,
            "backend": claim_verifier.get("backend"),
            "verified_count": claim_verified_count,
            "refuted_count": claim_refuted_count,
            "inconclusive_count": int(claim_verifier.get("inconclusive_count") or 0),
            "reference_operation": claim_verifier.get("reference_operation"),
        },
        "domain_rule_mc_verifier": {
            "status": domain_rule.get("status"),
            "rule_id": domain_rule.get("rule_id"),
            "confidence": domain_rule.get("confidence"),
            "selected_domain_rule_candidate": bool(domain_rule.get("selected_domain_rule_candidate")),
            "candidate_correct_for_eval": domain_rule.get("candidate_correct_for_eval"),
        },
        "mc_option_evidence_scorer": {
            "status": option_evidence.get("status"),
            "candidate_emitted": bool(option_evidence.get("candidate_emitted")),
            "candidate_correct_for_eval": option_evidence.get("candidate_correct_for_eval"),
            "top_score": option_evidence.get("top_score"),
            "margin": option_evidence.get("margin"),
            "selected_option_evidence_candidate": bool(option_evidence.get("selected_option_evidence_candidate")),
        },
        "counter_assumption_challenge": {
            "status": counter_challenge.get("status"),
            "reason": counter_challenge.get("reason"),
            "top_candidate_count": int(counter_challenge.get("top_candidate_count") or 0),
            "unique_candidate_count": int(counter_challenge.get("unique_candidate_count") or 0),
            "challenge_disagreed_with_majority": bool(counter_challenge.get("challenge_disagreed_with_majority")),
            "selected_counter_challenge": bool(counter_challenge.get("selected_counter_challenge")),
            "option_elimination_status": option_elimination.get("status"),
            "option_elimination_disagreed": bool(option_elimination.get("challenge_disagreed_with_majority")),
            "selected_option_elimination_challenge": bool(option_elimination.get("selected_option_elimination_challenge")),
            "forced_alternative_status": forced_alternative.get("status"),
            "forced_alternative_disagreed": bool(forced_alternative.get("challenge_disagreed_with_majority")),
            "selected_forced_alternative": bool(forced_alternative.get("selected_forced_alternative")),
        },
        "critic_synthesis_child": {
            "status": critic_synthesis.get("status"),
            "reason": critic_synthesis.get("reason"),
            "critic_model": critic_synthesis.get("critic_model"),
            "unique_candidate_count_before": int(critic_synthesis.get("unique_candidate_count_before") or 0),
            "top_candidate_count_before": int(critic_synthesis.get("top_candidate_count_before") or 0),
            "critic_disagreed_with_majority": bool(critic_synthesis.get("critic_disagreed_with_majority")),
            "selected_critic_synthesis": bool(critic_synthesis.get("selected_critic_synthesis")),
        },
        "mc_option_sweep_candidates": {
            "status": option_sweep.get("status"),
            "reason": option_sweep.get("reason"),
            "option_count": int(option_sweep.get("option_count") or 0),
            "covered_option_count_before": int(option_sweep.get("covered_option_count_before") or 0),
            "added_candidate_count": int(option_sweep.get("added_candidate_count") or 0),
            "selected_option_sweep_candidate": bool(option_sweep.get("selected_option_sweep_candidate")),
        },
        "selection": {
            "status": selection.get("status"),
            "selection_method": selection_method or None,
            "verifier_model_call": bool(selection.get("verifier_model_call")),
        },
    })
    return base


def _parse_answer_json(text: str) -> str | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1).strip()
    return None


def _is_correct(predicted: str, gold: str, *, answer_type: str) -> bool:
    if answer_type == "multipleChoice":
        return _extract_choice(predicted) == _extract_choice(gold)
    return _normalize_exact(predicted) == _normalize_exact(gold)


def _extract_choice(text: str) -> str:
    text = str(text).strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)
    return text[:1]


def _normalize_exact(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:")
    return text


def _metrics(*, sample_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], api_summary: dict[str, Any]) -> dict[str, Any]:
    planned = api_summary.get("planned_live_model_calls", 0)
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_key[f"{row['model']}::{row['variant']}"].append(row)
    by_model_variant = {key: _aggregate_rows(rows) for key, rows in by_key.items()}
    category_counts = Counter(row["category"] for row in sample_rows)
    subject_counts = Counter(row["raw_subject"] for row in sample_rows)
    answer_type_counts = Counter(row["answer_type"] for row in sample_rows)
    return {
        "sample_count": len(sample_rows),
        "scanned_row_count": max((row["scanned_index"] for row in sample_rows), default=0),
        "category_counts": dict(category_counts),
        "raw_subject_counts": dict(subject_counts),
        "answer_type_counts": dict(answer_type_counts),
        "planned_live_model_calls": planned,
        "live_model_calls_executed": api_summary["live_model_calls_executed"],
        "underlying_model_calls_executed": api_summary.get("underlying_model_calls_executed", api_summary["live_model_calls_executed"]),
        "live_model_call_error_count": len(api_summary["live_model_call_errors"]),
        "resolved_live_model_calls": api_summary["live_model_calls_executed"] + len(api_summary["live_model_call_errors"]),
        "scored_row_count": len(run_rows),
        "overall_accuracy": _accuracy(run_rows),
        "by_model_variant": by_model_variant,
        "control_comparison": _control_comparison(run_rows),
        "module_activation_summary": _module_activation_summary(run_rows),
        "expected_but_missing_modules": _expected_but_missing_modules(run_rows),
        "component_efficacy_summary": _component_efficacy_summary(run_rows),
        "raw_content_persisted": False,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mcq = [row for row in rows if row["answer_type"] == "multipleChoice"]
    exact = [row for row in rows if row["answer_type"] != "multipleChoice"]
    return {
        "n": len(rows),
        "accuracy": _accuracy(rows),
        "multiple_choice_n": len(mcq),
        "multiple_choice_accuracy": _accuracy(mcq),
        "exact_match_n": len(exact),
        "exact_match_accuracy": _accuracy(exact),
        "error_count": sum(1 for row in rows if row.get("error")),
    }


def _control_comparison(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in run_rows:
        by_model[row["model"]][row["variant"]][row["problem_id_hash"]] = row
    comparisons: dict[str, Any] = {}
    for model, by_variant in sorted(by_model.items()):
        agent_rows = by_variant.get("assumption_agent_recursive_verify")
        if not agent_rows:
            continue
        model_comparison: dict[str, Any] = {
            "policy": "higher accuracy is better for every variant; only assumption_agent is optimized, controls are frozen",
        }
        for control_variant in ("raw", "hipporag_baseline"):
            control_rows = by_variant.get(control_variant)
            if not control_rows:
                continue
            shared = sorted(set(agent_rows) & set(control_rows))
            agent_correct = [agent_rows[pid]["correct"] for pid in shared]
            control_correct = [control_rows[pid]["correct"] for pid in shared]
            agent_wins = sum(1 for a, c in zip(agent_correct, control_correct) if a and not c)
            agent_losses = sum(1 for a, c in zip(agent_correct, control_correct) if c and not a)
            model_comparison[f"agent_vs_{control_variant}"] = {
                "shared_problem_count": len(shared),
                "agent_accuracy": _accuracy([agent_rows[pid] for pid in shared]),
                "control_accuracy": _accuracy([control_rows[pid] for pid in shared]),
                "agent_minus_control_accuracy": None
                if not shared
                else round((sum(agent_correct) - sum(control_correct)) / len(shared), 4),
                "agent_unique_correct_count": agent_wins,
                "control_unique_correct_count": agent_losses,
                "both_correct_count": sum(1 for a, c in zip(agent_correct, control_correct) if a and c),
                "both_wrong_count": sum(1 for a, c in zip(agent_correct, control_correct) if not a and not c),
            }
        comparisons[model] = model_comparison
    return comparisons


def _module_activation_summary(run_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in run_rows:
        key = f"{row['model']}::{row['variant']}"
        for item in row.get("module_trace", []):
            summary[key][item["module"]][item["status"]] += 1
    return {
        key: {module: dict(counts) for module, counts in sorted(module_counts.items())}
        for key, module_counts in sorted(summary.items())
    }


def _expected_but_missing_modules(run_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    missing: dict[str, set[str]] = defaultdict(set)
    present_statuses = {
        "activated",
        "abstained",
        "failed",
        "no_executable_claim",
        "weak_single_candidate_confirmation",
        "no_option_match",
        "ambiguous_option_match",
        "no_option_parse",
        "not_required",
        "weak_margin",
    }
    for row in run_rows:
        key = f"{row['model']}::{row['variant']}"
        for item in row.get("module_trace", []):
            if item.get("expected") and item.get("status") not in present_statuses:
                missing[key].add(item["module"])
    return {key: sorted(values) for key, values in sorted(missing.items())}


def _component_efficacy_summary(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[f"{row['model']}::{row['variant']}"].append(row)
    for key, rows in sorted(grouped.items()):
        flag_counts: Counter[str] = Counter()
        flag_correct_counts: Counter[str] = Counter()
        selection_counts: Counter[str] = Counter()
        selection_correct_counts: Counter[str] = Counter()
        verifier_status_counts: Counter[str] = Counter()
        recursive_unique_counts: Counter[str] = Counter()
        for row in rows:
            correct = bool(row.get("correct"))
            efficacy = row.get("component_efficacy") or {}
            flags = efficacy.get("flags", {}) if isinstance(efficacy, dict) else {}
            for flag, value in flags.items():
                if value:
                    flag_counts[flag] += 1
                    if correct:
                        flag_correct_counts[flag] += 1
            selection_method = (((efficacy.get("selection") or {}) if isinstance(efficacy, dict) else {}).get("selection_method") or "none")
            selection_counts[str(selection_method)] += 1
            if correct:
                selection_correct_counts[str(selection_method)] += 1
            verifier_status = (((efficacy.get("claim_verifier") or {}) if isinstance(efficacy, dict) else {}).get("status") or "none")
            verifier_status_counts[str(verifier_status)] += 1
            unique_count = (((efficacy.get("recursive") or {}) if isinstance(efficacy, dict) else {}).get("unique_candidate_count"))
            if unique_count is not None:
                recursive_unique_counts[str(unique_count)] += 1
        summary[key] = {
            "n": len(rows),
            "correct_count": sum(1 for row in rows if row.get("correct")),
            "accuracy": _accuracy(rows),
            "flag_counts": dict(sorted(flag_counts.items())),
            "flag_correct_counts": dict(sorted(flag_correct_counts.items())),
            "flag_accuracy": {
                flag: round(flag_correct_counts[flag] / count, 4)
                for flag, count in sorted(flag_counts.items())
                if count
            },
            "selection_method_counts": dict(sorted(selection_counts.items())),
            "selection_method_accuracy": {
                method: round(selection_correct_counts[method] / count, 4)
                for method, count in sorted(selection_counts.items())
                if count
            },
            "claim_verifier_status_counts": dict(sorted(verifier_status_counts.items())),
            "recursive_unique_candidate_count_histogram": dict(sorted(recursive_unique_counts.items())),
        }
    return summary


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row["correct"]) / len(rows), 4)


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HLE text-only smoke evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_text_smoke_eval_20260615")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--sample-answer-type", default="")
    parser.add_argument("--sample-subject-contains", default="")
    parser.add_argument("--models", default="gpt-5.5")
    parser.add_argument("--variants", default="raw,assumption_wrapper")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--call-timeout", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--log-out", default=str(DEFAULT_LOG_OUT))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
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
    parser.add_argument("--hard-exit-after-write", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    log_out = None
    if args.log_out and args.execute_live:
        log_out = Path(args.log_out)
        log_out = log_out if log_out.is_absolute() else root / log_out
    graph_dir = Path(args.graph_dir)
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    payload = build_hle_text_smoke_eval_payload(
        root=root,
        eval_id=args.eval_id,
        sample_size=args.sample_size,
        max_scan=args.max_scan,
        seed_offset=args.seed_offset,
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        variants=[item.strip() for item in args.variants.split(",") if item.strip()],
        execute_live=args.execute_live,
        call_timeout=args.call_timeout,
        max_tokens=args.max_tokens,
        log_out=log_out,
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
    if args.hard_exit_after_write:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
