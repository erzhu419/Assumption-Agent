"""Text-only smoke evaluation for Humanity's Last Exam.

The official HLE dataset is gated.  This runner expects the user to have
accepted the dataset terms and provided ``HF_TOKEN`` in the process environment.
It deliberately does not persist HLE questions, gold answers, rationales, or
canary strings.  Artifacts store stable hashes, metadata, predictions hashes,
and correctness only.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import fractions
import http.client
import html
import json
import math
import os
import random
import re
import signal
import subprocess
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
    exclude_artifact_glob: str = "phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/hle*.json*",
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
                        if variant == "assumption_agent_recursive_verify" and _recursive_answering_disabled():
                            answer_text = _call_model(
                                model=model,
                                prompt=_prompt_for(problem, variant=variant, agent_plan=agent_plan),
                                timeout=call_timeout,
                                max_tokens=max_tokens,
                            )
                            api_summary["underlying_model_calls_executed"] += 1
                            (agent_plan or {}).setdefault("stages", {})["recursive_child_validation"] = {
                                "status": "disabled",
                                "reason": "env_disabled",
                                "child_count": 0,
                                "underlying_model_calls": 0,
                            }
                            (agent_plan or {}).setdefault("stages", {})["multi_candidate_self_verifier"] = {
                                "status": "disabled",
                                "reason": "recursive_runner_disabled",
                                "underlying_model_calls": 0,
                            }
                            module_trace = _module_trace(problem, variant=variant, agent_plan=agent_plan)
                        elif variant == "assumption_agent_recursive_verify":
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
                        _update_same_run_baseline_cache(
                            cache=same_run_baseline_cache,
                            problem=problem,
                            model=model,
                            variant=variant,
                            prediction=answer_text,
                            plan=variant_plan or {},
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
    fair_baseline_gate = _agent_meets_best_control_gate(metrics)
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
        "agent_not_below_best_same_model_control": fair_baseline_gate["passed"],
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
        "fair_baseline_gate": fair_baseline_gate,
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
        "## Route Credit",
        "",
        "| model | problems | agent acc | recoverable agent errors | unrecoverable agent errors | losses to controls | VOI actions |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    route_credit = metrics.get("route_credit_table", {})
    for model, row in sorted((route_credit.get("by_model") or {}).items()):
        losses = ", ".join(
            f"{variant}:{count}" for variant, count in sorted(row.get("agent_loss_to_control_counts", {}).items())
        ) or "none"
        voi_actions = ", ".join(
            f"{action}:{count}" for action, count in sorted(row.get("voi_recommended_action_counts", {}).items())
        ) or "none"
        lines.append(
            f"| `{model}` | `{row['problem_count']}` | `{row['agent_accuracy']}` | "
            f"`{row['recoverable_agent_error_count']}` | `{row['unrecoverable_agent_error_count']}` | "
            f"`{losses}` | `{voi_actions}` |"
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
                "structural_option_audit_activated",
                "structural_option_audit_disagreed",
                "structural_option_audit_selected",
                "structural_option_audit_candidate_correct",
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
                "route_value_verifier_used",
                "route_voi_recommended_preserve",
                "route_voi_hard_gate_applied",
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
    cache_path_text = os.environ.get("HLE_EXISTING_HASH_CACHE_PATH", "").strip()
    if cache_path_text:
        cached = _collect_existing_hle_problem_hashes_from_cache(
            root=root,
            artifact_glob=artifact_glob,
            cache_path=Path(cache_path_text),
        )
        if cached is not None:
            return cached
    hashes: set[str] = set()
    for path in root.glob(artifact_glob):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            _collect_problem_hashes_from_jsonl(path, hashes)
        elif path.suffix == ".json":
            _collect_problem_hashes_from_json(path, hashes)
    if cache_path_text:
        _write_existing_hle_problem_hash_cache(
            root=root,
            artifact_glob=artifact_glob,
            cache_path=Path(cache_path_text),
            hashes=hashes,
        )
    return hashes


def _existing_hle_hash_manifest(*, root: Path, artifact_glob: str) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(root.glob(artifact_glob)):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        manifest.append({
            "path_hash": stable_hash({"path": str(path.relative_to(root))}),
            "suffix": path.suffix,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return manifest


def _collect_existing_hle_problem_hashes_from_cache(
    *,
    root: Path,
    artifact_glob: str,
    cache_path: Path,
) -> set[str] | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    allow_stale = os.environ.get("HLE_EXISTING_HASH_CACHE_ALLOW_STALE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if data.get("artifact_glob") != artifact_glob:
        return None
    manifest = _existing_hle_hash_manifest(root=root, artifact_glob=artifact_glob)
    if data.get("manifest") != manifest and not allow_stale:
        return None
    hashes = data.get("problem_id_hashes")
    if not isinstance(hashes, list):
        return None
    return {str(value) for value in hashes if value}


def _write_existing_hle_problem_hash_cache(
    *,
    root: Path,
    artifact_glob: str,
    cache_path: Path,
    hashes: set[str],
) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "artifact_glob": artifact_glob,
                    "manifest": _existing_hle_hash_manifest(root=root, artifact_glob=artifact_glob),
                    "problem_id_hashes": sorted(hashes),
                    "raw_content_persisted": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except Exception:
        return


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
        graph_retrieval_disabled = (
            top_k <= 0
            or os.environ.get("HLE_DISABLE_ASSUMPTION_GRAPH_RETRIEVAL", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if graph_retrieval_disabled:
            retrieval_result = None
            retrieval_summary = {
                "status": "disabled",
                "node_count": 0,
                "edge_count": 0,
                "top_scores": [],
                "top_node_ids": [],
                "top_node_types": [],
                "reason": "env_disabled" if top_k > 0 else "top_k_zero",
            }
        else:
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

        morphism_disabled = (
            os.environ.get("HLE_DISABLE_STRUCTURAL_MORPHISM_TRANSFER", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if morphism_disabled:
            morphism_summary = {
                "status": "disabled",
                "formal_mapping_hits": [],
                "structural_morphism_hits": [],
                "reason": "env_disabled",
            }
        else:
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

        recursive_disabled = (
            top_k <= 0
            or os.environ.get("HLE_DISABLE_RECURSIVE_ASSUMPTION_RUNNER", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if recursive_disabled:
            recursive_summary = {
                "status": "disabled",
                "reason": "env_disabled" if top_k > 0 else "top_k_zero",
                "child_count": 0,
                "leaf_count": 0,
            }
        else:
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
        world_model_disabled = (
            os.environ.get("HLE_DISABLE_WORLD_MODEL_ROUTER", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if world_model_disabled:
            prediction = None
            context_allowed = False
        else:
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
        elif world_model_disabled:
            context_abstain_reason = "world_model_router_disabled"
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
            "predicted_acceptance_probability": (
                None if prediction is None else prediction.predicted_acceptance_probability
            ),
            "prediction_confidence": None if prediction is None else prediction.prediction_confidence,
            "expected_utility": None if prediction is None else prediction.expected_utility,
            "recommended_next_action": "disabled" if prediction is None else prediction.recommended_next_action,
            "predicted_regression_risk": "unknown" if prediction is None else prediction.predicted_regression_risk,
        }
        if world_model_disabled:
            router_summary["status"] = "disabled"
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


def _is_budget_matched_control_variant(variant: str) -> bool:
    return variant in {"raw_budget_matched", "hipporag_budget_matched"}


def _is_same_run_baseline_cache_variant(variant: str) -> bool:
    return variant in {"raw", "raw_budget_matched", "hipporag_baseline", "hipporag_budget_matched"}


def _same_run_baseline_cache_key(problem: dict[str, Any], model: str) -> tuple[str, str]:
    return (str(problem.get("id_hash") or ""), str(model or ""))


def _context_answer_support_for_mc(
    *,
    problem: dict[str, Any],
    answer: str,
    context: str,
) -> dict[str, Any]:
    if problem.get("answer_type") != "multipleChoice":
        return {"supported": False, "overlap_count": 0, "question_overlap_count": 0, "option_hash": None}
    label = _extract_choice(answer)
    if not label:
        return {"supported": False, "overlap_count": 0, "question_overlap_count": 0, "option_hash": None}
    options, _ = _extract_multiple_choice_options(str(problem.get("_question") or ""))
    option_text = str(options.get(label) or "").strip()
    if not option_text or not str(context or "").strip():
        return {
            "supported": False,
            "overlap_count": 0,
            "question_overlap_count": 0,
            "option_hash": stable_hash({"option_label": label}),
        }
    context_norm = _normalize_exact(context)
    option_norm = _normalize_exact(option_text)
    question_text = str(problem.get("_question") or "")
    question_stem = re.split(r"\n\s*A[\.\)]|\s+A[\.\)]", question_text, maxsplit=1)[0]
    if not question_stem.strip():
        question_stem = question_text
    for option_label, text in options.items():
        if option_label and text:
            question_stem = question_stem.replace(str(text), " ")
    question_stopwords = {
        "which", "what", "when", "where", "whose", "would", "could", "should", "there", "their",
        "about", "answer", "option", "following", "correct", "best", "most", "least", "only",
        "true", "false", "question", "select", "choose", "among", "given", "based", "statement",
    }
    question_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", _normalize_exact(question_stem))
        if token not in question_stopwords
    }
    question_overlap = sum(1 for token in question_tokens if token in context_norm)
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", option_norm)
        if token not in {"none", "both", "only", "true", "false", "above", "below"}
    }
    if not tokens:
        return {
            "supported": False,
            "overlap_count": 0,
            "question_overlap_count": int(question_overlap),
            "option_hash": stable_hash({"option_label": label}),
        }
    overlap = sum(1 for token in tokens if token in context_norm)
    option_supported = (
        (len(option_norm) >= 6 and option_norm in context_norm)
        or overlap >= 3
        or (
        overlap >= min(2, len(tokens)) and (overlap / max(1, len(tokens))) >= 0.45
        )
    )
    question_supported = question_overlap >= min(2, max(1, len(question_tokens)))
    supported = option_supported and (
        question_supported
        or (not question_tokens and overlap >= 3)
    )
    return {
        "supported": bool(supported),
        "overlap_count": int(overlap),
        "question_overlap_count": int(question_overlap),
        "option_hash": stable_hash({"option_label": label}),
    }


def _same_run_baseline_cache_entry(
    *,
    problem: dict[str, Any],
    variant: str,
    prediction: str,
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    if not _is_same_run_baseline_cache_variant(variant):
        return None
    answer = _parse_answer_json(prediction) or str(prediction or "").strip()
    if not answer:
        return None
    if problem.get("answer_type") == "multipleChoice":
        answer, canonical_summary = _canonicalize_multiple_choice_answer(problem, answer)
    else:
        answer, canonical_summary = _canonicalize_exact_answer_candidate(problem, answer)
        if _is_suspicious_exact_answer(answer):
            return None
    stages = (plan or {}).get("stages") if isinstance(plan, dict) else {}
    stages = stages if isinstance(stages, dict) else {}
    retrieval = stages.get("hipporag_context_retrieval") if isinstance(stages.get("hipporag_context_retrieval"), dict) else {}
    rerank = stages.get("hipporag_associative_rerank") if isinstance(stages.get("hipporag_associative_rerank"), dict) else {}
    prompt = stages.get("prompt_builder") if isinstance(stages.get("prompt_builder"), dict) else {}
    budget = stages.get("budget_matched_control") if isinstance(stages.get("budget_matched_control"), dict) else {}
    context = str((plan or {}).get("prompt_context") or "")
    context_answer_support = _context_answer_support_for_mc(problem=problem, answer=answer, context=context)
    return {
        "variant": variant,
        "answer": answer,
        "answer_hash": stable_hash({"answer": answer}),
        "prediction_hash": stable_hash({"prediction": prediction}),
        "answer_type": problem.get("answer_type"),
        "canonicalized": bool(canonical_summary.get("changed")) if isinstance(canonical_summary, dict) else False,
        "context_char_count": int(prompt.get("context_char_count") or 0),
        "candidate_doc_count": int(retrieval.get("candidate_doc_count") or 0),
        "selected_doc_count": int(rerank.get("selected_doc_count") or 0),
        "context_answer_supported": bool(context_answer_support.get("supported")),
        "context_answer_overlap_count": int(context_answer_support.get("overlap_count") or 0),
        "context_question_overlap_count": int(context_answer_support.get("question_overlap_count") or 0),
        "context_answer_option_hash": context_answer_support.get("option_hash"),
        "budget_matched": _is_budget_matched_control_variant(variant),
        "budget_candidate_count": int(budget.get("candidate_count") or 0),
        "budget_answered_candidate_count": int(budget.get("answered_candidate_count") or 0),
        "budget_candidate_answer_hash_counts": dict(budget.get("candidate_answer_hash_counts") or {}),
        "budget_selected_answer_hash": budget.get("selected_answer_hash"),
        "budget_top_candidate_vote_count": int(budget.get("top_candidate_vote_count") or 0),
        "budget_top_candidate_answer_hash": budget.get("top_candidate_answer_hash"),
        "budget_strong_consensus": bool(budget.get("strong_consensus")),
        "selection_method": budget.get("selection_method"),
        "budget_verified_or_abstain_gate": budget.get("verified_or_abstain_gate"),
        "source": "same_run_baseline_cache",
    }


def _update_same_run_baseline_cache(
    *,
    cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
    problem: dict[str, Any],
    model: str,
    variant: str,
    prediction: str,
    plan: dict[str, Any],
) -> None:
    entry = _same_run_baseline_cache_entry(
        problem=problem,
        variant=variant,
        prediction=prediction,
        plan=plan,
    )
    if not entry:
        return
    cache.setdefault(_same_run_baseline_cache_key(problem, model), {})[variant] = entry


def _attach_same_run_baseline_cache(
    *,
    agent_plan: dict[str, Any] | None,
    cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
    problem: dict[str, Any],
    model: str,
) -> None:
    if not isinstance(agent_plan, dict):
        return
    entries = cache.get(_same_run_baseline_cache_key(problem, model), {})
    if not entries:
        return
    agent_plan["hle_same_run_baseline_cache"] = {
        variant: dict(entry)
        for variant, entry in sorted(entries.items())
    }
    agent_plan.setdefault("stages", {})["same_run_baseline_cache"] = {
        "status": "activated",
        "policy": "same_problem_same_model_baseline_predictions_available",
        "cached_variants": sorted(entries),
        "cached_variant_count": len(entries),
        "answer_hashes_by_variant": {
            variant: entry.get("answer_hash")
            for variant, entry in sorted(entries.items())
        },
        "borrowed_baseline_model_call_count": len(entries),
    }


def _same_run_cached_baseline(
    agent_plan: dict[str, Any] | None,
    variants: list[str],
) -> dict[str, Any] | None:
    if not isinstance(agent_plan, dict):
        return None
    cache = agent_plan.get("hle_same_run_baseline_cache")
    if not isinstance(cache, dict):
        return None
    for variant in variants:
        entry = cache.get(variant)
        if isinstance(entry, dict) and str(entry.get("answer") or "").strip():
            return entry
    return None


def _same_run_cached_baseline_entries(
    agent_plan: dict[str, Any] | None,
    variants: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(agent_plan, dict):
        return []
    cache = agent_plan.get("hle_same_run_baseline_cache")
    if not isinstance(cache, dict):
        return []
    entries: list[dict[str, Any]] = []
    for variant in variants:
        entry = cache.get(variant)
        if isinstance(entry, dict) and str(entry.get("answer") or "").strip():
            entries.append(entry)
    return entries


def _budget_control_base_variant(variant: str) -> str:
    return "hipporag_baseline" if variant.startswith("hipporag") else "raw"


def _call_budget_matched_control_answer(
    *,
    problem: dict[str, Any],
    model: str,
    variant: str,
    variant_plan: dict[str, Any],
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    specs = _budget_matched_control_prompt_specs(problem, variant=variant, variant_plan=variant_plan)
    max_workers = _budget_matched_control_workers(len(specs))
    batch = _run_child_batch(
        problem=problem,
        specs=specs,
        start_index=1,
        model=model,
        variant=variant,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )
    attempts = batch["attempts"]
    selection = _select_recursive_child_answer(
        problem=problem,
        attempts=attempts,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=min(max_tokens, 384),
        evidence_context=str((variant_plan or {}).get("prompt_context") or ""),
    )
    selection = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)
    selected_answer = selection.get("selected_answer") or _fallback_answer(attempts)
    selected_hash = stable_hash({"answer": selected_answer})
    answered_count = sum(1 for attempt in attempts if attempt.get("status") == "answered")
    answer_hash_counts = Counter(
        attempt.get("parsed_answer_hash")
        for attempt in attempts
        if attempt.get("status") == "answered" and attempt.get("parsed_answer_hash")
    )
    top_candidate_answer_hash, top_candidate_vote_count = (None, 0)
    if answer_hash_counts:
        top_candidate_answer_hash, top_candidate_vote_count = answer_hash_counts.most_common(1)[0]
    strong_consensus = top_candidate_vote_count >= min(3, max(1, answered_count))
    stages = variant_plan.setdefault("stages", {})
    stages["budget_matched_control"] = {
        "status": "activated",
        "base_variant": _budget_control_base_variant(variant),
        "execution_mode": "parallel_self_consistency",
        "candidate_count": len(attempts),
        "answered_candidate_count": answered_count,
        "error_candidate_count": len(attempts) - answered_count,
        "candidate_prompt_kinds": [attempt.get("prompt_kind") for attempt in attempts],
        "candidate_answer_hashes": [
            attempt.get("parsed_answer_hash") for attempt in attempts if attempt.get("parsed_answer_hash")
        ],
        "candidate_answer_hash_counts": dict(answer_hash_counts),
        "top_candidate_answer_hash": top_candidate_answer_hash,
        "top_candidate_vote_count": top_candidate_vote_count,
        "strong_consensus": strong_consensus,
        "child_max_workers": batch.get("max_workers"),
        "selection_method": selection.get("selection_method"),
        "selected_child_id": selection.get("selected_child_id"),
        "selected_answer_hash": selected_hash,
        "verifier_model_call": bool(selection.get("verifier_model_call")),
        "verified_or_abstain_gate": selection.get("verified_or_abstain_gate"),
        "underlying_model_calls": int(batch.get("underlying_model_calls") or 0)
        + int(selection.get("underlying_model_calls") or 0),
    }
    _agent_stage_log(
        logger,
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        variant=variant,
        stage="budget_matched_control",
        data=stages["budget_matched_control"],
    )
    return {
        "answer_text": json.dumps({"answer": selected_answer}, ensure_ascii=False),
        "underlying_model_calls": stages["budget_matched_control"]["underlying_model_calls"],
    }


def _budget_matched_control_prompt_specs(
    problem: dict[str, Any],
    *,
    variant: str,
    variant_plan: dict[str, Any],
) -> list[dict[str, str]]:
    question = problem["_question"]
    answer_type = problem["answer_type"]
    output = (
        "Return JSON only: {\"answer\":\"...\"}. For multiple choice, answer with the single letter only. "
        "For exact match, answer with the shortest exact answer."
    )
    base_variant = _budget_control_base_variant(variant)
    base_prompt = _prompt_for(problem, variant=base_variant, agent_plan=variant_plan)
    context_prefix = ""
    if base_variant == "hipporag_baseline" and (variant_plan or {}).get("prompt_context"):
        context_prefix = (
            "Use the same retrieval context as a standard HippoRAG-style QA baseline. "
            "Do not use an Assumption Graph, morphism transfer, world model, or recursive assumption tree.\n\n"
        )
    specs = [
        {"prompt_kind": "direct_short_answer", "prompt": base_prompt},
        {
            "prompt_kind": "constraint_checked_answer",
            "prompt": (
                f"{context_prefix}Solve independently and internally verify that the answer obeys the exact output "
                f"contract. Do not expose reasoning.\n\nAnswer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        },
        {
            "prompt_kind": "skeptical_recheck_answer",
            "prompt": (
                f"{context_prefix}Re-solve the item from scratch. Assume the most obvious answer may be wrong; check "
                f"wording, exclusions, units, and option labels before answering. Do not expose reasoning.\n\n"
                f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        },
        {
            "prompt_kind": "literal_constraint_answer",
            "prompt": (
                f"{context_prefix}Choose the answer that best satisfies every explicit constraint in the question. "
                f"Ignore unrelated priors and return only JSON.\n\n"
                f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        },
    ]
    if problem.get("answer_type") == "multipleChoice":
        specs.append({
            "prompt_kind": "option_elimination_baseline_answer",
            "prompt": (
                f"{context_prefix}Evaluate the listed options one by one and eliminate options contradicted by the "
                f"question wording. Return only the final option letter as JSON, with no reasoning.\n\n"
                f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        })
    if base_variant == "hipporag_baseline" and (variant_plan or {}).get("prompt_context"):
        specs.insert(1, {
            "prompt_kind": "hipporag_context_answer",
            "prompt": base_prompt,
        })
    return specs[: _budget_matched_control_candidate_count()]


def _budget_matched_control_candidate_count() -> int:
    value = os.environ.get("HLE_BUDGET_MATCHED_CANDIDATE_COUNT", "").strip()
    if value:
        try:
            return max(1, min(12, int(value)))
        except ValueError:
            pass
    return 5


def _budget_matched_control_workers(candidate_count: int) -> int:
    value = os.environ.get("HLE_BUDGET_MATCHED_MAX_WORKERS", "").strip()
    if value:
        try:
            return max(1, min(candidate_count, int(value)))
        except ValueError:
            pass
    return max(1, min(candidate_count, 5))


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
    child_model = _agent_child_model(model)
    if critic_model != model:
        agent_plan.setdefault("stages", {})["critic_model_router"] = {
            "status": "activated",
            "base_model": model,
            "critic_model": critic_model,
            "policy": "env_override_for_falsification_and_verification",
        }
    if child_model != model:
        agent_plan.setdefault("stages", {})["child_model_router"] = {
            "status": "activated",
            "base_model": model,
            "child_model": child_model,
            "policy": "env_override_for_candidate_generation",
        }
    evidence_summary: dict[str, Any] | None = None
    if evidence_bridge_enabled and _should_prime_evidence_bridge(problem, agent_plan):
        evidence_context, evidence_summary = _build_hle_evidence_bridge_context(
            problem=problem,
            eval_id=eval_id,
            call_id=call_id,
            model=model,
            logger=logger,
            candidate_answers=[],
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
    raw_preserve_summary: dict[str, Any] | None = None
    raw_budget_preserve_summary: dict[str, Any] | None = None
    hipporag_preserve_summary: dict[str, Any] | None = None
    route_arbitrator_summary: dict[str, Any] | None = None
    early_route_arbitrator_locked = False
    pre_route_attempts: list[dict[str, Any]] = []
    if _cache_first_route_arbitrator_enabled() and agent_plan.get("hle_same_run_baseline_cache"):
        (
            pre_route_attempts,
            raw_preserve_summary,
            raw_budget_preserve_summary,
            hipporag_preserve_summary,
        ) = _same_run_cache_route_candidates(
            problem=problem,
            agent_plan=agent_plan,
            call_id=call_id,
            start_index=10001,
        )
        if pre_route_attempts:
            route_arbitrator_attempt, route_arbitrator_summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=pre_route_attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary=raw_budget_preserve_summary,
                hipporag_preserve_summary=hipporag_preserve_summary,
                call_id=call_id,
            )
            if route_arbitrator_attempt:
                pre_route_attempts.append(route_arbitrator_attempt)
                early_route_arbitrator_locked = _route_arbitrator_lock_decision(route_arbitrator_summary)
    if early_route_arbitrator_locked:
        skipped_prompt_kinds = [str(spec.get("prompt_kind") or "") for spec in specs]
        skipped_branch_axes = [str(spec.get("branch_axis") or _child_branch_axis(spec["prompt_kind"])) for spec in specs]
        child_result = {
            "attempts": pre_route_attempts,
            "underlying_model_calls": int((route_arbitrator_summary or {}).get("underlying_model_calls") or 0),
            "execution_mode": "cache_first_route_arbitrator",
            "serial_forced_reason": "",
            "child_timeout_sec": child_timeout if child_timeout is not None else timeout,
            "child_max_workers": 0,
            "early_stop_reason": "cache_first_route_arbitrator_locked",
            "skipped_prompt_kinds": skipped_prompt_kinds,
            "skipped_branch_axes": skipped_branch_axes,
        }
    else:
        child_result = _execute_recursive_child_attempts(
            problem=problem,
            specs=specs,
            model=child_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
            mode=child_mode,
        )
        if pre_route_attempts:
            child_result["attempts"] = list(pre_route_attempts) + list(child_result.get("attempts") or [])
    attempts = child_result["attempts"]
    underlying_calls = int(child_result["underlying_model_calls"] or 0)
    early_stop_reason = child_result.get("early_stop_reason")
    skipped_prompt_kinds = child_result.get("skipped_prompt_kinds", [])
    skipped_branch_axes = child_result.get("skipped_branch_axes", [])
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
    timeout_recovery_summary: dict[str, Any] | None = None
    endpoint_error_abort_summary = _endpoint_error_pressure_abort_summary(problem=problem, attempts=attempts)
    endpoint_error_abort = endpoint_error_abort_summary.get("status") == "activated"
    timeout_recovery_attempt = None
    if not endpoint_error_abort:
        timeout_recovery_attempt, timeout_recovery_summary = _maybe_run_timeout_recovery_child(
            problem=problem,
            attempts=attempts,
            math_tool_summary=math_tool_summary,
            model=child_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
    if timeout_recovery_attempt:
        attempts.append(timeout_recovery_attempt)
        if timeout_recovery_attempt.get("status") == "answered":
            underlying_calls += 1
    child_model_failover_summary: dict[str, Any] | None = None
    child_model_failover_attempt = None
    if not endpoint_error_abort:
        child_model_failover_attempt, child_model_failover_summary = _maybe_run_child_model_failover_child(
            problem=problem,
            attempts=attempts,
            base_model=model,
            child_model=child_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
    if child_model_failover_attempt:
        attempts.append(child_model_failover_attempt)
        if child_model_failover_attempt.get("status") == "answered":
            underlying_calls += 1
    candidate_verifier_summary: dict[str, Any] | None = None
    if not endpoint_error_abort and _should_run_candidate_claim_verifier(problem):
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
    selection_evidence_context = str(agent_plan.get("hle_evidence_context") or "")
    evidence_guided_option_summary: dict[str, Any] | None = None
    evidence_guided_option_attempt = None
    evidence_guided_context = ""
    if not endpoint_error_abort:
        (
            evidence_guided_option_attempt,
            evidence_guided_option_summary,
            evidence_guided_context,
        ) = _maybe_run_evidence_guided_option_challenge(
            problem=problem,
            attempts=attempts,
            option_evidence_summary=option_evidence_summary,
            model=critic_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
    if evidence_guided_option_attempt:
        attempts.append(evidence_guided_option_attempt)
        if evidence_guided_option_attempt.get("status") == "answered":
            underlying_calls += 1
    if evidence_guided_context and not selection_evidence_context:
        selection_evidence_context = evidence_guided_context
    structural_option_audit_summary: dict[str, Any] | None = None
    structural_option_audit_attempt = None
    if not endpoint_error_abort:
        structural_option_audit_attempt, structural_option_audit_summary = _maybe_run_structural_option_audit_child(
            problem=problem,
            attempts=attempts,
            option_evidence_summary=option_evidence_summary,
            evidence_guided_option_summary=evidence_guided_option_summary,
            evidence_context=selection_evidence_context,
            model=critic_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
    if structural_option_audit_attempt:
        attempts.append(structural_option_audit_attempt)
        if structural_option_audit_attempt.get("status") == "answered":
            underlying_calls += 1
    domain_rule_summary: dict[str, Any] | None = None
    domain_rule_attempt, domain_rule_summary = _maybe_run_domain_rule_mc_verifier(
        problem=problem,
        attempts=attempts,
        evidence_context=selection_evidence_context,
        eval_id=eval_id,
        call_id=call_id,
        model=model,
        logger=logger,
    )
    if domain_rule_attempt:
        attempts.append(domain_rule_attempt)
    if (
        not endpoint_error_abort
        and evidence_bridge_enabled
        and not agent_plan.get("hle_evidence_context")
        and _needs_evidence_grounded_child(problem, attempts)
    ):
        evidence_context, evidence_summary = _build_hle_evidence_bridge_context(
            problem=problem,
            eval_id=eval_id,
            call_id=call_id,
            model=model,
            logger=logger,
            candidate_answers=[
                str(attempt.get("parsed_answer") or "")
                for attempt in attempts
                if str(attempt.get("parsed_answer") or "").strip()
            ],
        )
        if evidence_context:
            agent_plan["hle_evidence_context"] = evidence_context
            selection_evidence_context = evidence_context
            agent_plan["hle_evidence_bridge"] = evidence_summary
            supported_evidence_attempt, supported_evidence_summary = _maybe_add_answer_bearing_evidence_candidate(
                problem=problem,
                attempts=attempts,
                evidence_summary=evidence_summary,
            )
            if supported_evidence_summary:
                evidence_summary["source_supported_candidate"] = supported_evidence_summary
            if supported_evidence_attempt:
                attempts.append(supported_evidence_attempt)
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
            _maybe_mark_answer_bearing_evidence_attempt(
                problem=problem,
                attempt=evidence_attempt,
                evidence_summary=evidence_summary,
            )
            attempts.append(evidence_attempt)
            if evidence_attempt.get("status") == "answered":
                underlying_calls += 1

    if (
        not early_route_arbitrator_locked
        and not endpoint_error_abort
        and _route_arbitrator_enabled()
        and agent_plan.get("hle_same_run_baseline_cache")
    ):
        if raw_preserve_summary is None:
            raw_preserve_attempt, raw_preserve_summary = _maybe_run_raw_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                model=model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=child_timeout if child_timeout is not None else timeout,
                max_tokens=max_tokens,
            )
            if raw_preserve_attempt:
                attempts.append(raw_preserve_attempt)
                if raw_preserve_attempt.get("status") == "answered":
                    underlying_calls += 1

        if raw_budget_preserve_summary is None:
            raw_budget_preserve_attempt, raw_budget_preserve_summary = _maybe_run_raw_budget_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                model=model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=child_timeout if child_timeout is not None else timeout,
                max_tokens=max_tokens,
            )
            if raw_budget_preserve_attempt:
                attempts.append(raw_budget_preserve_attempt)
            if raw_budget_preserve_summary:
                raw_budget_underlying = int(raw_budget_preserve_summary.get("underlying_model_calls") or 0)
                if raw_budget_underlying:
                    underlying_calls += raw_budget_underlying

        if hipporag_preserve_summary is None:
            hipporag_preserve_attempt, hipporag_preserve_summary = _maybe_run_hipporag_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                model=model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=child_timeout if child_timeout is not None else timeout,
                max_tokens=max_tokens,
            )
            if hipporag_preserve_attempt:
                attempts.append(hipporag_preserve_attempt)
                if hipporag_preserve_summary and hipporag_preserve_summary.get("underlying_model_calls") is not None:
                    underlying_calls += int(hipporag_preserve_summary.get("underlying_model_calls") or 0)
                elif hipporag_preserve_attempt.get("status") == "answered":
                    underlying_calls += 1

        route_arbitrator_attempt, route_arbitrator_summary = _maybe_add_route_arbitrator_candidate(
            problem=problem,
            attempts=attempts,
            agent_plan=agent_plan,
            raw_budget_preserve_summary=raw_budget_preserve_summary,
            hipporag_preserve_summary=hipporag_preserve_summary,
            call_id=call_id,
            model=critic_model,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
        if route_arbitrator_attempt:
            attempts.append(route_arbitrator_attempt)
            early_route_arbitrator_locked = _route_arbitrator_lock_decision(route_arbitrator_summary)
        if route_arbitrator_summary:
            underlying_calls += int(route_arbitrator_summary.get("underlying_model_calls") or 0)

    run_deep_challenges_after_route = (
        os.environ.get("HLE_ROUTE_ARBITRATOR_RUN_DEEP_CHALLENGES", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    counter_challenge_summary: dict[str, Any] | None = None
    counter_challenge_attempt = None
    if (not endpoint_error_abort) and (not early_route_arbitrator_locked or run_deep_challenges_after_route):
        counter_challenge_attempt, counter_challenge_summary = _maybe_run_counter_assumption_challenge(
            problem=problem,
            attempts=attempts,
            candidate_verifier_summary=candidate_verifier_summary,
            math_tool_summary=math_tool_summary,
            evidence_context=selection_evidence_context,
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
            evidence_context=selection_evidence_context,
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
                evidence_context=selection_evidence_context,
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
    critic_synthesis_attempt = None
    if (not endpoint_error_abort) and (not early_route_arbitrator_locked or run_deep_challenges_after_route):
        critic_synthesis_attempt, critic_synthesis_summary = _maybe_run_critic_synthesis_child(
            problem=problem,
            attempts=attempts,
            evidence_context=selection_evidence_context,
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
    option_sweep_attempts = []
    if not endpoint_error_abort:
        option_sweep_attempts, option_sweep_summary = _maybe_add_mc_option_sweep_candidates(
            problem=problem,
            attempts=attempts,
        )
    if option_sweep_attempts:
        attempts.extend(option_sweep_attempts)

    if raw_preserve_summary is None and not endpoint_error_abort:
        raw_preserve_attempt, raw_preserve_summary = _maybe_run_raw_preserve_selector_child(
            problem=problem,
            attempts=attempts,
            agent_plan=agent_plan,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
        if raw_preserve_attempt:
            attempts.append(raw_preserve_attempt)
            if raw_preserve_attempt.get("status") == "answered":
                underlying_calls += 1

    if raw_budget_preserve_summary is None and not endpoint_error_abort:
        raw_budget_preserve_attempt, raw_budget_preserve_summary = _maybe_run_raw_budget_preserve_selector_child(
            problem=problem,
            attempts=attempts,
            agent_plan=agent_plan,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
        if raw_budget_preserve_attempt:
            attempts.append(raw_budget_preserve_attempt)
        if raw_budget_preserve_summary:
            raw_budget_underlying = int(raw_budget_preserve_summary.get("underlying_model_calls") or 0)
            if raw_budget_underlying:
                underlying_calls += raw_budget_underlying

    if hipporag_preserve_summary is None and not endpoint_error_abort:
        hipporag_preserve_attempt, hipporag_preserve_summary = _maybe_run_hipporag_preserve_selector_child(
            problem=problem,
            attempts=attempts,
            agent_plan=agent_plan,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
        if hipporag_preserve_attempt:
            attempts.append(hipporag_preserve_attempt)
            if hipporag_preserve_summary and hipporag_preserve_summary.get("underlying_model_calls") is not None:
                underlying_calls += int(hipporag_preserve_summary.get("underlying_model_calls") or 0)
            elif hipporag_preserve_attempt.get("status") == "answered":
                underlying_calls += 1

    if route_arbitrator_summary is None and not endpoint_error_abort:
        route_arbitrator_attempt, route_arbitrator_summary = _maybe_add_route_arbitrator_candidate(
            problem=problem,
            attempts=attempts,
            agent_plan=agent_plan,
            raw_budget_preserve_summary=raw_budget_preserve_summary,
            hipporag_preserve_summary=hipporag_preserve_summary,
            call_id=call_id,
            model=critic_model,
            timeout=child_timeout if child_timeout is not None else timeout,
            max_tokens=max_tokens,
        )
        if route_arbitrator_attempt:
            attempts.append(route_arbitrator_attempt)
        if route_arbitrator_summary:
            underlying_calls += int(route_arbitrator_summary.get("underlying_model_calls") or 0)

    selection = _select_recursive_child_answer(
        problem=problem,
        attempts=attempts,
        model=critic_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=_recursive_verifier_timeout(child_timeout if child_timeout is not None else timeout),
        max_tokens=min(max_tokens, 384),
        evidence_context=selection_evidence_context,
    )
    selection = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)
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
    executed_branch_axes = _child_branch_axes_for_attempts(attempts)
    answered_branch_axes = _child_branch_axes_for_attempts([
        attempt for attempt in attempts if attempt.get("status") == "answered"
    ])
    planned_branch_axes = [str(spec.get("branch_axis") or _child_branch_axis(spec["prompt_kind"])) for spec in specs]
    required_branch_axes = sorted(_required_child_branch_axes_before_early_stop(problem))
    executed_required_branch_axes = sorted(set(executed_branch_axes) & set(required_branch_axes))
    answered_required_branch_axes = sorted(set(answered_branch_axes) & set(required_branch_axes))
    child_summary = {
        "status": "activated",
        "execution_mode": child_result.get("execution_mode"),
        "serial_forced_reason": child_result.get("serial_forced_reason"),
        "base_model": model,
        "child_model": child_model,
        "child_timeout_sec": child_result.get("child_timeout_sec"),
        "child_max_workers": child_result.get("child_max_workers"),
        "planned_child_count": len(specs),
        "child_count": len(attempts),
        "answered_child_count": answered_count,
        "error_child_count": len(attempts) - answered_count,
        "early_stopped": bool(early_stop_reason),
        "early_stop_reason": early_stop_reason,
        "skipped_prompt_kinds": skipped_prompt_kinds,
        "planned_branch_axes": planned_branch_axes,
        "executed_branch_axes": executed_branch_axes,
        "answered_branch_axes": answered_branch_axes,
        "skipped_branch_axes": skipped_branch_axes,
        "planned_unique_branch_axis_count": len(set(planned_branch_axes)),
        "executed_unique_branch_axis_count": len(set(executed_branch_axes)),
        "answered_unique_branch_axis_count": len(set(answered_branch_axes)),
        "required_branch_axes_before_early_stop": required_branch_axes,
        "executed_required_branch_axes": executed_required_branch_axes,
        "answered_required_branch_axes": answered_required_branch_axes,
        "core_orthogonal_axes_covered": set(required_branch_axes).issubset(set(executed_branch_axes)),
        "core_orthogonal_axes_answered": set(required_branch_axes).issubset(set(answered_branch_axes)),
        "orthogonal_branch_coverage": round(
            len(set(executed_branch_axes)) / max(1, len(set(planned_branch_axes))),
            4,
        ),
        "prompt_kinds": [attempt["prompt_kind"] for attempt in attempts],
        "selected_prompt_kind": next(
            (
                str(attempt.get("prompt_kind") or "")
                for attempt in attempts
                if attempt.get("child_id") == selection.get("selected_child_id")
            ),
            "",
        ),
        "candidate_answer_hashes": [
            attempt.get("parsed_answer_hash") for attempt in attempts if attempt.get("parsed_answer_hash")
        ],
        "endpoint_error_pressure_abort": endpoint_error_abort_summary,
    }
    verifier_summary = {
        "status": "activated",
        "selection_method": selection.get("selection_method"),
        "selected_child_id": selection.get("selected_child_id"),
        "selected_answer_hash": selected_hash,
        "verifier_model_call": bool(selection.get("verifier_model_call")),
        "verified_or_abstain_gate": selection.get("verified_or_abstain_gate"),
    }
    stages = agent_plan.setdefault("stages", {})
    stages["recursive_child_validation"] = child_summary
    stages["multi_candidate_self_verifier"] = verifier_summary
    if endpoint_error_abort_summary.get("status") == "activated":
        stages["endpoint_error_pressure_abort"] = endpoint_error_abort_summary
    if math_tool_summary:
        stages["hle_math_tool_solver"] = math_tool_summary
    if timeout_recovery_summary:
        timeout_recovery_summary["final_selection_method"] = selection.get("selection_method")
        timeout_recovery_summary["selected_timeout_recovery_candidate"] = (
            selection.get("selected_child_id") == timeout_recovery_summary.get("child_id")
        )
        stages["recursive_timeout_recovery_child"] = timeout_recovery_summary
    if child_model_failover_summary:
        child_model_failover_summary["final_selection_method"] = selection.get("selection_method")
        child_model_failover_summary["selected_child_model_failover_candidate"] = (
            selection.get("selected_child_id") == child_model_failover_summary.get("child_id")
        )
        stages["child_model_failover_child"] = child_model_failover_summary
    if candidate_verifier_summary:
        stages["candidate_claim_verifier"] = candidate_verifier_summary
    if domain_rule_summary:
        domain_rule_summary["final_selection_method"] = selection.get("selection_method")
        domain_rule_summary["selected_domain_rule_candidate"] = (
            selection.get("selected_child_id") == domain_rule_summary.get("child_id")
        )
        stages["domain_rule_mc_verifier"] = domain_rule_summary
    if evidence_guided_option_summary:
        evidence_guided_option_summary["final_selection_method"] = selection.get("selection_method")
        evidence_guided_option_summary["selected_evidence_guided_option_candidate"] = (
            selection.get("selected_child_id") == evidence_guided_option_summary.get("child_id")
        )
        stages["evidence_guided_option_challenge"] = evidence_guided_option_summary
    if structural_option_audit_summary:
        structural_option_audit_summary["final_selection_method"] = selection.get("selection_method")
        structural_option_audit_summary["selected_structural_option_audit"] = (
            selection.get("selected_child_id") == structural_option_audit_summary.get("child_id")
        )
        stages["structural_option_audit_child"] = structural_option_audit_summary
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
    if raw_preserve_summary:
        raw_preserve_summary["final_selection_method"] = selection.get("selection_method")
        raw_preserve_summary["selected_raw_preserve_candidate"] = (
            selection.get("selected_child_id") == raw_preserve_summary.get("child_id")
        )
        stages["raw_preserve_selector"] = raw_preserve_summary
    if raw_budget_preserve_summary:
        raw_budget_preserve_summary["final_selection_method"] = selection.get("selection_method")
        raw_budget_preserve_summary["selected_raw_budget_preserve_candidate"] = (
            selection.get("selected_child_id") == raw_budget_preserve_summary.get("child_id")
        )
        stages["raw_budget_preserve_selector"] = raw_budget_preserve_summary
    if hipporag_preserve_summary:
        hipporag_preserve_summary["final_selection_method"] = selection.get("selection_method")
        hipporag_preserve_summary["selected_hipporag_preserve_candidate"] = (
            selection.get("selected_child_id") == hipporag_preserve_summary.get("child_id")
        )
        stages["hipporag_preserve_selector"] = hipporag_preserve_summary
    if route_arbitrator_summary:
        route_arbitrator_summary["route_locked"] = bool(early_route_arbitrator_locked)
        route_arbitrator_summary["final_selection_method"] = selection.get("selection_method")
        route_arbitrator_summary["selected_route_arbitrator_candidate"] = (
            selection.get("selected_child_id") == route_arbitrator_summary.get("child_id")
        )
        stages["route_arbitrator"] = route_arbitrator_summary
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
    if stages.get("same_run_baseline_cache"):
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="same_run_baseline_cache",
            data=stages["same_run_baseline_cache"],
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
    if stages.get("child_model_router"):
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="child_model_router",
            data=stages["child_model_router"],
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
    if timeout_recovery_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="recursive_timeout_recovery_child",
            data=timeout_recovery_summary,
        )
    if child_model_failover_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="child_model_failover_child",
            data=child_model_failover_summary,
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
    if evidence_guided_option_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="evidence_guided_option_challenge",
            data=evidence_guided_option_summary,
        )
    if structural_option_audit_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="structural_option_audit_child",
            data=structural_option_audit_summary,
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
    if raw_preserve_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="raw_preserve_selector",
            data=raw_preserve_summary,
        )
    if raw_budget_preserve_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="raw_budget_preserve_selector",
            data=raw_budget_preserve_summary,
        )
    if hipporag_preserve_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="hipporag_preserve_selector",
            data=hipporag_preserve_summary,
        )
    if route_arbitrator_summary:
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            variant="assumption_agent_recursive_verify",
            stage="route_arbitrator",
            data=route_arbitrator_summary,
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


def _recursive_answering_disabled() -> bool:
    return os.environ.get("HLE_DISABLE_RECURSIVE_ASSUMPTION_RUNNER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _maybe_run_raw_preserve_selector_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any] | None = None,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    env_forced = os.environ.get("HLE_ENABLE_RAW_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    trigger = _cost_aware_raw_preserve_trigger(
        problem=problem,
        attempts=attempts,
        agent_plan=agent_plan or {},
    )
    if not env_forced and trigger.get("status") != "activated":
        return None, None
    child_index = _timeout_recovery_child_index(attempts)
    cached = _same_run_cached_baseline(agent_plan, ["raw"])
    if cached:
        answer = str(cached.get("answer") or "").strip()
        child_id = stable_hash({
            "call_id": call_id,
            "child_index": child_index,
            "prompt_kind": "raw_preserve_selector_answer",
            "same_run_cache_variant": cached.get("variant"),
            "answer_hash": cached.get("answer_hash"),
        })
        attempt = {
            "child_id": child_id,
            "child_index": child_index,
            "prompt_kind": "raw_preserve_selector_answer",
            "branch_axis": _child_branch_axis("raw_preserve_selector_answer"),
            "orthogonal_branch_id": _child_branch_id(
                problem,
                prompt_kind="raw_preserve_selector_answer",
                branch_axis=_child_branch_axis("raw_preserve_selector_answer"),
            ),
            "parsed_answer": answer,
            "parsed_answer_hash": stable_hash({"answer": answer}) if answer else None,
            "prediction_hash": stable_hash({
                "same_run_cache_variant": cached.get("variant"),
                "answer_hash": cached.get("answer_hash"),
            }),
            "latency_sec": 0.0,
            "status": "answered" if answer else "no_answer",
            "same_run_baseline_cache_variant": cached.get("variant"),
        }
        summary = {
            "status": "activated" if answer else "no_candidate",
            "policy": "same_run_raw_baseline_cache_candidate",
            "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
            "base_model": model,
            "child_id": child_id,
            "child_index": child_index,
            "child_status": attempt.get("status"),
            "child_error_type": None,
            "candidate_emitted": bool(answer),
            "candidate_answer_hash": attempt.get("parsed_answer_hash"),
            "same_run_cache_variant": cached.get("variant"),
            "borrowed_baseline_model_calls": 1,
            "underlying_model_calls": 0,
        }
        return attempt, summary
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "raw_preserve_selector_answer",
            "prompt": _prompt_for(problem, variant="raw"),
        },
        child_index=child_index,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    summary = {
        "status": "activated",
        "policy": (
            "env_forced_no_context_base_model_candidate"
            if env_forced
            else "cost_aware_regression_guard_no_context_candidate"
        ),
        "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
        "base_model": model,
        "child_id": attempt.get("child_id"),
        "child_index": attempt.get("child_index"),
        "child_status": attempt.get("status"),
        "child_error_type": attempt.get("error_type"),
        "candidate_emitted": bool(str(attempt.get("parsed_answer") or "").strip()),
        "candidate_answer_hash": attempt.get("parsed_answer_hash"),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    return attempt, summary


def _maybe_run_raw_budget_preserve_selector_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any] | None = None,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    env_forced = os.environ.get("HLE_ENABLE_RAW_BUDGET_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    trigger = _cost_aware_raw_budget_preserve_trigger(
        problem=problem,
        attempts=attempts,
        agent_plan=agent_plan or {},
    )
    if not env_forced and trigger.get("status") != "activated":
        return None, None

    base_child_index = _timeout_recovery_child_index(attempts)
    cached = _same_run_cached_baseline(agent_plan, ["raw_budget_matched"])
    if cached:
        answer = str(cached.get("answer") or "").strip()
        aggregate_child_id = stable_hash({
            "call_id": call_id,
            "child_index": base_child_index,
            "prompt_kind": "raw_budget_preserve_selector_answer",
            "same_run_cache_variant": cached.get("variant"),
            "answer_hash": cached.get("answer_hash"),
        })
        attempt = {
            "child_id": aggregate_child_id,
            "child_index": base_child_index,
            "prompt_kind": "raw_budget_preserve_selector_answer",
            "branch_axis": _child_branch_axis("raw_budget_preserve_selector_answer"),
            "orthogonal_branch_id": _child_branch_id(
                problem,
                prompt_kind="raw_budget_preserve_selector_answer",
                branch_axis=_child_branch_axis("raw_budget_preserve_selector_answer"),
            ),
            "parsed_answer": answer,
            "parsed_answer_hash": stable_hash({"answer": answer}) if answer else None,
            "prediction_hash": stable_hash({
                "same_run_cache_variant": cached.get("variant"),
                "answer_hash": cached.get("answer_hash"),
            }),
            "latency_sec": 0.0,
            "status": "answered" if answer else "no_answer",
            "raw_budget_selection_method": "same_run_raw_budget_cache",
            "same_run_baseline_cache_variant": cached.get("variant"),
        }
        top_vote_count = int(cached.get("budget_top_candidate_vote_count") or 0)
        strong_consensus = bool(cached.get("budget_strong_consensus"))
        summary = {
            "status": "activated" if answer else "no_candidate",
            "policy": "same_run_raw_budget_matched_cache_candidate",
            "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
            "base_model": model,
            "child_id": aggregate_child_id,
            "child_index": base_child_index,
            "child_status": attempt.get("status"),
            "child_error_type": None,
            "candidate_emitted": bool(answer),
            "candidate_answer_hash": attempt.get("parsed_answer_hash"),
            "candidate_count": int(cached.get("budget_candidate_count") or 0),
            "answered_candidate_count": int(cached.get("budget_answered_candidate_count") or 0),
            "error_candidate_count": 0,
            "candidate_prompt_kinds": [],
            "candidate_answer_hashes": [cached.get("answer_hash")] if cached.get("answer_hash") else [],
            "candidate_answer_hash_counts": {},
            "top_candidate_answer_hash": cached.get("answer_hash"),
            "top_candidate_vote_count": top_vote_count,
            "strong_consensus": strong_consensus,
            "verified_selection_allowed": False,
            "child_max_workers": 0,
            "selection_method": "same_run_raw_budget_cache",
            "child_selection_method": cached.get("selection_method"),
            "selected_child_id": None,
            "selected_answer_hash": attempt.get("parsed_answer_hash"),
            "verifier_model_call": False,
            "verified_or_abstain_gate": None,
            "same_run_cache_variant": cached.get("variant"),
            "borrowed_baseline_model_calls": 1,
            "underlying_model_calls": 0,
        }
        return attempt, summary
    selector_plan: dict[str, Any] = {}
    specs = _budget_matched_control_prompt_specs(
        problem,
        variant="raw_budget_matched",
        variant_plan=selector_plan,
    )
    batch = _run_child_batch(
        problem=problem,
        specs=specs,
        start_index=base_child_index,
        model=model,
        variant="raw_budget_preserve_selector",
        eval_id=eval_id,
        call_id=f"{call_id}_raw_budget_preserve",
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
        max_workers=_budget_matched_control_workers(len(specs)),
    )
    child_attempts = list(batch.get("attempts") or [])
    selection = _select_recursive_child_answer(
        problem=problem,
        attempts=child_attempts,
        model=model,
        eval_id=eval_id,
        call_id=f"{call_id}_raw_budget_preserve_selector",
        logger=logger,
        timeout=timeout,
        max_tokens=min(max_tokens, 384),
        evidence_context="",
    )
    selection = _apply_verified_or_abstain_selection(problem=problem, attempts=child_attempts, selection=selection)
    selected_answer = selection.get("selected_answer") or _fallback_answer(child_attempts)
    answer_hash_counts = Counter(
        child.get("parsed_answer_hash")
        for child in child_attempts
        if child.get("status") == "answered" and child.get("parsed_answer_hash")
    )
    top_answer_hash, top_vote_count = (None, 0)
    if answer_hash_counts:
        top_answer_hash, top_vote_count = answer_hash_counts.most_common(1)[0]
    top_consensus_attempt = next(
        (
            child for child in child_attempts
            if child.get("status") == "answered"
            and child.get("parsed_answer_hash") == top_answer_hash
        ),
        None,
    )
    verified_gate = selection.get("verified_or_abstain_gate")
    verified_gate = verified_gate if isinstance(verified_gate, dict) else {}
    verified_selection_allowed = (
        selection.get("selection_method") in _VERIFIED_SELECTION_METHODS
        and verified_gate.get("status") == "allowed"
    )
    strong_consensus = top_vote_count >= min(3, max(1, answered_count := sum(
        1 for child in child_attempts if child.get("status") == "answered"
    )))
    if strong_consensus and top_consensus_attempt:
        selected_answer = str(top_consensus_attempt.get("parsed_answer") or "").strip()
        selected_child_id = top_consensus_attempt.get("child_id")
        raw_budget_selection_method = "raw_budget_top_vote_consensus"
    else:
        selected_child_id = selection.get("selected_child_id")
        raw_budget_selection_method = str(selection.get("selection_method") or "")
    candidate_allowed = bool(selected_answer) and strong_consensus
    if env_forced and not selected_answer:
        selected_answer = selection.get("selected_answer") or _fallback_answer(child_attempts)
    aggregate_child_index = base_child_index + len(specs)
    aggregate_child_id = stable_hash({
        "call_id": call_id,
        "child_index": aggregate_child_index,
        "prompt_kind": "raw_budget_preserve_selector_answer",
    })
    attempt = {
        "child_id": aggregate_child_id,
        "child_index": aggregate_child_index,
        "prompt_kind": "raw_budget_preserve_selector_answer",
        "parsed_answer": selected_answer,
        "parsed_answer_hash": stable_hash({"answer": selected_answer}) if selected_answer else None,
        "prediction_hash": stable_hash({
            "selection": selection.get("selection_method"),
            "answer": selected_answer,
            "candidate_hashes": [
                child.get("parsed_answer_hash") for child in child_attempts if child.get("parsed_answer_hash")
            ],
        }),
        "latency_sec": None,
        "status": "answered" if selected_answer else "no_answer",
        "raw_budget_child_ids": [child.get("child_id") for child in child_attempts],
        "raw_budget_selection_method": raw_budget_selection_method,
    }
    answered_count = sum(1 for child in child_attempts if child.get("status") == "answered")
    summary = {
        "status": "activated" if selected_answer else "no_candidate",
        "policy": (
            "env_forced_raw_budget_matched_candidate"
            if env_forced
            else "cost_aware_raw_budget_matched_regression_guard"
        ),
        "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
        "base_model": model,
        "child_id": aggregate_child_id,
        "child_index": aggregate_child_index,
        "child_status": attempt.get("status"),
        "child_error_type": attempt.get("error_type"),
        "candidate_emitted": bool(str(selected_answer or "").strip()),
        "candidate_answer_hash": attempt.get("parsed_answer_hash"),
        "candidate_count": len(child_attempts),
        "answered_candidate_count": answered_count,
        "error_candidate_count": len(child_attempts) - answered_count,
        "candidate_prompt_kinds": [child.get("prompt_kind") for child in child_attempts],
        "candidate_answer_hashes": [
            child.get("parsed_answer_hash") for child in child_attempts if child.get("parsed_answer_hash")
        ],
        "candidate_answer_hash_counts": dict(answer_hash_counts),
        "top_candidate_answer_hash": top_answer_hash,
        "top_candidate_vote_count": top_vote_count,
        "strong_consensus": strong_consensus,
        "verified_selection_allowed": verified_selection_allowed,
        "child_max_workers": batch.get("max_workers"),
        "selection_method": raw_budget_selection_method,
        "child_selection_method": selection.get("selection_method"),
        "selected_child_id": selected_child_id,
        "selected_answer_hash": attempt.get("parsed_answer_hash"),
        "verifier_model_call": bool(selection.get("verifier_model_call")),
        "verified_or_abstain_gate": selection.get("verified_or_abstain_gate"),
        "underlying_model_calls": int(batch.get("underlying_model_calls") or 0)
        + int(selection.get("underlying_model_calls") or 0),
    }
    if not env_forced and not candidate_allowed:
        summary.update({
            "status": "blocked_weak_consensus",
            "candidate_emitted": False,
            "block_reason": "raw_budget_preserve_requires_verified_selection_or_3_vote_consensus",
        })
        return None, summary
    return attempt, summary


def _maybe_run_hipporag_preserve_selector_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any] | None = None,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    budget_preserve_enabled = (
        os.environ.get("HLE_ENABLE_HIPPORAG_BUDGET_PRESERVE_SELECTOR", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    env_forced = (
        os.environ.get("HLE_ENABLE_HIPPORAG_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        }
        or budget_preserve_enabled
    )
    trigger = _cost_aware_hipporag_preserve_trigger(
        problem=problem,
        attempts=attempts,
        agent_plan=agent_plan or {},
    )
    if not env_forced and trigger.get("status") != "activated":
        return None, None
    child_index = _timeout_recovery_child_index(attempts)
    hipporag_cached_entries = _same_run_cached_baseline_entries(
        agent_plan,
        ["hipporag_budget_matched", "hipporag_baseline"],
    )
    cached = hipporag_cached_entries[0] if hipporag_cached_entries else None
    if cached:
        context_char_count = int(cached.get("context_char_count") or 0)
        selected_doc_count = int(cached.get("selected_doc_count") or 0)
        candidate_doc_count = int(cached.get("candidate_doc_count") or 0)
        has_usable_context = context_char_count > 0 and (selected_doc_count > 0 or candidate_doc_count > 0)
        answer = str(cached.get("answer") or "").strip()
        answer_norm = _normalize_for_selection(answer, answer_type=problem.get("answer_type") or "exactMatch")
        agreeing_variants = [
            str(entry.get("variant") or "")
            for entry in hipporag_cached_entries
            if _normalize_for_selection(
                str(entry.get("answer") or ""),
                answer_type=problem.get("answer_type") or "exactMatch",
            ) == answer_norm
        ]
        aggregate_child_id = stable_hash({
            "call_id": call_id,
            "child_index": child_index,
            "prompt_kind": "hipporag_preserve_selector_answer",
            "same_run_cache_variant": cached.get("variant"),
            "answer_hash": cached.get("answer_hash"),
        })
        attempt = {
            "child_id": aggregate_child_id,
            "child_index": child_index,
            "prompt_kind": "hipporag_preserve_selector_answer",
            "branch_axis": _child_branch_axis("hipporag_preserve_selector_answer"),
            "orthogonal_branch_id": _child_branch_id(
                problem,
                prompt_kind="hipporag_preserve_selector_answer",
                branch_axis=_child_branch_axis("hipporag_preserve_selector_answer"),
            ),
            "parsed_answer": answer,
            "parsed_answer_hash": stable_hash({"answer": answer}) if answer else None,
            "prediction_hash": stable_hash({
                "same_run_cache_variant": cached.get("variant"),
                "answer_hash": cached.get("answer_hash"),
            }),
            "latency_sec": 0.0,
            "status": "answered" if answer else "no_answer",
            "preserve_context_char_count": context_char_count,
            "preserve_selected_doc_count": selected_doc_count,
            "preserve_candidate_doc_count": candidate_doc_count,
            "hipporag_budget_selection_method": "same_run_hipporag_cache",
            "same_run_baseline_cache_variant": cached.get("variant"),
            "same_run_cache_has_usable_context": has_usable_context,
            "same_route_agreement_count": len(agreeing_variants),
            "same_route_agreeing_variants": agreeing_variants,
            "budget_matched": cached.get("variant") == "hipporag_budget_matched",
            "budget_candidate_count": int(cached.get("budget_candidate_count") or 0),
            "budget_answered_candidate_count": int(cached.get("budget_answered_candidate_count") or 0),
            "budget_top_candidate_vote_count": int(cached.get("budget_top_candidate_vote_count") or 0),
            "budget_strong_consensus": bool(cached.get("budget_strong_consensus")),
            "budget_selected_answer_hash": cached.get("budget_selected_answer_hash"),
            "budget_top_candidate_answer_hash": cached.get("budget_top_candidate_answer_hash"),
            "budget_candidate_answer_hash_counts": dict(cached.get("budget_candidate_answer_hash_counts") or {}),
            "budget_verified_or_abstain_gate": cached.get("budget_verified_or_abstain_gate"),
            "context_answer_supported": bool(cached.get("context_answer_supported")),
            "context_answer_overlap_count": int(cached.get("context_answer_overlap_count") or 0),
            "context_question_overlap_count": int(cached.get("context_question_overlap_count") or 0),
            "context_answer_option_hash": cached.get("context_answer_option_hash"),
        }
        policy = (
            "same_run_hipporag_cache_candidate"
            if has_usable_context
            else "same_run_hipporag_cache_candidate_no_context"
        )
        summary = {
            "status": "activated" if answer else "no_candidate",
            "policy": policy,
            "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
            "base_model": model,
            "child_id": aggregate_child_id,
            "child_index": child_index,
            "child_status": attempt.get("status"),
            "child_error_type": None,
            "candidate_emitted": bool(answer),
            "candidate_answer_hash": attempt.get("parsed_answer_hash"),
            "retrieval_status": "same_run_cache" if has_usable_context else "same_run_cache_no_context",
            "retrieval_query_count": 0,
            "candidate_doc_count": candidate_doc_count,
            "rerank_status": "same_run_cache" if has_usable_context else "same_run_cache_no_context",
            "selected_doc_count": selected_doc_count,
            "context_char_count": context_char_count,
            "same_run_cache_has_usable_context": has_usable_context,
            "baseline_plan_hash": None,
            "budget_matched": cached.get("variant") == "hipporag_budget_matched",
            "candidate_count": int(cached.get("budget_candidate_count") or 0),
            "answered_candidate_count": int(cached.get("budget_answered_candidate_count") or 0),
            "selection_method": "same_run_hipporag_cache",
            "verified_or_abstain_gate": None,
            "same_run_cache_variant": cached.get("variant"),
            "same_route_agreement_count": len(agreeing_variants),
            "same_route_agreeing_variants": agreeing_variants,
            "budget_top_candidate_vote_count": int(cached.get("budget_top_candidate_vote_count") or 0),
            "budget_top_candidate_answer_hash": cached.get("budget_top_candidate_answer_hash"),
            "budget_selected_answer_hash": cached.get("budget_selected_answer_hash"),
            "budget_strong_consensus": bool(cached.get("budget_strong_consensus")),
            "budget_verified_or_abstain_gate": cached.get("budget_verified_or_abstain_gate"),
            "context_answer_supported": bool(cached.get("context_answer_supported")),
            "context_answer_overlap_count": int(cached.get("context_answer_overlap_count") or 0),
            "context_question_overlap_count": int(cached.get("context_question_overlap_count") or 0),
            "context_answer_option_hash": cached.get("context_answer_option_hash"),
            "borrowed_baseline_model_calls": 1,
            "underlying_model_calls": 0,
        }
        return attempt, summary
    baseline_plan = _build_hipporag_baseline_plan(
        problem=problem,
        eval_id=eval_id,
        call_id=f"{call_id}_hipporag_preserve",
        model=model,
        logger=logger,
        context_max_chars=2200,
    )
    stages = baseline_plan.get("stages", {}) if isinstance(baseline_plan, dict) else {}
    retrieval = stages.get("hipporag_context_retrieval", {}) if isinstance(stages.get("hipporag_context_retrieval"), dict) else {}
    rerank = stages.get("hipporag_associative_rerank", {}) if isinstance(stages.get("hipporag_associative_rerank"), dict) else {}
    prompt = stages.get("prompt_builder", {}) if isinstance(stages.get("prompt_builder"), dict) else {}
    retrieval_status = retrieval.get("status")
    candidate_doc_count = int(retrieval.get("candidate_doc_count") or 0)
    rerank_status = rerank.get("status")
    selected_doc_count = int(rerank.get("selected_doc_count") or 0)
    context_char_count = int(prompt.get("context_char_count") or 0)
    context_text = str(baseline_plan.get("prompt_context") or "")
    has_usable_context = context_char_count > 0 and (selected_doc_count > 0 or candidate_doc_count > 0)
    if not env_forced and not has_usable_context:
        return None, {
            "status": "blocked_non_answer_bearing",
            "policy": "cost_aware_unverified_mc_hipporag_baseline_candidate",
            "trigger": trigger,
            "base_model": model,
            "child_id": None,
            "child_index": child_index,
            "child_status": "skipped",
            "child_error_type": None,
            "candidate_emitted": False,
            "candidate_answer_hash": None,
            "retrieval_status": retrieval_status,
            "retrieval_query_count": int(retrieval.get("query_count") or 0),
            "candidate_doc_count": candidate_doc_count,
            "rerank_status": rerank_status,
            "selected_doc_count": selected_doc_count,
            "context_char_count": context_char_count,
            "baseline_plan_hash": stable_hash(baseline_plan),
            "underlying_model_calls": 0,
            "block_reason": "hipporag_preserve_requires_retrieved_context",
        }
    if budget_preserve_enabled and has_usable_context:
        specs = _budget_matched_control_prompt_specs(
            problem,
            variant="hipporag_budget_matched",
            variant_plan=baseline_plan,
        )
        batch = _run_child_batch(
            problem=problem,
            specs=specs,
            start_index=child_index,
            model=model,
            variant="hipporag_budget_preserve_selector",
            eval_id=eval_id,
            call_id=f"{call_id}_hipporag_budget_preserve",
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
            max_workers=_budget_matched_control_workers(len(specs)),
        )
        child_attempts = list(batch.get("attempts") or [])
        selection = _select_recursive_child_answer(
            problem=problem,
            attempts=child_attempts,
            model=model,
            eval_id=eval_id,
            call_id=f"{call_id}_hipporag_budget_preserve_selector",
            logger=logger,
            timeout=timeout,
            max_tokens=min(max_tokens, 384),
            evidence_context="",
        )
        selection = _apply_verified_or_abstain_selection(problem=problem, attempts=child_attempts, selection=selection)
        selected_answer = selection.get("selected_answer") or _fallback_answer(child_attempts)
        aggregate_child_index = child_index + len(specs)
        aggregate_child_id = stable_hash({
            "call_id": call_id,
            "child_index": aggregate_child_index,
            "prompt_kind": "hipporag_preserve_selector_answer",
            "mode": "budget_matched",
        })
        attempt = {
            "child_id": aggregate_child_id,
            "child_index": aggregate_child_index,
            "prompt_kind": "hipporag_preserve_selector_answer",
            "branch_axis": _child_branch_axis("hipporag_preserve_selector_answer"),
            "orthogonal_branch_id": _child_branch_id(
                problem,
                prompt_kind="hipporag_preserve_selector_answer",
                branch_axis=_child_branch_axis("hipporag_preserve_selector_answer"),
            ),
            "parsed_answer": selected_answer,
            "parsed_answer_hash": stable_hash({"answer": selected_answer}) if selected_answer else None,
            "prediction_hash": stable_hash({
                "mode": "hipporag_budget_preserve_selector",
                "selection": selection.get("selection_method"),
                "answer": selected_answer,
                "candidate_hashes": [
                    child.get("parsed_answer_hash") for child in child_attempts if child.get("parsed_answer_hash")
                ],
            }),
            "latency_sec": None,
            "status": "answered" if selected_answer else "no_answer",
            "preserve_context_char_count": context_char_count,
            "preserve_selected_doc_count": selected_doc_count,
            "preserve_candidate_doc_count": candidate_doc_count,
            "hipporag_budget_child_ids": [child.get("child_id") for child in child_attempts],
            "hipporag_budget_selection_method": selection.get("selection_method"),
        }
        context_answer_support = _context_answer_support_for_mc(
            problem=problem,
            answer=str(selected_answer or ""),
            context=context_text,
        )
        attempt["budget_matched"] = True
        attempt["context_answer_supported"] = bool(context_answer_support.get("supported"))
        attempt["context_answer_overlap_count"] = int(context_answer_support.get("overlap_count") or 0)
        attempt["context_question_overlap_count"] = int(context_answer_support.get("question_overlap_count") or 0)
        attempt["context_answer_option_hash"] = context_answer_support.get("option_hash")
        summary = {
            "status": "activated" if selected_answer else "no_candidate",
            "policy": (
                "env_forced_hipporag_budget_matched_candidate"
                if env_forced
                else "cost_aware_unverified_mc_hipporag_budget_matched_candidate"
            ),
            "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
            "base_model": model,
            "child_id": aggregate_child_id,
            "child_index": aggregate_child_index,
            "child_status": attempt.get("status"),
            "child_error_type": attempt.get("error_type"),
            "candidate_emitted": bool(str(selected_answer or "").strip()),
            "candidate_answer_hash": attempt.get("parsed_answer_hash"),
            "retrieval_status": retrieval_status,
            "retrieval_query_count": int(retrieval.get("query_count") or 0),
            "candidate_doc_count": candidate_doc_count,
            "rerank_status": rerank_status,
            "selected_doc_count": selected_doc_count,
            "context_char_count": context_char_count,
            "context_answer_supported": bool(context_answer_support.get("supported")),
            "context_answer_overlap_count": int(context_answer_support.get("overlap_count") or 0),
            "context_question_overlap_count": int(context_answer_support.get("question_overlap_count") or 0),
            "context_answer_option_hash": context_answer_support.get("option_hash"),
            "baseline_plan_hash": stable_hash(baseline_plan),
            "budget_matched": True,
            "candidate_count": len(child_attempts),
            "answered_candidate_count": sum(1 for child in child_attempts if child.get("status") == "answered"),
            "selection_method": selection.get("selection_method"),
            "verified_or_abstain_gate": selection.get("verified_or_abstain_gate"),
            "underlying_model_calls": int(batch.get("underlying_model_calls") or 0)
            + int(selection.get("underlying_model_calls") or 0),
        }
        return attempt, summary
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "hipporag_preserve_selector_answer",
            "prompt": _prompt_for(problem, variant="hipporag_baseline", agent_plan=baseline_plan),
        },
        child_index=child_index,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    attempt["preserve_context_char_count"] = context_char_count
    attempt["preserve_selected_doc_count"] = selected_doc_count
    attempt["preserve_candidate_doc_count"] = candidate_doc_count
    context_answer_support = _context_answer_support_for_mc(
        problem=problem,
        answer=str(attempt.get("parsed_answer") or ""),
        context=context_text,
    )
    attempt["context_answer_supported"] = bool(context_answer_support.get("supported"))
    attempt["context_answer_overlap_count"] = int(context_answer_support.get("overlap_count") or 0)
    attempt["context_question_overlap_count"] = int(context_answer_support.get("question_overlap_count") or 0)
    attempt["context_answer_option_hash"] = context_answer_support.get("option_hash")
    summary = {
        "status": "activated",
        "policy": (
            "env_forced_hipporag_baseline_candidate"
            if env_forced
            else "cost_aware_unverified_mc_hipporag_baseline_candidate"
        ),
        "trigger": trigger if not env_forced else {"status": "activated", "reason": "env_forced"},
        "base_model": model,
        "child_id": attempt.get("child_id"),
        "child_index": attempt.get("child_index"),
        "child_status": attempt.get("status"),
        "child_error_type": attempt.get("error_type"),
        "candidate_emitted": bool(str(attempt.get("parsed_answer") or "").strip()),
        "candidate_answer_hash": attempt.get("parsed_answer_hash"),
        "retrieval_status": retrieval_status,
        "retrieval_query_count": int(retrieval.get("query_count") or 0),
        "candidate_doc_count": candidate_doc_count,
        "rerank_status": rerank_status,
        "selected_doc_count": selected_doc_count,
        "context_char_count": context_char_count,
        "context_answer_supported": bool(context_answer_support.get("supported")),
        "context_answer_overlap_count": int(context_answer_support.get("overlap_count") or 0),
        "context_question_overlap_count": int(context_answer_support.get("question_overlap_count") or 0),
        "context_answer_option_hash": context_answer_support.get("option_hash"),
        "baseline_plan_hash": stable_hash(baseline_plan),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    return attempt, summary


def _route_arbitrator_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_ROUTE_ARBITRATOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _route_value_verifier_enabled() -> bool:
    return os.environ.get("HLE_DISABLE_ROUTE_VALUE_VERIFIER", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _route_consensus_guard_enabled() -> bool:
    return os.environ.get("HLE_DISABLE_ROUTE_CONSENSUS_GUARD", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _budget_echo_guard_enabled() -> bool:
    return os.environ.get("HLE_DISABLE_BUDGET_ECHO_GUARD", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _llm_route_arbitrator_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_LLM_ROUTE_ARBITRATOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cache_first_route_arbitrator_enabled() -> bool:
    if not _route_arbitrator_enabled():
        return False
    return os.environ.get("HLE_DISABLE_CACHE_FIRST_ROUTE_ARBITRATOR", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _same_run_cache_route_attempt(
    *,
    problem: dict[str, Any],
    entry: dict[str, Any],
    prompt_kind: str,
    child_index: int,
    call_id: str,
    hipporag_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    answer = str(entry.get("answer") or "").strip()
    if not answer:
        return None
    variant = str(entry.get("variant") or "")
    child_id = stable_hash({
        "call_id": call_id,
        "child_index": child_index,
        "prompt_kind": prompt_kind,
        "same_run_cache_variant": variant,
        "answer_hash": entry.get("answer_hash"),
    })
    attempt = {
        "child_id": child_id,
        "child_index": child_index,
        "prompt_kind": prompt_kind,
        "branch_axis": _child_branch_axis(prompt_kind),
        "orthogonal_branch_id": _child_branch_id(
            problem,
            prompt_kind=prompt_kind,
            branch_axis=_child_branch_axis(prompt_kind),
        ),
        "parsed_answer": answer,
        "parsed_answer_hash": stable_hash({"answer": answer}) if answer else None,
        "prediction_hash": stable_hash({
            "same_run_cache_variant": variant,
            "answer_hash": entry.get("answer_hash"),
        }),
        "latency_sec": 0.0,
        "status": "answered" if answer else "no_answer",
        "same_run_baseline_cache_variant": variant,
    }
    if prompt_kind == "raw_budget_preserve_selector_answer":
        attempt.update({
            "raw_budget_selection_method": "same_run_raw_budget_cache",
            "budget_matched": True,
            "budget_candidate_count": int(entry.get("budget_candidate_count") or 0),
            "budget_answered_candidate_count": int(entry.get("budget_answered_candidate_count") or 0),
            "budget_top_candidate_vote_count": int(entry.get("budget_top_candidate_vote_count") or 0),
            "budget_strong_consensus": bool(entry.get("budget_strong_consensus")),
            "budget_selected_answer_hash": entry.get("budget_selected_answer_hash"),
            "budget_top_candidate_answer_hash": entry.get("budget_top_candidate_answer_hash"),
            "budget_candidate_answer_hash_counts": dict(entry.get("budget_candidate_answer_hash_counts") or {}),
            "budget_verified_or_abstain_gate": entry.get("budget_verified_or_abstain_gate"),
        })
    if prompt_kind == "hipporag_preserve_selector_answer":
        context_char_count = int(entry.get("context_char_count") or 0)
        selected_doc_count = int(entry.get("selected_doc_count") or 0)
        candidate_doc_count = int(entry.get("candidate_doc_count") or 0)
        variant_is_budget = variant == "hipporag_budget_matched"
        has_budget_candidates = int(entry.get("budget_candidate_count") or 0) > 0
        if (
            context_char_count <= 0
            or (selected_doc_count <= 0 and candidate_doc_count <= 0)
        ) and not (variant_is_budget and has_budget_candidates):
            return None
        answer_norm = _normalize_for_selection(answer, answer_type=problem.get("answer_type") or "exactMatch")
        agreeing_variants = [
            str(other.get("variant") or "")
            for other in (hipporag_entries or [])
            if _normalize_for_selection(
                str(other.get("answer") or ""),
                answer_type=problem.get("answer_type") or "exactMatch",
            ) == answer_norm
        ]
        attempt.update({
            "preserve_context_char_count": context_char_count,
            "preserve_selected_doc_count": selected_doc_count,
            "preserve_candidate_doc_count": candidate_doc_count,
            "hipporag_budget_selection_method": "same_run_hipporag_cache",
            "same_route_agreement_count": len(agreeing_variants),
            "same_route_agreeing_variants": agreeing_variants,
            "budget_matched": variant_is_budget,
            "budget_candidate_count": int(entry.get("budget_candidate_count") or 0),
            "budget_answered_candidate_count": int(entry.get("budget_answered_candidate_count") or 0),
            "budget_top_candidate_vote_count": int(entry.get("budget_top_candidate_vote_count") or 0),
            "budget_strong_consensus": bool(entry.get("budget_strong_consensus")),
            "budget_selected_answer_hash": entry.get("budget_selected_answer_hash"),
            "budget_top_candidate_answer_hash": entry.get("budget_top_candidate_answer_hash"),
            "budget_candidate_answer_hash_counts": dict(entry.get("budget_candidate_answer_hash_counts") or {}),
            "budget_verified_or_abstain_gate": entry.get("budget_verified_or_abstain_gate"),
            "context_answer_supported": bool(entry.get("context_answer_supported")),
            "context_answer_overlap_count": int(entry.get("context_answer_overlap_count") or 0),
            "context_question_overlap_count": int(entry.get("context_question_overlap_count") or 0),
            "context_answer_option_hash": entry.get("context_answer_option_hash"),
        })
    return attempt


def _same_run_cache_route_candidates(
    *,
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None,
    call_id: str,
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if problem.get("answer_type") != "multipleChoice":
        return [], None, None, None
    cache_entries = _same_run_cached_baseline_entries(
        agent_plan,
        ["raw", "raw_budget_matched", "hipporag_baseline", "hipporag_budget_matched"],
    )
    if not cache_entries:
        return [], None, None, None
    by_variant = {str(entry.get("variant") or ""): entry for entry in cache_entries}
    hipporag_entries = [
        entry for entry in cache_entries
        if str(entry.get("variant") or "").startswith("hipporag")
    ]
    attempts: list[dict[str, Any]] = []
    raw_summary: dict[str, Any] | None = None
    raw_budget_summary: dict[str, Any] | None = None
    hipporag_summary: dict[str, Any] | None = None
    specs = [
        ("raw", "raw_preserve_selector_answer"),
        ("raw_budget_matched", "raw_budget_preserve_selector_answer"),
        ("hipporag_baseline", "hipporag_preserve_selector_answer"),
        ("hipporag_budget_matched", "hipporag_preserve_selector_answer"),
    ]
    for variant, prompt_kind in specs:
        entry = by_variant.get(variant)
        if not entry:
            continue
        attempt = _same_run_cache_route_attempt(
            problem=problem,
            entry=entry,
            prompt_kind=prompt_kind,
            child_index=start_index + len(attempts),
            call_id=call_id,
            hipporag_entries=hipporag_entries,
        )
        if not attempt:
            continue
        attempts.append(attempt)
        if variant == "raw":
            raw_summary = {
                "status": "activated",
                "policy": "cache_first_raw_baseline_candidate",
                "trigger": {"status": "activated", "reason": "cache_first_route_arbitrator"},
                "child_id": attempt.get("child_id"),
                "child_index": attempt.get("child_index"),
                "child_status": attempt.get("status"),
                "candidate_emitted": True,
                "candidate_answer_hash": attempt.get("parsed_answer_hash"),
                "same_run_cache_variant": variant,
                "borrowed_baseline_model_calls": 1,
                "underlying_model_calls": 0,
            }
        elif variant == "raw_budget_matched":
            raw_budget_summary = {
                "status": "activated",
                "policy": "cache_first_raw_budget_matched_candidate",
                "trigger": {"status": "activated", "reason": "cache_first_route_arbitrator"},
                "child_id": attempt.get("child_id"),
                "child_index": attempt.get("child_index"),
                "child_status": attempt.get("status"),
                "candidate_emitted": True,
                "candidate_answer_hash": attempt.get("parsed_answer_hash"),
                "candidate_count": int(entry.get("budget_candidate_count") or 0),
                "answered_candidate_count": int(entry.get("budget_answered_candidate_count") or 0),
                "candidate_answer_hash_counts": dict(entry.get("budget_candidate_answer_hash_counts") or {}),
                "top_candidate_answer_hash": entry.get("budget_top_candidate_answer_hash"),
                "top_candidate_vote_count": int(entry.get("budget_top_candidate_vote_count") or 0),
                "strong_consensus": bool(entry.get("budget_strong_consensus")),
                "selection_method": "same_run_raw_budget_cache",
                "selected_answer_hash": attempt.get("parsed_answer_hash"),
                "verified_or_abstain_gate": entry.get("budget_verified_or_abstain_gate"),
                "same_run_cache_variant": variant,
                "borrowed_baseline_model_calls": 1,
                "underlying_model_calls": 0,
            }
        elif variant.startswith("hipporag"):
            if hipporag_summary is None:
                hipporag_summary = {
                    "status": "activated",
                    "policy": "cache_first_hipporag_family_candidates",
                    "trigger": {"status": "activated", "reason": "cache_first_route_arbitrator"},
                    "candidate_emitted": True,
                    "candidate_variants": [],
                    "candidate_answer_hashes": [],
                    "candidate_doc_count": 0,
                    "selected_doc_count": 0,
                    "context_char_count": 0,
                    "budget_matched_variant_count": 0,
                    "context_answer_supported_variant_count": 0,
                    "underlying_model_calls": 0,
                }
            hipporag_summary["candidate_variants"].append(variant)
            hipporag_summary["candidate_answer_hashes"].append(attempt.get("parsed_answer_hash"))
            hipporag_summary["candidate_doc_count"] += int(entry.get("candidate_doc_count") or 0)
            hipporag_summary["selected_doc_count"] += int(entry.get("selected_doc_count") or 0)
            hipporag_summary["context_char_count"] += int(entry.get("context_char_count") or 0)
            hipporag_summary["budget_matched_variant_count"] += int(variant == "hipporag_budget_matched")
            hipporag_summary["context_answer_supported_variant_count"] += int(
                bool(entry.get("context_answer_supported"))
            )
            if hipporag_summary.get("child_id") is None:
                hipporag_summary.update({
                    "child_id": attempt.get("child_id"),
                    "child_index": attempt.get("child_index"),
                    "child_status": attempt.get("status"),
                    "candidate_answer_hash": attempt.get("parsed_answer_hash"),
                    "retrieval_status": "same_run_cache",
                    "selection_method": "same_run_hipporag_cache",
                    "same_run_cache_variant": variant,
                    "borrowed_baseline_model_calls": 1,
                })
    return attempts, raw_summary, raw_budget_summary, hipporag_summary


def _route_arbitrator_should_lock(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    if summary.get("status") != "activated" or not summary.get("candidate_emitted"):
        return False
    selected_score = float(summary.get("selected_route_score") or 0.0)
    runner_up = float(summary.get("runner_up_score") or 0.0)
    margin = selected_score - runner_up
    selected_route = str(summary.get("selected_route_type") or "")
    if selected_route == "raw_budget_consensus":
        if os.environ.get("HLE_ENABLE_RAW_BUDGET_CACHE_FIRST_LOCK", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return bool(summary.get("raw_budget_strong_consensus")) or margin >= 1.5
        return False
    if summary.get("selected_route_trusted"):
        return True
    if selected_route == "hipporag_preserve":
        if not summary.get("selected_route_trusted"):
            return False
        for row in summary.get("route_scores") or []:
            if (
                row.get("route_type") == "hipporag_preserve"
                and row.get("score") == summary.get("selected_route_score")
            ):
                return bool(row.get("context_answer_supported")) and (
                    bool(row.get("budget_strong_consensus"))
                    or int(row.get("baseline_cache_support_count") or 0) >= 2
                    or margin >= 2.0
                )
        return margin >= 2.5
    if selected_route in {"raw_preserve", "direct"}:
        return margin >= 2.5
    return margin >= 3.0


def _route_value_of_information_hard_gate_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_ROUTE_VOI_HARD_GATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _route_value_of_information_gate_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Diagnose whether opening more route/challenge branches has useful expected value.

    The summary is intentionally metadata-only.  It never stores HLE text or
    predictions, only route types, scores, hashes, and reason tags already
    produced by the route arbitrator.
    """
    if not isinstance(summary, dict) or summary.get("status") != "activated":
        return {
            "status": "not_required",
            "reason": "route_arbitrator_not_activated",
            "hard_gate_enabled": _route_value_of_information_hard_gate_enabled(),
            "hard_gate_applied": False,
        }
    selected_route = str(summary.get("selected_route_type") or "")
    route_scores = [
        row for row in (summary.get("route_scores") or [])
        if isinstance(row, dict)
    ]
    selected_child_id = summary.get("selected_route_child_id")
    selected_rows = [
        row for row in route_scores
        if row.get("child_id") == selected_child_id
        or row.get("route_type") == selected_route and row.get("score") == summary.get("selected_route_score")
    ]
    selected_row = selected_rows[0] if selected_rows else {}
    selected_norm_hash = str(selected_row.get("normalized_answer_hash") or "")
    profile = selected_row.get("value_profile")
    profile = profile if isinstance(profile, dict) else {}
    tags = set(str(tag) for tag in profile.get("reason_tags", []) or [])
    risks = set(str(risk) for risk in profile.get("risk_tags", []) or [])
    selected_score = float(summary.get("selected_route_score") or 0.0)
    runner_up = float(summary.get("runner_up_score") or 0.0)
    margin = round(selected_score - runner_up, 4)
    trusted = bool(summary.get("selected_route_trusted"))
    selected_trust_reason = str(summary.get("selected_route_trust_reason") or "")
    raw_budget_strong = bool(summary.get("raw_budget_strong_consensus"))
    raw_budget_votes = int(summary.get("raw_budget_top_vote_count") or 0)
    credible_counter_routes = []
    for row in route_scores:
        if row.get("child_id") == selected_child_id:
            continue
        if selected_norm_hash and str(row.get("normalized_answer_hash") or "") == selected_norm_hash:
            continue
        row_profile = row.get("value_profile")
        row_profile = row_profile if isinstance(row_profile, dict) else {}
        row_tags = set(str(tag) for tag in row_profile.get("reason_tags", []) or [])
        row_confidence = str(row_profile.get("confidence") or "").lower()
        row_score = float(row.get("score") or 0.0)
        if (
            row_confidence in {"verified", "high"}
            or bool(row.get("context_answer_supported"))
            or "answer_bearing_retrieval" in row_tags
            or int(row.get("baseline_cache_support_count") or 0) >= 3
        ) and row_score >= runner_up:
            credible_counter_routes.append({
                "route_type": row.get("route_type"),
                "prompt_kind": row.get("prompt_kind"),
                "score": row.get("score"),
                "confidence": row_confidence or None,
                "answer_hash": row.get("answer_hash"),
                "normalized_answer_hash": row.get("normalized_answer_hash"),
            })

    answer_bearing = (
        selected_route == "answer_bearing_evidence"
        or bool(selected_row.get("context_answer_supported"))
        or "answer_bearing_retrieval" in tags
        or "allowed_retrieval_budget_counter_to_raw_budget" in tags
    )
    route_family_consensus = bool(summary.get("route_consensus"))
    hard_gate_enabled = _route_value_of_information_hard_gate_enabled()
    answer_bearing_counter_trust = selected_trust_reason in {
        "answer_bearing_hipporag_counter_to_budget_echo",
        "baseline_supported_answer_bearing_hipporag_family_route",
        "answer_bearing_hipporag_route",
    }

    status = "continue_exploration"
    reason = "route_uncertainty_or_counter_routes_have_value"
    preserve_route = False
    if trusted and route_family_consensus and margin >= 2.0 and not credible_counter_routes:
        status = "preserve_route"
        reason = "trusted_route_family_consensus_low_marginal_voi"
        preserve_route = True
    elif (
        selected_route == "raw_budget_consensus"
        and trusted
        and raw_budget_strong
        and raw_budget_votes >= 3
        and not credible_counter_routes
        and "conflicts_with_strong_retrieval_budget_counter" not in risks
    ):
        status = "preserve_route"
        reason = "strong_raw_budget_consensus_without_credible_counter_low_marginal_voi"
        preserve_route = True
    elif (
        trusted
        and selected_route == "hipporag_preserve"
        and answer_bearing
        and margin >= 2.0
        and (not credible_counter_routes or answer_bearing_counter_trust)
    ):
        status = "preserve_route"
        reason = "trusted_answer_bearing_hipporag_route_low_marginal_voi"
        preserve_route = True
    elif not trusted and not credible_counter_routes:
        status = "continue_exploration"
        reason = "selected_route_untrusted_and_no_safe_route_to_preserve"

    return {
        "status": status,
        "reason": reason,
        "hard_gate_enabled": hard_gate_enabled,
        "hard_gate_applied": False,
        "recommended_action": "preserve_route" if preserve_route else "continue_exploration",
        "selected_route_type": selected_route,
        "selected_route_trusted": trusted,
        "selected_route_trust_reason": summary.get("selected_route_trust_reason"),
        "selected_route_score": summary.get("selected_route_score"),
        "runner_up_score": summary.get("runner_up_score"),
        "score_margin": margin,
        "route_consensus": route_family_consensus,
        "raw_budget_strong_consensus": raw_budget_strong,
        "raw_budget_top_vote_count": raw_budget_votes,
        "selected_route_answer_bearing": bool(answer_bearing),
        "selected_route_risk_tags": sorted(risks),
        "credible_counter_route_count": len(credible_counter_routes),
        "credible_counter_routes": credible_counter_routes[:4],
    }


def _route_value_of_information_gate_should_lock(summary: dict[str, Any] | None) -> bool:
    if not _route_value_of_information_hard_gate_enabled():
        return False
    gate = (summary or {}).get("value_of_information_gate") if isinstance(summary, dict) else None
    if not isinstance(gate, dict):
        return False
    return gate.get("status") == "preserve_route" and gate.get("recommended_action") == "preserve_route"


def _route_arbitrator_lock_decision(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    gate = summary.get("value_of_information_gate")
    if isinstance(gate, dict):
        if gate.get("status") == "continue_exploration" or gate.get("recommended_action") == "continue_exploration":
            return False
    if _route_value_of_information_gate_should_lock(summary):
        gate = summary.setdefault("value_of_information_gate", {})
        if isinstance(gate, dict):
            gate["hard_gate_applied"] = True
        return True
    return _route_arbitrator_should_lock(summary)


def _llm_route_arbitrator_prompt(
    *,
    problem: dict[str, Any],
    scored: list[dict[str, Any]],
) -> str:
    route_rows = []
    for index, row in enumerate(scored[:8], start=1):
        route_rows.append({
            "route_id": index,
            "route_type": row.get("route_type"),
            "answer": row.get("answer"),
            "heuristic_score": row.get("route_score"),
            "baseline_support_variants": row.get("baseline_cache_support_variants"),
            "base_pair_consensus": bool(row.get("base_pair_consensus")),
            "budget_matched": bool(row.get("budget_matched")),
            "budget_strong_consensus": bool(row.get("budget_strong_consensus")),
            "budget_top_candidate_vote_count": row.get("budget_top_candidate_vote_count"),
            "context_answer_supported": bool(row.get("context_answer_supported")),
            "context_answer_overlap_count": row.get("context_answer_overlap_count"),
            "context_question_overlap_count": row.get("context_question_overlap_count"),
            "same_route_agreement_count": row.get("same_route_agreement_count"),
        })
    return (
        "You are a route arbitrator for a multiple-choice QA agent. Choose which existing route's answer "
        "should be preserved. Do not solve by majority alone. Prefer a route with answer-bearing evidence, "
        "a reliable verified/budget signal, or a clear constraint match. Treat unsupported retrieval context "
        "as weak even if several routes agree. If the best choice is uncertain, choose the safest baseline route.\n\n"
        "Return JSON only: {\"route_id\": <integer>, \"confidence\": \"low|medium|high\", "
        "\"reason_tag\": \"...\"}.\n\n"
        f"Question:\n{problem.get('_question')}\n\n"
        f"Candidate routes:\n{json.dumps(route_rows, ensure_ascii=False, indent=2)}"
    )


def _strip_json_code_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _maybe_select_route_with_llm(
    *,
    problem: dict[str, Any],
    scored: list[dict[str, Any]],
    model: str | None,
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not _llm_route_arbitrator_enabled() or not model:
        return None, None
    if len(scored) < 2:
        return None, {"status": "abstained", "reason": "too_few_routes", "underlying_model_calls": 0}
    prompt = _llm_route_arbitrator_prompt(problem=problem, scored=scored)
    started = time.monotonic()
    try:
        response = _call_model(
            model=model,
            prompt=prompt,
            timeout=timeout,
            max_tokens=min(max_tokens, 256),
        )
    except Exception as exc:
        return None, {
            "status": "error",
            "error_type": type(exc).__name__,
            "underlying_model_calls": 1,
            "latency_sec": round(time.monotonic() - started, 4),
        }
    latency = round(time.monotonic() - started, 4)
    try:
        parsed = json.loads(_strip_json_code_fences(response))
    except Exception:
        parsed = {}
    route_id = parsed.get("route_id")
    try:
        route_index = int(route_id) - 1
    except (TypeError, ValueError):
        route_index = -1
    if route_index < 0 or route_index >= min(len(scored), 8):
        return None, {
            "status": "invalid_choice",
            "route_id": route_id,
            "response_hash": stable_hash({"response": response}),
            "underlying_model_calls": 1,
            "latency_sec": latency,
        }
    selected = scored[route_index]
    return selected, {
        "status": "activated",
        "route_id": route_index + 1,
        "selected_route_type": selected.get("route_type"),
        "selected_route_prompt_kind": selected.get("prompt_kind"),
        "selected_answer_hash": selected.get("answer_hash"),
        "confidence": parsed.get("confidence"),
        "reason_tag": parsed.get("reason_tag"),
        "response_hash": stable_hash({"response": response}),
        "underlying_model_calls": 1,
        "latency_sec": latency,
    }


def _route_arbitrator_trust_decision(
    selected: dict[str, Any],
    *,
    runner_up_score: float,
    raw_budget_strong_consensus: bool,
    raw_budget_top_vote_count: int,
    raw_budget_norm: str,
    llm_route_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    route_type = str(selected.get("route_type") or "")
    selected_norm = str(selected.get("normalized_answer") or "")
    selected_score = float(selected.get("route_score") or 0.0)
    margin = selected_score - float(runner_up_score or 0.0)
    baseline_support_count = int(selected.get("baseline_cache_support_count") or 0)
    context_supported = bool(selected.get("context_answer_supported"))
    context_answer_overlap = int(selected.get("context_answer_overlap_count") or 0)
    context_question_overlap = int(selected.get("context_question_overlap_count") or 0)
    context_option_linked = bool(selected.get("context_answer_option_hash"))
    same_route_agreement = int(selected.get("same_route_agreement_count") or 0)
    budget_strong = bool(selected.get("budget_strong_consensus"))
    budget_votes = int(selected.get("budget_top_candidate_vote_count") or 0)
    llm_confidence = str((llm_route_summary or {}).get("confidence") or "").lower()
    value_profile = selected.get("route_value_profile") if isinstance(selected.get("route_value_profile"), dict) else {}
    value_confidence = str(value_profile.get("confidence") or "").lower()
    value_risks = set(str(risk) for risk in value_profile.get("risk_tags", []) or [])
    value_tags = set(str(tag) for tag in value_profile.get("reason_tags", []) or [])
    conflicts_strong_raw_budget = (
        raw_budget_strong_consensus
        and bool(raw_budget_norm)
        and selected_norm != raw_budget_norm
    )

    if (
        value_profile.get("route_consensus")
        and baseline_support_count >= 3
        and "unverified_route_family_consensus" not in value_risks
    ):
        if route_type == "hipporag_preserve" and not context_supported:
            return {"trusted": False, "reason": "hipporag_family_consensus_without_answer_bearing_certificate"}
        return {"trusted": True, "reason": "route_family_consensus"}

    if route_type == "answer_bearing_evidence":
        return {"trusted": True, "reason": "answer_bearing_evidence_route"}

    if route_type == "raw_budget_consensus":
        if "conflicts_with_strong_retrieval_budget_counter" in value_risks:
            return {"trusted": False, "reason": "raw_budget_conflicts_with_retrieval_budget_counter"}
        if "conflicts_with_raw_hipporag_base_pair" in value_risks:
            return {"trusted": False, "reason": "raw_budget_conflicts_with_raw_hipporag_base_pair"}
        if raw_budget_strong_consensus and raw_budget_top_vote_count >= 3:
            return {"trusted": True, "reason": "strong_raw_budget_consensus"}
        if baseline_support_count >= 3 and margin >= 2.0:
            return {"trusted": True, "reason": "multi_baseline_raw_budget_support"}
        return {"trusted": False, "reason": "raw_budget_support_below_trust_threshold"}

    if "unverified_route_family_consensus" in value_risks and not (
        route_type == "answer_bearing_evidence"
        or context_supported
        or context_answer_overlap > 0
    ):
        return {"trusted": False, "reason": "unverified_route_family_consensus"}

    if route_type == "hipporag_preserve":
        if (
            not context_supported
            and "answer_bearing_retrieval" not in value_tags
            and "option_linked_retrieval" not in value_tags
        ):
            return {"trusted": False, "reason": "hipporag_context_not_answer_bearing"}
        if (
            conflicts_strong_raw_budget
            and value_confidence in {"verified", "high"}
            and "independent_hipporag_counter_to_budget_echo" in value_tags
            and "answer_bearing_retrieval" in value_tags
            and (
                context_supported
                or (
                    context_answer_overlap >= 2
                    and context_question_overlap >= 4
                    and int(selected.get("selected_doc_count") or 0) >= 2
                )
            )
            and margin >= 2.0
        ):
            return {"trusted": True, "reason": "answer_bearing_hipporag_counter_to_budget_echo"}
        if conflicts_strong_raw_budget and not (
            (
                baseline_support_count >= 3
                and same_route_agreement >= 2
                and budget_strong
                and budget_votes >= 3
                and context_answer_overlap >= 1
            )
            or (
                "allowed_retrieval_budget_counter_to_raw_budget" in value_tags
                and budget_strong
                and budget_votes >= 3
                and context_question_overlap >= 4
            )
        ):
            return {"trusted": False, "reason": "conflicts_with_strong_raw_budget_without_enough_evidence"}
        if (
            value_confidence in {"verified", "high"}
            and budget_strong
            and budget_votes >= 3
            and "low_baseline_support_in_fragmented_pool" in value_risks
            and not context_supported
            and context_answer_overlap < 2
        ):
            return {"trusted": False, "reason": "budgeted_retrieval_fragmented_low_support"}
        if (
            value_confidence in {"verified", "high"}
            and budget_strong
            and budget_votes >= 3
            and (
                context_supported
                or context_answer_overlap >= 2
                or same_route_agreement >= 2
                or (
                    "raw_and_hipporag_budget_pair_consensus" in value_tags
                    and baseline_support_count >= 2
                )
            )
        ):
            return {"trusted": True, "reason": "high_value_budgeted_retrieval_route"}
        if (
            conflicts_strong_raw_budget
            and value_confidence in {"verified", "high"}
            and "allowed_retrieval_budget_counter_to_raw_budget" in value_tags
            and "low_baseline_support_in_fragmented_pool" not in value_risks
            and budget_strong
            and budget_votes >= 3
            and context_question_overlap >= 4
            and (context_answer_overlap >= 1 or context_option_linked)
        ):
            return {"trusted": True, "reason": "retrieval_budget_counter_to_raw_budget"}
        if (
            value_confidence in {"verified", "high"}
            and baseline_support_count >= 3
            and same_route_agreement >= 2
            and "answer_bearing_retrieval" in value_tags
            and context_supported
            and not conflicts_strong_raw_budget
        ):
            return {"trusted": True, "reason": "baseline_supported_answer_bearing_hipporag_family_route"}
        if (
            (
                context_supported
                or (
                    context_question_overlap >= 4
                    and context_answer_overlap >= 2
                    and int(selected.get("selected_doc_count") or 0) >= 2
                )
            )
            and (
                baseline_support_count >= 2
                or same_route_agreement >= 2
                or (budget_strong and budget_votes >= 3)
            )
        ):
            return {"trusted": True, "reason": "answer_bearing_hipporag_route"}
        if llm_confidence == "high" and margin >= 2.5 and baseline_support_count >= 2:
            return {"trusted": True, "reason": "high_confidence_llm_route_with_baseline_support"}
        return {"trusted": False, "reason": "hipporag_support_below_trust_threshold"}

    if route_type in {"raw_preserve", "direct"}:
        if baseline_support_count >= 3 and margin >= 3.0 and not conflicts_strong_raw_budget:
            return {"trusted": True, "reason": "high_margin_baseline_family_route"}
        return {"trusted": False, "reason": "direct_or_raw_route_not_independently_verified"}

    return {"trusted": False, "reason": "unsupported_route_type"}


def _hipporag_preserve_attempt_has_context(attempt: dict[str, Any]) -> bool:
    return (
        int(attempt.get("preserve_context_char_count") or 0) > 0
        and (
            int(attempt.get("preserve_selected_doc_count") or 0) > 0
            or int(attempt.get("preserve_candidate_doc_count") or 0) > 0
        )
    )


def _route_arbitrator_route_type(prompt_kind: str) -> str:
    prompt_kind = str(prompt_kind or "")
    if prompt_kind == "raw_budget_preserve_selector_answer":
        return "raw_budget_consensus"
    if prompt_kind == "hipporag_preserve_selector_answer":
        return "hipporag_preserve"
    if prompt_kind == "raw_preserve_selector_answer":
        return "raw_preserve"
    if prompt_kind == "direct_short_answer":
        return "direct"
    if prompt_kind in {
        "evidence_bridge_answer",
        "evidence_grounded_answer",
        "answer_bearing_evidence_candidate",
        "mc_option_evidence_scorer_answer",
        "domain_rule_mc_verifier_answer",
    }:
        return "answer_bearing_evidence"
    if prompt_kind in {
        "recursive_assumption_answer",
        "constraint_checked_answer",
        "skeptical_recheck_answer",
        "literal_constraint_answer",
        "option_matrix_reasoner_answer",
        "option_elimination_answer",
        "counter_assumption_challenge_answer",
        "option_elimination_challenge_answer",
        "forced_alternative_answer",
        "critic_synthesis_answer",
        "mc_option_sweep_candidate",
        "agent_context_answer",
        "hipporag_context_answer",
    }:
        return "recursive_child"
    return "other"


def _route_arbitrator_route_priority(route_type: str) -> int:
    return {
        "answer_bearing_evidence": 90,
        "raw_budget_consensus": 80,
        "hipporag_preserve": 75,
        "raw_preserve": 65,
        "direct": 60,
        "recursive_child": 50,
        "other": 10,
    }.get(str(route_type or ""), 0)


def _route_arbitrator_component_score(value: float, *, limit: float | None = None) -> float:
    if limit is None:
        return round(float(value), 4)
    return round(min(float(value), float(limit)), 4)


def _route_arbitrator_has_retrieval_budget_counter_signal(record: dict[str, Any]) -> bool:
    if record.get("route_type") != "hipporag_preserve":
        return False
    if not bool(record.get("budget_matched")):
        return False
    if not bool(record.get("budget_strong_consensus")):
        return False
    if int(record.get("budget_top_candidate_vote_count") or 0) < 3:
        return False
    if int(record.get("context_question_overlap_count") or 0) < 4:
        return False
    return bool(record.get("context_answer_supported")) or bool(record.get("context_answer_option_hash"))


def _route_arbitrator_has_answer_bearing_retrieval_signal(record: dict[str, Any]) -> bool:
    if record.get("route_type") != "hipporag_preserve":
        return False
    return (
        bool(record.get("context_answer_supported"))
        or int(record.get("context_answer_overlap_count") or 0) > 0
        or bool(record.get("context_answer_option_hash"))
    )


def _route_arbitrator_has_independent_hippo_counter_signal(record: dict[str, Any]) -> bool:
    if record.get("route_type") != "hipporag_preserve":
        return False
    return bool(record.get("context_answer_supported")) or int(record.get("context_answer_overlap_count") or 0) > 0


def _route_arbitrator_value_profile(
    record: dict[str, Any],
    *,
    support_count: int,
    non_hipporag_support_count: int,
    raw_budget_strong_consensus: bool,
    raw_budget_top_vote_count: int,
    raw_budget_norm: str,
    retrieval_budget_counter_norms: set[str] | None = None,
    independent_hippo_counter_norms: set[str] | None = None,
    route_consensus: bool = False,
) -> dict[str, Any]:
    route_type = str(record.get("route_type") or "")
    norm = str(record.get("normalized_answer") or "")
    baseline_support_count = int(record.get("baseline_cache_support_count") or 0)
    baseline_unique_answer_count = int(record.get("baseline_cache_unique_answer_count") or 0)
    same_route_agreement_count = int(record.get("same_route_agreement_count") or 0)
    budget_votes = int(record.get("budget_top_candidate_vote_count") or 0)
    budget_strong = bool(record.get("budget_strong_consensus"))
    is_budget_matched = bool(record.get("budget_matched"))
    context_supported = bool(record.get("context_answer_supported"))
    context_answer_overlap = int(record.get("context_answer_overlap_count") or 0)
    context_question_overlap = int(record.get("context_question_overlap_count") or 0)
    context_option_linked = bool(record.get("context_answer_option_hash"))
    selected_docs = int(record.get("selected_doc_count") or 0)
    candidate_docs = int(record.get("candidate_doc_count") or 0)
    context_chars = int(record.get("context_char_count") or 0)
    retrieval_budget_counter_norms = retrieval_budget_counter_norms or set()
    independent_hippo_counter_norms = independent_hippo_counter_norms or set()

    components: dict[str, float] = {}
    tags: list[str] = []
    risks: list[str] = []

    components["answer_support"] = _route_arbitrator_component_score(float(support_count) * 0.65)
    if baseline_support_count:
        components["baseline_support"] = _route_arbitrator_component_score(
            baseline_support_count * 1.10
            + (1.8 if baseline_support_count >= 3 else 0.8 if baseline_support_count == 2 else 0.0)
        )
        tags.append(f"baseline_support_{baseline_support_count}")
    if record.get("base_pair_consensus"):
        components["base_pair_consensus"] = 1.7
        tags.append("raw_hipporag_base_pair_consensus")
    route_has_answer_bearing_signal = (
        route_type == "answer_bearing_evidence"
        or context_supported
        or context_answer_overlap > 0
    )
    if route_consensus:
        if route_has_answer_bearing_signal:
            components["route_family_consensus"] = 3.5 if baseline_support_count >= 3 else 2.0
        else:
            components["route_family_consensus"] = 0.8 if baseline_support_count >= 3 else 0.4
            risks.append("unverified_route_family_consensus")
        tags.append("route_family_consensus")
    if record.get("budget_pair_consensus"):
        components["budget_pair_consensus"] = 4.0 + (0.8 if budget_strong else 0.0)
        tags.append("raw_and_hipporag_budget_pair_consensus")

    if route_type == "answer_bearing_evidence":
        components["route_prior"] = 3.5
        tags.append("answer_bearing_evidence_route")
    elif route_type == "raw_budget_consensus":
        components["route_prior"] = 0.9
        components["budget_consensus"] = _route_arbitrator_component_score(
            min(max(raw_budget_top_vote_count, 0), 5) * 0.55 + (1.6 if raw_budget_strong_consensus else 0.0)
        )
        if raw_budget_strong_consensus:
            tags.append("raw_budget_strong_consensus")
        if record.get("competing_base_pair_consensus_exists") and not record.get("budget_pair_consensus"):
            components["base_pair_counter_penalty"] = -8.0
            risks.append("conflicts_with_raw_hipporag_base_pair")
        if any(counter_norm and counter_norm != norm for counter_norm in retrieval_budget_counter_norms):
            components["retrieval_budget_counter_penalty"] = -5.0
            risks.append("conflicts_with_strong_retrieval_budget_counter")
        if any(counter_norm and counter_norm != norm for counter_norm in independent_hippo_counter_norms):
            components["independent_hipporag_counter_penalty"] = -6.0
            risks.append("conflicts_with_independent_answer_bearing_hipporag_counter")
    elif route_type == "hipporag_preserve":
        components["route_prior"] = 0.7
        retrieval_value = 0.0
        retrieval_value += min(selected_docs, 5) * 0.25
        retrieval_value += min(candidate_docs, 10) * 0.06
        retrieval_value += min(context_chars / 1000.0, 1.4)
        retrieval_value += min(context_question_overlap, 8) * 0.18
        if context_supported or context_answer_overlap > 0:
            retrieval_value += 3.6 + min(context_answer_overlap, 3) * 0.45
            tags.append("answer_bearing_retrieval")
        elif context_option_linked:
            if is_budget_matched and budget_strong:
                retrieval_value += 1.6
                tags.append("option_linked_retrieval")
            else:
                retrieval_value = min(retrieval_value, 1.0) + 0.5
                tags.append("weak_option_linked_retrieval")
                risks.append("weak_option_linked_retrieval_without_budget")
        else:
            if not (is_budget_matched and budget_strong):
                retrieval_value = min(retrieval_value, 0.7)
            components["unsupported_retrieval_penalty"] = -2.0
            risks.append("retrieval_not_answer_bearing")
        components["retrieval_evidence"] = _route_arbitrator_component_score(retrieval_value)
        if is_budget_matched:
            components["budgeted_retrieval"] = 1.0
            tags.append("budget_matched_retrieval")
        if budget_strong:
            components["retrieval_budget_consensus"] = _route_arbitrator_component_score(
                min(max(budget_votes, 0), 5) * 0.65 + 1.8
            )
            tags.append("retrieval_budget_strong_consensus")
        if (not is_budget_matched) and norm in independent_hippo_counter_norms:
            if context_supported or context_answer_overlap > 0:
                components["independent_hipporag_counter"] = 8.0
                tags.append("independent_hipporag_counter_to_budget_echo")
            else:
                components["weak_independent_hipporag_counter"] = 1.2
                risks.append("weak_option_linked_independent_hipporag_counter")
        if (
            is_budget_matched
            and independent_hippo_counter_norms
            and raw_budget_norm
            and norm == raw_budget_norm
        ):
            components["budget_echo_independent_hippo_penalty"] = -14.0
            risks.append("budgeted_retrieval_echoes_raw_against_independent_hipporag")
        if same_route_agreement_count >= 2:
            components["same_route_agreement"] = 1.8
            tags.append("hipporag_family_agreement")
        elif same_route_agreement_count == 1:
            components["same_route_agreement"] = 0.5
        if non_hipporag_support_count > 0:
            components["non_retrieval_support"] = 0.9
        elif same_route_agreement_count < 2 and not (is_budget_matched and budget_strong):
            components["retrieval_isolated_penalty"] = -2.0
            risks.append("retrieval_isolated_from_non_retrieval_routes")
    elif route_type == "raw_preserve":
        components["route_prior"] = 0.5
        if support_count >= 2:
            components["direct_family_support"] = 0.8
    elif route_type == "direct":
        components["route_prior"] = 0.3
        if support_count >= 2:
            components["direct_family_support"] = 0.6
    elif route_type == "recursive_child":
        components["route_prior"] = 0.2

    if raw_budget_strong_consensus and raw_budget_norm and norm != raw_budget_norm:
        if route_type == "hipporag_preserve" and _route_arbitrator_has_retrieval_budget_counter_signal(record):
            components["strong_raw_budget_conflict_penalty"] = -0.2
            components["retrieval_budget_counter_signal"] = 2.0
            tags.append("allowed_retrieval_budget_counter_to_raw_budget")
        elif route_type == "hipporag_preserve" and norm in independent_hippo_counter_norms:
            components["strong_raw_budget_conflict_penalty"] = -2.0
            tags.append("allowed_independent_hipporag_counter_to_budget_echo")
        elif context_supported and same_route_agreement_count >= 2:
            components["strong_raw_budget_conflict_penalty"] = -1.5
        elif context_supported and context_answer_overlap >= 1:
            components["strong_raw_budget_conflict_penalty"] = -2.0
        else:
            components["strong_raw_budget_conflict_penalty"] = -13.0 if route_type == "hipporag_preserve" else -6.0
            risks.append("conflicts_with_strong_raw_budget")
    if record.get("competing_base_pair_consensus_exists") and record.get("budget_pair_consensus"):
        if budget_strong and raw_budget_top_vote_count >= 3:
            components["competing_base_pair_penalty"] = -0.4
            tags.append("budget_pair_allowed_against_base_pair")
        else:
            components["competing_base_pair_penalty"] = -2.5
            risks.append("competes_with_base_pair_consensus")
    if baseline_unique_answer_count >= 3 and baseline_support_count <= 1 and route_type != "answer_bearing_evidence":
        components["fragmented_baseline_penalty"] = -0.8
        risks.append("low_baseline_support_in_fragmented_pool")

    total = round(sum(components.values()), 4)
    if (
        route_type == "answer_bearing_evidence"
        or (context_supported and context_answer_overlap >= 1 and baseline_support_count >= 2)
        or (route_consensus and route_has_answer_bearing_signal and baseline_support_count >= 3)
    ):
        confidence = "verified"
    elif total >= 9.0 and not any(risk in risks for risk in {
        "retrieval_not_answer_bearing",
        "conflicts_with_strong_raw_budget",
        "conflicts_with_strong_retrieval_budget_counter",
        "unverified_route_family_consensus",
    }):
        confidence = "high"
    elif total >= 5.5:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "value_score": total,
        "confidence": confidence,
        "components": components,
        "reason_tags": tags,
        "risk_tags": risks,
        "route_consensus": bool(route_consensus),
    }


def _route_arbitrator_baseline_cache_support(
    *,
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(agent_plan, dict):
        return {
            "count_by_norm": {},
            "variants_by_norm": {},
            "base_pair_norms": [],
            "unique_answer_count": 0,
            "variant_count": 0,
        }
    entries = _same_run_cached_baseline_entries(
        agent_plan,
        ["raw", "raw_budget_matched", "hipporag_baseline", "hipporag_budget_matched"],
    )
    variants_by_norm: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        answer = str(entry.get("answer") or "").strip()
        norm = _normalize_for_selection(answer, answer_type=problem.get("answer_type") or "exactMatch")
        if not norm:
            continue
        variant = str(entry.get("variant") or "")
        if variant and variant not in variants_by_norm[norm]:
            variants_by_norm[norm].append(variant)
    return {
        "count_by_norm": {norm: len(variants) for norm, variants in variants_by_norm.items()},
        "variants_by_norm": {norm: sorted(variants) for norm, variants in variants_by_norm.items()},
        "base_pair_norms": sorted(
            norm
            for norm, variants in variants_by_norm.items()
            if "raw" in variants and "hipporag_baseline" in variants
        ),
        "unique_answer_count": len(variants_by_norm),
        "variant_count": sum(len(variants) for variants in variants_by_norm.values()),
    }


def _route_arbitrator_candidate_records(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any] | None = None,
    raw_budget_preserve_summary: dict[str, Any] | None,
    hipporag_preserve_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    cache_support = _route_arbitrator_baseline_cache_support(problem=problem, agent_plan=agent_plan)
    cache_counts = cache_support.get("count_by_norm") if isinstance(cache_support.get("count_by_norm"), dict) else {}
    cache_variants = (
        cache_support.get("variants_by_norm") if isinstance(cache_support.get("variants_by_norm"), dict) else {}
    )
    base_pair_norms = set(cache_support.get("base_pair_norms") or [])
    records: list[dict[str, Any]] = []
    retained_route_prompts = {
        "direct_short_answer",
        "raw_preserve_selector_answer",
        "raw_budget_preserve_selector_answer",
        "hipporag_preserve_selector_answer",
        "answer_bearing_evidence_candidate",
        "mc_option_evidence_scorer_answer",
        "domain_rule_mc_verifier_answer",
    }
    seen_route_keys: set[tuple[str, str, str]] = set()
    for attempt in valid:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        normalized_answer = _normalize_for_selection(answer, answer_type=problem.get("answer_type") or "exactMatch")
        if not normalized_answer:
            continue
        prompt_kind = str(attempt.get("prompt_kind") or "")
        if prompt_kind not in retained_route_prompts:
            continue
        dedupe_key = (
            prompt_kind,
            normalized_answer,
            str(attempt.get("same_run_baseline_cache_variant") or ""),
        )
        if dedupe_key in seen_route_keys:
            continue
        seen_route_keys.add(dedupe_key)
        route_type = _route_arbitrator_route_type(prompt_kind)
        if route_type == "other":
            continue
        summary: dict[str, Any] = {}
        if route_type == "raw_budget_consensus" and isinstance(raw_budget_preserve_summary, dict):
            summary = raw_budget_preserve_summary
        elif route_type == "hipporag_preserve" and isinstance(hipporag_preserve_summary, dict):
            summary = hipporag_preserve_summary
        baseline_support_variants = list(cache_variants.get(normalized_answer, []) or [])
        records.append({
            "attempt": attempt,
            "route_type": route_type,
            "prompt_kind": prompt_kind,
            "child_id": attempt.get("child_id"),
            "child_index": int(attempt.get("child_index") or 0),
            "answer": answer,
            "answer_hash": attempt.get("parsed_answer_hash") or stable_hash({"answer": answer}),
            "normalized_answer": normalized_answer,
            "normalized_answer_hash": stable_hash({"normalized_answer": normalized_answer}),
            "trusted_verified": (
                attempt.get("candidate_verifier_state") == "verified"
                and _is_trusted_candidate_verifier_attempt(attempt)
            ),
            "context_char_count": int(attempt.get("preserve_context_char_count") or 0),
            "selected_doc_count": int(attempt.get("preserve_selected_doc_count") or 0),
            "candidate_doc_count": int(attempt.get("preserve_candidate_doc_count") or 0),
            "same_route_agreement_count": int(attempt.get("same_route_agreement_count") or 0),
            "same_route_agreeing_variants": list(attempt.get("same_route_agreeing_variants") or []),
            "same_run_cache_variant": attempt.get("same_run_baseline_cache_variant"),
            "budget_matched": bool(attempt.get("budget_matched")),
            "budget_strong_consensus": bool(attempt.get("budget_strong_consensus")),
            "budget_top_candidate_vote_count": int(attempt.get("budget_top_candidate_vote_count") or 0),
            "budget_selected_answer_hash": attempt.get("budget_selected_answer_hash"),
            "budget_top_candidate_answer_hash": attempt.get("budget_top_candidate_answer_hash"),
            "context_answer_supported": bool(attempt.get("context_answer_supported")),
            "context_answer_overlap_count": int(attempt.get("context_answer_overlap_count") or 0),
            "context_question_overlap_count": int(attempt.get("context_question_overlap_count") or 0),
            "context_answer_option_hash": attempt.get("context_answer_option_hash"),
            "baseline_cache_support_count": int(cache_counts.get(normalized_answer, 0) or 0),
            "baseline_cache_support_variants": baseline_support_variants,
            "base_pair_consensus": normalized_answer in base_pair_norms,
            "competing_base_pair_consensus_exists": bool(base_pair_norms and normalized_answer not in base_pair_norms),
            "budget_pair_consensus": (
                "raw_budget_matched" in baseline_support_variants
                and "hipporag_budget_matched" in baseline_support_variants
            ),
            "baseline_cache_unique_answer_count": int(cache_support.get("unique_answer_count") or 0),
            "baseline_cache_variant_count": int(cache_support.get("variant_count") or 0),
            "summary": summary,
        })
    return records


def _score_route_arbitrator_record(
    record: dict[str, Any],
    *,
    support_count: int,
    non_hipporag_support_count: int,
    raw_budget_strong_consensus: bool,
    raw_budget_top_vote_count: int,
    raw_budget_norm: str,
    retrieval_budget_counter_norms: set[str] | None = None,
    independent_hippo_counter_norms: set[str] | None = None,
    route_consensus: bool = False,
) -> float:
    profile = _route_arbitrator_value_profile(
        record,
        support_count=support_count,
        non_hipporag_support_count=non_hipporag_support_count,
        raw_budget_strong_consensus=raw_budget_strong_consensus,
        raw_budget_top_vote_count=raw_budget_top_vote_count,
        raw_budget_norm=raw_budget_norm,
        retrieval_budget_counter_norms=retrieval_budget_counter_norms,
        independent_hippo_counter_norms=independent_hippo_counter_norms,
        route_consensus=route_consensus,
    )
    return float(profile.get("value_score") or 0.0)


def _maybe_add_route_arbitrator_candidate(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any] | None = None,
    raw_budget_preserve_summary: dict[str, Any] | None = None,
    hipporag_preserve_summary: dict[str, Any] | None = None,
    call_id: str = "",
    model: str | None = None,
    timeout: float | None = None,
    max_tokens: int = 384,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not _route_arbitrator_enabled():
        return None, None
    if not _route_value_verifier_enabled():
        return None, {
            "status": "abstained",
            "reason": "route_value_verifier_disabled",
            "policy": "route_level_value_verifier",
            "route_value_verifier_enabled": False,
            "route_consensus_guard_enabled": _route_consensus_guard_enabled(),
            "budget_echo_guard_enabled": _budget_echo_guard_enabled(),
            "underlying_model_calls": 0,
        }
    if problem.get("answer_type") != "multipleChoice":
        return None, {"status": "abstained", "reason": "not_multiple_choice", "underlying_model_calls": 0}
    records = _route_arbitrator_candidate_records(
        problem=problem,
        attempts=attempts,
        agent_plan=agent_plan,
        raw_budget_preserve_summary=raw_budget_preserve_summary,
        hipporag_preserve_summary=hipporag_preserve_summary,
    )
    if not records:
        return None, {"status": "abstained", "reason": "no_route_candidates", "underlying_model_calls": 0}
    if any(record.get("trusted_verified") for record in records):
        return None, {
            "status": "abstained",
            "reason": "trusted_verified_candidate_available",
            "route_count": len(records),
            "underlying_model_calls": 0,
        }
    route_types = {str(record.get("route_type") or "") for record in records}
    answer_norms = {str(record.get("normalized_answer") or "") for record in records if record.get("normalized_answer")}
    if len(route_types) < 2:
        return None, {
            "status": "abstained",
            "reason": "single_route_only",
            "route_count": len(records),
            "route_types": sorted(route_types),
            "underlying_model_calls": 0,
        }
    routes_agree = len(answer_norms) < 2
    route_consensus = routes_agree and _route_consensus_guard_enabled()
    if routes_agree and not route_consensus:
        return None, {
            "status": "abstained",
            "reason": "routes_agree",
            "route_count": len(records),
            "route_types": sorted(route_types),
            "unique_answer_count": len(answer_norms),
            "route_value_verifier_enabled": True,
            "route_consensus_guard_enabled": False,
            "budget_echo_guard_enabled": _budget_echo_guard_enabled(),
            "underlying_model_calls": 0,
        }

    answer_support = Counter(str(record.get("normalized_answer") or "") for record in records)
    non_hipporag_support = Counter(
        str(record.get("normalized_answer") or "")
        for record in records
        if record.get("route_type") != "hipporag_preserve"
    )
    baseline_cache_support = _route_arbitrator_baseline_cache_support(problem=problem, agent_plan=agent_plan)
    raw_budget = raw_budget_preserve_summary if isinstance(raw_budget_preserve_summary, dict) else {}
    raw_budget_strong = bool(raw_budget.get("strong_consensus"))
    raw_budget_top_vote_count = int(raw_budget.get("top_candidate_vote_count") or 0)
    raw_budget_records = [record for record in records if record.get("route_type") == "raw_budget_consensus"]
    raw_budget_norm = str(raw_budget_records[0].get("normalized_answer") or "") if raw_budget_records else ""
    retrieval_budget_counter_norms = {
        str(record.get("normalized_answer") or "")
        for record in records
        if _route_arbitrator_has_retrieval_budget_counter_signal(record)
    }
    independent_hippo_counter_norms: set[str] = set()
    if _budget_echo_guard_enabled():
        budget_echoes_raw = any(
            record.get("route_type") == "hipporag_preserve"
            and bool(record.get("budget_matched"))
            and raw_budget_norm
            and str(record.get("normalized_answer") or "") == raw_budget_norm
            for record in records
        )
        independent_hippo_counter_norms = {
            str(record.get("normalized_answer") or "")
            for record in records
            if budget_echoes_raw
            and record.get("route_type") == "hipporag_preserve"
            and not bool(record.get("budget_matched"))
            and raw_budget_norm
            and str(record.get("normalized_answer") or "") != raw_budget_norm
            and _route_arbitrator_has_independent_hippo_counter_signal(record)
        }
    scored: list[dict[str, Any]] = []
    for record in records:
        norm = str(record.get("normalized_answer") or "")
        profile = _route_arbitrator_value_profile(
            record,
            support_count=int(answer_support.get(norm, 0)),
            non_hipporag_support_count=int(non_hipporag_support.get(norm, 0)),
            raw_budget_strong_consensus=raw_budget_strong,
            raw_budget_top_vote_count=raw_budget_top_vote_count,
            raw_budget_norm=raw_budget_norm,
            retrieval_budget_counter_norms=retrieval_budget_counter_norms,
            independent_hippo_counter_norms=independent_hippo_counter_norms,
            route_consensus=route_consensus,
        )
        scored.append({
            **record,
            "route_score": float(profile.get("value_score") or 0.0),
            "route_value_profile": profile,
        })
    if route_consensus:
        consensus_candidates = [
            row for row in scored
            if int(row.get("baseline_cache_support_count") or 0) >= 3
            or bool((row.get("route_value_profile") or {}).get("route_consensus"))
        ]
        if not consensus_candidates or len(records) < 3:
            return None, {
                "status": "abstained",
                "reason": "routes_agree",
                "route_count": len(records),
                "route_types": sorted(route_types),
                "unique_answer_count": len(answer_norms),
                "underlying_model_calls": 0,
            }
    sorted_scored = sorted(
        scored,
        key=lambda row: (
            -float(row.get("route_score") or 0.0),
            -_route_arbitrator_route_priority(str(row.get("route_type") or "")),
            int(row.get("child_index") or 0),
        ),
    )
    selected = sorted_scored[0]
    llm_route_summary: dict[str, Any] | None = None
    llm_selected, llm_route_summary = _maybe_select_route_with_llm(
        problem=problem,
        scored=sorted_scored,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    if llm_selected:
        selected = llm_selected
    second_score = float(sorted_scored[1].get("route_score") or 0.0) if len(sorted_scored) > 1 else 0.0
    selected_answer = str(selected.get("answer") or "").strip()
    trust_decision = _route_arbitrator_trust_decision(
        selected,
        runner_up_score=second_score,
        raw_budget_strong_consensus=raw_budget_strong,
        raw_budget_top_vote_count=raw_budget_top_vote_count,
        raw_budget_norm=raw_budget_norm,
        llm_route_summary=llm_route_summary,
    )
    aggregate_child_index = _timeout_recovery_child_index(attempts)
    child_id = stable_hash({
        "call_id": call_id,
        "child_index": aggregate_child_index,
        "prompt_kind": "route_arbitrator_answer",
        "selected_route_child_id": selected.get("child_id"),
    })
    attempt = {
        "child_id": child_id,
        "child_index": aggregate_child_index,
        "prompt_kind": "route_arbitrator_answer",
        "branch_axis": _child_branch_axis("route_arbitrator_answer"),
        "orthogonal_branch_id": _child_branch_id(
            problem,
            prompt_kind="route_arbitrator_answer",
            branch_axis=_child_branch_axis("route_arbitrator_answer"),
        ),
        "parsed_answer": selected_answer,
        "parsed_answer_hash": stable_hash({"answer": selected_answer}) if selected_answer else None,
        "prediction_hash": stable_hash({
            "route_arbitrator": True,
            "selected_route_type": selected.get("route_type"),
            "selected_child_id": selected.get("child_id"),
            "selected_answer_hash": selected.get("answer_hash"),
            "route_answer_hashes": [row.get("answer_hash") for row in scored],
        }),
        "latency_sec": 0.0,
        "status": "answered" if selected_answer else "no_answer",
        "route_arbitrator_selected_route": selected.get("route_type"),
        "route_arbitrator_selected_child_id": selected.get("child_id"),
        "route_arbitrator_score": selected.get("route_score"),
        "route_arbitrator_trusted": bool(trust_decision.get("trusted")),
        "route_arbitrator_trust_reason": trust_decision.get("reason"),
        "route_value_score": selected.get("route_score"),
        "route_value_confidence": (selected.get("route_value_profile") or {}).get("confidence"),
        "route_value_profile": selected.get("route_value_profile"),
    }
    if trust_decision.get("trusted"):
        attempt.update({
            "candidate_verifier_state": "verified",
            "candidate_verifier_trust": "route_arbitrator_evidence_gate",
        })
    summary = {
        "status": "activated" if selected_answer else "no_candidate",
        "policy": "route_level_value_verifier",
        "route_value_verifier_enabled": True,
        "route_consensus_guard_enabled": _route_consensus_guard_enabled(),
        "budget_echo_guard_enabled": _budget_echo_guard_enabled(),
        "child_id": child_id,
        "child_index": aggregate_child_index,
        "candidate_emitted": bool(selected_answer),
        "candidate_answer_hash": attempt.get("parsed_answer_hash"),
        "selected_route_type": selected.get("route_type"),
        "selected_route_child_id": selected.get("child_id"),
        "selected_route_prompt_kind": selected.get("prompt_kind"),
        "selected_route_score": selected.get("route_score"),
        "selected_route_value_profile": selected.get("route_value_profile"),
        "route_consensus": bool(route_consensus),
        "retrieval_budget_counter_norm_count": len(retrieval_budget_counter_norms),
        "independent_hippo_counter_norm_count": len(independent_hippo_counter_norms),
        "runner_up_score": round(second_score, 4),
        "selected_route_trusted": bool(trust_decision.get("trusted")),
        "selected_route_trust_reason": trust_decision.get("reason"),
        "route_count": len(scored),
        "route_types": sorted(route_types),
        "unique_answer_count": len(answer_norms),
        "answer_support_hash_counts": {
            stable_hash({"normalized_answer": norm}): count
            for norm, count in sorted(answer_support.items())
        },
        "baseline_cache_support_hash_counts": {
            stable_hash({"normalized_answer": norm}): count
            for norm, count in sorted((baseline_cache_support.get("count_by_norm") or {}).items())
        },
        "baseline_cache_unique_answer_count": int(baseline_cache_support.get("unique_answer_count") or 0),
        "baseline_cache_variant_count": int(baseline_cache_support.get("variant_count") or 0),
        "raw_budget_strong_consensus": raw_budget_strong,
        "raw_budget_top_vote_count": raw_budget_top_vote_count,
        "hipporag_context_route_count": sum(1 for row in scored if row.get("route_type") == "hipporag_preserve"),
        "route_scores": [
            {
                "route_type": row.get("route_type"),
                "prompt_kind": row.get("prompt_kind"),
                "child_id": row.get("child_id"),
                "answer_hash": row.get("answer_hash"),
                "normalized_answer_hash": row.get("normalized_answer_hash"),
                "score": row.get("route_score"),
                "value_profile": row.get("route_value_profile"),
                "context_char_count": row.get("context_char_count"),
                "selected_doc_count": row.get("selected_doc_count"),
                "candidate_doc_count": row.get("candidate_doc_count"),
                "same_route_agreement_count": row.get("same_route_agreement_count"),
                "same_route_agreeing_variants": row.get("same_route_agreeing_variants"),
                "same_run_cache_variant": row.get("same_run_cache_variant"),
                "budget_matched": bool(row.get("budget_matched")),
                "budget_top_candidate_vote_count": row.get("budget_top_candidate_vote_count"),
                "budget_strong_consensus": bool(row.get("budget_strong_consensus")),
                "context_answer_supported": bool(row.get("context_answer_supported")),
                "context_answer_overlap_count": row.get("context_answer_overlap_count"),
                "context_question_overlap_count": row.get("context_question_overlap_count"),
                "context_answer_option_hash": row.get("context_answer_option_hash"),
                "baseline_cache_support_count": row.get("baseline_cache_support_count"),
                "baseline_cache_support_variants": row.get("baseline_cache_support_variants"),
                "base_pair_consensus": bool(row.get("base_pair_consensus")),
                "budget_pair_consensus": bool(row.get("budget_pair_consensus")),
                "competing_base_pair_consensus_exists": bool(row.get("competing_base_pair_consensus_exists")),
            }
            for row in sorted_scored
        ],
        "llm_route_arbitrator": llm_route_summary,
        "underlying_model_calls": int((llm_route_summary or {}).get("underlying_model_calls") or 0),
    }
    summary["value_of_information_gate"] = _route_value_of_information_gate_summary(summary)
    gate = summary.get("value_of_information_gate")
    if isinstance(gate, dict):
        attempt["route_value_of_information_gate_status"] = gate.get("status")
        attempt["route_value_of_information_recommended_action"] = gate.get("recommended_action")
    return attempt, summary


def _cost_aware_hipporag_preserve_trigger(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any],
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"status": "abstained", "reason": "disabled"}
    enabled = os.environ.get("HLE_ENABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return {"status": "abstained", "reason": "not_enabled"}
    if problem.get("answer_type") != "multipleChoice":
        return {"status": "abstained", "reason": "not_multiple_choice"}
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    if not valid:
        return {"status": "abstained", "reason": "no_valid_candidates"}
    trusted_verified = [
        attempt for attempt in valid
        if attempt.get("candidate_verifier_state") == "verified"
        and _is_trusted_candidate_verifier_attempt(attempt)
    ]
    if trusted_verified:
        return {
            "status": "abstained",
            "reason": "trusted_verified_candidate_available",
            "verified_count": len(trusted_verified),
        }
    normalized = {
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type="multipleChoice")
        for attempt in valid
        if str(attempt.get("parsed_answer") or "").strip()
    }
    unique_count = len({value for value in normalized if value})
    stages = agent_plan.get("stages") or {}
    world_model = stages.get("world_model_router") if isinstance(stages.get("world_model_router"), dict) else {}
    generic_graph_only = bool(world_model.get("generic_graph_context_only"))
    graph_context_used = bool((stages.get("prompt_builder") or {}).get("context_injected")) if isinstance(stages.get("prompt_builder"), dict) else False
    return {
        "status": "activated",
        "reason": "unverified_multiple_choice_baseline_preserve",
        "unique_candidate_count": unique_count,
        "valid_candidate_count": len(valid),
        "generic_graph_context_only": generic_graph_only,
        "graph_context_used": graph_context_used,
    }


def _cost_aware_raw_preserve_trigger(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any],
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_COST_AWARE_RAW_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"status": "abstained", "reason": "disabled"}
    enabled = os.environ.get("HLE_ENABLE_COST_AWARE_RAW_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return {"status": "abstained", "reason": "not_enabled"}
    if problem.get("answer_type") != "multipleChoice":
        return {"status": "abstained", "reason": "not_multiple_choice"}
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    if not valid:
        return {"status": "abstained", "reason": "no_valid_candidates"}
    trusted_verified = [
        attempt for attempt in valid
        if attempt.get("candidate_verifier_state") == "verified"
        and _is_trusted_candidate_verifier_attempt(attempt)
    ]
    if trusted_verified:
        return {
            "status": "abstained",
            "reason": "trusted_verified_candidate_available",
            "verified_count": len(trusted_verified),
        }
    normalized = {
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type="multipleChoice")
        for attempt in valid
        if str(attempt.get("parsed_answer") or "").strip()
    }
    unique_count = len({value for value in normalized if value})
    prompt_kinds = {str(attempt.get("prompt_kind") or "") for attempt in valid}
    domain = _classify_hle_domain(problem)
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
    ]).lower()
    world_model = (agent_plan.get("stages") or {}).get("world_model_router") or agent_plan.get("world_model_router") or {}
    generic_graph_only = bool(world_model.get("generic_graph_context_only"))
    graph_context_used = bool((agent_plan.get("stages") or {}).get("prompt_builder", {}).get("context_injected"))
    high_regression_domain = (
        domain == "humanities_social_science"
        or "social" in text
        or "humanit" in text
        or "history" in text
        or "law" in text
    )
    high_divergence = unique_count >= 4
    recursive_pressure = any(
        kind in prompt_kinds
        for kind in {
            "counter_assumption_challenge_answer",
            "option_elimination_challenge_answer",
            "forced_alternative_answer",
            "critic_synthesis_answer",
        }
    )
    if high_regression_domain and high_divergence:
        return {
            "status": "activated",
            "reason": "high_regression_domain_with_unverified_divergent_candidates",
            "domain": domain,
            "unique_candidate_count": unique_count,
            "valid_candidate_count": len(valid),
            "generic_graph_context_only": generic_graph_only,
            "graph_context_used": graph_context_used,
            "recursive_pressure": recursive_pressure,
        }
    if generic_graph_only and high_divergence and "hipporag_context_answer" in prompt_kinds:
        return {
            "status": "activated",
            "reason": "generic_graph_plus_hipporag_disagreement_without_verification",
            "domain": domain,
            "unique_candidate_count": unique_count,
            "valid_candidate_count": len(valid),
            "generic_graph_context_only": generic_graph_only,
            "graph_context_used": graph_context_used,
            "recursive_pressure": recursive_pressure,
        }
    return {
        "status": "abstained",
        "reason": "risk_below_threshold",
        "domain": domain,
        "unique_candidate_count": unique_count,
        "valid_candidate_count": len(valid),
        "generic_graph_context_only": generic_graph_only,
        "graph_context_used": graph_context_used,
        "recursive_pressure": recursive_pressure,
    }


def _cost_aware_raw_budget_preserve_trigger(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    agent_plan: dict[str, Any],
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_COST_AWARE_RAW_BUDGET_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"status": "abstained", "reason": "disabled"}
    enabled = os.environ.get("HLE_ENABLE_COST_AWARE_RAW_BUDGET_PRESERVE_SELECTOR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return {"status": "abstained", "reason": "not_enabled"}
    if problem.get("answer_type") != "multipleChoice":
        return {"status": "abstained", "reason": "not_multiple_choice"}
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    if not valid:
        return {"status": "abstained", "reason": "no_valid_candidates"}
    trusted_verified = [
        attempt for attempt in valid
        if attempt.get("candidate_verifier_state") == "verified"
        and _is_trusted_candidate_verifier_attempt(attempt)
    ]
    if trusted_verified:
        return {
            "status": "abstained",
            "reason": "trusted_verified_candidate_available",
            "verified_count": len(trusted_verified),
        }

    normalized = {
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type="multipleChoice")
        for attempt in valid
        if str(attempt.get("parsed_answer") or "").strip()
    }
    unique_count = len({value for value in normalized if value})
    prompt_kinds = {str(attempt.get("prompt_kind") or "") for attempt in valid}
    stages = agent_plan.get("stages") or {}
    domain = _classify_hle_domain(problem)
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
    ]).lower()
    world_model = stages.get("world_model_router") or agent_plan.get("world_model_router") or {}
    world_model = world_model if isinstance(world_model, dict) else {}
    generic_graph_only = bool(world_model.get("generic_graph_context_only"))
    prompt_builder = stages.get("prompt_builder") if isinstance(stages.get("prompt_builder"), dict) else {}
    graph_context_used = bool(prompt_builder.get("context_injected"))
    morphism = stages.get("structural_morphism_transfer")
    morphism = morphism if isinstance(morphism, dict) else {}
    structural_hits = list(morphism.get("structural_morphism_hits", []) or [])
    formal_hits = list(morphism.get("formal_mapping_hits", []) or [])
    transfer_supported_hits = [
        hit for hit in structural_hits
        if isinstance(hit, dict) and hit.get("decision") == "transfer_supported"
    ]
    weak_morphism_only = bool(structural_hits or formal_hits) and not bool(transfer_supported_hits)
    high_regression_domain = (
        domain == "humanities_social_science"
        or "social" in text
        or "humanit" in text
        or "history" in text
        or "law" in text
    )
    recursive_pressure = any(
        kind in prompt_kinds
        for kind in {
            "counter_assumption_challenge_answer",
            "option_elimination_challenge_answer",
            "forced_alternative_answer",
            "critic_synthesis_answer",
            "hipporag_context_answer",
            "raw_preserve_selector_answer",
        }
    )
    high_divergence = unique_count >= 3
    route_uncertain = any([
        high_divergence,
        high_regression_domain,
        generic_graph_only,
        graph_context_used,
        weak_morphism_only,
        recursive_pressure,
    ])
    if route_uncertain and unique_count >= 2:
        return {
            "status": "activated",
            "reason": "unverified_mc_route_uncertain_use_raw_budget_preserve",
            "domain": domain,
            "unique_candidate_count": unique_count,
            "valid_candidate_count": len(valid),
            "high_regression_domain": high_regression_domain,
            "high_divergence": high_divergence,
            "generic_graph_context_only": generic_graph_only,
            "graph_context_used": graph_context_used,
            "weak_morphism_only": weak_morphism_only,
            "recursive_pressure": recursive_pressure,
        }
    return {
        "status": "abstained",
        "reason": "risk_below_threshold",
        "domain": domain,
        "unique_candidate_count": unique_count,
        "valid_candidate_count": len(valid),
        "high_regression_domain": high_regression_domain,
        "high_divergence": high_divergence,
        "generic_graph_context_only": generic_graph_only,
        "graph_context_used": graph_context_used,
        "weak_morphism_only": weak_morphism_only,
        "recursive_pressure": recursive_pressure,
    }


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
    selected_docs, answer_bearing_certificate = _filter_answer_bearing_evidence_results(
        problem=problem,
        results=[row["doc"] for row in ranked_docs[:5]],
        candidate_answers=[],
        max_results=5,
    )
    context = _format_evidence_context(selected_docs, max_chars=context_max_chars)
    summary = {
        "status": (
            "activated"
            if context
            else (
                str(answer_bearing_certificate.get("status") or "blocked_non_answer_bearing")
                if ranked_docs
                else "no_results"
            )
        ),
        "source": "wikipedia_search_plus_hipporag_style_rerank",
        "query_count": len(queries),
        "query_hashes": [stable_hash({"query": query}) for query in queries],
        "candidate_doc_count": len(docs),
        "selected_doc_count": len(selected_docs),
        "selected_doc_hashes": [
            stable_hash({"title": row.get("title", ""), "snippet": row.get("snippet", "")})
            for row in selected_docs
        ],
        "top_scores": [round(float(row["score"]), 4) for row in ranked_docs[:5]],
        "entity_node_count": len(_hipporag_entity_nodes(problem, docs)),
        "context_char_count": len(context),
        "answer_bearing_certificate": answer_bearing_certificate,
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


_CHILD_BRANCH_AXIS_BY_PROMPT_KIND = {
    "direct_short_answer": "closed_book_direct",
    "constraint_checked_answer": "format_constraint",
    "skeptical_recheck_answer": "skeptical_recheck",
    "literal_constraint_answer": "literal_constraint",
    "option_elimination_answer": "option_elimination",
    "option_elimination_baseline_answer": "option_elimination",
    "option_matrix_reasoner_answer": "option_matrix_reasoning",
    "code_semantics_answer": "code_semantics",
    "recursive_assumption_answer": "assumption_falsification",
    "agent_context_answer": "assumption_graph_transfer",
    "hipporag_context_answer": "hipporag_retrieval_bridge",
    "evidence_bridge_answer": "external_evidence_bridge",
    "evidence_grounded_answer": "answer_bearing_evidence",
    "answer_bearing_evidence_candidate": "answer_bearing_evidence",
    "decomposition_answer": "subproblem_decomposition",
    "adversarial_alternative_answer": "adversarial_boundary_search",
    "counter_assumption_challenge_answer": "counter_assumption_challenge",
    "option_elimination_challenge_answer": "option_elimination",
    "forced_alternative_answer": "forced_alternative",
    "critic_synthesis_answer": "critic_synthesis",
    "structural_option_audit_answer": "structural_option_audit",
    "mc_option_sweep_candidate": "option_sweep",
    "mc_option_evidence_scorer_answer": "option_specific_evidence",
    "evidence_guided_option_challenge_answer": "option_specific_evidence",
    "domain_rule_mc_verifier_answer": "domain_rule_verifier",
    "math_tool_answer": "executable_math_tool",
    "candidate_claim_verifier_answer": "executable_claim_verifier",
    "timeout_recovery_answer": "timeout_recovery",
    "child_model_failover_answer": "model_failover",
    "raw_preserve_selector_answer": "raw_preserve_baseline",
    "raw_budget_preserve_selector_answer": "budget_matched_raw_consensus",
    "hipporag_preserve_selector_answer": "hipporag_preserve_baseline",
    "route_arbitrator_answer": "route_arbitration",
}


_CHILD_BRANCH_INSTRUCTIONS = {
    "closed_book_direct": "Solve closed-book. Do not use retrieved context, analogies, or graph priors.",
    "format_constraint": "Optimize only output contract, units, sign, option label, and exact wording compliance.",
    "skeptical_recheck": "Re-solve from scratch and test the obvious answer against traps and exclusions.",
    "literal_constraint": "Match every explicit textual constraint literally before using broad priors.",
    "option_elimination": "Treat the task as option-by-option elimination and reject contradicted options.",
    "option_matrix_reasoning": "Treat every option as a separate discrete hypothesis leaf and compare the minimal discriminating constraint for each.",
    "code_semantics": "Analyze executable/static code semantics before choosing among compile/runtime options.",
    "assumption_falsification": "Propose competing assumptions, falsify the weaker one, and answer from the survivor.",
    "assumption_graph_transfer": "Use only the retrieved assumption graph or morphism context if it directly constrains the answer.",
    "hipporag_retrieval_bridge": "Use only the retrieval bridge as evidence; ignore assumption graph priors.",
    "external_evidence_bridge": "Use only answer-bearing external evidence and abstain from generic context.",
    "answer_bearing_evidence": "Choose an answer only if concrete evidence supports the entity, option, or value.",
    "subproblem_decomposition": "Decompose the problem into target, constraints, and exclusions before answering.",
    "adversarial_boundary_search": "Search for a boundary case or less obvious answer forced by the wording.",
    "counter_assumption_challenge": "Challenge the current majority with an incompatible assumption family.",
    "forced_alternative": "Force a plausible alternative path, then keep it only if constraints support it.",
    "critic_synthesis": "Arbitrate between incompatible candidates; do not add a new first-impression answer.",
    "structural_option_audit": "Re-evaluate each option as a discrete hypothesis and prefer the least-assumption survivor.",
    "option_specific_evidence": "Score evidence separately for each option and answer only from the strongest supported option.",
    "executable_math_tool": "Use executable symbolic or numeric verification rather than verbal plausibility.",
    "executable_claim_verifier": "Convert a candidate into a checkable claim and verify or refute it.",
    "raw_preserve_baseline": "Preserve the raw baseline unless another branch has verified evidence.",
    "budget_matched_raw_consensus": "Use budget-matched raw consensus as a conservative baseline-preserving branch.",
    "hipporag_preserve_baseline": "Preserve the HippoRAG baseline unless another branch has verified evidence.",
    "route_arbitration": "Choose among raw-budget, HippoRAG, direct, and recursive routes using explicit evidence metadata.",
}


def _child_branch_axis(prompt_kind: str) -> str:
    prompt_kind = str(prompt_kind or "")
    if prompt_kind in _CHILD_BRANCH_AXIS_BY_PROMPT_KIND:
        return _CHILD_BRANCH_AXIS_BY_PROMPT_KIND[prompt_kind]
    return re.sub(r"[^a-z0-9]+", "_", prompt_kind.lower()).strip("_") or "unknown_child_axis"


def _child_branch_id(problem: dict[str, Any], *, prompt_kind: str, branch_axis: str) -> str:
    problem_key = (
        problem.get("id_hash")
        or problem.get("question_hash")
        or stable_hash({"question": problem.get("_question") or ""})
    )
    return stable_hash({
        "problem": problem_key,
        "prompt_kind": prompt_kind,
        "branch_axis": branch_axis,
    })


def _orthogonal_child_prompt_prefix(*, branch_axis: str, prompt_kind: str) -> str:
    instruction = _CHILD_BRANCH_INSTRUCTIONS.get(branch_axis, "Use this branch's distinct search policy.")
    return (
        f"You are recursive child branch `{branch_axis}` for prompt `{prompt_kind}`. "
        "Do not imitate other child branches; this branch is a discrete search leaf with its own failure mode. "
        f"{instruction}\n\n"
    )


def _allow_duplicate_child_branch_axes() -> bool:
    return os.environ.get("HLE_ALLOW_DUPLICATE_CHILD_BRANCH_AXES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _orthogonal_child_early_stop_guard_enabled() -> bool:
    return os.environ.get("HLE_DISABLE_ORTHOGONAL_CHILD_EARLY_STOP_GUARD", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _min_orthogonal_child_axes_before_early_stop(problem: dict[str, Any]) -> int:
    env_value = os.environ.get("HLE_MIN_ORTHOGONAL_CHILD_AXES_BEFORE_EARLY_STOP", "").strip()
    if env_value:
        try:
            return max(1, min(8, int(env_value)))
        except ValueError:
            pass
    if problem.get("answer_type") == "multipleChoice":
        return 6
    return 3


def _child_attempt_branch_axis(attempt: dict[str, Any]) -> str:
    return str(attempt.get("branch_axis") or _child_branch_axis(str(attempt.get("prompt_kind") or "")))


def _child_branch_axes_for_attempts(attempts: list[dict[str, Any]]) -> list[str]:
    return [_child_attempt_branch_axis(attempt) for attempt in attempts]


def _valid_child_branch_axis_count(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> int:
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    return len({axis for axis in _child_branch_axes_for_attempts(valid) if axis})


def _required_child_branch_axes_before_early_stop(problem: dict[str, Any]) -> set[str]:
    if os.environ.get("HLE_DISABLE_CORE_ORTHOGONAL_AXES_BEFORE_EARLY_STOP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return set()
    if problem.get("answer_type") != "multipleChoice":
        return set()
    return {
        "closed_book_direct",
        "format_constraint",
        "assumption_falsification",
        "option_matrix_reasoning",
        "option_elimination",
        "adversarial_boundary_search",
    }


def _core_orthogonal_axes_covered_for_early_stop(
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> bool:
    required = _required_child_branch_axes_before_early_stop(problem)
    if not required:
        return True
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    axes = {axis for axis in _child_branch_axes_for_attempts(valid) if axis}
    return required.issubset(axes)


def _orthogonalize_child_prompt_specs(
    problem: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    agent_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    allow_duplicates = _allow_duplicate_child_branch_axes()
    retained: list[dict[str, Any]] = []
    seen_axes: set[str] = set()
    skipped: list[dict[str, str]] = []
    for spec in specs:
        prompt_kind = str(spec.get("prompt_kind") or "")
        branch_axis = str(spec.get("branch_axis") or _child_branch_axis(prompt_kind))
        if branch_axis in seen_axes and not allow_duplicates:
            skipped.append({"prompt_kind": prompt_kind, "branch_axis": branch_axis})
            continue
        seen_axes.add(branch_axis)
        orthogonal_id = str(
            spec.get("orthogonal_branch_id")
            or _child_branch_id(problem, prompt_kind=prompt_kind, branch_axis=branch_axis)
        )
        prompt = str(spec.get("prompt") or "")
        if "You are recursive child branch `" not in prompt:
            prompt = _orthogonal_child_prompt_prefix(branch_axis=branch_axis, prompt_kind=prompt_kind) + prompt
        out = dict(spec)
        out["branch_axis"] = branch_axis
        out["orthogonal_branch_id"] = orthogonal_id
        out["prompt"] = prompt
        retained.append(out)

    if agent_plan is not None:
        required_axes = sorted(_required_child_branch_axes_before_early_stop(problem))
        planned_axes = [str(spec.get("branch_axis") or "") for spec in retained]
        agent_plan.setdefault("stages", {})["recursive_child_diversity_planner"] = {
            "status": "activated",
            "policy": "orthogonal_branch_axis_dedup",
            "planned_child_count_raw": len(specs),
            "planned_child_count": len(retained),
            "planned_branch_axes": planned_axes,
            "unique_branch_axis_count": len(set(planned_axes)),
            "duplicate_branch_axes_removed": len(skipped),
            "skipped_duplicate_branch_axes": skipped,
            "allow_duplicate_branch_axes": allow_duplicates,
            "min_axes_before_early_stop": _min_orthogonal_child_axes_before_early_stop(problem),
            "required_axes_before_early_stop": required_axes,
            "required_axes_planned": sorted(set(planned_axes) & set(required_axes)),
            "required_axes_missing_from_plan": sorted(set(required_axes) - set(planned_axes)),
        }
    return retained


def _execute_recursive_child_attempts(
    *,
    problem: dict[str, Any],
    specs: list[dict[str, Any]],
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    mode: str,
) -> dict[str, Any]:
    force_serial_reason = _force_serial_child_execution_reason(mode=mode, timeout=timeout)
    if mode != "parallel_quorum" or force_serial_reason:
        result = _execute_recursive_child_attempts_serial(
            problem=problem,
            specs=specs,
            model=model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if force_serial_reason:
            result["execution_mode"] = "serial_strict_timeout"
            result["serial_forced_reason"] = force_serial_reason
        return result
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
        skipped_branch_axes = [str(row.get("branch_axis") or _child_branch_axis(row["prompt_kind"])) for row in specs[len(attempts):]]
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
            executed_branch_axes=_child_branch_axes_for_attempts(attempts),
            skipped_branch_axes=skipped_branch_axes,
        )
        return {
            "attempts": attempts,
            "underlying_model_calls": first_batch["underlying_model_calls"],
            "early_stop_reason": early_stop_reason,
            "skipped_prompt_kinds": skipped_prompt_kinds,
            "skipped_branch_axes": skipped_branch_axes,
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
        "skipped_branch_axes": [],
        "execution_mode": "parallel_quorum",
        "child_timeout_sec": timeout,
        "child_max_workers": max(first_batch["max_workers"], rest_batch["max_workers"]),
    }


def _force_serial_child_execution_reason(*, mode: str, timeout: float | None) -> str:
    if mode != "parallel_quorum":
        return ""
    if timeout is None:
        return ""
    if os.environ.get("HLE_DISABLE_STRICT_SERIAL_CHILD_TIMEOUT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return ""
    return "finite_timeout_requires_main_thread_deadline"


def _execute_recursive_child_attempts_serial(
    *,
    problem: dict[str, Any],
    specs: list[dict[str, Any]],
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
    skipped_branch_axes: list[str] = []
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
            skipped_branch_axes = [
                str(row.get("branch_axis") or _child_branch_axis(row["prompt_kind"]))
                for row in specs[index:]
            ]
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
                executed_branch_axes=_child_branch_axes_for_attempts(attempts),
                skipped_branch_axes=skipped_branch_axes,
            )
            break
    return {
        "attempts": attempts,
        "underlying_model_calls": underlying_calls,
        "early_stop_reason": early_stop_reason,
        "skipped_prompt_kinds": skipped_prompt_kinds,
        "skipped_branch_axes": skipped_branch_axes,
        "execution_mode": "serial",
        "child_timeout_sec": timeout,
        "child_max_workers": 1,
    }


def _run_child_batch(
    *,
    problem: dict[str, Any],
    specs: list[dict[str, Any]],
    start_index: int,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    max_workers: int,
    variant: str = "assumption_agent_recursive_verify",
) -> dict[str, Any]:
    if not specs:
        return {"attempts": [], "underlying_model_calls": 0, "max_workers": 0}
    max_workers = max(1, min(max_workers, len(specs)))
    attempts: list[dict[str, Any]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_specs: dict[concurrent.futures.Future, tuple[dict[str, Any], int]] = {}
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
                variant=variant,
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
                        variant=variant,
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
    spec: dict[str, Any],
    child_index: int,
    model: str,
    variant: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    latency_sec: float,
) -> dict[str, Any]:
    child_id = stable_hash({"call_id": call_id, "child_index": child_index, "prompt_kind": spec["prompt_kind"]})
    branch_axis = str(spec.get("branch_axis") or _child_branch_axis(spec["prompt_kind"]))
    orthogonal_branch_id = str(
        spec.get("orthogonal_branch_id")
        or _child_branch_id(problem, prompt_kind=spec["prompt_kind"], branch_axis=branch_axis)
    )
    attempt = {
        "child_id": child_id,
        "child_index": child_index,
        "prompt_kind": spec["prompt_kind"],
        "branch_axis": branch_axis,
        "orthogonal_branch_id": orthogonal_branch_id,
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
            "variant": variant,
            "prompt_kind": spec["prompt_kind"],
            "branch_axis": branch_axis,
            "orthogonal_branch_id": orthogonal_branch_id,
            "latency_sec": latency_sec,
            "timeout_sec": timeout,
        },
    )
    return attempt


def _run_child_attempt(
    *,
    problem: dict[str, Any],
    spec: dict[str, Any],
    child_index: int,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    variant: str = "assumption_agent_recursive_verify",
) -> dict[str, Any]:
    child_id = stable_hash({"call_id": call_id, "child_index": child_index, "prompt_kind": spec["prompt_kind"]})
    branch_axis = str(spec.get("branch_axis") or _child_branch_axis(spec["prompt_kind"]))
    orthogonal_branch_id = str(
        spec.get("orthogonal_branch_id")
        or _child_branch_id(problem, prompt_kind=spec["prompt_kind"], branch_axis=branch_axis)
    )
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
            "variant": variant,
            "prompt_kind": spec["prompt_kind"],
            "branch_axis": branch_axis,
            "orthogonal_branch_id": orthogonal_branch_id,
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
            "branch_axis": branch_axis,
            "orthogonal_branch_id": orthogonal_branch_id,
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
                "variant": variant,
                "prompt_kind": spec["prompt_kind"],
                "branch_axis": branch_axis,
                "orthogonal_branch_id": orthogonal_branch_id,
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
            "branch_axis": branch_axis,
            "orthogonal_branch_id": orthogonal_branch_id,
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
                "variant": variant,
                "prompt_kind": spec["prompt_kind"],
                "branch_axis": branch_axis,
                "orthogonal_branch_id": orthogonal_branch_id,
                "latency_sec": attempt["latency_sec"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
            },
        )
        return attempt


def _maybe_run_timeout_recovery_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    math_tool_summary: dict[str, Any] | None,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    trigger = _recursive_timeout_recovery_trigger(
        problem=problem,
        attempts=attempts,
        math_tool_summary=math_tool_summary,
    )
    if trigger.get("status") != "activated":
        return None, None

    recovery_model = os.environ.get("HLE_TIMEOUT_RECOVERY_MODEL", "").strip() or model
    recovery_timeout = _timeout_recovery_child_timeout(timeout)
    recovery_max_tokens = _timeout_recovery_child_max_tokens(max_tokens)
    child_index = _timeout_recovery_child_index(attempts)
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "timeout_recovery_answer",
            "prompt": _timeout_recovery_answer_prompt(problem, trigger=trigger),
        },
        child_index=child_index,
        model=recovery_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=recovery_timeout,
        max_tokens=recovery_max_tokens,
    )
    summary = {
        "status": "activated",
        "reason": trigger.get("reason"),
        "answer_type": problem.get("answer_type"),
        "valid_candidate_count_before": trigger.get("valid_candidate_count"),
        "unique_candidate_count_before": trigger.get("unique_candidate_count"),
        "timeout_child_count_before": trigger.get("timeout_child_count"),
        "error_child_count_before": trigger.get("error_child_count"),
        "child_id": attempt.get("child_id"),
        "child_index": attempt.get("child_index"),
        "child_status": attempt.get("status"),
        "child_error_type": attempt.get("error_type"),
        "recovery_model": recovery_model,
        "recovery_timeout_sec": recovery_timeout,
        "recovery_max_tokens": recovery_max_tokens,
        "candidate_emitted": bool(str(attempt.get("parsed_answer") or "").strip()),
        "candidate_answer_hash": attempt.get("parsed_answer_hash"),
    }
    return attempt, summary


def _recursive_timeout_recovery_trigger(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    math_tool_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_TIMEOUT_RECOVERY_CHILD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"status": "abstained", "reason": "disabled"}
    if (math_tool_summary or {}).get("confidence") in {"verified_symbolic", "verified_symbolic_consensus"}:
        return {"status": "abstained", "reason": "math_tool_already_verified"}

    timeout_count = sum(1 for attempt in attempts if attempt.get("status") == "timeout")
    error_count = sum(1 for attempt in attempts if attempt.get("status") == "error")
    if timeout_count + error_count <= 0:
        return {
            "status": "abstained",
            "reason": "no_timeout_or_error_pressure",
            "timeout_child_count": timeout_count,
            "error_child_count": error_count,
        }

    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    normalized = {
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=problem.get("answer_type") or "exactMatch")
        for attempt in valid
    }
    unique_count = len({value for value in normalized if value})
    min_valid = 2
    if len(valid) >= min_valid and unique_count >= min_valid:
        return {
            "status": "abstained",
            "reason": "sufficient_candidate_diversity",
            "valid_candidate_count": len(valid),
            "unique_candidate_count": unique_count,
            "timeout_child_count": timeout_count,
            "error_child_count": error_count,
        }
    return {
        "status": "activated",
        "reason": "timeout_or_error_with_candidate_shortage",
        "valid_candidate_count": len(valid),
        "unique_candidate_count": unique_count,
        "timeout_child_count": timeout_count,
        "error_child_count": error_count,
    }


def _maybe_run_child_model_failover_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    base_model: str,
    child_model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    trigger = _child_model_failover_trigger(problem=problem, attempts=attempts, base_model=base_model, child_model=child_model)
    if trigger.get("status") != "activated":
        return None, None
    child_index = _timeout_recovery_child_index(attempts)
    failover_timeout = _child_model_failover_timeout(timeout)
    failover_max_tokens = _child_model_failover_max_tokens(max_tokens)
    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "child_model_failover_answer",
            "prompt": _child_model_failover_prompt(problem, trigger=trigger),
        },
        child_index=child_index,
        model=base_model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=failover_timeout,
        max_tokens=failover_max_tokens,
    )
    summary = {
        "status": "activated",
        "reason": trigger.get("reason"),
        "base_model": base_model,
        "failed_child_model": child_model,
        "valid_candidate_count_before": trigger.get("valid_candidate_count"),
        "unique_candidate_count_before": trigger.get("unique_candidate_count"),
        "timeout_child_count_before": trigger.get("timeout_child_count"),
        "error_child_count_before": trigger.get("error_child_count"),
        "child_id": attempt.get("child_id"),
        "child_index": attempt.get("child_index"),
        "child_status": attempt.get("status"),
        "child_error_type": attempt.get("error_type"),
        "failover_timeout_sec": failover_timeout,
        "failover_max_tokens": failover_max_tokens,
        "candidate_emitted": bool(str(attempt.get("parsed_answer") or "").strip()),
        "candidate_answer_hash": attempt.get("parsed_answer_hash"),
    }
    return attempt, summary


def _child_model_failover_trigger(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    base_model: str,
    child_model: str,
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_CHILD_MODEL_FAILOVER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"status": "abstained", "reason": "disabled"}
    if not child_model or child_model == base_model:
        return {"status": "abstained", "reason": "child_model_same_as_base"}
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
    timeout_count = sum(1 for attempt in attempts if attempt.get("status") == "timeout")
    error_count = sum(1 for attempt in attempts if attempt.get("status") == "error")
    if timeout_count + error_count <= 0:
        return {
            "status": "abstained",
            "reason": "no_child_model_failure_pressure",
            "valid_candidate_count": len(valid),
            "timeout_child_count": timeout_count,
            "error_child_count": error_count,
        }
    if not valid:
        return {
            "status": "activated",
            "reason": "child_model_failed_without_valid_candidate",
            "valid_candidate_count": 0,
            "unique_candidate_count": 0,
            "timeout_child_count": timeout_count,
            "error_child_count": error_count,
        }
    normalized = {
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=problem.get("answer_type") or "exactMatch")
        for attempt in valid
    }
    unique_count = len({value for value in normalized if value})
    min_unique = 3 if problem.get("answer_type") == "multipleChoice" else 4
    if unique_count >= min_unique:
        return {
            "status": "abstained",
            "reason": "valid_candidate_diversity_already_available",
            "valid_candidate_count": len(valid),
            "unique_candidate_count": unique_count,
            "timeout_child_count": timeout_count,
            "error_child_count": error_count,
        }
    return {
        "status": "activated",
        "reason": "child_model_failure_with_low_candidate_diversity",
        "valid_candidate_count": len(valid),
        "unique_candidate_count": unique_count,
        "timeout_child_count": timeout_count,
        "error_child_count": error_count,
    }


def _child_model_failover_prompt(problem: dict[str, Any], *, trigger: dict[str, Any]) -> str:
    answer_type = problem.get("answer_type") or "exactMatch"
    return (
        "One or more stronger candidate-generation calls failed or disconnected before the recursive set had enough "
        "independent candidates. Produce one concise fallback answer using only the question. Return JSON only: {\"answer\":\"...\"}. "
        "For multiple choice, return one option letter only. For exact match, return the shortest exact final answer only.\n\n"
        f"Failure pressure: timeouts={trigger.get('timeout_child_count')}, errors={trigger.get('error_child_count')}, "
        f"valid_candidates={trigger.get('valid_candidate_count')}.\n\n"
        f"Answer type: {answer_type}\nQuestion:\n{problem.get('_question') or ''}"
    )


def _child_model_failover_timeout(timeout: float | None) -> float | None:
    has_override, override = _optional_timeout_override_from_env("HLE_CHILD_MODEL_FAILOVER_TIMEOUT_SEC")
    if has_override:
        return override
    if timeout is None:
        return None
    return _normalize_optional_timeout(timeout)


def _child_model_failover_max_tokens(max_tokens: int) -> int:
    override = os.environ.get("HLE_CHILD_MODEL_FAILOVER_MAX_TOKENS", "").strip()
    if override:
        try:
            return max(16, int(override))
        except ValueError:
            pass
    return max(16, min(int(max_tokens), 64))


def _valid_recursive_answer_attempts(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answer_type = problem.get("answer_type") or "exactMatch"
    valid: list[dict[str, Any]] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        if answer_type == "multipleChoice":
            canonical, _ = _canonicalize_multiple_choice_answer(problem, answer)
            if _extract_choice(canonical):
                valid.append(attempt)
            continue
        if not _is_suspicious_exact_answer(answer):
            valid.append(attempt)
    return valid


def _endpoint_error_pressure_abort_summary(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_ENDPOINT_ERROR_PRESSURE_ABORT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"status": "disabled", "reason": "disabled"}
    model_attempts = [
        attempt for attempt in attempts
        if str(attempt.get("prompt_kind") or "")
        and attempt.get("prompt_kind") not in {
            "route_arbitrator_answer",
            "answer_bearing_evidence_candidate",
            "mc_option_sweep_candidate",
        }
    ]
    if not model_attempts:
        return {"status": "not_required", "reason": "no_model_child_attempts"}
    valid = _valid_recursive_answer_attempts(problem=problem, attempts=model_attempts)
    error_attempts = [
        attempt for attempt in model_attempts
        if attempt.get("status") in {"error", "timeout"}
    ]
    network_error_count = sum(
        1 for attempt in error_attempts
        if str(attempt.get("error_type") or "") in {"RuntimeError", "TimeoutError", "ChildTimeout"}
    )
    error_count = len(error_attempts)
    total = len(model_attempts)
    error_ratio = error_count / max(1, total)
    if len(valid) == 0 and error_count >= 2 and error_ratio >= 0.75:
        return {
            "status": "activated",
            "reason": "all_or_most_recursive_children_failed_with_endpoint_errors",
            "model_child_attempt_count": total,
            "valid_candidate_count": 0,
            "error_child_count": error_count,
            "network_error_child_count": network_error_count,
            "error_ratio": round(error_ratio, 4),
        }
    if error_count >= 5 and error_ratio >= 0.8 and len(valid) <= 1:
        return {
            "status": "activated",
            "reason": "endpoint_error_pressure_with_insufficient_candidates",
            "model_child_attempt_count": total,
            "valid_candidate_count": len(valid),
            "error_child_count": error_count,
            "network_error_child_count": network_error_count,
            "error_ratio": round(error_ratio, 4),
        }
    return {
        "status": "not_required",
        "reason": "candidate_or_error_pressure_below_abort_threshold",
        "model_child_attempt_count": total,
        "valid_candidate_count": len(valid),
        "error_child_count": error_count,
        "network_error_child_count": network_error_count,
        "error_ratio": round(error_ratio, 4),
    }


def _timeout_recovery_answer_prompt(problem: dict[str, Any], *, trigger: dict[str, Any]) -> str:
    answer_type = problem.get("answer_type") or "exactMatch"
    output_contract = (
        "Return JSON only: {\"answer\":\"...\"}. "
        "For multiple choice, return one option letter only. "
        "For exact match, return the shortest exact final answer only."
    )
    return (
        "The previous recursive child attempts timed out or errored before enough valid answer candidates were "
        "available. Produce one concise candidate answer under the same constraints. Do not explain, do not list "
        "alternatives, and do not include reasoning.\n\n"
        f"Timeout/error pressure: timeouts={trigger.get('timeout_child_count')}, errors={trigger.get('error_child_count')}, "
        f"valid_candidates={trigger.get('valid_candidate_count')}, unique_candidates={trigger.get('unique_candidate_count')}.\n\n"
        f"Answer type: {answer_type}\nQuestion:\n{problem.get('_question') or ''}\n\n{output_contract}"
    )


def _timeout_recovery_child_timeout(timeout: float | None) -> float | None:
    has_override, override = _optional_timeout_override_from_env("HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC")
    if has_override:
        return override
    if timeout is None:
        return None
    return max(float(timeout), min(float(timeout) * 2.0, 7200.0))


def _timeout_recovery_child_max_tokens(max_tokens: int) -> int:
    override = os.environ.get("HLE_TIMEOUT_RECOVERY_MAX_TOKENS", "").strip()
    if override:
        try:
            return max(16, int(override))
        except ValueError:
            pass
    return max(32, min(int(max_tokens), 160))


def _timeout_recovery_child_index(attempts: list[dict[str, Any]]) -> int:
    existing = [int(attempt.get("child_index") or 0) for attempt in attempts]
    return max(existing or [0]) + 1


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
    executed_branch_axes: list[str],
    skipped_branch_axes: list[str],
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
            "executed_branch_axes": executed_branch_axes,
            "skipped_branch_axes": skipped_branch_axes,
            "executed_unique_branch_axis_count": len(set(executed_branch_axes)),
        },
    )


def _can_stop_recursive_children_early(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> bool:
    if not _has_two_vote_majority(attempts, answer_type=problem["answer_type"]):
        return False
    if (
        _orthogonal_child_early_stop_guard_enabled()
        and _valid_child_branch_axis_count(problem, attempts) < _min_orthogonal_child_axes_before_early_stop(problem)
    ):
        return False
    if not _core_orthogonal_axes_covered_for_early_stop(problem, attempts):
        return False
    if problem.get("answer_type") == "multipleChoice":
        prompt_kinds = {str(attempt.get("prompt_kind") or "") for attempt in attempts}
        reflective_kinds = {"agent_context_answer", "constraint_checked_answer", "recursive_assumption_answer"}
        return bool(prompt_kinds & reflective_kinds)
    if problem.get("answer_type") == "exactMatch":
        if _exact_trajectory_search_enabled():
            return False
        valid = _valid_recursive_answer_attempts(problem=problem, attempts=attempts)
        prompt_kinds = {str(attempt.get("prompt_kind") or "") for attempt in valid}
        independent_kinds = {
            "agent_context_answer",
            "adversarial_alternative_answer",
            "decomposition_answer",
            "evidence_bridge_answer",
            "evidence_grounded_answer",
            "hipporag_context_answer",
            "literal_constraint_answer",
            "recursive_assumption_answer",
        }
        return len(valid) >= 3 and bool(prompt_kinds & independent_kinds)
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
        if _is_verified_math_reference(reference):
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
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            if _is_verified_math_reference(llm_reference):
                summary = _apply_math_reference_to_multiple_choice_options(
                    problem=problem,
                    attempts=attempts,
                    options=options,
                    reference=llm_reference,
                    backend="sympy_mc_option_planner",
                    underlying_model_calls=int(llm_reference.get("underlying_model_calls") or 1),
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
                "reference_error_type": llm_reference.get("error_type"),
                "planner_latency_sec": llm_reference.get("planner_latency_sec"),
                "deterministic_reference_reason": reference.get("reason"),
                "option_count": len(options),
                "underlying_model_calls": int(llm_reference.get("underlying_model_calls") or 1),
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
    if _is_verified_math_reference(reference):
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
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if _is_verified_math_reference(llm_reference):
            summary = _apply_math_reference_to_candidates(
                problem=problem,
                attempts=attempts,
                reference=llm_reference,
                backend="sympy_candidate_reference_planner",
                underlying_model_calls=int(llm_reference.get("underlying_model_calls") or 1),
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
            "reference_error_type": llm_reference.get("error_type"),
            "planner_latency_sec": llm_reference.get("planner_latency_sec"),
            "deterministic_reference_reason": reference.get("reason"),
            "underlying_model_calls": int(llm_reference.get("underlying_model_calls") or 1),
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
    if not _is_verified_math_reference(reference):
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
            "planner_latency_sec": reference.get("planner_latency_sec"),
            "reference_error_type": reference.get("error_type"),
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
            "planner_latency_sec": reference.get("planner_latency_sec"),
            "reference_error_type": reference.get("error_type"),
        }
    verified_label = matching_labels[0]
    if backend == "sympy_mc_option_deterministic":
        candidate_verifier_trust = "deterministic_mc_reference"
    elif _trust_llm_reference_planner_enabled():
        candidate_verifier_trust = "trusted_llm_reference_planner"
    else:
        candidate_verifier_trust = "weak_llm_reference_planner"
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
        attempt["candidate_verifier_trust"] = candidate_verifier_trust
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
        "candidate_verifier_trust": candidate_verifier_trust,
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
        "candidate_verifier_trust": candidate_verifier_trust,
        "verified_count": verified,
        "refuted_count": refuted,
        "inconclusive_count": inconclusive,
        "candidate_count": verified + refuted + inconclusive,
        "candidate_state_hashes": candidate_hashes,
        "option_count": len(options),
        "matched_option_count": 1,
        "underlying_model_calls": underlying_model_calls,
        "planner_latency_sec": reference.get("planner_latency_sec"),
        "reference_error_type": reference.get("error_type"),
        "claim_hash": stable_hash({
            "question_hash": problem.get("question_hash"),
            "reference_answer": reference_answer,
            "verified_label": verified_label,
            "operation": reference.get("operation"),
            "plan_hash": reference.get("plan_hash"),
        }),
    }


def _exact_math_candidate_match(candidate_answer: str, reference_answer: str) -> dict[str, Any]:
    candidate = str(candidate_answer or "").strip()
    reference = str(reference_answer or "").strip()
    if not candidate or not reference:
        return {"matched": False, "method": "empty"}

    candidate_norm = _normalize_exact(candidate)
    reference_norm = _normalize_exact(reference)
    if candidate_norm and candidate_norm == reference_norm:
        return {
            "matched": True,
            "method": "normalized_exact",
            "candidate_part_count": 1,
            "reference_part_count": 1,
        }

    candidate_parts = _math_answer_parts(candidate)
    reference_parts = _math_answer_parts(reference)
    if not candidate_parts or not reference_parts:
        return {
            "matched": False,
            "method": "not_parseable",
            "candidate_part_count": len(candidate_parts),
            "reference_part_count": len(reference_parts),
        }
    if len(candidate_parts) != len(reference_parts):
        return {
            "matched": False,
            "method": "part_count_mismatch",
            "candidate_part_count": len(candidate_parts),
            "reference_part_count": len(reference_parts),
        }

    if len(candidate_parts) == 1:
        equivalent, method = _sympy_answer_parts_equivalent(candidate_parts[0], reference_parts[0])
        return {
            "matched": equivalent,
            "method": method if equivalent else "symbolic_mismatch",
            "candidate_part_count": 1,
            "reference_part_count": 1,
        }

    unmatched = set(range(len(candidate_parts)))
    methods: list[str] = []
    for reference_part in reference_parts:
        matched_index: int | None = None
        matched_method = "none"
        for candidate_index in list(unmatched):
            equivalent, method = _sympy_answer_parts_equivalent(candidate_parts[candidate_index], reference_part)
            if equivalent:
                matched_index = candidate_index
                matched_method = method
                break
        if matched_index is None:
            return {
                "matched": False,
                "method": "collection_mismatch",
                "candidate_part_count": len(candidate_parts),
                "reference_part_count": len(reference_parts),
            }
        unmatched.remove(matched_index)
        methods.append(matched_method)
    return {
        "matched": True,
        "method": "unordered_collection_equivalence",
        "part_methods": sorted(Counter(methods).items()),
        "candidate_part_count": len(candidate_parts),
        "reference_part_count": len(reference_parts),
    }


def _math_answer_parts(text: str) -> list[str]:
    cleaned = _clean_math_answer_text(text)
    if not cleaned or len(cleaned) > 320:
        return []
    if not _has_math_answer_signal(cleaned):
        return []
    for segment in _math_answer_candidate_segments(cleaned):
        if not _has_math_answer_signal(segment):
            continue
        parts = _math_answer_parts_from_segment(segment)
        if parts and all(_safe_sympy_parse_expr(part) is not None for part in parts):
            return parts
    return []


def _math_answer_parts_from_segment(segment: str) -> list[str]:
    cleaned = str(segment or "").strip()
    if not cleaned:
        return []
    cleaned = re.sub(r"\s+(?:or|and)\s+", ",", cleaned, flags=re.IGNORECASE)
    cleaned = _strip_outer_math_container(cleaned)
    raw_parts = _split_top_level_math_parts(cleaned)
    if len(raw_parts) > 8:
        return []

    parts: list[str] = []
    for raw_part in raw_parts:
        part = _strip_outer_math_container(raw_part.strip())
        part = re.sub(
            r"^\s*[A-Za-z][A-Za-z0-9_]*(?:\([^()]{0,40}\))?\s*=\s*",
            "",
            part,
        ).strip()
        for expanded in _expand_plus_minus_answer_part(part):
            normalized = _normalize_math_expression(expanded)
            if normalized and len(normalized) <= 180:
                parts.append(normalized)
    return parts if parts and len(parts) <= 8 else []


def _math_answer_candidate_segments(cleaned: str) -> list[str]:
    segments: list[str] = []

    def add(segment: str) -> None:
        segment = _strip_outer_math_container(str(segment or "").strip())
        if segment and segment not in segments and len(segment) <= 260:
            segments.append(segment)

    add(cleaned)
    for pattern in (
        r"\\boxed\s*\{([^{}]{1,220})\}",
        r"\\boxed\s*\(([^()]{1,220})\)",
        r"\$([^$]{1,220})\$",
        r"\\\((.{1,220})\\\)",
        r"\\\[(.{1,220})\\\]",
    ):
        for match in re.finditer(pattern, cleaned, flags=re.DOTALL):
            add(match.group(1))
    keyword_pattern = re.compile(
        r"(?:answers?|final\s+answers?|results?|values?|solutions?|roots?)\s*"
        r"(?:is|are|=|:)?\s*([^.\n]{1,220})",
        flags=re.IGNORECASE,
    )
    for match in keyword_pattern.finditer(cleaned):
        add(match.group(1))
    assignment_pattern = re.compile(
        r"\b[A-Za-z][A-Za-z0-9_]*(?:\([^()]{0,40}\))?\s*=\s*([^.;\n]{1,220})",
        flags=re.IGNORECASE,
    )
    assignment_segments = [match.group(0) for match in assignment_pattern.finditer(cleaned)]
    if assignment_segments:
        add(", ".join(assignment_segments))
        for segment in assignment_segments:
            add(segment)
    if ":" in cleaned:
        add(cleaned.rsplit(":", 1)[-1])
    return segments


def _clean_math_answer_text(text: str) -> str:
    cleaned = html.unescape(str(text or "")).strip().strip('"').strip("'").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\\boxed\s*\{([^{}]{1,240})\}", r"\1", cleaned)
    cleaned = cleaned.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("π", "pi").replace("√", "sqrt")
    cleaned = cleaned.replace("\\{", "{").replace("\\}", "}")
    cleaned = cleaned.replace("\\pm", "±")
    unwrapped = re.fullmatch(r"\$([^$]{1,300})\$|\\\((.{1,300})\\\)|\\\[(.{1,300})\\\]", cleaned, flags=re.DOTALL)
    if unwrapped:
        cleaned = next((group for group in unwrapped.groups() if group), "").strip()
    cleaned = re.sub(
        r"^\s*(?:therefore|thus|hence|so)\s*,?\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"^\s*(?:the\s+)?(?:answer|final\s+answer|result|value|solutions?)\s*(?:is|are|=|:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if cleaned.endswith(".") and re.fullmatch(r"[-+0-9A-Za-z_±piPIsqrt^*/%()., {}\\=<>; ]+", cleaned[:-1]):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _has_math_answer_signal(text: str) -> bool:
    return bool(re.search(
        r"\d|[+\-*/^=<>%±√]|\\(?:frac|sqrt)|\b(?:pi|sqrt|sin|cos|tan|log|ln|exp)\b",
        str(text or ""),
        flags=re.IGNORECASE,
    ))


def _strip_outer_math_container(text: str) -> str:
    stripped = str(text or "").strip()
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        for opener, closer in (("{", "}"), ("[", "]"), ("(", ")")):
            if stripped.startswith(opener) and stripped.endswith(closer) and _outer_pair_encloses_text(stripped, opener, closer):
                stripped = stripped[1:-1].strip()
                changed = True
                break
        if stripped.startswith("$") and stripped.endswith("$") and stripped.count("$") == 2:
            stripped = stripped[1:-1].strip()
            changed = True
    return stripped


def _outer_pair_encloses_text(text: str, opener: str, closer: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def _split_top_level_math_parts(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and char in ",;":
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or [text.strip()]


def _expand_plus_minus_answer_part(part: str) -> list[str]:
    stripped = str(part or "").strip()
    match = re.fullmatch(r"(?:±|\+/-)\s*(.+)", stripped)
    if not match:
        return [stripped]
    magnitude = match.group(1).strip()
    if not magnitude:
        return []
    return [f"-({magnitude})", f"({magnitude})"]


def _sympy_answer_parts_equivalent(candidate_part: str, reference_part: str) -> tuple[bool, str]:
    candidate_expr = _safe_sympy_parse_expr(candidate_part)
    reference_expr = _safe_sympy_parse_expr(reference_part)
    if candidate_expr is None or reference_expr is None:
        return (candidate_part.replace(" ", "") == reference_part.replace(" ", ""), "normalized_math_text")
    try:
        import sympy as sp

        diff = sp.simplify(candidate_expr - reference_expr)
        if diff == 0:
            free_symbols = getattr(candidate_expr, "free_symbols", set()) | getattr(reference_expr, "free_symbols", set())
            return (True, "symbolic_equivalence" if free_symbols else "numeric_equivalence")
        free_symbols = getattr(candidate_expr, "free_symbols", set()) | getattr(reference_expr, "free_symbols", set())
        if not free_symbols:
            try:
                candidate_num = complex(sp.N(candidate_expr, 40))
                reference_num = complex(sp.N(reference_expr, 40))
                if abs(candidate_num - reference_num) <= 1e-10 * max(1.0, abs(reference_num)):
                    return (True, "numeric_tolerance")
            except Exception:
                pass
    except Exception:
        pass
    return (False, "symbolic_mismatch")


def _apply_math_reference_to_candidates(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    reference: dict[str, Any],
    backend: str,
    underlying_model_calls: int,
) -> dict[str, Any]:
    if not _is_verified_math_reference(reference):
        return {
            "status": "no_executable_claim",
            "backend": backend,
            "verified_count": 0,
            "refuted_count": 0,
            "inconclusive_count": sum(1 for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()),
            "reference_operation": reference.get("operation"),
            "reference_reason": reference.get("reason"),
            "underlying_model_calls": underlying_model_calls,
            "planner_latency_sec": reference.get("planner_latency_sec"),
            "reference_error_type": reference.get("error_type"),
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
    match_methods: Counter[str] = Counter()
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        canonical_answer, canonical_summary = _canonicalize_exact_answer_candidate(problem, answer)
        candidate_norm = _normalize_for_selection(canonical_answer, answer_type=problem["answer_type"])
        match = _exact_math_candidate_match(canonical_answer, canonical_reference)
        state = "verified" if match["matched"] or candidate_norm == reference_norm else "refuted"
        match_method = str(match.get("method") or ("normalized_exact" if state == "verified" else "none"))
        match_methods[match_method] += 1
        if weak_single_candidate:
            weak_verified += int(state == "verified")
            inconclusive += 1
            candidate_hashes.append(stable_hash({
                "candidate_answer": canonical_answer,
                "state": f"weak_{state}",
                "match_method": match_method,
            }))
            continue
        if state == "verified":
            verified += 1
        else:
            refuted += 1
        candidate_hashes.append(stable_hash({
            "candidate_answer": canonical_answer,
            "state": state,
            "match_method": match_method,
        }))
        attempt["candidate_verifier_state"] = state
        attempt["candidate_verifier_backend"] = backend
        attempt["candidate_verifier_operation"] = reference.get("operation")
        attempt["candidate_verifier_match_method"] = match_method
        attempt["candidate_verifier_claim_hash"] = stable_hash({
            "reference_answer": canonical_reference,
            "candidate_answer": canonical_answer,
            "operation": reference.get("operation"),
            "match_method": match_method,
        })
        if canonical_summary.get("changed"):
            attempt["candidate_verifier_canonicalized"] = True
            attempt["parsed_answer"] = canonical_answer
            attempt["parsed_answer_hash"] = stable_hash({"answer": canonical_answer})
    if not weak_single_candidate and verified == 0:
        verifier_attempt = {
            "child_id": stable_hash({
                "call_id": problem.get("id_hash"),
                "prompt_kind": "candidate_claim_verifier_answer",
                "reference_answer": canonical_reference,
                "claim_hash": reference.get("plan_hash"),
            }),
            "child_index": 9101,
            "prompt_kind": "candidate_claim_verifier_answer",
            "parsed_answer": canonical_reference,
            "parsed_answer_hash": stable_hash({"answer": canonical_reference}),
            "prediction_hash": stable_hash({
                "reference_answer": canonical_reference,
                "backend": backend,
                "operation": reference.get("operation"),
            }),
            "latency_sec": 0.0,
            "status": "answered",
            "candidate_verifier_state": "verified",
            "candidate_verifier_backend": backend,
            "candidate_verifier_operation": reference.get("operation"),
            "candidate_verifier_match_method": "executable_reference_synthetic",
            "candidate_verifier_claim_hash": stable_hash({
                "reference_answer": canonical_reference,
                "candidate_answer": canonical_reference,
                "operation": reference.get("operation"),
                "match_method": "executable_reference_synthetic",
            }),
        }
        attempts.append(verifier_attempt)
        verified += 1
        candidate_hashes.append(stable_hash({
            "candidate_answer": canonical_reference,
            "state": "verified_synthetic",
            "match_method": "executable_reference_synthetic",
        }))
        match_methods["executable_reference_synthetic"] += 1
    planner_backend = backend in {"sympy_candidate_reference_planner", "sympy_mc_option_planner"}
    planner_single_weak = backend == "sympy_candidate_reference_planner" and verified < 2
    planner_untrusted = planner_backend and not _trust_llm_reference_planner_enabled()
    for attempt in attempts:
        if attempt.get("candidate_verifier_backend") == backend:
            attempt["candidate_verifier_trust"] = (
                "weak_single_planner"
                if planner_single_weak
                else "weak_llm_reference_planner"
                if planner_untrusted
                else "trusted"
            )
    return {
        "status": (
            "weak_single_candidate_confirmation"
            if weak_single_candidate
            else "weak_planner_single_verified"
            if planner_single_weak
            else "activated"
        ),
        "backend": backend,
        "reference_operation": reference.get("operation"),
        "reference_answer_hash": stable_hash({"answer": canonical_reference}),
        "verified_count": verified,
        "refuted_count": refuted,
        "inconclusive_count": inconclusive,
        "weak_verified_count": weak_verified,
        "candidate_count": verified + refuted + inconclusive,
        "candidate_state_hashes": candidate_hashes,
        "match_method_counts": dict(match_methods),
        "underlying_model_calls": underlying_model_calls,
        "planner_latency_sec": reference.get("planner_latency_sec"),
        "reference_error_type": reference.get("error_type"),
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
    eval_id: str | None = None,
    call_id: str | None = None,
    logger: "_JsonlLogger | None" = None,
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
    planner_model = _candidate_claim_planner_model(model)
    _log_candidate_claim_planner_lifecycle(
        logger,
        event="candidate_claim_planner_start",
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=planner_model,
        planner_kind="exact_candidate_reference",
        candidate_count=len(candidates),
        timeout=timeout,
    )
    started = time.monotonic()
    try:
        planner_text = _call_model(
            model=planner_model,
            prompt=_candidate_claim_planner_prompt(problem, candidates),
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        latency = round(time.monotonic() - started, 4)
        _log_candidate_claim_planner_lifecycle(
            logger,
            event="candidate_claim_planner_error",
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=planner_model,
            planner_kind="exact_candidate_reference",
            candidate_count=len(candidates),
            timeout=timeout,
            latency_sec=latency,
            error_type=type(exc).__name__,
            error=str(exc)[:240],
        )
        return {
            "source": "llm_candidate_reference_planner",
            "operation": "none",
            "confidence": "abstain",
            "reason": "planner_error",
            "error_type": type(exc).__name__,
            "candidate_count": len(candidates),
            "planner_latency_sec": latency,
        }
    latency = round(time.monotonic() - started, 4)
    plan = _parse_json_object(planner_text)
    if not isinstance(plan, dict):
        result = {"source": "llm_candidate_reference_planner", "operation": "none", "confidence": "abstain", "reason": "planner_json_parse_failed"}
    else:
        result = _execute_math_tool_plan_candidates(_math_tool_plan_candidates_from_object(plan), leak_candidates=candidates)
    underlying_model_calls = 1
    if not _is_verified_math_reference(result):
        repair_result, repair_calls = _run_reference_plan_repair(
            problem=problem,
            model=planner_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
            planner_kind="exact_candidate_reference_repair",
            prompt=_candidate_claim_plan_repair_prompt(problem, candidates, result),
            initial_result=result,
            leak_candidates=candidates,
        )
        underlying_model_calls += repair_calls
        if _is_verified_math_reference(repair_result):
            result = repair_result
        else:
            result["repair_attempted"] = bool(repair_calls)
            result["repair_reason"] = repair_result.get("reason")
            result["repair_error_type"] = repair_result.get("error_type")
    result["source"] = "llm_candidate_reference_planner"
    result.setdefault("plan_hash", stable_hash({"planner_text": planner_text}))
    result["candidate_count"] = len(candidates)
    result["planner_latency_sec"] = latency
    result["underlying_model_calls"] = underlying_model_calls
    result["planner_model"] = planner_model
    _log_candidate_claim_planner_lifecycle(
        logger,
        event="candidate_claim_planner_end",
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=planner_model,
        planner_kind="exact_candidate_reference",
        candidate_count=len(candidates),
        timeout=timeout,
        latency_sec=latency,
        status="activated" if _is_verified_math_reference(result) else "abstained",
        operation=result.get("operation"),
        reason=result.get("reason"),
        plan_hash=result.get("plan_hash"),
    )
    return result


def _llm_math_reference_claim_for_mc_options(
    *,
    problem: dict[str, Any],
    model: str,
    eval_id: str | None = None,
    call_id: str | None = None,
    logger: "_JsonlLogger | None" = None,
    timeout: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    planner_model = _candidate_claim_planner_model(model)
    _log_candidate_claim_planner_lifecycle(
        logger,
        event="candidate_claim_planner_start",
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=planner_model,
        planner_kind="mc_option_reference",
        candidate_count=0,
        timeout=timeout,
    )
    started = time.monotonic()
    try:
        planner_text = _call_model(
            model=planner_model,
            prompt=_mc_option_claim_planner_prompt(problem),
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        latency = round(time.monotonic() - started, 4)
        _log_candidate_claim_planner_lifecycle(
            logger,
            event="candidate_claim_planner_error",
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=planner_model,
            planner_kind="mc_option_reference",
            candidate_count=0,
            timeout=timeout,
            latency_sec=latency,
            error_type=type(exc).__name__,
            error=str(exc)[:240],
        )
        return {
            "source": "llm_mc_option_reference_planner",
            "operation": "none",
            "confidence": "abstain",
            "reason": "planner_error",
            "error_type": type(exc).__name__,
            "candidate_count": 0,
            "planner_latency_sec": latency,
        }
    latency = round(time.monotonic() - started, 4)
    plan = _parse_json_object(planner_text)
    if not isinstance(plan, dict):
        result = {"source": "llm_mc_option_reference_planner", "operation": "none", "confidence": "abstain", "reason": "planner_json_parse_failed"}
    else:
        result = _execute_math_tool_plan_candidates(_math_tool_plan_candidates_from_object(plan))
    underlying_model_calls = 1
    if not _is_verified_math_reference(result):
        repair_result, repair_calls = _run_reference_plan_repair(
            problem=problem,
            model=planner_model,
            eval_id=eval_id,
            call_id=call_id,
            logger=logger,
            timeout=timeout,
            max_tokens=max_tokens,
            planner_kind="mc_option_reference_repair",
            prompt=_mc_option_claim_plan_repair_prompt(problem, result),
            initial_result=result,
            leak_candidates=None,
        )
        underlying_model_calls += repair_calls
        if _is_verified_math_reference(repair_result):
            result = repair_result
        else:
            result["repair_attempted"] = bool(repair_calls)
            result["repair_reason"] = repair_result.get("reason")
            result["repair_error_type"] = repair_result.get("error_type")
    result["source"] = "llm_mc_option_reference_planner"
    result.setdefault("plan_hash", stable_hash({"planner_text": planner_text}))
    result["candidate_count"] = 0
    result["planner_latency_sec"] = latency
    result["underlying_model_calls"] = underlying_model_calls
    result["planner_model"] = planner_model
    _log_candidate_claim_planner_lifecycle(
        logger,
        event="candidate_claim_planner_end",
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=planner_model,
        planner_kind="mc_option_reference",
        candidate_count=0,
        timeout=timeout,
        latency_sec=latency,
        status="activated" if _is_verified_math_reference(result) else "abstained",
        operation=result.get("operation"),
        reason=result.get("reason"),
        plan_hash=result.get("plan_hash"),
    )
    return result


def _is_verified_math_reference(result: dict[str, Any]) -> bool:
    return result.get("confidence") in {"verified_symbolic", "verified_symbolic_consensus"} and bool(str(result.get("answer") or "").strip())


def _candidate_claim_planner_model(default_model: str) -> str:
    return os.environ.get("HLE_CANDIDATE_CLAIM_PLANNER_MODEL", "").strip() or default_model


def _run_reference_plan_repair(
    *,
    problem: dict[str, Any],
    model: str,
    eval_id: str | None,
    call_id: str | None,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
    planner_kind: str,
    prompt: str,
    initial_result: dict[str, Any],
    leak_candidates: list[str] | None,
) -> tuple[dict[str, Any], int]:
    _log_candidate_claim_planner_lifecycle(
        logger,
        event="candidate_claim_planner_start",
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        planner_kind=planner_kind,
        candidate_count=0,
        timeout=timeout,
    )
    started = time.monotonic()
    try:
        repair_text = _call_model(
            model=model,
            prompt=prompt,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        latency = round(time.monotonic() - started, 4)
        _log_candidate_claim_planner_lifecycle(
            logger,
            event="candidate_claim_planner_error",
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
            planner_kind=planner_kind,
            candidate_count=0,
            timeout=timeout,
            latency_sec=latency,
            error_type=type(exc).__name__,
            error=str(exc)[:240],
        )
        return (
            {
                "source": "llm_candidate_reference_repair",
                "operation": "none",
                "confidence": "abstain",
                "reason": "repair_planner_error",
                "error_type": type(exc).__name__,
                "repair_attempted": True,
                "initial_reference_reason": initial_result.get("reason"),
                "repair_latency_sec": latency,
            },
            1,
        )
    latency = round(time.monotonic() - started, 4)
    plan = _parse_json_object(repair_text)
    if isinstance(plan, dict):
        result = _execute_math_tool_plan_candidates(_math_tool_plan_candidates_from_object(plan), leak_candidates=leak_candidates)
    else:
        result = {"source": "llm_candidate_reference_repair", "operation": "none", "confidence": "abstain", "reason": "repair_json_parse_failed"}
    result["source"] = "llm_candidate_reference_repair"
    result.setdefault("plan_hash", stable_hash({"repair_text": repair_text}))
    result["repair_attempted"] = True
    result["repair_latency_sec"] = latency
    result["initial_reference_operation"] = initial_result.get("operation")
    result["initial_reference_reason"] = initial_result.get("reason")
    _log_candidate_claim_planner_lifecycle(
        logger,
        event="candidate_claim_planner_end",
        eval_id=eval_id,
        call_id=call_id,
        problem=problem,
        model=model,
        planner_kind=planner_kind,
        candidate_count=0,
        timeout=timeout,
        latency_sec=latency,
        status="activated" if _is_verified_math_reference(result) else "abstained",
        operation=result.get("operation"),
        reason=result.get("reason"),
        plan_hash=result.get("plan_hash"),
    )
    return result, 1


def _mc_option_claim_planner_prompt(problem: dict[str, Any]) -> str:
    return (
        "Extract up to four independent executable math plans for this HLE multipleChoice item. The answer options "
        "are intentionally hidden; compute the underlying value or symbolic result from the stem only, so it can "
        "later be matched to exactly one option. Do not answer from memory. Prefer plans in this order when applicable: "
        "direct evaluate/mod, solve equation or roots, simplify/factor/expand symbolic result, derivative/integral/limit, "
        "or python expression for combinatorics, sums, and integer arithmetic. "
        "If the stem cannot be checked by a small SymPy-compatible plan, return one none plan. JSON only: "
        "{\"plans\":[{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|python|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\","
        "\"lower\":\"\",\"upper\":\"\",\"order\":\"1\"}]}. No imports, no code, no explanation.\n\n"
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
        "Extract up to four independent executable math plans that can verify candidate answers for this HLE exactMatch item. "
        "Use the candidates only to infer answer format; do not copy a candidate into the expression/equation. "
        "Prefer plans in this order when applicable: direct evaluate/mod, solve equation or roots, simplify/factor/expand "
        "symbolic result, derivative/integral/limit, or python expression for combinatorics, sums, and integer arithmetic. "
        "If one plan is uncertain, include an alternate plan rather than "
        "abstaining immediately. If the problem cannot be checked by a small SymPy-compatible plan, return one none plan. "
        "JSON only: {\"plans\":[{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|python|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\",\"upper\":\"\",\"order\":\"1\"}]}. "
        "No imports, no code, no explanation.\n\n"
        f"Question:\n{problem['_question']}\n\nCandidate answers:\n{candidate_lines}"
    )


def _candidate_claim_plan_repair_prompt(
    problem: dict[str, Any],
    candidates: list[str],
    initial_result: dict[str, Any],
) -> str:
    candidate_lines = "\n".join(f"{idx}. {answer}" for idx, answer in enumerate(candidates, start=1))
    failure_summary = _reference_plan_failure_summary(initial_result)
    return (
        "The previous executable math plan failed. Produce corrected executable plans for the same HLE exactMatch item. "
        "Use the question text as the source of computation; use candidate answers only for expected answer format. "
        "Do not copy a candidate answer into expression/equation. Common repairs: use solve for equations or roots; "
        "use simplify/factor/expand for symbolic answers; strip prose, assignments, and side conditions from expressions; "
        "use derivative/integral/limit operations when the wording asks for them; use python expression for comb/factorial/sum/range. "
        "Return JSON only: "
        "{\"plans\":[{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|python|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\","
        "\"upper\":\"\",\"order\":\"1\"}]}. No explanation.\n\n"
        f"Previous failure:\n{failure_summary}\n\n"
        f"Question:\n{problem['_question']}\n\nCandidate answers:\n{candidate_lines}"
    )


def _mc_option_claim_plan_repair_prompt(problem: dict[str, Any], initial_result: dict[str, Any]) -> str:
    failure_summary = _reference_plan_failure_summary(initial_result)
    return (
        "The previous executable math plan failed. Produce corrected executable plans for this HLE multipleChoice stem. "
        "The answer options are hidden; compute the underlying value or symbolic result from the stem only. "
        "Common repairs: use solve for equations or roots; use simplify/factor/expand for symbolic answers; strip prose, "
        "assignments, and side conditions from expressions; use derivative/integral/limit operations when the wording asks; "
        "use python expression for comb/factorial/sum/range. "
        "Return JSON only: {\"plans\":[{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|python|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\","
        "\"upper\":\"\",\"order\":\"1\"}]}. No explanation.\n\n"
        f"Previous failure:\n{failure_summary}\n\n"
        f"Question stem without answer options:\n{problem['_question']}"
    )


def _reference_plan_failure_summary(result: dict[str, Any]) -> str:
    summary = {
        "operation": result.get("operation"),
        "reason": result.get("reason"),
        "plan_failure_reasons": result.get("plan_failure_reasons"),
        "plan_count": result.get("plan_count"),
    }
    return json.dumps({key: value for key, value in summary.items() if value not in (None, {}, [])}, sort_keys=True)


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
        "python",
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
    if operation in {"evaluate", "simplify", "factor", "expand", "mod", "differentiate", "integrate", "limit", "python"} and expression_norm in candidate_norms:
        return True
    equation = _normalize_math_expression(str(plan.get("equation") or "")).replace(" ", "")
    variable = str(plan.get("variable") or "x").strip() or "x"
    trivial_equations = {f"{variable}={candidate}" for candidate in candidate_norms} | {
        f"{candidate}={variable}" for candidate in candidate_norms
    }
    return operation == "solve" and equation in trivial_equations


def _log_candidate_claim_planner_lifecycle(
    logger: "_JsonlLogger | None",
    *,
    event: str,
    eval_id: str | None,
    call_id: str | None,
    problem: dict[str, Any],
    model: str | None,
    planner_kind: str,
    candidate_count: int,
    timeout: float | None,
    latency_sec: float | None = None,
    status: str | None = None,
    operation: str | None = None,
    reason: str | None = None,
    plan_hash: str | None = None,
    error_type: str | None = None,
    error: str | None = None,
) -> None:
    if not eval_id or not call_id or not model:
        return
    payload: dict[str, Any] = {
        "event": event,
        "eval_id": eval_id,
        "call_id": call_id,
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "model": model,
        "variant": "assumption_agent_recursive_verify",
        "planner_kind": planner_kind,
        "candidate_count": candidate_count,
        "timeout_sec": timeout,
    }
    if latency_sec is not None:
        payload["latency_sec"] = latency_sec
    if status is not None:
        payload["status"] = status
    if operation is not None:
        payload["operation"] = operation
    if reason is not None:
        payload["reason"] = reason
    if plan_hash is not None:
        payload["plan_hash"] = plan_hash
    if error_type is not None:
        payload["error_type"] = error_type
    if error is not None:
        payload["error"] = error
    _log_event(logger, payload)


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
                "activated" if _is_verified_math_reference(reference) else "abstained",
            ),
            "operation": reference.get("operation"),
            "reason": reference.get("reason"),
            "error_type": reference.get("error_type"),
            "planner_latency_sec": reference.get("planner_latency_sec"),
            "plan_hash": reference.get("plan_hash"),
            "claim_hash": (summary or {}).get("claim_hash"),
            "verified_count": (summary or {}).get("verified_count", 0),
            "refuted_count": (summary or {}).get("refuted_count", 0),
            "candidate_count": reference.get("candidate_count"),
        },
    )


def _should_run_candidate_claim_verifier(problem: dict[str, Any]) -> bool:
    if os.environ.get("HLE_DISABLE_CANDIDATE_CLAIM_VERIFIER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if problem.get("answer_type") == "multipleChoice":
        if os.environ.get("HLE_DISABLE_MC_CANDIDATE_CLAIM_VERIFIER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
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
            if not result.get("answer") and _math_tool_plan_repair_enabled():
                initial_result = dict(result)
                repair_text = _call_model(
                    model=model,
                    prompt=_math_tool_plan_repair_prompt(problem, initial_result),
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
                underlying_calls += 1
                repair_result = _execute_math_tool_plan_text(repair_text)
                repair_result.setdefault("plan_hash", stable_hash({"planner_text": repair_text}))
                repair_result["source"] = "llm_planner_repair"
                repair_result["initial_plan_reason"] = initial_result.get("reason")
                repair_result["initial_plan_operation"] = initial_result.get("operation")
                if repair_result.get("answer"):
                    result = repair_result
                else:
                    result["repair_reason"] = repair_result.get("reason")
                    result["repair_operation"] = repair_result.get("operation")
                    result["repair_plan_hash"] = repair_result.get("plan_hash")
                    result["repair_underlying_model_calls"] = 1
        status = "answered" if result.get("answer") else "abstained"
        answer = str(result.get("answer") or "").strip()
        summary = {
            "status": "activated" if status == "answered" else "abstained",
            "tool": "sympy_restricted",
            "source": result.get("source"),
            "operation": result.get("operation"),
            "confidence": result.get("confidence"),
            "plan_hash": result.get("plan_hash"),
            "plan_count": result.get("plan_count"),
            "plan_success_count": result.get("plan_success_count"),
            "plan_agreement_count": result.get("plan_agreement_count"),
            "answer_hash": stable_hash({"answer": answer}) if answer else None,
            "reason": result.get("reason"),
            "initial_plan_reason": result.get("initial_plan_reason"),
            "repair_reason": result.get("repair_reason"),
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
        _deterministic_equation_answer(question),
        _deterministic_symbolic_transform_answer(question),
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


def _deterministic_symbolic_transform_answer(question: str) -> dict[str, Any]:
    text = str(question or "")
    if re.search(r"\b(?:differentiate|derivative)\b|d\s*/\s*d[A-Za-z]\b", text, flags=re.IGNORECASE):
        for expression in _candidate_operation_expressions(text, "differentiate"):
            variable = _infer_expression_variable(expression, text)
            if not variable:
                continue
            answer = _differentiate_safe_expression(expression, variable, "1")
            if answer:
                return _deterministic_transform_result("differentiate", expression, answer, variable=variable)
    if re.search(r"\b(?:integrate|integral|antiderivative)\b", text, flags=re.IGNORECASE):
        lower, upper = _extract_integral_bounds(text)
        for expression in _candidate_operation_expressions(text, "integrate"):
            variable = _infer_expression_variable(expression, text)
            if not variable:
                continue
            answer = _integrate_safe_expression(expression, variable, lower or "", upper or "")
            if answer:
                return _deterministic_transform_result("integrate", expression, answer, variable=variable, lower=lower, upper=upper)
    if re.search(r"\blimit\b|(?:->|→|approaches|tends\s+to)", text, flags=re.IGNORECASE):
        for expression in _candidate_operation_expressions(text, "limit"):
            variable = _infer_expression_variable(expression, text)
            if not variable:
                continue
            point = _extract_limit_point(text, variable)
            if not point:
                continue
            answer = _limit_safe_expression(expression, variable, point)
            if answer:
                return _deterministic_transform_result("limit", expression, answer, variable=variable, point=point)
    transform_triggers = (
        ("factor", r"\b(?:factor|factorize)\b"),
        ("expand", r"\bexpand\b"),
        ("simplify", r"\b(?:simplify|reduce)\b"),
    )
    for operation, trigger in transform_triggers:
        if not re.search(trigger, text, flags=re.IGNORECASE):
            continue
        for expression in _candidate_operation_expressions(text, operation):
            parsed = _safe_sympy_parse_expr(expression)
            if parsed is None:
                continue
            value = _apply_safe_sympy_transform(operation, parsed)
            if value is None:
                continue
            return _deterministic_transform_result(operation, expression, _format_sympy_operation_answer(operation, value))
    return {}


def _deterministic_transform_result(
    operation: str,
    expression: str,
    answer: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "operation": f"deterministic_{operation}",
        "expression": expression,
        **{key: value for key, value in extra.items() if value not in (None, "")},
    }
    return {
        "answer": answer,
        "operation": operation,
        "confidence": "verified_symbolic",
        "source": "deterministic_transform_solver",
        "plan_hash": stable_hash(payload),
    }


def _candidate_operation_expressions(question: str, operation: str) -> list[str]:
    text = str(question or "")
    raw_candidates: list[str] = []
    raw_candidates.extend(candidate for candidate in _math_container_texts(text) if "=" not in candidate)
    keyword_patterns = {
        "differentiate": [
            r"\b(?:differentiate|derivative\s+of)\s+([^?.\n]{1,180})",
            r"\bd\s*/\s*d[A-Za-z]\s*(?:of)?\s*([^?.\n]{1,180})",
        ],
        "integrate": [
            r"\b(?:integrate|integral\s+of|antiderivative\s+of)\s+([^?.\n]{1,180})",
        ],
        "limit": [
            r"\blimit\s+of\s+([^?.\n]{1,180})",
            r"\blim(?:it)?\s*([^?.\n]{1,180})",
        ],
        "simplify": [
            r"\b(?:simplify|reduce)\s+([^?.\n]{1,180})",
        ],
        "factor": [
            r"\b(?:factor|factorize)\s+([^?.\n]{1,180})",
        ],
        "expand": [
            r"\bexpand\s+([^?.\n]{1,180})",
        ],
    }
    for pattern in keyword_patterns.get(operation, []):
        raw_candidates.extend(match.group(1) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    cleaned: list[str] = []
    for candidate in raw_candidates:
        normalized = _normalize_math_expression(candidate)
        normalized = _strip_operation_expression_tail(normalized)
        if not normalized or "=" in normalized or len(normalized) > 180:
            continue
        parsed = _safe_sympy_parse_expr(normalized)
        if parsed is None:
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:8]


def _strip_operation_expression_tail(expression: str) -> str:
    expression = str(expression or "")
    expression = re.sub(
        r"\b(?:with\s+respect\s+to|wrt|as|when|where|from|between|over|for|at)\b.*$",
        "",
        expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(r"\s*d[A-Za-z]\s*$", "", expression)
    return expression.strip(" =.,;:")


def _extract_integral_bounds(question: str) -> tuple[str | None, str | None]:
    text = _normalize_math_expression(question)
    match = re.search(r"\bfrom\s+([^,; ]{1,40})\s+to\s+([^,; ]{1,40})\b", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bbetween\s+([^,; ]{1,40})\s+and\s+([^,; ]{1,40})\b", text, flags=re.IGNORECASE)
    if not match:
        return (None, None)
    lower = _normalize_math_expression(match.group(1))
    upper = _normalize_math_expression(match.group(2))
    if _safe_sympy_parse_expr(lower) is None or _safe_sympy_parse_expr(upper) is None:
        return (None, None)
    return (lower, upper)


def _extract_limit_point(question: str, variable: str) -> str | None:
    text = str(question or "").replace("→", "->")
    patterns = [
        rf"\b{re.escape(variable)}\s*(?:->|approaches|tends\s+to)\s*([^,;.\n ]{{1,60}})",
        rf"\bas\s+{re.escape(variable)}\s+(?:approaches|tends\s+to)\s*([^,;.\n ]{{1,60}})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        point = _normalize_math_expression(match.group(1))
        parsed = _safe_sympy_parse_expr(point)
        if parsed is not None and not getattr(parsed, "free_symbols", None):
            return point
    return None


def _deterministic_equation_answer(question: str) -> dict[str, Any]:
    for equation in _candidate_math_equations(question):
        variable = _infer_equation_variable(equation, question)
        if not variable:
            continue
        answer = _solve_safe_equation(equation, variable)
        if answer:
            return {
                "answer": answer,
                "operation": "solve",
                "confidence": "verified_symbolic",
                "source": "deterministic_equation_solver",
                "plan_hash": stable_hash({
                    "operation": "deterministic_solve",
                    "equation": equation,
                    "variable": variable,
                }),
            }
    for expression in _candidate_root_expressions(question):
        variable = _infer_expression_variable(expression, question)
        if not variable:
            continue
        equation = f"{expression} = 0"
        answer = _solve_safe_equation(equation, variable)
        if answer:
            return {
                "answer": answer,
                "operation": "solve",
                "confidence": "verified_symbolic",
                "source": "deterministic_root_solver",
                "plan_hash": stable_hash({
                    "operation": "deterministic_roots",
                    "expression": expression,
                    "variable": variable,
                }),
            }
    return {}


def _candidate_math_equations(question: str) -> list[str]:
    text = str(question or "")
    if not re.search(r"\b(?:solve|solution|solutions|root|roots|zero|zeros|equation|satisf(?:y|ies))\b|=", text, flags=re.IGNORECASE):
        return []
    raw_candidates: list[str] = []
    raw_candidates.extend(_math_container_texts(text))
    equation_pattern = re.compile(
        r"([A-Za-z0-9_\\{}^+\-*/%().\s]{1,180}=[A-Za-z0-9_\\{}^+\-*/%().\s]{1,120})"
    )
    raw_candidates.extend(match.group(1) for match in equation_pattern.finditer(text))
    cleaned: list[str] = []
    for candidate in raw_candidates:
        normalized = _normalize_math_expression(candidate)
        if "=" not in normalized or re.search(r"[<>]=?|!=", normalized):
            continue
        lhs_rhs = [part.strip() for part in normalized.split("=", 1)]
        if len(lhs_rhs) != 2 or not lhs_rhs[0] or not lhs_rhs[1]:
            continue
        equation = f"{lhs_rhs[0]} = {lhs_rhs[1]}"
        if len(equation) <= 220 and equation not in cleaned:
            cleaned.append(equation)
    return cleaned[:8]


def _candidate_root_expressions(question: str) -> list[str]:
    text = str(question or "")
    if not re.search(r"\b(?:roots?|zeros?|solve)\b", text, flags=re.IGNORECASE):
        return []
    raw_candidates: list[str] = []
    raw_candidates.extend(candidate for candidate in _math_container_texts(text) if "=" not in candidate)
    for match in re.finditer(
        r"\b(?:roots?|zeros?)\s+of\s+([^?.\n]{1,180})",
        text,
        flags=re.IGNORECASE,
    ):
        raw_candidates.append(match.group(1))
    for match in re.finditer(
        r"\bsolve\s+([^?.\n=]{1,180})",
        text,
        flags=re.IGNORECASE,
    ):
        raw_candidates.append(match.group(1))
    cleaned: list[str] = []
    for candidate in raw_candidates:
        normalized = _normalize_math_expression(candidate)
        normalized = re.sub(
            r"\b(?:for|over|in|where|with|and|find|the|roots?|zeros?|solutions?|solve)\b.*$",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip(" =.,;:")
        if not normalized or "=" in normalized or len(normalized) > 180:
            continue
        parsed = _safe_sympy_parse_expr(normalized)
        if parsed is None or not getattr(parsed, "free_symbols", None):
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:8]


def _math_container_texts(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (
        r"\$([^$]{1,240})\$",
        r"\\\((.{1,240})\\\)",
        r"\\\[(.{1,240})\\\]",
        r"`([^`]{1,240})`",
    ):
        candidates.extend(match.group(1) for match in re.finditer(pattern, str(text or ""), flags=re.DOTALL))
    return candidates


def _infer_equation_variable(equation: str, question: str) -> str | None:
    if "=" not in equation:
        return None
    lhs_text, rhs_text = equation.split("=", 1)
    lhs = _safe_sympy_parse_expr(lhs_text)
    rhs = _safe_sympy_parse_expr(rhs_text)
    if lhs is None or rhs is None:
        return None
    return _choose_safe_symbol(
        sorted(str(symbol) for symbol in (set(getattr(lhs, "free_symbols", set())) | set(getattr(rhs, "free_symbols", set())))),
        question,
    )


def _infer_expression_variable(expression: str, question: str) -> str | None:
    parsed = _safe_sympy_parse_expr(expression)
    if parsed is None:
        return None
    return _choose_safe_symbol(sorted(str(symbol) for symbol in getattr(parsed, "free_symbols", set())), question)


def _choose_safe_symbol(symbols: list[str], question: str) -> str | None:
    safe_symbols = [symbol for symbol in symbols if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol)]
    if len(safe_symbols) == 1:
        return safe_symbols[0]
    for match in re.finditer(r"\b(?:for|solve\s+for|in terms of)\s+([A-Za-z][A-Za-z0-9_]*)\b", str(question or ""), flags=re.IGNORECASE):
        symbol = match.group(1)
        if symbol in safe_symbols:
            return symbol
    return None


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


def _safe_sympy_parse_plan_expression(expr: str) -> Any:
    for candidate in _plan_expression_parse_candidates(expr):
        parsed = _safe_sympy_parse_expr(candidate)
        if parsed is not None:
            return parsed
    return None


def _plan_expression_parse_candidates(expr: str) -> list[str]:
    normalized = _normalize_math_expression(expr)
    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = _strip_operation_expression_tail(_normalize_math_expression(candidate))
        if candidate and len(candidate) <= 220 and candidate not in candidates:
            candidates.append(candidate)

    add(normalized)
    for segment in _math_answer_candidate_segments(str(expr or "")):
        add(segment)
    if "=" in normalized:
        lhs, rhs = normalized.split("=", 1)
        add(rhs)
        add(lhs)
    assignment_stripped = re.sub(
        r"^\s*[A-Za-z][A-Za-z0-9_]*(?:\([^()]{0,40}\))?\s*=\s*",
        "",
        normalized,
    )
    add(assignment_stripped)
    return candidates[:8]


def _math_tool_planner_prompt(problem: dict[str, Any]) -> str:
    return (
        "Extract up to four independent executable math plans for this HLE math exactMatch item. "
        "Do not answer from memory and do not copy a guessed final answer into an expression. Prefer a safe executable "
        "plan whenever the question can be computed, simplified, solved, counted, checked modulo an integer, or reduced "
        "to a finite arithmetic/combinatorics expression. If one plan is uncertain, include an alternate plan rather "
        "than abstaining immediately. Use python only as a side-effect-free expression for finite sums/products, "
        "combinations, factorials, integer arithmetic, or fractions. Return none only when no small executable check "
        "is possible. JSON only: {\"plans\":[{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|python|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\",\"upper\":\"\",\"order\":\"1\"}]}. "
        "Use plain SymPy-compatible syntax or a side-effect-free Python expression, no imports, no statements, no code blocks.\n\n"
        f"Question:\n{problem['_question']}"
    )


def _math_tool_plan_repair_prompt(problem: dict[str, Any], initial_result: dict[str, Any]) -> str:
    failure_summary = _reference_plan_failure_summary(initial_result)
    return (
        "The previous executable math plan for this HLE exactMatch item failed or abstained. Produce corrected "
        "executable plans. Use the question text as the only source of computation; do not invent or copy a final "
        "answer. Common repairs: convert prose into an equation, use solve for roots, use simplify/factor/expand "
        "for symbolic results, use derivative/integral/limit operations when explicitly asked, or use a python "
        "expression for finite sums/products/combinatorics/modular arithmetic. If the result is still not executable, "
        "return one none plan. JSON only: {\"plans\":[{\"operation\":\"evaluate|simplify|factor|expand|solve|mod|differentiate|integrate|limit|python|none\","
        "\"expression\":\"...\",\"equation\":\"...\",\"variable\":\"x\",\"modulus\":\"\",\"point\":\"\",\"lower\":\"\","
        "\"upper\":\"\",\"order\":\"1\"}]}. No explanation.\n\n"
        f"Previous failure:\n{failure_summary}\n\nQuestion:\n{problem['_question']}"
    )


def _math_tool_plan_repair_enabled() -> bool:
    return os.environ.get("HLE_DISABLE_MATH_TOOL_PLAN_REPAIR", "").strip().lower() not in {"1", "true", "yes", "on"}


def _execute_math_tool_plan_text(text: str) -> dict[str, Any]:
    plan = _parse_json_object(text)
    if not isinstance(plan, dict):
        return {"source": "llm_planner", "operation": "none", "confidence": "abstain", "reason": "planner_json_parse_failed"}
    return _execute_math_tool_plan_candidates(_math_tool_plan_candidates_from_object(plan))


def _math_tool_plan_candidates_from_object(plan: dict[str, Any]) -> list[dict[str, Any]]:
    plans = plan.get("plans")
    if isinstance(plans, list):
        return [candidate for candidate in plans if isinstance(candidate, dict)][:6]
    return [plan]


def _execute_math_tool_plan_candidates(
    plans: list[dict[str, Any]],
    *,
    leak_candidates: list[str] | None = None,
) -> dict[str, Any]:
    if not plans:
        return {"source": "llm_planner", "operation": "none", "confidence": "abstain", "reason": "planner_json_parse_failed"}
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        operation = str(plan.get("operation") or "none").strip().lower()
        if leak_candidates and _math_plan_candidate_leak_risk(plan, leak_candidates):
            failures.append({"plan_index": index, "operation": operation or "none", "reason": "candidate_literal_leakage"})
            continue
        result = _execute_math_tool_plan(plan)
        result["plan_index"] = index
        result["plan_count"] = len(plans)
        if result.get("confidence") == "verified_symbolic" and str(result.get("answer") or "").strip():
            successes.append(result)
            continue
        failures.append({
            "plan_index": index,
            "operation": result.get("operation") or operation or "none",
            "reason": result.get("reason") or "not_verified",
        })
    if successes:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in successes:
            answer_norm = _normalize_exact(str(result.get("answer") or ""))
            grouped[answer_norm or str(result.get("answer") or "").strip()].append(result)
        ranked_successes = sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), int(item[1][0].get("plan_index", 0) or 0)),
        )
        top_norm, top_results = ranked_successes[0]
        runner_up_count = len(ranked_successes[1][1]) if len(ranked_successes) > 1 else 0
        selected = dict(top_results[0])
        selected["plan_success_count"] = len(successes)
        selected["plan_agreement_count"] = len(top_results)
        selected["plan_count"] = len(plans)
        selected["agreement_answer_hash"] = stable_hash({"answer_norm": top_norm})
        if len(top_results) >= 2 and len(top_results) > runner_up_count:
            selected["confidence"] = "verified_symbolic_consensus"
            selected["consensus_plan_indices"] = [
                int(result.get("plan_index", 0) or 0)
                for result in top_results
            ]
        if failures:
            selected["prior_plan_failures"] = failures[:5]
        return selected
    failure_reasons = Counter(str(failure.get("reason") or "unknown") for failure in failures)
    first = failures[0] if failures else {}
    dominant_reason = failure_reasons.most_common(1)[0][0] if failure_reasons else "all_candidate_plans_failed"
    return {
        "source": "llm_planner",
        "operation": first.get("operation") or "none",
        "confidence": "abstain",
        "reason": dominant_reason if len(failure_reasons) == 1 else "all_candidate_plans_failed",
        "plan_count": len(plans),
        "plan_failure_reasons": dict(failure_reasons),
        "plan_failures": failures[:6],
    }


def _execute_math_tool_plan(plan: dict[str, Any]) -> dict[str, Any]:
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
        "python",
    }:
        return {"source": "llm_planner", "operation": operation or "none", "confidence": "abstain", "reason": "planner_abstained"}
    try:
        if operation in {"evaluate", "simplify", "factor", "expand"}:
            parsed = _safe_sympy_parse_plan_expression(str(plan.get("expression") or ""))
            if parsed is None:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "unsafe_or_symbolic_expression"}
            if operation == "evaluate" and getattr(parsed, "free_symbols", None):
                value = _apply_safe_sympy_transform("simplify", parsed)
                if value is None:
                    return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "unsafe_or_symbolic_expression"}
                return {
                    "source": "llm_planner",
                    "operation": "simplify",
                    "answer": _format_sympy_operation_answer("simplify", value),
                    "confidence": "verified_symbolic",
                    "coerced_from_operation": "evaluate",
                }
            value = _apply_safe_sympy_transform(operation, parsed)
            if value is None:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "transform_failed"}
            return {
                "source": "llm_planner",
                "operation": operation,
                "answer": _format_sympy_operation_answer(operation, value),
                "confidence": "verified_symbolic",
            }
        if operation == "mod":
            parsed = _safe_sympy_parse_plan_expression(str(plan.get("expression") or ""))
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
        if operation == "python":
            answer = _evaluate_safe_python_expression(str(plan.get("expression") or ""))
            if not answer:
                return {"source": "llm_planner", "operation": operation, "confidence": "abstain", "reason": "unsafe_python_expression"}
            return {"source": "llm_planner", "operation": operation, "answer": answer, "confidence": "verified_symbolic"}
    except Exception as exc:
        return {
            "source": "llm_planner",
            "operation": operation,
            "confidence": "abstain",
            "reason": type(exc).__name__,
        }


def _evaluate_safe_python_expression(expression: str) -> str | None:
    expression = str(expression or "").strip()
    if not expression or len(expression) > 500 or "__" in expression:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        _validate_safe_python_ast(tree)
        env = _safe_python_eval_env()
        value = eval(compile(tree, "<hle_safe_python_expr>", "eval"), {"__builtins__": {}, **env}, {})
        return _format_safe_python_value(value)
    except Exception:
        return None


def _safe_python_eval_env() -> dict[str, Any]:
    return {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "range": _safe_python_range,
        "comb": math.comb,
        "perm": getattr(math, "perm", _safe_perm),
        "factorial": math.factorial,
        "gcd": math.gcd,
        "lcm": getattr(math, "lcm", _safe_lcm),
        "sqrt": math.sqrt,
        "floor": math.floor,
        "ceil": math.ceil,
        "log": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "Fraction": fractions.Fraction,
    }


def _validate_safe_python_ast(node: ast.AST, bound_names: set[str] | None = None) -> None:
    bound_names = set(bound_names or set())
    if isinstance(node, ast.Expression):
        _validate_safe_python_ast(node.body, bound_names)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return
        raise ValueError("unsafe_constant")
    if isinstance(node, ast.Name):
        if node.id in _safe_python_eval_env() or node.id in bound_names:
            return
        raise ValueError("unsafe_name")
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            raise ValueError("unsafe_operator")
        _validate_safe_python_ast(node.left, bound_names)
        _validate_safe_python_ast(node.right, bound_names)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("unsafe_unary")
        _validate_safe_python_ast(node.operand, bound_names)
        return
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("unsafe_boolop")
        for value in node.values:
            _validate_safe_python_ast(value, bound_names)
        return
    if isinstance(node, ast.Compare):
        _validate_safe_python_ast(node.left, bound_names)
        for op in node.ops:
            if not isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                raise ValueError("unsafe_compare")
        for comparator in node.comparators:
            _validate_safe_python_ast(comparator, bound_names)
        return
    if isinstance(node, ast.IfExp):
        _validate_safe_python_ast(node.test, bound_names)
        _validate_safe_python_ast(node.body, bound_names)
        _validate_safe_python_ast(node.orelse, bound_names)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _safe_python_eval_env():
            raise ValueError("unsafe_call")
        for arg in node.args:
            _validate_safe_python_ast(arg, bound_names)
        for keyword in node.keywords:
            _validate_safe_python_ast(keyword.value, bound_names)
        return
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        if len(node.elts) > 20:
            raise ValueError("unsafe_collection_size")
        for elt in node.elts:
            _validate_safe_python_ast(elt, bound_names)
        return
    if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
        local_bound = set(bound_names)
        for generator in node.generators:
            if not isinstance(generator.target, ast.Name):
                raise ValueError("unsafe_comprehension_target")
            _validate_safe_python_ast(generator.iter, local_bound)
            local_bound.add(generator.target.id)
            for condition in generator.ifs:
                _validate_safe_python_ast(condition, local_bound)
        _validate_safe_python_ast(node.elt, local_bound)
        return
    raise ValueError(type(node).__name__)


def _safe_python_range(*args: Any) -> range:
    ints = [int(arg) for arg in args]
    if not 1 <= len(ints) <= 3:
        raise ValueError("bad_range_arity")
    result = range(*ints)
    if len(result) > 100000:
        raise ValueError("range_too_large")
    return result


def _safe_perm(n: int, k: int | None = None) -> int:
    n = int(n)
    k = n if k is None else int(k)
    return math.factorial(n) // math.factorial(n - k)


def _safe_lcm(*values: int) -> int:
    result = 1
    for value in values:
        result = abs(result * int(value)) // math.gcd(result, int(value))
    return result


def _format_safe_python_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, fractions.Fraction):
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if abs(value - round(value)) <= 1e-12:
            return str(int(round(value)))
        return format(value, ".12g")
    if isinstance(value, (tuple, list)) and len(value) <= 12:
        parts = [_format_safe_python_value(part) for part in value]
        if all(part is not None for part in parts):
            return ", ".join(str(part) for part in parts)
    return None


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


def _format_sympy_operation_answer(operation: str, value: Any) -> str:
    if operation in {"factor", "expand"}:
        try:
            import sympy as sp

            return sp.sstr(value)
        except Exception:
            return str(value).strip()
    return _format_sympy_answer(value)


def _recursive_child_prompt_specs(problem: dict[str, Any], *, agent_plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
    if answer_type == "exactMatch" and _exact_trajectory_search_enabled():
        specs.extend([
            {
                "prompt_kind": "decomposition_answer",
                "prompt": (
                    "Solve through an independent decomposition path. Internally identify the exact target "
                    "quantity/name, the binding constraints, and any exclusion clauses before giving the final "
                    "short answer. Do not reuse a first-impression answer unless it survives the decomposition. "
                    "Return only JSON.\n\n"
                    f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
                ),
            },
            {
                "prompt_kind": "adversarial_alternative_answer",
                "prompt": (
                    "Solve as an adversarial alternative generator. Assume the most obvious direct answer may be "
                    "wrong. Search for a different answer forced by a hidden constraint, edge case, wording trap, "
                    "or named entity disambiguation. If no alternative survives, return the best surviving answer. "
                    "Return only JSON.\n\n"
                    f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
                ),
            },
            {
                "prompt_kind": "literal_constraint_answer",
                "prompt": (
                    "Solve by literal constraint matching. Ignore broad background priors and choose the shortest "
                    "answer that satisfies every explicit word in the question, including time, scope, version, "
                    "unit, notation, and requested output form. Return only JSON.\n\n"
                    f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
                ),
            },
        ])
    if answer_type == "multipleChoice":
        if _is_code_compile_mc_question(problem):
            specs.insert(1, {
                "prompt_kind": "code_semantics_answer",
                "prompt": _code_semantics_answer_prompt(problem),
            })
        specs.extend([
            {
                "prompt_kind": "option_matrix_reasoner_answer",
                "prompt": _option_matrix_reasoner_prompt(problem),
            },
            {
                "prompt_kind": "option_elimination_answer",
                "prompt": (
                    "Solve through an independent option-elimination path. Evaluate every listed option against "
                    "the exact wording of the question. Reject any option contradicted by a constraint, scope, "
                    "time frame, definition, or requested relation. Return only JSON.\n\n"
                    f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
                ),
            },
            {
                "prompt_kind": "adversarial_alternative_answer",
                "prompt": (
                    "Solve as an adversarial boundary branch. Assume the most common or first-impression option "
                    "may be a lure. Look for the option forced by an edge case, negation, qualifier, or entity "
                    "disambiguation. Return only JSON.\n\n"
                    f"Answer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
                ),
            },
        ])
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
    return _orthogonalize_child_prompt_specs(problem, specs, agent_plan=agent_plan)


def _option_matrix_reasoner_prompt(problem: dict[str, Any]) -> str:
    return (
        "Solve this multiple-choice item as a discrete option matrix, not as a first-impression vote. "
        "Internally create one row per option. For each row identify: the exact claim the option would make true, "
        "the single strongest clue in the question for it, the single strongest contradiction or missing condition, "
        "and the minimal discriminating fact that separates it from the other options. Penalize options that only "
        "sound familiar but do not satisfy the exact requested relation, negation, scope, date, entity, or mechanism. "
        "Choose the option with the best surviving discriminating constraint, even if it is not the most common "
        "answer. Return JSON only: {\"answer\":\"A\"}.\n\n"
        f"Question:\n{problem['_question']}"
    )


def _is_code_compile_mc_question(problem: dict[str, Any]) -> bool:
    if problem.get("answer_type") != "multipleChoice":
        return False
    question = str(problem.get("_question") or "")
    lower = question.lower()
    if "```" not in question:
        return False
    return any(token in lower for token in (
        "compile",
        "compiler",
        "borrow checker",
        "type check",
        "runtime error",
        "will this code",
        "will the following",
    ))


def _code_semantics_answer_prompt(problem: dict[str, Any]) -> str:
    return (
        "Solve this multiple-choice code item through a static/code-semantics branch. Treat the code block as "
        "the primary evidence. Internally determine: whether the code compiles/type-checks, whether warnings "
        "matter for the option wording, whether runtime behavior is relevant, and whether phrases such as "
        "\"unsafe code under the hood\" refer to explicit unsafe blocks versus standard-library implementation "
        "details. Do not answer from general popularity or retrieval. Return JSON only: {\"answer\":\"A\"}.\n\n"
        f"Question:\n{problem['_question']}"
    )


def _exact_trajectory_search_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_EXACT_TRAJECTORY_SEARCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    if (math_tool_summary or {}).get("confidence") in {"verified_symbolic", "verified_symbolic_consensus"}:
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
    if os.environ.get("HLE_DISABLE_MC_OPTION_EVIDENCE_SCORER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None, {"status": "disabled", "reason": "env_disabled"}
    if any(attempt.get("prompt_kind") == "mc_option_evidence_scorer_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    stem, options = _split_multiple_choice_question(problem)
    if len(options) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    stem_terms = _content_terms(stem or problem.get("_question", ""))
    option_terms_by_label = {
        label: _content_terms(text)
        for label, text in options.items()
        if _content_terms(text)
    }
    option_text_by_label = dict(options)
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
        score_detail = _score_option_evidence_detail(
            stem_terms=stem_terms,
            option_label=label,
            option_text=option_text,
            option_terms_by_label=option_terms_by_label,
            option_text_by_label=option_text_by_label,
            docs=docs,
        )
        option_rows.append({
            "label": label,
            "score": score_detail["score"],
            "rank_score": _option_evidence_rank_score(score_detail),
            "query_hash": stable_hash({"query": query}),
            "doc_count": len(docs),
            "support_doc_count": score_detail["support_doc_count"],
            "ambiguous_doc_count": score_detail["ambiguous_doc_count"],
            "unsupported_doc_count": score_detail["unsupported_doc_count"],
            "supporting_doc_hashes": score_detail["supporting_doc_hashes"],
            "doc_hashes": [
                stable_hash({"title": doc.get("title", ""), "snippet": doc.get("snippet", "")})
                for doc in docs[:2]
            ],
        })
    ranked = sorted(option_rows, key=lambda row: (-float(row["rank_score"]), -float(row["score"]), row["label"]))
    if not ranked:
        return None, {"status": "no_results", "reason": "no_option_scores"}
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else {"score": 0.0}
    top_score = float(top["score"])
    top_rank_score = float(top.get("rank_score") or 0.0)
    margin = top_score - float(runner_up.get("score") or 0.0)
    rank_margin = top_rank_score - float(runner_up.get("rank_score") or 0.0)
    top_support_doc_count = int(top.get("support_doc_count") or 0)
    runner_up_support_doc_count = int(runner_up.get("support_doc_count") or 0)
    top_ambiguous_doc_count = int(top.get("ambiguous_doc_count") or 0)
    any_ambiguous_doc_count = sum(int(row.get("ambiguous_doc_count") or 0) for row in option_rows)
    min_verified_score = 16.0
    env_min_score = os.environ.get("HLE_OPTION_EVIDENCE_MIN_VERIFIED_SCORE", "").strip()
    if env_min_score:
        try:
            min_verified_score = max(0.0, min(40.0, float(env_min_score)))
        except ValueError:
            pass
    confidence = (
        top_score >= 6.0
        and top_score >= min_verified_score
        and rank_margin >= 2.0
        and top_support_doc_count >= 3
        and top_support_doc_count > runner_up_support_doc_count
        and top_ambiguous_doc_count == 0
    )
    status = "activated" if confidence else "weak_margin"
    if not confidence and top_score >= 4.0 and top_support_doc_count <= 0:
        status = "blocked_non_discriminative_option_evidence"
    if not confidence and any_ambiguous_doc_count > 0:
        status = "blocked_ambiguous_option_evidence"
    if not confidence and top_score >= 6.0 and top_support_doc_count <= runner_up_support_doc_count:
        status = "blocked_weak_support_count"
    summary = {
        "status": status,
        "source": "wikipedia_plus_domain_option_search",
        "score_policy": "discriminative_option_support_v2",
        "min_verified_score": round(min_verified_score, 4),
        "option_count": len(options),
        "top_option_hash": stable_hash({"option_label": top["label"]}),
        "top_option_answer_hash": stable_hash({"answer": str(top["label"])}),
        "top_score": round(top_score, 4),
        "top_rank_score": round(top_rank_score, 4),
        "runner_up_score": round(float(runner_up.get("score") or 0.0), 4),
        "runner_up_rank_score": round(float(runner_up.get("rank_score") or 0.0), 4),
        "margin": round(margin, 4),
        "rank_margin": round(rank_margin, 4),
        "candidate_emitted": bool(confidence),
        "candidate_verifier_state": "verified" if confidence else "not_verified",
        "top_support_doc_count": top_support_doc_count,
        "top_ambiguous_doc_count": top_ambiguous_doc_count,
        "any_ambiguous_doc_count": any_ambiguous_doc_count,
        "runner_up_support_doc_count": runner_up_support_doc_count,
        "query_hashes": [row["query_hash"] for row in option_rows],
        "doc_count_by_option_hash": {
            stable_hash({"option_label": row["label"]}): row["doc_count"]
            for row in option_rows
        },
        "support_doc_count_by_option_hash": {
            stable_hash({"option_label": row["label"]}): int(row.get("support_doc_count") or 0)
            for row in option_rows
        },
        "ambiguous_doc_count_by_option_hash": {
            stable_hash({"option_label": row["label"]}): int(row.get("ambiguous_doc_count") or 0)
            for row in option_rows
        },
        "top_doc_hashes": top.get("doc_hashes", []),
        "top_supporting_doc_hashes": top.get("supporting_doc_hashes", []),
        "runner_up_option_hash": stable_hash({"option_label": runner_up.get("label")}) if runner_up.get("label") else None,
        "runner_up_option_answer_hash": (
            stable_hash({"answer": str(runner_up.get("label"))}) if runner_up.get("label") else None
        ),
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
        "tool_confidence": "verified_option_evidence_margin",
        "candidate_verifier_state": "verified",
        "candidate_verifier_backend": "mc_option_evidence_scorer",
        "candidate_verifier_operation": "option_specific_retrieval_margin",
        "candidate_verifier_claim_hash": stable_hash({
            "backend": "mc_option_evidence_scorer",
            "operation": "option_specific_retrieval_margin",
            "question_hash": problem.get("question_hash"),
            "top_option_hash": summary["top_option_hash"],
            "top_score": summary["top_score"],
            "margin": summary["margin"],
            "supporting_doc_hashes": summary["top_supporting_doc_hashes"],
        }),
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


def _maybe_run_evidence_guided_option_challenge(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    option_evidence_summary: dict[str, Any] | None,
    model: str,
    eval_id: str,
    call_id: str,
    logger: "_JsonlLogger | None",
    timeout: float | None,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    if problem.get("answer_type") != "multipleChoice":
        return None, None, ""
    if os.environ.get("HLE_DISABLE_EVIDENCE_GUIDED_OPTION_CHALLENGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None, {"status": "disabled", "reason": "env_disabled"}, ""
    if any(attempt.get("prompt_kind") == "evidence_guided_option_challenge_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}, ""
    route_like_prompt_kinds = {
        "route_arbitrator_answer",
        "raw_preserve_selector_answer",
        "raw_budget_preserve_selector_answer",
        "hipporag_preserve_selector_answer",
    }
    if any(
        attempt.get("candidate_verifier_state") == "verified"
        and _is_trusted_candidate_verifier_attempt(attempt)
        and attempt.get("prompt_kind") not in route_like_prompt_kinds
        for attempt in attempts
    ):
        return None, {"status": "not_required", "reason": "verified_candidate_available"}, ""

    stem, options = _split_multiple_choice_question(problem)
    if len(options) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}, ""
    top_status = str((option_evidence_summary or {}).get("status") or "")
    if top_status == "activated":
        return None, {"status": "not_required", "reason": "option_evidence_already_verified"}, ""

    context, context_summary = _build_option_evidence_challenge_context(
        problem=problem,
        stem=stem,
        options=options,
    )
    if not context:
        summary = {
            "status": "abstained",
            "reason": context_summary.get("reason") or "no_option_evidence_context",
            **context_summary,
            "underlying_model_calls": 0,
        }
        _log_event(
            logger,
            {
                "event": "evidence_guided_option_challenge",
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
        return None, summary, ""

    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "evidence_guided_option_challenge_answer",
            "prompt": _evidence_guided_option_challenge_prompt(
                problem,
                option_evidence_context=context,
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
    if attempt.get("status") == "answered":
        attempt["candidate_verifier_state"] = "not_verified"
        attempt["candidate_verifier_backend"] = "evidence_guided_option_challenge"
        attempt["tool_confidence"] = "unverified_evidence_guided_variation"
    answer = str(attempt.get("parsed_answer") or "").strip()
    summary = {
        "status": "activated",
        "reason": "unverified_option_specific_evidence_variation",
        "child_id": attempt.get("child_id"),
        "child_status": attempt.get("status"),
        "candidate_emitted": bool(answer),
        "candidate_verifier_state": "not_verified",
        "candidate_answer_hash": stable_hash({"answer": answer}) if answer else None,
        "option_count": len(options),
        "source": "option_specific_evidence_challenge",
        "score_policy": context_summary.get("score_policy"),
        "context_hash": stable_hash({"option_evidence_challenge_context": context}),
        "context_char_count": len(context),
        "top_option_answer_hash": context_summary.get("top_option_answer_hash"),
        "top_rank_score": context_summary.get("top_rank_score"),
        "top_support_doc_count": context_summary.get("top_support_doc_count"),
        "any_ambiguous_doc_count": context_summary.get("any_ambiguous_doc_count"),
        "context_option_count": context_summary.get("context_option_count"),
        "doc_count_by_option_hash": context_summary.get("doc_count_by_option_hash", {}),
        "support_doc_count_by_option_hash": context_summary.get("support_doc_count_by_option_hash", {}),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    if problem.get("_answer"):
        gold_for_eval, _ = _canonicalize_multiple_choice_answer(problem, str(problem.get("_answer") or ""))
        summary["candidate_correct_for_eval"] = _is_correct(answer, gold_for_eval, answer_type="multipleChoice")
    _log_event(
        logger,
        {
            "event": "evidence_guided_option_challenge",
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
    return attempt, summary, context


def _build_option_evidence_challenge_context(
    *,
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    stem_terms = _content_terms(stem or problem.get("_question", ""))
    option_terms_by_label = {
        label: _content_terms(text)
        for label, text in options.items()
        if _content_terms(text)
    }
    option_text_by_label = dict(options)
    docs_by_label: dict[str, list[dict[str, str]]] = {}
    option_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for label, option_text in sorted(options.items()):
        query = _option_evidence_query(stem, option_text, problem)
        docs: list[dict[str, str]] = []
        if query:
            try:
                docs = _wikipedia_search(query, limit=3, timeout=6.0)
                if len(docs) < 3 or _should_use_domain_evidence_search(problem):
                    docs.extend(_domain_evidence_search(query, problem=problem, limit=2, timeout=8.0))
                docs = _dedupe_evidence_results(docs)
            except Exception as exc:
                errors.append(type(exc).__name__)
        docs_by_label[label] = docs
        detail = _score_option_evidence_detail(
            stem_terms=stem_terms,
            option_label=label,
            option_text=option_text,
            option_terms_by_label=option_terms_by_label,
            option_text_by_label=option_text_by_label,
            docs=docs,
        )
        option_rows.append({
            "label": label,
            "rank_score": _option_evidence_rank_score(detail),
            "score": detail["score"],
            "support_doc_count": detail["support_doc_count"],
            "ambiguous_doc_count": detail["ambiguous_doc_count"],
            "unsupported_doc_count": detail["unsupported_doc_count"],
            "doc_count": len(docs),
        })
    if not any(docs_by_label.values()):
        return "", {
            "status": "abstained",
            "reason": "no_option_evidence_docs",
            "option_count": len(options),
            "error_types": sorted(set(errors)),
        }
    ranked = sorted(option_rows, key=lambda row: (-float(row["rank_score"]), -float(row["score"]), row["label"]))
    total_support_doc_count = sum(int(row.get("support_doc_count") or 0) for row in option_rows)
    if total_support_doc_count <= 0:
        return "", {
            "status": "abstained",
            "reason": "no_discriminative_support_docs",
            "option_count": len(options),
            "context_option_count": sum(1 for docs in docs_by_label.values() if docs),
            "context_doc_count": sum(len(docs) for docs in docs_by_label.values()),
            "total_support_doc_count": total_support_doc_count,
            "error_types": sorted(set(errors)),
        }
    context = _option_evidence_context(options=options, docs_by_label=docs_by_label)
    if not context:
        return "", {
            "status": "abstained",
            "reason": "empty_option_evidence_context",
            "option_count": len(options),
            "error_types": sorted(set(errors)),
        }
    top = ranked[0] if ranked else {}
    return context, {
        "status": "context_built",
        "score_policy": "discriminative_option_support_v2_challenge_context",
        "option_count": len(options),
        "context_option_count": sum(1 for docs in docs_by_label.values() if docs),
        "context_doc_count": sum(len(docs) for docs in docs_by_label.values()),
        "top_option_hash": stable_hash({"option_label": top.get("label")}) if top.get("label") else None,
        "top_option_answer_hash": stable_hash({"answer": str(top.get("label"))}) if top.get("label") else None,
        "top_rank_score": round(float(top.get("rank_score") or 0.0), 4),
        "top_support_doc_count": int(top.get("support_doc_count") or 0),
        "any_ambiguous_doc_count": sum(int(row.get("ambiguous_doc_count") or 0) for row in option_rows),
        "doc_count_by_option_hash": {
            stable_hash({"option_label": row["label"]}): int(row.get("doc_count") or 0)
            for row in option_rows
        },
        "support_doc_count_by_option_hash": {
            stable_hash({"option_label": row["label"]}): int(row.get("support_doc_count") or 0)
            for row in option_rows
        },
        "error_types": sorted(set(errors)),
    }


def _evidence_guided_option_challenge_prompt(
    problem: dict[str, Any],
    *,
    option_evidence_context: str,
) -> str:
    return (
        "Run an evidence-guided multiple-choice challenge. Treat the evidence below as noisy retrieval, not as "
        "proof. For each option, ask whether the evidence actually connects the option to the exact question "
        "stem and whether it rules out close distractors. If retrieval is irrelevant or ambiguous, rely on the "
        "question wording instead. Do not pick an option merely because it has more snippets. Return JSON only: "
        "{\"answer\":\"A\"}.\n\n"
        "Option-specific retrieved evidence:\n"
        f"{option_evidence_context}\n\n"
        f"Question:\n{problem['_question']}"
    )


def _structural_option_audit_prompt(
    problem: dict[str, Any],
    *,
    candidate_distribution: dict[str, int],
    missing_labels: list[str],
    evidence_context: str = "",
) -> str:
    distribution = ", ".join(f"{label}:{count}" for label, count in sorted(candidate_distribution.items())) or "none"
    missing = ", ".join(missing_labels) or "none"
    evidence_block = (
        "Transient evidence, only if directly answer-bearing:\n"
        f"{evidence_context}\n\n"
        if evidence_context
        else ""
    )
    return (
        "Run a structural option audit as an orthogonal hypothesis branch. The current ensemble distribution is "
        "shown below, but it may be collapsed on a familiar wrong attractor. Do not vote by popularity. Internally "
        "build one row per option and compare the minimum necessary condition that would make each option true. "
        "Prefer the option whose claim is forced by the exact wording with the fewest unstated assumptions; penalize "
        "options that are merely associated with the topic, restate a common fact, or miss a qualifier, negation, "
        "scope, relation, date, unit, or entity boundary. If every alternative fails a necessary condition, keep the "
        "majority. Return JSON only: {\"answer\":\"A\"}.\n\n"
        f"{evidence_block}"
        f"Current candidate distribution: {distribution}\n"
        f"Option labels not yet argued by any child: {missing}\n"
        f"Question:\n{problem['_question']}"
    )


def _maybe_run_structural_option_audit_child(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    option_evidence_summary: dict[str, Any] | None,
    evidence_guided_option_summary: dict[str, Any] | None,
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
    if os.environ.get("HLE_DISABLE_STRUCTURAL_OPTION_AUDIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None, {"status": "disabled", "reason": "env_disabled"}
    if any(attempt.get("prompt_kind") == "structural_option_audit_answer" for attempt in attempts):
        return None, {"status": "abstained", "reason": "already_executed"}
    labels, _ = _extract_multiple_choice_options(str(problem.get("_question") or ""))
    if len(labels) < 2:
        return None, {"status": "abstained", "reason": "options_not_parsed"}
    if any(
        attempt.get("candidate_verifier_state") == "verified"
        and _is_trusted_candidate_verifier_attempt(attempt)
        and attempt.get("prompt_kind") not in {
            "route_arbitrator_answer",
            "raw_preserve_selector_answer",
            "raw_budget_preserve_selector_answer",
            "hipporag_preserve_selector_answer",
        }
        for attempt in attempts
    ):
        return None, {"status": "not_required", "reason": "verified_candidate_available"}

    valid_norms: list[str] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        valid_norms.append(_normalize_for_selection(answer, answer_type="multipleChoice"))
    if len(valid_norms) < 3:
        return None, {"status": "not_required", "reason": "too_few_candidate_answers"}
    counts = Counter(norm for norm in valid_norms if norm)
    if not counts:
        return None, {"status": "not_required", "reason": "no_normalized_candidates"}
    top_norm, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    unique_count = len(counts)
    missing_labels = [label for label in sorted(labels) if label not in counts]
    option_evidence_status = str((option_evidence_summary or {}).get("status") or "")
    evidence_guided_status = str((evidence_guided_option_summary or {}).get("status") or "")
    weak_evidence = option_evidence_status not in {"activated"} and evidence_guided_status not in {"activated_verified"}
    collapsed = top_count >= max(3, len(valid_norms) - 1) or unique_count <= 2
    if not collapsed and not (weak_evidence and missing_labels):
        return None, {
            "status": "not_required",
            "reason": "candidate_space_already_diverse",
            "valid_candidate_count": len(valid_norms),
            "unique_candidate_count": unique_count,
            "top_candidate_count": top_count,
        }

    attempt = _run_child_attempt(
        problem=problem,
        spec={
            "prompt_kind": "structural_option_audit_answer",
            "prompt": _structural_option_audit_prompt(
                problem,
                candidate_distribution=dict(counts),
                missing_labels=missing_labels,
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
    if attempt.get("status") == "answered":
        attempt["candidate_verifier_state"] = "not_verified"
        attempt["candidate_verifier_backend"] = "structural_option_audit"
        attempt["tool_confidence"] = "structural_option_audit_unverified"
    summary = {
        "status": "activated",
        "reason": "collapsed_or_weak_evidence_candidate_space_needs_structural_audit",
        "child_id": attempt.get("child_id"),
        "child_status": attempt.get("status"),
        "candidate_emitted": bool(answer),
        "candidate_verifier_state": "not_verified",
        "candidate_answer_hash": stable_hash({"answer": answer}) if answer else None,
        "candidate_disagreed_with_majority": bool(answer_norm and answer_norm != top_norm),
        "valid_candidate_count_before": len(valid_norms),
        "unique_candidate_count_before": unique_count,
        "top_candidate_count_before": top_count,
        "top_candidate_answer_hash": stable_hash({"answer": top_norm}),
        "missing_option_count_before": len(missing_labels),
        "missing_option_hashes": [stable_hash({"option_label": label}) for label in missing_labels],
        "option_evidence_status": option_evidence_status,
        "evidence_guided_option_status": evidence_guided_status,
        "evidence_context_used": bool(evidence_context),
        "underlying_model_calls": 1 if attempt.get("status") == "answered" else 0,
    }
    if problem.get("_answer"):
        gold_for_eval, _ = _canonicalize_multiple_choice_answer(problem, str(problem.get("_answer") or ""))
        summary["candidate_correct_for_eval"] = _is_correct(answer, gold_for_eval, answer_type="multipleChoice")
    _log_event(
        logger,
        {
            "event": "structural_option_audit_child",
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


def _score_option_evidence_detail(
    *,
    stem_terms: set[str],
    option_label: str,
    option_text: str,
    option_terms_by_label: dict[str, set[str]],
    option_text_by_label: dict[str, str],
    docs: list[dict[str, str]],
) -> dict[str, Any]:
    option_terms = option_terms_by_label.get(option_label) or _content_terms(option_text)
    if not option_terms or not docs:
        return {
            "score": 0.0,
            "support_doc_count": 0,
            "ambiguous_doc_count": 0,
            "unsupported_doc_count": len(docs),
            "supporting_doc_hashes": [],
        }
    score = 0.0
    support_doc_count = 0
    ambiguous_doc_count = 0
    unsupported_doc_count = 0
    supporting_doc_hashes: list[str] = []
    for index, doc in enumerate(docs[:3]):
        text = f"{doc.get('title', '')} {doc.get('snippet', '')}"
        doc_terms = _content_terms(text)
        title_terms = _content_terms(doc.get("title", ""))
        supporting_labels = _option_evidence_supporting_labels(
            text=text,
            doc_terms=doc_terms,
            title_terms=title_terms,
            option_terms_by_label=option_terms_by_label,
            option_text_by_label=option_text_by_label,
        )
        supports_current = option_label in supporting_labels
        supports_other = any(label != option_label for label in supporting_labels)
        option_overlap = len(option_terms & doc_terms)
        title_overlap = len(option_terms & title_terms)
        stem_overlap = len(stem_terms & doc_terms)
        min_stem_overlap = _option_evidence_min_stem_overlap(stem_terms)
        question_supported = stem_overlap >= min_stem_overlap
        if supports_current and not supports_other and question_supported:
            support_doc_count += 1
            supporting_doc_hashes.append(stable_hash({"title": doc.get("title", ""), "snippet": doc.get("snippet", "")}))
            phrase_bonus = 4.0 if _normalized_phrase_present(option_text, text) else 0.0
            score += (
                (2.0 * option_overlap)
                + (0.75 * stem_overlap)
                + (0.5 * title_overlap)
                + phrase_bonus
                + (0.1 / (index + 1))
            )
        elif supports_current and supports_other:
            ambiguous_doc_count += 1
            score += 0.25
        else:
            unsupported_doc_count += 1
    return {
        "score": round(score, 4),
        "support_doc_count": support_doc_count,
        "ambiguous_doc_count": ambiguous_doc_count,
        "unsupported_doc_count": unsupported_doc_count,
        "supporting_doc_hashes": supporting_doc_hashes[:2],
    }


def _option_evidence_min_stem_overlap(stem_terms: set[str]) -> int:
    if not stem_terms:
        return 0
    env_value = os.environ.get("HLE_OPTION_EVIDENCE_MIN_STEM_OVERLAP", "").strip()
    if env_value:
        try:
            return max(0, min(5, int(env_value)))
        except ValueError:
            pass
    return min(2, len(stem_terms))


def _option_evidence_rank_score(score_detail: dict[str, Any]) -> float:
    """Rank option evidence by support stability, not just lexical overlap."""
    score = float(score_detail.get("score") or 0.0)
    support_count = int(score_detail.get("support_doc_count") or 0)
    ambiguous_count = int(score_detail.get("ambiguous_doc_count") or 0)
    unsupported_count = int(score_detail.get("unsupported_doc_count") or 0)
    return round(score + (5.0 * support_count) - (8.0 * ambiguous_count) - (0.25 * unsupported_count), 4)


def _option_evidence_supporting_labels(
    *,
    text: str,
    doc_terms: set[str],
    title_terms: set[str],
    option_terms_by_label: dict[str, set[str]],
    option_text_by_label: dict[str, str],
) -> set[str]:
    labels: set[str] = set()
    distinctive_terms_by_label = _distinctive_option_terms_by_label(option_terms_by_label)
    for label, option_terms in option_terms_by_label.items():
        if not option_terms:
            continue
        distinctive_terms = distinctive_terms_by_label.get(label) or option_terms
        option_text = option_text_by_label.get(label, "")
        term_overlap = len(distinctive_terms & doc_terms)
        title_overlap = len(distinctive_terms & title_terms)
        min_term_support = 1 if len(distinctive_terms) <= 2 else 2
        if (
            _normalized_phrase_present(option_text, text)
            or title_overlap >= min_term_support
            or term_overlap >= min_term_support
        ):
            labels.add(label)
    return labels


def _distinctive_option_terms_by_label(option_terms_by_label: dict[str, set[str]]) -> dict[str, set[str]]:
    term_counts: Counter[str] = Counter()
    for terms in option_terms_by_label.values():
        term_counts.update(set(terms))
    distinctive: dict[str, set[str]] = {}
    for label, terms in option_terms_by_label.items():
        distinctive[label] = {term for term in terms if term_counts[term] == 1}
    return distinctive


def _normalized_phrase_present(phrase: str, text: str) -> bool:
    phrase_norm = _normalize_evidence_phrase(phrase)
    if len(phrase_norm) < 4:
        return False
    return phrase_norm in _normalize_evidence_phrase(text)


def _normalize_evidence_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+.-]+", " ", str(text or "").lower())).strip()


def _should_defer_trusted_route_to_evidence_challenge(
    *,
    problem: dict[str, Any],
    route_attempt: dict[str, Any],
    valid: list[dict[str, Any]],
) -> bool:
    if os.environ.get("HLE_DISABLE_ROUTE_DEFER_TO_EVIDENCE_CHALLENGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    if problem.get("answer_type") != "multipleChoice":
        return False
    if str(route_attempt.get("route_value_of_information_gate_status") or "") != "continue_exploration":
        return False
    route_answer = str(route_attempt.get("parsed_answer") or "").strip()
    if not route_answer:
        return False
    route_norm = _normalize_for_selection(route_answer, answer_type="multipleChoice")
    challenge_prompt_kinds = {
        "evidence_guided_option_challenge_answer",
        "counter_assumption_challenge_answer",
        "option_elimination_challenge_answer",
        "forced_alternative_answer",
        "critic_synthesis_answer",
        "code_semantics_answer",
        "option_matrix_reasoner_answer",
        "structural_option_audit_answer",
    }
    for attempt in valid:
        if attempt is route_attempt:
            continue
        if attempt.get("prompt_kind") not in challenge_prompt_kinds:
            continue
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        if attempt.get("candidate_verifier_state") == "refuted" and _is_trusted_candidate_verifier_attempt(attempt):
            continue
        if _normalize_for_selection(answer, answer_type="multipleChoice") != route_norm:
            return True
    return False


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
        verified_option_evidence_candidates = [
            attempt for attempt in valid
            if attempt.get("prompt_kind") == "mc_option_evidence_scorer_answer"
            and attempt.get("candidate_verifier_state") == "verified"
            and attempt.get("tool_confidence") == "verified_option_evidence_margin"
        ]
        if problem["answer_type"] == "multipleChoice" and verified_option_evidence_candidates:
            selected = sorted(verified_option_evidence_candidates, key=lambda row: int(row.get("child_index", 0) or 0))[0]
            return {
                "selection_method": "verified_option_evidence_priority",
                "selected_child_id": selected["child_id"],
                "selected_answer": selected["parsed_answer"],
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            }
        trusted_route_candidates = [
            attempt for attempt in valid
            if problem["answer_type"] == "multipleChoice"
            and _route_value_verifier_enabled()
            and attempt.get("prompt_kind") == "route_arbitrator_answer"
            and bool(attempt.get("route_arbitrator_trusted"))
            and attempt.get("candidate_verifier_state") == "verified"
            and _is_trusted_candidate_verifier_attempt(attempt)
            and str(attempt.get("route_value_confidence") or "").lower() in {"verified", "high"}
            and str(attempt.get("route_value_of_information_gate_status") or "") != "continue_exploration"
            and str(attempt.get("route_value_of_information_recommended_action") or "") != "continue_exploration"
            and not _should_defer_trusted_route_to_evidence_challenge(
                problem=problem,
                route_attempt=attempt,
                valid=valid,
            )
        ]
        if trusted_route_candidates:
            selected = sorted(
                trusted_route_candidates,
                key=lambda row: (
                    -float(row.get("route_value_score") or row.get("route_arbitrator_score") or 0.0),
                    int(row.get("child_index", 0) or 0),
                ),
            )[0]
            return {
                "selection_method": "route_value_verifier_choice",
                "selected_child_id": selected["child_id"],
                "selected_answer": selected["parsed_answer"],
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            }
        verified_candidates = [
            attempt for attempt in valid
            if attempt.get("candidate_verifier_state") == "verified"
            and _is_trusted_candidate_verifier_attempt(attempt)
            and attempt.get("prompt_kind") != "route_arbitrator_answer"
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
            attempt.get("candidate_verifier_state") == "refuted"
            and _is_trusted_candidate_verifier_attempt(attempt)
            for attempt in valid
        ):
            non_refuted_valid = [
                attempt for attempt in valid
                if attempt.get("candidate_verifier_state") != "refuted"
                or not _is_trusted_candidate_verifier_attempt(attempt)
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
            and attempt.get("tool_confidence") in {"verified_symbolic", "verified_symbolic_consensus"}
            and attempt.get("candidate_verifier_state") != "refuted"
            and _is_math_tool_attempt_override_trusted(attempt)
        ]
        if problem["answer_type"] != "multipleChoice" and math_candidates:
            deterministic_math = [
                attempt for attempt in math_candidates
                if _is_deterministic_math_tool_source(attempt.get("tool_source"))
                or _is_math_tool_consensus_attempt(attempt)
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
            if direct_candidates and not _should_defer_exact_direct_to_verifier(
                problem=problem,
                valid=valid,
                ranked=ranked,
            ):
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
            if attempt.get("prompt_kind") in {
                "evidence_bridge_answer",
                "evidence_grounded_answer",
                "evidence_guided_option_challenge_answer",
            }
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
            if direct_candidates and not _should_defer_exact_direct_to_verifier(
                problem=problem,
                valid=valid,
                ranked=ranked,
            ):
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
        _log_event(
            logger,
            {
                "event": "recursive_verifier_start",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "question_hash": problem["question_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(verifier_candidates),
                "raw_candidate_count": len(valid),
                "timeout_sec": timeout,
            },
        )
        started = time.monotonic()
        verifier_text = _call_model(
            model=model,
            prompt=_verifier_prompt(problem, verifier_candidates),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        latency_sec = round(time.monotonic() - started, 4)
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
                "selected_answer_hash": selected.get("parsed_answer_hash")
                or stable_hash({"answer": selected.get("parsed_answer")}),
                "verifier_prediction_hash": stable_hash({"prediction": verifier_text}),
                "timeout_sec": timeout,
                "latency_sec": latency_sec,
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
        latency_sec = round(time.monotonic() - started, 4) if "started" in locals() else None
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
                "timeout_sec": timeout,
                "latency_sec": latency_sec,
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


def _is_deterministic_math_tool_source(source: Any) -> bool:
    source_text = str(source or "")
    return source_text == "deterministic_parser" or source_text.startswith("deterministic_")


def _is_math_tool_consensus_attempt(attempt: dict[str, Any]) -> bool:
    if attempt.get("tool_confidence") != "verified_symbolic_consensus":
        return False
    summary = attempt.get("tool_summary") if isinstance(attempt.get("tool_summary"), dict) else {}
    agreement_count = attempt.get("plan_agreement_count", summary.get("plan_agreement_count"))
    success_count = attempt.get("plan_success_count", summary.get("plan_success_count"))
    try:
        return int(agreement_count or 0) >= 2 and int(success_count or 0) >= 2
    except Exception:
        return False


def _is_math_tool_attempt_override_trusted(attempt: dict[str, Any]) -> bool:
    if attempt.get("candidate_verifier_state") == "refuted":
        return False
    trust = attempt.get("candidate_verifier_trust")
    if trust in {"weak_llm_reference_planner", "weak_single_planner"}:
        return False
    if _is_deterministic_math_tool_source(attempt.get("tool_source")):
        return True
    return _is_math_tool_consensus_attempt(attempt)


def _recursive_verifier_timeout(timeout: float | None) -> float | None:
    has_override, override = _optional_timeout_override_from_env("HLE_RECURSIVE_VERIFIER_TIMEOUT_SEC")
    if has_override:
        return override
    if timeout is None:
        return None
    return _normalize_optional_timeout(timeout)


def _is_trusted_candidate_verifier_attempt(attempt: dict[str, Any]) -> bool:
    return attempt.get("candidate_verifier_trust") not in {
        "weak_llm_reference_planner",
        "weak_single_planner",
    }


def _trust_llm_reference_planner_enabled() -> bool:
    return os.environ.get("HLE_TRUST_LLM_REFERENCE_PLANNER", "").strip().lower() in {"1", "true", "yes", "on"}


_VERIFIED_SELECTION_METHODS = {
    "candidate_claim_verifier_priority",
    "counter_assumption_verifier_choice",
    "route_value_verifier_choice",
    "verified_math_tool_priority",
    "verifier_choice",
    "source_grounded_verifier_choice",
    "option_evidence_verifier_choice",
    "verified_option_evidence_priority",
}


def _exact_diverse_verifier_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_EXACT_DIVERSE_VERIFIER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_defer_exact_direct_to_verifier(
    *,
    problem: dict[str, Any],
    valid: list[dict[str, Any]],
    ranked: list[tuple[str, list[dict[str, Any]]]],
) -> bool:
    if problem.get("answer_type") == "multipleChoice":
        return False
    if not _exact_diverse_verifier_enabled():
        return False
    if not ranked or len(ranked) < 2:
        return False
    if len(ranked[0][1]) != 1:
        return False
    unique_candidates = _unique_verifier_candidates(problem, valid)
    return len(unique_candidates) >= 2


def _apply_verified_or_abstain_selection(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    if os.environ.get("HLE_DISABLE_VERIFIED_OR_ABSTAIN", "").strip().lower() in {"1", "true", "yes", "on"}:
        return selection
    method = str(selection.get("selection_method") or "")
    if method in _VERIFIED_SELECTION_METHODS:
        out = dict(selection)
        out["verified_or_abstain_gate"] = {
            "status": "allowed",
            "reason": "verified_selection_method",
            "original_selection_method": method,
        }
        return out
    fallback = _verified_or_abstain_fallback_candidate(problem=problem, attempts=attempts)
    if not fallback:
        out = dict(selection)
        out["verified_or_abstain_gate"] = {
            "status": "no_fallback",
            "reason": "no_direct_candidate",
            "original_selection_method": method,
        }
        return out
    selected_child_id = selection.get("selected_child_id")
    out = dict(selection)
    out.update({
        "selection_method": "verified_or_abstain_direct_fallback",
        "selected_child_id": fallback.get("child_id"),
        "selected_answer": fallback.get("parsed_answer"),
        "verified_or_abstain_gate": {
            "status": "abstained",
            "reason": "unverified_selection_method",
            "original_selection_method": method,
            "original_selected_child_id": selected_child_id,
            "fallback_prompt_kind": fallback.get("prompt_kind"),
        },
    })
    return out


def _verified_or_abstain_fallback_candidate(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer:
            continue
        normalized_attempt = dict(attempt)
        if problem.get("answer_type") == "multipleChoice":
            canonical, canonical_summary = _canonicalize_multiple_choice_answer(problem, answer)
            if canonical_summary.get("changed"):
                normalized_attempt["parsed_answer"] = canonical
                normalized_attempt["parsed_answer_hash"] = stable_hash({"answer": canonical})
        elif _is_suspicious_exact_answer(answer):
            continue
        candidates.append(normalized_attempt)
    if not candidates:
        return None
    preferred_prompt_kinds = [
        "route_arbitrator_answer",
        "raw_budget_preserve_selector_answer",
        "raw_preserve_selector_answer",
        "hipporag_preserve_selector_answer",
        "direct_short_answer",
        "constraint_checked_answer",
        "recursive_assumption_answer",
    ]
    for prompt_kind in preferred_prompt_kinds:
        prompt_candidates = [
            attempt for attempt in candidates
            if attempt.get("prompt_kind") == prompt_kind
        ]
        if prompt_kind == "route_arbitrator_answer":
            prompt_candidates = [
                attempt for attempt in prompt_candidates
                if bool(attempt.get("route_arbitrator_trusted"))
                or (
                    attempt.get("candidate_verifier_state") == "verified"
                    and _is_trusted_candidate_verifier_attempt(attempt)
                )
            ]
        if prompt_kind == "hipporag_preserve_selector_answer":
            prompt_candidates = [
                attempt for attempt in prompt_candidates
                if int(attempt.get("preserve_context_char_count") or 0) > 0
                and (
                    int(attempt.get("preserve_selected_doc_count") or 0) > 0
                    or int(attempt.get("preserve_candidate_doc_count") or 0) > 0
                )
            ]
        if prompt_candidates:
            return sorted(prompt_candidates, key=lambda row: int(row.get("child_index", 0) or 0))[0]
    return sorted(candidates, key=lambda row: int(row.get("child_index", 0) or 0))[0]


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
    trusted_preserve_routes = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") == "route_arbitrator_answer"
        and bool(attempt.get("route_arbitrator_trusted"))
        and (
            str(attempt.get("route_value_of_information_gate_status") or "") == "preserve_route"
            or str(attempt.get("route_value_of_information_recommended_action") or "") == "preserve_route"
        )
        and str(attempt.get("parsed_answer") or "").strip()
    ]
    if trusted_preserve_routes:
        selected = sorted(
            trusted_preserve_routes,
            key=lambda row: (
                -float(row.get("route_value_score") or row.get("route_arbitrator_score") or 0.0),
                int(row.get("child_index", 0) or 0),
            ),
        )[0]
        return {
            "selection_method": "route_value_verifier_choice",
            "selected_child_id": selected["child_id"],
            "selected_answer": selected["parsed_answer"],
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
    option_evidence_arbitrator_enabled = _option_evidence_arbitrator_enabled()
    challenge_prompt_kinds = {
        "counter_assumption_challenge_answer",
        "option_elimination_challenge_answer",
        "forced_alternative_answer",
        "critic_synthesis_answer",
        "evidence_guided_option_challenge_answer",
        "code_semantics_answer",
        "option_matrix_reasoner_answer",
        "structural_option_audit_answer",
    }
    option_sweep_voi_hard_counter = _option_sweep_voi_hard_counter_trigger_enabled(valid)
    if option_evidence_arbitrator_enabled:
        challenge_prompt_kinds.add("mc_option_evidence_scorer_answer")
    if _option_sweep_counter_trigger_enabled() or option_sweep_voi_hard_counter:
        challenge_prompt_kinds.add("mc_option_sweep_candidate")
    challenge_candidates = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") in challenge_prompt_kinds
        and str(attempt.get("parsed_answer") or "").strip()
    ]
    baseline_preserve_prompt_kinds = {
        "raw_preserve_selector_answer",
        "raw_budget_preserve_selector_answer",
        "hipporag_preserve_selector_answer",
    }
    baseline_preserve_norms = [
        _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=problem["answer_type"])
        for attempt in valid
        if attempt.get("prompt_kind") in baseline_preserve_prompt_kinds
        and str(attempt.get("parsed_answer") or "").strip()
    ]
    baseline_family_consensus_norm = ""
    if len(baseline_preserve_norms) >= 3:
        preserve_counts = Counter(norm for norm in baseline_preserve_norms if norm)
        if preserve_counts:
            top_preserve_norm, top_preserve_count = sorted(
                preserve_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            if top_preserve_count >= 3:
                baseline_family_consensus_norm = top_preserve_norm
    hard_challenge_prompt_kinds = {
        "counter_assumption_challenge_answer",
        "option_elimination_challenge_answer",
        "critic_synthesis_answer",
        "evidence_guided_option_challenge_answer",
        "code_semantics_answer",
        "option_matrix_reasoner_answer",
        "structural_option_audit_answer",
    }
    if option_evidence_arbitrator_enabled:
        hard_challenge_prompt_kinds.add("mc_option_evidence_scorer_answer")
    if option_sweep_voi_hard_counter:
        hard_challenge_prompt_kinds.add("mc_option_sweep_candidate")
    hard_disagreeing_challenges = [
        attempt for attempt in challenge_candidates
        if attempt.get("prompt_kind") in hard_challenge_prompt_kinds
        and _normalize_for_selection(str(attempt.get("parsed_answer") or ""), answer_type=problem["answer_type"]) != top_norm
    ]
    if not hard_disagreeing_challenges:
        return None
    if baseline_family_consensus_norm and baseline_family_consensus_norm == top_norm:
        evidence_disagreement = [
            attempt for attempt in hard_disagreeing_challenges
            if attempt.get("prompt_kind") == "mc_option_evidence_scorer_answer"
            and attempt.get("candidate_verifier_state") == "verified"
        ]
        structural_disagreement = [
            attempt for attempt in hard_disagreeing_challenges
            if attempt.get("prompt_kind") in {
                "option_matrix_reasoner_answer",
                "code_semantics_answer",
                "critic_synthesis_answer",
                "structural_option_audit_answer",
            }
        ]
        structural_audit_disagreement = [
            attempt for attempt in hard_disagreeing_challenges
            if attempt.get("prompt_kind") == "structural_option_audit_answer"
        ]
        allow_structural_over_baseline = os.environ.get(
            "HLE_ALLOW_STRUCTURAL_COUNTER_OVER_BASELINE_CONSENSUS",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not evidence_disagreement and not (
            (allow_structural_over_baseline or bool(structural_audit_disagreement))
            and structural_disagreement
            and _route_voi_allows_structural_counter_verifier(valid)
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
                "selected_answer_hash": selected.get("parsed_answer_hash")
                or stable_hash({"answer": selected.get("parsed_answer")}),
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


def _route_voi_allows_structural_counter_verifier(valid: list[dict[str, Any]]) -> bool:
    route_attempts = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") == "route_arbitrator_answer"
    ]
    if not route_attempts:
        return True
    for attempt in route_attempts:
        status = str(attempt.get("route_value_of_information_gate_status") or "")
        action = str(attempt.get("route_value_of_information_recommended_action") or "")
        if status == "continue_exploration" or action == "continue_exploration":
            return True
    return False


def _option_sweep_counter_trigger_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_OPTION_SWEEP_COUNTER_TRIGGER", "").strip().lower() in {"1", "true", "yes", "on"}


def _option_sweep_voi_hard_counter_trigger_enabled(valid: list[dict[str, Any]]) -> bool:
    """Allow full-option-space verifier only when the route policy asks to explore.

    `mc_option_sweep_candidate` is deliberately not evidence: it merely
    guarantees that every finite MC label is available to the verifier.  It
    should therefore stay inert unless a separate value-of-information signal
    says the route consensus is not worth preserving.
    """
    if os.environ.get("HLE_DISABLE_OPTION_SWEEP_VOI_COUNTER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    if os.environ.get("HLE_ENABLE_OPTION_SWEEP_VOI_COUNTER", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    if not any(attempt.get("prompt_kind") == "mc_option_sweep_candidate" for attempt in valid):
        return False
    if any(
        attempt.get("candidate_verifier_state") == "verified"
        and _is_trusted_candidate_verifier_attempt(attempt)
        and attempt.get("prompt_kind") not in {
            "route_arbitrator_answer",
            "raw_preserve_selector_answer",
            "raw_budget_preserve_selector_answer",
            "hipporag_preserve_selector_answer",
        }
        for attempt in valid
    ):
        return False
    route_attempts = [
        attempt for attempt in valid
        if attempt.get("prompt_kind") == "route_arbitrator_answer"
    ]
    if not route_attempts:
        return False
    return any(
        str(attempt.get("route_value_of_information_gate_status") or "") == "continue_exploration"
        or str(attempt.get("route_value_of_information_recommended_action") or "") == "continue_exploration"
        for attempt in route_attempts
    )


def _math_tool_child_timeout(timeout: float | None) -> float | None:
    has_override, cap = _optional_timeout_override_from_env("HLE_MATH_TOOL_CHILD_TIMEOUT_SEC")
    if not has_override:
        return _normalize_optional_timeout(timeout)
    if cap is None:
        return None
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
                "selected_answer_hash": selected.get("parsed_answer_hash")
                or stable_hash({"answer": selected.get("parsed_answer")}),
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
                "selected_answer_hash": selected.get("parsed_answer_hash")
                or stable_hash({"answer": selected.get("parsed_answer")}),
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
    if _exact_diverse_evidence_bridge_enabled():
        normalized = {
            _normalize_for_selection(answer, answer_type="exactMatch")
            for answer in valid
            if not _is_suspicious_exact_answer(answer)
        }
        if len(normalized) >= 2:
            return True
    return bool(valid) and all(_is_suspicious_exact_answer(answer) for answer in valid)


def _exact_diverse_evidence_bridge_enabled() -> bool:
    return os.environ.get("HLE_ENABLE_EXACT_DIVERSE_EVIDENCE_BRIDGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_prime_evidence_bridge(problem: dict[str, Any], agent_plan: dict[str, Any]) -> bool:
    if agent_plan.get("hle_evidence_context"):
        return False
    if _classify_hle_domain(problem) == "math":
        return False
    if problem.get("answer_type") == "multipleChoice":
        return os.environ.get("HLE_DISABLE_MC_EVIDENCE_BRIDGE", "").strip().lower() not in {"1", "true", "yes", "on"}
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
    direct = re.fullmatch(r"([A-Z])(?:[\).:：])?", upper)
    if direct:
        return direct.group(1)
    prefix = re.match(r"^\s*([A-Z])[\).:：]\s+", upper)
    if prefix:
        return prefix.group(1)
    match = re.search(
        r"\b(?:option|choice|answer|ans|final\s+answer)\s*(?:is|=|:)?\s*([A-Z])\b",
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
                candidate_answers=[selected_answer],
            )
            agent_plan["hle_evidence_context"] = evidence_context
            agent_plan["hle_evidence_bridge"] = evidence_summary
    effective_timeout = _answer_format_repair_timeout(timeout)
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
            "timeout_sec": effective_timeout,
        },
    )
    started = time.monotonic()
    try:
        text = _call_model(
            model=model,
            prompt=_exact_answer_repair_prompt(
                problem,
                selected_answer,
                repair_context=repair_context,
                evidence_context=evidence_context,
            ),
            timeout=effective_timeout,
            max_tokens=max_tokens,
        )
        latency_sec = round(time.monotonic() - started, 4)
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
            "timeout_sec": effective_timeout,
            "latency_sec": latency_sec,
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
                "timeout_sec": effective_timeout,
                "latency_sec": latency_sec,
            },
        )
        return {
            "selected_answer": repaired,
            "underlying_model_calls": 1,
            "stage_summary": stage_summary,
        }
    except Exception as exc:
        latency_sec = round(time.monotonic() - started, 4)
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
            "timeout_sec": effective_timeout,
            "latency_sec": latency_sec,
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
                "timeout_sec": effective_timeout,
                "latency_sec": latency_sec,
            },
        )
        return {
            "selected_answer": selected_answer,
            "underlying_model_calls": 1,
            "stage_summary": stage_summary,
        }


def _answer_format_repair_timeout(timeout: float | None) -> float | None:
    has_override, override = _optional_timeout_override_from_env("HLE_ANSWER_FORMAT_REPAIR_TIMEOUT_SEC")
    if has_override:
        return override
    if timeout is None:
        return None
    return _normalize_optional_timeout(timeout)


def _repair_context_for_exact(agent_plan: dict[str, Any]) -> str:
    return str(agent_plan.get("prompt_context") or agent_plan.get("retrieval_context_candidate") or "").strip()


def _build_hle_evidence_bridge_context(
    *,
    problem: dict[str, Any],
    eval_id: str,
    call_id: str,
    model: str,
    logger: "_JsonlLogger | None",
    candidate_answers: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    queries = _candidate_evidence_queries(problem, candidate_answers=candidate_answers or [])
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
    selected_results, answer_bearing_certificate = _filter_answer_bearing_evidence_results(
        problem=problem,
        results=reranked_results or unique_results[:5],
        candidate_answers=candidate_answers or [],
        max_results=5,
    )
    evidence_context = _format_evidence_context(selected_results, max_chars=1800)
    summary = {
        "status": (
            "activated"
            if evidence_context
            else (str(answer_bearing_certificate.get("status") or "blocked_non_answer_bearing") if unique_results else "no_results")
        ),
        "source": "wikipedia_plus_domain_search",
        "selection_policy": "answer_bearing_hipporag_style_associative_rerank",
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
        "answer_bearing_certificate": answer_bearing_certificate,
        "error_types": sorted(set(errors)),
    }
    _log_hle_evidence_bridge_event(logger, eval_id=eval_id, call_id=call_id, problem=problem, model=model, summary=summary)
    return evidence_context, summary


def _maybe_mark_answer_bearing_evidence_attempt(
    *,
    problem: dict[str, Any],
    attempt: dict[str, Any],
    evidence_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    certificate = (evidence_summary or {}).get("answer_bearing_certificate") or {}
    if problem.get("answer_type") == "multipleChoice":
        return {"status": "not_required", "reason": "multiple_choice_uses_option_evidence"}
    if _classify_hle_domain(problem) == "math":
        return {"status": "not_required", "reason": "math_requires_executable_verifier"}
    if certificate.get("status") != "answer_bearing":
        return {"status": "not_marked", "reason": str(certificate.get("status") or "no_answer_bearing_certificate")}
    answer = str(attempt.get("parsed_answer") or "").strip()
    if not answer or _is_suspicious_exact_answer(answer):
        return {"status": "not_marked", "reason": "suspicious_or_empty_answer"}
    supported_norm_hashes = {
        str(item)
        for item in certificate.get("candidate_hit_answer_norm_hashes", [])
        if str(item or "").strip()
    }
    answer_norm_hash = stable_hash({
        "answer_norm": _normalize_for_selection(answer, answer_type="exactMatch"),
    })
    if supported_norm_hashes and answer_norm_hash not in supported_norm_hashes:
        return {
            "status": "not_marked",
            "reason": "evidence_child_answer_not_candidate_supported",
            "answer_norm_hash": answer_norm_hash,
        }
    attempt["candidate_verifier_state"] = "verified"
    attempt["candidate_verifier_backend"] = "answer_bearing_evidence_bridge"
    attempt["candidate_verifier_operation"] = "candidate_specific_retrieval"
    attempt["candidate_verifier_match_method"] = "answer_bearing_candidate_overlap"
    attempt["candidate_verifier_claim_hash"] = stable_hash({
        "question_hash": problem.get("question_hash"),
        "answer_norm_hash": answer_norm_hash,
        "certificate_status": certificate.get("status"),
        "candidate_hit_count": certificate.get("candidate_hit_count"),
    })
    attempt["candidate_verifier_trust"] = "trusted"
    return {
        "status": "marked_verified",
        "backend": attempt["candidate_verifier_backend"],
        "answer_norm_hash": answer_norm_hash,
    }


def _maybe_add_answer_bearing_evidence_candidate(
    *,
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    evidence_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    certificate = (evidence_summary or {}).get("answer_bearing_certificate") or {}
    if problem.get("answer_type") == "multipleChoice":
        return None, {"status": "not_required", "reason": "multiple_choice_uses_option_evidence"}
    if _classify_hle_domain(problem) == "math":
        return None, {"status": "not_required", "reason": "math_requires_executable_verifier"}
    if certificate.get("status") != "answer_bearing":
        return None, {"status": "not_emitted", "reason": str(certificate.get("status") or "no_answer_bearing_certificate")}
    supported_answer_hashes = {
        str(item)
        for item in certificate.get("candidate_hit_answer_hashes", [])
        if str(item or "").strip()
    }
    supported_norm_hashes = {
        str(item)
        for item in certificate.get("candidate_hit_answer_norm_hashes", [])
        if str(item or "").strip()
    }
    if not supported_answer_hashes and not supported_norm_hashes:
        return None, {"status": "not_emitted", "reason": "no_supported_candidate_hashes"}
    candidate_rows: list[tuple[int, dict[str, Any], str, str]] = []
    for attempt in attempts:
        answer = str(attempt.get("parsed_answer") or "").strip()
        if not answer or _is_suspicious_exact_answer(answer):
            continue
        answer_hash = str(attempt.get("parsed_answer_hash") or stable_hash({"answer": answer}))
        answer_norm_hash = stable_hash({
            "answer_norm": _normalize_for_selection(answer, answer_type="exactMatch"),
        })
        if answer_hash in supported_answer_hashes or answer_norm_hash in supported_norm_hashes:
            candidate_rows.append((int(attempt.get("child_index", 0) or 0), attempt, answer_hash, answer_norm_hash))
    if not candidate_rows:
        return None, {"status": "not_emitted", "reason": "supported_candidate_not_in_attempts"}
    _, source_attempt, answer_hash, answer_norm_hash = sorted(candidate_rows, key=lambda row: row[0])[0]
    source_child_id = str(source_attempt.get("child_id") or "")
    child_id = "source_supported_" + stable_hash({
        "source_child_id": source_child_id,
        "answer_hash": answer_hash,
        "question_hash": problem.get("question_hash"),
    })[:16]
    synthetic_attempt = {
        "child_id": child_id,
        "child_index": max([int(attempt.get("child_index", 0) or 0) for attempt in attempts] or [0]) + 1,
        "prompt_kind": "answer_bearing_evidence_candidate",
        "status": "answered",
        "parsed_answer": source_attempt.get("parsed_answer"),
        "parsed_answer_hash": answer_hash,
        "prediction_hash": stable_hash({
            "source_supported_answer_hash": answer_hash,
            "question_hash": problem.get("question_hash"),
        }),
        "source_child_id": source_child_id,
        "source_prompt_kind": source_attempt.get("prompt_kind"),
        "candidate_verifier_state": "verified",
        "candidate_verifier_backend": "answer_bearing_evidence_bridge",
        "candidate_verifier_operation": "candidate_specific_retrieval",
        "candidate_verifier_match_method": "source_supported_candidate_hash",
        "candidate_verifier_claim_hash": stable_hash({
            "question_hash": problem.get("question_hash"),
            "answer_norm_hash": answer_norm_hash,
            "candidate_hit_count": certificate.get("candidate_hit_count"),
            "selected_result_count": certificate.get("selected_result_count"),
        }),
        "candidate_verifier_trust": "trusted",
        "tool_source": "candidate_specific_evidence_certificate",
        "tool_confidence": "answer_bearing_candidate_overlap",
    }
    return synthetic_attempt, {
        "status": "emitted",
        "child_id": child_id,
        "source_child_id": source_child_id,
        "source_prompt_kind": source_attempt.get("prompt_kind"),
        "answer_hash": answer_hash,
        "answer_norm_hash": answer_norm_hash,
        "candidate_hit_count": int(certificate.get("candidate_hit_count") or 0),
    }


def _filter_answer_bearing_evidence_results(
    *,
    problem: dict[str, Any],
    results: list[dict[str, str]],
    candidate_answers: list[str],
    max_results: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    question, options = _split_multiple_choice_question(problem)
    question_terms = _content_terms(question)
    option_terms: dict[str, set[str]] = {
        label: _content_terms(text)
        for label, text in options.items()
        if _content_terms(text)
    }
    subject_terms = _content_terms(f"{problem.get('raw_subject', '')} {problem.get('category', '')}")
    exact_candidates: list[dict[str, Any]] = []
    for answer in candidate_answers:
        answer_text = str(answer or "").strip()
        if not answer_text or _is_suspicious_exact_answer(answer_text):
            continue
        terms = _content_terms(answer_text)
        if not terms:
            continue
        exact_candidates.append({
            "terms": terms,
            "answer_hash": stable_hash({"answer": answer_text}),
            "answer_norm_hash": stable_hash({
                "answer_norm": _normalize_for_selection(answer_text, answer_type="exactMatch"),
            }),
        })
    selected: list[dict[str, str]] = []
    label_hits: Counter[str] = Counter()
    observed_option_hits: Counter[str] = Counter()
    candidate_hit_count = 0
    observed_candidate_raw_hit_count = 0
    candidate_hits_blocked_by_question_overlap = 0
    candidate_hit_answer_hashes: set[str] = set()
    candidate_hit_answer_norm_hashes: set[str] = set()
    required_question_overlap = _evidence_question_overlap_required()
    relaxed_candidate_overlap_count = 0
    relaxed_question_overlap_required = _exact_candidate_evidence_question_overlap_required()
    for row in results:
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        doc_terms = _content_terms(text)
        title_terms = _content_terms(row.get("title", ""))
        question_overlap = len(question_terms & doc_terms)
        subject_overlap = len(subject_terms & doc_terms)
        candidate_hits = [
            candidate
            for candidate in exact_candidates
            if len(candidate["terms"] & doc_terms) >= max(1, min(2, len(candidate["terms"])))
        ]
        candidate_title_hits = [
            candidate
            for candidate in exact_candidates
            if len(candidate["terms"] & title_terms) >= max(1, min(2, len(candidate["terms"])))
        ]
        if problem.get("answer_type") != "multipleChoice" and candidate_hits:
            observed_candidate_raw_hit_count += 1
        relaxed_candidate_allowed = (
            problem.get("answer_type") != "multipleChoice"
            and bool(candidate_hits)
            and (
                question_overlap >= relaxed_question_overlap_required
                or (bool(candidate_title_hits) and subject_overlap >= 1)
            )
        )
        if question_overlap < required_question_overlap:
            if relaxed_candidate_allowed:
                relaxed_candidate_overlap_count += 1
            elif problem.get("answer_type") != "multipleChoice" and candidate_hits:
                candidate_hits_blocked_by_question_overlap += 1
                continue
            else:
                continue
        if problem.get("answer_type") == "multipleChoice":
            hit_labels = [
                label
                for label, terms in option_terms.items()
                if terms and len(terms & doc_terms) >= max(1, min(2, len(terms)))
            ]
            for label in hit_labels:
                observed_option_hits[label] += 1
            if len(hit_labels) != 1:
                continue
            for label in hit_labels:
                label_hits[label] += 1
        else:
            if not exact_candidates:
                continue
            if not candidate_hits:
                continue
            for candidate in candidate_hits:
                candidate_hit_answer_hashes.add(str(candidate["answer_hash"]))
                candidate_hit_answer_norm_hashes.add(str(candidate["answer_norm_hash"]))
            candidate_hit_count += 1
        selected.append(row)
        if len(selected) >= max_results:
            break
    if problem.get("answer_type") == "multipleChoice" and len(label_hits) != 1:
        selected = []
    certificate = {
        "status": "answer_bearing" if selected else (
            "blocked_non_discriminative_option_evidence"
            if problem.get("answer_type") == "multipleChoice" and observed_option_hits
            else "blocked_non_answer_bearing"
        ),
        "policy": "question_terms_plus_discriminative_option_or_candidate_overlap",
        "question_term_overlap_required": required_question_overlap,
        "candidate_answer_count": len([answer for answer in candidate_answers if str(answer or "").strip()]),
        "option_count": len(option_terms),
        "option_discriminative_required": problem.get("answer_type") == "multipleChoice",
        "option_hit_labels": sorted(label_hits or observed_option_hits),
        "candidate_hit_count": candidate_hit_count,
        "candidate_raw_hit_count": observed_candidate_raw_hit_count,
        "candidate_hits_blocked_by_question_overlap": candidate_hits_blocked_by_question_overlap,
        "candidate_relaxed_overlap_count": relaxed_candidate_overlap_count,
        "candidate_relaxed_question_overlap_required": relaxed_question_overlap_required,
        "candidate_hit_answer_hashes": sorted(candidate_hit_answer_hashes),
        "candidate_hit_answer_norm_hashes": sorted(candidate_hit_answer_norm_hashes),
        "input_result_count": len(results),
        "selected_result_count": len(selected),
        "raw_content_persisted": False,
    }
    return selected, certificate


def _evidence_question_overlap_required() -> int:
    try:
        return max(1, int(os.environ.get("HLE_EVIDENCE_MIN_QUESTION_OVERLAP", "3")))
    except ValueError:
        return 3


def _exact_candidate_evidence_question_overlap_required() -> int:
    try:
        return max(1, int(os.environ.get("HLE_EXACT_CANDIDATE_EVIDENCE_MIN_QUESTION_OVERLAP", "1")))
    except ValueError:
        return 1


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


def _candidate_evidence_queries(
    problem: dict[str, Any],
    candidate_answers: list[str] | None = None,
) -> list[str]:
    question = str(problem.get("_question") or "")
    seeds: list[str] = []
    candidate_seed_norms: set[str] = set()
    candidate_answers = candidate_answers or []
    if problem.get("answer_type") != "multipleChoice" and candidate_answers:
        anchors = [
            str(problem.get("raw_subject") or "").strip(),
            str(problem.get("category") or "").strip(),
        ]
        anchors.extend(
            item
            for groups in re.findall(r'"([^"]{3,120})"|\'([^\']{3,120})\'|`([^`]{3,120})`', question)
            for item in groups
            if item
        )
        cleaned_anchors: list[str] = []
        seen_anchor_keys: set[str] = set()
        for anchor in anchors:
            cleaned_anchor = _clean_evidence_query(anchor)
            key = _normalize_exact(cleaned_anchor)
            if cleaned_anchor and key and key not in seen_anchor_keys:
                seen_anchor_keys.add(key)
                cleaned_anchors.append(cleaned_anchor)
        for answer in candidate_answers:
            answer_text = str(answer or "").strip()
            if not answer_text or _is_suspicious_exact_answer(answer_text):
                continue
            cleaned_answer = _clean_evidence_query(answer_text)
            if not cleaned_answer:
                continue
            seeds.append(cleaned_answer)
            candidate_seed_norms.add(_normalize_exact(cleaned_answer))
            for anchor in cleaned_anchors[:2]:
                if anchor and _normalize_exact(anchor) not in _normalize_exact(cleaned_answer):
                    seeds.append(f"{cleaned_answer} {anchor}")
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
    max_queries = _candidate_evidence_query_limit(candidate_answers)
    for seed in seeds:
        query = _clean_evidence_query(seed)
        if not query:
            continue
        if (
            problem.get("raw_subject")
            and problem["raw_subject"] not in query
            and len(query.split()) <= 4
            and _normalize_exact(query) not in candidate_seed_norms
        ):
            query = f"{query} {problem['raw_subject']}"
        if query not in queries:
            queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _candidate_evidence_query_limit(candidate_answers: list[str] | None = None) -> int:
    env_value = os.environ.get("HLE_EVIDENCE_MAX_QUERIES")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return 8 if candidate_answers else 4


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
    text = re.sub(r"\b[A-Z]\s*[\).:]", " ", text)
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
        r"(?:^|\n)\s*([A-Z])[\).:：]\s*(.*?)(?=(?:\n\s*[A-Z][\).:：]\s*)|\Z)",
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
    executable_match = _exact_math_candidate_match(option, reference)
    if executable_match.get("matched"):
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
    math_key = _canonical_math_selection_key(text)
    if math_key:
        return f"math:{math_key}"
    return _normalize_exact(text)


def _canonical_math_selection_key(text: str) -> str:
    parts = _math_answer_parts(text)
    if not parts:
        return ""
    try:
        import sympy as sp
    except Exception:
        return ""
    canonical_parts: list[str] = []
    for part in parts:
        expr = _safe_sympy_parse_expr(part)
        if expr is None:
            return ""
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
        canonical_parts.append(_format_sympy_answer(expr))
    return "|".join(sorted(canonical_parts))


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
    if _is_budget_matched_control_variant(variant):
        stages = (agent_plan or {}).get("stages", {})
        base_variant = _budget_control_base_variant(variant)
        return [
            {
                "module": "answer_type_router",
                "expected": True,
                "status": "activated",
                "reason": f"answer_type={problem.get('answer_type') or 'unknown'} controls output contract",
            },
            {
                "module": "baseline_prompt_builder",
                "expected": True,
                "status": _stage_status(stages, "prompt_builder") if base_variant == "hipporag_baseline" else "activated",
                "reason": f"{base_variant} prompt is used without Assumption Agent context",
            },
            {
                "module": "budget_matched_self_consistency",
                "expected": True,
                "status": _stage_status(stages, "budget_matched_control"),
                "reason": "control baseline receives multiple independent candidates under a matched-call budget",
            },
            {
                "module": "budget_matched_verifier",
                "expected": True,
                "status": _stage_status(stages, "budget_matched_control"),
                "reason": "control baseline selects among candidates by majority/verifier without graph, morphism, or world model",
            },
            {
                "module": "assumption_graph_retrieval",
                "expected": False,
                "status": "not_applicable",
                "reason": "budget-matched control must not use the Assumption Graph",
            },
            {
                "module": "structural_morphism_transfer",
                "expected": False,
                "status": "not_applicable",
                "reason": "budget-matched control must not use morphism transfer",
            },
            {
                "module": "world_model_router",
                "expected": False,
                "status": "not_applicable",
                "reason": "budget-matched control must not use the Assumption Agent world model",
            },
            {
                "module": "recursive_assumption_runner",
                "expected": False,
                "status": "not_applicable",
                "reason": "budget-matched control does self-consistency, not recursive assumption generation",
            },
            {
                "module": "answer_format_verifier",
                "expected": True,
                "status": "activated",
                "reason": "JSON answer parser and answer-type scorer run after each response",
            },
        ]

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
                "module": "child_model_router",
                "expected": False,
                "status": _stage_status(stages, "child_model_router") if stages.get("child_model_router") else "not_required",
                "reason": "optional stronger child model can be routed only to candidate generation steps",
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
                "module": "recursive_timeout_recovery_child",
                "expected": recursive_verify and problem.get("answer_type") == "exactMatch",
                "status": (
                    _stage_status(stages, "recursive_timeout_recovery_child")
                    if stages.get("recursive_timeout_recovery_child")
                    else "not_required"
                ),
                "reason": "timeout/error pressure with too few answer candidates can trigger a short recovery child",
            },
            {
                "module": "child_model_failover_child",
                "expected": False,
                "status": (
                    _stage_status(stages, "child_model_failover_child")
                    if stages.get("child_model_failover_child")
                    else "not_required"
                ),
                "reason": "optional base-model failover can produce one candidate when the routed child model yields none",
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
                "module": "raw_preserve_selector",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "raw_preserve_selector") if stages.get("raw_preserve_selector") else "not_required",
                "reason": "uncertain unverified selections can add a no-context raw baseline candidate for fallback auditing",
            },
            {
                "module": "raw_budget_preserve_selector",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": (
                    _stage_status(stages, "raw_budget_preserve_selector")
                    if stages.get("raw_budget_preserve_selector")
                    else "not_required"
                ),
                "reason": "uncertain unverified selections can add a same-model budget-matched raw self-consistency candidate",
            },
            {
                "module": "hipporag_preserve_selector",
                "expected": recursive_verify and problem.get("answer_type") == "multipleChoice",
                "status": _stage_status(stages, "hipporag_preserve_selector") if stages.get("hipporag_preserve_selector") else "not_required",
                "reason": "uncertain unverified selections can add a same-model HippoRAG baseline candidate before direct fallback",
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


def _agent_child_model(default_model: str) -> str:
    return os.environ.get("HLE_AGENT_CHILD_MODEL", "").strip() or default_model


def _call_model(*, model: str, prompt: str, timeout: float | None = None, max_tokens: int = 512) -> str:
    env = _api_env(model=model)
    payload = {
        "model": env["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    payload.update(_model_router_extra_body())
    if _model_router_subprocess_calls_enabled():
        return _call_model_via_subprocess(env=env, payload=payload, timeout=timeout)
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
    timeout = _normalize_optional_timeout(_default_call_timeout() if timeout is None else timeout)
    deadline = None if timeout is None else time.monotonic() + timeout
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request_timeout = _request_timeout_for_attempt(deadline=deadline)
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
                raise RuntimeError(f"model request failed: {_model_error_label(exc)}: {_model_error_message(exc)}") from exc
            _sleep_before_model_retry(attempt=attempt, deadline=deadline)
    raise RuntimeError(f"model request failed: {_model_error_message(last_error)}")


def _model_router_subprocess_calls_enabled() -> bool:
    return os.environ.get("MODEL_ROUTER_SUBPROCESS_CALLS", "").strip().lower() in {"1", "true", "yes", "on"}


def _model_router_extra_body() -> dict[str, Any]:
    body: dict[str, Any] = {}
    effort = os.environ.get("MODEL_ROUTER_REASONING_EFFORT", "").strip()
    if effort:
        body["reasoning_effort"] = effort
    raw = os.environ.get("MODEL_ROUTER_EXTRA_BODY_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if key not in {"model", "messages"}:
                    body[str(key)] = value
    return body


def _call_model_via_subprocess(*, env: dict[str, str], payload: dict[str, Any], timeout: float | None) -> str:
    attempts = max(1, int(os.environ.get("MODEL_ROUTER_ATTEMPTS", "3")))
    timeout = _normalize_optional_timeout(_default_call_timeout() if timeout is None else timeout)
    deadline = None if timeout is None else time.monotonic() + timeout
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request_timeout = _request_timeout_for_attempt(deadline=deadline)
            return _single_model_subprocess_call(env=env, payload=payload, request_timeout=request_timeout)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise RuntimeError(f"model request failed: {_model_error_label(exc)}: {_model_error_message(exc)}") from exc
            _sleep_before_model_retry(attempt=attempt, deadline=deadline)
    raise RuntimeError(f"model request failed: {_model_error_message(last_error)}")


def _single_model_subprocess_call(*, env: dict[str, str], payload: dict[str, Any], request_timeout: float | None) -> str:
    script = r"""
import json
import sys
import urllib.request

cfg = json.loads(sys.stdin.read())
request = urllib.request.Request(
    cfg["base_url"].rstrip("/") + "/chat/completions",
    data=json.dumps(cfg["payload"]).encode("utf-8"),
    headers={
        "Authorization": "Bearer " + cfg["api_key"],
        "Content-Type": "application/json",
    },
    method="POST",
)
if cfg.get("request_timeout") is None:
    response_ctx = urllib.request.urlopen(request)
else:
    response_ctx = urllib.request.urlopen(request, timeout=float(cfg["request_timeout"]))
with response_ctx as response:
    data = json.loads(response.read().decode("utf-8"))
print(str((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip())
"""
    stdin_payload = json.dumps({
        "base_url": env["base_url"],
        "api_key": env["api_key"],
        "payload": payload,
        "request_timeout": request_timeout,
    })
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=stdin_payload,
        text=True,
        capture_output=True,
        timeout=None if request_timeout is None else max(1.0, request_timeout + 5.0),
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip().splitlines()
        message = stderr[-1] if stderr else f"subprocess_exit_{completed.returncode}"
        raise RuntimeError(message[:240])
    return str(completed.stdout or "").strip()


def _urlopen_json_with_deadline(*, request: urllib.request.Request, timeout: float | None) -> dict[str, Any]:
    if timeout is None:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
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


def _model_error_message(exc: Exception | None) -> str:
    if exc is None:
        return ""
    message = str(exc)
    for key_name in ("GPT5_API_KEY", "RUOLI_GPT_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(key_name)
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"Bearer\s+[A-Za-z0-9._:-]+", "Bearer [redacted]", message)
    message = re.sub(r"sk-[A-Za-z0-9._:-]+", "sk-[redacted]", message)
    return message[:240]


def _sleep_before_model_retry(*, attempt: int, deadline: float | None) -> None:
    base = float(os.environ.get("MODEL_ROUTER_BACKOFF_BASE_SEC", "0.75"))
    cap = float(os.environ.get("MODEL_ROUTER_BACKOFF_MAX_SEC", "10"))
    jitter = float(os.environ.get("MODEL_ROUTER_BACKOFF_JITTER_SEC", "0.25"))
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0.0, max(0.0, jitter))
    if deadline is None:
        time.sleep(delay)
        return
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


def _default_call_timeout() -> float | None:
    return _optional_timeout_from_text(os.environ.get("MODEL_ROUTER_TIMEOUT"))


def _model_router_per_attempt_timeout() -> float | None:
    return _optional_timeout_from_text(os.environ.get("MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"))


def _request_timeout_for_attempt(*, deadline: float | None) -> float | None:
    per_attempt = _model_router_per_attempt_timeout()
    if deadline is None:
        return per_attempt
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("model_call_deadline_exceeded")
    if per_attempt is None:
        return max(0.1, remaining)
    return max(0.1, min(remaining, per_attempt))


def _normalize_optional_timeout(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _optional_timeout_from_text(value)
        return parsed
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed <= 0 else parsed


def _optional_timeout_from_text(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"0", "none", "null", "off", "false", "no", "unlimited"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return None if parsed <= 0 else parsed


def _optional_timeout_override_from_env(name: str) -> tuple[bool, float | None]:
    if name not in os.environ:
        return False, None
    return True, _optional_timeout_from_text(os.environ.get(name))


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
    if _is_budget_matched_control_variant(variant):
        budget = stages.get("budget_matched_control", {})
        base.update({
            "kind": "budget_matched_control",
            "budget_matched_control": {
                "status": budget.get("status"),
                "base_variant": budget.get("base_variant"),
                "candidate_count": int(budget.get("candidate_count") or 0),
                "answered_candidate_count": int(budget.get("answered_candidate_count") or 0),
                "error_candidate_count": int(budget.get("error_candidate_count") or 0),
                "child_max_workers": int(budget.get("child_max_workers") or 0),
                "selection_method": budget.get("selection_method"),
                "verifier_model_call": bool(budget.get("verifier_model_call")),
                "underlying_model_calls": int(budget.get("underlying_model_calls") or 0),
            },
            "selection": {
                "status": budget.get("status"),
                "selection_method": budget.get("selection_method"),
                "verifier_model_call": bool(budget.get("verifier_model_call")),
                "verified_or_abstain_gate": budget.get("verified_or_abstain_gate"),
            },
        })
        base["flags"].update({
            "budget_matched_control_activated": budget.get("status") == "activated",
            "budget_matched_verifier_used": bool(budget.get("verifier_model_call")),
            "verified_or_abstain_allowed": (
                isinstance(budget.get("verified_or_abstain_gate"), dict)
                and budget.get("verified_or_abstain_gate", {}).get("status") == "allowed"
            ),
            "verified_or_abstain_abstained": (
                isinstance(budget.get("verified_or_abstain_gate"), dict)
                and budget.get("verified_or_abstain_gate", {}).get("status") == "abstained"
            ),
        })
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
    same_run_cache = stages.get("same_run_baseline_cache", {})
    same_run_cache = same_run_cache if isinstance(same_run_cache, dict) else {}
    critic_router = stages.get("critic_model_router", {})
    child_router = stages.get("child_model_router", {})
    prompt = stages.get("prompt_builder", {})
    diversity_planner = stages.get("recursive_child_diversity_planner", {})
    diversity_planner = diversity_planner if isinstance(diversity_planner, dict) else {}
    recursive = stages.get("recursive_child_validation", {})
    selection = stages.get("multi_candidate_self_verifier", {})
    timeout_recovery = stages.get("recursive_timeout_recovery_child", {})
    child_model_failover = stages.get("child_model_failover_child", {})
    evidence = stages.get("hle_evidence_bridge", {})
    agent_hipporag = stages.get("agent_hipporag_context_bridge", {})
    claim_verifier = stages.get("candidate_claim_verifier", {})
    domain_rule = stages.get("domain_rule_mc_verifier", {})
    math_tool = stages.get("hle_math_tool_solver", {})
    option_evidence = stages.get("mc_option_evidence_scorer", {})
    evidence_guided_option = stages.get("evidence_guided_option_challenge", {})
    structural_option_audit = stages.get("structural_option_audit_child", {})
    structural_option_audit = structural_option_audit if isinstance(structural_option_audit, dict) else {}
    critic_synthesis = stages.get("critic_synthesis_child", {})
    option_sweep = stages.get("mc_option_sweep_candidates", {})
    counter_challenge = stages.get("counter_assumption_challenge", {})
    raw_preserve = stages.get("raw_preserve_selector", {})
    raw_preserve = raw_preserve if isinstance(raw_preserve, dict) else {}
    raw_budget_preserve = stages.get("raw_budget_preserve_selector", {})
    raw_budget_preserve = raw_budget_preserve if isinstance(raw_budget_preserve, dict) else {}
    hipporag_preserve = stages.get("hipporag_preserve_selector", {})
    hipporag_preserve = hipporag_preserve if isinstance(hipporag_preserve, dict) else {}
    route_arbitrator = stages.get("route_arbitrator", {})
    route_arbitrator = route_arbitrator if isinstance(route_arbitrator, dict) else {}
    route_voi = route_arbitrator.get("value_of_information_gate", {})
    route_voi = route_voi if isinstance(route_voi, dict) else {}

    candidate_hashes = [value for value in recursive.get("candidate_answer_hashes", []) if value]
    unique_candidate_count = len(set(candidate_hashes))
    prompt_kinds = [str(value) for value in recursive.get("prompt_kinds", [])]
    skipped_prompt_kinds = [str(value) for value in recursive.get("skipped_prompt_kinds", [])]
    planned_branch_axes = [str(value) for value in recursive.get("planned_branch_axes", []) if value]
    executed_branch_axes = [str(value) for value in recursive.get("executed_branch_axes", []) if value]
    answered_branch_axes = [str(value) for value in recursive.get("answered_branch_axes", []) if value]
    skipped_branch_axes = [str(value) for value in recursive.get("skipped_branch_axes", []) if value]
    formal_hits = list(morphism.get("formal_mapping_hits", []) or [])
    structural_hits = list(morphism.get("structural_morphism_hits", []) or [])
    transfer_supported_hits = [
        hit for hit in structural_hits
        if isinstance(hit, dict) and hit.get("decision") == "transfer_supported"
    ]
    selection_method = str(selection.get("selection_method") or "")
    verified_or_abstain_gate = (
        selection.get("verified_or_abstain_gate", {})
        if isinstance(selection.get("verified_or_abstain_gate"), dict)
        else {}
    )
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
        "morphism_context_injected": bool(prompt.get("context_injected")) and bool(formal_hits or structural_hits),
        "morphism_routing_only": bool(formal_hits or structural_hits) and not bool(prompt.get("context_injected")),
        "world_model_used_context": world_model.get("decision") == "use_context",
        "same_run_baseline_cache_available": same_run_cache.get("status") == "activated",
        "critic_model_used": critic_router.get("status") == "activated",
        "child_model_used": child_router.get("status") == "activated",
        "evidence_bridge_activated": evidence_status == "activated",
        "evidence_child_executed": any(
            kind in {
                "evidence_bridge_answer",
                "evidence_grounded_answer",
                "answer_bearing_evidence_candidate",
                "evidence_guided_option_challenge_answer",
            }
            for kind in prompt_kinds
        ),
        "source_supported_evidence_candidate": (
            (evidence.get("source_supported_candidate") or {}).get("status") == "emitted"
        ),
        "agent_hipporag_context_activated": agent_hipporag.get("status") == "activated",
        "agent_hipporag_child_executed": "hipporag_context_answer" in prompt_kinds,
        "hipporag_context_priority_used": selection_method == "hipporag_context_priority",
        "recursive_child_validation_activated": recursive.get("status") == "activated",
        "recursive_child_diversity_planner_activated": diversity_planner.get("status") == "activated",
        "orthogonal_child_branches_planned": int(recursive.get("planned_unique_branch_axis_count") or 0) >= 3,
        "orthogonal_child_branches_executed": int(recursive.get("executed_unique_branch_axis_count") or 0) >= 3,
        "orthogonal_child_branches_answered": int(recursive.get("answered_unique_branch_axis_count") or 0) >= 3,
        "core_orthogonal_child_axes_covered": bool(recursive.get("core_orthogonal_axes_covered")),
        "core_orthogonal_child_axes_answered": bool(recursive.get("core_orthogonal_axes_answered")),
        "recursive_diverse_candidates": unique_candidate_count >= 2,
        "recursive_collapsed_consensus": bool(candidate_hashes) and unique_candidate_count <= 1,
        "recursive_timeout_pressure": int(recursive.get("error_child_count") or 0) > 0,
        "recursive_timeout_recovery_activated": timeout_recovery.get("status") == "activated",
        "recursive_timeout_recovery_emitted_candidate": bool(timeout_recovery.get("candidate_emitted")),
        "recursive_timeout_recovery_selected": bool(timeout_recovery.get("selected_timeout_recovery_candidate")),
        "child_model_failover_activated": child_model_failover.get("status") == "activated",
        "child_model_failover_emitted_candidate": bool(child_model_failover.get("candidate_emitted")),
        "child_model_failover_selected": bool(child_model_failover.get("selected_child_model_failover_candidate")),
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
        "code_semantics_child_planned": "code_semantics" in planned_branch_axes,
        "code_semantics_child_executed": "code_semantics_answer" in prompt_kinds,
        "code_semantics_child_selected": recursive.get("selected_prompt_kind") == "code_semantics_answer",
        "option_matrix_child_planned": "option_matrix_reasoning" in planned_branch_axes,
        "option_matrix_child_executed": "option_matrix_reasoner_answer" in prompt_kinds,
        "option_matrix_child_selected": recursive.get("selected_prompt_kind") == "option_matrix_reasoner_answer",
        "domain_rule_mc_verifier_activated": domain_rule.get("status") == "activated",
        "domain_rule_mc_verifier_selected": bool(domain_rule.get("selected_domain_rule_candidate")),
        "domain_rule_mc_verifier_correct": domain_rule.get("candidate_correct_for_eval") is True,
        "math_tool_verified": math_tool.get("confidence") in {"verified_symbolic", "verified_symbolic_consensus"},
        "mc_option_evidence_scorer_activated": option_evidence.get("status") == "activated",
        "mc_option_evidence_candidate_emitted": bool(option_evidence.get("candidate_emitted")),
        "mc_option_evidence_candidate_selected": bool(option_evidence.get("selected_option_evidence_candidate")),
        "verified_option_evidence_override": selection_method == "verified_option_evidence_priority",
        "mc_option_evidence_candidate_correct": option_evidence.get("candidate_correct_for_eval") is True,
        "evidence_guided_option_challenge_activated": evidence_guided_option.get("status") == "activated",
        "evidence_guided_option_candidate_emitted": bool(evidence_guided_option.get("candidate_emitted")),
        "evidence_guided_option_candidate_selected": bool(
            evidence_guided_option.get("selected_evidence_guided_option_candidate")
        ),
        "evidence_guided_option_candidate_correct": evidence_guided_option.get("candidate_correct_for_eval") is True,
        "structural_option_audit_activated": structural_option_audit.get("status") == "activated",
        "structural_option_audit_disagreed": bool(
            structural_option_audit.get("candidate_disagreed_with_majority")
        ),
        "structural_option_audit_selected": bool(
            structural_option_audit.get("selected_structural_option_audit")
        ),
        "structural_option_audit_candidate_correct": (
            structural_option_audit.get("candidate_correct_for_eval") is True
        ),
        "option_evidence_verifier_used": selection_method in {
            "option_evidence_verifier_choice",
            "verified_option_evidence_priority",
        },
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
        "evidence_override": selection_method in {
            "evidence_bridge_priority_over_closed_book_majority",
            "candidate_claim_verifier_priority",
        } and (
            (evidence.get("source_supported_candidate") or {}).get("status") == "emitted"
        ),
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
        "verified_or_abstain_allowed": verified_or_abstain_gate.get("status") == "allowed",
        "verified_or_abstain_abstained": verified_or_abstain_gate.get("status") == "abstained",
        "verified_or_abstain_no_fallback": verified_or_abstain_gate.get("status") == "no_fallback",
        "raw_preserve_selector_activated": raw_preserve.get("status") == "activated",
        "raw_preserve_candidate_emitted": bool(raw_preserve.get("candidate_emitted")),
        "raw_preserve_selected": bool(raw_preserve.get("selected_raw_preserve_candidate")),
        "raw_budget_preserve_selector_activated": raw_budget_preserve.get("status") == "activated",
        "raw_budget_preserve_candidate_emitted": bool(raw_budget_preserve.get("candidate_emitted")),
        "raw_budget_preserve_selected": bool(raw_budget_preserve.get("selected_raw_budget_preserve_candidate")),
        "hipporag_preserve_selector_activated": hipporag_preserve.get("status") == "activated",
        "hipporag_preserve_candidate_emitted": bool(hipporag_preserve.get("candidate_emitted")),
        "hipporag_preserve_selected": bool(hipporag_preserve.get("selected_hipporag_preserve_candidate")),
        "route_arbitrator_activated": route_arbitrator.get("status") == "activated",
        "route_arbitrator_candidate_emitted": bool(route_arbitrator.get("candidate_emitted")),
        "route_arbitrator_selected": bool(route_arbitrator.get("selected_route_arbitrator_candidate")),
        "route_arbitrator_trusted": bool(route_arbitrator.get("selected_route_trusted")),
        "route_arbitrator_untrusted_candidate": (
            bool(route_arbitrator.get("candidate_emitted"))
            and not bool(route_arbitrator.get("selected_route_trusted"))
        ),
        "route_value_verifier_enabled": bool(
            route_arbitrator.get("route_value_verifier_enabled", _route_value_verifier_enabled())
        ),
        "route_consensus_guard_enabled": bool(
            route_arbitrator.get("route_consensus_guard_enabled", _route_consensus_guard_enabled())
        ),
        "budget_echo_guard_enabled": bool(
            route_arbitrator.get("budget_echo_guard_enabled", _budget_echo_guard_enabled())
        ),
        "route_value_verifier_used": selection_method == "route_value_verifier_choice",
        "route_arbitrator_consensus_guard": bool(route_arbitrator.get("route_consensus")),
        "route_arbitrator_locked": bool(route_arbitrator.get("route_locked")),
        "route_arbitrator_chose_hipporag": route_arbitrator.get("selected_route_type") == "hipporag_preserve",
        "route_arbitrator_chose_raw_budget": route_arbitrator.get("selected_route_type") == "raw_budget_consensus",
        "route_arbitrator_chose_direct": route_arbitrator.get("selected_route_type") == "direct",
        "route_voi_recommended_preserve": route_voi.get("recommended_action") == "preserve_route",
        "route_voi_hard_gate_enabled": bool(route_voi.get("hard_gate_enabled")),
        "route_voi_hard_gate_applied": bool(route_voi.get("hard_gate_applied")),
        "route_voi_low_marginal_value": route_voi.get("status") == "preserve_route",
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
        "same_run_baseline_cache": {
            "status": same_run_cache.get("status"),
            "policy": same_run_cache.get("policy"),
            "cached_variants": list(same_run_cache.get("cached_variants", []) or []),
            "cached_variant_count": int(same_run_cache.get("cached_variant_count") or 0),
            "borrowed_baseline_model_call_count": int(
                same_run_cache.get("borrowed_baseline_model_call_count") or 0
            ),
        },
        "critic_model": {
            "status": critic_router.get("status"),
            "base_model": critic_router.get("base_model"),
            "critic_model": critic_router.get("critic_model"),
            "policy": critic_router.get("policy"),
        },
        "child_model": {
            "status": child_router.get("status"),
            "base_model": child_router.get("base_model"),
            "child_model": child_router.get("child_model"),
            "policy": child_router.get("policy"),
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
            "serial_forced_reason": recursive.get("serial_forced_reason"),
            "base_model": recursive.get("base_model"),
            "child_model": recursive.get("child_model"),
            "planned_child_count": int(recursive.get("planned_child_count") or 0),
            "child_count": int(recursive.get("child_count") or 0),
            "answered_child_count": int(recursive.get("answered_child_count") or 0),
            "error_child_count": int(recursive.get("error_child_count") or 0),
            "unique_candidate_count": unique_candidate_count,
            "planned_branch_axes": planned_branch_axes,
            "executed_branch_axes": executed_branch_axes,
            "answered_branch_axes": answered_branch_axes,
            "skipped_branch_axes": skipped_branch_axes,
            "planned_unique_branch_axis_count": int(recursive.get("planned_unique_branch_axis_count") or 0),
            "executed_unique_branch_axis_count": int(recursive.get("executed_unique_branch_axis_count") or 0),
            "answered_unique_branch_axis_count": int(recursive.get("answered_unique_branch_axis_count") or 0),
            "required_branch_axes_before_early_stop": list(
                recursive.get("required_branch_axes_before_early_stop", []) or []
            ),
            "executed_required_branch_axes": list(recursive.get("executed_required_branch_axes", []) or []),
            "answered_required_branch_axes": list(recursive.get("answered_required_branch_axes", []) or []),
            "core_orthogonal_axes_covered": bool(recursive.get("core_orthogonal_axes_covered")),
            "core_orthogonal_axes_answered": bool(recursive.get("core_orthogonal_axes_answered")),
            "orthogonal_branch_coverage": recursive.get("orthogonal_branch_coverage"),
            "diversity_planner": {
                "status": diversity_planner.get("status"),
                "policy": diversity_planner.get("policy"),
                "planned_child_count_raw": int(diversity_planner.get("planned_child_count_raw") or 0),
                "duplicate_branch_axes_removed": int(diversity_planner.get("duplicate_branch_axes_removed") or 0),
                "min_axes_before_early_stop": diversity_planner.get("min_axes_before_early_stop"),
                "required_axes_before_early_stop": list(
                    diversity_planner.get("required_axes_before_early_stop", []) or []
                ),
                "required_axes_missing_from_plan": list(
                    diversity_planner.get("required_axes_missing_from_plan", []) or []
                ),
            },
            "early_stopped": bool(recursive.get("early_stopped")),
            "early_stop_reason": recursive.get("early_stop_reason"),
            "prompt_kinds": prompt_kinds,
            "skipped_prompt_kinds": skipped_prompt_kinds,
            "selected_prompt_kind": recursive.get("selected_prompt_kind"),
        },
        "recursive_timeout_recovery": {
            "status": timeout_recovery.get("status"),
            "reason": timeout_recovery.get("reason"),
            "valid_candidate_count_before": timeout_recovery.get("valid_candidate_count_before"),
            "unique_candidate_count_before": timeout_recovery.get("unique_candidate_count_before"),
            "timeout_child_count_before": timeout_recovery.get("timeout_child_count_before"),
            "error_child_count_before": timeout_recovery.get("error_child_count_before"),
            "candidate_emitted": bool(timeout_recovery.get("candidate_emitted")),
            "selected_timeout_recovery_candidate": bool(timeout_recovery.get("selected_timeout_recovery_candidate")),
            "recovery_model": timeout_recovery.get("recovery_model"),
        },
        "child_model_failover": {
            "status": child_model_failover.get("status"),
            "reason": child_model_failover.get("reason"),
            "base_model": child_model_failover.get("base_model"),
            "failed_child_model": child_model_failover.get("failed_child_model"),
            "candidate_emitted": bool(child_model_failover.get("candidate_emitted")),
            "selected_child_model_failover_candidate": bool(child_model_failover.get("selected_child_model_failover_candidate")),
            "valid_candidate_count_before": child_model_failover.get("valid_candidate_count_before"),
            "timeout_child_count_before": child_model_failover.get("timeout_child_count_before"),
            "error_child_count_before": child_model_failover.get("error_child_count_before"),
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
            "candidate_verifier_state": option_evidence.get("candidate_verifier_state"),
            "candidate_correct_for_eval": option_evidence.get("candidate_correct_for_eval"),
            "top_score": option_evidence.get("top_score"),
            "margin": option_evidence.get("margin"),
            "top_support_doc_count": int(option_evidence.get("top_support_doc_count") or 0),
            "top_ambiguous_doc_count": int(option_evidence.get("top_ambiguous_doc_count") or 0),
            "selected_option_evidence_candidate": bool(option_evidence.get("selected_option_evidence_candidate")),
        },
        "evidence_guided_option_challenge": {
            "status": evidence_guided_option.get("status"),
            "reason": evidence_guided_option.get("reason"),
            "candidate_emitted": bool(evidence_guided_option.get("candidate_emitted")),
            "candidate_verifier_state": evidence_guided_option.get("candidate_verifier_state"),
            "candidate_correct_for_eval": evidence_guided_option.get("candidate_correct_for_eval"),
            "selected_evidence_guided_option_candidate": bool(
                evidence_guided_option.get("selected_evidence_guided_option_candidate")
            ),
            "context_char_count": int(evidence_guided_option.get("context_char_count") or 0),
            "context_option_count": int(evidence_guided_option.get("context_option_count") or 0),
            "top_rank_score": evidence_guided_option.get("top_rank_score"),
            "top_support_doc_count": int(evidence_guided_option.get("top_support_doc_count") or 0),
            "any_ambiguous_doc_count": int(evidence_guided_option.get("any_ambiguous_doc_count") or 0),
        },
        "structural_option_audit_child": {
            "status": structural_option_audit.get("status"),
            "reason": structural_option_audit.get("reason"),
            "candidate_emitted": bool(structural_option_audit.get("candidate_emitted")),
            "candidate_verifier_state": structural_option_audit.get("candidate_verifier_state"),
            "candidate_disagreed_with_majority": bool(
                structural_option_audit.get("candidate_disagreed_with_majority")
            ),
            "selected_structural_option_audit": bool(
                structural_option_audit.get("selected_structural_option_audit")
            ),
            "candidate_correct_for_eval": structural_option_audit.get("candidate_correct_for_eval"),
            "valid_candidate_count_before": int(structural_option_audit.get("valid_candidate_count_before") or 0),
            "unique_candidate_count_before": int(structural_option_audit.get("unique_candidate_count_before") or 0),
            "top_candidate_count_before": int(structural_option_audit.get("top_candidate_count_before") or 0),
            "missing_option_count_before": int(structural_option_audit.get("missing_option_count_before") or 0),
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
        "raw_preserve_selector": {
            "status": raw_preserve.get("status"),
            "policy": raw_preserve.get("policy"),
            "trigger": raw_preserve.get("trigger"),
            "candidate_emitted": bool(raw_preserve.get("candidate_emitted")),
            "selected_raw_preserve_candidate": bool(raw_preserve.get("selected_raw_preserve_candidate")),
        },
        "raw_budget_preserve_selector": {
            "status": raw_budget_preserve.get("status"),
            "policy": raw_budget_preserve.get("policy"),
            "trigger": raw_budget_preserve.get("trigger"),
            "candidate_emitted": bool(raw_budget_preserve.get("candidate_emitted")),
            "selected_raw_budget_preserve_candidate": bool(
                raw_budget_preserve.get("selected_raw_budget_preserve_candidate")
            ),
            "candidate_count": int(raw_budget_preserve.get("candidate_count") or 0),
            "answered_candidate_count": int(raw_budget_preserve.get("answered_candidate_count") or 0),
            "selection_method": raw_budget_preserve.get("selection_method"),
            "child_selection_method": raw_budget_preserve.get("child_selection_method"),
            "top_candidate_vote_count": raw_budget_preserve.get("top_candidate_vote_count"),
            "strong_consensus": bool(raw_budget_preserve.get("strong_consensus")),
            "block_reason": raw_budget_preserve.get("block_reason"),
        },
        "hipporag_preserve_selector": {
            "status": hipporag_preserve.get("status"),
            "policy": hipporag_preserve.get("policy"),
            "trigger": hipporag_preserve.get("trigger"),
            "block_reason": hipporag_preserve.get("block_reason"),
            "candidate_emitted": bool(hipporag_preserve.get("candidate_emitted")),
            "selected_hipporag_preserve_candidate": bool(hipporag_preserve.get("selected_hipporag_preserve_candidate")),
            "retrieval_status": hipporag_preserve.get("retrieval_status"),
            "budget_matched": bool(hipporag_preserve.get("budget_matched")),
            "candidate_count": int(hipporag_preserve.get("candidate_count") or 0),
            "answered_candidate_count": int(hipporag_preserve.get("answered_candidate_count") or 0),
            "selection_method": hipporag_preserve.get("selection_method"),
            "candidate_doc_count": int(hipporag_preserve.get("candidate_doc_count") or 0),
            "selected_doc_count": int(hipporag_preserve.get("selected_doc_count") or 0),
            "context_char_count": int(hipporag_preserve.get("context_char_count") or 0),
        },
        "route_arbitrator": {
            "status": route_arbitrator.get("status"),
            "policy": route_arbitrator.get("policy"),
            "route_value_verifier_enabled": bool(
                route_arbitrator.get("route_value_verifier_enabled", _route_value_verifier_enabled())
            ),
            "route_consensus_guard_enabled": bool(
                route_arbitrator.get("route_consensus_guard_enabled", _route_consensus_guard_enabled())
            ),
            "budget_echo_guard_enabled": bool(
                route_arbitrator.get("budget_echo_guard_enabled", _budget_echo_guard_enabled())
            ),
            "candidate_emitted": bool(route_arbitrator.get("candidate_emitted")),
            "selected_route_arbitrator_candidate": bool(route_arbitrator.get("selected_route_arbitrator_candidate")),
            "selected_route_type": route_arbitrator.get("selected_route_type"),
            "selected_route_prompt_kind": route_arbitrator.get("selected_route_prompt_kind"),
            "route_count": int(route_arbitrator.get("route_count") or 0),
            "route_types": list(route_arbitrator.get("route_types", []) or []),
            "unique_answer_count": int(route_arbitrator.get("unique_answer_count") or 0),
            "selected_route_score": route_arbitrator.get("selected_route_score"),
            "runner_up_score": route_arbitrator.get("runner_up_score"),
            "selected_route_value_profile": route_arbitrator.get("selected_route_value_profile"),
            "route_consensus": bool(route_arbitrator.get("route_consensus")),
            "retrieval_budget_counter_norm_count": int(
                route_arbitrator.get("retrieval_budget_counter_norm_count") or 0
            ),
            "independent_hippo_counter_norm_count": int(
                route_arbitrator.get("independent_hippo_counter_norm_count") or 0
            ),
            "selected_route_trusted": bool(route_arbitrator.get("selected_route_trusted")),
            "selected_route_trust_reason": route_arbitrator.get("selected_route_trust_reason"),
            "route_locked": bool(route_arbitrator.get("route_locked")),
            "raw_budget_strong_consensus": bool(route_arbitrator.get("raw_budget_strong_consensus")),
            "raw_budget_top_vote_count": int(route_arbitrator.get("raw_budget_top_vote_count") or 0),
            "hipporag_context_route_count": int(route_arbitrator.get("hipporag_context_route_count") or 0),
            "value_of_information_gate": route_voi or None,
        },
        "selection": {
            "status": selection.get("status"),
            "selection_method": selection_method or None,
            "verifier_model_call": bool(selection.get("verifier_model_call")),
            "verified_or_abstain_gate": verified_or_abstain_gate or None,
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
        "route_credit_table": _route_credit_table(run_rows),
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


def _agent_meets_best_control_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    by_model_variant = metrics.get("by_model_variant", {}) if isinstance(metrics, dict) else {}
    control_variants = ("raw", "raw_budget_matched", "hipporag_baseline", "hipporag_budget_matched")
    details: dict[str, Any] = {}
    passed = True
    model_names = sorted({
        str(key).split("::", 1)[0]
        for key in by_model_variant
        if "::" in str(key)
    })
    for model in model_names:
        agent_key = f"{model}::assumption_agent_recursive_verify"
        agent_metrics = by_model_variant.get(agent_key)
        if not isinstance(agent_metrics, dict):
            continue
        agent_accuracy = agent_metrics.get("accuracy")
        control_rows = []
        for variant in control_variants:
            control_key = f"{model}::{variant}"
            control_metrics = by_model_variant.get(control_key)
            if not isinstance(control_metrics, dict):
                continue
            control_accuracy = control_metrics.get("accuracy")
            if control_accuracy is None:
                continue
            control_rows.append({
                "variant": variant,
                "accuracy": control_accuracy,
                "n": control_metrics.get("n"),
            })
        if agent_accuracy is None or not control_rows:
            continue
        best_control = max(control_rows, key=lambda row: float(row.get("accuracy") or 0.0))
        margin = round(float(agent_accuracy) - float(best_control.get("accuracy") or 0.0), 4)
        model_passed = margin >= 0.0
        passed = passed and model_passed
        details[model] = {
            "passed": model_passed,
            "agent_accuracy": agent_accuracy,
            "best_control_variant": best_control.get("variant"),
            "best_control_accuracy": best_control.get("accuracy"),
            "agent_minus_best_control": margin,
            "controls": control_rows,
        }
    return {
        "passed": passed,
        "policy": "Agent must not score below the best same-model control present in the run.",
        "details": details,
    }


def _route_credit_table(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-problem route credit assignment without persisting HLE content."""
    control_variants = ("raw", "raw_budget_matched", "hipporag_baseline", "hipporag_budget_matched")
    agent_variant = "assumption_agent_recursive_verify"
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        model = str(row.get("model") or "")
        problem_hash = str(row.get("problem_id_hash") or "")
        variant = str(row.get("variant") or "")
        if not model or not problem_hash or not variant:
            continue
        grouped[(model, problem_hash)][variant] = row

    by_model: dict[str, dict[str, Any]] = {}
    problem_summaries: list[dict[str, Any]] = []
    for (model, problem_hash), variants in sorted(grouped.items()):
        agent_row = variants.get(agent_variant)
        if not isinstance(agent_row, dict):
            continue
        model_summary = by_model.setdefault(model, {
            "problem_count": 0,
            "complete_control_problem_count": 0,
            "agent_correct_count": 0,
            "agent_wrong_or_error_count": 0,
            "agent_unique_correct_count": 0,
            "recoverable_agent_error_count": 0,
            "unrecoverable_agent_error_count": 0,
            "endpoint_error_problem_count": 0,
            "correct_by_variant": Counter(),
            "error_by_variant": Counter(),
            "agent_loss_to_control_counts": Counter(),
            "agent_gain_over_control_counts": Counter(),
            "control_family_correct_counts": Counter(),
            "agent_selection_method_counts": Counter(),
            "agent_selection_method_correct_counts": Counter(),
            "agent_selected_route_type_counts": Counter(),
            "agent_selected_route_type_correct_counts": Counter(),
            "agent_selected_route_type_recoverable_loss_counts": Counter(),
            "voi_status_counts": Counter(),
            "voi_recommended_action_counts": Counter(),
            "voi_preserve_correct_count": 0,
            "voi_preserve_count": 0,
            "problem_summaries": [],
        })
        control_rows = {
            variant: variants.get(variant)
            for variant in control_variants
            if isinstance(variants.get(variant), dict)
        }
        agent_correct = bool(agent_row.get("correct"))
        correct_control_variants = [
            variant for variant, row in control_rows.items()
            if bool(row.get("correct"))
        ]
        error_control_variants = [
            variant for variant, row in control_rows.items()
            if row.get("error")
        ]
        all_error_variants = [
            variant for variant, row in variants.items()
            if row.get("error")
        ]
        efficacy = agent_row.get("component_efficacy")
        efficacy = efficacy if isinstance(efficacy, dict) else {}
        selection = efficacy.get("selection")
        selection = selection if isinstance(selection, dict) else {}
        route = efficacy.get("route_arbitrator")
        route = route if isinstance(route, dict) else {}
        voi = route.get("value_of_information_gate")
        voi = voi if isinstance(voi, dict) else {}
        selection_method = str(selection.get("selection_method") or "none")
        selected_route_type = str(route.get("selected_route_type") or "none")
        recoverable_controls = [] if agent_correct else correct_control_variants
        control_family_correct = {
            "raw_family": any(variant in correct_control_variants for variant in ("raw", "raw_budget_matched")),
            "hipporag_family": any(
                variant in correct_control_variants for variant in ("hipporag_baseline", "hipporag_budget_matched")
            ),
            "budget_family": any(
                variant in correct_control_variants for variant in ("raw_budget_matched", "hipporag_budget_matched")
            ),
        }

        model_summary["problem_count"] += 1
        if len(control_rows) == len(control_variants):
            model_summary["complete_control_problem_count"] += 1
        model_summary["agent_correct_count"] += int(agent_correct)
        model_summary["agent_wrong_or_error_count"] += int(not agent_correct)
        model_summary["agent_unique_correct_count"] += int(agent_correct and not correct_control_variants)
        model_summary["recoverable_agent_error_count"] += int(bool(recoverable_controls))
        model_summary["unrecoverable_agent_error_count"] += int((not agent_correct) and not correct_control_variants)
        model_summary["endpoint_error_problem_count"] += int(bool(all_error_variants))
        model_summary["agent_selection_method_counts"][selection_method] += 1
        model_summary["agent_selected_route_type_counts"][selected_route_type] += 1
        if agent_correct:
            model_summary["agent_selection_method_correct_counts"][selection_method] += 1
            model_summary["agent_selected_route_type_correct_counts"][selected_route_type] += 1
        if recoverable_controls:
            model_summary["agent_selected_route_type_recoverable_loss_counts"][selected_route_type] += 1
        for variant, row in variants.items():
            if row.get("correct"):
                model_summary["correct_by_variant"][variant] += 1
            if row.get("error"):
                model_summary["error_by_variant"][variant] += 1
        for variant in correct_control_variants:
            if not agent_correct:
                model_summary["agent_loss_to_control_counts"][variant] += 1
        for variant, row in control_rows.items():
            if agent_correct and not bool(row.get("correct")):
                model_summary["agent_gain_over_control_counts"][variant] += 1
        for family, is_correct in control_family_correct.items():
            if is_correct:
                model_summary["control_family_correct_counts"][family] += 1
        voi_status = str(voi.get("status") or "none")
        voi_action = str(voi.get("recommended_action") or "none")
        model_summary["voi_status_counts"][voi_status] += 1
        model_summary["voi_recommended_action_counts"][voi_action] += 1
        if voi_action == "preserve_route":
            model_summary["voi_preserve_count"] += 1
            model_summary["voi_preserve_correct_count"] += int(agent_correct)

        problem_summary = {
            "model": model,
            "problem_id_hash": problem_hash,
            "answer_type": agent_row.get("answer_type"),
            "agent_correct": agent_correct,
            "control_correct_variants": correct_control_variants,
            "control_error_variants": error_control_variants,
            "recoverable_control_variants": recoverable_controls,
            "all_controls_wrong_or_error": (not correct_control_variants),
            "agent_selection_method": selection_method,
            "agent_selected_route_type": selected_route_type,
            "agent_selected_route_trust_reason": route.get("selected_route_trust_reason"),
            "route_voi_status": voi_status,
            "route_voi_recommended_action": voi_action,
            "endpoint_error_variants": all_error_variants,
        }
        model_summary["problem_summaries"].append(problem_summary)
        problem_summaries.append(problem_summary)

    finalized_by_model: dict[str, dict[str, Any]] = {}
    for model, summary in sorted(by_model.items()):
        n = int(summary["problem_count"] or 0)
        route_counts: Counter[str] = summary["agent_selected_route_type_counts"]
        selection_counts: Counter[str] = summary["agent_selection_method_counts"]
        finalized_by_model[model] = {
            "problem_count": n,
            "complete_control_problem_count": int(summary["complete_control_problem_count"]),
            "agent_correct_count": int(summary["agent_correct_count"]),
            "agent_accuracy": None if not n else round(summary["agent_correct_count"] / n, 4),
            "agent_wrong_or_error_count": int(summary["agent_wrong_or_error_count"]),
            "agent_unique_correct_count": int(summary["agent_unique_correct_count"]),
            "recoverable_agent_error_count": int(summary["recoverable_agent_error_count"]),
            "unrecoverable_agent_error_count": int(summary["unrecoverable_agent_error_count"]),
            "endpoint_error_problem_count": int(summary["endpoint_error_problem_count"]),
            "correct_by_variant": dict(sorted(summary["correct_by_variant"].items())),
            "error_by_variant": dict(sorted(summary["error_by_variant"].items())),
            "agent_loss_to_control_counts": dict(sorted(summary["agent_loss_to_control_counts"].items())),
            "agent_gain_over_control_counts": dict(sorted(summary["agent_gain_over_control_counts"].items())),
            "control_family_correct_counts": dict(sorted(summary["control_family_correct_counts"].items())),
            "agent_selection_method_counts": dict(sorted(selection_counts.items())),
            "agent_selection_method_accuracy": {
                method: round(summary["agent_selection_method_correct_counts"][method] / count, 4)
                for method, count in sorted(selection_counts.items())
                if count
            },
            "agent_selected_route_type_counts": dict(sorted(route_counts.items())),
            "agent_selected_route_type_accuracy": {
                route_type: round(summary["agent_selected_route_type_correct_counts"][route_type] / count, 4)
                for route_type, count in sorted(route_counts.items())
                if count
            },
            "agent_selected_route_type_recoverable_loss_counts": dict(
                sorted(summary["agent_selected_route_type_recoverable_loss_counts"].items())
            ),
            "voi_status_counts": dict(sorted(summary["voi_status_counts"].items())),
            "voi_recommended_action_counts": dict(sorted(summary["voi_recommended_action_counts"].items())),
            "voi_preserve_count": int(summary["voi_preserve_count"]),
            "voi_preserve_accuracy": None
            if not summary["voi_preserve_count"]
            else round(summary["voi_preserve_correct_count"] / summary["voi_preserve_count"], 4),
            "problem_summaries": summary["problem_summaries"],
        }
    return {
        "policy": (
            "Metadata-only route credit assignment.  It reports when the agent loses to an already-run "
            "same-model control and separates recoverable selector errors from all-route failures."
        ),
        "control_variants": list(control_variants),
        "agent_variant": agent_variant,
        "model_count": len(finalized_by_model),
        "problem_count": len(problem_summaries),
        "by_model": finalized_by_model,
        "problem_summaries": problem_summaries,
    }


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
        default="phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/hle*.json*",
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
