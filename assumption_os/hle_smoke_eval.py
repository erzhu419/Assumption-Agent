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
import html
import json
import os
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
) -> dict[str, Any]:
    root = root.resolve()
    models = models or ["gpt-5.5"]
    variants = variants or ["raw", "assumption_wrapper"]
    graph_dir = graph_dir or (root / DEFAULT_GRAPH_DIR)
    access = _access_preflight()
    sample_rows: list[dict[str, Any]] = []
    if access["dataset_accessible"]:
        sample_rows = _load_text_only_sample(sample_size=sample_size, max_scan=max_scan, seed_offset=seed_offset)
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
                    module_trace = _module_trace(problem, variant=variant, agent_plan=agent_plan)
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
                                prompt=_prompt_for(problem, variant=variant, agent_plan=agent_plan),
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
                                "agent_plan_hash": stable_hash(agent_plan or {}),
                            },
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
                        run_rows.append(
                            _error_row(
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
                                    "agent_plan_hash": stable_hash(agent_plan or {}),
                                },
                            )
                        )
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


def _load_text_only_sample(*, sample_size: int, max_scan: int, seed_offset: int) -> list[dict[str, Any]]:
    from datasets import Image, load_dataset

    dataset = load_dataset(DATASET_NAME, split="test", streaming=True, token=_hf_token())
    dataset = _cast_image_columns(dataset, Image)
    sample: list[dict[str, Any]] = []
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
        sample.append(_problem_from_row(row, scanned=scanned, skipped_before=skipped))
        if len(sample) >= sample_size or scanned >= max_scan:
            break
    return sample


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
        router_summary = {
            "status": "activated",
            "decision": "use_context" if context_allowed else "abstain_to_raw_prompt",
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

        if context_allowed and retrieval_result is not None:
            context = format_policy_context(retrieval_result, format_assumption_context, max_nodes=top_k)
            plan["retrieval_context_candidate"] = _trim_context(context, max_chars=context_max_chars)
            plan["prompt_context"] = plan["retrieval_context_candidate"]
        elif retrieval_result is not None:
            context = format_policy_context(retrieval_result, format_assumption_context, max_nodes=top_k)
            plan["retrieval_context_candidate"] = _trim_context(context, max_chars=context_max_chars)
        plan["stages"]["prompt_builder"] = {
            "status": "activated",
            "context_injected": bool(plan["prompt_context"]),
            "retrieval_context_candidate_char_count": len(plan.get("retrieval_context_candidate", "")),
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
    evidence_summary: dict[str, Any] | None = None
    if evidence_bridge_enabled and _needs_evidence_grounded_child(problem, attempts):
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
                model=model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=child_timeout if child_timeout is not None else timeout,
                max_tokens=max_tokens,
            )
            attempts.append(evidence_attempt)
            if evidence_attempt.get("status") == "answered":
                underlying_calls += 1

    selection = _select_recursive_child_answer(
        problem=problem,
        attempts=attempts,
        model=model,
        eval_id=eval_id,
        call_id=call_id,
        logger=logger,
        timeout=timeout,
        max_tokens=min(max_tokens, 384),
    )
    underlying_calls += int(selection.get("underlying_model_calls", 0) or 0)
    selected_answer = selection.get("selected_answer") or _fallback_answer(attempts)
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
    if _has_two_vote_majority(attempts, answer_type=problem["answer_type"]):
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
        if _has_two_vote_majority(attempts, answer_type=problem["answer_type"]):
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_child_attempt,
                problem=problem,
                spec=spec,
                child_index=start_index + offset,
                model=model,
                eval_id=eval_id,
                call_id=call_id,
                logger=logger,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            for offset, spec in enumerate(specs)
        ]
        for future in concurrent.futures.as_completed(futures):
            attempts.append(future.result())
    attempts.sort(key=lambda row: int(row.get("child_index", 0) or 0))
    return {
        "attempts": attempts,
        "underlying_model_calls": sum(1 for attempt in attempts if attempt.get("status") == "answered"),
        "max_workers": max_workers,
    }


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
    context = (agent_plan or {}).get("prompt_context", "")
    if context:
        specs.append({
            "prompt_kind": "agent_context_answer",
            "prompt": (
                "A bounded Assumption Agent retrieved the following graph/morphism context. Use it only if it "
                "directly constrains the answer; otherwise ignore it. Return only JSON.\n\n"
                f"{context}\n\nAnswer type: {answer_type}\nQuestion:\n{question}\n\n{output}"
            ),
        })
    return specs


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
) -> dict[str, Any]:
    valid = [attempt for attempt in attempts if str(attempt.get("parsed_answer") or "").strip()]
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
        if len(ranked[0][1]) >= 2 or len(ranked) == 1:
            selected = ranked[0][1][0]
            return {
                "selection_method": "normalized_majority",
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
        verifier_text = _call_model(
            model=model,
            prompt=_verifier_prompt(problem, valid),
            timeout=timeout,
            max_tokens=max_tokens,
        )
        choice = _parse_verifier_choice(verifier_text, max_index=len(valid))
        selected = valid[(choice or 1) - 1]
        _log_event(
            logger,
            {
                "event": "recursive_verifier_end",
                "eval_id": eval_id,
                "call_id": call_id,
                "problem_id_hash": problem["id_hash"],
                "model": model,
                "variant": "assumption_agent_recursive_verify",
                "candidate_count": len(valid),
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


def _verifier_prompt(problem: dict[str, Any], attempts: list[dict[str, Any]]) -> str:
    choices = "\n".join(
        f"{index}. prompt_kind={attempt['prompt_kind']}; answer={attempt['parsed_answer']}"
        for index, attempt in enumerate(attempts, start=1)
    )
    return (
        "Choose the candidate answer most likely to satisfy the HLE question. Prefer exact wording, correct "
        "multiple-choice letter, and answers that do not add unsupported qualifiers. Return JSON only: "
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
    if problem.get("answer_type") == "multipleChoice":
        return False
    valid = [
        str(attempt.get("parsed_answer") or "").strip()
        for attempt in attempts
        if str(attempt.get("parsed_answer") or "").strip()
    ]
    return bool(valid) and all(_is_suspicious_exact_answer(answer) for answer in valid)


def _is_suspicious_exact_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return True
    if re.fullmatch(r"[A-Z]", text):
        return True
    if text.lower() in {"unknown", "none", "n/a", "na"}:
        return True
    return False


def _evidence_grounded_answer_prompt(problem: dict[str, Any], *, evidence_context: str) -> str:
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
            results.extend(_wikipedia_search(query, limit=2, timeout=6.0))
        except Exception as exc:
            errors.append(type(exc).__name__)
    unique_results = _dedupe_evidence_results(results)[:5]
    evidence_context = _format_evidence_context(unique_results, max_chars=1800)
    summary = {
        "status": "activated" if evidence_context else "no_results",
        "source": "wikipedia_search",
        "query_count": len(queries),
        "query_hashes": query_hashes,
        "result_count": len(unique_results),
        "source_hashes": [
            stable_hash({"title": row.get("title", ""), "snippet": row.get("snippet", "")})
            for row in unique_results
        ],
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
        lines.append(f"[Evidence {index}] source=wikipedia; title={title}; snippet={snippet}")
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
                "expected": False,
                "status": _stage_status(stages, "hle_evidence_bridge") if stages.get("hle_evidence_bridge") else "not_required",
                "reason": "exactMatch repair can use transient external evidence; logs persist only hashes and counts",
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
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if threading.current_thread() is threading.main_thread():
                previous_handler = signal.signal(signal.SIGALRM, _raise_wallclock_timeout)
                signal.alarm(max(1, int(timeout)))
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        data = json.loads(response.read().decode("utf-8"))
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous_handler)
            else:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            return str((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise RuntimeError(f"model request failed: {type(exc).__name__}") from exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"model request failed: {last_error}")


def _raise_wallclock_timeout(signum: int, frame: Any) -> None:
    raise TimeoutError("wall_clock_model_timeout")


def _default_call_timeout() -> float:
    return float(os.environ.get("MODEL_ROUTER_TIMEOUT", "120"))


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
    correct = _is_correct(predicted, problem["_answer"], answer_type=answer_type)
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
        "module_activation_summary": _module_activation_summary(run_rows),
        "expected_but_missing_modules": _expected_but_missing_modules(run_rows),
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
    for row in run_rows:
        key = f"{row['model']}::{row['variant']}"
        for item in row.get("module_trace", []):
            if item.get("expected") and item.get("status") != "activated":
                missing[key].add(item["module"])
    return {key: sorted(values) for key, values in sorted(missing.items())}


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
