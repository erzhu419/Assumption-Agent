"""Text-only smoke evaluation for Humanity's Last Exam.

The official HLE dataset is gated.  This runner expects the user to have
accepted the dataset terms and provided ``HF_TOKEN`` in the process environment.
It deliberately does not persist HLE questions, gold answers, rationales, or
canary strings.  Artifacts store stable hashes, metadata, predictions hashes,
and correctness only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import urllib.error
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
                    if variant == "assumption_agent":
                        agent_plan = _build_assumption_agent_plan(
                            root=root,
                            graph_dir=graph_dir,
                            problem=problem,
                            eval_id=eval_id,
                            call_id=call_id,
                            model=model,
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
                        answer_text = _call_model(
                            model=model,
                            prompt=_prompt_for(problem, variant=variant, agent_plan=agent_plan),
                            timeout=call_timeout,
                            max_tokens=max_tokens,
                        )
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
) -> dict[str, Any]:
    question = problem["_question"]
    goal = f"Solve a text-only HLE item with answer_type={problem['answer_type']} and return exact JSON."
    plan: dict[str, Any] = {
        "agent_kind": "hle_assumption_agent_v1",
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
    _agent_stage_log(logger, eval_id=eval_id, call_id=call_id, problem=problem, model=model, stage="domain_router", data=plan["stages"]["domain_router"])

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
            stage="world_model_router",
            data=router_summary,
        )

        if context_allowed and retrieval_result is not None:
            context = format_policy_context(retrieval_result, format_assumption_context, max_nodes=top_k)
            plan["prompt_context"] = _trim_context(context, max_chars=context_max_chars)
        plan["stages"]["prompt_builder"] = {
            "status": "activated",
            "context_injected": bool(plan["prompt_context"]),
            "context_char_count": len(plan["prompt_context"]),
        }
        _agent_stage_log(
            logger,
            eval_id=eval_id,
            call_id=call_id,
            problem=problem,
            model=model,
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
    if variant == "assumption_agent":
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
            "variant": "assumption_agent",
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
    if variant == "assumption_agent":
        stages = (agent_plan or {}).get("stages", {})
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
                "module": "multi_candidate_self_verifier",
                "expected": False,
                "status": "not_implemented_for_hle_single_call",
                "reason": "this HLE variant keeps one answer call; external multi-candidate verification would require extra model calls",
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
            previous_handler = signal.signal(signal.SIGALRM, _raise_wallclock_timeout)
            signal.alarm(max(1, int(timeout)))
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
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
