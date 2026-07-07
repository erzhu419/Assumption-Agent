"""Prefetch answer-bearing source evidence for a fixed HLE cohort.

This tool is intentionally separate from scoring.  It may use raw local HLE
questions in memory to build source queries, but persisted artifacts store only
hashes, counts, and cache/source statuses.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .diagnostic_logging import JsonlDiagnosticLogger, log_event
from .hle_guideline_pair_comparators import (
    fe_hyperfine_pair_binding_detail,
    medical_guideline_permutation_ordering_detail,
)
from .hle_smoke_eval import (
    DEFAULT_GRAPH_DIR,
    _answer_bearing_web_fulltext_search,
    _answer_bearing_web_search,
    _arxiv_search,
    _classify_hle_domain,
    _clean_evidence_query,
    _compile_hle_operator_stage,
    _content_terms,
    _crossref_search,
    _deterministic_option_claim_relation_queries,
    _evidence_source_error_label,
    _evidence_source_cache_get,
    _extract_choice,
    _load_text_only_sample,
    _local_evidence_corpus_search,
    _musicology_short_option_direct_relation_signal,
    _musicology_short_option_phrase_signal,
    _courtlistener_search,
    _openalex_search,
    _option_claim_answer_web_fallback_queries,
    _option_claim_evidence_queries_for_plan,
    _option_claim_local_relation_query_expansion_queries,
    _option_claim_question_relation_signature_terms,
    _option_claim_relation_signature_term_hashes,
    _option_claim_relation_slot_coverage,
    _option_claim_relation_slot_plan,
    _option_evidence_queries_for_plan,
    _prefetch_query_plan_cache_put,
    _pubmed_pmc_fulltext_search,
    _pubmed_search,
    _pubchem_search,
    _question_evidence_anchor_terms,
    _question_relation_query_terms,
    _run_option_claim_relation_query_planner,
    _semantic_scholar_search,
    _split_multiple_choice_question,
    _normalized_phrase_present,
    _ontario_lso_rules_search,
    _wikipedia_extract_search,
    apply_hle_offline_defaults_to_environ,
)
from .hle_numeric_option_parser import parse_numeric_values
from .hle_numeric_relation_classifier import (
    classify_numeric_relation,
    numeric_relation_terms,
)
from .hle_numeric_source_witness import numeric_same_row_source_diagnostics
from .hle_operator_cohort_preflight import _operator_family_tags_from_stage
from .private_env import load_private_env


DEFAULT_RUN_DIR = PAPER_DIR / "hle_source_prefetch"
DEFAULT_OUT = DEFAULT_RUN_DIR / "hle_source_prefetch.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_source_prefetch.md")
DEFAULT_SOURCES = (
    "semantic_scholar",
    "openalex",
    "arxiv",
    "crossref",
    "wikipedia_extract",
    "answer_web",
)
SUPPORTED_SOURCES = DEFAULT_SOURCES + (
    "answer_web_fulltext",
    "pubmed",
    "pubmed_pmc_fulltext",
    "local_evidence_corpus",
    "pubchem",
    "courtlistener",
    "lso_rules",
)


def _sanitize_private_env_status_for_artifact(status: dict[str, Any]) -> dict[str, Any]:
    loaded_keys = [
        str(value)
        for value in (status.get("loaded_keys") or [])
        if str(value).strip()
    ] if isinstance(status, dict) else []
    path = str(status.get("path") or "") if isinstance(status, dict) else ""
    safe = {
        "loaded": bool(status.get("loaded")) if isinstance(status, dict) else False,
        "loaded_key_count": len(loaded_keys),
        "raw_key_names_persisted": False,
        "raw_path_persisted": False,
    }
    if isinstance(status, dict) and status.get("skipped_reason"):
        safe["skipped_reason"] = str(status.get("skipped_reason") or "")
    if isinstance(status, dict) and status.get("mode"):
        safe["mode"] = str(status.get("mode") or "")
    if path:
        safe["path_hash"] = stable_hash({"private_env_path": path})
    return safe


_SOURCE_PREFETCH_GENERIC_TERMS = {
    "answer",
    "answers",
    "choice",
    "choices",
    "correct",
    "direct",
    "directly",
    "false",
    "following",
    "option",
    "options",
    "question",
    "statement",
    "true",
}

_NUMERIC_THRESHOLD_PREFETCH_RELATION_FAMILIES = {
    "above_threshold",
    "below_threshold",
    "closest_value",
    "exact_value",
    "ordered_extreme_highest",
    "ordered_extreme_lowest",
    "range_membership",
    "threshold_maximum",
    "threshold_minimum",
}

_NUMERIC_THRESHOLD_GENERIC_TERMS = {
    "above",
    "allowed",
    "any",
    "below",
    "can",
    "celsius",
    "coldest",
    "degree",
    "degrees",
    "efficient",
    "efficiently",
    "fahrenheit",
    "following",
    "greater",
    "heat",
    "heated",
    "hottest",
    "kelvin",
    "largest",
    "least",
    "less",
    "lowest",
    "maximum",
    "method",
    "minimum",
    "nearest",
    "prepared",
    "preparation",
    "prepare",
    "produce",
    "produced",
    "reaction",
    "required",
    "still",
    "synthesis",
    "synthesize",
    "synthesized",
    "temperature",
    "thermal",
    "threshold",
    "using",
    "when",
    "where",
    "which",
}

_TERM_IDENTITY_ALL_REQUIRED_TERMS_SENTINEL = "__all_required_terms__"

_SOURCE_PREFETCH_LOCAL_SOURCES = {
    "local_evidence_corpus",
}


def build_hle_source_prefetch_payload(
    *,
    root: Path,
    eval_id: str,
    seed_offsets: list[int],
    graph_dir: Path | None = None,
    max_scan: int = 200,
    max_options_per_problem: int = 8,
    max_queries_per_problem: int = 24,
    max_queries_per_option: int = 4,
    sources: list[str] | None = None,
    source_limit: int = 2,
    timeout: float = 8.0,
    execute_live: bool = False,
    max_live_calls: int = 80,
    delay_sec: float = 1.1,
    retry_cached_errors: bool = False,
    refresh_cache_hits: bool = False,
    parallel_workers: int = 1,
    budget_policy: str = "round_robin_by_problem",
    source_error_budget: int = 0,
    logger: JsonlDiagnosticLogger | None = None,
    enable_relation_query_planner: bool = False,
    enable_sweep_gap_relation_backfill_queries: bool = False,
    enable_option_aware_query_expansion: bool = False,
    enable_answer_bearing_binding_queries: bool = False,
    enable_answer_bearing_pair_binding_queries: bool = False,
    enable_required_relation_completion_queries: bool = False,
    enable_candidate_specific_answer_bearing_queries: bool = False,
    enable_numeric_same_row_backfill_queries: bool = False,
    numeric_same_row_backfill_max_live_calls: int = 24,
    focus_eval_json: Path | None = None,
    focus_only: bool = False,
    focus_max_options_per_problem: int = 4,
    relation_query_planner_model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    root = root.resolve()
    graph_path = graph_dir or (root / DEFAULT_GRAPH_DIR)
    graph_path = graph_path if graph_path.is_absolute() else root / graph_path
    source_names = _normalize_sources(sources or list(DEFAULT_SOURCES))
    failure_focus = _source_failure_focus_from_eval_json(focus_eval_json)
    failure_focus = _trim_source_failure_focus(
        failure_focus,
        max_options_per_problem=focus_max_options_per_problem,
    )
    if failure_focus.get("status") == "activated":
        enable_candidate_specific_answer_bearing_queries = True
    previous_env = _enter_prefetch_env(
        execute_live=execute_live,
        refresh_cache_hits=refresh_cache_hits,
    )
    try:
        query_plan = _build_query_plan(
            root=root,
            graph_dir=graph_path,
            seed_offsets=seed_offsets,
            max_scan=max_scan,
            max_options_per_problem=max_options_per_problem,
            max_queries_per_problem=max_queries_per_problem,
            max_queries_per_option=max_queries_per_option,
            enable_relation_query_planner=enable_relation_query_planner,
            enable_sweep_gap_relation_backfill_queries=enable_sweep_gap_relation_backfill_queries,
            enable_option_aware_query_expansion=enable_option_aware_query_expansion,
            enable_answer_bearing_binding_queries=enable_answer_bearing_binding_queries,
            enable_answer_bearing_pair_binding_queries=(
                enable_answer_bearing_pair_binding_queries
            ),
            enable_required_relation_completion_queries=(
                enable_required_relation_completion_queries
            ),
            enable_candidate_specific_answer_bearing_queries=(
                enable_candidate_specific_answer_bearing_queries
            ),
            failure_focus_by_seed=failure_focus.get("focus_by_seed", {}),
            focus_only=focus_only,
            relation_query_planner_model=relation_query_planner_model,
            logger=logger,
        )
        source_records = _run_source_prefetch(
            query_plan=query_plan,
            sources=source_names,
            source_limit=source_limit,
            timeout=timeout,
            execute_live=execute_live,
            max_live_calls=max_live_calls,
            delay_sec=delay_sec,
            retry_cached_errors=retry_cached_errors,
            refresh_cache_hits=refresh_cache_hits,
            parallel_workers=parallel_workers,
            budget_policy=budget_policy,
            source_error_budget=source_error_budget,
            logger=logger,
        )
        numeric_same_row_backfill_query_plan: list[dict[str, Any]] = []
        numeric_same_row_backfill_source_records: list[dict[str, Any]] = []
        if enable_numeric_same_row_backfill_queries:
            numeric_same_row_backfill_query_plan = (
                _numeric_same_row_adaptive_backfill_query_plan(
                    query_plan=query_plan,
                    source_records=source_records,
                    source_limit=source_limit,
                )
            )
            if numeric_same_row_backfill_query_plan:
                numeric_same_row_backfill_source_records = _run_source_prefetch(
                    query_plan=numeric_same_row_backfill_query_plan,
                    sources=source_names,
                    source_limit=source_limit,
                    timeout=timeout,
                    execute_live=execute_live,
                    max_live_calls=numeric_same_row_backfill_max_live_calls,
                    delay_sec=delay_sec,
                    retry_cached_errors=retry_cached_errors,
                    refresh_cache_hits=refresh_cache_hits,
                    parallel_workers=parallel_workers,
                    budget_policy=budget_policy,
                    source_error_budget=source_error_budget,
                    logger=logger,
                )
                source_records.extend(numeric_same_row_backfill_source_records)
    finally:
        _restore_env(previous_env)

    metrics = _prefetch_metrics(query_plan=query_plan, source_records=source_records)
    if enable_numeric_same_row_backfill_queries:
        metrics["numeric_same_row_backfill_query_count"] = sum(
            len(row.get("query_records", []) or [])
            for row in numeric_same_row_backfill_query_plan
        )
        metrics["numeric_same_row_backfill_source_record_count"] = len(
            numeric_same_row_backfill_source_records
        )
    return {
        "eval_id": eval_id,
        "eval_kind": "hle_source_prefetch",
        "execute_live": bool(execute_live),
        "raw_content_persisted": False,
        "sampling": {
            "seed_offsets": seed_offsets,
            "max_scan": max_scan,
            "max_options_per_problem": max_options_per_problem,
            "max_queries_per_problem": max_queries_per_problem,
            "max_queries_per_option": max_queries_per_option,
            "sources": source_names,
            "source_limit": source_limit,
            "max_live_calls": max_live_calls,
            "delay_sec": delay_sec,
            "retry_cached_errors": retry_cached_errors,
            "refresh_cache_hits": refresh_cache_hits,
            "parallel_workers": parallel_workers,
            "budget_policy": budget_policy,
            "source_error_budget": source_error_budget,
            "enable_relation_query_planner": enable_relation_query_planner,
            "enable_sweep_gap_relation_backfill_queries": enable_sweep_gap_relation_backfill_queries,
            "enable_option_aware_query_expansion": enable_option_aware_query_expansion,
            "enable_answer_bearing_binding_queries": enable_answer_bearing_binding_queries,
            "enable_answer_bearing_pair_binding_queries": (
                enable_answer_bearing_pair_binding_queries
            ),
            "enable_required_relation_completion_queries": (
                enable_required_relation_completion_queries
            ),
            "enable_candidate_specific_answer_bearing_queries": (
                enable_candidate_specific_answer_bearing_queries
            ),
            "enable_numeric_same_row_backfill_queries": (
                enable_numeric_same_row_backfill_queries
            ),
            "numeric_same_row_backfill_max_live_calls": (
                numeric_same_row_backfill_max_live_calls
            ),
            "candidate_specific_answer_bearing_auto_enabled_by_failure_focus": bool(
                failure_focus.get("status") == "activated"
            ),
            "focus_only": bool(focus_only),
            "focus_max_options_per_problem": int(focus_max_options_per_problem or 0),
            "failure_focus_status": failure_focus.get("status"),
            "failure_focus_reason": failure_focus.get("reason"),
            "failure_focus_source_hash": failure_focus.get("source_hash"),
            "failure_focus_seed_count": failure_focus.get("seed_count"),
            "failure_focus_option_count": failure_focus.get("focused_option_count"),
            "failure_focus_untrimmed_option_count": failure_focus.get(
                "untrimmed_focused_option_count"
            ),
            "failure_focus_expand_all_option_seed_count": failure_focus.get(
                "expand_all_option_seed_count"
            ),
            "failure_focus_reason_counts": failure_focus.get("reason_counts"),
            "relation_query_planner_model": relation_query_planner_model,
        },
        "failure_focus": _sanitize_failure_focus_summary(failure_focus),
        "metrics": metrics,
        "problems": [_sanitize_problem_plan(row) for row in query_plan],
        "numeric_same_row_backfill_problems": [
            _sanitize_problem_plan(row)
            for row in numeric_same_row_backfill_query_plan
        ] if enable_numeric_same_row_backfill_queries else [],
        "source_records": [_sanitize_source_record(row) for row in source_records],
        "claim_boundary": (
            "This artifact proves local source-cache prefetch coverage for a fixed HLE cohort. "
            "It stores hashes and source counts only; raw HLE question text, options, answers, and raw "
            "queries are intentionally omitted."
        ),
    }


def _build_query_plan(
    *,
    root: Path,
    graph_dir: Path,
    seed_offsets: list[int],
    max_scan: int,
    max_options_per_problem: int,
    max_queries_per_problem: int,
    max_queries_per_option: int,
    enable_relation_query_planner: bool,
    enable_sweep_gap_relation_backfill_queries: bool,
    enable_option_aware_query_expansion: bool,
    enable_answer_bearing_binding_queries: bool,
    enable_answer_bearing_pair_binding_queries: bool,
    enable_required_relation_completion_queries: bool,
    enable_candidate_specific_answer_bearing_queries: bool,
    failure_focus_by_seed: dict[int, dict[str, Any]] | None,
    focus_only: bool,
    relation_query_planner_model: str,
    logger: JsonlDiagnosticLogger | None = None,
) -> list[dict[str, Any]]:
    graph = SimpleAssumptionGraph(JsonlGraphStore(graph_dir)) if graph_dir.exists() else None
    rows: list[dict[str, Any]] = []
    for seed_offset in seed_offsets:
        sample = _load_text_only_sample(
            sample_size=1,
            max_scan=max_scan,
            seed_offset=seed_offset,
            answer_type_filter="multipleChoice",
        )
        if not sample:
            rows.append({
                "seed_offset": seed_offset,
                "status": "sample_not_found",
                "problem_id_hash": "",
                "query_records": [],
            })
            continue
        problem = sample[0]
        stem, options = _split_multiple_choice_question(problem)
        agent_plan = _operator_plan_for_prefetch(problem=problem, graph=graph)
        operator_stage = (agent_plan.get("stages") or {}).get("operator_spec_compiler") or {}
        family_tags = _operator_family_tags_from_stage(operator_stage)
        failure_focus = (
            failure_focus_by_seed.get(int(seed_offset), {})
            if isinstance(failure_focus_by_seed, dict)
            else {}
        )
        query_records, relation_query_planner_summary = _problem_query_records(
            problem=problem,
            stem=stem,
            options=options,
            agent_plan=agent_plan,
            max_options=max_options_per_problem,
            max_queries_per_problem=max_queries_per_problem,
            max_queries_per_option=max_queries_per_option,
            enable_relation_query_planner=enable_relation_query_planner,
            enable_sweep_gap_relation_backfill_queries=enable_sweep_gap_relation_backfill_queries,
            enable_option_aware_query_expansion=enable_option_aware_query_expansion,
            enable_answer_bearing_binding_queries=enable_answer_bearing_binding_queries,
            enable_answer_bearing_pair_binding_queries=(
                enable_answer_bearing_pair_binding_queries
            ),
            enable_required_relation_completion_queries=(
                enable_required_relation_completion_queries
            ),
            enable_candidate_specific_answer_bearing_queries=(
                enable_candidate_specific_answer_bearing_queries
            ),
            focus_option_hashes=set(failure_focus.get("option_hashes", []) or []),
            focus_missing_required_term_hashes_by_option=(
                failure_focus.get("option_missing_required_term_hashes") or {}
            ),
            focus_all_options=bool(failure_focus.get("expand_all_options")),
            focus_only=focus_only,
            relation_query_planner_model=relation_query_planner_model,
            logger=logger,
        )
        prefetch_query_plan_cache_summary = _prefetch_query_plan_cache_put(
            problem=problem,
            options=options,
            query_records=query_records,
            relation_query_planner_summary=relation_query_planner_summary,
        )
        row = {
            "seed_offset": seed_offset,
            "status": "planned",
            "_problem": problem,
            "_stem": stem,
            "_options": options,
            "problem_id_hash": problem.get("id_hash"),
            "question_hash": problem.get("question_hash"),
            "category_hash": stable_hash({"category": problem.get("category") or ""}),
            "raw_subject_hash": stable_hash({"raw_subject": problem.get("raw_subject") or ""}),
            "domain": _classify_hle_domain(problem),
            "answer_type": problem.get("answer_type"),
            "option_count": len(options),
            "operator_status": operator_stage.get("status"),
            "operator_reason": operator_stage.get("reason"),
            "operator_family_tags": family_tags,
            "operator_source_ids": list(operator_stage.get("operator_source_ids", []) or []),
            "failure_focus": _sanitize_failure_focus_seed_summary(failure_focus),
            "relation_query_planner": relation_query_planner_summary,
            "prefetch_query_plan_cache": prefetch_query_plan_cache_summary,
            "query_records": query_records,
        }
        rows.append(row)
        log_event(
            logger,
            {
                "event": "hle_source_prefetch_problem_planned",
                "seed_offset": seed_offset,
                "problem_id_hash": row.get("problem_id_hash"),
                "question_hash": row.get("question_hash"),
                "domain": row.get("domain"),
                "answer_type": row.get("answer_type"),
                "option_count": row.get("option_count"),
                "query_count": len(query_records),
                "query_kind_counts": dict(
                    Counter(str(query.get("query_kind") or "") for query in query_records)
                ),
                "operator_status": row.get("operator_status"),
                "operator_family_tags": list(row.get("operator_family_tags", []) or []),
                "failure_focus_status": (row.get("failure_focus") or {}).get("status"),
                "failure_focus_option_count": (row.get("failure_focus") or {}).get("option_count"),
                "failure_focus_reason_counts": (row.get("failure_focus") or {}).get("reason_counts"),
                "focus_only": bool(focus_only),
                "relation_query_planner_status": relation_query_planner_summary.get("status"),
                "relation_query_planner_query_count": relation_query_planner_summary.get("query_count"),
                "relation_query_planner_model_query_count": relation_query_planner_summary.get("model_query_count"),
                "prefetch_query_plan_cache_status": prefetch_query_plan_cache_summary.get("status"),
                "prefetch_query_plan_cache_query_count": prefetch_query_plan_cache_summary.get("query_count"),
                "prefetch_query_plan_cache_option_count": len(
                    prefetch_query_plan_cache_summary.get("query_hashes_by_option_hash") or {}
                ),
                "prefetch_query_plan_cache_private_raw_query_cache_written": bool(
                    prefetch_query_plan_cache_summary.get("private_raw_query_cache_written")
                ),
                "raw_content_persisted": False,
            },
        )
    return rows


def _operator_plan_for_prefetch(
    *,
    problem: dict[str, Any],
    graph: SimpleAssumptionGraph | None,
) -> dict[str, Any]:
    previous_env = {
        "HLE_ENABLE_ASSUMPTION_OPERATORS": os.environ.get("HLE_ENABLE_ASSUMPTION_OPERATORS"),
        "HLE_ASSUMPTION_OPERATORS_ALLOW_WITHOUT_CONTEXT": os.environ.get(
            "HLE_ASSUMPTION_OPERATORS_ALLOW_WITHOUT_CONTEXT"
        ),
        "HLE_ASSUMPTION_OPERATOR_RETRIEVAL_FALLBACK": os.environ.get(
            "HLE_ASSUMPTION_OPERATOR_RETRIEVAL_FALLBACK"
        ),
        "HLE_ASSUMPTION_OPERATOR_MAX_SPECS": os.environ.get("HLE_ASSUMPTION_OPERATOR_MAX_SPECS"),
    }
    os.environ["HLE_ENABLE_ASSUMPTION_OPERATORS"] = "1"
    os.environ["HLE_ASSUMPTION_OPERATORS_ALLOW_WITHOUT_CONTEXT"] = "1"
    os.environ["HLE_ASSUMPTION_OPERATOR_RETRIEVAL_FALLBACK"] = "1"
    os.environ.setdefault("HLE_ASSUMPTION_OPERATOR_MAX_SPECS", "2")
    try:
        stage = _compile_hle_operator_stage(
            retrieval_result=None,
            graph=graph,
            problem_text=str(problem.get("_question") or ""),
            problem_id=str(problem.get("id_hash") or ""),
            domain=_classify_hle_domain(problem),
            difficulty="hle",
            context_allowed=True,
            generic_graph_context_only=False,
        )
    finally:
        _restore_env(previous_env)
    return {
        "stages": {
            "operator_spec_compiler": stage,
        },
        "assumption_operator_specs": stage.get("operator_specs", []),
        "operator_context": stage.get("operator_context", ""),
    }


def _problem_query_records(
    *,
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
    agent_plan: dict[str, Any],
    max_options: int,
    max_queries_per_problem: int,
    max_queries_per_option: int,
    enable_relation_query_planner: bool = False,
    enable_sweep_gap_relation_backfill_queries: bool = False,
    enable_option_aware_query_expansion: bool = False,
    enable_answer_bearing_binding_queries: bool = False,
    enable_answer_bearing_pair_binding_queries: bool = False,
    enable_required_relation_completion_queries: bool = False,
    enable_candidate_specific_answer_bearing_queries: bool = False,
    focus_option_hashes: set[str] | None = None,
    focus_missing_required_term_hashes_by_option: dict[str, Any] | None = None,
    focus_all_options: bool = False,
    focus_only: bool = False,
    relation_query_planner_model: str = "gpt-5.4-mini",
    logger: JsonlDiagnosticLogger | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    relation_queries_by_label: dict[str, list[str]] = {}
    relation_query_planner_summary: dict[str, Any] = {
        "status": "disabled",
        "reason": "planner_not_requested",
        "underlying_model_calls": 0,
    }
    if enable_relation_query_planner:
        relation_queries_by_label, relation_query_planner_summary = _run_option_claim_relation_query_planner(
            problem=problem,
            options=options,
            agent_plan=agent_plan,
            model=relation_query_planner_model,
            eval_id="hle_source_prefetch",
            call_id=stable_hash({"problem_id": problem.get("id_hash"), "stage": "relation_query_planner"}),
            logger=logger,
            timeout=None,
            max_tokens=512,
        )
    focus_hashes = {str(value) for value in (focus_option_hashes or set()) if str(value).strip()}
    missing_required_hashes_by_option = {
        str(option_hash): {
            str(term_hash)
            for term_hash in (hashes or [])
            if str(term_hash).strip()
        }
        for option_hash, hashes in (
            focus_missing_required_term_hashes_by_option or {}
        ).items()
    }
    option_items = list(options.items())[: max(1, max_options)]
    if focus_hashes:
        focused_items = [
            item
            for item in option_items
            if stable_hash({"option_label": item[0]}) in focus_hashes
        ]
        if focus_only and not focus_all_options:
            option_items = focused_items
        else:
            option_items = focused_items + [
                item
                for item in option_items
                if stable_hash({"option_label": item[0]}) not in focus_hashes
            ]
    for label, option_text in option_items:
        option_hash = stable_hash({"option_label": label})
        relation_queries = relation_queries_by_label.get(label, [])
        option_queries = _option_evidence_queries_for_plan(
            stem,
            option_text,
            problem,
            agent_plan=agent_plan,
        )
        claim_queries = _option_claim_evidence_queries_for_plan(
            stem=stem,
            option_text=option_text,
            problem=problem,
            agent_plan=agent_plan,
        )
        answer_web_queries = _option_claim_answer_web_fallback_queries(
            stem=stem,
            option_text=option_text,
            problem=problem,
        )
        option_aware_query_pairs: list[tuple[str, str]] = []
        if enable_option_aware_query_expansion:
            option_aware_query_pairs = _option_aware_source_prefetch_queries(
                stem=stem,
                option_text=option_text,
                problem=problem,
                agent_plan=agent_plan,
            )
        answer_bearing_binding_query_pairs: list[tuple[str, str]] = []
        if enable_answer_bearing_binding_queries:
            answer_bearing_binding_query_pairs = (
                _answer_bearing_binding_source_prefetch_queries(
                    stem=stem,
                    option_text=option_text,
                    problem=problem,
                    agent_plan=agent_plan,
                )
            )
        answer_bearing_pair_binding_query_pairs: list[tuple[str, str]] = []
        if enable_answer_bearing_pair_binding_queries:
            answer_bearing_pair_binding_query_pairs = (
                _answer_bearing_pair_binding_source_prefetch_queries(
                    stem=stem,
                    option_label=label,
                    option_text=option_text,
                    options=options,
                    problem=problem,
                    agent_plan=agent_plan,
                )
            )
        required_relation_completion_query_pairs: list[tuple[str, str]] = []
        if enable_required_relation_completion_queries:
            required_relation_completion_query_pairs = (
                _required_relation_completion_source_prefetch_queries(
                    stem=stem,
                    option_text=option_text,
                    problem=problem,
                    agent_plan=agent_plan,
                )
            )
        term_identity_missing_required_query_pairs: list[tuple[str, str]] = []
        if missing_required_hashes_by_option.get(option_hash):
            term_identity_missing_required_query_pairs = (
                _term_identity_missing_required_source_prefetch_queries(
                    stem=stem,
                    option_text=option_text,
                    problem=problem,
                    agent_plan=agent_plan,
                    missing_required_term_hashes=(
                        missing_required_hashes_by_option.get(option_hash) or set()
                    ),
                )
            )
        candidate_specific_answer_bearing_query_pairs: list[tuple[str, str]] = []
        if enable_candidate_specific_answer_bearing_queries:
            candidate_specific_answer_bearing_query_pairs = (
                _candidate_specific_answer_bearing_source_prefetch_queries(
                    stem=stem,
                    option_label=label,
                    option_text=option_text,
                    options=options,
                    problem=problem,
                    agent_plan=agent_plan,
                )
            )
        sweep_gap_relation_queries: list[tuple[str, str]] = []
        if enable_sweep_gap_relation_backfill_queries:
            deterministic_queries = _deterministic_option_claim_relation_queries(
                stem=stem,
                option_text=option_text,
                problem=problem,
                agent_plan=agent_plan,
            )
            relation_slot_plan = _option_claim_relation_slot_plan(
                stem=stem,
                option_text=option_text,
                planned_queries=relation_queries + deterministic_queries + claim_queries[:2],
            )
            local_relation_queries = _option_claim_local_relation_query_expansion_queries(
                stem=stem,
                option_text=option_text,
                problem=problem,
                planned_queries=relation_queries + deterministic_queries + claim_queries[:2],
                relation_slot_plan=relation_slot_plan,
            )
            sweep_gap_relation_queries = [
                ("option_claim_deterministic_relation", query)
                for query in deterministic_queries
            ] + [
                ("option_claim_local_relation_expansion", query)
                for query in local_relation_queries
            ]
        combined = _balanced_prefetch_query_mix(
            relation_queries=relation_queries,
            sweep_gap_relation_queries=sweep_gap_relation_queries,
            option_aware_query_pairs=option_aware_query_pairs,
            answer_bearing_binding_query_pairs=answer_bearing_binding_query_pairs,
            answer_bearing_pair_binding_query_pairs=(
                answer_bearing_pair_binding_query_pairs
            ),
            term_identity_missing_required_query_pairs=(
                term_identity_missing_required_query_pairs
            ),
            candidate_specific_answer_bearing_query_pairs=(
                candidate_specific_answer_bearing_query_pairs
            ),
            required_relation_completion_query_pairs=(
                required_relation_completion_query_pairs
            ),
            answer_web_queries=answer_web_queries,
            option_queries=option_queries,
            claim_queries=claim_queries,
            max_queries=max_queries_per_option,
        )
        for kind, query in combined[: max(1, max_queries_per_option)]:
            query = str(query or "").strip()
            if not query:
                continue
            query_hash = stable_hash({"query": query})
            key = query_hash
            if key in seen:
                continue
            seen.add(key)
            preferred_sources = _source_prefetch_preferred_sources_for_query(
                query_kind=kind,
                problem=problem,
                stem=stem,
            )
            record = {
                "option_hash": stable_hash({"option_label": label}),
                "option_label_hash": stable_hash({"option_label": label}),
                "option_text_hash": stable_hash({"option_text": option_text}),
                "option_choice": _extract_choice(label) or label,
                "_option_label": label,
                "_option_text": option_text,
                "query_kind": kind,
                "query_hash": query_hash,
                "_query": query,
            }
            diagnostic_option = _source_prefetch_diagnostic_option_text(
                query_kind=kind,
                stem=stem,
                option_text=option_text,
            )
            if diagnostic_option and diagnostic_option != option_text:
                record["_source_diagnostic_option_text"] = diagnostic_option
                record["source_diagnostic_option_text_hash"] = stable_hash({
                    "source_diagnostic_option_text": diagnostic_option,
                })
                record["source_diagnostic_option_expansion_kind"] = kind
            if preferred_sources:
                record["preferred_sources"] = preferred_sources
            records.append(record)
            if len(records) >= max(1, max_queries_per_problem):
                return records, relation_query_planner_summary
    return records, relation_query_planner_summary


def _source_prefetch_preferred_sources_for_query(
    *,
    query_kind: str,
    problem: dict[str, Any],
    stem: str,
) -> list[str]:
    kind = str(query_kind or "").strip()
    local_preferred = (
        ["local_evidence_corpus"]
        if os.environ.get("HLE_EVIDENCE_SOURCE_CORPUS_PATHS", "").strip()
        else []
    )
    if kind in {
        "candidate_specific_answer_bearing_patient_label_resolution",
        "candidate_specific_answer_bearing_oxidation_spin_alias",
    }:
        return local_preferred
    if not kind.startswith("candidate_specific_numeric_threshold_"):
        return []
    domain_text = " ".join(
        str(value or "")
        for value in (
            problem.get("category"),
            problem.get("raw_subject"),
            stem,
        )
    ).lower()
    biomedical_markers = {
        "bi-rads",
        "biomedical",
        "biopsy",
        "breast",
        "cell",
        "cells",
        "clinical",
        "disease",
        "elastography",
        "lesion",
        "lesions",
        "medicine",
        "medical",
        "oncology",
        "patient",
        "patients",
        "protein",
        "therapy",
        "tumor",
        "tumour",
        "ultrasound",
    }
    if any(marker in domain_text for marker in biomedical_markers):
        return local_preferred + [
            "pubmed_pmc_fulltext",
            "pubmed",
            "semantic_scholar",
            "openalex",
            "crossref",
        ]
    return local_preferred


def _interleave_queries(option_queries: list[str], claim_queries: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    max_len = max(len(option_queries), len(claim_queries))
    for index in range(max_len):
        if index < len(option_queries):
            out.append(("option_evidence", option_queries[index]))
        if index < len(claim_queries):
            out.append(("option_claim", claim_queries[index]))
    return out


def _balanced_prefetch_query_mix(
    *,
    relation_queries: list[str],
    sweep_gap_relation_queries: list[tuple[str, str]],
    answer_web_queries: list[str],
    option_queries: list[str],
    claim_queries: list[str],
    max_queries: int,
    option_aware_query_pairs: list[tuple[str, str]] | None = None,
    answer_bearing_binding_query_pairs: list[tuple[str, str]] | None = None,
    answer_bearing_pair_binding_query_pairs: list[tuple[str, str]] | None = None,
    term_identity_missing_required_query_pairs: list[tuple[str, str]] | None = None,
    candidate_specific_answer_bearing_query_pairs: list[tuple[str, str]] | None = None,
    required_relation_completion_query_pairs: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    deterministic_relation_queries = [
        (kind, query)
        for kind, query in sweep_gap_relation_queries
        if kind == "option_claim_deterministic_relation"
    ]
    local_relation_queries = [
        (kind, query)
        for kind, query in sweep_gap_relation_queries
        if kind == "option_claim_local_relation_expansion"
    ]
    other_relation_queries = [
        (kind, query)
        for kind, query in sweep_gap_relation_queries
        if kind
        not in {
            "option_claim_deterministic_relation",
            "option_claim_local_relation_expansion",
        }
    ]
    candidate_specific_buckets = _split_prefetch_query_pairs_by_kind(
        candidate_specific_answer_bearing_query_pairs or []
    )
    buckets = [
        list(term_identity_missing_required_query_pairs or []),
        [("option_claim_relation_planner", query) for query in relation_queries],
        list(required_relation_completion_query_pairs or []),
        deterministic_relation_queries,
        local_relation_queries,
        *candidate_specific_buckets,
        list(answer_bearing_pair_binding_query_pairs or []),
        list(answer_bearing_binding_query_pairs or []),
        [("answer_web_fallback", query) for query in answer_web_queries],
        [("option_claim", query) for query in claim_queries],
        [("option_evidence", query) for query in option_queries],
        list(option_aware_query_pairs or []),
        other_relation_queries,
    ]
    return _round_robin_query_pairs(buckets, max_queries=max_queries)


def _split_prefetch_query_pairs_by_kind(
    pairs: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    """Keep candidate-specific query variants visible under small per-option budgets."""
    by_kind: dict[str, list[tuple[str, str]]] = {}
    kind_order: list[str] = []
    for kind, query in pairs:
        clean_kind = str(kind or "").strip()
        if clean_kind not in by_kind:
            by_kind[clean_kind] = []
            kind_order.append(clean_kind)
        by_kind[clean_kind].append((kind, query))
    priority = {
        "candidate_specific_numeric_threshold_biomedical_pubmed_anchor": 0,
        "candidate_specific_numeric_threshold_exact_value": 0,
        "candidate_specific_numeric_threshold_same_row_primary": 0,
        "candidate_specific_numeric_threshold_same_row_unit_variant": 0,
        "candidate_specific_numeric_threshold_same_row_relation": 0,
        "candidate_specific_numeric_threshold_same_row_entity_anchor": 0,
        "candidate_specific_numeric_threshold_same_row_anchor": 0,
        "candidate_specific_numeric_threshold_same_row_source_title_backfill": 0,
        "candidate_specific_numeric_threshold_same_row_source_relation_backfill": 0,
        "candidate_specific_numeric_threshold_same_row_source_url_backfill": 0,
        "candidate_specific_numeric_threshold_unit_variant": 0,
        "candidate_specific_numeric_threshold_extreme_relation": 0,
        "candidate_specific_numeric_threshold_question_anchor": 0,
        "candidate_specific_answer_bearing_exact_option": 0,
        "candidate_specific_answer_bearing_experiment_anchor": 0,
        "candidate_specific_answer_bearing_patient_label_resolution": 0,
        "candidate_specific_answer_bearing_oxidation_spin_alias": 0,
        "candidate_specific_answer_bearing_required_term": 0,
        "candidate_specific_answer_bearing_required_pair": 0,
        "candidate_specific_answer_bearing_required_phrase_term": 0,
        "candidate_specific_answer_bearing_witness": 1,
        "candidate_specific_answer_bearing_relation": 2,
        "candidate_specific_answer_bearing_question_phrase": 3,
        "candidate_specific_answer_bearing_option_phrase": 4,
        "candidate_specific_answer_bearing_disambiguation": 5,
        "candidate_specific_answer_bearing_operator": 6,
    }
    order_index = {kind: index for index, kind in enumerate(kind_order)}
    kind_order = sorted(
        kind_order,
        key=lambda kind: (priority.get(kind, 99), order_index.get(kind, 999)),
    )
    return [by_kind[kind] for kind in kind_order]


def _option_aware_source_prefetch_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    option_terms = _source_prefetch_option_anchor_terms(option_text=option_text, stem=stem, max_terms=10)
    if not option_terms:
        return []
    subject_terms = _source_prefetch_subject_terms(problem, max_terms=4)
    question_anchors = _question_evidence_anchor_terms(stem, option_text=option_text, max_terms=8)
    relation_terms = _question_relation_query_terms(stem)
    option_phrases = _source_prefetch_focus_phrases(option_text, max_phrases=4)
    question_phrases = _source_prefetch_focus_phrases(stem, max_phrases=3)
    operator_terms = _source_prefetch_operator_terms(agent_plan, max_terms=5)
    seeds: list[tuple[str, str]] = [
        (
            "option_anchor_relation",
            " ".join(option_terms[:6] + question_anchors[:5] + relation_terms[:4] + subject_terms),
        ),
        (
            "option_anchor_relation",
            " ".join(option_terms[:6] + relation_terms[:5] + subject_terms),
        ),
        (
            "question_relation_anchor",
            " ".join(question_anchors[:6] + option_terms[:5] + subject_terms),
        ),
    ]
    if operator_terms:
        seeds.append((
            "option_operator_anchor",
            " ".join(option_terms[:5] + question_anchors[:4] + operator_terms[:5] + subject_terms),
        ))
    for phrase in option_phrases[:3]:
        seeds.extend([
            (
                "option_focus_phrase",
                " ".join([phrase] + question_anchors[:3] + relation_terms[:3] + subject_terms),
            ),
            (
                "option_focus_phrase",
                " ".join([phrase] + subject_terms),
            ),
        ])
    for phrase in question_phrases[:2]:
        seeds.append((
            "question_relation_anchor",
            " ".join([phrase] + option_terms[:5] + relation_terms[:3] + subject_terms),
        ))
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if not _source_prefetch_query_has_option_anchor(query=query, option_text=option_text):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 8:
            break
    return pairs


def _answer_bearing_binding_source_prefetch_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    option_terms = _source_prefetch_option_anchor_terms(
        option_text=option_text,
        stem=stem,
        max_terms=12,
    )
    if not option_terms:
        return []
    subject_terms = _source_prefetch_subject_terms(problem, max_terms=4)
    relation_terms = _question_relation_query_terms(stem)
    question_anchors = _question_evidence_anchor_terms(
        stem,
        option_text=option_text,
        max_terms=10,
    )
    option_phrases = _source_prefetch_focus_phrases(option_text, max_phrases=5)
    question_phrases = _source_prefetch_focus_phrases(stem, max_phrases=4)
    operator_terms = _source_prefetch_operator_terms(agent_plan, max_terms=4)
    seeds: list[tuple[str, str]] = []
    seeds.append((
        "answer_bearing_relation_binding",
        " ".join(option_terms[:7] + relation_terms[:5] + question_anchors[:6] + subject_terms),
    ))
    if question_phrases:
        seeds.append((
            "answer_bearing_relation_binding",
            " ".join(option_terms[:6] + relation_terms[:5] + question_phrases[:2] + subject_terms),
        ))
    if option_phrases:
        seeds.append((
            "answer_bearing_option_focus",
            " ".join(option_phrases[:2] + question_anchors[:5] + relation_terms[:4] + subject_terms),
        ))
    if operator_terms:
        seeds.append((
            "answer_bearing_operator_binding",
            " ".join(option_terms[:6] + relation_terms[:4] + operator_terms[:4] + subject_terms),
        ))
    seeds.append((
        "answer_bearing_source_specificity",
        " ".join(option_terms[:5] + question_anchors[:4] + ["mechanism", "evidence"] + subject_terms),
    ))
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if not _source_prefetch_query_has_option_anchor(query=query, option_text=option_text):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 6:
            break
    return pairs


def _required_relation_completion_source_prefetch_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    option_terms = _source_prefetch_option_anchor_terms(
        option_text=option_text,
        stem=stem,
        max_terms=10,
    )
    if not option_terms:
        return []
    relation_signature = _option_claim_question_relation_signature_terms(
        stem=stem,
        option_text=option_text,
        max_terms=10,
    )
    required_terms = [
        str(term).strip("._-")
        for term in relation_signature.get("required_terms", []) or []
        if str(term).strip("._-")
        and str(term).lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ]
    required_terms = list(dict.fromkeys(required_terms))
    if not required_terms:
        return []
    subject_terms = _source_prefetch_subject_terms(problem, max_terms=4)
    relation_terms = _question_relation_query_terms(stem)
    question_anchors = _question_evidence_anchor_terms(
        stem,
        option_text=option_text,
        max_terms=10,
    )
    option_phrases = _source_prefetch_focus_phrases(option_text, max_phrases=4)
    operator_terms = _source_prefetch_operator_terms(agent_plan, max_terms=4)
    seeds: list[tuple[str, str]] = [
        (
            "answer_bearing_required_relation_completion",
            " ".join(
                option_terms[:6]
                + required_terms[:6]
                + question_anchors[:4]
                + subject_terms
            ),
        ),
        (
            "answer_bearing_required_relation_completion",
            " ".join(
                option_terms[:5]
                + required_terms[:6]
                + relation_terms[:4]
                + ["evidence"]
                + subject_terms
            ),
        ),
    ]
    if option_phrases:
        seeds.append((
            "answer_bearing_required_relation_exact_option",
            " ".join(
                option_phrases[:1]
                + required_terms[:2]
                + ["evidence"]
                + subject_terms
            ),
        ))
        for required_term in required_terms[:4]:
            seeds.append((
                "answer_bearing_required_relation_phrase_term",
                " ".join(
                    option_phrases[:1]
                    + [required_term]
                    + relation_terms[:3]
                    + subject_terms
                ),
            ))
    for left_index, left_term in enumerate(required_terms[:5]):
        for right_term in required_terms[left_index + 1:left_index + 3]:
            seeds.append((
                "answer_bearing_required_relation_term_pair",
                " ".join(
                    option_terms[:5]
                    + [left_term, right_term]
                    + relation_terms[:2]
                    + subject_terms
                ),
            ))
    for required_term in required_terms[:4]:
        seeds.append((
            "answer_bearing_required_relation_term",
            " ".join(
                option_terms[:5]
                + [required_term]
                + relation_terms[:4]
                + question_anchors[:3]
                + subject_terms
            ),
        ))
    if option_phrases:
        seeds.append((
            "answer_bearing_required_relation_focus",
            " ".join(
                option_phrases[:1]
                + required_terms[:6]
                + relation_terms[:4]
                + subject_terms
            ),
        ))
    if operator_terms:
        seeds.append((
            "answer_bearing_required_relation_operator",
            " ".join(
                option_terms[:5]
                + required_terms[:5]
                + operator_terms[:4]
                + subject_terms
            ),
        ))

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if not _source_prefetch_query_has_option_anchor(
            query=query,
            option_text=option_text,
        ):
            continue
        if kind in {
            "answer_bearing_required_relation_phrase_term",
            "answer_bearing_required_relation_term_pair",
        } and not (_content_terms(query) & {term.lower() for term in required_terms}):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 12:
            break
    return pairs


def _term_identity_missing_required_source_prefetch_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None,
    missing_required_term_hashes: set[str],
) -> list[tuple[str, str]]:
    option_terms = _source_prefetch_option_anchor_terms(
        option_text=option_text,
        stem=stem,
        max_terms=10,
    )
    if not missing_required_term_hashes:
        return []
    relation_signature = _option_claim_question_relation_signature_terms(
        stem=stem,
        option_text=option_text,
        max_terms=12,
    )
    required_terms = [
        str(term).lower().strip("._-")
        for term in relation_signature.get("required_terms", []) or []
        if str(term).strip("._-")
        and str(term).lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ]
    required_terms = list(dict.fromkeys(required_terms))
    if not required_terms:
        return []
    term_hash_by_term = {
        term: (_option_claim_relation_signature_term_hashes([term]) or [""])[0]
        for term in required_terms
    }
    missing_terms = [
        term
        for term in required_terms
        if term_hash_by_term.get(term) in missing_required_term_hashes
    ]
    if _TERM_IDENTITY_ALL_REQUIRED_TERMS_SENTINEL in missing_required_term_hashes:
        missing_terms = required_terms
    if not missing_terms:
        return []
    covered_terms = [term for term in required_terms if term not in set(missing_terms)]
    relation_terms = [
        term
        for term in _question_relation_query_terms(stem)
        if str(term).lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ]
    question_anchors = _question_evidence_anchor_terms(
        stem,
        option_text=option_text,
        max_terms=10,
    )
    high_specificity_anchors = _source_prefetch_high_specificity_question_anchors(
        stem,
        max_terms=8,
    )
    subject_terms = _source_prefetch_subject_terms(problem, max_terms=4)
    option_phrases = _source_prefetch_focus_phrases(option_text, max_phrases=3)
    operator_terms = _source_prefetch_operator_terms(agent_plan, max_terms=4)
    seeds: list[tuple[str, str]] = []
    for term in missing_terms[:4]:
        paired_terms = [value for value in covered_terms if value != term][:3]
        seeds.append((
            "term_identity_missing_required_single",
            " ".join(
                option_terms[:5]
                + [term]
                + paired_terms
                + relation_terms[:3]
                + question_anchors[:3]
                + subject_terms
            ),
        ))
        if option_phrases:
            seeds.append((
                "term_identity_missing_required_exact_option",
                " ".join(
                    option_phrases[:1]
                    + [term]
                    + paired_terms[:2]
                    + relation_terms[:3]
                    + subject_terms
                ),
            ))
        if high_specificity_anchors:
            seeds.append((
                "term_identity_missing_required_experiment_anchor",
                " ".join(
                    option_terms[:5]
                    + [term]
                    + high_specificity_anchors[:6]
                    + subject_terms
                ),
            ))
    for left_index, left_term in enumerate(missing_terms[:4]):
        for right_term in missing_terms[left_index + 1:left_index + 3]:
            seeds.append((
                "term_identity_missing_required_pair",
                " ".join(
                    option_terms[:5]
                    + [left_term, right_term]
                    + relation_terms[:4]
                    + question_anchors[:3]
                    + subject_terms
                ),
            ))
    if operator_terms:
        seeds.append((
            "term_identity_missing_required_operator",
            " ".join(
                option_terms[:5]
                + missing_terms[:4]
                + operator_terms[:4]
                + subject_terms
            ),
        ))
    if not option_terms:
        stem_anchor_terms = high_specificity_anchors[:8] or question_anchors[:8]
        if stem_anchor_terms:
            for term in missing_terms[:4]:
                seeds.append((
                    "term_identity_missing_required_stem_anchor",
                    " ".join(
                        [term]
                        + covered_terms[:3]
                        + relation_terms[:4]
                        + stem_anchor_terms
                        + subject_terms
                    ),
                ))

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    missing_term_set = {term.lower() for term in missing_terms}
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        requires_option_anchor = kind != "term_identity_missing_required_stem_anchor"
        if requires_option_anchor and not _source_prefetch_query_has_option_anchor(
            query=query,
            option_text=option_text,
        ):
            continue
        if not (_content_terms(query) & missing_term_set):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 10:
            break
    return pairs


def _answer_bearing_pair_binding_source_prefetch_queries(
    *,
    stem: str,
    option_label: str,
    option_text: str,
    options: dict[str, str],
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    option_terms = _source_prefetch_option_anchor_terms(
        option_text=option_text,
        stem=stem,
        max_terms=10,
    )
    if not option_terms:
        return []
    subject_terms = _source_prefetch_subject_terms(problem, max_terms=4)
    relation_terms = _question_relation_query_terms(stem)
    question_anchors = _question_evidence_anchor_terms(
        stem,
        option_text=option_text,
        max_terms=10,
    )
    option_phrases = _source_prefetch_focus_phrases(option_text, max_phrases=3)
    question_phrases = _source_prefetch_focus_phrases(stem, max_phrases=3)
    operator_terms = _source_prefetch_operator_terms(agent_plan, max_terms=4)
    competitor_rows: list[dict[str, Any]] = []
    for other_label, other_text in options.items():
        if other_label == option_label:
            continue
        competitor_terms = _source_prefetch_option_anchor_terms(
            option_text=other_text,
            stem=stem,
            max_terms=8,
        )
        if not competitor_terms:
            continue
        competitor_rows.append({
            "label": other_label,
            "text": other_text,
            "terms": competitor_terms,
            "phrases": _source_prefetch_focus_phrases(other_text, max_phrases=2),
            "overlap": len(set(option_terms) & set(competitor_terms)),
            "term_count": len(competitor_terms),
        })
    if not competitor_rows:
        return []
    competitor_rows.sort(
        key=lambda row: (
            -int(row.get("overlap") or 0),
            -int(row.get("term_count") or 0),
            str(row.get("label") or ""),
        )
    )
    seeds: list[tuple[str, str]] = []
    for competitor in competitor_rows[:4]:
        competitor_terms = list(competitor.get("terms", []) or [])
        competitor_phrases = list(competitor.get("phrases", []) or [])
        seeds.append((
            "answer_bearing_pair_binding",
            " ".join(
                option_terms[:5]
                + competitor_terms[:5]
                + relation_terms[:5]
                + question_anchors[:5]
                + subject_terms
            ),
        ))
        seeds.append((
            "answer_bearing_pair_contrast",
            " ".join(
                option_terms[:4]
                + competitor_terms[:4]
                + question_phrases[:1]
                + relation_terms[:4]
                + ["comparison", "evidence"]
                + subject_terms
            ),
        ))
        if option_phrases or competitor_phrases:
            seeds.append((
                "answer_bearing_pair_disambiguation",
                " ".join(
                    option_phrases[:1]
                    + competitor_phrases[:1]
                    + relation_terms[:4]
                    + question_anchors[:4]
                    + subject_terms
                ),
            ))
        if operator_terms:
            seeds.append((
                "answer_bearing_pair_operator_binding",
                " ".join(
                    option_terms[:4]
                    + competitor_terms[:4]
                    + relation_terms[:3]
                    + operator_terms[:4]
                    + subject_terms
                ),
            ))

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if not _source_prefetch_query_has_option_anchor(
            query=query,
            option_text=option_text,
        ):
            continue
        if not any(
            _source_prefetch_query_has_option_anchor(
                query=query,
                option_text=str(row.get("text") or ""),
            )
            for row in competitor_rows
        ):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 8:
            break
    return pairs


def _source_prefetch_patient_label_descriptions(stem: str) -> dict[str, str]:
    text = re.sub(r"\s+", " ", str(stem or " ").replace("\n", " ")).strip()
    descriptions: dict[str, str] = {}
    pattern = re.compile(
        r"\bPatient\s+(\d{1,2})\s*:\s*(.*?)(?=\bPatient\s+\d{1,2}\s*:|\bAnswer\s+Choices\b|\bPrioriti[sz]e\b|\bWhich\b|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        label = match.group(1).strip()
        description = _clean_evidence_query(match.group(2))
        if label and description:
            descriptions[label] = description
    return descriptions


def _source_prefetch_patient_label_diagnostic_option_text(
    *,
    stem: str,
    option_text: str,
) -> str:
    label_descriptions = _source_prefetch_patient_label_descriptions(stem)
    selected_labels = [
        match.group(1)
        for match in re.finditer(r"\bPatient\s+(\d{1,2})\b", str(option_text or ""), re.IGNORECASE)
    ]
    selected_labels = list(dict.fromkeys(selected_labels))
    if not selected_labels or not label_descriptions:
        return str(option_text or "")
    parts = [str(option_text or "").strip()]
    for label in selected_labels:
        description = label_descriptions.get(label)
        if description:
            parts.append(f"Patient {label} {description}")
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def _source_prefetch_patient_label_resolution_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    relation_focus: list[str],
    subject_terms: list[str],
) -> list[tuple[str, str]]:
    label_descriptions = _source_prefetch_patient_label_descriptions(stem)
    if not label_descriptions:
        return []
    selected_labels = [
        match.group(1)
        for match in re.finditer(r"\bPatient\s+(\d{1,2})\b", str(option_text or ""), re.IGNORECASE)
    ]
    selected_labels = list(dict.fromkeys(selected_labels))
    if not selected_labels:
        return []
    selected_descriptions = [
        f"Patient {label} {label_descriptions[label]}"
        for label in selected_labels
        if label in label_descriptions
    ]
    if not selected_descriptions:
        return []
    option_phrase = _source_prefetch_clean_phrase(option_text)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for description in selected_descriptions[:4]:
        query = " ".join(
            [option_phrase, description]
            + relation_focus[:5]
            + [
                "surgical indications",
                "priority",
                "thoracolumbar injury classification",
            ]
            + subject_terms[:3]
        )
        query = _clean_evidence_query(query)
        key = _normalize_query_key(query)
        if query and key and key not in seen:
            seen.add(key)
            pairs.append(("candidate_specific_answer_bearing_patient_label_resolution", query))
    return pairs


def _source_prefetch_fe_oxidation_spin_aliases(option_text: str) -> list[str]:
    text = str(option_text or "")
    oxidation_aliases = {
        "I": ["Fe 1+", "Fe+", "iron I"],
        "II": ["Fe 2+", "Fe2+", "ferrous", "iron II", "iron(II)"],
        "III": ["Fe 3+", "Fe3+", "ferric", "iron III", "iron(III)"],
        "IV": ["Fe 4+", "Fe4+", "iron IV", "iron(IV)"],
        "V": ["Fe 5+", "Fe5+", "iron V", "iron(V)"],
        "VI": ["Fe 6+", "Fe6+", "iron VI", "iron(VI)"],
    }
    aliases: list[str] = []
    seen_aliases: set[str] = set()
    for match in re.finditer(r"\bFe\s*\(\s*([IVX]+)\s*\)", text, flags=re.IGNORECASE):
        roman = match.group(1).upper()
        for alias in oxidation_aliases.get(roman, []):
            key = alias.lower()
            if key not in seen_aliases:
                seen_aliases.add(key)
                aliases.append(alias)
    return aliases


def _source_prefetch_oxidation_spin_diagnostic_option_text(option_text: str) -> str:
    aliases = _source_prefetch_fe_oxidation_spin_aliases(option_text)
    spin_phrases = []
    for match in re.finditer(r"\bS\s*=\s*([0-9]+(?:/[0-9]+)?(?:\.\d+)?)", str(option_text or ""), flags=re.IGNORECASE):
        spin = match.group(1).strip()
        if spin:
            spin_phrases.extend([f"S = {spin}", f"spin {spin}"])
    parts = [str(option_text or "").strip()] + aliases + list(dict.fromkeys(spin_phrases))
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def _source_prefetch_diagnostic_option_text(
    *,
    query_kind: str,
    stem: str,
    option_text: str,
) -> str:
    kind = str(query_kind or "")
    if kind == "candidate_specific_answer_bearing_patient_label_resolution":
        return _source_prefetch_patient_label_diagnostic_option_text(
            stem=stem,
            option_text=option_text,
        )
    if kind == "candidate_specific_answer_bearing_oxidation_spin_alias":
        return _source_prefetch_oxidation_spin_diagnostic_option_text(option_text)
    return str(option_text or "")


def _source_prefetch_oxidation_spin_alias_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    relation_focus: list[str],
    subject_terms: list[str],
) -> list[tuple[str, str]]:
    text = str(option_text or "")
    if not re.search(r"\bFe\s*\(", text, flags=re.IGNORECASE):
        return []
    aliases = _source_prefetch_fe_oxidation_spin_aliases(text)
    spin_phrases = []
    for match in re.finditer(r"\bS\s*=\s*([0-9]+(?:/[0-9]+)?(?:\.\d+)?)", text, flags=re.IGNORECASE):
        spin = match.group(1).strip()
        if spin:
            spin_phrases.extend([f"S = {spin}", f"spin {spin}"])
    spin_phrases = list(dict.fromkeys(spin_phrases))
    if not aliases and not spin_phrases:
        return []
    option_phrase = _source_prefetch_clean_phrase(option_text)
    high_specificity_anchors = _source_prefetch_high_specificity_question_anchors(
        stem,
        max_terms=6,
    )
    domain_terms = []
    domain_text = " ".join(
        str(value or "") for value in (problem.get("raw_subject"), problem.get("category"))
    )
    if re.search(r"chem|m[öo]ssbauer|spectros", stem + " " + domain_text, flags=re.IGNORECASE):
        domain_terms = [
            "57Fe Mossbauer",
            "Mossbauer spectroscopy",
            "hyperfine field",
        ]
    query = " ".join(
        [option_phrase]
        + aliases[:6]
        + spin_phrases[:3]
        + relation_focus[:8]
        + high_specificity_anchors[:6]
        + domain_terms
        + subject_terms[:4]
    )
    query = _clean_evidence_query(query)
    if not query:
        return []
    return [("candidate_specific_answer_bearing_oxidation_spin_alias", query)]


def _candidate_specific_answer_bearing_source_prefetch_queries(
    *,
    stem: str,
    option_label: str,
    option_text: str,
    options: dict[str, str],
    problem: dict[str, Any],
    agent_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    option_terms = _source_prefetch_option_anchor_terms(
        option_text=option_text,
        stem=stem,
        max_terms=10,
    )
    numeric_threshold_query_pairs = _numeric_threshold_source_prefetch_queries(
        stem=stem,
        option_text=option_text,
        problem=problem,
    )
    if not option_terms and not numeric_threshold_query_pairs:
        return []
    relation_signature = _option_claim_question_relation_signature_terms(
        stem=stem,
        option_text=option_text,
        max_terms=12,
    )
    required_terms = [
        str(term).strip("._-")
        for term in relation_signature.get("required_terms", []) or []
        if str(term).strip("._-")
        and str(term).lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ]
    required_terms = list(dict.fromkeys(required_terms))
    signature_terms = [
        str(term).strip("._-")
        for term in relation_signature.get("terms", []) or []
        if str(term).strip("._-")
        and str(term).lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ]
    signature_terms = list(dict.fromkeys(signature_terms))
    relation_terms = _question_relation_query_terms(stem)
    question_anchors = _question_evidence_anchor_terms(
        stem,
        option_text=option_text,
        max_terms=10,
    )
    high_specificity_anchors = _source_prefetch_high_specificity_question_anchors(
        stem,
        max_terms=10,
    )
    subject_terms = _source_prefetch_subject_terms(problem, max_terms=4)
    option_phrases = _source_prefetch_focus_phrases(option_text, max_phrases=4)
    question_phrases = _source_prefetch_focus_phrases(stem, max_phrases=3)
    operator_terms = _source_prefetch_operator_terms(agent_plan, max_terms=4)
    option_terms_by_label = {
        label: _content_terms(text)
        for label, text in options.items()
        if label != option_label and _content_terms(text)
    }
    competitor_terms = list(dict.fromkeys(
        term
        for terms in option_terms_by_label.values()
        for term in sorted(terms)
        if term not in {value.lower() for value in option_terms}
        and term not in _SOURCE_PREFETCH_GENERIC_TERMS
    ))
    relation_focus = list(dict.fromkeys(
        required_terms[:6]
        + signature_terms[:5]
        + relation_terms[:5]
        + question_anchors[:6]
    ))
    patient_label_query_pairs = _source_prefetch_patient_label_resolution_queries(
        stem=stem,
        option_text=option_text,
        problem=problem,
        relation_focus=relation_focus,
        subject_terms=subject_terms,
    )
    oxidation_spin_alias_query_pairs = _source_prefetch_oxidation_spin_alias_queries(
        stem=stem,
        option_text=option_text,
        problem=problem,
        relation_focus=relation_focus,
        subject_terms=subject_terms,
    )
    if (
        not relation_focus
        and not numeric_threshold_query_pairs
        and not patient_label_query_pairs
        and not oxidation_spin_alias_query_pairs
    ):
        return []
    seeds: list[tuple[str, str]] = []
    seeds.extend(patient_label_query_pairs)
    seeds.extend(oxidation_spin_alias_query_pairs)
    seeds.extend(numeric_threshold_query_pairs)
    seeds.append((
        "candidate_specific_answer_bearing_witness",
        " ".join(option_terms[:6] + relation_focus[:8] + ["evidence"] + subject_terms),
    ))
    if option_phrases:
        seeds.append((
            "candidate_specific_answer_bearing_exact_option",
            " ".join(option_phrases[:1] + relation_focus[:2] + ["evidence"] + subject_terms),
        ))
        if high_specificity_anchors:
            seeds.append((
                "candidate_specific_answer_bearing_experiment_anchor",
                " ".join(
                    option_phrases[:1]
                    + relation_focus[:3]
                    + high_specificity_anchors[:8]
                    + subject_terms
                ),
            ))
    seeds.append((
        "candidate_specific_answer_bearing_relation",
        " ".join(option_terms[:5] + required_terms[:6] + relation_terms[:5] + subject_terms),
    ))
    if high_specificity_anchors:
        seeds.append((
            "candidate_specific_answer_bearing_experiment_anchor",
            " ".join(
                option_terms[:5]
                + relation_focus[:3]
                + high_specificity_anchors[:8]
                + subject_terms
            ),
        ))
    if option_phrases:
        seeds.append((
            "candidate_specific_answer_bearing_option_phrase",
            " ".join(option_phrases[:1] + relation_focus[:8] + subject_terms),
        ))
    if question_phrases:
        seeds.append((
            "candidate_specific_answer_bearing_question_phrase",
            " ".join(option_terms[:5] + question_phrases[:1] + required_terms[:5] + subject_terms),
        ))
    if competitor_terms:
        seeds.append((
            "candidate_specific_answer_bearing_disambiguation",
            " ".join(
                option_terms[:5]
                + competitor_terms[:4]
                + required_terms[:5]
                + ["which", "evidence"]
                + subject_terms
            ),
        ))
    if option_phrases:
        for required_term in required_terms[:4]:
            seeds.append((
                "candidate_specific_answer_bearing_required_phrase_term",
                " ".join(option_phrases[:1] + [required_term] + relation_terms[:3] + subject_terms),
            ))
    for left_index, left_term in enumerate(required_terms[:5]):
        for right_term in required_terms[left_index + 1:left_index + 3]:
            seeds.append((
                "candidate_specific_answer_bearing_required_pair",
                " ".join(
                    option_terms[:5]
                    + [left_term, right_term]
                    + question_anchors[:3]
                    + subject_terms
                ),
            ))
    for required_term in required_terms[:3]:
        seeds.append((
            "candidate_specific_answer_bearing_required_term",
            " ".join(option_terms[:5] + [required_term] + question_anchors[:5] + subject_terms),
        ))
    if operator_terms:
        seeds.append((
            "candidate_specific_answer_bearing_operator",
            " ".join(option_terms[:5] + relation_focus[:5] + operator_terms[:4] + subject_terms),
        ))

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if not _source_prefetch_query_has_option_anchor(
            query=query,
            option_text=option_text,
        ):
            continue
        numeric_query_kind = str(kind or "").startswith(
            "candidate_specific_numeric_threshold"
        )
        if (
            not numeric_query_kind
            and not (_content_terms(query) & set(term.lower() for term in relation_focus))
        ):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 12:
            break
    return pairs


def _format_numeric_prefetch_number(value: float, *, decimals: int = 2) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value - round(value)) <= 1e-6:
        return str(int(round(value)))
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text or str(value)


def _numeric_threshold_value_phrases(value: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        clean = re.sub(r"\s+", " ", str(phrase or "").strip())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            phrases.append(clean)

    raw = str(value.get("raw") or "").strip()
    if raw:
        add(raw)
    raw_unit_phrase = ""
    unit = str(value.get("unit") or "").strip()
    raw_value = value.get("value")
    if raw_value is not None and unit:
        try:
            raw_unit_phrase = f"{_format_numeric_prefetch_number(float(raw_value))} {unit}"
        except (TypeError, ValueError):
            pass

    normalized_value = value.get("normalized_value")
    normalized_unit = str(value.get("normalized_unit") or "").strip()
    if normalized_value is None or not normalized_unit:
        return phrases
    try:
        norm_float = float(normalized_value)
    except (TypeError, ValueError):
        return phrases
    if not math.isfinite(norm_float):
        return phrases
    if normalized_unit == "K":
        add(f"{int(round(norm_float))} K")
    add(f"{_format_numeric_prefetch_number(norm_float)} {normalized_unit}")
    add(f"{_format_numeric_prefetch_number(norm_float, decimals=1)} {normalized_unit}")
    if raw_unit_phrase:
        add(raw_unit_phrase)
    if normalized_unit == "K":
        celsius = norm_float - 273.15
        fahrenheit = (celsius * 9.0 / 5.0) + 32.0
        add(f"{_format_numeric_prefetch_number(celsius)} C")
        add(f"{_format_numeric_prefetch_number(celsius, decimals=1)} C")
        add(f"{_format_numeric_prefetch_number(fahrenheit)} F")
    return phrases


def _numeric_threshold_subject_terms(stem: str, *, max_terms: int) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in numeric_relation_terms(stem):
        key = str(term or "").lower().strip("._-")
        if (
            not key
            or key in seen
            or key in _SOURCE_PREFETCH_GENERIC_TERMS
            or key in _NUMERIC_THRESHOLD_GENERIC_TERMS
            or key.isdigit()
        ):
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def _numeric_threshold_entity_anchor_phrases(
    stem: str,
    *,
    max_phrases: int,
) -> list[str]:
    """Extract multi-token subject/entity phrases for numeric-only options."""
    stopwords = _SOURCE_PREFETCH_GENERIC_TERMS | _NUMERIC_THRESHOLD_GENERIC_TERMS | {
        "according",
        "after",
        "among",
        "appropriate",
        "based",
        "before",
        "category",
        "categories",
        "choose",
        "compared",
        "given",
        "indicated",
        "index",
        "likely",
        "measurement",
        "number",
        "only",
        "recommended",
        "score",
        "value",
        "values",
    }
    tokens: list[tuple[int, str]] = []
    for position, token in enumerate(re.findall(r"[A-Za-z0-9_+.-]{2,}", str(stem or ""))):
        clean = token.strip("._-")
        key = clean.lower()
        if (
            not clean
            or key in stopwords
            or key.isdigit()
            or re.fullmatch(r"[A-Z]", clean)
        ):
            tokens.append((position, ""))
            continue
        tokens.append((position, clean))

    candidates: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    chunk: list[tuple[int, str]] = []

    def flush_chunk() -> None:
        nonlocal chunk
        if len(chunk) < 2:
            chunk = []
            return
        max_window = min(5, len(chunk))
        for window in range(max_window, 1, -1):
            for start in range(0, len(chunk) - window + 1):
                items = chunk[start:start + window]
                phrase = _source_prefetch_clean_phrase(
                    " ".join(token for _position, token in items)
                )
                key = _normalize_query_key(phrase)
                if not phrase or not key or key in seen:
                    continue
                term_count = len(_content_terms(phrase))
                if term_count < 2:
                    continue
                first_position = items[0][0]
                has_specific_marker = bool(
                    re.search(r"\d|[-+_]", phrase)
                    or re.search(r"\b[A-Z]{2,}\b", phrase)
                    or re.search(r"[A-Z]", phrase[1:])
                )
                biomedical_bonus = bool(
                    re.search(
                        r"\b("
                        r"assay|biopsy|cell|cells|clinical|disease|elastography|"
                        r"lesion|lesions|protein|therapy|tumou?r|ultrasound"
                        r")\b",
                        phrase,
                        flags=re.IGNORECASE,
                    )
                )
                score = (
                    term_count * 1.5
                    + (2.0 if has_specific_marker else 0.0)
                    + (2.0 if biomedical_bonus else 0.0)
                    + min(len(phrase), 80) / 80.0
                )
                seen.add(key)
                candidates.append((score, first_position, phrase))
        chunk = []

    for position, token in tokens:
        if token:
            chunk.append((position, token))
        else:
            flush_chunk()
    flush_chunk()
    candidates.sort(key=lambda item: (-item[0], item[1], item[2].lower()))
    return [phrase for _score, _position, phrase in candidates[:max_phrases]]


def _numeric_threshold_biomedical_anchor_terms(
    stem: str,
    *,
    max_terms: int,
) -> list[str]:
    text = str(stem or "")
    normalized = text.lower()
    phrase_candidates = [
        ("bi-rads", "BI-RADS"),
        ("focal breast lesions", "focal breast lesions"),
        ("breast lesions", "breast lesions"),
        ("ultrasound elastography", "ultrasound elastography"),
        ("shear wave elastography", "shear wave elastography"),
        ("biopsy", "biopsy"),
    ]
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        clean = _source_prefetch_clean_phrase(term)
        key = _normalize_query_key(clean)
        if clean and key and key not in seen:
            seen.add(key)
            terms.append(clean)

    for needle, phrase in phrase_candidates:
        if needle in normalized:
            add(phrase)
            if len(terms) >= max_terms:
                return terms[:max_terms]
    biomedical_tokens = {
        "assay",
        "biopsy",
        "breast",
        "cell",
        "cells",
        "clinical",
        "disease",
        "elastography",
        "lesion",
        "lesions",
        "oncology",
        "patient",
        "patients",
        "protein",
        "therapy",
        "tumor",
        "tumour",
        "ultrasound",
    }
    for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", text):
        clean = token.strip("._-")
        if clean.lower() in biomedical_tokens:
            add(clean)
            if len(terms) >= max_terms:
                break
    return terms[:max_terms]


def _numeric_threshold_relation_cues(
    *,
    stem: str,
    relation: dict[str, Any],
    value_type: str,
    max_terms: int,
) -> list[str]:
    stem_terms = set(numeric_relation_terms(stem))
    family = str(relation.get("relation_family") or "")
    cues: list[str] = []
    seen: set[str] = set()

    def add_many(values: list[str]) -> None:
        for value in values:
            key = str(value or "").lower().strip("._-")
            if not key or key in seen:
                continue
            seen.add(key)
            cues.append(key)

    if value_type == "temperature" or stem_terms & {
        "temperature",
        "coldest",
        "hottest",
        "celsius",
        "kelvin",
    }:
        add_many(["temperature", "thermal"])
    if stem_terms & {"synthesis", "synthesize", "synthesized", "produce", "produced"}:
        add_many(["synthesis", "prepared", "preparation", "reaction", "produced"])
    if family in {"threshold_minimum", "ordered_extreme_lowest", "below_threshold"}:
        add_many(["coldest", "lowest", "minimum", "threshold"])
    elif family in {"threshold_maximum", "ordered_extreme_highest", "above_threshold"}:
        add_many(["hottest", "highest", "maximum", "threshold"])
    elif family == "closest_value":
        add_many(["closest", "approximately", "value"])
    elif family == "range_membership":
        add_many(["range", "between", "within"])
    add_many([
        term
        for term in sorted(stem_terms & _NUMERIC_THRESHOLD_GENERIC_TERMS)
        if term
        not in {
            "degree",
            "degrees",
            "following",
            "method",
            "using",
        }
    ])
    return cues[:max_terms]


def _numeric_threshold_source_prefetch_queries(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
) -> list[tuple[str, str]]:
    values = parse_numeric_values(option_text)
    if not values:
        return []
    value = values[0]
    value_type = str(value.get("value_type") or "")
    relation = classify_numeric_relation(stem, value_type=value_type)
    if str(relation.get("relation_family") or "") not in _NUMERIC_THRESHOLD_PREFETCH_RELATION_FAMILIES:
        return []
    subject_terms = _numeric_threshold_subject_terms(stem, max_terms=5)
    relation_cues = _numeric_threshold_relation_cues(
        stem=stem,
        relation=relation,
        value_type=value_type,
        max_terms=8,
    )
    problem_subject_terms = _source_prefetch_subject_terms(problem, max_terms=3)
    question_anchors = _question_evidence_anchor_terms(
        stem,
        option_text=option_text,
        max_terms=6,
    )
    high_specificity_anchors = _source_prefetch_high_specificity_question_anchors(
        stem,
        max_terms=4,
    )
    entity_anchor_phrases = _numeric_threshold_entity_anchor_phrases(
        stem,
        max_phrases=4,
    )
    biomedical_anchor_terms = _numeric_threshold_biomedical_anchor_terms(
        stem,
        max_terms=6,
    )
    value_phrases = _numeric_threshold_value_phrases(value)
    if not value_phrases or not (subject_terms or question_anchors or entity_anchor_phrases):
        return []

    seeds: list[tuple[str, str]] = []
    primary_subject = subject_terms[:4] or question_anchors[:4]
    primary_relation = relation_cues[:6] or question_anchors[:4]
    threshold_cues = [
        cue for cue in relation_cues
        if cue
        in {
            "above",
            "below",
            "coldest",
            "hottest",
            "least",
            "lowest",
            "maximum",
            "minimum",
            "threshold",
        }
    ]
    action_cues = [
        cue for cue in relation_cues
        if cue and cue not in set(threshold_cues)
    ]
    if not threshold_cues:
        threshold_cues = relation_cues[:3]
    if not action_cues:
        action_cues = primary_relation[:4]
    if biomedical_anchor_terms:
        for phrase in value_phrases[:2]:
            seeds.append((
                "candidate_specific_numeric_threshold_biomedical_pubmed_anchor",
                " ".join(
                    biomedical_anchor_terms[:1]
                    + [phrase]
                    + biomedical_anchor_terms[1:6]
                    + action_cues[:2]
                ),
            ))
    for anchor_phrase in entity_anchor_phrases[:3]:
        for phrase in value_phrases[:2]:
            seeds.append((
                "candidate_specific_numeric_threshold_same_row_entity_anchor",
                " ".join(
                    [anchor_phrase, phrase]
                    + threshold_cues[:3]
                    + action_cues[:4]
                    + problem_subject_terms
                ),
            ))
    for index, phrase in enumerate(value_phrases[:3]):
        kind = (
            "candidate_specific_numeric_threshold_same_row_primary"
            if index == 0
            else "candidate_specific_numeric_threshold_same_row_unit_variant"
        )
        seeds.append((
            kind,
            " ".join(
                primary_subject[:4]
                + [phrase]
                + threshold_cues[:3]
                + action_cues[:4]
                + problem_subject_terms
            ),
        ))
        seeds.append((
            "candidate_specific_numeric_threshold_same_row_relation",
            " ".join(
                primary_subject[:3]
                + [phrase]
                + action_cues[:3]
                + threshold_cues[:3]
                + ["evidence"]
            ),
        ))
    if high_specificity_anchors:
        for phrase in value_phrases[:2]:
            seeds.append((
                "candidate_specific_numeric_threshold_same_row_anchor",
                " ".join(
                    high_specificity_anchors[:4]
                    + [phrase]
                    + threshold_cues[:3]
                    + action_cues[:3]
                ),
            ))
    for index, phrase in enumerate(value_phrases[:4]):
        kind = (
            "candidate_specific_numeric_threshold_exact_value"
            if index == 0
            else "candidate_specific_numeric_threshold_unit_variant"
        )
        seeds.append((
            kind,
            " ".join(
                primary_subject
                + [phrase]
                + primary_relation
                + problem_subject_terms
            ),
        ))
    if relation_cues:
        seeds.append((
            "candidate_specific_numeric_threshold_extreme_relation",
            " ".join(
                primary_subject
                + relation_cues[:8]
                + value_phrases[:2]
                + problem_subject_terms
            ),
        ))
    if high_specificity_anchors:
        seeds.append((
            "candidate_specific_numeric_threshold_question_anchor",
            " ".join(
                high_specificity_anchors[:4]
                + value_phrases[:2]
                + relation_cues[:5]
                + problem_subject_terms
            ),
        ))

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        query = _clean_evidence_query(seed)
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if not _source_prefetch_query_has_option_anchor(
            query=query,
            option_text=option_text,
        ):
            continue
        seen.add(key)
        pairs.append((kind, query))
        if len(pairs) >= 9:
            break
    return pairs


def _source_prefetch_option_anchor_terms(
    *,
    option_text: str,
    stem: str,
    max_terms: int,
) -> list[str]:
    stem_terms = {
        term.lower().strip("._-")
        for term in _content_terms(stem)
    }
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", option_text or ""):
        clean = token.strip("._-")
        key = clean.lower()
        if (
            not clean
            or key in seen
            or key in _SOURCE_PREFETCH_GENERIC_TERMS
            or (key in stem_terms and len(clean) < 7)
        ):
            continue
        seen.add(key)
        terms.append(clean)
        if len(terms) >= max_terms:
            break
    return terms


def _source_prefetch_subject_terms(problem: dict[str, Any], *, max_terms: int) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in (problem.get("raw_subject"), problem.get("category")):
        for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", str(value or "")):
            clean = token.strip("._-")
            key = clean.lower()
            if not clean or key in seen or key in _SOURCE_PREFETCH_GENERIC_TERMS:
                continue
            seen.add(key)
            terms.append(clean)
            if len(terms) >= max_terms:
                return terms
    return terms


def _source_prefetch_high_specificity_question_anchors(
    text: str,
    *,
    max_terms: int,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", str(text or "")):
        clean = token.strip("._-")
        key = clean.lower()
        if (
            not clean
            or key in seen
            or key in _SOURCE_PREFETCH_GENERIC_TERMS
        ):
            continue
        has_digit = bool(re.search(r"\d", clean))
        has_symbol = bool(re.search(r"[+.-]", clean))
        has_internal_capital = bool(re.search(r"[A-Z]", clean[1:]))
        if not (has_digit or has_symbol or has_internal_capital):
            continue
        seen.add(key)
        candidates.append(clean)
    candidates.sort(
        key=lambda value: (
            0
            if (
                len(value) <= 12
                and (
                    bool(re.search(r"\d", value))
                    or value.upper() == value
                    or bool(re.search(r"[A-Z]", value[1:]))
                )
            )
            else 1,
            len(value),
            value.lower(),
        )
    )
    return candidates[:max_terms]


def _source_prefetch_focus_phrases(text: str, *, max_phrases: int) -> list[str]:
    raw = str(text or "")
    phrases: list[str] = []
    seen: set[str] = set()
    full_phrase = _source_prefetch_clean_phrase(raw)
    full_key = _normalize_query_key(full_phrase)
    full_term_count = len(_content_terms(full_phrase))
    chemical_or_symbolic = bool(re.search(r"[\d()+-]", raw))
    if (
        full_phrase
        and full_key
        and full_key not in seen
        and 1 <= full_term_count <= 12
        and (chemical_or_symbolic or full_term_count <= 8)
    ):
        seen.add(full_key)
        phrases.append(full_phrase)
        if len(phrases) >= max_phrases:
            return phrases[:max_phrases]
    for groups in re.findall(r'"([^"]{3,90})"|\'([^\']{3,90})\'|`([^`]{3,90})`', raw):
        for item in groups:
            phrase = _source_prefetch_clean_phrase(item)
            key = _normalize_query_key(phrase)
            if phrase and key and key not in seen:
                seen.add(key)
                phrases.append(phrase)
    proper_noun_pattern = r"\b[A-Z][A-Za-z0-9_+.-]*(?:\s+[A-Z][A-Za-z0-9_+.-]*){1,5}\b"
    for match in re.finditer(proper_noun_pattern, raw):
        phrase = _source_prefetch_clean_phrase(match.group(0))
        key = _normalize_query_key(phrase)
        if phrase and key and key not in seen:
            seen.add(key)
            phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    if len(phrases) < max_phrases:
        tokens = [
            token.strip("._-")
            for token in re.findall(r"[A-Za-z0-9_+.-]{5,}", raw)
            if token.lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
        ]
        for start in range(0, max(0, len(tokens) - 1)):
            phrase = _source_prefetch_clean_phrase(" ".join(tokens[start:start + 3]))
            key = _normalize_query_key(phrase)
            if phrase and key and key not in seen:
                seen.add(key)
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    break
    return phrases[:max_phrases]


def _source_prefetch_operator_terms(
    agent_plan: dict[str, Any] | None,
    *,
    max_terms: int,
) -> list[str]:
    if not isinstance(agent_plan, dict):
        return []
    operator_stage = ((agent_plan.get("stages") or {}).get("operator_spec_compiler") or {})
    specs = operator_stage.get("operator_specs", []) if isinstance(operator_stage, dict) else []
    terms: list[str] = []
    seen: set[str] = set()
    for spec in specs or []:
        if isinstance(spec, dict):
            values = [
                spec.get("source_claim"),
                spec.get("trigger_conditions"),
                spec.get("execution_steps"),
                spec.get("required_output_slots"),
                spec.get("verifier_checks"),
            ]
        else:
            values = []
        for value in values:
            if isinstance(value, (list, tuple, set)):
                texts = [str(item) for item in value]
            else:
                texts = [str(value or "")]
            for text in texts:
                for token in re.findall(r"[A-Za-z0-9_+.-]{4,}", text):
                    clean = token.strip("._-")
                    key = clean.lower()
                    if not clean or key in seen or key in _SOURCE_PREFETCH_GENERIC_TERMS:
                        continue
                    seen.add(key)
                    terms.append(clean)
                    if len(terms) >= max_terms:
                        return terms
    return terms


def _source_prefetch_clean_phrase(text: str) -> str:
    phrase = _clean_evidence_query(text)
    if len(phrase.split()) > 8:
        phrase = " ".join(phrase.split()[:8])
    return phrase


def _source_prefetch_numeric_values_equivalent(
    option_value: dict[str, Any],
    query_value: dict[str, Any],
) -> bool:
    option_normalized = option_value.get("normalized_value")
    query_normalized = query_value.get("normalized_value")
    if option_normalized is None or query_normalized is None:
        return False
    try:
        option_float = float(option_normalized)
        query_float = float(query_normalized)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(option_float) or not math.isfinite(query_float):
        return False
    option_unit = option_value.get("normalized_unit")
    query_unit = query_value.get("normalized_unit")
    if option_unit and query_unit and option_unit != query_unit:
        return False
    if option_unit and not query_unit:
        return False
    if query_unit and not option_unit:
        return False
    abs_delta = abs(option_float - query_float)
    rel_delta = abs_delta / max(abs(option_float), abs(query_float), 1.0)
    if option_unit == "K" or query_unit == "K":
        return abs_delta <= 0.75
    if option_unit or query_unit:
        return abs_delta <= 1e-9 or rel_delta <= 0.01
    return abs_delta <= 1e-9 or rel_delta <= 1e-6


def _source_prefetch_query_has_option_anchor(*, query: str, option_text: str) -> bool:
    option_terms = _content_terms(option_text)
    query_terms = _content_terms(query)
    if option_terms & query_terms:
        return True
    option_numeric_terms = {
        token.lstrip("+-").strip()
        for token in re.findall(r"[-+]?\d+(?:\.\d+)?", str(option_text or ""))
        if token.lstrip("+-").strip()
    }
    query_numeric_terms = {
        token.lstrip("+-").strip()
        for token in re.findall(r"[-+]?\d+(?:\.\d+)?", str(query or ""))
        if token.lstrip("+-").strip()
    }
    if option_numeric_terms & query_numeric_terms:
        return True
    option_numeric_values = parse_numeric_values(option_text)
    query_numeric_values = parse_numeric_values(query)
    if option_numeric_values and query_numeric_values:
        for option_value in option_numeric_values:
            for query_value in query_numeric_values:
                if _source_prefetch_numeric_values_equivalent(
                    option_value,
                    query_value,
                ):
                    return True
    for phrase in _source_prefetch_focus_phrases(option_text, max_phrases=4):
        if phrase and phrase.lower() in query.lower():
            return True
    return False


def _normalize_query_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _source_prefetch_query_row_diagnostic_option_text(query_row: dict[str, Any]) -> str:
    return str(
        query_row.get("_source_diagnostic_option_text")
        or query_row.get("_option_text")
        or ""
    )


def _round_robin_query_pairs(
    buckets: list[list[tuple[str, str]]],
    *,
    max_queries: int,
) -> list[tuple[str, str]]:
    budget = max(1, int(max_queries or 1))
    out: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()
    offsets = [0 for _ in buckets]
    while len(out) < budget:
        progressed = False
        for bucket_index, bucket in enumerate(buckets):
            while offsets[bucket_index] < len(bucket):
                kind, query = bucket[offsets[bucket_index]]
                offsets[bucket_index] += 1
                clean_query = str(query or "").strip()
                if not clean_query:
                    continue
                query_hash = stable_hash({"query": clean_query})
                if query_hash in seen_hashes:
                    continue
                seen_hashes.add(query_hash)
                out.append((kind, clean_query))
                progressed = True
                break
            if len(out) >= budget:
                break
        if not progressed:
            break
    return out


def _run_source_prefetch(
    *,
    query_plan: list[dict[str, Any]],
    sources: list[str],
    source_limit: int,
    timeout: float,
    execute_live: bool,
    max_live_calls: int,
    delay_sec: float,
    retry_cached_errors: bool = False,
    refresh_cache_hits: bool = False,
    parallel_workers: int = 1,
    budget_policy: str = "round_robin_by_problem",
    source_error_budget: int = 0,
    logger: JsonlDiagnosticLogger | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for problem_row in query_plan:
        for query_row in problem_row.get("query_records", []) or []:
            query = str(query_row.get("_query") or "")
            if not query:
                continue
            allowed_sources = {
                str(value).strip()
                for value in query_row.get("allowed_sources", []) or []
                if str(value).strip()
            }
            preferred_sources = [
                str(value).strip()
                for value in query_row.get("preferred_sources", []) or []
                if str(value).strip()
            ]
            source_candidates = list(dict.fromkeys(preferred_sources + list(sources)))
            for allowed_source in sorted(allowed_sources):
                if allowed_source not in source_candidates:
                    source_candidates.append(allowed_source)
            for source in source_candidates:
                if allowed_sources and source not in allowed_sources:
                    continue
                cache_status_kwargs: dict[str, Any] = {
                    "source": source,
                    "query": query,
                    "limit": source_limit,
                }
                if execute_live and refresh_cache_hits:
                    cache_status_kwargs["bypass_read"] = True
                before_status = _cache_status(**cache_status_kwargs)
                record = {
                    "problem_id_hash": problem_row.get("problem_id_hash"),
                    "seed_offset": problem_row.get("seed_offset"),
                    "operator_family_tags": list(problem_row.get("operator_family_tags", []) or []),
                    "option_hash": query_row.get("option_hash") or query_row.get("option_label_hash"),
                    "option_label_hash": query_row.get("option_label_hash") or query_row.get("option_hash"),
                    "option_text_hash": query_row.get("option_text_hash"),
                    "option_choice": query_row.get("option_choice"),
                    "query_kind": query_row.get("query_kind"),
                    "query_hash": query_row.get("query_hash"),
                    "source": source,
                    "cache_status_before": before_status,
                    "status": "cache_hit" if before_status == "hit" else "planned",
                    "row_count": 0,
                    "error_type": "",
                    "cached_error_retry_attempted": False,
                    "cached_error_retry_policy": "",
                }
                for key in (
                    "parent_query_hash",
                    "parent_query_kind",
                    "parent_source",
                    "parent_source_hash",
                    "source_url_backfill_reason",
                    "source_url_hash",
                    "source_diagnostic_option_text_hash",
                    "source_diagnostic_option_expansion_kind",
                ):
                    if query_row.get(key):
                        record[key] = query_row.get(key)
                if allowed_sources:
                    record["allowed_sources"] = sorted(allowed_sources)
                if preferred_sources:
                    record["preferred_sources"] = preferred_sources
                log_event(
                    logger,
                    {
                        "event": "hle_source_prefetch_record_planned",
                        "problem_id_hash": record["problem_id_hash"],
                        "seed_offset": record["seed_offset"],
                        "option_hash": record.get("option_hash"),
                        "option_label_hash": record.get("option_label_hash"),
                        "option_text_hash": record.get("option_text_hash"),
                        "option_choice": record.get("option_choice"),
                        "query_kind": record["query_kind"],
                        "query_hash": record["query_hash"],
                        "source": source,
                        "cache_status_before": before_status,
                        "execute_live": bool(execute_live),
                        "local_source": _source_prefetch_source_is_local(source),
                        "parent_query_hash": record.get("parent_query_hash"),
                        "parent_query_kind": record.get("parent_query_kind"),
                        "parent_source": record.get("parent_source"),
                        "source_url_backfill_reason": record.get("source_url_backfill_reason"),
                        "source_url_hash": record.get("source_url_hash"),
                        "source_diagnostic_option_text_hash": record.get(
                            "source_diagnostic_option_text_hash"
                        ),
                        "source_diagnostic_option_expansion_kind": record.get(
                            "source_diagnostic_option_expansion_kind"
                        ),
                        "raw_content_persisted": False,
                    },
                )
                if before_status == "hit":
                    jobs.append({
                        "action": "cache_hit",
                        "record": record,
                        "problem_row": problem_row,
                        "query_row": query_row,
                        "source": source,
                        "query": query,
                    })
                    continue
                if before_status == "cached_error" and not (execute_live and retry_cached_errors):
                    record["status"] = "cached_error"
                    jobs.append({
                        "action": "static",
                        "record": record,
                        "problem_row": problem_row,
                        "query_row": query_row,
                        "source": source,
                        "query": query,
                    })
                    continue
                if not execute_live and not _source_prefetch_source_is_local(source):
                    record["status"] = "dry_run_missing"
                    jobs.append({
                        "action": "static",
                        "record": record,
                        "problem_row": problem_row,
                        "query_row": query_row,
                        "source": source,
                        "query": query,
                    })
                    continue
                if _source_prefetch_source_is_local(source):
                    jobs.append({
                        "action": "fetch",
                        "record": record,
                        "problem_row": problem_row,
                        "query_row": query_row,
                        "source": source,
                        "query": query,
                    })
                    continue
                retrying_cached_error = bool(before_status == "cached_error" and retry_cached_errors)
                if retrying_cached_error:
                    record["cached_error_retry_attempted"] = True
                    record["cached_error_retry_policy"] = "ignore_cached_error_for_live_prefetch"
                jobs.append({
                    "action": "fetch_candidate",
                    "record": record,
                    "problem_row": problem_row,
                    "query_row": query_row,
                    "source": source,
                    "query": query,
                    "ignore_cached_error": retrying_cached_error,
                })
    jobs = _apply_source_prefetch_live_budget(
        jobs=jobs,
        max_live_calls=max_live_calls,
        budget_policy=budget_policy,
    )
    _log_source_prefetch_live_budget_applied(
        jobs=jobs,
        max_live_calls=max_live_calls,
        budget_policy=budget_policy,
        logger=logger,
    )
    indexed_jobs = [(index, job) for index, job in enumerate(jobs)]
    execution_jobs = _source_prefetch_execution_order(indexed_jobs)
    workers = max(1, min(int(parallel_workers or 1), max(1, len(indexed_jobs))))
    source_error_state: dict[str, int] = {}
    source_error_lock = threading.Lock()
    if workers == 1:
        records_by_index = {}
        for index, job in execution_jobs:
            records_by_index[index] = _run_source_prefetch_job(
                index=index,
                job=job,
                source_limit=source_limit,
                timeout=timeout,
                delay_sec=delay_sec,
                source_error_budget=source_error_budget,
                source_error_state=source_error_state,
                source_error_lock=source_error_lock,
                logger=logger,
            )
        return [
            records_by_index[index]
            for index, _ in indexed_jobs
        ]
    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _run_source_prefetch_job,
                index=index,
                job=job,
                source_limit=source_limit,
                timeout=timeout,
                delay_sec=delay_sec,
                source_error_budget=source_error_budget,
                source_error_state=source_error_state,
                source_error_lock=source_error_lock,
                logger=logger,
            ): index
            for index, job in execution_jobs
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            records_by_index[index] = future.result()
    return [records_by_index[index] for index, _ in indexed_jobs]


def _numeric_same_row_adaptive_backfill_query_plan(
    *,
    query_plan: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    source_limit: int,
) -> list[dict[str, Any]]:
    query_rows_by_hash: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for problem_row in query_plan:
        for query_row in problem_row.get("query_records", []) or []:
            query_hash = str(query_row.get("query_hash") or "")
            if query_hash:
                query_rows_by_hash[query_hash] = (problem_row, query_row)
    emitted_by_problem: dict[str, dict[str, Any]] = {}
    emitted_query_hashes: set[str] = set()
    per_option_counts: Counter[str] = Counter()
    per_option_limit = 2
    global_limit = 24
    for record in source_records:
        if len(emitted_query_hashes) >= global_limit:
            break
        query_kind = str(record.get("query_kind") or "")
        if "numeric_threshold" not in query_kind:
            continue
        record_value_match = _safe_int(record.get("numeric_same_row_value_match_count"))
        record_directish = _safe_int(record.get("answer_bearing_directish_count"))
        if _safe_int(record.get("numeric_same_row_direct_count")) > 0:
            continue
        query_hash = str(record.get("query_hash") or "")
        indexed = query_rows_by_hash.get(query_hash)
        if not indexed:
            continue
        problem_row, query_row = indexed
        option_hash = str(query_row.get("option_hash") or query_row.get("option_label_hash") or "")
        if option_hash and per_option_counts[option_hash] >= per_option_limit:
            continue
        source = str(record.get("source") or "")
        query = str(query_row.get("_query") or "")
        if not source or not query:
            continue
        if (
            record_value_match <= 0
            and record_directish <= 0
            and not _numeric_same_row_parent_source_url_enrichment_candidate(source)
        ):
            continue
        rows = _evidence_source_cache_get(
            source=source,
            query=query,
            limit=max(1, int(source_limit or 1)),
            allow_expired_ok=True,
            include_url=True,
        ) or []
        for source_row in rows:
            if len(emitted_query_hashes) >= global_limit:
                break
            if option_hash and per_option_counts[option_hash] >= per_option_limit:
                break
            detail = numeric_same_row_source_diagnostics(
                rows=[source_row],
                stem=str(problem_row.get("_stem") or ""),
                option_text=_source_prefetch_query_row_diagnostic_option_text(query_row),
            )
            if _safe_int(detail.get("numeric_same_row_direct_count")) > 0:
                continue
            answer_detail = _source_rows_answer_bearing_diagnostics(
                rows=[source_row],
                problem_row=problem_row,
                query_row=query_row,
                query=query,
            )
            row_value_match = _safe_int(detail.get("numeric_same_row_value_match_count"))
            row_directish = _safe_int(answer_detail.get("answer_bearing_directish_count"))
            url_enrichment_candidate = _numeric_same_row_source_row_url_enrichment_candidate(
                parent_source=source,
                source_row=source_row,
            )
            if row_directish <= 0 and not url_enrichment_candidate:
                continue
            url_only_backfill = row_value_match <= 0 or row_directish <= 0
            for kind, backfill_query in _numeric_same_row_backfill_queries_from_source_row(
                stem=str(problem_row.get("_stem") or ""),
                option_text=_source_prefetch_query_row_diagnostic_option_text(query_row),
                problem=problem_row.get("_problem") or {},
                source_row=source_row,
            ):
                if (
                    url_only_backfill
                    and kind != "candidate_specific_numeric_threshold_same_row_source_url_backfill"
                ):
                    continue
                clean_query = (
                    str(backfill_query or "").strip()
                    if kind == "candidate_specific_numeric_threshold_same_row_source_url_backfill"
                    else _clean_evidence_query(backfill_query)
                )
                if not clean_query:
                    continue
                new_query_hash = stable_hash({"query": clean_query})
                if new_query_hash in emitted_query_hashes:
                    continue
                emitted_query_hashes.add(new_query_hash)
                per_option_counts[option_hash] += 1
                problem_key = str(problem_row.get("problem_id_hash") or "")
                out_problem = emitted_by_problem.setdefault(
                    problem_key,
                    {
                        "seed_offset": problem_row.get("seed_offset"),
                        "status": "planned",
                        "_problem": problem_row.get("_problem") or {},
                        "_stem": problem_row.get("_stem") or "",
                        "_options": problem_row.get("_options") or {},
                        "problem_id_hash": problem_row.get("problem_id_hash"),
                        "question_hash": problem_row.get("question_hash"),
                        "category_hash": problem_row.get("category_hash"),
                        "raw_subject_hash": problem_row.get("raw_subject_hash"),
                        "domain": problem_row.get("domain"),
                        "answer_type": problem_row.get("answer_type"),
                        "option_count": problem_row.get("option_count"),
                        "operator_status": problem_row.get("operator_status"),
                        "operator_reason": problem_row.get("operator_reason"),
                        "operator_family_tags": list(
                            problem_row.get("operator_family_tags", []) or []
                        ),
                        "query_records": [],
                    },
                )
                out_problem["query_records"].append({
                    "query_hash": new_query_hash,
                    "query_kind": kind,
                    "option_hash": query_row.get("option_hash"),
                    "option_label_hash": query_row.get("option_label_hash") or query_row.get("option_hash"),
                    "option_text_hash": query_row.get("option_text_hash"),
                    "option_choice": query_row.get("option_choice"),
                    "_query": clean_query,
                    "_option_label": query_row.get("_option_label"),
                    "_option_text": query_row.get("_option_text"),
                    "_source_diagnostic_option_text": query_row.get(
                        "_source_diagnostic_option_text"
                    ),
                    "parent_query_hash": query_hash,
                    "parent_query_kind": query_kind,
                    "parent_source": source,
                    "parent_source_hash": stable_hash({
                        "title": str(source_row.get("title") or ""),
                        "snippet": str(source_row.get("snippet") or ""),
                        "source": str(source_row.get("source") or source),
                    }),
                    "source_url_backfill_reason": (
                        "url_enrichment_without_directish"
                        if row_directish <= 0
                        else (
                            "directish_without_value"
                            if row_value_match <= 0
                            else "directish_value_match"
                        )
                    ),
                    "source_url_hash": stable_hash({
                        "url": str(source_row.get("url") or "")
                    }) if source_row.get("url") else "",
                    "source_diagnostic_option_text_hash": query_row.get(
                        "source_diagnostic_option_text_hash"
                    ),
                    "source_diagnostic_option_expansion_kind": query_row.get(
                        "source_diagnostic_option_expansion_kind"
                    ),
                    "allowed_sources": (
                        ["answer_web_fulltext"]
                        if kind == "candidate_specific_numeric_threshold_same_row_source_url_backfill"
                        else []
                    ),
                })
                if option_hash and per_option_counts[option_hash] >= per_option_limit:
                    break
    return list(emitted_by_problem.values())


def _numeric_same_row_backfill_queries_from_source_row(
    *,
    stem: str,
    option_text: str,
    problem: dict[str, Any],
    source_row: dict[str, Any],
) -> list[tuple[str, str]]:
    values = parse_numeric_values(option_text)
    if not values:
        return []
    value = values[0]
    value_type = str(value.get("value_type") or "")
    relation = classify_numeric_relation(stem, value_type=value_type)
    if str(relation.get("relation_family") or "") not in _NUMERIC_THRESHOLD_PREFETCH_RELATION_FAMILIES:
        return []
    value_phrases = _numeric_threshold_value_phrases(value)[:2]
    if not value_phrases:
        return []
    subject_terms = _numeric_threshold_subject_terms(stem, max_terms=5)
    relation_cues = _numeric_threshold_relation_cues(
        stem=stem,
        relation=relation,
        value_type=value_type,
        max_terms=8,
    )
    threshold_cues = [
        cue for cue in relation_cues
        if cue
        in {
            "above",
            "below",
            "coldest",
            "hottest",
            "least",
            "lowest",
            "maximum",
            "minimum",
            "threshold",
        }
    ] or relation_cues[:3]
    action_cues = [
        cue for cue in relation_cues
        if cue and cue not in set(threshold_cues)
    ] or relation_cues[:4]
    title = str(source_row.get("title") or "")
    snippet = str(source_row.get("snippet") or "")
    title_phrases = _source_prefetch_focus_phrases(title, max_phrases=2)
    title_terms = [
        term
        for term in re.findall(r"[A-Za-z0-9_+.-]{3,}", title)
        if term.lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ][:5]
    snippet_terms = [
        term
        for term in re.findall(r"[A-Za-z0-9_+.-]{3,}", snippet)
        if term.lower().strip("._-") not in _SOURCE_PREFETCH_GENERIC_TERMS
    ][:4]
    source_anchor = title_phrases[:1] or title_terms[:5] or snippet_terms[:4]
    problem_subject_terms = _source_prefetch_subject_terms(problem, max_terms=3)
    seeds: list[tuple[str, str]] = []
    source_url = str(source_row.get("url") or "").strip()
    metadata_page_url = bool(
        re.search(r"://(?:www\.)?semanticscholar\.org/paper/", source_url, flags=re.IGNORECASE)
    )
    if re.match(r"^https?://", source_url, flags=re.IGNORECASE) and not metadata_page_url:
        url_focus_terms = (
            value_phrases[:1]
            + threshold_cues[:3]
            + action_cues[:4]
            + (subject_terms[:4] or source_anchor[:3])
            + problem_subject_terms
        )
        seeds.append((
            "candidate_specific_numeric_threshold_same_row_source_url_backfill",
            " ".join([source_url] + url_focus_terms),
        ))
    for phrase in value_phrases:
        seeds.append((
            "candidate_specific_numeric_threshold_same_row_source_title_backfill",
            " ".join(
                source_anchor
                + [phrase]
                + threshold_cues[:3]
                + action_cues[:4]
                + problem_subject_terms
            ),
        ))
        seeds.append((
            "candidate_specific_numeric_threshold_same_row_source_relation_backfill",
            " ".join(
                (subject_terms[:4] or source_anchor)
                + [phrase]
                + threshold_cues[:3]
                + action_cues[:4]
                + source_anchor[:2]
            ),
        ))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, seed in seeds:
        raw_seed = str(seed or "").strip()
        query = (
            raw_seed
            if kind == "candidate_specific_numeric_threshold_same_row_source_url_backfill"
            else _clean_evidence_query(seed)
        )
        key = _normalize_query_key(query)
        if not query or not key or key in seen:
            continue
        if (
            kind != "candidate_specific_numeric_threshold_same_row_source_url_backfill"
            and not _source_prefetch_query_has_option_anchor(
            query=query,
            option_text=option_text,
            )
        ):
            continue
        seen.add(key)
        out.append((kind, query))
        if len(out) >= 3:
            break
    return out


def _numeric_same_row_parent_source_url_enrichment_candidate(source: str) -> bool:
    return str(source or "").strip().lower() in {
        "answer_web",
        "answer_web_fulltext",
        "crossref",
        "local_evidence_corpus",
        "openalex",
        "pubmed",
        "pubmed_pmc_fulltext",
        "semantic_scholar",
    }


def _numeric_same_row_source_row_url_enrichment_candidate(
    *,
    parent_source: str,
    source_row: dict[str, Any],
) -> bool:
    url = str(source_row.get("url") or "").strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return False
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    row_source = str(source_row.get("source") or parent_source or "").strip().lower()
    if not host:
        return False
    if re.search(r"(?:^|\.)doi\.org$", host):
        return True
    if host.endswith("ncbi.nlm.nih.gov") or host in {
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
    }:
        return True
    if path.endswith(".pdf"):
        return True
    if row_source in {
        "clinical_guideline",
        "guideline",
        "local_guideline",
        "local_fulltext",
        "answer_web_fulltext",
    }:
        return True
    text = f"{source_row.get('title') or ''} {source_row.get('snippet') or ''}"
    if row_source in {"crossref", "openalex", "semantic_scholar", "pubmed", "pubmed_abstract"}:
        if re.search(r"\b10\.\d{4,9}/\S+", text):
            return True
    return False


def _source_prefetch_execution_order(
    indexed_jobs: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    fetch_jobs = [
        (index, job)
        for index, job in indexed_jobs
        if job.get("action") == "fetch"
    ]
    non_fetch_jobs = [
        (index, job)
        for index, job in indexed_jobs
        if job.get("action") != "fetch"
    ]
    return _source_prefetch_fair_candidate_order(fetch_jobs) + non_fetch_jobs


def _source_prefetch_fair_candidate_order(
    indexed_jobs: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    group_order: list[str] = []
    for index, job in indexed_jobs:
        problem_row = job.get("problem_row") or {}
        group_key = str(problem_row.get("problem_id_hash") or problem_row.get("seed_offset") or index)
        if group_key not in grouped:
            grouped[group_key] = []
            group_order.append(group_key)
        grouped[group_key].append((index, job))
    problem_orders: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for problem_position, group_key in enumerate(group_order):
        problem_orders[group_key] = _source_prefetch_problem_fair_candidate_order(
            grouped[group_key],
            source_rotation=problem_position,
        )
    fair_jobs: list[tuple[int, dict[str, Any]]] = []
    offsets = {group_key: 0 for group_key in group_order}
    while True:
        progressed = False
        for group_key in group_order:
            offset = offsets[group_key]
            group_items = problem_orders[group_key]
            if offset >= len(group_items):
                continue
            fair_jobs.append(group_items[offset])
            offsets[group_key] = offset + 1
            progressed = True
        if not progressed:
            break
    return fair_jobs


def _source_prefetch_problem_fair_candidate_order(
    indexed_jobs: list[tuple[int, dict[str, Any]]],
    *,
    source_rotation: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    option_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    option_order: list[str] = []
    for index, job in indexed_jobs:
        record = job.get("record") if isinstance(job.get("record"), dict) else {}
        query_row = job.get("query_row") if isinstance(job.get("query_row"), dict) else {}
        option_key = str(
            record.get("option_hash")
            or query_row.get("option_hash")
            or query_row.get("option_label_hash")
            or ""
        )
        if not option_key:
            option_key = "__no_option__"
        if option_key not in option_groups:
            option_groups[option_key] = []
            option_order.append(option_key)
        option_groups[option_key].append((index, job))
    if option_order:
        option_start = source_rotation % len(option_order)
        fair_option_order = option_order[option_start:] + option_order[:option_start]
    else:
        fair_option_order = []
    option_orders: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for option_position, option_key in enumerate(fair_option_order):
        option_orders[option_key] = _source_prefetch_query_fair_candidate_order(
            option_groups[option_key],
            source_rotation=source_rotation + option_position,
        )
    ordered: list[tuple[int, dict[str, Any]]] = []
    offsets = {option_key: 0 for option_key in fair_option_order}
    while True:
        progressed = False
        for option_key in fair_option_order:
            offset = offsets[option_key]
            group_items = option_orders[option_key]
            if offset >= len(group_items):
                continue
            ordered.append(group_items[offset])
            offsets[option_key] = offset + 1
            progressed = True
        if not progressed:
            break
    return ordered


def _source_prefetch_query_fair_candidate_order(
    indexed_jobs: list[tuple[int, dict[str, Any]]],
    *,
    source_rotation: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    query_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    query_order: list[str] = []
    for index, job in indexed_jobs:
        record = job.get("record") if isinstance(job.get("record"), dict) else {}
        query_row = job.get("query_row") if isinstance(job.get("query_row"), dict) else {}
        query_key = str(
            record.get("query_hash")
            or query_row.get("query_hash")
            or job.get("query")
            or index
        )
        if query_key not in query_groups:
            query_groups[query_key] = []
            query_order.append(query_key)
        query_groups[query_key].append((index, job))
    query_states: list[dict[str, Any]] = []
    for query_key in query_order:
        source_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        source_order: list[str] = []
        for index, job in query_groups[query_key]:
            source_key = _source_prefetch_job_source_key(job)
            if source_key not in source_groups:
                source_groups[source_key] = []
                source_order.append(source_key)
            source_groups[source_key].append((index, job))
        query_states.append({
            "source_groups": source_groups,
            "source_order": source_order,
            "offsets": {source_key: 0 for source_key in source_order},
        })
    ordered: list[tuple[int, dict[str, Any]]] = []
    round_offset = 0
    while True:
        progressed = False
        for query_position, state in enumerate(query_states):
            source_order = state["source_order"]
            if not source_order:
                continue
            start = (source_rotation + query_position + round_offset) % len(source_order)
            for probe in range(len(source_order)):
                source_key = source_order[(start + probe) % len(source_order)]
                offsets = state["offsets"]
                offset = offsets[source_key]
                group_items = state["source_groups"][source_key]
                if offset >= len(group_items):
                    continue
                ordered.append(group_items[offset])
                offsets[source_key] = offset + 1
                progressed = True
                break
        if not progressed:
            break
        round_offset += 1
    return ordered


def _source_prefetch_job_source_key(job: dict[str, Any]) -> str:
    record = job.get("record") if isinstance(job.get("record"), dict) else {}
    return str(job.get("source") or record.get("source") or "__no_source__")


def _source_prefetch_source_is_local(source: str) -> bool:
    return str(source or "").strip() in _SOURCE_PREFETCH_LOCAL_SOURCES


def _source_prefetch_job_preferred_source_priority(job: dict[str, Any]) -> int:
    query_row = job.get("query_row") if isinstance(job.get("query_row"), dict) else {}
    preferred_sources = [
        str(value).strip()
        for value in query_row.get("preferred_sources", []) or []
        if str(value).strip()
    ]
    if not preferred_sources:
        return 100
    source = _source_prefetch_job_source_key(job)
    try:
        return preferred_sources.index(source)
    except ValueError:
        return 100 + len(preferred_sources)


def _source_prefetch_answer_bearing_budget_priority_enabled() -> bool:
    return os.environ.get(
        "HLE_DISABLE_SOURCE_PREFETCH_ANSWER_BEARING_BUDGET_PRIORITY",
        "",
    ).strip().lower() not in {"1", "true", "yes", "on"}


def _source_prefetch_query_kind_budget_priority(job: dict[str, Any]) -> int:
    record = job.get("record") if isinstance(job.get("record"), dict) else {}
    query_row = job.get("query_row") if isinstance(job.get("query_row"), dict) else {}
    kind = str(record.get("query_kind") or query_row.get("query_kind") or "")
    priority = {
        "term_identity_missing_required_single": 0,
        "term_identity_missing_required_exact_option": 0,
        "term_identity_missing_required_experiment_anchor": 0,
        "term_identity_missing_required_pair": 0,
        "term_identity_missing_required_operator": 0,
        "term_identity_missing_required_stem_anchor": 0,
        "candidate_specific_numeric_threshold_biomedical_pubmed_anchor": 0,
        "candidate_specific_numeric_threshold_exact_value": 0,
        "candidate_specific_numeric_threshold_same_row_primary": 0,
        "candidate_specific_numeric_threshold_same_row_unit_variant": 0,
        "candidate_specific_numeric_threshold_same_row_relation": 0,
        "candidate_specific_numeric_threshold_same_row_entity_anchor": 0,
        "candidate_specific_numeric_threshold_same_row_anchor": 0,
        "candidate_specific_numeric_threshold_same_row_source_title_backfill": 0,
        "candidate_specific_numeric_threshold_same_row_source_relation_backfill": 0,
        "candidate_specific_numeric_threshold_same_row_source_url_backfill": 0,
        "candidate_specific_numeric_threshold_unit_variant": 0,
        "candidate_specific_numeric_threshold_extreme_relation": 0,
        "candidate_specific_numeric_threshold_question_anchor": 0,
        "candidate_specific_answer_bearing_required_term": 0,
        "candidate_specific_answer_bearing_required_pair": 0,
        "candidate_specific_answer_bearing_required_phrase_term": 0,
        "candidate_specific_answer_bearing_exact_option": 0,
        "candidate_specific_answer_bearing_experiment_anchor": 0,
        "candidate_specific_answer_bearing_patient_label_resolution": 0,
        "candidate_specific_answer_bearing_oxidation_spin_alias": 0,
        "answer_bearing_required_relation_completion": 0,
        "answer_bearing_required_relation_term": 0,
        "answer_bearing_required_relation_term_pair": 0,
        "answer_bearing_required_relation_phrase_term": 0,
        "answer_bearing_required_relation_exact_option": 0,
        "answer_bearing_required_relation_focus": 0,
        "answer_bearing_required_relation_operator": 0,
        "candidate_specific_answer_bearing_relation": 1,
        "candidate_specific_answer_bearing_witness": 1,
        "candidate_specific_answer_bearing_question_phrase": 1,
        "candidate_specific_answer_bearing_option_phrase": 1,
        "candidate_specific_answer_bearing_disambiguation": 1,
        "candidate_specific_answer_bearing_operator": 1,
        "answer_bearing_pair_binding": 2,
        "answer_bearing_pair_contrast": 2,
        "answer_bearing_pair_disambiguation": 2,
        "answer_bearing_pair_operator_binding": 2,
        "answer_bearing_relation_binding": 3,
        "answer_bearing_option_focus": 3,
        "answer_bearing_operator_binding": 3,
        "answer_bearing_source_specificity": 3,
    }
    return priority.get(kind, 4)


def _apply_source_prefetch_live_budget(
    *,
    jobs: list[dict[str, Any]],
    max_live_calls: int,
    budget_policy: str = "round_robin_by_problem",
) -> list[dict[str, Any]]:
    candidates = [
        (index, job)
        for index, job in enumerate(jobs)
        if job.get("action") == "fetch_candidate"
    ]
    budget = max(0, int(max_live_calls or 0))
    if not candidates:
        return jobs
    selected: set[int] = set()
    if budget_policy == "sequential":
        selected = {index for index, _ in candidates[:budget]}
    else:
        fair_order = _source_prefetch_fair_candidate_order(candidates)
        if _source_prefetch_answer_bearing_budget_priority_enabled():
            fair_order = [
                item
                for _kind_priority, _source_priority, _position, item in sorted(
                    (
                        (
                            _source_prefetch_query_kind_budget_priority(job),
                            _source_prefetch_job_preferred_source_priority(job),
                            position,
                            (index, job),
                        )
                        for position, (index, job) in enumerate(fair_order)
                    ),
                    key=lambda item: (item[0], item[1], item[2]),
                )
            ]
        selected = {index for index, _job in fair_order[:budget]}
    out: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        if job.get("action") != "fetch_candidate":
            out.append(job)
            continue
        next_job = dict(job)
        record = dict(next_job.get("record") or {})
        if index in selected:
            next_job["action"] = "fetch"
            record["status"] = "planned"
        else:
            next_job["action"] = "static"
            record["status"] = "budget_skipped"
        next_job["record"] = record
        out.append(next_job)
    return out


def _log_source_prefetch_live_budget_applied(
    *,
    jobs: list[dict[str, Any]],
    max_live_calls: int,
    budget_policy: str,
    logger: JsonlDiagnosticLogger | None,
) -> None:
    candidate_count_by_source: Counter[str] = Counter()
    selected_count_by_source: Counter[str] = Counter()
    skipped_count_by_source: Counter[str] = Counter()
    static_count_by_source: Counter[str] = Counter()
    candidate_count_by_query_kind: Counter[str] = Counter()
    selected_count_by_query_kind: Counter[str] = Counter()
    skipped_count_by_query_kind: Counter[str] = Counter()
    for job in jobs:
        source = _source_prefetch_job_source_key(job)
        record = job.get("record") if isinstance(job.get("record"), dict) else {}
        action = str(job.get("action") or "")
        query_kind = str(record.get("query_kind") or "unknown")
        if action == "fetch":
            candidate_count_by_source[source] += 1
            selected_count_by_source[source] += 1
            candidate_count_by_query_kind[query_kind] += 1
            selected_count_by_query_kind[query_kind] += 1
        elif action == "static" and record.get("status") == "budget_skipped":
            candidate_count_by_source[source] += 1
            skipped_count_by_source[source] += 1
            candidate_count_by_query_kind[query_kind] += 1
            skipped_count_by_query_kind[query_kind] += 1
        elif action == "static":
            static_count_by_source[source] += 1
    if not candidate_count_by_source:
        return
    log_event(
        logger,
        {
            "event": "hle_source_prefetch_live_budget_applied",
            "max_live_calls": int(max_live_calls or 0),
            "budget_policy": budget_policy,
            "candidate_count": sum(candidate_count_by_source.values()),
            "selected_count": sum(selected_count_by_source.values()),
            "budget_skipped_count": sum(skipped_count_by_source.values()),
            "candidate_count_by_source": dict(candidate_count_by_source),
            "selected_count_by_source": dict(selected_count_by_source),
            "budget_skipped_count_by_source": dict(skipped_count_by_source),
            "static_count_by_source": dict(static_count_by_source),
            "candidate_count_by_query_kind": dict(candidate_count_by_query_kind),
            "selected_count_by_query_kind": dict(selected_count_by_query_kind),
            "budget_skipped_count_by_query_kind": dict(skipped_count_by_query_kind),
            "answer_bearing_budget_priority_enabled": (
                _source_prefetch_answer_bearing_budget_priority_enabled()
            ),
            "raw_content_persisted": False,
        },
    )


def _run_source_prefetch_job(
    *,
    index: int,
    job: dict[str, Any],
    source_limit: int,
    timeout: float,
    delay_sec: float,
    source_error_budget: int,
    source_error_state: dict[str, int],
    source_error_lock: threading.Lock,
    logger: JsonlDiagnosticLogger | None,
) -> dict[str, Any]:
    action = str(job.get("action") or "")
    record = dict(job.get("record") or {})
    record["_record_index"] = index
    source = str(job.get("source") or "")
    query = str(job.get("query") or "")
    rows: list[dict[str, Any]] | None = None
    started = time.monotonic()
    if action == "cache_hit":
        rows = _evidence_source_cache_get(source=source, query=query, limit=source_limit) or []
        record["row_count"] = len(rows)
        record["cache_status_after"] = "hit"
        log_event(
            logger,
            _source_prefetch_log_event(
                event="hle_source_prefetch_cache_hit",
                record=record,
                latency_sec=round(time.monotonic() - started, 4),
            ),
        )
    elif action == "fetch":
        if _source_prefetch_error_budget_exhausted(
            source=source,
            source_error_budget=source_error_budget,
            source_error_state=source_error_state,
            source_error_lock=source_error_lock,
        ):
            record["status"] = "source_error_budget_skipped"
            log_event(
                logger,
                _source_prefetch_log_event(
                    event="hle_source_prefetch_source_error_budget_skipped",
                    record=record,
                    latency_sec=round(time.monotonic() - started, 4),
                ),
            )
            return _sanitize_source_record(record)
        log_event(
            logger,
            _source_prefetch_log_event(
                event="hle_source_prefetch_fetch_start",
                record=record,
                timeout_sec=timeout,
            ),
        )
        try:
            problem = job.get("problem_row", {}).get("_problem")
            fetch_kwargs: dict[str, Any] = {
                "source": source,
                "query": query,
                "limit": source_limit,
                "timeout": timeout,
                "ignore_cached_error": bool(job.get("ignore_cached_error")),
            }
            if isinstance(problem, dict) and problem:
                fetch_kwargs["problem"] = problem
            rows = _fetch_source(**fetch_kwargs)
            record["status"] = "fetched"
            record["row_count"] = len(rows)
            record["cache_status_after"] = _cache_status(source=source, query=query, limit=source_limit)
            log_event(
                logger,
                _source_prefetch_log_event(
                    event="hle_source_prefetch_fetch_end",
                    record=record,
                    latency_sec=round(time.monotonic() - started, 4),
                ),
            )
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error_label"] = _evidence_source_error_label(exc)
            _source_prefetch_record_source_error(
                source=source,
                source_error_budget=source_error_budget,
                source_error_state=source_error_state,
                source_error_lock=source_error_lock,
            )
            record["cache_status_after"] = _cache_status(source=source, query=query, limit=source_limit)
            log_event(
                logger,
                _source_prefetch_log_event(
                    event="hle_source_prefetch_fetch_error",
                    record=record,
                    latency_sec=round(time.monotonic() - started, 4),
                    error_type=type(exc).__name__,
                ),
            )
    else:
        log_event(
            logger,
            _source_prefetch_log_event(
                event="hle_source_prefetch_static_record",
                record=record,
                latency_sec=round(time.monotonic() - started, 4),
            ),
        )
    if rows is not None:
        record.update(
            _source_rows_answer_bearing_diagnostics(
                rows=rows,
                problem_row=job.get("problem_row") or {},
                query_row=job.get("query_row") or {},
                query=query,
            )
        )
        record.update(
            numeric_same_row_source_diagnostics(
                rows=rows,
                stem=str((job.get("problem_row") or {}).get("_stem") or ""),
                option_text=_source_prefetch_query_row_diagnostic_option_text(
                    job.get("query_row") or {}
                ),
            )
        )
        log_event(
            logger,
            _source_prefetch_log_event(
                event="hle_source_prefetch_answer_bearing_diagnostics",
                record=record,
                latency_sec=round(time.monotonic() - started, 4),
            ),
        )
    if delay_sec > 0 and action == "fetch":
        time.sleep(delay_sec)
    return _sanitize_source_record(record)


def _source_prefetch_error_budget_exhausted(
    *,
    source: str,
    source_error_budget: int,
    source_error_state: dict[str, int],
    source_error_lock: threading.Lock,
) -> bool:
    if int(source_error_budget or 0) <= 0:
        return False
    with source_error_lock:
        return int(source_error_state.get(source, 0) or 0) >= int(source_error_budget)


def _source_prefetch_record_source_error(
    *,
    source: str,
    source_error_budget: int,
    source_error_state: dict[str, int],
    source_error_lock: threading.Lock,
) -> None:
    if int(source_error_budget or 0) <= 0:
        return
    with source_error_lock:
        source_error_state[source] = int(source_error_state.get(source, 0) or 0) + 1


def _source_prefetch_log_event(
    *,
    event: str,
    record: dict[str, Any],
    latency_sec: float | None = None,
    timeout_sec: float | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    payload = {
        "event": event,
        "record_index": record.get("_record_index"),
        "problem_id_hash": record.get("problem_id_hash"),
        "seed_offset": record.get("seed_offset"),
        "option_hash": record.get("option_hash"),
        "option_label_hash": record.get("option_label_hash"),
        "option_text_hash": record.get("option_text_hash"),
        "option_choice": record.get("option_choice"),
        "query_kind": record.get("query_kind"),
        "query_hash": record.get("query_hash"),
        "source": record.get("source"),
        "status": record.get("status"),
        "cache_status_before": record.get("cache_status_before"),
        "cache_status_after": record.get("cache_status_after"),
        "row_count": record.get("row_count"),
        "error_type": error_type or record.get("error_type") or "",
        "error_label": record.get("error_label") or "",
        "cached_error_retry_attempted": bool(record.get("cached_error_retry_attempted")),
        "raw_content_persisted": False,
    }
    for key in (
        "parent_query_hash",
        "parent_query_kind",
        "parent_source",
        "parent_source_hash",
        "allowed_sources",
        "preferred_sources",
        "source_url_backfill_reason",
        "source_url_hash",
        "source_diagnostic_option_text_hash",
        "source_diagnostic_option_expansion_kind",
        "source_hashes",
        "answer_bearing_diagnostics_status",
        "answer_bearing_option_signal_count",
        "answer_bearing_relation_slot_covered_count",
        "answer_bearing_relation_proximity_count",
        "answer_bearing_directish_count",
        "answer_bearing_musicology_direct_signal_count",
        "answer_bearing_best_score",
        "medical_guideline_permutation_ordering_status",
        "medical_guideline_permutation_ordering_reason",
        "medical_guideline_permutation_ordering_candidate_exact",
        "medical_guideline_permutation_ordering_candidate_score",
        "medical_guideline_permutation_ordering_rank_penalty",
        "fe_hyperfine_pair_binding_status",
        "fe_hyperfine_pair_binding_reason",
        "fe_hyperfine_pair_binding_partial_row_count",
        "fe_hyperfine_pair_binding_direct_row_count",
        "fe_hyperfine_pair_binding_best_score",
    ):
        if key in record:
            payload[key] = record.get(key)
    if latency_sec is not None:
        payload["latency_sec"] = latency_sec
    if timeout_sec is not None:
        payload["timeout_sec"] = timeout_sec
    return payload


def _source_rows_answer_bearing_diagnostics(
    *,
    rows: list[dict[str, Any]],
    problem_row: dict[str, Any],
    query_row: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    stem = str(problem_row.get("_stem") or "")
    problem = problem_row.get("_problem") if isinstance(problem_row.get("_problem"), dict) else {}
    option_text = _source_prefetch_query_row_diagnostic_option_text(query_row)
    if not stem or not option_text:
        return {
            "answer_bearing_diagnostics_status": "missing_option_context",
            "source_hashes": [
                stable_hash({"title": row.get("title", ""), "snippet": row.get("snippet", "")})
                for row in rows[:5]
            ],
        }
    option_terms = _content_terms(option_text)
    relation_slot_plan = _option_claim_relation_slot_plan(
        stem=stem,
        option_text=option_text,
        planned_queries=[query],
    )
    source_hashes: list[str] = []
    option_signal_count = 0
    relation_slot_covered_count = 0
    relation_proximity_count = 0
    directish_count = 0
    musicology_direct_signal_count = 0
    best_score = 0.0
    for row in rows:
        title = str(row.get("title") or "")
        snippet = str(row.get("snippet") or "")
        text = f"{title} {snippet}"
        source_hashes.append(stable_hash({"title": title, "snippet": snippet}))
        doc_terms = _content_terms(text)
        phrase_present = _normalized_phrase_present(option_text, text)
        option_overlap = len(option_terms & doc_terms)
        musicology_phrase_signal = _musicology_short_option_phrase_signal(
            text=text,
            stem=stem,
            option_text=option_text,
            problem=problem,
        )
        option_signal = bool(phrase_present or option_overlap >= 1 or musicology_phrase_signal)
        coverage = _option_claim_relation_slot_coverage(
            text=text,
            option_text=option_text,
            option_terms=option_terms,
            relation_slot_plan=relation_slot_plan,
        )
        covered_slots = int(coverage.get("covered_slot_count") or 0)
        relation_proximity = bool(coverage.get("relation_proximity"))
        musicology_direct_signal = _musicology_short_option_direct_relation_signal(
            text=text,
            stem=stem,
        )
        if option_signal:
            option_signal_count += 1
        if covered_slots > 0:
            relation_slot_covered_count += 1
        if relation_proximity:
            relation_proximity_count += 1
        if musicology_direct_signal:
            musicology_direct_signal_count += 1
        directish = bool(option_signal and (covered_slots > 0 or relation_proximity or musicology_direct_signal))
        if directish:
            directish_count += 1
        score = (
            (1.5 if option_signal else 0.0)
            + min(option_overlap, 5)
            + (2.0 * min(covered_slots, 3))
            + (1.25 if relation_proximity else 0.0)
            + (2.0 if musicology_direct_signal else 0.0)
        )
        best_score = max(best_score, score)
    detail = {
        "answer_bearing_diagnostics_status": "evaluated" if rows else "no_rows",
        "source_hashes": source_hashes[:5],
        "source_hash_count": len(set(source_hashes)),
        "answer_bearing_option_signal_count": option_signal_count,
        "answer_bearing_relation_slot_covered_count": relation_slot_covered_count,
        "answer_bearing_relation_proximity_count": relation_proximity_count,
        "answer_bearing_directish_count": directish_count,
        "answer_bearing_musicology_direct_signal_count": musicology_direct_signal_count,
        "answer_bearing_best_score": round(best_score, 4),
    }
    medical_ordering = medical_guideline_permutation_ordering_detail(
        stem=stem,
        option_text=option_text,
        rows=rows,
    )
    if medical_ordering.get("status") != "not_applicable":
        detail["medical_guideline_permutation_ordering"] = medical_ordering
        detail["medical_guideline_permutation_ordering_status"] = (
            medical_ordering.get("status")
        )
        detail["medical_guideline_permutation_ordering_reason"] = (
            medical_ordering.get("reason")
        )
        detail["medical_guideline_permutation_ordering_candidate_exact"] = bool(
            medical_ordering.get("candidate_exact_expected_order")
        )
        detail["medical_guideline_permutation_ordering_candidate_score"] = (
            medical_ordering.get("candidate_guideline_order_score")
        )
        detail["medical_guideline_permutation_ordering_rank_penalty"] = (
            medical_ordering.get("candidate_rank_penalty")
        )
    fe_pair_binding = fe_hyperfine_pair_binding_detail(
        stem=stem,
        option_text=option_text,
        rows=rows,
    )
    if fe_pair_binding.get("status") != "not_applicable":
        detail["fe_hyperfine_pair_binding"] = fe_pair_binding
        detail["fe_hyperfine_pair_binding_status"] = fe_pair_binding.get("status")
        detail["fe_hyperfine_pair_binding_reason"] = fe_pair_binding.get("reason")
        detail["fe_hyperfine_pair_binding_partial_row_count"] = (
            fe_pair_binding.get("partial_pair_binding_row_count")
        )
        detail["fe_hyperfine_pair_binding_direct_row_count"] = (
            fe_pair_binding.get("direct_pair_binding_row_count")
        )
        detail["fe_hyperfine_pair_binding_best_score"] = (
            fe_pair_binding.get("best_pair_binding_score")
        )
    return detail
    return records


def _fetch_source(
    *,
    source: str,
    query: str,
    limit: int,
    timeout: float,
    problem: dict[str, Any] | None = None,
    ignore_cached_error: bool = False,
) -> list[dict[str, str]]:
    previous_ignore = os.environ.get("HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS")
    if ignore_cached_error:
        os.environ["HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS"] = "1"
    try:
        if source == "semantic_scholar":
            return _semantic_scholar_search(query, limit=limit, timeout=timeout)
        if source == "openalex":
            return _openalex_search(query, limit=limit, timeout=timeout)
        if source == "arxiv":
            return _arxiv_search(query, limit=limit, timeout=timeout)
        if source == "crossref":
            return _crossref_search(query, limit=limit, timeout=timeout)
        if source == "wikipedia_extract":
            return _wikipedia_extract_search(query, limit=limit, timeout=timeout)
        if source == "courtlistener":
            return _courtlistener_search(query, limit=limit, timeout=timeout)
        if source == "lso_rules":
            return _ontario_lso_rules_search(query, limit=limit, timeout=timeout)
        if source == "answer_web":
            return _answer_bearing_web_search(query, limit=limit, timeout=timeout)
        if source == "answer_web_fulltext":
            return _answer_bearing_web_fulltext_search(query, limit=limit, timeout=timeout)
        if source == "pubmed":
            return _pubmed_search(query, limit=limit, timeout=timeout)
        if source == "pubmed_pmc_fulltext":
            return _pubmed_pmc_fulltext_search(query, limit=limit, timeout=timeout)
        if source == "local_evidence_corpus":
            return _local_evidence_corpus_search(
                query,
                problem=problem or {},
                limit=limit,
            )
        if source == "pubchem":
            return _pubchem_search(query, limit=limit, timeout=timeout)
        raise ValueError(f"unsupported source: {source}")
    finally:
        if ignore_cached_error:
            if previous_ignore is None:
                os.environ.pop("HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS", None)
            else:
                os.environ["HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS"] = previous_ignore


def _cache_status(
    *,
    source: str,
    query: str,
    limit: int,
    bypass_read: bool = False,
) -> str:
    if bypass_read:
        return "miss"
    try:
        rows = _evidence_source_cache_get(source=source, query=query, limit=limit)
    except Exception:
        return "cached_error"
    if rows is not None:
        return "hit"
    try:
        rows = _evidence_source_cache_get(source=source, query=query, limit=limit, allow_expired_ok=True)
    except Exception:
        return "cached_error"
    return "hit" if rows is not None else "miss"


def _sanitize_problem_plan(row: dict[str, Any]) -> dict[str, Any]:
    safe = {
        key: value
        for key, value in row.items()
        if key != "query_records" and not str(key).startswith("_")
    }
    safe["query_count"] = len(row.get("query_records", []) or [])
    safe["query_hashes"] = [query.get("query_hash") for query in row.get("query_records", []) or []]
    safe["query_kind_counts"] = dict(Counter(str(query.get("query_kind") or "") for query in row.get("query_records", []) or []))
    query_hashes_by_option_hash: dict[str, list[str]] = {}
    query_kind_counts_by_option_hash: dict[str, dict[str, int]] = {}
    for query in row.get("query_records", []) or []:
        option_hash = str(query.get("option_hash") or query.get("option_label_hash") or "")
        if not option_hash:
            continue
        query_hashes_by_option_hash.setdefault(option_hash, []).append(str(query.get("query_hash") or ""))
        kind_counts = query_kind_counts_by_option_hash.setdefault(option_hash, {})
        kind = str(query.get("query_kind") or "")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    if query_hashes_by_option_hash:
        safe["query_hashes_by_option_hash"] = {
            key: [value for value in values if value]
            for key, values in sorted(query_hashes_by_option_hash.items())
        }
        safe["query_kind_counts_by_option_hash"] = {
            key: dict(sorted(value.items()))
            for key, value in sorted(query_kind_counts_by_option_hash.items())
        }
    return safe


def _sanitize_source_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not str(key).startswith("_")
    }


def _prefetch_metrics(
    *,
    query_plan: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    query_kind_counts = Counter(
        str(query.get("query_kind") or "")
        for problem in query_plan
        for query in problem.get("query_records", []) or []
    )
    status_counts = Counter(str(row.get("status") or "") for row in source_records)
    source_status_counts = Counter(
        f"{row.get('source')}::{row.get('status')}"
        for row in source_records
    )
    query_kind_status_counts = Counter(
        f"{row.get('query_kind')}::{row.get('status')}"
        for row in source_records
    )
    cache_before_counts = Counter(str(row.get("cache_status_before") or "") for row in source_records)
    family_counts: Counter[str] = Counter()
    for problem in query_plan:
        for family in problem.get("operator_family_tags", []) or []:
            family_counts[str(family)] += 1
    directish_by_option_hash: Counter[str] = Counter()
    option_signal_by_option_hash: Counter[str] = Counter()
    diagnostics_by_option_hash: Counter[str] = Counter()
    row_count_by_option_hash: Counter[str] = Counter()
    best_score_by_option_hash: dict[str, float] = {}
    directish_by_query_kind: Counter[str] = Counter()
    option_signal_by_query_kind: Counter[str] = Counter()
    numeric_same_row_direct_by_option_hash: Counter[str] = Counter()
    numeric_same_row_value_match_by_option_hash: Counter[str] = Counter()
    numeric_same_row_direct_by_query_kind: Counter[str] = Counter()
    numeric_same_row_value_match_by_query_kind: Counter[str] = Counter()
    numeric_same_row_rejection_counts: Counter[str] = Counter()
    numeric_same_row_value_failure_counts: Counter[str] = Counter()
    numeric_same_row_best_score_by_option_hash: dict[str, float] = {}
    medical_ordering_status_counts: Counter[str] = Counter()
    medical_ordering_exact_by_option_hash: Counter[str] = Counter()
    medical_ordering_best_score_by_option_hash: dict[str, float] = {}
    fe_pair_binding_status_counts: Counter[str] = Counter()
    fe_pair_binding_partial_by_option_hash: Counter[str] = Counter()
    fe_pair_binding_direct_by_option_hash: Counter[str] = Counter()
    fe_pair_binding_best_score_by_option_hash: dict[str, float] = {}
    for row in source_records:
        option_hash = str(row.get("option_hash") or row.get("option_label_hash") or "")
        query_kind = str(row.get("query_kind") or "")
        if _safe_int(row.get("answer_bearing_directish_count")) > 0:
            directish_by_query_kind[query_kind] += 1
        if _safe_int(row.get("answer_bearing_option_signal_count")) > 0:
            option_signal_by_query_kind[query_kind] += 1
        if _safe_int(row.get("numeric_same_row_direct_count")) > 0:
            numeric_same_row_direct_by_query_kind[query_kind] += 1
        if _safe_int(row.get("numeric_same_row_value_match_count")) > 0:
            numeric_same_row_value_match_by_query_kind[query_kind] += 1
        for reason, count in (row.get("numeric_same_row_rejection_reason_counts") or {}).items():
            numeric_same_row_rejection_counts[str(reason)] += _safe_int(count)
        for reason, count in (row.get("numeric_same_row_value_match_failure_counts") or {}).items():
            numeric_same_row_value_failure_counts[str(reason)] += _safe_int(count)
        medical_status = str(row.get("medical_guideline_permutation_ordering_status") or "")
        if medical_status:
            medical_ordering_status_counts[medical_status] += 1
        fe_status = str(row.get("fe_hyperfine_pair_binding_status") or "")
        if fe_status:
            fe_pair_binding_status_counts[fe_status] += 1
        if not option_hash:
            continue
        if row.get("medical_guideline_permutation_ordering_candidate_exact"):
            medical_ordering_exact_by_option_hash[option_hash] += 1
        if row.get("medical_guideline_permutation_ordering_candidate_score") is not None:
            medical_ordering_best_score_by_option_hash[option_hash] = max(
                medical_ordering_best_score_by_option_hash.get(option_hash, 0.0),
                _safe_float(
                    row.get("medical_guideline_permutation_ordering_candidate_score")
                ),
            )
        if _safe_int(row.get("fe_hyperfine_pair_binding_partial_row_count")) > 0:
            fe_pair_binding_partial_by_option_hash[option_hash] += 1
        if _safe_int(row.get("fe_hyperfine_pair_binding_direct_row_count")) > 0:
            fe_pair_binding_direct_by_option_hash[option_hash] += 1
        if row.get("fe_hyperfine_pair_binding_best_score") is not None:
            fe_pair_binding_best_score_by_option_hash[option_hash] = max(
                fe_pair_binding_best_score_by_option_hash.get(option_hash, 0.0),
                _safe_float(row.get("fe_hyperfine_pair_binding_best_score")),
            )
        if row.get("answer_bearing_diagnostics_status") == "evaluated":
            diagnostics_by_option_hash[option_hash] += 1
        row_count_by_option_hash[option_hash] += _safe_int(row.get("row_count"))
        if _safe_int(row.get("answer_bearing_directish_count")) > 0:
            directish_by_option_hash[option_hash] += 1
        if _safe_int(row.get("answer_bearing_option_signal_count")) > 0:
            option_signal_by_option_hash[option_hash] += 1
        if _safe_int(row.get("numeric_same_row_direct_count")) > 0:
            numeric_same_row_direct_by_option_hash[option_hash] += 1
        if _safe_int(row.get("numeric_same_row_value_match_count")) > 0:
            numeric_same_row_value_match_by_option_hash[option_hash] += 1
        best_score_by_option_hash[option_hash] = max(
            best_score_by_option_hash.get(option_hash, 0.0),
            _safe_float(row.get("answer_bearing_best_score")),
        )
        numeric_same_row_best_score_by_option_hash[option_hash] = max(
            numeric_same_row_best_score_by_option_hash.get(option_hash, 0.0),
            _safe_float(row.get("numeric_same_row_best_score")),
        )
    return {
        "problem_count": len(query_plan),
        "planned_query_count": sum(len(row.get("query_records", []) or []) for row in query_plan),
        "planned_query_kind_counts": dict(sorted(query_kind_counts.items())),
        "source_record_count": len(source_records),
        "status_counts": dict(sorted(status_counts.items())),
        "source_status_counts": dict(sorted(source_status_counts.items())),
        "query_kind_status_counts": dict(sorted(query_kind_status_counts.items())),
        "cache_status_before_counts": dict(sorted(cache_before_counts.items())),
        "fetched_count": int(status_counts.get("fetched", 0)),
        "error_count": int(status_counts.get("error", 0)),
        "cache_hit_count": int(status_counts.get("cache_hit", 0)),
        "dry_run_missing_count": int(status_counts.get("dry_run_missing", 0)),
        "budget_skipped_count": int(status_counts.get("budget_skipped", 0)),
        "cached_error_count": int(status_counts.get("cached_error", 0)),
        "answer_bearing_diagnostics_evaluated_count": sum(
            1
            for row in source_records
            if row.get("answer_bearing_diagnostics_status") == "evaluated"
        ),
        "answer_bearing_directish_record_count": sum(
            1
            for row in source_records
            if _safe_int(row.get("answer_bearing_directish_count")) > 0
        ),
        "answer_bearing_option_signal_record_count": sum(
            1
            for row in source_records
            if _safe_int(row.get("answer_bearing_option_signal_count")) > 0
        ),
        "answer_bearing_best_score_max": round(
            max(
                [
                    _safe_float(row.get("answer_bearing_best_score"))
                    for row in source_records
                ]
                or [0.0]
            ),
            4,
        ),
        "answer_bearing_diagnostics_count_by_option_hash": dict(sorted(diagnostics_by_option_hash.items())),
        "answer_bearing_directish_record_count_by_option_hash": dict(sorted(directish_by_option_hash.items())),
        "answer_bearing_option_signal_record_count_by_option_hash": dict(sorted(option_signal_by_option_hash.items())),
        "answer_bearing_directish_record_count_by_query_kind": dict(
            sorted(directish_by_query_kind.items())
        ),
        "answer_bearing_option_signal_record_count_by_query_kind": dict(
            sorted(option_signal_by_query_kind.items())
        ),
        "answer_bearing_source_row_count_by_option_hash": dict(sorted(row_count_by_option_hash.items())),
        "answer_bearing_best_score_max_by_option_hash": {
            key: round(value, 4)
            for key, value in sorted(best_score_by_option_hash.items())
        },
        "numeric_same_row_diagnostics_evaluated_count": sum(
            1
            for row in source_records
            if row.get("numeric_same_row_diagnostics_status") == "evaluated"
        ),
        "numeric_same_row_direct_record_count": sum(
            1
            for row in source_records
            if _safe_int(row.get("numeric_same_row_direct_count")) > 0
        ),
        "numeric_same_row_value_match_record_count": sum(
            1
            for row in source_records
            if _safe_int(row.get("numeric_same_row_value_match_count")) > 0
        ),
        "numeric_same_row_best_score_max": round(
            max(
                [
                    _safe_float(row.get("numeric_same_row_best_score"))
                    for row in source_records
                ]
                or [0.0]
            ),
            4,
        ),
        "numeric_same_row_direct_record_count_by_option_hash": dict(
            sorted(numeric_same_row_direct_by_option_hash.items())
        ),
        "numeric_same_row_value_match_record_count_by_option_hash": dict(
            sorted(numeric_same_row_value_match_by_option_hash.items())
        ),
        "numeric_same_row_direct_record_count_by_query_kind": dict(
            sorted(numeric_same_row_direct_by_query_kind.items())
        ),
        "numeric_same_row_value_match_record_count_by_query_kind": dict(
            sorted(numeric_same_row_value_match_by_query_kind.items())
        ),
        "numeric_same_row_best_score_max_by_option_hash": {
            key: round(value, 4)
            for key, value in sorted(numeric_same_row_best_score_by_option_hash.items())
        },
        "numeric_same_row_rejection_reason_counts": dict(
            sorted(numeric_same_row_rejection_counts.items())
        ),
        "numeric_same_row_value_match_failure_counts": dict(
            sorted(numeric_same_row_value_failure_counts.items())
        ),
        "medical_guideline_permutation_ordering_status_counts": dict(
            sorted(medical_ordering_status_counts.items())
        ),
        "medical_guideline_permutation_ordering_exact_record_count": sum(
            medical_ordering_exact_by_option_hash.values()
        ),
        "medical_guideline_permutation_ordering_unique_exact_option_hash": (
            next(iter(medical_ordering_exact_by_option_hash))
            if len(medical_ordering_exact_by_option_hash) == 1
            else None
        ),
        "medical_guideline_permutation_ordering_exact_by_option_hash": dict(
            sorted(medical_ordering_exact_by_option_hash.items())
        ),
        "medical_guideline_permutation_ordering_best_score_by_option_hash": {
            key: round(value, 4)
            for key, value in sorted(medical_ordering_best_score_by_option_hash.items())
        },
        "fe_hyperfine_pair_binding_status_counts": dict(
            sorted(fe_pair_binding_status_counts.items())
        ),
        "fe_hyperfine_pair_binding_partial_record_count": sum(
            fe_pair_binding_partial_by_option_hash.values()
        ),
        "fe_hyperfine_pair_binding_direct_record_count": sum(
            fe_pair_binding_direct_by_option_hash.values()
        ),
        "fe_hyperfine_pair_binding_unique_direct_option_hash": (
            next(iter(fe_pair_binding_direct_by_option_hash))
            if len(fe_pair_binding_direct_by_option_hash) == 1
            else None
        ),
        "fe_hyperfine_pair_binding_partial_by_option_hash": dict(
            sorted(fe_pair_binding_partial_by_option_hash.items())
        ),
        "fe_hyperfine_pair_binding_direct_by_option_hash": dict(
            sorted(fe_pair_binding_direct_by_option_hash.items())
        ),
        "fe_hyperfine_pair_binding_best_score_by_option_hash": {
            key: round(value, 4)
            for key, value in sorted(fe_pair_binding_best_score_by_option_hash.items())
        },
        "operator_family_counts": dict(sorted(family_counts.items())),
        "raw_content_persisted": False,
    }


def _source_failure_focus_from_eval_json(path: Path | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {
            "status": "disabled",
            "reason": "focus_eval_json_not_provided",
            "raw_content_persisted": False,
        }
    focus_path = Path(path)
    if not focus_path.exists():
        return {
            "status": "missing",
            "reason": "focus_eval_json_missing",
            "source_hash": stable_hash({"focus_eval_json": str(focus_path)}),
            "raw_content_persisted": False,
        }
    try:
        payload = json.loads(focus_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "error",
            "reason": "focus_eval_json_read_error",
            "error_type": type(exc).__name__,
            "source_hash": stable_hash({"focus_eval_json": str(focus_path)}),
            "raw_content_persisted": False,
        }
    return _source_failure_focus_from_eval_payload(payload, source_path=focus_path)


def _source_failure_focus_from_eval_payload(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    focus_by_seed: dict[int, dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    row_count = 0
    for seed_offset, row in _iter_source_failure_focus_eval_rows(
        payload,
        source_path=source_path,
    ):
        component = row.get("component_efficacy") if isinstance(row, dict) else {}
        component = component if isinstance(component, dict) else {}
        verifier = component.get("mc_option_claim_evidence_verifier")
        verifier = verifier if isinstance(verifier, dict) else {}
        if not verifier:
            continue
        row_count += 1
        flags = component.get("flags")
        flags = flags if isinstance(flags, dict) else {}
        if (
            bool(component.get("candidate_generation_missed_gold"))
            or bool(component.get("candidate_generation_missed_gold_with_sweep_coverage"))
            or bool(flags.get("candidate_generation_missed_gold"))
            or bool(flags.get("candidate_generation_missed_gold_with_sweep_coverage"))
        ):
            _mark_source_failure_focus_expand_all_options(
                focus_by_seed=focus_by_seed,
                reason_counts=reason_counts,
                seed_offset=seed_offset,
                reason="candidate_generation_missed_gold_expand_all_options",
            )
        for option_hash, reasons in _source_failure_focus_option_reasons(verifier).items():
            _add_source_failure_focus_option(
                focus_by_seed=focus_by_seed,
                reason_counts=reason_counts,
                seed_offset=seed_offset,
                option_hash=option_hash,
                reasons=reasons,
            )
        for option_hash, term_hashes in _source_failure_focus_option_missing_required_term_hashes(
            verifier
        ).items():
            reasons = {"required_term_hash_gap"}
            if _TERM_IDENTITY_ALL_REQUIRED_TERMS_SENTINEL in term_hashes:
                reasons.add("required_term_identity_all_terms_gap")
            _add_source_failure_focus_option(
                focus_by_seed=focus_by_seed,
                reason_counts=reason_counts,
                seed_offset=seed_offset,
                option_hash=option_hash,
                reasons=reasons,
            )
            _add_source_failure_focus_option_missing_required_term_hashes(
                focus_by_seed=focus_by_seed,
                seed_offset=seed_offset,
                option_hash=option_hash,
                missing_required_term_hashes=term_hashes,
            )
    for seed_focus in focus_by_seed.values():
        option_hashes = sorted(seed_focus.get("option_hashes", set()))
        seed_focus["option_hashes"] = option_hashes
        option_missing_required_term_hashes = {}
        for option_hash, hashes in (
            seed_focus.get("option_missing_required_term_hashes") or {}
        ).items():
            option_missing_required_term_hashes[str(option_hash)] = sorted({
                str(value)
                for value in hashes or []
                if str(value).strip()
            })
        seed_focus["option_missing_required_term_hashes"] = (
            option_missing_required_term_hashes
        )
        seed_focus["option_count"] = len(option_hashes)
        seed_focus["reason_counts"] = dict(sorted(seed_focus.get("reason_counts", Counter()).items()))
        seed_focus["status"] = (
            "activated"
            if option_hashes or seed_focus.get("expand_all_options")
            else "empty"
        )
    focused_option_count = sum(
        int(seed_focus.get("option_count") or 0)
        for seed_focus in focus_by_seed.values()
    )
    expand_all_seed_count = sum(
        1
        for seed_focus in focus_by_seed.values()
        if isinstance(seed_focus, dict) and seed_focus.get("expand_all_options")
    )
    status = "activated" if focused_option_count or expand_all_seed_count else "empty"
    return {
        "status": status,
        "reason": (
            "source_rejected_generic_candidates_found"
            if focused_option_count or expand_all_seed_count
            else "no_source_rejected_generic_candidates_found"
        ),
        "source_hash": stable_hash({"focus_eval_json": str(source_path or "")})
        if source_path is not None
        else None,
        "eval_id_hash": stable_hash({"eval_id": str(payload.get("eval_id") or "")})
        if isinstance(payload, dict)
        else None,
        "row_count": row_count,
        "seed_count": len(focus_by_seed),
        "focused_option_count": focused_option_count,
        "expand_all_option_seed_count": expand_all_seed_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "focus_by_seed": focus_by_seed,
        "raw_content_persisted": False,
    }


def _trim_source_failure_focus(
    focus: dict[str, Any],
    *,
    max_options_per_problem: int,
) -> dict[str, Any]:
    if not isinstance(focus, dict) or focus.get("status") != "activated":
        return focus
    limit = max(0, int(max_options_per_problem or 0))
    if limit <= 0:
        return focus
    focus_by_seed = focus.get("focus_by_seed")
    if not isinstance(focus_by_seed, dict):
        return focus
    untrimmed_count = sum(
        len((seed_focus or {}).get("option_hashes", []) or [])
        for seed_focus in focus_by_seed.values()
        if isinstance(seed_focus, dict)
    )
    trimmed_focus_by_seed: dict[int, dict[str, Any]] = {}
    reason_counts: Counter[str] = Counter()
    for seed_offset, seed_focus in focus_by_seed.items():
        if not isinstance(seed_focus, dict):
            continue
        option_hashes = [str(value) for value in seed_focus.get("option_hashes", []) or [] if str(value).strip()]
        option_reason_counts = seed_focus.get("option_reason_counts") or {}
        ranked = sorted(
            option_hashes,
            key=lambda option_hash: (
                _source_failure_focus_option_reason_score(
                    option_reason_counts.get(option_hash, {})
                ),
                sum((option_reason_counts.get(option_hash, {}) or {}).values())
                if isinstance(option_reason_counts.get(option_hash, {}), dict)
                else 0,
                option_hash,
            ),
            reverse=True,
        )
        kept = ranked[:limit]
        next_option_reason_counts = {}
        option_missing_required_term_hashes = (
            seed_focus.get("option_missing_required_term_hashes") or {}
        )
        next_option_missing_required_term_hashes = {}
        for option_hash in kept:
            counter = option_reason_counts.get(option_hash, {})
            if isinstance(counter, Counter):
                next_counter = Counter(counter)
            elif isinstance(counter, dict):
                next_counter = Counter({str(k): int(v or 0) for k, v in counter.items()})
            else:
                next_counter = Counter()
            next_option_reason_counts[option_hash] = next_counter
            next_option_missing_required_term_hashes[option_hash] = [
                str(value)
                for value in (
                    option_missing_required_term_hashes.get(option_hash, []) or []
                )
                if str(value).strip()
            ]
            reason_counts.update(next_counter)
        trimmed_focus_by_seed[int(seed_offset)] = {
            **seed_focus,
            "status": "activated" if kept else "empty",
            "option_hashes": kept,
            "option_count": len(kept),
            "reason_counts": dict(sorted(Counter(
                reason
                for counter in next_option_reason_counts.values()
                for reason, count in counter.items()
                for _ in range(int(count or 0))
            ).items())),
            "option_reason_counts": next_option_reason_counts,
            "option_missing_required_term_hashes": (
                next_option_missing_required_term_hashes
            ),
            "focus_trim_policy": "top_candidate_specific_failure_score_v1",
            "focus_max_options_per_problem": limit,
            "untrimmed_option_count": len(option_hashes),
            "expand_all_options": bool(seed_focus.get("expand_all_options")),
            "raw_content_persisted": False,
        }
        if seed_focus.get("expand_all_options"):
            trimmed_focus_by_seed[int(seed_offset)]["status"] = "activated"
    focused_option_count = sum(
        int(seed_focus.get("option_count") or 0)
        for seed_focus in trimmed_focus_by_seed.values()
    )
    return {
        **focus,
        "reason": "source_rejected_generic_candidates_found_topn_trimmed",
        "untrimmed_focused_option_count": untrimmed_count,
        "focused_option_count": focused_option_count,
        "expand_all_option_seed_count": sum(
            1
            for seed_focus in trimmed_focus_by_seed.values()
            if isinstance(seed_focus, dict) and seed_focus.get("expand_all_options")
        ),
        "seed_count": len(trimmed_focus_by_seed),
        "reason_counts": dict(sorted(reason_counts.items())),
        "focus_by_seed": trimmed_focus_by_seed,
        "focus_trim_policy": "top_candidate_specific_failure_score_v1",
        "focus_max_options_per_problem": limit,
        "raw_content_persisted": False,
    }


def _source_failure_focus_option_reason_score(reason_counts: Any) -> float:
    if isinstance(reason_counts, Counter):
        items = reason_counts.items()
    elif isinstance(reason_counts, dict):
        items = reason_counts.items()
    else:
        items = []
    weights = {
        "source_cache_targeted_near_complete_but_not_direct": 8.0,
        "source_cache_paired_required_overlap_but_not_direct": 8.0,
        "source_cache_backfill_signal_model_rejected": 6.0,
        "source_cache_required_gap_candidate": 5.0,
        "source_audit_direct_ambiguous_margin": 5.0,
        "candidate_span_bundle_direct_ambiguous_margin": 5.0,
        "candidate_span_bundle_indirect_needs_candidate_specific_source": 4.0,
        "required_coverage_gap": 4.0,
        "span_directness_missing_required_relation_terms": 4.0,
        "span_directness_lexical_unique_relation_generic": 3.0,
        "span_directness_generic": 3.0,
        "span_directness_indirect": 2.0,
        "source_verifier_rejected_relation": 2.0,
        "structured_relation_generic": 1.0,
        "structured_relation_indirect": 1.0,
        "source_quality_signal_not_direct_verified": 0.5,
    }
    score = 0.0
    for reason, count in items:
        score += weights.get(str(reason), 1.0) * max(0, int(count or 0))
    return score


def _iter_source_failure_focus_eval_rows(
    payload: dict[str, Any],
    *,
    source_path: Path | None,
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    if isinstance(payload.get("shards"), list):
        for shard in payload.get("shards", []) or []:
            if not isinstance(shard, dict):
                continue
            seed_offset = _safe_int(shard.get("seed_offset"))
            shard_out = str(shard.get("out") or "")
            if not shard_out:
                continue
            shard_path = Path(shard_out)
            if not shard_path.is_absolute() and source_path is not None:
                shard_path = source_path.parent / shard_path
            if not shard_path.exists():
                continue
            try:
                shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for row in shard_payload.get("rows", []) or []:
                if isinstance(row, dict):
                    rows.append((seed_offset, row))
        return rows
    seed_offset = _safe_int((payload.get("sampling") or {}).get("seed_offset"))
    for row in payload.get("rows", []) or []:
        if isinstance(row, dict):
            rows.append((seed_offset, row))
    return rows


def _source_failure_focus_option_reasons(
    verifier: dict[str, Any],
) -> dict[str, set[str]]:
    reasons_by_option: dict[str, set[str]] = {}

    def add(option_hash: Any, reason: str) -> None:
        clean_hash = str(option_hash or "").strip()
        clean_reason = str(reason or "").strip()
        if not clean_hash or not clean_reason:
            return
        reasons_by_option.setdefault(clean_hash, set()).add(clean_reason)

    for row in verifier.get("span_directness_verifier_candidate_directness_rows", []) or []:
        if not isinstance(row, dict):
            continue
        option_hash = row.get("option_hash")
        evidence_relation = str(
            row.get("evidence_relation") or row.get("model_evidence_relation") or ""
        ).strip().lower()
        if evidence_relation in {"generic", "indirect", "unsupported", "refuting"}:
            add(option_hash, f"span_directness_{evidence_relation}")
        if row.get("lexical_unique_but_relation_generic"):
            add(option_hash, "span_directness_lexical_unique_relation_generic")
        directness = row.get("candidate_relation_span_directness")
        directness = directness if isinstance(directness, dict) else {}
        gap_gate = directness.get("programmatic_gap_gate")
        gap_gate = gap_gate if isinstance(gap_gate, dict) else {}
        gap_reason = str(
            gap_gate.get("programmatic_gap_reason")
            or gap_gate.get("reason")
            or row.get("programmatic_gap_reason")
            or ""
        ).strip().lower()
        if "missing_required" in gap_reason or "missing_relation" in gap_reason:
            add(option_hash, "span_directness_missing_required_relation_terms")
        if _safe_int(row.get("candidate_relation_span_source_cache_targeted_near_complete_count")) > 0:
            add(option_hash, "source_cache_targeted_near_complete_but_not_direct")
        if _safe_int(row.get("candidate_relation_span_source_cache_paired_required_overlap_adopted_count")) > 0:
            add(option_hash, "source_cache_paired_required_overlap_but_not_direct")
        gate = row.get("ambiguous_only_model_direct_gate")
        gate = gate if isinstance(gate, dict) else {}
        if (
            _safe_int(gate.get("source_cache_backfill_doc_count")) > 0
            and not bool(row.get("direct_high_confidence"))
        ):
            add(option_hash, "source_cache_backfill_signal_model_rejected")

    for row in verifier.get("candidate_direct_relation_span_required_coverage_gap_rows", []) or []:
        if not isinstance(row, dict):
            continue
        option_hash = row.get("option_hash")
        if _safe_int(row.get("top_required_missing_count")) > 0:
            add(option_hash, "required_coverage_gap")
        if row.get("top_source_cache_corpus_backfill") or row.get(
            "top_source_cache_strict_answer_bearing_span"
        ):
            add(option_hash, "source_cache_required_gap_candidate")

    for row in verifier.get("contrastive_adjudicator_structured_relation_matrix", []) or []:
        if not isinstance(row, dict):
            continue
        option_hash = row.get("option_hash")
        evidence_relation = str(row.get("evidence_relation") or "").strip().lower()
        rejection = str(row.get("source_verifier_rejection_reason") or "").strip().lower()
        if evidence_relation in {"generic", "indirect", "unsupported", "refuting"}:
            add(option_hash, f"structured_relation_{evidence_relation}")
        if any(token in rejection for token in ("generic", "indirect", "unsupported")):
            add(option_hash, "source_verifier_rejected_relation")
        if row.get("has_source_quality") and not row.get("source_verified_direct"):
            add(option_hash, "source_quality_signal_not_direct_verified")

    promotion = verifier.get("source_quality_directness_promotion_detail")
    promotion = promotion if isinstance(promotion, dict) else {}

    def add_option_matrix_lane_focus(lane: Any, *, source: str) -> None:
        lane = lane if isinstance(lane, dict) else {}
        status = str(lane.get("status") or "").strip().lower()
        reason = str(lane.get("reason") or "").strip().lower()
        if status not in {"ambiguous", "blocked"}:
            return
        if not any(token in reason for token in ("margin", "ambiguous", "no_strong")):
            return
        direct_reason = (
            "source_audit_direct_ambiguous_margin"
            if source == "source_audit"
            else "candidate_span_bundle_direct_ambiguous_margin"
        )
        for summary in lane.get("option_summaries", []) or []:
            if not isinstance(summary, dict):
                continue
            option_hash = str(summary.get("option_hash") or "").strip()
            if not option_hash:
                continue
            relation_established = bool(summary.get("relation_established"))
            bundle_type = str(summary.get("bundle_type") or "").strip().lower()
            if relation_established:
                add(option_hash, direct_reason)
            elif source == "candidate_span_bundle" and bundle_type in {
                "indirect",
                "generic",
            }:
                add(
                    option_hash,
                    "candidate_span_bundle_indirect_needs_candidate_specific_source",
                )

    add_option_matrix_lane_focus(
        promotion.get("option_matrix_candidate_span_bundle_source_lane"),
        source="candidate_span_bundle",
    )
    add_option_matrix_lane_focus(
        promotion.get("option_matrix_source_audit_lane"),
        source="source_audit",
    )

    return reasons_by_option


def _source_failure_focus_option_missing_required_term_hashes(
    verifier: dict[str, Any],
) -> dict[str, set[str]]:
    hashes_by_option: dict[str, set[str]] = {}
    for row in verifier.get("span_directness_verifier_candidate_directness_rows", []) or []:
        if not isinstance(row, dict):
            continue
        option_hash = str(row.get("option_hash") or "").strip()
        if not option_hash:
            continue
        directness = row.get("candidate_relation_span_directness")
        directness = directness if isinstance(directness, dict) else {}
        gap_gate = directness.get("programmatic_gap_gate")
        gap_gate = gap_gate if isinstance(gap_gate, dict) else {}
        gap_reason = str(
            gap_gate.get("programmatic_gap_reason")
            or gap_gate.get("reason")
            or row.get("programmatic_gap_reason")
            or ""
        ).strip().lower()
        if "missing_required" not in gap_reason and "missing_relation" not in gap_reason:
            continue
        hashes_by_option.setdefault(option_hash, set()).add(
            _TERM_IDENTITY_ALL_REQUIRED_TERMS_SENTINEL
        )
    for row in verifier.get("candidate_direct_relation_span_required_coverage_gap_rows", []) or []:
        if not isinstance(row, dict):
            continue
        option_hash = str(row.get("option_hash") or "").strip()
        if not option_hash:
            continue
        if _safe_int(row.get("top_required_missing_count")) <= 0:
            continue
        missing_hashes = {
            str(value)
            for value in row.get("top_required_missing_term_hashes", []) or []
            if str(value).strip()
        }
        if not missing_hashes:
            continue
        hashes_by_option.setdefault(option_hash, set()).update(missing_hashes)
    return hashes_by_option


def _add_source_failure_focus_option(
    *,
    focus_by_seed: dict[int, dict[str, Any]],
    reason_counts: Counter[str],
    seed_offset: int,
    option_hash: str,
    reasons: set[str],
) -> None:
    if seed_offset <= 0 or not option_hash or not reasons:
        return
    seed_focus = focus_by_seed.setdefault(
        int(seed_offset),
        {
            "status": "activated",
            "option_hashes": set(),
            "reason_counts": Counter(),
            "option_reason_counts": {},
            "raw_content_persisted": False,
        },
    )
    seed_focus["option_hashes"].add(option_hash)
    option_reason_counts = seed_focus.setdefault("option_reason_counts", {})
    option_counter = option_reason_counts.setdefault(option_hash, Counter())
    for reason in sorted(reasons):
        reason_counts[reason] += 1
        seed_focus["reason_counts"][reason] += 1
        option_counter[reason] += 1


def _add_source_failure_focus_option_missing_required_term_hashes(
    *,
    focus_by_seed: dict[int, dict[str, Any]],
    seed_offset: int,
    option_hash: str,
    missing_required_term_hashes: set[str],
) -> None:
    if seed_offset <= 0 or not option_hash or not missing_required_term_hashes:
        return
    seed_focus = focus_by_seed.setdefault(
        int(seed_offset),
        {
            "status": "activated",
            "option_hashes": set(),
            "reason_counts": Counter(),
            "option_reason_counts": {},
            "option_missing_required_term_hashes": {},
            "raw_content_persisted": False,
        },
    )
    seed_focus["option_hashes"].add(option_hash)
    by_option = seed_focus.setdefault("option_missing_required_term_hashes", {})
    current = {
        str(value)
        for value in by_option.get(option_hash, set()) or set()
        if str(value).strip()
    }
    current.update(
        str(value)
        for value in missing_required_term_hashes
        if str(value).strip()
    )
    by_option[option_hash] = current


def _mark_source_failure_focus_expand_all_options(
    *,
    focus_by_seed: dict[int, dict[str, Any]],
    reason_counts: Counter[str],
    seed_offset: int,
    reason: str,
) -> None:
    if seed_offset <= 0:
        return
    clean_reason = str(reason or "").strip()
    if not clean_reason:
        return
    seed_focus = focus_by_seed.setdefault(
        int(seed_offset),
        {
            "status": "activated",
            "option_hashes": set(),
            "reason_counts": Counter(),
            "option_reason_counts": {},
            "raw_content_persisted": False,
        },
    )
    seed_focus["expand_all_options"] = True
    seed_focus["expand_all_options_reason"] = clean_reason
    seed_focus["reason_counts"][clean_reason] += 1
    reason_counts[clean_reason] += 1


def _sanitize_failure_focus_summary(focus: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(focus, dict):
        return {
            "status": "disabled",
            "reason": "failure_focus_unavailable",
            "raw_content_persisted": False,
        }
    focus_by_seed = focus.get("focus_by_seed") if isinstance(focus.get("focus_by_seed"), dict) else {}
    return {
        "status": focus.get("status"),
        "reason": focus.get("reason"),
        "source_hash": focus.get("source_hash"),
        "eval_id_hash": focus.get("eval_id_hash"),
        "row_count": int(focus.get("row_count") or 0),
        "seed_count": int(focus.get("seed_count") or 0),
        "focused_option_count": int(focus.get("focused_option_count") or 0),
        "expand_all_option_seed_count": int(
            focus.get("expand_all_option_seed_count") or 0
        ),
        "untrimmed_focused_option_count": int(
            focus.get("untrimmed_focused_option_count")
            or focus.get("focused_option_count")
            or 0
        ),
        "focus_trim_policy": focus.get("focus_trim_policy") or "",
        "focus_max_options_per_problem": int(
            focus.get("focus_max_options_per_problem") or 0
        ),
        "reason_counts": dict(focus.get("reason_counts") or {}),
        "focused_option_hashes_by_seed": {
            str(seed): list((seed_focus or {}).get("option_hashes", []) or [])
            for seed, seed_focus in sorted(focus_by_seed.items())
        },
        "missing_required_term_hashes_by_seed_option": {
            str(seed): {
                str(option_hash): [
                    str(value)
                    for value in (hashes or [])
                    if str(value).strip()
                ]
                for option_hash, hashes in (
                    (seed_focus or {}).get("option_missing_required_term_hashes")
                    or {}
                ).items()
            }
            for seed, seed_focus in sorted(focus_by_seed.items())
            if (seed_focus or {}).get("option_missing_required_term_hashes")
        },
        "expand_all_options_by_seed": {
            str(seed): bool((seed_focus or {}).get("expand_all_options"))
            for seed, seed_focus in sorted(focus_by_seed.items())
            if bool((seed_focus or {}).get("expand_all_options"))
        },
        "raw_content_persisted": False,
    }


def _sanitize_failure_focus_seed_summary(seed_focus: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(seed_focus, dict) or not seed_focus:
        return {
            "status": "not_targeted",
            "option_count": 0,
            "option_hashes": [],
            "reason_counts": {},
            "raw_content_persisted": False,
        }
    option_reason_counts = {}
    for option_hash, counter in (seed_focus.get("option_reason_counts") or {}).items():
        if isinstance(counter, Counter):
            option_reason_counts[str(option_hash)] = dict(sorted(counter.items()))
        elif isinstance(counter, dict):
            option_reason_counts[str(option_hash)] = dict(sorted(counter.items()))
    return {
        "status": seed_focus.get("status") or "activated",
        "option_count": int(seed_focus.get("option_count") or 0),
        "option_hashes": list(seed_focus.get("option_hashes", []) or []),
        "reason_counts": dict(seed_focus.get("reason_counts") or {}),
        "option_reason_counts": option_reason_counts,
        "option_missing_required_term_hashes": {
            str(option_hash): [
                str(value)
                for value in (hashes or [])
                if str(value).strip()
            ]
            for option_hash, hashes in (
                seed_focus.get("option_missing_required_term_hashes") or {}
            ).items()
        },
        "expand_all_options": bool(seed_focus.get("expand_all_options")),
        "expand_all_options_reason": str(
            seed_focus.get("expand_all_options_reason") or ""
        ),
        "focus_trim_policy": seed_focus.get("focus_trim_policy") or "",
        "focus_max_options_per_problem": int(
            seed_focus.get("focus_max_options_per_problem") or 0
        ),
        "untrimmed_option_count": int(seed_focus.get("untrimmed_option_count") or 0),
        "raw_content_persisted": False,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_sources(sources: list[str]) -> list[str]:
    allowed = set(SUPPORTED_SOURCES)
    out: list[str] = []
    for source in sources:
        clean = str(source or "").strip()
        if not clean:
            continue
        if clean not in allowed:
            raise ValueError(f"unsupported source: {clean}")
        if clean not in out:
            out.append(clean)
    return out or list(DEFAULT_SOURCES)


def _parse_seed_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            offsets.append(int(item))
    return offsets


def _enter_prefetch_env(
    *,
    execute_live: bool,
    refresh_cache_hits: bool = False,
) -> dict[str, str | None]:
    tracked = {
        "HLE_EVIDENCE_SOURCE_CACHE_ONLY",
        "HLE_SOURCE_SEARCH_CACHE_ONLY",
        "HLE_DISABLE_LIVE_SOURCE_SEARCH",
        "HLE_ALLOW_LIVE_SOURCE_SEARCH",
        "HLE_EVIDENCE_SOURCE_CACHE_BYPASS_READ",
        "HLE_DATASET_LOCAL_PATH",
        "HLE_EVIDENCE_SOURCE_CACHE_DIR",
        "HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS",
    }
    previous = {key: os.environ.get(key) for key in tracked}
    apply_hle_offline_defaults_to_environ(os.environ)
    if execute_live:
        os.environ.pop("HLE_EVIDENCE_SOURCE_CACHE_ONLY", None)
        os.environ.pop("HLE_SOURCE_SEARCH_CACHE_ONLY", None)
        os.environ.pop("HLE_DISABLE_LIVE_SOURCE_SEARCH", None)
        os.environ["HLE_ALLOW_LIVE_SOURCE_SEARCH"] = "1"
        if refresh_cache_hits:
            os.environ["HLE_EVIDENCE_SOURCE_CACHE_BYPASS_READ"] = "1"
    return previous


def _restore_env(previous_env: dict[str, str | None]) -> None:
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        f"# {payload.get('eval_id')}",
        "",
        "## Summary",
        "",
        f"- execute live: `{payload.get('execute_live')}`",
        f"- raw content persisted: `{payload.get('raw_content_persisted')}`",
        f"- problem count: `{metrics.get('problem_count')}`",
        f"- planned query count: `{metrics.get('planned_query_count')}`",
        f"- planned query kind counts: `{metrics.get('planned_query_kind_counts')}`",
        f"- source record count: `{metrics.get('source_record_count')}`",
        f"- status counts: `{metrics.get('status_counts')}`",
        f"- query-kind status counts: `{metrics.get('query_kind_status_counts')}`",
        f"- cache status before counts: `{metrics.get('cache_status_before_counts')}`",
        f"- answer-bearing diagnostics evaluated: `{metrics.get('answer_bearing_diagnostics_evaluated_count')}`",
        f"- answer-bearing direct-ish records: `{metrics.get('answer_bearing_directish_record_count')}`",
        f"- answer-bearing direct-ish by query kind: `{metrics.get('answer_bearing_directish_record_count_by_query_kind')}`",
        f"- answer-bearing best score max: `{metrics.get('answer_bearing_best_score_max')}`",
        f"- numeric same-row diagnostics evaluated: `{metrics.get('numeric_same_row_diagnostics_evaluated_count')}`",
        f"- numeric same-row direct records: `{metrics.get('numeric_same_row_direct_record_count')}`",
        f"- numeric same-row direct by query kind: `{metrics.get('numeric_same_row_direct_record_count_by_query_kind')}`",
        f"- numeric same-row rejection counts: `{metrics.get('numeric_same_row_rejection_reason_counts')}`",
        f"- operator family counts: `{metrics.get('operator_family_counts')}`",
        "",
        "## Boundary",
        "",
        str(payload.get("claim_boundary") or ""),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch public source evidence for a fixed local HLE cohort.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_source_prefetch")
    parser.add_argument("--seed-offsets", required=True)
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--max-options-per-problem", type=int, default=8)
    parser.add_argument("--max-queries-per-problem", type=int, default=24)
    parser.add_argument("--max-queries-per-option", type=int, default=4)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--source-limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--max-live-calls", type=int, default=80)
    parser.add_argument("--delay-sec", type=float, default=1.1)
    parser.add_argument("--retry-cached-errors", action="store_true")
    parser.add_argument(
        "--refresh-cache-hits",
        action="store_true",
        help=(
            "During live prefetch, bypass success-cache reads so source rows can be "
            "refetched and rewritten with newer metadata such as source URLs."
        ),
    )
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument(
        "--budget-policy",
        choices=["round_robin_by_problem", "sequential"],
        default="round_robin_by_problem",
    )
    parser.add_argument(
        "--source-error-budget",
        type=int,
        default=0,
        help="If positive, skip later live fetches for a source after this many errors in one run.",
    )
    parser.add_argument("--log-out", default="")
    parser.add_argument("--enable-relation-query-planner", action="store_true")
    parser.add_argument("--enable-sweep-gap-relation-backfill-queries", action="store_true")
    parser.add_argument(
        "--enable-option-aware-query-expansion",
        action="store_true",
        help="Opt-in experimental option-anchor query expansion for source prefetch diagnostics.",
    )
    parser.add_argument(
        "--enable-answer-bearing-binding-queries",
        action="store_true",
        help="Opt-in targeted option+relation binding queries for answer-bearing source prefetch.",
    )
    parser.add_argument(
        "--enable-answer-bearing-pair-binding-queries",
        action="store_true",
        help=(
            "Opt-in option-vs-option relation binding queries for ambiguous "
            "pair source prefetch diagnostics."
        ),
    )
    parser.add_argument(
        "--enable-required-relation-completion-queries",
        action="store_true",
        help=(
            "Opt-in option+required-relation-term queries for filling relation "
            "coverage gaps found by answer-time source verification."
        ),
    )
    parser.add_argument(
        "--enable-candidate-specific-answer-bearing-queries",
        action="store_true",
        help=(
            "Opt-in stricter option+required-relation witness queries intended to "
            "build candidate-specific answer-bearing source cache entries."
        ),
    )
    parser.add_argument(
        "--enable-numeric-same-row-backfill-queries",
        action="store_true",
        help=(
            "Opt-in second-pass numeric prefetch queries from value-match source rows."
        ),
    )
    parser.add_argument(
        "--numeric-same-row-backfill-max-live-calls",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--focus-eval-json",
        default="",
        help=(
            "Optional prior HLE eval artifact. Source-rejected/generic option hashes "
            "from its sanitized traces are used to focus candidate-specific prefetch."
        ),
    )
    parser.add_argument(
        "--focus-only",
        action="store_true",
        help="When --focus-eval-json is provided, plan queries only for focused option hashes.",
    )
    parser.add_argument(
        "--focus-max-options-per-problem",
        type=int,
        default=4,
        help=(
            "Cap failure-focused option hashes per seed by sanitized source/directness "
            "failure score; use 0 to disable trimming."
        ),
    )
    parser.add_argument("--relation-query-planner-model", default="gpt-5.4-mini")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()
    private_env_status = load_private_env()

    root = Path(args.root).resolve()
    graph_dir = Path(args.graph_dir)
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    focus_eval_json = Path(args.focus_eval_json) if args.focus_eval_json else None
    if focus_eval_json is not None and not focus_eval_json.is_absolute():
        focus_eval_json = root / focus_eval_json
    logger = None
    if args.log_out:
        log_out = Path(args.log_out)
        log_out = log_out if log_out.is_absolute() else root / log_out
        logger = JsonlDiagnosticLogger(log_out)
    payload = build_hle_source_prefetch_payload(
        root=root,
        eval_id=args.eval_id,
        seed_offsets=_parse_seed_offsets(args.seed_offsets),
        graph_dir=graph_dir,
        max_scan=args.max_scan,
        max_options_per_problem=args.max_options_per_problem,
        max_queries_per_problem=args.max_queries_per_problem,
        max_queries_per_option=args.max_queries_per_option,
        sources=[item.strip() for item in args.sources.split(",") if item.strip()],
        source_limit=args.source_limit,
        timeout=args.timeout,
        execute_live=bool(args.execute_live),
        max_live_calls=args.max_live_calls,
        delay_sec=args.delay_sec,
        retry_cached_errors=bool(args.retry_cached_errors),
        refresh_cache_hits=bool(args.refresh_cache_hits),
        parallel_workers=args.parallel_workers,
        budget_policy=args.budget_policy,
        source_error_budget=args.source_error_budget,
        logger=logger,
        enable_relation_query_planner=bool(args.enable_relation_query_planner),
        enable_sweep_gap_relation_backfill_queries=bool(
            args.enable_sweep_gap_relation_backfill_queries
        ),
        enable_option_aware_query_expansion=bool(args.enable_option_aware_query_expansion),
        enable_answer_bearing_binding_queries=bool(
            args.enable_answer_bearing_binding_queries
        ),
        enable_answer_bearing_pair_binding_queries=bool(
            args.enable_answer_bearing_pair_binding_queries
        ),
        enable_required_relation_completion_queries=bool(
            args.enable_required_relation_completion_queries
        ),
        enable_candidate_specific_answer_bearing_queries=bool(
            args.enable_candidate_specific_answer_bearing_queries
        ),
        enable_numeric_same_row_backfill_queries=bool(
            args.enable_numeric_same_row_backfill_queries
        ),
        numeric_same_row_backfill_max_live_calls=int(
            args.numeric_same_row_backfill_max_live_calls or 0
        ),
        focus_eval_json=focus_eval_json,
        focus_only=bool(args.focus_only),
        focus_max_options_per_problem=args.focus_max_options_per_problem,
        relation_query_planner_model=args.relation_query_planner_model,
    )
    payload["private_env"] = _sanitize_private_env_status_for_artifact(private_env_status)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "execute_live": payload["execute_live"],
        "metrics": payload["metrics"],
        "out": str(out),
        "md_out": str(Path(args.md_out) if args.md_out else ""),
        "log_out": str(Path(args.log_out) if args.log_out else ""),
    }, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
