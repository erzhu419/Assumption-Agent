"""Prefetch answer-bearing source evidence for a fixed HLE cohort.

This tool is intentionally separate from scoring.  It may use raw local HLE
questions in memory to build source queries, but persisted artifacts store only
hashes, counts, and cache/source statuses.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .diagnostic_logging import JsonlDiagnosticLogger, log_event
from .hle_smoke_eval import (
    DEFAULT_GRAPH_DIR,
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
    _musicology_short_option_direct_relation_signal,
    _musicology_short_option_phrase_signal,
    _openalex_search,
    _option_claim_answer_web_fallback_queries,
    _option_claim_evidence_queries_for_plan,
    _option_claim_local_relation_query_expansion_queries,
    _option_claim_relation_slot_coverage,
    _option_claim_relation_slot_plan,
    _option_evidence_queries_for_plan,
    _question_evidence_anchor_terms,
    _question_relation_query_terms,
    _run_option_claim_relation_query_planner,
    _semantic_scholar_search,
    _split_multiple_choice_question,
    _normalized_phrase_present,
    _wikipedia_extract_search,
    apply_hle_offline_defaults_to_environ,
)
from .hle_operator_cohort_preflight import _operator_family_tags_from_stage


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
    parallel_workers: int = 1,
    budget_policy: str = "round_robin_by_problem",
    source_error_budget: int = 0,
    logger: JsonlDiagnosticLogger | None = None,
    enable_relation_query_planner: bool = False,
    enable_sweep_gap_relation_backfill_queries: bool = False,
    enable_option_aware_query_expansion: bool = False,
    relation_query_planner_model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    root = root.resolve()
    graph_path = graph_dir or (root / DEFAULT_GRAPH_DIR)
    graph_path = graph_path if graph_path.is_absolute() else root / graph_path
    source_names = _normalize_sources(sources or list(DEFAULT_SOURCES))
    previous_env = _enter_prefetch_env(execute_live=execute_live)
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
            parallel_workers=parallel_workers,
            budget_policy=budget_policy,
            source_error_budget=source_error_budget,
            logger=logger,
        )
    finally:
        _restore_env(previous_env)

    metrics = _prefetch_metrics(query_plan=query_plan, source_records=source_records)
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
            "parallel_workers": parallel_workers,
            "budget_policy": budget_policy,
            "source_error_budget": source_error_budget,
            "enable_relation_query_planner": enable_relation_query_planner,
            "enable_sweep_gap_relation_backfill_queries": enable_sweep_gap_relation_backfill_queries,
            "enable_option_aware_query_expansion": enable_option_aware_query_expansion,
            "relation_query_planner_model": relation_query_planner_model,
        },
        "metrics": metrics,
        "problems": [_sanitize_problem_plan(row) for row in query_plan],
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
            relation_query_planner_model=relation_query_planner_model,
            logger=logger,
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
            "relation_query_planner": relation_query_planner_summary,
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
                "relation_query_planner_status": relation_query_planner_summary.get("status"),
                "relation_query_planner_query_count": relation_query_planner_summary.get("query_count"),
                "relation_query_planner_model_query_count": relation_query_planner_summary.get("model_query_count"),
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
    for label, option_text in list(options.items())[: max(1, max_options)]:
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
            records.append({
                "option_hash": stable_hash({"option_label": label}),
                "option_label_hash": stable_hash({"option_label": label}),
                "option_text_hash": stable_hash({"option_text": option_text}),
                "option_choice": _extract_choice(label) or label,
                "_option_label": label,
                "_option_text": option_text,
                "query_kind": kind,
                "query_hash": query_hash,
                "_query": query,
            })
            if len(records) >= max(1, max_queries_per_problem):
                return records, relation_query_planner_summary
    return records, relation_query_planner_summary


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
    buckets = [
        [("option_claim_relation_planner", query) for query in relation_queries],
        deterministic_relation_queries,
        local_relation_queries,
        [("answer_web_fallback", query) for query in answer_web_queries],
        [("option_claim", query) for query in claim_queries],
        [("option_evidence", query) for query in option_queries],
        list(option_aware_query_pairs or []),
        other_relation_queries,
    ]
    return _round_robin_query_pairs(buckets, max_queries=max_queries)


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


def _source_prefetch_focus_phrases(text: str, *, max_phrases: int) -> list[str]:
    raw = str(text or "")
    phrases: list[str] = []
    seen: set[str] = set()
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


def _source_prefetch_query_has_option_anchor(*, query: str, option_text: str) -> bool:
    option_terms = _content_terms(option_text)
    query_terms = _content_terms(query)
    if option_terms & query_terms:
        return True
    for phrase in _source_prefetch_focus_phrases(option_text, max_phrases=4):
        if phrase and phrase.lower() in query.lower():
            return True
    return False


def _normalize_query_key(text: str) -> str:
    return " ".join(str(text or "").lower().split())


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
            for source in sources:
                before_status = _cache_status(source=source, query=query, limit=source_limit)
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
                if not execute_live:
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
    problem_orders: dict[str, list[tuple[int, dict[str, Any]]]] = {
        group_key: _source_prefetch_problem_fair_candidate_order(grouped[group_key])
        for group_key in group_order
    }
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
            option_key = f"__no_option__:{index}"
        if option_key not in option_groups:
            option_groups[option_key] = []
            option_order.append(option_key)
        option_groups[option_key].append((index, job))
    option_orders: dict[str, list[tuple[int, dict[str, Any]]]] = {
        option_key: _source_prefetch_query_fair_candidate_order(option_groups[option_key])
        for option_key in option_order
    }
    ordered: list[tuple[int, dict[str, Any]]] = []
    offsets = {option_key: 0 for option_key in option_order}
    while True:
        progressed = False
        for option_key in option_order:
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
    ordered: list[tuple[int, dict[str, Any]]] = []
    offsets = {query_key: 0 for query_key in query_order}
    while True:
        progressed = False
        for query_key in query_order:
            offset = offsets[query_key]
            group_items = query_groups[query_key]
            if offset >= len(group_items):
                continue
            ordered.append(group_items[offset])
            offsets[query_key] = offset + 1
            progressed = True
        if not progressed:
            break
    return ordered


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
        selected = {
            index
            for index, _job in _source_prefetch_fair_candidate_order(candidates)[:budget]
        }
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
            rows = _fetch_source(
                source=source,
                query=query,
                limit=source_limit,
                timeout=timeout,
                ignore_cached_error=bool(job.get("ignore_cached_error")),
            )
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
        "source_hashes",
        "answer_bearing_diagnostics_status",
        "answer_bearing_option_signal_count",
        "answer_bearing_relation_slot_covered_count",
        "answer_bearing_relation_proximity_count",
        "answer_bearing_directish_count",
        "answer_bearing_musicology_direct_signal_count",
        "answer_bearing_best_score",
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
    option_text = str(query_row.get("_option_text") or "")
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
    return {
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
    return records


def _fetch_source(
    *,
    source: str,
    query: str,
    limit: int,
    timeout: float,
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
        if source == "answer_web":
            return _answer_bearing_web_search(query, limit=limit, timeout=timeout)
        raise ValueError(f"unsupported source: {source}")
    finally:
        if ignore_cached_error:
            if previous_ignore is None:
                os.environ.pop("HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS", None)
            else:
                os.environ["HLE_SOURCE_PREFETCH_RETRY_CACHED_ERRORS"] = previous_ignore


def _cache_status(*, source: str, query: str, limit: int) -> str:
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
    status_counts = Counter(str(row.get("status") or "") for row in source_records)
    source_status_counts = Counter(
        f"{row.get('source')}::{row.get('status')}"
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
    for row in source_records:
        option_hash = str(row.get("option_hash") or row.get("option_label_hash") or "")
        if not option_hash:
            continue
        if row.get("answer_bearing_diagnostics_status") == "evaluated":
            diagnostics_by_option_hash[option_hash] += 1
        row_count_by_option_hash[option_hash] += _safe_int(row.get("row_count"))
        if _safe_int(row.get("answer_bearing_directish_count")) > 0:
            directish_by_option_hash[option_hash] += 1
        if _safe_int(row.get("answer_bearing_option_signal_count")) > 0:
            option_signal_by_option_hash[option_hash] += 1
        best_score_by_option_hash[option_hash] = max(
            best_score_by_option_hash.get(option_hash, 0.0),
            _safe_float(row.get("answer_bearing_best_score")),
        )
    return {
        "problem_count": len(query_plan),
        "planned_query_count": sum(len(row.get("query_records", []) or []) for row in query_plan),
        "source_record_count": len(source_records),
        "status_counts": dict(sorted(status_counts.items())),
        "source_status_counts": dict(sorted(source_status_counts.items())),
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
        "answer_bearing_source_row_count_by_option_hash": dict(sorted(row_count_by_option_hash.items())),
        "answer_bearing_best_score_max_by_option_hash": {
            key: round(value, 4)
            for key, value in sorted(best_score_by_option_hash.items())
        },
        "operator_family_counts": dict(sorted(family_counts.items())),
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
    allowed = set(DEFAULT_SOURCES)
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


def _enter_prefetch_env(*, execute_live: bool) -> dict[str, str | None]:
    tracked = {
        "HLE_EVIDENCE_SOURCE_CACHE_ONLY",
        "HLE_SOURCE_SEARCH_CACHE_ONLY",
        "HLE_DISABLE_LIVE_SOURCE_SEARCH",
        "HLE_ALLOW_LIVE_SOURCE_SEARCH",
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
        f"- source record count: `{metrics.get('source_record_count')}`",
        f"- status counts: `{metrics.get('status_counts')}`",
        f"- cache status before counts: `{metrics.get('cache_status_before_counts')}`",
        f"- answer-bearing diagnostics evaluated: `{metrics.get('answer_bearing_diagnostics_evaluated_count')}`",
        f"- answer-bearing direct-ish records: `{metrics.get('answer_bearing_directish_record_count')}`",
        f"- answer-bearing best score max: `{metrics.get('answer_bearing_best_score_max')}`",
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
    parser.add_argument("--relation-query-planner-model", default="gpt-5.4-mini")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    graph_dir = Path(args.graph_dir)
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
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
        parallel_workers=args.parallel_workers,
        budget_policy=args.budget_policy,
        source_error_budget=args.source_error_budget,
        logger=logger,
        enable_relation_query_planner=bool(args.enable_relation_query_planner),
        enable_sweep_gap_relation_backfill_queries=bool(
            args.enable_sweep_gap_relation_backfill_queries
        ),
        enable_option_aware_query_expansion=bool(args.enable_option_aware_query_expansion),
        relation_query_planner_model=args.relation_query_planner_model,
    )
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
