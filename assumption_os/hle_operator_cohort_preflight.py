"""Select HLE rows where assumption operators are likely to be executable.

This preflight is intentionally metadata-only on disk.  It streams HLE rows in
memory, runs the same HLE OperatorSpec compiler used by the live agent, and
persists only stable hashes, public metadata, scanned offsets, and operator
summaries.  Raw HLE questions, gold answers, rationales, and predictions are
not written.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .autonomy_journal import PAPER_DIR, stable_hash
from .diagnostic_logging import JsonlDiagnosticLogger, log_event
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .hle_smoke_eval import (
    DATASET_NAME,
    DEFAULT_GRAPH_DIR,
    _call_model,
    _classify_hle_domain,
    _compile_hle_operator_stage,
    _domain_rule_mc_decision,
    _has_image_payload,
    _load_hle_test_dataset,
    _operator_application_answer_prompt,
    _operator_required_slots,
    _parse_operator_child_audit_json,
    _problem_from_row,
    _split_multiple_choice_question,
)


DEFAULT_OUT = PAPER_DIR / "hle_operator_cohort_preflight_20260621.json"
DEFAULT_MD_OUT = Path("reconstruction/md/hle_operator_cohort_preflight_20260621.md")

_BUILTIN_OPERATOR_FAMILIES = {
    "framework_dependency_aware_controlled_intervention": "controlled_variable",
    "framework_incremental_replacement_migration": "incremental_replacement",
    "framework_structural_transfer_analogy": "structural_transfer",
    "framework_answer_bearing_relation": "answer_bearing_relation",
}

_OPERATOR_FAMILY_ALIASES = {
    "causal": "controlled_variable",
    "causal_attribution": "controlled_variable",
    "control": "controlled_variable",
    "controlled": "controlled_variable",
    "controlled_variable_or_causal_attribution": "controlled_variable",
    "migration": "incremental_replacement",
    "incremental": "incremental_replacement",
    "replacement": "incremental_replacement",
    "system_migration": "incremental_replacement",
    "analogy": "structural_transfer",
    "structural": "structural_transfer",
    "structural_analogy": "structural_transfer",
    "structural_mapping": "structural_transfer",
    "relation": "answer_bearing_relation",
    "evidence_relation": "answer_bearing_relation",
}

_PROGRAMMATIC_DOMAIN_RULE_FAMILIES = {
    "sec_mals_mass_balance_affinity_monomer": "sec_mals_mass_balance",
    "bacterial_cross_resistance_minimal_extra_assumption": "cross_resistance_minimality",
}


def build_hle_operator_cohort_preflight_payload(
    *,
    root: Path,
    eval_id: str = "hle_operator_cohort_preflight_20260621",
    target_size: int = 12,
    max_scan: int = 2000,
    seed_offset: int = 0,
    answer_type_filter: str = "multipleChoice",
    subject_contains: str = "",
    graph_dir: Path | None = None,
    fallback_min_score: float = 0.145,
    max_specs: int = 2,
    family_targets: dict[str, int] | None = None,
    enable_applicability_probe: bool = False,
    applicability_probe_model: str = "gpt-5.4-mini",
    applicability_probe_max_tokens: int = 512,
    applicability_probe_min_slot_rate: float = 0.75,
    include_programmatic_domain_rules: bool = False,
    programmatic_domain_rule_target_size: int | None = None,
    log_out: Path | None = None,
    diagnostic_log_interval: int = 1000,
) -> dict[str, Any]:
    logger = JsonlDiagnosticLogger(log_out) if log_out else None
    log_event(
        logger,
        {
            "event": "hle_operator_cohort_preflight_started",
            "eval_id": eval_id,
            "target_size": int(target_size),
            "max_scan": int(max_scan),
            "seed_offset": int(seed_offset),
            "answer_type_filter": answer_type_filter,
            "subject_contains_filter_enabled": bool(subject_contains),
            "include_programmatic_domain_rules": bool(include_programmatic_domain_rules),
            "programmatic_domain_rule_target_size": programmatic_domain_rule_target_size,
            "diagnostic_log_interval": int(diagnostic_log_interval),
        },
    )
    graph_path = graph_dir or (root / DEFAULT_GRAPH_DIR)
    graph_path = graph_path if graph_path.is_absolute() else root / graph_path
    store = JsonlGraphStore(graph_path)
    graph = SimpleAssumptionGraph(store)

    previous_env = {
        "HLE_ENABLE_ASSUMPTION_OPERATORS": os.environ.get("HLE_ENABLE_ASSUMPTION_OPERATORS"),
        "HLE_ASSUMPTION_OPERATORS_ALLOW_WITHOUT_CONTEXT": os.environ.get(
            "HLE_ASSUMPTION_OPERATORS_ALLOW_WITHOUT_CONTEXT"
        ),
        "HLE_ASSUMPTION_OPERATOR_RETRIEVAL_FALLBACK": os.environ.get(
            "HLE_ASSUMPTION_OPERATOR_RETRIEVAL_FALLBACK"
        ),
        "HLE_ASSUMPTION_OPERATOR_FALLBACK_MIN_TOP_SCORE": os.environ.get(
            "HLE_ASSUMPTION_OPERATOR_FALLBACK_MIN_TOP_SCORE"
        ),
        "HLE_ASSUMPTION_OPERATOR_MAX_SPECS": os.environ.get("HLE_ASSUMPTION_OPERATOR_MAX_SPECS"),
    }
    os.environ["HLE_ENABLE_ASSUMPTION_OPERATORS"] = "1"
    os.environ["HLE_ASSUMPTION_OPERATORS_ALLOW_WITHOUT_CONTEXT"] = "1"
    os.environ["HLE_ASSUMPTION_OPERATOR_RETRIEVAL_FALLBACK"] = "1"
    os.environ["HLE_ASSUMPTION_OPERATOR_FALLBACK_MIN_TOP_SCORE"] = str(fallback_min_score)
    os.environ["HLE_ASSUMPTION_OPERATOR_MAX_SPECS"] = str(max_specs)
    canonical_family_targets = _canonicalize_family_targets(family_targets or {})
    effective_target_size = max(target_size, sum(canonical_family_targets.values()))
    try:
        rows, scan_summary = _scan_operator_rows(
            eval_id=eval_id,
            graph=graph,
            target_size=effective_target_size,
            max_scan=max_scan,
            seed_offset=seed_offset,
            answer_type_filter=answer_type_filter,
            subject_contains=subject_contains,
            family_targets=canonical_family_targets,
            enable_applicability_probe=enable_applicability_probe,
            applicability_probe_model=applicability_probe_model,
            applicability_probe_max_tokens=applicability_probe_max_tokens,
            applicability_probe_min_slot_rate=applicability_probe_min_slot_rate,
            logger=logger,
            log_interval=diagnostic_log_interval,
        )
        if include_programmatic_domain_rules:
            domain_rule_target_size = (
                target_size
                if programmatic_domain_rule_target_size is None
                else max(0, int(programmatic_domain_rule_target_size))
            )
            domain_rule_rows, domain_rule_scan_summary = _scan_programmatic_domain_rule_rows(
                eval_id=eval_id,
                target_size=domain_rule_target_size,
                max_scan=max_scan,
                seed_offset=seed_offset,
                answer_type_filter=answer_type_filter,
                subject_contains=subject_contains,
                logger=logger,
                log_interval=diagnostic_log_interval,
            )
            rows = _merge_preflight_rows(rows, domain_rule_rows)
            log_event(
                logger,
                {
                    "event": "hle_operator_cohort_preflight_rows_merged",
                    "eval_id": eval_id,
                    "operator_row_count": len(scan_summary.get("selected_rows", []) or [])
                    if isinstance(scan_summary.get("selected_rows"), list)
                    else int(scan_summary.get("selected") or 0),
                    "programmatic_domain_rule_row_count": len(domain_rule_rows),
                    "merged_row_count": len(rows),
                    "duplicate_row_count_removed": max(0, len(domain_rule_rows) + int(scan_summary.get("selected") or 0) - len(rows)),
                },
            )
            scan_summary = {
                "selected": len(rows),
                "operator": scan_summary,
                "programmatic_domain_rule": domain_rule_scan_summary,
            }
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    metrics = _cohort_metrics(rows, scan_summary)
    log_event(
        logger,
        {
            "event": "hle_operator_cohort_preflight_completed",
            "eval_id": eval_id,
            "selected_count": metrics["selected_count"],
            "row_kind_counts": metrics["row_kind_counts"],
            "cohort_family_counts": metrics["cohort_family_counts"],
            "operator_family_counts": metrics["operator_family_counts"],
            "programmatic_domain_rule_counts": metrics["programmatic_domain_rule_counts"],
            "pass": metrics["pass"],
        },
    )
    return {
        "eval_id": eval_id,
        "eval_kind": "hle_operator_cohort_preflight",
        "dataset": {
            "name": DATASET_NAME,
            "answer_type_filter": answer_type_filter,
            "subject_contains": subject_contains,
            "raw_content_persisted": False,
        },
        "sampling": {
            "target_size": effective_target_size,
            "max_scan": max_scan,
            "seed_offset": seed_offset,
            "operator_family_targets": canonical_family_targets,
            "include_programmatic_domain_rules": bool(include_programmatic_domain_rules),
            "programmatic_domain_rule_target_size": (
                target_size
                if programmatic_domain_rule_target_size is None
                else max(0, int(programmatic_domain_rule_target_size))
            ),
        },
        "operator_policy": {
            "fallback_enabled": True,
            "fallback_min_score": fallback_min_score,
            "max_specs": max_specs,
            "applicability_probe_enabled": enable_applicability_probe,
            "applicability_probe_model": applicability_probe_model if enable_applicability_probe else "",
            "applicability_probe_min_slot_rate": applicability_probe_min_slot_rate,
            "relevance_gate": (
                "fallback operators require answer-time operator-family trigger evidence; runtime "
                "retrieval/harness meta-surfaces are excluded by default"
            ),
        },
        "metrics": metrics,
        "rows": rows,
        "diagnostic_log_out": str(log_out) if log_out else None,
        "logging_policy": {
            "event_stream": "jsonl",
            "raw_content_persisted": False,
            "prediction_text_persisted": False,
            "gold_answer_persisted": False,
            "event_granularity": "scan progress, metadata-only candidate decisions, selected rows, merge, completion",
        },
        "claim_boundary": (
            "This preflight selects a candidate HLE operator-bearing cohort. It does not score model answers "
            "and does not persist raw HLE questions, gold answers, rationales, canary strings, or probe "
            "prediction text. Programmatic domain-rule rows are selected only when the same answer-time "
            "domain-rule verifier emits a self-contained decision without retrieved evidence."
        ),
    }


def _scan_operator_rows(
    *,
    eval_id: str,
    graph: SimpleAssumptionGraph,
    target_size: int,
    max_scan: int,
    seed_offset: int,
    answer_type_filter: str,
    subject_contains: str,
    family_targets: dict[str, int],
    enable_applicability_probe: bool,
    applicability_probe_model: str,
    applicability_probe_max_tokens: int,
    applicability_probe_min_slot_rate: float,
    logger: JsonlDiagnosticLogger | None = None,
    log_interval: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if target_size <= 0 and not family_targets:
        log_event(
            logger,
            {
                "event": "hle_operator_preflight_operator_scan_skipped",
                "eval_id": eval_id,
                "scan_kind": "operator_spec",
                "reason": "target_size_zero",
            },
        )
        return [], {"selected": 0, "scanned": 0, "skipped_target_size_zero": 1}
    log_event(
        logger,
        {
            "event": "hle_operator_preflight_operator_scan_started",
            "eval_id": eval_id,
            "scan_kind": "operator_spec",
            "target_size": int(target_size),
            "max_scan": int(max_scan),
            "seed_offset": int(seed_offset),
            "family_targets": dict(family_targets),
            "applicability_probe_enabled": bool(enable_applicability_probe),
        },
    )
    dataset = _load_hle_test_dataset()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    scanned = 0
    for raw_row in dataset:
        scanned += 1
        if scanned <= seed_offset:
            counts["skipped_seed_offset"] += 1
            continue
        if scanned > max_scan:
            break
        if logger and log_interval > 0 and scanned % log_interval == 0:
            log_event(
                logger,
                {
                    "event": "hle_operator_preflight_scan_progress",
                    "eval_id": eval_id,
                    "scan_kind": "operator_spec",
                    "scanned": int(scanned),
                    "selected": len(rows),
                    "counts": dict(sorted(counts.items())),
                    "family_counts": dict(sorted(family_counts.items())),
                },
            )
        if _has_image_payload(raw_row):
            counts["skipped_image_payload"] += 1
            continue
        if not str(raw_row.get("question") or "").strip() or not str(raw_row.get("answer") or "").strip():
            counts["skipped_missing_text_or_answer"] += 1
            continue
        if answer_type_filter and str(raw_row.get("answer_type") or "") != answer_type_filter:
            counts["skipped_answer_type"] += 1
            continue
        if subject_contains:
            haystack = " ".join([
                str(raw_row.get("category") or ""),
                str(raw_row.get("raw_subject") or ""),
            ]).lower()
            if subject_contains.lower() not in haystack:
                counts["skipped_subject"] += 1
                continue

        problem = _problem_from_row(raw_row, scanned=scanned, skipped_before=sum(counts.values()))
        domain = _classify_hle_domain(problem)
        stage = _compile_hle_operator_stage(
            retrieval_result=SimpleNamespace(subgraph=SimpleNamespace(nodes=[])),
            graph=graph,
            problem_text=str(problem.get("_question") or ""),
            problem_id=str(problem.get("id_hash") or ""),
            domain=domain,
            difficulty="hle",
            context_allowed=False,
            generic_graph_context_only=True,
        )
        if stage.get("status") != "activated":
            counts[f"operator_{stage.get('status') or 'missing'}"] += 1
            reason = str(stage.get("reason") or "")
            if reason:
                counts[f"reason_{reason}"] += 1
            continue

        cohort_family = ""
        family_tags = _operator_family_tags_from_stage(stage)
        if family_targets:
            cohort_family = _choose_underfilled_family(
                family_tags=family_tags,
                family_counts=family_counts,
                family_targets=family_targets,
            )
            if not cohort_family:
                log_event(
                    logger,
                    {
                        "event": "hle_operator_preflight_operator_candidate",
                        "eval_id": eval_id,
                        "scan_kind": "operator_spec",
                        "scanned_index": int(problem.get("scanned_index") or 0),
                        "seed_offset": int(problem.get("scanned_index") or 0) - 1,
                        "problem_id_hash": problem.get("id_hash"),
                        "question_hash": problem.get("question_hash"),
                        "domain": domain,
                        "operator_status": stage.get("status"),
                        "operator_reason": stage.get("reason"),
                        "operator_source_ids": list(stage.get("operator_source_ids", []) or []),
                        "operator_family_tags": family_tags,
                        "selected": False,
                        "decision_reason": "family_targets_full_or_missing",
                    },
                )
                counts["skipped_family_targets_full_or_missing"] += 1
                for tag in family_tags or ["none"]:
                    counts[f"skipped_family_{tag}"] += 1
                continue

        applicability_probe = None
        if enable_applicability_probe:
            applicability_probe = _operator_applicability_probe(
                problem=problem,
                stage=stage,
                model=applicability_probe_model,
                max_tokens=applicability_probe_max_tokens,
                min_slot_rate=applicability_probe_min_slot_rate,
            )
            probe_status = str(applicability_probe.get("status") or "unknown")
            counts[f"probe_{probe_status}"] += 1
            if probe_status != "passed":
                reason = str(applicability_probe.get("reason") or "")
                if reason:
                    counts[f"probe_reason_{reason}"] += 1
                log_event(
                    logger,
                    {
                        "event": "hle_operator_preflight_operator_candidate",
                        "eval_id": eval_id,
                        "scan_kind": "operator_spec",
                        "scanned_index": int(problem.get("scanned_index") or 0),
                        "seed_offset": int(problem.get("scanned_index") or 0) - 1,
                        "problem_id_hash": problem.get("id_hash"),
                        "question_hash": problem.get("question_hash"),
                        "domain": domain,
                        "operator_status": stage.get("status"),
                        "operator_reason": stage.get("reason"),
                        "operator_source_ids": list(stage.get("operator_source_ids", []) or []),
                        "operator_family_tags": family_tags,
                        "selected": False,
                        "decision_reason": "applicability_probe_failed",
                        "probe_status": probe_status,
                        "probe_reason": reason,
                        "probe_slot_completion_rate": float(applicability_probe.get("slot_completion_rate") or 0.0),
                        "probe_decorative_use": bool(applicability_probe.get("decorative_use")),
                    },
                )
                continue

        row = _cohort_row(
            problem=problem,
            domain=domain,
            stage=stage,
            applicability_probe=applicability_probe,
            cohort_family=cohort_family,
        )
        rows.append(row)
        counts["selected"] += 1
        log_event(
            logger,
            {
                "event": "hle_operator_preflight_row_selected",
                "eval_id": eval_id,
                "scan_kind": "operator_spec",
                "scanned_index": row.get("scanned_index"),
                "seed_offset": row.get("seed_offset"),
                "problem_id_hash": row.get("problem_id_hash"),
                "question_hash": row.get("question_hash"),
                "domain": row.get("domain"),
                "cohort_family": row.get("cohort_family"),
                "operator_source_ids": row.get("operator_source_ids"),
                "operator_family_tags": row.get("operator_family_tags"),
                "required_slot_count": row.get("required_slot_count"),
                "required_slots": row.get("required_slots"),
                "applicability_probe_status": (
                    row.get("applicability_probe", {}).get("status")
                    if isinstance(row.get("applicability_probe"), dict)
                    else None
                ),
            },
        )
        if cohort_family:
            family_counts[cohort_family] += 1
            counts[f"selected_family_{cohort_family}"] += 1
        if family_targets and all(family_counts[family] >= target for family, target in family_targets.items()):
            break
        if not family_targets and len(rows) >= target_size:
            break
    counts["scanned"] = scanned
    for family, value in sorted(family_counts.items()):
        counts[f"family_count_{family}"] = int(value)
    log_event(
        logger,
        {
            "event": "hle_operator_preflight_operator_scan_completed",
            "eval_id": eval_id,
            "scan_kind": "operator_spec",
            "scanned": int(scanned),
            "selected": len(rows),
            "counts": dict(sorted(counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
        },
    )
    return rows, dict(counts)


def _scan_programmatic_domain_rule_rows(
    *,
    eval_id: str,
    target_size: int,
    max_scan: int,
    seed_offset: int,
    answer_type_filter: str,
    subject_contains: str,
    logger: JsonlDiagnosticLogger | None = None,
    log_interval: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_size <= 0:
        log_event(
            logger,
            {
                "event": "hle_operator_preflight_programmatic_scan_skipped",
                "eval_id": eval_id,
                "scan_kind": "programmatic_domain_rule",
                "reason": "target_size_zero",
            },
        )
        return [], {"selected": 0, "scanned": 0, "skipped_target_size_zero": 1}
    log_event(
        logger,
        {
            "event": "hle_operator_preflight_programmatic_scan_started",
            "eval_id": eval_id,
            "scan_kind": "programmatic_domain_rule",
            "target_size": int(target_size),
            "max_scan": int(max_scan),
            "seed_offset": int(seed_offset),
        },
    )
    dataset = _load_hle_test_dataset()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    scanned = 0
    for raw_row in dataset:
        scanned += 1
        if scanned <= seed_offset:
            counts["skipped_seed_offset"] += 1
            continue
        if scanned > max_scan:
            break
        if logger and log_interval > 0 and scanned % log_interval == 0:
            log_event(
                logger,
                {
                    "event": "hle_operator_preflight_scan_progress",
                    "eval_id": eval_id,
                    "scan_kind": "programmatic_domain_rule",
                    "scanned": int(scanned),
                    "selected": len(rows),
                    "counts": dict(sorted(counts.items())),
                    "family_counts": dict(sorted(family_counts.items())),
                },
            )
        if _has_image_payload(raw_row):
            counts["skipped_image_payload"] += 1
            continue
        if not str(raw_row.get("question") or "").strip() or not str(raw_row.get("answer") or "").strip():
            counts["skipped_missing_text_or_answer"] += 1
            continue
        if answer_type_filter and str(raw_row.get("answer_type") or "") != answer_type_filter:
            counts["skipped_answer_type"] += 1
            continue
        if subject_contains:
            haystack = " ".join([
                str(raw_row.get("category") or ""),
                str(raw_row.get("raw_subject") or ""),
            ]).lower()
            if subject_contains.lower() not in haystack:
                counts["skipped_subject"] += 1
                continue

        problem = _problem_from_row(raw_row, scanned=scanned, skipped_before=sum(counts.values()))
        stem, options = _split_multiple_choice_question(problem)
        if len(options) < 2:
            counts["skipped_options_not_parsed"] += 1
            continue
        candidate_tags = _programmatic_domain_rule_candidate_tags(problem=problem, stem=stem, options=options)
        for tag in candidate_tags:
            counts[f"candidate_family_{tag}"] += 1
        decision = _domain_rule_mc_decision(
            problem=problem,
            stem=stem,
            options=options,
            evidence_context="",
        )
        if not decision:
            counts["rule_not_triggered"] += 1
            continue
        rule_id = str(decision.get("rule_id") or "")
        family = _programmatic_domain_rule_family(rule_id)
        decision_base = {
            "event": "hle_operator_preflight_programmatic_rule_candidate",
            "eval_id": eval_id,
            "scan_kind": "programmatic_domain_rule",
            "scanned_index": int(problem.get("scanned_index") or 0),
            "seed_offset": int(problem.get("scanned_index") or 0) - 1,
            "problem_id_hash": problem.get("id_hash"),
            "question_hash": problem.get("question_hash"),
            "domain": _classify_hle_domain(problem),
            "candidate_tags": candidate_tags,
            "rule_id": rule_id,
            "mapped_family": family,
            "confidence": decision.get("confidence"),
            "evidence_required": bool(decision.get("evidence_required")),
        }
        if bool(decision.get("evidence_required")):
            counts["skipped_evidence_required_rule"] += 1
            log_event(
                logger,
                {
                    **decision_base,
                    "selected": False,
                    "decision_reason": "evidence_required",
                },
            )
            continue
        if not family:
            counts["skipped_unmapped_rule"] += 1
            log_event(
                logger,
                {
                    **decision_base,
                    "selected": False,
                    "decision_reason": "unmapped_rule",
                },
            )
            continue
        domain = _classify_hle_domain(problem)
        row = _programmatic_domain_rule_cohort_row(
            problem=problem,
            domain=domain,
            decision=decision,
            family=family,
            candidate_tags=candidate_tags,
        )
        rows.append(row)
        counts["selected"] += 1
        family_counts[family] += 1
        counts[f"selected_family_{family}"] += 1
        log_event(
            logger,
            {
                **decision_base,
                "selected": True,
                "decision_reason": "selected",
                "cohort_family": family,
                "selected_option_hash": row.get("programmatic_domain_rule", {}).get("selected_option_hash")
                if isinstance(row.get("programmatic_domain_rule"), dict)
                else None,
            },
        )
        if len(rows) >= target_size:
            break
    counts["scanned"] = scanned
    for family, value in sorted(family_counts.items()):
        counts[f"family_count_{family}"] = int(value)
    log_event(
        logger,
        {
            "event": "hle_operator_preflight_programmatic_scan_completed",
            "eval_id": eval_id,
            "scan_kind": "programmatic_domain_rule",
            "scanned": int(scanned),
            "selected": len(rows),
            "counts": dict(sorted(counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
        },
    )
    return rows, dict(counts)


def _merge_preflight_rows(
    operator_rows: list[dict[str, Any]],
    programmatic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(operator_rows)
    seen = {str(row.get("problem_id_hash") or "") for row in merged if row.get("problem_id_hash")}
    for row in programmatic_rows:
        problem_hash = str(row.get("problem_id_hash") or "")
        if problem_hash and problem_hash in seen:
            continue
        merged.append(row)
        if problem_hash:
            seen.add(problem_hash)
    return merged


def _programmatic_domain_rule_family(rule_id: Any) -> str:
    return _PROGRAMMATIC_DOMAIN_RULE_FAMILIES.get(str(rule_id or ""), "")


def _programmatic_domain_rule_candidate_tags(
    *,
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
) -> list[str]:
    text = " ".join([
        str(problem.get("category") or ""),
        str(problem.get("raw_subject") or ""),
        stem,
        " ".join(str(value) for value in options.values()),
    ]).lower()
    tags: set[str] = set()
    if "sec-mals" in text or "multi-angle light scattering" in text:
        tags.add("sec_mals_mass_balance")
    if any(token in text for token in ("stoichiometry", "mass balance", "kda", "molecular mass")):
        tags.add("stoichiometry_mass_balance")
    if "bacteria" in text and "resistance" in text:
        tags.add("cross_resistance_minimality")
    if any(token in text for token in ("controlled variable", "control group", "all else equal", "controlled for")):
        tags.add("causal_controlled_variable")
    if any(token in text for token in ("experiment", "condition", "replacement", "migration", "substitution")):
        tags.add("experimental_condition_replacement")
    if any(token in text for token in ("none of the above", "contradiction", "except", "least likely")):
        tags.add("structured_option_contradiction")
    return sorted(tags)


def _programmatic_domain_rule_cohort_row(
    *,
    problem: dict[str, Any],
    domain: str,
    decision: dict[str, Any],
    family: str,
    candidate_tags: list[str],
) -> dict[str, Any]:
    return {
        "row_kind": "programmatic_domain_rule",
        "seed_offset": int(problem.get("scanned_index") or 0) - 1,
        "scanned_index": int(problem.get("scanned_index") or 0),
        "problem_id_hash": problem.get("id_hash"),
        "question_hash": problem.get("question_hash"),
        "category": problem.get("category"),
        "raw_subject": problem.get("raw_subject"),
        "answer_type": problem.get("answer_type"),
        "domain": domain,
        "operator_status": "not_applicable_programmatic_domain_rule",
        "operator_reason": "selected_by_answer_time_domain_rule_verifier",
        "operator_source_ids": [],
        "operator_source_types": [],
        "operator_family_tags": [],
        "cohort_family": family,
        "programmatic_domain_rule_family_tags": candidate_tags,
        "required_slot_count": 0,
        "verifier_check_count": 1,
        "fallback_top_node_ids": [],
        "fallback_top_scores": [],
        "relevance_kept_node_ids": [],
        "relevance_kept_answer_time_family_hits": {},
        "relevance_rejected_reasons": {},
        "required_slots": [],
        "programmatic_domain_rule": {
            "status": "activated",
            "rule_id": decision.get("rule_id"),
            "family": family,
            "confidence": decision.get("confidence"),
            "evidence_required": bool(decision.get("evidence_required")),
            "selected_option_hash": stable_hash({"option_label": decision.get("label")}),
        },
    }


def _operator_applicability_probe(
    *,
    problem: dict[str, Any],
    stage: dict[str, Any],
    model: str,
    max_tokens: int,
    min_slot_rate: float,
    call_model: Callable[..., str] = _call_model,
) -> dict[str, Any]:
    specs = [spec for spec in stage.get("operator_specs", []) or [] if isinstance(spec, dict)]
    source_ids = [str(value) for value in stage.get("operator_source_ids", []) or [] if str(value)]
    required_slots = _operator_required_slots(specs)
    base = {
        "status": "failed",
        "reason": "not_run",
        "model": model,
        "raw_content_persisted": False,
        "prediction_text_persisted": False,
        "used_operator_ids": [],
        "required_slots_filled": [],
        "slot_completion_rate": 0.0,
        "decorative_use": True,
    }
    if not specs or not source_ids:
        return {**base, "reason": "missing_operator_specs"}
    prompt = (
        f"{_operator_application_answer_prompt(specs=specs)}\n\n"
        f"Answer type: {problem.get('answer_type')}\n"
        f"Question:\n{problem.get('_question')}\n\n"
        "Keep the operator_audit object required above. For multiple choice, the `answer` field must be the "
        "single option letter only; for exact match, the `answer` field must be the shortest exact answer."
    )
    try:
        text = call_model(model=model, prompt=prompt, timeout=None, max_tokens=max_tokens)
    except Exception as exc:
        return {**base, "status": "error", "reason": type(exc).__name__}
    audit = _parse_operator_child_audit_json(
        text,
        source_ids=source_ids,
        required_slots=required_slots,
    )
    slot_rate = float(audit.get("slot_completion_rate") or 0.0) if audit else 0.0
    decorative = bool(audit.get("decorative_use")) if audit else True
    used_ids = list(audit.get("used_operator_ids", []) or []) if audit else []
    filled_slots = list(audit.get("required_slots_filled", []) or []) if audit else []
    passed = bool(used_ids and slot_rate >= min_slot_rate and not decorative)
    reason = "operator_child_audit_passed" if passed else "operator_child_audit_failed"
    return {
        **base,
        "status": "passed" if passed else "failed",
        "reason": reason,
        "used_operator_ids": used_ids,
        "required_slots_filled": filled_slots,
        "slot_completion_rate": round(slot_rate, 4),
        "decorative_use": decorative,
        "min_slot_rate": min_slot_rate,
    }


def _canonical_operator_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = _OPERATOR_FAMILY_ALIASES.get(text, text)
    allowed = {
        "answer_bearing_relation",
        "controlled_variable",
        "incremental_replacement",
        "structural_transfer",
    }
    return text if text in allowed else ""


def _canonicalize_family_targets(family_targets: dict[str, int]) -> dict[str, int]:
    canonical: Counter[str] = Counter()
    for family, target in (family_targets or {}).items():
        canonical_family = _canonical_operator_family(family)
        try:
            target_int = int(target)
        except (TypeError, ValueError):
            target_int = 0
        if canonical_family and target_int > 0:
            canonical[canonical_family] += target_int
    return dict(sorted(canonical.items()))


def _parse_family_targets(text: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for chunk in str(text or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" in item:
            family, raw_count = item.split("=", 1)
        elif ":" in item:
            family, raw_count = item.split(":", 1)
        else:
            family, raw_count = item, "1"
        canonical_family = _canonical_operator_family(family)
        if not canonical_family:
            continue
        try:
            count = int(raw_count.strip())
        except ValueError:
            continue
        if count > 0:
            targets[canonical_family] = targets.get(canonical_family, 0) + count
    return dict(sorted(targets.items()))


def _operator_family_tags_from_stage(stage: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    for source_id in stage.get("operator_source_ids", []) or []:
        family = _canonical_operator_family(_BUILTIN_OPERATOR_FAMILIES.get(str(source_id), ""))
        if family:
            tags.add(family)
    for summary in [
        stage.get("operator_admissibility") if isinstance(stage.get("operator_admissibility"), dict) else {},
        (
            stage.get("fallback_retrieval", {}).get("relevance_gate", {})
            if isinstance(stage.get("fallback_retrieval"), dict)
            and isinstance(stage.get("fallback_retrieval", {}).get("relevance_gate"), dict)
            else {}
        ),
    ]:
        for field in ("kept_answer_time_family_hits", "kept_family_hits"):
            family_map = summary.get(field) if isinstance(summary, dict) else {}
            if not isinstance(family_map, dict):
                continue
            for values in family_map.values():
                for value in values or []:
                    family = _canonical_operator_family(value)
                    if family:
                        tags.add(family)
    return sorted(tags)


def _choose_underfilled_family(
    *,
    family_tags: list[str],
    family_counts: Counter[str],
    family_targets: dict[str, int],
) -> str:
    candidates = [
        family
        for family in family_tags
        if family in family_targets and family_counts[family] < family_targets[family]
    ]
    if not candidates:
        return ""
    return sorted(
        candidates,
        key=lambda family: (
            -(family_targets[family] - family_counts[family]),
            family_counts[family],
            family,
        ),
    )[0]


def _cohort_row(
    *,
    problem: dict[str, Any],
    domain: str,
    stage: dict[str, Any],
    applicability_probe: dict[str, Any] | None = None,
    cohort_family: str = "",
) -> dict[str, Any]:
    fallback = stage.get("fallback_retrieval") if isinstance(stage.get("fallback_retrieval"), dict) else {}
    relevance = fallback.get("relevance_gate") if isinstance(fallback.get("relevance_gate"), dict) else {}
    specs = list(stage.get("operator_specs", []) or [])
    family_tags = _operator_family_tags_from_stage(stage)
    row = {
        "row_kind": "operator_spec",
        "seed_offset": int(problem.get("scanned_index") or 0) - 1,
        "scanned_index": int(problem.get("scanned_index") or 0),
        "problem_id_hash": problem.get("id_hash"),
        "question_hash": problem.get("question_hash"),
        "category": problem.get("category"),
        "raw_subject": problem.get("raw_subject"),
        "answer_type": problem.get("answer_type"),
        "domain": domain,
        "operator_status": stage.get("status"),
        "operator_reason": stage.get("reason"),
        "operator_source_ids": list(stage.get("operator_source_ids", []) or []),
        "operator_source_types": list(stage.get("operator_source_types", []) or []),
        "operator_family_tags": family_tags,
        "cohort_family": cohort_family or (family_tags[0] if len(family_tags) == 1 else ""),
        "required_slot_count": int(stage.get("required_slot_count") or 0),
        "verifier_check_count": int(stage.get("verifier_check_count") or 0),
        "fallback_top_node_ids": list(fallback.get("top_node_ids", []) or [])[:6],
        "fallback_top_scores": list(fallback.get("top_scores", []) or [])[:6],
        "relevance_kept_node_ids": list(relevance.get("kept_node_ids", []) or []),
        "relevance_kept_answer_time_family_hits": dict(relevance.get("kept_answer_time_family_hits", {}) or {}),
        "relevance_rejected_reasons": dict(relevance.get("rejected_reasons", {}) or {}),
        "required_slots": sorted({
            str(slot)
            for spec in specs
            if isinstance(spec, dict)
            for slot in spec.get("required_output_slots", []) or []
        }),
    }
    if applicability_probe is not None:
        row["applicability_probe"] = {
            "status": applicability_probe.get("status"),
            "reason": applicability_probe.get("reason"),
            "used_operator_ids": list(applicability_probe.get("used_operator_ids", []) or []),
            "required_slots_filled": list(applicability_probe.get("required_slots_filled", []) or []),
            "slot_completion_rate": float(applicability_probe.get("slot_completion_rate") or 0.0),
            "decorative_use": bool(applicability_probe.get("decorative_use")),
            "raw_content_persisted": False,
            "prediction_text_persisted": False,
        }
    return row


def _cohort_metrics(rows: list[dict[str, Any]], scan_summary: dict[str, Any]) -> dict[str, Any]:
    operator_source_counts: Counter[str] = Counter()
    operator_family_counts: Counter[str] = Counter()
    cohort_family_counts: Counter[str] = Counter()
    row_kind_counts: Counter[str] = Counter()
    programmatic_rule_counts: Counter[str] = Counter()
    programmatic_family_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    probe_status_counts: Counter[str] = Counter()
    probe_slot_rates: list[float] = []
    for row in rows:
        row_kind_counts[str(row.get("row_kind") or "operator_spec")] += 1
        domain_counts[str(row.get("domain") or "")] += 1
        subject_counts[str(row.get("raw_subject") or "")] += 1
        for source_id in row.get("operator_source_ids", []) or []:
            operator_source_counts[str(source_id)] += 1
        for family in row.get("operator_family_tags", []) or []:
            operator_family_counts[str(family)] += 1
        if row.get("cohort_family"):
            cohort_family_counts[str(row.get("cohort_family"))] += 1
        domain_rule = row.get("programmatic_domain_rule")
        domain_rule = domain_rule if isinstance(domain_rule, dict) else {}
        if domain_rule:
            programmatic_rule_counts[str(domain_rule.get("rule_id") or "")] += 1
            programmatic_family_counts[str(domain_rule.get("family") or row.get("cohort_family") or "")] += 1
        for slot in row.get("required_slots", []) or []:
            slot_counts[str(slot)] += 1
        probe = row.get("applicability_probe") if isinstance(row.get("applicability_probe"), dict) else {}
        if probe:
            probe_status_counts[str(probe.get("status") or "")] += 1
            probe_slot_rates.append(float(probe.get("slot_completion_rate") or 0.0))
    return {
        "selected_count": len(rows),
        "scan_summary": scan_summary,
        "row_kind_counts": dict(sorted(row_kind_counts.items())),
        "operator_source_counts": dict(sorted(operator_source_counts.items())),
        "operator_family_counts": dict(sorted(operator_family_counts.items())),
        "programmatic_domain_rule_counts": dict(sorted(programmatic_rule_counts.items())),
        "programmatic_domain_rule_family_counts": dict(sorted(programmatic_family_counts.items())),
        "cohort_family_counts": dict(sorted(cohort_family_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "raw_subject_counts": dict(sorted(subject_counts.items())),
        "required_slot_counts": dict(sorted(slot_counts.items())),
        "applicability_probe_status_counts": dict(sorted(probe_status_counts.items())),
        "applicability_probe_average_slot_completion": (
            round(sum(probe_slot_rates) / len(probe_slot_rates), 4) if probe_slot_rates else None
        ),
        "pass": bool(rows),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        f"# {payload.get('eval_id')}",
        "",
        f"- selected count: `{metrics.get('selected_count')}`",
        f"- scan summary: `{metrics.get('scan_summary')}`",
        f"- row kind counts: `{metrics.get('row_kind_counts')}`",
        f"- operator source counts: `{metrics.get('operator_source_counts')}`",
        f"- operator family counts: `{metrics.get('operator_family_counts')}`",
        f"- programmatic domain-rule counts: `{metrics.get('programmatic_domain_rule_counts')}`",
        f"- programmatic domain-rule family counts: `{metrics.get('programmatic_domain_rule_family_counts')}`",
        f"- cohort family counts: `{metrics.get('cohort_family_counts')}`",
        f"- domain counts: `{metrics.get('domain_counts')}`",
        "",
        "## Cohort Rows",
        "",
        "| seed offset | kind | problem hash | subject | domain | family | rule/operator ids | required slots |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("rows", []) or []:
        domain_rule = row.get("programmatic_domain_rule")
        domain_rule = domain_rule if isinstance(domain_rule, dict) else {}
        rule_or_operator_ids = domain_rule.get("rule_id") or row.get("operator_source_ids")
        lines.append(
            "| `{seed}` | `{kind}` | `{pid}` | `{subject}` | `{domain}` | `{family}` | `{rule_or_operator_ids}` | `{slots}` |".format(
                seed=row.get("seed_offset"),
                kind=row.get("row_kind") or "operator_spec",
                pid=row.get("problem_id_hash"),
                subject=row.get("raw_subject"),
                domain=row.get("domain"),
                family=row.get("cohort_family") or row.get("operator_family_tags"),
                rule_or_operator_ids=rule_or_operator_ids,
                slots=row.get("required_slots"),
            )
        )
    lines.extend(["", "## Claim Boundary", "", str(payload.get("claim_boundary") or "")])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a metadata-only HLE operator cohort preflight.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_operator_cohort_preflight_20260621")
    parser.add_argument("--target-size", type=int, default=12)
    parser.add_argument("--max-scan", type=int, default=2000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--sample-answer-type", default="multipleChoice")
    parser.add_argument("--sample-subject-contains", default="")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--assumption-operator-fallback-min-score", type=float, default=0.145)
    parser.add_argument("--assumption-operator-max-specs", type=int, default=2)
    parser.add_argument(
        "--operator-family-targets",
        default="",
        help=(
            "Optional comma-separated family quotas, e.g. "
            "controlled_variable=10,incremental_replacement=10,structural_transfer=10"
        ),
    )
    parser.add_argument("--enable-operator-applicability-probe", action="store_true")
    parser.add_argument("--operator-applicability-probe-model", default="gpt-5.4-mini")
    parser.add_argument("--operator-applicability-probe-max-tokens", type=int, default=512)
    parser.add_argument("--operator-applicability-probe-min-slot-rate", type=float, default=0.75)
    parser.add_argument("--include-programmatic-domain-rules", action="store_true")
    parser.add_argument("--programmatic-domain-rule-target-size", type=int, default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument(
        "--log-out",
        default="",
        help="Metadata-only JSONL diagnostic log. Defaults to the output JSON path with .jsonl suffix.",
    )
    parser.add_argument("--diagnostic-log-interval", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    graph_dir = Path(args.graph_dir)
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    md_out = Path(args.md_out)
    md_out = md_out if md_out.is_absolute() else root / md_out
    log_out = Path(args.log_out) if args.log_out else out.with_suffix(".jsonl")
    log_out = log_out if log_out.is_absolute() else root / log_out
    payload = build_hle_operator_cohort_preflight_payload(
        root=root,
        eval_id=args.eval_id,
        target_size=args.target_size,
        max_scan=args.max_scan,
        seed_offset=args.seed_offset,
        answer_type_filter=args.sample_answer_type,
        subject_contains=args.sample_subject_contains,
        graph_dir=graph_dir,
        fallback_min_score=args.assumption_operator_fallback_min_score,
        max_specs=args.assumption_operator_max_specs,
        family_targets=_parse_family_targets(args.operator_family_targets),
        enable_applicability_probe=args.enable_operator_applicability_probe,
        applicability_probe_model=args.operator_applicability_probe_model,
        applicability_probe_max_tokens=args.operator_applicability_probe_max_tokens,
        applicability_probe_min_slot_rate=args.operator_applicability_probe_min_slot_rate,
        include_programmatic_domain_rules=args.include_programmatic_domain_rules,
        programmatic_domain_rule_target_size=args.programmatic_domain_rule_target_size,
        log_out=log_out,
        diagnostic_log_interval=args.diagnostic_log_interval,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "selected_count": payload["metrics"]["selected_count"],
        "out": str(out),
        "md_out": str(md_out),
        "log_out": str(log_out),
        "pass": payload["metrics"]["pass"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
