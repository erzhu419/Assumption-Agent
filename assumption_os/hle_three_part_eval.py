"""Build a three-part HLE evaluation report from persisted metadata artifacts.

The report intentionally reads only aggregate metrics, shard rows, and JSONL
events that already avoid raw HLE content.  It separates the three claims called
out in GPT_analysis.md:

1. final answer-quality gain,
2. assumption/operator application fidelity,
3. residual-family or graph-evolution evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .diagnostic_logging import JsonlDiagnosticLogger, log_event


DEFAULT_RUN_DIR = PAPER_DIR / "hle_parallel_runs"
DEFAULT_MD_DIR = Path("reconstruction/md")


def build_three_part_eval_payload(
    *,
    run_dir: Path,
    eval_id: str,
    baseline_eval_id: str = "",
    preflight_path: Path | None = None,
    baseline_preflight_path: Path | None = None,
    residual_family_protocol: bool = False,
    log_out: Path | None = None,
) -> dict[str, Any]:
    logger = JsonlDiagnosticLogger(log_out) if log_out else None
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_started",
            "eval_id": eval_id,
            "baseline_eval_id": baseline_eval_id or None,
            "run_dir": str(run_dir),
            "preflight_path": str(preflight_path) if preflight_path else None,
            "baseline_preflight_path": str(baseline_preflight_path) if baseline_preflight_path else None,
            "residual_family_protocol": bool(residual_family_protocol),
        },
    )
    aggregate = _load_json(run_dir / f"{eval_id}.json")
    rows = _load_shard_rows(run_dir=run_dir, eval_id=eval_id)
    events = _load_jsonl_events(run_dir=run_dir, eval_id=eval_id)
    family_by_problem = _load_preflight_family_map(preflight_path) if preflight_path else {}
    baseline_family_by_problem = (
        _load_preflight_family_map(baseline_preflight_path)
        if baseline_preflight_path
        else family_by_problem
    )
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_artifacts_loaded",
            "eval_id": eval_id,
            "aggregate_loaded": True,
            "row_count": len(rows),
            "jsonl_event_count": len(events),
            "family_problem_count": len(family_by_problem),
            "baseline_family_problem_count": len(baseline_family_by_problem),
            "aggregate_failed_gates": list(aggregate.get("failed_gates") or []),
            "aggregate_paper_clean_failed_gates": list(aggregate.get("paper_clean_failed_gates") or []),
        },
    )

    baseline_payload: dict[str, Any] | None = None
    baseline_rows_for_family: list[dict[str, Any]] = []
    if baseline_eval_id:
        baseline_rows = _load_shard_rows(run_dir=run_dir, eval_id=baseline_eval_id)
        baseline_rows_for_family = baseline_rows
        baseline_aggregate = _load_json(run_dir / f"{baseline_eval_id}.json")
        baseline_payload = _baseline_comparison(
            current_rows=rows,
            baseline_rows=baseline_rows,
            current_aggregate=aggregate,
            baseline_aggregate=baseline_aggregate,
        )
        log_event(
            logger,
            {
                "event": "hle_three_part_eval_baseline_loaded",
                "eval_id": eval_id,
                "baseline_eval_id": baseline_eval_id,
                "baseline_row_count": len(baseline_rows),
                "shared_problem_count": baseline_payload.get("shared_problem_count"),
                "agent_improved_count": baseline_payload.get("agent_improved_count"),
                "agent_regressed_count": baseline_payload.get("agent_regressed_count"),
            },
        )

    answer_quality = _answer_quality_panel(aggregate=aggregate, rows=rows)
    application_fidelity = _application_fidelity_panel(aggregate=aggregate, rows=rows, events=events)
    programmatic_domain_rule_fidelity = _programmatic_domain_rule_fidelity_panel(rows=rows, events=events)
    residual_family = _residual_family_panel(
        rows=rows,
        family_by_problem=family_by_problem,
        baseline_family_by_problem=baseline_family_by_problem,
        baseline_payload=baseline_payload,
        baseline_rows=baseline_rows_for_family,
        residual_family_protocol=residual_family_protocol,
    )
    safety = _metadata_safety_panel(aggregate=aggregate, rows=rows)
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_panel_answer_quality",
            "eval_id": eval_id,
            "agent_above_raw": bool(answer_quality.get("agent_above_raw")),
            "agent_above_hipporag": bool(answer_quality.get("agent_above_hipporag")),
            "agent_minus_raw_accuracy": answer_quality.get("agent_minus_raw_accuracy"),
            "agent_minus_hipporag_accuracy": answer_quality.get("agent_minus_hipporag_accuracy"),
            "by_variant": answer_quality.get("by_variant"),
        },
    )
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_panel_operator_fidelity",
            "eval_id": eval_id,
            "passed": bool(application_fidelity.get("passed")),
            "application_coverage_present": bool(application_fidelity.get("application_coverage_present")),
            "applied_row_count": application_fidelity.get("applied_row_count"),
            "application_coverage_rate": application_fidelity.get("application_coverage_rate"),
            "direct_operator_selection_count": application_fidelity.get("direct_operator_selection_count"),
            "operator_defer_reason_counts": application_fidelity.get("operator_defer_reason_counts"),
            "source_defer_reason_counts": application_fidelity.get("operator_source_defer_reason_counts"),
        },
    )
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_panel_programmatic_domain_rule",
            "eval_id": eval_id,
            "passed": bool(programmatic_domain_rule_fidelity.get("passed")),
            "coverage_present": bool(programmatic_domain_rule_fidelity.get("coverage_present")),
            "activated_count": programmatic_domain_rule_fidelity.get("activated_count"),
            "selected_count": programmatic_domain_rule_fidelity.get("selected_count"),
            "selected_known_correct_rate": programmatic_domain_rule_fidelity.get("selected_known_correct_rate"),
            "rule_counts": programmatic_domain_rule_fidelity.get("rule_counts"),
        },
    )
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_panel_residual_family",
            "eval_id": eval_id,
            "residual_family_before_after_measured": bool(
                residual_family.get("residual_family_before_after_measured")
            ),
            "residual_family_learning_measured": bool(residual_family.get("residual_family_learning_measured")),
            "agent_error_rate_delta_by_family": residual_family.get("agent_error_rate_delta_by_family"),
            "family_problem_count": residual_family.get("family_problem_count"),
            "baseline_family_problem_count": residual_family.get("baseline_family_problem_count"),
        },
    )
    pass_summary = {
        "answer_quality_agent_above_raw_and_hipporag": bool(
            answer_quality.get("agent_above_raw") and answer_quality.get("agent_above_hipporag")
        ),
        "operator_application_fidelity_passed": bool(application_fidelity.get("passed")),
        "applied_row_fidelity_passed": bool(application_fidelity.get("applied_row_fidelity_passed")),
        "operator_application_coverage_present": bool(application_fidelity.get("application_coverage_present")),
        "operator_application_evidence_passed": bool(application_fidelity.get("operator_application_evidence_passed")),
        "programmatic_domain_rule_fidelity_passed": bool(programmatic_domain_rule_fidelity.get("passed")),
        "programmatic_domain_rule_coverage_present": bool(
            programmatic_domain_rule_fidelity.get("coverage_present")
        ),
        "programmatic_domain_rule_selected": bool(programmatic_domain_rule_fidelity.get("selected_count")),
        "residual_family_learning_measured": bool(residual_family.get("residual_family_learning_measured")),
        "residual_family_before_after_measured": bool(
            residual_family.get("residual_family_before_after_measured")
        ),
        "raw_content_not_persisted": bool(safety.get("raw_content_not_persisted")),
        "paper_clean_pass": bool(aggregate.get("paper_clean_pass")),
    }
    log_event(
        logger,
        {
            "event": "hle_three_part_eval_completed",
            "eval_id": eval_id,
            "pass_summary": pass_summary,
            "raw_content_not_persisted": bool(safety.get("raw_content_not_persisted")),
        },
    )
    return {
        "eval_id": eval_id,
        "baseline_eval_id": baseline_eval_id or None,
        "eval_kind": "hle_three_part_eval",
        "raw_content_persisted": False,
        "panels": {
            "answer_quality": answer_quality,
            "application_fidelity": application_fidelity,
            "programmatic_domain_rule_fidelity": programmatic_domain_rule_fidelity,
            "residual_family": residual_family,
            "metadata_safety": safety,
            "baseline_comparison": baseline_payload,
        },
        "pass_summary": pass_summary,
        "diagnostic_log_out": str(log_out) if log_out else None,
        "logging_policy": {
            "event_stream": "jsonl",
            "raw_content_persisted": False,
            "prediction_text_persisted": False,
            "gold_answer_persisted": False,
            "event_granularity": "artifact load counts, panel summaries, pass summary",
        },
        "claim_boundary": (
            "This artifact can support a small triggered-cohort answer-quality observation and "
            "operator no-harm selector diagnostics. It does not by itself prove residual-family "
            "learning or graph self-evolution unless a multi-round residual-family before/after "
            "run is supplied."
        ),
    }


def _answer_quality_panel(*, aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = _variant_accuracy(rows)
    agent = by_variant.get("assumption_agent_recursive_verify", {})
    raw = by_variant.get("raw", {})
    hippo = by_variant.get("hipporag_baseline", {})
    gain_loss = ((aggregate.get("failure_diagnostics") or {}).get("agent_gain_loss") or {})
    return {
        "by_variant": by_variant,
        "agent_minus_raw_accuracy": _round_delta(agent.get("accuracy"), raw.get("accuracy")),
        "agent_minus_hipporag_accuracy": _round_delta(agent.get("accuracy"), hippo.get("accuracy")),
        "agent_above_raw": _gt(agent.get("accuracy"), raw.get("accuracy")),
        "agent_above_hipporag": _gt(agent.get("accuracy"), hippo.get("accuracy")),
        "agent_gain_loss": gain_loss,
        "failed_gates": list(aggregate.get("failed_gates") or []),
        "paper_clean_failed_gates": list(aggregate.get("paper_clean_failed_gates") or []),
    }


def _application_fidelity_panel(
    *,
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = aggregate.get("metrics") or {}
    application = metrics.get("operator_application_summary") or {}
    activation = metrics.get("operator_activation_summary") or {}
    applied_row_count = _as_int(application.get("applied_row_count"))
    application_coverage_rate = _as_float(application.get("application_coverage_rate"))
    applied_row_fidelity_passed = bool(application.get("passed"))
    coverage_present = applied_row_count > 0 or application_coverage_rate > 0.0
    agent_rows = [row for row in rows if row.get("variant") == "assumption_agent_recursive_verify"]
    selection_counts: Counter[str] = Counter()
    selection_correct: Counter[str] = Counter()
    gate_reasons: Counter[str] = Counter()
    direct_operator_rows = 0
    direct_operator_correct = 0
    for row in agent_rows:
        selection = _row_selection(row)
        method = str(selection.get("selection_method") or "none")
        selection_counts[method] += 1
        if row.get("correct"):
            selection_correct[method] += 1
        gate = selection.get("verified_or_abstain_gate")
        if isinstance(gate, dict):
            gate_reasons[str(gate.get("reason") or "none")] += 1
        if method == "operator_application_fidelity_choice":
            direct_operator_rows += 1
            if row.get("correct"):
                direct_operator_correct += 1
    defer_reasons: Counter[str] = Counter()
    source_defer_reasons: Counter[str] = Counter()
    source_adjudicator_events = 0
    for event in events:
        if event.get("event") == "operator_application_selection_deferred":
            defer_reasons[str(event.get("reason") or "none")] += 1
            source = event.get("source_defer")
            source = source if isinstance(source, dict) else {}
            source_defer_reasons[str(source.get("reason") or "none")] += 1
        elif event.get("event") == "operator_source_grounded_adjudicator":
            source_adjudicator_events += 1
    return {
        "passed": bool(applied_row_fidelity_passed and coverage_present),
        "applied_row_fidelity_passed": applied_row_fidelity_passed,
        "application_coverage_present": coverage_present,
        "operator_application_evidence_passed": bool(applied_row_fidelity_passed and coverage_present),
        "application_coverage_rate": application_coverage_rate,
        "applied_row_count": applied_row_count,
        "operator_activation_summary": activation,
        "operator_application_summary": application,
        "agent_selection_method_counts": dict(sorted(selection_counts.items())),
        "agent_selection_method_accuracy": {
            method: round(selection_correct[method] / count, 4)
            for method, count in sorted(selection_counts.items())
            if count
        },
        "verified_or_abstain_gate_reasons": dict(sorted(gate_reasons.items())),
        "direct_operator_selection_count": direct_operator_rows,
        "direct_operator_selection_correct_count": direct_operator_correct,
        "operator_defer_reason_counts": dict(sorted(defer_reasons.items())),
        "operator_source_defer_reason_counts": dict(sorted(source_defer_reasons.items())),
        "operator_source_adjudicator_event_count": source_adjudicator_events,
    }


def _programmatic_domain_rule_fidelity_panel(
    *,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    agent_rows = [row for row in rows if row.get("variant") == "assumption_agent_recursive_verify"]
    status_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    activated_count = 0
    selected_count = 0
    selected_correct_known_count = 0
    selected_correct_count = 0
    selected_incorrect_count = 0
    short_circuit_count = 0
    disabled_count = 0
    evidence_required_count = 0
    for row in agent_rows:
        efficacy = row.get("component_efficacy")
        efficacy = efficacy if isinstance(efficacy, dict) else {}
        domain_rule = efficacy.get("domain_rule_mc_verifier")
        domain_rule = domain_rule if isinstance(domain_rule, dict) else {}
        recursive = efficacy.get("recursive")
        recursive = recursive if isinstance(recursive, dict) else {}
        selection = efficacy.get("selection")
        selection = selection if isinstance(selection, dict) else {}
        status = str(domain_rule.get("status") or "not_present")
        status_counts[status] += 1
        if status == "disabled":
            disabled_count += 1
        if status != "activated":
            continue
        activated_count += 1
        rule_id = str(domain_rule.get("rule_id") or "unknown")
        rule_counts[rule_id] += 1
        if bool(domain_rule.get("evidence_required")):
            evidence_required_count += 1
        method = str(selection.get("selection_method") or "")
        selection_counts[method or "none"] += 1
        selected = bool(domain_rule.get("selected_domain_rule_candidate")) or method == "domain_rule_verifier_priority"
        if selected:
            selected_count += 1
        if (
            bool(domain_rule.get("short_circuited_child_generation"))
            or recursive.get("execution_mode") == "domain_rule_preverified"
            or recursive.get("early_stop_reason") == "domain_rule_preverified"
        ):
            short_circuit_count += 1
        correctness = domain_rule.get("candidate_correct_for_eval")
        if selected and correctness is not None:
            selected_correct_known_count += 1
            if correctness is True:
                selected_correct_count += 1
            else:
                selected_incorrect_count += 1
    event_status_counts: Counter[str] = Counter()
    for event in events:
        if event.get("event") == "domain_rule_mc_verifier":
            event_status_counts[str(event.get("stage_status") or "unknown")] += 1
    coverage_present = activated_count > 0
    selected_rate = round(selected_count / activated_count, 4) if activated_count else 0.0
    known_correct_rate = (
        round(selected_correct_count / selected_correct_known_count, 4)
        if selected_correct_known_count
        else None
    )
    passed = bool(coverage_present and selected_count > 0 and selected_incorrect_count == 0)
    return {
        "passed": passed,
        "coverage_present": coverage_present,
        "agent_row_count": len(agent_rows),
        "activated_count": activated_count,
        "disabled_count": disabled_count,
        "selected_count": selected_count,
        "selected_rate": selected_rate,
        "selected_correct_known_count": selected_correct_known_count,
        "selected_correct_count": selected_correct_count,
        "selected_incorrect_count": selected_incorrect_count,
        "selected_known_correct_rate": known_correct_rate,
        "short_circuit_count": short_circuit_count,
        "short_circuit_rate": round(short_circuit_count / activated_count, 4) if activated_count else 0.0,
        "evidence_required_count": evidence_required_count,
        "status_counts": dict(sorted(status_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "selection_method_counts": dict(sorted(selection_counts.items())),
        "event_status_counts": dict(sorted(event_status_counts.items())),
    }


def _residual_family_panel(
    *,
    rows: list[dict[str, Any]],
    family_by_problem: dict[str, str],
    baseline_family_by_problem: dict[str, str],
    baseline_payload: dict[str, Any] | None,
    baseline_rows: list[dict[str, Any]],
    residual_family_protocol: bool,
) -> dict[str, Any]:
    by_family = _family_variant_accuracy(rows=rows, family_by_problem=family_by_problem)
    baseline_by_family = _family_variant_accuracy(
        rows=baseline_rows,
        family_by_problem=baseline_family_by_problem,
    )
    family_agent_error_delta = _family_agent_error_delta(
        current_by_family=by_family,
        baseline_by_family=baseline_by_family,
    )
    before_after_measured = bool(baseline_by_family and by_family)
    learning_measured = bool(residual_family_protocol and before_after_measured)
    if learning_measured:
        status = "measured_by_marked_residual_family_before_after_protocol"
    elif before_after_measured:
        status = "family_before_after_delta_measured_without_full_learning_claim"
    else:
        status = "not_measured_by_single_round_hle_ab"
    return {
        "family_map_provided": bool(family_by_problem),
        "baseline_family_map_provided": bool(baseline_rows and baseline_family_by_problem),
        "family_problem_count": len(set(family_by_problem.keys())),
        "baseline_family_problem_count": len(set(baseline_family_by_problem.keys())),
        "by_family": by_family,
        "baseline_by_family": baseline_by_family,
        "agent_error_rate_delta_by_family": family_agent_error_delta,
        "single_round_family_delta_measured": before_after_measured,
        "residual_family_before_after_measured": before_after_measured,
        "selector_before_after_comparison": baseline_payload,
        "residual_family_protocol_marked": bool(residual_family_protocol),
        "residual_family_learning_measured": learning_measured,
        "residual_family_learning_status": status,
        "required_next_experiment": (
            "Run a multi-round residual-family protocol: baseline residual exposure, candidate/operator "
            "generation, fresh validation, graph writeback, then unseen same-family test."
        ),
    }


def _metadata_safety_panel(*, aggregate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    persisted_rows = [
        row for row in rows
        if row.get("raw_question_persisted")
        or row.get("prediction_text_persisted")
        or row.get("gold_answer_persisted")
    ]
    row_errors = [row for row in rows if row.get("error")]
    return {
        "raw_content_not_persisted": aggregate.get("raw_content_persisted") is False and not persisted_rows,
        "aggregate_raw_content_persisted": bool(aggregate.get("raw_content_persisted")),
        "raw_content_persisted_row_count": len(persisted_rows),
        "row_error_count": len(row_errors),
        "pollution_pass": bool(aggregate.get("pollution_pass", True)),
        "runtime_policy": aggregate.get("runtime_policy") or {},
    }


def _baseline_comparison(
    *,
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    current_aggregate: dict[str, Any],
    baseline_aggregate: dict[str, Any],
) -> dict[str, Any]:
    current_agent = _agent_by_problem(current_rows)
    baseline_agent = _agent_by_problem(baseline_rows)
    shared = sorted(set(current_agent) & set(baseline_agent))
    improved = []
    regressed = []
    unchanged = 0
    for problem_hash in shared:
        old_correct = bool(baseline_agent[problem_hash].get("correct"))
        new_correct = bool(current_agent[problem_hash].get("correct"))
        if old_correct == new_correct:
            unchanged += 1
        elif new_correct:
            improved.append(problem_hash)
        else:
            regressed.append(problem_hash)
    return {
        "baseline_eval_id": baseline_aggregate.get("eval_id"),
        "current_eval_id": current_aggregate.get("eval_id"),
        "shared_problem_count": len(shared),
        "agent_improved_count": len(improved),
        "agent_regressed_count": len(regressed),
        "agent_unchanged_count": unchanged,
        "agent_improved_problem_hashes": improved,
        "agent_regressed_problem_hashes": regressed,
        "baseline_agent_accuracy": _variant_accuracy(baseline_rows).get(
            "assumption_agent_recursive_verify", {}
        ).get("accuracy"),
        "current_agent_accuracy": _variant_accuracy(current_rows).get(
            "assumption_agent_recursive_verify", {}
        ).get("accuracy"),
    }


def _variant_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for row in rows:
        variant = str(row.get("variant") or "unknown")
        counts[variant] += 1
        if row.get("correct"):
            correct[variant] += 1
        if row.get("error"):
            errors[variant] += 1
    return {
        variant: {
            "n": counts[variant],
            "correct": correct[variant],
            "accuracy": round(correct[variant] / counts[variant], 4) if counts[variant] else None,
            "error_count": errors[variant],
        }
        for variant in sorted(counts)
    }


def _family_variant_accuracy(
    *,
    rows: list[dict[str, Any]],
    family_by_problem: dict[str, str],
) -> dict[str, Any]:
    if not family_by_problem:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = family_by_problem.get(str(row.get("problem_id_hash") or ""))
        if family:
            grouped[family].append(row)
    return {
        family: _variant_accuracy(family_rows)
        for family, family_rows in sorted(grouped.items())
    }


def _family_agent_error_delta(
    *,
    current_by_family: dict[str, Any],
    baseline_by_family: dict[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for family in sorted(set(current_by_family) & set(baseline_by_family)):
        current_agent = (current_by_family.get(family) or {}).get("assumption_agent_recursive_verify") or {}
        baseline_agent = (baseline_by_family.get(family) or {}).get("assumption_agent_recursive_verify") or {}
        current_accuracy = current_agent.get("accuracy")
        baseline_accuracy = baseline_agent.get("accuracy")
        if current_accuracy is None or baseline_accuracy is None:
            continue
        current_error = 1.0 - float(current_accuracy)
        baseline_error = 1.0 - float(baseline_accuracy)
        deltas[family] = {
            "baseline_agent_error_rate": round(baseline_error, 4),
            "current_agent_error_rate": round(current_error, 4),
            "current_minus_baseline_error_rate": round(current_error - baseline_error, 4),
            "baseline_agent_n": baseline_agent.get("n"),
            "current_agent_n": current_agent.get("n"),
        }
    return deltas


def _agent_by_problem(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("problem_id_hash") or ""): row
        for row in rows
        if row.get("variant") == "assumption_agent_recursive_verify" and row.get("problem_id_hash")
    }


def _row_selection(row: dict[str, Any]) -> dict[str, Any]:
    efficacy = row.get("component_efficacy")
    efficacy = efficacy if isinstance(efficacy, dict) else {}
    selection = efficacy.get("selection")
    return selection if isinstance(selection, dict) else {}


def _load_shard_rows(*, run_dir: Path, eval_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = str(run_dir / f"{eval_id}_shard_*.json")
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        if path.name.endswith(".heartbeat.json"):
            continue
        payload = _load_json(path)
        rows.extend(payload.get("rows") or [])
    return rows


def _load_jsonl_events(*, run_dir: Path, eval_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pattern = str(run_dir / f"{eval_id}_shard_*.jsonl")
    for path_text in sorted(glob.glob(pattern)):
        with Path(path_text).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _load_preflight_family_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    rows = payload.get("rows") or payload.get("selected_rows") or payload.get("problems") or []
    out: dict[str, str] = {}
    for row in rows:
        problem_hash = str(row.get("problem_id_hash") or "")
        family = str(
            row.get("cohort_family")
            or row.get("operator_family")
            or _family_from_operator_tags(row.get("operator_family_tags"))
            or ""
        )
        if problem_hash and family:
            out[problem_hash] = family
    return out


def _family_from_operator_tags(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    tags = sorted({str(item).strip() for item in value if str(item).strip()})
    return "+".join(tags)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _round_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 4)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gt(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return float(left) > float(right)


def format_markdown(payload: dict[str, Any]) -> str:
    panels = payload.get("panels") or {}
    answer = panels.get("answer_quality") or {}
    fidelity = panels.get("application_fidelity") or {}
    domain_rule = panels.get("programmatic_domain_rule_fidelity") or {}
    residual = panels.get("residual_family") or {}
    safety = panels.get("metadata_safety") or {}
    lines = [
        f"# {payload.get('eval_id')}",
        "",
        "## Three-Part Verdict",
        "",
        f"- answer quality agent > raw: `{answer.get('agent_above_raw')}`",
        f"- answer quality agent > HippoRAG: `{answer.get('agent_above_hipporag')}`",
        f"- operator application evidence passed: `{fidelity.get('operator_application_evidence_passed')}`",
        f"- applied-row fidelity passed: `{fidelity.get('applied_row_fidelity_passed')}`",
        f"- application coverage present: `{fidelity.get('application_coverage_present')}`",
        f"- programmatic domain-rule fidelity passed: `{domain_rule.get('passed')}`",
        f"- programmatic domain-rule selected: `{domain_rule.get('selected_count')}`",
        f"- residual-family before/after measured: `{residual.get('residual_family_before_after_measured')}`",
        f"- residual-family learning measured: `{residual.get('residual_family_learning_measured')}`",
        f"- raw content not persisted: `{safety.get('raw_content_not_persisted')}`",
        "",
        "## A. Answer Quality",
        "",
        f"- by variant: `{answer.get('by_variant')}`",
        f"- agent minus raw accuracy: `{answer.get('agent_minus_raw_accuracy')}`",
        f"- agent minus HippoRAG accuracy: `{answer.get('agent_minus_hipporag_accuracy')}`",
        f"- gain/loss: `{answer.get('agent_gain_loss')}`",
        "",
        "## B1. OperatorSpec Fidelity",
        "",
        f"- application coverage rate: `{fidelity.get('application_coverage_rate')}`",
        f"- applied row count: `{fidelity.get('applied_row_count')}`",
        f"- direct operator selections: `{fidelity.get('direct_operator_selection_count')}`",
        f"- direct operator correct: `{fidelity.get('direct_operator_selection_correct_count')}`",
        f"- selection counts: `{fidelity.get('agent_selection_method_counts')}`",
        f"- gate reasons: `{fidelity.get('verified_or_abstain_gate_reasons')}`",
        f"- operator application summary: `{fidelity.get('operator_application_summary')}`",
        f"- operator defer reasons: `{fidelity.get('operator_defer_reason_counts')}`",
        f"- source defer reasons: `{fidelity.get('operator_source_defer_reason_counts')}`",
        "",
        "## B2. Programmatic Domain-Rule Fidelity",
        "",
        f"- coverage present: `{domain_rule.get('coverage_present')}`",
        f"- activated count: `{domain_rule.get('activated_count')}`",
        f"- selected count: `{domain_rule.get('selected_count')}`",
        f"- selected rate: `{domain_rule.get('selected_rate')}`",
        f"- selected known correct rate: `{domain_rule.get('selected_known_correct_rate')}`",
        f"- short-circuit count: `{domain_rule.get('short_circuit_count')}`",
        f"- status counts: `{domain_rule.get('status_counts')}`",
        f"- rule counts: `{domain_rule.get('rule_counts')}`",
        f"- selection counts: `{domain_rule.get('selection_method_counts')}`",
        "",
        "## C. Residual Family",
        "",
        f"- family map provided: `{residual.get('family_map_provided')}`",
        f"- baseline family map provided: `{residual.get('baseline_family_map_provided')}`",
        f"- by family: `{residual.get('by_family')}`",
        f"- baseline by family: `{residual.get('baseline_by_family')}`",
        f"- agent error-rate delta by family: `{residual.get('agent_error_rate_delta_by_family')}`",
        f"- before/after measured: `{residual.get('residual_family_before_after_measured')}`",
        f"- residual protocol marked: `{residual.get('residual_family_protocol_marked')}`",
        f"- selector before/after: `{residual.get('selector_before_after_comparison')}`",
        f"- status: `{residual.get('residual_family_learning_status')}`",
        "",
        "## Boundary",
        "",
        payload.get("claim_boundary", ""),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a three-part HLE evaluation report.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--baseline-eval-id", default="")
    parser.add_argument("--preflight", default="")
    parser.add_argument("--baseline-preflight", default="")
    parser.add_argument("--residual-family-protocol", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--md-out", default="")
    parser.add_argument(
        "--log-out",
        default="",
        help="Metadata-only JSONL diagnostic log. Defaults to the output JSON path with .jsonl suffix.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_dir = Path(args.run_dir)
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    preflight = Path(args.preflight) if args.preflight else None
    if preflight is not None and not preflight.is_absolute():
        preflight = root / preflight
    baseline_preflight = Path(args.baseline_preflight) if args.baseline_preflight else None
    if baseline_preflight is not None and not baseline_preflight.is_absolute():
        baseline_preflight = root / baseline_preflight
    out = Path(args.out) if args.out else run_dir / f"{args.eval_id}_three_part_eval.json"
    md_out = Path(args.md_out) if args.md_out else root / DEFAULT_MD_DIR / f"{args.eval_id}_three_part_eval.md"
    out = out if out.is_absolute() else root / out
    md_out = md_out if md_out.is_absolute() else root / md_out
    log_out = Path(args.log_out) if args.log_out else out.with_suffix(".jsonl")
    log_out = log_out if log_out.is_absolute() else root / log_out

    payload = build_three_part_eval_payload(
        run_dir=run_dir,
        eval_id=args.eval_id,
        baseline_eval_id=args.baseline_eval_id,
        preflight_path=preflight,
        baseline_preflight_path=baseline_preflight,
        residual_family_protocol=args.residual_family_protocol,
        log_out=log_out,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": args.eval_id,
        "out": str(out),
        "md_out": str(md_out),
        "log_out": str(log_out),
        "pass_summary": payload.get("pass_summary"),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
