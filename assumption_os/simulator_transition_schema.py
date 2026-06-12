"""Frozen simulator transition schema and dataset validator.

The Phase13 artifact reports 345 first-party transition-like rows, but it only
stores the count.  This module materializes those rows into a fixed redacted
JSONL dataset and validates the schema before any stronger simulator is trained.
Invalid rows are written to quarantine instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


SCHEMA_VERSION = "simulator_transition_schema_v0"
DEFAULT_SCHEMA_OUT = PAPER_DIR / "simulator_transition_schema_v0.json"
DEFAULT_DATASET_OUT = PAPER_DIR / "simulator_transition_dataset_v0.jsonl"
DEFAULT_QUARANTINE_OUT = PAPER_DIR / "simulator_transition_quarantine_v0.jsonl"
DEFAULT_VALIDATION_OUT = PAPER_DIR / "simulator_transition_schema_validation_20260612.json"

SOURCE_ARTIFACTS = {
    "phase10_reliability": PAPER_DIR / "full_v3_phase10_reliability_calibration_20260611.json",
    "residual_fresh_live": PAPER_DIR / "full_v3_residual_fresh_live_loop_20260611.json",
    "live_multigeneration": PAPER_DIR / "full_v3_live_multigeneration_expansion_20260612.json",
    "blinded_recursive_live": PAPER_DIR / "full_v3_blinded_recursive_live_line_20260612.json",
    "phase13_claim_lift": PAPER_DIR / "full_v3_phase13_general_autonomy_lift_20260612.json",
}

ACTION_TYPES = {
    "select_profile",
    "run_ablation",
    "repair_scope",
    "collect_evidence",
    "apply_candidate",
}
SPLITS = {"train", "validation", "test"}
FORBIDDEN_KEY_FRAGMENTS = {
    "prompt",
    "answer",
    "gold",
    "secret",
    "api_key",
    "apikey",
    "token",
    "password",
    "raw_text",
}
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9]{12,}|api[_-]?key|bearer\s+[A-Za-z0-9._-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
    row_id: str
    issue: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "issue": self.issue, "path": self.path}


@dataclass(frozen=True)
class ValidationReport:
    raw_row_count: int
    valid_row_count: int
    invalid_row_count: int
    quarantine_row_count: int
    issue_counts: dict[str, int]
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    provenance_hash_unique: bool
    redacted_row_count: int
    invalid_rows: list[dict[str, Any]] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_row_count": self.raw_row_count,
            "valid_row_count": self.valid_row_count,
            "invalid_row_count": self.invalid_row_count,
            "quarantine_row_count": self.quarantine_row_count,
            "issue_counts": self.issue_counts,
            "split_counts": self.split_counts,
            "source_counts": self.source_counts,
            "provenance_hash_unique": self.provenance_hash_unique,
            "redacted_row_count": self.redacted_row_count,
            "issues": [issue.to_dict() for issue in self.issues[:50]],
            "invalid_rows": self.invalid_rows[:10],
        }


def build_simulator_transition_schema_payload(
    *,
    root: Path,
    eval_id: str = "simulator_transition_schema_validation_20260612",
    schema_out: Path | None = None,
    dataset_out: Path | None = None,
    quarantine_out: Path | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    schema_out = schema_out or DEFAULT_SCHEMA_OUT
    dataset_out = dataset_out or DEFAULT_DATASET_OUT
    quarantine_out = quarantine_out or DEFAULT_QUARANTINE_OUT
    schema_path = _resolve(root, schema_out)
    dataset_path = _resolve(root, dataset_out)
    quarantine_path = _resolve(root, quarantine_out)

    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    schema = simulator_transition_schema()
    rows = build_current_transition_rows(artifacts)
    report = validate_transition_rows(rows)
    valid_rows = [row for row in rows if not validate_transition_row(row)]

    if write_artifacts:
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_jsonl(dataset_path, valid_rows)
        _write_jsonl(quarantine_path, report.invalid_rows)

    expected_rows = int(
        artifacts["phase13_claim_lift"].get("metrics", {}).get("simulator_first_party_transition_like_row_count")
        or 345
    )
    metrics = {
        "expected_transition_row_count": expected_rows,
        "raw_row_count": report.raw_row_count,
        "valid_row_count": report.valid_row_count,
        "invalid_row_count": report.invalid_row_count,
        "quarantine_row_count": report.quarantine_row_count,
        "redacted_row_count": report.redacted_row_count,
        "split_counts": report.split_counts,
        "source_counts": report.source_counts,
        "provenance_hash_unique": report.provenance_hash_unique,
        "schema_field_count": len(schema["required"]),
        "dataset_path": _display_path(root, dataset_path),
        "schema_path": _display_path(root, schema_path),
        "quarantine_path": _display_path(root, quarantine_path),
        "secret_or_prompt_payload_detected": bool(report.issue_counts.get("redaction_forbidden_payload")),
    }
    gates = {
        "source_artifacts_loaded": all(bool(artifact) for artifact in artifacts.values()),
        "row_count_matches_phase13": metrics["raw_row_count"] == expected_rows,
        "all_rows_valid": metrics["valid_row_count"] == metrics["raw_row_count"],
        "invalid_rows_quarantined": metrics["invalid_row_count"] == metrics["quarantine_row_count"],
        "no_invalid_rows_in_current_dataset": metrics["invalid_row_count"] == 0,
        "all_rows_redacted": metrics["redacted_row_count"] == metrics["raw_row_count"],
        "split_labels_present": set(metrics["split_counts"]) == SPLITS,
        "provenance_hashes_unique": metrics["provenance_hash_unique"] is True,
        "no_secret_or_prompt_payload": metrics["secret_or_prompt_payload_detected"] is False,
        "dataset_written": (not write_artifacts) or dataset_path.exists(),
        "schema_written": (not write_artifacts) or schema_path.exists(),
        "quarantine_written": (not write_artifacts) or quarantine_path.exists(),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_transition_schema_validation",
        "last_three_part_ticket": "B1_simulator_transition_schema",
        "schema_version": SCHEMA_VERSION,
        "performance_validation": True,
        "validation_scope": (
            "Materializes the current 345 first-party transition-like rows into a fixed redacted graph-action "
            "simulator transition schema.  Validates required fields, split labels, provenance hashes, redaction, "
            "and quarantine handling before stronger simulator training."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "schema": schema,
        "metrics": metrics,
        "validation_report": report.to_dict(),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "limitations": [
            "Some fresh live rows are aggregate-distilled from redacted call counts where source artifacts did not retain per-call judgment rows.",
            "This schema freezes transition data and validation discipline; it does not promote a raw simulator.",
        ],
    }


def simulator_transition_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": SCHEMA_VERSION,
        "type": "object",
        "required": ["row_id", "schema_version", "state", "action", "prediction", "outcome", "provenance"],
        "properties": {
            "row_id": {"type": "string"},
            "schema_version": {"const": SCHEMA_VERSION},
            "state": {
                "type": "object",
                "required": [
                    "domain",
                    "pattern",
                    "active_assumptions",
                    "residual_cluster",
                    "formal_gate_state",
                    "preflight_state",
                    "world_model_features",
                ],
            },
            "action": {
                "type": "object",
                "required": ["type", "arm"],
                "properties": {"type": {"enum": sorted(ACTION_TYPES)}, "arm": {"type": "string"}},
            },
            "prediction": {
                "type": "object",
                "required": ["p_accept", "p_regress", "expected_utility", "uncertainty"],
            },
            "outcome": {
                "type": "object",
                "required": ["accepted", "utility_vs_baseline", "control_harm", "regression", "cost"],
            },
            "provenance": {
                "type": "object",
                "required": ["artifact_id", "source_row_id", "split", "redacted", "provenance_hash"],
            },
        },
    }


def build_current_transition_rows(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_phase10_reliability_rows(artifacts["phase10_reliability"]))
    rows.extend(_residual_fresh_live_rows(artifacts["residual_fresh_live"]))
    rows.extend(_live_multigeneration_rows(artifacts["live_multigeneration"]))
    rows.extend(_blinded_recursive_rows(artifacts["blinded_recursive_live"]))
    return rows


def validate_transition_rows(rows: list[dict[str, Any]]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    invalid_rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    redacted_count = 0

    for row in rows:
        row_issues = validate_transition_row(row)
        row_id = str(row.get("row_id") or "")
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if provenance.get("redacted") is True:
            redacted_count += 1
        if provenance.get("split"):
            split_counts[str(provenance["split"])] += 1
        if provenance.get("artifact_id"):
            source_counts[str(provenance["artifact_id"])] += 1
        if provenance.get("provenance_hash"):
            hashes.append(str(provenance["provenance_hash"]))
        if row_issues:
            issues.extend(row_issues)
            invalid_rows.append({"row": row, "issues": [issue.to_dict() for issue in row_issues]})

    issue_counts = Counter(issue.issue for issue in issues)
    return ValidationReport(
        raw_row_count=len(rows),
        valid_row_count=len(rows) - len(invalid_rows),
        invalid_row_count=len(invalid_rows),
        quarantine_row_count=len(invalid_rows),
        issue_counts=dict(issue_counts),
        split_counts=dict(sorted(split_counts.items())),
        source_counts=dict(sorted(source_counts.items())),
        provenance_hash_unique=len(hashes) == len(set(hashes)) == len(rows),
        redacted_row_count=redacted_count,
        invalid_rows=invalid_rows,
        issues=issues,
    )


def validate_transition_row(row: dict[str, Any]) -> list[ValidationIssue]:
    row_id = str(row.get("row_id") or "<missing>")
    issues: list[ValidationIssue] = []
    for key in ["row_id", "schema_version", "state", "action", "prediction", "outcome", "provenance"]:
        if key not in row:
            issues.append(ValidationIssue(row_id, "missing_required_top_field", key))
    if row.get("schema_version") != SCHEMA_VERSION:
        issues.append(ValidationIssue(row_id, "schema_version_mismatch", "schema_version"))
    state = row.get("state")
    action = row.get("action")
    prediction = row.get("prediction")
    outcome = row.get("outcome")
    provenance = row.get("provenance")
    if not isinstance(state, dict):
        issues.append(ValidationIssue(row_id, "state_not_object", "state"))
    else:
        for key in [
            "domain",
            "pattern",
            "active_assumptions",
            "residual_cluster",
            "formal_gate_state",
            "preflight_state",
            "world_model_features",
        ]:
            if key not in state:
                issues.append(ValidationIssue(row_id, "missing_state_field", f"state.{key}"))
        if not isinstance(state.get("active_assumptions"), list):
            issues.append(ValidationIssue(row_id, "active_assumptions_not_list", "state.active_assumptions"))
        if not isinstance(state.get("world_model_features"), list):
            issues.append(ValidationIssue(row_id, "world_model_features_not_list", "state.world_model_features"))
    if not isinstance(action, dict):
        issues.append(ValidationIssue(row_id, "action_not_object", "action"))
    else:
        if action.get("type") not in ACTION_TYPES:
            issues.append(ValidationIssue(row_id, "invalid_action_type", "action.type"))
        if not action.get("arm"):
            issues.append(ValidationIssue(row_id, "missing_action_arm", "action.arm"))
    if not isinstance(prediction, dict):
        issues.append(ValidationIssue(row_id, "prediction_not_object", "prediction"))
    else:
        for key in ["p_accept", "p_regress", "expected_utility", "uncertainty"]:
            if key not in prediction:
                issues.append(ValidationIssue(row_id, "missing_prediction_field", f"prediction.{key}"))
            elif not _is_unit_float(prediction[key]):
                issues.append(ValidationIssue(row_id, "prediction_not_unit_float", f"prediction.{key}"))
    if not isinstance(outcome, dict):
        issues.append(ValidationIssue(row_id, "outcome_not_object", "outcome"))
    else:
        for key in ["accepted", "control_harm", "regression"]:
            if not isinstance(outcome.get(key), bool):
                issues.append(ValidationIssue(row_id, "outcome_bool_field_invalid", f"outcome.{key}"))
        if not _is_unit_float(outcome.get("utility_vs_baseline")):
            issues.append(ValidationIssue(row_id, "outcome_utility_not_unit_float", "outcome.utility_vs_baseline"))
        if not isinstance(outcome.get("cost"), (int, float)) or float(outcome.get("cost")) < 0:
            issues.append(ValidationIssue(row_id, "outcome_cost_invalid", "outcome.cost"))
    if not isinstance(provenance, dict):
        issues.append(ValidationIssue(row_id, "provenance_not_object", "provenance"))
    else:
        if provenance.get("split") not in SPLITS:
            issues.append(ValidationIssue(row_id, "invalid_split_label", "provenance.split"))
        if provenance.get("redacted") is not True:
            issues.append(ValidationIssue(row_id, "row_not_marked_redacted", "provenance.redacted"))
        if not provenance.get("artifact_id"):
            issues.append(ValidationIssue(row_id, "missing_artifact_id", "provenance.artifact_id"))
        expected_hash = _provenance_hash(row)
        if provenance.get("provenance_hash") != expected_hash:
            issues.append(ValidationIssue(row_id, "provenance_hash_mismatch", "provenance.provenance_hash"))
    if _has_forbidden_payload(row):
        issues.append(ValidationIssue(row_id, "redaction_forbidden_payload", "$"))
    return issues


def make_transition_row(
    *,
    row_id: str,
    state: dict[str, Any],
    action: dict[str, Any],
    prediction: dict[str, Any],
    outcome: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "row_id": row_id,
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "action": action,
        "prediction": {key: _unit(value) for key, value in prediction.items()},
        "outcome": {
            "accepted": bool(outcome["accepted"]),
            "utility_vs_baseline": _unit(outcome["utility_vs_baseline"]),
            "control_harm": bool(outcome["control_harm"]),
            "regression": bool(outcome["regression"]),
            "cost": round(float(outcome.get("cost", 1.0)), 4),
        },
        "provenance": {
            **provenance,
            "split": provenance.get("split") or _split_for(row_id),
            "redacted": True,
        },
    }
    row["provenance"]["provenance_hash"] = _provenance_hash(row)
    return row


def _phase10_reliability_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in payload.get("records", []):
        bits = [str(bit) for bit in record.get("state_bits", [])]
        domain = _bit_suffix(bits, "domain:") or "unknown"
        pattern = _bit_suffix(bits, "pattern:") or "unknown"
        route = _bit_suffix(bits, "route:") or "unknown"
        row_id = stable_hash(["phase10_reliability", record.get("problem_id"), record.get("arm")])
        observed = float(record.get("observed_reward") or 0.0)
        calibrated = float(record.get("calibrated_prediction") or record.get("raw_prediction") or 0.5)
        rows.append(
            make_transition_row(
                row_id=f"simtr_{row_id}",
                state={
                    "domain": domain,
                    "pattern": pattern,
                    "active_assumptions": [f"route:{route}", f"arm:{record.get('arm')}"],
                    "residual_cluster": "phase10_reliability",
                    "formal_gate_state": "not_applicable",
                    "preflight_state": "observed_arm_record",
                    "world_model_features": bits,
                },
                action={"type": "select_profile", "arm": str(record.get("arm") or "unknown")},
                prediction={
                    "p_accept": calibrated,
                    "p_regress": max(0.0, 1.0 - calibrated),
                    "expected_utility": calibrated,
                    "uncertainty": _unit(abs(float(record.get("raw_prediction") or calibrated) - calibrated)),
                },
                outcome={
                    "accepted": observed >= 0.5,
                    "utility_vs_baseline": observed,
                    "control_harm": False,
                    "regression": observed < 0.5,
                    "cost": 1.0,
                },
                provenance={
                    "artifact_id": "full_v3_phase10_reliability_calibration_20260611",
                    "source_row_id": f"{record.get('problem_id')}::{record.get('arm')}",
                    "source_granularity": "observed_arm_record",
                    "split": _split_for(f"phase10::{record.get('problem_id')}::{record.get('arm')}"),
                },
            )
        )
    return rows


def _residual_fresh_live_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    judgments = payload.get("live_judgment_payload", {}).get("judgments", {})
    summaries = payload.get("candidate_acceptance", {}).get("summaries", [])
    candidates = payload.get("selected_generation_one_candidates", [])
    candidate_by_proposal = {
        str(summary.get("proposal_id")): candidates[index] if index < len(candidates) else {}
        for index, summary in enumerate(summaries)
    }
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        proposal_id = str(summary.get("proposal_id"))
        candidate = candidate_by_proposal.get(proposal_id, {})
        for row_kind, ids in [
            ("trigger", summary.get("judged_trigger_problem_ids", [])),
            ("control", summary.get("judged_control_problem_ids", [])),
        ]:
            for problem_id in ids:
                judgment = judgments.get(problem_id, {})
                rows.append(
                    _fresh_judgment_row(
                        artifact_id="full_v3_residual_fresh_live_loop_20260611",
                        source_row_id=str(problem_id),
                        source_granularity="observed_fresh_judgment",
                        domain=str(candidate.get("source_domain") or "fresh_live_policy"),
                        pattern=str(candidate.get("source_pattern") or "fresh_live"),
                        residual_cluster=str(summary.get("parent_node_id") or candidate.get("source_cluster_id") or "residual"),
                        active_assumptions=[proposal_id, str(candidate.get("trajectory") or "residual_repair")],
                        row_kind=row_kind,
                        decision=str(summary.get("decision") or "unknown"),
                        winner=str(judgment.get("winner") or "tie"),
                        expected_utility=float(candidate.get("world_model_expected_utility") or summary.get("trigger_utility") or 0.5),
                        predicted_regression=float(candidate.get("predicted_regression_risk") or summary.get("control_loss_rate") or 0.0),
                    )
                )
    return rows


def _live_multigeneration_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generation in payload.get("generation_results", []):
        gen = int(generation.get("generation") or 0)
        decisions = _expand_decisions(generation.get("candidate_acceptance", {}).get("decision_counts", {}))
        selected = [str(candidate_id) for candidate_id in generation.get("selected_candidate_ids", [])]
        for index, candidate_id in enumerate(selected):
            decision = decisions[index] if index < len(decisions) else "reject_benefit"
            for row_kind, ordinal, winner in _distilled_outcomes_for_decision(decision):
                source_row_id = f"gen{gen}:{candidate_id}:{row_kind}:{ordinal}"
                rows.append(
                    _fresh_judgment_row(
                        artifact_id="full_v3_live_multigeneration_expansion_20260612",
                        source_row_id=source_row_id,
                        source_granularity="aggregate_distilled_fresh_judgment",
                        domain="live_multigeneration",
                        pattern=f"generation_{gen}",
                        residual_cluster=f"live_multigeneration_gen{gen}",
                        active_assumptions=[candidate_id, f"decision:{decision}"],
                        row_kind=row_kind,
                        decision=decision,
                        winner=winner,
                        expected_utility=_expected_utility_for_decision(decision, row_kind),
                        predicted_regression=_regression_for_decision(decision, row_kind),
                    )
                )
    return rows


def _blinded_recursive_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in payload.get("seed_results", []):
        for generation in seed.get("generation_results", []):
            summaries = {
                str(summary.get("candidate_id")): summary
                for summary in generation.get("candidate_acceptance", {}).get("summaries", [])
            }
            for judgment in generation.get("live_judgment", {}).get("judgment_rows", []):
                candidate_id = str(judgment.get("candidate_id") or "")
                summary = summaries.get(candidate_id, {})
                row_kind = str(judgment.get("row_kind") or "trigger")
                decision = str(summary.get("decision") or "unknown")
                winner = str(judgment.get("normalized_outcome") or judgment.get("winner") or "tie")
                candidate_family = str(judgment.get("candidate_family") or summary.get("candidate_family") or "unknown")
                pattern = _candidate_family_pattern(candidate_family, row_kind)
                source_row_id = str(judgment.get("synthetic_problem_id") or judgment.get("actual_problem_id"))
                rows.append(
                    _fresh_judgment_row(
                        artifact_id="full_v3_blinded_recursive_live_line_20260612",
                        source_row_id=source_row_id,
                        source_granularity="observed_blinded_fresh_judgment",
                        domain=str(judgment.get("domain") or "unknown"),
                        pattern=pattern,
                        residual_cluster=f"seed_{judgment.get('seed')}_gen_{judgment.get('generation')}",
                        active_assumptions=[
                            candidate_family,
                            str(judgment.get("selection_tier") or "unknown"),
                            candidate_id,
                        ],
                        row_kind=row_kind,
                        decision=decision,
                        winner=winner,
                        expected_utility=float(
                            summary.get("trigger_utility")
                            if row_kind == "trigger"
                            else 1.0 - float(summary.get("control_loss_rate") or 0.0)
                        )
                        if summary
                        else _winner_utility(winner),
                        predicted_regression=float(summary.get("control_loss_rate") or 0.0),
                    )
                )
    return rows


def _fresh_judgment_row(
    *,
    artifact_id: str,
    source_row_id: str,
    source_granularity: str,
    domain: str,
    pattern: str,
    residual_cluster: str,
    active_assumptions: list[str],
    row_kind: str,
    decision: str,
    winner: str,
    expected_utility: float,
    predicted_regression: float,
) -> dict[str, Any]:
    utility = _winner_utility(winner)
    control_harm = row_kind == "control" and utility < 0.5
    row_id = f"simtr_{stable_hash([artifact_id, source_row_id, row_kind, decision])}"
    return make_transition_row(
        row_id=row_id,
        state={
            "domain": domain,
            "pattern": pattern,
            "active_assumptions": active_assumptions,
            "residual_cluster": residual_cluster,
            "formal_gate_state": "not_applicable",
            "preflight_state": "ready_for_fresh_ablation",
            "world_model_features": [
                f"domain:{domain}",
                f"pattern:{pattern}",
                f"row_kind:{row_kind}",
                f"decision:{decision}",
            ],
        },
        action={"type": "run_ablation", "arm": "candidate_vs_baseline"},
        prediction={
            "p_accept": _decision_accept_probability(decision, expected_utility),
            "p_regress": predicted_regression,
            "expected_utility": expected_utility,
            "uncertainty": 0.2 if source_granularity.startswith("observed") else 0.35,
        },
        outcome={
            "accepted": decision == "accept",
            "utility_vs_baseline": utility,
            "control_harm": control_harm,
            "regression": control_harm or (row_kind == "trigger" and utility < 0.5),
            "cost": 1.0,
        },
        provenance={
            "artifact_id": artifact_id,
            "source_row_id": source_row_id,
            "source_granularity": source_granularity,
            "row_kind": row_kind,
            "split": _split_for(f"{artifact_id}:{source_row_id}"),
        },
    )


def _expand_decisions(counts: dict[str, int]) -> list[str]:
    order = ["reject_harm", "reject_benefit", "accept"]
    out: list[str] = []
    for decision in order:
        out.extend([decision] * int(counts.get(decision, 0)))
    return out


def _distilled_outcomes_for_decision(decision: str) -> list[tuple[str, int, str]]:
    if decision == "accept":
        return [("trigger", i, "candidate") for i in range(1, 5)] + [("control", 1, "tie"), ("control", 2, "candidate")]
    if decision == "reject_harm":
        return [("trigger", i, "candidate") for i in range(1, 4)] + [
            ("trigger", 4, "tie"),
            ("control", 1, "baseline"),
            ("control", 2, "tie"),
        ]
    return [("trigger", 1, "candidate"), ("trigger", 2, "tie"), ("trigger", 3, "baseline"), ("trigger", 4, "tie"), ("control", 1, "tie"), ("control", 2, "candidate")]


def _expected_utility_for_decision(decision: str, row_kind: str) -> float:
    if decision == "accept":
        return 0.82 if row_kind == "trigger" else 0.72
    if decision == "reject_harm":
        return 0.70 if row_kind == "trigger" else 0.30
    return 0.42 if row_kind == "trigger" else 0.60


def _regression_for_decision(decision: str, row_kind: str) -> float:
    if decision == "reject_harm" and row_kind == "control":
        return 0.70
    if decision == "accept":
        return 0.08
    return 0.25


def _decision_accept_probability(decision: str, expected_utility: float) -> float:
    if decision == "accept":
        return max(0.6, expected_utility)
    if decision == "reject_harm":
        return min(0.45, expected_utility)
    if decision == "reject_benefit":
        return min(0.45, expected_utility)
    return 0.5


def _winner_utility(winner: str) -> float:
    normalized = str(winner or "tie").lower()
    if normalized in {"candidate", "win"}:
        return 1.0
    if normalized in {"baseline", "loss"}:
        return 0.0
    return 0.5


def _candidate_family_pattern(candidate_family: str, row_kind: str) -> str:
    parts = [part for part in candidate_family.split("::") if part]
    if len(parts) >= 2:
        return parts[1]
    return row_kind


def _bit_suffix(bits: list[str], prefix: str) -> str | None:
    for bit in bits:
        if bit.startswith(prefix):
            return bit[len(prefix):]
    return None


def _split_for(value: str) -> str:
    bucket = int(stable_hash(value), 16) % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def _provenance_hash(row: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(row, sort_keys=True))
    provenance = clone.get("provenance", {})
    if isinstance(provenance, dict):
        provenance.pop("provenance_hash", None)
    return stable_hash(clone)


def _has_forbidden_payload(value: Any, *, key: str = "") -> bool:
    lowered_key = key.lower()
    if any(fragment in lowered_key for fragment in FORBIDDEN_KEY_FRAGMENTS):
        return True
    if isinstance(value, dict):
        return any(_has_forbidden_payload(v, key=str(k)) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_forbidden_payload(v, key=key) for v in value)
    if isinstance(value, str):
        if SECRET_RE.search(value):
            return True
        if len(value) > 1000:
            return True
    return False


def _is_unit_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def _unit(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator transition schema validation artifact.")
    parser.add_argument("--eval-id", default="simulator_transition_schema_validation_20260612")
    parser.add_argument("--root", default=".")
    parser.add_argument("--schema-out", default=str(DEFAULT_SCHEMA_OUT))
    parser.add_argument("--dataset-out", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--quarantine-out", default=str(DEFAULT_QUARANTINE_OUT))
    parser.add_argument("--out", default=str(DEFAULT_VALIDATION_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_transition_schema_payload(
        root=root,
        eval_id=args.eval_id,
        schema_out=Path(args.schema_out),
        dataset_out=Path(args.dataset_out),
        quarantine_out=Path(args.quarantine_out),
        write_artifacts=True,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
