"""Uncertainty and abstention layer for graph-action simulator triage.

B2 shows that a feature-similarity simulator can be useful, but it should not
be promoted as an oracle.  B3 wraps the split predictor with explicit
confidence intervals, calibration bins, abstention reasons, and verifier tiers.
The only allowed outputs are routing recommendations; auto-accept, auto-apply,
and judge replacement are intentionally impossible actions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .simulator_eval_splits import _ece, _evaluate_predictions, _label, _load_jsonl, _predict
from .simulator_transition_schema import DEFAULT_DATASET_OUT, validate_transition_rows


DEFAULT_OUT = PAPER_DIR / "simulator_uncertainty_20260612.json"
ALLOWED_ACTIONS = {
    "recommend_run_ablation",
    "recommend_collect_more_evidence",
    "recommend_repair_scope",
    "recommend_reject_low_value",
    "abstain_to_live_validation",
}
FORBIDDEN_ACTIONS = {
    "auto_accept_without_live",
    "auto_apply_policy_change",
    "replace_judge",
}
ECE_THRESHOLD = 0.12
MIN_SUPPORT = 20
MIN_CALIBRATION_BIN_SUPPORT = 8
MAX_INTERVAL_WIDTH = 0.36
MAX_CALIBRATION_ERROR = 0.24
DECISION_MARGIN = 0.03


def build_simulator_uncertainty_payload(
    *,
    root: Path,
    eval_id: str = "simulator_uncertainty_20260612",
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    dataset_path = dataset_path or DEFAULT_DATASET_OUT
    dataset_path = dataset_path if dataset_path.is_absolute() else root / dataset_path
    rows = _load_jsonl(dataset_path)
    validation = validate_transition_rows(rows)
    leave_pattern = _evaluate_leave_pattern_uncertainty(rows)
    stress_probe = _low_support_stress_probe(rows)
    metrics = {
        "row_count": len(rows),
        "valid_row_count": validation.valid_row_count,
        "decision_count": len(leave_pattern["decisions"]),
        "leave_pattern_base_rate_brier": leave_pattern["base_rate_metrics"]["brier"],
        "leave_pattern_base_rate_brier_with_abstain_as_half": leave_pattern["base_rate_metrics"][
            "brier_with_abstain_as_half"
        ],
        "leave_pattern_uncertainty_brier": leave_pattern["uncertainty_metrics"]["brier"],
        "leave_pattern_uncertainty_brier_with_abstain_as_half": leave_pattern["uncertainty_metrics"][
            "brier_with_abstain_as_half"
        ],
        "leave_pattern_uncertainty_ece": leave_pattern["uncertainty_metrics"]["ece"],
        "leave_pattern_abstention_rate": leave_pattern["uncertainty_metrics"]["abstention_rate"],
        "accepted_candidate_block_rate": leave_pattern["accepted_candidate_block_rate"],
        "harmful_candidate_block_rate": leave_pattern["harmful_candidate_block_rate"],
        "forbidden_action_recommended_count": leave_pattern["forbidden_action_recommended_count"],
        "allowed_action_coverage": leave_pattern["allowed_action_coverage"],
        "low_support_probe_abstained": stress_probe["action"] == "abstain_to_live_validation",
        "low_support_probe_reason": stress_probe["abstain_reason"],
        "production_simulator_replacement_allowed": False,
    }
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["row_count"] == 345,
        "required_prediction_fields_present": leave_pattern["required_prediction_fields_present"],
        "all_recommendations_are_allowed_actions": metrics["forbidden_action_recommended_count"] == 0
        and metrics["allowed_action_coverage"] == 1.0,
        "abstain_on_low_support": metrics["low_support_probe_abstained"]
        and metrics["low_support_probe_reason"] == "low_support",
        "accepted_candidate_block_rate_safe": metrics["accepted_candidate_block_rate"] <= 0.02,
        "calibration_ece_safe": metrics["leave_pattern_uncertainty_ece"] <= ECE_THRESHOLD,
        "leave_pattern_brier_beats_base_rate": (
            metrics["leave_pattern_uncertainty_brier_with_abstain_as_half"]
            < metrics["leave_pattern_base_rate_brier_with_abstain_as_half"]
        ),
        "production_replacement_blocked": metrics["production_simulator_replacement_allowed"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_uncertainty_abstention",
        "last_three_part_ticket": "B3_simulator_uncertainty_abstain",
        "performance_validation": True,
        "validation_scope": (
            "Adds confidence intervals, calibration bins, abstention reasons, and verifier-tier routing to the "
            "B2 feature simulator.  The simulator can route candidates to ablation, evidence collection, repair, "
            "or live validation, but cannot auto-accept, auto-apply, or replace judge evidence."
        ),
        "source": {
            "dataset_path": _display_path(root, dataset_path),
            "schema_validation_valid_row_count": validation.valid_row_count,
        },
        "policy": {
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "forbidden_actions": sorted(FORBIDDEN_ACTIONS),
            "ece_threshold": ECE_THRESHOLD,
            "min_support": MIN_SUPPORT,
            "min_calibration_bin_support": MIN_CALIBRATION_BIN_SUPPORT,
            "max_interval_width": MAX_INTERVAL_WIDTH,
            "max_calibration_error": MAX_CALIBRATION_ERROR,
            "decision_margin": DECISION_MARGIN,
        },
        "leave_pattern_evaluation": leave_pattern,
        "low_support_stress_probe": stress_probe,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundaries": [
            "This layer is a budget/verifier router, not a task-world simulator.",
            "No recommendation can auto-accept a proposal, auto-apply a graph policy change, or replace live judge evidence.",
            "The B3 gate is evaluated on leave-pattern split discipline and a synthetic low-support stress probe.",
        ],
    }


def _evaluate_leave_pattern_uncertainty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["state"]["pattern"])].append(row)

    decisions: list[dict[str, Any]] = []
    base_predictions: list[dict[str, Any]] = []
    for pattern, eval_rows in sorted(groups.items()):
        train_rows = [row for row in rows if str(row["state"]["pattern"]) != pattern]
        calibration_bins = _calibration_bins(train_rows)
        for row in eval_rows:
            base_pred = _predict("feature_similarity_simulator", row, train_rows)
            base_rate = _predict("base_rate_per_arm", row, train_rows)
            base_predictions.append(base_rate)
            decisions.append(_decision_from_prediction(row, base_pred, calibration_bins))

    uncertainty_predictions = [_prediction_view(decision) for decision in decisions]
    uncertainty_metrics = _evaluate_predictions(rows, uncertainty_predictions)
    base_rate_metrics = _evaluate_predictions(rows, base_predictions)
    action_counts = Counter(decision["action"] for decision in decisions)
    abstain_reason_counts = Counter(decision["abstain_reason"] for decision in decisions)
    blocked_ids = {decision["row_id"] for decision in decisions if decision["action"] == "recommend_reject_low_value"}
    accepted_ids = {row["row_id"] for row in rows if _label(row) == 1}
    harmful_ids = {row["row_id"] for row in rows if _label(row) == 0}
    required_fields = {
        "prediction",
        "confidence_interval",
        "calibration_bin",
        "abstain_reason",
        "required_verifier_tier",
        "action",
    }
    return {
        "split_name": "leave_pattern_out",
        "group_count": len(groups),
        "row_count": len(rows),
        "decisions": decisions,
        "uncertainty_metrics": uncertainty_metrics,
        "base_rate_metrics": base_rate_metrics,
        "action_counts": dict(sorted(action_counts.items())),
        "abstain_reason_counts": dict(sorted(abstain_reason_counts.items())),
        "accepted_candidate_block_rate": round(len(blocked_ids & accepted_ids) / max(1, len(accepted_ids)), 4),
        "harmful_candidate_block_rate": round(len(blocked_ids & harmful_ids) / max(1, len(harmful_ids)), 4),
        "forbidden_action_recommended_count": sum(1 for decision in decisions if decision["action"] in FORBIDDEN_ACTIONS),
        "allowed_action_coverage": round(
            sum(1 for decision in decisions if decision["action"] in ALLOWED_ACTIONS) / max(1, len(decisions)),
            4,
        ),
        "required_prediction_fields_present": all(required_fields.issubset(decision) for decision in decisions),
        "calibration_ece_recomputed": _ece(uncertainty_predictions),
    }


def _decision_from_prediction(
    row: dict[str, Any],
    prediction: dict[str, Any],
    calibration_bins: dict[str, dict[str, float]],
) -> dict[str, Any]:
    score = float(prediction["score"])
    support_count = int(prediction["support_count"])
    interval = _confidence_interval(score, support_count)
    interval_width = interval["upper"] - interval["lower"]
    calibration_bin = _bin_id(score)
    bin_stats = calibration_bins.get(calibration_bin, {"count": 0, "observed_rate": 0.5})
    calibration_error = abs(score - float(bin_stats["observed_rate"]))
    uncertainty = max(
        interval_width,
        calibration_error,
        1.0 / math.sqrt(max(1, support_count)),
    )
    abstain_reason = _abstain_reason(
        score=score,
        support_count=support_count,
        calibration_bin_count=int(bin_stats["count"]),
        interval_width=interval_width,
        calibration_error=calibration_error,
    )
    action = _route_action(score=score, uncertainty=uncertainty, abstain_reason=abstain_reason, interval=interval)
    return {
        "row_id": row["row_id"],
        "pattern": row["state"]["pattern"],
        "domain": row["state"]["domain"],
        "action": action,
        "prediction": {
            "p_accept": round(score, 4),
            "p_regress": round(float(row["prediction"].get("p_regress", 0.0)), 4),
            "expected_utility": round(float(row["prediction"].get("expected_utility", score)), 4),
            "uncertainty": round(uncertainty, 4),
        },
        "score": round(score, 4),
        "label": _label(row),
        "confidence_interval": {
            "method": "normal_approximation_with_support_floor",
            "support_count": support_count,
            "lower": round(interval["lower"], 4),
            "upper": round(interval["upper"], 4),
            "width": round(interval_width, 4),
        },
        "calibration_bin": {
            "bin": calibration_bin,
            "support_count": int(bin_stats["count"]),
            "observed_rate": round(float(bin_stats["observed_rate"]), 4),
            "absolute_error": round(calibration_error, 4),
        },
        "abstain": action == "abstain_to_live_validation",
        "abstain_reason": abstain_reason,
        "required_verifier_tier": _verifier_tier(action),
        "forbidden_action_blocked": action not in FORBIDDEN_ACTIONS,
    }


def _prediction_view(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "predictor": "simulator_uncertainty_abstention",
        "row_id": decision["row_id"],
        "score": decision["score"],
        "label": decision["label"],
        "abstain": decision["abstain"],
        "block": decision["action"] == "recommend_reject_low_value",
        "support_count": decision["confidence_interval"]["support_count"],
    }


def _calibration_bins(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    bins: dict[str, list[int]] = defaultdict(list)
    by_id = {row["row_id"]: row for row in train_rows}
    for row in train_rows:
        row_train = [candidate for candidate in train_rows if candidate["row_id"] != row["row_id"]]
        pred = _predict("feature_similarity_simulator", row, row_train)
        bins[_bin_id(float(pred["score"]))].append(_label(by_id[row["row_id"]]))
    return {
        key: {"count": float(len(labels)), "observed_rate": sum(labels) / max(1, len(labels))}
        for key, labels in bins.items()
    }


def _confidence_interval(score: float, support_count: int) -> dict[str, float]:
    n = max(1, support_count)
    half_width = 1.96 * math.sqrt(max(0.0001, score * (1.0 - score)) / n)
    return {
        "lower": max(0.0, score - half_width),
        "upper": min(1.0, score + half_width),
    }


def _abstain_reason(
    *,
    score: float,
    support_count: int,
    calibration_bin_count: int,
    interval_width: float,
    calibration_error: float,
) -> str:
    if support_count < MIN_SUPPORT:
        return "low_support"
    if calibration_bin_count < MIN_CALIBRATION_BIN_SUPPORT:
        return "low_calibration_bin_support"
    if interval_width > MAX_INTERVAL_WIDTH:
        return "wide_confidence_interval"
    if calibration_error > MAX_CALIBRATION_ERROR:
        return "calibration_error"
    if abs(score - 0.5) < DECISION_MARGIN:
        return "near_decision_boundary"
    return "none"


def _route_action(*, score: float, uncertainty: float, abstain_reason: str, interval: dict[str, float]) -> str:
    if abstain_reason != "none":
        return "abstain_to_live_validation"
    if score >= 0.58 and uncertainty <= 0.30:
        return "recommend_run_ablation"
    if score <= 0.25 and interval["upper"] < 0.40 and uncertainty <= 0.22:
        return "recommend_reject_low_value"
    if score < 0.5:
        return "recommend_repair_scope"
    return "recommend_collect_more_evidence"


def _verifier_tier(action: str) -> str:
    if action == "recommend_run_ablation":
        return "tier2_ablation_judge"
    if action == "recommend_collect_more_evidence":
        return "tier1_evidence_expand_then_rescreen"
    if action == "recommend_repair_scope":
        return "tier2_scope_repair_with_control"
    if action == "recommend_reject_low_value":
        return "tier2_reject_with_manifest_and_manual_audit"
    return "tier3_live_validation_or_human_review"


def _bin_id(score: float) -> str:
    index = min(9, max(0, int(score * 10)))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def _low_support_stress_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    prediction = {
        "score": 0.71,
        "support_count": MIN_SUPPORT - 1,
    }
    calibration_bins = {_bin_id(0.71): {"count": float(MIN_CALIBRATION_BIN_SUPPORT), "observed_rate": 0.70}}
    return _decision_from_prediction(row, prediction, calibration_bins)


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simulator uncertainty and abstention routing.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="simulator_uncertainty_20260612")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_uncertainty_payload(
        root=root,
        eval_id=args.eval_id,
        dataset_path=Path(args.dataset),
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
