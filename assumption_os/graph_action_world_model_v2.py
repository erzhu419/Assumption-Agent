"""V2 graph-action world model over discrete assumption graph operations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .process_model_zoo_v2 import build_process_model_zoo_v2_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "graph_action_world_model_v2_20260610.json"


@dataclass(frozen=True)
class GraphActionPrediction:
    action_id: str
    action_type: str
    source_process: str
    target_process: str
    predicted_accept_prob: float
    predicted_regression_prob: float
    predicted_failure_type: str
    predicted_value_delta: float
    expected_cost: float
    recommended_action: str
    label: int
    feature_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_graph_action_world_model_v2_payload(
    *,
    eval_id: str = "graph_action_world_model_v2_20260610",
    live_threshold: float = 0.50,
) -> dict[str, Any]:
    zoo = build_process_model_zoo_v2_payload(eval_id=f"{eval_id}_zoo")
    predictions = [
        predict_graph_action(row)
        for row in zoo["pair_judgments"]
    ]
    metrics = _metrics(predictions, live_threshold=live_threshold)
    gates = {
        "source_zoo_passes": bool(zoo.get("pass")),
        "has_labeled_graph_actions": metrics["labeled_count"] >= 16,
        "accept_auroc_high": metrics["accept_auroc"] >= 0.95,
        "accept_brier_beats_base_rate": metrics["accept_brier"] < metrics["base_rate_brier"],
        "screen_blocks_no_true_positive": metrics["accepted_blocked_count"] == 0,
        "screen_saves_negative_actions": metrics["negative_actions_saved"] >= 7,
        "regression_risk_ordered": metrics["mean_regression_positive"] < metrics["mean_regression_negative"],
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "graph_action_world_model_v2",
        "reconstruction_v2_phase": "phase3_graph_action_world_model",
        "performance_validation": True,
        "validation_scope": (
            "Discrete graph-action world model over Phase 2 process-pair actions.  It predicts whether "
            "adding an alignment hypothesis should be sent to live validation or screened out."
        ),
        "thresholds": {
            "live_threshold": live_threshold,
        },
        "source": {
            "process_zoo_eval_id": zoo.get("eval_id"),
            "label_semantics": "positive process-pair alignment = accepted graph action; negative control pair = rejected graph action",
        },
        "predictions": [prediction.to_dict() for prediction in predictions],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "This is the first v2 world model that predicts consequences of discrete graph actions.  It is "
            "not a physics simulator; it is a search-control model for whether a candidate graph overlay is "
            "worth live verifier spend."
        ),
    }


def predict_graph_action(row: dict[str, Any]) -> GraphActionPrediction:
    score = float(row.get("score", 0.0) or 0.0)
    family_count = len(row.get("family_overlap", []))
    role_count = len(row.get("role_overlap", []))
    action_type = "add_alignment_hypothesis"
    logit = -2.2 + 6.0 * score + 0.10 * family_count + 0.20 * role_count
    accept_prob = _sigmoid(logit)
    regression_prob = max(0.02, min(0.85, 0.10 + 0.55 * (1.0 - accept_prob) + 0.04 * max(0, 2 - role_count)))
    expected_value_delta = 0.24 * accept_prob - 0.11 * regression_prob - 0.04
    if accept_prob >= 0.70 and regression_prob <= 0.30:
        recommended = "run_live_or_promote_if_verified"
    elif accept_prob >= 0.50:
        recommended = "run_live_validation"
    else:
        recommended = "screen_or_collect_more_evidence"
    failure_type = (
        "none_expected"
        if accept_prob >= 0.70
        else "underpowered_or_surface_alignment"
        if accept_prob >= 0.50
        else "negative_control_or_low_family_overlap"
    )
    action_id = f"wm_action::{row.get('source_id')}::{row.get('target_id')}"
    label = 1 if row.get("gold_label") == "positive" else 0
    return GraphActionPrediction(
        action_id=action_id,
        action_type=action_type,
        source_process=row.get("source_id", ""),
        target_process=row.get("target_id", ""),
        predicted_accept_prob=round(accept_prob, 4),
        predicted_regression_prob=round(regression_prob, 4),
        predicted_failure_type=failure_type,
        predicted_value_delta=round(expected_value_delta, 4),
        expected_cost=1.0,
        recommended_action=recommended,
        label=label,
        feature_trace={
            "process_pair_score": score,
            "family_overlap_count": family_count,
            "role_overlap_count": role_count,
            "process_pair_decision": row.get("decision"),
            "family_overlap": row.get("family_overlap", []),
            "role_overlap": row.get("role_overlap", []),
        },
    )


def _metrics(predictions: list[GraphActionPrediction], *, live_threshold: float) -> dict[str, Any]:
    labels = [prediction.label for prediction in predictions]
    probs = [prediction.predicted_accept_prob for prediction in predictions]
    positives = [prediction for prediction in predictions if prediction.label == 1]
    negatives = [prediction for prediction in predictions if prediction.label == 0]
    run = [prediction for prediction in predictions if prediction.predicted_accept_prob >= live_threshold]
    blocked = [prediction for prediction in predictions if prediction.predicted_accept_prob < live_threshold]
    accepted_blocked = [prediction for prediction in blocked if prediction.label == 1]
    negative_saved = [prediction for prediction in blocked if prediction.label == 0]
    base_rate = sum(labels) / max(1, len(labels))
    return {
        "labeled_count": len(predictions),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "accept_auroc": round(_auroc(labels, probs), 4),
        "accept_brier": round(_brier(labels, probs), 4),
        "base_rate": round(base_rate, 4),
        "base_rate_brier": round(_brier(labels, [base_rate] * len(labels)), 4),
        "screen_live_count": len(run),
        "screen_block_count": len(blocked),
        "accepted_blocked_count": len(accepted_blocked),
        "negative_actions_saved": len(negative_saved),
        "screen_cost_reduction": round(len(blocked) / max(1, len(predictions)), 4),
        "mean_regression_positive": round(_mean([p.predicted_regression_prob for p in positives]), 4),
        "mean_regression_negative": round(_mean([p.predicted_regression_prob for p in negatives]), 4),
        "mean_value_delta_positive": round(_mean([p.predicted_value_delta for p in positives]), 4),
        "mean_value_delta_negative": round(_mean([p.predicted_value_delta for p in negatives]), 4),
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _brier(labels: list[int], probs: list[float]) -> float:
    if not labels:
        return 0.0
    return sum((prob - label) ** 2 for label, prob in zip(labels, probs)) / len(labels)


def _auroc(labels: list[int], probs: list[float]) -> float:
    positives = [prob for label, prob in zip(labels, probs) if label == 1]
    negatives = [prob for label, prob in zip(labels, probs) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 graph-action world model validation.")
    parser.add_argument("--eval-id", default="graph_action_world_model_v2_20260610")
    parser.add_argument("--live-threshold", type=float, default=0.50)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_graph_action_world_model_v2_payload(
        eval_id=args.eval_id,
        live_threshold=args.live_threshold,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
