"""Cheap verifier / world model for assumption candidates.

This is the executable version of the reconstruction "world model" gap.  It
does not replace real ablations or judges.  It predicts which candidate paths
are worth spending on, records simulator manifests, and marks where real
validation must override the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .graph_memory import JsonlGraphStore
from .manifest_logger import make_component_manifest
from .schema import ResidualType, TrialStatus, stable_id


@dataclass(frozen=True)
class AssumptionWorldModelPrediction:
    proposal_id: str
    candidate_node_id: str | None
    parent_node_id: str
    predicted_acceptance_probability: float
    prediction_confidence: float
    predicted_regression_risk: str
    expected_utility: float
    recommended_verifier_tier: str
    recommended_next_action: str
    predicted_failure_modes: list[str] = field(default_factory=list)
    feature_trace: dict = field(default_factory=dict)
    observed_acceptance_label: str | None = None
    calibration_error: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_world_model_payload(
    *,
    store: JsonlGraphStore | None,
    proposal_payload: dict,
    preflight_payload: dict | None = None,
    falsification_payload: dict | None = None,
    acceptance_payload: dict | None = None,
    regression_predictions: Iterable[dict] | None = None,
    formal_mapping_gate_payload: dict | None = None,
    calibration_payload: dict | None = None,
    eval_id: str,
    writeback: bool = False,
) -> dict:
    """Predict proposal outcomes and optionally log simulator manifests."""

    preflight_by_id = _index(preflight_payload, "summaries")
    falsification_by_id = _index(falsification_payload, "summaries")
    acceptance_by_id = _index(acceptance_payload, "summaries")
    regression_by_id = {
        row.get("proposal_id"): row
        for row in list(regression_predictions or [])
        if row.get("proposal_id")
    }
    formal_by_id = _index(formal_mapping_gate_payload, "gates")

    predictions: list[AssumptionWorldModelPrediction] = []
    manifests = []
    for proposal in proposal_payload.get("proposals", []):
        prediction = predict_proposal_outcome(
            store=store,
            proposal=proposal,
            preflight=preflight_by_id.get(proposal.get("proposal_id", ""), {}),
            falsification=falsification_by_id.get(proposal.get("proposal_id", ""), {}),
            acceptance=acceptance_by_id.get(proposal.get("proposal_id", ""), {}),
            regression=regression_by_id.get(proposal.get("proposal_id", ""), {}),
            formal_gate=formal_by_id.get(proposal.get("proposal_id", ""), {}),
            calibration_payload=calibration_payload,
        )
        predictions.append(prediction)
        manifest = _prediction_manifest(eval_id=eval_id, proposal=proposal, prediction=prediction)
        manifests.append(manifest)
        if store is not None and writeback:
            store.append_trial(manifest)
    if store is not None and writeback:
        store.flush()

    calibration = _calibration_summary(predictions)
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "writeback": writeback,
        "prediction_count": len(predictions),
        "risk_counts": dict(Counter(p.predicted_regression_risk for p in predictions)),
        "verifier_tier_counts": dict(Counter(p.recommended_verifier_tier for p in predictions)),
        "recommended_action_counts": dict(Counter(p.recommended_next_action for p in predictions)),
        "calibration": calibration,
        "calibration_model": _compact_calibration_model(calibration_payload),
        "predictions": [p.to_dict() for p in predictions],
        "simulator_manifests": [m.to_dict() for m in manifests],
    }


def predict_proposal_outcome(
    *,
    store: JsonlGraphStore | None,
    proposal: dict,
    preflight: dict | None = None,
    falsification: dict | None = None,
    acceptance: dict | None = None,
    regression: dict | None = None,
    formal_gate: dict | None = None,
    calibration_payload: dict | None = None,
) -> AssumptionWorldModelPrediction:
    """Score a single proposal with transparent, auditable features."""

    preflight = preflight or {}
    falsification = falsification or {}
    acceptance = acceptance or {}
    regression = regression or {}
    formal_gate = formal_gate or {}
    proposal_id = proposal.get("proposal_id", "")
    candidate = proposal.get("candidate_node") or {}
    parent_id = proposal.get("parent_node_id", "")

    score = 0.5
    confidence = 0.25
    failure_modes: list[str] = []
    feature_trace = {
        "proposal_type": proposal.get("proposal_type"),
        "priority": float(proposal.get("priority", 0.0) or 0.0),
        "readiness": preflight.get("readiness"),
        "falsification_decision": falsification.get("decision"),
        "acceptance_decision": acceptance.get("decision"),
        "regression_risk": regression.get("risk"),
        "formal_gate_decision": formal_gate.get("decision"),
        "parent_confidence": None,
    }

    raw_priority = max(0.0, float(proposal.get("priority", 0.0) or 0.0))
    priority = min(1.0, raw_priority / 2.5)
    feature_trace["priority"] = raw_priority
    feature_trace["normalized_priority"] = priority
    score += 0.16 * priority
    confidence += 0.05 if priority else 0.0

    parent = store.nodes.get(parent_id) if store is not None else None
    if parent is not None:
        feature_trace["parent_confidence"] = parent.confidence
        score += 0.08 * (parent.confidence - 0.5)
        score += 0.06 * parent.metaproductivity
        confidence += 0.08

    readiness = preflight.get("readiness")
    if readiness == "ready_for_fresh_ablation":
        score += 0.22
        confidence += 0.18
    elif readiness == "needs_scope_fix":
        score -= 0.22
        confidence += 0.12
        failure_modes.append("scope_no_fire_exposure")
    elif readiness == "needs_retrieval_fix":
        score -= 0.12
        confidence += 0.1
        failure_modes.append("retrieval_underfire")
    elif readiness == "manifest_only":
        score -= 0.18
        failure_modes.append("not_testable_yet")

    falsification_decision = falsification.get("decision")
    if falsification_decision == "ready_for_ablation":
        score += 0.14
        confidence += 0.14
    elif falsification_decision in {"reject_benefit", "reject_harm"}:
        score -= 0.35
        confidence += 0.25
        failure_modes.append(f"falsification_{falsification_decision}")
    elif falsification_decision in {"repair_scope", "repair_retrieval_before_ablation"}:
        score -= 0.1
        failure_modes.append(falsification_decision)

    acceptance_decision = acceptance.get("decision")
    if acceptance_decision == "accept":
        score += 0.35
        confidence += 0.3
    elif acceptance_decision == "reject_benefit":
        score -= 0.38
        confidence += 0.3
        failure_modes.append("observed_weak_trigger_benefit")
    elif acceptance_decision == "reject_harm":
        score -= 0.48
        confidence += 0.35
        failure_modes.append("observed_control_harm")
    elif acceptance_decision == "insufficient_judgments":
        score -= 0.08
        confidence += 0.08
        failure_modes.append("underpowered_acceptance")

    risk = regression.get("risk", "unknown")
    if risk == "high":
        score -= 0.24
        confidence += 0.12
        failure_modes.append("high_regression_risk")
    elif risk == "medium":
        score -= 0.1
        confidence += 0.08
        failure_modes.append("medium_regression_risk")
    elif risk == "low":
        score += 0.04
        confidence += 0.04

    formal_decision = formal_gate.get("decision")
    if formal_gate.get("blocks_policy_update"):
        score -= 0.32
        confidence += 0.16
        failure_modes.append(f"formal_gate_{formal_decision}")
    elif formal_decision == "allow":
        score += 0.04
        confidence += 0.04

    probability = _calibrated_probability(
        _sigmoid_logit(score),
        acceptance_decision,
        feature_trace=feature_trace,
        calibration_payload=calibration_payload,
    )
    expected_utility = _bounded((probability - 0.5) * 2.0 - _risk_penalty(risk))
    observed_label = _observed_label(acceptance_decision)
    calibration_error = (
        abs(probability - observed_label)
        if observed_label is not None
        else None
    )
    recommended_verifier_tier = _verifier_tier(
        probability=probability,
        confidence=confidence,
        readiness=readiness,
        risk=risk,
        acceptance_decision=acceptance_decision,
        formal_blocks=bool(formal_gate.get("blocks_policy_update")),
    )
    recommended_next_action = _recommended_next_action(
        probability=probability,
        readiness=readiness,
        risk=risk,
        acceptance_decision=acceptance_decision,
        formal_blocks=bool(formal_gate.get("blocks_policy_update")),
    )

    return AssumptionWorldModelPrediction(
        proposal_id=proposal_id,
        candidate_node_id=candidate.get("id"),
        parent_node_id=parent_id,
        predicted_acceptance_probability=round(probability, 4),
        prediction_confidence=round(min(0.95, max(0.0, confidence)), 4),
        predicted_regression_risk=risk,
        expected_utility=round(expected_utility, 4),
        recommended_verifier_tier=recommended_verifier_tier,
        recommended_next_action=recommended_next_action,
        predicted_failure_modes=sorted(set(failure_modes)),
        feature_trace=feature_trace,
        observed_acceptance_label=(
            None if observed_label is None else ("accepted" if observed_label == 1.0 else "rejected")
        ),
        calibration_error=None if calibration_error is None else round(calibration_error, 4),
    )


def _prediction_manifest(*, eval_id: str, proposal: dict, prediction: AssumptionWorldModelPrediction):
    candidate = proposal.get("candidate_node") or {}
    assumption = candidate.get("claim") or f"World model rollout for {prediction.proposal_id}"
    status = TrialStatus.OBSERVED
    residual = None
    residual_type = None
    if prediction.observed_acceptance_label == "rejected" and prediction.predicted_acceptance_probability >= 0.65:
        residual = "World model predicted a promising candidate but acceptance evidence rejected it."
        residual_type = ResidualType.SIMULATOR_DEFECT
        status = TrialStatus.FAILED
    return make_component_manifest(
        eval_id=eval_id,
        event_type="simulator_rollout",
        problem_id=f"world_model::{prediction.proposal_id}",
        component="assumption_world_model",
        assumption=assumption,
        why_selected="Cheap simulator screens candidate hypotheses before expensive ablation or graph mutation.",
        expected_effect="Predict acceptance probability, regression risk, and verifier tier for the proposal.",
        assumption_ids=[x for x in [prediction.parent_node_id, prediction.candidate_node_id] if x],
        verifier=prediction.recommended_verifier_tier,
        verification_plan="Override simulator with fresh ablation/judge evidence before promotion.",
        rollback_condition="Treat simulator as defective if accepted/rejected evidence contradicts high-confidence predictions.",
        status=status,
        observed_effect=(
            f"predicted_acceptance_probability={prediction.predicted_acceptance_probability}; "
            f"expected_utility={prediction.expected_utility}; action={prediction.recommended_next_action}"
        ),
        residual=residual,
        residual_type=residual_type,
        artifacts={"prediction": prediction.to_dict()},
        metadata={"proposal_id": prediction.proposal_id},
    )


def _index(payload: dict | None, key: str) -> dict[str, dict]:
    if not payload:
        return {}
    return {
        row.get("proposal_id"): row
        for row in payload.get(key, [])
        if row.get("proposal_id")
    }


def _calibration_summary(predictions: list[AssumptionWorldModelPrediction]) -> dict:
    labeled = [p for p in predictions if p.calibration_error is not None]
    if not labeled:
        return {"labeled_predictions": 0, "mean_absolute_error": None, "brier_score": None}
    brier = sum(
        (p.predicted_acceptance_probability - (1.0 if p.observed_acceptance_label == "accepted" else 0.0)) ** 2
        for p in labeled
    ) / len(labeled)
    mae = sum(float(p.calibration_error or 0.0) for p in labeled) / len(labeled)
    return {
        "labeled_predictions": len(labeled),
        "mean_absolute_error": round(mae, 4),
        "brier_score": round(brier, 4),
    }


def train_world_model_calibration(
    *,
    prediction_payload: dict,
    acceptance_payload: dict,
    eval_id: str,
) -> dict:
    """Train a small transparent calibration payload from labeled predictions."""

    labels = _acceptance_labels(acceptance_payload)
    rows = []
    matched_proposal_ids: set[str] = set()
    for prediction in prediction_payload.get("predictions", []):
        proposal_id = prediction.get("proposal_id")
        if proposal_id not in labels:
            continue
        matched_proposal_ids.add(proposal_id)
        feature_trace = prediction.get("feature_trace", {})
        rows.append({
            "proposal_id": proposal_id,
            "label": labels[proposal_id],
            "probability": float(prediction.get("predicted_acceptance_probability", 0.5) or 0.5),
            "raw_priority": float(feature_trace.get("priority", 0.0) or 0.0),
            "readiness": feature_trace.get("readiness"),
            "risk": feature_trace.get("regression_risk"),
        })

    accepted = [row for row in rows if row["label"] == "accept"]
    rejected = [row for row in rows if row["label"] == "reject"]
    decision_probabilities = _decision_probability_model(acceptance_payload)
    priority_model = _priority_calibration_model(rows)
    priority_boundary = priority_model.get("priority_boundary")
    high_priority_accept_floor = priority_model.get("high_priority_accept_floor")
    low_priority_probability_cap = priority_model.get("low_priority_probability_cap")

    raw_metrics = _probability_metrics(rows, probability_key="probability")
    calibrated_rows = [
        {
            **row,
            "calibrated_probability": _apply_probability_calibration(
                probability=row["probability"],
                feature_trace={"priority": row["raw_priority"], "readiness": row["readiness"], "regression_risk": row["risk"]},
                acceptance_decision=None,
                calibration_payload={
                    "priority_boundary": priority_boundary,
                    "high_priority_accept_floor": high_priority_accept_floor,
                    "low_priority_probability_cap": low_priority_probability_cap,
                    "decision_probabilities": decision_probabilities,
                },
            ),
        }
        for row in rows
    ]
    calibrated_metrics = _probability_metrics(calibrated_rows, probability_key="calibrated_probability")
    loo_metrics = _leave_one_out_priority_calibration_metrics(rows)
    return {
        "eval_id": eval_id,
        "source_prediction_eval_id": prediction_payload.get("eval_id"),
        "source_acceptance_eval_id": acceptance_payload.get("eval_id"),
        "status": "trained" if rows else "no_matched_labels",
        "labeled_count": len(rows),
        "matched_label_count": len(rows),
        "unmatched_label_count": len(set(labels) - matched_proposal_ids),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "priority_boundary": priority_boundary,
        "high_priority_accept_floor": high_priority_accept_floor,
        "low_priority_probability_cap": low_priority_probability_cap,
        "decision_probabilities": decision_probabilities,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "leave_one_out_calibrated_metrics": loo_metrics,
        "training_rows": calibrated_rows,
    }


def _priority_calibration_model(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["label"] == "accept"]
    rejected = [row for row in rows if row["label"] == "reject"]
    if not accepted or not rejected:
        return {
            "priority_boundary": None,
            "high_priority_accept_floor": None,
            "low_priority_probability_cap": None,
        }
    min_accept_priority = min(row["raw_priority"] for row in accepted)
    max_reject_priority = max(row["raw_priority"] for row in rejected)
    if min_accept_priority <= max_reject_priority:
        return {
            "priority_boundary": None,
            "high_priority_accept_floor": None,
            "low_priority_probability_cap": None,
        }
    return {
        "priority_boundary": round((min_accept_priority + max_reject_priority) / 2.0, 4),
        "high_priority_accept_floor": _smoothed_rate(len(accepted), len(accepted), alpha=1.0),
        "low_priority_probability_cap": _smoothed_rate(0, len(rejected), alpha=1.0),
    }


def _leave_one_out_priority_calibration_metrics(rows: list[dict]) -> dict:
    heldout = []
    for index, row in enumerate(rows):
        train_rows = [other for j, other in enumerate(rows) if j != index]
        model = _priority_calibration_model(train_rows)
        heldout.append({
            **row,
            "loo_calibrated_probability": _apply_probability_calibration(
                probability=row["probability"],
                feature_trace={"priority": row["raw_priority"], "readiness": row["readiness"], "regression_risk": row["risk"]},
                acceptance_decision=None,
                calibration_payload=model,
            ),
        })
    return _probability_metrics(heldout, probability_key="loo_calibrated_probability")


def _sigmoid_logit(score: float) -> float:
    return _bounded(1.0 / (1.0 + math.exp(-3.0 * (score - 0.5))))


def _calibrated_probability(
    probability: float,
    acceptance_decision: str | None,
    *,
    feature_trace: dict,
    calibration_payload: dict | None,
) -> float:
    """Use real acceptance evidence as calibration evidence when available."""

    calibrated = _apply_probability_calibration(
        probability=probability,
        feature_trace=feature_trace,
        acceptance_decision=acceptance_decision,
        calibration_payload=calibration_payload,
    )
    if calibrated is not None:
        return calibrated
    if acceptance_decision == "accept":
        return max(probability, 0.9)
    if acceptance_decision == "reject_harm":
        return min(probability, 0.12)
    if acceptance_decision == "reject_benefit":
        return min(probability, 0.22)
    if acceptance_decision == "insufficient_judgments":
        return min(probability, 0.52)
    return probability


def _apply_probability_calibration(
    *,
    probability: float,
    feature_trace: dict,
    acceptance_decision: str | None,
    calibration_payload: dict | None,
) -> float | None:
    if not calibration_payload:
        return None
    decision_probabilities = calibration_payload.get("decision_probabilities", {})
    if acceptance_decision and acceptance_decision in decision_probabilities:
        return float(decision_probabilities[acceptance_decision])

    priority_boundary = calibration_payload.get("priority_boundary")
    raw_priority = float(feature_trace.get("priority", 0.0) or 0.0)
    if priority_boundary is not None:
        if raw_priority >= float(priority_boundary) and calibration_payload.get("high_priority_accept_floor") is not None:
            return max(probability, float(calibration_payload["high_priority_accept_floor"]))
        if raw_priority < float(priority_boundary) and calibration_payload.get("low_priority_probability_cap") is not None:
            return min(probability, float(calibration_payload["low_priority_probability_cap"]))
    return probability


def _acceptance_labels(acceptance_payload: dict) -> dict[str, str]:
    labels = {}
    for row in acceptance_payload.get("summaries", []):
        decision = row.get("decision")
        if decision == "accept":
            labels[row.get("proposal_id", "")] = "accept"
        elif decision in {"reject_benefit", "reject_harm"}:
            labels[row.get("proposal_id", "")] = "reject"
    return {k: v for k, v in labels.items() if k}


def _decision_probability_model(acceptance_payload: dict) -> dict[str, float]:
    counts: dict[str, Counter[str]] = {}
    for row in acceptance_payload.get("summaries", []):
        decision = row.get("decision")
        if not decision:
            continue
        label = "accept" if decision == "accept" else "reject"
        counts.setdefault(decision, Counter())[label] += 1
    out = {}
    for decision, counter in counts.items():
        accepts = counter.get("accept", 0)
        total = sum(counter.values())
        out[decision] = round(_smoothed_rate(accepts, total, alpha=0.5), 4)
    if "accept" not in out:
        out["accept"] = 0.9
    if "reject_benefit" not in out:
        out["reject_benefit"] = 0.22
    if "reject_harm" not in out:
        out["reject_harm"] = 0.12
    if "insufficient_judgments" not in out:
        out["insufficient_judgments"] = 0.5
    return out


def _probability_metrics(rows: list[dict], *, probability_key: str) -> dict:
    if not rows:
        return {"labeled_count": 0, "brier_score": None, "mean_absolute_error": None}
    brier = 0.0
    mae = 0.0
    for row in rows:
        y = 1.0 if row["label"] == "accept" else 0.0
        p = float(row[probability_key])
        brier += (p - y) ** 2
        mae += abs(p - y)
    return {
        "labeled_count": len(rows),
        "brier_score": round(brier / len(rows), 4),
        "mean_absolute_error": round(mae / len(rows), 4),
    }


def _smoothed_rate(successes: int, total: int, *, alpha: float) -> float:
    return round((successes + alpha) / (total + 2 * alpha), 4)


def _compact_calibration_model(calibration_payload: dict | None) -> dict | None:
    if not calibration_payload:
        return None
    return {
        "eval_id": calibration_payload.get("eval_id"),
        "labeled_count": calibration_payload.get("labeled_count"),
        "priority_boundary": calibration_payload.get("priority_boundary"),
        "high_priority_accept_floor": calibration_payload.get("high_priority_accept_floor"),
        "low_priority_probability_cap": calibration_payload.get("low_priority_probability_cap"),
        "decision_probabilities": calibration_payload.get("decision_probabilities", {}),
    }


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _risk_penalty(risk: str) -> float:
    return {"high": 0.35, "medium": 0.15, "low": 0.0}.get(risk, 0.1)


def _observed_label(decision: str | None) -> float | None:
    if decision == "accept":
        return 1.0
    if decision in {"reject_benefit", "reject_harm"}:
        return 0.0
    return None


def _verifier_tier(
    *,
    probability: float,
    confidence: float,
    readiness: str | None,
    risk: str,
    acceptance_decision: str | None,
    formal_blocks: bool,
) -> str:
    if formal_blocks:
        return "V6_formal_repair_gate"
    if acceptance_decision:
        return "V5_acceptance_evidence"
    if readiness != "ready_for_fresh_ablation":
        return "V2_preflight_repair"
    if risk == "high":
        return "V4_control_harm_ablation"
    if probability >= 0.72 and confidence >= 0.45:
        return "V3_fresh_ablation"
    return "V1_collect_more_evidence"


def _recommended_next_action(
    *,
    probability: float,
    readiness: str | None,
    risk: str,
    acceptance_decision: str | None,
    formal_blocks: bool,
) -> str:
    if formal_blocks:
        return "repair_formal_mapping_before_policy_update"
    if acceptance_decision == "accept":
        return "apply_accepted_candidate_if_requested"
    if acceptance_decision in {"reject_benefit", "reject_harm"}:
        return "reject_or_revise_candidate"
    if readiness == "needs_scope_fix" or risk == "high":
        return "repair_scope_before_ablation"
    if readiness == "needs_retrieval_fix":
        return "repair_retrieval_before_ablation"
    if readiness == "ready_for_fresh_ablation" and probability >= 0.55:
        return "run_fresh_ablation"
    return "collect_more_evidence"


def _load_json(path: Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path | None, payload: dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--preflight", default=None)
    ap.add_argument("--falsification", default=None)
    ap.add_argument("--acceptance", default=None)
    ap.add_argument("--regression-predictions", default=None)
    ap.add_argument("--formal-gate", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--train-calibration-out", default=None,
                    help="train a reusable calibration artifact from raw pre-acceptance predictions and --acceptance")
    ap.add_argument("--raw-prediction-out", default=None,
                    help="optionally persist the raw pre-acceptance predictions used for calibration")
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    graph_dir = _resolve(root, args.graph_dir)
    proposal_payload = _load_json(_resolve(root, args.proposals)) or {}
    preflight_payload = _load_json(_resolve(root, args.preflight))
    falsification_payload = _load_json(_resolve(root, args.falsification))
    acceptance_payload = _load_json(_resolve(root, args.acceptance))
    regression_predictions = _load_json(_resolve(root, args.regression_predictions)) or []
    formal_gate_payload = _load_json(_resolve(root, args.formal_gate))
    calibration_payload = _load_json(_resolve(root, args.calibration))
    raw_prediction_payload = None

    if args.train_calibration_out:
        if acceptance_payload is None:
            raise SystemExit("--train-calibration-out requires --acceptance")
        raw_prediction_payload = build_world_model_payload(
            store=JsonlGraphStore(graph_dir),
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            falsification_payload=falsification_payload,
            acceptance_payload=None,
            regression_predictions=regression_predictions,
            formal_mapping_gate_payload=formal_gate_payload,
            calibration_payload=None,
            eval_id=f"{args.eval_id}_raw_for_calibration",
            writeback=False,
        )
        calibration_payload = train_world_model_calibration(
            prediction_payload=raw_prediction_payload,
            acceptance_payload=acceptance_payload,
            eval_id=f"{args.eval_id}_calibration",
        )
        _write_json(_resolve(root, args.train_calibration_out), calibration_payload)
    elif args.raw_prediction_out:
        raw_prediction_payload = build_world_model_payload(
            store=JsonlGraphStore(graph_dir),
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            falsification_payload=falsification_payload,
            acceptance_payload=None,
            regression_predictions=regression_predictions,
            formal_mapping_gate_payload=formal_gate_payload,
            calibration_payload=None,
            eval_id=f"{args.eval_id}_raw_for_calibration",
            writeback=False,
        )

    if raw_prediction_payload is not None:
        _write_json(_resolve(root, args.raw_prediction_out), raw_prediction_payload)

    payload = build_world_model_payload(
        store=JsonlGraphStore(graph_dir),
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        falsification_payload=falsification_payload,
        acceptance_payload=acceptance_payload,
        regression_predictions=regression_predictions,
        formal_mapping_gate_payload=formal_gate_payload,
        calibration_payload=calibration_payload,
        eval_id=args.eval_id,
        writeback=args.writeback,
    )
    if raw_prediction_payload is not None:
        payload["raw_prediction_eval_id"] = raw_prediction_payload.get("eval_id")
    if args.train_calibration_out and calibration_payload:
        payload["trained_calibration_artifact"] = {
            key: value for key, value in calibration_payload.items() if key != "training_rows"
        }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _write_json(_resolve(root, args.summary_out), payload)
    print(text)


if __name__ == "__main__":
    main()
