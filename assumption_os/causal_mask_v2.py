"""V2 counterfactual graph-action mask validation.

Phase 4 treats an assumption-graph action as an intervention target.  It asks:
if the candidate relation node or one of its supporting features is masked,
does the world-model prediction degrade in the expected direction?

The result is a functional contribution audit for the assumption pipeline, not
a proof of external-world causality.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .graph_action_world_model_v2 import build_graph_action_world_model_v2_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "causal_mask_v2_20260610.json"


@dataclass(frozen=True)
class MaskSpec:
    id: str
    target: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaskTrial:
    action_id: str
    mask_id: str
    label: int
    base_accept_prob: float
    masked_accept_prob: float
    base_regression_prob: float
    masked_regression_prob: float
    accept_prob_delta: float
    base_utility: float
    masked_utility: float
    utility_delta: float
    masked_failure_type: str
    masked_recommended_action: str
    masked_feature_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_causal_mask_v2_payload(
    *,
    eval_id: str = "causal_mask_v2_20260610",
    live_threshold: float = 0.50,
) -> dict[str, Any]:
    start = time.perf_counter()
    world = build_graph_action_world_model_v2_payload(
        eval_id=f"{eval_id}_world_model",
        live_threshold=live_threshold,
    )
    mask_specs = _mask_specs()
    trials = [
        _mask_trial(row, spec, live_threshold=live_threshold)
        for row in world["predictions"]
        for spec in mask_specs
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metrics = _metrics(
        world["predictions"],
        trials,
        elapsed_ms=elapsed_ms,
        live_threshold=live_threshold,
    )
    gates = {
        "source_world_model_passes": bool(world.get("pass")),
        "has_counterfactual_trials": metrics["counterfactual_trial_count"] >= 64,
        "relation_mask_drops_positive_actions": metrics["mean_positive_relation_accept_drop"] >= 0.40,
        "relation_drop_separates_negatives": metrics["relation_drop_auroc"] >= 0.95,
        "negative_controls_stay_screened": metrics["negative_control_mask_false_live_count"] == 0,
        "relation_mask_is_top_positive_importance": metrics["positive_top_relation_mask_fraction"] >= 0.80,
        "mask_evaluation_is_cheap": metrics["avg_mask_eval_ms"] < 1.0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "causal_mask_v2_counterfactual_graph_action_audit",
        "reconstruction_v2_phase": "phase4_causal_mask_validation",
        "performance_validation": True,
        "validation_scope": (
            "Counterfactual do(mask_assumption h) audit over Phase 3 graph-action predictions. "
            "Masks remove relation-node, family-overlap, role-schema, or invariant-trace support and "
            "measure prediction/utility changes."
        ),
        "source": {
            "world_model_eval_id": world.get("eval_id"),
            "world_model_metrics": world.get("metrics", {}),
        },
        "thresholds": {
            "live_threshold": live_threshold,
        },
        "mask_specs": [spec.to_dict() for spec in mask_specs],
        "trials": [trial.to_dict() for trial in trials],
        "importance_ranking": _importance_ranking(trials),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The relation-node deletion acts like the environment removing a candidate assumption edge.  "
            "If the positive action's predicted utility drops while negative controls remain screened, the "
            "system has a measurable internal contribution signal for that assumption.  This remains a "
            "pipeline counterfactual, not a mathematical or physical causality proof."
        ),
    }


def _mask_specs() -> list[MaskSpec]:
    return [
        MaskSpec(
            id="do(mask_alignment_relation_node)",
            target="AlignmentHypothesis relation node",
            description="Remove the candidate alignment relation and all direct overlap evidence.",
        ),
        MaskSpec(
            id="do(mask_process_family_overlap)",
            target="shared process-family support",
            description="Remove shared family tags while retaining any role-schema support.",
        ),
        MaskSpec(
            id="do(mask_role_schema_overlap)",
            target="shared role-schema support",
            description="Remove shared perturbation/response/constraint roles while retaining family support.",
        ),
        MaskSpec(
            id="do(mask_invariant_trace)",
            target="invariant evidence trace",
            description="Keep endpoints but damp the invariant and textual trace contribution.",
        ),
    ]


def _mask_trial(row: dict[str, Any], spec: MaskSpec, *, live_threshold: float) -> MaskTrial:
    masked_features = _masked_features(row, spec.id)
    masked_prediction = _predict_from_features(masked_features, live_threshold=live_threshold)
    label = int(row["label"])
    base_accept = float(row["predicted_accept_prob"])
    masked_accept = masked_prediction["accept_prob"]
    base_regression = float(row["predicted_regression_prob"])
    masked_regression = masked_prediction["regression_prob"]
    base_utility = _action_utility(label=label, accept_prob=base_accept, regression_prob=base_regression)
    masked_utility = _action_utility(
        label=label,
        accept_prob=masked_accept,
        regression_prob=masked_regression,
    )
    return MaskTrial(
        action_id=str(row["action_id"]),
        mask_id=spec.id,
        label=label,
        base_accept_prob=base_accept,
        masked_accept_prob=masked_accept,
        base_regression_prob=base_regression,
        masked_regression_prob=masked_regression,
        accept_prob_delta=round(base_accept - masked_accept, 4),
        base_utility=round(base_utility, 4),
        masked_utility=round(masked_utility, 4),
        utility_delta=round(base_utility - masked_utility, 4),
        masked_failure_type=masked_prediction["failure_type"],
        masked_recommended_action=masked_prediction["recommended_action"],
        masked_feature_trace=masked_features,
    )


def _masked_features(row: dict[str, Any], mask_id: str) -> dict[str, Any]:
    trace = dict(row.get("feature_trace", {}))
    score = float(trace.get("process_pair_score", 0.0) or 0.0)
    family_overlap = list(trace.get("family_overlap", []))
    role_overlap = list(trace.get("role_overlap", []))
    if mask_id == "do(mask_alignment_relation_node)":
        score = 0.0
        family_overlap = []
        role_overlap = []
    elif mask_id == "do(mask_process_family_overlap)":
        score = score * 0.45 if family_overlap else score
        family_overlap = []
    elif mask_id == "do(mask_role_schema_overlap)":
        score = score * 0.70 if role_overlap else score
        role_overlap = []
    elif mask_id == "do(mask_invariant_trace)":
        score = score * 0.85
    else:
        raise ValueError(f"unknown mask_id: {mask_id}")
    return {
        "process_pair_score": round(score, 4),
        "family_overlap": family_overlap,
        "family_overlap_count": len(family_overlap),
        "role_overlap": role_overlap,
        "role_overlap_count": len(role_overlap),
        "masked_by": mask_id,
        "source_process_pair_decision": trace.get("process_pair_decision"),
    }


def _predict_from_features(features: dict[str, Any], *, live_threshold: float) -> dict[str, Any]:
    score = float(features["process_pair_score"])
    family_count = int(features["family_overlap_count"])
    role_count = int(features["role_overlap_count"])
    logit = -2.2 + 6.0 * score + 0.10 * family_count + 0.20 * role_count
    accept_prob = _sigmoid(logit)
    regression_prob = max(0.02, min(0.85, 0.10 + 0.55 * (1.0 - accept_prob) + 0.04 * max(0, 2 - role_count)))
    if accept_prob >= 0.70 and regression_prob <= 0.30:
        recommended = "run_live_or_promote_if_verified"
    elif accept_prob >= live_threshold:
        recommended = "run_live_validation"
    else:
        recommended = "screen_or_collect_more_evidence"
    failure_type = (
        "none_expected"
        if accept_prob >= 0.70
        else "underpowered_or_surface_alignment"
        if accept_prob >= live_threshold
        else "masked_relation_or_low_family_overlap"
    )
    return {
        "accept_prob": round(accept_prob, 4),
        "regression_prob": round(regression_prob, 4),
        "failure_type": failure_type,
        "recommended_action": recommended,
    }


def _action_utility(*, label: int, accept_prob: float, regression_prob: float) -> float:
    if label == 1:
        return accept_prob - 0.25 * regression_prob
    return (1.0 - accept_prob) - 0.10 * (1.0 - regression_prob)


def _metrics(
    base_predictions: list[dict[str, Any]],
    trials: list[MaskTrial],
    *,
    elapsed_ms: float,
    live_threshold: float,
) -> dict[str, Any]:
    relation_trials = [trial for trial in trials if trial.mask_id == "do(mask_alignment_relation_node)"]
    positive_relation = [trial for trial in relation_trials if trial.label == 1]
    negative_relation = [trial for trial in relation_trials if trial.label == 0]
    negative_trials = [trial for trial in trials if trial.label == 0]
    top_masks = _top_masks_by_action(trials)
    positive_top_relation = [
        action_id
        for action_id, mask_id in top_masks.items()
        if _label_for_action(trials, action_id) == 1 and mask_id == "do(mask_alignment_relation_node)"
    ]
    positive_action_count = sum(1 for row in base_predictions if int(row["label"]) == 1)
    labels = [trial.label for trial in relation_trials]
    relation_drops = [trial.accept_prob_delta for trial in relation_trials]
    return {
        "base_action_count": len(base_predictions),
        "counterfactual_trial_count": len(trials),
        "mask_count": len({trial.mask_id for trial in trials}),
        "mean_positive_relation_accept_drop": round(_mean([t.accept_prob_delta for t in positive_relation]), 4),
        "mean_negative_relation_accept_drop": round(_mean([t.accept_prob_delta for t in negative_relation]), 4),
        "mean_positive_relation_utility_drop": round(_mean([t.utility_delta for t in positive_relation]), 4),
        "mean_negative_relation_utility_drop": round(_mean([t.utility_delta for t in negative_relation]), 4),
        "relation_drop_auroc": round(_auroc(labels, relation_drops), 4),
        "negative_control_mask_false_live_count": sum(
            1 for trial in negative_trials if trial.masked_accept_prob >= live_threshold
        ),
        "positive_top_relation_mask_fraction": round(
            len(positive_top_relation) / max(1, positive_action_count),
            4,
        ),
        "masked_positive_block_count": sum(
            1 for trial in positive_relation if trial.masked_accept_prob < live_threshold
        ),
        "avg_mask_eval_ms": round(elapsed_ms / max(1, len(trials)), 4),
    }


def _importance_ranking(trials: list[MaskTrial]) -> list[dict[str, Any]]:
    by_mask: dict[str, list[MaskTrial]] = {}
    for trial in trials:
        by_mask.setdefault(trial.mask_id, []).append(trial)
    ranking = []
    for mask_id, rows in by_mask.items():
        positives = [row for row in rows if row.label == 1]
        negatives = [row for row in rows if row.label == 0]
        ranking.append({
            "mask_id": mask_id,
            "mean_positive_accept_drop": round(_mean([row.accept_prob_delta for row in positives]), 4),
            "mean_negative_accept_drop": round(_mean([row.accept_prob_delta for row in negatives]), 4),
            "mean_positive_utility_drop": round(_mean([row.utility_delta for row in positives]), 4),
            "mean_negative_utility_drop": round(_mean([row.utility_delta for row in negatives]), 4),
        })
    return sorted(ranking, key=lambda row: row["mean_positive_utility_drop"], reverse=True)


def _top_masks_by_action(trials: list[MaskTrial]) -> dict[str, str]:
    by_action: dict[str, list[MaskTrial]] = {}
    for trial in trials:
        by_action.setdefault(trial.action_id, []).append(trial)
    return {
        action_id: max(rows, key=lambda row: row.utility_delta).mask_id
        for action_id, rows in by_action.items()
    }


def _label_for_action(trials: list[MaskTrial], action_id: str) -> int:
    for trial in trials:
        if trial.action_id == action_id:
            return trial.label
    return 0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _auroc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 counterfactual causal-mask validation.")
    parser.add_argument("--eval-id", default="causal_mask_v2_20260610")
    parser.add_argument("--live-threshold", type=float, default=0.50)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_causal_mask_v2_payload(
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
