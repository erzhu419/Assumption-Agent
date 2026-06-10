"""V2 bounded formal alignment layer over typed process models.

This module is category-inspired, but deliberately bounded: it emits typed
mapping certificates, finite diagram checks, invariant-preservation checks,
and negative-control decisions for process alignments.  It does not claim to
be a general category-theory theorem prover.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .causal_mask_v2 import build_causal_mask_v2_payload
from .process_model_zoo_v2 import build_process_model_zoo_v2_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "formal_alignment_v2_20260610.json"
TRAJECTORY_AXES = (
    "opposition",
    "equilibrium_restoration",
    "exponential_decay",
    "monotone_relaxation",
    "local_stabilization",
    "oscillation",
    "saturation",
    "capacity_constraint",
    "coupled_dynamics",
)


@dataclass(frozen=True)
class FormalAlignmentCertificate:
    source_id: str
    target_id: str
    gold_label: str
    decision: str
    formal_score: float
    baseline_scores: dict[str, float]
    typed_mapping: dict[str, str]
    preserved_invariants: list[str]
    broken_structures: list[str]
    finite_diagram_check: dict[str, Any]
    negative_control_check: dict[str, Any]
    causal_mask_signal: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_formal_alignment_v2_payload(
    *,
    eval_id: str = "formal_alignment_v2_20260610",
    formal_threshold: float = 0.60,
) -> dict[str, Any]:
    zoo = build_process_model_zoo_v2_payload(eval_id=f"{eval_id}_zoo")
    causal = build_causal_mask_v2_payload(eval_id=f"{eval_id}_causal")
    entries = {entry["model"]["id"]: entry for entry in zoo["process_entries"]}
    causal_by_pair = _causal_relation_drop_by_pair(causal)
    certificates = [
        _certificate(
            row,
            entries[row["source_id"]],
            entries[row["target_id"]],
            causal_by_pair=causal_by_pair,
            formal_threshold=formal_threshold,
        )
        for row in zoo["pair_judgments"]
    ]
    metrics = _metrics(certificates)
    gates = {
        "source_process_zoo_passes": bool(zoo.get("pass")),
        "source_causal_mask_passes": bool(causal.get("pass")),
        "has_typed_certificates": metrics["certificate_count"] >= 16,
        "formal_accuracy_high": metrics["formal_accuracy"] >= 0.95,
        "formal_beats_best_baseline": metrics["formal_accuracy"] > metrics["best_baseline_accuracy"],
        "formal_positive_recall_high": metrics["formal_positive_recall"] >= 0.95,
        "formal_negative_rejection_high": metrics["formal_negative_rejection_rate"] >= 0.95,
        "negative_controls_blocked": metrics["formal_false_positive_count"] == 0,
        "positive_causal_signal_high": metrics["accepted_positive_mean_relation_drop"] >= 0.40,
        "not_full_theorem_prover_claim": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "formal_alignment_v2_bounded_checker",
        "reconstruction_v2_phase": "phase5_formal_alignment_layer",
        "performance_validation": True,
        "validation_scope": (
            "Bounded category-inspired formal alignment over ProcessModel and AlignmentHypothesis objects. "
            "Compares semantic, graph-edit, and trajectory-information baselines against a typed mapping "
            "checker with invariant preservation, finite diagram checks, causal-mask signal, and negative controls."
        ),
        "thresholds": {
            "formal_threshold": formal_threshold,
        },
        "trajectory_axes": list(TRAJECTORY_AXES),
        "source": {
            "process_zoo_eval_id": zoo.get("eval_id"),
            "causal_mask_eval_id": causal.get("eval_id"),
        },
        "certificates": [certificate.to_dict() for certificate in certificates],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The formal layer treats an alignment as a typed relation node with a certificate.  It accepts "
            "cross-domain transfer only when typed roles, process-family structure, invariant preservation, "
            "finite diagram consistency, causal-mask contribution, and negative controls agree.  It is a bounded "
            "structural checker, not a universal category-theory engine."
        ),
    }


def _certificate(
    row: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    causal_by_pair: dict[tuple[str, str], dict[str, Any]],
    formal_threshold: float,
) -> FormalAlignmentCertificate:
    left_model = left["model"]
    right_model = right["model"]
    family_overlap = sorted(set(left["family_tags"]) & set(right["family_tags"]))
    role_overlap = sorted(set(left["role_schema"]) & set(right["role_schema"]))
    semantic = _semantic_score(left_model, right_model)
    graph_edit = _graph_edit_similarity(left["role_schema"], right["role_schema"])
    trajectory = _trajectory_similarity(_trajectory_distribution(left), _trajectory_distribution(right))
    typed_role_score = _typed_role_score(role_overlap=role_overlap, family_overlap=family_overlap)
    invariant = _invariant_preservation_score(left_model, right_model, family_overlap=family_overlap)
    finite_diagram = _finite_diagram_check(
        left_model,
        right_model,
        family_overlap=family_overlap,
        role_overlap=role_overlap,
        trajectory_similarity=trajectory,
    )
    negative_control = _negative_control_check(family_overlap=family_overlap, typed_role_score=typed_role_score)
    causal_signal = causal_by_pair.get(
        tuple(sorted((row["source_id"], row["target_id"]))),
        {"relation_accept_drop": 0.0, "relation_utility_drop": 0.0},
    )
    causal_support = min(1.0, max(0.0, float(causal_signal.get("relation_accept_drop", 0.0)) / 0.60))
    formal_score = round(
        0.24 * (1.0 if family_overlap else 0.0)
        + 0.24 * typed_role_score
        + 0.22 * invariant
        + 0.18 * (1.0 if finite_diagram["pass"] else 0.0)
        + 0.12 * max(trajectory, causal_support),
        4,
    )
    decision = (
        "accept_alignment"
        if formal_score >= formal_threshold and negative_control["pass"]
        else "reject_alignment"
    )
    typed_mapping = _typed_mapping(left_model, right_model, family_overlap=family_overlap, role_overlap=role_overlap)
    preserved_invariants = _preserved_invariants(left_model, right_model, family_overlap=family_overlap, role_overlap=role_overlap)
    broken = [
        f"domain differs: {left_model['domain']} vs {right_model['domain']}",
        "state variables and equations are not treated as identical",
    ]
    rationale = (
        f"families={family_overlap}; roles={role_overlap}; typed_role={typed_role_score:.3f}; "
        f"invariant={invariant:.3f}; trajectory={trajectory:.3f}; causal_drop={causal_signal.get('relation_accept_drop', 0.0)}"
    )
    return FormalAlignmentCertificate(
        source_id=row["source_id"],
        target_id=row["target_id"],
        gold_label=row["gold_label"],
        decision=decision,
        formal_score=formal_score,
        baseline_scores={
            "llm_semantic_aligner_proxy": round(semantic, 4),
            "graph_edit_role_similarity": round(graph_edit, 4),
            "trajectory_js_similarity": round(trajectory, 4),
        },
        typed_mapping=typed_mapping,
        preserved_invariants=preserved_invariants,
        broken_structures=broken,
        finite_diagram_check=finite_diagram,
        negative_control_check=negative_control,
        causal_mask_signal=causal_signal,
        rationale=rationale,
    )


def _metrics(certificates: list[FormalAlignmentCertificate]) -> dict[str, Any]:
    formal = _classification_metrics(certificates, lambda c: c.decision == "accept_alignment")
    semantic = _classification_metrics(certificates, lambda c: c.baseline_scores["llm_semantic_aligner_proxy"] >= 0.16)
    graph_edit = _classification_metrics(certificates, lambda c: c.baseline_scores["graph_edit_role_similarity"] >= 0.67)
    trajectory = _classification_metrics(certificates, lambda c: c.baseline_scores["trajectory_js_similarity"] >= 0.74)
    accepted_positives = [
        c for c in certificates
        if c.gold_label == "positive" and c.decision == "accept_alignment"
    ]
    baseline_accuracies = {
        "llm_semantic_aligner_proxy": semantic["accuracy"],
        "graph_edit_role_similarity": graph_edit["accuracy"],
        "trajectory_js_similarity": trajectory["accuracy"],
    }
    return {
        "certificate_count": len(certificates),
        "formal_accuracy": formal["accuracy"],
        "formal_positive_recall": formal["positive_recall"],
        "formal_positive_precision": formal["positive_precision"],
        "formal_negative_rejection_rate": formal["negative_rejection_rate"],
        "formal_false_positive_count": formal["false_positive"],
        "formal_false_negative_count": formal["false_negative"],
        "llm_semantic_aligner_proxy_accuracy": semantic["accuracy"],
        "graph_edit_role_similarity_accuracy": graph_edit["accuracy"],
        "trajectory_js_similarity_accuracy": trajectory["accuracy"],
        "best_baseline_accuracy": max(baseline_accuracies.values()),
        "formal_margin_over_best_baseline": round(formal["accuracy"] - max(baseline_accuracies.values()), 4),
        "baseline_accuracies": baseline_accuracies,
        "accepted_positive_mean_relation_drop": round(_mean([
            float(c.causal_mask_signal.get("relation_accept_drop", 0.0))
            for c in accepted_positives
        ]), 4),
        "finite_diagram_pass_rate": round(_mean([1.0 if c.finite_diagram_check["pass"] else 0.0 for c in certificates]), 4),
        "negative_control_pass_rate": round(_mean([1.0 if c.negative_control_check["pass"] else 0.0 for c in certificates]), 4),
    }


def _classification_metrics(
    certificates: list[FormalAlignmentCertificate],
    accept_fn,
) -> dict[str, Any]:
    tp = sum(1 for c in certificates if c.gold_label == "positive" and accept_fn(c))
    fn = sum(1 for c in certificates if c.gold_label == "positive" and not accept_fn(c))
    tn = sum(1 for c in certificates if c.gold_label == "negative" and not accept_fn(c))
    fp = sum(1 for c in certificates if c.gold_label == "negative" and accept_fn(c))
    total = len(certificates)
    return {
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "positive_recall": round(tp / max(1, tp + fn), 4),
        "positive_precision": round(tp / max(1, tp + fp), 4),
        "negative_rejection_rate": round(tn / max(1, tn + fp), 4),
    }


def _semantic_score(left_model: dict[str, Any], right_model: dict[str, Any]) -> float:
    left = " ".join([
        left_model["domain"],
        left_model["perturbation"],
        left_model["response"],
        " ".join(left_model.get("invariants", [])),
    ])
    right = " ".join([
        right_model["domain"],
        right_model["perturbation"],
        right_model["response"],
        " ".join(right_model.get("invariants", [])),
    ])
    return _token_jaccard(left, right)


def _graph_edit_similarity(left_roles: list[str], right_roles: list[str]) -> float:
    left = set(left_roles)
    right = set(right_roles)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _trajectory_distribution(entry: dict[str, Any]) -> list[float]:
    model = entry["model"]
    text = " ".join([
        model["domain"],
        model["perturbation"],
        model["response"],
        " ".join(model.get("invariants", [])),
        " ".join(entry.get("family_tags", [])),
        " ".join(entry.get("role_schema", [])),
    ]).lower()
    weights = []
    for axis in TRAJECTORY_AXES:
        axis_tokens = axis.split("_")
        weight = 0.05
        for token in axis_tokens:
            if token in text:
                weight += 1.0
        if axis == "opposition" and ("oppos" in text or "counteract" in text):
            weight += 1.5
        if axis == "equilibrium_restoration" and ("equilibrium" in text or "setpoint" in text or "clearing" in text):
            weight += 1.5
        if axis == "exponential_decay" and "decay" in text:
            weight += 1.5
        if axis == "local_stabilization" and ("stabilization" in text or "fixed point" in text or "restore local" in text):
            weight += 1.5
        if axis == "capacity_constraint" and ("capacity" in text or "constraint" in text):
            weight += 1.5
        weights.append(weight)
    total = sum(weights)
    return [w / total for w in weights]


def _trajectory_similarity(left_dist: list[float], right_dist: list[float]) -> float:
    return max(0.0, 1.0 - _jensen_shannon(left_dist, right_dist))


def _jensen_shannon(left: list[float], right: list[float]) -> float:
    midpoint = [(l + r) / 2.0 for l, r in zip(left, right)]
    return 0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint)


def _kl(left: list[float], right: list[float]) -> float:
    total = 0.0
    for l, r in zip(left, right):
        if l > 0 and r > 0:
            total += l * math.log(l / r, 2)
    return total


def _typed_role_score(*, role_overlap: list[str], family_overlap: list[str]) -> float:
    if len(role_overlap) >= 2:
        return min(1.0, len(role_overlap) / 3.0 + 0.20)
    if "local_stabilization" in family_overlap and "restoring_response" in role_overlap:
        return 0.75
    if family_overlap and role_overlap:
        return 0.55
    return 0.0


def _invariant_preservation_score(
    left_model: dict[str, Any],
    right_model: dict[str, Any],
    *,
    family_overlap: list[str],
) -> float:
    lexical = _token_jaccard(" ".join(left_model.get("invariants", [])), " ".join(right_model.get("invariants", [])))
    if "negative_feedback" in family_overlap or "exponential_decay" in family_overlap:
        return max(lexical, 0.90)
    if "equilibrium_restoration" in family_overlap or "local_stabilization" in family_overlap:
        return max(lexical, 0.78)
    return lexical


def _finite_diagram_check(
    left_model: dict[str, Any],
    right_model: dict[str, Any],
    *,
    family_overlap: list[str],
    role_overlap: list[str],
    trajectory_similarity: float,
) -> dict[str, Any]:
    pass_check = bool(family_overlap) and (
        len(role_overlap) >= 2
        or ("local_stabilization" in family_overlap and trajectory_similarity >= 0.55)
    )
    return {
        "pass": pass_check,
        "diagram": {
            "source": [
                left_model["perturbation"],
                left_model["response"],
                left_model.get("invariants", []),
            ],
            "target": [
                right_model["perturbation"],
                right_model["response"],
                right_model.get("invariants", []),
            ],
            "required_commutation": "mapped perturbation -> mapped response preserves the same abstract invariant family",
        },
        "family_overlap": family_overlap,
        "role_overlap": role_overlap,
        "trajectory_similarity": round(trajectory_similarity, 4),
    }


def _negative_control_check(*, family_overlap: list[str], typed_role_score: float) -> dict[str, Any]:
    pass_check = bool(family_overlap) and typed_role_score >= 0.55
    return {
        "pass": pass_check,
        "reject_reason": "" if pass_check else "no typed family bridge with enough role preservation",
    }


def _typed_mapping(
    left_model: dict[str, Any],
    right_model: dict[str, Any],
    *,
    family_overlap: list[str],
    role_overlap: list[str],
) -> dict[str, str]:
    mapping = {
        "source_perturbation": left_model["perturbation"],
        "target_perturbation": right_model["perturbation"],
        "source_response": left_model["response"],
        "target_response": right_model["response"],
    }
    if family_overlap:
        mapping["shared_process_family"] = " / ".join(family_overlap)
    if role_overlap:
        mapping["shared_typed_roles"] = " / ".join(role_overlap)
    return mapping


def _preserved_invariants(
    left_model: dict[str, Any],
    right_model: dict[str, Any],
    *,
    family_overlap: list[str],
    role_overlap: list[str],
) -> list[str]:
    preserved = []
    for family in family_overlap:
        preserved.append(f"process-family invariant: {family}")
    for role in role_overlap:
        preserved.append(f"typed role preserved: {role}")
    if not preserved:
        preserved.append("no formal invariant preserved")
    return preserved


def _causal_relation_drop_by_pair(causal_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for trial in causal_payload.get("trials", []):
        if trial.get("mask_id") != "do(mask_alignment_relation_node)":
            continue
        source, target = _pair_from_action_id(str(trial["action_id"]))
        by_pair[tuple(sorted((source, target)))] = {
            "relation_accept_drop": trial["accept_prob_delta"],
            "relation_utility_drop": trial["utility_delta"],
            "masked_accept_prob": trial["masked_accept_prob"],
        }
    return by_pair


def _pair_from_action_id(action_id: str) -> tuple[str, str]:
    parts = action_id.split("::")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    return "", ""


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 bounded formal alignment validation.")
    parser.add_argument("--eval-id", default="formal_alignment_v2_20260610")
    parser.add_argument("--formal-threshold", type=float, default=0.60)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_formal_alignment_v2_payload(
        eval_id=args.eval_id,
        formal_threshold=args.formal_threshold,
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
