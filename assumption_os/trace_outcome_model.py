"""Calibrate execution policy from trace-to-outcome datasets.

The proposal world model predicts whether candidate assumptions should be
tested or promoted.  This module is the paired execution world model: it learns
which runtime routes/components are currently producing judged wins or losses
and emits bounded policy updates from residual clusters in the trace dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .graph_memory import JsonlGraphStore
from .manifest_logger import redact_secrets
from .proposals import CandidateProposal, ProposalType
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    TrialManifest,
    TrialStatus,
    stable_id,
)


def build_trace_outcome_model_payload(
    *,
    trace_dataset_payload: dict[str, Any],
    eval_id: str,
    min_policy_group_size: int = 2,
) -> dict:
    rows = [
        row for row in trace_dataset_payload.get("rows", [])
        if row.get("trainable") and row.get("outcome") in {"win", "loss"}
    ]
    weighted_trainable_row_count = sum(_row_weight(row) for row in rows)
    predictions = [
        _predict_leave_one_out(row=row, train_rows=[other for other in rows if other is not row])
        for row in rows
    ]
    feature_predictions = [
        _predict_feature_leave_one_out(row=row, train_rows=[other for other in rows if other is not row])
        for row in rows
    ]
    route_stats = _group_stats(rows, key_fn=_route_key)
    component_stats = _group_stats(rows, key_fn=_component_key)
    domain_stats = _group_stats(rows, key_fn=lambda row: f"domain={row.get('domain') or 'unknown'}")
    residual_stats = _residual_stats(rows)
    policy_updates = _policy_updates(
        eval_id=eval_id,
        route_stats=route_stats,
        residual_stats=residual_stats,
        min_group_size=min_policy_group_size,
    )
    payload = {
        "eval_id": eval_id,
        "source_trace_dataset_eval_id": trace_dataset_payload.get("eval_id"),
        "trainable_row_count": len(rows),
        "weighted_trainable_row_count": round(weighted_trainable_row_count, 4),
        "trace_source_counts": dict(Counter(_trace_source(row) for row in rows)),
        "trace_source_weighted_counts": _trace_source_weighted_counts(rows),
        "route_group_count": len(route_stats),
        "component_group_count": len(component_stats),
        "domain_group_count": len(domain_stats),
        "residual_group_count": len(residual_stats),
        "leave_one_out_metrics": _prediction_metrics(predictions),
        "feature_leave_one_out_metrics": _prediction_metrics(feature_predictions),
        "feature_schema": _trace_feature_schema(rows),
        "route_stats": route_stats,
        "component_stats": component_stats,
        "domain_stats": domain_stats,
        "residual_stats": residual_stats,
        "policy_update_count": len(policy_updates),
        "policy_updates": policy_updates,
        "predictions": predictions,
        "feature_predictions": feature_predictions,
    }
    clean = redact_secrets(payload)
    clean["secret_leak_detected"] = _contains_secret(clean)
    return clean


def build_trace_policy_proposal_payload(
    *,
    store: JsonlGraphStore,
    trace_outcome_payload: dict[str, Any],
    eval_id: str,
    parent_surface_key: str = "domain_retrieval_policy",
) -> dict:
    """Convert trace policy updates into reviewable candidate proposals."""

    parent = _find_surface_parent(store, parent_surface_key)
    proposals: list[CandidateProposal] = []
    if parent is not None:
        proposals = [
            _proposal_from_policy_update(parent=parent, update=update, eval_id=eval_id)
            for update in trace_outcome_payload.get("policy_updates", [])
        ]
    payload = {
        "eval_id": eval_id,
        "source_trace_outcome_eval_id": trace_outcome_payload.get("eval_id"),
        "parent_surface_key": parent_surface_key,
        "parent_node_id": parent.id if parent else None,
        "proposal_count": len(proposals),
        "proposal_counts": dict(Counter(p.proposal_type.value for p in proposals)),
        "decision_counts": dict(Counter(p.source_action.get("decision") for p in proposals)),
        "proposals": [p.to_dict() for p in proposals],
    }
    clean = redact_secrets(payload)
    clean["secret_leak_detected"] = _contains_secret(clean)
    return clean


def _predict_leave_one_out(*, row: dict[str, Any], train_rows: list[dict[str, Any]]) -> dict:
    candidates = [
        ("route_component", _route_component_key(row)),
        ("route", _route_key(row)),
        ("domain", f"domain={row.get('domain') or 'unknown'}"),
        ("global", "global"),
    ]
    selected_level = "global"
    selected_key = "global"
    train_bucket = train_rows
    for level, key in candidates:
        if level == "global":
            bucket = train_rows
        else:
            bucket = [other for other in train_rows if _key_for_level(other, level) == key]
        if bucket:
            selected_level = level
            selected_key = key
            train_bucket = bucket
            break
    wins = sum(1 for other in train_bucket if other.get("outcome") == "win")
    count = len(train_bucket)
    support_weight = sum(_row_weight(other) for other in train_bucket)
    weighted_wins = sum(_row_weight(other) for other in train_bucket if other.get("outcome") == "win")
    probability = (weighted_wins + 1.0) / (support_weight + 2.0) if support_weight else 0.5
    label = 1.0 if row.get("outcome") == "win" else 0.0
    predicted_outcome = "win" if probability >= 0.5 else "loss"
    row_weight = _row_weight(row)
    return {
        "prediction_id": stable_id("trace_pred", row.get("row_id"), selected_key),
        "row_id": row.get("row_id"),
        "problem_id": row.get("problem_id"),
        "domain": row.get("domain"),
        "bypass_route": row.get("bypass_route"),
        "trace_source": _trace_source(row),
        "row_weight": row_weight,
        "selected_level": selected_level,
        "selected_key": selected_key,
        "support_count": count,
        "support_win_count": wins,
        "support_weight": round(support_weight, 4),
        "support_weighted_win_count": round(weighted_wins, 4),
        "predicted_win_probability": round(probability, 4),
        "predicted_outcome": predicted_outcome,
        "observed_outcome": row.get("outcome"),
        "label": label,
        "absolute_error": round(abs(probability - label), 4),
        "brier": round((probability - label) ** 2, 4),
        "residual_type": row.get("residual_type"),
    }


def _predict_feature_leave_one_out(*, row: dict[str, Any], train_rows: list[dict[str, Any]]) -> dict:
    row_features = _trace_feature_keys(row)
    global_weight = sum(_row_weight(other) for other in train_rows)
    global_weighted_wins = sum(_row_weight(other) for other in train_rows if other.get("outcome") == "win")
    global_probability = (
        (global_weighted_wins + 1.0) / (global_weight + 2.0)
        if global_weight else 0.5
    )
    terms = [{
        "feature": "global_prior",
        "support_count": len(train_rows),
        "support_weight": round(global_weight, 4),
        "weighted_win_count": round(global_weighted_wins, 4),
        "blend_weight": 2.0,
        "win_probability": round(global_probability, 4),
    }]
    train_feature_sets = [(other, _trace_feature_keys(other)) for other in train_rows]
    for feature in row_features:
        bucket = [other for other, feature_set in train_feature_sets if feature in feature_set]
        support_weight = sum(_row_weight(other) for other in bucket)
        if support_weight <= 0.0:
            continue
        weighted_wins = sum(_row_weight(other) for other in bucket if other.get("outcome") == "win")
        probability = (weighted_wins + 1.0) / (support_weight + 2.0)
        information_shift = abs(probability - global_probability)
        blend_weight = min(8.0, support_weight) * max(0.05, 8.0 * information_shift * information_shift)
        terms.append({
            "feature": feature,
            "support_count": len(bucket),
            "support_weight": round(support_weight, 4),
            "weighted_win_count": round(weighted_wins, 4),
            "information_shift": round(information_shift, 4),
            "blend_weight": round(blend_weight, 4),
            "win_probability": round(probability, 4),
        })
    total_blend_weight = sum(float(term["blend_weight"]) for term in terms)
    probability = (
        sum(float(term["blend_weight"]) * float(term["win_probability"]) for term in terms) / total_blend_weight
        if total_blend_weight else 0.5
    )
    label = 1.0 if row.get("outcome") == "win" else 0.0
    predicted_outcome = "win" if probability >= 0.5 else "loss"
    top_features = sorted(
        [term for term in terms if term["feature"] != "global_prior"],
        key=lambda term: (-float(term["blend_weight"]), str(term["feature"])),
    )[:8]
    return {
        "prediction_id": stable_id("trace_feature_pred", row.get("row_id")),
        "row_id": row.get("row_id"),
        "problem_id": row.get("problem_id"),
        "domain": row.get("domain"),
        "bypass_route": row.get("bypass_route"),
        "trace_source": _trace_source(row),
        "row_weight": _row_weight(row),
        "selected_level": "feature_blend",
        "selected_key": "feature_blend",
        "feature_count": len(row_features),
        "matched_feature_count": len(terms) - 1,
        "support_count": len(train_rows),
        "support_weight": round(global_weight, 4),
        "support_weighted_win_count": round(global_weighted_wins, 4),
        "top_features": top_features,
        "predicted_win_probability": round(probability, 4),
        "predicted_outcome": predicted_outcome,
        "observed_outcome": row.get("outcome"),
        "label": label,
        "absolute_error": round(abs(probability - label), 4),
        "brier": round((probability - label) ** 2, 4),
        "residual_type": row.get("residual_type"),
    }


def _group_stats(rows: list[dict[str, Any]], *, key_fn: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    stats = []
    for key, bucket in sorted(groups.items()):
        count = len(bucket)
        wins = sum(1 for row in bucket if row.get("outcome") == "win")
        losses = sum(1 for row in bucket if row.get("outcome") == "loss")
        weighted_count = sum(_row_weight(row) for row in bucket)
        weighted_wins = sum(_row_weight(row) for row in bucket if row.get("outcome") == "win")
        weighted_losses = sum(_row_weight(row) for row in bucket if row.get("outcome") == "loss")
        deltas = [float(row["score_delta"]) for row in bucket if row.get("score_delta") is not None]
        weighted_deltas = [
            (float(row["score_delta"]), _row_weight(row))
            for row in bucket
            if row.get("score_delta") is not None
        ]
        weighted_delta_weight = sum(weight for _, weight in weighted_deltas)
        residual_counts = Counter(str(row.get("residual_type") or "unknown") for row in bucket)
        stats.append({
            "key": key,
            "count": count,
            "weighted_count": round(weighted_count, 4),
            "win_count": wins,
            "loss_count": losses,
            "weighted_win_count": round(weighted_wins, 4),
            "weighted_loss_count": round(weighted_losses, 4),
            "win_rate": round(wins / count, 4) if count else 0.0,
            "smoothed_win_probability": round((wins + 1.0) / (count + 2.0), 4),
            "weighted_win_rate": round(weighted_wins / weighted_count, 4) if weighted_count else 0.0,
            "weighted_smoothed_win_probability": round((weighted_wins + 1.0) / (weighted_count + 2.0), 4)
            if weighted_count else 0.5,
            "mean_score_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "weighted_mean_score_delta": round(
                sum(delta * weight for delta, weight in weighted_deltas) / weighted_delta_weight,
                4,
            ) if weighted_delta_weight else None,
            "residual_type_counts": dict(residual_counts),
            "trace_source_counts": dict(Counter(_trace_source(row) for row in bucket)),
            "trace_source_weighted_counts": _trace_source_weighted_counts(bucket),
            "problem_ids": [str(row.get("problem_id")) for row in bucket],
        })
    return sorted(stats, key=lambda row: (-row["count"], row["key"]))


def _residual_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loss_rows = [row for row in rows if row.get("outcome") == "loss"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in loss_rows:
        groups[(
            str(row.get("residual_type") or "unknown"),
            str(row.get("bypass_route") or "no_route"),
            ",".join(row.get("components") or ["unknown"]),
        )].append(row)
    out = []
    for (rtype, route, components), bucket in sorted(groups.items()):
        out.append({
            "key": f"residual={rtype}|route={route}|components={components}",
            "residual_type": rtype,
            "bypass_route": route,
            "components": components.split(",") if components else [],
            "count": len(bucket),
            "weighted_count": round(sum(_row_weight(row) for row in bucket), 4),
            "problem_ids": [str(row.get("problem_id")) for row in bucket],
            "residual_previews": [_preview(row.get("residual"), limit=140) for row in bucket[:5]],
        })
    return sorted(out, key=lambda row: (-row["count"], row["key"]))


def _policy_updates(
    *,
    eval_id: str,
    route_stats: list[dict[str, Any]],
    residual_stats: list[dict[str, Any]],
    min_group_size: int,
) -> list[dict[str, Any]]:
    residual_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for residual in residual_stats:
        residual_by_route[residual["bypass_route"]].append(residual)
    updates = []
    for route in route_stats:
        route_name = _route_name_from_key(route["key"])
        if route["count"] < min_group_size:
            continue
        residuals = residual_by_route.get(route_name, [])
        if route["loss_count"] > 0:
            decision = "repair_before_scaling" if route["win_rate"] < 0.5 else "keep_with_targeted_repair"
            updates.append({
                "policy_update_id": stable_id("trace_policy", eval_id, route["key"], decision),
                "decision": decision,
                "scope": route["key"],
                "expected_effect": "Reduce route-specific losses while preserving the observed win rate.",
                "residual_type_counts": route["residual_type_counts"],
                "trigger_problem_ids": route["problem_ids"],
                "residual_groups": residuals,
                "verification_plan": [
                    "rerun heldout trigger rows for the route",
                    "include outside-control rows from other routes",
                    "reject if repair lowers route win rate or increases control losses",
                ],
            })
        elif route["win_count"] >= min_group_size:
            updates.append({
                "policy_update_id": stable_id("trace_policy", eval_id, route["key"], "reinforce"),
                "decision": "reinforce_route_prior",
                "scope": route["key"],
                "expected_effect": "Preserve a route with clean observed wins as a prior for similar cached traces.",
                "residual_type_counts": route["residual_type_counts"],
                "trigger_problem_ids": route["problem_ids"],
                "residual_groups": [],
                "verification_plan": [
                    "continue sampling outside controls",
                    "demote if new losses cluster under the same route",
                ],
            })
    return updates


def _prediction_metrics(predictions: list[dict[str, Any]]) -> dict:
    if not predictions:
        return {
            "prediction_count": 0,
            "brier_score": None,
            "mean_absolute_error": None,
            "accuracy_at_half": None,
        }
    brier = sum(float(row["brier"]) for row in predictions) / len(predictions)
    mae = sum(float(row["absolute_error"]) for row in predictions) / len(predictions)
    total_weight = sum(float(row.get("row_weight") or 0.0) for row in predictions)
    weighted_brier = (
        sum(float(row["brier"]) * float(row.get("row_weight") or 0.0) for row in predictions) / total_weight
        if total_weight else None
    )
    weighted_mae = (
        sum(float(row["absolute_error"]) * float(row.get("row_weight") or 0.0) for row in predictions) / total_weight
        if total_weight else None
    )
    accuracy = sum(1 for row in predictions if row["predicted_outcome"] == row["observed_outcome"]) / len(predictions)
    weighted_accuracy = (
        sum(
            float(row.get("row_weight") or 0.0)
            for row in predictions
            if row["predicted_outcome"] == row["observed_outcome"]
        ) / total_weight
        if total_weight else None
    )
    return {
        "prediction_count": len(predictions),
        "brier_score": round(brier, 4),
        "mean_absolute_error": round(mae, 4),
        "accuracy_at_half": round(accuracy, 4),
        "weighted_prediction_count": round(total_weight, 4),
        "weighted_brier_score": round(weighted_brier, 4) if weighted_brier is not None else None,
        "weighted_mean_absolute_error": round(weighted_mae, 4) if weighted_mae is not None else None,
        "weighted_accuracy_at_half": round(weighted_accuracy, 4) if weighted_accuracy is not None else None,
        "prediction_level_counts": dict(Counter(row["selected_level"] for row in predictions)),
    }


def _trace_source(row: dict[str, Any]) -> str:
    source = row.get("trace_source") or row.get("source_kind")
    if source:
        return str(source)
    if row.get("first_party_trace"):
        return "first_party_runtime"
    return "unspecified"


def _row_weight(row: dict[str, Any]) -> float:
    source = _trace_source(row).lower()
    if "artifact" in source or "replay" in source:
        return 0.5
    return 1.0


def _trace_source_weighted_counts(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for row in rows:
        counts[_trace_source(row)] += _row_weight(row)
    return {key: round(value, 4) for key, value in sorted(counts.items())}


def _trace_feature_keys(row: dict[str, Any]) -> set[str]:
    features = row.get("features") or {}
    out: set[str] = set()
    for key in (
        "domain",
        "difficulty",
        "frame",
        "bypass_route",
        "residual_type",
        "intervention_variant",
        "baseline_variant",
        "judgment_pair",
    ):
        value = row.get(key)
        if value is None:
            value = features.get(key)
        if value is not None:
            out.add(f"{key}={value}")
    out.add(f"trace_source={_trace_source(row)}")
    for component in row.get("components") or []:
        out.add(f"component={component}")
    for event_name in (row.get("event_counts") or {}).keys():
        out.add(f"event={event_name}")
    for component_name, count in (row.get("component_counts") or {}).items():
        out.add(f"component_count:{component_name}={count}")
    for key in ("active_assumption_count", "gold_hit", "trace_event_count"):
        if key in features:
            out.add(f"{key}={features[key]}")
    active_assumption_ids = row.get("activated_assumption_ids") or []
    out.add("active_assumption_count_bucket=0" if not active_assumption_ids else "active_assumption_count_bucket=1plus")
    for assumption_id in active_assumption_ids[:8]:
        out.add(f"active_assumption={assumption_id}")
    for prompt_kind in row.get("prompt_kinds") or []:
        out.add(f"prompt_kind={prompt_kind}")
    return out


def _trace_feature_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_counts = Counter(feature for row in rows for feature in _trace_feature_keys(row))
    family_counts = Counter(feature.split("=", 1)[0].split(":", 1)[0] for feature in feature_counts)
    return {
        "feature_count": len(feature_counts),
        "feature_family_counts": dict(sorted(family_counts.items())),
        "top_features": [
            {"feature": feature, "count": count}
            for feature, count in feature_counts.most_common(20)
        ],
    }


def _key_for_level(row: dict[str, Any], level: str) -> str:
    if level == "route_component":
        return _route_component_key(row)
    if level == "route":
        return _route_key(row)
    if level == "domain":
        return f"domain={row.get('domain') or 'unknown'}"
    return "global"


def _route_component_key(row: dict[str, Any]) -> str:
    return f"{_route_key(row)}|components={','.join(row.get('components') or ['none'])}"


def _route_key(row: dict[str, Any]) -> str:
    return f"route={row.get('bypass_route') or 'no_route'}"


def _component_key(row: dict[str, Any]) -> str:
    return f"components={','.join(row.get('components') or ['none'])}"


def _route_name_from_key(key: str) -> str:
    if key.startswith("route="):
        return key.split("=", 1)[1]
    return key


def _contains_secret(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return redact_secrets(text) != text


def _preview(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _find_surface_parent(store: JsonlGraphStore, surface_key: str) -> AssumptionNode | None:
    for node in store.nodes.values():
        if (node.payload or {}).get("surface_key") == surface_key:
            return node
    for node in store.nodes.values():
        if surface_key in node.tags:
            return node
    return None


def _proposal_from_policy_update(
    *,
    parent: AssumptionNode,
    update: dict[str, Any],
    eval_id: str,
) -> CandidateProposal:
    decision = str(update.get("decision") or "trace_policy_update")
    scope = str(update.get("scope") or "unknown_scope")
    route = _route_name_from_key(scope)
    cid = stable_id("cand", eval_id, update.get("policy_update_id"), route, decision)
    repair = decision in {"keep_with_targeted_repair", "repair_before_scaling"}
    claim = (
        f"For route {route}, keep the route but add a targeted repair verifier before scaling."
        if decision == "keep_with_targeted_repair"
        else f"For route {route}, repair the route policy before further scaling."
        if decision == "repair_before_scaling"
        else f"For route {route}, reinforce the route prior while monitoring new residuals."
    )
    candidate = AssumptionNode(
        id=cid,
        type=AssumptionType.RETRIEVAL,
        kind=HypothesisKind.RETRIEVAL_POLICY,
        claim=claim,
        context_conditions=[
            f"trace_policy_decision={decision}",
            f"scope={scope}",
            *[f"trigger={pid}" for pid in update.get("trigger_problem_ids", [])[:6]],
        ],
        predicted_effects=[
            update.get("expected_effect") or "Improve route-conditioned execution outcomes.",
            "convert trace outcome evidence into a falsifiable route policy before graph mutation",
        ],
        risk_predictions=[
            "route repair may overfit the small trace slice",
            "reinforcing a route without outside controls can hide future regressions",
        ],
        verifiers=[
            "trace_outcome_leave_one_out",
            "heldout_route_ablation",
            "outside_control_harm_check",
        ],
        confidence=0.46 if repair else 0.5,
        metaproductivity=0.08 if repair else 0.05,
        status="candidate",
        tags=[
            "candidate",
            "trace_policy",
            decision,
            route,
        ],
        payload={
            "source": "trace_outcome_model",
            "policy_update": update,
            "activation": {
                "problem_ids": update.get("trigger_problem_ids", []),
                "keywords": [route, decision],
                "min_keyword_hits": 1,
            },
            "validation_plan": {
                "trigger_problem_ids": update.get("trigger_problem_ids", []),
                "control_policy": "sample outside-control rows from other routes before promotion",
                "acceptance": "route trigger utility improves without outside-control harm",
            },
        },
    )
    edge = AssumptionEdge(
        source=parent.id,
        target=cid,
        type=EdgeType.SPECIALIZES,
        weight=0.62 if repair else 0.55,
        payload={"source": "trace_outcome_model", "decision": decision, "scope": scope},
    )
    manifest = TrialManifest(
        problem_id=f"trace_policy::{update.get('policy_update_id')}",
        action_type="trace_policy_proposal",
        component="trace_outcome_model",
        assumption=claim,
        why_selected=f"Trace outcome model emitted {decision} for {scope}.",
        expected_effect=update.get("expected_effect") or "Improve route-conditioned outcomes.",
        assumption_ids=[parent.id, cid],
        verifier="trace_policy_proposal_gate",
        verification_plan=json.dumps(update.get("verification_plan", []), ensure_ascii=False, sort_keys=True),
        rollback_condition="Reject if heldout trigger rows fail or outside-control rows regress.",
        status=TrialStatus.PENDING,
        artifacts={"policy_update": update, "candidate_node": candidate.to_dict()},
        metadata={"eval_id": eval_id, "decision": decision, "scope": scope},
        trial_id=stable_id("trial", eval_id, update.get("policy_update_id"), cid),
    )
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, update.get("policy_update_id"), cid),
        proposal_type=ProposalType.ASSUMPTION_REVISION,
        parent_node_id=parent.id,
        candidate_node=candidate.to_dict(),
        edges=[edge.to_dict()],
        manifest=manifest.to_dict(),
        rationale=f"Trace policy update {decision} for {scope}.",
        priority=0.72 if repair else 0.48,
        source_action=update,
    )


def _resolve(root: Path, path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--trace-dataset", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--min-policy-group-size", type=int, default=2)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--proposals-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    trace_dataset_path = _resolve(root, args.trace_dataset)
    payload = build_trace_outcome_model_payload(
        trace_dataset_payload=json.loads(trace_dataset_path.read_text(encoding="utf-8")),
        eval_id=args.eval_id,
        min_policy_group_size=args.min_policy_group_size,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.proposals_out:
        if not args.graph_dir:
            raise SystemExit("--proposals-out requires --graph-dir")
        proposal_payload = build_trace_policy_proposal_payload(
            store=JsonlGraphStore(_resolve(root, args.graph_dir)),
            trace_outcome_payload=payload,
            eval_id=f"{args.eval_id}_proposals",
        )
        out = _resolve(root, args.proposals_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(proposal_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
