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

from .manifest_logger import redact_secrets
from .schema import stable_id


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
    predictions = [
        _predict_leave_one_out(row=row, train_rows=[other for other in rows if other is not row])
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
        "route_group_count": len(route_stats),
        "component_group_count": len(component_stats),
        "domain_group_count": len(domain_stats),
        "residual_group_count": len(residual_stats),
        "leave_one_out_metrics": _prediction_metrics(predictions),
        "route_stats": route_stats,
        "component_stats": component_stats,
        "domain_stats": domain_stats,
        "residual_stats": residual_stats,
        "policy_update_count": len(policy_updates),
        "policy_updates": policy_updates,
        "predictions": predictions,
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
    probability = (wins + 1.0) / (count + 2.0) if count else 0.5
    label = 1.0 if row.get("outcome") == "win" else 0.0
    predicted_outcome = "win" if probability >= 0.5 else "loss"
    return {
        "prediction_id": stable_id("trace_pred", row.get("row_id"), selected_key),
        "row_id": row.get("row_id"),
        "problem_id": row.get("problem_id"),
        "domain": row.get("domain"),
        "bypass_route": row.get("bypass_route"),
        "selected_level": selected_level,
        "selected_key": selected_key,
        "support_count": count,
        "support_win_count": wins,
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
        deltas = [float(row["score_delta"]) for row in bucket if row.get("score_delta") is not None]
        residual_counts = Counter(str(row.get("residual_type") or "unknown") for row in bucket)
        stats.append({
            "key": key,
            "count": count,
            "win_count": wins,
            "loss_count": losses,
            "win_rate": round(wins / count, 4) if count else 0.0,
            "smoothed_win_probability": round((wins + 1.0) / (count + 2.0), 4),
            "mean_score_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "residual_type_counts": dict(residual_counts),
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
    accuracy = sum(1 for row in predictions if row["predicted_outcome"] == row["observed_outcome"]) / len(predictions)
    return {
        "prediction_count": len(predictions),
        "brier_score": round(brier, 4),
        "mean_absolute_error": round(mae, 4),
        "accuracy_at_half": round(accuracy, 4),
        "prediction_level_counts": dict(Counter(row["selected_level"] for row in predictions)),
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
    print(text)


if __name__ == "__main__":
    main()
