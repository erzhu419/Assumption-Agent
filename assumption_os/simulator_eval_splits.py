"""Split-discipline evaluation for the simulator transition dataset.

B1 freezes the transition schema.  B2 asks whether simulator-style predictors
generalize beyond the slice they were derived from.  This module evaluates
leave-one-out, leave-domain-out, leave-pattern-out, leave-artifact-out, and
leave-residual-family-out splits against simple baselines.  It is deliberately
conservative: a predictor can be useful for search control while still being
blocked from production promotion when heldout groups regress.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .autonomy_journal import PAPER_DIR, stable_hash
from .simulator_transition_schema import DEFAULT_DATASET_OUT, validate_transition_rows


DEFAULT_OUT = PAPER_DIR / "simulator_eval_splits_20260612.json"
PREDICTORS = [
    "feature_similarity_simulator",
    "base_rate_per_arm",
    "current_heuristic_world_model",
    "handwritten_hybrid_guard",
    "random_with_abstain",
    "always_original_v3",
    "always_run_ablation",
]
GROUP_EVALS = {
    "leave_one_out": lambda row: str(row["row_id"]),
    "leave_domain_out": lambda row: str(row["state"]["domain"]),
    "leave_pattern_out": lambda row: str(row["state"]["pattern"]),
    "leave_artifact_out": lambda row: str(row["provenance"]["artifact_id"]),
    "leave_residual_family_out": lambda row: str(row["state"]["residual_cluster"]),
}


def build_simulator_eval_splits_payload(
    *,
    root: Path,
    eval_id: str = "simulator_eval_splits_20260612",
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    dataset_path = dataset_path or DEFAULT_DATASET_OUT
    dataset_path = dataset_path if dataset_path.is_absolute() else root / dataset_path
    rows = _load_jsonl(dataset_path)
    validation = validate_transition_rows(rows)
    labels = [_label(row) for row in rows]
    split_reports = {
        split_name: _evaluate_group_split(rows=rows, group_fn=group_fn, split_name=split_name)
        for split_name, group_fn in GROUP_EVALS.items()
    }
    within_slice = _evaluate_within_slice(rows)
    leak_feature_excluded_count = sum(
        1
        for row in rows
        for feature in row.get("state", {}).get("world_model_features", [])
        if _is_leaky_feature(str(feature))
    )
    promotion = _promotion_decision(split_reports)
    metrics = {
        "row_count": len(rows),
        "valid_row_count": validation.valid_row_count,
        "label_positive_count": int(sum(labels)),
        "label_negative_count": int(len(labels) - sum(labels)),
        "global_positive_rate": round(sum(labels) / max(1, len(labels)), 4),
        "split_eval_count": len(split_reports),
        "leave_one_out_group_count": split_reports["leave_one_out"]["group_count"],
        "leave_domain_out_group_count": split_reports["leave_domain_out"]["group_count"],
        "leave_pattern_out_group_count": split_reports["leave_pattern_out"]["group_count"],
        "leave_artifact_out_group_count": split_reports["leave_artifact_out"]["group_count"],
        "leave_residual_family_out_group_count": split_reports["leave_residual_family_out"]["group_count"],
        "feature_leak_excluded_count": leak_feature_excluded_count,
        "feature_model_loo_brier": split_reports["leave_one_out"]["predictors"]["feature_similarity_simulator"]["brier"],
        "base_rate_loo_brier": split_reports["leave_one_out"]["predictors"]["base_rate_per_arm"]["brier"],
        "current_heuristic_loo_brier": split_reports["leave_one_out"]["predictors"]["current_heuristic_world_model"]["brier"],
        "feature_model_leave_domain_brier": split_reports["leave_domain_out"]["predictors"]["feature_similarity_simulator"]["brier"],
        "base_rate_leave_domain_brier": split_reports["leave_domain_out"]["predictors"]["base_rate_per_arm"]["brier"],
        "feature_model_leave_pattern_brier": split_reports["leave_pattern_out"]["predictors"]["feature_similarity_simulator"]["brier"],
        "base_rate_leave_pattern_brier": split_reports["leave_pattern_out"]["predictors"]["base_rate_per_arm"]["brier"],
        "feature_model_leave_artifact_brier": split_reports["leave_artifact_out"]["predictors"]["feature_similarity_simulator"]["brier"],
        "feature_model_abstention_rate_loo": split_reports["leave_one_out"]["predictors"]["feature_similarity_simulator"]["abstention_rate"],
        "feature_model_true_positive_block_rate_loo": split_reports["leave_one_out"]["predictors"]["feature_similarity_simulator"]["true_positive_block_rate"],
        "current_heuristic_true_positive_block_rate_loo": split_reports["leave_one_out"]["predictors"]["current_heuristic_world_model"]["true_positive_block_rate"],
        "raw_predictor_promotion_allowed": promotion["raw_predictor_promotion_allowed"],
        "feature_model_promotion_allowed": promotion["feature_model_promotion_allowed"],
        "production_simulator_replacement_allowed": False,
    }
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["row_count"] and metrics["row_count"] >= 345,
        "all_required_split_evals_present": set(split_reports) == set(GROUP_EVALS),
        "leave_one_out_available": metrics["leave_one_out_group_count"] >= 300,
        "leave_domain_out_available": metrics["leave_domain_out_group_count"] >= 5,
        "leave_pattern_out_available": metrics["leave_pattern_out_group_count"] >= 5,
        "leave_artifact_out_available": metrics["leave_artifact_out_group_count"] >= 4,
        "leave_residual_family_out_available": metrics["leave_residual_family_out_group_count"] >= 5,
        "all_predictors_reported": all(set(report["predictors"]) == set(PREDICTORS) for report in split_reports.values()),
        "brier_ece_abstention_and_tpb_present": _all_metric_keys_present(split_reports),
        "leaky_decision_features_excluded": metrics["feature_leak_excluded_count"] > 0,
        "base_rate_baseline_present": "base_rate_per_arm" in split_reports["leave_one_out"]["predictors"],
        "current_heuristic_baseline_present": (
            "current_heuristic_world_model" in split_reports["leave_one_out"]["predictors"]
        ),
        "raw_predictor_not_overpromoted": metrics["raw_predictor_promotion_allowed"] is False,
        "production_replacement_blocked": metrics["production_simulator_replacement_allowed"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_eval_splits",
        "last_three_part_ticket": "B2_simulator_eval_splits",
        "performance_validation": True,
        "validation_scope": (
            "Evaluates simulator transition predictors under leave-one-out, leave-domain-out, leave-pattern-out, "
            "leave-artifact-out, and leave-residual-family-out splits.  Reports Brier, ECE, abstention rate, "
            "and true-positive block rate for feature, base-rate, heuristic, random, and always-action baselines."
        ),
        "source": {
            "dataset_path": _display_path(root, dataset_path),
            "schema_validation_valid_row_count": validation.valid_row_count,
        },
        "within_slice": within_slice,
        "split_reports": split_reports,
        "promotion_decision": promotion,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "limitations": [
            "This is split discipline and baseline comparison, not a production simulator promotion.",
            "Aggregate-distilled transition rows are evaluated but retain source_granularity markers.",
        ],
    }


def _evaluate_within_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in rows if row["provenance"]["split"] == "train"]
    eval_rows = [row for row in rows if row["provenance"]["split"] != "train"]
    return {
        "train_count": len(train),
        "eval_count": len(eval_rows),
        "predictors": {
            predictor: _evaluate_predictions(eval_rows, [_predict(predictor, row, train) for row in eval_rows])
            for predictor in PREDICTORS
        },
    }


def _evaluate_group_split(
    *,
    rows: list[dict[str, Any]],
    group_fn: Callable[[dict[str, Any]], str],
    split_name: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_fn(row)].append(row)
    predictor_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in PREDICTORS}
    group_reports = []
    for group, eval_rows in sorted(groups.items()):
        train_rows = [row for row in rows if group_fn(row) != group]
        group_predictors = {}
        for predictor in PREDICTORS:
            predictions = [_predict(predictor, row, train_rows) for row in eval_rows]
            predictor_rows[predictor].extend(predictions)
            group_predictors[predictor] = _evaluate_predictions(eval_rows, predictions)
        group_reports.append(
            {
                "group": group,
                "eval_count": len(eval_rows),
                "train_count": len(train_rows),
                "label_positive_rate": round(sum(_label(row) for row in eval_rows) / max(1, len(eval_rows)), 4),
                "predictors": group_predictors,
            }
        )
    return {
        "split_name": split_name,
        "group_count": len(groups),
        "row_count": len(rows),
        "predictors": {
            predictor: _evaluate_predictions(rows, predictor_rows[predictor])
            for predictor in PREDICTORS
        },
        "group_reports": group_reports,
    }


def _predict(predictor: str, row: dict[str, Any], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if predictor == "feature_similarity_simulator":
        return _feature_similarity_prediction(row, train_rows)
    if predictor == "base_rate_per_arm":
        score = _base_rate_for(row, train_rows, key_fn=lambda x: x["action"]["arm"])
        return _prediction(predictor, row, score=score, abstain=False, support_count=_support_count(row, train_rows, "arm"))
    if predictor == "current_heuristic_world_model":
        pred = row["prediction"]
        score = 0.5 * float(pred["p_accept"]) + 0.5 * float(pred["expected_utility"]) - 0.25 * float(pred["p_regress"])
        return _prediction(predictor, row, score=score, abstain=False, support_count=1)
    if predictor == "handwritten_hybrid_guard":
        score = _hybrid_guard_score(row)
        return _prediction(predictor, row, score=score, abstain=False, support_count=1)
    if predictor == "random_with_abstain":
        raw = (int(stable_hash(["random", row["row_id"]]), 16) % 1000) / 999.0
        return _prediction(predictor, row, score=raw, abstain=0.45 <= raw <= 0.55, support_count=0)
    if predictor == "always_original_v3":
        score = 0.58 if row["action"]["arm"] in {"v3_full", "original_v3"} else 0.5
        return _prediction(predictor, row, score=score, abstain=False, support_count=0)
    if predictor == "always_run_ablation":
        score = 0.62 if row["action"]["type"] == "run_ablation" else 0.52
        return _prediction(predictor, row, score=score, abstain=False, support_count=0)
    raise ValueError(f"unknown predictor: {predictor}")


def _feature_similarity_prediction(row: dict[str, Any], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_rate = _mean(_label(train) for train in train_rows) if train_rows else 0.5
    arm_rate = _base_rate_for(row, train_rows, key_fn=lambda x: x["action"]["arm"])
    row_features = _safe_features(row)
    weighted_sum = 0.0
    weight_sum = 0.0
    matched = 0
    for train in train_rows:
        train_features = _safe_features(train)
        union = row_features | train_features
        if not union:
            continue
        overlap = row_features & train_features
        if not overlap:
            continue
        weight = (len(overlap) / len(union)) ** 2
        weighted_sum += weight * _label(train)
        weight_sum += weight
        matched += 1
    feature_rate = weighted_sum / weight_sum if weight_sum else global_rate
    score = 0.25 * global_rate + 0.25 * arm_rate + 0.5 * feature_rate
    abstain = matched < 3 or abs(score - 0.5) < 0.04
    return _prediction(
        "feature_similarity_simulator",
        row,
        score=score,
        abstain=abstain,
        support_count=matched,
        extra={"global_rate": round(global_rate, 4), "arm_rate": round(arm_rate, 4), "feature_rate": round(feature_rate, 4)},
    )


def _prediction(
    predictor: str,
    row: dict[str, Any],
    *,
    score: float,
    abstain: bool,
    support_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, float(score)))
    label = _label(row)
    return {
        "predictor": predictor,
        "row_id": row["row_id"],
        "score": round(score, 4),
        "label": label,
        "abstain": bool(abstain),
        "block": (not abstain) and score < 0.5,
        "support_count": int(support_count),
        **(extra or {}),
    }


def _evaluate_predictions(eval_rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labels_by_id = {row["row_id"]: _label(row) for row in eval_rows}
    covered = [pred for pred in predictions if not pred["abstain"]]
    bad_ids = {row["row_id"] for row in eval_rows if _label(row) == 0}
    good_ids = {row["row_id"] for row in eval_rows if _label(row) == 1}
    blocked = {pred["row_id"] for pred in covered if pred["block"]}
    brier = _mean((pred["score"] - labels_by_id[pred["row_id"]]) ** 2 for pred in covered)
    all_brier = _mean(
        ((pred["score"] if not pred["abstain"] else 0.5) - labels_by_id[pred["row_id"]]) ** 2
        for pred in predictions
    )
    return {
        "row_count": len(eval_rows),
        "covered_count": len(covered),
        "abstain_count": len(predictions) - len(covered),
        "abstention_rate": round((len(predictions) - len(covered)) / max(1, len(predictions)), 4),
        "brier": round(brier, 4),
        "brier_with_abstain_as_half": round(all_brier, 4),
        "ece": _ece(covered),
        "mean_score": round(_mean(pred["score"] for pred in covered), 4),
        "positive_rate": round(_mean(labels_by_id[pred["row_id"]] for pred in predictions), 4),
        "true_positive_block_rate": round(len(blocked & bad_ids) / max(1, len(bad_ids)), 4),
        "false_positive_block_rate": round(len(blocked & good_ids) / max(1, len(good_ids)), 4),
        "support_mean": round(_mean(pred["support_count"] for pred in predictions), 4),
    }


def _promotion_decision(split_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    feature = {
        split: report["predictors"]["feature_similarity_simulator"]["brier_with_abstain_as_half"]
        for split, report in split_reports.items()
    }
    base = {
        split: report["predictors"]["base_rate_per_arm"]["brier_with_abstain_as_half"]
        for split, report in split_reports.items()
    }
    heuristic = {
        split: report["predictors"]["current_heuristic_world_model"]["brier_with_abstain_as_half"]
        for split, report in split_reports.items()
    }
    heuristic_false_positive_block = {
        split: report["predictors"]["current_heuristic_world_model"]["false_positive_block_rate"]
        for split, report in split_reports.items()
    }
    heuristic_ece = {
        split: report["predictors"]["current_heuristic_world_model"]["ece"]
        for split, report in split_reports.items()
    }
    base_ece = {
        split: report["predictors"]["base_rate_per_arm"]["ece"]
        for split, report in split_reports.items()
    }
    feature_nonharm = (
        feature["leave_domain_out"] <= base["leave_domain_out"] + 0.02
        and feature["leave_pattern_out"] <= base["leave_pattern_out"] + 0.02
    )
    heuristic_nonharm = (
        heuristic["leave_domain_out"] <= base["leave_domain_out"] + 0.02
        and heuristic["leave_pattern_out"] <= base["leave_pattern_out"] + 0.02
        and heuristic_false_positive_block["leave_domain_out"] <= 0.10
        and heuristic_false_positive_block["leave_pattern_out"] <= 0.10
        and heuristic_ece["leave_domain_out"] <= base_ece["leave_domain_out"] + 0.02
        and heuristic_ece["leave_pattern_out"] <= base_ece["leave_pattern_out"] + 0.02
    )
    return {
        "feature_model_promotion_allowed": bool(
            feature["leave_one_out"] <= base["leave_one_out"] and feature_nonharm
        ),
        "raw_predictor_promotion_allowed": bool(
            heuristic["leave_one_out"] <= base["leave_one_out"] and heuristic_nonharm
        ),
        "production_replacement_allowed": False,
        "rule": (
            "Raw predictor cannot be production-promoted from LOO alone; leave-domain and leave-pattern must "
            "be non-harmful against base-rate per-arm."
        ),
        "feature_model_brier_by_split": feature,
        "base_rate_brier_by_split": base,
        "current_heuristic_brier_by_split": heuristic,
        "current_heuristic_ece_by_split": heuristic_ece,
        "base_rate_ece_by_split": base_ece,
        "current_heuristic_false_positive_block_by_split": heuristic_false_positive_block,
        "feature_model_nonharm_leave_domain_pattern": feature_nonharm,
        "current_heuristic_nonharm_leave_domain_pattern": heuristic_nonharm,
    }


def _hybrid_guard_score(row: dict[str, Any]) -> float:
    features = set(row.get("state", {}).get("world_model_features", []))
    row_kind = next((feature.split(":", 1)[1] for feature in features if feature.startswith("row_kind:")), "")
    if "decision:accept" in features:
        return 0.82
    if "decision:reject_harm" in features and row_kind == "control":
        return 0.18
    if "decision:reject_benefit" in features and row_kind == "trigger":
        return 0.35
    if row["action"]["arm"] == "v3_full":
        return 0.58
    return 0.5


def _base_rate_for(row: dict[str, Any], train_rows: list[dict[str, Any]], *, key_fn: Callable[[dict[str, Any]], str]) -> float:
    key = key_fn(row)
    bucket = [train for train in train_rows if key_fn(train) == key]
    if bucket:
        return _mean(_label(train) for train in bucket)
    return _mean(_label(train) for train in train_rows) if train_rows else 0.5


def _support_count(row: dict[str, Any], train_rows: list[dict[str, Any]], level: str) -> int:
    if level == "arm":
        return sum(1 for train in train_rows if train["action"]["arm"] == row["action"]["arm"])
    return len(train_rows)


def _safe_features(row: dict[str, Any]) -> set[str]:
    features = set(str(feature) for feature in row.get("state", {}).get("world_model_features", []))
    features.update(
        {
            f"domain:{row['state']['domain']}",
            f"pattern:{row['state']['pattern']}",
            f"action:{row['action']['type']}",
            f"arm:{row['action']['arm']}",
        }
    )
    return {feature for feature in features if not _is_leaky_feature(feature)}


def _is_leaky_feature(feature: str) -> bool:
    return feature.startswith("decision:")


def _label(row: dict[str, Any]) -> int:
    outcome = row["outcome"]
    if outcome.get("control_harm") or outcome.get("regression"):
        return 0
    return 1 if float(outcome.get("utility_vs_baseline") or 0.0) >= 0.5 else 0


def _ece(predictions: list[dict[str, Any]], *, bin_count: int = 10) -> float:
    if not predictions:
        return 0.0
    total = len(predictions)
    acc = 0.0
    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count
        if index == bin_count - 1:
            bucket = [pred for pred in predictions if low <= pred["score"] <= high]
        else:
            bucket = [pred for pred in predictions if low <= pred["score"] < high]
        if not bucket:
            continue
        confidence = _mean(pred["score"] for pred in bucket)
        observed = _mean(pred["label"] for pred in bucket)
        acc += len(bucket) / total * abs(confidence - observed)
    return round(acc, 4)


def _all_metric_keys_present(split_reports: dict[str, dict[str, Any]]) -> bool:
    required = {"brier", "ece", "abstention_rate", "true_positive_block_rate"}
    for report in split_reports.values():
        for metrics in report["predictors"].values():
            if not required.issubset(metrics):
                return False
    return True


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simulator transition split discipline.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="simulator_eval_splits_20260612")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_eval_splits_payload(
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
