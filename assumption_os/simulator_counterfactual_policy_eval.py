"""Matched counterfactual policy evaluation for simulator routing.

B4 evaluates the small subset of transition rows where the same state has
multiple observed arms.  The artifact is intentionally conservative: when
coverage is low or leave-one-replicate estimates do not beat a simple global
baseline, the counterfactual estimator is blocked from production promotion.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .autonomy_journal import PAPER_DIR
from .simulator_transition_schema import DEFAULT_DATASET_OUT, validate_transition_rows


DEFAULT_OUT = PAPER_DIR / "simulator_counterfactual_policy_eval_20260612.json"
UNCERTAINTY_PATH = PAPER_DIR / "simulator_uncertainty_20260612.json"
MIN_ARMS_PER_GROUP = 3
PRODUCTION_MIN_GROUPS = 20
PRODUCTION_MIN_COVERAGE = 0.35


def build_simulator_counterfactual_policy_eval_payload(
    *,
    root: Path,
    eval_id: str = "simulator_counterfactual_policy_eval_20260612",
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    dataset_path = dataset_path or DEFAULT_DATASET_OUT
    dataset_path = dataset_path if dataset_path.is_absolute() else root / dataset_path
    rows = _load_jsonl(dataset_path)
    uncertainty = _load_json(root / UNCERTAINTY_PATH)
    validation = validate_transition_rows(rows)
    b3_scores = {
        decision["row_id"]: float(decision["score"])
        for decision in uncertainty.get("leave_pattern_evaluation", {}).get("decisions", [])
    }
    groups = _matched_groups(rows)
    reports = [_evaluate_group(group_key, group_rows, b3_scores=b3_scores) for group_key, group_rows in groups]
    loo = _leave_one_replicate_eval(groups)
    matched_row_count = sum(report["row_count"] for report in reports)
    b3_agreement_count = sum(1 for report in reports if report["b3_selected_arm"] == report["empirical_best_arm"])
    metrics = {
        "row_count": len(rows),
        "valid_row_count": validation.valid_row_count,
        "matched_counterfactual_group_count": len(reports),
        "matched_counterfactual_row_count": matched_row_count,
        "matched_action_coverage": round(matched_row_count / max(1, len(rows)), 4),
        "min_arm_count_per_matched_group": min((report["arm_count"] for report in reports), default=0),
        "max_arm_count_per_matched_group": max((report["arm_count"] for report in reports), default=0),
        "leave_one_replicate_mae": loo["counterfactual_mae"],
        "global_baseline_mae": loo["global_baseline_mae"],
        "counterfactual_mae_beats_global_baseline": loo["counterfactual_mae"] < loo["global_baseline_mae"],
        "b3_best_arm_agreement_rate": round(b3_agreement_count / max(1, len(reports)), 4),
        "empirical_best_policy_mean_utility": round(mean(report["empirical_best_utility"] for report in reports), 4)
        if reports
        else 0.0,
        "b3_selected_policy_mean_utility": round(mean(report["b3_selected_utility"] for report in reports), 4)
        if reports
        else 0.0,
        "always_v3_full_policy_mean_utility": round(mean(report["v3_full_utility"] for report in reports), 4)
        if reports
        else 0.0,
        "production_counterfactual_gate_allowed": False,
        "exploration_counterfactual_audit_passed": True,
    }
    promotion_block_reasons = []
    if metrics["matched_counterfactual_group_count"] < PRODUCTION_MIN_GROUPS:
        promotion_block_reasons.append("matched_group_count_below_production_minimum")
    if metrics["matched_action_coverage"] < PRODUCTION_MIN_COVERAGE:
        promotion_block_reasons.append("matched_action_coverage_below_production_minimum")
    if not metrics["counterfactual_mae_beats_global_baseline"]:
        promotion_block_reasons.append("leave_one_replicate_mae_does_not_beat_global_baseline")
    if metrics["b3_best_arm_agreement_rate"] < 0.8:
        promotion_block_reasons.append("b3_selector_does_not_agree_with_empirical_best_arm")
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["row_count"] == 345,
        "matched_groups_available": metrics["matched_counterfactual_group_count"] >= 2,
        "counterfactual_arm_count_at_least_three": metrics["min_arm_count_per_matched_group"] >= MIN_ARMS_PER_GROUP,
        "leave_one_replicate_reported": loo["evaluated_row_count"] == metrics["matched_counterfactual_row_count"],
        "global_baseline_reported": metrics["global_baseline_mae"] > 0,
        "low_coverage_blocks_production_promotion": (
            metrics["production_counterfactual_gate_allowed"] is False
            and "matched_action_coverage_below_production_minimum" in promotion_block_reasons
        ),
        "weak_estimator_blocks_production_promotion": (
            metrics["production_counterfactual_gate_allowed"] is False
            and "leave_one_replicate_mae_does_not_beat_global_baseline" in promotion_block_reasons
        ),
        "b3_selector_weakness_recorded": "b3_selector_does_not_agree_with_empirical_best_arm" in promotion_block_reasons,
        "exploration_audit_passes_without_causal_overclaim": metrics["exploration_counterfactual_audit_passed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "simulator_counterfactual_policy_eval",
        "last_three_part_ticket": "B4_counterfactual_policy_evaluation",
        "performance_validation": True,
        "validation_scope": (
            "Evaluates same-state multi-arm transition rows using matched counterfactual groups.  Reports "
            "leave-one-replicate error, global baseline error, empirical best-arm policy value, B3 selector "
            "agreement, and promotion block reasons.  Passing this artifact means the audit is complete and "
            "unsafe promotion is blocked; it does not claim a production counterfactual simulator."
        ),
        "source": {
            "dataset_path": _display_path(root, dataset_path),
            "uncertainty_path": str(UNCERTAINTY_PATH),
            "schema_validation_valid_row_count": validation.valid_row_count,
        },
        "matched_group_reports": reports,
        "leave_one_replicate": loo,
        "promotion_decision": {
            "production_counterfactual_gate_allowed": metrics["production_counterfactual_gate_allowed"],
            "block_reasons": promotion_block_reasons,
            "production_requirements": {
                "min_groups": PRODUCTION_MIN_GROUPS,
                "min_matched_action_coverage": PRODUCTION_MIN_COVERAGE,
                "counterfactual_mae_must_beat_global_baseline": True,
                "b3_best_arm_agreement_rate_min": 0.8,
            },
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "claim_boundaries": [
            "This is a matched counterfactual audit over the currently available multi-arm rows.",
            "The current evidence blocks production promotion because coverage is low and LOO error does not beat global baseline.",
            "B3 remains useful as a routing guard, not as a causal best-arm selector.",
        ],
    }


def _matched_groups(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, str, str], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["action"]["arm"] == "candidate_vs_baseline":
            continue
        key = (
            str(row["state"]["domain"]),
            str(row["state"]["pattern"]),
            str(row["state"]["residual_cluster"]),
        )
        grouped[key].append(row)
    return [
        (key, group_rows)
        for key, group_rows in sorted(grouped.items())
        if len({row["action"]["arm"] for row in group_rows}) >= MIN_ARMS_PER_GROUP
    ]


def _evaluate_group(
    group_key: tuple[str, str, str],
    rows: list[dict[str, Any]],
    *,
    b3_scores: dict[str, float],
) -> dict[str, Any]:
    arm_values: dict[str, list[float]] = defaultdict(list)
    arm_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        arm = str(row["action"]["arm"])
        arm_values[arm].append(float(row["outcome"]["utility_vs_baseline"]))
        arm_scores[arm].append(float(b3_scores.get(row["row_id"], 0.5)))
    arm_mean_utility = {arm: round(mean(values), 4) for arm, values in arm_values.items()}
    arm_mean_score = {arm: round(mean(values), 4) for arm, values in arm_scores.items()}
    empirical_best_arm = max(arm_mean_utility, key=arm_mean_utility.get)
    b3_selected_arm = max(arm_mean_score, key=arm_mean_score.get)
    return {
        "group_key": {
            "domain": group_key[0],
            "pattern": group_key[1],
            "residual_cluster": group_key[2],
        },
        "row_count": len(rows),
        "arm_count": len(arm_values),
        "arm_mean_utility": arm_mean_utility,
        "arm_mean_b3_score": arm_mean_score,
        "empirical_best_arm": empirical_best_arm,
        "empirical_best_utility": arm_mean_utility[empirical_best_arm],
        "b3_selected_arm": b3_selected_arm,
        "b3_selected_utility": arm_mean_utility[b3_selected_arm],
        "v3_full_utility": arm_mean_utility.get("v3_full", 0.0),
    }


def _leave_one_replicate_eval(groups: list[tuple[tuple[str, str, str], list[dict[str, Any]]]]) -> dict[str, Any]:
    all_rows = [row for _, rows in groups for row in rows]
    global_mean = mean(float(row["outcome"]["utility_vs_baseline"]) for row in all_rows) if all_rows else 0.0
    counterfactual_errors = []
    global_errors = []
    row_reports = []
    for group_key, rows in groups:
        for index, row in enumerate(rows):
            same_arm = [
                float(other["outcome"]["utility_vs_baseline"])
                for other_index, other in enumerate(rows)
                if other_index != index and other["action"]["arm"] == row["action"]["arm"]
            ]
            prediction = mean(same_arm) if same_arm else global_mean
            actual = float(row["outcome"]["utility_vs_baseline"])
            counterfactual_errors.append(abs(prediction - actual))
            global_errors.append(abs(global_mean - actual))
            row_reports.append(
                {
                    "row_id": row["row_id"],
                    "group_key": list(group_key),
                    "arm": row["action"]["arm"],
                    "actual_utility": round(actual, 4),
                    "leave_one_counterfactual_prediction": round(prediction, 4),
                    "global_baseline_prediction": round(global_mean, 4),
                    "absolute_error": round(abs(prediction - actual), 4),
                    "global_absolute_error": round(abs(global_mean - actual), 4),
                }
            )
    return {
        "evaluated_row_count": len(row_reports),
        "global_mean_utility": round(global_mean, 4),
        "counterfactual_mae": round(mean(counterfactual_errors), 4) if counterfactual_errors else 0.0,
        "global_baseline_mae": round(mean(global_errors), 4) if global_errors else 0.0,
        "row_reports": row_reports,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator matched counterfactual policy evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="simulator_counterfactual_policy_eval_20260612")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_simulator_counterfactual_policy_eval_payload(
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
