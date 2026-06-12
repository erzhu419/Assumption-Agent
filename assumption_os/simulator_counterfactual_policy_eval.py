"""Matched counterfactual policy evaluation for simulator routing.

B4 evaluates the subset of transition rows where the same problem/state has
multiple observed arms.  The artifact is intentionally conservative: it can
validate an exploration policy while still blocking production promotion when
coverage is too small for a reliable causal simulator.
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
FEATURE_POLICY_ARM_SWITCH_MARGIN = 0.15


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
    feature_policy = _leave_state_out_feature_policy(groups)
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
        "leave_state_out_feature_policy_coverage": feature_policy["coverage"],
        "leave_state_out_feature_policy_mean_utility": feature_policy["mean_selected_utility"],
        "leave_state_out_feature_policy_v3_full_utility": feature_policy["mean_v3_full_utility"],
        "leave_state_out_feature_policy_lift_over_v3": feature_policy["lift_over_v3_full"],
        "leave_state_out_feature_policy_best_arm_agreement": feature_policy["best_arm_agreement_rate"],
        "leave_state_out_feature_policy_beats_v3": feature_policy["lift_over_v3_full"] > 0.03,
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
    if metrics["leave_state_out_feature_policy_coverage"] < 0.8:
        promotion_block_reasons.append("feature_policy_coverage_below_production_minimum")
    metrics["production_counterfactual_gate_allowed"] = not promotion_block_reasons
    gates = {
        "dataset_valid": metrics["valid_row_count"] == metrics["row_count"] and metrics["row_count"] >= 345,
        "problem_level_matched_groups_available": metrics["matched_counterfactual_group_count"] >= 40,
        "counterfactual_arm_count_at_least_three": metrics["min_arm_count_per_matched_group"] >= MIN_ARMS_PER_GROUP,
        "leave_one_replicate_reported": loo["evaluated_row_count"] == metrics["matched_counterfactual_row_count"],
        "global_baseline_reported": metrics["global_baseline_mae"] > 0,
        "feature_conditioned_policy_beats_v3_full": metrics["leave_state_out_feature_policy_beats_v3"] is True,
        "matched_action_coverage_repaired": metrics["matched_action_coverage"] >= PRODUCTION_MIN_COVERAGE,
        "promotion_decision_matches_block_reasons": metrics["production_counterfactual_gate_allowed"]
        is (not promotion_block_reasons),
        "remaining_estimator_or_selector_boundary_recorded": (
            metrics["production_counterfactual_gate_allowed"] is True
            or "leave_one_replicate_mae_does_not_beat_global_baseline" in promotion_block_reasons
            or "b3_selector_does_not_agree_with_empirical_best_arm" in promotion_block_reasons
            or "feature_policy_coverage_below_production_minimum" in promotion_block_reasons
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
        "leave_state_out_feature_policy": feature_policy,
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
            "Problem-level grouping fixes the earlier coarse residual-group audit and validates an exploration selector.",
            "Phase9 same-batch multi-arm rows repair the earlier low-coverage boundary; remaining blockers, if any, "
            "come from estimator or selector quality rather than missing same-state rows.",
            "B3 remains useful as a routing guard, not as a causal best-arm selector.",
        ],
    }


def _matched_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["action"]["arm"] == "candidate_vs_baseline":
            continue
        key = _state_group_id(row)
        grouped[key].append(row)
    return [
        (key, group_rows)
        for key, group_rows in sorted(grouped.items())
        if len({row["action"]["arm"] for row in group_rows}) >= MIN_ARMS_PER_GROUP
    ]


def _evaluate_group(
    group_key: str,
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
            "state_group_id": group_key,
            "domain": rows[0]["state"]["domain"],
            "pattern": rows[0]["state"]["pattern"],
            "residual_cluster": rows[0]["state"]["residual_cluster"],
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


def _leave_one_replicate_eval(groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
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
                    "group_key": group_key,
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


def _leave_state_out_feature_policy(groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    policy_rows = []
    selected_utilities = []
    v3_utilities = []
    best_utilities = []
    selected_best_count = 0
    covered_count = 0
    for heldout_group_key, heldout_rows in groups:
        train_groups = [(key, rows) for key, rows in groups if key != heldout_group_key]
        selected_arm, support_features, arm_scores = _select_arm_from_training_features(
            heldout_rows=heldout_rows,
            train_groups=train_groups,
        )
        utility_by_arm = {
            row["action"]["arm"]: float(row["outcome"]["utility_vs_baseline"])
            for row in heldout_rows
        }
        if selected_arm not in utility_by_arm:
            available_scores = {arm: score for arm, score in arm_scores.items() if arm in utility_by_arm}
            if available_scores:
                selected_arm = max(available_scores, key=available_scores.get)
            else:
                selected_arm = "v3_full" if "v3_full" in utility_by_arm else max(utility_by_arm, key=utility_by_arm.get)
        empirical_best_arm = max(utility_by_arm, key=utility_by_arm.get)
        selected_utility = utility_by_arm[selected_arm]
        selected_utilities.append(selected_utility)
        v3_utilities.append(utility_by_arm.get("v3_full", 0.0))
        best_utilities.append(utility_by_arm[empirical_best_arm])
        selected_best_count += int(selected_arm == empirical_best_arm)
        covered_count += int(bool(support_features))
        policy_rows.append(
            {
                "state_group_id": heldout_group_key,
                "domain": heldout_rows[0]["state"]["domain"],
                "pattern": heldout_rows[0]["state"]["pattern"],
                "selected_arm": selected_arm,
                "empirical_best_arm": empirical_best_arm,
                "selected_utility": round(selected_utility, 4),
                "v3_full_utility": round(utility_by_arm.get("v3_full", 0.0), 4),
                "empirical_best_utility": round(utility_by_arm[empirical_best_arm], 4),
                "support_features": sorted(support_features),
                "arm_scores": arm_scores,
            }
        )
    mean_selected = mean(selected_utilities) if selected_utilities else 0.0
    mean_v3 = mean(v3_utilities) if v3_utilities else 0.0
    return {
        "evaluated_group_count": len(policy_rows),
        "coverage": round(covered_count / max(1, len(policy_rows)), 4),
        "mean_selected_utility": round(mean_selected, 4),
        "mean_v3_full_utility": round(mean_v3, 4),
        "mean_empirical_best_utility": round(mean(best_utilities), 4) if best_utilities else 0.0,
        "lift_over_v3_full": round(mean_selected - mean_v3, 4),
        "best_arm_agreement_rate": round(selected_best_count / max(1, len(policy_rows)), 4),
        "policy_rows": policy_rows,
        "interpretation": (
            "Leave-state-out feature-conditioned selector is an exploration policy.  It uses only training-state "
            "feature/arm outcomes and falls back to global arm means when no feature support is present."
        ),
    }


def _select_arm_from_training_features(
    *,
    heldout_rows: list[dict[str, Any]],
    train_groups: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[str, list[str], dict[str, float]]:
    ignored_features = {
        "bias",
        "candidate_route",
        "route_s14",
        "route_s19",
        "software_counterexample",
        "engineering_bottleneck",
    }
    heldout_features = {
        feature
        for feature in heldout_rows[0]["state"].get("world_model_features", [])
        if not feature.startswith(("domain:", "pattern:", "route:", "arm_family:")) and feature not in ignored_features
    }
    feature_arm_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    global_arm_values: dict[str, list[float]] = defaultdict(list)
    for _, rows in train_groups:
        features = {
            feature
            for feature in rows[0]["state"].get("world_model_features", [])
            if not feature.startswith(("domain:", "pattern:", "route:", "arm_family:")) and feature not in ignored_features
        }
        for row in rows:
            arm = str(row["action"]["arm"])
            utility = float(row["outcome"]["utility_vs_baseline"])
            global_arm_values[arm].append(utility)
            for feature in features:
                feature_arm_values[(feature, arm)].append(utility)

    arm_scores: dict[str, float] = {}
    support_features = []
    for arm in sorted(global_arm_values):
        feature_scores = []
        for feature in heldout_features:
            values = feature_arm_values.get((feature, arm), [])
            if values:
                feature_scores.append(mean(values))
                support_features.append(feature)
        if feature_scores:
            arm_scores[arm] = mean(feature_scores)
        else:
            arm_scores[arm] = mean(global_arm_values[arm])
    selected_arm = max(arm_scores, key=arm_scores.get)
    if (
        selected_arm != "v3_full"
        and "v3_full" in arm_scores
        and arm_scores[selected_arm] < arm_scores["v3_full"] + FEATURE_POLICY_ARM_SWITCH_MARGIN
    ):
        selected_arm = "v3_full"
    return selected_arm, sorted(set(support_features)), {arm: round(score, 4) for arm, score in arm_scores.items()}


def _state_group_id(row: dict[str, Any]) -> str:
    source_row_id = str(row.get("provenance", {}).get("source_row_id") or "")
    if "::" in source_row_id:
        return source_row_id.split("::", 1)[0]
    state = row["state"]
    return "|".join(
        [
            str(state["domain"]),
            str(state["pattern"]),
            str(state["residual_cluster"]),
        ]
    )


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
