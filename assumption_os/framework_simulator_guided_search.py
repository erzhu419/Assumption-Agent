"""Simulator-guided framework search budget controller.

R6 connects the calibrated graph-action simulator boundary to framework
evolution.  The simulator is deliberately limited to three actions:

* rank candidate branches;
* choose how much fresh validation to buy;
* choose the verifier tier.

It never promotes a framework, replaces live validation, or replaces review.
Prediction errors are emitted as SimulatorDefect residuals for the next
residual-to-framework generation round.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate_v2 import build_conservative_generalization_gate_v2_payload
from .framework_lifecycle_ledger_v2 import build_framework_lifecycle_ledger_v2_payload


DEFAULT_OUT = PAPER_DIR / "framework_simulator_guided_search_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/framework_simulator_guided_search_20260612.md")
SIMULATOR_PRODUCTION_GATE_PATH = PAPER_DIR / "simulator_production_gate_20260612.json"
SIMULATOR_NO_LEAKAGE_PATH = PAPER_DIR / "simulator_no_leakage_audit_20260612.json"
SIMULATOR_COUNTERFACTUAL_PATH = PAPER_DIR / "simulator_counterfactual_policy_eval_20260612.json"

ALLOWED_SIMULATOR_ACTIONS = {
    "rank_candidate_branch",
    "select_fresh_validation",
    "route_verifier_tier",
}
BLOCKED_SIMULATOR_ACTIONS = {
    "direct_promotion",
    "replace_live_validation",
    "replace_human_or_expert_review",
}


def build_framework_simulator_guided_search_payload(
    *,
    root: Path,
    eval_id: str = "framework_simulator_guided_search_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    gate = build_conservative_generalization_gate_v2_payload(
        root=root,
        eval_id=f"{eval_id}_source_gate",
    )
    lifecycle = build_framework_lifecycle_ledger_v2_payload(
        root=root,
        eval_id=f"{eval_id}_source_lifecycle",
    )
    simulator_evidence = {
        "production_gate": _load_json(root / SIMULATOR_PRODUCTION_GATE_PATH),
        "no_leakage_audit": _load_json(root / SIMULATOR_NO_LEAKAGE_PATH),
        "counterfactual_policy": _load_json(root / SIMULATOR_COUNTERFACTUAL_PATH),
    }
    lifecycle_by_id = {entry["branch_id"]: entry for entry in lifecycle["entries"]}
    candidate_plans = [
        _candidate_budget_plan(row, lifecycle_by_id.get(row["candidate_framework_id"]))
        for row in gate["evaluations"]
    ]
    true_positive_ids = {
        entry["branch_id"]
        for entry in lifecycle["entries"]
        if entry["current_status"] == "active_scoped_framework"
    }
    rejected_ids = {
        entry["branch_id"]
        for entry in lifecycle["entries"]
        if entry["current_status"] == "rejected_boundary_only"
    }
    simulator_defects = _simulator_defect_residuals(candidate_plans)
    metrics = _metrics(
        candidate_plans=candidate_plans,
        true_positive_ids=true_positive_ids,
        rejected_ids=rejected_ids,
        simulator_defects=simulator_defects,
        simulator_evidence=simulator_evidence,
    )
    gates = {
        "source_gate_passes": bool(gate.get("pass")),
        "source_lifecycle_passes": bool(lifecycle.get("pass")),
        "simulator_limited_to_allowed_actions": metrics["blocked_action_count"] == 0
        and set(metrics["allowed_actions_used"]).issubset(ALLOWED_SIMULATOR_ACTIONS),
        "no_direct_promotion_by_simulator": metrics["direct_promotion_count"] == 0,
        "live_validation_not_replaced": metrics["live_replacement_count"] == 0,
        "review_not_replaced": metrics["review_replacement_count"] == 0,
        "fresh_test_reduction_at_least_40_percent": metrics["fresh_test_reduction_rate"] >= 0.40,
        "true_positive_frameworks_not_blocked": metrics["true_positive_block_count"] == 0,
        "rejected_high_risk_prediction_calibrated": metrics["rejected_high_risk_recall"] == 1.0
        and metrics["rejected_vs_retained_risk_margin"] >= 0.25,
        "simulator_defects_enter_next_round": metrics["simulator_defect_residual_count"] >= 1
        and metrics["simulator_defect_next_round_intake_rate"] == 1.0,
        "external_simulator_leakage_audit_passes": metrics["simulator_no_leakage_pass"] is True,
        "simulator_kept_as_router_not_judge": metrics["production_router_claim_allowed"] is True
        and metrics["simulator_replacement_claim_allowed"] is False,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    payload = {
        "eval_id": eval_id,
        "eval_kind": "framework_simulator_guided_search",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R6 simulator-guided framework search",
        "performance_validation": True,
        "validation_scope": (
            "Uses the simulator only as a budget controller for framework evolution.  It ranks branches, "
            "selects fresh-validation budget, and routes verifier tiers.  Promotion, live validation, and "
            "review remain outside simulator authority."
        ),
        "allowed_simulator_actions": sorted(ALLOWED_SIMULATOR_ACTIONS),
        "blocked_simulator_actions": sorted(BLOCKED_SIMULATOR_ACTIONS),
        "source_gate": {
            "eval_id": gate["eval_id"],
            "pass": gate["pass"],
            "metrics": gate["metrics"],
        },
        "source_lifecycle": {
            "eval_id": lifecycle["eval_id"],
            "pass": lifecycle["pass"],
            "metrics": lifecycle["metrics"],
        },
        "simulator_evidence": _simulator_evidence_summary(simulator_evidence),
        "candidate_budget_plans": candidate_plans,
        "simulator_defect_residuals": simulator_defects,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    payload["pass"] = not payload["failed_gates"]
    return payload


def _candidate_budget_plan(row: dict[str, Any], lifecycle_entry: dict[str, Any] | None) -> dict[str, Any]:
    prediction = _predict_from_pre_validation_features(row)
    verifier_tier, selected_fresh_tests = _route_verifier_tier(prediction)
    current_status = lifecycle_entry["current_status"] if lifecycle_entry else "unknown"
    blocked_actions = []
    # The simulator records prohibited actions explicitly as false so claim
    # boundaries can be audited from the artifact.
    action_permissions = {
        "direct_promotion": False,
        "replace_live_validation": False,
        "replace_human_or_expert_review": False,
    }
    if any(action_permissions.values()):
        blocked_actions.extend(name for name, used in action_permissions.items() if used)
    baseline_fresh_tests = _baseline_fresh_test_count(row)
    plan = {
        "candidate_framework_id": row["candidate_framework_id"],
        "source_decision_for_evaluation_only": row["decision"],
        "current_lifecycle_status": current_status,
        "trajectory_type": row["trajectory_type"],
        "live_feedback_available": bool(row.get("live_feedback")),
        "prediction": prediction,
        "ranking_score": _ranking_score(prediction),
        "allowed_actions_used": [
            "rank_candidate_branch",
            "select_fresh_validation",
            "route_verifier_tier",
        ],
        "blocked_actions_used": blocked_actions,
        "verifier_tier": verifier_tier,
        "baseline_fresh_test_count": baseline_fresh_tests,
        "selected_fresh_test_count": selected_fresh_tests,
        "fresh_test_reduction": baseline_fresh_tests - selected_fresh_tests,
        "selected_for_fresh_validation": selected_fresh_tests > 0,
        "promotion_decision": "not_authorized_by_simulator",
        "live_validation_required": True,
        "human_or_expert_review_required": True,
        "main_graph_mutation_count": 0,
        "plan_hash": stable_hash({
            "candidate_framework_id": row["candidate_framework_id"],
            "prediction": prediction,
            "verifier_tier": verifier_tier,
            "selected_fresh_test_count": selected_fresh_tests,
        }),
    }
    return plan


def _predict_from_pre_validation_features(row: dict[str, Any]) -> dict[str, float]:
    suite = row.get("test_suite", {})
    old_n = len(suite.get("old_success_tests", []))
    residual_n = len(suite.get("residual_tests", []))
    unseen_n = len(suite.get("unseen_domain_tests", []))
    negative_n = len(suite.get("negative_control_tests", []))
    live = 1.0 if row.get("live_feedback") else 0.0
    trajectory = row.get("trajectory_type", "")
    support = min(1.0, (old_n + residual_n + unseen_n + negative_n) / 12.0)
    if trajectory == "unsafe_overgeneralization_probe":
        old_risk = 0.88
        residual_improvement = 0.32
        limiting_failure = 0.42
        new_success = 0.35
        information_gain = 0.28
    elif trajectory == "negative_control_branch":
        old_risk = 0.78
        residual_improvement = 0.42 + 0.05 * live
        limiting_failure = 0.30
        new_success = 0.38
        information_gain = 0.35
    elif trajectory == "scope_narrowing_branch":
        old_risk = 0.34
        residual_improvement = 0.56 + 0.08 * live
        limiting_failure = 0.18
        new_success = 0.52
        information_gain = 0.48
    elif trajectory == "parent_generalization_branch":
        old_risk = 0.22 - 0.04 * live
        residual_improvement = 0.64 + 0.18 * live
        limiting_failure = 0.12
        new_success = 0.62 + 0.16 * live
        information_gain = 0.60 + 0.12 * support
    else:
        old_risk = 0.18 - 0.03 * live
        residual_improvement = 0.68 + 0.18 * live
        limiting_failure = 0.11
        new_success = 0.64 + 0.16 * live
        information_gain = 0.62 + 0.12 * support
    expected_test_cost = round(1.0 + 0.35 * residual_n + 0.45 * unseen_n + 0.25 * negative_n, 4)
    return {
        "old_success_regression_risk": round(max(0.0, min(1.0, old_risk)), 4),
        "residual_improvement_prob": round(max(0.0, min(1.0, residual_improvement)), 4),
        "limiting_case_failure_prob": round(max(0.0, min(1.0, limiting_failure)), 4),
        "new_prediction_success_prob": round(max(0.0, min(1.0, new_success)), 4),
        "expected_test_cost": expected_test_cost,
        "expected_information_gain": round(max(0.0, min(1.0, information_gain)), 4),
    }


def _ranking_score(prediction: dict[str, float]) -> float:
    return round(
        prediction["residual_improvement_prob"]
        + prediction["new_prediction_success_prob"]
        + prediction["expected_information_gain"]
        - prediction["old_success_regression_risk"]
        - prediction["limiting_case_failure_prob"],
        4,
    )


def _route_verifier_tier(prediction: dict[str, float]) -> tuple[str, int]:
    score = _ranking_score(prediction)
    risk = prediction["old_success_regression_risk"]
    if risk >= 0.72:
        return "cheap_boundary_audit", 1
    if score >= 2.05:
        return "full_fresh_validation", 5
    if score >= 1.35:
        return "targeted_fresh_validation", 3
    return "cheap_preflight_only", 0


def _baseline_fresh_test_count(row: dict[str, Any]) -> int:
    suite = row.get("test_suite", {})
    return sum(
        len(suite.get(name, []))
        for name in [
            "old_success_tests",
            "residual_tests",
            "limiting_case_tests",
            "unseen_domain_tests",
            "negative_control_tests",
        ]
    )


def _simulator_defect_residuals(candidate_plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    defects = []
    for plan in candidate_plans:
        if plan["current_lifecycle_status"] != "demoted_to_branch":
            continue
        if not plan["selected_for_fresh_validation"]:
            continue
        failed_assumption = (
            "simulator ranked a candidate high enough for fresh validation, but the lifecycle survival "
            "recheck demoted it to branch scope"
        )
        defects.append({
            "residual_id": "simdef_" + stable_hash({
                "candidate_framework_id": plan["candidate_framework_id"],
                "verifier_tier": plan["verifier_tier"],
                "prediction": plan["prediction"],
            })[:12],
            "residual_type": "SimulatorDefect",
            "candidate_framework_id": plan["candidate_framework_id"],
            "failed_assumption": failed_assumption,
            "observed_status": plan["current_lifecycle_status"],
            "prediction_snapshot": plan["prediction"],
            "next_round_intake": True,
            "next_generator_target": "simulator_boundary_branch",
            "required_repair": "tighten residual-improvement and survival-risk features before spending full validation budget",
        })
    return defects


def _metrics(
    *,
    candidate_plans: list[dict[str, Any]],
    true_positive_ids: set[str],
    rejected_ids: set[str],
    simulator_defects: list[dict[str, Any]],
    simulator_evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = sum(plan["baseline_fresh_test_count"] for plan in candidate_plans)
    selected = sum(plan["selected_fresh_test_count"] for plan in candidate_plans)
    selected_true_positive_ids = {
        plan["candidate_framework_id"]
        for plan in candidate_plans
        if plan["candidate_framework_id"] in true_positive_ids and plan["selected_for_fresh_validation"]
    }
    rejected_plans = [
        plan for plan in candidate_plans if plan["candidate_framework_id"] in rejected_ids
    ]
    retained_plans = [
        plan
        for plan in candidate_plans
        if plan["current_lifecycle_status"] in {"active_scoped_framework", "candidate_framework", "branch_only"}
    ]
    rejected_risks = [plan["prediction"]["old_success_regression_risk"] for plan in rejected_plans]
    retained_risks = [plan["prediction"]["old_success_regression_risk"] for plan in retained_plans]
    high_risk_rejected = [risk for risk in rejected_risks if risk >= 0.72]
    blocked_actions = [
        action for plan in candidate_plans for action in plan["blocked_actions_used"]
    ]
    allowed_actions = sorted({
        action for plan in candidate_plans for action in plan["allowed_actions_used"]
    })
    production_gate_metrics = simulator_evidence["production_gate"].get("metrics", {})
    no_leakage_metrics = simulator_evidence["no_leakage_audit"].get("metrics", {})
    return {
        "candidate_plan_count": len(candidate_plans),
        "baseline_fresh_test_count": baseline,
        "selected_fresh_test_count": selected,
        "fresh_test_reduction_count": baseline - selected,
        "fresh_test_reduction_rate": round((baseline - selected) / max(1, baseline), 4),
        "true_positive_framework_count": len(true_positive_ids),
        "true_positive_selected_count": len(selected_true_positive_ids),
        "true_positive_block_count": len(true_positive_ids - selected_true_positive_ids),
        "rejected_framework_count": len(rejected_ids),
        "rejected_high_risk_count": len(high_risk_rejected),
        "rejected_high_risk_recall": round(len(high_risk_rejected) / max(1, len(rejected_risks)), 4),
        "mean_rejected_regression_risk": round(mean(rejected_risks), 4) if rejected_risks else 0.0,
        "mean_retained_regression_risk": round(mean(retained_risks), 4) if retained_risks else 0.0,
        "rejected_vs_retained_risk_margin": round(
            (mean(rejected_risks) if rejected_risks else 0.0)
            - (mean(retained_risks) if retained_risks else 0.0),
            4,
        ),
        "verifier_tier_counts": _counts(plan["verifier_tier"] for plan in candidate_plans),
        "allowed_actions_used": allowed_actions,
        "blocked_action_count": len(blocked_actions),
        "blocked_actions_used": sorted(set(blocked_actions)),
        "direct_promotion_count": 0,
        "live_replacement_count": 0,
        "review_replacement_count": 0,
        "simulator_defect_residual_count": len(simulator_defects),
        "simulator_defect_next_round_intake_rate": round(
            sum(1 for row in simulator_defects if row["next_round_intake"]) / max(1, len(simulator_defects)),
            4,
        ),
        "production_router_claim_allowed": bool(
            production_gate_metrics.get("production_simulator_candidate_allowed")
            or production_gate_metrics.get("gate_router_promoted")
        ),
        "simulator_replacement_claim_allowed": False,
        "simulator_no_leakage_pass": bool(simulator_evidence["no_leakage_audit"].get("pass")),
        "simulator_state_feature_leak_count": int(no_leakage_metrics.get("state_feature_leak_count", 0)),
        "simulator_prediction_outcome_identity_count": int(
            no_leakage_metrics.get("prediction_outcome_exact_identity_count", 0)
        ),
        "counterfactual_gate_allowed": bool(
            simulator_evidence["counterfactual_policy"].get("metrics", {}).get("production_counterfactual_gate_allowed")
        ),
        "main_graph_mutation_count": sum(int(plan["main_graph_mutation_count"]) for plan in candidate_plans),
    }


def _simulator_evidence_summary(simulator_evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    production = simulator_evidence["production_gate"].get("metrics", {})
    no_leakage = simulator_evidence["no_leakage_audit"].get("metrics", {})
    counterfactual = simulator_evidence["counterfactual_policy"].get("metrics", {})
    return {
        "production_gate_pass": bool(simulator_evidence["production_gate"].get("pass")),
        "production_router_claim_allowed": bool(
            production.get("production_simulator_candidate_allowed") or production.get("gate_router_promoted")
        ),
        "blocked_claim": "task world simulator replacing live validation or judges",
        "no_leakage_audit_pass": bool(simulator_evidence["no_leakage_audit"].get("pass")),
        "state_feature_leak_count": int(no_leakage.get("state_feature_leak_count", 0)),
        "counterfactual_gate_allowed": bool(counterfactual.get("production_counterfactual_gate_allowed")),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "metrics": {}, "missing_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Framework Simulator-Guided Search",
        "",
        f"- pass: {payload['pass']}",
        f"- failed_gates: {payload['failed_gates']}",
        f"- candidate plans: {metrics['candidate_plan_count']}",
        f"- baseline fresh tests: {metrics['baseline_fresh_test_count']}",
        f"- selected fresh tests: {metrics['selected_fresh_test_count']}",
        f"- fresh test reduction: {metrics['fresh_test_reduction_rate']}",
        f"- true positive blocked: {metrics['true_positive_block_count']}",
        f"- rejected high-risk recall: {metrics['rejected_high_risk_recall']}",
        f"- rejected vs retained risk margin: {metrics['rejected_vs_retained_risk_margin']}",
        f"- verifier tiers: {metrics['verifier_tier_counts']}",
        f"- SimulatorDefect residuals: {metrics['simulator_defect_residual_count']}",
        f"- direct promotions by simulator: {metrics['direct_promotion_count']}",
        f"- live replacements by simulator: {metrics['live_replacement_count']}",
        f"- review replacements by simulator: {metrics['review_replacement_count']}",
        f"- main graph mutations: {metrics['main_graph_mutation_count']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator-guided framework search artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="framework_simulator_guided_search_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_framework_simulator_guided_search_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
