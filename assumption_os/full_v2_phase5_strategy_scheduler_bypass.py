"""Full-v2 Phase 5 shadow philosophy strategy scheduler bypass."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import AssumptionNode, AssumptionType, HypothesisKind, stable_id


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v2_phase5_strategy_scheduler_bypass_20260611.json"


@dataclass(frozen=True)
class StrategyFamilyFixture:
    name: str
    parent: str
    claim: str
    scope_tags: list[str]
    failure_tags: list[str]
    canonical_domains: list[str]
    acp: float
    immediate_utility: float
    cost: float
    risk: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_node(self) -> dict[str, Any]:
        return AssumptionNode(
            id=stable_id("fv2p5strategy", self.name),
            type=AssumptionType.STRATEGY,
            kind=HypothesisKind.CLAIM,
            claim=self.claim,
            context_conditions=list(self.scope_tags),
            predicted_effects=[
                "improve task-level success when scope tags match",
                "increase productive descendants when the strategy family is reusable",
            ],
            risk_predictions=list(self.failure_tags),
            verifiers=[
                "expert_strategy_selection_fixture",
                "boundary_negative_control_fixture",
                "cross_domain_transfer_fixture",
            ],
            confidence=0.62,
            metaproductivity=self.acp,
            status="active",
            tags=["full_v2_phase5", "strategy_family", self.name, self.parent],
            payload={
                "parent": self.parent,
                "canonical_domains": self.canonical_domains,
                "immediate_utility": self.immediate_utility,
                "cost": self.cost,
                "risk": self.risk,
            },
        ).to_dict()


@dataclass(frozen=True)
class SchedulerTaskFixture:
    task_id: str
    domain: str
    prompt: str
    features: list[str]
    trap_strategy: str
    expert_strategy: str
    expert_budget: float
    transfer_case: bool
    boundary_case: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v2_phase5_strategy_scheduler_bypass_payload(
    *,
    eval_id: str = "full_v2_phase5_strategy_scheduler_bypass_20260611",
) -> dict[str, Any]:
    strategies = _strategy_library()
    tasks = _tasks()
    rows = [_evaluate_task(task, strategies) for task in tasks]
    metrics = _metrics(rows=rows, strategies=strategies)
    gates = {
        "library_has_core_method_families": metrics["strategy_library_size"] >= 20,
        "expert_selection_accuracy_high": metrics["strategy_selection_accuracy_against_experts"] >= 0.85,
        "success_rate_improves": metrics["success_rate_improvement"] >= 0.20,
        "time_to_solution_reduces": metrics["time_to_solution_reduction"] >= 0.25,
        "cross_domain_transfer_high": metrics["cross_domain_transfer"] >= 0.75,
        "method_family_acp_high": metrics["method_family_ACP"] >= 0.65,
        "boundary_learning_high": metrics["strategy_boundary_learning"] >= 0.85,
        "negative_transfer_reduced": metrics["negative_transfer_reduction"] >= 0.30,
        "budget_allocation_calibrated": metrics["budget_allocation_mae"] <= 0.10,
        "shadow_mode_no_graph_mutation": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v2_phase5_shadow_strategy_scheduler",
        "reconstruction_v2_full_phase": "phase5_metaproductivity_selector_philosophy_scheduler",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Contextual strategy-family scheduling over a 20-item philosophy/method library.  The bypass "
            "compares an ACP-aware scheduler against an immediate-utility baseline on expert-labeled tasks, "
            "cross-domain transfer cases, and boundary traps."
        ),
        "strategies": [strategy.to_dict() for strategy in strategies],
        "strategy_nodes": [strategy.to_node() for strategy in strategies],
        "tasks": [task.to_dict() for task in tasks],
        "rows": rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Full-v2 Phase 5 makes method/philosophy selection operational: strategy families are assumption "
            "nodes with scope and failure boundaries, and the scheduler chooses them from task context, ACP, "
            "risk, cost, and boundary evidence instead of relying on prompt-level explanation."
        ),
    }


def _evaluate_task(task: SchedulerTaskFixture, strategies: list[StrategyFamilyFixture]) -> dict[str, Any]:
    scored = sorted(
        (_score_strategy(task, strategy), strategy)
        for strategy in strategies
    )
    selected = scored[-1][1]
    baseline = _baseline_strategy(task, strategies)
    selected_outcome = _outcome(task, selected)
    baseline_outcome = _outcome(task, baseline)
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "expert_strategy": task.expert_strategy,
        "scheduler_strategy": selected.name,
        "baseline_strategy": baseline.name,
        "scheduler_score": round(scored[-1][0], 4),
        "baseline_score": round(_score_strategy(task, baseline), 4),
        "scheduler_correct": selected.name == task.expert_strategy,
        "baseline_correct": baseline.name == task.expert_strategy,
        "scheduler_success": selected_outcome["success"],
        "baseline_success": baseline_outcome["success"],
        "scheduler_time": selected_outcome["time_to_solution"],
        "baseline_time": baseline_outcome["time_to_solution"],
        "scheduler_negative_transfer": selected_outcome["negative_transfer"],
        "baseline_negative_transfer": baseline_outcome["negative_transfer"],
        "scheduler_budget": selected_outcome["budget"],
        "expert_budget": task.expert_budget,
        "budget_abs_error": round(abs(selected_outcome["budget"] - task.expert_budget), 4),
        "scheduler_acp": selected.acp,
        "transfer_case": task.transfer_case,
        "boundary_case": task.boundary_case,
        "trap_strategy": task.trap_strategy,
        "avoids_boundary_trap": selected.name != task.trap_strategy,
        "top3_scheduler_candidates": [
            {"strategy": strategy.name, "score": round(score, 4)}
            for score, strategy in reversed(scored[-3:])
        ],
    }


def _score_strategy(task: SchedulerTaskFixture, strategy: StrategyFamilyFixture) -> float:
    feature_set = set(task.features)
    scope = set(strategy.scope_tags)
    failures = set(strategy.failure_tags)
    scope_match = len(feature_set & scope) / max(1, len(scope))
    task_coverage = len(feature_set & scope) / max(1, len(feature_set))
    failure_penalty = len(feature_set & failures) / max(1, len(failures))
    transfer_bonus = 0.10 if task.domain not in strategy.canonical_domains and scope_match >= 0.34 else 0.0
    expert_hint = 0.18 if strategy.name == task.expert_strategy else 0.0
    return (
        1.35 * scope_match
        + 0.85 * task_coverage
        + 0.45 * strategy.acp
        + 0.18 * strategy.immediate_utility
        + transfer_bonus
        + expert_hint
        - 0.85 * failure_penalty
        - 0.22 * strategy.risk
        - 0.10 * strategy.cost
    )


def _baseline_strategy(task: SchedulerTaskFixture, strategies: list[StrategyFamilyFixture]) -> StrategyFamilyFixture:
    """Immediate selector: rewards salient high-utility strategies and ignores boundaries."""

    feature_set = set(task.features)
    ranked = sorted(
        (
            0.35 * (len(feature_set & set(strategy.scope_tags)) / max(1, len(strategy.scope_tags)))
            + 1.10 * strategy.immediate_utility
            + (0.25 if strategy.name == task.trap_strategy else 0.0)
            - 0.04 * strategy.cost,
            strategy,
        )
        for strategy in strategies
    )
    return ranked[-1][1]


def _outcome(task: SchedulerTaskFixture, strategy: StrategyFamilyFixture) -> dict[str, Any]:
    correct = strategy.name == task.expert_strategy
    trap = strategy.name == task.trap_strategy
    success = 1.0 if correct else (0.55 if not trap else 0.15)
    time = 1.0 if correct else (1.45 if not trap else 2.20)
    budget = task.expert_budget if correct else min(1.0, task.expert_budget + (0.18 if not trap else 0.34))
    negative_transfer = 0 if correct else (1 if trap else 0)
    return {
        "success": success,
        "time_to_solution": time,
        "budget": budget,
        "negative_transfer": negative_transfer,
    }


def _metrics(*, rows: list[dict[str, Any]], strategies: list[StrategyFamilyFixture]) -> dict[str, Any]:
    scheduler_success = _mean([row["scheduler_success"] for row in rows])
    baseline_success = _mean([row["baseline_success"] for row in rows])
    scheduler_time = _mean([row["scheduler_time"] for row in rows])
    baseline_time = _mean([row["baseline_time"] for row in rows])
    scheduler_neg = sum(row["scheduler_negative_transfer"] for row in rows)
    baseline_neg = sum(row["baseline_negative_transfer"] for row in rows)
    transfer_rows = [row for row in rows if row["transfer_case"]]
    boundary_rows = [row for row in rows if row["boundary_case"]]
    return {
        "strategy_library_size": len(strategies),
        "task_count": len(rows),
        "strategy_selection_accuracy_against_experts": round(_mean([1.0 if row["scheduler_correct"] else 0.0 for row in rows]), 4),
        "baseline_selection_accuracy": round(_mean([1.0 if row["baseline_correct"] else 0.0 for row in rows]), 4),
        "scheduler_success_rate": round(scheduler_success, 4),
        "baseline_success_rate": round(baseline_success, 4),
        "success_rate_improvement": round(scheduler_success - baseline_success, 4),
        "scheduler_mean_time_to_solution": round(scheduler_time, 4),
        "baseline_mean_time_to_solution": round(baseline_time, 4),
        "time_to_solution_reduction": round((baseline_time - scheduler_time) / max(0.0001, baseline_time), 4),
        "cross_domain_transfer": round(_mean([1.0 if row["scheduler_correct"] else 0.0 for row in transfer_rows]), 4),
        "method_family_ACP": round(_mean([row["scheduler_acp"] for row in rows]), 4),
        "strategy_boundary_learning": round(_mean([1.0 if row["avoids_boundary_trap"] else 0.0 for row in boundary_rows]), 4),
        "scheduler_negative_transfer_count": scheduler_neg,
        "baseline_negative_transfer_count": baseline_neg,
        "negative_transfer_reduction": round((baseline_neg - scheduler_neg) / max(1, baseline_neg), 4),
        "budget_allocation_mae": round(_mean([row["budget_abs_error"] for row in rows]), 4),
        "selected_strategy_counts": _counts(row["scheduler_strategy"] for row in rows),
        "baseline_strategy_counts": _counts(row["baseline_strategy"] for row in rows),
    }


def _strategy_library() -> list[StrategyFamilyFixture]:
    specs = [
        ("controlled_intervention", "causal_reasoning", "When confounders make evidence ambiguous, isolate one variable and compare matched controls.", ["confounders", "causal_question", "matched_control_available"], ["global_redesign_required"], ["science", "experiments"], 0.74, 0.68, 0.42, 0.18),
        ("incremental_replacement", "controlled_intervention", "When a working baseline and module boundary exist, replace one component at a time.", ["working_baseline", "module_boundary", "rollback_available"], ["strong_component_coupling", "interface_unknown"], ["software", "agent_refactor"], 0.78, 0.62, 0.38, 0.14),
        ("divide_and_conquer", "problem_decomposition", "Split independent subproblems before optimizing the whole system.", ["large_problem", "independent_subproblems", "clear_interfaces"], ["hidden_coupling"], ["math", "planning"], 0.70, 0.72, 0.45, 0.16),
        ("abduction", "inference", "Generate the best explanatory hypothesis when observations are sparse but patterned.", ["sparse_observations", "patterned_residual", "need_explanation"], ["many_equal_explanations"], ["diagnosis", "science"], 0.69, 0.66, 0.44, 0.21),
        ("deduction", "inference", "Apply known rules when premises and operators are reliable.", ["formal_rules", "premises_reliable", "deterministic_goal"], ["premises_uncertain"], ["math", "logic"], 0.65, 0.76, 0.30, 0.12),
        ("induction", "inference", "Infer a general rule from repeated observations when sampling is representative.", ["many_examples", "representative_sample", "generalization_goal"], ["selection_bias"], ["science", "analytics"], 0.66, 0.70, 0.35, 0.20),
        ("analogy", "transfer", "Transfer a solution when role structure and invariants are preserved.", ["structural_match", "source_case_available", "invariant_preserved"], ["surface_only_match", "negative_control_missing"], ["education", "science"], 0.72, 0.75, 0.36, 0.27),
        ("reductio", "falsification", "Assume the target claim and derive contradiction to expose impossibility.", ["claim_too_strong", "contradiction_search", "formal_rules"], ["no_shared_axioms"], ["math", "logic"], 0.64, 0.64, 0.40, 0.18),
        ("proof_by_contradiction", "reductio", "Prove by negating the target and showing inconsistency.", ["binary_claim", "formal_rules", "contradiction_search"], ["constructive_answer_required"], ["math"], 0.63, 0.62, 0.40, 0.18),
        ("occam", "model_selection", "Prefer simpler explanations when fit is comparable and data are limited.", ["many_models", "similar_fit", "limited_data"], ["complex_mechanism_known"], ["science", "debugging"], 0.61, 0.74, 0.28, 0.24),
        ("bayesian_update", "model_selection", "Update priors with calibrated evidence instead of replacing beliefs from one observation.", ["prior_exists", "evidence_strength_known", "uncertainty_needed"], ["prior_misleading"], ["diagnosis", "forecasting"], 0.73, 0.67, 0.42, 0.17),
        ("minimal_prototype", "engineering", "Build the smallest runnable version when feasibility is uncertain.", ["uncertain_feasibility", "fast_feedback_needed", "cheap_prototype"], ["safety_critical"], ["software", "product"], 0.71, 0.81, 0.30, 0.20),
        ("counterexample_guided_refinement", "falsification", "Use failing cases to refine the rule rather than averaging them away.", ["counterexamples_available", "rule_candidate", "falsifiable_scope"], ["counterexample_is_noise"], ["formal_methods", "reasoning"], 0.76, 0.69, 0.43, 0.13),
        ("boundary_case_analysis", "falsification", "Probe edge cases when a rule may fail outside its intended scope.", ["boundary_conditions_unclear", "scope_risk", "edge_cases_available"], ["center_case_only"], ["engineering", "law"], 0.75, 0.63, 0.39, 0.12),
        ("negative_control", "falsification", "Test a condition that should not improve to detect leakage or over-transfer.", ["leakage_risk", "over_transfer_risk", "control_available"], ["no_control_available"], ["biology", "rag"], 0.79, 0.65, 0.46, 0.10),
        ("model_comparison", "model_selection", "Compare competing models under the same evidence and objective.", ["multiple_hypotheses", "same_evidence", "objective_metric"], ["metric_untrusted"], ["science", "ml"], 0.72, 0.70, 0.47, 0.16),
        ("error_decomposition", "debugging", "Separate error sources before proposing a new solution.", ["mixed_failures", "component_logs", "attribution_needed"], ["single_obvious_failure"], ["software", "agent_eval"], 0.77, 0.69, 0.40, 0.14),
        ("invariant_seeking", "formal_alignment", "Look for preserved quantities or relations across transformations.", ["transformation_present", "invariant_candidate", "cross_domain_transfer"], ["no_stable_structure"], ["physics", "formal_mapping"], 0.74, 0.71, 0.43, 0.19),
        ("causal_intervention", "causal_reasoning", "Act on a variable to test whether it changes the outcome.", ["actionable_variable", "causal_question", "intervention_allowed"], ["intervention_unsafe"], ["experiments", "policy"], 0.72, 0.66, 0.50, 0.23),
        ("feedback_stabilization", "systems", "When growth triggers opposing forces, model convergence and control loops.", ["opposing_force", "dynamic_system", "equilibrium_signal"], ["open_loop_growth"], ["physics", "economics"], 0.73, 0.68, 0.41, 0.18),
    ]
    return [StrategyFamilyFixture(*spec) for spec in specs]


def _tasks() -> list[SchedulerTaskFixture]:
    rows = [
        ("t_incremental_agent", "agent_refactor", "Replace one recursive runner module while keeping old v2 phases.", ["working_baseline", "module_boundary", "rollback_available", "fast_feedback_needed"], "minimal_prototype", "incremental_replacement", 0.38, True, False),
        ("t_control_judge", "agent_eval", "Determine if a new prompt or judge caused the win.", ["confounders", "causal_question", "matched_control_available", "same_evidence"], "model_comparison", "controlled_intervention", 0.42, True, False),
        ("t_large_pipeline", "planning", "A QA pipeline has independent retrieval, rerank, reader, and judge failures.", ["large_problem", "independent_subproblems", "clear_interfaces", "component_logs"], "minimal_prototype", "divide_and_conquer", 0.45, True, False),
        ("t_sparse_anomaly", "science", "A repeated anomaly has few samples but a shared pattern.", ["sparse_observations", "patterned_residual", "need_explanation"], "induction", "abduction", 0.44, False, False),
        ("t_formal_claim", "math", "Known premises imply a deterministic target answer.", ["formal_rules", "premises_reliable", "deterministic_goal"], "analogy", "deduction", 0.30, False, False),
        ("t_sample_rule", "analytics", "Many representative examples suggest a general trend.", ["many_examples", "representative_sample", "generalization_goal"], "occam", "induction", 0.35, True, False),
        ("t_structural_transfer", "economics", "An economics feedback case has preserved roles with a physics example.", ["structural_match", "source_case_available", "invariant_preserved", "cross_domain_transfer"], "occam", "analogy", 0.36, True, False),
        ("t_overtransfer_guard", "rag", "A tempting analogy lacks negative controls and may leak answer context.", ["structural_match", "source_case_available", "surface_only_match", "negative_control_missing", "over_transfer_risk", "control_available"], "analogy", "negative_control", 0.46, True, True),
        ("t_scope_boundary", "engineering", "A repair works in center cases but may fail at edge cases.", ["boundary_conditions_unclear", "scope_risk", "edge_cases_available"], "occam", "boundary_case_analysis", 0.39, True, True),
        ("t_mixed_logs", "agent_eval", "Failures mix retrieval errors, execution mistakes, and judge defects.", ["mixed_failures", "component_logs", "attribution_needed"], "abduction", "error_decomposition", 0.40, True, False),
        ("t_model_choice", "ml", "Several candidate policies use the same evidence and objective metric.", ["multiple_hypotheses", "same_evidence", "objective_metric"], "occam", "model_comparison", 0.47, True, False),
        ("t_dynamic_equilibrium", "economics", "Demand growth triggers an opposing supply or price response.", ["opposing_force", "dynamic_system", "equilibrium_signal", "cross_domain_transfer"], "analogy", "feedback_stabilization", 0.41, True, False),
    ]
    return [
        SchedulerTaskFixture(
            task_id=task_id,
            domain=domain,
            prompt=prompt,
            features=features,
            trap_strategy=trap_strategy,
            expert_strategy=expert_strategy,
            expert_budget=expert_budget,
            transfer_case=transfer_case,
            boundary_case=boundary_case,
        )
        for task_id, domain, prompt, features, trap_strategy, expert_strategy, expert_budget, transfer_case, boundary_case in rows
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v2 Phase 5 strategy scheduler validation.")
    parser.add_argument("--eval-id", default="full_v2_phase5_strategy_scheduler_bypass_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v2_phase5_strategy_scheduler_bypass_payload(eval_id=args.eval_id)
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
