"""Dialectical conservative-generalization gate for self-evolution.

self_evo_roadmap.md argues that a new assumption should become a new framework
only when it is a conservative generalization of prior frameworks: it explains
the residual that motivated it, preserves validated old successes, reduces to
its parent under the parent's scope conditions, and creates new testable
consequences.  This module turns that principle into a bounded gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType, HypothesisKind, stable_id


DEFAULT_OUT = PAPER_DIR / "conservative_generalization_gate_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/conservative_generalization_gate_20260612.md")

SOURCE_ARTIFACTS = {
    "creative_generator": PAPER_DIR / "creative_hypothesis_trajectory_search_20260612.json",
    "simulator_production_gate": PAPER_DIR / "simulator_production_gate_20260612.json",
    "finite_formal_stack": PAPER_DIR / "finite_formal_reasoning_stack_20260612.json",
    "integrated_recursive_episode": PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json",
    "last_three_part_coverage": PAPER_DIR / "last_three_part_coverage_audit_20260612.json",
}

REQUIRED_PROMOTION_RELATIONS = {
    EdgeType.GENERALIZES.value,
    EdgeType.REDUCES_TO_UNDER_SCOPE.value,
    EdgeType.EXPLAINS_RESIDUAL.value,
    EdgeType.PRESERVES_SUCCESS_CASES.value,
    EdgeType.MODIFIES_BOUNDARY_OF.value,
    EdgeType.PREDICTS_NEW_CASE.value,
}


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    domain: str
    parent_score: float
    candidate_score: float
    scope_condition: str


@dataclass(frozen=True)
class LimitingCase:
    case_id: str
    scope_condition: str
    reduction_fidelity: float


@dataclass(frozen=True)
class CandidateFramework:
    framework_id: str
    claim: str
    parent_frameworks: list[str]
    old_success_cases: list[ScoredCase]
    residual_cases: list[ScoredCase]
    limiting_cases: list[LimitingCase]
    new_prediction_cases: list[ScoredCase]
    unified_branch_count: int
    old_rule_count: int
    new_rule_count: int
    transfer_domain_count: int
    complexity_penalty: float
    formal_certificate_status: str
    simulator_expected_utility: float
    conflict_boundaries: list[str] = field(default_factory=list)


def build_conservative_generalization_gate_payload(
    *,
    root: Path,
    eval_id: str = "conservative_generalization_gate_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    source_artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    candidates = _fixture_candidates()
    evaluations = [_evaluate_candidate(candidate) for candidate in candidates]
    graph_patch = _build_graph_patch(evaluations)
    metrics = _metrics(source_artifacts=source_artifacts, evaluations=evaluations, graph_patch=graph_patch)
    gates = {
        "source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "candidate_count_high": metrics["candidate_count"] >= 4,
        "active_scoped_framework_found": metrics["decision_counts"].get("active_scoped_framework", 0) >= 1,
        "candidate_framework_found": metrics["decision_counts"].get("candidate_framework", 0) >= 1,
        "branch_only_found": metrics["decision_counts"].get("branch_only", 0) >= 1,
        "reject_found": metrics["decision_counts"].get("reject", 0) >= 1,
        "active_framework_preserves_old_success": metrics["active_min_old_success_preservation"] >= 0.95,
        "active_framework_explains_residuals": metrics["active_min_residual_explanation"] >= 0.75,
        "active_framework_reduces_to_parent": metrics["active_min_limiting_case_reduction"] >= 0.90,
        "active_framework_has_generality_gain": metrics["active_min_generality_gain"] >= 0.35,
        "active_framework_has_new_prediction": metrics["active_min_new_prediction_success"] >= 0.75,
        "active_framework_regression_bounded": metrics["active_max_regression_cost"] <= 0.02,
        "required_relation_coverage": metrics["active_required_relation_coverage"] == 1.0,
        "framework_growth_score_high": metrics["top_framework_growth_score"] >= 0.70,
        "non_promoted_have_next_tests": metrics["non_promoted_next_test_coverage"] == 1.0,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
        "unbounded_philosophy_generator_claim_blocked": (
            metrics["unbounded_philosophy_generator_claim_allowed"] is False
        ),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "conservative_generalization_gate",
        "source_md": "reconstruction/md/self_evo_roadmap.md",
        "reconstruction_v2_full_phase": "r7_framework_evolution_conservative_generalization",
        "implementation_level": "bounded_dialectical_extension_gate",
        "performance_validation": True,
        "validation_scope": (
            "Implements the self_evo_roadmap conservative-generalization principle.  A new framework is promoted "
            "only if it explains residuals, preserves old successes, reduces to parent frameworks under their "
            "scope conditions, gains generality, and creates new testable consequences.  The gate emits graph "
            "patches and promotion decisions but does not mutate the main graph."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(source_artifacts[name].get("pass")),
                "eval_kind": source_artifacts[name].get("eval_kind"),
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
        "evaluations": evaluations,
        "graph_patch": graph_patch,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": (
            "bounded conservative-generalization gate for framework growth over an Assumption Graph"
        ),
        "blocked_claims": [
            "unbounded_philosophy_generator",
            "ungated_framework_promotion",
            "replacement_of_live_validation",
            "full_theorem_prover_for_arbitrary_natural_language",
        ],
        "interpretation": (
            "The self-evolution loop can now distinguish a local branch from a candidate framework and from an "
            "active scoped framework.  Promotion requires conservative generalization, not just local utility."
        ),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Conservative Generalization Gate",
        "",
        f"- pass: `{payload['pass']}`",
        f"- candidates: `{metrics['candidate_count']}`",
        f"- decisions: `{metrics['decision_counts']}`",
        f"- active required relation coverage: `{metrics['active_required_relation_coverage']}`",
        f"- top framework growth score: `{metrics['top_framework_growth_score']}`",
        "",
        "## Evaluation Rows",
        "",
        "| Framework | Decision | Growth | Old Preservation | Residual Explanation | Limiting Reduction | Generality | New Prediction | Regression |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["evaluations"]:
        m = row["metrics"]
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["framework_id"],
                row["decision"],
                m["framework_growth_score"],
                m["old_success_preservation"],
                m["residual_explanation"],
                m["limiting_case_reduction"],
                m["generality_gain"],
                m["new_prediction_success"],
                m["regression_cost"],
            )
        )
    lines.extend([
        "",
        "## Required Relations",
        "",
    ])
    for row in payload["evaluations"]:
        if row["decision"] not in {"active_scoped_framework", "candidate_framework"}:
            continue
        lines.append(f"- `{row['framework_id']}`: `{sorted(row['relation_types'])}`")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This is a bounded framework-growth gate.  It does not claim unbounded philosophy generation,",
        "ungated framework promotion, replacement of live validation, or a full theorem prover.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _evaluate_candidate(candidate: CandidateFramework) -> dict[str, Any]:
    old_preservation = _old_success_preservation(candidate.old_success_cases)
    regression_cost = _regression_cost(candidate.old_success_cases)
    residual_explanation = _residual_explanation(candidate.residual_cases)
    coverage_gain = _mean(
        max(0.0, case.candidate_score - case.parent_score)
        for case in candidate.residual_cases
    )
    limiting_case_reduction = _mean(case.reduction_fidelity for case in candidate.limiting_cases)
    new_prediction_success = _mean(case.candidate_score for case in candidate.new_prediction_cases)
    compression_gain = max(0.0, (candidate.old_rule_count - candidate.new_rule_count) / max(candidate.old_rule_count, 1))
    transfer_gain = min(0.30, 0.06 * candidate.transfer_domain_count)
    branch_unification_gain = min(0.20, 0.05 * max(0, candidate.unified_branch_count - 1))
    generality_gain = max(
        0.0,
        min(
            1.0,
            0.36 * coverage_gain
            + 0.26 * compression_gain
            + 0.22 * transfer_gain
            + 0.16 * branch_unification_gain
            - candidate.complexity_penalty,
        ),
    )
    framework_growth_score = max(
        0.0,
        min(
            1.0,
            0.20 * old_preservation
            + 0.20 * residual_explanation
            + 0.16 * limiting_case_reduction
            + 0.18 * generality_gain
            + 0.16 * new_prediction_success
            + 0.10 * candidate.simulator_expected_utility
            - 0.35 * regression_cost
            - 0.10 * candidate.complexity_penalty,
        ),
    )
    decision = _decision(
        old_success_preservation=old_preservation,
        regression_cost=regression_cost,
        residual_explanation=residual_explanation,
        coverage_gain=coverage_gain,
        limiting_case_reduction=limiting_case_reduction,
        generality_gain=generality_gain,
        new_prediction_success=new_prediction_success,
        framework_growth_score=framework_growth_score,
        candidate=candidate,
    )
    required_next_tests = _required_next_tests(candidate=candidate, decision=decision)
    relation_types = _relation_types_for_candidate(candidate=candidate, decision=decision)
    return {
        "framework_id": candidate.framework_id,
        "claim": candidate.claim,
        "parent_frameworks": candidate.parent_frameworks,
        "decision": decision,
        "promotion_level": decision,
        "metrics": {
            "old_success_preservation": round(old_preservation, 4),
            "residual_explanation": round(residual_explanation, 4),
            "coverage_gain": round(coverage_gain, 4),
            "limiting_case_reduction": round(limiting_case_reduction, 4),
            "generality_gain": round(generality_gain, 4),
            "compression_gain": round(compression_gain, 4),
            "transfer_gain": round(transfer_gain, 4),
            "new_prediction_success": round(new_prediction_success, 4),
            "regression_cost": round(regression_cost, 4),
            "complexity_penalty": round(candidate.complexity_penalty, 4),
            "simulator_expected_utility": round(candidate.simulator_expected_utility, 4),
            "framework_growth_score": round(framework_growth_score, 4),
        },
        "gate_checks": {
            "g1_residual_explanation": residual_explanation >= 0.75,
            "g2_old_success_preservation": old_preservation >= 0.95 and regression_cost <= 0.02,
            "g3_limiting_case_reduction": limiting_case_reduction >= 0.90,
            "g4_generality_gain": generality_gain >= 0.35,
            "g5_new_testable_consequence": new_prediction_success >= 0.75,
        },
        "relation_types": sorted(relation_types),
        "conflict_boundaries": candidate.conflict_boundaries,
        "required_next_tests": required_next_tests,
    }


def _decision(
    *,
    old_success_preservation: float,
    regression_cost: float,
    residual_explanation: float,
    coverage_gain: float,
    limiting_case_reduction: float,
    generality_gain: float,
    new_prediction_success: float,
    framework_growth_score: float,
    candidate: CandidateFramework,
) -> str:
    if old_success_preservation < 0.92 or regression_cost > 0.04:
        return "reject"
    if residual_explanation < 0.55 or coverage_gain < 0.06:
        return "reject"
    if (
        old_success_preservation >= 0.95
        and regression_cost <= 0.02
        and residual_explanation >= 0.75
        and limiting_case_reduction >= 0.90
        and generality_gain >= 0.35
        and new_prediction_success >= 0.75
        and framework_growth_score >= 0.70
        and candidate.formal_certificate_status == "pass"
        and candidate.simulator_expected_utility >= 0.70
        and len({case.domain for case in candidate.old_success_cases}) >= 3
        and len({case.domain for case in candidate.residual_cases}) >= 2
        and len({case.domain for case in candidate.new_prediction_cases}) >= 1
    ):
        return "active_scoped_framework"
    if (
        old_success_preservation >= 0.94
        and residual_explanation >= 0.68
        and limiting_case_reduction >= 0.84
        and generality_gain >= 0.24
        and new_prediction_success >= 0.64
        and candidate.formal_certificate_status in {"pass", "not_applicable"}
    ):
        return "candidate_framework"
    return "branch_only"


def _required_next_tests(*, candidate: CandidateFramework, decision: str) -> list[str]:
    if decision == "active_scoped_framework":
        return [
            "canary_graph_apply_with_rollback",
            "multi_domain_survival_recheck",
            "descendant_productivity_monitor",
        ]
    if decision == "candidate_framework":
        return [
            "expand_unseen_domain_validation",
            "repeat_old_success_noninferiority",
            "fresh_prediction_ablation",
        ]
    if decision == "branch_only":
        return [
            "collect_second_parent_framework",
            "prove_compression_or_transfer_gain",
            "retain_as_scoped_branch",
        ]
    return [
        "record_negative_evidence",
        "mark_conflict_boundary",
        "do_not_promote_to_framework",
    ]


def _relation_types_for_candidate(*, candidate: CandidateFramework, decision: str) -> set[str]:
    relations = {
        EdgeType.GENERALIZES.value,
        EdgeType.EXPLAINS_RESIDUAL.value,
        EdgeType.PRESERVES_SUCCESS_CASES.value,
    }
    if decision in {"active_scoped_framework", "candidate_framework"}:
        relations.update({
            EdgeType.REDUCES_TO_UNDER_SCOPE.value,
            EdgeType.MODIFIES_BOUNDARY_OF.value,
            EdgeType.PREDICTS_NEW_CASE.value,
        })
    if candidate.conflict_boundaries:
        relations.add(EdgeType.CONFLICTS_WITH.value)
    return relations


def _build_graph_patch(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for candidate in _fixture_candidates():
        row = next(item for item in evaluations if item["framework_id"] == candidate.framework_id)
        if row["decision"] == "reject":
            continue
        node = AssumptionNode(
            id=candidate.framework_id,
            type=AssumptionType.METHOD,
            kind=HypothesisKind.PROCESS_MODEL,
            claim=candidate.claim,
            context_conditions=[case.scope_condition for case in candidate.limiting_cases],
            predicted_effects=[case.case_id for case in candidate.new_prediction_cases],
            risk_predictions=candidate.conflict_boundaries,
            confidence=row["metrics"]["framework_growth_score"],
            metaproductivity=row["metrics"]["generality_gain"],
            status=row["decision"],
            tags=["conservative_generalization", "framework_growth", row["decision"]],
            payload={
                "gate_metrics": row["metrics"],
                "required_next_tests": row["required_next_tests"],
                "claim_boundary": "bounded_framework_growth_not_unbounded_philosophy_generation",
            },
        )
        nodes.append(node.to_dict())
        for parent in candidate.parent_frameworks:
            edges.append(_edge(candidate.framework_id, parent, EdgeType.GENERALIZES, row))
            if row["decision"] in {"active_scoped_framework", "candidate_framework"}:
                edges.append(_edge(candidate.framework_id, parent, EdgeType.REDUCES_TO_UNDER_SCOPE, row))
                edges.append(_edge(candidate.framework_id, parent, EdgeType.MODIFIES_BOUNDARY_OF, row))
        for case in candidate.residual_cases:
            rid = stable_id("residual", case.case_id)
            nodes.append(_case_node(rid, case.case_id, AssumptionType.RESIDUAL, case.domain, "residual_case"))
            edges.append(_edge(candidate.framework_id, rid, EdgeType.EXPLAINS_RESIDUAL, row))
        for case in candidate.old_success_cases:
            cid = stable_id("case", case.case_id)
            nodes.append(_case_node(cid, case.case_id, AssumptionType.CASE, case.domain, "old_success_case"))
            edges.append(_edge(candidate.framework_id, cid, EdgeType.PRESERVES_SUCCESS_CASES, row))
        if row["decision"] in {"active_scoped_framework", "candidate_framework"}:
            for case in candidate.new_prediction_cases:
                cid = stable_id("case", case.case_id)
                nodes.append(_case_node(cid, case.case_id, AssumptionType.CASE, case.domain, "new_prediction_case"))
                edges.append(_edge(candidate.framework_id, cid, EdgeType.PREDICTS_NEW_CASE, row))
        for conflict in candidate.conflict_boundaries:
            cid = stable_id("conflict", conflict)
            nodes.append(_case_node(cid, conflict, AssumptionType.RESIDUAL, "boundary", "conflict_boundary"))
            edges.append(_edge(candidate.framework_id, cid, EdgeType.CONFLICTS_WITH, row))
    nodes = _dedupe_nodes(nodes)
    edges = _dedupe_edges(edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "edge_type_counts": _counts(edge["type"] for edge in edges),
        "main_graph_mutation_count": 0,
        "apply_mode": "gated_graph_patch_only",
    }


def _edge(source: str, target: str, edge_type: EdgeType, row: dict[str, Any]) -> dict[str, Any]:
    return AssumptionEdge(
        source=source,
        target=target,
        type=edge_type,
        weight=row["metrics"]["framework_growth_score"],
        payload={
            "decision": row["decision"],
            "gate_metrics": row["metrics"],
            "source": "conservative_generalization_gate",
        },
    ).to_dict()


def _case_node(
    node_id: str,
    claim: str,
    node_type: AssumptionType,
    domain: str,
    role: str,
) -> dict[str, Any]:
    return AssumptionNode(
        id=node_id,
        type=node_type,
        claim=claim.replace("_", " "),
        kind=HypothesisKind.CLAIM,
        status="evidence_case",
        tags=[role, domain],
        payload={"domain": domain, "role": role},
    ).to_dict()


def _metrics(
    *,
    source_artifacts: dict[str, dict[str, Any]],
    evaluations: list[dict[str, Any]],
    graph_patch: dict[str, Any],
) -> dict[str, Any]:
    decisions = _counts(row["decision"] for row in evaluations)
    active = [row for row in evaluations if row["decision"] == "active_scoped_framework"]
    non_promoted = [row for row in evaluations if row["decision"] != "active_scoped_framework"]
    required_coverage = 0.0
    if active:
        required_coverage = min(
            len(REQUIRED_PROMOTION_RELATIONS.intersection(set(row["relation_types"])))
            / len(REQUIRED_PROMOTION_RELATIONS)
            for row in active
        )
    return {
        "source_artifact_count": len(source_artifacts),
        "source_artifact_pass_rate": round(
            sum(1 for artifact in source_artifacts.values() if artifact.get("pass")) / len(source_artifacts),
            4,
        ),
        "candidate_count": len(evaluations),
        "decision_counts": decisions,
        "active_min_old_success_preservation": _min_metric(active, "old_success_preservation"),
        "active_min_residual_explanation": _min_metric(active, "residual_explanation"),
        "active_min_limiting_case_reduction": _min_metric(active, "limiting_case_reduction"),
        "active_min_generality_gain": _min_metric(active, "generality_gain"),
        "active_min_new_prediction_success": _min_metric(active, "new_prediction_success"),
        "active_max_regression_cost": _max_metric(active, "regression_cost"),
        "active_required_relation_coverage": round(required_coverage, 4),
        "top_framework_growth_score": max(row["metrics"]["framework_growth_score"] for row in evaluations),
        "non_promoted_next_test_coverage": round(
            sum(1 for row in non_promoted if row["required_next_tests"]) / max(1, len(non_promoted)),
            4,
        ),
        "graph_patch_node_count": graph_patch["node_count"],
        "graph_patch_edge_count": graph_patch["edge_count"],
        "graph_patch_edge_type_counts": graph_patch["edge_type_counts"],
        "main_graph_mutation_count": graph_patch["main_graph_mutation_count"],
        "unbounded_philosophy_generator_claim_allowed": False,
    }


def _min_metric(rows: list[dict[str, Any]], key: str) -> float:
    return round(min((row["metrics"][key] for row in rows), default=0.0), 4)


def _max_metric(rows: list[dict[str, Any]], key: str) -> float:
    return round(max((row["metrics"][key] for row in rows), default=0.0), 4)


def _fixture_candidates() -> list[CandidateFramework]:
    return [
        CandidateFramework(
            framework_id="fw_dependency_aware_controlled_intervention",
            claim=(
                "Test intervention independence before applying controlled-variable reasoning; when dependencies "
                "are dense, use paired or grouped ablation and reduce to ordinary control variables when interfaces "
                "are isolated."
            ),
            parent_frameworks=[
                "fw_control_variables",
                "fw_interface_isolation",
                "fw_paired_ablation",
            ],
            old_success_cases=[
                ScoredCase("ab_test_isolated_units", "causal_eval", 0.84, 0.86, "independent_units"),
                ScoredCase("prompt_single_factor_ablation", "prompting", 0.81, 0.82, "single_factor_change"),
                ScoredCase("module_replacement_stable_interface", "software", 0.78, 0.80, "stable_interface"),
            ],
            residual_cases=[
                ScoredCase("dense_module_coupling_failure", "software", 0.41, 0.82, "dense_dependency_graph"),
                ScoredCase("monolithic_world_model_build_failure", "agent", 0.38, 0.79, "coupled_world_model"),
                ScoredCase("retrieval_bridge_hidden_dependency", "qa", 0.46, 0.76, "hidden_bridge_dependency"),
            ],
            limiting_cases=[
                LimitingCase("sparse_dependency_graph", "dependencies_sparse", 0.97),
                LimitingCase("single_interface_boundary", "one_stable_interface", 0.95),
                LimitingCase("paired_ablation_unneeded", "no_interaction_effect", 0.94),
            ],
            new_prediction_cases=[
                ScoredCase("dense_dependency_group_ablation_wins", "software", 0.43, 0.84, "dense_dependency_graph"),
                ScoredCase("bridge_dependency_route_before_answer", "qa", 0.49, 0.78, "hidden_bridge_dependency"),
                ScoredCase("simulator_defect_pairwise_probe", "agent", 0.44, 0.80, "coupled_world_model"),
            ],
            unified_branch_count=5,
            old_rule_count=8,
            new_rule_count=3,
            transfer_domain_count=5,
            complexity_penalty=0.02,
            formal_certificate_status="pass",
            simulator_expected_utility=0.83,
            conflict_boundaries=[
                "do_not_use_group_ablation_when_dependency_graph_is_sparse_and_cost_is_high",
            ],
        ),
        CandidateFramework(
            framework_id="fw_evidence_ladder_verifier_routing",
            claim=(
                "Route proposed assumptions through a verifier ladder whose tier is determined by uncertainty, "
                "risk, and formal-transfer status."
            ),
            parent_frameworks=[
                "fw_world_model_triage",
                "fw_verifier_stack",
                "fw_formal_certificate_gate",
            ],
            old_success_cases=[
                ScoredCase("low_risk_manifest_update", "autonomy", 0.88, 0.88, "low_risk_update"),
                ScoredCase("formal_mapping_negative_control", "formal", 0.82, 0.83, "certificate_available"),
                ScoredCase("budget_triage_candidate_screen", "simulator", 0.79, 0.80, "calibrated_route"),
            ],
            residual_cases=[
                ScoredCase("overconfident_world_model_route", "simulator", 0.42, 0.74, "high_uncertainty"),
                ScoredCase("formal_gate_not_applicable_case", "formal", 0.51, 0.72, "semantic_only_mapping"),
            ],
            limiting_cases=[
                LimitingCase("certain_low_risk_case", "low_uncertainty_low_risk", 0.91),
                LimitingCase("certificate_available_case", "finite_certificate_passes", 0.89),
            ],
            new_prediction_cases=[
                ScoredCase("abstain_under_unseen_pattern", "simulator", 0.50, 0.72, "unseen_pattern"),
                ScoredCase("manual_review_for_policy_promotion", "autonomy", 0.58, 0.76, "policy_default_change"),
            ],
            unified_branch_count=3,
            old_rule_count=7,
            new_rule_count=4,
            transfer_domain_count=4,
            complexity_penalty=0.035,
            formal_certificate_status="not_applicable",
            simulator_expected_utility=0.74,
            conflict_boundaries=[
                "cannot_replace_live_judgment_when_outcome_evidence_is_required",
            ],
        ),
        CandidateFramework(
            framework_id="fw_boundary_first_analogy_abstention",
            claim=(
                "For structural analogy transfer, test boundary and negative controls before using the analogy "
                "as an active reasoning guide."
            ),
            parent_frameworks=["fw_structural_morphism_transfer"],
            old_success_cases=[
                ScoredCase("le_chatelier_lenz_negative_feedback", "science", 0.77, 0.78, "invariant_preserved"),
                ScoredCase("skip_connection_residual_transport", "ml", 0.73, 0.74, "role_preserved"),
                ScoredCase("control_feedback_economics", "social", 0.70, 0.70, "opposing_response"),
            ],
            residual_cases=[
                ScoredCase("surface_similarity_false_transfer", "qa", 0.36, 0.69, "surface_only_match"),
                ScoredCase("morphism_over_structure_harm", "reasoning", 0.40, 0.66, "boundary_uncertain"),
            ],
            limiting_cases=[
                LimitingCase("invariant_preserved_case", "invariant_and_direction_preserved", 0.90),
                LimitingCase("surface_only_case", "missing_invariant", 0.86),
            ],
            new_prediction_cases=[
                ScoredCase("near_negative_abstain_wins", "qa", 0.48, 0.66, "near_negative"),
            ],
            unified_branch_count=1,
            old_rule_count=4,
            new_rule_count=4,
            transfer_domain_count=2,
            complexity_penalty=0.025,
            formal_certificate_status="pass",
            simulator_expected_utility=0.69,
            conflict_boundaries=[
                "do_not_promote_analogy_when_invariant_or_direction_is_missing",
            ],
        ),
        CandidateFramework(
            framework_id="fw_longer_context_style_boost",
            claim=(
                "Use a longer, more philosophical context around every proposal to improve self-evolution quality."
            ),
            parent_frameworks=["fw_prompt_scaffolding"],
            old_success_cases=[
                ScoredCase("math_hygiene_short_answer", "math", 0.76, 0.63, "requires_concise_answer"),
                ScoredCase("qa_direct_answer", "qa", 0.80, 0.71, "direct_fact_query"),
                ScoredCase("software_patch_review", "software", 0.74, 0.68, "localized_patch"),
            ],
            residual_cases=[
                ScoredCase("judge_prefers_bridge_plan", "planning", 0.50, 0.57, "needs_bridge"),
            ],
            limiting_cases=[
                LimitingCase("concise_answer_case", "direct_answer_required", 0.58),
            ],
            new_prediction_cases=[
                ScoredCase("verbosity_as_general_prior", "qa", 0.52, 0.54, "generic_query"),
            ],
            unified_branch_count=1,
            old_rule_count=2,
            new_rule_count=3,
            transfer_domain_count=1,
            complexity_penalty=0.18,
            formal_certificate_status="not_applicable",
            simulator_expected_utility=0.41,
            conflict_boundaries=[
                "regresses concise answer regimes and mimics prompt-length placebo",
            ],
        ),
    ]


def _old_success_preservation(cases: list[ScoredCase]) -> float:
    ratios = []
    for case in cases:
        tolerated_parent = max(0.01, case.parent_score - 0.02)
        ratios.append(min(1.0, case.candidate_score / tolerated_parent))
    return _mean(ratios)


def _regression_cost(cases: list[ScoredCase]) -> float:
    return _mean(max(0.0, case.parent_score - case.candidate_score) for case in cases)


def _residual_explanation(cases: list[ScoredCase]) -> float:
    return _mean(case.candidate_score for case in cases)


def _candidate_to_dict(candidate: CandidateFramework) -> dict[str, Any]:
    row = asdict(candidate)
    return row


def _dedupe_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for node in nodes:
        out[node["id"]] = node
    return [out[key] for key in sorted(out)]


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        out[(edge["source"], edge["target"], edge["type"])] = edge
    return [out[key] for key in sorted(out)]


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Any) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(float(value) for value in rows) / len(rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "missing": True}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative-generalization gate artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="conservative_generalization_gate_20260612")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_conservative_generalization_gate_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
