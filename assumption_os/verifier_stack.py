"""Unified verifier stack for candidate assumptions.

The individual gates already exist: preflight, world-model screening,
falsification, acceptance, and formal mapping.  This module combines those
signals into one ordered verifier protocol so a candidate has a single
auditable verdict before graph mutation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerifierStage:
    tier: str
    name: str
    status: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VerifierProtocol:
    proposal_type: str
    protocol_id: str
    required_stages: list[str]
    required_negative_controls: list[str]
    required_objective_evidence: list[str]
    acceptance_thresholds: dict[str, float]
    manual_review_policy: dict[str, str]
    default_next_action: str
    blocked_claims: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VerifierStackSummary:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    verdict: str
    confidence: str
    next_action: str
    protocol: VerifierProtocol
    protocol_report: dict[str, Any]
    stages: list[VerifierStage]
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["protocol"] = self.protocol.to_dict()
        d["stages"] = [stage.to_dict() for stage in self.stages]
        return d


def build_verifier_stack_payload(
    *,
    proposal_payload: dict,
    preflight_payload: dict | None = None,
    world_model_payload: dict | None = None,
    falsification_payload: dict | None = None,
    acceptance_payload: dict | None = None,
    formal_mapping_gate_payload: dict | None = None,
    structural_morphism_gate_payload: dict | None = None,
    objective_benchmark_payload: dict | None = None,
    eval_id: str,
) -> dict:
    """Combine gate outputs into a unified per-proposal verifier verdict."""

    preflight_by_id = _index(preflight_payload, "summaries")
    world_by_id = _index(world_model_payload, "predictions")
    falsification_by_id = _index(falsification_payload, "summaries")
    acceptance_by_id = _index(acceptance_payload, "summaries")
    formal_by_id = _index(formal_mapping_gate_payload, "gates")
    structural_by_id = _index(structural_morphism_gate_payload, "gates")
    objective_by_id = _index(objective_benchmark_payload, "summaries")
    summaries = [
        _summarize(
            proposal=proposal,
            preflight=preflight_by_id.get(proposal.get("proposal_id", ""), {}),
            world=world_by_id.get(proposal.get("proposal_id", ""), {}),
            falsification=falsification_by_id.get(proposal.get("proposal_id", ""), {}),
            acceptance=acceptance_by_id.get(proposal.get("proposal_id", ""), {}),
            formal=formal_by_id.get(proposal.get("proposal_id", ""), {}),
            structural=structural_by_id.get(proposal.get("proposal_id", ""), {}),
            objective=objective_by_id.get(proposal.get("proposal_id", ""), {}),
        )
        for proposal in proposal_payload.get("proposals", [])
    ]
    protocol_ids = sorted({s.protocol.protocol_id for s in summaries})
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "source_preflight_eval_id": (preflight_payload or {}).get("eval_id"),
        "source_world_model_eval_id": (world_model_payload or {}).get("eval_id"),
        "source_falsification_eval_id": (falsification_payload or {}).get("eval_id"),
        "source_acceptance_eval_id": (acceptance_payload or {}).get("eval_id"),
        "source_formal_gate_eval_id": (formal_mapping_gate_payload or {}).get("eval_id"),
        "source_structural_gate_eval_id": (structural_morphism_gate_payload or {}).get("eval_id"),
        "source_objective_benchmark_eval_id": (objective_benchmark_payload or {}).get("eval_id"),
        "proposal_count": len(summaries),
        "protocol_count": len(protocol_ids),
        "protocol_ids": protocol_ids,
        "protocol_pass_count": sum(1 for s in summaries if s.protocol_report.get("protocol_pass")),
        "protocol_violation_counts": dict(Counter(
            violation
            for summary in summaries
            for violation in summary.protocol_report.get("violations", [])
        )),
        "verdict_counts": dict(Counter(s.verdict for s in summaries)),
        "confidence_counts": dict(Counter(s.confidence for s in summaries)),
        "next_action_counts": dict(Counter(s.next_action for s in summaries)),
        "summaries": [s.to_dict() for s in summaries],
    }


def _summarize(
    *,
    proposal: dict,
    preflight: dict,
    world: dict,
    falsification: dict,
    acceptance: dict,
    formal: dict,
    structural: dict,
    objective: dict,
) -> VerifierStackSummary:
    proposal_id = proposal.get("proposal_id", "")
    candidate = proposal.get("candidate_node") or {}
    protocol = _protocol_for(proposal.get("proposal_type", ""))
    stages = [
        _preflight_stage(preflight),
        _world_stage(world),
        _formal_stage(formal),
        _structural_stage(structural),
        _falsification_stage(falsification),
        _acceptance_stage(acceptance),
        _objective_stage(
            preflight=preflight,
            falsification=falsification,
            acceptance=acceptance,
            objective=objective,
        ),
        _manual_review_stage(
            proposal=proposal,
            world=world,
            formal=formal,
            structural=structural,
            acceptance=acceptance,
        ),
    ]
    verdict, confidence, next_action, rationale = _verdict(
        preflight=preflight,
        world=world,
        falsification=falsification,
        acceptance=acceptance,
        formal=formal,
        structural=structural,
        objective=objective,
    )
    protocol_report = _protocol_report(
        protocol=protocol,
        stages=stages,
        verdict=verdict,
        next_action=next_action,
    )
    return VerifierStackSummary(
        proposal_id=proposal_id,
        proposal_type=proposal.get("proposal_type", ""),
        parent_node_id=proposal.get("parent_node_id", ""),
        candidate_node_id=candidate.get("id"),
        verdict=verdict,
        confidence=confidence,
        next_action=next_action,
        protocol=protocol,
        protocol_report=protocol_report,
        stages=stages,
        rationale=rationale,
    )


def _protocol_for(proposal_type: str | None) -> VerifierProtocol:
    p = str(proposal_type or "").lower()
    if "formal" in p or "morphism" in p or "mapping" in p:
        return VerifierProtocol(
            proposal_type=proposal_type or "",
            protocol_id="bounded_structural_morphism_candidate",
            required_stages=[
                "V0:candidate_preflight",
                "V1:world_model_screen",
                "V2:formal_mapping_gate",
                "V2b:structural_morphism_gate",
                "V3:sequential_falsification",
                "V4:fresh_ablation_acceptance",
                "V5:objective_task_regression",
                "V6:manual_review_gate",
            ],
            required_negative_controls=[
                "negative_control_hits_absent",
                "broken_or_uncertain_invariants_absent",
                "outside_active_problem_ids",
            ],
            required_objective_evidence=[
                "preserved_invariants",
                "functor_check",
                "trigger_lcb90",
                "fresh_cross_judge_replay",
            ],
            acceptance_thresholds={
                "trigger_lcb90_min": 0.54,
                "control_loss_ucb90_max": 0.35,
                "world_model_min_probability": 0.55,
            },
            manual_review_policy={
                "accepted_candidates": "required",
                "policy_sensitive_candidates": "required",
                "graph_mutation": "explicit_apply_or_writeback_required",
            },
            default_next_action="run_fresh_ablation_after_formal_and_structural_gates",
            blocked_claims=[
                "full_category_theory_theorem_prover",
                "formal_transfer_without_negative_controls",
                "default_policy_update_without_manual_apply",
            ],
            notes=[
                "This is a bounded structural protocol, not a mathematical proof certificate.",
                "Accepted morphisms must preserve stated invariants and survive negative controls.",
            ],
        )
    if "retrieval" in p or "rag" in p or "meta_qa" in p:
        return VerifierProtocol(
            proposal_type=proposal_type or "",
            protocol_id="retrieval_policy_candidate",
            required_stages=[
                "V0:candidate_preflight",
                "V1:world_model_screen",
                "V3:sequential_falsification",
                "V4:fresh_ablation_acceptance",
                "V5:objective_task_regression",
                "V6:manual_review_gate",
            ],
            required_negative_controls=[
                "control_problem_ids",
                "outside_active_problem_ids",
                "retrieval_negative_control",
            ],
            required_objective_evidence=[
                "target_trigger_coverage",
                "answer_f1_or_em_noninferiority",
                "fresh_cross_judge_replay",
            ],
            acceptance_thresholds={
                "trigger_lcb90_min": 0.54,
                "control_loss_ucb90_max": 0.25,
                "world_model_min_probability": 0.55,
            },
            manual_review_policy={
                "accepted_candidates": "required",
                "default_retrieval_policy": "requires_global_noninferiority",
                "graph_mutation": "explicit_apply_or_writeback_required",
            },
            default_next_action="run_retrieval_ablation_with_negative_controls",
            blocked_claims=[
                "answer_leakage",
                "gold_title_routing",
                "default_rag_policy_without_qa_metric",
            ],
        )
    if "world" in p or "calibration" in p or "selector" in p:
        return VerifierProtocol(
            proposal_type=proposal_type or "",
            protocol_id="world_model_calibration_candidate",
            required_stages=[
                "V0:candidate_preflight",
                "V1:world_model_screen",
                "V3:sequential_falsification",
                "V5:objective_task_regression",
                "V6:manual_review_gate",
            ],
            required_negative_controls=[
                "leave_domain_out_split",
                "base_rate_baseline",
                "raw_predictor_not_promoted_when_uncalibrated",
            ],
            required_objective_evidence=[
                "auroc_or_rank_auc",
                "brier_score",
                "expected_utility_noninferiority",
            ],
            acceptance_thresholds={
                "world_model_min_probability": 0.55,
                "calibration_error_max": 0.12,
                "expected_utility_delta_min": 0.0,
            },
            manual_review_policy={
                "production_selector": "requires_guarded_promotion",
                "raw_predictor": "shadow_until_calibrated",
                "graph_mutation": "explicit_apply_or_writeback_required",
            },
            default_next_action="promote_guarded_policy_only_if_calibrated",
            blocked_claims=[
                "world_model_as_task_simulator",
                "raw_uncalibrated_selector_as_production_policy",
            ],
        )
    if "memory" in p or "consolidation" in p or "sleep" in p:
        return VerifierProtocol(
            proposal_type=proposal_type or "",
            protocol_id="memory_consolidation_candidate",
            required_stages=[
                "V0:candidate_preflight",
                "V3:sequential_falsification",
                "V5:objective_task_regression",
                "V6:manual_review_gate",
            ],
            required_negative_controls=[
                "retrieval_before_after_nonregression",
                "archived_nodes_recoverable",
                "no_main_graph_mutation_in_dry_run",
            ],
            required_objective_evidence=[
                "duplicate_reduction",
                "retrieval_probe_noninferiority",
                "manifested_archive_write",
            ],
            acceptance_thresholds={
                "retrieval_noninferiority_min": 0.0,
                "duplicate_reduction_min": 0.1,
            },
            manual_review_policy={
                "apply_mode": "requires_explicit_apply",
                "deletions": "archive_only_unless_reviewed",
                "graph_mutation": "explicit_apply_or_writeback_required",
            },
            default_next_action="run_sleep_job_dry_run_then_apply_on_review",
            blocked_claims=[
                "irreversible_memory_delete",
                "main_graph_mutation_from_shadow_sleep",
            ],
        )
    if "prompt" in p or "guard" in p or "phase9" in p:
        return VerifierProtocol(
            proposal_type=proposal_type or "",
            protocol_id="prompt_guard_candidate",
            required_stages=[
                "V0:candidate_preflight",
                "V1:world_model_screen",
                "V3:sequential_falsification",
                "V4:fresh_ablation_acceptance",
                "V5:objective_task_regression",
                "V6:manual_review_gate",
            ],
            required_negative_controls=[
                "formal_high_risk_abstention",
                "math_science_noninferiority",
                "original_v3_noninferiority",
            ],
            required_objective_evidence=[
                "target_trigger_lift",
                "family_split_noninferiority",
                "heldout_noninferiority_to_v1",
            ],
            acceptance_thresholds={
                "trigger_lcb90_min": 0.54,
                "control_loss_ucb90_max": 0.25,
                "world_model_min_probability": 0.55,
            },
            manual_review_policy={
                "default_policy": "requires_global_win_threshold",
                "accepted_candidates": "required",
                "graph_mutation": "explicit_apply_or_writeback_required",
            },
            default_next_action="promote_as_selective_guard_only",
            blocked_claims=[
                "longer_prompt_is_sufficient_explanation",
                "default_injection_without_family_split",
                "over_structuring_high_risk_rows",
            ],
        )
    return VerifierProtocol(
        proposal_type=proposal_type or "",
        protocol_id="method_hypothesis_candidate",
        required_stages=[
            "V0:candidate_preflight",
            "V1:world_model_screen",
            "V3:sequential_falsification",
            "V4:fresh_ablation_acceptance",
            "V5:objective_task_regression",
            "V6:manual_review_gate",
        ],
        required_negative_controls=[
            "control_problem_ids",
            "outside_active_problem_ids",
            "control_harm_sequential",
        ],
        required_objective_evidence=[
            "trigger_lcb90",
            "control_harm_sequential",
            "fresh_cross_judge_replay",
        ],
        acceptance_thresholds={
            "trigger_lcb90_min": 0.54,
            "control_loss_ucb90_max": 0.35,
            "world_model_min_probability": 0.55,
        },
        manual_review_policy={
            "accepted_candidates": "required",
            "high_risk_candidates": "required",
            "graph_mutation": "explicit_apply_or_writeback_required",
        },
        default_next_action="run_fresh_ablation",
        blocked_claims=[
            "automatic_graph_mutation_without_apply",
            "acceptance_without_trigger_control_evidence",
            "general_method_claim_without_scope",
        ],
    )


def _protocol_report(
    *,
    protocol: VerifierProtocol,
    stages: list[VerifierStage],
    verdict: str,
    next_action: str,
) -> dict[str, Any]:
    by_key = {_stage_key(stage): stage for stage in stages}
    missing_required_stages = [
        stage_key
        for stage_key in protocol.required_stages
        if stage_key not in by_key
    ]
    mutating_verdict = verdict == "accepted_for_gated_apply"
    negative_control_satisfied = _negative_control_satisfied(stages)
    objective_evidence_satisfied = _objective_evidence_satisfied(protocol, stages)
    threshold_violations = _threshold_violations(protocol, stages)
    manual_review_satisfied = _manual_review_satisfied(stages, mutating_verdict=mutating_verdict)

    violations: list[str] = []
    violations.extend(f"missing_stage:{stage_key}" for stage_key in missing_required_stages)
    if mutating_verdict:
        blocking_statuses = _blocking_statuses_for_accepted(protocol, by_key)
        violations.extend(blocking_statuses)
        if not negative_control_satisfied:
            violations.append("accepted_without_negative_control_evidence")
        if not objective_evidence_satisfied:
            violations.append("accepted_without_required_objective_evidence")
        if threshold_violations:
            violations.extend(threshold_violations)
        if not manual_review_satisfied:
            violations.append("accepted_without_manual_review_gate")
    elif verdict.startswith("rejected"):
        if not _has_falsifying_stage(stages):
            violations.append("rejected_without_falsifying_stage")
    elif "apply" in next_action:
        violations.append("nonaccepted_verdict_requests_apply")

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_pass": not violations,
        "verdict": verdict,
        "next_action": next_action,
        "missing_required_stages": missing_required_stages,
        "negative_control_satisfied": negative_control_satisfied,
        "objective_evidence_satisfied": objective_evidence_satisfied,
        "threshold_violations": threshold_violations,
        "manual_review_satisfied": manual_review_satisfied,
        "violations": violations,
        "stage_status": {
            _stage_key(stage): stage.status
            for stage in stages
        },
    }


def _stage_key(stage: VerifierStage) -> str:
    return f"{stage.tier}:{stage.name}"


def _blocking_statuses_for_accepted(protocol: VerifierProtocol, by_key: dict[str, VerifierStage]) -> list[str]:
    statuses: list[str] = []
    for key in protocol.required_stages:
        stage = by_key.get(key)
        if not stage:
            continue
        if key == "V6:manual_review_gate":
            if stage.status != "required":
                statuses.append(f"accepted_manual_gate_not_required:{key}")
            continue
        if stage.status in {"missing", "block", "fail", "repair", "defer"}:
            statuses.append(f"accepted_blocking_stage:{key}:{stage.status}")
        if stage.status == "not_applicable" and key in {"V2:formal_mapping_gate", "V2b:structural_morphism_gate"}:
            statuses.append(f"accepted_required_stage_not_applicable:{key}")
    return statuses


def _negative_control_satisfied(stages: list[VerifierStage]) -> bool:
    for stage in stages:
        evidence = stage.evidence
        if int(evidence.get("control_n") or 0) > 0 or int(evidence.get("outside_active_n") or 0) > 0:
            return True
        if "control_harm_sequential" in evidence.get("passed_required_experiments", []):
            return True
        if "control_harm_sequential" in evidence.get("experiment_name_counts", {}):
            return True
        if evidence.get("control_outcomes"):
            return True
        if evidence.get("negative_control_hits") == [] and "negative_control_hits" in evidence:
            return True
    return False


def _objective_evidence_satisfied(protocol: VerifierProtocol, stages: list[VerifierStage]) -> bool:
    evidence = _merged_evidence(stages)
    for required in protocol.required_objective_evidence:
        if required in {"trigger_lcb90", "target_trigger_lift"}:
            if evidence.get("trigger_lcb90") is None and evidence.get("target_trigger_lift") is None:
                return False
        elif required == "control_harm_sequential":
            if "control_harm_sequential" not in evidence.get("passed_required_experiments", []):
                return False
        elif required == "fresh_cross_judge_replay":
            if "fresh_cross_judge_replay" not in evidence.get("passed_required_experiments", []):
                return False
        elif required == "preserved_invariants":
            if not evidence.get("preserved_invariants"):
                return False
        elif required == "functor_check":
            if not evidence.get("functor_check"):
                return False
        else:
            if required not in evidence or evidence.get(required) in {None, False, []}:
                return False
    v5 = next((stage for stage in stages if _stage_key(stage) == "V5:objective_task_regression"), None)
    return bool(v5 and v5.evidence.get("objective_gate_passed"))


def _threshold_violations(protocol: VerifierProtocol, stages: list[VerifierStage]) -> list[str]:
    evidence = _merged_evidence(stages)
    violations: list[str] = []
    trigger_min = protocol.acceptance_thresholds.get("trigger_lcb90_min")
    if trigger_min is not None:
        trigger_lcb = evidence.get("trigger_lcb90")
        if trigger_lcb is None or float(trigger_lcb) < trigger_min:
            violations.append("threshold:trigger_lcb90_min")
    control_max = protocol.acceptance_thresholds.get("control_loss_ucb90_max")
    if control_max is not None:
        control_ucb = evidence.get("control_loss_ucb90")
        control_exp_ok = "control_harm_sequential" in evidence.get("passed_required_experiments", [])
        if control_ucb is not None and float(control_ucb) > control_max:
            violations.append("threshold:control_loss_ucb90_max")
        elif control_ucb is None and not control_exp_ok:
            violations.append("threshold:control_loss_ucb90_missing")
    world_min = protocol.acceptance_thresholds.get("world_model_min_probability")
    if world_min is not None:
        probability = evidence.get("predicted_acceptance_probability")
        if probability is None or float(probability) < world_min:
            violations.append("threshold:world_model_min_probability")
    return violations


def _manual_review_satisfied(stages: list[VerifierStage], *, mutating_verdict: bool) -> bool:
    if not mutating_verdict:
        return True
    v6 = next((stage for stage in stages if _stage_key(stage) == "V6:manual_review_gate"), None)
    return bool(v6 and v6.status == "required" and v6.evidence.get("permission_boundary") == "explicit_apply_or_writeback_required")


def _has_falsifying_stage(stages: list[VerifierStage]) -> bool:
    return any(
        stage.status == "fail" and stage.tier in {"V3", "V4"}
        for stage in stages
    )


def _merged_evidence(stages: list[VerifierStage]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for stage in stages:
        merged.update(stage.evidence)
    return merged


def _preflight_stage(preflight: dict) -> VerifierStage:
    readiness = preflight.get("readiness")
    status = "missing"
    detail = "No preflight summary is available."
    if readiness == "ready_for_fresh_ablation":
        status = "pass"
        detail = "Candidate has trigger exposure and no blocking no-fire exposure."
    elif readiness in {"needs_scope_fix", "needs_retrieval_fix", "needs_more_trigger_rows"}:
        status = "repair"
        detail = f"Preflight readiness is {readiness}."
    elif readiness:
        status = "defer"
        detail = f"Preflight readiness is {readiness}."
    return VerifierStage(
        tier="V0",
        name="candidate_preflight",
        status=status,
        detail=detail,
        evidence={
            "readiness": readiness,
            "trigger_n": len(preflight.get("trigger_problem_ids", [])),
            "control_n": len(preflight.get("control_problem_ids", [])),
            "outside_active_n": len(preflight.get("outside_active_problem_ids", [])),
        },
    )


def _world_stage(world: dict) -> VerifierStage:
    if not world:
        return VerifierStage("V1", "world_model_screen", "missing", "No world-model prediction is available.")
    probability = float(world.get("predicted_acceptance_probability", 0.5) or 0.5)
    risk = world.get("predicted_regression_risk", "unknown")
    action = world.get("recommended_next_action")
    if risk == "high":
        status = "risk"
    elif probability >= 0.72:
        status = "pass"
    elif probability >= 0.55:
        status = "weak_pass"
    else:
        status = "defer"
    return VerifierStage(
        tier="V1",
        name="world_model_screen",
        status=status,
        detail=f"p_accept={probability:.4f}; risk={risk}; action={action}",
        evidence={
            "predicted_acceptance_probability": probability,
            "expected_utility": world.get("expected_utility"),
            "predicted_regression_risk": risk,
            "recommended_next_action": action,
            "predicted_failure_modes": world.get("predicted_failure_modes", []),
        },
    )


def _formal_stage(formal: dict) -> VerifierStage:
    if not formal:
        return VerifierStage("V2", "formal_mapping_gate", "not_applicable", "No formal gate applies.")
    decision = formal.get("decision", "not_applicable")
    blocks = bool(formal.get("blocks_policy_update"))
    status = "block" if blocks else ("pass" if decision == "allow" else "not_applicable")
    return VerifierStage(
        tier="V2",
        name="formal_mapping_gate",
        status=status,
        detail=f"formal_gate={decision}; blocks_policy_update={blocks}",
        evidence={
            "decision": decision,
            "blocks_policy_update": blocks,
            "reason": formal.get("reason"),
        },
    )


def _structural_stage(structural: dict) -> VerifierStage:
    if not structural:
        return VerifierStage("V2b", "structural_morphism_gate", "not_applicable", "No structural morphism gate applies.")
    decision = structural.get("decision", "not_applicable")
    blocks = bool(structural.get("blocks_policy_update"))
    if blocks:
        status = "block"
    elif decision == "allow":
        status = "pass"
    elif decision == "not_applicable":
        status = "not_applicable"
    else:
        status = "shadow"
    return VerifierStage(
        tier="V2b",
        name="structural_morphism_gate",
        status=status,
        detail=f"structural_gate={decision}; blocks_policy_update={blocks}",
        evidence={
            "decision": decision,
            "blocks_policy_update": blocks,
            "reason": structural.get("reason"),
            "source_pattern_id": structural.get("source_pattern_id"),
            "functor_check": structural.get("functor_check", {}),
            "preserved_invariants": structural.get("preserved_invariants", []),
            "broken_or_uncertain_invariants": structural.get("broken_or_uncertain_invariants", []),
            "negative_control_hits": structural.get("negative_control_hits", []),
        },
    )


def _falsification_stage(falsification: dict) -> VerifierStage:
    if not falsification:
        return VerifierStage("V3", "sequential_falsification", "missing", "No falsification gate summary is available.")
    decision = falsification.get("decision")
    experiments = falsification.get("experiments", [])
    if decision in {"accept", "ready_for_ablation"}:
        status = "pass"
    elif decision in {"reject_benefit", "reject_harm"}:
        status = "fail"
    elif str(decision or "").startswith("blocked"):
        status = "block"
    else:
        status = "defer"
    return VerifierStage(
        tier="V3",
        name="sequential_falsification",
        status=status,
        detail=f"falsification_decision={decision}",
        evidence={
            "decision": decision,
            "next_action": falsification.get("next_action"),
            "ordered_checks": falsification.get("ordered_checks", []),
            "experiment_count": len(experiments),
            "experiment_status_counts": dict(Counter(e.get("status") for e in experiments)),
            "experiment_name_counts": dict(Counter(e.get("name") for e in experiments)),
            "experiments": experiments,
        },
    )


def _acceptance_stage(acceptance: dict) -> VerifierStage:
    if not acceptance:
        return VerifierStage("V4", "fresh_ablation_acceptance", "missing", "No fresh acceptance result is available.")
    decision = acceptance.get("decision")
    if decision == "accept":
        status = "pass"
    elif decision in {"reject_benefit", "reject_harm"}:
        status = "fail"
    else:
        status = "defer"
    return VerifierStage(
        tier="V4",
        name="fresh_ablation_acceptance",
        status=status,
        detail=f"acceptance_decision={decision}",
        evidence={
            "decision": decision,
            "trigger_outcomes": acceptance.get("trigger_outcomes", {}),
            "control_outcomes": acceptance.get("control_outcomes", {}),
            "trigger_lcb90": acceptance.get("trigger_lcb90"),
            "control_loss_ucb90": acceptance.get("control_loss_ucb90"),
        },
    )


def _objective_stage(*, preflight: dict, falsification: dict, acceptance: dict, objective: dict) -> VerifierStage:
    decision = acceptance.get("decision")
    if not acceptance:
        readiness = preflight.get("readiness")
        status = "planned" if readiness == "ready_for_fresh_ablation" else "defer"
        return VerifierStage(
            "V5",
            "objective_task_regression",
            status,
            "Objective trigger/control regression gate awaits fresh acceptance evidence.",
            evidence={
                "readiness": readiness,
                "acceptance_decision": None,
                "objective_gate_passed": False,
                "objective_gate_source": "external_objective_task_benchmark" if objective else "internal_trigger_control",
                "external_objective_available": bool(objective),
            },
        )
    internal_ok = _internal_objective_acceptance_ok(acceptance=acceptance, falsification=falsification)
    external_available = bool(objective)
    external_ok = bool(objective.get("objective_gate_passed")) if external_available else True
    objective_ok = internal_ok and external_ok
    if decision == "accept":
        status = "pass" if objective_ok else "block"
        detail = (
            "Accepted candidate satisfies internal and external objective criteria."
            if objective_ok and external_available
            else "Accepted candidate satisfies objective trigger/control and falsification criteria."
            if objective_ok
            else "Accepted candidate is blocked because objective criteria are incomplete."
        )
    elif decision in {"reject_benefit", "reject_harm"}:
        status = "pass"
        detail = f"Objective gate recorded a falsifying fresh-ablation decision: {decision}."
    else:
        status = "defer"
        detail = f"Objective gate cannot decide from acceptance decision {decision}."
    return VerifierStage(
        "V5",
        "objective_task_regression",
        status,
        detail,
        evidence={
            "acceptance_decision": decision,
            "trigger_lcb90": acceptance.get("trigger_lcb90"),
            "control_loss_ucb90": acceptance.get("control_loss_ucb90"),
            "objective_gate_passed": objective_ok or decision in {"reject_benefit", "reject_harm"},
            "objective_gate_source": "external_objective_task_benchmark" if external_available else "internal_trigger_control",
            "internal_objective_passed": internal_ok,
            "external_objective_available": external_available,
            "external_objective_passed": external_ok if external_available else None,
            "external_objective_decision": objective.get("objective_decision"),
            "external_task_count": objective.get("external_task_count", 0),
            "external_family_count": objective.get("family_count", 0),
            "external_mean_score_delta": objective.get("mean_score_delta"),
            "external_loss_rate": objective.get("loss_rate"),
            "required_experiments": [
                "trigger_benefit_sequential",
                "control_harm_sequential",
                "fresh_cross_judge_replay",
            ],
            "passed_required_experiments": _passed_required_experiments(falsification),
        },
    )


def _manual_review_stage(*, proposal: dict, world: dict, formal: dict, structural: dict, acceptance: dict) -> VerifierStage:
    decision = acceptance.get("decision")
    risk = world.get("predicted_regression_risk", "unknown")
    policy_sensitive = (
        bool(formal.get("blocks_policy_update"))
        or bool(structural.get("blocks_policy_update"))
        or decision == "accept"
        or risk == "high"
    )
    if decision == "accept":
        status = "required"
        detail = "Accepted candidate is ready only for gated manual apply, not automatic graph mutation."
    elif policy_sensitive:
        status = "required"
        detail = "Policy-sensitive or high-risk candidate requires manual review before mutation."
    else:
        status = "not_required"
        detail = "Manual review is not required before the next verifier action."
    return VerifierStage(
        "V6",
        "manual_review_gate",
        status,
        detail,
        evidence={
            "manual_gate_required": policy_sensitive,
            "acceptance_decision": decision,
            "predicted_regression_risk": risk,
            "formal_blocks_policy_update": bool(formal.get("blocks_policy_update")),
            "structural_blocks_policy_update": bool(structural.get("blocks_policy_update")),
            "permission_boundary": "explicit_apply_or_writeback_required",
            "proposal_type": proposal.get("proposal_type"),
        },
    )


def _verdict(
    *,
    preflight: dict,
    world: dict,
    falsification: dict,
    acceptance: dict,
    formal: dict,
    structural: dict,
    objective: dict,
) -> tuple[str, str, str, str]:
    if formal.get("blocks_policy_update"):
        return (
            "blocked_formal_gate",
            "high",
            "repair_formal_mapping_before_policy_update",
            "Formal mapping gate blocks policy-sensitive promotion.",
        )
    if structural.get("blocks_policy_update"):
        return (
            "blocked_structural_morphism_gate",
            "high",
            "repair_structural_morphism_before_policy_update",
            "Structural morphism gate blocks promotion because invariants or negative controls are unsafe.",
        )

    acceptance_decision = acceptance.get("decision")
    if acceptance_decision == "accept" and not _objective_acceptance_ok(
        acceptance=acceptance,
        falsification=falsification,
        objective=objective,
    ):
        return (
            "blocked_objective_gate",
            "high",
            "repair_objective_gate_before_apply",
            "Fresh acceptance passed, but V5 objective trigger/control criteria are incomplete.",
        )
    if acceptance_decision == "accept":
        return (
            "accepted_for_gated_apply",
            "high",
            "apply_accepted_candidate_if_requested",
            "Fresh acceptance gate passed after trigger/control checks.",
        )
    if acceptance_decision == "reject_harm":
        return (
            "rejected_control_harm",
            "high",
            "reject_or_narrow_scope",
            "Fresh acceptance observed control harm.",
        )
    if acceptance_decision == "reject_benefit":
        return (
            "rejected_weak_benefit",
            "high",
            "reject_or_revise_candidate",
            "Fresh acceptance did not find enough trigger benefit.",
        )

    readiness = preflight.get("readiness")
    if readiness in {"needs_scope_fix", "needs_retrieval_fix", "needs_more_trigger_rows"}:
        return (
            "needs_preflight_repair",
            "medium",
            _preflight_repair_action(readiness),
            f"Preflight requires repair before expensive validation: {readiness}.",
        )

    falsification_decision = falsification.get("decision")
    if falsification_decision in {"reject_benefit", "reject_harm"}:
        return (
            f"rejected_by_falsification_{falsification_decision}",
            "high",
            "reject_or_revise_candidate",
            "Sequential falsification gate rejected the candidate.",
        )

    probability = float(world.get("predicted_acceptance_probability", 0.5) or 0.5)
    risk = world.get("predicted_regression_risk", "unknown")
    if readiness == "ready_for_fresh_ablation" and risk != "high" and probability >= 0.55:
        return (
            "ready_for_fresh_ablation",
            "medium",
            "run_fresh_ablation",
            "Preflight passed and world model says the candidate is worth testing.",
        )
    if risk == "high":
        return (
            "needs_risk_repair",
            "medium",
            "repair_scope_before_ablation",
            "World model or regression screen predicts high risk.",
        )
    return (
        "collect_more_evidence",
        "low",
        "collect_more_evidence",
        "Evidence is insufficient for repair, ablation, or rejection.",
    )


def _preflight_repair_action(readiness: str | None) -> str:
    return {
        "needs_scope_fix": "narrow_scope_before_ablation",
        "needs_retrieval_fix": "repair_retrieval_before_ablation",
        "needs_more_trigger_rows": "collect_more_trigger_rows",
    }.get(str(readiness), "run_candidate_preflight")


def _objective_acceptance_ok(*, acceptance: dict, falsification: dict, objective: dict) -> bool:
    internal_ok = _internal_objective_acceptance_ok(acceptance=acceptance, falsification=falsification)
    if not internal_ok:
        return False
    if not objective:
        return True
    return bool(objective.get("objective_gate_passed"))


def _internal_objective_acceptance_ok(*, acceptance: dict, falsification: dict) -> bool:
    if acceptance.get("decision") != "accept":
        return False
    trigger_lcb = acceptance.get("trigger_lcb90")
    control_ucb = acceptance.get("control_loss_ucb90")
    trigger_ok = trigger_lcb is not None and float(trigger_lcb) >= 0.54
    control_ok = control_ucb is None or float(control_ucb) <= 0.35
    required = {"trigger_benefit_sequential", "control_harm_sequential", "fresh_cross_judge_replay"}
    return trigger_ok and control_ok and required <= set(_passed_required_experiments(falsification))


def _passed_required_experiments(falsification: dict) -> list[str]:
    return sorted({
        str(experiment.get("name"))
        for experiment in falsification.get("experiments", [])
        if experiment.get("status") == "passed"
    })


def _index(payload: dict | None, key: str) -> dict[str, dict]:
    if not payload:
        return {}
    return {
        row.get("proposal_id"): row
        for row in payload.get(key, [])
        if row.get("proposal_id")
    }


def _load_json(path: Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--preflight", default=None)
    ap.add_argument("--world-model", default=None)
    ap.add_argument("--falsification", default=None)
    ap.add_argument("--acceptance", default=None)
    ap.add_argument("--formal-gate", default=None)
    ap.add_argument("--structural-gate", default=None)
    ap.add_argument("--objective-benchmark", default=None)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_verifier_stack_payload(
        proposal_payload=_load_json(_resolve(root, args.proposals)) or {},
        preflight_payload=_load_json(_resolve(root, args.preflight)),
        world_model_payload=_load_json(_resolve(root, args.world_model)),
        falsification_payload=_load_json(_resolve(root, args.falsification)),
        acceptance_payload=_load_json(_resolve(root, args.acceptance)),
        formal_mapping_gate_payload=_load_json(_resolve(root, args.formal_gate)),
        structural_morphism_gate_payload=_load_json(_resolve(root, args.structural_gate)),
        objective_benchmark_payload=_load_json(_resolve(root, args.objective_benchmark)),
        eval_id=args.eval_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
