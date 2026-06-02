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
class VerifierStackSummary:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    verdict: str
    confidence: str
    next_action: str
    stages: list[VerifierStage]
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
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
    return VerifierStackSummary(
        proposal_id=proposal_id,
        proposal_type=proposal.get("proposal_type", ""),
        parent_node_id=proposal.get("parent_node_id", ""),
        candidate_node_id=candidate.get("id"),
        verdict=verdict,
        confidence=confidence,
        next_action=next_action,
        stages=stages,
        rationale=rationale,
    )


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
