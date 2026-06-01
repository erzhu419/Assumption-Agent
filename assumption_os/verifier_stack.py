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
    eval_id: str,
) -> dict:
    """Combine gate outputs into a unified per-proposal verifier verdict."""

    preflight_by_id = _index(preflight_payload, "summaries")
    world_by_id = _index(world_model_payload, "predictions")
    falsification_by_id = _index(falsification_payload, "summaries")
    acceptance_by_id = _index(acceptance_payload, "summaries")
    formal_by_id = _index(formal_mapping_gate_payload, "gates")
    summaries = [
        _summarize(
            proposal=proposal,
            preflight=preflight_by_id.get(proposal.get("proposal_id", ""), {}),
            world=world_by_id.get(proposal.get("proposal_id", ""), {}),
            falsification=falsification_by_id.get(proposal.get("proposal_id", ""), {}),
            acceptance=acceptance_by_id.get(proposal.get("proposal_id", ""), {}),
            formal=formal_by_id.get(proposal.get("proposal_id", ""), {}),
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
) -> VerifierStackSummary:
    proposal_id = proposal.get("proposal_id", "")
    candidate = proposal.get("candidate_node") or {}
    stages = [
        _preflight_stage(preflight),
        _world_stage(world),
        _formal_stage(formal),
        _falsification_stage(falsification),
        _acceptance_stage(acceptance),
    ]
    verdict, confidence, next_action, rationale = _verdict(
        preflight=preflight,
        world=world,
        falsification=falsification,
        acceptance=acceptance,
        formal=formal,
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


def _verdict(
    *,
    preflight: dict,
    world: dict,
    falsification: dict,
    acceptance: dict,
    formal: dict,
) -> tuple[str, str, str, str]:
    if formal.get("blocks_policy_update"):
        return (
            "blocked_formal_gate",
            "high",
            "repair_formal_mapping_before_policy_update",
            "Formal mapping gate blocks policy-sensitive promotion.",
        )

    acceptance_decision = acceptance.get("decision")
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
