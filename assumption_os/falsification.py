"""Sequential falsification gate for candidate assumptions.

This module makes the POPPER-style ordering explicit: a candidate must first be
testable, then have enough routed trigger/control coverage, then pass fresh
ablation acceptance before it can update graph policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum


class FalsificationDecision(str, Enum):
    MANIFEST_ONLY = "manifest_only"
    BLOCKED_UNDERPOWERED = "blocked_underpowered"
    BLOCKED_RETRIEVAL = "blocked_retrieval"
    BLOCKED_SCOPE_RISK = "blocked_scope_risk"
    READY_FOR_ABLATION = "ready_for_ablation"
    ACCEPT = "accept"
    REJECT_BENEFIT = "reject_benefit"
    REJECT_HARM = "reject_harm"
    INSUFFICIENT_JUDGMENTS = "insufficient_judgments"


@dataclass(frozen=True)
class FalsificationGateSummary:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    decision: FalsificationDecision
    ordered_checks: list[dict] = field(default_factory=list)
    next_action: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def build_falsification_payload(
    *,
    proposal_payload: dict,
    preflight_payload: dict,
    acceptance_payload: dict | None = None,
) -> dict:
    preflight_by_id = {s.get("proposal_id"): s for s in preflight_payload.get("summaries", [])}
    acceptance_by_id = {
        s.get("proposal_id"): s
        for s in (acceptance_payload or {}).get("summaries", [])
    }
    summaries = [
        _summarize(
            proposal=proposal,
            preflight=preflight_by_id.get(proposal.get("proposal_id"), {}),
            acceptance=acceptance_by_id.get(proposal.get("proposal_id")),
        )
        for proposal in proposal_payload.get("proposals", [])
    ]
    return {
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "source_preflight_eval_id": preflight_payload.get("eval_id"),
        "source_acceptance_eval_id": (acceptance_payload or {}).get("eval_id"),
        "decision_counts": dict(Counter(s.decision.value for s in summaries)),
        "summaries": [s.to_dict() for s in summaries],
    }


def _summarize(*, proposal: dict, preflight: dict, acceptance: dict | None) -> FalsificationGateSummary:
    candidate = proposal.get("candidate_node") or {}
    proposal_id = proposal.get("proposal_id", "")
    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    if not candidate:
        add_check("candidate_exists", False, "Proposal has no candidate node to falsify.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.MANIFEST_ONLY,
            checks=checks,
            next_action="record_manifest_only",
            rationale="The proposal records evidence or promotion metadata only.",
        )
    add_check("candidate_exists", True, f"Candidate node {candidate.get('id')} exists.")

    readiness = preflight.get("readiness")
    if readiness in {None, "manifest_only"}:
        add_check("preflight_available", False, "No candidate preflight summary is available.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.BLOCKED_UNDERPOWERED,
            checks=checks,
            next_action="run_candidate_preflight",
            rationale="The candidate cannot enter ablation until preflight exists.",
        )
    add_check("preflight_available", True, f"Preflight readiness is {readiness}.")

    if readiness == "needs_scope_fix":
        add_check("scope_safety", False, "Preflight found outside/no-fire exposure.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.BLOCKED_SCOPE_RISK,
            checks=checks,
            next_action="narrow_scope_before_ablation",
            rationale="A candidate with no-fire exposure should be falsified by scope repair first.",
        )
    add_check("scope_safety", True, "No blocking no-fire exposure was reported.")

    if readiness == "needs_retrieval_fix":
        add_check("trigger_retrieval", False, "Candidate misses routed trigger rows.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.BLOCKED_RETRIEVAL,
            checks=checks,
            next_action="repair_retrieval_before_ablation",
            rationale="The candidate is not exposed enough to be fairly tested.",
        )
    if readiness == "needs_more_trigger_rows":
        add_check("trigger_power", False, "Sample has too few routed trigger rows.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.BLOCKED_UNDERPOWERED,
            checks=checks,
            next_action="collect_more_trigger_rows",
            rationale="The candidate needs more trigger examples before ablation.",
        )
    add_check("trigger_power", True, "Trigger coverage is sufficient for fresh ablation.")

    if not acceptance:
        add_check("fresh_ablation_acceptance", False, "No fresh ablation acceptance result yet.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.READY_FOR_ABLATION,
            checks=checks,
            next_action="run_fresh_ablation",
            rationale="The candidate passed preflight and is ready to be falsified by fresh judgments.",
        )

    decision = acceptance.get("decision")
    if decision == "accept":
        add_check("fresh_ablation_acceptance", True, "Trigger benefit passed and control harm stayed within gate.")
        return _summary(
            proposal=proposal,
            candidate=candidate,
            decision=FalsificationDecision.ACCEPT,
            checks=checks,
            next_action="apply_accepted_candidate_if_requested",
            rationale="The candidate passed the ordered falsification gate.",
        )
    if decision == "reject_harm":
        add_check("fresh_ablation_acceptance", False, "Control harm exceeded the gate.")
        mapped = FalsificationDecision.REJECT_HARM
        next_action = "reject_or_narrow_scope"
    elif decision == "reject_benefit":
        add_check("fresh_ablation_acceptance", False, "Trigger benefit did not pass the gate.")
        mapped = FalsificationDecision.REJECT_BENEFIT
        next_action = "reject_or_revise_candidate"
    else:
        add_check("fresh_ablation_acceptance", False, "Fresh judgments are insufficient.")
        mapped = FalsificationDecision.INSUFFICIENT_JUDGMENTS
        next_action = "collect_more_judgments"
    return _summary(
        proposal=proposal,
        candidate=candidate,
        decision=mapped,
        checks=checks,
        next_action=next_action,
        rationale=acceptance.get("rationale", "Acceptance gate did not approve the candidate."),
    )


def _summary(
    *,
    proposal: dict,
    candidate: dict,
    decision: FalsificationDecision,
    checks: list[dict],
    next_action: str,
    rationale: str,
) -> FalsificationGateSummary:
    return FalsificationGateSummary(
        proposal_id=proposal.get("proposal_id", ""),
        proposal_type=proposal.get("proposal_type", ""),
        parent_node_id=proposal.get("parent_node_id", ""),
        candidate_node_id=candidate.get("id"),
        decision=decision,
        ordered_checks=checks,
        next_action=next_action,
        rationale=rationale,
    )
