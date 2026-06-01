"""Sequential falsification gate for candidate assumptions.

This module makes the POPPER-style ordering explicit: a candidate must first be
testable, then have enough routed trigger/control coverage, then pass fresh
ablation acceptance before it can update graph policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum

from .schema import stable_id


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
class FalsificationExperiment:
    experiment_id: str
    proposal_id: str
    layer: str
    name: str
    status: str
    hypothesis: str
    falsifier: str
    trigger_problem_ids: list[str] = field(default_factory=list)
    control_problem_ids: list[str] = field(default_factory=list)
    command_hint: str = ""
    measurement: str = ""
    stop_rule: str = ""
    pass_rule: str = ""
    fail_rule: str = ""
    type_i_control: str = ""
    observed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FalsificationGateSummary:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    decision: FalsificationDecision
    ordered_checks: list[dict] = field(default_factory=list)
    experiments: list[FalsificationExperiment] = field(default_factory=list)
    next_action: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["experiments"] = [e.to_dict() for e in self.experiments]
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
        "experiment_count": sum(len(s.experiments) for s in summaries),
        "experiment_status_counts": dict(Counter(e.status for s in summaries for e in s.experiments)),
        "experiment_name_counts": dict(Counter(e.name for s in summaries for e in s.experiments)),
        "summaries": [s.to_dict() for s in summaries],
    }


def _summarize(*, proposal: dict, preflight: dict, acceptance: dict | None) -> FalsificationGateSummary:
    candidate = proposal.get("candidate_node") or {}
    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    def finish(
        decision: FalsificationDecision,
        next_action: str,
        rationale: str,
    ) -> FalsificationGateSummary:
        return _summary(
            proposal=proposal,
            candidate=candidate,
            preflight=preflight,
            acceptance=acceptance,
            decision=decision,
            checks=checks,
            next_action=next_action,
            rationale=rationale,
        )

    if not candidate:
        add_check("candidate_exists", False, "Proposal has no candidate node to falsify.")
        return finish(
            FalsificationDecision.MANIFEST_ONLY,
            "record_manifest_only",
            "The proposal records evidence or promotion metadata only.",
        )
    add_check("candidate_exists", True, f"Candidate node {candidate.get('id')} exists.")

    readiness = preflight.get("readiness")
    if readiness in {None, "manifest_only"}:
        add_check("preflight_available", False, "No candidate preflight summary is available.")
        return finish(
            FalsificationDecision.BLOCKED_UNDERPOWERED,
            "run_candidate_preflight",
            "The candidate cannot enter ablation until preflight exists.",
        )
    add_check("preflight_available", True, f"Preflight readiness is {readiness}.")

    if readiness == "needs_scope_fix":
        add_check("scope_safety", False, "Preflight found outside/no-fire exposure.")
        return finish(
            FalsificationDecision.BLOCKED_SCOPE_RISK,
            "narrow_scope_before_ablation",
            "A candidate with no-fire exposure should be falsified by scope repair first.",
        )
    add_check("scope_safety", True, "No blocking no-fire exposure was reported.")

    if readiness == "needs_retrieval_fix":
        add_check("trigger_retrieval", False, "Candidate misses routed trigger rows.")
        return finish(
            FalsificationDecision.BLOCKED_RETRIEVAL,
            "repair_retrieval_before_ablation",
            "The candidate is not exposed enough to be fairly tested.",
        )
    if readiness == "needs_more_trigger_rows":
        add_check("trigger_power", False, "Sample has too few routed trigger rows.")
        return finish(
            FalsificationDecision.BLOCKED_UNDERPOWERED,
            "collect_more_trigger_rows",
            "The candidate needs more trigger examples before ablation.",
        )
    add_check("trigger_power", True, "Trigger coverage is sufficient for fresh ablation.")

    if not acceptance:
        add_check("fresh_ablation_acceptance", False, "No fresh ablation acceptance result yet.")
        return finish(
            FalsificationDecision.READY_FOR_ABLATION,
            "run_fresh_ablation",
            "The candidate passed preflight and is ready to be falsified by fresh judgments.",
        )

    decision = acceptance.get("decision")
    if decision == "accept":
        add_check("fresh_ablation_acceptance", True, "Trigger benefit passed and control harm stayed within gate.")
        return finish(
            FalsificationDecision.ACCEPT,
            "apply_accepted_candidate_if_requested",
            "The candidate passed the ordered falsification gate.",
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
    return finish(
        mapped,
        next_action,
        acceptance.get("rationale", "Acceptance gate did not approve the candidate."),
    )


def _summary(
    *,
    proposal: dict,
    candidate: dict,
    preflight: dict,
    acceptance: dict | None,
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
        experiments=_build_experiments(
            proposal=proposal,
            candidate=candidate,
            preflight=preflight,
            acceptance=acceptance,
            decision=decision,
        ),
        next_action=next_action,
        rationale=rationale,
    )


def _build_experiments(
    *,
    proposal: dict,
    candidate: dict,
    preflight: dict,
    acceptance: dict | None,
    decision: FalsificationDecision,
) -> list[FalsificationExperiment]:
    if not candidate:
        return []

    proposal_id = proposal.get("proposal_id", "")
    readiness = preflight.get("readiness")
    trigger_ids = list(preflight.get("trigger_problem_ids", []))
    control_ids = list(preflight.get("control_problem_ids", []))
    outside_ids = list(preflight.get("outside_active_problem_ids", []))
    command_hint = preflight.get("command_hint", "")
    blocked = decision in {
        FalsificationDecision.BLOCKED_RETRIEVAL,
        FalsificationDecision.BLOCKED_SCOPE_RISK,
        FalsificationDecision.BLOCKED_UNDERPOWERED,
    }

    experiments = [
        _experiment(
            proposal_id=proposal_id,
            layer="V0",
            name="route_power_and_scope_probe",
            status=_route_probe_status(decision, readiness),
            hypothesis="The candidate activates on enough routed trigger rows and stays quiet on no-fire rows.",
            falsifier="A scoped preflight falsifies the candidate if it misses trigger coverage or activates outside its route.",
            trigger_problem_ids=trigger_ids,
            control_problem_ids=control_ids + outside_ids,
            command_hint=command_hint,
            measurement="trigger_n, active_trigger_n, outside_active_n",
            stop_rule="Stop before expensive model calls if trigger power or scope safety fails.",
            pass_rule="readiness == ready_for_fresh_ablation",
            fail_rule="readiness in {needs_retrieval_fix, needs_scope_fix, needs_more_trigger_rows}",
            type_i_control="Preflight is a routing gate only; it cannot promote a candidate.",
            observed={
                "readiness": readiness,
                "trigger_n": len(trigger_ids),
                "control_n": len(control_ids),
                "outside_active_n": len(outside_ids),
            },
        )
    ]

    experiments.extend([
        _experiment(
            proposal_id=proposal_id,
            layer="V3",
            name="trigger_benefit_sequential",
            status=_trigger_benefit_status(decision, acceptance, blocked),
            hypothesis="On routed trigger rows, forcing the candidate improves answer quality over the parent/baseline.",
            falsifier="Reject if fresh trigger judgments do not clear the lower-confidence benefit gate.",
            trigger_problem_ids=trigger_ids,
            command_hint=command_hint,
            measurement="win/tie/loss utility and trigger_lcb90 on judged trigger rows",
            stop_rule="Evaluate until min_trigger_judgments is met or the candidate is rejected by benefit.",
            pass_rule="trigger_lcb90 >= acceptance threshold and min_trigger_judgments is satisfied",
            fail_rule="trigger_lcb90 below threshold, or insufficient judgments after the budget is exhausted",
            type_i_control="Sequential lower-bound gate; no graph mutation from raw win count alone.",
            observed=_acceptance_observed(acceptance, "trigger"),
        ),
        _experiment(
            proposal_id=proposal_id,
            layer="V3",
            name="control_harm_sequential",
            status=_control_harm_status(decision, acceptance, blocked),
            hypothesis="On routed controls/no-fire rows, the candidate does not introduce regression harm.",
            falsifier="Reject or narrow scope if control loss upper bound exceeds the harm gate.",
            control_problem_ids=control_ids,
            command_hint=command_hint,
            measurement="control loss rate and control_loss_ucb90",
            stop_rule="Run after trigger benefit is testable; stop immediately on high-confidence control harm.",
            pass_rule="control_loss_ucb90 is absent or <= acceptance threshold",
            fail_rule="control_loss_ucb90 > acceptance threshold",
            type_i_control="Separate control-harm upper-bound gate prevents trigger-only overfitting.",
            observed=_acceptance_observed(acceptance, "control"),
        ),
        _experiment(
            proposal_id=proposal_id,
            layer="V3",
            name="placebo_context_control",
            status="blocked" if blocked else "planned",
            hypothesis="The observed gain comes from the candidate assumption, not from longer or more salient prompt context.",
            falsifier="Compare against a length-matched placebo context on the same routed rows.",
            trigger_problem_ids=trigger_ids,
            control_problem_ids=control_ids,
            command_hint=_append_placeholder(command_hint, "--assumption-placebo-context length_matched"),
            measurement="candidate-vs-placebo pairwise judgments on routed trigger/control rows",
            stop_rule="Run only after preflight passes and before accepting high-impact candidates.",
            pass_rule="candidate beats length-matched placebo on trigger rows without extra control harm",
            fail_rule="placebo matches or beats candidate, indicating exemplar/context boost rather than a valid assumption",
            type_i_control="Placebo split guards against selection bias and generic-context boost.",
            observed={"acceptance_decision": (acceptance or {}).get("decision")},
        ),
        _experiment(
            proposal_id=proposal_id,
            layer="V4",
            name="fresh_cross_judge_replay",
            status=_fresh_replay_status(decision, acceptance, blocked),
            hypothesis="The candidate survives fresh-row and judge-direction replay, not just cached or one-sided scoring.",
            falsifier="Replay candidate-vs-baseline and baseline-vs-candidate judgments on fresh routed rows.",
            trigger_problem_ids=trigger_ids,
            control_problem_ids=control_ids,
            command_hint=_append_placeholder(command_hint, "--fresh-split --bidirectional-judge"),
            measurement="bidirectional fresh split acceptance decision",
            stop_rule="Stop after accepted/rejected decision is produced for the proposal id.",
            pass_rule="candidate decision == accept after trigger/control gates",
            fail_rule="decision in {reject_benefit, reject_harm}",
            type_i_control="Bidirectional fresh replay reduces cached signal, side bias, and single-judge style preference.",
            observed=_acceptance_observed(acceptance, "fresh"),
        ),
    ])
    return experiments


def _experiment(
    *,
    proposal_id: str,
    layer: str,
    name: str,
    status: str,
    hypothesis: str,
    falsifier: str,
    trigger_problem_ids: list[str] | None = None,
    control_problem_ids: list[str] | None = None,
    command_hint: str = "",
    measurement: str = "",
    stop_rule: str = "",
    pass_rule: str = "",
    fail_rule: str = "",
    type_i_control: str = "",
    observed: dict | None = None,
) -> FalsificationExperiment:
    return FalsificationExperiment(
        experiment_id=stable_id("fexp", proposal_id, name),
        proposal_id=proposal_id,
        layer=layer,
        name=name,
        status=status,
        hypothesis=hypothesis,
        falsifier=falsifier,
        trigger_problem_ids=trigger_problem_ids or [],
        control_problem_ids=control_problem_ids or [],
        command_hint=command_hint,
        measurement=measurement,
        stop_rule=stop_rule,
        pass_rule=pass_rule,
        fail_rule=fail_rule,
        type_i_control=type_i_control,
        observed=observed or {},
    )


def _route_probe_status(decision: FalsificationDecision, readiness: str | None) -> str:
    if decision == FalsificationDecision.MANIFEST_ONLY:
        return "not_applicable"
    if readiness == "ready_for_fresh_ablation":
        return "passed"
    if readiness:
        return "failed"
    return "blocked"


def _trigger_benefit_status(
    decision: FalsificationDecision,
    acceptance: dict | None,
    blocked: bool,
) -> str:
    if blocked:
        return "blocked"
    if not acceptance:
        return "planned"
    if decision == FalsificationDecision.ACCEPT:
        return "passed"
    if decision == FalsificationDecision.REJECT_BENEFIT:
        return "failed"
    if decision == FalsificationDecision.REJECT_HARM:
        return "passed"
    return "inconclusive"


def _control_harm_status(
    decision: FalsificationDecision,
    acceptance: dict | None,
    blocked: bool,
) -> str:
    if blocked:
        return "blocked"
    if not acceptance:
        return "planned"
    if decision == FalsificationDecision.REJECT_HARM:
        return "failed"
    if decision in {FalsificationDecision.ACCEPT, FalsificationDecision.REJECT_BENEFIT}:
        return "passed"
    return "inconclusive"


def _fresh_replay_status(
    decision: FalsificationDecision,
    acceptance: dict | None,
    blocked: bool,
) -> str:
    if blocked:
        return "blocked"
    if not acceptance:
        return "planned"
    if decision == FalsificationDecision.ACCEPT:
        return "passed"
    if decision in {FalsificationDecision.REJECT_BENEFIT, FalsificationDecision.REJECT_HARM}:
        return "failed"
    return "inconclusive"


def _acceptance_observed(acceptance: dict | None, view: str) -> dict:
    if not acceptance:
        return {}
    if view == "trigger":
        return {
            "decision": acceptance.get("decision"),
            "trigger_outcomes": acceptance.get("trigger_outcomes", {}),
            "trigger_lcb90": acceptance.get("trigger_lcb90"),
            "judged_trigger_problem_ids": acceptance.get("judged_trigger_problem_ids", []),
        }
    if view == "control":
        return {
            "decision": acceptance.get("decision"),
            "control_outcomes": acceptance.get("control_outcomes", {}),
            "control_loss_ucb90": acceptance.get("control_loss_ucb90"),
            "judged_control_problem_ids": acceptance.get("judged_control_problem_ids", []),
        }
    return {
        "decision": acceptance.get("decision"),
        "rationale": acceptance.get("rationale"),
        "trigger_outcomes": acceptance.get("trigger_outcomes", {}),
        "control_outcomes": acceptance.get("control_outcomes", {}),
    }


def _append_placeholder(command_hint: str, suffix: str) -> str:
    if not command_hint:
        return suffix.strip()
    return f"{command_hint} {suffix}".strip()
