"""Recursive assumption runner.

This module is the outer loop that turns one-shot self-evolution artifacts into
an auditable recursive problem-solving tree.  It does not pretend that external
experiments have already been run: open child frames record exactly which
verification, evidence, or repair subproblem must be solved before the parent
hypothesis can be updated.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .record_phase2_eval import PRIMARY_TYPES
from .schema import AssumptionNode, ResidualType, TrialManifest, TrialStatus, stable_id


class RecursiveFrameType(str, Enum):
    ROOT_PROBLEM = "root_problem"
    CANDIDATE_HYPOTHESIS = "candidate_hypothesis"
    VERIFICATION_SUBPROBLEM = "verification_subproblem"
    EVIDENCE_SUBPROBLEM = "evidence_subproblem"
    REPAIR_SUBPROBLEM = "repair_subproblem"
    TERMINAL_RECORD = "terminal_record"


class RecursiveFrameStatus(str, Enum):
    OPEN = "open"
    READY_TO_ACT = "ready_to_act"
    WAITING_FOR_EVIDENCE = "waiting_for_evidence"
    RESOLVED = "resolved"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass
class RecursiveFrame:
    frame_id: str
    frame_type: RecursiveFrameType
    status: RecursiveFrameStatus
    depth: int
    problem_id: str
    goal: str
    hypothesis: str
    expected_observation: str
    verifier: str
    parent_frame_id: str | None = None
    assumption_ids: list[str] = field(default_factory=list)
    child_frame_ids: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    argument: dict = field(default_factory=dict)
    next_action: str = ""
    command_hint: str = ""
    residual_type: str | None = None
    return_update: dict = field(default_factory=dict)
    priority: float = 0.0
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frame_type"] = self.frame_type.value
        d["status"] = self.status.value
        return d


def build_recursive_assumption_run(
    *,
    graph_dir: Path,
    problem: str,
    goal: str,
    eval_id: str,
    problem_id: str | None = None,
    evolution_payload: dict | None = None,
    acceptance_payload: dict | None = None,
    top_k: int = 6,
    max_children: int = 8,
    max_depth: int = 3,
    writeback: bool = False,
) -> dict:
    """Build a recursive assumption tree and optionally log frame manifests."""

    store = JsonlGraphStore(graph_dir)
    graph = SimpleAssumptionGraph(store)
    problem_id = problem_id or stable_id("problem", problem, goal)
    activated = graph.retrieve(
        "\n".join([problem, goal]),
        seeds=[problem_id, goal],
        top_k=top_k,
        candidate_types=PRIMARY_TYPES,
    )

    frames: list[RecursiveFrame] = []
    root = _root_frame(
        eval_id=eval_id,
        problem_id=problem_id,
        problem=problem,
        goal=goal,
        activated_nodes=activated.nodes,
    )
    frames.append(root)

    if evolution_payload:
        _extend_from_evolution_payload(
            frames=frames,
            root_frame=root,
            evolution_payload=evolution_payload,
            acceptance_payload=acceptance_payload,
            eval_id=eval_id,
            max_children=max_children,
            max_depth=max_depth,
        )
    else:
        _extend_from_retrieval_only(
            frames=frames,
            root_frame=root,
            activated_nodes=activated.nodes,
            eval_id=eval_id,
            max_children=max_children,
            max_depth=max_depth,
        )

    if writeback:
        for frame in frames:
            store.append_trial(TrialManifest.from_dict(frame.manifest))
        store.flush()

    payload = {
        "eval_id": eval_id,
        "mode": {
            "writeback": writeback,
            "top_k": top_k,
            "max_children": max_children,
            "max_depth": max_depth,
            "source": "evolution_payload" if evolution_payload else "retrieval_only",
            "acceptance_payload": bool(acceptance_payload),
        },
        "root": {
            "problem_id": problem_id,
            "problem": problem,
            "goal": goal,
            "activated_assumption_ids": [node.id for node in activated.nodes],
        },
        "frame_counts": dict(Counter(frame.frame_type.value for frame in frames)),
        "status_counts": dict(Counter(frame.status.value for frame in frames)),
        "depth_counts": dict(Counter(str(frame.depth) for frame in frames)),
        "recursion_edges": [
            {"parent_frame_id": frame.frame_id, "child_frame_id": child_id}
            for frame in frames
            for child_id in frame.child_frame_ids
        ],
        "open_frame_ids": [
            frame.frame_id
            for frame in frames
            if frame.status in {
                RecursiveFrameStatus.OPEN,
                RecursiveFrameStatus.READY_TO_ACT,
                RecursiveFrameStatus.WAITING_FOR_EVIDENCE,
            }
        ],
        "next_actions": _next_actions(frames),
        "frames": [frame.to_dict() for frame in frames],
    }
    return payload


def _extend_from_evolution_payload(
    *,
    frames: list[RecursiveFrame],
    root_frame: RecursiveFrame,
    evolution_payload: dict,
    acceptance_payload: dict | None,
    eval_id: str,
    max_children: int,
    max_depth: int,
) -> None:
    proposals = evolution_payload.get("proposals", {}).get("proposals", [])
    if not proposals:
        return
    indexes = _EvolutionIndexes(evolution_payload, acceptance_payload=acceptance_payload)
    for proposal in _ranked_proposals(proposals, indexes, max_children):
        candidate_frame = _candidate_frame(
            proposal=proposal,
            indexes=indexes,
            eval_id=eval_id,
            parent_frame_id=root_frame.frame_id,
            parent_trial_id=root_frame.manifest.get("trial_id"),
            depth=1,
        )
        frames.append(candidate_frame)
        root_frame.child_frame_ids.append(candidate_frame.frame_id)
        child = _child_frame_for_candidate(
            candidate_frame=candidate_frame,
            proposal=proposal,
            indexes=indexes,
            eval_id=eval_id,
            max_depth=max_depth,
        )
        if child:
            frames.append(child)
            candidate_frame.child_frame_ids.append(child.frame_id)


def _extend_from_retrieval_only(
    *,
    frames: list[RecursiveFrame],
    root_frame: RecursiveFrame,
    activated_nodes: list[AssumptionNode],
    eval_id: str,
    max_children: int,
    max_depth: int,
) -> None:
    if max_depth < 1:
        return
    for node in activated_nodes[:max_children]:
        frame = _applicability_frame(
            node=node,
            eval_id=eval_id,
            parent_frame_id=root_frame.frame_id,
            depth=1,
        )
        frames.append(frame)
        root_frame.child_frame_ids.append(frame.frame_id)


@dataclass
class _EvolutionIndexes:
    preflight_by_id: dict
    falsification_by_id: dict
    bayes_by_id: dict
    policy_by_id: dict
    regression_by_id: dict
    formal_gate_by_id: dict
    acceptance_by_id: dict

    def __init__(self, payload: dict, *, acceptance_payload: dict | None = None):
        self.preflight_by_id = {
            row.get("proposal_id"): row
            for row in payload.get("candidate_preflight", {}).get("summaries", [])
        }
        self.falsification_by_id = {
            row.get("proposal_id"): row
            for row in payload.get("falsification_gate", {}).get("summaries", [])
        }
        self.bayes_by_id = {
            row.get("proposal_id"): row
            for row in payload.get("bayesian_policy", {}).get("scores", [])
        }
        self.policy_by_id = {
            row.get("proposal_id"): row
            for row in payload.get("policy_update_plan", {}).get("actions", [])
        }
        self.regression_by_id = {
            row.get("proposal_id"): row
            for row in payload.get("regression_predictions", [])
        }
        self.formal_gate_by_id = {
            row.get("proposal_id"): row
            for row in payload.get("formal_mapping_gate", {}).get("gates", [])
        }
        source_acceptance = (
            acceptance_payload
            if acceptance_payload is not None
            else payload.get("candidate_acceptance") or {}
        )
        self.acceptance_by_id = {
            row.get("proposal_id"): row
            for row in source_acceptance.get("summaries", [])
        }


def _root_frame(
    *,
    eval_id: str,
    problem_id: str,
    problem: str,
    goal: str,
    activated_nodes: list[AssumptionNode],
) -> RecursiveFrame:
    assumption_ids = [node.id for node in activated_nodes]
    hypothesis = "The goal can be solved by recursively selecting, testing, and updating explicit assumptions."
    manifest = TrialManifest(
        problem_id=problem_id,
        action_type="recursive_root",
        component="recursive_assumption_runner",
        assumption=hypothesis,
        why_selected="User requested recursive self-argument rather than a one-shot proposal queue.",
        expected_effect="Produce an auditable stack of hypothesis frames and child verification tasks.",
        assumption_ids=assumption_ids,
        verifier="recursive_frame_audit",
        verification_plan="Every open child frame must define the observation needed to update its parent.",
        rollback_condition="Reject a child hypothesis if its verifier fails or creates unacceptable regression risk.",
        status=TrialStatus.PENDING,
        metadata={"eval_id": eval_id, "frame_type": RecursiveFrameType.ROOT_PROBLEM.value},
        trial_id=stable_id("trial", eval_id, problem_id, "recursive_root"),
    )
    return RecursiveFrame(
        frame_id=stable_id("rframe", eval_id, problem_id, "root"),
        frame_type=RecursiveFrameType.ROOT_PROBLEM,
        status=RecursiveFrameStatus.OPEN,
        depth=0,
        problem_id=problem_id,
        goal=goal,
        hypothesis=hypothesis,
        expected_observation="Child frames either resolve the goal or return a structured residual.",
        verifier="recursive_frame_audit",
        assumption_ids=assumption_ids,
        source={"problem": problem, "activated_assumption_ids": assumption_ids},
        argument={
            "support": [
                f"activated {len(assumption_ids)} assumptions from graph memory",
                "existing self-evolution artifacts can provide candidate, gate, and policy evidence",
            ],
            "objections": [
                "external verification is still required for open child frames",
            ],
            "falsification_tests": [
                "each child frame must specify a verifier and parent return update",
            ],
        },
        next_action="expand_highest_priority_child_hypotheses",
        manifest=manifest.to_dict(),
    )


def _candidate_frame(
    *,
    proposal: dict,
    indexes: _EvolutionIndexes,
    eval_id: str,
    parent_frame_id: str,
    parent_trial_id: str | None,
    depth: int,
) -> RecursiveFrame:
    proposal_id = proposal.get("proposal_id", "")
    candidate = proposal.get("candidate_node") or {}
    preflight = indexes.preflight_by_id.get(proposal_id, {})
    falsification = indexes.falsification_by_id.get(proposal_id, {})
    bayes = indexes.bayes_by_id.get(proposal_id, {})
    policy = indexes.policy_by_id.get(proposal_id, {})
    regression = indexes.regression_by_id.get(proposal_id, {})
    formal_gate = indexes.formal_gate_by_id.get(proposal_id, {})
    acceptance = indexes.acceptance_by_id.get(proposal_id, {})
    action = _recommended_action(policy, bayes, falsification, preflight, acceptance)
    status = _status_for_action(action)
    hypothesis = candidate.get("claim") or (proposal.get("manifest") or {}).get("assumption", "")
    if not hypothesis:
        hypothesis = f"Proposal {proposal_id} records a lifecycle decision for {proposal.get('parent_node_id', '')}."
    assumption_ids = [
        x
        for x in [proposal.get("parent_node_id"), candidate.get("id")]
        if x
    ]
    priority = float(bayes.get("posterior_priority", proposal.get("priority", 0.0)) or 0.0)
    manifest = TrialManifest(
        problem_id=f"recursive::{proposal_id}",
        action_type="recursive_candidate_hypothesis",
        component="recursive_assumption_runner",
        assumption=hypothesis,
        why_selected=_why_selected(proposal, bayes, policy),
        expected_effect=_expected_effect(proposal, preflight, bayes),
        assumption_ids=assumption_ids,
        verifier=falsification.get("next_action") or bayes.get("recommended_action") or "recursive_gate",
        verification_plan="Resolve the child verification or repair frame, then update this candidate frame.",
        rollback_condition="Do not promote if acceptance, scope, retrieval, or formal-mapping gates fail.",
        status=TrialStatus.PENDING,
        parent_trial_id=parent_trial_id,
        metadata={
            "eval_id": eval_id,
            "frame_type": RecursiveFrameType.CANDIDATE_HYPOTHESIS.value,
            "proposal_id": proposal_id,
            "recommended_action": action,
        },
        artifacts={
            "proposal": _compact_proposal(proposal),
            "preflight": preflight,
            "falsification": falsification,
            "bayesian_policy": bayes,
            "formal_mapping_gate": formal_gate,
            "regression_prediction": regression,
            "acceptance": acceptance,
        },
        trial_id=stable_id("trial", eval_id, proposal_id, "recursive_candidate"),
    )
    return RecursiveFrame(
        frame_id=stable_id("rframe", eval_id, proposal_id, "candidate"),
        frame_type=RecursiveFrameType.CANDIDATE_HYPOTHESIS,
        status=status,
        depth=depth,
        parent_frame_id=parent_frame_id,
        problem_id=f"proposal::{proposal_id}",
        goal=_candidate_goal(action),
        hypothesis=hypothesis,
        expected_observation=_expected_effect(proposal, preflight, bayes),
        verifier=manifest.verifier or "recursive_gate",
        assumption_ids=assumption_ids,
        source={
            "proposal_id": proposal_id,
            "proposal_type": proposal.get("proposal_type"),
            "parent_node_id": proposal.get("parent_node_id"),
            "candidate_node_id": candidate.get("id"),
        },
        argument=_argument_map(
            proposal=proposal,
            preflight=preflight,
            falsification=falsification,
            bayes=bayes,
            policy=policy,
            regression=regression,
            formal_gate=formal_gate,
            acceptance=acceptance,
        ),
        next_action=action,
        command_hint=bayes.get("command_hint") or preflight.get("command_hint", ""),
        residual_type=_residual_type_for_action(action),
        return_update={
            "parent_frame_id": parent_frame_id,
            "on_success": "candidate can advance to acceptance/application or close the parent gap",
            "on_failure": "classify residual and generate a narrower child frame",
        },
        priority=priority,
        manifest=manifest.to_dict(),
    )


def _child_frame_for_candidate(
    *,
    candidate_frame: RecursiveFrame,
    proposal: dict,
    indexes: _EvolutionIndexes,
    eval_id: str,
    max_depth: int,
) -> RecursiveFrame | None:
    if candidate_frame.depth + 1 > max_depth:
        return None
    action = candidate_frame.next_action
    proposal_id = proposal.get("proposal_id", "")
    preflight = indexes.preflight_by_id.get(proposal_id, {})
    falsification = indexes.falsification_by_id.get(proposal_id, {})
    bayes = indexes.bayes_by_id.get(proposal_id, {})
    acceptance = indexes.acceptance_by_id.get(proposal_id, {})

    if acceptance:
        return _verification_child(
            candidate_frame=candidate_frame,
            proposal_id=proposal_id,
            preflight=preflight,
            falsification=falsification,
            bayes=bayes,
            acceptance=acceptance,
            eval_id=eval_id,
        )
    if action in {"run_fresh_ablation", "run_ablation", "run_fresh_ablation_before_promotion"}:
        return _verification_child(
            candidate_frame=candidate_frame,
            proposal_id=proposal_id,
            preflight=preflight,
            falsification=falsification,
            bayes=bayes,
            acceptance={},
            eval_id=eval_id,
        )
    if action in {"collect_more_evidence", "collect_evidence", "collect_more_trigger_rows"}:
        return _evidence_child(
            candidate_frame=candidate_frame,
            proposal_id=proposal_id,
            preflight=preflight,
            eval_id=eval_id,
        )
    if action in {"repair_scope", "narrow_scope_before_ablation", "block_unsafe_formal_mapping"}:
        return _repair_child(
            candidate_frame=candidate_frame,
            proposal_id=proposal_id,
            repair_kind="scope",
            source=preflight or falsification,
            eval_id=eval_id,
        )
    if action in {"repair_retrieval", "repair_retrieval_before_ablation"}:
        return _repair_child(
            candidate_frame=candidate_frame,
            proposal_id=proposal_id,
            repair_kind="retrieval",
            source=preflight or falsification,
            eval_id=eval_id,
        )
    return None


def _verification_child(
    *,
    candidate_frame: RecursiveFrame,
    proposal_id: str,
    preflight: dict,
    falsification: dict,
    bayes: dict,
    acceptance: dict,
    eval_id: str,
) -> RecursiveFrame:
    trigger_ids = preflight.get("trigger_problem_ids", [])
    control_ids = preflight.get("control_problem_ids", [])
    acceptance_decision = acceptance.get("decision")
    child_status = _verification_child_status(acceptance_decision)
    next_action = _verification_child_next_action(acceptance_decision)
    hypothesis = "Fresh ablation on routed trigger/control rows can decide whether the parent candidate should be promoted."
    manifest = _child_manifest(
        eval_id=eval_id,
        proposal_id=proposal_id,
        child_kind=RecursiveFrameType.VERIFICATION_SUBPROBLEM,
        parent=candidate_frame,
        hypothesis=hypothesis,
        expected_effect="Accepted judgments should show trigger benefit without control harm.",
        verifier="candidate_acceptance_gate",
        action_type="recursive_fresh_ablation_subproblem",
    )
    return RecursiveFrame(
        frame_id=stable_id("rframe", eval_id, proposal_id, "fresh_ablation"),
        frame_type=RecursiveFrameType.VERIFICATION_SUBPROBLEM,
        status=child_status,
        depth=candidate_frame.depth + 1,
        parent_frame_id=candidate_frame.frame_id,
        problem_id=f"verify::{proposal_id}",
        goal=(
            "Return fresh ablation evidence to the parent candidate."
            if acceptance
            else "Run the minimal fresh ablation needed to update the parent candidate."
        ),
        hypothesis=hypothesis,
        expected_observation=_verification_expected_observation(acceptance_decision),
        verifier="candidate_acceptance_gate",
        assumption_ids=candidate_frame.assumption_ids,
        source={
            "proposal_id": proposal_id,
            "trigger_problem_ids": trigger_ids,
            "control_problem_ids": control_ids,
            "falsification_decision": falsification.get("decision"),
            "acceptance": acceptance,
        },
        argument={
            "support": [
                f"preflight readiness={preflight.get('readiness')}",
                f"trigger_rows={len(trigger_ids)}",
                f"control_rows={len(control_ids)}",
                f"bayesian_priority={bayes.get('posterior_priority')}",
                *(_acceptance_support(acceptance) if acceptance else []),
            ],
            "objections": _acceptance_objections(acceptance),
            "falsification_tests": [
                "trigger benefit lower bound clears acceptance gate",
                "control loss upper bound stays below harm gate",
            ],
        },
        next_action=next_action,
        command_hint=bayes.get("command_hint") or preflight.get("command_hint", ""),
        return_update=_verification_return_update(candidate_frame.frame_id, acceptance_decision),
        priority=candidate_frame.priority,
        manifest=manifest.to_dict(),
    )


def _verification_child_status(decision: str | None) -> RecursiveFrameStatus:
    if not decision:
        return RecursiveFrameStatus.READY_TO_ACT
    if decision in {"accept", "reject_benefit", "reject_harm"}:
        return RecursiveFrameStatus.RESOLVED
    if decision == "insufficient_judgments":
        return RecursiveFrameStatus.WAITING_FOR_EVIDENCE
    if decision in {"deferred_not_ready", "manifest_only"}:
        return RecursiveFrameStatus.DEFERRED
    return RecursiveFrameStatus.OPEN


def _verification_child_next_action(decision: str | None) -> str:
    if decision == "accept":
        return "return_accept_to_parent"
    if decision == "reject_harm":
        return "return_harm_rejection_to_parent"
    if decision == "reject_benefit":
        return "return_benefit_rejection_to_parent"
    if decision == "insufficient_judgments":
        return "collect_more_judgments"
    if decision in {"deferred_not_ready", "manifest_only"}:
        return "wait_for_parent_preflight"
    return "run_fresh_ablation"


def _verification_expected_observation(decision: str | None) -> str:
    if decision == "accept":
        return "Fresh judgments accepted the candidate; parent can request gated application."
    if decision == "reject_harm":
        return "Fresh judgments found control harm; parent should narrow scope or reject."
    if decision == "reject_benefit":
        return "Fresh judgments did not show enough trigger benefit; parent should revise or reject."
    if decision == "insufficient_judgments":
        return "More judgments are needed before the parent can be updated."
    return "Candidate wins on trigger rows and does not add outside-control harm."


def _acceptance_support(acceptance: dict) -> list[str]:
    decision = acceptance.get("decision")
    if not decision:
        return []
    fields = [f"acceptance_decision={decision}"]
    if acceptance.get("trigger_utility") is not None:
        fields.append(f"trigger_utility={acceptance.get('trigger_utility')}")
    if acceptance.get("trigger_lcb90") is not None:
        fields.append(f"trigger_lcb90={acceptance.get('trigger_lcb90')}")
    if acceptance.get("control_loss_ucb90") is not None:
        fields.append(f"control_loss_ucb90={acceptance.get('control_loss_ucb90')}")
    return fields


def _acceptance_objections(acceptance: dict) -> list[str]:
    if not acceptance:
        return ["fresh candidate judgments are not available yet"]
    decision = acceptance.get("decision")
    rationale = acceptance.get("rationale", "")
    if decision == "accept":
        return []
    if decision == "reject_harm":
        return [f"control harm rejected candidate: {rationale}".strip()]
    if decision == "reject_benefit":
        return [f"trigger benefit rejected candidate: {rationale}".strip()]
    if decision == "insufficient_judgments":
        return [f"insufficient judgments: {rationale}".strip()]
    return [f"acceptance gate returned {decision}: {rationale}".strip()]


def _verification_return_update(parent_frame_id: str, decision: str | None) -> dict:
    if decision == "accept":
        return {
            "parent_frame_id": parent_frame_id,
            "outcome": "accepted",
            "parent_next_action": "apply_accepted_candidate_if_requested",
            "on_success": "mark parent candidate accepted and eligible for gated apply",
            "on_failure": "not applicable; verification already accepted",
        }
    if decision == "reject_harm":
        return {
            "parent_frame_id": parent_frame_id,
            "outcome": "rejected_harm",
            "parent_next_action": "reject_or_narrow_scope",
            "on_success": "parent should create a scope-repair child or reject the candidate",
            "on_failure": "keep parent blocked until scope risk is addressed",
        }
    if decision == "reject_benefit":
        return {
            "parent_frame_id": parent_frame_id,
            "outcome": "rejected_benefit",
            "parent_next_action": "reject_or_revise_candidate",
            "on_success": "parent should create a revision child or reject the candidate",
            "on_failure": "keep parent unresolved until a stronger hypothesis exists",
        }
    if decision == "insufficient_judgments":
        return {
            "parent_frame_id": parent_frame_id,
            "outcome": "underpowered",
            "parent_next_action": "collect_more_judgments",
            "on_success": "rerun acceptance when enough judgments exist",
            "on_failure": "parent remains waiting for evidence",
        }
    return {
        "parent_frame_id": parent_frame_id,
        "on_success": "mark parent candidate accepted or ready to apply",
        "on_failure": "reject, narrow, or revise the parent candidate",
    }


def _evidence_child(
    *,
    candidate_frame: RecursiveFrame,
    proposal_id: str,
    preflight: dict,
    eval_id: str,
) -> RecursiveFrame:
    hypothesis = "Collecting more routed trigger rows will make the parent candidate falsifiable."
    manifest = _child_manifest(
        eval_id=eval_id,
        proposal_id=proposal_id,
        child_kind=RecursiveFrameType.EVIDENCE_SUBPROBLEM,
        parent=candidate_frame,
        hypothesis=hypothesis,
        expected_effect="Additional trigger rows should turn an underpowered gate into a fair ablation setup.",
        verifier="candidate_preflight_gate",
        action_type="recursive_evidence_subproblem",
    )
    return RecursiveFrame(
        frame_id=stable_id("rframe", eval_id, proposal_id, "collect_evidence"),
        frame_type=RecursiveFrameType.EVIDENCE_SUBPROBLEM,
        status=RecursiveFrameStatus.WAITING_FOR_EVIDENCE,
        depth=candidate_frame.depth + 1,
        parent_frame_id=candidate_frame.frame_id,
        problem_id=f"evidence::{proposal_id}",
        goal="Find or sample enough trigger rows for the parent candidate.",
        hypothesis=hypothesis,
        expected_observation="The parent candidate reaches the minimum trigger and active-trigger counts.",
        verifier="candidate_preflight_gate",
        assumption_ids=candidate_frame.assumption_ids,
        source={
            "proposal_id": proposal_id,
            "readiness": preflight.get("readiness"),
            "trigger_problem_ids": preflight.get("trigger_problem_ids", []),
            "missed_trigger_problem_ids": preflight.get("missed_trigger_problem_ids", []),
        },
        argument={
            "support": [
                f"current_trigger_rows={len(preflight.get('trigger_problem_ids', []))}",
            ],
            "objections": [
                "sample is underpowered for a promotion-sensitive decision",
            ],
            "falsification_tests": [
                "rerun preflight after collecting more examples",
            ],
        },
        next_action="collect_more_trigger_rows",
        return_update={
            "parent_frame_id": candidate_frame.frame_id,
            "on_success": "parent moves to fresh ablation",
            "on_failure": "parent remains manifest-only or is rejected as untestable",
        },
        priority=candidate_frame.priority,
        manifest=manifest.to_dict(),
    )


def _repair_child(
    *,
    candidate_frame: RecursiveFrame,
    proposal_id: str,
    repair_kind: str,
    source: dict,
    eval_id: str,
) -> RecursiveFrame:
    is_scope = repair_kind == "scope"
    hypothesis = (
        "A narrower activation scope will remove no-fire exposure before ablation."
        if is_scope
        else "A retrieval repair will expose the candidate on its routed trigger rows."
    )
    manifest = _child_manifest(
        eval_id=eval_id,
        proposal_id=proposal_id,
        child_kind=RecursiveFrameType.REPAIR_SUBPROBLEM,
        parent=candidate_frame,
        hypothesis=hypothesis,
        expected_effect="Repair should unblock the parent candidate without changing its core claim.",
        verifier="candidate_preflight_gate",
        action_type=f"recursive_{repair_kind}_repair_subproblem",
    )
    return RecursiveFrame(
        frame_id=stable_id("rframe", eval_id, proposal_id, f"repair_{repair_kind}"),
        frame_type=RecursiveFrameType.REPAIR_SUBPROBLEM,
        status=RecursiveFrameStatus.OPEN,
        depth=candidate_frame.depth + 1,
        parent_frame_id=candidate_frame.frame_id,
        problem_id=f"repair::{proposal_id}",
        goal=f"Repair {repair_kind} before the parent candidate can be judged.",
        hypothesis=hypothesis,
        expected_observation="Preflight readiness improves without introducing new regression risk.",
        verifier="candidate_preflight_gate",
        assumption_ids=candidate_frame.assumption_ids,
        source={"proposal_id": proposal_id, "repair_kind": repair_kind, "gate_source": source},
        argument={
            "support": [f"parent next_action={candidate_frame.next_action}"],
            "objections": [source.get("rationale", "gate blocked direct promotion")],
            "falsification_tests": ["rerun candidate preflight after repair"],
        },
        next_action=f"repair_{repair_kind}",
        residual_type=(
            ResidualType.ASSUMPTION_DEFECT.value
            if is_scope
            else ResidualType.MEMORY_DEFECT.value
        ),
        return_update={
            "parent_frame_id": candidate_frame.frame_id,
            "on_success": "parent moves to fresh ablation",
            "on_failure": "generate a narrower residual-derived child or reject parent",
        },
        priority=candidate_frame.priority,
        manifest=manifest.to_dict(),
    )


def _applicability_frame(
    *,
    node: AssumptionNode,
    eval_id: str,
    parent_frame_id: str,
    depth: int,
) -> RecursiveFrame:
    hypothesis = f"{node.id} is relevant to the root problem: {node.claim}"
    manifest = TrialManifest(
        problem_id=f"recursive_applicability::{node.id}",
        action_type="recursive_applicability_probe",
        component="recursive_assumption_runner",
        assumption=hypothesis,
        why_selected="Graph retrieval activated this assumption for the root problem.",
        expected_effect="A verifier can decide whether this assumption should shape the next action.",
        assumption_ids=[node.id],
        verifier="applicability_check",
        verification_plan="Ask whether the assumption should fire, then create a candidate or residual child frame.",
        rollback_condition="If the assumption is irrelevant, mark as no-fire for this root context.",
        status=TrialStatus.PENDING,
        metadata={"eval_id": eval_id, "frame_type": RecursiveFrameType.CANDIDATE_HYPOTHESIS.value},
        trial_id=stable_id("trial", eval_id, node.id, "recursive_applicability"),
    )
    return RecursiveFrame(
        frame_id=stable_id("rframe", eval_id, node.id, "applicability"),
        frame_type=RecursiveFrameType.CANDIDATE_HYPOTHESIS,
        status=RecursiveFrameStatus.OPEN,
        depth=depth,
        parent_frame_id=parent_frame_id,
        problem_id=f"applicability::{node.id}",
        goal="Decide whether the retrieved assumption should shape a concrete action.",
        hypothesis=hypothesis,
        expected_observation="Applicability verifier returns should_fire with a concrete expected effect.",
        verifier="applicability_check",
        assumption_ids=[node.id],
        source={"node_id": node.id, "node_type": str(getattr(node.type, "value", node.type))},
        argument={
            "support": [f"confidence={node.confidence}", f"metaproductivity={node.metaproductivity}"],
            "objections": ["retrieval relevance has not been verified by a task-specific gate"],
            "falsification_tests": ["conditioned route should_fire/no_fire check"],
        },
        next_action="verify_applicability",
        priority=node.confidence + node.metaproductivity,
        manifest=manifest.to_dict(),
    )


def _child_manifest(
    *,
    eval_id: str,
    proposal_id: str,
    child_kind: RecursiveFrameType,
    parent: RecursiveFrame,
    hypothesis: str,
    expected_effect: str,
    verifier: str,
    action_type: str,
) -> TrialManifest:
    return TrialManifest(
        problem_id=f"{child_kind.value}::{proposal_id}",
        action_type=action_type,
        component="recursive_assumption_runner",
        assumption=hypothesis,
        why_selected=f"Parent frame {parent.frame_id} cannot be resolved without this child subproblem.",
        expected_effect=expected_effect,
        assumption_ids=parent.assumption_ids,
        verifier=verifier,
        verification_plan="Resolve this child frame and propagate success/failure to its parent frame.",
        rollback_condition="If this child fails, keep parent unresolved and generate a repair or rejection residual.",
        status=TrialStatus.PENDING,
        parent_trial_id=parent.manifest.get("trial_id"),
        metadata={
            "eval_id": eval_id,
            "frame_type": child_kind.value,
            "proposal_id": proposal_id,
            "parent_frame_id": parent.frame_id,
        },
        trial_id=stable_id("trial", eval_id, proposal_id, child_kind.value),
    )


def _ranked_proposals(proposals: list[dict], indexes: _EvolutionIndexes, max_children: int) -> list[dict]:
    def key(proposal: dict) -> tuple[float, str]:
        proposal_id = proposal.get("proposal_id", "")
        bayes = indexes.bayes_by_id.get(proposal_id, {})
        priority = float(bayes.get("posterior_priority", proposal.get("priority", 0.0)) or 0.0)
        return (-priority, proposal_id)

    return sorted(proposals, key=key)[:max_children]


def _recommended_action(
    policy: dict,
    bayes: dict,
    falsification: dict,
    preflight: dict,
    acceptance: dict,
) -> str:
    decision = acceptance.get("decision")
    if decision == "accept":
        return "apply_accepted_candidate_if_requested"
    if decision == "reject_harm":
        return "reject_or_narrow_scope"
    if decision == "reject_benefit":
        return "reject_or_revise_candidate"
    if decision == "insufficient_judgments":
        return "collect_more_judgments"
    if decision in {"deferred_not_ready", "manifest_only"}:
        return "wait_for_parent_preflight"
    if decision:
        return "inspect_acceptance_result"
    if policy.get("policy_action"):
        return policy["policy_action"]
    if bayes.get("recommended_action"):
        return bayes["recommended_action"]
    if falsification.get("next_action"):
        return falsification["next_action"]
    readiness = preflight.get("readiness")
    if readiness == "ready_for_fresh_ablation":
        return "run_fresh_ablation"
    if readiness == "needs_more_trigger_rows":
        return "collect_more_trigger_rows"
    if readiness == "needs_retrieval_fix":
        return "repair_retrieval"
    if readiness == "needs_scope_fix":
        return "repair_scope"
    return "record_only"


def _status_for_action(action: str) -> RecursiveFrameStatus:
    if action in {
        "run_fresh_ablation",
        "run_ablation",
        "run_fresh_ablation_before_promotion",
        "verify_applicability",
        "apply_accepted_candidate_if_requested",
        "reject_or_narrow_scope",
        "reject_or_revise_candidate",
        "inspect_acceptance_result",
    }:
        return RecursiveFrameStatus.READY_TO_ACT
    if action in {
        "collect_more_evidence",
        "collect_evidence",
        "collect_more_trigger_rows",
        "collect_more_judgments",
    }:
        return RecursiveFrameStatus.WAITING_FOR_EVIDENCE
    if action in {
        "repair_scope",
        "repair_retrieval",
        "narrow_scope_before_ablation",
        "repair_retrieval_before_ablation",
    }:
        return RecursiveFrameStatus.OPEN
    if action in {
        "record_manifest_only_no_graph_policy_change",
        "record_only",
        "applied_to_graph",
        "apply_accepted",
        "reject",
    }:
        return RecursiveFrameStatus.RESOLVED
    if action == "wait_for_parent_preflight":
        return RecursiveFrameStatus.DEFERRED
    if action == "block_unsafe_formal_mapping":
        return RecursiveFrameStatus.BLOCKED
    return RecursiveFrameStatus.OPEN


def _candidate_goal(action: str) -> str:
    mapping = {
        "run_fresh_ablation_before_promotion": "Test whether this candidate should be promoted.",
        "run_ablation": "Test whether this candidate should be promoted.",
        "collect_more_evidence": "Collect enough evidence to make this candidate falsifiable.",
        "collect_evidence": "Collect enough evidence to make this candidate falsifiable.",
        "collect_more_judgments": "Collect enough fresh judgments to decide this candidate.",
        "repair_scope": "Narrow candidate scope before judging quality.",
        "repair_retrieval": "Repair retrieval exposure before judging quality.",
        "apply_accepted_candidate_if_requested": "Apply this accepted candidate through the gated graph mutation path.",
        "reject_or_narrow_scope": "Reject this harmful candidate or create a narrower scope-repair child.",
        "reject_or_revise_candidate": "Reject this weak candidate or create a stronger revision child.",
        "wait_for_parent_preflight": "Wait for preflight readiness before judging this candidate.",
        "record_manifest_only_no_graph_policy_change": "Record the lifecycle decision without mutating graph policy.",
    }
    return mapping.get(action, f"Resolve candidate action: {action}")


def _why_selected(proposal: dict, bayes: dict, policy: dict) -> str:
    bits = []
    if proposal.get("rationale"):
        bits.append(proposal["rationale"])
    if bayes.get("rationale"):
        bits.extend(str(x) for x in bayes["rationale"])
    if policy.get("policy_action"):
        bits.append(f"policy_action={policy['policy_action']}")
    return " ".join(bits) or "Selected by recursive proposal ranking."


def _expected_effect(proposal: dict, preflight: dict, bayes: dict) -> str:
    candidate = proposal.get("candidate_node") or {}
    effects = candidate.get("predicted_effects") or []
    if effects:
        return "; ".join(str(x) for x in effects[:2])
    if preflight.get("readiness"):
        return f"Advance from preflight readiness={preflight.get('readiness')} to a resolved parent update."
    if bayes.get("recommended_action"):
        return f"Carry out recommended_action={bayes.get('recommended_action')}."
    return "Resolve the proposal through the recursive verifier stack."


def _argument_map(
    *,
    proposal: dict,
    preflight: dict,
    falsification: dict,
    bayes: dict,
    policy: dict,
    regression: dict,
    formal_gate: dict,
    acceptance: dict,
) -> dict:
    support = []
    objections = []
    tests = []
    if preflight:
        support.append(f"preflight readiness={preflight.get('readiness')}")
        support.append(f"active_trigger_rows={len(preflight.get('active_trigger_problem_ids', []))}")
    if bayes:
        support.append(f"bayesian recommended_action={bayes.get('recommended_action')}")
        support.append(f"posterior_priority={bayes.get('posterior_priority')}")
        support.append(f"expected_value={bayes.get('expected_value')}")
    if formal_gate:
        decision = formal_gate.get("decision")
        if formal_gate.get("blocks_policy_update"):
            objections.append(f"formal_mapping_gate blocks update: {decision}")
        else:
            support.append(f"formal_mapping_gate={decision}")
    if regression:
        risk = regression.get("risk")
        if risk in {"medium", "high"}:
            objections.append(f"regression_risk={risk}: {'; '.join(regression.get('reasons', []))}")
        else:
            support.append(f"regression_risk={risk}")
    decision = falsification.get("decision")
    if decision:
        tests.extend(check.get("name", "") for check in falsification.get("ordered_checks", []))
        if decision in {"ready_for_ablation", "accept"}:
            support.append(f"falsification_decision={decision}")
        elif decision != "manifest_only":
            objections.append(f"falsification_decision={decision}: {falsification.get('rationale', '')}")
    if acceptance:
        support.extend(_acceptance_support(acceptance))
        objections.extend(_acceptance_objections(acceptance))
        tests.append("candidate_acceptance_gate")
    if not proposal.get("candidate_node"):
        objections.append("proposal has no candidate node")
    if not tests:
        tests.append(policy.get("policy_action") or bayes.get("recommended_action") or "recursive_gate")
    return {
        "support": [x for x in support if x],
        "objections": [x for x in objections if x],
        "falsification_tests": [x for x in tests if x],
    }


def _residual_type_for_action(action: str) -> str | None:
    if "retrieval" in action:
        return ResidualType.MEMORY_DEFECT.value
    if "scope" in action or "formal" in action:
        return ResidualType.ASSUMPTION_DEFECT.value
    if "collect" in action:
        return ResidualType.UNKNOWN.value
    return None


def _compact_proposal(proposal: dict) -> dict:
    candidate = proposal.get("candidate_node") or {}
    return {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "parent_node_id": proposal.get("parent_node_id"),
        "candidate_node_id": candidate.get("id"),
        "priority": proposal.get("priority"),
    }


def _next_actions(frames: list[RecursiveFrame]) -> list[dict]:
    frames_by_id = {frame.frame_id: frame for frame in frames}

    def child_returned(child: RecursiveFrame) -> bool:
        return bool(child.return_update.get("parent_next_action"))

    def children_returned_or_terminal(frame: RecursiveFrame) -> bool:
        terminal = {
            RecursiveFrameStatus.RESOLVED,
            RecursiveFrameStatus.BLOCKED,
            RecursiveFrameStatus.DEFERRED,
        }
        for child_id in frame.child_frame_ids:
            child = frames_by_id.get(child_id)
            if not child:
                continue
            if child.status not in terminal and not child_returned(child):
                return False
        return True

    actionable = [
        frame
        for frame in frames
        if frame.status in {
            RecursiveFrameStatus.OPEN,
            RecursiveFrameStatus.READY_TO_ACT,
            RecursiveFrameStatus.WAITING_FOR_EVIDENCE,
        }
        and not child_returned(frame)
        and (not frame.child_frame_ids or children_returned_or_terminal(frame))
        and frame.next_action
    ]
    actionable = sorted(actionable, key=lambda f: (-f.priority, f.depth, f.frame_id))
    return [
        {
            "frame_id": frame.frame_id,
            "parent_frame_id": frame.parent_frame_id,
            "status": frame.status.value,
            "frame_type": frame.frame_type.value,
            "problem_id": frame.problem_id,
            "next_action": frame.next_action,
            "command_hint": frame.command_hint,
            "priority": frame.priority,
            "return_update": frame.return_update,
        }
        for frame in actionable
    ]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--goal", default="Solve the problem by recursive assumption testing.")
    ap.add_argument("--problem-id", default=None)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--evolution-payload", default=None)
    ap.add_argument("--acceptance-payload", default=None)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--max-children", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--writeback", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_recursive_assumption_run(
        graph_dir=_resolve(root, args.graph_dir),
        problem=args.problem,
        goal=args.goal,
        eval_id=args.eval_id,
        problem_id=args.problem_id,
        evolution_payload=(
            _load_json(_resolve(root, args.evolution_payload))
            if args.evolution_payload
            else None
        ),
        acceptance_payload=(
            _load_json(_resolve(root, args.acceptance_payload))
            if args.acceptance_payload
            else None
        ),
        top_k=args.top_k,
        max_children=args.max_children,
        max_depth=args.max_depth,
        writeback=args.writeback,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
