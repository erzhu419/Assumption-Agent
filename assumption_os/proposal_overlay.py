"""Temporary proposal overlays for candidate testing.

Candidate proposals should be tested before they are committed to the graph.
This module applies proposal nodes, edges, and optionally manifests to an
in-memory ``JsonlGraphStore`` without flushing the store to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .conditioned_eval import ConditionedEvalRow, RouteLabel, route_problem_to_node
from .graph_memory import JsonlGraphStore
from .proposals import ProposalType
from .schema import AssumptionEdge, AssumptionNode, TrialManifest


def load_proposal_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_csv_set(raw: str | None) -> set[str]:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


def iter_matching_proposals(
    proposal_payload: dict,
    *,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
):
    ids = set(proposal_ids or [])
    parents = set(parent_node_ids or [])
    types = set(proposal_types or [])
    for proposal in proposal_payload.get("proposals", []):
        if ids and proposal.get("proposal_id") not in ids:
            continue
        if parents and proposal.get("parent_node_id") not in parents:
            continue
        if types and proposal.get("proposal_type") not in types:
            continue
        yield proposal


def apply_proposal_overlay(
    store: JsonlGraphStore,
    proposal_payload: dict,
    *,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
    include_manifests: bool = False,
) -> list[str]:
    """Apply matching proposal objects to ``store`` without flushing to disk."""

    applied_candidate_ids: list[str] = []
    for proposal in iter_matching_proposals(
        proposal_payload,
        proposal_ids=proposal_ids,
        parent_node_ids=parent_node_ids,
        proposal_types=proposal_types,
    ):
        candidate = proposal.get("candidate_node")
        if candidate:
            node = AssumptionNode.from_dict(candidate)
            store.upsert_node(node)
            applied_candidate_ids.append(node.id)
        for edge in proposal.get("edges", []):
            store.add_edge(AssumptionEdge.from_dict(edge))
        if include_manifests and proposal.get("manifest"):
            store.append_trial(TrialManifest.from_dict(proposal["manifest"]))
    return applied_candidate_ids


def apply_proposal_overlay_file(
    store: JsonlGraphStore,
    proposal_path: str | Path,
    *,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
    include_manifests: bool = False,
) -> list[str]:
    return apply_proposal_overlay(
        store,
        load_proposal_payload(proposal_path),
        proposal_ids=proposal_ids,
        parent_node_ids=parent_node_ids,
        proposal_types=proposal_types,
        include_manifests=include_manifests,
    )


def proposal_route_target_ids(
    store: JsonlGraphStore,
    proposal_payload: dict,
    *,
    problem: dict,
    meta: dict,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
) -> list[str]:
    """Return candidate/parent ids that should be forced for this problem.

    Retrieval-policy proposals force their parent node on the parent's routed
    trigger subset.  Revision/scope proposals force the candidate child on the
    candidate child's routed trigger subset.  Neutral and no-fire rows are left
    untouched.
    """

    row = ConditionedEvalRow(
        problem_id=problem.get("problem_id", ""),
        domain=problem.get("domain", ""),
        difficulty=problem.get("difficulty", ""),
        description=problem.get("description", ""),
        coverage_tags=problem.get("coverage_tags", []),
        outcome="tie",
        active_assumption_ids=[],
        meta=meta,
    )
    target_ids: list[str] = []
    for proposal in iter_matching_proposals(
        proposal_payload,
        proposal_ids=proposal_ids,
        parent_node_ids=parent_node_ids,
        proposal_types=proposal_types,
    ):
        parent = store.nodes.get(proposal.get("parent_node_id", ""))
        if not parent:
            continue
        if proposal.get("proposal_type") == ProposalType.RETRIEVAL_POLICY.value:
            route_node = parent
            target_id = parent.id
        elif proposal.get("candidate_node"):
            route_node = AssumptionNode.from_dict(proposal["candidate_node"])
            target_id = route_node.id
        else:
            continue
        if target_id in store.nodes and route_problem_to_node(route_node, row) == RouteLabel.SHOULD_FIRE:
            target_ids.append(target_id)
    return target_ids
