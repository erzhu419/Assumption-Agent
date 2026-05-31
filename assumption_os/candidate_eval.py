"""Preflight evaluation for candidate Assumption Graph proposals.

This is not the final quality judge.  It answers the cheaper question before
spending model calls: if a candidate proposal is overlaid on the graph, does it
route to a meaningful trigger subset, get retrieved on that subset, and avoid
retrieval on explicit no-fire rows?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .conditioned_eval import ConditionedEvalRow, GateThresholds, RouteLabel, route_problem_to_node
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .proposal_overlay import (
    apply_proposal_overlay,
    iter_matching_proposals,
    load_proposal_payload,
    parse_csv_set,
)
from .proposals import ProposalType
from .record_phase2_eval import retrieve_eval_subgraph
from .schema import AssumptionNode


class CandidateReadiness(str, Enum):
    READY_FOR_FRESH_ABLATION = "ready_for_fresh_ablation"
    NEEDS_RETRIEVAL_FIX = "needs_retrieval_fix"
    NEEDS_SCOPE_FIX = "needs_scope_fix"
    NEEDS_MORE_TRIGGER_ROWS = "needs_more_trigger_rows"
    MANIFEST_ONLY = "manifest_only"
    MISSING_PARENT = "missing_parent"


@dataclass(frozen=True)
class CandidateEvalSummary:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    target_node_id: str | None
    route_node_id: str | None
    readiness: CandidateReadiness
    route_counts: dict[str, int]
    active_counts: dict[str, int]
    trigger_problem_ids: list[str] = field(default_factory=list)
    active_trigger_problem_ids: list[str] = field(default_factory=list)
    missed_trigger_problem_ids: list[str] = field(default_factory=list)
    outside_active_problem_ids: list[str] = field(default_factory=list)
    control_problem_ids: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    command_hint: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["readiness"] = self.readiness.value
        return d


def build_candidate_eval_payload(
    *,
    graph_dir: Path,
    proposal_payload: dict,
    sample: list[dict],
    meta_by_pid: dict,
    eval_id: str,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
    top_k: int = 8,
    policy_rerank: bool = False,
    skip_domains: set[str] | None = None,
    skip_missing_meta: bool = True,
    min_trigger_n: int = 3,
    min_active_trigger_n: int = 3,
    force_proposal_route: bool = False,
    max_control_ids: int = 8,
    command_prefix: str = 'python3 "phase one/scripts/validation/phase2_v20_framework.py"',
    graph_arg: str = "phase four/assumption_graph",
    proposals_arg: str = "phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json",
    sample_arg: str = "sample_21_50.json",
) -> dict:
    proposals = list(iter_matching_proposals(
        proposal_payload,
        proposal_ids=proposal_ids,
        parent_node_ids=parent_node_ids,
        proposal_types=proposal_types,
    ))
    summaries = [
        _evaluate_proposal(
            graph_dir=graph_dir,
            proposal_payload=proposal_payload,
            proposal=proposal,
            sample=sample,
            meta_by_pid=meta_by_pid,
            top_k=top_k,
            policy_rerank=policy_rerank,
            skip_domains=skip_domains,
            skip_missing_meta=skip_missing_meta,
            min_trigger_n=min_trigger_n,
            min_active_trigger_n=min_active_trigger_n,
            force_proposal_route=force_proposal_route,
            max_control_ids=max_control_ids,
            command_prefix=command_prefix,
            graph_arg=graph_arg,
            proposals_arg=proposals_arg,
            sample_arg=sample_arg,
        )
        for proposal in proposals
    ]
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "thresholds": {
            "top_k": top_k,
            "policy_rerank": policy_rerank,
            "skip_domains": sorted(skip_domains or []),
            "skip_missing_meta": skip_missing_meta,
            "min_trigger_n": min_trigger_n,
            "min_active_trigger_n": min_active_trigger_n,
            "force_proposal_route": force_proposal_route,
        },
        "readiness_counts": dict(Counter(s.readiness.value for s in summaries)),
        "summaries": [s.to_dict() for s in summaries],
    }


def _evaluate_proposal(
    *,
    graph_dir: Path,
    proposal_payload: dict,
    proposal: dict,
    sample: list[dict],
    meta_by_pid: dict,
    top_k: int,
    policy_rerank: bool,
    skip_domains: set[str] | None,
    skip_missing_meta: bool,
    min_trigger_n: int,
    min_active_trigger_n: int,
    force_proposal_route: bool,
    max_control_ids: int,
    command_prefix: str,
    graph_arg: str,
    proposals_arg: str,
    sample_arg: str,
) -> CandidateEvalSummary:
    store = JsonlGraphStore(graph_dir)
    parent = store.nodes.get(proposal.get("parent_node_id", ""))
    if not parent:
        return CandidateEvalSummary(
            proposal_id=proposal.get("proposal_id", ""),
            proposal_type=proposal.get("proposal_type", ""),
            parent_node_id=proposal.get("parent_node_id", ""),
            candidate_node_id=None,
            target_node_id=None,
            route_node_id=None,
            readiness=CandidateReadiness.MISSING_PARENT,
            route_counts={},
            active_counts={},
            rationale="Parent node is missing from the graph store.",
        )

    candidate = (
        AssumptionNode.from_dict(proposal["candidate_node"])
        if proposal.get("candidate_node")
        else None
    )
    proposal_type = proposal.get("proposal_type", "")
    if not candidate:
        return CandidateEvalSummary(
            proposal_id=proposal["proposal_id"],
            proposal_type=proposal_type,
            parent_node_id=parent.id,
            candidate_node_id=None,
            target_node_id=parent.id,
            route_node_id=parent.id,
            readiness=CandidateReadiness.MANIFEST_ONLY,
            route_counts={},
            active_counts={},
            acceptance_criteria=_acceptance_criteria(proposal_type),
            rationale="This proposal only requests more evidence or records promotion; no candidate node can be preflighted.",
        )

    apply_proposal_overlay(
        store,
        proposal_payload,
        proposal_ids=[proposal["proposal_id"]],
        include_manifests=False,
    )
    graph = SimpleAssumptionGraph(store)
    rows = _build_probe_rows(
        graph=graph,
        sample=sample,
        meta_by_pid=meta_by_pid,
        top_k=top_k,
        policy_rerank=policy_rerank,
        skip_domains=skip_domains,
        skip_missing_meta=skip_missing_meta,
    )

    if proposal_type == ProposalType.RETRIEVAL_POLICY.value:
        route_node = parent
        target_node_id = parent.id
    else:
        route_node = candidate
        target_node_id = candidate.id

    routes: Counter[str] = Counter()
    active_counts: Counter[str] = Counter()
    trigger_ids: list[str] = []
    active_trigger_ids: list[str] = []
    missed_trigger_ids: list[str] = []
    outside_active_ids: list[str] = []
    control_ids: list[str] = []
    for row in rows:
        route = route_problem_to_node(route_node, row, thresholds=GateThresholds())
        routes[route.value] += 1
        active = target_node_id in row.active_assumption_ids or (
            force_proposal_route and route == RouteLabel.SHOULD_FIRE
        )
        if active:
            active_counts[route.value] += 1
        if route == RouteLabel.SHOULD_FIRE:
            trigger_ids.append(row.problem_id)
            if active:
                active_trigger_ids.append(row.problem_id)
            else:
                missed_trigger_ids.append(row.problem_id)
        elif route == RouteLabel.NO_FIRE and active:
            outside_active_ids.append(row.problem_id)
        elif route == RouteLabel.NEUTRAL and len(control_ids) < max_control_ids:
            control_ids.append(row.problem_id)

    readiness = _readiness(
        should_n=len(trigger_ids),
        active_should_n=len(active_trigger_ids),
        outside_active_n=len(outside_active_ids),
        min_trigger_n=min_trigger_n,
        min_active_trigger_n=min_active_trigger_n,
    )
    return CandidateEvalSummary(
        proposal_id=proposal["proposal_id"],
        proposal_type=proposal_type,
        parent_node_id=parent.id,
        candidate_node_id=candidate.id,
        target_node_id=target_node_id,
        route_node_id=route_node.id,
        readiness=readiness,
        route_counts=dict(routes),
        active_counts=dict(active_counts),
        trigger_problem_ids=trigger_ids,
        active_trigger_problem_ids=active_trigger_ids,
        missed_trigger_problem_ids=missed_trigger_ids,
        outside_active_problem_ids=outside_active_ids,
        control_problem_ids=control_ids,
        acceptance_criteria=_acceptance_criteria(proposal_type),
        command_hint=_command_hint(
            command_prefix=command_prefix,
            proposal=proposal,
            graph_arg=graph_arg,
            proposals_arg=proposals_arg,
            sample_arg=sample_arg,
            force_proposal_route=force_proposal_route,
        ),
        rationale=_rationale(readiness),
    )


def _build_probe_rows(
    *,
    graph: SimpleAssumptionGraph,
    sample: list[dict],
    meta_by_pid: dict,
    top_k: int,
    policy_rerank: bool,
    skip_domains: set[str] | None,
    skip_missing_meta: bool,
) -> list[ConditionedEvalRow]:
    rows = []
    for problem in sample:
        pid = problem.get("problem_id", "")
        meta = meta_by_pid.get(pid, {})
        if skip_missing_meta and not meta:
            continue
        subgraph = retrieve_eval_subgraph(
            graph,
            problem,
            meta,
            top_k=top_k,
            policy_rerank=policy_rerank,
            skip_domains=skip_domains,
        )
        if subgraph is None:
            continue
        rows.append(ConditionedEvalRow(
            problem_id=pid,
            domain=problem.get("domain", ""),
            difficulty=problem.get("difficulty", ""),
            description=problem.get("description", ""),
            coverage_tags=problem.get("coverage_tags", []),
            outcome="tie",
            active_assumption_ids=[node.id for node in subgraph.nodes],
            meta=meta,
        ))
    return rows


def _readiness(
    *,
    should_n: int,
    active_should_n: int,
    outside_active_n: int,
    min_trigger_n: int,
    min_active_trigger_n: int,
) -> CandidateReadiness:
    if should_n < min_trigger_n:
        return CandidateReadiness.NEEDS_MORE_TRIGGER_ROWS
    if outside_active_n:
        return CandidateReadiness.NEEDS_SCOPE_FIX
    if active_should_n < min_active_trigger_n:
        return CandidateReadiness.NEEDS_RETRIEVAL_FIX
    return CandidateReadiness.READY_FOR_FRESH_ABLATION


def _acceptance_criteria(proposal_type: str) -> list[str]:
    common = [
        "candidate beats the same-model parent/baseline on routed trigger rows",
        "candidate does not increase active no-fire retrieval or judged losses on outside controls",
        "fresh conditioned gate passes with heldout rows not used to create the proposal",
    ]
    if proposal_type == ProposalType.RETRIEVAL_POLICY.value:
        return [
            "active trigger coverage increases before quality judging",
            *common,
        ]
    if proposal_type in {ProposalType.ASSUMPTION_REVISION.value, ProposalType.SCOPE_NARROWING.value}:
        return [
            "candidate child beats parent on the same routed trigger subset",
            *common,
        ]
    return common


def _command_hint(
    *,
    command_prefix: str,
    proposal: dict,
    graph_arg: str,
    proposals_arg: str,
    sample_arg: str,
    force_proposal_route: bool,
) -> str:
    variant = f"proposal_{proposal['proposal_id'].replace('prop_', '')}"
    force_arg = " --assumption-force-proposal-route" if force_proposal_route else ""
    return (
        f"{command_prefix} --variant {variant} "
        f'--sample "{sample_arg}" '
        f'--assumption-graph "{graph_arg}" '
        f'--assumption-graph-skip-domains "" '
        f'--assumption-proposals "{proposals_arg}" '
        f"--assumption-proposal-ids {proposal['proposal_id']}"
        f"{force_arg}"
    )


def _rationale(readiness: CandidateReadiness) -> str:
    if readiness == CandidateReadiness.READY_FOR_FRESH_ABLATION:
        return "Overlay retrieval reaches enough routed trigger rows and no explicit no-fire active row was observed."
    if readiness == CandidateReadiness.NEEDS_RETRIEVAL_FIX:
        return "The candidate has a trigger subset, but overlay retrieval does not expose it enough for a fair quality test."
    if readiness == CandidateReadiness.NEEDS_SCOPE_FIX:
        return "The candidate is retrieved on at least one explicit no-fire row and needs tighter activation scope before judging."
    if readiness == CandidateReadiness.NEEDS_MORE_TRIGGER_ROWS:
        return "The available sample does not contain enough routed trigger rows for a meaningful test."
    return "No candidate node was available for preflight."


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--proposal-ids", nargs="*", default=None)
    ap.add_argument("--proposal-types", default="")
    ap.add_argument("--parent-node-ids", nargs="*", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--policy-rerank", action="store_true")
    ap.add_argument("--assumption-graph-skip-domains", default="")
    ap.add_argument("--include-missing-meta", action="store_true")
    ap.add_argument("--min-trigger-n", type=int, default=3)
    ap.add_argument("--min-active-trigger-n", type=int, default=3)
    ap.add_argument("--force-proposal-route", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    proposals_path = _resolve(root, args.proposals)
    sample_path = _resolve(root, args.sample)
    payload = build_candidate_eval_payload(
        graph_dir=_resolve(root, args.graph_dir),
        proposal_payload=load_proposal_payload(proposals_path),
        sample=_load_json(sample_path),
        meta_by_pid=_load_json(_resolve(root, args.meta)),
        eval_id=args.eval_id,
        proposal_ids=args.proposal_ids,
        parent_node_ids=args.parent_node_ids,
        proposal_types=parse_csv_set(args.proposal_types),
        top_k=args.top_k,
        policy_rerank=args.policy_rerank,
        skip_domains=parse_csv_set(args.assumption_graph_skip_domains),
        skip_missing_meta=not args.include_missing_meta,
        min_trigger_n=args.min_trigger_n,
        min_active_trigger_n=args.min_active_trigger_n,
        force_proposal_route=args.force_proposal_route,
        proposals_arg=args.proposals,
        sample_arg=str(sample_path),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
