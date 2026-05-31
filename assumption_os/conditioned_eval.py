"""Relevance-conditioned evaluation for Assumption Graph nodes.

Self-evolution gates are noisy if they pool every problem together.  This module
first routes each judged problem into the subset where a node should fire, should
not fire, or is neutral, then computes benefit and harm only on the relevant
subsets.  The output is a gate decision for lifecycle updates; it does not edit
the graph by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .activation import build_activation_profile, keyword_hit_count
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph, cosine_counter, tokenize
from .record_phase2_eval import PRIMARY_TYPES, retrieve_eval_subgraph
from .schema import AssumptionNode


class RouteLabel(str, Enum):
    SHOULD_FIRE = "should_fire"
    NO_FIRE = "no_fire"
    NEUTRAL = "neutral"


class GateDecision(str, Enum):
    PROMOTE = "promote"
    KEEP = "keep"
    EXPAND_RETRIEVAL = "expand_retrieval"
    NARROW_SCOPE = "narrow_scope"
    REVISE = "revise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ConditionedEvalRow:
    problem_id: str
    domain: str
    difficulty: str
    description: str
    coverage_tags: list[str]
    outcome: str
    active_assumption_ids: list[str]
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GateThresholds:
    min_benefit_n: int = 3
    min_harm_n: int = 3
    benefit_lcb90: float = 0.54
    harm_ucb90: float = 0.35
    min_coverage: float = 0.5
    lexical_should_fire: float = 0.16


@dataclass
class ConditionedNodeSummary:
    node_id: str
    claim: str
    decision: GateDecision
    route_counts: dict[str, int]
    active_counts: dict[str, int]
    active_should_fire_outcomes: dict[str, int]
    active_no_fire_outcomes: dict[str, int]
    should_fire_coverage: float | None
    utility_when_active_should_fire: float | None
    utility_lcb90: float | None
    harm_rate_when_active_no_fire: float | None
    harm_ucb90: float | None
    reasons: list[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def evaluate_node(
    node: AssumptionNode,
    rows: Iterable[ConditionedEvalRow],
    *,
    thresholds: GateThresholds | None = None,
) -> ConditionedNodeSummary:
    thresholds = thresholds or GateThresholds()
    rows = list(rows)
    route_counts: Counter[str] = Counter()
    active_counts: Counter[str] = Counter()
    active_should: Counter[str] = Counter()
    active_no: Counter[str] = Counter()

    for row in rows:
        route = route_problem_to_node(node, row, thresholds=thresholds)
        route_counts[route.value] += 1
        active = node.id in row.active_assumption_ids
        if active:
            active_counts[route.value] += 1
            if route == RouteLabel.SHOULD_FIRE:
                active_should[row.outcome] += 1
            elif route == RouteLabel.NO_FIRE:
                active_no[row.outcome] += 1

    should_n = route_counts[RouteLabel.SHOULD_FIRE.value]
    active_should_n = sum(active_should.values())
    active_no_n = sum(active_no.values())
    coverage = active_should_n / should_n if should_n else None
    utility = _utility(active_should) if active_should_n else None
    utility_lcb = _normal_bound(utility, active_should_n, sign=-1) if utility is not None else None
    harm_rate = (active_no.get("loss", 0) / active_no_n) if active_no_n else None
    harm_ucb = _normal_bound(harm_rate, active_no_n, sign=1) if harm_rate is not None else None
    decision, reasons = _decide(
        thresholds=thresholds,
        should_n=should_n,
        active_should_n=active_should_n,
        active_no_n=active_no_n,
        coverage=coverage,
        utility_lcb=utility_lcb,
        harm_ucb=harm_ucb,
    )
    return ConditionedNodeSummary(
        node_id=node.id,
        claim=node.claim,
        decision=decision,
        route_counts=dict(route_counts),
        active_counts=dict(active_counts),
        active_should_fire_outcomes=dict(active_should),
        active_no_fire_outcomes=dict(active_no),
        should_fire_coverage=coverage,
        utility_when_active_should_fire=utility,
        utility_lcb90=utility_lcb,
        harm_rate_when_active_no_fire=harm_rate,
        harm_ucb90=harm_ucb,
        reasons=reasons,
    )


def route_problem_to_node(
    node: AssumptionNode,
    row: ConditionedEvalRow,
    *,
    thresholds: GateThresholds | None = None,
) -> RouteLabel:
    thresholds = thresholds or GateThresholds()
    profile = build_activation_profile(node)
    if row.problem_id in profile.problem_ids:
        return RouteLabel.SHOULD_FIRE
    if row.domain in profile.excluded_domains:
        return RouteLabel.NO_FIRE
    if profile.family not in {"wisdom"} and profile.domains and row.domain not in profile.domains:
        return RouteLabel.NO_FIRE
    if profile.difficulties and row.difficulty not in profile.difficulties:
        return RouteLabel.NO_FIRE
    if profile.keywords:
        hits = keyword_hit_count(profile, _row_text(row))
        domain_ok = not profile.domains or row.domain in profile.domains
        if hits >= profile.min_keyword_hits and (domain_ok or hits >= profile.min_keyword_hits + 1):
            return RouteLabel.SHOULD_FIRE
        if profile.family == "wisdom":
            return RouteLabel.NEUTRAL

    gold_ids = {f"strategy_{sid}" for sid in row.coverage_tags}
    if node.id in gold_ids:
        return RouteLabel.SHOULD_FIRE
    if profile.coverage_tags & set(row.coverage_tags):
        return RouteLabel.SHOULD_FIRE
    tag_set = {str(tag) for tag in node.tags}
    if tag_set & set(row.coverage_tags):
        return RouteLabel.SHOULD_FIRE
    if profile.family == "strategy":
        return RouteLabel.NEUTRAL

    if profile.domains:
        return RouteLabel.SHOULD_FIRE if row.domain in profile.domains else RouteLabel.NO_FIRE
    if not profile.allow_lexical_fallback:
        return RouteLabel.NEUTRAL

    lexical = cosine_counter(tokenize(_node_route_text(node)), tokenize(_row_text(row)))
    if lexical >= thresholds.lexical_should_fire:
        return RouteLabel.SHOULD_FIRE
    return RouteLabel.NEUTRAL


def build_conditioned_rows(
    *,
    graph: SimpleAssumptionGraph,
    sample: list[dict],
    meta_by_pid: dict,
    judgment_paths: Iterable[Path],
    intervention_variant: str,
    baseline_variant: str,
    top_k: int = 8,
    policy_rerank: bool = False,
    skip_domains: set[str] | None = None,
    skip_missing_meta: bool = True,
) -> list[ConditionedEvalRow]:
    problems = {p["problem_id"]: p for p in sample if "problem_id" in p}
    rows: list[ConditionedEvalRow] = []
    for judgment_path in judgment_paths:
        judgments = _load_json(judgment_path)
        for pid, judgment in judgments.items():
            problem = problems.get(pid)
            if not problem:
                continue
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
            active_ids = [node.id for node in subgraph.nodes] if subgraph else []
            rows.append(ConditionedEvalRow(
                problem_id=pid,
                domain=problem.get("domain", ""),
                difficulty=problem.get("difficulty", ""),
                description=problem.get("description", ""),
                coverage_tags=problem.get("coverage_tags", []),
                outcome=_normalize_outcome(judgment.get("winner"), intervention_variant, baseline_variant),
                active_assumption_ids=active_ids,
                meta=meta,
            ))
    return rows


def evaluate_graph_nodes(
    graph: SimpleAssumptionGraph,
    rows: Iterable[ConditionedEvalRow],
    *,
    thresholds: GateThresholds | None = None,
    node_ids: Iterable[str] | None = None,
) -> list[ConditionedNodeSummary]:
    rows = list(rows)
    ids = sorted(set(node_ids or _active_node_ids(rows)))
    summaries = [
        evaluate_node(graph.store.nodes[nid], rows, thresholds=thresholds)
        for nid in ids
        if nid in graph.store.nodes
    ]
    return sorted(summaries, key=_summary_sort_key)


def _decide(
    *,
    thresholds: GateThresholds,
    should_n: int,
    active_should_n: int,
    active_no_n: int,
    coverage: float | None,
    utility_lcb: float | None,
    harm_ucb: float | None,
) -> tuple[GateDecision, list[str]]:
    reasons = []
    if active_should_n < thresholds.min_benefit_n:
        if should_n >= thresholds.min_benefit_n:
            reasons.append("node should fire but retrieval did not expose it enough for benefit evidence")
            return GateDecision.EXPAND_RETRIEVAL, reasons
        reasons.append("not enough should-fire evidence")
        return GateDecision.INSUFFICIENT_EVIDENCE, reasons

    benefit_pass = utility_lcb is not None and utility_lcb >= thresholds.benefit_lcb90
    harm_pass = active_no_n < thresholds.min_harm_n or (harm_ucb is not None and harm_ucb <= thresholds.harm_ucb90)
    coverage_pass = coverage is not None and coverage >= thresholds.min_coverage

    if not harm_pass:
        reasons.append("active no-fire harm exceeds threshold")
        return GateDecision.NARROW_SCOPE, reasons
    if not benefit_pass:
        reasons.append("conditioned benefit lower confidence bound is below threshold")
        return GateDecision.REVISE, reasons
    if not coverage_pass:
        reasons.append("beneficial when active but under-retrieved on should-fire subset")
        return GateDecision.EXPAND_RETRIEVAL, reasons
    if active_should_n >= 2 * thresholds.min_benefit_n:
        reasons.append("benefit passes conditioned gate with adequate exposure")
        return GateDecision.PROMOTE, reasons
    reasons.append("benefit passes but exposure is still small")
    return GateDecision.KEEP, reasons


def _normalize_outcome(winner: str | None, intervention_variant: str, baseline_variant: str) -> str:
    if winner == intervention_variant:
        return "win"
    if winner == baseline_variant:
        return "loss"
    return "tie"


def _utility(outcomes: Counter[str]) -> float:
    n = sum(outcomes.values())
    return (outcomes.get("win", 0) + 0.5 * outcomes.get("tie", 0)) / n if n else 0.0


def _normal_bound(value: float, n: int, *, sign: int) -> float:
    if n <= 0:
        return value
    se = math.sqrt(max(value * (1.0 - value), 0.0) / n)
    return max(0.0, min(1.0, value + sign * 1.28 * se))


def _node_route_text(node: AssumptionNode) -> str:
    return "\n".join([
        node.claim,
        " ".join(node.context_conditions),
        " ".join(node.predicted_effects),
        " ".join(node.tags),
    ])


def _row_text(row: ConditionedEvalRow) -> str:
    return "\n".join([
        row.description,
        row.domain,
        row.difficulty,
        " ".join(row.coverage_tags),
        row.meta.get("critical_reframe", ""),
        row.meta.get("rewritten_problem", ""),
        row.meta.get("what_changed", ""),
    ])


def _active_node_ids(rows: Iterable[ConditionedEvalRow]) -> set[str]:
    return {nid for row in rows for nid in row.active_assumption_ids}


def _summary_sort_key(summary: ConditionedNodeSummary):
    decision_rank = {
        GateDecision.PROMOTE: 0,
        GateDecision.KEEP: 1,
        GateDecision.EXPAND_RETRIEVAL: 2,
        GateDecision.NARROW_SCOPE: 3,
        GateDecision.REVISE: 4,
        GateDecision.INSUFFICIENT_EVIDENCE: 5,
    }
    return (
        decision_rank[summary.decision],
        -(summary.utility_lcb90 or 0.0),
        -(summary.route_counts.get(RouteLabel.SHOULD_FIRE.value, 0)),
        summary.node_id,
    )


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--policy-rerank", action="store_true")
    ap.add_argument("--assumption-graph-skip-domains", default="")
    ap.add_argument("--include-missing-meta", action="store_true")
    ap.add_argument("--node-ids", nargs="*", default=None)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--min-benefit-n", type=int, default=3)
    ap.add_argument("--min-harm-n", type=int, default=3)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    graph = SimpleAssumptionGraph(JsonlGraphStore(_resolve(root, args.graph_dir)))
    rows = build_conditioned_rows(
        graph=graph,
        sample=_load_json(_resolve(root, args.sample)),
        meta_by_pid=_load_json(_resolve(root, args.meta)),
        judgment_paths=[_resolve(root, p) for p in args.judgments],
        intervention_variant=args.intervention,
        baseline_variant=args.baseline,
        top_k=args.top_k,
        policy_rerank=args.policy_rerank,
        skip_domains={x.strip() for x in args.assumption_graph_skip_domains.split(",") if x.strip()},
        skip_missing_meta=not args.include_missing_meta,
    )
    thresholds = GateThresholds(min_benefit_n=args.min_benefit_n, min_harm_n=args.min_harm_n)
    summaries = evaluate_graph_nodes(graph, rows, thresholds=thresholds, node_ids=args.node_ids)
    payload = {
        "rows": len(rows),
        "thresholds": asdict(thresholds),
        "decision_counts": dict(Counter(s.decision.value for s in summaries)),
        "summaries": [s.to_dict() for s in summaries[: args.top_n]],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
