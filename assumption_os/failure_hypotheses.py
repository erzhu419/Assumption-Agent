"""Generate candidate hypotheses from judged failure rows.

This is the deterministic first pass of the self-evolution loop: every
attributed loss is converted into a reviewable candidate node and manifest
before any graph mutation can happen.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .proposals import CandidateProposal, ProposalType
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    TrialManifest,
    TrialStatus,
    stable_id,
)


def build_failure_hypothesis_payload(
    *,
    graph,
    sample: list[dict],
    meta_by_pid: dict,
    writeback_summary: dict,
    eval_id: str,
    max_hypotheses: int | None = None,
    judgment_paths: Iterable[Path] | None = None,
    intervention_variant: str | None = None,
    baseline_variant: str | None = None,
    skip_domains: set[str] | None = None,
    skip_missing_meta: bool = True,
    include_skipped_losses: bool = True,
) -> dict:
    problems = {p.get("problem_id"): p for p in sample if p.get("problem_id")}
    processed_loss_groups = _loss_groups(writeback_summary.get("processed_trials", []))
    skipped_rows, skipped_scan = _skipped_outcome_rows(
        sample=sample,
        meta_by_pid=meta_by_pid,
        processed_trials=writeback_summary.get("processed_trials", []),
        judgment_paths=list(judgment_paths or []),
        intervention_variant=intervention_variant,
        baseline_variant=baseline_variant,
        skip_domains=skip_domains or set(),
        skip_missing_meta=skip_missing_meta,
        include_skipped_losses=include_skipped_losses,
        source_eval_id=writeback_summary.get("eval_id") or eval_id,
    )
    skipped_loss_groups = _loss_groups(skipped_rows)
    loss_groups = [*processed_loss_groups, *skipped_loss_groups]
    proposals: list[CandidateProposal] = []
    skipped: Counter[str] = Counter()
    for group in loss_groups:
        parent = _select_parent(graph, group)
        if not parent:
            skipped["missing_parent"] += 1
            continue
        proposal = _proposal_from_group(
            parent=parent,
            group=group,
            problems=problems,
            meta_by_pid=meta_by_pid,
            eval_id=eval_id,
        )
        proposals.append(proposal)
    proposals = sorted(proposals, key=lambda p: (-p.priority, p.parent_node_id, p.proposal_id))
    if max_hypotheses is not None:
        proposals = proposals[:max_hypotheses]
    return {
        "eval_id": eval_id,
        "source_eval_id": writeback_summary.get("eval_id"),
        "loss_problem_count": len(loss_groups),
        "processed_loss_problem_count": len(processed_loss_groups),
        "skipped_loss_problem_count": len(skipped_loss_groups),
        "skipped_loss_scan": skipped_scan,
        "skipped": dict(skipped),
        "proposal_counts": dict(Counter(p.proposal_type.value for p in proposals)),
        "proposals": [p.to_dict() for p in proposals],
    }


def merge_proposal_payloads(*, eval_id: str, payloads: list[dict]) -> dict:
    proposals = [proposal for payload in payloads for proposal in payload.get("proposals", [])]
    return {
        "eval_id": eval_id,
        "source_eval_ids": [payload.get("eval_id") for payload in payloads if payload.get("eval_id")],
        "proposal_counts": dict(Counter(p.get("proposal_type", "") for p in proposals)),
        "proposals": proposals,
    }


def _loss_groups(processed_trials: list[dict]) -> list[dict]:
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for row in processed_trials:
        by_pid[row.get("problem_id", "")].append(row)

    groups = []
    for pid, rows in by_pid.items():
        outcomes = Counter(row.get("outcome") for row in rows)
        if outcomes.get("loss", 0) <= outcomes.get("win", 0):
            continue
        first_loss = next(row for row in rows if row.get("outcome") == "loss")
        groups.append({
            "problem_id": pid,
            "rows": rows,
            "outcomes": dict(outcomes),
            "domain": first_loss.get("domain", ""),
            "difficulty": first_loss.get("difficulty", ""),
            "residual_type": first_loss.get("residual_type", ""),
            "gold_hit": bool(first_loss.get("gold_hit")),
            "gold_ids": first_loss.get("gold_ids", []),
            "active_assumption_ids": first_loss.get("active_assumption_ids", []),
            "trial_ids": [row.get("trial_id") for row in rows if row.get("trial_id")],
            "source_kind": first_loss.get("source_kind", "processed_trial"),
            "source_skipped_reason": first_loss.get("source_skipped_reason"),
        })
    return groups


def _skipped_outcome_rows(
    *,
    sample: list[dict],
    meta_by_pid: dict,
    processed_trials: list[dict],
    judgment_paths: list[Path],
    intervention_variant: str | None,
    baseline_variant: str | None,
    skip_domains: set[str],
    skip_missing_meta: bool,
    include_skipped_losses: bool,
    source_eval_id: str,
) -> tuple[list[dict], dict]:
    """Recover judged rows that writeback skipped before creating manifests."""

    scan = {
        "enabled": bool(include_skipped_losses),
        "candidate_rows": 0,
        "outcome_counts": {},
        "reason_counts": {},
        "loss_problem_ids": [],
        "net_loss_problem_ids": [],
    }
    if (
        not include_skipped_losses
        or not judgment_paths
        or not intervention_variant
        or not baseline_variant
    ):
        return [], scan

    problems = {p.get("problem_id"): p for p in sample if p.get("problem_id")}
    processed_keys = {
        (row.get("problem_id"), _path_key(row.get("judgment_path", "")))
        for row in processed_trials
    }
    rows = []
    reason_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    loss_problem_ids = set()
    for judgment_path in judgment_paths:
        judgments = _load_json(Path(judgment_path))
        path_key = _path_key(str(judgment_path))
        for pid, judgment in judgments.items():
            problem = problems.get(pid)
            if not problem:
                continue
            if (pid, path_key) in processed_keys:
                continue
            reason = _skip_reason(problem, meta_by_pid.get(pid), skip_domains, skip_missing_meta)
            if not reason:
                continue
            outcome = _outcome(judgment, intervention_variant, baseline_variant)
            reason_counts[reason] += 1
            outcome_counts[outcome] += 1
            if outcome == "loss":
                loss_problem_ids.add(pid)
            rows.append({
                "trial_id": stable_id(
                    "skipped",
                    source_eval_id,
                    judgment_path.name,
                    pid,
                    intervention_variant,
                    baseline_variant,
                    reason,
                ),
                "problem_id": pid,
                "domain": problem.get("domain", ""),
                "difficulty": problem.get("difficulty", ""),
                "outcome": outcome,
                "winner": judgment.get("winner", "tie"),
                "residual_type": _skipped_residual_type(reason),
                "gold_hit": False,
                "gold_ids": _gold_ids(problem),
                "active_assumption_ids": [],
                "judgment_path": str(judgment_path),
                "source_kind": "skipped_judgment",
                "source_skipped_reason": reason,
            })

    net_loss_problem_ids = []
    outcomes_by_pid: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        outcomes_by_pid[row["problem_id"]][row["outcome"]] += 1
    for pid, outcomes in outcomes_by_pid.items():
        if outcomes.get("loss", 0) > outcomes.get("win", 0):
            net_loss_problem_ids.append(pid)

    return rows, {
        "enabled": True,
        "candidate_rows": len(rows),
        "outcome_counts": dict(outcome_counts),
        "reason_counts": dict(reason_counts),
        "loss_problem_ids": sorted(loss_problem_ids),
        "net_loss_problem_ids": sorted(net_loss_problem_ids),
    }


def _select_parent(graph, group: dict) -> AssumptionNode | None:
    for node_id in [*group.get("gold_ids", []), *group.get("active_assumption_ids", [])]:
        node = graph.store.nodes.get(node_id)
        if node:
            return node
    return None


def _proposal_from_group(
    *,
    parent: AssumptionNode,
    group: dict,
    problems: dict[str, dict],
    meta_by_pid: dict,
    eval_id: str,
) -> CandidateProposal:
    pid = group["problem_id"]
    problem = problems.get(pid, {})
    meta = meta_by_pid.get(pid, {})
    residual_type = group.get("residual_type", "unknown")
    cid = stable_id("cand", eval_id, pid, parent.id, residual_type)
    claim = _claim(parent, group, problem, meta)
    parent_type = getattr(parent.type, "value", parent.type)
    coverage_codes = _strategy_codes(group.get("gold_ids", []))
    candidate = AssumptionNode(
        id=cid,
        type=parent.type if parent_type != AssumptionType.RETRIEVAL.value else AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=claim,
        context_conditions=_context_conditions(group, problem, meta),
        predicted_effects=[
            "turn the recorded failure residual into a testable assumption update",
            "beat the current graph route on the same failure pattern without outside-control harm",
        ],
        risk_predictions=[
            "single-failure hypothesis may overfit the current heldout slice",
            "must pass preflight trigger coverage before any fresh judge spend",
        ],
        verifiers=["candidate_preflight", "sequential_falsification", "fresh_ablation"],
        confidence=0.25,
        metaproductivity=0.0,
        status="candidate",
        tags=[
            "candidate",
            "failure_hypothesis",
            str(group.get("domain", "")),
            str(residual_type),
            pid,
            parent.id,
            *coverage_codes,
        ],
        source_refs=[*parent.source_refs, f"failure:{pid}", f"parent:{parent.id}"],
        payload={
            "parent_node_id": parent.id,
            "source_problem_id": pid,
            "source_trial_ids": group.get("trial_ids", []),
            "source_kind": group.get("source_kind", "processed_trial"),
            "source_skipped_reason": group.get("source_skipped_reason"),
            "outcomes": group.get("outcomes", {}),
            "residual_type": residual_type,
            "gold_hit": group.get("gold_hit"),
            "gold_ids": group.get("gold_ids", []),
            "active_assumption_ids": group.get("active_assumption_ids", []),
            "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
            "activation": {
                "problem_ids": [pid],
                "coverage_tags": coverage_codes,
                "keywords": _activation_keywords(problem, meta),
                "min_keyword_hits": 3,
            },
        },
    )
    edge = AssumptionEdge(
        source=parent.id,
        target=candidate.id,
        type=EdgeType.GENERATED_FROM_RESIDUAL,
        weight=0.65,
        payload={
            "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
            "eval_id": eval_id,
            "problem_id": pid,
            "residual_type": residual_type,
        },
    )
    manifest = _manifest(parent, candidate, group, eval_id)
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, pid, parent.id, ProposalType.FAILURE_HYPOTHESIS.value),
        proposal_type=ProposalType.FAILURE_HYPOTHESIS,
        parent_node_id=parent.id,
        candidate_node=candidate.to_dict(),
        edges=[edge.to_dict()],
        manifest=manifest.to_dict(),
        rationale=_rationale(group),
        priority=_priority(group),
        source_action={
            "action_type": "generate_failure_hypothesis",
            "source_problem_id": pid,
            "source_kind": group.get("source_kind", "processed_trial"),
            "source_skipped_reason": group.get("source_skipped_reason"),
            "residual_type": residual_type,
            "outcomes": group.get("outcomes", {}),
        },
    )


def _claim(parent: AssumptionNode, group: dict, problem: dict, meta: dict) -> str:
    pid = group["problem_id"]
    residual_type = group.get("residual_type", "unknown")
    skipped_reason = group.get("source_skipped_reason")
    if skipped_reason == "missing_meta":
        return (
            f"For bypassed math/science failures like {pid}, create a graph-compatible "
            f"bridge before answering: connect {parent.id} to the concrete research route, "
            "candidate counterexample/test, and next evidence step instead of leaving the "
            "hygiene bypass to answer alone."
        )
    if skipped_reason == "policy_skipped":
        return (
            f"For skipped-domain failures like {pid}, replace the blanket domain bypass with "
            f"a scoped execution bridge from {parent.id}: retrieve only when the task needs "
            "acceptance metrics, rollback/monitoring, migration risk, or implementation sequencing."
        )
    if residual_type == "memory_defect":
        return (
            f"For failures like {pid}, repair retrieval before changing the core assumption: "
            f"surface {', '.join(group.get('gold_ids', []) or [parent.id])} when the task matches "
            f"{_short(problem.get('description', ''), 160)}"
        )
    if residual_type == "optimization":
        return (
            f"For failures like {pid}, keep {parent.id} but require a concrete execution bridge "
            f"from the activated assumption to the final answer, using the frame "
            f"{_short(meta.get('frame', ''), 80)}."
        )
    return (
        f"Investigate a new assumption for {pid}: the current graph route around {parent.id} "
        "did not beat the baseline and needs a falsifiable repair candidate."
    )


def _context_conditions(group: dict, problem: dict, meta: dict) -> list[str]:
    return [
        f"source_problem={group['problem_id']}",
        f"domain={group.get('domain', '')}",
        f"difficulty={group.get('difficulty', '')}",
        f"residual_type={group.get('residual_type', '')}",
        f"source_kind={group.get('source_kind', 'processed_trial')}",
        f"source_skipped_reason={group.get('source_skipped_reason') or ''}",
        f"gold_hit={group.get('gold_hit')}",
        f"coverage={','.join(group.get('gold_ids', []))}",
        _short(problem.get("description", ""), 220),
        _short(meta.get("critical_reframe", "") or meta.get("rewritten_problem", ""), 220),
    ]


def _manifest(parent: AssumptionNode, candidate: AssumptionNode, group: dict, eval_id: str) -> TrialManifest:
    return TrialManifest(
        problem_id=f"failure::{group['problem_id']}",
        action_type=f"proposal_{ProposalType.FAILURE_HYPOTHESIS.value}",
        component="failure_hypothesis_generator",
        assumption=f"Failure-derived candidate for {group['problem_id']} under parent {parent.id}",
        why_selected=_rationale(group),
        expected_effect="Candidate should convert a judged loss into an auditable repair hypothesis.",
        assumption_ids=[parent.id, candidate.id],
        verifier="candidate_preflight_then_fresh_ablation",
        verification_plan="Preflight the routed trigger subset, then run fresh ablation only if the candidate is retrievable and scoped.",
        rollback_condition="Reject if preflight is underpowered or fresh judgments do not beat the current graph/baseline.",
        status=TrialStatus.PENDING,
        artifacts={"source_group": group},
        metadata={
            "eval_id": eval_id,
            "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
            "parent_node_id": parent.id,
            "candidate_node_id": candidate.id,
            "source_problem_id": group["problem_id"],
        },
        trial_id=stable_id("trial", eval_id, group["problem_id"], parent.id, ProposalType.FAILURE_HYPOTHESIS.value),
    )


def _rationale(group: dict) -> str:
    if group.get("source_kind") == "skipped_judgment":
        return (
            f"Intervention lost on skipped row {group['problem_id']} "
            f"with source_skipped_reason={group.get('source_skipped_reason')} "
            f"and outcomes={group.get('outcomes')}."
        )
    return (
        f"Intervention lost on {group['problem_id']} with residual_type={group.get('residual_type')} "
        f"and outcomes={group.get('outcomes')}."
    )


def _priority(group: dict) -> float:
    outcomes = group.get("outcomes", {})
    priority = 0.55 + 0.08 * outcomes.get("loss", 0)
    if group.get("residual_type") == "memory_defect":
        priority += 0.08
    if group.get("gold_hit"):
        priority += 0.04
    if group.get("source_kind") == "skipped_judgment":
        priority += 0.06
    if group.get("source_skipped_reason") == "policy_skipped":
        priority += 0.04
    return min(0.95, priority)


def _skip_reason(
    problem: dict,
    meta: dict | None,
    skip_domains: set[str],
    skip_missing_meta: bool,
) -> str:
    if skip_missing_meta and not meta:
        return "missing_meta"
    if skip_domains and problem.get("domain", "") in skip_domains:
        return "policy_skipped"
    return ""


def _outcome(judgment: dict, intervention_variant: str, baseline_variant: str) -> str:
    winner = judgment.get("winner")
    if winner == intervention_variant:
        return "win"
    if winner == baseline_variant:
        return "loss"
    return "tie"


def _gold_ids(problem: dict) -> list[str]:
    return sorted(f"strategy_{sid}" for sid in problem.get("coverage_tags", []) if sid)


def _skipped_residual_type(reason: str) -> str:
    if reason == "policy_skipped":
        return "assumption_defect"
    if reason == "missing_meta":
        return "memory_defect"
    return "unknown"


def _path_key(path: str) -> str:
    return Path(path).name if path else ""


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_codes(gold_ids: list[str]) -> list[str]:
    codes = []
    for node_id in gold_ids:
        if isinstance(node_id, str) and node_id.startswith("strategy_"):
            codes.append(node_id.split("_", 1)[1])
    return codes


def _activation_keywords(problem: dict, meta: dict) -> list[str]:
    text = " ".join([
        str(problem.get("description", "")),
        str(meta.get("critical_reframe", "")),
        str(meta.get("rewritten_problem", "")),
    ]).lower()
    words = []
    seen = set()
    for raw in text.replace("/", " ").replace(",", " ").replace(".", " ").split():
        word = raw.strip("()[]{}:;!?\"'")
        if len(word) < 4 or word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) >= 12:
            break
    return words


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit].rstrip()
