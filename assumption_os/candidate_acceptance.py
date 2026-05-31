"""Acceptance gate for proposal candidate experiments.

After a fresh ablation produces pairwise judgments, this module maps those
judgments back to proposal ids. Accepted candidates can then be applied to the
graph; rejected or underpowered candidates remain as audit records.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from .graph_memory import JsonlGraphStore
from .proposal_overlay import iter_matching_proposals, load_proposal_payload
from .schema import AssumptionEdge, AssumptionNode, TrialManifest, TrialStatus


class AcceptanceDecision(str, Enum):
    ACCEPT = "accept"
    REJECT_BENEFIT = "reject_benefit"
    REJECT_HARM = "reject_harm"
    INSUFFICIENT_JUDGMENTS = "insufficient_judgments"
    DEFERRED_NOT_READY = "deferred_not_ready"
    MANIFEST_ONLY = "manifest_only"


@dataclass(frozen=True)
class CandidateAcceptanceSummary:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    decision: AcceptanceDecision
    trigger_outcomes: dict[str, int]
    control_outcomes: dict[str, int]
    trigger_utility: float | None
    trigger_lcb90: float | None
    control_loss_rate: float | None
    control_loss_ucb90: float | None
    judged_trigger_problem_ids: list[str]
    judged_control_problem_ids: list[str]
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def build_acceptance_payload(
    *,
    proposal_payload: dict,
    preflight_payload: dict,
    judgment_paths: Iterable[Path],
    candidate_variant: str,
    baseline_variant: str,
    eval_id: str,
    proposal_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
    min_trigger_judgments: int = 3,
    benefit_lcb90: float = 0.54,
    control_loss_ucb90: float = 0.35,
) -> dict:
    judgments = _load_judgments(judgment_paths)
    preflight_by_id = {s["proposal_id"]: s for s in preflight_payload.get("summaries", [])}
    summaries = []
    for proposal in iter_matching_proposals(
        proposal_payload,
        proposal_ids=proposal_ids,
        proposal_types=proposal_types,
    ):
        summaries.append(_summarize_proposal(
            proposal=proposal,
            preflight=preflight_by_id.get(proposal.get("proposal_id", ""), {}),
            judgments=judgments,
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            min_trigger_judgments=min_trigger_judgments,
            benefit_lcb90=benefit_lcb90,
            control_loss_ucb90=control_loss_ucb90,
        ))
    return {
        "eval_id": eval_id,
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "source_preflight_eval_id": preflight_payload.get("eval_id"),
        "candidate_variant": candidate_variant,
        "baseline_variant": baseline_variant,
        "thresholds": {
            "min_trigger_judgments": min_trigger_judgments,
            "benefit_lcb90": benefit_lcb90,
            "control_loss_ucb90": control_loss_ucb90,
        },
        "decision_counts": dict(Counter(s.decision.value for s in summaries)),
        "accepted_proposal_ids": [s.proposal_id for s in summaries if s.decision == AcceptanceDecision.ACCEPT],
        "summaries": [s.to_dict() for s in summaries],
    }


def apply_accepted_candidates(
    store: JsonlGraphStore,
    proposal_payload: dict,
    acceptance_payload: dict,
) -> list[str]:
    """Apply only accepted candidate proposals to the graph store."""

    accepted = set(acceptance_payload.get("accepted_proposal_ids", []))
    applied: list[str] = []
    summary_by_id = {s["proposal_id"]: s for s in acceptance_payload.get("summaries", [])}
    for proposal in proposal_payload.get("proposals", []):
        if proposal.get("proposal_id") not in accepted:
            continue
        if proposal.get("candidate_node"):
            node = AssumptionNode.from_dict(proposal["candidate_node"])
            node.status = "active"
            store.upsert_node(node)
            applied.append(node.id)
        for edge in proposal.get("edges", []):
            store.add_edge(AssumptionEdge.from_dict(edge))
        if proposal.get("manifest"):
            manifest = TrialManifest.from_dict(proposal["manifest"])
            manifest.observe(
                "Accepted by proposal acceptance gate.",
                status=TrialStatus.ACCEPTED,
            )
            manifest.metadata["acceptance_summary"] = summary_by_id.get(proposal["proposal_id"], {})
            store.append_trial(manifest)
    if applied:
        store.flush()
    return applied


def _summarize_proposal(
    *,
    proposal: dict,
    preflight: dict,
    judgments: dict[str, list[dict]],
    candidate_variant: str,
    baseline_variant: str,
    min_trigger_judgments: int,
    benefit_lcb90: float,
    control_loss_ucb90: float,
) -> CandidateAcceptanceSummary:
    proposal_id = proposal.get("proposal_id", "")
    candidate_id = proposal.get("candidate_node", {}).get("id") if proposal.get("candidate_node") else None
    if preflight.get("readiness") == "manifest_only" or not candidate_id:
        return _empty_summary(proposal, AcceptanceDecision.MANIFEST_ONLY, "No candidate node was tested.")
    if preflight.get("readiness") != "ready_for_fresh_ablation":
        return _empty_summary(proposal, AcceptanceDecision.DEFERRED_NOT_READY, "Preflight did not mark this proposal ready.")

    trigger_ids = preflight.get("trigger_problem_ids", [])
    control_ids = preflight.get("control_problem_ids", [])
    trigger_outcomes, judged_trigger = _outcomes_for_ids(
        trigger_ids, judgments, candidate_variant=candidate_variant, baseline_variant=baseline_variant)
    control_outcomes, judged_control = _outcomes_for_ids(
        control_ids, judgments, candidate_variant=candidate_variant, baseline_variant=baseline_variant)
    trigger_n = sum(trigger_outcomes.values())
    control_n = sum(control_outcomes.values())
    trigger_utility = _utility(trigger_outcomes) if trigger_n else None
    trigger_lcb = _normal_bound(trigger_utility, trigger_n, sign=-1) if trigger_utility is not None else None
    control_loss_rate = control_outcomes.get("loss", 0) / control_n if control_n else None
    control_loss_ucb = _normal_bound(control_loss_rate, control_n, sign=1) if control_loss_rate is not None else None

    if trigger_n < min_trigger_judgments:
        decision = AcceptanceDecision.INSUFFICIENT_JUDGMENTS
        rationale = "Not enough judged trigger rows for an acceptance decision."
    elif trigger_lcb is None or trigger_lcb < benefit_lcb90:
        decision = AcceptanceDecision.REJECT_BENEFIT
        rationale = "Trigger-row benefit lower bound did not clear the acceptance gate."
    elif control_loss_ucb is not None and control_loss_ucb > control_loss_ucb90:
        decision = AcceptanceDecision.REJECT_HARM
        rationale = "Control-row loss upper bound exceeded the harm gate."
    else:
        decision = AcceptanceDecision.ACCEPT
        rationale = "Trigger benefit passed and control harm stayed within threshold."

    return CandidateAcceptanceSummary(
        proposal_id=proposal_id,
        proposal_type=proposal.get("proposal_type", ""),
        parent_node_id=proposal.get("parent_node_id", ""),
        candidate_node_id=candidate_id,
        decision=decision,
        trigger_outcomes=dict(trigger_outcomes),
        control_outcomes=dict(control_outcomes),
        trigger_utility=trigger_utility,
        trigger_lcb90=trigger_lcb,
        control_loss_rate=control_loss_rate,
        control_loss_ucb90=control_loss_ucb,
        judged_trigger_problem_ids=judged_trigger,
        judged_control_problem_ids=judged_control,
        rationale=rationale,
    )


def _empty_summary(proposal: dict, decision: AcceptanceDecision, rationale: str) -> CandidateAcceptanceSummary:
    return CandidateAcceptanceSummary(
        proposal_id=proposal.get("proposal_id", ""),
        proposal_type=proposal.get("proposal_type", ""),
        parent_node_id=proposal.get("parent_node_id", ""),
        candidate_node_id=proposal.get("candidate_node", {}).get("id") if proposal.get("candidate_node") else None,
        decision=decision,
        trigger_outcomes={},
        control_outcomes={},
        trigger_utility=None,
        trigger_lcb90=None,
        control_loss_rate=None,
        control_loss_ucb90=None,
        judged_trigger_problem_ids=[],
        judged_control_problem_ids=[],
        rationale=rationale,
    )


def _load_judgments(paths: Iterable[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for pid, judgment in rows.items():
            out.setdefault(pid, []).append(judgment)
    return out


def _outcomes_for_ids(
    problem_ids: Iterable[str],
    judgments: dict[str, list[dict]],
    *,
    candidate_variant: str,
    baseline_variant: str,
) -> tuple[Counter[str], list[str]]:
    outcomes: Counter[str] = Counter()
    judged: list[str] = []
    for pid in problem_ids:
        for judgment in judgments.get(pid, []):
            outcome = _normalize_winner(judgment, candidate_variant, baseline_variant)
            outcomes[outcome] += 1
            judged.append(pid)
    return outcomes, judged


def _normalize_winner(judgment: dict, candidate_variant: str, baseline_variant: str) -> str:
    winner = judgment.get("winner", "tie")
    if winner == candidate_variant:
        return "win"
    if winner == baseline_variant:
        return "loss"
    if winner == "tie":
        return "tie"
    if winner in {"A", "B"}:
        a_was = judgment.get("a_was", "A")
        if a_was == candidate_variant:
            candidate_side = "A"
        elif a_was == baseline_variant:
            candidate_side = "B"
        else:
            candidate_side = a_was if a_was in {"A", "B"} else None
        if candidate_side and winner == candidate_side:
            return "win"
        if candidate_side and winner != candidate_side:
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


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--preflight", required=True)
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--candidate-variant", required=True)
    ap.add_argument("--baseline-variant", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--proposal-ids", nargs="*", default=None)
    ap.add_argument("--proposal-types", default="")
    ap.add_argument("--min-trigger-judgments", type=int, default=3)
    ap.add_argument("--benefit-lcb90", type=float, default=0.54)
    ap.add_argument("--control-loss-ucb90", type=float, default=0.35)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--apply-accepted", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    proposals = load_proposal_payload(_resolve(root, args.proposals))
    preflight = json.loads(_resolve(root, args.preflight).read_text(encoding="utf-8"))
    payload = build_acceptance_payload(
        proposal_payload=proposals,
        preflight_payload=preflight,
        judgment_paths=[_resolve(root, p) for p in args.judgments],
        candidate_variant=args.candidate_variant,
        baseline_variant=args.baseline_variant,
        eval_id=args.eval_id,
        proposal_ids=args.proposal_ids,
        proposal_types={x.strip() for x in args.proposal_types.split(",") if x.strip()},
        min_trigger_judgments=args.min_trigger_judgments,
        benefit_lcb90=args.benefit_lcb90,
        control_loss_ucb90=args.control_loss_ucb90,
    )
    if args.apply_accepted:
        payload["applied_candidate_node_ids"] = apply_accepted_candidates(
            JsonlGraphStore(_resolve(root, args.graph_dir)),
            proposals,
            payload,
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
