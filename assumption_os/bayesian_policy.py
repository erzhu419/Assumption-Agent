"""Bayesian policy scoring for self-evolution candidates.

This is a lightweight meta-reasoning layer over the existing graph evidence. It
does not replace the falsification gate.  It ranks which safe next action is
worth spending budget on by combining parent-node posterior performance,
preflight readiness, regression risk, and information value.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .graph_memory import JsonlGraphStore


class BayesianPolicyAction(str, Enum):
    APPLY_ACCEPTED = "apply_accepted"
    RUN_ABLATION = "run_ablation"
    COLLECT_EVIDENCE = "collect_evidence"
    REPAIR_SCOPE = "repair_scope"
    REPAIR_RETRIEVAL = "repair_retrieval"
    RECORD_ONLY = "record_only"
    REJECT = "reject"


@dataclass(frozen=True)
class BetaBelief:
    alpha: float
    beta: float
    n: float
    mean: float
    lcb90: float
    ucb90: float
    variance: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BayesianPolicyScore:
    proposal_id: str
    proposal_type: str
    parent_node_id: str
    candidate_node_id: str | None
    parent_belief: BetaBelief
    readiness: str | None
    falsification_decision: str | None
    regression_risk: str
    expected_value: float
    information_value: float
    risk_penalty: float
    posterior_priority: float
    recommended_action: BayesianPolicyAction
    rationale: list[str] = field(default_factory=list)
    command_hint: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parent_belief"] = self.parent_belief.to_dict()
        d["recommended_action"] = self.recommended_action.value
        return d


def build_bayesian_policy_payload(
    *,
    store: JsonlGraphStore,
    proposal_payload: dict,
    preflight_payload: dict,
    falsification_payload: dict | None = None,
    acceptance_payload: dict | None = None,
    regression_predictions: list[dict] | None = None,
) -> dict:
    preflight_by_id = {s.get("proposal_id"): s for s in preflight_payload.get("summaries", [])}
    falsification_by_id = {
        s.get("proposal_id"): s
        for s in (falsification_payload or {}).get("summaries", [])
    }
    acceptance_by_id = {
        s.get("proposal_id"): s
        for s in (acceptance_payload or {}).get("summaries", [])
    }
    regression_by_id = {r.get("proposal_id"): r for r in regression_predictions or []}
    scores = [
        score_proposal(
            store=store,
            proposal=proposal,
            preflight=preflight_by_id.get(proposal.get("proposal_id"), {}),
            falsification=falsification_by_id.get(proposal.get("proposal_id"), {}),
            acceptance=acceptance_by_id.get(proposal.get("proposal_id"), {}),
            regression=regression_by_id.get(proposal.get("proposal_id"), {}),
        )
        for proposal in proposal_payload.get("proposals", [])
    ]
    scores = sorted(scores, key=lambda s: (-s.posterior_priority, s.proposal_id))
    return {
        "source_proposal_eval_id": proposal_payload.get("eval_id"),
        "source_preflight_eval_id": preflight_payload.get("eval_id"),
        "source_falsification_eval_id": (falsification_payload or {}).get("source_proposal_eval_id"),
        "decision_counts": dict(Counter(s.recommended_action.value for s in scores)),
        "scores": [s.to_dict() for s in scores],
    }


def score_proposal(
    *,
    store: JsonlGraphStore,
    proposal: dict,
    preflight: dict,
    falsification: dict,
    acceptance: dict,
    regression: dict,
) -> BayesianPolicyScore:
    proposal_id = proposal.get("proposal_id", "")
    parent_id = proposal.get("parent_node_id", "")
    candidate = proposal.get("candidate_node") or {}
    belief = parent_belief(store, parent_id)
    readiness = preflight.get("readiness")
    falsification_decision = falsification.get("decision")
    regression_risk = regression.get("risk", "unknown")

    risk_penalty = _risk_penalty(regression_risk)
    information_value = _information_value(belief, readiness, falsification_decision)
    expected_value = belief.mean - risk_penalty

    action, rationale = _recommend_action(
        readiness=readiness,
        falsification_decision=falsification_decision,
        acceptance_decision=acceptance.get("decision"),
        expected_value=expected_value,
        information_value=information_value,
        regression_risk=regression_risk,
        candidate_exists=bool(candidate),
    )
    posterior_priority = _priority(
        action=action,
        expected_value=expected_value,
        information_value=information_value,
        risk_penalty=risk_penalty,
        belief=belief,
    )

    return BayesianPolicyScore(
        proposal_id=proposal_id,
        proposal_type=proposal.get("proposal_type", ""),
        parent_node_id=parent_id,
        candidate_node_id=candidate.get("id"),
        parent_belief=belief,
        readiness=readiness,
        falsification_decision=falsification_decision,
        regression_risk=regression_risk,
        expected_value=round(expected_value, 4),
        information_value=round(information_value, 4),
        risk_penalty=round(risk_penalty, 4),
        posterior_priority=round(posterior_priority, 4),
        recommended_action=action,
        rationale=rationale,
        command_hint=preflight.get("command_hint", ""),
    )


def parent_belief(store: JsonlGraphStore, node_id: str, *, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> BetaBelief:
    alpha = prior_alpha
    beta = prior_beta
    n = 0.0
    evidence_trial_ids = set()

    for ev in store.evidence.values():
        if ev.node_id != node_id:
            continue
        value = _evidence_value(ev.value, ev.outcome)
        if value is None:
            continue
        alpha += value
        beta += 1.0 - value
        n += 1.0
        if ev.details.get("trial_id"):
            evidence_trial_ids.add(ev.details["trial_id"])

    for trial in store.trials.values():
        if trial.trial_id in evidence_trial_ids:
            continue
        if node_id not in trial.assumption_ids:
            continue
        value = _trial_value(trial.metadata.get("outcome"), trial.status)
        if value is None:
            continue
        alpha += value
        beta += 1.0 - value
        n += 1.0

    return _beta_belief(alpha, beta, n)


def _beta_belief(alpha: float, beta: float, n: float) -> BetaBelief:
    total = alpha + beta
    mean = alpha / total if total else 0.5
    variance = (alpha * beta) / (total * total * (total + 1.0)) if total > 0 else 0.0
    sd = math.sqrt(max(variance, 0.0))
    return BetaBelief(
        alpha=round(alpha, 4),
        beta=round(beta, 4),
        n=round(n, 4),
        mean=round(mean, 4),
        lcb90=round(max(0.0, mean - 1.28 * sd), 4),
        ucb90=round(min(1.0, mean + 1.28 * sd), 4),
        variance=round(variance, 6),
    )


def _evidence_value(value: float | None, outcome: str | None) -> float | None:
    if value is not None:
        return max(0.0, min(1.0, float(value)))
    if outcome in {"success", "accepted", "win"}:
        return 1.0
    if outcome in {"failed", "rejected", "loss"}:
        return 0.0
    if outcome in {"tie", "observed"}:
        return 0.5
    return None


def _trial_value(outcome: str | None, status) -> float | None:
    status_value = getattr(status, "value", status)
    if outcome == "win" or status_value == "accepted":
        return 1.0
    if outcome == "loss" or status_value in {"failed", "rejected"}:
        return 0.0
    if outcome == "tie" or status_value == "observed":
        return 0.5
    return None


def _risk_penalty(risk: str) -> float:
    return {
        "low": 0.02,
        "unknown": 0.06,
        "medium": 0.12,
        "high": 0.28,
    }.get(risk or "unknown", 0.06)


def _information_value(belief: BetaBelief, readiness: str | None, falsification_decision: str | None) -> float:
    readiness_weight = {
        "ready_for_fresh_ablation": 1.0,
        "needs_more_trigger_rows": 0.45,
        "needs_retrieval_fix": 0.35,
        "needs_scope_fix": 0.2,
        "manifest_only": 0.05,
    }.get(readiness or "", 0.15)
    if falsification_decision == "ready_for_ablation":
        readiness_weight = max(readiness_weight, 1.0)
    if falsification_decision in {"blocked_underpowered", "blocked_retrieval"}:
        readiness_weight = min(readiness_weight, 0.45)
    if falsification_decision == "blocked_scope_risk":
        readiness_weight = min(readiness_weight, 0.15)
    return 2.0 * math.sqrt(max(belief.variance, 0.0)) * readiness_weight


def _recommend_action(
    *,
    readiness: str | None,
    falsification_decision: str | None,
    acceptance_decision: str | None,
    expected_value: float,
    information_value: float,
    regression_risk: str,
    candidate_exists: bool,
) -> tuple[BayesianPolicyAction, list[str]]:
    reasons = []
    if not candidate_exists:
        return BayesianPolicyAction.RECORD_ONLY, ["proposal has no candidate node"]
    if acceptance_decision == "accept":
        return BayesianPolicyAction.APPLY_ACCEPTED, ["candidate passed acceptance gate"]
    if acceptance_decision in {"reject_benefit", "reject_harm"}:
        return BayesianPolicyAction.REJECT, [f"acceptance gate returned {acceptance_decision}"]
    if falsification_decision == "blocked_scope_risk" or readiness == "needs_scope_fix":
        return BayesianPolicyAction.REPAIR_SCOPE, ["scope risk blocks ablation"]
    if falsification_decision == "blocked_retrieval" or readiness == "needs_retrieval_fix":
        return BayesianPolicyAction.REPAIR_RETRIEVAL, ["retrieval miss blocks fair ablation"]
    if falsification_decision == "blocked_underpowered" or readiness == "needs_more_trigger_rows":
        return BayesianPolicyAction.COLLECT_EVIDENCE, ["not enough routed trigger rows"]
    if falsification_decision == "ready_for_ablation" or readiness == "ready_for_fresh_ablation":
        reasons.append("candidate is preflight-ready")
        if regression_risk == "high":
            reasons.append("high regression risk requires repair before spending ablation budget")
            return BayesianPolicyAction.REPAIR_SCOPE, reasons
        if expected_value + information_value >= 0.45:
            reasons.append("posterior expected value plus information value clears budget threshold")
            return BayesianPolicyAction.RUN_ABLATION, reasons
        reasons.append("posterior value is too weak for fresh ablation")
        return BayesianPolicyAction.COLLECT_EVIDENCE, reasons
    return BayesianPolicyAction.COLLECT_EVIDENCE, ["candidate lacks enough evidence for a stronger action"]


def _priority(
    *,
    action: BayesianPolicyAction,
    expected_value: float,
    information_value: float,
    risk_penalty: float,
    belief: BetaBelief,
) -> float:
    action_bonus = {
        BayesianPolicyAction.APPLY_ACCEPTED: 1.0,
        BayesianPolicyAction.RUN_ABLATION: 0.75,
        BayesianPolicyAction.COLLECT_EVIDENCE: 0.35,
        BayesianPolicyAction.REPAIR_RETRIEVAL: 0.25,
        BayesianPolicyAction.REPAIR_SCOPE: 0.15,
        BayesianPolicyAction.RECORD_ONLY: 0.05,
        BayesianPolicyAction.REJECT: -0.25,
    }[action]
    uncertainty_bonus = 0.15 if belief.n < 3 else 0.0
    return action_bonus + 0.55 * expected_value + 0.9 * information_value - 0.4 * risk_penalty + uncertainty_bonus


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
    ap.add_argument("--preflight", required=True)
    ap.add_argument("--falsification", default=None)
    ap.add_argument("--acceptance", default=None)
    ap.add_argument("--regression-predictions", default=None)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_bayesian_policy_payload(
        store=JsonlGraphStore(_resolve(root, args.graph_dir)),
        proposal_payload=_load_json(_resolve(root, args.proposals)),
        preflight_payload=_load_json(_resolve(root, args.preflight)),
        falsification_payload=_load_json(_resolve(root, args.falsification)) if args.falsification else None,
        acceptance_payload=_load_json(_resolve(root, args.acceptance)) if args.acceptance else None,
        regression_predictions=_load_json(_resolve(root, args.regression_predictions)) if args.regression_predictions else None,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
