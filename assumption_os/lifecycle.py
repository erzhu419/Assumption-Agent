"""Lifecycle action planning for self-evolving assumptions.

Conditioned evaluation says what happened.  This module turns those gate results
into auditable next actions without mutating the graph automatically.  It is the
bridge between evaluation and self-modification: promote only when evidence is
clean, expand retrieval when a useful node is under-exposed, narrow scope when a
node harms no-fire rows, revise when conditioned utility is low, and collect
more evidence when the signal is too small.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .conditioned_eval import GateDecision
from .schema import TrialManifest, TrialStatus, stable_id


class LifecycleActionType(str, Enum):
    PROMOTE_ASSUMPTION = "promote_assumption"
    KEEP_COLLECT_EVIDENCE = "keep_collect_evidence"
    EXPAND_RETRIEVAL = "expand_retrieval"
    NARROW_SCOPE = "narrow_scope"
    REVISE_ASSUMPTION = "revise_assumption"
    DEFER = "defer"


@dataclass(frozen=True)
class LifecycleAction:
    node_id: str
    action_type: LifecycleActionType
    priority: float
    rationale: str
    proposed_updates: dict = field(default_factory=dict)
    verification_plan: str = ""
    rollback_condition: str = ""
    source: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d

    def to_trial_manifest(self, *, eval_id: str) -> TrialManifest:
        return TrialManifest(
            problem_id=f"lifecycle::{self.node_id}",
            action_type=f"lifecycle_{self.action_type.value}",
            component="assumption_lifecycle",
            assumption=f"Lifecycle action {self.action_type.value} for {self.node_id}",
            why_selected=self.rationale,
            expected_effect=self.proposed_updates.get("expected_effect", "Improve future conditioned utility or reduce harm."),
            assumption_ids=[self.node_id],
            verifier="conditioned_eval_gate",
            verification_plan=self.verification_plan,
            rollback_condition=self.rollback_condition,
            cost=float(self.proposed_updates.get("estimated_cost", 0.0)),
            status=TrialStatus.PENDING,
            artifacts={"source_summary": self.source},
            metadata={
                "eval_id": eval_id,
                "priority": self.priority,
                "action_type": self.action_type.value,
                "proposed_updates": self.proposed_updates,
            },
            trial_id=stable_id("trial", eval_id, self.node_id, self.action_type.value),
        )


def plan_lifecycle_actions(
    summaries: Iterable[dict],
    *,
    eval_id: str,
    include_deferred: bool = False,
    max_actions: int | None = None,
) -> list[LifecycleAction]:
    actions = [_plan_one(_normalize_summary(s), eval_id=eval_id) for s in summaries]
    if not include_deferred:
        actions = [a for a in actions if a.action_type != LifecycleActionType.DEFER]
    actions = sorted(actions, key=lambda a: (-a.priority, a.node_id, a.action_type.value))
    return actions[:max_actions] if max_actions else actions


def build_lifecycle_payload(
    conditioned_payload: dict,
    *,
    eval_id: str,
    include_deferred: bool = False,
    max_actions: int | None = None,
) -> dict:
    actions = plan_lifecycle_actions(
        conditioned_payload.get("summaries", []),
        eval_id=eval_id,
        include_deferred=include_deferred,
        max_actions=max_actions,
    )
    return {
        "eval_id": eval_id,
        "source_rows": conditioned_payload.get("rows"),
        "source_thresholds": conditioned_payload.get("thresholds", {}),
        "source_decision_counts": conditioned_payload.get("decision_counts", {}),
        "action_counts": _count_actions(actions),
        "actions": [a.to_dict() for a in actions],
        "trial_manifests": [a.to_trial_manifest(eval_id=eval_id).to_dict() for a in actions],
    }


def _plan_one(summary: dict, *, eval_id: str) -> LifecycleAction:
    decision = GateDecision(summary.get("decision", GateDecision.INSUFFICIENT_EVIDENCE.value))
    node_id = summary["node_id"]
    utility = _none_as(summary.get("utility_when_active_should_fire"), None)
    utility_lcb = _none_as(summary.get("utility_lcb90"), None)
    coverage = _none_as(summary.get("should_fire_coverage"), None)
    harm_ucb = _none_as(summary.get("harm_ucb90"), None)
    route_counts = summary.get("route_counts", {})
    active_counts = summary.get("active_counts", {})
    should_n = int(route_counts.get("should_fire", 0))
    active_should_n = int(active_counts.get("should_fire", 0))
    exposure_gap = max(0, should_n - active_should_n)
    source = {
        "decision": decision.value,
        "route_counts": route_counts,
        "active_counts": active_counts,
        "active_should_fire_outcomes": summary.get("active_should_fire_outcomes", {}),
        "active_no_fire_outcomes": summary.get("active_no_fire_outcomes", {}),
        "utility_lcb90": utility_lcb,
        "harm_ucb90": harm_ucb,
        "reasons": summary.get("reasons", []),
    }

    if decision == GateDecision.PROMOTE:
        return LifecycleAction(
            node_id=node_id,
            action_type=LifecycleActionType.PROMOTE_ASSUMPTION,
            priority=0.95,
            rationale="Conditioned utility passes the benefit gate with enough exposure and no observed no-fire harm.",
            proposed_updates={
                "confidence_delta": 0.04,
                "metaproductivity_delta": 0.03,
                "tags_add": ["conditioned_promoted"],
                "expected_effect": "Preserve this node as an active high-priority assumption for similar routed problems.",
            },
            verification_plan="Re-run conditioned gate on a fresh split before increasing default prompt budget.",
            rollback_condition="Demote if fresh conditioned utility LCB90 falls below threshold or no-fire harm appears.",
            source=source,
        )

    if decision == GateDecision.KEEP:
        return LifecycleAction(
            node_id=node_id,
            action_type=LifecycleActionType.KEEP_COLLECT_EVIDENCE,
            priority=0.65,
            rationale="Conditioned utility is positive but exposure is still too small for promotion.",
            proposed_updates={
                "confidence_delta": 0.01,
                "expected_effect": "Keep active while collecting additional should-fire examples.",
                "needed_rows": max(0, 6 - active_should_n),
            },
            verification_plan="Collect more should-fire rows and rerun the conditioned gate.",
            rollback_condition="Revise if added rows push utility LCB90 below threshold.",
            source=source,
        )

    if decision == GateDecision.EXPAND_RETRIEVAL:
        if utility is not None and utility >= 0.65:
            return LifecycleAction(
                node_id=node_id,
                action_type=LifecycleActionType.EXPAND_RETRIEVAL,
                priority=min(0.9, 0.55 + 0.02 * exposure_gap + 0.2 * utility),
                rationale="The node looks useful when active but misses many should-fire rows.",
                proposed_updates={
                    "retrieval_policy": "add activation hints or stronger tags for the should-fire subset",
                    "target_should_fire_rows": should_n,
                    "active_should_fire_rows": active_should_n,
                    "expected_effect": "Increase should-fire coverage without increasing active no-fire harm.",
                },
                verification_plan="Apply retrieval hints on a copy of the graph, rerun retrieval audit, then rerun conditioned gate.",
                rollback_condition="Undo retrieval hints if no-fire active rows increase with losses.",
                source=source,
            )
        return LifecycleAction(
            node_id=node_id,
            action_type=LifecycleActionType.KEEP_COLLECT_EVIDENCE,
            priority=0.45 + min(0.25, 0.02 * should_n),
            rationale="The gate says under-exposed, but current active evidence is absent or not strong enough to expand retrieval safely.",
            proposed_updates={
                "needed_rows": max(1, 3 - active_should_n),
                "expected_effect": "Disambiguate whether this is a retrieval miss or a weak assumption before editing retrieval policy.",
            },
            verification_plan="Manually inspect should-fire misses and collect at least three active examples before expansion.",
            rollback_condition="Convert to revise if new active should-fire examples are losses.",
            source=source,
        )

    if decision == GateDecision.NARROW_SCOPE:
        return LifecycleAction(
            node_id=node_id,
            action_type=LifecycleActionType.NARROW_SCOPE,
            priority=0.85,
            rationale="The node creates active no-fire harm and needs tighter applicability conditions.",
            proposed_updates={
                "context_conditions_add": ["exclude the no-fire conditions that produced losses"],
                "risk_predictions_add": ["negative transfer outside routed should-fire subset"],
                "expected_effect": "Reduce off-scope activation while preserving should-fire utility.",
            },
            verification_plan="Add explicit exclusion conditions, rerun retrieval audit, and verify no-fire harm UCB90 falls.",
            rollback_condition="Undo narrowing if should-fire coverage collapses without reducing harm.",
            source=source,
        )

    if decision == GateDecision.REVISE:
        return LifecycleAction(
            node_id=node_id,
            action_type=LifecycleActionType.REVISE_ASSUMPTION,
            priority=min(0.8, 0.45 + 0.02 * active_should_n + (0.2 if utility_lcb is not None and utility_lcb < 0.2 else 0.0)),
            rationale="The node was exposed on should-fire rows but conditioned utility did not clear the lower-bound gate.",
            proposed_updates={
                "revision_focus": "tighten claim, context conditions, or executor template; do not promote on pooled wins",
                "observed_utility_lcb90": utility_lcb,
                "expected_effect": "Turn a broad or weak assumption into a narrower, testable variant.",
            },
            verification_plan="Create a candidate child assumption and test it against the same routed subset plus outside controls.",
            rollback_condition="Reject the child if it fails to beat the parent on conditioned utility.",
            source=source,
        )

    return LifecycleAction(
        node_id=node_id,
        action_type=LifecycleActionType.DEFER,
        priority=0.15 + min(0.2, 0.03 * active_should_n),
        rationale="The conditioned gate does not have enough evidence for a lifecycle edit.",
        proposed_updates={
            "expected_effect": "Avoid overfitting lifecycle edits to tiny routed subsets.",
            "needed_rows": max(0, 3 - active_should_n),
        },
        verification_plan="Collect more routed examples before making graph edits.",
        rollback_condition="No graph edit was made.",
        source=source,
    )


def _normalize_summary(summary) -> dict:
    if hasattr(summary, "to_dict"):
        return summary.to_dict()
    return dict(summary)


def _none_as(value, default):
    return default if value is None else value


def _count_actions(actions: Iterable[LifecycleAction]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        key = action.action_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--conditioned-summary", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--include-deferred", action="store_true")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = build_lifecycle_payload(
        _load_json(_resolve(root, args.conditioned_summary)),
        eval_id=args.eval_id,
        include_deferred=args.include_deferred,
        max_actions=args.top_n,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
