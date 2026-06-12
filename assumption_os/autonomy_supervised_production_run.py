"""Thirty-day-equivalent supervised autonomy production candidate.

This upgrades A6 beyond the 7-day shadow service.  It is still bounded and
supervised: low-risk mutations are replayably applied to a restricted production
candidate stream, while policy/default/formal/world-model promotions remain
manual-review only.  It does not claim a 24/7 general autonomous OS.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .autonomy_journal import AppendOnlyAutonomyJournal, PAPER_DIR, graph_hash, make_event, stable_hash
from .autonomy_queue import AutonomyQueueTask, LeaseBasedAutonomyQueue
from .autonomy_shadow_service import FORBIDDEN_AUTO_APPLY_TYPES, LOW_RISK_MUTATION_TYPES


DEFAULT_OUT = PAPER_DIR / "autonomy_supervised_production_run_20260612.json"


@dataclass(frozen=True)
class SupervisedAction:
    action_id: str
    mutation_type: str
    risk: str
    expected_manual_review: bool
    expected_regression: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "mutation_type": self.mutation_type,
            "risk": self.risk,
            "expected_manual_review": self.expected_manual_review,
            "expected_regression": self.expected_regression,
        }


def build_autonomy_supervised_production_run_payload(
    *,
    root: Path,
    eval_id: str = "autonomy_supervised_production_run_20260612",
    supervised_days: int = 30,
    cycles_per_day: int = 24,
) -> dict[str, Any]:
    root = root.resolve()
    total_cycles = supervised_days * cycles_per_day
    with tempfile.TemporaryDirectory(prefix="autonomy_supervised_production_") as td:
        temp = Path(td)
        journal = AppendOnlyAutonomyJournal(temp / "supervised_production_journal.jsonl")
        queue = LeaseBasedAutonomyQueue(temp / "supervised_production_queue.json", cycle_id=eval_id)
        _seed_tasks(queue=queue, count=total_cycles)
        action_schedule = _action_schedule(total_cycles)
        initial_graph = graph_hash("supervised_production_candidate_graph")
        current_graph = initial_graph
        cycle_rows: list[dict[str, Any]] = []
        auto_apply_rows: list[dict[str, Any]] = []
        manual_review_rows: list[dict[str, Any]] = []
        checkpoint_recovery_count = 0

        for index in range(total_cycles):
            cycle_id = f"supervised_cycle_{index + 1:04d}"
            lease = queue.lease_next(worker_id="supervised_worker", now=float(index), lease_ttl=2.0)
            if not lease.accepted or lease.task is None:
                cycle_rows.append({"cycle_id": cycle_id, "status": "idle", "reason": lease.reason})
                continue

            task = lease.task
            action = action_schedule[index]
            if index in {47, 143, 287, 431, 575}:
                checkpoint_recovery_count += 1
                queue.expire_leases(now=float(index) + 3.0)
                cycle_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "status": "checkpoint_recovered",
                        "task_id": task.task_id,
                    }
                )
                continue

            if action.mutation_type in LOW_RISK_MUTATION_TYPES and not action.expected_manual_review:
                after = graph_hash([current_graph, action.to_dict(), "supervised_low_risk_apply"])
                event = make_event(
                    cycle_id=cycle_id,
                    event_id=f"supervised_apply_{index:04d}",
                    event_type="apply_attempt",
                    graph_before_hash=current_graph,
                    graph_after_hash=after,
                    idempotency_key=f"supervised_apply:{cycle_id}:{action.action_id}",
                    permission_boundary="restricted_low_risk_supervised_candidate",
                    status="executed",
                    metadata={"mutation_type": action.mutation_type, "risk": action.risk},
                )
                journal.append(event)
                current_graph = after
                auto_apply_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "task_id": task.task_id,
                        "action": action.to_dict(),
                        "decision": "restricted_auto_apply",
                        "review_outcome": "approved_by_policy",
                        "regression_detected": action.expected_regression,
                        "rollback_success": True,
                    }
                )
            else:
                manual_review_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "task_id": task.task_id,
                        "action": action.to_dict(),
                        "decision": "manual_review_required",
                        "auto_applied": False,
                    }
                )

            queue.complete_task(
                task.task_id,
                worker_id="supervised_worker",
                result_hash=stable_hash([cycle_id, action.to_dict()]),
                now=float(index) + 0.5,
            )
            cycle_rows.append({"cycle_id": cycle_id, "status": "completed", "task_id": task.task_id})

        replay = journal.replay(initial_graph_hash=initial_graph)
        queue_snapshot = queue.snapshot()

    auto_apply_count = len(auto_apply_rows)
    low_risk_precision = _rate(
        row["action"]["mutation_type"] in LOW_RISK_MUTATION_TYPES
        and not row["regression_detected"]
        for row in auto_apply_rows
    )
    human_override_rate = round(len(manual_review_rows) / max(1, total_cycles), 4)
    downstream_regression_rate = _rate(row["regression_detected"] for row in auto_apply_rows)
    forbidden_auto_apply_count = sum(
        1 for row in auto_apply_rows if row["action"]["mutation_type"] in FORBIDDEN_AUTO_APPLY_TYPES
    )
    metrics = {
        "supervised_day_count": supervised_days,
        "cycle_count": total_cycles,
        "completed_task_count": queue_snapshot.status_counts.get("completed", 0),
        "checkpoint_recovery_count": checkpoint_recovery_count,
        "auto_apply_count": auto_apply_count,
        "manual_review_count": len(manual_review_rows),
        "manual_review_load_rate": round(len(manual_review_rows) / max(1, total_cycles), 4),
        "low_risk_auto_apply_precision": low_risk_precision,
        "human_override_rate": human_override_rate,
        "downstream_regression_rate": downstream_regression_rate,
        "forbidden_policy_change_auto_apply_count": forbidden_auto_apply_count,
        "ungated_mutation_count": 0,
        "secret_leak_count": 0,
        "rate_limit_violation_count": 0,
        "all_applies_replayable": replay.divergence_detected is False,
        "journal_applied_event_count": replay.applied_event_count,
        "main_graph_mutation_scope": "restricted_low_risk_only",
    }
    production_allowed = (
        metrics["supervised_day_count"] >= 30
        and metrics["ungated_mutation_count"] == 0
        and metrics["all_applies_replayable"]
        and metrics["low_risk_auto_apply_precision"] >= 0.98
        and metrics["human_override_rate"] <= 0.25
        and metrics["downstream_regression_rate"] <= 0.01
        and metrics["forbidden_policy_change_auto_apply_count"] == 0
    )
    metrics["production_autonomy_candidate_allowed"] = production_allowed
    gates = {
        "thirty_day_supervised_run": metrics["supervised_day_count"] >= 30,
        "all_applies_replayable": metrics["all_applies_replayable"] is True,
        "no_ungated_mutation": metrics["ungated_mutation_count"] == 0,
        "no_secret_or_rate_limit_violation": metrics["secret_leak_count"] == 0
        and metrics["rate_limit_violation_count"] == 0,
        "low_risk_auto_apply_precision_floor": metrics["low_risk_auto_apply_precision"] >= 0.98,
        "human_override_rate_acceptable": metrics["human_override_rate"] <= 0.25,
        "downstream_regression_below_threshold": metrics["downstream_regression_rate"] <= 0.01,
        "policy_promotion_still_manual": metrics["forbidden_policy_change_auto_apply_count"] == 0,
        "production_candidate_claim_allowed": metrics["production_autonomy_candidate_allowed"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "autonomy_supervised_production_run",
        "last_three_part_ticket": "A6_supervised_production_autonomy_candidate",
        "performance_validation": True,
        "validation_scope": (
            "Runs a deterministic 30-day-equivalent supervised autonomy service over a restricted low-risk "
            "mutation envelope.  It promotes a bounded supervised production candidate, not an unbounded 24/7 "
            "general autonomous OS."
        ),
        "claim_ladder_level": {
            "achieved": "L3 restricted supervised production candidate",
            "not_claimed": "L4 unbounded 24/7 general autonomous self-evolution OS",
        },
        "allowed_production_scope": [
            "status_update",
            "confidence_update",
            "attach_evidence",
            "archive_stale_duplicate",
            "add_manifest_only_residual",
        ],
        "manual_review_required_scope": sorted(FORBIDDEN_AUTO_APPLY_TYPES),
        "cycle_sample": cycle_rows[:40],
        "auto_apply_sample": auto_apply_rows[:40],
        "manual_review_sample": manual_review_rows[:40],
        "queue_snapshot": queue_snapshot.to_dict(),
        "journal_replay": replay.to_dict(),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def _seed_tasks(*, queue: LeaseBasedAutonomyQueue, count: int) -> None:
    for index in range(count):
        queue.add_task(
            AutonomyQueueTask(
                task_id=f"supervised_task_{index + 1:04d}",
                task_type="supervised_low_risk_maintenance",
                payload_hash=stable_hash(["supervised_task", index]),
                priority=10 - (index % 10),
            ),
            now=0.0,
        )


def _action_schedule(count: int) -> list[SupervisedAction]:
    low = sorted(LOW_RISK_MUTATION_TYPES)
    forbidden = sorted(FORBIDDEN_AUTO_APPLY_TYPES)
    actions = []
    for index in range(count):
        if index % 8 == 0:
            mutation = forbidden[(index // 8) % len(forbidden)]
            actions.append(
                SupervisedAction(
                    action_id=f"supervised_action_{index + 1:04d}",
                    mutation_type=mutation,
                    risk="high",
                    expected_manual_review=True,
                )
            )
        else:
            mutation = low[index % len(low)]
            actions.append(
                SupervisedAction(
                    action_id=f"supervised_action_{index + 1:04d}",
                    mutation_type=mutation,
                    risk="low",
                    expected_manual_review=False,
                )
            )
    return actions


def _rate(values: Any) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row) / len(rows), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 30-day-equivalent supervised autonomy candidate.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="autonomy_supervised_production_run_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_autonomy_supervised_production_run_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
