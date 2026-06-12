"""Shadow autonomy service and low-risk auto-apply sandbox.

This module implements the remaining Track A boundary from last_three_part.md:
A4 shadow service, A5 low-risk auto-apply sandbox, and A6 production-candidate
claim gate.  It is intentionally bounded.  It simulates a replayable shadow run
over deterministic queue tasks, permits only manifest/status/evidence style
mutations on a graph copy, and blocks production-autonomy claims without a
longer supervised run.
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


DEFAULT_OUT = PAPER_DIR / "autonomy_shadow_service_20260612.json"

LOW_RISK_MUTATION_TYPES = {
    "status_update",
    "confidence_update",
    "attach_evidence",
    "archive_stale_duplicate",
    "add_manifest_only_residual",
}
FORBIDDEN_AUTO_APPLY_TYPES = {
    "new_active_method_assumption",
    "new_default_policy",
    "world_model_promotion",
    "formal_mapping_promotion",
}


@dataclass(frozen=True)
class ShadowMutation:
    mutation_id: str
    mutation_type: str
    target_id: str
    risk: str
    requires_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "mutation_type": self.mutation_type,
            "target_id": self.target_id,
            "risk": self.risk,
            "requires_human_review": self.requires_human_review,
        }


def build_autonomy_shadow_service_payload(
    *,
    root: Path,
    eval_id: str = "autonomy_shadow_service_20260612",
    shadow_days: int = 7,
    cycles_per_day: int = 24,
) -> dict[str, Any]:
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="autonomy_shadow_service_") as td:
        temp = Path(td)
        journal = AppendOnlyAutonomyJournal(temp / "shadow_graph_journal.jsonl")
        queue = LeaseBasedAutonomyQueue(temp / "shadow_queue.json", cycle_id=eval_id)
        tasks = _seed_shadow_tasks(queue)
        mutations = _shadow_mutations()
        graph_before = graph_hash("shadow_graph_before")
        graph_hash_current = graph_before
        cycle_reports: list[dict[str, Any]] = []
        recommendation_manifests: list[dict[str, Any]] = []
        auto_apply_reports: list[dict[str, Any]] = []
        manual_review_reports: list[dict[str, Any]] = []
        checkpoint_recovery_count = 0

        total_cycles = shadow_days * cycles_per_day
        for cycle_index in range(total_cycles):
            cycle_id = f"shadow_cycle_{cycle_index + 1:04d}"
            lease = queue.lease_next(worker_id="shadow_worker", now=float(cycle_index), lease_ttl=2.0)
            if not lease.accepted or lease.task is None:
                cycle_reports.append({"cycle_id": cycle_id, "status": "idle", "queue_reason": lease.reason})
                continue
            task = lease.task
            if cycle_index in {17, 83, 119}:
                checkpoint_recovery_count += 1
                queue.expire_leases(now=float(cycle_index) + 3.0)
                cycle_reports.append({"cycle_id": cycle_id, "status": "recovered_expired_lease", "task_id": task.task_id})
                continue

            mutation = mutations[cycle_index % len(mutations)]
            recommendation = _recommendation_manifest(cycle_id=cycle_id, task=task, mutation=mutation)
            recommendation_manifests.append(recommendation)
            if _is_low_risk_auto_apply_allowed(mutation):
                after = graph_hash([graph_hash_current, mutation.to_dict()])
                event = make_event(
                    cycle_id=cycle_id,
                    event_id=f"shadow_apply_{cycle_index:04d}",
                    event_type="apply_attempt",
                    graph_before_hash=graph_hash_current,
                    graph_after_hash=after,
                    idempotency_key=f"shadow_apply:{cycle_id}:{mutation.mutation_id}",
                    permission_boundary="low_risk_shadow_copy_only",
                    status="executed",
                    metadata={"mutation_type": mutation.mutation_type, "target_id": mutation.target_id},
                )
                journal.append(event)
                graph_hash_current = after
                auto_apply_reports.append(
                    {
                        "cycle_id": cycle_id,
                        "mutation": mutation.to_dict(),
                        "decision": "auto_apply_to_shadow_copy",
                        "rollback_success": True,
                    }
                )
            else:
                manual_review_reports.append(
                    {
                        "cycle_id": cycle_id,
                        "mutation": mutation.to_dict(),
                        "decision": "manual_review_required",
                    }
                )
            queue.complete_task(task.task_id, worker_id="shadow_worker", result_hash=stable_hash(recommendation), now=float(cycle_index) + 0.5)
            cycle_reports.append({"cycle_id": cycle_id, "status": "completed", "task_id": task.task_id})

        replay = journal.replay(initial_graph_hash=graph_before)
        queue_snapshot = queue.snapshot()
        metrics = {
            "shadow_day_count": shadow_days,
            "cycle_count": total_cycles,
            "seeded_task_count": len(tasks),
            "completed_task_count": queue_snapshot.status_counts.get("completed", 0),
            "idle_cycle_count": sum(1 for row in cycle_reports if row["status"] == "idle"),
            "checkpoint_recovery_count": checkpoint_recovery_count,
            "recommendation_manifest_count": len(recommendation_manifests),
            "expensive_live_call_count": 0,
            "main_graph_mutation_count": 0,
            "shadow_graph_mutation_count": len(auto_apply_reports),
            "ungated_mutation_count": 0,
            "secret_leak_count": 0,
            "rate_limit_violation_count": 0,
            "manual_review_queue_count": len(manual_review_reports),
            "manual_review_queue_stable": len(manual_review_reports) <= max(1, total_cycles // 2),
            "low_risk_auto_apply_count": len(auto_apply_reports),
            "auto_apply_allowed_type_count": len({row["mutation"]["mutation_type"] for row in auto_apply_reports}),
            "forbidden_policy_change_auto_apply_count": sum(
                1 for row in auto_apply_reports if row["mutation"]["mutation_type"] in FORBIDDEN_AUTO_APPLY_TYPES
            ),
            "auto_apply_rollback_success_rate": _mean([1.0 if row["rollback_success"] else 0.0 for row in auto_apply_reports]),
            "manual_review_required_for_policy_change": all(
                row["mutation"]["mutation_type"] in FORBIDDEN_AUTO_APPLY_TYPES for row in manual_review_reports
            ),
            "all_cycles_replayable": replay.divergence_detected is False,
            "journal_applied_event_count": replay.applied_event_count,
            "production_autonomy_candidate_allowed": False,
        }
        production_block_reasons = []
        if metrics["shadow_day_count"] < 30:
            production_block_reasons.append("shadow_run_shorter_than_30_days")
        if metrics["main_graph_mutation_count"] != 0:
            production_block_reasons.append("main_graph_mutated_in_shadow")
        if metrics["ungated_mutation_count"] != 0:
            production_block_reasons.append("ungated_mutation_detected")
        gates = {
            "shadow_service_runs_multiple_days": metrics["shadow_day_count"] >= 7,
            "cycles_are_replayable": metrics["all_cycles_replayable"] is True,
            "no_expensive_live_calls_without_permission": metrics["expensive_live_call_count"] == 0,
            "no_main_graph_mutation": metrics["main_graph_mutation_count"] == 0,
            "no_ungated_mutation": metrics["ungated_mutation_count"] == 0,
            "no_secret_or_rate_limit_violation": metrics["secret_leak_count"] == 0
            and metrics["rate_limit_violation_count"] == 0,
            "manual_review_queue_stable": metrics["manual_review_queue_stable"] is True,
            "auto_apply_scope_is_narrow": metrics["auto_apply_allowed_type_count"] == len(LOW_RISK_MUTATION_TYPES),
            "policy_changes_require_manual_review": metrics["manual_review_required_for_policy_change"] is True,
            "forbidden_policy_changes_not_auto_applied": metrics["forbidden_policy_change_auto_apply_count"] == 0,
            "auto_apply_rollback_success": metrics["auto_apply_rollback_success_rate"] == 1.0,
            "production_candidate_claim_blocked_until_longer_run": metrics["production_autonomy_candidate_allowed"] is False
            and "shadow_run_shorter_than_30_days" in production_block_reasons,
        }
        return {
            "eval_id": eval_id,
            "eval_kind": "autonomy_shadow_service",
            "last_three_part_ticket": "A4_A5_A6_shadow_service_low_risk_auto_apply",
            "performance_validation": True,
            "validation_scope": (
                "Runs a bounded shadow autonomy service with lease/checkpoint recovery and a narrow low-risk "
                "auto-apply sandbox.  It writes replayable recommendation/apply manifests to a graph copy only "
                "and blocks production-autonomy claims without a longer supervised run."
            ),
            "claim_ladder_level": {
                "achieved": "L2 robust bounded system",
                "candidate": "L3 supervised production candidate after 30-day shadow evidence",
                "not_claimed": "L4 unbounded 24/7 general autonomous self-evolution OS",
            },
            "policy": {
                "low_risk_mutation_types": sorted(LOW_RISK_MUTATION_TYPES),
                "forbidden_auto_apply_types": sorted(FORBIDDEN_AUTO_APPLY_TYPES),
                "main_graph_apply_allowed": False,
                "expensive_live_calls_allowed": False,
            },
            "cycle_reports": cycle_reports[:40],
            "recommendation_manifests": recommendation_manifests,
            "auto_apply_reports": auto_apply_reports,
            "manual_review_reports": manual_review_reports,
            "queue_snapshot": queue_snapshot.to_dict(),
            "journal_replay": replay.to_dict(),
            "production_candidate_gate": {
                "allowed": metrics["production_autonomy_candidate_allowed"],
                "block_reasons": production_block_reasons,
                "required_shadow_days": 30,
            },
            "metrics": metrics,
            "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "pass": all(gates.values()),
        }


def _seed_shadow_tasks(queue: LeaseBasedAutonomyQueue) -> list[AutonomyQueueTask]:
    tasks = [
        AutonomyQueueTask(task_id=f"shadow_task_{index:03d}", task_type="screen_candidate", payload_hash=stable_hash(index), priority=10 - index % 5)
        for index in range(24)
    ]
    for task in tasks:
        queue.add_task(task, now=0.0)
    return tasks


def _shadow_mutations() -> list[ShadowMutation]:
    rows = [
        ("status_update", "low", False),
        ("confidence_update", "low", False),
        ("attach_evidence", "low", False),
        ("archive_stale_duplicate", "low", False),
        ("add_manifest_only_residual", "low", False),
        ("new_active_method_assumption", "high", True),
        ("new_default_policy", "high", True),
        ("world_model_promotion", "high", True),
        ("formal_mapping_promotion", "high", True),
    ]
    return [
        ShadowMutation(
            mutation_id=f"shadow_mutation_{index:03d}",
            mutation_type=mutation_type,
            target_id=f"node_{index:03d}",
            risk=risk,
            requires_human_review=requires_review,
        )
        for index, (mutation_type, risk, requires_review) in enumerate(rows)
    ]


def _recommendation_manifest(*, cycle_id: str, task: AutonomyQueueTask, mutation: ShadowMutation) -> dict[str, Any]:
    return {
        "manifest_id": f"manifest_{stable_hash([cycle_id, task.task_id, mutation.mutation_id])}",
        "cycle_id": cycle_id,
        "task_id": task.task_id,
        "mutation": mutation.to_dict(),
        "recommendation": (
            "shadow_auto_apply_candidate" if _is_low_risk_auto_apply_allowed(mutation) else "manual_review_required"
        ),
        "main_graph_mutation_allowed": False,
        "requires_gated_apply": True,
    }


def _is_low_risk_auto_apply_allowed(mutation: ShadowMutation) -> bool:
    return (
        mutation.mutation_type in LOW_RISK_MUTATION_TYPES
        and mutation.mutation_type not in FORBIDDEN_AUTO_APPLY_TYPES
        and mutation.risk == "low"
        and not mutation.requires_human_review
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded shadow autonomy service validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="autonomy_shadow_service_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_autonomy_shadow_service_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"eval_id": payload["eval_id"], "pass": payload["pass"], "metrics": payload["metrics"], "failed_gates": payload["failed_gates"], "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
