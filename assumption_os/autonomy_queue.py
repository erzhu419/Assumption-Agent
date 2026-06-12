"""Lease-based autonomy queue with checkpoint-safe task ownership.

This queue is the second autonomy substrate after the append-only journal.  It
does not decide which hypothesis is good.  It only makes daemon execution
bounded and auditable: one worker owns one task lease, expired leases requeue
within a retry budget, completed tasks are idempotent no-ops, and blocked tasks
never become runnable without an explicit state change.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .autonomy_journal import AppendOnlyAutonomyJournal, PAPER_DIR, make_event, stable_hash


DEFAULT_OUT = PAPER_DIR / "autonomy_queue_lease_20260612.json"
DEFAULT_LEASE_TTL = 30.0
RUNNABLE_STATUSES = {"pending"}
TERMINAL_STATUSES = {"completed", "failed", "blocked", "expired"}
VALID_STATUSES = RUNNABLE_STATUSES | TERMINAL_STATUSES | {"leased", "deferred"}


@dataclass
class AutonomyQueueTask:
    task_id: str
    task_type: str
    payload_hash: str
    priority: int = 0
    budget_class: str = "default"
    retry_limit: int = 2
    status: str = "pending"
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    retry_count: int = 0
    requires_human_review: bool = False
    requires_fresh_ablation: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    blocked_reason: str | None = None
    deferred_reason: str | None = None
    result_hash: str | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid task status: {self.status}")
        if self.retry_limit < 0:
            raise ValueError("retry_limit must be non-negative")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AutonomyQueueTask":
        return cls(
            task_id=str(row["task_id"]),
            task_type=str(row["task_type"]),
            payload_hash=str(row["payload_hash"]),
            priority=int(row.get("priority", 0)),
            budget_class=str(row.get("budget_class", "default")),
            retry_limit=int(row.get("retry_limit", 2)),
            status=str(row.get("status", "pending")),
            lease_owner=row.get("lease_owner"),
            lease_expires_at=_maybe_float(row.get("lease_expires_at")),
            retry_count=int(row.get("retry_count", 0)),
            requires_human_review=bool(row.get("requires_human_review", False)),
            requires_fresh_ablation=bool(row.get("requires_fresh_ablation", False)),
            created_at=float(row.get("created_at", 0.0)),
            updated_at=float(row.get("updated_at", 0.0)),
            blocked_reason=row.get("blocked_reason"),
            deferred_reason=row.get("deferred_reason"),
            result_hash=row.get("result_hash"),
            last_error=row.get("last_error"),
            metadata=dict(row.get("metadata") or {}),
        )

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class QueueOperationResult:
    accepted: bool
    reason: str
    task_id: str | None = None
    task: AutonomyQueueTask | None = None
    queue_hash_before: str | None = None
    queue_hash_after: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if self.task is not None:
            row["task"] = self.task.to_dict()
        return row


@dataclass(frozen=True)
class QueueSnapshot:
    task_count: int
    status_counts: dict[str, int]
    checkpoint_hash: str
    tasks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseBasedAutonomyQueue:
    def __init__(
        self,
        path: Path | str,
        *,
        journal: AppendOnlyAutonomyJournal | None = None,
        cycle_id: str = "autonomy_queue",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.journal = journal
        self.cycle_id = cycle_id

    def add_task(self, task: AutonomyQueueTask, *, now: float | None = None) -> QueueOperationResult:
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            existing = tasks.get(task.task_id)
            if existing is not None:
                if existing.digest == task.digest:
                    return QueueOperationResult(False, "duplicate_task_noop", task.task_id, existing), False
                return QueueOperationResult(False, "duplicate_task_conflict_blocked", task.task_id, existing), False
            row = AutonomyQueueTask.from_dict(task.to_dict())
            row.created_at = row.created_at or now
            row.updated_at = now
            tasks[row.task_id] = row
            return QueueOperationResult(True, "task_added", row.task_id, row), True

        return self._mutate("add_task", mutate, task_id=task.task_id, now=now)

    def lease_next(
        self,
        *,
        worker_id: str,
        now: float | None = None,
        lease_ttl: float = DEFAULT_LEASE_TTL,
    ) -> QueueOperationResult:
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            expired = _expire_leases_in_state(tasks, now=now)
            candidates = [task for task in tasks.values() if task.status in RUNNABLE_STATUSES]
            if not candidates:
                return QueueOperationResult(
                    False,
                    "no_pending_task",
                    metadata={"expired_task_ids": expired},
                ), bool(expired)
            task = sorted(candidates, key=lambda row: (-row.priority, row.created_at, row.task_id))[0]
            task.status = "leased"
            task.lease_owner = worker_id
            task.lease_expires_at = now + lease_ttl
            task.updated_at = now
            return QueueOperationResult(
                True,
                "task_leased",
                task.task_id,
                AutonomyQueueTask.from_dict(task.to_dict()),
                metadata={"expired_task_ids": expired, "worker_id": worker_id},
            ), True

        return self._mutate("lease_next", mutate, worker_id=worker_id, now=now)

    def expire_leases(self, *, now: float | None = None) -> QueueOperationResult:
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            expired = _expire_leases_in_state(tasks, now=now)
            return QueueOperationResult(
                bool(expired),
                "expired_leases_processed" if expired else "no_expired_leases",
                metadata={"expired_task_ids": expired},
            ), bool(expired)

        return self._mutate("expire_leases", mutate, now=now)

    def complete_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        result_hash: str,
        now: float | None = None,
    ) -> QueueOperationResult:
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            task = tasks.get(task_id)
            if task is None:
                return QueueOperationResult(False, "task_not_found", task_id), False
            if task.status == "completed":
                return QueueOperationResult(False, "already_completed_noop", task_id, task), False
            if task.status != "leased":
                return QueueOperationResult(False, f"task_not_leased:{task.status}", task_id, task), False
            if task.lease_owner != worker_id:
                return QueueOperationResult(False, "lease_owner_mismatch", task_id, task), False
            if task.lease_expires_at is not None and task.lease_expires_at <= now:
                _expire_one_task(task, now=now)
                return QueueOperationResult(False, "stale_lease_rejected", task_id, task), True
            task.status = "completed"
            task.result_hash = result_hash
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = now
            return QueueOperationResult(True, "task_completed", task_id, task), True

        return self._mutate("complete_task", mutate, task_id=task_id, worker_id=worker_id, now=now)

    def fail_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        error: str,
        now: float | None = None,
    ) -> QueueOperationResult:
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            task = tasks.get(task_id)
            if task is None:
                return QueueOperationResult(False, "task_not_found", task_id), False
            if task.status != "leased":
                return QueueOperationResult(False, f"task_not_leased:{task.status}", task_id, task), False
            if task.lease_owner != worker_id:
                return QueueOperationResult(False, "lease_owner_mismatch", task_id, task), False
            task.last_error = error
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = now
            if task.retry_count < task.retry_limit:
                task.retry_count += 1
                task.status = "pending"
                reason = "task_failed_requeued"
            else:
                task.status = "failed"
                reason = "task_failed_terminal"
            return QueueOperationResult(True, reason, task_id, task), True

        return self._mutate("fail_task", mutate, task_id=task_id, worker_id=worker_id, now=now)

    def block_task(self, task_id: str, *, reason: str, now: float | None = None) -> QueueOperationResult:
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            task = tasks.get(task_id)
            if task is None:
                return QueueOperationResult(False, "task_not_found", task_id), False
            if task.status == "completed":
                return QueueOperationResult(False, "completed_task_not_blocked", task_id, task), False
            if task.status == "blocked" and task.blocked_reason == reason:
                return QueueOperationResult(False, "already_blocked_noop", task_id, task), False
            task.status = "blocked"
            task.blocked_reason = reason
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = now
            return QueueOperationResult(True, "task_blocked", task_id, task), True

        return self._mutate("block_task", mutate, task_id=task_id, now=now)

    def defer_task(self, task_id: str, *, reason: str, now: float | None = None) -> QueueOperationResult:
        now = _now(now)

        def mutate(tasks: dict[str, AutonomyQueueTask]) -> tuple[QueueOperationResult, bool]:
            task = tasks.get(task_id)
            if task is None:
                return QueueOperationResult(False, "task_not_found", task_id), False
            if task.status in TERMINAL_STATUSES:
                return QueueOperationResult(False, f"terminal_task_not_deferred:{task.status}", task_id, task), False
            task.status = "deferred"
            task.deferred_reason = reason
            task.lease_owner = None
            task.lease_expires_at = None
            task.updated_at = now
            return QueueOperationResult(True, "task_deferred", task_id, task), True

        return self._mutate("defer_task", mutate, task_id=task_id, now=now)

    def snapshot(self) -> QueueSnapshot:
        with self._locked():
            tasks = self._read_tasks_unlocked()
            return _snapshot(tasks)

    def get_task(self, task_id: str) -> AutonomyQueueTask | None:
        with self._locked():
            return self._read_tasks_unlocked().get(task_id)

    def checkpoint_hash(self) -> str:
        return self.snapshot().checkpoint_hash

    def _mutate(
        self,
        operation: str,
        mutate: Callable[[dict[str, AutonomyQueueTask]], tuple[QueueOperationResult, bool]],
        *,
        task_id: str | None = None,
        worker_id: str | None = None,
        now: float,
    ) -> QueueOperationResult:
        with self._locked():
            tasks = self._read_tasks_unlocked()
            before_hash = _queue_hash(tasks)
            result, changed = mutate(tasks)
            after_hash = _queue_hash(tasks)
            result = QueueOperationResult(
                accepted=result.accepted,
                reason=result.reason,
                task_id=result.task_id,
                task=result.task,
                queue_hash_before=before_hash,
                queue_hash_after=after_hash,
                metadata=result.metadata,
            )
            if changed:
                self._write_tasks_unlocked(tasks)
                self._append_journal_event(
                    operation=operation,
                    task_id=task_id or result.task_id,
                    worker_id=worker_id,
                    before_hash=before_hash,
                    after_hash=after_hash,
                    reason=result.reason,
                    now=now,
                )
            return result

    @contextlib.contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_tasks_unlocked(self) -> dict[str, AutonomyQueueTask]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        rows = raw.get("tasks", raw if isinstance(raw, list) else [])
        tasks = [AutonomyQueueTask.from_dict(row) for row in rows]
        return {task.task_id: task for task in tasks}

    def _write_tasks_unlocked(self, tasks: dict[str, AutonomyQueueTask]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "lease_based_autonomy_queue_v1",
            "checkpoint_hash": _queue_hash(tasks),
            "tasks": [task.to_dict() for task in sorted(tasks.values(), key=lambda row: row.task_id)],
        }
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=self.path.name + ".tmp.",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, self.path)

    def _append_journal_event(
        self,
        *,
        operation: str,
        task_id: str | None,
        worker_id: str | None,
        before_hash: str,
        after_hash: str,
        reason: str,
        now: float,
    ) -> None:
        if self.journal is None:
            return
        event_id = stable_hash(
            {
                "queue_event": operation,
                "task_id": task_id,
                "worker_id": worker_id,
                "before": before_hash,
                "after": after_hash,
                "now": now,
            }
        )
        self.journal.append(
            make_event(
                cycle_id=self.cycle_id,
                event_id=f"queue_{event_id}",
                event_type=f"queue_{operation}",
                graph_before_hash=before_hash,
                graph_after_hash=after_hash,
                idempotency_key=f"queue:{operation}:{task_id or 'none'}:{after_hash}",
                status="executed",
                metadata={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "reason": reason,
                    "queue_path": str(self.path),
                },
            )
        )


def make_task(
    task_id: str,
    *,
    task_type: str = "hypothesis_ablation",
    payload: Any | None = None,
    priority: int = 0,
    retry_limit: int = 2,
    budget_class: str = "default",
    requires_human_review: bool = False,
    requires_fresh_ablation: bool = False,
    created_at: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> AutonomyQueueTask:
    return AutonomyQueueTask(
        task_id=task_id,
        task_type=task_type,
        payload_hash=stable_hash(payload if payload is not None else {"task_id": task_id}),
        priority=priority,
        budget_class=budget_class,
        retry_limit=retry_limit,
        requires_human_review=requires_human_review,
        requires_fresh_ablation=requires_fresh_ablation,
        created_at=created_at,
        metadata=metadata or {},
    )


def build_autonomy_queue_lease_payload(
    *,
    eval_id: str = "autonomy_queue_lease_20260612",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="autonomy_queue_") as td:
        root = Path(td)
        journal = AppendOnlyAutonomyJournal(root / "queue_journal.jsonl")
        queue = LeaseBasedAutonomyQueue(root / "queue.json", journal=journal, cycle_id=eval_id)
        add_results = [
            queue.add_task(make_task("task_crash", priority=10, retry_limit=2, created_at=1.0), now=1.0),
            queue.add_task(make_task("task_done", priority=9, retry_limit=1, created_at=2.0), now=2.0),
            queue.add_task(make_task("task_expire", priority=8, retry_limit=0, created_at=3.0), now=3.0),
            queue.add_task(make_task("task_blocked", priority=7, retry_limit=2, created_at=4.0), now=4.0),
        ]

        crash_lease = queue.lease_next(worker_id="worker_a", now=10.0, lease_ttl=5.0)
        double_lease_attempt = queue.lease_next(worker_id="worker_b", now=11.0, lease_ttl=5.0)
        crash_expire = queue.expire_leases(now=16.0)
        crash_after_expire = queue.get_task("task_crash")

        done_lease = queue.lease_next(worker_id="worker_b", now=17.0, lease_ttl=5.0)
        done_complete = queue.complete_task(
            "task_crash",
            worker_id="worker_b",
            result_hash=stable_hash({"result": "accepted"}),
            now=18.0,
        )
        done_second_complete = queue.complete_task(
            "task_crash",
            worker_id="worker_b",
            result_hash=stable_hash({"result": "accepted"}),
            now=19.0,
        )

        expire_lease = queue.lease_next(worker_id="worker_c", now=20.0, lease_ttl=2.0)
        expire_once = queue.expire_leases(now=23.0)
        expired_task = queue.get_task("task_done")

        queue.block_task("task_blocked", reason="needs_human_review", now=24.0)
        queue.expire_leases(now=999.0)
        blocked_after_time = queue.get_task("task_blocked")

        queue_reloaded = LeaseBasedAutonomyQueue(root / "queue.json", journal=journal, cycle_id=eval_id)
        snapshot = queue.snapshot()
        reloaded_snapshot = queue_reloaded.snapshot()
        journal_replay = journal.replay()

    metrics = {
        "task_count": snapshot.task_count,
        "add_accepted_count": sum(1 for result in add_results if result.accepted),
        "double_lease_attempt_task_id": double_lease_attempt.task_id,
        "double_lease_blocked_for_original_task": double_lease_attempt.task_id != "task_crash",
        "worker_crash_releases_lease": (
            crash_after_expire is not None
            and crash_after_expire.status == "pending"
            and crash_after_expire.retry_count == 1
            and crash_after_expire.lease_owner is None
        ),
        "expired_task_requeues": (
            crash_expire.accepted
            and crash_after_expire is not None
            and crash_after_expire.status == "pending"
        ),
        "same_task_not_executed_twice": (
            done_lease.accepted
            and done_complete.accepted
            and done_second_complete.reason == "already_completed_noop"
        ),
        "retry_limit_expires_terminal": (
            expire_lease.accepted
            and expire_once.accepted
            and expired_task is not None
            and expired_task.status == "expired"
        ),
        "blocked_task_not_auto_unblocked": blocked_after_time is not None and blocked_after_time.status == "blocked",
        "checkpoint_reload_same_state": snapshot.checkpoint_hash == reloaded_snapshot.checkpoint_hash,
        "journal_event_count": journal_replay.event_count,
        "journal_replay_divergence_detected": journal_replay.divergence_detected,
    }
    gates = {
        "all_tasks_added": metrics["add_accepted_count"] == 4,
        "no_double_lease_original_task": metrics["double_lease_blocked_for_original_task"] is True,
        "worker_crash_releases_lease": metrics["worker_crash_releases_lease"] is True,
        "expired_task_requeues": metrics["expired_task_requeues"] is True,
        "same_task_not_executed_twice": metrics["same_task_not_executed_twice"] is True,
        "retry_limit_expires_terminal": metrics["retry_limit_expires_terminal"] is True,
        "blocked_task_not_auto_unblocked": metrics["blocked_task_not_auto_unblocked"] is True,
        "checkpoint_reload_same_state": metrics["checkpoint_reload_same_state"] is True,
        "journal_replay_clean": metrics["journal_event_count"] >= 8
        and metrics["journal_replay_divergence_detected"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "autonomy_queue_lease",
        "last_three_part_ticket": "A2_lease_based_autonomy_queue",
        "performance_validation": True,
        "validation_scope": (
            "Validates lease-based queue semantics needed for supervised autonomy: no double lease, worker "
            "crash lease expiry, retry-bounded requeue/terminal expiry, idempotent completion, blocked task "
            "isolation, checkpoint reload, and clean journal replay."
        ),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "snapshot": snapshot.to_dict(),
    }


def _expire_leases_in_state(tasks: dict[str, AutonomyQueueTask], *, now: float) -> list[str]:
    expired: list[str] = []
    for task in sorted(tasks.values(), key=lambda row: row.task_id):
        if task.status != "leased":
            continue
        if task.lease_expires_at is None or task.lease_expires_at > now:
            continue
        _expire_one_task(task, now=now)
        expired.append(task.task_id)
    return expired


def _expire_one_task(task: AutonomyQueueTask, *, now: float) -> None:
    task.lease_owner = None
    task.lease_expires_at = None
    task.updated_at = now
    if task.retry_count < task.retry_limit:
        task.retry_count += 1
        task.status = "pending"
    else:
        task.status = "expired"


def _snapshot(tasks: dict[str, AutonomyQueueTask]) -> QueueSnapshot:
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for task in tasks.values():
        counts[task.status] = counts.get(task.status, 0) + 1
    return QueueSnapshot(
        task_count=len(tasks),
        status_counts=counts,
        checkpoint_hash=_queue_hash(tasks),
        tasks=[task.to_dict() for task in sorted(tasks.values(), key=lambda row: row.task_id)],
    )


def _queue_hash(tasks: dict[str, AutonomyQueueTask]) -> str:
    return stable_hash([task.to_dict() for task in sorted(tasks.values(), key=lambda row: row.task_id)])


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _now(value: float | None) -> float:
    return float(time.time() if value is None else value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build autonomy queue lease validation artifact.")
    parser.add_argument("--eval-id", default="autonomy_queue_lease_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_autonomy_queue_lease_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
