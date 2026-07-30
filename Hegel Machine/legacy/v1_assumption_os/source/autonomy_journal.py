"""Append-only autonomy journal with deterministic replay.

The journal is intentionally small: it records cycle events, idempotency keys,
graph hashes, permission boundaries, and status transitions.  It does not make
policy decisions.  Its job is to make autonomy runs replayable and crash-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "autonomy_journal_replay_20260612.json"
MUTATING_STATUSES = {"executed", "completed", "recovered"}


@dataclass(frozen=True)
class AutonomyJournalEvent:
    cycle_id: str
    event_id: str
    event_type: str
    input_hash: str
    output_hash: str
    graph_before_hash: str
    graph_after_hash: str
    idempotency_key: str
    permission_boundary: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AutonomyJournalEvent":
        return cls(
            cycle_id=str(row["cycle_id"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            input_hash=str(row["input_hash"]),
            output_hash=str(row["output_hash"]),
            graph_before_hash=str(row["graph_before_hash"]),
            graph_after_hash=str(row["graph_after_hash"]),
            idempotency_key=str(row["idempotency_key"]),
            permission_boundary=str(row["permission_boundary"]),
            status=str(row["status"]),
            metadata=dict(row.get("metadata") or {}),
        )

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class AppendResult:
    accepted: bool
    reason: str
    event_id: str
    idempotency_key: str


@dataclass(frozen=True)
class ReplayReport:
    event_count: int
    applied_event_count: int
    duplicate_event_count: int
    duplicate_idempotency_count: int
    divergence_detected: bool
    divergence_count: int
    divergences: list[dict[str, Any]]
    final_graph_hash: str | None
    applied_event_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AppendOnlyAutonomyJournal:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AutonomyJournalEvent) -> AppendResult:
        event_by_id: dict[str, AutonomyJournalEvent] = {}
        event_by_key: dict[str, AutonomyJournalEvent] = {}
        for existing in self.read_events():
            event_by_id[existing.event_id] = existing
            event_by_key[existing.idempotency_key] = existing

        if event.event_id in event_by_id:
            return AppendResult(False, "duplicate_event_id_noop", event.event_id, event.idempotency_key)

        if event.idempotency_key in event_by_key:
            existing = event_by_key[event.idempotency_key]
            if existing.digest == event.digest:
                reason = "duplicate_idempotency_key_noop"
            else:
                reason = "idempotency_key_conflict_blocked"
            return AppendResult(False, reason, event.event_id, event.idempotency_key)

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return AppendResult(True, "appended", event.event_id, event.idempotency_key)

    def read_events(self) -> list[AutonomyJournalEvent]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(AutonomyJournalEvent.from_dict(json.loads(line)))
        return events

    def replay(self, *, initial_graph_hash: str | None = None) -> ReplayReport:
        return replay_events(self.read_events(), initial_graph_hash=initial_graph_hash)


def replay_events(
    events: list[AutonomyJournalEvent],
    *,
    initial_graph_hash: str | None = None,
) -> ReplayReport:
    current = initial_graph_hash
    seen_event_ids: set[str] = set()
    seen_idempotency_keys: set[str] = set()
    applied_ids: list[str] = []
    duplicate_events = 0
    duplicate_keys = 0
    divergences: list[dict[str, Any]] = []

    for event in events:
        if event.event_id in seen_event_ids:
            duplicate_events += 1
            continue
        seen_event_ids.add(event.event_id)

        if event.idempotency_key in seen_idempotency_keys:
            duplicate_keys += 1
            continue
        seen_idempotency_keys.add(event.idempotency_key)

        if current is None:
            current = event.graph_before_hash

        if event.graph_before_hash != current:
            divergences.append(
                {
                    "event_id": event.event_id,
                    "expected_graph_before_hash": current,
                    "observed_graph_before_hash": event.graph_before_hash,
                }
            )
            continue

        if event.status in MUTATING_STATUSES:
            current = event.graph_after_hash
            applied_ids.append(event.event_id)

    return ReplayReport(
        event_count=len(events),
        applied_event_count=len(applied_ids),
        duplicate_event_count=duplicate_events,
        duplicate_idempotency_count=duplicate_keys,
        divergence_detected=bool(divergences),
        divergence_count=len(divergences),
        divergences=divergences,
        final_graph_hash=current,
        applied_event_ids=applied_ids,
    )


def build_autonomy_journal_replay_payload(
    *,
    eval_id: str = "autonomy_journal_replay_20260612",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="autonomy_journal_") as td:
        journal = AppendOnlyAutonomyJournal(Path(td) / "journal.jsonl")
        genesis = graph_hash("genesis")
        after_queue = graph_hash("after_queue")
        after_apply = graph_hash("after_apply")
        after_recovery = graph_hash("after_recovery")

        first = make_event(
            cycle_id="cycle_001",
            event_id="event_queue_read",
            event_type="queue_read",
            graph_before_hash=genesis,
            graph_after_hash=after_queue,
            idempotency_key="queue_read:cycle_001",
            status="executed",
        )
        second = make_event(
            cycle_id="cycle_001",
            event_id="event_apply",
            event_type="apply_attempt",
            graph_before_hash=after_queue,
            graph_after_hash=after_apply,
            idempotency_key="apply:candidate_001",
            status="executed",
        )
        crash = make_event(
            cycle_id="cycle_002",
            event_id="event_crash",
            event_type="apply_attempt",
            graph_before_hash=after_apply,
            graph_after_hash=after_apply,
            idempotency_key="apply:candidate_002",
            status="failed",
        )
        recovery = make_event(
            cycle_id="cycle_002",
            event_id="event_recovery",
            event_type="recovery",
            graph_before_hash=after_apply,
            graph_after_hash=after_recovery,
            idempotency_key="recovery:candidate_002",
            status="recovered",
        )

        append_results = [
            journal.append(first),
            journal.append(second),
            journal.append(first),
            journal.append(_replace_event_id(first, "event_queue_read_retry")),
            journal.append(crash),
            journal.append(recovery),
        ]
        replay = journal.replay(initial_graph_hash=genesis)
        replay_again = journal.replay(initial_graph_hash=genesis)
        divergent = replay_events(
            [
                first,
                make_event(
                    cycle_id="cycle_bad",
                    event_id="event_bad_hash",
                    event_type="apply_attempt",
                    graph_before_hash=graph_hash("wrong_before"),
                    graph_after_hash=graph_hash("bad_after"),
                    idempotency_key="bad:hash",
                    status="executed",
                ),
            ],
            initial_graph_hash=genesis,
        )

    metrics = {
        "append_attempt_count": len(append_results),
        "append_accepted_count": sum(1 for row in append_results if row.accepted),
        "duplicate_event_noop_count": sum(1 for row in append_results if row.reason == "duplicate_event_id_noop"),
        "idempotency_conflict_blocked_count": sum(
            1 for row in append_results if row.reason == "idempotency_key_conflict_blocked"
        ),
        "replay_event_count": replay.event_count,
        "replay_applied_event_count": replay.applied_event_count,
        "replay_final_graph_hash": replay.final_graph_hash,
        "replay_again_final_graph_hash": replay_again.final_graph_hash,
        "replay_same_journal_same_state": replay.final_graph_hash == replay_again.final_graph_hash,
        "duplicate_event_no_double_apply": replay.duplicate_event_count == 0
        and replay.applied_event_ids.count("event_queue_read") == 1,
        "crash_mid_cycle_recoverable": replay.final_graph_hash == graph_hash("after_recovery"),
        "graph_hash_divergence_detected": divergent.divergence_detected,
    }
    gates = {
        "append_only_accepts_unique_events": metrics["append_accepted_count"] == 4,
        "duplicate_event_noop": metrics["duplicate_event_noop_count"] == 1,
        "idempotency_conflict_blocked": metrics["idempotency_conflict_blocked_count"] == 1,
        "replay_same_journal_same_state": metrics["replay_same_journal_same_state"] is True,
        "duplicate_event_no_double_apply": metrics["duplicate_event_no_double_apply"] is True,
        "crash_mid_cycle_recoverable": metrics["crash_mid_cycle_recoverable"] is True,
        "graph_hash_divergence_detected": metrics["graph_hash_divergence_detected"] is True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "autonomy_journal_replay",
        "last_three_part_ticket": "A1_autonomy_journal",
        "performance_validation": True,
        "validation_scope": (
            "Validates append-only autonomy journal semantics: idempotent event append, duplicate no-op, "
            "crash recovery through replay, and graph-hash divergence detection."
        ),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def make_event(
    *,
    cycle_id: str,
    event_id: str,
    event_type: str,
    graph_before_hash: str,
    graph_after_hash: str,
    idempotency_key: str,
    status: str,
    permission_boundary: str = "gated_apply_required",
    metadata: dict[str, Any] | None = None,
) -> AutonomyJournalEvent:
    return AutonomyJournalEvent(
        cycle_id=cycle_id,
        event_id=event_id,
        event_type=event_type,
        input_hash=stable_hash({"event_id": event_id, "input": "redacted"}),
        output_hash=stable_hash({"event_id": event_id, "output": "redacted"}),
        graph_before_hash=graph_before_hash,
        graph_after_hash=graph_after_hash,
        idempotency_key=idempotency_key,
        permission_boundary=permission_boundary,
        status=status,
        metadata=metadata or {},
    )


def graph_hash(value: str) -> str:
    return stable_hash({"graph": value})


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _replace_event_id(event: AutonomyJournalEvent, event_id: str) -> AutonomyJournalEvent:
    row = event.to_dict()
    row["event_id"] = event_id
    return AutonomyJournalEvent.from_dict(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build autonomy journal replay validation artifact.")
    parser.add_argument("--eval-id", default="autonomy_journal_replay_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_autonomy_journal_replay_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
