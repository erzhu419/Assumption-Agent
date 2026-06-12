import tempfile
import unittest
from pathlib import Path

from assumption_os.autonomy_journal import (
    AppendOnlyAutonomyJournal,
    build_autonomy_journal_replay_payload,
    graph_hash,
    make_event,
    replay_events,
)


class AutonomyJournalTest(unittest.TestCase):
    def test_append_replay_and_idempotency(self):
        with tempfile.TemporaryDirectory() as td:
            journal = AppendOnlyAutonomyJournal(Path(td) / "journal.jsonl")
            genesis = graph_hash("genesis")
            after_a = graph_hash("after_a")
            after_b = graph_hash("after_b")
            first = make_event(
                cycle_id="cycle_001",
                event_id="event_a",
                event_type="queue_read",
                graph_before_hash=genesis,
                graph_after_hash=after_a,
                idempotency_key="cycle_001:queue_read",
                status="executed",
            )
            second = make_event(
                cycle_id="cycle_001",
                event_id="event_b",
                event_type="apply_attempt",
                graph_before_hash=after_a,
                graph_after_hash=after_b,
                idempotency_key="cycle_001:apply",
                status="executed",
            )

            self.assertTrue(journal.append(first).accepted)
            self.assertTrue(journal.append(second).accepted)
            self.assertFalse(journal.append(first).accepted)

            conflict = make_event(
                cycle_id="cycle_001",
                event_id="event_a_retry",
                event_type="queue_read",
                graph_before_hash=genesis,
                graph_after_hash=graph_hash("different_after"),
                idempotency_key="cycle_001:queue_read",
                status="executed",
            )
            self.assertEqual(journal.append(conflict).reason, "idempotency_key_conflict_blocked")

            replay = journal.replay(initial_graph_hash=genesis)
            self.assertFalse(replay.divergence_detected)
            self.assertEqual(replay.applied_event_count, 2)
            self.assertEqual(replay.final_graph_hash, after_b)

    def test_crash_recovery_and_hash_divergence_detection(self):
        genesis = graph_hash("genesis")
        after_a = graph_hash("after_a")
        after_recovery = graph_hash("after_recovery")
        events = [
            make_event(
                cycle_id="cycle_001",
                event_id="event_a",
                event_type="queue_read",
                graph_before_hash=genesis,
                graph_after_hash=after_a,
                idempotency_key="cycle_001:queue_read",
                status="executed",
            ),
            make_event(
                cycle_id="cycle_002",
                event_id="event_crash",
                event_type="apply_attempt",
                graph_before_hash=after_a,
                graph_after_hash=after_a,
                idempotency_key="cycle_002:apply",
                status="failed",
            ),
            make_event(
                cycle_id="cycle_002",
                event_id="event_recovery",
                event_type="recovery",
                graph_before_hash=after_a,
                graph_after_hash=after_recovery,
                idempotency_key="cycle_002:recovery",
                status="recovered",
            ),
        ]
        replay = replay_events(events, initial_graph_hash=genesis)
        self.assertFalse(replay.divergence_detected)
        self.assertEqual(replay.final_graph_hash, after_recovery)

        divergent = replay_events(
            events
            + [
                make_event(
                    cycle_id="cycle_003",
                    event_id="event_bad",
                    event_type="apply_attempt",
                    graph_before_hash=graph_hash("wrong_before"),
                    graph_after_hash=graph_hash("bad_after"),
                    idempotency_key="cycle_003:bad",
                    status="executed",
                )
            ],
            initial_graph_hash=genesis,
        )
        self.assertTrue(divergent.divergence_detected)
        self.assertEqual(divergent.divergence_count, 1)

    def test_autonomy_journal_replay_payload_passes(self):
        payload = build_autonomy_journal_replay_payload(eval_id="unit_autonomy_journal_replay")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["replay_same_journal_same_state"])
        self.assertTrue(metrics["duplicate_event_no_double_apply"])
        self.assertTrue(metrics["crash_mid_cycle_recoverable"])
        self.assertTrue(metrics["graph_hash_divergence_detected"])


if __name__ == "__main__":
    unittest.main()
