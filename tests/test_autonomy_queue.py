import tempfile
import unittest
from pathlib import Path

from assumption_os.autonomy_journal import AppendOnlyAutonomyJournal, stable_hash
from assumption_os.autonomy_queue import (
    LeaseBasedAutonomyQueue,
    build_autonomy_queue_lease_payload,
    make_task,
)


class AutonomyQueueTest(unittest.TestCase):
    def test_no_double_lease_and_completed_noop(self):
        with tempfile.TemporaryDirectory() as td:
            queue = LeaseBasedAutonomyQueue(Path(td) / "queue.json")
            self.assertTrue(queue.add_task(make_task("task_a", priority=10), now=1.0).accepted)

            first = queue.lease_next(worker_id="worker_a", now=2.0, lease_ttl=10.0)
            second = queue.lease_next(worker_id="worker_b", now=3.0, lease_ttl=10.0)
            self.assertTrue(first.accepted)
            self.assertEqual(first.task_id, "task_a")
            self.assertFalse(second.accepted)
            self.assertEqual(second.reason, "no_pending_task")

            complete = queue.complete_task(
                "task_a",
                worker_id="worker_a",
                result_hash=stable_hash({"ok": True}),
                now=4.0,
            )
            duplicate_complete = queue.complete_task(
                "task_a",
                worker_id="worker_a",
                result_hash=stable_hash({"ok": True}),
                now=5.0,
            )
            self.assertTrue(complete.accepted)
            self.assertEqual(duplicate_complete.reason, "already_completed_noop")
            self.assertEqual(queue.get_task("task_a").status, "completed")

    def test_worker_crash_releases_lease_and_retry_limit_expires(self):
        with tempfile.TemporaryDirectory() as td:
            queue = LeaseBasedAutonomyQueue(Path(td) / "queue.json")
            queue.add_task(make_task("task_retry", retry_limit=1), now=1.0)

            first = queue.lease_next(worker_id="worker_a", now=2.0, lease_ttl=5.0)
            self.assertTrue(first.accepted)
            expired = queue.expire_leases(now=8.0)
            task = queue.get_task("task_retry")
            self.assertTrue(expired.accepted)
            self.assertEqual(task.status, "pending")
            self.assertEqual(task.retry_count, 1)
            self.assertIsNone(task.lease_owner)

            second = queue.lease_next(worker_id="worker_b", now=9.0, lease_ttl=5.0)
            self.assertTrue(second.accepted)
            queue.expire_leases(now=15.0)
            self.assertEqual(queue.get_task("task_retry").status, "expired")

    def test_blocked_task_not_auto_unblocked_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "queue.json"
            journal = AppendOnlyAutonomyJournal(Path(td) / "journal.jsonl")
            queue = LeaseBasedAutonomyQueue(path, journal=journal, cycle_id="unit_queue")
            queue.add_task(make_task("task_blocked", priority=1), now=1.0)
            self.assertTrue(queue.block_task("task_blocked", reason="needs_review", now=2.0).accepted)

            queue.expire_leases(now=999.0)
            blocked = queue.get_task("task_blocked")
            self.assertEqual(blocked.status, "blocked")
            self.assertEqual(blocked.blocked_reason, "needs_review")

            reloaded = LeaseBasedAutonomyQueue(path)
            self.assertEqual(queue.checkpoint_hash(), reloaded.checkpoint_hash())
            self.assertFalse(journal.replay().divergence_detected)

    def test_stale_completion_rejected_after_lease_expiry(self):
        with tempfile.TemporaryDirectory() as td:
            queue = LeaseBasedAutonomyQueue(Path(td) / "queue.json")
            queue.add_task(make_task("task_stale", retry_limit=1), now=1.0)
            queue.lease_next(worker_id="worker_a", now=2.0, lease_ttl=5.0)

            stale = queue.complete_task(
                "task_stale",
                worker_id="worker_a",
                result_hash=stable_hash({"late": True}),
                now=8.0,
            )
            self.assertFalse(stale.accepted)
            self.assertEqual(stale.reason, "stale_lease_rejected")
            self.assertEqual(queue.get_task("task_stale").status, "pending")

    def test_autonomy_queue_lease_payload_passes(self):
        payload = build_autonomy_queue_lease_payload(eval_id="unit_autonomy_queue_lease")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["double_lease_blocked_for_original_task"])
        self.assertTrue(metrics["worker_crash_releases_lease"])
        self.assertTrue(metrics["same_task_not_executed_twice"])
        self.assertTrue(metrics["blocked_task_not_auto_unblocked"])


if __name__ == "__main__":
    unittest.main()
