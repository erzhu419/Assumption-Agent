import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_os.l4_wallclock_autonomy_run import build_l4_wallclock_autonomy_run_payload


class L4WallclockAutonomyRunTest(unittest.TestCase):
    def test_short_real_elapsed_run_records_wallclock_cycles(self):
        with TemporaryDirectory() as td:
            payload = build_l4_wallclock_autonomy_run_payload(
                root=Path(td),
                eval_id="unit_l4_wallclock_autonomy_run",
                duration_seconds=0.05,
                cycle_interval_seconds=0.01,
                max_cycles=2,
                seed_count=4,
                inject_faults=True,
            )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreater(metrics["observed_wallclock_seconds"], 0.0)
        self.assertGreaterEqual(metrics["cycle_count"], 1)
        self.assertTrue(metrics["all_cycles_have_required_fields"])
        self.assertTrue(metrics["real_wallclock_smoke_claim_allowed"])
        self.assertEqual(metrics["forbidden_auto_apply_count"], 0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)
        self.assertFalse(metrics["l4_mini_72h_claim_allowed"])

    def test_long_run_mode_uses_bounded_seed_and_heartbeat(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            heartbeat = root / "heartbeat.json"
            payload = build_l4_wallclock_autonomy_run_payload(
                root=root,
                eval_id="unit_l4_wallclock_longmode",
                duration_seconds=5.0,
                cycle_interval_seconds=0.0,
                max_cycles=6,
                seed_count=3,
                heartbeat_path=heartbeat,
                inject_faults=False,
            )
            heartbeat_payload = json.loads(heartbeat.read_text(encoding="utf-8"))

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(heartbeat_payload["status"], "completed")
        self.assertEqual(heartbeat_payload["cycle_count"], payload["metrics"]["cycle_count"])
        self.assertGreaterEqual(payload["metrics"]["cycle_count"], 3)
        self.assertTrue(payload["metrics"]["queue_journal_replayable"])


if __name__ == "__main__":
    unittest.main()
