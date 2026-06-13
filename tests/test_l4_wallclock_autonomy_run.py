import unittest
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


if __name__ == "__main__":
    unittest.main()
