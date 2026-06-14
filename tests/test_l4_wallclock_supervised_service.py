import unittest
from pathlib import Path

from assumption_os.l4_wallclock_supervised_service import (
    build_l4_wallclock_supervised_service_payload,
)


class L4WallclockSupervisedServiceTest(unittest.TestCase):
    def test_service_contract_passes_without_fabricating_wallclock_claims(self):
        payload = build_l4_wallclock_supervised_service_payload(
            root=Path("."),
            eval_id="unit_l4_wallclock_supervised_service",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["wallclock_service_preflight_claim_allowed"])
        self.assertEqual(metrics["service_level_count"], 4)
        self.assertGreaterEqual(metrics["source_cycle_count"], 720)
        self.assertGreaterEqual(metrics["observed_wallclock_seconds"], 0.0)
        self.assertEqual(
            metrics["twenty_four_hour_cumulative_claim_allowed"],
            metrics["observed_wallclock_seconds"] >= 24 * 3600 and metrics["observed_uptime"] >= 0.95,
        )
        self.assertFalse(metrics["twenty_four_hour_continuous_claim_allowed"])
        self.assertFalse(metrics["l4a_wallclock_completed_claim_allowed"])
        self.assertFalse(metrics["thirty_day_wallclock_claim_allowed"])
        self.assertEqual(metrics["ungated_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
