import unittest
from pathlib import Path

from assumption_os.paper_fresh_rerun_result_integration import (
    build_paper_fresh_rerun_result_integration_payload,
)


class PaperFreshRerunResultIntegrationTest(unittest.TestCase):
    def test_completed_fresh_run_supports_selective_retention_claim(self):
        payload = build_paper_fresh_rerun_result_integration_payload(
            root=Path("."),
            eval_id="unit_paper_fresh_rerun_result_integration",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["fresh_api_call_count"], 720)
        self.assertEqual(metrics["planned_fresh_api_call_count"], 720)
        self.assertEqual(metrics["live_error_count"], 0)
        self.assertGreaterEqual(metrics["accepted_count"], 1)
        self.assertGreater(metrics["accepted_trigger_ci95_lower"], 0.5)
        self.assertLessEqual(metrics["accepted_control_loss_ci95_upper"], 0.1)
        self.assertTrue(metrics["paper_selective_retention_claim_allowed"])

    def test_unfiltered_generator_overclaim_remains_blocked(self):
        payload = build_paper_fresh_rerun_result_integration_payload(
            root=Path("."),
            eval_id="unit_paper_fresh_rerun_result_integration_boundary",
        )
        metrics = payload["metrics"]

        self.assertFalse(metrics["fresh_live_raw_pass"])
        self.assertIn(
            "all_candidate_trigger_exploration_not_catastrophic",
            metrics["fresh_live_failed_gates"],
        )
        self.assertFalse(metrics["paper_unfiltered_generator_claim_allowed"])
        self.assertLess(metrics["trigger_ci95_upper"], 0.5)


if __name__ == "__main__":
    unittest.main()
