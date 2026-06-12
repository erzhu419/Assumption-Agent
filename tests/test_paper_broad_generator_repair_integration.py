import unittest
from pathlib import Path

from assumption_os.paper_broad_generator_repair_integration import (
    build_paper_broad_generator_repair_integration_payload,
)


class PaperBroadGeneratorRepairIntegrationTest(unittest.TestCase):
    def test_evidence_calibrated_repair_passes_fresh_720(self):
        payload = build_paper_broad_generator_repair_integration_payload(
            root=Path("."),
            eval_id="unit_paper_broad_generator_repair_integration",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertFalse(metrics["original_raw_pass"])
        self.assertTrue(metrics["repair_v2_live_pass"])
        self.assertEqual(metrics["repair_v2_fresh_api_call_count"], 720)
        self.assertEqual(metrics["repair_v2_live_error_count"], 0)
        self.assertGreaterEqual(metrics["trigger_utility_delta_vs_original"], 0.10)
        self.assertGreater(
            metrics["repair_v2_trigger_ci95_lower"],
            metrics["original_trigger_ci95_upper"],
        )

    def test_repair_keeps_claim_boundary_against_raw_frontier(self):
        payload = build_paper_broad_generator_repair_integration_payload(
            root=Path("."),
            eval_id="unit_paper_broad_generator_repair_claim_boundary",
        )
        metrics = payload["metrics"]
        raw_boundary = next(
            row for row in payload["claim_boundaries"]
            if row["claim_id"] == "raw_unfiltered_generator_frontier"
        )
        calibrated_boundary = next(
            row for row in payload["claim_boundaries"]
            if row["claim_id"] == "evidence_calibrated_broad_frontier"
        )

        self.assertFalse(raw_boundary["allowed"])
        self.assertTrue(calibrated_boundary["allowed"])
        self.assertLessEqual(metrics["selected_candidate_count_delta_vs_original"], -40)
        self.assertEqual(
            metrics["repair_v2_fresh_api_call_count"],
            metrics["original_fresh_api_call_count"],
        )


if __name__ == "__main__":
    unittest.main()
