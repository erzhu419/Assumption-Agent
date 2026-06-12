import unittest
from pathlib import Path

from assumption_os.simulator_uncertainty import (
    ALLOWED_ACTIONS,
    FORBIDDEN_ACTIONS,
    build_simulator_uncertainty_payload,
)


class SimulatorUncertaintyTest(unittest.TestCase):
    def test_uncertainty_payload_passes_performance_gates(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_uncertainty_payload(
            root=root,
            eval_id="unit_simulator_uncertainty",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["row_count"], 531)
        self.assertEqual(metrics["valid_row_count"], 531)
        self.assertLess(
            metrics["leave_pattern_uncertainty_brier_with_abstain_as_half"],
            metrics["leave_pattern_base_rate_brier_with_abstain_as_half"],
        )
        self.assertLessEqual(metrics["leave_pattern_uncertainty_ece"], 0.12)
        self.assertLessEqual(metrics["accepted_candidate_block_rate"], 0.02)

    def test_decisions_have_required_uncertainty_fields_and_allowed_actions(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_uncertainty_payload(
            root=root,
            eval_id="unit_simulator_uncertainty_fields",
        )

        for decision in payload["leave_pattern_evaluation"]["decisions"]:
            self.assertIn("prediction", decision)
            self.assertIn("confidence_interval", decision)
            self.assertIn("calibration_bin", decision)
            self.assertIn("abstain_reason", decision)
            self.assertIn("required_verifier_tier", decision)
            self.assertIn(decision["action"], ALLOWED_ACTIONS)
            self.assertNotIn(decision["action"], FORBIDDEN_ACTIONS)

    def test_low_support_probe_abstains_to_live_validation(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_uncertainty_payload(
            root=root,
            eval_id="unit_simulator_uncertainty_low_support",
        )
        probe = payload["low_support_stress_probe"]

        self.assertEqual(probe["action"], "abstain_to_live_validation")
        self.assertEqual(probe["abstain_reason"], "low_support")
        self.assertEqual(probe["required_verifier_tier"], "tier3_live_validation_or_human_review")


if __name__ == "__main__":
    unittest.main()
