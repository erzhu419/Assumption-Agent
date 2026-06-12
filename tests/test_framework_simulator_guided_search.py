import unittest
from pathlib import Path

from assumption_os.framework_simulator_guided_search import (
    ALLOWED_SIMULATOR_ACTIONS,
    build_framework_simulator_guided_search_payload,
)


class FrameworkSimulatorGuidedSearchTest(unittest.TestCase):
    def test_simulator_reduces_fresh_tests_without_blocking_true_positive_frameworks(self):
        payload = build_framework_simulator_guided_search_payload(
            root=Path("."),
            eval_id="unit_framework_simulator_guided_search",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["fresh_test_reduction_rate"], 0.40)
        self.assertEqual(metrics["true_positive_block_count"], 0)
        self.assertGreaterEqual(metrics["true_positive_selected_count"], 12)
        self.assertGreater(metrics["baseline_fresh_test_count"], metrics["selected_fresh_test_count"])

    def test_simulator_is_limited_to_ranking_selection_and_verifier_routing(self):
        payload = build_framework_simulator_guided_search_payload(
            root=Path("."),
            eval_id="unit_framework_simulator_guided_search_claim_boundary",
        )
        metrics = payload["metrics"]

        self.assertTrue(set(metrics["allowed_actions_used"]).issubset(ALLOWED_SIMULATOR_ACTIONS))
        self.assertEqual(metrics["blocked_action_count"], 0)
        self.assertEqual(metrics["direct_promotion_count"], 0)
        self.assertEqual(metrics["live_replacement_count"], 0)
        self.assertEqual(metrics["review_replacement_count"], 0)
        self.assertTrue(metrics["production_router_claim_allowed"])
        self.assertFalse(metrics["simulator_replacement_claim_allowed"])

    def test_rejected_risk_and_simulator_defects_are_recorded(self):
        payload = build_framework_simulator_guided_search_payload(
            root=Path("."),
            eval_id="unit_framework_simulator_guided_search_defects",
        )
        metrics = payload["metrics"]

        self.assertEqual(metrics["rejected_high_risk_recall"], 1.0)
        self.assertGreaterEqual(metrics["rejected_vs_retained_risk_margin"], 0.25)
        self.assertGreaterEqual(metrics["simulator_defect_residual_count"], 1)
        self.assertEqual(metrics["simulator_defect_next_round_intake_rate"], 1.0)
        residual = payload["simulator_defect_residuals"][0]
        self.assertEqual(residual["residual_type"], "SimulatorDefect")
        self.assertTrue(residual["next_round_intake"])


if __name__ == "__main__":
    unittest.main()
