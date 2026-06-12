import unittest
from pathlib import Path

from assumption_os.simulator_counterfactual_policy_eval import (
    build_simulator_counterfactual_policy_eval_payload,
)


class SimulatorCounterfactualPolicyEvalTest(unittest.TestCase):
    def test_counterfactual_audit_passes_while_blocking_production_promotion(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_counterfactual_policy_eval_payload(
            root=root,
            eval_id="unit_simulator_counterfactual_policy_eval",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["row_count"], 345)
        self.assertEqual(metrics["valid_row_count"], 345)
        self.assertGreaterEqual(metrics["matched_counterfactual_group_count"], 2)
        self.assertGreaterEqual(metrics["min_arm_count_per_matched_group"], 3)
        self.assertFalse(metrics["production_counterfactual_gate_allowed"])
        self.assertTrue(metrics["exploration_counterfactual_audit_passed"])

    def test_low_coverage_and_weak_estimator_are_explicit_block_reasons(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_counterfactual_policy_eval_payload(
            root=root,
            eval_id="unit_simulator_counterfactual_policy_eval_blocks",
        )
        reasons = set(payload["promotion_decision"]["block_reasons"])
        metrics = payload["metrics"]

        self.assertLess(metrics["matched_action_coverage"], 0.35)
        self.assertFalse(metrics["counterfactual_mae_beats_global_baseline"])
        self.assertIn("matched_action_coverage_below_production_minimum", reasons)
        self.assertIn("leave_one_replicate_mae_does_not_beat_global_baseline", reasons)
        self.assertIn("b3_selector_does_not_agree_with_empirical_best_arm", reasons)

    def test_matched_group_reports_include_best_arm_values(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_counterfactual_policy_eval_payload(
            root=root,
            eval_id="unit_simulator_counterfactual_policy_eval_groups",
        )

        for report in payload["matched_group_reports"]:
            self.assertIn("empirical_best_arm", report)
            self.assertIn("b3_selected_arm", report)
            self.assertIn("arm_mean_utility", report)
            self.assertIn("arm_mean_b3_score", report)
            self.assertGreaterEqual(report["arm_count"], 3)


if __name__ == "__main__":
    unittest.main()
