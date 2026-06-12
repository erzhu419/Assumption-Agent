import unittest
from pathlib import Path

from assumption_os.simulator_eval_splits import (
    build_simulator_eval_splits_payload,
)


class SimulatorEvalSplitsTest(unittest.TestCase):
    def test_split_evaluation_payload_passes_and_blocks_raw_promotion(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_eval_splits_payload(
            root=root,
            eval_id="unit_simulator_eval_splits",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["row_count"], 345)
        self.assertEqual(metrics["valid_row_count"], 345)
        self.assertGreaterEqual(metrics["leave_one_out_group_count"], 300)
        self.assertGreaterEqual(metrics["leave_domain_out_group_count"], 5)
        self.assertGreaterEqual(metrics["leave_pattern_out_group_count"], 5)
        self.assertFalse(metrics["raw_predictor_promotion_allowed"])
        self.assertFalse(metrics["production_simulator_replacement_allowed"])

    def test_all_split_reports_include_required_predictor_metrics(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_eval_splits_payload(
            root=root,
            eval_id="unit_simulator_eval_splits_metrics",
        )
        required = {"brier", "ece", "abstention_rate", "true_positive_block_rate"}

        for report in payload["split_reports"].values():
            self.assertEqual(
                set(report["predictors"]),
                {
                    "feature_similarity_simulator",
                    "base_rate_per_arm",
                    "current_heuristic_world_model",
                    "handwritten_hybrid_guard",
                    "random_with_abstain",
                    "always_original_v3",
                    "always_run_ablation",
                },
            )
            for metrics in report["predictors"].values():
                self.assertTrue(required.issubset(metrics))

    def test_decision_features_are_excluded_from_feature_model(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_eval_splits_payload(
            root=root,
            eval_id="unit_simulator_eval_splits_leak_guard",
        )

        self.assertGreater(payload["metrics"]["feature_leak_excluded_count"], 0)
        self.assertTrue(payload["gates"]["leaky_decision_features_excluded"])


if __name__ == "__main__":
    unittest.main()
