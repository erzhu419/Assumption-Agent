import unittest
from pathlib import Path

from assumption_os.framework_growth_ablation_suite import (
    build_framework_growth_ablation_suite_payload,
)


class FrameworkGrowthAblationSuiteTest(unittest.TestCase):
    def test_full_framework_growth_beats_toggle_offs_and_baselines(self):
        payload = build_framework_growth_ablation_suite_payload(
            root=Path("."),
            eval_id="unit_framework_growth_ablation_suite",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["ablation_count"], 8)
        self.assertGreaterEqual(metrics["full_margin_vs_best_toggle_off"], 0.12)
        self.assertGreaterEqual(metrics["full_margin_vs_local_patch"], 0.20)
        self.assertGreaterEqual(metrics["full_margin_vs_raw_wisdom"], 0.30)
        self.assertFalse(metrics["full_prompt_trick_retained"])

    def test_toggle_offs_expose_expected_failure_modes(self):
        payload = build_framework_growth_ablation_suite_payload(
            root=Path("."),
            eval_id="unit_framework_growth_ablation_suite_failures",
        )
        metrics = payload["metrics"]
        rows = {row["variant"]: row for row in payload["ablation_rows"]}

        self.assertGreaterEqual(metrics["max_old_success_drop_vs_full"], 0.10)
        self.assertGreaterEqual(metrics["max_regression_increase_vs_full"], 0.08)
        self.assertGreaterEqual(metrics["no_ledger_unsafe_promotion_count"], 1)
        self.assertGreaterEqual(metrics["no_graph_lifecycle_readback_penalty"], 0.15)
        self.assertTrue(rows["no_conservative_gate_residual_only"]["prompt_trick_retained"])
        self.assertGreater(rows["no_branch_ledger_no_pruning"]["unsafe_promotion_count"], 0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
