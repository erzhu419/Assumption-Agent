import unittest
from pathlib import Path

from assumption_os.multigeneration_framework_evolution_benchmark import (
    build_multigeneration_framework_evolution_benchmark_payload,
)


class MultigenerationFrameworkEvolutionBenchmarkTest(unittest.TestCase):
    def test_runs_five_generation_framework_evolution_line(self):
        payload = build_multigeneration_framework_evolution_benchmark_payload(
            root=Path("."),
            eval_id="unit_multigeneration_framework_evolution_benchmark",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["generation_count"], 5)
        self.assertGreaterEqual(metrics["candidate_count"], 30)
        self.assertGreaterEqual(metrics["input_residual_cluster_count"], 10)
        self.assertGreaterEqual(metrics["input_parent_framework_count"], 30)
        self.assertGreaterEqual(metrics["cross_generation_active_survival_count"], 3)

    def test_full_agent_beats_baselines_with_accept_and_reject_validation(self):
        payload = build_multigeneration_framework_evolution_benchmark_payload(
            root=Path("."),
            eval_id="unit_multigeneration_framework_evolution_benchmark_baselines",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["full_margin_vs_local_patch"], 0.20)
        self.assertGreaterEqual(metrics["full_margin_vs_raw_wisdom"], 0.30)
        self.assertGreater(metrics["full_vs_best_ablation_ci_lower"], 0.0)
        self.assertGreaterEqual(metrics["fresh_validation_accepted_count"], 1)
        self.assertGreaterEqual(metrics["fresh_validation_rejected_count"], 1)

    def test_preserves_core_safety_gates_across_generations(self):
        payload = build_multigeneration_framework_evolution_benchmark_payload(
            root=Path("."),
            eval_id="unit_multigeneration_framework_evolution_benchmark_safety",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["old_success_preservation"], 0.95)
        self.assertGreaterEqual(metrics["residual_explanation"], 0.75)
        self.assertEqual(metrics["prompt_trick_retained_count"], 0)
        self.assertEqual(metrics["core_philosophy_prior_promotion_count"], 0)
        self.assertGreaterEqual(metrics["simulator_fresh_test_reduction_rate"], 0.40)
        self.assertEqual(metrics["simulator_true_positive_block_count"], 0)
        self.assertEqual(metrics["formal_applicable_certificate_coverage"], 1.0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
