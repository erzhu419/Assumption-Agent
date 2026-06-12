import unittest
from pathlib import Path

from assumption_os.creative_hypothesis_trajectory_search import (
    build_creative_hypothesis_trajectory_search_payload,
)
from assumption_os.finite_theorem_fragment import extract_natural_language_diagram
from assumption_os.main_graph_controlled_apply_monitor import (
    build_main_graph_controlled_apply_monitor_payload,
)
from assumption_os.nl_to_diagram_scale_benchmark import (
    build_nl_to_diagram_scale_benchmark_payload,
)
from assumption_os.paper_frozen_main_experiment_v2 import (
    build_paper_frozen_main_experiment_v2_payload,
)


class LastThreePartClosureModuleTests(unittest.TestCase):
    def test_scaled_nl_to_diagram_benchmark_passes(self):
        payload = build_nl_to_diagram_scale_benchmark_payload(
            eval_id="unit_nl_to_diagram_scale",
            examples_per_family=8,
        )
        metrics = payload["metrics"]
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["family_count"], 13)
        self.assertEqual(metrics["positive_accuracy"], 1.0)
        self.assertEqual(metrics["negative_specificity"], 1.0)
        self.assertFalse(metrics["full_theorem_prover_claim_allowed"])

    def test_new_finite_diagram_families_extract(self):
        bottleneck = extract_natural_language_diagram(
            "The enzyme saturates, so more substrate cannot increase throughput past the bottleneck capacity."
        )
        randomized = extract_natural_language_diagram(
            "A clinical trial and an A/B test both randomize units into treatment and control groups."
        )
        near_negative = extract_natural_language_diagram(
            "A residual complaint from a user is not the same as a ResNet residual transport diagram."
        )
        self.assertEqual(bottleneck["family"], "bottleneck_capacity_limit")
        self.assertEqual(randomized["family"], "randomized_counterfactual_evaluation")
        self.assertEqual(near_negative["status"], "not_applicable")

    def test_creative_hypothesis_trajectory_search_passes(self):
        payload = build_creative_hypothesis_trajectory_search_payload(
            root=Path("."),
            eval_id="unit_creative_hypothesis_trajectory_search",
            generations=5,
            seed_limit=14,
            frontier_width=14,
        )
        metrics = payload["metrics"]
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["candidate_count"], 300)
        self.assertGreaterEqual(metrics["retained_family_count"], 18)
        self.assertGreaterEqual(metrics["nonlocal_candidate_ratio"], 0.40)
        self.assertEqual(metrics["graph_mutation_count"], 0)

    def test_paper_frozen_main_experiment_v2_passes(self):
        payload = build_paper_frozen_main_experiment_v2_payload(
            root=Path("."),
            eval_id="unit_paper_frozen_main_experiment_v2",
            bootstrap_samples=100,
        )
        metrics = payload["metrics"]
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["problem_count"], 1000)
        self.assertGreater(metrics["full_v3_margin_over_best_baseline_score"], 0.0)
        self.assertGreater(metrics["min_pairwise_utility"], 0.55)
        self.assertEqual(metrics["new_api_call_count"], 0)

    def test_main_graph_controlled_apply_monitor_passes(self):
        payload = build_main_graph_controlled_apply_monitor_payload(
            root=Path("."),
            eval_id="unit_main_graph_controlled_apply_monitor",
            monitor_days=30,
        )
        metrics = payload["metrics"]
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["source_main_graph_mutated"])
        self.assertEqual(metrics["regression_alert_count"], 0)
        self.assertGreaterEqual(metrics["rollback_entry_count"], metrics["source_planned_archive_count"])


if __name__ == "__main__":
    unittest.main()
