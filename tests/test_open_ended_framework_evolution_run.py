import unittest
from pathlib import Path

from assumption_os.open_ended_framework_evolution_run import (
    build_open_ended_framework_evolution_run_payload,
)


class OpenEndedFrameworkEvolutionRunTest(unittest.TestCase):
    def test_multigeneration_framework_evolution_passes_bounded_gates(self):
        payload = build_open_ended_framework_evolution_run_payload(
            root=Path("."),
            eval_id="unit_open_ended_framework_evolution_run",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["generation_count"], 6)
        self.assertGreaterEqual(metrics["candidate_count"], 30)
        self.assertGreaterEqual(metrics["retained_count"], 20)
        self.assertGreaterEqual(metrics["active_framework_count"], 12)
        self.assertGreaterEqual(metrics["negative_evidence_retained_count"], 4)
        self.assertGreaterEqual(metrics["max_lineage_depth"], 6)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_open_run_preserves_conservative_obligations_and_claim_boundaries(self):
        payload = build_open_ended_framework_evolution_run_payload(
            root=Path("."),
            eval_id="unit_open_ended_framework_evolution_run_claims",
        )
        metrics = payload["metrics"]

        self.assertEqual(metrics["conservative_obligation_coverage"], 1.0)
        self.assertEqual(metrics["parent_compatibility_relation_coverage"], 1.0)
        self.assertGreaterEqual(metrics["limiting_case_survival_rate"], 0.95)
        self.assertGreaterEqual(metrics["generation_productivity_nonnegative_rate"], 0.80)
        self.assertGreaterEqual(metrics["margin_vs_best_toggle_off"], 0.12)
        self.assertEqual(metrics["core_philosophy_prior_promotion_count"], 0)
        self.assertTrue(metrics["bounded_open_ended_framework_evolution_claim_allowed"])
        self.assertFalse(metrics["unbounded_open_ended_os_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
