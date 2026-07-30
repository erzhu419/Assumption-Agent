import unittest
from pathlib import Path

from assumption_os.conservative_generalization_gate import (
    REQUIRED_PROMOTION_RELATIONS,
    build_conservative_generalization_gate_payload,
)


class ConservativeGeneralizationGateTest(unittest.TestCase):
    def test_gate_promotes_only_conservative_generalization(self):
        payload = build_conservative_generalization_gate_payload(
            root=Path("."),
            eval_id="unit_conservative_generalization_gate",
        )
        metrics = payload["metrics"]
        decisions = metrics["decision_counts"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(decisions.get("active_scoped_framework"), 1)
        self.assertEqual(decisions.get("candidate_framework"), 1)
        self.assertEqual(decisions.get("branch_only"), 1)
        self.assertEqual(decisions.get("reject"), 1)
        self.assertGreaterEqual(metrics["active_min_old_success_preservation"], 0.95)
        self.assertGreaterEqual(metrics["active_min_residual_explanation"], 0.75)
        self.assertGreaterEqual(metrics["active_min_limiting_case_reduction"], 0.90)
        self.assertGreaterEqual(metrics["active_min_generality_gain"], 0.35)
        self.assertGreaterEqual(metrics["active_min_new_prediction_success"], 0.75)
        self.assertLessEqual(metrics["active_max_regression_cost"], 0.02)

    def test_active_framework_has_required_graph_relations(self):
        payload = build_conservative_generalization_gate_payload(
            root=Path("."),
            eval_id="unit_conservative_generalization_gate_graph",
        )
        active = next(row for row in payload["evaluations"] if row["decision"] == "active_scoped_framework")
        edge_types = set(payload["graph_patch"]["edge_type_counts"])

        self.assertTrue(REQUIRED_PROMOTION_RELATIONS.issubset(set(active["relation_types"])))
        self.assertTrue(REQUIRED_PROMOTION_RELATIONS.issubset(edge_types))
        self.assertEqual(payload["graph_patch"]["main_graph_mutation_count"], 0)

    def test_prompt_trick_is_rejected_and_claim_boundary_is_blocked(self):
        payload = build_conservative_generalization_gate_payload(
            root=Path("."),
            eval_id="unit_conservative_generalization_gate_rejects_prompt_trick",
        )
        rejected = next(row for row in payload["evaluations"] if row["decision"] == "reject")

        self.assertEqual(rejected["framework_id"], "fw_longer_context_style_boost")
        self.assertFalse(rejected["gate_checks"]["g2_old_success_preservation"])
        self.assertFalse(payload["metrics"]["unbounded_philosophy_generator_claim_allowed"])
        self.assertIn("ungated_framework_promotion", payload["blocked_claims"])


if __name__ == "__main__":
    unittest.main()
