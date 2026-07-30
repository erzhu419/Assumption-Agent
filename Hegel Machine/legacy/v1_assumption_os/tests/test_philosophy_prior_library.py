import unittest
from pathlib import Path

from assumption_os.philosophy_prior_library import (
    build_philosophy_prior_library_payload,
)
from assumption_os.schema import EdgeType


class PhilosophyPriorLibraryTest(unittest.TestCase):
    def test_prior_library_has_falsifiable_principle_records(self):
        payload = build_philosophy_prior_library_payload(
            root=Path("."),
            eval_id="unit_philosophy_prior_library",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["principle_count"], 30)
        self.assertEqual(metrics["framework_prior_node_count"], 30)
        self.assertGreaterEqual(metrics["min_success_case_count"], 2)
        self.assertGreaterEqual(metrics["min_negative_case_count"], 1)
        self.assertEqual(metrics["scope_condition_coverage"], 1.0)
        self.assertEqual(metrics["failure_condition_coverage"], 1.0)
        self.assertEqual(metrics["verifier_protocol_coverage"], 1.0)

    def test_priors_are_gate_ready_and_do_not_auto_promote_core_priors(self):
        payload = build_philosophy_prior_library_payload(
            root=Path("."),
            eval_id="unit_philosophy_prior_library_gate_ready",
        )
        metrics = payload["metrics"]
        edge_types = set(payload["graph"]["edge_type_counts"])

        self.assertEqual(metrics["conservative_gate_ready_coverage"], 1.0)
        self.assertEqual(metrics["required_prior_edge_coverage"], 1.0)
        self.assertIn(EdgeType.PRESERVES_SUCCESS_CASES.value, edge_types)
        self.assertIn(EdgeType.CONFLICTS_WITH.value, edge_types)
        self.assertIn(EdgeType.EXPLAINS_RESIDUAL.value, edge_types)
        self.assertIn(EdgeType.PREDICTS_NEW_CASE.value, edge_types)
        self.assertEqual(metrics["core_prior_auto_promotion_count"], 0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_retrieval_agrees_with_expert_top3_and_roundtrips(self):
        payload = build_philosophy_prior_library_payload(
            root=Path("."),
            eval_id="unit_philosophy_prior_library_retrieval",
        )
        metrics = payload["metrics"]
        retrieval = payload["retrieval_benchmark"]

        self.assertGreaterEqual(metrics["retrieval_query_count"], 10)
        self.assertGreaterEqual(metrics["top3_expert_agreement"], 0.8)
        self.assertGreaterEqual(metrics["top1_expert_agreement"], 0.5)
        self.assertTrue(all(row["top3_hit"] for row in retrieval["rows"]))
        self.assertTrue(metrics["roundtrip_exact"])
        self.assertEqual(payload["roundtrip"]["framework_prior_node_count_after"], 30)


if __name__ == "__main__":
    unittest.main()
