import unittest
from pathlib import Path

from assumption_os.conservative_generalization_gate_v2 import (
    build_conservative_generalization_gate_v2_payload,
)


class ConservativeGeneralizationGateV2Test(unittest.TestCase):
    def test_gate_v2_uses_real_residual_candidate_suites(self):
        payload = build_conservative_generalization_gate_v2_payload(
            root=Path("."),
            eval_id="unit_conservative_generalization_gate_v2",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["source_generator_candidate_count"], 50)
        self.assertGreaterEqual(metrics["evaluated_candidate_count"], 24)
        self.assertGreaterEqual(metrics["real_residual_candidate_rate"], 0.8)
        self.assertGreaterEqual(metrics["old_success_test_count"], metrics["evaluated_candidate_count"] * 2)
        self.assertGreaterEqual(metrics["residual_test_count"], metrics["evaluated_candidate_count"] * 2)
        self.assertGreaterEqual(metrics["unseen_domain_test_count"], metrics["evaluated_candidate_count"])

    def test_gate_v2_promotes_only_certified_frameworks_and_keeps_negative_evidence(self):
        payload = build_conservative_generalization_gate_v2_payload(
            root=Path("."),
            eval_id="unit_conservative_generalization_gate_v2_certificates",
        )
        metrics = payload["metrics"]
        decisions = metrics["decision_counts"]

        self.assertGreaterEqual(decisions.get("active_scoped_framework", 0), 1)
        self.assertGreaterEqual(decisions.get("candidate_framework", 0), 1)
        self.assertGreaterEqual(decisions.get("branch_only", 0), 1)
        self.assertGreaterEqual(metrics["old_success_reject_count"], 1)
        self.assertEqual(metrics["promoted_certificate_coverage"], 1.0)
        self.assertGreaterEqual(metrics["rejected_negative_evidence_count"], 1)
        self.assertEqual(metrics["required_relation_coverage"], 1.0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_gate_v2_records_branch_to_active_transition(self):
        payload = build_conservative_generalization_gate_v2_payload(
            root=Path("."),
            eval_id="unit_conservative_generalization_gate_v2_transition",
        )
        metrics = payload["metrics"]
        transition_rows = [
            row for row in payload["evaluations"]
            if row.get("stage") in {"initial_branch", "promoted_after_evidence"}
        ]

        self.assertGreaterEqual(metrics["branch_to_active_transition_count"], 1)
        self.assertEqual([row["decision"] for row in transition_rows], ["branch_only", "active_scoped_framework"])
        self.assertEqual(transition_rows[-1].get("promotion_from"), "branch_only")


if __name__ == "__main__":
    unittest.main()
