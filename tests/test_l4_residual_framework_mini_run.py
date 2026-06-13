import unittest
from pathlib import Path

from assumption_os.l4_residual_framework_mini_run import (
    build_l4_residual_framework_mini_run_payload,
)


class L4ResidualFrameworkMiniRunTest(unittest.TestCase):
    def test_bounded_l4_mini_run_has_candidates_validations_and_expert_packet(self):
        payload = build_l4_residual_framework_mini_run_payload(
            root=Path("."),
            eval_id="unit_l4_residual_framework_mini_run",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["real_residual_cluster_count"], 10)
        self.assertGreaterEqual(metrics["candidate_framework_count"], 20)
        self.assertGreaterEqual(metrics["llm_contract_candidate_count"], 10)
        self.assertGreaterEqual(metrics["conservative_validation_count"], 5)
        self.assertGreaterEqual(metrics["bounded_fresh_validation_row_count"], 5)
        self.assertGreaterEqual(metrics["expert_review_packet_row_count"], 5)
        self.assertGreaterEqual(metrics["active_scoped_framework_count"], 1)
        self.assertGreaterEqual(metrics["negative_evidence_count"], 1)
        self.assertFalse(metrics["human_expert_panel_claim_allowed"])
        self.assertFalse(metrics["l4_external_completion_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
