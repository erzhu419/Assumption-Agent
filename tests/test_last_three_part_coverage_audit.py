import unittest
from pathlib import Path

from assumption_os.last_three_part_coverage_audit import (
    build_last_three_part_coverage_audit_payload,
)


class LastThreePartCoverageAuditTest(unittest.TestCase):
    def test_all_actionable_last_three_part_tickets_are_covered(self):
        payload = build_last_three_part_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_last_three_part_coverage_audit",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["engineering_ticket_count"], 30)
        self.assertEqual(metrics["engineering_open_gap_count"], 0)
        self.assertEqual(metrics["overclaim_leak_count"], 0)
        self.assertEqual(metrics["source_artifact_pass_rate"], 1.0)

    def test_formal_markov_geometry_and_transfer_tickets_are_explicit(self):
        payload = build_last_three_part_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_last_three_part_coverage_audit_formal",
        )
        rows = {row["ticket_id"]: row for row in payload["tickets"]}

        for ticket_id in [
            "C5_markov_kernel_extension",
            "C6_information_geometry_plugin",
            "C7_formal_transfer_benchmark",
            "C8_claim_gate",
        ]:
            self.assertIn(ticket_id, rows)
            self.assertEqual(rows[ticket_id]["status"], "pass")

        self.assertGreaterEqual(rows["C5_markov_kernel_extension"]["key_metrics"]["markov_kernel_count"], 5)
        self.assertTrue(rows["C6_information_geometry_plugin"]["key_metrics"]["not_truth_oracle"])
        self.assertGreaterEqual(rows["C7_formal_transfer_benchmark"]["key_metrics"]["pairwise_auc"], 0.95)

    def test_unbounded_claims_remain_boundaries_not_gaps(self):
        payload = build_last_three_part_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_last_three_part_coverage_audit_boundaries",
        )
        claims = {row["claim_id"]: row for row in payload["claim_boundaries"]}

        for claim_id in [
            "unbounded_24_7_general_autonomous_os",
            "raw_world_simulator_replaces_live_validation",
            "complete_category_theory_theorem_prover",
            "brand_new_live_api_main_paper_experiment",
            "ungated_default_policy_or_main_graph_mutation",
        ]:
            self.assertIn(claim_id, claims)
            self.assertTrue(claims[claim_id]["blocked"])


if __name__ == "__main__":
    unittest.main()
