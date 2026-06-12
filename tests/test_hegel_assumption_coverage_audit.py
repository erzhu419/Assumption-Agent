import unittest
from pathlib import Path

from assumption_os.hegel_assumption_coverage_audit import (
    build_hegel_assumption_coverage_audit_payload,
)


class HegelAssumptionCoverageAuditTest(unittest.TestCase):
    def test_review_and_deep_items_are_closed(self):
        payload = build_hegel_assumption_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_hegel_assumption_coverage",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["review_open_gap_count"], 0)
        self.assertEqual(metrics["deep_open_gap_count"], 0)
        self.assertEqual(metrics["review_item_pass_count"], metrics["review_item_count"])
        self.assertEqual(metrics["deep_item_pass_count"], metrics["deep_item_count"])

    def test_paper_delivery_and_llm_experiment_are_accounted_for(self):
        payload = build_hegel_assumption_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_hegel_assumption_delivery",
        )
        metrics = payload["metrics"]
        llm = payload["llm_candidate_experiment_summary"]

        self.assertGreaterEqual(metrics["paper_delivery_file_count"], 2)
        self.assertTrue(llm["pass"])
        self.assertEqual(llm["llm_candidate_count"], 10)
        self.assertTrue(llm["paper_preflight_claim_allowed"])
        self.assertFalse(llm["strong_live_llm_claim_allowed"])

    def test_unbounded_and_external_claim_boundaries_remain_blocked(self):
        payload = build_hegel_assumption_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_hegel_assumption_claims",
        )
        metrics = payload["metrics"]
        claims = {row["claim_id"]: row for row in payload["claim_boundaries"]}

        self.assertEqual(metrics["overclaim_leak_count"], 0)
        self.assertTrue(claims["unbounded_l4_autonomous_os"]["blocked"])
        self.assertTrue(claims["human_expert_panel_completed"]["blocked"])
        self.assertTrue(claims["full_category_theory_theorem_prover"]["blocked"])


if __name__ == "__main__":
    unittest.main()
