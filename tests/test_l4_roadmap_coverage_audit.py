import unittest
from pathlib import Path

from assumption_os.l4_roadmap_coverage_audit import build_l4_roadmap_coverage_audit_payload


class L4RoadmapCoverageAuditTest(unittest.TestCase):
    def test_all_seven_l4_stages_have_preflight_evidence(self):
        payload = build_l4_roadmap_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_l4_roadmap_coverage_audit",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["stage_count"], 7)
        self.assertEqual(metrics["stage_preflight_pass_count"], 7)
        self.assertGreaterEqual(metrics["l4_mini_requirement_preflight_count"], 8)
        self.assertGreaterEqual(metrics["real_residual_candidate_count"], 20)
        self.assertTrue(metrics["l4a_preflight_claim_allowed"])

    def test_l4_completion_and_l4b_claims_remain_blocked(self):
        payload = build_l4_roadmap_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_l4_roadmap_claim_boundaries",
        )
        metrics = payload["metrics"]

        self.assertFalse(metrics["completed_l4a_claim_allowed"])
        self.assertFalse(metrics["l4b_unbounded_claim_allowed"])
        self.assertEqual(metrics["overclaim_leak_count"], 0)
        self.assertEqual(metrics["blocked_claim_boundary_count"], metrics["claim_boundary_count"])


if __name__ == "__main__":
    unittest.main()
