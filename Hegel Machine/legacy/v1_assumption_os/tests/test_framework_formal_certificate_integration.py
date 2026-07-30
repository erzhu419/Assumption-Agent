import unittest
from pathlib import Path

from assumption_os.framework_formal_certificate_integration import (
    build_framework_formal_certificate_integration_payload,
)


class FrameworkFormalCertificateIntegrationTest(unittest.TestCase):
    def test_formal_applicable_frameworks_have_certificate_coverage(self):
        payload = build_framework_formal_certificate_integration_payload(
            root=Path("."),
            eval_id="unit_framework_formal_certificate_integration",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["formal_applicable_count"], 8)
        self.assertEqual(metrics["formal_applicable_certificate_coverage"], 1.0)
        self.assertEqual(metrics["formal_applicable_proof_obligation_pass_rate"], 1.0)
        self.assertTrue(metrics["bounded_formal_stack_claim_allowed"])
        self.assertFalse(metrics["full_theorem_prover_claim_allowed"])

    def test_unsafe_mapping_blocks_and_non_formalizable_controls_abstain(self):
        payload = build_framework_formal_certificate_integration_payload(
            root=Path("."),
            eval_id="unit_framework_formal_certificate_integration_controls",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["unsafe_mapping_block_count"], 1)
        self.assertGreaterEqual(metrics["not_formalizable_control_count"], 2)
        self.assertEqual(metrics["non_formalizable_theorem_prover_invocation_count"], 0)
        self.assertEqual(metrics["semi_formal_theorem_prover_invocation_count"], 0)
        self.assertTrue(all(not row["theorem_prover_invoked"] for row in payload["non_formalizable_controls"]))

    def test_lean_reproducibility_and_negative_controls_are_preserved(self):
        payload = build_framework_formal_certificate_integration_payload(
            root=Path("."),
            eval_id="unit_framework_formal_certificate_integration_lean",
        )
        metrics = payload["metrics"]

        self.assertTrue(metrics["external_lean_check_passed"])
        self.assertGreaterEqual(metrics["external_lean_theorem_count"], 20)
        self.assertTrue(metrics["lean_verified_finite_fragment_claim_allowed"])
        self.assertGreaterEqual(metrics["negative_control_blocked_count"], 7)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
