import unittest
from pathlib import Path

from assumption_os.finite_formal_reasoning_stack import build_finite_formal_reasoning_stack_payload


class FiniteFormalReasoningStackTest(unittest.TestCase):
    def test_formal_reasoning_stack_passes_bounded_claim_gate(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_formal_reasoning_stack_payload(
            root=root,
            eval_id="unit_finite_formal_reasoning_stack",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["dsl_valid"])
        self.assertTrue(metrics["kernel_composition_pass"])
        self.assertTrue(metrics["kernel_negative_control_rejected"])
        self.assertTrue(metrics["finite_theorem_fragment_external_lean_passed"])
        self.assertGreaterEqual(metrics["finite_theorem_fragment_external_lean_theorem_count"], 20)
        self.assertTrue(metrics["lean_verified_finite_theorem_fragment_claim_allowed"])
        self.assertTrue(metrics["bounded_formal_stack_claim_allowed"])
        self.assertFalse(metrics["full_theorem_prover_claim_allowed"])

    def test_information_geometry_is_metric_plugin_not_truth_oracle(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_formal_reasoning_stack_payload(
            root=root,
            eval_id="unit_finite_formal_reasoning_stack_geometry",
        )
        geometry = payload["information_geometry"]

        self.assertGreaterEqual(geometry["metric_count"], 5)
        self.assertTrue(geometry["not_truth_oracle"])
        self.assertIn("jensen_shannon", geometry["metrics"])
        self.assertIn("frobenius_kernel_distance", geometry["metrics"])

    def test_transfer_benchmark_records_overreach_residual(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_finite_formal_reasoning_stack_payload(
            root=root,
            eval_id="unit_finite_formal_reasoning_stack_transfer",
        )
        transfer = payload["formal_transfer_benchmark"]

        self.assertGreaterEqual(transfer["pairwise_auc"], 0.95)
        self.assertEqual(transfer["negative_control_rejection_rate"], 1.0)
        self.assertGreaterEqual(transfer["overreach_residual_count"], 1)


if __name__ == "__main__":
    unittest.main()
