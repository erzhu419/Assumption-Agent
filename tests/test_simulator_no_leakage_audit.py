import unittest
from pathlib import Path

from assumption_os.simulator_no_leakage_audit import build_simulator_no_leakage_audit_payload


class SimulatorNoLeakageAuditTest(unittest.TestCase):
    def test_production_simulator_evidence_has_no_label_or_outcome_leakage(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_no_leakage_audit_payload(
            root=root,
            eval_id="unit_simulator_no_leakage_audit",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["state_feature_leak_count"], 0)
        self.assertEqual(metrics["provenance_leak_count"], 0)
        self.assertEqual(metrics["row_id_leak_count"], 0)
        self.assertEqual(metrics["prediction_outcome_exact_identity_count"], 0)
        self.assertLessEqual(metrics["prediction_outcome_near_identity_rate"], 0.02)
        self.assertGreaterEqual(metrics["mean_abs_prediction_outcome_gap"], 0.025)

    def test_audit_preserves_bounded_router_claim_boundary(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_no_leakage_audit_payload(
            root=root,
            eval_id="unit_simulator_no_leakage_audit_boundary",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["best_arm_agreement_rate"], 0.80)
        self.assertLessEqual(metrics["best_arm_agreement_rate"], 0.98)
        self.assertLess(metrics["counterfactual_mae"], metrics["global_baseline_mae"])
        self.assertLess(metrics["feature_model_loo_brier"], metrics["base_rate_loo_brier"])
        self.assertFalse(metrics["raw_simulator_promoted"])
        self.assertTrue(metrics["gate_router_promoted"])
        self.assertIn("oracle simulator replacing live validation", payload["blocked_claims"])


if __name__ == "__main__":
    unittest.main()
