import tempfile
import unittest
from pathlib import Path

from assumption_os.simulator_production_evidence_expansion import build_simulator_production_evidence_payload


class SimulatorProductionEvidenceExpansionTest(unittest.TestCase):
    def test_production_evidence_passes_router_promotion_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = build_simulator_production_evidence_payload(
                root=root,
                eval_id="unit_simulator_production_evidence",
                out_dataset=Path(temp_dir) / "production_dataset.jsonl",
                write_artifacts=True,
            )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["transition_row_count"], 2000)
        self.assertGreaterEqual(metrics["pattern_count"], 20)
        self.assertTrue(metrics["counterfactual_production_allowed"])
        self.assertTrue(metrics["counterfactual_mae_beats_global_baseline"])
        self.assertGreaterEqual(metrics["best_arm_agreement_rate"], 0.8)
        self.assertLessEqual(metrics["best_arm_agreement_rate"], 0.98)
        self.assertLess(metrics["feature_model_loo_brier"], metrics["base_rate_loo_brier"])
        self.assertTrue(metrics["production_simulator_candidate_allowed"])
        self.assertEqual(payload["claim_boundary"]["blocked_claim"], "task-world simulator replacing live validation or judges")


if __name__ == "__main__":
    unittest.main()
