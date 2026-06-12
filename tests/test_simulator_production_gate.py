import unittest
from pathlib import Path

from assumption_os.simulator_production_gate import build_simulator_production_gate_payload


class SimulatorProductionGateTest(unittest.TestCase):
    def test_production_gate_passes_as_audit_and_blocks_overclaim(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_production_gate_payload(root=root, eval_id="unit_simulator_production_gate")
        metrics = payload["metrics"]
        blockers = set(payload["promotion_decision"]["blockers"])

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["transition_row_count"], 531)
        self.assertFalse(metrics["production_simulator_candidate_allowed"])
        self.assertIn("transition_rows_minimum", blockers)
        self.assertIn("pattern_count_minimum", blockers)
        self.assertIn("counterfactual_gate_allowed", blockers)
        self.assertFalse(metrics["raw_simulator_promoted"])
        self.assertTrue(metrics["gate_router_promoted"])

    def test_split_and_calibration_requirements_are_reported(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_production_gate_payload(root=root, eval_id="unit_simulator_production_gate_metrics")
        metrics = payload["metrics"]
        requirements = payload["requirement_results"]

        self.assertGreaterEqual(metrics["domain_count"], 8)
        self.assertGreater(metrics["leave_domain_nonnegative_rate"], 0.0)
        self.assertTrue(requirements["brier_beats_base_rate"])
        self.assertTrue(requirements["ece_below_threshold"])
        self.assertTrue(requirements["true_positive_block_rate_safe"])
        self.assertTrue(requirements["raw_simulator_not_promoted"])


if __name__ == "__main__":
    unittest.main()
