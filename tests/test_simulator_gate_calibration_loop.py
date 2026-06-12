import tempfile
import unittest
from pathlib import Path

from assumption_os.simulator_gate_calibration_loop import (
    FORBIDDEN_ACTIONS,
    FORBIDDEN_ORACLE_LEVELS,
    build_simulator_gate_calibration_loop_payload,
)


class SimulatorGateCalibrationLoopTest(unittest.TestCase):
    def test_gate_calibration_loop_passes_and_keeps_raw_simulator_unpromoted(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = build_simulator_gate_calibration_loop_payload(
                root=root,
                eval_id="unit_simulator_gate_calibration_loop",
                writeback_out=Path(temp_dir) / "writeback.jsonl",
            )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertTrue(metrics["b3_uncertainty_pass"])
        self.assertTrue(metrics["i2_episode_pass"])
        self.assertFalse(metrics["raw_simulator_promoted"])
        self.assertTrue(metrics["gate_router_promoted"])
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_routing_policy_blocks_oracle_behavior(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_gate_calibration_loop_payload(
            root=root,
            eval_id="unit_simulator_gate_calibration_loop_policy",
            write_artifact=False,
        )

        for row in payload["routing_policy"]:
            self.assertNotIn(row["routing_level"], FORBIDDEN_ORACLE_LEVELS)
            self.assertNotIn(row["recommended_action"], FORBIDDEN_ACTIONS)
            self.assertFalse(row["can_auto_accept"])
            self.assertFalse(row["can_auto_apply_policy_change"])
            self.assertFalse(row["can_replace_judge"])

    def test_writeback_rows_emit_simulator_defect_residuals(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_simulator_gate_calibration_loop_payload(
            root=root,
            eval_id="unit_simulator_gate_calibration_loop_residuals",
            write_artifact=False,
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["fresh_writeback_row_count"], 6)
        self.assertGreaterEqual(metrics["deferred_writeback_row_count"], 2)
        self.assertEqual(metrics["high_confidence_wrong_count"], metrics["simulator_defect_residual_count"])
        self.assertGreaterEqual(metrics["simulator_defect_residual_count"], 2)
        self.assertTrue(
            all(row["residual_type"] == "SIMULATOR_DEFECT" for row in payload["simulator_defect_residuals"])
        )


if __name__ == "__main__":
    unittest.main()
