import unittest
from pathlib import Path

from assumption_os.integrated_recursive_episode_b3_c2 import (
    build_integrated_recursive_episode_b3_c2_payload,
)


class IntegratedRecursiveEpisodeB3C2Test(unittest.TestCase):
    def test_b3_c2_episode_passes_core_gates(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_integrated_recursive_episode_b3_c2_payload(
            root=root,
            eval_id="unit_integrated_recursive_episode_b3_c2",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["candidate_count"], 9)
        self.assertTrue(metrics["b3_pass"])
        self.assertTrue(metrics["c2_pass"])
        self.assertEqual(metrics["b3_forbidden_action_recommended_count"], 0)
        self.assertEqual(metrics["c2_forbidden_generator_output_count"], 0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)
        self.assertTrue(metrics["autonomy_replay_exact"])

    def test_uncertainty_abstain_is_deferred_not_auto_executed(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_integrated_recursive_episode_b3_c2_payload(
            root=root,
            eval_id="unit_integrated_recursive_episode_b3_c2_abstain",
        )
        metrics = payload["metrics"]
        deferred = [row for row in payload["cycle_rows"] if row["cycle_action"] == "defer_live_validation"]

        self.assertGreaterEqual(metrics["b3_abstain_selected_count"], 1)
        self.assertEqual(metrics["abstained_candidate_auto_execute_count"], 0)
        self.assertTrue(deferred)
        self.assertTrue(all(row["auto_executed"] is False for row in deferred))

    def test_c2_lean_checked_formal_gate_and_fresh_readback_are_both_used(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_integrated_recursive_episode_b3_c2_payload(
            root=root,
            eval_id="unit_integrated_recursive_episode_b3_c2_formal",
        )
        metrics = payload["metrics"]

        self.assertTrue(metrics["c2_external_lean_check_passed"])
        self.assertGreaterEqual(metrics["formal_gate_block_count"], 1)
        self.assertGreaterEqual(metrics["formal_gate_lean_checked_count"], 1)
        self.assertGreaterEqual(metrics["fresh_ablation_accept_count"], 1)
        self.assertGreaterEqual(metrics["fresh_ablation_reject_count"], 1)
        self.assertTrue(metrics["accepted_candidate_survival_on_recheck"])
        self.assertGreater(metrics["calibration_row_count_after"], metrics["calibration_row_count_before"])


if __name__ == "__main__":
    unittest.main()
