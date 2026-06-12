import unittest
from pathlib import Path

from assumption_os.integrated_recursive_episode import build_integrated_recursive_episode_payload


class IntegratedRecursiveEpisodeTest(unittest.TestCase):
    def test_integrated_episode_passes_core_acceptance_gates(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_integrated_recursive_episode_payload(
            root=root,
            eval_id="unit_integrated_recursive_episode",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["residual_cluster_count"], 3)
        self.assertEqual(metrics["candidate_proposal_count"], 9)
        self.assertEqual(metrics["contract_invalid_admitted_count"], 0)
        self.assertGreaterEqual(metrics["fresh_ablation_accept_count"], 1)
        self.assertGreaterEqual(metrics["fresh_ablation_reject_count"], 1)
        self.assertTrue(metrics["accepted_candidate_survival_on_recheck"])
        self.assertTrue(metrics["autonomy_replay_exact"])
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_episode_connects_simulator_formal_queue_and_calibration(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_integrated_recursive_episode_payload(
            root=root,
            eval_id="unit_integrated_recursive_episode_links",
        )
        metrics = payload["metrics"]

        self.assertEqual(metrics["simulator_selected_count"], 3)
        self.assertEqual(metrics["simulator_true_positive_block_count"], 0)
        self.assertFalse(metrics["raw_predictor_promotion_allowed"])
        self.assertGreaterEqual(
            metrics["formal_gate_block_count"] + metrics["formal_gate_not_applicable_count"],
            1,
        )
        self.assertGreater(metrics["world_model_calibration_row_count_after"], metrics["world_model_calibration_row_count_before"])
        self.assertEqual(metrics["finite_certificate_valid_count"], 16)

    def test_episode_runs_ten_queue_cycles(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_integrated_recursive_episode_payload(
            root=root,
            eval_id="unit_integrated_recursive_episode_cycles",
        )

        self.assertEqual(len(payload["cycle_rows"]), 10)
        self.assertEqual(payload["metrics"]["queue_cycle_count"], 10)
        self.assertEqual(payload["metrics"]["queue_completed_count"], 10)


if __name__ == "__main__":
    unittest.main()
