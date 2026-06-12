import unittest
from pathlib import Path

from assumption_os.framework_evolution_graph_episode import (
    build_framework_evolution_graph_episode_payload,
)


class FrameworkEvolutionGraphEpisodeTest(unittest.TestCase):
    def test_framework_growth_enters_graph_lifecycle_copy_only(self):
        payload = build_framework_evolution_graph_episode_payload(
            root=Path("."),
            eval_id="unit_framework_evolution_graph_episode",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["contract_admitted_count"], 1)
        self.assertEqual(metrics["contract_quarantined_count"], 0)
        self.assertGreaterEqual(metrics["graft_added_node_count"], 6)
        self.assertEqual(metrics["required_relation_coverage"], 1.0)
        self.assertEqual(metrics["readback_relation_coverage"], 1.0)
        self.assertLessEqual(metrics["readback_active_rank"], 3)
        self.assertTrue(metrics["rollback_success"])
        self.assertTrue(metrics["journal_replay_exact"])
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_framework_episode_generates_descendant_seeds_without_core_promotion(self):
        payload = build_framework_evolution_graph_episode_payload(
            root=Path("."),
            eval_id="unit_framework_evolution_graph_episode_descendants",
        )
        metrics = payload["metrics"]
        seeds = payload["descendant_seeds"]

        self.assertGreaterEqual(metrics["descendant_seed_count"], 3)
        self.assertEqual(metrics["core_philosophy_prior_promotion_count"], 0)
        self.assertTrue(all(seed["next_action"] == "generate_child_branch_and_run_conservative_gate" for seed in seeds))
        self.assertGreaterEqual(metrics["negative_evidence_retained_count"], 1)


if __name__ == "__main__":
    unittest.main()
