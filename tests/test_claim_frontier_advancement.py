import unittest
from pathlib import Path

from assumption_os.claim_frontier_advancement import build_claim_frontier_advancement_payload


class ClaimFrontierAdvancementTest(unittest.TestCase):
    def test_frontier_advances_bounded_next_claims_without_overclaim(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_claim_frontier_advancement_payload(
            root=root,
            eval_id="unit_claim_frontier_advancement",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["source_artifact_pass_rate"], 1.0)
        self.assertEqual(metrics["frontier_track_pass_count"], 3)
        self.assertGreaterEqual(metrics["frontier_advancement_score"], 0.90)
        self.assertEqual(metrics["autonomy_downstream_regression_rate"], 0.0)
        self.assertLessEqual(metrics["simulator_counterfactual_mae"], 0.01)
        self.assertGreaterEqual(metrics["formal_lean_theorem_count"], 30)
        self.assertFalse(metrics["full_theorem_prover_claim_allowed"])

        blocked = {row["claim_id"]: row["allowed"] for row in payload["blocked_claims"]}
        self.assertFalse(blocked["unbounded_24_7_autonomous_self_evolution_os"])
        self.assertFalse(blocked["world_simulator_replacing_live_ablation_or_judges"])
        self.assertFalse(blocked["full_category_theory_theorem_prover"])

    def test_frontier_tracks_keep_next_evidence_requirements(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_claim_frontier_advancement_payload(
            root=root,
            eval_id="unit_claim_frontier_advancement_evidence",
        )

        self.assertEqual(len(payload["frontier_tracks"]), 3)
        for row in payload["frontier_tracks"]:
            self.assertTrue(row["frontier_l3p5_allowed"])
            self.assertIn("L3.5", row["next_bounded_claim"])
            self.assertTrue(row["next_evidence_required"])
            self.assertIn("L4", row["blocked_upper_claim"])


if __name__ == "__main__":
    unittest.main()
