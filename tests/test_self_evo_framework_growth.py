import unittest
from pathlib import Path

from assumption_os.framework_branch_ledger import build_framework_branch_ledger_payload
from assumption_os.philosophy_growth_benchmark import build_philosophy_growth_benchmark_payload
from assumption_os.residual_to_framework_generator import build_residual_to_framework_generator_payload
from assumption_os.self_evo_roadmap_coverage_audit import build_self_evo_roadmap_coverage_audit_payload
from assumption_os.self_evo_paper_evidence_pack import build_self_evo_paper_evidence_pack_payload


class SelfEvoFrameworkGrowthTest(unittest.TestCase):
    def test_residual_to_framework_generator_emits_structured_candidates(self):
        payload = build_residual_to_framework_generator_payload(
            root=Path("."),
            eval_id="unit_residual_to_framework_generator",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["anomaly_family_count"], 20)
        self.assertGreaterEqual(metrics["real_residual_cluster_count"], 20)
        self.assertGreaterEqual(metrics["candidate_framework_count"], 50)
        self.assertGreaterEqual(metrics["trajectory_type_count"], 6)
        self.assertGreaterEqual(metrics["non_scope_narrowing_candidate_rate"], 0.2)
        self.assertGreaterEqual(metrics["framework_combination_or_generalization_rate"], 0.2)
        self.assertGreaterEqual(metrics["negative_evidence_candidate_count"], 10)
        self.assertGreaterEqual(metrics["live_feedback_candidate_count"], 10)
        self.assertEqual(metrics["structured_candidate_coverage"], 1.0)
        self.assertEqual(metrics["raw_wisdom_candidate_count"], 0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)
        sample = payload["candidate_frameworks"][0]
        self.assertIn("candidate_framework_id", sample)
        self.assertIn("generation_trace", sample)
        self.assertIn("risk_predictions", sample)
        self.assertIn("required_tests", sample)

    def test_branch_ledger_records_promotions_and_negative_evidence(self):
        payload = build_framework_branch_ledger_payload(
            root=Path("."),
            eval_id="unit_framework_branch_ledger",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["ledger_entry_count"], 4)
        self.assertEqual(metrics["status_counts"]["active_scoped_framework"], 1)
        self.assertGreaterEqual(metrics["negative_evidence_retained_count"], 1)
        self.assertEqual(metrics["core_promotion_count"], 0)
        self.assertEqual(payload["replay"]["replay_hash"], payload["replay"]["replay_again_hash"])

    def test_philosophy_growth_benchmark_beats_local_patch_and_raw_wisdom(self):
        payload = build_philosophy_growth_benchmark_payload(
            root=Path("."),
            eval_id="unit_philosophy_growth_benchmark",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["active_framework_survival_count"], 5)
        self.assertGreater(metrics["conservative_growth_score"], metrics["local_patch_growth_score"])
        self.assertGreater(metrics["local_patch_growth_score"], metrics["raw_wisdom_growth_score"])
        self.assertLess(metrics["conservative_regression_cost"], metrics["local_patch_regression_cost"])
        self.assertEqual(metrics["core_philosophy_prior_promotion_count"], 0)

    def test_self_evo_roadmap_coverage_closes_r7_bounded_claim(self):
        payload = build_self_evo_roadmap_coverage_audit_payload(
            root=Path("."),
            eval_id="unit_self_evo_roadmap_coverage",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["open_roadmap_item_count"], 0)
        self.assertEqual(metrics["r7_item_pass_count"], metrics["r7_item_count"])
        self.assertGreaterEqual(metrics["bounded_ugse_score"], 0.90)
        self.assertTrue(metrics["fresh_broad_generator_repair_passed"])
        self.assertEqual(metrics["fresh_broad_generator_repair_calls"], 720)
        self.assertGreaterEqual(metrics["fresh_broad_generator_repair_delta"], 0.10)
        self.assertFalse(metrics["unbounded_self_evolution_os_claim_allowed"])

    def test_self_evo_paper_evidence_pack_closes_reviewer_facing_artifact(self):
        payload = build_self_evo_paper_evidence_pack_payload(
            root=Path("."),
            eval_id="unit_self_evo_paper_evidence_pack",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["source_artifact_pass_rate"], 1.0)
        self.assertGreaterEqual(metrics["roadmap_bounded_ugse_score"], 0.90)
        self.assertEqual(metrics["fresh_repair_fresh_api_call_count"], 720)
        self.assertGreaterEqual(metrics["fresh_repair_delta_vs_original"], 0.10)
        self.assertGreater(metrics["fresh_repair_ci_lower_minus_original_ci_upper"], 0.0)
        self.assertGreaterEqual(metrics["paper_skeleton_section_count"], 10)
        self.assertFalse(metrics["full_theorem_prover_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
