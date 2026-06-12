import unittest
from pathlib import Path

from assumption_os.llm_framework_candidate_experiment import (
    build_llm_framework_candidate_experiment_payload,
)


class LlmFrameworkCandidateExperimentTest(unittest.TestCase):
    def test_llm_contract_candidates_are_built_from_real_residuals(self):
        payload = build_llm_framework_candidate_experiment_payload(
            root=Path("."),
            eval_id="unit_llm_framework_candidate_experiment",
            execute_live=False,
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["llm_candidate_count"], 10)
        self.assertGreaterEqual(metrics["real_residual_source_count"], 10)
        self.assertEqual(metrics["llm_contract_field_coverage"], 1.0)
        self.assertGreaterEqual(metrics["framework_combination_or_generalization_count"], 4)

    def test_top2_validation_and_negative_control_are_present(self):
        payload = build_llm_framework_candidate_experiment_payload(
            root=Path("."),
            eval_id="unit_llm_framework_candidate_validation",
            execute_live=False,
        )
        metrics = payload["metrics"]

        self.assertEqual(metrics["top2_validation_count"], 2)
        self.assertGreaterEqual(metrics["top2_min_old_success_preservation"], 0.90)
        self.assertGreaterEqual(metrics["top2_min_residual_explanation"], 0.70)
        self.assertGreaterEqual(metrics["accepted_or_candidate_validation_count"], 1)
        self.assertGreaterEqual(metrics["negative_control_validation_count"], 1)

    def test_live_llm_claim_is_blocked_without_execute_live(self):
        payload = build_llm_framework_candidate_experiment_payload(
            root=Path("."),
            eval_id="unit_llm_framework_candidate_claim_boundary",
            execute_live=False,
        )
        metrics = payload["metrics"]

        self.assertFalse(metrics["live_llm_api_executed"])
        self.assertFalse(metrics["strong_live_llm_claim_allowed"])
        self.assertTrue(metrics["paper_preflight_claim_allowed"])
        self.assertIn("fresh_live_llm_candidate_generation_completed", payload["blocked_claims"])
        self.assertEqual(metrics["secret_scan_match_count"], 0)


if __name__ == "__main__":
    unittest.main()
