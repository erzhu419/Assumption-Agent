import unittest
from pathlib import Path

from assumption_os.autonomy_shadow_service import build_autonomy_shadow_service_payload


class AutonomyShadowServiceTest(unittest.TestCase):
    def test_shadow_service_passes_without_main_graph_mutation(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_shadow_service_payload(root=root, eval_id="unit_autonomy_shadow_service")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["shadow_day_count"], 7)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)
        self.assertEqual(metrics["ungated_mutation_count"], 0)
        self.assertEqual(metrics["expensive_live_call_count"], 0)
        self.assertTrue(metrics["all_cycles_replayable"])

    def test_low_risk_auto_apply_is_narrow_and_policy_changes_are_manual(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_shadow_service_payload(root=root, eval_id="unit_autonomy_shadow_service_policy")
        metrics = payload["metrics"]
        manual_types = {row["mutation"]["mutation_type"] for row in payload["manual_review_reports"]}

        self.assertEqual(metrics["auto_apply_allowed_type_count"], 5)
        self.assertEqual(metrics["forbidden_policy_change_auto_apply_count"], 0)
        self.assertEqual(metrics["auto_apply_rollback_success_rate"], 1.0)
        self.assertTrue(metrics["manual_review_required_for_policy_change"])
        self.assertIn("new_default_policy", manual_types)
        self.assertIn("world_model_promotion", manual_types)

    def test_production_candidate_claim_is_blocked_without_30_day_shadow(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_shadow_service_payload(root=root, eval_id="unit_autonomy_shadow_service_claim")

        self.assertFalse(payload["metrics"]["production_autonomy_candidate_allowed"])
        self.assertIn("shadow_run_shorter_than_30_days", payload["production_candidate_gate"]["block_reasons"])


if __name__ == "__main__":
    unittest.main()
