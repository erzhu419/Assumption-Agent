import unittest
from pathlib import Path

from assumption_os.autonomy_supervised_production_run import build_autonomy_supervised_production_run_payload


class AutonomySupervisedProductionRunTest(unittest.TestCase):
    def test_thirty_day_supervised_run_promotes_restricted_candidate(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_supervised_production_run_payload(
            root=root,
            eval_id="unit_autonomy_supervised_production_run",
            cycles_per_day=2,
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["supervised_day_count"], 30)
        self.assertTrue(metrics["production_autonomy_candidate_allowed"])
        self.assertEqual(metrics["ungated_mutation_count"], 0)
        self.assertEqual(metrics["forbidden_policy_change_auto_apply_count"], 0)
        self.assertGreaterEqual(metrics["low_risk_auto_apply_precision"], 0.98)
        self.assertLessEqual(metrics["human_override_rate"], 0.25)
        self.assertLessEqual(metrics["downstream_regression_rate"], 0.01)
        self.assertTrue(metrics["all_applies_replayable"])
        self.assertEqual(payload["claim_ladder_level"]["achieved"], "L3 restricted supervised production candidate")


if __name__ == "__main__":
    unittest.main()
