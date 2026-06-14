import unittest
from pathlib import Path

from assumption_os.historical_morphine_rediscovery import (
    KEY_DISCOVERY_OBLIGATIONS,
    build_historical_morphine_rediscovery_payload,
)


class HistoricalMorphineRediscoveryTest(unittest.TestCase):
    def test_safe_historical_rediscovery_passes(self):
        payload = build_historical_morphine_rediscovery_payload(
            root=Path("."),
            eval_id="unit_historical_morphine_rediscovery",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["modern_knowledge_leak_count"], 0)
        self.assertEqual(metrics["operational_protocol_leak_count"], 0)
        self.assertEqual(metrics["era_constraint_violation_count"], 0)
        self.assertTrue(metrics["historical_rediscovery_claim_allowed"])
        self.assertFalse(metrics["wet_lab_reproduction_claim_allowed"])

    def test_retains_basic_active_principle_framework(self):
        payload = build_historical_morphine_rediscovery_payload(
            root=Path("."),
            eval_id="unit_historical_morphine_retention",
        )
        retained = payload["hypothesis_tree"]["retained"]

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["hypothesis_id"], "h_salt_forming_basic_active_principle")
        self.assertEqual(set(retained[0]["obligations_satisfied"]), set(KEY_DISCOVERY_OBLIGATIONS))

    def test_beats_non_recursive_baselines(self):
        payload = build_historical_morphine_rediscovery_payload(
            root=Path("."),
            eval_id="unit_historical_morphine_baselines",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["margin_vs_best_baseline"], 0.20)
        self.assertGreaterEqual(metrics["recursive_round_count"], 5)
        self.assertGreaterEqual(metrics["control_count"], 3)


if __name__ == "__main__":
    unittest.main()
