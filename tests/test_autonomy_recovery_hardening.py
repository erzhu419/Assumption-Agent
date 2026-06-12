import unittest
from pathlib import Path

from assumption_os.autonomy_recovery_hardening import (
    ALLOWED_RESOLUTIONS,
    FAULTS,
    build_autonomy_recovery_hardening_payload,
)


class AutonomyRecoveryHardeningTest(unittest.TestCase):
    def test_fault_injection_payload_passes_recovery_gates(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_recovery_hardening_payload(
            root=root,
            eval_id="unit_autonomy_recovery_hardening",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["fault_count"], len(FAULTS))
        self.assertEqual(metrics["resolved_fault_count"], len(FAULTS))
        self.assertEqual(metrics["allowed_resolution_coverage"], 1.0)
        self.assertGreaterEqual(metrics["rollback_success_rate"], 0.99)
        self.assertEqual(metrics["ungated_mutation_count"], 0)
        self.assertEqual(metrics["replay_divergence_count"], 0)

    def test_each_fault_resolves_to_allowed_state(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_recovery_hardening_payload(
            root=root,
            eval_id="unit_autonomy_recovery_hardening_states",
        )

        observed_faults = {row["fault"] for row in payload["fault_results"]}
        self.assertEqual(observed_faults, set(FAULTS))
        for row in payload["fault_results"]:
            self.assertIn(row["resolution"], ALLOWED_RESOLUTIONS)
            self.assertEqual(row["orphan_manifest_count"], 0)
            self.assertEqual(row["dangling_candidate_count"], 0)
            self.assertEqual(row["ungated_mutation_count"], 0)
            self.assertFalse(row["replay_divergence_detected"])

    def test_manual_review_defer_and_rollback_paths_are_exercised(self):
        root = Path(__file__).resolve().parents[1]
        payload = build_autonomy_recovery_hardening_payload(
            root=root,
            eval_id="unit_autonomy_recovery_hardening_paths",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["manual_review_required_count"], 1)
        self.assertGreaterEqual(metrics["defer_count"], 1)
        self.assertGreaterEqual(metrics["rollback_count"], 1)


if __name__ == "__main__":
    unittest.main()
