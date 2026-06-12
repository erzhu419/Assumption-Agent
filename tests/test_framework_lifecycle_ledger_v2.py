import unittest
from pathlib import Path

from assumption_os.framework_lifecycle_ledger_v2 import (
    STATE_MACHINE,
    build_framework_lifecycle_ledger_v2_payload,
)


class FrameworkLifecycleLedgerV2Test(unittest.TestCase):
    def test_records_hegel_lifecycle_state_machine_from_real_gate_rows(self):
        payload = build_framework_lifecycle_ledger_v2_payload(
            root=Path("."),
            eval_id="unit_framework_lifecycle_ledger_v2",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(payload["state_machine"], STATE_MACHINE)
        self.assertEqual(metrics["source_evaluation_count"], 26)
        self.assertGreaterEqual(metrics["ledger_entry_count"], 24)
        self.assertGreaterEqual(metrics["active_framework_count"], 12)
        self.assertGreaterEqual(metrics["source_active_framework_count"], metrics["active_framework_count"])
        self.assertGreaterEqual(metrics["branch_to_framework_transition_count"], 8)
        self.assertEqual(metrics["core_prior_promotion_count"], 0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)

    def test_promoted_and_rejected_frameworks_have_auditable_ledger_entries(self):
        payload = build_framework_lifecycle_ledger_v2_payload(
            root=Path("."),
            eval_id="unit_framework_lifecycle_ledger_v2_coverage",
        )
        metrics = payload["metrics"]

        self.assertEqual(metrics["promoted_framework_ledger_coverage"], 1.0)
        self.assertEqual(metrics["rejected_framework_rejection_reason_coverage"], 1.0)
        self.assertGreaterEqual(metrics["negative_evidence_retained_count"], 2)
        self.assertEqual(metrics["deleted_branch_count"], 0)
        rejected = [
            entry
            for entry in payload["entries"]
            if entry["current_status"] == "rejected_boundary_only"
        ]
        self.assertTrue(rejected)
        self.assertTrue(all(entry["rejection_reason"] for entry in rejected))
        self.assertTrue(all(entry["negative_evidence"] for entry in rejected))

    def test_demotion_rollback_and_survival_recheck_are_measurable(self):
        payload = build_framework_lifecycle_ledger_v2_payload(
            root=Path("."),
            eval_id="unit_framework_lifecycle_ledger_v2_rollback",
        )
        metrics = payload["metrics"]

        self.assertEqual(metrics["active_recheck_count"], metrics["source_active_framework_count"])
        self.assertLess(metrics["active_framework_survival_rate"], 1.0)
        self.assertGreaterEqual(metrics["current_active_survival_rate"], 0.95)
        self.assertGreaterEqual(metrics["demoted_after_recheck_count"], 1)
        self.assertGreaterEqual(metrics["limiting_case_survival_rate"], 0.95)
        self.assertGreaterEqual(metrics["demotion_event_count"], 1)
        self.assertGreaterEqual(metrics["rollback_replay_count"], 1)
        self.assertEqual(metrics["rollback_final_status"], "active_scoped_framework")
        self.assertEqual(
            payload["rollback_drill"]["replay_hash"],
            payload["rollback_drill"]["replay_again_hash"],
        )
        self.assertEqual(metrics["prompt_trick_retained_count"], 0)


if __name__ == "__main__":
    unittest.main()
