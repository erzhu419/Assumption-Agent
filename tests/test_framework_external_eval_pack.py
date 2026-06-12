import unittest
from pathlib import Path

from assumption_os.framework_external_eval_pack import build_framework_external_eval_pack_payload


class FrameworkExternalEvalPackTest(unittest.TestCase):
    def test_expert_packet_and_proxy_preflight_are_ready_without_fabricating_humans(self):
        payload = build_framework_external_eval_pack_payload(
            root=Path("."),
            eval_id="unit_framework_external_eval_pack",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["expert_annotation_packet_row_count"], 30)
        self.assertGreaterEqual(metrics["expert_proxy_agreement_with_system"], 0.65)
        self.assertFalse(metrics["human_panel_completed"])
        self.assertTrue(metrics["human_panel_status_recorded"])

    def test_fresh_rerun_protocol_and_repro_pack_are_complete(self):
        payload = build_framework_external_eval_pack_payload(
            root=Path("."),
            eval_id="unit_framework_external_eval_pack_repro",
        )
        metrics = payload["metrics"]

        self.assertTrue(metrics["framework_specific_fresh_rerun_protocol_ready"])
        self.assertTrue(metrics["old_evidence_reuse_blocked_in_protocol"])
        self.assertEqual(metrics["artifact_hash_coverage"], 1.0)
        self.assertGreaterEqual(metrics["exact_command_count"], 8)
        self.assertEqual(metrics["secret_scan_match_count"], 0)

    def test_claim_ledger_formula_and_bounded_definition_are_present(self):
        payload = build_framework_external_eval_pack_payload(
            root=Path("."),
            eval_id="unit_framework_external_eval_pack_claims",
        )
        metrics = payload["metrics"]

        self.assertGreaterEqual(metrics["claim_ledger_entry_count"], 10)
        self.assertGreaterEqual(metrics["overclaim_blocked_count"], 4)
        self.assertEqual(metrics["framework_growth_formula_term_count"], 8)
        self.assertGreaterEqual(metrics["bounded_90_definition_item_count"], 10)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
