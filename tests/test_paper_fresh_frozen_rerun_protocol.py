import unittest
from pathlib import Path

from assumption_os.paper_fresh_frozen_rerun_protocol import (
    build_paper_fresh_frozen_rerun_protocol_payload,
)


class PaperFreshFrozenRerunProtocolTest(unittest.TestCase):
    def test_protocol_is_ready_without_overclaiming_live_result(self):
        payload = build_paper_fresh_frozen_rerun_protocol_payload(
            root=Path("."),
            eval_id="unit_paper_fresh_frozen_rerun_protocol",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["target_fresh_api_call_count"], 720)
        self.assertEqual(metrics["dry_run_planned_fresh_api_call_count"], 720)
        self.assertGreaterEqual(metrics["fresh_pilot_api_call_count"], 180)
        self.assertGreaterEqual(metrics["available_problem_count"], 720)
        self.assertGreaterEqual(metrics["available_problem_domain_count"], 6)
        self.assertEqual(metrics["command_secret_hit_count"], 0)
        self.assertTrue(metrics["fresh_protocol_ready_claim_allowed"])
        self.assertFalse(metrics["target_fresh_result_claim_allowed"])

    def test_problem_manifest_is_redacted_and_disjoint(self):
        payload = build_paper_fresh_frozen_rerun_protocol_payload(
            root=Path("."),
            eval_id="unit_paper_fresh_frozen_rerun_protocol_redaction",
        )
        manifest = payload["problem_manifest"]
        sample = manifest["sample_problem_rows"][0]

        self.assertFalse(manifest["raw_payload_exposed"])
        self.assertTrue(manifest["disjoint_from_existing_samples"])
        self.assertEqual(
            sorted(sample),
            ["difficulty", "domain", "problem_hash", "problem_id"],
        )
        for forbidden in ["description", "reference_answer", "prompt", "api_secret"]:
            self.assertNotIn(forbidden, sample)


if __name__ == "__main__":
    unittest.main()
