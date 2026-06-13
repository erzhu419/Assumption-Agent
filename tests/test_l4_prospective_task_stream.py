import unittest
from pathlib import Path

from assumption_os.l4_prospective_task_stream import build_l4_prospective_task_stream_payload


class L4ProspectiveTaskStreamTest(unittest.TestCase):
    def test_manifest_is_large_redacted_and_pre_registered(self):
        payload = build_l4_prospective_task_stream_payload(
            root=Path("."),
            eval_id="unit_l4_prospective_task_stream",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["manifest_task_count"], 100)
        self.assertGreaterEqual(metrics["manifest_domain_count"], 6)
        self.assertGreaterEqual(metrics["baseline_count"], 10)
        self.assertEqual(metrics["outcome_field_count"], 0)
        self.assertFalse(metrics["raw_prompt_or_answer_exposed"])
        self.assertTrue(metrics["prospective_manifest_claim_allowed"])
        self.assertFalse(metrics["completed_external_benchmark_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
