import unittest

from assumption_os.hle_fast_policy_miner import (
    POLICY_MINER_VERSION,
    mine_fast_policy_hypotheses,
)


class HleFastPolicyMinerTests(unittest.TestCase):
    def test_mines_candidate_gap_and_no_fallback_policies(self):
        dataset = {
            "records": [
                _row(
                    "q1",
                    correct=False,
                    failure_bucket="candidate_generation_missed_gold",
                    gate_status="no_fallback",
                    latency=360.0,
                ),
                _row(
                    "q2",
                    correct=False,
                    failure_bucket="candidate_generation_missed_gold",
                    gate_status="no_fallback",
                    latency=340.0,
                ),
                _row(
                    "q3",
                    correct=True,
                    failure_bucket="none",
                    gate_status="abstained",
                    latency=320.0,
                ),
            ]
        }

        report = mine_fast_policy_hypotheses(dataset, min_support=2)

        self.assertEqual(report["miner_version"], POLICY_MINER_VERSION)
        self.assertFalse(report["raw_content_persisted"])
        self.assertEqual(report["summary"]["no_fallback_count"], 2)
        actions = {policy["action"] for policy in report["hypotheses"]}
        self.assertIn("deterministic_option_coverage_and_required_term_source_bundle", actions)
        self.assertIn("preserve_slow_baseline_when_verified_gate_has_no_direct_candidate", actions)
        self.assertIn("batch_or_cap_source_directness_calls_before_slow_baseline_fallback", actions)
        self.assertIn("no_fallback_present", report["promotion_gate_blockers"])
        for policy in report["hypotheses"]:
            self.assertEqual(policy["promotion_status"], "candidate")
            self.assertEqual(policy["fallback_behavior"], "preserve_slow_baseline")
            self.assertFalse(policy["raw_content_persisted"])

    def test_report_stays_redacted(self):
        dataset = {
            "records": [
                {
                    **_row("q1", correct=False, failure_bucket="candidate_generation_missed_gold"),
                    "question": "RAW QUESTION SHOULD NEVER APPEAR",
                },
                _row("q2", correct=False, failure_bucket="candidate_generation_missed_gold"),
            ]
        }

        report = mine_fast_policy_hypotheses(dataset, min_support=2)

        self.assertNotIn("RAW QUESTION", str(report))
        evidence = report["hypotheses"][0]["evidence_rows"][0]
        self.assertIn("record_ref_hash", evidence)
        self.assertFalse(evidence["raw_content_persisted"])


def _row(
    question_id: str,
    *,
    correct: bool,
    failure_bucket: str,
    gate_status: str = "unknown",
    latency: float = 10.0,
) -> dict:
    return {
        "question_id": question_id,
        "question_hash": f"{question_id}-hash",
        "action": "verified_or_abstain_direct_fallback",
        "category": "Chemistry",
        "domain": "Chemistry",
        "correct": correct,
        "failure_bucket": failure_bucket,
        "selected_label_hash": f"{question_id}-pred",
        "gold_after_run_label_hash": f"{question_id}-gold",
        "path_hashes": {"verified_or_abstain_gate_status": gate_status},
        "latency_seconds": latency,
        "cost": 9.0,
        "raw_content_persisted": False,
    }


if __name__ == "__main__":
    unittest.main()
