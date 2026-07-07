import unittest

from assumption_os.hle_fast_policy_promotion_report import (
    PROMOTION_REPORT_VERSION,
    build_hle_fast_policy_promotion_report_from_rows,
)


class HleFastPolicyPromotionReportTests(unittest.TestCase):
    def test_report_separates_triads_from_agent_failure_mining(self):
        rows = [
            _row("p1", "raw", True, "A"),
            _row("p1", "hipporag_baseline", True, "A"),
            _row("p1", "assumption_agent_recursive_verify", False, "B", gate="no_fallback"),
            _row("p2", "raw", False, "C"),
            _row("p2", "hipporag_baseline", False, "C"),
            _row("p2", "assumption_agent_recursive_verify", False, "C", failure_bucket="candidate_generation_missed_gold"),
            _row("p3", "raw", False, "D"),
            _row("p3", "hipporag_baseline", False, "D"),
            _row("p3", "assumption_agent_recursive_verify", False, "D", failure_bucket="candidate_generation_missed_gold"),
        ]

        report = build_hle_fast_policy_promotion_report_from_rows(
            rows,
            min_unseen_triads=3,
        )

        self.assertEqual(report["report_version"], PROMOTION_REPORT_VERSION)
        self.assertEqual(report["triad_metrics"]["complete_triad_count"], 3)
        agent_vs_control = report["triad_metrics"]["agent_vs_best_control"]["gpt-5.4-mini"]
        self.assertFalse(agent_vs_control["passed"])
        self.assertEqual(agent_vs_control["agent_minus_best_control"], -0.3333)
        self.assertIn("agent_below_best_control:gpt-5.4-mini", report["promotion_gate"]["blockers"])
        self.assertIn("agent_no_fallback_present", report["promotion_gate"]["blockers"])
        self.assertIn("deterministic_option_coverage_and_required_term_source_bundle", report["agent_mined_policy_summary"]["hypothesis_actions"])
        self.assertFalse(report["raw_content_persisted"])
        self.assertNotIn("RAW QUESTION", str(report))

    def test_selector_simulation_detects_verified_else_raw_gain(self):
        rows = [
            _row("p1", "raw", True, "A"),
            _row("p1", "hipporag_baseline", False, "B"),
            _row("p1", "assumption_agent_recursive_verify", False, "B", gate="abstained"),
            _row("p2", "raw", True, "C"),
            _row("p2", "hipporag_baseline", True, "C"),
            _row("p2", "assumption_agent_recursive_verify", True, "C", gate="allowed"),
        ]

        report = build_hle_fast_policy_promotion_report_from_rows(
            rows,
            min_unseen_triads=2,
        )

        simulation = report["selector_policy_simulation"]
        self.assertEqual(simulation["policy_table"]["agent_current"]["correct"], 1)
        self.assertEqual(simulation["policy_table"]["verified_else_raw"]["correct"], 2)
        self.assertEqual(simulation["best_policy"], "verified_else_raw")
        self.assertGreater(simulation["best_delta_vs_agent_current"], 0)
        self.assertNotIn("no_selector_policy_gain", report["promotion_gate"]["blockers"])


def _row(
    problem_id: str,
    variant: str,
    correct: bool,
    prediction_hash: str,
    *,
    gate: str = "unknown",
    failure_bucket: str = "unknown_failure",
) -> dict:
    row = {
        "problem_id_hash": problem_id,
        "question_hash": f"q-{problem_id}",
        "answer_hash": f"a-{problem_id}",
        "model": "gpt-5.4-mini",
        "variant": variant,
        "category": "Science",
        "raw_subject": "Physics",
        "answer_type": "multipleChoice",
        "correct": correct,
        "prediction_hash": prediction_hash,
        "prediction_text_persisted": False,
        "raw_question_persisted": False,
        "gold_answer_persisted": False,
        "call_metadata": {"variant_watchdog": {"model_call_count": 1}},
        "error": None,
        "question": "RAW QUESTION SHOULD NOT SURVIVE",
    }
    if variant == "assumption_agent_recursive_verify":
        row["component_efficacy"] = {
            "flags": {
                "candidate_generation_missed_gold": failure_bucket == "candidate_generation_missed_gold",
                "verified_or_abstain_no_fallback": gate == "no_fallback",
            },
            "selection": {
                "selection_method": "verified_or_abstain_direct_fallback",
                "verified_or_abstain_gate": {"status": gate},
            },
        }
    else:
        row["component_efficacy"] = {}
    return row


if __name__ == "__main__":
    unittest.main()
