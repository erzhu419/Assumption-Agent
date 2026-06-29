import unittest

from assumption_os.operator_failure_diagnostics import (
    classify_operator_failure,
    summarize_operator_failure_taxonomy,
)


class OperatorFailureDiagnosticsTest(unittest.TestCase):
    def test_correct_row_is_not_operator_failure(self) -> None:
        result = classify_operator_failure({
            "kind": "assumption_agent_recursive_verify",
            "final_correct": True,
            "flags": {"final_correct": True},
        })

        self.assertEqual(result["category"], "NotOperatorFailure")

    def test_semantic_failure_after_slot_pass_is_verifier_false_positive(self) -> None:
        result = classify_operator_failure({
            "kind": "assumption_agent_recursive_verify",
            "final_correct": False,
            "flags": {
                "operator_specs_requested": True,
                "operator_specs_activated": True,
                "operator_application_applied": True,
                "operator_application_passed": True,
            },
            "operator_specs": {
                "status": "activated",
                "operator_policy": {
                    "p_trigger": 0.82,
                    "p_harm": 0.18,
                    "selected_operator_family_ids": ["O5_evidence_grounding"],
                },
            },
            "operator_application_verifier": {
                "status": "activated",
                "pass": True,
                "operator_application_applied": True,
                "operator_changed_candidate": True,
                "semantic_fidelity": {"semantic_pass": False},
            },
        })

        self.assertEqual(result["category"], "VerifierFalsePositive")

    def test_compiled_but_unapplied_operator_is_deferred(self) -> None:
        result = classify_operator_failure({
            "kind": "assumption_agent_recursive_verify",
            "final_correct": False,
            "flags": {
                "operator_specs_requested": True,
                "operator_specs_activated": True,
                "operator_application_applied": False,
            },
            "operator_specs": {
                "status": "activated",
                "operator_policy": {"p_trigger": 0.7, "p_harm": 0.2},
            },
            "operator_application_verifier": {
                "status": "deferred_not_applied",
                "operator_application_applied": False,
            },
        })

        self.assertEqual(result["category"], "OperatorDeferred")

    def test_source_verifier_exhaustion_is_source_evidence_missing(self) -> None:
        result = classify_operator_failure({
            "kind": "assumption_agent_recursive_verify",
            "final_correct": False,
            "flags": {
                "operator_specs_requested": True,
                "operator_specs_activated": False,
                "operator_specs_blocked": True,
                "gold_option_source_verifier_direct_source_insufficient": True,
                "gold_option_source_verifier_indirect_or_generic": True,
                "source_supported_evidence_candidate": False,
            },
            "operator_specs": {
                "status": "skipped",
                "reason": "generic_harness_graph_context_only",
            },
            "mc_option_claim_evidence_verifier": {
                "status": "blocked_source_grounded_claim_verifier",
                "source_verifier_attempt_count": 4,
                "source_verifier_accepted_attempt_count": 0,
                "source_verifier_direct_high_confidence_count": 0,
                "span_directness_verifier_status": "blocked_not_direct_relation",
            },
        })

        self.assertEqual(result["category"], "SourceEvidenceMissing")

    def test_summary_counts_categories(self) -> None:
        payload = summarize_operator_failure_taxonomy([
            {"kind": "assumption_agent_recursive_verify", "final_correct": True},
            {"kind": "raw", "final_correct": False},
        ])

        self.assertEqual(payload["category_counts"]["NotOperatorFailure"], 1)
        self.assertEqual(payload["category_counts"]["BaseGenerationNoise"], 1)
        self.assertFalse(payload["raw_content_persisted"])


if __name__ == "__main__":
    unittest.main()
