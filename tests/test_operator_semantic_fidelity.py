import unittest

from assumption_os.operator_semantic_fidelity import audit_answer_semantic_fidelity
from assumption_os.operator_specs import OperatorSpec


class OperatorSemanticFidelityTest(unittest.TestCase):
    def _causal_spec(self) -> OperatorSpec:
        return OperatorSpec(
            source_id="framework_dependency_aware_controlled_intervention",
            source_type="framework",
            source_claim="Use controlled variable reasoning.",
            trigger_conditions=["causal attribution with a baseline"],
            execution_steps=["change one cause", "hold baseline fixed", "measure outcome"],
            required_output_slots=[
                "variable_or_cause_changed",
                "control_or_baseline",
                "observed_metric",
            ],
            negative_controls=[],
            verifier_checks=[],
            fallback_policy="ignore if not causal",
            confidence=0.8,
        )

    def test_surface_slot_mentions_without_problem_anchors_fail(self) -> None:
        audit = audit_answer_semantic_fidelity(
            problem_text="Which intervention lowered blood pressure compared with placebo?",
            answer_text="Use controlled variables, a baseline, and a metric, then choose carefully.",
            specs=[self._causal_spec()],
            decision_changed=True,
        )

        self.assertFalse(audit["semantic_pass"])
        self.assertLess(audit["slot_substance_rate"], 0.6)
        self.assertFalse(audit["raw_content_persisted"])

    def test_problem_relevant_slots_and_changed_decision_pass(self) -> None:
        audit = audit_answer_semantic_fidelity(
            problem_text="Which intervention lowered blood pressure compared with placebo?",
            answer_text=(
                "Choose B: change the treatment dose only, keep the placebo baseline fixed, "
                "and use blood pressure as the observed metric."
            ),
            specs=[self._causal_spec()],
            decision_changed=True,
        )

        self.assertTrue(audit["semantic_pass"])
        self.assertGreaterEqual(audit["slot_substance_rate"], 0.6)
        self.assertGreaterEqual(audit["problem_relevance_rate"], 0.5)

    def test_no_decision_change_blocks_semantic_pass_when_observed(self) -> None:
        audit = audit_answer_semantic_fidelity(
            problem_text="Which intervention lowered blood pressure compared with placebo?",
            answer_text=(
                "Choose B: change the treatment dose only, keep the placebo baseline fixed, "
                "and use blood pressure as the observed metric."
            ),
            specs=[self._causal_spec()],
            decision_changed=False,
        )

        self.assertFalse(audit["semantic_pass"])
        self.assertTrue(audit["decision_change_observed"])
        self.assertFalse(audit["decision_changed"])


if __name__ == "__main__":
    unittest.main()
