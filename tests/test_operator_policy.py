import unittest

from assumption_os.operator_policy import classify_operator_families, decide_operator_policy
from assumption_os.operator_specs import OperatorSpec


class OperatorPolicyTest(unittest.TestCase):
    def test_daily_life_causal_operator_is_selected_with_required_strength(self) -> None:
        spec = OperatorSpec(
            source_id="framework_dependency_aware_controlled_intervention",
            source_type="framework",
            source_claim="Use controlled variable and dependency-aware intervention reasoning.",
            trigger_conditions=["causal attribution with control or contrast"],
            execution_steps=["identify cause", "hold baseline fixed"],
            required_output_slots=[
                "candidate_cause_or_intervention",
                "outcome_or_metric",
                "control_or_contrast",
                "limiting_case",
            ],
            negative_controls=[],
            verifier_checks=["selected option preserves the control contrast"],
            fallback_policy="ignore if not causal",
            confidence=0.86,
        )

        decision = decide_operator_policy(
            problem_text="Which treatment caused the observed effect compared with the control baseline?",
            specs=[spec],
            domain="daily_life",
        )

        self.assertTrue(decision.enabled)
        self.assertIn(spec.source_id, decision.selected_operator_ids)
        self.assertIn("O1_causal_control_variable", decision.selected_operator_family_ids)
        self.assertIn(decision.operator_strength, {"required", "strict", "repair"})
        self.assertGreater(decision.p_trigger, decision.p_harm)
        self.assertIsNone(decision.abstain_reason)

    def test_caution_domain_generic_operator_abstains(self) -> None:
        spec = OperatorSpec(
            source_id="generic_advice",
            source_type="strategy",
            source_claim="Apply the assumption as a useful constraint.",
            trigger_conditions=["the assumption may help"],
            execution_steps=["state trigger", "apply constraint"],
            required_output_slots=["trigger", "applied_constraint", "evidence_or_boundary"],
            negative_controls=["do not merely restate the assumption"],
            verifier_checks=["answer changes because of the assumption"],
            fallback_policy="ignore if weak",
            confidence=0.2,
        )

        decision = decide_operator_policy(
            problem_text="Which deployment option is best for this software rollout?",
            specs=[spec],
            domain="software_engineering",
        )

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.operator_strength, "off")
        self.assertEqual(decision.abstain_reason, "policy_trigger_below_floor")
        self.assertEqual(classify_operator_families(spec), ["generic_operator"])

    def test_empty_specs_returns_no_operator_decision(self) -> None:
        decision = decide_operator_policy(problem_text="anything", specs=[], domain="hle_general")

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.abstain_reason, "no_operator_specs")
        self.assertEqual(decision.selected_operator_ids, [])


if __name__ == "__main__":
    unittest.main()
