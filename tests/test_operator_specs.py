import unittest

from assumption_os.application_fidelity import audit_answer_application, audit_operator_application
from assumption_os.operator_specs import (
    build_operator_specs,
    format_operator_specs,
    operator_gate_decision,
    operator_spec_from_node,
    operator_trace_summary,
)
from assumption_os.schema import AssumptionNode, AssumptionType


class OperatorSpecsTest(unittest.TestCase):
    def test_explicit_operator_payload_is_preserved(self) -> None:
        node = AssumptionNode(
            id="op_explicit",
            type=AssumptionType.METHOD,
            claim="use an explicit execution policy",
            payload={
                "operator_spec": {
                    "trigger_conditions": ["route is ambiguous"],
                    "execution_steps": ["compare the two routes"],
                    "required_output_slots": ["chosen_route", "rejection_reason"],
                    "negative_controls": ["do not vote by popularity"],
                    "verifier_checks": ["route has literal support"],
                    "fallback_policy": "abstain if neither route is supported",
                }
            },
            confidence=0.82,
        )

        spec = operator_spec_from_node(node)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.source_id, "op_explicit")
        self.assertEqual(spec.execution_steps, ["compare the two routes"])
        self.assertEqual(spec.required_output_slots, ["chosen_route", "rejection_reason"])
        self.assertEqual(spec.fallback_policy, "abstain if neither route is supported")

    def test_controlled_variable_claim_gets_executable_slots(self) -> None:
        node = AssumptionNode(
            id="strategy_S01",
            type=AssumptionType.METHOD,
            claim="Use controlled variables for causal attribution before changing the system.",
            context_conditions=["multiple causes explain the regression"],
            confidence=0.7,
        )

        spec = operator_spec_from_node(node)

        self.assertIsNotNone(spec)
        self.assertIn("variable_or_cause_changed", spec.required_output_slots)
        self.assertIn("control_or_baseline", spec.required_output_slots)
        self.assertTrue(any("held" in step.lower() or "fixed" in step.lower() for step in spec.execution_steps))

    def test_format_operator_specs_marks_constraints_as_not_background_context(self) -> None:
        nodes = [
            AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled variables for causal attribution.",
                confidence=0.7,
            ),
            AssumptionNode(
                id="residual_noise",
                type=AssumptionType.RESIDUAL,
                claim="historical residual only",
            ),
        ]

        specs = build_operator_specs(nodes, max_specs=2)
        text = format_operator_specs(specs)
        summary = operator_trace_summary(specs)

        self.assertEqual(summary["operator_source_ids"], ["strategy_S01"])
        self.assertIn("Operatorized Assumption Constraints", text)
        self.assertIn("not background context", text)
        self.assertIn("Required answer slots", text)
        self.assertNotIn("residual_noise", text)

    def test_operator_gate_selects_domains_without_polluting_retrieval(self) -> None:
        enabled = operator_gate_decision("daily_life", allowed_domains={"daily_life"})
        blocked = operator_gate_decision("business", allowed_domains={"daily_life"})
        skipped = operator_gate_decision(
            "software_engineering",
            allowed_domains={"all"},
            skipped_domains={"software_engineering"},
        )

        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.status, "enabled")
        self.assertFalse(blocked.enabled)
        self.assertEqual(blocked.reason, "domain_not_in_operator_allow_list")
        self.assertFalse(skipped.enabled)
        self.assertEqual(skipped.reason, "domain_in_operator_skip_list")

    def test_application_fidelity_passes_when_required_slots_are_filled(self) -> None:
        node = AssumptionNode(
            id="strategy_S01",
            type=AssumptionType.METHOD,
            claim="Use controlled variables for causal attribution before changing the system.",
            confidence=0.7,
        )
        spec = operator_spec_from_node(node)
        answer = (
            "先只改变一个因素：把A渠道预算提高10%，其他城市、投放时段和素材保持固定。"
            "用B城市同预算组作对照基线，观察获客成本、核销率和复购指标。"
            "如果成本低于阈值且核销率提升，再加码；否则停止并回到基线方案。"
        )

        audit = audit_operator_application(answer, spec)

        self.assertTrue(audit.used)
        self.assertGreaterEqual(audit.slot_completion_rate, 0.8)
        self.assertFalse(audit.decorative)

    def test_application_fidelity_flags_decorative_operator_mentions(self) -> None:
        node = AssumptionNode(
            id="strategy_S01",
            type=AssumptionType.METHOD,
            claim="Use controlled variables for causal attribution before changing the system.",
            confidence=0.7,
        )
        spec = operator_spec_from_node(node)
        answer = "这里要使用控制变量法，综合分析各种可能因素，然后给出更稳妥的建议。"

        audit = audit_answer_application(answer, [spec])

        self.assertFalse(audit["pass"])
        self.assertEqual(audit["decorative_use_count"], 1)
        self.assertEqual(audit["used_assumption_ids"], [])


if __name__ == "__main__":
    unittest.main()
