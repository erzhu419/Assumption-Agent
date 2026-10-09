"""Exercise the actual study-to-agent operator wiring without a model call."""

from random import Random

from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from scripts.run_online_inquiry_llm import run_study
from scripts.run_temporal_operator_llm import laboratory, protocol


class ScriptedOperatorUser:
    def __init__(self, workspace):
        self.random = Random(71)

    def next_action(self, request):
        assert "operator:'history'" in request["task_interface"]
        step = len(request["events"])
        if step == 0:
            length = request["capabilities"]["max_probe_length"]
            count = 12 if length == 3 else 4
            action, arguments = "query", {"actions": [
                {"kind": "trajectory", "channels": [], "sequence": [
                    {"u": self.random.uniform(-1., 1.)} for _ in range(length)]}
                for _ in range(count)]}
        elif step == 1:
            action, arguments = "run", {"operator": "history", "path": "model.py"}
        else:
            action, arguments = "finish", {"model": "model.py"}
        return {"action": action, "arguments": arguments,
                "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}


def test_study_dispatches_shared_operators_and_evaluates_both_scopes(tmp_path):
    plan = protocol(cases=1, arms=["code_react"])
    summary = run_study(plan, tmp_path, provider_factory=ScriptedOperatorUser,
                        lab_factory=laboratory, agent_class=TemporalOperatorAgent)
    assert summary["evidence_kind"] == "interface_test_not_llm_performance"
    assert summary["arms"]["code_react"]["confirmed_and_correct"] == 2
    assert summary["arms"]["code_react"]["costs_including_source"]["model_calls"] == 6
