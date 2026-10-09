"""The advice ablation changes visible feedback and measures actual next actions."""

from copy import deepcopy
import json
from random import Random
from types import SimpleNamespace

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from scripts.run_python_advice_ablation import RecordingAdviceAgent, NoAdviceAgent, advice_evidence


def prepared(path, cls):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="recurrent")
    agent = cls(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=384, compute_budget=360000,
                               confirmation_rows=24, execution_seconds=120.))
    rng = Random(71)
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]} for _ in range(3)]}, lab)
    agent._save("model.py", "_mean=0.\ndef fit(rows):\n global _mean\n"
        " _mean=sum(row['target'] for row in rows)/len(rows)\ndef predict(row):\n return _mean\n")
    agent._advice_learning = True
    return agent, lab


def execute(agent, lab, action):
    result = agent._execute(action["action"], action["arguments"], lab)
    agent.events.append({"step": len(agent.events), **deepcopy(action), "feedback": result})
    return result


def test_hidden_advice_stays_private_with_identical_real_actions_and_paid_costs(tmp_path):
    visible, lab = prepared(tmp_path / "visible", RecordingAdviceAgent)
    hidden, hidden_lab = prepared(tmp_path / "hidden", NoAdviceAgent)
    assert visible.common_interface == hidden.common_interface
    action = {"action": "evaluate", "arguments": {"model": "model.py"}}
    # Failed evaluation, failed refit, then the recommended real metered query
    # exercise both ordinary feedback and nested instrument/assessment feedback.
    for _ in range(3):
        shown = execute(visible, lab, action)
        withheld = execute(hidden, hidden_lab, action)
        assert "research_advice" in shown
        assert "research_advice" not in json.dumps(withheld)
        assert {k: v for k, v in visible.costs.items() if k != "python_execution_seconds"} == {
            k: v for k, v in hidden.costs.items() if k != "python_execution_seconds"}
        assert lab.costs == hidden_lab.costs
        action = shown["research_advice"]["next_action"]
    assert len(hidden._advice_ledger) == len(visible._advice_ledger) >= 5
    assert hidden._advice_ledger[-1]["trigger"]["action"] == "query"
    assert any(item["trigger"]["operator"] == "assess_instrument" for item in hidden._advice_ledger)
    assert "research_advice" not in json.dumps(hidden.events)
    assert "research_advice" not in json.dumps(hidden.operator_results)
    hidden._prepare_program_state()
    state = (hidden.workspace / "research_state.json").read_text()
    assert "research_advice" not in state and "_advice_ledger" not in state
    assert all("_advice_ledger" not in content for content in hidden.files.values())
    assert hidden.costs["model_calls"] == visible.costs["model_calls"] == 0


def test_only_final_advice_per_event_matches_the_next_exact_action():
    refit = {"action": "run", "arguments": {"operator": "refit", "model": "old.py"}}
    response = {"action": "run", "arguments": {"operator": "response", "instrument": "meter.json"}}
    finish = {"action": "finish", "arguments": {"model": "confirmed.py"}}
    ledger = [{"event_index": step, "learning": True,
               "trigger": {"action": "query", "operator": operator},
               "advice": {"diagnosis": operator, "next_action": action}}
              for step, operator, action in [(0, "instrument", refit),
                                             (0, "assess_instrument", response),
                                             (1, "response", finish)]]
    events = [{"action": "query", "arguments": {}}, response,
              {"action": "finish", "arguments": {"model": "different.py"}}]
    agent = SimpleNamespace(_advice_ledger=deepcopy(ledger))
    visible = advice_evidence(agent, events, visible=True)
    hidden = advice_evidence(agent, events, visible=False)
    assert visible["rendered_opportunities"] == 3
    assert visible["superseded_opportunities"] == 1
    assert visible["actionable_final_opportunities"] == visible["opportunities_with_next_event"] == 2
    assert visible["visible_next_action_matches"] == 1
    assert visible["counterfactual_next_action_matches"] == 0
    assert hidden["visible_next_action_matches"] == 0
    assert hidden["counterfactual_next_action_matches"] == 1
    assert visible["advice_ledger"][0]["superseded"]
    assert not visible["advice_ledger"][2]["next_action_matches"]
    assert agent._advice_ledger == ledger
