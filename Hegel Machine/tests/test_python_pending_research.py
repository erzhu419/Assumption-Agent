"""An open instrument question survives unrelated, ordinary Python work."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


class SilentAdviceAgent(TemporalOperatorAgent):
    def _render_research_advice(self, advice):
        return None


def open_instrument_question(path, *, cls=TemporalOperatorAgent, mode="recurrent"):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode=mode)
    agent = cls(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=384, compute_budget=360000,
            confirmation_rows=24, execution_seconds=120.))
    rng = Random(71)
    agent._execute("query", {"actions": [
        {"kind": "trajectory", "channels": [],
         "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}
        for _ in range(3)]}, lab)
    agent._save("mean.py", "_mean = 0.\ndef fit(rows):\n global _mean\n"
        " _mean = sum(row['target'] for row in rows) / len(rows)\n"
        "def predict(row):\n return _mean\n")
    evaluated = agent._execute("evaluate", {"model": "mean.py"}, lab)
    assert evaluated["mse"] > 1e-4
    rng = Random(89)
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": ["meter"],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}]}, lab)
    fitted = agent._execute("run", {"operator": "instrument", "records": [3],
        "path": "instrument.json", "compute_limit": 100000}, lab)
    assert fitted["status"] == "requires_fresh_validation"
    return agent, lab, fitted


def state(agent):
    before = deepcopy(agent.costs)
    agent._prepare_program_state()
    assert agent.costs == before
    return json.loads((agent.workspace / "research_state.json").read_text())


def execute_next(agent, lab, feedback):
    action = feedback["research_advice"]["next_action"]
    return agent._execute(action["action"], action["arguments"], lab)


@pytest.mark.parametrize("action,arguments", [
    ("run", {"code": "print('checking an alternative history basis')"}),
    ("write", {"path": "alternative.py", "code": "def predict(row):\n return 0.\n"}),
    ("run", {"operator": "history", "base_lags": 3, "path": "history.py"}),
])
def test_unfinished_discrimination_survives_ordinary_actions(tmp_path, action, arguments):
    agent, lab, fitted = open_instrument_question(tmp_path)
    assert fitted["research_advice"]["next_action"]["arguments"]["operator"] == "disagreement"
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    result = agent._execute(action, arguments, lab)
    assert "error" not in result
    next_action = result["research_advice"]["next_action"]
    assert next_action["arguments"]["operator"] == "disagreement"
    assert next_action["arguments"]["instrument"] == "instrument.json"
    assert lab.costs == lab_before
    expected = result["compute_cost"] if arguments.get("operator") == "history" else 0
    assert agent.costs["compute_cost"] - before["compute_cost"] == expected
    snapshot = state(agent)
    assert snapshot["research_advice"] == result["research_advice"]
    assert "research_advice" not in json.dumps(snapshot["operator_results"])
    assert agent.model is None and not agent.confirmations


def test_saved_disagreement_plan_survives_python_then_real_validation_advances_it(tmp_path):
    agent, lab, fitted = open_instrument_question(tmp_path)
    planned = execute_next(agent, lab, fitted)
    assert planned["operator"] == "disagreement" and planned["experiment"]
    offered = planned["research_advice"]["next_action"]
    assert offered["action"] == "query"
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    paused = agent._execute("run", {"code": "print('unrelated numerical work')"}, lab)
    assert paused["research_advice"]["next_action"] == offered
    assert lab.costs == lab_before
    assert agent.costs["compute_cost"] == before["compute_cost"]
    assessed = execute_next(agent, lab, paused)
    assert any(item.get("status") == "supported_on_measured_trajectory"
               for item in assessed["after_observation_results"])
    assert assessed["research_advice"]["next_action"]["arguments"]["operator"] == "response"
    resumed = agent._execute("run", {"code": "print('return to the parent question')"}, lab)
    assert resumed["research_advice"]["next_action"]["arguments"]["operator"] == "response"
    response = execute_next(agent, lab, resumed)
    assert any(candidate["validation_mse"] < 1e-10 for candidate in response["candidates"])
    later = agent._execute("write", {"path": "notes.txt", "code": "Response fit available."}, lab)
    assert later["research_advice"]["next_action"]["action"] == "evaluate"
    assert sum(row["operator"] == "disagreement" for row in agent.operator_results) == 1


def test_independent_correct_predictor_takes_priority_and_finishing_clears_pending_advice(tmp_path):
    agent, lab, _ = open_instrument_question(tmp_path, mode="finite2")
    learned = agent._execute("run", {"operator": "history", "base_lags": 3,
        "path": "independent.py"}, lab)
    evaluated = agent._execute("evaluate", {"model": learned["path"]}, lab)
    assert evaluated["mse"] < 1e-10
    result = agent._execute("run", {"code": "print('retain the independently fitted response')"}, lab)
    action = result["research_advice"]["next_action"]
    assert action == {"action": "finish", "arguments": {"model": evaluated["evaluated_model"]["model"]}}
    confirmed = agent._execute(action["action"], action["arguments"], lab)
    assert confirmed["fresh_confirmation"]["passed"]
    assert "research_advice" not in confirmed
    assert "research_advice" not in state(agent)
    after = agent._execute("run", {"code": "print('finished')"}, lab)
    assert "research_advice" not in after
    assert "research_advice" not in state(agent)


def test_validated_refit_stays_next_for_evaluation_after_ordinary_python(tmp_path):
    agent, lab, _ = open_instrument_question(tmp_path, mode="finite2")
    learned = agent._execute("run", {"operator": "history", "base_lags": 3,
        "path": "independent.py"}, lab)
    refitted = agent._execute("run", {"operator": "refit", "model": learned["path"],
        "path": "shared/refitted"}, lab)
    assert refitted["status"] == "refit_supported_on_validation"
    assert refitted["final_fitting_rows"] == len(agent.rows)
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    paused = agent._execute("run", {"code": "print('review the refit result')"}, lab)
    expected = {"action": "evaluate", "arguments": {"model": refitted["path"]}}
    assert paused["research_advice"]["next_action"] == expected
    assert state(agent)["research_advice"]["next_action"] == expected
    assert agent.costs["compute_cost"] == before["compute_cost"]
    assert lab.costs == lab_before
    evaluated = execute_next(agent, lab, paused)
    assert evaluated["mse"] < 1e-10
    assert evaluated["research_advice"]["next_action"]["action"] == "finish"
    confirmed = execute_next(agent, lab, evaluated)
    assert confirmed["fresh_confirmation"]["passed"]


def test_hidden_pending_advice_is_absent_from_state_and_operator_results(tmp_path):
    visible, visible_lab, _ = open_instrument_question(tmp_path / "visible")
    silent, silent_lab, hidden = open_instrument_question(tmp_path / "silent", cls=SilentAdviceAgent)
    assert "research_advice" not in hidden
    for action, arguments in [("run", {"code": "print('another fitting calculation')"}),
                              ("write", {"path": "notes.txt", "code": "temporary notes"})]:
        normal = visible._execute(action, arguments, visible_lab)
        hidden = silent._execute(action, arguments, silent_lab)
        assert "research_advice" in normal and "research_advice" not in hidden
        normal_state, hidden_state = state(visible), state(silent)
        assert "research_advice" in normal_state and "research_advice" not in hidden_state
        assert "research_advice" not in json.dumps(hidden_state["operator_results"])
        assert {key: value for key, value in visible.costs.items() if key != "python_execution_seconds"} == {
            key: value for key, value in silent.costs.items() if key != "python_execution_seconds"}
        assert visible_lab.costs == silent_lab.costs
    assert visible.common_interface == silent.common_interface
