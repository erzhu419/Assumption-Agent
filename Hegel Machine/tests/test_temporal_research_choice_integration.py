"""Execute the chooser's returned purchases through the existing agent tools."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.instrument_method_lab import InstrumentMethodLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import (
    INSTRUMENT_METHODS_PATH, TemporalOperatorAgent,
)
from hegel_machine.cognition import temporal_research_choice as choice_module
from scripts.run_temporal_legacy_method_llm import prepare_episode


def config():
    return InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
        compute_budget=100000, execution_seconds=30.)


def execute_next(agent, lab, result):
    action = result["next_action"]
    return agent._execute(action["action"], action["arguments"], lab)


def method_book(agent):
    return json.loads((agent.workspace / INSTRUMENT_METHODS_PATH).read_text())


def test_response_choice_executes_old_method_once_then_confirms_and_passes_terminal(tmp_path, monkeypatch):
    # Generate the prior through the same old paid fixture and method learner;
    # the test remains runnable without large historical run artifacts.
    source, source_lab = prepare_episode(tmp_path / "source", dict(case="event", stage=0,
        seed=1154, center=.21, half_width=1e-4, response_gain=2.), config())
    proposal = source._execute("run", {"operator": "response_disagreement",
        "hypotheses": "hypotheses.json", "instrument": "instrument.json"}, source_lab)
    source._execute("query", {"actions": [proposal["action"]], "after_observation": [{
        "operator": "assess_response", "experiment": proposal["experiment"], "records": "acquired"}]}, source_lab)
    old_book = method_book(source)
    assert len(old_book) == 1 and len(old_book[0]["support"]) == 1
    agent, lab = prepare_episode(tmp_path / "target", dict(case="event", stage=4,
        seed=1376, center=.37, half_width=1e-4, response_gain=1.), config(), book=old_book)
    assert agent.costs["observation_cost"] == 72
    assert all(row["target"] == 0. for record in agent.observations for row in record["rows"])
    calls, original = [], choice_module.plan_response_experiment

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(choice_module, "plan_response_experiment", counted)
    arguments = {"operator": "research_choice", "hypotheses": "hypotheses.json",
                 "instrument": "instrument.json"}
    before_costs, before_lab = deepcopy(agent.costs), deepcopy(lab.costs)
    choice = agent._execute("run", arguments, lab)
    assert choice["diagnosis"] == "response_ambiguity"
    assert choice["proposal_source"] == "learned_method"
    assert choice["fresh_synthesis_skipped"]
    assert choice["proposal"]["method_id"] == old_book[0]["method_id"]
    assert choice["next_action"]["action"] == "query"
    assert len(choice["action"]["sequence"]) == 4
    assert len(calls) == 1 and lab.costs == before_lab
    assert agent.costs["compute_cost"] - before_costs["compute_cost"] == choice["compute_cost"]
    assert method_book(agent) == old_book
    saved = json.loads((agent.workspace / choice["experiment"]).read_text())
    assert saved["first_new_record"] == 3
    assert saved["action"] == choice["action"]

    before_compute = agent.costs["compute_cost"]
    acquired = execute_next(agent, lab, choice)
    assert acquired["record_indices"] == [3]
    assessed = acquired["after_observation_results"][0]
    assert assessed["status"] == "distinguished"
    assert assessed["method_feedback"]["action"] == "use_program_experiment_method"
    assert assessed["method_feedback"]["cross_task"]
    assert len(method_book(agent)[0]["support"]) == 2
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 76
    assert agent.costs["measurement_cost"] == lab.costs["measurement_cost"] == 228
    assert agent.costs["compute_cost"] - before_compute == assessed["compute_cost"]
    assert len(calls) == 1

    retained = agent._execute("run", arguments, lab)
    assert retained["diagnosis"] == "one_response_survives_paid_evidence"
    assert retained["next_action"] == {"action": "finish", "arguments": {"model": "models/candidate_02.py"}}
    assert len(calls) == 1
    assert len(list((agent.workspace / "response_experiments").glob("*.json"))) == 1
    finished = execute_next(agent, lab, retained)
    assert finished["fresh_confirmation"]["passed"]
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 124
    terminal = evaluate_python_model(agent.workspace, agent.model,
        lab.evaluation_rows(length=128, seed=1000011), timeout_seconds=10.)
    assert terminal["returncode"] == 0 and terminal["mse"] <= 1e-10
    assert len(method_book(agent)[0]["support"]) == 2
    assert agent.costs["model_calls"] == 0


@pytest.mark.parametrize("mode", ["finite", "recurrent"])
def test_instrument_choice_saves_plan_consumable_by_existing_paid_assessor(tmp_path, mode):
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False, config=config())
    lab = TemporalInquiryLab(task_index=1)
    lab._laboratory = InstrumentMethodLab(0, 157, 739, mode)
    rng = Random(157)
    actions = [{"kind": "trajectory", "channels": ["meter"],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(3)]} for _ in range(7)]
    agent._execute("query", {"actions": actions[:6]}, lab)
    agent._execute("run", {"operator": "instrument", "path": "instrument.json"}, lab)
    fresh = agent._execute("query", {"actions": actions[6:]}, lab)
    assessed = agent._execute("run", {"operator": "assess_instrument",
        "instrument": "instrument.json", "records": fresh["record_indices"]}, lab)
    assert assessed["status"] == "ambiguous_on_measured_trajectory"
    # This is the old fixture's legal 3->4 horizon expansion, not a new plant.
    lab._past_costs = lab.costs
    lab._laboratory = InstrumentMethodLab(1, 157, 739, mode)
    agent._save("hypotheses.json", json.dumps([{"hypothesis_id": "null",
        "program": Program("const", value=0.).to_dict()}]))
    before = deepcopy(agent.costs), deepcopy(lab.costs)
    choice = agent._execute("run", {"operator": "research_choice",
        "hypotheses": "hypotheses.json", "instrument": "instrument.json"}, lab)
    assert choice["diagnosis"] == "instrument_ambiguity"
    assert len(choice["action"]["sequence"]) == 4
    assert lab.costs == before[1]
    assert agent.costs["compute_cost"] - before[0]["compute_cost"] == choice["compute_cost"]
    plan = json.loads((agent.workspace / choice["experiment"]).read_text())
    assert plan["instrument"] == "instrument.json"
    assert plan["first_new_record"] == 7
    assert plan["instrument_condition"]["programs"]["meter"]
    hypotheses = len(plan["instrument_condition"]["programs"]["meter"])
    assert choice["decision"]["quoted_compute_cost"] == 4 * hypotheses + 21 + 4 + 48
    before_compute = agent.costs["compute_cost"]
    acquired = execute_next(agent, lab, choice)
    assert acquired["record_indices"] == [7]
    outcome = acquired["after_observation_results"][0]
    assert outcome["status"] == "supported_on_measured_trajectory"
    assert outcome["method_feedback"]["action"] == "learn_instrument_experiment_method"
    assert agent.costs["compute_cost"] - before_compute == outcome["compute_cost"]
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"] == 25
    assert agent.costs["measurement_cost"] == lab.costs["measurement_cost"] == 75
    assert len(method_book(agent)[0]["support"]) == 1
    assert len(list((agent.workspace / "instrument_experiments").glob("*.json"))) == 1
