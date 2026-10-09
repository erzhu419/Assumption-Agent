"""The existing procedure executor can follow research in the same paid session."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from random import Random
import shutil

import pytest

from hegel_machine.cognition.code_agent import archive_confirmed_python_model, evaluate_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_paired_method import (
    DEFAULT_RESEARCH_PROCEDURE, ProcedureConsumerAgent,
)


PROCEDURE = "shared/research_procedure.py"


def configuration():
    return InquiryLLMConfig(observation_budget=384, compute_budget=360000,
        confirmation_rows=24, execution_seconds=120.)


def purchase(agent, lab, seed=71):
    rng = Random(seed)
    return agent._execute("query", {"actions": [
        {"kind": "trajectory", "channels": [],
         "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}
        for _ in range(3)]}, lab)


@pytest.fixture
def source_archive(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="finite2")
    agent = ProcedureConsumerAgent(None, tmp_path / "source", arm="code_react",
        source_task=False, config=configuration())
    purchase(agent, lab)
    learned = agent._execute("run", {"operator": "history", "base_lags": 3,
        "path": "source_model.py"}, lab)
    assert learned["base"]["heldout_mse"] < 1e-10
    confirmed = agent._execute("finish", {"model": learned["path"]}, lab)
    assert confirmed["fresh_confirmation"]["passed"]
    return archive_confirmed_python_model(agent, tmp_path / "bank" / "_evaluated_model")


def consumer(path, archive, *, mode="finite2", changed=True, paid=True,
             agent_class=ProcedureConsumerAgent):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910079 if changed and mode == "finite2" else 910061,
        instrument_mode=mode)
    agent = agent_class(None, path, arm="code_react", source_task=False,
        config=configuration())
    destination = agent.workspace / "shared/old/_evaluated_model"
    shutil.copytree(archive["snapshot_workspace"], destination)
    model = str(Path("shared/old/_evaluated_model") / archive["inner_model"])
    agent._save(PROCEDURE, DEFAULT_RESEARCH_PROCEDURE)
    if paid:
        purchase(agent, lab, seed=79)
        scored = agent._execute("evaluate", {"model": model}, lab)
        if changed or mode == "recurrent":
            assert scored["mse"] > 1e-4
    return agent, lab, model


def execute_procedure(agent, lab, **arguments):
    return agent._execute("run", {"operator": "procedure", "path": PROCEDURE,
        "max_steps": 12, **arguments}, lab)


def assert_no_llm_activity(agent, previous_events):
    assert agent.events == previous_events
    assert agent.costs["model_calls"] == agent.costs["reported_tokens"] == 0


def assert_terminal(agent, lab):
    terminal = evaluate_python_model(agent.workspace, agent.model,
        lab.evaluation_rows(length=128, seed=1000291), timeout_seconds=10.)
    assert terminal["returncode"] == 0 and terminal["mse"] < 1e-10


@pytest.mark.parametrize("mode,operators", [
    ("finite2", ["refit"]),
    ("recurrent", ["refit", "instrument", "assess_instrument", "response"]),
])
def test_selected_procedure_completes_parameter_or_instrument_research_in_same_agent(
        tmp_path, source_archive, mode, operators):
    agent, lab, _ = consumer(tmp_path / "consumer", source_archive, mode=mode)
    events = deepcopy(agent.events)
    result = execute_procedure(agent, lab, confirm=True)
    assert result["status"] == "completed"
    assert agent.model is not None and agent.confirmations[-1]["passed"]
    assert [item["operator"] for item in agent.operator_results] == operators
    assert len(result["steps"]) == len(agent.method_results)
    assert all(step["action"] == "query" and step["arguments"] == {"path": PROCEDURE}
               for step in result["steps"])
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"]
    assert agent.costs["measurement_cost"] == lab.costs["measurement_cost"]
    assert agent.costs["python_execution_seconds"] > 0
    assert_no_llm_activity(agent, events)
    assert_terminal(agent, lab)


def test_direct_model_declaration_needs_confirmation_authorization_and_no_preliminary_query(
        tmp_path, source_archive):
    agent, lab, model = consumer(tmp_path / "consumer", source_archive, changed=False, paid=False)
    agent._save(PROCEDURE, f"def select_actions(spec, observations, memory):\n return {{'model': {model!r}}}\n")
    denied = execute_procedure(agent, lab)
    assert denied["status"] == "confirmation_not_authorized"
    assert len(denied["steps"]) == 1 and agent.model is None
    assert not agent.rows and not agent.observations and not agent.confirmations
    assert agent.costs["observation_cost"] == agent.costs["compute_cost"] == 0
    allowed = execute_procedure(agent, lab, confirm=True)
    assert allowed["status"] == "completed" and len(allowed["steps"]) == 1
    assert agent.model == model and agent.confirmations[-1]["passed"]
    assert agent.costs["observation_cost"] == agent.costs["compute_cost"] == 24
    assert agent.costs["measurement_cost"] == 0 and agent.operator_results == []
    assert_no_llm_activity(agent, [])
    assert_terminal(agent, lab)

    # A caller may invoke the procedure again after confirmation. It must not
    # acquire another confirmation trajectory or execute the program again.
    previous_costs = deepcopy(agent.costs)
    previous_results = deepcopy(agent.method_results)
    repeated = execute_procedure(agent, lab, confirm=True)
    assert repeated["status"] == "completed" and repeated["steps"] == []
    assert agent.costs == previous_costs and agent.method_results == previous_results
    assert len(agent.confirmations) == 1


def test_failed_confirmation_stops_once_and_retains_its_actual_counterexamples(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
    agent = ProcedureConsumerAgent(None, tmp_path, arm="code_react", source_task=False,
        config=configuration())
    agent._save("wrong.py", "def predict(row):\n return 0.\n")
    agent._save(PROCEDURE, "def select_actions(spec, observations, memory):\n"
        " return {'action': 'finish', 'arguments': {'model': 'wrong.py'}}\n")
    result = execute_procedure(agent, lab, confirm=True)
    assert result["status"] == "confirmation_failed" and len(result["steps"]) == 1
    assert len(agent.confirmations) == len(agent.method_results) == 1
    assert not agent.confirmations[0]["passed"] and agent.model is None
    assert len(agent.rows) == 24 and sum(len(record["rows"]) for record in agent.observations) == 24
    assert agent.costs["observation_cost"] == agent.costs["compute_cost"] == 24
    assert agent.method_results[0]["feedback"]["fresh_confirmation"]["passed"] is False
    assert_no_llm_activity(agent, [])


def test_unaffordable_program_query_preserves_previous_paid_evidence_and_costs(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
    agent = ProcedureConsumerAgent(None, tmp_path, arm="code_react", source_task=False,
        config=replace(configuration(), observation_budget=96))
    purchase(agent, lab)
    agent._save(PROCEDURE, "def select_actions(spec, observations, memory):\n"
        " return {'actions': [{'kind': 'trajectory', 'channels': ['meter'],\n"
        "  'sequence': [{'u': 0.5}] * 24}] * 2}\n")
    observations, before, lab_before = deepcopy(agent.observations), deepcopy(agent.costs), deepcopy(lab.costs)
    result = execute_procedure(agent, lab, confirm=True)
    assert result["status"] == "method_error" and len(result["steps"]) == 1
    assert agent.method_results[-1]["feedback"]["error"] == "observation budget exhausted"
    assert agent.observations == observations and len(agent.rows) == 72
    assert lab.costs == lab_before
    for key, value in before.items():
        if key != "python_execution_seconds":
            assert agent.costs[key] == value
    assert agent.model is None and not agent.confirmations
    assert_no_llm_activity(agent, [])


def test_procedure_keeps_numeric_confirmation_reserve_before_any_step(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
    agent = ProcedureConsumerAgent(None, tmp_path, arm="code_react", source_task=False,
        config=replace(configuration(), observation_budget=80))
    purchase(agent, lab)
    agent._save(PROCEDURE, "def select_actions(spec, observations, memory):\n"
        " return {'action': 'write', 'arguments': {'path': 'executed.txt', 'code': 'ran'}}\n")
    before = deepcopy(agent.costs)
    result = execute_procedure(agent, lab, confirm=True)
    assert result["status"] == "confirmation_budget_exhausted" and result["steps"] == []
    assert agent.costs == before and agent.method_results == []
    assert not (agent.workspace / "executed.txt").exists()


def test_step_limit_resumes_actual_pending_research_without_restarting_or_llm_events(tmp_path, source_archive):
    agent, lab, _ = consumer(tmp_path / "consumer", source_archive, mode="recurrent")
    first = execute_procedure(agent, lab, max_steps=1, confirm=True)
    assert first["status"] == "method_step_limit" and len(first["steps"]) == 1
    assert [item["operator"] for item in agent.operator_results] == ["refit"]
    assert agent.operator_results[0]["status"] == "refit_inadequate_on_validation"
    assert agent.model is None and not agent.confirmations and len(agent.rows) == 72
    partial = deepcopy(agent.method_results)
    resumed = execute_procedure(agent, lab, confirm=True)
    assert resumed["status"] == "completed"
    assert agent.method_results[:len(partial)] == partial
    assert len(agent.method_results) == len(partial) + len(resumed["steps"])
    assert sum(item["operator"] == "refit" for item in agent.operator_results) == 1
    assert agent.confirmations[-1]["passed"]
    assert_no_llm_activity(agent, [])
    assert_terminal(agent, lab)


def test_ordinary_program_query_still_executes_exactly_one_selected_action(tmp_path, source_archive):
    agent, lab, _ = consumer(tmp_path / "consumer", source_archive, mode="recurrent")
    feedback = agent._execute("query", {"path": PROCEDURE}, lab)
    assert feedback["status"] == "refit_inadequate_on_validation"
    assert len(agent.method_results) == 1
    assert agent.method_results[0]["proposal"]["action"] == "run"
    assert [item["operator"] for item in agent.operator_results] == ["refit"]
    assert len(agent.rows) == 72 and agent.model is None and not agent.confirmations
    assert_no_llm_activity(agent, [])
