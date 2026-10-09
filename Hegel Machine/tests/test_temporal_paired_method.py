"""Frozen scripted method interface tests, not LLM performance evidence."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from hegel_machine.cognition.temporal_paired_method import FrozenMethodAgent, ProcedureConsumerAgent


PAIRED_PROGRAM = """import json
from random import Random
def select_actions(spec, observations, memory):
    roles = memory['research_roles']
    state = json.load(open('research_state.json'))
    outcomes = state['operator_results']
    if not outcomes:
        rng = Random(193)
        actions = [{'kind':'trajectory','sequence':[{'u':rng.uniform(-1.,1.)}
                   for _ in range(spec['max_probe_length'])], 'channels':['meter']} for _ in range(2)]
        return {'actions':actions, 'after_observation':[
            {'operator':'assess_instrument','instrument':roles['instrument'],'records':'acquired'}]}
    latest = outcomes[-1]
    if latest['operator'] == 'assess_instrument':
        if not latest.get('error') and any(item['status'] == 'supported' for item in latest['hypotheses']):
            return {'model':roles['predictor'], 'lesson':'The measured instrument still agrees with the retained law.'}
        return {'actions':[], 'after_observation':[
            {'operator':'instrument','path':roles['instrument']},
            {'operator':'response','instrument':roles['instrument'],'output_dir':'shared/revised'}]}
    assert latest['operator'] == 'response'
    candidate = min(latest['candidates'], key=lambda item:item['validation_mse'])
    return {'model':candidate['path'], 'lesson':'Current measurements required a new instrument and response fit.'}
"""

DIRECT_PRIOR_PROGRAM = """import json
from random import Random
def select_actions(spec, observations, memory):
    roles = memory['research_roles']
    state = json.load(open('research_state.json'))
    if not state['method_results']:
        return {'model':roles['predictor']}
    confirmation = state['method_results'][-1]['feedback'].get('fresh_confirmation')
    if confirmation and not confirmation['passed']:
        rng = Random(197)
        return {'actions':[{'kind':'trajectory','sequence':[{'u':rng.uniform(-1.,1.)}
                for _ in range(spec['max_probe_length'])], 'channels':['meter']} for _ in range(2)],
                'after_observation':[{'operator':'instrument','path':roles['instrument']},
                                    {'operator':'response','instrument':roles['instrument'],
                                     'output_dir':'shared/revised'}]}
    latest = state['operator_results'][-1]
    assert latest['operator'] == 'response'
    candidate = min(latest['candidates'],key=lambda item:item['validation_mse'])
    return {'model':candidate['path']}
"""


@pytest.fixture
def calibrated_knowledge(tmp_path):
    workspace = tmp_path / "calibration"
    config = InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                              execution_seconds=30., compute_budget=100000)
    agent = TemporalOperatorAgent(None, workspace, arm="code_react", config=config)
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")
    rng = Random(192)

    def action():
        return {"kind": "trajectory", "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)],
                "channels": ["meter"]}

    agent._execute("query", {"actions": [action(), action()]}, lab)
    instrument = agent._execute("run", {"operator": "instrument", "path": "instrument.json"}, lab)
    assert "error" not in instrument
    agent._execute("query", {"actions": [action()]}, lab)
    assessed = agent._execute("run", {"operator": "assess_instrument", "instrument": "instrument.json",
                                      "records": [2]}, lab)
    assert len([item for item in assessed["hypotheses"] if item["status"] == "supported"]) == 1
    response = agent._execute("run", {"operator": "response", "instrument": "instrument.json"}, lab)
    candidate = min(response["candidates"], key=lambda item: item["validation_mse"])
    assert candidate["validation_mse"] < 1e-10
    predictor = (workspace / candidate["path"]).read_text()
    agent._execute("write", {"path": "shared/prior_response.py", "code": predictor}, lab)
    confirmed = agent._execute("finish", {"model": "shared/prior_response.py"}, lab)
    assert confirmed["fresh_confirmation"]["passed"]
    return {"files": {"instrument.json": (workspace / "instrument.json").read_text(),
                       "shared/prior_response.py": predictor},
            "research_roles": {"instrument": "instrument.json", "predictor": "shared/prior_response.py"},
            "manifest": {"hypotheses": [{"id": "prior_response", "path": "shared/prior_response.py"}],
                         "research_programs": [], "support_questions": []},
            "lessons": ["The retained law was identified and confirmed using paid trajectories."]}


def make_frozen(tmp_path, knowledge, method_source=PAIRED_PROGRAM):
    return FrozenMethodAgent(tmp_path, knowledge=deepcopy(knowledge), method_source=method_source,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
                               execution_seconds=30., compute_budget=100000), confirmation_seed=195)


def test_identical_program_reuses_supported_prior_and_refits_refuted_prior(
        tmp_path, calibrated_knowledge):
    original = deepcopy(calibrated_knowledge)
    runs = []
    specs = []
    for stage in (1, 2):
        agent = make_frozen(tmp_path / f"stage_{stage}", calibrated_knowledge)
        lab = TemporalInquiryLab(task_index=stage, apparatus_seed=99106007, instrument_mode="recurrent")
        specs.append(lab.public_spec())
        for path, source in calibrated_knowledge["files"].items():
            assert (agent.workspace / path).read_text() == source
        result = agent.execute_method(lab, max_steps=8)
        assert result["status"] == "completed"
        assert result["fresh_confirmation"]["passed"]
        assert result["fresh_confirmation"]["rows"] == 48
        assert result["costs"]["observation_cost"] == lab.costs["observation_cost"] == 96
        assert result["costs"]["measurement_cost"] == 144
        assert result["costs"]["compute_cost"] > 0 and result["costs"]["python_execution_seconds"] > 0
        assert result["costs"]["model_calls"] == result["costs"]["reported_tokens"] == 0
        assert (agent.workspace / METHOD_PATH).read_text() == PAIRED_PROGRAM
        assert (agent.workspace / "shared/prior_response.py").read_text() == original["files"]["shared/prior_response.py"]
        operators = [item["operator"] for item in agent.operator_results]
        if stage == 1:
            assert operators == ["assess_instrument"]
            assert result["model"] == "shared/prior_response.py"
            assert any(item["status"] == "supported" for item in agent.operator_results[0]["hypotheses"])
        else:
            assert operators == ["assess_instrument", "instrument", "response"]
            assert all(item["status"] == "refuted" for item in agent.operator_results[0]["hypotheses"])
            assert result["model"].startswith("shared/revised/")
        runs.append(result)
    assert specs[0] == specs[1]
    assert calibrated_knowledge == original
    assert [record["action"] for record in runs[0]["paid_observations"]] == [
        record["action"] for record in runs[1]["paid_observations"]]


def test_frozen_runtime_does_not_submit_before_program_declares_a_model(
        tmp_path, calibrated_knowledge):
    agent = make_frozen(tmp_path / "one_step", calibrated_knowledge)
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")
    result = agent.execute_method(lab, max_steps=1)
    assert result["status"] == "method_step_limit"
    assert result["model"] is None and result["fresh_confirmation"] is None
    assert result["costs"]["observation_cost"] == 48
    assert [item["operator"] for item in agent.operator_results] == ["assess_instrument"]
    assert any(item["status"] == "supported" for item in agent.operator_results[0]["hypotheses"])
    assert (agent.workspace / "shared/prior_response.py").read_text() == calibrated_knowledge["files"]["shared/prior_response.py"]
    assert result["costs"]["model_calls"] == 0


def test_frozen_program_receives_paid_error_feedback_and_recovers_without_host_policy(
        tmp_path, calibrated_knowledge):
    memory = deepcopy(calibrated_knowledge)
    memory["research_roles"]["instrument"] = "recovered_instrument.json"
    agent = make_frozen(tmp_path / "recover", memory)
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")
    result = agent.execute_method(lab, max_steps=8)
    assert result["status"] == "completed" and result["fresh_confirmation"]["passed"]
    assert agent.operator_results[0]["operator"] == "assess_instrument"
    assert agent.operator_results[0]["error"]
    assert [item["operator"] for item in agent.operator_results] == ["assess_instrument", "instrument", "response"]
    assert result["model"].startswith("shared/revised/")
    state = json.loads((agent.workspace / "research_state.json").read_text())
    assert any(item["feedback"].get("after_observation_error") for item in state["method_results"])
    assert state["operator_results"][0]["error"] == agent.operator_results[0]["error"]
    assert result["costs"]["observation_cost"] == 96
    assert result["costs"]["model_calls"] == result["costs"]["reported_tokens"] == 0
    assert (agent.workspace / METHOD_PATH).read_text() == PAIRED_PROGRAM


def test_initial_prior_submission_confirms_without_preliminary_acquisition_or_refit(
        tmp_path, calibrated_knowledge):
    agent = make_frozen(tmp_path / "direct_stable", calibrated_knowledge, DIRECT_PRIOR_PROGRAM)
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=99106007, instrument_mode="recurrent")
    result = agent.execute_method(lab)
    assert result["status"] == "completed"
    assert result["model"] == calibrated_knowledge["research_roles"]["predictor"]
    assert result["fresh_confirmation"]["passed"] and result["fresh_confirmation"]["rows"] == 48
    assert result["fresh_confirmation"]["mse"] < 1e-10
    assert result["costs"]["observation_cost"] == result["costs"]["compute_cost"] == 48
    assert result["costs"]["measurement_cost"] == 0
    assert result["costs"]["model_calls"] == 0
    assert agent.operator_results == []
    assert len(result["method_results"]) == 1
    assert result["method_results"][0]["proposal"] == {"model": result["model"]}
    assert (agent.workspace / result["model"]).read_text() == calibrated_knowledge["files"][result["model"]]


def test_initial_failed_confirmation_drives_program_owned_measurement_and_revision(
        tmp_path, calibrated_knowledge):
    agent = make_frozen(tmp_path / "direct_changed", calibrated_knowledge, DIRECT_PRIOR_PROGRAM)
    lab = TemporalInquiryLab(task_index=2, apparatus_seed=99106007, instrument_mode="recurrent")
    checkpoints = []
    result = agent.execute_method(lab, on_checkpoint=lambda data: checkpoints.append(deepcopy(data)))
    first = checkpoints[0]
    assert first["model"] is None
    assert first["fresh_confirmation"]["passed"] is False
    assert first["fresh_confirmation"]["rows"] == 48 and first["fresh_confirmation"]["mse"] > 1e-4
    assert first["costs"]["observation_cost"] == first["costs"]["compute_cost"] == 48
    assert first["costs"]["measurement_cost"] == 0
    assert first["method_results"][0]["proposal"] == {"model": calibrated_knowledge["research_roles"]["predictor"]}
    assert result["status"] == "completed" and result["fresh_confirmation"]["passed"]
    assert result["model"].startswith("shared/revised/")
    assert result["costs"]["observation_cost"] == 144  # failed check + chosen measurement + fresh check
    assert result["costs"]["measurement_cost"] == 144
    assert result["costs"]["model_calls"] == result["costs"]["reported_tokens"] == 0
    assert [item["operator"] for item in agent.operator_results] == ["instrument", "response"]
    assert [len(item["proposal"].get("actions", [])) for item in result["method_results"]] == [0, 2, 0]
    state = json.loads((agent.workspace / "research_state.json").read_text())
    assert state["method_results"][0]["feedback"]["fresh_confirmation"]["mse"] == first["fresh_confirmation"]["mse"]
    assert state["method_results"][0]["feedback"]["fresh_confirmation"]["passed"] is False
    assert (agent.workspace / METHOD_PATH).read_text() == DIRECT_PRIOR_PROGRAM


@pytest.mark.parametrize("stage,expected_calls,expected_rows", [(1, 1, 48), (2, 3, 144)])
def test_live_consumer_executes_one_method_step_per_model_decision(
        tmp_path, calibrated_knowledge, stage, expected_calls, expected_rows):
    class RepeatedQueryProvider:
        def __init__(self):
            self.seen = []

        def next_action(self, request):
            self.seen.append(deepcopy(request))
            return {"action": "query", "arguments": {"path": METHOD_PATH},
                    "usage": {"backend": "scripted_test", "total_tokens": 1}}

    memory = deepcopy(calibrated_knowledge)
    memory["files"][METHOD_PATH] = DIRECT_PRIOR_PROGRAM
    provider = RepeatedQueryProvider()
    agent = ProcedureConsumerAgent(provider, tmp_path / "consumer", arm="hegel",
        knowledge=memory, source_task=False, confirmation_seed=195,
        config=InquiryLLMConfig(observation_budget=240, confirmation_rows=48,
            execution_seconds=30., compute_budget=100000, max_model_calls=3))
    lab = TemporalInquiryLab(task_index=stage, apparatus_seed=99106007, instrument_mode="recurrent")
    result = agent.run(lab)
    assert result["status"] == "completed" and result["fresh_confirmation"]["passed"]
    assert result["costs"]["model_calls"] == len(provider.seen) == expected_calls
    assert len(result["method_results"]) == expected_calls
    assert result["costs"]["observation_cost"] == expected_rows
    assert all(event["feedback"]["executed_path"] == METHOD_PATH for event in result["events"])
    assert all(event["feedback"]["inherited_source_unchanged"] for event in result["events"])
    first = result["events"][0]["feedback"]
    assert first["method_submission"] == calibrated_knowledge["research_roles"]["predictor"]
    assert first["submitted_model_inherited"] and first["submitted_model_unchanged"]
    if stage == 2:
        assert first["fresh_confirmation"]["passed"] is False
        assert provider.seen[1]["costs"]["observation_cost"] == 48
        assert provider.seen[1]["events"][-1]["feedback"]["fresh_confirmation"]["passed"] is False
        assert [op["operator"] for op in agent.operator_results] == ["instrument", "response"]
    else:
        assert agent.operator_results == []
