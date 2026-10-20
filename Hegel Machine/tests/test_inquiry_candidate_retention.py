"""Retained candidates are executable paid evaluations, not printed claims."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.online_inquiry import InquiryLab
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from hegel_machine.cognition.temporal_paired_method import ProcedureConsumerAgent


def prepared_agent(path, values=(.1, .3, .5, .8)):
    laboratory = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)
    agent = TemporalOperatorAgent(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=24, confirmation_rows=3,
            compute_budget=10000, execution_seconds=60., max_model_calls=2))
    buy(agent, laboratory, values)
    return agent, laboratory


def buy(agent, laboratory, values):
    controls = laboratory.public_spec()["controls"]
    return agent._execute("query", {"actions": [
        {"kind": "response", "point": {controls[0]: value, controls[1]: 0.}}
        for value in values]}, laboratory)


def write(agent, laboratory, source, path="models/model.py"):
    return agent._execute("write", {"path": path, "code": source}, laboratory)


def test_consumer_evaluation_without_paid_rows_has_no_retained_candidate(tmp_path):
    laboratory = InquiryLab(seed=8, mechanism="linear", driver=0, orientation=1)
    agent = ProcedureConsumerAgent(None, tmp_path, arm="code_react", source_task=False)
    write(agent, laboratory, "def predict(row):\n return 0.\n")
    result = agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    assert result["mse"] is None and result["rows"] == 0
    assert result["evaluated_model"] is None and agent.evaluated_model is None
    assert not (agent.workspace / "_evaluated_model").exists()
    assert agent.model is None and not agent.confirmations
    assert agent.costs["observation_cost"] == agent.costs["compute_cost"] == 0


@pytest.mark.parametrize("fitted", [False, True])
def test_worse_same_path_edit_preserves_evaluated_candidate_until_explicit_finish(tmp_path, fitted):
    agent, laboratory = prepared_agent(tmp_path)
    control = laboratory.public_spec()["controls"][0]
    source = (f"def predict(row):\n return row['features'][{control!r}]\n" if not fitted else
        "scale = None\n"
        "def fit(rows):\n global scale\n"
        f" scale = sum(row['target'] for row in rows) / sum(row['features'][{control!r}] for row in rows)\n"
        f"def predict(row):\n return scale * row['features'][{control!r}]\n")
    write(agent, laboratory, source)
    first = agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    saved = deepcopy(first["evaluated_model"])
    assert first["mse"] < 1e-20
    assert first['retention_decision']['status'] == 'candidate_retained'
    assert first['retention_decision']['incumbent'] is None
    assert saved["model"] != "models/model.py" and saved["source_model"] == "models/model.py"
    assert saved["support"] == "paid_fit" and saved["evaluated_rows"] == 4
    assert (agent.workspace / saved["model"]).read_text() == source
    assert agent.model is None and not agent.confirmations

    write(agent, laboratory, "def predict(row):\n return -100.\n")
    second = agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    assert second["mse"] > 1000.
    assert second['retention_decision']['status'] == 'incumbent_retained'
    assert second['retention_decision']['candidate']['model'] == 'models/model.py'
    assert second['retention_decision']['incumbent']['mse'] < 1e-20
    assert second['retention_decision']['selected_model'] == saved['model']
    assert second['evaluated_model']['selection'] == saved['selection']
    assert second["evaluated_model"]["model"] == saved["model"]
    assert second["evaluated_model"]["mse"] < 1e-20
    assert (agent.workspace / saved["model"]).read_text() == source
    assert agent.result("researching")["model"] is None
    assert agent.result("researching")["evaluated_model"]["model"] == saved["model"]
    before = deepcopy(agent.costs)
    submitted = agent._execute("finish", {"model": saved["model"]}, laboratory)
    assert submitted["fresh_confirmation"]["passed"]
    assert agent.model == saved["model"]
    assert agent.costs["observation_cost"] - before["observation_cost"] == 3
    assert agent.costs["compute_cost"] - before["compute_cost"] == 4 + 3


def test_fitted_snapshot_keeps_original_data_and_new_paid_counterexample_refutes_it(tmp_path):
    agent, laboratory = prepared_agent(tmp_path, values=(1., 1., 1., 1.))
    source = ("mean = None\ndef fit(rows):\n global mean\n"
              " mean = sum(row['target'] for row in rows) / len(rows)\n"
              "def predict(row):\n return mean\n")
    write(agent, laboratory, source)
    first = agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    saved = first["evaluated_model"]
    fitting = agent.workspace / "_evaluated_model" / "model_observations.json"
    assert len(json.loads(fitting.read_text())) == 4
    buy(agent, laboratory, [0.])
    stale = agent.result("researching")["evaluated_model"]
    assert stale["support"] == "new_paid_rows_unscored"
    assert stale["evaluated_rows"] == 4 and stale["current_paid_rows"] == 5
    assert len(json.loads(fitting.read_text())) == 4
    before = agent.costs["compute_cost"]
    scored = agent._execute("evaluate", {"model": saved["model"]}, laboratory)
    # The frozen four-row fit predicts one. Refitting on the current five rows
    # would silently change that prediction to 0.8 and give the wrong receipt.
    assert scored["mse"] == pytest.approx(.2)
    assert scored['retention_decision']['status'] == 'retained_rescored'
    assert scored['retention_decision']['candidate']['fitting_view']['rows'] == 4
    assert scored['retention_decision']['candidate']['evaluated_rows'] == 5
    assert scored['evaluated_model']['selection'] == saved['selection']
    assert agent.costs["compute_cost"] - before == 5
    assert scored["evaluated_model"]["mse"] == pytest.approx(.2)
    assert scored["evaluated_model"]["evaluated_rows"] == 5
    assert len(json.loads(fitting.read_text())) == 4
    assert agent.model is None and not agent.confirmations


def test_new_candidate_competes_with_saved_model_on_same_new_paid_rows_and_pays_both(tmp_path):
    agent, laboratory = prepared_agent(tmp_path, values=(1., 1., 1., 1.))
    write(agent, laboratory, "def predict(row):\n return 1.\n")
    agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    buy(agent, laboratory, [0.])
    control = laboratory.public_spec()["controls"][0]
    source = f"def predict(row):\n return row['features'][{control!r}]\n"
    write(agent, laboratory, source, path="models/revision.py")
    before = agent.costs["compute_cost"]
    revised = agent._execute("evaluate", {"model": "models/revision.py"}, laboratory)
    assert agent.costs["compute_cost"] - before == 2 * 5
    assert revised["mse"] < 1e-20
    assert revised["evaluated_model"]["source_model"] == "models/revision.py"
    assert revised["evaluated_model"]["evaluated_rows"] == 5
    decision = revised['retention_decision']
    assert decision['status'] == 'candidate_retained' and decision['same_paid_rows']
    assert decision['candidate']['mse'] < 1e-20
    assert decision['incumbent']['mse'] == pytest.approx(.2)
    assert decision['candidate']['fitting_view']['rows'] == 5
    assert decision['incumbent']['fitting_view']['rows'] == 4
    assert decision['selected_model'] == revised['evaluated_model']['model']
    assert revised['evaluated_model']['selection'] == decision
    assert agent.result("researching")["model"] is None


def test_paid_rejected_finish_keeps_better_refit_before_worse_candidate(tmp_path):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="finite2")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=24, confirmation_rows=3,
            compute_budget=1000, execution_seconds=60.))
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": value}]} for value in (0., .5, 1.)]}, laboratory)
    write(agent, laboratory, "def predict(row):\n return -100.\n")
    agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": -.5}]}]}, laboratory)
    source = ("mean = None\ndef fit(rows):\n global mean\n"
              " mean = sum(row['target'] for row in rows) / len(rows)\n"
              "def predict(row):\n return mean\n")
    write(agent, laboratory, source)
    before = deepcopy(agent.costs)
    rejected = agent._execute("finish", {"model": "models/model.py"}, laboratory)
    assert "paid evidence contradicts" in rejected["error"]
    assert rejected["mse"] > agent.config.confirmation_mse_threshold
    assert rejected["retained_comparison"]["mse"] > rejected["mse"]
    assert rejected['retention_decision']['status'] == 'candidate_retained'
    saved = deepcopy(rejected["evaluated_model"])
    assert saved["mse"] == rejected["mse"] and saved["evaluated_rows"] == 4
    assert (agent.workspace / saved["model"]).read_text() == source
    assert agent.costs["compute_cost"] - before["compute_cost"] == 8
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert "fresh_confirmation" not in rejected
    assert agent.model is None and not agent.confirmations

    write(agent, laboratory, source.replace("return mean", "return mean + 1."))
    worse = agent._execute("evaluate", {"model": "models/model.py"}, laboratory)
    assert worse["mse"] == pytest.approx(rejected["mse"] + 1.)
    assert worse["evaluated_model"] == saved
    assert (agent.workspace / saved["model"]).read_text() == source
    assert agent.model is None and not agent.confirmations


def test_unaffordable_current_row_comparison_reports_deferred_without_selecting_candidate(tmp_path):
    agent, laboratory = prepared_agent(tmp_path, values=(1., 1., 1., 1.))
    write(agent, laboratory, 'def predict(row): return 1.\n')
    first = agent._execute('evaluate', {'model': 'models/model.py'}, laboratory)
    buy(agent, laboratory, [0.])
    control = laboratory.public_spec()['controls'][0]
    write(agent, laboratory, f"def predict(row): return row['features'][{control!r}]\n")
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 5)

    result = agent._execute('evaluate', {'model': 'models/model.py'}, laboratory)

    decision = result['retention_decision']
    assert decision['status'] == 'deferred' and decision['same_paid_rows'] is False
    assert decision['candidate']['evaluated_rows'] == 5
    assert decision['incumbent']['evaluated_rows'] == 4
    assert decision['selected_model'] == first['evaluated_model']['model']
    assert result['evaluated_model']['selection'] == first['evaluated_model']['selection']
    assert (agent.workspace / decision['selected_model']).read_text() == 'def predict(row): return 1.\n'
    assert agent.model is None and not agent.confirmations


def test_procedure_replacement_decision_reaches_current_notebook_and_compact_provider(tmp_path):
    from hegel_machine.cognition.code_provider_context import prepare_code_context
    from test_common_archive_online_research import prepared
    from test_common_archive_resume import execute
    from test_inquiry_revision import PublicLab, action

    agent, laboratory = prepared(tmp_path, 'hegel'), PublicLab()
    execute(agent, laboratory, 'query', {'actions': [action(2)]})
    execute(agent, laboratory, 'evaluate', {'model': 'old.py'})
    execute(agent, laboratory, 'query', {'actions': [action(4)]})
    execute(agent, laboratory, 'write', {'path': 'procedure.py', 'code': (
        "def select_actions(spec, observations, memory):\n"
        " return {'action':'evaluate','arguments':{'model':'new.py'}}\n")})

    result = execute(agent, laboratory, 'run', {'operator': 'procedure',
        'path': 'procedure.py', 'max_steps': 1, 'confirm': False})

    feedback = result['steps'][0]['feedback']
    decision = feedback['retention_decision']
    assert decision['status'] == 'candidate_retained'
    assert decision['candidate']['mse'] == 0.
    assert decision['incumbent']['mse'] == 10.
    assert decision['selected_model'] == '_evaluated_model/new.py'
    request = agent._request(laboratory)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='runtime test')
    fact = context['evidence_notebook']['facts']['current_paid_evaluation']
    assert fact['selection'] == decision
    assert context['events'][-1]['feedback']['steps'][0]['feedback']['retention_decision'] == decision
    assert not agent.confirmations and agent.model is None

def test_printed_zero_error_is_not_a_candidate_or_a_submission(tmp_path):
    agent, laboratory = prepared_agent(tmp_path)
    printed = agent._execute("run", {"code": "print('training_mse=0, validation_mse=0')"}, laboratory)
    assert printed["returncode"] == 0 and "validation_mse=0" in printed["stdout"]
    result = agent.result("researching")
    assert result["evaluated_model"] is None
    assert result["model"] is None and result["fresh_confirmation"] is None
    assert agent.costs["compute_cost"] == 0


def test_python_research_choice_uses_existing_policy_and_recommends_only_paid_confirmation(
        tmp_path, monkeypatch):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="finite2")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=192, confirmation_rows=12,
            compute_budget=300000, execution_seconds=60.))
    rng = Random(71)
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}
        for _ in range(3)]}, laboratory)
    exported = agent._execute("run", {"operator": "history", "base_lags": 2,
        "path": "models/history.py"}, laboratory)
    assert agent.result("researching")["evaluated_model"] is None
    agent._prepare_program_state()
    state = json.loads((agent.workspace / "research_state.json").read_text())
    assert state["exported_candidate_paths"] == [exported["path"]]
    assert state["evaluated_model"] is None
    calls, original_choose = [], ResearchActionPolicy.choose

    def choose(self, options, **kwargs):
        calls.append([option.name for option in options])
        return original_choose(self, options, **kwargs)

    monkeypatch.setattr(ResearchActionPolicy, "choose", choose)
    before = deepcopy(agent.costs)
    result = agent._execute("run", {"operator": "research_choice", "models": [exported["path"]]}, laboratory)
    assert calls == [["retain_response"]]
    assert result["decision"]["selected_action"] == "retain_response"
    saved = agent.result("researching")["evaluated_model"]
    assert saved["mse"] < 1e-10
    assert result["next_action"] == {"action": "finish", "arguments": {"model": saved["model"]}}
    assert result["decision"]["quoted_compute_cost"] == 72 + 12
    assert result["decision"]["quoted_observation_cost"] == 12
    assert agent.costs["compute_cost"] - before["compute_cost"] == 72
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert agent.model is None and not agent.confirmations


def test_default_research_choice_rescores_saved_python_after_real_counterexample(tmp_path):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="finite2")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=24, confirmation_rows=3,
            compute_budget=1000, execution_seconds=60.))
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": 0.}]} for _ in range(3)]}, laboratory)
    write(agent, laboratory, "mean = None\ndef fit(rows):\n global mean\n"
        " mean = sum(row['target'] for row in rows) / len(rows)\n"
        "def predict(row):\n return mean\n")
    first = agent._execute("run", {"operator": "research_choice", "models": ["models/model.py"]}, laboratory)
    assert first["next_action"]["action"] == "finish"
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": 1.}]}]}, laboratory)
    before = deepcopy(agent.costs)
    second = agent._execute("run", {"operator": "research_choice"}, laboratory)
    assert second["diagnosis"] == "response_counterexample"
    assert second["decision"]["selected_action"] == "refit_response"
    assert second["next_action"]["arguments"]["operator"] == "refit"
    assert agent.costs["compute_cost"] - before["compute_cost"] == 4
    assert agent.costs["observation_cost"] == before["observation_cost"]
    assert agent.result("researching")["evaluated_model"]["mse"] > 1e-4
    assert agent.model is None and not agent.confirmations


def test_python_choice_keeps_candidate_without_unaffordable_finish_recommendation(tmp_path):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="finite2")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=24, confirmation_rows=3,
            compute_budget=8, execution_seconds=60.))
    agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
        "sequence": [{"u": 0.}]} for _ in range(3)]}, laboratory)
    write(agent, laboratory, "mean = None\ndef fit(rows):\n global mean\n"
        " mean = sum(row['target'] for row in rows) / len(rows)\n"
        "def predict(row):\n return mean\n")
    result = agent._execute("run", {"operator": "research_choice", "models": ["models/model.py"]}, laboratory)
    # Three scoring units leave five: enough for three fresh rows alone, but
    # finish must also score all three earlier paid rows before buying them.
    assert agent.costs["compute_cost"] == 3
    assert agent.result("researching")["evaluated_model"]["mse"] < 1e-10
    assert result["decision"]["reason"] == "no_affordable_action"
    assert result["recommendation"] is None and result.get("next_action") is None
    assert agent.model is None and not agent.confirmations
