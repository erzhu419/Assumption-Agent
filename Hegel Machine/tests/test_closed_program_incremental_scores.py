"""Actual closed predictors append paid predictions without replaying the prefix."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_inquiry_revision import action
from test_python_paid_score_reuse import ChoiceLab, choose, no_execution


MODEL = 'frozen/_evaluated_model/model.py'


def scene(path, monkeypatch, *, arm='hegel', program=None):
    agent = CommonArchiveInquiryAgent(None, path, arm=arm, source_task=False,
        config=InquiryLLMConfig(compute_budget=1000, execution_seconds=30.,
            observation_budget=100, max_model_calls=0))
    agent._save(MODEL, export_temporal_program(program or Program('const', value=.1)))
    agent._save('frozen/_evaluated_model/model_observations.json', '[]')
    agent._save('frozen/_evaluated_model/params.json', '{"version": 1}')
    lab, calls = ChoiceLab(), []
    worker = evaluate_python_model

    def recorded(workspace, model, rows, **kwargs):
        calls.append((model, deepcopy(rows)))
        return worker(workspace, model, rows, **kwargs)

    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', recorded)
    agent._execute('query', {'actions': [action(.2, .3)], 'question': {
        'id': 'q', 'question': 'Predict the measured response', 'context_ref': {'fixture': 'append'}}}, lab)
    return agent, lab, calls


def state(agent, evaluation):
    loaded, = load_source_values([evaluation['closed_prediction_state_ref']], agent.workspace)
    return loaded['value']


@pytest.mark.parametrize('arm', ['code_react', 'hegel', 'hegel'], ids=['archive_A', 'fixed_B', 'learned_C'])
def test_all_arms_execute_only_suffix_and_match_full_loss_and_residuals(tmp_path, monkeypatch, arm):
    agent, lab, calls = scene(tmp_path, monkeypatch, arm=arm)
    first = choose(agent, lab, [MODEL])['evaluations'][0]
    assert len(calls) == 1 and len(calls[0][1]) == 2
    before = agent.costs['compute_cost']
    agent._execute('query', {'actions': [action(.4, .7)]}, lab)
    result = choose(agent, lab, [MODEL])['evaluations'][0]
    assert len(calls) == 2 and calls[-1][1] == agent.rows[2:]
    assert agent.costs['compute_cost'] - before == 2
    full = evaluate_python_model(agent.workspace, MODEL, agent.rows)
    assert result['mse'] == full['mse'] and result['loss'] == full['loss']
    assert result['representation_diagnostic'] == full['representation_diagnostic']
    assert result['paid_residuals']['evaluated_rows'] == len(agent.rows)
    assert result['paid_score_reuse']['reused_prediction_rows'] == 2
    assert result['paid_score_reuse']['newly_predicted_rows'] == 2
    delta = state(agent, result)
    assert len(delta['predictions']) == len(delta['representations']) == 2
    assert delta['prefix_ref'] == first['closed_prediction_state_ref']
    assert delta['row_count'] == 4
    assert '_closed_prediction_state' not in result and 'predictions' not in result


def test_prefix_and_suffix_collision_is_detected_after_merge(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch, program=Program('feature', name='u'))
    # Each segment has unique u; only the union contains equal represented u
    # with different paid targets. Neither segment's summary alone can show it.
    first = choose(agent, lab, [MODEL])['evaluations'][0]
    assert first['representation_diagnostic']['conflicting_groups'] == 0
    agent._execute('query', {'actions': [action(.2, .8)]}, lab)
    agent.rows[2] = replace(agent.rows[2], target=1.2)
    result = choose(agent, lab, [MODEL])['evaluations'][0]
    assert len(calls[-1][1]) == 2
    full = evaluate_python_model(agent.workspace, MODEL, agent.rows)
    assert result['representation_diagnostic'] == full['representation_diagnostic']
    assert result['representation_diagnostic']['conflicting_groups'] == 1
    assert result['representation_diagnostic']['empirical_mse_floor'] == pytest.approx(.125)


@pytest.mark.parametrize('change', ['source', 'params', 'fit_view', 'old_target', 'old_history', 'meaning'])
def test_changed_model_fit_or_prefix_falls_back_to_complete_execution(tmp_path, monkeypatch, change):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    choose(agent, lab, [MODEL])
    agent.evaluated_model = None
    agent._execute('query', {'actions': [action(.4)]}, lab)
    if change == 'source':
        agent._save(MODEL, export_temporal_program(Program('const', value=.2)))
    elif change == 'params':
        agent._save('frozen/_evaluated_model/params.json', '{"version": 2}')
    elif change == 'fit_view':
        agent._save('frozen/_evaluated_model/model_observations.json', '[{"target": 0}]')
    elif change == 'old_target':
        agent.rows[0] = replace(agent.rows[0], target=99.)
    elif change == 'old_history':
        agent.rows[0] = replace(agent.rows[0], history=({'u': -1.},))
    else:
        agent.questions[-1]['meaning_version'] += 1
    result = choose(agent, lab, [MODEL])['evaluations'][0]
    assert calls[-1][1] == agent.rows
    assert 'reused_prediction_rows' not in result['paid_score_reuse']


def test_arbitrary_python_still_reexecutes_all_rows(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    agent._save(MODEL, 'def predict(row): return .1\n')
    first = choose(agent, lab, [MODEL])['evaluations'][0]
    assert 'closed_prediction_state_ref' not in first
    agent.evaluated_model = None
    agent._execute('query', {'actions': [action(.4)]}, lab)
    choose(agent, lab, [MODEL])
    assert calls[-1][1] == agent.rows


def test_cold_restore_reuses_delta_chain_and_zero_cost_complete_score(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path / 'source', monkeypatch)
    choose(agent, lab, [MODEL])
    agent._execute('query', {'actions': [action(.4)]}, lab)
    last = choose(agent, lab, [MODEL])['evaluations'][0]
    agent._prepare_program_state()
    saved = agent.result('researching')
    cold = CommonArchiveInquiryAgent(None, tmp_path / 'cold', arm='hegel', source_task=False,
        config=agent.config)
    cold_lab = ChoiceLab()
    cold_lab.costs = {'observation_cost': 0}
    cold.restore_practice_research(saved, agent.workspace, cold_lab, confirmation_seed=0)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    repeated = choose(cold, ChoiceLab(), [MODEL])['evaluations'][0]
    assert repeated['paid_score_reuse']['status'] == 'reused'
    assert repeated['closed_prediction_state_ref'] == last['closed_prediction_state_ref']
    delta = state(cold, repeated)
    assert delta['row_count'] == 3 and len(delta['predictions']) == 1
    suffix_calls = []

    def recorded(workspace, model, rows, **kwargs):
        suffix_calls.append(deepcopy(rows))
        return evaluate_python_model(workspace, model, rows, **kwargs)

    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', recorded)
    cold._execute('query', {'actions': [action(.7)]}, cold_lab)
    appended = choose(cold, cold_lab, [MODEL])['evaluations'][0]
    assert suffix_calls == [cold.rows[3:]]
    assert state(cold, appended)['prefix_ref'] == last['closed_prediction_state_ref']
    assert appended['mse'] == evaluate_python_model(cold.workspace, MODEL, cold.rows)['mse']


def test_suffix_only_budget_also_finishes_incumbent_comparison(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    choose(agent, lab, [MODEL])
    agent._execute('query', {'actions': [action(.4)]}, lab)
    agent.config = replace(agent.config, compute_budget=agent.costs['compute_cost'] + 1)
    result = choose(agent, lab, [MODEL])['evaluations'][0]
    assert len(calls) == 2 and len(calls[-1][1]) == 1
    assert 'retention_deferred' not in result
    assert result['retained_comparison']['paid_score_reuse']['status'] == 'reused'
    assert agent.evaluated_model['evaluated_rows'] == len(agent.rows)
    assert agent.costs['compute_cost'] == agent.config.compute_budget


def test_old_complete_score_stays_reusable_after_output_capture_change(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    first = choose(agent, lab, [MODEL])['evaluations'][0]
    assessment = json.loads(Path(first['paid_score_reuse']['execution_basis_ref']['file']).read_text())
    scope_path = Path(assessment['scope_ref']['file'])
    scope = json.loads(scope_path.read_text())
    contract = scope['evaluation_contract']
    contract['evaluator_source'] = contract['evaluator_source'].replace(', prediction_state=None', '').replace(
        ', prediction_state=prediction_state', '').replace(
        '        if temporal_representation and prediction_state is not None:\n'
        '            prediction_state.update(predictions=predictions, representations=representations)\n', '')
    scope_path.write_text(json.dumps(scope))
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    result = choose(agent, lab, [MODEL])['evaluations'][0]
    assert result['paid_score_reuse']['status'] == 'reused'
