"""Reuse already scored paid predictions as row-aligned model feedback."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import online_inquiry_llm
from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_online_research import prepared
from test_common_archive_resume import execute
from test_inquiry_revision import PublicLab, action


class HistoryLab(PublicLab):
    def execute(self, requested):
        record = super().execute(requested)
        for row in record['rows']:
            row['target'] += sum(point['u'] for point in row['history'])
        return record


class HistoryConfirmationLab(HistoryLab):
    def confirmation_actions(self, length, rng):
        return [action(*([2.] * length))]


def scene(path, arm):
    agent, lab = prepared(path, arm), HistoryLab()
    execute(agent, lab, 'query', {'actions': [action(1., 1., 1., 1., 1.)]})
    return agent, lab


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_existing_local_checks_reach_candidate_feedback_without_binding(tmp_path, arm):
    agent, lab = scene(tmp_path, arm)
    answer = {'status': 'supported', 'answer': 'Test an authored local contrast',
        'evidence': [0, 4], 'checks': [{'id': 'end_minus_start',
            'terms': [{'row': 4, 'coefficient': 1}, {'row': 0, 'coefficient': -1}],
            'relation': 'eq', 'expected': 0., 'atol': 1e-9,
            'scope': {'trajectory': 'already purchased'}}]}
    execute(agent, lab, 'write', {'path': 'local.py', 'code':
        f'def readout(result,spec,memory): return {answer!r}\n'})
    execute(agent, lab, 'run', {'operator': 'analyze_evidence', 'path': 'local.py',
        'research_question': {'id': 'local', 'question': 'Does this measured contrast vanish?',
                              'parent_id': 'task', 'depends_on': []}})
    before = deepcopy(agent.costs)
    result = execute(agent, lab, 'evaluate', {'model': 'relation_model.py',
        'code': "def predict(row):\n    assert 'target' not in row\n    return row['features']['u']\n"})
    diagnostic = result['paid_relation_diagnostic']
    check = diagnostic['revisions'][0]['examples'][0]
    assert check['measured_value'] == 4. and check['predicted_value'] == 0.
    assert check['residual'] == -4. and check['evidence_status'] == 'violated'
    assert diagnostic['model'] == 'relation_model.py'
    assert diagnostic['workspace'] == str(tmp_path)
    assert diagnostic['evaluation_scope'] == result['evaluation_scope'] == 'paid_fit'
    assert result['returncode'] == 0 and result['mse'] == 6.
    assert agent.research_input_bindings == {} and result['research_bindings'] == []
    assert agent.costs['compute_cost'] - before['compute_cost'] == 7
    assert agent.costs['observation_cost'] == before['observation_cost']
    # execute() accounts one scripted decision; scoring adds no provider turn.
    assert agent.costs['model_calls'] == before['model_calls'] + 1 and not agent.confirmations
    assert agent.research_candidates['relation_model.py']['current_evaluation'][
        'paid_relation_diagnostic'] == diagnostic
    request = agent._request(lab)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    assert context['events'][-1]['feedback']['paid_relation_diagnostic'] == diagnostic


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_same_controls_distinct_histories_return_exact_paid_error_rows(tmp_path, arm):
    agent, lab = scene(tmp_path, arm)
    result = execute(agent, lab, 'evaluate', {'model': 'current_only.py',
        'code': "def predict(row): return row['features']['u']\n"})
    diagnostic = result['paid_residuals']

    assert diagnostic['evaluation_scope'] == 'paid_response_evaluation'
    assert diagnostic['model'] == 'current_only.py' and diagnostic['evaluated_rows'] == 5
    assert [(row['row_index'], row['prediction'], row['target'], row['squared_error'])
            for row in diagnostic['examples']] == [(4, 1., 5., 16.), (3, 1., 4., 9.), (2, 1., 3., 4.)]
    for row in diagnostic['examples']:
        assert row['features'] == {'u': 1.}
        assert tuple(row['history']) == agent.rows[row['row_index']].history
        assert len(row['history']) == row['row_index']
    assert result['mse'] == 6.0 and 'predictions' not in result
    assert 'fresh_confirmation' not in result and not agent.confirmations

    # Shared provider compaction must not discard the history that explains
    # why otherwise identical controls have different prediction errors.
    request = agent._request(lab)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='test')
    shown = context['events'][-1]['feedback']['paid_residuals']
    assert shown == diagnostic


def test_attempted_same_path_edit_is_not_misattributed_to_retained_incumbent(tmp_path):
    agent, lab = scene(tmp_path, 'hegel')
    execute(agent, lab, 'write', {'path': 'contrast.py', 'code':
        "def readout(result,spec,memory):\n"
        " return {'status':'supported','answer':'Check the measured difference',"
        "'evidence':[0,4],'checks':[{'id':'difference','terms':[{'row':4,'coefficient':1.},"
        "{'row':0,'coefficient':-1.}],'relation':'eq','expected':0.,'atol':1e-9}]}\n"})
    execute(agent, lab, 'run', {'operator': 'analyze_evidence', 'path': 'contrast.py',
        'research_question': {'id': 'contrast', 'question': 'Does the difference vanish?',
                              'parent_id': 'task', 'depends_on': []}})
    correct = "def predict(row): return row['features']['u'] + sum(p['u'] for p in row['history'])\n"
    incumbent = execute(agent, lab, 'evaluate', {'model': 'model.py', 'code': correct})
    retained = deepcopy(incumbent['evaluated_model'])

    wrong = execute(agent, lab, 'evaluate', {'model': 'model.py',
        'code': "def predict(row): return row['features']['u']\n"})

    assert wrong['paid_residuals']['model'] == 'model.py'
    assert wrong['paid_residuals']['model'] != retained['model']
    assert wrong['paid_residuals']['examples'][0]['prediction'] == 1.
    assert wrong['paid_relation_diagnostic']['model'] == 'model.py'
    assert wrong['paid_relation_diagnostic']['model'] != retained['model']
    assert wrong['paid_relation_diagnostic']['workspace'] == str(tmp_path)
    assert wrong['paid_relation_diagnostic']['revisions'][0]['examples'][0]['residual'] == -4.
    assert incumbent['paid_relation_diagnostic']['revisions'][0]['examples'][0]['residual'] == 0.
    assert wrong['evaluated_model'] == retained and retained['mse'] == 0.
    assert (tmp_path / retained['model']).read_text() == correct


def test_execution_failure_does_not_invent_prediction_witnesses(tmp_path):
    agent, lab = scene(tmp_path, 'code_react')
    result = execute(agent, lab, 'evaluate', {'model': 'broken.py',
        'code': "def predict(row): raise ValueError('model shape mismatch')\n"})

    assert result['returncode'] != 0 and 'model shape mismatch' in result['stderr']
    assert 'paid_residuals' not in result and 'mse' not in result
    assert agent.evaluated_model is None and not agent.confirmations


def test_error_examples_use_one_existing_score_without_extra_costs(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path, 'hegel')
    execute(agent, lab, 'write', {'path': 'model.py', 'code': 'def predict(row): return 0.\n'})
    before = deepcopy(agent.costs)
    calls = []
    original = online_inquiry_llm.evaluate_python_model

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    times = iter((10., 12.5))
    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model', counted)
    monkeypatch.setattr(online_inquiry_llm, 'monotonic', lambda: next(times))
    result = agent._score('model.py')

    assert calls == ['model.py'] and len(result['paid_residuals']['examples']) == 3
    assert agent.costs == {**before, 'compute_cost': before['compute_cost'] + 5,
                          'python_execution_seconds': before['python_execution_seconds'] + 2.5}
    assert lab.paid == 5 and len(lab.actions) == 1
    assert not agent.confirmations


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_fresh_residuals_align_new_rows_and_frozen_same_path_attempts(tmp_path, arm, monkeypatch):
    agent, lab = prepared(tmp_path, arm), HistoryConfirmationLab()
    execute(agent, lab, 'query', {'actions': [action(1., 1., 1., 1., 1.)]})
    first_source = 'def predict(row): return 1.\n'
    execute(agent, lab, 'write', {'path': 'model.py', 'code': first_source})
    before = deepcopy(agent.costs)
    calls = []
    original = online_inquiry_llm.evaluate_python_model

    def counted(workspace, model, rows, **kwargs):
        calls.append((model, len(rows), len(agent.rows)))
        return original(workspace, model, rows, **kwargs)

    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model', counted)
    first = execute(agent, lab, 'finish', {'model': 'model.py'})['fresh_confirmation']
    residuals = first['residuals']
    assert first['passed'] is False and first['mse'] == 5.
    assert residuals['evaluation_scope'] == 'fresh_confirmation'
    assert residuals['model'] == first['submitted_candidate']['model'] != 'model.py'
    assert [(row['row_index'], row['target'], row['prediction'], row['squared_error'])
            for row in residuals['examples']] == [(6, 4., 1., 9.), (5, 2., 1., 1.)]
    assert tuple(residuals['examples'][0]['history']) == ({'u': 2.},)
    assert calls == [('model.py', 5, 5), ('model.py', 2, 5)]
    assert agent.costs['compute_cost'] - before['compute_cost'] == 7
    assert agent.costs['observation_cost'] - before['observation_cost'] == 2
    assert first['observation_indices'] == [5, 6] and len(agent.rows) == 7

    execute(agent, lab, 'write', {'path': 'model.py', 'code': 'def predict(row): return -1.\n'})
    second = execute(agent, lab, 'finish', {'model': 'model.py'})['fresh_confirmation']
    assert second['residuals']['model'] != residuals['model']
    assert [(row['row_index'], row['prediction'], row['squared_error'])
            for row in second['residuals']['examples']] == [(8, -1., 25.), (7, -1., 9.)]
    assert (tmp_path / residuals['model']).read_text() == first_source
    assert first['residuals'] == residuals
    assert agent._evidence_notebook()['facts']['confirmation:0']['residuals'] == residuals
    assert agent._evidence_notebook()['facts']['confirmation:1']['residuals'] == second['residuals']
    # finish retains its paid fit before confirming; the next finish must pay
    # to compare that saved candidate on the two newly acquired rows as well.
    assert calls == [('model.py', 5, 5), ('model.py', 2, 5), ('model.py', 7, 7),
                     ('_evaluated_model/model.py', 7, 7), ('model.py', 2, 7)]
    assert agent.costs['compute_cost'] - before['compute_cost'] == 23


def test_failed_fresh_prediction_retains_acquisition_without_inventing_residuals(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), HistoryConfirmationLab()
    execute(agent, lab, 'query', {'actions': [action(1.)]})
    execute(agent, lab, 'write', {'path': 'model.py', 'code':
        "def predict(row):\n if row['features']['u'] == 2.: raise ValueError('fresh input error')\n return 1.\n"})
    result = execute(agent, lab, 'finish', {'model': 'model.py'})
    fresh = result['fresh_confirmation']
    assert fresh['returncode'] != 0 and 'fresh input error' in fresh['stderr']
    assert fresh['passed'] is False and 'residuals' not in fresh
    assert fresh['observation_indices'] == [1, 2]
    assert lab.paid == len(agent.rows) == 3


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
@pytest.mark.parametrize('later_value', [0., 2.])
def test_frozen_response_distinguishes_fitting_rows_from_later_predictions(tmp_path, monkeypatch,
                                                                        arm, later_value):
    agent, lab = prepared(tmp_path, arm), HistoryLab()
    execute(agent, lab, 'query', {'actions': [action(0., 0., 0., 0., 0.)]})
    fitted = execute(agent, lab, 'evaluate', {'model': 'mean.py', 'code':
        '_mean = None\ndef fit(rows):\n global _mean\n'
        " _mean = sum(row['target'] for row in rows) / len(rows)\n"
        'def predict(row): return _mean\n'})
    model = fitted['evaluated_model']['model']
    assert fitted['evaluation_scope'] == 'paid_fit'
    execute(agent, lab, 'query', {'actions': [action(later_value, later_value)]})
    before, paid = deepcopy(agent.costs), lab.paid
    calls = []
    original = online_inquiry_llm.evaluate_python_model

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model', counted)
    result = execute(agent, lab, 'evaluate', {'model': model})
    expected_later = (later_value ** 2 + (2 * later_value) ** 2) / 2
    assert result['mse'] == pytest.approx(expected_later * 2 / 7)
    assert result['evaluation_scope'] == 'paid_prediction'
    assert result['fitting_view'] == {'scope': 'current_paid_prefix',
        'file': '_evaluated_model/model_observations.json', 'rows': 5,
        'response_row_range': [0, 4]}
    assert result['later_readings'] == {'rows': 2, 'response_row_range': [5, 6],
        'mse': expected_later, 'max_abs_error': 2 * later_value}
    summary = result['evaluated_model']
    assert summary['support'] == summary['evaluation_scope'] == 'paid_prediction'
    assert summary['later_readings'] == result['later_readings']
    fact = agent._evidence_notebook()['facts']['current_paid_evaluation']
    assert fact['kind'] == 'paid_prediction' and fact['later_readings'] == result['later_readings']
    assert calls == [model] and agent.costs['compute_cost'] - before['compute_cost'] == 7
    assert lab.paid == paid == 7 and agent.costs['observation_cost'] == before['observation_cost']
    assert not agent.confirmations


def test_frozen_current_score_keeps_existing_confirmation_and_routing_cost(tmp_path):
    from test_python_research_advice import prepared as temporal_scene
    from hegel_machine.cognition.python_research_routing import evaluate_action

    agent, lab = temporal_scene(tmp_path, mode='finite2')
    learned = agent._execute('run', {'operator': 'history', 'base_lags': 3,
                                   'path': 'history.py'}, lab)
    fitted = agent._execute('evaluate', {'model': learned['path']}, lab)
    model = fitted['evaluated_model']['model']
    agent._execute('query', {'actions': [action(.2, -.4, .8)]}, lab)
    before = deepcopy(agent.costs)
    scored = agent._execute('evaluate', {'model': model}, lab)
    assert scored['evaluated_model']['support'] == 'paid_prediction'
    assert scored['research_advice']['next_action'] == {
        'action': 'finish', 'arguments': {'model': model}}
    assert agent.costs['compute_cost'] - before['compute_cost'] == len(agent.rows)
    offer = evaluate_action(agent, model, 'current_evaluated_candidate')
    assert offer['compute_cost'] == 2 * len(agent.rows) + agent.config.confirmation_rows


def test_nonprefix_fitting_view_does_not_invent_a_later_row_cutoff(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), HistoryLab()
    execute(agent, lab, 'query', {'actions': [action(1., 2., 3.)]})
    fitted = execute(agent, lab, 'evaluate', {'model': 'model.py', 'code':
        '_mean = None\ndef fit(rows):\n global _mean\n'
        " _mean = sum(row['target'] for row in rows) / len(rows)\n"
        'def predict(row): return _mean\n'})
    # An explicit selected fitting view can contain a subset, as trajectory refit does.
    view = tmp_path / '_evaluated_model/model_observations.json'
    rows = json.loads(view.read_text())
    view.write_text(json.dumps([rows[1]]))
    result = execute(agent, lab, 'evaluate', {'model': fitted['evaluated_model']['model']})
    assert result['evaluation_scope'] == 'paid_evaluation'
    assert result['fitting_view']['rows'] == 1 and result['fitting_view']['scope'] == 'unknown'
    assert 'response_row_range' not in result['fitting_view'] and 'later_readings' not in result
    assert result['evaluated_model']['support'] == 'paid_evaluation'
    assert agent._evidence_notebook()['facts']['current_paid_evaluation']['kind'] == 'paid_evaluation'


def test_same_scene_restore_keeps_frozen_prediction_scope(tmp_path):
    from test_common_archive_resume import checkpoint, agent_at, SEED
    from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab

    old, lab, _ = checkpoint(tmp_path)
    frozen = old._evaluated_summary()['model']
    scored = execute(old, lab, 'evaluate', {'model': frozen})
    assert scored['evaluation_scope'] == 'paid_prediction'
    previous = old.result('model_call_budget_exhausted')
    resumed, apparatus = agent_at(tmp_path / 'continued'), TemporalInquiryLab(task_index=1)
    restored = resumed.restore_research(previous, old.workspace, apparatus, confirmation_seed=SEED)
    assert restored['new_observations'] == 0 and resumed.costs == old.costs
    assert resumed._evaluated_summary() == old._evaluated_summary()
    fact = resumed._evidence_notebook()['facts']['current_paid_evaluation']
    assert fact['kind'] == 'paid_prediction'
    assert fact['fitting_view'] == scored['fitting_view']
    assert fact['later_readings'] == scored['later_readings']
