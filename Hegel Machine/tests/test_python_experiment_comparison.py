"""Runtime checks for author-supplied Python experiments, not discovery results."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_common_archive_online_research import (
    CHILD, NEW_MODEL, PARENT, READOUT, ConfirmationLab, prepared,
)
from test_inquiry_revision import PublicLab, action, write


def compare(agent, lab, experiments, models=None):
    result = agent._execute('run', {
        'operator': 'compare_experiments',
        'models': models or ['old.py', 'new.py'],
        'experiments': deepcopy(experiments),
    }, lab)
    assert result['operator'] == 'compare_experiments'
    return result['comparison']


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_authored_pool_is_compared_before_purchase_and_result_returns_to_parent(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), PublicLab()
    agent._execute('query', {'actions': [action(0)], 'question': PARENT}, lab)
    write(agent, 'readout.py', READOUT, lab)
    experiments = [
        {'id': 'agreement', 'actions': [action(0, 0)]},
        {'id': 'contrast', 'actions': [action(2), action(3)]},
    ]
    before_actions = deepcopy(lab.actions)
    before_rows = deepcopy(agent.rows)

    comparison = compare(agent, lab, experiments)

    assert lab.actions == before_actions and lab.paid == 1
    assert agent.rows == before_rows and agent.costs['observation_cost'] == 1
    assert agent.revisions == []  # A forecast comparison has not run an experiment.
    assert comparison['evidence_cutoff'] == {'records': 1, 'rows': 1}
    assert comparison['eligible_models'] == ['old.py', 'new.py']
    assert all(item['mse'] == 0 for item in comparison['prior_fit'])
    selected = comparison['selected_experiment']
    assert selected['id'] == 'contrast'
    assert selected['actions'] == experiments[1]['actions']
    panels = {item['id']: item for item in comparison['comparisons']}
    assert panels['agreement']['predicted_separation'] == 0
    assert panels['contrast']['predicted_separation'] > 0
    assert [item['predictions'] for item in panels['contrast']['forecasts']] == [[0., 0.], [2., 3.]]
    start = panels['contrast']['row_start']
    stop = start + panels['contrast']['row_count']
    for forecast in comparison['forecasts']:
        inputs = forecast['prediction_inputs'][start:stop]
        assert all(not row['history'] for row in inputs)
        assert all('target' not in row for row in inputs)
    assert agent.experiment_comparisons[0] == comparison
    state = json.loads((tmp_path / 'research_state.json').read_text())
    assert state['experiment_comparisons'][0]['selected_experiment']['id'] == 'contrast'

    proposal = {**selected['revision'], 'readout': 'readout.py'}
    observed = agent._execute('query', {
        'actions': selected['actions'], 'revision': proposal, 'question': CHILD,
    }, lab)
    record = observed['revision']
    assert lab.actions == [action(0), action(2), action(3)]
    assert record['evidence_cutoff'] == comparison['evidence_cutoff']
    assert record['proposal']['comparison_index'] == comparison['index']
    assert [item['prospective_mse'] for item in record['forecasts']] == [6.5, 0.]
    assert record['authored_answer']['answer'] == {'rejected': ['old.py'], 'surviving': ['new.py']}

    request = agent._request(lab)
    returned = next(item for item in request['online_research']['parent_returns']
                    if item['parent_id'] == 'parent')['answers'][-1]
    assert returned['status'] == 'refuted'
    selected_model = returned['answer']['surviving'][0]
    evaluation = agent._execute('evaluate', {
        'model': selected_model, 'research_question': {'id': 'parent'},
    }, lab)
    assert evaluation['mse'] == 0
    parent = next(item for item in agent.questions if item['id'] == 'parent')
    assert parent['reviewed_child_answers'][-1]['answer']['surviving'] == [selected_model]
    assert lab.paid == 3  # Analysis and the parent's revised decision do not buy labels.


def test_comparison_fits_only_current_paid_rows_and_does_not_mutate_training_view(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    source = """def fit(rows):
    assert len(rows) == 1 and rows[0]['target'] == 0, 'future labels reached fitting'
def predict(row): return row['features']['u']
"""
    write(agent, 'new.py', source, lab)
    evidence_before = (tmp_path / 'observations.json').read_text()
    result = compare(agent, lab, [{'id': 'candidate', 'actions': [action(4, 5)]}])
    assert result['selected_experiment']['id'] == 'candidate'
    forecast = next(item for item in result['forecasts'] if item['model'] == 'new.py')
    assert forecast['returncode'] == 0 and forecast['predictions'] == [4., 5.]
    assert forecast['source']['files']['new.py'] == source
    assert (tmp_path / 'observations.json').read_text() == evidence_before
    assert lab.paid == 1


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_failed_or_paid_refuted_models_cannot_drive_experiment_selection(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), PublicLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    # This already-refuted predictor would make the first experiment appear
    # more informative than the useful second one if it remained eligible.
    write(agent, 'refuted.py', "def predict(row): return 1000. if row['features']['u'] <= 1 else 0.\n", lab)
    write(agent, 'broken.py', "def predict(row): raise ValueError('unfinished candidate')\n", lab)
    result = compare(agent, lab, [
        {'id': 'refuted_distraction', 'actions': [action(1)]},
        {'id': 'supported_contrast', 'actions': [action(4)]},
    ], models=['old.py', 'new.py', 'refuted.py', 'broken.py'])
    assert result['eligible_models'] == ['old.py', 'new.py']
    assert result['selected_experiment']['id'] == 'supported_contrast'
    fits = {item['model']: item for item in result['prior_fit']}
    assert fits['refuted.py']['mse'] > result['mse_threshold']
    assert fits['broken.py'].get('returncode', 1) != 0
    assert lab.paid == 1 and len(lab.actions) == 1


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_agreement_needs_no_experiment_and_does_not_block_fresh_finish(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), ConfirmationLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    write(agent, 'old.py', NEW_MODEL, lab)
    result = compare(agent, lab, [
        {'id': 'first', 'actions': [action(1)]},
        {'id': 'second', 'actions': [action(2)]},
    ])
    assert result['selected_experiment'] is None
    assert result['eligible_models'] == ['old.py', 'new.py']
    assert all(item['predicted_separation'] == 0 for item in result['comparisons'])
    assert lab.paid == 1
    agent._execute('evaluate', {'model': 'final.py', 'code': NEW_MODEL}, lab)
    finished = agent._execute('finish', {'model': 'final.py'}, lab)
    assert finished['fresh_confirmation']['passed']
    assert lab.paid == 3 and len(lab.actions) == 2


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
@pytest.mark.parametrize('resource', ['compute', 'python'])
def test_exhausted_comparison_budget_does_not_purchase_or_recommend(tmp_path, arm, resource, monkeypatch):
    from hegel_machine.cognition import inquiry_revision

    agent, lab = prepared(tmp_path, arm), PublicLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    if resource == 'compute':
        agent.config = replace(agent.config, compute_budget=0)
    else:
        agent.costs['python_execution_seconds'] = agent.config.execution_seconds

    def must_not_execute(*args, **kwargs):
        raise AssertionError('candidate executed after its budget was exhausted')

    monkeypatch.setattr(inquiry_revision, 'evaluate_python_model', must_not_execute)
    result = compare(agent, lab, [{'id': 'candidate', 'actions': [action(2)]}])
    assert result['selected_experiment'] is None
    assert agent.costs['compute_cost'] <= agent.config.compute_budget
    assert agent.costs['python_execution_seconds'] <= agent.config.execution_seconds
    assert lab.paid == 1 and len(lab.actions) == 1


def test_partial_compute_budget_cannot_choose_from_an_incomplete_comparison(tmp_path):
    agent, lab = prepared(tmp_path, 'hegel'), PublicLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    agent.config = replace(agent.config, compute_budget=3)
    result = compare(agent, lab, [{'id': 'candidate', 'actions': [action(2, 3)]}])
    assert result['selected_experiment'] is None
    assert agent.costs['compute_cost'] <= 3
    assert lab.paid == 1
