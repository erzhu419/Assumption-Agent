"""Task-value selection through the real Python comparison and inquiry state."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_common_archive_online_research import prepared
from test_inquiry_revision import PublicLab, action, write
from scripts.run_anonymous_online_research import experimental_design_trace


def value_problem(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm), PublicLab()
    agent._execute('query', {'actions': [action(0)]}, lab)
    models = []
    for relevant in (0, 1):
        for nuisance in (0, 1):
            model = f'candidate_{relevant}_{nuisance}.py'
            write(agent, model, f'''def fit(rows):
    assert len(rows) == 1 and rows[0]['target'] == 0

def predict(row):
    u = row['features']['u']
    if u == 1: return {100 * nuisance}.
    if u in (2, 3): return {relevant}.
    return 0.
''', lab)
            models.append(model)
    args = {'operator': 'compare_experiments', 'models': models,
            'experiments': [
                {'id': 'nuisance', 'actions': [action(1)]},
                {'id': 'task_relevant', 'actions': [action(2)]}],
            'reference_actions': [action(3)]}
    return agent, lab, args


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_reference_risk_changes_actual_recommendation_without_buying_reference_labels(tmp_path, arm):
    agent, lab, args = value_problem(tmp_path, arm)
    before = deepcopy(agent.rows)
    result = agent._execute('run', args, lab)['comparison']
    assert result['selected_experiment']['id'] == 'task_relevant'
    assert result['diagnosis'] == 'task_relevant_experiment_available'
    assert result['selection_policy'] == 'expected_reference_risk_reduction'
    assert [p['predicted_separation'] for p in result['comparisons']] == [2500., .25]
    values = {p['id']: p for p in result['task_value']['comparisons']}
    assert values['nuisance']['expected_risk_reduction'] == pytest.approx(0)
    assert values['task_relevant']['expected_risk_reduction'] == pytest.approx(.5)
    assert result['task_value']['reference_rows'] == 1
    assert agent.rows == before and lab.actions == [action(0)] and lab.paid == 1
    assert all('target' not in row for f in result['forecasts'] for row in f['prediction_inputs'])
    state = json.loads((tmp_path / 'research_state.json').read_text())
    assert state['experiment_comparisons'][-1]['task_value'] == result['task_value']
    assert agent._request(lab)['online_research']['latest_comparison']['task_value'] == result['task_value']
    trace = experimental_design_trace({'experiment_comparisons': agent.experiment_comparisons})
    assert trace['comparisons'][0]['task_value'] == result['task_value']
    acquired = agent._execute('query', result['query_arguments'], lab)
    assert lab.actions == [action(0), action(2)]
    assert acquired['revision']['proposal']['comparison_index'] == result['index']
    assert acquired['revision']['evidence_cutoff']['rows'] == 1


def test_unchanged_variance_interface_keeps_its_historical_behavior(tmp_path):
    agent, lab, args = value_problem(tmp_path, 'hegel')
    args.pop('reference_actions')
    result = agent._execute('run', args, lab)['comparison']
    assert result['selected_experiment']['id'] == 'nuisance'
    assert 'task_value' not in result
    assert lab.paid == 1


def test_reference_agreement_does_not_turn_nuisance_variance_into_task_value(tmp_path):
    agent, lab, args = value_problem(tmp_path, 'hegel')
    args['reference_actions'] = [action(0)]
    result = agent._execute('run', args, lab)['comparison']
    assert result['selected_experiment'] is None
    assert result['query_arguments'] is None
    assert result['diagnosis'] == 'no_expected_task_risk_reduction'
    assert result['task_value']['reference_risk'] == 0
    assert lab.paid == 1


def test_value_budget_exhaustion_does_not_fall_back_to_unrelated_variance(tmp_path):
    agent, lab, args = value_problem(tmp_path, 'hegel')
    agent.config = replace(agent.config, compute_budget=24)
    result = agent._execute('run', args, lab)['comparison']
    assert result['selected_experiment'] is None
    assert result['query_arguments'] is None
    assert result['diagnosis'] == 'comparison_budget_exhausted'
    assert result['task_value']['status'] == 'budget_exhausted'
    assert agent.costs['compute_cost'] <= 24
    assert lab.paid == 1
