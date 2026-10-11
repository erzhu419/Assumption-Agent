"""Explicit experiments carry control preparation through the public operator."""

import json

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from test_world_authored_response import _ready
from test_world_control_preparation import SaturatingLab
from test_world_reference_hypothesis import PATH


def arguments():
    return {'operator': 'reference_response', 'path': PATH, 'experiment': {
        'number': 1, 'setting': 15., 'restore_frequency': 15.8,
        'control_preparation': {'boundary': 0.,
            'grid': {'origin': 0., 'step': 1., 'lower': 0., 'upper': 10000.},
            'restore_settings': [16., 15.]}}}


def test_explicit_prepared_reference_observes_boundary_and_tests_recovery(tmp_path):
    agent, lab = _ready(tmp_path, lab_type=SaturatingLab, intercept=5.8)
    ordinary = {'operator': 'reference_response', 'path': PATH,
                'experiment': {'number': 1, 'setting': 15.}}
    before = len(lab.actions)
    assert agent._execute('run', ordinary, lab)['status'] == 'unreachable_setting'
    assert len(lab.actions) == before

    result = agent._execute('run', arguments(), lab)

    assert result['requested_setting_reached'] and result['prediction_status'] == 'refuted'
    assert result['answer']['actual_frequency'] == 15.
    state = json.loads((agent.workspace / recipe.STATE).read_text())
    trial = next(item for item in state['tests'] if item['question'] == result['question_id'])
    assert trial['control_preparation'] == arguments()['experiment']['control_preparation']
    assert recipe._dialog(recipe._view(agent.observations[trial['preparation_evidence'][0]])) == (1, 0.)
    recovery = result['setting_recoveries'][-1]
    assert recovery['actual_frequency'] == 16. and recovery['restored']
    assert recovery['original_frequency'] == pytest.approx(15.8)
    assert lab.active(1) and agent.costs['model_calls'] == 0


@pytest.mark.parametrize('revise', (False, True))
def test_interrupted_preparation_resumes_or_restores_before_a_new_experiment(tmp_path, revise):
    agent, lab = _ready(tmp_path, lab_type=SaturatingLab, intercept=5.8)
    first = agent._execute('run', {**arguments(), 'max_steps': 2}, lab)
    assert first['status'] == 'method_step_limit'
    assert lab.frequencies[1] == 0.
    followup = {'operator': 'reference_response', 'path': PATH, 'experiment': {
        'number': 4 if revise else 1, 'setting': 166. if revise else 15.}}
    result = agent._execute('run', followup, lab)
    assert result['requested_setting_reached']
    assert result['setting_recoveries'][-1]['actual_frequency'] == 16.
    assert result['setting_recoveries'][-1]['restored'] and lab.active(1)
    if revise:
        assert result['superseded_experiment']['question'] == first['question_id']
        assert result['prediction_status'] == 'supported'
        assert not any(row['frequency'] == 15. for row in result['observed_responses'])
    else:
        assert result['question_id'] == first['question_id']
        assert result['prediction_status'] == 'refuted'
    assert agent.costs['model_calls'] == 0
