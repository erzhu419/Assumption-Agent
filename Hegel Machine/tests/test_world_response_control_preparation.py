"""A new control origin is an experiment, with separately tested restoration."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition import world_response_hypotheses as response
from hegel_machine.cognition.world_response_consequences import select_response_grid_trial


def case():
    references = [{'crystal': n, 'measurements': {'signal': {'value': n}},
                   'frequency': {'value': frequency}}
                  for n, frequency in ((1, 12.3), (2, 14.3))]
    state = {'research_table': {'references': references, 'targets': [
        {'crystal': 3, 'measurements': {'signal': {'value': 10}}}]},
        'response_settings': [{'crystal': n, 'frequency': frequency + delta,
                               'activated': delta == 0}
            for n, frequency in ((1, 12.3), (2, 14.3)) for delta in (-1, 0, 1)],
        'control_grid': {'step': 1., 'origin': 0., 'lower': 0., 'upper': 100.},
        'control_grids': {'1': {'step': 1., 'origin': .3, 'lower': 0., 'upper': 100.}}}
    family = {'id': 'linear', 'basis': [Program('feature', name='signal').to_dict()]}
    result = {'target_number': 3, 'status': 'family_research_needed', 'selection': None,
        'selected_number': None, 'selected_setting': None, 'compute_cost': 0,
        'assumptions': [], 'family_research': {'research_obligation': {}, 'no_progress_cycles': 4},
        'nested_supports': [family],
        'candidates': [{'id': 'point', 'status': 'compatible', 'fit_sources': [{'family': 'linear'}]}],
        'hypotheses': [{'id': 'point', 'lower': 0., 'upper': 0.,
                       'predictions_by_crystal': {'1': 12.3, '2': 14.3, '3': 30.3}}],
        'experiments': [{'id': 'reference', 'number': 1, 'prediction': 14.3,
                        'restore_frequency': 12.3,
                        'procedure': {'path': 'shared/research.py', 'max_steps': 512},
                        'execution_binding': 'parent resumes the selected experiment',
                        'predictions': {'point': False}}]}
    return state, family, result


def test_preparation_changes_candidates_without_assuming_a_changed_or_restored_control():
    state, _, result = case()
    before = deepcopy((state, result))
    prepared = response._prepared_reference_experiments(state, result)
    assert (state, result) == before
    assert [experiment['prediction'] for experiment in prepared] == [12., 13.]
    assert all(experiment['predictions'] == {'point': False} for experiment in prepared)
    for experiment in prepared:
        assert experiment['restore_frequency'] == 12.3
        assert experiment['procedure'] == result['experiments'][0]['procedure']
        assert experiment['execution_binding'] == result['experiments'][0]['execution_binding']
        assert experiment['control_preparation'] == {
            'boundary': 0., 'grid': {'step': 1., 'origin': 0., 'lower': 0., 'upper': 100.},
            'restore_settings': [12., 13.]}


@pytest.mark.parametrize('origin', (0., 12.))
def test_already_available_lattice_does_not_propose_preparation(origin):
    state, _, result = case()
    state['control_grids']['1']['origin'] = origin
    assert response._prepared_reference_experiments(state, result) == []


def test_preparation_requires_a_paid_positive_and_omits_observed_trials():
    state, _, result = case()
    state['response_settings'].append({'crystal': 1, 'frequency': 12., 'activated': False})
    assert [e['prediction'] for e in response._prepared_reference_experiments(state, result)] == [13.]
    state['response_settings'] = [row for row in state['response_settings']
                                  if (row['crystal'], row['frequency']) != (1, 12.3)]
    assert response._prepared_reference_experiments(state, result) == []


def test_full_family_consequences_can_value_a_prepared_test_with_all_false_members():
    state, family, result = case()
    result['experiments'] = response._prepared_reference_experiments(state, result)
    selected = select_response_grid_trial(state, [family], result)
    experiment = selected['selection']['selected_experiment']
    assert experiment['predictions'] == {'point': False}
    assert 'control_preparation' in experiment
    assessment = selected['family_research']['operational_search']
    assert assessment['support']['grid_count'] == 18
    value = next(item for item in assessment['experiments'] if item['crystal'] == 1
                 and item['frequency'] == experiment['prediction'])
    assert min(value['reductions_by_outcome']) == 2
    assert value['outcome_statuses'] == ['complete', 'complete']
    assert selected['selection']['guaranteed_search_reduction'] == 2
    assert selected['selection']['guaranteed_nonlocal_eliminations'] == 0


@pytest.mark.parametrize('phase', ('before_revision', 'pending_joint_trial', 'exhausted'))
def test_preparation_is_adopted_only_after_the_paid_joint_batch_is_exhausted(monkeypatch, phase):
    from hegel_machine.cognition import world_response_consequences as consequences

    state, family, members = case()
    monkeypatch.setattr(response, '_propose_response_members', lambda *_: deepcopy(members))
    monkeypatch.setattr(consequences, 'assess_response_consequences',
                        lambda _state, _families, result, **_: result)
    revision = None if phase == 'before_revision' else {'models': []}
    trial = ({'selected_experiment': {'id': 'joint', 'number': 3, 'prediction': 31.}}
             if phase == 'pending_joint_trial' else None)
    monkeypatch.setattr(response, '_revise_stalled_support',
                        lambda *_: (deepcopy(revision), deepcopy(trial)))
    result = response.propose_response_research(state, [family], 'shared/research.py')
    experiment = (result['selection'] or {}).get('selected_experiment', {})
    assert ('control_preparation' in experiment) == (phase == 'exhausted')
    if phase == 'exhausted':
        assert result['conditional_simplification']['adopted']
        assert result['selection']['guaranteed_search_reduction'] == 2
        assert result['selection']['consequence_scope'] == 'assumed_simpler_supports'
        assert experiment['restore_frequency'] == 12.3
        assert result['compute_cost'] >= result['conditional_simplification']['compute_cost'] > 0
    elif phase == 'pending_joint_trial':
        assert result['status'] == 'conditional_joint_response_trial'
        assert experiment['prediction'] == 31.
