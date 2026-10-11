"""Renew exhausted finite proposals only after informative reference research."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_response_hypotheses as response
from hegel_machine.cognition.world_response_constraints import response_family_grid_support
from test_world_response_control_preparation import case


PATH = 'shared/research.py'


def informed_state(*, target_failure=False):
    state, family, _ = case()
    state.update(target_numbers=[3], completed=[])
    rows = state['research_table']['references'] + state['research_table']['targets']
    for row in rows:
        for measurement in row['measurements'].values():
            measurement['evidence'] = 0
        if 'frequency' in row:
            row['frequency']['evidence'] = 0
    members = response._propose_fitted_responses(state, [family], PATH)
    prior, _ = response._revise_stalled_support(state, [family], members,
        {'family_research': {'no_progress_cycles': 2}}, PATH)
    assert prior['models']
    for model in prior['models']:
        state['response_settings'].append({'crystal': 3, 'frequency': model['target_setting'],
                                          'activated': False, 'evidence': [10]})
    assert not response._compatible_joint_models(state, prior, members)
    features = {row['crystal']: {key: value['value']
                for key, value in row['measurements'].items()} for row in rows}

    def projection():
        answer = response_family_grid_support(features, state['response_settings'],
                                             family, 3, state['control_grid'])
        assert answer['status'] == 'COMPLETE'
        return {'status': 'complete', 'target_number': 3, 'grid': deepcopy(state['control_grid']),
                'support': {key: answer[key] for key in ('intervals', 'grid_count')}}

    before = projection()
    if target_failure:
        observed = {(row['crystal'], row['frequency']) for row in state['response_settings']}
        grid = before['grid']
        setting = next(grid['origin'] + grid['step'] * index
            for left, right in before['support']['intervals'] for index in range(left, right + 1)
            if (3, grid['origin'] + grid['step'] * index) not in observed)
        event = {'crystal': 3, 'frequency': setting, 'activated': False, 'evidence': [100]}
    else:
        event = {'crystal': 1, 'frequency': 12., 'activated': True, 'evidence': [100]}
    previous = {'joint_revision': prior, 'assumption': {'supports': [family]},
        'family_research': {'selected_test': {key: event[key] for key in ('crystal', 'frequency')},
            'response_count': len(state['response_settings']), 'operational_search': before,
            'no_progress_cycles': 4}}
    state['response_research'] = {'conditional_simplification': previous}
    state['response_settings'].append(event)
    members = response._propose_fitted_responses(state, [family], PATH)
    after = projection()
    assert after['support']['grid_count'] < before['support']['grid_count']
    members['family_research'] = {'operational_search': after, 'no_progress_cycles': 5}
    return state, family, members, previous, event


def revise(state, family, members, previous):
    # The real caller has already replaced this argument's research with the
    # new comparison; the previous experiment must come from the saved state.
    return response._revise_stalled_support(state, [family], members,
        {**previous, 'family_research': members['family_research']}, PATH)


def record_fits(monkeypatch):
    calls = []
    original = response._append_joint_responses

    def record(result, state, supports, path):
        calls.append([family['id'] for family in supports])
        return original(result, state, supports, path)

    monkeypatch.setattr(response, '_append_joint_responses', record)
    return calls


def test_new_reference_information_renews_one_finite_batch(monkeypatch):
    state, family, members, previous, event = informed_state()
    old_batch = deepcopy(previous['joint_revision']['models'])
    before_state = deepcopy(state)
    # Earlier ordinary proposal work does not consume this research result.
    members['joint_parameter_search'] = [{'family': family['id']}]
    calls = record_fits(monkeypatch)
    revision, trial = revise(state, family, members, previous)
    assert calls == [[family['id']]]
    assert state == before_state
    assert trial is not None
    assert revision['models'] != old_batch
    before_count = previous['family_research']['operational_search']['support']['grid_count']
    after_count = members['family_research']['operational_search']['support']['grid_count']
    assert after_count < before_count
    assert revision['renewal'] == {
        'event': event, 'support_before_after': [before_count, after_count]}
    assert response._compatible_joint_models(state, revision, members)

    # Recomparison retains the new batch. Its own failed predictions do not
    # consume the same reference result a second time, even with old metadata.
    previous['joint_revision'] = deepcopy(revision)
    for model in revision['models']:
        state['response_settings'].append({'crystal': 3, 'frequency': model['target_setting'],
                                          'activated': False, 'evidence': [101]})
    repeated_members = response._propose_fitted_responses(state, [family], PATH)
    repeated_members['family_research'] = deepcopy(members['family_research'])
    repeated, pending = revise(state, family, repeated_members, previous)
    assert calls == [[family['id']]]
    assert pending is None
    assert repeated == revision


def test_target_failure_does_not_renew_even_when_its_grid_point_is_eliminated(monkeypatch):
    state, family, members, previous, _ = informed_state(target_failure=True)
    calls = record_fits(monkeypatch)
    revision, trial = revise(state, family, members, previous)
    assert calls == [] and trial is None
    assert revision == previous['joint_revision']


@pytest.mark.parametrize('unavailable', ('no_contraction', 'unknown', 'changed_grid', 'no_new_response'))
def test_reference_result_requires_a_comparable_paid_contraction(monkeypatch, unavailable):
    state, family, members, previous, _ = informed_state()
    after = members['family_research']['operational_search']
    if unavailable == 'no_contraction':
        after['support'] = deepcopy(previous['family_research']['operational_search']['support'])
    elif unavailable == 'unknown':
        after['status'] = 'unknown'
    elif unavailable == 'changed_grid':
        after['grid']['step'] = .5
    else:
        previous['family_research']['response_count'] = len(state['response_settings'])
    calls = record_fits(monkeypatch)
    revision, trial = revise(state, family, members, previous)
    assert calls == [] and trial is None
    assert revision == previous['joint_revision']


def test_useful_paid_models_survive_informative_reference_return(monkeypatch):
    state, family, members, previous, _ = informed_state()
    valid_members = response._propose_fitted_responses(state, [family], PATH)
    valid, _ = response._revise_stalled_support(state, [family], valid_members,
        {'family_research': {'no_progress_cycles': 2}}, PATH)
    assert response._compatible_joint_models(state, valid, valid_members)
    previous['joint_revision'] = deepcopy(valid)
    calls = record_fits(monkeypatch)
    revision, trial = revise(state, family, members, previous)
    assert calls == [] and trial is not None
    assert revision['models'] == valid['models']
    assert 'renewal' not in revision
