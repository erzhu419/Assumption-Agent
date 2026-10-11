"""Saved coefficient programs must answer new evidence with newly fitted bands."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import world_response_hypotheses as response
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from test_world_response_nested_support import PATH, X, X2, state_for


def retained_state(location):
    state = state_for((10., 12.), [(1, 9.), (1, 11.), (2, 11.), (2, 13.)])
    family = {'id': 'linear', 'basis': [X.to_dict()]}
    initial = response._propose_fitted_responses(state, [family], PATH)
    candidate = deepcopy(initial['candidates'][0])
    program = Program.from_dict(candidate['program'])
    assert program.predict(Observation({'measurement': 2.}, 0.)) == pytest.approx(14.)
    if location == 'current_candidates':
        state['response_research'] = {'candidates': [candidate]}
    else:
        state['response_research_history'] = [{
            'question_id': 'previous_response', 'hypotheses': deepcopy(initial['hypotheses'])}]
    state['research_table']['references'][0]['frequency'] = {'value': 10.2, 'evidence': 100}
    state['response_settings'].append({'crystal': 1, 'frequency': 10.2,
                                       'activated': True, 'evidence': [100]})
    return state, family, program


def for_program(result, program):
    return [candidate for candidate in result['candidates']
            if 'program' in candidate and Program.from_dict(candidate['program']) == program]


@pytest.mark.parametrize('location', ('current_candidates', 'history_hypotheses'))
def test_replaced_reference_retains_program_and_recomputes_its_response_band(location):
    state, family, program = retained_state(location)
    before = deepcopy(state)
    result = response._propose_fitted_responses(state, [family], PATH)
    assert state == before
    candidates = for_program(result, program)
    assert len(candidates) == 1
    retained = candidates[0]
    assert retained['status'] == 'compatible'
    assert retained['positive_residual_span'] == pytest.approx([0., .2], abs=1e-8)
    assert retained['response_count'] == len(state['response_settings'])
    hypotheses = [item for item in result['hypotheses'] if item['candidate_id'] == retained['id']]
    assert hypotheses
    for hypothesis in hypotheses:
        for row in state['response_settings']:
            residual = row['frequency'] - hypothesis['predictions_by_crystal'][str(row['crystal'])]
            assert (hypothesis['lower'] <= residual <= hypothesis['upper']) == row['activated']


@pytest.mark.parametrize('location', ('current_candidates', 'history_hypotheses'))
def test_old_program_is_rejected_when_new_negative_splits_its_positive_hull(location):
    state, family, program = retained_state(location)
    state['response_settings'].append({'crystal': 2, 'frequency': 12.1,
                                       'activated': False, 'evidence': [101]})
    result = response._propose_fitted_responses(state, [family], PATH)
    retained = for_program(result, program)
    assert len(retained) == 1
    assert retained[0]['status'] == 'rejected'
    assert retained[0]['reason'] == 'negative_response_inside_required_positive_band'
    assert any(row['evidence'] == [101] for row in retained[0]['conflicting_responses'])
    assert not any(Program.from_dict(item['program']) == program for item in result['hypotheses'])


@pytest.mark.parametrize('location', ('current_candidates', 'history_hypotheses'))
def test_history_does_not_reintroduce_a_removed_family(location):
    state, _, program = retained_state(location)
    replacement = {'id': 'quadratic', 'basis': [X2.to_dict()]}
    result = response._propose_fitted_responses(state, [replacement], PATH)
    assert not for_program(result, program)
    assert all(source['family'] != 'linear' for candidate in result['candidates']
               for source in candidate.get('fit_sources', []))


@pytest.mark.parametrize('location', ('current_candidates', 'history_hypotheses'))
def test_reused_family_name_does_not_retain_program_outside_its_changed_basis(location):
    state, _, program = retained_state(location)
    replacement = {'id': 'linear', 'basis': [X2.to_dict()]}
    result = response._propose_fitted_responses(state, [replacement], PATH)
    assert not for_program(result, program)


def open_support_state():
    state = state_for((10.4, 12.4), [(1, 9.4), (1, 11.4), (3, 14.)])
    family = {'id': 'linear', 'basis': [X.to_dict()]}
    result = response._propose_fitted_responses(state, [family], PATH)
    assert not [e for e in result['experiments'] if e['number'] == 3]
    return state, family, result


def assert_paid_consistency(model, state):
    program = Program.from_dict(model['program'])
    rows = state['research_table']['references'] + state['research_table']['targets']
    features = {row['crystal']: {key: value['value']
                for key, value in row['measurements'].items()} for row in rows}
    for row in state['response_settings']:
        residual = row['frequency'] - program.predict(Observation(features[row['crystal']], 0.))
        assert (model['lower'] <= residual <= model['upper']) == row['activated']
    return program, features


def test_complete_fixed_program_support_adds_a_missing_reachable_band_witness():
    state, family, result = open_support_state()
    before = deepcopy(state)
    response._append_fixed_program_responses(result, state, [family], PATH)
    assert state == before
    candidates = {item['id']: item for item in result['candidates']}
    witnesses = [item for item in result['hypotheses']
        if any(source.get('origin') == 'fixed_program_response_support'
               for source in candidates[item['candidate_id']]['fit_sources'])]
    assert witnesses
    # 15 is a synthetic example: 14.4 + .6 lies inside the complete open
    # residual support, although the former midpoint upper bound was .5.
    trial = next(e for e in result['experiments'] if e['number'] == 3 and e['prediction'] == 15.)
    assert any(trial['predictions'][item['id']] for item in witnesses)
    for model in witnesses:
        assert_paid_consistency(model, state)


def test_fixed_program_witnesses_share_the_existing_finite_batch_limit(monkeypatch):
    state, family, members = open_support_state()
    calls = []
    original = response._append_fixed_program_responses

    def record(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(response, '_append_fixed_program_responses', record)
    revision, trial = response._revise_stalled_support(state, [family], members,
        {'family_research': {'no_progress_cycles': 2}}, PATH)
    assert trial is not None and calls == [True]
    fixed = [model for model in revision['models']
             if model['fit_provenance']['origin'] == 'fixed_program_response_support']
    assert fixed
    for model in revision['models']:
        program, features = assert_paid_consistency(model, state)
        residual = model['target_setting'] - program.predict(
            Observation(features[model['design_target_number']], 0.))
        assert model['lower'] <= residual <= model['upper']
    batch = deepcopy(revision['models'])
    state['response_research'] = {'candidates': deepcopy(members['candidates']),
        'conditional_simplification': {'joint_revision': deepcopy(revision),
            'assumption': {'supports': [family]}, 'family_research': {'no_progress_cycles': 3}}}
    for index, point in enumerate(sorted({(model['design_target_number'], model['target_setting'])
                                          for model in batch})):
        state['response_settings'].append({'crystal': point[0], 'frequency': point[1],
                                           'activated': False, 'evidence': [100 + index]})
    for _ in range(2):
        members = response._propose_fitted_responses(state, [family], PATH)
        previous = state['response_research']['conditional_simplification']
        stopped, trial = response._revise_stalled_support(state, [family], members, previous, PATH)
        assert trial is None
        assert stopped['models'] == batch
        assert calls == [True]
        previous['joint_revision'] = deepcopy(stopped)
