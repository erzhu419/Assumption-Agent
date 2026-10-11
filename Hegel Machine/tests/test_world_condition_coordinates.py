"""Revised coordinates extend research without reinterpreting retained rules."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.world_response_conditions import (
    _experiments,
    _inputs,
    _screen,
    _target_consequences,
    predict_response_condition,
    synthesize_response_conditions,
)
from test_world_response_conditions import band_packet, compatible


def _coordinate_revision(packet, kind):
    revised = deepcopy(packet)
    density = Program('feature', name='density')
    if kind == 'bound_coordinate':
        for row in revised['bound_rows']:
            row['features']['density'] /= 2.
        program = Program('mul', children=(Program('const', value=20.), density))
        revised['paid_family_checks']['linear']['witness']['program'] = program.to_dict()
    elif kind == 'residual_source':
        centre = Program.from_dict(revised['paid_family_checks']['linear']['witness']['program'])
        program = Program('add', children=(centre, Program('const', value=.5)))
        revised['paid_family_checks']['linear']['witness']['program'] = program.to_dict()
    else:
        for row in revised['measurements']:
            row['measurements']['new_signal'] = {
                'value': 10. * row['crystal'], 'evidence': [100 + row['crystal']]}
    return revised


@pytest.mark.parametrize('kind', ['bound_coordinate', 'residual_source', 'measured_column'])
def test_real_representation_change_extends_compatible_pool_once_then_reuses_it(kind):
    packet = band_packet()
    before = synthesize_response_conditions(packet)
    assert compatible(before)
    untouched = deepcopy(before)
    revised = _coordinate_revision(packet, kind)

    after = synthesize_response_conditions(revised, previous=before)

    assert after['regenerated'] and after['generation_compute_cost'] > 0
    old = {row['id']: row for row in before['candidates']}
    retained = {row['id']: row for row in after['candidates']}
    assert old.keys() <= retained.keys()
    assert retained.keys() - old.keys(), 'new coordinates must enter the retained candidate pool'
    for identifier, candidate in old.items():
        current = retained[identifier]
        for key in ('program', 'feature_derivations', 'proposed_evidence', 'status', 'evidence'):
            assert current[key] == candidate[key]
    assert before == untouched
    resumed = json.loads(json.dumps(after))

    cached = synthesize_response_conditions(deepcopy(revised), previous=resumed)

    assert cached['cached'] and cached['compute_cost'] == 0
    assert cached['candidates'] == after['candidates']


def test_saved_candidates_without_coordinates_keep_the_previous_input_binding():
    packet = band_packet()
    before = synthesize_response_conditions(packet)
    for candidate in before['candidates']:
        candidate.pop('bound_rows', None)
    saved = json.loads(json.dumps(before))
    revised = _coordinate_revision(packet, 'bound_coordinate')

    after = synthesize_response_conditions(revised, previous=saved)

    by_id = {row['id']: row for row in after['candidates']}
    original_binding = before['inputs']['bound_rows']
    for candidate in before['candidates']:
        retained = by_id[candidate['id']]
        assert retained['bound_rows'] == original_binding
        assert retained['status'] == candidate['status']
        assert retained['program'] == candidate['program']
        assert retained['feature_derivations'] == candidate['feature_derivations']
    assert all('bound_rows' not in row for row in saved['candidates'])


def test_screen_probe_and_target_projection_share_each_rules_frozen_coordinate():
    packet = band_packet()
    packet['targets'] = [1]
    inputs = _inputs(packet)
    frozen = deepcopy(inputs['bound_rows'])
    coordinate = Program('feature', name='residual_0')
    source = packet['paid_family_checks']['linear']['witness']['program']
    candidates = []
    for identifier, upper in [('narrow', 1.), ('broad', 1.5)]:
        program = Program('and', children=(
            Program('ge', children=(coordinate, Program('const', value=-1.))),
            Program('lt', children=(coordinate, Program('const', value=upper)))))
        candidates.append({'id': identifier, 'status': 'compatible',
            'program': program.to_dict(), 'feature_derivations': {'residual_0': deepcopy(source)},
            'bound_rows': deepcopy(frozen)})
    for row in inputs['bound_rows'].values():
        row['density'] = 999.
    prior_candidates = deepcopy(candidates)

    screened = [_screen(candidate, inputs, SearchBudget(10000)) for candidate in candidates]

    assert all(candidate['status'] == 'compatible' for candidate in screened)
    result = {'candidates': screened, 'selected_experiment': None, 'experiment_action': None}
    _experiments(result, inputs, SearchBudget(10000))
    trial = result['selected_experiment']
    assert trial is not None
    number = str(trial['number'])
    assert trial['predictions'] == {candidate['id']: predict_response_condition(
        candidate, inputs['measurements'][number], frozen[number], trial['prediction'])
        for candidate in screened}
    assert set(trial['predictions'].values()) == {False, True}

    _target_consequences(result, inputs, SearchBudget(10000))

    target, = result['target_consequences']['targets']
    assert target['status'] == 'complete'
    assert target['support'] == {'intervals': [[18, 22]], 'grid_count': 5}
    assert candidates == prior_candidates
