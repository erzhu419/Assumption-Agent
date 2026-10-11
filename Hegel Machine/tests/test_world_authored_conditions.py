"""Explicit conditional hypotheses compete using the old paid Boolean researcher."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_response_conditions import (
    predict_response_condition,
    synthesize_response_conditions,
)
from test_world_response_conditions import band_packet, compatible, forecast


def feature(name):
    return Program('feature', name=name)


def constant(value):
    return Program('const', value=value)


def band(expression, lower, upper):
    return Program('and', children=(
        Program('ge', children=(expression, constant(lower))),
        Program('lt', children=(expression, constant(upper)))))


def branched_rule():
    return Program('or', children=(band(feature('residual_0'), -1., 1.),
        Program('and', children=(Program('ge', children=(feature('density'), constant(2.))),
                                band(feature('residual_0'), 4., 5.)))))


def authored(result):
    return [item for item in result['candidates'] if item.get('origin') == 'authored']


def test_explicit_disconnected_guarded_rule_enters_existing_compatible_bank_and_cache_key():
    packet = band_packet()
    before = synthesize_response_conditions(packet)
    packet['candidate_additions'] = [{'id': 'second_branch', 'claim': 'a measured class has a second band',
                                      'program': branched_rule().to_dict()}]

    after = synthesize_response_conditions(packet, before)

    candidate, = authored(after)
    assert not after['cached'] and not after['regenerated']
    assert candidate['authored_id'] == 'second_branch'
    assert candidate['claim'] == 'a measured class has a second band'
    assert candidate['status'] == 'compatible'
    assert candidate['checked_response_count'] == len(packet['responses'])
    assert forecast(candidate, packet, 2, 24.) is True
    assert forecast(candidate, packet, 1, 14.) is False
    assert all(not forecast(item, packet, 2, 24.) for item in compatible(before))
    assert all(item in after['candidates'] for item in before['candidates'])
    again = synthesize_response_conditions(packet, after)
    assert again['cached'] and again['compute_cost'] == 0
    assert again['candidates'] == after['candidates']


def test_paid_counterexample_refutes_wrong_branch_and_removal_keeps_frozen_rejected_rule():
    packet = band_packet()
    packet['candidate_additions'] = [band(feature('residual_0'), -1., 1.).to_dict(),
        {'id': 'branch', 'program': branched_rule().to_dict()}]
    before = synthesize_response_conditions(packet)
    assert len(compatible(before)) == 2
    packet['responses'].append({'crystal': 2, 'frequency': 24., 'activated': False, 'evidence': [99]})
    packet['candidate_additions'] = []

    after = synthesize_response_conditions(packet, before)

    narrow, branch = after['candidates']
    assert narrow['status'] == 'compatible' and branch['status'] == 'refuted'
    assert branch['conflicting_responses'] == [dict(packet['responses'][-1], predicted_activation=True)]
    assert branch['program'] == before['candidates'][1]['program']
    assert branch['bound_rows'] == before['candidates'][1]['bound_rows']
    assert branch['feature_derivations'] == before['candidates'][1]['feature_derivations']
    assert all(item['status'] != 'supported' for item in after['candidates'])


def test_same_authored_name_after_coordinate_revision_appends_without_rebinding_old_rule():
    packet = band_packet()
    packet['candidate_additions'] = [{'id': 'branch', 'program': branched_rule().to_dict()}]
    before = synthesize_response_conditions(packet)
    original = deepcopy(before['candidates'][0])
    for row in packet['bound_rows']:
        row['features']['density'] /= 2.
    packet['paid_family_checks']['linear']['witness']['program'] = Program('mul', children=(
        constant(20.), feature('density'))).to_dict()
    revised = Program('or', children=(band(feature('residual_0'), -1., 1.),
                                     band(feature('residual_0'), 6., 7.)))
    packet['candidate_additions'][0]['program'] = revised.to_dict()

    after = synthesize_response_conditions(packet, before)

    rules = authored(after)
    assert len(rules) == 2 and rules[0] == original
    assert rules[0]['authored_id'] == rules[1]['authored_id'] == 'branch'
    assert rules[0]['id'] != rules[1]['id']
    assert rules[0]['bound_rows'] != rules[1]['bound_rows']
    assert rules[0]['feature_derivations'] != rules[1]['feature_derivations']
    measured = packet['measurements'][1]['measurements']
    assert predict_response_condition(rules[0], measured, rules[0]['bound_rows']['2'], 24.)
    assert not predict_response_condition(rules[1], measured, rules[1]['bound_rows']['2'], 24.)


def test_composed_boundary_uses_measured_object_constants_in_probe_and_target_projection():
    packet = band_packet()
    packet['targets'] = [2]
    coordinate = Program('mul', children=(feature('bound_density'),
        Program('sub', children=(feature('frequency'),
            Program('mul', children=(constant(10.), feature('density')))))))
    packet['candidate_additions'] = [band(coordinate, -1., upper).to_dict() for upper in (1., 2.)]

    result = synthesize_response_conditions(packet)

    assert len(result['candidates']) == 2
    trial = result['selected_experiment']
    assert trial and set(trial['predictions'].values()) == {True, False}
    assert trial['predictions'] == {item['id']: forecast(item, packet, trial['number'], trial['prediction'])
                                    for item in result['candidates']}
    target, = result['target_consequences']['targets']
    assert target['status'] == 'complete'
    projected = {i for left, right in target['support']['intervals'] for i in range(left, right + 1)}
    expected = {i for i in range(61) if any(forecast(item, packet, 2, i * .5)
                                           for item in result['candidates'])}
    assert projected == expected


def test_nonlinear_authored_control_keeps_projection_unresolved():
    packet = band_packet()
    packet['targets'] = [2]
    residual = feature('residual_0')
    packet['candidate_additions'] = [Program('lt', children=(
        Program('mul', children=(residual, residual)), constant(1.))).to_dict()]

    result = synthesize_response_conditions(packet)

    candidate, = result['candidates']
    assert candidate['status'] == 'compatible'
    target, = result['target_consequences']['targets']
    assert target['status'] == 'unresolved'
    assert target['reason'] == 'nonaffine_control_boundary'
    assert 'support' not in target


def test_unprocessed_authored_additions_are_pending_not_cached_after_budget_exhaustion():
    packet = band_packet()
    packet['candidate_additions'] = [band(feature('residual_0'), -1., 1.).to_dict(),
                                     branched_rule().to_dict()]

    partial = synthesize_response_conditions(packet, compute_limit=12)
    assert len(partial['candidates']) == 1
    assert partial['status'] == 'search_budget_exhausted'
    assert partial['pending_candidate_additions'] == packet['candidate_additions'][1:]
    repeated = synthesize_response_conditions(packet, partial, compute_limit=12)
    assert not repeated['cached'] and repeated['status'] == 'search_budget_exhausted'
    finished = synthesize_response_conditions(packet, partial, compute_limit=10000)
    assert not finished['cached'] and len(authored(finished)) == 2
    assert not finished.get('pending_candidate_additions')


@pytest.mark.parametrize('program', [
    feature('frequency'),
    Program('ge', children=(feature('invented_reading'), constant(1.))),
    Program('ge', children=(Program('lag', name='frequency', lag=1), constant(1.))),
    Program('and', children=(feature('frequency'), feature('frequency'))),
])
def test_invalid_authored_conditions_do_not_silently_become_boolean_predictions(program):
    packet = band_packet()
    packet['candidate_additions'] = [program.to_dict()]
    with pytest.raises(ValueError, match='authored response condition'):
        synthesize_response_conditions(packet)
