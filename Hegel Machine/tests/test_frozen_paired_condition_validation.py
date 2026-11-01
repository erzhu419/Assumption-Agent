"""A validation return judges the actual earlier condition and its prediction."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import method_conditions
from hegel_machine.cognition.models import SearchBudget
from test_v8_paired_method_conditions import METHOD, learn, pair


BASELINE = {'id': 'original_method', 'version': 1}


def frozen_study():
    development = [pair(0), pair(1), pair(2, stage='complete', challenger_cost=200)]
    frozen = learn(development)
    frozen['baseline_method_ref'] = deepcopy(BASELINE)
    frozen['candidates'] = frozen['candidates'][:1]
    validation = [pair(3, role='condition_validation'),
        pair(4, stage='complete', challenger_cost=200, role='condition_validation')]
    cohort = [item['physical_source_ref'] for item in validation]
    for item in validation:
        item['baseline']['method_ref'] = deepcopy(BASELINE)
        item['challenger']['method_ref'] = deepcopy(METHOD)
        item['frozen_conditions_ref'] = {'file': 'frozen_conditions.json'}
        item['condition_prediction_ref'] = {'file': 'predictions.json',
            'json_path': deepcopy(item['feedback_ref']['json_path'])}
        item['frozen_conditions'] = [deepcopy(frozen)]
        item['condition_validation_cohort_refs'] = deepcopy(cohort)
        item['condition_prediction'] = {
            'frozen_conditions_ref': deepcopy(item['frozen_conditions_ref']),
            'profile': deepcopy(item['baseline']['profile']),
            'before_method_execution': True,
            'predictions': [{'candidate_predictions': method_conditions._conflict_predictions(
                frozen, item['baseline']['profile']), 'public_coverage': {}}]}
    return development, validation, frozen


def prohibit_late_learning(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError('A frozen validation cannot resynthesize or replace its earlier prediction')
    monkeypatch.setattr(method_conditions, 'research_method_conflict', prohibited)
    monkeypatch.setattr(method_conditions, '_conflict_predictions', prohibited)


def evaluate(items):
    budget = SearchBudget(0)
    result = method_conditions.research_paired_method_conditions(METHOD, items, budget)
    assert budget.spent == 0
    return result


def test_frozen_condition_waits_for_whole_cohort_then_uses_saved_predictions(monkeypatch):
    development, validation, frozen = frozen_study()
    original = deepcopy(validation)
    prohibit_late_learning(monkeypatch)
    partial = evaluate([*development, validation[0]])
    assert partial['status'] == 'awaiting_registered_condition_validation'
    assert not partial['general_guard_supported']
    assert partial['candidates'] == frozen['candidates']
    result = evaluate([*development, *validation])
    assert result['status'] == 'paired_condition_supported_on_declared_groups'
    assert result['general_guard_supported']
    assert result['condition_validation_mode'] == 'frozen_before_method_call'
    assert result['coordinate_paths'] == frozen['coordinate_paths']
    assert result['candidates'] == frozen['candidates']
    assert result['frozen_condition_prediction_refs'] == [item['condition_prediction_ref']
        for item in validation]
    assert result['compute_cost'] == 0
    assert validation == original


def test_negative_validation_cannot_relearn_a_more_favorable_condition(monkeypatch):
    development, validation, frozen = frozen_study()
    validation[1]['challenger']['costs']['compute_cost'] = 50
    prohibit_late_learning(monkeypatch)
    result = evaluate([*development, *validation])
    assert result['status'] == 'development_condition_without_group_support'
    assert not result['general_guard_supported'] and result['condition'] is None
    assert result['candidates'] == frozen['candidates']


def test_actual_prior_prediction_is_not_replaced_by_a_later_prediction(monkeypatch):
    development, validation, _ = frozen_study()
    saved = validation[0]['condition_prediction']['predictions'][0]['candidate_predictions'][0]
    saved['helpful'] = not saved['helpful']
    prohibit_late_learning(monkeypatch)
    result = evaluate([*development, *validation])
    assert result['status'] == 'development_condition_without_group_support'
    assert result['frozen_validation_predictions'][0]['helpful'] == [saved['helpful']]
    assert not result['general_guard_supported']


@pytest.mark.parametrize('change,reason', [
    ('condition', 'validation_sources_used_different_frozen_conditions'),
    ('coordinates', 'validation_sources_used_different_frozen_conditions'),
    ('method', 'frozen_condition_method_pair_differs'),
    ('profile', 'frozen_condition_prediction_profile_differs'),
    ('late', 'condition_prediction_not_bound_before_method_call'),
    ('reference', 'condition_prediction_not_bound_before_method_call'),
    ('missing_prediction', 'frozen_condition_predictions_unavailable'),
    ('unbound_source', 'validation_condition_not_bound_to_frozen_prediction'),
])
def test_changed_or_unbound_validation_stays_unresolved(monkeypatch, change, reason):
    development, validation, _ = frozen_study()
    item = validation[1]
    if change == 'condition':
        item['frozen_conditions'][0]['candidates'][0]['condition'] = {'op': 'const', 'value': 1.}
    elif change == 'coordinates':
        item['frozen_conditions'][0]['coordinate_paths'][0]['path'] = ['residual', 'loss']
    elif change == 'method':
        item['baseline']['method_ref']['version'] += 1
    elif change == 'profile':
        item['baseline']['profile']['observation_count'] += 1
        item['challenger']['profile'] = deepcopy(item['baseline']['profile'])
    elif change == 'late':
        item['condition_prediction']['before_method_execution'] = False
    elif change == 'reference':
        item['condition_prediction']['frozen_conditions_ref'] = {'file': 'later_conditions.json'}
    elif change == 'missing_prediction':
        item['condition_prediction']['predictions'][0]['candidate_predictions'] = []
    elif change == 'unbound_source':
        item.pop('condition_prediction_ref')
    prohibit_late_learning(monkeypatch)
    result = evaluate([*development, *validation])
    assert result['status'] == 'frozen_condition_validation_unresolved'
    assert result['reason'] == reason
    assert not result['general_guard_supported'] and result['condition'] is None
