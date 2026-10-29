"""A new matched experiment consumes related prior reset constraints."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.native_condition_research import (
    _assess_matched_group, _spec, accumulate_matched_pairs,
    matched_assessment_compute_upper, research_native_conditions,
)
from test_native_condition_research import public_spec, pulse_levels_fixture


def scope(records, *, background_drive=.6):
    return {'control': 'switch', 'background': {'drive': background_drive, 'switch': -.8},
            'record_indices': list(range(len(records)))}


def test_prior_paid_levels_exclude_the_new_pair_smooth_explanation_before_generation():
    # All labels are a fixed synthetic control. No apparatus or production
    # prediction is invoked; the old matched-delta/fitting solver runs normally.
    records = pulse_levels_fixture()
    before = deepcopy(records)
    new = [{'reference_record_index': 0, 'record_index': 5}]
    only_new = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        matched_pairs=new)
    accumulated = research_native_conditions(public_spec(), records, SearchBudget(600_000),
        matched_pairs=new, accumulate_matched_scope=scope(records))
    assert only_new['evidence'][0]['status'] == 'smooth'
    evidence, = accumulated['evidence']
    assert evidence['status'] == 'conditional'
    assert evidence['smooth_loss'] > 1e-4
    assert evidence['conditional_loss'] < 1e-10
    assert set(evidence['record_indices']) == {1, 2, 3, 4, 5}
    assert accumulated['accumulated_matched_scope']['matched_rows'] == 80
    old_programs = [candidate['program'] for candidate in only_new['candidates']]
    assert any(candidate['program'] not in old_programs for candidate in accumulated['candidates'])
    assert any(step.get('operation') == 'discover_conditional_coordinates'
        for candidate in accumulated['candidates']
        for step in candidate.get('research_program', {}).get('selection_steps', []))
    assert accumulated['fit_record_indices'] == only_new['fit_record_indices'] == list(range(6))
    assert accumulated['fit_evidence'] == only_new['fit_evidence']
    assert min(candidate['fit_mse'] for candidate in accumulated['candidates']) < 1e-8
    assert records == before


def test_accumulation_keeps_background_and_source_meaning_scopes_separate():
    records = pulse_levels_fixture() + pulse_levels_fixture(-.3)[:2]
    for index, record in enumerate(records):
        record['row_indices'] = list(range(index * 16, (index + 1) * 16))
    new = [{'reference_record_index': 0, 'record_index': 5}]
    pairs = accumulate_matched_pairs(public_spec(), records, new, scope(records))
    assert {pair['record_index'] for pair in pairs} == {1, 2, 3, 4, 5}
    assert all(pair['background'] == {'drive': .6, 'switch': -.8} for pair in pairs)
    restricted = {**scope(records), 'record_indices': [0, 5]}
    assert [pair['record_index'] for pair in accumulate_matched_pairs(
        public_spec(), records, new, restricted)] == [5]
    assert len(pairs) == len({(p['reference_record_index'], p['record_index']) for p in pairs})


@pytest.mark.parametrize('change', ['record_scope', 'background', 'control', 'reset'])
def test_new_pair_must_obey_the_actual_accumulated_scope(change):
    records = pulse_levels_fixture()
    selected, public = scope(records), public_spec()
    if change == 'record_scope':
        selected['record_indices'].remove(5)
    elif change == 'background':
        selected['background']['drive'] = -.3
    elif change == 'control':
        selected['control'] = 'drive'
    else:
        public['reset'] = False
    with pytest.raises(ValueError):
        accumulate_matched_pairs(public, records,
            [{'reference_record_index': 0, 'record_index': 5}], selected)


def test_discovery_uses_the_scoped_reference_not_a_longer_outside_record():
    records = pulse_levels_fixture()
    # The full archive supports a longer reference, but this question meaning
    # cites only the original records. Its discovery must keep their addresses.
    outside = deepcopy(records[0])
    for step in range(16, 20):
        outside['rows'].append({'features': {'drive': .6, 'switch': -.8}, 'target': 0.,
            'history': [{'drive': .6, 'switch': -.8} for _ in range(step)]})
    records.append(outside)
    selected = {**scope(records), 'record_indices': list(range(6))}
    pairs = accumulate_matched_pairs(public_spec(), records,
        [{'reference_record_index': 0, 'record_index': 5}], selected)
    assert {pair['reference_record_index'] for pair in pairs} == {0}


def test_matched_preparation_upper_funds_the_real_varying_response_fitter():
    records = pulse_levels_fixture()
    for record in records:
        for row in record['rows']:
            window = [*row['history'], row['features']][-3:]
            row['target'] = sum((.2 + .1 * point['switch'] + .3 * point['switch'] ** 2)
                if point['switch'] >= .1 else
                (-.4 + .2 * point['switch'] - .1 * point['switch'] ** 2)
                for point in window)
    pairs = accumulate_matched_pairs(public_spec(), records,
        [{'reference_record_index': 0, 'record_index': 5}], scope(records))
    upper = matched_assessment_compute_upper(public_spec(), records, pairs)
    budget = SearchBudget(upper)
    outcome = _assess_matched_group(_spec(public_spec()), records, pairs, budget)
    assert outcome['response_model'] == 'varying_response'
    assert outcome['conditional_loss'] < 1e-10
    assert 0 < outcome['compute_cost'] == budget.spent <= upper
    assert budget.denied_requests == 0


def test_pre_purchase_matched_quote_does_not_read_unpurchased_targets():
    records = pulse_levels_fixture()
    pairs = accumulate_matched_pairs(public_spec(), records,
        [{'reference_record_index': 0, 'record_index': 5}], scope(records))
    paid_upper = matched_assessment_compute_upper(public_spec(), records, pairs)
    for record in records:
        for row in record['rows']:
            del row['target']
    assert matched_assessment_compute_upper(public_spec(), records, pairs) == paid_upper

