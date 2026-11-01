"""Paid condition search must reach distinct evidence, not fill with aliases."""

from copy import deepcopy

from hegel_machine.cognition import condition_learning
from hegel_machine.cognition.method_conditions import (
    _conflict_predictions, _method_coordinates, _method_observables,
    research_method_conflict,
)
from hegel_machine.cognition.models import SearchBudget
from test_method_conflict_conditions import outcome
from test_method_use_conditions import METHOD, profile


def bound_profile(ready):
    """The saved transfer binding summary's actual ready/unknown contract."""
    result = profile()
    groups = [
        {'parameter_index': 0, 'term_indices': [0, 1],
         'occurrence_depths': [0, 2], 'source_interval_count': 0},
        {'parameter_index': 1, 'term_indices': [2],
         'occurrence_depths': [1], 'source_interval_count': 1},
    ] if ready else []
    result['research'] = {'parameter_groups': {
        'status': 'ready' if ready else 'binding_unavailable',
        'count': len(groups) if ready else None,
        'fit_columns': 8 if ready else None,
        'bound_occurrences': 3 if ready else None,
        'groups': groups,
    }}
    # This independent scientific input merely co-occurs in these examples.
    result['boundary'] = {'observed_interval': 2. if ready else None}
    return result


def test_binding_availability_aliases_require_the_producer_contract():
    public = [bound_profile(True), bound_profile(False)]
    old = deepcopy(public)
    coordinates = _method_coordinates([_method_observables(p) for p in public])
    parameter = ['research', 'parameter_groups']
    canonical = next(item for item in coordinates
        if item['path'] == parameter + ['count'] and item['kind'] == 'available')
    aliases = [item['path'] for item in canonical['aliases']]
    assert parameter + ['fit_columns'] in aliases
    assert parameter + ['bound_occurrences'] in aliases
    assert parameter + ['groups', 0, 'parameter_index'] in aliases
    assert parameter + ['groups', 0, 'term_indices', 0] in aliases
    paths = [item['path'] for item in coordinates]
    assert parameter + ['fit_columns'] not in paths
    assert parameter + ['groups', 0, 'term_indices', 1] in paths
    assert any(path[:4] == parameter + ['groups', 1] for path in paths)
    assert ['boundary', 'observed_interval'] in paths
    assert public == old


def test_old_frozen_available_coordinates_keep_their_original_predictions():
    # New normalization must not rewrite a previous frozen public_N mapping.
    frozen = {'coordinate_paths': [{'path': [
        'research', 'parameter_groups', 'fit_columns'], 'kind': 'available'}],
        'candidates': [{'condition': {'op': 'feature', 'name': 'public_0',
                                    'children': []}}]}
    original = deepcopy(frozen)
    assert _conflict_predictions(frozen, bound_profile(True)) == [
        {'candidate': 0, 'helpful': True}]
    assert _conflict_predictions(frozen, bound_profile(False)) == [
        {'candidate': 0, 'helpful': False}]
    assert frozen == original


def diverse_returns():
    records = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 2, False)]
    for index, record in enumerate(records):
        record['profile']['boundary'] = {
            f'candidate_{column:02}': [2., 3., 0.][index] for column in range(8)}
        record['profile']['residual']['transition'] = {'error_rows': [4, 6, 1][index]}
    return records


def test_output_cap_does_not_skip_late_residual_or_real_affine_research(monkeypatch):
    records = diverse_returns()
    original = deepcopy(records)
    fitted = []
    fit = condition_learning._fit_linear

    def observed_fit(terms, rows, *args, **kwargs):
        fitted.append((len(terms), len(rows)))
        return fit(terms, rows, *args, **kwargs)

    monkeypatch.setattr(condition_learning, '_fit_linear', observed_fit)
    budget = SearchBudget(100000)
    result = research_method_conflict(METHOD, records, budget, max_conditions=3)
    assert result['search_complete']
    assert len(result['candidates']) == 3
    assert len(result['candidate_pool']) > len(result['candidates'])
    late = [item for item in result['diagnostics']
        if item.get('synthesis') == 'coordinate' and any(
            coordinate['path'] == ['residual', 'transition', 'error_rows']
            for coordinate in item.get('coordinates', []))]
    assert len(late) == 1 and late[0]['status'] == 'separated'
    assert any(item.get('synthesis') == 'affine' for item in result['diagnostics'])
    assert fitted and fitted[-1] == (9, 3)
    # All early boundary columns fit, but cannot monopolize the output quota.
    assert any(len(item['coordinates']) == 1 and item['coordinates'][0]['path'] == [
        'residual', 'transition', 'error_rows'] for item in result['candidates'])
    assert len([item for item in result['candidate_pool']
        if len(item['coordinates']) == 1 and item['coordinates'][0]['path'][0] == 'boundary']) == 8
    assert result['compute_cost'] == budget.spent <= result['required_compute']
    assert records == original


def test_old_early_exit_cache_cannot_hide_the_completed_search_contract():
    records = diverse_returns()
    finished = research_method_conflict(METHOD, records, SearchBudget(100000), max_conditions=3)
    old = deepcopy(finished)
    old['basis']['condition_search'] = 'complete_public_coordinate_and_affine_pass'
    old.pop('search_complete', None)
    retained_old = deepcopy(old)
    unavailable = SearchBudget(0)
    retry = research_method_conflict(METHOD, records, unavailable,
                                    previous=old, max_conditions=3)
    assert retry['budget_incomplete'] and not retry.get('reused')
    assert retry['compute_cost'] == unavailable.spent == 0
    assert old == retained_old
    completed_budget = SearchBudget(retry['required_compute'])
    updated = research_method_conflict(METHOD, records, completed_budget,
                                      previous=old, max_conditions=3)
    assert updated['search_complete'] and not updated.get('reused')
    reused_budget = SearchBudget(0)
    reused = research_method_conflict(METHOD, records, reused_budget,
                                     previous=updated, max_conditions=3)
    assert reused['reused'] and reused['compute_cost'] == reused_budget.spent == 0
    assert reused['candidate_pool'] == updated['candidate_pool']
