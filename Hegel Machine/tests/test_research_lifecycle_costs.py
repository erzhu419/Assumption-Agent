"""Source reuse, failed trials and incomplete bills change net comparisons."""

from copy import deepcopy
import pytest

from hegel_machine.cognition.research_lifecycle_costs import COSTS, summarize_research_lifecycle


def occurrence(name, cost, arms=('A', 'B', 'C'), phase='source', **extra):
    return {'occurrence_ref': {'file': name}, 'scope': 'incremental_execution',
        'charged_to': list(arms), 'phase': phase,
        'costs': {key: cost if key == 'compute_cost' else 0 for key in COSTS}, **extra}


def test_sources_and_included_failures_charged_once_not_per_target():
    shared = occurrence('original-source.json', 100)
    failure = occurrence('failed.json', 20, ('C',), failed=True)
    target = occurrence('target.json', 30, ('C',), 'successor')
    child = occurrence('nested-fit.json', 12, ('C',), 'successor', included_in_ref=target['occurrence_ref'])
    result = summarize_research_lifecycle([shared, shared, failure, target, child], arms=('A', 'B', 'C'),
        outcomes=[{'arm': 'C', 'predictor_available': False, 'prediction_pass': False}])
    assert result['cost_intervals']['C']['compute_cost'] == [150, 150]
    assert result['actual_experiment_cost_intervals']['compute_cost'] == [150, 150]
    assert result['paired_differences']['C_minus_B']['cost_intervals']['compute_cost'] == [50, 50]
    assert result['duplicate_references_ignored'] == [shared['occurrence_ref']]
    assert result['failed_occurrence_refs'] == [failure['occurrence_ref']]
    assert result['outcomes']['C'] == {'registered': 1, 'delivered': 0, 'confirmed': 0, 'unmeasured': 1}


def test_shared_unknown_bill_cancels_only_in_difference_and_unique_unknown_does_not():
    shared = occurrence('old-unknown.json', None)
    c = occurrence('C.json', None, ('C',), 'successor', cost_intervals={'compute_cost': [5, 20]})
    b = occurrence('B.json', 10, ('B',), 'successor')
    result = summarize_research_lifecycle([shared, b, c], arms=('B', 'C'))
    assert result['cost_intervals']['C']['compute_cost'] == [5, None]
    assert result['paired_differences']['C_minus_B']['cost_intervals']['compute_cost'] == [-5, 10]
    unknown = occurrence('C-source-unknown.json', None, ('C',))
    result = summarize_research_lifecycle([shared, b, c, unknown], arms=('B', 'C'))
    assert result['paired_differences']['C_minus_B']['cost_intervals']['compute_cost'] == [-5, None]
    assert result['net_utility'] is None


def test_inconsistent_original_bill_and_cumulative_receipt_require_reconciliation():
    source = occurrence('source.json', 3)
    changed = deepcopy(source)
    changed['costs']['compute_cost'] = 4
    with pytest.raises(ValueError, match='inconsistent'):
        summarize_research_lifecycle([source, changed], arms=('A', 'B', 'C'))
    source['scope'] = 'cumulative_lineage'
    with pytest.raises(ValueError, match='incremental'):
        summarize_research_lifecycle([source], arms=('A', 'B', 'C'))


def test_registered_new_sources_are_four_factors_and_quote_covers_shared_evaluation():
    from run_partial_role_successors import program_successor_protocol
    source = {'seed': 7, 'objects': 4, 'apparatus_seed': 7, 'name_seed': 2, 'wiring_seed': 3,
        'relation_radius': .8, 'actuator_curvature': .12, 'relation_direction': 'symmetric'}
    plan = program_successor_protocol(source)
    assert [case['factor'] for case in plan['factor_cases']] == ['000', '100', '010', '001']
    assert len({task['seed'] for task in plan['tasks']}) == 4
    assert plan['tasks'][0]['relation_radius'] == .8 and plan['tasks'][1]['relation_radius'] == 1.
    assert plan['tasks'][2]['relation_direction'] == 'right'
    assert plan['tasks'][3]['name_seed'] != source['name_seed']
    quote = plan['budget_quote']
    assert quote['total_new_compute_upper'] == 5762136
    assert quote['total_new_observation_upper'] == 7568
    assert quote['total_new_measurement_upper'] == 84408
    assert not plan['current_budget_admitted']
