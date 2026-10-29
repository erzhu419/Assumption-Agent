"""An output bound must not discard later legal observable-condition searches."""

from hegel_machine.cognition.method_conditions import research_method_conflict
from hegel_machine.cognition.models import SearchBudget
from test_method_conflict_conditions import outcome
from test_method_use_conditions import METHOD


def returns_with_boundary_columns(columns):
    results = [outcome(0, 2, True), outcome(1, 2, True), outcome(2, 2, False)]
    for index, result in enumerate(results):
        result['profile']['boundary'] = {name: values[index] for name, values in columns.items()}
    return results


def test_late_coordinate_after_eight_unseparable_columns_can_supply_a_condition():
    columns = {f'a{index:02}': [0., 2., 1.] for index in range(8)}
    columns['z09'] = [2., 2., 0.]
    budget = SearchBudget(100000)
    result = research_method_conflict(METHOD, returns_with_boundary_columns(columns), budget,
                                     max_conditions=1)
    assert result['status'] == 'development_conditions_found'
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['coordinates'] == [{'path': ['boundary', 'z09'], 'kind': 'number'}]
    assert len(result['diagnostics']) == 10
    assert result['compute_cost'] == budget.spent <= result['required_compute']


def test_affine_pass_remains_reachable_when_each_single_coordinate_fails():
    results = returns_with_boundary_columns({'a': [0., 2., 0.], 'b': [2., 0., 0.]})
    result = research_method_conflict(METHOD, results, SearchBudget(100000), max_conditions=1)
    assert result['condition']
    assert len(result['candidates']) == 1
    assert len(result['candidates'][0]['coordinates']) == 2
    assert [item['status'] for item in result['diagnostics'][1:]] == [
        'not_separable', 'not_separable', 'separated']


def test_output_still_has_at_most_eight_actual_conditions_and_insufficient_budget_is_unresolved():
    results = returns_with_boundary_columns({f'v{index:02}': [2., 2., 0.] for index in range(12)})
    denied_budget = SearchBudget(1)
    denied = research_method_conflict(METHOD, results, denied_budget, max_conditions=8)
    assert denied['budget_incomplete'] and not denied['candidates']
    assert denied_budget.spent == 0
    budget = SearchBudget(denied['required_compute'])
    result = research_method_conflict(METHOD, results, budget, max_conditions=8)
    assert len(result['candidates']) == 8
    assert result['compute_cost'] == budget.spent <= result['required_compute']
