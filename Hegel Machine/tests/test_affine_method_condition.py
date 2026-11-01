from hegel_machine.cognition import condition_learning
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget


def separating_rows():
    support = [Observation({'x': 1., 'y': 1.}, 0.), Observation({'x': 2., 'y': -1.}, 0.)]
    negative = [Observation({'x': -1., 'y': 1.}, 0.), Observation({'x': -2., 'y': -1.}, 0.)]
    return support, negative


def test_joint_entry_fits_even_when_ordinary_entry_stops_at_one_coordinate(monkeypatch):
    support, negative = separating_rows()
    calls = []
    original = condition_learning._fit_linear

    def recorded(terms, rows):
        calls.append((terms, rows))
        return original(terms, rows)

    monkeypatch.setattr(condition_learning, '_fit_linear', recorded)
    ordinary = condition_learning.learn_condition(support, negative, SearchBudget(100), max_history=0)
    assert ordinary['status'] == 'separated' and ordinary['compute_cost'] == 4
    assert calls == []

    budget = SearchBudget(8)
    joint = condition_learning.learn_affine_condition(support, negative, budget)
    assert len(calls) == 1 and len(calls[0][0]) == 2
    assert [row.target for row in calls[0][1]] == [1., 1., 0., 0.]
    assert joint['status'] == 'separated'
    assert joint['affine_fit_executed'] and joint['execution_complete']
    assert joint['compute_cost'] == joint['required_compute'] == budget.spent == 8
    condition = Program.from_dict(joint['condition'])
    assert all(condition.predict(row) for row in support)
    assert not any(condition.predict(row) for row in negative)


def test_joint_entry_requires_the_complete_fit_and_test_cost(monkeypatch):
    support, negative = separating_rows()

    def not_funded(*_):
        raise AssertionError('A partial joint fit must not run')

    monkeypatch.setattr(condition_learning, '_fit_linear', not_funded)
    budget = SearchBudget(7)
    result = condition_learning.learn_affine_condition(support, negative, budget)
    assert result['status'] == 'insufficient_budget' and result['budget_incomplete']
    assert not result['affine_fit_executed'] and not result['execution_complete']
    assert result['condition'] is None and result['compute_cost'] == budget.spent == 0
    assert result['required_compute'] == 8


def test_completed_inseparable_joint_fit_is_not_reported_as_unfunded():
    support = [Observation({'x': 0., 'y': 0.}, 0.), Observation({'x': 1., 'y': 1.}, 0.)]
    negative = [Observation({'x': 0., 'y': 0.}, 0.)]
    budget = SearchBudget(6)
    result = condition_learning.learn_affine_condition(support, negative, budget)
    assert result['status'] == 'not_separable' and result['condition'] is None
    assert result['affine_fit_executed'] and result['execution_complete']
    assert not result['budget_incomplete']
    assert result['compute_cost'] == budget.spent == 6


def test_joint_entry_does_not_relabel_a_single_coordinate_as_a_joint_fit(monkeypatch):
    support = [Observation({'x': 1.}, 0.), Observation({'x': 2.}, 0.)]
    negative = [Observation({'x': -1.}, 0.)]

    def not_joint(*_):
        raise AssertionError('A single observable is already covered by coordinate research')

    monkeypatch.setattr(condition_learning, '_fit_linear', not_joint)
    result = condition_learning.learn_affine_condition(support, negative, SearchBudget(100))
    assert result['status'] == 'insufficient_joint_observables'
    assert result['compute_cost'] == 0 and not result['affine_fit_executed']
