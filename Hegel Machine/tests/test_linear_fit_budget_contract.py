"""A short linear allowance must not silently choose a different fitter."""

import pytest

from hegel_machine.cognition import abstractions
from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget


def linear_batch(slots):
    x = Program('feature', name='x')
    basis = Program('const', value=1.)
    terms, paths = [], []
    for index in range(slots):
        if index:
            basis = Program('mul', children=(basis, x))
        terms.append(Program('mul', children=(Program('const', value=0.), basis)))
    pattern = terms[0]
    paths = [(0,)]
    for term in terms[1:]:
        pattern = Program('add', children=(pattern, term))
        paths = [(0, *path) for path in paths] + [(1, 0)]
    template = AbstractTemplate('linear_budget', 'fixture', pattern, (), (), tuple(paths))
    rows = [Observation({'x': value}, sum((index + 1.) / 3. * value ** index
        for index in range(slots))) for value in (i / 48. - 1. for i in range(97))]
    return template, rows


@pytest.mark.parametrize('slots', [4, 5, 6])
@pytest.mark.parametrize('boundary', ['initial_short', 'initial_only', 'basis_short',
                                    'ridge_exact', 'stable_short', 'complete'])
def test_linear_budget_boundaries_keep_method_losses_and_actual_debit(slots, boundary, monkeypatch):
    template, rows = linear_batch(slots)
    count = len(rows)
    limits = {'initial_short': count - 1, 'initial_only': count,
        'basis_short': (slots + 3) * count - 1, 'ridge_exact': (slots + 3) * count,
        'stable_short': (slots + 4) * count - 1, 'complete': (slots + 4) * count}
    budget, report, evaluations = SearchBudget(limits[boundary]), {}, []
    predict = abstractions.predict_rows

    def tracked(program, actual_rows):
        evaluations.append(len(actual_rows))
        return predict(program, actual_rows)

    def forbidden(*args, **kwargs):
        pytest.fail('A linear basis shortfall must not execute coordinate trials')

    monkeypatch.setattr(abstractions, 'predict_rows', tracked)
    monkeypatch.setattr(abstractions, '_comparison_slot_uses', forbidden)
    programs = fit_parameter_slots(template, {}, rows, budget, report=report)
    expected = ([] if boundary == 'initial_short' else ['initial']
        if boundary in {'initial_only', 'basis_short'} else ['initial', 'ridge_lstsq']
        if boundary in {'ridge_exact', 'stable_short'} else ['initial', 'ridge_lstsq', 'stable_lstsq'])
    assert report['candidate_methods'] == expected
    assert len(programs) == len(expected)
    assert sum(evaluations) == budget.spent == report['actual_compute_cost'] <= budget.limit
    assert set(report['losses']) == set(programs)
    assert report['selected_slot_degree'] == 1
    assert not report['scientific_support']
    assert report['search_completed'] is (boundary == 'complete')
    if boundary in {'initial_only', 'basis_short'}:
        assert budget.spent == count
        assert report['linear_diagnostics']['reason'] == 'linear_basis_budget_exhausted'
        assert budget.last_denied_request['required_compute'] == (slots + 2) * count
        assert programs == [template.instantiate({})]
    if boundary != 'complete':
        assert report['status'] == 'UNKNOWN' and report['budget_exhausted']


@pytest.mark.parametrize('slots', [4, 5, 6])
def test_fully_funded_linear_batch_preserves_programs_and_losses(slots):
    template, rows = linear_batch(slots)
    bounded, full = {}, {}
    programs = fit_parameter_slots(template, {}, rows,
        SearchBudget((slots + 4) * len(rows)), report=bounded)
    reference = fit_parameter_slots(template, {}, rows,
        SearchBudget(10000), report=full)
    assert programs == reference
    assert bounded['losses'] == full['losses']
    assert bounded['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert bounded['linear_diagnostics'] == full['linear_diagnostics']
