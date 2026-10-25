"""A complete finite fit is quoted without running or changing its search."""

from copy import deepcopy

import pytest

from hegel_machine.cognition import abstractions
from hegel_machine.cognition.abstractions import (
    AbstractTemplate, fit_parameter_slots, quote_parameter_slot_fit,
)
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from test_conditional_linear_fitting import const, fixture, op


def fit(template, rows, limit):
    report, budget = {}, SearchBudget(limit)
    programs = fit_parameter_slots(template, {}, rows, budget, report=report)
    return programs, report, budget.spent


def test_quote_does_not_predict_fit_or_mutate_the_template_and_rows(monkeypatch):
    template, rows = fixture()
    saved = deepcopy((template, rows))

    def forbidden(*args, **kwargs):
        pytest.fail('a fit quote must not execute numerical research')

    monkeypatch.setattr(Program, 'predict', forbidden)
    monkeypatch.setattr(abstractions, 'predict_rows', forbidden)
    monkeypatch.setattr(abstractions, '_fit_slots', forbidden)
    quote = quote_parameter_slot_fit(template, rows)
    assert quote['scope'] == 'cold_complete_finite_fit_only'
    assert quote['compute'] > 0 and quote['maximum_outputs'] > 3
    assert (template, rows) == saved


@pytest.mark.parametrize('options', [{}, {'duplicated_offset': True}, {'nonlinear_ordinary': True}])
def test_quoted_finite_call_completes_and_preserves_unquoted_outputs(options):
    template, rows = fixture(**options)
    original, original_report, original_cost = fit(template, rows, 100_000)
    quote = quote_parameter_slot_fit(template, rows)
    actual, report, cost = fit(template, rows, quote['compute'])
    assert actual == original
    assert report['candidate_methods'] == original_report['candidate_methods']
    assert report['losses'] == original_report['losses']
    assert report['search_completed']
    assert cost == original_cost <= quote['compute']
    assert len(actual) <= quote['maximum_outputs']
    assert all(program.size == quote['expanded_program_nodes'] for program in actual)
    assert len(report.get('conditional_linear_profiles', [])) <= quote['conditional_profile_count_bound']
    assert not report['scientific_support'] and not report['global_optimum_established']


def test_linear_quote_includes_both_solvers_and_no_profiles():
    x = Program('feature', name='x')
    template = AbstractTemplate('linear', 'fixture', op('add', op('mul', const(0), x), const(0)),
                                (), (), ((0, 0), (1,)))
    rows = [Observation({'x': value}, .37 * value + .23) for value in (-2., -1., 0., 1., 2.)]
    quote = quote_parameter_slot_fit(template, rows)
    programs, report, cost = fit(template, rows, quote['compute'])
    assert quote['compute'] == 6 * len(rows) == cost
    assert quote['maximum_outputs'] == len(programs) == 3
    assert quote['conditional_profile_count_bound'] == 0
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert report['search_completed']


def test_quote_counts_each_comparison_operand_and_its_actual_history_domain():
    gain, theta, offset = (Program('arg', name=name) for name in ('gain', 'theta', 'offset'))
    lag = Program('lag', name='x', lag=1)
    body = Program('if', children=(Program('available', lag=1),
        op('add', op('mul', gain, op('ge', lag, theta)), offset), offset))
    definition = Definition('guarded', ('gain', 'theta', 'offset'), body)
    call = Program('call', name='guarded', definition=definition,
                   children=(const(0), const(0), const(0)))
    template = AbstractTemplate('guarded', 'fixture', call, (), (), ((0,), (1,), (2,)))
    rows = [Observation({'x': float(index)}, float(index) / 3,
                        () if index < 2 else ({'x': float(index - 1)},)) for index in range(5)]
    quote = quote_parameter_slot_fit(template, rows)
    # Three mature operand rows yield at most two midpoints; the startup rows
    # remain in every whole-Program fit and comparison.
    assert quote['eligible_rows'] == 5
    assert quote['coordinate_compute'] == 5 + 2 * ((3 + 3 + 3 + 2) * 5 + 3)
    assert quote['conditional_profile_count_bound'] == 1 + 2 * (3 + 2)
    programs, report, cost = fit(template, rows, quote['compute'])
    assert report['search_completed'] and cost <= quote['compute']
    assert len(programs) <= quote['maximum_outputs']


def test_repeated_lexical_threshold_quotes_both_observed_operands():
    gain, theta, offset = (Program('arg', name=name) for name in ('gain', 'theta', 'offset'))
    indicators = op('add', op('ge', Program('feature', name='x'), theta),
                    op('ge', Program('feature', name='y'), theta))
    definition = Definition('repeated', ('gain', 'theta', 'offset'),
                            op('add', op('mul', gain, indicators), offset))
    call = Program('call', name='repeated', definition=definition,
                   children=(const(0), const(0), const(0)))
    template = AbstractTemplate('repeated', 'fixture', call, (), (), ((0,), (1,), (2,)))
    rows = [Observation({'x': float(index), 'y': float(index * 2)}, float(index) / 3)
            for index in range(5)]
    quote = quote_parameter_slot_fit(template, rows)
    assert quote['coordinate_compute'] == 5 + 2 * ((3 + 3 + 3 + 4 + 4) * 5 + 5 + 5)
    programs, report, cost = fit(template, rows, quote['compute'])
    assert report['search_completed'] and cost <= quote['compute']
    assert len(programs) <= quote['maximum_outputs']


def test_empty_domain_and_fixed_program_do_not_buy_nonexistent_slot_work():
    template = AbstractTemplate('fixed', 'fixture', Program('feature', name='original'), (), (), ())
    rows = [Observation({'mapped': 2.}, 3.)]
    empty = quote_parameter_slot_fit(template, rows)
    assert empty['compute'] == empty['maximum_outputs'] == empty['eligible_rows'] == 0
    mapped = quote_parameter_slot_fit(template, rows, {'original': 'mapped'})
    assert mapped['compute'] == mapped['maximum_outputs'] == mapped['eligible_rows'] == 1


def test_quote_rejects_the_same_invalid_selected_slot_as_the_fitter():
    template = AbstractTemplate('invalid', 'fixture', Program('feature', name='x'), (), (), ((),))
    with pytest.raises(ValueError, match='actual const slots'):
        quote_parameter_slot_fit(template, [Observation({'x': 2.}, 3.)])
