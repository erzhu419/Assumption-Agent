"""Conditional slots reuse the fixed-structure linear solver after finite trials."""
from collections import Counter
from dataclasses import replace

import pytest

from hegel_machine.cognition import abstractions
from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget


def const(value):
    return Program('const', value=float(value))


def op(name, *children):
    return Program(name, children=children)


def fixture(*, duplicated_offset=False, nonlinear_ordinary=False):
    x = Program('feature', name='x')
    parent = op('mul', const(5.), x)
    before, coefficient, theta, offset = [Program('arg', name=name)
                                          for name in ('before', 'coefficient', 'theta', 'offset')]
    basis = Program('if', children=(op('ge', x, theta), x, const(0.)))
    coefficient_term = op('mul', coefficient, coefficient) if nonlinear_ordinary else coefficient
    body = op('add', before, op('add', op('mul', coefficient_term, basis), offset))
    if duplicated_offset:
        body = op('add', body, Program('arg', name='second_offset'))
    names = ('before', 'coefficient', 'theta', 'offset') + (('second_offset',) if duplicated_offset else ())
    definition = Definition('conditional_fixture', names, body)
    pattern = Program('call', name=definition.definition_id, definition=definition,
                      children=(parent, const(0.), const(0.), const(0.))
                               + ((const(0.),) if duplicated_offset else ()))
    paths = ((1,), (2,), (3,)) + (((4,),) if duplicated_offset else ())
    template = AbstractTemplate('conditional_fit', 'fixture', pattern, (), (), paths)
    rows = [Observation({'x': value}, 5. * value + .37 * max(value, 0.) + .23)
            for value in (-2., -1., 0., 1., 2.)]
    return template, rows


def run(template, rows, limit=10000):
    report, budget = {}, SearchBudget(limit)
    programs = fit_parameter_slots(template, {}, rows, budget, report=report)
    return programs, report, budget


def test_profiles_add_continuous_candidate_preserving_old_finite_outputs_and_conditions():
    template, rows = fixture()
    programs, report, budget = run(template, rows)
    profiles = report['conditional_linear_profiles']
    finite_cost = budget.spent - sum(profile['actual_compute_cost'] for profile in profiles)
    old, old_report, bounded = run(template, rows, finite_cost)
    assert old and programs[:len(old)] == old
    assert report['candidate_methods'][:len(old)] == old_report['candidate_methods']
    assert all(report['losses'][program] == old_report['losses'][program] for program in old)
    assert bounded.spent == finite_cost and not old_report['search_completed']
    assert all('conditional_' not in method for method in old_report['candidate_methods'])
    assert min(report['losses'][program] for program in programs) < 1e-20
    assert min(old_report['losses'][program] for program in old) > 1e-3
    assert 'conditional_stable_lstsq' in report['candidate_methods']
    visited = {program.children[2].value for program in old_report['losses']}
    assert {tuple(profile['threshold_values']) for profile in profiles} == {(value,) for value in visited}
    assert all(program.children[2].value in visited for program in programs)
    assert all(program.definition == template.pattern.definition
               and program.children[0] == template.pattern.children[0] for program in programs)
    assert report['status'] == 'UNKNOWN' and report['search_completed']
    assert not report['scientific_support'] and not report['global_optimum_established']
    assert report['reason'] == 'nonlinear_parameter_search_is_local'


def test_paid_whole_program_predictions_are_reused_between_profiles(monkeypatch):
    template, rows = fixture()
    calls = []
    predict = abstractions.predict_rows

    def tracked(program, actual_rows):
        calls.append((program, len(actual_rows)))
        return predict(program, actual_rows)

    monkeypatch.setattr(abstractions, 'predict_rows', tracked)
    _, report, budget = run(template, rows)
    assert sum(count for _, count in calls) == budget.spent == report['actual_compute_cost']
    whole = Counter(program for program, _ in calls if program.op == 'call')
    assert whole and set(whole.values()) == {1}
    assert report['conditional_linear_profiles']


def test_profile_basis_shortfall_preserves_old_results_and_reports_denied_request():
    template, rows = fixture()
    _, full, full_budget = run(template, rows)
    finite_cost = full_budget.spent - sum(item['actual_compute_cost']
        for item in full['conditional_linear_profiles'])
    programs, report, budget = run(template, rows, finite_cost)
    assert programs and all('conditional_' not in method for method in report['candidate_methods'])
    assert report['conditional_linear_profiles'][0]['linear_diagnostics']['reason'] == 'linear_basis_budget_exhausted'
    assert report['budget_exhausted'] and not report['search_completed']
    assert report['status'] == 'UNKNOWN' and budget.last_denied_request['additional_compute'] > 0
    assert budget.spent == report['actual_compute_cost'] == finite_cost


def test_rank_deficient_profiles_keep_both_solvers_and_unknown_identifiability():
    template, rows = fixture(duplicated_offset=True)
    _, report, _ = run(template, rows)
    profiles = report['conditional_linear_profiles']
    assert profiles
    assert all(profile['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
               for profile in profiles)
    assert all(profile['linear_diagnostics']['rank_deficient'] for profile in profiles)
    assert all(profile['parameter_identifiability'] == 'UNKNOWN' for profile in profiles)
    assert report['parameter_identifiability'] == 'UNKNOWN' and report['status'] == 'UNKNOWN'


def test_non_affine_remaining_slots_keep_the_old_finite_policy():
    template, rows = fixture(nonlinear_ordinary=True)
    _, report, _ = run(template, rows)
    assert 'conditional_linear_profiles' not in report
    assert all('conditional_' not in method for method in report['candidate_methods'])
    assert report['search_completed'] and report['status'] == 'UNKNOWN'


def test_pure_linear_contract_cost_and_methods_are_unchanged():
    x = Program('feature', name='x')
    pattern = op('add', op('mul', const(0.), x), const(0.))
    template = AbstractTemplate('linear', 'fixture', pattern, (), (), ((0, 0), (1,)))
    rows = [Observation({'x': value}, .37 * value + .23) for value in (-2., -1., 0., 1., 2.)]
    _, report, budget = run(template, rows)
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert 'conditional_linear_profiles' not in report
    assert budget.spent == (2 + 4) * len(rows) and report['status'] == 'completed'


def test_legacy_template_calibration_keeps_its_original_budget_and_outputs():
    template, rows = fixture()
    _, selected_report, selected_budget = run(template, rows)
    finite_cost = selected_budget.spent - sum(item['actual_compute_cost']
        for item in selected_report['conditional_linear_profiles'])
    finite, finite_report, _ = run(template, rows, finite_cost)
    finite_best = min(finite, key=finite_report['losses'].__getitem__)
    expected_calibrated = abstractions._fit_linear((finite_best,), rows)
    budget, report = SearchBudget(finite_cost + 2 * len(rows)), {}
    actual = abstractions.fit_template(template, {}, rows, budget, report=report)
    assert actual[:len(finite)] == finite
    assert actual[-1] == expected_calibrated
    assert report['candidate_methods'] == [*finite_report['candidate_methods'], 'output_calibration']
    assert 'conditional_linear_profiles' not in report
    assert budget.spent == report['actual_compute_cost'] == finite_cost + 2 * len(rows)
