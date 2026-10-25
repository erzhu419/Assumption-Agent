"""Fit declared slots inside a complete after Program with a fixed parent."""

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots, fit_template
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget


def constant(value):
    return Program('const', value=value)


def binary(op, left, right):
    return Program(op, children=(left, right))


X = Program('feature', name='x')
PARENT = binary('mul', constant(5.), X)


def template(program, paths):
    return AbstractTemplate('anchored_fixture', 'retained_before', program, (), (), tuple(paths))


def observations(target):
    return [Observation({'x': value}, target(value)) for value in (-2., -1., 1., 2.)]


def best(programs, report):
    return min(programs, key=report['losses'].__getitem__)


def test_offset_fits_the_whole_after_and_both_fitters_retain_the_stable_candidate():
    pattern = binary('add', PARENT, constant(0.))
    selected = template(pattern, [(1,)])
    rows = observations(lambda x: 5. * x + 7.)
    report, work = {}, SearchBudget(1000)
    programs = fit_parameter_slots(selected, {}, rows, work, report=report)
    fitted = best(programs, report)
    assert fitted.children[1].value == pytest.approx(7., abs=1e-8)
    assert all(program.op == 'add' and program.children[0] == PARENT for program in programs)
    assert work.spent == report['actual_compute_cost'] == 5 * len(rows)
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    assert report['status'] == 'completed' and report['actual_fitted_rows'] == len(rows)
    assert not report['scientific_support'] and not report['global_optimum_established']
    legacy_work, legacy_report = SearchBudget(1000), {}
    legacy = fit_template(selected, {}, rows, legacy_work, report=legacy_report)
    # The old fitter still attempts its normal whole-output calibration. Here
    # its result is cached as the same Program, so that attempt costs one pass.
    assert legacy == programs and legacy_work.spent == 6 * len(rows)
    assert legacy_report['candidate_methods'] == report['candidate_methods']
    assert legacy_report['linear_diagnostics'] == report['linear_diagnostics']


def test_data_only_guard_and_fixed_parent_constants_leave_startup_parameters_linear():
    parent_arg, startup_arg = Program('arg', name='before'), Program('arg', name='startup')
    guard = binary('ge', X, constant(0.))
    definition = Definition('conditional_context', ('before', 'startup'),
        Program('if', children=(guard, parent_arg, startup_arg)))
    startup = binary('add', binary('mul', constant(0.), X), constant(0.))
    pattern = Program('call', name=definition.definition_id, children=(PARENT, startup), definition=definition)
    selected = template(pattern, [(1, 0, 0), (1, 1)])
    rows = observations(lambda x: 5. * x if x >= 0. else 3. * x + 7.)
    report, work = {}, SearchBudget(1000)
    fitted = best(fit_parameter_slots(selected, {}, rows, work, report=report), report)
    assert fitted.op == 'call' and fitted.definition == definition and fitted.children[0] == PARENT
    assert fitted.children[1].children[0].children[0].value == pytest.approx(3., abs=1e-7)
    assert fitted.children[1].children[1].value == pytest.approx(7., abs=1e-7)
    assert all(fitted.predict(row) == pytest.approx(row.target, abs=1e-7) for row in rows)
    assert work.spent == report['actual_compute_cost'] == 6 * len(rows)
    assert definition.body.children[0] == guard and report['status'] == 'completed'


def test_one_lexical_hole_parameter_stays_shared_at_both_definition_occurrences():
    parent_arg, offset_arg = Program('arg', name='before'), Program('arg', name='offset')
    definition = Definition('repeated_hole', ('before', 'offset'),
        binary('add', parent_arg, binary('add', offset_arg, offset_arg)))
    pattern = Program('call', name=definition.definition_id, children=(PARENT, constant(0.)), definition=definition)
    report, work = {}, SearchBudget(1000)
    rows = observations(lambda x: 5. * x + 14.)
    fitted = best(fit_parameter_slots(template(pattern, [(1,)]), {}, rows, work, report=report), report)
    assert fitted.children[0] == PARENT and fitted.children[1].value == pytest.approx(7., abs=1e-8)
    assert fitted.definition == definition
    expanded = fitted.expanded()
    assert expanded.children[1].children[0] == expanded.children[1].children[1] == fitted.children[1]
    assert work.spent == 5 * len(rows) and report['status'] == 'completed'


def test_nonlinear_trials_preserve_the_parent_and_do_not_claim_a_global_fit():
    parent_arg, offset_arg = Program('arg', name='before'), Program('arg', name='offset')
    definition = Definition('nonlinear_hole', ('before', 'offset'),
        binary('add', parent_arg, binary('mul', offset_arg, offset_arg)))
    pattern = Program('call', name=definition.definition_id, children=(PARENT, constant(0.)), definition=definition)
    selected = template(pattern, [(1,)])
    rows = observations(lambda x: 5. * x + 49.)
    report, work = {}, SearchBudget(1000)
    programs = fit_parameter_slots(selected, {}, rows, work, report=report)
    assert report['status'] == 'UNKNOWN' and report['losses'][best(programs, report)] > 1.
    assert report['reason'] == 'nonlinear_parameter_search_is_local'
    assert report['search_completed'] and not report['budget_exhausted']
    assert not report['global_optimum_established'] and not report['scientific_support']
    assert all(program.definition == definition and program.children[0] == PARENT for program in programs)
    assert work.spent == report['actual_compute_cost'] == 3 * len(rows)
    truncated, limited = {}, SearchBudget(len(rows))
    partial = fit_parameter_slots(selected, {}, rows, limited, report=truncated)
    assert partial == [pattern]
    assert truncated['status'] == 'UNKNOWN' and truncated['budget_exhausted']
    assert truncated['actual_compute_cost'] == limited.spent == len(rows)
    assert not truncated['search_completed'] and not truncated['scientific_support']


def test_budget_before_first_fit_and_nonconstant_paths_never_change_history_structure():
    pattern = binary('add', PARENT, constant(0.))
    rows = observations(lambda x: 5. * x + 7.)
    report, work = {}, SearchBudget(len(rows) - 1)
    assert fit_parameter_slots(template(pattern, [(1,)]), {}, rows, work, report=report) == []
    assert report['status'] == 'UNKNOWN' and report['actual_fitted_rows'] == 0
    assert report['eligible_fitting_rows'] == len(rows)
    assert report['actual_compute_cost'] == work.spent == 0 and report['losses'] == {}
    lagged = Program('lag', name='x', lag=3)
    with pytest.raises(ValueError, match='actual const slots'):
        fit_parameter_slots(template(lagged, [()]), {}, rows, SearchBudget(100))
    assert lagged.lag == 3 and PARENT == binary('mul', constant(5.), X)
