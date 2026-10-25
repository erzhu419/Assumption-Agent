"""D1 uses actual slot fitting, fixed targets and the production error scale."""

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.synthesis import _loss


def const(value):
    return Program('const', value=float(value))


def binary(op, left, right):
    return Program(op, children=(left, right))


X = Program('feature', name='x')


def affine_template():
    pattern = binary('add', const(0.), binary('mul', const(1.), X))
    return AbstractTemplate('exact_slots', 'mechanism_fixture', pattern, (), (), ((0,), (1, 0)))


@pytest.mark.parametrize('training,held', [
    ([1., 2., 4., 5.], [3., 6.]),
    ([.5000, .5001, .5003, .5004], [.5002, .5005]),
])
def test_stable_candidate_passes_actual_1e10_contract_while_ridge_is_retained(training, held):
    rows = [Observation({'x': value}, 2. * value + 1.) for value in training]
    report, budget = {}, SearchBudget(24)
    fitted = fit_parameter_slots(affine_template(), {}, rows, budget, report=report)
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq', 'stable_lstsq']
    ridge, stable = fitted[1:]
    assert max(abs(ridge.predict(row) - row.target) for row in rows) > 1e-10
    assert max(abs(stable.predict(row) - row.target) for row in rows) <= 1e-10
    assert all(abs(stable.predict(Observation({'x': value}, 0.)) - (2. * value + 1.)) <= 1e-10
               for value in held)
    assert report['status'] == 'completed' and report['search_completed']
    assert budget.spent == report['actual_compute_cost'] == (2 + 4) * len(rows)
    assert report['parameter_paths'] == [[0], [1, 0]] and report['selected_slot_degree'] == 1
    diagnostic = report['linear_diagnostics']
    assert diagnostic['rank'] == 2 and not diagnostic['rank_deficient']
    assert diagnostic['parameter_identifiability'] == 'full_rank_linear_design'
    assert len(diagnostic['singular_values']) == 2 and diagnostic['condition_number'] >= 1.
    assert diagnostic['ridge']['penalty'] == 1e-9 and diagnostic['stable']['penalty'] == 0.
    assert diagnostic['train_residual_max_abs'] <= 1e-10
    assert diagnostic['ridge']['objective'] > diagnostic['ridge']['train_residual_sum_squares']
    assert diagnostic['stable']['objective'] == diagnostic['stable']['train_residual_sum_squares']
    assert diagnostic['scope'] == 'linear_training_rows_only'
    assert not report['scientific_support'] and not report['global_optimum_established']


def test_rank_deficient_representative_can_predict_without_identifying_parameters():
    pattern = binary('add', binary('mul', const(1.), X), binary('mul', const(0.), X))
    template = AbstractTemplate('redundant_slots', 'fixture', pattern, (), (), ((0, 0), (1, 0)))
    rows = [Observation({'x': value}, 3. * value) for value in (-2., -1., 1., 2.)]
    report, budget = {}, SearchBudget(24)
    fitted = fit_parameter_slots(template, {}, rows, budget, report=report)
    stable = fitted[report['candidate_methods'].index('stable_lstsq')]
    assert all(abs(stable.predict(row) - row.target) <= 1e-10 for row in rows)
    assert report['status'] == report['parameter_identifiability'] == 'UNKNOWN'
    assert report['reason'] == 'rank_deficient_parameter_fit'
    assert report['linear_diagnostics']['rank'] == 1
    assert report['linear_diagnostics']['rank_deficient']
    assert report['search_completed'] and not report['budget_exhausted']
    assert budget.spent == report['actual_compute_cost'] == 24


def test_stable_diagnostic_does_not_spend_beyond_old_candidate_allowance():
    rows = [Observation({'x': value}, 2. * value + 1.) for value in (1., 2., 4., 5.)]
    report, budget = {}, SearchBudget(20)
    fitted = fit_parameter_slots(affine_template(), {}, rows, budget, report=report)
    assert report['candidate_methods'] == ['initial', 'ridge_lstsq'] and len(fitted) == 2
    assert report['status'] == 'UNKNOWN' and report['budget_exhausted']
    assert not report['search_completed']
    assert report['linear_diagnostics']['reason'] == 'stable_linear_candidate_budget_exhausted'
    assert report['linear_diagnostics']['rank'] is None
    assert budget.spent == report['actual_compute_cost'] == 20


def test_nonlinear_slot_trial_failure_is_unknown_with_bounded_search_completed():
    # Multiplicative dependence on both selected slots has a solution, but the
    # existing coordinate trials cannot improve either coordinate from (0, 0).
    pattern = binary('mul', const(0.), const(0.))
    template = AbstractTemplate('local_trials', 'fixture', pattern, (), (), ((0,), (1,)))
    rows = [Observation({'x': value}, 2.25) for value in (0., 1., 2., 3.)]
    report, budget = {}, SearchBudget(100)
    fitted = fit_parameter_slots(template, {}, rows, budget, report=report)
    assert fitted == [pattern]
    assert report['selected_slot_degree'] == 2
    assert report['status'] == report['parameter_identifiability'] == 'UNKNOWN'
    assert report['reason'] == 'nonlinear_parameter_search_is_local'
    assert report['search_completed'] and not report['budget_exhausted']
    assert report['actual_compute_cost'] == budget.spent == 5 * len(rows)
    assert 'linear_diagnostics' not in report and not report['global_optimum_established']


def test_nonlinear_observable_and_quantile_evaluation_keep_their_original_objective():
    square = binary('mul', X, X)
    pattern = binary('add', const(0.), binary('mul', const(1.), square))
    template = AbstractTemplate('nonlinear_input', 'fixture', pattern, (), (), ((0,), (1, 0)))
    rows = [Observation({'x': value}, 3. * value * value + 7.) for value in (-2., -1., 1., 2.)]
    report, budget = {}, SearchBudget(100)
    programs = fit_parameter_slots(template, {}, rows, budget, q=.75, report=report)
    stable = programs[report['candidate_methods'].index('stable_lstsq')]
    assert report['selected_slot_degree'] == 1
    assert report['linear_diagnostics']['evaluation_objective'] == 'quantile_loss'
    assert all(report['losses'][program] == _loss(program, rows, .75) for program in programs)
    assert all(abs(stable.predict(row) - row.target) <= 1e-10 for row in rows)
    assert not report['global_optimum_established']
