"""Pure prediction workers must not change fitting or finite research budgets."""

import os

import pytest

from hegel_machine.cognition.abstractions import AbstractTemplate, fit_parameter_slots
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition import parallel_prediction as kernel


@pytest.fixture(autouse=True)
def reset_prediction_pool(monkeypatch):
    kernel.configure_prediction_workers(())
    monkeypatch.setattr(kernel, '_MIN_PARALLEL_SECONDS', 0.)
    yield
    kernel.configure_prediction_workers(())


def parallel_cpus():
    return tuple(sorted(os.sched_getaffinity(0))[-2:])


def rows():
    return [Observation({'x': float(index)}, 2. * index + 1.,
                        tuple({'x': float(previous)} for previous in range(index)))
            for index in range(33)]


def binary(op, left, right):
    return Program(op, children=(left, right))


def const(value):
    return Program('const', value=float(value))


def test_parallel_rows_preserve_guarded_history_call_values_and_reuse_pool():
    history = Program('window_sum', lag=3, children=(Program('feature', name='x'),))
    definition = Definition('actual_history', ('before',), binary('add',
        Program('arg', name='before'), Program('event_delta', lag=2,
            children=(Program('feature', name='x'),))))
    program = Program('call', name=definition.definition_id, children=(history,), definition=definition)
    observations = rows()
    expected = [program.predict(row) for row in observations]
    kernel.configure_prediction_workers(parallel_cpus())
    assert kernel.predict_rows(program, observations) == expected
    executor = kernel._EXECUTOR
    assert executor is not None
    assert kernel.predict_rows(program, list(observations)) == expected
    assert kernel._EXECUTOR is executor
    assert set().union(*(os.sched_getaffinity(pid) for pid in executor._processes)) <= set(parallel_cpus())
    changed = observations[:-1]
    assert kernel.predict_rows(program, changed) == expected[:-1]
    assert kernel._EXECUTOR is not executor


@pytest.mark.parametrize('passes', [0, 1, 5, 6])
def test_fixed_and_exhausted_fit_budgets_keep_all_programs_losses_and_diagnostics(passes):
    observations = rows()
    x = Program('feature', name='x')
    pattern = binary('add', const(0.), binary('mul', const(1.), x))
    template = AbstractTemplate('slots', 'source', pattern, (), (), ((0,), (1, 0)))
    limit = passes * len(observations) if passes else len(observations) - 1
    serial_budget, serial_report = SearchBudget(limit), {}
    serial = fit_parameter_slots(template, {}, observations, serial_budget, report=serial_report)
    kernel.configure_prediction_workers(parallel_cpus())
    parallel_budget, parallel_report = SearchBudget(limit), {}
    parallel = fit_parameter_slots(template, {}, observations, parallel_budget, report=parallel_report)
    assert parallel == serial
    assert parallel_report == serial_report
    assert parallel_budget == serial_budget
    if passes == 0:
        assert kernel._EXECUTOR is None


def test_nonlinear_negative_result_and_quantile_loss_are_identical():
    observations = rows()
    pattern = binary('mul', const(0.), const(0.))
    template = AbstractTemplate('local_failure', 'source', pattern, (), (), ((0,), (1,)))
    serial_budget, serial_report = SearchBudget(1000), {}
    serial = fit_parameter_slots(template, {}, observations, serial_budget, q=.75, report=serial_report)
    kernel.configure_prediction_workers(parallel_cpus())
    parallel_budget, parallel_report = SearchBudget(1000), {}
    parallel = fit_parameter_slots(template, {}, observations, parallel_budget, q=.75, report=parallel_report)
    assert parallel == serial == [pattern]
    assert parallel_report == serial_report
    assert parallel_report['reason'] == 'nonlinear_parameter_search_is_local'
    assert parallel_budget == serial_budget


def test_prediction_failure_propagates_without_turning_into_a_zero_score():
    observations = rows()
    observations[19].features.clear()
    program = Program('feature', name='x')
    with pytest.raises(KeyError) as serial:
        kernel.predict_rows(program, observations)
    kernel.configure_prediction_workers(parallel_cpus())
    with pytest.raises(KeyError) as parallel:
        kernel.predict_rows(program, observations)
    assert parallel.value.args == serial.value.args == ('x',)


def test_cheap_or_short_batches_do_not_start_a_pool(monkeypatch):
    monkeypatch.setattr(kernel, '_MIN_PARALLEL_SECONDS', 100.)
    program = Program('feature', name='x')
    observations = rows()
    kernel.configure_prediction_workers(parallel_cpus())
    assert kernel.predict_rows(program, observations) == [row.features['x'] for row in observations]
    assert kernel.predict_rows(program, observations[:8]) == [row.features['x'] for row in observations[:8]]
    assert kernel._EXECUTOR is None
