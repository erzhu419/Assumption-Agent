"""Research fit consumers preserve paid work when using the existing row pool."""
from random import Random

import pytest

from hegel_machine.cognition import parallel_prediction as kernel
from hegel_machine.cognition import research_programs as research
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.experiment_runtime import physical_worker_cpus
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget


def binary(op, left, right):
    return Program(op, children=(left, right))


def inherited_terms():
    argument = Program('arg', name='input')
    square = binary('mul', argument, argument)
    window = Program('window_sum', lag=3, children=(square,))
    gated = Program('if', children=(binary('ge', argument, Program('const', value=.17)),
        window, binary('add', window, argument)))
    definition = Definition('retained_nonlinear_parent', ('input',), gated, 'retained_source')
    parent = Program('call', name=definition.definition_id, definition=definition,
        children=(Program('feature', name='u'),))
    return (parent,
        Program('accumulate', value=.63, children=(parent, Program('const', value=.2))),
        Program('window_sum', lag=4, children=(parent,)))


def observations(terms):
    rng, rows = Random(418), []
    for length in (25, 33):
        history = []
        for _ in range(length):
            features = {'u': rng.uniform(-1., 1.)}
            row = Observation(features, 0., tuple(history))
            values = [term.predict(row) for term in terms]
            target = 1.7 * values[0] + .2 * values[1] - .8 * values[2] + .13
            rows.append(Observation(features, target, tuple(history)))
            history.append(dict(features))
    return rows


@pytest.fixture(autouse=True)
def reset_prediction_workers(monkeypatch):
    kernel.configure_prediction_workers(())
    monkeypatch.setattr(kernel, '_MIN_PARALLEL_SECONDS', 0.)
    yield
    kernel.configure_prediction_workers(())


def enable_parallel():
    cpus = physical_worker_cpus()[:2]
    if len(cpus) < 2:
        pytest.skip('Two assigned physical CPUs are required')
    kernel.configure_prediction_workers(cpus)


@pytest.mark.parametrize('quantile', [.5, .8])
@pytest.mark.parametrize('passes', [4, 5])
def test_fitter_columns_loss_candidates_and_paid_budget_match_serial(quantile, passes, monkeypatch):
    terms = inherited_terms()
    rows = observations(terms)
    serial_budget = SearchBudget(passes * len(rows))
    serial_fitter = research._Fitter(rows, serial_budget, quantile)
    serial = serial_fitter.fit(terms)
    expected_columns = {term: [term.predict(row) for row in rows] for term in terms}
    enable_parallel()
    calls, predict_rows = [], research.predict_rows

    def observed(program, examples):
        calls.append(program)
        return predict_rows(program, examples)

    monkeypatch.setattr(research, 'predict_rows', observed)
    parallel_budget = SearchBudget(passes * len(rows))
    parallel_fitter = research._Fitter(rows, parallel_budget, quantile)
    parallel = parallel_fitter.fit(terms)
    assert parallel == serial
    assert parallel_budget == serial_budget
    if passes == 4:
        assert parallel is None and not calls
        assert kernel._EXECUTOR is None
        return
    assert kernel._EXECUTOR is not None
    assert parallel_fitter.columns == serial_fitter.columns == expected_columns
    assert len(calls) == len(terms) + 1  # Three funded columns plus final candidate scoring.
    assert parallel['loss'] == loss(parallel['program'], rows,
                                   parallel_fitter.fallback, quantile)
    assert parallel_budget.spent == 5 * len(rows)
    assert parallel_budget.completed_fits == 1
    assert parallel_fitter.fit(terms) == parallel
    assert len(calls) == len(terms) + 1
    path = [{'operation': 'transform_component', 'question': 'Retain paid composed parent',
             'program': serial['program'], 'loss': serial['loss']}]
    assert research._entry(parallel, path, len(rows), parallel_budget.spent) == research._entry(
        serial, path, len(rows), serial_budget.spent)


def test_saved_research_candidate_replay_and_cost_are_identical():
    terms = inherited_terms()
    rows = observations(terms)
    source_budget = SearchBudget(1000)
    source = research._Fitter(rows, source_budget, .5).fit(terms)
    path = [{'operation': 'transform_component', 'question': 'Refit retained temporal terms',
             'program': source['program'], 'loss': source['loss']}]
    payload = research._entry(source, path, len(rows), source_budget.spent)['research_program']
    serial_budget = SearchBudget(1000)
    serial = research.replay_research_program(payload, rows, serial_budget)
    enable_parallel()
    parallel_budget = SearchBudget(1000)
    parallel = research.replay_research_program(payload, rows, parallel_budget)
    assert parallel == serial
    assert parallel_budget == serial_budget
    assert kernel._EXECUTOR is not None


@pytest.mark.parametrize('failure', ['missing_feature', 'missing_history', 'nonfinite'])
def test_invalid_columns_remain_paid_failures_and_cache_none(failure):
    terms = inherited_terms()
    rows = observations(terms)
    term = terms[0]
    if failure == 'missing_feature':
        rows[19].features.clear()  # Beyond the serial timing prefix, in a real worker block.
    elif failure == 'missing_history':
        term = Program('lag', name='u', lag=2)
    else:
        term = binary('mul', Program('const', value=1e308),
                      Program('const', value=1e308))
    serial_budget = SearchBudget(1000)
    serial = research._Fitter(rows, serial_budget, .5)
    assert serial.column(term) is None
    enable_parallel()
    parallel_budget = SearchBudget(1000)
    parallel = research._Fitter(rows, parallel_budget, .5)
    assert parallel.column(term) is None
    assert parallel.column(term) is None
    assert parallel_budget == serial_budget
    assert parallel_budget.spent == len(rows)
    assert parallel.columns == serial.columns == {term: None}
