"""A purchased prefix retains the full experiment's pairwise response distinctions."""
from copy import deepcopy

from hegel_machine.cognition.competing_object_experiments import (
    _forecast, assess_competing_object_probe, shorten_discriminating_probe)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget


def plan(programs, points):
    return {'probe': Probe('separation', tuple(points)), 'forecast_status': 'predicted_distinction',
        'hypotheses': [{'hypothesis_id': str(i), 'candidate_id': str(i), 'program': p.to_dict(),
                       'predictions': _forecast(p, points)} for i, p in enumerate(programs)],
        'predicted_fields': {'x': [point['x'] for point in points]}, 'compute_cost': 0,
        'predicted_question_phases': ['enter', 'maintain', 'exit']}


def observe(program, points):
    rows, history = [], []
    for point in points:
        target = program.predict(Observation(point, 0., tuple(history)))
        row = Observation(point, target, tuple(history))
        rows.append((row, row))
        history.append({**point, '__target__': target})
    return rows


def test_forecast_prefix_keeps_history_and_real_refutation_without_buying_tail():
    x = Program('feature', name='x')
    event = Program('event_delta', children=(x,), lag=2)
    law = Program('window_sum', children=(event,), lag=3)
    programs = [law, Program('mul', children=(Program('const', value=2.), law)), Program('const', value=0.)]
    points = [{'x': value} for value in (0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.)]
    original = plan(programs, points)
    frozen = deepcopy(original)
    budget = SearchBudget(1000)
    short = shorten_discriminating_probe(original, budget)
    assert 2 < short['probe'].cost < len(points)
    assert short['probe'].sequence == tuple(points[:short['probe'].cost])
    assert len(short['predicted_fields']['x']) == short['probe'].cost
    assert 'predicted_question_phases' not in short
    assert original == frozen
    assert budget.spent == 3 * len(points)
    for truth in programs:
        full = assess_competing_object_probe(original, observe(truth, points), SearchBudget(1000))
        prefix = assess_competing_object_probe(short, observe(truth, short['probe'].sequence), SearchBudget(1000))
        assert prefix['status'] == full['status'] == 'distinguished'
        assert [h['status'] for h in prefix['hypotheses']] == [h['status'] for h in full['hypotheses']]
        assert prefix['observation_rows'] < full['observation_rows']


def test_late_distinction_of_a_third_hypothesis_is_not_discarded():
    x = Program('feature', name='x')
    programs = [Program('const', value=0.), x, Program('window_sum', children=(x,), lag=2)]
    points = [{'x': value} for value in (0., 0., 0., 1., 1.)]
    original = plan(programs, points)
    short = shorten_discriminating_probe(original, SearchBudget(1000))
    # First two separate at row four; second and third require the last row.
    assert short['probe'] == original['probe']


def test_calibration_or_insufficient_compute_keeps_the_original_experiment():
    programs = [Program('const', value=0.), Program('feature', name='x')]
    original = plan(programs, [{'x': 1.}, {'x': 0.}, {'x': 0.}])
    unforecast = {**original, 'forecast_status': 'requires_paired_measurements'}
    assert shorten_discriminating_probe(unforecast, SearchBudget(100)) is unforecast
    assert shorten_discriminating_probe(original, SearchBudget(2))['probe'] == original['probe']
