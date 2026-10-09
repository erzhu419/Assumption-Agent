"""Paid sensor trajectories identify recurrent observation models."""

from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.instrument_models import fit_instrument_model, instrument_binding
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


COMMAND = TaskSpec('stateful_meter', 'public actuator', ('u',), (), (),
                   feature_bounds={'u': (-1., 1.)})
MEASURED = replace(COMMAND, feature_names=('meter',))


def measurements(seed, *, lengths=(32, 32), decay=.873, initial=.8, square=False):
    random, result = Random(seed), []
    for length in lengths:
        state, history = initial, []
        for _ in range(length):
            control = random.uniform(-1., 1.)
            state = decay * state + 1.1 * control
            value = .17 + .4 * control + 1.3 * (state ** 2 if square else state)
            command = Observation({'u': control}, 8765., tuple(history))
            measured = Observation({'meter': value}, -5432., tuple(
                {'meter': 7777., '__target__': 5555.} for _ in history))
            result.append((command, measured))
            history.append({'u': control, '__target__': -4321., 'response_only': 3333.})
    return result


def squared_errors(program, pairs):
    return [(program.predict(command) - measured.features['meter']) ** 2
            for command, measured in pairs]


@pytest.mark.parametrize('square,decay,initial', [(False, .873, .8), (True, .84, .3)])
def test_recurrent_field_generalizes_beyond_finite_control_window(square, decay, initial):
    training = measurements(151, square=square, decay=decay, initial=initial)
    budget = SearchBudget(24000)
    model = fit_instrument_model(COMMAND, MEASURED, training, budget)
    assert model['fields']['meter']['kind'] == 'continuous_state'
    program = instrument_binding(json.loads(json.dumps(model)))['meter']
    assert 'accumulate' in program.describe()
    fresh = measurements(617, lengths=(1, 3, 19, 81), square=square, decay=decay, initial=initial)
    errors = squared_errors(program, fresh)
    assert sum(errors) / len(errors) < 1e-8
    assert max(errors) < 1e-6
    assert {name for name, _ in program.observed_variables()} == {'u'}
    assert model['compute_cost'] == budget.spent <= 24000
    assert model['status'] == 'requires_fresh_validation'


def test_state_identification_uses_only_its_purchased_field_targets():
    training = measurements(193)
    changed = [(replace(command, target=float('nan'), history=tuple(
        {**past, '__target__': float('inf'), 'response_only': float('nan')}
        for past in command.history)), replace(measured, target=float('inf'), history=tuple(
            {'meter': float('nan'), '__target__': float('inf')} for _ in measured.history)))
        for command, measured in training]
    original = fit_instrument_model(COMMAND, MEASURED, training, SearchBudget(24000))
    independent = fit_instrument_model(COMMAND, MEASURED, changed, SearchBudget(24000))
    assert original == independent
    assert original['fields']['meter']['kind'] == 'continuous_state'


def test_disabled_history_and_unfunded_state_do_not_create_an_instrument_binding():
    training = measurements(211)
    for history, limit in [(0, 24000), (4, 700)]:
        budget = SearchBudget(limit)
        model = fit_instrument_model(COMMAND, MEASURED, training, budget, max_history=history)
        assert not model['fields'] and 'meter' in model['unmodeled_fields']
        assert not instrument_binding(model)
        assert model['compute_cost'] == budget.spent <= limit


def test_two_persistent_instrument_modes_remain_executable_on_longer_trajectories():
    def pairs(seed, lengths):
        random, result = Random(seed), []
        for length in lengths:
            first, second, history = .6, -.3, []
            for _ in range(length):
                control = random.uniform(-1., 1.)
                first, second = .91 * first + .8 * control, -.36 * second - .45 * control
                value = .2 * control + first + second
                result.append((Observation({'u': control}, 9876., tuple(history)),
                               Observation({'meter': value}, -8765.)))
                history.append({'u': control, '__target__': 1234.})
        return result

    budget = SearchBudget(24000)
    model = fit_instrument_model(COMMAND, MEASURED, pairs(307, (32, 32)), budget)
    program = instrument_binding(json.loads(json.dumps(model)))['meter']
    assert model['fields']['meter']['kind'] == 'continuous_state'
    assert program.describe().count('accumulate') == 2
    errors = squared_errors(program, pairs(919, (1, 3, 96)))
    assert max(errors) < 1e-8
    assert budget.spent == model['compute_cost'] <= 24000
