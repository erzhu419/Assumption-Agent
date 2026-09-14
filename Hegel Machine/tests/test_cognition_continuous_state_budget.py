from random import Random

import pytest

from hegel_machine.cognition.continuous_state import continuous_state_candidates
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import _pilot_probe, measured_view
from hegel_machine.cognition.models import Observation, SearchBudget


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


@pytest.mark.parametrize("seed", [28320, 68348])
def test_small_instrument_pilot_fits_the_actual_compressed_state(seed):
    # The two observed failures had validation MSE 2.20 and .52 even when the
    # old search was given 40,000 units. Only the first paid trajectory fits
    # the program; the other trajectory never chooses the candidate or pole.
    with make_lab(1, seed=seed) as lab:
        train, validation = [measured_view(lab.observe(
            _pilot_probe(lab.spec, seed, index, 24), ("sensor_a", "sensor_b")),
            ("u", "sensor_a")) for index in range(2)]
    budget = SearchBudget(6000)
    candidates = continuous_state_candidates(train, budget=budget)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    assert mse(best, validation) < 1e-6
    assert 0 < budget.spent <= budget.limit


def trajectory(seed, decay, initial, length, *, second_decay=None):
    rng, history, result = Random(seed), [], []
    state, second = initial, -.2
    for _ in range(length):
        features = {name: rng.uniform(-1, 1) for name in ("drive", "reading")}
        state = decay * state + .55 * features["drive"] - .9 * features["reading"]
        if second_decay is None:
            target = state * state + .17 * features["drive"]
        else:
            second = second_decay * second + .3 * features["drive"] + .2 * features["reading"]
            target = state + second + .17 * features["drive"]
        result.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return result


@pytest.mark.parametrize("decay,initial,seed", [(.78, .4, 42), (.91, -.3, 102)])
def test_nonlinear_finite_budget_search_learns_other_drives_and_initial_states(decay, initial, seed):
    train = trajectory(seed, decay, initial, 24)
    budget = SearchBudget(7480)
    candidates = continuous_state_candidates(train, budget=budget)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    validation = trajectory(seed + 187, decay, initial, 40)
    assert mse(best, validation) < 1e-8
    assert budget.spent <= budget.limit


def test_small_budget_still_identifies_two_distinct_driven_states():
    train = trajectory(26, .89, .4, 24, second_decay=.32)
    budget = SearchBudget(6000)
    candidates = continuous_state_candidates(train, budget=budget)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    validation = trajectory(403, .89, .4, 51, second_decay=.32)
    no_outputs = [Observation(row.features, row.target,
        tuple({name: value for name, value in past.items() if name != "__target__"}
              for past in row.history)) for row in validation]
    assert mse(best, no_outputs) < 1e-10
    assert best.describe().count("accumulate") == 2
    assert budget.spent <= budget.limit
