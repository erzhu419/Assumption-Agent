import json
from random import Random

import pytest

from hegel_machine.cognition.continuous_state import continuous_state_candidates
from hegel_machine.cognition.language import Definition, EvaluationContext, Program
from hegel_machine.cognition.models import Observation, SearchBudget


def trajectories(lengths, *, seed, decay=0.873, names=("u", "sensor"), square=False,
                 gains=(1.2, -0.4), initial=0.0, second_decay=None):
    rng = Random(seed)
    result = []
    for length in lengths:
        state, second, history = initial, 0.0, []
        for _ in range(length):
            values = {name: rng.uniform(-1.0, 1.0) for name in names}
            state = decay * state + sum(gain * values[name] for gain, name in zip(gains, names))
            if second_decay is not None:
                second = second_decay * second + 0.6 * values[names[0]] + 0.25 * values[names[1]]
            target = (state * state if square else state) + 0.1 * values[names[0]] + second
            result.append(Observation(values, target, tuple(history)))
            history.append({**values, "__target__": target})
    return result


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_accumulate_replays_prefix_and_resets_with_executable_calls():
    drive = Program("arg", name="input")
    definition = Definition("memory", ("input",), Program("accumulate", value=0.5,
                            children=(drive, Program("const", value=2.0))))
    call = Program("call", name="memory", children=(Program("feature", name="u"),), definition=definition)
    program = Program("add", children=(call, call))
    row = Observation({"u": 4.0}, -999.0, ({"u": 1.0, "__target__": 900.0},
                                                  {"u": 2.0, "__target__": -800.0}))
    context = EvaluationContext(row)
    assert program.predict(row, context=context) == 11.0
    assert program.expanded().predict(row) == 11.0
    assert context.call_evaluations == 1
    assert context.cache_hits == 1
    assert context.primitive_steps == 9
    assert call.predict(Observation({"u": 4.0}, 0.0)) == 5.0
    assert call.minimum_history_depth == 0
    assert call.history_depth >= 1
    renamed = Program.from_dict(json.loads(json.dumps(program.to_dict()))).remap_features({"u": "voltage"})
    renamed_row = Observation({"voltage": 4.0}, 0.0, ({"voltage": 1.0}, {"voltage": 2.0}))
    assert renamed.predict(renamed_row) == 11.0


def test_linear_memory_is_identified_and_runs_without_past_outputs():
    train = trajectories([7, 15, 25], seed=41, initial=0.8)
    budget = SearchBudget(6000)
    candidates = continuous_state_candidates(train, budget=budget)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    test = trajectories([1, 3, 31, 71], seed=442, initial=0.8)
    without_targets = [Observation(row.features, row.target,
                      tuple({name: value for name, value in past.items() if name != "__target__"}
                            for past in row.history)) for row in test]
    assert mse(best, without_targets) < 1e-12
    assert "accumulate" in best.describe()
    assert all(name != "__target__" for name, _ in best.observed_variables())
    assert 0 < budget.spent <= budget.limit


def test_two_continuous_modes_are_composed_from_observed_inputs():
    train = trajectories([17, 29, 37], seed=84, decay=0.91, second_decay=0.36)
    budget = SearchBudget(9000)
    candidates = continuous_state_candidates(train, budget=budget, max_states=2)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    test = trajectories([5, 43, 63], seed=894, decay=0.91, second_decay=0.36)
    assert mse(best, test) < 1e-10
    assert best.describe().count("accumulate") == 2
    assert budget.spent <= budget.limit


def test_square_readout_learns_shared_decay_and_cross_input_drive():
    train = trajectories([11, 19, 23], seed=16, decay=0.84, square=True, gains=(0.8, 0.55))
    budget = SearchBudget(24000)
    candidates = continuous_state_candidates(train, budget=budget)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    test = trajectories([3, 33, 67], seed=93, decay=0.84, square=True, gains=(0.8, 0.55))
    assert mse(best, test) < 1e-6
    assert "accumulate" in best.describe()
    assert "mul" in best.describe()
    assert budget.spent <= budget.limit
    def states(program):
        return ({program} if program.op == "accumulate" else set()).union(
            *(states(child) for child in program.children))
    assert all(len(states(candidate.program)) <= 2 for candidate in candidates)


def test_reused_state_is_refitted_after_role_and_parameter_change():
    source = trajectories([15, 25], seed=35, decay=0.65)
    candidates = continuous_state_candidates(source, budget=SearchBudget(5000))
    learned = min(candidates, key=lambda candidate: mse(candidate.program, source)).program
    restored = Program.from_dict(json.loads(json.dumps(learned.to_dict())))
    restored = restored.remap_features({"u": "force", "sensor": "reading"})
    target = trajectories([13, 27], seed=73, decay=0.93, names=("force", "reading"), gains=(-0.3, 1.1))
    revised = continuous_state_candidates(target, budget=SearchBudget(6000), seed_programs=(restored,))
    best = min(revised, key=lambda candidate: mse(candidate.program, target)).program
    test = trajectories([7, 57], seed=344, decay=0.93, names=("force", "reading"), gains=(-0.3, 1.1))
    assert mse(best, test) < 1e-12
    assert mse(restored, test) > 0.1


def test_small_budget_reserves_a_complete_candidate_and_binary_path_is_untouched():
    rows = trajectories([8, 10], seed=51)
    budget = SearchBudget(7 * len(rows))
    candidates = continuous_state_candidates(rows, budget=budget)
    assert candidates
    assert budget.spent <= budget.limit
    binary = [Observation(row.features, float(row.target > 0), row.history) for row in rows]
    untouched = SearchBudget(1000)
    assert continuous_state_candidates(binary, budget=untouched) == []
    assert untouched.spent == 0


def test_accumulate_uses_history_older_than_a_fixed_lag_window():
    program = Program("accumulate", value=0.95, children=(Program("feature", name="u"),
                                                        Program("const", value=0.0)))
    zero = [{"u": 0.0}] * 15
    previous = Observation({"u": 0.0}, 0.0, tuple([{"u": 1.0}, *zero]))
    no_previous = Observation({"u": 0.0}, 0.0, tuple([{"u": 0.0}, *zero]))
    assert program.predict(previous) == pytest.approx(0.95 ** 16)
    assert program.predict(no_previous) == 0.0


def test_short_measured_trajectory_returns_the_low_rank_model_that_search_found():
    from hegel_machine.cognition.measurement_lab import make_lab
    from hegel_machine.cognition.measurement_research import _pilot_probe, measured_view

    # This is the public acquisition sequence of the failed measurement run.
    # Its almost rank-one fit used to stop optimization and then be discarded,
    # leaving only an unrelated, poorly predicting linear state candidate.
    laboratory = make_lab(1, seed=10320)
    pilots = [laboratory.observe(_pilot_probe(laboratory.spec, 10320, index, 24),
                                ("sensor_a", "sensor_b")) for index in range(2)]
    train, validation = [measured_view(rows, ("u", "sensor_a")) for rows in pilots]
    budget = SearchBudget(12000)
    candidates = continuous_state_candidates(train, budget=budget)
    best = min(candidates, key=lambda candidate: mse(candidate.program, train)).program
    assert mse(best, validation) < 1e-7
    independent = laboratory.evaluation_rows(("sensor_a",), seed=762, length=64, trajectories=3)
    assert mse(best, independent) < 1e-7
    assert budget.spent <= budget.limit
    laboratory.close()
