"""Learn transition conditions without confusing state, direction or controls."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.state_search import state_candidates


def _transition_rows(sequences, *, direction=1, levels=(0.0, 1.0)):
    rows = []
    for values in sequences:
        history, active = [], False
        for index, value in enumerate(values):
            active = value >= 0.6 or (active and value >= -0.6)
            features = {"signal": direction * value, "distractor": float(index % 3 - 1)}
            target = levels[int(active)]
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
    return rows


_SEQUENCES = (
    (-1.0, 0.0), (1.0, 0.0),
    (-1.2, 1.2, 0.2, -0.2, -1.2),
    (1.2, 0.4, -0.4, -1.2, 0.0, 0.8),
    (0.0, 0.8, 0.0, -0.8, 0.0),
    (-0.8, 0.8, 0.5, -0.5, -0.8),
    (0.8, 0.2, -0.2, -0.8),
)


@pytest.mark.parametrize("direction, levels", ((1, (0.0, 1.0)), (-1, (0.0, 1.0)), (1, (-2.0, 3.0))))
def test_learns_both_transition_directions_and_observed_output_states(direction, levels):
    training = _transition_rows(_SEQUENCES, direction=direction, levels=levels)
    budget = SearchBudget(900)
    candidates = state_candidates(training, budget=budget)
    winner = min(candidates, key=lambda candidate: (loss(candidate.program, training, 0.0, 0.5),
                                                   candidate.program.size)).program
    assert winner.minimum_history_depth == 0 and winner.history_depth == 1
    assert winner.observed_variables() == {("signal", 0), ("__target__", 1)}
    assert loss(winner, training, 0.0, 0.5) == 0.0
    # Exactly the same current features must lead to different outputs after
    # different recorded states. A rule on a correlated distractor cannot pass.
    features = {"signal": 0.0, "distractor": 0.0}
    for state in levels:
        row = Observation(features, state, ({**features, "__target__": state},))
        assert winner.predict(row) == state
    fresh = _transition_rows(((0.9, 0.1, -0.1, -0.9, 0.1), (-0.9, 0.1, 0.9, -0.1)),
                             direction=direction, levels=levels)
    assert loss(winner, fresh, 0.0, 0.5) == 0.0
    assert any(candidate.program.history_depth == 0 for candidate in candidates)
    assert 0 < budget.spent <= 900


def test_equally_fitting_transition_boundaries_disagree_on_an_unobserved_control():
    rows = _transition_rows(_SEQUENCES)
    candidates = state_candidates(rows, budget=SearchBudget(900))
    fitted = [candidate.program for candidate in candidates if loss(candidate.program, rows, 0.0, 0.5) == 0.0]
    # The entry boundary lies between observed controls 0.5 and 0.8. Preserve
    # alternative cuts until an experiment inside that interval resolves them.
    features = {"signal": 0.6, "distractor": 0.0}
    unseen = Observation(features, 1.0, ({"signal": -1.0, "distractor": 0.0, "__target__": 0.0},))
    assert {program.predict(unseen) for program in fitted} == {0.0, 1.0}


def test_memoryless_law_wins_when_previous_states_do_not_change_the_transition():
    rows = [Observation({"control": value}, float(value >= 0.2),
                        ({"control": -value, "__target__": previous},))
            for value in (-1.0, -0.3, 0.4, 1.0) for previous in (0.0, 1.0)]
    rows.extend(Observation({"control": value}, float(value >= 0.2)) for value in (-0.3, 0.4))
    candidates = state_candidates(rows, budget=SearchBudget(500))
    winner = min(candidates, key=lambda candidate: (loss(candidate.program, rows, 0.0, 0.5),
                                                   candidate.program.size)).program
    assert winner.history_depth == 0
    assert loss(winner, rows, 0.0, 0.5) == 0.0


def test_continuous_responses_do_not_get_forced_into_two_output_states():
    rows = [Observation({"control": value}, value / 3) for value in range(5)]
    budget = SearchBudget(100)
    assert state_candidates(rows, budget=budget) == []
    assert 0 < budget.spent <= 100


def test_state_rule_improves_on_actual_old_n10_acquired_trajectories():
    root = Path(__file__).resolve().parents[1]
    frozen = root / "artifacts" / "cognition_v4_reference"
    name = "_state_search_v4_reference"
    spec = importlib.util.spec_from_file_location(name, frozen / "__init__.py",
                                                 submodule_search_locations=[str(frozen)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    environment_module = __import__(f"{name}.environments", fromlist=["make_suite"])
    environment = next(item for item in environment_module.make_suite(401, "continual") if item.spec.task_id == "n10")
    report = json.loads((root / "artifacts" / "cognition_v4_final_20260908" / "continual_runs.json").read_text())
    trace = next(item for item in report["case_traces"] if item["system"] == "cognitive_v4" and item["task_id"] == "n10")
    observed = []
    for action in trace["actions"]:
        if action["action"] == "observe" and action["round"] <= 2:
            details = action["details"]
            observed.extend(environment.observe(Probe(details["probe_id"], tuple(details["sequence"]))))
    assert len(observed) == 92
    budget = SearchBudget(900)
    candidates = state_candidates(observed, budget=budget)
    winner = min(candidates, key=lambda candidate: (loss(candidate.program, observed, 0.0, 0.5),
                                                   candidate.program.size)).program
    assert loss(winner, observed, 0.0, 0.5) == 0.0
    final_loss = loss(winner, environment.evaluation_rows(), 0.0, 0.5)
    assert final_loss < 0.02 < trace["evaluation"]["loss"]
    assert 0 < budget.spent <= 900
