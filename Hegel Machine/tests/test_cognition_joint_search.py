"""Joint search recovers sparse laws from misleading early representations."""

import importlib.util
import json
from pathlib import Path
from random import Random
import sys

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.sparse_search import sparse_candidates


@pytest.fixture(scope="module")
def captured_development_cases():
    # Replay the exact acquired v0.5.0 trajectories. The frozen simulator keeps
    # this regression independent of any subsequently introduced environment.
    root = Path(__file__).resolve().parents[1]
    frozen = root / "artifacts" / "cognition_v3_reference"
    name = "_joint_search_v3_reference"
    spec = importlib.util.spec_from_file_location(name, frozen / "__init__.py",
                                                 submodule_search_locations=[str(frozen)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    module = __import__(f"{name}.environments", fromlist=["make_suite"])
    environments = module.make_suite(101, "forward")
    report = json.loads((root / "artifacts" / "cognition_v3_final_20260908" / "forward_runs.json").read_text())
    result = {}
    for task_id in ("f03", "f09"):
        environment = next(item for item in environments if item.spec.task_id == task_id)
        trace = next(item for item in report["case_traces"]
                     if item["system"] == "cognitive_v3" and item["task_id"] == task_id)
        rounds = []
        for index in (0, 1):
            observed = []
            for action in trace["actions"]:
                if action["action"] == "observe" and action["round"] <= index:
                    details = action["details"]
                    observed.extend(environment.observe(Probe(details["probe_id"], tuple(details["sequence"]))))
            rounds.append(observed)
        result[task_id] = (rounds, list(environment.evaluation_rows()))
    return result


@pytest.mark.parametrize("round_index, expected_rows", ((0, 23), (1, 37)))
def test_joint_search_recovers_four_terms_from_actual_f09_acquired_trajectories(
        captured_development_cases, round_index, expected_rows):
    rounds, final_rows = captured_development_cases["f09"]
    training = rounds[round_index]
    assert len(training) == expected_rows
    common_training = [row for row in training if len(row.history) >= 4]
    fresh = [row for row in final_rows if len(row.history) >= 4]
    winners = {}
    for strategy in ("greedy", "beam"):
        budget = SearchBudget(2200)
        candidates = sparse_candidates(training, budget=budget, max_history=4, strategy=strategy)
        assert candidates and 0 < budget.spent <= 2200
        # Candidate choice uses only acquired evidence, on the same rows for
        # every history representation; the final rows are scored afterwards.
        winners[strategy] = min(candidates, key=lambda item: loss(item.program, common_training, 0.0, 0.5)).program
    assert winners["beam"].history_depth == 2
    assert loss(winners["beam"], fresh, 0.0, 0.5) < 1e-12
    assert loss(winners["greedy"], fresh, 0.0, 0.5) > 0.1


@pytest.mark.parametrize("max_history", (0, 4))
def test_joint_refit_removes_collinear_alias_after_actual_f03_queries(captured_development_cases, max_history):
    rounds, fresh = captured_development_cases["f03"]
    training = rounds[0]
    assert len(training) == 23
    assert all(row.features["x"] == row.features["z"] for row in training[:9])
    assert any(row.features["x"] != row.features["z"] for row in training[9:])
    budget = SearchBudget(2200)
    candidates = sparse_candidates(training, budget=budget, max_history=max_history)
    current = [candidate for candidate in candidates if candidate.program.history_depth == 0]
    winner = min(current, key=lambda item: loss(item.program, training, 0.0, 0.5)).program
    assert winner.observed_variables() == {("x", 0)}
    assert loss(winner, fresh, 0.0, 0.5) < 1e-12
    assert 0 < budget.spent <= 2200


def test_support_exchanges_combine_two_distinct_historical_coordinates():
    def rows(seed, count):
        rng, result = Random(seed), []
        for _ in range(count):
            history = tuple({"x": rng.uniform(-1.5, 1.5), "z": rng.uniform(-1.5, 1.5)} for _ in range(2))
            x, z = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            target = 0.4 * x + 1.1 * history[-1]["x"] - 1.2 * history[-2]["z"] + 0.6 * x * z
            result.append(Observation({"x": x, "z": z}, target, history))
        return result

    training, fresh = rows(31, 40), rows(43, 24)
    budget = SearchBudget(6000)
    candidates = sparse_candidates(training, budget=budget, max_history=2)
    winner = min(candidates, key=lambda item: loss(item.program, training, 0.0, 0.5)).program
    # The joint initialization contains at most one historical coordinate.
    # Both lags in the result therefore require a subsequent support revision.
    assert {("x", 1), ("z", 2)} <= winner.observed_variables()
    assert loss(winner, fresh, 0.0, 0.5) < 1e-12
    assert 0 < budget.spent <= 6000
