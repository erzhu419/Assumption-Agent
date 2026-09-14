"""A deployed goal explanation can lose to a GP on later, paid evidence."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from test_cognition_sc_kg_environment import project_root, _saved_terminal_failure_optimizer


CASE = "objective_reentry_1_mechanism_change_noise_0.03"


@pytest.fixture
def installed_goal(project_root, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "artifacts/sc_kg_goal_completion_20260911/runs"
        / f"{CASE}_hegel_goal_completion.json").read_text())
    algorithm, adapter, environment = _saved_terminal_failure_optimizer(
        project_root, monkeypatch, CASE, frozen_directory="sc_kg_objective_reentry_v052_20260911")
    monkeypatch.setattr(adapter, "simulate", lambda *args, **kwargs:
                        pytest.fail("this test reuses existing simulator vectors"))
    monkeypatch.setattr(environment, "observe", lambda *args, **kwargs:
                        pytest.fail("this test needs no new research observation"))
    algorithm._fit_initial_belief(algorithm._initial_samples())
    history = [(tuple(point), np.asarray(output)) for point, output in
               zip(saved["initial_points"], saved["initial_outputs"])]
    history.extend((tuple(row["point"]), np.asarray(row["output"])) for row in saved["simulations"])
    algorithm.history = deepcopy(history[:54])
    algorithm.observations = {}
    for point, output in algorithm.history:
        algorithm.observations.setdefault(point, []).append(output)
    revision = saved["objective_scope_revisions"][0]
    algorithm.objective_research.state["accepted_revision"] = deepcopy(revision)
    algorithm._rebase_objective_scope(revision, 54)
    algorithm.recovery.state["phase"] = "monitoring"
    return algorithm, adapter, history


def _consume(algorithm, point, output):
    index = len(algorithm.history)
    features = dict(zip(algorithm.objective_research.spec.feature_names,
                        (value / 100. for value in point)))
    algorithm.objective_replacement.observe(features, float(output[0]), global_index=index)
    noise = algorithm.variance_model.predict_variance(0, point, algorithm.problem)
    algorithm.gpr[0].update(point, float(output[0]), noise)
    algorithm.history.append((point, output.copy()))
    algorithm.observations.setdefault(point, []).append(output.copy())
    algorithm._consider_objective_replacement(index + 1)


def test_installed_model_waits_for_actual_counterexamples_before_reopening(installed_goal):
    algorithm, adapter, history = installed_goal
    objective = algorithm.gpr[0]
    algorithm._prepare_objective_replacement(history[54][0])
    assert algorithm.objective_replacement.state["scope_start"] is None
    assert algorithm.gpr[0] is objective and objective.basis_map is not None
    assert adapter.simulator_calls == 0


def test_saved_later_vectors_replace_failed_program_with_an_independent_raw_basis(installed_goal):
    algorithm, adapter, history = installed_goal
    objective = algorithm.gpr[0]
    constraint = algorithm.gpr[1]
    constraint_parameters = constraint.a.copy(), constraint.C.copy()
    comparisons = []
    for point, output in history[54:]:
        algorithm.objective_research._monitor(algorithm._objective_evidence())
        index = len(algorithm.history)
        algorithm._prepare_objective_replacement(point)
        pending = deepcopy(algorithm.objective_replacement.state["pending"])
        if pending is not None:
            assert index >= 55 and pending["stage"] == index
            prefix, observations = algorithm._objective_prefix(44)
            raw = algorithm._fit_objective_model(prefix, observations)
            assert raw.basis_map is None and objective.basis_map is not None
            assert pending["replacement_mean"] == pytest.approx(raw.posterior_mean(point))
            comparisons.append(pending)
        _consume(algorithm, point, output)
    assert len(comparisons) >= 4
    assert algorithm.gpr[0] is not objective and algorithm.gpr[0].basis_map is None
    assert algorithm.active_objective_scope_start == 44
    attempt = algorithm.objective_replacement.state["attempts"][-1]
    assert attempt["reason"] == "supported_gp_replacement"
    assert min(row["global_index"] for row in attempt["validation"]) == 55
    assert algorithm.gpr[1] is constraint
    np.testing.assert_array_equal(constraint.a, constraint_parameters[0])
    np.testing.assert_array_equal(constraint.C, constraint_parameters[1])
    assert len(algorithm.history) == 64 and adapter.simulator_calls == 0


def test_failed_raw_comparison_preserves_the_installed_program(installed_goal):
    algorithm, adapter, history = installed_goal
    goal = algorithm.objective_research
    goal.state["scope_research"] = {"scope_start": 54, "phase": "collecting",
        "trigger": [{"global_index": 54}], "candidate": None, "validation": []}
    objective = algorithm.gpr[0]
    for point, _ in history[54:58]:
        algorithm._prepare_objective_replacement(point)
        pending = algorithm.objective_replacement.state["pending"]
        output = np.asarray([pending["incumbent_mean"], -.2])
        _consume(algorithm, point, output)
    assert algorithm.gpr[0] is objective
    assert algorithm.objective_replacement.state["decisions"][-1]["replace"] is False
    assert adapter.simulator_calls == 0
