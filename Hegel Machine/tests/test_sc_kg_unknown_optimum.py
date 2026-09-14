"""Unknown interior goals retain public controls and one complete paid vector.

Deterministic truth checks and injected vectors make no simulator acquisitions.
"""

from copy import deepcopy
import importlib.util
from itertools import product
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.sc_kg_environment import (
    HegelProgramProblem, SCKGEnvironment, compile_sc_kg_program,
)
from test_cognition_sc_kg_environment import project_root


PATH = Path(__file__).resolve().parents[1] / "scripts/sc_kg_unknown_optimum.py"
SPEC = importlib.util.spec_from_file_location("unknown_optimum_test_task", PATH)
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)
PILOT_SEEDS = (1843012269, 1843112287, 1843212301)


def test_hidden_center_does_not_change_public_controls_or_native_structured_candidates(project_root):
    import numpy as np

    environment_class = task.configure(SCKGEnvironment)
    environment = environment_class(project_root, seed=PILOT_SEEDS[0])
    ordinary = SCKGEnvironment(project_root, problem="state_policy_rzdt1",
                               dimension=3, seed=PILOT_SEEDS[0], noise=.03)
    assert environment.spec == ordinary.spec
    adapter = compile_sc_kg_program(environment, Program("feature", name="x1"))
    native = compile_sc_kg_program(ordinary, Program("feature", name="x1"))
    before = adapter.structured_candidates(n=42, rng=np.random.default_rng(17))
    assert before == native.structured_candidates(n=42, rng=np.random.default_rng(17))
    public_spec = deepcopy(environment.spec)
    environment._problem._center_integer = (38, 58, 82)
    assert adapter.structured_candidates(n=42, rng=np.random.default_rng(17)) == before
    assert environment.spec == public_spec
    assert environment.costs["simulator_calls"] == adapter.simulator_calls == 0


def test_task_optima_change_remain_feasible_and_evaluator_uses_the_same_truth(project_root):
    environment_class = task.configure(SCKGEnvironment)
    from problems.single_objective import ScalarizedProblem

    centers, objective_at_old_optimum = [], []
    for seed in PILOT_SEEDS:
        environment = environment_class(project_root, seed=seed)
        evaluation = task.evaluation_environment(project_root, problem=task.PROBLEM,
            dimension=3, seed=seed + task.EVALUATION_SEED_OFFSET, noise=.03)
        adapter = task.controlled_problem(HegelProgramProblem, environment,
            Program("feature", name="x1"), {"task": {"seed": seed}})
        scalar = ScalarizedProblem(environment._problem)
        center = environment._problem._center_integer
        centers.append(center)
        assert scalar.true_objective(center) == pytest.approx(.35)
        assert scalar.is_truly_feasible(center)
        # Adjacent integer controls expose a misplaced/flat optimum immediately.
        neighbors = [tuple(x + step for x, step in zip(center, offset))
                     for offset in product((-1, 0, 1), repeat=3) if any(offset)]
        assert all(scalar.true_objective(point) > .35 for point in neighbors)
        for point in (center, (24, 70, 70), (0, 0, 0), (100, 100, 100)):
            expected = environment._problem.true_objectives(point)
            assert evaluation._problem.true_objectives(point) == expected
            assert adapter.base.base.true_objectives(point) == expected
        objective_at_old_optimum.append(scalar.true_objective((24, 70, 70)))
        assert environment.costs["simulator_calls"] == evaluation.costs["simulator_calls"] == 0
        assert adapter.simulator_calls == 0
    assert len(set(centers)) == len(PILOT_SEEDS)
    assert len(set(objective_at_old_optimum)) == len(PILOT_SEEDS)
    assert all(value > .35 for value in objective_at_old_optimum)


def test_each_paid_vector_preserves_both_optimizer_outputs_without_a_second_draw(project_root, monkeypatch):
    import numpy as np

    environment = task.configure(SCKGEnvironment)(project_root, seed=PILOT_SEEDS[0])
    calls = []

    def initial_vector(point, rng=None):
        calls.append(tuple(point))
        return np.asarray([2., 6., -.1])

    monkeypatch.setattr(environment._problem, "simulate", initial_vector)
    point = {"x1": .254, "x2": .694, "x3": .754}
    observation, = environment.observe(Probe("one_paid_vector", (point,)))
    assert calls == [(25, 69, 75)]
    assert observation.target == -.1
    assert environment.acquisitions[0]["vector_outputs"] == [[2., 6., -.1]]
    adapter = task.controlled_problem(HegelProgramProblem, environment,
        Program("feature", name="x1"), {"task": {"seed": PILOT_SEEDS[0]}})
    assert len(adapter.paid_observations) == 1
    assert adapter.paid_observations[0]["output"] == [4., -.1]
    assert adapter.research_simulator_calls == 1 and adapter.simulator_calls == 0

    def online_vector(point, rng=None):
        calls.append(tuple(point))
        return np.asarray([7., 11., -.2])

    monkeypatch.setattr(adapter.base.base, "simulate", online_vector)
    actual = adapter.simulate((26, 68, 74), np.random.default_rng(31))
    np.testing.assert_array_equal(actual, [9., -.2])
    assert calls == [(25, 69, 75), (26, 68, 74)]
    assert adapter.simulation_log[0]["vector_outputs"] == [7., 11., -.2]
    assert adapter.simulation_log[0]["output"] == [9., -.2]
    assert environment.costs["simulator_calls"] == adapter.simulator_calls == 1
