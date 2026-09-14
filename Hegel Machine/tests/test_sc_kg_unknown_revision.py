"""Controlled change clocks and interacting objectives, without physical draws."""

from itertools import product
import json
from pathlib import Path
import sys

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.sc_kg_environment import HegelProgramProblem, SCKGEnvironment
from test_cognition_sc_kg_environment import project_root


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sc_kg_unknown_revision as task


def _case(profile, seed):
    return {"profile": profile,
        "task": {"problem": "stable_interaction" if profile == "stable_interaction" else "initial_separable",
                 "dimension": 3, "seed": seed, "noise": .03},
        "next_problem": profile, "change_after_new_calls": 8}


def test_native_and_hegel_change_only_after_eight_online_vectors_with_the_same_paid_prefix(project_root, monkeypatch):
    import numpy as np
    import run_sc_kg_unknown_revision_study as runner

    spec = json.loads((ROOT / "artifacts/sc_kg_unknown_optimum_pilot_20260911/protocol.json").read_text())
    environment_class = task.configure(SCKGEnvironment)
    # Load the actual backend before replacing its stochastic draw with a spy.
    environment_class(project_root, seed=runner.SEEDS[0])
    from problems.rzdt import TestProblem

    draws = []

    def vector_spy(problem, point, rng=None):
        draws.append(tuple(point))
        # Unequal first two outputs reveal dropped/reconstructed objective labels.
        return np.asarray(problem.true_objectives(point)) + [.11, -.07, .013]

    monkeypatch.setattr(TestProblem, "simulate", vector_spy)
    for profile in task.PROFILES:
        case = _case(profile, runner.SEEDS[0])
        environment = environment_class(project_root, **case["task"])
        controls = tuple(point for probe in (*environment.spec.initial_probes,
                         *environment.spec.available_probes[:2]) for point in probe.sequence)
        before = len(draws)
        environment.observe(Probe("shared_paid32", controls))
        assert len(draws) == before + 32
        acquisition = environment.acquisitions[0]
        records = [{"point": point, "output": [.5 * vector[0] + .5 * vector[1], vector[2]]}
                   for point, vector in zip(acquisition["integer_points"], acquisition["vector_outputs"])]
        hegel = task.controlled_problem(HegelProgramProblem, environment,
                                        Program("feature", name="x1"), case)
        _, native = runner.build_native_optimizer(spec, case, records)
        assert len(draws) == before + 32
        assert hegel.research_simulator_calls == native.global_calls == 32
        assert hegel.simulator_calls == native.actual_simulator_calls == 0
        assert [row["output"] for row in hegel.paid_observations] == [row["output"] for row in records]
        point = hegel.base.base._center_integer
        old = np.asarray(hegel.base.base.true_objectives(point)) + [.11, -.07, .013]
        changed = np.asarray(hegel.next_base.base.true_objectives(point)) + [.11, -.07, .013]
        for index in range(1, 10):
            expected_vector = changed if index > 8 and profile != "stable_interaction" else old
            expected = [.5 * expected_vector[0] + .5 * expected_vector[1], expected_vector[2]]
            np.testing.assert_allclose(hegel.simulate(point), expected, atol=1e-14)
            np.testing.assert_allclose(native.simulate(point), expected, atol=1e-14)
            np.testing.assert_array_equal(hegel.simulation_log[-1]["vector_outputs"],
                                          native.simulation_log[-1]["vector_outputs"])
            assert hegel.simulator_calls == native.actual_simulator_calls == index
            assert native.global_calls == 32 + index
            assert len(draws) == before + 32 + 2 * index
        assert environment.costs["simulator_calls"] == 32
        if profile != "stable_interaction":
            assert old[0] != changed[0]
            assert hegel.simulation_log[7]["mechanism_phase"] == "initial"
            assert hegel.simulation_log[8]["mechanism_phase"] == "changed"
        else:
            assert all(row["mechanism_phase"] == "initial" for row in hegel.simulation_log)


def test_final_evaluation_matches_each_task_and_preserves_its_constraint(project_root):
    import run_sc_kg_unknown_revision_study as runner

    environment_class = task.configure(SCKGEnvironment)
    for profile, seed in product(task.PROFILES, runner.SEEDS):
        case = _case(profile, seed)
        environment = environment_class(project_root, **case["task"])
        from problems.single_objective import ScalarizedProblem
        adapter = task.controlled_problem(HegelProgramProblem, environment,
                                          Program("feature", name="x1"), case)
        evaluation = task.evaluation_environment(project_root, problem=profile,
            dimension=3, seed=seed + task.EVALUATION_SEED_OFFSET, noise=.03)
        initial, final = adapter.base.base, adapter.next_base.base
        center = final._center_integer
        assert ScalarizedProblem(final).true_objective(center) == pytest.approx(.35)
        assert ScalarizedProblem(final).is_truly_feasible(center)
        for point in (initial._center_integer, center, (24, 70, 70), (100, 100, 100)):
            assert evaluation._problem.true_objectives(point) == final.true_objectives(point)
            assert initial.true_objectives(point)[2] == final.true_objectives(point)[2]
        if profile != "stable_interaction":
            assert center != initial._center_integer
            assert final.true_objectives(initial._center_integer)[0] > .35
        assert environment.costs["simulator_calls"] == evaluation.costs["simulator_calls"] == 0
        assert adapter.simulator_calls == 0


def test_interaction_is_present_in_the_executed_quadratic_and_the_optimum_is_interior(project_root):
    import numpy as np
    import run_sc_kg_unknown_revision_study as runner

    environment_class = task.configure(SCKGEnvironment)
    for kind in ("stable_interaction", "center_interaction_change"):
        environment = environment_class(project_root, problem=kind, seed=runner.SEEDS[0])
        problem = environment._problem
        center = np.asarray(problem._center_integer)
        assert np.all(center > 0) and np.all(center < 100)

        def value(offset):
            return problem.true_objectives(tuple(center + offset))[0]

        step = np.eye(3, dtype=int)
        q = np.zeros((3, 3))
        for i in range(3):
            q[i, i] = (value(step[i]) - 2 * value(np.zeros(3, dtype=int))
                        + value(-step[i])) / (2 * .01 ** 2)
            for j in range(i):
                q[i, j] = q[j, i] = (value(step[i] + step[j]) - value(step[i] - step[j])
                    - value(-step[i] + step[j]) + value(-step[i] - step[j])) / (8 * .01 ** 2)
        assert np.min(np.linalg.eigvalsh(q)) > 0.
        assert np.all(np.abs(q[np.triu_indices(3, 1)]) > .1)
        # Check the recovered interaction beyond the finite-difference neighborhood.
        for offset in (np.asarray([7, -11, 13]), np.asarray([-9, 15, -14])):
            delta = offset / 100.
            assert value(offset) == pytest.approx(.35 + float(delta @ q @ delta))
            assert value(offset) > .35
        assert environment.costs["simulator_calls"] == 0
