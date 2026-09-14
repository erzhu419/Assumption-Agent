"""An interaction must predict new controls and change the joint experiment."""

from copy import deepcopy
from random import Random

import pytest

from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.decision_experiments import _predict, objective_pool
from hegel_machine.cognition.objective_research import (
    ObjectiveResearch, _fit, objective_program,
)


def spec():
    return TaskSpec("interaction_goal", "Learn a measured joint objective", ("x", "y", "z"), (), (),
        feature_bounds={"x": (0., 100.), "y": (0., 100.), "z": (0., 100.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0.0"})


def goal(point, interaction=True):
    x, y, z = (point[name] / 100 - center for name, center in zip(("x", "y", "z"), (.2, .7, .8)))
    return .35 + 2 * x*x + 3 * y*y + 4 * z*z + (2*x*y - 2*y*z if interaction else 0.)


def rows(*, interaction=True, repeats=1):
    rng = Random(127)
    points = [{name: rng.uniform(0., 100.) for name in ("x", "y", "z")} for _ in range(40)]
    return [{"observation": {"features": deepcopy(point), "target": -1.},
             "context": {"objective": goal(point, interaction), "global_index": index}}
            for index, point in enumerate(point for point in points for _ in range(repeats))]


def test_measured_interaction_predicts_new_controls_and_compiles_the_same_relation():
    observations = rows()
    before = deepcopy(observations)
    model = _fit(spec(), observations, SearchBudget(10000))
    assert model["interactions"] == [(0, 1), (0, 2), (1, 2)]
    assert model["interaction_comparison"]["decision"] == "supported"
    program = objective_program(spec(), model)
    for point in ({"x": 13., "y": 65., "z": 90.}, {"x": 45., "y": 18., "z": 57.}):
        assert _predict(spec(), model, point) == pytest.approx(goal(point), abs=1e-5)
        assert program.predict(Observation(point, 0.)) == pytest.approx(_predict(spec(), model, point))
    assert observations == before


def test_interaction_changes_joint_optimum_and_charges_its_solve():
    model = _fit(spec(), rows(), SearchBudget(10000))
    budget = SearchBudget(100)
    control = objective_pool(spec(), model, budget)[0]
    assert control == pytest.approx({"x": 20., "y": 70., "z": 80.}, abs=.002)
    assert goal(control) == pytest.approx(.35, abs=1e-9)
    assert budget.spent == 27


def test_additive_evidence_keeps_the_smaller_representation():
    model = _fit(spec(), rows(interaction=False), SearchBudget(10000))
    assert model["degree"] == 2 and not model.get("interactions")
    assert model["interaction_comparison"]["decision"] != "supported"


def test_replicates_are_not_independent_support_for_more_terms():
    once = _fit(spec(), rows(), SearchBudget(10000))
    repeated = _fit(spec(), rows(repeats=3), SearchBudget(30000))
    assert once["interaction_comparison"]["count"] == repeated["interaction_comparison"]["count"] == 40


def test_unidentifiable_cross_terms_do_not_replace_a_supported_additive_model():
    rng = Random(17)
    observations = []
    # At most one nonzero coordinate means all cross terms are zero. The
    # additive quadratic is identifiable, but the relational one is not.
    for index in range(45):
        point = {name: 0. for name in ("x", "y", "z")}
        point[("x", "y", "z")[index % 3]] = rng.uniform(1., 100.)
        observations.append({"observation": {"features": point, "target": -1.},
                             "context": {"objective": goal(point, False), "global_index": index}})
    model = _fit(spec(), observations, SearchBudget(10000))
    assert model is not None and not model.get("interactions")


def test_scope_collection_waits_for_controls_that_can_distinguish_interactions():
    observations = rows()
    controller = ObjectiveResearch(spec())
    controller.state["scope_start"] = 0
    ready, offers = controller._identifying_controls(observations[:12], SearchBudget(10000),
                                                    centers=observations[:12])
    assert not ready and next(offers) is not None
    ready, _ = controller._identifying_controls(observations[:13], SearchBudget(10000),
                                               centers=observations[:13])
    assert ready
    model = _fit(spec(), observations[:13], SearchBudget(10000))
    assert model.get("interactions")
