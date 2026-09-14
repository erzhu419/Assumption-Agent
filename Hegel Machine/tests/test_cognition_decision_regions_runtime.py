"""Retain the regional and final-call experiments under their frozen methods."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from hegel_machine.cognition.decision_experiments import _predict
from hegel_machine.cognition.models import SearchBudget, TaskSpec


@pytest.fixture(scope="module")
def frozen_region_modules():
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_decision_regions_20260911/revision4/method_source")

    def load(name):
        module_spec = importlib.util.spec_from_file_location(
            f"hegel_machine.cognition._decision_regions_revision4_{name}", source / f"{name}.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    regions, reentry, research = (load(name) for name in (
        "objective_regions", "objective_reentry", "objective_research"))
    reentry.propose_decision_regions = regions.propose_decision_regions
    research.propose_decision_regions = regions.propose_decision_regions
    research.select_reentry_control = reentry.select_reentry_control
    research.fit_repaired_objective = reentry.fit_repaired_objective
    return research, reentry


@pytest.fixture(scope="module")
def frozen_tail_research():
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_decision_regions_20260911/revision5/method_source/objective_research.py")
    module_spec = importlib.util.spec_from_file_location(
        "hegel_machine.cognition._decision_regions_revision5_objective_research", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.ObjectiveResearch


def setting():
    spec = TaskSpec("regional-goal", "Choose an informative useful control", ("x1", "x2"), (), (),
        feature_bounds={"x1": (0., 1.), "x2": (0., 1.)},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0"})
    model = {"degree": 2, "coefficients": [.85, -1., -1., 1., 1.],
             "cv_loss": .001, "reference_loss": 1., "training_end": 9,
             "training_rows": 10}
    model["folds"] = [deepcopy(model) for _ in range(5)]
    research = {"scope_start": 0, "phase": "validating", "candidate": model,
                "validation": [], "validation_target": 4,
                "training_indices": list(range(10)), "coverage_calls": 0}
    return spec, research


def forecast(point):
    actual = {name: round(value, 2) for name, value in point.items()}
    distance = sum((value - .5) ** 2 for value in actual.values())
    return {"features": actual, "objective_mean": .35 + distance + 1. + 10*distance,
            "objective_noise": .001, "feasible": True}


def test_frozen_active_validation_uses_future_controls_near_the_predicted_action(frozen_region_modules):
    spec, state = setting()
    goal = frozen_region_modules[0].ObjectiveResearch(spec)
    goal.state.update(scope_start=0, evidence_start=0, scope_research=state)
    frozen = deepcopy(state["candidate"])
    rows = []
    for stage in range(10, 14):
        request = goal._research_scope(rows, forecast, stage, 20-stage)
        assert request is not None and request["reason"] == "validate_objective_scope"
        point = request["features"]
        assert all(abs(value - .5) <= .13 for value in point.values())
        assert goal.accepted_revision is None
        assert state["candidate"] == frozen
        target = _predict(spec, frozen, point)
        assert request["candidate_mean"] == pytest.approx(target)
        context = {"global_index": stage, "stage": stage+1,
                   "purpose": request["purpose"], "objective": target}
        goal.observe(point, [target, -1.], context=context)
        rows.append({"observation": {"features": point, "target": -1.}, "context": context})
    assert goal.accepted_revision["validation_indices"] == [10, 11, 12, 13]
    assert goal.accepted_revision["model"] == frozen
    assert goal.state["compute_cost"] <= 20000


def test_frozen_reentry_uses_low_value_region_even_with_larger_remote_disagreement(frozen_region_modules):
    spec, research = setting()
    budget = SearchBudget(1000)
    selected = frozen_region_modules[1].select_reentry_control(spec, research, [], forecast, budget)
    assert selected is not None
    assert all(abs(value - .5) <= .13 for value in selected["features"].values())
    remote = {"x1": 0., "x2": 0.}
    assert (forecast(remote)["objective_mean"] - _predict(spec, research["candidate"], remote)
            > abs(selected["objective_mean"] - selected["candidate_mean"]))
    assert 0 < budget.spent <= budget.limit


def test_frozen_unfinished_scope_uses_last_calls_to_retest_old_recommendation_then_new_winner(
        frozen_tail_research):
    spec, research = setting()
    goal = frozen_tail_research(spec)
    old = {"x1": .2, "x2": .2}
    new = {"x1": .7, "x2": .7}
    rows = [{"observation": {"features": old, "target": -1.},
             "context": {"global_index": 8, "objective": .1}},
            {"observation": {"features": new, "target": -1.},
             "context": {"global_index": 10, "objective": .3}}]
    research.update(scope_start=11, phase="collecting", candidate=None)
    goal.state.update(scope_start=0, evidence_start=0, scope_research=research,
                      scope_attempts=[{"scope_start": 10}])

    def current_prediction(point):
        return {"features": point, "objective_mean": .1 if point == old else .3,
                "objective_noise": .01, "feasible": True}

    for index, expected in ((11, old), (12, old), (13, new)):
        request = goal.next_control(rows, current_prediction,
            scope_start=0, stage=index, remaining_calls=14-index)
        assert request["features"] == expected
        assert request["reason"] == "resolve_current_goal_incumbent"
        assert request["decision_scope"] == 10
        assert goal.accepted_revision is None
        target = .7 if expected == old else .3
        context = {"global_index": index, "stage": index+1,
                   "purpose": request["purpose"], "objective": target}
        goal.observe(expected, [target, -1.], context=context)
        rows.append({"observation": {"features": expected, "target": -1.}, "context": context})
    assert goal.state["compute_cost"] <= 20000
