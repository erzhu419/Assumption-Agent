"""Retain the rejected joint experiment and test live shared goal validation.

These are synthetic mechanism labels, not new SC-KG simulator calls. The
joint acquisition tests explicitly load frozen revision 5, which was not
adopted after its outcome comparison. Shared validation and compiled programs
exercise the production class. The identification question is explicit:
single-axis evidence alone cannot reveal its missing interaction.
"""

from copy import deepcopy
import importlib.util
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import pytest

from hegel_machine.cognition.decision_experiments import _design, _predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.objective_research import ObjectiveResearch, _fit


NAMES = ("x1", "x2", "x3")
PAIRS = tuple(combinations(range(3), 2))
CENTER = dict.fromkeys(NAMES, .5)


def spec():
    return TaskSpec("joint-identification", "Identify a questioned interaction",
        NAMES, (), (), feature_bounds={name: (0., 1.) for name in NAMES},
        context={"objective_direction": "minimize", "constraint_upper_bound": "0"})


def additive(point):
    x, y, z = (point[name] - CENTER[name] for name in NAMES)
    return .35 + 2*x*x + 3*y*y + 4*z*z


def mechanism(point):
    x, y, z = (point[name] - CENTER[name] for name in NAMES)
    return additive(point) + 2*x*y - 2*y*z


def row(point, index, *, target=mechanism):
    return {"observation": {"features": deepcopy(point), "target": -1.},
            "context": {"global_index": index, "objective": target(point)}}


def axes(values=(0., .25, .75, 1.)):
    points = [CENTER, *({**CENTER, name: value} for name in NAMES for value in values)]
    return [row(point, index) for index, point in enumerate(points)]


def controller(research_class=ObjectiveResearch):
    research = research_class(spec())
    research.state.update(scope_start=0, evidence_start=0)
    return research


@pytest.fixture(scope="module")
def frozen_joint_class():
    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_joint_identification_20260911/revision5/"
        "method_source/objective_research.py")
    module_spec = importlib.util.spec_from_file_location(
        "hegel_machine.cognition._joint_identification_revision5_tests", source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.ObjectiveResearch


def grouped_ranks(rows, interactions):
    """Independently check held-out whole controls, including paid replicates."""
    keys = [tuple(item["observation"]["features"][name] for name in NAMES) for item in rows]
    groups = {key: index % 5 for index, key in enumerate(dict.fromkeys(keys))}
    design = np.asarray([_design(spec(), item["observation"]["features"], 2, interactions)
                         for item in rows])
    return [np.linalg.matrix_rank(design[[groups[key] != fold for key in keys]])
            for fold in sorted(set(groups.values()))]


def test_frozen_joint_query_separates_explanations_identical_on_all_paid_axis_controls(
        frozen_joint_class):
    rows = axes()
    before = deepcopy(rows)

    def interaction(point):
        return additive(point) + (point["x1"] - .5) * (point["x2"] - .5)

    assert all(additive(item["observation"]["features"]) ==
               interaction(item["observation"]["features"]) for item in rows)
    # The normal remaining planning budget must reach a distinguishing query;
    # spending it again on the same axis is unable to answer this question.
    budget = SearchBudget(1500)
    ready, controls = controller(frozen_joint_class)._identifying_controls(rows, budget,
        centers=rows, interactions=((0, 1),))
    point = next(iter(controls), None)
    assert not ready and point is not None
    assert interaction(point) != pytest.approx(additive(point))
    assert point not in [item["observation"]["features"] for item in rows]
    assert all(0. <= value <= 1. for value in point.values())
    assert 0 < budget.spent <= budget.limit
    assert rows == before


def test_frozen_joint_identification_supplies_a_goal_for_live_future_validation(
        frozen_joint_class):
    rows = axes()
    original_count = len(rows)
    budget = SearchBudget(20000)
    previous = _fit(spec(), rows, budget)
    assert previous is not None and not previous.get("interactions")
    assert previous["cv_loss"] < 1e-12
    experimental = controller(frozen_joint_class)
    for _ in range(32 - 4):
        ready, controls = experimental._identifying_controls(rows, budget,
            centers=rows, interactions=PAIRS)
        if ready:
            break
        point = next(iter(controls), None)
        assert point is not None, "the remaining budget must buy an identifying control"
        assert point not in [item["observation"]["features"] for item in rows]
        rows.append(row(point, len(rows)))
    else:
        pytest.fail("joint identification consumed the four future validation calls")
    assert grouped_ranks(rows, PAIRS) == [10] * 5
    # Only acquisition comes from the rejected experiment. The live validator
    # must still qualify an explanation supplied by those observations.
    research = controller()
    research.state["scope_research"] = {
        "scope_start": 0, "phase": "collecting", "candidate": None,
        "trigger": [{"threshold": 9e-7}], "validation": [], "coverage_calls": 0}
    assessment = research._collect_candidate(rows, budget,
        remaining_calls=32 - (len(rows) - original_count))
    research.state["compute_cost"] = budget.spent
    candidate = deepcopy(research.state["scope_research"]["candidate"])
    assert assessment["ready"] and candidate["interactions"]
    assert candidate["interaction_comparison"]["decision"] == "supported"
    assert candidate["cv_loss"] < 1e-12
    assert research.accepted_revision is None
    training_indices = [item["context"]["global_index"] for item in rows]

    def forecast(point):
        return {"features": point, "objective_mean": _predict(spec(), previous, point),
                "objective_noise": 0.}

    # Four independent future controls share their complete goal/constraint
    # vector. Their labels are formed only after both forecasts were recorded.
    future = [{"x1": x, "x2": y, "x3": .5} for x, y in product((.25, .75), repeat=2)]
    for count, point in enumerate(future):
        index = len(rows)
        request = research.prepare_shared_control(rows, forecast, point,
            scope_start=0, stage=index, remaining_calls=32-(index-original_count),
            purpose="optimization")
        assert request is not None and research.accepted_revision is None
        assert request["expected_global_index"] == index
        assert research.state["scope_research"]["candidate"] == candidate
        target = mechanism(point)
        assert request["candidate_mean"] == pytest.approx(target, abs=1e-6)
        research.observe(point, [target, -1.], context={"global_index": index,
            "stage": index + 1, "purpose": "optimization"})
        rows.append(row(point, index))
        if count < 3:
            assert research.accepted_revision is None
    accepted = research.accepted_revision
    assert accepted is not None and accepted["model"] == candidate
    assert accepted["validation_mse"] < 1e-12
    assert accepted["previous_mse"] > .01
    assert accepted["training_indices"] == training_indices
    assert len(accepted["validation_indices"]) == 4
    assert set(accepted["validation_indices"]).isdisjoint(training_indices)
    assert len(rows) - original_count <= 32
    assert research.state["compute_cost"] <= 20000
    assert len(research.state["shared_outcomes"]) == 4


def test_additive_identification_keeps_the_single_axis_path():
    rows = axes((0., 1.))
    budget = SearchBudget(20000)
    research = controller()
    for _ in range(28):
        ready, controls = research._identifying_controls(rows, budget,
            centers=rows, interactions=())
        if ready:
            break
        point = next(iter(controls), None)
        assert point is not None
        assert sum(point[name] != CENTER[name] for name in NAMES) == 1
        rows.append(row(point, len(rows), target=additive))
    else:
        pytest.fail("the existing additive identification no longer completes")
    assert grouped_ranks(rows, ()) == [7] * 5
    assert budget.spent <= 20000


def test_frozen_joint_identification_stops_when_no_candidate_pool_can_be_priced(
        frozen_joint_class):
    rows = axes()
    before = deepcopy(rows)
    # Cover the existing-data assessment but no unobserved control evaluation.
    budget = SearchBudget(len(rows) * 10)
    ready, controls = controller(frozen_joint_class)._identifying_controls(rows, budget,
        centers=rows, interactions=PAIRS)
    assert not ready and list(controls) == []
    assert budget.remaining == 0 and rows == before


@pytest.fixture
def saved_inconclusive_interaction():
    path = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_joint_identification_20260911/runs/"
        "goal_completion_2_center_interaction_change_hegel_joint_identification.json")
    record = json.loads(path.read_text())
    rows = [{"observation": {"features": dict(zip(NAMES,
                (value / 100 for value in item["point"]))), "target": item["output"][1]},
             "context": {"global_index": 31 + item["new_call"],
                         "objective": item["output"][0]}}
            for item in record["simulations"] if 40 <= 31 + item["new_call"] <= 52]
    previous = _fit(spec(), rows, SearchBudget(20000))
    research = controller()
    research.state["model"] = deepcopy(previous)
    research.state["scope_research"] = {"scope_start": 40, "phase": "collecting",
        "candidate": None, "validation": [], "coverage_calls": 0,
        "trigger": deepcopy(record["objective_research"]["scope_attempts"][0]["trigger"])}
    return rows, previous, research


def test_saved_positive_inconclusive_interaction_survives_without_changing_predictor(
        saved_inconclusive_interaction):
    rows, previous, _ = saved_inconclusive_interaction
    challenger = previous["interaction_challenger"]
    assert len(rows) == 13 and grouped_ranks(rows, PAIRS) == [10] * 5
    assert not previous.get("interactions") and challenger["interactions"]
    assert previous["cv_loss"] == pytest.approx(.022837, abs=1e-6)
    assert challenger["cv_loss"] == pytest.approx(.005377, abs=1e-6)
    assert challenger["interaction_comparison"]["decision"] == "inconclusive"
    assert challenger["interaction_comparison"]["mean_gain"] > 0
    assert len(challenger["folds"]) == 5
    # Retaining another explanation must not silently change current forecasts.
    without_challenger = {key: value for key, value in previous.items()
                          if key != "interaction_challenger"}
    point = {"x1": .25, "x2": .75, "x3": .5}
    assert _predict(spec(), previous, point) == _predict(spec(), without_challenger, point)
    assert _predict(spec(), challenger, point) != pytest.approx(_predict(spec(), previous, point))


def test_saved_challenger_predicts_before_the_same_future_experiment_is_observed(
        saved_inconclusive_interaction):
    rows, previous, research = saved_inconclusive_interaction
    budget = SearchBudget(20000)
    result = research._collect_candidate(rows, budget, remaining_calls=11)
    candidate = research.state["scope_research"]["candidate"]
    assert result["ready"] and not candidate.get("interactions")
    assert candidate["coefficients"] == previous["coefficients"]
    assert candidate["interaction_challenger"] == previous["interaction_challenger"]
    assert candidate["training_end"] == 52
    assert research.state["scope_research"]["phase"] == "validating"
    assert research.state["scope_research"]["validation"] == []
    assert research.accepted_revision is None and research.state["model"] == previous
    assert budget.remaining >= result["validation_reserve"]
    research.state["compute_cost"] = budget.spent
    frozen, paid_prefix = deepcopy(candidate), deepcopy(rows)
    forecasts = []

    def forecast(point):
        assert research.state["scope_research"]["validation"] == []
        assert research.state["outcomes"] == []
        forecasts.append(deepcopy(point))
        return {"features": point, "feasible": True,
            "objective_mean": _predict(spec(), previous, point) + 1., "objective_noise": 0.}

    offered = research.next_control(rows, forecast, scope_start=0, stage=53, remaining_calls=11)
    assert offered is not None and offered["features"] in forecasts
    assert offered["candidate_mean"] == pytest.approx(
        _predict(spec(), frozen, offered["features"]))
    assert offered["previous_mean"] == pytest.approx(
        _predict(spec(), previous, offered["features"]) + 1.)
    assert offered["reason"] == "validate_objective_scope"
    assert offered["interaction_mean"] == pytest.approx(
        _predict(spec(), frozen["interaction_challenger"], offered["features"]))
    assert research.state["scope_research"]["candidate"] == frozen
    assert budget.spent < research.state["compute_cost"] <= budget.limit
    assert research.accepted_revision is None and research.state["scope_research"]["validation"] == []
    assert research.state["outcomes"] == [] and rows == paid_prefix

    # Produce the synthetic future label only after all three frozen forecasts
    # exist. One paid experiment supplies both candidate errors at the same point.
    target = offered["interaction_mean"] + .25
    research.observe(offered["features"], [target, -1.], context={"global_index": 53,
        "stage": 54, "purpose": offered["purpose"]})
    validation = research.state["scope_research"]["validation"]
    assert len(validation) == 1 and len(research.state["outcomes"]) == 1
    assert validation[0]["features"] == offered["features"]
    assert validation[0]["global_index"] == 53 and validation[0]["target"] == target
    for error, prediction in (("candidate_error", "candidate_mean"),
                              ("interaction_error", "interaction_mean"),
                              ("previous_error", "previous_mean")):
        assert validation[0][error] == pytest.approx((target - offered[prediction]) ** 2)
    assert research.state["pending"] is None
    assert research.state["scope_research"]["candidate"] == frozen
    assert research.accepted_revision is None and research.state["model"] == previous
    assert research.state["compute_cost"] <= budget.limit and rows == paid_prefix


@pytest.mark.parametrize("correct_model", ["interaction", "base"])
def test_shared_future_vectors_choose_the_supported_goal_explanation(
        saved_inconclusive_interaction, correct_model):
    rows, previous, research = saved_inconclusive_interaction
    budget = SearchBudget(20000)
    research._collect_candidate(rows, budget, remaining_calls=11)
    research.state["compute_cost"] = budget.spent
    frozen = deepcopy(research.state["scope_research"]["candidate"])
    challenger = frozen["interaction_challenger"]

    def forecast(point):
        # A deployed GP biased relative to both fitted explanations lets their
        # common future evidence distinguish representation choice from adoption.
        return {"features": point, "objective_mean": _predict(spec(), previous, point) + 1.,
                "objective_noise": 0.}

    # Both possible futures use the same public controls and fixed proposals.
    # The label is produced only after the GP, base, and full forecasts exist.
    future = [{"x1": x, "x2": y, "x3": .5} for x, y in product((.24, .26), (.74, .76))]
    generating_model = challenger if correct_model == "interaction" else frozen
    for count, point in enumerate(future):
        index = rows[-1]["context"]["global_index"] + 1
        spent = research.state["compute_cost"]
        request = research.prepare_shared_control(rows, forecast, point,
            scope_start=0, stage=index, remaining_calls=64-index, purpose="optimization")
        assert request is not None
        assert research.state["compute_cost"] == spent + 2
        assert research.state["scope_research"]["candidate"] == frozen
        assert request["interaction_mean"] == pytest.approx(_predict(spec(), challenger, point))
        assert research.accepted_revision is None
        target = _predict(spec(), generating_model, point)
        research.observe(point, [target, -1.], context={"global_index": index,
            "stage": index + 1, "purpose": "optimization"})
        rows.append(row(point, index, target=lambda unused: target))
        if count < 3:
            assert research.accepted_revision is None
    accepted = research.accepted_revision
    assert accepted is not None and research.state["model"] == previous
    assert accepted["validation_indices"] == [53, 54, 55, 56]
    assert accepted["training_indices"] == list(range(40, 53))
    assert accepted["validation_mse"] == pytest.approx(0., abs=1e-15)
    assert bool(accepted["model"].get("interactions")) == (correct_model == "interaction")
    assert accepted["model"]["coefficients"] == generating_model["coefficients"]
    compiled = Program.from_dict(accepted["program"])
    for point in [*future, {"x1": .1, "x2": .6, "x3": .8}]:
        assert compiled.predict(Observation(point, 0.)) == pytest.approx(
            _predict(spec(), generating_model, point))
    assert len(research.state["shared_outcomes"]) == 4
    assert research.state["compute_cost"] <= 20000
