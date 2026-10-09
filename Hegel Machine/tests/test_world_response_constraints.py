"""Binary activation constrains a response interval, not an equality label."""

from copy import deepcopy
from math import isclose

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.world_response_constraints import fit_response_constraints
from hegel_machine.cognition.world_response_hypotheses import propose_response_research


X = Program("feature", name="measurement")
ESTIMATORS = [{"id": "linear", "basis": [X.to_dict()]}]


def response(number, setting, activated):
    return {"crystal": number, "frequency": setting, "activated": activated,
            "evidence": [number]}


def assert_respects_evidence(result, features, responses, target, grid):
    """Check the executable programs, including the proposed physical witness."""
    assert result["models"]
    for model in result["models"]:
        program = model["program"]
        assert isinstance(program, Program)
        for number, values in features.items():
            prediction = program.predict(Observation(values, 0.))
            assert model["predictions_by_crystal"][str(number)] == prediction
        for item in responses:
            predicted = program.predict(Observation(features[item["crystal"]], 0.))
            active = model["lower"] <= item["frequency"] - predicted <= model["upper"]
            assert active == item["activated"], (model, item)
        setting = model["target_setting"]
        assert grid["lower"] <= setting <= grid["upper"]
        steps = (setting - grid["origin"]) / grid["step"]
        assert isclose(steps, round(steps), abs_tol=1e-8)
        predicted = program.predict(Observation(features[target], 0.))
        assert model["lower"] <= setting - predicted <= model["upper"]
        assert all(item["crystal"] != target or item["frequency"] != setting
                   for item in responses)


def test_joint_response_fit_recovers_reachable_hypotheses_missed_by_point_labels():
    features = {n: {"measurement": x} for n, x in enumerate((0., 1., 2., 8.), 1)}
    responses = []
    for n, positive, negatives in ((1, 7., (6., 14.)), (2, 15., (11., 19.)),
                                    (3, 23., (16., 24.))):
        responses.append(response(n, positive, True))
        responses.extend(response(n, value, False) for value in negatives)
    responses.append(response(4, 55., False))
    grid = {"lower": 0., "upper": 60., "step": 1., "origin": 0.}
    # Exact interpolation through successful settings extrapolates outside the
    # instrument. The same observations also permit center=10+5*x, band=[-3,3].
    point_fit = Program("add", children=(Program("const", value=7.),
        Program("mul", children=(Program("const", value=8.), X))))
    assert point_fit.predict(Observation(features[4], 0.)) == 71.
    seeds = {"linear": point_fit}
    inputs = (features, responses, ESTIMATORS, grid, seeds)
    before = deepcopy(inputs)
    result = fit_response_constraints(features, responses, ESTIMATORS, 4, grid, seeds)
    assert inputs == before
    assert result["compute_cost"] > 0
    assert_respects_evidence(result, features, responses, 4, grid)
    assert any(model["lower"] < model["upper"] for model in result["models"])


@pytest.mark.parametrize("observations", (
    [response(1, 10., True), response(1, 10., False)],
    [response(1, 9., True), response(1, 11., True), response(1, 10., False)],
), ids=("contradictory_same_setting", "negative_between_two_positives"))
def test_inconsistent_connected_band_preserves_conflicting_observations(observations):
    features = {1: {"measurement": 0.}, 2: {"measurement": 1.}}
    before = deepcopy(observations)
    result = fit_response_constraints(features, observations, ESTIMATORS, 2,
        {"lower": 0., "upper": 40., "step": 1., "origin": 0.}, {})
    assert observations == before
    assert result["models"] == []
    assert result["search"]
    assert all(item["status"] in {"inconsistent_response_band", "infeasible"}
               for item in result["search"])


def test_negative_only_target_retains_possible_activation_on_both_sides():
    features = {1: {"measurement": 0.}, 2: {"measurement": 1.}}
    observations = [response(1, 10., True), response(1, 9., False),
                    response(1, 11., False), response(2, 20., False)]
    grid = {"lower": 0., "upper": 40., "step": 1., "origin": 0.}
    result = fit_response_constraints(features, observations, ESTIMATORS, 2, grid, {})
    assert_respects_evidence(result, features, observations, 2, grid)
    settings = [model["target_setting"] for model in result["models"]]
    assert min(settings) < 20. < max(settings)
    # Neither branch is an observation of success on the target.
    assert not any(item["crystal"] == 2 and item["activated"] for item in observations)


def test_reachable_witness_uses_declared_fractional_grid():
    features = {1: {"measurement": 0.}, 2: {"measurement": 1.}}
    observations = [response(1, 10., True), response(1, 9., False),
                    response(1, 11., False), response(2, 20.25, False)]
    grid = {"lower": .25, "upper": 40.25, "step": .5, "origin": .25}
    result = fit_response_constraints(features, observations, ESTIMATORS, 2, grid, {})
    assert_respects_evidence(result, features, observations, 2, grid)
    assert all(not isclose(model["target_setting"], round(model["target_setting"]), abs_tol=1e-9)
               for model in result["models"])


def test_public_response_research_uses_joint_revision_and_learns_from_new_failure():
    observations = []
    references = []
    for n, positive, negatives in ((1, 7., (6., 14.)), (2, 15., (11., 19.)),
                                    (3, 23., (16., 24.))):
        references.append({"crystal": n,
            "measurements": {"measurement": {"value": float(n - 1), "evidence": n}},
            "frequency": {"value": positive, "evidence": n}})
        observations.append(response(n, positive, True))
        observations.extend(response(n, value, False) for value in negatives)
    observations.append(response(4, 55., False))
    state = {"research_table": {"references": references,
                "targets": [{"crystal": 4, "measurements": {
                    "measurement": {"value": 8., "evidence": 4}}}]},
             "response_settings": observations, "target_numbers": [4], "completed": [],
             "control_grid": {"lower": 0., "upper": 60., "step": 1., "origin": 0.}}
    before = deepcopy(state)
    first = propose_response_research(state, ESTIMATORS, "shared/research.py")
    assert state == before
    selected = first["selection"]["selected_experiment"]
    assert selected["candidate_id"].startswith("joint_coefficient_")
    assert selected["procedure"]["path"] == "shared/research.py"
    assert first["selected_number"] == 4
    assert 0. <= first["selected_setting"] <= 60.
    old_candidates = [candidate for candidate in first["candidates"]
                      if candidate["id"].startswith("coefficient_")]
    assert old_candidates
    assert all(candidate["target_prediction"] > 60. for candidate in old_candidates)

    failed_setting = first["selected_setting"]
    state["response_settings"].append({**response(4, failed_setting, False), "evidence": [99]})
    before_feedback = deepcopy(state)
    revised = propose_response_research(state, ESTIMATORS, "shared/research.py")
    assert state == before_feedback
    assert revised["selected_setting"] is not None
    assert revised["selected_setting"] != failed_setting
    assert set(first["evidence"]) < set(revised["evidence"])
    assert 99 in revised["evidence"]
    assert revised["response_count"] == first["response_count"] + 1
    assert any(candidate["family"] == "linear" and candidate["status"] == "compatible"
               for candidate in revised["candidates"])
    # Failure removes the supported coefficient/band pair, not the estimator
    # family or the original point-fit alternatives and their provenance.
    for old in old_candidates:
        assert any(candidate["program"] == old["program"]
                   and candidate["fit_sources"] == old["fit_sources"]
                   for candidate in revised["candidates"])
    for hypothesis in revised["hypotheses"]:
        program = Program.from_dict(hypothesis["program"])
        for item in state["response_settings"]:
            x = 8. if item["crystal"] == 4 else float(item["crystal"] - 1)
            prediction = program.predict(Observation({"measurement": x}, 0.))
            active = hypothesis["lower"] <= item["frequency"] - prediction <= hypothesis["upper"]
            assert active == item["activated"]
