"""A shared response law can be tested on another unfinished measured object.

These fixtures reuse the Boolean chooser and public physical controller; they
check integration behavior, not scientific discovery or benchmark performance.
"""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_reference_response as wrapper
from hegel_machine.cognition import world_response_hypotheses as response
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from test_world_reference_hypothesis import PATH, ReferenceLab, prepared, square
from test_world_response_controller import BoundedReferenceLab, _exhausted_grammar, _state, _submit_point_prefix


def _pool():
    first_grid = {"step": 1., "origin": 0., "lower": 0., "upper": 20.}
    second_grid = {"step": .5, "origin": .25, "lower": 0., "upper": 5.}
    state = {"target_numbers": [4, 5], "completed": [], "response_settings": [],
             "control_grid": first_grid,
             "control_grids": {"4": first_grid, "5": second_grid},
             "research_table": {"references": [], "targets": [
                 {"crystal": n, "measurements": {"x": {"value": n, "evidence": n}}}
                 for n in (4, 5)]}}
    hypotheses = [{"id": name, "claim": "A fixed shared activation law",
                   "family": name, "candidate_id": name, "lower": 0., "upper": 0.,
                   "target_prediction": 10.,
                   "predictions_by_crystal": {"4": 10., "5": value}}
                  for name, value in (("a", 2.25), ("b", 3.25))]
    result = {"target_number": 4, "hypotheses": hypotheses,
              "candidates": [{"id": name, "fit_mse": 0.} for name in ("a", "b")],
              "selection": None, "selected_setting": None, "selected_number": None,
              "compute_cost": 0}
    return state, result


def test_shared_chooser_uses_second_target_to_separate_laws_on_its_own_grid():
    state, result = _pool()
    before = deepcopy(state)
    chosen = response._select_response_experiment(result, state, PATH)

    assert state == before
    assert chosen["target_number"] == 4
    assert chosen["selected_number"] == 5
    experiment = chosen["selection"]["selected_experiment"]
    assert experiment["prediction"] in (2.25, 3.25)
    assert set(experiment["predictions"].values()) == {True, False}
    assert chosen["selection"]["predicted_separation"] > 0
    assert "restore_frequency" not in experiment
    assert experiment["procedure"]["path"] == PATH


def test_same_frequency_on_two_targets_has_distinct_experiment_identity(monkeypatch):
    state, result = _pool()
    state["control_grids"]["5"] = deepcopy(state["control_grid"])
    result["hypotheses"][0]["predictions_by_crystal"]["5"] = 10.
    result["hypotheses"][1]["predictions_by_crystal"]["5"] = 11.
    actual_choose = response.select_experiment
    designs = []

    def inspect(design):
        designs.append(deepcopy(design))
        return actual_choose(design)

    monkeypatch.setattr(response, "select_experiment", inspect)
    chosen = response._select_response_experiment(result, state, PATH)

    experiments = designs[0]["experiments"]
    same_setting = [item for item in experiments if item["prediction"] == 10.]
    assert {item["number"] for item in same_setting} == {4, 5}
    assert len({item["id"] for item in same_setting}) == 2
    assert chosen["selected_number"] == 5


@pytest.mark.parametrize("excluded", ["completed", "no_own_grid", "not_measured"])
def test_other_target_must_be_unfinished_measured_and_have_its_own_grid(excluded):
    state, result = _pool()
    if excluded == "completed":
        state["completed"] = [5]
    elif excluded == "no_own_grid":
        state["control_grids"].pop("5")
    else:
        state["research_table"]["targets"] = state["research_table"]["targets"][:1]
    chosen = response._select_response_experiment(result, state, PATH)

    assert chosen["selected_number"] == 4
    assert chosen["selected_setting"] == 10.
    assert chosen["status"] == "confirmation_trial"
    assert chosen["selection"]["predicted_separation"] == 0.


@pytest.mark.parametrize("activated", [True, False])
def test_feedback_refutes_using_actual_test_object_not_current_target(tmp_path, monkeypatch, activated):
    lab = ReferenceLab()
    lab.installed.add(5)
    lab.frequencies[5] = 350.
    lab.required[5] = 350. if activated else 351.
    record = {"index": 0, **lab.act({"action": "TALK", "arg1": 105})}
    state = {"test": {"number": 5, "crystal": 205, "reactor": 105,
                      "family": "shared_law", "prediction": 350., "started": 0,
                      "response_question": "response_4"},
             "response_research": {"target_number": 4, "hypotheses": [
                 {"id": "a", "lower": 0., "upper": 0., "target_prediction": 100.,
                  "predictions_by_crystal": {"4": 100., "5": 350.}},
                 {"id": "b", "lower": 0., "upper": 0., "target_prediction": 350.,
                  "predictions_by_crystal": {"4": 350., "5": 400.}}]}}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(recipe, "STATE", state_path)
    answer = recipe.answer({"current_observation": lab.public_observation()}, [record], {})

    assert answer["status"] == ("supported" if activated else "refuted")
    feedback = answer["answer"]["response_feedback"]
    assert feedback["remaining_hypotheses"] == (["a"] if activated else ["b"])
    assert feedback["refuted_hypotheses"] == (["b"] if activated else ["a"])
    assert {item["crystal"] for item in feedback["observed_responses"]} == {5}
    assert answer["evidence"] == [0]


def test_completed_secondary_target_invalidates_saved_experiment_without_changing_primary(monkeypatch):
    state, result = _pool()
    state["expanded_features"] = ["x"]
    monkeypatch.setattr(recipe, "ESTIMATORS", [{"id": "linear", "basis": [{"op": "feature", "name": "x"}]}])
    memory = {"program_arguments": {}, "limits": {"max_model_calls": 0},
              "costs": {"model_calls": 0}, "model_available": False}
    request = recipe._extend_hypotheses(state, memory, 4, PATH)
    state["response_context"] = deepcopy(request["arguments"]["context"])
    state["response_research"] = response._select_response_experiment(result, state, PATH)
    state["response_research"]["question_id"] = "response_4"
    old_trial = recipe._extend_hypotheses(state, memory, 4, PATH)["response_trial"]
    assert old_trial["number"] == 5

    # Completion can change after a paid observation has already been refreshed.
    # The remaining primary and all numerical evidence are unchanged here.
    state["completed"] = [5]
    revised = recipe._extend_hypotheses(state, memory, 4, PATH)
    assert revised["arguments"]["operator"] == "reference_response"
    assert "response_trial" not in revised


def test_public_joint_refit_finds_second_target_when_first_has_no_reachable_response():
    references, observations = [], []
    for number, positive, negatives in ((1, 7., (6., 14.)), (2, 15., (11., 19.)),
                                         (3, 23., (16., 24.))):
        references.append({"crystal": number,
            "measurements": {"x": {"value": float(number - 1), "evidence": number}},
            "frequency": {"value": positive, "evidence": number}})
        observations.append({"crystal": number, "frequency": positive,
                             "activated": True, "evidence": [number]})
        observations.extend({"crystal": number, "frequency": setting,
                             "activated": False, "evidence": [number]}
                            for setting in negatives)
    state = {"research_table": {"references": references, "targets": [
                {"crystal": number, "measurements": {"x": {"value": x, "evidence": number}}}
                for number, x in ((4, 8.), (5, 3.))]},
             "response_settings": observations, "target_numbers": [4, 5], "completed": [],
             "control_grids": {"4": {"lower": 0., "upper": 10., "step": 1., "origin": 0.},
                               "5": {"lower": 0., "upper": 25., "step": 1., "origin": 0.}}}
    state["control_grid"] = state["control_grids"]["4"]
    estimators = [{"id": "linear", "basis": [Program("feature", name="x").to_dict()]}]
    before = deepcopy((state, estimators))
    result = response.propose_response_research(state, estimators, PATH)

    assert (state, estimators) == before
    assert result["target_number"] == 4 and result["selected_number"] == 5
    assert {item["design_target_number"] for item in result["joint_parameter_search"]} == {4, 5}
    selected = result["selection"]["selected_experiment"]
    assert selected["candidate_id"].startswith("joint_coefficient_")
    assert 0. <= selected["prediction"] <= 25.
    assert selected["prediction"] == round(selected["prediction"])
    assert not any(item["crystal"] == 5 for item in state["response_settings"])
    # Equality fitting extrapolates to 31 on object 5. Inequalities also admit
    # center 10 + 5*x and residual band [-3, 3], with reachable support there.
    old = [item for item in result["candidates"] if item.get("id", "").startswith("coefficient_")]
    assert old and all(Program.from_dict(item["program"]).predict(Observation({"x": 3.}, 0.)) > 25.
                       for item in old)
    for hypothesis in result["hypotheses"]:
        program = Program.from_dict(hypothesis["program"])
        for item in observations:
            prediction = program.predict(Observation({"x": float(item["crystal"] - 1)}, 0.))
            active = hypothesis["lower"] <= item["frequency"] - prediction <= hypothesis["upper"]
            assert active == item["activated"]
        prediction = program.predict(Observation({"x": 3.}, 0.))
        assert selected["predictions"][hypothesis["id"]] == (
            hypothesis["lower"] <= selected["prediction"] - prediction <= hypothesis["upper"])


def test_controller_installs_other_target_and_preserves_its_success(tmp_path, monkeypatch):
    agent, _ = prepared(tmp_path)
    lab = BoundedReferenceLab()
    agent.lab = lab
    _submit_point_prefix(agent, lab, square("gap", "gap_square"))
    assert not lab.active(4) and not lab.active(5)
    _exhausted_grammar(agent)
    before = len(lab.actions)
    calls = []

    def choose(state, estimators, path):
        calls.append(deepcopy(state))
        hypotheses = [{"id": name, "family": "gap_square", "candidate_id": name,
                       "claim": "Fixed response hypothesis", "lower": 0., "upper": 0.,
                       "target_prediction": 350.,
                       "predictions_by_crystal": {"4": 350., "5": value}}
                      for name, value in (("a", 350.), ("b", 400.))]
        experiment = {"id": "test_other_target", "number": 5, "prediction": 350.,
                      "family": "gap_square", "candidate_id": "a",
                      "predictions": {"a": True, "b": False},
                      "outcome": "Crystal 5 activates at 350 Hz",
                      "procedure": {"path": path, "max_steps": 128}}
        selected = not any(item["crystal"] == 5 and item["activated"]
                           for item in state["response_settings"])
        return {"status": "discriminating_experiment_available" if selected else "no_untried_supported_setting",
                "target_number": 4, "assumptions": ["One fixed shared response law"],
                "joint_parameter_search": [], "hypotheses": hypotheses,
                "candidates": [{"id": name, "status": "compatible"} for name in ("a", "b")],
                "selection": {"selected_experiment": experiment if selected else None},
                "selected_number": 5 if selected else None,
                "selected_setting": 350. if selected else None, "compute_cost": 0,
                "response_count": len(state["response_settings"]),
                "evidence": sorted({i for item in state["response_settings"] for i in item["evidence"]})}

    monkeypatch.setattr(wrapper, "propose_response_research", choose)
    result = agent._run_procedure(lab, PATH, max_steps=128)

    assert result["status"] == "method_stopped" and not lab.success
    assert lab.active(5) and not lab.active(4)
    state = _state(agent)
    assert state["completed"] == [5]
    test = state["tests"][-1]
    assert test["number"] == 5 and test["status"] == "supported"
    assert "restore_frequency" not in test and "recovery" not in state
    assert {"action": "PICKUP", "arg1": 205} in lab.actions[before:]
    assert {"action": "PUT", "arg1": 205, "arg2": 105} in lab.actions[before:]
    assert calls[-1]["completed"] == [5]
    physical = next(item for item in agent.questions if item["id"] == test["question"])
    feedback = physical["answer"]["response_feedback"]
    assert feedback["remaining_hypotheses"] == ["a"]
    assert feedback["refuted_hypotheses"] == ["b"]
    assert agent.costs["model_calls"] == 0
