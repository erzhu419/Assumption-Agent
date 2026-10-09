"""Response research must learn from actual successes and failures together."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.world_response_hypotheses import propose_response_research


PATH = "shared/consolidated_method.py"


def fixture():
    x = Program("feature", name="measurement")
    estimators = [{"id": "quadratic", "basis": [x.to_dict(), Program("mul", children=(x, x)).to_dict()]}]
    references = [{"crystal": n, "measurements": {"measurement": {"value": n, "evidence": n}},
                   "frequency": {"value": 5 + 5 * n * n, "evidence": n}} for n in (1, 2, 3)]
    targets = [{"crystal": n, "measurements": {"measurement": {"value": n, "evidence": n}}}
               for n in (4, 5)]
    responses = [{"crystal": n, "frequency": 5 + 5 * n * n, "activated": True, "evidence": [n]}
                 for n in (1, 2, 3)]
    responses += [{"crystal": 4, "frequency": setting, "activated": setting in (85, 86),
                   "evidence": [index]} for index, setting in enumerate((84, 85, 86, 87), 6)]
    return {"research_table": {"references": references, "targets": targets},
            "response_settings": responses, "target_numbers": [4, 5], "completed": [4],
            "control_grid": {"step": 1., "origin": 0., "lower": 0., "upper": 10000.}}, estimators


def test_successes_form_coefficient_alternatives_and_every_response_is_consumed():
    state, estimators = fixture()
    before = deepcopy((state, estimators))
    result = propose_response_research(state, estimators, PATH)
    assert (state, estimators) == before
    sources = [source for candidate in result["candidates"] for source in candidate["fit_sources"]]
    assert {item["frequency"] for source in sources for item in source["settings"]
            if item["crystal"] == 4} == {85, 86}
    assert any("leave_out_" in source["origin"] for source in sources)
    assert all(len(source["settings"]) >= 3 for source in sources)
    assert all(item["response_count"] == len(state["response_settings"]) for item in result["candidates"])
    assert result["evidence"] == list(range(1, 10))
    assert result["target_number"] == 5
    assert result["compute_cost"] > 0


def test_every_fixed_band_respects_all_positive_and_negative_responses():
    state, estimators = fixture()
    result = propose_response_research(state, estimators, PATH)
    assert result["hypotheses"]
    for hypothesis in result["hypotheses"]:
        program = Program.from_dict(hypothesis["program"])
        for response in state["response_settings"]:
            prediction = program.predict(Observation({"measurement": response["crystal"]}, 0.))
            outcome = hypothesis["lower"] <= response["frequency"] - prediction <= hypothesis["upper"]
            assert outcome == response["activated"]
        selected = result["selection"]["selected_experiment"]
        residual = selected["prediction"] - program.predict(Observation({"measurement": 5}, 0.))
        assert selected["predictions"][hypothesis["id"]] == (hypothesis["lower"] <= residual <= hypothesis["upper"])


def test_same_input_opposite_responses_refute_joint_models_instead_of_dropping_data():
    state, estimators = fixture()
    state["response_settings"].append({"crystal": 4, "frequency": 85, "activated": False, "evidence": [20]})
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "no_compatible_response_hypotheses"
    assert result["hypotheses"] == []
    assert all(item["status"] == "rejected" for item in result["candidates"])
    assert all(any(20 in response["evidence"] for response in item["conflicting_responses"])
               for item in result["candidates"])


def test_experiment_uses_public_noninteger_grid_and_changes_after_negative_response():
    state, estimators = fixture()
    state["control_grid"] = {"step": .5, "origin": .25, "lower": 0., "upper": 10000.}
    first = propose_response_research(state, estimators, PATH)
    setting = first["selected_setting"]
    assert (setting - .25) / .5 == int((setting - .25) / .5)
    experiment = first["selection"]["selected_experiment"]
    assert experiment["procedure"]["path"] == PATH
    assert "arguments" not in experiment["procedure"]
    assert experiment["execution_binding"] == "parent resumes selection stored by reference_response in current episode"
    state["response_settings"].append({"crystal": 5, "frequency": setting, "activated": False, "evidence": [20]})
    second = propose_response_research(state, estimators, PATH)
    assert second["selected_setting"] != setting
    assert second["response_count"] == first["response_count"] + 1
    for hypothesis in second["hypotheses"]:
        assert not hypothesis["lower"] <= setting - hypothesis["target_prediction"] <= hypothesis["upper"]


def test_boolean_disagreement_is_required_to_claim_a_discriminating_experiment():
    state, estimators = fixture()
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "discriminating_experiment_available"
    forecasts = result["selection"]["selected_experiment"]["predictions"]
    assert set(forecasts.values()) == {True, False}
    assert result["selection"]["predicted_separation"] > 0


def test_single_fixed_response_hypothesis_gets_confirmation_without_inventing_width():
    state, _ = fixture()
    references = state["research_table"]["references"][:1]
    references[0]["frequency"]["value"] = 10.
    state["research_table"]["references"] = references
    state["response_settings"] = [{"crystal": 1, "frequency": 10., "activated": True, "evidence": [1]}]
    result = propose_response_research(state, [{"id": "constant", "basis": []}], PATH)
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["lower"] == result["hypotheses"][0]["upper"] == 0.
    assert result["status"] == "confirmation_trial"
    assert result["selected_setting"] == 10.
    assert result["selection"]["predicted_separation"] == 0.


def test_no_reachable_support_stops_without_extending_control_range_or_band():
    state, estimators = fixture()
    state["control_grid"]["upper"] = 50.
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "no_untried_supported_setting"
    assert result["selected_setting"] is None
    assert result["hypotheses"]


def test_equal_programs_share_candidate_but_preserve_fit_provenance():
    state, estimators = fixture()
    first = propose_response_research(state, estimators, PATH)
    duplicate = {**estimators[0], "id": "same_basis_other_name"}
    second = propose_response_research(state, estimators + [duplicate], PATH)
    assert len(second["candidates"]) == len(first["candidates"])
    assert len(second["hypotheses"]) == len(first["hypotheses"])
    assert all({source["family"] for source in item["fit_sources"]}
               == {"quadratic", "same_basis_other_name"} for item in second["candidates"])


def test_absent_control_grid_does_not_silently_assume_integer_hz():
    state, estimators = fixture()
    state.pop("control_grid")
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "control_grid_needed"
    assert result["selection"] is None
    assert result["hypotheses"]


def test_exhausted_target_can_select_disagreement_on_a_measured_reference():
    state, estimators = fixture()
    state["control_grid"]["upper"] = 50.
    state["control_grids"] = {str(n): {"step": .25, "origin": 0., "lower": 0., "upper": 100.}
                              for n in (1, 2, 3)}
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "discriminating_experiment_available"
    assert result["target_number"] == 5
    assert result["selected_number"] in (1, 2, 3)
    selected = result["selection"]["selected_experiment"]
    number = selected["number"]
    assert selected["restore_frequency"] == 5 + 5 * number * number
    assert selected["procedure"]["path"] == PATH
    assert set(selected["predictions"].values()) == {True, False}
    for hypothesis in result["hypotheses"]:
        program = Program.from_dict(hypothesis["program"])
        prediction = program.predict(Observation({"measurement": number}, 0.))
        assert hypothesis["predictions_by_crystal"][str(number)] == prediction
        assert hypothesis["target_prediction"] == hypothesis["predictions_by_crystal"]["5"]
        assert selected["predictions"][hypothesis["id"]] == (
            hypothesis["lower"] <= selected["prediction"] - prediction <= hypothesis["upper"])


def test_reference_uses_its_own_noninteger_grid_and_no_other_objects_grid():
    state, estimators = fixture()
    for row in state["research_table"]["references"]:
        row["frequency"]["value"] += .25
    for response in state["response_settings"]:
        response["frequency"] += .25
    state["control_grid"]["upper"] = 1.
    state["control_grids"] = {"2": {"step": .125, "origin": .25, "lower": 0., "upper": 100.},
                              "4": {"step": 1., "origin": 0., "lower": 0., "upper": 10000.}}
    result = propose_response_research(state, estimators, PATH)
    assert result["status"] == "discriminating_experiment_available"
    assert result["selected_number"] == 2
    selected = result["selection"]["selected_experiment"]
    assert selected["restore_frequency"] == 25.25
    setting = selected["prediction"]
    assert (setting - .25) / .125 == int((setting - .25) / .125)
    assert setting != 25.25
    assert set(selected["predictions"].values()) == {True, False}


@pytest.mark.parametrize("basis", ([], [Program("const", value=1.).to_dict()]),
                         ids=("equivalent_predictions", "single_window"))
def test_reference_agreement_does_not_become_a_confirmation_trial(basis):
    from hegel_machine.cognition.world_response_hypotheses import _propose_fitted_responses
    state, _ = fixture()
    state["research_table"]["references"] = state["research_table"]["references"][:2]
    for row, value in zip(state["research_table"]["references"], (10., 12.)):
        row["frequency"]["value"] = value
    state["response_settings"] = [{"crystal": n, "frequency": value,
        "activated": True, "evidence": [n]} for n, value in ((1, 10.), (2, 12.))]
    state["control_grid"]["upper"] = 5.
    state["control_grids"] = {str(n): {"step": 1., "origin": 0., "lower": 0., "upper": 20.}
                              for n in (1, 2)}
    estimators = [{"id": "constant", "basis": basis}]
    result = _propose_fitted_responses(state, estimators, PATH)
    assert all(hypothesis["lower"] <= 11. - hypothesis["predictions_by_crystal"][str(n)]
               <= hypothesis["upper"] for hypothesis in result["hypotheses"] for n in (1, 2))
    if basis:
        assert len(result["hypotheses"]) == 1
    assert result["status"] == "no_untried_supported_setting"
    assert result["selection"] is None
    assert result["selected_number"] is None
    # Joint revision can widen the previously unbounded response interval and
    # propose a target experiment; it must not confirm an agreed reference.
    revised = propose_response_research(state, estimators, PATH)
    assert revised['selected_number'] == revised['target_number']
    assert revised['selection']['selected_experiment']['candidate_id'].startswith('joint_coefficient_')


def test_available_target_experiment_keeps_priority_over_references():
    state, estimators = fixture()
    first = propose_response_research(state, estimators, PATH)
    state["control_grids"] = {str(n): {"step": .25, "origin": 0., "lower": 0., "upper": 100.}
                              for n in (1, 2, 3)}
    second = propose_response_research(state, estimators, PATH)
    assert second["selected_number"] == second["target_number"] == 5
    assert second["selection"] == first["selection"]
    assert "restore_frequency" not in second["selection"]["selected_experiment"]


def test_saved_exhausted_point_fit_resumes_untried_parameter_research_once(monkeypatch):
    from hegel_machine.cognition import world_reference_recalibration as recipe

    state, estimators = fixture()
    state['expanded_features'] = ['measurement']
    monkeypatch.setattr(recipe, 'ESTIMATORS', estimators)
    memory = {'program_arguments': {}, 'limits': {'max_model_calls': 0},
              'costs': {'model_calls': 0}, 'model_available': False}
    request = recipe._extend_hypotheses(state, memory, 5, PATH)
    state['response_context'] = deepcopy(request['arguments']['context'])
    state['response_research'] = {'status': 'no_untried_supported_setting', 'selection': None}
    assert recipe._extend_hypotheses(state, memory, 5, PATH)['arguments']['operator'] == 'reference_response'
    state['response_research']['joint_parameter_search'] = []
    assert 'stop' in recipe._extend_hypotheses(state, memory, 5, PATH)
