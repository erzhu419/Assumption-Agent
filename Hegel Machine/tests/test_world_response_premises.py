"""Acquired cross-object contrasts distinguish a weak representation from its premise."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.world_response_consequences import (
    _shared_premise_test,
    assess_response_consequences,
)
from hegel_machine.cognition.world_response_hypotheses import propose_response_research


ESTIMATORS = [
    {"id": "shared_x", "basis": [{"op": "feature", "name": "x"}]},
    {"id": "shared_x_z", "basis": [{"op": "feature", "name": name}
                                      for name in ("x", "z")]},
]


def public_state(*, extra_measurement):
    """A paid public prefix; no response oracle or task label reaches selection."""
    z, transferred_setting = (2., 4.) if extra_measurement else (0., 3.)
    coordinates = [(0., 0.), (0., z), (2., 0.),
                   (1., 1. if extra_measurement else 0.), (1.5, 0.)]
    rows = [{"crystal": number, "measurements": {
                name: {"value": value, "evidence": number - 1}
                for name, value in zip(("x", "z"), values)}}
            for number, values in enumerate(coordinates, 1)]
    responses = [{"crystal": number, "frequency": setting,
                  "activated": True, "evidence": [number + 4]}
                 for number, setting in enumerate((0., transferred_setting, 10.), 1)]
    for row, response in zip(rows, responses):
        row["frequency"] = {"value": response["frequency"],
                            "evidence": response["evidence"][0]}
    grid = {"lower": 0., "upper": 14., "step": 1., "origin": 0.}
    return {"research_table": {"references": rows[:3], "targets": rows[3:]},
            "response_settings": responses, "target_numbers": [4, 5], "completed": [],
            "control_grid": grid,
            "control_grids": {str(number): deepcopy(grid) for number in range(1, 6)},
            "consequence_panel": [
                {"crystal": 4, "frequency": 6. if extra_measurement else 4.},
                {"crystal": 5, "frequency": 6.}]}


def propose(state):
    return propose_response_research(state, deepcopy(ESTIMATORS), "public_lab.py")


def stalled_proposal(state):
    previous = propose(state)
    previous["family_research"]["no_progress_cycles"] = 2
    state["response_research"] = previous
    return propose(state)


@pytest.mark.parametrize("extra_measurement", (False, True))
def test_stalled_research_transfers_paid_positive_through_existing_action_pool(extra_measurement):
    state = public_state(extra_measurement=extra_measurement)
    initial = propose(state)
    assert initial["family_research"]["selected_role"] != "shared_premise_test"

    result = stalled_proposal(state)
    research = result["family_research"]
    assert result["status"] == research["selected_role"] == "shared_premise_test"
    selected = result["selection"]["selected_experiment"]
    assert selected in result["experiments"]
    assert selected["number"] == 1
    source = research["premise_test"]["source_response"]
    assert source in state["response_settings"] and source["activated"]
    assert selected["prediction"] == source["frequency"]
    assert (selected["number"], selected["prediction"]) not in {
        (row["crystal"], row["frequency"]) for row in state["response_settings"]}
    expected = ["shared_x"] if extra_measurement else ["shared_x", "shared_x_z"]
    assert research["premise_test"]["required_by_families"] == expected
    # A counterfactual negative motivates the experiment; it has not occurred.
    assert research["refuted_family_ids"] == []
    assert all(family["status"] == "SAT_WITNESS" for family in research["families"])


def test_acquired_negative_refutes_only_the_coarse_representation():
    state = public_state(extra_measurement=True)
    selected = stalled_proposal(state)
    state["response_research"] = selected
    state["response_settings"].append({
        "crystal": selected["selected_number"], "frequency": selected["selected_setting"],
        "activated": False, "evidence": [8]})

    result = propose(state)
    research = result["family_research"]
    statuses = {family["id"]: family["status"] for family in research["families"]}
    assert statuses == {"shared_x": "UNSAT_VERIFIED", "shared_x_z": "SAT_WITNESS"}
    assert research["refuted_family_ids"] == ["shared_x"]
    assert research["nonlocal_progress"] == 0
    assert research["no_progress_cycles"] == 0
    assert research["possible_signatures"]
    assert "research_obligation" not in research
    assert state["completed"] == []
    assert all(row["crystal"] <= 3 for row in state["response_settings"])


def test_identical_public_measurements_require_new_information_after_actual_conflict():
    state = public_state(extra_measurement=False)
    selected = stalled_proposal(state)
    state["response_research"] = selected
    negative = {"crystal": selected["selected_number"],
                "frequency": selected["selected_setting"], "activated": False, "evidence": [8]}
    state["response_settings"].append(negative)

    result = propose(state)
    research = result["family_research"]
    assert research["refuted_family_ids"] == ["shared_x", "shared_x_z"]
    assert research["possible_signatures"] == []
    assert research["admitted_signatures"] == []
    assert result["status"] == "family_research_needed"
    assert result["selection"] is None
    assert result["selected_number"] is None and result["selected_setting"] is None
    assert research["selected_role"] == "shared_premise_research"
    obligation = research["research_obligation"]
    assert obligation["reason"] == "declared_shared_response_families_refuted"
    assert "additional measurement or observable state" in obligation["question"]
    positive, actual_negative = obligation["same_measurement_conflict"]
    assert positive == selected["family_research"]["premise_test"]["source_response"]
    assert actual_negative == negative
    assert positive["frequency"] == actual_negative["frequency"]
    assert positive["activated"] != actual_negative["activated"]
    measurements = {row["crystal"]: {name: reading["value"]
                    for name, reading in row["measurements"].items()}
                    for row in state["research_table"]["references"]}
    assert measurements[positive["crystal"]] == measurements[actual_negative["crystal"]]
    # All signatures may be formally excluded when the entire family is false.
    # That empty set never licenses a target prediction or successful completion.
    assert state["completed"] == []


def test_already_refuted_shared_families_request_research_without_waiting_for_stagnation():
    state = public_state(extra_measurement=False)
    state["response_settings"].append({"crystal": 1, "frequency": 3.,
                                       "activated": False, "evidence": [8]})
    result = propose(state)
    research = result["family_research"]
    assert research["no_progress_cycles"] == 0
    assert research["research_obligation"]["reason"] == "declared_shared_response_families_refuted"
    assert result["selection"] is None
    assert research["refuted_family_ids"] == []  # No earlier feasible snapshot is invented.


def test_guaranteed_target_consequence_takes_priority_over_shared_premise_challenge():
    def measured(number, value):
        return {"crystal": number, "measurements": {"x": {"value": value, "evidence": number}}}

    state = {"research_table": {"references": [measured(1, 0.), measured(2, 2.)],
                                "targets": [measured(3, 1.), measured(4, 0.)]},
             "response_settings": [
                 {"crystal": 1, "frequency": 10., "activated": True},
                 {"crystal": 1, "frequency": 9., "activated": False},
                 {"crystal": 2, "frequency": 12., "activated": True}],
             "target_numbers": [3, 4], "completed": [],
             "consequence_panel": [{"crystal": 4, "frequency": 11.}],
             "response_research": {"family_research": {"no_progress_cycles": 2}}}
    actions = [{"id": name, "number": number, "prediction": frequency,
                "predictions": {"first": True, "second": False}}
               for name, number, frequency in (("premise", 4, 10.), ("consequence", 1, 11.))]
    result = assess_response_consequences(state, ESTIMATORS[:1], {
        "compute_cost": 0, "experiments": actions,
        "selection": {"selected_experiment": actions[0]}})
    assert result["selection"]["selected_experiment"]["id"] == "consequence"
    assert result["family_research"]["selected_role"] == "nonlocal_family_test"
    assert result["selection"]["guaranteed_nonlocal_eliminations"] == 1
    assert "premise_test" not in result["family_research"]


def fractional_prefix():
    state = public_state(extra_measurement=False)
    for row, response, setting in zip(state["research_table"]["references"],
                                      state["response_settings"], (.2, 3.7, 10.2)):
        row["frequency"]["value"] = response["frequency"] = setting
        state["control_grids"][str(row["crystal"])]["origin"] = setting % 1.
    return state


def test_mismatched_reference_grids_test_existing_interior_of_shared_positive_band():
    state = fractional_prefix()
    result = stalled_proposal(state)
    research = result["family_research"]
    assert research["selected_role"] == "shared_premise_test"
    selected = result["selection"]["selected_experiment"]
    assert selected in result["experiments"]
    assert selected["number"] in (1, 2)
    assert .2 < selected["prediction"] < 3.7
    grid = state["control_grids"][str(selected["number"])]
    position = (selected["prediction"] - grid["origin"]) / grid["step"]
    assert position == pytest.approx(round(position))
    premise = research["premise_test"]
    assert premise["inference"] == "shared_connected_band_interval"
    assert premise["source_responses"] == state["response_settings"][:2]
    assert premise["source_response"] in premise["source_responses"]
    assert all(row["activated"] for row in premise["source_responses"])
    assert research["refuted_family_ids"] == []
    assert all(family["status"] == "SAT_WITNESS" for family in research["families"])


def test_actual_interior_negative_refutes_shared_band_without_rounding_source_settings():
    state = fractional_prefix()
    selected = stalled_proposal(state)
    assert selected["family_research"]["selected_role"] == "shared_premise_test"
    state["response_research"] = selected
    state["response_settings"].append({"crystal": selected["selected_number"],
        "frequency": selected["selected_setting"], "activated": False, "evidence": [8]})
    result = propose(state)
    research = result["family_research"]
    assert research["refuted_family_ids"] == ["shared_x", "shared_x_z"]
    assert all(family["status"] == "UNSAT_VERIFIED" for family in research["families"])
    assert result["status"] == "family_research_needed"
    assert result["selection"] is None
    assert state["completed"] == []
    assert [row["frequency"] for row in state["response_settings"][:2]] == [.2, 3.7]


def test_unresolved_family_interval_requires_equal_basis_and_existing_unmeasured_action():
    features = {1: {"x": 0., "z": 0.}, 2: {"x": 0., "z": 2.},
                3: {"x": 1., "z": 0.}}
    responses = [{"crystal": 1, "frequency": .2, "activated": True, "evidence": [1]},
                 {"crystal": 2, "frequency": 3.7, "activated": True, "evidence": [2]}]
    actions = [{"number": 1, "prediction": 1.2}, {"number": 3, "prediction": 1.2}]
    families = [{"id": family["id"], "status": "UNKNOWN"} for family in ESTIMATORS]
    premise = _shared_premise_test(features, responses, ESTIMATORS, actions, families)
    assert premise is not None
    assert premise["experiment"] == actions[0]
    assert premise["required_by_families"] == ["shared_x"]
    assert premise["source_responses"] == responses
    assert _shared_premise_test(features, responses, ESTIMATORS, actions[1:], families) is None
    paid = responses + [{"crystal": 1, "frequency": 1.2, "activated": True, "evidence": [3]}]
    assert _shared_premise_test(features, paid, ESTIMATORS, actions, families) is None
