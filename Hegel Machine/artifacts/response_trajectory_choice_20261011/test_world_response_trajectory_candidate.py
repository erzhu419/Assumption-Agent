"""Tuning forecasts follow the public controller and buy joint information once."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as controller
from hegel_machine.cognition.world_reference_response import _reference_response_inputs
from hegel_machine.cognition.world_experiment_choice import (
    forecast_equivalence_classes, select_experiment,
)
from hegel_machine.cognition.world_response_trajectory import tuning_path, trajectory_experiments
from hegel_machine.cognition.world_response_transfer import reuse_response_knowledge
from scripts.run_response_matched_adoption import prepare_bundle
from test_world_control_preparation import SaturatingLab
from test_world_reference_hypothesis import PATH
from test_world_response_forecast_classes import _frozen_case


def _menu(lab, number, frequency, index=0):
    lab.frequencies[number] = frequency
    record = {"index": index, **lab.act({"action": "TALK", "arg1": 100 + number})}
    return controller._control_menus([record])[str(number)]


@pytest.mark.parametrize("initial,setting,restore,expected", [
    (0., 9.495, None, [10., 9.]),
    (9.8, 0., None, [0.]),
    (110., 121., 110., [120., 121., 111., 110.]),
])
def test_forecast_matches_actual_public_tuning_and_saturation(initial, setting, restore, expected):
    lab = SaturatingLab()
    menu = _menu(lab, 1, initial, index=7)
    # Archived JSON menus have string keys; the same options must be executed.
    forecast = tuning_path(json.loads(json.dumps(menu)), setting, restore)
    actual = []
    for target in ([setting] if restore is None else [setting, restore]):
        for _ in range(16):
            proposal = controller._tune(lab.public_observation(), target)
            if proposal is None:
                break
            action, = proposal["actions"]
            lab.act(action)
            actual.append({**action, "frequency": lab.frequencies[1]})
        else:
            pytest.fail("The actual public controller did not settle")
    assert forecast == actual
    assert [row["frequency"] for row in forecast] == expected
    assert menu["evidence"] == 7 and menu["frequency"] == initial


def test_saturated_unreachable_endpoint_does_not_claim_a_complete_forecast():
    lab = SaturatingLab()
    menu = _menu(lab, 1, 9999.)
    assert tuning_path(menu, 10001.) is None
    for _ in range(2):
        action, = controller._tune(lab.public_observation(), 10001.)["actions"]
        lab.act(action)
    assert lab.frequencies[1] == 10000.
    assert tuning_path(menu, 9990., max_steps=1) is None


def _design(sequences):
    identifiers = tuple(sequences[0][0])
    return {"hypotheses": [{"id": name, "claim": "Fixed physical explanation " + name}
                           for name in identifiers],
            "experiments": [{"id": str(index), "outcome": "The crystal responds along tuning",
                             "procedure": {"path": PATH, "max_steps": 512},
                             "predictions": deepcopy(rows[-1]), "forecast_sequence": deepcopy(rows)}
                            for index, rows in enumerate(sequences)]}


def test_repeated_and_complementary_readouts_add_cost_not_independent_information():
    split = {"a": True, "b": True, "c": False, "d": False}
    complement = {key: not value for key, value in split.items()}
    one = select_experiment(_design([[split]]))
    repeated = select_experiment(_design([[split, split]]))
    returning = select_experiment(_design([[split, complement, split]]))
    assert one["predicted_separation"] == .25
    assert repeated["predicted_separation"] == .125
    assert returning["predicted_separation"] == pytest.approx(1 / 12)
    assert [result["compute_cost"] for result in (one, repeated, returning)] == [4, 8, 12]
    endpoint = _design([[split]])
    del endpoint["experiments"][0]["forecast_sequence"]
    assert select_experiment(endpoint)["predicted_separation"] == one["predicted_separation"]


def test_new_split_is_valuable_and_intermediate_outcomes_keep_hypotheses_distinct():
    first = {"a": True, "b": True, "c": False, "d": False}
    second = {"a": True, "b": False, "c": True, "d": False}
    design = _design([[second, second], [first, second]])
    result = select_experiment(design)
    assert result["selected_experiment"]["id"] == "1"
    assert result["predicted_separation"] == .1875
    assert len(forecast_equivalence_classes(design["experiments"])) == 4
    endpoints = [{key: value for key, value in experiment.items() if key != "forecast_sequence"}
                 for experiment in design["experiments"]]
    assert len(forecast_equivalence_classes(endpoints)) == 2


def test_reference_restoration_is_in_forecast_and_paid_cost():
    lab = SaturatingLab()
    menu = _menu(lab, 1, 99., index=12)
    hypotheses = [{"id": key, "claim": "Activation only at " + str(frequency),
                   "lower": 0., "upper": 0., "predictions_by_crystal": {"1": frequency}}
                  for key, frequency in (("a", 100.), ("b", 101.))]
    experiment = {"id": "reference", "number": 1, "prediction": 101.,
                  "restore_frequency": 99., "outcome": "Reference activation",
                  "predictions": {"a": False, "b": True},
                  "procedure": {"path": PATH, "max_steps": 512}}
    original = deepcopy((experiment, menu, hypotheses))
    forecast, = trajectory_experiments([experiment], hypotheses, {"1": menu})
    assert [step["frequency"] for step in forecast["tuning_path"]] == [100., 101., 100., 99.]
    assert forecast["control_evidence"] == 12
    chosen = select_experiment({"hypotheses": hypotheses, "experiments": [forecast]})
    assert chosen["compute_cost"] == 8
    assert chosen["predicted_separation"] == .0625
    assert (experiment, menu, hypotheses) == original


def test_frozen_selection_keeps_all_members_and_original_endpoint_experiments():
    state, estimators = _frozen_case(copies=6)
    lab = SaturatingLab()
    state["control_menus"] = {"5": _menu(lab, 5, 95., index=19)}
    before = deepcopy(state)
    result = reuse_response_knowledge(state, estimators, PATH)
    selected = result["selection"]["selected_experiment"]
    assert result["selection"]["comparison_scope"] == "prospective_tuning_paths"
    assert len(result["hypotheses"]) == 10
    ids = {hypothesis["id"] for hypothesis in result["hypotheses"]}
    assert all(set(row) == ids for row in selected["forecast_sequence"])
    assert set(selected["predictions"]) == ids
    assert result["selection"]["hypotheses"] == result["hypotheses"]
    assert all("forecast_sequence" not in experiment for experiment in result["experiments"])
    assert result["trajectory_inputs"] == state["control_menus"]
    assert set(result["tuning_paths"]) == {experiment["id"] for experiment in result["experiments"]}
    assert state == before


def test_missing_one_object_menu_falls_back_for_the_whole_comparison_pool():
    state, estimators = _frozen_case()
    state["target_numbers"].append(4)
    state["research_table"]["targets"].append({"crystal": 4,
        "measurements": {"measurement": {"value": 4., "evidence": 3}}})
    state["control_grids"] = {"4": deepcopy(state["control_grid"])}
    baseline = reuse_response_knowledge(state, estimators, PATH)
    lab = SaturatingLab()
    state["control_menus"] = {"5": _menu(lab, 5, 95., index=19)}
    partial = reuse_response_knowledge(state, estimators, PATH)
    assert {experiment["number"] for experiment in partial["experiments"]} == {4, 5}
    assert partial["selection"] == baseline["selection"]
    assert "tuning_paths" not in partial and "trajectory_inputs" not in partial
    state["control_menus"]["4"] = _menu(lab, 4, 95., index=20)
    complete = reuse_response_knowledge(state, estimators, PATH)
    assert complete["selection"]["comparison_scope"] == "prospective_tuning_paths"


def test_menu_cache_survives_json_and_ignores_only_record_identity():
    state, estimators = _frozen_case()
    state["control_grids"] = {"5": deepcopy(state["control_grid"])}
    state["control_menus"] = {"5": _menu(SaturatingLab(), 5, 95., index=19)}
    original = _reference_response_inputs(state, estimators)
    restored = json.loads(json.dumps(state))
    assert json.loads(json.dumps(original)) == _reference_response_inputs(restored, estimators)
    restored["control_menus"]["5"]["evidence"] = 20
    assert _reference_response_inputs(restored, estimators) == original
    restored["control_menus"]["5"]["frequency"] += 1
    assert _reference_response_inputs(restored, estimators) != original


def test_archived_bundle_carries_shared_tuning_helpers_and_runs_actual_controls():
    archive = Path(__file__).resolve().parents[1] / (
        "artifacts/response_exhausted_stage_20261011/after/retained/sources/"
        "cubic_original_1__gain_lower/numeric_bundle.json")
    bundle = json.loads(archive.read_text())
    original = deepcopy(bundle)
    exported = prepare_bundle(bundle)
    assert prepare_bundle(exported) == exported
    namespace = {}
    exec(compile(exported["files"][PATH], PATH, "exec"), namespace)
    lab = SaturatingLab()
    menu = _menu(lab, 1, 110.)
    actual = []
    while proposal := namespace["_tune"](lab.public_observation(), 121.):
        action, = proposal["actions"]
        lab.act(action)
        actual.append({**action, "frequency": lab.frequencies[1]})
        assert len(actual) <= 10
    assert actual == tuning_path(menu, 121.)
    assert {key: value for key, value in exported.items() if key != "files"} == {
        key: value for key, value in original.items() if key != "files"}
    assert bundle == original
