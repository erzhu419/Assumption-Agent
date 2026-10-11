"""A new instrument must be measured before a failed parent can recover."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_response_measurement_recovery.py"
spec = importlib.util.spec_from_file_location("response_measurement_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("measurement_recovery") / "run"
    brief = recovery.run(path)
    return brief, json.loads((path / "prefix.json").read_text()), json.loads((path / "result.json").read_text())


def test_parent_acquires_the_refutation_before_author_supplies_new_measurement(completed_run):
    brief, prefix, _ = completed_run
    assert not prefix["agent"]["score"]["completedSuccessfully"]
    assert brief["prefix_family_statuses"]
    assert all(row["status"] == "UNSAT_VERIFIED" for row in brief["prefix_family_statuses"])
    assert brief["prefix_research_obligation"]["reason"] == "declared_shared_response_families_refuted"
    assert brief["candidate_origin"] == "author_supplied_after_refutation"
    assert not any(row["action"].get("action") == "USE" and row["action"].get("arg1") == 93
                   for row in prefix["observations"])


def test_new_readings_change_representation_then_failed_point_gets_actual_boolean_support(completed_run):
    brief, prefix, final = completed_run
    assert brief["completed_successfully"]
    before = len(prefix["observations"])
    indices = brief["new_sensor_observation_indices"]
    assert len(indices) == 5 and min(indices) >= before
    readings = final["state"]["research_table"]["references"]
    assert readings[0]["measurements"]["density"]["value"] == readings[1]["measurements"]["density"]["value"]
    assert readings[0]["measurements"]["signal"]["value"] != readings[1]["measurements"]["signal"]["value"]
    failed = next(q for q in brief["new_point_refutations"] if q["id"] == "reference_test_5_signal_square")
    assert failed["evidence"] and min(failed["evidence"]) > max(indices)
    physical = brief["new_boolean_physical_answers"]
    supported = [q for q in physical if q["status"] == "supported"]
    assert supported and all(q["answer"]["activated"] for q in supported)
    assert all(min(q["evidence"]) > max(failed["evidence"]) for q in supported)
    old_tests = {q["id"]: q for q in prefix["agent"]["research"]["questions"] if q["status"] == "refuted"}
    final_tests = {q["id"]: q for q in final["agent"]["research"]["questions"]}
    assert all(final_tests[qid] == question for qid, question in old_tests.items())
    assert brief["costs"]["model_calls"] == 0


def test_signal_instrument_reports_its_own_reading_not_aliased_density():
    lab = recovery.AliasedSignalLab()
    one = lab.act({"action": "USE", "arg1": 93, "arg2": 201})
    two = lab.act({"action": "USE", "arg1": 93, "arg2": 202})
    assert lab.density[1] == lab.density[2]
    assert one["observation"]["lastActionMessage"] != two["observation"]["lastActionMessage"]
    assert "signal magnitude = 2.0 units" in two["observation"]["lastActionMessage"]


@pytest.mark.parametrize("start,choice,boundary", [(770., 6, 0.), (9770., 2, 10000.)])
def test_declared_control_range_clamps_readout_and_actual_activation(start, choice, boundary):
    lab = recovery.AliasedSignalLab()
    lab.frequencies[5] = start
    lab.required[5] = boundary
    lab.act({"action": "PICKUP", "arg1": 205})
    lab.act({"action": "PUT", "arg1": 205, "arg2": 105})
    lab.act({"action": "TALK", "arg1": 105})

    result = lab.act({"chosen_dialog_option_int": choice})

    view = result["observation"]
    assert recovery.recipe._dialog(view) == (5, boundary)
    assert lab.frequencies[5] == boundary and lab.active(5)
    reactor = next(obj for obj in view["accessibleEnvironmentObjects"] if obj["uuid"] == 105)
    assert reactor["name"] == "crystal reactor (activated)"
    # A boundary readout remains usable by the existing control program.
    next_setting = 190.62727269291065 if boundary == 0 else 9010.
    tuning = recovery.recipe._tune(view, next_setting)
    assert tuning and tuning["actions"][0]["chosen_dialog_option_int"] != 1


def test_uninformative_new_measurement_does_not_confirm_refuted_family(tmp_path, completed_run):
    brief = recovery.run(tmp_path / "uninformative", case="uninformative")
    assert brief["prefix_family_statuses"] == completed_run[0]["prefix_family_statuses"]
    assert brief["prefix_actions"] == completed_run[0]["prefix_actions"]
    assert len(brief["new_sensor_observation_indices"]) == 5
    assert not brief["completed_successfully"]
    assert not brief["new_boolean_physical_answers"]
    assert brief["costs"]["model_calls"] == 0
    result = json.loads((tmp_path / "uninformative" / "result.json").read_text())
    family = result["state"]["response_research"]["family_research"]
    assert all(row["status"] == "UNSAT_VERIFIED" for row in family["families"])
    assert family["research_obligation"]["reason"] == "declared_shared_response_families_refuted"
    assert result["execution"]["status"] == "method_stopped"
