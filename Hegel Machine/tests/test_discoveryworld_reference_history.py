import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_prerequisite_research import (
    intervention_history,
    public_reference_history,
)
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_prerequisite import ControlLab


def record(index, target=7, *, name="device (active)", objects=None, action=None, dialog=None):
    view = {"accessibleEnvironmentObjects": objects if objects is not None else
            [{"uuid": target, "name": name, "description": "public device"}],
            "lastActionMessage": f"public message {index}"}
    if dialog is not None:
        view["dialog_box"] = dialog
    return {"index": index, "action": action or {"action": "TALK", "arg1": target},
            "feedback": {"success": True}, "observation": view}


def test_keeps_first_and_latest_per_state_and_preserves_earlier_working_description():
    records = [record(0), record(1), record(2), record(3, name="device (inactive)"),
               record(4, name="device (inactive)"), record(5, name="device (inactive)")]
    projected = public_reference_history(records)
    assert [row["index"] for row in projected] == [0, 2, 3, 5]
    assert projected[0]["public_objects"][0]["name"] == "device (active)"
    assert projected[-1]["public_objects"][0]["name"] == "device (inactive)"
    assert all("status" not in row and "supported" not in row for row in projected)


def test_early_other_object_survives_later_failure_window():
    earlier = record(3, target=7)
    later = record(20, target=8, name="device (uncalibrated)")
    later["feedback"] = {"success": False}
    records = [earlier, later]
    assert [row["index"] for row in intervention_history(records, 8, 10)] == [20]
    assert [(row["target"], row["index"]) for row in public_reference_history(records)] == [(7, 3), (8, 20)]


def test_use_dialog_options_and_exit_stay_associated_with_public_container():
    child = {"uuid": 9, "name": "sample", "description": "public sample"}
    container = {"uuid": 7, "name": "device", "description": "contains sample [uuid: 9]"}
    opened = record(0, action={"action": "USE", "arg1": 4, "arg2": 7},
        objects=[child, container], dialog={"is_in_dialog": True, "dialogIn": "setting 1"})
    changed = record(1, action={"chosen_dialog_option_int": 1}, objects=[container, child],
        dialog={"is_in_dialog": True, "dialogIn": "setting 2"})
    exited = record(2, action={"chosen_dialog_option_int": 9}, objects=[container, child],
        dialog={"is_in_dialog": False})
    unrelated = record(3, action={"chosen_dialog_option_int": 1}, objects=[container, child])
    projected = public_reference_history([opened, changed, exited, unrelated])
    device_rows = [row for row in projected if row["target"] == 7]
    assert [row["index"] for row in device_rows] == [0, 2]
    assert device_rows[0]["dialog_box"]["dialogIn"] == "setting 1"
    assert device_rows[-1]["dialog_box"] == {"is_in_dialog": False}
    assert [item["uuid"] for item in device_rows[0]["public_objects"]] == [7, 9]
    assert not any(row["index"] == 3 for row in projected)


def test_missing_object_preserves_only_actual_feedback_without_failure_inference():
    missing = record(4, objects=[])
    projected = public_reference_history([missing])
    assert projected == [{"target": 7, "index": 4, "action": missing["action"],
        "feedback": {"success": True}, "lastActionMessage": "public message 4", "public_objects": []}]


def test_navigation_and_uninteracted_visible_floor_are_not_reference_records():
    floor = {"uuid": 99, "name": "floor", "description": "public floor"}
    walk = record(0, objects=[floor], action={"action": "MOVE_DIRECTION", "arg1": "north"})
    visit = record(1, objects=[floor, {"uuid": 7, "name": "device", "description": "public device"}])
    projected = public_reference_history([walk, visit])
    assert len(projected) == 1
    assert projected[0]["target"] == 7
    assert [item["uuid"] for item in projected[0]["public_objects"]] == [7]


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
def test_early_reference_reaches_both_provider_contexts_outside_recent_window(tmp_path, arm):
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    agent.lab = ControlLab()
    agent._question({"id": "task", "question": "Explain the actual device response"})
    agent.observations = [record(0, target=8, name="other device (active)")]
    agent.observations.extend(record(index, objects=[],
        action={"action": "MOVE_DIRECTION", "arg1": "north"}) for index in range(1, 15))
    agent.observations.append(record(15, target=7, name="device (inactive)"))
    agent._save_state()

    request = agent._request()
    context = prepare_code_context(tmp_path, request, schema={}, instructions="")
    assert 0 not in [row["index"] for row in context["evidence"]["recent_action_results"]]
    history = context["evidence"]["reference_history"]
    assert [(row["target"], row["index"]) for row in history] == [(8, 0), (7, 15)]
    assert history[0]["public_objects"][0]["name"] == "other device (active)"
    assert history == public_reference_history(agent.observations)
    assert all(set(row) == {"target", "index", "action", "feedback", "lastActionMessage", "public_objects"}
               for row in history)
    assert context["research"]["answers"] == {"task": {"status": "open", "evidence": []}}
