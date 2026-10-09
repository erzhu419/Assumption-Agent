"""Keep observed physical state distinct from successful command execution."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_prerequisite_research import (
    intervention_history,
    next_prerequisite,
    public_object_context,
)
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_prerequisite import ControlLab


CONTROLLER = {"uuid": 7, "name": "controller (unadjusted)",
              "description": "controller (unadjusted)"}
SAMPLE = {"uuid": 8, "name": "sample",
          "description": "sample (in controller (unadjusted) [uuid: 7])"}


def public_view(message="", objects=None):
    return {"accessibleEnvironmentObjects": deepcopy(
                [CONTROLLER, SAMPLE] if objects is None else objects),
            "lastActionMessage": message}


def record(index, action, message, *, success=True, objects=None):
    return {"index": index, "action": action,
            "feedback": {"success": success, "errors": []},
            "observation": public_view(message, objects)}


def test_context_preserves_direct_public_relations_and_copies_preferred_objects():
    controller = {**CONTROLLER, "description": "controller (on bench [uuid: 9])"}
    bench = {"uuid": 9, "name": "bench", "description": "bench"}
    view = {"inventoryObjects": [controller],
            "accessibleEnvironmentObjects": [{**CONTROLLER, "description": "duplicate"}, SAMPLE],
            "nearbyObjects": {"objects": {"north": [bench,
                {"uuid": 10, "name": "wall", "description": "wall"}]}}}
    original = deepcopy(view)
    projected = public_object_context(view, [7])

    assert {obj["uuid"] for obj in projected} == {7, 8, 9}
    assert next(obj for obj in projected if obj["uuid"] == 7) == controller
    assert len(projected) == 3
    projected[0]["description"] = "local edit"
    assert view == original


def test_successful_put_retains_observed_negative_state_without_changing_api_success():
    observed = record(160, {"action": "PUT", "arg1": 8, "arg2": 7},
                      "I put the sample into the controller (empty).")
    original = deepcopy(observed)
    compact = DiscoveryWorldAgent._action_observation(observed)

    assert compact["feedback"] == {"success": True, "errors": []}
    assert compact["public_objects"] == [CONTROLLER, SAMPLE]
    assert compact["lastActionMessage"].endswith("controller (empty).")
    assert compact["public_objects"][0]["name"] == "controller (unadjusted)"
    assert observed == original


def test_dialog_readback_preserves_physical_state_without_fabricated_outcome_label():
    observed = record(162, {"chosen_dialog_option_int": 3}, "Current setting is 1310.")
    observed["observation"]["dialog_box"] = {
        "is_in_dialog": True, "dialogIn": "Current setting is 1310.",
        "dialogOptions": {"9": "Exit"}}
    compact = DiscoveryWorldAgent._action_observation(observed)

    assert compact["lastActionMessage"] == "Current setting is 1310."
    assert compact["public_objects"] == [CONTROLLER, SAMPLE]
    assert compact["dialog_box"] == observed["observation"]["dialog_box"]
    assert all(set(obj) == {"uuid", "name", "description"}
               for obj in compact["public_objects"])


def test_unseen_object_remains_missing_instead_of_becoming_negative_outcome():
    observed = record(164, {"action": "TALK", "arg1": 7},
                      "Could not find object with UUID '7'.", success=False,
                      objects=[{"uuid": 40, "name": "floor", "description": "floor"}])

    assert public_object_context(observed["observation"], [7]) == []
    compact = DiscoveryWorldAgent._action_observation(observed)
    assert compact["public_objects"] == []
    assert compact["feedback"] == observed["feedback"]


def test_intervention_history_keeps_successes_between_failures_in_actual_order():
    unsupported = "That object (controller) is not activatable/deactivatable."
    observations = [
        record(100, {"action": "TALK", "arg1": 7}, "Old setting is 5000."),
        record(151, {"action": "ACTIVATE", "arg1": 7}, unsupported, success=False),
        record(160, {"action": "PUT", "arg1": 8, "arg2": 7}, "Sample inserted."),
        record(162, {"action": "TALK", "arg1": 7}, "Current setting is 1310."),
        record(164, {"action": "ACTIVATE", "arg1": 7}, unsupported, success=False),
    ]
    operation = {"id": "operation_7", "target": 7, "parent_id": "task",
                 "status": "supported", "answer": {"operation": "TALK"},
                 "evidence": [104], "reviewed_through_observation": 104}
    original = deepcopy((observations, operation))
    focus = next_prerequisite(observations, [operation], public_view(), after_index=200)
    history = focus["intervention_history"]

    assert [item["index"] for item in history] == [151, 160, 162, 164]
    assert [item["feedback"]["success"] for item in history] == [False, True, True, False]
    assert history[1]["action"] == {"action": "PUT", "arg1": 8, "arg2": 7}
    assert history[2]["lastActionMessage"] == "Current setting is 1310."
    assert all(item["public_objects"] == [CONTROLLER, SAMPLE] for item in history)
    assert focus["known_operation"] == operation
    assert (observations, operation) == original


def test_nearby_controller_dialog_choices_are_not_assigned_to_the_previous_target():
    visible = [CONTROLLER, SAMPLE,
               {"uuid": 9, "name": "second controller", "description": "second controller"}]
    observations = [
        record(100, {"action": "TALK", "arg1": 7}, "First controls.", objects=visible),
        record(151, {"chosen_dialog_option_int": 3}, "First setting is 1310.", objects=visible),
        record(152, {"action": "TALK", "arg1": 9}, "Second controls.", objects=visible),
        record(153, {"chosen_dialog_option_int": 3}, "Second setting is 870.", objects=visible),
    ]
    for item in observations:
        item["observation"]["dialog_box"] = {"is_in_dialog": True}

    history = intervention_history(observations, 7, after_index=104)

    assert [item["index"] for item in history] == [151]
    assert history[0]["lastActionMessage"] == "First setting is 1310."


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
def test_both_arms_receive_the_same_public_outcome_in_feedback_context_and_saved_evidence(tmp_path, arm):
    class PlacementLab(ControlLab):
        def public_observation(self):
            return {**super().public_observation(), **public_view(self.message)}

        def capabilities(self):
            capabilities = super().capabilities()
            capabilities["actions"]["PUT"] = {"args": ["arg1", "arg2"]}
            return capabilities

        def act(self, action):
            self.actions.append(deepcopy(action))
            self.environment_steps += 1
            self.message = ("Sample inserted." if action["action"] == "PUT"
                            else "Current setting is 1310.")
            return {"action": deepcopy(action), "feedback": {"success": True, "errors": []},
                    "observation": self.public_observation()}

    provider = ScriptedProvider([
        ("query", {"actions": [{"action": "PUT", "arg1": 8, "arg2": 7},
                                {"action": "TALK", "arg1": 7}]}),
        ("finish", {}),
    ])
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm, limits=LIMITS)
    result = agent.run(PlacementLab())
    request = provider.requests[-1]
    context = prepare_code_context(tmp_path, request, schema={}, instructions="")
    saved = json.loads((tmp_path / "action_observations.json").read_text())
    feedback = result["events"][0]["feedback"]["action_results"]

    assert feedback == saved == context["evidence"]["recent_action_results"]
    assert [item["index"] for item in saved] == [0, 1]
    assert all(item["public_objects"] == [CONTROLLER, SAMPLE] for item in saved)
    assert all(item["feedback"]["success"] is True for item in saved)
    assert not result["score"]["completedSuccessfully"]
