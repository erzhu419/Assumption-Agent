"""Ground official actions in publicly listed roles, including equipment agents."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab


class PublicActions:
    def __init__(self, *, talk=True):
        self.talk = talk

    def listKnownActions(self):
        actions = {"READ": {"args": ["arg1"], "desc": "read an object"}}
        if self.talk:
            actions["TALK"] = {"args": ["arg1"], "desc": "talk to another agent"}
        return actions

    def additionalActionDescriptionString(self):
        return "Choose one published action."

    def listTeleportLocationsDict(self):
        return {}


@pytest.mark.parametrize("accessible_field", ["accessibleEnvironmentObjects", "inventoryObjects"])
def test_public_agent_role_grounds_talk_for_equipment_and_preserves_accessibility(accessible_field):
    observation = {
        "nearbyAgents": {"list_of_agents": {
            "control cabinet uuid 71": [], "research assistant uuid 89": []}},
        "accessibleEnvironmentObjects": [{"name": "speaking cabinet", "uuid": 99}],
        "inventoryObjects": [],
    }
    observation[accessible_field].append({"uuid": 71, "name": "control cabinet"})
    lab = DiscoveryWorldLab.__new__(DiscoveryWorldLab)
    lab._api = PublicActions()
    lab._observation = deepcopy(observation)

    capabilities = lab.capabilities()

    assert capabilities["interaction_targets"] == [
        {"uuid": 71, "name": "control cabinet", "role": "agent",
         "action": {"action": "TALK", "arg1": 71}, "accessible": True},
        {"uuid": 89, "name": "research assistant", "role": "agent",
         "action": {"action": "TALK", "arg1": 89}, "accessible": False},
    ]
    assert lab.public_observation() == observation
    assert capabilities["actions"]["TALK"] == lab._api.listKnownActions()["TALK"]


def test_role_does_not_invent_an_unpublished_action():
    lab = DiscoveryWorldLab.__new__(DiscoveryWorldLab)
    lab._api = PublicActions(talk=False)
    lab._observation = {"nearbyAgents": {"list_of_agents": {"control cabinet uuid 71": []}},
                        "accessibleEnvironmentObjects": [{"uuid": 71}], "inventoryObjects": []}

    assert lab.capabilities()["interaction_targets"] == []
