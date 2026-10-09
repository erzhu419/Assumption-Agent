"""Exercise the real public dialog interface, without model calls or task answers."""

from pathlib import Path

import pytest

from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab


RUNTIME = Path.home() / ".cache/hegel-machine/discoveryworld"


@pytest.mark.skipif(not (RUNTIME / "discoveryworld/DiscoveryWorldAPI.py").is_file(),
                    reason="DiscoveryWorld runtime is not installed")
@pytest.mark.parametrize("entry", ["USE", "TALK"])
def test_public_dialog_entry_choice_and_exit(tmp_path, entry):
    task = {"scenario_name": "Reactor Lab", "difficulty": "Normal", "random_seed": 0}
    with DiscoveryWorldLab(task, RUNTIME, tmp_path) as lab:
        observed = lab.act({"action": "TELEPORT_TO_LOCATION", "arg1": "reactor lab"})
        objects = [obj for group in observed["observation"]["nearbyObjects"]["objects"].values()
                   for obj in group]
        reactor = next(obj for obj in objects if obj["name"] == "crystal reactor (activated)")
        observed = lab.act({"action": "TELEPORT_TO_OBJECT", "arg1": reactor["uuid"]})
        crystal = next(obj for obj in observed["observation"]["accessibleEnvironmentObjects"]
                       if obj["name"].startswith("quantum crystal"))
        action = {"action": entry, "arg1": reactor["uuid"]}
        if entry == "USE":
            action["arg2"] = crystal["uuid"]

        opened = lab.act(action)
        assert opened["feedback"]["success"]
        dialog = opened["observation"]["dialog_box"]
        assert dialog["is_in_dialog"]
        options = dialog["dialogOptions"]
        change = next(key for key, value in options.items() if value == "Increase frequency by 100 Hz")
        changed = lab.act({"chosen_dialog_option_int": int(change)})
        assert changed["feedback"]["success"]
        updated = changed["observation"]["dialog_box"]
        assert updated["is_in_dialog"] and updated["dialogIn"] != dialog["dialogIn"]

        leave = next(key for key, value in updated["dialogOptions"].items() if value == "Exit")
        closed = lab.act({"chosen_dialog_option_int": int(leave)})
        assert closed["feedback"]["success"]
        assert closed["observation"]["dialog_box"] == {"is_in_dialog": False}
        # This must now accept a normal action rather than request another choice.
        moved = lab.act({"action": "TELEPORT_TO_LOCATION", "arg1": "reactor lab"})
        assert moved["feedback"]["success"]
