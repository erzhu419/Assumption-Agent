"""Existing finite research can test observed responses to chosen controls."""

import re

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_relation_research import research_relation


def test_binary_response_uses_public_state_then_discriminates_and_confirms():
    # The record shape is the public world's shape. Every command succeeds,
    # including a command whose physical response was not publicly observed.
    def record(index, setting, response):
        name = "test actuator" + (f" ({response})" if response is not None else "")
        return {
            "index": index,
            "action": {"chosen_dialog_option_int": 1},
            "feedback": {"success": True, "errors": []},
            "observation": {
                "dialog_box": {"dialogIn": f"The current control setting is: {setting}."},
                "accessibleEnvironmentObjects": [{"uuid": 71, "name": name}],
            },
        }

    setting = Program("feature", name="setting")
    hypotheses = [Program("ge", children=(setting, Program("const", value=threshold))).to_dict()
                  for threshold in (1., 2.)]

    # An authored binder extracts the physical response, not command success
    # or the chosen setting. Unknown state produces no numerical target.
    def bind(records):
        rows = []
        for item in records:
            public = item["observation"]
            names = {obj["name"] for obj in public["accessibleEnvironmentObjects"]
                     if obj["uuid"] == 71}
            if "test actuator (active)" in names:
                target = 1.
            elif "test actuator (inactive)" in names:
                target = 0.
            else:
                continue
            control = float(re.search(r"setting is: ([0-9]+)",
                                      public["dialog_box"]["dialogIn"]).group(1))
            rows.append({"features": {"setting": control}, "target": target,
                         "evidence": [item["index"]]})
        return {"rows": rows, "hypotheses": hypotheses, "tolerance": 0.,
                "probes": [{"id": f"control_{control}", "features": {"setting": control},
                            "actions": [{"chosen_dialog_option_int": option}]}
                           for option, control in enumerate((1., 3.), start=1)]}

    records = [record(0, 0, "inactive"), record(1, 1, None)]
    initial = bind(records)
    assert initial["rows"] == [{"features": {"setting": 0.}, "target": 0., "evidence": [0]}]
    first = research_relation(initial, records)
    assert first["status"] == "inconclusive"
    assert first["diagnosis"] == "discriminating_experiment_available"
    assert first["selected_probe"]["id"] == "control_1.0"
    assert set(first["selected_probe"]["predictions"]) == {0., 1.}

    records.append(record(2, 1, "active"))
    selected = research_relation(bind(records), records, first)
    assert [item["program"] for item in selected["candidates"]] == hypotheses
    assert [item["status"] for item in selected["candidates"]] == ["plausible", "refuted"]
    assert selected["fit_evidence"] == [0]
    assert selected["validation_evidence"] == [2]
    assert selected["selected_program"] == hypotheses[0]
    assert selected["status"] == "inconclusive"
    assert selected["diagnosis"] == "needs_confirmation"
    assert selected["selected_probe"]["id"] == "control_3.0"
    assert selected["selected_probe"]["purpose"] == "confirmation"

    records.append(record(3, 3, "active"))
    confirmed = research_relation(bind(records), records, selected)
    assert confirmed["status"] == "supported"
    assert confirmed["selected_program"] == hypotheses[0]
    assert confirmed["confirmation_evidence"] == [3]
    assert confirmed["selected_probe"] is None
