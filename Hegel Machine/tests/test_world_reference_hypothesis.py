"""Estimator declarations use the real reference controller and physical answer.

The small laboratory below is an integration fixture, not a DiscoveryWorld
benchmark or evidence of model discovery.
"""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS


PATH = "shared/consolidated_method.py"


def square(feature, name):
    term = {"op": "feature", "name": feature}
    return {"id": name, "basis": [{"op": "mul", "children": [term, deepcopy(term)]}],
            "rationale": "Test whether the measured quantity enters quadratically."}


class ReferenceLab:
    """Three installed references; target activation depends on the actual setting."""

    def __init__(self, *, intercept=100, gain=10, linear=False):
        self.density = {n: float(n) for n in range(1, 6)}
        self.gap = {n: float(n) for n in range(1, 6)}
        self.gap.update({4: 5., 5: 4.})
        self.required = {n: intercept + gain * (x if linear else x * x)
                         for n, x in self.density.items()}
        self.frequencies = {n: self.required[n] if n <= 3 else 500 for n in range(1, 6)}
        self.installed = {1, 2, 3}
        self.inventory = {91, 92}
        self.dialog_number = None
        self.in_dialog = False
        self.last_message = ""
        self.environment_steps = 0
        self.actions = []
        self.completed = self.success = False

    def active(self, number):
        return number in self.installed and abs(self.frequencies[number] - self.required[number]) < .51

    def public_observation(self):
        instruments = [{"uuid": uid, "name": name} for uid, name in
                       ((91, "densitometer"), (92, "microscope"), (93, "signal meter"))]
        objects = [obj for obj in instruments if obj["uuid"] not in self.inventory]
        inventory = [obj for obj in instruments if obj["uuid"] in self.inventory]
        for number in range(1, 6):
            status = ("activated" if self.active(number) else "uncalibrated") if number in self.installed else "no crystal present"
            objects.append({"uuid": 100 + number, "name": "crystal reactor (" + status + ")"})
            crystal = {"uuid": 200 + number, "name": "quantum crystal " + str(number),
                       "description": "quantum crystal " + str(number) + (
                           " (in crystal reactor (" + status + ") [uuid: " + str(100 + number) + "])"
                           if number in self.installed else " (on table [uuid: 300])")}
            (inventory if crystal["uuid"] in self.inventory else objects).append(crystal)
        dialog = {"is_in_dialog": self.in_dialog, "dialogOptions": {}}
        if self.dialog_number is not None:
            dialog["dialogIn"] = ("Reactor #" + str(self.dialog_number) +
                ". The current resonance frequence is: " + str(self.frequencies[self.dialog_number]) + " Hertz.")
        if self.in_dialog:
            dialog["dialogOptions"] = {"1": "Exit"}
            for index, delta in enumerate((1000, 100, 10, 1, -1000, -100, -10, -1), start=2):
                dialog["dialogOptions"][str(index)] = (
                    ("Increase" if delta > 0 else "Decrease") + " frequency by " + str(abs(delta)) + " Hz")
        return {"inventoryObjects": inventory, "accessibleEnvironmentObjects": objects,
                "dialog_box": dialog, "lastActionMessage": self.last_message,
                "taskProgress": [{"description": "Calibrate targets from current reference measurements",
                                  "completed": self.completed}]}

    def capabilities(self):
        return {"teleport_locations": {"reactor lab": {}, "crystal bench": {}, "instruments": {}}}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        self.last_message = ""
        if "chosen_dialog_option_int" in action:
            choice = action["chosen_dialog_option_int"]
            if choice == 1:
                self.in_dialog = False
                self.completed = self.success = all(self.active(n) for n in (4, 5))
            else:
                self.frequencies[self.dialog_number] += (1000, 100, 10, 1, -1000, -100, -10, -1)[choice - 2]
        elif action["action"] == "TALK":
            self.dialog_number = action["arg1"] - 100
            self.in_dialog = True
        elif action["action"] == "PICKUP":
            self.inventory.add(action["arg1"])
            self.installed.discard(action["arg1"] - 200)
        elif action["action"] == "PUT":
            assert action["arg1"] - 200 == action["arg2"] - 100
            self.inventory.remove(action["arg1"])
            self.installed.add(action["arg1"] - 200)
        elif action["action"] == "USE":
            number = action["arg2"] - 200
            readings = {91: "density is approximately " + str(self.density[number]) + " grams.",
                        92: "quantum gap is estimated to be " + str(self.gap[number]) + " nm.",
                        93: "signal magnitude = " + str(self.density[number]) + " units."}
            self.last_message = "You examine quantum crystal " + str(number) + ". The " + readings[action["arg1"]]
        else:
            assert action["action"] in ("TELEPORT_TO_OBJECT", "TELEPORT_TO_LOCATION", "ROTATE_DIRECTION")
        return {"action": deepcopy(action), "feedback": {"success": True},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": self.completed, "completedSuccessfully": self.success,
                "scoreNormalized": float(self.success)}


def prepared(tmp_path, *, source=None, **world):
    lab = ReferenceLab(**world)
    agent = DiscoveryWorldAgent(None, tmp_path, arm="hegel", limits={**LIMITS,
        "environment_steps": 512, "python_seconds_per_task": 120, "task_wall_seconds": 180})
    agent.lab = lab
    agent._question({"id": "task", "question": "Calibrate the two targets"})
    agent._execute("write", {"path": PATH,
        "code": source or Path(recipe.__file__).read_text()}, lab)
    return agent, lab


def submit(agent, lab, hypothesis):
    return agent._execute("run", {"operator": "reference_hypothesis", "path": PATH,
                                 "hypothesis": hypothesis}, lab)


def source_without_estimators(source):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "ESTIMATORS" for target in node.targets))
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def test_declaring_estimator_changes_only_declaration_and_identical_retry_keeps_failure(tmp_path):
    agent, lab = prepared(tmp_path)
    original = (tmp_path / PATH).read_text()
    hypothesis = square("gap", "gap_square")
    first = submit(agent, lab, hypothesis)
    revised = (tmp_path / PATH).read_text()

    assert source_without_estimators(revised) == source_without_estimators(original)
    assert agent.files[PATH] == revised
    assert first["fitted_predictions"][0]["selected_probe"]["predictions"] == pytest.approx([350.])
    assert [(test["family"], test["status"]) for test in first["tests"]] == [("gap_square", "refuted")]
    assert not lab.success
    count = len(lab.actions)
    repeated = submit(agent, lab, hypothesis)
    assert len(lab.actions) == count and repeated["tests"] == []
    assert (tmp_path / PATH).read_text() == revised
    assert agent.costs["model_calls"] == 0


def test_failed_physical_prediction_leaves_new_family_available_and_parent_tests_both_targets(tmp_path):
    agent, lab = prepared(tmp_path)
    failed = submit(agent, lab, square("gap", "gap_square"))
    assert failed["tests"][0]["status"] == "refuted"
    result = submit(agent, lab, square("density", "density_square"))

    assert result["status"] == "completed" and lab.success
    assert [lab.frequencies[n] for n in (4, 5)] == [260, 350]
    questions = {q["id"]: q for q in agent.questions}
    assert questions["reference_test_4_gap_square"]["status"] == "refuted"
    for number in (4, 5):
        answer = questions["reference_test_" + str(number) + "_density_square"]
        assert answer["status"] == "supported"
        assert answer["answer"]["activated"] is True
        assert answer["evidence"]
    state = json.loads((tmp_path / recipe.STATE).read_text())
    assert state["rejected"]["4"] == ["density", "gap", "gap_square"]
    assert state["reference_numbers"] == [1, 2, 3] and state["target_numbers"] == [4, 5]
    assert not agent.llm_inquiries and agent.costs["model_calls"] == 0


def test_revision_requires_new_id_without_erasing_prior_test_or_rewriting_source(tmp_path):
    agent, lab = prepared(tmp_path)
    submit(agent, lab, square("gap", "quadratic"))
    before = (tmp_path / PATH).read_text()
    with pytest.raises(ValueError, match="new id"):
        submit(agent, lab, square("density", "quadratic"))
    assert (tmp_path / PATH).read_text() == before
    question = next(q for q in agent.questions if q["id"] == "reference_test_4_quadratic")
    assert question["status"] == "refuted" and question["evidence"]


def test_new_measurement_is_acquired_and_its_declaration_runs_frozen_on_changed_parameters(tmp_path):
    agent, lab = prepared(tmp_path / "source")
    hypothesis = square("signal", "signal_square")
    hypothesis["measurements"] = {"signal": {
        "instrument": "signal meter", "pattern": r"signal magnitude = ([\d.]+) units", "group": 1}}
    result = submit(agent, lab, hypothesis)
    assert result["status"] == "completed" and lab.success
    assert {row["arg2"] for row in lab.actions if row.get("action") == "USE" and row["arg1"] == 93} == set(range(201, 206))
    assert {"action": "PICKUP", "arg1": 93} in lab.actions

    source = (agent.workspace / PATH).read_text()
    frozen, fresh = prepared(tmp_path / "target", source=source, intercept=40, gain=30)
    execution = frozen._run_procedure(fresh, PATH, max_steps=256)
    assert execution["status"] == "completed" and fresh.success
    assert [fresh.frequencies[n] for n in (4, 5)] == [520, 790]
    assert (frozen.workspace / PATH).read_text() == source
    assert frozen.costs["model_calls"] == 0 and not frozen.llm_inquiries
    questions = {q["id"]: q for q in frozen.questions}
    assert questions["reference_test_4_signal_square"]["answer"]["prediction"] == pytest.approx(520.)


def test_retained_estimator_keeps_existing_linear_case_on_original_family(tmp_path):
    agent, lab = prepared(tmp_path / "source")
    submit(agent, lab, square("density", "density_square"))
    source = (agent.workspace / PATH).read_text()
    old, linear = prepared(tmp_path / "linear", source=source, intercept=60, gain=20, linear=True)
    result = old._run_procedure(linear, PATH, max_steps=256)
    assert result["status"] == "completed" and linear.success
    assert [linear.frequencies[n] for n in (4, 5)] == [140, 160]
    assert set(old.relation_research) == {"reference_fit_density_4", "reference_fit_density_5"}
    assert old.costs["model_calls"] == 0 and (old.workspace / PATH).read_text() == source
