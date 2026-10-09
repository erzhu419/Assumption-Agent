"""Reference response experiments reuse paid records, tuning and restoration.

The selector is fixed here to isolate the real controller and physical analysis;
these small public laboratories are integration fixtures, not discovery evidence.
"""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_reference_response as wrapper
from test_world_reference_hypothesis import PATH, prepared, square, submit
from test_world_response_controller import BoundedReferenceLab, _exhausted_grammar, _state


class ReferenceResearchLab(BoundedReferenceLab):
    def __init__(self, *, wide_reference=False, failed_restore=False):
        self.wide_reference = wide_reference
        self.failed_restore = failed_restore
        self.reference_disturbed = self.reference_broken = False
        super().__init__(intercept=100.3)

    def active(self, number):
        if number == 1:
            if self.reference_broken:
                return False
            if self.wide_reference:
                return number in self.installed and abs(self.frequencies[number] - self.required[number]) < 1.1
        return super().active(number)

    def act(self, action):
        result = super().act(action)
        if self.dialog_number == 1:
            if abs(self.frequencies[1] - self.required[1]) > .9:
                self.reference_disturbed = True
            elif self.failed_restore and self.reference_disturbed:
                self.reference_broken = True
            result["observation"] = self.public_observation()
        return result


def stopped_target(tmp_path, *, lab_type=ReferenceResearchLab, **options):
    agent, _ = prepared(tmp_path, intercept=100.3)
    lab = lab_type(**options)
    agent.lab = lab
    lab.density[5] = 395.5 ** .5
    lab.required[5] = 4077.3
    submit(agent, lab, square("density", "density_square"))
    assert lab.active(4) and not lab.active(5)
    _exhausted_grammar(agent)
    return agent, lab


def fixed_reference_result(state, path, *, selected):
    target = next(n for n in state["target_numbers"] if n not in state["completed"])
    reference = next(row for row in state["research_table"]["references"] if row["crystal"] == 1)
    baseline = reference["frequency"]["value"]
    hypotheses = [{"id": name, "family": "density_square", "candidate_id": "coefficient_0",
                   "claim": "A fixed response interval around the recorded crystal prediction",
                   "lower": -.4, "upper": upper, "target_prediction": 4055.3,
                   "predictions_by_crystal": {"1": baseline, str(target): 4055.3}}
                  for name, upper in (("narrow", .4), ("wide", 1.1))]
    experiment = {"id": "reference_boundary", "number": 1, "restore_frequency": baseline,
                  "prediction": baseline + 1, "family": "density_square", "candidate_id": "coefficient_0",
                  "predictions": {"narrow": False, "wide": True},
                  "outcome": "The installed reference activates at the proposed setting",
                  "procedure": {"path": path, "max_steps": 128}}
    return {"status": "discriminating_experiment_available" if selected else "no_untried_supported_setting",
            "target_number": target, "assumptions": ["A fixed shared response interval is testable"],
            "joint_parameter_search": [],
            "candidates": [{"id": "coefficient_0", "status": "compatible", "family": "density_square"}],
            "hypotheses": hypotheses, "selection": {
                "selected_experiment": experiment if selected else None,
                "hypotheses": deepcopy(hypotheses), "predicted_separation": 1. if selected else 0., "compute_cost": 0},
            "selected_setting": experiment["prediction"] if selected else None,
            "compute_cost": 0, "response_count": len(state["response_settings"]),
            "evidence": sorted({i for response in state["response_settings"] for i in response["evidence"]})}


def test_public_response_call_refreshes_ordinary_paid_records_and_reference_control_grids(tmp_path, monkeypatch):
    agent, lab = stopped_target(tmp_path)
    cached = _state(agent)
    previous_count = len(agent.observations)
    agent._execute("query", {"actions": [
        {"action": "TALK", "arg1": 101}, {"chosen_dialog_option_int": 5},
        {"chosen_dialog_option_int": 1}, {"action": "TALK", "arg1": 102},
        {"chosen_dialog_option_int": 1}]}, lab)
    assert _state(agent)["response_settings"] == cached["response_settings"]
    seen = []

    def capture(state, estimators, path):
        seen.append(deepcopy(state))
        return fixed_reference_result(state, path, selected=False)

    monkeypatch.setattr(wrapper, "propose_response_research", capture)
    before = len(lab.actions)
    result = agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)

    fresh, = seen
    state = _state(agent)
    negative = next(row for row in fresh["response_settings"] if row["crystal"] == 1
                    and row["frequency"] == pytest.approx(111.3) and not row["activated"])
    assert previous_count + 1 in negative["evidence"]
    assert result["response_count"] == len(fresh["response_settings"]) > len(cached["response_settings"])
    assert previous_count + 1 in result["evidence"]
    assert state["response_settings"] == fresh["response_settings"]
    for number in ("1", "2", "3"):
        grid = fresh["control_grids"][number]
        assert grid["step"] == 1 and grid["origin"] == pytest.approx(.3)
        assert grid["lower"] == 0 and grid["upper"] == 10000
    assert len(lab.actions) == before and agent.costs["model_calls"] == 0


@pytest.mark.parametrize("positive", [False, True])
def test_reference_trial_uses_its_own_prediction_and_restores_before_research_continues(tmp_path, monkeypatch, positive):
    agent, lab = stopped_target(tmp_path, wide_reference=positive)
    initial = _state(agent)
    before = len(agent.observations)
    baseline = lab.frequencies[1]
    target_setting = lab.frequencies[5]
    calls = []

    def choose(state, estimators, path):
        calls.append(deepcopy(state))
        if len(calls) > 1:
            assert lab.active(1) and lab.frequencies[1] == pytest.approx(baseline)
            assert state["setting_recoveries"][-1]["restored"]
        return fixed_reference_result(state, path, selected=len(calls) == 1)

    monkeypatch.setattr(wrapper, "propose_response_research", choose)
    result = agent._run_procedure(lab, PATH, max_steps=128)

    assert result["status"] == "method_stopped" and len(calls) == 2
    state = _state(agent)
    test = state["tests"][-1]
    assert test["number"] == 1 and test["prediction"] == pytest.approx(baseline + 1)
    assert test["restore_frequency"] == pytest.approx(baseline)
    assert test["status"] == ("supported" if positive else "refuted")
    assert state["completed"] == initial["completed"]
    assert state["rejected"] == initial["rejected"]
    assert lab.frequencies[1] == pytest.approx(baseline) and lab.active(1)
    assert not lab.active(5) and lab.frequencies[5] == target_setting
    recovery = state["setting_recoveries"][-1]
    assert recovery["number"] == 1 and recovery["restored"]
    assert recovery["evidence"] and min(recovery["evidence"]) >= before
    physical = next(q for q in agent.questions if q["id"] == test["question"])
    feedback = physical["answer"]["response_feedback"]
    assert physical["answer"]["actual_frequency"] == pytest.approx(baseline + 1)
    assert feedback["remaining_hypotheses"] == (["wide"] if positive else ["narrow"])
    assert feedback["refuted_hypotheses"] == (["narrow"] if positive else ["wide"])
    assert all(row["crystal"] == 1 for row in feedback["observed_responses"])
    dialogs = [recipe._dialog(record["observation"]) for record in agent.observations[before:]
               if record["observation"].get("dialog_box", {}).get("is_in_dialog")]
    assert any(number == 1 and frequency == pytest.approx(baseline + 1) for number, frequency in dialogs)
    assert all(number == 1 for number, _ in dialogs)
    assert agent.costs["model_calls"] == 0


def test_failed_reference_restoration_stops_without_restarting_response_research(tmp_path, monkeypatch):
    agent, lab = stopped_target(tmp_path, failed_restore=True)
    initial = _state(agent)
    calls = []

    def choose(state, estimators, path):
        calls.append(deepcopy(state))
        return fixed_reference_result(state, path, selected=True)

    monkeypatch.setattr(wrapper, "propose_response_research", choose)
    result = agent._run_procedure(lab, PATH, max_steps=128)

    assert result["status"] == "method_stopped" and len(calls) == 1
    state = _state(agent)
    assert state["setting_recoveries"][-1]["number"] == 1
    assert state["setting_recoveries"][-1]["restored"] is False
    assert not lab.active(1)
    assert state["completed"] == initial["completed"] and state["rejected"] == initial["rejected"]
    assert "restore" in result["steps"][-1]["feedback"]["reason"].lower()
    assert agent.costs["model_calls"] == 0


class QuantizedReferenceLab(ReferenceResearchLab):
    """Public adjustment feedback uses a grid distinct from its saved baseline."""

    def __init__(self, *, control_origin=0., **options):
        self.control_origin = control_origin
        super().__init__(**options)

    def act(self, action):
        result = super().act(action)
        if self.dialog_number == 1 and action.get("chosen_dialog_option_int") not in (None, 1):
            setting = self.control_origin + round(self.frequencies[1] - self.control_origin)
            # Returning close to the recorded reference restores its original
            # successful decimal reading; this readout is not the step origin.
            self.frequencies[1] = self.required[1] if abs(setting - self.required[1]) < .51 else setting
            result["observation"] = self.public_observation()
        return result


class IntegerMenuReferenceLab(QuantizedReferenceLab):
    """The public in-memory UI uses integer option keys before JSON storage."""

    def public_observation(self):
        view = super().public_observation()
        options = view["dialog_box"]["dialogOptions"]
        view["dialog_box"]["dialogOptions"] = {int(key): label for key, label in options.items()}
        return view


class RoundedReferenceLab(ReferenceResearchLab):
    """Restoration remains on the control grid instead of snapping to baseline."""

    def act(self, action):
        result = super().act(action)
        if self.dialog_number == 1 and action.get("chosen_dialog_option_int") not in (None, 1):
            self.frequencies[1] = float(round(self.frequencies[1]))
            result["observation"] = self.public_observation()
        return result


def test_wrapper_refresh_matches_integer_ui_menu_and_json_reloaded_menu(tmp_path, monkeypatch):
    agent, lab = stopped_target(tmp_path, lab_type=IntegerMenuReferenceLab)
    seen = []

    def capture(state, estimators, path):
        seen.append(deepcopy(state))
        return fixed_reference_result(state, path, selected=False)

    monkeypatch.setattr(wrapper, "propose_response_research", capture)
    agent._execute("query", {"actions": [{"action": "TALK", "arg1": 101},
                                         {"chosen_dialog_option_int": 5}]}, lab)
    adjusted = agent.observations[-1]
    assert recipe._dialog(adjusted["observation"]) == (1, 111.)
    assert 5 in adjusted["observation"]["dialog_box"]["dialogOptions"]
    agent._execute("query", {"actions": [{"chosen_dialog_option_int": 9},
                                         {"chosen_dialog_option_int": 1},
                                         {"action": "TALK", "arg1": 101}]}, lab)
    assert lab.active(1) and lab.frequencies[1] == pytest.approx(110.3)
    before = len(lab.actions)
    memory_result = agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)

    # A saved program request has exactly the same paid records, but JSON
    # changes only the dictionary key representation of the public menu.
    agent.observations = json.loads(json.dumps(agent.observations))
    public_observation = lab.public_observation
    monkeypatch.setattr(lab, "public_observation", lambda: json.loads(json.dumps(public_observation())))
    assert "5" in agent.observations[adjusted["index"]]["observation"]["dialog_box"]["dialogOptions"]
    saved_result = agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)

    memory, saved = seen
    assert saved["control_grids"]["1"]["origin"] == pytest.approx(0.)
    assert memory["control_grids"] == saved["control_grids"]
    assert memory["response_settings"] == saved["response_settings"]
    assert adjusted["index"] in memory_result["evidence"] == saved_result["evidence"]
    assert len(lab.actions) == before and agent.costs["model_calls"] == 0


@pytest.mark.parametrize("origin", [0., .25])
def test_adjustment_feedback_learns_grid_without_baseline_readout_overwriting_it(tmp_path, monkeypatch, origin):
    agent, lab = stopped_target(tmp_path, lab_type=QuantizedReferenceLab, control_origin=origin)
    seen = []

    def capture(state, estimators, path):
        seen.append(deepcopy(state))
        return fixed_reference_result(state, path, selected=False)

    monkeypatch.setattr(wrapper, "propose_response_research", capture)
    agent._execute("query", {"actions": [{"action": "TALK", "arg1": 101},
                                         {"chosen_dialog_option_int": 5}]}, lab)
    adjusted = agent.observations[-1]
    assert recipe._dialog(adjusted["observation"]) == (1, 111. + origin)
    agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)
    assert seen[-1]["control_grids"]["1"]["origin"] == pytest.approx(origin)

    agent._execute("query", {"actions": [{"chosen_dialog_option_int": 9},
                                         {"chosen_dialog_option_int": 1},
                                         {"action": "TALK", "arg1": 101}]}, lab)
    assert lab.active(1) and lab.frequencies[1] == pytest.approx(110.3)
    agent._execute("run", {"operator": "reference_response", "path": PATH}, lab)

    grid = seen[-1]["control_grids"]["1"]
    assert grid["step"] == 1 and grid["origin"] == pytest.approx(origin)
    assert _state(agent)["control_grids"]["1"] == grid
    assert adjusted["index"] in next(row["evidence"] for row in seen[-1]["response_settings"]
                                     if row["crystal"] == 1 and row["frequency"] == 111. + origin)
    assert agent.costs["model_calls"] == 0


@pytest.mark.parametrize("positive", [False, True])
def test_unreachable_reference_request_returns_actual_evidence_restores_and_reselects(tmp_path, monkeypatch, positive):
    agent, lab = stopped_target(tmp_path, lab_type=QuantizedReferenceLab, wide_reference=positive)
    initial = _state(agent)
    before = len(agent.observations)
    calls = []

    def choose(state, estimators, path):
        calls.append(deepcopy(state))
        if len(calls) > 1:
            assert lab.active(1) and lab.frequencies[1] == pytest.approx(110.3)
            assert state["setting_recoveries"][-1]["restored"]
            assert state["control_grids"]["1"]["origin"] == pytest.approx(0.)
        return fixed_reference_result(state, path, selected=len(calls) == 1)

    monkeypatch.setattr(wrapper, "propose_response_research", choose)
    result = agent._run_procedure(lab, PATH, max_steps=128)

    assert result["status"] == "method_stopped" and len(calls) == 2
    state = _state(agent)
    new_tests = state["tests"][len(initial["tests"]):]
    assert len(new_tests) == 1 and new_tests[0]["status"] == "inconclusive"
    assert new_tests[0]["prediction"] == pytest.approx(111.3)
    physical = next(q for q in agent.questions if q["id"] == new_tests[0]["question"])
    assert physical["status"] == "inconclusive" and physical["evidence"]
    feedback = physical["answer"].get("response_feedback", physical["answer"])
    actual = [row for row in feedback["observed_responses"] if row["frequency"] == 111.]
    assert actual and all(row["activated"] is positive for row in actual)
    assert not any(row["frequency"] == pytest.approx(111.3) for row in feedback["observed_responses"])
    assert all(index >= before for index in physical["evidence"])
    assert not state.get("test") and lab.frequencies[1] == pytest.approx(110.3) and lab.active(1)
    assert state["completed"] == initial["completed"] and state["rejected"] == initial["rejected"]
    assert len(state["response_research_history"]) == len(initial.get("response_research_history", [])) + 2
    assert agent.costs["model_calls"] == 0


def test_nearest_reachable_active_reference_restores_with_actual_frequency_and_reselects(tmp_path, monkeypatch):
    agent, lab = stopped_target(tmp_path, lab_type=RoundedReferenceLab)
    initial = _state(agent)
    baseline = lab.frequencies[1]
    before = len(agent.observations)
    calls = []

    def choose(state, estimators, path):
        calls.append(deepcopy(state))
        if len(calls) > 1:
            assert lab.active(1) and lab.frequencies[1] == 110.
            assert state["setting_recoveries"][-1]["restored"]
        result = fixed_reference_result(state, path, selected=len(calls) == 1)
        if len(calls) == 1:
            # The experiment itself is reachable; only the old successful
            # decimal baseline is no longer an exact control setting.
            result["selection"]["selected_experiment"]["prediction"] = 111.
            result["selected_setting"] = 111.
        return result

    monkeypatch.setattr(wrapper, "propose_response_research", choose)
    result = agent._run_procedure(lab, PATH, max_steps=128)

    assert result["status"] == "method_stopped" and len(calls) == 2
    state = _state(agent)
    assert state["tests"][-1]["status"] == "refuted"
    recovery = state["setting_recoveries"][-1]
    assert recovery["restored"] and recovery["number"] == 1
    assert recovery["frequency"] == baseline == pytest.approx(110.3)
    assert recovery["actual_frequency"] == lab.frequencies[1] == 110.
    assert recovery["crystal"] == 201 and recovery["reactor"] == 101
    assert recovery["evidence"] and min(recovery["evidence"]) >= before
    assert not state.get("recovery") and lab.active(1)
    assert state["completed"] == initial["completed"] and state["rejected"] == initial["rejected"]
    assert agent.costs["model_calls"] == 0
