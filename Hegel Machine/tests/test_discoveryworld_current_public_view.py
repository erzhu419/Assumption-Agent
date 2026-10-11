"""Current and historical public views expose the same existing object projection."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_prerequisite_research import public_object_context
from test_discoveryworld_agent import LIMITS, InstrumentLab, ScriptedProvider


class PublicStateLab(InstrumentLab):
    def __init__(self):
        super().__init__()
        self.active = True

    def public_observation(self):
        state = "activated" if self.active else "uncalibrated"
        return {**super().public_observation(),
            "inventoryObjects": [{"uuid": 91, "name": "densitometer"}],
            "accessibleEnvironmentObjects": [
                {"uuid": 104, "name": f"crystal reactor ({state})"},
                {"uuid": 204, "name": "quantum crystal 4",
                 "description": f"quantum crystal 4 (in crystal reactor ({state}) [uuid: 104])"}],
            "lastActionMessage": "Control adjusted.",
            "dialog_box": {"is_in_dialog": True,
                "dialogIn": "Reactor #4. The current resonance frequence is: 104 Hertz.",
                "dialogOptions": {"1": "Exit"}}}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        self.active = not self.active
        return {"action": deepcopy(action), "feedback": {"success": True},
                "observation": self.public_observation()}


# The unmodified analysis written in changed_mechanism/repeat_03/hegel.
RECORDED_ANALYSIS = """def answer(public_spec, observations, memory):
    # The live public state identifies the reactor by UUID 104, not by a 'Reactor #4' name.
    state = public_spec.get('current_observation', public_spec) if isinstance(public_spec, dict) else public_spec
    def get(o,k,d=None): return o.get(k,d) if isinstance(o,dict) else getattr(o,k,d)
    dialog=get(state,'dialog_box',{}) or {}
    txt=str(get(dialog,'dialogIn','') or '')
    objs=get(state,'public_objects',[]) or []
    reactor=next((o for o in objs if get(o,'uuid')==104),None)
    if reactor is None:
        reactor=next((o for o in objs if 'reactor #4' in str(get(o,'name','')).lower()),None)
    name=str(get(reactor,'name','unknown'))
    if '104' not in txt:
        return {'status':'inconclusive','answer':{'reason':'Current readout does not verify 104 Hz','reactor_state':name},'evidence':[270]}
    active='activated' in name.lower() and 'uncalibrated' not in name.lower()
    return {'status':'supported' if active else 'refuted','answer':{'setting_hz':104,'activated':active,'reactor_state':name,'scope':'After the recorded 105-Hz setting, option 9 decreased frequency to 104 Hz and the public reactor state was read.'},'evidence':[243,270]}
"""


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_recorded_analysis_reads_actual_current_object_without_changing_its_source(tmp_path, arm):
    lab = PublicStateLab()
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    agent.lab = lab
    raw = deepcopy(lab.public_observation())
    namespace = {}
    exec(RECORDED_ANALYSIS, namespace)
    old = namespace["answer"]({"current_observation": raw}, [], {})
    assert old["status"] == "refuted" and old["answer"]["reactor_state"] == "unknown"

    agent._execute("write", {"path": "recorded_analysis.py", "code": RECORDED_ANALYSIS}, lab)
    result = agent._program("recorded_analysis.py", "answer")

    assert result["status"] == "supported"
    assert result["answer"]["reactor_state"] == "crystal reactor (activated)"
    assert (tmp_path / "recorded_analysis.py").read_text() == RECORDED_ANALYSIS
    current = agent._spec()["current_observation"]
    assert {key: current[key] for key in raw} == raw
    assert current["public_objects"] == public_object_context(raw)
    assert current["public_relations"] == [{"subject": 204, "relation": "in", "object": 104,
        "text": "(in crystal reactor (activated) [uuid: 104])"}]
    assert json.loads((tmp_path / "current_observation.json").read_text()) == current
    assert lab.public_observation() == raw and not lab.actions
    assert agent.costs["model_calls"] == 0


def test_current_projection_refreshes_after_action_without_rewriting_historical_records(tmp_path):
    lab = PublicStateLab()
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    agent.lab = lab
    agent._execute("query", {"actions": [{"chosen_dialog_option_int": 9}]}, lab)
    first = deepcopy(agent.observations[0])
    first_view = agent._spec()["current_observation"]
    assert first_view["public_objects"][1]["name"] == "crystal reactor (uncalibrated)"

    agent._execute("query", {"actions": [{"chosen_dialog_option_int": 5}]}, lab)

    current = agent._request()["current_observation"]
    assert current["public_objects"][1]["name"] == "crystal reactor (activated)"
    assert json.loads((tmp_path / "current_observation.json").read_text()) == current
    assert agent.observations[0] == first
    assert json.loads((tmp_path / "laboratory_observations.json").read_text())[0] == first
    assert "public_objects" not in first["observation"]
    assert agent._action_observation(first)["public_objects"] == first_view["public_objects"]
    assert "activated" not in current and "physical_outcome" not in current


def test_missing_current_object_stays_missing_and_projection_cannot_mutate_lab(tmp_path):
    lab = PublicStateLab()
    raw = lab.public_observation()
    raw["accessibleEnvironmentObjects"] = []
    lab.public_observation = lambda: raw
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    agent.lab = lab
    current = agent._spec()["current_observation"]

    assert current["public_objects"] == [{"uuid": 91, "name": "densitometer"}]
    assert current["public_relations"] == []
    current["public_objects"][0]["name"] = "local edit"
    current["inventoryObjects"][0]["name"] = "local raw edit"
    assert lab.public_observation()["inventoryObjects"][0]["name"] == "densitometer"


# Unmodified call 13 in response_current_public_view_live_20261010,
# changed_mechanism/repeat_03/hegel. Exact archived snapshots are replayed separately.
RECORDED_HISTORICAL_ANALYSIS = """def answer(public_spec, observations, memory):
    # The procedure reached 500 Hz at record 263. Judge the live public reactor state, not the control command alone.
    r = next((r for r in observations if r.get('index') == 263), None)
    if r is None:
        return {'status':'inconclusive','answer':{'reason':'No record of the reactor at 500 Hz.'},'evidence':[]}
    ob = r.get('observation', {})
    msg = ob.get('dialog_box', {}).get('dialogIn', '')
    objs = ob.get('public_objects', [])
    reactor = next((o for o in objs if o.get('uuid') == 105), {})
    if '500 Hertz' not in msg:
        return {'status':'inconclusive','answer':{'reason':'The actual dialog does not verify 500 Hz.','dialog':msg},'evidence':[263]}
    state = reactor.get('name', '')
    if 'uncalibrated' in state:
        return {'status':'refuted','answer':{'condition':'crystal 5 in reactor 5 at 500 Hz','activated':False,'observed_reactor_state':state,'scope':'This setting only; other frequencies remain untested.'},'evidence':[263]}
    if 'activated' in state:
        return {'status':'supported','answer':{'condition':'crystal 5 in reactor 5 at 500 Hz','activated':True,'observed_reactor_state':state,'scope':'This setting only; not a general calibration law.'},'evidence':[263]}
    return {'status':'inconclusive','answer':{'reason':'No explicit activation state in the public object name.','observed_reactor_state':state},'evidence':[263]}
"""


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_recorded_history_analysis_reads_paid_state_without_rewriting_raw_evidence(tmp_path, arm):
    lab = PublicStateLab()
    raw = lab.public_observation()
    raw["accessibleEnvironmentObjects"] = [
        {"uuid": 105, "name": "crystal reactor (uncalibrated)"},
        {"uuid": 205, "name": "quantum crystal 5",
         "description": "quantum crystal 5 (in crystal reactor (uncalibrated) [uuid: 105])"}]
    raw["dialog_box"]["dialogIn"] = "Reactor #5. The current resonance frequence is: 500 Hertz."
    record = {"index": 263, "action": {"action": "TALK", "arg1": 105},
              "feedback": {"success": True}, "observation": raw}
    original = deepcopy(record)
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    agent.lab, agent.observations = lab, [record]
    namespace = {}
    exec(RECORDED_HISTORICAL_ANALYSIS, namespace)
    old = namespace["answer"]({}, [{**record, **agent._action_observation(record)}], {})
    assert old["status"] == "inconclusive"

    agent._execute("write", {"path": "recorded_history.py", "code": RECORDED_HISTORICAL_ANALYSIS}, lab)
    result = agent._program("recorded_history.py", "answer")

    assert result["status"] == "refuted" and result["answer"]["activated"] is False
    assert result["evidence"] == [263]
    assert (tmp_path / "recorded_history.py").read_text() == RECORDED_HISTORICAL_ANALYSIS
    assert agent.observations == [original]
    assert json.loads((tmp_path / "laboratory_observations.json").read_text()) == [original]
    projected = agent._program_observation(record)
    assert {o["uuid"] for o in projected["public_objects"]} == {105, 205}
    assert {o["uuid"] for o in projected["observation"]["public_objects"]} == {91, 105, 205}
    assert projected["observation"]["public_relations"] == projected["public_relations"]
    assert projected["observation"] == agent._public_observation(raw)
    projected["observation"]["public_objects"][1]["name"] = "local edit"
    assert record == original
    assert not lab.actions and agent.costs["model_calls"] == 0


def test_history_projection_does_not_fill_an_absent_dialog_or_object(tmp_path):
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    raw = {"accessibleEnvironmentObjects": [{"uuid": 205, "name": "quantum crystal 5"}],
           "dialog_box": {"is_in_dialog": False, "dialogOptions": {}}}
    record = {"index": 0, "action": {"chosen_dialog_option_int": 1},
              "feedback": {"success": True}, "observation": raw}
    view = agent._program_observation(record)["observation"]
    assert view["public_objects"] == raw["accessibleEnvironmentObjects"]
    assert view["public_relations"] == []
    assert view["dialog_box"] == raw["dialog_box"]
    assert "dialogIn" not in view["dialog_box"]
    assert "activated" not in view and "physical_outcome" not in view
