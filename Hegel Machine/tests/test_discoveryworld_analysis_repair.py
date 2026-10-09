"""Paid public counterexamples can repair authored analysis without rerunning a test."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import ScriptedProvider
from test_discoveryworld_child_return import reviewed_parent
from test_discoveryworld_parent_research import question


def public_record(index=0, *, relation="in", visible=True):
    # DiscoveryWorld can expose a container state only inside its content's
    # description. UUID 8 need not have a separate visible-object entry.
    objects = [{"uuid": 9, "name": "quantum crystal",
                "description": f"A crystal ({relation} crystal reactor (uncalibrated) [uuid: 8])"}]
    return {"index": index, "action": {"action": "TALK", "arg1": 8},
            "feedback": {"success": True},
            "observation": {"accessibleEnvironmentObjects": objects if visible else [],
                "lastActionMessage": "Set resonance frequency to 886 Hz.",
                "dialog_box": {"is_in_dialog": True, "dialog_options": {"1": "Exit"}},
                "taskProgress": [{"description": "Explain the response", "completed": False}]}}


@pytest.mark.parametrize("relation", ["in", "on"])
def test_public_containment_state_is_exposed_without_a_synthetic_parent_object(relation):
    record = public_record(relation=relation)
    saved = deepcopy(record)
    compact = DiscoveryWorldAgent._action_observation(record)

    assert compact["public_objects"] == record["observation"]["accessibleEnvironmentObjects"]
    assert len(compact["public_relations"]) == 1
    link = compact["public_relations"][0]
    assert {key: link[key] for key in ("subject", "relation", "object")} == {
        "subject": 9, "relation": relation, "object": 8}
    assert "crystal reactor (uncalibrated) [uuid: 8]" in link["text"]
    assert set(link) == {"subject", "relation", "object", "text"}
    assert record == saved


def test_missing_public_object_does_not_generate_absence_or_physical_outcome():
    compact = DiscoveryWorldAgent._action_observation(public_record(visible=False))
    assert compact["public_objects"] == compact["public_relations"] == []
    assert "activated" not in compact and "physical_outcome" not in compact
    assert compact["feedback"]["success"] is True


ECHO = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': observations, 'evidence': [0]}
"""


def test_both_arms_receive_the_same_augmented_program_records_and_intact_raw_data(tmp_path):
    received = []
    for arm in ("hegel", "code_archive"):
        agent, lab = reviewed_parent(tmp_path / arm)
        agent.agent_arm = arm
        agent.observations = [public_record()]
        before = deepcopy(agent.observations)
        agent.research_focus = None
        agent._execute("write", {"path": "echo.py", "code": ECHO}, lab)
        payload = agent._program("echo.py", "answer")["answer"]
        received.append(payload)
        assert payload[0]["observation"] == before[0]["observation"]
        assert payload[0]["lastActionMessage"] == before[0]["observation"]["lastActionMessage"]
        assert payload[0]["dialog_box"] == before[0]["observation"]["dialog_box"]
        assert payload[0]["public_relations"][0]["object"] == 8
        assert agent.observations == before
        assert json.loads((agent.workspace / "laboratory_observations.json").read_text()) == before
    assert received[0] == received[1]


PUBLIC_ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    relevant = [relation['text'] for relation in record['public_relations']
                if relation['subject'] == 9 and relation['relation'] == 'in'
                and relation['object'] == 8]
    state = relevant[-1] if relevant else None
    rejected = state is not None and 'uncalibrated' in state
    return {'status': 'refuted' if rejected else 'inconclusive',
            'answer': {'public_state': state, 'tested_setting': record['lastActionMessage']},
            'evidence': [record['index']]}
"""


@pytest.mark.parametrize("visible, status", [(True, "refuted"), (False, "inconclusive")])
def test_authored_analysis_distinguishes_observed_uncalibrated_state_from_missing_state(tmp_path, visible, status):
    agent, lab = reviewed_parent(tmp_path)
    agent.observations.append(public_record(1, visible=visible))
    child = agent._question({"id": "response", "question": "Does 886 Hz activate this reactor?",
                             "parent_id": "task"})
    child["acquired_observations"] = [1]
    result = agent._execute("evaluate", {"path": "repaired_analysis.py", "question_id": "response",
                                          "code": PUBLIC_ANALYSIS}, lab)
    assert result["question"]["status"] == status
    assert result["question"]["evidence"] == [1]
    assert lab.environment_steps == 1
    if visible:
        assert "uncalibrated" in child["answer"]["public_state"]
    else:
        assert child["answer"]["public_state"] is None


def test_same_paid_counterexample_revises_answer_and_returns_to_retained_parent(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    parent = question(agent)
    parent_path = parent["parent_program_path"]
    retained_parent = (tmp_path / parent_path).read_text()
    agent.observations.append(public_record(1))
    child = agent._question({"id": "response", "question": "Does 886 Hz activate this reactor?",
                             "parent_id": "task"})
    child["acquired_observations"] = [1]
    experiment_path = "shared/test_886.py"
    experiment_source = "def design(spec, observations, memory):\n    return {'tested_setting': 886}\n"
    agent._execute("write", {"path": experiment_path, "code": experiment_source}, lab)
    child["design_path"] = child["experiment_path"] = experiment_path
    agent._record_answer(child, {"status": "inconclusive", "answer": "Missing direct reactor entry",
                                 "evidence": [1]})
    parent["reviewed_child_answers"] = agent._child_answer_snapshots(parent)
    assert not agent._request().get("research_focus")
    before = deepcopy(agent.observations)
    actions = deepcopy(lab.actions)

    result = agent._execute("evaluate", {"path": "shared/repaired_response.py", "question_id": "response",
                                          "code": PUBLIC_ANALYSIS}, lab)

    assert result["analysis_is_agent_authored"]
    assert child["status"] == "refuted" and child["evidence"] == [1]
    assert child["reviewed_through_observation"] == 1
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "parent_research" and focus["stage"] == "child_return"
    assert focus["changed_child_ids"] == ["response"]
    assert agent.observations == before and lab.actions == actions
    assert (tmp_path / parent_path).read_text() == retained_parent
    assert (tmp_path / experiment_path).read_text() == experiment_source
    assert child["design_path"] == experiment_path


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_failed_repeat_retains_earlier_frozen_experiment_and_its_paid_indices(tmp_path, arm, adaptive):
    if adaptive:
        from test_discoveryworld_adaptive_contrast import PATH, QUESTION, SOURCE, prepared
        agent, lab = prepared(tmp_path, arm=arm)
        later_source = SOURCE  # An unchanged design also keeps distinct execution histories.
        expected = [[0, 1, 2], [3, 4]]
    else:
        from test_discoveryworld_contrast import PATH, QUESTION, SOURCE, prepared
        agent, lab = prepared(tmp_path, arm=arm, outcome=True)
        later_source = SOURCE.replace("The extra operation is necessary", "A revised condition is necessary")
        expected = [[1], [2]]

    class RepeatProvider(ScriptedProvider):
        def next_action(self, request):
            if len(self.requests) == 1:
                if adaptive:
                    lab.fail_adjustment = True
                else:
                    lab.failed = True
            return super().next_action(request)

    agent.provider = RepeatProvider([
        ("run", {"operator": "contrast", "path": PATH, "code": SOURCE, "question": QUESTION}),
        ("run", {"operator": "contrast", "path": PATH, "code": later_source, "question": QUESTION}),
        ("finish", {"lesson": "Keep the successful trial and inspect the failed repeat"}),
    ])
    result = agent.run(lab)
    assert result["status"] == "agent_finished"
    events = [row for row in result["events"] if row["action"] == "run"]
    assert len(events) == 2 and events[0]["feedback"]["operator"] == "contrast"
    trials = agent._state()["experiment_trials"]
    assert len(trials) == 2
    assert [row["observation_indices"] for row in trials] == expected
    for index, (trial, event) in enumerate(zip(trials, events)):
        assert trial["event_index"] == index
        assert trial["question_id"] == QUESTION["id"] and trial["path"] == PATH
        assert trial["hypotheses"] == event["feedback"]["hypotheses"]
        assert trial["selected_experiment"] == event["feedback"]["selected_experiment"]
        assert trial["execution_status"]
    assert trials[0]["execution_status"] != trials[1]["execution_status"]
    active = question(agent, QUESTION["id"])["experiment_choice"]
    assert active["started_before_observation"] == expected[1][0]
    assert active["hypotheses"] == trials[1]["hypotheses"]
    view = json.loads(agent._request()["files"]["method_decision.json"])
    entry = next(row for row in view["questions"] if row["id"] == QUESTION["id"])
    assert entry["experiment_trials"] == trials
    assert next(row for row in view["questions"] if row["id"] == "task")["experiment_trials"] == []


def test_parent_procedure_records_executed_child_contrast_without_counting_its_proposals(tmp_path):
    from test_discoveryworld_contrast import PATH, QUESTION, SOURCE, prepared

    agent, lab = prepared(tmp_path, arm="code_archive", outcome=True)
    proposal = {"action": "run", "arguments": {"operator": "contrast", "path": PATH,
        "code": SOURCE, "question": QUESTION}, "planned_only": {
            "operator": "contrast", "question_id": "not_executed", "path": "unused.py",
            "hypotheses": [{"id": "guess", "claim": "Unexecuted proposal"}],
            "selected_experiment": {"id": "planned"},
            "experiment": {"observation_indices": [999]}}}
    parent_source = ("def select_actions(spec, observations, memory):\n"
        "    if memory['program_results']:\n"
        "        return {'stop': 'The actual child contrast returned'}\n"
        f"    return {proposal!r}\n")
    agent._execute("write", {"path": "shared/parent_contrast.py", "code": parent_source}, lab)
    agent.provider = ScriptedProvider([
        ("run", {"operator": "procedure", "path": "shared/parent_contrast.py", "max_steps": 3}),
        ("finish", {"lesson": "Inspect the returned physical trial"}),
    ])
    result = agent.run(lab)

    assert result["status"] == "agent_finished"
    execution = result["events"][0]["feedback"]
    assert execution["operator"] == "procedure" and execution["status"] == "method_stopped"
    actual = execution["steps"][0]["feedback"]
    assert actual["operator"] == "contrast" and actual["experiment"]["observation_indices"] == [1]
    assert execution["steps"][0]["proposal"]["planned_only"]["experiment"]["observation_indices"] == [999]
    trials = agent._state()["experiment_trials"]
    assert len(trials) == 1
    assert trials[0]["event_index"] == 0
    assert trials[0]["step_path"] == ["steps", 0, "feedback"]
    assert trials[0]["question_id"] == QUESTION["id"] and trials[0]["path"] == PATH
    assert trials[0]["observation_indices"] == [1]
    assert trials[0]["selected_experiment"] == actual["selected_experiment"]
    assert lab.environment_steps == 2
