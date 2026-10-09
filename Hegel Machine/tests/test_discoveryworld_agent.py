"""Executable research contracts, not DiscoveryWorld or LLM performance evidence."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.code_provider_context import prepare_code_context


LIMITS = {
    "environment_steps": 20,
    "score_checkpoints_environment_steps": [1, 2],
    "max_model_calls": 30,
    "task_wall_seconds": 30,
    "provider_request_timeout_seconds": 5,
    "python_seconds_per_call": 5,
    "python_seconds_per_task": 20,
}


class ScriptedProvider:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def next_action(self, request):
        self.requests.append(deepcopy(request))
        action, arguments = next(self.steps)
        return {"action": action, "arguments": deepcopy(arguments),
                "usage": {"total_tokens": 1}}


class InstrumentLab:
    """An instrument must be read before choosing its matching intervention."""

    def __init__(self, readings=(3,)):
        self.readings = iter(readings)
        self.environment_steps = 0
        self.actions = []
        self.completed = self.success = False

    def public_observation(self):
        return {"taskProgress": [{"description": "Read the instrument and tune to twice its reading",
                                  "completed": self.completed}],
                "world_steps": self.environment_steps}

    def capabilities(self):
        return {"actions": ["MEASURE", "TUNE"]}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        if action["action"] == "MEASURE":
            feedback = {"reading": next(self.readings)}
        else:
            self.completed = True
            self.success = action["arg1"] == 6
            feedback = {"message": "Intervention performed"}
        return {"action": deepcopy(action), "feedback": feedback,
                "observation": self.public_observation()}

    def score(self):
        return {"completed": self.completed, "completedSuccessfully": self.success,
                "scoreNormalized": float(self.success),
                "criticalHypotheses": ["EVALUATOR_ONLY_GOLD_RULE"]}


MEASURE = """def select_actions(spec, observations, memory):
    return [{'action': 'MEASURE'}]
"""

ANALYZE = """def answer(spec, observations, memory):
    index = len(observations) - 1
    reading = observations[index]['feedback']['reading']
    return {'status': 'supported' if reading > 0 else 'refuted',
            'answer': {'gain': reading}, 'evidence': [index]}
"""

PARENT_ANALYSIS = """def answer(spec, observations, memory):
    child = memory['answers']['instrument']
    return {'status': child['status'], 'answer': child['answer'],
            'evidence': child['evidence']}
"""


def run_steps(tmp_path, steps, *, arm="hegel", readings=(3,)):
    provider = ScriptedProvider(steps)
    lab = InstrumentLab(readings)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm, limits=LIMITS)
    return agent.run(lab), provider, lab


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
def test_executed_child_answer_changes_parent_action_and_success_needs_no_manifest(tmp_path, arm):
    parent_program = """def select_actions(spec, observations, memory):
    child = memory['answers']['instrument']
    assert child['status'] == 'supported'
    assert child['evidence'] == [0]
    return [{'action': 'TUNE', 'arg1': 2 * child['answer']['gain']}]
"""
    result, provider, lab = run_steps(tmp_path, [
        ("run", {"operator": "question", "id": "parent", "question": "How should it be tuned?"}),
        ("write", {"path": "shared/measure.py", "code": MEASURE}),
        ("query", {"path": "shared/measure.py", "question": {
            "id": "instrument", "question": "What does the instrument report?", "parent_id": "parent",
            "analysis_path": "analysis.py"}}),
        ("write", {"path": "analysis.py", "code": ANALYZE}),
        ("evaluate", {"path": "analysis.py", "question_id": "instrument"}),
        ("write", {"path": "shared/tune.py", "code": parent_program}),
        ("query", {"path": "shared/tune.py", "question": {"id": "parent"}}),
    ], arm=arm)

    assert lab.actions == [{"action": "MEASURE"}, {"action": "TUNE", "arg1": 6}]
    child = next(q for q in result["research"]["questions"] if q["id"] == "instrument")
    assert child["opened_before_observation"] == 0
    assert child["answer"] == {"gain": 3}
    assert result["research"]["child_answers_returned"] == 1
    assert result["status"] == "completed" and result["score"]["completedSuccessfully"]
    assert not (tmp_path / "shared/knowledge.json").exists()
    assert not any(event["action"] == "finish" for event in result["events"])

    # Check both pre- and post-acquisition contexts, including recorded checkpoints.
    assert result["score_checkpoints"][0]["criticalHypotheses"]
    for request in provider.requests:
        rendered = json.dumps(request)
        assert "EVALUATOR_ONLY_GOLD_RULE" not in rendered
        assert "criticalHypotheses" not in rendered
        assert "scoreNormalized" not in rendered
    records = json.loads((tmp_path / "laboratory_observations.json").read_text())
    assert records[0]["feedback"]["reading"] == 3

    if arm == "hegel":
        # The decision view retains the unresolved parent and its actual child
        # answer; it must not equate a successful analysis with solving the task.
        pending = json.loads(provider.requests[3]["files"]["method_decision.json"])
        parent = next(q for q in pending["questions"] if q["id"] == "parent")
        assert "instrument" in parent["pending_dependencies"]
        returned = json.loads(provider.requests[5]["files"]["method_decision.json"])
        child = next(q for q in returned["questions"] if q["id"] == "instrument")
        assert child["answer"] == {"gain": 3} and child["evidence"] == [0]
        assert child["unreviewed_observation_indices"] == []
        assert "answer_semantics" in returned
    else:
        assert all("method_decision.json" in r["files"] for r in provider.requests)
        assert all("research_focus" not in r for r in provider.requests)


def supported_method_steps():
    return [
        ("run", {"operator": "question", "id": "root", "question": "Does the explanation hold?"}),
        ("run", {"operator": "question", "id": "parent", "question": "Can the instrument guide research?",
                 "parent_id": "root", "depends_on": ["instrument"]}),
        ("query", {"actions": [{"action": "MEASURE"}], "question": {
            "id": "instrument", "question": "Is the instrument usable?", "parent_id": "parent",
            "analysis_path": "child_analysis.py"}}),
        ("write", {"path": "child_analysis.py", "code": ANALYZE}),
        ("evaluate", {"path": "child_analysis.py", "question_id": "instrument"}),
        ("write", {"path": "parent_analysis.py", "code": PARENT_ANALYSIS}),
        ("evaluate", {"path": "parent_analysis.py", "question_id": "parent"}),
        ("write", {"path": "shared/method.py", "code": MEASURE}),
        ("run", {"operator": "method", "id": "method", "path": "shared/method.py",
                 "requires": ["parent"], "applicability": "Instrument support remains valid"}),
    ]


def test_counterevidence_reopens_ancestors_and_blocks_the_previously_executable_method(tmp_path):
    result, provider, lab = run_steps(tmp_path, [*supported_method_steps(),
        ("query", {"method_id": "method", "question": {"id": "instrument"}}),
        ("evaluate", {"path": "child_analysis.py", "question_id": "instrument"}),
        ("query", {"method_id": "method"}),
        ("finish", {"lesson": "The new measurement refuted the instrument support"}),
    ], readings=(3, -3))

    assert provider.requests[9]["research"]["methods"][0]["available"]
    assert len(lab.actions) == 2  # The third attempt must not acquire another observation.
    questions = {q["id"]: q for q in result["research"]["questions"]}
    assert questions["instrument"]["status"] == "refuted"
    assert questions["instrument"]["evidence"] == [1]
    assert questions["parent"]["status"] == questions["root"]["status"] == "reopened"
    assert not result["research"]["methods"][0]["available"]
    assert "error" in result["events"][-2]["feedback"]
    assert not result["score"]["completedSuccessfully"]  # finish is not proof of success.

    # A new acquisition has not yet been interpreted: expose it for review,
    # preserving the supported answer until the authored analysis changes it.
    before_review = json.loads(provider.requests[10]["files"]["method_decision.json"])
    child = next(q for q in before_review["questions"] if q["id"] == "instrument")
    assert child["status"] == "supported"
    assert child["acquired_observations"] == [0, 1]
    assert provider.requests[10]["research"]["answers"]["instrument"]["evidence"] == [0]
    assert child["unreviewed_observation_indices"] == [1]
    assert child["evaluate_action"] == {
        "action": "evaluate", "arguments": {"path": "child_analysis.py", "question_id": "instrument"}}
    assert before_review["methods"][0]["available"]
    after_review = json.loads(provider.requests[11]["files"]["method_decision.json"])
    assert provider.requests[11]["research"]["answers"]["instrument"]["evidence"] == [1]
    method = after_review["methods"][0]
    assert not method["available"] and "parent" in method["blocked_by"]
    assert method["execute_action"] == {"action": "query", "arguments": {"method_id": "method"}}


def test_revising_support_requirements_invalidates_the_old_method_admission(tmp_path):
    result, provider, lab = run_steps(tmp_path, [*supported_method_steps(),
        ("run", {"operator": "question", "id": "new_calibration",
                 "question": "Is the new calibration adequate?"}),
        ("run", {"operator": "question", "id": "parent", "depends_on": ["new_calibration"]}),
        ("query", {"method_id": "method"}),
        ("finish", {}),
    ], readings=(3, 4))

    assert provider.requests[9]["research"]["methods"][0]["available"]
    assert len(lab.actions) == 1
    parent = next(q for q in result["research"]["questions"] if q["id"] == "parent")
    assert parent["status"] != "supported"
    assert not result["research"]["methods"][0]["available"]
    assert "error" in result["events"][-2]["feedback"]


def test_batched_measurement_remains_visible_after_movement_replaces_current_message(tmp_path):
    class MessageLab(InstrumentLab):
        message = "Ready"

        def capabilities(self):
            return {"actions": ["USE", "TELEPORT_TO_LOCATION"]}

        def public_observation(self):
            return {**super().public_observation(), "lastActionMessage": self.message,
                    "extended_action_message": self.message,
                    "nearbyObjects": {"objects": {"north": [{"uuid": 33, "name": "FULL_WORLD_STATE_ONLY"}]}},
                    "dialog_box": {"is_in_dialog": False}}

        def act(self, action):
            self.actions.append(deepcopy(action))
            self.environment_steps += 1
            self.message = ("Instrument reading: 17.25" if action["action"] == "USE"
                            else "Teleported to laboratory")
            return {"action": deepcopy(action), "feedback": {"success": True},
                    "observation": self.public_observation()}

    actions = [{"action": "USE", "arg1": 11, "arg2": 22},
               {"action": "TELEPORT_TO_LOCATION", "arg1": "laboratory"}]
    provider = ScriptedProvider([("query", {"actions": actions}), ("finish", {})])
    agent = DiscoveryWorldAgent(provider, tmp_path, arm="hegel", limits=LIMITS)
    agent.run(MessageLab())

    request = provider.requests[1]
    assert request["current_observation"]["lastActionMessage"] == "Teleported to laboratory"
    feedback = request["events"][-1]["feedback"]
    assert [r["action"] for r in feedback["action_results"]] == actions
    assert [r["index"] for r in feedback["action_results"]] == [0, 1]
    assert feedback["action_results"][0]["lastActionMessage"] == "Instrument reading: 17.25"
    assert "FULL_WORLD_STATE_ONLY" not in json.dumps(feedback)
    assert all("observation" not in r and "extended_action_message" not in r
               and "dialog_box" not in r for r in feedback["action_results"])


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
def test_paid_evidence_and_run_written_notes_survive_the_recent_event_window(tmp_path, arm):
    class MessageLab(InstrumentLab):
        def public_observation(self):
            return {**super().public_observation(),
                    "lastActionMessage": "Instrument reading: 3" if self.actions else "Ready",
                    "nearbyObjects": [{"name": "FULL_WORLD_STATE_ONLY"}]}

    initial_note = "The instrument calibration remains unresolved."
    saved_note = "Observation 0 reports 3. Next distinguish the two calibration hypotheses."
    final_note = "The evidence and outstanding calibration question are saved."
    steps = [
        ("query", {"actions": [{"action": "MEASURE"}], "notes": initial_note}),
        ("run", {"code": "from pathlib import Path\n"
            f"Path('research_notes.md').write_text({saved_note!r})\n"}),
        *[("run", {"code": f"print('Unrelated local computation {i}')"}) for i in range(7)],
        ("finish", {"notes": final_note}),
    ]
    provider, lab = ScriptedProvider(steps), MessageLab()
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm, limits=LIMITS)
    agent.run(lab)

    assert provider.requests[1]["research_notes"] == initial_note
    request = provider.requests[-1]
    assert request["research_notes"] == saved_note
    assert all(event["action"] == "run" for event in request["events"])
    assert (tmp_path / "research_notes.md").read_text() == final_note
    assert not (tmp_path / "observations.json").exists()
    assert not (tmp_path / "model_observations.json").exists()

    # Use the real provider compactor: the old integration fabricated an empty
    # numerical evidence file and re-numbered scratch files within this window.
    context = prepare_code_context(tmp_path / "provider", request, schema={}, instructions="")
    assert context["evidence"]["observations_file"] == "laboratory_observations.json"
    assert context["evidence"]["observation_count"] == 1
    assert context["evidence"]["compact_observations_file"] == "action_observations.json"
    assert context["research_notes"] == saved_note
    records = json.loads((tmp_path / context["evidence"]["compact_observations_file"]).read_text())
    assert len(records) == 1
    assert records[0]["index"] == 0 and records[0]["action"] == {"action": "MEASURE"}
    assert records[0]["feedback"]["reading"] == 3
    assert records[0]["lastActionMessage"] == "Instrument reading: 3"
    assert "FULL_WORLD_STATE_ONLY" not in json.dumps(records)
    for event in context["events"]:
        assert event["arguments"]["path"] == event["feedback"]["executed_path"]
        assert (tmp_path / event["arguments"]["path"]).exists()
    rendered = json.dumps(context)
    assert "EVALUATOR_ONLY_GOLD_RULE" not in rendered and "scoreNormalized" not in rendered
    if arm == "hegel":
        # The public task is bookkeeping, not an authored child or an answer.
        questions = context["method_decision"]["questions"]
        assert len(questions) == 1
        root = questions[0]
        assert root["id"] == "task" and root["origin"] == "public_task"
        assert root["question"] == lab.public_observation()["taskProgress"][0]["description"]
        assert root["parent_id"] is None and root["depends_on"] == []
        assert root["evidence"] == [] and "answer" not in root
        assert root["status"] == "open"
        assert context["research_root"] == {"id": "task", "question": root["question"]}
    else:
        assert context["method_decision"]["methods"] == []
        assert context["method_decision"]["questions"] == []
        assert "research_root" not in context


def test_action_declared_question_registers_before_execution_without_claiming_support(tmp_path):
    child = {"id": "instrument", "question": "What does the instrument reading mean?",
             "parent_id": "task", "depends_on": []}
    result, provider, lab = run_steps(tmp_path, [
        ("run", {"research_question": child, "code": "print('Analysis still needed')"}),
        ("query", {"research_question": child, "actions": [{"action": "MEASURE"}]}),
        ("finish", {}),
    ])
    assert len(provider.requests[0]["research"]["questions"]) == 1
    declared = next(q for q in provider.requests[1]["research"]["questions"] if q["id"] == "instrument")
    assert declared["opened_before_observation"] == 0 and declared["status"] == "open"
    assert declared["evidence"] == [] and "answer" not in declared
    questions = result["research"]["questions"]
    assert len(questions) == 2  # Referencing the stable ID did not create another child.
    recorded = next(q for q in questions if q["id"] == "instrument")
    assert recorded["acquired_observations"] == [0] and recorded["evidence"] == []
    assert "answer" not in recorded and result["research"]["child_answers_returned"] == 0
    assert lab.actions == [{"action": "MEASURE"}]


def test_decision_question_does_not_replace_explicit_or_program_acquisition_question(tmp_path):
    metadata = {"id": "task", "question": InstrumentLab().public_observation()["taskProgress"][0]["description"],
                "parent_id": None, "depends_on": []}
    explicit = {"id": "explicit_probe", "question": "What is the first reading?",
                "parent_id": "task", "depends_on": [], "analysis_path": "first_analysis.py"}
    authored = {"id": "program_probe", "question": "What is the second reading?",
                "parent_id": "task", "depends_on": [], "analysis_path": "second_analysis.py"}
    program = ("def select_actions(spec, observations, memory):\n"
               f"    return {{'actions': [{{'action': 'MEASURE'}}], 'question': {authored!r}}}\n")
    result, _, _ = run_steps(tmp_path, [
        ("write", {"path": "probe.py", "code": program}),
        ("query", {"path": "probe.py", "question": explicit, "research_question": metadata}),
        ("query", {"path": "probe.py", "research_question": metadata}),
        ("finish", {}),
    ], readings=(3, 4))
    questions = {q["id"]: q for q in result["research"]["questions"]}
    assert questions["explicit_probe"]["acquired_observations"] == [0]
    assert questions["explicit_probe"]["analysis_path"] == "first_analysis.py"
    assert questions["program_probe"]["acquired_observations"] == [1]
    assert questions["program_probe"]["analysis_path"] == "second_analysis.py"
    assert questions["task"]["acquired_observations"] == []
    assert all(q["evidence"] == [] and "answer" not in q for q in questions.values())
