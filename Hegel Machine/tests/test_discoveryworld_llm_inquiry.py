"""A child LLM can execute, repair and resume research without ending its parent."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_method_reuse import ReferenceLab
from test_discoveryworld_recovery import DECODE_ERROR, USAGE


@pytest.mark.parametrize("kind", ["analysis", "intervention", "prerequisite"])
def test_scheduled_child_can_actually_select_the_model_loop(tmp_path, monkeypatch, kind):
    from hegel_machine.cognition import code_agent

    arguments = {"operator": "inquire", "question": CHILD, "max_model_calls": 8}

    def run(command, prompt, timeout, cwd):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert "run" in schema["properties"]["action"]["enum"]
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        assert "continuing local LLM loop" in instructions
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({
            "action": "run", "arguments_json": json.dumps(arguments), "research_notes": "Delegate this child."}))
        return 0, "", "", False

    monkeypatch.setattr(code_agent, "_run", run)
    provider = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test")
    reply = provider.next_action({"task_interface": "run operator:'inquire' is available",
        "research_notes": "", "research_focus": {"kind": kind, "question": CHILD}, "files": {}})
    assert reply["action"] == "run" and reply["arguments"]["operator"] == "inquire"


CHILD = {"id": "response", "question": "Does the observed reference remain active?",
         "parent_id": "task", "depends_on": []}
PATH = "shared/child_experiment.py"
BROKEN = "def select_actions(spec, observations, memory):\n    return [}\n"
EXPERIMENT = """def select_actions(spec, observations, memory):
    return [{'action': 'TALK', 'arg1': 7}]
"""
ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    active = record['observation']['active']
    return {'status': 'supported' if active else 'refuted',
            'answer': {'observed_active': active}, 'evidence': [record['index']]}
"""


class Provider(ScriptedProvider):
    def next_action(self, request):
        result = super().next_action(request)
        result["usage"] = deepcopy(USAGE)
        if result["action"] == "provider_error":
            result["error"] = deepcopy(result["arguments"])
            result["arguments"] = {}
        return result


def prepared(tmp_path, steps, *, limits=None, **lab_options):
    provider, lab = Provider(steps), ReferenceLab(**lab_options)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm="hegel", limits={**LIMITS, **(limits or {})})
    agent.lab = lab
    agent._question({"id": "task", "question": "Explain the control response"})
    (tmp_path / "research_notes.md").write_text("Parent: compare current controls.")
    return agent, provider, lab


def inquire(agent, lab, *, question=None, calls=8):
    return agent._execute("run", {"operator": "inquire", "question": question or CHILD,
                                  "max_model_calls": calls}, lab)


def evaluate(question_id="response"):
    return ("evaluate", {"question_id": question_id,
                         "path": f"shared/answer_{question_id}.py", "code": ANALYSIS})


def test_child_repairs_its_actual_syntax_failure_then_parent_uses_returned_answer(tmp_path):
    steps = [
        ("run", {"operator": "inquire", "question": CHILD, "max_model_calls": 8,
                 "notes": "Parent: compare current controls."}),
        ("write", {"path": PATH, "code": BROKEN, "notes": "Child: execute this draft."}),
        ("query", {"path": PATH, "question": CHILD}),
        ("write", {"path": PATH, "code": EXPERIMENT, "notes": "Child: repair the syntax."}),
        ("query", {"path": PATH, "question": CHILD}),
        evaluate(),
        ("query", {"question": {"id": "task"},
                   "actions": [{"action": "TELEPORT_TO_OBJECT", "arg1": 9}]}),
        ("finish", {"lesson": "The child answered; a second reference still needs research."}),
    ]
    agent, provider, lab = prepared(tmp_path, steps)
    result = agent.run(lab)

    repair_request = provider.requests[3]
    assert "SyntaxError" in repair_request["events"][-1]["feedback"]["error"]
    assert repair_request["research_notes"] == "Child: execute this draft."
    assert repair_request["evidence"]["observation_count"] == 0
    assert repair_request["llm_inquiry"]["question_id"] == "response"
    assert lab.actions == [{"action": "TALK", "arg1": 7},
                           {"action": "TELEPORT_TO_OBJECT", "arg1": 9}]
    assert (tmp_path / PATH).read_text() == EXPERIMENT
    returned = provider.requests[6]
    assert not returned.get("llm_inquiry")
    assert returned["research_notes"] == "Parent: compare current controls."
    question = next(q for q in result["research"]["questions"] if q["id"] == "response")
    assert question["status"] == "supported" and question["evidence"] == [0]
    assert result["research"]["child_answers_returned"] == 1
    assert result["status"] == "agent_finished"
    assert result["costs"]["model_calls"] == len(provider.requests) == 8
    assert result["costs"]["reported_tokens"] == 8 * USAGE["total_tokens"]
    assert result["costs"]["cached_input_tokens"] == 8 * USAGE["cached_input_tokens"]


def test_child_repairs_empty_analysis_evidence_with_actual_branch_and_global_indices(tmp_path):
    wrong_layer = """def answer(spec, observations, memory):
    record = observations[-1]
    if 'active' not in record:
        return {'status': 'inconclusive',
                'answer': {'missing_field': 'active in record'}, 'evidence': []}
    return {'status': 'supported', 'answer': record['active'], 'evidence': [record['index']]}
"""
    agent, provider, lab = prepared(tmp_path, [
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        ("evaluate", {"question_id": CHILD["id"], "path": "shared/answer_response.py",
                      "code": wrong_layer}),
        evaluate(),
    ])
    agent._execute("query", {"question": {"id": "task"}, "actions": [
        {"action": "TALK", "arg1": 7}, {"action": "TELEPORT_TO_OBJECT", "arg1": 9}]}, lab)

    result = inquire(agent, lab)

    repair = provider.requests[2]
    error = repair["events"][-1]["feedback"]["error"]
    assert "Analysis must reference actual observation indices" in error
    assert "evidence=[]" in error
    assert "0 <= index < 3" in error and "global" in error
    assert "inconclusive" in error and "missing_field" in error and "active in record" in error
    assert repair["llm_inquiry"]["question"]["status"] not in {"supported", "refuted"}
    assert result["status"] == "answered" and result["question"]["status"] == "supported"
    assert result["question"]["evidence"] == [2]
    assert result["model_calls"] == 3 and len(lab.actions) == 3
    assert agent.status == "running"


def test_child_finish_is_local_and_same_question_resumes_its_notebook(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"code": "print('A measurement is still required')",
                 "notes": "Child: inspect the public reference next."}),
        ("finish", {"lesson": "No measurement yet."}),
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        evaluate(),
    ])
    first = inquire(agent, lab)
    assert first["status"] == "inquiry_finished"
    assert first["question"]["status"] == "open"
    assert "answer" not in first["question"] and not lab.actions
    assert agent.status == "running" and not agent.lesson
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."

    second = inquire(agent, lab)
    assert second["status"] == "answered" and second["question"]["status"] == "supported"
    assert second["question"]["evidence"] == [0]
    assert provider.requests[2]["research_notes"] == "Child: inspect the public reference next."
    assert provider.requests[2]["events"][-1]["action"] == "finish"
    assert provider.requests[2]["llm_inquiry"]["question_id"] == "response"
    assert agent.status == "running"
    assert agent.costs["model_calls"] == 4
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."


@pytest.mark.parametrize("active,verdict", [(True, "supported"), (False, "refuted")])
def test_child_retitles_experiment_without_losing_delegated_program_deliverable(tmp_path, active, verdict):
    assignment = {**CHILD, "question":
        "Revise shared/consolidated_method.py from new measurements and execute the reusable estimator."}
    measurement = {**CHILD, "question": "What does the first reference currently show?"}
    experiment = {**CHILD, "question": "Is the second reference active after approaching it?"}
    agent, provider, lab = prepared(tmp_path, [
        ("query", {"research_question": measurement,
                   "actions": [{"action": "TALK", "arg1": 7}]}),
        ("query", {"research_question": experiment,
                   "actions": [{"action": "TELEPORT_TO_OBJECT", "arg1": 9}]}),
        evaluate(),
        ("finish", {"lesson": "The reference answer does not yet revise the delegated estimator."}),
    ], baseline_active=active)

    result = inquire(agent, lab, question=assignment)

    for request in provider.requests:
        assert request["research_root"]["question"] == assignment["question"]
        assert request["llm_inquiry"]["assignment"]["question"] == assignment["question"]
    assert provider.requests[1]["llm_inquiry"]["question"]["question"] == measurement["question"]
    assert provider.requests[2]["llm_inquiry"]["question"]["question"] == experiment["question"]
    assert provider.requests[3]["llm_inquiry"]["question"]["status"] == verdict
    assert result["status"] == "inquiry_finished" and result["model_calls"] == 4
    assert result["question"]["question"] == experiment["question"]
    assert result["question"]["status"] == verdict and result["question"]["evidence"] == [1]
    assert agent.status == "running"


def test_explicit_redelegation_updates_assignment_and_keeps_existing_notebook(tmp_path):
    first = {**CHILD, "question": "Repair shared/source_method.py and test its reference access."}
    experiment = {**CHILD, "question": "Does the current reference respond?"}
    second = {**CHILD, "question": "Use the saved readings to revise shared/target_method.py."}
    agent, provider, lab = prepared(tmp_path, [
        ("query", {"research_question": experiment,
                   "actions": [{"action": "TALK", "arg1": 7}],
                   "notes": "The source reference was measured; retain this evidence."}),
        ("finish", {"lesson": "Return the measured reference before the next assignment."}),
        evaluate(),
    ])

    assert inquire(agent, lab, question=first)["status"] == "inquiry_finished"
    result = inquire(agent, lab, question=second)

    assert provider.requests[1]["llm_inquiry"]["assignment"]["question"] == first["question"]
    resumed = provider.requests[2]
    assert resumed["research_root"]["question"] == second["question"]
    assert resumed["llm_inquiry"]["assignment"]["question"] == second["question"]
    assert resumed["llm_inquiry"]["invocation"] == 2
    assert resumed["research_notes"] == "The source reference was measured; retain this evidence."
    assert resumed["events"][-1]["action"] == "finish"
    assert result["status"] == "answered" and result["question"]["evidence"] == [0]
    assert len(lab.actions) == 1 and agent.costs["model_calls"] == 3


def test_child_python_notebook_edits_survive_resume_without_reading_parent_notes(tmp_path):
    note = "Child: Python computed the next reference to inspect."
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"code": "from pathlib import Path\n"
                          f"Path('research_notes.md').write_text({note!r})\n",
                 "notes": "Child: run the calculation."}),
        ("finish", {"lesson": "Return now; retain the calculation."}),
        ("run", {"code": "from pathlib import Path\nprint(Path('research_notes.md').read_text())\n"}),
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        evaluate(),
    ])
    assert inquire(agent, lab)["status"] == "inquiry_finished"
    assert provider.requests[1]["research_notes"] == note
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."
    resumed = inquire(agent, lab)
    assert provider.requests[2]["research_notes"] == note
    assert resumed["steps"][0]["feedback"]["stdout"].strip() == note
    assert resumed["status"] == "answered" and len(lab.actions) == 1
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."


@pytest.mark.parametrize("active,verdict", [(True, "supported"), (False, "refuted")])
def test_actual_positive_and_negative_child_answers_return_automatically(tmp_path, active, verdict):
    agent, provider, lab = prepared(tmp_path, [
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        evaluate(),
    ], baseline_active=active)
    result = inquire(agent, lab)
    assert result["status"] == "answered"
    assert result["question"]["status"] == verdict and result["question"]["evidence"] == [0]
    assert result["model_calls"] == len(provider.requests) == 2
    assert agent.status == "running"
    assert next(q for q in agent.questions if q["id"] == "task")["status"] != "supported"


def test_local_call_limit_returns_unresolved_and_can_resume_paid_measurement(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        evaluate(),
    ])
    first = inquire(agent, lab, calls=1)
    assert first["status"] == "model_call_budget_exhausted"
    assert first["question"]["status"] != "supported" and "answer" not in first["question"]
    assert agent.status == "running" and len(provider.requests) == 1
    second = inquire(agent, lab, calls=1)
    assert second["status"] == "answered" and second["question"]["evidence"] == [0]
    assert len(lab.actions) == 1 and agent.costs["model_calls"] == 2


def test_child_cannot_spend_beyond_parent_global_call_budget(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "inquire", "question": CHILD, "max_model_calls": 8}),
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
    ], limits={"max_model_calls": 2})
    result = agent.run(lab)
    assert result["status"] == "model_call_budget_exhausted"
    assert result["costs"]["model_calls"] == len(provider.requests) == 2
    assert result["costs"]["reported_tokens"] == 2 * USAGE["total_tokens"]
    assert len(lab.actions) == 1
    assert all(q["status"] != "supported" for q in agent.questions)


def test_decode_failure_returns_to_same_child_before_real_repair(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("provider_error", DECODE_ERROR),
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        evaluate(),
    ])
    result = inquire(agent, lab)
    retry = provider.requests[1]
    assert retry["llm_inquiry"]["question_id"] == "response"
    assert retry["events"][-1]["feedback"]["error"] == DECODE_ERROR["decode_error"]
    assert retry["costs"]["model_calls"] == 1 and retry["evidence"]["observation_count"] == 0
    assert result["status"] == "answered" and result["model_calls"] == 3
    assert agent.costs["reported_tokens"] == 3 * USAGE["total_tokens"]
    assert len(lab.actions) == 1


def test_service_failure_never_becomes_a_supported_child_answer(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("provider_error", {"returncode": -15, "timeout": True}),
    ])
    result = inquire(agent, lab)
    assert result["status"] == "provider_failed"
    assert result["question"]["status"] != "supported" and "answer" not in result["question"]
    assert agent.costs["model_calls"] == len(provider.requests) == 1
    assert agent.costs["reported_tokens"] == USAGE["total_tokens"]
    assert not lab.actions
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."


def test_nested_child_llm_research_executes_and_returns_to_each_parent(tmp_path):
    grandchild = {"id": "reference", "question": "Is the first reference active?",
                  "parent_id": "response", "depends_on": []}
    local_experiment = {**grandchild, "question": "Does the reference respond to TALK?"}
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "inquire", "question": grandchild, "max_model_calls": 4,
                 "notes": "Child: ask about reference one, then inspect reference two."}),
        ("query", {"question": local_experiment, "actions": [{"action": "TALK", "arg1": 7}],
                   "notes": "Grandchild: reference one is observed."}),
        evaluate("reference"),
        ("finish", {"lesson": "Return the narrower TALK result to the caller."}),
        ("query", {"question": CHILD, "actions": [{"action": "TELEPORT_TO_OBJECT", "arg1": 9}]}),
        evaluate(),
    ])
    result = inquire(agent, lab, calls=7)
    assert result["status"] == "answered" and result["model_calls"] == 6
    assert result["question"]["status"] == "supported" and result["question"]["evidence"] == [1]
    assert [request["llm_inquiry"]["question_id"] for request in provider.requests] == [
        "response", "reference", "reference", "reference", "response", "response"]
    assert [request["llm_inquiry"]["depth"] for request in provider.requests] == [1, 2, 2, 2, 1, 1]
    assert [request["llm_inquiry"]["assignment"]["question"] for request in provider.requests] == [
        CHILD["question"], grandchild["question"], grandchild["question"], grandchild["question"],
        CHILD["question"], CHILD["question"]]
    assert provider.requests[2]["llm_inquiry"]["question"]["question"] == local_experiment["question"]
    assert provider.requests[3]["llm_inquiry"]["question"]["status"] == "supported"
    assert provider.requests[4]["research_root"]["question"] == CHILD["question"]
    assert provider.requests[4]["research_notes"] == "Child: ask about reference one, then inspect reference two."
    inner = next(q for q in agent.questions if q["id"] == "reference")
    assert inner["status"] == "supported" and inner["evidence"] == [0]
    assert lab.actions == [{"action": "TALK", "arg1": 7},
                           {"action": "TELEPORT_TO_OBJECT", "arg1": 9}]
    assert agent.costs["model_calls"] == 6 and agent.costs["reported_tokens"] == 6 * USAGE["total_tokens"]
    assert agent.child_answers_returned == 2 and agent.status == "running"
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."


def test_descendant_calls_also_consume_ancestor_local_allowance(tmp_path):
    grandchild = {"id": "reference", "question": "Is the first reference active?",
                  "parent_id": "response", "depends_on": []}
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "inquire", "question": grandchild, "max_model_calls": 8}),
        ("query", {"question": grandchild, "actions": [{"action": "TALK", "arg1": 7}]}),
    ])
    result = inquire(agent, lab, calls=2)
    assert result["status"] == "model_call_budget_exhausted"
    assert result["model_calls"] == len(provider.requests) == agent.costs["model_calls"] == 2
    assert all(q["status"] != "supported" for q in agent.questions)
    assert len(lab.actions) == 1 and agent.status == "running"


def test_existing_procedure_calls_child_llm_then_uses_answer_in_its_next_real_step(tmp_path):
    parent_path, child_path = "shared/parent_method.py", "shared/child_measurement.py"
    delegation = {"action": "run", "arguments": {
        "operator": "inquire", "question": CHILD, "max_model_calls": 4}}
    parent_source = (
        "def select_actions(spec, observations, memory):\n"
        "    answer = memory['answers'].get('response', {})\n"
        "    if answer.get('status') != 'supported':\n"
        f"        return {delegation!r}\n"
        "    if spec['current_observation']['target'] == 9:\n"
        "        return {'stop': 'Child evidence used; second reference reached.'}\n"
        "    return {'actions': [{'action': 'TELEPORT_TO_OBJECT', 'arg1': 9}],\n"
        "            'used_child_evidence': answer['evidence']}\n"
    )
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": child_path, "code": EXPERIMENT}),
        ("query", {"path": child_path, "question": CHILD}),
        evaluate(),
    ])
    agent._execute("write", {"path": parent_path, "code": parent_source}, lab)
    result = agent._execute("run", {"operator": "procedure", "path": parent_path,
                                     "question": {"id": "task"}, "max_steps": 3}, lab)

    assert result["status"] == "method_stopped"
    assert lab.actions == [{"action": "TALK", "arg1": 7},
                           {"action": "TELEPORT_TO_OBJECT", "arg1": 9}]
    assert result["steps"][0]["proposal"] == delegation
    assert result["steps"][0]["feedback"]["status"] == "answered"
    assert result["steps"][0]["feedback"]["question"]["evidence"] == [0]
    assert result["steps"][1]["proposal"]["used_child_evidence"] == [0]
    assert result["steps"][1]["feedback"]["observation_indices"] == [1]
    assert [row["path"] for row in agent.method_results] == [child_path, parent_path, parent_path, parent_path]
    assert agent.method_results[0]["proposal"] == [{"action": "TALK", "arg1": 7}]
    assert agent.method_results[1]["proposal"] == delegation
    assert agent._current_proposal == {"stop": "Child evidence used; second reference reached."}
    assert agent.costs["model_calls"] == len(provider.requests) == 3
    assert agent.costs["reported_tokens"] == sum(event["usage"]["total_tokens"] for event in agent.events) == 3 * USAGE["total_tokens"]
    assert agent.costs["environment_steps"] == 2 and agent.status == "running"
