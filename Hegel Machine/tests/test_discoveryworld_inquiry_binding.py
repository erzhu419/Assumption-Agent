"""Retained program bindings reach local research without supplying an answer."""

from copy import deepcopy
import json

import pytest

from test_discoveryworld_llm_inquiry import CHILD, evaluate, inquire, prepared


PATH = "shared/retained_research.py"
OTHER = "shared/other_research.py"
STOP = """def select_actions(spec, observations, memory):
    return {'stop': 'The paid observations do not distinguish the remaining explanations'}
"""


def stopped(agent, lab, path):
    agent._execute("write", {"path": path, "code": STOP}, lab)
    agent._execute("query", {"path": path}, lab)
    return deepcopy(agent.method_results[-1])


def decision_question(request):
    summary = json.loads(request["files"]["method_decision.json"])
    return next(row for row in summary["questions"] if row["id"] == CHILD["id"])


def test_question_binding_survives_references_and_retitling_without_granting_support(tmp_path):
    agent, _, lab = prepared(tmp_path, [])
    question = agent._question({**CHILD, "parent_program_path": PATH})
    assert question["parent_program_path"] == PATH
    assert question["status"] == "open" and "answer" not in question
    assert agent._question({"id": CHILD["id"]}) is question
    assert question["parent_program_path"] == PATH

    agent._execute("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    action, arguments = evaluate()
    agent._execute(action, arguments, lab)
    answer = deepcopy(question["answer"])
    assert question["status"] == "supported"
    agent._question({"id": CHILD["id"], "parent_program_path": OTHER})
    assert question["status"] == "supported" and question["answer"] == answer
    assert question["parent_program_path"] == OTHER

    agent._question({"id": CHILD["id"], "question": "Does the earlier result apply after a change?"})
    assert question["status"] == "reopened"
    assert question["parent_program_path"] == OTHER
    assert question["evidence"] == [0] and question["answer"] == answer


def test_binding_can_be_added_to_an_existing_question_without_erasing_its_evidence(tmp_path):
    agent, _, lab = prepared(tmp_path, [])
    agent._execute("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    question = agent._question({"id": CHILD["id"], "parent_program_path": PATH})
    assert question["parent_program_path"] == PATH
    assert question["acquired_observations"] == [0]
    assert question["status"] == "open" and "answer" not in question


@pytest.mark.parametrize("explicit", [False, True])
def test_program_delegation_exposes_matching_stop_and_executable_continuation(tmp_path, explicit):
    bound = OTHER if explicit else PATH
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "procedure", "path": bound}),
        ("finish", {"lesson": "The retained research still needs a distinguishing observation."}),
    ])
    previous = stopped(agent, lab, bound)
    stopped(agent, lab, "shared/unrelated.py")
    declared = {**CHILD, **({"parent_program_path": bound} if explicit else {})}
    delegation = {"action": "run", "arguments": {
        "operator": "inquire", "question": declared, "max_model_calls": 2}}
    source = ("def select_actions(spec, observations, memory):\n"
              "    if memory.get('llm_inquiries', {}).get('response', {}).get('invocations'):\n"
              "        return {'stop': 'Retained research resumed and remains unresolved'}\n"
              f"    return {delegation!r}\n")
    agent._execute("write", {"path": PATH, "code": source}, lab)

    result = agent._execute("query", {"path": PATH}, lab)

    assert result["status"] == "inquiry_finished"
    first = provider.requests[0]
    assert first["llm_inquiry"]["assignment"]["parent_program_path"] == bound
    entry = decision_question(first)
    assert entry["procedure_action"] == {"action": "run", "arguments": {
        "operator": "procedure", "path": bound}}
    assert entry["last_program_result"] == previous
    actual = decision_question(provider.requests[1])["last_program_result"]
    assert actual["path"] == bound and actual["step"] > previous["step"]
    assert actual["feedback"]["status"] == "method_stopped"
    assert "answer" not in result["question"] and result["question"]["status"] == "open"
    assert agent.status == "running" and lab.environment_steps == 0


def test_plain_inquiry_does_not_guess_a_program_from_parent_or_latest_execution(tmp_path):
    agent, provider, lab = prepared(tmp_path, [("finish", {"lesson": "No new evidence yet."})])
    agent._question({"id": "task", "parent_program_path": PATH})
    stopped(agent, lab, PATH)
    stopped(agent, lab, OTHER)

    result = inquire(agent, lab)

    assert result["status"] == "inquiry_finished"
    entry = decision_question(provider.requests[0])
    assert "parent_program_path" not in entry
    assert "procedure_action" not in entry and "last_program_result" not in entry
    assert "parent_program_path" not in provider.requests[0]["llm_inquiry"]["assignment"]
    assert len(agent.method_results) == 2 and lab.environment_steps == 0
