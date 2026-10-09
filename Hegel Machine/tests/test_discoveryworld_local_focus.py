"""Existing research scheduling stays local to an active model inquiry."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_discoveryworld_inquiry_binding import PATH, stopped
from test_discoveryworld_llm_inquiry import ANALYSIS, CHILD, Provider, inquire, prepared


LEAF = {"id": "reference", "question": "Does this reference respond?",
        "parent_id": CHILD["id"], "depends_on": []}
SIBLING = {"id": "other", "question": "Does the unrelated reference respond?",
           "parent_id": "task", "depends_on": []}


def question(agent, qid):
    return next(row for row in agent.questions if row["id"] == qid)


def forbid_global_scan(monkeypatch):
    from hegel_machine.cognition import world_prerequisite_research

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("A local inquiry must not select outer operational questions")

    monkeypatch.setattr(world_prerequisite_research, "next_prerequisite", unexpected_scan)


def test_local_analysis_returns_child_evidence_to_its_parent_and_preserves_outer_focus(tmp_path, monkeypatch):
    agent, provider, lab = prepared(tmp_path, [
        ("write", {"path": "shared/analysis_reference.py", "code": ANALYSIS}),
        ("finish", {"lesson": "The child returned; the local parent still needs a next experiment."}),
    ])
    agent._question(CHILD)
    agent._question(LEAF)
    # The unrelated, later-registered sibling would win the global reverse scan.
    agent._execute("query", {"question": SIBLING,
                              "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    agent._execute("query", {"question": LEAF,
                              "actions": [{"action": "TELEPORT_TO_OBJECT", "arg1": 9}]}, lab)
    outer_focus = deepcopy(agent._request()["research_focus"])
    assert outer_focus["question"]["id"] == SIBLING["id"]
    cursor = agent._prerequisite_checked_through
    forbid_global_scan(monkeypatch)

    result = inquire(agent, lab)

    first, returned = provider.requests
    assert first["research_focus"]["kind"] == "analysis"
    assert first["research_focus"]["question"]["id"] == LEAF["id"]
    assert first["research_focus"]["evidence"] == [1]
    assert {q["id"] for q in first["research"]["questions"]} == {
        "task", CHILD["id"], LEAF["id"], SIBLING["id"]}
    assert first["evidence"]["observation_count"] == 2
    focus = returned["research_focus"]
    assert focus["kind"] == "parent_research" and focus["question"]["id"] == CHILD["id"]
    assert focus["returned_children"][0]["evidence"] == [1]
    assert focus["changed_child_ids"] == [LEAF["id"]]
    assert question(agent, LEAF["id"])["status"] == "supported"
    assert question(agent, SIBLING["id"])["status"] == "open"
    assert result["status"] == "inquiry_finished"
    assert agent.research_focus == outer_focus
    assert agent._prerequisite_checked_through == cursor
    assert len(lab.actions) == 2  # Scheduling and analysis buy no replacement evidence.


def test_bound_leaf_stop_is_reviewed_without_children_or_forced_execution(tmp_path, monkeypatch):
    agent, _, lab = prepared(tmp_path, [])
    previous = stopped(agent, lab, PATH)
    stopped(agent, lab, "shared/unrelated.py")
    executions = agent.program_executions
    repeated = []

    class RepeatedRequestProvider(Provider):
        def next_action(self, request):
            if not self.requests:
                repeated.append(agent._request())
            return super().next_action(request)

    provider = RepeatedRequestProvider([
        ("run", {"code": "print('The existing stop requires a different experiment')"}),
        ("finish", {"lesson": "No new experiment was selected."}),
    ])
    agent.provider = provider
    cursor = agent._prerequisite_checked_through
    forbid_global_scan(monkeypatch)

    result = inquire(agent, lab, question={**CHILD, "parent_program_path": PATH})

    first, after_review = provider.requests
    focus = first["research_focus"]
    assert focus == repeated[0]["research_focus"]
    assert focus["stage"] == "review" and focus["path"] == PATH
    assert focus["previous_execution"] == previous and focus["returned_children"] == []
    assert focus["question"]["status"] == "open"
    assert "reviewed_program_step" not in focus["question"]
    assert not after_review.get("research_focus")
    assert question(agent, CHILD["id"])["reviewed_program_step"] == previous["step"]
    assert "answer" not in result["question"]
    assert agent.program_executions == executions and len(agent.method_results) == 2
    assert not lab.actions and agent._prerequisite_checked_through == cursor


def test_unrelated_stopped_parent_never_becomes_plain_local_inquiry_focus(tmp_path, monkeypatch):
    agent, provider, lab = prepared(tmp_path, [("finish", {"lesson": "No local evidence yet."})])
    agent._question({"id": "task", "parent_program_path": PATH})
    stopped(agent, lab, PATH)
    cursor = agent._prerequisite_checked_through
    forbid_global_scan(monkeypatch)

    result = inquire(agent, lab)

    assert not provider.requests[0].get("research_focus")
    assert result["status"] == "inquiry_finished"
    assert agent._prerequisite_checked_through == cursor and not lab.actions


def test_nested_inquiry_returns_to_its_unconsumed_parent_analysis_focus(tmp_path):
    nested = {"id": "secondary", "question": "What extra observation would resolve this?",
              "parent_id": CHILD["id"], "depends_on": []}
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "inquire", "question": nested, "max_model_calls": 1}),
        ("finish", {"lesson": "Return without a new claim."}),
        ("finish", {"lesson": "The earlier paid result still requires analysis."}),
    ])
    agent._execute("query", {"question": CHILD,
                              "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    outer_focus = deepcopy(agent._request()["research_focus"])

    result = inquire(agent, lab)

    first, inner, resumed = provider.requests
    assert first["research_focus"] == outer_focus
    assert not inner.get("research_focus")
    assert inner["research_root"]["id"] == nested["id"]
    assert resumed["research_focus"] == outer_focus
    assert resumed["research_root"]["id"] == CHILD["id"]
    assert result["status"] == "inquiry_finished" and agent.research_focus == outer_focus
    assert len(lab.actions) == 1 and agent.costs["model_calls"] == 3


def test_local_analysis_focus_keeps_experiment_revision_available(tmp_path):
    from test_discoveryworld_llm_inquiry import EXPERIMENT, evaluate

    experiment_path = "shared/local_experiment.py"
    revised = EXPERIMENT.replace("'TALK', 'arg1': 7", "'TELEPORT_TO_OBJECT', 'arg1': 9")
    agent, provider, lab = prepared(tmp_path, [
        ("query", {"path": experiment_path, "question": CHILD}),
        ("write", {"path": experiment_path, "code": revised}),
        ("query", {"path": experiment_path, "question": CHILD}),
        evaluate(),
    ])
    agent._execute("write", {"path": experiment_path, "code": EXPERIMENT}, lab)

    result = inquire(agent, lab)

    focus = provider.requests[1]["research_focus"]
    assert focus["kind"] == "analysis" and focus["path"] != experiment_path
    assert not result["steps"][1]["feedback"].get("error")
    assert provider.requests[2]["research_focus"] == focus
    assert (tmp_path / experiment_path).read_text() == revised
    assert lab.actions == [{"action": "TALK", "arg1": 7},
                           {"action": "TELEPORT_TO_OBJECT", "arg1": 9}]
    assert result["status"] == "answered" and result["question"]["evidence"] == [1]


def test_local_analysis_provider_preserves_query_action(tmp_path, monkeypatch):
    from hegel_machine.cognition import code_agent

    arguments = {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}

    def run(command, prompt, timeout, cwd):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert schema["properties"]["action"]["enum"] == code_agent._ACTION_SCHEMA["properties"]["action"]["enum"]
        assert "query" in schema["properties"]["action"]["enum"]
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({
            "action": "query", "arguments_json": json.dumps(arguments),
            "research_notes": "The existing result requires a further discriminating observation."}))
        return 0, "", "", False

    monkeypatch.setattr(code_agent, "_run", run)
    provider = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test")
    reply = provider.next_action({"task_interface": "query and evaluate use the public world",
        "llm_inquiry": {"question_id": CHILD["id"], "assignment": CHILD},
        "research_focus": {"kind": "analysis", "question": CHILD,
                           "path": "shared/analysis_response.py", "evidence": [0]},
        "research_notes": "", "files": {}})
    assert reply["action"] == "query"
    assert all(reply["arguments"][key] == value for key, value in arguments.items())


def test_answer_with_pending_child_does_not_finish_the_local_assignment(tmp_path):
    from test_discoveryworld_llm_inquiry import evaluate

    agent, provider, lab = prepared(tmp_path, [
        ("query", {"question": CHILD, "actions": [{"action": "TALK", "arg1": 7}]}),
        ("run", {"operator": "question", **LEAF}),
        ("finish", {"lesson": "The scheduled child has no answer yet."}),
        evaluate(),
        ("finish", {"lesson": "The initial answer does not resolve the remaining child."}),
    ])

    result = inquire(agent, lab)

    assert result["status"] == "inquiry_finished" and result["model_calls"] == 5
    assert provider.requests[2]["llm_inquiry"]["question_id"] == LEAF["id"]
    assert provider.requests[-1]["llm_inquiry"]["question"]["status"] == "supported"
    assert result["question"]["evidence"] == [0]
    assert agent._pending_dependencies(question(agent, CHILD["id"])) == [LEAF["id"]]


def test_automatic_repair_does_not_resume_on_supported_retitled_measurement(tmp_path):
    from test_discoveryworld_automatic_repair import (
        BROKEN, PATH as repair_path, RepairProvider, prepared as repair_agent, run_parent)
    from test_discoveryworld_method_reuse import ReferenceLab

    def measurement(request):
        qid = request["llm_inquiry"]["question_id"]
        return "query", {"question": {"id": qid, "question": "Is this one reference active?"},
                         "actions": [{"action": "TALK", "arg1": 7}]}

    def answer(request):
        return "evaluate", {"question_id": request["llm_inquiry"]["question_id"],
                            "path": "shared/reference_answer.py", "code": ANALYSIS}

    provider = RepairProvider([measurement, answer,
        ("finish", {"lesson": "The reference responds; the retained broken procedure was not repaired."})])
    agent, lab = repair_agent(tmp_path, provider, lab=ReferenceLab())

    result = run_parent(agent)

    outcome = result["repair"]["outcome"]
    assert outcome["question"]["status"] == "supported" and outcome["status"] == "inquiry_finished"
    assert result["status"] == "method_error" and "continuation" not in result
    assert (tmp_path / repair_path).read_text() == BROKEN
    assert lab.actions == [{"action": "TALK", "arg1": 7}]
    assert sum(row["path"] == repair_path for row in agent.method_results) == 1
    assert len(provider.requests) == 3


@pytest.mark.parametrize("field,value", [
    ("question", "Does the first reference currently respond?"),
    ("parent_id", "other_parent"),
    ("depends_on", ["premise"]),
])
def test_answer_to_changed_assignment_keeps_local_loop_open(tmp_path, field, value):
    from test_discoveryworld_llm_inquiry import evaluate

    agent, provider, lab = prepared(tmp_path, [
        ("query", {"question": {**CHILD, field: value},
                   "actions": [{"action": "TALK", "arg1": 7}]}),
        evaluate(),
        ("finish", {"lesson": "Return this revised question's evidence, not a completed assignment."}),
    ])
    if field == "depends_on":
        premise = agent._question({"id": "premise", "question": "A previously established premise"})
        # This case tests assignment changes, not unresolved-premise rejection.
        premise["status"] = "supported"

    result = inquire(agent, lab)

    assert result["status"] == "inquiry_finished" and result["model_calls"] == 3
    assert provider.requests[-1]["llm_inquiry"]["question"]["status"] == "supported"
    assert provider.requests[-1]["llm_inquiry"]["assignment"][field] == CHILD[field]
    assert result["question"][field] == value and result["question"]["evidence"] == [0]
