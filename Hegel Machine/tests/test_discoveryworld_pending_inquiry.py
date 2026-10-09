"""Pending local questions use the existing researcher and paid evidence account."""

from copy import deepcopy

import pytest

from test_discoveryworld_llm_inquiry import CHILD, Provider, evaluate, inquire, prepared
from test_discoveryworld_recovery import USAGE


LEAF = {"id": "reference", "question": "Does this reference currently respond?",
        "parent_id": CHILD["id"], "depends_on": []}


def question(agent, qid):
    return next(row for row in agent.questions if row["id"] == qid)


def register(agent, *questions):
    agent._question(CHILD)
    for item in questions:
        agent._question(item)


def host_events(agent):
    return [row for row in agent.events if row.get("trigger_origin") == "host_pending_child_research"]


@pytest.mark.parametrize("active,verdict,target", [(True, "supported", 9), (False, "refuted", 7)])
def test_unmeasured_child_returns_real_analysis_before_parent_selects_its_action(
        tmp_path, active, verdict, target):
    agent, _, lab = prepared(tmp_path, [], baseline_active=active)
    register(agent, LEAF)
    outer_focus = {"kind": "analysis", "question": {"id": "task"}, "path": "outer.py"}
    agent.research_focus = deepcopy(outer_focus)
    agent.llm_inquiries[CHILD["id"]] = {
        "question_id": CHILD["id"], "notes": "Parent: select an action from the child answer.",
        "events": [], "invocations": 0, "model_calls": 0, "status": "open"}

    class BranchingProvider(Provider):
        def next_action(self, request):
            if request["llm_inquiry"]["question_id"] == CHILD["id"] and not self.parent_returned:
                returned = next(q for q in request["research"]["questions"] if q["id"] == LEAF["id"])
                assert returned["status"] == verdict and returned["evidence"] == [0]
                assert request["research_notes"] == "Parent: select an action from the child answer."
                assert request["research_focus"]["kind"] == "parent_research"
                assert request["research_focus"]["question"]["id"] == CHILD["id"]
                self.parent_returned = True
                self.requests.append(deepcopy(request))
                return {"action": "query", "arguments": {
                    "question": {"id": CHILD["id"]}, "actions": [{"action": "TELEPORT_TO_OBJECT",
                        "arg1": 9 if returned["answer"]["observed_active"] else 7}]},
                    "usage": deepcopy(USAGE)}
            return super().next_action(request)

    provider = BranchingProvider([
        ("query", {"question": LEAF, "actions": [{"action": "TALK", "arg1": 7}],
                   "notes": "Child: analyze the actual reference response."}),
        evaluate(LEAF["id"]),
        ("finish", {"lesson": "The child answer changed the parent experiment."}),
    ])
    provider.parent_returned = False
    agent.provider = provider

    result = inquire(agent, lab)

    assert [r["llm_inquiry"]["question_id"] for r in provider.requests] == [
        LEAF["id"], LEAF["id"], CHILD["id"], CHILD["id"]]
    assert provider.requests[0]["llm_inquiry"]["assignment"]["question"] == LEAF["question"]
    assert lab.actions == [{"action": "TALK", "arg1": 7},
                           {"action": "TELEPORT_TO_OBJECT", "arg1": target}]
    assert question(agent, LEAF["id"])["acquired_observations"] == [0]
    assert question(agent, CHILD["id"])["acquired_observations"] == [1]
    assert not agent._pending_dependencies(question(agent, CHILD["id"]))
    assert result["status"] == "inquiry_finished" and result["model_calls"] == 4
    assert agent.llm_inquiries[LEAF["id"]]["model_calls"] == 2
    assert agent.llm_inquiries[CHILD["id"]]["model_calls"] == 2
    assert agent.costs["model_calls"] == 4 and agent.costs["reported_tokens"] == 4 * USAGE["total_tokens"]
    assert agent.research_focus == outer_focus
    assert (tmp_path / "research_notes.md").read_text() == "Parent: compare current controls."
    event, = host_events(agent)
    assert event["action"] == "run" and event["arguments"]["operator"] == "inquire"
    assert event["arguments"]["question"]["id"] == LEAF["id"]
    assert "model_call" not in event and "usage" not in event


@pytest.mark.parametrize("verdict", ["supported", "refuted", "inconclusive"])
def test_child_can_analyze_existing_parent_evidence_without_relabeling_its_acquisition(tmp_path, verdict):
    from test_discoveryworld_llm_inquiry import ANALYSIS

    analysis = ANALYSIS.replace("'supported' if active else 'refuted'", repr(verdict))
    steps = [("evaluate", {"question_id": LEAF["id"], "path": "shared/old_answer.py", "code": analysis})]
    if verdict == "inconclusive":
        steps.append(("finish", {"lesson": "The old evidence is inconclusive."}))
    steps.append(("finish", {"lesson": "Use the returned analysis for the next parent decision."}))
    agent, provider, lab = prepared(tmp_path, steps)
    register(agent, LEAF)
    agent._execute("query", {"question": {"id": CHILD["id"]},
                             "actions": [{"action": "TALK", "arg1": 7}]}, lab)

    result = inquire(agent, lab)

    child = question(agent, LEAF["id"])
    assert child["status"] == verdict and child["evidence"] == [0]
    assert child["acquired_observations"] == []
    assert question(agent, CHILD["id"])["acquired_observations"] == [0]
    assert not agent._pending_dependencies(question(agent, CHILD["id"]))
    assert len(lab.actions) == 1 and result["status"] == "inquiry_finished"
    assert provider.requests[-1]["llm_inquiry"]["question_id"] == CHILD["id"]
    assert len(host_events(agent)) == 1


def test_local_finish_is_attempted_once_per_parent_invocation_then_can_be_resumed(tmp_path):
    agent, provider, lab = prepared(tmp_path, [("finish", {"lesson": "No evidence yet."})] * 4)
    register(agent, LEAF)

    first = inquire(agent, lab)
    second = inquire(agent, lab)

    assert first["status"] == second["status"] == "inquiry_finished"
    assert [r["llm_inquiry"]["question_id"] for r in provider.requests] == [
        LEAF["id"], CHILD["id"], LEAF["id"], CHILD["id"]]
    assert agent.llm_inquiries[LEAF["id"]]["invocations"] == 2
    assert agent.llm_inquiries[CHILD["id"]]["model_calls"] == 2
    assert len(host_events(agent)) == 2 and not lab.actions
    assert "answer" not in question(agent, LEAF["id"])


def test_explicit_child_inquiry_is_not_automatically_reopened_after_local_finish(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("run", {"operator": "inquire", "question": LEAF, "max_model_calls": 2}),
        ("finish", {"lesson": "Return an unresolved child."}),
        ("finish", {"lesson": "Parent accepts that research is unresolved."}),
    ])

    result = inquire(agent, lab)

    assert result["status"] == "inquiry_finished" and result["model_calls"] == 3
    assert [r["llm_inquiry"]["question_id"] for r in provider.requests] == [
        CHILD["id"], LEAF["id"], CHILD["id"]]
    assert agent.llm_inquiries[LEAF["id"]]["invocations"] == 1
    assert not host_events(agent) and not lab.actions


def test_nested_unmeasured_dependencies_are_researched_from_leaf_back_to_parent(tmp_path):
    middle = {"id": "instrument", "question": "Can this instrument be used?",
              "parent_id": CHILD["id"], "depends_on": []}
    leaf = {**LEAF, "parent_id": middle["id"]}
    agent, provider, lab = prepared(tmp_path, [evaluate(leaf["id"]), evaluate(middle["id"]),
        ("finish", {"lesson": "Both dependencies returned their actual evidence."})])
    register(agent, middle, leaf)
    agent._execute("query", {"question": {"id": "task"},
                             "actions": [{"action": "TALK", "arg1": 7}]}, lab)

    result = inquire(agent, lab)

    assert [r["llm_inquiry"]["question_id"] for r in provider.requests] == [
        leaf["id"], middle["id"], CHILD["id"]]
    assert [r["llm_inquiry"]["depth"] for r in provider.requests] == [3, 2, 1]
    assert result["model_calls"] == 3 and len(lab.actions) == 1
    assert len(host_events(agent)) == 2
    assert question(agent, leaf["id"])["parent_id"] == middle["id"]
    assert not agent._pending_dependencies(question(agent, CHILD["id"]))


@pytest.mark.parametrize("excluded", ["outside", "numeric", "measured", "reviewed", "unknown"])
def test_auto_inquiry_does_not_steal_other_scopes_or_existing_research_routes(tmp_path, excluded):
    agent, provider, lab = prepared(tmp_path, [("finish", {"lesson": "Return to the caller."})])
    register(agent)
    if excluded == "unknown":
        agent._question({"id": CHILD["id"], "depends_on": ["not_registered"]})
    else:
        child = agent._question({**LEAF, "parent_id": "task" if excluded == "outside" else CHILD["id"]})
        if excluded == "outside":
            agent._question({"id": CHILD["id"], "depends_on": [child["id"]]})
        elif excluded == "numeric":
            child["binding_path"] = "shared/numerical_relation.py"
        elif excluded == "measured":
            agent._execute("query", {"question": LEAF, "actions": [{"action": "TALK", "arg1": 7}]}, lab)
        elif excluded == "reviewed":
            agent._execute("query", {"question": {"id": "task"}, "actions": [{"action": "TALK", "arg1": 7}]}, lab)
            agent._record_answer(child, {"status": "refuted", "answer": "The condition does not hold", "evidence": [0]})
            # Explicit premises still need support, but a completed negative
            # child is not a reason to rerun its model researcher automatically.
            agent._question({"id": CHILD["id"], "depends_on": [child["id"]]})

    assignment = deepcopy(question(agent, CHILD["id"]))
    result = inquire(agent, lab, question=assignment)

    assert result["status"] == "inquiry_finished" and result["model_calls"] == 1
    assert len(provider.requests) == 1
    assert provider.requests[0]["llm_inquiry"]["question_id"] == CHILD["id"]
    assert not host_events(agent) and LEAF["id"] not in agent.llm_inquiries


def test_cyclic_parent_links_never_reenter_an_active_inquiry(tmp_path):
    agent, provider, lab = prepared(tmp_path, [
        ("finish", {"lesson": "The active parent cannot supply its own prerequisite yet."}),
        ("finish", {"lesson": "The dependency remains unresolved."}),
    ])
    register(agent, LEAF)
    root = agent._question({"id": CHILD["id"], "parent_id": LEAF["id"]})

    result = inquire(agent, lab, question=deepcopy(root))

    assert result["status"] == "inquiry_finished" and result["model_calls"] == 2
    assert [r["llm_inquiry"]["question_id"] for r in provider.requests] == [LEAF["id"], CHILD["id"]]
    assert [r["llm_inquiry"]["depth"] for r in provider.requests] == [2, 1]
    assert agent.llm_inquiries[CHILD["id"]]["invocations"] == 1
    assert len(host_events(agent)) == 1


def test_ordinary_code_arm_retains_model_control_of_delegation(tmp_path):
    agent, provider, lab = prepared(tmp_path, [("finish", {"lesson": "No delegation selected."})])
    agent = type(agent)(provider, tmp_path, arm="code_archive", limits=agent.limits)
    agent.lab = lab
    agent._question({"id": "task", "question": "Explain the control response"})
    register(agent, LEAF)

    result = inquire(agent, lab)

    assert result["model_calls"] == 1 and not host_events(agent)
    assert provider.requests[0]["llm_inquiry"]["question_id"] == CHILD["id"]


@pytest.mark.parametrize("calls,total_limit,child_calls", [
    (1, 30, 0), (2, 30, 1), (4, 30, 3), (11, 30, 8), (8, 3, 2)])
def test_auto_child_shares_hard_call_ceiling_and_leaves_parent_one_real_decision(
        tmp_path, calls, total_limit, child_calls):
    steps = [("run", {"code": "print('The research remains unresolved')"})] * child_calls
    steps.append(("finish", {"lesson": "Parent records the unresolved child."}))
    agent, provider, lab = prepared(tmp_path, steps, limits={"max_model_calls": total_limit})
    register(agent, LEAF)

    result = inquire(agent, lab, calls=calls)

    assert [r["llm_inquiry"]["question_id"] for r in provider.requests] == [LEAF["id"]] * child_calls + [CHILD["id"]]
    assert result["model_calls"] == agent.costs["model_calls"] == child_calls + 1
    assert agent.llm_inquiries[CHILD["id"]]["model_calls"] == 1
    assert agent.costs["reported_tokens"] == (child_calls + 1) * USAGE["total_tokens"]
    if child_calls:
        assert agent.llm_inquiries[LEAF["id"]]["model_calls"] == child_calls
        assert agent.llm_inquiries[LEAF["id"]]["status"] == "model_call_budget_exhausted"
    else:
        assert LEAF["id"] not in agent.llm_inquiries
    assert result["status"] == "inquiry_finished" and not lab.actions
