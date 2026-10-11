"""Premise obligations reuse authored contrasts without replacing response bindings."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.world_reference_response import (
    response_premise_design_action,
    return_response_to_author,
)
from test_discoveryworld_contrast import SOURCE, prepared, question
from test_discoveryworld_llm_inquiry import Provider


BINDING = "shared/retained_response.py"
DECLARATION = {"id": "response_family", "parent_id": "task", "depends_on": [],
               "question": "Which response family explains the paid outcomes?"}
OLD_SOURCE = "def bind(spec, observations, memory):\n    return {'retained': True}\n"


def comparison(role="shared_premise_research"):
    return {"binding_path": BINDING, "question_id": DECLARATION["id"],
            "experiment_action": None,
            "revision_action": {"action": "run", "arguments": {
                "operator": "propose_explanations", "path": BINDING,
                "response_path": "shared/controller.py", "question": deepcopy(DECLARATION)}},
            "family_research": {"selected_role": role, "research_obligation": {
                "reason": "completed_tests_changed_members_without_task_or_nonlocal_progress",
                "question": "Which shared response premise needs a distinguishing observation?"}}}


@pytest.mark.parametrize("role", [None, "family_discrimination"])
def test_no_premise_design_action_without_a_premise_obligation(role):
    value = comparison(role)
    before = deepcopy(value)
    assert response_premise_design_action(value) is None
    assert value == before


def test_premise_offer_runs_ordinary_proposal_then_parent_executes_authored_contrast(tmp_path):
    agent, lab = prepared(tmp_path)
    agent._question(DECLARATION)
    agent._execute("write", {"path": BINDING, "code": OLD_SOURCE}, lab)
    value = comparison()
    unchanged = deepcopy(value)
    action = response_premise_design_action(value)
    arguments = action["arguments"]
    assert arguments["operator"] == "propose_explanations"
    assert "response_path" not in arguments
    assert arguments["path"] != BINDING
    qid = arguments["question"]["id"]
    assert qid != DECLARATION["id"]
    assert arguments["question"]["parent_id"] == DECLARATION["parent_id"]
    source = SOURCE.replace("== 'physical'", f"== {qid!r}")
    provider = agent.provider = Provider([
        ("write", {"path": arguments["path"], "code": source}),
        ("finish", {"lesson": "Return the authored predictions for the parent's choice."}),
    ])
    questions_before, actions_before = deepcopy(agent.questions), deepcopy(lab.actions)

    result = agent._execute(action["action"], {**arguments, "max_model_calls": 2}, lab)

    assert result["status"] == "inquiry_finished"
    assert result["model_calls"] == len(provider.requests) == 2
    assert agent.status == "running"
    assert agent.questions == questions_before and lab.actions == actions_before
    assert all("response_path" not in request["llm_inquiry"] for request in provider.requests)
    assert result["proposal"]["comparison_action"] is None
    adopted = result["proposal"]["contrast_action"]
    assert adopted["arguments"]["operator"] == "contrast"

    executed = agent._execute(adopted["action"], adopted["arguments"], lab)

    assert executed["diagnosis"] == "discriminating_experiment_available"
    assert lab.actions[len(actions_before):] == [{"action": "USE", "arg1": 8, "arg2": 7}]
    answer = question(agent, qid)
    assert answer["status"] == "refuted"
    assert answer["evidence"] == [1]
    assert answer["answer"] == {"observed_outcome": False, "tested_prediction": True}
    assert question(agent, DECLARATION["parent_id"])["status"] == "reopened"
    assert "answer" not in question(agent, DECLARATION["id"])
    assert (tmp_path / BINDING).read_text() == OLD_SOURCE
    assert value == unchanged


def test_restored_proposal_derives_latest_premise_offer_without_research_or_acquisition(tmp_path):
    agent, lab = prepared(tmp_path)
    agent._question(DECLARATION)
    latest = comparison()
    # Saved sessions predate the new action field; their comparison is already paid.
    saved = {"question_id": DECLARATION["id"], "notes": "The premise remains unresolved.",
             "events": [], "proposal": {"path": BINDING, "comparison_result": latest,
                 "experiment_action": None, "revision_action": latest["revision_action"]}}
    agent.llm_inquiries["proposal:" + BINDING] = json.loads(json.dumps(saved))
    before_costs, before_actions = deepcopy(agent.costs), deepcopy(lab.actions)

    agent._prepare_program_state()

    view = json.loads((tmp_path / "method_decision.json").read_text())
    retained, = view["explanation_proposals"]
    assert retained["premise_design_action"] == response_premise_design_action(latest)
    assert agent.costs == before_costs and lab.actions == before_actions
    assert not agent.provider.requests

    # A later paid return removes that obligation. Its old offer must not survive.
    returned = comparison("family_discrimination")
    return_response_to_author(agent, returned)
    session = agent.llm_inquiries["proposal:" + BINDING]
    assert session["proposal"]["comparison_result"] == returned
    assert session["proposal"]["premise_design_action"] is None
    agent._prepare_program_state()
    refreshed = json.loads((tmp_path / "method_decision.json").read_text())
    assert refreshed["explanation_proposals"][0]["premise_design_action"] is None
    assert agent.costs == before_costs and lab.actions == before_actions
    assert not agent.provider.requests
