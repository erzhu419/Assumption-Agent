"""A research procedure is an explicit option, never an automatic action."""

from copy import deepcopy

import pytest

from scripts.run_python_procedure_offer_study import FeedbackOfferAgent
from test_python_research_procedure import (
    PROCEDURE, assert_no_llm_activity, assert_terminal, consumer, purchase,
    source_archive,
)


PROCEDURE_ACTION = {"action": "run", "arguments": {"operator": "procedure",
    "path": PROCEDURE, "max_steps": 12, "confirm": True}}


def unevaluated_consumer(tmp_path, archive, *, changed=True):
    agent, lab, model = consumer(tmp_path / "consumer", archive,
        mode="recurrent" if changed else "finite2", changed=changed, paid=False,
        agent_class=FeedbackOfferAgent)
    purchase(agent, lab, seed=79)
    return agent, lab, model


@pytest.mark.parametrize("trigger", ["evaluate", "refit"])
def test_failure_offers_a_procedure_without_replacing_the_next_action_or_executing_it(
        tmp_path, source_archive, trigger):
    agent, lab, model = unevaluated_consumer(tmp_path, source_archive)
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    feedback = agent._execute("evaluate", {"model": model}, lab)
    assert feedback["mse"] > 1e-4
    assert agent.costs["compute_cost"] - before["compute_cost"] == len(agent.rows)
    assert lab.costs == lab_before
    if trigger == "refit":
        next_action = feedback["research_advice"]["next_action"]
        assert next_action["action"] == "run" and next_action["arguments"]["operator"] == "refit"
        before = deepcopy(agent.costs)
        feedback = agent._execute(next_action["action"], next_action["arguments"], lab)
        assert feedback["status"] == "refit_inadequate_on_validation"
        assert agent.costs["compute_cost"] - before["compute_cost"] == len(agent.rows)
    advice = feedback["research_advice"]
    assert advice["procedure_action"] == PROCEDURE_ACTION
    ordinary = advice["next_action"]
    if trigger == "evaluate":
        assert ordinary["action"] == "run" and ordinary["arguments"]["operator"] == "refit"
        assert ordinary["arguments"]["model"] == feedback["evaluated_model"]["model"]
    else:
        assert ordinary["action"] == "query"
        assert [step["operator"] for step in ordinary["arguments"]["after_observation"]] == [
            "instrument", "assess_instrument"]
    assert lab.costs == lab_before
    assert not agent.method_results and not agent.confirmations and agent.model is None
    assert_no_llm_activity(agent, [])

    # Only the caller's explicit selection now executes the complete procedure.
    selected = advice["procedure_action"]
    result = agent._execute(selected["action"], selected["arguments"], lab)
    assert result["status"] == "completed" and agent.confirmations[-1]["passed"]
    assert agent.method_results and len(agent.confirmations) == 1
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"]
    assert agent.costs["measurement_cost"] == lab.costs["measurement_cost"]
    assert_no_llm_activity(agent, [])
    assert_terminal(agent, lab)


def test_correct_candidate_keeps_direct_confirmation_without_a_procedure_offer(tmp_path, source_archive):
    agent, lab, model = unevaluated_consumer(tmp_path, source_archive, changed=False)
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    feedback = agent._execute("evaluate", {"model": model}, lab)
    assert feedback["mse"] < 1e-10
    assert feedback["research_advice"]["next_action"] == {
        "action": "finish", "arguments": {"model": feedback["evaluated_model"]["model"]}}
    assert "procedure_action" not in feedback["research_advice"]
    assert "available_procedure_action" not in feedback
    assert agent.costs["compute_cost"] - before["compute_cost"] == len(agent.rows)
    assert lab.costs == lab_before and not agent.method_results


def test_missing_procedure_file_keeps_ordinary_repair_without_an_unexecutable_offer(tmp_path, source_archive):
    agent, lab, model = unevaluated_consumer(tmp_path, source_archive)
    (agent.workspace / PROCEDURE).unlink()
    feedback = agent._execute("evaluate", {"model": model}, lab)
    assert feedback["mse"] > 1e-4
    assert feedback["research_advice"]["next_action"]["arguments"]["operator"] == "refit"
    assert "procedure_action" not in feedback["research_advice"]
    assert "available_procedure_action" not in feedback
    assert not agent.method_results and not agent.confirmations


def test_available_action_fallback_can_offer_the_procedure_when_detailed_advice_is_hidden(
        tmp_path, source_archive, monkeypatch):
    agent, lab, model = unevaluated_consumer(tmp_path, source_archive)
    monkeypatch.setattr(agent, "_render_research_advice", lambda advice: None)
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    feedback = agent._execute("evaluate", {"model": model}, lab)
    assert feedback["mse"] > 1e-4 and "research_advice" not in feedback
    assert feedback["available_research_action"] == {
        "action": "run", "arguments": {"operator": "research_choice"}}
    assert feedback["available_procedure_action"] == PROCEDURE_ACTION
    assert agent.costs["compute_cost"] - before["compute_cost"] == len(agent.rows)
    assert lab.costs == lab_before and not agent.method_results
    assert_no_llm_activity(agent, [])
