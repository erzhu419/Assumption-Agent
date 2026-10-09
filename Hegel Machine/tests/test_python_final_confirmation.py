"""A separately validated repair can use its last live call for confirmation."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.python_research_routing import evaluate_action
from test_python_research_procedure import assert_terminal, consumer, source_archive


def follow(agent, lab, advice):
    action = advice["next_action"]
    return agent._execute(action["action"], action["arguments"], lab)


def ready_to_validate(tmp_path, archive, mode):
    agent, lab, _ = consumer(tmp_path / "consumer", archive, mode=mode)
    advice = agent._current_research_advice
    if mode == "recurrent":
        refitted = follow(agent, lab, advice)
        assert refitted["status"] == "refit_inadequate_on_validation"
        measured = follow(agent, lab, refitted["research_advice"])
        assert any(item.get("status") == "supported_on_measured_trajectory"
                   for item in measured["after_observation_results"])
        advice = measured["research_advice"]
        assert advice["next_action"]["arguments"]["operator"] == "response"
    else:
        assert advice["next_action"]["arguments"]["operator"] == "refit"
    return agent, lab, advice


@pytest.mark.parametrize("mode", ["finite2", "recurrent"])
@pytest.mark.parametrize("remaining_calls", [1, 2])
def test_validated_repairs_use_the_final_call_without_skipping_fresh_confirmation(
        tmp_path, source_archive, mode, remaining_calls):
    agent, lab, proposal = ready_to_validate(tmp_path, source_archive, mode)
    agent.costs["model_calls"] = agent.config.max_model_calls - remaining_calls
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    result = follow(agent, lab, proposal)
    if mode == "finite2":
        assert result["status"] == "refit_supported_on_validation"
        model = result["path"]
    else:
        best = min(result["candidates"], key=lambda item: item["validation_mse"])
        assert best["validation_mse"] < 1e-10
        model = best["path"]
    advice = result["research_advice"]
    expected_action = "finish" if remaining_calls == 1 else "evaluate"
    assert advice["next_action"] == {"action": expected_action, "arguments": {"model": model}}
    assert lab.costs == lab_before
    assert agent.costs["compute_cost"] - before["compute_cost"] == result["compute_cost"]
    assert agent.model is None and not agent.confirmations
    assert "procedure_action" not in advice  # The unadopted live offer is no longer production behavior.

    if remaining_calls == 1:
        needed = len(agent.rows) + agent.config.confirmation_rows
        assert advice["decision"]["quoted_compute_cost"] == needed
        assert advice["decision"]["quoted_observation_cost"] == agent.config.confirmation_rows
        if mode == "recurrent":
            # The old incumbent is stale after the meter purchase. Direct
            # finish does not compare it again, so exactly this budget suffices.
            agent.config = replace(agent.config, compute_budget=agent.costs["compute_cost"] + needed)
            resumed = agent._execute("run", {"code": "print('ready for confirmation')"}, lab)
            assert resumed["research_advice"]["next_action"] == advice["next_action"]
    else:
        evaluated = follow(agent, lab, advice)
        assert evaluated["mse"] < 1e-10
        advice = evaluated["research_advice"]
        assert advice["next_action"]["action"] == "finish"
    before = deepcopy(agent.costs)
    submitted = follow(agent, lab, advice)
    assert submitted["fresh_confirmation"]["passed"]
    assert agent.costs["observation_cost"] - before["observation_cost"] == agent.config.confirmation_rows
    assert agent.costs["compute_cost"] - before["compute_cost"] == len(agent.rows)
    assert len(agent.confirmations) == 1 and not agent.events
    assert_terminal(agent, lab)


def test_unvalidated_candidate_still_needs_evaluation_with_one_live_call(tmp_path, source_archive):
    agent, lab, model = consumer(tmp_path / "consumer", source_archive)
    agent.costs["model_calls"] = agent.config.max_model_calls - 1
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    offer = evaluate_action(agent, model, "unvalidated_candidate")
    assert offer["next_action"] == {"action": "evaluate", "arguments": {"model": model}}
    assert agent.costs == before and lab.costs == lab_before
    assert agent.model is None and not agent.confirmations


def test_a_changed_candidate_still_fails_actual_fresh_confirmation(tmp_path, source_archive):
    agent, lab, proposal = ready_to_validate(tmp_path, source_archive, "finite2")
    agent.costs["model_calls"] = agent.config.max_model_calls - 1
    validated = follow(agent, lab, proposal)
    advice = validated["research_advice"]
    assert advice["next_action"]["action"] == "finish"
    model = advice["next_action"]["arguments"]["model"]
    # Alter the actual candidate, not the evaluator's return value. Its earlier
    # validation cannot authorize acceptance after new predictive errors.
    agent._execute("write", {"path": model, "code": "def predict(row):\n return 0.\n"}, lab)
    before = deepcopy(agent.costs)
    failed = follow(agent, lab, advice)
    assert failed["fresh_confirmation"]["passed"] is False
    assert failed["fresh_confirmation"]["mse"] > 1e-4
    assert agent.model is None and len(agent.confirmations) == 1
    assert agent.costs["observation_cost"] - before["observation_cost"] == agent.config.confirmation_rows


@pytest.mark.parametrize("resource", ["compute", "observations"])
def test_final_confirmation_is_not_offered_without_its_actual_remaining_budget(
        tmp_path, source_archive, resource):
    agent, lab, proposal = ready_to_validate(tmp_path, source_archive, "finite2")
    agent.costs["model_calls"] = agent.config.max_model_calls - 1
    if resource == "compute":
        agent.config = replace(agent.config, compute_budget=agent.costs["compute_cost"]
            + 2 * len(agent.rows) + agent.config.confirmation_rows - 1)
    else:
        agent.config = replace(agent.config, observation_budget=agent.costs["observation_cost"]
            + agent.config.confirmation_rows - 1)
    lab_before = deepcopy(lab.costs)
    validated = follow(agent, lab, proposal)
    assert validated["status"] == "refit_supported_on_validation"
    assert not validated["research_advice"].get("next_action")
    assert lab.costs == lab_before and agent.model is None and not agent.confirmations
