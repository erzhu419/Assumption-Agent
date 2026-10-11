"""Anonymous consumers retain failed Python evidence without a meter workflow."""

import json
from random import Random

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.anonymous_method_learning import AnonymousPreparedMethodAgent
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


@pytest.mark.parametrize("agent_class,arm", [
    (CommonArchiveInquiryAgent, "code_react"),
    (CommonArchiveInquiryAgent, "hegel"),
    (AnonymousPreparedMethodAgent, "hegel"),
])
def test_mismatched_predictor_without_fit_keeps_feedback_without_meter_routing(
        agent_class, arm, tmp_path):
    prior = "archive/source_000/_evaluated_model/model.py"
    code = "def predict(row):\n    return 0.0\n"
    agent = agent_class(None, tmp_path / "consumer", arm=arm,
        knowledge={"files": {prior: code}, "research_roles": {"predictor": prior}},
        source_task=False, config=InquiryLLMConfig(observation_budget=576,
            confirmation_rows=96, compute_budget=250000, execution_seconds=30.,
            max_model_calls=0, token_budget=0))
    lab = AnonymousHistoryLab("memoryless", seed=31)
    assert lab.public_spec()["channels"] == []
    prefix = lab.confirmation_actions(72, Random(101))
    acquired = agent._execute("query", {"actions": prefix}, lab)
    assert "error" not in acquired

    scored = agent._execute("evaluate", {"model": prior}, lab)
    assert scored["returncode"] == 0 and scored["mse"] > 1e-10
    assert "research_advice" not in scored
    assert "available_research_action" not in scored

    # The old runtime proposed this real action after a mismatch, then crashed
    # while seeking a nonexistent meter when this fixed predictor had no fit.
    refit = agent._execute("run", {"operator": "refit", "model": prior,
        "path": "shared/local_refit"}, lab)
    assert refit["status"] == "refit_unavailable"
    assert refit["validation"]["fit_available"] is False
    assert refit["path"] is None
    assert "research_advice" not in refit
    assert "available_research_action" not in refit
    assert agent.operator_results[-1]["status"] == "refit_unavailable"
    assert agent.costs["observation_cost"] == 72
    assert agent.costs["measurement_cost"] == 0
    assert agent.costs["compute_cost"] >= 144
    assert agent.costs["model_calls"] == 0
    assert (agent.workspace / prior).read_text() == code

    agent._prepare_program_state()
    state = json.loads((agent.workspace / "research_state.json").read_text())
    assert "research_advice" not in state
    assert state["operator_results"][-1]["status"] == "refit_unavailable"
