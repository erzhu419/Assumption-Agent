"""Exercise live-session orchestration using decisions without external API calls."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.legacy_research_bridge import LEGACY_KNOWLEDGE_PATH, METHODS_PATH
from hegel_machine.cognition.legacy_session_bridge import LegacyInquirySession
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from scripts import run_legacy_session_llm as runner


@pytest.fixture
def one_case(monkeypatch):
    plan = runner.protocol()
    plan["cases"] = plan["cases"][:1]
    monkeypatch.setattr(runner, "protocol", lambda: deepcopy(plan))
    return plan


def test_matching_preparation_changes_library_without_running_bound_research(tmp_path, monkeypatch, one_case):
    common = runner.prepare_case(one_case["cases"][0], tmp_path / "common", one_case)

    def no_host_action(*args, **kwargs):
        raise AssertionError("binding the session cannot run research or finish for the model")

    with monkeypatch.context() as patch:
        patch.setattr(LegacyInquirySession, "run", no_host_action)
        patch.setattr(TemporalOperatorAgent, "_execute", no_host_action)
        arms = [runner.prepare_arm(common, condition, tmp_path / condition, one_case,
                                   provider_factory=lambda path: None)
                for condition in runner.CONDITIONS]
    retained, local = [arm["episodes"][0]["agent"] for arm in arms]
    assert retained.observations == local.observations
    assert retained.rows == local.rows
    assert retained.costs == local.costs == common["setup_costs"]["consumer_prefix"]
    assert retained.common_interface == local.common_interface
    assert retained.config == local.config
    assert retained.confirmation_rng.getstate() == local.confirmation_rng.getstate()
    assert retained.model is local.model is None
    assert retained.confirmations == local.confirmations == []
    assert json.loads(retained.files[LEGACY_KNOWLEDGE_PATH])["concepts"]
    assert json.loads(local.files[LEGACY_KNOWLEDGE_PATH])["concepts"] == []
    for path in set(retained.files) | set(local.files):
        if path not in {LEGACY_KNOWLEDGE_PATH, METHODS_PATH}:
            assert retained.files[path] == local.files[path]
    for arm in arms:
        state = arm["bridge"].to_dict()
        assert state["revisits"] == state["pending_revisits"] == state["evaluation"] == []
        assert arm["episodes"][1]["agent"].costs == common["setup_costs"]["source"]
        assert arm["setup_costs"] == common["setup_costs"]


def test_model_selected_native_actions_finish_before_terminal_and_charge_source_once(
        tmp_path, monkeypatch, one_case):
    one_case["limits"]["max_model_calls"] = 2
    requests, completed, evaluated = {}, [], []

    class ResearchThenFinish:
        def __init__(self, path):
            self.path = path
            self.condition = path.parent.name
            self.calls = 0

        def next_action(self, request):
            self.calls += 1
            requests.setdefault(self.condition, []).append(deepcopy(request))
            state = json.loads((self.path / "research_state.json").read_text())
            assert state["costs"]["observation_cost"] == request["costs"]["observation_cost"]
            if self.calls == 1:
                assert not request["events"]
                assert state["legacy_session"]["tasks"][0]["status"] == "unresolved"
                operator = "legacy_revisit" if self.condition == "retained" else "legacy_research"
                return {"action": "run", "arguments": {"operator": operator},
                        "usage": {"total_tokens": 7, "input_tokens": 6, "cached_input_tokens": 3}}
            assert self.calls == 2
            feedback = request["events"][-1]["feedback"]
            assert "error" not in feedback
            model = feedback.get("model_path") or feedback.get("path")
            assert model
            assert state["operator_results"][-1]["operator"] == feedback["operator"]
            return {"action": "finish", "arguments": {"model": model},
                    "usage": {"total_tokens": 5, "input_tokens": 4, "cached_input_tokens": 2}}

    real_run = TemporalOperatorAgent.run
    real_evaluate = TemporalInquiryLab.evaluation_rows

    def record_completion(self, laboratory, **kwargs):
        result = real_run(self, laboratory, **kwargs)
        completed.append(self.workspace.parent.name)
        return result

    def terminal_after_training(self, **kwargs):
        assert sorted(completed) == sorted(runner.CONDITIONS)
        evaluated.append(self)
        return real_evaluate(self, **kwargs)

    monkeypatch.setattr(TemporalOperatorAgent, "run", record_completion)
    monkeypatch.setattr(TemporalInquiryLab, "evaluation_rows", terminal_after_training)
    result = runner.run_study(tmp_path / "run", provider_factory=ResearchThenFinish, max_workers=1)
    assert result["registered_episodes"] == result["scored_episodes"] == 2
    assert not result["failures"] and len(evaluated) == 2
    for item in result["episodes"]:
        condition = item["condition"]
        assert len(requests[condition]) == 2
        assert item["llm_actions"] == {"run": 1, "finish": 1}
        operator = "legacy_revisit" if condition == "retained" else "legacy_research"
        assert item["llm_operator_choices"] == {operator: 1}
        if condition == "retained":
            assert item["status"] == "completed" and item["confirmed_and_correct"]
            assert item["submitted_source_binding"] == "matching_frozen_session_predictor"
            assert item["confirmed_positive_final_source_returns"] == 1
            assert item["final_source_returns"][0]["confirmation_status"] == "confirmed"
        else:
            # The native task-local researcher failed this paid-prefix case.
            # Its attempted finish must remain a measured failure, without a
            # host-supplied predictor or inherited-source credit.
            assert item["status"] == "model_call_budget_exhausted"
            assert not item["fresh_confirmed"] and not item["confirmed_and_correct"]
            assert item["confirmed_positive_final_source_returns"] == 0
            assert item["final_source_returns"] == []
        assert item["consumer_costs"]["model_calls"] == 2
        assert item["consumer_costs"]["reported_tokens"] == 12
        assert item["cached_input_tokens"] == 5
        for metric in ("observation_cost", "measurement_cost", "compute_cost"):
            assert item["session_costs"][metric] == (
                item["consumer_costs"][metric] + item["setup_costs"]["source"][metric])
            assert item["consumer_continuation_costs"][metric] == (
                item["consumer_costs"][metric] - item["setup_costs"]["consumer_prefix"][metric])
    retained, local = [requests[condition][0] for condition in runner.CONDITIONS]
    for field in ("limits", "capabilities", "observations", "laboratory_observations", "task_interface"):
        assert retained[field] == local[field]


def test_provider_failure_is_saved_once_without_host_research_finish_or_replacement(tmp_path, one_case):
    calls = []

    class FailedProvider:
        def __init__(self, path):
            self.path = path

        def next_action(self, request):
            calls.append(self.path)
            assert not request["events"]
            return {"action": "provider_error", "error": {"message": "test transport failure"},
                    "usage": {"total_tokens": 7}}

    output = tmp_path / "failed"
    result = runner.run_study(output, provider_factory=FailedProvider, max_workers=1)
    assert len(calls) == len(set(calls)) == 2
    assert result["registered_episodes"] == result["scored_episodes"] == 2
    for item in result["episodes"]:
        assert item["status"] == "provider_failed" and item["model"] is None
        assert item["llm_operator_choices"] == {}
        assert item["session_revisits"] == []
        assert not item["fresh_confirmed"] and not item["confirmed_and_correct"]
        assert item["consumer_costs"]["model_calls"] == 1
        assert item["consumer_costs"]["reported_tokens"] == 7
        assert item["consumer_continuation_costs"]["observation_cost"] == 0
        assert item["consumer_continuation_costs"]["compute_cost"] == 0
        path = output / f"case_{item['apparatus_seed']}" / item["condition"] / "training.json"
        saved = json.loads(path.read_text())
        assert saved["events"][0]["error"]["message"] == "test transport failure"
    with pytest.raises(ValueError, match="preserved"):
        runner.run_study(output, provider_factory=FailedProvider)
    assert len(calls) == 2
