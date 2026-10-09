"""Keep the historical paid JSON repair behavior in the world adapter."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import InstrumentLab, LIMITS


USAGE = {"input_tokens": 11, "output_tokens": 2, "cached_input_tokens": 7,
         "total_tokens": 13}
DECODE_ERROR = {"decode_error": "JSONDecodeError: Unterminated string"}


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
def test_paid_decode_error_is_feedback_before_repaired_world_action(tmp_path, arm):
    class RepairingProvider:
        def next_action(self, request):
            if not request["events"]:
                return {"action": "provider_error", "arguments": {
                    "actions": [{"action": "TUNE", "arg1": 6}],
                    "notes": "This malformed reply must not change the notebook."},
                    "error": DECODE_ERROR, "usage": USAGE}
            previous = request["events"][-1]
            assert previous["action"] == "provider_error"
            assert previous["feedback"]["error"] == DECODE_ERROR["decode_error"]
            assert request["costs"]["model_calls"] == 1
            assert request["costs"]["reported_tokens"] == 13
            assert request["costs"]["environment_steps"] == 0
            assert request["evidence"]["observation_count"] == 0
            assert not request["research_notes"]
            return {"action": "query", "arguments": {
                "actions": [{"action": "MEASURE"}, {"action": "TUNE", "arg1": 6}]},
                "usage": USAGE}

    lab = InstrumentLab()
    checkpoints = []
    agent = DiscoveryWorldAgent(RepairingProvider(), tmp_path, arm=arm,
                                limits={**LIMITS, "max_model_calls": 2})
    result = agent.run(lab, on_checkpoint=lambda state: checkpoints.append(deepcopy(state)))

    assert result["status"] == "completed"
    assert lab.actions == [{"action": "MEASURE"}, {"action": "TUNE", "arg1": 6}]
    assert [event["action"] for event in result["events"]] == ["provider_error", "query"]
    assert result["costs"]["model_calls"] == 2
    assert result["costs"]["reported_tokens"] == 26
    assert result["costs"]["input_tokens"] == 22
    assert result["costs"]["output_tokens"] == 4
    assert result["costs"]["cached_input_tokens"] == 14
    assert result["costs"]["environment_steps"] == 2
    assert any(state["status"] == "running" and len(state["events"]) == 1
               and state["costs"]["reported_tokens"] == 13 for state in checkpoints)


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
@pytest.mark.parametrize("usage", [USAGE, {}])
def test_repeated_decode_errors_exhaust_original_call_limit(tmp_path, arm, usage):
    class MalformedProvider:
        def next_action(self, request):
            return {"action": "provider_error", "arguments": {},
                    "error": DECODE_ERROR, "usage": usage}

    lab = InstrumentLab()
    agent = DiscoveryWorldAgent(MalformedProvider(), tmp_path, arm=arm,
                                limits={**LIMITS, "max_model_calls": 3})
    result = agent.run(lab)

    assert result["status"] == "model_call_budget_exhausted"
    assert result["costs"]["model_calls"] == len(result["events"]) == 3
    assert result["costs"]["reported_tokens"] == 3 * usage.get("total_tokens", 0)
    assert result["costs"]["cached_input_tokens"] == 3 * usage.get("cached_input_tokens", 0)
    assert result["costs"]["usage_missing_calls"] == (0 if usage else 3)
    assert result["costs"]["environment_steps"] == 0
    assert not lab.actions


@pytest.mark.parametrize("error,usage", [
    ({"returncode": 1, "timeout": False, "messages": ["service unavailable"]}, USAGE),
    ({"returncode": -15, "timeout": True}, {}),
])
def test_service_failure_still_stops(tmp_path, error, usage):
    class FailedProvider:
        def next_action(self, request):
            return {"action": "provider_error", "arguments": {},
                    "error": error, "usage": usage}

    lab = InstrumentLab()
    agent = DiscoveryWorldAgent(FailedProvider(), tmp_path, arm="hegel", limits=LIMITS)
    result = agent.run(lab)

    assert result["status"] == "provider_failed"
    assert result["costs"]["model_calls"] == len(result["events"]) == 1
    assert result["costs"]["reported_tokens"] == usage.get("total_tokens", 0)
    assert result["costs"]["usage_missing_calls"] == (not usage)
    assert result["events"][0]["error"] == error
    assert not lab.actions
