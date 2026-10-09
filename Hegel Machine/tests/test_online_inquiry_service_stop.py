"""The observed service outage must not consume the remaining registered cohort."""

import json
from pathlib import Path

import pytest

from scripts.run_online_inquiry_llm import protocol, run_study, service_failure_reason


ERRORS = {
    "empty_response": {"kind": "empty_response", "message": "completed response contains no output"},
    "model_unavailable": {"status": 500, "message": json.dumps({
        "error": {"type": "server_error", "code": "model_unavailable", "message": "model unavailable"}})},
    "decode_error": {"decode_error": "expected exactly one research_action function call"},
}


def failure_provider(reason, calls):
    class Provider:
        def __init__(self, workspace):
            self.workspace = Path(workspace)

        def next_action(self, request):
            calls.append(self.workspace)
            return {"action": "provider_error", "arguments": {}, "error": ERRORS[reason],
                    "usage": {"backend": "scripted_interface_test", "total_tokens": None}}
    return Provider


@pytest.mark.parametrize("reason", ["empty_response", "model_unavailable", "decode_error"])
def test_service_failure_reason_separates_bad_actions_from_unavailable_service(reason):
    chain = {"tasks": [{"status": "provider_failed", "events": [
        {"action": "provider_error", "error": ERRORS[reason]}]}]}
    assert service_failure_reason(chain) == (None if reason == "decode_error" else reason)


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("reason", ["empty_response", "model_unavailable"])
def test_outage_stops_pending_chains_and_preserves_in_flight_results(tmp_path, workers, reason):
    calls = []
    plan = protocol(model="scripted-test-only", profile="main", cases=6)
    summary = run_study(plan, tmp_path, provider_factory=failure_provider(reason, calls), workers=workers)

    assert summary["status"] == "interrupted_service_failure"
    assert summary["stop_reason"] == reason
    # Three failed completions trigger the stop. At most workers-1 additional
    # calls can already be in flight; no queued jobs or implicit retries start.
    assert 3 <= len(calls) <= 3 + workers - 1
    assert len(set(calls)) == len(calls)
    assert all(path.name == "task_000" for path in calls)
    assert len(list(tmp_path.glob("*/*/result.json"))) == len(calls)
    attempted = sum(arm["attempted_tasks"] for arm in summary["arms"].values())
    assert attempted == len(calls)
    assert sum(arm["registered_tasks"] for arm in summary["arms"].values()) == 36
    assert sum(arm["not_attempted_tasks"] for arm in summary["arms"].values()) == 36 - attempted
    assert sum(arm["infrastructure_failed_tasks"] for arm in summary["arms"].values()) == attempted
    assert sum(arm["provider_protocol_failed_tasks"] for arm in summary["arms"].values()) == 0
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_nonempty_invalid_tool_output_is_not_service_outage(tmp_path):
    calls = []
    plan = protocol(model="scripted-test-only", profile="main", cases=2)
    summary = run_study(plan, tmp_path,
                        provider_factory=failure_provider("decode_error", calls), workers=1)

    assert summary["status"] == "completed"
    assert len(calls) == 6  # Each arm's failed source stops that chain, not the study.
    assert sum(arm["infrastructure_failed_tasks"] for arm in summary["arms"].values()) == 0
    assert sum(arm["provider_protocol_failed_tasks"] for arm in summary["arms"].values()) == 6
