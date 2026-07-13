from __future__ import annotations

import hashlib
import os
import urllib.error
from typing import Any, Mapping

import pytest

from assumption_agent.events import MemoryEventSink
from assumption_agent.model_client import (
    ACTION_QUALITY_SYSTEM_PROMPT_ADDENDUM,
    PROPOSAL_SYSTEM_PROMPT,
    OpenAICompatibleConfig,
    OpenAICompatibleProposalModel,
)
from assumption_agent.proposer import TRAIN_ACTION_DESIGN_POLICY_VERSION
from assumption_agent.secure_env import (
    configured_api_origin,
    configured_model,
    map_legacy_model_env,
)


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": "```json\n{\"hypotheses\": []}\n```",
                    }
                }
            ]
        }


def test_openai_compatible_proposal_model_uses_json_contract(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ASSUMPTION_API_KEY", "secret-value")
    transport = FakeTransport()
    model = OpenAICompatibleProposalModel(
        OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="gpt-5.3-codex-spark",
            api_key_env="TEST_ASSUMPTION_API_KEY",
            attempts=1,
        ),
        transport=transport,
    )

    response = model.complete({"request_kind": "propose_hypothesis_programs"})

    assert response == {"hypotheses": []}
    assert transport.calls[0]["url"] == "https://provider.example/v1/chat/completions"
    assert transport.calls[0]["payload"]["model"] == "gpt-5.3-codex-spark"
    assert transport.calls[0]["payload"]["temperature"] == 0
    assert transport.calls[0]["payload"]["response_format"] == {"type": "json_object"}
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer secret-value"
    system_prompt = transport.calls[0]["payload"]["messages"][0]["content"]
    assert "complete imperative, task-local sentence" in system_prompt
    assert "TRAIN residual context.task_instruction" in system_prompt
    assert "enum-only value" in system_prompt
    assert "mapping, mode, or check label" in system_prompt
    assert "repair_request_scope_policy" in system_prompt
    assert "repair_response_contract" in system_prompt
    assert "singular top-level field hypothesis" in system_prompt
    assert (
        "never claim preserve_baseline inside an activated action node"
        in system_prompt
    )
    assert "top-level fallback field" in system_prompt
    assert system_prompt == PROPOSAL_SYSTEM_PROMPT
    assert ACTION_QUALITY_SYSTEM_PROMPT_ADDENDUM not in system_prompt
    assert hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() == (
        "a726e60d76516379ad021cd2ad7fedd465593ac203b0774e038d7f0772e6c66e"
    )


def test_action_quality_addendum_is_request_local_to_supported_contract(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ASSUMPTION_API_KEY", "secret-value")
    transport = FakeTransport()
    model = OpenAICompatibleProposalModel(
        OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="gpt-5.4-mini",
            api_key_env="TEST_ASSUMPTION_API_KEY",
            attempts=1,
        ),
        transport=transport,
    )

    model.complete(
        {
            "request_kind": "propose_hypothesis_programs",
            "capabilities": {
                "action_quality_contract": {
                    "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
                }
            },
        }
    )

    system_prompt = transport.calls[0]["payload"]["messages"][0]["content"]
    assert system_prompt == (
        f"{PROPOSAL_SYSTEM_PROMPT} {ACTION_QUALITY_SYSTEM_PROMPT_ADDENDUM}"
    )
    assert "task_instruction as the baseline requirement" in system_prompt
    assert "action-quality audit is diagnostic only" in system_prompt


@pytest.mark.parametrize(
    "payload",
    [
        {
            "request_kind": "propose_hypothesis_programs",
            "action_quality_contract": {
                "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
            },
        },
        {
            "request_kind": "propose_hypothesis_programs",
            "capabilities": {
                "action_quality_contract": {"policy": "unsupported_policy"},
            },
        },
    ],
)
def test_action_quality_addendum_does_not_leak_to_other_requests(
    monkeypatch,
    payload,
) -> None:
    monkeypatch.setenv("TEST_ASSUMPTION_API_KEY", "secret-value")
    transport = FakeTransport()
    model = OpenAICompatibleProposalModel(
        OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="gpt-5.4-mini",
            api_key_env="TEST_ASSUMPTION_API_KEY",
            attempts=1,
        ),
        transport=transport,
    )

    model.complete(payload)

    system_prompt = transport.calls[0]["payload"]["messages"][0]["content"]
    assert system_prompt == PROPOSAL_SYSTEM_PROMPT
    assert ACTION_QUALITY_SYSTEM_PROMPT_ADDENDUM not in system_prompt


def test_model_attempt_events_are_sanitized(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ASSUMPTION_API_KEY", "secret-value")
    sink = MemoryEventSink()
    model = OpenAICompatibleProposalModel(
        OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="gpt-5.3-codex-spark",
            api_key_env="TEST_ASSUMPTION_API_KEY",
            attempts=1,
        ),
        transport=FakeTransport(),
        event_sink=sink,
    )

    model.complete({"request_kind": "propose_hypothesis_programs"}, trace_id="attempt-audit")

    assert [row["event"] for row in sink.events] == [
        "model_attempt_started",
        "model_attempt_succeeded",
    ]
    assert "secret-value" not in repr(sink.events)


class UnauthorizedTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def post(self, **kwargs):
        self.call_count += 1
        raise urllib.error.HTTPError(kwargs["url"], 401, "Unauthorized", {}, None)


def test_nonretryable_auth_error_stops_after_one_attempt(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ASSUMPTION_API_KEY", "secret-value")
    transport = UnauthorizedTransport()
    sink = MemoryEventSink()
    model = OpenAICompatibleProposalModel(
        OpenAICompatibleConfig(
            base_url="https://provider.example/v1",
            model="gpt-5.3-codex-spark",
            api_key_env="TEST_ASSUMPTION_API_KEY",
            attempts=3,
        ),
        transport=transport,
        event_sink=sink,
    )

    with pytest.raises(RuntimeError, match=r"failed after 1 attempt\(s\)"):
        model.complete({"request_kind": "propose_hypothesis_programs"})

    assert transport.call_count == 1
    failed = next(row for row in sink.events if row["event"] == "model_attempt_failed")
    assert failed["payload"]["http_status"] == 401
    assert failed["payload"]["retryable"] is False


def test_legacy_env_mapping_prefers_ruoli_aliases(monkeypatch) -> None:
    monkeypatch.delenv("ASSUMPTION_V2_API_BASE", raising=False)
    monkeypatch.delenv("ASSUMPTION_V2_API_KEY", raising=False)
    monkeypatch.setenv("GPT5_BASE_URL", "https://old.example")
    monkeypatch.setenv("GPT5_API_KEY", "old-key")
    monkeypatch.setenv("RUOLI_BASE_URL", "https://ruoli.example")
    monkeypatch.setenv("RUOLI_GPT_KEY", "ruoli-key")

    presence = map_legacy_model_env()

    assert presence["api_key_present"] is True
    assert presence["base_url_present"] is True
    assert presence["secret_value_persisted"] is False
    assert presence["model"] == "gpt-5.4-mini"
    assert os.environ["ASSUMPTION_V2_API_BASE"] == "https://ruoli.example"
    assert os.environ["ASSUMPTION_V2_API_KEY"] == "ruoli-key"


def test_ruoli_fallback_model_is_protocol_approved(monkeypatch) -> None:
    monkeypatch.setenv("ASSUMPTION_V2_MODEL", "gpt-5.4-mini")
    monkeypatch.delenv("ASSUMPTION_V2_ALLOW_ALTERNATE_MODEL", raising=False)

    assert configured_model() == "gpt-5.4-mini"


def test_configured_api_origin_discards_route_path(monkeypatch) -> None:
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev/v1")

    assert configured_api_origin() == "https://ruoli.dev"
