from __future__ import annotations

import os
import urllib.error
from typing import Any, Mapping

import pytest

from assumption_agent.events import MemoryEventSink
from assumption_agent.model_client import OpenAICompatibleConfig, OpenAICompatibleProposalModel
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
