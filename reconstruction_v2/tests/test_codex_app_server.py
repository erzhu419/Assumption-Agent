from __future__ import annotations

from typing import Any, Mapping

import pytest

from assumption_agent.codex_app_server import (
    CodexAppServerConfig,
    CodexAppServerProposalModel,
    CodexAppServerToolUseError,
    _TurnAccumulator,
    _output_schema_for_payload,
    _safe_codex_environment,
)
from assumption_agent.events import MemoryEventSink


class FakeRunner:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        config: CodexAppServerConfig,
        payload: Mapping[str, Any],
        developer_instructions: str,
    ) -> str:
        self.calls.append(
            {
                "config": config,
                "payload": dict(payload),
                "developer_instructions": developer_instructions,
            }
        )
        return self.response


def test_codex_model_parses_json_and_logs_sanitized_transport() -> None:
    sink = MemoryEventSink()
    runner = FakeRunner('{"ok": true}')
    model = CodexAppServerProposalModel(
        CodexAppServerConfig(codex_path="/test/codex", attempts=1),
        runner=runner,
        event_sink=sink,
    )

    result = model.complete({"request_kind": "health_probe"}, trace_id="codex-test")

    assert result == {"ok": True}
    assert runner.calls[0]["config"].model == "gpt-5.3-codex-spark"
    assert "Never use tools" in runner.calls[0]["developer_instructions"]
    succeeded = next(row for row in sink.events if row["event"] == "model_attempt_succeeded")
    assert succeeded["payload"]["provider"] == "codex_app_server"
    assert succeeded["payload"]["tool_use_observed"] is False


def test_proposal_output_schemas_are_strict_objects() -> None:
    for request_kind in (
        "health_probe",
        "propose_hypothesis_programs",
        "repair_hypothesis_program",
    ):
        schema = _output_schema_for_payload({"request_kind": request_kind})
        assert schema is not None
        _assert_strict_schema_objects(schema)


def test_runtime_tool_item_is_rejected() -> None:
    accumulator = _TurnAccumulator(turn_id="turn-1")

    with pytest.raises(CodexAppServerToolUseError):
        accumulator.consume(
            {
                "method": "item/started",
                "params": {
                    "turnId": "turn-1",
                    "item": {"type": "commandExecution", "id": "item-1"},
                },
            }
        )


def test_codex_child_environment_excludes_provider_secrets(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_SQLITE_HOME", "/safe/sqlite")
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", "must-not-propagate")
    monkeypatch.setenv("RUOLI_GPT_KEY", "must-not-propagate-either")

    child_env = _safe_codex_environment()

    assert child_env["CODEX_SQLITE_HOME"] == "/safe/sqlite"
    assert "ASSUMPTION_V2_API_KEY" not in child_env
    assert "RUOLI_GPT_KEY" not in child_env
    assert "must-not-propagate" not in repr(child_env)


def _assert_strict_schema_objects(value: Any) -> None:
    if isinstance(value, Mapping):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
            properties = value.get("properties")
            assert isinstance(properties, Mapping)
            assert set(value.get("required", [])) == set(properties)
        for child in value.values():
            _assert_strict_schema_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_strict_schema_objects(child)
