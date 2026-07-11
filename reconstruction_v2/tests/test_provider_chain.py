from __future__ import annotations

from typing import Any, Mapping

from assumption_agent.events import MemoryEventSink
from assumption_agent.provider_chain import (
    ProviderBinding,
    ProviderChainProposalModel,
    configured_provider_chain,
)


class FailingModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_trace(self, payload: Mapping[str, Any], *, trace_id: str):
        self.calls += 1
        raise RuntimeError("provider failed with private diagnostics")


class SuccessfulModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete_with_trace(self, payload: Mapping[str, Any], *, trace_id: str):
        self.calls += 1
        return {"ok": True, "call": self.calls}


def test_provider_chain_fails_over_and_opens_primary_circuit() -> None:
    sink = MemoryEventSink()
    primary = FailingModel()
    fallback = SuccessfulModel()
    chain = ProviderChainProposalModel(
        [
            ProviderBinding("openai_compatible", primary),
            ProviderBinding("secondary_openai_compatible", fallback),
        ],
        model_name="gpt-5.3-codex-spark",
        event_sink=sink,
    )

    assert chain.complete({"request_kind": "health_probe"}, trace_id="first") == {
        "ok": True,
        "call": 1,
    }
    assert chain.complete({"request_kind": "health_probe"}, trace_id="second") == {
        "ok": True,
        "call": 2,
    }

    assert primary.calls == 1
    assert fallback.calls == 2
    failed = next(row for row in sink.events if row["event"] == "model_provider_failed")
    assert failed["payload"]["circuit_opened"] is True
    selected = [row for row in sink.events if row["event"] == "model_provider_selected"]
    assert selected[0]["payload"]["failover_used"] is True
    assert selected[1]["payload"]["failover_used"] is False
    assert "private diagnostics" not in repr(sink.events)


def test_default_provider_chain_uses_direct_openai_compatible_route(monkeypatch) -> None:
    monkeypatch.delenv("ASSUMPTION_V2_PROVIDER_CHAIN", raising=False)

    assert configured_provider_chain() == ("openai_compatible",)
