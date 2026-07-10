from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .codex_app_server import CodexAppServerConfig, CodexAppServerProposalModel
from .events import Event, EventSink, NullEventSink
from .model_client import OpenAICompatibleConfig, OpenAICompatibleProposalModel
from .models import stable_hash
from .secure_env import configured_model


OPENAI_COMPATIBLE = "openai_compatible"
CODEX_APP_SERVER = "codex_app_server"
SUPPORTED_PROVIDERS = (OPENAI_COMPATIBLE, CODEX_APP_SERVER)


class TracedProposalModel(Protocol):
    def complete_with_trace(
        self,
        payload: Mapping[str, Any],
        *,
        trace_id: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ProviderBinding:
    provider_id: str
    model: TracedProposalModel


class ProviderChainProposalModel:
    """Fixed-order failover with a per-run circuit breaker and sanitized provenance."""

    def __init__(
        self,
        providers: Sequence[ProviderBinding],
        *,
        model_name: str,
        event_sink: EventSink | None = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one proposal provider is required")
        ids = [binding.provider_id for binding in providers]
        if len(ids) != len(set(ids)):
            raise ValueError("proposal provider IDs must be unique")
        self.providers = tuple(providers)
        self.model_name = model_name
        self.event_sink = event_sink or NullEventSink()
        self._open_circuits: set[str] = set()
        self.chain_hash = stable_hash({"providers": ids, "model": model_name})

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(binding.provider_id for binding in self.providers)

    def complete(
        self,
        payload: Mapping[str, Any],
        *,
        trace_id: str = "proposal_model",
    ) -> Mapping[str, Any]:
        request_hash = stable_hash(payload)
        failures: list[tuple[str, str]] = []
        active = [binding for binding in self.providers if binding.provider_id not in self._open_circuits]
        if not active:
            raise RuntimeError("all proposal provider circuits are open")
        for index, binding in enumerate(active):
            self.event_sink.emit(
                Event(
                    event="model_provider_attempted",
                    stage="model.provider_chain",
                    trace_id=trace_id,
                    payload={
                        "provider": binding.provider_id,
                        "provider_position": index,
                        "provider_count": len(active),
                        "provider_chain_hash": self.chain_hash,
                        "request_hash": request_hash,
                        "model": self.model_name,
                    },
                )
            )
            try:
                response = binding.model.complete_with_trace(payload, trace_id=trace_id)
            except Exception as exc:
                failures.append((binding.provider_id, type(exc).__name__))
                has_fallback = index + 1 < len(active)
                if has_fallback:
                    self._open_circuits.add(binding.provider_id)
                self.event_sink.emit(
                    Event(
                        event="model_provider_failed",
                        stage="model.provider_chain",
                        trace_id=trace_id,
                        payload={
                            "provider": binding.provider_id,
                            "provider_chain_hash": self.chain_hash,
                            "request_hash": request_hash,
                            "model": self.model_name,
                            "error_type": type(exc).__name__,
                            "fallback_available": has_fallback,
                            "circuit_opened": has_fallback,
                            "raw_error_persisted": False,
                        },
                    )
                )
                continue
            self.event_sink.emit(
                Event(
                    event="model_provider_selected",
                    stage="model.provider_chain",
                    trace_id=trace_id,
                    payload={
                        "provider": binding.provider_id,
                        "provider_chain_hash": self.chain_hash,
                        "request_hash": request_hash,
                        "response_hash": stable_hash(response),
                        "model": self.model_name,
                        "failover_used": bool(failures),
                        "prior_failure_count": len(failures),
                    },
                )
            )
            return response
        self.event_sink.emit(
            Event(
                event="model_provider_chain_exhausted",
                stage="model.provider_chain",
                trace_id=trace_id,
                payload={
                    "provider_chain_hash": self.chain_hash,
                    "request_hash": request_hash,
                    "model": self.model_name,
                    "failure_types": [
                        {"provider": provider, "error_type": error_type}
                        for provider, error_type in failures
                    ],
                    "raw_error_persisted": False,
                },
            )
        )
        summary = ", ".join(f"{provider}:{error_type}" for provider, error_type in failures)
        raise RuntimeError(f"proposal provider chain exhausted ({summary})")

    def complete_with_trace(self, payload: Mapping[str, Any], *, trace_id: str) -> Mapping[str, Any]:
        return self.complete(payload, trace_id=trace_id)


def configured_provider_chain() -> tuple[str, ...]:
    raw = os.environ.get(
        "ASSUMPTION_V2_PROVIDER_CHAIN",
        f"{CODEX_APP_SERVER},{OPENAI_COMPATIBLE}",
    )
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("ASSUMPTION_V2_PROVIDER_CHAIN cannot be empty")
    unsupported = sorted(set(values) - set(SUPPORTED_PROVIDERS))
    if unsupported:
        raise ValueError(f"unsupported proposal providers: {unsupported}")
    if len(values) != len(set(values)):
        raise ValueError("ASSUMPTION_V2_PROVIDER_CHAIN contains duplicates")
    return values


def build_proposal_model(*, event_sink: EventSink | None = None) -> ProviderChainProposalModel:
    sink = event_sink or NullEventSink()
    requested = configured_provider_chain()
    bindings: list[ProviderBinding] = []
    unavailable: list[str] = []
    for provider_id in requested:
        if provider_id == OPENAI_COMPATIBLE:
            if not _openai_compatible_config_present():
                unavailable.append(provider_id)
                continue
            bindings.append(
                ProviderBinding(
                    provider_id=provider_id,
                    model=OpenAICompatibleProposalModel(
                        OpenAICompatibleConfig.from_env(),
                        event_sink=sink,
                    ),
                )
            )
            continue
        if provider_id == CODEX_APP_SERVER:
            try:
                config = CodexAppServerConfig.from_env()
            except (RuntimeError, ValueError):
                unavailable.append(provider_id)
                continue
            bindings.append(
                ProviderBinding(
                    provider_id=provider_id,
                    model=CodexAppServerProposalModel(config, event_sink=sink),
                )
            )
    if not bindings:
        raise RuntimeError(f"no configured proposal provider is available: {unavailable}")
    model_name = configured_model()
    sink.emit(
        Event(
            event="model_provider_chain_built",
            stage="model.provider_chain",
            trace_id="provider-chain-config",
            payload={
                "requested_providers": list(requested),
                "active_providers": [binding.provider_id for binding in bindings],
                "unavailable_providers": unavailable,
                "model": model_name,
                "provider_chain_hash": stable_hash(
                    {
                        "providers": [binding.provider_id for binding in bindings],
                        "model": model_name,
                    }
                ),
                "secret_value_persisted": False,
            },
        )
    )
    return ProviderChainProposalModel(bindings, model_name=model_name, event_sink=sink)


def proposal_provider_status() -> dict[str, Any]:
    try:
        requested = configured_provider_chain()
        chain_valid = True
    except ValueError:
        requested = ()
        chain_valid = False
    codex_path = _codex_path()
    codex_login = _codex_chatgpt_login_present(codex_path) if codex_path else False
    ready = {
        OPENAI_COMPATIBLE: _openai_compatible_config_present(),
        CODEX_APP_SERVER: bool(codex_path and codex_login),
    }
    active_ready = [provider_id for provider_id in requested if ready.get(provider_id, False)]
    return {
        "passed": bool(chain_valid and active_ready),
        "provider_chain_valid": chain_valid,
        "requested_providers": list(requested),
        "ready_providers": active_ready,
        "openai_compatible_config_present": ready[OPENAI_COMPATIBLE],
        "codex_cli_present": bool(codex_path),
        "codex_chatgpt_login_present": codex_login,
        "model": configured_model(enforce_policy=False),
        "secret_value_persisted": False,
    }


def _openai_compatible_config_present() -> bool:
    return bool(
        os.environ.get("ASSUMPTION_V2_API_BASE", "").strip()
        and os.environ.get("ASSUMPTION_V2_API_KEY", "").strip()
    )


def _codex_path() -> str:
    configured = os.environ.get("ASSUMPTION_V2_CODEX_PATH", "").strip()
    if configured:
        return configured if Path(configured).expanduser().is_file() else ""
    return shutil.which("codex") or ""


def _codex_chatgpt_login_present(codex_path: str) -> bool:
    try:
        result = subprocess.run(
            [codex_path, "login", "status"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_login_status_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    status = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode == 0 and "logged in using chatgpt" in status


def _login_status_environment() -> dict[str, str]:
    allowed = {"CODEX_HOME", "HOME", "USERPROFILE", "PATH", "LANG", "LC_ALL"}
    return {key: value for key, value in os.environ.items() if key in allowed and value}
