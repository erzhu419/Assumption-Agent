from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .events import Event, EventSink, NullEventSink
from .models import stable_hash
from .secure_env import LOCKED_MODEL, configured_model


PROPOSAL_SYSTEM_PROMPT = (
    "You are the proposal component of a controlled self-evolution system. "
    "Return one JSON object matching the supplied contract. Do not include markdown. "
    "Never infer or request hidden test answers. Never use tools, files, shell commands, "
    "network search, or external context. Every hypothesis needs structured triggers, "
    "an action graph whose semantics match the supplied backend capabilities, an "
    "external-anchor verifier used only by the evaluator unless explicitly declared "
    "agent-local, and the supplied prospective fallback contract. When a "
    "proposal_batch_contract is present, return exactly its requested hypothesis "
    "count and keep the JSON output compact. Treat distinct activation signatures "
    "on the specified training failure rows as a search preference, not a hard "
    "response requirement: candidates may share a signature when they diversify "
    "the action graph or backend treatment. When action_semantics contains "
    "prompt_directive, every activated action value must be a complete imperative, "
    "task-local sentence grounded in the provided TRAIN residual "
    "context.task_instruction. Never use an enum-only value or a mapping, mode, or "
    "check label as the action value, and never claim preserve_baseline inside an "
    "activated action node; preserve_baseline remains only in the unchanged "
    "top-level fallback field."
)


class JsonTransport(Protocol):
    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
        if not raw:
            raise RuntimeError("model endpoint returned no bytes")
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("model endpoint did not return a JSON object")
        return decoded


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str = LOCKED_MODEL
    api_key_env: str = "ASSUMPTION_V2_API_KEY"
    timeout_seconds: float = 300.0
    attempts: int = 2
    max_tokens: int = 4000

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        base_url = os.environ.get("ASSUMPTION_V2_API_BASE", "").strip()
        if not base_url:
            raise RuntimeError("ASSUMPTION_V2_API_BASE is required")
        model = configured_model()
        return cls(
            base_url=base_url,
            model=model,
        )


class OpenAICompatibleProposalModel:
    """Secure JSON-only adapter for an OpenAI-compatible chat-completions route."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: JsonTransport | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        self.event_sink = event_sink or NullEventSink()

    def complete(self, payload: Mapping[str, Any], *, trace_id: str = "proposal_model") -> Mapping[str, Any]:
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"{self.config.api_key_env} is required")
        body = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": PROPOSAL_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        attempts_used = 0
        request_hash = stable_hash(body)
        for attempt in range(max(1, self.config.attempts)):
            attempts_used = attempt + 1
            started = time.monotonic()
            self.event_sink.emit(
                Event(
                    event="model_attempt_started",
                    stage="model.transport",
                    trace_id=trace_id,
                    payload={
                        "request_hash": request_hash,
                        "attempt": attempt + 1,
                        "attempt_limit": max(1, self.config.attempts),
                        "model": self.config.model,
                        "timeout_seconds": self.config.timeout_seconds,
                        "endpoint_hash": stable_hash({"url": _chat_completions_url(self.config.base_url)}),
                    },
                )
            )
            try:
                response = self.transport.post(
                    url=_chat_completions_url(self.config.base_url),
                    headers=headers,
                    payload=body,
                    timeout_seconds=self.config.timeout_seconds,
                )
                content = _message_content(response)
                parsed = parse_json_object_content(content)
                self.event_sink.emit(
                    Event(
                        event="model_attempt_succeeded",
                        stage="model.transport",
                        trace_id=trace_id,
                        payload={
                            "request_hash": request_hash,
                            "response_hash": stable_hash(parsed),
                            "attempt": attempt + 1,
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                            "model": self.config.model,
                        },
                    )
                )
                return parsed
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                last_error = exc
                retryable = _is_retryable(exc)
                self.event_sink.emit(
                    Event(
                        event="model_attempt_failed",
                        stage="model.transport",
                        trace_id=trace_id,
                        payload={
                            "request_hash": request_hash,
                            "attempt": attempt + 1,
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                            "error_type": type(exc).__name__,
                            "http_status": getattr(exc, "code", None),
                            "retryable": retryable,
                            "model": self.config.model,
                        },
                    )
                )
                if retryable and attempt + 1 < max(1, self.config.attempts):
                    time.sleep(min(4.0, 0.5 * (2**attempt)))
                    continue
                break
        raise RuntimeError(
            f"proposal model failed after {attempts_used} attempt(s): {type(last_error).__name__}"
        ) from last_error

    def complete_with_trace(self, payload: Mapping[str, Any], *, trace_id: str) -> Mapping[str, Any]:
        return self.complete(payload, trace_id=trace_id)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _message_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("chat completion choice is malformed")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat completion response has no message")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping)
        )
        if text.strip():
            return text
    raise ValueError("chat completion response has no textual content")


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_json_object_content(content: str) -> Mapping[str, Any]:
    parsed = json.loads(_strip_json_fence(content))
    if not isinstance(parsed, Mapping):
        raise ValueError("proposal response content must decode to a JSON object")
    return parsed


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 408 or error.code == 429 or 500 <= error.code < 600
    return True
