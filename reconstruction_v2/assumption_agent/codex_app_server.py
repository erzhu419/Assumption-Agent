from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .events import Event, EventSink, NullEventSink
from .model_client import PROPOSAL_SYSTEM_PROMPT, parse_json_object_content
from .models import stable_hash
from .secure_env import LOCKED_MODEL, configured_model


class CodexAppServerToolUseError(RuntimeError):
    """Raised when the isolated proposal turn attempts any runtime tool."""


@dataclass(frozen=True)
class CodexAppServerConfig:
    codex_path: str
    model: str = LOCKED_MODEL
    request_idle_timeout_seconds: float = 60.0
    turn_idle_timeout_seconds: float = 300.0
    attempts: int = 1
    reasoning_effort: str = "low"

    @classmethod
    def from_env(cls) -> "CodexAppServerConfig":
        configured_path = os.environ.get("ASSUMPTION_V2_CODEX_PATH", "").strip()
        codex_path = configured_path or shutil.which("codex") or ""
        if not codex_path:
            raise RuntimeError("codex CLI is required for the codex_app_server provider")
        effort = os.environ.get("ASSUMPTION_V2_CODEX_REASONING_EFFORT", "low").strip().lower()
        if effort not in {"minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported Codex reasoning effort")
        return cls(
            codex_path=codex_path,
            model=configured_model(),
            request_idle_timeout_seconds=_positive_float_env(
                "ASSUMPTION_V2_CODEX_REQUEST_IDLE_SECONDS", 60.0
            ),
            turn_idle_timeout_seconds=_positive_float_env(
                "ASSUMPTION_V2_CODEX_TURN_IDLE_SECONDS", 300.0
            ),
            attempts=_positive_int_env("ASSUMPTION_V2_CODEX_ATTEMPTS", 1),
            reasoning_effort=effort,
        )


class CodexTurnRunner(Protocol):
    def run(
        self,
        *,
        config: CodexAppServerConfig,
        payload: Mapping[str, Any],
        developer_instructions: str,
    ) -> str: ...


class CodexAppServerProposalModel:
    """JSON-only proposal adapter backed by the user's local Codex subscription."""

    provider_id = "codex_app_server"

    def __init__(
        self,
        config: CodexAppServerConfig,
        *,
        runner: CodexTurnRunner | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessCodexTurnRunner()
        self.event_sink = event_sink or NullEventSink()

    def complete(
        self,
        payload: Mapping[str, Any],
        *,
        trace_id: str = "proposal_model",
    ) -> Mapping[str, Any]:
        request_hash = stable_hash(
            {
                "provider": self.provider_id,
                "model": self.config.model,
                "developer_instructions": PROPOSAL_SYSTEM_PROMPT,
                "payload": payload,
            }
        )
        last_error: Exception | None = None
        attempts_used = 0
        for attempt in range(max(1, self.config.attempts)):
            attempts_used = attempt + 1
            started = time.monotonic()
            self.event_sink.emit(
                Event(
                    event="model_attempt_started",
                    stage="model.transport",
                    trace_id=trace_id,
                    payload={
                        "provider": self.provider_id,
                        "request_hash": request_hash,
                        "attempt": attempt + 1,
                        "attempt_limit": max(1, self.config.attempts),
                        "model": self.config.model,
                        "request_idle_timeout_seconds": self.config.request_idle_timeout_seconds,
                        "turn_idle_timeout_seconds": self.config.turn_idle_timeout_seconds,
                        "transport_hash": stable_hash(
                            {
                                "transport": self.provider_id,
                                "binary_name": Path(self.config.codex_path).name,
                            }
                        ),
                        "isolated_workdir": True,
                        "tool_use_allowed": False,
                    },
                )
            )
            try:
                content = self.runner.run(
                    config=self.config,
                    payload=payload,
                    developer_instructions=PROPOSAL_SYSTEM_PROMPT,
                )
                parsed = parse_json_object_content(content)
                self.event_sink.emit(
                    Event(
                        event="model_attempt_succeeded",
                        stage="model.transport",
                        trace_id=trace_id,
                        payload={
                            "provider": self.provider_id,
                            "request_hash": request_hash,
                            "response_hash": stable_hash(parsed),
                            "attempt": attempt + 1,
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                            "model": self.config.model,
                            "tool_use_observed": False,
                            "raw_content_persisted": False,
                        },
                    )
                )
                return parsed
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                self.event_sink.emit(
                    Event(
                        event="model_attempt_failed",
                        stage="model.transport",
                        trace_id=trace_id,
                        payload={
                            "provider": self.provider_id,
                            "request_hash": request_hash,
                            "attempt": attempt + 1,
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                            "error_type": type(exc).__name__,
                            "tool_use_rejected": isinstance(exc, CodexAppServerToolUseError),
                            "model": self.config.model,
                            "raw_error_persisted": False,
                        },
                    )
                )
                if attempt + 1 < max(1, self.config.attempts):
                    time.sleep(min(2.0, 0.25 * (2**attempt)))
        raise RuntimeError(
            f"codex app-server proposal failed after {attempts_used} attempt(s): "
            f"{type(last_error).__name__}"
        ) from last_error

    def complete_with_trace(self, payload: Mapping[str, Any], *, trace_id: str) -> Mapping[str, Any]:
        return self.complete(payload, trace_id=trace_id)


class SubprocessCodexTurnRunner:
    """One ephemeral app-server process per call, with byte-activity idle watchdogs."""

    def run(
        self,
        *,
        config: CodexAppServerConfig,
        payload: Mapping[str, Any],
        developer_instructions: str,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="assumption-v2-codex-") as isolated_dir:
            process = _JsonLineAppServer.start(config, isolated_dir)
            try:
                notifications: list[Mapping[str, Any]] = []
                process.send(
                    {
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "clientInfo": {
                                "name": "assumption-agent-v2",
                                "title": "Assumption Agent v2",
                                "version": "0.1.0",
                            },
                            "capabilities": {"experimentalApi": True},
                        },
                    }
                )
                process.wait_response(
                    1,
                    idle_timeout_seconds=config.request_idle_timeout_seconds,
                    notifications=notifications,
                )
                process.send({"method": "initialized", "params": None})
                process.send(
                    {
                        "id": 2,
                        "method": "thread/start",
                        "params": {
                            "model": config.model,
                            "cwd": isolated_dir,
                            "ephemeral": True,
                            "approvalPolicy": "never",
                            "sandbox": "read-only",
                            "developerInstructions": developer_instructions,
                            "dynamicTools": [],
                            "environments": [],
                            "runtimeWorkspaceRoots": [isolated_dir],
                            "selectedCapabilityRoots": [],
                        },
                    }
                )
                thread_result = process.wait_response(
                    2,
                    idle_timeout_seconds=config.request_idle_timeout_seconds,
                    notifications=notifications,
                )
                thread_id = _nested_id(thread_result, "thread")
                if not thread_id:
                    raise RuntimeError("codex app-server did not return a thread ID")
                turn_params: dict[str, Any] = {
                    "threadId": thread_id,
                    "input": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=True, sort_keys=True),
                        }
                    ],
                    "model": config.model,
                    "effort": config.reasoning_effort,
                    "summary": "auto",
                    "environments": [],
                }
                output_schema = _output_schema_for_payload(payload)
                if output_schema is not None:
                    turn_params["outputSchema"] = output_schema
                process.send({"id": 3, "method": "turn/start", "params": turn_params})
                turn_result = process.wait_response(
                    3,
                    idle_timeout_seconds=config.request_idle_timeout_seconds,
                    notifications=notifications,
                )
                turn_id = _nested_id(turn_result, "turn")
                if not turn_id:
                    raise RuntimeError("codex app-server did not return a turn ID")
                accumulator = _TurnAccumulator(turn_id=turn_id)
                for message in notifications:
                    accumulator.consume(message)
                while not accumulator.completed:
                    message = process.next_message(config.turn_idle_timeout_seconds)
                    if _is_server_request(message):
                        process.reject_server_request(message)
                        raise CodexAppServerToolUseError("codex app-server requested a runtime action")
                    try:
                        accumulator.consume(message)
                    except CodexAppServerToolUseError:
                        process.send(
                            {
                                "id": 4,
                                "method": "turn/interrupt",
                                "params": {"threadId": thread_id, "turnId": turn_id},
                            }
                        )
                        raise
                if accumulator.status != "completed":
                    raise RuntimeError("codex app-server turn did not complete successfully")
                content = accumulator.final_text.strip()
                if not content:
                    raise RuntimeError("codex app-server returned no assistant text")
                return content
            finally:
                process.close()


class _JsonLineAppServer:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.selector = selectors.DefaultSelector()
        if process.stdout is None or process.stderr is None or process.stdin is None:
            raise RuntimeError("codex app-server pipes were not created")
        self.stdout = process.stdout
        self.stderr = process.stderr
        self.stdin = process.stdin
        self.selector.register(self.stdout, selectors.EVENT_READ, "stdout")
        self.selector.register(self.stderr, selectors.EVENT_READ, "stderr")
        self.stdout_buffer = b""
        self.messages: list[Mapping[str, Any]] = []
        self.stderr_byte_count = 0

    @classmethod
    def start(cls, config: CodexAppServerConfig, isolated_dir: str) -> "_JsonLineAppServer":
        process = subprocess.Popen(
            [
                config.codex_path,
                "app-server",
                "--stdio",
                "-c",
                "mcp_servers={}",
            ],
            cwd=isolated_dir,
            env=_safe_codex_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        return cls(process)

    def send(self, message: Mapping[str, Any]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError("codex app-server process is not running")
        encoded = json.dumps(message, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.stdin.write(encoded + b"\n")
        self.stdin.flush()

    def wait_response(
        self,
        request_id: int,
        *,
        idle_timeout_seconds: float,
        notifications: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        while True:
            message = self.next_message(idle_timeout_seconds)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError("codex app-server request returned a protocol error")
                result = message.get("result")
                return result if isinstance(result, Mapping) else {}
            if _is_server_request(message):
                self.reject_server_request(message)
                raise CodexAppServerToolUseError("codex app-server requested a runtime action")
            notifications.append(message)

    def reject_server_request(self, message: Mapping[str, Any]) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, int):
            return
        self.send(
            {
                "id": request_id,
                "error": {"code": -32601, "message": "runtime actions disabled for proposal turns"},
            }
        )

    def next_message(self, idle_timeout_seconds: float) -> Mapping[str, Any]:
        if self.messages:
            return self.messages.pop(0)
        deadline = time.monotonic() + idle_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("codex app-server idle watchdog expired")
            ready = self.selector.select(timeout=min(1.0, remaining))
            if not ready:
                if self.process.poll() is not None:
                    raise RuntimeError("codex app-server process closed unexpectedly")
                continue
            for key, _ in ready:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    try:
                        self.selector.unregister(key.fileobj)
                    except (KeyError, ValueError):
                        pass
                    if key.data == "stdout":
                        raise RuntimeError("codex app-server stdout closed unexpectedly")
                    continue
                deadline = time.monotonic() + idle_timeout_seconds
                if key.data == "stderr":
                    self.stderr_byte_count += len(chunk)
                    continue
                self.stdout_buffer += chunk
                lines = self.stdout_buffer.split(b"\n")
                self.stdout_buffer = lines.pop()
                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        decoded = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(decoded, Mapping):
                        self.messages.append(decoded)
            if self.messages:
                return self.messages.pop(0)

    def close(self) -> None:
        try:
            self.selector.close()
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)


@dataclass
class _TurnAccumulator:
    turn_id: str
    completed: bool = False
    status: str = ""
    delta_text: str = ""
    completed_text: str = ""

    @property
    def final_text(self) -> str:
        return self.completed_text or self.delta_text

    def consume(self, message: Mapping[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params")
        if not isinstance(method, str) or not isinstance(params, Mapping):
            return
        event_turn_id = _turn_id_from_params(params)
        if event_turn_id and event_turn_id != self.turn_id:
            return
        if method == "item/started":
            item = params.get("item")
            item_type = _normalized_item_type(item)
            if item_type and item_type not in {
                "agentmessage",
                "assistantmessage",
                "message",
                "reasoning",
                "usermessage",
            }:
                raise CodexAppServerToolUseError("proposal turn attempted a runtime tool")
            return
        if method == "item/agentMessage/delta":
            self.delta_text += _normalize_text(params.get("delta") or params.get("text"))
            return
        if method == "item/completed":
            item = params.get("item")
            item_type = _normalized_item_type(item)
            if item_type in {"agentmessage", "assistantmessage", "message"}:
                text = _normalize_text(item)
                if text:
                    self.completed_text = text
            return
        if method == "turn/completed":
            turn = params.get("turn")
            if isinstance(turn, Mapping):
                self.status = str(turn.get("status") or "")
            else:
                self.status = str(params.get("status") or "")
            self.completed = True


def _nested_id(result: Mapping[str, Any], key: str) -> str:
    direct = result.get("id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    nested = result.get(key)
    if isinstance(nested, Mapping):
        value = nested.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _turn_id_from_params(params: Mapping[str, Any]) -> str:
    direct = params.get("turnId")
    if isinstance(direct, str):
        return direct
    turn = params.get("turn")
    if isinstance(turn, Mapping) and isinstance(turn.get("id"), str):
        return str(turn["id"])
    return ""


def _normalized_item_type(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    value = item.get("type")
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.lower() if character.isalnum())


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_normalize_text(row) for row in value)
    if not isinstance(value, Mapping):
        return ""
    for key in ("text", "content", "message", "outputText", "output_text"):
        text = _normalize_text(value.get(key))
        if text:
            return text
    return ""


def _is_server_request(message: Mapping[str, Any]) -> bool:
    return isinstance(message.get("id"), int) and isinstance(message.get("method"), str)


def _safe_codex_environment() -> dict[str, str]:
    allowed = {
        "CODEX_HOME",
        "CODEX_SQLITE_HOME",
        "HOME",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "LOGNAME",
        "PATH",
        "LANG",
        "LC_ALL",
        "APPDATA",
        "LOCALAPPDATA",
        "TMP",
        "TEMP",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
    return {key: value for key, value in os.environ.items() if key in allowed and value}


def _output_schema_for_payload(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    request_kind = payload.get("request_kind")
    if request_kind == "health_probe":
        return _strict_object({"ok": {"type": "boolean"}}, required=("ok",))
    if request_kind == "propose_hypothesis_programs":
        return _strict_object(
            {
                "hypotheses": {
                    "type": "array",
                    "items": _hypothesis_program_output_schema(),
                }
            },
            required=("hypotheses",),
        )
    if request_kind == "repair_hypothesis_program":
        return _strict_object(
            {"hypothesis": _hypothesis_program_output_schema()},
            required=("hypothesis",),
        )
    return None


def _hypothesis_program_output_schema() -> Mapping[str, Any]:
    scalar = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
        ]
    }
    predicate = _strict_object(
        {
            "key": {"type": "string"},
            "op": {
                "type": "string",
                "enum": ["eq", "ne", "in", "contains", "exists", "gte", "lte"],
            },
            "value": scalar,
        },
        required=("key", "op", "value"),
    )
    predicate_group = _strict_object(
        {
            "all_of": {"type": "array", "items": predicate},
            "any_of": {"type": "array", "items": predicate},
            "none_of": {"type": "array", "items": predicate},
        },
        required=("all_of", "any_of", "none_of"),
    )
    action = _strict_object(
        {
            "id": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": [
                    "enable_lane",
                    "disable_lane",
                    "prioritize_lane",
                    "set_parameter",
                    "require_verifier",
                    "abstain",
                    "execute_step",
                    "check_condition",
                    "produce_artifact",
                    "request_evidence",
                ],
            },
            "target": {"type": "string"},
            "value": scalar,
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        required=("id", "operation", "target", "value", "depends_on"),
    )
    expected_effect = _strict_object(
        {
            "metric": {"type": "string"},
            "minimum_delta": {"type": "number"},
            "maximum_harm_rate": {"type": "number"},
            "maximum_cost_ratio": {"type": "number"},
        },
        required=(
            "metric",
            "minimum_delta",
            "maximum_harm_rate",
            "maximum_cost_ratio",
        ),
    )
    verifier = _strict_object(
        {
            "checks": {"type": "array", "items": {"type": "string"}},
            "required_evidence": {"type": "array", "items": {"type": "string"}},
            "anchor_id": {"type": "string"},
            "repair_on_failure": {"type": "boolean"},
            "max_repair_depth": {"type": "integer", "minimum": 0, "maximum": 4},
        },
        required=(
            "checks",
            "required_evidence",
            "anchor_id",
            "repair_on_failure",
            "max_repair_depth",
        ),
    )
    return _strict_object(
        {
            "id": {"type": "string"},
            "kind": {"type": "string", "enum": ["task", "policy", "evaluator"]},
            "statement": {"type": "string"},
            "trigger": predicate_group,
            "anti_trigger": predicate_group,
            "action_graph": {"type": "array", "items": action, "minItems": 1},
            "expected_effect": expected_effect,
            "verifier": verifier,
            "fallback": {"type": "string", "enum": ["preserve_baseline"]},
            "status": {"type": "string", "enum": ["candidate"]},
        },
        required=(
            "id",
            "kind",
            "statement",
            "trigger",
            "anti_trigger",
            "action_graph",
            "expected_effect",
            "verifier",
            "fallback",
            "status",
        ),
    )


def _strict_object(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...],
) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _positive_float_env(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    value = float(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _positive_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    value = int(raw) if raw else default
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value
