#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assumption_agent.benchmarks.codex_execution_policy import (
    MODEL_ONLY_ACTION_BUDGET_POLICY,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    _openai_compatible_codex_config_values,
)
from assumption_agent.models import stable_hash


DIAGNOSTIC_VERSION = "codex_model_only_wire_probe_v1"
EXPECTED_CODEX_VERSION = "codex-cli 0.144.1"
MODEL = "gpt-5.4-mini"


class _CaptureServer(ThreadingHTTPServer):
    captures: list[dict[str, Any]]
    errors: list[str]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        payload = json.dumps({"object": "list", "data": []}).encode("utf-8")
        self._write_response("application/json", payload)

    def do_POST(self) -> None:
        server = self.server
        assert isinstance(server, _CaptureServer)
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
            encoded = self.rfile.read(content_length)
            decoded = _decode_body(
                encoded,
                str(self.headers.get("Content-Encoding") or ""),
            )
            payload = json.loads(decoded)
            if not isinstance(payload, Mapping):
                raise TypeError("request body is not an object")
            tools = payload.get("tools")
            if not isinstance(tools, list):
                raise TypeError("request tools are not a list")
            projection = sorted(
                (_tool_projection(row) for row in tools if isinstance(row, Mapping)),
                key=lambda row: json.dumps(row, sort_keys=True),
            )
            server.captures.append(
                {
                    "path": self.path,
                    "model": str(payload.get("model") or ""),
                    "stream": payload.get("stream") is True,
                    "tool_count": len(tools),
                    "tool_projection_count": len(projection),
                    "tool_projection": projection,
                    "tool_projection_hash": stable_hash(projection),
                    "request_body_sha256": hashlib.sha256(decoded).hexdigest(),
                    "hosted_web_search_present": any(
                        _is_web_tool(row) for row in projection
                    ),
                    "image_generation_present": any(
                        _is_image_tool(row) for row in projection
                    ),
                    "external_web_access_present": any(
                        row.get("external_web_access") is True
                        for row in projection
                    ),
                }
            )
        except Exception as exc:
            server.errors.append(type(exc).__name__)
        self._write_response("text/event-stream", _canned_sse())

    def _write_response(self, content_type: str, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


def _decode_body(payload: bytes, encoding: str) -> bytes:
    normalized = encoding.strip().lower()
    if normalized in {"", "identity"}:
        return payload
    if normalized == "gzip":
        return gzip.decompress(payload)
    if normalized == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstandard is required for the Codex wire probe") from exc
        return zstandard.ZstdDecompressor().decompress(payload)
    raise ValueError("unsupported request content encoding")


def _tool_projection(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": str(tool.get("type") or ""),
        "name": str(tool.get("name") or ""),
        "namespace": str(tool.get("namespace") or ""),
        "external_web_access": tool.get("external_web_access") is True,
    }


def _is_web_tool(tool: Mapping[str, Any]) -> bool:
    tool_type = str(tool.get("type") or "").lower()
    name = str(tool.get("name") or "").lower()
    namespace = str(tool.get("namespace") or "").lower()
    return (
        tool_type.startswith("web_search")
        or name in {"web.run", "web_run"}
        or namespace == "web"
    )


def _is_image_tool(tool: Mapping[str, Any]) -> bool:
    return str(tool.get("type") or "").lower().startswith("image_generation")


def _canned_sse() -> bytes:
    events = (
        {
            "type": "response.created",
            "response": {"id": "resp-wire-probe"},
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "id": "msg-wire-probe",
                "content": [
                    {"type": "output_text", "text": "wire probe complete"}
                ],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-wire-probe",
                "usage": {
                    "input_tokens": 0,
                    "input_tokens_details": None,
                    "output_tokens": 0,
                    "output_tokens_details": None,
                    "total_tokens": 0,
                },
            },
        },
    )
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
        for event in events
    ).encode("utf-8")


def _without_canonical_web_disable(values: tuple[str, ...]) -> tuple[str, ...]:
    filtered: list[str] = []
    index = 0
    while index < len(values):
        if (
            values[index] == "--config"
            and index + 1 < len(values)
            and values[index + 1] == 'web_search="disabled"'
        ):
            index += 2
            continue
        filtered.append(values[index])
        index += 1
    filtered.extend(("--config", "tools.web_search=false"))
    return tuple(filtered)


def _trace_projection(stdout: str) -> dict[str, Any]:
    event_types: list[str] = []
    item_types: list[str] = []
    turn_completed = False
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping):
            continue
        event_type = str(row.get("type") or "")
        if event_type:
            event_types.append(event_type)
        if event_type == "turn.completed":
            turn_completed = True
        item = row.get("item")
        if isinstance(item, Mapping):
            item_type = str(item.get("type") or "")
            if item_type:
                item_types.append(item_type)
    return {
        "event_types": sorted(set(event_types)),
        "item_types": sorted(set(item_types)),
        "turn_completed": turn_completed,
        "web_search_event_present": any(
            value.lower().startswith("web_search") for value in item_types
        ),
        "trace_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
    }


def _run_variant(
    *,
    codex: str,
    canonical_disable: bool,
) -> dict[str, Any]:
    server = _CaptureServer(("127.0.0.1", 0), _Handler)
    server.captures = []
    server.errors = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    base_url = f"http://127.0.0.1:{port}/v1"
    config_values = _openai_compatible_codex_config_values(
        policy=MODEL_ONLY_ACTION_BUDGET_POLICY,
        codex_base_url=base_url,
    )
    if not canonical_disable:
        config_values = _without_canonical_web_disable(config_values)
    linux_temp_root = Path("/tmp") if Path("/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="codex-wire-probe-",
        dir=linux_temp_root,
    ) as temporary:
        root = Path(temporary)
        (root / "codex-home").mkdir()
        env = dict(os.environ)
        env.update(
            {
                "CODEX_HOME": str(root / "codex-home"),
                "OPENAI_API_KEY": "local-wire-probe-only",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        command = [
            codex,
            "exec",
            *config_values,
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "--model",
            MODEL,
            "--",
            "Return only: wire probe complete.",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    capture = server.captures[0] if len(server.captures) == 1 else {}
    trace = _trace_projection(completed.stdout)
    return {
        "canonical_web_search_disabled": canonical_disable,
        "codex_return_code": completed.returncode,
        "response_request_count": len(server.captures),
        "capture_errors": list(server.errors),
        "request": capture,
        "trace": trace,
        "command_hash": stable_hash(command),
        "passed": (
            completed.returncode == 0
            and len(server.captures) == 1
            and not server.errors
            and capture.get("path") == "/v1/responses"
            and capture.get("model") == MODEL
            and capture.get("stream") is True
            and int(capture.get("tool_count") or 0) > 0
            and capture.get("tool_projection_count")
            == capture.get("tool_count")
            and trace["turn_completed"] is True
            and trace["web_search_event_present"] is False
            and (
                (
                    capture.get("hosted_web_search_present") is False
                    and capture.get("image_generation_present") is False
                    and capture.get("external_web_access_present") is False
                )
                if canonical_disable
                else capture.get("hosted_web_search_present") is True
            )
        ),
        "raw_request_persisted": False,
        "raw_trace_persisted": False,
        "authorization_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a zero-model Codex Responses wire-tool diagnostic."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--codex", default=shutil.which("codex"))
    args = parser.parse_args()
    if not args.codex:
        raise SystemExit("Codex executable not found")
    version = subprocess.run(
        [str(args.codex), "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout.strip()
    canonical = _run_variant(codex=str(args.codex), canonical_disable=True)
    stale_control = _run_variant(codex=str(args.codex), canonical_disable=False)
    report = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "codex_version": version,
        "expected_codex_version": EXPECTED_CODEX_VERSION,
        "model": MODEL,
        "codex_agent_execution_policy": (
            MODEL_ONLY_ACTION_BUDGET_POLICY.to_dict()
        ),
        "codex_agent_execution_policy_hash": (
            MODEL_ONLY_ACTION_BUDGET_POLICY.policy_hash
        ),
        "canonical_disabled": canonical,
        "stale_boolean_control": stale_control,
        "model_inference_performed": False,
        "scoring_performed": False,
        "raw_content_persisted": False,
        "secret_value_persisted": False,
        "passed": (
            version == EXPECTED_CODEX_VERSION
            and canonical["passed"] is True
            and stale_control["passed"] is True
        ),
    }
    report["report_hash"] = stable_hash(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
