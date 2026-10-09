"""Direct structured Responses provider for the ordinary code agent."""

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from time import monotonic

from .code_agent import _ACTION_SCHEMA, _INSTRUCTIONS
from .code_provider_context import prepare_code_context


class ResponsesCodeProvider:
    """Use an explicitly selected Responses model for code actions."""

    def __init__(self, workspace, *, base_url, model, env_key="RUOLI_API_KEY", timeout_seconds=180,
                 reasoning_effort=None, max_output_tokens=8192):
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/responses"):
            self.base_url += "/responses"
        self.model = model
        self.env_key = env_key
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort, self.max_output_tokens = reasoning_effort, max_output_tokens
        self.calls = max([0, *(int(path.stem.rsplit("_", 1)[1])
                               for path in self.workspace.glob("api_response_[0-9][0-9][0-9].json"))])

    def next_action(self, request):
        self.calls += 1
        context = prepare_code_context(self.workspace, request, schema=_ACTION_SCHEMA,
                                       instructions=_INSTRUCTIONS)
        instructions = (_INSTRUCTIONS + "\nCall research_action exactly once for your next action. "
                        "The host executes it and supplies real results in the next request. "
                        "Wait for that feedback before choosing another action.")
        body = {"model": self.model, "instructions": instructions, "stream": True,
                "input": json.dumps(context), "max_output_tokens": self.max_output_tokens,
                "tools": [{"type": "function", "name": "research_action", "strict": True,
                           "description": "Submit one laboratory or Python action for the host to execute.",
                           "parameters": _ACTION_SCHEMA}],
                "tool_choice": {"type": "function", "name": "research_action"},
                "parallel_tool_calls": False}
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        key = os.environ.get(self.env_key)
        usage = {"backend": "responses", "runtime": self.base_url,
                 "calls": 1, "input_tokens": None, "output_tokens": None,
                 "total_tokens": None, "model": self.model}
        started = monotonic()
        request_data = urllib.request.Request(
            self.base_url, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}" if key else "",
                     "Content-Type": "application/json"}, method="POST")
        try:
            payload = None
            with urllib.request.urlopen(request_data, timeout=self.timeout_seconds) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                        payload = event["response"]
                        break
                    if event.get("type") == "error":
                        raise ValueError(event.get("message", "response stream failed"))
            if payload is None:
                raise ValueError("response stream ended before completion")
            (self.workspace / f"api_response_{self.calls:03d}.json").write_text(
                json.dumps(payload), encoding="utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[-4000:]
            usage["wall_time_seconds"] = monotonic() - started
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"status": error.code, "message": detail}}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            usage["wall_time_seconds"] = monotonic() - started
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"message": f"{type(error).__name__}: {error}"}}
        usage["wall_time_seconds"] = monotonic() - started
        usage["returned_model"] = payload.get("model")
        reported = payload.get("usage") or {}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(reported.get(name), int):
                usage[name] = reported[name]
        output = payload.get("output", [])
        if not output or all(item.get("type") == "message" and not any(
                part.get("text", "").strip() for part in item.get("content", []))
                for item in output):
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"kind": "empty_response",
                              "message": "Provider returned no tool action or nonempty text",
                              "response_status": payload.get("status")}}
        try:
            calls = [item for item in payload["output"] if item.get("type") == "function_call"]
            if len(calls) != 1 or calls[0].get("name") != "research_action":
                raise ValueError("expected exactly one research_action function call")
            action_payload = json.loads(calls[0]["arguments"])
            arguments = json.loads(action_payload["arguments_json"], strict=False)
            return {"action": action_payload["action"], "arguments": arguments, "usage": usage}
        except (ValueError, KeyError, TypeError, IndexError) as error:
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"decode_error": f"{type(error).__name__}: {error}"}}
