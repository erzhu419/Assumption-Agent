"""Direct native Chat Completions tools for the ordinary code agent."""

import json
import os
from pathlib import Path
from time import monotonic
import urllib.error
import urllib.request

from .code_agent import _ACTION_SCHEMA, _INSTRUCTIONS
from .code_provider_context import prepare_code_context


CHAT_INSTRUCTIONS = (_INSTRUCTIONS + "\nCall research_action exactly once for your next action. "
                     "The host executes it and supplies real results in the next request. "
                     "Wait for that feedback before choosing another action.")


class ChatCompletionsCodeProvider:
    """Use one explicitly selected Chat Completions model without retries."""

    def __init__(self, workspace, *, base_url, model, env_key="RUOLI_API_KEY",
                 timeout_seconds=180, max_output_tokens=8192,
                 cached_content=None, cached_task_interface=None):
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if cached_content and cached_task_interface is None:
            raise ValueError("cached_task_interface is required with cached_content")
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/chat/completions"):
            self.base_url += "/chat/completions"
        self.model, self.env_key = model, env_key
        self.cached_content, self.cached_task_interface = cached_content, cached_task_interface
        self.timeout_seconds, self.max_output_tokens = timeout_seconds, max_output_tokens
        self.calls = max([0, *(int(path.stem.rsplit("_", 1)[1])
                               for path in self.workspace.glob("api_response_[0-9][0-9][0-9].json"))])

    def next_action(self, request):
        if self.cached_content and request.get("task_interface") != self.cached_task_interface:
            raise ValueError("task_interface differs from the cached task interface")
        self.calls += 1
        context = prepare_code_context(self.workspace, request, schema=_ACTION_SCHEMA,
                                       instructions=_INSTRUCTIONS)
        if self.cached_content:
            context.pop("task_interface")
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": json.dumps(context)}],
                "max_tokens": self.max_output_tokens}
        if self.cached_content:
            body["extra_body"] = {"google": {"cached_content": self.cached_content}}
        else:
            body["messages"].insert(0, {"role": "system", "content": CHAT_INSTRUCTIONS})
            body["tools"] = [{"type": "function", "function": {
                    "name": "research_action",
                    "description": "Submit one laboratory or Python action for the host to execute.",
                    "parameters": _ACTION_SCHEMA}}]
        key = os.environ.get(self.env_key)
        usage = {"backend": "chat_completions", "runtime": self.base_url,
                 "calls": 1, "input_tokens": None, "cached_input_tokens": None, "output_tokens": None,
                 "total_tokens": None, "model": self.model}
        started = monotonic()
        request_data = urllib.request.Request(
            self.base_url, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}" if key else "",
                     "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request_data, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
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
        for source, target in (("prompt_tokens", "input_tokens"),
                               ("completion_tokens", "output_tokens"),
                               ("total_tokens", "total_tokens")):
            if isinstance(reported.get(source), int):
                usage[target] = reported[source]
        cached_tokens = (reported.get("prompt_tokens_details") or {}).get("cached_tokens")
        if isinstance(cached_tokens, int):
            usage["cached_input_tokens"] = cached_tokens
        choices = payload.get("choices", [])
        if not choices or all(not (choice.get("message", {}).get("tool_calls")
                                   or (choice.get("message", {}).get("content") or "").strip())
                              for choice in choices):
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"kind": "empty_response",
                              "message": "Provider returned no tool action or nonempty text"}}
        try:
            calls = [call for choice in choices
                     for call in choice.get("message", {}).get("tool_calls", [])]
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != "research_action":
                raise ValueError("expected exactly one research_action function call")
            action_payload = json.loads(calls[0]["function"]["arguments"])
            arguments = json.loads(action_payload["arguments_json"], strict=False)
            return {"action": action_payload["action"], "arguments": arguments, "usage": usage}
        except (ValueError, KeyError, TypeError, IndexError) as error:
            return {"action": "provider_error", "arguments": {}, "usage": usage,
                    "error": {"decode_error": f"{type(error).__name__}: {error}"}}
