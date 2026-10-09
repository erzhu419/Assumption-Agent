import json
import urllib.error

import pytest

from hegel_machine.cognition.responses_code_provider import ResponsesCodeProvider


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        yield b"event: response.created\n"
        yield b'data: {"type":"response.created"}\n'
        yield b"\n"
        yield b"data: " + json.dumps({"type": "response.completed", "response": self.payload}).encode() + b"\n"
        raise AssertionError("a completed response must not wait for the connection to close")


def message(text, phase):
    return {"type": "message", "role": "assistant", "phase": phase,
            "content": [{"type": "output_text", "text": text}]}


def function_call(action, arguments_json):
    return {"type": "function_call", "name": "research_action",
            "arguments": json.dumps({"action": action, "arguments_json": arguments_json}),
            "call_id": "call-1"}


def test_responses_provider_uses_native_function_and_actual_usage(tmp_path, monkeypatch):
    seen = {}

    def open_url(request, timeout):
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Response({"output": [message("intermediate commentary", "commentary"),
                                     function_call("run", '{"code":"print(1)"}')],
                         "usage": {"input_tokens": 13, "output_tokens": 5, "total_tokens": 18}})

    monkeypatch.setenv("TEST_RESPONSES_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    result = ResponsesCodeProvider(tmp_path, base_url="https://example.test/v1", model="terra",
                                   env_key="TEST_RESPONSES_KEY", timeout_seconds=181,
                                   reasoning_effort="low").next_action(
        {"observations": [{"value": index} for index in range(20)], "events": []})
    assert result["action"] == "run"
    assert result["arguments"] == {"code": "print(1)"}
    assert result["usage"]["total_tokens"] == 18
    assert seen["timeout"] == 181
    assert seen["body"]["model"] == "terra"
    assert seen["body"]["stream"] is True
    assert seen["body"]["reasoning"] == {"effort": "low"}
    assert seen["body"]["tools"][0]["type"] == "function"
    assert seen["body"]["tools"][0]["name"] == "research_action"
    assert seen["body"]["tool_choice"] == {"type": "function", "name": "research_action"}
    assert seen["body"]["parallel_tool_calls"] is False
    assert json.loads(seen["body"]["input"])["observations"] == [
        {"value": index, "row_index": index} for index in range(8, 20)]
    assert (tmp_path / "api_response_001.json").exists()


def test_responses_provider_keeps_missing_usage_and_reports_decode_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESPONSES_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response(
        {"output": [function_call("run", '{"code":"{"')]}))
    result = ResponsesCodeProvider(tmp_path, base_url="https://example.test/responses", model="terra",
                                   env_key="TEST_RESPONSES_KEY").next_action({})
    assert result["action"] == "provider_error"
    assert result["usage"]["input_tokens"] is None
    assert "JSONDecodeError" in result["error"]["decode_error"]


def test_responses_provider_rejects_missing_or_multiple_function_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESPONSES_KEY", "secret")
    outputs = [[message("I cannot provide a tool action", "final")],
               [{**function_call("run", '{"code":"1"}'), "name": "unknown_tool"}],
               [function_call("run", '{"code":"1"}'), function_call("finish", "{}")]]
    for output in outputs:
        monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout, output=output:
                            Response({"output": output, "usage": {}}))
        result = ResponsesCodeProvider(tmp_path, base_url="https://example.test", model="terra",
                                       env_key="TEST_RESPONSES_KEY").next_action({})
        assert result["action"] == "provider_error"
        assert result["error"].get("kind") != "empty_response"
        assert "exactly one research_action function call" in result["error"]["decode_error"]


@pytest.mark.parametrize("output", [[], [message("", "final")], [message("  \n", "final")]])
def test_completed_blank_response_is_service_failure_without_retry(tmp_path, monkeypatch, output):
    calls = []
    payload = {"status": "completed", "output": output, "model": "chosen-model"}

    def open_url(request, timeout):
        calls.append(request.full_url)
        return Response(payload)

    monkeypatch.setenv("TEST_RESPONSES_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    result = ResponsesCodeProvider(tmp_path, base_url="https://example.test/v1",
        model="chosen-model", env_key="TEST_RESPONSES_KEY").next_action({})

    assert calls == ["https://example.test/v1/responses"]
    assert result["action"] == "provider_error"
    assert result["error"]["kind"] == "empty_response"
    assert result["usage"]["total_tokens"] is None
    assert json.loads((tmp_path / "api_response_001.json").read_text()) == payload


def test_responses_provider_reports_http_failure_without_retry(tmp_path, monkeypatch):
    calls = []

    def open_url(request, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setenv("TEST_RESPONSES_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    result = ResponsesCodeProvider(tmp_path, base_url="https://example.test", model="terra",
                                   env_key="TEST_RESPONSES_KEY").next_action({})
    assert calls == [1]
    assert result["action"] == "provider_error"
    assert result["error"]["status"] == 503
