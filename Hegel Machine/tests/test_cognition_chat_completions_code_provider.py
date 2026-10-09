"""Native Chat tools carry the same research protocol without provider retries."""

import io
import json
import urllib.error

import pytest

from hegel_machine.cognition.chat_completions_code_provider import CHAT_INSTRUCTIONS, ChatCompletionsCodeProvider
from hegel_machine.cognition.code_agent import _ACTION_SCHEMA, _INSTRUCTIONS
from hegel_machine.cognition.code_provider_context import prepare_code_context


def function_call(action, arguments):
    return {"id": "call-1", "type": "function", "function": {
        "name": "research_action", "arguments": json.dumps({
            "action": action, "arguments_json": json.dumps(arguments)})}}


def invoke(tmp_path, monkeypatch, payload, request=None, **provider_options):
    calls = []

    def open_url(request, timeout):
        calls.append({"url": request.full_url, "body": json.loads(request.data), "timeout": timeout})
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setenv("TEST_CHAT_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    provider = ChatCompletionsCodeProvider(tmp_path, base_url="https://example.test/v1",
        model="chosen-model", env_key="TEST_CHAT_KEY", timeout_seconds=181, max_output_tokens=2048,
        **provider_options)
    return provider.next_action(request or {}), calls


def test_native_chat_keeps_scientific_context_schema_and_actual_usage(tmp_path, monkeypatch):
    request = {"capabilities": {"controls": ["left", "right"]},
               "observations": [{"target": index} for index in range(20)], "events": [],
               "session_knowledge": {"lessons": ["probe both controls"]}}
    payload = {"model": "returned-model", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning_content": "private reasoning",
        "tool_calls": [function_call("run", {"code": "print(1)"})]}}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}}
    result, calls = invoke(tmp_path, monkeypatch, payload, request)

    assert len(calls) == 1
    assert calls[0]["url"] == "https://example.test/v1/chat/completions"
    assert calls[0]["timeout"] == 181
    body = calls[0]["body"]
    assert body["model"] == "chosen-model"
    assert body["stream"] is False
    assert body["max_tokens"] == 2048
    assert "tool_choice" not in body and "parallel_tool_calls" not in body
    assert body["tools"][0]["type"] == "function"
    function = body["tools"][0]["function"]
    assert function["name"] == "research_action"
    assert function["parameters"] == _ACTION_SCHEMA
    assert "strict" not in function
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == CHAT_INSTRUCTIONS
    assert CHAT_INSTRUCTIONS == (_INSTRUCTIONS + "\nCall research_action exactly once for your next action. "
        "The host executes it and supplies real results in the next request. "
        "Wait for that feedback before choosing another action.")
    assert "extra_body" not in body
    expected = prepare_code_context(tmp_path / "expected", request,
                                    schema=_ACTION_SCHEMA, instructions=_INSTRUCTIONS)
    assert json.loads(body["messages"][1]["content"]) == expected
    assert result["action"] == "run"
    assert result["arguments"] == {"code": "print(1)"}
    assert result["usage"]["backend"] == "chat_completions"
    assert result["usage"]["input_tokens"] == 13
    assert result["usage"]["output_tokens"] == 5
    assert result["usage"]["total_tokens"] == 18
    assert result["usage"]["returned_model"] == "returned-model"
    assert json.loads((tmp_path / "api_response_001.json").read_text()) == payload


def test_nested_code_escaping_is_preserved_without_extra_unescape(tmp_path, monkeypatch):
    code = 'import re\npattern = r"\\bitem\\d+\\b"\nprint("line\\nnext")\n'
    payload = {"choices": [{"message": {"content": None,
        "reasoning_content": '{"action":"finish","arguments_json":"{}"}',
        "tool_calls": [function_call("write", {"path": "model.py", "code": code})]}}]}
    result, calls = invoke(tmp_path, monkeypatch, payload)

    assert len(calls) == 1
    assert result["action"] == "write"
    assert result["arguments"] == {"path": "model.py", "code": code}
    assert result["usage"]["total_tokens"] is None


@pytest.mark.parametrize("message", [
    {"content": '{"action":"run","arguments_json":"{}"}'},
    {"content": None, "tool_calls": [function_call("run", {}), function_call("finish", {})]},
    {"content": None, "tool_calls": [{"type": "function", "function": {
        "name": "wrong_tool", "arguments": "{}"}}]},
    {"content": None, "tool_calls": [{"type": "function", "function": {
        "name": "research_action", "arguments": "not json"}}]},
])
def test_nonempty_invalid_actions_are_decode_failures_without_retry(tmp_path, monkeypatch, message):
    result, calls = invoke(tmp_path, monkeypatch, {"choices": [{"message": message}]})
    assert len(calls) == 1
    assert result["action"] == "provider_error"
    assert "decode_error" in result["error"]
    assert result["error"].get("kind") != "empty_response"


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_completed_blank_content_is_empty_response_even_with_reasoning(tmp_path, monkeypatch, content):
    payload = {"choices": [{"finish_reason": "stop", "message": {
        "content": content, "reasoning_content": "I should issue a tool call."}}]}
    result, calls = invoke(tmp_path, monkeypatch, payload)
    assert len(calls) == 1
    assert result["action"] == "provider_error"
    assert result["error"]["kind"] == "empty_response"
    assert result["usage"]["total_tokens"] is None


def test_http_failure_retains_status_and_body_without_retry(tmp_path, monkeypatch):
    calls = []
    detail = json.dumps({"error": {"code": "model_unavailable", "message": "model unavailable"}})

    def open_url(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 500, "unavailable", {},
                                     io.BytesIO(detail.encode()))

    monkeypatch.setenv("TEST_CHAT_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    provider = ChatCompletionsCodeProvider(tmp_path,
        base_url="https://example.test/v1/chat/completions", model="chosen-model", env_key="TEST_CHAT_KEY")
    result = provider.next_action({})
    assert calls == ["https://example.test/v1/chat/completions"]
    assert result["action"] == "provider_error"
    assert result["error"]["status"] == 500
    assert result["error"]["message"] == detail
    assert result["usage"]["total_tokens"] is None


def test_cached_static_prefix_is_omitted_and_dynamic_research_context_is_preserved(tmp_path, monkeypatch):
    interface = "query buys an observation; finish freezes the model"
    request = {"task_interface": interface, "knowledge_condition": "hegel",
               "capabilities": {"controls": ["left", "right"]},
               "observations": [{"target": index} for index in range(20)],
               "events": [{"action": "query", "arguments": {"kind": "response"},
                           "feedback": {"observations": [1.0]}}],
               "session_knowledge": {"lessons": ["distinguish the two controls"]}}
    payload = {"choices": [{"message": {
        "content": None, "tool_calls": [function_call("run", {"code": "print(1)"})]}}]}
    result, calls = invoke(tmp_path, monkeypatch, payload, request,
                           cached_content="cachedContents/research-prefix",
                           cached_task_interface=interface)

    assert len(calls) == 1
    body = calls[0]["body"]
    assert body["extra_body"] == {"google": {"cached_content": "cachedContents/research-prefix"}}
    assert "tools" not in body
    assert [message["role"] for message in body["messages"]] == ["user"]
    expected = prepare_code_context(tmp_path / "expected", request,
                                    schema=_ACTION_SCHEMA, instructions=_INSTRUCTIONS)
    del expected["task_interface"]
    assert json.loads(body["messages"][0]["content"]) == expected
    assert json.loads((tmp_path / "research_context.json").read_text()) == request
    assert result["action"] == "run"


def test_cached_interface_change_is_rejected_before_network_request(tmp_path, monkeypatch):
    calls = []

    def open_url(request, timeout):
        calls.append(request.full_url)
        raise AssertionError("a mismatched cached research interface must not reach the service")

    monkeypatch.setenv("TEST_CHAT_KEY", "secret")
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    provider = ChatCompletionsCodeProvider(tmp_path, base_url="https://example.test/v1",
        model="chosen-model", env_key="TEST_CHAT_KEY", cached_content="cachedContents/research-prefix",
        cached_task_interface="the original scientific interface")
    with pytest.raises(ValueError):
        provider.next_action({"task_interface": "a changed scientific interface", "events": []})
    assert calls == []


@pytest.mark.parametrize("details,expected", [(None, None), ({"cached_tokens": 0}, 0),
                                            ({"cached_tokens": 11}, 11)])
def test_cached_input_usage_preserves_zero_and_does_not_infer_missing_counts(
        tmp_path, monkeypatch, details, expected):
    usage = {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}
    if details is not None:
        usage["prompt_tokens_details"] = details
    payload = {"choices": [{"message": {
        "content": None, "tool_calls": [function_call("run", {"code": "print(1)"})]}}],
        "usage": usage}
    result, calls = invoke(tmp_path, monkeypatch, payload)
    assert len(calls) == 1
    assert result["usage"]["cached_input_tokens"] == expected
    assert result["usage"]["input_tokens"] == 13
    assert result["usage"]["total_tokens"] == 18
