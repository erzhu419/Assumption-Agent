"""The real CLI can put literal line breaks inside its nested arguments JSON."""

import json
from pathlib import Path

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


def returned_action(monkeypatch, payload):
    def run(command, prompt, timeout, cwd):
        destination = Path(command[command.index("--output-last-message") + 1])
        destination.write_text(json.dumps(payload), encoding="utf-8")
        stdout = json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 31, "output_tokens": 7}}) + "\n"
        return 0, stdout, "", False
    monkeypatch.setattr(code_agent, "_run", run)


def test_literal_line_breaks_in_nested_code_reach_executor_unchanged(tmp_path, monkeypatch):
    returned_action(monkeypatch, {"action": "run",
        "arguments_json": '{"code":"print(1)\nprint(2)"}'})
    result = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action({})
    assert result["action"] == "run"
    assert result["arguments"] == {"code": "print(1)\nprint(2)"}
    assert result["usage"]["total_tokens"] == 38


def test_malformed_arguments_keep_actual_reported_usage(tmp_path, monkeypatch):
    returned_action(monkeypatch, {"action": "run", "arguments_json": '{"code":'})
    result = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action({})
    assert result["action"] == "provider_error"
    assert result["usage"]["total_tokens"] == 38
    assert "JSONDecodeError" in result["error"]["decode_error"]


def test_model_can_repair_its_arguments_within_original_call_budget(tmp_path):
    class Repair:
        def next_action(self, request):
            index = len(request["events"])
            if index == 0:
                return {"action": "provider_error", "arguments": {},
                    "error": {"decode_error": "JSONDecodeError: Invalid escape"},
                    "usage": {"total_tokens": 11}}
            if index == 1:
                assert "JSONDecodeError" in request["events"][-1]["feedback"]["error"]
                action, arguments = "query", {"sequence": [{"u": 0.}], "channels": []}
            elif index == 2:
                action, arguments = "write", {"path": "model.py", "code": "def predict(row): return 0."}
            else:
                action, arguments = "finish", {"model": "model.py", "channels": []}
            return {"action": action, "arguments": arguments, "usage": {"total_tokens": 7}}

    agent = code_agent.CodeAgent(Repair(), tmp_path, code_agent.CodeAgentConfig(max_model_calls=4))
    with UnknownTaskLab(task_index=0, seed=91) as lab:
        result = agent.run(lab)
    assert result["status"] == "completed"
    assert result["costs"]["model_calls"] == 4
    assert result["costs"]["reported_tokens"] == 32
    assert result["events"][0]["action"] == "provider_error"
