"""Decode authored source exactly, without a second nested source serialization."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

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


def assert_codex_source_schema(schema):
    ordinary = deepcopy(schema)
    source = ordinary["properties"].pop("source_code")
    assert source["type"] == ["string", "null"]
    ordinary["required"].remove("source_code")
    assert ordinary == code_agent._ACTION_SCHEMA
    assert "source_code" not in code_agent._ACTION_SCHEMA["properties"]


@pytest.mark.parametrize("action", ["write", "run"])
def test_native_source_keeps_regex_backslashes_and_generated_python_exactly(tmp_path, monkeypatch, action):
    generated = "import re\ndef predict(row):\n    return re.search(r'\\d+\\.\\d+', row['text']).group()\n"
    source = ("from pathlib import Path\nimport re\n"
              "pattern = re.compile(r'\\d+(?:\\.\\d+)?\\s*units')\n"
              f"Path('model.py').write_text({generated!r})\n"
              "literal = r'alpha\\beta\\gamma'\n")
    returned_action(monkeypatch, {"action": action, "arguments_json": '{"path":"author.py"}',
                                  "source_code": source})
    result = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action({})
    assert result["action"] == action
    assert result["arguments"] == {"path": "author.py", "code": source}
    assert result["usage"]["total_tokens"] == 38
    assert not (tmp_path / "author.py").exists()
    assert not (tmp_path / "model.py").exists()
    schema = json.loads((tmp_path / "code_action_schema.json").read_text())
    assert_codex_source_schema(schema)
    assert "top-level source_code" in (tmp_path / "code_agent_instructions.txt").read_text()


@pytest.mark.parametrize("source", [None, "print(1)"])
def test_null_or_identical_native_source_does_not_change_existing_arguments(tmp_path, monkeypatch, source):
    returned_action(monkeypatch, {"action": "run", "arguments_json": '{"code":"print(1)"}',
                                  "source_code": source})
    result = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action({})
    assert result["arguments"] == {"code": "print(1)"}


def test_conflicting_source_is_returned_for_author_revision_without_guessing(tmp_path, monkeypatch):
    returned_action(monkeypatch, {"action": "run", "arguments_json": '{"code":"print(1)"}',
                                  "source_code": "print(2)"})
    result = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action({})
    assert result["action"] == "provider_error" and result["arguments"] == {}
    assert "source_code conflicts" in result["error"]["decode_error"]
    assert result["usage"]["total_tokens"] == 38


def test_invalid_nested_escape_remains_an_error_even_with_native_source(tmp_path, monkeypatch):
    returned_action(monkeypatch, {"action": "run", "arguments_json": r'{"path":"bad\q.py"}',
                                  "source_code": "print(1)"})
    result = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action({})
    assert result["action"] == "provider_error"
    assert "JSONDecodeError" in result["error"]["decode_error"]
    assert result["usage"]["total_tokens"] == 38


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


def test_notebook_contract_requires_authored_memory_without_changing_numerical_contract(tmp_path, monkeypatch):
    previous = "Observed instrument reading in record 0; interpretation remains unresolved."
    revised = "Record 0 reports 3. Hypothesis: gain is 2. Next: vary input to distinguish it from an offset."
    calls = []

    def run(command, prompt, timeout, cwd):
        context = json.loads(prompt)
        schema_path = Path(command[command.index("--output-schema") + 1])
        schema = json.loads(schema_path.read_text())
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        payload = {"action": "run", "arguments_json": json.dumps({"code": "print(3)"})}
        if "research_notes" in context:
            assert context["research_notes"] == previous
            assert schema["properties"]["research_notes"]["type"] == "string"
            assert "research_notes" in schema["required"]
            assert "COMPLETE current notebook" in instructions
            assert "actual feedback" in instructions
            assert "has not happened yet" in instructions
            payload["research_notes"] = revised
        else:
            assert_codex_source_schema(schema)
            assert "research_notes" not in schema["properties"]
            assert instructions == code_agent._INSTRUCTIONS + code_agent._CODEX_SOURCE_INSTRUCTIONS
        calls.append(context)
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(payload))
        return 0, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 31, "output_tokens": 7}}), "", False

    monkeypatch.setattr(code_agent, "_run", run)
    provider = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test")
    result = provider.next_action({"task_interface": "Use world actions.", "research_notes": previous})
    assert result["arguments"] == {"code": "print(3)", "notes": revised}
    assert result["usage"]["total_tokens"] == 38
    numerical = provider.next_action({"observations": [{"features": {"u": 0.}, "target": 0.}]})
    assert numerical["arguments"] == {"code": "print(3)"}
    assert len(calls) == 2


def test_research_root_accepts_an_authored_child_or_null_without_changing_ordinary_tasks(tmp_path, monkeypatch):
    root = {"id": "task", "question": "Which intervention solves the public task?"}
    child = {"id": "calibration", "question": "Does the reading depend on the sensor offset?",
             "parent_id": "task", "depends_on": []}
    acquisition = {"id": "probe", "question": "What does the next probe report?",
                   "parent_id": "calibration", "depends_on": [], "analysis_path": "analyze_probe.py"}
    answers = iter([child, None])
    contexts = []

    def run(command, prompt, timeout, cwd):
        context = json.loads(prompt)
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        payload = {"action": "run", "arguments_json": json.dumps({"code": "print('inspect evidence')"})}
        if "research_root" in context:
            assert context["research_root"] == root
            assert "research_question" in schema["required"]
            variants = schema["properties"]["research_question"]["anyOf"]
            assert {"type": "null"} in variants
            question = next(item for item in variants if item["type"] == "object")
            assert question["additionalProperties"] is False
            assert set(question["required"]) == {"id", "question", "parent_id", "depends_on"}
            assert question["properties"]["parent_id"]["type"] == ["string", "null"]
            assert question["properties"]["depends_on"]["items"] == {"type": "string"}
            assert "stable question IDs" in instructions
            assert "no scientific support" in instructions
            payload["research_question"] = next(answers)
            if payload["research_question"] is not None:
                payload["arguments_json"] = json.dumps({"code": "print('inspect evidence')",
                                                         "question": acquisition})
        else:
            assert_codex_source_schema(schema)
            assert "research_question" not in schema["properties"]
            assert instructions == code_agent._INSTRUCTIONS + code_agent._CODEX_SOURCE_INSTRUCTIONS
        contexts.append(context)
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(payload))
        return 0, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 31, "output_tokens": 7}}), "", False

    monkeypatch.setattr(code_agent, "_run", run)
    provider = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test")
    request = {"task_interface": "Use the task's public evidence.", "research_root": root}
    registered = provider.next_action(request)
    assert registered["arguments"] == {"code": "print('inspect evidence')",
                                      "question": acquisition, "research_question": child}
    routine = provider.next_action(request)
    assert routine["arguments"] == {"code": "print('inspect evidence')"}
    ordinary = provider.next_action({"observations": []})
    assert ordinary["arguments"] == routine["arguments"]
    assert all(result["usage"]["total_tokens"] == 38 for result in (registered, routine, ordinary))
    assert len(contexts) == 3


def test_intermediate_write_is_not_executed_before_final_practice(tmp_path, monkeypatch):
    method = tmp_path / "shared/consolidated_method.py"
    method.parent.mkdir()
    original = "def select_actions(spec, records, memory): return {'stop': 'original'}\n"
    method.write_text(original)
    draft = {"action": "write", "arguments_json": json.dumps({
        "path": "shared/consolidated_method.py", "code": "raise RuntimeError('draft')\n"})}
    final = {"action": "run", "arguments_json": json.dumps({"operator": "practice", "case": "case_0"})}
    events = [
        {"type": "item.completed", "item": {"id": "draft", "type": "agent_message",
            "text": json.dumps(draft)}},
        {"type": "item.completed", "item": {"id": "final", "type": "agent_message",
            "text": json.dumps(final)}},
        {"type": "turn.completed", "usage": {
            "input_tokens": 147776, "cached_input_tokens": 104704, "output_tokens": 814}},
    ]

    def run(command, prompt, timeout, cwd):
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(final))
        return 0, "\n".join(json.dumps(event) for event in events), "", False

    monkeypatch.setattr(code_agent, "_run", run)
    reply = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action(
        {"task_interface": "Author a method using write and practice."})
    assert reply["action"] == "run"
    assert reply["arguments"] == {"operator": "practice", "case": "case_0"}
    assert method.read_text() == original
    assert reply["usage"]["total_tokens"] == 148590
    assert reply["usage"]["cached_input_tokens"] == 104704
    assert json.loads((tmp_path / "code_call_001.jsonl").read_text().splitlines()[0]) == events[0]


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
