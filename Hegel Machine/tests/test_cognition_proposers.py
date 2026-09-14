"""Proposal transport tests use a labeled CLI fixture; no real LLM claim here."""

import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.proposers import CodexProposer, _parse_proposals, build_prompt


def task():
    return TaskSpec("PRIVATE_TASK_LABEL", "PRIVATE_MECHANISM_DESCRIPTION", ("x",), (),
                    (Probe("PRIVATE_PROBE_LABEL", ({"x": 0.0}, {"x": 1.0})),),
                    context={"family": "PRIVATE_FAMILY_LABEL"},
                    feature_bounds={"x": (-2, 2)}, max_probe_length=3)


def rows():
    return [Observation({"x": x}, 2 * x + 1, ({"x": x - 1, "__target__": 2 * x - 1},))
            for x in (-1.0, 0.0, 1.0)]


def linear_program():
    return Program("add", children=(Program("mul", children=(Program("const", value=2.0),
                        Program("feature", name="x"))), Program("const", value=1.0)))


def test_prompt_contains_only_observed_data_and_public_capabilities():
    prompt = build_prompt(rows(), Program("const", value=0.0), task(), SearchBudget(100))
    assert "PRIVATE_" not in prompt
    payload = json.loads(prompt[prompt.index('{"training_observations"'):])
    assert payload["training_observations"][0]["history"][0]["__target__"] == -3.0
    assert payload["capabilities"]["feature_bounds"]["x"] == [-2, 2]
    assert payload["incumbent"] == Program("const", value=0.0).to_dict()


def test_actual_ast_execution_and_probe_capabilities_are_checked():
    payload = {"candidates": [
        {"program": linear_program().to_dict(), "rationale": "fit the observed slope"},
        {"program": Program("feature", name="mechanism_answer").to_dict()}],
        "probes": [{"sequence": [{"x": -2}, {"x": 2}]},
                   {"sequence": [{"x": 3}]}]}
    budget = SearchBudget(100)
    candidates, probes, errors = _parse_proposals(payload, rows(), task(), budget,
        call_number=1, max_candidates=4, max_probes=2, max_history=4)
    assert len(candidates) == 1 and candidates[0].program.predict(rows()[2]) == 3
    assert len(probes) == 1 and probes[0].cost == 2
    assert len(errors) == 2 and budget.spent == len(rows())


def fixture_cli(tmp_path: Path, *, tools=False, runtime_error=False) -> Path:
    script = tmp_path / "fixture_codex"
    response = {"candidates": [{"program": linear_program().to_dict(), "rationale": "fixture"}], "probes": []}
    script.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
        "if '--version' in sys.argv:\n print('codex-cli protocol-test');sys.exit()\n"
        "prompt=sys.stdin.read()\nassert 'PRIVATE_' not in prompt\n"
        "assert '--ephemeral' in sys.argv and 'read-only' in sys.argv\n"
        "assert any(a=='--enable' and b=='code_mode_host' for a,b in zip(sys.argv,sys.argv[1:]))\n"
        "assert not any(a=='--disable' and b=='code_mode_host' for a,b in zip(sys.argv,sys.argv[1:]))\n"
        "assert '-m' not in sys.argv and '--model' not in sys.argv\n"
        "assert 'project_doc_max_bytes=0' in sys.argv and 'notify=[]' in sys.argv\n"
        "instruction_arg=next(v for v in sys.argv if v.startswith('model_instructions_file='))\n"
        "instruction_path=Path(json.loads(instruction_arg.split('=',1)[1]))\n"
        "assert 'bounded program proposal component' in instruction_path.read_text()\n"
        f"response={response!r}\n"
        "Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(response))\n"
        "print(json.dumps({'type':'thread.started','model':'fixture-model'}))\n"
        + ("print(json.dumps({'type':'item.completed','item':{'type':'command_execution','command':'fixture'}}))\n" if tools else "")
        + ("print(json.dumps({'type':'item.completed','item':{'type':'error','message':'code_mode_host must be enabled'}}))\n" if runtime_error else "")
        + "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'cached_input_tokens':2,'output_tokens':7}}))\n")
    script.chmod(0o755)
    return script


def test_cli_protocol_records_usage_and_stops_after_token_budget(tmp_path):
    proposer = CodexProposer(executable=str(fixture_cli(tmp_path)), token_budget=15, max_calls=2)
    batch = proposer.propose(rows(), Program("const", value=0), task(), SearchBudget(100))
    assert batch.usage["status"] == "completed"
    assert batch.usage["runtime_version"] == "codex-cli protocol-test"
    assert batch.usage["model"] == "fixture-model" and batch.usage["total_tokens"] == 19
    assert batch.usage["token_budget_exceeded"] and batch.usage["calls"] == 1
    assert batch.candidates[0].program.predict(rows()[0]) == -1
    assert not Path(batch.usage["command"][batch.usage["command"].index('--cd') + 1]).exists()
    skipped = proposer.propose(rows(), Program("const", value=0), task(), SearchBudget(100))
    assert skipped.usage["status"] == "llm_budget_exhausted" and skipped.usage["calls"] == 0


def test_tool_using_completion_is_not_admitted_as_observation_only_baseline(tmp_path):
    proposer = CodexProposer(executable=str(fixture_cli(tmp_path, tools=True)))
    batch = proposer.propose(rows(), Program("const", value=0), task(), SearchBudget(100))
    assert batch.usage["status"] == "unexpected_tool_use"
    assert not batch.candidates and batch.usage["total_tokens"] == 19


def test_runtime_error_item_is_recorded_alongside_valid_proposals(tmp_path):
    proposer = CodexProposer(executable=str(fixture_cli(tmp_path, runtime_error=True)))
    batch = proposer.propose(rows(), Program("const", value=0), task(), SearchBudget(100))
    assert batch.usage["runtime_errors"] == ["code_mode_host must be enabled"]
    assert batch.usage["status"] == "completed" and batch.usage["total_tokens"] == 19
    assert batch.candidates[0].program.predict(rows()[0]) == -1


def test_missing_runtime_is_recorded_instead_of_synthetic_fallback():
    proposer = CodexProposer(executable="/does/not/exist")
    batch = proposer.propose(rows(), Program("const", value=0), task(), SearchBudget(100))
    assert batch.usage["status"] == "runtime_failed"
    assert not batch.candidates and batch.usage["calls"] == 0
