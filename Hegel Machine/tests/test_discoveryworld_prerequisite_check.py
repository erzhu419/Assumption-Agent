"""A second continuation must retain the first one's programs, answers and costs."""

from copy import deepcopy
import json

import pytest

from scripts import run_discoveryworld_prerequisite_check as runner
from test_discoveryworld_agent import LIMITS
from test_discoveryworld_prerequisite import ControlLab, PROBE
from test_discoveryworld_recovery import USAGE


def source_continuation(tmp_path):
    root = tmp_path / "source"
    source = root / "task" / "hegel"
    workspace = source / "agent"
    (workspace / "shared").mkdir(parents=True)
    task = {"id": "task"}
    plan = {"study": "prior_check", "tasks": [task], "runtime": {},
            "model": {"backend": "codex_cli", "requested_alias": "gpt-6-luna", "reasoning_effort": "medium"},
            "limits": {**LIMITS, "max_model_calls": 1}}
    runner.write_json(root / "protocol.json", plan)
    lab = ControlLab()
    records = []
    for action in ("READ", "ACTIVATE", "TALK"):
        records.append({"index": len(records), **lab.act({"action": action, "arg1": 7}),
                        "environment_steps": lab.environment_steps})
    root_question = {"id": "task", "question": "Find the control operation.", "parent_id": None,
                     "depends_on": [], "status": "reopened", "evidence": [],
                     "acquired_observations": [0, 1], "opened_before_observation": 0}
    child = {"id": "operation_7", "question": "Does TALK open the control interface?", "parent_id": "task",
             "depends_on": [], "status": "supported", "answer": {"operation": "TALK"}, "evidence": [2],
             "acquired_observations": [2], "opened_before_observation": 2,
             "analysis_path": "shared/operation_7.py", "reviewed_through_observation": 2}
    events = [{"action": "query", "arguments": {"research_question": root_question},
               "feedback": {"observation_indices": [0, 1]}, "phase": "frozen_prefix"},
              {"action": "run", "arguments": {"path": "scratch.py", "code": "print('old')"},
               "feedback": {}, "phase": "frozen_prefix"},
              {"action": "write", "arguments": {"path": "shared/operation_7.py", "code": PROBE,
                                                   "notes": "TALK is supported by observation 2."},
               "feedback": {"observation_indices": [2]}, "usage": USAGE}]
    research = {"questions": [root_question, child], "methods": [], "relation_research": {},
                "research_advice": None, "child_answers_returned": 1, "program_executions": 2}
    active = {"status": "model_call_budget_exhausted", "costs": {
        "model_calls": 1, "environment_steps": 3, "reported_tokens": 13,
        "input_tokens": 11, "output_tokens": 2, "cached_input_tokens": 7,
        "usage_missing_calls": 0, "python_execution_seconds": .1, "wall_time_seconds": 2},
        "prior_costs": {"model_calls": 2, "environment_steps": 2, "reported_tokens": 26,
                        "input_tokens": 22, "output_tokens": 4, "cached_input_tokens": 14},
        "prefix_event_count": 2, "score": {"agent_actions": 3}, "events": events,
        "research": research, "method_results": [{"step": 0, "path": "shared/operation_7.py"}]}
    runner.write_json(source / "active.json", active)
    runner.write_json(workspace / "laboratory_observations.json", records)
    runner.write_json(workspace / "research_state.json", {**research, "research_focus": None})
    (workspace / "shared/operation_7.py").write_text(PROBE + "\n# edited after the original write\n")
    (workspace / "shared/world_research_driver.py").write_text("def select_actions(*args): return []\n")
    (workspace / "scratch.py").write_text("print('updated')")
    return source, plan, active


def test_full_prefix_accumulates_costs_and_keeps_current_programs(tmp_path):
    source, _, active = source_continuation(tmp_path)
    output = tmp_path / "frozen"
    output.mkdir()
    _, prefix = runner.freeze_prefix(source, output)

    assert prefix["prior_costs"]["model_calls"] == 3
    assert prefix["prior_costs"]["reported_tokens"] == 39
    assert prefix["prior_costs"]["cached_input_tokens"] == 21
    assert prefix["prior_costs"]["environment_steps"] == 3
    assert prefix["source_checkpoint_costs"] == active["costs"]
    assert prefix["source_prior_costs"] == active["prior_costs"]
    assert prefix["research"]["program_executions"] == 2
    assert prefix["research"]["prerequisite_checked_through"] == 1
    assert prefix["notes"].startswith("TALK is supported")
    assert (output / "prefix_files/shared/operation_7.py").read_text().endswith("# edited after the original write\n")
    assert (output / "prefix_files/scratch.py").read_text() == "print('updated')"
    assert "shared/world_research_driver.py" in prefix["registered_files"]


def test_resume_preserves_research_and_charges_only_new_calls(tmp_path, monkeypatch):
    source, plan, _ = source_continuation(tmp_path)
    output = tmp_path / "resumed"
    output.mkdir()
    _, prefix = runner.freeze_prefix(source, output)
    runner.write_json(output / "protocol.json", plan)
    requests = []

    class Lab(ControlLab):
        def __init__(self, *_):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def score(self):
            return {**super().score(), "agent_actions": len(self.actions)}

    class Provider:
        def __init__(self, *_args, **_kwargs):
            pass

        def next_action(self, request):
            requests.append(deepcopy(request))
            return {"action": "finish", "arguments": {"lesson": "Use the supported control interface."},
                    "usage": USAGE}

    monkeypatch.setattr(runner, "DiscoveryWorldLab", Lab)
    monkeypatch.setattr(runner, "CodexCodeProvider", Provider)
    result = runner.continue_arm(plan, "hegel", output)
    request = requests[0]
    assert request["costs"]["model_calls"] == request["costs"]["reported_tokens"] == 0
    assert "edited after the original write" in request["files"]["shared/operation_7.py"]
    assert "shared/world_research_driver.py" in request["files"]
    assert request["research"]["answers"]["operation_7"]["status"] == "supported"
    assert result["research"]["program_executions"] == 2
    assert result["research"]["child_answers_returned"] == 1
    assert result["method_results"] == prefix["method_results"]
    assert result["costs"]["model_calls"] == 1
    assert result["costs"]["reported_tokens"] == 13
    assert result["new_environment_steps"] == 0
    assert all("usage" not in event for event in result["events"][:3])

    summary = runner.check_summary({**plan, "arms": ["hegel"], "jobs": [{"arm": "hegel"}],
        "prefix_source": str(source), "prefix_prior_costs": prefix["prior_costs"],
        "prefix_question_ids": [q["id"] for q in prefix["research"]["questions"]]}, [result])
    arm = summary["arms"]["hegel"]
    assert arm["new_program_executions"] == arm["new_child_answers_returned"] == 0
    assert arm["program_executions"] == 2
    assert arm["total_tokens_known_lower_bound"] == 13

    next_output = tmp_path / "next"
    next_output.mkdir()
    _, second = runner.freeze_prefix(output / "task/hegel", next_output)
    assert second["prior_costs"]["model_calls"] == 4
    assert second["prior_costs"]["reported_tokens"] == 52
    assert second["prior_costs"]["environment_steps"] == 3
    assert second["research"]["program_executions"] == 2


def test_root_only_historical_cut_still_uses_cut_events_not_later_files(tmp_path):
    source, _, active = source_continuation(tmp_path)
    # An ordinary source has one event and token accounting per call.
    active.pop("prior_costs")
    active.pop("prefix_event_count")
    active["costs"]["model_calls"] = 3
    for event in active["events"]:
        event["usage"] = USAGE
    runner.write_json(source / "active.json", active)
    output = tmp_path / "cut"
    output.mkdir()
    _, prefix = runner.freeze_prefix(source, output, prefix_calls=2)
    assert len(prefix["events"]) == len(prefix["records"]) == 2
    assert prefix["prior_costs"]["model_calls"] == 2
    assert prefix["prior_costs"]["reported_tokens"] == 26
    assert prefix["research"]["program_executions"] == 0
    assert len(prefix["research"]["questions"]) == 1
    assert (output / "prefix_files/scratch.py").read_text() == "print('old')"
    assert "shared/world_research_driver.py" not in prefix["copied_files"]


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_resuming_shared_knowledge_does_not_give_code_the_hegel_host_focus(tmp_path, monkeypatch, arm):
    source, plan, active = source_continuation(tmp_path)
    state_path = source / "agent/research_state.json"
    state = json.loads(state_path.read_text())
    state["research_focus"] = {"kind": "parent_research", "stage": "contrast",
        "question": deepcopy(active["research"]["questions"][0]),
        "comparison_question": deepcopy(active["research"]["questions"][1]),
        "path": "shared/contrast_operation_7.py", "origin": "host_parent_contrast_continuation"}
    runner.write_json(state_path, state)
    output = tmp_path / "resumed"
    output.mkdir()
    _, prefix = runner.freeze_prefix(source, output)
    runner.write_json(output / "protocol.json", plan)
    assert prefix["research"]["research_focus"] is not None
    requests = []

    class Lab(ControlLab):
        def __init__(self, *_):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def score(self):
            return {**super().score(), "agent_actions": len(self.actions)}

    class Provider:
        def __init__(self, *_args, **_kwargs):
            pass

        def next_action(self, request):
            requests.append(deepcopy(request))
            return {"action": "finish", "arguments": {"lesson": "Keep the shared evidence."}, "usage": USAGE}

    monkeypatch.setattr(runner, "DiscoveryWorldLab", Lab)
    monkeypatch.setattr(runner, "CodexCodeProvider", Provider)
    result = runner.continue_arm(plan, arm, output)
    assert result["status"] == "agent_finished" and len(requests) == 1
    request = requests[0]
    if arm == "code_archive":
        assert "research_focus" not in request
        assert request["research"]["research_focus"] is None
        assert result["research"]["research_focus"] is None
    else:
        assert request["research_focus"]["kind"] == "parent_research"
        assert request["research_focus"]["origin"] == "host_parent_research_continuation"
        assert request["research_focus"]["returned_children"][0]["id"] == "operation_7"
    assert request["research"]["answers"]["operation_7"] == {
        "status": "supported", "answer": {"operation": "TALK"}, "evidence": [2]}
    for path in ("shared/operation_7.py", "shared/world_research_driver.py", "scratch.py"):
        assert request["files"][path] == (output / "prefix_files" / path).read_text()
    assert request["evidence"]["observation_count"] == len(prefix["records"])
    assert result["new_environment_steps"] == 0 and result["research"]["program_executions"] == 2
