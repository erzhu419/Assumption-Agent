"""Match the rescue's real starting evidence and account, not scientific discovery."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_anonymous_online_research import representation_laboratory, representation_protocol
from scripts.run_observable_dependency_rescue import (
    prepare_rescue, rescue_continuation_protocol, rescue_protocol, run_rescue,
)
from test_common_archive_resume import SEED, checkpoint


def saved_origin(tmp_path):
    original = tmp_path / "original"
    old, lab, previous = checkpoint(original / "case_001")
    plan = representation_protocol(tmp_path / "archive")
    plan["cases"] = [{"id": "case_001", "kind": "recurrent", "seed": 113,
        "apparatus_seed": None, "prefix_seed": 10113, "confirmation_seed": SEED,
        "evaluation_seed": 30113, "parameter_pair": 0}]
    (original / "protocol.json").write_text(json.dumps(plan))
    (old.workspace / "training.json").write_text(json.dumps(previous))
    (old.workspace / "evaluation.json").write_text('{"terminal_label": 987654321}')
    return old, lab, json.loads((old.workspace / "training.json").read_text())


def test_protocol_adds_allowance_to_actual_usage_and_retains_shared_source(tmp_path):
    old, _, prior = saved_origin(tmp_path)
    plan = rescue_protocol(old.workspace)
    assert plan["limits"] == {
        "max_model_calls": prior["costs"]["model_calls"] + 24,
        "token_budget": prior["costs"]["reported_tokens"] + 1_600_000,
        "execution_seconds": prior["costs"]["python_execution_seconds"] + 120.,
        "compute_budget": prior["costs"]["compute_cost"] + 25_000,
        "observation_budget": 576, "confirmation_rows": 96}
    assert plan["cases"][0]["arms"] == ["code_react", "hegel"]
    assert plan["model"] == "gpt-6-luna" and plan["provider"]["reasoning_effort"] == "medium"
    assert plan["source_actor"] == "hegel"
    assert "src/hegel_machine/cognition/observable_research.py" in plan["runtime_snapshot"]
    assert "scripts/run_observable_dependency_rescue.py" in plan["runtime_snapshot"]
    assert "independent_parameter_pairs" not in plan


def test_both_arms_restore_identical_author_work_without_new_labels_or_code_focus(tmp_path):
    old, lab, prior = saved_origin(tmp_path)
    plan = rescue_protocol(old.workspace)
    prepare = prepare_rescue(plan)
    requests = []
    expected = lab.confirmation_actions(96, old.confirmation_rng)
    for arm in plan["arms"]:
        agent = CommonArchiveInquiryAgent(None, tmp_path / arm, arm=arm,
            config=InquiryLLMConfig(**plan["limits"]), source_task=False)
        target = representation_laboratory(plan["cases"][0])
        account = prepare(agent, target, plan["cases"][0])
        assert account["prior_research_cost"] == prior["costs"] == agent.costs
        assert agent.events == prior["events"] and agent.questions == prior["online_questions"]
        assert agent.rows == old.rows and agent.observations == prior["paid_observations"]
        assert agent.source_task is False
        assert target.confirmation_actions(96, agent.confirmation_rng) == expected
        assert not (agent.workspace / "evaluation.json").exists()
        assert (agent.workspace / "latest.py").read_bytes() == (old.workspace / "latest.py").read_bytes()
        assert (agent.workspace / "research_notes.md").read_bytes() == (old.workspace / "research_notes.md").read_bytes()
        assert agent._active_question_id == (None if arm == "code_react" else "new_focus")
        requests.append(agent._request(target))
    for field in ("session_knowledge", "observations", "laboratory_observations", "events", "research_notes"):
        assert requests[0][field] == requests[1][field]
    assert "987654321" not in json.dumps(requests)


def test_runner_counts_only_new_actions_as_rescue_cost(tmp_path):
    old, _, prior = saved_origin(tmp_path)
    plan = rescue_protocol(old.workspace)
    plan["limits"]["max_model_calls"] = prior["costs"]["model_calls"] + 1
    requests = []

    class InspectProvider:
        def __init__(self, workspace):
            self.workspace = Path(workspace)
            self.calls = 0

        def next_action(self, request):
            requests.append(deepcopy(request))
            (self.workspace / "research_context.json").write_text(json.dumps(request))
            return {"action": "run", "arguments": {"code": "print('accounting fixture')"},
                    "usage": {"backend": "scripted_rescue_check", "total_tokens": 3,
                              "cached_input_tokens": 1}}

    output = tmp_path / "rescue"
    summary = run_rescue(plan, output, provider_factory=InspectProvider)
    assert len(requests) == len(summary["trials"]) == 2
    assert summary["independent_source_count"] == 1
    assert summary["evidence_kind"] == "scripted_interface_check"
    for trial in summary["trials"]:
        assert trial["prior_research_cost"] == prior["costs"]
        assert trial["continuation_cost"]["model_calls"] == 1
        assert trial["continuation_cost"]["reported_tokens"] == 3
        assert trial["continuation_cost"]["observation_cost"] == 0
        assert trial["continuation_cached_input_tokens"] == 1
        assert trial["continuation_calls_with_cache_hits"] == 1
        result = json.loads((output / trial["case"] / trial["arm"] / "training.json").read_text())
        assert result["events"][:len(prior["events"])] == prior["events"]
    assert (output / "source_at_run/scripts/run_observable_dependency_rescue.py").is_file()


def truncated_rescue(tmp_path):
    old, _, prior = saved_origin(tmp_path)
    plan = rescue_protocol(old.workspace)
    plan["limits"]["max_model_calls"] = prior["costs"]["model_calls"] + 2
    plan["limits"]["token_budget"] = prior["costs"]["reported_tokens"] + 1

    class BranchProvider:
        def __init__(self, workspace):
            self.workspace = Path(workspace)
            self.calls = 0

        def next_action(self, request):
            (self.workspace / "research_context.json").write_text(json.dumps(request))
            return {"action": "write", "arguments": {"path": "shared/own_branch.py",
                "code": "# " + self.workspace.name, "notes": self.workspace.name + " notebook"},
                "usage": {"backend": "scripted_rescue_check", "total_tokens": 3,
                          "cached_input_tokens": 2}}

    output = tmp_path / "stopped_rescue"
    summary = run_rescue(plan, output, provider_factory=BranchProvider)
    assert [trial["status"] for trial in summary["trials"]] == ["token_budget_exhausted"] * 2
    return output, plan, summary


def test_rescue_continuation_changes_only_token_cap_after_all_branches_stop(tmp_path):
    source, original, summary = truncated_rescue(tmp_path)
    # A saved study predates the newly connected runtime module.
    original["runtime_snapshot"].remove("src/hegel_machine/cognition/conditional_research.py")
    (source / "protocol.json").write_text(json.dumps(original))
    continued = rescue_continuation_protocol(source)
    assert continued["limits"] == {**original["limits"], "token_budget": 10_000_000}
    assert continued["original_limits"] == original["limits"]
    assert continued["cases"] == original["cases"]
    assert continued["source_workspace"] == original["source_workspace"]
    assert continued["previous_runtime_snapshot"] == str(source / "source_at_run")
    assert "src/hegel_machine/cognition/conditional_research.py" in continued["runtime_snapshot"]
    summary["trials"].pop()
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="all registered"):
        rescue_continuation_protocol(source)


def test_rescue_continuation_restores_each_own_branch_and_counts_only_new_cost(tmp_path):
    source, original, _ = truncated_rescue(tmp_path)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    plan = rescue_continuation_protocol(source)
    requests = {}

    class ResumeProvider:
        def __init__(self, workspace):
            self.workspace = Path(workspace)
            self.calls = 0

        def next_action(self, request):
            arm = self.workspace.name
            requests[arm] = deepcopy(request)
            (self.workspace / "research_context.json").write_text(json.dumps(request))
            assert request["research_notes"] == arm + " notebook"
            assert (self.workspace / "shared/own_branch.py").read_text() == "# " + arm
            assert not (self.workspace / "evaluation.json").exists()
            return {"action": "write", "arguments": {"path": "shared/continued.py", "code": "# resumed"},
                "usage": {"backend": "scripted_rescue_check", "total_tokens": 5,
                          "cached_input_tokens": 1}}

    output = tmp_path / "continued_rescue"
    summary = run_rescue(plan, output, provider_factory=ResumeProvider)
    assert summary["source_count"] == summary["independent_source_count"] == 1
    assert summary["shared_origin"] == original["source_workspace"]
    assert set(requests) == {"code_react", "hegel"}
    for trial in summary["trials"]:
        prior = json.loads((source / trial["case"] / trial["arm"] / "training.json").read_text())
        assert trial["status"] == "model_call_budget_exhausted"
        assert trial["costs"]["model_calls"] == original["limits"]["max_model_calls"]
        assert trial["prior_research_cost"] == prior["costs"]
        assert trial["continuation_cost"]["model_calls"] == 1
        assert trial["continuation_cost"]["reported_tokens"] == 5
        assert trial["continuation_cost"]["observation_cost"] == 0
        assert trial["continuation_cached_input_tokens"] == 1
        assert trial["continuation_calls_with_cache_hits"] == 1
        result = json.loads((output / trial["case"] / trial["arm"] / "training.json").read_text())
        assert result["events"][:len(prior["events"])] == prior["events"]
        assert result["resume_origin"]["training"] == str(source / trial["case"] / trial["arm"] / "training.json")
    assert "987654321" not in json.dumps(requests)
    assert (output / "previous_summary.json").read_text() == (source / "summary.json").read_text()
    assert before == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_rescue_continuation_selects_every_token_truncated_branch_only(tmp_path):
    source, _, summary = truncated_rescue(tmp_path)
    finished = summary["trials"][0]
    finished["status"] = "model_call_budget_exhausted"
    path = source / finished["case"] / finished["arm"] / "training.json"
    training = json.loads(path.read_text())
    training["status"] = finished["status"]
    path.write_text(json.dumps(training))
    (source / "summary.json").write_text(json.dumps(summary))
    plan = rescue_continuation_protocol(source)
    assert plan["cases"][0]["arms"] == ["hegel"]
