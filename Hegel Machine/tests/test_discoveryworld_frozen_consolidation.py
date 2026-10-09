"""A live author must consolidate the program instead of borrowing a child model."""

import json
import shutil

import pytest

from hegel_machine.cognition import world_method_learning
from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts import prepare_discoveryworld_transfer_method as author
from test_discoveryworld_agent import ScriptedProvider
from test_discoveryworld_source_transfer import successful_source
from test_world_method_learning import PracticeLab, SOLVE, STOP


DELEGATE = """def select_actions(spec, observations, memory):
    return {'action': 'run', 'arguments': {'operator': 'inquire', 'question': {
        'id': 'measurement', 'question': 'Measure and tune', 'parent_id': 'task'},
        'max_model_calls': 8}}
"""
BROKEN = "def select_actions(spec, observations, memory):\n    return [}\n"
SOURCE_CONSTANT = """def select_actions(spec, observations, memory):
    if not observations:
        return {'actions': [{'action': 'MEASURE'}]}
    return {'actions': [{'action': 'TUNE', 'arg1': 6}]}
"""
OPENED_TASK_ID = "dw_04_challenge_s1"


@pytest.fixture
def frozen_source(successful_source, tmp_path, monkeypatch):
    source, result, _, _, plan = successful_source
    task = {"id": "dw_04_normal_s0", "random_seed": 0,
            "difficulty": "Normal", "split": "development"}
    plan["tasks"] = [task]
    (source.parents[1] / "protocol.json").write_text(json.dumps(plan))
    original = PracticeLab()
    records = [{"index": 0, **original.act({"action": "MEASURE"})},
               {"index": 1, **original.act({"action": "TUNE", "arg1": 6})}]
    result["prefix_environment_steps"] = 1
    (source / "result.json").write_text(json.dumps(result))
    (source / "agent/laboratory_observations.json").write_text(json.dumps(records))
    providers, agents, labs = [], [], []
    pending_providers = []

    def provider_factory(*args, **kwargs):
        provider = pending_providers.pop(0)
        providers.append(provider)
        return provider

    original_agent = world_method_learning.DiscoveryWorldAgent

    class ObservedPracticeAgent(original_agent):
        def __init__(self, provider, *args, **kwargs):
            super().__init__(provider, *args, **kwargs)
            agents.append(self)

    def laboratory(*args):
        lab = PracticeLab()
        labs.append(lab)
        return lab

    monkeypatch.setattr(author, "verify_codex_subscription", lambda: "test_subscription")
    monkeypatch.setattr(author, "CodexCodeProvider", provider_factory)
    monkeypatch.setattr(world_method_learning, "DiscoveryWorldAgent", ObservedPracticeAgent)
    monkeypatch.setattr(world_method_learning, "DiscoveryWorldLab", laboratory)
    return source, pending_providers, providers, agents, labs


def initial_bundle(tmp_path, program):
    bundle = tmp_path / "initial_bundle.json"
    bundle.write_text(json.dumps({"methods": [{"path": METHOD_PATH}],
        "files": {METHOD_PATH: program}, "consolidation_costs": {"model_calls": 7}}))
    return bundle


@pytest.mark.parametrize("program", [DELEGATE, BROKEN], ids=["explicit_child", "automatic_repair"])
def test_frozen_practice_returns_failure_to_live_author_for_executable_revision(
        frozen_source, tmp_path, program):
    source, pending, providers, agents, labs = frozen_source
    live_author = ScriptedProvider([
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"path": METHOD_PATH, "lesson": "Use the current reading directly"}),
    ])
    pending.append(live_author)
    output = tmp_path / "consolidation"

    status = author.prepare(source, output, calls=5,
        initial_bundle=initial_bundle(tmp_path, program), frozen_practice=True)

    assert status == "published_source_practiced"
    assert providers == [live_author] and len(live_author.requests) == 2
    assert len(agents) == len(labs) == 3
    assert all(agent.provider is None and agent.limits["max_model_calls"] > 0 for agent in agents)
    first_request = live_author.requests[0]
    assert first_request["costs"]["model_calls"] == 0
    assert "every practice executes without an LLM" in first_request["task_interface"]
    failed = first_request["events"][0]["feedback"]
    assert failed["status"] == "method_error" and not failed["fresh_confirmation"]["passed"]
    assert failed["costs"]["model_calls"] == 0
    result = json.loads((output / "result.json").read_text())
    assert result["method_source"] == SOLVE and result["costs"]["model_calls"] == 2
    assert [trial["fresh_confirmation"]["passed"] for trial in result["practice_trials"]] == [
        False, True, True]
    assert all(trial["costs"]["model_calls"] == 0 for trial in result["practice_trials"])
    failed_result = json.loads((output / "author/trials/trial_000/result.json").read_text())
    assert not failed_result["automatic_repair_used"]
    assert not failed_result["research"]["llm_inquiries"]
    if program == DELEGATE:
        assert "inquire needs the live model provider" in json.dumps(failed_result["execution"])
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["frozen_practice"] is True and protocol["max_model_calls"] == 5
    bundle = json.loads((output / "method_bundle.json").read_text())
    assert bundle["files"][METHOD_PATH] == SOLVE


def test_resume_preserves_frozen_practice_and_paid_budget_without_replaying_old_trial(
        frozen_source, tmp_path):
    source, pending, providers, agents, labs = frozen_source
    interrupted_author = ScriptedProvider([])
    pending.append(interrupted_author)
    interrupted = tmp_path / "interrupted"
    assert author.prepare(source, interrupted, calls=5,
        initial_bundle=initial_bundle(tmp_path, DELEGATE), frozen_practice=True) == "provider_failed"
    previous_path = interrupted / "result.json"
    previous_text = previous_path.read_text()
    previous = json.loads(previous_text)
    assert previous["costs"]["model_calls"] == 1 and len(labs) == 1
    resumed_author = ScriptedProvider([
        ("run", {"operator": "practice", "case": "source_initial"}),
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"path": METHOD_PATH}),
    ])
    pending.append(resumed_author)
    output = tmp_path / "resumed"

    # No repeat CLI flag: the saved study protocol owns the practice mode.
    assert author.prepare(source, output, resume_author=previous_path) == "published_source_practiced"

    assert providers == [interrupted_author, resumed_author]
    assert len(labs) == len(agents) == 4
    assert all(agent.provider is None and agent.limits["max_model_calls"] > 0 for agent in agents)
    first_request = resumed_author.requests[0]
    assert first_request["limits"]["max_model_calls"] == 5
    assert first_request["costs"]["model_calls"] == 1
    assert "every practice executes without an LLM" in first_request["task_interface"]
    failed = resumed_author.requests[1]["events"][-1]["feedback"]
    assert not failed["fresh_confirmation"]["passed"] and failed["costs"]["model_calls"] == 0
    result = json.loads((output / "result.json").read_text())
    assert result["practice_trials"][0] == previous["practice_trials"][0]
    assert result["costs"]["model_calls"] == 4
    assert all(trial["costs"]["model_calls"] == 0 for trial in result["practice_trials"])
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["frozen_practice"] is True and protocol["max_model_calls"] == 5
    assert protocol["resume_author"]["remaining_model_calls"] == 4
    assert previous_path.read_text() == previous_text
    with pytest.raises(ValueError, match="original total"):
        author.prepare(source, tmp_path / "different_budget", calls=6, resume_author=previous_path)


def test_source_notes_and_section_links_survive_provider_compaction_without_becoming_current_evidence(
        frozen_source, tmp_path):
    source, pending, _, _, _ = frozen_source
    result_path = source / "result.json"
    source_result = json.loads(result_path.read_text())
    source_result["events"][1]["feedback"]["stdout"] = "Public source execution log.\n" * 1000
    result_path.write_text(json.dumps(source_result))
    source_context_path = source / "agent/research_context.json"
    source_context = json.loads(source_context_path.read_text())
    source_context["research_notes"] = "Recorded source note: inspect the earlier instrument and test records."
    source_context_path.write_text(json.dumps(source_context))
    extra_source = tmp_path / "additional_successful_source"
    shutil.copytree(source, extra_source)
    extra_result = {**source_result, "task_id": "dw_04_normal_s2"}
    (extra_source / "result.json").write_text(json.dumps(extra_result))
    extra_task = {"id": "dw_04_normal_s2", "random_seed": 2,
                  "difficulty": "Normal", "split": "evaluation"}
    (extra_source / "protocol.json").write_text(json.dumps({"task": extra_task}))
    extra_context = {**source_context,
        "research_notes": "Recorded additional-source note: retain the successful measured steps."}
    (extra_source / "agent/research_context.json").write_text(json.dumps(extra_context))
    live_author = ScriptedProvider([
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"path": METHOD_PATH}),
    ])
    pending.append(live_author)

    assert author.prepare(source, tmp_path / "large_archive_author", calls=5,
        initial_bundle=initial_bundle(tmp_path, STOP), additional_sources=[extra_source],
        frozen_practice=True) == "published_source_practiced"

    request = live_author.requests[0]
    context = prepare_code_context(tmp_path / "provider_context", request, schema={}, instructions="")
    expected_files = ["shared/source_episode.json", "shared/source_dw_04_normal_s2.json"]
    expected_notes = [source_context["research_notes"], extra_context["research_notes"]]
    experiences = context["source_experiences"]
    assert experiences == request["source_experiences"]
    assert [item["file"] for item in experiences] == expected_files
    assert [item["notes"] for item in experiences] == expected_notes
    for experience, note in zip(experiences, expected_notes):
        source_file = experience["file"]
        assert len(request["files"][source_file]) > 12000
        preview = context["files"][source_file]
        assert preview["full_content_file"] == "research_context.json"
        assert preview["json_path"] == ["files", source_file]
        assert note not in preview["preview"]
        assert experience["sections"] == {"measurement_history": 0, "reference_history": 1,
            "continuation_observations": 1, "events": 2}
        assert set(experience) == {"file", "notes", "sections"}
    assert context["observations"] == []
    assert context["evidence"]["observation_count"] == 0


def changed_condition_labs(monkeypatch, labs):
    """An old source constant works until the actual current reading changes."""
    class ChangedConditionLab(PracticeLab):
        def __init__(self, task):
            self.task = task
            self.target = 8 if task["id"] == OPENED_TASK_ID else 6
            super().__init__(readings=(self.target / 2,))

        def act(self, action):
            observed = super().act(action)
            if action["action"] == "TUNE":
                self.success = action["arg1"] == self.target
            return observed

    def laboratory(task, *args):
        lab = ChangedConditionLab(task)
        labs.append(lab)
        return lab

    monkeypatch.setattr(world_method_learning, "DiscoveryWorldLab", laboratory)


def test_opened_failed_task_is_practiced_and_revised_without_fabricating_successful_source(
        frozen_source, tmp_path, monkeypatch):
    source, pending, _, agents, labs = frozen_source
    changed_condition_labs(monkeypatch, labs)
    case = OPENED_TASK_ID + "_initial"
    live_author = ScriptedProvider([
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"path": METHOD_PATH}),
    ])
    pending.append(live_author)
    output = tmp_path / "opened_task_author"

    assert author.prepare(source, output, calls=5,
        initial_bundle=initial_bundle(tmp_path, SOURCE_CONSTANT),
        frozen_practice=True, practice_task_ids=[OPENED_TASK_ID]) == "published_source_practiced"

    first_request = live_author.requests[0]
    assert case in first_request["practice_cases"]
    assert [item["file"] for item in first_request["source_experiences"]] == [
        "shared/source_episode.json"]
    assert f"shared/source_{OPENED_TASK_ID}.json" not in first_request["files"]
    failure = first_request["events"][-1]["feedback"]
    assert not failure["fresh_confirmation"]["passed"]
    actual = json.loads((output / "author" / failure["directory"] / "result.json").read_text())
    assert actual["paid_observations"][0]["feedback"]["reading"] == 4
    assert not actual["score"]["completedSuccessfully"]
    result = json.loads((output / "result.json").read_text())
    trials = result["practice_trials"]
    assert [trial["fresh_confirmation"]["passed"] for trial in trials] == [
        True, False, True, True, True]
    assert {trial["case"] for trial in trials if trial["method_source"] == SOLVE} == {
        "source_initial", "source_continuation", case}
    assert all(agent.provider is None for agent in agents)
    assert all(trial["costs"]["model_calls"] == 0 for trial in trials)
    challenge_labs = [lab for lab in labs if lab.task["id"] == OPENED_TASK_ID]
    assert [lab.success for lab in challenge_labs] == [False, True]
    assert challenge_labs[-1].actions == [
        {"action": "MEASURE"}, {"action": "TUNE", "arg1": 8}]
    protocol = json.loads((output / "protocol.json").read_text())
    assert [task["id"] for task in protocol["additional_practice_tasks"]] == [OPENED_TASK_ID]
    assert protocol["additional_sources"] == []
    bundle = json.loads((output / "method_bundle.json").read_text())
    assert bundle["development_task_ids"] == ["dw_04_normal_s0", OPENED_TASK_ID]
    assert f"shared/source_{OPENED_TASK_ID}.json" not in bundle["files"]


def test_resume_restores_opened_practice_tasks_and_original_budget(
        frozen_source, tmp_path, monkeypatch):
    source, pending, _, _, labs = frozen_source
    changed_condition_labs(monkeypatch, labs)
    pending.append(ScriptedProvider([]))
    interrupted = tmp_path / "interrupted_opened_task"
    assert author.prepare(source, interrupted, calls=5,
        initial_bundle=initial_bundle(tmp_path, SOURCE_CONSTANT),
        frozen_practice=True, practice_task_ids=[OPENED_TASK_ID]) == "provider_failed"
    previous_path = interrupted / "result.json"
    previous_text = previous_path.read_text()
    resumed_author = ScriptedProvider([
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"path": METHOD_PATH}),
    ])
    pending.append(resumed_author)
    output = tmp_path / "resumed_opened_task"

    assert author.prepare(source, output, resume_author=previous_path) == "published_source_practiced"

    first_request = resumed_author.requests[0]
    assert OPENED_TASK_ID + "_initial" in first_request["practice_cases"]
    assert first_request["costs"]["model_calls"] == 1
    assert first_request["limits"]["max_model_calls"] == 5
    result = json.loads((output / "result.json").read_text())
    assert result["costs"]["model_calls"] == 3
    assert len(labs) == len(result["practice_trials"]) == 5
    protocol = json.loads((output / "protocol.json").read_text())
    previous_protocol = json.loads((interrupted / "protocol.json").read_text())
    assert protocol["additional_practice_tasks"] == previous_protocol["additional_practice_tasks"]
    assert protocol["frozen_practice"] is True
    bundle = json.loads((output / "method_bundle.json").read_text())
    assert OPENED_TASK_ID in bundle["development_task_ids"]
    assert previous_path.read_text() == previous_text
    with pytest.raises(ValueError, match="existing draft"):
        author.prepare(source, tmp_path / "changed_opened_task_set", resume_author=previous_path,
            practice_task_ids=[OPENED_TASK_ID])
