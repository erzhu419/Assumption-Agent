"""World practice uses paid execution and the old exact-source publisher."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.world_method_learning import (
    WorldMethodLearner, compact_author_events, run_world_method_author,
)
from test_discoveryworld_agent import InstrumentLab, LIMITS, ScriptedProvider


STOP = "def select_actions(spec, observations, memory):\n    return {'stop': 'Needs research'}\n"
SOLVE = """def select_actions(spec, observations, memory):
    if not observations:
        return {'actions': [{'action': 'MEASURE'}]}
    reading = observations[-1]['feedback']['reading']
    return {'actions': [{'action': 'TUNE', 'arg1': 2 * reading}]}
"""


class PracticeLab(InstrumentLab):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def prepared(tmp_path, source=STOP, *, child_steps=(), calls=8, completed=False):
    original = PracticeLab()
    records = [{"index": 0, **original.act({"action": "MEASURE"})},
               {"index": 1, **original.act({"action": "TUNE", "arg1": 6})}]
    labs, providers = [], []

    def laboratory(workspace):
        lab = PracticeLab()
        if completed:
            lab.completed = lab.success = True
        labs.append(lab)
        return lab

    def provider(workspace):
        instance = ScriptedProvider(child_steps)
        providers.append(instance)
        return instance

    learner = WorldMethodLearner(None, tmp_path, knowledge={"files": {METHOD_PATH: source}},
        task={"id": "source"}, records=records, prefix_steps=1, public_spec={},
        limits=LIMITS, calls=calls, runtime="unused", provider_factory=provider,
        laboratory_factory=laboratory, max_steps=6)
    return learner, labs, providers


def finish(learner):
    return learner._execute("finish", {"path": METHOD_PATH}, None)


def test_shared_author_loop_publishes_with_current_limits_and_terminal_checkpoint(tmp_path):
    learner, _, _ = prepared(tmp_path, calls=2)
    learner.provider = ScriptedProvider([
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"path": METHOD_PATH}),
    ])
    checkpoints = []
    experiences = [{"file": "shared/source_episode.json", "notes": "Earlier paid source"}]

    status = run_world_method_author(learner, task_interface="Author from source experience",
        source_experiences=experiences, on_checkpoint=checkpoints.append)

    assert status == checkpoints[-1] == "published_source_practiced"
    assert checkpoints[0] == "authoring"
    assert learner.published_source == SOLVE and learner.costs["model_calls"] == 2
    request = learner.provider.requests[0]
    assert request["limits"]["max_model_calls"] == 2
    assert request["source_experiences"] == experiences
    assert all(trial["fresh_confirmation"]["passed"] for trial in learner.practice_trials)


def test_shared_author_loop_preserves_provider_failure_in_terminal_checkpoint(tmp_path):
    learner, _, _ = prepared(tmp_path, calls=2)
    learner.provider = ScriptedProvider([])
    checkpoints = []

    status = run_world_method_author(learner, task_interface="Author from source experience",
        source_experiences=[], on_checkpoint=checkpoints.append)

    assert status == checkpoints[-1] == "provider_failed"
    assert learner.costs["model_calls"] == 1 and learner.published_source is None
    assert learner.events[-1]["action"] == "provider_error"


def test_registration_must_execute_then_edit_practice_and_publication_cover_both_cases(tmp_path):
    learner, labs, _ = prepared(tmp_path)
    refused = learner._execute("run", {"operator": "method", "path": METHOD_PATH}, None)
    assert "published" not in refused and learner.published_source is None
    assert len(learner.practice_trials) == 1
    assert not learner.practice_trials[0]["fresh_confirmation"]["passed"]
    edited = learner._execute("write", {"path": METHOD_PATH, "code": SOLVE}, None)
    assert edited["edit_practice"]["fresh_confirmation"]["passed"]
    assert edited["current_source_coverage"]["missing_cases"] == ["source_continuation"]
    published = finish(learner)
    assert published["published"] == METHOD_PATH
    assert learner.published_source == SOLVE
    assert published["current_source_coverage"]["confirmed_cases"] == [
        "source_continuation", "source_initial"]
    trial = json.loads((tmp_path / "trials/trial_002/result.json").read_text())
    assert trial["replayed_environment_steps"] == 1 and trial["new_environment_steps"] == 1
    assert trial["research"]["questions"][0]["status"] == "open"
    assert [row["index"] for row in trial["paid_observations"]] == [0, 1]
    assert all(lab.actions[-1]["action"] == "TUNE" for lab in labs[1:])
    before = deepcopy(learner.costs)
    assert finish(learner)["publication_practices"] == []
    assert learner.costs == before


def test_failed_revision_retains_an_actually_supported_old_source_through_original_finish(tmp_path):
    learner, _, _ = prepared(tmp_path, SOLVE)
    learner._practice("source_initial")
    learner._execute("write", {"path": METHOD_PATH, "code": STOP}, None)
    published = finish(learner)
    assert published["incumbent_comparison"]["status"] == "incumbent_published"
    assert published["incumbent_comparison"]["selected"]
    assert learner.published_source == (tmp_path / METHOD_PATH).read_text() == SOLVE
    assert len(learner.practice_trials) == 4
    assert [trial["status"] for trial in learner.practice_trials] == [
        "completed", "method_stopped", "method_stopped", "completed"]
    for key in ("observation_cost", "compute_cost", "python_execution_seconds"):
        assert learner.costs[key] == sum(trial["costs"][key] for trial in learner.practice_trials)


def test_practice_keeps_runtime_entry_revision_but_does_not_confirm_the_old_source(tmp_path):
    mutating = f"""def select_actions(spec, observations, memory):
    return {{'action': 'write', 'arguments': {{'path': {METHOD_PATH!r}, 'code': {SOLVE!r}}}}}
"""
    learner, _, _ = prepared(tmp_path, mutating)
    feedback = learner._practice("source_initial")
    result = json.loads((tmp_path / feedback["directory"] / "result.json").read_text())
    assert result["score"]["completedSuccessfully"]
    assert result["executed_source"] == SOLVE and result["entry_source_changed"]
    assert feedback["status"] == "entry_source_revised"
    assert feedback["fresh_confirmation"]["passed"] is False
    assert (tmp_path / feedback["directory"] / METHOD_PATH).read_text() == mutating
    confirmation = feedback["repaired_candidate_confirmation"]
    assert confirmation["retained"] and confirmation["practice"]["fresh_confirmation"]["passed"]
    assert confirmation["practice"]["costs"]["model_calls"] == 0
    assert (tmp_path / METHOD_PATH).read_text() == SOLVE
    assert learner._current_source_coverage()["confirmed_cases"] == ["source_initial"]
    assert len(learner.practice_trials) == 2


def test_nested_model_calls_consume_the_remaining_author_account_exactly_once(tmp_path):
    delegating = """def select_actions(spec, observations, memory):
    return {'action':'run','arguments':{'operator':'inquire','question':{
        'id':'measurement','question':'Measure and tune from the reading','parent_id':'task'},
        'max_model_calls':8}}
"""
    steps = [("query", {"actions": [{"action": "MEASURE"}]}),
             ("query", {"actions": [{"action": "TUNE", "arg1": 6}]})]
    learner, _, providers = prepared(tmp_path, delegating, child_steps=steps, calls=3)
    learner.costs["model_calls"] = 1  # One already-paid outer author request.
    feedback = learner._practice("source_initial")
    assert feedback["fresh_confirmation"]["passed"]
    assert len(providers[0].requests) == 2
    assert learner.costs["model_calls"] == 3 and learner.costs["reported_tokens"] == 2
    assert feedback["costs"]["model_calls"] == 2
    assert learner.costs["observation_cost"] == learner.costs["environment_steps"] == 2
    result = json.loads((tmp_path / feedback["directory"] / "result.json").read_text())
    assert len(result["events"]) == 2


def test_already_completed_source_cannot_pass_by_zero_action_replay(tmp_path):
    learner, _, _ = prepared(tmp_path, SOLVE, completed=True)
    with pytest.raises(ValueError, match="before world completion"):
        learner._practice("source_initial")
    assert not learner.practice_trials and learner.published_source is None


def test_author_practice_feedback_does_not_repeat_entire_world_executions(tmp_path):
    learner, _, _ = prepared(tmp_path)
    feedback = learner._practice("source_initial")
    learner.events = [
        {"action": "write", "arguments": {"path": METHOD_PATH},
         "feedback": {"edit_practice": deepcopy(feedback)}} for _ in range(20)]
    steps = [{"step": n, "feedback": {"error": f"Execution failure {n}"}}
             for n in range(64)]
    for event in learner.events:
        event["feedback"]["edit_practice"]["steps"] = deepcopy(steps)
    original = deepcopy(learner.events)
    projected = compact_author_events(learner.events)
    assert len(projected) == len(original)
    assert learner.events == original
    for event in projected:
        practice = event["feedback"]["edit_practice"]
        assert practice["step_count"] == 64
        assert practice["fresh_confirmation"]["passed"] is False
        assert (tmp_path / practice["full_trace_file"]).exists()
    assert all(not item["feedback"]["edit_practice"]["steps"] for item in projected[:-1])
    assert projected[-1]["feedback"]["edit_practice"]["steps"] == steps[-8:]


def test_world_author_retains_current_notes_across_practice_and_revision(tmp_path):
    learner, _, _ = prepared(tmp_path)
    learner._execute("write", {"path": METHOD_PATH, "code": SOLVE,
                              "notes": "Test initial world before publication."})
    assert (tmp_path / "research_notes.md").read_text() == "Test initial world before publication."
    learner._execute("run", {"operator": "practice", "case": "source_continuation",
                             "notes": "Initial passed; execute continuation."})
    assert (tmp_path / "research_notes.md").read_text() == "Initial passed; execute continuation."


def test_source_practice_adopts_repair_only_after_separate_unchanged_source_execution(tmp_path):
    broken = "def select_actions(spec, observations, memory):\n    return [}\n"
    learner, _, providers = prepared(tmp_path, broken, child_steps=[
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"lesson": "Syntax repaired; execute the revised procedure."}),
    ])
    feedback = learner._practice("source_initial")
    trial = json.loads((tmp_path / feedback["directory"] / "result.json").read_text())
    assert trial["score"]["completedSuccessfully"]
    assert trial["execution"]["repair"]["trigger"]["kind"] == "execution_error"
    assert trial["execution"]["continuation"]["status"] == "completed"
    assert feedback["status"] == "entry_source_revised" and not feedback["fresh_confirmation"]["passed"]
    assert (tmp_path / METHOD_PATH).read_text() == SOLVE
    assert (tmp_path / feedback["directory"] / METHOD_PATH).read_text() == broken
    assert (tmp_path / feedback["revised_source_path"]).read_text() == trial["executed_source"] == SOLVE
    assert learner.costs["model_calls"] == len(providers[0].requests) == 2
    confirmation = feedback["repaired_candidate_confirmation"]
    assert confirmation["retained"]
    assert confirmation["practice"]["fresh_confirmation"]["passed"]
    assert confirmation["practice"]["directory"] != feedback["directory"]
    assert confirmation["practice"]["costs"]["model_calls"] == 0
    frozen = json.loads((tmp_path / confirmation["practice"]["directory"] / "result.json").read_text())
    assert not frozen["entry_source_changed"] and not frozen["automatic_repair_used"]
    assert frozen["score"]["completedSuccessfully"] and frozen["executed_source"] == SOLVE
    assert learner._current_source_coverage()["confirmed_cases"] == ["source_initial"]
    assert len(learner.practice_trials) == 2 and len(providers) == 1
    assert finish(learner)["published"] == METHOD_PATH
    assert learner.costs["model_calls"] == 2


def test_finish_can_repair_confirm_and_publish_the_new_source_in_one_request(tmp_path):
    broken = "def select_actions(spec, observations, memory):\n    return [}\n"
    learner, _, providers = prepared(tmp_path, broken, child_steps=[
        ("write", {"path": METHOD_PATH, "code": SOLVE}),
        ("finish", {"lesson": "Syntax repaired; execute the revised procedure."}),
    ])

    published = finish(learner)

    assert published["published"] == METHOD_PATH
    assert learner.published_source == (tmp_path / METHOD_PATH).read_text() == SOLVE
    assert learner.costs["model_calls"] == sum(len(provider.requests) for provider in providers) == 2
    assert len(learner.practice_trials) == 3
    original, initial_confirmation, continuation = learner.practice_trials
    assert original["method_source"] == broken and original["status"] == "entry_source_revised"
    assert not original["fresh_confirmation"]["passed"]
    for trial in (initial_confirmation, continuation):
        assert trial["method_source"] == SOLVE
        assert trial["fresh_confirmation"]["passed"] and trial["costs"]["model_calls"] == 0
    assert published["current_source_coverage"]["confirmed_cases"] == [
        "source_continuation", "source_initial"]


def test_failed_other_case_and_incumbent_comparison_retain_the_repaired_candidate(tmp_path):
    initial_only = SOLVE.replace("    if not observations:",
        "    if memory['program_arguments']['procedure_start']:\n"
        "        return {'stop': 'Continuation still needs research'}\n"
        "    if not observations:")
    incumbent = "# Earlier initial-only method\n" + initial_only
    candidate = "# Newly repaired initial-only method\n" + initial_only
    broken = "def select_actions(spec, observations, memory):\n    return [}\n"
    learner, _, providers = prepared(tmp_path, incumbent, child_steps=[
        ("write", {"path": METHOD_PATH, "code": candidate}),
        ("finish", {"lesson": "The initial-world code is repaired."}),
    ])
    assert learner._practice("source_initial")["fresh_confirmation"]["passed"]
    # An already edited draft reaches finish with an earlier supported candidate.
    learner.files[METHOD_PATH] = broken
    (tmp_path / METHOD_PATH).write_text(broken)

    refused = finish(learner)

    assert "published" not in refused and learner.published_source is None
    assert refused["incumbent_comparison"]["status"] == "incumbent_practice_failed"
    assert refused["incumbent_comparison"]["current_failure_case"] == "source_continuation"
    assert (tmp_path / METHOD_PATH).read_text() == candidate
    assert learner.files[METHOD_PATH] == candidate
    assert learner._current_source_coverage()["confirmed_cases"] == ["source_initial"]
    assert learner._current_source_coverage()["missing_cases"] == ["source_continuation"]
    assert learner.costs["model_calls"] == sum(len(provider.requests) for provider in providers) == 2


def test_late_phase_repair_is_not_retained_when_it_fails_from_the_original_start(tmp_path):
    broken = "def select_actions(spec, observations, memory):\n    return [}\n"
    late_phase = """def select_actions(spec, observations, memory):
    reading = observations[-1]['feedback']['reading']
    return {'actions': [{'action': 'TUNE', 'arg1': 2 * reading}]}
"""
    learner, _, providers = prepared(tmp_path, broken, child_steps=[
        ("query", {"actions": [{"action": "MEASURE"}]}),
        ("write", {"path": METHOD_PATH, "code": late_phase}),
        ("finish", {"lesson": "Parent can now use the reading."}),
    ])
    feedback = learner._practice("source_initial")
    repaired = json.loads((tmp_path / feedback["directory"] / "result.json").read_text())
    assert repaired["score"]["completedSuccessfully"] and repaired["entry_source_changed"]
    assert repaired["executed_source"] == late_phase
    assert not feedback["fresh_confirmation"]["passed"]

    confirmation = feedback["repaired_candidate_confirmation"]
    assert not confirmation["retained"]
    assert not confirmation["practice"]["fresh_confirmation"]["passed"]
    frozen = json.loads((tmp_path / confirmation["practice"]["directory"] / "result.json").read_text())
    assert not frozen["score"]["completedSuccessfully"]
    assert frozen["execution"]["status"] == "method_error"
    assert frozen["costs"]["model_calls"] == 0 and not frozen["automatic_repair_used"]
    assert (tmp_path / METHOD_PATH).read_text() == broken
    assert (tmp_path / feedback["revised_source_path"]).read_text() == late_phase
    assert learner._current_source_coverage()["confirmed_cases"] == []
    assert len(learner.practice_trials) == 2 and len(providers) == 1
    assert learner.costs["model_calls"] == len(providers[0].requests) == 3


def test_automatic_child_completion_does_not_confirm_an_unchanged_broken_program(tmp_path):
    broken = "def select_actions(spec, observations, memory):\n    return [}\n"
    learner, _, providers = prepared(tmp_path, broken, calls=2, child_steps=[
        ("query", {"actions": [{"action": "MEASURE"}]}),
        ("query", {"actions": [{"action": "TUNE", "arg1": 6}]}),
    ])
    feedback = learner._practice("source_initial")
    trial = json.loads((tmp_path / feedback["directory"] / "result.json").read_text())
    assert trial["score"]["completedSuccessfully"]
    assert trial["execution"]["repair"]["trigger"]["kind"] == "execution_error"
    assert trial["executed_source"] == broken and not trial["entry_source_changed"]
    assert trial["status"] == feedback["status"] == "repair_assisted"
    assert not feedback["fresh_confirmation"]["passed"]
    assert learner._current_source_coverage()["confirmed_cases"] == []
    assert learner.costs["model_calls"] == len(providers[0].requests) == 2

    # The exact-source publisher must not retain this draft as a supported method.
    refused = finish(learner)
    assert "published" not in refused and learner.published_source is None
    assert refused["incumbent_comparison"]["status"] == "no_supported_incumbent"
    assert refused["current_source_coverage"]["confirmed_cases"] == []
