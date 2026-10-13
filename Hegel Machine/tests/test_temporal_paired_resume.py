"""Paid scripted resume checks, not evidence of autonomous LLM learning."""

from copy import deepcopy
import json
import shutil

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition import temporal_paired_practice as practice
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_practice import PairedMethodLearner
from test_temporal_paired_method import DIRECT_PRIOR_PROGRAM, calibrated_knowledge


CASES = {
    "case_a": {"task_index": 2, "apparatus_seed": 99106007, "confirmation_seed": 195},
    "case_b": {"task_index": 1, "apparatus_seed": 99106007, "confirmation_seed": 195},
}
CURRENT_SOURCE = DIRECT_PRIOR_PROGRAM
EARLIER_SOURCE = DIRECT_PRIOR_PROGRAM + "\n# Earlier authored version.\n"


def laboratory(task):
    return TemporalInquiryLab(task_index=task["task_index"], apparatus_seed=task["apparatus_seed"],
                              instrument_mode="recurrent")


def make_learner(workspace, knowledge, provider=None):
    return PairedMethodLearner(provider, workspace, knowledge=deepcopy(knowledge), cases=CASES,
        lab_factory=laboratory, config=InquiryLLMConfig(observation_budget=960, confirmation_rows=0,
            compute_budget=300000, execution_seconds=120., max_model_calls=8, token_budget=1000))


@pytest.fixture
def interrupted_practice(tmp_path, calibrated_knowledge):
    original = deepcopy(calibrated_knowledge)
    original["files"][METHOD_PATH] = DIRECT_PRIOR_PROGRAM + "\n# Inherited seed.\n"
    original["manifest"]["research_programs"].append({"id": "seed", "path": METHOD_PATH})
    actions = [
        ("write", {"path": METHOD_PATH, "code": EARLIER_SOURCE}),
        ("run", {"operator": "practice", "case": "case_a"}),
        ("write", {"path": METHOD_PATH, "code": CURRENT_SOURCE}),
        ("run", {"operator": "practice", "case": "case_b"}),
    ]

    class InterruptedAuthor:
        def __init__(self):
            self.calls = 0

        def next_action(self, request):
            step = self.calls
            self.calls += 1
            if step == len(actions):
                return {"action": "provider_error", "arguments": {},
                        "error": {"status": 503, "message": "scripted provider interruption"},
                        "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}
            action, arguments = actions[step]
            return {"action": action, "arguments": arguments,
                    "usage": {"backend": "scripted_interface_test", "total_tokens": 1}}

    previous = make_learner(tmp_path / "previous", original, InterruptedAuthor())
    result = previous.run()
    assert result["status"] == "provider_failed"
    assert result["method_path"] is None
    assert [trial["case"] for trial in result["practice_trials"]] == ["case_a", "case_b"]
    assert all(trial["status"] == "completed" for trial in result["practice_trials"])
    assert [trial["selected_prior_predictor"] for trial in result["practice_trials"]] == [False, True]
    assert result["costs"]["observation_cost"] == 192
    path = previous.workspace / "result.json"
    path.write_text(json.dumps(result))
    return {"original": original, "previous": previous, "result": result, "path": path}


def test_resume_preserves_only_current_support_and_requires_new_paid_missing_case(
        tmp_path, interrupted_practice, monkeypatch):
    prior = interrupted_practice
    resumed = make_learner(tmp_path / "resumed", prior["original"])
    context = resumed.restore_practice(prior["result"])
    assert context["current_source_coverage"] == {
        "confirmed_cases": ["case_b"], "missing_cases": ["case_a"],
        "prior_reuse_cases": ["case_b"], "revised_cases": [],
    }
    assert context["last_action"]["action"] == "run"
    assert context["last_action"]["arguments"] == {"operator": "practice", "case": "case_b"}
    assert context["last_action"]["feedback"]["fresh_confirmation"]["passed"]
    assert json.loads(resumed.files["resume_context.json"]) == context
    assert json.loads((resumed.workspace / "resume_context.json").read_text()) == context
    assert resumed.files[METHOD_PATH] == (resumed.workspace / METHOD_PATH).read_text() == CURRENT_SOURCE
    assert resumed.practice_trials == prior["result"]["practice_trials"]
    assert resumed.knowledge == prior["original"]
    assert resumed.events == resumed.observations == []
    assert all(value == 0 for value in resumed.costs.values())
    assert resumed.model is None

    instrument = prior["original"]["research_roles"]["instrument"]
    original_instrument = prior["original"]["files"][instrument]
    for trial in prior["result"]["practice_trials"]:
        old = prior["previous"].workspace / trial["directory"]
        restored = resumed.workspace / trial["directory"]
        assert restored.is_symlink() and restored.readlink() == old.resolve()
        for filename in ("result.json", "laboratory_observations.json", instrument):
            assert (restored / filename).samefile(old / filename)
    historical_fit = resumed.workspace / prior["result"]["practice_trials"][0]["directory"] / instrument
    assert historical_fit.read_text() != original_instrument
    assert (resumed.workspace / instrument).read_text() == original_instrument

    refused = resumed._execute("finish", {"path": METHOD_PATH}, resumed.interface)
    assert refused["unconfirmed_cases"] == ["case_a"]
    assert resumed.model is None and all(value == 0 for value in resumed.costs.values())

    initial_files = []
    actual_agent = practice.FrozenMethodAgent

    def capture_initial_prior(*args, **kwargs):
        agent = actual_agent(*args, **kwargs)
        initial_files.append({path: (agent.workspace / path).read_text()
                              for path in prior["original"]["research_roles"].values()})
        return agent

    monkeypatch.setattr(practice, "FrozenMethodAgent", capture_initial_prior)
    paid = resumed._execute("run", {"operator": "practice", "case": "case_a"}, resumed.interface)
    assert paid["status"] == "completed" and paid["fresh_confirmation"]["passed"]
    assert paid["selected_prior_predictor"] is False
    assert paid["directory"] == "trials/trial_002"
    assert not (resumed.workspace / paid["directory"]).is_symlink()
    assert paid["costs"]["observation_cost"] == 144
    assert initial_files == [{path: prior["original"]["files"][path]
                              for path in prior["original"]["research_roles"].values()}]
    for key in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert resumed.costs[key] == paid["costs"][key]
    assert resumed.costs["model_calls"] == resumed.costs["reported_tokens"] == 0
    published = resumed._execute("finish", {"path": METHOD_PATH, "lesson": "Verified both cases."},
                                 resumed.interface)
    assert published["published"] == METHOD_PATH
    assert published["confirmed_cases"] == ["case_a", "case_b"]
    assert resumed.result("completed")["method_source"] == CURRENT_SOURCE
    assert prior["result"]["costs"]["observation_cost"] == 192
    assert len(prior["result"]["practice_trials"]) == 2


def test_editing_restored_source_invalidates_coverage_without_recharging_history(
        tmp_path, interrupted_practice):
    prior = interrupted_practice
    resumed = make_learner(tmp_path / "edited_resume", prior["original"])
    resumed.restore_practice(prior["result"])
    feedback = resumed._execute("write", {"path": METHOD_PATH,
        "code": CURRENT_SOURCE + "\n# Newly edited, not yet practiced.\n"}, resumed.interface)
    assert feedback["current_source_coverage"] == {
        "confirmed_cases": [], "missing_cases": ["case_a", "case_b"],
        "prior_reuse_cases": [], "revised_cases": [],
    }
    refused = resumed._execute("finish", {"path": METHOD_PATH}, resumed.interface)
    assert refused["unconfirmed_cases"] == ["case_a", "case_b"]
    assert resumed.model is None
    assert resumed.practice_trials == prior["result"]["practice_trials"]
    assert all(value == 0 for value in resumed.costs.values())
    assert resumed.knowledge == prior["original"]


def test_restore_reuses_relocated_paid_trials_without_copying_or_replaying(
        tmp_path, interrupted_practice, monkeypatch):
    prior = interrupted_practice
    resumed = make_learner(tmp_path / "relocated", prior["original"])
    old_trials = prior["previous"].workspace / "trials"
    moved_trials = resumed.workspace / "trials"
    old_trials.rename(moved_trials)
    old_trials.symlink_to(moved_trials, target_is_directory=True)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("relocated paid history was copied again")

    monkeypatch.setattr(shutil, "copytree", unexpected_copy)
    context = resumed.restore_practice(prior["result"])
    assert context["previous_practice"] == [
        {k: v for k, v in t.items() if k != "method_source"}
        for t in prior["result"]["practice_trials"]]
    assert resumed.practice_trials == prior["result"]["practice_trials"]
    assert resumed.events == resumed.observations == []
    assert all(value == 0 for value in resumed.costs.values())
    for trial in resumed.practice_trials:
        source = prior["previous"].workspace / trial["directory"] / "result.json"
        target = resumed.workspace / trial["directory"] / "result.json"
        assert source.samefile(target)
    assert (resumed.workspace / METHOD_PATH).read_text() == CURRENT_SOURCE


def test_repeated_resume_references_original_trials_without_copying_or_moving(
        tmp_path, interrupted_practice, monkeypatch):
    prior = interrupted_practice
    original_trials = prior["previous"].workspace / "trials"
    before = {path.relative_to(original_trials): path.read_bytes()
              for path in original_trials.rglob("*") if path.is_file()}
    first = make_learner(tmp_path / "first_resume", prior["original"])
    second = make_learner(tmp_path / "second_resume", prior["original"])

    def unexpected_copy(*args, **kwargs):
        pytest.fail("paid history must be referenced, not copied")

    monkeypatch.setattr(shutil, "copytree", unexpected_copy)
    first.restore_practice(prior["result"])
    second.restore_practice(first.result("model_call_budget_exhausted"))
    assert first.practice_trials == second.practice_trials == prior["result"]["practice_trials"]
    for trial in second.practice_trials:
        original = prior["previous"].workspace / trial["directory"]
        reference = second.workspace / trial["directory"]
        assert reference.is_symlink() and reference.readlink() == original.resolve()
        assert (reference / "result.json").samefile(original / "result.json")
        assert not original.is_symlink()
    restored = second._execute("run", {"operator": "restore_method",
        "trial": second.practice_trials[0]["directory"]}, second.interface)
    assert restored["restored_method"] == "trials/trial_000"
    assert (second.workspace / METHOD_PATH).read_text() == EARLIER_SOURCE
    assert all(value == 0 for value in second.costs.values())
    assert before == {path.relative_to(original_trials): path.read_bytes()
                      for path in original_trials.rglob("*") if path.is_file()}


def test_fifty_seven_retained_trials_start_new_practice_at_trial_057(
        tmp_path, interrupted_practice, monkeypatch):
    prior = interrupted_practice
    previous = deepcopy(prior["result"])
    # Small stopped-trial fixtures exercise the current real continuation count.
    for index in range(2, 57):
        trial = deepcopy(previous["practice_trials"][0])
        trial.update(directory=f"trials/trial_{index:03d}", status="method_stopped",
                     fresh_confirmation=None, costs={})
        directory = prior["previous"].workspace / trial["directory"]
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps({"status": "method_stopped"}))
        previous["practice_trials"].append(trial)
    old_tree = prior["previous"].workspace / "trials"
    before = {path.relative_to(old_tree): path.read_bytes()
              for path in old_tree.rglob("*") if path.is_file()}
    resumed = make_learner(tmp_path / "resume_57", prior["original"])

    def unexpected_copy(*args, **kwargs):
        pytest.fail("restoration must not copy the 57 retained trial trees")

    with monkeypatch.context() as context:
        context.setattr(shutil, "copytree", unexpected_copy)
        resumed.restore_practice(previous)
    assert len(resumed.practice_trials) == 57
    outcome = resumed._execute("run", {"operator": "practice", "case": "case_a"}, resumed.interface)
    assert outcome["directory"] == "trials/trial_057"
    assert outcome["fresh_confirmation"]["passed"]
    current = resumed.workspace / outcome["directory"]
    assert current.is_dir() and not current.is_symlink()
    assert not (old_tree / "trial_057").exists()
    assert before == {path.relative_to(old_tree): path.read_bytes()
                      for path in old_tree.rglob("*") if path.is_file()}
