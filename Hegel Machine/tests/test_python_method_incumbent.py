"""A failed draft can only yield to an older draft with its own measured support."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from scripts import run_python_method_reopening_study as runner
from test_python_method_edit_practice import change, learner_with_real_practice
from test_python_method_learning import (
    ADAPTIVE_METHOD, DEPENDENCY_EDIT_METHOD, archived_predictor, make_learner,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = ROOT / "artifacts/python_method_edit_practice_luna_20261006"
FAILED = ROOT / "artifacts/python_method_recall_compact_luna_20261007/author_2/learning/result.json"


def resumed_author(tmp_path):
    plan = json.loads((ORIGIN / "protocol.json").read_text())
    sources = json.loads((ORIGIN / "existing_source_outcomes.json").read_text())["sources"]
    seed = (ORIGIN / "unchanged_method.py").read_text()
    learner = runner.prepare_author(None, tmp_path, plan, plan["author_replicas"][2],
        sources, seed, ORIGIN / "paid_source_counterexample.json")
    previous = json.loads(FAILED.read_text())
    learner.restore_practice(previous)
    assert learner.cases == plan["author_replicas"][2]["practice_cases"]
    assert learner._current_source_coverage()["missing_cases"] == ["case_4"]
    return learner, previous


def finish(learner):
    return learner._execute("finish", {"path": runner.METHOD_PATH}, learner.interface)


def test_failed_current_draft_yields_to_old_source_only_after_its_five_missing_practices(tmp_path):
    learner, previous = resumed_author(tmp_path)
    original_source = (learner.workspace / runner.METHOD_PATH).read_text()
    old = previous["practice_trials"][3]
    assert old["directory"] == "trials/trial_003" and old["case"] == "case_5"
    count = len(learner.practice_trials)
    result = finish(learner)
    comparison = result["incumbent_comparison"]
    assert comparison["old_trial"] == old["directory"]
    assert comparison["current_failure_case"] == "case_4"
    assert comparison["selected"] is True and comparison["status"] == "incumbent_published"
    assert result["published"] == runner.METHOD_PATH
    assert learner.published_source == (learner.workspace / runner.METHOD_PATH).read_text() == old["method_source"]
    assert learner.published_source != original_source
    assert result["current_source_coverage"]["confirmed_cases"] == [f"case_{i}" for i in range(6)]
    assert result["current_source_coverage"]["scoped_revision"]["distinct_cases"] is True
    paid = learner.practice_trials[count:]
    assert len(paid) == 6
    assert paid[0]["method_source"] == original_source and paid[0]["case"] == "case_4"
    assert paid[0]["status"] != "completed"
    assert [trial["case"] for trial in paid[1:]] == ["case_4", "case_0", "case_1", "case_2", "case_3"]
    assert all(trial["method_source"] == old["method_source"] and trial["status"] == "completed"
               for trial in paid[1:])
    assert [trial["case"] for trial in comparison["practices"]] == [trial["case"] for trial in paid[1:]]
    for resource in ("observation_cost", "measurement_cost", "compute_cost", "python_execution_seconds"):
        assert learner.costs[resource] == sum(trial["costs"][resource] for trial in paid)
    assert learner.costs["model_calls"] == 0
    # Publication repeats neither the retained case_5 nor any newly confirmed case.
    costs = deepcopy(learner.costs)
    assert finish(learner)["publication_practices"] == []
    assert len(learner.practice_trials) == count + 6 and learner.costs == costs


def test_current_and_old_draft_both_fail_without_publishing_or_mixing_support(tmp_path):
    learner, previous = resumed_author(tmp_path)
    current_source = (learner.workspace / runner.METHOD_PATH).read_text()
    old_variant = current_source + "\n# Separate old draft with the same case_4 failure.\n"
    assert change(learner, "write", old_variant)["edit_practice"]["fresh_confirmation"]["passed"] is True
    old_trial = learner.practice_trials[-1]["directory"]
    current_trial = next(trial["directory"] for trial in previous["practice_trials"]
                         if trial["method_source"] == current_source)
    learner._execute("run", {"operator": "restore_method", "trial": current_trial}, learner.interface)
    count, costs = len(learner.practice_trials), deepcopy(learner.costs)
    result = finish(learner)
    comparison = result["incumbent_comparison"]
    assert comparison["old_trial"] == old_trial
    assert comparison["selected"] is False and comparison["status"] == "incumbent_practice_failed"
    assert "published" not in result and learner.published_source is learner.model is None
    assert learner.files[runner.METHOD_PATH] == (learner.workspace / runner.METHOD_PATH).read_text() == current_source
    assert result["current_source_coverage"]["missing_cases"] == ["case_4"]
    paid = learner.practice_trials[count:]
    assert len(paid) == 2 and all(trial["case"] == "case_4" and trial["status"] != "completed" for trial in paid)
    assert [trial["method_source"] for trial in paid] == [current_source, old_variant]
    for resource in ("observation_cost", "measurement_cost", "compute_cost"):
        assert learner.costs[resource] == costs[resource] + sum(trial["costs"][resource] for trial in paid)
    # The newly refuted old draft is ineligible on the next comparison; a success
    # on case_5 cannot erase its unresolved case_4 failure.
    assert learner._incumbent_trial(current_source)["method_source"] == previous["practice_trials"][3]["method_source"]


def test_insufficient_remaining_budget_never_publishes_the_partly_supported_old_draft(tmp_path):
    learner, _ = resumed_author(tmp_path)
    current_source = (learner.workspace / runner.METHOD_PATH).read_text()
    learner.config = replace(learner.config, observation_budget=95)
    count, costs = len(learner.practice_trials), deepcopy(learner.costs)
    result = finish(learner)
    assert result["incumbent_comparison"]["selected"] is False
    assert result["incumbent_comparison"]["practices"][0]["error"]
    assert "published" not in result and learner.published_source is learner.model is None
    assert (learner.workspace / runner.METHOD_PATH).read_text() == current_source
    assert result["current_source_coverage"]["missing_cases"] == ["case_4"]
    assert len(learner.practice_trials) == count and learner.costs == costs


def test_later_outcome_updates_same_source_support_without_borrowing_other_source_cases(tmp_path):
    learner, source = learner_with_real_practice(tmp_path, "case_5")
    other = source + "\n# Another candidate.\n"

    def outcome(code, case, passed):
        # Bookkeeping records only: this test makes no numerical-success claim.
        return {"method_source": code, "case": case,
            "directory": f"trials/trial_{len(learner.practice_trials):03d}",
            "status": "completed" if passed else "method_stopped",
            "fresh_confirmation": {"passed": passed}, "selected_prior_predictor": False,
            "selective_revision": {"zero_meter_refit_confirmations": [{}] if passed else [],
                                   "recoveries": []},
            "scoped_revision": {"zero_meter_scoped_recoveries": [{}] if passed else [],
                                "measured_recoveries": []}}

    learner.practice_trials.append(outcome(source, "case_0", True))
    learner.practice_trials.append(outcome(other, "case_1", True))
    assert learner._current_source_coverage()["confirmed_cases"] == ["case_0"]
    assert learner._incumbent_trial(other)["case"] == "case_0"
    learner.practice_trials.append(outcome(source, "case_0", False))
    failed = learner._current_source_coverage()
    assert failed["confirmed_cases"] == []
    assert failed["selective_revision"]["zero_meter_refit_cases"] == []
    assert failed["scoped_revision"]["zero_meter_scoped_recovery_cases"] == []
    assert learner._incumbent_trial(other) is None
    learner.practice_trials.append(outcome(source, "case_0", True))
    recovered = learner._current_source_coverage()
    assert recovered["confirmed_cases"] == ["case_0"]
    assert recovered["selective_revision"]["zero_meter_refit_cases"] == ["case_0"]
    assert learner._incumbent_trial(other) == learner.practice_trials[-1]


def test_numeric_success_without_reuse_compares_old_supported_source_before_publication(tmp_path, archived_predictor):
    learner, _ = make_learner(tmp_path, archived_predictor, source=ADAPTIVE_METHOD)
    supported = learner._execute("run", {"operator": "practice", "case": "case_a"}, learner.interface)
    assert supported["fresh_confirmation"]["passed"] and supported["selected_prior_predictor"]
    change(learner, "write", DEPENDENCY_EDIT_METHOD)
    result = finish(learner)
    assert len(result["publication_practices"]) == 2
    assert all(trial["fresh_confirmation"]["passed"] for trial in result["publication_practices"])
    comparison = result["incumbent_comparison"]
    assert comparison["current_failure_case"] is None
    assert comparison["current_gate_error"]["confirmed_prior_reuse_cases"] == []
    assert comparison["current_gate_error"]["confirmed_revision_cases"] == ["case_a", "case_b"]
    assert comparison["selected"] is True and comparison["status"] == "incumbent_published"
    assert [trial["case"] for trial in comparison["practices"]] == ["case_b"]
    assert result["published"] == runner.METHOD_PATH and learner.published_source == ADAPTIVE_METHOD
    assert result["current_source_coverage"]["prior_reuse_cases"] == ["case_a"]
    assert result["current_source_coverage"]["revised_cases"] == ["case_b"]
    assert len(learner.practice_trials) == 4
