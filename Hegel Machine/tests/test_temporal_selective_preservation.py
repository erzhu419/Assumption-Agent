"""Keep the preservation repair tied to its observed regression and strong control."""

from copy import deepcopy
import json

import pytest

from scripts import run_temporal_failure_selectivity as base
from scripts import run_temporal_selective_revision as selective
from scripts import run_temporal_selective_preservation as runner
from scripts.run_temporal_operator_llm import laboratory


@pytest.fixture
def preceding_phase(tmp_path, monkeypatch):
    scope = tmp_path / "successful_scope_phase"
    scope.mkdir()
    (scope / "protocol.json").write_text(json.dumps(base.protocol()))
    monkeypatch.setattr(selective, "PREVIOUS", scope)
    old = selective.protocol()
    previous = tmp_path / "regressed_selective_phase"
    previous.mkdir()
    (previous / "protocol.json").write_text(json.dumps(old))
    first_failure = next(case for case in old["frozen_cases"]
        if case["prior"] == "first" and case["change"] == "stationary_instrument_only")
    second_failure = next(case for case in old["frozen_cases"]
        if case["prior"] == "second" and case["change"] == "stationary_instrument_only")
    # An earlier listed failure with another donor must not become first-donor practice.
    (previous / "summary.json").write_text(json.dumps({
        "revised_method": {"confirmed_and_correct": 36, "cases": [
            {"case_id": case["case_id"], "prior": case["prior"], "confirmed_and_correct": False}
            for case in (second_failure, first_failure)]}, "new_learning_costs": {"model_calls": 8}}))
    monkeypatch.setattr(runner, "PREVIOUS", previous)
    return previous, old, first_failure


def test_added_practice_keeps_actual_failed_physics_and_common_public_interface(preceding_phase):
    _, old, failed = preceding_phase
    plan = runner.protocol()
    assert {key: plan["practice_cases"][key] for key in old["practice_cases"]} == old["practice_cases"]
    new_task = plan["practice_cases"]["case_e"]
    assert {key: value for key, value in new_task.items() if key != "calibration_change"} == failed["task"]
    original = laboratory(failed["task"])
    practice = laboratory(new_task)
    assert practice.public_spec() == laboratory(plan["practice_cases"]["case_a"]).public_spec()
    action = {"kind": "trajectory", "channels": ["meter"],
              "sequence": [{"u": u} for u in [-1., .4, .9, -.6, .1, 0.] * 4]}
    # Cross the declared checkpoint: this task already starts with the failed
    # calibration, so adding a public checkpoint must introduce no further shift.
    for _ in range(4):
        assert original.execute(deepcopy(action)) == practice.execute(deepcopy(action))
    assert original.costs == practice.costs


def test_preservation_freezes_old_tasks_and_compares_against_successful_ancestor(preceding_phase):
    previous, old, _ = preceding_phase
    plan = runner.protocol()
    assert plan["seed_method"] == str(previous / "study/learning/result.json")
    assert plan["comparison_method"] == old["seed_method"]
    assert plan["seed_method"] != plan["comparison_method"]
    assert plan["regression_case_count"] == 44
    assert plan["frozen_cases"][:44] == old["frozen_cases"]
    added = plan["frozen_cases"][44:]
    assert len(added) == 12
    prior_tasks = [case["task"] for case in old["frozen_cases"]] + list(plan["practice_cases"].values())
    prior_seeds = {value for task in prior_tasks
                   for mapping in (task, task.get("calibration_change", {}))
                   for key, value in mapping.items() if key.endswith("seed")}
    for prior, (original_seed, _) in runner.CALIBRATIONS.items():
        for change in ("response_shift", "instrument_shift", "joint_shift"):
            group = [case for case in added if case["prior"] == prior and case["change"] == change]
            assert len(group) == 2
            unseen = {value for case in group for key, value in case["task"]["calibration_change"].items()
                      if key.endswith("seed") and value != original_seed}
            assert len(unseen) == 2 and unseen.isdisjoint(prior_seeds)
    assert len({case["case_id"] for case in plan["frozen_cases"]}) == 56
