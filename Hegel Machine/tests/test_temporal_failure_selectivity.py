"""Protocol and orchestration checks; scripted outcomes are not learning evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import run_temporal_counterexample_learning as earlier
from scripts import run_temporal_failure_selectivity as runner
from scripts.run_temporal_operator_llm import laboratory


def test_selectivity_keeps_previous_tasks_and_separates_new_shift_calibrations():
    plan = runner.protocol()
    cases, practices = plan["frozen_cases"], plan["practice_cases"]
    assert len(cases) == 32
    assert set(practices) == {"case_a", "case_b", "case_c", "case_d"}
    old_cases = earlier.protocol()["frozen_cases"]
    assert len(old_cases) == 18
    for actual, old in zip(cases[14:], old_cases):
        assert actual["task"] == old["task"]
        assert actual["prior"] == old["prior"]
        assert actual["unchanged"] == old["unchanged"]
        assert actual["change"] == "stationary_" + old["change"]
    assert [case["case_id"] for case in cases] == [f"episode_{i:02d}" for i in range(32)]

    practice_seeds = {value for task in practices.values()
                      for key, value in task["calibration_change"].items()
                      if key.endswith("seed")}
    for prior, (initial_seed, _) in runner.CALIBRATIONS.items():
        cohort = [case for case in cases[:14] if case["prior"] == prior]
        assert len(cohort) == 7
        assert sum(case["change"] == "stable_checkpoint" for case in cohort) == 1
        for change in ("response_shift", "instrument_shift", "joint_shift"):
            shifted = [case for case in cohort if case["change"] == change]
            assert len(shifted) == 2
            post_seeds = set()
            for case in shifted:
                task, post = case["task"], case["task"]["calibration_change"]
                assert task["apparatus_seed"] == initial_seed
                assert post["after_rows"] == 72
                for component, changes in (("apparatus_seed", {"instrument_shift", "joint_shift"}),
                                           ("response_seed", {"response_shift", "joint_shift"})):
                    if change in changes:
                        assert post[component] not in practice_seeds
                        post_seeds.add(post[component])
                    else:
                        assert post[component] == initial_seed
                assert task["confirmation_seed"] != task["evaluation_seed"]
            assert len(post_seeds) == 2


def test_practice_causes_share_public_checkpoint_and_identical_paid_prefix():
    practices = runner.protocol()["practice_cases"]
    labs = [laboratory(task) for task in practices.values()]
    public = labs[0].public_spec()
    assert public["settled_after_observation_rows"] == 72
    assert all(lab.public_spec() == public for lab in labs)
    public_text = json.dumps(public)
    assert "apparatus_seed" not in public_text and "response_seed" not in public_text
    assert "instrument_shift" not in public_text and "response_shift" not in public_text
    assert "joint_shift" not in public_text
    action = {"kind": "trajectory", "channels": ["meter"],
              "sequence": [{"u": u} for u in [-1., .4, 0., .8, -.7, .1] * 4]}
    records = [[lab.execute(deepcopy(action)) for _ in range(3)] for lab in labs]
    assert all(prefix == records[0] for prefix in records)
    assert all(lab.costs["observation_cost"] == 72 for lab in labs)
    assert all(lab.costs["measurement_cost"] == 216 for lab in labs)


@pytest.mark.parametrize("baseline_passes", [False, True])
def test_preflight_freezes_plan_and_only_actual_failure_enters_authoring(
        tmp_path, monkeypatch, baseline_passes):
    output = tmp_path / "experiment"
    calls = []
    costs = {"observation_cost": 240, "measurement_cost": 216, "compute_cost": 19872,
             "python_execution_seconds": .25, "model_calls": 0, "reported_tokens": 0}
    provider_factory = object()

    def snapshot(path):
        (Path(path) / "source_at_run/scripts").mkdir(parents=True)

    def preflight(seed, path, challenge):
        saved = json.loads((output / "protocol.json").read_text())
        assert saved["seed_method"] == seed
        assert saved["provider"]["auth_mode"] == "scripted-test"
        assert len(saved["frozen_cases"]) == 32
        assert challenge["frozen_cases"] == [{"case_id": "episode_00", "prior": "first",
            "change": "instrument_shift", "unchanged": False,
            "task": saved["practice_cases"]["case_c"]}]
        assert (output / "source_at_run/scripts/run_temporal_failure_selectivity.py").is_file()
        trace = Path(saved["counterexample"])
        assert trace == (path / "episode_00/training.json").resolve()
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({"tasks": [{"status": "completed" if baseline_passes
                                                else "confirmation_budget_exhausted",
            "events": [{"feedback": {"fresh_confirmation": {"passed": baseline_passes}}}],
            "costs": costs}], "evaluation": None}))
        calls.append("preflight")
        return {"confirmed_and_correct": int(baseline_passes), "costs": costs}

    def author(path, *, provider_factory, auth_mode, study_protocol):
        assert not baseline_passes
        assert calls == ["preflight"] and path == output / "study"
        assert provider_factory is factory and auth_mode == "scripted-test"
        assert study_protocol == json.loads((output / "protocol.json").read_text())
        trace = json.loads(Path(study_protocol["counterexample"]).read_text())
        assert trace["tasks"][0]["events"][0]["feedback"]["fresh_confirmation"]["passed"] is False
        assert trace["tasks"][0]["costs"] == costs
        calls.append("author")
        return {"status": "method_not_published"}

    factory = provider_factory
    monkeypatch.setattr(runner, "snapshot", snapshot)
    monkeypatch.setattr(runner, "run_frozen", preflight)
    monkeypatch.setattr(runner, "run_study", author)
    summary = runner.run_experiment(output, provider_factory=provider_factory,
                                    auth_mode="scripted-test")
    assert summary == json.loads((output / "summary.json").read_text())
    if baseline_passes:
        assert calls == ["preflight"] and summary["status"] == "no_counterexample"
        assert "new_counterexample_costs" not in summary
    else:
        assert calls == ["preflight", "author"]
        assert summary["new_counterexample_costs"] == costs
