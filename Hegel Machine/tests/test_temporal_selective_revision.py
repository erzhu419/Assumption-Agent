"""Keep selective-repair authoring tied to published evidence and new calibrations."""

import json
from pathlib import Path

import pytest

from scripts import run_temporal_failure_selectivity as preceding
from scripts import run_temporal_selective_revision as runner


@pytest.fixture
def previous_phase(tmp_path, monkeypatch):
    previous = tmp_path / "published_phase"
    previous.mkdir()
    plan = preceding.protocol()
    # Preserve arbitrary existing metadata as well as the actual task settings.
    plan["frozen_cases"][0]["existing_note"] = "keep this control unchanged"
    (previous / "protocol.json").write_text(json.dumps(plan))
    monkeypatch.setattr(runner, "PREVIOUS", previous)
    return previous, plan


def test_new_calibration_block_preserves_all_previous_controls(previous_phase):
    previous, old = previous_phase
    plan = runner.protocol()
    controls = plan["frozen_cases"][:plan["regression_case_count"]]
    added = plan["frozen_cases"][plan["regression_case_count"]:]
    assert controls == old["frozen_cases"]
    assert len(controls) == 32 and len(added) == 12
    assert plan["practice_cases"] == old["practice_cases"]
    assert Path(plan["seed_method"]) == previous / "study/learning/result.json"
    assert plan["seed_method"] != old["seed_method"]

    old_tasks = [case["task"] for case in controls] + list(old["practice_cases"].values())
    old_seeds = {value for task in old_tasks
                 for mapping in (task, task.get("calibration_change", {}))
                 for key, value in mapping.items() if key.endswith("seed")}
    old_tasks_text = {json.dumps(task, sort_keys=True) for task in old_tasks}
    for prior, (prior_seed, _) in runner.CALIBRATIONS.items():
        for change in ("response_shift", "instrument_shift", "joint_shift"):
            group = [case for case in added if case["prior"] == prior and case["change"] == change]
            assert len(group) == 2
            new_seeds = set()
            for case in group:
                task = case["task"]
                assert json.dumps(task, sort_keys=True) not in old_tasks_text
                post = task["calibration_change"]
                for key in ("apparatus_seed", "response_seed"):
                    value = post[key]
                    if value != prior_seed:
                        assert value not in old_seeds
                        new_seeds.add(value)
            assert len(new_seeds) == 2
    assert len({case["case_id"] for case in plan["frozen_cases"]}) == 44


def test_author_uses_current_published_recovery_after_protocol_freeze(
        tmp_path, monkeypatch, previous_phase):
    previous, _ = previous_phase
    learning = previous / "study/learning"
    selected = learning / "trials/current_recovery/result.json"
    selected.parent.mkdir(parents=True)
    source = "published selective-repair starting source"
    recovery = {"status": "completed", "method_source": source,
                "paid_observations": [{"row_indices": [120, 121]}],
                "method_results": [{"feedback": {"fresh_confirmation": {"passed": False}}},
                                   {"feedback": {"fresh_confirmation": {"passed": True}}}]}
    selected.write_text(json.dumps(recovery))
    # None of these other trials has the required successful current-source case_b.
    trials = [
        {"case": "case_b", "status": "completed", "method_source": "older source", "directory": "old"},
        {"case": "case_b", "status": "confirmation_budget_exhausted", "method_source": source,
         "directory": "failed"},
        {"case": "case_a", "status": "completed", "method_source": source, "directory": "stable"},
        {"case": "case_b", "status": "completed", "method_source": source,
         "directory": "trials/current_recovery"}]
    (learning / "result.json").write_text(json.dumps(
        {"status": "completed", "method_source": source, "practice_trials": trials}))
    output = tmp_path / "experiment"
    observed = []

    def provider_factory(path):
        saved = json.loads((output / "protocol.json").read_text())
        assert len(saved["frozen_cases"]) == 44
        assert Path(saved["supplied_trace_origin"]) == selected
        assert json.loads(Path(saved["counterexample"]).read_text()) == {
            "tasks": [recovery], "evaluation": None}
        observed.append(path)
        return object()

    def author(path, *, provider_factory, auth_mode, study_protocol, learner_class):
        assert path == output / "study"
        assert auth_mode == "scripted-test"
        assert learner_class is runner.SelectiveMethodLearner
        assert study_protocol == json.loads((output / "protocol.json").read_text())
        assert study_protocol["counterexample_case"] == "case_b"
        assert json.loads(Path(study_protocol["seed_method"]).read_text())["method_source"] == source
        assert (output / "source_at_run/scripts/run_temporal_selective_revision.py").is_file()
        provider_factory(path / "learning")
        return {"status": "method_not_published"}

    monkeypatch.setattr(runner, "run_study", author)
    summary = runner.run_experiment(output, provider_factory=provider_factory, auth_mode="scripted-test")
    assert observed == [output / "study/learning"]
    assert summary == json.loads((output / "summary.json").read_text())
    assert "selective_transfer" not in summary
