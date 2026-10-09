"""A late paid failure must require method revision without resetting old evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from scripts import run_python_method_reopening_study as runner


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "artifacts/python_late_response_diagnostic_20261006/training.json"


def study_inputs():
    sources = json.loads(runner.SOURCE_OUTCOMES.read_text())["sources"]
    reference_plan = json.loads((runner.BRIDGE / "protocol.json").read_text())
    reference = json.loads((runner.BRIDGE / "summary.json").read_text())
    source = (runner.BRIDGE / "historical_bridge.py").read_text()
    return sources, reference_plan, reference, source


def test_registered_extension_preserves_five_practices_ten_targets_and_episode_limits():
    sources, original, _, _ = study_inputs()
    plan = runner.study_protocol(sources)
    assert {key: plan["practice_cases"][key] for key in original["practice_cases"]} == original["practice_cases"]
    assert set(plan["practice_cases"]) == {*original["practice_cases"], "case_5"}
    assert plan["transfer_cases"][:10] == original["transfer_cases"]
    extra = plan["transfer_cases"][10:]
    assert [spec["id"] for spec in extra] == ["target_0_5", "target_1_5"]
    for spec in [plan["practice_cases"]["case_5"], *extra]:
        assert spec["kind"] == runner.KIND
        assert spec["task"]["instrument_mode"] == "recurrent"
        assert spec["task"]["calibration_change"]["after_rows"] == 167
    for field in ("limits", "procedure_step_limit", "evaluation"):
        assert plan[field] == original[field]
    replicas = plan["author_replicas"]
    assert [item["id"] for item in replicas] == ["author_0", "author_1", "author_2"]
    for case, spec in plan["practice_cases"].items():
        practice = [item["practice_cases"][case] for item in replicas]
        assert len({item["confirmation_seed"] for item in practice}) == 3
        assert all({key: value for key, value in item.items() if key != "confirmation_seed"}
                   == {key: value for key, value in spec.items() if key != "confirmation_seed"}
                   for item in practice)


def test_existing_paid_trace_demonstrates_response_validation_then_rejected_confirmation():
    trial = json.loads(DIAGNOSTIC.read_text())
    assert trial["status"] == "method_stopped"
    response_index, response = next((index, step) for index, step in enumerate(trial["method_results"])
                                   if step["feedback"].get("operator") == "response")
    candidate = response["feedback"]["candidates"][0]
    assert candidate["validation_mse"] < 1e-10
    confirmation = trial["method_results"][response_index + 1]
    assert confirmation["proposal"]["model"] == candidate["path"]
    assert confirmation["feedback"]["fresh_confirmation"]["passed"] is False
    assert confirmation["feedback"]["fresh_confirmation"]["mse"] > .6
    assert confirmation["feedback"]["fresh_confirmation"]["observation_indices"][0] == 167
    assert "stop" in trial["method_results"][-1]["proposal"]
    assert trial["costs"]["model_calls"] == 0


def test_ten_existing_control_outcomes_are_reused_without_mutating_the_archive():
    sources, original, reference, _ = study_inputs()
    saved = deepcopy(reference)
    reused = runner.reusable_controls(runner.study_protocol(sources), original, reference)
    assert len(reused) == 10
    assert {row["target_id"] for row in reused} == {spec["id"] for spec in original["transfer_cases"]}
    assert all(row["confirmed_and_correct"] for row in reused)
    reused[0]["costs"]["observation_cost"] = -1
    assert reference == saved
    changed = runner.study_protocol(sources)
    changed["limits"]["observation_budget"] += 1
    with pytest.raises(ValueError, match="identical target tasks"):
        runner.reusable_controls(changed, original, reference)


def test_author_receives_only_prior_source_code_and_paid_failure_feedback(tmp_path):
    sources, _, _, source = study_inputs()
    plan = runner.study_protocol(sources)
    prior = json.loads(DIAGNOSTIC.read_text())
    prior["terminal"] = "unseen_terminal_result"
    prior["task"] = {"mechanism": "private_response_parameters"}
    evidence = tmp_path / "prior.json"
    evidence.write_text(json.dumps({"tasks": [prior]}))
    learner = runner.prepare_author(None, tmp_path / "author", plan,
        plan["author_replicas"][0], sources, source, evidence)
    assert (learner.workspace / runner.METHOD_PATH).read_text() == source
    visible = json.loads((learner.workspace / "counterexample.json").read_text())
    assert visible["practice_case"] == "case_5"
    assert visible["status"] == "method_stopped"
    for name in ("counterexample.json", "counterexample_trace.json", "counterexample_observations.json"):
        text = (learner.workspace / name).read_text()
        assert "unseen_terminal_result" not in text
        assert "private_response_parameters" not in text
    assert learner.practice_trials == []
    assert set(learner._current_source_coverage()["missing_cases"]) == set(plan["practice_cases"])


def test_publication_failure_skips_twelve_and_changed_publication_schedules_all_twelve(tmp_path):
    sources, _, _, source = study_inputs()
    plan = runner.study_protocol(sources)
    jobs, skipped = runner.target_jobs(plan, "author_0",
        {"status": "provider_failed", "method_source": source + "\n# unpublished edit\n"},
        source, tmp_path)
    assert jobs == [] and len(skipped) == 12
    assert {row["target_id"] for row in skipped} == {spec["id"] for spec in plan["transfer_cases"]}
    changed = source + "\n# published fixture only; no numerical success is claimed\n"
    jobs, skipped = runner.target_jobs(plan, "author_1",
        {"status": "completed", "method_source": changed}, source, tmp_path)
    assert skipped == [] and len(jobs) == 12
    assert [job["spec"] for job in jobs] == plan["transfer_cases"]
    assert all(job["method_source"] == changed and job["condition"] == "revised" for job in jobs)
    assert all(job["path"] == tmp_path / "author_1" / job["spec"]["id"] for job in jobs)


def test_failed_authors_execute_only_two_new_controls_and_charge_no_reused_trials(tmp_path):
    seen = []
    output = tmp_path / "study"

    class FailedProvider:
        def __init__(self, workspace):
            self.workspace = Path(workspace)

        def next_action(self, request):
            seen.append(self.workspace)
            # The new controls ran, but terminal results have not been computed.
            assert len(list((output / "unchanged").glob("*/training.json"))) == 2
            assert not list(output.rglob("evaluation.json"))
            assert "counterexample.json" in request["files"]
            return {"action": "provider_error", "arguments": {},
                "error": {"message": "offline fixture supplies no method revision"},
                "usage": {"total_tokens": 0, "input_tokens": 0,
                          "output_tokens": 0, "cached_input_tokens": 0}}

    result = runner.run_study(output, provider_factory=FailedProvider,
                              auth_mode="offline_scripted_provider")
    assert len(set(seen)) == len(seen) == result["registered_authors"] == 3
    assert result["published_authors"] == result["changed_published_authors"] == 0
    assert result["reused_control_count"] == 10
    assert result["new_control_count"] == 2
    assert result["new_revised_target_count"] == 0
    assert result["skipped_target_count"] == 36
    assert all(len(author["skipped"]) == 12 and author["results"] == []
               and author["practice_trials"] == 0 for author in result["authors"])
    controls = result["unchanged_control"]["results"]
    assert len(controls) == 12
    new = [row for row in controls if row["evidence_kind"] == "new_frozen_execution"]
    assert {row["target_id"] for row in new} == {"target_0_5", "target_1_5"}
    for row in new:
        trial = json.loads((Path(row["workspace"]) / "training.json").read_text())
        assert trial["costs"]["model_calls"] == 0
        assert trial["method_source"] == (runner.BRIDGE / "historical_bridge.py").read_text()
    assert result["new_target_and_control_costs"] == runner.total_costs([row["costs"] for row in new])
    assert result["new_work_costs"] == runner.total_costs([
        *[author["author_costs"] for author in result["authors"]],
        *[row["costs"] for row in new]])
    assert result["new_work_costs"]["reported_tokens"] == 0


def test_edit_practice_comparison_reuses_all_twelve_controls_with_matched_author_budgets(tmp_path, monkeypatch):
    from scripts.python_method_edit_practice import EditPracticeMethodLearner

    baseline_plan = json.loads((runner.BASELINE / "protocol.json").read_text())
    baseline_path = runner.BASELINE / "summary.json"
    original_summary = baseline_path.read_bytes()
    baseline = json.loads(original_summary)
    seed = (runner.BASELINE / "unchanged_method.py").read_text()
    prepared = []
    original_prepare = runner.prepare_author

    def prepare(*args, **kwargs):
        learner = original_prepare(*args, **kwargs)
        assert isinstance(learner, EditPracticeMethodLearner)
        prepared.append(learner)
        return learner

    monkeypatch.setattr(runner, "prepare_author", prepare)

    class FailedProvider:
        def __init__(self, workspace):
            self.workspace = workspace

        def next_action(self, request):
            assert request["files"][runner.METHOD_PATH] == seed
            return {"action": "provider_error", "arguments": {},
                "error": {"message": "offline fixture leaves the source unchanged"},
                "usage": {"total_tokens": 0, "input_tokens": 0,
                          "output_tokens": 0, "cached_input_tokens": 0}}

    output = tmp_path / "edit_comparison"
    result = runner.run_study(output, provider_factory=FailedProvider,
        auth_mode="offline_scripted_provider", edit_practice=True)
    plan = json.loads((output / "protocol.json").read_text())
    assert len(prepared) == 3
    for field in ("author_limits", "author_replicas", "practice_cases", "transfer_cases",
                  "limits", "procedure_step_limit", "evaluation"):
        assert plan[field] == baseline_plan[field]
    for learner, replica in zip(prepared, baseline_plan["author_replicas"]):
        assert asdict(learner.config) == baseline_plan["author_limits"]
        assert learner.cases == replica["practice_cases"]
        assert (learner.workspace / runner.METHOD_PATH).read_text() == seed
    assert result["reused_control_count"] == 12
    assert result["new_control_count"] == result["new_revised_target_count"] == 0
    assert result["skipped_target_count"] == 36
    assert not list(output.rglob("training.json"))
    assert result["new_target_and_control_costs"] == runner.total_costs([])
    assert result["new_work_costs"] == result["new_author_costs"]
    assert result["new_work_costs"]["observation_cost"] == 0
    assert result["new_work_costs"]["reported_tokens"] == 0
    for current, prior in zip(result["unchanged_control"]["results"], baseline["unchanged_control"]["results"]):
        assert current["target_id"] == prior["target_id"]
        assert current["costs"] == prior["costs"]
        assert current["confirmed_and_correct"] == prior["confirmed_and_correct"]
        assert current["evidence_kind"] == "reused_frozen_execution"
    assert baseline_path.read_bytes() == original_summary
