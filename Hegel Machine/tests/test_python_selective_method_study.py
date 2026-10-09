"""Keep the four-arm comparison conditional on actual method publication."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import run_python_selective_method_study as runner
from scripts.run_temporal_counterexample_learning import supply_counterexample


def sources():
    return [{"case": {"apparatus_seed": 971106001 + i * 1009,
                      "changed_response_seed": 981106001 + i * 1009,
                      "seed": 991106001 + i * 1009},
             "candidate": None if i == 0 else {"candidate_id": "candidate_000"}}
            for i in range(4)]


def methods():
    return {"seed_selective": "unchanged selective source",
            "prior_published": "unchanged published source", "fixed": "fixed source"}


def test_protocol_registers_four_arms_eight_targets_and_bounded_author(tmp_path):
    plan = runner.study_protocol(sources(), tmp_path / "diagnostic")
    assert tuple(plan["conditions"]) == runner.CONDITIONS
    assert len(plan["transfer_cases"]) == 8
    assert len(plan["practice_cases"]) == 4
    assert plan["author_limits"]["max_model_calls"] == 12
    assert plan["author_limits"]["token_budget"] == 500000
    assert plan["limits"]["max_model_calls"] == plan["limits"]["token_budget"] == 0
    assert plan["practice_cases"]["case_3"]["task"]["calibration_change"]["after_rows"] == 96


@pytest.mark.parametrize("status", ["token_budget_exhausted", "offline_candidate_replay_finished"])
def test_unpublished_author_keeps_all_reference_slots_even_with_replayed_source(tmp_path, status):
    inputs = sources()
    plan = runner.study_protocol(inputs, tmp_path / "diagnostic")
    learned = {"status": status, "method_source": "not actually published"}
    jobs, skipped = runner.target_jobs(inputs, tmp_path, plan, learned, methods())
    assert len(jobs) == 24 and len(skipped) == 8
    assert {job["condition"] for job in jobs} == {"seed_selective", "prior_published", "fixed"}
    assert all(item["condition"] == "revised" and item["reason"] == "method_not_published"
               for item in skipped)
    for condition in methods():
        group = [job for job in jobs if job["condition"] == condition]
        assert len(group) == 8 and all(job["method_source"] == methods()[condition] for job in group)


def test_publication_unlocks_exact_source_and_rotates_matched_conditions(tmp_path):
    inputs = sources()
    plan = runner.study_protocol(inputs, tmp_path / "diagnostic")
    original = deepcopy(plan)
    published_source = "def select_actions(spec, records, memory):\n    return {'stop':'literal'}\n"
    jobs, skipped = runner.target_jobs(inputs, tmp_path, plan,
        {"status": "completed", "method_source": published_source}, methods())
    assert not skipped and len(jobs) == 32 and len({job["path"] for job in jobs}) == 32
    assert plan == original
    for index, spec in enumerate(plan["transfer_cases"]):
        group = jobs[index * 4:(index + 1) * 4]
        order = runner.CONDITIONS[index % 4:] + runner.CONDITIONS[:index % 4]
        assert tuple(job["condition"] for job in group) == order
        assert all(job["spec"] == spec and job["source"] == inputs[spec["source_index"]] for job in group)
        assert next(job for job in group if job["condition"] == "revised")["method_source"] == published_source
        assert {job["condition"]: job["method_source"] for job in group if job["condition"] != "revised"} == methods()


def test_wrong_case_counterexample_is_rejected_before_author_or_files(tmp_path, monkeypatch):
    inputs = sources()
    outcome = tmp_path / "sources.json"
    outcome.write_text(json.dumps({"sources": inputs}))
    monkeypatch.setattr(runner, "SOURCE_OUTCOMES", outcome)
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    spec = deepcopy(runner.study_protocol(inputs, diagnostic)["practice_cases"]["case_3"])
    spec["confirmation_seed"] += 1
    (diagnostic / "protocol.json").write_text(json.dumps({"diagnostic_case": spec}))

    def no_author(path):
        raise AssertionError("a mismatched diagnostic must not start authoring")

    output = tmp_path / "study"
    with pytest.raises(ValueError, match="exact practice case_3"):
        runner.run_study(output, diagnostic, provider_factory=no_author)
    assert not output.exists()


def test_paid_counterexample_preserves_real_refit_validation_then_fresh_rejection(tmp_path):
    # Numerical/event excerpt from python_selective_method_diagnostic_20261006/selective/training.json.
    # Keep it in this test so regression coverage does not require untracked run artifacts.
    prior_path = "shared/confirmed_models/candidate_000/_evaluated_model/model.py"
    refit_path = "shared/refits/revision/_evaluated_model/model.py"
    prior = {"status": "method_step_limit", "costs": {"observation_cost": 336, "measurement_cost": 0},
        "paid_observations": [{"row_indices": list(range(start, start + 24)), "measurements": []}
                              for start in range(0, 336, 24)],
        "events": [
            {"action": "query", "feedback": {"method_submission": prior_path,
                "fresh_confirmation": {"passed": False, "mse": 0.6609043837984635,
                    "rows": 24, "observation_indices": list(range(72, 96))}}},
            {"action": "query", "feedback": {"operator": "refit", "path": refit_path,
                "source_model": prior_path, "validation": {"returncode": 0,
                    "fit_available": True, "mse": 5.2440845058968494e-30},
                "final_fitting_rows": 96, "paid_rows_at_refit": 96,
                "split": {"records": [0, 1, 2, 3], "training_records": [0, 1],
                    "validation_records": [2, 3], "train_rows": 48, "holdout_rows": 48}}},
            {"action": "query", "feedback": {"method_submission": refit_path,
                "fresh_confirmation": {"passed": False, "mse": 0.44702558551277183,
                    "rows": 24, "observation_indices": list(range(96, 120))}}}],
        "task": {"response_seed": "private_physics_marker"},
        "terminal": {"mse": "terminal_outcome_marker"}}
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"tasks": [prior]}))
    learner = SimpleNamespace(workspace=tmp_path, files={}, common_interface="")
    assert supply_counterexample(learner, path, case_label="case_3") == prior["costs"]
    trace = json.loads(learner.files["counterexample.json"])
    assert trace["practice_case"] == "case_3"
    refit = trace["steps"][1]["feedback"]["operators"][0]
    assert refit["source_model"] == prior_path and refit["path"] == refit_path
    assert refit["validation"]["mse"] == 5.2440845058968494e-30
    assert refit["final_fitting_rows"] == refit["paid_rows_at_refit"] == 96
    assert refit["training_records"] == [0, 1] and refit["validation_records"] == [2, 3]
    rejected = trace["steps"][2]["feedback"]
    assert rejected["method_submission"] == refit_path
    assert rejected["fresh_confirmation"] == {"passed": False, "mse": 0.44702558551277183,
        "rows": 24, "row_interval": [96, 119]}
    for filename in ("counterexample.json", "counterexample_trace.json", "counterexample_observations.json"):
        shown = (tmp_path / filename).read_text()
        assert "private_physics_marker" not in shown and "terminal_outcome_marker" not in shown


def interrupted_fixture(tmp_path):
    root = tmp_path / "interrupted"
    learning = root / "learning"
    learning.mkdir(parents=True)
    plan = runner.study_protocol(sources(), tmp_path / "diagnostic")
    costs = runner.total_costs([{"model_calls": 1, "reported_tokens": None}])
    previous = {"status": "provider_failed", "method_source": None, "costs": costs,
        "events": [{"action": "provider_error", "usage": {"total_tokens": None}}],
        "practice_trials": [], "outgoing_knowledge": {"files": {runner.METHOD_PATH: "old source"}}}
    reference_cost = runner.total_costs([{"observation_cost": 96, "compute_cost": 168}])
    references = [{"target_id": spec["id"], "kind": spec["kind"], "condition": condition,
        "status": "completed", "confirmed_and_correct": True, "false_confirmation": False,
        "unchanged_prior_selected": True, "costs": reference_cost}
        for spec in plan["transfer_cases"] for condition in ("seed_selective", "prior_published", "fixed")]
    diagnostic_cost = runner.total_costs([{"observation_cost": 528, "measurement_cost": 216}])
    total = runner.total_costs([costs, diagnostic_cost, *[item["costs"] for item in references]])
    summary = {"results": references, "cumulative_author_costs": costs,
        "cumulative_author_usage": {"known_tokens": 0, "input_tokens": 0,
            "cached_input_tokens": 0, "missing_usage_calls": 1}, "all_method_development_costs": total}
    for path, value in ((root / "protocol.json", plan), (root / "summary.json", summary),
                        (learning / "result.json", previous)):
        path.write_text(json.dumps(value))
    return learning / "result.json", plan, previous, summary, diagnostic_cost


def test_provider_interruption_deducts_calls_and_known_usage_without_resetting_unknown(tmp_path):
    path, original, previous, summary, _ = interrupted_fixture(tmp_path)
    plan, restored, prior = runner.interrupted_state(path)
    assert plan["author_limits"] == {**original["author_limits"], "max_model_calls": 11}
    assert plan["author_limits"]["token_budget"] == 500000
    assert plan["resume_author"]["previous_costs"]["reported_tokens"] is None
    assert restored == previous and prior == summary
    assert json.loads(path.read_text()) == previous
    previous["status"] = "token_budget_exhausted"
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="provider-failed"):
        runner.interrupted_state(path)


def test_resume_reuses_all_reference_work_and_charges_only_new_author(tmp_path, monkeypatch):
    path, plan, previous, prior, diagnostic_cost = interrupted_fixture(tmp_path)
    inputs_path = tmp_path / "sources.json"
    inputs_path.write_text(json.dumps({"sources": sources()}))
    monkeypatch.setattr(runner, "SOURCE_OUTCOMES", inputs_path)
    old = tmp_path / "old.json"
    old.write_text(json.dumps(previous))
    monkeypatch.setattr(runner, "OLD_AUTHOR_RESULT", old)
    published = tmp_path / "published/learning/result.json"
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps({"method_source": "published source"}))
    (published.parent.parent / "analysis.json").write_text(json.dumps({"accounting": {
        "separate_existing_source_acquisition": runner.total_costs([])}}))
    monkeypatch.setattr(runner, "PUBLISHED_AUTHOR_RESULT", published)
    diagnostic = tmp_path / "diagnostic"
    (diagnostic / "selective").mkdir(parents=True)
    (diagnostic / "protocol.json").write_text(json.dumps({"diagnostic_case": plan["practice_cases"]["case_3"]}))
    (diagnostic / "summary.json").write_text(json.dumps({"costs": diagnostic_cost}))
    (diagnostic / "selective/training.json").write_text(json.dumps({
        "status": "method_step_limit", "costs": diagnostic_cost, "events": [], "paid_observations": []}))
    monkeypatch.setattr(runner, "archive_memory", lambda source: {"files": {}})
    monkeypatch.setattr(runner.shutil, "copytree", lambda source, destination, **kwargs: None)
    restored = []
    new_costs = runner.total_costs([{"model_calls": 1, "reported_tokens": 20}])

    class InterruptedAgain:
        def __init__(self, provider, workspace, **kwargs):
            self.workspace, self.files, self.common_interface = workspace, {}, ""
            workspace.mkdir(parents=True)
            assert kwargs["config"].max_model_calls == 11

        def restore_practice(self, value):
            restored.append(value)
            return {"restored_trials": 0}

        def run(self, **kwargs):
            return {"status": "model_call_budget_exhausted", "method_source": None,
                "costs": new_costs, "events": [{"usage": {"total_tokens": 20,
                    "input_tokens": 18, "cached_input_tokens": 10}}]}

    def no_reference_execution(*args, **kwargs):
        raise AssertionError("completed reference acquisition must not run again")

    monkeypatch.setattr(runner, "SelectivePythonMethodLearner", InterruptedAgain)
    monkeypatch.setattr(runner, "trial_preparer", lambda *args: no_reference_execution)
    result = runner.run_study(tmp_path / "resumed", diagnostic,
        provider_factory=lambda _: None, resume_author=path)
    assert restored == [previous]
    assert result["reused_reference_episodes"] == 24 and result["results"] == prior["results"]
    assert len(result["skipped"]) == 8
    assert result["new_diagnostic_costs"] == runner.total_costs([])
    assert result["new_author_and_transfer_costs"] == result["new_work_including_diagnostic"] == new_costs
    assert result["all_method_development_costs"] == runner.total_costs([prior["all_method_development_costs"], new_costs])
    assert result["cumulative_author_costs"]["model_calls"] == 2
    assert result["cumulative_author_costs"]["reported_tokens"] is None
    assert result["cumulative_author_usage"] == {"known_tokens": 20, "input_tokens": 18,
        "cached_input_tokens": 10, "missing_usage_calls": 1}


def test_extra_development_adds_only_declared_call_token_allowance_after_actual_usage(tmp_path):
    path, original, previous, summary, _ = interrupted_fixture(tmp_path)
    with pytest.raises(ValueError, match="budget-exhausted"):
        runner.interrupted_state(path, additional_development=True)
    original["author_limits"]["max_model_calls"] = 11
    original["resume_author"] = {"result": "earlier actual provider interruption"}
    (path.parent.parent / "protocol.json").write_text(json.dumps(original))
    previous.update(status="model_call_budget_exhausted",
        costs=runner.total_costs([{"model_calls": 11, "reported_tokens": 345000,
            "observation_cost": 400, "compute_cost": 8000, "python_execution_seconds": 2.5}]),
        events=[{"usage": {"total_tokens": 345000, "input_tokens": 340000, "cached_input_tokens": 50000}}],
        practice_trials=[{"case": "case_3", "status": "method_step_limit"}])
    path.write_text(json.dumps(previous))
    plan, restored, parent = runner.interrupted_state(path, additional_development=True)
    assert plan["author_limits"] == {**original["author_limits"], "max_model_calls": 6,
        "token_budget": 355000, "observation_budget": 2600, "compute_budget": 2992000,
        "execution_seconds": 237.5}
    assert "resume_author" not in plan
    assert plan["continue_author"]["added_allowance"] == {"model_calls": 6, "tokens": 200000}
    assert restored == previous and json.loads(path.read_text()) == previous
    assert plan["practice_cases"] == original["practice_cases"]
    assert plan["transfer_cases"] == original["transfer_cases"]
    assert parent == summary and len(parent["results"]) == 24
    assert parent["cumulative_author_costs"]["reported_tokens"] is None
    assert parent["cumulative_author_usage"]["missing_usage_calls"] == 1
