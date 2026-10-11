"""Preserve real prior inputs, paid-prefix accounting and unpublished outcomes."""

from copy import deepcopy
import json
import pytest

from scripts.run_python_method_learning_study import (
    CONTINUATION_ALLOWANCE, archive_memory, continue_author_state, laboratory, protocol,
    resume_author_state, summarize, target_jobs, trial_preparer,
)
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import PythonArchiveMethodLearner, SEED_PYTHON_METHOD
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_method import DEFAULT_RESEARCH_PROCEDURE


def sources_for(tmp_path):
    sources = []
    for index in range(4):
        case = {"apparatus_seed": 971106001 + 1009 * index,
                "changed_response_seed": 981106001 + 1009 * index,
                "seed": 991106001 + 1009 * index}
        candidate = None if index == 0 else {"candidate_id": "candidate_000", "training_rows": 72,
            "snapshot_workspace": str(tmp_path / f"archive_{index}"), "inner_model": "model.py",
            "model": "shared/confirmed_models/candidate_000/_evaluated_model/model.py"}
        sources.append({"case": case, "candidate": candidate,
            "status": "completed" if candidate else "model_call_budget_exhausted",
            "costs": {"model_calls": 3 if candidate else 12, "reported_tokens": 100,
                      "observation_cost": 96}})
    return sources


def test_callback_preserves_archive_training_view_and_charges_only_actual_prefix(tmp_path):
    sources = sources_for(tmp_path)
    source = sources[1]
    archive = tmp_path / "archive_1"
    archive.mkdir()
    code = "import support\ndef predict(row):\n    return support.OFFSET\n"
    (archive / "model.py").write_text(code)
    (archive / "support.py").write_text("OFFSET = 0.0\n")
    old_rows = [{"features": {"u": .1}, "target": 17.}]
    (archive / "model_observations.json").write_text(json.dumps(old_rows))
    plan = protocol(sources)
    memory = archive_memory(source)
    assert memory["files"][METHOD_PATH] == SEED_PYTHON_METHOD
    assert DEFAULT_RESEARCH_PROCEDURE not in memory["files"].values()
    assert memory["research_roles"] == {"predictor": source["candidate"]["model"]}
    assert not (tmp_path / "archive_2").exists() and not (tmp_path / "archive_3").exists()
    assert plan["author_limits"]["max_model_calls"] == 24
    assert plan["author_limits"]["confirmation_rows"] == 0
    limits = InquiryLLMConfig(observation_budget=144, compute_budget=12345,
        confirmation_rows=24, execution_seconds=40, max_model_calls=0, token_budget=0)
    spec = plan["practice_cases"]["case_0"]
    payload = trial_preparer(source, plan)(spec, tmp_path / "trial", SEED_PYTHON_METHOD, limits)
    agent = payload["agent"]
    assert len(agent.observations) == 3 and len(agent.rows) == 72
    assert agent.costs["observation_cost"] == 72 and agent.costs["measurement_cost"] == 0
    assert agent.costs["model_calls"] == agent.costs["reported_tokens"] == 0
    assert agent.config == limits and agent.frozen_method_source == SEED_PYTHON_METHOD
    assert agent.knowledge["research_roles"] == memory["research_roles"]
    copied = agent.workspace / source["candidate"]["model"]
    assert copied.read_text() == code
    assert (copied.parent / "support.py").read_text() == "OFFSET = 0.0\n"
    assert json.loads((copied.parent / "model_observations.json").read_text()) == old_rows
    assert payload["inherited_candidates"] == [source["candidate"]]
    assert all(item["case"]["seed"] != sources[item["source_index"]]["case"]["seed"]
               for item in plan["transfer_cases"])
    assert plan["transfer_cases"][1]["task"]["response_seed"] != sources[2]["case"]["changed_response_seed"]


def test_unpublished_author_keeps_blocked_slots_fixed_reference_and_unknown_usage(tmp_path):
    sources = sources_for(tmp_path)
    plan = protocol(sources)
    learned = {"status": "provider_failed", "method_source": None,
        "candidate_replay_eligible": True,
        "costs": {"model_calls": 3, "reported_tokens": None, "observation_cost": 192,
                  "measurement_cost": 72, "compute_cost": 5000},
        "events": [{"usage": {"total_tokens": 100, "input_tokens": 90, "cached_input_tokens": 40}},
                   {"usage": {"total_tokens": 120, "input_tokens": 110, "cached_input_tokens": 60}},
                   {"usage": {"total_tokens": None, "input_tokens": None}}]}
    jobs, skipped = target_jobs(sources, tmp_path, plan, learned)
    assert len(jobs) == len(skipped) == 6
    assert all(job["condition"] == "fixed" and job["method_source"] == DEFAULT_RESEARCH_PROCEDURE for job in jobs)
    assert len({job["path"] for job in jobs}) == 6
    assert {job["spec"]["source_index"] for job in jobs} == {2, 3}
    assert all(item["reason"] == "method_not_published" for item in skipped)
    replay_only = {**deepcopy(learned), "status": "offline_candidate_replay_finished",
                   "method_source": "successfully_replayed_but_unpublished_source"}
    replay_jobs, replay_skipped = target_jobs(sources, tmp_path, plan, replay_only)
    assert len(replay_skipped) == 6 and all(job["condition"] == "fixed" for job in replay_jobs)
    results = [{"condition": "fixed", "kind": "unchanged", "status": "completed",
        "confirmed_and_correct": True, "false_confirmation": False, "unchanged_prior_selected": True,
        "costs": {"observation_cost": 96, "measurement_cost": 0, "compute_cost": 168,
                  "model_calls": 0, "reported_tokens": 0},
        "known_tokens": 0, "input_tokens": 0, "cached_input_tokens": 0, "missing_usage_calls": 0,
        "method_steps": 3}]
    summary = summarize(sources, learned, results, skipped, plan)
    assert summary["registered_target_episodes"] == 12
    assert not summary["method_published"] and not summary["paired_comparison_available"]
    assert summary["conditions"]["learned"]["author_blocked_targets"] == 6
    assert summary["actual_new_author_plus_transfer_costs"]["model_calls"] == 3
    assert summary["actual_new_author_plus_transfer_costs"]["reported_tokens"] is None
    assert summary["actual_new_author_plus_transfer_costs"]["observation_cost"] == 288
    assert summary["known_tokens"] == 220 and summary["missing_usage_calls"] == 1
    assert summary["existing_all_source_acquisition_costs"]["model_calls"] == 21
    published = {**deepcopy(learned), "status": "completed", "method_source": "published_actual_source"}
    pairs, blocked = target_jobs(sources, tmp_path, plan, published)
    assert len(pairs) == 12 and blocked == []
    for identifier in {job["spec"]["id"] for job in pairs}:
        pair = [job for job in pairs if job["spec"]["id"] == identifier]
        assert pair[0]["spec"] == pair[1]["spec"] and pair[0]["source"] == pair[1]["source"]
        assert {job["condition"] for job in pair} == {"learned", "fixed"}


def test_resume_reuses_paid_practice_and_deducts_actual_original_author_budget(tmp_path):
    sources = sources_for(tmp_path)
    original = protocol(sources)
    old = tmp_path / "previous"
    learning = old / "learning"
    paid = learning / "trials/trial_000"
    paid.mkdir(parents=True)
    (paid / "result.json").write_text(
        '{"status":"method_stopped","paid_rows":72,"fresh_confirmation":null}')
    current_source = "def select_actions(spec, records, memory):\n    return {'stop':'current revision'}\n"
    prior = {"status": "provider_failed", "workspace": str(learning),
        "costs": {"model_calls": 5, "reported_tokens": None, "observation_cost": 72,
                  "measurement_cost": 0, "compute_cost": 2208, "python_execution_seconds": .25},
        "events": [{"action": "write", "arguments": {"path": METHOD_PATH, "code": current_source},
                    "usage": {"total_tokens": 82939, "input_tokens": 82000, "cached_input_tokens": 40000}},
                   {"action": "provider_error", "usage": {"total_tokens": None}}],
        "outgoing_knowledge": {"files": {METHOD_PATH: current_source}},
        "practice_trials": [{"case": "case_0", "directory": "trials/trial_000",
            "method_source": SEED_PYTHON_METHOD, "status": "method_stopped",
            "selected_prior_predictor": False, "fresh_confirmation": None}]}
    (learning / "result.json").write_text(json.dumps(prior))
    (old / "protocol.json").write_text(json.dumps(original))
    (old / "existing_source_outcomes.json").write_text(json.dumps({"sources": sources}))
    fixed = {"attempted_targets": 6, "costs": {"observation_cost": 672, "measurement_cost": 288,
        "compute_cost": 10406, "python_execution_seconds": 5., "model_calls": 0, "reported_tokens": 0},
        "known_tokens": 0, "input_tokens": 0, "cached_input_tokens": 0, "missing_usage_calls": 0}
    (old / "summary.json").write_text(json.dumps({"conditions": {"fixed": fixed}}))
    resumed, restored_sources, previous = resume_author_state(learning / "result.json")
    assert resumed["practice_cases"] == original["practice_cases"]
    assert resumed["transfer_cases"] == original["transfer_cases"] and restored_sources == sources
    assert resumed["author_limits"] == {**original["author_limits"], "max_model_calls": 19,
        "observation_budget": 2328, "compute_budget": 3597792,
        "execution_seconds": 359.75, "token_budget": 817061}
    assert resumed["resume_author"]["prior_unknown_usage_calls"] == 1
    assert resumed["prior_fixed_reference_runs"][0]["costs"]["observation_cost"] == 672

    def no_replay(*args):
        raise AssertionError("restoration must not run a paid practice")

    learner = PythonArchiveMethodLearner(None, tmp_path / "resumed/learning",
        knowledge={"files": {METHOD_PATH: SEED_PYTHON_METHOD, "prior.py": "def predict(row): return 0."},
                   "research_roles": {"predictor": "prior.py"}},
        cases=resumed["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**resumed["author_limits"]), prepare_trial=no_replay)
    context = learner.restore_practice(previous)
    assert (learner.workspace / METHOD_PATH).read_text() == current_source
    assert (learner.workspace / "trials/trial_000/result.json").read_text() == (paid / "result.json").read_text()
    assert learner.practice_trials == prior["practice_trials"]
    assert context["current_source_coverage"]["missing_cases"] == list(original["practice_cases"])
    assert learner.costs["observation_cost"] == learner.costs["compute_cost"] == learner.costs["model_calls"] == 0
    assert "fixed" not in json.dumps(context)
    current = {"status": "model_call_budget_exhausted", "method_source": None,
        "costs": {"observation_cost": 96, "compute_cost": 500, "model_calls": 1, "reported_tokens": 20},
        "events": [{"usage": {"total_tokens": 20, "input_tokens": 19, "cached_input_tokens": 10}}]}
    _, skipped = target_jobs(sources, tmp_path, resumed, current)
    summary = summarize(sources, current, [], skipped, resumed)
    assert summary["cumulative_author_costs"]["model_calls"] == 6
    assert summary["cumulative_author_costs"]["reported_tokens"] is None
    assert summary["cumulative_author_usage"]["known_tokens"] == 82959
    assert summary["cumulative_author_usage"]["missing_usage_calls"] == 1
    assert summary["actual_all_author_phases_plus_all_transfer_costs"]["observation_cost"] == 840


@pytest.mark.parametrize("exhausted_status", ["token_budget_exhausted", "model_call_budget_exhausted"])
def test_development_continuation_adds_allowance_and_retains_paid_coverage(tmp_path, exhausted_status):
    sources = sources_for(tmp_path)
    original = protocol(sources)
    old = tmp_path / "previous"
    learning = old / "learning"
    current_source = "def select_actions(spec, records, memory): return {'model': 'prior.py'}\n"
    trials = [
        {"case": "case_0", "directory": "trials/trial_000", "method_source": SEED_PYTHON_METHOD,
         "status": "method_stopped", "selected_prior_predictor": False, "fresh_confirmation": None},
        {"case": "case_0", "directory": "trials/trial_001", "method_source": current_source,
         "status": "completed", "selected_prior_predictor": True,
         "fresh_confirmation": {"passed": True, "mse": 0., "rows": 24}},
    ]
    for trial in trials:
        directory = learning / trial["directory"]
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"status": trial["status"], "paid_rows": 96,
            "fresh_confirmation": trial["fresh_confirmation"]}))
    first = {"result": str(tmp_path / "original/learning/result.json"), "status": "provider_failed",
        "costs": {"observation_cost": 72, "measurement_cost": 0, "compute_cost": 2208,
                  "python_execution_seconds": .25, "model_calls": 5, "reported_tokens": None},
        "usage": {"known_tokens": 82939, "input_tokens": 81384,
                  "cached_input_tokens": 34560, "missing_usage_calls": 1}}
    fixed = {"attempted_targets": 6, "costs": {"observation_cost": 672, "measurement_cost": 288,
        "compute_cost": 10406, "python_execution_seconds": 5., "model_calls": 0, "reported_tokens": 0},
        "known_tokens": 0, "input_tokens": 0, "cached_input_tokens": 0, "missing_usage_calls": 0}
    original.update(author_total_limits=deepcopy(original["author_limits"]),
        prior_author_phases=[first], prior_fixed_reference_runs=[{
            "summary": str(tmp_path / "original/summary.json"), "attempted_targets": 6,
            "costs": fixed["costs"], "usage": {key: 0 for key in first["usage"]}}])
    previous = {"status": exhausted_status, "method_source": None, "workspace": str(learning),
        "costs": {"observation_cost": 960, "measurement_cost": 0, "compute_cost": 13560,
                  "python_execution_seconds": 6., "model_calls": 12, "reported_tokens": 909796},
        "events": [{"action": "finish", "arguments": {}, "feedback": {"error": "missing path"},
            "usage": {"total_tokens": 909796, "input_tokens": 900193, "cached_input_tokens": 506624}}],
        "outgoing_knowledge": {"files": {METHOD_PATH: current_source}}, "practice_trials": trials}
    result_path = learning / "result.json"
    result_path.write_text(json.dumps(previous))
    original_bytes = result_path.read_bytes()
    (old / "protocol.json").write_text(json.dumps(original))
    (old / "existing_source_outcomes.json").write_text(json.dumps({"sources": sources}))
    (old / "summary.json").write_text(json.dumps({"conditions": {"fixed": fixed}}))

    plan, restored_sources, restored = continue_author_state(result_path)
    assert plan["author_phase"] == "additional_development"
    assert plan["author_limits"] == CONTINUATION_ALLOWANCE
    assert plan["author_limits"]["max_model_calls"] == 8 and plan["author_limits"]["token_budget"] == 300000
    assert plan["author_total_limits"]["max_model_calls"] == 17 + CONTINUATION_ALLOWANCE["max_model_calls"]
    assert plan["author_total_limits"]["token_budget"] == 992735 + CONTINUATION_ALLOWANCE["token_budget"]
    assert plan["author_original_limits"] == original["author_total_limits"]
    for case, old_spec in original["practice_cases"].items():
        assert plan["practice_cases"][case] == {
            **old_spec, "confirmation_seed": old_spec["confirmation_seed"] + 700001}
    assert plan["prefix"] == original["prefix"] and plan["limits"] == original["limits"]
    assert restored_sources == sources and restored == previous
    assert len(plan["transfer_cases"]) == len(original["transfer_cases"]) == 6
    for old_spec, new_spec in zip(original["transfer_cases"], plan["transfer_cases"]):
        for key in ("id", "kind", "source_index"):
            assert new_spec[key] == old_spec[key]
        assert new_spec["case"] == {**old_spec["case"],
            "seed": old_spec["case"]["seed"] + 700003,
            "changed_response_seed": old_spec["case"]["changed_response_seed"] + 700009}
        expected_response = old_spec["task"]["response_seed"] + (
            700009 if old_spec["kind"] == "response_parameter_change" else 0)
        assert new_spec["task"] == {**old_spec["task"],
            "seed": old_spec["task"]["seed"] + 700003, "response_seed": expected_response}
        assert new_spec["confirmation_seed"] == old_spec["confirmation_seed"] + 700003
    repeated, _, _ = continue_author_state(result_path)
    assert repeated == plan
    assert plan["continue_author"]["prior_unknown_usage_calls"] == 1
    assert [phase["status"] for phase in plan["prior_author_phases"]] == ["provider_failed", exhausted_status]

    def no_replay(*args):
        raise AssertionError("restoration must not replay a paid practice")

    learner = PythonArchiveMethodLearner(None, tmp_path / "continued/learning",
        knowledge={"files": {METHOD_PATH: SEED_PYTHON_METHOD, "prior.py": "def predict(row): return 0."},
                   "research_roles": {"predictor": "prior.py"}},
        cases=plan["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]), prepare_trial=no_replay)
    context = learner.restore_practice(restored)
    assert (learner.workspace / METHOD_PATH).read_text() == current_source
    assert learner.practice_trials == trials
    for trial in trials:
        relative = trial["directory"] + "/result.json"
        assert (learner.workspace / relative).read_bytes() == (learning / relative).read_bytes()
    assert context["current_source_coverage"]["confirmed_cases"] == ["case_0"]
    assert context["current_source_coverage"]["missing_cases"] == ["case_1", "case_2"]
    assert "fixed" not in json.dumps(context)
    assert learner.costs["observation_cost"] == learner.costs["model_calls"] == 0
    current = {"status": "model_call_budget_exhausted", "method_source": None,
        "costs": {"observation_cost": 96, "compute_cost": 500, "model_calls": 2, "reported_tokens": 400},
        "events": [{"usage": {"total_tokens": 400, "input_tokens": 390, "cached_input_tokens": 200}}]}
    _, skipped = target_jobs(sources, tmp_path, plan, current)
    summary = summarize(sources, current, [], skipped, plan)
    assert summary["author_phase"] == "additional_development"
    assert summary["additional_author_allowance"] == CONTINUATION_ALLOWANCE
    assert summary["cumulative_author_costs"]["model_calls"] == 19
    assert summary["cumulative_author_costs"]["reported_tokens"] is None
    assert summary["cumulative_author_usage"]["known_tokens"] == 993135
    assert summary["cumulative_author_usage"]["missing_usage_calls"] == 1
    assert summary["prior_fixed_reference_costs"]["observation_cost"] == 1344
    assert summary["actual_all_author_phases_plus_all_transfer_costs"]["observation_cost"] == 2472
    assert result_path.read_bytes() == original_bytes

    # Successful publication cannot be silently turned into another author attempt.
    completed_path = learning / "completed.json"
    completed_path.write_text(json.dumps({**previous, "status": "completed", "method_source": current_source}))
    with pytest.raises(ValueError, match="unpublished"):
        continue_author_state(completed_path)
    with pytest.raises(ValueError, match="provider_failed"):
        resume_author_state(result_path)
