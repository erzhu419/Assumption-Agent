"""Preserve prior outcomes while charging and admitting only newly executed work."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import run_python_scoped_method_study as runner
from scripts.python_selective_method_cases import protocol as previous_protocol


def inputs():
    return [{"case": {"apparatus_seed": 971106001 + 1009 * i,
                      "changed_response_seed": 981106001 + 1009 * i,
                      "seed": 991106001 + 1009 * i},
             "candidate": None if i == 0 else {"candidate_id": "candidate_000"}}
            for i in range(4)]


def old_results(plan):
    costs = runner.total_costs([{"observation_cost": 96, "measurement_cost": 12,
                                "compute_cost": 168}])
    return [{"target_id": spec["id"], "kind": spec["kind"], "condition": condition,
             "status": "completed", "confirmed_and_correct": condition != "fixed",
             "false_confirmation": condition == "fixed", "unchanged_prior_selected": False,
             "costs": deepcopy(costs), "terminal": {"private_marker": "terminal_secret"}}
            for spec in plan["transfer_cases"]
            for condition in ("revised", "fixed", "seed_selective")]


def test_reference_reuse_preserves_failures_and_rejects_a_changed_confirmation_stream():
    sources = inputs()
    plan, old_plan = runner.study_protocol(sources), previous_protocol(sources)
    old = {"results": old_results(old_plan)}
    saved = deepcopy(old)
    reused = runner.reusable_references(plan, old_plan, old)
    assert len(reused) == 16
    assert {item["condition"] for item in reused} == {"prior_published", "fixed"}
    assert sum(item["false_confirmation"] for item in reused) == 8
    assert all(item["condition"] != "seed_selective" for item in reused)
    reused[0]["costs"]["observation_cost"] = -1
    assert old == saved
    plan["transfer_cases"][0]["confirmation_seed"] += 1
    with pytest.raises(ValueError, match="same task and confirmation stream"):
        runner.reusable_references(plan, old_plan, old)


@pytest.fixture
def study(tmp_path, monkeypatch):
    sources = inputs()
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps({"sources": sources}))
    previous = tmp_path / "previous"
    learning = previous / "learning"
    learning.mkdir(parents=True)
    old_plan = previous_protocol(sources)
    source = "def select_actions(spec, records, memory): return {'stop': 'original'}\n"
    seed = {"method_source": source, "practice_trials": [
        {"case": "case_3", "method_source": "obsolete source", "directory": "missing_old_trial"},
        {"case": "case_3", "method_source": source, "directory": "trials/current"}]}
    prior_costs = runner.total_costs([{"observation_cost": 4000, "compute_cost": 9000,
                                     "model_calls": 7, "reported_tokens": None}])
    summary = {"results": old_results(old_plan),
        "cumulative_author_costs": runner.total_costs([{"observation_cost": 500,
            "model_calls": 7, "reported_tokens": None}]),
        "cumulative_author_usage": {"known_tokens": 1000, "missing_usage_calls": 1,
                                   "input_tokens": 800, "cached_input_tokens": 200},
        "all_method_development_costs": prior_costs,
        "source_acquisition_costs": runner.total_costs([{"observation_cost": 432}])}
    paid = {"status": "completed", "costs": runner.total_costs([
                {"observation_cost": 216, "measurement_cost": 216, "compute_cost": 6269}]),
            "events": [], "paid_observations": [], "terminal": "terminal_secret"}
    for path, data in [(learning / "result.json", seed), (previous / "protocol.json", old_plan),
                       (previous / "summary.json", summary),
                       (learning / "trials/current/result.json", paid)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "SOURCE_OUTCOMES", source_path)
    monkeypatch.setattr(runner, "PREVIOUS", previous)
    monkeypatch.setattr(runner, "archive_memory", lambda source: {"files": {}})
    monkeypatch.setattr(runner.shutil, "copytree", lambda *args, **kwargs: None)
    state = SimpleNamespace(author_done=False, published=False, trained=[], preparations=[],
        scored=[], restored=[], configs=[], status="provider_failed", old_summary=summary,
        source=source, expected_executed=4, tmp_path=tmp_path)
    state.author_costs = runner.total_costs([{"observation_cost": 120, "compute_cost": 312,
        "python_execution_seconds": 1., "model_calls": 2, "reported_tokens": None}])
    state.target_costs = runner.total_costs([{"observation_cost": 144,
        "measurement_cost": 72, "compute_cost": 1200}])

    class Author:
        def __init__(self, provider, workspace, **kwargs):
            self.workspace, self.files, self.common_interface = workspace, {}, ""
            workspace.mkdir(parents=True)
            state.configs.append(kwargs["config"])
            assert len(kwargs["cases"]) == 5
            assert kwargs["knowledge"]["files"][runner.METHOD_PATH] == source

        def restore_practice(self, prior):
            state.restored.append(deepcopy(prior))
            return {"restored_trials": len(prior["practice_trials"])}

        def run(self, **kwargs):
            assert "terminal_secret" not in json.dumps(self.files)
            assert not state.trained
            state.author_done = True
            state.published = state.status == "completed"
            trial = self.workspace / "trials/paid_failure/result.json"
            trial.parent.mkdir(parents=True)
            trial.write_text(json.dumps({"status": "method_stopped",
                "costs": {"observation_cost": 120}, "paid_observations": [],
                "events": [{"action": "query", "feedback": {"error": "earlier source failed"}}],
                "outgoing_knowledge": {"files": {runner.METHOD_PATH: "earlier source"}}}))
            return {"status": state.status, "method_source": "exact author revision",
                "costs": deepcopy(state.author_costs), "practice_trials": [
                    {"case": "case_3", "status": "method_stopped", "method_source": "earlier source",
                     "directory": "trials/paid_failure", "costs": {"observation_cost": 120}}],
                "outgoing_knowledge": {"files": {runner.METHOD_PATH: "exact author revision"}},
                "events": deepcopy(getattr(state, "author_events", [
                    {"action": "run", "usage": {"total_tokens": 100, "input_tokens": 80, "cached_input_tokens": 32}},
                    {"action": "provider_error", "usage": {"total_tokens": None}}]))}

    class Frozen:
        def __init__(self, path, method):
            self.path, self.method = path, method

        def execute_method(self, laboratory, max_steps):
            assert state.author_done and max_steps == 12
            if self.path.name.endswith("_revised"):
                assert state.published and self.method == "exact author revision"
            elif self.path.name.endswith("_prior_published"):
                assert self.method == source
            else:
                assert self.method == runner.DEFAULT_RESEARCH_PROCEDURE
            state.trained.append(self.path.name)
            return {"status": "completed", "costs": deepcopy(state.target_costs)}

    def prepare(source, plan):
        def one(spec, path, method, limits):
            assert state.author_done
            assert limits.max_model_calls == limits.token_budget == 0
            state.preparations.append((spec["id"], path.name, method))
            path.mkdir(parents=True)
            return {"agent": Frozen(path, method), "path": path, "laboratory": None}
        return one

    def score(payload, plan):
        assert len(state.trained) == state.expected_executed
        state.scored.append(payload["path"].name)
        return {"status": "completed", "confirmed_and_correct": True,
                "false_confirmation": False, "costs": payload["training"]["costs"]}

    monkeypatch.setattr(runner, "ScopedPythonMethodLearner", Author)
    monkeypatch.setattr(runner, "trial_preparer", prepare)
    monkeypatch.setattr(runner, "score_episode", score)
    monkeypatch.setattr(runner, "execution_evidence", lambda *args: {})
    monkeypatch.setattr(runner, "selected_candidate", lambda payload: None)
    return state


@pytest.mark.parametrize("published", [False, True])
def test_only_published_source_runs_new_targets_and_all_costs_are_charged_once(study, published):
    study.status = "completed" if published else "provider_failed"
    study.expected_executed = 14 if published else 4
    result = runner.run_study(study.tmp_path / "run", provider_factory=lambda _: None)
    assert result["method_published"] is published
    assert result["reused_reference_episodes"] == 16
    assert len(study.trained) == len(study.scored) == study.expected_executed
    new_refs = [(case, name) for case, name, _ in study.preparations if not name.endswith("_revised")]
    assert len(new_refs) == 4 and {case for case, _ in new_refs} == {"target_0_4", "target_1_4"}
    assert len(result["skipped"]) == (0 if published else 10)
    assert result["conditions"]["revised"]["attempted_targets"] == (10 if published else 0)
    assert result["conditions"]["fixed"]["false_confirmations"] == 8
    expected = runner.total_costs([study.author_costs] + [study.target_costs] * study.expected_executed)
    assert result["new_author_and_transfer_costs"] == expected
    assert result["all_method_development_costs"] == runner.total_costs([
        study.old_summary["all_method_development_costs"], expected])
    assert result["cumulative_author_costs"]["observation_cost"] == 620
    assert result["cumulative_author_usage"] == {"known_tokens": 1100,
        "missing_usage_calls": 2, "input_tokens": 880, "cached_input_tokens": 232}
    plan = json.loads((study.tmp_path / "run/protocol.json").read_text())
    assert len(plan["transfer_cases"]) == 10 and plan["conditions"] == list(runner.CONDITIONS)
    assert plan["author_limits"]["max_model_calls"] == 16
    assert plan["author_limits"]["token_budget"] == 650000


def test_service_resume_preserves_paid_trials_and_references_without_extra_allowance(study):
    first = runner.run_study(study.tmp_path / "first", provider_factory=lambda _: None)
    path = study.tmp_path / "first/learning/result.json"
    old_bytes = path.read_bytes()
    old_result = json.loads(old_bytes)
    study.trained.clear()
    study.scored.clear()
    study.preparations.clear()
    study.author_done, study.expected_executed = False, 0
    resumed = runner.run_study(study.tmp_path / "resumed", provider_factory=lambda _: None,
                               resume_author=path)
    assert study.restored == [old_result] and path.read_bytes() == old_bytes
    assert study.restored[0]["practice_trials"][0]["status"] == "method_stopped"
    remaining = study.configs[-1]
    assert remaining.max_model_calls == 14 and remaining.token_budget == 649900
    assert remaining.observation_budget == 4680 and remaining.compute_budget == 2999688
    assert remaining.execution_seconds == 299.
    assert not study.trained and not study.preparations
    assert resumed["reused_reference_episodes"] == 20
    assert resumed["results"] == first["results"] and len(resumed["skipped"]) == 10
    assert resumed["new_author_and_transfer_costs"] == study.author_costs
    assert resumed["all_method_development_costs"] == runner.total_costs([
        first["all_method_development_costs"], study.author_costs])
    assert resumed["cumulative_author_usage"]["known_tokens"] == 1200
    assert resumed["cumulative_author_usage"]["missing_usage_calls"] == 3
    assert resumed["cumulative_author_costs"]["reported_tokens"] is None


@pytest.mark.parametrize("last_action", ["run", "write"])
def test_exhausted_development_restores_failed_source_and_declares_only_new_allowance(study, last_action):
    # Actual first scoped-author totals: token exhaustion leaves three original calls.
    study.status = "token_budget_exhausted"
    study.author_costs = runner.total_costs([{"observation_cost": 1272,
        "measurement_cost": 432, "compute_cost": 11568,
        "python_execution_seconds": 3.8166569171880838,
        "model_calls": 13, "reported_tokens": 652609}])
    arguments = ({"path": runner.METHOD_PATH, "code": "exact author revision"}
                 if last_action == "write" else {"code": "print('inspect paid feedback')"})
    study.author_events = [{"action": last_action, "arguments": arguments, "usage": {
        "total_tokens": 652609, "input_tokens": 621623, "cached_input_tokens": 126208},
        "feedback": {"returncode": 0}}]
    first = runner.run_study(study.tmp_path / "exhausted", provider_factory=lambda _: None)
    path = study.tmp_path / "exhausted/learning/result.json"
    failed = json.loads(path.read_text())
    # Later entries from another source and successful trials must not replace
    # the latest actual failure of the restored source.
    failed["practice_trials"].extend([
        {"case": "case_1", "status": "method_stopped", "method_source": "exact author revision",
         "directory": "trials/earlier_current_failure"},
        {"case": "case_2", "status": "method_stopped", "method_source": "exact author revision",
         "directory": "trials/latest_current_failure"},
        {"case": "case_3", "status": "method_stopped", "method_source": "other source",
         "directory": "trials/later_other_source_failure"},
        {"case": "case_4", "status": "completed", "method_source": "exact author revision",
         "directory": "trials/later_success"}])
    path.write_text(json.dumps(failed))
    old_bytes = path.read_bytes()
    actual_failure_path = path.parent / "trials/latest_current_failure/result.json"
    actual_failure_path.parent.mkdir(parents=True)
    actual_failure = {"status": "method_stopped", "costs": runner.total_costs([
        {"observation_cost": 192, "measurement_cost": 216, "compute_cost": 3979}]),
        "paid_observations": [{"row_indices": list(range(96, 120)), "measurements": []}],
        "events": [{"step": 0, "action": "query", "feedback": {
            "operator": "response", "error": "current response has no validated candidate"}}],
        "terminal": "terminal_secret", "task": {"mechanism": "private_mechanism_secret"},
        "outgoing_knowledge": {"files": {runner.METHOD_PATH: "exact author revision"}}}
    actual_failure_path.write_text(json.dumps(actual_failure))
    study.trained.clear()
    study.scored.clear()
    study.preparations.clear()
    study.author_done, study.expected_executed = False, 0
    study.status = "model_call_budget_exhausted"
    study.author_costs = runner.total_costs([{"model_calls": 1, "reported_tokens": 50}])
    study.author_events = [{"action": "write", "usage": {
        "total_tokens": 50, "input_tokens": 40, "cached_input_tokens": 16}}]
    result = runner.run_study(study.tmp_path / "continued", provider_factory=lambda _: None,
                              continue_author=path)
    assert study.restored == [failed] and path.read_bytes() == old_bytes
    assert study.restored[0]["outgoing_knowledge"]["files"][runner.METHOD_PATH] == "exact author revision"
    assert study.restored[0]["practice_trials"][0]["status"] == "method_stopped"
    assert result["reused_paid_trace"] == str(actual_failure_path)
    visible = json.loads((study.tmp_path / "continued/learning/counterexample.json").read_text())
    assert visible["practice_case"] == "case_2" and visible["status"] == "method_stopped"
    assert visible["steps"][0]["feedback"]["error"] == "current response has no validated candidate"
    assert visible["costs"] == actual_failure["costs"]
    assert visible["executed_method"]["source"] == "exact author revision"
    for name in ("counterexample.json", "counterexample_trace.json", "counterexample_observations.json"):
        text = (study.tmp_path / "continued/learning" / name).read_text()
        assert "terminal_secret" not in text and "private_mechanism_secret" not in text
    plan = json.loads((study.tmp_path / "continued/protocol.json").read_text())
    phase = plan["continue_author"]
    assert phase["added_allowance"] == {"model_calls": 6, "tokens": 400000}
    assert phase["previous_status"] == "token_budget_exhausted"
    assert plan["author_guidance"] == phase["author_guidance"]
    if last_action == "write":
        assert "last action rewrote the research source" in plan["author_guidance"]
        assert "using finish now to obtain paid practice feedback" in plan["author_guidance"]
    else:
        assert "restored source and actual paid practice failures" in plan["author_guidance"]
        assert "using finish now" not in plan["author_guidance"]
    remaining = study.configs[-1]
    assert remaining.max_model_calls == 9 and remaining.token_budget == 400000
    assert remaining.observation_budget == 3528 and remaining.compute_budget == 2988432
    assert remaining.execution_seconds == pytest.approx(296.1833430828119)
    assert phase["remaining_limits"] == plan["author_limits"]
    assert not study.trained and not study.preparations
    assert result["reused_reference_episodes"] == 20 and result["results"] == first["results"]
    assert len(result["skipped"]) == 10 and not result["method_published"]
    assert result["new_author_and_transfer_costs"] == study.author_costs
    assert result["all_method_development_costs"] == runner.total_costs([
        first["all_method_development_costs"], study.author_costs])
    assert result["cumulative_author_costs"]["observation_cost"] == 500 + 1272
    assert result["cumulative_author_usage"]["known_tokens"] == 1000 + 652609 + 50
    assert result["cumulative_author_usage"]["missing_usage_calls"] == 1


def test_researcher_comparison_is_visible_not_coverage_and_charged_only_on_explicit_supply(study, monkeypatch):
    directory = study.tmp_path / "researcher_comparison"
    directory.mkdir()
    comparison = directory / "scope_comparison.json"
    text = json.dumps({"experimenter": "researcher", "cases": [
        {"case": "case_3", "validation_mse": 1e-25, "fresh_confirmation": {"passed": True}},
        {"case": "case_4", "validation_mse": .031, "fresh_confirmation": {"passed": False}}]}, indent=2)
    comparison.write_text(text)
    costs = runner.total_costs([{"observation_cost": 360, "measurement_cost": 144,
                                "compute_cost": 4400, "python_execution_seconds": 1.5}])
    (directory / "summary.json").write_text(json.dumps({"costs": costs}))
    visible = []
    original_author = runner.ScopedPythonMethodLearner

    class InspectingAuthor(original_author):
        def run(self, **kwargs):
            visible.append((self.files["scope_comparison.json"],
                            (self.workspace / "scope_comparison.json").read_text(),
                            self.common_interface))
            return super().run(**kwargs)

    monkeypatch.setattr(runner, "ScopedPythonMethodLearner", InspectingAuthor)
    first = runner.run_study(study.tmp_path / "with_comparison", provider_factory=lambda _: None,
                             scope_comparison=comparison)
    assert first["new_scope_comparison_costs"] == costs
    assert first["new_work_including_diagnostic"] == runner.total_costs([
        first["new_author_and_transfer_costs"], costs])
    assert first["all_method_development_costs"] == runner.total_costs([
        study.old_summary["all_method_development_costs"], first["new_author_and_transfer_costs"], costs])
    assert not first["method_published"] and len(first["skipped"]) == 10
    assert first["conditions"]["revised"]["attempted_targets"] == 0
    path = study.tmp_path / "with_comparison/learning/result.json"
    original_result = json.loads(path.read_text())
    assert len(original_result["practice_trials"]) == 1
    assert original_result["practice_trials"][0]["status"] == "method_stopped"
    study.trained.clear()
    study.scored.clear()
    study.preparations.clear()
    study.author_done, study.expected_executed = False, 0
    resumed = runner.run_study(study.tmp_path / "reshown_comparison", provider_factory=lambda _: None,
                               resume_author=path)
    assert len(visible) == 2
    for files_text, disk_text, interface in visible:
        assert files_text == disk_text
        assert json.loads(files_text) == json.loads(text)
        assert len(files_text) < len(text)
        assert "researcher-arranged experiments" in interface
        assert "not chosen by your program and do not confer current-source practice coverage" in interface
    assert study.restored == [original_result]
    assert resumed["new_scope_comparison_costs"] == runner.total_costs([])
    assert resumed["new_work_including_diagnostic"] == study.author_costs
    assert resumed["all_method_development_costs"] == runner.total_costs([
        first["all_method_development_costs"], study.author_costs])
    assert resumed["reused_reference_episodes"] == 20 and resumed["results"] == first["results"]
    assert not resumed["method_published"] and len(resumed["skipped"]) == 10
    assert not study.trained and not study.preparations
