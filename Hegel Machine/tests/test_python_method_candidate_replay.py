"""Offline candidate replay preserves publication status, gates and actual costs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import run_python_method_candidate_replay as runner
from scripts.run_python_method_learning_study import protocol
from test_python_method_learning_study import sources_for


@pytest.mark.parametrize("gate_passes", [True, False])
def test_frozen_unpublished_candidate_replay_keeps_all_cases_costs_and_terminal_order(
        tmp_path, monkeypatch, gate_passes):
    sources = sources_for(tmp_path)
    archive = Path(sources[1]["candidate"]["snapshot_workspace"])
    archive.mkdir()
    (archive / "model.py").write_text("def predict(row): return 0.\n")
    source = "def select_actions(spec, records, memory): return {'stop': 'exact last candidate'}\n"
    old = tmp_path / "previous"
    learning = old / "learning"
    paid = learning / "trials/trial_000"
    paid.mkdir(parents=True)
    (paid / "result.json").write_text('{"status":"method_stopped","paid_rows":72}')
    prior = {"status": "model_call_budget_exhausted", "method_source": None,
        "workspace": str(learning), "events": [{"action": "write", "arguments": {"path": runner.METHOD_PATH}}],
        "outgoing_knowledge": {"files": {runner.METHOD_PATH: source}},
        "practice_trials": [{"case": "case_0", "directory": "trials/trial_000", "method_source": source,
            "status": "method_stopped", "selected_prior_predictor": False}]}
    previous_path = learning / "result.json"
    previous_path.write_text(json.dumps(prior))
    previous_bytes = previous_path.read_bytes()
    plan = protocol(sources)
    plan["author_limits"].update(max_model_calls=16, token_budget=600000)
    (old / "protocol.json").write_text(json.dumps(plan))
    (old / "existing_source_outcomes.json").write_text(json.dumps({"sources": sources}))
    old_costs = {"model_calls": 16, "reported_tokens": None, "observation_cost": 200,
                 "measurement_cost": 0, "compute_cost": 800, "python_execution_seconds": 1.}
    fixed_costs = {**old_costs, "model_calls": 0, "reported_tokens": 0, "observation_cost": 672}
    old_summary = {"cumulative_author_costs": old_costs,
        "cumulative_author_usage": {"known_tokens": 900000, "missing_usage_calls": 1},
        "prior_fixed_reference_costs": fixed_costs, "conditions": {"fixed": {"costs": fixed_costs}},
        "actual_all_author_phases_plus_all_transfer_costs": {**old_costs, "observation_cost": 1544}}
    (old / "summary.json").write_text(json.dumps(old_summary))
    practiced, trained, models, scored = [], [], [], []
    cost = {"model_calls": 0, "reported_tokens": 0, "observation_cost": 96,
            "measurement_cost": 0, "compute_cost": 192, "python_execution_seconds": .1}

    def practice(learner, case):
        assert learner.provider is None
        assert learner.config.max_model_calls == learner.config.token_budget == 0
        assert (learner.workspace / runner.METHOD_PATH).read_text() == source
        assert (learner.workspace / "trials/trial_000/result.json").read_bytes() == (paid / "result.json").read_bytes()
        assert len(learner.practice_trials) == 1
        practiced.append(case)
        for key, value in cost.items():
            learner.costs[key] += value
        passed = gate_passes or case != "case_0"
        return {"case": case, "status": "completed" if passed else "method_step_limit",
            "fresh_confirmation": {"passed": passed}, "selected_prior_predictor": case == "case_0",
            "costs": deepcopy(cost)}

    def preparer(source_record, execution_plan):
        def prepare(spec, path, method_source, limits):
            assert practiced == list(plan["practice_cases"])
            assert limits.max_model_calls == limits.token_budget == 0
            assert spec in plan["transfer_cases"]
            assert method_source in (source, runner.DEFAULT_RESEARCH_PROCEDURE)
            path.mkdir(parents=True)
            models.append(method_source)

            class Agent:
                def execute_method(self, lab, max_steps):
                    assert not scored
                    trained.append(spec["id"])
                    return {"status": "completed", "costs": deepcopy(cost)}

            return {"agent": Agent(), "path": path, "laboratory": spec, "spec": spec}
        return prepare

    def score(payload, execution_plan):
        assert len(trained) == (12 if gate_passes else 6)
        scored.append(payload["spec"]["id"])
        return {"status": "completed", "confirmed_and_correct": True,
                "false_confirmation": False, "costs": deepcopy(cost)}

    monkeypatch.setattr(runner.PythonArchiveMethodLearner, "_practice", practice)
    monkeypatch.setattr(runner, "trial_preparer", preparer)
    monkeypatch.setattr(runner, "score_episode", score)
    monkeypatch.setattr(runner, "selected_candidate", lambda payload: None)
    monkeypatch.setattr(runner, "execution_evidence", lambda *args: {"method_steps": 1})
    # Snapshotting is irrelevant to these accounting and ordering assertions.
    original_copytree = runner.shutil.copytree
    monkeypatch.setattr(runner.shutil, "copytree", lambda src, dst, **kw:
        original_copytree(src, dst, **kw) if "trials" in str(src) else None)
    summary = runner.run_replay(previous_path, tmp_path / "replay")
    assert summary["author_status"] == "model_call_budget_exhausted"
    assert summary["method_published"] is False
    assert summary["candidate_replay_eligible"] is gate_passes
    assert summary["conditions"]["candidate"]["registered_targets"] == 6
    assert summary["conditions"]["candidate"]["attempted_targets"] == (6 if gate_passes else 0)
    assert summary["conditions"]["fixed"]["attempted_targets"] == 6
    assert len(summary["skipped"]) == (0 if gate_passes else 6)
    assert len(summary["practices"]) == 3 and models.count(source) == (6 if gate_passes else 0)
    assert summary["new_practice_costs"]["observation_cost"] == 288
    assert summary["new_replay_costs"]["observation_cost"] == (1440 if gate_passes else 864)
    assert summary["new_model_calls"] == summary["new_reported_tokens"] == 0
    assert summary["cumulative_author_costs"] == old_costs
    assert summary["actual_cumulative_costs"]["reported_tokens"] is None
    assert summary["prior_fixed_reference_costs"]["observation_cost"] == 1344
    assert previous_path.read_bytes() == previous_bytes
    assert (tmp_path / "replay/candidate.py").read_text() == source
