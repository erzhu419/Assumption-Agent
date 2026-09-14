"""Continuation returns use independent copies and genuinely subsequent data."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ActionRecord, Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.research_methods import ResearchMethods
from hegel_machine.cognition.session import ResearchSession


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_method_continuation_study.py"
SPEC = importlib.util.spec_from_file_location("method_continuation_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def saved_checkpoint():
    config = ResearchConfig(seed=61, observation_budget=24, compute_budget=5000, max_rounds=1)
    return ResearchSession(config, make_suite(61), observation_budget=120,
                           compute_budget=30000, max_visits=1).run(max_steps=1)


def test_method_ablation_restores_selection_even_when_a_visit_fails():
    choose, opportunity, observe = ResearchMethods.choose, ResearchMethods.opportunity, ResearchMethods.observe
    action_choose = ResearchActionPolicy.choose
    with pytest.raises(RuntimeError, match="visit failed"):
        with study.without_method_selection():
            methods = ResearchMethods()
            assert methods.choose() is None and methods.opportunity() is None
            assert ResearchMethods.observe is observe
            assert ResearchActionPolicy.choose is action_choose
            raise RuntimeError("visit failed")
    assert ResearchMethods.choose is choose and ResearchMethods.opportunity is opportunity


def test_two_real_continuations_share_the_paid_start_and_evaluation_arrives_only_after_both(monkeypatch):
    checkpoint = saved_checkpoint()
    original = deepcopy(checkpoint)
    run = study.CognitiveAgent.run
    inputs, finished = [], []

    def observed_run(agent, environment, *, research_state):
        inputs.append({"seed": agent.config.seed, "config": agent.config,
                       "workspace": deepcopy(research_state), "knowledge": agent.library.to_dict()})
        result = run(agent, environment, research_state=research_state)
        finished.append(result)
        return result

    evaluate_rows = study._new_evaluation_rows

    def after_both(environment, seed, trajectories):
        assert len(finished) == 2
        return evaluate_rows(environment, seed, trajectories)

    monkeypatch.setattr(study.CognitiveAgent, "run", observed_run)
    monkeypatch.setattr(study, "_new_evaluation_rows", after_both)
    report = study.run_study(checkpoint, tasks=[0], observation_budget=16,
                             compute_budget=5000, rounds=1, evaluation_trajectories=2)
    assert inputs[0] == inputs[1]
    assert inputs[0]["config"].learn_research_policy is True
    assert checkpoint == original
    result = report["tasks"][0]
    assert result["source_visits"] == checkpoint["limits"]["max_visits"] == 1
    for arm in ("methods", "ordinary"):
        row = result["arms"][arm]
        assert 0 < row["compute_cost"] <= 5000 and 0 <= row["observation_cost"] <= 16
        assert row["observations_after"] - result["saved_observations"] == row["observation_cost"]
        resume = next(action for action in row["actions"] if action["action"] == "resume_research")
        assert resume["details"]["restored_observations"] == result["saved_observations"]
        before = result["arms"]["no_additional"]["evaluation"]["objective_loss"]
        assert row["change_from_saved"]["improvement"] == before - row["evaluation"]["objective_loss"]
        assert "actions_awaiting_feedback" in row["pending"]
    assert result["arms"]["no_additional"]["observation_cost"] == result["arms"]["no_additional"]["compute_cost"] == 0
    assert result["arms"]["no_additional"]["fallback"] == checkpoint["workspaces"]["0"]["fallback"]
    assert result["arms"]["no_additional"]["uncertainty_model"] == checkpoint["workspaces"]["0"]["uncertainty_model"]
    assert result["evaluation_costs"]["candidate_row_compute"] == 3 * result["evaluation_costs"]["observation_rows"]


def test_cli_persists_new_costs_and_leaves_the_source_checkpoint_untouched(tmp_path, capsys):
    checkpoint = tmp_path / "learning.json"
    saved = saved_checkpoint()
    checkpoint.write_text(json.dumps(saved))
    original = checkpoint.read_bytes()
    output = tmp_path / "study.json"
    assert study.main(["--checkpoint", str(checkpoint), "--tasks", "0", "--output", str(output),
                       "--observation-budget", "16", "--compute-budget", "5000", "--rounds", "1",
                       "--evaluation-trajectories", "2"]) == 0
    report = json.loads(output.read_text())
    summary = json.loads(capsys.readouterr().out)
    assert report["status"] == summary["status"] == "completed"
    assert summary["arms"]["methods"]["compute_cost"] == report["tasks"][0]["arms"]["methods"]["compute_cost"]
    assert checkpoint.read_bytes() == original
    assert output.with_name("study_summary.json").exists()


def test_decision_evaluation_uses_the_saved_fallback_and_residual_action():
    program = Program("lag", name="x", lag=1)
    rows = [Observation({"x": 0.}, 4.)]
    spec = TaskSpec("decision", "", ("x",), (), (), risk_quantile=.9, decision_costs=(.9, .1))
    model = {"kind": "global_interval", "residual_interval": [-1., 2.],
             "action_rule": "empirical_cost_quantile", "action_offset": 1.}
    calibrated = study._measure(program, 3., model, rows, spec)
    nominal = study._measure(program, 3., {}, rows, spec)
    assert calibrated["prediction_loss"] == nominal["prediction_loss"] == .9
    assert calibrated["objective_loss"] == 0.
    assert nominal["objective_loss"] == .9


def test_small_changes_are_separated_from_material_returns_and_learning_feedback_is_retained():
    details = {"gain": .0001, "evidence": "adaptive_learning_trajectories", "reference_rows": 8}
    action = ActionRecord(4, "continuation_feedback", "", details=details)
    assert study._action_trace([action])[0]["details"] == details
    tasks = []
    for gain in (.0001, -.0001, .003, -.004):
        arms = {}
        for arm in ("methods", "ordinary", "no_additional"):
            record = {"observation_cost": 0, "compute_cost": 0,
                      "evaluation": {"loss_kind": "squared_error", "objective_loss": 1., "prediction_loss": 1.}}
            if arm != "no_additional":
                record["change_from_saved"] = study._change(1., 1. - gain)
            arms[arm] = record
        tasks.append({"arms": arms, "method_advantage": study._change(1., 1.),
                      "evaluation_costs": {"observation_rows": 8, "candidate_row_compute": 24}})
    summary = study.summarize({"config": {"minimum_improvement": .002}, "tasks": tasks})
    assert summary["arms"]["methods"]["outcomes"] == {"improved": 2, "worsened": 2}
    assert summary["arms"]["methods"]["material_outcomes"] == {"unchanged": 2, "improved": 1, "worsened": 1}
    assert summary["method_advantage_material_outcomes"] == {"unchanged": 4}
