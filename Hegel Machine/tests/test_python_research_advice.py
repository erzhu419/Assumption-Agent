"""Action feedback exposes existing paid-evidence decisions without new research."""

from copy import deepcopy
import json
from random import Random

import pytest

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


MEAN = ("_mean = 0.\ndef fit(rows):\n global _mean\n"
        " _mean = sum(row['target'] for row in rows) / len(rows)\n"
        "def predict(row):\n return _mean\n")
LOOKUP = ("_targets = {}\ndef fit(rows):\n global _targets\n"
          " _targets = {row['features']['u']: row['target'] for row in rows}\n"
          "def predict(row):\n return _targets.get(row['features']['u'], 0.)\n")


class SilentAdviceAgent(TemporalOperatorAgent):
    def _render_research_advice(self, advice):
        return None


def prepared(path, *, cls=TemporalOperatorAgent, mode="recurrent", code=MEAN):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode=mode)
    agent = cls(None, path, arm="code_react", source_task=False,
        config=InquiryLLMConfig(observation_budget=384, compute_budget=360000,
            confirmation_rows=24, execution_seconds=120.))
    rng = Random(71)
    agent._execute("query", {"actions": [
        {"kind": "trajectory", "channels": [],
         "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}
        for _ in range(3)]}, lab)
    agent._save("model.py", code)
    return agent, lab


def costs_without_wall_time(agent):
    return {key: value for key, value in agent.costs.items()
            if key != "python_execution_seconds"}


def test_failed_evaluation_supplies_refit_advice_without_another_scoring_pass(tmp_path):
    agent, lab = prepared(tmp_path)
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    result = agent._execute("evaluate", {"model": "model.py"}, lab)
    assert result["mse"] > 1e-4
    advice = result["research_advice"]
    assert advice["next_action"]["action"] == "run"
    assert advice["next_action"]["arguments"]["operator"] == "refit"
    assert advice["evaluated_model"]["evaluated_rows"] == 72
    assert agent.costs["compute_cost"] - before["compute_cost"] == 72
    assert lab.costs == lab_before
    assert agent.model is None and not agent.confirmations


def test_failed_refit_supplies_meter_advice_without_rescoring_or_buying(tmp_path):
    agent, lab = prepared(tmp_path)
    evaluated = agent._execute("evaluate", {"model": "model.py"}, lab)
    action = evaluated["research_advice"]["next_action"]
    before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
    result = agent._execute(action["action"], action["arguments"], lab)
    assert result["status"] == "refit_inadequate_on_validation"
    advice = result["research_advice"]
    assert advice["next_action"]["action"] == "query"
    assert all(action["channels"] == ["meter"]
               for action in advice["next_action"]["arguments"]["actions"])
    assert agent.costs["compute_cost"] - before["compute_cost"] == 72
    assert lab.costs == lab_before
    assert "research_advice" not in json.dumps(agent.operator_results)
    assert agent.model is None and not agent.confirmations


def test_failed_fresh_confirmation_cannot_recommend_finish_from_stale_support(tmp_path):
    agent, lab = prepared(tmp_path, code=LOOKUP)
    evaluated = agent._execute("evaluate", {"model": "model.py"}, lab)
    assert evaluated["mse"] == 0.
    retained = evaluated["evaluated_model"]["model"]
    before = deepcopy(agent.costs)
    result = agent._execute("finish", {"model": retained}, lab)
    assert result["fresh_confirmation"]["passed"] is False
    assert result["fresh_confirmation"]["mse"] > 1e-4
    advice = result["research_advice"]
    current = advice["evaluated_model"]
    assert current["current_paid_rows"] == 96
    assert advice.get("next_action", {}).get("action") != "finish"
    assert current["support"] == "fresh_counterexample"
    assert current["mse"] is None
    assert current["fresh_mse"] == result["fresh_confirmation"]["mse"]
    stored = agent._evaluated_summary()
    assert stored["support"] == "new_paid_rows_unscored"
    assert stored["evaluated_rows"] == 72
    assert agent.costs["compute_cost"] - before["compute_cost"] == 72 + 24
    assert agent.costs["observation_cost"] - before["observation_cost"] == 24
    assert agent.model is None and len(agent.confirmations) == 1


def test_worse_evaluation_advises_confirming_the_correct_saved_model(tmp_path):
    agent, lab = prepared(tmp_path, mode="finite2")
    learned = agent._execute("run", {"operator": "history", "base_lags": 3,
        "path": "history.py"}, lab)
    correct = agent._execute("evaluate", {"model": learned["path"]}, lab)
    assert correct["mse"] < 1e-10
    saved = deepcopy(correct["evaluated_model"])
    before = deepcopy(agent.costs)
    worse = agent._execute("evaluate", {"model": "model.py"}, lab)
    assert worse["mse"] > 1e-4
    assert worse["evaluated_model"] == saved
    action = worse["research_advice"]["next_action"]
    assert action == {"action": "finish", "arguments": {"model": saved["model"]}}
    assert agent.costs["compute_cost"] - before["compute_cost"] == 72
    assert agent.costs["observation_cost"] == before["observation_cost"]
    confirmed = agent._execute(action["action"], action["arguments"], lab)
    assert confirmed["fresh_confirmation"]["passed"]


@pytest.mark.parametrize("next_step", ["refit", "finish"])
def test_hiding_advice_keeps_tools_costs_and_operator_history_unchanged(tmp_path, next_step):
    code = MEAN if next_step == "refit" else LOOKUP
    visible, visible_lab = prepared(tmp_path / "visible", code=code)
    silent, silent_lab = prepared(tmp_path / "silent", cls=SilentAdviceAgent, code=code)
    assert visible.common_interface == silent.common_interface
    assert visible_lab.public_spec() == silent_lab.public_spec()
    normal = visible._execute("evaluate", {"model": "model.py"}, visible_lab)
    hidden = silent._execute("evaluate", {"model": "model.py"}, silent_lab)
    assert "research_advice" not in hidden
    assert costs_without_wall_time(visible) == costs_without_wall_time(silent)
    action = (normal["research_advice"]["next_action"] if next_step == "refit" else
              {"action": "finish", "arguments": {"model": normal["evaluated_model"]["model"]}})
    normal = visible._execute(action["action"], action["arguments"], visible_lab)
    hidden = silent._execute(action["action"], action["arguments"], silent_lab)
    assert "research_advice" in normal and "research_advice" not in hidden
    assert costs_without_wall_time(visible) == costs_without_wall_time(silent)
    assert visible_lab.costs == silent_lab.costs
    silent._prepare_program_state()
    state = json.loads((silent.workspace / "research_state.json").read_text())
    assert "research_advice" not in json.dumps(state["operator_results"])
    # Only unsolicited feedback is hidden; both agents retain the same explicit
    # research-choice tool and pay for its candidate evaluations normally.
    normal = visible._execute("run", {"operator": "research_choice"}, visible_lab)
    hidden = silent._execute("run", {"operator": "research_choice"}, silent_lab)
    assert normal["next_action"] == hidden["next_action"]
    assert costs_without_wall_time(visible) == costs_without_wall_time(silent)


def test_feedback_alone_connects_recurrent_counterexample_to_confirmed_response(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path)

    def unexpected_explicit_choice(*args, **kwargs):
        raise AssertionError("the feedback chain must not invoke research_choice")

    monkeypatch.setattr(agent, "_python_research_choice", unexpected_explicit_choice)
    feedback = agent._execute("evaluate", {"model": "model.py"}, lab)
    assert feedback["mse"] > 1e-4
    actions = []
    for _ in range(8):
        action = feedback["research_advice"]["next_action"]
        arguments = action["arguments"]
        actions.append((action["action"], arguments.get("operator")))
        before, lab_before = deepcopy(agent.costs), deepcopy(lab.costs)
        assert agent.model is None
        feedback = agent._execute(action["action"], arguments, lab)
        assert "error" not in feedback, feedback
        if action["action"] not in {"query", "finish"}:
            assert lab.costs == lab_before
        if arguments.get("operator") == "refit":
            assert feedback["status"] == "refit_inadequate_on_validation"
        elif action["action"] == "query":
            requested = sum(lab.action_cost(item) for item in arguments["actions"])
            assert agent.costs["observation_cost"] - before["observation_cost"] == requested
            assert [item["operator"] for item in agent.operator_results] == [
                "refit", "instrument", "assess_instrument"]
            # The public query result must reflect its completed assessment,
            # not the earlier instrument fit's now-outdated next step.
            assert feedback["research_advice"]["next_action"]["arguments"]["operator"] == "response"
        elif action["action"] == "evaluate":
            assert feedback["mse"] < 1e-10
            # The incumbent was last scored before the meter purchases. Both
            # candidates legitimately pay for comparison on the same 120 rows.
            assert feedback["retained_comparison"]["mse"] > 1e-4
            assert agent.costs["compute_cost"] - before["compute_cost"] == 2 * len(agent.rows)
        elif action["action"] == "finish":
            assert feedback["fresh_confirmation"]["passed"]
            assert agent.costs["observation_cost"] - before["observation_cost"] == 24
            assert agent.costs["compute_cost"] - before["compute_cost"] == len(agent.rows)
            break
    assert actions == [("run", "refit"), ("query", None), ("run", "response"),
                       ("evaluate", None), ("finish", None)]
    assert agent.model is not None and len(agent.confirmations) == 1
    terminal = evaluate_python_model(agent.workspace, agent.model,
        lab.evaluation_rows(length=128, seed=1000291), timeout_seconds=10.)
    assert terminal["returncode"] == 0 and terminal["mse"] < 1e-10
    assert agent.costs["model_calls"] == 0
