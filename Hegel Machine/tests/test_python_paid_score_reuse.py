"""Only a complete execution on the same paid inputs can avoid another score."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.research_episodes import load_source_values
from test_inquiry_revision import PublicLab, action


class ChoiceLab(PublicLab):
    def public_spec(self):
        return {**super().public_spec(), "feature_bounds": {"u": [-3., 3.]},
                "max_probe_length": 2}


def scene(path):
    lab = ChoiceLab()
    agent = CommonArchiveInquiryAgent(None, path, arm="hegel", source_task=False,
        config=InquiryLLMConfig(observation_budget=30, confirmation_rows=2,
            compute_budget=1000, execution_seconds=30., max_model_calls=0))
    agent._execute("query", {"actions": [action(1., 2.)],
        "question": {"id": "q", "question": "Predict the public response",
                     "context_ref": {"apparatus": "local_fixture"}}}, lab)
    for name, value in (("old", 0.), ("new", 1.)):
        agent._save(f"{name}/_evaluated_model/model.py", f"def predict(row): return {value!r}\n")
        agent._save(f"{name}/_evaluated_model/params.json", '{"intercept": 0}')
    return agent, lab


MODELS = ["old/_evaluated_model/model.py", "new/_evaluated_model/model.py"]


def choose(agent, lab, models=MODELS):
    return agent._execute("run", {"operator": "research_choice", "models": models}, lab)


def no_execution(*args, **kwargs):
    raise AssertionError("An exact paid score must not predict or fit again")


def scoring_costs(agent):
    # The existing shared-operator ledger includes its host wall time, even
    # when it only reads references and invokes no Python model worker.
    return {key: value for key, value in agent.costs.items() if key != "python_execution_seconds"}


def test_repeat_reuses_actual_scores_and_reapplies_current_retention(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    first = choose(agent, lab)
    assert first["new_scored_count"] == 2 and first["reused_score_count"] == 0
    assert agent.evaluated_model["source_model"] == MODELS[1]
    assert first["evaluations"][1]["retention_decision"]["status"] == "candidate_retained"
    costs, incumbent = scoring_costs(agent), deepcopy(agent.evaluated_model)
    monkeypatch.setattr("hegel_machine.cognition.online_inquiry_llm.evaluate_python_model", no_execution)
    second = choose(agent, lab)
    assert second["reused_score_count"] == 2 and second["new_scored_count"] == 0
    assert scoring_costs(agent) == costs and agent.evaluated_model == incumbent
    assert second["evaluations"][1]["retention_decision"]["status"] == "incumbent_retained"
    for evaluation in second["evaluations"]:
        reuse = evaluation["paid_score_reuse"]
        loaded, = load_source_values([reuse["assessment_ref"]], agent.workspace)
        original = loaded["value"]
        assert original["mse"] == evaluation["mse"]
        assert "retention_decision" not in original
        assert reuse["original_costs"]["compute_cost"] == len(agent.rows)
        assert reuse["additional_scoring_compute"] == 0
    assert second["compute_cost"] == 0 and agent.model is None and not agent.confirmations


@pytest.mark.parametrize("change", ["source", "params", "dependency", "fit_view", "bindings",
                                   "row_order", "target", "meaning", "context", "public_contract"])
def test_changed_execution_inputs_are_scored_again(tmp_path, change):
    agent, lab = scene(tmp_path)
    choose(agent, lab, MODELS[:1])
    context = agent.workspace / "old/_evaluated_model"
    if change == "source":
        (context / "model.py").write_text("def predict(row): return -1.\n")
    elif change == "params":
        (context / "params.json").write_text('{"intercept": 3}')
    elif change == "dependency":
        (context / "dependency.py").write_text("parameter = 7\n")
    elif change == "fit_view":
        (context / "model_observations.json").write_text('[{"features":{"u":1},"target":9}]')
    elif change == "bindings":
        (context / "model_bindings.json").write_text('{"model.py":{}}')
    elif change == "row_order":
        agent.rows.reverse()
    elif change == "target":
        agent.rows[0] = replace(agent.rows[0], target=5.)
    elif change == "meaning":
        agent.questions[-1]["meaning_version"] += 1
    elif change == "context":
        agent.questions[-1]["context_ref"] = {"apparatus": "different"}
    elif change == "public_contract":
        original = lab.public_spec
        lab.public_spec = lambda: {**original(), "prediction_contract": "changed contract"}
    before = agent.costs["compute_cost"]
    result = choose(agent, lab, MODELS[:1])
    assert result["new_scored_count"] == 1 and result["reused_score_count"] == 0
    assert agent.costs["compute_cost"] - before == len(agent.rows)
    assert result["evaluations"][0]["paid_score_reuse"]["reuse_miss_reason"] != "no_retained_complete_execution_basis"


@pytest.mark.parametrize("failed", ["exception", "truncated", "partial_relations"])
def test_failed_or_truncated_execution_never_becomes_a_reusable_score(tmp_path, monkeypatch, failed):
    agent, lab = scene(tmp_path)
    if failed == "exception":
        agent._save(MODELS[0], "def predict(row): raise ValueError('fixture failure')\n")
    elif failed == "truncated":
        monkeypatch.setattr("hegel_machine.cognition.online_inquiry_llm.evaluate_python_model",
            lambda *args, **kwargs: {"returncode": 0, "truncated": True, "status": "scoped_unknown"})
    else:
        monkeypatch.setattr("hegel_machine.cognition.online_inquiry_llm.compare_paid_relations",
            lambda *args, **kwargs: {"check_count": 1, "term_count": 0, "truncated": True})
    first = choose(agent, lab, MODELS[:1])
    second = choose(agent, lab, MODELS[:1])
    assert first["new_scored_count"] == second["new_scored_count"] == 1
    assert first["reused_score_count"] == second["reused_score_count"] == 0
    assert second["evaluations"][0]["paid_score_reuse"]["unavailable_reason"] == "incomplete_or_failed_execution"
    assert agent.costs["compute_cost"] == 2 * len(agent.rows)


def test_fit_mutation_is_not_misrepresented_as_an_unchanged_execution(tmp_path):
    agent, lab = scene(tmp_path)
    agent._save(MODELS[0], "from pathlib import Path\n"
        "def fit(rows):\n p=Path('counter.txt'); p.write_text(str(int(p.read_text())+1))\n"
        "def predict(row): return 0.\n")
    agent._save("old/_evaluated_model/counter.txt", "0")
    agent._save("old/_evaluated_model/model_observations.json", "[]")
    for expected in ("1", "2"):
        result = choose(agent, lab, MODELS[:1])
        reuse = result["evaluations"][0]["paid_score_reuse"]
        assert not reuse["reusable"] and reuse["unavailable_reason"] == "execution_context_changed_during_scoring"
        assert (agent.workspace / "old/_evaluated_model/counter.txt").read_text() == expected


def test_completed_scores_survive_later_budget_failure_and_cold_Common_restore(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path / "source")
    agent.config = replace(agent.config, compute_budget=len(agent.rows))
    with pytest.raises(ValueError, match="scoring budget exhausted"):
        choose(agent, lab)
    # The first candidate was fully executed before the second hit its budget.
    before = scoring_costs(agent)
    agent.config = replace(agent.config, compute_budget=1000)
    one = choose(agent, lab, MODELS[:1])
    assert one["reused_score_count"] == 1 and scoring_costs(agent) == before
    agent._prepare_program_state()
    previous = agent.result("researching")
    cold, cold_lab = scene(tmp_path / "cold")
    # A real restore needs an unused destination and apparatus.
    import shutil
    shutil.rmtree(cold.workspace)
    cold.workspace.mkdir()
    cold_lab = ChoiceLab()
    cold_lab.costs = {"observation_cost": 0}
    cold.restore_practice_research(previous, agent.workspace, cold_lab, confirmation_seed=0)
    monkeypatch.setattr("hegel_machine.cognition.online_inquiry_llm.evaluate_python_model", no_execution)
    before = scoring_costs(cold)
    repeated = choose(cold, cold_lab, MODELS[:1])
    assert repeated["reused_score_count"] == 1 and repeated["new_scored_count"] == 0
    assert scoring_costs(cold) == before
    reference = repeated["evaluations"][0]["paid_score_reuse"]["execution_basis_ref"]
    loaded, = load_source_values([reference], cold.workspace)
    assessment = loaded["value"]
    assert Path(assessment["execution_context"]["workspace"]).is_dir()
    assert "rows" not in assessment and "predictions" not in assessment["result"]


def test_old_source_flag_and_partial_residuals_do_not_fabricate_execution_basis(tmp_path):
    agent, lab = scene(tmp_path)
    agent.operator_results.append({"operator": "research_choice", "evaluations": [
        {"path": MODELS[0], "returncode": 0, "mse": 0.,
         "inherited_source_unchanged": True, "paid_residuals": {"evaluated_rows": 2, "largest": []}}]})
    choice = choose(agent, lab, MODELS[:1])
    reuse = choice["evaluations"][0]["paid_score_reuse"]
    assert choice["new_scored_count"] == 1 and choice["reused_score_count"] == 0
    assert reuse["reuse_miss_reason"] == "no_retained_complete_execution_basis"
    assert choice["evaluations"][0]["mse"] > 0.


def test_root_predictor_matches_snapshot_despite_unbound_host_forecast_files(tmp_path, monkeypatch):
    agent, lab = scene(tmp_path)
    agent._save("model.py", "def predict(row): return 0.\n")
    agent._save("_experiment_models/python_forecasts_0000.json", '{"saved_public_forecasts": []}')
    first = choose(agent, lab, ["model.py"])
    assert first["evaluations"][0]["paid_score_reuse"]["reusable"]
    agent._save("_experiment_models/python_forecasts_0001.json", '{"saved_public_forecasts": []}')
    monkeypatch.setattr("hegel_machine.cognition.online_inquiry_llm.evaluate_python_model", no_execution)
    repeated = choose(agent, lab, ["model.py"])
    assert repeated["reused_score_count"] == 1 and repeated["new_scored_count"] == 0


def test_declared_child_in_host_context_is_still_compared(tmp_path):
    agent, lab = scene(tmp_path)
    agent._save("model.py", "_models={}\ndef bind_models(models):\n global _models\n _models=models\n"
        "def predict(row): return _models['child'](row)\n")
    child = "_experiment_models/child/_evaluated_model/model.py"
    agent._save(child, "def predict(row): return 0.\n")
    agent._save("model_bindings.json", '{"model.py":{"child":' + repr(child).replace("'", '"') + '}}')
    first = choose(agent, lab, ["model.py"])
    assert first["evaluations"][0]["paid_score_reuse"]["reusable"]
    cached = choose(agent, lab, ["model.py"])
    assert cached["reused_score_count"] == 1
    agent._save(child, "def predict(row): return -1.\n")
    changed = choose(agent, lab, ["model.py"])
    assert changed["new_scored_count"] == 1 and changed["reused_score_count"] == 0
    assert changed["evaluations"][0]["mse"] > first["evaluations"][0]["mse"]
