"""The Python chooser must turn real counterexamples into executable research."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from random import Random
import shutil

import pytest

from hegel_machine.cognition.code_agent import archive_confirmed_python_model, evaluate_python_model
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent


def config():
    return InquiryLLMConfig(observation_budget=384, compute_budget=360000,
        confirmation_rows=24, execution_seconds=120.)


def purchase(agent, lab, *, count=3, channels=(), seed=71):
    rng = Random(seed)
    return agent._execute("query", {"actions": [
        {"kind": "trajectory", "channels": list(channels),
         "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}
        for _ in range(count)]}, lab)


@pytest.fixture
def source_archive(tmp_path):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="finite2")
    agent = TemporalOperatorAgent(None, tmp_path / "source", arm="code_react",
        source_task=False, config=config())
    purchase(agent, laboratory)
    learned = agent._execute("run", {"operator": "history", "base_lags": 3,
        "path": "shared/predictor.py"}, laboratory)
    assert learned["base"]["heldout_mse"] < 1e-10
    submitted = agent._execute("finish", {"model": learned["path"]}, laboratory)
    assert submitted["fresh_confirmation"]["passed"]
    return archive_confirmed_python_model(agent, tmp_path / "bank" / "_evaluated_model")


def target(tmp_path, source_archive, *, mode="finite2"):
    laboratory = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910079 if mode == "finite2" else 910061, instrument_mode=mode)
    agent = TemporalOperatorAgent(None, tmp_path / "target", arm="code_react",
        source_task=False, config=config())
    destination = agent.workspace / "shared/old/_evaluated_model"
    shutil.copytree(source_archive["snapshot_workspace"], destination)
    model = str(Path("shared/old/_evaluated_model") / source_archive["inner_model"])
    agent.candidate_paths.append(model)
    purchase(agent, laboratory, seed=79)
    return agent, laboratory, model


def choose(agent, lab, **arguments):
    previous = deepcopy(lab.costs)
    before_compute = agent.costs["compute_cost"]
    choice = agent._execute("run", {"operator": "research_choice", **arguments}, lab)
    assert lab.costs == previous  # Advice and paid-data scoring buy no evidence.
    assert agent.costs["compute_cost"] - before_compute == choice["compute_cost"]
    assert agent.model is None
    return choice


def execute_next(agent, lab, choice):
    action = choice["next_action"]
    result = agent._execute(action["action"], action["arguments"], lab)
    assert "error" not in result, result
    assert agent.costs["observation_cost"] == lab.costs["observation_cost"]
    assert agent.costs["measurement_cost"] == lab.costs["measurement_cost"]
    if action["action"] != "finish":
        assert agent.config.observation_budget - agent.costs["observation_cost"] >= agent.config.confirmation_rows
        assert agent._remaining_compute() >= len(agent.rows) + agent.config.confirmation_rows
    return result


def assert_final(agent, lab, choice):
    assert choice["next_action"]["action"] == "finish"
    before = deepcopy(agent.costs)
    result = execute_next(agent, lab, choice)
    assert result["fresh_confirmation"]["passed"]
    assert agent.costs["observation_cost"] - before["observation_cost"] == agent.config.confirmation_rows
    terminal = evaluate_python_model(agent.workspace, agent.model,
        lab.evaluation_rows(length=128, seed=1000291), timeout_seconds=10.)
    assert terminal["returncode"] == 0 and terminal["mse"] < 1e-10
    assert agent.costs["model_calls"] == 0


def test_parameter_counterexample_routes_through_original_refit_to_confirmation(tmp_path, source_archive):
    agent, laboratory, old_model = target(tmp_path, source_archive)
    first = choose(agent, laboratory, models=[old_model])
    assert first["evaluations"][0]["mse"] > 1e-4
    assert first["next_action"]["arguments"]["operator"] == "refit"
    refitted = execute_next(agent, laboratory, first)
    assert refitted["status"] == "refit_supported_on_validation"
    assert refitted["tested_model"] == refitted["path"]
    assert (agent.workspace / refitted["tested_model"]).read_bytes() == (agent.workspace / old_model).read_bytes()
    second = choose(agent, laboratory)
    assert any(row["path"] == refitted["path"] and row["mse"] < 1e-10
               for row in second["evaluations"])
    assert_final(agent, laboratory, second)


def test_recurrence_counterexample_routes_from_failed_refit_to_confirmed_measurement(tmp_path, source_archive):
    agent, laboratory, old_model = target(tmp_path, source_archive, mode="recurrent")
    first = choose(agent, laboratory, models=[old_model])
    assert first["next_action"]["arguments"]["operator"] == "refit"
    refitted = execute_next(agent, laboratory, first)
    assert refitted["status"] == "refit_inadequate_on_validation"
    assert refitted["validation"]["mse"] > 1e-10

    measure = choose(agent, laboratory)
    assert measure["next_action"]["action"] == "query"
    arguments = measure["next_action"]["arguments"]
    assert all(action["channels"] == ["meter"] for action in arguments["actions"])
    assert any(step["operator"] == "instrument" for step in arguments["after_observation"])
    assert any(step["operator"] == "assess_instrument" for step in arguments["after_observation"])
    instrument = execute_next(agent, laboratory, measure)
    assert any(step["operator"] == "instrument" for step in instrument["after_observation_results"])
    assert any(step.get("status") == "supported_on_measured_trajectory"
               for step in instrument["after_observation_results"])

    response = choose(agent, laboratory)
    assert response["next_action"]["arguments"]["operator"] == "response"
    fitted = execute_next(agent, laboratory, response)
    assert any(item["validation_mse"] < 1e-10 for item in fitted["candidates"])
    train_sequences = [agent.observations[index]["action"]["sequence"]
                       for index in fitted["split"]["training_records"]]
    validation_sequences = [agent.observations[index]["action"]["sequence"]
                            for index in fitted["split"]["validation_records"]]
    assert all(sequence not in validation_sequences for sequence in train_sequences)
    final = choose(agent, laboratory)
    assert any(item["mse"] < 1e-10 for item in final["evaluations"])
    assert_final(agent, laboratory, final)
    assert sum(row["operator"] == "refit" for row in agent.operator_results) == 1


def test_missing_fit_uses_existing_meter_evidence_without_rebuying_it(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False, config=config())
    purchase(agent, lab)
    purchase(agent, lab, count=1, channels=("meter",), seed=89)
    agent._save("constant.py", "def predict(row):\n return 0.\n")
    first = choose(agent, lab, models=["constant.py"])
    assert first["next_action"]["arguments"]["operator"] == "refit"
    refitted = execute_next(agent, lab, first)
    assert refitted["status"] == "refit_unavailable"
    second = choose(agent, lab)
    assert second["next_action"]["action"] == "run"
    assert second["next_action"]["arguments"]["operator"] == "instrument"
    before = deepcopy(lab.costs)
    execute_next(agent, lab, second)
    assert lab.costs == before
    third = choose(agent, lab)
    # Both finite and recurrent instruments can fit this one real metered
    # trajectory. Their distinguishing plan is advice, followed by a purchase.
    if third["next_action"]["action"] == "run":
        assert third["next_action"]["arguments"]["operator"] == "disagreement"
        execute_next(agent, lab, third)
        third = choose(agent, lab)
    assert third["next_action"]["action"] == "query"
    assert any(step["operator"] == "assess_instrument"
               for step in third["next_action"]["arguments"]["after_observation"])
    assert sum(row["operator"] == "refit" for row in agent.operator_results) == 1
    execute_next(agent, lab, third)
    response = choose(agent, lab)
    assert response["next_action"]["arguments"]["operator"] == "response"
    execute_next(agent, lab, response)
    assert_final(agent, lab, choose(agent, lab))


def test_zero_paired_instrument_attempt_recovers_with_real_measurements_and_confirmation(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="recurrent")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False, config=config())
    purchase(agent, lab)
    before = deepcopy(lab.costs)
    attempted = agent._execute("run", {"operator": "instrument", "records": [0, 1, 2],
        "path": "empty_instrument.json", "compute_limit": 100000}, lab)
    assert attempted["support_rows"] == 0 and attempted["hypotheses"] == []
    assert attempted["unmodeled_fields"]["meter"]["reason"] == "insufficient_paired_measurements"
    assert lab.costs == before
    proposed = attempted["research_advice"]
    assert proposed["next_action"]["action"] == "query"
    arguments = proposed["next_action"]["arguments"]
    assert len(arguments["actions"]) == 2
    assert all(action["channels"] == ["meter"] for action in arguments["actions"])
    assert [step["operator"] for step in arguments["after_observation"]] == [
        "instrument", "assess_instrument"]
    measured = execute_next(agent, lab, proposed)
    fitted, assessed = measured["after_observation_results"]
    assert fitted["path"] == "empty_instrument.json" and fitted["support_rows"] >= 6
    assert assessed["status"] == "supported_on_measured_trajectory"
    assert measured["research_advice"]["next_action"]["arguments"]["operator"] == "response"
    response = execute_next(agent, lab, measured["research_advice"])
    assert any(candidate["validation_mse"] < 1e-10 for candidate in response["candidates"])
    evaluated = execute_next(agent, lab, response["research_advice"])
    assert evaluated["mse"] < 1e-10
    assert_final(agent, lab, evaluated["research_advice"])


def test_unaffordable_research_is_not_recommended_over_confirmation_reserve(tmp_path, source_archive):
    agent, laboratory, old_model = target(tmp_path, source_archive)
    # Scoring the old 72-row predictor leaves 108 units, enough to finish but
    # insufficient for another 72-row refit followed by 96-unit confirmation.
    agent.config = replace(agent.config, compute_budget=180, observation_budget=96)
    result = choose(agent, laboratory, models=[old_model])
    assert not result.get("next_action")
    assert agent.costs["observation_cost"] == 72
    assert agent._remaining_compute() >= len(agent.rows) + agent.config.confirmation_rows


@pytest.mark.parametrize("evaluate_original", [False, True])
def test_failed_validation_is_not_erased_by_interpolating_all_paid_rows(tmp_path, evaluate_original):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False, config=config())
    purchase(agent, lab)
    agent._save("lookup.py", "_targets = {}\n"
        "def fit(rows):\n global _targets\n"
        " _targets = {row['features']['u']: row['target'] for row in rows}\n"
        "def predict(row):\n return _targets.get(row['features']['u'], 0.)\n")
    validation = agent._execute("run", {"operator": "refit", "model": "lookup.py",
        "path": "shared/lookup_refit"}, lab)
    assert validation["status"] == "refit_inadequate_on_validation"
    model = "lookup.py" if evaluate_original else validation["path"]
    all_paid = agent._execute("evaluate", {"model": model}, lab)
    assert all_paid["mse"] == 0.
    choice = choose(agent, lab)
    assert choice["evaluated_model"]["mse"] == 0.
    assert choice["next_action"]["action"] == "query"
    assert all(action["channels"] == ["meter"] for action in choice["next_action"]["arguments"]["actions"])
    assert agent.model is None and not agent.confirmations


def test_rewritten_predictor_does_not_inherit_another_sources_failed_refit(tmp_path, source_archive):
    agent, lab, old_model = target(tmp_path, source_archive, mode="recurrent")
    first = choose(agent, lab, models=[old_model])
    previous = execute_next(agent, lab, first)
    assert previous["status"] == "refit_inadequate_on_validation"
    model = agent._evaluated_summary()["model"]
    agent._execute("write", {"path": model, "code":
        "_mean = 0.\ndef fit(rows):\n global _mean\n"
        " _mean = sum(row['target'] for row in rows) / len(rows)\n"
        "def predict(row):\n return _mean\n"}, lab)
    choice = choose(agent, lab, models=[model])
    assert choice["evaluated_model"]["mse"] > 1e-10
    assert choice["next_action"]["arguments"]["operator"] == "refit"
    assert choice["next_action"]["arguments"]["model"] == model


def test_failed_new_refit_does_not_block_correct_original_frozen_coefficients(tmp_path):
    source_lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="finite2")
    source = TemporalOperatorAgent(None, tmp_path / "source", arm="code_react",
        source_task=False, config=config())
    purchase(source, source_lab)
    learned = source._execute("run", {"operator": "history", "base_lags": 5,
        "path": "shared/predictor.py"}, source_lab)
    submitted = source._execute("finish", {"model": learned["path"]}, source_lab)
    assert submitted["fresh_confirmation"]["passed"]
    archive = archive_confirmed_python_model(source, tmp_path / "bank" / "_evaluated_model")

    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061,
        response_seed=910061, instrument_mode="finite2")
    agent = TemporalOperatorAgent(None, tmp_path / "return", arm="code_react",
        source_task=False, config=config())
    shutil.copytree(archive["snapshot_workspace"], agent.workspace / "shared/old/_evaluated_model")
    original = str(Path("shared/old/_evaluated_model") / archive["inner_model"])
    purchase(agent, lab, seed=79)
    # One training trajectory underdetermines this larger same-source model;
    # that refit failure does not refute coefficients fitted on 72 source rows.
    refitted = agent._execute("run", {"operator": "refit", "model": original,
        "path": "shared/underidentified"}, lab)
    assert refitted["status"] == "refit_inadequate_on_validation"
    assert refitted["validation"]["mse"] > 1e-4
    assert (agent.workspace / refitted["tested_model"]).read_bytes() == (agent.workspace / original).read_bytes()
    choice = choose(agent, lab, models=[original])
    assert choice["evaluated_model"]["source_model"] == original
    assert choice["evaluated_model"]["mse"] < 1e-10
    assert_final(agent, lab, choice)


def test_changed_imported_fitting_helper_does_not_inherit_previous_refit_failure(tmp_path):
    lab = TemporalInquiryLab(task_index=1, apparatus_seed=910061, instrument_mode="recurrent")
    agent = TemporalOperatorAgent(None, tmp_path, arm="code_react", source_task=False, config=config())
    purchase(agent, lab)
    agent._save("model.py", "from helper import fit, predict\n")
    agent._save("helper.py", "_mean = 0.\ndef fit(rows):\n global _mean\n"
        " _mean = sum(row['target'] for row in rows) / len(rows)\n"
        "def predict(row):\n return _mean\n")
    first = choose(agent, lab, models=["model.py"])
    refitted = execute_next(agent, lab, first)
    assert refitted["status"] == "refit_inadequate_on_validation"
    retained = agent._evaluated_summary()["model"]
    helper = str(Path(retained).parent / "helper.py")
    agent._execute("write", {"path": helper, "code":
        "_slope = 0.\ndef fit(rows):\n global _slope\n"
        " _slope = sum(r['features']['u'] * r['target'] for r in rows) / "
        "sum(r['features']['u'] ** 2 for r in rows)\n"
        "def predict(row):\n return _slope * row['features']['u']\n"}, lab)
    assert (agent.workspace / retained).read_bytes() == (agent.workspace / refitted["tested_model"]).read_bytes()
    choice = choose(agent, lab, models=[retained])
    assert choice["evaluated_model"]["mse"] > 1e-10
    assert choice["next_action"]["arguments"]["operator"] == "refit"
    assert choice["next_action"]["arguments"]["model"] == retained
