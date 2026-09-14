"""Paired apparatus summaries retain failed outcomes and all paid attempts."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("replication_summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_dynamic_research_replication.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
ARMS = ("code", "hegel_v078", "hegel_revised")


def protocol():
    return {"paired_seeds": [101, 202], "stages": ["source", "return"],
        "arms": {arm: {"session": "code" if arm == "code" else "hegel"} for arm in ARMS},
        "terminal": {"solved_threshold": 1e-6}, "statistical_unit": "apparatus sequence",
        "development_evidence": "old_one_instance.json", "compute_note": "different units"}


def report(observations, measurements, queries):
    results, evaluation = [], []
    for position, (rows, cost, calls) in enumerate(zip(observations, measurements, queries)):
        results.append({"task_position": position, "status": "completed", "model_origin": "finish",
            "selected_channels": [],
            "costs": {"observation_cost": rows, "measurement_cost": cost, "compute_cost": rows * 2},
            "actual_costs": {"queries": calls}})
        evaluation.append({"task_position": position, "status": "completed", "loss": 0.,
            "observation_cost": 256, "measurement_cost": 0, "compute_cost": 256})
    return {"session": {"status": "completed", "results": results, "evaluation": evaluation,
        "costs": {"observation_cost": sum(observations), "measurement_cost": sum(measurements),
                  "compute_cost": sum(observations) * 2}}}


def reports():
    return {seed: {"code": report([40, 40], [60, 60], [1, 1]),
        "hegel_v078": report([60, 20], [80, 0], [2, 2]),
        "hegel_revised": report([50, 10], [70, 0], [2, 1])} for seed in (101, 202)}


def test_sequence_pairs_and_stage_tradeoffs_do_not_treat_stages_as_independent_seeds():
    result = summary.summarize(protocol(), reports())
    pair = result["paired"]["hegel_revised_minus_code"]
    assert pair["observation_cost"] == {"pairs": 2, "sum": -40, "median": -20,
        "range": [-20, -20], "lower": 2, "equal": 0, "higher": 0}
    assert pair["query_delta"]["sum"] == 2
    assert result["stages"]["source"]["paired"]["hegel_revised_minus_code"]["measurement_cost"]["sum"] == 20
    assert result["stages"]["return"]["paired"]["hegel_revised_minus_code"]["measurement_cost"]["sum"] == -120
    assert result["arms"]["code"]["fully_solved_sequences"] == 2
    assert result["arms"]["code"]["measurement_free_solved"] == 4
    assert result["development_evidence"] == "old_one_instance.json"


def test_failed_model_retains_cost_and_denominator_while_unscored_tasks_remain_pending():
    data = reports()
    failed = data[101]["code"]["session"]
    failed["evaluation"][1] = {"task_position": 1, "status": "model_failed", "loss": None}
    failed["results"][1]["model_origin"] = "last_successful_evaluation"
    failed["results"][0]["selected_channels"] = ["meter"]
    pending = data[202]["code"]["session"]
    pending["results"] = pending["results"][:1]
    pending["evaluation"] = []
    pending["status"] = "paused"
    pending["costs"] = {"observation_cost": 55, "measurement_cost": 70, "compute_cost": 110}
    result = summary.summarize(protocol(), data)
    code = result["arms"]["code"]
    assert (code["expected_tasks"], code["solved"], code["failed"], code["pending_terminal"]) == (4, 1, 1, 2)
    assert code["mean_loss"] is None and code["recorded_learning_costs"]["observation_cost"] == 120
    assert code["recorded_session_costs"]["observation_cost"] == 135  # Includes the active task's paid 15 rows.
    assert code["model_origins"]["last_successful_evaluation"] == 1
    assert code["measurement_free_solved"] == 0
    pair = result["paired"]["hegel_revised_minus_code"]
    assert pair["evaluated_pairs"] == 1 and pair["solved_delta"]["sum"] == 1
    assert pair["observation_cost"]["pairs"] == 1  # The partial cheap stream is not a completed efficiency win.


def test_reader_falls_back_to_checkpoints_without_turning_missing_arms_into_zero_cost(tmp_path):
    saved = reports()[101]["code"]
    checkpoint = tmp_path / "runs/code/101/code_checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(saved))
    data = summary.read_reports(tmp_path, protocol())
    result = summary.summarize(protocol(), data)
    missing = result["arms"]["hegel_revised"]
    assert missing["pending_terminal"] == 4 and missing["failed"] == 0
    assert missing["recorded_learning_costs"]["observation_cost"] is None
    assert result["paired"]["hegel_revised_minus_code"]["observation_cost"]["pairs"] == 0
    replacement = deepcopy(saved)
    replacement["session"]["evaluation"][0]["loss"] = 2.
    comparison = tmp_path / "comparisons/101.json"
    comparison.parent.mkdir()
    comparison.write_text(json.dumps({"arms": {"code": replacement}}))
    assert summary.summarize(protocol(), summary.read_reports(tmp_path, protocol()))["arms"]["code"]["failed"] == 1
