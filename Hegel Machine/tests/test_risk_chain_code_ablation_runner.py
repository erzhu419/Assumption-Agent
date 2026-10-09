import json
import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[1] / "scripts/run_risk_chain_code_ablation.py"
_SPEC = importlib.util.spec_from_file_location("risk_chain_runner", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_protocol_has_equal_budgets_and_four_risk_cases(tmp_path):
    protocol = runner.prepare(tmp_path)
    assert [task["task_index"] for task in protocol["tasks"]] == [12, 12, 13, 12]
    from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
    with UnknownTaskLab(**protocol["tasks"][0]) as lab:
        assert lab.spec.risk_quantile == .9
    with UnknownTaskLab(**protocol["tasks"][2]) as lab:
        assert lab.spec.risk_quantile == .9
    assert protocol["research_config"]["observation_budget"] == protocol["code_config"]["observation_budget"]
    assert protocol["measurement_budget"] == protocol["code_config"]["measurement_budget"]
    assert protocol["research_config"]["compute_budget"] == protocol["code_config"]["compute_budget"]
    assert protocol["model"] == "gpt-5.6-terra"


def test_summary_does_not_claim_missing_code_arms(tmp_path):
    runner.prepare(tmp_path)
    result = runner.summary(tmp_path)
    assert set(result["arms"]) == set()


def test_summary_handles_null_evaluation_and_reports_failure(tmp_path):
    runner.prepare(tmp_path)
    (tmp_path / "code_no_memory.json").write_text(json.dumps({
        "results": [{"status": "provider_failed"}], "evaluation": None,
        "costs": {"observation_cost": 1, "measurement_cost": 2, "compute_cost": 3}}))
    result = runner.summary(tmp_path)["arms"]["code_no_memory"]
    assert result["evaluation_coverage"] == 0.
    assert result["result_status_counts"] == {"provider_failed": 1}
    assert result["evaluation_status_counts"] == {}


def test_summary_requires_all_planned_losses_for_whole_chain_mean(tmp_path):
    state = {
        "tasks": [{"task_index": 12}, {"task_index": 13}],
        "results": [{"status": "completed"}, {"status": "model_call_budget_exhausted"}],
        "evaluation": [
            {"status": "completed", "loss_kind": "pinball", "loss": .25},
            {"status": "missing_model", "loss_kind": "pinball", "loss": None}],
    }
    path = tmp_path / "code_full.json"
    path.write_text(json.dumps(state))
    result = runner.summary(tmp_path)["arms"]["code_full"]
    assert result["planned_tasks"] == 2 and result["valid_loss_scores"] == 1
    assert result["evaluation_coverage"] == .5
    assert result["mean_loss"] is None
    assert result["observed_mean_loss"] == .25
    assert result["evaluation_status_counts"] == {"completed": 1, "missing_model": 1}
    assert result["result_status_counts"]["model_call_budget_exhausted"] == 1

    state["evaluation"][1] = {"status": "completed", "loss_kind": "pinball", "loss": .75}
    path.write_text(json.dumps(state))
    result = runner.summary(tmp_path)["arms"]["code_full"]
    assert result["evaluation_coverage"] == 1.
    assert result["mean_loss"] == result["observed_mean_loss"] == .5
