"""Finished laboratory research survives CLI pause/resume with actual costs."""

import json

import pytest

from hegel_machine.cli import main
from hegel_machine.cognition import measurement_session
from hegel_machine.cognition.measurement_session import MeasurementSession
from hegel_machine.cognition.models import ResearchConfig


def small_config():
    return ResearchConfig(observation_budget=32, compute_budget=12000, max_rounds=1, seed=313)


def test_cli_process_pause_resume_keeps_learning_and_never_repeats_finished_tasks(tmp_path, capsys, monkeypatch):
    checkpoint = tmp_path / "lab.json"
    args = ["cognition", "lab", "--checkpoint", str(checkpoint)]
    assert main([*args, "--tasks", "0", "2", "--steps", "1", "--observation-budget", "32",
                 "--compute-budget", "12000", "--measurement-budget", "160", "--rounds", "1",
                 "--evaluation-trajectories", "2"]) == 0
    first_summary = json.loads(capsys.readouterr().out)
    first = json.loads(checkpoint.read_text())
    assert first_summary["status"] == "paused"
    assert first["backend"] == "process" and first["completed_tasks"] == [0] and first["pending_tasks"] == [2]
    assert first["knowledge"]["concepts"] and first["knowledge"]["definitions"]
    assert first["results"][0]["selected_channels"] == ["sensor_a"]
    assert first["results"][0]["evaluation"]["rows"] == 128
    assert first["results"][0]["evaluation"]["mse"] < 1e-6

    actual_factory = measurement_session.make_lab
    requested_tasks = []

    def record_factory(task_index, seed, *, backend):
        requested_tasks.append(task_index)
        return actual_factory(task_index, seed, backend=backend)

    monkeypatch.setattr(measurement_session, "make_lab", record_factory)
    output = tmp_path / "report.json"
    assert main([*args, "--resume", "--steps", "1", "--output", str(output)]) == 0
    resumed_summary = json.loads(capsys.readouterr().out)
    resumed = json.loads(checkpoint.read_text())
    assert requested_tasks == [2]
    assert resumed["status"] == "completed" and resumed["completed_tasks"] == [0, 2]
    assert resumed["results"][0] == first["results"][0]
    assert resumed["config"] == first["config"]
    old_ids = {concept["concept_id"] for concept in first["knowledge"]["concepts"]}
    assert old_ids <= {concept["concept_id"] for concept in resumed["knowledge"]["concepts"]}
    for kind, cap in (("observation", 32), ("measurement", 160), ("compute", 12000)):
        assert resumed[f"total_{kind}_cost"] == sum(row["costs"][f"{kind}_cost"] for row in resumed["results"])
        assert all(row["costs"][f"{kind}_cost"] <= cap for row in resumed["results"])
        assert resumed_summary[f"total_{kind}_cost"] == resumed[f"total_{kind}_cost"]
    assert json.loads(output.read_text()) == resumed
    assert json.loads(output.with_name("report_summary.json").read_text())["results"][0]["evaluation_mse"] < 1e-6

    requested_tasks.clear()
    assert main([*args, "--resume"]) == 0
    capsys.readouterr()
    assert requested_tasks == []
    assert json.loads(checkpoint.read_text()) == resumed


@pytest.mark.parametrize("strategy,channels", [("learned", ["sensor_a"]), ("controls_only", []),
                                               ("all", ["sensor_a", "sensor_b"])])
def test_strategy_results_save_paid_pilots_and_new_measurements_separately_from_evaluation(strategy, channels):
    saved = []
    session = MeasurementSession(small_config(), tasks=[0], backend="local", strategy=strategy,
                                 measurement_budget=160, evaluation_trajectories=2)
    report = session.run(on_checkpoint=saved.append)
    result = report["results"][0]
    assert result["selected_channels"] == channels
    pilot_rows = [row for trajectory in result["pilot_observations"] for row in trajectory]
    rates = {"sensor_a": 1, "sensor_b": 4}
    pilot_units = sum(sum(rates.get(name, 0) for name in row["features"]) for row in pilot_rows)
    new_rows = result["episode"]["observation_cost"]
    assert result["costs"]["observation_cost"] == len(pilot_rows) + new_rows <= 32
    assert result["costs"]["measurement_cost"] == pilot_units + new_rows * sum(rates[name] for name in channels)
    assert result["costs"]["measurement_cost"] <= 160
    assert 0 < result["costs"]["compute_cost"] <= 12000
    assert result["evaluation"]["rows"] == 128
    assert result["evaluation"]["compute_cost"] == 128
    assert saved[-1] == report
    assert saved[-1]["knowledge"] == session.library.to_dict()
    if strategy == "controls_only":
        assert result["evaluation"]["mse"] > .05
    elif strategy == "learned":
        assert result["evaluation"]["mse"] < 1e-6
    else:
        assert result["evaluation"]["mse"] < .05


def test_each_completed_task_is_saved_before_next_apparatus_opens(monkeypatch):
    saved = []
    factory = measurement_session.make_lab

    def fail_second(task_index, seed, *, backend):
        if task_index == 2:
            raise RuntimeError("apparatus unavailable for the next task")
        return factory(task_index, seed, backend=backend)

    monkeypatch.setattr(measurement_session, "make_lab", fail_second)
    session = MeasurementSession(small_config(), tasks=[0, 2], backend="local", measurement_budget=160)
    with pytest.raises(RuntimeError, match="next task"):
        session.run(on_checkpoint=saved.append)
    assert len(saved) >= 2 and saved[0]["active_task"]["task_index"] == 0
    assert saved[0]["active_task"]["result"]["research_state"]["stage"] == "research"
    assert not saved[0]["completed_tasks"] and saved[0]["pending_tasks"] == [0, 2]
    completed = saved[-1]
    assert completed["completed_tasks"] == [0] and completed["active_task"] is None
    assert completed["pending_tasks"] == [2] and completed["knowledge"]["concepts"]
    restored = MeasurementSession(state=json.loads(json.dumps(completed)))
    assert restored.library.to_dict() == completed["knowledge"]
    assert restored.state["results"] == json.loads(json.dumps(completed["results"]))


def test_measurement_stage_extends_existing_knowledge_and_restores_it():
    first = MeasurementSession(small_config(), tasks=[0], backend="local",
                               measurement_budget=160, evaluation_trajectories=1)
    first.run()
    library = first.library
    previous_ids = {item["concept_id"] for item in library.to_dict()["concepts"]}
    assert previous_ids
    next_stage = MeasurementSession(small_config(), tasks=[2], backend="local",
                                    library=library, measurement_budget=160,
                                    evaluation_trajectories=1)
    report = next_stage.run()
    assert next_stage.library is library
    assert previous_ids < {item["concept_id"] for item in report["knowledge"]["concepts"]}
    restored = MeasurementSession(state=json.loads(json.dumps(report)))
    assert restored.library.to_dict() == report["knowledge"]
    assert report["results"][0]["evaluation"]["mse"] < 1e-6
