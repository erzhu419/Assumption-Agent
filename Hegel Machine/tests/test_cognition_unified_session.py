"""Cross-task knowledge must survive interruption before separate scoring."""

import json
from dataclasses import asdict

import pytest

from hegel_machine.cli import main
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_session import UnifiedResearchSession, task_sequence
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


def config():
    return ResearchConfig(observation_budget=192, compute_budget=120000,
                          minimum_improvement=1e-7, max_rounds=4, seed=347)


def test_real_joint_knowledge_survives_pause_and_is_called_with_renamed_measurements():
    opened, evaluated, checkpoints = [], [], []

    class RecordedLab(UnknownTaskLab):
        def evaluation_rows(self, *args, **kwargs):
            evaluated.append(self.task_index)
            return super().evaluation_rows(*args, **kwargs)

    def factory(item):
        opened.append(item["task_index"])
        return RecordedLab(**item)

    session = UnifiedResearchSession(config(), tasks=task_sequence(347, (0, 2, 3)),
                                     lab_factory=factory)
    first = session.run(max_tasks=1, on_checkpoint=checkpoints.append)
    assert first["status"] == "paused" and first["pending_tasks"] == [1, 2]
    assert opened == [0] and evaluated == []
    assert first["evaluation"] == []
    assert first["knowledge"]["definitions"]
    assert checkpoints == [first]
    with pytest.raises(ValueError, match="finish learning"):
        session.evaluate()
    assert evaluated == [] and session.to_dict() == first

    persisted = json.loads(json.dumps(first))
    resumed = UnifiedResearchSession(state=persisted, lab_factory=factory)
    completed = resumed.run()
    assert completed["status"] == "completed" and completed["pending_tasks"] == []
    # New concepts can open old public interfaces for paid-evidence screening;
    # this is not an acquisition or independent terminal evaluation.
    assert list(dict.fromkeys(opened)) == [0, 2, 3] and evaluated == []
    assert completed["results"][0] == persisted["results"][0]
    assert completed["config"] == first["config"]
    assert {row["concept_id"] for row in first["knowledge"]["concepts"]} <= {
        row["concept_id"] for row in completed["knowledge"]["concepts"]}
    joint, renamed = [row["research"] for row in completed["results"][1:]]
    assert joint["selected_channels"] == ["sensor_a"]
    assert renamed["selected_channels"] == ["channel_p"]
    assert Program.from_dict(joint["program"]).history_depth >= 1
    assert Program.from_dict(renamed["program"]).history_depth >= 1
    assert set(_sources(Program.from_dict(renamed["program"]))) & set(joint["concepts_added"])
    assert set(renamed["used_sources"]) <= set(completed["results"][2]["available_concepts"])
    for key in ("observation_cost", "measurement_cost", "compute_cost"):
        expected = sum(row["research"]["costs"][key] for row in completed["results"])
        if key == "compute_cost":
            expected += completed["screening_compute_cost"]
        assert completed["costs"][key] == expected

    opened_before_resume = list(opened)
    unchanged = resumed.run()
    assert unchanged == completed and opened == opened_before_resume
    scored = resumed.evaluate(length=32, trajectories=2)
    assert evaluated == [0, 2, 3]
    assert all(row["mse"] < 1e-6 and row["rows"] == 64 for row in scored["evaluation"])
    assert scored["costs"] == completed["costs"]
    opened_before, evaluated_before = list(opened), list(evaluated)
    assert resumed.evaluate(length=32, trajectories=2) == scored
    assert opened == opened_before and evaluated == evaluated_before


def test_finished_task_checkpoint_is_available_when_next_apparatus_fails():
    saved = []

    def factory(item):
        if item["task_index"] == 2:
            raise RuntimeError("next apparatus unavailable")
        return UnknownTaskLab(**item)

    session = UnifiedResearchSession(config(), tasks=task_sequence(811, (0, 2)), lab_factory=factory)
    with pytest.raises(RuntimeError, match="next apparatus unavailable"):
        session.run(on_checkpoint=saved.append)
    assert len(saved) == 1
    assert saved[0]["status"] == "paused" and saved[0]["pending_tasks"] == [1]
    assert len(saved[0]["results"]) == 1 and saved[0]["evaluation"] == []
    assert saved[0]["knowledge"]["concepts"]
    persisted = json.loads(json.dumps(saved[0]))
    restored = UnifiedResearchSession(state=persisted)
    assert restored.to_dict() == persisted


def test_cli_resumes_request_sequence_and_evaluates_only_after_finishing(tmp_path, capsys, monkeypatch):
    checkpoint = tmp_path / "unified.json"
    memory = tmp_path / "shared_knowledge.json"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"config": asdict(config()), "measurement_budget": 768,
                                  "tasks": task_sequence(991, (0, 2))}))
    args = ["cognition", "unified", "--checkpoint", str(checkpoint), "--memory", str(memory)]
    assert main([*args, "--request", str(request), "--steps", "1"]) == 0
    first_summary = json.loads(capsys.readouterr().out)
    first = json.loads(checkpoint.read_text())
    assert first_summary["status"] == "paused" and first_summary["pending_tasks"] == [1]
    assert first_summary["evaluation"] == [] and first["evaluation"] == []
    assert first["knowledge"]["definitions"]
    assert json.loads(memory.read_text()) == first["knowledge"]

    assert main([*args, "--evaluate"]) == 0
    summary = json.loads(capsys.readouterr().out)
    completed = json.loads(checkpoint.read_text())
    assert summary["status"] == "completed" and summary["pending_tasks"] == []
    assert completed["results"][0] == first["results"][0]
    assert completed["results"][1]["research"]["selected_channels"] == ["sensor_a"]
    assert completed["config"] == first["config"]
    assert all(row["rows"] == 256 and row["mse"] < 1e-6 for row in completed["evaluation"])
    assert summary["costs"] == completed["costs"]
    assert json.loads(checkpoint.with_name("unified_summary.json").read_text()) == summary
    assert json.loads(memory.read_text()) == completed["knowledge"]

    def no_new_experiment(*args, **kwargs):
        raise AssertionError("completed session repeated an experiment")

    with monkeypatch.context() as patch:
        patch.setattr(UnknownTaskLab, "acquire", no_new_experiment)
        assert main([*args, "--evaluate"]) == 0
        assert json.loads(capsys.readouterr().out) == summary
        assert json.loads(checkpoint.read_text()) == completed

    next_checkpoint = tmp_path / "successor.json"
    request.write_text(json.dumps({"config": asdict(config()), "measurement_budget": 768,
                                  "tasks": task_sequence(1087, (3,))}))
    assert main(["cognition", "unified", "--checkpoint", str(next_checkpoint),
                 "--memory", str(memory), "--request", str(request), "--evaluate"]) == 0
    capsys.readouterr()
    successor = json.loads(next_checkpoint.read_text())
    prior_ids = {concept["concept_id"] for concept in completed["knowledge"]["concepts"]}
    assert set(successor["results"][0]["available_concepts"]) == prior_ids
    inherited = successor["results"][0]["research"]
    assert inherited["selected_channels"] == ["channel_p"]
    assert set(inherited["used_sources"]) & set(completed["results"][1]["research"]["concepts_added"])
    assert successor["evaluation"][0]["mse"] < 1e-6
    assert json.loads(memory.read_text()) == successor["knowledge"]
