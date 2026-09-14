import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_revisable_inputs_study.py"
SPEC = importlib.util.spec_from_file_location("revisable_inputs_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_relation_and_joint_tasks_keep_the_original_eleven_task_prefix():
    tasks = study.tasks_for(71)
    assert tasks[:11] == study.instrument_tasks(71)
    assert len(tasks) == 15
    assert len({task["seed"] for task in tasks}) == 15
    assert all(task["apparatus_seed"] == 9013 for task in tasks[1:])
    assert [task["variant"] for task in tasks[11:]] == ["sink", "sink", "nonlocal", "nonlocal"]
    assert [task["actuator_curvature"] for task in tasks[10:]] == [.31, .31, .31, .47, .47]
    assert [task["readout"] for task in tasks[11:]] == [
        "total_event", "window_total_event", "total_event", "window_total_event"]
    assert all(tasks[index]["rename_prefix"] for index in (12, 14))
    config = json.loads((SCRIPT.parents[1] / "config/cognition_revisable_inputs.json").read_text())
    assert config["tasks"] == tasks
    assert config["config"]["seed"] == 71


def test_relation_change_cannot_be_explained_by_a_changed_input_instrument():
    exchange = UnifiedObjectLab(apparatus_seed=9013, readout="total_event", actuator_curvature=.31)
    sink = UnifiedObjectLab(apparatus_seed=9013, readout="total_event", variant="sink", actuator_curvature=.31)
    nonlocal_same_input = UnifiedObjectLab(apparatus_seed=9013, readout="total_event",
                                         variant="nonlocal", actuator_curvature=.31)
    joint = UnifiedObjectLab(apparatus_seed=9013, readout="total_event",
                             variant="nonlocal", actuator_curvature=.47)
    probe = exchange.random_probe(3127, length=12)
    batches = [lab.acquire(probe, ("fine",)) for lab in (exchange, sink, nonlocal_same_input, joint)]
    features = [[row.features for row in batch.observations] for batch in batches]
    targets = [[row.target for row in batch.observations] for batch in batches]
    assert features[0] == features[1] == features[2]
    assert features[2] != features[3]
    assert targets[0] != targets[1]
    assert targets[1] != targets[2]
    assert targets[2] != targets[3]
    for lab in (sink, nonlocal_same_input, joint):
        assert lab.channels == exchange.channels
        for channels in exchange.measurement_views:
            assert lab.spec_for(channels) == exchange.spec_for(channels)
    for point, row in zip(probe.sequence, batches[-1].observations):
        for index in range(4):
            command = point[f"q_o{index}"]
            assert row.features[f"measured_q_o{index}"] == pytest.approx(command + .47 * command ** 3)


@pytest.mark.parametrize("variant,curvature", [("sink", .31), ("nonlocal", .47)])
def test_new_relation_history_and_calibration_survive_fresh_evaluation_and_restore(variant, curvature):
    lab = UnifiedObjectLab(apparatus_seed=9013, readout="window_total_event", variant=variant,
                           actuator_curvature=curvature, rename_prefix="other_")
    probe = lab.random_probe(3137, length=9)
    batch = lab.acquire(probe, ("fine",))
    assert not batch.observations[0].history
    assert lab.project_rows(batch.observations, ("fine",), ("coarse",)) == lab.observe(probe, ("coarse",))
    cost = dict(lab.costs)
    assert lab.evaluation_rows(("fine",), seed=3137, length=9, trajectories=1) == batch.observations
    assert lab.costs == cost
    lab.restore(json.loads(json.dumps(lab.acquisitions)))
    assert lab.costs == cost
    assert lab.acquire(probe, ("fine",)) == batch


def test_current_revision_arms_count_all_fifteen_learning_and_terminal_bills(tmp_path):
    tasks = study.tasks_for(71)
    protocol = study.prepare(tmp_path, seeds=[71])
    assert protocol["planned_new_tasks"] == protocol["planned_logical_tasks"] == 30
    assert "prefixes" not in protocol
    costs = dict(zip(study.COST_KEYS, (3, 2, 7)))
    research = {"status": "completed", "costs": costs, "selected_channels": [],
                "selected_kind": "test_model", "actions": [{"action": "execute_research_programs",
                    "alternatives": [{"reused_procedure": True, "training_loss": .2}]}],
                "research_program": {"kind": "residual_composition", "source_ids": ["concept_1"]}}
    scores = [{"task_position": index, "observation_cost": 256,
               "measurement_cost": 12, "compute_cost": 256, "loss": 0.} for index in range(15)]
    state = {"tasks": tasks, "results": [{"research": research} for _ in range(15)],
        "evaluation": scores, "knowledge": {"meta_priors": []},
        "costs": {key: 15 * value for key, value in costs.items()}}
    runs = tmp_path / "runs"
    runs.mkdir()
    for arm in study.ARMS:
        (runs / f"71_{arm}.json").write_text(json.dumps(state))
    summary = study.summarize(tmp_path, protocol)
    assert summary["actual_new_learning_costs"] == {key: 30 * value for key, value in costs.items()}
    assert summary["actual_new_terminal_evaluation_costs"] == dict(zip(study.COST_KEYS, (7680, 360, 7680)))
    for arm in study.ARMS:
        assert summary["arms"][arm]["all"]["solved"] == 15
        assert summary["arms"][arm]["calibration_only"]["tasks"] == 3
        assert summary["arms"][arm]["relation_only"]["tasks"] == 2
        assert summary["arms"][arm]["joint_change"]["tasks"] == 2
    first = summary["runs"][0]["tasks"][0]
    assert first["replayed_program_candidates"] == 1
    assert first["selected_research_program"]["reused_procedure"] is False
