import importlib.util
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.structure_lab import ObjectSchema


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_object_induction_study.py"
SPEC = importlib.util.spec_from_file_location("object_induction_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def laboratory(task):
    return AnonymousObjectLab(**{key: value for key, value in task.items() if key != "environment"})


def test_task_stream_has_true_same_interface_rewiring_then_renamed_successors(tmp_path):
    tasks = study.tasks_for(71)
    assert [task["readout"] for task in tasks] == [
        "focus", "total", "event", "total_event", "window_total_event",
        "focus", "window_total_event", "focus", "window_total_event"]
    before, rewired, renamed = (laboratory(tasks[index]) for index in (0, 5, 7))
    for channels in before.measurement_views:
        assert before.spec_for(channels) == rewired.spec_for(channels)
        assert before.spec_for(channels).feature_names != renamed.spec_for(channels).feature_names
    assert before._evaluation_schema() != rewired._evaluation_schema()
    protocol = study.prepare(tmp_path, seeds=[71], task_count=2)
    assert protocol["tasks_by_seed"]["71"] == tasks[:2]
    assert protocol["planned_positive_tasks"] == 4
    assert protocol["negative_tasks_by_seed"]["71"][0]["readout"] == "total"
    assert protocol["research_config"]["observation_budget"] == 512
    assert protocol["research_config"]["compute_budget"] == 480000
    assert protocol["measurement_budget"] == 6144


def test_recovery_ignores_object_id_order_but_detects_wrong_focus_pairing():
    truth = AnonymousObjectLab(objects=3)._evaluation_schema()
    rename = {obj: f"learned_{index}" for index, obj in enumerate(reversed(truth.objects))}
    guessed = ObjectSchema(tuple(rename[obj] for obj in reversed(truth.objects)),
        {rename[obj]: dict(truth.fields[obj]) for obj in truth.objects},
        {rename[obj]: (rename[obj],) for obj in truth.objects},
        {rename[obj]: dict(truth.controls[obj]) for obj in truth.objects}, truth.focus_control)
    assert study.object_recovery(guessed, truth)["exact"]
    first, second = guessed.objects[:2]
    guessed.fields[first]["f"], guessed.fields[second]["f"] = guessed.fields[second]["f"], guessed.fields[first]["f"]
    wrong = study.object_recovery(guessed, truth)
    assert not wrong["exact"]
    assert wrong["complete_objects"] == wrong["correct_focus_pairs"] == 1
    unresolved = study.object_recovery(None, truth)
    assert not unresolved["exact"]
    assert unresolved["proposed_objects"] == 0


def test_object_evaluation_uses_each_tasks_actual_schema_snapshot():
    tasks = [study.tasks_for(71)[index] for index in (0, 5)]
    report = {"tasks": tasks, "results": [{"task_position": index,
        "research": {"learned_object_schema": laboratory(task)._evaluation_schema().to_dict()}}
        for index, task in enumerate(tasks)]}
    assert all(row["exact"] for row in study.evaluate_objects(report)["object_evaluation"])
    report["results"][0]["research"]["learned_object_schema"] = report["results"][1]["research"]["learned_object_schema"]
    scores = study.evaluate_objects(report)["object_evaluation"]
    assert not scores[0]["exact"]
    assert scores[1]["exact"]


def test_total_negative_stays_separate_from_positive_prediction_coverage(tmp_path):
    protocol = study.prepare(tmp_path, seeds=[71], arms=["default"], task_count=2)
    costs = dict(zip(study.COST_KEYS, (3, 2, 7)))
    research = {"status": "completed", "costs": costs, "selected_channels": [],
                "selected_kind": "test_model", "actions": [], "learned_object_schema": None}
    def report(tasks):
        return {"tasks": tasks, "results": [{"task_position": index, "research": research}
            for index in range(len(tasks))], "evaluation": [{"task_position": index, "loss": 0.,
            "observation_cost": 256, "measurement_cost": 0, "compute_cost": 256}
            for index in range(len(tasks))]}
    for directory, key in (("runs", "tasks_by_seed"), ("negative_runs", "negative_tasks_by_seed")):
        path = tmp_path / directory
        path.mkdir()
        state = study.evaluate_objects(report(protocol[key]["71"]))
        (path / "71_default.json").write_text(json.dumps(state))
    summary = study.summarize(tmp_path, protocol)
    assert summary["arms"]["default"]["all"]["solved"] == 2
    assert summary["arms"]["default"]["all"]["tasks"] == 2
    assert len(summary["negative_runs"]) == 1
    assert summary["negative_runs"][0]["tasks"][0]["object_recovery"]["exact"] is False
