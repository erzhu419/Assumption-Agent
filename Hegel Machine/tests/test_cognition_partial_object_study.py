import importlib.util
import json
from pathlib import Path

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_partial_object_study.py"
SPEC = importlib.util.spec_from_file_location("partial_object_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def laboratory(task):
    return AnonymousObjectLab(**{key: value for key, value in task.items() if key != "environment"})


def test_partial_first_stream_changes_dependency_without_changing_public_interface(tmp_path):
    tasks = study.tasks_for(71)
    assert [task["readout"] for task in tasks] == [
        "total", "total", "focus", "event", "total_event", "window_total_event",
        "focus", "window_total_event"]
    before, changed = (laboratory(tasks[index]) for index in (2, 6))
    for channels in before.measurement_views:
        assert before.spec_for(channels) == changed.spec_for(channels)
    assert before._evaluation_schema() == changed._evaluation_schema()
    probe = before.random_probe(881, length=12)
    assert [row.target for row in before.observe(probe)] != [row.target for row in changed.observe(probe)]
    protocol = study.prepare(tmp_path, seeds=[71])
    assert protocol["planned_tasks"] == 16
    assert protocol["research_config"]["observation_budget"] == 512
    assert protocol["research_config"]["compute_budget"] == 480000
    assert protocol["measurement_budget"] == 6144
    assert [study.tasks_for(seed)[0]["objects"] for seed in study.SEEDS] == [3, 4, 3]
    assert {study.tasks_for(seed)[0]["apparatus_seed"] for seed in study.SEEDS} == {13007, 14009, 15013}


def partial_report(task):
    truth = laboratory(task)._evaluation_schema()
    groups = [{"object_id": f"learned_{index}",
        **{role: truth.fields[obj][role] for role in ("q", "x", "u")},
        "controls": dict(truth.controls[obj])} for index, obj in enumerate(truth.objects)]
    research = {"learned_object_schema": None,
        "object_research": {"groups": groups, "focus_control": truth.focus_control,
                            "unresolved": [f"learned_{index}:f" for index in range(len(groups))]}}
    return {"tasks": [task], "results": [{"task_position": 0, "research": research}]}


def test_partial_role_scoring_keeps_correct_knowledge_without_claiming_membership():
    report = partial_report(study.tasks_for(71)[0])
    row = study.score_objects(report)["object_evaluation"][0]
    assert row["correct_known_roles"] == row["proposed_known_roles"] == 18
    assert not row["exact"] and not row["adopted_schema"]
    assert row["complete_objects"] == 0 and len(row["unresolved"]) == 3
    groups = report["results"][0]["research"]["object_research"]["groups"]
    groups[0]["x"], groups[1]["x"] = groups[1]["x"], groups[0]["x"]
    assert study.score_objects(report)["object_evaluation"][0]["correct_known_roles"] == 16


def test_summary_counts_prediction_success_independently_of_full_object_recovery(tmp_path):
    protocol = study.prepare(tmp_path, seeds=[71])
    report = partial_report(protocol["tasks_by_seed"]["71"][0])
    costs = dict(zip(study.COST_KEYS, (24, 72, 160)))
    report["results"][0]["research"].update(status="completed", costs=costs,
        selected_kind="partial_total", selected_channels=[], actions=[])
    report["evaluation"] = [{"task_position": 0, "loss": 0.,
        "observation_cost": 256, "measurement_cost": 0, "compute_cost": 256}]
    study.score_objects(report)
    path = tmp_path / "runs/71_partial_objects.json"
    path.parent.mkdir()
    path.write_text(json.dumps(report))
    summary = study.summarize(tmp_path, protocol)
    assert summary["arms"]["partial_objects"]["partial"]["solved"] == 1
    assert not summary["runs"][0]["tasks"][0]["object_evaluation"]["exact"]
