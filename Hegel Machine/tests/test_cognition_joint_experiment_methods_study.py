"""The study must test method transfer after a mechanism and channel change."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_joint_experiment_methods_study.py"
SPEC = importlib.util.spec_from_file_location("joint_experiment_methods_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_transfer_stream_changes_dynamics_and_channels_without_inflating_budget(tmp_path):
    protocol = study.prepare(tmp_path)
    assert protocol["planned_tasks"] == 30
    assert protocol["arms"] == ["v069", "joint_methods"]
    assert protocol["research_config"]["observation_budget"] == 512
    assert protocol["research_config"]["compute_budget"] == 480000
    assert protocol["measurement_budget"] == 6144
    assert protocol["evaluation"] == {"length": 64, "trajectories": 4, "seed_offset": 1000003}
    assert protocol["deterministic_solved_threshold"] == 1e-6
    plants = set()
    for tasks in protocol["tasks_by_seed"].values():
        assert [task["readout"] for task in tasks] == [
            "focus", "position_focus", "position_focus", "position_focus", "event"]
        assert tasks[0]["apparatus_seed"] == tasks[1]["apparatus_seed"]
        assert len({tasks[index]["apparatus_seed"] for index in (0, 2, 3)}) == 3
        assert tasks[3]["apparatus_seed"] == tasks[4]["apparatus_seed"]
        assert tasks[1]["name_seed"] == tasks[2]["name_seed"] != tasks[3]["name_seed"]
        assert tasks[1]["wiring_seed"] == tasks[2]["wiring_seed"] != tasks[3]["wiring_seed"]
        assert tasks[3]["name_seed"] == tasks[4]["name_seed"]
        assert tasks[3]["wiring_seed"] == tasks[4]["wiring_seed"]
        plants.update(task["apparatus_seed"] for task in tasks)
    assert len(plants) == 9


def test_changed_apparatus_is_not_exposed_as_a_learner_label():
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    tasks = study.tasks_for(107)
    labs = [AnonymousObjectLab(**{key: value for key, value in task.items()
             if key != "environment"}) for task in tasks]
    try:
        # The same public command plan produces different actual observations,
        # while the learner sees identical names/ranges for the first change.
        assert labs[1].spec == labs[2].spec
        probe = labs[1].random_probe(117, length=8)
        before = labs[1].observe(probe, ("fine",))
        after = labs[2].observe(probe, ("fine",))
        assert any(a.target != b.target for a, b in zip(before, after))
        assert not labs[2].spec.context
        assert set(labs[2].spec.feature_names).isdisjoint(labs[3].spec.feature_names)
        assert labs[3].spec == labs[4].spec
    finally:
        for lab in labs:
            lab.close()
