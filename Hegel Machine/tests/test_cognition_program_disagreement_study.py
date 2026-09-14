"""The comparison uses distinct apparatus cohorts and immutable runtime arms."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_program_disagreement_study.py"
SPEC = importlib.util.spec_from_file_location("program_disagreement_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_fresh_streams_preserve_budget_and_evaluation_contract(tmp_path):
    protocol = study.prepare(tmp_path)
    assert protocol["planned_tasks"] == 24
    assert protocol["arms"] == ["v068", "directed_programs"]
    assert protocol["research_config"]["observation_budget"] == 512
    assert protocol["research_config"]["compute_budget"] == 480000
    assert protocol["measurement_budget"] == 6144
    assert protocol["evaluation"]["length"] == 64
    assert protocol["evaluation"]["trajectories"] == 4
    assert protocol["deterministic_solved_threshold"] == 1e-6
    assert {tasks[0]["apparatus_seed"] for tasks in protocol["tasks_by_seed"].values()} == {
        41011, 42013, 43003}
    for tasks in protocol["tasks_by_seed"].values():
        assert [task["readout"] for task in tasks] == [
            "focus", "position_focus", "position_focus", "event"]
        assert len({task["name_seed"] for task in tasks}) == 1
        assert len({task["wiring_seed"] for task in tasks}) == 1
        assert len({task["apparatus_seed"] for task in tasks}) == 1


def test_complete_arm_snapshot_is_not_overwritten_by_later_source_changes(tmp_path):
    source = tmp_path / "original"
    (source / "hegel_machine/cognition").mkdir(parents=True)
    (source / "hegel_machine/__init__.py").write_text('__version__ = "old"\n')
    module = source / "hegel_machine/cognition/program.py"
    module.write_text("value = 1\n")
    snapshot = study.freeze(tmp_path / "study", "v068", source)
    module.write_text("value = 2\n")
    assert (snapshot / "hegel_machine/cognition/program.py").read_text() == "value = 1\n"
    assert (snapshot / "hegel_machine/__init__.py").read_text() == '__version__ = "old"\n'
    with pytest.raises(ValueError, match="already prepared"):
        study.freeze(tmp_path / "study", "v068", source)
