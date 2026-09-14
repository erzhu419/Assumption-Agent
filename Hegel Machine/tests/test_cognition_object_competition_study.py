"""The new readout needs observation evidence, not a changed physical role."""

import importlib.util
from pathlib import Path

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.models import Probe


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_object_competition_study.py"
SPEC = importlib.util.spec_from_file_location("object_competition_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def laboratory(task):
    return AnonymousObjectLab(**{key: value for key, value in task.items() if key != "environment"})


def test_changed_readout_preserves_public_interface_and_physical_transitions(tmp_path):
    tasks = study.tasks_for(97)
    before, changed = (laboratory(task) for task in tasks[:2])
    assert [task["readout"] for task in tasks] == ["focus", "position_focus", "position_focus", "event"]
    for view in before.measurement_views:
        assert before.spec_for(view) == changed.spec_for(view)
        assert before.spec_for(view).context == {}
    probe = before.random_probe(7301, length=12)
    original, observed = (lab.acquire(probe, ("fine",)) for lab in (before, changed))
    assert original.transitions == observed.transitions
    assert [row.features for row in original.observations] == [row.features for row in observed.observations]
    assert any(abs(old.target - new.target) > .1 for old, new in zip(original.observations, observed.observations))
    assert original.measurement_cost == observed.measurement_cost
    protocol = study.prepare(tmp_path, seeds=[97])
    assert protocol["planned_tasks"] == 8
    assert protocol["research_config"]["observation_budget"] == 512
    assert protocol["research_config"]["compute_budget"] == 480000
    assert protocol["measurement_budget"] == 6144
    assert len({study.tasks_for(seed)[0]["apparatus_seed"] for seed in study.SEEDS}) == 3


def test_equal_position_evidence_can_hide_membership_but_independent_intervention_exposes_it():
    lab = laboratory(study.tasks_for(97)[1])
    schema = lab._evaluation_schema()
    left, right = schema.objects
    first = lab.random_probe(7703, length=1).sequence[0]
    xleft, xright = (schema.controls[obj]["x"] for obj in (left, right))
    points = tuple({**first, xleft: value, xright: value} for value in (-1., -.5, .5, 1.))
    old = lab.acquire(Probe("ordinary_linked_positions", points), ("fine",))
    assert all(row.features[schema.fields[left]["x"]] == row.features[schema.fields[right]["x"]]
               for row in old.observations)
    one = {**first, xleft: -1., xright: 1., schema.focus_control: 0.}
    two = {**one, xleft: 1., xright: -1.}
    acquired = lab.acquire(Probe("independent_position_intervention", (one, two)), ("fine",))
    for obj in schema.objects:
        q = schema.fields[obj]["q"]
        assert acquired.transitions[0]["after"][q] == acquired.transitions[1]["after"][q]
    assert abs(acquired.observations[0].target - acquired.observations[1].target) > .4
