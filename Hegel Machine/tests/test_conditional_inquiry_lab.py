"""The historical conditional apparatus serves the shared inquiry interface."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path
from random import Random

import pytest

from hegel_machine.cognition.models import Probe


@pytest.fixture
def factory(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("run_conditional_discovery_study").laboratory


def task(stage=2, noise_sigma=0.):
    return {"stage": stage, "seed": 71, "apparatus_seed": 113, "noise_sigma": noise_sigma}


def trajectory(lab, *, length=4):
    return {"kind": "trajectory", "channels": [],
        "sequence": list(lab.random_probe(823, length=length).sequence)}


@pytest.mark.parametrize("stage", range(7))
def test_query_preserves_legacy_physics_paid_costs_and_restore(factory, stage):
    legacy, inquiry = factory(task(stage, .03)), factory(task(stage, .03))
    action = trajectory(inquiry)
    supplied = deepcopy(action)
    for index in range(2):
        original = legacy.acquire(Probe(f"legacy_{index}", tuple(action["sequence"])))
        record = inquiry.execute(action)
        observed = inquiry.observation_rows(record)
        assert [row.target for row in observed] == [row.target for row in original.observations]
        assert [row.features for row in observed] == [row.features for row in original.observations]
        assert record["row_indices"] == list(range(4 * index, 4 * (index + 1)))
        assert record["observation_index"] == 4 * index
        assert record["observation_cost"] == 4 and record["measurement_cost"] == 0
        assert record["measurements"] == []
        assert "__target__" in original.observations[1].history[0]
        assert all(set(point) == set(inquiry.spec.feature_names)
                   for row in observed for point in row.history)
    assert action == supplied
    assert inquiry.costs == legacy.costs == {
        "observation_cost": 8, "measurement_cost": 0, "queries": 2}
    restored = factory(task(stage, .03))
    restored.restore(inquiry.acquisitions)
    assert restored.costs == inquiry.costs
    assert restored.execute(action)["rows"] == inquiry.execute(action)["rows"]


def test_public_interface_does_not_disclose_private_stage_or_parameters(factory):
    labs = [factory({**task(stage), "apparatus_seed": seed})
            for stage in (0, 1, 2, 3, 5, 6) for seed in (113, 823)]
    assert all(lab.public_spec() == labs[0].public_spec() for lab in labs)
    assert all(lab.task_interface == labs[0].task_interface for lab in labs)
    assert set(labs[0].public_spec()) == {"controls", "feature_bounds", "action_kinds",
        "channels", "max_probe_length", "observation_cost_per_row", "reset", "prediction_contract"}


@pytest.mark.parametrize("stage", [2, 4])
def test_prospective_rows_have_no_labels_costs_or_physics_access(factory, stage, monkeypatch):
    lab = factory(task(stage))
    action, before = trajectory(lab), lab.costs

    def unavailable(*args, **kwargs):
        raise AssertionError("forecast inputs must not acquire labels")

    with monkeypatch.context() as context:
        context.setattr(lab, "acquire", unavailable)
        rows = lab.prediction_rows(action)
    assert lab.costs == before
    assert all(row.target == 0. for row in rows)
    assert rows[0].history == ()
    for index, row in enumerate(rows):
        assert row.history == tuple(action["sequence"][:index])
    measured = lab.observation_rows(lab.execute(action))
    assert [(row.features, row.history) for row in rows] == [
        (row.features, row.history) for row in measured]
    rows[-1].history[0][lab.spec.feature_names[0]] = 999
    assert lab.prediction_rows(action)[-1].history[0] == action["sequence"][0]


def test_renamed_controls_and_public_bounds_are_respected(factory):
    lab = factory(task(4))
    public = lab.public_spec()
    assert public["controls"] == ["p", "q", "r"]
    assert all(bounds == pytest.approx([-3.45, 3.45])
               for bounds in public["feature_bounds"].values())
    actions = lab.confirmation_actions(3, Random(19))
    assert all(set(point) == {"p", "q", "r"} for point in actions[0]["sequence"])
    assert len(lab.execute(actions[0])["rows"]) == 3
    with pytest.raises(ValueError, match="exactly the public controls"):
        lab.execute({"kind": "trajectory", "sequence": [{"x": 0., "z": 0., "c": 0.}]})
    assert lab.costs["observation_cost"] == 3


def test_resets_full_horizon_and_separate_terminal_evaluation(factory):
    lab = factory(task())
    action = trajectory(lab, length=64)
    first = lab.execute(action)
    lab.execute(trajectory(lab, length=2))
    repeated = lab.execute(action)
    assert first["rows"] == repeated["rows"]
    assert first["rows"][0]["history"] == ()
    assert len(first["rows"][-1]["history"]) == 63
    actions = lab.confirmation_actions(130, Random(149))
    assert [lab.action_cost(item) for item in actions] == [64, 64, 2]
    before = lab.costs
    terminal = lab.evaluation_rows(length=130, seed=947)
    assert len(terminal) == 130
    assert [index for index, row in enumerate(terminal) if not row.history] == [0, 64, 128]
    assert terminal == lab.evaluation_rows(length=130, seed=947)
    assert terminal != lab.evaluation_rows(length=130, seed=953)
    assert all(set(point) == set(lab.spec.feature_names) for row in terminal for point in row.history)
    assert lab.costs == before


@pytest.mark.parametrize("change", ["empty", "too_long", "kind", "channel", "bounds"])
def test_invalid_public_action_does_not_purchase(factory, change):
    lab = factory(task())
    action = trajectory(lab)
    if change == "empty":
        action["sequence"] = []
    elif change == "too_long":
        action = trajectory(lab, length=65)
    elif change == "kind":
        action["kind"] = "response"
    elif change == "channel":
        action["channels"] = ["meter"]
    else:
        action["sequence"][0]["x"] = 1.6
    with pytest.raises(ValueError):
        lab.execute(action)
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}
