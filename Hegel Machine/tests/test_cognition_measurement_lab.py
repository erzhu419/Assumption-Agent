"""Instrument choice changes usable evidence, cost and attainable prediction."""

import json
import os
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, make_lab
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.synthesis import _fit_linear


def experiment(name, length=32, seed=7):
    random = Random(seed)
    return Probe(name, tuple({"u": random.uniform(-1., 1.)} for _ in range(length)))


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_instruments_change_available_evidence_and_have_separate_real_costs():
    lab = make_lab()
    probe = experiment("first", length=5)
    control_only = lab.acquire(probe)
    measured = lab.acquire(probe, ("sensor_a",))
    both = lab.acquire(experiment("second", length=3), ("sensor_a", "sensor_b"))

    assert [row.target for row in control_only.observations] == [row.target for row in measured.observations]
    for batch, names in ((control_only, {"u"}), (measured, {"u", "sensor_a"}),
                         (both, {"u", "sensor_a", "sensor_b"})):
        assert batch.observations[0].history == ()
        for index, row in enumerate(batch.observations):
            assert set(row.features) == names
            assert len(row.history) == index
            assert row.history == tuple({**previous.features, "__target__": previous.target}
                                        for previous in batch.observations[:index])
    assert measured.observation_cost == measured.measurement_cost == 5
    assert both.observation_cost == 3 and both.measurement_cost == 15
    assert lab.costs == {"observation_cost": 13, "measurement_cost": 20, "queries": 3}
    assert MeasurementBatch.from_dict(json.loads(json.dumps(measured.to_dict()))) == measured


def test_public_schema_exposes_instruments_but_only_controls_are_intervenable():
    lab = make_lab()
    spec = lab.spec_for(("sensor_b", "sensor_a"))
    assert spec.feature_names == ("u", "sensor_b", "sensor_a")
    assert spec.feature_bounds == {"u": (-1., 1.)}
    assert all(set(point) == {"u"} for probe in spec.initial_probes for point in probe.sequence)
    assert [channel.cost_per_row for channel in lab.channels] == [1, 4]
    assert spec.context == {}
    with pytest.raises(ValueError, match="controllable input"):
        lab.observe(Probe("cannot-control-sensor", ({"u": 0., "sensor_a": 0.},)))
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}


def test_useful_sensor_removes_unpredictable_innovation_while_expensive_sensor_does_not():
    lab = make_lab(task_index=0)
    training = []
    for index in range(4):
        training.extend(lab.observe(experiment(f"train_{index}", 64, index), ("sensor_a", "sensor_b")))
    training = [row for row in training if len(row.history) >= 4]
    current = Program("feature", name="u")
    # Give the baseline four previous controls and responses. The new physical
    # disturbance is still unavailable until the useful instrument is selected.
    baseline_terms = [current] + [Program("lag", name=name, lag=lag)
                                 for lag in range(1, 5) for name in ("u", "__target__")]
    old = _fit_linear(baseline_terms, training)
    unrelated = _fit_linear([*baseline_terms, Program("feature", name="sensor_b")], training)
    measured = _fit_linear([*baseline_terms, Program("feature", name="sensor_a")], training)
    fresh = [row for row in lab.evaluation_rows(("sensor_a", "sensor_b"), seed=733)
             if len(row.history) >= 4]
    assert mse(old, fresh) > .1
    assert mse(unrelated, fresh) > .1
    assert mse(measured, fresh) < 1e-12


def test_evaluation_is_a_separate_stream_and_does_not_consume_learning_state_or_costs():
    learning, unchanged = make_lab(2, seed=71), make_lab(2, seed=71)
    first, next_probe = experiment("first"), experiment("next", seed=9)
    assert learning.observe(first, ("sensor_a",)) == unchanged.observe(first, ("sensor_a",))
    before = dict(learning.costs)
    evaluation = learning.evaluation_rows(("sensor_a",), seed=36, length=11, trajectories=3)
    assert len(evaluation) == 33
    assert [index for index, row in enumerate(evaluation) if not row.history] == [0, 11, 22]
    assert learning.costs == before
    assert learning.observe(next_probe, ("sensor_a",)) == unchanged.observe(next_probe, ("sensor_a",))


@pytest.mark.parametrize("task_index", range(3))
def test_real_subprocess_holds_the_same_experiment_state_and_measurement_history(task_index, monkeypatch):
    # CLI entry points add src to sys.path; it need not already be in the
    # operator's PYTHONPATH for the actual apparatus process to start.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    with make_lab(task_index, 29) as local, make_lab(task_index, 29, backend="process") as process:
        assert process.process_id != os.getpid()
        assert process.spec_for(("sensor_a",)) == local.spec_for(("sensor_a",))
        first = experiment("repeatable", 9)
        for probe, channels in ((first, ()), (first, ("sensor_a",)),
                                (experiment("new", 13, 19), ("sensor_a", "sensor_b"))):
            assert process.acquire(probe, channels) == local.acquire(probe, channels)
            assert process.costs == local.costs
        before = dict(process.costs)
        assert process.evaluation_rows(("sensor_a",), seed=20, length=17, trajectories=2) == \
            local.evaluation_rows(("sensor_a",), seed=20, length=17, trajectories=2)
        assert process.costs == before
        process.reset()
        local.reset()
        after = experiment("after_reset", 6, 51)
        assert process.observe(after, ("sensor_a",)) == local.observe(after, ("sensor_a",))
    assert process._process.poll() == 0
