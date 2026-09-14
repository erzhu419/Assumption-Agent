"""A joint measurement/history gap must come from observations, not labels."""

from dataclasses import asdict
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.evaluation import point_loss
from hegel_machine.cognition.models import Probe
from hegel_machine.cognition.synthesis import _fit_linear
from hegel_machine.cognition.unknown_task_lab import TASK_COUNT, UnknownTaskLab


def probe(lab, name, seed=7, length=48):
    random = Random(seed)
    return Probe(name, tuple({key: random.uniform(*bounds) for key, bounds in lab.spec.feature_bounds.items()}
                            for _ in range(length)))


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_public_interface_has_no_mechanism_or_history_labels_and_paid_histories_only():
    for index in range(8):
        lab = UnknownTaskLab(index, seed=21)
        assert lab.spec.context == {}
        assert lab.spec.description == UnknownTaskLab().spec.description
        assert set(asdict(lab.spec)) == set(asdict(UnknownTaskLab().spec))
        assert all(set(point) == set(lab.spec.feature_bounds)
                   for item in lab.spec.initial_probes for point in item.sequence)
        experiment = probe(lab, "same", length=7)
        bare = lab.acquire(experiment)
        selected = (lab.channels[-1].name,)
        measured = lab.acquire(experiment, selected)
        assert [row.target for row in bare.observations] == [row.target for row in measured.observations]
        for batch, names in ((bare, set(lab.spec.feature_names)),
                             (measured, {*lab.spec.feature_names, *selected})):
            for position, row in enumerate(batch.observations):
                assert set(row.features) == names
                assert row.history == tuple({**prior.features, "__target__": prior.target}
                                            for prior in batch.observations[:position])
        assert lab.costs == {"observation_cost": 14, "measurement_cost": 21, "queries": 2}
        assert measured.observations[0].features[selected[0]] == 0.


def test_joint_task_requires_both_measuring_and_using_measurement_history():
    lab = UnknownTaskLab(2, seed=19)
    channels = tuple(channel.name for channel in lab.channels)
    training = [row for index in range(6)
                for row in lab.observe(probe(lab, f"train_{index}", index), channels) if len(row.history) >= 4]
    current = Program("feature", name="u")
    history = [Program("lag", name=name, lag=lag)
               for lag in range(1, 5) for name in ("u", "__target__")]
    measured = [Program("feature", name=name) for name in channels]
    prior_sensor = Program("lag", name=channels[0], lag=1)
    models = [_fit_linear([current, *history], training),
              _fit_linear([current, *measured], training),
              _fit_linear([current, prior_sensor], training)]
    fresh = [row for row in lab.evaluation_rows(channels, seed=861) if len(row.history) >= 4]
    assert mse(models[0], fresh) > .2
    assert mse(models[1], fresh) > .2
    assert mse(models[2], fresh) < 1e-12


def test_related_renamed_task_and_delay_change_need_different_executable_lags():
    related, changed = UnknownTaskLab(3), UnknownTaskLab(4)
    assert related.spec.feature_names == changed.spec.feature_names == ("x",)
    assert related.spec_for(("channel_p",)).feature_names == ("x", "channel_p")
    for lab, lag in ((related, 1), (changed, 2)):
        rows = [row for row in lab.observe(probe(lab, "fit"), ("channel_p",)) if len(row.history) >= 2]
        best = _fit_linear([Program("feature", name="x"), Program("lag", name="channel_p", lag=lag)], rows)
        wrong = _fit_linear([Program("feature", name="x"),
                             Program("lag", name="channel_p", lag=3 - lag)], rows)
        fresh = [row for row in lab.evaluation_rows(("channel_p",), seed=17) if len(row.history) >= 2]
        assert mse(best, fresh) < 1e-12
        assert mse(wrong, fresh) > .1


def test_replay_is_order_independent_and_restore_does_not_duplicate_costs():
    lab, independent = UnknownTaskLab(6, 37), UnknownTaskLab(6, 37)
    first, second = probe(lab, "first"), probe(lab, "second", seed=13)
    expected = independent.acquire(second, ("reading_n",))
    lab.acquire(first, ("reading_m",))
    assert lab.acquire(second, ("reading_n",)) == expected
    recorded, costs = list(lab.acquisitions), dict(lab.costs)
    lab.restore(recorded)
    assert lab.acquisitions == recorded
    assert lab.costs == costs
    assert lab.acquire(second, ("reading_n",)) == expected
    assert lab.costs["observation_cost"] == costs["observation_cost"] + second.cost


def test_evaluation_has_fresh_trajectories_and_preserves_learning_state():
    lab, unchanged = UnknownTaskLab(7, 71), UnknownTaskLab(7, 71)
    experiment = probe(lab, "paid")
    assert lab.acquire(experiment, ("meter_r",)) == unchanged.acquire(experiment, ("meter_r",))
    costs = dict(lab.costs)
    rows = lab.evaluation_rows(("meter_r",), seed=36, length=11, trajectories=3)
    assert len(rows) == 33
    assert [index for index, row in enumerate(rows) if not row.history] == [0, 11, 22]
    assert all(rows[index].features["meter_r"] == 0. for index in (0, 11, 22))
    assert lab.costs == costs
    next_probe = probe(lab, "next", seed=29)
    assert lab.acquire(next_probe, ("meter_r",)) == unchanged.acquire(next_probe, ("meter_r",))


def test_cannot_intervene_on_an_instrument_or_overrun_supported_probe_length():
    lab = UnknownTaskLab(2)
    with pytest.raises(ValueError, match="controllable inputs"):
        lab.acquire(Probe("invalid", ({"u": 0., "sensor_a": 1.},)))
    with pytest.raises(ValueError, match="maximum experiment length"):
        lab.acquire(probe(lab, "too_long", length=97))
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}


@pytest.mark.parametrize("task,control,instrument,decay", [
    (8, "u", None, .83), (9, "u", "sensor_a", .83),
    (10, "x", "channel_p", .83), (11, "x", "channel_p", .94)])
def test_continuous_state_retains_information_beyond_four_input_steps(task, control, instrument, decay):
    lab = UnknownTaskLab(task, seed=173)
    channels = (instrument,) if instrument else ()
    names = (control, *channels)
    training = [row for index in range(4)
                for row in lab.observe(probe(lab, f"state_{index}", 31 + index), channels)
                if len(row.history) >= 4]
    finite_terms = [Program("feature", name=name) for name in names]
    finite_terms.extend(Program("lag", name=name, lag=lag)
                        for lag in range(1, 5) for name in names)
    finite = _fit_linear(finite_terms, training)
    state = Program("accumulate", value=decay, children=(
        Program("feature", name=instrument or control), Program("const", value=0.)))
    continuous = _fit_linear([Program("feature", name=control), state], training)
    fresh = [row for row in lab.evaluation_rows(channels, seed=683) if len(row.history) >= 4]
    assert mse(finite, fresh) > .02
    assert mse(continuous, fresh) < 1e-12
    # Past measured responses are part of the public interface. They can also
    # carry this linear state through an autoregression; finite-input failure
    # does not establish that every finite-output representation must fail.
    autoregressive = _fit_linear([*finite_terms, Program("lag", name="__target__", lag=1)], training)
    assert mse(autoregressive, fresh) < 1e-12


def test_new_tasks_expose_fixed_loss_but_not_noise_or_state_parameters():
    assert TASK_COUNT == 14
    for task in range(8, 14):
        lab = UnknownTaskLab(task)
        assert lab.spec.context == {}
        assert lab.spec.description == UnknownTaskLab().spec.description
        assert lab.spec.risk_quantile == (.9 if task >= 12 else .5)
        assert set(lab.spec.feature_bounds) == set(lab.spec.feature_names)
        measured = lab.acquire(probe(lab, "public", length=5), (lab.channels[0].name,))
        assert set(measured.observations[-1].features) == {
            *lab.spec.feature_names, lab.channels[0].name}


@pytest.mark.parametrize("task", [12, 13])
def test_noisy_trials_are_paid_reproducible_and_oracle_scoring_is_separate(task):
    lab = UnknownTaskLab(task, seed=271)
    channels = tuple(channel.name for channel in lab.channels)
    controls = probe(lab, "first_noise", seed=73)
    first = lab.acquire(controls, channels)
    second = lab.acquire(Probe("new_noise", controls.sequence), channels)
    assert first != second
    assert first == lab.acquire(controls, channels)
    assert lab.costs == {"observation_cost": 144, "measurement_cost": 576, "queries": 3}
    unchanged = dict(lab.costs)
    oracle = lab.evaluation_oracle(channels, seed=791, length=64, trajectories=16)
    assert lab.costs == unchanged
    assert oracle["costs"] == {"observation_cost": 1024, "measurement_cost": 4096, "queries": 16}
    rows, means, quantiles, scales = (oracle[key] for key in (
        "observations", "conditional_means", "conditional_quantiles", "noise_scales"))
    assert len(rows) == len(means) == len(quantiles) == len(scales) == 1024
    assert oracle["risk_quantile"] == .9
    for row, mean, quantile, scale in zip(rows, means, quantiles, scales):
        expected_mean = .15 + .55 * row.features["u"]
        if task == 13 and row.history:
            expected_mean += 1.2 * row.history[-1]["sensor_a"]
        assert mean == pytest.approx(expected_mean)
        if row.history:
            driver = row.features["u" if task == 12 else "sensor_a"]
            assert scale == pytest.approx(.08 + .20 * driver ** 2)
        else:
            assert scale == 0. and row.target == mean == quantile
        assert abs(row.target - mean) <= scale + 1e-12
        assert quantile == pytest.approx(mean + .8 * scale)
    risk = sum(point_loss(value, row.target, .9) for value, row in zip(quantiles, rows)) / len(rows)
    mean_risk = sum(point_loss(value, row.target, .9) for value, row in zip(means, rows)) / len(rows)
    assert risk < mean_risk * .6
    coverage = sum(row.target <= value for value, row in zip(quantiles, rows)) / len(rows)
    assert .86 < coverage < .94
    assert rows == lab.evaluation_rows(channels, seed=791, length=64, trajectories=16)
    assert lab.costs == unchanged
