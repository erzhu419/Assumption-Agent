from hegel_machine.cognition.temporal_method_lab import TemporalMethodLab
from hegel_machine.cognition.models import Probe


def test_calibration_drift_occurs_after_paid_initial_measurements_with_reset_history():
    lab = TemporalMethodLab("window", 3)
    control, sensor = lab.control_name, lab.sensor_name
    probe = Probe("impulse", tuple({control: value} for value in (1., 0., 0., 0., 0., 0.)))
    first = lab.acquire(probe, (sensor,)).observations
    lab.acquire(Probe("complete_initial_calibration", tuple({control: 0.} for _ in range(42))), (sensor,))
    changed = lab.acquire(probe, (sensor,)).observations
    assert first[2].features[sensor] > .7
    assert changed[2].features[sensor] == 0. and changed[3].features[sensor] < -.5
    assert first[0].history == changed[0].history == ()
    assert lab.costs == {"observation_cost": 54, "measurement_cost": 162, "queries": 3}


def test_terminal_has_real_activation_and_negative_events_without_changing_learning_costs():
    for stage in range(5):
        lab = TemporalMethodLab("event", stage)
        before = lab.costs.copy()
        rows = lab.evaluation_rows(seed=811, length=32, trajectories=4)
        assert any(row.target > 0. for row in rows)
        assert any(row.target < 0. for row in rows)
        assert rows[0].target == rows[32].target == rows[64].target == rows[96].target == 0.
        assert lab.costs == before


def test_no_true_delay_or_mechanism_label_appears_in_the_public_spec():
    source = TemporalMethodLab("window", 0)
    returned = TemporalMethodLab("event", 4)
    assert source.spec == returned.spec
    changed = [TemporalMethodLab("event", stage) for stage in (1, 2, 3)]
    assert changed[0].spec == changed[1].spec == changed[2].spec
    assert changed[0].spec.context == {}
