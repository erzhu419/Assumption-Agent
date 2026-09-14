from hegel_machine.cognition.dynamic_instrument_lab import DynamicInstrumentLab
from hegel_machine.cognition.models import Probe


def impulse():
    return Probe("impulse", tuple({"u": value} for value in (1., 0., 0., -.5, 0., 0.)))


def test_private_instrument_and_response_changes_have_identical_public_specs():
    labs = [DynamicInstrumentLab(stage, 100 + stage, apparatus_seed=151) for stage in range(5)]
    assert all(lab.spec == labs[0].spec for lab in labs)
    assert all(lab.spec_for(("meter",)) == labs[0].spec_for(("meter",)) for lab in labs)
    assert labs[0].spec.context == {}
    rows = [lab.observe(impulse(), ("meter",)) for lab in labs]
    assert rows[0] == rows[1] == rows[4]
    assert rows[0][0].features["meter"] == rows[2][0].features["meter"] == .4
    assert rows[0][1].features["meter"] > 1. and rows[2][1].features["meter"] == 0.
    assert [row.features for row in rows[2]] == [row.features for row in rows[3]]
    assert [row.target for row in rows[2]] != [row.target for row in rows[3]]


def test_measurement_projection_resets_and_independent_evaluation_costs():
    lab = DynamicInstrumentLab(2, apparatus_seed=151)
    command = lab.acquire(impulse())
    measured = lab.acquire(impulse(), ("meter",))
    assert [row.target for row in command.observations] == [row.target for row in measured.observations]
    assert all(set(row.features) == {"u"} for row in command.observations)
    assert all(all("meter" not in past for past in row.history) for row in command.observations)
    assert lab.costs == {"observation_cost": 12, "measurement_cost": 18, "queries": 2}
    assert command.observations[0].history == measured.observations[0].history == ()
    assert lab.observe(Probe("repeat", impulse().sequence), ("meter",)) == measured.observations
    before = lab.costs.copy()
    terminal = lab.evaluation_rows(seed=313, length=8, trajectories=2)
    assert len(terminal) == 16 and terminal[0].history == terminal[8].history == ()
    assert lab.costs == before
    saved = list(lab.acquisitions)
    lab.restore(saved)
    assert lab.costs == before


def test_future_control_suffix_does_not_change_already_observed_readings():
    lab = DynamicInstrumentLab(3, apparatus_seed=151)
    first = lab.observe(Probe("first", ({"u": .8}, {"u": -.6}, {"u": 1.})), ("meter",))
    second = lab.observe(Probe("second", ({"u": .8}, {"u": -.6}, {"u": -1.})), ("meter",))
    assert first[:2] == second[:2]
    assert first[2].features["meter"] != second[2].features["meter"]


def test_confirmation_varies_delays_and_responses_with_the_same_public_interface():
    probe = Probe("single_impulse", tuple({"u": value} for value in (1., 0., 0., 0., 0.)))
    observed_delays, first_readings, initial_targets = set(), set(), set()
    public = DynamicInstrumentLab().spec
    for seed in range(101, 113):
        labs = [DynamicInstrumentLab(stage, apparatus_seed=seed, profile="confirmation")
                for stage in range(5)]
        assert all(lab.spec == public for lab in labs)
        rows = [lab.observe(probe, ("meter",)) for lab in labs]
        assert rows[0] == rows[1] == rows[4]
        assert [row.features for row in rows[2]] == [row.features for row in rows[3]]
        assert [row.target for row in rows[2]] != [row.target for row in rows[3]]
        delays = [next(i for i in range(1, 4) if stage[i].features["meter"] != 0.)
                  for stage in (rows[0], rows[2])]
        assert delays[0] != delays[1]
        observed_delays.update(delays)
        first_readings.add(rows[0][0].features["meter"])
        initial_targets.add(rows[0][0].target)
    assert observed_delays == {1, 2, 3}
    assert len(first_readings) == len(initial_targets) == 12


def test_confirmation_restore_and_terminal_keep_the_sampled_apparatus():
    lab = DynamicInstrumentLab(3, apparatus_seed=202, profile="confirmation")
    before = lab.acquire(impulse(), ("meter",)).observations
    acquisitions, costs = list(lab.acquisitions), lab.costs
    lab.restore(acquisitions)
    assert lab.costs == costs
    assert lab.acquire(impulse(), ("meter",)).observations == before
    expected = DynamicInstrumentLab(3, apparatus_seed=202, profile="confirmation")
    assert lab.evaluation_rows(("meter",), seed=313, length=8, trajectories=2) == (
        expected.evaluation_rows(("meter",), seed=313, length=8, trajectories=2))
