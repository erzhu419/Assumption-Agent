from hegel_machine.cognition.growth_lab import GrowthLab
from hegel_machine.cognition.models import Probe


def controls(lab, pairs):
    return Probe("shared_controls", tuple(dict(zip(lab._roles, pair)) for pair in pairs))


def test_renaming_preserves_response_and_original_return():
    pairs = ((-.9, .2), (-.3, -.5), (.25, .8), (.7, -.4))
    labs = [GrowthLab(seed=31, stage=stage) for stage in (0, 1, 7)]
    targets = [[row.target for row in lab.observe(controls(lab, pairs))] for lab in labs]
    assert targets[0] == targets[1] == targets[2]
    assert labs[0].spec.feature_names != labs[1].spec.feature_names
    assert labs[0].spec.feature_names == labs[2].spec.feature_names


def test_piecewise_change_preserves_the_old_negative_side():
    pairs = ((-.9, .2), (-.3, -.5), (.25, .8), (.7, -.4))
    original, changed = GrowthLab(seed=31, stage=1), GrowthLab(seed=31, stage=2)
    before = original.observe(controls(original, pairs))
    after = changed.observe(controls(changed, pairs))
    assert [row.target for row in before[:2]] == [row.target for row in after[:2]]
    assert all(new.target > old.target for old, new in zip(before[2:], after[2:]))


def test_only_the_paid_signal_explains_repeated_hidden_variation():
    original, hidden = GrowthLab(seed=31, stage=1), GrowthLab(seed=31, stage=3)
    pair = (.4, -.2)
    base = original.observe(controls(original, (pair,)))[0].target
    rows = hidden.observe(controls(hidden, (pair,) * 32), (hidden._signal, hidden._decoy))
    assert all(abs(row.target - row.features[hidden._signal] - base) < 1e-12 for row in rows)
    assert max(abs(row.target - row.features[hidden._decoy] - base) for row in rows) > .1
    assert hidden.costs["observation_cost"] == 32
    assert hidden.costs["measurement_cost"] == 128
    unmeasured = hidden.observe(controls(hidden, (pair,)))[0]
    assert set(unmeasured.features) == set(hidden.spec.feature_names)


def test_repeating_the_same_probe_draws_independent_response_noise():
    laboratory = GrowthLab(seed=31, stage=5)
    probe = controls(laboratory, ((.4, -.2),) * 32)
    first = laboratory.observe(probe)
    second = laboratory.observe(probe)
    assert [row.target for row in first] != [row.target for row in second]
    assert first[0].features == second[0].features


def test_stage_truth_is_absent_from_the_public_specification():
    labs = [GrowthLab(seed=31, stage=stage) for stage in range(8)]
    assert len({lab.spec.task_id for lab in labs}) == 1
    assert len({lab.spec.description for lab in labs}) == 1
    assert all(lab.spec.context == {} for lab in labs)
    assert all({channel.cost_per_row for channel in lab.channels} == {2} for lab in labs)


def test_terminal_rows_are_independent_and_do_not_change_training_acquisitions():
    laboratory = GrowthLab(seed=31, stage=5)
    probe = controls(laboratory, ((.4, -.2),) * 4)
    laboratory.observe(probe)
    before = dict(laboratory.costs)
    evaluated = laboratory.evaluation_rows(seed=731, length=8, trajectories=2)
    assert laboratory.costs == before
    assert len(evaluated["observations"]) == 16
    assert evaluated["irreducible_noise_variance"] == .12 ** 2
    assert any(actual.target != expected.target for actual, expected in
               zip(evaluated["observations"], evaluated["mean_observations"]))
