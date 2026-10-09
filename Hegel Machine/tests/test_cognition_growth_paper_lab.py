import pytest

from hegel_machine.cognition.growth_paper_lab import GrowthPaperLab
from hegel_machine.cognition.models import Probe


def controls(lab, pairs):
    return Probe("physical_controls", tuple({name: offset + scale * value
        for name, (offset, scale), value in zip(lab._roles, lab._units, pair)} for pair in pairs))


@pytest.mark.parametrize("family", range(3))
def test_independent_algebra_and_changed_units_preserve_canonical_response(family):
    source = GrowthPaperLab(seed=9301, stage=0, family=family)
    target = GrowthPaperLab(seed=9301, stage=1, family=family)
    pairs = ((-.9, .3), (-.25, -.7), (.4, .2), (.95, -.6))
    before = source.observe(controls(source, pairs))
    after = target.observe(controls(target, pairs))
    assert [row.target for row in before] == pytest.approx([row.target for row in after], abs=1e-12)
    assert source.spec.feature_names != target.spec.feature_names
    assert set(source.spec.feature_bounds.values()) != set(target.spec.feature_bounds.values())
    assert source.spec.context == target.spec.context == {}


def test_instrument_rank_reverses_without_publishing_the_role():
    source, target = (GrowthPaperLab(seed=9301, stage=stage) for stage in (3, 4))
    rank = lambda lab: [channel.name for channel in lab.channels].index(lab._signal)
    assert rank(source) != rank(target)
    assert {channel.description for lab in (source, target) for channel in lab.channels} == {
        "Read an auxiliary process instrument."}
    assert {channel.cost_per_row for lab in (source, target) for channel in lab.channels} == {2}


def test_controls_only_preserves_hidden_response_and_cannot_acquire_instruments():
    complete = GrowthPaperLab(seed=9301, stage=3)
    limited = GrowthPaperLab(seed=9301, stage=3, controls_only=True)
    pairs = ((.4, -.2),) * 12
    measured = complete.observe(controls(complete, pairs), (complete._signal,))
    unmeasured = limited.observe(controls(limited, pairs))
    assert [row.target for row in measured] == [row.target for row in unmeasured]
    assert len({row.target for row in unmeasured}) > 1
    assert limited.channels == ()
    with pytest.raises(ValueError, match="unknown instrument"):
        limited.acquire(controls(limited, pairs), (complete._signal,))


def test_noise_is_coordinate_dependent_and_terminal_truth_is_separate():
    lab = GrowthPaperLab(seed=9301, stage=5)
    rows = lab.observe(controls(lab, ((0., .2), (.9, .2))))
    assert lab._last_variances[1] > 4 * lab._last_variances[0]
    assert set(rows[0].features) == set(lab.spec.feature_names)
    paid = lab.costs
    evaluation = lab.evaluation_rows(seed=799991, length=16, trajectories=2)
    assert lab.costs == paid
    assert len(evaluation["noise_variances"]) == 32
    assert len(set(evaluation["noise_variances"])) > 1


def test_novel_expression_is_a_separate_post_prefix_intervention():
    source = GrowthPaperLab(seed=9301, stage=0)
    changed = GrowthPaperLab(seed=9301, stage=0, new_expression=True)
    pairs = ((.4, 0.), (.4, .7), (.4, -.7))
    original = source.observe(controls(source, pairs))
    revised = changed.observe(controls(changed, pairs))
    assert revised[0].target == original[0].target
    assert revised[1].target > original[1].target
    assert revised[2].target < original[2].target
