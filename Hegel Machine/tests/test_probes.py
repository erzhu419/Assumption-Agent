import pytest

from hegel_machine.probes import (
    CandidateFit,
    TaskGeometry,
    choose_discriminating_probe,
    hypothesis_membership,
    quotient_classes,
    total_variation,
)
from hegel_machine.schema import ProbeSpec


def test_total_variation_normalizes_mass():
    assert total_variation({"yes": 2, "no": 0}, {"yes": 0, "no": 3}) == 1.0


def test_probe_geometry_and_quotient():
    geometry = TaskGeometry("task", "epoch", (("q1", 1.0), ("q2", 2.0)))
    hypotheses = {
        "h1": {"q1": {"a": 1}, "q2": {"a": 1}},
        "h1_alias": {"q1": {"a": 2}, "q2": {"a": 3}},
        "h2": {"q1": {"b": 1}, "q2": {"b": 1}},
    }
    classes = quotient_classes(hypotheses, geometry, tolerance=0.0)
    assert ("h1", "h1_alias") in classes
    assert ("h2",) in classes


def test_tolerance_chain_does_not_merge_distinguishable_endpoints():
    geometry = TaskGeometry("task", "epoch", (("q", 1.0),))
    hypotheses = {
        "a": {"q": {"yes": 0.0, "no": 1.0}},
        "b": {"q": {"yes": 0.4, "no": 0.6}},
        "c": {"q": {"yes": 0.8, "no": 0.2}},
    }
    classes = quotient_classes(hypotheses, geometry, tolerance=0.5)
    assert not any(set(group) == {"a", "b", "c"} for group in classes)
    for group in classes:
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                assert geometry.distance(
                    hypotheses[left], hypotheses[right]
                ) <= 0.5


def test_discriminating_probe_excludes_semantic_only_probe():
    probes = (
        ProbeSpec(
            "semantic",
            "1",
            "x",
            "y",
            "cosine",
            ("task",),
            "epoch",
            ("anchor",),
            "cutoff",
            cost=0.1,
            semantic_only=True,
        ),
        ProbeSpec(
            "structural",
            "1",
            "x",
            "y",
            "tv",
            ("task",),
            "epoch",
            ("anchor",),
            "cutoff",
        ),
    )
    outcomes = {
        "h1": {
            "semantic": {"a": 1},
            "structural": {"left": 1},
        },
        "h2": {
            "semantic": {"b": 1},
            "structural": {"right": 1},
        },
    }
    geometry = TaskGeometry("task", "epoch", (("structural", 1.0),))
    assert (
        choose_discriminating_probe(
            probes,
            outcomes,
            ("h1", "h2"),
            geometry=geometry,
            data_cutoff="cutoff",
        )
        == "structural"
    )


def test_discriminating_probe_rejects_cross_epoch_task_and_cutoff_leakage():
    probes = (
        ProbeSpec(
            "wrong_epoch",
            "1",
            "x",
            "y",
            "tv",
            ("task",),
            "old_epoch",
            ("anchor",),
            "cutoff",
        ),
        ProbeSpec(
            "wrong_task",
            "1",
            "x",
            "y",
            "tv",
            ("other_task",),
            "epoch",
            ("anchor",),
            "cutoff",
        ),
        ProbeSpec(
            "current",
            "1",
            "x",
            "y",
            "tv",
            ("task",),
            "epoch",
            ("anchor",),
            "cutoff",
        ),
    )
    outcomes = {
        "h1": {
            "wrong_epoch": {"left": 1},
            "wrong_task": {"left": 1},
            "current": {"same": 1},
        },
        "h2": {
            "wrong_epoch": {"right": 1},
            "wrong_task": {"right": 1},
            "current": {"same": 1},
        },
    }
    geometry = TaskGeometry(
        "task",
        "epoch",
        (("wrong_epoch", 1.0), ("wrong_task", 1.0), ("current", 1.0)),
    )
    assert (
        choose_discriminating_probe(
            probes,
            outcomes,
            ("h1", "h2"),
            geometry=geometry,
            data_cutoff="cutoff",
        )
        == "current"
    )


@pytest.mark.parametrize("tolerance", (float("nan"), float("inf")))
def test_quotient_rejects_nonfinite_tolerance(tolerance):
    geometry = TaskGeometry("task", "epoch", (("q", 1.0),))
    with pytest.raises(ValueError, match="finite"):
        quotient_classes(
            {"a": {"q": {"x": 1}}},
            geometry,
            tolerance=tolerance,
        )


def test_membership_uses_infimum_only_over_admissible_bindings():
    fits = (
        CandidateFit("invalid", "s1", "r1", 0.0, -1.0, 1.0, True, 0.0),
        CandidateFit("valid", "s2", "r2", 0.1, 0.5, 0.8, True, 0.2),
    )
    result = hypothesis_membership(
        fits,
        maximum_violation=0.2,
        minimum_hard_negative_margin=0.1,
        minimum_unseen_prediction=0.5,
        complexity_weight=0.1,
    )
    assert result.accepted
    assert result.best_fit is fits[1]
    assert result.score == pytest.approx(0.12)
