"""Source-free unit tests for the SCAR CSSM append-only repair contract.

These tests intentionally exercise only deterministic contract primitives.  They
must not read the consumed SCAR cohort, labels, scorer outputs, or formal result.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import math

import pytest

from assumption_agent import gscl_scar_cssm_repair_contract_v2 as subject


_DOMAIN = "ASSUMPTION_AGENT/GSCL_SCAR/REPAIR_TEST/V1"
_BINDING_ROOT = "4" * 64


def test_canonical_hash_is_order_invariant_domain_separated_and_self_sealed() -> None:
    left = {"schema": "fixture_v1", "nested": {"z": 3, "a": [1, 2]}}
    right = {"nested": {"a": [1, 2], "z": 3}, "schema": "fixture_v1"}

    assert subject.canonical_bytes(left) == subject.canonical_bytes(right)
    assert subject.domain_hash(_DOMAIN, left) == subject.domain_hash(_DOMAIN, right)
    assert subject.domain_hash(_DOMAIN, left) != subject.domain_hash(
        _DOMAIN + "_OTHER", left
    )
    assert len(subject.domain_hash(_DOMAIN, left)) == 64

    sealed = subject.seal_payload(_DOMAIN, left)
    assert "self_sha256" not in left
    assert sealed["self_sha256"] == subject.domain_hash(_DOMAIN, left)
    assert subject.validate_self_seal(_DOMAIN, sealed) is True

    tampered = copy.deepcopy(sealed)
    tampered["nested"]["z"] = 4
    with pytest.raises(subject.ScarRepairContractError):
        subject.validate_self_seal(_DOMAIN, tampered)

    missing = dict(sealed)
    missing.pop("self_sha256")
    with pytest.raises(subject.ScarRepairContractError):
        subject.validate_self_seal(_DOMAIN, missing)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        ("M", "M", "M"),
        ("M", "V", "V"),
        ("M", "U", "U"),
        ("V", "M", "V"),
        ("V", "V", "V"),
        ("V", "U", "V"),
        ("U", "M", "U"),
        ("U", "V", "V"),
        ("U", "U", "U"),
    ),
)
def test_mvu_conjunction_full_truth_table(
    left: str, right: str, expected: str
) -> None:
    assert subject.conjoin_evidence(left, right) == expected


@pytest.mark.parametrize(
    ("states", "domain_complete", "expected"),
    (
        (("M", "M"), False, "M"),
        (("M", "V"), False, "M"),
        (("M", "U"), False, "M"),
        (("V", "M"), False, "M"),
        (("V", "V"), False, "U"),
        (("V", "U"), False, "U"),
        (("U", "M"), False, "M"),
        (("U", "V"), False, "U"),
        (("U", "U"), False, "U"),
        (("M", "M"), True, "M"),
        (("M", "V"), True, "M"),
        (("M", "U"), True, "M"),
        (("V", "M"), True, "M"),
        (("V", "V"), True, "V"),
        (("V", "U"), True, "U"),
        (("U", "M"), True, "M"),
        (("U", "V"), True, "U"),
        (("U", "U"), True, "U"),
        (("M", "U", "V"), True, "M"),
    ),
)
def test_mvu_exhaustive_or_requires_a_complete_domain_for_all_violation(
    states: tuple[str, ...], domain_complete: bool, expected: str
) -> None:
    assert (
        subject.exhaustive_or_evidence(
            states, domain_complete=domain_complete
        )
        == expected
    )


def test_mvu_operators_reject_empty_or_noncanonical_states() -> None:
    with pytest.raises(subject.ScarRepairContractError):
        subject.exhaustive_or_evidence((), domain_complete=True)
    with pytest.raises(subject.ScarRepairContractError):
        subject.conjoin_evidence("MATCH", "M")
    with pytest.raises(subject.ScarRepairContractError):
        subject.exhaustive_or_evidence(("M", "?"), domain_complete=True)


def _group_to_fold(
    groups: tuple[str, ...], folds: tuple[int, ...]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for group, fold in zip(groups, folds, strict=True):
        if group in result:
            assert result[group] == fold
        result[group] = fold
    return result


def test_grouped_folds_are_stable_and_never_split_a_group() -> None:
    groups = ("g3", "g1", "g2", "g1", "g4", "g3", "g5")
    folds = subject.assign_grouped_folds(
        groups, fold_count=3, binding_root=_BINDING_ROOT
    )
    assert len(folds) == len(groups)
    assert all(type(fold) is int and 0 <= fold < 3 for fold in folds)
    mapping = _group_to_fold(groups, folds)

    reordered = tuple(reversed(groups))
    reordered_folds = subject.assign_grouped_folds(
        reordered, fold_count=3, binding_root=_BINDING_ROOT
    )
    assert _group_to_fold(reordered, reordered_folds) == mapping
    assert (
        subject.assign_grouped_folds(
            groups, fold_count=3, binding_root=_BINDING_ROOT
        )
        == folds
    )


def test_grouped_folds_fail_closed_on_bad_roots_or_fold_counts() -> None:
    with pytest.raises(subject.ScarRepairContractError):
        subject.assign_grouped_folds(
            ("g1", "g2"), fold_count=1, binding_root=_BINDING_ROOT
        )
    with pytest.raises(subject.ScarRepairContractError):
        subject.assign_grouped_folds(
            ("g1", "g2"), fold_count=2, binding_root="not-a-root"
        )
    with pytest.raises(subject.ScarRepairContractError):
        subject.assign_grouped_folds(
            ("g1", ""), fold_count=2, binding_root=_BINDING_ROOT
        )


def test_standardized_ridge_is_deterministic_and_zero_sd_is_inert() -> None:
    features = (
        (0.0, 7.0, 1.0),
        (1.0, 7.0, 0.0),
        (2.0, 7.0, 1.0),
        (3.0, 7.0, 0.0),
    )
    targets = (0.0, 1.0, 2.0, 3.0)

    first = subject.fit_standardized_ridge(features, targets, l2=1.0)
    second = subject.fit_standardized_ridge(features, targets, l2=1.0)
    assert first == second
    assert first.feature_means == (1.5, 7.0, 0.5)
    assert first.feature_scales[1] == 1.0
    assert first.coefficients[1] == 0.0
    assert first.predict(features) == second.predict(features)
    assert all(math.isfinite(value) for value in first.predict(features))


def _threshold_example(
    score: int,
    delta: int,
    *,
    old_success_count: int = 0,
    override_preserved_count: int = 0,
) -> subject.ThresholdExample:
    return subject.ThresholdExample(
        selector_score=score,
        utility_delta=delta,
        old_success_count=old_success_count,
        override_preserved_count=override_preserved_count,
    )


def test_threshold_tie_uses_higher_threshold_then_fewer_overrides() -> None:
    # At threshold 50 both rows act; at 150 only the second acts.  Net utility
    # is five in both cases, so the more conservative threshold must win.
    examples = (
        _threshold_example(100, 0),
        _threshold_example(200, 5),
    )
    selected = subject.select_override_threshold(
        examples,
        thresholds=(50, 150),
        minimum_preservation=Fraction(1, 1),
    )
    assert selected.threshold == 150
    assert selected.net_utility_delta == 5
    assert selected.override_count == 1


def test_threshold_no_positive_gain_selects_explicit_all_noop() -> None:
    selected = subject.select_override_threshold(
        (
            _threshold_example(100, -1),
            _threshold_example(200, 0),
        ),
        thresholds=(0, 100, 200),
        minimum_preservation=Fraction(1, 1),
    )
    assert selected.threshold == 200
    assert selected.net_utility_delta == 0
    assert selected.override_count == 0


def test_threshold_discards_higher_utility_candidate_that_breaks_preservation() -> None:
    examples = (
        _threshold_example(100, 10),
        _threshold_example(
            300,
            -1,
            old_success_count=1,
            override_preserved_count=0,
        ),
    )
    unconstrained = subject.select_override_threshold(
        examples,
        thresholds=(0, 100, 300),
        minimum_preservation=Fraction(0, 1),
    )
    assert unconstrained.threshold == 0
    constrained = subject.select_override_threshold(
        examples,
        thresholds=(0, 100, 300),
        minimum_preservation=Fraction(1, 1),
    )
    assert constrained.threshold == 300
    assert constrained.override_count == 0
    assert constrained.preservation == Fraction(1, 1)


def test_action_selection_is_strict_and_noop_is_byte_exact() -> None:
    baseline = b'{ "pairs" : [["a","b"]] }\n'
    override = b'{"pairs":[["a","c"]]}'

    assert (
        subject.select_action_output(
            baseline,
            override,
            structurally_eligible=False,
            selector_score=101,
            threshold=100,
        )
        == baseline
    )
    assert (
        subject.select_action_output(
            baseline,
            override,
            structurally_eligible=True,
            selector_score=100,
            threshold=100,
        )
        == baseline
    )
    assert (
        subject.select_action_output(
            baseline,
            override,
            structurally_eligible=True,
            selector_score=101,
            threshold=100,
        )
        == override
    )


def test_pair_f1_and_pair_level_old_success_preservation() -> None:
    gold = (
        (("a", "1"), ("b", "2")),
        (("c", "3"),),
        (("d", "4"),),
    )
    baseline = (
        (("a", "1"), ("b", "2")),
        (("c", "3"), ("wrong", "x")),
        (("wrong", "y"),),
    )
    successor = (
        (("a", "1"),),
        (("c", "3"),),
        (("d", "4"),),
    )

    assert subject.pair_f1((('a', '1'),), (('a', '1'), ('b', '2'))) == Fraction(
        2, 3
    )
    preservation = subject.old_success_preservation(
        baseline, successor, gold
    )
    assert preservation.old_success_count == 3
    assert preservation.preserved_count == 2
    assert preservation.fraction == Fraction(2, 3)


def test_bootstrap_is_seeded_deterministic_and_small_counts_are_supported() -> None:
    successor = (Fraction(1), Fraction(2), Fraction(4), Fraction(3))
    baseline = (Fraction(0), Fraction(2), Fraction(1), Fraction(5))
    first = subject.paired_bootstrap_mean_delta(
        successor, baseline, seed=17, replicate_count=31
    )
    second = subject.paired_bootstrap_mean_delta(
        successor, baseline, seed=17, replicate_count=31
    )
    other_seed = subject.paired_bootstrap_mean_delta(
        successor, baseline, seed=18, replicate_count=31
    )
    assert first == second
    assert first.replicate_count == 31
    assert len(first.bootstrap_mean_deltas) == 31
    assert first.bootstrap_mean_deltas != other_seed.bootstrap_mean_deltas
    assert first.observed_mean_delta == Fraction(1, 2)


@pytest.mark.parametrize(
    ("implementation_valid", "preservation", "lower", "expected"),
    (
        (
            False,
            Fraction(0, 1),
            Fraction(1, 1),
            "REPAIR_DEVELOPMENT_IMPLEMENTATION_INVALID",
        ),
        (
            True,
            Fraction(99, 100),
            Fraction(1, 1),
            "REPAIR_DEVELOPMENT_UNSAFE_OLD_SUCCESS_REGRESSION",
        ),
        (
            True,
            Fraction(1, 1),
            Fraction(1, 100),
            "REPAIR_DEVELOPMENT_NO_PRACTICALLY_IMPORTANT_GAIN",
        ),
        (
            True,
            Fraction(1, 1),
            Fraction(101, 10_000),
            "POSTHOC_REPAIR_DEVELOPMENT_QUALIFIED",
        ),
    ),
)
def test_verdict_precedence_and_strict_mid_boundary(
    implementation_valid: bool,
    preservation: Fraction,
    lower: Fraction,
    expected: str,
) -> None:
    assert subject.decide_repair_development_verdict(
        implementation_valid=implementation_valid,
        old_success_preservation=preservation,
        minimum_old_success_preservation=Fraction(1, 1),
        primary_ci_lower_bound=lower,
        minimum_practically_important_gain=Fraction(1, 100),
    ) == expected


@pytest.mark.parametrize(
    "call",
    (
        lambda: subject.canonical_bytes({"value": float("nan")}),
        lambda: subject.canonical_bytes({"value": float("inf")}),
        lambda: subject.fit_standardized_ridge(
            ((0.0,), (float("nan"),)), (0.0, 1.0), l2=1.0
        ),
        lambda: subject.fit_standardized_ridge(
            ((0.0,),), (0.0, 1.0), l2=1.0
        ),
        lambda: subject.select_override_threshold(
            (_threshold_example(100, 1),),
            thresholds=(),
            minimum_preservation=Fraction(1, 1),
        ),
        lambda: subject.pair_f1(
            (("duplicate", "pair"), ("duplicate", "pair")),
            (("duplicate", "pair"),),
        ),
        lambda: subject.paired_bootstrap_mean_delta(
            (Fraction(1),), (), seed=1, replicate_count=3
        ),
        lambda: subject.paired_bootstrap_mean_delta(
            (float("nan"),), (0.0,), seed=1, replicate_count=3
        ),
        lambda: subject.decide_repair_development_verdict(
            implementation_valid=True,
            old_success_preservation=float("nan"),
            minimum_old_success_preservation=Fraction(1, 1),
            primary_ci_lower_bound=Fraction(1, 10),
            minimum_practically_important_gain=Fraction(1, 100),
        ),
    ),
)
def test_malformed_and_nonfinite_inputs_fail_closed(call) -> None:
    with pytest.raises(subject.ScarRepairContractError):
        call()
