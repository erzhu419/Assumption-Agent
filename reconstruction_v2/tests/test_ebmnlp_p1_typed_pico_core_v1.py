from __future__ import annotations

from fractions import Fraction
import hashlib
import hmac
import json
import math

import pytest

from assumption_agent.benchmarks import ebmnlp_p1_typed_pico_core_v1 as core


def _synthetic_inputs(
    token_count: int = 192,
) -> tuple[
    tuple[core.EvidenceWindow, ...],
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    windows = core.build_evidence_windows(
        tuple(f"tok{index:03d}" for index in range(token_count))
    )
    count = len(windows)
    target = tuple(600_000 for _ in range(count))
    probabilities = {
        core.PARTICIPANT: target,
        core.INTERVENTION: tuple(200_000 for _ in range(count)),
        core.OUTCOME: tuple(100_000 for _ in range(count)),
    }
    query_cosines = tuple(500_000 + index * 10_000 for index in range(count))
    embeddings = tuple(
        (1_000_000 if index % 2 == 0 else -1_000_000, 1)
        for index in range(count)
    )
    return windows, probabilities, query_cosines, embeddings


def _synthetic_slate() -> core.RecipeSlate:
    windows, probabilities, query_cosines, embeddings = _synthetic_inputs()
    return core.build_recipe_slate(
        windows=windows,
        target_role=core.PARTICIPANT,
        role_probabilities=probabilities,
        query_cosines=query_cosines,
        embeddings=embeddings,
    )


def test_canonical_json_hash_role_registry_and_queries_are_exact() -> None:
    value = {"unicode": "α", "b": [2, 1], "a": Fraction(2, 3)}
    raw = core.canonical_json_bytes(value)
    assert raw.endswith(b"\n")
    assert json.loads(raw) == {
        "a": {"denominator": 3, "numerator": 2},
        "b": [2, 1],
        "unicode": "α",
    }
    assert core.canonical_sha256(value) == hashlib.sha256(raw).hexdigest()
    assert core.ROLE_ORDER == (
        "PARTICIPANT",
        "INTERVENTION",
        "OUTCOME",
    )
    assert core.ROLE_QUERIES == {
        "PARTICIPANT": (
            "Which text describes the participants or patient population in "
            "this clinical trial?"
        ),
        "INTERVENTION": (
            "Which text describes the intervention or treatment in this "
            "clinical trial?"
        ),
        "OUTCOME": (
            "Which text describes the outcomes or endpoints measured in this "
            "clinical trial?"
        ),
    }
    with pytest.raises(core.EbmNlpP1CoreError, match="non-finite"):
        core.canonical_json_bytes({"bad": float("nan")})


@pytest.mark.parametrize(
    ("token_count", "intervals"),
    (
        (1, ((0, 1),)),
        (48, ((0, 48),)),
        (49, ((0, 48), (1, 49))),
        (72, ((0, 48), (24, 72))),
        (73, ((0, 48), (24, 72), (25, 73))),
    ),
)
def test_fixed_windows_have_one_exact_tail(
    token_count: int, intervals: tuple[tuple[int, int], ...]
) -> None:
    tokens = tuple(f"t{index}" for index in range(token_count))
    windows = core.build_evidence_windows(tokens)
    assert tuple((row.start, row.end) for row in windows) == intervals
    assert tuple(row.ordinal for row in windows) == tuple(range(len(windows)))
    assert windows[-1].end == token_count
    for row in windows:
        assert row.window_id == f"W:{row.start:08d}:{row.end:08d}"
        assert row.text == " ".join(tokens[row.start : row.end])
    with pytest.raises(core.EbmNlpP1CoreError, match="whitespace-free"):
        core.build_evidence_windows(("two tokens",))


def test_probe_fit_is_independent_exact_and_one_class_totalizes() -> None:
    class FakeLogisticRegression:
        instances: list["FakeLogisticRegression"] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.classes_ = (0, 1)
            self.fit_labels: tuple[int, ...] = ()
            self.instances.append(self)

        def fit(
            self, rows: object, labels: tuple[int, ...]
        ) -> "FakeLogisticRegression":
            self.fit_labels = tuple(labels)
            return self

        def predict_proba(
            self, rows: object
        ) -> tuple[tuple[float, float], ...]:
            return tuple((0.25, 0.75) for _ in rows)  # type: ignore[arg-type]

    embeddings = ((1, 0), (0, 1), (1, 1), (-1, 1))
    probes = core.fit_independent_role_probes(
        embeddings,
        {
            core.PARTICIPANT: (0, 1, 0, 1),
            core.INTERVENTION: (0, 0, 0, 0),
            core.OUTCOME: (1, 1, 1, 1),
        },
        logistic_regression_cls=FakeLogisticRegression,
    )
    assert len(FakeLogisticRegression.instances) == 1
    fitted = FakeLogisticRegression.instances[0]
    assert fitted.kwargs == {
        "solver": "liblinear",
        "penalty": "l2",
        "C": 1,
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-6,
        "random_state": 0,
    }
    assert fitted.fit_labels == (0, 1, 0, 1)
    scored = probes.score_quantized(embeddings)
    assert scored[core.PARTICIPANT] == (750_000,) * 4
    assert scored[core.INTERVENTION] == (0,) * 4
    assert scored[core.OUTCOME] == (1_000_000,) * 4


def test_six_recipes_are_closed_deterministic_totalizers_with_frozen_ties() -> None:
    windows, probabilities, query_cosines, embeddings = _synthetic_inputs()
    first = core.materialize_recipe_actions(
        windows=windows,
        target_role=core.PARTICIPANT,
        role_probabilities=probabilities,
        query_cosines=query_cosines,
        embeddings=embeddings,
    )
    second = core.materialize_recipe_actions(
        windows=windows,
        target_role=core.PARTICIPANT,
        role_probabilities=probabilities,
        query_cosines=query_cosines,
        embeddings=embeddings,
    )
    assert first == second
    assert tuple(action.recipe_id for action in first) == core.RECIPE_IDS
    assert all(
        len(action.window_ordinals) == 5
        and len(set(action.window_ordinals)) == 5
        for action in first
    )
    # Exact posterior ties resolve by lower start/end.
    assert first[0].window_ordinals == (0, 1, 2, 3, 4)
    assert first[1].window_ordinals == (0, 1, 2, 3, 4)
    assert first[2].window_ordinals == (0, 1, 2, 3, 4)
    assert first[5].window_ordinals == (0, 1, 2, 3, 4)
    # The two set operators are substantive: posterior coverage jumps to the
    # first wholly uncovered interval and semantic diversity takes the first
    # opposite-direction embedding.
    assert first[3].window_ordinals[:3] == (0, 2, 4)
    assert first[4].window_ordinals[:2] == (0, 1)
    assert all(len(action.behavior_sha256) == 64 for action in first)
    with pytest.raises(core.EbmNlpP1CoreError, match="frozen at five"):
        core.materialize_recipe_actions(
            windows=windows,
            target_role=core.PARTICIPANT,
            role_probabilities=probabilities,
            query_cosines=query_cosines,
            embeddings=embeddings,
            top_k=4,
        )


def test_candidate_features_e0_raw_and_registry_tie_are_exact() -> None:
    slate = _synthetic_slate()
    assert len(slate.features) == 6
    assert all(len(row.values) == 16 for row in slate.features)
    for ordinal, features in enumerate(slate.features):
        assert features.values[10:] == tuple(
            core.INTEGER_SCALE if index == ordinal else 0
            for index in range(6)
        )
        mapping = features.as_mapping()
        assert 0 <= mapping["selected_union_target_posterior_mass_fraction"] <= 1_000_000
        assert 0 <= mapping["mean_pairwise_MiniLM_diversity"] <= 1_000_000
        assert 0 <= mapping["selected_window_overlap_fraction"] <= 1_000_000
    assert core.raw_probe_ranking(slate) == slate.actions[0]

    base = (
        800_000,
        700_000,
        900_000,
        300_000,
        600_000,
        500_000,
        400_000,
        900_000,
        200_000,
        500_000,
    )
    tied_features = tuple(
        core.CandidateFeatures(
            base
            + tuple(
                core.INTEGER_SCALE if index == ordinal else 0
                for index in range(6)
            )
        )
        for ordinal in range(6)
    )
    tied = core.RecipeSlate(slate.actions, tied_features)
    assert core.e0_score(tied.features[0]) == (
        Fraction(5, 20) * 800_000
        + Fraction(4, 20) * 700_000
        + Fraction(3, 20) * 300_000
        + Fraction(3, 20) * 600_000
        + Fraction(2, 20) * 500_000
        + Fraction(2, 20) * 400_000
        + Fraction(1, 20) * 900_000
        - Fraction(2, 20) * 200_000
    )
    assert core.select_e0(tied).registry_ordinal == 0


def test_e1_feature_contract_and_optional_torch_fit_are_full_slate() -> None:
    slate = _synthetic_slate()
    shifted = core.RecipeSlate(
        slate.actions,
        tuple(
            core.CandidateFeatures(
                (features.values[0] + 1_000,)
                + features.values[1:]
            )
            for features in slate.features
        ),
    )
    features, targets, means, deviations, zero = core.e1_feature_tensors(
        (slate, slate),
        (
            tuple(Fraction(index, 10) for index in range(6)),
            tuple(Fraction(5 - index, 10) for index in range(6)),
        ),
    )
    assert len(features) == len(targets) == 2
    assert all(len(row) == 6 for row in features)
    assert len(means) == len(deviations) == len(zero) == 16
    assert all(
        math.isfinite(value)
        for slate_row in features
        for candidate in slate_row
        for value in candidate
    )
    population_features = core.e1_feature_tensors(
        (slate,),
        (tuple(Fraction(index, 10) for index in range(6)),),
        standardization_slates=(slate, shifted),
    )
    assert population_features[2][0] == means[0] + 500.0

    torch = pytest.importorskip("torch")
    model = core.fit_e1_deepsets(
        (slate, slate),
        (
            tuple(Fraction(index, 10) for index in range(6)),
            tuple(Fraction(5 - index, 10) for index in range(6)),
        ),
        standardization_slates=(slate, shifted, shifted),
        torch_module=torch,
    )
    assert model.training_slate_count == 2
    assert model.standardization_slate_count == 3
    prediction = model.predict_slate(slate)
    assert len(prediction) == 6
    assert all(math.isfinite(value) for value in prediction)
    assert core.select_e1(model, slate).recipe_id in core.RECIPE_IDS


def test_exact_incremental_token_coverage_and_abstract_aggregation() -> None:
    windows = core.build_evidence_windows(tuple(f"t{i}" for i in range(96)))
    score = core.score_ranked_token_coverage(
        windows=windows,
        ranking=(0, 1, 2),
        positive_token_positions=(0, 24, 48, 49, 49),
    )
    assert score.defined
    assert score.newly_covered_positive_counts == (2, 2, 0)
    assert score.primary_utility == Fraction(45, 137)
    assert score.undiscounted_coverage_at_5 == 1
    assert score.complete_at_5 == 1
    zero = core.score_ranked_token_coverage(
        windows=windows,
        ranking=(0, 1, 2),
        positive_token_positions=(),
    )
    assert not zero.defined and zero.primary_utility is None

    abstract = {
        core.PARTICIPANT: Fraction(1, 2),
        core.INTERVENTION: None,
        core.OUTCOME: Fraction(1, 4),
    }
    assert core.aggregate_abstract_role_utilities(abstract) == Fraction(3, 8)
    aggregates = core.family_aggregate(
        (
            abstract,
            {
                core.PARTICIPANT: Fraction(1),
                core.INTERVENTION: Fraction(1, 3),
                core.OUTCOME: None,
            },
        )
    )
    assert aggregates == {
        core.PARTICIPANT: Fraction(3, 4),
        core.INTERVENTION: Fraction(1, 3),
        core.OUTCOME: Fraction(1, 4),
    }


def test_exact_one_sided_sign_test_and_cluster_comparison() -> None:
    test = core.exact_one_sided_sign_test(
        (Fraction(1),) * 5 + (Fraction(0), Fraction(-1))
    )
    assert (test.gains, test.harms, test.ties) == (5, 1, 1)
    assert test.one_sided_p == Fraction(7, 64)
    assert core.exact_one_sided_sign_test((0, 0)).one_sided_p == 1

    left = tuple(
        {
            core.PARTICIPANT: Fraction(1),
            core.INTERVENTION: Fraction(1, 2),
            core.OUTCOME: Fraction(1, 4),
        }
        for _ in range(6)
    ) + (
        {
            core.PARTICIPANT: None,
            core.INTERVENTION: None,
            core.OUTCOME: None,
        },
    )
    right = tuple(
        {
            core.PARTICIPANT: Fraction(0),
            core.INTERVENTION: Fraction(0),
            core.OUTCOME: Fraction(0),
        }
        for _ in range(6)
    ) + (
        {
            core.PARTICIPANT: None,
            core.INTERVENTION: None,
            core.OUTCOME: None,
        },
    )
    comparison = core.compare_abstract_arms(left, right)
    assert comparison.paired_deltas == (Fraction(7, 12),) * 6
    assert comparison.mean_delta == Fraction(7, 12)
    assert comparison.zero_defined_abstract_count == 1
    assert comparison.sign_test.one_sided_p == Fraction(1, 64)
    assert comparison.family_deltas == {
        core.PARTICIPANT: Fraction(1),
        core.INTERVENTION: Fraction(1, 2),
        core.OUTCOME: Fraction(1, 4),
    }


def test_hmac_assignment_uses_exact_message_and_no_replacement() -> None:
    secret = bytes(range(32))
    pmid = "12345678"
    expected = hmac.new(
        secret,
        (
            core.STUDY_ID.encode("utf-8")
            + b"\x00TRAIN\x00"
            + pmid.encode("ascii")
        ),
        hashlib.sha256,
    ).digest()
    assert core.hmac_assignment_digest(secret, "TRAIN", pmid) == expected
    pmids = ("9", "2", "10", "1")
    ordered = core.hmac_assignment_order(pmids, secret, "TRAIN")
    manual = tuple(
        row[2]
        for row in sorted(
            (
                core.hmac_assignment_digest(secret, "TRAIN", value),
                value.encode("ascii"),
                value,
            )
            for value in pmids
        )
    )
    assert ordered == manual
    assignment = core.assign_hmac_blocks(
        pmids,
        secret,
        "TRAIN",
        (("G_form", 2), ("A_form", 1)),
    )
    assert assignment.as_mapping() == {
        "G_form": ordered[:2],
        "A_form": ordered[2:3],
    }
    assert assignment.unused == ordered[3:]
    with pytest.raises(core.EbmNlpP1CoreError, match="capacity"):
        core.assign_hmac_blocks(
            pmids, secret, "TRAIN", (("too_large", 5),)
        )
