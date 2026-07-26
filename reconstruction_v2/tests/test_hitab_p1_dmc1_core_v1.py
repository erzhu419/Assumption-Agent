from __future__ import annotations

from fractions import Fraction

import pytest

from assumption_agent.benchmarks.hitab_p1_dmc1_core_v1 import (
    FEATURE_NAMES,
    HMAC_EXPLORATION,
    TOP_K,
    TOP_V0,
    HitabDmc1CoreError,
    PrecomputedView,
    ProofDNF,
    TypedEdge,
    action_features,
    build_and_seal_aform_registry,
    compare_paired,
    fit_e1,
    label_sealed_registry,
    model_payload,
    registry_payload,
    select_e0,
    select_e1,
    set_utility,
    stable_hash,
    view_from_mapping,
)


def _view(
    *,
    suffix: str,
    permutation: tuple[int, ...] = tuple(range(10)),
) -> tuple[PrecomputedView, ProofDNF]:
    """Ten-unit complementarity fixture with permutable local ordinals."""

    # Logical units 0..4 are high-relevance copies of one requirement.  Logical
    # units 5..7 are lower-base-score but complementary requirements.  Units
    # 8..9 are low-scoring distractors that keep the fixture inside the formal
    # 10-unit minimum.
    ce_logical = (
        (950_000, 940_000, 930_000, 920_000, 910_000, 320_000, 300_000, 280_000, 20_000, 10_000),
        (160_000, 150_000, 140_000, 130_000, 120_000, 850_000, 820_000, 790_000, 30_000, 20_000),
        (110_000, 100_000, 90_000, 80_000, 70_000, 240_000, 860_000, 810_000, 10_000, 30_000),
    )
    minilm_logical = (
        (920_000, 910_000, 900_000, 890_000, 880_000, 300_000, 280_000, 260_000, 20_000, 10_000),
        (180_000, 170_000, 160_000, 150_000, 140_000, 830_000, 800_000, 770_000, 30_000, 20_000),
        (120_000, 110_000, 100_000, 90_000, 80_000, 220_000, 840_000, 790_000, 10_000, 30_000),
    )
    types_logical = (
        "DATA_CELL",
        "DATA_CELL",
        "DATA_CELL",
        "DATA_CELL",
        "DATA_CELL",
        "ROW_HEADER",
        "COLUMN_HEADER",
        "DERIVED_VALUE",
        "DATA_CELL",
        "DATA_CELL",
    )

    inverse = {logical: local for local, logical in enumerate(permutation)}
    ce = tuple(
        tuple(row[logical] for logical in permutation) for row in ce_logical
    )
    minilm = tuple(
        tuple(row[logical] for logical in permutation)
        for row in minilm_logical
    )
    unit_types = tuple(types_logical[logical] for logical in permutation)
    similarities = []
    for logical_left in permutation:
        row = []
        for logical_right in permutation:
            if logical_left == logical_right:
                row.append(1_000_000)
            elif logical_left < 5 and logical_right < 5:
                row.append(940_000)
            elif (
                logical_left >= 5
                and logical_right >= 5
                and logical_left != logical_right
            ):
                row.append(180_000)
            else:
                row.append(230_000)
        similarities.append(tuple(row))
    logical_edges = (
        (0, 5),
        (5, 6),
        (6, 7),
    )
    edges = tuple(
        sorted(
            TypedEdge(
                min(inverse[left], inverse[right]),
                max(inverse[left], inverse[right]),
                "FORWARD_SHARED_AXIS_OR_HEADER",
            )
            for left, right in logical_edges
        )
    )
    view = PrecomputedView(
        corpus_commitment=stable_hash(
            [f"fixture-unit-{logical}" for logical in permutation]
        ),
        question_facets=(
            f"entity facet {suffix}",
            f"metric facet {suffix}",
            f"operation facet {suffix}",
        ),
        unit_keys=tuple(f"U:{index}" for index in range(10)),
        unit_types=unit_types,
        typed_edges=edges,
        ce_facet_unit=ce,
        minilm_facet_unit=minilm,
        minilm_unit_unit=tuple(similarities),
    )
    proof = ProofDNF(
        alternatives=(
            (
                tuple(sorted(inverse[row] for row in range(5))),
                (inverse[5],),
                (inverse[6],),
                (inverse[7],),
            ),
        ),
        corpus_commitment=view.corpus_commitment,
    )
    return view, proof


def test_prelabel_archive_and_features_have_no_qrel_or_baseline_channel() -> None:
    view, proof = _view(suffix="train-a")
    registry = build_and_seal_aform_registry(
        view, exploration_key=b"fixed-a-form-key-000000000000"
    )
    before_payload = registry_payload(registry)
    before_hash = registry.seal_sha256
    before_features = tuple(
        action.phi.values
        for state in registry.states
        for action in state.actions
    )

    labelled = label_sealed_registry(registry, proof)
    after_features = tuple(
        action.phi.values
        for state in labelled.labelled_states
        for action in state.actions
    )
    assert registry.seal_sha256 == before_hash
    assert registry_payload(registry) == before_payload
    assert before_features == after_features
    serialized = str(before_payload).casefold()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "qrel",
            "gold",
            "proof",
            "family",
            "raw_rank",
            "hipporag",
            "recipe_id",
        )
    )
    assert any(action.target_y > 0 for state in labelled.labelled_states for action in state.actions)


def test_aform_keeps_v0_and_hmac_states_and_all_remaining_actions() -> None:
    view, _proof = _view(suffix="topology")
    registry = build_and_seal_aform_registry(
        view, exploration_key=b"fixed-topology-key-00000000000"
    )
    assert {state.state_class for state in registry.states} == {
        TOP_V0,
        HMAC_EXPLORATION,
    }
    for depth in range(TOP_K):
        states = [state for state in registry.states if state.depth == depth]
        assert 1 <= sum(state.state_class == TOP_V0 for state in states) <= 8
        assert sum(state.state_class == HMAC_EXPLORATION for state in states) <= 8
        assert len(states) <= 16
        for state in states:
            assert len(state.actions) == view.unit_count - depth
            assert tuple(action.candidate_ordinal for action in state.actions) == tuple(
                row for row in range(view.unit_count) if row not in state.selected_ordinals
            )
            assert all(len(action.phi.values) == len(FEATURE_NAMES) for action in state.actions)


def test_e1_learns_complementarity_and_improves_untouched_synthetic() -> None:
    training = []
    for index, permutation in enumerate(
        (
            tuple(range(10)),
            (3, 4, 5, 6, 7, 8, 9, 0, 1, 2),
            (6, 2, 7, 1, 5, 8, 0, 9, 4, 3),
        )
    ):
        view, proof = _view(suffix=f"train-{index}", permutation=permutation)
        registry = build_and_seal_aform_registry(
            view,
            exploration_key=f"training-key-{index:02d}-fixed-000000".encode(),
        )
        training.append(label_sealed_registry(registry, proof))
    model = fit_e1(training)

    untouched_view, untouched_proof = _view(
        suffix="untouched",
        permutation=(7, 4, 1, 6, 0, 8, 5, 3, 9, 2),
    )
    e0 = select_e0(untouched_view)
    e1 = select_e1(untouched_view, model)
    e0_utility = set_utility(
        tuple(sorted(e0)), untouched_proof, unit_count=untouched_view.unit_count
    )
    e1_utility = set_utility(
        tuple(sorted(e1)), untouched_proof, unit_count=untouched_view.unit_count
    )
    assert len(e0) == len(set(e0)) == TOP_K
    assert len(e1) == len(set(e1)) == TOP_K
    assert e1_utility > e0_utility
    assert model_payload(model)["intercept"] is False
    assert model_payload(model)["ridge_lambda"] == {
        "denominator": 1,
        "numerator": 1,
    }
    assert model_payload(model)["training_corpus_set_commitment"] == stable_hash(
        sorted(row.corpus_commitment for row in training)
    )
    assert model_payload(model)[
        "training_corpus_qrel_binding_set_commitment"
    ] == stable_hash(
        sorted(
            [row.corpus_commitment, row.ordinal_mapping_commitment]
            for row in training
        )
    )


def test_family_is_external_only_and_rejected_from_feature_input() -> None:
    view, _proof = _view(suffix="family")
    payload = view.payload()
    payload["family"] = "LOOKUP"
    with pytest.raises(HitabDmc1CoreError, match="forbidden=.*family"):
        view_from_mapping(payload)

    payload = view.payload()
    payload["raw_output"] = [0, 1, 2, 3, 4]
    with pytest.raises(HitabDmc1CoreError, match="raw_output"):
        view_from_mapping(payload)

    phi = action_features(view, (), 0)
    assert tuple(FEATURE_NAMES) == (
        "candidate_ce_max",
        "candidate_ce_mean",
        "candidate_minilm_max",
        "candidate_minilm_mean",
        "ce_residual_facet_coverage_gain",
        "minilm_residual_facet_coverage_gain",
        "source_native_type_novelty",
        "typed_incoming_from_selected_count",
        "typed_outgoing_to_selected_count",
        "pairwise_minilm_nonredundancy_gain",
    )
    assert len(phi.values) == 10


def test_dnf_alternatives_are_never_unioned() -> None:
    proof = ProofDNF(
        alternatives=(
            ((0,), (1,)),
            ((2,), (3,)),
        ),
        corpus_commitment=stable_hash(
            [f"dnf-unit-{ordinal}" for ordinal in range(5)]
        ),
    )
    # Unit 0 covers half of alternative A and unit 2 covers half of B.  A
    # forbidden union would call this complete and return 2; max-per-proof is
    # exactly one half.
    assert set_utility((0, 2), proof, unit_count=5) == Fraction(1, 2)
    assert set_utility((0, 1), proof, unit_count=5) == Fraction(2, 1)


def test_qrel_corpus_binding_rejects_same_size_cross_item_swap() -> None:
    left_view, left_proof = _view(suffix="binding-left")
    right_view, _right_proof = _view(
        suffix="binding-right",
        permutation=(1, 0, 2, 3, 4, 5, 6, 7, 8, 9),
    )
    right_registry = build_and_seal_aform_registry(
        right_view,
        exploration_key=b"binding-right-key-000000000000",
    )
    assert left_view.unit_count == right_view.unit_count
    assert left_proof.corpus_commitment != right_registry.corpus_commitment
    with pytest.raises(HitabDmc1CoreError, match="corpus commitments differ"):
        label_sealed_registry(right_registry, left_proof)


def test_same_input_is_bit_deterministic_and_exact_sign_flip_is_exact() -> None:
    view, proof = _view(suffix="determinism")
    key = b"deterministic-key-0000000000000"
    left = build_and_seal_aform_registry(view, exploration_key=key)
    right = build_and_seal_aform_registry(view, exploration_key=key)
    assert registry_payload(left) == registry_payload(right)
    assert left.seal_sha256 == stable_hash(
        {key: value for key, value in registry_payload(left).items() if key != "self_sha256"}
    )
    assert label_sealed_registry(left, proof) == label_sealed_registry(right, proof)
    assert select_e0(view) == select_e0(view)

    comparison = compare_paired(
        [Fraction(2), Fraction(3, 2), Fraction(1)],
        [Fraction(1), Fraction(1), Fraction(1)],
    )
    assert comparison.net_utility == Fraction(3, 2)
    assert comparison.positive_count == 2
    assert comparison.tie_count == 1
    assert comparison.reference_tail == Fraction(1, 4)
