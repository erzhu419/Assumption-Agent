from __future__ import annotations

from fractions import Fraction
import inspect
import itertools
import math

import numpy as np
import pytest

from assumption_agent.benchmarks import docred_structured_set_decoder_g8_e1_v1 as core


ZERO_HASH = "0" * 64


def _item_kwargs(
    tag: str = "base",
    *,
    sentence_embeddings: tuple[tuple[float, float], ...] | None = None,
) -> dict[str, object]:
    if sentence_embeddings is None:
        sentence_embeddings = tuple(
            (float(10 - ordinal), float(ordinal + 1))
            for ordinal in range(10)
        )
    entities = (
        core.Entity(
            0,
            (
                core.Mention(f"Head-{tag}", 0, "ORG"),
                core.Mention(f"head-{tag}", 1, "ORG"),
                core.Mention(f"H-{tag}", 4, "ORG"),
            ),
        ),
        core.Entity(
            1,
            (
                core.Mention(f"Tail-{tag}", 2, "ORG"),
                core.Mention(f"tail-{tag}", 3, "ORG"),
                core.Mention(f"T-{tag}", 4, "ORG"),
            ),
        ),
        core.Entity(
            2,
            (
                core.Mention(f"Bridge-{tag}", 0, "LOC"),
                core.Mention(f"Bridge-{tag}", 2, "LOC"),
            ),
        ),
        core.Entity(
            3,
            (
                core.Mention(f"Aux-{tag}", 8, "MISC"),
                core.Mention(f"Aux-{tag}", 9, "MISC"),
            ),
        ),
    )
    return {
        "sentence_tokens": tuple(
            (f"Sentence-{tag}-{ordinal}", "evidence")
            for ordinal in range(10)
        ),
        "sentence_embeddings": sentence_embeddings,
        "entities": entities,
        "head_entity": 0,
        "tail_entity": 1,
        "relation_description": "is structurally related to",
        "full_query_embedding": (1.0, 0.0),
        "relation_description_embedding": (1.0, 0.0),
    }


def _item(tag: str = "base") -> core.ValidatedActionItem:
    return core.validate_action_item(**_item_kwargs(tag))


def _g8(weights: tuple[float, ...] | None = None) -> core.G8Model:
    return core.G8Model(
        weights=weights or (0.0,) * len(core.G8_FEATURE_ORDER),
        normal_equation_sha256=ZERO_HASH,
        observation_weight_sha256=ZERO_HASH,
        centered_target_sha256=ZERO_HASH,
        coefficient_sha256=ZERO_HASH,
        fit_sha256=ZERO_HASH,
    )


def _e1(
    *,
    weights: tuple[float, ...] | None = None,
    stds: tuple[float, ...] | None = None,
) -> core.E1Model:
    return core.E1Model(
        weights=weights or (0.0,) * len(core.E1_FEATURE_ORDER),
        feature_stds=stds or (1.0,) * len(core.E1_FEATURE_ORDER),
        normal_equation_sha256=ZERO_HASH,
        observation_weight_sha256=ZERO_HASH,
        target_sha256=ZERO_HASH,
        coefficient_sha256=ZERO_HASH,
        fit_sha256=ZERO_HASH,
    )


def _labelled_items(*, per_family: int) -> tuple[core.LabelledItem, ...]:
    rows: list[core.LabelledItem] = []
    for family in core.FAMILY_ORDER:
        for index in range(per_family):
            item = _item(f"{family}-{index}")
            gold_by_index = ((0,), (0, 2), (0, 2, 4))
            rows.append(
                core.labelled_item(
                    item,
                    gold_by_index[index % len(gold_by_index)],
                    family,
                )
            )
    return tuple(rows)


def test_render_query_aliases_and_label_free_api_boundary() -> None:
    assert core.render_sentence(("New York", "works")) == "New York works"
    assert core.canonical_aliases((" Alpha ", "alpha", "ＡＬＰＨＡ", "Beta")) == (
        " Alpha ",
        "Beta",
    )
    assert core.serialize_common_query(("H",), "owns", ("T",)) == (
        "HEAD: H\nRELATION: owns\nTAIL: T"
    )
    assert "gold" not in inspect.signature(core.validate_action_item).parameters
    with pytest.raises(core.DocredStructuredSetDecoderError):
        core.render_sentence(("bad\x00token",))
    with pytest.raises(core.DocredStructuredSetDecoderError):
        core.render_sentence(("\ud800",))


def test_validate_item_is_strict_and_gold_is_separate() -> None:
    item = _item()
    assert item.sentence_count == 10
    assert item.embedding_dimension == 2
    assert item.common_query.startswith("HEAD: Head-base | H-base\n")
    assert core.validate_gold(item, (4, 0)).ordinals == (0, 4)

    bad = _item_kwargs("nan")
    embeddings = list(bad["sentence_embeddings"])
    embeddings[0] = (math.nan, 1.0)
    bad["sentence_embeddings"] = tuple(embeddings)
    with pytest.raises(core.DocredStructuredSetDecoderError, match="finite"):
        core.validate_action_item(**bad)

    bad = _item_kwargs("ids")
    entities = list(bad["entities"])
    entities[0] = core.Entity(7, entities[0].mentions)
    bad["entities"] = tuple(entities)
    with pytest.raises(core.DocredStructuredSetDecoderError, match="contiguous"):
        core.validate_action_item(**bad)

    with pytest.raises(core.DocredStructuredSetDecoderError, match="unique"):
        core.validate_gold(item, (0, 0))
    with pytest.raises(core.DocredStructuredSetDecoderError, match="one through three"):
        core.validate_gold(item, (0, 1, 2, 3))


def test_q6_ties_raw3_and_complete_typed_grammar() -> None:
    item = _item()
    space = core.build_action_space(item)
    authorization = space.authorization_map()

    assert space.authorized_ordinals == tuple(range(10))
    assert all("QUERY" in authorization[index].kinds for index in range(8))
    assert all("QUERY" not in authorization[index].kinds for index in (8, 9))
    assert "HEAD" in authorization[0].kinds
    assert "TAIL" in authorization[2].kinds
    assert "DIRECT" in authorization[4].kinds
    bridge = core.Witness("BRIDGE", 2, 0, 2)
    assert bridge in authorization[0].witnesses
    assert bridge in authorization[2].witnesses
    assert all(
        set(row.kinds).issubset(core.AUTHORITY_KIND_ORDER)
        for row in space.authorizations
    )

    sets = tuple(core.iter_authorized_set3(space))
    assert len(sets) == math.comb(10, 3)
    assert sets == tuple(itertools.combinations(range(10), 3))

    without_bridge = core.build_action_space(item, deleted_witnesses=(bridge,))
    assert bridge not in without_bridge.authorization_map()[0].witnesses
    assert "BRIDGE" not in without_bridge.authorization_map()[0].kinds

    tied = core.validate_action_item(
        **_item_kwargs(
            "tie",
            sentence_embeddings=((1.0, 0.0),) * 10,
        )
    )
    assert core.q6_cosine((1.0, 0.0), (1.0, 0.0)) == 1_000_000
    assert core.raw3(tied) == (0, 1, 2)
    tied_space = core.build_action_space(tied)
    assert tuple(
        index
        for index, row in enumerate(tied_space.authorizations)
        if "QUERY" in row.kinds
    ) == tuple(range(8))


def test_phi_and_exact_x6_utility_for_all_gold_sizes() -> None:
    item = _item()
    space = core.build_action_space(item)
    phi = core.phi_features(space, (0, 2, 4))
    position = core.G8_FEATURE_ORDER.index
    assert len(phi) == 12
    assert phi[position("head_mention_terminal_fraction")] == pytest.approx(2 / 3)
    assert phi[position("tail_mention_terminal_fraction")] == pytest.approx(2 / 3)
    assert phi[position("direct_head_tail_terminal_fraction")] == pytest.approx(1 / 3)
    assert phi[position("head_and_tail_set_coverage_indicator")] == 1.0
    assert phi[position("one_bridge_witness_pair_fraction")] == pytest.approx(1 / 3)
    assert phi[position("shared_entity_connected_pair_fraction")] == 1.0
    assert phi[position("query_atom_terminal_fraction")] == 1.0
    assert phi[position("authorization_kind_coverage_fraction")] == 1.0

    gold1 = core.validate_gold(item, (0,))
    gold2 = core.validate_gold(item, (0, 2))
    gold3 = core.validate_gold(item, (0, 2, 4))
    assert core.utility_x6((0, 8, 9), gold1) == 12
    assert core.utility_x6((0, 8, 9), gold2) == 3
    assert core.utility_x6((0, 2, 9), gold2) == 12
    assert core.utility_x6((0, 2, 9), gold3) == 4
    assert core.utility_x6((0, 2, 4), gold3) == 12
    assert core.utility((0, 2, 9), gold3) == Fraction(2, 3)


def test_streaming_g8_statistics_match_brute_force() -> None:
    item = _item("stats")
    gold = core.validate_gold(item, (0, 2, 4))
    space = core.build_action_space(item)
    selected_rows = tuple(core.iter_authorized_set3(space))
    features = np.asarray(
        [core.phi_features(space, selected) for selected in selected_rows],
        dtype=np.float64,
    )
    targets = np.asarray(
        [core.utility_x6(selected, gold) / 6.0 for selected in selected_rows],
        dtype=np.float64,
    )
    centered_features = features - features.mean(axis=0)
    centered_targets = targets - targets.mean()

    stats = core.g8_item_sufficient_statistics(item, gold, space=space)
    assert stats.set_count == 120
    np.testing.assert_allclose(stats.mean_phi, features.mean(axis=0), rtol=0, atol=1e-15)
    assert stats.mean_target == pytest.approx(float(targets.mean()), abs=1e-15)
    np.testing.assert_allclose(
        stats.centered_xx,
        centered_features.T @ centered_features,
        rtol=0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        stats.centered_xy,
        centered_features.T @ centered_targets,
        rtol=0,
        atol=2e-13,
    )
    assert len(stats.centered_target_sha256) == 64


def test_frontier_is_exact_top16_and_psi_is_fixed() -> None:
    item = _item("frontier")
    space = core.build_action_space(item)
    frontier = core.g8_frontier(item, _g8(), space=space)
    expected = tuple(itertools.islice(core.iter_authorized_set3(space), 16))
    assert tuple(entry.ordinals for entry in frontier.entries) == expected
    assert all(entry.generator_energy == 0.0 for entry in frontier.entries)
    psi = core.psi_features(space, frontier.e0)
    assert len(psi) == len(core.E1_FEATURE_ORDER) == 8
    assert all(math.isfinite(value) for value in psi)
    assert core.e1_select(space, frontier, _e1()).entry == frontier.e0


def test_pairwise_ridge_and_deployment_formula_use_frozen_scales() -> None:
    solution = core.solve_standardized_pairwise_ridge(
        ((2.0, 0.0), (-2.0, 0.0)),
        (1.0, -1.0),
        row_weight=0.5,
    )
    assert solution.feature_stds == (2.0, 0.0)
    assert solution.weights == pytest.approx((0.5, 0.0))

    model = _e1(
        weights=(0.5,) + (7.0,) * 7,
        stds=(2.0,) + (0.0,) * 7,
    )
    assert core.e1_score(model, (4.0,) + (999.0,) * 7) == 1.0
    assert core.E1_DEPLOYMENT_FORMULA == (
        "sum_beta_std_times_Psi_div_sigma_pair_zero_variance_zero_no_intercept"
    )
    with pytest.raises(core.DocredStructuredSetDecoderError, match="sum to one"):
        core.solve_standardized_pairwise_ridge(
            ((1.0,), (-1.0,)),
            (1.0, -1.0),
            row_weight=0.4,
        )


def test_formal_g8_and_e1_fit_exact_declared_counts() -> None:
    g_examples = _labelled_items(per_family=32)
    with pytest.raises(core.DocredStructuredSetDecoderError, match="count"):
        core.fit_g8(g_examples[:-1])
    g8_model = core.fit_g8(g_examples)
    assert g8_model.item_count == 96
    assert g8_model.set_observation_count == 96 * 120
    assert len(g8_model.weights) == 12
    assert all(math.isfinite(value) for value in g8_model.weights)
    assert len(g8_model.fit_sha256) == 64

    a_examples = _labelled_items(per_family=16)
    with pytest.raises(core.DocredStructuredSetDecoderError, match="count"):
        core.fit_e1(a_examples[:-1], g8_model)
    e1_model = core.fit_e1(a_examples, g8_model)
    assert e1_model.item_count == 48
    assert e1_model.oriented_pair_count == 48 * 16 * 15
    assert len(e1_model.weights) == 8
    assert len(e1_model.feature_stds) == 8
    assert all(math.isfinite(value) for value in e1_model.weights)
    assert len(e1_model.fit_sha256) == 64


def test_controls_are_deterministic_label_free_and_exact() -> None:
    item = _item("controls")
    kind_weights = [0.0] * len(core.G8_FEATURE_ORDER)
    kind_weights[
        core.G8_FEATURE_ORDER.index("authorization_kind_coverage_fraction")
    ] = 10.0
    model = _g8(tuple(kind_weights))
    space = core.build_action_space(item)
    frontier = core.g8_frontier(item, model, space=space)
    first_hash = core.behavior_hash(item, space, frontier, frontier.e0.ordinals)
    assert first_hash == core.behavior_hash(
        item, space, frontier, frontier.e0.ordinals
    )
    assert len(first_hash) == 64

    receipts = core.edge_deletion_redecode(item, model)
    assert receipts
    assert all(row.e1_before is None and row.e1_after is None for row in receipts)
    assert any(row.witness.kind == "DIRECT" for row in receipts)
    assert core.edge_deletion_action_change_count((receipts,)) in {0, 1}

    result = core.exact_sign_flip_x6((1, 2, 0))
    assert result.observed_sum_x6 == 3
    assert result.nonzero_pair_count == 2
    assert result.assignment_count == 4
    assert result.tail_count == 1
    assert result.p_value == Fraction(1, 4)

    assert core.outside_raw3_count(((0, 3, 4),), ((0, 1, 2),)) == 2
    with pytest.raises(core.DocredStructuredSetDecoderError, match="unique"):
        core.outside_raw3_count(((0, 0, 1),), ((0, 1, 2),))
