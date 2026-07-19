from __future__ import annotations

from fractions import Fraction
import math

import pytest

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _item(index: int = 0, family: str = "CAUSAL") -> core.ValidatedActionItem:
    sentences = tuple(f"Synthetic document {index} sentence {ordinal}." for ordinal in range(8))
    embeddings = tuple(
        _normalized((1.0, float(ordinal + 1) / 20.0, 0.1, 0.2))
        for ordinal in range(8)
    )
    events = (
        core.Event(0, "Attack", (core.Mention("attacked", 0), core.Mention("assault", 2))),
        core.Event(1, "Response", (core.Mention("responded", 1), core.Mention("reply", 3))),
        core.Event(2, "Process", (core.Mention("mediation", 4),)),
        core.Event(3, "Other", (core.Mention("unrelated", 6),)),
    )
    query = core.serialize_common_query(events[0], events[1])
    family_index = core.FAMILY_ORDER.index(family)
    wrong_index = (family_index + 1) % len(core.FAMILY_ORDER)
    scores: list[tuple[int, int, int]] = []
    for ordinal in range(8):
        row = [-3_000_000, -3_000_000, -3_000_000]
        row[wrong_index] = 2_000_000
        if ordinal == 4:
            row[family_index] = 7_000_000
            row[wrong_index] = -2_000_000
        scores.append(tuple(row))  # type: ignore[arg-type]
    return core.validate_action_item(
        sentences=sentences,
        sentence_embeddings=embeddings,
        events=events,
        head_event=0,
        tail_event=1,
        generic_relations=(core.GenericRelation(0, 2), core.GenericRelation(1, 2)),
        common_query=query,
        query_embedding=_normalized((1.0, 0.0, 0.1, 0.2)),
        sentence_family_nli_scores=scores,
    )


def _manual_g8(path_weight: float = 20.0) -> core.G8Model:
    weights = [0.0] * len(core.G8_FEATURE_ORDER)
    weights[core.G8_FEATURE_ORDER.index("generic_two_edge_path_terminal_fraction")] = path_weight
    return core.G8Model(
        weights=tuple(weights),
        normal_equation_sha256="1" * 64,
        observation_weight_sha256="2" * 64,
        centered_target_sha256="3" * 64,
        coefficient_sha256="4" * 64,
        fit_sha256="5" * 64,
    )


def test_action_item_has_no_family_argument_and_rejects_query_pair_edge() -> None:
    item = _item()
    assert "CAUSAL" not in item.common_query
    assert core.predict_family(item, (0, 1, 4)) == "CAUSAL"
    with pytest.raises(core.MavenEreG8E1Error, match="leaked"):
        core.validate_action_item(
            sentences=item.sentences,
            sentence_embeddings=item.sentence_embeddings,
            events=item.events,
            head_event=0,
            tail_event=1,
            generic_relations=(core.GenericRelation(0, 1),),
            common_query=item.common_query,
            query_embedding=item.query_embedding,
            sentence_family_nli_scores=item.sentence_family_nli_scores,
        )


def test_closed_action_space_and_features_are_deterministic() -> None:
    item = _item()
    first = core.build_action_space(item)
    second = core.build_action_space(item)
    assert first == second
    assert first.authorized_ordinals == tuple(range(8))
    assert len(tuple(core.iter_authorized_set3(first))) == 56
    phi = core.phi_features(first, (0, 1, 4))
    assert len(phi) == len(core.G8_FEATURE_ORDER)
    assert phi[core.G8_FEATURE_ORDER.index("head_and_tail_set_coverage_indicator")] == 1.0
    assert phi[core.G8_FEATURE_ORDER.index("generic_two_edge_path_terminal_fraction")] > 0


def test_classifier_and_binary_utility_use_only_late_label_api() -> None:
    item = _item(family="SUBEVENT")
    assert core.predict_family(item, (0, 1, 4)) == "SUBEVENT"
    assert core.utility((0, 1, 4), "SUBEVENT", item) == 1
    assert core.utility((0, 1, 2), "SUBEVENT", item) == 0
    with pytest.raises(core.MavenEreG8E1Error, match="unknown"):
        core.labelled_item(item, "CAUSE")


def test_manual_generator_is_behavior_identifiable_under_path_deletion() -> None:
    item = _item()
    model = _manual_g8()
    frontier = core.g8_frontier(item, model)
    assert 4 in frontier.e0.ordinals
    receipts = core.edge_deletion_redecode(item, model)
    path_rows = [row for row in receipts if row.witness.kind == "GENERIC_TWO_EDGE_PATH"]
    assert path_rows
    assert any(row.e0_changed for row in path_rows)
    assert core.behavior_hash(item, core.build_action_space(item), frontier, frontier.e0.ordinals)


def test_formal_size_g8_and_e1_fits_are_repeat_exact() -> None:
    g_examples = [
        core.labelled_item(_item(family_index * 32 + offset, family), family)
        for family_index, family in enumerate(core.FAMILY_ORDER)
        for offset in range(32)
    ]
    first_g8 = core.fit_g8(g_examples)
    second_g8 = core.fit_g8(tuple(reversed(g_examples)))
    assert first_g8 == second_g8
    assert first_g8.item_count == 96
    assert first_g8.set_observation_count == 96 * 56

    a_examples = [
        core.labelled_item(_item(10_000 + family_index * 16 + offset, family), family)
        for family_index, family in enumerate(core.FAMILY_ORDER)
        for offset in range(16)
    ]
    first_e1 = core.fit_e1(a_examples, first_g8)
    second_e1 = core.fit_e1(tuple(reversed(a_examples)), first_g8)
    assert first_e1 == second_e1
    assert first_e1.oriented_pair_count == 11_520
    item = a_examples[0].item
    space = core.build_action_space(item)
    frontier = core.g8_frontier(item, first_g8, space=space)
    selected = core.e1_select(space, frontier, first_e1)
    assert selected.entry in frontier.entries
    assert core.g8_model_payload(first_g8)["fit_sha256"] == first_g8.fit_sha256
    assert core.e1_model_payload(first_e1)["fit_sha256"] == first_e1.fit_sha256


def test_raw3_and_exact_sign_flip() -> None:
    item = _item()
    raw = core.raw3(item)
    assert raw == (0, 1, 2)
    result = core.exact_sign_flip((1, 1, 0, -1))
    assert result.observed_sum == 1
    assert result.nonzero_pair_count == 3
    assert result.p_value == Fraction(1, 2)
    with pytest.raises(core.MavenEreG8E1Error):
        core.exact_sign_flip((2,))
