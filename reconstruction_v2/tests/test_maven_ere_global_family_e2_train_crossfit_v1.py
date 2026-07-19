from __future__ import annotations

import math

import pytest

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_global_family_e2_train_crossfit_v1 as e2


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _item(index: int, family: str) -> core.ValidatedActionItem:
    family_index = core.FAMILY_ORDER.index(family)
    wrong = (family_index + 1) % len(core.FAMILY_ORDER)
    sentences = tuple(f"E2 synthetic {index} sentence {ordinal}." for ordinal in range(8))
    embeddings = tuple(
        _normalized((1.0, (8 - ordinal) / 50.0, 0.1, 0.2))
        for ordinal in range(8)
    )
    events = (
        core.Event(0, "Attack", (core.Mention("attacked", 0), core.Mention("assault", 2))),
        core.Event(1, "Response", (core.Mention("responded", 1), core.Mention("reply", 3))),
        core.Event(2, "Process", (core.Mention("mediation", 4),)),
    )
    scores = []
    for ordinal in range(8):
        row = [-5_000_000] * len(core.FAMILY_ORDER)
        if ordinal < 3:
            row[wrong] = 5_000_000 + index
            row[family_index] = -2_000_000
        else:
            row[family_index] = 8_000_000 + index
            row[wrong] = -3_000_000
        scores.append(tuple(row))
    query = core.serialize_common_query(events[0], events[1])
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


def _g8() -> core.G8Model:
    return core.G8Model(
        tuple(0.0 for _ in core.G8_FEATURE_ORDER),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )


def _examples() -> tuple[core.LabelledItem, ...]:
    return tuple(
        core.labelled_item(_item(family_index * 4 + repeat, family), family)
        for family_index, family in enumerate(core.FAMILY_ORDER)
        for repeat in range(4)
    )


def test_global_family_features_and_ridge_fit_are_deterministic() -> None:
    examples = _examples()
    model1 = e2.fit_e2(examples)
    model2 = e2.fit_e2(tuple(reversed(examples)))
    assert model1 == model2
    assert model1.item_count == 12
    assert len(model1.feature_means) == len(e2.FEATURE_ORDER) == 18
    assert all(
        e2.predict_item_family(model1, example.item) == example.family
        for example in examples
    )
    payload = e2.model_payload(model1)
    assert payload["fit_sha256"] == model1.fit_sha256
    assert payload["feature_order"] == list(e2.FEATURE_ORDER)


def test_e2_full_set_selection_can_recover_target_outside_e0() -> None:
    model = e2.fit_e2(_examples())
    item = _item(99, "CAUSAL")
    space = core.build_action_space(item)
    g8 = _g8()
    frontier = core.g8_frontier(item, g8, space=space)
    e0 = frontier.e0.ordinals
    selection = e2.e2_select(
        item,
        g8,
        model,
        space=space,
        frontier=frontier,
    )
    assert selection.target_family == "CAUSAL"
    assert selection.used_fallback is False
    assert core.utility(e0, "CAUSAL", item) == 0
    assert core.utility(selection.selected, "CAUSAL", item) == 1
    assert selection.selected != e0
    assert all(value in space.authorized_ordinals for value in selection.selected)


def test_e2_fit_rejects_family_imbalance() -> None:
    examples = _examples()
    with pytest.raises(e2.MavenEreE2Error, match="family-balanced"):
        e2.fit_e2(examples[:-1])
