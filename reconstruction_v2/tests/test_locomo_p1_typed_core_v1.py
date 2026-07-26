from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import inspect

import pytest

from assumption_agent.benchmarks import locomo_p1_typed_core_v1 as core


def _entity_turns(*, suffix: str = "") -> tuple[core.Turn, ...]:
    distractors = [
        core.Turn(
            ordinal=index,
            session_ordinal=index // 3,
            speaker="Bob",
            date=f"2020-01-{index + 1:02d}",
            text=f"color choose decision report color choose {suffix}".strip(),
        )
        for index in range(7)
    ]
    alice = [
        core.Turn(
            ordinal=ordinal,
            session_ordinal=5 + index,
            speaker="Alice",
            date=f"2022-02-{index + 1:02d}",
            text=f"private response {index} {suffix}".strip(),
        )
        for index, ordinal in enumerate((10, 12, 14, 16, 18))
    ]
    return tuple(distractors + alice)


def _utility(
    action: core.RecipeAction, relevant_ordinals: frozenset[int]
) -> Fraction:
    return Fraction(
        len(set(action.top5_turn_ordinals) & relevant_ordinals),
        len(relevant_ordinals),
    )


def _utilities(
    slate: core.ActionSlate, relevant_ordinals: frozenset[int]
) -> dict[str, Fraction]:
    return {
        recipe: _utility(slate.action(recipe), relevant_ordinals)
        for recipe in core.RECIPE_IDS
    }


def test_public_projection_and_action_api_exclude_private_or_label_fields() -> None:
    assert tuple(field.name for field in fields(core.Turn)) == (
        "ordinal",
        "session_ordinal",
        "speaker",
        "date",
        "text",
    )
    assert tuple(inspect.signature(core.build_action_slate).parameters) == (
        "question",
        "turns",
    )
    assert tuple(field.name for field in fields(core.AFormExample)) == (
        "features",
        "utilities",
    )
    public = {
        "ordinal": 0,
        "session_ordinal": 0,
        "speaker": "Alice",
        "date": "2022-01-01",
        "text": "A public turn.",
    }
    assert core.turn_public_payload(core.turn_from_public_fields(public)) == public
    for forbidden in (
        "answer",
        "category",
        "conversation_id",
        "evidence",
        "family",
    ):
        with pytest.raises(
            core.LocomoP1TypedCoreError,
            match="exact public field set",
        ):
            core.turn_from_public_fields({**public, forbidden: "forbidden"})


def test_six_recipes_are_deterministic_unique_and_raw_is_full_query_bm25() -> None:
    question = "What color did Alice choose?"
    turns = _entity_turns()
    first = core.build_action_slate(question, turns)
    second = core.build_action_slate(question, tuple(reversed(turns)))
    assert first == second
    assert tuple(action.recipe_id for action in first.actions) == core.RECIPE_IDS
    assert len(core.RECIPE_IDS) == 6
    raw_scores = core.bm25_scores(question, [turn.text for turn in turns])
    expected_raw = tuple(
        turns[index].ordinal
        for index in sorted(
            range(len(turns)),
            key=lambda index: (-raw_scores[index], turns[index].ordinal),
        )[: core.TOP_K]
    )
    raw = first.action(core.R0_RAW_BM25)
    assert raw.top5_turn_ordinals == expected_raw
    for action in first.actions:
        assert len(action.top5_turn_ordinals) == core.TOP_K
        assert len(set(action.top5_turn_ordinals)) == core.TOP_K
        assert action.raw_top5_turn_ordinals == expected_raw
    audit = first.audit_payload()
    assert audit["feature_names"] == list(core.FEATURE_NAMES)
    assert len(audit["turn_structures"]) == len(turns)
    assert len(audit["features"]) == len(core.RECIPE_IDS)
    claimed = audit.pop("self_sha256")
    assert claimed == core.stable_hash(audit)


def test_entity_metadata_has_a_causal_effect_over_raw_text_only() -> None:
    question = "What color did Alice choose?"
    slate = core.build_action_slate(question, _entity_turns())
    alice_ordinals = {10, 12, 14, 16, 18}
    raw = slate.action(core.R0_RAW_BM25)
    entity = slate.action(core.R1_ENTITY_FOCUS)
    assert set(raw.top5_turn_ordinals).isdisjoint(alice_ordinals)
    assert set(entity.top5_turn_ordinals) == alice_ordinals
    entity_features = slate.feature_row(core.R1_ENTITY_FOCUS).payload()
    assert entity_features["entity_turn_count"] == 5
    assert entity_features["speaker_turn_count"] == 5
    assert entity_features["raw_overlap_count"] == 0


def test_date_metadata_has_a_causal_effect_over_raw_text_only() -> None:
    question = "What was approved in March 2021?"
    turns = tuple(
        [
            core.Turn(
                ordinal=index,
                session_ordinal=0,
                speaker="Nora",
                date=f"2020-01-{index + 1:02d}",
                text="approved report approved report",
            )
            for index in range(7)
        ]
        + [
            core.Turn(
                ordinal=20,
                session_ordinal=2,
                speaker="Nora",
                date="2021-03-14",
                text="the launch passed quietly",
            )
        ]
    )
    slate = core.build_action_slate(question, turns)
    assert 20 not in slate.action(core.R0_RAW_BM25).top5_turn_ordinals
    assert 20 in slate.action(core.R2_TEMPORAL_FOCUS).top5_turn_ordinals
    row = next(
        row for row in slate.turn_structures if row.turn_ordinal == 20
    )
    assert row.raw_bm25 == 0
    assert row.temporal_hits > 0


def test_typed_cascade_recovers_an_adjacent_turn_without_query_overlap() -> None:
    question = "Why was the Orion launch delayed?"
    turns = tuple(
        [
            core.Turn(
                ordinal=index,
                session_ordinal=0,
                speaker="Dana",
                date=f"2020-01-{index + 1:02d}",
                text="routine unrelated update",
            )
            for index in range(7)
        ]
        + [
            core.Turn(8, 2, "Dana", "2022-01-01", "Orion launch delayed"),
            core.Turn(9, 2, "Eli", "2022-01-01", "Valve failure"),
            core.Turn(10, 3, "Fran", "2023-01-01", "other note"),
        ]
    )
    slate = core.build_action_slate(question, turns)
    raw = slate.action(core.R0_RAW_BM25)
    cascade = slate.action(core.R5_TYPED_CASCADE)
    assert 9 not in raw.top5_turn_ordinals
    assert cascade.top5_turn_ordinals[:2] == (8, 9)
    row = next(
        row for row in slate.turn_structures if row.turn_ordinal == 9
    )
    assert row.raw_bm25 == 0
    assert (
        slate.feature_row(core.R5_TYPED_CASCADE)
        .payload()["adjacent_pair_count"]
        > 0
    )


def test_conservative_aform_e1_is_distinct_improves_and_applies_unchanged() -> None:
    relevant = frozenset({10, 12, 14, 16, 18})
    training_slates = tuple(
        core.build_action_slate(
            "What color did Alice choose?",
            _entity_turns(suffix=f"train{index}"),
        )
        for index in range(core.MIN_SIGNATURE_SUPPORT)
    )
    examples = tuple(
        core.make_aform_example(slate, _utilities(slate, relevant))
        for slate in training_slates
    )
    model = core.fit_e1(examples)
    assert model.training_stage == "A_form"
    assert model.training_item_count == core.MIN_SIGNATURE_SUPPORT
    assert any(
        rule.recipe_id == core.R1_ENTITY_FOCUS and rule.qualified
        for rule in model.rules
    )
    model_payload = model.payload()
    assert "family" not in model_payload["signature_feature_names"]
    assert "conversation_id" not in model_payload["signature_feature_names"]

    held = core.build_action_slate(
        "What color did Alice choose?",
        _entity_turns(suffix="held"),
    )
    e0 = core.apply_e0(held, stage="A_hold")
    decisions = tuple(
        core.apply_e1(model, held, stage=stage)
        for stage in core.POLICY_STAGES
    )
    assert e0.selected_recipe_id == core.R5_TYPED_CASCADE
    assert {decision.selected_recipe_id for decision in decisions} == {
        core.R1_ENTITY_FOCUS
    }
    assert {
        decision.top5_turn_ordinals for decision in decisions
    } == {held.action(core.R1_ENTITY_FOCUS).top5_turn_ordinals}
    e1 = decisions[1]
    assert e1.selected_recipe_id != e0.selected_recipe_id
    assert _utility(
        held.action(e1.selected_recipe_id), relevant
    ) > _utility(held.action(e0.selected_recipe_id), relevant)


def test_contradicted_or_under_supported_signature_falls_back_to_e0() -> None:
    slate = core.build_action_slate(
        "What color did Alice choose?", _entity_turns()
    )
    relevant = frozenset({10, 12, 14, 16, 18})
    favorable = _utilities(slate, relevant)
    contradicted = dict(favorable)
    contradicted[core.R1_ENTITY_FOCUS] = Fraction(0, 1)
    contradicted[core.R3_SPEAKER_EVENT] = Fraction(0, 1)
    examples = (
        core.make_aform_example(slate, favorable),
        core.make_aform_example(slate, favorable),
        core.make_aform_example(slate, contradicted),
    )
    model = core.fit_e1(examples)
    assert not any(rule.qualified for rule in model.rules)
    decision = core.apply_e1(model, slate, stage="F_search")
    assert decision.fallback_to_e0
    assert decision.selected_recipe_id == core.E0_RECIPE_ID
