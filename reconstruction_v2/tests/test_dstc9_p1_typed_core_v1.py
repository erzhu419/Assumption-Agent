from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import pytest

from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as core


def _history() -> tuple[core.DialogueTurn, ...]:
    return (
        core.DialogueTurn("U", "I need a hotel in the centre."),
        core.DialogueTurn("S", "I found Acorn Guest House."),
        core.DialogueTurn("U", "Does it allow pets?"),
    )


def _snippets(count: int = 25) -> tuple[core.KnowledgeSnippet, ...]:
    return tuple(
        core.KnowledgeSnippet(
            ordinal=index,
            entity_name=(
                f"Entity {index // 5}" if index % 2 == 0 else None
            ),
            title=f"Question {index}",
            body=f"Answer body {index}",
        )
        for index in range(count)
    )


def _group_scores(
    count: int,
    group_start: int,
) -> tuple[int, ...]:
    group = list(range(group_start, min(group_start + 5, count)))
    remainder = [index for index in range(count) if index not in group]
    order = group + remainder
    scores = [0] * count
    for rank, index in enumerate(order):
        scores[index] = count - rank
    return tuple(scores)


def _score_bundle(
    count: int = 25,
) -> dict[str, tuple[int, ...]]:
    return {
        "global_ce_scores": _group_scores(count, 0),
        "last_turn_ce_scores": _group_scores(count, 5),
        "minilm_scores": _group_scores(count, 20),
        "entity_scores": _group_scores(count, 5),
        "title_scores": _group_scores(count, 10),
        "body_scores": _group_scores(count, 15),
    }


def _slate(*, bucket: int = 0) -> core.ActionSlate:
    snippets = _snippets()
    return core.build_action_slate(
        _history(),
        snippets,
        **_score_bundle(len(snippets)),
        predicted_bucket=bucket,
    )


def _utility_vector(
    *,
    selected_recipe: str | None = None,
    selected_utility: int = 800_000,
    e0_utility: int = 500_000,
    default_utility: int = 400_000,
) -> tuple[int, ...]:
    values = [default_utility] * len(core.RECIPE_IDS)
    values[core.RECIPE_IDS.index(core.E0_RECIPE_ID)] = e0_utility
    if selected_recipe is not None:
        values[core.RECIPE_IDS.index(selected_recipe)] = selected_utility
    return tuple(values)


def test_query_canonicalization_uses_full_history_and_allows_same_speaker() -> None:
    first = (
        core.DialogueTurn("U", "  Ｈｅｌｌｏ\u00a0world\t"),
        core.DialogueTurn("U", "same\u2003speaker follow-up"),
        core.DialogueTurn("S", "Line one\nline two"),
        core.DialogueTurn("U", "Final question"),
    )
    second = (
        core.DialogueTurn("U", "Hello world"),
        core.DialogueTurn("U", "same speaker follow-up"),
        core.DialogueTurn("S", "Line one line two"),
        core.DialogueTurn("U", "Final question"),
    )
    expected = (
        "U: Hello world\n"
        "U: same speaker follow-up\n"
        "S: Line one line two\n"
        "U: Final question"
    )
    assert core.serialize_model_query(first) == expected
    assert core.serialize_model_query(second) == expected
    assert core.normalized_query_payload(first) == (
        core.normalized_query_payload(second)
    )
    assert core.normalized_query_sha256(first) == (
        core.normalized_query_sha256(second)
    )
    payload = core.normalized_query_payload(first)
    assert payload["model_query"] == expected
    assert [turn["speaker"] for turn in payload["turns"]] == [
        "U",
        "U",
        "S",
        "U",
    ]

    with pytest.raises(
        core.Dstc9P1TypedCoreError,
        match="forbidden control",
    ):
        core.DialogueTurn("U", "zero\u200bwidth")
    with pytest.raises(core.Dstc9P1TypedCoreError, match="empty"):
        core.DialogueTurn("U", "\t\u2003\n")
    with pytest.raises(core.Dstc9P1TypedCoreError, match="invalid"):
        core.DialogueTurn(
            "U",
            "x" * (core.MAX_TURN_CHARACTERS + 1),
        )


def test_public_projection_and_passage_contract_exclude_every_label_field() -> None:
    assert tuple(field.name for field in fields(core.DialogueTurn)) == (
        "speaker",
        "text",
    )
    assert tuple(field.name for field in fields(core.KnowledgeSnippet)) == (
        "ordinal",
        "entity_name",
        "title",
        "body",
    )
    assert tuple(inspect.signature(core.build_action_slate).parameters) == (
        "history",
        "snippets",
        "global_ce_scores",
        "last_turn_ce_scores",
        "minilm_scores",
        "entity_scores",
        "title_scores",
        "body_scores",
        "predicted_bucket",
    )
    assert tuple(inspect.signature(core.fit_e1).parameters) == ("examples",)
    assert tuple(field.name for field in fields(core.AFormExample)) == (
        "predicted_bucket",
        "utility_vector",
    )

    turn = {"speaker": "U", "text": "  A\u00a0question "}
    assert core.turn_public_payload(
        core.turn_from_public_fields(turn)
    ) == {"speaker": "U", "text": "A question"}

    public = {
        "ordinal": 7,
        "entity_name": "  Acorn\u00a0Guest House ",
        "title": " Are pets allowed? ",
        "body": " Pets are allowed. ",
    }
    snippet = core.snippet_from_public_fields(public)
    assert core.snippet_public_payload(snippet) == {
        "body": "Pets are allowed.",
        "entity_name": "Acorn Guest House",
        "ordinal": 7,
        "title": "Are pets allowed?",
    }
    assert core.serialize_passage(snippet) == (
        "ENTITY: Acorn Guest House\n"
        "TITLE: Are pets allowed?\n"
        "BODY: Pets are allowed."
    )

    no_entity = core.snippet_from_public_fields(
        {
            "ordinal": 8,
            "title": "Taxi payments",
            "body": "Cash and card are accepted.",
        }
    )
    assert no_entity.entity_name is None
    assert core.serialize_passage(no_entity) == (
        "TITLE: Taxi payments\nBODY: Cash and card are accepted."
    )

    forbidden_fields = (
        "domain",
        "entity_id",
        "doc_id",
        "family",
        "qrel",
        "response",
        "split",
        "target",
    )
    for forbidden in forbidden_fields:
        with pytest.raises(
            core.Dstc9P1TypedCoreError,
            match="exact public field set",
        ):
            core.turn_from_public_fields(
                {**turn, forbidden: "forbidden"}
            )
        with pytest.raises(
            core.Dstc9P1TypedCoreError,
            match="exact public field set",
        ):
            core.snippet_from_public_fields(
                {**public, forbidden: "forbidden"}
            )


def test_whitespace_controls_lengths_and_duplicate_ordinals_fail_closed() -> None:
    with pytest.raises(core.Dstc9P1TypedCoreError, match="empty"):
        core.KnowledgeSnippet(0, None, " ", "body")
    with pytest.raises(
        core.Dstc9P1TypedCoreError,
        match="forbidden control",
    ):
        core.KnowledgeSnippet(0, None, "title", "bad\x00body")
    with pytest.raises(core.Dstc9P1TypedCoreError, match="invalid"):
        core.KnowledgeSnippet(
            0,
            None,
            "t" * (core.MAX_TITLE_CHARACTERS + 1),
            "body",
        )
    with pytest.raises(core.Dstc9P1TypedCoreError, match="invalid"):
        core.KnowledgeSnippet(
            0,
            "e" * (core.MAX_ENTITY_NAME_CHARACTERS + 1),
            "title",
            "body",
        )
    with pytest.raises(core.Dstc9P1TypedCoreError, match="invalid"):
        core.KnowledgeSnippet(
            0,
            None,
            "title",
            "b" * (core.MAX_BODY_CHARACTERS + 1),
        )

    snippets = list(_snippets(5))
    snippets[-1] = core.KnowledgeSnippet(
        snippets[0].ordinal,
        None,
        "Duplicate",
        "Duplicate ordinal",
    )
    zeros = (0,) * len(snippets)
    with pytest.raises(
        core.Dstc9P1TypedCoreError,
        match="ordinals are duplicated",
    ):
        core.build_action_slate(
            _history(),
            snippets,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            0,
        )


def test_five_typed_combiners_and_e0_change_rankings_without_gold() -> None:
    slate = _slate(bucket=0)
    assert tuple(action.recipe_id for action in slate.actions) == (
        core.RECIPE_IDS
    )
    assert len(core.TYPED_RECIPE_IDS) == 5
    typed_top5 = {
        slate.action(recipe_id).top5_ordinals
        for recipe_id in core.TYPED_RECIPE_IDS
    }
    assert len(typed_top5) == len(core.TYPED_RECIPE_IDS)
    assert slate.action(core.E0_RECIPE_ID).top5_ordinals not in (
        typed_top5
    )
    for action in slate.actions:
        assert len(action.ranked_ordinals) == len(_snippets())
        assert set(action.ranked_ordinals) == set(range(len(_snippets())))
        assert len(action.top5_ordinals) == core.TOP_K

    audit = slate.audit_payload()
    assert audit["label_bearing_action_inputs"] is False
    assert audit["score_names"] == list(core.SCORE_NAMES)
    assert audit["public_snippet_fields"] == list(
        core.PUBLIC_SNIPPET_FIELDS
    )
    assert all(
        forbidden not in audit
        for forbidden in (
            "domain",
            "entity_id",
            "doc_id",
            "family",
            "qrel",
            "response",
            "target",
        )
    )
    claimed = audit.pop("self_sha256")
    assert claimed == core.stable_hash(audit)


def test_integer_rank_ties_break_by_ordinal_and_input_order_is_irrelevant() -> None:
    snippets = _snippets(10)
    tied = (17,) * len(snippets)
    first = core.build_action_slate(
        _history(),
        snippets,
        tied,
        tied,
        tied,
        tied,
        tied,
        tied,
        3,
    )
    second = core.build_action_slate(
        _history(),
        tuple(reversed(snippets)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        3,
    )
    assert first == second
    expected = tuple(range(len(snippets)))
    assert {
        action.ranked_ordinals for action in first.actions
    } == {expected}
    with pytest.raises(
        core.Dstc9P1TypedCoreError,
        match="bounded integers",
    ):
        core.build_action_slate(
            _history(),
            snippets,
            (True,) * len(snippets),
            tied,
            tied,
            tied,
            tied,
            tied,
            3,
        )


def test_e1_tie_and_negative_evidence_fall_back_to_e0() -> None:
    tie_rows = []
    for _index in range(core.MIN_BUCKET_SUPPORT):
        values = list(_utility_vector())
        values[core.RECIPE_IDS.index(core.R1_GLOBAL_CONTEXT)] = 800_000
        values[
            core.RECIPE_IDS.index(core.R2_LAST_TURN_ENTITY)
        ] = 800_000
        tie_rows.append(core.AFormExample(0, tuple(values)))
    tied = core.fit_e1(tie_rows)
    assert tied.rule(0).selected_recipe_id == core.E0_RECIPE_ID
    assert tied.rule(0).fallback_reason == "tie_to_e0"
    tied_decision = core.apply_e1(tied, _slate(bucket=0), stage="A_hold")
    assert tied_decision.fallback_to_e0 is True
    assert tied_decision.selected_recipe_id == core.E0_RECIPE_ID

    negative_rows = []
    selected_index = core.RECIPE_IDS.index(core.R3_TITLE_ANCHOR)
    for utility in (800_000, 800_000, 400_000):
        values = list(_utility_vector())
        values[selected_index] = utility
        negative_rows.append(core.AFormExample(2, tuple(values)))
    negative = core.fit_e1(negative_rows)
    evidence = negative.rule(2).evidence[
        core.TYPED_RECIPE_IDS.index(core.R3_TITLE_ANCHOR)
    ]
    assert evidence.minimum_delta < 0
    assert evidence.qualified is False
    assert negative.rule(2).selected_recipe_id == core.E0_RECIPE_ID
    assert negative.rule(2).fallback_reason == "no_qualified_recipe"

    under_supported = core.fit_e1(
        tuple(
            core.AFormExample(
                3,
                _utility_vector(
                    selected_recipe=core.R4_BODY_EVIDENCE
                ),
            )
            for _index in range(core.MIN_BUCKET_SUPPORT - 1)
        )
    )
    assert under_supported.rule(3).selected_recipe_id == (
        core.E0_RECIPE_ID
    )


def test_positive_e1_program_is_immutable_and_reused_on_hold_and_search() -> None:
    slate = _slate(bucket=1)
    examples = tuple(
        core.make_aform_example(
            slate,
            _utility_vector(
                selected_recipe=core.R2_LAST_TURN_ENTITY
            ),
        )
        for _index in range(core.MIN_BUCKET_SUPPORT)
    )
    program = core.fit_e1(examples)
    rule = program.rule(1)
    assert rule.selected_recipe_id == core.R2_LAST_TURN_ENTITY
    assert rule.fallback_reason == "selected"
    evidence = rule.evidence[
        core.TYPED_RECIPE_IDS.index(core.R2_LAST_TURN_ENTITY)
    ]
    assert evidence.qualified is True
    assert 0 < evidence.shrunken_mean_delta < 300_000

    frozen_payload = program.payload()
    frozen_sha = program.program_sha256
    hold_decisions = tuple(
        core.apply_e1(program, slate, stage="A_hold")
        for _index in range(2)
    )
    search_decision = core.apply_e1(
        program,
        slate,
        stage="M_search",
    )
    assert {
        decision.selected_recipe_id
        for decision in (*hold_decisions, search_decision)
    } == {core.R2_LAST_TURN_ENTITY}
    assert {
        decision.program_sha256
        for decision in (*hold_decisions, search_decision)
    } == {frozen_sha}
    assert all(
        decision.fallback_to_e0 is False
        for decision in (*hold_decisions, search_decision)
    )
    assert program.payload() == frozen_payload
    assert program.program_sha256 == frozen_sha
    with pytest.raises(FrozenInstanceError):
        program.training_item_count = 99  # type: ignore[misc]

    summary = core.summarize_e1_behavior(
        program,
        hold_decisions,
        stage="A_hold",
    )
    assert summary.program_sha256 == frozen_sha
    assert summary.item_count == 2
    assert summary.fallback_count == 0
    assert summary.bucket_recipe_counts == (
        (1, core.R2_LAST_TURN_ENTITY, 2),
    )
    summary_payload = summary.payload()
    assert "utility_vector" not in summary_payload
    claimed = summary_payload.pop("self_sha256")
    assert claimed == core.stable_hash(summary_payload)
