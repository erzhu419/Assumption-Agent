from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import hashlib
import inspect

import pytest

from assumption_agent.benchmarks import techqa_p1_typed_core_v1 as core


QUESTION_TITLE = "Fix ERR-X91 in Widget Server 8.5.5"
QUESTION_TEXT = (
    "Connection fails after upgrade; how can this be resolved?"
)


def _focused_documents(
    *,
    suffix: str = "",
) -> tuple[core.Document, ...]:
    suffix = f" {suffix}" if suffix else ""
    distractors = [
        core.Document(
            ordinal=index,
            title=(
                f"Connection fails upgrade resolved {index}{suffix}"
            ),
            text=(
                "Connection fails after upgrade how can this be resolved "
                f"connection fails upgrade{suffix}"
            ),
        )
        for index in range(5)
    ]
    typed = [
        core.Document(
            10,
            f"Widget Server ERR-X91{suffix}",
            "Use cleanup mode.",
        ),
        core.Document(
            11,
            f"Widget Server version 8.5.5{suffix}",
            "Migration notes.",
        ),
        core.Document(
            12,
            f"Fix Widget Server{suffix}",
            "ERR-X91 is corrected by patch 8.5.5.",
        ),
        core.Document(
            13,
            f"Widget Server administration{suffix}",
            "Header-focused answer for connection issue.",
        ),
        core.Document(
            14,
            f"Unrelated manual{suffix}",
            "ERR-X91 8.5.5 cleanup.",
        ),
        core.Document(
            15,
            f"Widget Server troubleshooting{suffix}",
            "Connection reset steps.",
        ),
    ]
    return tuple(distractors + typed)


def _distinction_documents() -> tuple[core.Document, ...]:
    rows = (
        ("widget", "after server migration trouble"),
        (
            "other fails err-x91",
            "guide other noise migration trouble mode after random mode "
            "random trouble after",
        ),
        ("noise guide migration", "reset 8.5.5 fix fails widget"),
        (
            "reset migration trouble connection admin",
            "admin random patch patch after other other server patch",
        ),
        (
            "migration after patch upgrade",
            "mode migration after trouble trouble cleanup reset random "
            "admin cleanup",
        ),
        (
            "mode connection",
            "connection random fails other admin mode mode guide reset "
            "fails 8.5.5 mode",
        ),
        ("widget mode fix fails server", "reset fix reset"),
        ("migration", "trouble 8.5.5 8.5.5 trouble upgrade"),
        (
            "err-x91",
            "after 8.5.5 fix widget err-x91 fix random after",
        ),
        (
            "connection other",
            "guide after fix other server fix fails noise admin widget fails",
        ),
        (
            "cleanup",
            "reset noise fix connection upgrade noise server other "
            "err-x91 guide",
        ),
        (
            "fix",
            "patch random mode patch cleanup fails trouble fails "
            "connection other",
        ),
        ("trouble", "guide connection widget server cleanup"),
        (
            "migration err-x91",
            "8.5.5 upgrade connection reset noise migration",
        ),
    )
    return tuple(
        core.Document(index, title, text)
        for index, (title, text) in enumerate(rows)
    )


def _utility(
    action: core.RecipeAction,
    relevant_ordinals: frozenset[int],
) -> Fraction:
    return Fraction(
        len(
            set(action.top5_document_ordinals)
            & relevant_ordinals
        ),
        len(relevant_ordinals),
    )


def _utilities(
    slate: core.ActionSlate,
    relevant_ordinals: frozenset[int],
) -> dict[str, Fraction]:
    return {
        recipe_id: _utility(
            slate.action(recipe_id),
            relevant_ordinals,
        )
        for recipe_id in core.RECIPE_IDS
    }


def test_strict_public_projection_and_api_have_no_label_entrypoint() -> None:
    assert tuple(field.name for field in fields(core.Document)) == (
        "ordinal",
        "title",
        "text",
    )
    assert tuple(inspect.signature(core.build_action_slate).parameters) == (
        "question_title",
        "question_text",
        "documents",
    )
    assert tuple(field.name for field in fields(core.AFormExample)) == (
        "features",
        "utilities",
    )
    public = {
        "ordinal": 7,
        "title": "Public title",
        "text": "Public body",
    }
    projected = core.document_from_public_fields(public)
    assert core.document_public_payload(projected) == public
    for forbidden in (
        "answer_span",
        "answerable",
        "cluster",
        "document_id",
        "family",
        "qrel",
        "source",
        "stage",
    ):
        with pytest.raises(
            core.TechqaP1TypedCoreError,
            match="exact public field set",
        ):
            core.document_from_public_fields(
                {**public, forbidden: "forbidden"}
            )


def test_raw_is_full_unchanged_query_and_identical_document_bytes() -> None:
    documents = _focused_documents()
    raw_query = QUESTION_TITLE + "\n" + QUESTION_TEXT
    assert (
        core.serialize_query_text(QUESTION_TITLE, QUESTION_TEXT)
        == raw_query
    )
    assert core.serialize_document_bytes(documents[0]) == (
        documents[0].title + "\n\n" + documents[0].text
    ).encode("utf-8")
    assert core.QUERY_SERIALIZATION_SEPARATOR == "\n"
    assert core.DOCUMENT_SERIALIZATION_SEPARATOR == "\n\n"
    assert core.MAX_QUERY_CHARACTERS == 4_000
    assert core.MAX_DOCUMENT_FIELD_CHARACTERS == 1_000_000
    with pytest.raises(
        core.TechqaP1TypedCoreError,
        match="official-worker bound",
    ):
        core.serialize_query_text("q" * 4_000, "x")
    slate = core.build_action_slate(
        QUESTION_TITLE,
        QUESTION_TEXT,
        tuple(reversed(documents)),
    )
    ordered = tuple(sorted(documents, key=lambda document: document.ordinal))
    raw_scores = core.bm25_scores(
        raw_query,
        [core.serialize_document_text(document) for document in ordered],
    )
    expected = tuple(
        ordered[index].ordinal
        for index in sorted(
            range(len(ordered)),
            key=lambda index: (-raw_scores[index], ordered[index].ordinal),
        )[: core.TOP_K]
    )
    assert slate.action(core.R0_RAW_BM25).top5_document_ordinals == expected
    for action in slate.actions:
        assert len(action.top5_document_ordinals) == core.TOP_K
        assert len(set(action.top5_document_ordinals)) == core.TOP_K
        assert action.raw_top5_document_ordinals == expected
    assert slate.raw_query_bytes_sha256 == hashlib.sha256(
        raw_query.encode("utf-8")
    ).hexdigest()
    assert tuple(
        row.serialized_sha256 for row in slate.document_structures
    ) == tuple(
        hashlib.sha256(core.serialize_document_bytes(document)).hexdigest()
        for document in ordered
    )


def test_six_recipes_are_frozen_deterministic_and_behaviorally_distinct() -> None:
    documents = _distinction_documents()
    first = core.build_action_slate(
        QUESTION_TITLE,
        "connection fails after upgrade cleanup mode reset",
        documents,
    )
    second = core.build_action_slate(
        QUESTION_TITLE,
        "connection fails after upgrade cleanup mode reset",
        tuple(reversed(documents)),
    )
    assert first == second
    assert len(core.RECIPE_IDS) == 6
    assert tuple(action.recipe_id for action in first.actions) == (
        core.R0_RAW_BM25,
        core.R1_TITLE_FOCUSED,
        core.R2_LITERAL_SIGNATURE_ANCHOR,
        core.R3_FIELD_AWARE_COVERAGE,
        core.R4_MULTI_SEED_MARGINAL,
        core.R5_TYPED_CASCADE,
    )
    assert len(
        {
            action.top5_document_ordinals
            for action in first.actions
        }
    ) == len(core.RECIPE_IDS)
    audit = first.audit_payload()
    assert audit["public_document_fields"] == [
        "ordinal",
        "title",
        "text",
    ]
    assert audit["feature_names"] == list(core.FEATURE_NAMES)
    claimed = audit.pop("self_sha256")
    assert claimed == core.stable_hash(audit)


def test_exact_error_version_and_header_anchors_have_causal_effect() -> None:
    documents = _focused_documents() + (
        core.Document(
            20,
            "Widget Server ERR-X92",
            "Use cleanup mode for version 8.5.6.",
        ),
    )
    slate = core.build_action_slate(
        QUESTION_TITLE,
        QUESTION_TEXT,
        documents,
    )
    rows = {
        row.document_ordinal: row
        for row in slate.document_structures
    }
    assert rows[10].error_hits == 1
    assert rows[20].error_hits == 0
    assert rows[11].version_hits == 1
    assert rows[20].version_hits == 0
    assert rows[10].title_hits > 0
    assert rows[14].title_hits == 0
    assert 10 not in slate.action(
        core.R0_RAW_BM25
    ).top5_document_ordinals
    assert 10 in slate.action(
        core.R1_TITLE_FOCUSED
    ).top5_document_ordinals
    literal_top5 = slate.action(
        core.R2_LITERAL_SIGNATURE_ANCHOR
    ).top5_document_ordinals
    assert {10, 11, 12, 14} <= set(literal_top5)
    assert 20 not in literal_top5


def test_e1_uses_exact_aform_utilities_and_improves_held_item() -> None:
    relevant = frozenset({0, 1, 10, 11, 12})
    training_slates = tuple(
        core.build_action_slate(
            QUESTION_TITLE,
            QUESTION_TEXT,
            _focused_documents(suffix=f"train{index}"),
        )
        for index in range(core.MIN_SIGNATURE_SUPPORT)
    )
    examples = tuple(
        core.make_aform_example(
            slate,
            _utilities(slate, relevant),
        )
        for slate in training_slates
    )
    model = core.fit_e1(examples)
    assert model.training_stage == "A_form"
    assert model.training_item_count == core.MIN_SIGNATURE_SUPPORT
    title_rules = [
        rule
        for rule in model.rules
        if rule.recipe_id == core.R1_TITLE_FOCUSED
    ]
    assert len(title_rules) == 1
    assert title_rules[0].qualified
    assert title_rules[0].minimum_delta == Fraction(1, 5)

    held = core.build_action_slate(
        QUESTION_TITLE,
        QUESTION_TEXT,
        _focused_documents(suffix="held"),
    )
    e0 = core.apply_e0(held, stage="A_hold")
    decisions = tuple(
        core.apply_e1(model, held, stage=stage)
        for stage in core.POLICY_STAGES
    )
    assert {
        decision.selected_recipe_id for decision in decisions
    } == {core.R1_TITLE_FOCUSED}
    assert {
        decision.top5_document_ordinals for decision in decisions
    } == {
        held.action(
            core.R1_TITLE_FOCUSED
        ).top5_document_ordinals
    }
    assert all(not decision.fallback_to_e0 for decision in decisions)
    assert _utility(
        held.action(decisions[0].selected_recipe_id),
        relevant,
    ) > _utility(held.action(e0.selected_recipe_id), relevant)
    model_payload = model.payload()
    assert model_payload["training_stage"] == "A_form"
    assert "family" not in model_payload["signature_feature_names"]
    assert "stage" not in model_payload["signature_feature_names"]


def test_under_support_conflict_and_unseen_signature_fall_back_to_e0() -> None:
    slate = core.build_action_slate(
        QUESTION_TITLE,
        QUESTION_TEXT,
        _focused_documents(),
    )
    relevant = frozenset({0, 1, 10, 11, 12})
    favorable = _utilities(slate, relevant)
    contradicted = dict(favorable)
    contradicted[core.R1_TITLE_FOCUSED] = Fraction(0, 1)

    under_supported = core.fit_e1(
        tuple(
            core.make_aform_example(slate, favorable)
            for _ in range(core.MIN_SIGNATURE_SUPPORT - 1)
        )
    )
    under = core.apply_e1(
        under_supported,
        slate,
        stage="F_search",
    )
    assert under.fallback_to_e0
    assert under.selected_recipe_id == core.E0_RECIPE_ID

    conflicted = core.fit_e1(
        (
            core.make_aform_example(slate, favorable),
            core.make_aform_example(slate, favorable),
            core.make_aform_example(slate, contradicted),
        )
    )
    assert not any(rule.qualified for rule in conflicted.rules)
    conflict = core.apply_e1(
        conflicted,
        slate,
        stage="A_hold",
    )
    assert conflict.fallback_to_e0
    assert conflict.selected_recipe_id == core.E0_RECIPE_ID

    qualified = core.fit_e1(
        tuple(
            core.make_aform_example(slate, favorable)
            for _ in range(core.MIN_SIGNATURE_SUPPORT)
        )
    )
    unseen_slate = core.build_action_slate(
        "General network setup guide",
        "Where are the default preferences stored?",
        tuple(
            core.Document(
                index,
                f"Manual section {index}",
                f"General preferences storage notes {index}",
            )
            for index in range(8)
        ),
    )
    unseen = core.apply_e1(
        qualified,
        unseen_slate,
        stage="M_search",
    )
    assert unseen.fallback_to_e0
    assert unseen.selected_recipe_id == core.E0_RECIPE_ID
