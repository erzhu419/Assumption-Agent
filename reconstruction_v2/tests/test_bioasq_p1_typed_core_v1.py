from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import pytest

from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core


def _passages() -> tuple[core.Passage, ...]:
    return tuple(
        core.Passage(index, f"Biomedical evidence passage {index}.")
        for index in range(core.CORPUS_SIZE)
    )


def _group_scores(group_start: int) -> tuple[int, ...]:
    group = list(range(group_start, group_start + core.TOP_K))
    remainder = [
        index for index in range(core.CORPUS_SIZE)
        if index not in group
    ]
    order = group + remainder
    scores = [0] * core.CORPUS_SIZE
    for rank, index in enumerate(order):
        scores[index] = core.CORPUS_SIZE - rank
    return tuple(scores)


def _score_bundle() -> dict[str, tuple[int, ...]]:
    return {
        "raw_ce_scores": _group_scores(0),
        "focus_ce_scores": _group_scores(5),
        "dense_base_scores": _group_scores(10),
        "dense_support_scores": _group_scores(15),
        "dense_contrast_scores": _group_scores(20),
        "dense_coverage_scores": _group_scores(25),
    }


def _slate(query: str = "Are statins effective?") -> core.ActionSlate:
    return core.build_action_slate(
        query,
        _passages(),
        **_score_bundle(),
    )


def _utility_vector(
    *,
    selected_recipe: str | None = None,
    selected_utility: int = 500_000,
    e0_utility: int = 300_000,
    default_utility: int = 200_000,
) -> tuple[int, ...]:
    values = [default_utility] * len(core.RECIPE_IDS)
    values[core.RECIPE_IDS.index(core.E0_RECIPE_ID)] = e0_utility
    if selected_recipe is not None:
        values[core.RECIPE_IDS.index(selected_recipe)] = selected_utility
    return tuple(values)


def test_frozen_registry_and_public_boundary_match_preregistered_design() -> None:
    assert core.STUDY_ID == (
        "BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1"
    )
    assert core.CORPUS_SIZE == 2_900
    assert core.TOP_K == 5
    assert core.SCALE == 300_000
    assert core.MAX_UTILITY == 600_000
    assert core.BUCKET_NAMES == ("claim", "entity", "list", "aspect")
    assert core.POLICY_STAGES == ("F_search", "A_hold", "M_search")
    assert core.SCORE_NAMES == (
        "raw_ce",
        "focus_ce",
        "dense_base",
        "dense_support",
        "dense_contrast",
        "dense_coverage",
    )
    assert core.TYPED_RECIPE_IDS == (
        "claim_polarity_balanced_evidence_set",
        "entity_focused_evidence_set",
        "list_redundancy_controlled_evidence_set",
        "multi_aspect_coverage_evidence_set",
    )
    assert core.E0_RECIPE_ID == (
        "global_raw_dense_reciprocal_rank_fusion"
    )
    assert core.MIN_BUCKET_SUPPORT == 6
    assert core.MIN_NET_POSITIVE_MARGIN_COUNT == 2
    assert core.SHRINKAGE_PSEUDOCOUNT == 4

    assert tuple(field.name for field in fields(core.Passage)) == (
        "ordinal",
        "text",
    )
    assert tuple(
        inspect.signature(core.build_action_slate).parameters
    ) == (
        "query_text",
        "passages",
        "raw_ce_scores",
        "focus_ce_scores",
        "dense_base_scores",
        "dense_support_scores",
        "dense_contrast_scores",
        "dense_coverage_scores",
    )
    assert tuple(inspect.signature(core.fit_e1).parameters) == (
        "examples",
    )
    assert tuple(field.name for field in fields(core.AFormExample)) == (
        "predicted_bucket",
        "utility_vector",
    )

    public = {"ordinal": 7, "text": "  NF-κB\u00a0evidence. "}
    assert core.passage_public_payload(
        core.passage_from_public_fields(public)
    ) == {"ordinal": 7, "text": "NF-κB evidence."}
    for forbidden in (
        "answer",
        "document_id",
        "family",
        "ideal_answer",
        "qrel",
        "question_type",
        "source_identifier",
        "split",
        "target",
    ):
        with pytest.raises(
            core.BioasqP1TypedCoreError,
            match="exact public field set",
        ):
            core.passage_from_public_fields(
                {**public, forbidden: "forbidden"}
            )


def test_query_only_predictor_and_all_scorer_serializers_are_deterministic() -> None:
    examples = (
        ("Are statins effective?", core.B0_CLAIM),
        ("Which gene encodes the p53 protein?", core.B1_ENTITY),
        ("List drugs used to treat pulmonary hypertension.", core.B2_LIST),
        ("Explain how autophagy affects cell survival.", core.B3_ASPECT),
    )
    for query, expected_bucket in examples:
        first = core.predict_question_structure(query)
        second = core.predict_question_structure(f"  {query}\n")
        assert first == second
        assert first.predicted_bucket == expected_bucket
        payload = first.payload()
        claimed = payload.pop("self_sha256")
        assert claimed == core.stable_hash(payload)

        bundle = core.serialize_score_queries(query)
        assert bundle.predicted_bucket == expected_bucket
        assert tuple(
            getattr(bundle, name) for name in core.SCORE_NAMES
        ) == tuple(
            core.serialize_query_for_score(query, name)
            for name in core.SCORE_NAMES
        )
        assert bundle.raw_ce == core.validate_query_text(query)
        assert bundle.dense_base == core.validate_query_text(query)
        assert "family" not in core.canonical_bytes(
            bundle.payload()
        ).decode("ascii").casefold()
        assert bundle.bundle_sha256 == core.stable_hash(
            bundle.body_payload()
        )

    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="forbidden control",
    ):
        core.validate_query_text("bad\u200bquery")
    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="no lexical token",
    ):
        core.validate_query_text(" ?! ")
    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="score name",
    ):
        core.serialize_query_for_score("A query", "qrel")


def test_four_typed_evidence_sets_and_global_rrf_e0_change_top5() -> None:
    slate = _slate()
    assert slate.predicted_bucket == core.B0_CLAIM
    assert tuple(action.recipe_id for action in slate.actions) == (
        core.RECIPE_IDS
    )
    typed_top5 = tuple(
        slate.action(recipe_id).top5_ordinals
        for recipe_id in core.TYPED_RECIPE_IDS
    )
    assert len(set(typed_top5)) == len(core.TYPED_RECIPE_IDS)
    assert slate.action(core.E0_RECIPE_ID).top5_ordinals not in (
        typed_top5
    )
    assert typed_top5[0] == (15, 20, 0, 5, 25)
    assert typed_top5[2] == (25, 10, 5, 15, 0)
    for action in slate.actions:
        assert len(action.ranked_ordinals) == core.CORPUS_SIZE
        assert set(action.ranked_ordinals) == set(
            range(core.CORPUS_SIZE)
        )
        assert len(action.top5_trace) == core.TOP_K

    e0 = slate.action(core.E0_RECIPE_ID)
    assert all(trace.startswith("rrf:") for trace in e0.top5_trace)
    audit = slate.audit_payload()
    assert audit["label_bearing_action_inputs"] is False
    assert audit["score_names"] == list(core.SCORE_NAMES)
    assert audit["public_passage_fields"] == ["ordinal", "text"]
    serialized = core.canonical_bytes(audit).decode("ascii").casefold()
    for forbidden in (
        "answer",
        "document_id",
        "family",
        "ideal_answer",
        "qrel",
        "question_type",
        "source_identifier",
        "target",
    ):
        assert forbidden not in serialized
    claimed = audit.pop("self_sha256")
    assert claimed == core.stable_hash(audit)


def test_list_recipe_removes_near_duplicate_from_fixed_prefix_deterministically() -> None:
    passages = list(_passages())
    duplicate = (
        "BRCA1 mutation disrupts homologous recombination DNA repair "
        "and increases genomic instability."
    )
    passages[25] = core.Passage(25, duplicate)
    passages[10] = core.Passage(
        10,
        " BRCA1\u00a0mutation disrupts homologous recombination DNA "
        "repair and increases genomic instability. ",
    )
    bundle = _score_bundle()
    first = core.build_action_slate(
        "List mechanisms that maintain genomic stability.",
        passages,
        **bundle,
    )
    action = first.action(core.R3_LIST_DIVERSITY)
    assert action.top5_ordinals[0] == 25
    assert 10 not in action.top5_ordinals
    # The unselected near duplicate is appended in the unchanged multiview
    # base order immediately after the five greedily selected passages.
    assert action.ranked_ordinals[core.TOP_K] == 10
    assert all(
        trace.startswith("greedy:")
        for trace in action.top5_trace
    )
    assert len(action.ranked_ordinals) == core.CORPUS_SIZE
    assert set(action.ranked_ordinals) == set(range(core.CORPUS_SIZE))

    reversed_bundle = {
        name: tuple(reversed(values))
        for name, values in bundle.items()
    }
    second = core.build_action_slate(
        "List mechanisms that maintain genomic stability.",
        tuple(reversed(passages)),
        **reversed_bundle,
    )
    assert second.action(core.R3_LIST_DIVERSITY) == action
    audit = first.audit_payload()
    assert audit["list_diversity"] == {
        "candidate_prefix": 128,
        "novelty_weight": 1,
        "passage_similarity": "token_set_jaccard",
        "relevance_weight": 1,
    }


def test_exact_corpus_integer_scores_and_ordinal_tie_break_fail_closed() -> None:
    passages = _passages()
    tied = (17,) * core.CORPUS_SIZE
    first = core.build_action_slate(
        "Which protein is involved?",
        passages,
        tied,
        tied,
        tied,
        tied,
        tied,
        tied,
    )
    second = core.build_action_slate(
        "Which protein is involved?",
        tuple(reversed(passages)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
        tuple(reversed(tied)),
    )
    assert first == second
    expected = tuple(range(core.CORPUS_SIZE))
    assert {
        action.ranked_ordinals for action in first.actions
    } == {expected}

    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="exactly 2900",
    ):
        core.build_action_slate(
            "Question",
            passages[:-1],
            tied[:-1],
            tied[:-1],
            tied[:-1],
            tied[:-1],
            tied[:-1],
            tied[:-1],
        )
    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="bounded integers",
    ):
        core.build_action_slate(
            "Question",
            passages,
            (True,) * core.CORPUS_SIZE,
            tied,
            tied,
            tied,
            tied,
            tied,
        )
    malformed = list(passages)
    malformed[-1] = core.Passage(0, "duplicate ordinal")
    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="frozen corpus universe",
    ):
        core.build_action_slate(
            "Question",
            malformed,
            tied,
            tied,
            tied,
            tied,
            tied,
            tied,
        )


def test_e1_uses_only_registered_support_margin_total_and_shrinkage_rules() -> None:
    recipe = core.R1_CLAIM_BALANCE
    recipe_index = core.RECIPE_IDS.index(recipe)
    e0_index = core.RECIPE_IDS.index(core.E0_RECIPE_ID)
    rows = []
    # Four positive and two negative rows: support=6, net margin=2,
    # total delta=600k.  Negative evidence is allowed by the preregistered
    # rule; there is intentionally no minimum-delta or 3-of-4 gate.
    for delta in (200_000, 200_000, 200_000, 200_000, -100_000, -100_000):
        values = list(_utility_vector())
        values[e0_index] = 300_000
        values[recipe_index] = 300_000 + delta
        rows.append(core.AFormExample(core.B0_CLAIM, tuple(values)))
    program = core.fit_e1(rows)
    rule = program.rule(core.B0_CLAIM)
    evidence = rule.evidence[
        core.TYPED_RECIPE_IDS.index(recipe)
    ]
    assert evidence.support_count == 6
    assert evidence.positive_count == 4
    assert evidence.negative_count == 2
    assert evidence.total_delta == 600_000
    assert evidence.qualified is True
    assert rule.selected_recipe_id == recipe

    under_supported = core.fit_e1(rows[:-1])
    assert under_supported.rule(core.B0_CLAIM).selected_recipe_id == (
        core.E0_RECIPE_ID
    )

    insufficient_margin = []
    for delta in (200_000, 200_000, 200_000, -100_000, -100_000, 0):
        values = list(_utility_vector())
        values[recipe_index] = 300_000 + delta
        insufficient_margin.append(
            core.AFormExample(core.B0_CLAIM, tuple(values))
        )
    margin_program = core.fit_e1(insufficient_margin)
    margin_evidence = margin_program.rule(core.B0_CLAIM).evidence[
        core.TYPED_RECIPE_IDS.index(recipe)
    ]
    assert margin_evidence.total_delta > 0
    assert margin_evidence.qualified is False

    zero_total = []
    for delta in (100_000, 100_000, 100_000, 100_000, -200_000, -200_000):
        values = list(_utility_vector())
        values[recipe_index] = 300_000 + delta
        zero_total.append(
            core.AFormExample(core.B0_CLAIM, tuple(values))
        )
    total_program = core.fit_e1(zero_total)
    total_evidence = total_program.rule(core.B0_CLAIM).evidence[
        core.TYPED_RECIPE_IDS.index(recipe)
    ]
    assert total_evidence.positive_count - total_evidence.negative_count == 2
    assert total_evidence.total_delta == 0
    assert total_evidence.qualified is False


def test_e1_unique_max_tie_fallback_and_unchanged_hold_search_application() -> None:
    tie_rows = []
    for _index in range(core.MIN_BUCKET_SUPPORT):
        values = list(_utility_vector())
        values[
            core.RECIPE_IDS.index(core.R1_CLAIM_BALANCE)
        ] = 500_000
        values[
            core.RECIPE_IDS.index(core.R2_ENTITY_FOCUS)
        ] = 500_000
        tie_rows.append(core.AFormExample(core.B0_CLAIM, tuple(values)))
    tied = core.fit_e1(tie_rows)
    assert tied.rule(core.B0_CLAIM).selected_recipe_id == (
        core.E0_RECIPE_ID
    )
    assert tied.rule(core.B0_CLAIM).fallback_reason == "tie_to_e0"

    slate = _slate()
    examples = tuple(
        core.make_aform_example(
            slate,
            _utility_vector(
                selected_recipe=core.R1_CLAIM_BALANCE
            ),
        )
        for _index in range(core.MIN_BUCKET_SUPPORT)
    )
    program = core.fit_e1(examples)
    assert program.rule(core.B0_CLAIM).selected_recipe_id == (
        core.R1_CLAIM_BALANCE
    )
    frozen_payload = program.payload()
    hold = core.apply_e1(program, slate, stage="A_hold")
    label_free_search = core.apply_e1(
        program,
        slate,
        stage="F_search",
    )
    search = core.apply_e1(program, slate, stage="M_search")
    assert hold.selected_recipe_id == core.R1_CLAIM_BALANCE
    assert label_free_search.selected_recipe_id == (
        core.R1_CLAIM_BALANCE
    )
    assert search.selected_recipe_id == core.R1_CLAIM_BALANCE
    assert hold.program_sha256 == search.program_sha256
    assert hold.top5_ordinals == search.top5_ordinals
    assert core.apply_e0(
        slate,
        stage="A_hold",
    ).selected_recipe_id == core.E0_RECIPE_ID
    assert program.payload() == frozen_payload
    with pytest.raises(FrozenInstanceError):
        program.training_item_count = 0  # type: ignore[misc]

    summary = core.summarize_e1_behavior(
        program,
        (hold, hold),
        stage="A_hold",
    )
    assert summary.item_count == 2
    assert summary.fallback_count == 0
    assert summary.bucket_recipe_counts == (
        (core.B0_CLAIM, core.R1_CLAIM_BALANCE, 2),
    )
    summary_payload = summary.payload()
    assert "utility_vector" not in summary_payload
    claimed = summary_payload.pop("self_sha256")
    assert claimed == core.stable_hash(summary_payload)


def test_dataclass_and_canonical_hash_contracts_reject_tampering() -> None:
    structure = core.predict_question_structure("Is aspirin effective?")
    with pytest.raises(core.BioasqP1TypedCoreError, match="bucket drifted"):
        core.QuestionStructure(
            predicted_bucket=core.B1_ENTITY,
            bucket_name="entity",
            yes_no_cue=True,
            list_cue=False,
            summary_cue=False,
            wh_head=None,
            query_sha256=structure.query_sha256,
        )
    with pytest.raises(core.BioasqP1TypedCoreError, match="malformed"):
        core.EvidenceAction(
            recipe_id=core.R1_CLAIM_BALANCE,
            ranked_ordinals=(0,) * core.CORPUS_SIZE,
            top5_trace=("trace",) * core.TOP_K,
            behavior_digest="0" * 64,
        )
    with pytest.raises(
        core.BioasqP1TypedCoreError,
        match="canonical JSON",
    ):
        core.stable_hash({"bad": float("nan")})
    with pytest.raises(FrozenInstanceError):
        structure.predicted_bucket = core.B1_ENTITY  # type: ignore[misc]
