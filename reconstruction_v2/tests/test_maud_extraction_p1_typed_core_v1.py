from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math

import pytest

from assumption_agent.benchmarks.maud_extraction_p1_typed_core_v1 import (
    AFormSlate,
    ActionFeatures,
    CharacterInterval,
    ClusterItem,
    CONDITION_OBLIGATION,
    ContractCluster,
    DEFINITION_REFERENCE,
    EDGE_FAMILIES,
    E1_FEATURE_COUNT,
    EXCEPTION_REMEDY,
    FAMILY_CONDITION_OBLIGATION,
    FAMILY_DEFINITION_REFERENCE,
    FAMILY_PROTECTION_EXCEPTION_REMEDY,
    FEATURE_ORDER,
    GoldAnswer,
    INTEGER_SCALE,
    MaudExtractionP1CoreError,
    Passage,
    QUERY_FAMILIES,
    RECIPE_IDS,
    RecipeSlate,
    SECTION_XREF,
    TypedEdge,
    bm25_scores,
    build_coordinate_table,
    build_coordinate_table_from_quantized,
    build_passages,
    build_recipe_slate,
    build_typed_edges,
    compare_contract_clusters,
    e0_score,
    exact_contract_sign_flip,
    expanded_family_features,
    fit_e1_ridge,
    materialize_recipe_actions,
    quantize_half_even,
    score_evidence_coverage,
    select_e0,
    select_e1,
    serialized_passage_corpus,
    validate_gold_intervals,
)


def _manual_corpus() -> tuple[str, tuple[Passage, ...]]:
    parts = (
        'Section 1. "Defined Term" means the transaction value.\n',
        "Section 2. The Defined Term shall be paid subject to Section 4.\n",
        "Section 3. Except as provided in Section 2, breach gives a remedy.\n",
        "Section 4. Closing mechanics and covenants apply.\n",
        "Section 5. Neutral evidence about the consideration.\n",
        "Section 6. Remote definition evidence mentions Defined Term.\n",
        "Section 7. Remote condition evidence applies at closing.\n",
        "Section 8. Remote remedy evidence supports enforcement.\n",
    )
    context = "".join(parts)
    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    passages: list[Passage] = []
    start = 0
    for ordinal, text in enumerate(parts):
        end = start + len(text)
        passages.append(
            Passage(
                ordinal=ordinal,
                context_sha256=context_sha256,
                start=start,
                end=end,
                text=text,
                exact_substring_sha256=hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            )
        )
        start = end
    return context, tuple(passages)


def _quantized_coordinates(passages: tuple[Passage, ...]):
    width = len(passages)
    ce = (1_000_000, 900_000, 800_000, 700_000, 600_000, 0, 0, 0)
    minilm = (950_000, 850_000, 750_000, 650_000, 550_000, 0, 0, 0)
    pairwise = tuple(
        tuple(
            INTEGER_SCALE if left == right else 100_000
            for right in range(width)
        )
        for left in range(width)
    )
    return build_coordinate_table_from_quantized(
        query="transaction consideration closing enforcement",
        passages=passages,
        cross_encoder_sigmoid=ce,
        minilm_unit_interval=minilm,
        pairwise_minilm_unit_interval=pairwise,
    )


def _swap_edges() -> tuple[TypedEdge, ...]:
    return (
        TypedEdge(0, 6, SECTION_XREF),
        TypedEdge(0, 5, DEFINITION_REFERENCE),
        TypedEdge(0, 7, CONDITION_OBLIGATION),
        TypedEdge(0, 6, EXCEPTION_REMEDY),
    )


def _base_slate() -> RecipeSlate:
    _, passages = _manual_corpus()
    return build_recipe_slate(
        query="transaction consideration closing enforcement",
        passages=passages,
        coordinates=_quantized_coordinates(passages),
        edges=_swap_edges(),
    )


def _feature_slate(item_offset: int) -> RecipeSlate:
    actions = _base_slate().actions
    features = []
    for recipe_index in range(len(RECIPE_IDS)):
        signal = 100_000 + item_offset * 2_000 + recipe_index * 40_000
        features.append(
            ActionFeatures(
                (
                    signal,
                    signal - 10_000,
                    signal - 20_000,
                    signal - 30_000,
                    200_000 + recipe_index * 10_000,
                    300_000,
                    400_000 + recipe_index * 5_000,
                    recipe_index * 5_000,
                    100_000 + recipe_index * 2_000,
                    200_000,
                    50_000 + item_offset * 1_000,
                    0,
                )
            )
        )
    return RecipeSlate(actions=actions, features=tuple(features))


def test_half_even_quantization_is_signed_and_deterministic() -> None:
    assert quantize_half_even(0.0000005) == 0
    assert quantize_half_even(0.0000015) == 2
    assert quantize_half_even(-0.0000005) == 0
    assert quantize_half_even(-0.0000015) == -2
    with pytest.raises(MaudExtractionP1CoreError, match="finite real"):
        quantize_half_even(float("nan"))
    with pytest.raises(MaudExtractionP1CoreError, match="finite real"):
        quantize_half_even(True)


def test_passage_windows_are_exact_bounded_overlapping_and_serialized() -> None:
    short = "α beta"
    one = build_passages(short)
    assert len(one) == 1
    assert one[0].text == short
    assert json.loads(serialized_passage_corpus(one)[0]) == {
        "text": short,
        "title": "MAUD passage 000000",
    }

    context = " ".join(f"token{index:04d}" for index in range(900))
    passages = build_passages(context)
    assert len(passages) > 5
    assert tuple(row.ordinal for row in passages) == tuple(
        range(len(passages))
    )
    assert all(
        row.text == context[row.start : row.end]
        and 0 < row.end - row.start <= 1_400
        for row in passages
    )
    assert all(
        right.start <= left.end - 240
        for left, right in zip(passages, passages[1:])
    )
    assert tuple(
        json.loads(raw)["title"] for raw in serialized_passage_corpus(passages)
    ) == tuple(f"MAUD passage {index:06d}" for index in range(len(passages)))


def test_bm25_and_both_coordinate_entrypoints_are_local_and_strict() -> None:
    documents = (
        b'{"text":"alpha alpha alpha","title":"first"}\n',
        b'{"text":"alpha","title":"second"}\n',
        b'{"text":"beta","title":"third"}\n',
    )
    scores = bm25_scores("alpha", documents)
    assert scores[0] > scores[2] == 0.0
    assert scores[1] > scores[2]

    _, passages = _manual_corpus()
    quantized = _quantized_coordinates(passages)
    assert quantized.passage_count == len(passages)
    assert quantized.bm25
    assert quantized.fused[0] == (
        5 * quantized.cross_encoder[0]
        + 3 * quantized.minilm[0]
        + 2 * quantized.bm25[0]
    ) // 10

    raw = build_coordinate_table(
        query="transaction",
        passages=passages,
        cross_encoder_logits=(8, 7, 6, 5, 4, -4, -5, -6),
        minilm_cosines=(1, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4),
        pairwise_minilm_cosines=tuple(
            tuple(1.0 if left == right else 0.0 for right in range(8))
            for left in range(8)
        ),
    )
    assert all(0 <= value <= INTEGER_SCALE for value in raw.fused)
    assert raw.pairwise_minilm[0][1] == 500_000

    with pytest.raises(MaudExtractionP1CoreError, match="exact integer"):
        build_coordinate_table_from_quantized(
            query="transaction",
            passages=passages,
            cross_encoder_sigmoid=(
                True,
                900_000,
                800_000,
                700_000,
                600_000,
                0,
                0,
                0,
            ),
            minilm_unit_interval=quantized.minilm,
            pairwise_minilm_unit_interval=quantized.pairwise_minilm,
        )
    asymmetric = [list(row) for row in quantized.pairwise_minilm]
    asymmetric[0][1] += 1
    with pytest.raises(MaudExtractionP1CoreError, match="symmetric"):
        build_coordinate_table_from_quantized(
            query="transaction",
            passages=passages,
            cross_encoder_sigmoid=quantized.cross_encoder,
            minilm_unit_interval=quantized.minilm,
            pairwise_minilm_unit_interval=asymmetric,
        )


def test_typed_parser_emits_only_the_four_frozen_edge_families() -> None:
    _, passages = _manual_corpus()
    edges = build_typed_edges(passages)
    triples = {
        (row.source_ordinal, row.target_ordinal, row.edge_family)
        for row in edges
    }
    assert (0, 1, DEFINITION_REFERENCE) in triples
    assert (0, 5, DEFINITION_REFERENCE) in triples
    assert (1, 2, CONDITION_OBLIGATION) in triples
    assert (1, 3, CONDITION_OBLIGATION) in triples
    assert (2, 1, EXCEPTION_REMEDY) in triples
    assert (1, 3, SECTION_XREF) in triples
    assert {row.edge_family for row in edges} == set(EDGE_FAMILIES)
    assert edges == tuple(
        sorted(
            edges,
            key=lambda row: (
                EDGE_FAMILIES.index(row.edge_family),
                row.source_ordinal,
                row.target_ordinal,
            ),
        )
    )


def test_nine_recipes_are_fixed_deterministic_typed_swaps() -> None:
    _, passages = _manual_corpus()
    coordinates = _quantized_coordinates(passages)
    edges = _swap_edges()
    actions = materialize_recipe_actions(
        passages=passages,
        coordinates=coordinates,
        edges=edges,
    )
    reversed_actions = materialize_recipe_actions(
        passages=passages,
        coordinates=coordinates,
        edges=tuple(reversed(edges)),
    )

    assert tuple(row.recipe_id for row in actions) == RECIPE_IDS
    assert actions == reversed_actions
    assert all(len(row.passage_ordinals) == 5 for row in actions)
    assert all(set(row.passage_ordinals) != set(range(5)) for row in actions)
    assert tuple(len(row.accepted_edges) for row in actions) == (
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
    )
    assert all(
        row.accepted_edges
        and all(edge.edge_family in EDGE_FAMILIES for edge in row.accepted_edges)
        for row in actions
    )
    assert len({row.behavior_sha256 for row in actions}) == len(RECIPE_IDS)


def test_action_features_e0_and_tie_break_are_integer_exact() -> None:
    slate = _base_slate()
    assert len(slate.features) == len(RECIPE_IDS)
    assert all(len(row.values) == len(FEATURE_ORDER) for row in slate.features)
    assert all(
        0 <= value <= INTEGER_SCALE
        for row in slate.features
        for value in row.values
    )
    # R0 closes the accepted SECTION_XREF relation over two of five passages.
    r0 = slate.features[0].as_mapping()
    assert r0["section_cross_reference_closure"] == 400_000

    fixed = ActionFeatures(
        (
            800_000,
            700_000,
            600_000,
            500_000,
            400_000,
            300_000,
            200_000,
            100_000,
            300_000,
            300_000,
            300_000,
            300_000,
        )
    )
    assert e0_score(fixed) == (
        Fraction(11, 20) * 800_000
        + Fraction(1, 5) * 600_000
        + Fraction(1, 10) * 400_000
        + Fraction(1, 10) * 300_000
        + Fraction(1, 20) * 200_000
        - Fraction(1, 20) * 100_000
    )
    tied = RecipeSlate(
        actions=slate.actions,
        features=tuple(fixed for _ in RECIPE_IDS),
    )
    assert select_e0(tied).registry_ordinal == 0


def test_e1_is_one_no_intercept_aform_ridge_with_48_features() -> None:
    aform = tuple(
        AFormSlate(
            family=family,
            slate=_feature_slate(index),
            recipe_utilities=tuple(
                100_000 + recipe * 75_000 for recipe in range(len(RECIPE_IDS))
            ),
        )
        for index, family in enumerate(QUERY_FAMILIES)
    )
    model = fit_e1_ridge(aform)
    same_model = fit_e1_ridge(aform)

    assert model == same_model
    assert model.training_row_count == len(aform) * len(RECIPE_IDS)
    assert len(model.means) == E1_FEATURE_COUNT == 48
    assert len(model.weights) == E1_FEATURE_COUNT
    assert any(model.zero_variance_columns)
    assert all(math.isfinite(value) for value in model.weights)
    expanded = expanded_family_features(
        aform[0].slate.features[0], FAMILY_DEFINITION_REFERENCE
    )
    assert len(expanded) == 48
    assert expanded[12:24] == expanded[:12]
    assert expanded[24:] == (0.0,) * 24
    selected = select_e1(
        model, aform[0].slate, FAMILY_DEFINITION_REFERENCE
    )
    assert selected.recipe_id in RECIPE_IDS
    assert 0 <= selected.registry_ordinal < len(RECIPE_IDS)


def test_exact_character_union_coverage_and_rank_discount() -> None:
    context, passages = _manual_corpus()
    answer_start = context.index("Remote definition")
    answers = (
        GoldAnswer(answer_start, "Remote"),
        GoldAnswer(answer_start, "Remote"),
        GoldAnswer(answer_start + len("Remote"), " definition"),
    )
    merged = validate_gold_intervals(context, answers)
    assert merged == (
        CharacterInterval(
            answer_start, answer_start + len("Remote definition")
        ),
    )

    first_rank = score_evidence_coverage(
        passages=passages,
        selected_ordinals=(5, 0, 1, 2, 3),
        merged_gold_intervals=merged,
    )
    last_rank = score_evidence_coverage(
        passages=passages,
        selected_ordinals=(0, 1, 2, 3, 5),
        merged_gold_intervals=merged,
    )
    missed = score_evidence_coverage(
        passages=passages,
        selected_ordinals=(0, 1, 2, 3, 4),
        merged_gold_intervals=merged,
    )
    assert first_rank.primary_utility == INTEGER_SCALE
    assert first_rank.complete_at_5 == first_rank.coverage_at_least_half == 1
    assert first_rank.rank_discounted_incremental_utility == INTEGER_SCALE
    assert (
        0
        < last_rank.rank_discounted_incremental_utility
        < first_rank.rank_discounted_incremental_utility
    )
    assert missed.primary_utility == 0
    assert missed.complete_at_5 == missed.coverage_at_least_half == 0

    unanswerable = score_evidence_coverage(
        passages=passages,
        selected_ordinals=(0, 1, 2, 3, 4),
        merged_gold_intervals=validate_gold_intervals(context, ()),
    )
    assert not unanswerable.answerable
    assert unanswerable.primary_utility is None
    with pytest.raises(MaudExtractionP1CoreError, match="exactly match"):
        validate_gold_intervals(context, (GoldAnswer(answer_start, "wrong"),))


def test_contract_clustering_is_within_family_then_equal_contract() -> None:
    first = ContractCluster(
        (
            ClusterItem(
                FAMILY_DEFINITION_REFERENCE, {"Agent": 1_000, "RAW": 0}
            ),
            ClusterItem(
                FAMILY_DEFINITION_REFERENCE, {"Agent": 1_000, "RAW": 0}
            ),
            ClusterItem(
                FAMILY_CONDITION_OBLIGATION, {"Agent": 0, "RAW": 0}
            ),
        )
    )
    remaining = tuple(
        ContractCluster(
            (
                ClusterItem(
                    FAMILY_DEFINITION_REFERENCE,
                    {"Agent": value, "RAW": 0},
                ),
            )
        )
        for value in (1_000_000, 800_000, 600_000)
    )
    zero = ContractCluster(
        (
            ClusterItem(
                FAMILY_PROTECTION_EXCEPTION_REMEDY,
                {"Agent": None, "RAW": None},
            ),
        )
    )
    comparison = compare_contract_clusters(
        (first, *remaining, zero),
        left_arm="Agent",
        right_arm="RAW",
    )

    assert first.utility("Agent") == Fraction(500)
    assert comparison.paired_contract_deltas == (
        Fraction(500),
        Fraction(1_000_000),
        Fraction(800_000),
        Fraction(600_000),
    )
    assert comparison.equal_weight_contract_mean_delta == Fraction(600_125)
    assert comparison.zero_answerable_contract_count == 1
    assert comparison.family_deltas[FAMILY_DEFINITION_REFERENCE] == Fraction(
        600_250
    )
    assert comparison.sign_flip.reference_tail == Fraction(1, 16)
    assert comparison.promoted

    sign_flip = exact_contract_sign_flip(
        (Fraction(1, 2), Fraction(1, 3), 0)
    )
    assert sign_flip.nonzero_contract_count == 2
    assert sign_flip.reference_tail == Fraction(1, 4)


def test_contract_fails_closed_on_short_recipes_and_arm_answerability_drift() -> None:
    one = build_passages("one short context")
    coordinates = build_coordinate_table_from_quantized(
        query="context",
        passages=one,
        cross_encoder_sigmoid=(500_000,),
        minilm_unit_interval=(500_000,),
        pairwise_minilm_unit_interval=((INTEGER_SCALE,),),
    )
    with pytest.raises(MaudExtractionP1CoreError, match="at least five"):
        materialize_recipe_actions(
            passages=one, coordinates=coordinates, edges=()
        )
    with pytest.raises(MaudExtractionP1CoreError, match="answerability"):
        ClusterItem(
            FAMILY_DEFINITION_REFERENCE,
            {"Agent": None, "RAW": 0},
        )
