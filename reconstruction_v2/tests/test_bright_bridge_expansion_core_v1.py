from __future__ import annotations

import numpy as np
import pytest

from reconstruction_v2.assumption_agent.benchmarks.bright_bridge_expansion_core_v1 import (
    BRIDGE_QUERY_CAP,
    BRIDGE_RETRIEVAL_DEPTH,
    MAX_EXPANDED_POOL_SIZE,
    BridgeAnchor,
    BrightBridgeExpansionError,
    build_bridge_queries,
    candidate_expansion_diagnostics,
    expand_candidate_pool,
    extract_bridge_anchors,
    integer_ndcg_at_10,
    rank_p10,
    select_seed_rows,
)


def _base_pool() -> tuple[int, ...]:
    return tuple(range(32))


def _bridge_scores(*tops: tuple[int, ...], size: int = 400) -> tuple[np.ndarray, ...]:
    output = []
    for rows in tops:
        values = np.arange(size, dtype=np.int64) * -1
        for rank, row in enumerate(rows):
            values[row] = 10_000 - rank
        output.append(values)
    return tuple(output)


def test_select_seed_rows_uses_combined_score_and_row_tie_break() -> None:
    relation = [0] * 32
    mechanism = [0] * 32
    relation[7] = 10
    mechanism[3] = 10
    assert select_seed_rows(_base_pool(), relation, mechanism) == (3, 7, 0, 1)


def test_select_seed_rows_rejects_non_pool32() -> None:
    with pytest.raises(BrightBridgeExpansionError):
        select_seed_rows(tuple(range(31)), [0] * 31, [0] * 31)


def test_extract_bridge_anchors_is_deterministic_and_query_exclusive() -> None:
    seeds = (3, 7, 0, 1)
    documents = {
        3: "The Bayesian Surprise framework predicts attention. Cortical adaptation follows.",
        7: "Community composting reduces methane emissions through anaerobic diversion.",
        0: "Princess Celestia governs Equestria while Twilight studies friendship.",
        1: "Behavioral reinforcement can create persistent avoidance responses.",
    }
    kwargs = {
        "original_query": "How does attention adaptation work?",
        "relation_query": "attention adaptation relationship",
        "mechanism_query": "mechanism causing attention adaptation",
        "seed_rows": seeds,
        "documents_by_row": documents,
    }
    first = extract_bridge_anchors(**kwargs)
    second = extract_bridge_anchors(**kwargs)
    assert first == second
    assert 1 <= len(first) <= BRIDGE_QUERY_CAP
    forbidden = {"attention", "adaptation", "relationship", "mechanism", "causing", "does", "work"}
    assert all(anchor.normalized not in forbidden for anchor in first)
    assert len({anchor.normalized for anchor in first}) == len(first)


def test_extract_bridge_anchors_returns_empty_without_eligible_terms() -> None:
    anchors = extract_bridge_anchors(
        original_query="what does this mean",
        relation_query="what does this mean",
        mechanism_query="what does this mean",
        seed_rows=(0,),
        documents_by_row={0: "this does mean that this does mean"},
    )
    assert anchors == ()


def test_extract_bridge_anchors_requires_exact_seed_mapping() -> None:
    with pytest.raises(BrightBridgeExpansionError):
        extract_bridge_anchors(
            original_query="original query",
            relation_query="relation query",
            mechanism_query="mechanism query",
            seed_rows=(0,),
            documents_by_row={1: "Novel Anchor Phrase"},
        )


def test_build_bridge_queries_alternates_typed_query_kind() -> None:
    anchors = (
        BridgeAnchor(1, 0, 0, 0, "Bayesian Surprise", "bayesian surprise"),
        BridgeAnchor(2, 1, 0, 0, "methane emissions", "methane emissions"),
        BridgeAnchor(3, 2, 0, 0, "Princess Celestia", "princess celestia"),
    )
    queries = build_bridge_queries(
        relation_query="relation base",
        mechanism_query="mechanism base",
        anchors=anchors,
    )
    assert [value.query_kind for value in queries] == [
        "relation_query",
        "mechanism_query",
        "relation_query",
    ]
    assert queries[0].text == "relation base Bayesian Surprise"
    assert queries[1].text == "mechanism base methane emissions"


def test_build_bridge_queries_rejects_duplicate_anchors() -> None:
    anchor = BridgeAnchor(1, 0, 0, 0, "Novel Anchor", "novel anchor")
    with pytest.raises(BrightBridgeExpansionError):
        build_bridge_queries(
            relation_query="relation base",
            mechanism_query="mechanism base",
            anchors=(anchor, anchor),
        )


def test_expand_candidate_pool_adds_outside_candidates_and_retains_base() -> None:
    scores = _bridge_scores(tuple(range(100, 164)), tuple(range(200, 264)))
    expanded = expand_candidate_pool(base_pool=_base_pool(), bridge_score_vectors=scores)
    assert expanded.base_pool == _base_pool()
    assert len(expanded.bridge_rankings) == 2
    assert expanded.outside_base_count == 128
    assert len(expanded.expanded_pool) == 160
    assert set(_base_pool()) <= set(expanded.expanded_pool)


def test_expand_candidate_pool_obeys_frozen_cap() -> None:
    tops = tuple(tuple(range(32 + 64 * i, 32 + 64 * (i + 1))) for i in range(4))
    expanded = expand_candidate_pool(
        base_pool=_base_pool(),
        bridge_score_vectors=_bridge_scores(*tops, size=400),
    )
    assert len(expanded.expanded_pool) == MAX_EXPANDED_POOL_SIZE
    assert expanded.outside_base_count == 4 * BRIDGE_RETRIEVAL_DEPTH


def test_expand_candidate_pool_rejects_more_than_four_vectors() -> None:
    with pytest.raises(BrightBridgeExpansionError):
        expand_candidate_pool(
            base_pool=_base_pool(),
            bridge_score_vectors=tuple(np.arange(400, dtype=np.int64) for _ in range(5)),
        )


def test_expand_candidate_pool_respects_excluded_rows() -> None:
    scores = _bridge_scores(tuple(range(100, 164)))
    expanded = expand_candidate_pool(
        base_pool=_base_pool(),
        bridge_score_vectors=scores,
        excluded_rows=(100, 101),
    )
    assert 100 not in expanded.bridge_rankings[0]
    assert 101 not in expanded.bridge_rankings[0]


def test_rank_p10_can_promote_a_bridge_only_candidate() -> None:
    size = 400
    scores = _bridge_scores(tuple(range(100, 164)), size=size)
    expanded = expand_candidate_pool(
        base_pool=_base_pool(),
        bridge_score_vectors=scores,
    )
    original = np.arange(size, dtype=np.int64) * -1
    relation = original.copy()
    mechanism = original.copy()
    pool = expanded.expanded_pool
    ce_relation = [0] * len(pool)
    ce_mechanism = [0] * len(pool)
    bridge_index = pool.index(100)
    ce_relation[bridge_index] = 1_000_000
    ce_mechanism[bridge_index] = 1_000_000
    ranked = rank_p10(
        expanded=expanded,
        original_scores=original,
        relation_scores=relation,
        mechanism_scores=mechanism,
        cross_encoder_relation_scores=ce_relation,
        cross_encoder_mechanism_scores=ce_mechanism,
    )
    assert 100 in ranked.rows
    assert len(ranked.rows) == 10
    assert len(set(ranked.rows)) == 10


def test_rank_p10_is_byte_stable_for_equal_inputs() -> None:
    size = 400
    expanded = expand_candidate_pool(
        base_pool=_base_pool(),
        bridge_score_vectors=_bridge_scores(tuple(range(100, 164)), size=size),
    )
    direct = np.arange(size, dtype=np.int64) * -1
    ce = list(range(len(expanded.expanded_pool)))
    kwargs = {
        "expanded": expanded,
        "original_scores": direct,
        "relation_scores": direct,
        "mechanism_scores": direct,
        "cross_encoder_relation_scores": ce,
        "cross_encoder_mechanism_scores": ce,
    }
    assert rank_p10(**kwargs) == rank_p10(**kwargs)


def test_rank_p10_rejects_cross_encoder_shape_drift() -> None:
    size = 400
    expanded = expand_candidate_pool(
        base_pool=_base_pool(),
        bridge_score_vectors=_bridge_scores(tuple(range(100, 164)), size=size),
    )
    direct = np.arange(size, dtype=np.int64)
    with pytest.raises(BrightBridgeExpansionError):
        rank_p10(
            expanded=expanded,
            original_scores=direct,
            relation_scores=direct,
            mechanism_scores=direct,
            cross_encoder_relation_scores=[0],
            cross_encoder_mechanism_scores=[0],
        )


def test_candidate_expansion_diagnostics_counts_causal_span() -> None:
    result = candidate_expansion_diagnostics(
        base_pool=_base_pool(),
        expanded_pool=tuple(range(40)),
        p10_rows=(0, 1, 2, 3, 4, 5, 32, 33, 34, 35),
        gold_rows=(33, 50),
    )
    assert result == {
        "expanded_pool_size": 40,
        "unique_bridge_candidates_outside_base_pool": 8,
        "P10_top10_documents_outside_base_pool": 4,
        "gold_documents_absent_from_base_pool_but_recovered_by_P10_top10": 1,
    }


def test_integer_ndcg_at_10_is_exactly_scaled() -> None:
    assert integer_ndcg_at_10(tuple(range(10)), (0,)) == 1_000_000_000
    assert 0 < integer_ndcg_at_10(tuple(range(10)), (9,)) < 1_000_000_000


@pytest.mark.parametrize(
    "bad_document",
    ["", "   ", None],
)
def test_extract_bridge_anchors_rejects_invalid_document(bad_document: object) -> None:
    with pytest.raises(BrightBridgeExpansionError):
        extract_bridge_anchors(
            original_query="original query",
            relation_query="relation query",
            mechanism_query="mechanism query",
            seed_rows=(0,),
            documents_by_row={0: bad_document},  # type: ignore[dict-item]
        )
