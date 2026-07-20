from __future__ import annotations

from fractions import Fraction

import pytest

from assumption_agent.benchmarks import bright_reasoning_retrieval_cross_encoder_formation_v1 as formation


def test_p9_rows_is_exact_equal_rrf_k60() -> None:
    pool = tuple(range(100, 132))
    ordinals = tuple(reversed(range(32)))
    raw = pool[:10]
    hippo = pool[5:15]
    actual = formation.p9_rows(
        candidate_rows=pool,
        cross_encoder_ranked_ordinals=ordinals,
        raw_rows=raw,
        hipporag_rows=hippo,
    )
    cross = tuple(pool[index] for index in ordinals)
    totals = {row: Fraction(0, 1) for row in pool}
    for ranking in (cross, raw, hippo):
        for rank, row in enumerate(ranking, start=1):
            totals[row] += Fraction(1, 60 + rank)
    expected = tuple(sorted(pool, key=lambda row: (-totals[row], row))[:10])
    assert actual == expected


def test_p9_rows_ties_break_by_corpus_row() -> None:
    pool = tuple(reversed(range(32)))
    actual = formation.p9_rows(
        candidate_rows=pool,
        cross_encoder_ranked_ordinals=tuple(range(32)),
        raw_rows=tuple(range(31, 21, -1)),
        hipporag_rows=tuple(range(31, 21, -1)),
    )
    assert actual[0] == 31
    assert len(actual) == 10
    assert len(set(actual)) == 10


def test_p9_rows_rejects_nonpermutation_cross_encoder_output() -> None:
    with pytest.raises(formation.BrightCrossEncoderFormationError):
        formation.p9_rows(
            candidate_rows=tuple(range(32)),
            cross_encoder_ranked_ordinals=(0,) * 32,
            raw_rows=tuple(range(10)),
            hipporag_rows=tuple(range(10, 20)),
        )
