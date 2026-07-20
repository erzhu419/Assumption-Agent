"""Frozen P11 ranker: RAW-preserving weighted RRF with expanded-pool CE."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral
from typing import Sequence


TOP_K = 10
RRF_K = 60
RAW_WEIGHT = 1
CROSS_ENCODER_WEIGHT = 2
CANDIDATE_NAME = "P11_RAW1_CE_SUM2_EXPANDED_RRF_K60"


class P11RankingError(RuntimeError):
    """The frozen P11 ranking contract failed closed."""


def _integer_rows(value: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise P11RankingError(f"{name} is not a row sequence")
    rows = tuple(value)
    if any(isinstance(row, bool) or not isinstance(row, Integral) or row < 0 for row in rows):
        raise P11RankingError(f"{name} contains an invalid row")
    result = tuple(int(row) for row in rows)
    if len(set(result)) != len(result):
        raise P11RankingError(f"{name} contains duplicates")
    return result


def rank_p11(
    *,
    expanded_pool: Sequence[int],
    raw_top10: Sequence[int],
    cross_encoder_relation_scores: Sequence[int],
    cross_encoder_mechanism_scores: Sequence[int],
) -> tuple[int, ...]:
    """Return the exact P11 top-10 without any label-dependent decision."""

    pool = _integer_rows(expanded_pool, "expanded pool")
    raw = _integer_rows(raw_top10, "RAW top10")
    if len(pool) < TOP_K or len(raw) != TOP_K or not set(raw) <= set(pool):
        raise P11RankingError("candidate pool or RAW top10 drifted")
    relation = tuple(cross_encoder_relation_scores)
    mechanism = tuple(cross_encoder_mechanism_scores)
    if len(relation) != len(pool) or len(mechanism) != len(pool):
        raise P11RankingError("cross-encoder score shape drifted")
    for scores, name in ((relation, "relation"), (mechanism, "mechanism")):
        if any(isinstance(score, bool) or not isinstance(score, Integral) for score in scores):
            raise P11RankingError(f"{name} score is not an integer")
    combined = {
        row: int(relation[index]) + int(mechanism[index])
        for index, row in enumerate(pool)
    }
    cross_encoder = tuple(sorted(pool, key=lambda row: (-combined[row], row)))
    totals = {row: Fraction(0, 1) for row in pool}
    for rank, row in enumerate(raw, start=1):
        totals[row] += Fraction(RAW_WEIGHT, RRF_K + rank)
    for rank, row in enumerate(cross_encoder, start=1):
        totals[row] += Fraction(CROSS_ENCODER_WEIGHT, RRF_K + rank)
    result = tuple(sorted(pool, key=lambda row: (-totals[row], row))[:TOP_K])
    if len(result) != TOP_K or len(set(result)) != TOP_K:
        raise P11RankingError("P11 top10 drifted")
    return result


__all__ = [
    "CANDIDATE_NAME",
    "CROSS_ENCODER_WEIGHT",
    "P11RankingError",
    "RAW_WEIGHT",
    "RRF_K",
    "TOP_K",
    "rank_p11",
]
