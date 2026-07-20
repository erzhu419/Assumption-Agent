from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_p11_formation_v1 as formation,
)
from reconstruction_v2.assumption_agent.benchmarks import p11_raw_ce_rrf_v1 as p11


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_p11_exact_weighted_rrf() -> None:
    pool = tuple(range(12))
    raw = tuple(range(10))
    relation = tuple(range(12))
    mechanism = tuple(range(12))
    observed = p11.rank_p11(
        expanded_pool=pool,
        raw_top10=raw,
        cross_encoder_relation_scores=relation,
        cross_encoder_mechanism_scores=mechanism,
    )
    ce = tuple(reversed(pool))
    totals = {row: Fraction() for row in pool}
    for rank, row in enumerate(raw, 1):
        totals[row] += Fraction(1, 60 + rank)
    for rank, row in enumerate(ce, 1):
        totals[row] += Fraction(2, 60 + rank)
    expected = tuple(sorted(pool, key=lambda row: (-totals[row], row))[:10])
    assert observed == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expanded_pool": range(9), "raw_top10": range(10)},
        {"expanded_pool": range(12), "raw_top10": [0] * 10},
        {"expanded_pool": range(12), "raw_top10": range(3, 13)},
    ],
)
def test_p11_rejects_bad_candidate_shapes(kwargs: dict[str, object]) -> None:
    with pytest.raises(p11.P11RankingError):
        p11.rank_p11(
            expanded_pool=kwargs["expanded_pool"],  # type: ignore[arg-type]
            raw_top10=kwargs["raw_top10"],  # type: ignore[arg-type]
            cross_encoder_relation_scores=[0] * 12,
            cross_encoder_mechanism_scores=[0] * 12,
        )


def test_frozen_formation_artifacts_match() -> None:
    formation._verify_artifacts(PROJECT_ROOT / "reconstruction_v2")


def test_formation_is_one_shot(tmp_path: Path) -> None:
    base = tmp_path / "project" / "reconstruction_v2"
    result = base / formation.RESULT_RELATIVE
    result.parent.mkdir(parents=True)
    result.write_text("consumed")
    with pytest.raises(formation.FiqaP11FormationError):
        formation.run_formation(tmp_path / "project")
