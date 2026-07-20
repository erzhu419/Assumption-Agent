from __future__ import annotations

import numpy as np

from assumption_agent.benchmarks import bright_reasoning_retrieval_core_v1 as core


def _score_vectors() -> list[np.ndarray]:
    base = np.arange(100, dtype=np.int32)
    return [base, -base, np.roll(base, 7), np.roll(base, 13), np.roll(base, 29)]


def test_candidate_pool_preserves_raw_and_all_recipe_contracts() -> None:
    result = core.build_local_retrieval(_score_vectors(), excluded_rows=(99,))
    assert len(result.candidate_rows) == 32
    assert tuple(result.recipe_rows) == core.RECIPE_ORDER
    assert set(result.raw_rows) <= set(result.candidate_rows)
    assert 99 not in result.candidate_rows
    assert all(len(rows) == 10 and len(set(rows)) == 10 for rows in result.recipe_rows.values())


def test_invalid_generation_collapses_every_recipe_to_raw() -> None:
    result = core.build_local_retrieval(_score_vectors()[:1])
    assert all(rows == result.raw_rows for rows in result.recipe_rows.values())
    assert len(result.candidate_rows) == 32


def test_binary_ndcg_and_portfolio_selection() -> None:
    retrieved = tuple(f"d{index}" for index in range(10))
    assert core.integer_ndcg_at_10(retrieved, ("d0",)) == core.UTILITY_SCALE
    assert core.integer_ndcg_at_10(retrieved, ("missing",)) == 0
    rows = []
    for _ in range(3):
        rows.append(
            {
                recipe: index * 10
                for index, recipe in enumerate(core.RECIPE_ORDER, start=1)
            }
        )
    assert core.select_f_portfolio(rows) == tuple(reversed(core.RECIPE_ORDER[-4:]))


def test_evaluator_challenger_and_routing_are_deterministic() -> None:
    count = 60
    families = tuple(core.FAMILY_ORDER[index // 20] for index in range(count))
    matrix = np.zeros((count, 384), dtype=np.float32)
    for index in range(count):
        matrix[index, index % 20] = 1.0
    rows = []
    for index in range(count):
        preferred = core.RECIPE_ORDER[index % 4]
        rows.append(
            {
                recipe: (core.UTILITY_SCALE if recipe == preferred else 0)
                for recipe in core.RECIPE_ORDER
            }
        )
    portfolio = core.RECIPE_ORDER[:4]
    spec, total = core.select_evaluator_challenger(
        families=families,
        query_embeddings=matrix,
        utility_rows=rows,
        portfolio=portfolio,
    )
    assert spec in core.evaluator_specs()
    assert total >= 0
    selected = core.route_with_evaluator(
        target_family="BIOLOGY",
        target_embedding=matrix[0],
        training_families=families,
        training_embeddings=matrix,
        training_utility_rows=rows,
        portfolio=portfolio,
        spec=spec,
    )
    assert selected in portfolio
