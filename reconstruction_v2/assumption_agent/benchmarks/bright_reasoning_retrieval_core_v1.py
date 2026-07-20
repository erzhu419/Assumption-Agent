"""Pure frozen retrieval, scoring, and evaluator logic for the BRIGHT study."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


FAMILY_ORDER = ("BIOLOGY", "ECONOMICS", "ROBOTICS")
RECIPE_ORDER = (
    "P1_RRF_EQUAL",
    "P2_RRF_ANCHOR2",
    "P3_MAX_SIM",
    "P4_MEAN_SIM",
    "P5_TOP2_MEAN",
    "P6_RELATION_MECHANISM_RRF",
    "P7_ENTITY_CONSTRAINT_RRF",
    "P8_ROUND_ROBIN",
)
EVALUATOR_SCOPES = (
    "ALL_OTHER_ITEMS",
    "SAME_FAMILY_ONLY",
    "OTHER_FAMILIES_ONLY",
)
EVALUATOR_K_VALUES = (3, 5, 9)
EVALUATOR_ALPHA_VALUES = (1, 4, 16)
POOL_SIZE = 32
TOP_K = 10
GLOBAL_QUERY_DEPTH = 64
RRF_K = 60
UTILITY_SCALE = 1_000_000_000


class BrightStudyCoreError(RuntimeError):
    """A frozen pure-study contract failed closed."""


@dataclass(frozen=True)
class LocalRetrieval:
    candidate_rows: tuple[int, ...]
    query_global_rankings: tuple[tuple[int, ...], ...]
    recipe_rows: Mapping[str, tuple[int, ...]]
    raw_rows: tuple[int, ...]


@dataclass(frozen=True)
class EvaluatorSpec:
    scope: str
    k: int
    alpha: int

    @property
    def evaluator_id(self) -> str:
        return f"E1_{self.scope}_K{self.k:02d}_A{self.alpha:02d}"


def _scores(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise BrightStudyCoreError("score vector must be one-dimensional integers")
    return array.astype(np.int64, copy=False)


def stable_top_rows(
    scores: object, *, k: int, excluded_rows: Sequence[int] = ()
) -> tuple[int, ...]:
    values = _scores(scores)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(values):
        raise BrightStudyCoreError("top-k is outside the score vector")
    excluded: set[int] = set()
    for raw in excluded_rows:
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise BrightStudyCoreError("excluded row is invalid")
        row = int(raw)
        if not 0 <= row < len(values) or row in excluded:
            raise BrightStudyCoreError("excluded row is invalid")
        excluded.add(row)
    eligible = np.ones(len(values), dtype=bool)
    if excluded:
        eligible[np.fromiter(sorted(excluded), dtype=np.int64)] = False
    available = int(eligible.sum())
    if available < k:
        raise BrightStudyCoreError("insufficient eligible documents")
    eligible_scores = values[eligible]
    threshold = int(np.partition(eligible_scores, available - k)[available - k])
    greater = np.flatnonzero(eligible & (values > threshold)).tolist()
    equal = np.flatnonzero(eligible & (values == threshold)).tolist()
    greater.sort(key=lambda row: (-int(values[row]), row))
    selected = greater + equal[: k - len(greater)]
    if len(selected) != k:
        raise BrightStudyCoreError("stable top-k construction drifted")
    return tuple(selected)


def _rank_pool(pool: Sequence[int], scores: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(pool, key=lambda row: (-int(scores[row]), row)))


def _rrf_ranking(
    rankings: Sequence[Sequence[int]],
    pool: Sequence[int],
    weights: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if not rankings:
        raise BrightStudyCoreError("RRF requires a ranking")
    if weights is None:
        weights = (1,) * len(rankings)
    if len(weights) != len(rankings) or any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
        for weight in weights
    ):
        raise BrightStudyCoreError("RRF weights drifted")
    allowed = set(pool)
    totals = {row: Fraction(0, 1) for row in pool}
    for ranking, weight in zip(rankings, weights):
        seen: set[int] = set()
        for rank, row in enumerate(ranking, start=1):
            if row not in allowed or row in seen:
                raise BrightStudyCoreError("RRF ranking contains an invalid row")
            seen.add(row)
            totals[row] += Fraction(weight, RRF_K + rank)
    return tuple(sorted(pool, key=lambda row: (-totals[row], row)))


def _round_robin(rankings: Sequence[Sequence[int]], pool: Sequence[int]) -> tuple[int, ...]:
    allowed = set(pool)
    output: list[int] = []
    seen: set[int] = set()
    for depth in range(len(pool)):
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            row = ranking[depth]
            if row not in allowed:
                raise BrightStudyCoreError("round-robin ranking contains an invalid row")
            if row not in seen:
                seen.add(row)
                output.append(row)
                if len(output) == TOP_K:
                    return tuple(output)
    raise BrightStudyCoreError("round-robin did not produce top-k")


def build_local_retrieval(
    query_scores: Sequence[object], *, excluded_rows: Sequence[int] = ()
) -> LocalRetrieval:
    if not 1 <= len(query_scores) <= 5:
        raise BrightStudyCoreError("query score count drifted")
    matrices = tuple(_scores(value) for value in query_scores)
    if len({len(value) for value in matrices}) != 1 or len(matrices[0]) < POOL_SIZE:
        raise BrightStudyCoreError("query score shapes drifted")
    global_rankings = tuple(
        stable_top_rows(value, k=GLOBAL_QUERY_DEPTH, excluded_rows=excluded_rows)
        for value in matrices
    )
    raw_top = global_rankings[0][:TOP_K]
    if len(matrices) == 1:
        pool_members = global_rankings[0][:POOL_SIZE]
    else:
        union = tuple(sorted(set().union(*global_rankings)))
        union_rankings = tuple(
            tuple(row for row in ranking if row in set(union))
            for ranking in global_rankings
        )
        fill_ranking = _rrf_ranking(union_rankings, union)
        selected = list(raw_top)
        selected_set = set(selected)
        for row in fill_ranking:
            if row not in selected_set:
                selected.append(row)
                selected_set.add(row)
                if len(selected) == POOL_SIZE:
                    break
        if len(selected) != POOL_SIZE:
            raise BrightStudyCoreError("candidate pool fill failed")
        pool_members = tuple(selected)
    pool = tuple(sorted(pool_members))
    pool_rankings = tuple(_rank_pool(pool, value) for value in matrices)
    if len(matrices) == 1:
        recipes = {recipe: raw_top for recipe in RECIPE_ORDER}
    else:
        recipes: dict[str, tuple[int, ...]] = {}
        recipes["P1_RRF_EQUAL"] = _rrf_ranking(pool_rankings, pool)[:TOP_K]
        recipes["P2_RRF_ANCHOR2"] = _rrf_ranking(
            pool_rankings, pool, (2, 1, 1, 1, 1)
        )[:TOP_K]
        recipes["P3_MAX_SIM"] = tuple(
            sorted(
                pool,
                key=lambda row: (
                    -max(int(value[row]) for value in matrices),
                    row,
                ),
            )[:TOP_K]
        )
        recipes["P4_MEAN_SIM"] = tuple(
            sorted(
                pool,
                key=lambda row: (
                    -sum(int(value[row]) for value in matrices),
                    row,
                ),
            )[:TOP_K]
        )
        recipes["P5_TOP2_MEAN"] = tuple(
            sorted(
                pool,
                key=lambda row: (
                    -sum(
                        sorted(
                            (int(value[row]) for value in matrices), reverse=True
                        )[:2]
                    ),
                    row,
                ),
            )[:TOP_K]
        )
        recipes["P6_RELATION_MECHANISM_RRF"] = _rrf_ranking(
            (pool_rankings[0], pool_rankings[2], pool_rankings[3]), pool
        )[:TOP_K]
        recipes["P7_ENTITY_CONSTRAINT_RRF"] = _rrf_ranking(
            (pool_rankings[0], pool_rankings[1], pool_rankings[4]), pool
        )[:TOP_K]
        recipes["P8_ROUND_ROBIN"] = _round_robin(pool_rankings, pool)
    if tuple(recipes) != RECIPE_ORDER:
        raise BrightStudyCoreError("recipe registry order drifted")
    for value in recipes.values():
        if len(value) != TOP_K or len(set(value)) != TOP_K or not set(value) <= set(pool):
            raise BrightStudyCoreError("recipe output drifted")
    return LocalRetrieval(
        candidate_rows=pool,
        query_global_rankings=global_rankings,
        recipe_rows=recipes,
        raw_rows=raw_top,
    )


def integer_ndcg_at_10(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> int:
    if len(retrieved_ids) != TOP_K or len(set(retrieved_ids)) != TOP_K:
        raise BrightStudyCoreError("retrieved IDs are not an exact top10")
    gold = tuple(gold_ids)
    if not gold or len(set(gold)) != len(gold):
        raise BrightStudyCoreError("gold IDs are empty or duplicated")
    gold_set = set(gold)
    dcg = math.fsum(
        (1.0 / math.log2(rank + 1)) if value in gold_set else 0.0
        for rank, value in enumerate(retrieved_ids, start=1)
    )
    ideal = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), TOP_K) + 1)
    )
    result = int(round((dcg / ideal) * UTILITY_SCALE))
    if not 0 <= result <= UTILITY_SCALE:
        raise BrightStudyCoreError("integer nDCG drifted")
    return result


def select_f_portfolio(
    utility_rows: Sequence[Mapping[str, int]], size: int = 4
) -> tuple[str, ...]:
    if not utility_rows or size != 4:
        raise BrightStudyCoreError("F selection shape drifted")
    totals = {recipe: 0 for recipe in RECIPE_ORDER}
    for row in utility_rows:
        if tuple(row) != RECIPE_ORDER:
            raise BrightStudyCoreError("F utility registry drifted")
        for recipe, value in row.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UTILITY_SCALE:
                raise BrightStudyCoreError("F utility is invalid")
            totals[recipe] += value
    order = {recipe: index for index, recipe in enumerate(RECIPE_ORDER)}
    return tuple(
        sorted(RECIPE_ORDER, key=lambda recipe: (-totals[recipe], order[recipe]))[:size]
    )


def evaluator_specs() -> tuple[EvaluatorSpec, ...]:
    return tuple(
        EvaluatorSpec(scope=scope, k=k, alpha=alpha)
        for scope in EVALUATOR_SCOPES
        for k in EVALUATOR_K_VALUES
        for alpha in EVALUATOR_ALPHA_VALUES
    )


def _neighbor_indices(
    *,
    target: int,
    families: Sequence[str],
    similarities: np.ndarray,
    scope: str,
    k: int,
) -> tuple[int, ...]:
    if scope not in EVALUATOR_SCOPES or families[target] not in FAMILY_ORDER:
        raise BrightStudyCoreError("evaluator scope or family drifted")
    eligible = []
    for index, family in enumerate(families):
        if index == target:
            continue
        if scope == "SAME_FAMILY_ONLY" and family != families[target]:
            continue
        if scope == "OTHER_FAMILIES_ONLY" and family == families[target]:
            continue
        eligible.append(index)
    if len(eligible) < k:
        raise BrightStudyCoreError("evaluator neighbor capacity drifted")
    return tuple(sorted(eligible, key=lambda index: (-int(similarities[index]), index))[:k])


def _predict_policy(
    *,
    target: int,
    families: Sequence[str],
    similarities: np.ndarray,
    utility_rows: Sequence[Mapping[str, int]],
    portfolio: Sequence[str],
    spec: EvaluatorSpec,
    leave_target_out: bool,
) -> str:
    if tuple(recipe for recipe in portfolio if recipe in RECIPE_ORDER) != tuple(portfolio):
        raise BrightStudyCoreError("evaluator portfolio drifted")
    neighbors = _neighbor_indices(
        target=target,
        families=families,
        similarities=similarities,
        scope=spec.scope,
        k=spec.k,
    )
    global_indices = [index for index in range(len(utility_rows)) if not leave_target_out or index != target]
    if not global_indices:
        raise BrightStudyCoreError("evaluator global training set is empty")
    predictions: dict[str, Fraction] = {}
    for recipe in portfolio:
        global_mean = Fraction(
            sum(utility_rows[index][recipe] for index in global_indices),
            len(global_indices),
        )
        predictions[recipe] = Fraction(
            sum(utility_rows[index][recipe] for index in neighbors), 1
        ) + spec.alpha * global_mean
        predictions[recipe] /= spec.k + spec.alpha
    order = {recipe: index for index, recipe in enumerate(RECIPE_ORDER)}
    return min(portfolio, key=lambda recipe: (-predictions[recipe], order[recipe]))


def select_evaluator_challenger(
    *,
    families: Sequence[str],
    query_embeddings: object,
    utility_rows: Sequence[Mapping[str, int]],
    portfolio: Sequence[str],
) -> tuple[EvaluatorSpec, int]:
    matrix = np.asarray(query_embeddings, dtype=np.float32)
    if matrix.shape != (len(utility_rows), 384) or len(families) != len(utility_rows):
        raise BrightStudyCoreError("A_form evaluator tensors drifted")
    scores: list[tuple[int, str, EvaluatorSpec]] = []
    for spec in evaluator_specs():
        total = 0
        for target in range(len(utility_rows)):
            similarities = np.rint(
                (matrix @ matrix[target]).astype(np.float64) * 1_000_000
            ).astype(np.int32)
            selected = _predict_policy(
                target=target,
                families=families,
                similarities=similarities,
                utility_rows=utility_rows,
                portfolio=portfolio,
                spec=spec,
                leave_target_out=True,
            )
            total += utility_rows[target][selected]
        scores.append((total, spec.evaluator_id, spec))
    total, _identifier, winner = min(scores, key=lambda row: (-row[0], row[1]))
    return winner, total


def route_with_evaluator(
    *,
    target_family: str,
    target_embedding: object,
    training_families: Sequence[str],
    training_embeddings: object,
    training_utility_rows: Sequence[Mapping[str, int]],
    portfolio: Sequence[str],
    spec: EvaluatorSpec,
) -> str:
    matrix = np.asarray(training_embeddings, dtype=np.float32)
    query = np.asarray(target_embedding, dtype=np.float32)
    if matrix.shape != (len(training_utility_rows), 384) or query.shape != (384,):
        raise BrightStudyCoreError("evaluator routing tensors drifted")
    families = tuple(training_families) + (target_family,)
    expanded_utility = tuple(training_utility_rows) + (training_utility_rows[0],)
    similarities = np.rint((matrix @ query).astype(np.float64) * 1_000_000).astype(np.int32)
    similarities = np.concatenate((similarities, np.asarray([0], dtype=np.int32)))
    return _predict_policy(
        target=len(training_utility_rows),
        families=families,
        similarities=similarities,
        utility_rows=expanded_utility,
        portfolio=portfolio,
        spec=spec,
        leave_target_out=True,
    )
