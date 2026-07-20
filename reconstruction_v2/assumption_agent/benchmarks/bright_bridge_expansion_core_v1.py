"""Pure typed bridge-candidate expansion for the second BRIGHT source epoch.

The module is deliberately independent of parquet readers, labels, model
runtimes, and filesystem state.  It turns a frozen first-stage 32-document
pool into auditable bridge queries, expands that pool with full-corpus score
vectors, and produces a deterministic P10 ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks.bright_reasoning_retrieval_core_v1 import (
    BrightStudyCoreError,
    GLOBAL_QUERY_DEPTH,
    POOL_SIZE,
    RRF_K,
    TOP_K,
    stable_top_rows,
)


BRIDGE_QUERY_CAP = 4
BRIDGE_RETRIEVAL_DEPTH = GLOBAL_QUERY_DEPTH
MAX_EXPANDED_POOL_SIZE = POOL_SIZE + BRIDGE_QUERY_CAP * BRIDGE_RETRIEVAL_DEPTH
MAX_ANCHOR_CHARACTERS = 96
MAX_BRIDGE_QUERY_CHARACTERS = 768
SEED_DOCUMENT_CAP = 4

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?:\r?\n)+|(?<=[.!?;])\s+")
_SPACE_RE = re.compile(r"\s+")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "also",
        "among",
        "another",
        "because",
        "before",
        "being",
        "between",
        "both",
        "could",
        "does",
        "doing",
        "during",
        "each",
        "either",
        "from",
        "further",
        "have",
        "having",
        "here",
        "into",
        "itself",
        "many",
        "might",
        "more",
        "most",
        "other",
        "over",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "using",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "within",
        "would",
        "your",
    }
)


class BrightBridgeExpansionError(BrightStudyCoreError):
    """The frozen bridge-expansion contract failed closed."""


@dataclass(frozen=True)
class BridgeAnchor:
    seed_row: int
    seed_rank: int
    sentence_rank: int
    token_start: int
    text: str
    normalized: str


@dataclass(frozen=True)
class BridgeQuery:
    seed_row: int
    anchor: str
    query_kind: str
    text: str


@dataclass(frozen=True)
class ExpandedCandidatePool:
    base_pool: tuple[int, ...]
    bridge_rankings: tuple[tuple[int, ...], ...]
    expanded_pool: tuple[int, ...]
    outside_base_count: int


@dataclass(frozen=True)
class P10Ranking:
    rows: tuple[int, ...]
    cross_encoder_ranking: tuple[int, ...]
    component_rankings: tuple[tuple[int, ...], ...]
    expanded_pool: tuple[int, ...]


@dataclass(frozen=True)
class _Token:
    surface: str
    normalized: str
    start: int
    capitalized: bool


@dataclass(frozen=True)
class _AnchorCandidate:
    seed_row: int
    seed_rank: int
    sentence_rank: int
    token_start: int
    text: str
    normalized: str
    capitalized_count: int
    token_count: int
    character_count: int


def _clean_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise BrightBridgeExpansionError(f"{name} is not text")
    cleaned = _SPACE_RE.sub(" ", value).strip()
    if not cleaned:
        raise BrightBridgeExpansionError(f"{name} is empty")
    return cleaned


def _tokens(value: str) -> tuple[_Token, ...]:
    output: list[_Token] = []
    for match in _TOKEN_RE.finditer(value):
        surface = match.group(0)
        normalized = surface.casefold()
        output.append(
            _Token(
                surface=surface,
                normalized=normalized,
                start=match.start(),
                capitalized=surface[0].isupper(),
            )
        )
    return tuple(output)


def _forbidden_terms(*queries: str) -> frozenset[str]:
    return frozenset(token.normalized for query in queries for token in _tokens(query))


def select_seed_rows(
    base_pool: Sequence[int],
    relation_scores: Sequence[int],
    mechanism_scores: Sequence[int],
    *,
    seed_cap: int = SEED_DOCUMENT_CAP,
) -> tuple[int, ...]:
    pool = tuple(base_pool)
    if len(pool) != POOL_SIZE or len(set(pool)) != POOL_SIZE:
        raise BrightBridgeExpansionError("base pool is not an exact unique pool32")
    if isinstance(seed_cap, bool) or not isinstance(seed_cap, int) or not 1 <= seed_cap <= 4:
        raise BrightBridgeExpansionError("seed cap drifted")
    if len(relation_scores) != len(pool) or len(mechanism_scores) != len(pool):
        raise BrightBridgeExpansionError("seed score shape drifted")
    scored: list[tuple[int, int]] = []
    for row, relation, mechanism in zip(pool, relation_scores, mechanism_scores):
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or row < 0
            or isinstance(relation, bool)
            or not isinstance(relation, int)
            or isinstance(mechanism, bool)
            or not isinstance(mechanism, int)
        ):
            raise BrightBridgeExpansionError("seed score value drifted")
        scored.append((row, relation + mechanism))
    return tuple(row for row, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:seed_cap])


def _candidate_sentences(
    *,
    seed_row: int,
    seed_rank: int,
    document: str,
    forbidden: frozenset[str],
) -> tuple[_AnchorCandidate, ...]:
    raw_sentences = tuple(
        value.strip() for value in _SENTENCE_RE.split(document) if value.strip()
    )
    candidates: list[_AnchorCandidate] = []
    for sentence_rank, sentence in enumerate(raw_sentences):
        tokens = _tokens(sentence)
        eligible = tuple(
            token
            for token in tokens
            if len(token.normalized) >= 4
            and token.normalized not in _STOPWORDS
            and token.normalized not in forbidden
        )
        if not eligible:
            continue
        for index, token in enumerate(eligible):
            candidates.append(
                _AnchorCandidate(
                    seed_row=seed_row,
                    seed_rank=seed_rank,
                    sentence_rank=sentence_rank,
                    token_start=token.start,
                    text=token.surface[:MAX_ANCHOR_CHARACTERS],
                    normalized=token.normalized,
                    capitalized_count=int(token.capitalized),
                    token_count=1,
                    character_count=len(token.surface),
                )
            )
            if index + 1 >= len(eligible):
                continue
            right = eligible[index + 1]
            if right.start - (token.start + len(token.surface)) > 4:
                continue
            text = f"{token.surface} {right.surface}"
            normalized = f"{token.normalized} {right.normalized}"
            if len(text) <= MAX_ANCHOR_CHARACTERS:
                candidates.append(
                    _AnchorCandidate(
                        seed_row=seed_row,
                        seed_rank=seed_rank,
                        sentence_rank=sentence_rank,
                        token_start=token.start,
                        text=text,
                        normalized=normalized,
                        capitalized_count=int(token.capitalized)
                        + int(right.capitalized),
                        token_count=2,
                        character_count=len(token.surface) + len(right.surface),
                    )
                )
    return tuple(candidates)


def extract_bridge_anchors(
    *,
    original_query: str,
    relation_query: str,
    mechanism_query: str,
    seed_rows: Sequence[int],
    documents_by_row: Mapping[int, str],
    anchor_cap: int = BRIDGE_QUERY_CAP,
) -> tuple[BridgeAnchor, ...]:
    original = _clean_text(original_query, "original query")
    relation = _clean_text(relation_query, "relation query")
    mechanism = _clean_text(mechanism_query, "mechanism query")
    seeds = tuple(seed_rows)
    if (
        isinstance(anchor_cap, bool)
        or not isinstance(anchor_cap, int)
        or not 1 <= anchor_cap <= BRIDGE_QUERY_CAP
    ):
        raise BrightBridgeExpansionError("anchor cap drifted")
    if not seeds or len(seeds) > SEED_DOCUMENT_CAP or len(set(seeds)) != len(seeds):
        raise BrightBridgeExpansionError("seed rows drifted")
    if any(isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in seeds):
        raise BrightBridgeExpansionError("seed row is invalid")
    if set(documents_by_row) != set(seeds):
        raise BrightBridgeExpansionError("seed document mapping drifted")

    forbidden = _forbidden_terms(original, relation, mechanism)
    per_seed: dict[int, tuple[_AnchorCandidate, ...]] = {}
    document_frequency: dict[str, int] = {}
    for seed_rank, row in enumerate(seeds):
        document = _clean_text(documents_by_row[row], "seed document")
        candidates = _candidate_sentences(
            seed_row=row,
            seed_rank=seed_rank,
            document=document,
            forbidden=forbidden,
        )
        per_seed[row] = candidates
        for normalized in {candidate.normalized for candidate in candidates}:
            document_frequency[normalized] = document_frequency.get(normalized, 0) + 1

    selected: list[BridgeAnchor] = []
    used: set[str] = set()
    for seed_rank, row in enumerate(seeds):
        ranked = sorted(
            per_seed[row],
            key=lambda candidate: (
                -candidate.capitalized_count,
                document_frequency[candidate.normalized],
                -candidate.token_count,
                -candidate.character_count,
                candidate.sentence_rank,
                candidate.token_start,
                candidate.normalized,
            ),
        )
        chosen = next(
            (candidate for candidate in ranked if candidate.normalized not in used),
            None,
        )
        if chosen is None:
            continue
        used.add(chosen.normalized)
        selected.append(
            BridgeAnchor(
                seed_row=row,
                seed_rank=seed_rank,
                sentence_rank=chosen.sentence_rank,
                token_start=chosen.token_start,
                text=chosen.text,
                normalized=chosen.normalized,
            )
        )
        if len(selected) == anchor_cap:
            break
    return tuple(selected)


def build_bridge_queries(
    *,
    relation_query: str,
    mechanism_query: str,
    anchors: Sequence[BridgeAnchor],
) -> tuple[BridgeQuery, ...]:
    relation = _clean_text(relation_query, "relation query")
    mechanism = _clean_text(mechanism_query, "mechanism query")
    values = tuple(anchors)
    if len(values) > BRIDGE_QUERY_CAP:
        raise BrightBridgeExpansionError("too many bridge anchors")
    if len({value.normalized for value in values}) != len(values):
        raise BrightBridgeExpansionError("bridge anchors are duplicated")
    queries: list[BridgeQuery] = []
    seen: set[str] = set()
    for index, anchor in enumerate(values):
        if (
            not isinstance(anchor, BridgeAnchor)
            or not anchor.text.strip()
            or not anchor.normalized.strip()
        ):
            raise BrightBridgeExpansionError("bridge anchor drifted")
        kind = "relation_query" if index % 2 == 0 else "mechanism_query"
        base = relation if kind == "relation_query" else mechanism
        text = _SPACE_RE.sub(" ", f"{base} {anchor.text}").strip()
        if len(text) > MAX_BRIDGE_QUERY_CHARACTERS:
            text = text[:MAX_BRIDGE_QUERY_CHARACTERS].rstrip()
        normalized = text.casefold()
        if not text or normalized in seen:
            raise BrightBridgeExpansionError("bridge query is empty or duplicated")
        seen.add(normalized)
        queries.append(
            BridgeQuery(
                seed_row=anchor.seed_row,
                anchor=anchor.text,
                query_kind=kind,
                text=text,
            )
        )
    return tuple(queries)


def expand_candidate_pool(
    *,
    base_pool: Sequence[int],
    bridge_score_vectors: Sequence[object],
    excluded_rows: Sequence[int] = (),
) -> ExpandedCandidatePool:
    base = tuple(base_pool)
    if len(base) != POOL_SIZE or len(set(base)) != POOL_SIZE:
        raise BrightBridgeExpansionError("base pool is not an exact unique pool32")
    matrices = tuple(np.asarray(value) for value in bridge_score_vectors)
    if len(matrices) > BRIDGE_QUERY_CAP:
        raise BrightBridgeExpansionError("too many bridge score vectors")
    if matrices:
        if len({len(value) for value in matrices}) != 1:
            raise BrightBridgeExpansionError("bridge score vector shapes drifted")
        if len(matrices[0]) < BRIDGE_RETRIEVAL_DEPTH:
            raise BrightBridgeExpansionError("bridge corpus is too small")
    rankings = tuple(
        stable_top_rows(
            value,
            k=BRIDGE_RETRIEVAL_DEPTH,
            excluded_rows=excluded_rows,
        )
        for value in matrices
    )
    expanded = tuple(sorted(set(base).union(*rankings)))
    if not set(base) <= set(expanded):
        raise BrightBridgeExpansionError("base pool was not retained")
    if len(expanded) > MAX_EXPANDED_POOL_SIZE:
        raise BrightBridgeExpansionError("expanded pool exceeds the frozen cap")
    return ExpandedCandidatePool(
        base_pool=base,
        bridge_rankings=rankings,
        expanded_pool=expanded,
        outside_base_count=len(set(expanded) - set(base)),
    )


def _integer_vector(value: object, expected: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or len(array) != expected
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise BrightBridgeExpansionError(f"{name} score shape drifted")
    return array.astype(np.int64, copy=False)


def _rank_rows(rows: Sequence[int], scores: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(rows, key=lambda row: (-int(scores[row]), row)))


def _rrf(
    rankings: Sequence[Sequence[int]],
    pool: Sequence[int],
) -> tuple[int, ...]:
    allowed = set(pool)
    if not rankings or not allowed:
        raise BrightBridgeExpansionError("RRF inputs are empty")
    totals = {row: Fraction(0, 1) for row in pool}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, row in enumerate(ranking, start=1):
            if row not in allowed or row in seen:
                raise BrightBridgeExpansionError("RRF ranking drifted")
            seen.add(row)
            totals[row] += Fraction(1, RRF_K + rank)
    return tuple(sorted(pool, key=lambda row: (-totals[row], row)))


def rank_p10(
    *,
    expanded: ExpandedCandidatePool,
    original_scores: object,
    relation_scores: object,
    mechanism_scores: object,
    cross_encoder_relation_scores: Sequence[int],
    cross_encoder_mechanism_scores: Sequence[int],
) -> P10Ranking:
    pool = expanded.expanded_pool
    if not pool:
        raise BrightBridgeExpansionError("expanded pool is empty")
    corpus_size = max(pool) + 1
    original = np.asarray(original_scores)
    relation = np.asarray(relation_scores)
    mechanism = np.asarray(mechanism_scores)
    if (
        original.ndim != 1
        or relation.ndim != 1
        or mechanism.ndim != 1
        or len({len(original), len(relation), len(mechanism)}) != 1
        or len(original) < corpus_size
        or not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (original, relation, mechanism)
        )
    ):
        raise BrightBridgeExpansionError("direct score vector shapes drifted")
    ce_relation = _integer_vector(
        cross_encoder_relation_scores,
        len(pool),
        "cross encoder relation",
    )
    ce_mechanism = _integer_vector(
        cross_encoder_mechanism_scores,
        len(pool),
        "cross encoder mechanism",
    )
    ce_by_row = {
        row: int(ce_relation[index]) + int(ce_mechanism[index])
        for index, row in enumerate(pool)
    }
    ce_ranking = tuple(sorted(pool, key=lambda row: (-ce_by_row[row], row)))
    component_rankings = (
        ce_ranking,
        _rank_rows(pool, original),
        _rank_rows(pool, relation),
        _rank_rows(pool, mechanism),
        *expanded.bridge_rankings,
    )
    ranking = _rrf(component_rankings, pool)
    rows = ranking[:TOP_K]
    if len(rows) != TOP_K or len(set(rows)) != TOP_K:
        raise BrightBridgeExpansionError("P10 top10 drifted")
    return P10Ranking(
        rows=rows,
        cross_encoder_ranking=ce_ranking,
        component_rankings=tuple(component_rankings),
        expanded_pool=pool,
    )


def candidate_expansion_diagnostics(
    *,
    base_pool: Sequence[int],
    expanded_pool: Sequence[int],
    p10_rows: Sequence[int],
    gold_rows: Sequence[int] | None = None,
) -> Mapping[str, int]:
    base = set(base_pool)
    expanded = set(expanded_pool)
    p10 = tuple(p10_rows)
    if len(p10) != TOP_K or len(set(p10)) != TOP_K or not set(p10) <= expanded:
        raise BrightBridgeExpansionError("diagnostic P10 ranking drifted")
    result = {
        "expanded_pool_size": len(expanded),
        "unique_bridge_candidates_outside_base_pool": len(expanded - base),
        "P10_top10_documents_outside_base_pool": len(set(p10) - base),
    }
    if gold_rows is not None:
        gold = set(gold_rows)
        result["gold_documents_absent_from_base_pool_but_recovered_by_P10_top10"] = len(
            (gold - base).intersection(p10)
        )
    return result


def integer_ndcg_at_10(
    retrieved_rows: Sequence[int],
    gold_rows: Sequence[int],
) -> int:
    retrieved = tuple(retrieved_rows)
    gold = tuple(gold_rows)
    if len(retrieved) != TOP_K or len(set(retrieved)) != TOP_K:
        raise BrightBridgeExpansionError("retrieved rows are not an exact top10")
    if not gold or len(set(gold)) != len(gold):
        raise BrightBridgeExpansionError("gold rows are empty or duplicated")
    gold_set = set(gold)
    dcg = math.fsum(
        (1.0 / math.log2(rank + 1)) if row in gold_set else 0.0
        for rank, row in enumerate(retrieved, start=1)
    )
    ideal = math.fsum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), TOP_K) + 1)
    )
    return int(round((dcg / ideal) * 1_000_000_000))
