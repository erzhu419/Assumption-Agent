"""Closed typed-operator and causal-evaluator algebra for MultiHop-RAG v2.

This module is deliberately row-free and model-free.  The frozen offline NER,
embedding, official-HippoRAG, source reader, and late-label adapters live at
separate boundaries.  Here, an article is represented only by label-free
metadata, typed entity keys, reciprocal topic neighbors, and one dense query
relevance integer.  Neither evidence URLs, evidence facts, answers, nor gold
article IDs are accepted by any action or evaluator function.

All six actions scan every ordered article pair and every remaining article
during two core extensions.  HippoRAG results are a separate comparator and
cannot limit the Agent candidate space.  The challenger evaluator is a fixed
causal lexicographic rule over query-grounded leave-one-out necessity and typed
path connectivity.  Type-preserving replacement is recorded as a diagnostic
only; there is no learned weight, prompt, keyword threshold, or runner-up.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import date
from fractions import Fraction
import hashlib
import json
import math
import re
import unicodedata
from typing import Iterable, Mapping, Sequence


VERSION = "multihoprag_typed_operator_v2"
TOP_K = 5
CORE_SIZE = 4
INTEGER_SCALE = 1_000_000
TOPIC_K = 4

CAPABILITIES = ("comparison_query", "inference_query", "temporal_query")
ENTITY_TYPES = ("LOC", "MISC", "ORG", "PER")
ACTION_IDS = (
    "P0_IND_SUM",
    "P1_IND_MAXIMIN",
    "P2_ENTITY_BRIDGE",
    "P3_TOPIC_BRIDGE",
    "P4_META_ASSIGN",
    "P5_FAMILY_UNION",
)

SAME_TYPED_ENTITY = "SAME_TYPED_ENTITY"
CROSS_SOURCE_TYPED_ENTITY = "CROSS_SOURCE_TYPED_ENTITY"
TYPED_ENTITY_TEMPORAL_ORDER = "TYPED_ENTITY_TEMPORAL_ORDER"
RECIPROCAL_TOPIC_KNN = "RECIPROCAL_TOPIC_KNN"
SAME_SOURCE = "SAME_SOURCE"
EDGE_FAMILIES = (
    SAME_TYPED_ENTITY,
    CROSS_SOURCE_TYPED_ENTITY,
    TYPED_ENTITY_TEMPORAL_ORDER,
    RECIPROCAL_TOPIC_KNN,
    SAME_SOURCE,
)

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_ISO_DATE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-/]([01]?\d)[-/]([0-3]?\d)(?!\d)"
)
_ISO_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})[-/]([01]?\d)(?![-/\d])")
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_MONTH_DAY_YEAR = re.compile(
    rf"(?<![A-Za-z])({_MONTH_NAMES})(?![A-Za-z])\s+([0-3]?\d)(?:st|nd|rd|th)?"
    rf"(?:\s*,\s*|\s+)((?:19|20)\d{{2}})(?!\d)",
    flags=re.IGNORECASE,
)
_DAY_MONTH_YEAR = re.compile(
    rf"(?<!\d)([0-3]?\d)(?:st|nd|rd|th)?\s+({_MONTH_NAMES})(?![A-Za-z])"
    rf"(?:\s*,\s*|\s+)((?:19|20)\d{{2}})(?!\d)",
    flags=re.IGNORECASE,
)
_MONTH_YEAR = re.compile(
    rf"(?<![A-Za-z])({_MONTH_NAMES})(?![A-Za-z])\s*,?\s*((?:19|20)\d{{2}})(?!\d)",
    flags=re.IGNORECASE,
)
_MONTH_NUMBER = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


class MultiHopRAGTypedOperatorV2Error(ValueError):
    """Raised when a typed action or evaluator boundary drifts."""


def _valid_calendar_day(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _valid_date_ordinal(value: int) -> bool:
    year, remainder = divmod(value, 10_000)
    month, day = divmod(remainder, 100)
    if not 1900 <= year <= 2099:
        return False
    if month == 0:
        return day == 0
    if not 1 <= month <= 12:
        return False
    return day == 0 or _valid_calendar_day(year, month, day)


@dataclass(frozen=True)
class FrozenMapping(Mapping[object, object]):
    """Small pickle-safe immutable mapping used inside process-shared receipts."""

    rows: tuple[tuple[object, object], ...]

    def __getitem__(self, key: object) -> object:
        for candidate, value in self.rows:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self.rows)

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, order=True)
class EntityKey:
    entity_type: str
    normalized_span: str


@dataclass(frozen=True)
class ArticleRecord:
    article_i: int
    normalized_source: str
    normalized_category: str
    published_ordinal: int | None
    entities: tuple[EntityKey, ...]
    reciprocal_topic_neighbors: tuple[int, ...]


@dataclass(frozen=True, order=True)
class TypedEdge:
    family_order: int
    left_article_i: int
    right_article_i: int

    @property
    def family(self) -> str:
        if self.family_order not in range(len(EDGE_FAMILIES)):
            raise MultiHopRAGTypedOperatorV2Error("edge family order is invalid")
        return EDGE_FAMILIES[self.family_order]

    def public_tuple(self) -> tuple[str, int, int]:
        return (self.family, self.left_article_i, self.right_article_i)


@dataclass(frozen=True)
class TypedCorpusGraph:
    articles: tuple[ArticleRecord, ...]
    sources: tuple[str, ...]
    entity_documents: Mapping[EntityKey, tuple[int, ...]]
    edges: tuple[TypedEdge, ...]
    neighbors: Mapping[str, tuple[tuple[int, ...], ...]]
    temporal_successors: tuple[tuple[int, ...], ...]
    graph_sha256: str


@dataclass(frozen=True)
class QueryPlan:
    capability: str
    capability_similarity_ints: tuple[int, int, int]
    normalized_sources: tuple[str, ...]
    entities: tuple[EntityKey, ...]
    date_ordinals: tuple[int, ...]
    graph_sha256: str
    query_sha256: str
    plan_sha256: str


@dataclass(frozen=True)
class CoverageSignature:
    covered: int
    total: int
    value: Fraction
    slot_keys: tuple[str, ...]
    covered_slot_keys: tuple[str, ...]


@dataclass(frozen=True)
class CausalSignature:
    necessary_count: int
    necessary_fraction: Fraction
    minimum_leave_one_out_loss: Fraction
    minimum_replacement_loss: Fraction
    path_connectivity: Fraction


@dataclass(frozen=True)
class ActionTrace:
    action_id: str
    output_top5: tuple[int, int, int, int, int]
    core: tuple[int, int, int, int]
    core_quality: tuple[Fraction | int, ...]
    coverage: CoverageSignature
    causal: CausalSignature
    e0_key: tuple[Fraction | int, ...]
    e1_key: tuple[Fraction | int, ...]
    ordered_pair_scan_count: int
    extension_scan_count: int
    graph_sha256: str
    plan_sha256: str
    query_sha256: str
    relevance_sha256: str
    trace_sha256: str


@dataclass(frozen=True)
class EvaluationObservation:
    traces_by_action: Mapping[str, ActionTrace]


@dataclass(frozen=True)
class PolicySelection:
    evaluator_id: str
    action_id: str
    observation_count: int
    macro_key: tuple[Fraction, ...]
    per_action_macro_keys: tuple[tuple[str, tuple[Fraction, ...]], ...]
    input_receipt_sha256: str
    selection_sha256: str


@dataclass(frozen=True)
class PairedUtilitySummary:
    count: int
    left_total: Fraction
    right_total: Fraction
    delta_total: Fraction
    gains: int
    harms: int
    ties: int
    exact_one_sided_p: Fraction


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def make_entity_key(entity_type: str, span: str) -> EntityKey:
    if entity_type not in ENTITY_TYPES:
        raise MultiHopRAGTypedOperatorV2Error("entity type is outside the frozen registry")
    normalized = normalize_text(span)
    if not normalized or "\x00" in normalized or len(normalized) > 512:
        raise MultiHopRAGTypedOperatorV2Error("entity span is invalid")
    return EntityKey(entity_type, normalized)


def parse_date_ordinals(value: str) -> tuple[int, ...]:
    """Return non-overlapping date operands in mention order.

    The parser intentionally accepts only ISO dates/months, English month-date
    forms, English month-year forms, and standalone four-digit years.  A more
    specific span occupies its characters before a less specific rule runs, so
    an ISO date cannot also emit its year and separate month/year mentions can
    never form a Cartesian product.
    """

    if not isinstance(value, str):
        raise TypeError("date text must be a string")
    value = unicodedata.normalize("NFKC", value)
    occupied: list[tuple[int, int]] = []
    candidates: list[tuple[int, int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < old_end and old_start < end for old_start, old_end in occupied)

    def record(match: re.Match[str], ordinal: int) -> None:
        start, end = match.span()
        if not overlaps(start, end):
            occupied.append((start, end))
            candidates.append((start, end, ordinal))

    def block(match: re.Match[str]) -> None:
        start, end = match.span()
        if not overlaps(start, end):
            occupied.append((start, end))

    for match in _ISO_DATE.finditer(value):
        year, month, day = (int(part) for part in match.groups())
        if _valid_calendar_day(year, month, day):
            record(match, year * 10_000 + month * 100 + day)
        else:
            block(match)
    for match in _MONTH_DAY_YEAR.finditer(value):
        month_name, day_text, year_text = match.groups()
        month = _MONTH_NUMBER[month_name.casefold()]
        day = int(day_text)
        if _valid_calendar_day(int(year_text), month, day):
            record(match, int(year_text) * 10_000 + month * 100 + day)
        else:
            block(match)
    for match in _DAY_MONTH_YEAR.finditer(value):
        day_text, month_name, year_text = match.groups()
        month = _MONTH_NUMBER[month_name.casefold()]
        day = int(day_text)
        if _valid_calendar_day(int(year_text), month, day):
            record(match, int(year_text) * 10_000 + month * 100 + day)
        else:
            block(match)
    for match in _ISO_MONTH.finditer(value):
        year, month = (int(part) for part in match.groups())
        if 1 <= month <= 12:
            record(match, year * 10_000 + month * 100)
        else:
            block(match)
    for match in _MONTH_YEAR.finditer(value):
        month_name, year_text = match.groups()
        record(match, int(year_text) * 10_000 + _MONTH_NUMBER[month_name.casefold()] * 100)
    for match in _YEAR.finditer(value):
        record(match, int(match.group(0)) * 10_000)

    seen: set[int] = set()
    ordered: list[int] = []
    for _start, _end, ordinal in sorted(candidates):
        if ordinal not in seen:
            seen.add(ordinal)
            ordered.append(ordinal)
    return tuple(ordered)


def _edge(family: str, left: int, right: int) -> TypedEdge:
    if family not in EDGE_FAMILIES or left == right:
        raise MultiHopRAGTypedOperatorV2Error("typed edge is invalid")
    lo, hi = sorted((left, right))
    return TypedEdge(EDGE_FAMILIES.index(family), lo, hi)


def _graph_receipt_body(
    *,
    articles: Sequence[ArticleRecord],
    sources: Sequence[str],
    entity_documents: Mapping[EntityKey, Sequence[int]],
    edges: Sequence[TypedEdge],
    neighbors: Mapping[str, Sequence[Sequence[int]]],
    temporal_successors: Sequence[Sequence[int]],
) -> dict[str, object]:
    return {
        "articles": [
            {
                "article_i": row.article_i,
                "category": row.normalized_category,
                "entities": [[entity.entity_type, entity.normalized_span] for entity in row.entities],
                "published_ordinal": row.published_ordinal,
                "source": row.normalized_source,
                "topic_neighbors": list(row.reciprocal_topic_neighbors),
            }
            for row in articles
        ],
        "edges": [edge.public_tuple() for edge in edges],
        "entity_documents": [
            [entity.entity_type, entity.normalized_span, list(documents)]
            for entity, documents in sorted(entity_documents.items())
        ],
        "neighbors": {
            family: [list(values) for values in neighbors[family]] for family in EDGE_FAMILIES
        },
        "sources": list(sources),
        "temporal_successors": [list(values) for values in temporal_successors],
        "version": VERSION,
    }


def _validated_graph(graph: TypedCorpusGraph) -> TypedCorpusGraph:
    if not isinstance(graph, TypedCorpusGraph):
        raise MultiHopRAGTypedOperatorV2Error("graph has the wrong type")
    count = len(graph.articles)
    if (
        count < TOP_K
        or set(graph.neighbors) != set(EDGE_FAMILIES)
        or any(len(graph.neighbors[family]) != count for family in EDGE_FAMILIES)
        or len(graph.temporal_successors) != count
        or tuple(sorted(set(graph.sources))) != graph.sources
    ):
        raise MultiHopRAGTypedOperatorV2Error("graph topology drifted")
    body = _graph_receipt_body(
        articles=graph.articles,
        sources=graph.sources,
        entity_documents=graph.entity_documents,
        edges=graph.edges,
        neighbors=graph.neighbors,
        temporal_successors=graph.temporal_successors,
    )
    if graph.graph_sha256 != _stable_hash(body):
        raise MultiHopRAGTypedOperatorV2Error("graph receipt drifted")
    return graph


def build_typed_corpus_graph(articles: Sequence[ArticleRecord]) -> TypedCorpusGraph:
    """Validate compiled article features and build the fixed typed graph."""

    if isinstance(articles, (str, bytes)) or not isinstance(articles, Sequence):
        raise MultiHopRAGTypedOperatorV2Error("articles must be a sequence")
    rows = tuple(articles)
    if len(rows) < TOP_K:
        raise MultiHopRAGTypedOperatorV2Error("corpus is smaller than top-k")
    checked: list[ArticleRecord] = []
    entity_documents_mutable: dict[EntityKey, list[int]] = {}
    for position, article in enumerate(rows):
        if not isinstance(article, ArticleRecord) or article.article_i != position:
            raise MultiHopRAGTypedOperatorV2Error("article IDs must be contiguous corpus order")
        source = normalize_text(article.normalized_source)
        category = normalize_text(article.normalized_category)
        if not source or not category:
            raise MultiHopRAGTypedOperatorV2Error("source and category must be non-empty")
        if article.published_ordinal is not None and (
            isinstance(article.published_ordinal, bool)
            or not isinstance(article.published_ordinal, int)
            or not _valid_date_ordinal(article.published_ordinal)
        ):
            raise MultiHopRAGTypedOperatorV2Error("published ordinal is invalid")
        raw_entities = tuple(article.entities)
        if any(
            not isinstance(entity, EntityKey)
            or entity.entity_type not in ENTITY_TYPES
            or not isinstance(entity.normalized_span, str)
            or not entity.normalized_span
            or normalize_text(entity.normalized_span) != entity.normalized_span
            or "\x00" in entity.normalized_span
            or len(entity.normalized_span) > 512
            for entity in raw_entities
        ):
            raise MultiHopRAGTypedOperatorV2Error("article entity is invalid")
        entities = tuple(sorted(set(raw_entities)))
        topic_neighbors = tuple(article.reciprocal_topic_neighbors)
        if len(topic_neighbors) > TOPIC_K or len(set(topic_neighbors)) != len(topic_neighbors):
            raise MultiHopRAGTypedOperatorV2Error("topic neighbors violate k or uniqueness")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(rows)
            or value == position
            for value in topic_neighbors
        ):
            raise MultiHopRAGTypedOperatorV2Error("topic neighbor is invalid")
        checked_row = ArticleRecord(
            position,
            source,
            category,
            article.published_ordinal,
            entities,
            tuple(sorted(topic_neighbors)),
        )
        checked.append(checked_row)
        for entity in entities:
            entity_documents_mutable.setdefault(entity, []).append(position)

    # Reciprocal means every declared direction must be declared by its peer.
    for article in checked:
        for neighbor in article.reciprocal_topic_neighbors:
            if article.article_i not in checked[neighbor].reciprocal_topic_neighbors:
                raise MultiHopRAGTypedOperatorV2Error("topic neighbor relation is not reciprocal")

    entity_documents = {
        entity: tuple(documents)
        for entity, documents in sorted(entity_documents_mutable.items())
        if 2 <= len(documents) <= 32
    }
    retained_entities = set(entity_documents)
    checked = [
        ArticleRecord(
            article.article_i,
            article.normalized_source,
            article.normalized_category,
            article.published_ordinal,
            tuple(entity for entity in article.entities if entity in retained_entities),
            article.reciprocal_topic_neighbors,
        )
        for article in checked
    ]

    edge_set: set[TypedEdge] = set()
    for documents in entity_documents.values():
        for offset, left in enumerate(documents):
            for right in documents[offset + 1 :]:
                edge_set.add(_edge(SAME_TYPED_ENTITY, left, right))
                if checked[left].normalized_source != checked[right].normalized_source:
                    edge_set.add(_edge(CROSS_SOURCE_TYPED_ENTITY, left, right))
                left_date = checked[left].published_ordinal
                right_date = checked[right].published_ordinal
                if left_date is not None and right_date is not None and left_date != right_date:
                    edge_set.add(_edge(TYPED_ENTITY_TEMPORAL_ORDER, left, right))
    for article in checked:
        for neighbor in article.reciprocal_topic_neighbors:
            edge_set.add(_edge(RECIPROCAL_TOPIC_KNN, article.article_i, neighbor))
    by_source: dict[str, list[int]] = {}
    for article in checked:
        by_source.setdefault(article.normalized_source, []).append(article.article_i)
    for documents in by_source.values():
        for offset, left in enumerate(documents):
            for right in documents[offset + 1 :]:
                edge_set.add(_edge(SAME_SOURCE, left, right))

    edges = tuple(sorted(edge_set))
    neighbor_sets: dict[str, list[set[int]]] = {
        family: [set() for _ in checked] for family in EDGE_FAMILIES
    }
    for edge in edges:
        neighbor_sets[edge.family][edge.left_article_i].add(edge.right_article_i)
        neighbor_sets[edge.family][edge.right_article_i].add(edge.left_article_i)
    neighbors = {
        family: tuple(tuple(sorted(values)) for values in neighbor_sets[family])
        for family in EDGE_FAMILIES
    }
    temporal_successor_sets: list[set[int]] = [set() for _ in checked]
    for edge in edges:
        if edge.family != TYPED_ENTITY_TEMPORAL_ORDER:
            continue
        left_date = checked[edge.left_article_i].published_ordinal
        right_date = checked[edge.right_article_i].published_ordinal
        if left_date is None or right_date is None or left_date == right_date:
            raise MultiHopRAGTypedOperatorV2Error("temporal edge lost its strict date order")
        earlier, later = (
            (edge.left_article_i, edge.right_article_i)
            if left_date < right_date
            else (edge.right_article_i, edge.left_article_i)
        )
        temporal_successor_sets[earlier].add(later)
    temporal_successors = tuple(tuple(sorted(values)) for values in temporal_successor_sets)
    sources = tuple(sorted(by_source))
    graph_body = _graph_receipt_body(
        articles=checked,
        sources=sources,
        entity_documents=entity_documents,
        edges=edges,
        neighbors=neighbors,
        temporal_successors=temporal_successors,
    )
    return TypedCorpusGraph(
        articles=tuple(checked),
        sources=sources,
        entity_documents=FrozenMapping(tuple(sorted(entity_documents.items()))),
        edges=edges,
        neighbors=FrozenMapping(tuple((family, neighbors[family]) for family in EDGE_FAMILIES)),
        temporal_successors=temporal_successors,
        graph_sha256=_stable_hash(graph_body),
    )


def compile_query_plan(
    *,
    graph: TypedCorpusGraph,
    query: str,
    capability_similarity_ints: Mapping[str, int],
    query_entities: Sequence[EntityKey],
) -> QueryPlan:
    """Compile typed query metadata using the frozen semantic prototype router."""

    graph = _validated_graph(graph)
    if not isinstance(query, str) or not query.strip() or "\x00" in query:
        raise MultiHopRAGTypedOperatorV2Error("query is invalid")
    if (
        not isinstance(capability_similarity_ints, Mapping)
        or set(capability_similarity_ints) != set(CAPABILITIES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -INTEGER_SCALE <= value <= INTEGER_SCALE
            for value in capability_similarity_ints.values()
        )
    ):
        raise MultiHopRAGTypedOperatorV2Error("capability prototype scores are invalid")
    score_tuple = tuple(capability_similarity_ints[name] for name in CAPABILITIES)
    capability = min(
        CAPABILITIES,
        key=lambda name: (-capability_similarity_ints[name], CAPABILITIES.index(name)),
    )
    if isinstance(query_entities, (str, bytes)) or not isinstance(query_entities, Sequence):
        raise MultiHopRAGTypedOperatorV2Error("query entities must be a sequence")
    if any(
        not isinstance(entity, EntityKey)
        or entity.entity_type not in ENTITY_TYPES
        or not isinstance(entity.normalized_span, str)
        or not entity.normalized_span
        or normalize_text(entity.normalized_span) != entity.normalized_span
        for entity in query_entities
    ):
        raise MultiHopRAGTypedOperatorV2Error("query entity is invalid")
    entities = tuple(sorted(set(query_entities) & set(graph.entity_documents)))
    normalized_query = normalize_text(query)
    sources = tuple(
        source
        for source in graph.sources
        if re.search(rf"(?<![\w]){re.escape(source)}(?![\w])", normalized_query)
    )
    query_sha256 = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
    plan_body = {
        "capability": capability,
        "capability_similarity_ints": list(score_tuple),
        "date_ordinals": list(parse_date_ordinals(query)),
        "entities": [[entity.entity_type, entity.normalized_span] for entity in entities],
        "graph_sha256": graph.graph_sha256,
        "normalized_sources": list(sorted(sources)),
        "query_sha256": query_sha256,
        "version": VERSION,
    }
    return QueryPlan(
        capability=capability,
        capability_similarity_ints=score_tuple,
        normalized_sources=tuple(sorted(sources)),
        entities=entities,
        date_ordinals=tuple(plan_body["date_ordinals"]),
        graph_sha256=graph.graph_sha256,
        query_sha256=query_sha256,
        plan_sha256=_stable_hash(plan_body),
    )


def _validated_plan(graph: TypedCorpusGraph, plan: QueryPlan) -> QueryPlan:
    if not isinstance(plan, QueryPlan):
        raise MultiHopRAGTypedOperatorV2Error("query plan has the wrong type")
    if plan.graph_sha256 != graph.graph_sha256:
        raise MultiHopRAGTypedOperatorV2Error("query plan belongs to another graph")
    scores = tuple(plan.capability_similarity_ints)
    if (
        len(scores) != len(CAPABILITIES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -INTEGER_SCALE <= value <= INTEGER_SCALE
            for value in scores
        )
    ):
        raise MultiHopRAGTypedOperatorV2Error("query plan scores drifted")
    expected_capability = min(
        CAPABILITIES,
        key=lambda name: (-scores[CAPABILITIES.index(name)], CAPABILITIES.index(name)),
    )
    if plan.capability != expected_capability:
        raise MultiHopRAGTypedOperatorV2Error("query plan capability drifted")
    if (
        tuple(sorted(set(plan.normalized_sources))) != plan.normalized_sources
        or not set(plan.normalized_sources) <= set(graph.sources)
    ):
        raise MultiHopRAGTypedOperatorV2Error("query plan sources drifted")
    if (
        tuple(sorted(set(plan.entities))) != plan.entities
        or not set(plan.entities) <= set(graph.entity_documents)
        or any(
            entity.entity_type not in ENTITY_TYPES
            or not isinstance(entity.normalized_span, str)
            or normalize_text(entity.normalized_span) != entity.normalized_span
            for entity in plan.entities
        )
    ):
        raise MultiHopRAGTypedOperatorV2Error("query plan entities drifted")
    if len(set(plan.date_ordinals)) != len(plan.date_ordinals) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not _valid_date_ordinal(value)
        for value in plan.date_ordinals
    ):
        raise MultiHopRAGTypedOperatorV2Error("query plan dates drifted")
    if not re.fullmatch(r"[0-9a-f]{64}", plan.query_sha256):
        raise MultiHopRAGTypedOperatorV2Error("query hash drifted")
    body = {
        "capability": plan.capability,
        "capability_similarity_ints": list(scores),
        "date_ordinals": list(plan.date_ordinals),
        "entities": [[entity.entity_type, entity.normalized_span] for entity in plan.entities],
        "graph_sha256": plan.graph_sha256,
        "normalized_sources": list(plan.normalized_sources),
        "query_sha256": plan.query_sha256,
        "version": VERSION,
    }
    if plan.plan_sha256 != _stable_hash(body):
        raise MultiHopRAGTypedOperatorV2Error("query plan receipt drifted")
    return plan


def _validated_relevance(values: Sequence[int], count: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MultiHopRAGTypedOperatorV2Error("relevance must be a sequence")
    rows = tuple(values)
    if len(rows) != count:
        raise MultiHopRAGTypedOperatorV2Error("relevance count differs from corpus")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -INTEGER_SCALE <= value <= INTEGER_SCALE
        for value in rows
    ):
        raise MultiHopRAGTypedOperatorV2Error("relevance is not frozen cosine integer scale")
    return rows


def _pair_count(values: Sequence[int]) -> int:
    return len(values) * (len(values) - 1) // 2


def _shared_entities(graph: TypedCorpusGraph, left: int, right: int) -> int:
    return len(set(graph.articles[left].entities) & set(graph.articles[right].entities))


def _has_edge(graph: TypedCorpusGraph, family: str, left: int, right: int) -> bool:
    return right in graph.neighbors[family][left]


def _connected_by_families(
    graph: TypedCorpusGraph,
    core: Sequence[int],
    families: Sequence[str],
) -> bool:
    rows = tuple(core)
    if len(rows) <= 1:
        return bool(rows)
    allowed = tuple(families)
    if not allowed or any(family not in EDGE_FAMILIES for family in allowed):
        raise MultiHopRAGTypedOperatorV2Error("connectivity family registry drifted")
    target = set(rows)
    seen = {rows[0]}
    queue = deque([rows[0]])
    while queue:
        current = queue.popleft()
        for family in allowed:
            for neighbor in graph.neighbors[family][current]:
                if neighbor in target and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return seen == target


def _capability_path_families(capability: str) -> tuple[str, ...]:
    if capability == "inference_query":
        return (SAME_TYPED_ENTITY, CROSS_SOURCE_TYPED_ENTITY, RECIPROCAL_TOPIC_KNN)
    if capability == "comparison_query":
        return (
            SAME_TYPED_ENTITY,
            CROSS_SOURCE_TYPED_ENTITY,
            RECIPROCAL_TOPIC_KNN,
            SAME_SOURCE,
        )
    if capability == "temporal_query":
        return (
            SAME_TYPED_ENTITY,
            CROSS_SOURCE_TYPED_ENTITY,
            TYPED_ENTITY_TEMPORAL_ORDER,
            RECIPROCAL_TOPIC_KNN,
            SAME_SOURCE,
        )
    raise MultiHopRAGTypedOperatorV2Error("capability drifted")


def _temporal_order_key(
    graph: TypedCorpusGraph, core: Sequence[int]
) -> tuple[int, int, int]:
    """Directed temporal closure, tuple-order consistency, and dated count."""

    rows = tuple(core)
    directed_edges = 0
    ordered_pairs = 0
    for offset, left in enumerate(rows):
        left_date = graph.articles[left].published_ordinal
        for right in rows[offset + 1 :]:
            right_date = graph.articles[right].published_ordinal
            if right in graph.temporal_successors[left] or left in graph.temporal_successors[right]:
                directed_edges += 1
            if left_date is not None and right_date is not None and left_date < right_date:
                ordered_pairs += 1
    dated_count = sum(graph.articles[index].published_ordinal is not None for index in rows)
    return (directed_edges, ordered_pairs, dated_count)


def _slot_rows(plan: QueryPlan) -> tuple[tuple[str, object], ...]:
    rows: list[tuple[str, object]] = []
    rows.extend((f"source:{source}", source) for source in plan.normalized_sources)
    rows.extend(
        (f"entity:{entity.entity_type}:{entity.normalized_span}", entity)
        for entity in plan.entities
    )
    rows.extend((f"date:{value}", value) for value in plan.date_ordinals)
    return tuple(rows)


def _article_covers_slot(
    article: ArticleRecord, slot: tuple[str, object]
) -> bool:
    key, value = slot
    if key.startswith("source:"):
        return article.normalized_source == value
    if key.startswith("entity:"):
        return value in article.entities
    if key.startswith("date:"):
        if article.published_ordinal is None:
            return False
        requested = int(value)
        if requested % 10_000 == 0:
            return article.published_ordinal // 10_000 == requested // 10_000
        if requested % 100 == 0:
            return article.published_ordinal // 100 == requested // 100
        return article.published_ordinal == requested
    raise MultiHopRAGTypedOperatorV2Error("unknown slot type")


def _maximum_slot_matching(
    graph: TypedCorpusGraph, core: Sequence[int], slots: Sequence[tuple[str, object]]
) -> int:
    """Exact small bipartite matching between declared operands and articles."""

    match_for_doc: dict[int, int] = {}

    def augment(slot_i: int, seen: set[int]) -> bool:
        for article_i in core:
            if article_i in seen or not _article_covers_slot(graph.articles[article_i], slots[slot_i]):
                continue
            seen.add(article_i)
            if article_i not in match_for_doc or augment(match_for_doc[article_i], seen):
                match_for_doc[article_i] = slot_i
                return True
        return False

    matched = 0
    for slot_i in range(len(slots)):
        matched += int(augment(slot_i, set()))
    return matched


def coverage_signature(
    graph: TypedCorpusGraph, plan: QueryPlan, core: Sequence[int]
) -> CoverageSignature:
    """Capability-specific typed requirement coverage C(S)."""

    rows = tuple(core)
    if len(set(rows)) != len(rows) or any(not 0 <= value < len(graph.articles) for value in rows):
        raise MultiHopRAGTypedOperatorV2Error("coverage core is invalid")
    declared = list(_slot_rows(plan))
    covered_keys = {
        key
        for key, value in declared
        if any(_article_covers_slot(graph.articles[index], (key, value)) for index in rows)
    }
    requirements: list[tuple[str, bool]] = [
        (key, key in covered_keys) for key, _value in declared
    ]
    if plan.capability == "inference_query":
        requirements.append(
            (
                "relation:entity_or_topic_connected",
                _connected_by_families(graph, rows, _capability_path_families(plan.capability)),
            )
        )
        requirements.append(("relation:multi_document", len(rows) >= 2))
    elif plan.capability == "comparison_query":
        matching = _maximum_slot_matching(graph, rows, declared)
        requirements.append(("relation:one_to_one", matching == min(len(declared), len(rows))))
        requirements.append(
            ("relation:distinct_sources", len({graph.articles[index].normalized_source for index in rows}) >= 2)
        )
    elif plan.capability == "temporal_query":
        dates = [graph.articles[index].published_ordinal for index in rows]
        valid_dates = [value for value in dates if value is not None]
        requirements.append(("relation:two_dates", len(set(valid_dates)) >= 2))
        requirements.append(
            ("relation:total_order", len(valid_dates) == len(rows) and len(set(valid_dates)) == len(rows))
        )
        requirements.append(
            (
                "relation:typed_connected",
                _connected_by_families(graph, rows, _capability_path_families(plan.capability)),
            )
        )
    else:  # defensive even though QueryPlan is validated
        raise MultiHopRAGTypedOperatorV2Error("capability drifted")
    if not requirements:
        requirements.append(("relation:multi_document", len(rows) >= 2))
    covered = sum(flag for _key, flag in requirements)
    return CoverageSignature(
        covered=covered,
        total=len(requirements),
        value=Fraction(covered, len(requirements)),
        slot_keys=tuple(key for key, _flag in requirements),
        covered_slot_keys=tuple(key for key, flag in requirements if flag),
    )


def _metadata_slots(plan: QueryPlan) -> tuple[tuple[str, object], ...]:
    return tuple(
        slot for slot in _slot_rows(plan) if slot[0].startswith("source:") or slot[0].startswith("date:")
    )


def _metadata_coverage(graph: TypedCorpusGraph, plan: QueryPlan, core: Sequence[int]) -> Fraction:
    slots = _metadata_slots(plan)
    if not slots:
        return Fraction(0)
    covered = sum(
        any(_article_covers_slot(graph.articles[index], slot) for index in core)
        for slot in slots
    )
    return Fraction(covered, len(slots))


def _redundancy(graph: TypedCorpusGraph, output: Sequence[int]) -> int:
    total = 0
    rows = tuple(output)
    for offset, left in enumerate(rows):
        for right in rows[offset + 1 :]:
            total += int(
                graph.articles[left].normalized_source == graph.articles[right].normalized_source
            )
            total += int(bool(set(graph.articles[left].entities) & set(graph.articles[right].entities)))
    return total


def _relation_quality(
    action_id: str,
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    core: Sequence[int],
    relevance: Sequence[int],
) -> tuple[Fraction | int, ...]:
    rows = tuple(core)
    values = [relevance[index] for index in rows]
    sum_relevance = sum(values)
    minimum_relevance = min(values)
    if action_id == "P0_IND_SUM":
        return (sum_relevance, minimum_relevance)
    if action_id == "P1_IND_MAXIMIN":
        return (minimum_relevance, sum_relevance)
    if action_id == "P2_ENTITY_BRIDGE":
        entity_edges = sum(
            _shared_entities(graph, left, right)
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
        )
        return (entity_edges, minimum_relevance, sum_relevance)
    if action_id == "P3_TOPIC_BRIDGE":
        topic_edges = sum(
            _has_edge(graph, RECIPROCAL_TOPIC_KNN, left, right)
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
        )
        return (
            topic_edges,
            int(_connected_by_families(graph, rows, (RECIPROCAL_TOPIC_KNN,))),
            minimum_relevance,
            sum_relevance,
        )
    if action_id == "P4_META_ASSIGN":
        metadata_slots = _metadata_slots(plan)
        temporal_key = _temporal_order_key(graph, rows)
        return (
            _maximum_slot_matching(graph, rows, metadata_slots),
            *temporal_key,
            _metadata_coverage(graph, plan, rows),
            minimum_relevance,
            sum_relevance,
        )
    if action_id == "P5_FAMILY_UNION":
        entity_edges = sum(
            _shared_entities(graph, left, right)
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
        )
        topic_edges = sum(
            _has_edge(graph, RECIPROCAL_TOPIC_KNN, left, right)
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
        )
        coverage = coverage_signature(graph, plan, rows)
        if plan.capability == "inference_query":
            return (
                int(
                    _connected_by_families(
                        graph, rows, _capability_path_families(plan.capability)
                    )
                ),
                entity_edges + topic_edges,
                coverage.value,
                minimum_relevance,
                sum_relevance,
            )
        if plan.capability == "comparison_query":
            slots = _slot_rows(plan)
            return (
                _maximum_slot_matching(graph, rows, slots),
                coverage.value,
                len({graph.articles[index].normalized_source for index in rows}),
                minimum_relevance,
                sum_relevance,
            )
        date_slots = tuple(slot for slot in _slot_rows(plan) if slot[0].startswith("date:"))
        return (
            _maximum_slot_matching(graph, rows, date_slots),
            *_temporal_order_key(graph, rows),
            int(
                _connected_by_families(
                    graph, rows, _capability_path_families(plan.capability)
                )
            ),
            coverage.value,
            minimum_relevance,
            sum_relevance,
        )
    raise MultiHopRAGTypedOperatorV2Error("unknown action")


def _best_pair(
    action_id: str,
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    relevance: Sequence[int],
) -> tuple[int, int]:
    best_key: tuple[Fraction | int, ...] | None = None
    best_pair: tuple[int, int] | None = None
    for left in range(len(graph.articles)):
        for right in range(len(graph.articles)):
            if left == right:
                continue
            quality = _relation_quality(action_id, graph, plan, (left, right), relevance)
            key = (*quality, -left, -right)
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (left, right)
    if best_pair is None:  # pragma: no cover - corpus validation makes this impossible
        raise MultiHopRAGTypedOperatorV2Error("pair scan produced no pair")
    return best_pair


def _extend_core(
    action_id: str,
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    relevance: Sequence[int],
    core: Sequence[int],
) -> tuple[int, ...]:
    rows = tuple(core)
    while len(rows) < CORE_SIZE:
        old_quality = _relation_quality(action_id, graph, plan, rows, relevance)
        candidates: list[tuple[tuple[Fraction | int, ...], int]] = []
        for article_i in range(len(graph.articles)):
            if article_i in rows:
                continue
            new_rows = (*rows, article_i)
            new_quality = _relation_quality(action_id, graph, plan, new_rows, relevance)
            # The complete quality, not a lossy scalar, is the frozen marginal key.
            candidates.append(((*new_quality, *old_quality, relevance[article_i], -article_i), article_i))
        _key, selected = max(candidates, key=lambda row: row[0])
        rows = (*rows, selected)
    return rows


def _fill_tail(
    action_id: str,
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    relevance: Sequence[int],
    core: Sequence[int],
) -> tuple[int, ...]:
    rows = tuple(core)
    if len(rows) != CORE_SIZE or len(set(rows)) != CORE_SIZE:
        raise MultiHopRAGTypedOperatorV2Error("tail received an invalid core")
    typed_tail = action_id in {"P3_TOPIC_BRIDGE", "P4_META_ASSIGN", "P5_FAMILY_UNION"}
    candidates: list[tuple[tuple[Fraction | int, ...], int]] = []
    old_coverage = coverage_signature(graph, plan, rows).value
    for article_i in range(len(graph.articles)):
        if article_i in rows:
            continue
        if typed_tail:
            new_rows = (*rows, article_i)
            new_coverage = coverage_signature(graph, plan, new_rows).value
            key = (new_coverage - old_coverage, relevance[article_i], -article_i)
        else:
            key = (relevance[article_i], -article_i)
        candidates.append((key, article_i))
    _key, tail = max(candidates, key=lambda row: row[0])
    return (*rows, tail)


def _path_connectivity(
    graph: TypedCorpusGraph, plan: QueryPlan, core: Sequence[int]
) -> Fraction:
    rows = tuple(core)
    slots = _slot_rows(plan)
    seeds = {
        article_i
        for article_i in rows
        if any(_article_covers_slot(graph.articles[article_i], slot) for slot in slots)
    }
    if not seeds:
        return Fraction(0)
    allowed = _capability_path_families(plan.capability)
    target = set(rows)
    reachable = set(seeds)
    queue = deque(sorted(seeds))
    while queue:
        current = queue.popleft()
        for family in allowed:
            for neighbor in graph.neighbors[family][current]:
                if neighbor in target and neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
    return Fraction(len(reachable), len(rows))


def _replacement_for(
    graph: TypedCorpusGraph,
    relevance: Sequence[int],
    core: Sequence[int],
    removed: int,
) -> int | None:
    article = graph.articles[removed]
    candidates = [
        row.article_i
        for row in graph.articles
        if row.article_i not in core and row.normalized_source == article.normalized_source
    ]
    if not candidates:
        candidates = [
            row.article_i
            for row in graph.articles
            if row.article_i not in core and row.normalized_category == article.normalized_category
        ]
    if not candidates:
        return None

    def date_distance(candidate_i: int) -> int:
        left = article.published_ordinal
        right = graph.articles[candidate_i].published_ordinal
        if left is None or right is None:
            return 10**12
        return abs(left - right)

    return min(
        candidates,
        key=lambda candidate_i: (
            abs(relevance[candidate_i] - relevance[removed]),
            date_distance(candidate_i),
            candidate_i,
        ),
    )


def causal_signature(
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    core: Sequence[int],
    relevance: Sequence[int],
) -> CausalSignature:
    rows = tuple(core)
    if len(rows) != CORE_SIZE or len(set(rows)) != CORE_SIZE:
        raise MultiHopRAGTypedOperatorV2Error("causal signature requires the frozen core size")
    original = coverage_signature(graph, plan, rows).value
    leave_losses: list[Fraction] = []
    replacement_losses: list[Fraction] = []
    for removed in rows:
        reduced = tuple(value for value in rows if value != removed)
        leave_losses.append(original - coverage_signature(graph, plan, reduced).value)
        replacement = _replacement_for(graph, relevance, rows, removed)
        if replacement is None:
            replacement_losses.append(original)
        else:
            swapped = tuple(replacement if value == removed else value for value in rows)
            replacement_losses.append(original - coverage_signature(graph, plan, swapped).value)
    necessary_count = sum(loss > 0 for loss in leave_losses)
    return CausalSignature(
        necessary_count=necessary_count,
        necessary_fraction=Fraction(necessary_count, len(rows)),
        minimum_leave_one_out_loss=min(leave_losses),
        minimum_replacement_loss=min(replacement_losses),
        path_connectivity=_path_connectivity(graph, plan, rows),
    )


def _e0_key(
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    output: Sequence[int],
    relevance: Sequence[int],
) -> tuple[Fraction | int, ...]:
    return (
        _metadata_coverage(graph, plan, output),
        sum(relevance[index] for index in output),
        -_redundancy(graph, output),
    )


def _json_number(value: Fraction | int) -> int | list[int]:
    return [value.numerator, value.denominator] if isinstance(value, Fraction) else value


def _action_trace_receipt_body(trace: ActionTrace) -> dict[str, object]:
    return {
        "action_id": trace.action_id,
        "causal": [
            trace.causal.necessary_count,
            _json_number(trace.causal.necessary_fraction),
            _json_number(trace.causal.minimum_leave_one_out_loss),
            _json_number(trace.causal.minimum_replacement_loss),
            _json_number(trace.causal.path_connectivity),
        ],
        "core": list(trace.core),
        "core_quality": [_json_number(value) for value in trace.core_quality],
        "coverage": [trace.coverage.covered, trace.coverage.total],
        "e0": [_json_number(value) for value in trace.e0_key],
        "e1": [_json_number(value) for value in trace.e1_key],
        "extension_scan_count": trace.extension_scan_count,
        "graph_sha256": trace.graph_sha256,
        "output_top5": list(trace.output_top5),
        "ordered_pair_scan_count": trace.ordered_pair_scan_count,
        "plan_sha256": trace.plan_sha256,
        "query_sha256": trace.query_sha256,
        "relevance_sha256": trace.relevance_sha256,
        "version": VERSION,
    }


def recompute_action_trace_sha256(trace: ActionTrace) -> str:
    """Recompute the complete action receipt for persistence/postflight checks."""

    if not isinstance(trace, ActionTrace):
        raise MultiHopRAGTypedOperatorV2Error("action trace has the wrong type")
    return _stable_hash(_action_trace_receipt_body(trace))


def run_action(
    *,
    action_id: str,
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    relevance_ints: Sequence[int],
) -> ActionTrace:
    """Run one frozen all-corpus action and compute both evaluator signatures."""

    if action_id not in ACTION_IDS:
        raise MultiHopRAGTypedOperatorV2Error("action is outside the registry")
    graph = _validated_graph(graph)
    plan = _validated_plan(graph, plan)
    relevance = _validated_relevance(relevance_ints, len(graph.articles))
    relevance_sha256 = _stable_hash({"integer_scale": INTEGER_SCALE, "values": list(relevance)})
    pair = _best_pair(action_id, graph, plan, relevance)
    core = _extend_core(action_id, graph, plan, relevance, pair)
    output = _fill_tail(action_id, graph, plan, relevance, core)
    coverage = coverage_signature(graph, plan, core)
    causal = causal_signature(graph, plan, core, relevance)
    e0 = _e0_key(graph, plan, output, relevance)
    e1 = (
        causal.necessary_fraction,
        causal.minimum_leave_one_out_loss,
        causal.path_connectivity,
        *e0,
    )
    quality = _relation_quality(action_id, graph, plan, core, relevance)
    trace = ActionTrace(
        action_id=action_id,
        output_top5=output,
        core=core,
        core_quality=quality,
        coverage=coverage,
        causal=causal,
        e0_key=e0,
        e1_key=e1,
        ordered_pair_scan_count=len(graph.articles) * (len(graph.articles) - 1),
        extension_scan_count=(len(graph.articles) - 2) + (len(graph.articles) - 3),
        graph_sha256=graph.graph_sha256,
        plan_sha256=plan.plan_sha256,
        query_sha256=plan.query_sha256,
        relevance_sha256=relevance_sha256,
        trace_sha256="0" * 64,
    )
    return replace(trace, trace_sha256=recompute_action_trace_sha256(trace))


def run_all_actions(
    *,
    graph: TypedCorpusGraph,
    plan: QueryPlan,
    relevance_ints: Sequence[int],
) -> tuple[ActionTrace, ...]:
    """Run exactly six actions.  Callers may parallelize these independent units."""

    return tuple(
        run_action(
            action_id=action_id,
            graph=graph,
            plan=plan,
            relevance_ints=relevance_ints,
        )
        for action_id in ACTION_IDS
    )


def _validated_observations(
    observations: Sequence[EvaluationObservation],
) -> tuple[EvaluationObservation, ...]:
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise MultiHopRAGTypedOperatorV2Error("observations must be a sequence")
    rows = tuple(observations)
    if not rows:
        raise MultiHopRAGTypedOperatorV2Error("observations are empty")
    for row in rows:
        if not isinstance(row, EvaluationObservation):
            raise MultiHopRAGTypedOperatorV2Error("observation has the wrong type")
        if not isinstance(row.traces_by_action, Mapping) or set(row.traces_by_action) != set(ACTION_IDS):
            raise MultiHopRAGTypedOperatorV2Error("observation action registry drifted")
        for action_id, trace in row.traces_by_action.items():
            if not isinstance(trace, ActionTrace) or trace.action_id != action_id:
                raise MultiHopRAGTypedOperatorV2Error("observation trace identity drifted")
            if any(
                not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (
                    trace.graph_sha256,
                    trace.plan_sha256,
                    trace.query_sha256,
                    trace.relevance_sha256,
                    trace.trace_sha256,
                )
            ):
                raise MultiHopRAGTypedOperatorV2Error("observation receipt drifted")
            if trace.trace_sha256 != recompute_action_trace_sha256(trace):
                raise MultiHopRAGTypedOperatorV2Error("action trace receipt does not match its content")
            expected_e1 = (
                trace.causal.necessary_fraction,
                trace.causal.minimum_leave_one_out_loss,
                trace.causal.path_connectivity,
                *trace.e0_key,
            )
            if (
                len(trace.core) != CORE_SIZE
                or len(set(trace.core)) != CORE_SIZE
                or len(trace.output_top5) != TOP_K
                or len(set(trace.output_top5)) != TOP_K
                or trace.output_top5[:CORE_SIZE] != trace.core
                or len(trace.e0_key) != 3
                or trace.e1_key != expected_e1
            ):
                raise MultiHopRAGTypedOperatorV2Error("action trace algebra drifted")
        input_receipts = {
            (
                trace.graph_sha256,
                trace.plan_sha256,
                trace.query_sha256,
                trace.relevance_sha256,
            )
            for trace in row.traces_by_action.values()
        }
        if len(input_receipts) != 1:
            raise MultiHopRAGTypedOperatorV2Error("actions do not share one observation input")
    return rows


def _to_fraction_tuple(values: Sequence[Fraction | int]) -> tuple[Fraction, ...]:
    return tuple(value if isinstance(value, Fraction) else Fraction(value) for value in values)


def _observation_input_receipt(rows: Sequence[EvaluationObservation]) -> str:
    input_receipts = [
        [row.traces_by_action[action_id].trace_sha256 for action_id in ACTION_IDS]
        for row in rows
    ]
    return _stable_hash(
        {"action_ids": list(ACTION_IDS), "observations": input_receipts, "version": VERSION}
    )


def _policy_selection_receipt_body(policy: PolicySelection) -> dict[str, object]:
    return {
        "action_id": policy.action_id,
        "evaluator_id": policy.evaluator_id,
        "input_receipt_sha256": policy.input_receipt_sha256,
        "observation_count": policy.observation_count,
        "per_action": [
            [name, [[value.numerator, value.denominator] for value in key]]
            for name, key in policy.per_action_macro_keys
        ],
        "version": VERSION,
    }


def recompute_policy_selection_sha256(policy: PolicySelection) -> str:
    if not isinstance(policy, PolicySelection):
        raise MultiHopRAGTypedOperatorV2Error("policy selection has the wrong type")
    return _stable_hash(_policy_selection_receipt_body(policy))


def select_global_policy(
    *,
    evaluator_id: str,
    observations: Sequence[EvaluationObservation],
) -> PolicySelection:
    """Select one recipe by exact mean lexicographic key on a frozen balanced block."""

    if evaluator_id not in {"E0_INDEPENDENT_V2", "E1_CAUSAL_NECESSITY_V2"}:
        raise MultiHopRAGTypedOperatorV2Error("evaluator is outside the frozen registry")
    rows = _validated_observations(observations)
    per_action: list[tuple[str, tuple[Fraction, ...]]] = []
    for action_id in ACTION_IDS:
        keys = [
            _to_fraction_tuple(
                row.traces_by_action[action_id].e0_key
                if evaluator_id == "E0_INDEPENDENT_V2"
                else row.traces_by_action[action_id].e1_key
            )
            for row in rows
        ]
        width = len(keys[0])
        macro = tuple(sum(key[index] for key in keys) / len(keys) for index in range(width))
        per_action.append((action_id, macro))
    action_id, macro_key = min(per_action, key=lambda row: (tuple(-value for value in row[1]), row[0]))
    input_receipt_sha256 = _observation_input_receipt(rows)
    policy = PolicySelection(
        evaluator_id=evaluator_id,
        action_id=action_id,
        observation_count=len(rows),
        macro_key=macro_key,
        per_action_macro_keys=tuple(per_action),
        input_receipt_sha256=input_receipt_sha256,
        selection_sha256="0" * 64,
    )
    return replace(policy, selection_sha256=recompute_policy_selection_sha256(policy))


def policies_identifiable(
    e0: PolicySelection,
    e1: PolicySelection,
    observations: Sequence[EvaluationObservation],
) -> bool:
    rows = _validated_observations(observations)
    if e0.evaluator_id != "E0_INDEPENDENT_V2" or e1.evaluator_id != "E1_CAUSAL_NECESSITY_V2":
        raise MultiHopRAGTypedOperatorV2Error("policy evaluator identities drifted")
    expected_input_receipt = _observation_input_receipt(rows)
    expected_policies = (
        select_global_policy(evaluator_id="E0_INDEPENDENT_V2", observations=rows),
        select_global_policy(evaluator_id="E1_CAUSAL_NECESSITY_V2", observations=rows),
    )
    for policy, expected_policy in zip((e0, e1), expected_policies, strict=True):
        if (
            policy.observation_count != len(rows)
            or policy.input_receipt_sha256 != expected_input_receipt
            or policy.selection_sha256 != recompute_policy_selection_sha256(policy)
            or tuple(name for name, _key in policy.per_action_macro_keys) != ACTION_IDS
        ):
            raise MultiHopRAGTypedOperatorV2Error("policy does not bind the supplied observations")
        expected_action, expected_key = min(
            policy.per_action_macro_keys,
            key=lambda row: (tuple(-value for value in row[1]), row[0]),
        )
        if policy.action_id != expected_action or policy.macro_key != expected_key:
            raise MultiHopRAGTypedOperatorV2Error("policy selection algebra drifted")
        if policy != expected_policy:
            raise MultiHopRAGTypedOperatorV2Error("policy differs from recomputation on observations")
    if e0.action_id == e1.action_id:
        return False
    return any(
        row.traces_by_action[e0.action_id].output_top5
        != row.traces_by_action[e1.action_id].output_top5
        for row in rows
    )


def item_utility(output_top5: Sequence[int], gold_article_ids: Sequence[int]) -> Fraction:
    """Late exact distinct-article recall plus complete bonus."""

    output = tuple(output_top5)
    gold = tuple(gold_article_ids)
    if len(output) != TOP_K or len(set(output)) != TOP_K or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in output
    ):
        raise MultiHopRAGTypedOperatorV2Error("output top5 is invalid")
    if not 2 <= len(gold) <= 4 or len(set(gold)) != len(gold) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gold
    ):
        raise MultiHopRAGTypedOperatorV2Error("gold article set is invalid")
    hits = len(set(output) & set(gold))
    return Fraction(hits, len(gold)) + int(hits == len(gold))


def exact_magnitude_signflip_p(deltas: Sequence[Fraction]) -> Fraction:
    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, Sequence):
        raise MultiHopRAGTypedOperatorV2Error("deltas must be a sequence")
    rows = tuple(deltas)
    if not rows or any(not isinstance(value, Fraction) for value in rows):
        raise MultiHopRAGTypedOperatorV2Error("deltas must be non-empty Fractions")
    scale = 1
    for value in rows:
        scale = math.lcm(scale, value.denominator)
    observed = sum(value.numerator * (scale // value.denominator) for value in rows)
    magnitudes = [abs(value.numerator * (scale // value.denominator)) for value in rows]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for total, count in distribution.items():
            updated[total + magnitude] += count
            updated[total - magnitude] += count
        distribution = updated
    return Fraction(
        sum(count for total, count in distribution.items() if total >= observed),
        2 ** len(rows),
    )


def paired_utility_summary(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> PairedUtilitySummary:
    left_rows = tuple(left)
    right_rows = tuple(right)
    if not left_rows or len(left_rows) != len(right_rows) or any(
        not isinstance(value, Fraction) for value in left_rows + right_rows
    ):
        raise MultiHopRAGTypedOperatorV2Error("paired utility vectors are invalid")
    deltas = tuple(a - b for a, b in zip(left_rows, right_rows, strict=True))
    return PairedUtilitySummary(
        count=len(deltas),
        left_total=sum(left_rows, Fraction(0)),
        right_total=sum(right_rows, Fraction(0)),
        delta_total=sum(deltas, Fraction(0)),
        gains=sum(value > 0 for value in deltas),
        harms=sum(value < 0 for value in deltas),
        ties=sum(value == 0 for value in deltas),
        exact_one_sided_p=exact_magnitude_signflip_p(deltas),
    )


__all__ = [
    "ACTION_IDS",
    "ArticleRecord",
    "CAPABILITIES",
    "CORE_SIZE",
    "CausalSignature",
    "CoverageSignature",
    "EDGE_FAMILIES",
    "ENTITY_TYPES",
    "EntityKey",
    "EvaluationObservation",
    "INTEGER_SCALE",
    "MultiHopRAGTypedOperatorV2Error",
    "PairedUtilitySummary",
    "PolicySelection",
    "QueryPlan",
    "TOPIC_K",
    "TOP_K",
    "TypedCorpusGraph",
    "TypedEdge",
    "VERSION",
    "build_typed_corpus_graph",
    "causal_signature",
    "compile_query_plan",
    "coverage_signature",
    "exact_magnitude_signflip_p",
    "item_utility",
    "make_entity_key",
    "normalize_text",
    "paired_utility_summary",
    "parse_date_ordinals",
    "policies_identifiable",
    "run_action",
    "run_all_actions",
    "select_global_policy",
]
