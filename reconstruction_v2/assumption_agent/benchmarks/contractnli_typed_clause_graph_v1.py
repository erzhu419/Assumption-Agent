"""Frozen, row-free ContractNLI typed-clause graph and evaluator core.

The module deliberately has no dataset, archive, model, filesystem, or network
loader.  Source spans and the frozen quantized query/span similarity tensors are
injected by the caller.  Gold evidence enters only through the isolated
formation/statistical helpers; graph construction, actions, coverage, and
F-search have narrow label-free interfaces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
import unicodedata
from typing import Mapping, Sequence


VERSION = "contractnli_typed_clause_graph_v1"
DESIGN_COMMIT = "f3bdc201"
DESIGN_SHA256 = "49f8f6dba1ddffd63dc97d1a23ced48ddd70fcc6657404a4c71b33431e6a7157"
TOP_K = 5
INTEGER_SCALE = 1_000_000
PROMOTION_ALPHA = Fraction(1, 10)

MENTIONS_DEFINITION = "MENTIONS_DEFINITION"
EXCEPTION_SCOPE = "EXCEPTION_SCOPE"
LIST_SIBLING = "LIST_SIBLING"
EXPLICIT_CROSS_REFERENCE = "EXPLICIT_CROSS_REFERENCE"
EDGE_FAMILIES = (
    MENTIONS_DEFINITION,
    EXCEPTION_SCOPE,
    LIST_SIBLING,
    EXPLICIT_CROSS_REFERENCE,
)
EDGE_FAMILY_ORDER = {family: order for order, family in enumerate(EDGE_FAMILIES)}

_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_LEXICAL_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DEFINITION = re.compile(
    r'(?:^|[.;:]\s*)(?:["“]([^"”]{1,80})["”]|'
    r"([a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,7}))\s+"
    r"(?:shall\s+mean|means|refers\s+to)\b",
    flags=re.UNICODE,
)
_DEONTIC = re.compile(
    r"\b(?:shall|must|may not|shall not|agrees? to|is required to|will not)\b",
    flags=re.UNICODE,
)
_EXCEPTION = re.compile(
    r"\b(?:except(?: that| as)?|provided(?:, however,?)?(?: that)?|"
    r"subject to|notwithstanding|unless)\b",
    flags=re.UNICODE,
)
_HEADING = re.compile(
    r"^\s*(?:section|clause|paragraph)\s+([0-9]+(?:\.[0-9]+)*|[a-z])\b",
    flags=re.UNICODE,
)
_REFERENCE = re.compile(
    r"\b(?:section|clause|paragraph)\s+([0-9]+(?:\.[0-9]+)*|[a-z])\b",
    flags=re.UNICODE,
)
_LIST_MARKER = re.compile(
    r"^\s*(?:(\((?:[a-z]|[ivxlcdm]+|[0-9]{1,3})\))|"
    r"((?:[a-z]|[ivxlcdm]+|[0-9]{1,3})[.)]))\s+",
    flags=re.UNICODE,
)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


class ContractNLITypedClauseGraphError(ValueError):
    """Raised when an input violates the frozen row-free contract."""


@dataclass(frozen=True)
class SourceSpan:
    """A source node whose identity remains the original source-list index."""

    span_i: int
    start: int
    end: int
    identity_text: str

    @property
    def embedding_text(self) -> str:
        return embedding_text(self.identity_text)

    @property
    def pattern_text(self) -> str:
        return pattern_text(self.identity_text)


@dataclass(frozen=True, order=True)
class TypedEdge:
    """One canonical undirected typed edge; parallel families are retained."""

    edge_family_order: int
    left_span_i: int
    right_span_i: int

    @property
    def edge_family(self) -> str:
        if self.edge_family_order not in range(len(EDGE_FAMILIES)):
            raise ContractNLITypedClauseGraphError("edge family order is invalid")
        return EDGE_FAMILIES[self.edge_family_order]

    def as_tuple(self) -> tuple[str, int, int]:
        return (self.edge_family, self.left_span_i, self.right_span_i)


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    budget: int
    edge_families: frozenset[str]


@dataclass(frozen=True)
class EvaluatorSpec:
    evaluator_id: str
    archetype: str
    edge_weights: tuple[Fraction, Fraction, Fraction, Fraction]
    closure_lambda: Fraction


@dataclass(frozen=True)
class CandidateRecord:
    """The exact six-field common raw candidate record."""

    edge_family_order: int
    official_seed_rank: int
    seed_span_i: int
    neighbor_span_i: int
    query_similarity_int: int
    absolute_start_offset_distance: int

    @property
    def edge_family(self) -> str:
        if self.edge_family_order not in range(len(EDGE_FAMILIES)):
            raise ContractNLITypedClauseGraphError("candidate edge family is invalid")
        return EDGE_FAMILIES[self.edge_family_order]

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.edge_family_order,
            self.official_seed_rank,
            self.seed_span_i,
            self.neighbor_span_i,
            self.query_similarity_int,
            self.absolute_start_offset_distance,
        )


@dataclass(frozen=True)
class CandidateDecision:
    record: CandidateRecord
    disposition: str
    dropped_span_i: int | None


@dataclass(frozen=True)
class ActionTrace:
    recipe_id: str
    output_top5: tuple[int, int, int, int, int]
    swap_count: int
    records_visited: int
    accepted_count: int
    rejected_count: int
    common_scan_sha256: str
    decisions: tuple[CandidateDecision, ...]


@dataclass(frozen=True)
class CoverageComponents:
    Sem: int
    Lex: int
    Diversity: int
    Churn: int
    Closure: tuple[int, int, int, int]

    def as_mapping(self) -> dict[str, object]:
        return {
            "Sem": self.Sem,
            "Lex": self.Lex,
            "Diversity": self.Diversity,
            "Churn": self.Churn,
            "Closure": {
                family: self.Closure[order]
                for order, family in enumerate(EDGE_FAMILIES)
            },
        }


@dataclass(frozen=True)
class FormationItem:
    """Late A-form input after all nine actions are terminal."""

    components_by_recipe: Mapping[str, CoverageComponents]
    utility_by_recipe: Mapping[str, int]
    complete_by_recipe: Mapping[str, bool]


@dataclass(frozen=True)
class EvaluatorFormationResult:
    evaluator_id: str
    chosen_recipe_ids: tuple[str, ...]
    item_regrets: tuple[int, ...]
    sum_regret: int
    total_true_utility: int
    total_complete: int
    total_churn: int
    coverage_comparisons: int


@dataclass(frozen=True)
class AFormationSelection:
    evaluator_id: str
    chosen_recipe_ids: tuple[str, ...]
    sum_regret: int
    total_true_utility: int
    total_complete: int
    total_churn: int
    evaluator_results: tuple[EvaluatorFormationResult, ...]


@dataclass(frozen=True)
class FSearchSelection:
    evaluator_id: str
    recipe_id: str
    total_exact_coverage: Fraction
    recipe_totals: tuple[tuple[str, Fraction], ...]
    coverage_comparisons: int


_RECIPES = (
    RecipeSpec("R0_HIPPO_TOP5", 0, frozenset()),
    RecipeSpec("R1_DEFINITION_1SWAP", 1, frozenset({MENTIONS_DEFINITION})),
    RecipeSpec("R2_EXCEPTION_1SWAP", 1, frozenset({EXCEPTION_SCOPE})),
    RecipeSpec("R3_LIST_1SWAP", 1, frozenset({LIST_SIBLING})),
    RecipeSpec(
        "R4_CROSS_REFERENCE_1SWAP",
        1,
        frozenset({EXPLICIT_CROSS_REFERENCE}),
    ),
    RecipeSpec(
        "R5_DEFINITION_EXCEPTION_2SWAP",
        2,
        frozenset({MENTIONS_DEFINITION, EXCEPTION_SCOPE}),
    ),
    RecipeSpec(
        "R6_DEFINITION_LIST_2SWAP",
        2,
        frozenset({MENTIONS_DEFINITION, LIST_SIBLING}),
    ),
    RecipeSpec(
        "R7_EXCEPTION_LIST_2SWAP",
        2,
        frozenset({EXCEPTION_SCOPE, LIST_SIBLING}),
    ),
    RecipeSpec("R8_ALL_TYPED_2SWAP", 2, frozenset(EDGE_FAMILIES)),
)
_RECIPE_BY_ID = {recipe.recipe_id: recipe for recipe in _RECIPES}

_ARCHETYPE_WEIGHTS = {
    "DEF_HEAVY": (
        Fraction(11, 20),
        Fraction(3, 20),
        Fraction(3, 20),
        Fraction(3, 20),
    ),
    "EXCEPTION_HEAVY": (
        Fraction(3, 20),
        Fraction(11, 20),
        Fraction(3, 20),
        Fraction(3, 20),
    ),
    "LIST_XREF_HEAVY": (
        Fraction(1, 10),
        Fraction(1, 10),
        Fraction(2, 5),
        Fraction(2, 5),
    ),
    "UNIFORM": (
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 4),
    ),
}
_LAMBDA_BY_ID = {
    "L025": Fraction(1, 4),
    "L050": Fraction(1, 2),
    "L100": Fraction(1, 1),
    "L200": Fraction(2, 1),
}
_EVALUATORS = tuple(
    EvaluatorSpec(
        evaluator_id=f"E_{archetype}_{lambda_id}",
        archetype=archetype,
        edge_weights=weights,
        closure_lambda=closure_lambda,
    )
    for archetype, weights in _ARCHETYPE_WEIGHTS.items()
    for lambda_id, closure_lambda in _LAMBDA_BY_ID.items()
)
_EVALUATOR_BY_ID = {evaluator.evaluator_id: evaluator for evaluator in _EVALUATORS}


def recipe_registry() -> tuple[RecipeSpec, ...]:
    return _RECIPES


def evaluator_registry() -> tuple[EvaluatorSpec, ...]:
    return _EVALUATORS


def embedding_text(identity_text: str) -> str:
    """Collapse whitespace while preserving case and all non-whitespace codepoints."""

    if not isinstance(identity_text, str):
        raise ContractNLITypedClauseGraphError("identity text must be text")
    return _WHITESPACE.sub(" ", identity_text).strip()


def pattern_text(identity_text: str) -> str:
    """Apply the frozen NFKC, casefold, and whitespace-only pattern transform."""

    if not isinstance(identity_text, str):
        raise ContractNLITypedClauseGraphError("identity text must be text")
    normalized = unicodedata.normalize("NFKC", identity_text).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def unicode_tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ContractNLITypedClauseGraphError("token input must be text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _LEXICAL_TOKEN.finditer(normalized))


def _validated_spans(spans: Sequence[SourceSpan]) -> tuple[SourceSpan, ...]:
    if isinstance(spans, (str, bytes)):
        raise ContractNLITypedClauseGraphError("source spans must be a sequence")
    normalized = tuple(spans)
    for expected_i, span in enumerate(normalized):
        if not isinstance(span, SourceSpan):
            raise ContractNLITypedClauseGraphError("source span has the wrong type")
        if type(span.span_i) is not int or span.span_i != expected_i:
            raise ContractNLITypedClauseGraphError(
                "source span identities must be contiguous source-list indices"
            )
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or span.start < 0
            or span.end <= span.start
        ):
            raise ContractNLITypedClauseGraphError("source span offsets are invalid")
        if not isinstance(span.identity_text, str):
            raise ContractNLITypedClauseGraphError("source identity text must be text")
    return normalized


def _edge(order: int, left: int, right: int) -> TypedEdge | None:
    if left == right:
        return None
    return TypedEdge(order, min(left, right), max(left, right))


def _definition_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for match in _DEFINITION.finditer(text):
        captured = match.group(1) or match.group(2) or ""
        term = pattern_text(captured)
        tokens = unicode_tokens(term)
        if not 1 <= len(tokens) <= 8 or all(token in STOPWORDS for token in tokens):
            continue
        if term not in terms:
            terms.append(term)
    return tuple(terms)


def build_typed_clause_graph(spans: Sequence[SourceSpan]) -> tuple[TypedEdge, ...]:
    """Build all four frozen regex edge families without model or label access."""

    source_spans = _validated_spans(spans)
    texts = tuple(span.pattern_text for span in source_spans)
    edges: set[TypedEdge] = set()

    for definition_i, text in enumerate(texts):
        for term in _definition_terms(text):
            mention = re.compile(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                flags=re.UNICODE,
            )
            for mention_i, candidate_text in enumerate(texts):
                if mention_i != definition_i and mention.search(candidate_text) is not None:
                    edge = _edge(EDGE_FAMILY_ORDER[MENTIONS_DEFINITION], definition_i, mention_i)
                    if edge is not None:
                        edges.add(edge)

    deontic_indices = {
        span_i for span_i, text in enumerate(texts) if _DEONTIC.search(text) is not None
    }
    for exception_i, text in enumerate(texts):
        if _EXCEPTION.search(text) is None:
            continue
        preceding = [
            span_i
            for span_i in range(max(0, exception_i - 8), exception_i)
            if span_i in deontic_indices
        ]
        if preceding:
            edge = _edge(
                EDGE_FAMILY_ORDER[EXCEPTION_SCOPE],
                exception_i,
                preceding[-1],
            )
            if edge is not None:
                edges.add(edge)

    headings: dict[str, list[int]] = {}
    heading_indices: set[int] = set()
    for span_i, text in enumerate(texts):
        match = _HEADING.search(text)
        if match is not None:
            anchor = match.group(1).casefold()
            headings.setdefault(anchor, []).append(span_i)
            heading_indices.add(span_i)
    for reference_i, text in enumerate(texts):
        if reference_i in heading_indices:
            continue
        anchors = {match.group(1).casefold() for match in _REFERENCE.finditer(text)}
        for anchor in sorted(anchors):
            for heading_i in headings.get(anchor, ()):  # all matching headings
                edge = _edge(
                    EDGE_FAMILY_ORDER[EXPLICIT_CROSS_REFERENCE],
                    reference_i,
                    heading_i,
                )
                if edge is not None:
                    edges.add(edge)

    marker_families: list[str | None] = []
    for text in texts:
        match = _LIST_MARKER.search(text)
        marker_families.append(
            "parenthesized" if match is not None and match.group(1) is not None
            else "suffix" if match is not None and match.group(2) is not None
            else None
        )
    for left_i in range(len(source_spans) - 1):
        family = marker_families[left_i]
        if family is not None and family == marker_families[left_i + 1]:
            edge = _edge(EDGE_FAMILY_ORDER[LIST_SIBLING], left_i, left_i + 1)
            if edge is not None:
                edges.add(edge)

    return tuple(sorted(edges))


def _validated_integer_vector(
    values: Sequence[int], expected_length: int, field: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ContractNLITypedClauseGraphError(f"{field} must be an integer vector")
    normalized = tuple(values)
    if len(normalized) != expected_length or any(type(value) is not int for value in normalized):
        raise ContractNLITypedClauseGraphError(f"{field} must be a length-N integer vector")
    return normalized


def _validated_top5(
    top5: Sequence[int], source_count: int, field: str = "top5"
) -> tuple[int, int, int, int, int]:
    if isinstance(top5, (str, bytes)):
        raise ContractNLITypedClauseGraphError(f"{field} is malformed")
    normalized = tuple(top5)
    if (
        len(normalized) != TOP_K
        or len(set(normalized)) != TOP_K
        or any(type(value) is not int or not 0 <= value < source_count for value in normalized)
    ):
        raise ContractNLITypedClauseGraphError(
            f"{field} must contain five unique bounded source indices"
        )
    return normalized  # type: ignore[return-value]


def _validated_edges(
    edges: Sequence[TypedEdge], source_count: int
) -> tuple[TypedEdge, ...]:
    if isinstance(edges, (str, bytes)):
        raise ContractNLITypedClauseGraphError("typed edges must be a sequence")
    normalized = tuple(edges)
    seen: set[tuple[int, int, int]] = set()
    for edge in normalized:
        if not isinstance(edge, TypedEdge):
            raise ContractNLITypedClauseGraphError("typed edge has the wrong type")
        key = (edge.edge_family_order, edge.left_span_i, edge.right_span_i)
        if (
            edge.edge_family_order not in range(len(EDGE_FAMILIES))
            or not 0 <= edge.left_span_i < edge.right_span_i < source_count
            or key in seen
        ):
            raise ContractNLITypedClauseGraphError(
                "typed edges must be unique canonical bounded pairs"
            )
        seen.add(key)
    return tuple(sorted(normalized))


def _candidate_sort_key(record: CandidateRecord) -> tuple[int, int, int, int, int]:
    return (
        -record.query_similarity_int,
        record.absolute_start_offset_distance,
        record.neighbor_span_i,
        record.official_seed_rank,
        record.edge_family_order,
    )


def build_common_candidate_table(
    spans: Sequence[SourceSpan],
    typed_edges: Sequence[TypedEdge],
    official_top5: Sequence[int],
    query_span_similarities: Sequence[int],
) -> tuple[CandidateRecord, ...]:
    """Build one shared raw record for each family/seed/neighbor triple."""

    source_spans = _validated_spans(spans)
    similarities = _validated_integer_vector(
        query_span_similarities, len(source_spans), "query-span similarities"
    )
    top5 = _validated_top5(official_top5, len(source_spans), "official Hippo top5")
    edges = _validated_edges(typed_edges, len(source_spans))
    seed_rank = {span_i: rank for rank, span_i in enumerate(top5, 1)}
    records: list[CandidateRecord] = []
    seen: set[tuple[int, int, int]] = set()
    for edge in edges:
        for seed_i, neighbor_i in (
            (edge.left_span_i, edge.right_span_i),
            (edge.right_span_i, edge.left_span_i),
        ):
            if seed_i not in seed_rank:
                continue
            triple = (edge.edge_family_order, seed_i, neighbor_i)
            if triple in seen:
                raise ContractNLITypedClauseGraphError("duplicate candidate triple")
            seen.add(triple)
            records.append(
                CandidateRecord(
                    edge_family_order=edge.edge_family_order,
                    official_seed_rank=seed_rank[seed_i],
                    seed_span_i=seed_i,
                    neighbor_span_i=neighbor_i,
                    query_similarity_int=similarities[neighbor_i],
                    absolute_start_offset_distance=abs(
                        source_spans[seed_i].start - source_spans[neighbor_i].start
                    ),
                )
            )
    return tuple(sorted(records, key=_candidate_sort_key))


def _scan_sha256(records: Sequence[CandidateRecord]) -> str:
    payload = json.dumps(
        [record.as_tuple() for record in records],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_candidate_table(
    records: Sequence[CandidateRecord],
    top5: tuple[int, int, int, int, int],
    similarities: tuple[int, ...],
) -> tuple[CandidateRecord, ...]:
    if isinstance(records, (str, bytes)):
        raise ContractNLITypedClauseGraphError("candidate table must be a sequence")
    normalized = tuple(records)
    if any(not isinstance(record, CandidateRecord) for record in normalized):
        raise ContractNLITypedClauseGraphError("candidate record has the wrong type")
    if tuple(sorted(normalized, key=_candidate_sort_key)) != normalized:
        raise ContractNLITypedClauseGraphError("candidate table is not in common sort order")
    rank = {span_i: position for position, span_i in enumerate(top5, 1)}
    seen: set[tuple[int, int, int]] = set()
    for record in normalized:
        triple = (record.edge_family_order, record.seed_span_i, record.neighbor_span_i)
        if (
            record.edge_family_order not in range(len(EDGE_FAMILIES))
            or record.seed_span_i not in rank
            or record.official_seed_rank != rank[record.seed_span_i]
            or not 0 <= record.neighbor_span_i < len(similarities)
            or record.neighbor_span_i == record.seed_span_i
            or type(record.query_similarity_int) is not int
            or record.query_similarity_int != similarities[record.neighbor_span_i]
            or type(record.absolute_start_offset_distance) is not int
            or record.absolute_start_offset_distance < 0
            or triple in seen
        ):
            raise ContractNLITypedClauseGraphError("candidate record violates the frozen schema")
        seen.add(triple)
    return normalized


def execute_recipe(
    official_top5: Sequence[int],
    common_candidate_table: Sequence[CandidateRecord],
    query_span_similarities: Sequence[int],
    recipe_id: str,
) -> ActionTrace:
    """Execute one recipe while always visiting the complete shared table."""

    if not isinstance(recipe_id, str) or recipe_id not in _RECIPE_BY_ID:
        raise ContractNLITypedClauseGraphError("recipe ID is not in the frozen registry")
    similarities = _validated_integer_vector(
        query_span_similarities, len(query_span_similarities), "query-span similarities"
    )
    top5 = _validated_top5(official_top5, len(similarities), "official Hippo top5")
    records = _validated_candidate_table(common_candidate_table, top5, similarities)
    recipe = _RECIPE_BY_ID[recipe_id]
    official_rank = {span_i: rank for rank, span_i in enumerate(top5, 1)}
    retained = set(top5)
    added: list[int] = []
    accepted_neighbors: set[int] = set()
    protected_origins: set[int] = set()
    decisions: list[CandidateDecision] = []

    for record in records:
        disposition: str
        dropped: int | None = None
        if record.edge_family not in recipe.edge_families:
            disposition = "edge_family_masked"
        elif len(added) >= recipe.budget:
            disposition = "budget_exhausted"
        elif record.seed_span_i not in retained:
            disposition = "origin_seed_not_selected"
        elif record.neighbor_span_i in retained or record.neighbor_span_i in accepted_neighbors:
            disposition = "neighbor_already_selected"
        else:
            protected = protected_origins | {record.seed_span_i}
            drop_candidates = [
                span_i
                for span_i in retained
                if span_i in official_rank
                and span_i not in accepted_neighbors
                and span_i not in protected
            ]
            if not drop_candidates:
                disposition = "no_drop_candidate"
            else:
                dropped = min(
                    drop_candidates,
                    key=lambda span_i: (
                        similarities[span_i],
                        -official_rank[span_i],
                        -span_i,
                    ),
                )
                retained.remove(dropped)
                retained.add(record.neighbor_span_i)
                added.append(record.neighbor_span_i)
                accepted_neighbors.add(record.neighbor_span_i)
                protected_origins.add(record.seed_span_i)
                disposition = "accepted"
        decisions.append(CandidateDecision(record, disposition, dropped))

    retained_official = [span_i for span_i in top5 if span_i in retained and span_i not in added]
    output = tuple(retained_official + added)
    if len(output) != TOP_K or len(set(output)) != TOP_K or set(output) != retained:
        raise ContractNLITypedClauseGraphError("recipe failed to return five unique source indices")
    accepted_count = sum(decision.disposition == "accepted" for decision in decisions)
    swap_count = TOP_K - len(set(output).intersection(top5))
    return ActionTrace(
        recipe_id=recipe_id,
        output_top5=output,  # type: ignore[arg-type]
        swap_count=swap_count,
        records_visited=len(decisions),
        accepted_count=accepted_count,
        rejected_count=len(decisions) - accepted_count,
        common_scan_sha256=_scan_sha256(records),
        decisions=tuple(decisions),
    )


def execute_all_recipes(
    official_top5: Sequence[int],
    common_candidate_table: Sequence[CandidateRecord],
    query_span_similarities: Sequence[int],
) -> tuple[ActionTrace, ...]:
    """Execute all nine recipes against one exact common candidate table."""

    return tuple(
        execute_recipe(
            official_top5,
            common_candidate_table,
            query_span_similarities,
            recipe.recipe_id,
        )
        for recipe in _RECIPES
    )


def _validated_similarity_matrix(
    values: Sequence[Sequence[int]], source_count: int
) -> tuple[tuple[int, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise ContractNLITypedClauseGraphError("span-span similarities must be a matrix")
    rows = tuple(tuple(row) for row in values)
    if len(rows) != source_count:
        raise ContractNLITypedClauseGraphError("span-span similarity matrix must be N by N")
    for row in rows:
        if len(row) != source_count or any(
            type(value) is not int
            or not -INTEGER_SCALE <= value <= INTEGER_SCALE
            for value in row
        ):
            raise ContractNLITypedClauseGraphError("span-span similarity matrix must be integer N by N")
    for left in range(source_count):
        for right in range(left + 1, source_count):
            if rows[left][right] != rows[right][left]:
                raise ContractNLITypedClauseGraphError("span-span similarity matrix must be symmetric")
    return rows


def semantic_coverage(
    candidate_top5: Sequence[int], query_span_similarities: Sequence[int]
) -> int:
    similarities = _validated_integer_vector(
        query_span_similarities, len(query_span_similarities), "query-span similarities"
    )
    candidate = _validated_top5(candidate_top5, len(similarities), "candidate top5")
    values = sorted(
        (
            (min(INTEGER_SCALE, max(-INTEGER_SCALE, similarities[span_i])) + INTEGER_SCALE)
            // 2
            for span_i in candidate
        ),
        reverse=True,
    )
    return (3 * values[0] + values[1]) // 4


def lexical_coverage(
    query_text: str, spans: Sequence[SourceSpan], candidate_top5: Sequence[int]
) -> int:
    source_spans = _validated_spans(spans)
    candidate = _validated_top5(candidate_top5, len(source_spans), "candidate top5")
    if not isinstance(query_text, str):
        raise ContractNLITypedClauseGraphError("query text must be text")
    span_token_sets = tuple(set(unicode_tokens(span.identity_text)) for span in source_spans)
    document_frequency: Counter[str] = Counter()
    for tokens in span_token_sets:
        document_frequency.update(tokens)
    query_tokens = set(unicode_tokens(query_text))
    usable = {
        token
        for token in query_tokens
        if token not in STOPWORDS
        and not token.isnumeric()
        and len(token) >= 2
        and document_frequency[token] <= len(source_spans) // 2
    }
    if not usable:
        return 0
    weights = {
        token: len(source_spans) + 1 - document_frequency[token] for token in usable
    }
    covered = set().union(*(span_token_sets[span_i] for span_i in candidate))
    numerator = sum(weight for token, weight in weights.items() if token in covered)
    return INTEGER_SCALE * numerator // sum(weights.values())


def coverage_components(
    query_text: str,
    spans: Sequence[SourceSpan],
    candidate_top5: Sequence[int],
    official_top5: Sequence[int],
    typed_edges: Sequence[TypedEdge],
    query_span_similarities: Sequence[int],
    span_span_similarities: Sequence[Sequence[int]],
) -> CoverageComponents:
    """Compute the five exact integer coverage components for one candidate."""

    source_spans = _validated_spans(spans)
    candidate = _validated_top5(candidate_top5, len(source_spans), "candidate top5")
    official = _validated_top5(official_top5, len(source_spans), "official Hippo top5")
    similarities = _validated_integer_vector(
        query_span_similarities, len(source_spans), "query-span similarities"
    )
    matrix = _validated_similarity_matrix(span_span_similarities, len(source_spans))
    edges = _validated_edges(typed_edges, len(source_spans))

    sem = semantic_coverage(candidate, similarities)
    lex = lexical_coverage(query_text, source_spans, candidate)
    pair_sum = sum(
        max(0, matrix[candidate[left]][candidate[right]])
        for left in range(TOP_K)
        for right in range(left + 1, TOP_K)
    )
    diversity = INTEGER_SCALE - pair_sum // 10
    churn = 200_000 * (TOP_K - len(set(candidate).intersection(official)))

    official_set = set(official)
    candidate_set = set(candidate)
    closures: list[int] = []
    for order in range(len(EDGE_FAMILIES)):
        demand = [
            edge
            for edge in edges
            if edge.edge_family_order == order
            and (edge.left_span_i in official_set or edge.right_span_i in official_set)
        ]
        closed = sum(
            edge.left_span_i in candidate_set and edge.right_span_i in candidate_set
            for edge in demand
        )
        closures.append(INTEGER_SCALE * closed // len(demand) if demand else 0)
    return CoverageComponents(sem, lex, diversity, churn, tuple(closures))  # type: ignore[arg-type]


def _validated_components(components: CoverageComponents) -> CoverageComponents:
    if not isinstance(components, CoverageComponents):
        raise ContractNLITypedClauseGraphError("coverage components have the wrong type")
    scalar_values = (components.Sem, components.Lex, components.Diversity, components.Churn)
    if any(type(value) is not int for value in scalar_values):
        raise ContractNLITypedClauseGraphError("coverage components must be integers")
    if (
        not 0 <= components.Sem <= INTEGER_SCALE
        or not 0 <= components.Lex <= INTEGER_SCALE
        or not 0 <= components.Diversity <= INTEGER_SCALE
        or not 0 <= components.Churn <= INTEGER_SCALE
        or components.Churn % 200_000
    ):
        raise ContractNLITypedClauseGraphError("coverage component is out of range")
    if (
        len(components.Closure) != len(EDGE_FAMILIES)
        or any(
            type(value) is not int or not 0 <= value <= INTEGER_SCALE
            for value in components.Closure
        )
    ):
        raise ContractNLITypedClauseGraphError("closure vector is invalid")
    return components


def score_coverage(components: CoverageComponents, evaluator_id: str) -> Fraction:
    """Score one component row with one of the sixteen exact Fraction evaluators."""

    components = _validated_components(components)
    if not isinstance(evaluator_id, str) or evaluator_id not in _EVALUATOR_BY_ID:
        raise ContractNLITypedClauseGraphError("evaluator ID is not in the frozen registry")
    evaluator = _EVALUATOR_BY_ID[evaluator_id]
    weighted_closure = sum(
        weight * closure
        for weight, closure in zip(evaluator.edge_weights, components.Closure)
    )
    return (
        Fraction(2, 5) * components.Sem
        + Fraction(1, 5) * components.Lex
        + Fraction(1, 10) * components.Diversity
        - Fraction(1, 20) * components.Churn
        + Fraction(3, 10) * evaluator.closure_lambda * weighted_closure
    )


def score_all_evaluators(
    components: CoverageComponents,
) -> tuple[tuple[str, Fraction], ...]:
    return tuple(
        (evaluator.evaluator_id, score_coverage(components, evaluator.evaluator_id))
        for evaluator in _EVALUATORS
    )


def _validate_recipe_mapping(mapping: Mapping[str, object], field: str) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != set(_RECIPE_BY_ID):
        raise ContractNLITypedClauseGraphError(
            f"{field} must contain the complete nine-recipe registry"
        )


def _validated_formation_item(item: FormationItem) -> FormationItem:
    if not isinstance(item, FormationItem):
        raise ContractNLITypedClauseGraphError("formation item has the wrong type")
    _validate_recipe_mapping(item.components_by_recipe, "component table")
    _validate_recipe_mapping(item.utility_by_recipe, "utility table")
    _validate_recipe_mapping(item.complete_by_recipe, "complete table")
    for recipe in _RECIPES:
        _validated_components(item.components_by_recipe[recipe.recipe_id])
        utility = item.utility_by_recipe[recipe.recipe_id]
        complete = item.complete_by_recipe[recipe.recipe_id]
        if type(utility) is not int or utility < 0 or type(complete) is not bool:
            raise ContractNLITypedClauseGraphError("formation outcomes are malformed")
    return item


def select_a_evaluator(items: Sequence[FormationItem]) -> AFormationSelection:
    """Perform the frozen late-label A-form regret selection over all 16 evaluators."""

    if isinstance(items, (str, bytes)) or not items:
        raise ContractNLITypedClauseGraphError("A-form items must be nonempty")
    normalized = tuple(_validated_formation_item(item) for item in items)
    evaluator_results: list[EvaluatorFormationResult] = []
    for evaluator in _EVALUATORS:
        chosen: list[str] = []
        regrets: list[int] = []
        total_utility = 0
        total_complete = 0
        total_churn = 0
        comparisons = 0
        for item in normalized:
            scored: list[tuple[Fraction, int, str]] = []
            for recipe in _RECIPES:
                components = item.components_by_recipe[recipe.recipe_id]
                scored.append(
                    (
                        score_coverage(components, evaluator.evaluator_id),
                        components.Churn,
                        recipe.recipe_id,
                    )
                )
                comparisons += 1
            selected = min(scored, key=lambda row: (-row[0], row[1], row[2]))
            recipe_id = selected[2]
            utility = item.utility_by_recipe[recipe_id]
            chosen.append(recipe_id)
            regrets.append(max(item.utility_by_recipe.values()) - utility)
            total_utility += utility
            total_complete += int(item.complete_by_recipe[recipe_id])
            total_churn += item.components_by_recipe[recipe_id].Churn
        evaluator_results.append(
            EvaluatorFormationResult(
                evaluator_id=evaluator.evaluator_id,
                chosen_recipe_ids=tuple(chosen),
                item_regrets=tuple(regrets),
                sum_regret=sum(regrets),
                total_true_utility=total_utility,
                total_complete=total_complete,
                total_churn=total_churn,
                coverage_comparisons=comparisons,
            )
        )
    selected = min(
        evaluator_results,
        key=lambda result: (
            result.sum_regret,
            -result.total_true_utility,
            -result.total_complete,
            result.total_churn,
            result.evaluator_id,
        ),
    )
    return AFormationSelection(
        evaluator_id=selected.evaluator_id,
        chosen_recipe_ids=selected.chosen_recipe_ids,
        sum_regret=selected.sum_regret,
        total_true_utility=selected.total_true_utility,
        total_complete=selected.total_complete,
        total_churn=selected.total_churn,
        evaluator_results=tuple(evaluator_results),
    )


def select_f_recipe(
    component_tables: Sequence[Mapping[str, CoverageComponents]], evaluator_id: str
) -> FSearchSelection:
    """Select the frozen F recipe from label-free component tables only."""

    if isinstance(component_tables, (str, bytes)) or not component_tables:
        raise ContractNLITypedClauseGraphError("F component tables must be nonempty")
    if not isinstance(evaluator_id, str) or evaluator_id not in _EVALUATOR_BY_ID:
        raise ContractNLITypedClauseGraphError("evaluator ID is not in the frozen registry")
    normalized: list[Mapping[str, CoverageComponents]] = []
    for table in component_tables:
        _validate_recipe_mapping(table, "F component table")
        for recipe in _RECIPES:
            _validated_components(table[recipe.recipe_id])
        normalized.append(table)
    totals: list[tuple[str, Fraction]] = []
    comparisons = 0
    for recipe in _RECIPES:
        total = Fraction(0, 1)
        for table in normalized:
            total += score_coverage(table[recipe.recipe_id], evaluator_id)
            comparisons += 1
        totals.append((recipe.recipe_id, total))
    selected_id, selected_total = min(
        totals,
        key=lambda row: (-row[1], _RECIPE_BY_ID[row[0]].budget, row[0]),
    )
    return FSearchSelection(
        evaluator_id=evaluator_id,
        recipe_id=selected_id,
        total_exact_coverage=selected_total,
        recipe_totals=tuple(totals),
        coverage_comparisons=comparisons,
    )


def has_identifiable_transition(
    selected_recipe_id: str,
    selected_outputs: Sequence[Sequence[int]],
    official_outputs: Sequence[Sequence[int]],
) -> bool:
    """Return false for R0 or all-item membership identity, without a runner-up."""

    if selected_recipe_id not in _RECIPE_BY_ID:
        raise ContractNLITypedClauseGraphError("recipe ID is not in the frozen registry")
    if len(selected_outputs) != len(official_outputs) or not selected_outputs:
        raise ContractNLITypedClauseGraphError("transition output tables are malformed")
    if selected_recipe_id == "R0_HIPPO_TOP5":
        return False
    for selected, official in zip(selected_outputs, official_outputs):
        source_count = max((*selected, *official), default=-1) + 1
        selected_top5 = _validated_top5(selected, source_count, "selected output")
        official_top5 = _validated_top5(official, source_count, "official output")
        if set(selected_top5) != set(official_top5):
            return True
    return False


def item_utility(
    top5: Sequence[int], gold_span_indices: Sequence[int], *, source_count: int
) -> tuple[int, int, int]:
    """Compute hits, complete, and U inside the isolated late gold controller."""

    if type(source_count) is not int or source_count < TOP_K:
        raise ContractNLITypedClauseGraphError("source count is invalid")
    selected = _validated_top5(top5, source_count)
    if isinstance(gold_span_indices, (str, bytes)):
        raise ContractNLITypedClauseGraphError("gold span indices are malformed")
    gold = tuple(gold_span_indices)
    if (
        not gold
        or tuple(sorted(gold)) != gold
        or len(set(gold)) != len(gold)
        or any(type(value) is not int or not 0 <= value < source_count for value in gold)
    ):
        raise ContractNLITypedClauseGraphError(
            "gold span indices must be sorted, distinct, and bounded"
        )
    selected_set = set(selected)
    hits = len(selected_set.intersection(gold))
    complete = int(set(gold).issubset(selected_set))
    return hits, complete, hits + complete


def exact_magnitude_preserving_sign_flip(deltas: Sequence[int]) -> dict[str, object]:
    """Compute the frozen one-sided exact sign-flip test without Monte Carlo."""

    if isinstance(deltas, (str, bytes)) or not deltas or any(type(value) is not int for value in deltas):
        raise ContractNLITypedClauseGraphError("paired U deltas are malformed")
    observed = sum(deltas)
    magnitudes = [abs(value) for value in deltas if value]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    p_value = Fraction(
        sum(count for subtotal, count in distribution.items() if subtotal >= observed),
        1 << len(magnitudes),
    )
    positive = observed > 0
    exact = p_value <= PROMOTION_ALPHA
    return {
        "test": "one_sided_exact_magnitude_preserving_sign_flip_v1",
        "observed_net_U": observed,
        "nonzero_pair_count": len(magnitudes),
        "p_value_numerator": p_value.numerator,
        "p_value_denominator": p_value.denominator,
        "p_value": float(p_value),
        "alpha_numerator": PROMOTION_ALPHA.numerator,
        "alpha_denominator": PROMOTION_ALPHA.denominator,
        "positive_observed_net": positive,
        "exact_p_at_or_below_alpha": exact,
        "promoted": positive and exact,
        "sole_promotion_criterion": True,
    }


__all__ = [
    "AFormationSelection",
    "ActionTrace",
    "CandidateDecision",
    "CandidateRecord",
    "ContractNLITypedClauseGraphError",
    "CoverageComponents",
    "DESIGN_COMMIT",
    "DESIGN_SHA256",
    "EDGE_FAMILIES",
    "EDGE_FAMILY_ORDER",
    "EvaluatorFormationResult",
    "EvaluatorSpec",
    "EXCEPTION_SCOPE",
    "EXPLICIT_CROSS_REFERENCE",
    "FSearchSelection",
    "FormationItem",
    "INTEGER_SCALE",
    "LIST_SIBLING",
    "MENTIONS_DEFINITION",
    "PROMOTION_ALPHA",
    "RecipeSpec",
    "STOPWORDS",
    "SourceSpan",
    "TOP_K",
    "TypedEdge",
    "VERSION",
    "build_common_candidate_table",
    "build_typed_clause_graph",
    "coverage_components",
    "embedding_text",
    "evaluator_registry",
    "exact_magnitude_preserving_sign_flip",
    "execute_all_recipes",
    "execute_recipe",
    "has_identifiable_transition",
    "item_utility",
    "lexical_coverage",
    "pattern_text",
    "recipe_registry",
    "score_all_evaluators",
    "score_coverage",
    "select_a_evaluator",
    "select_f_recipe",
    "semantic_coverage",
    "unicode_tokens",
]
