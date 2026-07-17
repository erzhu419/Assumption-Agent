"""Frozen row-free scientific typed graph and evaluator algebra for EvidenceBench.

This module is deliberately source agnostic.  It has no dataset, filesystem,
network, or model loader.  Graph construction receives exactly 32 ordered
source nodes and inspects only their identity text.  The query is used only by
the separately injected coverage/action algebra.  Gold aspect evidence enters
only through :func:`item_utility` after every label-free action is terminal.
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


VERSION = "evidencebench_typed_scientific_graph_v1"
SOURCE_NODE_COUNT = 32
TOP_K = 5
INTEGER_SCALE = 1_000_000
UTILITY_RECALL_SCALE = 1_000
UTILITY_COMPLETE_BONUS = 1_000
PROMOTION_ALPHA = Fraction(1, 10)

ADJACENT_BUCKET = "ADJACENT_BUCKET"
ABBREVIATION_DEFINITION = "ABBREVIATION_DEFINITION"
EXPLICIT_SCIENTIFIC_XREF = "EXPLICIT_SCIENTIFIC_XREF"
RARE_ENTITY_BRIDGE = "RARE_ENTITY_BRIDGE"
EDGE_FAMILIES = (
    ADJACENT_BUCKET,
    ABBREVIATION_DEFINITION,
    EXPLICIT_SCIENTIFIC_XREF,
    RARE_ENTITY_BRIDGE,
)
EDGE_FAMILY_ORDER = {family: order for order, family in enumerate(EDGE_FAMILIES)}

_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)
_LEXICAL_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_PARENTHETICAL = re.compile(r"\(([^()]{2,24})\)", flags=re.UNICODE)
_LONG_FORM_TOKEN = re.compile(r"[^\W_]+(?:[-/][^\W_]+)*", flags=re.UNICODE)
_SCIENTIFIC_XREF = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<kind>fig(?:ure)?s?|tab(?:le)?s?|eq(?:uation)?s?)\.?\s*"
    r"(?:\(\s*)?(?P<label>[sS]?\d{1,4}[A-Za-z]?)(?:\s*\))?"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE | re.UNICODE,
)
_ENTITY_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*"
    r"(?![A-Za-z0-9_])",
    flags=re.UNICODE,
)

# This is a language-level stop list, not a scientific or biomedical keyword
# list.  It is frozen and used only by lexical coverage and surface filtering.
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


class EvidenceBenchTypedScientificGraphError(ValueError):
    """Raised when an input violates the frozen row-free contract."""


@dataclass(frozen=True)
class SourceNode:
    """One ordered item-local source node with its source-list identity."""

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
            raise EvidenceBenchTypedScientificGraphError("edge family order is invalid")
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
            raise EvidenceBenchTypedScientificGraphError(
                "candidate edge family is invalid"
            )
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
    """Late formation input after all nine label-free actions are terminal."""

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
    RecipeSpec("R1_ADJACENT_1SWAP", 1, frozenset({ADJACENT_BUCKET})),
    RecipeSpec(
        "R2_ABBREVIATION_1SWAP", 1, frozenset({ABBREVIATION_DEFINITION})
    ),
    RecipeSpec("R3_XREF_1SWAP", 1, frozenset({EXPLICIT_SCIENTIFIC_XREF})),
    RecipeSpec("R4_RARE_ENTITY_1SWAP", 1, frozenset({RARE_ENTITY_BRIDGE})),
    RecipeSpec(
        "R5_ADJACENT_ABBREVIATION_2SWAP",
        2,
        frozenset({ADJACENT_BUCKET, ABBREVIATION_DEFINITION}),
    ),
    RecipeSpec(
        "R6_ADJACENT_XREF_2SWAP",
        2,
        frozenset({ADJACENT_BUCKET, EXPLICIT_SCIENTIFIC_XREF}),
    ),
    RecipeSpec(
        "R7_ABBREVIATION_XREF_2SWAP",
        2,
        frozenset({ABBREVIATION_DEFINITION, EXPLICIT_SCIENTIFIC_XREF}),
    ),
    RecipeSpec("R8_ALL_TYPED_2SWAP", 2, frozenset(EDGE_FAMILIES)),
)
_RECIPE_BY_ID = {recipe.recipe_id: recipe for recipe in _RECIPES}

_ARCHETYPE_WEIGHTS = {
    "ADJACENCY_HEAVY": (
        Fraction(11, 20),
        Fraction(3, 20),
        Fraction(3, 20),
        Fraction(3, 20),
    ),
    "ABBREVIATION_HEAVY": (
        Fraction(3, 20),
        Fraction(11, 20),
        Fraction(3, 20),
        Fraction(3, 20),
    ),
    "XREF_ENTITY_HEAVY": (
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
    if not isinstance(identity_text, str):
        raise EvidenceBenchTypedScientificGraphError("identity text must be text")
    return _WHITESPACE.sub(" ", identity_text).strip()


def pattern_text(identity_text: str) -> str:
    if not isinstance(identity_text, str):
        raise EvidenceBenchTypedScientificGraphError("identity text must be text")
    normalized = unicodedata.normalize("NFKC", identity_text).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def unicode_tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise EvidenceBenchTypedScientificGraphError("token input must be text")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _LEXICAL_TOKEN.finditer(normalized))


def _validated_nodes(nodes: Sequence[SourceNode]) -> tuple[SourceNode, ...]:
    if isinstance(nodes, (str, bytes)):
        raise EvidenceBenchTypedScientificGraphError("source nodes must be a sequence")
    normalized = tuple(nodes)
    if len(normalized) != SOURCE_NODE_COUNT:
        raise EvidenceBenchTypedScientificGraphError(
            "each item must contain exactly 32 source nodes"
        )
    prior_end = -1
    for expected_i, node in enumerate(normalized):
        if not isinstance(node, SourceNode):
            raise EvidenceBenchTypedScientificGraphError("source node has the wrong type")
        if type(node.span_i) is not int or node.span_i != expected_i:
            raise EvidenceBenchTypedScientificGraphError(
                "source node identities must be contiguous source-list indices"
            )
        if (
            type(node.start) is not int
            or type(node.end) is not int
            or node.start < 0
            or node.end <= node.start
            or node.start < prior_end
        ):
            raise EvidenceBenchTypedScientificGraphError(
                "source node offsets must be ordered and non-overlapping"
            )
        if not isinstance(node.identity_text, str) or not embedding_text(node.identity_text):
            raise EvidenceBenchTypedScientificGraphError(
                "source node identity text must be nonempty text"
            )
        prior_end = node.end
    return normalized


def _edge(order: int, left: int, right: int) -> TypedEdge | None:
    if left == right:
        return None
    return TypedEdge(order, min(left, right), max(left, right))


def _valid_short_form(surface: str) -> bool:
    stripped = surface.strip()
    compact = "".join(character for character in stripped if character.isalnum())
    return (
        2 <= len(compact) <= 10
        and len(stripped.split()) <= 2
        and stripped[:1].isalnum()
        and any(character.isalpha() for character in compact)
        and any(character.isupper() or character.isdigit() for character in stripped)
        and all(character.isalnum() or character in "-/. " for character in stripped)
    )


def _schwartz_hearst_long_form(short_form: str, candidate: str) -> str | None:
    """Return the minimal aligned long form using frozen surface alignment."""

    short_characters = [
        character.casefold() for character in short_form if character.isalnum()
    ]
    if not short_characters:
        return None
    folded = candidate.casefold()
    cursor = len(candidate)
    start = -1
    for short_i in range(len(short_characters) - 1, -1, -1):
        character = short_characters[short_i]
        found = folded.rfind(character, 0, cursor)
        if short_i == 0:
            while found > 0 and candidate[found - 1].isalnum():
                found = folded.rfind(character, 0, found)
        if found < 0:
            return None
        start = found
        cursor = found
    long_form = candidate[start:].strip(" \t\r\n,;:-")
    long_tokens = tuple(_LONG_FORM_TOKEN.finditer(long_form))
    maximum_words = min(len(short_characters) + 5, len(short_characters) * 2)
    if (
        not long_tokens
        or len(long_tokens) > maximum_words
        or len("".join(unicode_tokens(long_form))) < len(short_characters)
        or pattern_text(long_form) == pattern_text(short_form)
    ):
        return None
    return long_form


def _abbreviation_definitions(text: str) -> tuple[tuple[str, str], ...]:
    definitions: list[tuple[str, str]] = []
    for match in _PARENTHETICAL.finditer(text):
        short_form = match.group(1).strip()
        if not _valid_short_form(short_form):
            continue
        prefix = text[: match.start()].rstrip()
        prefix_tokens = tuple(_LONG_FORM_TOKEN.finditer(prefix))
        compact_short = "".join(
            character for character in short_form if character.isalnum()
        )
        maximum_words = min(len(compact_short) + 5, len(compact_short) * 2)
        if not prefix_tokens:
            continue
        candidate_start = prefix_tokens[max(0, len(prefix_tokens) - maximum_words)].start()
        candidate = prefix[candidate_start:]
        boundary = max(candidate.rfind(mark) for mark in ".;:!?")
        if boundary >= 0:
            candidate = candidate[boundary + 1 :].lstrip()
        long_form = _schwartz_hearst_long_form(short_form, candidate)
        if long_form is None:
            continue
        pair = (pattern_text(short_form), pattern_text(long_form))
        if pair not in definitions:
            definitions.append(pair)
    return tuple(definitions)


def _contains_surface(pattern: str, surface: str) -> bool:
    return (
        re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", pattern) is not None
    )


def _xref_keys(text: str) -> tuple[tuple[str, str], ...]:
    kind_map = {
        "fig": "figure",
        "figs": "figure",
        "figure": "figure",
        "figures": "figure",
        "tab": "table",
        "tabs": "table",
        "table": "table",
        "tables": "table",
        "eq": "equation",
        "eqs": "equation",
        "equation": "equation",
        "equations": "equation",
    }
    keys = {
        (kind_map[match.group("kind").casefold()], match.group("label").casefold())
        for match in _SCIENTIFIC_XREF.finditer(text)
    }
    return tuple(sorted(keys))


def _overlaps(interval: tuple[int, int], ranges: Sequence[tuple[int, int]]) -> bool:
    return any(interval[0] < end and start < interval[1] for start, end in ranges)


def _identifier_surface(surface: str) -> bool:
    letters = [character for character in surface if character.isalpha()]
    uppercase = sum(character.isupper() for character in letters)
    return (
        2 <= len(surface) <= 24
        and len(letters) >= 2
        and (uppercase >= 2 or (uppercase >= 1 and any(c.isdigit() for c in surface)))
    )


def _title_surface(surface: str) -> bool:
    return (
        surface[:1].isupper()
        and any(character.islower() for character in surface[1:])
        and pattern_text(surface) not in STOPWORDS
    )


def _surface_entities(text: str) -> frozenset[str]:
    """Extract frozen casing-based surfaces; no domain lexicon is consulted."""

    xref_ranges = tuple(match.span() for match in _SCIENTIFIC_XREF.finditer(text))
    matches = tuple(_ENTITY_TOKEN.finditer(text))
    surfaces: set[str] = set()
    for match in matches:
        if not _overlaps(match.span(), xref_ranges) and _identifier_surface(match.group(0)):
            surfaces.add(pattern_text(match.group(0)))

    run: list[re.Match[str]] = []

    def flush() -> None:
        if 2 <= len(run) <= 5:
            start, end = run[0].start(), run[-1].end()
            if not _overlaps((start, end), xref_ranges):
                surfaces.add(pattern_text(text[start:end]))
        run.clear()

    previous_end: int | None = None
    for match in matches:
        separated_only_by_space = (
            previous_end is not None and not text[previous_end : match.start()].strip()
        )
        if not _title_surface(match.group(0)) or (
            run and not separated_only_by_space
        ):
            flush()
        if _title_surface(match.group(0)):
            run.append(match)
        previous_end = match.end()
    flush()
    return frozenset(surface for surface in surfaces if surface)


def build_typed_scientific_graph(nodes: Sequence[SourceNode]) -> tuple[TypedEdge, ...]:
    """Build four frozen surface-only edge families from exactly 32 nodes."""

    source_nodes = _validated_nodes(nodes)
    raw_texts = tuple(node.embedding_text for node in source_nodes)
    patterns = tuple(node.pattern_text for node in source_nodes)
    edges: set[TypedEdge] = set()

    for left_i in range(SOURCE_NODE_COUNT - 1):
        edges.add(
            TypedEdge(EDGE_FAMILY_ORDER[ADJACENT_BUCKET], left_i, left_i + 1)
        )

    for definition_i, raw_text in enumerate(raw_texts):
        for short_form, long_form in _abbreviation_definitions(raw_text):
            for mention_i, candidate in enumerate(patterns):
                if mention_i == definition_i:
                    continue
                if _contains_surface(candidate, short_form) or _contains_surface(
                    candidate, long_form
                ):
                    edge = _edge(
                        EDGE_FAMILY_ORDER[ABBREVIATION_DEFINITION],
                        definition_i,
                        mention_i,
                    )
                    if edge is not None:
                        edges.add(edge)

    xref_nodes: dict[tuple[str, str], list[int]] = {}
    for node_i, text in enumerate(raw_texts):
        for key in _xref_keys(text):
            xref_nodes.setdefault(key, []).append(node_i)
    for node_indices in xref_nodes.values():
        for left_position in range(len(node_indices)):
            for right_position in range(left_position + 1, len(node_indices)):
                edge = _edge(
                    EDGE_FAMILY_ORDER[EXPLICIT_SCIENTIFIC_XREF],
                    node_indices[left_position],
                    node_indices[right_position],
                )
                if edge is not None:
                    edges.add(edge)

    entity_nodes: dict[str, list[int]] = {}
    for node_i, text in enumerate(raw_texts):
        for entity in _surface_entities(text):
            entity_nodes.setdefault(entity, []).append(node_i)
    for node_indices in entity_nodes.values():
        distinct = sorted(set(node_indices))
        if not 2 <= len(distinct) <= 3:
            continue
        for left_position in range(len(distinct)):
            for right_position in range(left_position + 1, len(distinct)):
                edge = _edge(
                    EDGE_FAMILY_ORDER[RARE_ENTITY_BRIDGE],
                    distinct[left_position],
                    distinct[right_position],
                )
                if edge is not None:
                    edges.add(edge)

    return tuple(sorted(edges))


def _validated_integer_vector(
    values: Sequence[int], expected_length: int, field: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise EvidenceBenchTypedScientificGraphError(
            f"{field} must be an integer vector"
        )
    normalized = tuple(values)
    if len(normalized) != expected_length or any(
        type(value) is not int for value in normalized
    ):
        raise EvidenceBenchTypedScientificGraphError(
            f"{field} must be a length-N integer vector"
        )
    return normalized


def _validated_top5(
    top5: Sequence[int], source_count: int, field: str = "top5"
) -> tuple[int, int, int, int, int]:
    if isinstance(top5, (str, bytes)):
        raise EvidenceBenchTypedScientificGraphError(f"{field} is malformed")
    normalized = tuple(top5)
    if (
        len(normalized) != TOP_K
        or len(set(normalized)) != TOP_K
        or any(
            type(value) is not int or not 0 <= value < source_count
            for value in normalized
        )
    ):
        raise EvidenceBenchTypedScientificGraphError(
            f"{field} must contain five unique bounded source indices"
        )
    return normalized  # type: ignore[return-value]


def _validated_edges(
    edges: Sequence[TypedEdge], source_count: int
) -> tuple[TypedEdge, ...]:
    if isinstance(edges, (str, bytes)):
        raise EvidenceBenchTypedScientificGraphError("typed edges must be a sequence")
    normalized = tuple(edges)
    seen: set[tuple[int, int, int]] = set()
    for edge in normalized:
        if not isinstance(edge, TypedEdge):
            raise EvidenceBenchTypedScientificGraphError(
                "typed edge has the wrong type"
            )
        key = (edge.edge_family_order, edge.left_span_i, edge.right_span_i)
        if (
            edge.edge_family_order not in range(len(EDGE_FAMILIES))
            or not 0 <= edge.left_span_i < edge.right_span_i < source_count
            or key in seen
        ):
            raise EvidenceBenchTypedScientificGraphError(
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
    nodes: Sequence[SourceNode],
    typed_edges: Sequence[TypedEdge],
    official_top5: Sequence[int],
    query_node_similarities: Sequence[int],
) -> tuple[CandidateRecord, ...]:
    """Build one shared raw record for every family/seed/neighbor triple."""

    source_nodes = _validated_nodes(nodes)
    similarities = _validated_integer_vector(
        query_node_similarities, SOURCE_NODE_COUNT, "query-node similarities"
    )
    top5 = _validated_top5(official_top5, SOURCE_NODE_COUNT, "official Hippo top5")
    edges = _validated_edges(typed_edges, SOURCE_NODE_COUNT)
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
                raise EvidenceBenchTypedScientificGraphError(
                    "duplicate candidate triple"
                )
            seen.add(triple)
            records.append(
                CandidateRecord(
                    edge_family_order=edge.edge_family_order,
                    official_seed_rank=seed_rank[seed_i],
                    seed_span_i=seed_i,
                    neighbor_span_i=neighbor_i,
                    query_similarity_int=similarities[neighbor_i],
                    absolute_start_offset_distance=abs(
                        source_nodes[seed_i].start - source_nodes[neighbor_i].start
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
        raise EvidenceBenchTypedScientificGraphError(
            "candidate table must be a sequence"
        )
    normalized = tuple(records)
    if any(not isinstance(record, CandidateRecord) for record in normalized):
        raise EvidenceBenchTypedScientificGraphError(
            "candidate record has the wrong type"
        )
    if tuple(sorted(normalized, key=_candidate_sort_key)) != normalized:
        raise EvidenceBenchTypedScientificGraphError(
            "candidate table is not in common sort order"
        )
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
            raise EvidenceBenchTypedScientificGraphError(
                "candidate record violates the frozen schema"
            )
        seen.add(triple)
    return normalized


def execute_recipe(
    official_top5: Sequence[int],
    common_candidate_table: Sequence[CandidateRecord],
    query_node_similarities: Sequence[int],
    recipe_id: str,
) -> ActionTrace:
    """Execute one recipe while always visiting the complete shared table."""

    if not isinstance(recipe_id, str) or recipe_id not in _RECIPE_BY_ID:
        raise EvidenceBenchTypedScientificGraphError(
            "recipe ID is not in the frozen registry"
        )
    similarities = _validated_integer_vector(
        query_node_similarities, SOURCE_NODE_COUNT, "query-node similarities"
    )
    top5 = _validated_top5(official_top5, SOURCE_NODE_COUNT, "official Hippo top5")
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

    retained_official = [
        span_i for span_i in top5 if span_i in retained and span_i not in added
    ]
    output = tuple(retained_official + added)
    if len(output) != TOP_K or len(set(output)) != TOP_K or set(output) != retained:
        raise EvidenceBenchTypedScientificGraphError(
            "recipe failed to return five unique source indices"
        )
    accepted_count = sum(
        decision.disposition == "accepted" for decision in decisions
    )
    return ActionTrace(
        recipe_id=recipe_id,
        output_top5=output,  # type: ignore[arg-type]
        swap_count=TOP_K - len(set(output).intersection(top5)),
        records_visited=len(decisions),
        accepted_count=accepted_count,
        rejected_count=len(decisions) - accepted_count,
        common_scan_sha256=_scan_sha256(records),
        decisions=tuple(decisions),
    )


def execute_all_recipes(
    official_top5: Sequence[int],
    common_candidate_table: Sequence[CandidateRecord],
    query_node_similarities: Sequence[int],
) -> tuple[ActionTrace, ...]:
    return tuple(
        execute_recipe(
            official_top5,
            common_candidate_table,
            query_node_similarities,
            recipe.recipe_id,
        )
        for recipe in _RECIPES
    )


def _validated_similarity_matrix(
    values: Sequence[Sequence[int]], source_count: int
) -> tuple[tuple[int, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise EvidenceBenchTypedScientificGraphError(
            "node-node similarities must be a matrix"
        )
    rows = tuple(tuple(row) for row in values)
    if len(rows) != source_count:
        raise EvidenceBenchTypedScientificGraphError(
            "node-node similarity matrix must be N by N"
        )
    for row in rows:
        if len(row) != source_count or any(
            type(value) is not int
            or not -INTEGER_SCALE <= value <= INTEGER_SCALE
            for value in row
        ):
            raise EvidenceBenchTypedScientificGraphError(
                "node-node similarity matrix must be integer N by N"
            )
    for left in range(source_count):
        for right in range(left + 1, source_count):
            if rows[left][right] != rows[right][left]:
                raise EvidenceBenchTypedScientificGraphError(
                    "node-node similarity matrix must be symmetric"
                )
    return rows


def semantic_coverage(
    candidate_top5: Sequence[int], query_node_similarities: Sequence[int]
) -> int:
    similarities = _validated_integer_vector(
        query_node_similarities,
        SOURCE_NODE_COUNT,
        "query-node similarities",
    )
    candidate = _validated_top5(candidate_top5, SOURCE_NODE_COUNT, "candidate top5")
    values = sorted(
        (
            (
                min(INTEGER_SCALE, max(-INTEGER_SCALE, similarities[node_i]))
                + INTEGER_SCALE
            )
            // 2
            for node_i in candidate
        ),
        reverse=True,
    )
    return (3 * values[0] + values[1]) // 4


def lexical_coverage(
    query_text: str, nodes: Sequence[SourceNode], candidate_top5: Sequence[int]
) -> int:
    source_nodes = _validated_nodes(nodes)
    candidate = _validated_top5(
        candidate_top5, SOURCE_NODE_COUNT, "candidate top5"
    )
    if not isinstance(query_text, str):
        raise EvidenceBenchTypedScientificGraphError("query text must be text")
    node_token_sets = tuple(set(unicode_tokens(node.identity_text)) for node in source_nodes)
    document_frequency: Counter[str] = Counter()
    for tokens in node_token_sets:
        document_frequency.update(tokens)
    query_tokens = set(unicode_tokens(query_text))
    usable = {
        token
        for token in query_tokens
        if token not in STOPWORDS
        and not token.isnumeric()
        and len(token) >= 2
        and document_frequency[token] <= SOURCE_NODE_COUNT // 2
    }
    if not usable:
        return 0
    weights = {
        token: SOURCE_NODE_COUNT + 1 - document_frequency[token]
        for token in usable
    }
    covered = set().union(*(node_token_sets[node_i] for node_i in candidate))
    numerator = sum(weight for token, weight in weights.items() if token in covered)
    return INTEGER_SCALE * numerator // sum(weights.values())


def coverage_components(
    query_text: str,
    nodes: Sequence[SourceNode],
    candidate_top5: Sequence[int],
    official_top5: Sequence[int],
    typed_edges: Sequence[TypedEdge],
    query_node_similarities: Sequence[int],
    node_node_similarities: Sequence[Sequence[int]],
) -> CoverageComponents:
    """Compute the five frozen integer components for one candidate."""

    source_nodes = _validated_nodes(nodes)
    candidate = _validated_top5(
        candidate_top5, SOURCE_NODE_COUNT, "candidate top5"
    )
    official = _validated_top5(
        official_top5, SOURCE_NODE_COUNT, "official Hippo top5"
    )
    similarities = _validated_integer_vector(
        query_node_similarities,
        SOURCE_NODE_COUNT,
        "query-node similarities",
    )
    matrix = _validated_similarity_matrix(
        node_node_similarities, SOURCE_NODE_COUNT
    )
    edges = _validated_edges(typed_edges, SOURCE_NODE_COUNT)

    sem = semantic_coverage(candidate, similarities)
    lex = lexical_coverage(query_text, source_nodes, candidate)
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
            and (
                edge.left_span_i in official_set
                or edge.right_span_i in official_set
            )
        ]
        closed = sum(
            edge.left_span_i in candidate_set
            and edge.right_span_i in candidate_set
            for edge in demand
        )
        closures.append(INTEGER_SCALE * closed // len(demand) if demand else 0)
    return CoverageComponents(sem, lex, diversity, churn, tuple(closures))  # type: ignore[arg-type]


def _validated_components(components: CoverageComponents) -> CoverageComponents:
    if not isinstance(components, CoverageComponents):
        raise EvidenceBenchTypedScientificGraphError(
            "coverage components have the wrong type"
        )
    scalar_values = (
        components.Sem,
        components.Lex,
        components.Diversity,
        components.Churn,
    )
    if any(type(value) is not int for value in scalar_values):
        raise EvidenceBenchTypedScientificGraphError(
            "coverage components must be integers"
        )
    if (
        not 0 <= components.Sem <= INTEGER_SCALE
        or not 0 <= components.Lex <= INTEGER_SCALE
        or not 0 <= components.Diversity <= INTEGER_SCALE
        or not 0 <= components.Churn <= INTEGER_SCALE
        or components.Churn % 200_000
    ):
        raise EvidenceBenchTypedScientificGraphError(
            "coverage component is out of range"
        )
    if (
        len(components.Closure) != len(EDGE_FAMILIES)
        or any(
            type(value) is not int or not 0 <= value <= INTEGER_SCALE
            for value in components.Closure
        )
    ):
        raise EvidenceBenchTypedScientificGraphError("closure vector is invalid")
    return components


def score_coverage(components: CoverageComponents, evaluator_id: str) -> Fraction:
    """Score one component row with one of sixteen exact-Fraction evaluators."""

    components = _validated_components(components)
    if not isinstance(evaluator_id, str) or evaluator_id not in _EVALUATOR_BY_ID:
        raise EvidenceBenchTypedScientificGraphError(
            "evaluator ID is not in the frozen registry"
        )
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
        raise EvidenceBenchTypedScientificGraphError(
            f"{field} must contain the complete nine-recipe registry"
        )


def _validated_formation_item(item: FormationItem) -> FormationItem:
    if not isinstance(item, FormationItem):
        raise EvidenceBenchTypedScientificGraphError(
            "formation item has the wrong type"
        )
    _validate_recipe_mapping(item.components_by_recipe, "component table")
    _validate_recipe_mapping(item.utility_by_recipe, "utility table")
    _validate_recipe_mapping(item.complete_by_recipe, "complete table")
    for recipe in _RECIPES:
        _validated_components(item.components_by_recipe[recipe.recipe_id])
        utility = item.utility_by_recipe[recipe.recipe_id]
        complete = item.complete_by_recipe[recipe.recipe_id]
        if (
            type(utility) is not int
            or not 0 <= utility <= UTILITY_RECALL_SCALE + UTILITY_COMPLETE_BONUS
            or type(complete) is not bool
        ):
            raise EvidenceBenchTypedScientificGraphError(
                "formation outcomes are malformed"
            )
    return item


def select_a_evaluator(items: Sequence[FormationItem]) -> AFormationSelection:
    """Perform frozen late-label regret selection over all sixteen evaluators."""

    if isinstance(items, (str, bytes)) or not items:
        raise EvidenceBenchTypedScientificGraphError("A-form items must be nonempty")
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
        raise EvidenceBenchTypedScientificGraphError(
            "F component tables must be nonempty"
        )
    if not isinstance(evaluator_id, str) or evaluator_id not in _EVALUATOR_BY_ID:
        raise EvidenceBenchTypedScientificGraphError(
            "evaluator ID is not in the frozen registry"
        )
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
    """Return false for R0 or all-item membership identity."""

    if selected_recipe_id not in _RECIPE_BY_ID:
        raise EvidenceBenchTypedScientificGraphError(
            "recipe ID is not in the frozen registry"
        )
    if len(selected_outputs) != len(official_outputs) or not selected_outputs:
        raise EvidenceBenchTypedScientificGraphError(
            "transition output tables are malformed"
        )
    if selected_recipe_id == "R0_HIPPO_TOP5":
        return False
    for selected, official in zip(selected_outputs, official_outputs):
        selected_top5 = _validated_top5(
            selected, SOURCE_NODE_COUNT, "selected output"
        )
        official_top5 = _validated_top5(
            official, SOURCE_NODE_COUNT, "official output"
        )
        if set(selected_top5) != set(official_top5):
            return True
    return False


def item_utility(
    top5: Sequence[int],
    aspect_bucket_sets: Sequence[Sequence[int]],
    *,
    source_count: int = SOURCE_NODE_COUNT,
) -> tuple[int, int, int]:
    """Return covered aspects, completion, and half-up recall-plus-bonus U."""

    if type(source_count) is not int or source_count != SOURCE_NODE_COUNT:
        raise EvidenceBenchTypedScientificGraphError(
            "source count must equal the frozen 32-node item size"
        )
    selected = _validated_top5(top5, source_count)
    if isinstance(aspect_bucket_sets, (str, bytes)):
        raise EvidenceBenchTypedScientificGraphError(
            "aspect evidence bucket sets are malformed"
        )
    aspects = tuple(aspect_bucket_sets)
    if not aspects:
        raise EvidenceBenchTypedScientificGraphError(
            "aspect evidence bucket sets must be nonempty"
        )
    normalized_aspects: list[tuple[int, ...]] = []
    for bucket_set in aspects:
        if isinstance(bucket_set, (str, bytes)):
            raise EvidenceBenchTypedScientificGraphError(
                "each aspect evidence bucket set must be a sequence"
            )
        buckets = tuple(bucket_set)
        if (
            not buckets
            or tuple(sorted(buckets)) != buckets
            or len(set(buckets)) != len(buckets)
            or any(
                type(bucket_i) is not int or not 0 <= bucket_i < source_count
                for bucket_i in buckets
            )
        ):
            raise EvidenceBenchTypedScientificGraphError(
                "each aspect evidence bucket set must be nonempty, sorted, "
                "distinct, and bounded"
            )
        normalized_aspects.append(buckets)
    selected_set = set(selected)
    covered_aspects = sum(
        bool(selected_set.intersection(bucket_set))
        for bucket_set in normalized_aspects
    )
    aspect_count = len(normalized_aspects)
    # Integer half-up rounding of 1000 * covered_aspects / aspect_count.
    recall = (
        2 * UTILITY_RECALL_SCALE * covered_aspects + aspect_count
    ) // (2 * aspect_count)
    complete = int(covered_aspects == aspect_count)
    utility = recall + UTILITY_COMPLETE_BONUS * complete
    return covered_aspects, complete, utility


def exact_magnitude_preserving_sign_flip(deltas: Sequence[int]) -> dict[str, object]:
    """Compute the frozen one-sided exact sign-flip test without Monte Carlo."""

    if (
        isinstance(deltas, (str, bytes))
        or not deltas
        or any(type(value) is not int for value in deltas)
    ):
        raise EvidenceBenchTypedScientificGraphError(
            "paired U deltas are malformed"
        )
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
    "ABBREVIATION_DEFINITION",
    "ADJACENT_BUCKET",
    "AFormationSelection",
    "ActionTrace",
    "CandidateDecision",
    "CandidateRecord",
    "CoverageComponents",
    "EDGE_FAMILIES",
    "EDGE_FAMILY_ORDER",
    "EXPLICIT_SCIENTIFIC_XREF",
    "EvaluatorFormationResult",
    "EvaluatorSpec",
    "EvidenceBenchTypedScientificGraphError",
    "FSearchSelection",
    "FormationItem",
    "INTEGER_SCALE",
    "PROMOTION_ALPHA",
    "RARE_ENTITY_BRIDGE",
    "RecipeSpec",
    "SOURCE_NODE_COUNT",
    "STOPWORDS",
    "SourceNode",
    "TOP_K",
    "TypedEdge",
    "UTILITY_COMPLETE_BONUS",
    "UTILITY_RECALL_SCALE",
    "VERSION",
    "build_common_candidate_table",
    "build_typed_scientific_graph",
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
