"""Row-free FEVEROUS P6 query-anchored residual typed operator.

The module is intentionally dataset-, model-, filesystem-, network-, label-,
and HippoRAG-free.  A caller injects a closed ordered atomic-unit corpus and a
complete, quantized claim-facet-by-corpus semantic tensor.  Every P6 recipe
starts from the same dense order, retains its first three units, scans the
whole corpus, and fills two residual slots.  Typed paths are useful only when
they are reachable from a direct claim-facet anchor; a disconnected corpus
component therefore cannot acquire a query-independent clique score.

The tensor separates continuous semantic coverage from direct-anchor
strength.  The offline semantic adapter is responsible for constructing both
for every facet and every corpus unit (including the fixed MiniLM/NLI and exact
entity/numeric rules).  This core never receives a shortlist or a HippoRAG
output and cannot silently turn either into its candidate universe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from itertools import combinations
import re
import unicodedata
from typing import Iterable, Sequence


VERSION = "feverous_p6_query_anchored_operator_v1"
TOP_K = 5
RAW_RETAINED = 3
RESIDUAL_BUDGET = 2
INTEGER_SCALE = 1_000_000
CORPUS_UNIT_COUNT = 8192

UNIT_TYPES = (
    "sentence",
    "item",
    "cell",
    "header_cell",
    "table_caption",
)
FACET_TYPES = ("entity", "numeric_or_date", "relation_clause")
ENTITY_TYPES = ("LOC", "MISC", "ORG", "PER")

SAME_PAGE_ADJACENT_OFFICIAL_ORDER = "same_page_adjacent_official_order"
SAME_TABLE_ROW = "same_table_row"
CELL_TO_APPLICABLE_HEADER = "cell_to_applicable_header"
SAME_LIST_PARENT_PATH = "same_list_parent_path"
RECIPROCAL_SHARED_NORMALIZED_ENTITY = "reciprocal_shared_normalized_entity"
EDGE_FAMILIES = (
    SAME_PAGE_ADJACENT_OFFICIAL_ORDER,
    SAME_TABLE_ROW,
    CELL_TO_APPLICABLE_HEADER,
    SAME_LIST_PARENT_PATH,
    RECIPROCAL_SHARED_NORMALIZED_ENTITY,
)
EDGE_FAMILY_ORDER = {family: order for order, family in enumerate(EDGE_FAMILIES)}

RECIPE_IDS = (
    "R0_DENSE5",
    "R1_P6_DIRECT_B2",
    "R2_P6_PATH1_B2",
    "R3_P6_PATH2_B2",
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FeverousP6OperatorError(ValueError):
    """An input or receipt violates the frozen P6 operator contract."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stream_hash(header: object, rows: Iterable[object]) -> str:
    """Hash a complete ordered scan without retaining its 8192-row receipt."""

    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    for row in rows:
        digest.update(b"\n")
        digest.update(_canonical_bytes(row))
    return digest.hexdigest()


def normalize_key(value: str) -> str:
    """NFKC, whitespace-collapsed, case-folded identity normalization."""

    if not isinstance(value, str):
        raise TypeError("identity text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeverousP6OperatorError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise FeverousP6OperatorError(f"{field} is below its minimum")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise FeverousP6OperatorError(f"{field} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True, order=True)
class EntityKey:
    entity_type: str
    normalized_span: str

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise FeverousP6OperatorError("entity type is outside the frozen registry")
        if not self.normalized_span or normalize_key(self.normalized_span) != self.normalized_span:
            raise FeverousP6OperatorError("entity span is not canonically normalized")


def make_entity_key(entity_type: str, span: str) -> EntityKey:
    normalized = normalize_key(span)
    if not normalized or "\x00" in normalized:
        raise FeverousP6OperatorError("entity span is empty or contains NUL")
    return EntityKey(entity_type=entity_type, normalized_span=normalized)


@dataclass(frozen=True)
class AtomicUnit:
    """One public-sidecar atomic unit in the fixed closed corpus.

    ``corpus_ordinal`` is the only action identity.  Opaque page/table keys and
    coordinates are structural sidecars, never gold/evidence identifiers.
    ``applicable_header_ordinals`` must point to exact ``header_cell`` units;
    a normal ``cell`` is deliberately not accepted as a header substitute.
    """

    corpus_ordinal: int
    unit_type: str
    page_key: str
    official_order: int
    section_path: tuple[str, ...] = ()
    table_key: str | None = None
    table_row: int | None = None
    applicable_header_ordinals: tuple[int, ...] = ()
    list_parent_path: tuple[str, ...] = ()
    entities: tuple[EntityKey, ...] = ()

    def __post_init__(self) -> None:
        _require_int(self.corpus_ordinal, "corpus ordinal", minimum=0)
        _require_int(self.official_order, "official order", minimum=0)
        if self.unit_type not in UNIT_TYPES:
            raise FeverousP6OperatorError("atomic unit type is outside the registry")
        if not isinstance(self.page_key, str) or not self.page_key or "\x00" in self.page_key:
            raise FeverousP6OperatorError("page key is invalid")
        if self.table_key is not None and (
            not isinstance(self.table_key, str) or not self.table_key or "\x00" in self.table_key
        ):
            raise FeverousP6OperatorError("table key is invalid")
        if self.table_row is not None:
            _require_int(self.table_row, "table row", minimum=0)
        for field, value in (
            ("section path", self.section_path),
            ("applicable headers", self.applicable_header_ordinals),
            ("list parent path", self.list_parent_path),
            ("entities", self.entities),
        ):
            if not isinstance(value, tuple):
                raise FeverousP6OperatorError(f"{field} must be an immutable tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.applicable_header_ordinals
        ):
            raise FeverousP6OperatorError("applicable header ordinal is invalid")
        if any(
            not isinstance(value, str) or not value or "\x00" in value
            for value in self.section_path
        ):
            raise FeverousP6OperatorError("section path is invalid")
        if tuple(sorted(set(self.applicable_header_ordinals))) != self.applicable_header_ordinals:
            raise FeverousP6OperatorError("applicable header ordinals are not canonical")
        if any(not isinstance(value, str) or not value or "\x00" in value for value in self.list_parent_path):
            raise FeverousP6OperatorError("list parent path is invalid")
        if tuple(sorted(set(self.entities))) != self.entities:
            raise FeverousP6OperatorError("entities are not a canonical set")
        if self.unit_type in {"cell", "header_cell"}:
            if self.table_key is None or self.table_row is None:
                raise FeverousP6OperatorError("cell types require table coordinates")
        elif self.table_row is not None:
            raise FeverousP6OperatorError("only exact cell types may have a table row")
        if self.unit_type != "cell" and self.applicable_header_ordinals:
            raise FeverousP6OperatorError("only an exact cell may declare applicable headers")
        if self.unit_type != "item" and self.list_parent_path:
            raise FeverousP6OperatorError("only an exact item may declare a list parent path")


@dataclass(frozen=True, order=True)
class TypedEdge:
    family_order: int
    left_ordinal: int
    right_ordinal: int
    strength_int: int

    @property
    def family(self) -> str:
        if self.family_order not in range(len(EDGE_FAMILIES)):
            raise FeverousP6OperatorError("typed edge family order is invalid")
        return EDGE_FAMILIES[self.family_order]

    def public_tuple(self) -> tuple[str, int, int, int]:
        return (self.family, self.left_ordinal, self.right_ordinal, self.strength_int)


@dataclass(frozen=True, order=True)
class Neighbor:
    neighbor_ordinal: int
    family_order: int
    strength_int: int


@dataclass(frozen=True)
class TypedCorpusGraph:
    units: tuple[AtomicUnit, ...]
    edges: tuple[TypedEdge, ...]
    neighbors: tuple[tuple[Neighbor, ...], ...]
    graph_sha256: str


def _unit_receipt(unit: AtomicUnit) -> list[object]:
    return [
        unit.corpus_ordinal,
        unit.unit_type,
        unit.page_key,
        unit.official_order,
        list(unit.section_path),
        unit.table_key,
        unit.table_row,
        list(unit.applicable_header_ordinals),
        list(unit.list_parent_path),
        [[entity.entity_type, entity.normalized_span] for entity in unit.entities],
    ]


def _graph_receipt_body(graph: TypedCorpusGraph) -> dict[str, object]:
    return {
        "edges": [list(edge.public_tuple()) for edge in graph.edges],
        "units": [_unit_receipt(unit) for unit in graph.units],
        "version": VERSION,
    }


def recompute_graph_sha256(graph: TypedCorpusGraph) -> str:
    if not isinstance(graph, TypedCorpusGraph):
        raise FeverousP6OperatorError("graph has the wrong type")
    return stable_hash(_graph_receipt_body(graph))


def _validated_graph(graph: TypedCorpusGraph) -> TypedCorpusGraph:
    if not isinstance(graph.units, tuple) or not isinstance(graph.edges, tuple) or not isinstance(graph.neighbors, tuple):
        raise FeverousP6OperatorError("graph containers must be immutable tuples")
    if len(graph.units) != CORPUS_UNIT_COUNT:
        raise FeverousP6OperatorError("closed corpus is not exactly 8192 units")
    if tuple(unit.corpus_ordinal for unit in graph.units) != tuple(range(len(graph.units))):
        raise FeverousP6OperatorError("corpus ordinals are not complete source order")
    if len(graph.neighbors) != len(graph.units) or any(not isinstance(row, tuple) for row in graph.neighbors):
        raise FeverousP6OperatorError("graph neighbor matrix is malformed")
    _require_sha256(graph.graph_sha256, "graph hash")
    if recompute_graph_sha256(graph) != graph.graph_sha256:
        raise FeverousP6OperatorError("graph self hash drifted")
    expected: list[list[Neighbor]] = [[] for _unit in graph.units]
    previous: TypedEdge | None = None
    for edge in graph.edges:
        if previous is not None and not previous < edge:
            raise FeverousP6OperatorError("typed edges are not a strict canonical set")
        previous = edge
        if edge.family_order not in range(len(EDGE_FAMILIES)):
            raise FeverousP6OperatorError("typed edge family order is invalid")
        if not 0 <= edge.left_ordinal < edge.right_ordinal < len(graph.units):
            raise FeverousP6OperatorError("typed edge endpoints are invalid")
        _require_int(edge.strength_int, "typed edge strength", minimum=1)
        expected[edge.left_ordinal].append(
            Neighbor(edge.right_ordinal, edge.family_order, edge.strength_int)
        )
        expected[edge.right_ordinal].append(
            Neighbor(edge.left_ordinal, edge.family_order, edge.strength_int)
        )
    canonical_neighbors = tuple(tuple(sorted(row)) for row in expected)
    if graph.neighbors != canonical_neighbors:
        raise FeverousP6OperatorError("graph neighbor matrix does not match typed edges")
    return graph


def build_typed_graph(units: Sequence[AtomicUnit]) -> TypedCorpusGraph:
    """Build the five-family immutable graph without inspecting unit text."""

    rows = tuple(units)
    if any(not isinstance(unit, AtomicUnit) for unit in rows):
        raise FeverousP6OperatorError("closed corpus contains a non-atomic unit")
    if len(rows) != CORPUS_UNIT_COUNT:
        raise FeverousP6OperatorError("closed corpus is not exactly 8192 units")
    if tuple(unit.corpus_ordinal for unit in rows) != tuple(range(len(rows))):
        raise FeverousP6OperatorError("corpus ordinals are not complete source order")

    by_page_section: dict[tuple[str, tuple[str, ...]], list[AtomicUnit]] = {}
    by_table_row: dict[tuple[str, str, int], list[int]] = {}
    by_list_parent: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    by_entity: dict[EntityKey, list[int]] = {}
    for unit in rows:
        by_page_section.setdefault(
            (unit.page_key, unit.section_path), []
        ).append(unit)
        if unit.unit_type in {"cell", "header_cell"}:
            assert unit.table_key is not None and unit.table_row is not None
            by_table_row.setdefault(
                (unit.page_key, unit.table_key, unit.table_row), []
            ).append(unit.corpus_ordinal)
        if unit.unit_type == "item" and unit.list_parent_path:
            by_list_parent.setdefault(
                (unit.page_key, unit.list_parent_path), []
            ).append(unit.corpus_ordinal)
        for entity in unit.entities:
            by_entity.setdefault(entity, []).append(unit.corpus_ordinal)

    # Parallel edge families are retained.  Duplicate witnesses within one
    # family collapse to the greatest integer strength.
    edge_strength: dict[tuple[int, int, int], int] = {}

    def add(family: str, left: int, right: int, strength: int = INTEGER_SCALE) -> None:
        if left == right:
            return
        left, right = sorted((left, right))
        key = (EDGE_FAMILY_ORDER[family], left, right)
        edge_strength[key] = max(edge_strength.get(key, 0), strength)

    for page_units in by_page_section.values():
        ordered = sorted(page_units, key=lambda unit: (unit.official_order, unit.corpus_ordinal))
        if len({unit.official_order for unit in ordered}) != len(ordered):
            raise FeverousP6OperatorError("official order is not unique within a page")
        for left, right in zip(ordered, ordered[1:]):
            if right.official_order == left.official_order + 1:
                add(
                    SAME_PAGE_ADJACENT_OFFICIAL_ORDER,
                    left.corpus_ordinal,
                    right.corpus_ordinal,
                )

    for ordinals in by_table_row.values():
        for left, right in combinations(sorted(ordinals), 2):
            add(SAME_TABLE_ROW, left, right)

    for unit in rows:
        if unit.unit_type != "cell":
            continue
        for header_ordinal in unit.applicable_header_ordinals:
            if header_ordinal >= len(rows):
                raise FeverousP6OperatorError("applicable header is outside the corpus")
            header = rows[header_ordinal]
            if (
                header.unit_type != "header_cell"
                or header.page_key != unit.page_key
                or header.table_key != unit.table_key
            ):
                raise FeverousP6OperatorError(
                    "applicable header does not point to an exact same-table header_cell"
                )
            add(CELL_TO_APPLICABLE_HEADER, unit.corpus_ordinal, header_ordinal)

    for ordinals in by_list_parent.values():
        for left, right in combinations(sorted(ordinals), 2):
            add(SAME_LIST_PARENT_PATH, left, right)

    entity_witness_count: dict[tuple[int, int], int] = {}
    for ordinals in by_entity.values():
        for pair in combinations(sorted(set(ordinals)), 2):
            entity_witness_count[pair] = entity_witness_count.get(pair, 0) + 1
    for (left, right), count in entity_witness_count.items():
        add(
            RECIPROCAL_SHARED_NORMALIZED_ENTITY,
            left,
            right,
            count * INTEGER_SCALE,
        )

    edges = tuple(
        TypedEdge(family_order, left, right, strength)
        for (family_order, left, right), strength in sorted(edge_strength.items())
    )
    neighbors: list[list[Neighbor]] = [[] for _unit in rows]
    for edge in edges:
        neighbors[edge.left_ordinal].append(
            Neighbor(edge.right_ordinal, edge.family_order, edge.strength_int)
        )
        neighbors[edge.right_ordinal].append(
            Neighbor(edge.left_ordinal, edge.family_order, edge.strength_int)
        )
    graph = TypedCorpusGraph(
        units=rows,
        edges=edges,
        neighbors=tuple(tuple(sorted(row)) for row in neighbors),
        graph_sha256="0" * 64,
    )
    graph = replace(graph, graph_sha256=recompute_graph_sha256(graph))
    return _validated_graph(graph)


@dataclass(frozen=True)
class ClaimFacet:
    facet_i: int
    facet_type: str
    normalized_text: str

    def __post_init__(self) -> None:
        _require_int(self.facet_i, "facet index", minimum=0)
        if self.facet_type not in FACET_TYPES:
            raise FeverousP6OperatorError("facet type is outside the frozen registry")
        if not self.normalized_text or normalize_key(self.normalized_text) != self.normalized_text:
            raise FeverousP6OperatorError("facet text is not canonically normalized")


def make_claim_facet(facet_i: int, facet_type: str, text: str) -> ClaimFacet:
    normalized = normalize_key(text)
    if not normalized or "\x00" in normalized:
        raise FeverousP6OperatorError("facet text is empty or contains NUL")
    return ClaimFacet(facet_i=facet_i, facet_type=facet_type, normalized_text=normalized)


@dataclass(frozen=True)
class FacetSemanticRow:
    facet_i: int
    semantic_coverage_ints: tuple[int, ...]
    direct_anchor_strength_ints: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int(self.facet_i, "semantic row facet index", minimum=0)
        if not isinstance(self.semantic_coverage_ints, tuple) or not isinstance(self.direct_anchor_strength_ints, tuple):
            raise FeverousP6OperatorError("semantic rows must use immutable tuples")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.semantic_coverage_ints):
            raise FeverousP6OperatorError("semantic coverage contains a non-integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.direct_anchor_strength_ints
        ):
            raise FeverousP6OperatorError("direct-anchor strength is not a nonnegative integer")


@dataclass(frozen=True)
class QuerySemanticTensor:
    """Complete claim semantics over every unit in the closed corpus."""

    query_sha256: str
    facets: tuple[ClaimFacet, ...]
    rows: tuple[FacetSemanticRow, ...]
    dense_relevance_ints: tuple[int, ...]
    tensor_sha256: str


def _tensor_receipt_body(tensor: QuerySemanticTensor) -> dict[str, object]:
    return {
        "dense_relevance_ints": list(tensor.dense_relevance_ints),
        "facets": [
            [facet.facet_i, facet.facet_type, facet.normalized_text]
            for facet in tensor.facets
        ],
        "integer_scale": INTEGER_SCALE,
        "query_sha256": tensor.query_sha256,
        "rows": [
            [
                row.facet_i,
                list(row.semantic_coverage_ints),
                list(row.direct_anchor_strength_ints),
            ]
            for row in tensor.rows
        ],
        "version": VERSION,
    }


def recompute_tensor_sha256(tensor: QuerySemanticTensor) -> str:
    if not isinstance(tensor, QuerySemanticTensor):
        raise FeverousP6OperatorError("query semantic tensor has the wrong type")
    return stable_hash(_tensor_receipt_body(tensor))


def _validate_facet_schema(facets: tuple[ClaimFacet, ...]) -> None:
    if not 1 <= len(facets) <= 8:
        raise FeverousP6OperatorError("claim facet count is outside one through eight")
    if tuple(facet.facet_i for facet in facets) != tuple(range(len(facets))):
        raise FeverousP6OperatorError("claim facets are not in complete source order")
    if len({facet.normalized_text for facet in facets}) != len(facets):
        raise FeverousP6OperatorError("claim facets were not deduplicated")
    counts = {kind: sum(facet.facet_type == kind for facet in facets) for kind in FACET_TYPES}
    if counts["entity"] > 4 or counts["numeric_or_date"] > 2 or counts["relation_clause"] > 2:
        raise FeverousP6OperatorError("claim facet type limit drifted")
    observed = tuple(FACET_TYPES.index(facet.facet_type) for facet in facets)
    if observed != tuple(sorted(observed)):
        raise FeverousP6OperatorError("claim facet types are outside the frozen ordering")


def _validated_tensor(
    tensor: QuerySemanticTensor, corpus_size: int
) -> QuerySemanticTensor:
    _require_sha256(tensor.query_sha256, "query hash")
    _require_sha256(tensor.tensor_sha256, "tensor hash")
    if not isinstance(tensor.facets, tuple) or not isinstance(tensor.rows, tuple) or not isinstance(tensor.dense_relevance_ints, tuple):
        raise FeverousP6OperatorError("query tensor containers must be immutable tuples")
    _validate_facet_schema(tensor.facets)
    if tuple(row.facet_i for row in tensor.rows) != tuple(range(len(tensor.facets))):
        raise FeverousP6OperatorError("semantic rows do not match claim facets")
    for row in tensor.rows:
        if len(row.semantic_coverage_ints) != corpus_size or len(row.direct_anchor_strength_ints) != corpus_size:
            raise FeverousP6OperatorError("semantic row does not cover the complete corpus")
    if len(tensor.dense_relevance_ints) != corpus_size or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in tensor.dense_relevance_ints
    ):
        raise FeverousP6OperatorError("dense relevance does not cover the complete corpus")
    if recompute_tensor_sha256(tensor) != tensor.tensor_sha256:
        raise FeverousP6OperatorError("query semantic tensor self hash drifted")
    return tensor


def make_query_semantic_tensor(
    *,
    query_sha256: str,
    facets: Sequence[ClaimFacet],
    semantic_coverage_ints: Sequence[Sequence[int]],
    direct_anchor_strength_ints: Sequence[Sequence[int]],
    dense_relevance_ints: Sequence[int],
) -> QuerySemanticTensor:
    """Construct and self-hash one complete all-corpus semantic tensor."""

    facet_rows = tuple(facets)
    coverage = tuple(tuple(row) for row in semantic_coverage_ints)
    anchors = tuple(tuple(row) for row in direct_anchor_strength_ints)
    if len(coverage) != len(facet_rows) or len(anchors) != len(facet_rows):
        raise FeverousP6OperatorError("semantic matrix row count does not match facets")
    rows = tuple(
        FacetSemanticRow(facet_i, coverage[facet_i], anchors[facet_i])
        for facet_i in range(len(facet_rows))
    )
    tensor = QuerySemanticTensor(
        query_sha256=query_sha256,
        facets=facet_rows,
        rows=rows,
        dense_relevance_ints=tuple(dense_relevance_ints),
        tensor_sha256="0" * 64,
    )
    tensor = replace(tensor, tensor_sha256=recompute_tensor_sha256(tensor))
    # The corpus size is learned only from the dense vector here; run_recipe
    # binds it independently to the graph.
    return _validated_tensor(tensor, len(tensor.dense_relevance_ints))


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    maximum_typed_path_length: int | None
    residual_budget: int


def recipe_registry() -> tuple[RecipeSpec, ...]:
    return (
        RecipeSpec("R0_DENSE5", None, 0),
        RecipeSpec("R1_P6_DIRECT_B2", 0, RESIDUAL_BUDGET),
        RecipeSpec("R2_P6_PATH1_B2", 1, RESIDUAL_BUDGET),
        RecipeSpec("R3_P6_PATH2_B2", 2, RESIDUAL_BUDGET),
    )


def _recipe_by_id(recipe_id: str) -> RecipeSpec:
    for recipe in recipe_registry():
        if recipe.recipe_id == recipe_id:
            return recipe
    raise FeverousP6OperatorError("recipe is outside the frozen four-recipe registry")


@dataclass(frozen=True)
class ReachabilityRecord:
    unit_ordinal: int
    direct_anchor: bool
    path_length: int | None
    path_strength_int: int
    anchor_facet_i: int | None
    anchor_unit_ordinal: int | None
    path_family_orders: tuple[int, ...]
    path_unit_ordinals: tuple[int, ...]


def _reachability_receipt(record: ReachabilityRecord) -> list[object]:
    return [
        record.unit_ordinal,
        record.direct_anchor,
        record.path_length,
        record.path_strength_int,
        record.anchor_facet_i,
        record.anchor_unit_ordinal,
        list(record.path_family_orders),
        list(record.path_unit_ordinals),
    ]


def _query_anchored_reachability(
    graph: TypedCorpusGraph, tensor: QuerySemanticTensor
) -> tuple[ReachabilityRecord, ...]:
    size = len(graph.units)
    best: list[ReachabilityRecord | None] = [None] * size
    for unit_ordinal in range(size):
        witnesses = [
            (row.direct_anchor_strength_ints[unit_ordinal], row.facet_i)
            for row in tensor.rows
            if row.direct_anchor_strength_ints[unit_ordinal] > 0
        ]
        if witnesses:
            strength, facet_i = max(witnesses, key=lambda value: (value[0], -value[1]))
            best[unit_ordinal] = ReachabilityRecord(
                unit_ordinal=unit_ordinal,
                direct_anchor=True,
                path_length=0,
                path_strength_int=strength,
                anchor_facet_i=facet_i,
                anchor_unit_ordinal=unit_ordinal,
                path_family_orders=(),
                path_unit_ordinals=(unit_ordinal,),
            )

    # Expand exactly two contiguous prefixes.  Each new record copies the
    # original direct anchor and its path, so no disconnected component can be
    # introduced by a corpus-only score.
    for depth in (1, 2):
        updates: dict[int, ReachabilityRecord] = {}
        for current in range(size):
            prefix = best[current]
            if prefix is None or prefix.path_length != depth - 1:
                continue
            for edge in graph.neighbors[current]:
                candidate = edge.neighbor_ordinal
                if candidate in prefix.path_unit_ordinals:
                    continue
                record = ReachabilityRecord(
                    unit_ordinal=candidate,
                    direct_anchor=False,
                    path_length=depth,
                    path_strength_int=min(prefix.path_strength_int, edge.strength_int),
                    anchor_facet_i=prefix.anchor_facet_i,
                    anchor_unit_ordinal=prefix.anchor_unit_ordinal,
                    path_family_orders=(*prefix.path_family_orders, edge.family_order),
                    path_unit_ordinals=(*prefix.path_unit_ordinals, candidate),
                )
                incumbent = best[candidate] or updates.get(candidate)
                if incumbent is None:
                    updates[candidate] = record
                    continue
                if incumbent.path_length is not None and incumbent.path_length < depth:
                    continue
                new_key = (
                    record.path_strength_int,
                    -(record.anchor_facet_i or 0),
                    tuple(-value for value in record.path_family_orders),
                    tuple(-value for value in record.path_unit_ordinals),
                )
                old_key = (
                    incumbent.path_strength_int,
                    -(incumbent.anchor_facet_i or 0),
                    tuple(-value for value in incumbent.path_family_orders),
                    tuple(-value for value in incumbent.path_unit_ordinals),
                )
                if new_key > old_key:
                    updates[candidate] = record
        for candidate, record in updates.items():
            if best[candidate] is None:
                best[candidate] = record

    return tuple(
        record
        if record is not None
        else ReachabilityRecord(
            unit_ordinal=unit_ordinal,
            direct_anchor=False,
            path_length=None,
            path_strength_int=0,
            anchor_facet_i=None,
            anchor_unit_ordinal=None,
            path_family_orders=(),
            path_unit_ordinals=(),
        )
        for unit_ordinal, record in enumerate(best)
    )


@dataclass(frozen=True)
class SelectionStep:
    output_slot: int
    selected_unit_ordinal: int
    disposition: str
    residual_facet_coverage_gain_int: int
    direct_anchor: bool
    path_length: int | None
    path_strength_int: int


@dataclass(frozen=True)
class ActionTrace:
    recipe_id: str
    output_top5: tuple[int, int, int, int, int]
    retained_raw_top3: tuple[int, int, int]
    selection_steps: tuple[SelectionStep, ...]
    raw_dense_order_sha256: str
    graph_sha256: str
    query_sha256: str
    semantic_tensor_sha256: str
    reachability_sha256: str
    candidate_scan_sha256: str
    candidate_universe_size: int
    candidate_score_evaluations: int
    semantic_cell_scan_count: int
    hipporag_candidate_or_feature_count: int
    trace_sha256: str


def _step_receipt(step: SelectionStep) -> list[object]:
    return [
        step.output_slot,
        step.selected_unit_ordinal,
        step.disposition,
        step.residual_facet_coverage_gain_int,
        step.direct_anchor,
        step.path_length,
        step.path_strength_int,
    ]


def _trace_receipt_body(trace: ActionTrace) -> dict[str, object]:
    return {
        "candidate_scan_sha256": trace.candidate_scan_sha256,
        "candidate_score_evaluations": trace.candidate_score_evaluations,
        "candidate_universe_size": trace.candidate_universe_size,
        "graph_sha256": trace.graph_sha256,
        "hipporag_candidate_or_feature_count": trace.hipporag_candidate_or_feature_count,
        "output_top5": list(trace.output_top5),
        "query_sha256": trace.query_sha256,
        "raw_dense_order_sha256": trace.raw_dense_order_sha256,
        "reachability_sha256": trace.reachability_sha256,
        "recipe_id": trace.recipe_id,
        "retained_raw_top3": list(trace.retained_raw_top3),
        "selection_steps": [_step_receipt(step) for step in trace.selection_steps],
        "semantic_cell_scan_count": trace.semantic_cell_scan_count,
        "semantic_tensor_sha256": trace.semantic_tensor_sha256,
        "version": VERSION,
    }


def recompute_action_trace_sha256(trace: ActionTrace) -> str:
    if not isinstance(trace, ActionTrace):
        raise FeverousP6OperatorError("action trace has the wrong type")
    return stable_hash(_trace_receipt_body(trace))


def verify_action_trace(trace: ActionTrace) -> str:
    _require_sha256(trace.trace_sha256, "trace hash")
    observed = recompute_action_trace_sha256(trace)
    if observed != trace.trace_sha256:
        raise FeverousP6OperatorError("action trace self hash drifted")
    if trace.recipe_id not in RECIPE_IDS:
        raise FeverousP6OperatorError("action trace recipe drifted")
    if len(trace.output_top5) != TOP_K or len(set(trace.output_top5)) != TOP_K:
        raise FeverousP6OperatorError("action trace output is not an exact top five")
    if trace.hipporag_candidate_or_feature_count != 0:
        raise FeverousP6OperatorError("HippoRAG contaminated the Agent trace")
    return observed


def _raw_dense_order(tensor: QuerySemanticTensor) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(tensor.dense_relevance_ints)),
            key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
        )
    )


def _current_facet_coverage(
    tensor: QuerySemanticTensor, selected: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        max(row.semantic_coverage_ints[ordinal] for ordinal in selected)
        for row in tensor.rows
    )


def _residual_gain(
    tensor: QuerySemanticTensor,
    current_coverage: Sequence[int],
    candidate: int,
) -> int:
    return sum(
        max(0, row.semantic_coverage_ints[candidate] - current_coverage[row.facet_i])
        for row in tensor.rows
    )


@dataclass(frozen=True)
class _PreparedRecipeInputs:
    graph: TypedCorpusGraph
    tensor: QuerySemanticTensor
    size: int
    raw_order: tuple[int, ...]
    retained: tuple[int, ...]
    raw_hash: str
    reachability: tuple[ReachabilityRecord, ...]
    reachability_hash: str


def _prepare_recipe_inputs(
    *,
    graph: TypedCorpusGraph,
    semantic_tensor: QuerySemanticTensor,
) -> _PreparedRecipeInputs:
    """Validate and derive every recipe-independent query artifact once."""

    graph = _validated_graph(graph)
    tensor = _validated_tensor(semantic_tensor, len(graph.units))
    size = len(graph.units)
    raw_order = _raw_dense_order(tensor)
    retained = raw_order[:RAW_RETAINED]
    raw_hash = stable_hash(
        {
            "corpus_size": size,
            "dense_relevance_ints": list(tensor.dense_relevance_ints),
            "raw_dense_order": list(raw_order),
        }
    )
    reachability = _query_anchored_reachability(graph, tensor)
    reachability_hash = _stream_hash(
        {"corpus_size": size, "query_sha256": tensor.query_sha256},
        (_reachability_receipt(record) for record in reachability),
    )
    return _PreparedRecipeInputs(
        graph=graph,
        tensor=tensor,
        size=size,
        raw_order=raw_order,
        retained=retained,
        raw_hash=raw_hash,
        reachability=reachability,
        reachability_hash=reachability_hash,
    )


def _execute_prepared_recipe(
    *,
    recipe: RecipeSpec,
    prepared: _PreparedRecipeInputs,
) -> ActionTrace:
    """Execute one recipe without repeating shared validation or derivation."""

    graph = prepared.graph
    tensor = prepared.tensor
    size = prepared.size
    raw_order = prepared.raw_order
    retained = prepared.retained
    raw_hash = prepared.raw_hash
    reachability = prepared.reachability
    reachability_hash = prepared.reachability_hash

    if recipe.recipe_id == "R0_DENSE5":
        output = raw_order[:TOP_K]
        raw_rank = [0] * size
        for rank, ordinal in enumerate(raw_order):
            raw_rank[ordinal] = rank
        steps = tuple(
            SelectionStep(
                output_slot=slot,
                selected_unit_ordinal=output[slot],
                disposition="raw_dense",
                residual_facet_coverage_gain_int=0,
                direct_anchor=reachability[output[slot]].direct_anchor,
                path_length=reachability[output[slot]].path_length,
                path_strength_int=reachability[output[slot]].path_strength_int,
            )
            for slot in range(TOP_K)
        )
        candidate_scan_hash = _stream_hash(
            {"recipe_id": recipe.recipe_id, "scan": "complete_dense_order"},
            (
                [ordinal, tensor.dense_relevance_ints[ordinal], raw_rank[ordinal]]
                for ordinal in range(size)
            ),
        )
        candidate_evaluations = size
    else:
        assert recipe.maximum_typed_path_length is not None
        selected = list(retained)
        step_rows: list[SelectionStep] = []
        scan_hasher = hashlib.sha256()
        scan_hasher.update(
            _canonical_bytes(
                {
                    "recipe_id": recipe.recipe_id,
                    "scan": "two_complete_residual_scans",
                }
            )
        )
        for residual_slot in range(recipe.residual_budget):
            current = _current_facet_coverage(tensor, selected)
            ranked: list[tuple[tuple[int, int, int, int, int, int], int, int]] = []
            for ordinal in range(size):
                record = reachability[ordinal]
                reachable = (
                    ordinal not in selected
                    and record.path_length is not None
                    and record.path_length <= recipe.maximum_typed_path_length
                )
                gain = _residual_gain(tensor, current, ordinal) if ordinal not in selected else 0
                path_length_key = -(record.path_length if record.path_length is not None else size + 1)
                rank_key = (
                    gain,
                    int(record.direct_anchor),
                    path_length_key,
                    record.path_strength_int,
                    tensor.dense_relevance_ints[ordinal],
                    -ordinal,
                )
                scan_hasher.update(b"\n")
                scan_hasher.update(
                    _canonical_bytes(
                        [
                            residual_slot,
                            ordinal,
                            ordinal in selected,
                            reachable,
                            gain,
                            int(record.direct_anchor),
                            record.path_length,
                            record.path_strength_int,
                            tensor.dense_relevance_ints[ordinal],
                        ]
                    )
                )
                if reachable:
                    ranked.append((rank_key, ordinal, gain))
            positive = [row for row in ranked if row[2] > 0]
            if positive:
                _key, chosen, gain = max(positive, key=lambda row: row[0])
                record = reachability[chosen]
                disposition = "query_anchored_residual"
            else:
                chosen = next(ordinal for ordinal in raw_order if ordinal not in selected)
                gain = 0
                record = reachability[chosen]
                disposition = "unused_raw_fallback"
            selected.append(chosen)
            step_rows.append(
                SelectionStep(
                    output_slot=RAW_RETAINED + residual_slot,
                    selected_unit_ordinal=chosen,
                    disposition=disposition,
                    residual_facet_coverage_gain_int=gain,
                    direct_anchor=record.direct_anchor,
                    path_length=record.path_length,
                    path_strength_int=record.path_strength_int,
                )
            )
        output = tuple(
            sorted(
                selected,
                key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
            )
        )
        steps = tuple(step_rows)
        candidate_scan_hash = scan_hasher.hexdigest()
        candidate_evaluations = size * recipe.residual_budget

    if len(output) != TOP_K or len(set(output)) != TOP_K:
        raise FeverousP6OperatorError("recipe failed to produce five unique units")
    trace = ActionTrace(
        recipe_id=recipe.recipe_id,
        output_top5=(output[0], output[1], output[2], output[3], output[4]),
        retained_raw_top3=(retained[0], retained[1], retained[2]),
        selection_steps=steps,
        raw_dense_order_sha256=raw_hash,
        graph_sha256=graph.graph_sha256,
        query_sha256=tensor.query_sha256,
        semantic_tensor_sha256=tensor.tensor_sha256,
        reachability_sha256=reachability_hash,
        candidate_scan_sha256=candidate_scan_hash,
        candidate_universe_size=size,
        candidate_score_evaluations=candidate_evaluations,
        semantic_cell_scan_count=len(tensor.facets) * size,
        hipporag_candidate_or_feature_count=0,
        trace_sha256="0" * 64,
    )
    trace = replace(trace, trace_sha256=recompute_action_trace_sha256(trace))
    verify_action_trace(trace)
    return trace


def run_recipe(
    *,
    recipe_id: str,
    graph: TypedCorpusGraph,
    semantic_tensor: QuerySemanticTensor,
) -> ActionTrace:
    """Execute one recipe from a complete graph/tensor, with no Hippo input."""

    recipe = _recipe_by_id(recipe_id)
    prepared = _prepare_recipe_inputs(
        graph=graph,
        semantic_tensor=semantic_tensor,
    )
    return _execute_prepared_recipe(recipe=recipe, prepared=prepared)


def run_all_recipes(
    *, graph: TypedCorpusGraph, semantic_tensor: QuerySemanticTensor
) -> tuple[ActionTrace, ...]:
    """Return the complete four-recipe action matrix in registry order."""

    prepared = _prepare_recipe_inputs(
        graph=graph,
        semantic_tensor=semantic_tensor,
    )
    return tuple(
        _execute_prepared_recipe(recipe=recipe, prepared=prepared)
        for recipe in recipe_registry()
    )


# Narrow compatibility names for sibling formal runners.
execute_recipe = run_recipe
execute_all_recipes = run_all_recipes


__all__ = [
    "ActionTrace",
    "AtomicUnit",
    "CELL_TO_APPLICABLE_HEADER",
    "ClaimFacet",
    "CORPUS_UNIT_COUNT",
    "EDGE_FAMILIES",
    "ENTITY_TYPES",
    "EntityKey",
    "FACET_TYPES",
    "FacetSemanticRow",
    "FeverousP6OperatorError",
    "INTEGER_SCALE",
    "Neighbor",
    "QuerySemanticTensor",
    "RECIPE_IDS",
    "RAW_RETAINED",
    "RECIPROCAL_SHARED_NORMALIZED_ENTITY",
    "RESIDUAL_BUDGET",
    "ReachabilityRecord",
    "RecipeSpec",
    "SAME_LIST_PARENT_PATH",
    "SAME_PAGE_ADJACENT_OFFICIAL_ORDER",
    "SAME_TABLE_ROW",
    "SelectionStep",
    "TOP_K",
    "TypedCorpusGraph",
    "TypedEdge",
    "UNIT_TYPES",
    "VERSION",
    "build_typed_graph",
    "execute_all_recipes",
    "execute_recipe",
    "make_claim_facet",
    "make_entity_key",
    "make_query_semantic_tensor",
    "normalize_key",
    "recipe_registry",
    "recompute_action_trace_sha256",
    "recompute_graph_sha256",
    "recompute_tensor_sha256",
    "run_all_recipes",
    "run_recipe",
    "stable_hash",
    "verify_action_trace",
]
