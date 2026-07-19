"""Dataset-free HybridQA query-anchored residual typed operator.

The operator accepts only an immutable 609-unit typed corpus graph and a
complete, caller-injected, quantized query-facet-by-corpus tensor.  It has no
dataset reader, filesystem path, network client, labels, answer nodes,
evaluator output, or HippoRAG candidate input.  Every P6 recipe retains the
same dense top three, scans all 609 units for each of two residual slots, and
admits typed-path candidates only when they descend from an injected direct
query anchor.  A disconnected corpus component therefore cannot win through
query-independent graph density.

The public deletion and exact-same-type replacement helpers expose only
ordinal interventions over already validated graph/tensor objects.  They are
sufficient for a sibling runner to build causal action features without
weakening the operator contract or introducing a hidden shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import json
import re
import unicodedata
from typing import Iterable, Sequence


VERSION = "hybridqa_query_anchored_operator_v1"
TOP_K = 5
RAW_RETAINED = 3
RESIDUAL_BUDGET = 2
INTEGER_SCALE = 1_000_000
CORPUS_UNIT_COUNT = 609

UNIT_TYPES = ("table_row", "linked_passage")
FACET_TYPES = ("entity", "numeric_or_date", "relation_clause")

SAME_TABLE_ADJACENT_ROW = "same_table_adjacent_row"
ROW_TO_LINKED_PASSAGE = "row_to_linked_passage"
SAME_TABLE_SHARED_LINK_TARGET = "same_table_shared_link_target"
EDGE_FAMILIES = (
    SAME_TABLE_ADJACENT_ROW,
    ROW_TO_LINKED_PASSAGE,
    SAME_TABLE_SHARED_LINK_TARGET,
)
EDGE_FAMILY_ORDER = {family: index for index, family in enumerate(EDGE_FAMILIES)}

RECIPE_IDS = (
    "R0_DENSE5",
    "R1_P6_DIRECT_B2",
    "R2_P6_PATH1_B2",
    "R3_P6_PATH2_B2",
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HybridQaOperatorError(ValueError):
    """An input or self-hashed output violates the frozen operator contract."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HybridQaOperatorError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stream_hash(header: object, rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    for row in rows:
        digest.update(b"\n")
        digest.update(_canonical_bytes(row))
    return digest.hexdigest()


def normalize_key(value: str) -> str:
    """Return the frozen NFKC/space-collapsed/case-folded query identity."""

    if not isinstance(value, str):
        raise TypeError("identity text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HybridQaOperatorError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise HybridQaOperatorError(f"{field} is below its minimum")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise HybridQaOperatorError(f"{field} is not a lowercase SHA-256")
    return value


def _require_key(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise HybridQaOperatorError(f"{field} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise HybridQaOperatorError(f"{field} is invalid Unicode") from exc
    return value


@dataclass(frozen=True)
class AtomicUnit:
    """One text-free public-sidecar unit in the fixed closed corpus.

    A ``table_row`` has an exact table key, nonnegative source row ordinal,
    and a canonical set of exact link-target keys.  A ``linked_passage`` has
    the same originating table key, no row ordinal, and exactly one target
    key.  Corpus ordinal is the sole action identity.
    """

    corpus_ordinal: int
    unit_type: str
    table_key: str
    row_ordinal: int | None
    link_target_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_int(self.corpus_ordinal, "corpus ordinal", minimum=0)
        if self.unit_type not in UNIT_TYPES:
            raise HybridQaOperatorError("atomic unit type is outside the registry")
        _require_key(self.table_key, "table key")
        if not isinstance(self.link_target_keys, tuple):
            raise HybridQaOperatorError("link target keys must be an immutable tuple")
        for value in self.link_target_keys:
            _require_key(value, "link target key")
        if tuple(sorted(set(self.link_target_keys))) != self.link_target_keys:
            raise HybridQaOperatorError("link target keys are not a canonical set")
        if self.unit_type == "table_row":
            _require_int(self.row_ordinal, "row ordinal", minimum=0)
        else:
            if self.row_ordinal is not None:
                raise HybridQaOperatorError(
                    "a linked passage must not declare a row ordinal"
                )
            if len(self.link_target_keys) != 1:
                raise HybridQaOperatorError(
                    "a linked passage requires exactly one target key"
                )


@dataclass(frozen=True, order=True)
class TypedEdge:
    family_order: int
    left_ordinal: int
    right_ordinal: int
    strength_int: int

    @property
    def family(self) -> str:
        if self.family_order not in range(len(EDGE_FAMILIES)):
            raise HybridQaOperatorError("typed edge family order is invalid")
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
        unit.table_key,
        unit.row_ordinal,
        list(unit.link_target_keys),
    ]


def _graph_receipt_body(graph: TypedCorpusGraph) -> dict[str, object]:
    return {
        "edges": [list(edge.public_tuple()) for edge in graph.edges],
        "units": [_unit_receipt(unit) for unit in graph.units],
        "version": VERSION,
    }


def recompute_graph_sha256(graph: TypedCorpusGraph) -> str:
    if not isinstance(graph, TypedCorpusGraph):
        raise HybridQaOperatorError("graph has the wrong type")
    return stable_hash(_graph_receipt_body(graph))


def _derived_edges(rows: tuple[AtomicUnit, ...]) -> tuple[TypedEdge, ...]:
    by_table_rows: dict[str, list[AtomicUnit]] = {}
    passage_by_target: dict[tuple[str, str], int] = {}
    rows_by_target: dict[tuple[str, str], list[int]] = {}
    for unit in rows:
        if unit.unit_type == "table_row":
            by_table_rows.setdefault(unit.table_key, []).append(unit)
            for target in unit.link_target_keys:
                rows_by_target.setdefault((unit.table_key, target), []).append(
                    unit.corpus_ordinal
                )
        else:
            target = unit.link_target_keys[0]
            key = (unit.table_key, target)
            if key in passage_by_target:
                raise HybridQaOperatorError(
                    "linked passage target is duplicated within a table"
                )
            passage_by_target[key] = unit.corpus_ordinal

    edge_strength: dict[tuple[int, int, int], int] = {}

    def add(family: str, left: int, right: int, strength: int) -> None:
        if left == right:
            return
        left, right = sorted((left, right))
        key = (EDGE_FAMILY_ORDER[family], left, right)
        edge_strength[key] = max(edge_strength.get(key, 0), strength)

    for table_rows in by_table_rows.values():
        ordered = sorted(
            table_rows,
            key=lambda unit: (unit.row_ordinal, unit.corpus_ordinal),
        )
        ordinals = [unit.row_ordinal for unit in ordered]
        if len(ordinals) != len(set(ordinals)):
            raise HybridQaOperatorError("row ordinal is duplicated within a table")
        for left, right in zip(ordered, ordered[1:]):
            assert left.row_ordinal is not None and right.row_ordinal is not None
            if right.row_ordinal == left.row_ordinal + 1:
                add(
                    SAME_TABLE_ADJACENT_ROW,
                    left.corpus_ordinal,
                    right.corpus_ordinal,
                    INTEGER_SCALE,
                )

    for target_key, row_ordinals in rows_by_target.items():
        passage_ordinal = passage_by_target.get(target_key)
        if passage_ordinal is not None:
            for row_ordinal in sorted(set(row_ordinals)):
                add(
                    ROW_TO_LINKED_PASSAGE,
                    row_ordinal,
                    passage_ordinal,
                    INTEGER_SCALE,
                )

    shared_witness_count: dict[tuple[int, int], int] = {}
    for row_ordinals in rows_by_target.values():
        for pair in combinations(sorted(set(row_ordinals)), 2):
            shared_witness_count[pair] = shared_witness_count.get(pair, 0) + 1
    for (left, right), witness_count in shared_witness_count.items():
        add(
            SAME_TABLE_SHARED_LINK_TARGET,
            left,
            right,
            witness_count * INTEGER_SCALE,
        )

    return tuple(
        TypedEdge(family_order, left, right, strength)
        for (family_order, left, right), strength in sorted(edge_strength.items())
    )


def _neighbors_from_edges(
    edges: Sequence[TypedEdge], size: int
) -> tuple[tuple[Neighbor, ...], ...]:
    rows: list[list[Neighbor]] = [[] for _ in range(size)]
    for edge in edges:
        rows[edge.left_ordinal].append(
            Neighbor(edge.right_ordinal, edge.family_order, edge.strength_int)
        )
        rows[edge.right_ordinal].append(
            Neighbor(edge.left_ordinal, edge.family_order, edge.strength_int)
        )
    return tuple(tuple(sorted(row)) for row in rows)


def _validated_graph(graph: TypedCorpusGraph) -> TypedCorpusGraph:
    if not isinstance(graph, TypedCorpusGraph):
        raise HybridQaOperatorError("graph has the wrong type")
    if (
        not isinstance(graph.units, tuple)
        or not isinstance(graph.edges, tuple)
        or not isinstance(graph.neighbors, tuple)
    ):
        raise HybridQaOperatorError("graph containers must be immutable tuples")
    if len(graph.units) != CORPUS_UNIT_COUNT:
        raise HybridQaOperatorError("closed corpus is not exactly 609 units")
    if any(not isinstance(unit, AtomicUnit) for unit in graph.units):
        raise HybridQaOperatorError("closed corpus contains a non-atomic unit")
    if tuple(unit.corpus_ordinal for unit in graph.units) != tuple(
        range(CORPUS_UNIT_COUNT)
    ):
        raise HybridQaOperatorError("corpus ordinals are not complete source order")
    if len(graph.neighbors) != CORPUS_UNIT_COUNT or any(
        not isinstance(row, tuple) for row in graph.neighbors
    ):
        raise HybridQaOperatorError("graph neighbor matrix is malformed")
    _require_sha256(graph.graph_sha256, "graph hash")
    if recompute_graph_sha256(graph) != graph.graph_sha256:
        raise HybridQaOperatorError("graph self hash drifted")
    previous: TypedEdge | None = None
    for edge in graph.edges:
        if not isinstance(edge, TypedEdge):
            raise HybridQaOperatorError("graph contains a non-typed edge")
        if previous is not None and not previous < edge:
            raise HybridQaOperatorError("typed edges are not a strict canonical set")
        previous = edge
        if edge.family_order not in range(len(EDGE_FAMILIES)):
            raise HybridQaOperatorError("typed edge family order is invalid")
        if not 0 <= edge.left_ordinal < edge.right_ordinal < CORPUS_UNIT_COUNT:
            raise HybridQaOperatorError("typed edge endpoints are invalid")
        _require_int(edge.strength_int, "typed edge strength", minimum=1)
    expected_edges = _derived_edges(graph.units)
    if graph.edges != expected_edges:
        raise HybridQaOperatorError("typed edges do not match unit sidecars")
    if graph.neighbors != _neighbors_from_edges(expected_edges, CORPUS_UNIT_COUNT):
        raise HybridQaOperatorError("graph neighbor matrix does not match typed edges")
    return graph


def build_typed_graph(units: Sequence[AtomicUnit]) -> TypedCorpusGraph:
    """Build the exact immutable three-family graph without unit text."""

    rows = tuple(units)
    if len(rows) != CORPUS_UNIT_COUNT:
        raise HybridQaOperatorError("closed corpus is not exactly 609 units")
    if any(not isinstance(unit, AtomicUnit) for unit in rows):
        raise HybridQaOperatorError("closed corpus contains a non-atomic unit")
    if tuple(unit.corpus_ordinal for unit in rows) != tuple(
        range(CORPUS_UNIT_COUNT)
    ):
        raise HybridQaOperatorError("corpus ordinals are not complete source order")
    edges = _derived_edges(rows)
    graph = TypedCorpusGraph(
        units=rows,
        edges=edges,
        neighbors=_neighbors_from_edges(edges, CORPUS_UNIT_COUNT),
        graph_sha256="0" * 64,
    )
    graph = replace(graph, graph_sha256=recompute_graph_sha256(graph))
    return _validated_graph(graph)


def verify_typed_graph(graph: TypedCorpusGraph) -> str:
    return _validated_graph(graph).graph_sha256


@dataclass(frozen=True)
class QueryFacet:
    facet_i: int
    facet_type: str
    normalized_text: str

    def __post_init__(self) -> None:
        _require_int(self.facet_i, "facet index", minimum=0)
        if self.facet_type not in FACET_TYPES:
            raise HybridQaOperatorError("facet type is outside the frozen registry")
        if (
            not self.normalized_text
            or normalize_key(self.normalized_text) != self.normalized_text
            or "\x00" in self.normalized_text
        ):
            raise HybridQaOperatorError("facet text is not canonically normalized")


def make_query_facet(facet_i: int, facet_type: str, text: str) -> QueryFacet:
    normalized = normalize_key(text)
    if not normalized or "\x00" in normalized:
        raise HybridQaOperatorError("facet text is empty or contains NUL")
    return QueryFacet(facet_i=facet_i, facet_type=facet_type, normalized_text=normalized)


@dataclass(frozen=True)
class FacetSemanticRow:
    facet_i: int
    semantic_coverage_ints: tuple[int, ...]
    direct_anchor_strength_ints: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int(self.facet_i, "semantic row facet index", minimum=0)
        if (
            not isinstance(self.semantic_coverage_ints, tuple)
            or not isinstance(self.direct_anchor_strength_ints, tuple)
        ):
            raise HybridQaOperatorError("semantic rows must use immutable tuples")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.semantic_coverage_ints
        ):
            raise HybridQaOperatorError("semantic coverage contains a non-integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.direct_anchor_strength_ints
        ):
            raise HybridQaOperatorError(
                "direct-anchor strength is not a nonnegative integer"
            )


@dataclass(frozen=True)
class QuerySemanticTensor:
    """Complete injected query semantics over all 609 closed-corpus units."""

    query_sha256: str
    facets: tuple[QueryFacet, ...]
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
        raise HybridQaOperatorError("query semantic tensor has the wrong type")
    return stable_hash(_tensor_receipt_body(tensor))


def _validate_facet_schema(facets: tuple[QueryFacet, ...]) -> None:
    if not 1 <= len(facets) <= 8:
        raise HybridQaOperatorError("query facet count is outside one through eight")
    if any(not isinstance(facet, QueryFacet) for facet in facets):
        raise HybridQaOperatorError("query facets contain a wrong type")
    if tuple(facet.facet_i for facet in facets) != tuple(range(len(facets))):
        raise HybridQaOperatorError("query facets are not in complete source order")
    if len({facet.normalized_text for facet in facets}) != len(facets):
        raise HybridQaOperatorError("query facets were not deduplicated")
    counts = {
        kind: sum(facet.facet_type == kind for facet in facets)
        for kind in FACET_TYPES
    }
    if (
        counts["entity"] > 4
        or counts["numeric_or_date"] > 2
        or counts["relation_clause"] > 2
    ):
        raise HybridQaOperatorError("query facet type limit drifted")
    observed = tuple(FACET_TYPES.index(facet.facet_type) for facet in facets)
    if observed != tuple(sorted(observed)):
        raise HybridQaOperatorError("query facet types are outside frozen ordering")


def _validated_tensor(tensor: QuerySemanticTensor) -> QuerySemanticTensor:
    if not isinstance(tensor, QuerySemanticTensor):
        raise HybridQaOperatorError("query semantic tensor has the wrong type")
    _require_sha256(tensor.query_sha256, "query hash")
    _require_sha256(tensor.tensor_sha256, "tensor hash")
    if (
        not isinstance(tensor.facets, tuple)
        or not isinstance(tensor.rows, tuple)
        or not isinstance(tensor.dense_relevance_ints, tuple)
    ):
        raise HybridQaOperatorError("query tensor containers must be immutable tuples")
    _validate_facet_schema(tensor.facets)
    if any(not isinstance(row, FacetSemanticRow) for row in tensor.rows):
        raise HybridQaOperatorError("semantic rows contain a wrong type")
    if tuple(row.facet_i for row in tensor.rows) != tuple(
        range(len(tensor.facets))
    ):
        raise HybridQaOperatorError("semantic rows do not match query facets")
    for row in tensor.rows:
        if (
            len(row.semantic_coverage_ints) != CORPUS_UNIT_COUNT
            or len(row.direct_anchor_strength_ints) != CORPUS_UNIT_COUNT
        ):
            raise HybridQaOperatorError(
                "semantic row does not cover the complete 609-unit corpus"
            )
    if len(tensor.dense_relevance_ints) != CORPUS_UNIT_COUNT or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in tensor.dense_relevance_ints
    ):
        raise HybridQaOperatorError(
            "dense relevance does not cover the complete 609-unit corpus"
        )
    if recompute_tensor_sha256(tensor) != tensor.tensor_sha256:
        raise HybridQaOperatorError("query semantic tensor self hash drifted")
    return tensor


def make_query_semantic_tensor(
    *,
    query_sha256: str,
    facets: Sequence[QueryFacet],
    semantic_coverage_ints: Sequence[Sequence[int]],
    direct_anchor_strength_ints: Sequence[Sequence[int]],
    dense_relevance_ints: Sequence[int],
) -> QuerySemanticTensor:
    """Construct and self-hash a complete all-609 query semantic tensor."""

    facet_rows = tuple(facets)
    coverage = tuple(tuple(row) for row in semantic_coverage_ints)
    anchors = tuple(tuple(row) for row in direct_anchor_strength_ints)
    if len(coverage) != len(facet_rows) or len(anchors) != len(facet_rows):
        raise HybridQaOperatorError("semantic matrix row count does not match facets")
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
    return _validated_tensor(tensor)


def verify_query_semantic_tensor(tensor: QuerySemanticTensor) -> str:
    return _validated_tensor(tensor).tensor_sha256


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
    raise HybridQaOperatorError("recipe is outside the frozen four-recipe registry")


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
    graph: TypedCorpusGraph,
    tensor: QuerySemanticTensor,
) -> tuple[ReachabilityRecord, ...]:
    best: list[ReachabilityRecord | None] = [None] * CORPUS_UNIT_COUNT
    for unit_ordinal in range(CORPUS_UNIT_COUNT):
        witnesses = [
            (row.direct_anchor_strength_ints[unit_ordinal], row.facet_i)
            for row in tensor.rows
            if row.direct_anchor_strength_ints[unit_ordinal] > 0
        ]
        if witnesses:
            strength, facet_i = max(
                witnesses,
                key=lambda value: (value[0], -value[1]),
            )
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

    for depth in (1, 2):
        updates: dict[int, ReachabilityRecord] = {}
        for current in range(CORPUS_UNIT_COUNT):
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
                incumbent = best[candidate]
                if incumbent is not None and (
                    incumbent.path_length is not None
                    and incumbent.path_length < depth
                ):
                    continue
                incumbent = incumbent or updates.get(candidate)
                if incumbent is None:
                    updates[candidate] = record
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
        "hipporag_candidate_or_feature_count": (
            trace.hipporag_candidate_or_feature_count
        ),
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
        raise HybridQaOperatorError("action trace has the wrong type")
    return stable_hash(_trace_receipt_body(trace))


def verify_action_trace(trace: ActionTrace) -> str:
    if not isinstance(trace, ActionTrace):
        raise HybridQaOperatorError("action trace has the wrong type")
    for value, field in (
        (trace.trace_sha256, "trace hash"),
        (trace.raw_dense_order_sha256, "raw dense order hash"),
        (trace.graph_sha256, "graph hash"),
        (trace.query_sha256, "query hash"),
        (trace.semantic_tensor_sha256, "semantic tensor hash"),
        (trace.reachability_sha256, "reachability hash"),
        (trace.candidate_scan_sha256, "candidate scan hash"),
    ):
        _require_sha256(value, field)
    if recompute_action_trace_sha256(trace) != trace.trace_sha256:
        raise HybridQaOperatorError("action trace self hash drifted")
    if trace.recipe_id not in RECIPE_IDS:
        raise HybridQaOperatorError("action trace recipe drifted")
    if (
        not isinstance(trace.output_top5, tuple)
        or len(trace.output_top5) != TOP_K
        or len(set(trace.output_top5)) != TOP_K
        or any(
            type(ordinal) is not int
            or not 0 <= ordinal < CORPUS_UNIT_COUNT
            for ordinal in trace.output_top5
        )
    ):
        raise HybridQaOperatorError("action trace output is not an exact top five")
    if (
        not isinstance(trace.retained_raw_top3, tuple)
        or len(trace.retained_raw_top3) != RAW_RETAINED
        or len(set(trace.retained_raw_top3)) != RAW_RETAINED
        or not set(trace.retained_raw_top3).issubset(trace.output_top5)
    ):
        raise HybridQaOperatorError("retained raw top three drifted")
    if not isinstance(trace.selection_steps, tuple) or any(
        not isinstance(step, SelectionStep) for step in trace.selection_steps
    ):
        raise HybridQaOperatorError("selection steps are not immutable typed rows")
    expected_slots = (
        tuple(range(TOP_K))
        if trace.recipe_id == "R0_DENSE5"
        else tuple(range(RAW_RETAINED, TOP_K))
    )
    if tuple(step.output_slot for step in trace.selection_steps) != expected_slots:
        raise HybridQaOperatorError("selection step slots drifted")
    if any(
        step.selected_unit_ordinal not in trace.output_top5
        or step.disposition
        not in {"raw_dense", "query_anchored_residual", "unused_raw_fallback"}
        or type(step.residual_facet_coverage_gain_int) is not int
        or step.residual_facet_coverage_gain_int < 0
        or type(step.direct_anchor) is not bool
        or (
            step.path_length is not None
            and (
                type(step.path_length) is not int
                or not 0 <= step.path_length <= 2
            )
        )
        or type(step.path_strength_int) is not int
        or step.path_strength_int < 0
        for step in trace.selection_steps
    ):
        raise HybridQaOperatorError("selection step content drifted")
    selected_step_ordinals = tuple(
        step.selected_unit_ordinal for step in trace.selection_steps
    )
    if trace.recipe_id == "R0_DENSE5":
        if selected_step_ordinals != trace.output_top5 or any(
            step.disposition != "raw_dense"
            or step.residual_facet_coverage_gain_int != 0
            for step in trace.selection_steps
        ):
            raise HybridQaOperatorError("dense action steps drifted")
    else:
        recipe = _recipe_by_id(trace.recipe_id)
        assert recipe.maximum_typed_path_length is not None
        if (
            len(set(selected_step_ordinals)) != RESIDUAL_BUDGET
            or set(selected_step_ordinals)
            != set(trace.output_top5).difference(trace.retained_raw_top3)
            or any(
                (
                    step.disposition == "query_anchored_residual"
                    and (
                        step.residual_facet_coverage_gain_int <= 0
                        or step.path_length is None
                        or step.path_length > recipe.maximum_typed_path_length
                    )
                )
                or (
                    step.disposition == "unused_raw_fallback"
                    and step.residual_facet_coverage_gain_int != 0
                )
                or step.disposition == "raw_dense"
                for step in trace.selection_steps
            )
        ):
            raise HybridQaOperatorError("residual action steps drifted")
    if any(
        (step.direct_anchor and step.path_length != 0)
        or (step.path_length == 0 and not step.direct_anchor)
        or (step.path_length is None and step.path_strength_int != 0)
        or (step.path_length is not None and step.path_strength_int <= 0)
        for step in trace.selection_steps
    ):
        raise HybridQaOperatorError("selection step reachability drifted")
    expected_evaluations = (
        CORPUS_UNIT_COUNT
        if trace.recipe_id == "R0_DENSE5"
        else CORPUS_UNIT_COUNT * RESIDUAL_BUDGET
    )
    if (
        trace.candidate_universe_size != CORPUS_UNIT_COUNT
        or trace.candidate_score_evaluations != expected_evaluations
        or type(trace.semantic_cell_scan_count) is not int
        or trace.semantic_cell_scan_count < CORPUS_UNIT_COUNT
        or trace.semantic_cell_scan_count % CORPUS_UNIT_COUNT != 0
    ):
        raise HybridQaOperatorError("action trace complete-scan counts drifted")
    if trace.hipporag_candidate_or_feature_count != 0:
        raise HybridQaOperatorError("HippoRAG contaminated the Agent trace")
    return trace.trace_sha256


def _raw_dense_order(tensor: QuerySemanticTensor) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(CORPUS_UNIT_COUNT),
            key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
        )
    )


def _current_facet_coverage(
    tensor: QuerySemanticTensor,
    selected: Sequence[int],
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
    graph = _validated_graph(graph)
    tensor = _validated_tensor(semantic_tensor)
    raw_order = _raw_dense_order(tensor)
    retained = raw_order[:RAW_RETAINED]
    raw_hash = stable_hash(
        {
            "corpus_size": CORPUS_UNIT_COUNT,
            "dense_relevance_ints": list(tensor.dense_relevance_ints),
            "raw_dense_order": list(raw_order),
        }
    )
    reachability = _query_anchored_reachability(graph, tensor)
    reachability_hash = _stream_hash(
        {
            "corpus_size": CORPUS_UNIT_COUNT,
            "graph_sha256": graph.graph_sha256,
            "query_sha256": tensor.query_sha256,
            "semantic_tensor_sha256": tensor.tensor_sha256,
        },
        (_reachability_receipt(record) for record in reachability),
    )
    return _PreparedRecipeInputs(
        graph=graph,
        tensor=tensor,
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
    graph = prepared.graph
    tensor = prepared.tensor
    raw_order = prepared.raw_order
    retained = prepared.retained
    reachability = prepared.reachability

    if recipe.recipe_id == "R0_DENSE5":
        output = raw_order[:TOP_K]
        raw_rank = [0] * CORPUS_UNIT_COUNT
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
                for ordinal in range(CORPUS_UNIT_COUNT)
            ),
        )
        candidate_evaluations = CORPUS_UNIT_COUNT
    else:
        assert recipe.maximum_typed_path_length is not None
        selected = list(retained)
        step_rows: list[SelectionStep] = []
        scan_hasher = hashlib.sha256()
        scan_hasher.update(
            _canonical_bytes(
                {
                    "recipe_id": recipe.recipe_id,
                    "scan": "two_complete_query_anchored_residual_scans",
                }
            )
        )
        for residual_slot in range(recipe.residual_budget):
            current = _current_facet_coverage(tensor, selected)
            ranked: list[tuple[tuple[int, int, int, int, int, int], int, int]] = []
            for ordinal in range(CORPUS_UNIT_COUNT):
                record = reachability[ordinal]
                already_selected = ordinal in selected
                reachable = (
                    not already_selected
                    and record.path_length is not None
                    and record.path_length <= recipe.maximum_typed_path_length
                )
                gain = (
                    _residual_gain(tensor, current, ordinal)
                    if not already_selected
                    else 0
                )
                path_length_key = -(
                    record.path_length
                    if record.path_length is not None
                    else CORPUS_UNIT_COUNT + 1
                )
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
                            already_selected,
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
                _rank, chosen, gain = max(positive, key=lambda row: row[0])
                disposition = "query_anchored_residual"
            else:
                chosen = next(
                    ordinal for ordinal in raw_order if ordinal not in selected
                )
                gain = 0
                disposition = "unused_raw_fallback"
            record = reachability[chosen]
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
        candidate_evaluations = CORPUS_UNIT_COUNT * recipe.residual_budget

    if len(output) != TOP_K or len(set(output)) != TOP_K:
        raise HybridQaOperatorError("recipe failed to produce five unique units")
    trace = ActionTrace(
        recipe_id=recipe.recipe_id,
        output_top5=(output[0], output[1], output[2], output[3], output[4]),
        retained_raw_top3=(retained[0], retained[1], retained[2]),
        selection_steps=steps,
        raw_dense_order_sha256=prepared.raw_hash,
        graph_sha256=graph.graph_sha256,
        query_sha256=tensor.query_sha256,
        semantic_tensor_sha256=tensor.tensor_sha256,
        reachability_sha256=prepared.reachability_hash,
        candidate_scan_sha256=candidate_scan_hash,
        candidate_universe_size=CORPUS_UNIT_COUNT,
        candidate_score_evaluations=candidate_evaluations,
        semantic_cell_scan_count=len(tensor.facets) * CORPUS_UNIT_COUNT,
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
    """Execute one frozen recipe from a complete graph/tensor."""

    prepared = _prepare_recipe_inputs(graph=graph, semantic_tensor=semantic_tensor)
    return _execute_prepared_recipe(
        recipe=_recipe_by_id(recipe_id),
        prepared=prepared,
    )


def run_all_recipes(
    *,
    graph: TypedCorpusGraph,
    semantic_tensor: QuerySemanticTensor,
) -> tuple[ActionTrace, ...]:
    """Execute all four recipes while reusing one complete preparation."""

    prepared = _prepare_recipe_inputs(graph=graph, semantic_tensor=semantic_tensor)
    return tuple(
        _execute_prepared_recipe(recipe=recipe, prepared=prepared)
        for recipe in recipe_registry()
    )


def _validated_action_ordinals(
    selected: Sequence[int], *, expected_count: int
) -> tuple[int, ...]:
    rows = tuple(selected)
    if (
        len(rows) != expected_count
        or len(set(rows)) != expected_count
        or any(
            type(ordinal) is not int
            or not 0 <= ordinal < CORPUS_UNIT_COUNT
            for ordinal in rows
        )
    ):
        raise HybridQaOperatorError("action ordinals are invalid")
    return rows


def deletion_action(
    selected_top5: Sequence[int], *, slot: int
) -> tuple[int, int, int, int]:
    """Delete one exact action slot while preserving the other four order."""

    selected = _validated_action_ordinals(selected_top5, expected_count=TOP_K)
    _require_int(slot, "deletion slot", minimum=0)
    if slot >= TOP_K:
        raise HybridQaOperatorError("deletion slot is outside top five")
    reduced = selected[:slot] + selected[slot + 1 :]
    return (reduced[0], reduced[1], reduced[2], reduced[3])


def same_type_replacement_candidates(
    graph: TypedCorpusGraph,
    selected_top5: Sequence[int],
    *,
    slot: int,
) -> tuple[int, ...]:
    """Return every unselected exact-same-type unit in ordinal order."""

    graph = _validated_graph(graph)
    selected = _validated_action_ordinals(selected_top5, expected_count=TOP_K)
    _require_int(slot, "replacement slot", minimum=0)
    if slot >= TOP_K:
        raise HybridQaOperatorError("replacement slot is outside top five")
    removed_type = graph.units[selected[slot]].unit_type
    selected_set = set(selected)
    return tuple(
        ordinal
        for ordinal in range(CORPUS_UNIT_COUNT)
        if ordinal not in selected_set
        and graph.units[ordinal].unit_type == removed_type
    )


def replace_action_same_type(
    graph: TypedCorpusGraph,
    selected_top5: Sequence[int],
    *,
    slot: int,
    replacement_ordinal: int,
) -> tuple[int, int, int, int, int]:
    """Apply one exact-same-type replacement without silently reordering."""

    selected = _validated_action_ordinals(selected_top5, expected_count=TOP_K)
    candidates = same_type_replacement_candidates(graph, selected, slot=slot)
    _require_int(replacement_ordinal, "replacement ordinal", minimum=0)
    if replacement_ordinal not in candidates:
        raise HybridQaOperatorError(
            "replacement is not an unselected exact-same-type unit"
        )
    output = list(selected)
    output[slot] = replacement_ordinal
    return (output[0], output[1], output[2], output[3], output[4])


def facet_maxima_ints(
    semantic_tensor: QuerySemanticTensor,
    selected_ordinals: Sequence[int],
) -> tuple[int, ...]:
    """Return exact facet maxima for a deletion/replacement action."""

    tensor = _validated_tensor(semantic_tensor)
    selected = tuple(selected_ordinals)
    if not selected or len(selected) != len(set(selected)) or any(
        type(ordinal) is not int
        or not 0 <= ordinal < CORPUS_UNIT_COUNT
        for ordinal in selected
    ):
        raise HybridQaOperatorError("feature action ordinals are invalid")
    return _current_facet_coverage(tensor, selected)


# Narrow compatibility names for future sibling runners.
execute_recipe = run_recipe
execute_all_recipes = run_all_recipes


__all__ = [
    "ActionTrace",
    "AtomicUnit",
    "CORPUS_UNIT_COUNT",
    "EDGE_FAMILIES",
    "FACET_TYPES",
    "FacetSemanticRow",
    "HybridQaOperatorError",
    "INTEGER_SCALE",
    "Neighbor",
    "QueryFacet",
    "QuerySemanticTensor",
    "RAW_RETAINED",
    "RECIPE_IDS",
    "RESIDUAL_BUDGET",
    "ROW_TO_LINKED_PASSAGE",
    "ReachabilityRecord",
    "RecipeSpec",
    "SAME_TABLE_ADJACENT_ROW",
    "SAME_TABLE_SHARED_LINK_TARGET",
    "SelectionStep",
    "TOP_K",
    "TypedCorpusGraph",
    "TypedEdge",
    "UNIT_TYPES",
    "VERSION",
    "build_typed_graph",
    "deletion_action",
    "execute_all_recipes",
    "execute_recipe",
    "facet_maxima_ints",
    "make_query_facet",
    "make_query_semantic_tensor",
    "normalize_key",
    "recipe_registry",
    "recompute_action_trace_sha256",
    "recompute_graph_sha256",
    "recompute_tensor_sha256",
    "replace_action_same_type",
    "run_all_recipes",
    "run_recipe",
    "same_type_replacement_candidates",
    "stable_hash",
    "verify_action_trace",
    "verify_query_semantic_tensor",
    "verify_typed_graph",
]
