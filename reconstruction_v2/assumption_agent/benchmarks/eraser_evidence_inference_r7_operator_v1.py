"""Pure-offline ERASER Evidence Inference R0/R7 sentence operator.

The module has no dataset reader, filesystem access, model client, label,
rationale, classifier, or HippoRAG input.  A caller injects an immutable
sentence sidecar and a complete, quantized MiniLM tensor for the exact full
query plus the three official Intervention/Comparator/Outcome facets.

R7 derives at most eight strictly-positive semantic anchor edges per facet,
exhaustively enumerates every simple anchor-to-sentence path containing zero,
one, or two consecutive-sentence edges, and keeps one canonical path for each
reachable ``(facet, terminal)`` pair.  Five terminals are then selected by the
frozen lexicographic rule without retaining or gating on any RAW member.
Every edge used by a selected canonical witness is deleted once and R7 is
rerun from the otherwise unchanged graph and tensor.  The base behavior and
the complete causal action receipt have independent, recursively verified
SHA-256 identities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Iterable, Sequence


VERSION = "eraser_evidence_inference_r7_operator_v1"
TOP_K = 5
ANCHOR_FANOUT = 8
MAX_ADJACENCY_HOPS = 2
INTEGER_SCALE = 1_000_000

FACET_TYPES = ("INTERVENTION", "COMPARATOR", "OUTCOME")
OFFICIAL_ICO_ANCHOR = "OFFICIAL_ICO_ANCHOR"
ADJACENT_SENTENCE = "ADJACENT_SENTENCE"
EDGE_TYPES = (OFFICIAL_ICO_ANCHOR, ADJACENT_SENTENCE)
RECIPE_IDS = ("R0_DENSE5", "R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE")

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EraserR7OperatorError(ValueError):
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
        raise EraserR7OperatorError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stream_hash(header: object, rows: Iterable[object]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    for row in rows:
        digest.update(b"\n")
        digest.update(_canonical_bytes(row))
    return digest.hexdigest()


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EraserR7OperatorError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise EraserR7OperatorError(f"{field} is below its minimum")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise EraserR7OperatorError(f"{field} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SentenceUnit:
    """One nonempty official sentence line, represented without its text."""

    sentence_ordinal: int
    start_token: int
    end_token: int
    sentence_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.sentence_ordinal, "sentence ordinal", minimum=0)
        _require_int(self.start_token, "sentence start token", minimum=0)
        _require_int(self.end_token, "sentence end token", minimum=1)
        if self.end_token <= self.start_token:
            raise EraserR7OperatorError("sentence token span must be nonempty")
        _require_sha256(self.sentence_sha256, "sentence hash")


@dataclass(frozen=True)
class OfficialIcoFacet:
    facet_i: int
    facet_type: str
    value_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.facet_i, "facet index", minimum=0)
        if self.facet_i >= len(FACET_TYPES):
            raise EraserR7OperatorError("facet index is outside official ICO")
        if self.facet_type != FACET_TYPES[self.facet_i]:
            raise EraserR7OperatorError("facet type is outside official ICO order")
        _require_sha256(self.value_sha256, "facet value hash")


def make_official_ico_facets(
    *,
    intervention_sha256: str,
    comparator_sha256: str,
    outcome_sha256: str,
) -> tuple[OfficialIcoFacet, OfficialIcoFacet, OfficialIcoFacet]:
    return (
        OfficialIcoFacet(0, FACET_TYPES[0], intervention_sha256),
        OfficialIcoFacet(1, FACET_TYPES[1], comparator_sha256),
        OfficialIcoFacet(2, FACET_TYPES[2], outcome_sha256),
    )


@dataclass(frozen=True)
class FacetSemanticRow:
    facet_i: int
    similarity_ints: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int(self.facet_i, "semantic row facet index", minimum=0)
        if not isinstance(self.similarity_ints, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.similarity_ints
        ):
            raise EraserR7OperatorError("facet similarities must be an integer tuple")


@dataclass(frozen=True)
class QuerySemanticTensor:
    """Exact full-query and three-by-all-sentence quantized MiniLM tensor."""

    query_sha256: str
    facets: tuple[OfficialIcoFacet, ...]
    rows: tuple[FacetSemanticRow, ...]
    dense_relevance_ints: tuple[int, ...]
    tensor_sha256: str


def _tensor_receipt_body(tensor: QuerySemanticTensor) -> dict[str, object]:
    return {
        "dense_relevance_ints": list(tensor.dense_relevance_ints),
        "facets": [
            [facet.facet_i, facet.facet_type, facet.value_sha256]
            for facet in tensor.facets
        ],
        "integer_scale": INTEGER_SCALE,
        "query_sha256": tensor.query_sha256,
        "rows": [
            [row.facet_i, list(row.similarity_ints)] for row in tensor.rows
        ],
        "version": VERSION,
    }


def recompute_tensor_sha256(tensor: QuerySemanticTensor) -> str:
    if not isinstance(tensor, QuerySemanticTensor):
        raise EraserR7OperatorError("query semantic tensor has the wrong type")
    return stable_hash(_tensor_receipt_body(tensor))


def _validated_tensor(tensor: QuerySemanticTensor) -> QuerySemanticTensor:
    if not isinstance(tensor, QuerySemanticTensor):
        raise EraserR7OperatorError("query semantic tensor has the wrong type")
    _require_sha256(tensor.query_sha256, "query hash")
    _require_sha256(tensor.tensor_sha256, "tensor hash")
    if (
        not isinstance(tensor.facets, tuple)
        or not isinstance(tensor.rows, tuple)
        or not isinstance(tensor.dense_relevance_ints, tuple)
    ):
        raise EraserR7OperatorError("query tensor containers must be tuples")
    if len(tensor.facets) != len(FACET_TYPES) or any(
        not isinstance(facet, OfficialIcoFacet) for facet in tensor.facets
    ):
        raise EraserR7OperatorError("query tensor must contain exact official ICO facets")
    if tuple(facet.facet_i for facet in tensor.facets) != tuple(
        range(len(FACET_TYPES))
    ):
        raise EraserR7OperatorError("official ICO facets are out of order")
    if len(tensor.rows) != len(FACET_TYPES) or any(
        not isinstance(row, FacetSemanticRow) for row in tensor.rows
    ):
        raise EraserR7OperatorError("query tensor must contain three semantic rows")
    if tuple(row.facet_i for row in tensor.rows) != tuple(range(len(FACET_TYPES))):
        raise EraserR7OperatorError("semantic rows do not match official ICO facets")
    width = len(tensor.dense_relevance_ints)
    if width < TOP_K or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in tensor.dense_relevance_ints
    ):
        raise EraserR7OperatorError("full-query dense row is incomplete or malformed")
    if any(len(row.similarity_ints) != width for row in tensor.rows):
        raise EraserR7OperatorError("ICO semantic tensor does not cover every sentence")
    if recompute_tensor_sha256(tensor) != tensor.tensor_sha256:
        raise EraserR7OperatorError("query semantic tensor self hash drifted")
    return tensor


def make_query_semantic_tensor(
    *,
    query_sha256: str,
    facets: Sequence[OfficialIcoFacet],
    facet_similarity_ints: Sequence[Sequence[int]],
    dense_relevance_ints: Sequence[int],
) -> QuerySemanticTensor:
    facet_rows = tuple(facets)
    similarity_rows = tuple(tuple(row) for row in facet_similarity_ints)
    if len(similarity_rows) != len(FACET_TYPES):
        raise EraserR7OperatorError("semantic matrix must have exact I/C/O rows")
    tensor = QuerySemanticTensor(
        query_sha256=query_sha256,
        facets=facet_rows,
        rows=tuple(
            FacetSemanticRow(facet_i, similarity_rows[facet_i])
            for facet_i in range(len(FACET_TYPES))
        ),
        dense_relevance_ints=tuple(dense_relevance_ints),
        tensor_sha256="0" * 64,
    )
    tensor = replace(tensor, tensor_sha256=recompute_tensor_sha256(tensor))
    return _validated_tensor(tensor)


def verify_query_semantic_tensor(tensor: QuerySemanticTensor) -> str:
    return _validated_tensor(tensor).tensor_sha256


@dataclass(frozen=True, order=True)
class TypedEdge:
    edge_i: int
    edge_type: str
    facet_i: int | None
    left_sentence_ordinal: int | None
    right_sentence_ordinal: int
    strength_int: int

    def public_tuple(self) -> tuple[object, ...]:
        return (
            self.edge_i,
            self.edge_type,
            self.facet_i,
            self.left_sentence_ordinal,
            self.right_sentence_ordinal,
            self.strength_int,
        )


@dataclass(frozen=True, order=True)
class AdjacencyNeighbor:
    sentence_ordinal: int
    edge_i: int


@dataclass(frozen=True)
class QueryAnchoredSentenceGraph:
    units: tuple[SentenceUnit, ...]
    edges: tuple[TypedEdge, ...]
    adjacency_neighbors: tuple[tuple[AdjacencyNeighbor, ...], ...]
    semantic_tensor_sha256: str
    graph_sha256: str


def _unit_receipt(unit: SentenceUnit) -> list[object]:
    return [
        unit.sentence_ordinal,
        unit.start_token,
        unit.end_token,
        unit.sentence_sha256,
    ]


def _graph_receipt_body(graph: QueryAnchoredSentenceGraph) -> dict[str, object]:
    return {
        "edges": [list(edge.public_tuple()) for edge in graph.edges],
        "semantic_tensor_sha256": graph.semantic_tensor_sha256,
        "units": [_unit_receipt(unit) for unit in graph.units],
        "version": VERSION,
    }


def recompute_graph_sha256(graph: QueryAnchoredSentenceGraph) -> str:
    if not isinstance(graph, QueryAnchoredSentenceGraph):
        raise EraserR7OperatorError("query-anchored graph has the wrong type")
    return stable_hash(_graph_receipt_body(graph))


def _derived_edges(tensor: QuerySemanticTensor) -> tuple[TypedEdge, ...]:
    raw: list[tuple[str, int | None, int | None, int, int]] = []
    for row in tensor.rows:
        anchors = sorted(
            (
                (score, ordinal)
                for ordinal, score in enumerate(row.similarity_ints)
                if score > 0
            ),
            key=lambda value: (-value[0], value[1]),
        )[:ANCHOR_FANOUT]
        for score, ordinal in sorted(anchors, key=lambda value: value[1]):
            raw.append((OFFICIAL_ICO_ANCHOR, row.facet_i, None, ordinal, score))
    for left in range(len(tensor.dense_relevance_ints) - 1):
        raw.append(
            (ADJACENT_SENTENCE, None, left, left + 1, INTEGER_SCALE)
        )
    return tuple(
        TypedEdge(edge_i, edge_type, facet_i, left, right, strength)
        for edge_i, (edge_type, facet_i, left, right, strength) in enumerate(raw)
    )


def _adjacency_neighbors(
    edges: Sequence[TypedEdge], sentence_count: int
) -> tuple[tuple[AdjacencyNeighbor, ...], ...]:
    rows: list[list[AdjacencyNeighbor]] = [[] for _ in range(sentence_count)]
    for edge in edges:
        if edge.edge_type != ADJACENT_SENTENCE:
            continue
        assert edge.left_sentence_ordinal is not None
        rows[edge.left_sentence_ordinal].append(
            AdjacencyNeighbor(edge.right_sentence_ordinal, edge.edge_i)
        )
        rows[edge.right_sentence_ordinal].append(
            AdjacencyNeighbor(edge.left_sentence_ordinal, edge.edge_i)
        )
    return tuple(tuple(sorted(row)) for row in rows)


def _validated_graph(
    graph: QueryAnchoredSentenceGraph,
    tensor: QuerySemanticTensor,
) -> QueryAnchoredSentenceGraph:
    tensor = _validated_tensor(tensor)
    if not isinstance(graph, QueryAnchoredSentenceGraph):
        raise EraserR7OperatorError("query-anchored graph has the wrong type")
    if (
        not isinstance(graph.units, tuple)
        or not isinstance(graph.edges, tuple)
        or not isinstance(graph.adjacency_neighbors, tuple)
    ):
        raise EraserR7OperatorError("graph containers must be immutable tuples")
    if len(graph.units) != len(tensor.dense_relevance_ints):
        raise EraserR7OperatorError("graph and tensor sentence counts differ")
    if any(not isinstance(unit, SentenceUnit) for unit in graph.units):
        raise EraserR7OperatorError("graph contains a non-sentence unit")
    if tuple(unit.sentence_ordinal for unit in graph.units) != tuple(
        range(len(graph.units))
    ):
        raise EraserR7OperatorError("sentence ordinals are not complete source order")
    expected_start = 0
    for unit in graph.units:
        if unit.start_token != expected_start:
            raise EraserR7OperatorError("sentence token spans are not contiguous")
        expected_start = unit.end_token
    _require_sha256(graph.semantic_tensor_sha256, "graph tensor hash")
    _require_sha256(graph.graph_sha256, "graph hash")
    if graph.semantic_tensor_sha256 != tensor.tensor_sha256:
        raise EraserR7OperatorError("graph is bound to a different semantic tensor")
    if recompute_graph_sha256(graph) != graph.graph_sha256:
        raise EraserR7OperatorError("query-anchored graph self hash drifted")
    expected_edges = _derived_edges(tensor)
    if graph.edges != expected_edges:
        raise EraserR7OperatorError("typed graph edges drifted from the complete tensor")
    for edge in graph.edges:
        if edge.edge_type not in EDGE_TYPES:
            raise EraserR7OperatorError("typed edge family is invalid")
        if edge.edge_type == OFFICIAL_ICO_ANCHOR:
            if (
                edge.facet_i not in range(len(FACET_TYPES))
                or edge.left_sentence_ordinal is not None
                or edge.strength_int <= 0
            ):
                raise EraserR7OperatorError("official ICO anchor edge is malformed")
        elif (
            edge.facet_i is not None
            or edge.left_sentence_ordinal is None
            or edge.right_sentence_ordinal != edge.left_sentence_ordinal + 1
            or edge.strength_int != INTEGER_SCALE
        ):
            raise EraserR7OperatorError("sentence adjacency edge is malformed")
    expected_neighbors = _adjacency_neighbors(expected_edges, len(graph.units))
    if graph.adjacency_neighbors != expected_neighbors:
        raise EraserR7OperatorError("graph adjacency matrix drifted")
    return graph


def build_query_anchored_graph(
    *,
    units: Sequence[SentenceUnit],
    semantic_tensor: QuerySemanticTensor,
) -> QueryAnchoredSentenceGraph:
    tensor = _validated_tensor(semantic_tensor)
    rows = tuple(units)
    if len(rows) != len(tensor.dense_relevance_ints):
        raise EraserR7OperatorError("sentence sidecar and tensor widths differ")
    edges = _derived_edges(tensor)
    graph = QueryAnchoredSentenceGraph(
        units=rows,
        edges=edges,
        adjacency_neighbors=_adjacency_neighbors(edges, len(rows)),
        semantic_tensor_sha256=tensor.tensor_sha256,
        graph_sha256="0" * 64,
    )
    graph = replace(graph, graph_sha256=recompute_graph_sha256(graph))
    return _validated_graph(graph, tensor)


def verify_query_anchored_graph(
    graph: QueryAnchoredSentenceGraph,
    semantic_tensor: QuerySemanticTensor,
) -> str:
    return _validated_graph(graph, semantic_tensor).graph_sha256


@dataclass(frozen=True)
class AtomicPath:
    facet_i: int
    terminal_sentence_ordinal: int
    anchor_sentence_ordinal: int
    sentence_ordinals: tuple[int, ...]
    edge_ids: tuple[int, ...]
    anchor_strength_int: int
    adjacency_hop_count: int


def _path_receipt(path: AtomicPath) -> list[object]:
    return [
        path.facet_i,
        path.terminal_sentence_ordinal,
        path.anchor_sentence_ordinal,
        list(path.sentence_ordinals),
        list(path.edge_ids),
        path.anchor_strength_int,
        path.adjacency_hop_count,
    ]


def _validate_path(path: AtomicPath) -> None:
    if not isinstance(path, AtomicPath):
        raise EraserR7OperatorError("path receipt has the wrong type")
    if type(path.facet_i) is not int or path.facet_i not in range(len(FACET_TYPES)):
        raise EraserR7OperatorError("path facet is invalid")
    if (
        type(path.terminal_sentence_ordinal) is not int
        or path.terminal_sentence_ordinal < 0
        or type(path.anchor_sentence_ordinal) is not int
        or path.anchor_sentence_ordinal < 0
        or not isinstance(path.sentence_ordinals, tuple)
        or not isinstance(path.edge_ids, tuple)
        or any(type(value) is not int or value < 0 for value in path.sentence_ordinals)
        or any(type(value) is not int or value < 0 for value in path.edge_ids)
        or not 1 <= len(path.sentence_ordinals) <= MAX_ADJACENCY_HOPS + 1
        or len(set(path.sentence_ordinals)) != len(path.sentence_ordinals)
        or len(set(path.edge_ids)) != len(path.edge_ids)
        or len(path.edge_ids) != len(path.sentence_ordinals)
        or path.anchor_sentence_ordinal != path.sentence_ordinals[0]
        or path.terminal_sentence_ordinal != path.sentence_ordinals[-1]
        or type(path.adjacency_hop_count) is not int
        or path.adjacency_hop_count != len(path.sentence_ordinals) - 1
        or type(path.anchor_strength_int) is not int
        or path.anchor_strength_int <= 0
    ):
        raise EraserR7OperatorError("atomic path is malformed")


@dataclass(frozen=True)
class TerminalFacetPathMap:
    terminal_sentence_ordinal: int
    facet_paths: tuple[AtomicPath, ...]


def _terminal_map_receipt(row: TerminalFacetPathMap) -> list[object]:
    return [
        row.terminal_sentence_ordinal,
        [_path_receipt(path) for path in row.facet_paths],
    ]


def enumerate_atomic_paths(
    *,
    graph: QueryAnchoredSentenceGraph,
    semantic_tensor: QuerySemanticTensor,
    excluded_edge_i: int | None = None,
) -> tuple[AtomicPath, ...]:
    """Exhaustively enumerate all simple anchor + zero-to-two adjacency paths."""

    graph = _validated_graph(graph, semantic_tensor)
    edge_ids = {edge.edge_i for edge in graph.edges}
    if excluded_edge_i is not None:
        _require_int(excluded_edge_i, "excluded edge index", minimum=0)
        if excluded_edge_i not in edge_ids:
            raise EraserR7OperatorError("excluded edge is outside the graph")
    paths: list[AtomicPath] = []

    def visit(
        *,
        facet_i: int,
        anchor_sentence: int,
        anchor_strength: int,
        current: int,
        sentence_path: tuple[int, ...],
        path_edge_ids: tuple[int, ...],
        depth: int,
    ) -> None:
        paths.append(
            AtomicPath(
                facet_i=facet_i,
                terminal_sentence_ordinal=current,
                anchor_sentence_ordinal=anchor_sentence,
                sentence_ordinals=sentence_path,
                edge_ids=path_edge_ids,
                anchor_strength_int=anchor_strength,
                adjacency_hop_count=depth,
            )
        )
        if depth == MAX_ADJACENCY_HOPS:
            return
        for neighbor in graph.adjacency_neighbors[current]:
            if neighbor.edge_i == excluded_edge_i:
                continue
            if neighbor.sentence_ordinal in sentence_path:
                continue
            visit(
                facet_i=facet_i,
                anchor_sentence=anchor_sentence,
                anchor_strength=anchor_strength,
                current=neighbor.sentence_ordinal,
                sentence_path=(*sentence_path, neighbor.sentence_ordinal),
                path_edge_ids=(*path_edge_ids, neighbor.edge_i),
                depth=depth + 1,
            )

    for edge in graph.edges:
        if edge.edge_type != OFFICIAL_ICO_ANCHOR or edge.edge_i == excluded_edge_i:
            continue
        assert edge.facet_i is not None
        visit(
            facet_i=edge.facet_i,
            anchor_sentence=edge.right_sentence_ordinal,
            anchor_strength=edge.strength_int,
            current=edge.right_sentence_ordinal,
            sentence_path=(edge.right_sentence_ordinal,),
            path_edge_ids=(edge.edge_i,),
            depth=0,
        )
    ordered = tuple(
        sorted(
            paths,
            key=lambda path: (
                path.facet_i,
                path.terminal_sentence_ordinal,
                path.anchor_sentence_ordinal,
                path.adjacency_hop_count,
                path.sentence_ordinals,
                path.edge_ids,
            ),
        )
    )
    for path in ordered:
        _validate_path(path)
    return ordered


def canonical_facet_terminal_maps(
    paths: Sequence[AtomicPath],
) -> tuple[TerminalFacetPathMap, ...]:
    """Choose exactly one frozen best path for every reachable facet/terminal."""

    grouped: dict[tuple[int, int], list[AtomicPath]] = {}
    for path in paths:
        _validate_path(path)
        grouped.setdefault((path.terminal_sentence_ordinal, path.facet_i), []).append(
            path
        )
    by_terminal: dict[int, list[AtomicPath]] = {}
    for (terminal, _facet_i), candidates in grouped.items():
        chosen = min(
            candidates,
            key=lambda path: (
                -path.anchor_strength_int,
                path.adjacency_hop_count,
                path.sentence_ordinals,
                path.edge_ids,
            ),
        )
        by_terminal.setdefault(terminal, []).append(chosen)
    rows = tuple(
        TerminalFacetPathMap(
            terminal_sentence_ordinal=terminal,
            facet_paths=tuple(sorted(facet_paths, key=lambda path: path.facet_i)),
        )
        for terminal, facet_paths in sorted(by_terminal.items())
    )
    for row in rows:
        if (
            not row.facet_paths
            or tuple(path.facet_i for path in row.facet_paths)
            != tuple(sorted({path.facet_i for path in row.facet_paths}))
            or any(
                path.terminal_sentence_ordinal != row.terminal_sentence_ordinal
                for path in row.facet_paths
            )
        ):
            raise EraserR7OperatorError("canonical facet-to-path map is malformed")
    return rows


@dataclass(frozen=True)
class SelectionStep:
    output_slot: int
    selected_sentence_ordinal: int
    disposition: str
    newly_covered_facets: tuple[int, ...]
    reachable_facets: tuple[int, ...]
    facet_paths: tuple[AtomicPath, ...]


def _step_receipt(step: SelectionStep) -> list[object]:
    return [
        step.output_slot,
        step.selected_sentence_ordinal,
        step.disposition,
        list(step.newly_covered_facets),
        list(step.reachable_facets),
        [_path_receipt(path) for path in step.facet_paths],
    ]


@dataclass(frozen=True)
class BehaviorTrace:
    recipe_id: str
    output_top5: tuple[int, int, int, int, int]
    selection_steps: tuple[SelectionStep, ...]
    selected_facet_maxima_ints: tuple[int, int, int]
    graph_sha256: str
    query_sha256: str
    semantic_tensor_sha256: str
    raw_dense_order_sha256: str
    exhaustive_path_scan_sha256: str
    exhaustive_path_count: int
    terminal_path_map_sha256: str
    terminal_path_map_count: int
    candidate_scan_sha256: str
    candidate_score_evaluations: int
    semantic_cell_scan_count: int
    dense_fill_count: int
    hipporag_candidate_or_feature_count: int
    excluded_edge_i: int | None
    behavior_sha256: str


def _behavior_receipt_body(trace: BehaviorTrace) -> dict[str, object]:
    return {
        "behavior_sha256_schema": VERSION,
        "candidate_scan_sha256": trace.candidate_scan_sha256,
        "candidate_score_evaluations": trace.candidate_score_evaluations,
        "dense_fill_count": trace.dense_fill_count,
        "excluded_edge_i": trace.excluded_edge_i,
        "exhaustive_path_count": trace.exhaustive_path_count,
        "exhaustive_path_scan_sha256": trace.exhaustive_path_scan_sha256,
        "graph_sha256": trace.graph_sha256,
        "hipporag_candidate_or_feature_count": (
            trace.hipporag_candidate_or_feature_count
        ),
        "output_top5": list(trace.output_top5),
        "query_sha256": trace.query_sha256,
        "raw_dense_order_sha256": trace.raw_dense_order_sha256,
        "recipe_id": trace.recipe_id,
        "selected_facet_maxima_ints": list(trace.selected_facet_maxima_ints),
        "selection_steps": [_step_receipt(step) for step in trace.selection_steps],
        "semantic_cell_scan_count": trace.semantic_cell_scan_count,
        "semantic_tensor_sha256": trace.semantic_tensor_sha256,
        "terminal_path_map_count": trace.terminal_path_map_count,
        "terminal_path_map_sha256": trace.terminal_path_map_sha256,
        "version": VERSION,
    }


def recompute_behavior_sha256(trace: BehaviorTrace) -> str:
    if not isinstance(trace, BehaviorTrace):
        raise EraserR7OperatorError("behavior trace has the wrong type")
    return stable_hash(_behavior_receipt_body(trace))


def _facet_maxima(
    tensor: QuerySemanticTensor, selected: Sequence[int]
) -> tuple[int, int, int]:
    maxima = tuple(
        max(row.similarity_ints[ordinal] for ordinal in selected)
        for row in tensor.rows
    )
    return (maxima[0], maxima[1], maxima[2])


def facet_maxima_ints(
    semantic_tensor: QuerySemanticTensor,
    selected_ordinals: Sequence[int],
) -> tuple[int, int, int]:
    tensor = _validated_tensor(semantic_tensor)
    selected = tuple(selected_ordinals)
    if not selected or len(selected) != len(set(selected)) or any(
        type(ordinal) is not int
        or not 0 <= ordinal < len(tensor.dense_relevance_ints)
        for ordinal in selected
    ):
        raise EraserR7OperatorError("feature action ordinals are invalid")
    return _facet_maxima(tensor, selected)


def _raw_dense_order(tensor: QuerySemanticTensor) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(tensor.dense_relevance_ints)),
            key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
        )
    )


def _path_only_step_receipts(trace: BehaviorTrace) -> list[object]:
    return [
        [step.selected_sentence_ordinal, [_path_receipt(path) for path in step.facet_paths]]
        for step in trace.selection_steps
    ]


def run_action_core(
    *,
    recipe_id: str,
    graph: QueryAnchoredSentenceGraph,
    semantic_tensor: QuerySemanticTensor,
    excluded_edge_i: int | None = None,
) -> BehaviorTrace:
    """Execute one recipe without recursively collecting edge deletions."""

    tensor = _validated_tensor(semantic_tensor)
    graph = _validated_graph(graph, tensor)
    if recipe_id not in RECIPE_IDS:
        raise EraserR7OperatorError("recipe is outside the frozen registry")
    if recipe_id == "R0_DENSE5" and excluded_edge_i is not None:
        raise EraserR7OperatorError("R0 has no edge-deletion execution")
    if excluded_edge_i is not None:
        _require_int(excluded_edge_i, "excluded edge index", minimum=0)
        if excluded_edge_i not in {edge.edge_i for edge in graph.edges}:
            raise EraserR7OperatorError("excluded edge is outside the graph")

    sentence_count = len(graph.units)
    raw_order = _raw_dense_order(tensor)
    raw_hash = stable_hash(
        {
            "dense_relevance_ints": list(tensor.dense_relevance_ints),
            "raw_dense_order": list(raw_order),
            "sentence_count": sentence_count,
        }
    )

    if recipe_id == "R0_DENSE5":
        output = raw_order[:TOP_K]
        raw_rank = [0] * sentence_count
        for rank, ordinal in enumerate(raw_order):
            raw_rank[ordinal] = rank
        steps = tuple(
            SelectionStep(slot, ordinal, "raw_dense", (), (), ())
            for slot, ordinal in enumerate(output)
        )
        path_hash = _stream_hash(
            {"excluded_edge_i": None, "scan": "no_paths_for_R0"}, ()
        )
        map_hash = _stream_hash(
            {"excluded_edge_i": None, "scan": "no_terminal_maps_for_R0"}, ()
        )
        candidate_hash = _stream_hash(
            {"recipe_id": recipe_id, "scan": "complete_dense_order"},
            (
                [ordinal, tensor.dense_relevance_ints[ordinal], raw_rank[ordinal]]
                for ordinal in range(sentence_count)
            ),
        )
        path_count = 0
        map_count = 0
        candidate_evaluations = sentence_count
        dense_fill_count = 0
    else:
        paths = enumerate_atomic_paths(
            graph=graph,
            semantic_tensor=tensor,
            excluded_edge_i=excluded_edge_i,
        )
        maps = canonical_facet_terminal_maps(paths)
        path_hash = _stream_hash(
            {
                "excluded_edge_i": excluded_edge_i,
                "graph_sha256": graph.graph_sha256,
                "scan": "all_simple_anchor_plus_zero_to_two_adjacency_paths",
            },
            (_path_receipt(path) for path in paths),
        )
        map_hash = _stream_hash(
            {
                "excluded_edge_i": excluded_edge_i,
                "scan": "canonical_facet_terminal_maps",
            },
            (_terminal_map_receipt(row) for row in maps),
        )
        map_by_terminal = {row.terminal_sentence_ordinal: row for row in maps}
        selected: list[int] = []
        covered: set[int] = set()
        step_rows: list[SelectionStep] = []
        dense_fill_count = 0
        scan_hasher = hashlib.sha256()
        scan_hasher.update(
            _canonical_bytes(
                {
                    "excluded_edge_i": excluded_edge_i,
                    "recipe_id": recipe_id,
                    "scan": "five_complete_terminal_scans",
                }
            )
        )
        for slot in range(TOP_K):
            ranked: list[tuple[tuple[int, int, int, int, int], int, tuple[int, ...]]] = []
            for ordinal in range(sentence_count):
                terminal_map = map_by_terminal.get(ordinal)
                reachable = (
                    tuple(path.facet_i for path in terminal_map.facet_paths)
                    if terminal_map is not None
                    else ()
                )
                new_facets = tuple(facet for facet in reachable if facet not in covered)
                score_facets = new_facets or reachable
                if terminal_map is not None and score_facets:
                    path_by_facet = {
                        path.facet_i: path for path in terminal_map.facet_paths
                    }
                    minimum_strength = min(
                        path_by_facet[facet].anchor_strength_int
                        for facet in score_facets
                    )
                    minimum_hops = min(
                        path_by_facet[facet].adjacency_hop_count
                        for facet in score_facets
                    )
                else:
                    minimum_strength = 0
                    minimum_hops = MAX_ADJACENCY_HOPS + 1
                already_selected = ordinal in selected
                eligible = terminal_map is not None and not already_selected
                scan_hasher.update(b"\n")
                scan_hasher.update(
                    _canonical_bytes(
                        [
                            slot,
                            ordinal,
                            already_selected,
                            eligible,
                            list(reachable),
                            list(new_facets),
                            minimum_strength,
                            tensor.dense_relevance_ints[ordinal],
                            minimum_hops,
                        ]
                    )
                )
                if eligible:
                    rank = (
                        len(new_facets),
                        minimum_strength,
                        tensor.dense_relevance_ints[ordinal],
                        -minimum_hops,
                        -ordinal,
                    )
                    ranked.append((rank, ordinal, new_facets))
            if ranked:
                _rank, chosen, new_facets = max(ranked, key=lambda row: row[0])
                terminal_map = map_by_terminal[chosen]
                reachable = tuple(path.facet_i for path in terminal_map.facet_paths)
                facet_paths = terminal_map.facet_paths
                disposition = "query_anchored_path"
                covered.update(reachable)
            else:
                chosen = next(ordinal for ordinal in raw_order if ordinal not in selected)
                new_facets = ()
                reachable = ()
                facet_paths = ()
                disposition = "dense_fill_after_path_exhaustion"
                dense_fill_count += 1
            selected.append(chosen)
            step_rows.append(
                SelectionStep(
                    output_slot=slot,
                    selected_sentence_ordinal=chosen,
                    disposition=disposition,
                    newly_covered_facets=new_facets,
                    reachable_facets=reachable,
                    facet_paths=facet_paths,
                )
            )
        output = tuple(selected)
        steps = tuple(step_rows)
        candidate_hash = scan_hasher.hexdigest()
        path_count = len(paths)
        map_count = len(maps)
        candidate_evaluations = sentence_count * TOP_K

    if len(output) != TOP_K or len(set(output)) != TOP_K:
        raise EraserR7OperatorError("recipe failed to produce five unique sentences")
    behavior = BehaviorTrace(
        recipe_id=recipe_id,
        output_top5=(output[0], output[1], output[2], output[3], output[4]),
        selection_steps=steps,
        selected_facet_maxima_ints=_facet_maxima(tensor, output),
        graph_sha256=graph.graph_sha256,
        query_sha256=tensor.query_sha256,
        semantic_tensor_sha256=tensor.tensor_sha256,
        raw_dense_order_sha256=raw_hash,
        exhaustive_path_scan_sha256=path_hash,
        exhaustive_path_count=path_count,
        terminal_path_map_sha256=map_hash,
        terminal_path_map_count=map_count,
        candidate_scan_sha256=candidate_hash,
        candidate_score_evaluations=candidate_evaluations,
        semantic_cell_scan_count=len(FACET_TYPES) * sentence_count,
        dense_fill_count=dense_fill_count,
        hipporag_candidate_or_feature_count=0,
        excluded_edge_i=excluded_edge_i,
        behavior_sha256="0" * 64,
    )
    behavior = replace(
        behavior, behavior_sha256=recompute_behavior_sha256(behavior)
    )
    verify_behavior_trace(behavior)
    return behavior


def verify_behavior_trace(
    trace: BehaviorTrace,
    *,
    graph: QueryAnchoredSentenceGraph | None = None,
    semantic_tensor: QuerySemanticTensor | None = None,
) -> str:
    if not isinstance(trace, BehaviorTrace):
        raise EraserR7OperatorError("behavior trace has the wrong type")
    for value, field in (
        (trace.behavior_sha256, "behavior hash"),
        (trace.graph_sha256, "graph hash"),
        (trace.query_sha256, "query hash"),
        (trace.semantic_tensor_sha256, "semantic tensor hash"),
        (trace.raw_dense_order_sha256, "raw dense order hash"),
        (trace.exhaustive_path_scan_sha256, "path scan hash"),
        (trace.terminal_path_map_sha256, "terminal map hash"),
        (trace.candidate_scan_sha256, "candidate scan hash"),
    ):
        _require_sha256(value, field)
    if recompute_behavior_sha256(trace) != trace.behavior_sha256:
        raise EraserR7OperatorError("behavior trace self hash drifted")
    if trace.recipe_id not in RECIPE_IDS:
        raise EraserR7OperatorError("behavior recipe is outside the registry")
    if trace.excluded_edge_i is not None:
        _require_int(trace.excluded_edge_i, "excluded edge index", minimum=0)
    if (
        not isinstance(trace.output_top5, tuple)
        or len(trace.output_top5) != TOP_K
        or len(set(trace.output_top5)) != TOP_K
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in trace.output_top5)
    ):
        raise EraserR7OperatorError("behavior output is not an exact top five")
    if not isinstance(trace.selection_steps, tuple) or len(trace.selection_steps) != TOP_K:
        raise EraserR7OperatorError("behavior selection steps drifted")
    if any(not isinstance(step, SelectionStep) for step in trace.selection_steps):
        raise EraserR7OperatorError("selection step has the wrong type")
    if (
        tuple(step.output_slot for step in trace.selection_steps)
        != tuple(range(TOP_K))
        or tuple(step.selected_sentence_ordinal for step in trace.selection_steps)
        != trace.output_top5
    ):
        raise EraserR7OperatorError("behavior selection steps drifted")
    if (
        not isinstance(trace.selected_facet_maxima_ints, tuple)
        or len(trace.selected_facet_maxima_ints) != len(FACET_TYPES)
        or any(type(value) is not int for value in trace.selected_facet_maxima_ints)
    ):
        raise EraserR7OperatorError("behavior facet maxima are malformed")
    for step in trace.selection_steps:
        if (
            type(step.output_slot) is not int
            or type(step.selected_sentence_ordinal) is not int
            or not isinstance(step.newly_covered_facets, tuple)
            or not isinstance(step.reachable_facets, tuple)
            or not isinstance(step.facet_paths, tuple)
            or any(
                type(facet) is not int or facet not in range(len(FACET_TYPES))
                for facet in (*step.newly_covered_facets, *step.reachable_facets)
            )
            or step.newly_covered_facets
            != tuple(sorted(set(step.newly_covered_facets)))
            or step.reachable_facets != tuple(sorted(set(step.reachable_facets)))
            or not set(step.newly_covered_facets).issubset(step.reachable_facets)
        ):
            raise EraserR7OperatorError("selection facet map is malformed")
        for path in step.facet_paths:
            _validate_path(path)
        if step.disposition == "query_anchored_path":
            if (
                not step.facet_paths
                or tuple(path.facet_i for path in step.facet_paths)
                != step.reachable_facets
                or any(
                    path.terminal_sentence_ordinal != step.selected_sentence_ordinal
                    for path in step.facet_paths
                )
            ):
                raise EraserR7OperatorError("selected canonical path map drifted")
        elif step.disposition in {"raw_dense", "dense_fill_after_path_exhaustion"}:
            if step.newly_covered_facets or step.reachable_facets or step.facet_paths:
                raise EraserR7OperatorError("dense selection contains a path witness")
        else:
            raise EraserR7OperatorError("selection disposition is invalid")
    if trace.recipe_id == "R0_DENSE5":
        if (
            trace.excluded_edge_i is not None
            or any(step.disposition != "raw_dense" for step in trace.selection_steps)
            or trace.exhaustive_path_count != 0
            or trace.terminal_path_map_count != 0
            or trace.dense_fill_count != 0
        ):
            raise EraserR7OperatorError("R0 behavior semantics drifted")
    else:
        if any(step.disposition == "raw_dense" for step in trace.selection_steps):
            raise EraserR7OperatorError("R7 retained a RAW-only disposition")
        expected_fill = sum(
            step.disposition == "dense_fill_after_path_exhaustion"
            for step in trace.selection_steps
        )
        if trace.dense_fill_count != expected_fill:
            raise EraserR7OperatorError("R7 dense-fill count drifted")
    for value, field in (
        (trace.exhaustive_path_count, "path count"),
        (trace.terminal_path_map_count, "terminal map count"),
        (trace.candidate_score_evaluations, "candidate evaluations"),
        (trace.semantic_cell_scan_count, "semantic cell scans"),
        (trace.dense_fill_count, "dense fill count"),
        (trace.hipporag_candidate_or_feature_count, "HippoRAG feature count"),
    ):
        _require_int(value, field, minimum=0)
    if trace.semantic_cell_scan_count % len(FACET_TYPES) != 0:
        raise EraserR7OperatorError("semantic cell scan count is not three-by-sentence")
    sentence_count = trace.semantic_cell_scan_count // len(FACET_TYPES)
    if sentence_count < TOP_K or any(
        ordinal >= sentence_count for ordinal in trace.output_top5
    ):
        raise EraserR7OperatorError("behavior output is outside its complete sentence scan")
    expected_evaluations = (
        sentence_count
        if trace.recipe_id == "R0_DENSE5"
        else sentence_count * TOP_K
    )
    if trace.candidate_score_evaluations != expected_evaluations:
        raise EraserR7OperatorError("candidate scan count drifted")
    if trace.hipporag_candidate_or_feature_count != 0:
        raise EraserR7OperatorError("HippoRAG contaminated the Agent behavior")
    if (graph is None) != (semantic_tensor is None):
        raise EraserR7OperatorError("behavior input verification requires graph and tensor")
    if graph is not None and semantic_tensor is not None:
        expected = run_action_core(
            recipe_id=trace.recipe_id,
            graph=graph,
            semantic_tensor=semantic_tensor,
            excluded_edge_i=trace.excluded_edge_i,
        )
        if trace != expected:
            raise EraserR7OperatorError("behavior trace does not reconstruct from inputs")
    return trace.behavior_sha256


@dataclass(frozen=True)
class EdgeDeletionWitness:
    edge_i: int
    counterfactual_behavior: BehaviorTrace
    selected_ordinals_changed: bool
    witness_path_receipts_changed: bool
    ico_coverage_changed: bool
    ico_coverage_drop_ints: tuple[int, int, int]
    witness_sha256: str


def _edge_deletion_receipt_body(witness: EdgeDeletionWitness) -> dict[str, object]:
    return {
        "counterfactual_behavior": {
            **_behavior_receipt_body(witness.counterfactual_behavior),
            "behavior_sha256": witness.counterfactual_behavior.behavior_sha256,
        },
        "edge_i": witness.edge_i,
        "ico_coverage_changed": witness.ico_coverage_changed,
        "ico_coverage_drop_ints": list(witness.ico_coverage_drop_ints),
        "selected_ordinals_changed": witness.selected_ordinals_changed,
        "version": VERSION,
        "witness_path_receipts_changed": witness.witness_path_receipts_changed,
    }


def recompute_edge_deletion_witness_sha256(witness: EdgeDeletionWitness) -> str:
    if not isinstance(witness, EdgeDeletionWitness):
        raise EraserR7OperatorError("edge deletion witness has the wrong type")
    return stable_hash(_edge_deletion_receipt_body(witness))


@dataclass(frozen=True)
class ActionTrace:
    recipe_id: str
    behavior: BehaviorTrace
    used_edge_ids: tuple[int, ...]
    edge_deletion_witnesses: tuple[EdgeDeletionWitness, ...]
    trace_sha256: str

    @property
    def behavior_sha256(self) -> str:
        return self.behavior.behavior_sha256

    @property
    def output_top5(self) -> tuple[int, int, int, int, int]:
        return self.behavior.output_top5

    @property
    def selection_steps(self) -> tuple[SelectionStep, ...]:
        return self.behavior.selection_steps


def _action_receipt_body(trace: ActionTrace) -> dict[str, object]:
    return {
        "behavior": {
            **_behavior_receipt_body(trace.behavior),
            "behavior_sha256": trace.behavior.behavior_sha256,
        },
        "edge_deletion_witnesses": [
            {
                **_edge_deletion_receipt_body(witness),
                "witness_sha256": witness.witness_sha256,
            }
            for witness in trace.edge_deletion_witnesses
        ],
        "recipe_id": trace.recipe_id,
        "used_edge_ids": list(trace.used_edge_ids),
        "version": VERSION,
    }


def recompute_action_trace_sha256(trace: ActionTrace) -> str:
    if not isinstance(trace, ActionTrace):
        raise EraserR7OperatorError("action trace has the wrong type")
    return stable_hash(_action_receipt_body(trace))


def _used_edge_ids(behavior: BehaviorTrace) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                edge_i
                for step in behavior.selection_steps
                for path in step.facet_paths
                for edge_i in path.edge_ids
            }
        )
    )


def _make_edge_deletion_witness(
    *,
    edge_i: int,
    base: BehaviorTrace,
    counterfactual: BehaviorTrace,
) -> EdgeDeletionWitness:
    drop = tuple(
        base_value - counterfactual_value
        for base_value, counterfactual_value in zip(
            base.selected_facet_maxima_ints,
            counterfactual.selected_facet_maxima_ints,
        )
    )
    witness = EdgeDeletionWitness(
        edge_i=edge_i,
        counterfactual_behavior=counterfactual,
        selected_ordinals_changed=base.output_top5 != counterfactual.output_top5,
        witness_path_receipts_changed=(
            _path_only_step_receipts(base)
            != _path_only_step_receipts(counterfactual)
        ),
        ico_coverage_changed=any(value != 0 for value in drop),
        ico_coverage_drop_ints=(drop[0], drop[1], drop[2]),
        witness_sha256="0" * 64,
    )
    return replace(
        witness,
        witness_sha256=recompute_edge_deletion_witness_sha256(witness),
    )


def run_action(
    *,
    recipe_id: str,
    graph: QueryAnchoredSentenceGraph,
    semantic_tensor: QuerySemanticTensor,
) -> ActionTrace:
    """Execute R0 or R7 and collect every required used-edge counterfactual."""

    base = run_action_core(
        recipe_id=recipe_id,
        graph=graph,
        semantic_tensor=semantic_tensor,
    )
    used = _used_edge_ids(base)
    witnesses: list[EdgeDeletionWitness] = []
    if recipe_id == "R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE":
        for edge_i in used:
            counterfactual = run_action_core(
                recipe_id=recipe_id,
                graph=graph,
                semantic_tensor=semantic_tensor,
                excluded_edge_i=edge_i,
            )
            witnesses.append(
                _make_edge_deletion_witness(
                    edge_i=edge_i,
                    base=base,
                    counterfactual=counterfactual,
                )
            )
    trace = ActionTrace(
        recipe_id=recipe_id,
        behavior=base,
        used_edge_ids=used,
        edge_deletion_witnesses=tuple(witnesses),
        trace_sha256="0" * 64,
    )
    trace = replace(trace, trace_sha256=recompute_action_trace_sha256(trace))
    verify_action_trace(trace)
    return trace


def verify_action_trace(
    trace: ActionTrace,
    *,
    graph: QueryAnchoredSentenceGraph | None = None,
    semantic_tensor: QuerySemanticTensor | None = None,
) -> str:
    if not isinstance(trace, ActionTrace):
        raise EraserR7OperatorError("action trace has the wrong type")
    _require_sha256(trace.trace_sha256, "action trace hash")
    if recompute_action_trace_sha256(trace) != trace.trace_sha256:
        raise EraserR7OperatorError("action trace self hash drifted")
    verify_behavior_trace(trace.behavior)
    if trace.recipe_id != trace.behavior.recipe_id or trace.recipe_id not in RECIPE_IDS:
        raise EraserR7OperatorError("action and behavior recipe identities differ")
    if trace.behavior.excluded_edge_i is not None:
        raise EraserR7OperatorError("base behavior unexpectedly deletes an edge")
    expected_used = _used_edge_ids(trace.behavior)
    if (
        not isinstance(trace.used_edge_ids, tuple)
        or trace.used_edge_ids != expected_used
        or tuple(sorted(set(trace.used_edge_ids))) != trace.used_edge_ids
    ):
        raise EraserR7OperatorError("used edge registry drifted")
    if not isinstance(trace.edge_deletion_witnesses, tuple):
        raise EraserR7OperatorError("edge deletion witnesses must be a tuple")
    if any(
        not isinstance(witness, EdgeDeletionWitness)
        for witness in trace.edge_deletion_witnesses
    ):
        raise EraserR7OperatorError("edge deletion witness has the wrong type")
    if tuple(witness.edge_i for witness in trace.edge_deletion_witnesses) != expected_used:
        raise EraserR7OperatorError("used edges do not have exactly one deletion witness")
    if trace.recipe_id == "R0_DENSE5" and expected_used:
        raise EraserR7OperatorError("R0 unexpectedly used a typed edge")
    for witness in trace.edge_deletion_witnesses:
        _require_int(witness.edge_i, "edge deletion index", minimum=0)
        _require_sha256(witness.witness_sha256, "edge deletion witness hash")
        if (
            recompute_edge_deletion_witness_sha256(witness)
            != witness.witness_sha256
        ):
            raise EraserR7OperatorError("edge deletion witness self hash drifted")
        verify_behavior_trace(witness.counterfactual_behavior)
        if witness.counterfactual_behavior.excluded_edge_i != witness.edge_i:
            raise EraserR7OperatorError("counterfactual deleted the wrong edge")
        expected_drop = tuple(
            base_value - counterfactual_value
            for base_value, counterfactual_value in zip(
                trace.behavior.selected_facet_maxima_ints,
                witness.counterfactual_behavior.selected_facet_maxima_ints,
            )
        )
        if (
            witness.selected_ordinals_changed
            != (
                trace.behavior.output_top5
                != witness.counterfactual_behavior.output_top5
            )
            or witness.witness_path_receipts_changed
            != (
                _path_only_step_receipts(trace.behavior)
                != _path_only_step_receipts(witness.counterfactual_behavior)
            )
            or witness.ico_coverage_changed
            != any(value != 0 for value in expected_drop)
            or witness.ico_coverage_drop_ints != expected_drop
        ):
            raise EraserR7OperatorError("edge deletion causal flags drifted")
    if (graph is None) != (semantic_tensor is None):
        raise EraserR7OperatorError("action input verification requires graph and tensor")
    if graph is not None and semantic_tensor is not None:
        expected = run_action(
            recipe_id=trace.recipe_id,
            graph=graph,
            semantic_tensor=semantic_tensor,
        )
        if trace != expected:
            raise EraserR7OperatorError("action trace does not reconstruct from inputs")
    return trace.trace_sha256


def run_all_actions(
    *,
    graph: QueryAnchoredSentenceGraph,
    semantic_tensor: QuerySemanticTensor,
) -> tuple[ActionTrace, ActionTrace]:
    return (
        run_action(
            recipe_id="R0_DENSE5",
            graph=graph,
            semantic_tensor=semantic_tensor,
        ),
        run_action(
            recipe_id="R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE",
            graph=graph,
            semantic_tensor=semantic_tensor,
        ),
    )


def sentence_leave_one_out_coverage_deltas(
    semantic_tensor: QuerySemanticTensor,
    selected_top5: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """Return the exact five selected-minus-leave-one-out ICO coverage deltas."""

    tensor = _validated_tensor(semantic_tensor)
    selected = tuple(selected_top5)
    if len(selected) != TOP_K or len(set(selected)) != TOP_K:
        raise EraserR7OperatorError("leave-one-out action must be an exact top five")
    base = facet_maxima_ints(tensor, selected)
    rows: list[tuple[int, int, int]] = []
    for slot in range(TOP_K):
        reduced = selected[:slot] + selected[slot + 1 :]
        counterfactual = facet_maxima_ints(tensor, reduced)
        delta = tuple(
            base_value - counterfactual_value
            for base_value, counterfactual_value in zip(base, counterfactual)
        )
        rows.append((delta[0], delta[1], delta[2]))
    return tuple(rows)


# Narrow compatibility aliases for sibling formal runners.
execute_recipe = run_action
run_recipe = run_action
recompute_evaluator_behavior_sha256 = recompute_behavior_sha256


__all__ = [
    "ADJACENT_SENTENCE",
    "ANCHOR_FANOUT",
    "ActionTrace",
    "AdjacencyNeighbor",
    "AtomicPath",
    "BehaviorTrace",
    "EDGE_TYPES",
    "EdgeDeletionWitness",
    "EraserR7OperatorError",
    "FACET_TYPES",
    "FacetSemanticRow",
    "INTEGER_SCALE",
    "MAX_ADJACENCY_HOPS",
    "OFFICIAL_ICO_ANCHOR",
    "OfficialIcoFacet",
    "QueryAnchoredSentenceGraph",
    "QuerySemanticTensor",
    "RECIPE_IDS",
    "SelectionStep",
    "SentenceUnit",
    "TOP_K",
    "TerminalFacetPathMap",
    "TypedEdge",
    "VERSION",
    "build_query_anchored_graph",
    "canonical_facet_terminal_maps",
    "enumerate_atomic_paths",
    "execute_recipe",
    "facet_maxima_ints",
    "make_official_ico_facets",
    "make_query_semantic_tensor",
    "recompute_action_trace_sha256",
    "recompute_behavior_sha256",
    "recompute_edge_deletion_witness_sha256",
    "recompute_evaluator_behavior_sha256",
    "recompute_graph_sha256",
    "recompute_tensor_sha256",
    "run_action",
    "run_action_core",
    "run_all_actions",
    "run_recipe",
    "sentence_leave_one_out_coverage_deltas",
    "stable_hash",
    "verify_action_trace",
    "verify_behavior_trace",
    "verify_query_anchored_graph",
    "verify_query_semantic_tensor",
]
