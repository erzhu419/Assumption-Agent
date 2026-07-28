"""Source-free relational set comparator for the post-stop QuAC study.

This module is deliberately dataset agnostic.  It accepts only already
quantized, item-local dialogue/evidence graphs and exact offline utilities.
It has no filesystem, network, API, text parsing, source-loading, or baseline
execution capability.

RJMC-V1 compares complete five-unit evidence sets relative to RAW.  Its
challenger is a two-layer relational attention network:

* layer one passes typed messages inside each evidence set;
* layer two performs typed cross-set attention from one set to the other; and
* ``f(left, right) = g(left, right) - g(right, left)`` is antisymmetric.

Consequently ``f(RAW, RAW)`` is structurally zero, independently of fitted
parameters.  Five deterministic component-jackknife models are fitted with an
item-balanced listwise utility-delta loss.  Selection maximizes the minimum
score across those five heads.  RAW wins every exact tie, followed by fewer
replacements and canonical state order.

The state constructor enumerates RAW plus *every* distinct one- and
two-replacement set.  Candidate pruning and sampled state spaces are therefore
not expressible through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn


VERSION = "quac_rjmc_evaluator_v1"
ARCHITECTURE_DECISION_SELF_SHA256 = (
    "9efb416359c1efc315846523a67382b0b942a8a827976cece72175085fe79462"
)

TOP_K = 5
MAX_REPLACEMENTS = 2
COMPONENT_COUNT = 5
NODE_FEATURE_WIDTH = 4
DIALOGUE_FACET_WIDTH = 4
RELATION_TYPES = (
    "same_section",
    "adjacent_window",
    "entity_chain",
)
RELATION_WIDTH = len(RELATION_TYPES)
DEFAULT_STATE_BATCH_SIZE = 128


class RjmcEvaluatorError(ValueError):
    """A source-free graph, state, fit, or selection contract drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise RjmcEvaluatorError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_float_tuple(
    value: object,
    *,
    field: str,
    width: int,
    minimum: float = -8.0,
    maximum: float = 8.0,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RjmcEvaluatorError(f"{field} must be a numeric vector")
    rows = tuple(value)
    if len(rows) != width:
        raise RjmcEvaluatorError(f"{field} width drifted")
    if any(type(row) not in (int, float) for row in rows):
        raise RjmcEvaluatorError(f"{field} must contain exact finite numbers")
    normalized = tuple(float(row) for row in rows)
    if any(
        not math.isfinite(row) or row < minimum or row > maximum
        for row in normalized
    ):
        raise RjmcEvaluatorError(f"{field} is outside the frozen numeric range")
    return normalized


def _binary_tuple(value: object, *, field: str, width: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RjmcEvaluatorError(f"{field} must be a binary vector")
    rows = tuple(value)
    if len(rows) != width:
        raise RjmcEvaluatorError(f"{field} width drifted")
    if any(type(row) is not int or row not in (0, 1) for row in rows):
        raise RjmcEvaluatorError(f"{field} must contain exact zero/one values")
    return rows  # type: ignore[return-value]


@dataclass(frozen=True)
class EvidenceUnit:
    """One content-free evidence-window node.

    ``node_features`` are fixed upstream action features in this order:
    dense relevance, direct dialogue-anchor strength, normalized turn recency,
    and normalized section proximity.  ``dialogue_facets`` are four binary,
    parser-derived facet incidences.  Neither vector may contain gold labels,
    family/split identity, item identity, or HippoRAG output.
    """

    unit_id: str
    node_features: tuple[float, ...]
    dialogue_facets: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.unit_id, str)
            or not self.unit_id
            or self.unit_id.strip() != self.unit_id
            or any(ord(character) > 127 for character in self.unit_id)
        ):
            raise RjmcEvaluatorError("unit ID must be nonempty canonical ASCII")
        object.__setattr__(
            self,
            "node_features",
            _exact_float_tuple(
                self.node_features,
                field="node features",
                width=NODE_FEATURE_WIDTH,
            ),
        )
        object.__setattr__(
            self,
            "dialogue_facets",
            _binary_tuple(
                self.dialogue_facets,
                field="dialogue facets",
                width=DIALOGUE_FACET_WIDTH,
            ),
        )


@dataclass(frozen=True)
class TypedEdge:
    """One canonical undirected typed graph edge."""

    left: str
    right: str
    relation: str
    strength: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.left, str)
            or not isinstance(self.right, str)
            or not self.left < self.right
        ):
            raise RjmcEvaluatorError(
                "typed edge endpoints must use strict canonical order"
            )
        if self.relation not in RELATION_TYPES:
            raise RjmcEvaluatorError("typed edge relation is outside the registry")
        if (
            type(self.strength) not in (int, float)
            or not math.isfinite(float(self.strength))
            or not 0.0 < float(self.strength) <= 1.0
        ):
            raise RjmcEvaluatorError("typed edge strength must be in (0, 1]")
        object.__setattr__(self, "strength", float(self.strength))


@dataclass(frozen=True)
class RelationalGraph:
    """A complete item-local graph supplied by a future source adapter."""

    units: tuple[EvidenceUnit, ...]
    edges: tuple[TypedEdge, ...]

    def __post_init__(self) -> None:
        if len(self.units) < TOP_K:
            raise RjmcEvaluatorError("graph must contain at least five units")
        unit_ids = tuple(unit.unit_id for unit in self.units)
        if len(set(unit_ids)) != len(unit_ids):
            raise RjmcEvaluatorError("graph contains duplicate unit IDs")
        if unit_ids != tuple(sorted(unit_ids)):
            raise RjmcEvaluatorError("graph units must use canonical ID order")
        known = set(unit_ids)
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if edge.left not in known or edge.right not in known:
                raise RjmcEvaluatorError("typed edge references an unknown unit")
            key = (edge.left, edge.right, edge.relation)
            if key in edge_keys:
                raise RjmcEvaluatorError("graph contains a duplicate typed edge")
            edge_keys.add(key)
        if tuple(
            sorted(
                self.edges,
                key=lambda row: (
                    row.left,
                    row.right,
                    RELATION_TYPES.index(row.relation),
                ),
            )
        ) != self.edges:
            raise RjmcEvaluatorError("typed edges must use canonical order")

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.units)

    def ordinal(self, unit_id: str) -> int:
        try:
            return self.unit_ids.index(unit_id)
        except ValueError as exc:
            raise RjmcEvaluatorError("set references an unknown unit") from exc

    def canonical_set(self, unit_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(unit_ids, (str, bytes)):
            raise RjmcEvaluatorError("evidence set must be a unit-ID sequence")
        rows = tuple(unit_ids)
        if len(rows) != TOP_K or len(set(rows)) != TOP_K:
            raise RjmcEvaluatorError("evidence set must contain five distinct units")
        ordinal = {unit_id: index for index, unit_id in enumerate(self.unit_ids)}
        if any(unit_id not in ordinal for unit_id in rows):
            raise RjmcEvaluatorError("evidence set references an unknown unit")
        return tuple(sorted(rows, key=ordinal.__getitem__))


@dataclass(frozen=True)
class SetState:
    """One canonical complete-state-space member."""

    unit_ids: tuple[str, ...]
    replacements: int

    def __post_init__(self) -> None:
        if len(self.unit_ids) != TOP_K or len(set(self.unit_ids)) != TOP_K:
            raise RjmcEvaluatorError("state must contain five distinct units")
        if (
            type(self.replacements) is not int
            or not 0 <= self.replacements <= MAX_REPLACEMENTS
        ):
            raise RjmcEvaluatorError("replacement count is outside the registry")


def complete_state_count(candidate_count: int) -> int:
    """Exact RAW + all distinct one/two replacement state count."""

    if type(candidate_count) is not int or candidate_count < 0:
        raise RjmcEvaluatorError("candidate count must be a nonnegative integer")
    return (
        1
        + TOP_K * candidate_count
        + math.comb(TOP_K, 2) * math.comb(candidate_count, 2)
    )


def enumerate_complete_states(
    graph: RelationalGraph,
    *,
    raw_top5: Sequence[str],
) -> tuple[SetState, ...]:
    """Enumerate RAW plus every state over the fixed ``graph \\ RAW`` domain.

    There is intentionally no candidate argument.  Once RAW is supplied, every
    other graph node is a replacement candidate, including the zero- and
    one-candidate boundary cases.
    """

    raw = graph.canonical_set(raw_top5)
    ordinal = {unit_id: index for index, unit_id in enumerate(graph.unit_ids)}
    raw_set = set(raw)
    candidate_rows = tuple(
        unit_id for unit_id in graph.unit_ids if unit_id not in raw_set
    )

    states = [SetState(raw, 0)]
    for removed in itertools.combinations(raw, 1):
        retained = raw_set.difference(removed)
        for added in itertools.combinations(candidate_rows, 1):
            output = graph.canonical_set(tuple(retained) + added)
            states.append(SetState(output, 1))
    for removed in itertools.combinations(raw, 2):
        retained = raw_set.difference(removed)
        for added in itertools.combinations(candidate_rows, 2):
            output = graph.canonical_set(tuple(retained) + added)
            states.append(SetState(output, 2))

    raw_state = states[0]
    tail = sorted(states[1:], key=lambda row: (row.replacements, row.unit_ids))
    result = (raw_state, *tail)
    if (
        len(result) != complete_state_count(len(candidate_rows))
        or len({row.unit_ids for row in result}) != len(result)
    ):
        raise RjmcEvaluatorError("complete state enumeration drifted")
    return result


def _graph_arrays(
    graph: RelationalGraph,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    unit_ids = graph.unit_ids
    ordinals = {unit_id: index for index, unit_id in enumerate(unit_ids)}
    nodes = np.asarray(
        [
            (*unit.node_features, *(float(value) for value in unit.dialogue_facets))
            for unit in graph.units
        ],
        dtype=np.float64,
    )
    relation = np.zeros(
        (len(unit_ids), len(unit_ids), RELATION_WIDTH), dtype=np.float64
    )
    for edge in graph.edges:
        left = ordinals[edge.left]
        right = ordinals[edge.right]
        channel = RELATION_TYPES.index(edge.relation)
        relation[left, right, channel] = edge.strength
        relation[right, left, channel] = edge.strength
    return nodes, relation, ordinals


class RelationalSetComparator(nn.Module):
    """Two-layer, small-width relational cross-set comparator."""

    def __init__(self, *, width: int = 8) -> None:
        super().__init__()
        if type(width) is not int or not 4 <= width <= 32:
            raise RjmcEvaluatorError("attention width must be an integer in [4, 32]")
        self.width = width
        input_width = NODE_FEATURE_WIDTH + DIALOGUE_FACET_WIDTH

        self.input_projection = nn.Linear(input_width, width)

        self.intra_query = nn.Linear(width, width, bias=False)
        self.intra_key = nn.Linear(width, width, bias=False)
        self.intra_value = nn.Linear(width, width, bias=False)
        self.intra_residual = nn.Linear(width, width, bias=False)
        self.intra_relation_bias = nn.Parameter(torch.empty(RELATION_WIDTH))
        self.intra_relation_message = nn.Parameter(
            torch.empty(RELATION_WIDTH, width)
        )

        self.cross_query = nn.Linear(width, width, bias=False)
        self.cross_key = nn.Linear(width, width, bias=False)
        self.cross_value = nn.Linear(width, width, bias=False)
        self.cross_residual = nn.Linear(width, width, bias=False)
        self.cross_relation_bias = nn.Parameter(torch.empty(RELATION_WIDTH))
        self.cross_relation_message = nn.Parameter(
            torch.empty(RELATION_WIDTH, width)
        )

        self.pool_projection = nn.Linear(width * 2, width)
        self.output = nn.Linear(width, 1)
        self.reset_parameters()
        self.to(dtype=torch.float64, device=torch.device("cpu"))

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.intra_relation_bias)
        nn.init.xavier_uniform_(self.intra_relation_message)
        nn.init.zeros_(self.cross_relation_bias)
        nn.init.xavier_uniform_(self.cross_relation_message)

    def _intra(
        self, nodes: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        hidden = torch.tanh(self.input_projection(nodes))
        query = self.intra_query(hidden)
        key = self.intra_key(hidden)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.width)
        logits = logits + torch.einsum(
            "bijn,n->bij", relation, self.intra_relation_bias
        )
        attention = torch.softmax(logits, dim=-1)
        values = self.intra_value(hidden).unsqueeze(1)
        relation_message = torch.einsum(
            "bijn,nw->bijw", relation, self.intra_relation_message
        )
        aggregate = torch.sum(
            attention.unsqueeze(-1) * (values + relation_message), dim=2
        )
        return torch.tanh(self.intra_residual(hidden) + aggregate)

    def _cross(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        query = self.cross_query(left)
        key = self.cross_key(right)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.width)
        logits = logits + torch.einsum(
            "bijn,n->bij", relation, self.cross_relation_bias
        )
        attention = torch.softmax(logits, dim=-1)
        values = self.cross_value(right).unsqueeze(1)
        relation_message = torch.einsum(
            "bijn,nw->bijw", relation, self.cross_relation_message
        )
        aggregate = torch.sum(
            attention.unsqueeze(-1) * (values + relation_message), dim=2
        )
        return torch.tanh(self.cross_residual(left) + aggregate)

    def _directed_score(
        self,
        left_nodes: torch.Tensor,
        right_nodes: torch.Tensor,
        left_relation: torch.Tensor,
        right_relation: torch.Tensor,
        cross_relation: torch.Tensor,
    ) -> torch.Tensor:
        left_hidden = self._intra(left_nodes, left_relation)
        right_hidden = self._intra(right_nodes, right_relation)
        compared = self._cross(left_hidden, right_hidden, cross_relation)
        pooled = torch.cat(
            (torch.mean(compared, dim=1), torch.amax(compared, dim=1)), dim=-1
        )
        return self.output(torch.tanh(self.pool_projection(pooled))).squeeze(-1)

    def compare_tensors(
        self,
        *,
        left_nodes: torch.Tensor,
        right_nodes: torch.Tensor,
        left_relation: torch.Tensor,
        right_relation: torch.Tensor,
        left_to_right_relation: torch.Tensor,
        identical: torch.Tensor,
    ) -> torch.Tensor:
        """Return an antisymmetric batch of set-to-set scores."""

        forward = self._directed_score(
            left_nodes,
            right_nodes,
            left_relation,
            right_relation,
            left_to_right_relation,
        )
        reverse = self._directed_score(
            right_nodes,
            left_nodes,
            right_relation,
            left_relation,
            left_to_right_relation.transpose(1, 2),
        )
        score = forward - reverse
        # This is an architectural identity, not a learned threshold.
        return torch.where(identical, score * 0.0, score)


@dataclass(frozen=True)
class _CompiledComparison:
    left_nodes: torch.Tensor
    right_nodes: torch.Tensor
    left_relation: torch.Tensor
    right_relation: torch.Tensor
    cross_relation: torch.Tensor
    identical: torch.Tensor


def _compile_comparisons(
    graph: RelationalGraph,
    *,
    left_sets: Sequence[Sequence[str]],
    right_set: Sequence[str],
) -> _CompiledComparison:
    nodes_array, relation_array, ordinals = _graph_arrays(graph)
    right = graph.canonical_set(right_set)
    left = tuple(graph.canonical_set(row) for row in left_sets)
    if not left:
        raise RjmcEvaluatorError("comparison batch must be nonempty")

    left_index = np.asarray(
        [[ordinals[unit_id] for unit_id in row] for row in left], dtype=np.int64
    )
    right_index = np.asarray(
        [ordinals[unit_id] for unit_id in right], dtype=np.int64
    )
    batch_size = len(left)
    right_batch = np.broadcast_to(right_index, (batch_size, TOP_K))

    left_nodes = nodes_array[left_index]
    right_nodes = nodes_array[right_batch]
    left_relation = relation_array[
        left_index[:, :, None], left_index[:, None, :]
    ]
    right_relation = relation_array[
        right_batch[:, :, None], right_batch[:, None, :]
    ]
    cross_relation = relation_array[
        left_index[:, :, None], right_batch[:, None, :]
    ]
    identical = np.asarray([row == right for row in left], dtype=np.bool_)
    return _CompiledComparison(
        left_nodes=torch.from_numpy(np.ascontiguousarray(left_nodes)),
        right_nodes=torch.from_numpy(np.ascontiguousarray(right_nodes)),
        left_relation=torch.from_numpy(np.ascontiguousarray(left_relation)),
        right_relation=torch.from_numpy(np.ascontiguousarray(right_relation)),
        cross_relation=torch.from_numpy(np.ascontiguousarray(cross_relation)),
        identical=torch.from_numpy(identical),
    )


def compare_sets(
    model: RelationalSetComparator,
    graph: RelationalGraph,
    *,
    left: Sequence[str],
    right: Sequence[str],
) -> float:
    """Score one set against another after canonical permutation removal."""

    compiled = _compile_comparisons(
        graph, left_sets=(left,), right_set=right
    )
    model.eval()
    with torch.no_grad():
        score = model.compare_tensors(
            left_nodes=compiled.left_nodes,
            right_nodes=compiled.right_nodes,
            left_relation=compiled.left_relation,
            right_relation=compiled.right_relation,
            left_to_right_relation=compiled.cross_relation,
            identical=compiled.identical,
        )
    return float(score[0])


def _compiled_scores(
    model: RelationalSetComparator,
    compiled: _CompiledComparison,
) -> torch.Tensor:
    return model.compare_tensors(
        left_nodes=compiled.left_nodes,
        right_nodes=compiled.right_nodes,
        left_relation=compiled.left_relation,
        right_relation=compiled.right_relation,
        left_to_right_relation=compiled.cross_relation,
        identical=compiled.identical,
    )


def _state_batches(
    states: Sequence[SetState], *, batch_size: int
) -> Iterator[tuple[SetState, ...]]:
    if type(batch_size) is not int or batch_size < 1:
        raise RjmcEvaluatorError("state batch size must be a positive integer")
    for start in range(0, len(states), batch_size):
        yield tuple(states[start : start + batch_size])


def _score_complete_states(
    model: RelationalSetComparator,
    graph: RelationalGraph,
    *,
    states: Sequence[SetState],
    raw_top5: Sequence[str],
    state_batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in _state_batches(states, batch_size=state_batch_size):
            compiled = _compile_comparisons(
                graph,
                left_sets=tuple(state.unit_ids for state in batch),
                right_set=raw_top5,
            )
            rows.append(
                _compiled_scores(model, compiled)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=True)
            )
    return np.concatenate(rows)


def score_states(
    model: RelationalSetComparator,
    graph: RelationalGraph,
    *,
    raw_top5: Sequence[str],
    state_batch_size: int = DEFAULT_STATE_BATCH_SIZE,
) -> np.ndarray:
    """Score the internally derived, complete ``graph \\ RAW`` state space."""

    raw = graph.canonical_set(raw_top5)
    states = enumerate_complete_states(graph, raw_top5=raw)
    return _score_complete_states(
        model,
        graph,
        states=states,
        raw_top5=raw,
        state_batch_size=state_batch_size,
    )


def _proof_coverage_key(
    graph: RelationalGraph, state: SetState
) -> tuple[int, int, int, int]:
    ordinal = {unit_id: index for index, unit_id in enumerate(graph.unit_ids)}
    units = tuple(graph.units[ordinal[unit_id]] for unit_id in state.unit_ids)
    union_coverage = sum(
        any(unit.dialogue_facets[index] for unit in units)
        for index in range(DIALOGUE_FACET_WIDTH)
    )
    covered_unit_count = sum(any(unit.dialogue_facets) for unit in units)
    total_coverage = sum(sum(unit.dialogue_facets) for unit in units)
    state_set = set(state.unit_ids)
    typed_edge_count = sum(
        1
        for edge in graph.edges
        if edge.left in state_set and edge.right in state_set
    )
    return union_coverage, covered_unit_count, total_coverage, typed_edge_count


def select_e0_proof_coverage(
    graph: RelationalGraph, *, raw_top5: Sequence[str]
) -> int:
    """Select E0 from discrete counts and canonical lexicographic tie breaks."""

    raw = graph.canonical_set(raw_top5)
    states = enumerate_complete_states(graph, raw_top5=raw)
    best_index = 0
    best_key = _proof_coverage_key(graph, states[0])
    for index, state in enumerate(states[1:], start=1):
        key = _proof_coverage_key(graph, state)
        if key > best_key:
            best_index = index
            best_key = key
    return best_index


@dataclass(frozen=True)
class ListwiseTrainingItem:
    """One exact item-balanced listwise training item."""

    item_id: str
    component: int
    graph: RelationalGraph
    raw_top5: tuple[str, ...]
    utility: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise RjmcEvaluatorError("training item ID must be nonempty")
        if type(self.component) is not int or not 0 <= self.component < COMPONENT_COUNT:
            raise RjmcEvaluatorError("jackknife component is outside [0, 5)")
        raw = self.graph.canonical_set(self.raw_top5)
        states = enumerate_complete_states(self.graph, raw_top5=raw)
        if (
            len(self.utility) != len(states)
            or any(type(value) is not int or not 0 <= value <= 4 for value in self.utility)
        ):
            raise RjmcEvaluatorError(
                "utility must align to every complete state and use integer [0, 4]"
            )
        object.__setattr__(self, "raw_top5", raw)

    @property
    def states(self) -> tuple[SetState, ...]:
        return enumerate_complete_states(self.graph, raw_top5=self.raw_top5)


@dataclass(frozen=True)
class FitConfig:
    """Frozen deterministic training controls."""

    width: int = 8
    epochs: int = 36
    learning_rate: float = 0.035
    weight_decay: float = 1.0e-5
    target_temperature: float = 0.5
    state_batch_size: int = DEFAULT_STATE_BATCH_SIZE
    seed: int = 1729

    def __post_init__(self) -> None:
        if type(self.width) is not int or not 4 <= self.width <= 32:
            raise RjmcEvaluatorError("fit width is outside [4, 32]")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 10_000:
            raise RjmcEvaluatorError("fit epochs are outside [1, 10000]")
        for value, field in (
            (self.learning_rate, "learning rate"),
            (self.weight_decay, "weight decay"),
            (self.target_temperature, "target temperature"),
        ):
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise RjmcEvaluatorError(f"{field} must be finite")
        if not 0.0 < self.learning_rate <= 1.0:
            raise RjmcEvaluatorError("learning rate is outside (0, 1]")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise RjmcEvaluatorError("weight decay is outside [0, 1]")
        if not 0.0 < self.target_temperature <= 4.0:
            raise RjmcEvaluatorError("target temperature is outside (0, 4]")
        if (
            type(self.state_batch_size) is not int
            or not 1 <= self.state_batch_size <= 4096
        ):
            raise RjmcEvaluatorError("state batch size is outside [1, 4096]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise RjmcEvaluatorError("fit seed is outside the deterministic range")


class JackknifeMinimaxComparator(nn.Module):
    """Five component-jackknife heads with minimum-score inference."""

    def __init__(self, heads: Sequence[RelationalSetComparator]) -> None:
        super().__init__()
        if len(heads) != COMPONENT_COUNT:
            raise RjmcEvaluatorError("RJMC requires exactly five jackknife heads")
        if len({head.width for head in heads}) != 1:
            raise RjmcEvaluatorError("jackknife head widths drifted")
        self.heads = nn.ModuleList(heads)

    def score_matrix(
        self,
        graph: RelationalGraph,
        *,
        raw_top5: Sequence[str],
        state_batch_size: int = DEFAULT_STATE_BATCH_SIZE,
    ) -> np.ndarray:
        raw = graph.canonical_set(raw_top5)
        states = enumerate_complete_states(graph, raw_top5=raw)
        rows = tuple(
            _score_complete_states(
                head,
                graph,
                states=states,
                raw_top5=raw,
                state_batch_size=state_batch_size,
            )
            for head in self.heads
        )
        return np.stack(rows, axis=0)

    def minimax_scores(
        self,
        graph: RelationalGraph,
        *,
        raw_top5: Sequence[str],
        state_batch_size: int = DEFAULT_STATE_BATCH_SIZE,
    ) -> np.ndarray:
        return np.min(
            self.score_matrix(
                graph,
                raw_top5=raw_top5,
                state_batch_size=state_batch_size,
            ),
            axis=0,
        )

    def select(
        self,
        graph: RelationalGraph,
        *,
        raw_top5: Sequence[str],
        state_batch_size: int = DEFAULT_STATE_BATCH_SIZE,
    ) -> tuple[int, np.ndarray]:
        scores = self.minimax_scores(
            graph,
            raw_top5=raw_top5,
            state_batch_size=state_batch_size,
        )
        if not np.isfinite(scores).all() or scores[0] != 0.0:
            raise RjmcEvaluatorError("minimax scores violated RAW structural zero")
        # np.argmax returns the first maximum.  The enumerator orders RAW,
        # fewer replacements, then canonical state identity.
        return int(np.argmax(scores)), scores


def fit_component_jackknife(
    items: Sequence[ListwiseTrainingItem],
    *,
    config: FitConfig = FitConfig(),
) -> JackknifeMinimaxComparator:
    """Fit five deterministic leave-one-component-out listwise heads."""

    item_rows = tuple(items)
    if len(item_rows) < COMPONENT_COUNT * 2:
        raise RjmcEvaluatorError("jackknife fit needs at least two items per component")
    if len({item.item_id for item in item_rows}) != len(item_rows):
        raise RjmcEvaluatorError("training item IDs must be unique")
    component_counts = {
        component: sum(item.component == component for item in item_rows)
        for component in range(COMPONENT_COUNT)
    }
    if any(count < 2 for count in component_counts.values()):
        raise RjmcEvaluatorError("each jackknife component needs at least two items")

    # CPU float64 plus a single BLAS thread gives repeat-exact same-host fits.
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    ordered_items = tuple(sorted(item_rows, key=lambda row: row.item_id))
    heads: list[RelationalSetComparator] = []
    for held_component in range(COMPONENT_COUNT):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + held_component)
            model = RelationalSetComparator(width=config.width)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        train_rows = tuple(
            row for row in ordered_items if row.component != held_component
        )
        for _epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            for row in train_rows:
                states = row.states
                target_utility = torch.asarray(row.utility, dtype=torch.float64)
                target_probability = torch.softmax(
                    (target_utility - target_utility[0])
                    / config.target_temperature,
                    dim=0,
                )

                # First pass: obtain the current listwise normalizer without
                # retaining any per-item autograd graph or compiled item.
                predicted_chunks: list[torch.Tensor] = []
                model.eval()
                with torch.no_grad():
                    for batch in _state_batches(
                        states, batch_size=config.state_batch_size
                    ):
                        comparison = _compile_comparisons(
                            row.graph,
                            left_sets=tuple(state.unit_ids for state in batch),
                            right_set=row.raw_top5,
                        )
                        predicted_chunks.append(
                            _compiled_scores(model, comparison).detach()
                        )
                predicted_scores = torch.cat(predicted_chunks)
                predicted_probability = torch.softmax(predicted_scores, dim=0)
                score_gradient = (
                    predicted_probability - target_probability
                ) / len(train_rows)
                if not torch.isfinite(score_gradient).all():
                    raise RjmcEvaluatorError(
                        "jackknife listwise gradient became nonfinite"
                    )

                # Second pass: recompute one bounded state batch at a time and
                # backpropagate its exact slice of d(listwise_loss)/d(score).
                model.train()
                offset = 0
                for batch in _state_batches(
                    states, batch_size=config.state_batch_size
                ):
                    comparison = _compile_comparisons(
                        row.graph,
                        left_sets=tuple(state.unit_ids for state in batch),
                        right_set=row.raw_top5,
                    )
                    scores = _compiled_scores(model, comparison)
                    next_offset = offset + len(batch)
                    scores.backward(score_gradient[offset:next_offset])
                    offset = next_offset
                if offset != len(states):
                    raise RjmcEvaluatorError(
                        "streaming listwise state boundary drifted"
                    )
            optimizer.step()
        model.eval()
        heads.append(model)
    return JackknifeMinimaxComparator(heads)


def model_parameter_sha256(model: nn.Module) -> str:
    """Hash exact CPU float64 parameters in stable name order."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        encoded = name.encode("ascii")
        values = (
            parameter.detach()
            .cpu()
            .to(dtype=torch.float64)
            .contiguous()
            .numpy()
            .astype("<f8", copy=False)
        )
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(tuple(values.shape).__repr__().encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def behavior_sha256(
    ensemble: JackknifeMinimaxComparator,
    items: Sequence[ListwiseTrainingItem],
) -> str:
    """Hash exact actions and float64 score matrices for synthetic replay."""

    payload = bytearray()
    for item in sorted(items, key=lambda row: row.item_id):
        matrix = ensemble.score_matrix(
            item.graph, raw_top5=item.raw_top5
        )
        selected = int(np.argmax(np.min(matrix, axis=0)))
        payload.extend(item.item_id.encode("ascii"))
        payload.extend(selected.to_bytes(8, "big"))
        payload.extend(
            np.ascontiguousarray(matrix, dtype="<f8").tobytes(order="C")
        )
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ARCHITECTURE_DECISION_SELF_SHA256",
    "COMPONENT_COUNT",
    "DEFAULT_STATE_BATCH_SIZE",
    "DIALOGUE_FACET_WIDTH",
    "EvidenceUnit",
    "FitConfig",
    "JackknifeMinimaxComparator",
    "ListwiseTrainingItem",
    "MAX_REPLACEMENTS",
    "NODE_FEATURE_WIDTH",
    "RELATION_TYPES",
    "RelationalGraph",
    "RelationalSetComparator",
    "RjmcEvaluatorError",
    "SetState",
    "TOP_K",
    "TypedEdge",
    "VERSION",
    "behavior_sha256",
    "canonical_bytes",
    "compare_sets",
    "complete_state_count",
    "enumerate_complete_states",
    "fit_component_jackknife",
    "model_parameter_sha256",
    "score_states",
    "select_e0_proof_coverage",
    "stable_hash",
]
