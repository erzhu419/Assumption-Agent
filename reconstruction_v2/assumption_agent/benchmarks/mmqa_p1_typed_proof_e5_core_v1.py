"""Pure typed-proof action and E5 evaluator core for MultiModalQA P1.

``MMQA_P1_LOCAL_PROOF_E5_V1`` is deliberately a source-free, offline core.
The outer runtime may provide already-computed MiniLM and cross-encoder
coordinates, but this module has no source reader, text parser, filesystem,
network, model, API, credential, retry, item-ID, or family-ID surface.

The core implements four frozen pieces:

* typed row/text nodes and exact directed row-to-text or text-to-row links;
* a two-hop query-local closure and at most 256 connected row/text bundles;
* a fixed E0 proof energy and a lambda-one conditional log-linear E5 model
  fitted with deterministic NumPy L-BFGS and multi-gold log-sum-exp; and
* offline binary-evidence nDCG@5, integer utility, an exact ties-excluded
  one-sided gain-vs-harm binomial tail, and A_hold/M_search decisions.

Local integer ordinals exist only to connect nodes and labels within one item.
They are never coordinates of either E0 or E5.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence

import numpy as np


STUDY_ID = "MMQA_P1_LOCAL_PROOF_E5_V1"
VERSION = "mmqa_p1_typed_proof_e5_core_v1"

ROW = "ROW"
TEXT = "TEXT"
NODE_TYPES = (ROW, TEXT)
ROW_TO_TEXT = "ROW_TO_TEXT"
TEXT_TO_ROW = "TEXT_TO_ROW"
EDGE_TYPES = (ROW_TO_TEXT, TEXT_TO_ROW)

MAX_CLOSURE_NODES = 96
MAX_CLOSURE_HOPS = 2
MAX_BUNDLE_SIZE = 5
MAX_BUNDLES = 256
TOP_K = 5
INTEGER_UTILITY_SCALE = 1_000_000_000
E5_L2 = 1.0
E5_MAX_ITER = 256
PROMOTION_ALPHA = Fraction(1, 10)

FEATURE_ORDER = (
    "cross_encoder_mean",
    "cross_encoder_minimum",
    "minilm_mean",
    "minilm_minimum",
    "entity_anchor_fraction",
    "relation_anchor_fraction",
    "numeric_temporal_fraction",
    "directed_link_density",
    "reciprocal_pair_fraction",
    "row_text_balance",
    "cardinality_penalty",
)

# E0 is a fixed proof energy, not a learned selector or a recipe gate.  The
# final coordinate is non-positive, so its positive coefficient penalizes
# gratuitous cardinality.
E0_WEIGHTS = (
    4.0,
    1.0,
    2.0,
    0.5,
    1.0,
    1.5,
    0.5,
    2.0,
    0.5,
    0.5,
    0.25,
)

FORBIDDEN_FEATURES = frozenset(
    {
        "answer",
        "candidate_id",
        "document_id",
        "family",
        "family_id",
        "gold",
        "gold_bundle",
        "hipporag_rank",
        "item_id",
        "node_id",
        "node_ordinal",
        "query_id",
        "raw_rank",
        "recipe_id",
        "source_id",
    }
)


class MmqaP1CoreError(ValueError):
    """Fail-closed error for malformed offline graph, model, or score input."""


def _strict_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise MmqaP1CoreError(f"{field} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise MmqaP1CoreError(f"{field} must be at least {minimum}")
    return result


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise MmqaP1CoreError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise MmqaP1CoreError(f"{field} must be a finite real number")
    return 0.0 if result == 0.0 else result


def _unit_interval(value: object, field: str) -> float:
    result = _finite_float(value, field)
    if not 0.0 <= result <= 1.0:
        raise MmqaP1CoreError(f"{field} must lie in [0, 1]")
    return result


def _binary(value: object, field: str) -> int:
    result = _strict_int(value, field)
    if result not in (0, 1):
        raise MmqaP1CoreError(f"{field} must be binary")
    return result


@dataclass(frozen=True)
class ProofNode:
    """One item-local row or text node with frozen, label-free coordinates."""

    ordinal: int
    node_type: str
    minilm_similarity: float
    cross_encoder_relevance: float
    entity_anchor: int
    relation_anchor: int
    numeric_or_temporal_anchor: int

    def __post_init__(self) -> None:
        ordinal = _strict_int(self.ordinal, "node ordinal", minimum=0)
        if self.node_type not in NODE_TYPES:
            raise MmqaP1CoreError("node type must be ROW or TEXT")
        minilm = _unit_interval(self.minilm_similarity, "MiniLM similarity")
        cross_encoder = _unit_interval(
            self.cross_encoder_relevance, "cross-encoder relevance"
        )
        entity = _binary(self.entity_anchor, "entity anchor")
        relation = _binary(self.relation_anchor, "relation anchor")
        numeric = _binary(
            self.numeric_or_temporal_anchor, "numeric/temporal anchor"
        )
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "minilm_similarity", minilm)
        object.__setattr__(self, "cross_encoder_relevance", cross_encoder)
        object.__setattr__(self, "entity_anchor", entity)
        object.__setattr__(self, "relation_anchor", relation)
        object.__setattr__(self, "numeric_or_temporal_anchor", numeric)


@dataclass(frozen=True)
class TypedLinkEdge:
    """One exact directed cross-modal link; no inferred similarity edge exists."""

    source_ordinal: int
    target_ordinal: int
    edge_type: str

    def __post_init__(self) -> None:
        source = _strict_int(self.source_ordinal, "edge source", minimum=0)
        target = _strict_int(self.target_ordinal, "edge target", minimum=0)
        if source == target:
            raise MmqaP1CoreError("typed links cannot be self-loops")
        if self.edge_type not in EDGE_TYPES:
            raise MmqaP1CoreError("edge type is outside the frozen registry")
        object.__setattr__(self, "source_ordinal", source)
        object.__setattr__(self, "target_ordinal", target)


@dataclass(frozen=True)
class ProofGraph:
    """A canonical item-local bipartite graph of rows, texts, and exact links."""

    nodes: tuple[ProofNode, ...]
    edges: tuple[TypedLinkEdge, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.nodes, (str, bytes))
            or not isinstance(self.nodes, Sequence)
            or not self.nodes
            or not all(isinstance(node, ProofNode) for node in self.nodes)
        ):
            raise MmqaP1CoreError("proof graph requires ProofNode rows")
        nodes = tuple(self.nodes)
        ordinals = tuple(node.ordinal for node in nodes)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise MmqaP1CoreError("nodes must have distinct ascending local ordinals")
        if (
            isinstance(self.edges, (str, bytes))
            or not isinstance(self.edges, Sequence)
            or not all(isinstance(edge, TypedLinkEdge) for edge in self.edges)
        ):
            raise MmqaP1CoreError("proof graph edges must be TypedLinkEdge rows")
        edges = tuple(self.edges)
        expected_edges = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.source_ordinal,
                    edge.target_ordinal,
                    EDGE_TYPES.index(edge.edge_type),
                ),
            )
        )
        if edges != expected_edges or len(set(edges)) != len(edges):
            raise MmqaP1CoreError("typed edges must be distinct and canonically sorted")
        by_ordinal = {node.ordinal: node for node in nodes}
        for edge in edges:
            source = by_ordinal.get(edge.source_ordinal)
            target = by_ordinal.get(edge.target_ordinal)
            if source is None or target is None:
                raise MmqaP1CoreError("typed edge refers to a missing local ordinal")
            expected = (
                ROW_TO_TEXT
                if source.node_type == ROW and target.node_type == TEXT
                else TEXT_TO_ROW
                if source.node_type == TEXT and target.node_type == ROW
                else None
            )
            if edge.edge_type != expected:
                raise MmqaP1CoreError(
                    "edge direction/type must exactly match ROW<->TEXT endpoints"
                )

    @property
    def node_by_ordinal(self) -> Mapping[int, ProofNode]:
        return {node.ordinal: node for node in self.nodes}


@dataclass(frozen=True)
class ProofClosure:
    graph: ProofGraph
    anchor_ordinals: tuple[int, ...]
    hop_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.graph, ProofGraph):
            raise MmqaP1CoreError("closure graph is malformed")
        anchors = tuple(self.anchor_ordinals)
        if (
            not anchors
            or anchors != tuple(sorted(anchors))
            or len(set(anchors)) != len(anchors)
            or any(type(value) is not int for value in anchors)
        ):
            raise MmqaP1CoreError("closure anchors must be distinct sorted ordinals")
        node_ordinals = {node.ordinal for node in self.graph.nodes}
        if not set(anchors).issubset(node_ordinals):
            raise MmqaP1CoreError("closure lost a query anchor")
        hops = _strict_int(self.hop_limit, "closure hop limit", minimum=0)
        if hops > MAX_CLOSURE_HOPS:
            raise MmqaP1CoreError("closure hop limit exceeds the frozen maximum")
        object.__setattr__(self, "hop_limit", hops)


def _node_priority(node: ProofNode) -> tuple[float, ...]:
    return (
        node.cross_encoder_relevance,
        node.minilm_similarity,
        float(node.entity_anchor + node.relation_anchor),
        float(node.numeric_or_temporal_anchor),
    )


def build_query_local_closure(
    graph: ProofGraph,
    anchor_ordinals: Sequence[int],
    *,
    hop_limit: int = MAX_CLOSURE_HOPS,
    max_nodes: int = MAX_CLOSURE_NODES,
) -> ProofClosure:
    """Follow exact directed links from frozen anchors for at most two hops."""

    if not isinstance(graph, ProofGraph):
        raise MmqaP1CoreError("closure input must be a ProofGraph")
    if isinstance(anchor_ordinals, (str, bytes)) or not isinstance(
        anchor_ordinals, Sequence
    ):
        raise MmqaP1CoreError("anchor ordinals must be an array")
    anchors = tuple(sorted(_strict_int(value, "anchor ordinal", minimum=0) for value in anchor_ordinals))
    if not anchors or len(set(anchors)) != len(anchors):
        raise MmqaP1CoreError("anchors must be nonempty and distinct")
    hops = _strict_int(hop_limit, "closure hop limit", minimum=0)
    if hops > MAX_CLOSURE_HOPS:
        raise MmqaP1CoreError("closure hop limit exceeds two")
    cap = _strict_int(max_nodes, "closure node cap", minimum=1)
    if cap > MAX_CLOSURE_NODES:
        raise MmqaP1CoreError("closure node cap exceeds 96")
    by_ordinal = graph.node_by_ordinal
    if not set(anchors).issubset(by_ordinal):
        raise MmqaP1CoreError("an anchor ordinal is absent from the graph")
    if len(anchors) > cap:
        raise MmqaP1CoreError("anchor count exceeds the closure cap")

    outgoing: dict[int, list[int]] = {ordinal: [] for ordinal in by_ordinal}
    for edge in graph.edges:
        outgoing[edge.source_ordinal].append(edge.target_ordinal)
    for values in outgoing.values():
        values.sort()

    selected = set(anchors)
    frontier = set(anchors)
    for _hop in range(hops):
        candidates = {
            target
            for source in frontier
            for target in outgoing[source]
            if target not in selected
        }
        if not candidates or len(selected) == cap:
            break
        ordered = sorted(
            candidates,
            key=lambda ordinal: (
                *(-value for value in _node_priority(by_ordinal[ordinal])),
                ordinal,
            ),
        )
        chosen = tuple(ordered[: cap - len(selected)])
        selected.update(chosen)
        frontier = set(chosen)

    nodes = tuple(node for node in graph.nodes if node.ordinal in selected)
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.source_ordinal in selected and edge.target_ordinal in selected
    )
    return ProofClosure(ProofGraph(nodes, edges), anchors, hops)


@dataclass(frozen=True, order=True)
class ProofBundle:
    """A canonical two-to-five-node proof-set identity within one item."""

    node_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        values = tuple(self.node_ordinals)
        if (
            not 2 <= len(values) <= MAX_BUNDLE_SIZE
            or values != tuple(sorted(values))
            or len(set(values)) != len(values)
            or any(type(value) is not int or value < 0 for value in values)
        ):
            raise MmqaP1CoreError(
                "bundle ordinals must be two-to-five distinct sorted integers"
            )


def _undirected_adjacency(graph: ProofGraph) -> Mapping[int, frozenset[int]]:
    adjacency: dict[int, set[int]] = {node.ordinal: set() for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source_ordinal].add(edge.target_ordinal)
        adjacency[edge.target_ordinal].add(edge.source_ordinal)
    return {ordinal: frozenset(values) for ordinal, values in adjacency.items()}


def validate_connected_bundle(graph: ProofGraph, bundle: ProofBundle) -> ProofBundle:
    if not isinstance(graph, ProofGraph) or not isinstance(bundle, ProofBundle):
        raise MmqaP1CoreError("connected-bundle validation requires frozen types")
    by_ordinal = graph.node_by_ordinal
    members = set(bundle.node_ordinals)
    if not members.issubset(by_ordinal):
        raise MmqaP1CoreError("bundle contains an ordinal outside the closure")
    kinds = {by_ordinal[ordinal].node_type for ordinal in members}
    if kinds != {ROW, TEXT}:
        raise MmqaP1CoreError("each proof bundle must contain a row and a text node")
    adjacency = _undirected_adjacency(graph)
    visited = {bundle.node_ordinals[0]}
    frontier = list(visited)
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current].intersection(members - visited):
            visited.add(neighbor)
            frontier.append(neighbor)
    if visited != members:
        raise MmqaP1CoreError("proof bundle must be connected by exact typed links")
    return bundle


def validate_bundle_features(
    value: Mapping[str, object] | Sequence[object],
) -> tuple[float, ...]:
    """Accept exactly the fixed content-free E0/E5 feature schema."""

    if isinstance(value, Mapping):
        supplied = set(value)
        expected = set(FEATURE_ORDER)
        if supplied != expected:
            forbidden = sorted(supplied.intersection(FORBIDDEN_FEATURES))
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise MmqaP1CoreError(
                "bundle feature schema drifted; "
                f"forbidden={forbidden}, missing={missing}, extra={extra}"
            )
        raw = tuple(value[name] for name in FEATURE_ORDER)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
        if len(raw) != len(FEATURE_ORDER):
            raise MmqaP1CoreError("bundle feature width drifted")
    else:
        raise MmqaP1CoreError("bundle features must be a mapping or sequence")
    result = tuple(
        _finite_float(feature, f"bundle feature {FEATURE_ORDER[index]}")
        for index, feature in enumerate(raw)
    )
    if any(not 0.0 <= result[index] <= 1.0 for index in range(10)):
        raise MmqaP1CoreError("the first ten bundle features must lie in [0, 1]")
    if not -1.0 <= result[10] <= 0.0:
        raise MmqaP1CoreError("cardinality penalty must lie in [-1, 0]")
    return result


def bundle_feature_vector(graph: ProofGraph, bundle: ProofBundle) -> tuple[float, ...]:
    """Project one connected bundle onto the frozen label-free coordinates."""

    validate_connected_bundle(graph, bundle)
    by_ordinal = graph.node_by_ordinal
    nodes = tuple(by_ordinal[ordinal] for ordinal in bundle.node_ordinals)
    members = set(bundle.node_ordinals)
    rows = tuple(node for node in nodes if node.node_type == ROW)
    texts = tuple(node for node in nodes if node.node_type == TEXT)
    induced = tuple(
        edge
        for edge in graph.edges
        if edge.source_ordinal in members and edge.target_ordinal in members
    )
    directed_pairs = {
        (edge.source_ordinal, edge.target_ordinal) for edge in induced
    }
    reciprocal = 0
    for row in rows:
        for text in texts:
            if (
                (row.ordinal, text.ordinal) in directed_pairs
                and (text.ordinal, row.ordinal) in directed_pairs
            ):
                reciprocal += 1
    size = len(nodes)
    cross_modal_pair_count = len(rows) * len(texts)
    features = (
        math.fsum(node.cross_encoder_relevance for node in nodes) / size,
        min(node.cross_encoder_relevance for node in nodes),
        math.fsum(node.minilm_similarity for node in nodes) / size,
        min(node.minilm_similarity for node in nodes),
        math.fsum(node.entity_anchor for node in nodes) / size,
        math.fsum(node.relation_anchor for node in nodes) / size,
        math.fsum(node.numeric_or_temporal_anchor for node in nodes) / size,
        len(induced) / (2 * cross_modal_pair_count),
        reciprocal / cross_modal_pair_count,
        2 * min(len(rows), len(texts)) / size,
        -(size - 2) / (MAX_BUNDLE_SIZE - 2),
    )
    return validate_bundle_features(features)


def e0_proof_energy(
    graph_or_features: ProofGraph | Mapping[str, object] | Sequence[object],
    bundle: ProofBundle | None = None,
) -> float:
    """Return the single frozen, non-learned E0 proof energy."""

    if isinstance(graph_or_features, ProofGraph):
        if bundle is None:
            raise MmqaP1CoreError("E0 graph scoring requires a proof bundle")
        features = bundle_feature_vector(graph_or_features, bundle)
    else:
        if bundle is not None:
            raise MmqaP1CoreError("standalone E0 features cannot carry a bundle")
        features = validate_bundle_features(graph_or_features)
    return float(math.fsum(weight * value for weight, value in zip(E0_WEIGHTS, features)))


def _bundle_sort_key(graph: ProofGraph, bundle: ProofBundle) -> tuple[object, ...]:
    return (-e0_proof_energy(graph, bundle), bundle.node_ordinals)


def _prune_bundles(
    graph: ProofGraph, bundles: Sequence[ProofBundle], cap: int
) -> tuple[ProofBundle, ...]:
    unique = set(bundles)
    return tuple(sorted(unique, key=lambda row: _bundle_sort_key(graph, row))[:cap])


def enumerate_connected_bundles(
    closure: ProofClosure | ProofGraph,
    *,
    max_bundle_size: int = MAX_BUNDLE_SIZE,
    max_bundles: int = MAX_BUNDLES,
) -> tuple[ProofBundle, ...]:
    """Deterministically beam-enumerate at most 256 connected row/text sets."""

    graph = closure.graph if isinstance(closure, ProofClosure) else closure
    if not isinstance(graph, ProofGraph):
        raise MmqaP1CoreError("bundle enumeration requires a closure or ProofGraph")
    size_cap = _strict_int(max_bundle_size, "bundle size cap", minimum=2)
    if size_cap > MAX_BUNDLE_SIZE:
        raise MmqaP1CoreError("bundle size cap exceeds five")
    bundle_cap = _strict_int(max_bundles, "bundle count cap", minimum=1)
    if bundle_cap > MAX_BUNDLES:
        raise MmqaP1CoreError("bundle count cap exceeds 256")

    initial = tuple(
        ProofBundle(tuple(sorted((edge.source_ordinal, edge.target_ordinal))))
        for edge in graph.edges
    )
    frontier = _prune_bundles(graph, initial, bundle_cap)
    retained = list(frontier)
    adjacency = _undirected_adjacency(graph)
    for target_size in range(3, size_cap + 1):
        expanded: set[ProofBundle] = set()
        for bundle in frontier:
            members = set(bundle.node_ordinals)
            candidates = set().union(
                *(adjacency[ordinal] for ordinal in bundle.node_ordinals)
            ) - members
            for candidate in candidates:
                row = ProofBundle(tuple(sorted((*members, candidate))))
                validate_connected_bundle(graph, row)
                expanded.add(row)
        if not expanded:
            break
        frontier = _prune_bundles(graph, tuple(expanded), bundle_cap)
        if any(len(row.node_ordinals) != target_size for row in frontier):
            raise MmqaP1CoreError("bundle frontier cardinality drifted")
        retained.extend(frontier)
    result = _prune_bundles(graph, tuple(retained), bundle_cap)
    for bundle in result:
        validate_connected_bundle(graph, bundle)
    return result


def select_e0_bundle(
    graph: ProofGraph, bundles: Sequence[ProofBundle]
) -> ProofBundle:
    checked = _validate_bundle_registry(graph, bundles)
    return min(checked, key=lambda row: _bundle_sort_key(graph, row))


def _validate_bundle_registry(
    graph: ProofGraph, bundles: Sequence[ProofBundle]
) -> tuple[ProofBundle, ...]:
    if (
        not isinstance(graph, ProofGraph)
        or isinstance(bundles, (str, bytes))
        or not isinstance(bundles, Sequence)
        or not bundles
        or len(bundles) > MAX_BUNDLES
        or not all(isinstance(bundle, ProofBundle) for bundle in bundles)
    ):
        raise MmqaP1CoreError("bundle registry is empty, oversized, or malformed")
    checked = tuple(bundles)
    if len(set(checked)) != len(checked):
        raise MmqaP1CoreError("bundle registry contains duplicates")
    for bundle in checked:
        validate_connected_bundle(graph, bundle)
    return checked


@dataclass(frozen=True)
class E5TrainingItem:
    """One anonymous conditional bundle slate with one or more gold sets."""

    graph: ProofGraph
    bundles: tuple[ProofBundle, ...]
    admissible_gold_bundles: tuple[ProofBundle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, ProofGraph):
            raise MmqaP1CoreError("E5 training graph is malformed")
        bundles = tuple(sorted(_validate_bundle_registry(self.graph, self.bundles)))
        gold = tuple(sorted(self.admissible_gold_bundles))
        if (
            not gold
            or len(set(gold)) != len(gold)
            or not set(gold).issubset(bundles)
        ):
            raise MmqaP1CoreError(
                "gold bundles must be a nonempty subset of the bundle slate"
            )
        object.__setattr__(self, "bundles", bundles)
        object.__setattr__(self, "admissible_gold_bundles", gold)


def make_e5_training_item(
    graph: ProofGraph,
    bundles: Sequence[ProofBundle],
    admissible_gold_bundles: Sequence[ProofBundle],
) -> E5TrainingItem:
    return E5TrainingItem(
        graph, tuple(bundles), tuple(admissible_gold_bundles)
    )


def _logsumexp(values: np.ndarray) -> float:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise MmqaP1CoreError("log-sum-exp requires a nonempty finite vector")
    maximum = float(np.max(values))
    return maximum + math.log(
        math.fsum(math.exp(float(value - maximum)) for value in values)
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    normalizer = _logsumexp(values)
    probabilities = np.exp(values - normalizer)
    if not np.isfinite(probabilities).all():
        raise MmqaP1CoreError("conditional probabilities are nonfinite")
    return probabilities


def _conditional_loss_gradient(
    beta: np.ndarray,
    feature_slates: Sequence[np.ndarray],
    gold_indices: Sequence[np.ndarray],
) -> tuple[float, np.ndarray]:
    """Lambda-one negative marginal likelihood with multi-gold log-sum-exp."""

    width = len(FEATURE_ORDER)
    if beta.shape != (width,) or not np.isfinite(beta).all():
        raise MmqaP1CoreError("E5 coefficient vector drifted")
    loss = 0.5 * E5_L2 * float(beta @ beta)
    gradient = E5_L2 * beta.copy()
    for features, gold in zip(feature_slates, gold_indices, strict=True):
        if (
            features.ndim != 2
            or features.shape[1] != width
            or gold.ndim != 1
            or gold.size == 0
            or len(set(int(index) for index in gold)) != gold.size
            or np.any(gold < 0)
            or np.any(gold >= features.shape[0])
        ):
            raise MmqaP1CoreError("E5 conditional slate shape drifted")
        # A source-valid item whose entire support was removed by the frozen
        # closure cap has no distinguishable admissible target bundle.  Its
        # preregistered neutral target is the complete slate: p(all | x)=1,
        # hence exactly zero conditional loss and gradient.  Skip explicitly
        # to avoid a floating add/subtract residual while retaining the item
        # in cohort and audit counts.
        if gold.size == features.shape[0]:
            continue
        logits = features @ beta
        gold_logits = logits[gold]
        loss += _logsumexp(logits) - _logsumexp(gold_logits)
        all_probability = _softmax(logits)
        gold_probability = _softmax(gold_logits)
        gradient += features.T @ all_probability
        gradient -= features[gold].T @ gold_probability
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise MmqaP1CoreError("E5 objective became nonfinite")
    return float(loss), gradient


def _numpy_lbfgs(
    objective_gradient,
    width: int,
    *,
    max_iter: int = E5_MAX_ITER,
    memory: int = 10,
) -> tuple[np.ndarray, float, int, bool]:
    """Deterministic float64 L-BFGS with fixed Armijo backtracking."""

    x = np.zeros(width, dtype=np.float64)
    value, gradient = objective_gradient(x)
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    rho_history: list[float] = []
    converged = float(np.max(np.abs(gradient))) <= 1.0e-9
    iterations = 0
    for iteration in range(max_iter):
        if converged:
            break
        q = gradient.copy()
        alphas: list[float] = []
        for s_value, y_value, rho in zip(
            reversed(s_history), reversed(y_history), reversed(rho_history)
        ):
            alpha = rho * float(s_value @ q)
            alphas.append(alpha)
            q -= alpha * y_value
        if s_history:
            latest_s = s_history[-1]
            latest_y = y_history[-1]
            yy = float(latest_y @ latest_y)
            gamma = float(latest_s @ latest_y) / yy if yy > 0.0 else 1.0
        else:
            gamma = 1.0
        direction = gamma * q
        for index, (s_value, y_value, rho) in enumerate(
            zip(s_history, y_history, rho_history)
        ):
            correction = rho * float(y_value @ direction)
            alpha = alphas[len(alphas) - 1 - index]
            direction += s_value * (alpha - correction)
        direction = -direction
        directional = float(gradient @ direction)
        if not math.isfinite(directional) or directional >= 0.0:
            direction = -gradient
            directional = -float(gradient @ gradient)

        step = 1.0
        accepted = False
        for _line_search in range(64):
            candidate = x + step * direction
            candidate_value, candidate_gradient = objective_gradient(candidate)
            if (
                math.isfinite(candidate_value)
                and candidate_value
                <= value + 1.0e-4 * step * directional
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

        s_value = candidate - x
        y_value = candidate_gradient - gradient
        curvature = float(s_value @ y_value)
        threshold = 1.0e-12 * max(
            1.0,
            float(np.linalg.norm(s_value)) * float(np.linalg.norm(y_value)),
        )
        if curvature > threshold:
            if len(s_history) == memory:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_value)
            y_history.append(y_value)
            rho_history.append(1.0 / curvature)
        x = candidate
        value = float(candidate_value)
        gradient = candidate_gradient
        iterations = iteration + 1
        converged = float(np.max(np.abs(gradient))) <= 1.0e-9
        if float(np.max(np.abs(s_value))) <= 1.0e-13 * max(
            1.0, float(np.max(np.abs(x)))
        ):
            converged = float(np.max(np.abs(gradient))) <= 1.0e-7
            break
    return x, value, iterations, converged


@dataclass(frozen=True)
class E5Model:
    """One shared standardized conditional bundle-energy model."""

    population_mean: tuple[float, ...]
    population_std: tuple[float, ...]
    coefficients: tuple[float, ...]
    training_item_count: int
    training_bundle_count: int
    solver: str
    iterations: int
    converged: bool
    objective: float

    def __post_init__(self) -> None:
        mean = validate_bundle_features(self.population_mean)
        std = tuple(
            _finite_float(value, "E5 population standard deviation")
            for value in self.population_std
        )
        coefficients = tuple(
            _finite_float(value, "E5 coefficient") for value in self.coefficients
        )
        if (
            len(std) != len(FEATURE_ORDER)
            or len(coefficients) != len(FEATURE_ORDER)
            or any(value < 0.0 for value in std)
        ):
            raise MmqaP1CoreError("E5 model width or scale drifted")
        _strict_int(self.training_item_count, "E5 training item count", minimum=1)
        _strict_int(self.training_bundle_count, "E5 training bundle count", minimum=2)
        _strict_int(self.iterations, "E5 iteration count", minimum=0)
        if self.solver != "numpy_deterministic_lbfgs_m10_v1":
            raise MmqaP1CoreError("E5 solver identity drifted")
        if type(self.converged) is not bool:
            raise MmqaP1CoreError("E5 convergence flag must be boolean")
        objective = _finite_float(self.objective, "E5 objective")
        object.__setattr__(self, "population_mean", mean)
        object.__setattr__(self, "population_std", std)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "objective", objective)

    def standardize(
        self, features: Mapping[str, object] | Sequence[object]
    ) -> tuple[float, ...]:
        checked = validate_bundle_features(features)
        return tuple(
            0.0 if std == 0.0 else (value - mean) / std
            for value, mean, std in zip(
                checked, self.population_mean, self.population_std
            )
        )

    def energy(self, features: Mapping[str, object] | Sequence[object]) -> float:
        standardized = self.standardize(features)
        value = math.fsum(
            coefficient * feature
            for coefficient, feature in zip(self.coefficients, standardized)
        )
        if not math.isfinite(value):
            raise MmqaP1CoreError("E5 energy is nonfinite")
        return float(value)

    def payload(self) -> dict[str, object]:
        return {
            "study_id": STUDY_ID,
            "version": VERSION,
            "feature_order": list(FEATURE_ORDER),
            "forbidden_id_or_family_features": sorted(FORBIDDEN_FEATURES),
            "population_mean_float64_hex": [
                value.hex() for value in self.population_mean
            ],
            "population_std_float64_hex": [
                value.hex() for value in self.population_std
            ],
            "coefficient_float64_hex": [
                value.hex() for value in self.coefficients
            ],
            "L2": E5_L2,
            "multi_gold_objective": "logsumexp_all_minus_logsumexp_gold",
            "training_item_count": self.training_item_count,
            "training_bundle_count": self.training_bundle_count,
            "solver": self.solver,
            "max_iter": E5_MAX_ITER,
            "iterations": self.iterations,
            "converged": self.converged,
            "objective_float64_hex": self.objective.hex(),
        }


def _training_item_key(item: E5TrainingItem) -> tuple[object, ...]:
    gold = set(item.admissible_gold_bundles)
    return tuple(
        coordinate.hex()
        for bundle in item.bundles
        for coordinate in bundle_feature_vector(item.graph, bundle)
    ) + tuple(int(bundle in gold) for bundle in item.bundles)


def fit_e5_conditional_maxent(
    items: Sequence[E5TrainingItem],
) -> E5Model:
    """Fit the one lambda-one E5 model from anonymous TRAIN-only slates."""

    if (
        isinstance(items, (str, bytes))
        or not isinstance(items, Sequence)
        or not items
        or not all(isinstance(item, E5TrainingItem) for item in items)
    ):
        raise MmqaP1CoreError("E5 fit requires nonempty E5TrainingItem rows")
    checked = tuple(sorted(items, key=_training_item_key))
    raw_slates = tuple(
        np.asarray(
            [bundle_feature_vector(item.graph, bundle) for bundle in item.bundles],
            dtype=np.float64,
        )
        for item in checked
    )
    all_features = np.vstack(raw_slates)
    means = np.mean(all_features, axis=0, dtype=np.float64)
    stds = np.std(all_features, axis=0, ddof=0, dtype=np.float64)
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    standardized_slates = tuple((slate - means) / safe_stds for slate in raw_slates)
    for slate in standardized_slates:
        slate[:, stds == 0.0] = 0.0
    gold_indices = tuple(
        np.asarray(
            [
                index
                for index, bundle in enumerate(item.bundles)
                if bundle in set(item.admissible_gold_bundles)
            ],
            dtype=np.int64,
        )
        for item in checked
    )

    def objective_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        return _conditional_loss_gradient(beta, standardized_slates, gold_indices)

    beta, objective, iterations, converged = _numpy_lbfgs(
        objective_gradient, len(FEATURE_ORDER), max_iter=E5_MAX_ITER
    )
    objective, gradient = objective_gradient(beta)
    if not converged and float(np.max(np.abs(gradient))) > 1.0e-6:
        raise MmqaP1CoreError("deterministic E5 L-BFGS did not converge")
    return E5Model(
        population_mean=tuple(float(value) for value in means),
        population_std=tuple(float(value) for value in stds),
        coefficients=tuple(float(value) for value in beta),
        training_item_count=len(checked),
        training_bundle_count=sum(len(item.bundles) for item in checked),
        solver="numpy_deterministic_lbfgs_m10_v1",
        iterations=iterations,
        converged=converged,
        objective=float(objective),
    )


fit_e5 = fit_e5_conditional_maxent


def select_e5_bundle(
    model: E5Model, graph: ProofGraph, bundles: Sequence[ProofBundle]
) -> ProofBundle:
    if not isinstance(model, E5Model):
        raise MmqaP1CoreError("E5 selection requires a frozen E5Model")
    checked = _validate_bundle_registry(graph, bundles)
    return min(
        checked,
        key=lambda bundle: (
            -model.energy(bundle_feature_vector(graph, bundle)),
            bundle.node_ordinals,
        ),
    )


def rank_bundle_evidence(
    graph: ProofGraph, bundle: ProofBundle
) -> tuple[int, ...]:
    """Order selected evidence without gold, IDs, family, RAW, or HippoRAG rank."""

    validate_connected_bundle(graph, bundle)
    by_ordinal = graph.node_by_ordinal

    def node_score(node: ProofNode) -> float:
        return (
            4.0 * node.cross_encoder_relevance
            + 2.0 * node.minilm_similarity
            + node.entity_anchor
            + 1.5 * node.relation_anchor
            + 0.5 * node.numeric_or_temporal_anchor
        )

    return tuple(
        sorted(
            bundle.node_ordinals,
            key=lambda ordinal: (-node_score(by_ordinal[ordinal]), ordinal),
        )
    )


def _validated_ranking(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MmqaP1CoreError("evidence ranking must be an integer sequence")
    checked = tuple(_strict_int(row, "evidence ordinal", minimum=0) for row in value)
    if not 1 <= len(checked) <= TOP_K or len(set(checked)) != len(checked):
        raise MmqaP1CoreError("evidence ranking must contain one-to-five distinct ordinals")
    return checked


def _validated_gold(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MmqaP1CoreError("binary gold evidence must be an ordinal sequence")
    checked = tuple(_strict_int(row, "gold evidence ordinal", minimum=0) for row in value)
    if not checked or len(set(checked)) != len(checked):
        raise MmqaP1CoreError("binary gold evidence must be nonempty and distinct")
    return checked


def binary_evidence_ndcg_at_5(
    ranking: Sequence[int], gold_evidence_ordinals: Sequence[int]
) -> float:
    """Compute nDCG@5 with binary evidence relevance and no answer metric."""

    ranked = _validated_ranking(ranking)
    gold = frozenset(_validated_gold(gold_evidence_ordinals))
    dcg = math.fsum(
        (1.0 / math.log2(rank + 2)) if ordinal in gold else 0.0
        for rank, ordinal in enumerate(ranked[:TOP_K])
    )
    ideal_hits = min(len(gold), TOP_K)
    ideal = math.fsum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    value = dcg / ideal
    if not 0.0 <= value <= 1.0 + 1.0e-15:
        raise MmqaP1CoreError("binary evidence nDCG escaped [0, 1]")
    return min(1.0, max(0.0, float(value)))


def integer_utility_from_ndcg(ndcg: object) -> int:
    value = _finite_float(ndcg, "binary evidence nDCG@5")
    if not 0.0 <= value <= 1.0:
        raise MmqaP1CoreError("binary evidence nDCG@5 must lie in [0, 1]")
    return math.floor(INTEGER_UTILITY_SCALE * value)


def integer_binary_evidence_utility(
    ranking: Sequence[int], gold_evidence_ordinals: Sequence[int]
) -> int:
    return integer_utility_from_ndcg(
        binary_evidence_ndcg_at_5(ranking, gold_evidence_ordinals)
    )


@dataclass(frozen=True)
class EvidenceScore:
    ndcg_at_5: float
    integer_utility: int
    selected_evidence_count: int
    gold_evidence_count: int


def score_bundle_evidence(
    graph: ProofGraph,
    bundle: ProofBundle,
    gold_evidence_ordinals: Sequence[int],
) -> EvidenceScore:
    ranking = rank_bundle_evidence(graph, bundle)
    gold = _validated_gold(gold_evidence_ordinals)
    ndcg = binary_evidence_ndcg_at_5(ranking, gold)
    return EvidenceScore(
        ndcg_at_5=ndcg,
        integer_utility=integer_utility_from_ndcg(ndcg),
        selected_evidence_count=len(ranking),
        gold_evidence_count=len(gold),
    )


def exact_gain_vs_harm_binomial_tail(gains: int, harms: int) -> Fraction:
    """Return P[Binomial(gains+harms, 1/2) >= gains], excluding ties.

    This exact tail is O(n) in the number of non-tied pairs and therefore
    remains bounded for the frozen 45-item A_hold and M_search blocks.
    """

    gains_checked = _strict_int(gains, "gain count", minimum=0)
    harms_checked = _strict_int(harms, "harm count", minimum=0)
    nonzero = gains_checked + harms_checked
    if nonzero == 0:
        return Fraction(1)
    numerator = sum(
        math.comb(nonzero, value)
        for value in range(gains_checked, nonzero + 1)
    )
    return Fraction(numerator, 2**nonzero)


@dataclass(frozen=True)
class PairedUtilitySummary:
    item_count: int
    total_integer_delta: int
    gains: int
    harms: int
    ties: int
    exact_one_sided_p: Fraction

    def __post_init__(self) -> None:
        for field, value in (
            ("item count", self.item_count),
            ("gain count", self.gains),
            ("harm count", self.harms),
            ("tie count", self.ties),
        ):
            _strict_int(value, field, minimum=0)
        if type(self.total_integer_delta) is not int:
            raise MmqaP1CoreError("total integer delta must be an exact integer")
        if self.gains + self.harms + self.ties != self.item_count:
            raise MmqaP1CoreError("paired utility counts do not sum to item count")
        expected = exact_gain_vs_harm_binomial_tail(self.gains, self.harms)
        if self.exact_one_sided_p != expected:
            raise MmqaP1CoreError("exact gain-vs-harm tail is inconsistent")

    @property
    def positive_total(self) -> bool:
        return self.total_integer_delta > 0

    @property
    def tail_at_most_alpha(self) -> bool:
        return self.exact_one_sided_p <= PROMOTION_ALPHA

    @property
    def passed(self) -> bool:
        return self.positive_total and self.tail_at_most_alpha


def paired_utility_summary(
    challenger_utilities: Sequence[int], incumbent_utilities: Sequence[int]
) -> PairedUtilitySummary:
    if (
        isinstance(challenger_utilities, (str, bytes))
        or isinstance(incumbent_utilities, (str, bytes))
        or not isinstance(challenger_utilities, Sequence)
        or not isinstance(incumbent_utilities, Sequence)
        or not challenger_utilities
        or len(challenger_utilities) != len(incumbent_utilities)
    ):
        raise MmqaP1CoreError("paired utility vectors must be nonempty and aligned")

    def utility(value: object) -> int:
        result = _strict_int(value, "integer utility", minimum=0)
        if result > INTEGER_UTILITY_SCALE:
            raise MmqaP1CoreError("integer utility exceeds its frozen scale")
        return result

    deltas = tuple(
        utility(left) - utility(right)
        for left, right in zip(
            challenger_utilities, incumbent_utilities, strict=True
        )
    )
    gains = sum(value > 0 for value in deltas)
    harms = sum(value < 0 for value in deltas)
    ties = len(deltas) - gains - harms
    return PairedUtilitySummary(
        item_count=len(deltas),
        total_integer_delta=sum(deltas),
        gains=gains,
        harms=harms,
        ties=ties,
        exact_one_sided_p=exact_gain_vs_harm_binomial_tail(gains, harms),
    )


@dataclass(frozen=True)
class PromotionDecision:
    comparison: PairedUtilitySummary
    promoted: bool
    m_search_authorized: bool
    status: str


def decide_a_hold_promotion(
    e5_integer_utilities: Sequence[int], e0_integer_utilities: Sequence[int]
) -> PromotionDecision:
    """Apply the sole A_hold rule; there is no identifiability or family gate."""

    comparison = paired_utility_summary(
        e5_integer_utilities, e0_integer_utilities
    )
    promoted = comparison.passed
    return PromotionDecision(
        comparison=comparison,
        promoted=promoted,
        m_search_authorized=promoted,
        status="promoted_open_M_search" if promoted else "valid_nonpromotion_M_search_sealed",
    )


@dataclass(frozen=True)
class MSearchDecision:
    authorized: bool
    comparison: PairedUtilitySummary | None
    improved: bool
    status: str


def decide_m_search(
    promotion: PromotionDecision,
    e5_integer_utilities: Sequence[int] | None = None,
    e0_integer_utilities: Sequence[int] | None = None,
) -> MSearchDecision:
    """Keep M sealed after nonpromotion, otherwise apply the same exact rule."""

    if not isinstance(promotion, PromotionDecision):
        raise MmqaP1CoreError("M_search requires the committed A_hold decision")
    if not promotion.promoted:
        if e5_integer_utilities is not None or e0_integer_utilities is not None:
            raise MmqaP1CoreError("M_search utilities cannot be supplied after nonpromotion")
        return MSearchDecision(False, None, False, "sealed_after_A_hold_nonpromotion")
    if e5_integer_utilities is None or e0_integer_utilities is None:
        raise MmqaP1CoreError("authorized M_search requires both utility vectors")
    comparison = paired_utility_summary(
        e5_integer_utilities, e0_integer_utilities
    )
    improved = comparison.passed
    return MSearchDecision(
        True,
        comparison,
        improved,
        "valid_L5_improvement" if improved else "valid_no_L5_improvement",
    )


__all__ = [
    "STUDY_ID",
    "VERSION",
    "ROW",
    "TEXT",
    "NODE_TYPES",
    "ROW_TO_TEXT",
    "TEXT_TO_ROW",
    "EDGE_TYPES",
    "MAX_CLOSURE_NODES",
    "MAX_CLOSURE_HOPS",
    "MAX_BUNDLE_SIZE",
    "MAX_BUNDLES",
    "TOP_K",
    "INTEGER_UTILITY_SCALE",
    "E5_L2",
    "E5_MAX_ITER",
    "PROMOTION_ALPHA",
    "FEATURE_ORDER",
    "E0_WEIGHTS",
    "FORBIDDEN_FEATURES",
    "MmqaP1CoreError",
    "ProofNode",
    "TypedLinkEdge",
    "ProofGraph",
    "ProofClosure",
    "ProofBundle",
    "E5TrainingItem",
    "E5Model",
    "EvidenceScore",
    "PairedUtilitySummary",
    "PromotionDecision",
    "MSearchDecision",
    "build_query_local_closure",
    "validate_connected_bundle",
    "validate_bundle_features",
    "bundle_feature_vector",
    "e0_proof_energy",
    "enumerate_connected_bundles",
    "select_e0_bundle",
    "make_e5_training_item",
    "fit_e5_conditional_maxent",
    "fit_e5",
    "select_e5_bundle",
    "rank_bundle_evidence",
    "binary_evidence_ndcg_at_5",
    "integer_utility_from_ndcg",
    "integer_binary_evidence_utility",
    "score_bundle_evidence",
    "exact_gain_vs_harm_binomial_tail",
    "paired_utility_summary",
    "decide_a_hold_promotion",
    "decide_m_search",
]
