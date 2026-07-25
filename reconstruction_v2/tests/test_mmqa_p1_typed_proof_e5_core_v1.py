from __future__ import annotations

import ast
from fractions import Fraction
import inspect
import math

import numpy as np
import pytest

from assumption_agent.benchmarks import mmqa_p1_typed_proof_e5_core_v1 as core


def _node(
    ordinal: int,
    node_type: str,
    *,
    minilm: float = 0.5,
    cross_encoder: float = 0.5,
    entity: int = 0,
    relation: int = 0,
    numeric: int = 0,
) -> core.ProofNode:
    return core.ProofNode(
        ordinal,
        node_type,
        minilm,
        cross_encoder,
        entity,
        relation,
        numeric,
    )


def _edge(source: int, target: int, edge_type: str) -> core.TypedLinkEdge:
    return core.TypedLinkEdge(source, target, edge_type)


def _graph(
    nodes: tuple[core.ProofNode, ...],
    edges: tuple[core.TypedLinkEdge, ...],
) -> core.ProofGraph:
    ordered = tuple(
        sorted(
            edges,
            key=lambda row: (
                row.source_ordinal,
                row.target_ordinal,
                core.EDGE_TYPES.index(row.edge_type),
            ),
        )
    )
    return core.ProofGraph(tuple(sorted(nodes, key=lambda row: row.ordinal)), ordered)


def _reciprocal(row: int, text: int) -> tuple[core.TypedLinkEdge, ...]:
    return (
        _edge(row, text, core.ROW_TO_TEXT),
        _edge(text, row, core.TEXT_TO_ROW),
    )


def _training_graph() -> tuple[core.ProofGraph, tuple[core.ProofBundle, ...]]:
    graph = _graph(
        (
            _node(
                0,
                core.ROW,
                minilm=0.8,
                cross_encoder=0.8,
                entity=1,
            ),
            _node(
                1,
                core.TEXT,
                minilm=0.8,
                cross_encoder=0.8,
                entity=1,
            ),
            _node(
                2,
                core.ROW,
                minilm=0.3,
                cross_encoder=0.3,
                relation=1,
            ),
            _node(
                3,
                core.TEXT,
                minilm=0.3,
                cross_encoder=0.3,
                relation=1,
            ),
        ),
        tuple(
            edge
            for row, text in ((0, 1), (0, 3), (2, 1), (2, 3))
            for edge in _reciprocal(row, text)
        ),
    )
    bundles = (
        core.ProofBundle((0, 1)),
        core.ProofBundle((0, 3)),
        core.ProofBundle((1, 2)),
        core.ProofBundle((2, 3)),
    )
    return graph, bundles


def test_module_is_the_frozen_offline_study_and_imports_no_io_stack() -> None:
    assert core.STUDY_ID == "MMQA_P1_LOCAL_PROOF_E5_V1"
    assert core.E5_L2 == 1.0
    assert core.E5_MAX_ITER == 256
    assert core.MAX_BUNDLES == 256
    assert core.MAX_BUNDLE_SIZE == 5
    assert not ({"family", "item_id", "node_ordinal"} & set(core.FEATURE_ORDER))

    tree = ast.parse(inspect.getsource(core))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported_roots.intersection(
        {
            "aiohttp",
            "httpx",
            "os",
            "pathlib",
            "requests",
            "socket",
            "subprocess",
            "transformers",
            "urllib",
        }
    )


def test_nodes_and_exact_directed_edges_fail_closed() -> None:
    with pytest.raises(core.MmqaP1CoreError, match=r"\[0, 1\]"):
        _node(0, core.ROW, minilm=1.01)
    with pytest.raises(core.MmqaP1CoreError, match="binary"):
        _node(0, core.ROW, entity=2)
    with pytest.raises(core.MmqaP1CoreError, match="ROW or TEXT"):
        _node(0, "IMAGE")

    nodes = (_node(0, core.ROW), _node(1, core.TEXT), _node(2, core.ROW))
    with pytest.raises(core.MmqaP1CoreError, match="direction/type"):
        _graph(nodes, (_edge(0, 1, core.TEXT_TO_ROW),))
    with pytest.raises(core.MmqaP1CoreError, match="direction/type"):
        _graph(nodes, (_edge(0, 2, core.ROW_TO_TEXT),))
    with pytest.raises(core.MmqaP1CoreError, match="canonically sorted"):
        core.ProofGraph(
            nodes,
            (
                _edge(1, 0, core.TEXT_TO_ROW),
                _edge(0, 1, core.ROW_TO_TEXT),
            ),
        )


def test_closure_follows_direction_for_exactly_the_frozen_hops() -> None:
    graph = _graph(
        (
            _node(0, core.ROW),
            _node(1, core.TEXT),
            _node(2, core.ROW),
            _node(3, core.TEXT),
        ),
        (
            _edge(0, 1, core.ROW_TO_TEXT),
            _edge(1, 2, core.TEXT_TO_ROW),
            _edge(2, 3, core.ROW_TO_TEXT),
        ),
    )
    closure = core.build_query_local_closure(graph, (0,))
    assert tuple(node.ordinal for node in closure.graph.nodes) == (0, 1, 2)
    assert tuple(
        (edge.source_ordinal, edge.target_ordinal) for edge in closure.graph.edges
    ) == ((0, 1), (1, 2))

    reverse_dead_end = core.build_query_local_closure(graph, (3,))
    assert tuple(node.ordinal for node in reverse_dead_end.graph.nodes) == (3,)
    assert reverse_dead_end.graph.edges == ()
    with pytest.raises(core.MmqaP1CoreError, match="exceeds two"):
        core.build_query_local_closure(graph, (0,), hop_limit=3)


def test_closure_cap_uses_frozen_feature_priority_not_ordinal() -> None:
    graph = _graph(
        (
            _node(0, core.ROW),
            _node(1, core.TEXT, minilm=0.1, cross_encoder=0.1),
            _node(2, core.TEXT, minilm=0.9, cross_encoder=0.9),
        ),
        (
            _edge(0, 1, core.ROW_TO_TEXT),
            _edge(0, 2, core.ROW_TO_TEXT),
        ),
    )
    closure = core.build_query_local_closure(graph, (0,), max_nodes=2)
    assert tuple(node.ordinal for node in closure.graph.nodes) == (0, 2)
    assert closure.graph.edges == (_edge(0, 2, core.ROW_TO_TEXT),)


def test_bundle_features_are_connected_typed_and_ordinal_invariant() -> None:
    graph = _graph(
        (
            _node(
                0,
                core.ROW,
                minilm=0.7,
                cross_encoder=0.9,
                entity=1,
                numeric=1,
            ),
            _node(
                1,
                core.TEXT,
                minilm=0.5,
                cross_encoder=0.3,
                relation=1,
            ),
        ),
        _reciprocal(0, 1),
    )
    bundle = core.ProofBundle((0, 1))
    features = core.bundle_feature_vector(graph, bundle)
    assert features == (
        0.6,
        0.3,
        0.6,
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        0.0,
    )

    renumbered = _graph(
        (
            _node(
                10,
                core.ROW,
                minilm=0.7,
                cross_encoder=0.9,
                entity=1,
                numeric=1,
            ),
            _node(
                20,
                core.TEXT,
                minilm=0.5,
                cross_encoder=0.3,
                relation=1,
            ),
        ),
        _reciprocal(10, 20),
    )
    assert core.bundle_feature_vector(
        renumbered, core.ProofBundle((10, 20))
    ) == features

    invalid = dict(zip(core.FEATURE_ORDER, features, strict=True))
    invalid["family"] = 1
    with pytest.raises(core.MmqaP1CoreError, match="family"):
        core.validate_bundle_features(invalid)


def test_bundle_requires_both_modalities_and_exact_link_connectivity() -> None:
    graph = _graph(
        (
            _node(0, core.ROW),
            _node(1, core.TEXT),
            _node(2, core.TEXT),
        ),
        (_edge(0, 1, core.ROW_TO_TEXT),),
    )
    with pytest.raises(core.MmqaP1CoreError, match="row and a text"):
        core.validate_connected_bundle(graph, core.ProofBundle((1, 2)))
    with pytest.raises(core.MmqaP1CoreError, match="connected"):
        core.validate_connected_bundle(graph, core.ProofBundle((0, 2)))


def test_connected_bundle_enumeration_is_deterministic_and_hard_capped() -> None:
    nodes = tuple(
        [_node(index, core.ROW) for index in range(17)]
        + [_node(17 + index, core.TEXT) for index in range(17)]
    )
    edges = tuple(
        edge
        for row in range(17)
        for text in range(17, 34)
        for edge in _reciprocal(row, text)
    )
    graph = _graph(nodes, edges)
    first = core.enumerate_connected_bundles(graph, max_bundle_size=2)
    second = core.enumerate_connected_bundles(graph, max_bundle_size=2)
    assert first == second
    assert len(first) == 256
    assert all(len(bundle.node_ordinals) == 2 for bundle in first)
    assert all(
        core.validate_connected_bundle(graph, bundle) == bundle for bundle in first
    )
    with pytest.raises(core.MmqaP1CoreError, match="exceeds 256"):
        core.enumerate_connected_bundles(graph, max_bundles=257)


def test_fixed_e0_prefers_relevance_while_e5_can_rewrite_bundle_energy() -> None:
    graph, bundles = _training_graph()
    assert core.select_e0_bundle(graph, bundles) == core.ProofBundle((0, 1))

    single_gold = core.make_e5_training_item(graph, bundles, (bundles[3],))
    multi_gold = core.make_e5_training_item(
        graph, bundles, (bundles[2], bundles[3])
    )
    items = (single_gold, multi_gold, single_gold, multi_gold) * 2
    first = core.fit_e5(items)
    second = core.fit_e5(tuple(reversed(items)))

    assert first.payload() == second.payload()
    assert first.converged
    assert first.solver == "numpy_deterministic_lbfgs_m10_v1"
    assert first.payload()["L2"] == 1.0
    assert (
        first.payload()["multi_gold_objective"]
        == "logsumexp_all_minus_logsumexp_gold"
    )
    assert core.select_e5_bundle(first, graph, bundles) == core.ProofBundle((2, 3))
    assert first.coefficients[5] > 0.0
    assert first.coefficients[0] < 0.0


def test_e5_training_requires_gold_as_a_nonempty_subset_of_enumerated_bundles() -> None:
    graph, bundles = _training_graph()
    with pytest.raises(core.MmqaP1CoreError, match="nonempty subset"):
        core.make_e5_training_item(graph, bundles, ())
    neutral = core.make_e5_training_item(graph, bundles, bundles)
    assert neutral.admissible_gold_bundles == tuple(sorted(bundles))
    with pytest.raises(core.MmqaP1CoreError, match="nonempty subset"):
        core.make_e5_training_item(
            graph, bundles, (core.ProofBundle((0, 1, 2)),)
        )


def test_full_slate_neutral_target_has_exactly_zero_loss_and_gradient() -> None:
    graph, bundles = _training_graph()
    neutral = core.make_e5_training_item(graph, bundles, bundles)
    width = len(core.FEATURE_ORDER)
    features = np.asarray(
        [core.bundle_feature_vector(graph, bundle) for bundle in neutral.bundles],
        dtype=np.float64,
    )
    beta = np.linspace(-0.2, 0.2, width, dtype=np.float64)
    all_indices = np.arange(len(neutral.bundles), dtype=np.int64)
    neutral_value, neutral_gradient = core._conditional_loss_gradient(
        beta, (features,), (all_indices,)
    )
    assert neutral_value == 0.5 * float(beta @ beta)
    assert np.array_equal(neutral_gradient, beta)

    informative_indices = np.asarray([0], dtype=np.int64)
    informative = core._conditional_loss_gradient(
        beta, (features,), (informative_indices,)
    )
    mixed = core._conditional_loss_gradient(
        beta,
        (features, features),
        (informative_indices, all_indices),
    )
    assert mixed[0] == informative[0]
    assert np.array_equal(mixed[1], informative[1])

    neutral_model = core.fit_e5((neutral,))
    assert neutral_model.training_item_count == 1
    assert neutral_model.training_bundle_count == len(bundles)
    assert neutral_model.objective == 0.0
    assert neutral_model.iterations == 0
    assert all(value == 0.0 for value in neutral_model.coefficients)

    informative_item = core.make_e5_training_item(graph, bundles, (bundles[0],))
    mixed_model = core.fit_e5((informative_item, neutral))
    assert mixed_model.training_item_count == 2
    assert mixed_model.training_bundle_count == 2 * len(bundles)
    assert mixed_model.converged


def test_multi_gold_logsumexp_objective_gradient_matches_finite_difference() -> None:
    width = len(core.FEATURE_ORDER)
    features = np.asarray(
        [
            np.linspace(-0.4, 0.5, width),
            np.linspace(0.3, -0.2, width),
            np.linspace(-0.1, 0.6, width),
        ],
        dtype=np.float64,
    )
    gold = np.asarray([1, 2], dtype=np.int64)
    beta = np.linspace(-0.2, 0.2, width, dtype=np.float64)
    value, gradient = core._conditional_loss_gradient(beta, (features,), (gold,))

    logits = features @ beta
    expected = (
        0.5 * float(beta @ beta)
        + float(np.logaddexp.reduce(logits))
        - float(np.logaddexp.reduce(logits[gold]))
    )
    assert value == pytest.approx(expected, abs=1.0e-12)
    epsilon = 1.0e-6
    numerical = []
    for index in range(width):
        offset = np.zeros(width, dtype=np.float64)
        offset[index] = epsilon
        upper, _ = core._conditional_loss_gradient(
            beta + offset, (features,), (gold,)
        )
        lower, _ = core._conditional_loss_gradient(
            beta - offset, (features,), (gold,)
        )
        numerical.append((upper - lower) / (2 * epsilon))
    assert gradient == pytest.approx(numerical, abs=1.0e-7)


def test_binary_evidence_ndcg_at_5_and_integer_utility_are_offline_and_fixed() -> None:
    assert core.binary_evidence_ndcg_at_5((0, 1), (0, 1)) == 1.0
    expected = (1.0 + 1.0 / math.log2(4)) / (
        1.0 + 1.0 / math.log2(3)
    )
    value = core.binary_evidence_ndcg_at_5((0, 2, 1), (0, 1))
    assert value == pytest.approx(expected)
    assert core.integer_binary_evidence_utility((0, 2, 1), (0, 1)) == math.floor(
        core.INTEGER_UTILITY_SCALE * expected
    )
    assert core.binary_evidence_ndcg_at_5((2, 3), (0, 1)) == 0.0
    with pytest.raises(core.MmqaP1CoreError, match="distinct"):
        core.binary_evidence_ndcg_at_5((0, 0), (0, 1))


def test_bundle_evidence_ranking_and_scoring_never_use_gold_for_order() -> None:
    graph = _graph(
        (
            _node(0, core.ROW, minilm=0.2, cross_encoder=0.2),
            _node(1, core.TEXT, minilm=0.9, cross_encoder=0.9),
            _node(2, core.TEXT, minilm=0.5, cross_encoder=0.5),
        ),
        (
            *_reciprocal(0, 1),
            *_reciprocal(0, 2),
        ),
    )
    bundle = core.ProofBundle((0, 1, 2))
    assert core.rank_bundle_evidence(graph, bundle) == (1, 2, 0)
    score = core.score_bundle_evidence(graph, bundle, (0, 1))
    assert score.selected_evidence_count == 3
    assert score.gold_evidence_count == 2
    assert score.integer_utility == core.integer_utility_from_ndcg(score.ndcg_at_5)


@pytest.mark.parametrize(
    ("gains", "harms", "expected"),
    [
        (0, 0, Fraction(1)),
        (4, 0, Fraction(1, 16)),
        (1, 1, Fraction(3, 4)),
        (3, 2, Fraction(1, 2)),
        (0, 4, Fraction(1)),
    ],
)
def test_exact_ties_excluded_gain_vs_harm_binomial_tail(
    gains: int, harms: int, expected: Fraction
) -> None:
    assert core.exact_gain_vs_harm_binomial_tail(gains, harms) == expected


def test_paired_summary_excludes_ties_and_retains_total_integer_delta() -> None:
    result = core.paired_utility_summary(
        (12, 4, 7, 10, 10),
        (10, 5, 7, 9, 10),
    )
    assert result.item_count == 5
    assert result.total_integer_delta == 2
    assert (result.gains, result.harms, result.ties) == (2, 1, 2)
    assert result.exact_one_sided_p == Fraction(1, 2)
    assert not result.passed


def test_a_hold_promotion_is_the_only_authority_for_m_search() -> None:
    nonpromotion = core.decide_a_hold_promotion((10, 9), (9, 10))
    assert not nonpromotion.promoted
    sealed = core.decide_m_search(nonpromotion)
    assert not sealed.authorized
    assert sealed.comparison is None
    assert sealed.status == "sealed_after_A_hold_nonpromotion"
    with pytest.raises(core.MmqaP1CoreError, match="cannot be supplied"):
        core.decide_m_search(nonpromotion, (10,), (9,))

    promotion = core.decide_a_hold_promotion((10, 10, 10, 10), (9, 9, 9, 9))
    assert promotion.promoted
    assert promotion.m_search_authorized
    assert promotion.comparison.exact_one_sided_p == Fraction(1, 16)
    improved = core.decide_m_search(
        promotion, (100, 100, 100, 100), (90, 90, 90, 90)
    )
    assert improved.authorized
    assert improved.improved
    assert improved.status == "valid_L5_improvement"

    failed = core.decide_m_search(
        promotion, (90, 90, 90, 90), (100, 100, 100, 100)
    )
    assert failed.authorized
    assert not failed.improved
    assert failed.status == "valid_no_L5_improvement"


def test_utility_vectors_are_integer_scaled_and_aligned() -> None:
    with pytest.raises(core.MmqaP1CoreError, match="aligned"):
        core.paired_utility_summary((1, 2), (1,))
    with pytest.raises(core.MmqaP1CoreError, match="exact integer"):
        core.paired_utility_summary((1.0,), (0,))
    with pytest.raises(core.MmqaP1CoreError, match="exceeds"):
        core.paired_utility_summary(
            (core.INTEGER_UTILITY_SCALE + 1,), (0,)
        )
