from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from assumption_agent.benchmarks import eraser_evidence_inference_r7_operator_v1 as subject


def _units(count: int = 12) -> tuple[subject.SentenceUnit, ...]:
    return tuple(
        subject.SentenceUnit(
            sentence_ordinal=ordinal,
            start_token=ordinal * 2,
            end_token=(ordinal + 1) * 2,
            sentence_sha256=f"{ordinal + 1:064x}",
        )
        for ordinal in range(count)
    )


def _facets() -> tuple[
    subject.OfficialIcoFacet,
    subject.OfficialIcoFacet,
    subject.OfficialIcoFacet,
]:
    return subject.make_official_ico_facets(
        intervention_sha256="a" * 64,
        comparator_sha256="b" * 64,
        outcome_sha256="c" * 64,
    )


def _tensor(
    rows: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    *,
    dense: tuple[int, ...],
    query_sha256: str = "d" * 64,
) -> subject.QuerySemanticTensor:
    return subject.make_query_semantic_tensor(
        query_sha256=query_sha256,
        facets=_facets(),
        facet_similarity_ints=rows,
        dense_relevance_ints=dense,
    )


def _path_fixture() -> tuple[
    subject.QueryAnchoredSentenceGraph,
    subject.QuerySemanticTensor,
]:
    # Dense ranks 0..4 first and sentence 10 sixth.  All three official facets
    # have exactly one direct anchor at sentence 8, so 8 -> 9 -> 10 is the
    # only witness that makes sentence 10 path-authorized.
    count = 12
    anchor_row = tuple(100 if ordinal == 8 else 0 for ordinal in range(count))
    tensor = _tensor(
        (anchor_row, anchor_row, anchor_row),
        dense=(100, 99, 98, 97, 96, -100, -100, -100, -100, -100, 95, -101),
    )
    return (
        subject.build_query_anchored_graph(
            units=_units(count), semantic_tensor=tensor
        ),
        tensor,
    )


def test_exact_official_ico_tensor_and_query_bound_graph_are_self_hashed() -> None:
    graph, tensor = _path_fixture()

    assert tuple(facet.facet_type for facet in tensor.facets) == subject.FACET_TYPES
    assert subject.verify_query_semantic_tensor(tensor) == tensor.tensor_sha256
    assert (
        subject.verify_query_anchored_graph(graph, tensor) == graph.graph_sha256
    )
    adjacency = [
        edge for edge in graph.edges if edge.edge_type == subject.ADJACENT_SENTENCE
    ]
    anchors = [
        edge for edge in graph.edges if edge.edge_type == subject.OFFICIAL_ICO_ANCHOR
    ]
    assert len(adjacency) == len(graph.units) - 1
    assert {
        (edge.left_sentence_ordinal, edge.right_sentence_ordinal)
        for edge in adjacency
    } == {(ordinal, ordinal + 1) for ordinal in range(len(graph.units) - 1)}
    assert {(edge.facet_i, edge.right_sentence_ordinal) for edge in anchors} == {
        (0, 8),
        (1, 8),
        (2, 8),
    }


def test_direct_anchor_fanout_is_exact_top_eight_positive_without_threshold_gate() -> None:
    count = 12
    tied_positive = (1,) * 10 + (0, -1)
    tensor = _tensor(
        (tied_positive, tied_positive, tied_positive),
        dense=tuple(range(count)),
    )
    graph = subject.build_query_anchored_graph(
        units=_units(count), semantic_tensor=tensor
    )
    anchors_by_facet = {
        facet_i: tuple(
            edge.right_sentence_ordinal
            for edge in graph.edges
            if edge.edge_type == subject.OFFICIAL_ICO_ANCHOR
            and edge.facet_i == facet_i
        )
        for facet_i in range(3)
    }

    assert anchors_by_facet == {0: tuple(range(8)), 1: tuple(range(8)), 2: tuple(range(8))}
    assert "raw" not in inspect.signature(subject.run_action).parameters
    assert "hippo" not in inspect.signature(subject.run_action).parameters


def test_exhaustive_paths_and_canonical_facet_terminal_map_are_complete() -> None:
    graph, tensor = _path_fixture()

    paths = subject.enumerate_atomic_paths(graph=graph, semantic_tensor=tensor)
    maps = subject.canonical_facet_terminal_maps(paths)
    terminal_ten = next(row for row in maps if row.terminal_sentence_ordinal == 10)

    # For each facet: the anchor, two one-hop directions, and two simple
    # two-hop directions in the interior of a sentence chain.
    assert len(paths) == 3 * 5
    assert tuple(row.terminal_sentence_ordinal for row in maps) == (6, 7, 8, 9, 10)
    assert tuple(path.facet_i for path in terminal_ten.facet_paths) == (0, 1, 2)
    assert all(path.sentence_ordinals == (8, 9, 10) for path in terminal_ten.facet_paths)
    assert all(path.adjacency_hop_count == 2 for path in terminal_ten.facet_paths)


def test_r7_selects_rank_six_endpoint_and_retains_no_raw_member() -> None:
    graph, tensor = _path_fixture()

    raw = subject.run_action(
        recipe_id="R0_DENSE5", graph=graph, semantic_tensor=tensor
    )
    r7 = subject.run_action(
        recipe_id="R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE",
        graph=graph,
        semantic_tensor=tensor,
    )

    assert raw.behavior.output_top5 == (0, 1, 2, 3, 4)
    assert r7.behavior.output_top5 == (10, 8, 7, 9, 6)
    assert set(r7.behavior.output_top5).isdisjoint(raw.behavior.output_top5)
    first = r7.behavior.selection_steps[0]
    assert first.selected_sentence_ordinal == 10
    assert first.newly_covered_facets == first.reachable_facets == (0, 1, 2)
    assert tuple(path.sentence_ordinals for path in first.facet_paths) == (
        (8, 9, 10),
        (8, 9, 10),
        (8, 9, 10),
    )
    assert r7.behavior.dense_fill_count == 0
    assert all(step.disposition == "query_anchored_path" for step in r7.behavior.selection_steps)


def test_no_positive_anchor_edges_dense_fills_only_after_path_exhaustion() -> None:
    count = 8
    no_anchors = ((0,) * count, (-1,) * count, (0,) * count)
    tensor = _tensor(no_anchors, dense=tuple(100 - ordinal for ordinal in range(count)))
    graph = subject.build_query_anchored_graph(
        units=_units(count), semantic_tensor=tensor
    )

    raw, r7 = subject.run_all_actions(graph=graph, semantic_tensor=tensor)

    assert not [edge for edge in graph.edges if edge.edge_type == subject.OFFICIAL_ICO_ANCHOR]
    assert r7.behavior.output_top5 == raw.behavior.output_top5 == (0, 1, 2, 3, 4)
    assert r7.behavior.exhaustive_path_count == 0
    assert r7.behavior.terminal_path_map_count == 0
    assert r7.behavior.dense_fill_count == subject.TOP_K
    assert not r7.used_edge_ids and not r7.edge_deletion_witnesses


def test_every_used_edge_has_one_nonrecursive_deletion_rerun() -> None:
    graph, tensor = _path_fixture()
    trace = subject.run_action(
        recipe_id="R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE",
        graph=graph,
        semantic_tensor=tensor,
    )
    by_edge = {witness.edge_i: witness for witness in trace.edge_deletion_witnesses}
    bridge = next(
        edge
        for edge in graph.edges
        if edge.edge_type == subject.ADJACENT_SENTENCE
        and edge.left_sentence_ordinal == 9
        and edge.right_sentence_ordinal == 10
    )
    redundant_anchor = next(
        edge
        for edge in graph.edges
        if edge.edge_type == subject.OFFICIAL_ICO_ANCHOR and edge.facet_i == 0
    )

    assert tuple(by_edge) == trace.used_edge_ids
    assert len(by_edge) == len(set(trace.used_edge_ids))
    assert by_edge[bridge.edge_i].selected_ordinals_changed is True
    assert by_edge[bridge.edge_i].counterfactual_behavior.output_top5 != trace.behavior.output_top5
    assert by_edge[redundant_anchor.edge_i].selected_ordinals_changed is False
    assert by_edge[redundant_anchor.edge_i].witness_path_receipts_changed is True
    assert all(
        witness.counterfactual_behavior.excluded_edge_i == witness.edge_i
        for witness in by_edge.values()
    )


def test_action_and_behavior_hashes_are_independent_and_reconstruct_from_inputs() -> None:
    graph, tensor = _path_fixture()
    trace = subject.run_action(
        recipe_id="R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE",
        graph=graph,
        semantic_tensor=tensor,
    )

    assert trace.trace_sha256 != trace.behavior.behavior_sha256
    assert subject.verify_behavior_trace(
        trace.behavior, graph=graph, semantic_tensor=tensor
    ) == trace.behavior.behavior_sha256
    assert subject.verify_action_trace(
        trace, graph=graph, semantic_tensor=tensor
    ) == trace.trace_sha256
    assert subject.recompute_behavior_sha256(trace.behavior) == trace.behavior.behavior_sha256
    assert subject.recompute_action_trace_sha256(trace) == trace.trace_sha256


def test_nested_graph_tensor_behavior_and_deletion_tamper_fail_closed() -> None:
    graph, tensor = _path_fixture()
    trace = subject.run_action(
        recipe_id="R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE",
        graph=graph,
        semantic_tensor=tensor,
    )

    with pytest.raises(subject.EraserR7OperatorError, match="graph self hash drifted"):
        subject.verify_query_anchored_graph(
            replace(graph, edges=graph.edges[:-1]), tensor
        )

    changed = list(tensor.rows[0].similarity_ints)
    changed[-1] += 1
    tampered_tensor = replace(
        tensor,
        rows=(replace(tensor.rows[0], similarity_ints=tuple(changed)), *tensor.rows[1:]),
    )
    with pytest.raises(subject.EraserR7OperatorError, match="tensor self hash drifted"):
        subject.verify_query_semantic_tensor(tampered_tensor)

    tampered_behavior = replace(trace.behavior, output_top5=(0, 8, 7, 9, 6))
    with pytest.raises(subject.EraserR7OperatorError, match="behavior trace self hash drifted"):
        subject.verify_behavior_trace(tampered_behavior)

    first_witness = trace.edge_deletion_witnesses[0]
    tampered_witness = replace(first_witness, selected_ordinals_changed=True)
    tampered_trace = replace(
        trace,
        edge_deletion_witnesses=(tampered_witness, *trace.edge_deletion_witnesses[1:]),
    )
    with pytest.raises(subject.EraserR7OperatorError, match="action trace self hash drifted"):
        subject.verify_action_trace(tampered_trace)


def test_complete_scan_receipts_and_five_way_sentence_deletion_are_exact() -> None:
    graph, tensor = _path_fixture()
    raw, r7 = subject.run_all_actions(graph=graph, semantic_tensor=tensor)

    assert raw.behavior.candidate_score_evaluations == len(graph.units)
    assert r7.behavior.candidate_score_evaluations == subject.TOP_K * len(graph.units)
    assert raw.behavior.semantic_cell_scan_count == r7.behavior.semantic_cell_scan_count == 3 * len(graph.units)
    assert r7.behavior.exhaustive_path_count == 15
    deltas = subject.sentence_leave_one_out_coverage_deltas(
        tensor, r7.behavior.output_top5
    )
    assert len(deltas) == subject.TOP_K
    assert all(len(row) == 3 for row in deltas)


def test_wrong_ico_order_incomplete_tensor_and_noncontiguous_spans_are_rejected() -> None:
    bad_facets = (
        subject.OfficialIcoFacet(0, "INTERVENTION", "a" * 64),
        subject.OfficialIcoFacet(1, "COMPARATOR", "b" * 64),
        subject.OfficialIcoFacet(2, "OUTCOME", "c" * 64),
    )
    with pytest.raises(subject.EraserR7OperatorError, match="exact I/C/O rows"):
        subject.make_query_semantic_tensor(
            query_sha256="d" * 64,
            facets=bad_facets,
            facet_similarity_ints=((0,) * 5, (0,) * 5),
            dense_relevance_ints=(0,) * 5,
        )

    tensor = _tensor(((0,) * 5, (0,) * 5, (0,) * 5), dense=(0,) * 5)
    units = list(_units(5))
    units[2] = replace(units[2], start_token=units[2].start_token + 1)
    with pytest.raises(subject.EraserR7OperatorError, match="not contiguous"):
        subject.build_query_anchored_graph(units=units, semantic_tensor=tensor)
