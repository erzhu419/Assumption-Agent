from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as subject


def _row(
    ordinal: int,
    *,
    table: str | None = None,
    row: int = 0,
    links: tuple[str, ...] = (),
) -> subject.AtomicUnit:
    return subject.AtomicUnit(
        corpus_ordinal=ordinal,
        unit_type="table_row",
        table_key=table or f"table-{ordinal}",
        row_ordinal=row,
        link_target_keys=links,
    )


def _passage(
    ordinal: int,
    *,
    table: str | None = None,
    target: str | None = None,
) -> subject.AtomicUnit:
    return subject.AtomicUnit(
        corpus_ordinal=ordinal,
        unit_type="linked_passage",
        table_key=table or f"table-{ordinal}",
        row_ordinal=None,
        link_target_keys=(target or f"target-{ordinal}",),
    )


def _pad_units(
    rows: tuple[subject.AtomicUnit, ...],
) -> tuple[subject.AtomicUnit, ...]:
    if tuple(unit.corpus_ordinal for unit in rows) != tuple(range(len(rows))):
        raise AssertionError("synthetic units must begin in complete ordinal order")
    return rows + tuple(
        _row(ordinal) for ordinal in range(len(rows), subject.CORPUS_UNIT_COUNT)
    )


def _path_units() -> tuple[subject.AtomicUnit, ...]:
    # 5 -> 6 -> 7 is the only anchor-connected length-two component.
    # 8 <-> 9 is a deliberately disconnected shared-target component.
    return _pad_units(
        (
            _row(0),
            _row(1),
            _row(2),
            _row(3),
            _row(4),
            _row(5, table="anchored", row=0),
            _row(6, table="anchored", row=1),
            _row(7, table="anchored", row=2),
            _row(8, table="disconnected", row=0, links=("shared",)),
            _row(9, table="disconnected", row=2, links=("shared",)),
        )
    )


def _facets() -> tuple[subject.QueryFacet, ...]:
    return (
        subject.make_query_facet(0, "entity", "Anchor Entity"),
        subject.make_query_facet(1, "relation_clause", "won the relation"),
    )


def _tensor(
    coverage: tuple[tuple[int, ...], ...],
    anchors: tuple[tuple[int, ...], ...],
    *,
    dense: tuple[int, ...] | None = None,
    query_byte: str = "a",
) -> subject.QuerySemanticTensor:
    width = len(coverage[0])
    if any(len(row) != width for row in (*coverage, *anchors)):
        raise AssertionError("synthetic semantic rows have inconsistent widths")
    coverage = tuple(
        (*row, *((0,) * (subject.CORPUS_UNIT_COUNT - width)))
        for row in coverage
    )
    anchors = tuple(
        (*row, *((0,) * (subject.CORPUS_UNIT_COUNT - width)))
        for row in anchors
    )
    if dense is None:
        dense = tuple(
            1_000 - ordinal for ordinal in range(subject.CORPUS_UNIT_COUNT)
        )
    else:
        dense = dense + tuple(
            -10_000_000 - ordinal
            for ordinal in range(width, subject.CORPUS_UNIT_COUNT)
        )
    return subject.make_query_semantic_tensor(
        query_sha256=query_byte * 64,
        facets=_facets(),
        semantic_coverage_ints=coverage,
        direct_anchor_strength_ints=anchors,
        dense_relevance_ints=dense,
    )


def _path_tensor(*, query_byte: str = "a") -> subject.QuerySemanticTensor:
    return _tensor(
        (
            (0, 0, 0, 0, 0, 10, 0, 0, 1_000, 900),
            (0, 0, 0, 0, 0, 0, 50, 100, 1_000, 900),
        ),
        (
            (0, 0, 0, 0, 0, 100, 0, 0, 0, 0),
            (0,) * 10,
        ),
        query_byte=query_byte,
    )


def test_graph_has_only_three_frozen_exact_sidecar_edge_families() -> None:
    units = _pad_units(
        (
            _row(0, table="t", row=0, links=("p",)),
            _row(1, table="t", row=1, links=("p", "q")),
            _passage(2, table="t", target="p"),
            _passage(3, table="t", target="q"),
        )
    )
    graph = subject.build_typed_graph(units)
    observed = {edge.public_tuple() for edge in graph.edges}

    assert subject.EDGE_FAMILIES == (
        "same_table_adjacent_row",
        "row_to_linked_passage",
        "same_table_shared_link_target",
    )
    assert observed == {
        (subject.SAME_TABLE_ADJACENT_ROW, 0, 1, subject.INTEGER_SCALE),
        (subject.ROW_TO_LINKED_PASSAGE, 0, 2, subject.INTEGER_SCALE),
        (subject.ROW_TO_LINKED_PASSAGE, 1, 2, subject.INTEGER_SCALE),
        (subject.ROW_TO_LINKED_PASSAGE, 1, 3, subject.INTEGER_SCALE),
        (
            subject.SAME_TABLE_SHARED_LINK_TARGET,
            0,
            1,
            subject.INTEGER_SCALE,
        ),
    }
    assert subject.verify_typed_graph(graph) == graph.graph_sha256


def test_disconnected_high_score_component_cannot_win_query_anchored_path() -> None:
    graph = subject.build_typed_graph(_path_units())
    tensor = _path_tensor()

    trace = subject.run_recipe(
        recipe_id="R3_P6_PATH2_B2",
        graph=graph,
        semantic_tensor=tensor,
    )

    assert trace.selection_steps[0].selected_unit_ordinal == 7
    assert trace.selection_steps[0].path_length == 2
    assert trace.selection_steps[1].selected_unit_ordinal == 5
    assert trace.selection_steps[1].direct_anchor is True
    assert 8 not in trace.output_top5 and 9 not in trace.output_top5


def test_direct_path1_and_path2_recipes_enforce_exact_reachability_depths() -> None:
    graph = subject.build_typed_graph(_path_units())
    tensor = _path_tensor()

    direct = subject.run_recipe(
        recipe_id="R1_P6_DIRECT_B2", graph=graph, semantic_tensor=tensor
    )
    path1 = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=tensor
    )
    path2 = subject.run_recipe(
        recipe_id="R3_P6_PATH2_B2", graph=graph, semantic_tensor=tensor
    )

    assert direct.selection_steps[0].selected_unit_ordinal == 5
    assert direct.selection_steps[0].path_length == 0
    assert direct.selection_steps[1].disposition == "unused_raw_fallback"
    assert tuple(step.selected_unit_ordinal for step in path1.selection_steps) == (6, 5)
    assert tuple(step.path_length for step in path1.selection_steps) == (1, 0)
    assert tuple(step.selected_unit_ordinal for step in path2.selection_steps) == (7, 5)
    assert tuple(step.path_length for step in path2.selection_steps) == (2, 0)


@pytest.mark.parametrize(
    "recipe_id",
    ("R1_P6_DIRECT_B2", "R2_P6_PATH1_B2", "R3_P6_PATH2_B2"),
)
def test_no_positive_query_anchored_candidate_falls_back_exactly_to_dense(
    recipe_id: str,
) -> None:
    graph = subject.build_typed_graph(_path_units())
    tensor = _tensor(((0,) * 10, (0,) * 10), ((0,) * 10, (0,) * 10))

    raw = subject.run_recipe(
        recipe_id="R0_DENSE5", graph=graph, semantic_tensor=tensor
    )
    p6 = subject.run_recipe(
        recipe_id=recipe_id, graph=graph, semantic_tensor=tensor
    )

    assert raw.output_top5 == p6.output_top5 == (0, 1, 2, 3, 4)
    assert tuple(step.disposition for step in p6.selection_steps) == (
        "unused_raw_fallback",
        "unused_raw_fallback",
    )


def test_deterministic_ties_use_ascending_corpus_ordinal() -> None:
    graph = subject.build_typed_graph(_path_units())
    tensor = _tensor(
        (
            (0, 0, 0, 10, 10, 10, 0, 0, 0, 0),
            (0,) * 10,
        ),
        (
            (0, 0, 0, 100, 100, 100, 0, 0, 0, 0),
            (0,) * 10,
        ),
        dense=(0,) * 10,
    )

    trace = subject.run_recipe(
        recipe_id="R1_P6_DIRECT_B2", graph=graph, semantic_tensor=tensor
    )

    assert trace.output_top5 == (0, 1, 2, 3, 4)
    assert tuple(step.selected_unit_ordinal for step in trace.selection_steps) == (3, 4)


def test_every_tensor_cell_and_all_609_candidates_are_scanned() -> None:
    graph = subject.build_typed_graph(_path_units())
    coverage = (
        (0, 0, 0, 0, 0, 10, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 50, 0, 0, 0),
    )
    anchors = (
        (0, 0, 0, 0, 0, 100, 0, 0, 0, 0),
        (0,) * 10,
    )
    first_tensor = _tensor(coverage, anchors)
    changed_rows = [list(row.semantic_coverage_ints) for row in first_tensor.rows]
    changed_rows[1][-1] = 1
    last_cell_changed = subject.make_query_semantic_tensor(
        query_sha256="b" * 64,
        facets=first_tensor.facets,
        semantic_coverage_ints=changed_rows,
        direct_anchor_strength_ints=[
            row.direct_anchor_strength_ints for row in first_tensor.rows
        ],
        dense_relevance_ints=first_tensor.dense_relevance_ints,
    )

    first = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=first_tensor
    )
    second = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=last_cell_changed
    )

    assert "hippo" not in inspect.signature(subject.run_recipe).parameters
    assert first.candidate_universe_size == subject.CORPUS_UNIT_COUNT == 609
    assert first.candidate_score_evaluations == 2 * subject.CORPUS_UNIT_COUNT
    assert first.semantic_cell_scan_count == len(_facets()) * subject.CORPUS_UNIT_COUNT
    assert first.hipporag_candidate_or_feature_count == 0
    assert first.candidate_scan_sha256 != second.candidate_scan_sha256


def test_tensor_constructor_rejects_any_incomplete_facet_by_609_matrix() -> None:
    facets = _facets()
    with pytest.raises(
        subject.HybridQaOperatorError,
        match="complete 609-unit corpus",
    ):
        subject.make_query_semantic_tensor(
            query_sha256="a" * 64,
            facets=facets,
            semantic_coverage_ints=((0,) * 608, (0,) * 608),
            direct_anchor_strength_ints=((0,) * 608, (0,) * 608),
            dense_relevance_ints=(0,) * 608,
        )


def test_query_counterfactual_changes_p6_but_not_dense_recipe() -> None:
    graph = subject.build_typed_graph(_path_units())
    dense = tuple(900 - index * 100 for index in range(10))
    first = _tensor(
        (
            (0, 0, 0, 0, 0, 10, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 50, 0, 0, 0),
        ),
        (
            (0, 0, 0, 0, 0, 100, 0, 0, 0, 0),
            (0,) * 10,
        ),
        dense=dense,
        query_byte="a",
    )
    second = _tensor(
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 10, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 50),
        ),
        (
            (0, 0, 0, 0, 0, 0, 0, 0, 100, 0),
            (0,) * 10,
        ),
        dense=dense,
        query_byte="b",
    )

    raw_first = subject.run_recipe(
        recipe_id="R0_DENSE5", graph=graph, semantic_tensor=first
    )
    raw_second = subject.run_recipe(
        recipe_id="R0_DENSE5", graph=graph, semantic_tensor=second
    )
    p6_first = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=first
    )
    p6_second = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=second
    )

    assert raw_first.output_top5 == raw_second.output_top5 == (0, 1, 2, 3, 4)
    assert p6_first.output_top5 != p6_second.output_top5
    assert {5, 6}.issubset(p6_first.output_top5)
    assert {8, 9}.issubset(p6_second.output_top5)


def test_graph_tensor_and_action_trace_hashes_fail_on_tamper() -> None:
    graph = subject.build_typed_graph(_path_units())
    tensor = _path_tensor()
    trace = subject.run_recipe(
        recipe_id="R3_P6_PATH2_B2", graph=graph, semantic_tensor=tensor
    )

    with pytest.raises(subject.HybridQaOperatorError, match="graph self hash drifted"):
        subject.verify_typed_graph(replace(graph, edges=graph.edges[:-1]))

    changed = list(tensor.rows[0].semantic_coverage_ints)
    changed[-1] += 1
    tampered_row = replace(tensor.rows[0], semantic_coverage_ints=tuple(changed))
    tampered_tensor = replace(tensor, rows=(tampered_row, tensor.rows[1]))
    with pytest.raises(subject.HybridQaOperatorError, match="tensor self hash drifted"):
        subject.verify_query_semantic_tensor(tampered_tensor)

    tampered_trace = replace(trace, output_top5=(1, 0, 2, 5, 7))
    assert subject.recompute_action_trace_sha256(tampered_trace) != trace.trace_sha256
    with pytest.raises(subject.HybridQaOperatorError, match="trace self hash drifted"):
        subject.verify_action_trace(tampered_trace)


def test_sidecars_are_immutable_canonical_and_type_specific() -> None:
    with pytest.raises(subject.HybridQaOperatorError, match="immutable tuple"):
        subject.AtomicUnit(0, "table_row", "t", 0, ["x"])
    with pytest.raises(subject.HybridQaOperatorError, match="canonical set"):
        _row(0, links=("z", "a"))
    with pytest.raises(subject.HybridQaOperatorError, match="must not declare"):
        subject.AtomicUnit(0, "linked_passage", "t", 0, ("x",))
    with pytest.raises(subject.HybridQaOperatorError, match="exactly one"):
        subject.AtomicUnit(0, "linked_passage", "t", None, ())


def test_row_adjacency_never_bridges_gap_or_crosses_table() -> None:
    graph = subject.build_typed_graph(
        _pad_units(
            (
                _row(0, table="gap", row=0),
                _row(1, table="gap", row=2),
                _row(2, table="left", row=0),
                _row(3, table="right", row=1),
            )
        )
    )
    adjacency = {
        (edge.left_ordinal, edge.right_ordinal)
        for edge in graph.edges
        if edge.family == subject.SAME_TABLE_ADJACENT_ROW
    }
    assert (0, 1) not in adjacency
    assert (2, 3) not in adjacency


def test_deletion_and_exact_same_type_replacement_helpers_support_features() -> None:
    graph = subject.build_typed_graph(
        _pad_units(
            (
                _row(0),
                _passage(1),
                _row(2),
                _passage(3),
                _row(4),
                _passage(5),
            )
        )
    )
    selected = (0, 1, 2, 3, 4)
    tensor = _tensor(
        ((10, 20, 30, 40, 50, 60), (0, 5, 0, 10, 0, 20)),
        ((0,) * 6, (0,) * 6),
    )

    assert subject.deletion_action(selected, slot=1) == (0, 2, 3, 4)
    candidates = subject.same_type_replacement_candidates(graph, selected, slot=1)
    assert candidates == (5,)
    replaced = subject.replace_action_same_type(
        graph,
        selected,
        slot=1,
        replacement_ordinal=5,
    )
    assert replaced == (0, 5, 2, 3, 4)
    assert subject.facet_maxima_ints(tensor, selected) == (50, 10)
    assert subject.facet_maxima_ints(tensor, replaced) == (60, 20)
    with pytest.raises(subject.HybridQaOperatorError, match="exact-same-type"):
        subject.replace_action_same_type(
            graph,
            selected,
            slot=1,
            replacement_ordinal=6,
        )


def test_complete_four_recipe_registry_and_action_matrix_are_stable() -> None:
    assert tuple(recipe.recipe_id for recipe in subject.recipe_registry()) == subject.RECIPE_IDS
    assert tuple(
        recipe.maximum_typed_path_length for recipe in subject.recipe_registry()
    ) == (None, 0, 1, 2)
    graph = subject.build_typed_graph(_path_units())
    tensor = _path_tensor()

    traces = subject.run_all_recipes(graph=graph, semantic_tensor=tensor)

    assert tuple(trace.recipe_id for trace in traces) == subject.RECIPE_IDS
    assert all(len(trace.output_top5) == 5 for trace in traces)
    assert all(len(set(trace.output_top5)) == 5 for trace in traces)
    assert all(subject.verify_action_trace(trace) == trace.trace_sha256 for trace in traces)


def test_run_all_reuses_shared_preparation_and_matches_independent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = subject.build_typed_graph(_path_units())
    tensor = _path_tensor()
    independent = tuple(
        subject.run_recipe(
            recipe_id=recipe_id,
            graph=graph,
            semantic_tensor=tensor,
        )
        for recipe_id in subject.RECIPE_IDS
    )
    calls = {"_validated_graph": 0, "_validated_tensor": 0, "_raw_dense_order": 0}

    for name in calls:
        original = getattr(subject, name)

        def wrapper(*args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(subject, name, wrapper)

    shared = subject.run_all_recipes(graph=graph, semantic_tensor=tensor)

    assert calls == {name: 1 for name in calls}
    assert shared == independent
