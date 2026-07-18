from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from assumption_agent.benchmarks import feverous_p6_query_anchored_operator_v1 as subject


def _unit(
    ordinal: int,
    *,
    page: str | None = None,
    order: int | None = None,
    section: tuple[str, ...] = (),
    unit_type: str = "sentence",
    table: str | None = None,
    row: int | None = None,
    headers: tuple[int, ...] = (),
    parent: tuple[str, ...] = (),
    entities: tuple[subject.EntityKey, ...] = (),
) -> subject.AtomicUnit:
    return subject.AtomicUnit(
        corpus_ordinal=ordinal,
        unit_type=unit_type,
        page_key=page or f"page-{ordinal}",
        official_order=ordinal if order is None else order,
        section_path=section,
        table_key=table,
        table_row=row,
        applicable_header_ordinals=headers,
        list_parent_path=parent,
        entities=entities,
    )


def _base_units() -> tuple[subject.AtomicUnit, ...]:
    # 5 -> 6 is a query-anchored path.  7 <-> 8 is a deliberately disconnected
    # high-semantic-score clique.
    clique_entity = subject.make_entity_key("ORG", "Disconnected Clique")
    rows = (
        _unit(0),
        _unit(1),
        _unit(2),
        _unit(3),
        _unit(4),
        _unit(5, page="anchored", order=0),
        _unit(6, page="anchored", order=1),
        _unit(7, page="clique", order=0, entities=(clique_entity,)),
        _unit(8, page="clique", order=1, entities=(clique_entity,)),
    )
    return _pad_units(rows)


def _pad_units(
    rows: tuple[subject.AtomicUnit, ...],
) -> tuple[subject.AtomicUnit, ...]:
    if tuple(unit.corpus_ordinal for unit in rows) != tuple(range(len(rows))):
        raise AssertionError("synthetic units must begin in complete ordinal order")
    return rows + tuple(
        _unit(ordinal)
        for ordinal in range(len(rows), subject.CORPUS_UNIT_COUNT)
    )


def _facets() -> tuple[subject.ClaimFacet, ...]:
    return (
        subject.make_claim_facet(0, "entity", "Anchor Entity"),
        subject.make_claim_facet(1, "relation_clause", "won the relation"),
    )


def _tensor(
    coverage: tuple[tuple[int, ...], ...],
    anchors: tuple[tuple[int, ...], ...],
    *,
    dense: tuple[int, ...] | None = None,
    query_byte: str = "a",
) -> subject.QuerySemanticTensor:
    if any(len(row) != len(coverage[0]) for row in (*coverage, *anchors)):
        raise AssertionError("synthetic semantic rows have inconsistent widths")
    initial_size = len(coverage[0])
    coverage = tuple(
        (*row, *((0,) * (subject.CORPUS_UNIT_COUNT - initial_size)))
        for row in coverage
    )
    anchors = tuple(
        (*row, *((0,) * (subject.CORPUS_UNIT_COUNT - initial_size)))
        for row in anchors
    )
    if dense is None:
        dense = tuple(
            1000 - ordinal for ordinal in range(subject.CORPUS_UNIT_COUNT)
        )
    else:
        dense = dense + tuple(
            -10_000_000 - ordinal
            for ordinal in range(initial_size, subject.CORPUS_UNIT_COUNT)
        )
    return subject.make_query_semantic_tensor(
        query_sha256=query_byte * 64,
        facets=_facets(),
        semantic_coverage_ints=coverage,
        direct_anchor_strength_ints=anchors,
        dense_relevance_ints=dense,
    )


def test_disconnected_high_score_clique_is_ignored_by_query_anchored_path() -> None:
    graph = subject.build_typed_graph(_base_units())
    tensor = _tensor(
        (
            (0, 0, 0, 0, 0, 10, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 50, 1000, 900),
        ),
        (
            (0, 0, 0, 0, 0, 100, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
    )

    trace = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=tensor
    )

    assert 6 in trace.output_top5
    assert 5 in trace.output_top5
    assert 7 not in trace.output_top5 and 8 not in trace.output_top5
    assert trace.selection_steps[0].selected_unit_ordinal == 6
    assert trace.selection_steps[0].path_length == 1
    assert trace.selection_steps[1].selected_unit_ordinal == 5
    assert trace.selection_steps[1].direct_anchor is True


def test_query_counterfactual_changes_p6_but_not_r0_dense_order() -> None:
    graph = subject.build_typed_graph(_base_units())
    dense = (900, 800, 700, 600, 500, 400, 300, 200, 100)
    first = _tensor(
        (
            (0, 0, 0, 0, 0, 10, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 50, 0, 0),
        ),
        (
            (0, 0, 0, 0, 0, 100, 0, 0, 0),
            (0,) * 9,
        ),
        dense=dense,
        query_byte="a",
    )
    second = _tensor(
        (
            (0, 0, 0, 0, 0, 0, 0, 10, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 50),
        ),
        (
            (0, 0, 0, 0, 0, 0, 0, 100, 0),
            (0,) * 9,
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
    assert {7, 8}.issubset(p6_second.output_top5)


@pytest.mark.parametrize(
    "recipe_id",
    ("R1_P6_DIRECT_B2", "R2_P6_PATH1_B2", "R3_P6_PATH2_B2"),
)
def test_no_positive_residual_candidate_falls_back_exactly_to_raw(
    recipe_id: str,
) -> None:
    graph = subject.build_typed_graph(_base_units())
    tensor = _tensor(
        ((0,) * 9, (0,) * 9),
        ((0,) * 9, (0,) * 9),
    )

    raw = subject.run_recipe(
        recipe_id="R0_DENSE5", graph=graph, semantic_tensor=tensor
    )
    p6 = subject.run_recipe(
        recipe_id=recipe_id, graph=graph, semantic_tensor=tensor
    )

    assert p6.output_top5 == raw.output_top5 == (0, 1, 2, 3, 4)
    assert tuple(step.disposition for step in p6.selection_steps) == (
        "unused_raw_fallback",
        "unused_raw_fallback",
    )


def test_cell_to_header_requires_exact_header_cell_type() -> None:
    units = _pad_units((
        _unit(0),
        _unit(1),
        _unit(2),
        _unit(3, page="table-page", order=0, unit_type="cell", table="t", row=1, headers=(4,)),
        _unit(4, page="table-page", order=1, unit_type="header_cell", table="t", row=0),
        _unit(5, page="table-page", order=2, unit_type="cell", table="t", row=2),
    ))
    graph = subject.build_typed_graph(units)
    header_edges = {
        (edge.left_ordinal, edge.right_ordinal)
        for edge in graph.edges
        if edge.family == subject.CELL_TO_APPLICABLE_HEADER
    }
    assert header_edges == {(3, 4)}

    wrong_exact_type = (*units[:4], replace(units[4], unit_type="cell", table_row=0), *units[5:])
    with pytest.raises(subject.FeverousP6OperatorError, match="exact same-table header_cell"):
        subject.build_typed_graph(wrong_exact_type)


def test_every_corpus_unit_and_semantic_cell_is_considered_without_hippo_input() -> None:
    graph = subject.build_typed_graph(_base_units())
    base_coverage = (
        (0, 0, 0, 0, 0, 10, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 50, 0, 0),
    )
    anchors = (
        (0, 0, 0, 0, 0, 100, 0, 0, 0),
        (0,) * 9,
    )
    original = _tensor(base_coverage, anchors)
    changed_only_at_last_unreachable_unit = _tensor(
        (base_coverage[0], (*base_coverage[1][:-1], 1)), anchors, query_byte="b"
    )

    first = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2", graph=graph, semantic_tensor=original
    )
    second = subject.run_recipe(
        recipe_id="R2_P6_PATH1_B2",
        graph=graph,
        semantic_tensor=changed_only_at_last_unreachable_unit,
    )

    assert "hippo" not in inspect.signature(subject.run_recipe).parameters
    assert first.candidate_universe_size == len(graph.units) == subject.CORPUS_UNIT_COUNT
    assert first.candidate_score_evaluations == 2 * len(graph.units)
    assert first.semantic_cell_scan_count == len(_facets()) * len(graph.units)
    assert first.hipporag_candidate_or_feature_count == 0
    # Even a non-selected last-corpus-unit semantic change alters the complete
    # scan receipt, proving it was not removed by a hidden shortlist.
    assert first.candidate_scan_sha256 != second.candidate_scan_sha256


def test_same_page_adjacency_never_bridges_an_official_order_gap() -> None:
    graph = subject.build_typed_graph(
        _pad_units(
            (
                _unit(0, page="gap", order=0),
                _unit(1, page="gap", order=2),
                _unit(2, page="contiguous", order=4),
                _unit(3, page="contiguous", order=5),
                _unit(4),
            )
        )
    )
    adjacency = {
        (edge.left_ordinal, edge.right_ordinal)
        for edge in graph.edges
        if edge.family == subject.SAME_PAGE_ADJACENT_OFFICIAL_ORDER
    }
    assert (0, 1) not in adjacency
    assert (2, 3) in adjacency


def test_same_page_adjacency_never_crosses_a_section_boundary() -> None:
    graph = subject.build_typed_graph(
        _pad_units(
            (
                _unit(0, page="sectioned", order=0, section=("section_0",)),
                _unit(1, page="sectioned", order=1, section=("section_1",)),
                _unit(2),
                _unit(3),
                _unit(4),
            )
        )
    )
    adjacency = {
        (edge.left_ordinal, edge.right_ordinal)
        for edge in graph.edges
        if edge.family == subject.SAME_PAGE_ADJACENT_OFFICIAL_ORDER
    }
    assert (0, 1) not in adjacency


def test_action_trace_self_hash_detects_tampering() -> None:
    graph = subject.build_typed_graph(_base_units())
    tensor = _tensor(((0,) * 9, (0,) * 9), ((0,) * 9, (0,) * 9))
    trace = subject.run_recipe(
        recipe_id="R0_DENSE5", graph=graph, semantic_tensor=tensor
    )
    assert subject.verify_action_trace(trace) == trace.trace_sha256

    tampered = replace(trace, output_top5=(1, 0, 2, 3, 4))
    assert subject.recompute_action_trace_sha256(tampered) != trace.trace_sha256
    with pytest.raises(subject.FeverousP6OperatorError, match="self hash drifted"):
        subject.verify_action_trace(tampered)


def test_registry_and_complete_four_recipe_matrix_are_exact_and_stable() -> None:
    assert tuple(recipe.recipe_id for recipe in subject.recipe_registry()) == subject.RECIPE_IDS
    assert tuple(recipe.maximum_typed_path_length for recipe in subject.recipe_registry()) == (
        None,
        0,
        1,
        2,
    )
    graph = subject.build_typed_graph(_base_units())
    tensor = _tensor(
        (
            (0, 0, 0, 0, 0, 10, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 50, 0, 0),
        ),
        (
            (0, 0, 0, 0, 0, 100, 0, 0, 0),
            (0,) * 9,
        ),
    )
    matrix = subject.run_all_recipes(graph=graph, semantic_tensor=tensor)

    assert tuple(trace.recipe_id for trace in matrix) == subject.RECIPE_IDS
    assert all(len(trace.output_top5) == 5 for trace in matrix)
    assert all(len(set(trace.output_top5)) == 5 for trace in matrix)
    assert all(subject.verify_action_trace(trace) == trace.trace_sha256 for trace in matrix)


def test_run_all_reuses_shared_preparation_once_and_matches_independent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = subject.build_typed_graph(_base_units())
    tensor = _tensor(
        (
            (0, 0, 0, 0, 0, 10, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 50, 0, 0),
        ),
        (
            (0, 0, 0, 0, 0, 100, 0, 0, 0),
            (0,) * 9,
        ),
    )
    independent = tuple(
        subject.run_recipe(
            recipe_id=recipe_id,
            graph=graph,
            semantic_tensor=tensor,
        )
        for recipe_id in subject.RECIPE_IDS
    )

    counters = {
        "_validated_graph": 0,
        "_validated_tensor": 0,
        "_raw_dense_order": 0,
        "_query_anchored_reachability": 0,
    }

    def counted(name: str):
        original = getattr(subject, name)

        def wrapper(*args: object, **kwargs: object):
            counters[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in counters:
        monkeypatch.setattr(subject, name, counted(name))

    shared = subject.run_all_recipes(graph=graph, semantic_tensor=tensor)

    assert counters == {name: 1 for name in counters}
    assert shared == independent
    assert tuple(trace.trace_sha256 for trace in shared) == tuple(
        trace.trace_sha256 for trace in independent
    )


def test_stable_ties_use_ascending_corpus_ordinal() -> None:
    graph = subject.build_typed_graph(_base_units())
    tensor = _tensor(
        (
            (0, 0, 0, 10, 10, 10, 0, 0, 0),
            (0,) * 9,
        ),
        (
            (0, 0, 0, 100, 100, 100, 0, 0, 0),
            (0,) * 9,
        ),
        dense=(0,) * 9,
    )
    trace = subject.run_recipe(
        recipe_id="R1_P6_DIRECT_B2", graph=graph, semantic_tensor=tensor
    )

    # 0/1/2 are retained; 3, 4 and 5 have equal P6 keys, so 3 then 4 win.
    assert trace.output_top5 == (0, 1, 2, 3, 4)
    assert tuple(step.selected_unit_ordinal for step in trace.selection_steps) == (3, 4)
