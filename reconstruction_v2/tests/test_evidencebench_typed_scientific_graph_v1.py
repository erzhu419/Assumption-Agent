from __future__ import annotations

import ast
from fractions import Fraction
import inspect
import itertools

import pytest

from assumption_agent.benchmarks import evidencebench_typed_scientific_graph_v1 as subject


def _nodes(overrides: dict[int, str] | None = None) -> tuple[subject.SourceNode, ...]:
    texts = [f"ordinary lowercase filler sentence {node_i}." for node_i in range(32)]
    for node_i, text in (overrides or {}).items():
        texts[node_i] = text
    nodes: list[subject.SourceNode] = []
    cursor = 0
    for node_i, text in enumerate(texts):
        nodes.append(subject.SourceNode(node_i, cursor, cursor + len(text), text))
        cursor += len(text) + 1
    return tuple(nodes)


def _zero_matrix() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(32)) for _ in range(32))


def _zero_components() -> subject.CoverageComponents:
    return subject.CoverageComponents(0, 0, 0, 0, (0, 0, 0, 0))


def _edge_tuples(nodes: tuple[subject.SourceNode, ...]) -> set[tuple[str, int, int]]:
    return {edge.as_tuple() for edge in subject.build_typed_scientific_graph(nodes)}


def test_registry_is_frozen_at_nine_recipes_and_sixteen_evaluators() -> None:
    assert len(subject.recipe_registry()) == 9
    assert tuple(recipe.recipe_id for recipe in subject.recipe_registry()) == (
        "R0_HIPPO_TOP5",
        "R1_ADJACENT_1SWAP",
        "R2_ABBREVIATION_1SWAP",
        "R3_XREF_1SWAP",
        "R4_RARE_ENTITY_1SWAP",
        "R5_ADJACENT_ABBREVIATION_2SWAP",
        "R6_ADJACENT_XREF_2SWAP",
        "R7_ABBREVIATION_XREF_2SWAP",
        "R8_ALL_TYPED_2SWAP",
    )
    evaluators = subject.evaluator_registry()
    assert len(evaluators) == 16
    assert len({evaluator.evaluator_id for evaluator in evaluators}) == 16
    assert all(sum(evaluator.edge_weights) == 1 for evaluator in evaluators)


def test_surface_edge_goldens_cover_all_four_families() -> None:
    nodes = _nodes(
        {
            0: "tumor necrosis factor (TNF) mediates response.",
            1: "Figure 2 reports the analysis.",
            2: "BRCA1 was detected.",
            5: "TNF was measured.",
            7: "As shown in Fig. 2, the estimate changed.",
            10: "BRCA1 abundance increased.",
            20: "The BRCA1 result replicated.",
        }
    )
    edges = _edge_tuples(nodes)
    assert {
        (subject.ADJACENT_BUCKET, 0, 1),
        (subject.ABBREVIATION_DEFINITION, 0, 5),
        (subject.EXPLICIT_SCIENTIFIC_XREF, 1, 7),
        (subject.RARE_ENTITY_BRIDGE, 2, 10),
        (subject.RARE_ENTITY_BRIDGE, 2, 20),
        (subject.RARE_ENTITY_BRIDGE, 10, 20),
    }.issubset(edges)
    assert sum(edge[0] == subject.ADJACENT_BUCKET for edge in edges) == 31


def test_surface_extractors_fail_closed_on_frozen_false_positives() -> None:
    nodes = _nodes(
        {
            0: "Measurement occurred in 2024 (2024).",
            1: "Cells were maintained (in vitro).",
            2: "plain words (XYZ) were included.",
            3: "configuration 2 was varied.",
            4: "Figure 2 reports one result.",
            5: "Figure 3 reports another result.",
            6: "COMMONMARKER changed.",
            7: "COMMONMARKER changed again.",
            8: "COMMONMARKER remained.",
            9: "COMMONMARKER was measured.",
        }
    )
    assert {
        edge for edge in _edge_tuples(nodes) if edge[0] != subject.ADJACENT_BUCKET
    } == set()


def test_parallel_families_are_retained_for_one_node_pair() -> None:
    edges = _edge_tuples(
        _nodes(
            {
                0: "tumor protein p53 (TP53) controls the response.",
                1: "TP53 abundance increased.",
            }
        )
    )
    families = {family for family, left, right in edges if (left, right) == (0, 1)}
    assert families == {
        subject.ADJACENT_BUCKET,
        subject.ABBREVIATION_DEFINITION,
        subject.RARE_ENTITY_BRIDGE,
    }


@pytest.mark.parametrize("count", (31, 33))
def test_graph_requires_exactly_32_ordered_source_nodes(count: int) -> None:
    with pytest.raises(subject.EvidenceBenchTypedScientificGraphError, match="exactly 32"):
        subject.build_typed_scientific_graph(_nodes()[:count] if count == 31 else _nodes() + (
            subject.SourceNode(32, 2000, 2001, "x"),
        ))


def test_every_recipe_visits_the_identical_common_scan() -> None:
    nodes = _nodes(
        {
            0: "tumor protein p53 (TP53) controls response.",
            1: "TP53 abundance increased; Figure 4 summarizes it.",
            6: "Fig. 4 presents TP53 in another cohort.",
        }
    )
    graph = subject.build_typed_scientific_graph(nodes)
    similarities = tuple(node_i * 1000 for node_i in range(32))
    official = (0, 1, 2, 3, 4)
    table = subject.build_common_candidate_table(
        nodes, graph, official, similarities
    )
    traces = subject.execute_all_recipes(official, table, similarities)
    assert len(traces) == 9
    assert table
    assert {trace.records_visited for trace in traces} == {len(table)}
    assert len({trace.common_scan_sha256 for trace in traces}) == 1
    assert traces[0].recipe_id == "R0_HIPPO_TOP5"
    assert traces[0].output_top5 == official
    assert traces[0].accepted_count == 0


def test_candidate_ties_and_drop_ties_use_the_frozen_total_orders() -> None:
    official = (0, 1, 2, 3, 4)
    similarities = tuple([10, 10, 10, 0, 0] + [100] * 27)
    # Same similarity and distance: lower neighbor index is the earlier record.
    table = (
        subject.CandidateRecord(0, 1, 0, 5, 100, 50),
        subject.CandidateRecord(0, 2, 1, 6, 100, 50),
    )
    trace = subject.execute_recipe(
        official, table, similarities, "R1_ADJACENT_1SWAP"
    )
    assert trace.decisions[0].record.neighbor_span_i == 5
    assert trace.decisions[0].disposition == "accepted"
    # Spans 3 and 4 share the minimum query score; the worse official rank (4)
    # is dropped, while origin 0 is protected.
    assert trace.decisions[0].dropped_span_i == 4
    assert trace.output_top5 == (0, 1, 2, 3, 5)
    assert trace.decisions[1].disposition == "budget_exhausted"


def test_action_graph_and_components_are_deterministic() -> None:
    nodes = _nodes(
        {
            0: "tumor necrosis factor (TNF) mediates rare response.",
            5: "TNF was measured.",
        }
    )
    similarities = tuple(
        [1_000_000, 500_000, 0, -500_000, -1_000_000] + [0] * 27
    )
    graph_one = subject.build_typed_scientific_graph(nodes)
    graph_two = subject.build_typed_scientific_graph(tuple(nodes))
    assert graph_one == graph_two
    table_one = subject.build_common_candidate_table(
        nodes, graph_one, (0, 1, 2, 3, 4), similarities
    )
    table_two = subject.build_common_candidate_table(
        nodes, graph_two, (0, 1, 2, 3, 4), similarities
    )
    assert table_one == table_two
    assert subject.execute_all_recipes(
        (0, 1, 2, 3, 4), table_one, similarities
    ) == subject.execute_all_recipes((0, 1, 2, 3, 4), table_two, similarities)

    manual_edges = (
        subject.TypedEdge(0, 0, 1),
        subject.TypedEdge(0, 0, 5),
        subject.TypedEdge(2, 2, 3),
    )
    args = (
        "rare missing 123 the",
        nodes,
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 4),
        manual_edges,
        similarities,
        _zero_matrix(),
    )
    components = subject.coverage_components(*args)
    assert components == subject.coverage_components(*args)
    assert components.Sem == 937_500
    assert components.Diversity == 1_000_000
    assert components.Churn == 0
    assert components.Closure == (500_000, 0, 1_000_000, 0)
    assert isinstance(
        subject.score_coverage(components, "E_UNIFORM_L100"), Fraction
    )


def test_a_regret_selection_and_label_free_f_selection() -> None:
    components = {
        recipe.recipe_id: _zero_components()
        for recipe in subject.recipe_registry()
    }
    components["R1_ADJACENT_1SWAP"] = subject.CoverageComponents(
        0, 0, 0, 0, (1_000_000, 0, 0, 0)
    )
    components["R2_ABBREVIATION_1SWAP"] = subject.CoverageComponents(
        0, 0, 0, 0, (0, 1_000_000, 0, 0)
    )
    utility = {recipe.recipe_id: 0 for recipe in subject.recipe_registry()}
    complete = {recipe.recipe_id: False for recipe in subject.recipe_registry()}
    utility["R1_ADJACENT_1SWAP"] = 2_000
    complete["R1_ADJACENT_1SWAP"] = True
    selection = subject.select_a_evaluator(
        [subject.FormationItem(components, utility, complete)]
    )
    assert selection.evaluator_id == "E_ADJACENCY_HEAVY_L025"
    assert selection.chosen_recipe_ids == ("R1_ADJACENT_1SWAP",)
    assert selection.sum_regret == 0
    assert len(selection.evaluator_results) == 16
    assert all(
        result.coverage_comparisons == 9
        for result in selection.evaluator_results
    )

    f_selection = subject.select_f_recipe([components], selection.evaluator_id)
    assert f_selection.recipe_id == "R1_ADJACENT_1SWAP"
    assert isinstance(f_selection.total_exact_coverage, Fraction)
    assert f_selection.coverage_comparisons == 9
    zero_f = subject.select_f_recipe(
        [
            {
                recipe.recipe_id: _zero_components()
                for recipe in subject.recipe_registry()
            }
        ],
        selection.evaluator_id,
    )
    assert zero_f.recipe_id == "R0_HIPPO_TOP5"


def test_aspect_utility_uses_alternative_buckets_and_half_up_recall() -> None:
    assert subject.item_utility(
        (0, 1, 2, 3, 4), ((0, 7), (2, 8), (4, 9))
    ) == (3, 1, 2_000)
    assert subject.item_utility(
        (0, 1, 2, 3, 4), ((0, 7), (8, 9), (10, 11))
    ) == (1, 0, 333)
    assert subject.item_utility(
        (0, 1, 2, 3, 4), ((0,), (6,), (7,), (8,), (9,), (10,))
    ) == (1, 0, 167)
    # There is no gold-cardinality gate: one aspect may have all 32 alternatives.
    assert subject.item_utility((0, 1, 2, 3, 4), (tuple(range(32)),)) == (
        1,
        1,
        2_000,
    )


@pytest.mark.parametrize(
    "aspects",
    (
        (),
        ((),),
        ((2, 0),),
        ((0, 0),),
        ((32,),),
    ),
)
def test_aspect_utility_rejects_malformed_evidence_sets(
    aspects: tuple[tuple[int, ...], ...]
) -> None:
    with pytest.raises(subject.EvidenceBenchTypedScientificGraphError):
        subject.item_utility((0, 1, 2, 3, 4), aspects)
    with pytest.raises(subject.EvidenceBenchTypedScientificGraphError, match="32-node"):
        subject.item_utility(
            (0, 1, 2, 3, 4), ((0,),), source_count=31
        )


def _brute_sign_flip(deltas: list[int]) -> Fraction:
    magnitudes = [abs(value) for value in deltas if value]
    observed = sum(deltas)
    possible = [
        sum(sign * magnitude for sign, magnitude in zip(signs, magnitudes))
        for signs in itertools.product((-1, 1), repeat=len(magnitudes))
    ]
    return Fraction(sum(value >= observed for value in possible), len(possible))


@pytest.mark.parametrize(
    "deltas",
    ([0, 0], [1, 1, 1, 1], [3, -1, 2, -3, 0, 1], [-3, -1, 0, 1]),
)
def test_exact_signflip_matches_brute_magnitude_preserving_test(
    deltas: list[int],
) -> None:
    result = subject.exact_magnitude_preserving_sign_flip(deltas)
    expected = _brute_sign_flip(deltas)
    assert Fraction(
        result["p_value_numerator"], result["p_value_denominator"]
    ) == expected
    assert result["observed_net_U"] == sum(deltas)
    assert result["promoted"] is (
        sum(deltas) > 0 and expected <= Fraction(1, 10)
    )


def test_label_free_interfaces_exclude_metadata_and_gold() -> None:
    forbidden = {
        "gold",
        "metadata",
        "paper_id",
        "document_id",
        "file_name",
        "url",
        "aspect",
        "answer",
    }
    label_free_functions = (
        subject.build_typed_scientific_graph,
        subject.build_common_candidate_table,
        subject.execute_recipe,
        subject.execute_all_recipes,
        subject.coverage_components,
        subject.select_f_recipe,
    )
    for function in label_free_functions:
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    assert tuple(inspect.signature(subject.build_typed_scientific_graph).parameters) == (
        "nodes",
    )
    assert tuple(subject.SourceNode.__dataclass_fields__) == (
        "span_i",
        "start",
        "end",
        "identity_text",
    )
    with pytest.raises(TypeError):
        subject.build_typed_scientific_graph(_nodes(), query="forbidden")
    with pytest.raises(TypeError):
        subject.select_f_recipe([], "E_UNIFORM_L100", gold=())


def test_core_has_no_dataset_network_model_loader_or_legal_core_import() -> None:
    source = inspect.getsource(subject)
    tree = ast.parse(source)
    allowed_roots = {
        "__future__",
        "collections",
        "dataclasses",
        "fractions",
        "hashlib",
        "json",
        "re",
        "unicodedata",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= allowed_roots
    assert "contractnli_typed_clause_graph" not in source
    assert not ({"requests", "urllib", "datasets", "torch", "transformers"} & imported_roots)
