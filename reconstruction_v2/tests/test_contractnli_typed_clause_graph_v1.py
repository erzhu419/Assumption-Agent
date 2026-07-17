from __future__ import annotations

from fractions import Fraction
import inspect
import itertools

import pytest

import assumption_agent.benchmarks.contractnli_typed_clause_graph_v1 as subject


def _spans(texts: list[str]) -> tuple[subject.SourceSpan, ...]:
    return tuple(
        subject.SourceSpan(i, i * 100, i * 100 + len(text), text)
        for i, text in enumerate(texts)
    )


def _zero_matrix(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def _zero_components() -> subject.CoverageComponents:
    return subject.CoverageComponents(0, 0, 0, 0, (0, 0, 0, 0))


def test_registries_are_exact_and_all_scores_are_fractions() -> None:
    assert [(recipe.recipe_id, recipe.budget) for recipe in subject.recipe_registry()] == [
        ("R0_HIPPO_TOP5", 0),
        ("R1_DEFINITION_1SWAP", 1),
        ("R2_EXCEPTION_1SWAP", 1),
        ("R3_LIST_1SWAP", 1),
        ("R4_CROSS_REFERENCE_1SWAP", 1),
        ("R5_DEFINITION_EXCEPTION_2SWAP", 2),
        ("R6_DEFINITION_LIST_2SWAP", 2),
        ("R7_EXCEPTION_LIST_2SWAP", 2),
        ("R8_ALL_TYPED_2SWAP", 2),
    ]
    evaluator_ids = [evaluator.evaluator_id for evaluator in subject.evaluator_registry()]
    assert len(evaluator_ids) == len(set(evaluator_ids)) == 16
    assert evaluator_ids[:4] == [
        "E_DEF_HEAVY_L025",
        "E_DEF_HEAVY_L050",
        "E_DEF_HEAVY_L100",
        "E_DEF_HEAVY_L200",
    ]
    scores = subject.score_all_evaluators(_zero_components())
    assert len(scores) == 16
    assert all(isinstance(score, Fraction) for _, score in scores)


def test_unicode_casefold_regex_graph_builds_all_four_edge_families() -> None:
    spans = _spans(
        [
            "  “Résumé”   Means a short summary. ",
            "RE\u0301SUME\u0301 shall be supplied.",
            "Except that a delay may occur.",
            "Clause 7. Payment terms",
            "See CLAUSE 7 for payment.",
            "(a) first item",
            "(iv) second item",
            "unmatched ordinary prose",
        ]
    )
    assert spans[0].identity_text.startswith("  “")
    assert spans[0].embedding_text == "“Résumé” Means a short summary."
    assert spans[1].pattern_text.startswith("résumé shall")
    edges = {edge.as_tuple() for edge in subject.build_typed_clause_graph(spans)}
    assert (subject.MENTIONS_DEFINITION, 0, 1) in edges
    assert (subject.EXCEPTION_SCOPE, 1, 2) in edges
    assert (subject.EXPLICIT_CROSS_REFERENCE, 3, 4) in edges
    assert (subject.LIST_SIBLING, 5, 6) in edges
    assert all(left != right for _, left, right in edges)


def test_common_table_preserves_multiseed_and_parallel_typed_records() -> None:
    spans = _spans([f"span {i}" for i in range(7)])
    edges = (
        subject.TypedEdge(0, 0, 5),
        subject.TypedEdge(2, 0, 5),
        subject.TypedEdge(0, 1, 5),
        subject.TypedEdge(3, 0, 1),
    )
    table = subject.build_common_candidate_table(
        spans, edges, (0, 1, 2, 3, 4), (10, 20, 30, 40, 50, 99, 0)
    )
    triples = {
        (record.edge_family, record.seed_span_i, record.neighbor_span_i)
        for record in table
    }
    assert {
        (subject.MENTIONS_DEFINITION, 0, 5),
        (subject.LIST_SIBLING, 0, 5),
        (subject.MENTIONS_DEFINITION, 1, 5),
        (subject.EXPLICIT_CROSS_REFERENCE, 0, 1),
        (subject.EXPLICIT_CROSS_REFERENCE, 1, 0),
    } <= triples
    assert len(table) == 5
    assert [record.as_tuple() for record in table] == [
        record.as_tuple()
        for record in sorted(
            table,
            key=lambda record: (
                -record.query_similarity_int,
                record.absolute_start_offset_distance,
                record.neighbor_span_i,
                record.official_seed_rank,
                record.edge_family_order,
            ),
        )
    ]


def test_origin_dropped_record_is_skipped_and_r0_performs_equal_full_scan() -> None:
    spans = _spans([f"span {i}" for i in range(8)])
    similarities = (500, 100, 200, 300, 400, 1000, 900, 800)
    table = subject.build_common_candidate_table(
        spans,
        (
            subject.TypedEdge(0, 0, 5),
            subject.TypedEdge(1, 1, 6),
            subject.TypedEdge(2, 2, 7),
        ),
        (0, 1, 2, 3, 4),
        similarities,
    )
    all_typed = subject.execute_recipe(
        (0, 1, 2, 3, 4), table, similarities, "R8_ALL_TYPED_2SWAP"
    )
    assert all_typed.output_top5 == (0, 2, 4, 5, 7)
    assert all_typed.swap_count == all_typed.accepted_count == 2
    assert [decision.disposition for decision in all_typed.decisions] == [
        "accepted",
        "origin_seed_not_selected",
        "accepted",
    ]

    identity = subject.execute_recipe(
        (0, 1, 2, 3, 4), table, similarities, "R0_HIPPO_TOP5"
    )
    assert identity.output_top5 == (0, 1, 2, 3, 4)
    assert identity.records_visited == all_typed.records_visited == len(table)
    assert identity.common_scan_sha256 == all_typed.common_scan_sha256
    assert identity.accepted_count == 0


def test_sem_parentheses_clipping_and_integer_component_formulas() -> None:
    spans = _spans(["rare token", "beta", "gamma", "delta", "epsilon", "zeta"])
    query_sims = (1_000_000, 0, -1_000_000, -1_000_000, -1_000_000, 0)
    assert subject.semantic_coverage((0, 1, 2, 3, 4), query_sims) == 875_000
    assert subject.semantic_coverage(
        (0, 1, 2, 3, 4), (2_000_000, 0, -2_000_000, -2_000_000, -2_000_000, 0)
    ) == 875_000

    edges = (
        subject.TypedEdge(0, 0, 1),
        subject.TypedEdge(0, 0, 5),
        subject.TypedEdge(2, 2, 3),
    )
    components = subject.coverage_components(
        "rare missing 123 the",
        spans,
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 4),
        edges,
        query_sims,
        _zero_matrix(6),
    )
    assert components.Sem == 875_000
    assert components.Lex == 1_000_000 * 6 // 13
    assert components.Diversity == 1_000_000
    assert components.Churn == 0
    assert components.Closure == (500_000, 0, 1_000_000, 0)
    assert isinstance(subject.score_coverage(components, "E_UNIFORM_L100"), Fraction)


def test_graph_action_and_coverage_are_deterministic() -> None:
    spans = _spans(
        [
            '"service" means support',
            "service shall continue",
            "unless service ends",
            "(a) x",
            "(b) y",
            "other service",
        ]
    )
    similarities = (10, 20, 30, 40, 50, 60)
    graph1 = subject.build_typed_clause_graph(spans)
    graph2 = subject.build_typed_clause_graph(tuple(spans))
    assert graph1 == graph2
    table1 = subject.build_common_candidate_table(
        spans, graph1, (0, 1, 2, 3, 4), similarities
    )
    table2 = subject.build_common_candidate_table(
        spans, graph2, (0, 1, 2, 3, 4), similarities
    )
    assert table1 == table2
    assert subject.execute_all_recipes(
        (0, 1, 2, 3, 4), table1, similarities
    ) == subject.execute_all_recipes((0, 1, 2, 3, 4), table2, similarities)
    args = (
        "service support",
        spans,
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 4),
        graph1,
        similarities,
        _zero_matrix(6),
    )
    assert subject.coverage_components(*args) == subject.coverage_components(*args)


def test_a_regret_selection_and_label_free_f_recipe_selection() -> None:
    components = {recipe.recipe_id: _zero_components() for recipe in subject.recipe_registry()}
    components["R1_DEFINITION_1SWAP"] = subject.CoverageComponents(
        0, 0, 0, 0, (1_000_000, 0, 0, 0)
    )
    components["R2_EXCEPTION_1SWAP"] = subject.CoverageComponents(
        0, 0, 0, 0, (0, 1_000_000, 0, 0)
    )
    utility = {recipe.recipe_id: 0 for recipe in subject.recipe_registry()}
    complete = {recipe.recipe_id: False for recipe in subject.recipe_registry()}
    utility["R1_DEFINITION_1SWAP"] = 3
    complete["R1_DEFINITION_1SWAP"] = True
    selection = subject.select_a_evaluator(
        [subject.FormationItem(components, utility, complete)]
    )
    assert selection.evaluator_id == "E_DEF_HEAVY_L025"
    assert selection.chosen_recipe_ids == ("R1_DEFINITION_1SWAP",)
    assert selection.sum_regret == 0
    assert len(selection.evaluator_results) == 16
    assert all(result.coverage_comparisons == 9 for result in selection.evaluator_results)

    f_selection = subject.select_f_recipe([components], selection.evaluator_id)
    assert f_selection.recipe_id == "R1_DEFINITION_1SWAP"
    assert isinstance(f_selection.total_exact_coverage, Fraction)
    assert f_selection.coverage_comparisons == 9
    zero_f = subject.select_f_recipe(
        [{recipe.recipe_id: _zero_components() for recipe in subject.recipe_registry()}],
        selection.evaluator_id,
    )
    assert zero_f.recipe_id == "R0_HIPPO_TOP5"  # minimum declared budget


def test_action_and_f_interfaces_have_no_forbidden_metadata_or_gold_inputs() -> None:
    forbidden = {
        "gold",
        "metadata",
        "document_id",
        "file_name",
        "url",
        "choice",
        "hypothesis_id",
        "short_description",
    }
    functions = (
        subject.build_typed_clause_graph,
        subject.build_common_candidate_table,
        subject.execute_recipe,
        subject.execute_all_recipes,
        subject.select_f_recipe,
    )
    for function in functions:
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    assert tuple(subject.SourceSpan.__dataclass_fields__) == (
        "span_i",
        "start",
        "end",
        "identity_text",
    )
    with pytest.raises(TypeError):
        subject.execute_recipe((), (), (), "R0_HIPPO_TOP5", gold=())


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
def test_late_utility_and_exact_signflip_match_frozen_rules(deltas: list[int]) -> None:
    result = subject.exact_magnitude_preserving_sign_flip(deltas)
    expected = _brute_sign_flip(deltas)
    assert Fraction(result["p_value_numerator"], result["p_value_denominator"]) == expected
    assert result["promoted"] is (sum(deltas) > 0 and expected <= Fraction(1, 10))


def test_gold_utility_is_sorted_distinct_bounded_and_has_no_complete_gate() -> None:
    assert subject.item_utility((0, 1, 2, 3, 4), (0, 2), source_count=6) == (2, 1, 3)
    assert subject.item_utility((0, 1, 2, 3, 4), (0, 5), source_count=6) == (1, 0, 1)
    with pytest.raises(subject.ContractNLITypedClauseGraphError, match="sorted"):
        subject.item_utility((0, 1, 2, 3, 4), (2, 0), source_count=6)


def test_empty_source_span_and_out_of_range_pair_cosine_fail_closed() -> None:
    with pytest.raises(subject.ContractNLITypedClauseGraphError, match="offsets"):
        subject.build_typed_clause_graph((subject.SourceSpan(0, 1, 1, ""),))

    spans = _spans([f"span {index}" for index in range(5)])
    matrix = [list(row) for row in _zero_matrix(5)]
    matrix[0][1] = matrix[1][0] = 1_000_001
    with pytest.raises(subject.ContractNLITypedClauseGraphError, match="integer N by N"):
        subject.coverage_components(
            "query",
            spans,
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 4),
            (),
            (0, 0, 0, 0, 0),
            matrix,
        )
