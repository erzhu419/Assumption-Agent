from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect

import pytest

from assumption_agent.benchmarks import (
    feverous_e2_feature_producer_v1 as producer,
)
from assumption_agent.benchmarks import (
    feverous_p6_query_anchored_operator_v1 as operator,
)
from assumption_agent.benchmarks.feverous_e2_evaluator_v1 import RecipeTrace


OPERATOR_RECEIPT = hashlib.sha256(b"synthetic-operator-receipt").hexdigest()
SEMANTIC_RECEIPT = hashlib.sha256(b"synthetic-semantic-receipt").hexdigest()
ITEM = hashlib.sha256(b"synthetic-item").hexdigest()


def _unit(
    ordinal: int,
    *,
    unit_type: str = "sentence",
    page: str | None = None,
) -> operator.AtomicUnit:
    table = f"table-{ordinal}" if unit_type in {"cell", "header_cell"} else None
    row = 0 if table is not None else None
    return operator.AtomicUnit(
        corpus_ordinal=ordinal,
        unit_type=unit_type,
        page_key=page or f"page-{ordinal}",
        official_order=0,
        table_key=table,
        table_row=row,
    )


def _fixture() -> tuple[
    operator.TypedCorpusGraph,
    operator.QuerySemanticTensor,
    tuple[operator.ActionTrace, ...],
]:
    # The selected RAW top five have types sentence/cell/item/sentence/cell.
    # Unit 5 is a stronger sentence replacement, unit 6 a stronger cell
    # replacement, and no unselected item exists.
    leading = (
        _unit(0),
        _unit(1, unit_type="cell"),
        _unit(2, unit_type="item"),
        _unit(3),
        _unit(4, unit_type="cell"),
        _unit(5),
        _unit(6, unit_type="cell"),
    )
    units = leading + tuple(
        _unit(ordinal)
        for ordinal in range(len(leading), operator.CORPUS_UNIT_COUNT)
    )
    graph = operator.build_typed_graph(units)

    facets = (
        operator.make_claim_facet(0, "entity", "anchor entity"),
        operator.make_claim_facet(1, "relation_clause", "frozen relation"),
    )
    f0 = [0] * operator.CORPUS_UNIT_COUNT
    f1 = [0] * operator.CORPUS_UNIT_COUNT
    f0[0], f0[3], f0[5] = 10, 20, 30
    f1[1], f1[2], f1[4], f1[6] = 10, 5, 20, 30
    a0 = [0] * operator.CORPUS_UNIT_COUNT
    a1 = [0] * operator.CORPUS_UNIT_COUNT
    a0[0] = 100
    a1[1] = 100
    dense = [
        -10_000_000 - ordinal
        for ordinal in range(operator.CORPUS_UNIT_COUNT)
    ]
    dense[:7] = [100, 90, 80, 70, 60, 50, 40]
    tensor = operator.make_query_semantic_tensor(
        query_sha256="a" * 64,
        facets=facets,
        semantic_coverage_ints=(f0, f1),
        direct_anchor_strength_ints=(a0, a1),
        dense_relevance_ints=dense,
    )
    traces = operator.run_all_recipes(
        graph=graph, semantic_tensor=tensor
    )
    assert all(trace.output_top5 == (0, 1, 2, 3, 4) for trace in traces)
    return graph, tensor, traces


@pytest.fixture(scope="module")
def frozen_inputs():
    return _fixture()


def _produce_matrix(frozen_inputs):
    graph, tensor, traces = frozen_inputs
    return producer.produce_complete_e2_recipe_matrix(
        item_commitment_sha256=ITEM,
        graph=graph,
        semantic_tensor=tensor,
        action_traces=traces,
        external_operator_receipt_sha256=OPERATOR_RECEIPT,
        external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
    )


def test_exact_eight_features_and_intervention_semantics(frozen_inputs) -> None:
    produced = _produce_matrix(frozen_inputs)
    first = produced[0]
    assert isinstance(first.recipe_trace, RecipeTrace)
    assert tuple(first.exact_features.as_mapping()) == producer.FEATURE_ORDER
    assert first.exact_features.as_mapping() == {
        "direct_facet_coverage": Fraction(20, producer.INTEGER_SCALE),
        "residual_facet_coverage": Fraction(10, producer.INTEGER_SCALE),
        "deletion_mean_coverage_drop": Fraction(2, producer.INTEGER_SCALE),
        "deletion_minimum_coverage_drop": Fraction(0),
        "same_type_replacement_mean_coverage_drop": Fraction(
            -4, producer.INTEGER_SCALE
        ),
        "query_anchored_path_coverage": Fraction(2, 5),
        "dense_relevance_mass": Fraction(400, producer.INTEGER_SCALE),
        "negative_pairwise_redundancy": Fraction(
            -3, 2 * producer.INTEGER_SCALE
        ),
    }
    diagnostics = first.diagnostics
    assert diagnostics.full_facet_maxima_ints == (20, 20)
    assert diagnostics.retained_raw_top3_facet_maxima_ints == (10, 10)
    assert diagnostics.deletion_coverage_drops == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(5, producer.INTEGER_SCALE),
        Fraction(5, producer.INTEGER_SCALE),
    )
    assert diagnostics.replacement_coverage_drops == (
        Fraction(-5, producer.INTEGER_SCALE),
        Fraction(-5, producer.INTEGER_SCALE),
        Fraction(0),
        Fraction(-5, producer.INTEGER_SCALE),
        Fraction(-5, producer.INTEGER_SCALE),
    )
    assert diagnostics.replacement_available == (True, True, False, True, True)
    assert diagnostics.replacement_same_type_candidate_counts[2] == 0
    assert diagnostics.query_anchored_reachable_selected_count == 2
    assert diagnostics.positive_pairwise_redundancy_sum_int == 30


def test_complete_8192_replacement_scan_and_four_recipe_behavior_reuse(
    frozen_inputs,
) -> None:
    produced = _produce_matrix(frozen_inputs)
    assert tuple(row.recipe_trace.recipe_id for row in produced) == producer.RECIPE_IDS
    assert len(produced) == 4
    assert {
        row.recipe_trace.behavior_sha256 for row in produced
    } == {produced[0].recipe_trace.behavior_sha256}
    assert all(
        row.diagnostics.replacement_candidate_consideration_count
        == 5 * producer.CORPUS_UNIT_COUNT
        for row in produced
    )
    # Shared ordered behavior has shared exact features and scan receipts, but
    # action-specific operator provenance remains distinct.
    assert all(row.exact_features == produced[0].exact_features for row in produced)
    assert all(row.diagnostics == produced[0].diagnostics for row in produced)
    assert len(
        {
            row.provenance.operator_action_trace_sha256
            for row in produced
        }
    ) == 4


def test_external_receipts_and_internal_operator_semantics_are_bound(
    frozen_inputs,
) -> None:
    graph, tensor, traces = frozen_inputs
    produced = producer.produce_e2_recipe_trace(
        item_commitment_sha256=ITEM,
        graph=graph,
        semantic_tensor=tensor,
        action_trace=traces[0],
        external_operator_receipt_sha256=OPERATOR_RECEIPT,
        external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
    )
    assert produced.provenance.operator_graph_sha256 == graph.graph_sha256
    assert produced.provenance.semantic_tensor_sha256 == tensor.tensor_sha256
    assert produced.provenance.operator_action_trace_sha256 == traces[0].trace_sha256
    assert produced.provenance.external_operator_receipt_sha256 == OPERATOR_RECEIPT
    assert produced.provenance.external_semantic_receipt_sha256 == SEMANTIC_RECEIPT
    assert produced.provenance.feature_behavior_sha256 == producer.FEATURE_BEHAVIOR_SHA256
    assert producer.verify_feature_production_trace(
        produced,
        external_operator_receipt_sha256=OPERATOR_RECEIPT,
        external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
    ) == produced.production_trace_sha256
    with pytest.raises(
        producer.FeverousFeatureProducerError, match="external provenance freeze"
    ):
        producer.verify_feature_production_trace(
            produced,
            external_operator_receipt_sha256="b" * 64,
            external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
        )


def test_permuted_or_rehashed_operator_action_is_rejected(frozen_inputs) -> None:
    graph, tensor, traces = frozen_inputs
    tampered = replace(
        traces[0],
        output_top5=(1, 0, 2, 3, 4),
        trace_sha256="0" * 64,
    )
    tampered = replace(
        tampered,
        trace_sha256=operator.recompute_action_trace_sha256(tampered),
    )
    # A valid self hash is insufficient: the producer replays the frozen
    # operator and compares the entire semantic action trace.
    assert operator.verify_action_trace(tampered) == tampered.trace_sha256
    with pytest.raises(
        producer.FeverousFeatureProducerError, match="semantic replay"
    ):
        producer.produce_e2_recipe_trace(
            item_commitment_sha256=ITEM,
            graph=graph,
            semantic_tensor=tensor,
            action_trace=tampered,
            external_operator_receipt_sha256=OPERATOR_RECEIPT,
            external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
        )


def test_feature_provenance_and_nested_e2_tampering_fail_closed(
    frozen_inputs,
) -> None:
    produced = _produce_matrix(frozen_inputs)[0]
    changed_features = replace(
        produced.exact_features,
        direct_facet_coverage=(
            produced.exact_features.direct_facet_coverage + 1
        ),
    )
    changed = replace(produced, exact_features=changed_features)
    with pytest.raises(
        producer.FeverousFeatureProducerError,
        match="nested E2 trace disagrees with exact features",
    ):
        producer.verify_feature_production_trace(
            changed,
            external_operator_receipt_sha256=OPERATOR_RECEIPT,
            external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
        )

    changed_provenance = replace(
        produced.provenance,
        query_sha256="b" * 64,
    )
    changed = replace(produced, provenance=changed_provenance)
    with pytest.raises(
        producer.FeverousFeatureProducerError, match="provenance self hash"
    ):
        producer.verify_feature_production_trace(
            changed,
            external_operator_receipt_sha256=OPERATOR_RECEIPT,
            external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
        )

    changed_trace_hash = replace(
        produced, production_trace_sha256="f" * 64
    )
    with pytest.raises(
        producer.FeverousFeatureProducerError, match="trace self hash"
    ):
        producer.verify_feature_production_trace(
            changed_trace_hash,
            external_operator_receipt_sha256=OPERATOR_RECEIPT,
            external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
        )


def test_consistently_rehashed_feature_forgery_fails_trusted_input_replay(
    frozen_inputs,
) -> None:
    graph, tensor, traces = frozen_inputs
    produced = _produce_matrix(frozen_inputs)[0]
    forged_features = replace(
        produced.exact_features,
        dense_relevance_mass=produced.exact_features.dense_relevance_mass + 1,
    )
    forged_recipe = RecipeTrace.from_mapping(
        item_commitment_sha256=ITEM,
        recipe_id=produced.recipe_trace.recipe_id,
        behavior_sha256=produced.recipe_trace.behavior_sha256,
        features=forged_features.as_mapping(),
    )
    forged = replace(
        produced,
        exact_features=forged_features,
        recipe_trace=forged_recipe,
        production_trace_sha256="0" * 64,
    )
    forged = replace(
        forged,
        production_trace_sha256=producer.recompute_feature_production_sha256(
            forged
        ),
    )
    # Internal semantic hashing is coherent, but trusted input replay still
    # distinguishes the forged coordinate from the frozen computation.
    assert producer.verify_feature_production_trace(
        forged,
        external_operator_receipt_sha256=OPERATOR_RECEIPT,
        external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
    ) == forged.production_trace_sha256
    with pytest.raises(
        producer.FeverousFeatureProducerError, match="trusted input replay"
    ):
        producer.verify_feature_production_against_inputs(
            forged,
            graph=graph,
            semantic_tensor=tensor,
            action_trace=traces[0],
            external_operator_receipt_sha256=OPERATOR_RECEIPT,
            external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
        )


def test_matrix_requires_each_of_four_recipes_exactly_once(frozen_inputs) -> None:
    graph, tensor, traces = frozen_inputs
    for invalid in (traces[:3], (traces[0], traces[0], traces[2], traces[3])):
        with pytest.raises(
            producer.FeverousFeatureProducerError,
            match="exactly four|duplicate recipe",
        ):
            producer.produce_complete_e2_recipe_matrix(
                item_commitment_sha256=ITEM,
                graph=graph,
                semantic_tensor=tensor,
                action_traces=invalid,
                external_operator_receipt_sha256=OPERATOR_RECEIPT,
                external_semantic_receipt_sha256=SEMANTIC_RECEIPT,
            )


def test_public_producer_api_has_no_forbidden_result_or_comparator_input() -> None:
    forbidden = ("label", "family", "gold", "evidence", "hippo", "utility")
    for function in (
        producer.produce_e2_recipe_trace,
        producer.produce_complete_e2_recipe_matrix,
    ):
        names = tuple(inspect.signature(function).parameters)
        assert not any(token in name.casefold() for name in names for token in forbidden)
    assert producer.CORPUS_UNIT_COUNT == 8192
    assert producer.FEATURE_ORDER == (
        "direct_facet_coverage",
        "residual_facet_coverage",
        "deletion_mean_coverage_drop",
        "deletion_minimum_coverage_drop",
        "same_type_replacement_mean_coverage_drop",
        "query_anchored_path_coverage",
        "dense_relevance_mass",
        "negative_pairwise_redundancy",
    )
