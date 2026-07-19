from __future__ import annotations

from dataclasses import fields, replace
from fractions import Fraction
import inspect
from itertools import combinations

import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_exact_feature_bridge_v1 as subject,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_operator_v1 as operator,
)


def _operator_fixture() -> tuple[
    operator.QueryAnchoredSentenceGraph,
    operator.QuerySemanticTensor,
    operator.ActionTrace,
    operator.ActionTrace,
]:
    count = 12
    units = tuple(
        operator.SentenceUnit(
            sentence_ordinal=ordinal,
            start_token=ordinal * 2,
            end_token=(ordinal + 1) * 2,
            sentence_sha256=f"{ordinal + 1:064x}",
        )
        for ordinal in range(count)
    )
    facets = operator.make_official_ico_facets(
        intervention_sha256="a" * 64,
        comparator_sha256="b" * 64,
        outcome_sha256="c" * 64,
    )
    anchor_row = tuple(100 if ordinal == 8 else 0 for ordinal in range(count))
    tensor = operator.make_query_semantic_tensor(
        query_sha256="d" * 64,
        facets=facets,
        facet_similarity_ints=(anchor_row, anchor_row, anchor_row),
        dense_relevance_ints=(
            100,
            99,
            98,
            97,
            96,
            -100,
            -100,
            -100,
            -100,
            -100,
            95,
            -101,
        ),
    )
    graph = operator.build_query_anchored_graph(
        units=units, semantic_tensor=tensor
    )
    r0, r7 = operator.run_all_actions(graph=graph, semantic_tensor=tensor)
    assert r0.output_top5 == (0, 1, 2, 3, 4)
    assert r7.output_top5 == (10, 8, 7, 9, 6)
    return graph, tensor, r0, r7


def _pair_rows(
    r0_top5: tuple[int, ...] = (0, 1, 2, 3, 4),
    r7_top5: tuple[int, ...] = (10, 8, 7, 9, 6),
) -> tuple[tuple[int, int, int], ...]:
    # R0's ten unordered pairs have cosine +100.  R7's disjoint ten pairs
    # have cosine -100.  Everything else is zero and the exact diagonal is 1.
    r0_set = set(range(5))
    r7_set = set(range(6, 11))
    pairs = sorted(
        set(combinations(sorted(r0_top5), 2))
        | set(combinations(sorted(r7_top5), 2))
    )
    rows: list[tuple[int, int, int]] = []
    for left, right in pairs:
        if left in r0_set and right in r0_set:
            value = 100
        elif left in r7_set and right in r7_set:
            value = -100
        else:
            value = 0
        rows.append((left, right, value))
    return tuple(rows)


def _build() -> tuple[
    subject.ExactDifferenceTraceBuild,
    subject.SelectedPairSemanticReceipt,
    operator.QueryAnchoredSentenceGraph,
    operator.QuerySemanticTensor,
    operator.ActionTrace,
    operator.ActionTrace,
]:
    graph, tensor, r0, r7 = _operator_fixture()
    pair_receipt = subject.build_selected_pair_semantic_receipt(
        graph=graph,
        semantic_tensor=tensor,
        r0_top5=r0.output_top5,
        r7_top5=r7.output_top5,
        pair_rows=_pair_rows(r0.output_top5, r7.output_top5),
    )
    built = subject.build_exact_difference_trace(
        item_commitment_sha256="e" * 64,
        graph=graph,
        semantic_tensor=tensor,
        r0_action=r0,
        r7_action=r7,
        selected_pair_semantic_receipt=pair_receipt,
    )
    return built, pair_receipt, graph, tensor, r0, r7


def test_selected_pair_union_is_complete_quantized_ordinal_bound_and_self_hashed() -> None:
    graph, tensor, r0, r7 = _operator_fixture()
    receipt = subject.build_selected_pair_semantic_receipt(
        graph=graph,
        semantic_tensor=tensor,
        r0_top5=r0.output_top5,
        r7_top5=r7.output_top5,
        pair_rows=_pair_rows(r0.output_top5, r7.output_top5),
    )

    assert receipt.sentence_count == len(graph.units) == 12
    assert receipt.minilm_asset_manifest_sha256 == subject.MINILM_ASSET_MANIFEST_SHA256
    assert receipt.payload()["full_square_scan_required"] is False
    assert receipt.payload()["required_pair_count"] == 20
    assert receipt.payload()["quantization_scale"] == 1_000_000
    assert subject.verify_selected_pair_semantic_receipt(
        receipt, graph=graph, semantic_tensor=tensor
    ) == receipt.receipt_sha256
    assert subject.recompute_selected_pair_receipt_sha256(receipt) == receipt.receipt_sha256


def test_selected_pair_api_accepts_exact_integer_rows_without_an_encoder() -> None:
    parameters = inspect.signature(
        subject.build_selected_pair_semantic_receipt
    ).parameters

    assert tuple(parameters) == (
        "graph",
        "semantic_tensor",
        "r0_top5",
        "r7_top5",
        "pair_rows",
        "minilm_asset_manifest_sha256",
    )
    assert "encoder" not in parameters
    assert tuple(field.name for field in fields(subject.SelectedPairSemanticReceipt)) == (
        "graph_sha256",
        "semantic_tensor_sha256",
        "sentence_sha256s",
        "sentence_identity_sha256",
        "minilm_asset_manifest_sha256",
        "r0_top5",
        "r7_top5",
        "pair_rows",
        "receipt_sha256",
    )


@pytest.mark.parametrize("kind", ("missing", "noncanonical", "bounds", "dimension"))
def test_selected_pair_receipt_fails_closed_on_registry_quantization_and_dimension(kind: str) -> None:
    graph, tensor, r0, r7 = _operator_fixture()
    rows = [list(row) for row in _pair_rows()]
    if kind == "missing":
        rows.pop()
        match = "incomplete"
    elif kind == "noncanonical":
        rows[0], rows[1] = rows[1], rows[0]
        match = "noncanonical"
    elif kind == "bounds":
        rows[0][2] = 1_000_001
        match = "malformed"
    else:
        rows[0] = rows[0][:-1]
        match = "incomplete|malformed"

    with pytest.raises(subject.EraserExactFeatureBridgeError, match=match):
        subject.build_selected_pair_semantic_receipt(
            graph=graph,
            semantic_tensor=tensor,
            r0_top5=r0.output_top5,
            r7_top5=r7.output_top5,
            pair_rows=rows,
        )


def test_exact_synthetic_vector_causally_covers_all_eight_features() -> None:
    built, pair_receipt, graph, tensor, r0, r7 = _build()
    trace = built.difference_trace
    receipt = built.feature_receipt

    expected = (
        Fraction(5),  # every R7 terminal is outside RAW/R0 top five
        Fraction(100),  # mean I/C/O maximum rises from 0 to 100
        Fraction(3),  # all three facet maxima strictly improve
        Fraction(100),  # every selected canonical witness uses strength 100
        Fraction(-795),  # exact R7 dense mass minus exact R0 dense mass
        Fraction(20),  # (300 R7 - 0 R0) / (5 deletions * 3 facets)
        Fraction(1),  # at least one used-edge deletion changes ordinal tuple
        Fraction(200),  # negative(-100) minus negative(+100)
    )
    assert trace.features == expected
    assert tuple(zip(subject.FEATURE_ORDER, trace.features)) == tuple(
        zip(runner.FEATURE_ORDER, expected)
    )
    assert receipt.r0_facet_maxima_ints == (0, 0, 0)
    assert receipt.r7_facet_maxima_ints == (100, 100, 100)
    assert len(receipt.r0_leave_one_out_coverage_deltas) == 5
    assert len(receipt.r7_leave_one_out_coverage_deltas) == 5
    assert sum(
        value for row in receipt.r7_leave_one_out_coverage_deltas for value in row
    ) == 300
    assert receipt.r0_pair_similarity_ints == (100,) * 10
    assert receipt.r7_pair_similarity_ints == (-100,) * 10
    assert receipt.edge_deletion_action_change_indicator == 1
    assert trace.r0_action_trace_sha256 == r0.trace_sha256
    assert trace.r7_action_trace_sha256 == r7.trace_sha256
    assert receipt.r0_operator_behavior_sha256 == r0.behavior_sha256
    assert receipt.r7_operator_behavior_sha256 == r7.behavior_sha256
    assert trace.r0_behavior_sha256 == runner.behavior_sha256(
        item_commitment_sha256="e" * 64,
        recipe_id=runner.RECIPE_IDS[0],
        selected_ordinals=r0.output_top5,
    )
    assert trace.r0_behavior_sha256 != receipt.r0_operator_behavior_sha256
    assert subject.verify_exact_difference_trace_build(
        built,
        item_commitment_sha256="e" * 64,
        graph=graph,
        semantic_tensor=tensor,
        r0_action=r0,
        r7_action=r7,
        selected_pair_semantic_receipt=pair_receipt,
    ) == receipt.feature_receipt_sha256


def test_bridge_accepts_no_external_feature_mapping() -> None:
    parameters = inspect.signature(subject.build_exact_difference_trace).parameters

    assert "features" not in parameters
    assert "feature_map" not in parameters
    assert tuple(parameters) == (
        "item_commitment_sha256",
        "graph",
        "semantic_tensor",
        "r0_action",
        "r7_action",
        "selected_pair_semantic_receipt",
    )


def test_pair_action_and_receipt_tamper_fail_closed() -> None:
    built, pair_receipt, graph, tensor, r0, r7 = _build()

    changed_rows = [list(row) for row in pair_receipt.pair_rows]
    changed_rows[0][2] += 1
    tampered_pairs = replace(
        pair_receipt,
        pair_rows=tuple(tuple(row) for row in changed_rows),
    )
    with pytest.raises(subject.EraserExactFeatureBridgeError, match="self hash drifted"):
        subject.verify_selected_pair_semantic_receipt(tampered_pairs)

    tampered_action = replace(r7, trace_sha256="f" * 64)
    with pytest.raises(subject.EraserExactFeatureBridgeError, match="action verification"):
        subject.build_exact_difference_trace(
            item_commitment_sha256="e" * 64,
            graph=graph,
            semantic_tensor=tensor,
            r0_action=r0,
            r7_action=tampered_action,
            selected_pair_semantic_receipt=pair_receipt,
        )

    tampered_receipt = replace(
        built.feature_receipt,
        minimum_positive_anchor_strength=(
            built.feature_receipt.minimum_positive_anchor_strength + 1
        ),
    )
    with pytest.raises(subject.EraserExactFeatureBridgeError, match="feature vector drifted"):
        subject.verify_feature_computation_receipt(tampered_receipt)


def test_pair_receipt_rejects_wrong_model_and_cross_graph_sentence_identity() -> None:
    graph, tensor, r0, r7 = _operator_fixture()
    with pytest.raises(subject.EraserExactFeatureBridgeError, match="unfrozen model"):
        subject.build_selected_pair_semantic_receipt(
            graph=graph,
            semantic_tensor=tensor,
            r0_top5=r0.output_top5,
            r7_top5=r7.output_top5,
            pair_rows=_pair_rows(r0.output_top5, r7.output_top5),
            minilm_asset_manifest_sha256="f" * 64,
        )

    pair_receipt = subject.build_selected_pair_semantic_receipt(
        graph=graph,
        semantic_tensor=tensor,
        r0_top5=r0.output_top5,
        r7_top5=r7.output_top5,
        pair_rows=_pair_rows(r0.output_top5, r7.output_top5),
    )
    swapped = list(graph.units)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    # Reordering units is invalid even before pair measurements can be consumed.
    with pytest.raises(subject.EraserExactFeatureBridgeError, match="graph/tensor"):
        subject.build_selected_pair_semantic_receipt(
            graph=replace(graph, units=tuple(swapped)),
            semantic_tensor=tensor,
            r0_top5=r0.output_top5,
            r7_top5=r7.output_top5,
            pair_rows=pair_receipt.pair_rows,
        )
