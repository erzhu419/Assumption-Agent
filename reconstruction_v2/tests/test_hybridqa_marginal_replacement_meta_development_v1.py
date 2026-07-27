from __future__ import annotations

from fractions import Fraction

import numpy as np

from assumption_agent.benchmarks import (
    hybridqa_marginal_replacement_meta_development_v1 as subject,
)
from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as operator


def test_exact_sign_flip_p_preserves_magnitudes() -> None:
    assert subject.exact_sign_flip_p(
        (Fraction(1), Fraction(1), Fraction(1))
    ) == Fraction(1, 8)
    assert subject.exact_sign_flip_p(
        (Fraction(1), Fraction(-1), Fraction(0))
    ) == Fraction(1)


def test_self_hash_round_trip() -> None:
    value = subject.self_hashed(
        {
            "schema": "synthetic",
            "version": "v1",
            "rows": [1, 2, 3],
        },
        "self_sha256",
    )
    assert subject.verify_self_hash(value, "self_sha256") == value["self_sha256"]


def test_feature_registry_is_fixed_and_unique() -> None:
    assert len(subject.FEATURE_ORDER) == 22
    assert len(set(subject.FEATURE_ORDER)) == len(subject.FEATURE_ORDER)
    assert subject.MAX_REPLACEMENTS == 2
    assert subject.MAX_PATH_LENGTH == 2
    assert subject.RIDGE_LAMBDA == 1.0
    assert subject.EXPECTED_GPU_CANARY["repeat_exact"] is True
    assert subject.EXPECTED_GPU_RUNTIME_VERSIONS["torch"] == "2.4.1+cu118"


def test_no_op_boundary_is_strictly_zero() -> None:
    weights = np.asarray([1.0, -1.0], dtype=np.float64)
    no_op = np.zeros(2, dtype=np.float64)
    assert float(no_op @ weights) == 0.0


def _synthetic_item() -> tuple[subject.DiagnosticItem, operator.TypedCorpusGraph]:
    units = [
        operator.AtomicUnit(
            index,
            "table_row",
            "shared" if index < 6 else f"table_{index}",
            index if index < 6 else 0,
            (),
        )
        for index in range(operator.CORPUS_UNIT_COUNT)
    ]
    graph = operator.build_typed_graph(units)
    facet = operator.make_query_facet(0, "relation_clause", "synthetic relation")
    dense = [0] * operator.CORPUS_UNIT_COUNT
    for ordinal, score in enumerate((1_000_000, 900_000, 800_000, 700_000, 600_000)):
        dense[ordinal] = score
    dense[5] = 500_000
    coverage = [[0] * operator.CORPUS_UNIT_COUNT]
    coverage[0][5] = 1_000_000
    anchors = [[0] * operator.CORPUS_UNIT_COUNT]
    anchors[0][4] = 600_000
    tensor = operator.make_query_semantic_tensor(
        query_sha256="1" * 64,
        facets=(facet,),
        semantic_coverage_ints=coverage,
        direct_anchor_strength_ints=anchors,
        dense_relevance_ints=dense,
    )
    order = tuple(
        sorted(
            range(operator.CORPUS_UNIT_COUNT),
            key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
        )
    )
    ranks = [0] * operator.CORPUS_UNIT_COUNT
    for rank, ordinal in enumerate(order):
        ranks[ordinal] = rank
    reachability = operator._query_anchored_reachability(graph, tensor)
    candidates = tuple(
        ordinal
        for ordinal, record in enumerate(reachability)
        if ordinal not in set(order[:5])
        and record.path_length is not None
        and record.path_length <= 2
    )
    return (
        subject.DiagnosticItem(
            block="A_form",
            family="TABLE_ONLY",
            commitment="2" * 64,
            gold=(5,),
            tensor=tensor,
            raw_top5=tuple(order[:5]),
            raw_rank=tuple(ranks),
            reachability=reachability,
            candidates=candidates,
        ),
        graph,
    )


def test_typed_candidate_expands_outside_raw_without_gain_filter() -> None:
    item, graph = _synthetic_item()
    assert item.candidates == (5,)
    actions = subject.enumerate_actions(
        item=item,
        graph=graph,
        state=item.raw_top5,
        available_slots=range(5),
        step=0,
    )
    assert len(actions) == 5
    assert all(action.candidate == 5 for action in actions)
    outcome, _rows = subject.oracle_trajectory(
        item=item,
        graph=graph,
        collect_training_rows=False,
    )
    assert 5 in outcome.output
    assert outcome.replacements == 1
