from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from assumption_agent.benchmarks import (
    hybridqa_set_interaction_meta_development_v1 as subject,
)
from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as operator


def _synthetic_item() -> tuple[subject.DiagnosticItem, operator.TypedCorpusGraph]:
    units: list[operator.AtomicUnit] = []
    for index in range(operator.CORPUS_UNIT_COUNT):
        if index <= 5:
            units.append(
                operator.AtomicUnit(
                    index,
                    "table_row",
                    "shared",
                    index,
                    ("target",) if index == 5 else (),
                )
            )
        elif index == 6:
            units.append(
                operator.AtomicUnit(
                    index,
                    "linked_passage",
                    "shared",
                    None,
                    ("target",),
                )
            )
        else:
            units.append(
                operator.AtomicUnit(
                    index,
                    "table_row",
                    f"table_{index}",
                    0,
                    (),
                )
            )
    graph = operator.build_typed_graph(units)
    facets = (
        operator.make_query_facet(0, "entity", "synthetic entity"),
        operator.make_query_facet(1, "relation_clause", "synthetic relation"),
    )
    dense = [0] * operator.CORPUS_UNIT_COUNT
    for ordinal, score in enumerate(
        (1_000_000, 900_000, 800_000, 700_000, 600_000, 500_000, 400_000)
    ):
        dense[ordinal] = score
    coverage = [[0] * operator.CORPUS_UNIT_COUNT for _ in facets]
    coverage[0][:7] = [
        101_003,
        202_007,
        303_011,
        404_017,
        505_019,
        923_457,
        246_802,
    ]
    coverage[1][:7] = [
        111_013,
        222_023,
        333_031,
        444_049,
        555_061,
        135_791,
        876_543,
    ]
    anchors = [[0] * operator.CORPUS_UNIT_COUNT for _ in facets]
    anchors[0][4] = 800_000
    tensor = operator.make_query_semantic_tensor(
        query_sha256="1" * 64,
        facets=facets,
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
        and record.path_length <= subject.MAX_PATH_LENGTH
    )
    assert candidates == (5, 6)
    return (
        subject.DiagnosticItem(
            block="A_form",
            family="DUAL_TABLE_PASSAGE",
            commitment="2" * 64,
            gold=(5, 6),
            tensor=tensor,
            raw_top5=tuple(order[:5]),
            raw_rank=tuple(ranks),
            reachability=reachability,
            candidates=candidates,
        ),
        graph,
    )


def test_fixed_registry_and_complete_state_formula() -> None:
    assert len(subject.FEATURE_ORDER) == 48
    assert len(set(subject.FEATURE_ORDER)) == len(subject.FEATURE_ORDER)
    assert subject.MAX_REPLACEMENTS == 2
    assert subject.RIDGE_LAMBDA == 1.0
    assert subject.MAX_WORKERS == 8
    assert subject.complete_state_count(2) == 21
    assert subject.complete_state_count(64) == 20_481
    assert subject.EXPECTED_LEGACY_FULL_PROJECTION_SHA256 == (
        "395f755517bd756bb16eb83c02dd15a6c9a46e10023561be2eaf89850fefe943"
    )
    assert subject.EXPECTED_SET_ENERGY_NUMERIC_CANARY_SHA256 == (
        "81e04ae053607c568c4b98885acbad1e5700ffe882eff7aeda26aa99aace6d7c"
    )


def test_source_free_numeric_canary_exercises_frozen_shape() -> None:
    receipt = subject.compute_set_energy_numeric_canary()
    assert receipt["matrix_shape"] == [257, 48]
    assert receipt["score_count"] == 37
    assert len(receipt["float64_payload_sha256"]) == 64


def test_complete_set_enumeration_has_no_assignment_duplicates() -> None:
    item, graph = _synthetic_item()
    batches = tuple(subject.iter_state_batches(item=item, graph=graph))
    outputs = [
        tuple(sorted(int(value) for value in output))
        for batch in batches
        for output in batch.outputs
    ]
    assert len(outputs) == subject.complete_state_count(len(item.candidates)) - 1
    assert len(set(outputs)) == len(outputs)
    assert sum(batch.replacement_count == 1 for batch in batches) == 5
    assert sum(batch.replacement_count == 2 for batch in batches) == 10
    assert all(
        batch.features.shape == (len(batch.outputs), len(subject.FEATURE_ORDER))
        for batch in batches
    )


def test_candidate_universe_cannot_be_pruned() -> None:
    item, graph = _synthetic_item()
    pruned = replace(item, candidates=item.candidates[:-1])
    with pytest.raises(
        subject.HybridQaSetInteractionError,
        match="candidate universe",
    ):
        tuple(subject.iter_state_batches(item=pruned, graph=graph))


def test_pair_features_are_invariant_to_candidate_slot_assignment() -> None:
    item, graph = _synthetic_item()
    context = subject._build_feature_context(item=item, graph=graph)
    left = np.asarray([[5, 6]], dtype=np.int64)
    left_output = np.asarray([[5, 6, 2, 3, 4]], dtype=np.int64)
    right = np.asarray([[6, 5]], dtype=np.int64)
    right_output = np.asarray([[6, 5, 2, 3, 4]], dtype=np.int64)
    left_features = subject._state_features(
        item=item,
        context=context,
        slots=(0, 1),
        candidate_local=left,
        outputs_local=left_output,
    )
    right_features = subject._state_features(
        item=item,
        context=context,
        slots=(0, 1),
        candidate_local=right,
        outputs_local=right_output,
    )
    assert np.array_equal(left_features, right_features)


def test_noop_is_bitwise_zero_and_global_oracle_can_choose_pair() -> None:
    item, graph = _synthetic_item()
    no_op = np.zeros(len(subject.FEATURE_ORDER), dtype=np.float64)
    assert no_op.tobytes() == bytes(no_op.nbytes)
    learned, oracle = subject.select_set_and_oracle(
        item=item,
        graph=graph,
        weights=no_op,
    )
    assert learned == subject.PolicyOutcome(item.raw_top5, 0)
    assert oracle.replacements == 2
    assert {5, 6}.issubset(oracle.output)


def test_item_statistics_cover_every_state_and_are_symmetric() -> None:
    item, graph = _synthetic_item()
    statistics = subject.item_sufficient_statistics(item=item, graph=graph)
    assert statistics.state_count == subject.complete_state_count(2)
    assert statistics.non_noop_state_count == statistics.state_count - 1
    assert statistics.stratum_count >= 2
    assert statistics.gram.shape == (
        len(subject.FEATURE_ORDER),
        len(subject.FEATURE_ORDER),
    )
    assert np.array_equal(statistics.gram, statistics.gram.T)
    assert np.isfinite(statistics.target).all()


def test_features_ignore_commitment_gold_family_and_block() -> None:
    item, graph = _synthetic_item()
    changed = replace(
        item,
        block="M_search",
        family="TABLE_ONLY",
        commitment="3" * 64,
        gold=(0,),
    )
    original_features = np.concatenate(
        [
            batch.features
            for batch in subject.iter_state_batches(item=item, graph=graph)
        ],
        axis=0,
    )
    changed_features = np.concatenate(
        [
            batch.features
            for batch in subject.iter_state_batches(item=changed, graph=graph)
        ],
        axis=0,
    )
    assert np.array_equal(original_features, changed_features)


def test_connectivity_lookup_matches_simple_graphs() -> None:
    assert subject._COMPONENT_COUNT_BY_MASK[0] == 5
    assert subject._LARGEST_COMPONENT_BY_MASK[0] == 1
    assert subject._COMPONENT_COUNT_BY_MASK[1] == 4
    assert subject._LARGEST_COMPONENT_BY_MASK[1] == 2
    assert subject._COMPONENT_COUNT_BY_MASK[(1 << 10) - 1] == 1
    assert subject._LARGEST_COMPONENT_BY_MASK[(1 << 10) - 1] == 5


def test_self_hash_round_trip() -> None:
    value = subject.self_hashed(
        {"schema": "synthetic", "version": "v1", "rows": [1, 2, 3]},
        "self_sha256",
    )
    assert subject.verify_self_hash(value, "self_sha256") == value["self_sha256"]
