from __future__ import annotations

import inspect

import numpy as np
import pytest

from assumption_agent.benchmarks import tatqa_p19_label_free_runtime_v1 as runtime
from assumption_agent.benchmarks import tatqa_p19_typed_evaluator_core_v1 as core


def _plan() -> core.TypedPlan:
    return core.TypedPlan(
        entity_facets=("Acme",),
        metric_facets=("revenue",),
        time_facets=("2024",),
        operation="COMPARE",
        relation_query="Acme revenue in 2024",
    )


def _item() -> runtime.LabelFreeRuntimeItem:
    return runtime.LabelFreeRuntimeItem(
        item_id="a" * 64,
        question="How did Acme revenue change in 2024?",
        units=(
            runtime.RuntimeUnit("T:0", "TABLE HEADER | company | revenue | year"),
            runtime.RuntimeUnit("T:1", "Acme | 100 | 2023"),
            runtime.RuntimeUnit("T:2", "Acme | 130 | 2024"),
            runtime.RuntimeUnit("P:1", "Acme reported stronger sales."),
            runtime.RuntimeUnit("P:2", "Revenue increased in the 2024 period."),
            runtime.RuntimeUnit("P:3", "The appendix supplies a period anchor."),
        ),
    )


def _normalized(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    values = np.pad(
        values,
        ((0, 0), (0, runtime.EMBEDDING_DIMENSION - values.shape[1])),
        mode="constant",
    )
    values /= np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True).astype(
        np.float32
    )
    return values


def _matrix() -> np.ndarray:
    # Input order is question, four plan facets, then six canonical units.
    return _normalized(
        [
            [1.0, 1.0, 0.0, 0.0],  # question
            [1.0, 0.0, 0.0, 0.0],  # entity
            [0.0, 1.0, 0.0, 0.0],  # metric
            [0.0, 0.0, 1.0, 0.0],  # time
            [1.0, 1.0, 1.0, 0.0],  # relation
            [0.1, 0.1, 0.1, 1.0],  # T:0
            [0.8, 0.1, 0.1, 0.1],  # T:1 entity table winner
            [0.4, 0.8, 0.9, 0.1],  # T:2 metric/time table winner
            [0.9, 0.1, 0.1, 0.1],  # P:1 entity paragraph winner
            [0.2, 0.9, 0.8, 0.1],  # P:2 metric/time paragraph winner
            [0.0, 0.0, 1.0, 0.0],  # P:3 time facet, low full-query relevance
        ]
    )


def test_runtime_item_is_canonical_label_free_and_content_bounded() -> None:
    item = _item()
    assert tuple(row.unit_id for row in item.units) == (
        "T:0",
        "T:1",
        "T:2",
        "P:1",
        "P:2",
        "P:3",
    )
    with pytest.raises(runtime.TatqaP19LabelFreeRuntimeError, match="order"):
        runtime.LabelFreeRuntimeItem(
            item_id="b" * 64,
            question="q",
            units=(item.units[1], item.units[0], *item.units[2:]),
        )
    with pytest.raises(runtime.TatqaP19LabelFreeRuntimeError, match="header"):
        runtime.LabelFreeRuntimeItem(
            item_id="b" * 64,
            question="q",
            units=tuple(row for row in item.units if row.unit_id != "T:0"),
        )
    with pytest.raises(runtime.TatqaP19LabelFreeRuntimeError, match="positive"):
        runtime.RuntimeUnit("P:0", "invalid paragraph zero")


def test_embedding_order_is_frozen_and_has_no_identity_or_label_text() -> None:
    texts = runtime.embedding_texts(_item(), _plan())
    assert texts[:5] == (
        _item().question,
        "Acme",
        "revenue",
        "2024",
        "Acme revenue in 2024",
    )
    assert texts[5:] == tuple(row.text for row in _item().units)
    assert _item().item_id not in texts


def test_semantic_top_one_assignment_has_no_threshold_and_ties_are_canonical() -> None:
    compiled = runtime.compile_from_embeddings(_item(), _plan(), _matrix())
    by_id = {row.unit_id: row for row in compiled.units}
    assert sum(sum(row.facet_coverage) for row in compiled.units) == _plan().facet_width
    assert any(
        row.typed_edge_features[0] != sum(row.facet_coverage)
        for row in compiled.units
    )
    # The entity winner is P:1; metric/time/relation are assigned by exact
    # semantic maxima, independently of any hand-tuned threshold.
    assert by_id["P:1"].facet_coverage[0] == 1
    assert by_id["P:2"].facet_coverage[1] == 1
    assert by_id["P:3"].facet_coverage[2] == 1
    assert by_id["T:2"].typed_edge_features[2] >= 1
    assert by_id["P:1"].typed_edge_features[3] >= 1
    assert by_id["P:2"].typed_edge_features[3] >= 1
    assert sum(row.typed_edge_features[3] for row in compiled.units) > 0
    relation_changed = _matrix().copy()
    relation_changed[4] = np.roll(relation_changed[4], 1)
    changed = runtime.compile_from_embeddings(_item(), _plan(), relation_changed)
    assert sum(row.typed_edge_features[3] for row in changed.units) == sum(
        row.typed_edge_features[3] for row in compiled.units
    )
    assert len(compiled.raw_top5) == 5
    assert len(set(compiled.raw_top5)) == 5
    assert len(compiled.tensor_sha256) == 64

    tied = _matrix().copy()
    tied[5 + 4] = tied[5 + 3]  # P:2 equals P:1 for all facets.
    first = runtime.compile_from_embeddings(_item(), _plan(), tied)
    second = runtime.compile_from_embeddings(_item(), _plan(), tied)
    assert first == second


def test_compiler_builds_actions_raw_and_nonnegative_canonical_redundancy() -> None:
    compiled = runtime.compile_from_embeddings(_item(), _plan(), _matrix())
    p0, p1 = core.build_action_pair(
        compiled.plan,
        compiled.units,
        redundancy_features=compiled.redundancy_features,
    )
    assert len(p0.selected_unit_ids) == len(p1.selected_unit_ids) == 5
    assert p0.selected_unit_ids != p1.selected_unit_ids
    assert p1.feature_mapping()["P1_outside_P0_unit_count"] >= 1
    assert compiled.raw_top5 == tuple(
        sorted(
            (row.unit_id for row in compiled.units),
            key=lambda unit_id: (
                -next(
                    row.full_question_similarity
                    for row in compiled.units
                    if row.unit_id == unit_id
                ),
                (0 if unit_id.startswith("T:") else 1, int(unit_id[2:])),
            ),
        )[:5]
    )
    canonical = lambda value: (0 if value.startswith("T:") else 1, int(value[2:]))
    assert all(
        canonical(left) < canonical(right)
        for left, right in compiled.redundancy_features
    )
    assert all(value >= 0 for value in compiled.redundancy_features.values())


class _Encoder:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts):
        self.calls.append(tuple(texts))
        return self.matrix


def test_encoder_is_called_once_with_complete_frozen_batch() -> None:
    encoder = _Encoder(_matrix())
    compiled = runtime.compile_with_encoder(_item(), _plan(), encoder)
    assert len(encoder.calls) == 1
    assert encoder.calls[0] == runtime.embedding_texts(_item(), _plan())
    assert compiled == runtime.compile_from_embeddings(_item(), _plan(), _matrix())


@pytest.mark.parametrize(
    "bad",
    (
        np.zeros((11, runtime.EMBEDDING_DIMENSION), dtype=np.float32),
        np.full((11, runtime.EMBEDDING_DIMENSION), np.nan, dtype=np.float32),
        np.ones((10, runtime.EMBEDDING_DIMENSION), dtype=np.float32),
        np.ones((11, runtime.EMBEDDING_DIMENSION), dtype=np.float64),
        np.ones((11, 4), dtype=np.float32),
    ),
)
def test_embedding_tensor_fails_closed(bad: np.ndarray) -> None:
    with pytest.raises(runtime.TatqaP19LabelFreeRuntimeError):
        runtime.compile_from_embeddings(_item(), _plan(), bad)


def test_public_surface_has_no_gold_family_answer_or_baseline_inputs() -> None:
    for cls in (runtime.RuntimeUnit, runtime.LabelFreeRuntimeItem):
        assert not set(cls.__dataclass_fields__).intersection(
            {"gold", "answer", "answer_from", "family", "raw", "hipporag"}
        )
    for function in (
        runtime.embedding_texts,
        runtime.compile_from_embeddings,
        runtime.compile_with_encoder,
    ):
        assert not set(inspect.signature(function).parameters).intersection(
            {"gold", "answer", "answer_from", "family", "raw", "hipporag"}
        )
