from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import (
    eraser_evidence_inference_local_runtime_v1 as runtime,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_r7_e3_runner_v1 as runner,
)
from assumption_agent.benchmarks import (
    eraser_evidence_inference_three_arm_scheduler_v1 as subject,
)


def _views(count: int = 30, *, sentence_count: int = 50) -> tuple[runtime.ItemTextView, ...]:
    return tuple(
        runtime.ItemTextView(
            item_commitment_sha256=hashlib.sha256(
                f"item-{item}".encode("ascii")
            ).hexdigest(),
            query=f"Q exact query {item}",
            intervention=f"I exact intervention {item}",
            comparator=f"C exact comparator {item}",
            outcome=f"O exact outcome {item}",
            official_tokenized_sentences=tuple(
                ("S", f"item-{item}", "sentence", str(ordinal))
                for ordinal in range(sentence_count)
            ),
        )
        for item in range(count)
    )


class _OrthogonalEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        values = tuple(texts)
        self.calls.append(values)
        matrix = np.zeros((len(values), 384), dtype=np.float32)
        for index, value in enumerate(values):
            # Query and I/C/O are mutually identical in semantic direction;
            # every sentence is exactly orthogonal, so no positive R7 anchor
            # exists and R0/R7 both dense-fill deterministically.
            matrix[index, 1 if value.startswith("S ") else 0] = 1.0
        return matrix


@dataclass
class _BarrierState:
    expected_submits: int
    submit_count: int = 0
    result_count: int = 0
    first_result_submit_count: int | None = None


class _LazyFuture:
    def __init__(
        self,
        state: _BarrierState,
        function: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        self._state = state
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._done = False
        self._value: object = None
        self.cancelled = False

    def result(self) -> object:
        if self._state.first_result_submit_count is None:
            self._state.first_result_submit_count = self._state.submit_count
        assert self._state.submit_count == self._state.expected_submits
        self._state.result_count += 1
        if not self._done:
            self._value = self._function(*self._args, **self._kwargs)
            self._done = True
        return self._value

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _LazyExecutor:
    def __init__(
        self,
        *,
        state: _BarrierState,
        pool_name: str,
        max_workers: int,
        observed_workers: dict[str, int],
    ) -> None:
        self.state = state
        self.pool_name = pool_name
        observed_workers[pool_name] = max_workers

    def __enter__(self) -> "_LazyExecutor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(
        self,
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> _LazyFuture:
        self.state.submit_count += 1
        return _LazyFuture(self.state, function, args, kwargs)


class _FakeHippo:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.prepared_blocks: tuple[str, ...] = ()

    def prepare_blocks(self, blocks: Sequence[str]) -> tuple[object, ...]:
        self.prepared_blocks = tuple(blocks)
        self.events.append(("hippo_prepare_blocks", self.prepared_blocks))
        return ()

    def retrieve_artifact(
        self,
        *,
        block: str,
        view: runtime.ItemTextView,
    ) -> runtime.HippoExecutionArtifact:
        self.events.append(("hippo_item", view.item_commitment_sha256))
        return runtime.HippoExecutionArtifact(
            block=block,
            item_commitment_sha256=view.item_commitment_sha256,
            top5=(0, 1, 2, 3, 4),
        )


class _FakeRuntimeBundle:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.encoder = _OrthogonalEncoder()
        self.hippo = _FakeHippo(events)

    def prepare(
        self,
        items_by_block: dict[str, tuple[runtime.ItemTextView, ...]],
    ) -> runtime.PreparedBatchArtifact:
        self.events.append(("semantic_prepare", tuple(items_by_block)))
        return runtime.prepare_semantic_batch(
            items_by_block=items_by_block,
            encoder=self.encoder,
        )


def _executor_factories(
    state: _BarrierState,
) -> tuple[Callable[..., _LazyExecutor], Callable[..., _LazyExecutor], dict[str, int]]:
    observed: dict[str, int] = {}

    def local_factory(*, max_workers: int) -> _LazyExecutor:
        return _LazyExecutor(
            state=state,
            pool_name="local",
            max_workers=max_workers,
            observed_workers=observed,
        )

    def hippo_factory(*, max_workers: int) -> _LazyExecutor:
        return _LazyExecutor(
            state=state,
            pool_name="hippo",
            max_workers=max_workers,
            observed_workers=observed,
        )

    return local_factory, hippo_factory, observed


def test_all_3n_futures_are_submitted_before_first_result_and_outputs_are_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _views()
    state = _BarrierState(expected_submits=3 * len(views))
    local_factory, hippo_factory, workers = _executor_factories(state)
    events: list[tuple[str, object]] = []
    bundle = _FakeRuntimeBundle(events)

    quantized_calls: list[tuple[object, object]] = []
    original_quantized = runtime.qasper_binding.quantized_cosine_similarity

    def counted_quantized(left: object, right: object) -> int:
        quantized_calls.append((left, right))
        return original_quantized(left, right)

    monkeypatch.setattr(
        runtime.qasper_binding,
        "quantized_cosine_similarity",
        counted_quantized,
    )
    result = subject.run_three_arm_schedule(
        items_by_block={"A_hold": views},
        runtime_bundle=bundle,
        local_executor_factory=local_factory,
        hippo_executor_factory=hippo_factory,
    )

    assert events[:2] == [
        ("semantic_prepare", ("A_hold",)),
        ("hippo_prepare_blocks", ("A_hold",)),
    ]
    assert len(bundle.encoder.calls) == 1
    assert state.submit_count == state.result_count == 90
    assert state.first_result_submit_count == 90
    assert workers == {"local": 32, "hippo": 30}
    assert result.total_item_count == 30
    assert result.submitted_task_count == 90
    assert result.local_pool_max_workers == 32
    assert result.hippo_pool_max_workers == 30

    block = result.by_block["A_hold"]
    assert isinstance(block.feature_seal, runner.FeatureSeal)
    assert isinstance(block.hippo_arm_seal, subject.HippoArmSeal)
    assert isinstance(block.raw_arm_seal, subject.RawArmSeal)
    assert isinstance(block.hippo_retrieval_seal, runner.HippoRetrievalSeal)
    assert isinstance(block.raw_retrieval_seal, runner.RawRetrievalSeal)
    assert block.hippo_retrieval_seal.rows == block.hippo_arm_seal.rows
    assert block.raw_retrieval_seal.rows == block.raw_arm_seal.rows
    assert len(block.items) == 30
    assert all(
        item.raw.r0_action is not item.agent.r0_action
        and item.raw.r0_action.trace_sha256 == item.agent.r0_action.trace_sha256
        and item.raw.top5 == item.agent.r0_action.output_top5
        for item in block.items
    )
    assert all(len(item.agent.pair_rows) == 10 for item in block.items)
    # 4 query/facet-by-sentence rows during preparation, then only the ten
    # selected top-five pairs.  A full 50x50 item scan would be much larger.
    assert len(quantized_calls) == 30 * (4 * 50 + 10)
    assert block.archive_payload["all_3n_tasks_submitted_before_first_result"] is True
    assert block.archive_payload["full_square_pair_scan_performed"] is False
    assert block.receipt["labels_evaluator_source_or_network_calls"] == 0
    assert result.receipt["labels_evaluator_source_or_network_calls"] == 0

    encoded_archive = json.dumps(block.archive_payload, sort_keys=True)
    assert "Q exact query" not in encoded_archive
    assert "S item-" not in encoded_archive
    assert "exact intervention" not in encoded_archive
    assert block.archive_payload["raw_content_persisted"] is False


def test_pre_anchor_block_uses_scheduler_owned_hippo_and_raw_seals() -> None:
    views = _views(count=48, sentence_count=5)
    state = _BarrierState(expected_submits=144)
    local_factory, hippo_factory, workers = _executor_factories(state)
    result = subject.run_three_arm_schedule(
        items_by_block={"A_form": views},
        runtime_bundle=_FakeRuntimeBundle([]),
        local_executor_factory=local_factory,
        hippo_executor_factory=hippo_factory,
    )

    block = result.by_block["A_form"]
    assert state.first_result_submit_count == 144
    assert workers == {"local": 32, "hippo": 32}
    assert isinstance(block.feature_seal, runner.FeatureSeal)
    assert isinstance(block.hippo_arm_seal, subject.HippoArmSeal)
    assert isinstance(block.raw_arm_seal, subject.RawArmSeal)
    assert block.hippo_retrieval_seal is None
    assert block.raw_retrieval_seal is None
    assert len(block.hippo_arm_seal.rows) == 48
    assert len(block.raw_arm_seal.rows) == 48
    assert (
        block.hippo_arm_seal.item_commitment_set_sha256
        == block.raw_arm_seal.item_commitment_set_sha256
        == block.feature_seal.item_commitment_set_sha256
    )
    assert block.receipt["anchor_hipporag_retrieval_matrix_sha256"] is None
    assert block.receipt["anchor_raw_retrieval_receipt_sha256"] is None


def test_scheduler_rejects_raw_action_object_reuse_even_when_hashes_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = _views(sentence_count=5)
    state = _BarrierState(expected_submits=90)
    local_factory, hippo_factory, _workers = _executor_factories(state)
    bundle = _FakeRuntimeBundle([])
    real_agent = runtime.execute_agent
    by_item: dict[str, runtime.AgentExecutionArtifact] = {}

    def remembered_agent(
        prepared: runtime.PreparedItemArtifact,
    ) -> runtime.AgentExecutionArtifact:
        result = real_agent(prepared)
        by_item[prepared.item_commitment_sha256] = result
        return result

    def reused_raw(
        prepared: runtime.PreparedItemArtifact,
    ) -> runtime.RawExecutionArtifact:
        agent = by_item[prepared.item_commitment_sha256]
        return runtime.RawExecutionArtifact(
            item_commitment_sha256=prepared.item_commitment_sha256,
            graph_sha256=prepared.graph.graph_sha256,
            semantic_tensor_sha256=prepared.semantic_tensor.tensor_sha256,
            r0_action=agent.r0_action,
        )

    monkeypatch.setattr(runtime, "execute_agent", remembered_agent)
    monkeypatch.setattr(runtime, "execute_raw", reused_raw)
    with pytest.raises(
        subject.EraserEvidenceInferenceThreeArmSchedulerError,
        match="RAW reused",
    ):
        subject.run_three_arm_schedule(
            items_by_block={"A_hold": views},
            runtime_bundle=bundle,
            local_executor_factory=local_factory,
            hippo_executor_factory=hippo_factory,
        )
    assert state.first_result_submit_count == 90


@pytest.mark.parametrize(
    "items_by_block,match",
    (
        ({"unknown": _views()}, "four-block registry"),
        ({"A_hold": _views(count=29)}, "count"),
    ),
)
def test_invalid_block_or_count_fails_before_runtime_preparation(
    items_by_block: dict[str, tuple[runtime.ItemTextView, ...]],
    match: str,
) -> None:
    class NeverRuntime:
        def prepare(self, _rows: object) -> object:
            raise AssertionError("invalid scheduler input reached runtime")

    with pytest.raises(
        subject.EraserEvidenceInferenceThreeArmSchedulerError,
        match=match,
    ):
        subject.run_three_arm_schedule(
            items_by_block=items_by_block,
            runtime_bundle=NeverRuntime(),
        )
