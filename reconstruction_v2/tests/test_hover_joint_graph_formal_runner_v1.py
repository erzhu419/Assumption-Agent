from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor as ActualThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import hover_joint_graph_formal_runner_v1 as runner
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ACTION_IDS,
    SAME_SOURCE,
    CausalSignature,
    recompute_action_trace_sha256,
)
from replication_runtime.multihoprag_minilm_v1 import frozen_minilm_runtime_identity
from replication_runtime.multihoprag_ner_v1 import EntitySpan
from replication_runtime.multihoprag_official_hipporag_v1 import RetrievalBatch


class FakeEncoder:
    def __init__(self) -> None:
        identity = frozen_minilm_runtime_identity()
        self.runtime_receipt = {
            "asset_file_sha256": identity["asset_file_sha256"],
            "asset_manifest_path": "/synthetic/asset.json",
            "asset_sha256": identity["asset_sha256"],
            "embedding_dimension": identity["embedding_dimension"],
            "maximum_sequence_length": identity["maximum_sequence_length"],
            "model_root": "/synthetic/model",
            "model_tree_sha256": identity["model_tree_sha256"],
            "runtime_versions": identity["runtime_versions"],
            "status": identity["status"],
            "weights_sha256": identity["weights_sha256"],
        }
        self.canary_receipt = {
            "float32_bytes_sha256": identity["canary_float32_bytes_sha256"],
            "quantized_embedding_matrix_sha256": identity[
                "canary_quantized_embedding_sha256"
            ],
            "qasper_rows_or_archives_accessed_by_canary": False,
            "repeat_count": 2,
            "repeat_exact": True,
            "sentence_count": identity["canary_sentence_count"],
            "status": "passed_exact_row_free_synthetic_canary",
            "text_vector_sha256": identity["canary_text_vector_sha256"],
        }
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(tuple(texts))
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        matrix[:, 0] = np.float32(1.0)
        return matrix


class FakeNER:
    def __init__(self) -> None:
        self.runtime_binding = {"binding_sha256": "a" * 64, "status": "verified"}
        self.canary_receipt = {
            "status": "passed_exact_row_free_synthetic_canary",
            "receipt_sha256": "b" * 64,
        }
        self.batch_sizes: list[int] = []
        self.inputs: list[Mapping[str, object]] = []

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        self.batch_sizes.append(len(values))
        self.inputs.extend(values)
        return tuple(
            (EntitySpan(entity_type="ORG", start=0, end=4, text="Acme"),)
            for _value in values
        )


class FakeHippo:
    def __init__(self, agent_started: threading.Event | None = None) -> None:
        self.agent_started = agent_started
        self.retrieve_started = threading.Event()
        self.build_calls = 0
        self.retrieve_calls = 0
        self.built_articles: tuple[Mapping[str, object], ...] = ()

    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
        self.build_calls += 1
        self.built_articles = tuple(articles)
        return {
            "status": "synthetic_global_index_built",
            "corpus_count": len(articles),
            "receipt_sha256": runner.stable_hash(list(articles)),
        }

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch:
        self.retrieve_calls += 1
        self.retrieve_started.set()
        if self.agent_started is not None:
            assert self.agent_started.wait(timeout=5)
        sizes = [min(8, len(queries) - i) for i in range(0, len(queries), 8)]
        return RetrievalBatch(
            indices=tuple((0, 1, 2, 3, 4) for _query in queries),
            receipt={
                "block": block,
                "batch_sizes": sizes,
                "query_count": len(queries),
                "receipt_sha256": runner.stable_hash(list(queries)),
            },
        )


class GuardedFuture(Future[Any]):
    def __init__(self, owner: "EagerExecutor") -> None:
        super().__init__()
        self.owner = owner

    def result(self, timeout: float | None = None) -> Any:
        assert self.owner.submit_count == self.owner.expected_count
        if self.owner.event_log is not None:
            self.owner.event_log.append("agent_result")
        return super().result(timeout=timeout)


class EagerExecutor:
    expected_count = 0
    started_event: threading.Event | None = None
    event_log: list[str] | None = None
    last: "EagerExecutor | None" = None

    def __init__(
        self,
        *,
        max_workers: int,
        initializer: Any,
        initargs: tuple[Any, ...],
    ) -> None:
        assert 1 <= max_workers <= 32
        self.expected_count = type(self).expected_count
        self.submit_count = 0
        self.max_workers = max_workers
        initializer(*initargs)
        type(self).last = self

    def __enter__(self) -> "EagerExecutor":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def submit(self, fn: Any, /, *args: Any) -> Future[Any]:
        self.submit_count += 1
        if self.event_log is not None:
            self.event_log.append("agent_submit")
        if type(self).started_event is not None:
            type(self).started_event.set()
        future = GuardedFuture(self)
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class LoggedHippoFuture:
    def __init__(self, delegate: Future[Any], event_log: list[str]) -> None:
        self.delegate = delegate
        self.event_log = event_log

    def result(self, timeout: float | None = None) -> Any:
        self.event_log.append("hippo_result")
        return self.delegate.result(timeout=timeout)


class LoggedHippoExecutor:
    event_log: list[str] = []

    def __init__(self, *, max_workers: int) -> None:
        assert max_workers == 1
        self.delegate = ActualThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self) -> "LoggedHippoExecutor":
        self.delegate.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.delegate.__exit__(exc_type, exc, traceback)

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> LoggedHippoFuture:
        self.event_log.append("hippo_submit")
        return LoggedHippoFuture(self.delegate.submit(fn, *args, **kwargs), self.event_log)


def _self_hash(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: runner.stable_hash(body)}


def _corpus_view(count: int = 10) -> dict[str, Any]:
    articles = [
        {
            "article_id": index,
            "title": f"HoVer synthetic document {index}",
            "body": f"Acme synthetic evidence document {index}.",
        }
        for index in range(count)
    ]
    return _self_hash(
        {
            "schema": runner.DEFAULT_ACQUISITION_ADAPTER.corpus_view_schema,
            "version": runner.ACQUISITION_VERSION,
            "article_count": len(articles),
            "origin_or_gold_membership_included": False,
            "articles": articles,
        },
        "corpus_view_sha256",
    )


def _block_view(block: str, count: int = 6) -> dict[str, Any]:
    items = [
        {
            "schema": runner.DEFAULT_ACQUISITION_ADAPTER.view_item_schema,
            "block": block,
            "ordinal": ordinal,
            "claim": f"Acme makes synthetic claim {ordinal} for {block}",
        }
        for ordinal in range(count)
    ]
    return _self_hash(
        {
            "schema": runner.DEFAULT_ACQUISITION_ADAPTER.block_view_schema,
            "version": runner.ACQUISITION_VERSION,
            "block": block,
            "item_count": len(items),
            "late_utility_fields_included": False,
            "items": items,
        },
        "block_view_sha256",
    )


def _labels(stage: runner.StageExecution) -> dict[str, Any]:
    rows = []
    gold_by_stratum = {
        "2_hop": [5, 6],
        "3_hop": [5, 6, 7],
        "4_hop": [5, 6, 7, 8],
    }
    for ordinal, view_item in enumerate(stage.view["items"]):
        stratum = runner.HOP_STRATA[ordinal % 3]
        rows.append(
            {
                "schema": runner.DEFAULT_ACQUISITION_ADAPTER.label_item_schema,
                "block": stage.block,
                "ordinal": ordinal,
                "view_sha256": runner.stable_hash(view_item),
                "identity_commitment_sha256": hashlib.sha256(
                    f"identity-{stage.block}-{ordinal}".encode()
                ).hexdigest(),
                "source_record_commitment_sha256": hashlib.sha256(
                    f"source-{stage.block}-{ordinal}".encode()
                ).hexdigest(),
                "hop_stratum": stratum,
                "gold_article_ids": gold_by_stratum[stratum],
            }
        )
    return _self_hash(
        {
            "schema": runner.DEFAULT_ACQUISITION_ADAPTER.block_label_schema,
            "version": runner.ACQUISITION_VERSION,
            "block": stage.block,
            "item_count": len(rows),
            "source_or_verdict_payload_included": False,
            "items": rows,
        },
        "block_labels_sha256",
    )


@pytest.fixture
def runtime_bundle() -> tuple[FakeEncoder, FakeNER, FakeHippo, runner.PreparedCorpus]:
    encoder = FakeEncoder()
    ner = FakeNER()
    hippo = FakeHippo()
    prepared = runner.prepare_offline_corpus(
        corpus_view=_corpus_view(),
        encoder=encoder,
        ner=ner,
        hippo=hippo,
        ner_batch_size=4,
        formal_shape=False,
    )
    return encoder, ner, hippo, prepared


def _execute(
    block: str,
    bundle: tuple[FakeEncoder, FakeNER, FakeHippo, runner.PreparedCorpus],
) -> runner.StageExecution:
    encoder, ner, hippo, prepared = bundle
    EagerExecutor.expected_count = 6
    EagerExecutor.started_event = None
    return runner.execute_gold_free_stage(
        block=block,
        view=_block_view(block),
        prepared=prepared,
        encoder=encoder,
        ner=ner,
        hippo=hippo,
        ner_batch_size=4,
        local_worker_cap=32,
        formal_shape=False,
        executor_factory=EagerExecutor,
    )


def _controlled_stage(stage: runner.StageExecution) -> runner.StageExecution:
    outputs = {
        "P0_IND_SUM": (5, 6, 0, 1, 2),
        "P1_IND_MAXIMIN": (1, 2, 3, 4, 5),
        "P2_ENTITY_BRIDGE": (5, 6, 7, 8, 9),
        "P3_TOPIC_BRIDGE": (2, 3, 4, 5, 6),
        "P4_META_ASSIGN": (3, 4, 5, 6, 7),
        "P5_FAMILY_UNION": (4, 5, 6, 7, 8),
    }
    items = []
    for item in stage.items:
        traces = []
        for action_index, trace in enumerate(item.traces):
            e0 = (Fraction(0), 100 - action_index, 0)
            causal = CausalSignature(
                necessary_count=(4 if trace.action_id == "P2_ENTITY_BRIDGE" else 0),
                necessary_fraction=(
                    Fraction(1)
                    if trace.action_id == "P2_ENTITY_BRIDGE"
                    else Fraction(0)
                ),
                minimum_leave_one_out_loss=Fraction(0),
                minimum_replacement_loss=Fraction(0),
                path_connectivity=Fraction(0),
            )
            provisional = replace(
                trace,
                output_top5=outputs[trace.action_id],
                core=outputs[trace.action_id][:4],
                causal=causal,
                e0_key=e0,
                e1_key=(
                    causal.necessary_fraction,
                    causal.minimum_leave_one_out_loss,
                    causal.path_connectivity,
                    *e0,
                ),
                trace_sha256="0" * 64,
            )
            traces.append(
                replace(
                    provisional,
                    trace_sha256=recompute_action_trace_sha256(provisional),
                )
            )
        items.append(replace(item, traces=tuple(traces)))
    item_rows = tuple(items)
    return replace(
        stage,
        items=item_rows,
        execution_matrix_sha256=runner.stage_execution_matrix_sha256(
            item_rows,
            expected_embedding_index_sha256=stage.embedding_index_sha256,
        ),
    )


def test_corpus_view_is_title_body_only_and_missing_metadata_creates_no_false_edges(
    runtime_bundle: tuple[FakeEncoder, FakeNER, FakeHippo, runner.PreparedCorpus],
) -> None:
    encoder, ner, hippo, prepared = runtime_bundle
    assert encoder.calls and len(encoder.calls) == 1
    assert hippo.build_calls == 1
    assert all(set(row) == {"idx", "title", "body"} for row in hippo.built_articles)
    assert all("reserved_missing" not in repr(row) for row in hippo.built_articles)
    assert all(set(row) == {"kind", "title", "body"} for row in ner.inputs)
    assert len(prepared.graph.sources) == len(prepared.articles) == 10
    assert all(not neighbors for neighbors in prepared.graph.neighbors[SAME_SOURCE])
    assert all(
        article.published_ordinal is None
        and "reserved_missing_source" in article.normalized_source
        and "reserved_missing_category" in article.normalized_category
        for article in prepared.graph.articles
    )


def test_formal_shapes_are_fixed_609_and_48_36_30_30() -> None:
    adapter = runner.DEFAULT_ACQUISITION_ADAPTER
    assert len(adapter.validate_corpus_view(_corpus_view(609), formal_shape=True)) == 609
    with pytest.raises(runner.HoVerFormalRunnerError, match="corpus view"):
        adapter.validate_corpus_view(_corpus_view(608), formal_shape=True)
    for block, count in runner.BLOCK_COUNTS.items():
        assert len(
            adapter.validate_block_view(
                _block_view(block, count), block=block, formal_shape=True
            )
        ) == count
        with pytest.raises(runner.HoVerFormalRunnerError, match="block view"):
            adapter.validate_block_view(
                _block_view(block, count - 1), block=block, formal_shape=True
            )


def test_gold_free_signature_rejects_label_fields_and_agent_wave_is_eager_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {"gold", "hop", "verdict", "support", "label"}
    parameters = set(inspect.signature(runner.execute_gold_free_stage).parameters)
    assert not any(any(term in parameter.casefold() for term in forbidden) for parameter in parameters)

    encoder = FakeEncoder()
    ner = FakeNER()
    started = threading.Event()
    hippo = FakeHippo(agent_started=started)
    prepared = runner.prepare_offline_corpus(
        corpus_view=_corpus_view(),
        encoder=encoder,
        ner=ner,
        hippo=hippo,
        formal_shape=False,
    )
    EagerExecutor.expected_count = 6
    EagerExecutor.started_event = started
    event_log: list[str] = []
    monkeypatch.setattr(EagerExecutor, "event_log", event_log)
    monkeypatch.setattr(LoggedHippoExecutor, "event_log", event_log)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", LoggedHippoExecutor)
    original_raw_top5 = runner.raw_top5

    def logged_raw_top5(relevance_ints: Sequence[int]) -> tuple[int, ...]:
        output = original_raw_top5(relevance_ints)
        event_log.append("raw_materialized")
        return output

    monkeypatch.setattr(runner, "raw_top5", logged_raw_top5)
    stage = runner.execute_gold_free_stage(
        block="F_search",
        view=_block_view("F_search"),
        prepared=prepared,
        encoder=encoder,
        ner=ner,
        hippo=hippo,
        local_worker_cap=32,
        formal_shape=False,
        executor_factory=EagerExecutor,
    )
    assert hippo.retrieve_started.is_set()
    assert hippo.retrieve_calls == 1
    assert EagerExecutor.last is not None
    assert EagerExecutor.last.submit_count == 6
    assert all(len(item.traces) == len(ACTION_IDS) for item in stage.items)
    assert all(item.plan.normalized_sources == () for item in stage.items)
    assert all(item.raw_top5 == (0, 1, 2, 3, 4) for item in stage.items)
    assert event_log.count("hippo_submit") == 1
    assert event_log.count("raw_materialized") == 6
    assert event_log.count("agent_submit") == 6
    first_result = min(
        index for index, event in enumerate(event_log) if event.endswith("_result")
    )
    assert event_log.index("hippo_submit") < event_log.index("raw_materialized")
    assert max(
        index for index, event in enumerate(event_log) if event == "raw_materialized"
    ) < min(index for index, event in enumerate(event_log) if event == "agent_submit")
    assert max(
        index for index, event in enumerate(event_log) if event == "agent_submit"
    ) < first_result

    bad = _block_view("F_search")
    del bad["block_view_sha256"]
    bad["items"][0]["verdict"] = "SUPPORTED"
    bad = _self_hash(bad, "block_view_sha256")
    with pytest.raises(runner.HoVerFormalRunnerError, match="view item"):
        runner.DEFAULT_ACQUISITION_ADAPTER.validate_block_view(
            bad, block="F_search", formal_shape=False
        )
    with pytest.raises(runner.HoVerFormalRunnerError, match="1..32"):
        runner.execute_agent_actions_eager(
            graph=prepared.graph,
            plans=[stage.items[0].plan],
            relevance_vectors=[stage.items[0].query_feature.dense_relevance_ints],
            local_worker_cap=33,
            executor_factory=EagerExecutor,
        )


def test_late_scoring_accepts_gold_2_3_4_and_enforces_fixed_id_bounds(
    runtime_bundle: tuple[FakeEncoder, FakeNER, FakeHippo, runner.PreparedCorpus],
) -> None:
    stage = _execute("A_form", runtime_bundle)
    labels = _labels(stage)
    joined = runner.DEFAULT_ACQUISITION_ADAPTER.join_late_labels(stage, labels)
    assert [len(row.gold_article_ids) for row in joined] == [2, 3, 4, 2, 3, 4]
    report = runner.descriptive_stage_scores(stage=stage, labels=labels)
    assert report["item_count"] == 6
    assert report["exact_hop_stratum_counts"] == {
        "2_hop": 2,
        "3_hop": 2,
        "4_hop": 2,
    }

    boundary = _labels(stage)
    del boundary["block_labels_sha256"]
    boundary["items"][0]["gold_article_ids"] = [5, 608]
    boundary = _self_hash(boundary, "block_labels_sha256")
    assert runner.DEFAULT_ACQUISITION_ADAPTER.join_late_labels(stage, boundary)[
        0
    ].gold_article_ids == (5, 608)

    invalid = _labels(stage)
    del invalid["block_labels_sha256"]
    invalid["items"][0]["gold_article_ids"] = [5, 609]
    invalid = _self_hash(invalid, "block_labels_sha256")
    with pytest.raises(runner.HoVerFormalRunnerError, match="row drifted"):
        runner.DEFAULT_ACQUISITION_ADAPTER.join_late_labels(stage, invalid)


def test_label_free_policy_selection_a_hold_primary_and_l5_scoring(
    runtime_bundle: tuple[FakeEncoder, FakeNER, FakeHippo, runner.PreparedCorpus],
) -> None:
    f_stage = _controlled_stage(_execute("F_search", runtime_bundle))
    e0, e1, identifiable = runner.select_f_policies(f_stage=f_stage)
    assert identifiable is True
    assert e0.action_id == "P0_IND_SUM"
    assert e1.action_id == "P2_ENTITY_BRIDGE"

    a_hold = _controlled_stage(_execute("A_hold", runtime_bundle))
    decision = runner.decide_a_hold_promotion(
        stage=a_hold,
        labels=_labels(a_hold),
        f_stage=f_stage,
        e0_policy=e0,
        e1_policy=e1,
    )
    assert decision.promoted is True
    assert decision.primary_passed is True
    assert decision.promotion_delta_total > 0
    assert decision.e0_minus_hippo_delta_total > 0
    assert dict(decision.e0_minus_hippo_stratum_deltas).keys() == set(
        runner.HOP_STRATA
    )

    m_stage = _controlled_stage(_execute("M_search", runtime_bundle))
    assessment = runner.assess_m_search(
        stage=m_stage,
        labels=_labels(m_stage),
        f_stage=f_stage,
        e0_policy=e0,
        e1_policy=e1,
    )
    assert assessment.l5_passed is True
    assert assessment.l5_delta_total > 0
    assert assessment.e1_minus_hippo_delta_total > 0
