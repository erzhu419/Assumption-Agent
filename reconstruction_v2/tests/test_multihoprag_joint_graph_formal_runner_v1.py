from __future__ import annotations

from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import multihoprag_direct_acquisition_v1 as acquisition
from assumption_agent.benchmarks import multihoprag_joint_graph_formal_runner_v1 as runner
from assumption_agent.benchmarks.multihoprag_joint_graph_formal_runner_v1 import (
    MultiHopRAGFormalRunnerError,
    FormalRuntimeConfig,
    StageExecution,
    assess_m_search,
    build_canonical_stage_records,
    build_stage_runtime_binding,
    decide_a_hold_promotion,
    default_formal_runtime_config,
    descriptive_a_form,
    execute_gold_free_stage,
    execute_agent_actions_eager,
    make_result_receipt,
    prepare_offline_corpus,
    preflight_formal_runtime_config,
    select_f_policies,
    stable_hash,
    stage_execution_matrix_sha256,
    write_json_exclusive,
)
from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ACTION_IDS,
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
            "asset_manifest_path": "/synthetic/fixed/asset.json",
            "asset_sha256": identity["asset_sha256"],
            "embedding_dimension": identity["embedding_dimension"],
            "maximum_sequence_length": identity["maximum_sequence_length"],
            "model_root": "/synthetic/fixed/model",
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

    def extract_inputs(
        self, values: Sequence[Mapping[str, object]]
    ) -> tuple[tuple[EntitySpan, ...], ...]:
        self.batch_sizes.append(len(values))
        return tuple(
            (EntitySpan(entity_type="ORG", start=0, end=4, text="Acme"),)
            for _value in values
        )


class FakeHippo:
    def __init__(self) -> None:
        self.build_calls = 0
        self.retrieval_blocks: list[str] = []

    def build(self, articles: Sequence[Mapping[str, object]]) -> Mapping[str, Any]:
        self.build_calls += 1
        return {
            "status": "synthetic_global_index_built",
            "corpus_count": len(articles),
            "receipt_sha256": stable_hash(list(articles)),
        }

    def retrieve(self, *, block: str, queries: Sequence[str]) -> RetrievalBatch:
        self.retrieval_blocks.append(block)
        batch_sizes = [
            min(8, len(queries) - start) for start in range(0, len(queries), 8)
        ]
        return RetrievalBatch(
            indices=tuple((0, 1, 2, 3, 4) for _query in queries),
            receipt={
                "block": block,
                "batch_sizes": batch_sizes,
                "query_count": len(queries),
                "receipt_sha256": stable_hash(list(queries)),
            },
        )


class GuardedFuture(Future[Any]):
    def __init__(self, owner: "EagerExecutor") -> None:
        super().__init__()
        self.owner = owner

    def result(self, timeout: float | None = None) -> Any:
        assert self.owner.submit_count == self.owner.expected_count
        return super().result(timeout=timeout)


class EagerExecutor:
    expected_count = 0
    last: "EagerExecutor | None" = None

    def __init__(
        self,
        *,
        max_workers: int,
        initializer: Any,
        initargs: tuple[Any, ...],
    ) -> None:
        assert 1 <= max_workers <= 64
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
        future = GuardedFuture(self)
        try:
            future.set_result(fn(*args))
        except BaseException as exc:  # pragma: no cover - exercises propagation
            future.set_exception(exc)
        return future


def _self_hash(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(body)
    payload[field] = stable_hash(body)
    return payload


def _complete_trace_matrix_sha256(stage: StageExecution) -> str:
    return stable_hash(
        [
            [trace.trace_sha256 for trace in item.traces]
            for item in stage.items
        ]
    )


def _a_form_policy_freeze(
    stage: StageExecution,
    *,
    archive_file_sha256: str = "4" * 64,
    archive_semantic_sha256: str = "5" * 64,
    action_seal_sha256: str = "6" * 64,
) -> dict[str, Any]:
    observations = stage.observations()
    e0 = runner.select_global_policy(
        evaluator_id=runner.E0_ID, observations=observations
    )
    e1 = runner.select_global_policy(
        evaluator_id=runner.E1_ID, observations=observations
    )
    return _self_hash(
        {
            "schema": acquisition.A_FORM_POLICY_FREEZE_SCHEMA,
            "version": "v1",
            "status": "A_form_prelabel_descriptive_policies_frozen",
            "acquisition_sha256": "1" * 64,
            "a_form_view_file_sha256": "2" * 64,
            "a_form_view_semantic_sha256": stage.view_sha256,
            "a_form_item_count": len(stage.items),
            "a_form_output_archive_file_sha256": archive_file_sha256,
            "a_form_output_archive_semantic_sha256": (
                archive_semantic_sha256
            ),
            "a_form_action_seal_sha256": action_seal_sha256,
            "complete_a_form_trace_matrix_receipt_sha256": (
                _complete_trace_matrix_sha256(stage)
            ),
            "e0_action_id": e0.action_id,
            "e0_policy_sha256": e0.selection_sha256,
            "e1_action_id": e1.action_id,
            "e1_policy_sha256": e1.selection_sha256,
            "policies_identifiable": runner.policies_identifiable(
                e0, e1, observations
            ),
            "selection_purpose": (
                "prelabel_descriptive_only_not_F_policy"
            ),
            "A_form_gold_opened_before_policy_freeze": False,
            "created_with_O_EXCL": True,
            "same_stage_replay_or_policy_reselection_authorized": False,
        },
        "a_form_policy_freeze_sha256",
    )


def _corpus_view() -> dict[str, Any]:
    articles = [
        {
            "article_id": index,
            "title": f"Acme article {index}",
            "author": f"Author {index}",
            "source": f"Source {index % 4}",
            "published_at": f"202{index % 4}-01-0{index % 8 + 1}",
            "category": f"Category {index % 3}",
            "body": f"Acme synthetic body {index}",
        }
        for index in range(8)
    ]
    return _self_hash(
        {
            "schema": acquisition.CORPUS_VIEW_SCHEMA,
            "version": acquisition.VERSION,
            "article_count": len(articles),
            "corpus_locator_fields_included": False,
            "articles": articles,
        },
        "corpus_view_sha256",
    )


def _block_view(block: str, count: int = 6) -> dict[str, Any]:
    items = [
        {
            "schema": acquisition.VIEW_ITEM_SCHEMA,
            "block": block,
            "ordinal": ordinal,
            "query": f"Compare Acme synthetic item {ordinal} in {block}",
        }
        for ordinal in range(count)
    ]
    return _self_hash(
        {
            "schema": acquisition.BLOCK_VIEW_SCHEMA,
            "version": acquisition.VERSION,
            "block": block,
            "item_count": len(items),
            "late_label_fields_included": False,
            "items": items,
        },
        "block_view_sha256",
    )


def _labels(stage: StageExecution) -> dict[str, Any]:
    items = []
    for ordinal, view_item in enumerate(stage.view["items"]):
        items.append(
            {
                "schema": acquisition.LABEL_ITEM_SCHEMA,
                "block": stage.block,
                "ordinal": ordinal,
                "view_sha256": stable_hash(view_item),
                "identity_commitment_sha256": hashlib.sha256(
                    f"identity-{stage.block}-{ordinal}".encode()
                ).hexdigest(),
                "source_record_commitment_sha256": hashlib.sha256(
                    f"source-record-{stage.block}-{ordinal}".encode()
                ).hexdigest(),
                "question_type": acquisition.FAMILIES[ordinal % 3],
                "answer": "synthetic answer",
                "gold_article_ids": [5, 6],
            }
        )
    return _self_hash(
        {
            "schema": acquisition.BLOCK_LABEL_SCHEMA,
            "version": acquisition.VERSION,
            "block": stage.block,
            "item_count": len(items),
            "source_locator_payload_included": False,
            "items": items,
        },
        "block_labels_sha256",
    )


def _runner_marker() -> dict[str, Any]:
    return _self_hash(
        {
            "schema": runner.RUNNER_MARKER_SCHEMA,
            "version": runner.VERSION,
            "phase": "formal_A_form_F_A_hold_M_one_shot",
            "bindings": {
                "acquisition_sha256": "1" * 64,
                "implementation_freeze_sha256": "2" * 64,
            },
            "replay_retry_resample_replacement_authorized": False,
        },
        "marker_sha256",
    )


def _authoritative_m_assessment(
    assessment: Any, *, promotion_sha256: str, action_seal_sha256: str
) -> dict[str, Any]:
    return {
        "status": "M_search_authoritatively_assessed",
        "l5_delta_total": assessment.l5_delta_total,
        "l5_signflip_p": assessment.l5_signflip_p,
        "l5_passed": assessment.l5_passed,
        "agent_minus_hippo_delta_total": assessment.agent_minus_hippo_delta_total,
        "agent_minus_hippo_signflip_p": assessment.agent_minus_hippo_signflip_p,
        "agent_minus_hippo_family_deltas": dict(
            assessment.agent_minus_hippo_family_deltas
        ),
        "cross_family_agent_over_hippo_passed": (
            assessment.cross_family_agent_over_hippo_passed
        ),
        "agent_minus_raw_delta_total": assessment.agent_minus_raw_delta_total,
        "agent_minus_raw_signflip_p": assessment.agent_minus_raw_signflip_p,
        "agent_complete_count": assessment.agent_complete_count,
        "raw_complete_count": assessment.raw_complete_count,
        "agent_minus_raw_complete_delta": assessment.agent_minus_raw_complete_delta,
        "raw_complete_advantage_overcome": assessment.raw_complete_advantage_overcome,
        "promotion_sha256": promotion_sha256,
        "m_search_action_seal_sha256": action_seal_sha256,
        "m_search_output_archive_file_sha256": "5" * 64,
        "m_search_output_archive_semantic_sha256": "6" * 64,
    }


def _controlled_stage(stage: StageExecution) -> StageExecution:
    controlled_items = []
    outputs = {
        "P0_IND_SUM": (0, 1, 2, 3, 4),
        "P1_IND_MAXIMIN": (1, 2, 3, 4, 5),
        "P2_ENTITY_BRIDGE": (5, 6, 0, 1, 2),
        "P3_TOPIC_BRIDGE": (2, 3, 4, 5, 6),
        "P4_META_ASSIGN": (3, 4, 5, 6, 7),
        "P5_FAMILY_UNION": (4, 5, 6, 7, 0),
    }
    for item in stage.items:
        traces = []
        for action_index, trace in enumerate(item.traces):
            e0 = (Fraction(0), 100 - action_index, 0)
            causal = CausalSignature(
                necessary_count=(4 if trace.action_id == "P2_ENTITY_BRIDGE" else 0),
                necessary_fraction=(
                    Fraction(1) if trace.action_id == "P2_ENTITY_BRIDGE" else Fraction(0)
                ),
                minimum_leave_one_out_loss=Fraction(0),
                minimum_replacement_loss=Fraction(0),
                path_connectivity=Fraction(0),
            )
            output = outputs[trace.action_id]
            provisional = replace(
                trace,
                output_top5=output,
                core=output[:4],
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
        controlled_items.append(replace(item, traces=tuple(traces)))
    items = tuple(controlled_items)
    return replace(
        stage,
        items=items,
        execution_matrix_sha256=stage_execution_matrix_sha256(
            items,
            expected_embedding_index_sha256=stage.embedding_index_sha256,
        ),
    )


def _formal_scan_receipt_stage(stage: StageExecution) -> StageExecution:
    items = []
    for item in stage.items:
        dense = (runner.INTEGER_SCALE,) * acquisition.CORPUS_RECORD_COUNT
        relevance_sha256 = stable_hash(
            {"integer_scale": runner.INTEGER_SCALE, "values": list(dense)}
        )
        traces = []
        for trace in item.traces:
            provisional = replace(
                trace,
                ordered_pair_scan_count=609 * 608,
                extension_scan_count=(609 - 2) + (609 - 3),
                relevance_sha256=relevance_sha256,
                trace_sha256="0" * 64,
            )
            traces.append(
                replace(
                    provisional,
                    trace_sha256=recompute_action_trace_sha256(provisional),
                )
            )
        items.append(
            replace(
                item,
                query_feature=replace(
                    item.query_feature, dense_relevance_ints=dense
                ),
                traces=tuple(traces),
            )
        )
    item_rows = tuple(items)
    return replace(
        stage,
        items=item_rows,
        execution_matrix_sha256=stage_execution_matrix_sha256(
            item_rows,
            expected_embedding_index_sha256=stage.embedding_index_sha256,
        ),
    )


@pytest.fixture
def runtime_bundle():
    encoder = FakeEncoder()
    ner = FakeNER()
    hippo = FakeHippo()
    prepared = prepare_offline_corpus(
        corpus_view=_corpus_view(),
        encoder=encoder,
        ner=ner,
        hippo=hippo,
        ner_batch_size=3,
        formal_shape=False,
    )
    return encoder, ner, hippo, prepared


def _execute(
    block: str,
    runtime_bundle: tuple[FakeEncoder, FakeNER, FakeHippo, Any],
) -> StageExecution:
    encoder, ner, hippo, prepared = runtime_bundle
    EagerExecutor.expected_count = 6
    return _controlled_stage(
        execute_gold_free_stage(
            block=block,
            view=_block_view(block),
            prepared=prepared,
            encoder=encoder,
            ner=ner,
            hippo=hippo,
            ner_batch_size=3,
            local_worker_cap=64,
            formal_shape=False,
            executor_factory=EagerExecutor,
        )
    )


def test_corpus_compiles_once_and_query_stage_is_maximally_batched(runtime_bundle) -> None:
    encoder, ner, hippo, prepared = runtime_bundle
    assert len(encoder.calls) == 1
    assert ner.batch_sizes == [3, 3, 2]
    assert hippo.build_calls == 1
    assert len(prepared.graph.articles) == 8
    assert prepared.embedding_index.article_count == 8

    stage = _execute("A_form", runtime_bundle)
    assert len(encoder.calls) == 2
    assert len(encoder.calls[-1]) == 6 + 3
    assert ner.batch_sizes == [3, 3, 2, 3, 3]
    assert hippo.retrieval_blocks == ["A_form"]
    assert EagerExecutor.last is not None
    assert EagerExecutor.last.submit_count == 6
    assert EagerExecutor.last.max_workers == 6
    assert all(item.raw_top5 == (0, 1, 2, 3, 4) for item in stage.items)
    assert all(len(item.traces) == 6 for item in stage.items)
    runtime_binding = build_stage_runtime_binding(
        prepared=prepared, stage=stage
    )
    assert set(runtime_binding) == set(acquisition.STAGE_RUNTIME_BINDING_KEYS)


def test_canonical_records_hold_full_traces_and_labels_are_descriptive_only(runtime_bundle) -> None:
    _encoder, _ner, _hippo, _prepared = runtime_bundle
    stage = _execute("A_form", runtime_bundle)
    records = build_canonical_stage_records(_formal_scan_receipt_stage(stage))
    assert len(records) == len(stage.items)
    trace = records[0]["agent_action_traces"][0]
    assert trace["trace_sha256"] == stage.items[0].traces[0].trace_sha256 or len(
        trace["trace_sha256"]
    ) == 64
    assert set(trace["trace"]) >= {
        "core",
        "core_quality",
        "coverage",
        "causal",
        "e0",
        "e1",
        "ordered_pair_scan_count",
        "extension_scan_count",
        "graph_sha256",
        "plan_sha256",
        "query_sha256",
        "relevance_sha256",
    }
    policy_freeze = _a_form_policy_freeze(stage)
    report = descriptive_a_form(
        stage=stage,
        policy_freeze=policy_freeze,
        labels=_labels(stage),
    )
    assert report["status"] == "descriptive_only_no_policy_or_threshold_change"
    assert report["outcome_used_to_change_action_evaluator_or_threshold"] is False

    forged = _labels(stage)
    forged["items"][0]["gold_article_ids"] = [0, 1]
    with pytest.raises(MultiHopRAGFormalRunnerError, match="self-hash"):
        descriptive_a_form(
            stage=stage, policy_freeze=policy_freeze, labels=forged
        )

    self_hash_tamper = dict(policy_freeze)
    self_hash_tamper["a_form_output_archive_file_sha256"] = "9" * 64
    with pytest.raises(MultiHopRAGFormalRunnerError, match="self-hash"):
        descriptive_a_form(
            stage=stage,
            policy_freeze=self_hash_tamper,
            labels=_labels(stage),
        )

    policy_tamper_body = dict(policy_freeze)
    policy_tamper_body.pop("a_form_policy_freeze_sha256")
    policy_tamper_body["e0_policy_sha256"] = "9" * 64
    policy_tamper = _self_hash(
        policy_tamper_body, "a_form_policy_freeze_sha256"
    )
    with pytest.raises(
        MultiHopRAGFormalRunnerError, match="sealed action evidence"
    ):
        descriptive_a_form(
            stage=stage,
            policy_freeze=policy_tamper,
            labels=_labels(stage),
        )

    rebound_body = dict(policy_freeze)
    rebound_body.pop("a_form_policy_freeze_sha256")
    rebound_body["a_form_output_archive_file_sha256"] = "9" * 64
    rebound_freeze = _self_hash(
        rebound_body, "a_form_policy_freeze_sha256"
    )
    rebound_report = descriptive_a_form(
        stage=stage,
        policy_freeze=rebound_freeze,
        labels=_labels(stage),
    )
    assert rebound_report["a_form_policy_freeze_sha256"] != report[
        "a_form_policy_freeze_sha256"
    ]
    assert rebound_report["descriptive_sha256"] != report[
        "descriptive_sha256"
    ]


def test_f_freeze_a_hold_promotion_and_m_exact_boundaries(runtime_bundle) -> None:
    _encoder, _ner, _hippo, prepared = runtime_bundle
    f_stage = _execute("F_search", runtime_bundle)
    e0, e1, identifiable = select_f_policies(f_stage=f_stage)
    assert identifiable
    assert e0.action_id == "P0_IND_SUM"
    assert e1.action_id == "P2_ENTITY_BRIDGE"

    a_hold = _execute("A_hold", runtime_bundle)
    decision = decide_a_hold_promotion(
        stage=a_hold,
        labels=_labels(a_hold),
        f_stage=f_stage,
        e0_policy=e0,
        e1_policy=e1,
    )
    assert decision.promoted
    assert decision.delta_total == Fraction(12)
    assert decision.signflip_p == Fraction(1, 64)
    assert all(value == Fraction(4) for _family, value in decision.family_delta_totals)
    promotion = {"promotion_sha256": "3" * 64}

    m_stage = _execute("M_search", runtime_bundle)
    assessment = assess_m_search(
        stage=m_stage,
        labels=_labels(m_stage),
        f_stage=f_stage,
        e0_policy=e0,
        e1_policy=e1,
    )
    assert assessment.l5_passed
    assert assessment.cross_family_agent_over_hippo_passed
    assert assessment.raw_complete_advantage_overcome
    assert assessment.agent_minus_hippo_delta_total == Fraction(12)
    seal_sha256 = "4" * 64
    result = make_result_receipt(
        assessment=_authoritative_m_assessment(
            assessment,
            promotion_sha256=promotion["promotion_sha256"],
            action_seal_sha256=seal_sha256,
        ),
        promotion=promotion,
        m_action_seal={"action_seal_sha256": seal_sha256},
        runner_marker=_runner_marker(),
    )
    assert result["external_network_calls"] == 0
    assert result["online_evaluator_calls"] == 0
    with pytest.raises(TypeError):
        make_result_receipt(  # type: ignore[call-arg]
            assessment={},
            promotion=promotion,
            m_action_seal={"action_seal_sha256": seal_sha256},
        )
    with pytest.raises(MultiHopRAGFormalRunnerError, match="marker"):
        make_result_receipt(
            assessment=_authoritative_m_assessment(
                assessment,
                promotion_sha256=promotion["promotion_sha256"],
                action_seal_sha256=seal_sha256,
            ),
            promotion=promotion,
            m_action_seal={"action_seal_sha256": seal_sha256},
            runner_marker={"marker_sha256": "0" * 64},
        )


def test_real_spawn_process_pool_initializer_and_pickling_smoke(runtime_bundle) -> None:
    _encoder, _ner, _hippo, prepared = runtime_bundle
    stage = _execute("F_search", runtime_bundle)
    traces = execute_agent_actions_eager(
        graph=prepared.graph,
        plans=[item.plan for item in stage.items[:2]],
        relevance_vectors=[
            item.query_feature.dense_relevance_ints for item in stage.items[:2]
        ],
        local_worker_cap=2,
    )
    assert len(traces) == 2
    assert all(tuple(trace.action_id for trace in row) == ACTION_IDS for row in traces)
    assert all(
        trace.trace_sha256 == recompute_action_trace_sha256(trace)
        for row in traces
        for trace in row
    )


def test_default_config_routes_committed_v3_attestation_before_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path.cwd().resolve()
    config = default_formal_runtime_config(project)
    assert config.hippo_llm_model == Path.home() / ".hr5/models/smollm2-135m-instruct"
    assert config.hippo_attestation_receipt == (
        project / "manifests/musique_official_hipporag_runtime_attestation_v3.json"
    )
    observed: dict[str, Any] = {}

    def fake_verify(**kwargs: Any) -> Mapping[str, Any]:
        observed.update(kwargs)
        return {"status": "synthetic_preflight_pass"}

    monkeypatch.setattr(runner, "verify_formal_runtime_attestation_v3", fake_verify)
    assert preflight_formal_runtime_config(config)["status"] == "synthetic_preflight_pass"
    assert observed["attestation_receipt_path"] == config.hippo_attestation_receipt
    wrong = replace(
        config,
        hippo_attestation_receipt=(
            project
            / "artifacts/multihoprag_official_hipporag_global_qualification_v1/stage/runtime.attestation_receipt.json"
        ),
    )
    with pytest.raises(MultiHopRAGFormalRunnerError, match="committed v3"):
        preflight_formal_runtime_config(wrong)
    substituted_minilm = replace(
        config,
        minilm_asset_manifest=project / "artifacts/substituted-minilm.json",
    )
    with pytest.raises(MultiHopRAGFormalRunnerError, match="exact default"):
        preflight_formal_runtime_config(substituted_minilm)


def test_formal_controller_rejects_runtime_override_before_marker() -> None:
    root = Path("/tmp/multihoprag-formal-override-rejected")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(mode=0o700)
    config = default_formal_runtime_config(root)
    with pytest.raises(TypeError):
        runner.run_formal_lifecycle(  # type: ignore[call-arg]
            config, encoder_factory=lambda _config: FakeEncoder()
        )
    assert not (root / runner.RUNNER_MARKER_RELATIVE).exists()
    shutil.rmtree(root)


def test_formal_controller_rejects_loaded_checkout_mismatch_before_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp/multihoprag-formal-cross-checkout-rejected")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(mode=0o700)
    typed_relative = acquisition.FIXED_FREEZE_ROLE_PATHS["typed_core"]
    typed_placeholder = root / typed_relative
    typed_placeholder.parent.mkdir(parents=True)
    typed_placeholder.write_text("# checkout B placeholder\n", encoding="ascii")
    config = default_formal_runtime_config(root)
    monkeypatch.setattr(
        runner, "preflight_formal_runtime_config", lambda _config: {}
    )
    monkeypatch.setattr(
        acquisition,
        "load_committed_acquisition_receipt",
        lambda project: ({"acquisition_sha256": "1" * 64}, {}),
    )
    monkeypatch.setattr(
        acquisition,
        "verify_committed_implementation_freeze",
        lambda project: {
            "implementation_freeze_sha256": "2" * 64,
            "all_bindings_byte_match_committed_HEAD": True,
            "required_role_count": len(acquisition.REQUIRED_FREEZE_ROLES),
        },
    )
    with pytest.raises(
        MultiHopRAGFormalRunnerError, match="outside the frozen project"
    ):
        runner.run_formal_lifecycle(config)
    assert not (root / runner.RUNNER_MARKER_RELATIVE).exists()
    shutil.rmtree(root)


def _controller_scenario(
    *,
    monkeypatch: pytest.MonkeyPatch,
    runtime_bundle: tuple[FakeEncoder, FakeNER, FakeHippo, Any],
    name: str,
    force_nonidentifiable: bool = False,
    force_nonpromotion: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    root = Path("/tmp") / f"multihoprag-controller-{name}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(mode=0o700)
    (root / runner._SYNTHETIC_SENTINEL).write_text(
        runner._SYNTHETIC_SENTINEL_CONTENT, encoding="ascii"
    )
    config = replace(
        default_formal_runtime_config(root),
        hippo_stage_root=root / "hippo-stage",
        hippo_work_root=root / "hippo-work",
    )
    stages = {
        block: _execute(block, runtime_bundle)
        for block in ("A_form", "F_search", "A_hold", "M_search")
    }
    labels = {block: _labels(stage) for block, stage in stages.items()}
    events: list[str] = []
    archive_bindings: dict[str, dict[str, str]] = {}
    archives: dict[str, dict[str, Any]] = {}
    f_freezes: list[dict[str, Any]] = []

    monkeypatch.setattr(
        acquisition,
        "load_committed_acquisition_receipt",
        lambda project: ({"acquisition_sha256": "1" * 64}, {}),
    )
    monkeypatch.setattr(
        acquisition,
        "verify_committed_implementation_freeze",
        lambda project: {"implementation_freeze_sha256": "2" * 64},
    )
    monkeypatch.setattr(acquisition, "load_corpus_view", lambda **kwargs: _corpus_view())

    def load_view(*, project: Path, expected_block: str) -> Mapping[str, Any]:
        events.append(f"view:{expected_block}")
        return _block_view(expected_block)

    def load_labels(*, project: Path, expected_block: str) -> Mapping[str, Any]:
        events.append(f"labels:{expected_block}")
        return labels[expected_block]

    monkeypatch.setattr(acquisition, "load_block_view", load_view)
    monkeypatch.setattr(acquisition, "load_block_labels", load_labels)

    def persist_stage(*, project: Path, prepared: Any, stage: StageExecution):
        events.append(f"archive:{stage.block}")
        binding = {
            "file_sha256": hashlib.sha256(
                f"archive-file-{stage.block}".encode()
            ).hexdigest(),
            "semantic_sha256": hashlib.sha256(
                f"archive-semantic-{stage.block}".encode()
            ).hexdigest(),
        }
        archive_bindings[stage.block] = binding
        runtime_binding = {
            field: hashlib.sha256(
                f"{stage.block}-{field}".encode()
            ).hexdigest()
            for field in acquisition.STAGE_RUNTIME_BINDING_KEYS
        }
        archive = {
            "agent_complete_six_action_trace_matrix_sha256": (
                _complete_trace_matrix_sha256(stage)
            ),
            "stage_runtime_binding": runtime_binding,
        }
        archives[stage.block] = archive
        return archive, binding, runtime_binding

    monkeypatch.setattr(runner, "persist_canonical_stage_archive", persist_stage)

    def create_seal(*, project: Path, block: str) -> Mapping[str, Any]:
        events.append(f"seal:{block}")
        return {
            "action_seal_sha256": hashlib.sha256(
                f"seal-{block}".encode()
            ).hexdigest(),
            "stage_output_archive_file_sha256": archive_bindings[block][
                "file_sha256"
            ],
            "stage_output_archive_semantic_sha256": archive_bindings[block][
                "semantic_sha256"
            ],
        }

    monkeypatch.setattr(acquisition, "create_action_seal_once", create_seal)

    def create_a_form_freeze(**kwargs: Any) -> Mapping[str, Any]:
        events.append("freeze:A_form")
        binding = archive_bindings["A_form"]
        return _a_form_policy_freeze(
            stages["A_form"],
            archive_file_sha256=binding["file_sha256"],
            archive_semantic_sha256=binding["semantic_sha256"],
            action_seal_sha256=hashlib.sha256(b"seal-A_form").hexdigest(),
        )

    monkeypatch.setattr(
        acquisition,
        "create_a_form_policy_freeze_once",
        create_a_form_freeze,
    )

    f_e0, f_e1, _f_identifiable = select_f_policies(
        f_stage=stages["F_search"]
    )

    def recompute_f_selections(_archive: Mapping[str, Any]):
        assert _archive is archives["F_search"]
        return f_e0, f_e1, not force_nonidentifiable

    monkeypatch.setattr(
        acquisition,
        "_recompute_f_search_policy_selections",
        recompute_f_selections,
    )

    def create_freeze(**kwargs: Any) -> Mapping[str, Any]:
        events.append("freeze:F_search")
        if force_nonidentifiable:
            raise acquisition.MultiHopRAGAcquisitionError(
                "F policies are not identifiable"
            )
        payload = _self_hash({
            "f_search_output_archive_file_sha256": archive_bindings[
                "F_search"
            ]["file_sha256"],
            "f_search_output_archive_semantic_sha256": archive_bindings[
                "F_search"
            ]["semantic_sha256"],
            "e0_action_id": f_e0.action_id,
            "e0_policy_sha256": f_e0.selection_sha256,
            "e1_action_id": f_e1.action_id,
            "e1_policy_sha256": f_e1.selection_sha256,
        }, "policy_freeze_sha256")
        f_freezes.append(payload)
        return payload

    monkeypatch.setattr(
        acquisition, "create_f_search_policy_freeze_once", create_freeze
    )

    def assess_a_hold(*, project: Path) -> Mapping[str, Any]:
        events.append("assess:A_hold")
        return {
            "status": "valid_nonpromotion" if force_nonpromotion else "promote",
            "challenger_promoted": not force_nonpromotion,
            "family_balanced_delta_total": (
                Fraction(0) if force_nonpromotion else Fraction(12)
            ),
            "one_sided_magnitude_signflip_p": (
                Fraction(1) if force_nonpromotion else Fraction(1, 64)
            ),
            "e0_policy_sha256": f_e0.selection_sha256,
            "e1_policy_sha256": f_e1.selection_sha256,
            "f_search_policy_freeze_sha256": f_freezes[0][
                "policy_freeze_sha256"
            ],
            "a_hold_action_seal_sha256": hashlib.sha256(
                b"seal-A_hold"
            ).hexdigest(),
            "a_hold_output_archive_file_sha256": archive_bindings[
                "A_hold"
            ]["file_sha256"],
            "a_hold_output_archive_semantic_sha256": archive_bindings[
                "A_hold"
            ]["semantic_sha256"],
        }

    monkeypatch.setattr(acquisition, "assess_a_hold_promotion", assess_a_hold)
    monkeypatch.setattr(
        acquisition,
        "create_a_hold_promotion_once",
        lambda **kwargs: events.append("promotion:A_hold")
        or {"promotion_sha256": "3" * 64},
    )

    def assess_m(*, project: Path) -> Mapping[str, Any]:
        events.append("assess:M_search")
        seal_sha = hashlib.sha256(b"seal-M_search").hexdigest()
        return {
            "status": "M_search_authoritatively_assessed",
            "l5_delta_total": Fraction(12),
            "l5_signflip_p": Fraction(1, 64),
            "l5_passed": True,
            "agent_minus_hippo_delta_total": Fraction(12),
            "agent_minus_hippo_signflip_p": Fraction(1, 64),
            "agent_minus_hippo_family_deltas": {
                family: Fraction(4) for family in acquisition.FAMILIES
            },
            "cross_family_agent_over_hippo_passed": True,
            "agent_minus_raw_delta_total": Fraction(12),
            "agent_minus_raw_signflip_p": Fraction(1, 64),
            "agent_complete_count": 6,
            "raw_complete_count": 0,
            "agent_minus_raw_complete_delta": 6,
            "raw_complete_advantage_overcome": True,
            "promotion_sha256": "3" * 64,
            "m_search_action_seal_sha256": seal_sha,
            "m_search_output_archive_file_sha256": archive_bindings[
                "M_search"
            ]["file_sha256"],
            "m_search_output_archive_semantic_sha256": archive_bindings[
                "M_search"
            ]["semantic_sha256"],
        }

    monkeypatch.setattr(acquisition, "assess_m_search", assess_m)

    def execute_stage(**kwargs: Any) -> StageExecution:
        block = kwargs["block"]
        source = stages[block]
        return replace(
            source,
            view=kwargs["view"],
            view_sha256=kwargs["view"]["block_view_sha256"],
            formal_shape=False,
        )

    result = runner._run_synthetic_lifecycle_core(
        config,
        encoder_factory=lambda _config: runtime_bundle[0],
        ner_factory=lambda _config: nullcontext(runtime_bundle[1]),
        hippo_factory=lambda _config: runtime_bundle[2],
        prepare_corpus_fn=lambda **kwargs: runtime_bundle[3],
        execute_stage_fn=execute_stage,
        executor_factory=EagerExecutor,
    )
    shutil.rmtree(root, ignore_errors=True)
    return result, events


def test_controller_guards_and_canonical_promotion_order(
    monkeypatch: pytest.MonkeyPatch, runtime_bundle
) -> None:
    with monkeypatch.context() as context:
        result, events = _controller_scenario(
            monkeypatch=context,
            runtime_bundle=runtime_bundle,
            name="nonidentifiable",
            force_nonidentifiable=True,
        )
        assert "nonidentifiable" in result["status"]
        assert not any(event.startswith("view:A_hold") for event in events)
        assert not any(event.startswith("view:M_search") for event in events)

    with monkeypatch.context() as context:
        result, events = _controller_scenario(
            monkeypatch=context,
            runtime_bundle=runtime_bundle,
            name="nonpromotion",
            force_nonpromotion=True,
        )
        assert "nonpromotion" in result["status"]
        assert "view:A_hold" in events
        assert "assess:A_hold" in events
        assert "promotion:A_hold" not in events
        assert not any(event.startswith("view:M_search") for event in events)

    with monkeypatch.context() as context:
        result, events = _controller_scenario(
            monkeypatch=context,
            runtime_bundle=runtime_bundle,
            name="promotion",
        )
        assert result["status"] == "formal_M_search_complete"
        assert events == [
            "view:A_form",
            "archive:A_form",
            "seal:A_form",
            "freeze:A_form",
            "labels:A_form",
            "view:F_search",
            "archive:F_search",
            "freeze:F_search",
            "view:A_hold",
            "archive:A_hold",
            "seal:A_hold",
            "assess:A_hold",
            "promotion:A_hold",
            "view:M_search",
            "archive:M_search",
            "seal:M_search",
            "assess:M_search",
        ]


def test_exclusive_writer_and_tampered_stage_fail_closed(tmp_path: Path, runtime_bundle) -> None:
    # The configured pytest temp root is DrvFs, whose synthetic mode bits cannot
    # satisfy the formal Linux permission postflight; use the local Linux fs.
    path = Path("/tmp") / f"multihoprag-formal-{hashlib.sha256(str(tmp_path).encode()).hexdigest()}.json"
    path.unlink(missing_ok=True)
    payload = {"schema": "synthetic", "value": 1}
    write_json_exclusive(path, payload, mode=0o600)
    with pytest.raises(MultiHopRAGFormalRunnerError, match="already exists"):
        write_json_exclusive(path, payload, mode=0o600)
    path.unlink()
    unsafe_root = Path("/tmp") / f"multihoprag-symlink-{hashlib.sha256(str(tmp_path).encode()).hexdigest()}"
    shutil.rmtree(unsafe_root, ignore_errors=True)
    (unsafe_root / "real").mkdir(parents=True)
    (unsafe_root / "linked").symlink_to(unsafe_root / "real", target_is_directory=True)
    with pytest.raises(MultiHopRAGFormalRunnerError, match="ancestor"):
        write_json_exclusive(
            unsafe_root / "linked/forbidden.json", payload, mode=0o600
        )
    assert not (unsafe_root / "real/forbidden.json").exists()
    shutil.rmtree(unsafe_root)

    _encoder, _ner, _hippo, _prepared = runtime_bundle
    stage = _execute("A_form", runtime_bundle)
    tampered_item = replace(stage.items[0], raw_top5=(1, 0, 2, 3, 4))
    tampered = replace(stage, items=(tampered_item, *stage.items[1:]))
    with pytest.raises(MultiHopRAGFormalRunnerError, match="RAW output"):
        stage_execution_matrix_sha256(
            tampered.items,
            expected_embedding_index_sha256=tampered.embedding_index_sha256,
        )
    first_trace = stage.items[0].traces[0]
    wrong_input = replace(
        first_trace,
        relevance_sha256="f" * 64,
        trace_sha256="0" * 64,
    )
    wrong_input = replace(
        wrong_input,
        trace_sha256=recompute_action_trace_sha256(wrong_input),
    )
    wrong_item = replace(
        stage.items[0],
        traces=(wrong_input, *stage.items[0].traces[1:]),
    )
    with pytest.raises(MultiHopRAGFormalRunnerError, match="exact query/plan/graph/relevance"):
        stage_execution_matrix_sha256(
            (wrong_item, *stage.items[1:]),
            expected_embedding_index_sha256=stage.embedding_index_sha256,
        )
