from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import synthetic_typed_graph_causal_grammar_v1 as grammar
from assumption_agent.benchmarks import synthetic_typed_graph_multiseed_runner_v2 as runner
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding


PRIVATE_MODE = 0o600
FIXED_OFFICIAL = (0, 1, 2, 3, 4)


def _linux_tmp(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp"))


def _semantic_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _write_private(path: Path, value: Mapping[str, object]) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")
    path.chmod(PRIVATE_MODE)


class RealBoundFakeEncoder:
    """Cheap deterministic fake that enforces the real frozen call bound."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) > minilm_binding.MAXIMUM_TEXTS_PER_CALL:
            raise minilm_binding.QasperMiniLMError(
                "text count is outside the frozen bound"
            )
        self.calls.append((len(texts), texts[0], texts[-1]))
        base = np.ones((minilm_binding.EMBEDDING_DIMENSION,), dtype=np.float32)
        base /= np.linalg.norm(base)
        return np.broadcast_to(base, (len(texts), len(base))).copy()


class FakeOfficialRuntime:
    def __init__(self, *, postflight_drift: bool = False) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.postflights = 0
        self.postflight_drift = postflight_drift

    @property
    def safe_binding(self) -> Mapping[str, object]:
        return {"runtime": "offline_public_test_fake", "revision": 1}

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]:
        assert question
        assert len(paragraphs) == runner.NODE_COUNT
        assert all(
            paragraph.get("title") == "synthetic_typed_graph_causal_v1"
            for paragraph in paragraphs
        )
        assert work_root.name
        with self._lock:
            self.calls += 1
        time.sleep(0.001)
        return FIXED_OFFICIAL

    def fresh_reverify(self) -> Mapping[str, object]:
        with self._lock:
            self.postflights += 1
        if self.postflight_drift:
            return {"runtime": "offline_public_test_fake", "revision": 2}
        return dict(self.safe_binding)


@pytest.fixture(scope="module")
def pack_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for seed_index in range(runner.SEED_COUNT):
        seed = hashlib.sha256(f"v2-public-unit-seed-{seed_index}".encode("ascii")).digest()
        items = grammar.generate_block(seed, runner.BLOCK)
        assert len(items) == runner.ITEMS_PER_SEED
        for seed_ordinal, item in enumerate(items):
            ordinal = seed_index * runner.ITEMS_PER_SEED + seed_ordinal
            nodes = [
                {
                    "span_i": node.span_i,
                    "start": node.start,
                    "end": node.end,
                    "identity_text": node.identity_text,
                }
                for node in item.nodes
            ]
            edges = [
                {
                    "edge_family": edge.edge_family,
                    "left_span_i": edge.left_span_i,
                    "right_span_i": edge.right_span_i,
                }
                for edge in item.designated_edges
            ]
            action_body = {
                "schema": runner.ACTION_ITEM_SCHEMA,
                "global_ordinal": ordinal,
                "seed_index": seed_index,
                "seed_ordinal": seed_ordinal,
                "question": item.question,
                "context": item.context,
                "nodes": nodes,
                "designated_edges": edges,
            }
            action = {
                **action_body,
                "action_item_sha256": _semantic_hash(action_body),
            }
            action_rows.append(action)
            label_body = {
                "schema": runner.LABEL_ITEM_SCHEMA,
                "global_ordinal": ordinal,
                "seed_index": seed_index,
                "seed_ordinal": seed_ordinal,
                "action_item_sha256": action["action_item_sha256"],
                "gold_node_indices": list(item.gold_node_indices),
                "family_id": item.family_id,
                "family_role": item.family_role,
                "polarity": item.polarity,
                "edge_family": item.edge_family,
            }
            label_rows.append(
                {**label_body, "label_item_sha256": _semantic_hash(label_body)}
            )
            # Deliberately include every prohibited label/outcome field.  The
            # public diagnostic projection must ignore all of them.
            public_rows.append(
                {
                    "schema": "synthetic_typed_graph_causal_grammar_v1_compiled_item",
                    "global_ordinal": ordinal,
                    "seed_index": seed_index,
                    "seed_ordinal": seed_ordinal,
                    "question": item.question,
                    "context": item.context,
                    "nodes": [
                        {**node, "latent_role": "must_not_cross_projection"}
                        for node in nodes
                    ],
                    "designated_edges": edges,
                    "label_free_commitment_sha256": _semantic_hash(
                        ["public-label-free", ordinal]
                    ),
                    "gold_node_indices": list(item.gold_node_indices),
                    "family_id": item.family_id,
                    "family_role": item.family_role,
                    "polarity": item.polarity,
                    "edge_family": item.edge_family,
                    "negative_kind": item.negative_kind,
                    "endpoint_permutation": item.endpoint_permutation,
                    "matching_signature_sha256": item.matching_signature_sha256,
                }
            )
    return action_rows, label_rows, public_rows


@pytest.fixture(scope="module")
def private_packs(
    tmp_path_factory: pytest.TempPathFactory,
    pack_rows: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
) -> tuple[runner.ActionPack, runner.LabelPack]:
    action_rows, label_rows, _public_rows = pack_rows
    root = _linux_tmp("synthetic-multiseed-v2-packs-")
    action_body = {
        "schema": runner.ACTION_PACK_SCHEMA,
        "version": runner.DESIGN_VERSION,
        "block": runner.BLOCK,
        "seed_count": runner.SEED_COUNT,
        "item_count_per_seed": runner.ITEMS_PER_SEED,
        "total_item_count": runner.TOTAL_ITEMS,
        "labels_included": False,
        "items": action_rows,
    }
    label_body = {
        "schema": runner.LABEL_PACK_SCHEMA,
        "version": runner.DESIGN_VERSION,
        "block": runner.BLOCK,
        "seed_count": runner.SEED_COUNT,
        "item_count_per_seed": runner.ITEMS_PER_SEED,
        "total_item_count": runner.TOTAL_ITEMS,
        "items": label_rows,
    }
    action_json = {**action_body, "pack_sha256": _semantic_hash(action_body)}
    label_json = {**label_body, "pack_sha256": _semantic_hash(label_body)}
    action_path = root / "action_pack.json"
    label_path = root / "label_pack.json"
    _write_private(action_path, action_json)
    _write_private(label_path, label_json)
    return runner.load_action_pack(action_path), runner.load_label_pack(label_path)


@pytest.fixture(scope="module")
def prepared(
    private_packs: tuple[runner.ActionPack, runner.LabelPack],
) -> tuple[runner.ActionPack, runner.LabelPack, tuple[runner.LocalTensor, ...], runner.MiniLMChunkAudit, RealBoundFakeEncoder]:
    actions, labels = private_packs
    encoder = RealBoundFakeEncoder()
    tensors, audit = runner.precompute_local_tensors(actions, encoder)
    return actions, labels, tensors, audit, encoder


def test_public_projection_is_exactly_label_free(
    pack_rows: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    _actions, _labels, public_rows = pack_rows
    observed_gets: set[str] = set()

    class AuditedMapping(dict[str, Any]):
        def get(self, key: str, default: object = None) -> Any:
            observed_gets.add(key)
            return super().get(key, default)

    audited = [AuditedMapping(row) for row in public_rows]
    pack, source_set = runner.project_public_v1_label_free_rows(
        audited, source_file_sha256="a" * 64
    )
    assert len(pack.rows) == runner.TOTAL_ITEMS
    assert source_set == _semantic_hash(
        [row["label_free_commitment_sha256"] for row in public_rows]
    )
    expected_accesses = {
        "global_ordinal",
        "seed_index",
        "seed_ordinal",
        "question",
        "context",
        "nodes",
        "designated_edges",
        "label_free_commitment_sha256",
    }
    assert observed_gets == expected_accesses
    forbidden = {
        "gold_node_indices",
        "family_id",
        "family_role",
        "polarity",
        "negative_kind",
        "endpoint_permutation",
        "matching_signature_sha256",
    }
    assert observed_gets.isdisjoint(forbidden)
    assert all(not forbidden.intersection(vars(row)) for row in pack.rows)
    assert {
        paragraph["title"]
        for row in pack.rows
        for paragraph in row.paragraphs
    } == {"synthetic_typed_graph_causal_v1"}


def test_real_bound_encoder_receives_exact_two_contiguous_8448_text_calls(
    prepared: tuple[runner.ActionPack, runner.LabelPack, tuple[runner.LocalTensor, ...], runner.MiniLMChunkAudit, RealBoundFakeEncoder],
) -> None:
    actions, _labels, tensors, audit, encoder = prepared
    assert minilm_binding.MAXIMUM_TEXTS_PER_CALL == 16_384
    assert [call[0] for call in encoder.calls] == [8448, 8448]
    assert len(encoder.calls) == 2
    assert encoder.calls[0][1] == actions.rows[0].question
    assert encoder.calls[1][1] == actions.rows[256].question
    assert encoder.calls[0][2] == runner.core.embedding_text(
        actions.rows[255].nodes[-1].identity_text
    )
    assert encoder.calls[1][2] == runner.core.embedding_text(
        actions.rows[511].nodes[-1].identity_text
    )
    assert len(tensors) == 512
    assert audit.chunk_schedule_sha256 == runner.CHUNK_SCHEDULE_SHA256
    assert audit.observed_input_row_counts == (8448, 8448)
    assert audit.observed_output_row_counts == (8448, 8448)


def test_three_arm_wave_submits_all_futures_before_any_join_and_opens_labels_after_seal(
    prepared: tuple[runner.ActionPack, runner.LabelPack, tuple[runner.LocalTensor, ...], runner.MiniLMChunkAudit, RealBoundFakeEncoder],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _linux_tmp("synthetic-multiseed-v2-wave-")
    actions, labels, tensors, audit, _encoder = prepared
    real_executor = ThreadPoolExecutor
    submit_events: list[int] = []

    class RecordingExecutor(real_executor):
        def submit(self, fn: object, /, *args: object, **kwargs: object) -> Future[object]:
            submit_events.append(int(self._max_workers))
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(
        runner,
        "precompute_local_tensors",
        lambda _pack, _encoder: (tensors, audit),
    )
    monkeypatch.setattr(runner, "ThreadPoolExecutor", RecordingExecutor)
    real_result = Future.result
    main_thread = threading.get_ident()
    main_join_counts: list[int] = []
    worker_join_counts: list[int] = []

    def audited_result(self: Future[object], *args: object, **kwargs: object) -> object:
        if threading.get_ident() == main_thread:
            main_join_counts.append(len(submit_events))
        else:
            worker_join_counts.append(len(submit_events))
        return real_result(self, *args, **kwargs)

    monkeypatch.setattr(Future, "result", audited_result)
    runtime = FakeOfficialRuntime()
    seal_path = tmp_path / "formal-action-seal.json"
    label_open_observations: list[tuple[int, int, bool]] = []

    def load_labels() -> runner.LabelPack:
        label_open_observations.append(
            (runtime.calls, runtime.postflights, seal_path.is_file())
        )
        return labels

    outcome = runner.run_multiseed_replication(
        actions,
        label_loader=load_labels,
        encoder=RealBoundFakeEncoder(),
        runtime=runtime,
        work_root=tmp_path / "official-work",
        action_seal_path=seal_path,
    )
    assert Counter(submit_events) == {8: 512, 64: 1024}
    assert main_join_counts and min(main_join_counts) == 1536
    assert len(worker_join_counts) == 512
    assert min(worker_join_counts) == 1536
    assert runtime.calls == 512
    assert runtime.postflights == 1
    assert label_open_observations == [(512, 1, True)]
    assert seal_path.stat().st_mode & 0o777 == PRIVATE_MODE
    seal = json.loads(seal_path.read_text(encoding="ascii"))
    assert seal["labels_opened_before_action_seal"] is False
    assert seal["labels_opened_before_seal"] is False
    assert seal["observed_encoder_output_row_counts"] == [8448, 8448]
    assert outcome.chunk_audit == audit
    assert set(outcome.aggregates) == set(runner.ARM_IDS)
    assert set(outcome.cluster_differences) == {
        "Agent_R1_minus_official_HippoRAG",
        "Agent_R1_minus_RAW",
    }


def test_diagnostic_seal_and_receipt_persist_no_action_rows_indices_or_scores(
    prepared: tuple[runner.ActionPack, runner.LabelPack, tuple[runner.LocalTensor, ...], runner.MiniLMChunkAudit, RealBoundFakeEncoder],
    tmp_path: Path,
) -> None:
    tmp_path = _linux_tmp("synthetic-multiseed-v2-diagnostic-")
    actions, _labels, tensors, audit, _encoder = prepared
    item_actions = tuple(
        runner.ItemActions(
            item.global_ordinal,
            item.action_item_sha256,
            tensor.raw_top5,
            FIXED_OFFICIAL,
            (4, 3, 2, 1, 0),
            "b" * 64,
            tensor.tensor_sha256,
        )
        for item, tensor in zip(actions.rows, tensors)
    )
    binding = _semantic_hash({"runtime": "offline_public_test_fake", "revision": 1})
    wave = runner.ActionWaveOutcome(item_actions, binding, binding, 8, 64)
    seal_path = tmp_path / "diagnostic-seal.json"
    seal_sha, seal_file_sha, table_sha = runner._persist_action_seal(
        path=seal_path,
        pack=actions,
        wave=wave,
        chunk_audit=audit,
        purpose="public_integration_diagnostic",
    )
    seal = json.loads(seal_path.read_text(encoding="ascii"))
    assert set(seal) == {
        "schema",
        "total_action_count",
        "arm_terminal_counts",
        "ordered_action_commitment_set_sha256",
        "official_peak_concurrency",
        "local_peak_concurrency",
        "postflight_receipt_sha256",
        "action_rows_or_ranked_indices_persisted",
    }
    assert seal["action_rows_or_ranked_indices_persisted"] is False
    assert seal_sha == _semantic_hash(seal)
    assert seal_file_sha == hashlib.sha256(seal_path.read_bytes()).hexdigest()
    assert table_sha == seal["ordered_action_commitment_set_sha256"]

    binding_rows = [
        {
            "relative_path": relative,
            "file_sha256": "c" * 64,
            "git_blob_sha1": "d" * 40,
        }
        for relative in (
            runner.ACQUISITION_MODULE_RELATIVE_PATH.as_posix(),
            runner.RUNNER_MODULE_RELATIVE_PATH.as_posix(),
            runner.ACQUISITION_TEST_RELATIVE_PATH.as_posix(),
            runner.RUNNER_TEST_RELATIVE_PATH.as_posix(),
        )
    ]
    source = {
        "file_sha256": runner.V1_PUBLICATION_FILE_SHA256,
        "reproducibility_sha256": runner.V1_PUBLICATION_REPRODUCIBILITY_SHA256,
        "generated_item_commitment_set_sha256": (
            runner.V1_GENERATED_ITEM_COMMITMENT_SET_SHA256
        ),
        "projected_action_pack_sha256": actions.pack_sha256,
        "projected_action_item_commitment_set_sha256": (
            actions.item_commitment_set_sha256
        ),
        "source_label_free_commitment_set_sha256": "e" * 64,
    }
    marker = {"marker_sha256": "f" * 64}
    receipt = runner._diagnostic_success_receipt(
        actual_head="1" * 40,
        bindings=binding_rows,
        source_binding=source,
        chunk_audit=audit,
        wave=wave,
        action_table_sha256=table_sha,
        seal_sha256=seal_sha,
        seal_file_sha256=seal_file_sha,
        marker=marker,
        marker_file_sha256="2" * 64,
    )
    body = dict(receipt)
    assert body.pop("diagnostic_sha256") == _semantic_hash(body)
    serialized = json.dumps(receipt, sort_keys=True).casefold()
    for forbidden in (
        "action_rows",
        "ranked_indices",
        "gold_node_indices",
        "support_hit_count",
        "complete_count",
        "total_u",
        "cluster_differences",
    ):
        assert re.search(
            rf"(?<![a-z0-9_]){re.escape(forbidden)}(?![a-z0-9_])", serialized
        ) is None
    assert receipt["labels_opened"] is False
    assert receipt["scores_computed"] is False
    assert receipt["estimands_computed"] is False
    assert receipt["claims_made"] is False
    assert receipt["network_calls"] == 0
    assert receipt["action_identity_or_quality_used_for_decision"] is False


def test_postflight_failure_never_opens_labels_or_creates_seal(
    prepared: tuple[runner.ActionPack, runner.LabelPack, tuple[runner.LocalTensor, ...], runner.MiniLMChunkAudit, RealBoundFakeEncoder],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _linux_tmp("synthetic-multiseed-v2-postflight-")
    actions, _labels, tensors, audit, _encoder = prepared
    monkeypatch.setattr(
        runner,
        "precompute_local_tensors",
        lambda _pack, _encoder: (tensors, audit),
    )
    label_opens = 0

    def forbidden_labels() -> runner.LabelPack:
        nonlocal label_opens
        label_opens += 1
        raise AssertionError("labels must remain closed")

    seal_path = tmp_path / "must-not-exist.json"
    with pytest.raises(
        runner.SyntheticTypedGraphMultiseedRunnerV2Error, match="postflight|binding"
    ):
        runner.run_multiseed_replication(
            actions,
            label_loader=forbidden_labels,
            encoder=RealBoundFakeEncoder(),
            runtime=FakeOfficialRuntime(postflight_drift=True),
            work_root=tmp_path / "failed-work",
            action_seal_path=seal_path,
        )
    assert label_opens == 0
    assert not seal_path.exists()
