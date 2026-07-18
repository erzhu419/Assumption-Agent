from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import replication_runtime.morehopqa_official_hipporag_v1.adapter as adapter
from replication_runtime.morehopqa_official_hipporag_v1.contract import (
    CORPUS_INPUT_SCHEMA,
    MAX_QUERY_BATCH,
    MIN_CORPUS_SIZE,
    MoreHopQAOfficialHippoRAGError,
    QUERY_INPUT_SCHEMA,
    RETRIEVAL_OUTPUT_SCHEMA,
    canonical_json_bytes,
    make_build_receipt,
    make_retrieval_receipt,
    parse_retrieval_output,
    serialize_article,
    serialize_corpus,
    snapshot_index_tree,
    validate_build_receipt,
    validate_corpus,
)
from replication_runtime.morehopqa_official_hipporag_v1.worker import (
    _build_official_core,
    build_index_with_core,
    retrieve_batches_with_core,
)


PROJECT = Path(__file__).parents[1]
PACKAGE = PROJECT / "replication_runtime/morehopqa_official_hipporag_v1"
TEST_CORPUS_SIZE = 17


def _articles(count: int = TEST_CORPUS_SIZE) -> list[dict[str, object]]:
    return [
        {
            "body": f"Synthetic offline body {idx}.",
            "idx": idx,
            "title": f"Synthetic title {idx}",
        }
        for idx in range(count)
    ]


@dataclass
class _Store:
    documents: list[str] | None = None
    index_calls: int = 0
    index_root: Path | None = None


class _FakeCore:
    def __init__(self, store: _Store, *, build_mode: bool) -> None:
        self.store = store
        self.build_mode = build_mode
        self.retrieve_calls: list[tuple[tuple[str, ...], int]] = []

    def index(self, documents: list[str]) -> None:
        if not self.build_mode:
            raise AssertionError("reopen core must never index")
        self.store.index_calls += 1
        self.store.documents = list(documents)
        if self.store.index_root is not None:
            self.store.index_root.mkdir(parents=True, exist_ok=True)
            (self.store.index_root / "synthetic.index").write_bytes(
                canonical_json_bytes(documents)
            )

    def retrieve(self, queries: list[str], *, num_to_retrieve: int) -> list[object]:
        assert self.store.documents is not None
        self.retrieve_calls.append((tuple(queries), num_to_retrieve))
        documents = list(reversed(self.store.documents))
        scores = [1.0] * len(documents)
        return [
            SimpleNamespace(docs=list(documents), doc_scores=list(scores))
            for _query in queries
        ]


def test_dynamic_corpus_contract_and_title_blank_body_serialization() -> None:
    rows = validate_corpus(_articles())
    assert len(rows) == TEST_CORPUS_SIZE
    assert serialize_article(rows[7]) == (
        "Synthetic title 7\n\nSynthetic offline body 7."
    )
    assert len(set(serialize_corpus(rows))) == TEST_CORPUS_SIZE

    noncontiguous = _articles()
    noncontiguous[8]["idx"] = 9
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="contiguous"):
        validate_corpus(noncontiguous)

    duplicate = _articles()
    duplicate[1]["title"] = duplicate[0]["title"]
    duplicate[1]["body"] = duplicate[0]["body"]
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="unique"):
        validate_corpus(duplicate)

    leaked = _articles()
    leaked[0]["evidence"] = True
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="only idx"):
        validate_corpus(leaked)

    for count in (MIN_CORPUS_SIZE, 31, 73):
        assert len(validate_corpus(_articles(count))) == count
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="size"):
        validate_corpus(_articles(MIN_CORPUS_SIZE - 1))


def test_build_once_then_batch_retrieve_caps_each_call_at_eight(
    tmp_path: Path,
) -> None:
    articles = _articles()
    index_root = tmp_path / "index"
    store = _Store(index_root=index_root)
    build_core = _FakeCore(store, build_mode=True)
    build_receipt = build_index_with_core(
        core=build_core,
        articles=articles,
        index_root=index_root,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    assert store.index_calls == 1
    assert build_receipt["index_call_count"] == 1
    assert build_receipt["force_index_from_scratch"] is True

    reopen_core = _FakeCore(store, build_mode=False)
    queries = [f"Synthetic query {idx}?" for idx in range(17)]
    indices, batch_sizes = retrieve_batches_with_core(
        core=reopen_core,
        articles=articles,
        queries=queries,
    )
    assert indices == ((0, 1, 2, 3, 4),) * len(queries)
    assert [len(call[0]) for call in reopen_core.retrieve_calls] == [8, 8, 1]
    assert all(call[1] == TEST_CORPUS_SIZE for call in reopen_core.retrieve_calls)
    assert max(len(call[0]) for call in reopen_core.retrieve_calls) == MAX_QUERY_BATCH
    assert store.index_calls == 1
    assert batch_sizes == (8, 8, 1)
    snapshot = snapshot_index_tree(index_root)
    receipt = make_retrieval_receipt(
        documents=serialize_corpus(validate_corpus(articles)),
        queries=queries,
        indices=indices,
        batch_sizes=batch_sizes,
        build_receipt=build_receipt,
        index_snapshot_before=snapshot,
        index_snapshot_after=snapshot,
    )
    assert receipt["batch_sizes"] == [8, 8, 1]
    assert receipt["index_call_count"] == 0
    assert receipt["force_index_from_scratch"] is False


def test_fresh_core_and_reopened_core_are_synthetically_equivalent(
    tmp_path: Path,
) -> None:
    articles = _articles()
    queries = ["Synthetic alpha?", "Synthetic beta?"]
    index_root = tmp_path / "index"
    store = _Store(index_root=index_root)
    fresh_core = _FakeCore(store, build_mode=True)
    build_index_with_core(
        core=fresh_core,
        articles=articles,
        index_root=index_root,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    fresh_result, fresh_batches = retrieve_batches_with_core(
        core=fresh_core,
        articles=articles,
        queries=queries,
    )
    reopened_core = _FakeCore(store, build_mode=False)
    reopened_result, reopened_batches = retrieve_batches_with_core(
        core=reopened_core,
        articles=articles,
        queries=queries,
    )
    assert fresh_result == reopened_result
    assert fresh_batches == reopened_batches
    assert store.index_calls == 1


def test_official_core_factory_freezes_true_build_and_false_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[dict[str, Any]] = []

    class _Config:
        def __init__(self, **kwargs: Any) -> None:
            observed.append(dict(kwargs))

    class _Hippo:
        def __init__(self, *, global_config: object) -> None:
            self.global_config = global_config
            self.llm_model = SimpleNamespace(
                llm_config=SimpleNamespace(generate_params={})
            )

    hipporag = ModuleType("hipporag")
    hipporag.HippoRAG = _Hippo  # type: ignore[attr-defined]
    utils = ModuleType("hipporag.utils")
    config_utils = ModuleType("hipporag.utils.config_utils")
    config_utils.BaseConfig = _Config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hipporag", hipporag)
    monkeypatch.setitem(sys.modules, "hipporag.utils", utils)
    monkeypatch.setitem(sys.modules, "hipporag.utils.config_utils", config_utils)
    for force in (True, False):
        _build_official_core(
            save_dir=tmp_path / "index",
            llm_model=tmp_path / "llm",
            embedding_model=tmp_path / "embedding",
            force_index_from_scratch=force,
            corpus_count=TEST_CORPUS_SIZE,
        )
    assert [row["force_index_from_scratch"] for row in observed] == [True, False]
    assert all(row["retrieval_top_k"] == TEST_CORPUS_SIZE for row in observed)


def test_idx_receipt_output_is_exact_and_rejects_scores_or_documents(
    tmp_path: Path,
) -> None:
    documents = serialize_corpus(validate_corpus(_articles()))
    queries = ["Synthetic alpha?", "Synthetic beta?"]
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "index.bin").write_bytes(b"frozen synthetic index")
    snapshot = snapshot_index_tree(index_root)
    build_receipt = make_build_receipt(
        documents,
        index_snapshot=snapshot,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    indices = ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0))
    receipt = make_retrieval_receipt(
        documents=documents,
        queries=queries,
        indices=indices,
        batch_sizes=[2],
        build_receipt=build_receipt,
        index_snapshot_before=snapshot,
        index_snapshot_after=snapshot,
    )
    output = {
        "receipt": receipt,
        "retrieved_idx": [list(row) for row in indices],
        "schema": RETRIEVAL_OUTPUT_SCHEMA,
    }
    parsed = parse_retrieval_output(
        canonical_json_bytes(output),
        queries=queries,
        expected_build_receipt=build_receipt,
        expected_index_snapshot_after=snapshot,
    )
    assert parsed.indices == indices
    assert set(output) == {"receipt", "retrieved_idx", "schema"}

    output["scores"] = [[1.0] * 5] * 2
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="only indices"):
        parse_retrieval_output(
            canonical_json_bytes(output),
            queries=queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=snapshot,
        )


def test_query_hash_rejects_equal_length_replay(tmp_path: Path) -> None:
    documents = serialize_corpus(validate_corpus(_articles()))
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "index.bin").write_bytes(b"query-bound index")
    snapshot = snapshot_index_tree(index_root)
    build_receipt = make_build_receipt(
        documents,
        index_snapshot=snapshot,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    original_queries = ["Original alpha?", "Original beta?"]
    indices = ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0))
    receipt = make_retrieval_receipt(
        documents=documents,
        queries=original_queries,
        indices=indices,
        batch_sizes=[2],
        build_receipt=build_receipt,
        index_snapshot_before=snapshot,
        index_snapshot_after=snapshot,
    )
    output = {
        "receipt": receipt,
        "retrieved_idx": [list(row) for row in indices],
        "schema": RETRIEVAL_OUTPUT_SCHEMA,
    }
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="contract drifted"):
        parse_retrieval_output(
            canonical_json_bytes(output),
            queries=["Replacement one?", "Replacement two?"],
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=snapshot,
        )


def test_build_receipt_rejects_replaced_index_and_index_symlink(
    tmp_path: Path,
) -> None:
    documents = serialize_corpus(validate_corpus(_articles()))
    index_root = tmp_path / "index"
    index_root.mkdir()
    payload = index_root / "index.bin"
    payload.write_bytes(b"original index")
    original = snapshot_index_tree(index_root)
    receipt = make_build_receipt(
        documents,
        index_snapshot=original,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    payload.write_bytes(b"replaced index")
    replaced = snapshot_index_tree(index_root)
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="contract drifted"):
        validate_build_receipt(
            receipt,
            expected_corpus_sha256=receipt["corpus_sha256"],
            expected_corpus_count=TEST_CORPUS_SIZE,
            expected_index_snapshot=replaced,
            expected_runtime_attestation_receipt_sha256="a" * 64,
        )

    payload.unlink()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (index_root / "linked.bin").symlink_to(target)
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="symbolic link"):
        snapshot_index_tree(index_root)


def test_parser_independently_rejects_forged_clone_post_snapshot(
    tmp_path: Path,
) -> None:
    documents = serialize_corpus(validate_corpus(_articles()))
    queries = ["Synthetic query?"]
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "index.bin").write_bytes(b"real clone post")
    snapshot = snapshot_index_tree(index_root)
    build_receipt = make_build_receipt(
        documents,
        index_snapshot=snapshot,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    indices = ((0, 1, 2, 3, 4),)
    receipt = make_retrieval_receipt(
        documents=documents,
        queries=queries,
        indices=indices,
        batch_sizes=[1],
        build_receipt=build_receipt,
        index_snapshot_before=snapshot,
        index_snapshot_after=snapshot,
    )
    forged = dict(receipt)
    forged["index_post_tree_sha256"] = "f" * 64
    forged["index_changed_during_retrieve"] = True
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            forged, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output = {
        "receipt": forged,
        "retrieved_idx": [list(row) for row in indices],
        "schema": RETRIEVAL_OUTPUT_SCHEMA,
    }
    with pytest.raises(MoreHopQAOfficialHippoRAGError, match="contract drifted"):
        parse_retrieval_output(
            canonical_json_bytes(output),
            queries=queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=snapshot,
        )


def test_public_adapter_persists_one_stage_and_supports_repeat_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _Store()
    runtime_receipt = {
        "attestation_receipt_sha256": "a" * 64,
        "implementation_set_sha256": "b" * 64,
    }
    runtime = tmp_path / "runtime-python"
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    base = tmp_path / "manifests" / "binding.json"
    attestation = tmp_path / "manifests" / "attestation.json"

    monkeypatch.setattr(
        adapter,
        "_validated_runtime",
        lambda **_kwargs: (
            PROJECT,
            runtime,
            llm,
            embedding,
            base,
            dict(runtime_receipt),
        ),
    )

    def fake_launch(**kwargs: object) -> None:
        stage = kwargs["stage"]
        corpus_payload = json.loads(Path(kwargs["corpus_input"]).read_text())
        articles = corpus_payload["articles"]
        output_path = Path(kwargs["output_path"])
        index_root = Path(kwargs["index_root"])
        runtime_receipt_sha256 = str(
            kwargs["runtime_attestation_receipt_sha256"]
        )
        if stage == "build":
            store.index_root = index_root
            receipt = build_index_with_core(
                core=_FakeCore(store, build_mode=True),
                articles=articles,
                index_root=index_root,
                runtime_attestation_receipt_sha256=runtime_receipt_sha256,
            )
            output_path.write_bytes(canonical_json_bytes(receipt))
            return
        query_payload = json.loads(Path(kwargs["query_input"]).read_text())
        build_receipt = json.loads(Path(kwargs["build_receipt"]).read_text())
        before = snapshot_index_tree(index_root)
        indices, batch_sizes = retrieve_batches_with_core(
            core=_FakeCore(store, build_mode=False),
            articles=articles,
            queries=query_payload["queries"],
        )
        after = snapshot_index_tree(index_root)
        receipt = make_retrieval_receipt(
            documents=serialize_corpus(validate_corpus(articles)),
            queries=query_payload["queries"],
            indices=indices,
            batch_sizes=batch_sizes,
            build_receipt=build_receipt,
            index_snapshot_before=before,
            index_snapshot_after=after,
        )
        output_path.write_bytes(
            canonical_json_bytes(
                {
                    "receipt": receipt,
                    "retrieved_idx": [list(row) for row in indices],
                    "schema": RETRIEVAL_OUTPUT_SCHEMA,
                }
            )
        )

    monkeypatch.setattr(adapter, "_launch_worker", fake_launch)
    stage_root = tmp_path / "persistent-stage"
    build_receipt = adapter.build_morehopqa_official_hipporag_global_index_v1(
        articles=_articles(),
        runtime_python=runtime,
        local_llm_model=llm,
        local_embedding_model=embedding,
        base_binding_receipt_path=base,
        attestation_receipt_path=attestation,
        stage_root=stage_root,
    )
    assert build_receipt["index_call_count"] == 1
    assert stage_root.is_dir()
    for run in range(2):
        result = adapter.retrieve_morehopqa_official_hipporag_global_index_v1(
            queries=["Synthetic query?"],
            runtime_python=runtime,
            local_llm_model=llm,
            local_embedding_model=embedding,
            base_binding_receipt_path=base,
            attestation_receipt_path=attestation,
            stage_root=stage_root,
            work_root=tmp_path / f"query-work-{run}",
        )
        assert result.indices == ((0, 1, 2, 3, 4),)
        assert not (tmp_path / f"query-work-{run}").exists()
    assert store.index_calls == 1
    assert not (stage_root / adapter.QUERY_LOCK_FILENAME).exists()


def test_launcher_is_network_isolated_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_json_bytes(
                {
                    "corpus_count": TEST_CORPUS_SIZE,
                    "index_call_count": 1,
                    "stage": "build",
                    "status": "passed",
                }
            ),
            stderr=b"",
        )

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("home", "cache", "tmp"):
        (stage / name).mkdir()
    adapter._launch_worker(
        stage="build",
        project_root=PROJECT,
        runtime_python=Path(sys.executable),
        local_llm_model=tmp_path / "llm",
        local_embedding_model=tmp_path / "embedding",
        corpus_input=stage / "corpus.json",
        output_path=stage / "receipt.json",
        index_root=stage / "index",
        stage_root=stage,
        writable_root=stage,
        timeout_seconds=10,
        runtime_attestation_receipt_sha256="a" * 64,
        expected_corpus_count=TEST_CORPUS_SIZE,
    )
    command = observed["command"]
    environment = observed["environment"]
    assert isinstance(command, list) and "--unshare-net" in command
    assert isinstance(environment, dict)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_retrieve_mounts_persisted_stage_read_only_and_work_clone_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        observed.extend(command)
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_json_bytes(
                {
                    "batch_count": 1,
                    "index_call_count": 0,
                    "query_count": 1,
                    "stage": "retrieve",
                    "status": "passed",
                }
            ),
            stderr=b"",
        )

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    stage = tmp_path / "stage"
    work = tmp_path / "work"
    stage.mkdir()
    work.mkdir()
    for root in (stage, work):
        for name in ("home", "cache", "tmp"):
            (root / name).mkdir()
    adapter._launch_worker(
        stage="retrieve",
        project_root=PROJECT,
        runtime_python=Path(sys.executable),
        local_llm_model=tmp_path / "llm",
        local_embedding_model=tmp_path / "embedding",
        corpus_input=stage / "corpus.json",
        query_input=work / "queries.json",
        build_receipt=stage / "build.json",
        output_path=work / "output.json",
        index_root=work / "index-clone",
        stage_root=stage,
        writable_root=work,
        timeout_seconds=10,
        runtime_attestation_receipt_sha256="a" * 64,
        expected_corpus_count=TEST_CORPUS_SIZE,
    )
    stage_position = observed.index(str(stage))
    work_position = observed.index(str(work))
    assert observed[stage_position - 1] == "--ro-bind"
    assert observed[work_position - 1] == "--bind"


def test_worker_surface_has_no_answer_evidence_or_generation_channel() -> None:
    worker_source = (PACKAGE / "worker.py").read_text(encoding="utf-8")
    assert "rag_qa" not in worker_source
    assert "question_type" not in worker_source
    assert "gold_docs" not in worker_source
    assert "answer" not in worker_source
    assert "CORPUS_INPUT_SCHEMA" in worker_source
    assert "QUERY_INPUT_SCHEMA" in worker_source
