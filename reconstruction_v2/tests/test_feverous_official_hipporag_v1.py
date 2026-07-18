from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import replication_runtime.feverous_official_hipporag_v1.adapter as adapter
import replication_runtime.feverous_official_hipporag_v1.contract as contract
import replication_runtime.feverous_official_hipporag_v1.worker as worker
from replication_runtime.feverous_official_hipporag_v1.contract import (
    BENCHMARK,
    CORPUS_INPUT_SCHEMA,
    CORPUS_SIZE,
    FORMAL_QUERY_COUNT_UPPER_BOUND,
    MAX_CORPUS_SIZE,
    MAX_QUERY_BATCH,
    MAX_QUERY_COUNT,
    MIN_CORPUS_SIZE,
    OFFICIAL_HIPPORAG_COMMIT,
    QUERY_INPUT_SCHEMA,
    RETRIEVAL_OUTPUT_SCHEMA,
    FeverousOfficialHippoRAGError,
    canonical_json_bytes,
    corpus_text_multiplicity,
    make_build_receipt,
    make_retrieval_receipt,
    parse_retrieval_output,
    serialize_unit,
    serialize_corpus,
    snapshot_index_tree,
    stable_top_five_from_official_result,
    validate_build_receipt,
    validate_corpus,
    validate_queries,
)
from replication_runtime.feverous_official_hipporag_v1.worker import (
    _build_official_core,
    build_index_with_core,
    retrieve_batches_with_core,
)


PROJECT = Path(__file__).parents[1]
PACKAGE = PROJECT / "replication_runtime/feverous_official_hipporag_v1"


def _recompute_receipt_self_hash(payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    raw = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(raw).hexdigest()


def _units(count: int = CORPUS_SIZE) -> list[dict[str, object]]:
    return [
        {
            "idx": idx,
            "text": (
                f"TARGET Synthetic offline FEVEROUS atomic unit {idx}.\n"
                f"[PAGE_TITLE] Synthetic FEVEROUS page {idx // 16}"
            ),
        }
        for idx in range(count)
    ]


def _units_with_one_exact_duplicate() -> list[dict[str, object]]:
    units = _units()
    units[-1]["text"] = units[0]["text"]
    return units


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
            raise AssertionError("reopened core must never index")
        self.store.index_calls += 1
        self.store.documents = list(documents)
        if self.store.index_root is not None:
            self.store.index_root.mkdir(parents=True, exist_ok=True)
            (self.store.index_root / "synthetic.index").write_bytes(
                b"synthetic official FEVEROUS index"
            )

    def retrieve(self, queries: list[str], *, num_to_retrieve: int) -> list[object]:
        assert self.store.documents is not None
        assert num_to_retrieve == CORPUS_SIZE
        self.retrieve_calls.append((tuple(queries), num_to_retrieve))
        # Reverse the official return order and tie every score.  The adapter must
        # nevertheless select unit idx 0..4 after seeing all 8,192 scores.
        documents = list(reversed(tuple(dict.fromkeys(self.store.documents))))
        scores = [
            2.0 if document == self.store.documents[0] else 1.0
            for document in documents
        ]
        return [
            SimpleNamespace(docs=list(documents), doc_scores=list(scores))
            for _query in queries
        ]


def test_contract_requires_exactly_8192_label_free_units() -> None:
    assert MIN_CORPUS_SIZE == MAX_CORPUS_SIZE == CORPUS_SIZE == 8192
    rows = validate_corpus(_units())
    assert len(rows) == CORPUS_SIZE
    exact = str(_units()[7]["text"])
    assert serialize_unit(rows[7]) == exact
    assert serialize_corpus(rows)[7].encode("utf-8") == exact.encode("utf-8")
    assert not serialize_unit(rows[7]).startswith("Synthetic FEVEROUS page 0\n\n")
    assert len(set(serialize_corpus(rows))) == CORPUS_SIZE

    legacy_prefixed = _units()
    legacy_prefixed[0] = {
        "idx": 0,
        "title": "Forbidden extra title",
        "body": str(legacy_prefixed[0]["text"]),
    }
    with pytest.raises(FeverousOfficialHippoRAGError, match="only idx"):
        validate_corpus(legacy_prefixed)

    for count in (CORPUS_SIZE - 1, CORPUS_SIZE + 1):
        with pytest.raises(FeverousOfficialHippoRAGError, match="exactly 8192"):
            validate_corpus(_units(count))

    leaked = _units()
    leaked[0]["label"] = "SUPPORTS"
    with pytest.raises(FeverousOfficialHippoRAGError, match="only idx"):
        validate_corpus(leaked)

    duplicate_rows = validate_corpus(_units_with_one_exact_duplicate())
    duplicate_documents = serialize_corpus(duplicate_rows)
    assert len(duplicate_documents) == CORPUS_SIZE
    assert len(set(duplicate_documents)) == CORPUS_SIZE - 1
    assert corpus_text_multiplicity(duplicate_documents) == {
        "duplicate_text_group_count": 1,
        "duplicate_text_unit_count": 2,
        "official_unique_text_count": CORPUS_SIZE - 1,
    }
    leaked = _units()
    leaked[0]["gold_evidence"] = [0]
    with pytest.raises(FeverousOfficialHippoRAGError, match="only idx"):
        validate_corpus(leaked)


def test_distinct_text_content_hash_collision_fails_before_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _CollidingMd5:
        def hexdigest(self) -> str:
            return "0" * 32

    monkeypatch.setattr(
        contract.hashlib,
        "md5",
        lambda _raw: _CollidingMd5(),
    )
    index_root = tmp_path / "must-not-be-created"
    store = _Store(index_root=index_root)
    core = _FakeCore(store, build_mode=True)
    with pytest.raises(
        FeverousOfficialHippoRAGError,
        match="distinct corpus texts collide",
    ):
        build_index_with_core(
            core=core,
            units=_units(),
            index_root=index_root,
            runtime_attestation_receipt_sha256="a" * 64,
        )
    assert store.index_calls == 0
    assert not index_root.exists()

def test_query_bounds_cover_288_formal_and_4096_expansion() -> None:
    assert FORMAL_QUERY_COUNT_UPPER_BOUND == 288
    assert MAX_QUERY_COUNT == 4096
    assert len(validate_queries([f"formal query {idx}" for idx in range(288)])) == 288
    assert (
        len(validate_queries([f"expanded query {idx}" for idx in range(4096)]))
        == 4096
    )
    with pytest.raises(FeverousOfficialHippoRAGError, match="query count"):
        validate_queries([f"overflow query {idx}" for idx in range(4097)])


def test_stable_top5_requires_all_official_unique_text_scores() -> None:
    documents = serialize_corpus(validate_corpus(_units()))
    document_to_indices = {
        document: (idx,) for idx, document in enumerate(documents)
    }
    reversed_documents = list(reversed(documents))
    result = stable_top_five_from_official_result(
        retrieved_documents=reversed_documents,
        retrieved_scores=[1.0] * CORPUS_SIZE,
        document_to_indices=document_to_indices,
    )
    assert result == (0, 1, 2, 3, 4)

    with pytest.raises(FeverousOfficialHippoRAGError, match="unique global corpus"):
        stable_top_five_from_official_result(
            retrieved_documents=reversed_documents[:-1],
            retrieved_scores=[1.0] * (CORPUS_SIZE - 1),
            document_to_indices=document_to_indices,
        )


def test_exact_text_quotient_expands_one_official_score_to_all_logical_idx() -> None:
    documents = serialize_corpus(validate_corpus(_units_with_one_exact_duplicate()))
    document_to_indices: dict[str, list[int]] = {}
    for idx, document in enumerate(documents):
        document_to_indices.setdefault(document, []).append(idx)
    unique_documents = list(reversed(tuple(document_to_indices)))
    scores = [2.0 if document == documents[0] else 1.0 for document in unique_documents]
    result = stable_top_five_from_official_result(
        retrieved_documents=unique_documents,
        retrieved_scores=scores,
        document_to_indices=document_to_indices,
    )
    assert result == (0, CORPUS_SIZE - 1, 1, 2, 3)

    with pytest.raises(FeverousOfficialHippoRAGError, match="unique global corpus"):
        stable_top_five_from_official_result(
            retrieved_documents=unique_documents[:-1],
            retrieved_scores=scores[:-1],
            document_to_indices=document_to_indices,
        )


def test_build_once_reopen_and_query_batches_never_exceed_eight(
    tmp_path: Path,
) -> None:
    units = _units()
    index_root = tmp_path / "index"
    store = _Store(index_root=index_root)
    build_receipt = build_index_with_core(
        core=_FakeCore(store, build_mode=True),
        units=units,
        index_root=index_root,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    assert store.index_calls == 1
    assert store.documents == [str(row["text"]) for row in units]
    assert build_receipt["benchmark"] == BENCHMARK == "FEVEROUS"
    assert build_receipt["corpus_count"] == CORPUS_SIZE
    assert build_receipt["serialization"] == "exact_linearized_text_utf8_v1"
    assert build_receipt["official_hipporag_commit"] == OFFICIAL_HIPPORAG_COMMIT
    assert build_receipt["official_unique_text_count"] == CORPUS_SIZE
    assert build_receipt["duplicate_text_group_count"] == 0
    assert build_receipt["duplicate_text_unit_count"] == 0

    reopen_core = _FakeCore(store, build_mode=False)
    queries = [f"Synthetic query {idx}?" for idx in range(17)]
    indices, batch_sizes = retrieve_batches_with_core(
        core=reopen_core,
        units=units,
        queries=queries,
    )
    assert indices == ((0, 1, 2, 3, 4),) * len(queries)
    assert batch_sizes == (8, 8, 1)
    assert [len(call[0]) for call in reopen_core.retrieve_calls] == [8, 8, 1]
    assert all(call[1] == CORPUS_SIZE for call in reopen_core.retrieve_calls)
    assert max(batch_sizes) == MAX_QUERY_BATCH
    assert store.index_calls == 1


def test_duplicate_corpus_build_reopen_lift_and_receipt_tamper_are_auditable(
    tmp_path: Path,
) -> None:
    units = _units_with_one_exact_duplicate()
    documents = tuple(str(row["text"]) for row in units)
    index_root = tmp_path / "index"
    store = _Store(index_root=index_root)
    build_receipt = build_index_with_core(
        core=_FakeCore(store, build_mode=True),
        units=units,
        index_root=index_root,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    assert store.documents == list(documents)
    assert len(store.documents) == CORPUS_SIZE
    assert len(set(store.documents)) == CORPUS_SIZE - 1
    assert {
        key: build_receipt[key]
        for key in (
            "official_unique_text_count",
            "duplicate_text_group_count",
            "duplicate_text_unit_count",
        )
    } == {
        "official_unique_text_count": CORPUS_SIZE - 1,
        "duplicate_text_group_count": 1,
        "duplicate_text_unit_count": 2,
    }
    snapshot = snapshot_index_tree(index_root)
    validate_build_receipt(
        build_receipt,
        expected_documents=documents,
        expected_index_snapshot=snapshot,
        expected_runtime_attestation_receipt_sha256="a" * 64,
    )

    queries = ["Synthetic duplicate quotient query?"]
    indices, batch_sizes = retrieve_batches_with_core(
        core=_FakeCore(store, build_mode=False),
        units=units,
        queries=queries,
    )
    assert indices == ((0, CORPUS_SIZE - 1, 1, 2, 3),)
    assert batch_sizes == (1,)
    retrieval_receipt = make_retrieval_receipt(
        documents=documents,
        queries=queries,
        indices=indices,
        batch_sizes=batch_sizes,
        build_receipt=build_receipt,
        index_snapshot_before=snapshot,
        index_snapshot_after=snapshot,
    )
    output = {
        "receipt": retrieval_receipt,
        "retrieved_idx": [list(indices[0])],
        "schema": RETRIEVAL_OUTPUT_SCHEMA,
    }
    parsed = parse_retrieval_output(
        canonical_json_bytes(output),
        queries=queries,
        expected_build_receipt=build_receipt,
        expected_index_snapshot_after=snapshot,
    )
    assert parsed.indices == indices

    forged_build = dict(build_receipt)
    forged_build["official_unique_text_count"] = CORPUS_SIZE
    _recompute_receipt_self_hash(forged_build)
    with pytest.raises(FeverousOfficialHippoRAGError, match="contract drifted"):
        validate_build_receipt(
            forged_build,
            expected_documents=documents,
            expected_index_snapshot=snapshot,
            expected_runtime_attestation_receipt_sha256="a" * 64,
        )

    forged_retrieval = dict(retrieval_receipt)
    forged_retrieval["duplicate_text_unit_count"] = 0
    _recompute_receipt_self_hash(forged_retrieval)
    forged_output = {**output, "receipt": forged_retrieval}
    with pytest.raises(FeverousOfficialHippoRAGError, match="contract drifted"):
        parse_retrieval_output(
            canonical_json_bytes(forged_output),
            queries=queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=snapshot,
        )


def test_receipts_are_feverous_specific_and_output_has_no_content_channel(
    tmp_path: Path,
) -> None:
    documents = serialize_corpus(validate_corpus(_units()))
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "index.bin").write_bytes(b"synthetic index")
    snapshot = snapshot_index_tree(index_root)
    build_receipt = make_build_receipt(
        documents,
        index_snapshot=snapshot,
        runtime_attestation_receipt_sha256="a" * 64,
    )
    queries = ["Synthetic alpha?", "Synthetic beta?"]
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
    assert receipt["benchmark"] == "FEVEROUS"
    assert receipt["serialization"] == "exact_linearized_text_utf8_v1"
    assert str(receipt["schema"]).startswith("feverous_")
    assert set(output) == {"receipt", "retrieved_idx", "schema"}

    for forbidden in ("scores", "documents", "label", "gold_evidence"):
        forged = dict(output)
        forged[forbidden] = []
        with pytest.raises(FeverousOfficialHippoRAGError, match="only indices"):
            parse_retrieval_output(
                canonical_json_bytes(forged),
                queries=queries,
                expected_build_receipt=build_receipt,
                expected_index_snapshot_after=snapshot,
            )


def test_worker_cli_builds_8192_once_then_reopens_without_label_or_gold_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    units = _units()
    corpus_input = tmp_path / "corpus.json"
    corpus_input.write_bytes(
        canonical_json_bytes(
            {"units": units, "schema": CORPUS_INPUT_SCHEMA}
        )
    )
    query_input = tmp_path / "queries.json"
    queries = [f"CLI query {idx}?" for idx in range(9)]
    query_input.write_bytes(
        canonical_json_bytes({"queries": queries, "schema": QUERY_INPUT_SCHEMA})
    )
    index_root = tmp_path / "index"
    store = _Store(index_root=index_root)
    observed: list[tuple[bool, int]] = []

    def fake_factory(**kwargs: object) -> _FakeCore:
        force = kwargs["force_index_from_scratch"]
        count = kwargs["corpus_count"]
        assert isinstance(force, bool)
        assert count == CORPUS_SIZE
        observed.append((force, int(count)))
        return _FakeCore(store, build_mode=force)

    monkeypatch.setattr(worker, "_build_official_core", fake_factory)
    build_output = tmp_path / "build.receipt.json"
    common = [
        "--corpus-input",
        str(corpus_input),
        "--index-root",
        str(index_root),
        "--llm-model",
        str(tmp_path / "llm"),
        "--embedding-model",
        str(tmp_path / "embedding"),
        "--runtime-attestation-receipt-sha256",
        "a" * 64,
    ]
    assert worker.main(["--stage", "build", "--output", str(build_output), *common]) == 0
    assert store.index_calls == 1
    assert store.documents == [str(row["text"]) for row in units]
    build_payload = json.loads(build_output.read_text(encoding="utf-8"))
    assert build_payload["benchmark"] == "FEVEROUS"
    assert build_payload["corpus_count"] == CORPUS_SIZE
    corpus_payload = json.loads(corpus_input.read_text(encoding="utf-8"))
    assert set(corpus_payload) == {"schema", "units"}
    assert all(set(row) == {"idx", "text"} for row in corpus_payload["units"])

    retrieval_output = tmp_path / "retrieval.output.json"
    assert (
        worker.main(
            [
                "--stage",
                "retrieve",
                "--query-input",
                str(query_input),
                "--build-receipt",
                str(build_output),
                "--output",
                str(retrieval_output),
                *common,
            ]
        )
        == 0
    )
    assert observed == [(True, CORPUS_SIZE), (False, CORPUS_SIZE)]
    assert store.index_calls == 1
    value = json.loads(retrieval_output.read_text(encoding="utf-8"))
    assert set(value) == {"receipt", "retrieved_idx", "schema"}
    assert value["receipt"]["benchmark"] == "FEVEROUS"
    assert value["receipt"]["batch_sizes"] == [8, 1]
    assert value["retrieved_idx"] == [[0, 1, 2, 3, 4]] * len(queries)
    encoded = retrieval_output.read_text(encoding="utf-8").lower()
    assert '"label"' not in encoded
    assert '"gold' not in encoded
    capsys.readouterr()


def test_official_core_factory_has_no_4096_corpus_ceiling(
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
            corpus_count=CORPUS_SIZE,
        )
    assert [row["force_index_from_scratch"] for row in observed] == [True, False]
    assert all(row["retrieval_top_k"] == CORPUS_SIZE for row in observed)


def test_public_adapter_build_once_and_repeat_reopen(
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
        corpus_payload = json.loads(Path(kwargs["corpus_input"]).read_text())
        units = corpus_payload["units"]
        output_path = Path(kwargs["output_path"])
        index_root = Path(kwargs["index_root"])
        runtime_hash = str(kwargs["runtime_attestation_receipt_sha256"])
        if kwargs["stage"] == "build":
            store.index_root = index_root
            receipt = build_index_with_core(
                core=_FakeCore(store, build_mode=True),
                units=units,
                index_root=index_root,
                runtime_attestation_receipt_sha256=runtime_hash,
            )
            output_path.write_bytes(canonical_json_bytes(receipt))
            return
        query_payload = json.loads(Path(kwargs["query_input"]).read_text())
        build_receipt = json.loads(Path(kwargs["build_receipt"]).read_text())
        before = snapshot_index_tree(index_root)
        indices, batch_sizes = retrieve_batches_with_core(
            core=_FakeCore(store, build_mode=False),
            units=units,
            queries=query_payload["queries"],
        )
        receipt = make_retrieval_receipt(
            documents=serialize_corpus(validate_corpus(units)),
            queries=query_payload["queries"],
            indices=indices,
            batch_sizes=batch_sizes,
            build_receipt=build_receipt,
            index_snapshot_before=before,
            index_snapshot_after=snapshot_index_tree(index_root),
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
    receipt = adapter.build_feverous_official_hipporag_global_index_v1(
        units=_units(),
        runtime_python=runtime,
        local_llm_model=llm,
        local_embedding_model=embedding,
        base_binding_receipt_path=base,
        attestation_receipt_path=attestation,
        stage_root=stage_root,
    )
    assert receipt["benchmark"] == "FEVEROUS"
    assert receipt["index_call_count"] == 1
    for run in range(2):
        result = adapter.retrieve_feverous_official_hipporag_global_index_v1(
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
    assert store.index_calls == 1


def test_systemd_capability_preflight_freezes_both_network_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    adapter._preflight_systemd_transport()
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:6] == [
        "/usr/bin/systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        "--quiet",
    ]
    assert command[-5:] == [
        "--",
        "/usr/bin/python3",
        "-I",
        "-c",
        adapter.SYSTEMD_PREFLIGHT_SCRIPT,
    ]
    assert "IPAddressDeny=any" in command
    assert "RestrictAddressFamilies=AF_UNIX" in command
    assert observed["timeout"] == adapter.SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS


def test_launcher_is_systemd_network_isolated_offline_gpu_visible_and_feverous(
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
                    "corpus_count": CORPUS_SIZE,
                    "index_call_count": 1,
                    "stage": "build",
                    "status": "passed",
                }
            ),
            stderr=b"",
        )

    monkeypatch.setattr(adapter, "_preflight_systemd_transport", lambda: None)
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
        expected_corpus_count=CORPUS_SIZE,
    )
    command = observed["command"]
    environment = observed["environment"]
    assert isinstance(command, list)
    assert command[:6] == [
        "/usr/bin/systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        "--quiet",
    ]
    assert "IPAddressDeny=any" in command
    assert "RestrictAddressFamilies=AF_UNIX" in command
    assert "--setenv=HF_HUB_OFFLINE=1" in command
    assert "--setenv=TRANSFORMERS_OFFLINE=1" in command
    assert not any("CUDA_VISIBLE_DEVICES" in value for value in command)
    assert "--unshare-net" not in command
    assert "replication_runtime.feverous_official_hipporag_v1.worker" in command
    assert isinstance(environment, dict)
    assert "XDG_RUNTIME_DIR" not in environment or environment["XDG_RUNTIME_DIR"]


def test_launcher_rejects_nonzero_systemd_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "_preflight_systemd_transport", lambda: None)
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=23, stdout=b"child stdout", stderr=b"child stderr"
        ),
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in ("home", "cache", "tmp"):
        (stage / name).mkdir()
    with pytest.raises(FeverousOfficialHippoRAGError, match="returncode=23"):
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
            expected_corpus_count=CORPUS_SIZE,
        )


def test_worker_surface_has_no_generation_or_supervision_channel() -> None:
    source = (PACKAGE / "worker.py").read_text(encoding="utf-8")
    for forbidden in ("rag_qa", "question_type", "gold_docs", "gold_evidence", "label"):
        assert forbidden not in source
    assert "replication_runtime.morehopqa" not in source
    assert "CORPUS_INPUT_SCHEMA" in source
    assert "QUERY_INPUT_SCHEMA" in source
