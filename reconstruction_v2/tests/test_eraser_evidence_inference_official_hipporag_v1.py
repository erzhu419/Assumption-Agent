from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    adapter,
    contract,
    worker,
)


def _sentences(count: int) -> list[str]:
    return [f"Exact synthetic sentence text {ordinal}." for ordinal in range(count)]


@dataclass
class _FakeCore:
    ranked_documents: list[str] | None = None
    indexed_documents: list[str] = field(default_factory=list)
    calls: list[tuple[object, ...]] = field(default_factory=list)

    def index(self, documents: list[str]) -> None:
        self.calls.append(("index", tuple(documents)))
        self.indexed_documents = list(documents)

    def retrieve(self, queries: list[str], *, num_to_retrieve: int) -> list[object]:
        self.calls.append(("retrieve", tuple(queries), num_to_retrieve))
        ranked = (
            list(self.ranked_documents)
            if self.ranked_documents is not None
            else list(reversed(self.indexed_documents))
        )
        # Deliberately contradictory scores prove that quotient order, not an
        # adapter-side score sort, supplies the official rank.
        return [
            SimpleNamespace(
                docs=ranked,
                doc_scores=[float(position) for position in range(len(ranked))],
            )
        ]


def test_more_than_128_sentences_are_passed_exactly_without_prefix_or_truncation() -> None:
    sentence_texts = _sentences(257)
    core = _FakeCore()
    result = worker.retrieve_ordinals_with_core(
        core=core,
        query="Exact structured ICO query?",
        sentence_texts=sentence_texts,
    )
    assert result == (256, 255, 254, 253, 252)
    assert core.indexed_documents == sentence_texts
    assert len(core.indexed_documents) == 257
    assert all(not document.startswith("{") for document in core.indexed_documents)
    assert all("ordinal" not in document for document in core.indexed_documents)
    assert core.calls[-1] == (
        "retrieve",
        ("Exact structured ICO query?",),
        257,
    )
    assert contract.FROZEN_CORE_CONFIG["candidate_sentence_count_upper_bound"] is None
    assert contract.FROZEN_CORE_CONFIG["candidate_sentence_truncation"] is False


def test_exact_text_quotient_expands_each_rank_member_in_ascending_ordinals() -> None:
    sentence_texts = [
        "duplicate exact sentence",
        "unique sentence one",
        "duplicate exact sentence",
        "duplicate exact sentence",
        "unique sentence two",
        "unique sentence three",
        "unique sentence four",
    ]
    ranked = [
        "duplicate exact sentence",
        "unique sentence three",
        "unique sentence two",
        "unique sentence one",
        "unique sentence four",
    ]
    core = _FakeCore(ranked_documents=ranked)
    result = worker.retrieve_ordinals_with_core(
        core=core,
        query="Synthetic query?",
        sentence_texts=sentence_texts,
    )
    assert core.indexed_documents == [
        "duplicate exact sentence",
        "unique sentence one",
        "unique sentence two",
        "unique sentence three",
        "unique sentence four",
    ]
    assert result == (0, 2, 3, 5, 4)


def test_content_hash_collision_fails_before_official_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Digest:
        def hexdigest(self) -> str:
            return "0" * 32

    monkeypatch.setattr(contract.hashlib, "md5", lambda _raw: _Digest())
    core = _FakeCore()
    with pytest.raises(
        contract.EraserEvidenceInferenceOfficialHippoRAGError,
        match="content hash",
    ):
        worker.retrieve_ordinals_with_core(
            core=core,
            query="Synthetic query?",
            sentence_texts=_sentences(6),
        )
    assert core.calls == []


@pytest.mark.parametrize("mode", ["duplicate", "foreign", "omitted"])
def test_official_ranked_quotient_tamper_fails_closed(mode: str) -> None:
    sentences = _sentences(7)
    quotient, mapping = contract.exact_text_quotient(sentences)
    ranked: list[object] = list(quotient)
    if mode == "duplicate":
        ranked[-1] = ranked[0]
    elif mode == "foreign":
        ranked[-1] = "foreign cross-item sentence"
    else:
        ranked.pop()
    with pytest.raises(
        contract.EraserEvidenceInferenceOfficialHippoRAGError,
        match="omitted|unknown|duplicate|quotient member",
    ):
        contract.expand_ranked_quotient_to_top_five(
            retrieved_documents=ranked,
            document_to_ordinals=mapping,
            logical_sentence_count=len(sentences),
        )


def test_ordinal_partition_tamper_fails_closed() -> None:
    with pytest.raises(
        contract.EraserEvidenceInferenceOfficialHippoRAGError,
        match="ascending|partition",
    ):
        contract.expand_ranked_quotient_to_top_five(
            retrieved_documents=["a", "b", "c", "d", "e"],
            document_to_ordinals={
                "a": (1, 0),
                "b": (2,),
                "c": (3,),
                "d": (4,),
                "e": (5,),
            },
            logical_sentence_count=6,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"[0, 1, 2, 3, 4]\n",
        b'{"ordinals":[0,1,2,3,4]}\n',
        b"[0,0,1,2,3]\n",
        b"[0,1,2,3,99]\n",
        b"[0,1,2,3,4] trailing\n",
    ],
)
def test_worker_output_is_canonical_ordinals_only_and_tamper_closed(raw: bytes) -> None:
    with pytest.raises(contract.EraserEvidenceInferenceOfficialHippoRAGError):
        contract.parse_ordinals_only_output(raw, logical_sentence_count=7)
    assert contract.parse_ordinals_only_output(
        b"[0,1,2,3,4]\n", logical_sentence_count=7
    ) == (0, 1, 2, 3, 4)


def test_worker_input_envelope_is_exact_and_canonical(tmp_path: Path) -> None:
    valid = {
        "query": "Synthetic query?",
        "schema": contract.INPUT_SCHEMA,
        "sentence_texts": _sentences(6),
    }
    path = tmp_path / "input.json"
    path.write_bytes(contract.canonical_json_bytes(valid))
    assert worker._load_input(path) == ("Synthetic query?", _sentences(6))

    path.write_text(json.dumps(valid, indent=2), encoding="utf-8")
    with pytest.raises(
        contract.EraserEvidenceInferenceOfficialHippoRAGError,
        match="canonical",
    ):
        worker._load_input(path)

    extra = {**valid, "answer": "must never enter the adapter"}
    path.write_bytes(contract.canonical_json_bytes(extra))
    with pytest.raises(
        contract.EraserEvidenceInferenceOfficialHippoRAGError,
        match="envelope",
    ):
        worker._load_input(path)

    target = tmp_path / "target.json"
    target.write_bytes(contract.canonical_json_bytes(valid))
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(
        contract.EraserEvidenceInferenceOfficialHippoRAGError,
        match="unavailable",
    ):
        worker._load_input(path)


def test_two_item_calls_have_disjoint_exact_text_corpora() -> None:
    first = [f"FIRST ITEM sentence {ordinal}" for ordinal in range(7)]
    second = [f"SECOND ITEM sentence {ordinal}" for ordinal in range(9)]
    first_core = _FakeCore()
    second_core = _FakeCore()
    worker.retrieve_ordinals_with_core(
        core=first_core,
        query="First query?",
        sentence_texts=first,
    )
    worker.retrieve_ordinals_with_core(
        core=second_core,
        query="Second query?",
        sentence_texts=second,
    )
    assert first_core.indexed_documents == first
    assert second_core.indexed_documents == second
    assert set(first_core.indexed_documents).isdisjoint(second_core.indexed_documents)
    assert all(call[0] == "index" for call in first_core.calls[:1])
    assert all(call[0] == "index" for call in second_core.calls[:1])


def test_public_adapter_uses_fresh_ephemeral_root_for_every_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    manifests = project / "manifests"
    manifests.mkdir(parents=True)
    binding = manifests / "binding.json"
    binding.write_text("{}", encoding="utf-8")
    attestation = manifests / "attestation.json"
    attestation.write_text("{}", encoding="utf-8")
    runtime_python = tmp_path / "python"
    runtime_python.write_bytes(b"synthetic")
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    llm.mkdir()
    embedding.mkdir()
    work_root = tmp_path / "item_work"
    observed: list[dict[str, Any]] = []

    verified: list[dict[str, Any]] = []
    monkeypatch.setattr(
        adapter,
        "verify_formal_runtime_attestation_v3",
        lambda **kwargs: verified.append(kwargs),
    )

    def fake_launch(**kwargs: Any) -> None:
        input_payload = json.loads(kwargs["input_path"].read_text(encoding="utf-8"))
        observed.append(input_payload)
        assert kwargs["writable_root"] == work_root
        assert not kwargs["index_root"].exists()
        kwargs["output_path"].write_bytes(b"[0,1,2,3,4]\n")

    monkeypatch.setattr(adapter, "_launch_worker", fake_launch)
    first = _sentences(129)
    second = [f"Second isolated sentence {ordinal}" for ordinal in range(131)]
    for query, sentences in (("First query?", first), ("Second query?", second)):
        result = adapter.run_item_local_official_hipporag_v1(
            query=query,
            sentence_texts=sentences,
            runtime_python=runtime_python,
            local_llm_model=llm,
            local_embedding_model=embedding,
            base_binding_receipt_path=binding,
            attestation_receipt_path=attestation,
            work_root=work_root,
        )
        assert result == (0, 1, 2, 3, 4)
        assert not work_root.exists()
    assert observed == [
        {"query": "First query?", "schema": contract.INPUT_SCHEMA, "sentence_texts": first},
        {"query": "Second query?", "schema": contract.INPUT_SCHEMA, "sentence_texts": second},
    ]
    assert len(verified) == 2
    assert all(row["project_root"] == project for row in verified)


def test_worker_factory_thinly_reuses_frozen_musique_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    observed: dict[str, Path] = {}

    def fake_factory(**kwargs: Path) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(worker, "_build_musique_official_core", fake_factory)
    assert worker._build_official_core(
        save_dir=tmp_path / "index",
        llm_model=tmp_path / "llm",
        embedding_model=tmp_path / "embedding",
    ) is sentinel
    assert observed == {
        "save_dir": tmp_path / "index",
        "llm_model": tmp_path / "llm",
        "embedding_model": tmp_path / "embedding",
    }


def test_worker_source_has_no_answer_gold_or_persistent_reopen_channel() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert "answer" not in source.casefold()
    assert "gold" not in source.casefold()
    assert "force_index_from_scratch=False" not in source
    assert "128" not in source
    assert "index(list(quotient))" in source
