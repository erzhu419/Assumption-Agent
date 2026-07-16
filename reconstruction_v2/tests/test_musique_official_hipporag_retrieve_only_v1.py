from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.models import stable_hash
from replication_runtime.musique_official_hipporag_v1.binding import (
    OFFICIAL_COMMIT,
    OFFICIAL_SOURCE_TREE_SHA256,
    QUALIFICATION_SHA256,
    current_implementation_binding,
    validate_binding_receipt,
)
from replication_runtime.musique_official_hipporag_v1.contract import (
    FROZEN_CORE_CONFIG,
    MuSiQueOfficialHippoRAGError,
    parse_idx_only_output,
    serialize_candidate_corpus,
    validate_single_item,
)
from replication_runtime.musique_official_hipporag_v1.worker import (
    retrieve_idx_with_core,
)


PROJECT = Path(__file__).parents[1]
BINDING = PROJECT / "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
PACKAGE = PROJECT / "replication_runtime/musique_official_hipporag_v1"


def _paragraphs(count: int = 7) -> list[dict[str, object]]:
    return [
        {
            "idx": idx,
            "title": f"Synthetic title {idx}",
            "paragraph_text": f"Locally generated paragraph signal {idx}.",
        }
        for idx in range(count)
    ]


def test_exact_gold_free_item_contract_and_canonical_idx() -> None:
    question, paragraphs = validate_single_item("Synthetic question?", _paragraphs())
    assert question == "Synthetic question?"
    assert tuple(row.idx for row in paragraphs) == tuple(range(7))

    leaked = _paragraphs()
    leaked[0]["is_supporting"] = True
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="only idx"):
        validate_single_item("Synthetic question?", leaked)

    missing = _paragraphs()
    missing[2]["idx"] = 3
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="contiguous"):
        validate_single_item("Synthetic question?", missing)


def test_document_serialization_is_unique_even_for_duplicate_text() -> None:
    rows = _paragraphs(5)
    rows[1]["title"] = rows[0]["title"]
    rows[1]["paragraph_text"] = rows[0]["paragraph_text"]
    _, validated = validate_single_item("Synthetic question?", rows)
    documents = serialize_candidate_corpus(validated)
    assert len(documents) == len(set(documents)) == 5
    assert json.loads(documents[0])["paragraph_idx"] == 0
    assert json.loads(documents[1])["paragraph_idx"] == 1


@dataclass
class _FakeCore:
    mode: str = "valid"

    def __post_init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.documents: list[str] = []

    def index(self, documents: list[str]) -> None:
        self.calls.append(("index", len(documents)))
        self.documents = list(documents)

    def retrieve(self, queries: list[str], *, num_to_retrieve: int) -> list[object]:
        self.calls.append(("retrieve", tuple(queries), num_to_retrieve))
        documents = list(reversed(self.documents))
        if self.mode == "duplicate":
            documents[-1] = documents[0]
        elif self.mode == "cross_item":
            documents[-1] = "foreign item document"
        return [SimpleNamespace(docs=documents, doc_scores=[1.0] * len(documents))]


def test_fake_official_core_indexes_one_item_and_stably_tie_breaks() -> None:
    core = _FakeCore()
    result = retrieve_idx_with_core(
        core=core,
        question="Synthetic question?",
        paragraphs=_paragraphs(7),
    )
    assert result == (0, 1, 2, 3, 4)
    assert core.calls == [
        ("index", 7),
        ("retrieve", ("Synthetic question?",), 7),
    ]


@pytest.mark.parametrize("mode", ["duplicate", "cross_item"])
def test_retrieval_mapping_rejects_duplicate_or_cross_item_docs(mode: str) -> None:
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="duplicate|cross-corpus"):
        retrieve_idx_with_core(
            core=_FakeCore(mode),
            question="Synthetic question?",
            paragraphs=_paragraphs(7),
        )


def test_idx_only_output_is_exact_and_fail_closed() -> None:
    assert parse_idx_only_output(b"[4,3,2,1,0]\n", candidate_count=6) == (4, 3, 2, 1, 0)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="duplicate"):
        parse_idx_only_output(b"[0,0,1,2,3]\n", candidate_count=6)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="canonical"):
        parse_idx_only_output(b"[0, 1, 2, 3, 4]\n", candidate_count=6)


def test_worker_is_retrieve_only_and_has_no_answer_generation_api() -> None:
    worker_source = (PACKAGE / "worker.py").read_text(encoding="utf-8")
    assert "getattr(core, \"index\"" in worker_source
    assert "getattr(core, \"retrieve\"" in worker_source
    assert "rag_qa" not in worker_source
    assert "gold_docs" not in worker_source


def test_safe_binding_receipt_freezes_official_runtime_and_deviation() -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    validate_binding_receipt(payload, project_root=PROJECT)
    assert payload["qualification_binding"]["qualification_sha256"] == QUALIFICATION_SHA256
    assert payload["official_source_binding"]["commit"] == OFFICIAL_COMMIT
    assert (
        payload["official_source_binding"]["python_source_tree_sha256"]
        == OFFICIAL_SOURCE_TREE_SHA256
    )
    assert payload["official_source_binding"]["python_source_file_count"] == 52
    assert payload["runtime_binding"]["official_openai_pin"] == "1.91.1"
    assert payload["runtime_binding"]["runtime_openai_version"] == "1.91.0"
    assert payload["runtime_binding"]["openai_pin_satisfied"] is False
    assert payload["runtime_binding"]["openai_1_91_0_deviation_explicitly_bound"] is True
    assert payload["config_binding"] == {
        "payload": FROZEN_CORE_CONFIG,
        "config_sha256": stable_hash(FROZEN_CORE_CONFIG),
    }
    assert payload["implementation_binding"] == current_implementation_binding(PROJECT)
    assert [
        row["path"] for row in payload["implementation_binding"]["files"][:3]
    ] == [
        "assumption_agent/__init__.py",
        "assumption_agent/models.py",
        "replication_runtime/__init__.py",
    ]
    assert payload["scope"]["official_core_calls"] == ["index", "retrieve"]
    assert payload["scope"]["answer_generation_calls"] == 0
    assert payload["scope"]["benchmark_rows_read_while_binding"] == 0
    assert (
        payload["synthetic_local_qualification"]["status"]
        == "passed_non_scoring_synthetic_local_retrieve_only"
    )


def test_binding_rejects_implementation_or_receipt_tamper() -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    payload["scope"]["answer_generation_calls"] = 1
    payload["receipt_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="contract drifted"):
        validate_binding_receipt(payload, project_root=PROJECT)

    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    payload["receipt_sha256"] = "0" * 64
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="self-hash"):
        validate_binding_receipt(payload, project_root=PROJECT)


def _rehash(payload: dict[str, object]) -> None:
    payload["receipt_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )


@pytest.mark.parametrize(
    ("section", "extra_key"),
    [
        (None, "unexpected_top_level"),
        ("qualification_binding", "unexpected_qualification_field"),
        ("official_source_binding", "unexpected_source_field"),
        ("runtime_binding", "unexpected_runtime_field"),
        ("asset_binding", "unexpected_asset_field"),
        ("config_binding", "unexpected_config_field"),
        ("scope", "unexpected_scope_field"),
        ("synthetic_local_qualification", "unexpected_synthetic_field"),
    ],
)
def test_binding_rejects_rehashed_extra_keys(
    section: str | None, extra_key: str
) -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    target = payload if section is None else payload[section]
    target[extra_key] = "not allowed"
    _rehash(payload)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="key set mismatch"):
        validate_binding_receipt(
            payload, project_root=PROJECT, verify_implementation=False
        )


def test_binding_rejects_rehashed_transitive_file_or_dependency_schema_tamper() -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    payload["implementation_binding"]["files"][0]["unexpected"] = True
    _rehash(payload)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="file row key set"):
        validate_binding_receipt(
            payload, project_root=PROJECT, verify_implementation=False
        )

    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    payload["runtime_binding"]["dependency_versions"]["unexpected-package"] = "1"
    _rehash(payload)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="dependency versions key set"):
        validate_binding_receipt(
            payload, project_root=PROJECT, verify_implementation=False
        )

    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    del payload["scope"]["online_evaluator_calls"]
    _rehash(payload)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="scope key set"):
        validate_binding_receipt(
            payload, project_root=PROJECT, verify_implementation=False
        )
