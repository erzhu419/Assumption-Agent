from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks.official_hipporag_runtime_adapter_qualification_v1 import (
    ANSWER_SCHEMA,
    OFFICIAL_COMMIT,
    _CORE_PROBE_PROGRAM,
    build_synthetic_fixture,
    documents_from_custom_json,
    normalize_multi_answers,
    questions_from_custom_json,
    write_answer_json,
)
from assumption_agent.models import stable_hash


PROJECT = Path(__file__).parents[1]
RECEIPT = PROJECT / "manifests" / "official_hipporag_runtime_adapter_qualification_v1.json"


def test_synthetic_hierarchy_flattens_deterministically() -> None:
    fixture = build_synthetic_fixture()
    documents = documents_from_custom_json(fixture)
    questions = questions_from_custom_json(fixture)

    assert len(documents) == 4
    assert len(set(documents)) == 4
    assert all("Hierarchy:" in document for document in documents)
    assert "root/branch-a/leaf-a" in documents[2]
    assert len(questions) == 1
    assert len(questions[0]["normalized_aliases"]) == 1
    assert stable_hash(documents) == stable_hash(documents_from_custom_json(fixture))


def test_multi_answer_normalization_is_unicode_stable_and_deduplicated() -> None:
    assert normalize_multi_answers(["  Alpha-Beta ", "alpha beta", "ALPHA—BETA"]) == (
        "alpha beta",
    )
    with pytest.raises(TypeError):
        normalize_multi_answers("not-a-sequence-of-aliases")


def test_answer_writer_emits_predictions_only_and_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    digest = write_answer_json(
        path,
        question_ids=["q1"],
        predictions={"q1": "synthetic-prediction"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "schema": ANSWER_SCHEMA,
        "answers": [{"question_id": "q1", "answer": "synthetic-prediction"}],
    }
    assert len(digest) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "accepted_aliases" not in path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_answer_json(
            path,
            question_ids=["q1"],
            predictions={"q1": "replacement"},
        )


def test_hierarchy_rejects_cycles_and_external_provenance() -> None:
    fixture = build_synthetic_fixture()
    fixture["corpus"][0]["parent_id"] = "leaf-b"
    with pytest.raises(Exception, match="cycle"):
        documents_from_custom_json(fixture)

    fixture = build_synthetic_fixture()
    fixture["provenance"]["external_corpus_rows"] = 1
    with pytest.raises(PermissionError):
        documents_from_custom_json(fixture)


def test_core_probe_uses_normal_imports_and_all_required_paths() -> None:
    assert "from hipporag import HippoRAG" in _CORE_PROBE_PROGRAM
    assert ".index(documents)" in _CORE_PROBE_PROGRAM
    assert ".retrieve(queries" in _CORE_PROBE_PROGRAM
    assert ".rag_qa(queries)" in _CORE_PROBE_PROGRAM
    assert "monkeypatch" not in _CORE_PROBE_PROGRAM
    assert "sys.modules" not in _CORE_PROBE_PROGRAM


def test_safe_receipt_if_materialized() -> None:
    if not RECEIPT.exists():
        pytest.skip("qualification receipt has not been materialized")
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    declared = payload.pop("qualification_sha256")
    assert stable_hash(payload) == declared
    assert payload["qualified"] is True
    assert payload["source_binding"]["commit"] == OFFICIAL_COMMIT
    assert payload["scope"]["benchmark_performance_claim"] is False
    assert payload["safety"]["online_model_calls"] == 0
    assert payload["safety"]["online_evaluator_calls"] == 0
    assert payload["safety"]["network_namespace_isolated"] is True
    assert payload["safety"]["external_network_transport_possible"] is False
    assert payload["safety"]["official_source_modified"] is False
