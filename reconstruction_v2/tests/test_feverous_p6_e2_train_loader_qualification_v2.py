from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter
from assumption_agent.benchmarks import (
    feverous_p6_e2_train_loader_qualification_v2 as qualification,
)


PROJECT = Path(__file__).resolve().parents[1]


def _official_blank_sentinel() -> dict[str, str]:
    return {field: "" for field in source_adapter.OFFICIAL_RECORD_FIELDS}


def test_persisted_real_receipt_is_canonical_exact_and_content_free() -> None:
    expected = qualification.expected_qualification_receipt(PROJECT)
    observed = qualification.verify_train_loader_qualification(PROJECT)
    raw = (PROJECT / qualification.MANIFEST_RELATIVE).read_bytes()

    assert dict(observed) == expected
    assert raw == qualification._canonical_bytes(expected) + b"\n"
    assert expected["annotation_size_bytes"] == 177_565_233
    assert expected["annotation_file_sha256"] == (
        "0c29ccba41e27c5b988ca5132085e8d67c7921f265707bea170bfbde12bceee7"
    )
    assert expected["annotation_physical_rows"] == 71_292
    assert expected["annotation_nonblank_rows"] == 71_291
    assert expected["annotation_blank_sentinel_rows"] == 1
    assert expected["exact_count_identity"] == "71292=71291+1"
    assert expected["annotation_records_retained_after_scan"] == 0
    assert expected["selection_secret_generated_or_read"] is False
    assert expected["candidate_adapter_invoked"] is False
    assert expected["cohort_or_block_selection_invoked"] is False
    assert (
        expected["wikipedia_database_stated_hashed_opened_or_queried"] is False
    )
    assert expected["retrieval_action_utility_evaluator_or_scoring_calls"] == 0
    assert expected["claim_corpus_gold_label_or_outcome_rows_persisted"] is False
    assert expected["development_or_test_source_accessed"] is False
    assert expected["online_evaluator_calls"] == 0


def test_exact_sentinel_predicate_and_code_bindings_are_strict() -> None:
    sentinel = _official_blank_sentinel()
    assert source_adapter._is_blank_sentinel(sentinel) is True

    class StringSubclass(str):
        pass

    near_sentinels = (
        {},
        {**sentinel, "evidence": []},
        {**sentinel, "unexpected": ""},
        {**sentinel, "label": StringSubclass("")},
        {**sentinel, "label": " "},
    )
    assert all(
        source_adapter._is_blank_sentinel(row) is False
        for row in near_sentinels
    )

    receipt = qualification.expected_qualification_receipt(PROJECT)
    predicate_source = inspect.getsource(
        source_adapter._is_blank_sentinel
    ).encode("utf-8")
    decoder_source = inspect.getsource(formal_source._decode_json_line).encode(
        "utf-8"
    )
    assert receipt["exact_blank_sentinel_predicate_source_sha256"] == (
        hashlib.sha256(predicate_source).hexdigest()
    )
    assert receipt["strict_json_decoder_source_sha256"] == hashlib.sha256(
        decoder_source
    ).hexdigest()


def test_synthetic_stream_exhaustion_uses_same_aggregate_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = tmp_path / "synthetic_train.jsonl"
    rows = (_official_blank_sentinel(), {"synthetic": "nonblank"})
    raw = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )
    annotation.write_bytes(raw)
    annotation.chmod(0o600)

    monkeypatch.setattr(
        qualification, "ANNOTATION_RELATIVE", Path("synthetic_train.jsonl")
    )
    monkeypatch.setattr(
        formal_source, "FROZEN_ANNOTATION_SIZE_BYTES", len(raw)
    )
    monkeypatch.setattr(
        formal_source,
        "FROZEN_ANNOTATION_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(formal_source, "FROZEN_ANNOTATION_NONBLANK_ROWS", 1)
    monkeypatch.setattr(
        formal_source, "FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS", 1
    )
    monkeypatch.setattr(
        qualification,
        "_code_bindings",
        lambda _project: {
            "formal_source_file_sha256": "0" * 64,
            "source_adapter_file_sha256": "1" * 64,
            "strict_json_decoder_source_sha256": "2" * 64,
            "exact_blank_sentinel_predicate_source_sha256": "3" * 64,
        },
    )

    observed = qualification.run_real_train_loader_qualification(tmp_path)
    assert observed["annotation_physical_rows"] == 2
    assert observed["annotation_nonblank_rows"] == 1
    assert observed["annotation_blank_sentinel_rows"] == 1
    assert observed["annotation_records_retained_after_scan"] == 0


def test_verifier_rejects_even_self_consistent_aggregate_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = qualification.expected_qualification_receipt(PROJECT)
    tampered["annotation_nonblank_rows"] = 71_290
    tampered["exact_count_identity"] = "71292=71290+1"
    tampered.pop("qualification_sha256")
    tampered["qualification_sha256"] = qualification.stable_hash(tampered)

    path = tmp_path / "tampered.json"
    path.write_bytes(qualification._canonical_bytes(tampered) + b"\n")
    monkeypatch.setattr(qualification, "MANIFEST_RELATIVE", path)
    with pytest.raises(
        qualification.FeverousTrainLoaderQualificationError,
        match="receipt drifted",
    ):
        qualification.verify_train_loader_qualification(PROJECT)
