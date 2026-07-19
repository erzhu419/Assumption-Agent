from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unicodedata

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_adapter_compatibility_qualification_v3 as qualification,
)
from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as adapter


PROJECT = Path(__file__).resolve().parents[1]
PRIVATE_CANDIDATE_SENTINEL = "PRIVATE-CANDIDATE-MUST-NOT-PERSIST"


def _annotation_receipt() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": formal_source.ANNOTATION_RECEIPT_SCHEMA,
        "version": formal_source.VERSION,
        "status": "train_annotation_read_once_and_verified",
        "source_split": "TRAIN",
        "source_spec_sha256": formal_source.FROZEN_TRAIN_SOURCE_SPEC.spec_sha256,
        "source_binding_sha256": adapter.FROZEN_TRAIN_BINDING.binding_sha256,
        "formal_source_opener_source_sha256": "a" * 64,
        "formal_source": True,
        "annotation_basename": formal_source.FROZEN_ANNOTATION_BASENAME,
        "annotation_size_bytes": formal_source.FROZEN_ANNOTATION_SIZE_BYTES,
        "annotation_file_sha256": formal_source.FROZEN_ANNOTATION_SHA256,
        "annotation_nonblank_rows": formal_source.FROZEN_ANNOTATION_NONBLANK_ROWS,
        "annotation_blank_sentinel_rows": (
            formal_source.FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS
        ),
        "annotation_file_read_count": 1,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return formal_source._self_hashed(body, "annotation_receipt_sha256")


def _adapter_receipt() -> dict[str, Any]:
    receipt = adapter._aggregate_receipt(
        decisions=(),
        pages=(),
        corpus_units=(),
        binding=adapter.FROZEN_TRAIN_BINDING,
    )
    receipt.update(
        {
            "input_record_count": qualification.EXPECTED_TRAIN_RECORD_COUNT,
            "record_status_counts": {
                "blank_sentinel": 1,
                "no_eligible_canonical_set": (
                    qualification.EXPECTED_TRAIN_RECORD_COUNT - 1
                ),
            },
            "candidate_count": 0,
            "official_evidence_set_count": (
                qualification.EXPECTED_ADAPTER_OFFICIAL_EVIDENCE_SET_COUNT
            ),
            "official_evidence_reference_count": (
                qualification.EXPECTED_ADAPTER_OFFICIAL_CONTENT_REFERENCE_COUNT
            ),
            "excluded_family_structure_set_count": 7,
            "excluded_nonexact_title_context_set_count": 2,
            "excluded_nonexact_title_context_reference_count": 2,
            "records_with_excluded_nonexact_title_context_count": 2,
        }
    )
    receipt.pop("adapter_receipt_sha256")
    receipt["adapter_receipt_sha256"] = adapter._stable_hash(receipt)
    adapter.verify_adapter_receipt(receipt)
    return receipt


def _raw_topology() -> qualification.RawTrainTopology:
    return qualification.RawTrainTopology(
        record_count=qualification.EXPECTED_TRAIN_RECORD_COUNT,
        blank_sentinel_record_count=(
            qualification.EXPECTED_BLANK_SENTINEL_RECORD_COUNT
        ),
        evidence_set_count=qualification.EXPECTED_RAW_EVIDENCE_SET_COUNT,
        content_reference_count=qualification.EXPECTED_RAW_CONTENT_REFERENCE_COUNT,
    )


def _rebind_adapter_receipt(receipt: dict[str, Any]) -> None:
    receipt.pop("adapter_receipt_sha256", None)
    receipt["adapter_receipt_sha256"] = adapter._stable_hash(receipt)


def test_forms_and_validates_content_free_exact_aggregate_receipt() -> None:
    receipt = qualification.form_adapter_compatibility_qualification_receipt(
        project=PROJECT,
        annotation_receipt=_annotation_receipt(),
        adapter_receipt=_adapter_receipt(),
        raw_topology=_raw_topology(),
    )
    assert qualification.validate_adapter_compatibility_qualification_receipt(
        receipt, project=PROJECT
    ) == receipt["qualification_sha256"]
    assert receipt["raw_evidence_set_count"] == 77_492
    assert receipt["raw_content_reference_count"] == 349_556
    assert receipt["adapter_evidence_set_count"] == 75_221
    assert receipt["adapter_content_reference_count"] == 338_063
    assert (
        receipt["exact_context_evidence_set_count_after_nonexact_exclusion"]
        == 75_219
    )
    assert (
        receipt["exact_context_reference_count_after_nonexact_exclusion"]
        == 338_061
    )
    assert receipt["invalid_nonexact_title_context_evidence_set_count"] == 2
    assert receipt["invalid_nonexact_title_context_reference_count"] == 2
    assert receipt["records_with_invalid_nonexact_title_context_count"] == 2
    assert receipt["qualification_runner_file_sha256"] == qualification._sha256_file(
        PROJECT / qualification.QUALIFICATION_RUNNER_CODE_RELATIVE
    )
    assert receipt["wikipedia_resolver_file_sha256"] == qualification._sha256_file(
        PROJECT / qualification.WIKIPEDIA_RESOLVER_CODE_RELATIVE
    )
    assert receipt["atomic_corpus_file_sha256"] == qualification._sha256_file(
        PROJECT / qualification.ATOMIC_CORPUS_CODE_RELATIVE
    )
    assert receipt["acquisition_core_file_sha256"] == qualification._sha256_file(
        PROJECT / qualification.ACQUISITION_CORE_CODE_RELATIVE
    )
    assert receipt["unicode_database_version"] == unicodedata.unidata_version
    assert receipt["strict_json_decoder_source_sha256"] == hashlib.sha256(
        inspect.getsource(formal_source._decode_json_line).encode("utf-8")
    ).hexdigest()
    assert receipt[
        "exact_blank_sentinel_predicate_source_sha256"
    ] == hashlib.sha256(
        inspect.getsource(adapter._is_blank_sentinel).encode("utf-8")
    ).hexdigest()
    assert (
        receipt["adapter_aggregate_receipt"][
            "excluded_family_structure_set_count"
        ]
        == 7
    )
    assert receipt["selection_secret_generated_or_read"] is False
    assert receipt["cohort_block_or_canonical_set_selected"] is False
    assert receipt["retrieval_action_utility_evaluator_or_scoring_calls"] == 0
    serialized = json.dumps(receipt, sort_keys=True)
    assert PRIVATE_CANDIDATE_SENTINEL not in serialized


@pytest.mark.parametrize(
    "field",
    (
        "qualification_runner_file_sha256",
        "wikipedia_resolver_file_sha256",
        "atomic_corpus_file_sha256",
        "acquisition_core_file_sha256",
        "strict_json_decoder_source_sha256",
        "exact_blank_sentinel_predicate_source_sha256",
    ),
)
def test_receipt_rejects_tampered_runner_resolver_or_loader_binding(
    field: str,
) -> None:
    receipt = qualification.form_adapter_compatibility_qualification_receipt(
        project=PROJECT,
        annotation_receipt=_annotation_receipt(),
        adapter_receipt=_adapter_receipt(),
        raw_topology=_raw_topology(),
    )
    receipt[field] = "b" * 64
    receipt.pop("qualification_sha256")
    receipt["qualification_sha256"] = qualification.stable_hash(receipt)
    with pytest.raises(
        qualification.FeverousAdapterCompatibilityQualificationError,
        match="receipt drifted",
    ):
        qualification.validate_adapter_compatibility_qualification_receipt(
            receipt, project=PROJECT
        )


def test_receipt_rejects_tampered_unicode_database_binding() -> None:
    receipt = qualification.form_adapter_compatibility_qualification_receipt(
        project=PROJECT,
        annotation_receipt=_annotation_receipt(),
        adapter_receipt=_adapter_receipt(),
        raw_topology=_raw_topology(),
    )
    receipt["unicode_database_version"] = "0.0.0"
    receipt.pop("qualification_sha256")
    receipt["qualification_sha256"] = qualification.stable_hash(receipt)
    with pytest.raises(
        qualification.FeverousAdapterCompatibilityQualificationError,
        match="receipt drifted",
    ):
        qualification.validate_adapter_compatibility_qualification_receipt(
            receipt, project=PROJECT
        )


@pytest.mark.parametrize(
    "field",
    (
        "excluded_nonexact_title_context_set_count",
        "excluded_nonexact_title_context_reference_count",
        "records_with_excluded_nonexact_title_context_count",
    ),
)
def test_each_preregistered_mismatch_aggregate_must_equal_two(field: str) -> None:
    adapter_receipt = _adapter_receipt()
    adapter_receipt[field] = 3
    _rebind_adapter_receipt(adapter_receipt)
    with pytest.raises(
        qualification.FeverousAdapterCompatibilityQualificationError,
        match="preregistered topology",
    ):
        qualification.form_adapter_compatibility_qualification_receipt(
            project=PROJECT,
            annotation_receipt=_annotation_receipt(),
            adapter_receipt=adapter_receipt,
            raw_topology=_raw_topology(),
        )


def test_raw_topology_counts_all_records_without_retaining_content() -> None:
    blank = {field: "" for field in adapter.OFFICIAL_RECORD_FIELDS}
    record = {
        "annotator_operations": [],
        "challenge": "Entity Disambiguation",
        "claim": "PRIVATE CLAIM",
        "evidence": [
            {
                "content": ["Private_Page_sentence_0", "Private_Page_sentence_1"],
                "context": {},
            },
            {
                "content": ["Private_Page_sentence_2"],
                "context": {},
            },
        ],
        "id": 1,
        "label": "SUPPORTS",
    }
    topology = qualification._raw_train_topology((blank, record))
    assert topology == qualification.RawTrainTopology(2, 1, 2, 3)
    assert "PRIVATE" not in repr(topology)


def test_real_runner_orchestrates_only_train_resolver_and_aggregate_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    records = ({field: "" for field in adapter.OFFICIAL_RECORD_FIELDS},)

    class FakeSource:
        def __init__(self, *, annotation_path: Path, database_path: Path) -> None:
            calls["annotation_path"] = annotation_path
            calls["database_path"] = database_path
            self.annotation_receipt = _annotation_receipt()

        def __enter__(self) -> "FakeSource":
            return self

        def __exit__(self, *_exc: object) -> None:
            calls["closed"] = True

        def read_annotations_once(self) -> tuple[dict[str, str], ...]:
            calls["annotation_reads"] = 1
            return records

        def exact_resolver_for_candidate_screen(self) -> object:
            resolver = object()
            calls["resolver"] = resolver
            return resolver

    def adapt(
        observed_records: object,
        *,
        source_split: str,
        resolver: object,
        binding: object,
    ) -> SimpleNamespace:
        calls["adapter_records"] = observed_records
        calls["adapter_kwargs"] = {
            "source_split": source_split,
            "resolver": resolver,
            "binding": binding,
        }
        return SimpleNamespace(
            receipt=_adapter_receipt(),
            candidates=(PRIVATE_CANDIDATE_SENTINEL,),
            corpus_units=(),
        )

    monkeypatch.setattr(formal_source, "ControlledTrainSource", FakeSource)
    monkeypatch.setattr(qualification, "_raw_train_topology", lambda _r: _raw_topology())
    monkeypatch.setattr(adapter, "adapt_train_candidate_records", adapt)

    receipt = qualification.run_real_adapter_compatibility_qualification(PROJECT)
    assert calls["annotation_path"] == PROJECT / qualification.ANNOTATION_RELATIVE
    assert calls["database_path"] == PROJECT / qualification.DATABASE_RELATIVE
    assert calls["annotation_reads"] == 1
    assert calls["closed"] is True
    assert calls["adapter_records"] is records
    assert calls["adapter_kwargs"] == {
        "source_split": "TRAIN",
        "resolver": calls["resolver"],
        "binding": adapter.FROZEN_TRAIN_BINDING,
    }
    assert PRIVATE_CANDIDATE_SENTINEL not in json.dumps(receipt, sort_keys=True)
    assert tuple(
        inspect.signature(
            qualification.run_real_adapter_compatibility_qualification
        ).parameters
    ) == ("project",)


def test_future_manifest_verifier_requires_canonical_aggregate_only_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = qualification.form_adapter_compatibility_qualification_receipt(
        project=PROJECT,
        annotation_receipt=_annotation_receipt(),
        adapter_receipt=_adapter_receipt(),
        raw_topology=_raw_topology(),
    )
    path = tmp_path / "compatibility.json"
    path.write_bytes(qualification._canonical_bytes(receipt) + b"\n")
    monkeypatch.setattr(qualification, "MANIFEST_RELATIVE", path)
    observed = qualification.verify_adapter_compatibility_qualification(PROJECT)
    assert dict(observed) == receipt
