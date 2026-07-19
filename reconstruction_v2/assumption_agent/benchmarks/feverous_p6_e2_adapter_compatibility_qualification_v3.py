"""Aggregate-only FEVEROUS TRAIN/adapter compatibility qualification v3.

The real runner opens only the frozen TRAIN annotation and immutable Wikipedia
resolver, scans every TRAIN record, and exercises the production candidate
adapter.  It returns aggregate receipts only.  It never accepts, creates, or
reads a selection secret; never selects a cohort or fixed corpus; never scores;
and never opens DEV or TEST.  Receipt persistence is intentionally outside this
module so the formal real scan can be executed exactly once and reviewed before
its aggregate receipt is committed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter


VERSION = "feverous_p6_e2_adapter_compatibility_qualification_v3"
SCHEMA = f"{VERSION}_receipt"
MANIFEST_RELATIVE = Path(
    "manifests/feverous_p6_e2_adapter_compatibility_qualification_v3.json"
)
SOURCE_ROOT_RELATIVE = Path("artifacts/feverous_official_source_v1")
ANNOTATION_RELATIVE = SOURCE_ROOT_RELATIVE / formal_source.FROZEN_ANNOTATION_BASENAME
DATABASE_RELATIVE = SOURCE_ROOT_RELATIVE / formal_source.FROZEN_DATABASE_BASENAME
FORMAL_SOURCE_CODE_RELATIVE = Path(
    "assumption_agent/benchmarks/feverous_p6_e2_formal_source_v1.py"
)
SOURCE_ADAPTER_CODE_RELATIVE = Path(
    "assumption_agent/benchmarks/feverous_p6_e2_source_adapter_v1.py"
)
QUALIFICATION_RUNNER_CODE_RELATIVE = Path(
    "assumption_agent/benchmarks/"
    "feverous_p6_e2_adapter_compatibility_qualification_v3.py"
)
WIKIPEDIA_RESOLVER_CODE_RELATIVE = Path(
    "assumption_agent/benchmarks/"
    "feverous_wikipedia_source_qualification_v1.py"
)

# These aggregates were fixed by the pre-action source qualification/topology
# audit.  They contain no source ids, claims, labels, page ids, or evidence ids.
EXPECTED_TRAIN_RECORD_COUNT = 71_292
EXPECTED_BLANK_SENTINEL_RECORD_COUNT = 1
EXPECTED_RAW_EVIDENCE_SET_COUNT = 77_492
EXPECTED_RAW_CONTENT_REFERENCE_COUNT = 349_556
EXPECTED_ADAPTER_EVIDENCE_SET_COUNT = 75_219
EXPECTED_ADAPTER_CONTENT_REFERENCE_COUNT = 338_061
EXPECTED_NONEXACT_TITLE_CONTEXT_SET_COUNT = 2
EXPECTED_NONEXACT_TITLE_CONTEXT_REFERENCE_COUNT = 2
EXPECTED_NONEXACT_TITLE_CONTEXT_RECORD_COUNT = 2


class FeverousAdapterCompatibilityQualificationError(RuntimeError):
    """The aggregate TRAIN topology or exact resolver compatibility drifted."""


@dataclass(frozen=True)
class RawTrainTopology:
    """Content-free counts over all physical TRAIN records."""

    record_count: int
    blank_sentinel_record_count: int
    evidence_set_count: int
    content_reference_count: int


_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "status",
        "source_split",
        "source_binding_sha256",
        "annotation_relative",
        "annotation_file_sha256",
        "database_relative",
        "database_file_sha256",
        "formal_source_version",
        "source_adapter_version",
        "formal_source_file_sha256",
        "source_adapter_file_sha256",
        "qualification_runner_file_sha256",
        "wikipedia_resolver_file_sha256",
        "strict_json_decoder_source_sha256",
        "exact_blank_sentinel_predicate_source_sha256",
        "annotation_aggregate_receipt",
        "adapter_aggregate_receipt",
        "train_record_count",
        "blank_sentinel_record_count",
        "raw_evidence_set_count",
        "raw_content_reference_count",
        "adapter_evidence_set_count",
        "adapter_content_reference_count",
        "invalid_nonexact_title_context_evidence_set_count",
        "invalid_nonexact_title_context_reference_count",
        "records_with_invalid_nonexact_title_context_count",
        "nonexact_title_context_policy",
        "annotation_file_read_count",
        "database_file_hash_pass_count",
        "exact_resolver_scan_completed",
        "raw_id_claim_label_page_or_evidence_persisted",
        "train_records_or_candidate_objects_persisted",
        "selection_secret_generated_or_read",
        "cohort_block_or_canonical_set_selected",
        "fixed_8192_corpus_formed",
        "retrieval_action_utility_evaluator_or_scoring_calls",
        "development_or_test_source_accessed",
        "online_evaluator_calls",
        "qualification_sha256",
    }
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "qualification value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "qualification code file cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _callable_source_sha256(function: Callable[..., object]) -> str:
    try:
        source = inspect.getsource(function).encode("utf-8", errors="strict")
    except (OSError, TypeError, UnicodeEncodeError) as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "qualification loader predicate source is unavailable"
        ) from exc
    return hashlib.sha256(source).hexdigest()


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "project root is unavailable"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise FeverousAdapterCompatibilityQualificationError(
            "project root is unsafe"
        )
    return root


def _raw_train_topology(
    records: Sequence[Mapping[str, Any]],
) -> RawTrainTopology:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise FeverousAdapterCompatibilityQualificationError(
            "TRAIN records are not a finite sequence"
        )
    blank_count = 0
    evidence_set_count = 0
    content_reference_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise FeverousAdapterCompatibilityQualificationError(
                "TRAIN record is not an object"
            )
        if source_adapter._is_blank_sentinel(record):
            blank_count += 1
            continue
        try:
            source_adapter._require_official_record(record)
        except source_adapter.FeverousSourceAdapterError as exc:
            raise FeverousAdapterCompatibilityQualificationError(
                "TRAIN record schema drifted during topology scan"
            ) from exc
        evidence = record.get("evidence")
        assert isinstance(evidence, list)
        evidence_set_count += len(evidence)
        for evidence_set in evidence:
            if not isinstance(evidence_set, Mapping):
                raise FeverousAdapterCompatibilityQualificationError(
                    "TRAIN evidence set is not an object"
                )
            content = evidence_set.get("content")
            if not isinstance(content, list) or any(
                not isinstance(value, str) for value in content
            ):
                raise FeverousAdapterCompatibilityQualificationError(
                    "TRAIN evidence content topology drifted"
                )
            content_reference_count += len(content)
    return RawTrainTopology(
        record_count=len(records),
        blank_sentinel_record_count=blank_count,
        evidence_set_count=evidence_set_count,
        content_reference_count=content_reference_count,
    )


def _validate_aggregate_topology(
    *,
    raw_topology: RawTrainTopology,
    adapter_receipt: Mapping[str, Any],
) -> None:
    if not isinstance(raw_topology, RawTrainTopology):
        raise FeverousAdapterCompatibilityQualificationError(
            "raw TRAIN topology is unavailable"
        )
    try:
        source_adapter.verify_adapter_receipt(adapter_receipt)
    except source_adapter.FeverousSourceAdapterError as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "adapter aggregate receipt drifted"
        ) from exc
    status_counts = adapter_receipt.get("record_status_counts")
    if (
        raw_topology.record_count != EXPECTED_TRAIN_RECORD_COUNT
        or raw_topology.blank_sentinel_record_count
        != EXPECTED_BLANK_SENTINEL_RECORD_COUNT
        or raw_topology.evidence_set_count != EXPECTED_RAW_EVIDENCE_SET_COUNT
        or raw_topology.content_reference_count
        != EXPECTED_RAW_CONTENT_REFERENCE_COUNT
        or adapter_receipt.get("input_record_count")
        != EXPECTED_TRAIN_RECORD_COUNT
        or not isinstance(status_counts, Mapping)
        or status_counts.get("blank_sentinel")
        != EXPECTED_BLANK_SENTINEL_RECORD_COUNT
        or adapter_receipt.get("official_evidence_set_count")
        != EXPECTED_ADAPTER_EVIDENCE_SET_COUNT
        or adapter_receipt.get("official_evidence_reference_count")
        != EXPECTED_ADAPTER_CONTENT_REFERENCE_COUNT
        or adapter_receipt.get("excluded_nonexact_title_context_set_count")
        != EXPECTED_NONEXACT_TITLE_CONTEXT_SET_COUNT
        or adapter_receipt.get(
            "excluded_nonexact_title_context_reference_count"
        )
        != EXPECTED_NONEXACT_TITLE_CONTEXT_REFERENCE_COUNT
        or adapter_receipt.get(
            "records_with_excluded_nonexact_title_context_count"
        )
        != EXPECTED_NONEXACT_TITLE_CONTEXT_RECORD_COUNT
        or adapter_receipt.get("raw_claim_page_or_evidence_serialized") is not False
        or adapter_receipt.get("per_record_or_per_source_digest_serialized")
        is not False
        or adapter_receipt.get("cohort_block_or_canonical_set_selected") is not False
        or adapter_receipt.get("fixed_8192_corpus_formed") is not False
        or adapter_receipt.get("utility_recipe_or_model_accessed") is not False
        or adapter_receipt.get("development_or_test_source_accessed") is not False
        or adapter_receipt.get("online_evaluator_calls") != 0
    ):
        raise FeverousAdapterCompatibilityQualificationError(
            "TRAIN/adapter aggregate differs from preregistered topology"
        )


def _validate_annotation_aggregate(
    annotation_receipt: Mapping[str, Any],
) -> str:
    try:
        declared = formal_source.verify_annotation_receipt(annotation_receipt)
    except formal_source.FeverousFormalSourceError as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "annotation aggregate receipt drifted"
        ) from exc
    if (
        annotation_receipt.get("formal_source") is not True
        or annotation_receipt.get("annotation_basename")
        != formal_source.FROZEN_ANNOTATION_BASENAME
        or annotation_receipt.get("annotation_size_bytes")
        != formal_source.FROZEN_ANNOTATION_SIZE_BYTES
        or annotation_receipt.get("annotation_file_sha256")
        != formal_source.FROZEN_ANNOTATION_SHA256
        or annotation_receipt.get("annotation_nonblank_rows")
        != formal_source.FROZEN_ANNOTATION_NONBLANK_ROWS
        or annotation_receipt.get("annotation_blank_sentinel_rows")
        != formal_source.FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS
    ):
        raise FeverousAdapterCompatibilityQualificationError(
            "annotation aggregate differs from frozen TRAIN"
        )
    return declared


def form_adapter_compatibility_qualification_receipt(
    *,
    project: str | Path,
    annotation_receipt: Mapping[str, Any],
    adapter_receipt: Mapping[str, Any],
    raw_topology: RawTrainTopology,
) -> dict[str, Any]:
    """Form a content-free receipt after complete TRAIN/resolver exhaustion."""

    root = _canonical_project(project)
    annotation_sha = _validate_annotation_aggregate(annotation_receipt)
    _validate_aggregate_topology(
        raw_topology=raw_topology,
        adapter_receipt=adapter_receipt,
    )
    adapter_sha = adapter_receipt.get("adapter_receipt_sha256")
    if not _is_sha256(annotation_sha) or not _is_sha256(adapter_sha):
        raise FeverousAdapterCompatibilityQualificationError(
            "aggregate receipt binding is invalid"
        )
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "real_train_exact_resolver_adapter_compatible_before_v3",
        "source_split": "TRAIN",
        "source_binding_sha256": source_adapter.FROZEN_TRAIN_BINDING.binding_sha256,
        "annotation_relative": ANNOTATION_RELATIVE.as_posix(),
        "annotation_file_sha256": formal_source.FROZEN_ANNOTATION_SHA256,
        "database_relative": DATABASE_RELATIVE.as_posix(),
        "database_file_sha256": formal_source.FROZEN_DATABASE_SHA256,
        "formal_source_version": formal_source.VERSION,
        "source_adapter_version": source_adapter.VERSION,
        "formal_source_file_sha256": _sha256_file(
            root / FORMAL_SOURCE_CODE_RELATIVE
        ),
        "source_adapter_file_sha256": _sha256_file(
            root / SOURCE_ADAPTER_CODE_RELATIVE
        ),
        "qualification_runner_file_sha256": _sha256_file(
            root / QUALIFICATION_RUNNER_CODE_RELATIVE
        ),
        "wikipedia_resolver_file_sha256": _sha256_file(
            root / WIKIPEDIA_RESOLVER_CODE_RELATIVE
        ),
        "strict_json_decoder_source_sha256": _callable_source_sha256(
            formal_source._decode_json_line
        ),
        "exact_blank_sentinel_predicate_source_sha256": (
            _callable_source_sha256(source_adapter._is_blank_sentinel)
        ),
        "annotation_aggregate_receipt": dict(annotation_receipt),
        "adapter_aggregate_receipt": dict(adapter_receipt),
        "train_record_count": raw_topology.record_count,
        "blank_sentinel_record_count": raw_topology.blank_sentinel_record_count,
        "raw_evidence_set_count": raw_topology.evidence_set_count,
        "raw_content_reference_count": raw_topology.content_reference_count,
        "adapter_evidence_set_count": adapter_receipt[
            "official_evidence_set_count"
        ],
        "adapter_content_reference_count": adapter_receipt[
            "official_evidence_reference_count"
        ],
        "invalid_nonexact_title_context_evidence_set_count": adapter_receipt[
            "excluded_nonexact_title_context_set_count"
        ],
        "invalid_nonexact_title_context_reference_count": adapter_receipt[
            "excluded_nonexact_title_context_reference_count"
        ],
        "records_with_invalid_nonexact_title_context_count": adapter_receipt[
            "records_with_excluded_nonexact_title_context_count"
        ],
        "nonexact_title_context_policy": (
            "exclude_whole_set_only_when_exact_page_differs_but_NFD_casefold_matches;"
            "_never_guess_repair_or_accept_set"
        ),
        "annotation_file_read_count": 1,
        "database_file_hash_pass_count": 1,
        "exact_resolver_scan_completed": True,
        "raw_id_claim_label_page_or_evidence_persisted": False,
        "train_records_or_candidate_objects_persisted": False,
        "selection_secret_generated_or_read": False,
        "cohort_block_or_canonical_set_selected": False,
        "fixed_8192_corpus_formed": False,
        "retrieval_action_utility_evaluator_or_scoring_calls": 0,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "qualification_sha256": stable_hash(body)}


def validate_adapter_compatibility_qualification_receipt(
    receipt: Mapping[str, Any], *, project: str | Path
) -> str:
    """Validate one aggregate receipt without reopening TRAIN or the database."""

    root = _canonical_project(project)
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise FeverousAdapterCompatibilityQualificationError(
            "compatibility qualification receipt schema drifted"
        )
    body = dict(receipt)
    declared = body.pop("qualification_sha256", None)
    annotation_receipt = receipt.get("annotation_aggregate_receipt")
    adapter_receipt = receipt.get("adapter_aggregate_receipt")
    if not isinstance(annotation_receipt, Mapping) or not isinstance(
        adapter_receipt, Mapping
    ):
        raise FeverousAdapterCompatibilityQualificationError(
            "nested aggregate receipt is unavailable"
        )
    raw_topology = RawTrainTopology(
        record_count=receipt.get("train_record_count", -1),
        blank_sentinel_record_count=receipt.get(
            "blank_sentinel_record_count", -1
        ),
        evidence_set_count=receipt.get("raw_evidence_set_count", -1),
        content_reference_count=receipt.get("raw_content_reference_count", -1),
    )
    annotation_sha = _validate_annotation_aggregate(annotation_receipt)
    _validate_aggregate_topology(
        raw_topology=raw_topology,
        adapter_receipt=adapter_receipt,
    )
    if (
        not _is_sha256(declared)
        or stable_hash(body) != declared
        or receipt.get("schema") != SCHEMA
        or receipt.get("version") != VERSION
        or receipt.get("status")
        != "real_train_exact_resolver_adapter_compatible_before_v3"
        or receipt.get("source_split") != "TRAIN"
        or receipt.get("source_binding_sha256")
        != source_adapter.FROZEN_TRAIN_BINDING.binding_sha256
        or receipt.get("annotation_relative") != ANNOTATION_RELATIVE.as_posix()
        or receipt.get("annotation_file_sha256")
        != formal_source.FROZEN_ANNOTATION_SHA256
        or receipt.get("database_relative") != DATABASE_RELATIVE.as_posix()
        or receipt.get("database_file_sha256")
        != formal_source.FROZEN_DATABASE_SHA256
        or receipt.get("formal_source_version") != formal_source.VERSION
        or receipt.get("source_adapter_version") != source_adapter.VERSION
        or receipt.get("formal_source_file_sha256")
        != _sha256_file(root / FORMAL_SOURCE_CODE_RELATIVE)
        or receipt.get("source_adapter_file_sha256")
        != _sha256_file(root / SOURCE_ADAPTER_CODE_RELATIVE)
        or receipt.get("qualification_runner_file_sha256")
        != _sha256_file(root / QUALIFICATION_RUNNER_CODE_RELATIVE)
        or receipt.get("wikipedia_resolver_file_sha256")
        != _sha256_file(root / WIKIPEDIA_RESOLVER_CODE_RELATIVE)
        or receipt.get("strict_json_decoder_source_sha256")
        != _callable_source_sha256(formal_source._decode_json_line)
        or receipt.get("exact_blank_sentinel_predicate_source_sha256")
        != _callable_source_sha256(source_adapter._is_blank_sentinel)
        or annotation_receipt.get("annotation_receipt_sha256") != annotation_sha
        or receipt.get("adapter_evidence_set_count")
        != EXPECTED_ADAPTER_EVIDENCE_SET_COUNT
        or receipt.get("adapter_content_reference_count")
        != EXPECTED_ADAPTER_CONTENT_REFERENCE_COUNT
        or receipt.get("invalid_nonexact_title_context_evidence_set_count")
        != EXPECTED_NONEXACT_TITLE_CONTEXT_SET_COUNT
        or receipt.get("invalid_nonexact_title_context_reference_count")
        != EXPECTED_NONEXACT_TITLE_CONTEXT_REFERENCE_COUNT
        or receipt.get("records_with_invalid_nonexact_title_context_count")
        != EXPECTED_NONEXACT_TITLE_CONTEXT_RECORD_COUNT
        or receipt.get("nonexact_title_context_policy")
        != (
            "exclude_whole_set_only_when_exact_page_differs_but_NFD_casefold_matches;"
            "_never_guess_repair_or_accept_set"
        )
        or receipt.get("annotation_file_read_count") != 1
        or receipt.get("database_file_hash_pass_count") != 1
        or receipt.get("exact_resolver_scan_completed") is not True
        or receipt.get("raw_id_claim_label_page_or_evidence_persisted") is not False
        or receipt.get("train_records_or_candidate_objects_persisted") is not False
        or receipt.get("selection_secret_generated_or_read") is not False
        or receipt.get("cohort_block_or_canonical_set_selected") is not False
        or receipt.get("fixed_8192_corpus_formed") is not False
        or receipt.get("retrieval_action_utility_evaluator_or_scoring_calls") != 0
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("online_evaluator_calls") != 0
    ):
        raise FeverousAdapterCompatibilityQualificationError(
            "compatibility qualification receipt drifted"
        )
    return str(declared)


def run_real_adapter_compatibility_qualification(
    project: str | Path,
) -> Mapping[str, Any]:
    """Perform the sole full real TRAIN/resolver scan and return aggregates."""

    root = _canonical_project(project)
    with formal_source.ControlledTrainSource(
        annotation_path=root / ANNOTATION_RELATIVE,
        database_path=root / DATABASE_RELATIVE,
    ) as source:
        records = source.read_annotations_once()
        raw_topology = _raw_train_topology(records)
        annotation_receipt = dict(source.annotation_receipt)
        resolver = source.exact_resolver_for_candidate_screen()
        batch = source_adapter.adapt_train_candidate_records(
            records,
            source_split="TRAIN",
            resolver=resolver,
            binding=source_adapter.FROZEN_TRAIN_BINDING,
        )
        adapter_receipt = dict(batch.receipt)
        del batch
        del records
    receipt = form_adapter_compatibility_qualification_receipt(
        project=root,
        annotation_receipt=annotation_receipt,
        adapter_receipt=adapter_receipt,
        raw_topology=raw_topology,
    )
    validate_adapter_compatibility_qualification_receipt(receipt, project=root)
    return receipt


def verify_adapter_compatibility_qualification(
    project: str | Path,
) -> Mapping[str, Any]:
    """Verify the future committed aggregate receipt without a real rescan."""

    root = _canonical_project(project)
    path = root / MANIFEST_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise FeverousAdapterCompatibilityQualificationError(
            "compatibility qualification receipt is unavailable"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousAdapterCompatibilityQualificationError(
            "compatibility qualification receipt is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_bytes(value) + b"\n"
    ):
        raise FeverousAdapterCompatibilityQualificationError(
            "compatibility qualification receipt is noncanonical"
        )
    validate_adapter_compatibility_qualification_receipt(value, project=root)
    return value


__all__ = [
    "ANNOTATION_RELATIVE",
    "DATABASE_RELATIVE",
    "EXPECTED_ADAPTER_CONTENT_REFERENCE_COUNT",
    "EXPECTED_ADAPTER_EVIDENCE_SET_COUNT",
    "EXPECTED_BLANK_SENTINEL_RECORD_COUNT",
    "EXPECTED_NONEXACT_TITLE_CONTEXT_RECORD_COUNT",
    "EXPECTED_NONEXACT_TITLE_CONTEXT_REFERENCE_COUNT",
    "EXPECTED_NONEXACT_TITLE_CONTEXT_SET_COUNT",
    "EXPECTED_RAW_CONTENT_REFERENCE_COUNT",
    "EXPECTED_RAW_EVIDENCE_SET_COUNT",
    "EXPECTED_TRAIN_RECORD_COUNT",
    "FeverousAdapterCompatibilityQualificationError",
    "MANIFEST_RELATIVE",
    "RawTrainTopology",
    "SCHEMA",
    "VERSION",
    "form_adapter_compatibility_qualification_receipt",
    "run_real_adapter_compatibility_qualification",
    "stable_hash",
    "validate_adapter_compatibility_qualification_receipt",
    "verify_adapter_compatibility_qualification",
]
