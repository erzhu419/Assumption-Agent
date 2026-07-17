"""Aggregate-only qualification of the frozen ContractNLI TRAIN source.

The public entry point delegates to a clean ``python -I`` worker.  The worker
lists the ZIP central directory, opens only ``contract-nli/train.json``, and
emits fixed-schema aggregate diagnostics.  It never opens DEV, TEST, raw
contracts, a selection secret, or an individual output row.

Malformed documents are counted and excluded.  Archive, root JSON, duplicate
JSON keys, and aggregate-redaction failures remain fatal infrastructure errors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import heapq
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any
import unicodedata
import zipfile
import zlib


VERSION = "contractnli_fresh_source_qualification_v1"
SCHEMA = VERSION
SOURCE_RELEASE = "ContractNLI_official_commit_eced6528_TRAIN_only"
QUALIFICATION_CLASS = "outcome_blind_row_free_aggregate_source_qualification"
FORMAL_ATTEMPT_MARKER_RELATIVE_PATH = (
    "artifacts/contractnli_graph_evaluator_custody_v1/"
    "source_qualification_attempt_v1.marker"
)

TRAIN_MEMBER = "contract-nli/train.json"
FORMAL_DATA_COMMIT = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
FORMAL_ARCHIVE_SHA256 = (
    "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"
)
FORMAL_ARCHIVE_SIZE = 65_362_913
FORMAL_ARCHIVE_GIT_BLOB = "757fd1dafd29a997fba00c60c6d40b2930a36159"
FORMAL_TRAIN_SIZE = 7_608_211
FORMAL_TRAIN_CRC32 = "7788f20e"

FORMAL_CUSTODY_SCHEMA = "contractnli_graph_evaluator_source_custody_v1"
FORMAL_CUSTODY_COMMIT = "f3bdc201a93b724bf9c017f6c426c7f6505e5430"
FORMAL_CUSTODY_FILE_SHA256 = (
    "e635ee6257ea54d02f739a82aac17b2e5f4e755eff798f708596519f8b79764f"
)
FORMAL_CUSTODY_SHA256 = (
    "9bbb3d09fc474240fa4631974582da31613c16d70d33b306cd2c8dbeaadbcc4e"
)
FORMAL_ADDENDUM_SCHEMA = "contractnli_source_access_addendum_v1"
FORMAL_ADDENDUM_COMMIT = "d5636ec4570a2d5ba851a26ca9a02a19dc53e3a6"
FORMAL_ADDENDUM_FILE_SHA256 = (
    "518031ef43fd6222fbce3b358970bf3bf101d38e5d6347af96a868805e455637"
)
FORMAL_ADDENDUM_SHA256 = (
    "c9dc2bb8c94faaa9a540747b22f7479a2a0f9d83831ff882e9abe244c42aa693"
)
FORMAL_MEMBER_SCHEMA = "contractnli_source_member_binding_v1"
FORMAL_MEMBER_BINDING_COMMIT = "fdd1c57a05e86ab2e0873dc3053245a024d3c237"
FORMAL_MEMBER_FILE_SHA256 = (
    "2c4230aab716db315863f749b57f3401b11fba5c2d3b104005cabfe6e201cb97"
)
FORMAL_MEMBER_BINDING_SHA256 = (
    "1f0e48d06232cfe003c51dd754e30f156a56cb6164e5e63dfeb1adf625723f5b"
)

FORMAL_READER_COMMIT = "058c56fd62d56897bb4fcfbf1be71b17aee3a79c"
FORMAL_READER_ARCHIVE_SHA256 = (
    "24c78d09c31d4e75ab600b7f18d02204021ffe3f19a648138c0153e90fb8ec25"
)
FORMAL_READER_ARCHIVE_SIZE = 35_860
FORMAL_READER_DATASET_SHA256 = (
    "21ff58e1b63c8ee1f18632385c6ee8c36c352d6312746db4420f007f8fc19087"
)
FORMAL_READER_LOADER_SHA256 = (
    "9c63102a01576c230da94695f593fd7bd4fe2d0b20672b74e8a7f62aba426fc4"
)

MIN_NODES = 18
MAX_NODES = 128
MIN_GOLD = 1
MAX_GOLD = 5
MIN_CONTENT_GROUPS = 256
LABEL_COUNT = 17
MAX_ARCHIVE_MEMBERS = 100_000
MAX_TRAIN_BYTES = 64 * 1024 * 1024
CHOICES = frozenset({"Entailment", "Contradiction", "NotMentioned"})
DOCUMENT_TYPES = frozenset({"search-pdf", "sec-text", "sec-html"})

FULL_TEXT_SIGNATURES = (
    "confidential information: means all confidential information (however recorded, preserved or disclosed) disclosed by a party or its representatives",
    "brooks technical solutions, inc.",
    "brookstech",
    "recipient shall not disclose confidential information to any person or entity, except its employees or partners",
    "the receiving party undertakes to permit access to the confidential information only to its representatives",
    "representatives shall mean directors, employees, professional advisors or anyone involved with the party in a professional or business capacity",
)
DENYLIST_FILE_NAME = "example.pdf"
DENYLIST_URL_SUBSTRING = "examplecontract.com"

ANOMALY_KINDS = (
    "document_not_object",
    "required_field_type",
    "duplicate_document_id",
    "invalid_document_type",
    "invalid_span",
    "invalid_annotation_set",
    "invalid_annotation_keys",
    "invalid_annotation",
    "invalid_choice",
    "invalid_choice_span_consistency",
    "invalid_gold_index",
)

STATUS_QUALIFIED = "source_qualified_for_frozen_content_group_capacity"
STATUS_INFEASIBLE = "terminal_source_infeasible_for_frozen_content_group_capacity"
STATUS_DIAGNOSTIC = "synthetic_or_nonformal_aggregate_diagnostic"
SELECTION_STATUS = "not_performed"

HEX_RE = re.compile(r"(?:[0-9a-f]{8}|[0-9a-f]{40}|[0-9a-f]{64})\Z")
PUBLIC_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")


class ContractNliQualificationError(RuntimeError):
    """The source or public qualification receipt violated the contract."""


class DocumentAnomaly(RuntimeError):
    """A private document is invalid and must be counted, not serialized."""

    def __init__(self, kind: str):
        if kind not in ANOMALY_KINDS:
            raise AssertionError(kind)
        self.kind = kind
        super().__init__(kind)


@dataclass(frozen=True)
class DocumentRecord:
    """Private per-document state; instances are never serialized."""

    node_count: int
    eligible_item_count: int
    normalized_text_sha256: str
    duplicate_node_text_count: int
    shared_start_count: int
    repeated_boundary_count: int
    overlapping_span_pair_count: int
    duplicate_gold_index_count: int
    full_text_signature_exposed: bool
    metadata_file_exposed: bool
    metadata_url_exposed: bool

    @property
    def node_eligible(self) -> bool:
        return MIN_NODES <= self.node_count <= MAX_NODES

    @property
    def exposed(self) -> bool:
        return (
            self.full_text_signature_exposed
            or self.metadata_file_exposed
            or self.metadata_url_exposed
        )

    @property
    def population_eligible(self) -> bool:
        return self.node_eligible and self.eligible_item_count > 0 and not self.exposed


def _canonical_json(payload: Mapping[str, Any], *, ensure_ascii: bool = True) -> str:
    return json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_no_duplicate_keys(raw: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ContractNliQualificationError(f"{label} contains a non-finite number")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractNliQualificationError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except ContractNliQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractNliQualificationError(f"{label} is not strict JSON") from exc


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractNliQualificationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractNliQualificationError(f"{label} must be a non-symlink regular file")


def _hash_archive(path: Path) -> tuple[str, int, str]:
    _require_regular_file(path, label="archive")
    try:
        size = path.stat().st_size
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1()
        git_blob.update(f"blob {size}\0".encode("ascii"))
        observed = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)
                git_blob.update(chunk)
                observed += len(chunk)
    except OSError as exc:
        raise ContractNliQualificationError("archive hashing failed") from exc
    if observed != size:
        raise ContractNliQualificationError("archive changed while hashing")
    return sha256.hexdigest(), size, git_blob.hexdigest()


def _read_manifest(
    path: Path,
    *,
    schema: str,
    hash_field: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    _require_regular_file(path, label="manifest")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractNliQualificationError("manifest read failed") from exc
    payload = _json_no_duplicate_keys(raw, label="manifest")
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ContractNliQualificationError("manifest schema mismatch")
    declared = payload.get(hash_field)
    if not isinstance(declared, str) or HEX_RE.fullmatch(declared) is None:
        raise ContractNliQualificationError("manifest semantic hash is invalid")
    body = dict(payload)
    del body[hash_field]
    observed = _semantic_hash(body)
    if observed != declared:
        raise ContractNliQualificationError("manifest semantic hash mismatch")
    return payload, {
        "schema": schema,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
        "semantic_sha256": observed,
    }


def normalize_full_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise DocumentAnomaly("required_field_type")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise DocumentAnomaly("required_field_type") from exc
    return value


def _validate_root(payload: Any) -> tuple[list[Any], Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ContractNliQualificationError("TRAIN root must be an object")
    documents = payload.get("documents")
    labels = payload.get("labels")
    if not isinstance(documents, list) or not isinstance(labels, Mapping):
        raise ContractNliQualificationError("TRAIN root fields have invalid types")
    if len(labels) != LABEL_COUNT:
        raise ContractNliQualificationError("TRAIN label ontology must contain 17 entries")
    for label_id, label_value in labels.items():
        if not isinstance(label_id, str) or not isinstance(label_value, Mapping):
            raise ContractNliQualificationError("TRAIN label ontology is malformed")
        hypothesis = label_value.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ContractNliQualificationError("TRAIN label hypothesis is malformed")
        try:
            hypothesis.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ContractNliQualificationError(
                "TRAIN label hypothesis is malformed"
            ) from exc
    return documents, labels


def _parse_document(
    document: Any,
    *,
    label_keys: tuple[str, ...],
    duplicate_ids: frozenset[int],
) -> DocumentRecord:
    if not isinstance(document, Mapping):
        raise DocumentAnomaly("document_not_object")
    document_id = document.get("id")
    if type(document_id) is not int:
        raise DocumentAnomaly("required_field_type")
    if document_id in duplicate_ids:
        raise DocumentAnomaly("duplicate_document_id")
    text = _require_string(document, "text")
    file_name = _require_string(document, "file_name")
    url = _require_string(document, "url")
    document_type = _require_string(document, "document_type")
    spans_raw = document.get("spans")
    annotation_sets = document.get("annotation_sets")
    if not isinstance(spans_raw, list) or not isinstance(annotation_sets, list):
        raise DocumentAnomaly("required_field_type")
    if document_type not in DOCUMENT_TYPES:
        raise DocumentAnomaly("invalid_document_type")

    spans: list[tuple[int, int]] = []
    for span_value in spans_raw:
        if (
            not isinstance(span_value, list)
            or len(span_value) != 2
            or type(span_value[0]) is not int
            or type(span_value[1]) is not int
        ):
            raise DocumentAnomaly("invalid_span")
        start, end = span_value
        if not (0 <= start < end <= len(text)):
            raise DocumentAnomaly("invalid_span")
        spans.append((start, end))

    if len(annotation_sets) != 1 or not isinstance(annotation_sets[0], Mapping):
        raise DocumentAnomaly("invalid_annotation_set")
    annotations = annotation_sets[0].get("annotations")
    if not isinstance(annotations, Mapping):
        raise DocumentAnomaly("invalid_annotation_set")
    if frozenset(annotations.keys()) != frozenset(label_keys):
        raise DocumentAnomaly("invalid_annotation_keys")

    eligible_item_count = 0
    duplicate_gold_index_count = 0
    for label_key in label_keys:
        annotation = annotations[label_key]
        if not isinstance(annotation, Mapping):
            raise DocumentAnomaly("invalid_annotation")
        choice = annotation.get("choice")
        gold_raw = annotation.get("spans")
        if not isinstance(choice, str) or not isinstance(gold_raw, list):
            raise DocumentAnomaly("invalid_annotation")
        if choice not in CHOICES:
            raise DocumentAnomaly("invalid_choice")
        if any(type(index) is not int for index in gold_raw):
            raise DocumentAnomaly("invalid_gold_index")
        if any(index < 0 or index >= len(spans) for index in gold_raw):
            raise DocumentAnomaly("invalid_gold_index")
        if choice == "NotMentioned" and gold_raw:
            raise DocumentAnomaly("invalid_choice_span_consistency")
        distinct_gold = set(gold_raw)
        duplicate_gold_index_count += len(gold_raw) - len(distinct_gold)
        if choice in {"Entailment", "Contradiction"} and MIN_GOLD <= len(distinct_gold) <= MAX_GOLD:
            eligible_item_count += 1

    node_texts = [text[start:end] for start, end in spans]
    starts = [start for start, _ in spans]
    repeated_boundary_count = len(spans) - len(set(spans))
    overlapping_pairs = 0
    active_ends: list[int] = []
    for start, end in sorted(spans):
        while active_ends and active_ends[0] <= start:
            heapq.heappop(active_ends)
        overlapping_pairs += len(active_ends)
        heapq.heappush(active_ends, end)

    normalized_text = normalize_full_text(text)
    normalized_url = normalize_full_text(url)
    normalized_file = file_name.casefold()
    return DocumentRecord(
        node_count=len(spans),
        eligible_item_count=eligible_item_count,
        normalized_text_sha256=hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        duplicate_node_text_count=len(node_texts) - len(set(node_texts)),
        shared_start_count=len(starts) - len(set(starts)),
        repeated_boundary_count=repeated_boundary_count,
        overlapping_span_pair_count=overlapping_pairs,
        duplicate_gold_index_count=duplicate_gold_index_count,
        full_text_signature_exposed=any(
            signature in normalized_text for signature in FULL_TEXT_SIGNATURES
        ),
        metadata_file_exposed=normalized_file == DENYLIST_FILE_NAME,
        metadata_url_exposed=DENYLIST_URL_SUBSTRING in normalized_url,
    )


def _aggregate_train(payload: Any) -> dict[str, Any]:
    documents, labels = _validate_root(payload)
    id_counts: Counter[int] = Counter(
        document.get("id")
        for document in documents
        if isinstance(document, Mapping) and type(document.get("id")) is int
    )
    duplicate_ids = frozenset(key for key, count in id_counts.items() if count > 1)
    anomaly_counts: Counter[str] = Counter()
    records: list[DocumentRecord] = []
    label_keys = tuple(labels.keys())
    for document in documents:
        try:
            records.append(
                _parse_document(
                    document,
                    label_keys=label_keys,
                    duplicate_ids=duplicate_ids,
                )
            )
        except DocumentAnomaly as exc:
            anomaly_counts[exc.kind] += 1

    groups = Counter(
        record.normalized_text_sha256
        for record in records
        if record.population_eligible
    )
    node_counts = [record.node_count for record in records]
    node_eligible = [record for record in records if record.node_eligible]
    candidate_docs = [record for record in node_eligible if record.eligible_item_count > 0]
    eligible_docs = [record for record in candidate_docs if not record.exposed]
    duplicate_groups = [count for count in groups.values() if count > 1]

    return {
        "root": {
            "label_count": len(labels),
            "document_count": len(documents),
            "valid_document_count": len(records),
            "invalid_document_count": len(documents) - len(records),
        },
        "document_anomalies": {
            kind: anomaly_counts[kind] for kind in ANOMALY_KINDS
        },
        "addressable_graph": {
            "valid_document_node_count_total": sum(node_counts),
            "valid_document_node_count_min": min(node_counts) if node_counts else 0,
            "valid_document_node_count_max": max(node_counts) if node_counts else 0,
            "node_eligible_document_count": len(node_eligible),
            "duplicate_node_text_count": sum(
                record.duplicate_node_text_count for record in records
            ),
            "shared_start_count": sum(record.shared_start_count for record in records),
            "repeated_boundary_count": sum(
                record.repeated_boundary_count for record in records
            ),
            "overlapping_span_pair_count": sum(
                record.overlapping_span_pair_count for record in records
            ),
        },
        "eligibility": {
            "eligible_item_count_before_one_per_document_cap": sum(
                record.eligible_item_count for record in node_eligible
            ),
            "duplicate_gold_index_count": sum(
                record.duplicate_gold_index_count for record in records
            ),
            "document_with_eligible_item_count": len(candidate_docs),
            "exposure_excluded_document_count": sum(
                record.exposed for record in candidate_docs
            ),
            "full_text_signature_excluded_document_count": sum(
                record.full_text_signature_exposed for record in candidate_docs
            ),
            "metadata_file_excluded_document_count": sum(
                record.metadata_file_exposed for record in candidate_docs
            ),
            "metadata_url_excluded_document_count": sum(
                record.metadata_url_exposed for record in candidate_docs
            ),
            "eligible_document_count_after_exposure": len(eligible_docs),
            "eligible_normalized_content_group_count": len(groups),
            "duplicate_content_group_count": len(duplicate_groups),
            "documents_in_duplicate_content_groups": sum(duplicate_groups),
            "maximum_content_group_size": max(groups.values()) if groups else 0,
            "minimum_required_content_groups": MIN_CONTENT_GROUPS,
            "capacity_satisfied": len(groups) >= MIN_CONTENT_GROUPS,
        },
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ContractNliQualificationError("formal manifest binding is incomplete")
        value = value[key]
    return value


def _expect(mapping: Mapping[str, Any], keys: tuple[str, ...], expected: Any) -> None:
    if _nested(mapping, *keys) != expected:
        raise ContractNliQualificationError("formal manifest binding mismatch")


def _validate_formal_manifests(
    custody: Mapping[str, Any],
    custody_receipt: Mapping[str, Any],
    addendum: Mapping[str, Any],
    addendum_receipt: Mapping[str, Any],
    member: Mapping[str, Any],
    member_receipt: Mapping[str, Any],
) -> None:
    if custody_receipt != {
        "schema": FORMAL_CUSTODY_SCHEMA,
        "file_sha256": FORMAL_CUSTODY_FILE_SHA256,
        "byte_size": custody_receipt.get("byte_size"),
        "semantic_sha256": FORMAL_CUSTODY_SHA256,
    }:
        raise ContractNliQualificationError("formal custody manifest mismatch")
    if addendum_receipt != {
        "schema": FORMAL_ADDENDUM_SCHEMA,
        "file_sha256": FORMAL_ADDENDUM_FILE_SHA256,
        "byte_size": addendum_receipt.get("byte_size"),
        "semantic_sha256": FORMAL_ADDENDUM_SHA256,
    }:
        raise ContractNliQualificationError("formal addendum manifest mismatch")
    if member_receipt != {
        "schema": FORMAL_MEMBER_SCHEMA,
        "file_sha256": FORMAL_MEMBER_FILE_SHA256,
        "byte_size": member_receipt.get("byte_size"),
        "semantic_sha256": FORMAL_MEMBER_BINDING_SHA256,
    }:
        raise ContractNliQualificationError("formal member manifest mismatch")

    _expect(
        custody,
        ("official_source_contract", "data_repository", "fixed_commit"),
        FORMAL_DATA_COMMIT,
    )
    _expect(
        custody,
        ("official_source_contract", "source_archive", "expected_content_length_from_HEAD"),
        FORMAL_ARCHIVE_SIZE,
    )
    _expect(
        custody,
        ("official_source_contract", "source_archive", "git_blob"),
        FORMAL_ARCHIVE_GIT_BLOB,
    )
    _expect(
        custody,
        ("official_source_contract", "baseline_repository", "fixed_commit"),
        FORMAL_READER_COMMIT,
    )
    _expect(
        custody,
        ("qualification_and_population_contract", "minimum_eligible_TRAIN_content_groups_after_representative_selection"),
        MIN_CONTENT_GROUPS,
    )
    _expect(
        custody,
        ("exposure_and_exclusion_contract", "whole_document_normalized_substring_signatures_v1"),
        list(FULL_TEXT_SIGNATURES),
    )
    _expect(
        custody,
        ("exposure_and_exclusion_contract", "synthetic_literal_metadata_denylist", "file_name_casefolded"),
        [DENYLIST_FILE_NAME],
    )
    _expect(
        custody,
        ("exposure_and_exclusion_contract", "synthetic_literal_metadata_denylist", "URL_normalized_substrings"),
        [DENYLIST_URL_SUBSTRING],
    )

    _expect(
        addendum,
        ("official_dataset_archive_binding", "archive_sha256"),
        FORMAL_ARCHIVE_SHA256,
    )
    _expect(
        addendum,
        ("official_dataset_archive_binding", "byte_size"),
        FORMAL_ARCHIVE_SIZE,
    )
    _expect(
        addendum,
        ("official_dataset_archive_binding", "git_blob"),
        FORMAL_ARCHIVE_GIT_BLOB,
    )
    _expect(
        addendum,
        ("official_dataset_archive_binding", "fixed_commit"),
        FORMAL_DATA_COMMIT,
    )
    _expect(
        addendum,
        ("custody_v1_binding", "canonical_custody_sha256"),
        FORMAL_CUSTODY_SHA256,
    )
    _expect(
        addendum,
        ("custody_v1_binding", "custody_file_sha256"),
        FORMAL_CUSTODY_FILE_SHA256,
    )
    _expect(addendum, ("custody_v1_binding", "commit"), FORMAL_CUSTODY_COMMIT)
    _expect(
        addendum,
        ("official_reader_code_archive_binding", "fixed_commit"),
        FORMAL_READER_COMMIT,
    )
    _expect(
        addendum,
        ("official_reader_code_archive_binding", "archive_sha256"),
        FORMAL_READER_ARCHIVE_SHA256,
    )
    _expect(
        addendum,
        ("official_reader_code_archive_binding", "byte_size"),
        FORMAL_READER_ARCHIVE_SIZE,
    )
    opened_code_members = _nested(
        addendum, "official_reader_code_archive_binding", "opened_code_members"
    )
    if not isinstance(opened_code_members, list) or len(opened_code_members) != 2:
        raise ContractNliQualificationError("official reader member binding mismatch")
    reader_hashes: dict[str, str] = {}
    for value in opened_code_members:
        if not isinstance(value, Mapping):
            raise ContractNliQualificationError("official reader member binding mismatch")
        archive_member = value.get("archive_member")
        member_sha = value.get("sha256")
        if not isinstance(archive_member, str) or not isinstance(member_sha, str):
            raise ContractNliQualificationError("official reader member binding mismatch")
        if archive_member.endswith("/contract_nli/dataset/dataset.py"):
            reader_hashes["dataset"] = member_sha
        elif archive_member.endswith("/contract_nli/dataset/loader.py"):
            reader_hashes["loader"] = member_sha
        else:
            raise ContractNliQualificationError("official reader member binding mismatch")
    if reader_hashes != {
        "dataset": FORMAL_READER_DATASET_SHA256,
        "loader": FORMAL_READER_LOADER_SHA256,
    }:
        raise ContractNliQualificationError("official reader member hash mismatch")

    _expect(member, ("archive_binding", "sha256"), FORMAL_ARCHIVE_SHA256)
    _expect(member, ("archive_binding", "byte_size"), FORMAL_ARCHIVE_SIZE)
    _expect(member, ("member_selection", "TRAIN_exact_member"), TRAIN_MEMBER)
    _expect(
        member,
        ("member_selection", "TRAIN_declared_uncompressed_size"),
        FORMAL_TRAIN_SIZE,
    )
    _expect(
        member,
        ("member_selection", "TRAIN_crc32_from_central_directory"),
        FORMAL_TRAIN_CRC32,
    )
    _expect(member, ("member_selection", "TRAIN_basename_count"), 1)
    _expect(
        member,
        ("source_access_addendum_binding", "addendum_sha256"),
        FORMAL_ADDENDUM_SHA256,
    )
    addendum_commit = _nested(
        member, "source_access_addendum_binding", "commit"
    )
    if not isinstance(addendum_commit, str) or not FORMAL_ADDENDUM_COMMIT.startswith(
        addendum_commit
    ):
        raise ContractNliQualificationError("member addendum commit binding mismatch")


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise ContractNliQualificationError("ZIP contains an unsafe member path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part == ".." for part in name.split("/"))
    ):
        raise ContractNliQualificationError("ZIP contains an unsafe member path")
    if info.flag_bits & 0x1:
        raise ContractNliQualificationError("ZIP contains an encrypted member")
    if info.file_size < 0:
        raise ContractNliQualificationError("ZIP member size is invalid")
    if info.create_system == 3:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode:
            file_kind = stat.S_IFMT(unix_mode)
            allowed_kind = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
            if file_kind not in {0, allowed_kind}:
                raise ContractNliQualificationError("ZIP contains a nonregular member")


def _read_train_member(
    path: Path,
    *,
    formal: bool,
    initial_archive_binding: tuple[str, int, str],
) -> tuple[bytes, dict[str, Any]]:
    try:
        with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ContractNliQualificationError("ZIP member count exceeds the limit")
            names: set[str] = set()
            train_basename_members: list[zipfile.ZipInfo] = []
            for info in members:
                _validate_zip_info(info)
                if info.filename in names:
                    raise ContractNliQualificationError("ZIP contains duplicate member paths")
                names.add(info.filename)
                if not info.is_dir() and PurePosixPath(info.filename).name == "train.json":
                    train_basename_members.append(info)
            if len(train_basename_members) != 1:
                raise ContractNliQualificationError("ZIP TRAIN basename is not unique")
            train_info = train_basename_members[0]
            if train_info.filename != TRAIN_MEMBER or train_info.is_dir():
                raise ContractNliQualificationError("ZIP exact TRAIN member is missing")
            if train_info.file_size > MAX_TRAIN_BYTES:
                raise ContractNliQualificationError("TRAIN member exceeds the byte limit")
            train_crc = f"{train_info.CRC & 0xFFFFFFFF:08x}"
            if formal and (
                train_info.file_size != FORMAL_TRAIN_SIZE
                or train_crc != FORMAL_TRAIN_CRC32
            ):
                raise ContractNliQualificationError("formal TRAIN member binding mismatch")
            chunks: list[bytes] = []
            observed_size = 0
            observed_crc = 0
            observed_sha256 = hashlib.sha256()
            with archive.open(train_info, "r") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    observed_size += len(chunk)
                    if observed_size > MAX_TRAIN_BYTES:
                        raise ContractNliQualificationError("TRAIN member exceeds the byte limit")
                    observed_crc = zlib.crc32(chunk, observed_crc)
                    observed_sha256.update(chunk)
                    chunks.append(chunk)
            if observed_size != train_info.file_size:
                raise ContractNliQualificationError("TRAIN member size mismatch")
            if f"{observed_crc & 0xFFFFFFFF:08x}" != train_crc:
                raise ContractNliQualificationError("TRAIN member CRC mismatch")
    except ContractNliQualificationError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        OSError,
    ) as exc:
        raise ContractNliQualificationError("ZIP reader failed") from exc

    if _hash_archive(path) != initial_archive_binding:
        raise ContractNliQualificationError("archive changed during qualification")
    return b"".join(chunks), {
        "member_count": len(members),
        "train_member": TRAIN_MEMBER,
        "train_declared_byte_size": train_info.file_size,
        "train_crc32": train_crc,
        "train_member_sha256": observed_sha256.hexdigest(),
        "archive_stable_after_train_read": True,
        "archive_extracted": False,
    }


def build_qualification(
    archive_path: str | Path,
    custody_manifest_path: str | Path,
    source_access_addendum_path: str | Path,
    source_member_binding_path: str | Path,
    *,
    enforce_formal_bindings: bool = False,
) -> dict[str, Any]:
    """Build one row-free receipt.  This function is intended for the worker."""

    archive_path = Path(archive_path).absolute()
    custody_manifest_path = Path(custody_manifest_path).absolute()
    source_access_addendum_path = Path(source_access_addendum_path).absolute()
    source_member_binding_path = Path(source_member_binding_path).absolute()

    archive_binding = _hash_archive(archive_path)
    custody, custody_receipt = _read_manifest(
        custody_manifest_path,
        schema=FORMAL_CUSTODY_SCHEMA,
        hash_field="custody_sha256",
    )
    addendum, addendum_receipt = _read_manifest(
        source_access_addendum_path,
        schema=FORMAL_ADDENDUM_SCHEMA,
        hash_field="addendum_sha256",
    )
    member, member_receipt = _read_manifest(
        source_member_binding_path,
        schema=FORMAL_MEMBER_SCHEMA,
        hash_field="source_member_binding_sha256",
    )

    if enforce_formal_bindings:
        if archive_binding != (
            FORMAL_ARCHIVE_SHA256,
            FORMAL_ARCHIVE_SIZE,
            FORMAL_ARCHIVE_GIT_BLOB,
        ):
            raise ContractNliQualificationError("formal outer archive binding mismatch")
        _validate_formal_manifests(
            custody,
            custody_receipt,
            addendum,
            addendum_receipt,
            member,
            member_receipt,
        )

    train_raw, zip_receipt = _read_train_member(
        archive_path,
        formal=enforce_formal_bindings,
        initial_archive_binding=archive_binding,
    )
    train_payload = _json_no_duplicate_keys(train_raw, label="TRAIN member")
    aggregate = _aggregate_train(train_payload)
    capacity = aggregate["eligibility"]["capacity_satisfied"]
    if enforce_formal_bindings:
        status = STATUS_QUALIFIED if capacity else STATUS_INFEASIBLE
    else:
        status = STATUS_DIAGNOSTIC

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "source_release": SOURCE_RELEASE,
        "qualification_class": QUALIFICATION_CLASS,
        "formal_binding_mode": enforce_formal_bindings,
        "formal_public_bindings": {
            "data_commit": FORMAL_DATA_COMMIT,
            "archive_sha256": FORMAL_ARCHIVE_SHA256,
            "archive_byte_size": FORMAL_ARCHIVE_SIZE,
            "archive_git_blob": FORMAL_ARCHIVE_GIT_BLOB,
            "train_member": TRAIN_MEMBER,
            "train_byte_size": FORMAL_TRAIN_SIZE,
            "train_crc32": FORMAL_TRAIN_CRC32,
            "custody_commit": FORMAL_CUSTODY_COMMIT,
            "custody_file_sha256": FORMAL_CUSTODY_FILE_SHA256,
            "custody_semantic_sha256": FORMAL_CUSTODY_SHA256,
            "addendum_commit": FORMAL_ADDENDUM_COMMIT,
            "addendum_file_sha256": FORMAL_ADDENDUM_FILE_SHA256,
            "addendum_semantic_sha256": FORMAL_ADDENDUM_SHA256,
            "member_binding_commit": FORMAL_MEMBER_BINDING_COMMIT,
            "member_binding_file_sha256": FORMAL_MEMBER_FILE_SHA256,
            "member_binding_semantic_sha256": FORMAL_MEMBER_BINDING_SHA256,
        },
        "archive": {
            "file_sha256": archive_binding[0],
            "file_byte_size": archive_binding[1],
            "git_blob": archive_binding[2],
            **zip_receipt,
        },
        "manifest_bindings": {
            "custody": custody_receipt,
            "source_access_addendum": addendum_receipt,
            "source_member_binding": member_receipt,
        },
        "official_reader_branch": {
            "commit": FORMAL_READER_COMMIT,
            "archive_sha256": FORMAL_READER_ARCHIVE_SHA256,
            "archive_byte_size": FORMAL_READER_ARCHIVE_SIZE,
            "dataset_reader_sha256": FORMAL_READER_DATASET_SHA256,
            "loader_sha256": FORMAL_READER_LOADER_SHA256,
        },
        "aggregate": aggregate,
        "qualification_operations": {
            "train_members_opened": 1,
            "dev_members_opened": 0,
            "test_members_opened": 0,
            "raw_contract_members_opened": 0,
            "selection_or_sampling_operations": 0,
            "selection_secret_files_opened": 0,
            "concrete_document_or_label_identifiers_emitted": 0,
            "source_text_span_gold_or_annotation_rows_emitted": 0,
            "source_member_provenance_fingerprints_emitted": 1,
            "private_row_content_fingerprints_emitted": 0,
        },
        "selection_status": SELECTION_STATUS,
        "status": status,
    }
    receipt["qualification_sha256"] = _semantic_hash(receipt)
    _validate_child_receipt(receipt)
    return receipt


def _require_exact_keys(value: Any, expected: set[str], *, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value.keys()) != expected:
        raise ContractNliQualificationError(f"aggregate receipt {section} shape mismatch")
    return value


def _validate_redacted_values(payload: Any) -> None:
    public_strings = {
        SCHEMA,
        SOURCE_RELEASE,
        QUALIFICATION_CLASS,
        FORMAL_CUSTODY_SCHEMA,
        FORMAL_ADDENDUM_SCHEMA,
        FORMAL_MEMBER_SCHEMA,
        TRAIN_MEMBER,
        STATUS_QUALIFIED,
        STATUS_INFEASIBLE,
        STATUS_DIAGNOSTIC,
        SELECTION_STATUS,
    }
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str) or PUBLIC_KEY_RE.fullmatch(key) is None:
                raise ContractNliQualificationError("aggregate receipt contains a private key")
            _validate_redacted_values(value)
        return
    if isinstance(payload, (list, tuple)):
        raise ContractNliQualificationError("aggregate receipt must not contain row arrays")
    if isinstance(payload, str):
        if payload not in public_strings and HEX_RE.fullmatch(payload) is None:
            raise ContractNliQualificationError("aggregate receipt contains a private string")
        return
    if payload is None or type(payload) in {bool, int}:
        if type(payload) is int and payload < 0:
            raise ContractNliQualificationError("aggregate receipt contains a negative count")
        return
    raise ContractNliQualificationError("aggregate receipt contains a nonaggregate value")


def _validate_child_receipt(payload: Any) -> dict[str, Any]:
    top = _require_exact_keys(
        payload,
        {
            "schema",
            "source_release",
            "qualification_class",
            "formal_binding_mode",
            "formal_public_bindings",
            "archive",
            "manifest_bindings",
            "official_reader_branch",
            "aggregate",
            "qualification_operations",
            "selection_status",
            "status",
            "qualification_sha256",
        },
        section="root",
    )
    if (
        top.get("schema") != SCHEMA
        or top.get("source_release") != SOURCE_RELEASE
        or top.get("qualification_class") != QUALIFICATION_CLASS
        or type(top.get("formal_binding_mode")) is not bool
        or top.get("selection_status") != SELECTION_STATUS
    ):
        raise ContractNliQualificationError("aggregate receipt root binding mismatch")

    formal_bindings = _require_exact_keys(
        top["formal_public_bindings"],
        {
            "data_commit",
            "archive_sha256",
            "archive_byte_size",
            "archive_git_blob",
            "train_member",
            "train_byte_size",
            "train_crc32",
            "custody_commit",
            "custody_file_sha256",
            "custody_semantic_sha256",
            "addendum_commit",
            "addendum_file_sha256",
            "addendum_semantic_sha256",
            "member_binding_commit",
            "member_binding_file_sha256",
            "member_binding_semantic_sha256",
        },
        section="formal_public_bindings",
    )
    expected_formal = {
        "data_commit": FORMAL_DATA_COMMIT,
        "archive_sha256": FORMAL_ARCHIVE_SHA256,
        "archive_byte_size": FORMAL_ARCHIVE_SIZE,
        "archive_git_blob": FORMAL_ARCHIVE_GIT_BLOB,
        "train_member": TRAIN_MEMBER,
        "train_byte_size": FORMAL_TRAIN_SIZE,
        "train_crc32": FORMAL_TRAIN_CRC32,
        "custody_commit": FORMAL_CUSTODY_COMMIT,
        "custody_file_sha256": FORMAL_CUSTODY_FILE_SHA256,
        "custody_semantic_sha256": FORMAL_CUSTODY_SHA256,
        "addendum_commit": FORMAL_ADDENDUM_COMMIT,
        "addendum_file_sha256": FORMAL_ADDENDUM_FILE_SHA256,
        "addendum_semantic_sha256": FORMAL_ADDENDUM_SHA256,
        "member_binding_commit": FORMAL_MEMBER_BINDING_COMMIT,
        "member_binding_file_sha256": FORMAL_MEMBER_FILE_SHA256,
        "member_binding_semantic_sha256": FORMAL_MEMBER_BINDING_SHA256,
    }
    if dict(formal_bindings) != expected_formal:
        raise ContractNliQualificationError("formal public binding drift")

    archive = _require_exact_keys(
        top["archive"],
        {
            "file_sha256",
            "file_byte_size",
            "git_blob",
            "member_count",
            "train_member",
            "train_declared_byte_size",
            "train_crc32",
            "train_member_sha256",
            "archive_stable_after_train_read",
            "archive_extracted",
        },
        section="archive",
    )
    if (
        archive.get("train_member") != TRAIN_MEMBER
        or not isinstance(archive.get("train_member_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", archive["train_member_sha256"]) is None
        or archive.get("archive_stable_after_train_read") is not True
        or archive.get("archive_extracted") is not False
    ):
        raise ContractNliQualificationError("archive operation binding drift")

    manifests = _require_exact_keys(
        top["manifest_bindings"],
        {"custody", "source_access_addendum", "source_member_binding"},
        section="manifest_bindings",
    )
    for key, schema in (
        ("custody", FORMAL_CUSTODY_SCHEMA),
        ("source_access_addendum", FORMAL_ADDENDUM_SCHEMA),
        ("source_member_binding", FORMAL_MEMBER_SCHEMA),
    ):
        binding = _require_exact_keys(
            manifests[key],
            {"schema", "file_sha256", "byte_size", "semantic_sha256"},
            section=key,
        )
        if binding.get("schema") != schema:
            raise ContractNliQualificationError("manifest receipt schema drift")

    reader = _require_exact_keys(
        top["official_reader_branch"],
        {
            "commit",
            "archive_sha256",
            "archive_byte_size",
            "dataset_reader_sha256",
            "loader_sha256",
        },
        section="official_reader_branch",
    )
    if dict(reader) != {
        "commit": FORMAL_READER_COMMIT,
        "archive_sha256": FORMAL_READER_ARCHIVE_SHA256,
        "archive_byte_size": FORMAL_READER_ARCHIVE_SIZE,
        "dataset_reader_sha256": FORMAL_READER_DATASET_SHA256,
        "loader_sha256": FORMAL_READER_LOADER_SHA256,
    }:
        raise ContractNliQualificationError("official reader branch drift")

    aggregate = _require_exact_keys(
        top["aggregate"],
        {"root", "document_anomalies", "addressable_graph", "eligibility"},
        section="aggregate",
    )
    root = _require_exact_keys(
        aggregate["root"],
        {
            "label_count",
            "document_count",
            "valid_document_count",
            "invalid_document_count",
        },
        section="aggregate_root",
    )
    if (
        root.get("label_count") != LABEL_COUNT
        or root.get("valid_document_count", -1) + root.get("invalid_document_count", -1)
        != root.get("document_count")
    ):
        raise ContractNliQualificationError("aggregate root arithmetic mismatch")
    _require_exact_keys(
        aggregate["document_anomalies"],
        set(ANOMALY_KINDS),
        section="document_anomalies",
    )
    _require_exact_keys(
        aggregate["addressable_graph"],
        {
            "valid_document_node_count_total",
            "valid_document_node_count_min",
            "valid_document_node_count_max",
            "node_eligible_document_count",
            "duplicate_node_text_count",
            "shared_start_count",
            "repeated_boundary_count",
            "overlapping_span_pair_count",
        },
        section="addressable_graph",
    )
    eligibility = _require_exact_keys(
        aggregate["eligibility"],
        {
            "eligible_item_count_before_one_per_document_cap",
            "duplicate_gold_index_count",
            "document_with_eligible_item_count",
            "exposure_excluded_document_count",
            "full_text_signature_excluded_document_count",
            "metadata_file_excluded_document_count",
            "metadata_url_excluded_document_count",
            "eligible_document_count_after_exposure",
            "eligible_normalized_content_group_count",
            "duplicate_content_group_count",
            "documents_in_duplicate_content_groups",
            "maximum_content_group_size",
            "minimum_required_content_groups",
            "capacity_satisfied",
        },
        section="eligibility",
    )
    if (
        eligibility.get("minimum_required_content_groups") != MIN_CONTENT_GROUPS
        or type(eligibility.get("capacity_satisfied")) is not bool
        or eligibility.get("capacity_satisfied")
        != (eligibility.get("eligible_normalized_content_group_count", -1) >= MIN_CONTENT_GROUPS)
    ):
        raise ContractNliQualificationError("aggregate capacity arithmetic mismatch")

    operations = _require_exact_keys(
        top["qualification_operations"],
        {
            "train_members_opened",
            "dev_members_opened",
            "test_members_opened",
            "raw_contract_members_opened",
            "selection_or_sampling_operations",
            "selection_secret_files_opened",
            "concrete_document_or_label_identifiers_emitted",
            "source_text_span_gold_or_annotation_rows_emitted",
            "source_member_provenance_fingerprints_emitted",
            "private_row_content_fingerprints_emitted",
        },
        section="qualification_operations",
    )
    one_fields = {
        "train_members_opened",
        "source_member_provenance_fingerprints_emitted",
    }
    if any(operations.get(key) != 1 for key in one_fields) or any(
        operations.get(key) != 0 for key in operations if key not in one_fields
    ):
        raise ContractNliQualificationError("aggregate receipt violates row-free isolation")

    expected_status = STATUS_DIAGNOSTIC
    if top["formal_binding_mode"]:
        expected_status = (
            STATUS_QUALIFIED
            if eligibility["capacity_satisfied"]
            else STATUS_INFEASIBLE
        )
        if (
            archive.get("file_sha256") != FORMAL_ARCHIVE_SHA256
            or archive.get("file_byte_size") != FORMAL_ARCHIVE_SIZE
            or archive.get("git_blob") != FORMAL_ARCHIVE_GIT_BLOB
            or archive.get("train_declared_byte_size") != FORMAL_TRAIN_SIZE
            or archive.get("train_crc32") != FORMAL_TRAIN_CRC32
        ):
            raise ContractNliQualificationError("formal receipt source binding drift")
    if top.get("status") != expected_status:
        raise ContractNliQualificationError("aggregate receipt status mismatch")

    declared = top.get("qualification_sha256")
    if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
        raise ContractNliQualificationError("aggregate receipt semantic hash missing")
    body = dict(top)
    del body["qualification_sha256"]
    if _semantic_hash(body) != declared:
        raise ContractNliQualificationError("aggregate receipt semantic hash mismatch")
    _validate_redacted_values(top)
    return dict(top)


def _validate_receipt_against_inputs(
    receipt: dict[str, Any],
    archive_path: Path,
    custody_manifest_path: Path,
    source_access_addendum_path: Path,
    source_member_binding_path: Path,
) -> None:
    archive_binding = _hash_archive(archive_path)
    if (
        receipt["archive"]["file_sha256"],
        receipt["archive"]["file_byte_size"],
        receipt["archive"]["git_blob"],
    ) != archive_binding:
        raise ContractNliQualificationError("worker archive binding mismatch")
    for key, path, schema, hash_field in (
        ("custody", custody_manifest_path, FORMAL_CUSTODY_SCHEMA, "custody_sha256"),
        (
            "source_access_addendum",
            source_access_addendum_path,
            FORMAL_ADDENDUM_SCHEMA,
            "addendum_sha256",
        ),
        (
            "source_member_binding",
            source_member_binding_path,
            FORMAL_MEMBER_SCHEMA,
            "source_member_binding_sha256",
        ),
    ):
        _, expected = _read_manifest(path, schema=schema, hash_field=hash_field)
        if receipt["manifest_bindings"][key] != expected:
            raise ContractNliQualificationError("worker manifest binding mismatch")


def _formal_attempt_marker_path() -> Path:
    return Path(__file__).resolve().parents[2] / FORMAL_ATTEMPT_MARKER_RELATIVE_PATH


def _consume_formal_attempt() -> Path:
    """Irreversibly consume the one formal attempt before any source read."""

    marker = _formal_attempt_marker_path()
    parent = marker.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise ContractNliQualificationError("formal attempt marker parent is unavailable") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ContractNliQualificationError(
            "formal attempt marker parent must be a non-symlink directory"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        raw = f"{SCHEMA}\nformal_attempt_consumed\n".encode("ascii")
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise ContractNliQualificationError(
            "formal source qualification attempt is already consumed"
        ) from exc
    except OSError as exc:
        raise ContractNliQualificationError("formal attempt marker creation failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return marker


def run_clean_qualification(
    archive_path: str | Path,
    custody_manifest_path: str | Path,
    source_access_addendum_path: str | Path,
    source_member_binding_path: str | Path,
    *,
    enforce_formal_bindings: bool = False,
) -> dict[str, Any]:
    """Run the only member-opening operation in an isolated Python worker."""

    inputs = [
        Path(value).absolute()
        for value in (
            archive_path,
            custody_manifest_path,
            source_access_addendum_path,
            source_member_binding_path,
        )
    ]
    if enforce_formal_bindings:
        _consume_formal_attempt()
    for path, label in zip(
        inputs,
        ("archive", "custody manifest", "source access addendum", "member binding"),
    ):
        _require_regular_file(path, label=label)
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--_aggregate-worker",
        "--archive",
        str(inputs[0]),
        "--custody-manifest",
        str(inputs[1]),
        "--source-access-addendum",
        str(inputs[2]),
        "--source-member-binding",
        str(inputs[3]),
    ]
    if enforce_formal_bindings:
        command.append("--formal")
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        check=False,
        close_fds=True,
        env={
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C.UTF-8",
        },
        cwd=str(Path(__file__).resolve().parent),
    )
    if completed.returncode != 0:
        raise ContractNliQualificationError("clean aggregate worker failed")
    payload = _json_no_duplicate_keys(completed.stdout, label="clean worker receipt")
    receipt = _validate_child_receipt(payload)
    _validate_receipt_against_inputs(receipt, *inputs)
    return receipt


def _atomic_write_exclusive(destination: Path, raw: bytes, *, mode: int) -> None:
    parent = destination.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise ContractNliQualificationError("output parent is unavailable") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ContractNliQualificationError("output parent must be a non-symlink directory")

    temporary = parent / f".{destination.name}.{os.urandom(12).hex()}.tmp"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if published:
            destination.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).absolute()
    raw = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    _atomic_write_exclusive(destination, raw, mode=0o644)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--custody-manifest", required=True, type=Path)
    parser.add_argument("--source-access-addendum", required=True, type=Path)
    parser.add_argument("--source-member-binding", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="parent-process-only exclusive public receipt destination",
    )
    parser.add_argument("--_aggregate-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments._aggregate_worker:
        if arguments.output is not None:
            parser.error("--output is parent-process-only")
        receipt = build_qualification(
            arguments.archive,
            arguments.custody_manifest,
            arguments.source_access_addendum,
            arguments.source_member_binding,
            enforce_formal_bindings=arguments.formal,
        )
    else:
        if arguments.formal and arguments.output is None:
            parser.error("--formal requires --output")
        receipt = run_clean_qualification(
            arguments.archive,
            arguments.custody_manifest,
            arguments.source_access_addendum,
            arguments.source_member_binding,
            enforce_formal_bindings=arguments.formal,
        )
        if arguments.output is not None:
            _write_json_exclusive(arguments.output, receipt)
    sys.stdout.write(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
