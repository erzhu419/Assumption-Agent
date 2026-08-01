"""Mechanical, non-scoring protocol for a future GSCL ARN intrinsic run.

This module closes custody, split, pack, action-seal, and aggregate-scoring
mechanics without implementing an ARN narrative adapter or any of the four
prediction arms.  It is therefore *not* a measurement runner and its public
protocol receipt must remain ``freeze_ready == False`` until independently
qualified implementation closures are supplied.

The official source verifier reads the complete files only to establish byte
identity.  It decodes the CSV header, but never decodes an item row.  All row
and lifecycle tests use synthetic fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Protocol, Sequence
import unicodedata


VERSION = "gscl_arn_intrinsic_protocol_v1"
SOURCE_SCHEMA = f"{VERSION}.source_verification.v1"
SYNTHETIC_PREDICTOR_PACK_SCHEMA = f"{VERSION}.synthetic_predictor_pack.v2"
SYNTHETIC_LINKAGE_PACK_SCHEMA = f"{VERSION}.synthetic_linkage_pack.v2"
SYNTHETIC_LABEL_PACK_SCHEMA = f"{VERSION}.synthetic_label_pack.v2"
OFFICIAL_PREDICTOR_PACK_SCHEMA = f"{VERSION}.official_predictor_pack.v2"
OFFICIAL_LINKAGE_PACK_SCHEMA = f"{VERSION}.official_linkage_pack.v2"
OFFICIAL_LABEL_PACK_SCHEMA = f"{VERSION}.official_label_pack.v2"
PREDICTION_PACK_SCHEMA = f"{VERSION}.prediction_pack.v1"
QUALIFICATION_ACTION_SEAL_SCHEMA = (
    f"{VERSION}.synthetic_qualification_action_seal.v2"
)
FORMAL_ACTION_SEAL_SCHEMA = f"{VERSION}.formal_all_arm_action_seal.v2"
QUALIFICATION_SCORE_RECEIPT_SCHEMA = (
    f"{VERSION}.synthetic_qualification_score_receipt.v2"
)
FORMAL_SCORE_RECEIPT_SCHEMA = f"{VERSION}.formal_aggregate_score_receipt.v2"
PROTOCOL_RECEIPT_SCHEMA = f"{VERSION}.safe_protocol_receipt.v1"
IMPLEMENTATION_QUALIFICATION_SCHEMA = (
    f"{VERSION}.implementation_qualification.v1"
)
MATERIALIZATION_RECEIPT_SCHEMA = (
    f"{VERSION}.capability_materialization.v1"
)
RUNTIME_ACCESS_RECEIPT_SCHEMA = f"{VERSION}.runtime_access_qualification.v1"
READY_FREEZE_MANIFEST_SCHEMA = f"{VERSION}.ready_freeze_manifest.v1"
ADAPTER_OUTPUT_RECEIPT_SCHEMA = f"{VERSION}.adapter_output_receipt.v1"
ADAPTER_INVOCATION_RECEIPT_SCHEMA = (
    f"{VERSION}.adapter_invocation_receipt.v1"
)
_ADAPTER_INVOCATION_VALIDATION_TOKEN = object()

OFFICIAL_DOI = "10.5281/zenodo.11044026"
OFFICIAL_CONCEPT_DOI = "10.5281/zenodo.11044025"
OFFICIAL_ZENODO_REVISION = 4
OFFICIAL_LICENSE_ID = "cc-by-4.0"
SOURCE_QUALIFICATION_REPORT_SHA256 = (
    "fb0a0afc1bb0e98e1c17d05f9490486c72cf90f7bb802ebb204e4d8f11c7fa9e"
)
OFFICIAL_DATASET_FILENAME = (
    "Analogical Reasoning on Narratives (ARN) dataset.xlsx - Sheet1.csv"
)
OFFICIAL_DATASET_SIZE = 1_256_913
OFFICIAL_DATASET_MD5 = "38484f48176fd0bfa0b569acb55f1176"
OFFICIAL_DATASET_SHA256 = (
    "a866fe5341ce4a29f00f24987a12278303b2b8ad788352f549b0fe051ad4a7a8"
)
OFFICIAL_METADATA_SIZE = 5_562
OFFICIAL_METADATA_SHA256 = (
    "c9e91d7a49ea383eeccec5421cce9f1b0d8713c243187d840482eb1764f3317f"
)
OFFICIAL_HEADER = (
    "id",
    "proverb",
    "query_narrative",
    "first_choice",
    "second_choice",
    "distractor_similarity",
    "analogy_level",
    "correct_answer",
)
OFFICIAL_HEADER_BYTES = (",".join(OFFICIAL_HEADER) + "\r\n").encode("ascii")
OFFICIAL_ROW_COUNT = 1_095
OFFICIAL_ID_MINIMUM = 1
OFFICIAL_ID_MAXIMUM = 1_096
OFFICIAL_MISSING_IDS = (753,)
OFFICIAL_CELL_COUNTS = {
    "far_high": 254,
    "far_low": 294,
    "near_high": 253,
    "near_low": 294,
}

SPLIT_SALT = b"GSCL_ARN_INTRINSIC_V1"
FROZEN_UNIDATA_VERSION = "13.0.0"
CALIBRATION_BUCKET = 0
MEASUREMENT_BUCKETS = (1, 2, 3, 4)

ARM_IDS = (
    "semantic_only",
    "legacy_keyword",
    "flat_label_no_verifier",
    "full_gscl",
)
CHOICE_IDS = ("first_choice", "second_choice")
DISPOSITIONS = ("ANSWER", "ABSTAIN", "ERROR")
ERROR_CODES = (
    "ARM_RUNTIME_ERROR",
    "INVALID_ARM_OUTPUT",
    "NO_VALID_PREDICTION",
)
ANALOGY_LEVELS = ("far", "near")
DISTRACTOR_SIMILARITIES = ("high", "low")

COLUMN_ACCESS_MATRIX = {
    "splitter": ("id", "proverb"),
    "arms": ("query_narrative", "first_choice", "second_choice"),
    "scorer_only": (
        "correct_answer",
        "analogy_level",
        "distractor_similarity",
    ),
}

# Source row 1 was exposed accidentally during implementation.  Its whole
# proverb group is identified only inside the private HMAC linkage domain.
# No public dictionary-attackable proverb digest is retained or emitted.
IMPLEMENTATION_EXPOSURE_SOURCE_ID = "1"
IMPLEMENTATION_EXPOSURE_BUCKET = 2
IMPLEMENTATION_EXPOSURE = {
    "status": "implementation_exposure_quarantine",
    "exposed_field_classes": ["proverb", "query_narrative_prefix"],
    "exposed_choice_count": 0,
    "exposed_answer_count": 0,
    "exposed_cell_label_count": 0,
    "quarantine_anchor": "source_row_1_whole_proverb_group",
    "public_group_digest_emitted": False,
    "immutable_mod5_bucket": IMPLEMENTATION_EXPOSURE_BUCKET,
    "measurement_excluded": True,
    "label_unseen_disposition": True,
    "label_unseen_is_cryptographic_claim": False,
    "split_assignment_changed": False,
    "rebalanced": False,
    "fallback_replacement": False,
}


class ArnIntrinsicProtocolError(RuntimeError):
    """A frozen mechanical protocol, source, or lifecycle invariant drifted."""


class ArnImplementationNotReady(ArnIntrinsicProtocolError):
    """A content-sensitive implementation is deliberately not available."""


@dataclass(frozen=True)
class SourceBinding:
    doi: str
    concept_doi: str
    revision: int
    license_id: str
    dataset_filename: str
    dataset_size: int
    dataset_md5: str
    dataset_sha256: str
    metadata_size: int
    metadata_sha256: str
    header_bytes: bytes


OFFICIAL_SOURCE_BINDING = SourceBinding(
    doi=OFFICIAL_DOI,
    concept_doi=OFFICIAL_CONCEPT_DOI,
    revision=OFFICIAL_ZENODO_REVISION,
    license_id=OFFICIAL_LICENSE_ID,
    dataset_filename=OFFICIAL_DATASET_FILENAME,
    dataset_size=OFFICIAL_DATASET_SIZE,
    dataset_md5=OFFICIAL_DATASET_MD5,
    dataset_sha256=OFFICIAL_DATASET_SHA256,
    metadata_size=OFFICIAL_METADATA_SIZE,
    metadata_sha256=OFFICIAL_METADATA_SHA256,
    header_bytes=OFFICIAL_HEADER_BYTES,
)


@dataclass(frozen=True)
class SplitAssignment:
    private_group_id: str
    bucket: int
    hash_partition: str
    effective_partition: str
    measurement_eligible: bool
    exclusion_codes: tuple[str, ...]


@dataclass(frozen=True)
class AdaptedArnRow:
    """Normalized row shape expected *after* the missing raw adapter.

    ``gold_choice`` is deliberately canonical.  No code in this module maps
    the official ``correct_answer`` representation into this field.
    """

    source_id: str
    proverb: str
    query_narrative: str
    first_choice: str
    second_choice: str
    gold_choice: str
    analogy_level: str
    distractor_similarity: str


@dataclass(frozen=True)
class PrivatePackBundle:
    lineage: str
    predictor_pack: Mapping[str, Any]
    linkage_pack: Mapping[str, Any]
    label_pack: Mapping[str, Any]
    pack_commitments: Mapping[str, str]
    safe_split_aggregates: Mapping[str, Any]
    adapter_output_receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedAdapterInvocation:
    rows: tuple[AdaptedArnRow, ...]
    receipt: Mapping[str, Any]
    adapter_qualification: ValidatedImplementationQualification
    _validation_token: object


class RawNarrativeAdapter(Protocol):
    """Interface a future independently qualified ARN adapter must satisfy."""

    implementation_sha256: str
    qualification_receipt_sha256: str

    def adapt(self, source_path: Path) -> Sequence[AdaptedArnRow]:
        """Map the exact raw source into canonical rows without model access."""


class ArnArmAlgorithm(Protocol):
    """Interface each future frozen arm implementation must satisfy."""

    arm_id: str
    implementation_sha256: str
    qualification_receipt_sha256: str

    def predict(self, predictor_pack: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one strict prediction pack over the common predictor pack."""


def build_raw_narrative_adapter() -> RawNarrativeAdapter:
    """Refuse content-sensitive adapter construction until closure exists."""

    raise ArnImplementationNotReady(
        "ARN raw narrative adapter implementation closure is not available"
    )


def build_arm_algorithm(arm_id: str) -> ArnArmAlgorithm:
    """Refuse arm construction until the named implementation is qualified."""

    if arm_id not in ARM_IDS:
        raise ArnIntrinsicProtocolError("unknown frozen arm identifier")
    raise ArnImplementationNotReady(
        "ARN arm implementation closure is not available"
    )


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ArnIntrinsicProtocolError(
            "protocol value is not canonical JSON"
        ) from exc


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ArnIntrinsicProtocolError(f"{label} schema drifted")


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArnIntrinsicProtocolError(
                "Zenodo metadata contains a duplicate object key"
            )
        result[key] = value
    return result


def _read_exact_regular_file(
    path: Path, *, size: int, sha256: str, label: str
) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ArnIntrinsicProtocolError(
            f"{label} is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != size
    ):
        raise ArnIntrinsicProtocolError(
            f"{label} topology or exact size drifted"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArnIntrinsicProtocolError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_nlink != 1
        ):
            raise ArnIntrinsicProtocolError(
                f"{label} changed while opening"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or after.st_nlink != 1
        ):
            raise ArnIntrinsicProtocolError(
                f"{label} changed while hashing"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise ArnIntrinsicProtocolError(
            f"{label} exact SHA256 identity drifted"
        )
    return raw


def _verify_source_files(
    dataset_path: Path,
    metadata_path: Path,
    binding: SourceBinding,
) -> dict[str, Any]:
    """Verify a bound source without decoding any dataset row."""

    if (
        not _is_sha256(binding.dataset_sha256)
        or not _is_sha256(binding.metadata_sha256)
        or len(binding.dataset_md5) != 32
    ):
        raise ArnIntrinsicProtocolError("source binding hashes are invalid")
    dataset_raw = _read_exact_regular_file(
        dataset_path,
        size=binding.dataset_size,
        sha256=binding.dataset_sha256,
        label="ARN dataset",
    )
    if hashlib.md5(dataset_raw).hexdigest() != binding.dataset_md5:  # noqa: S324
        raise ArnIntrinsicProtocolError("ARN dataset Zenodo MD5 drifted")
    header_end = dataset_raw.find(b"\n")
    if header_end < 0 or dataset_raw[: header_end + 1] != binding.header_bytes:
        raise ArnIntrinsicProtocolError("ARN CSV exact header drifted")

    metadata_raw = _read_exact_regular_file(
        metadata_path,
        size=binding.metadata_size,
        sha256=binding.metadata_sha256,
        label="ARN Zenodo metadata",
    )
    try:
        metadata = json.loads(
            metadata_raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArnIntrinsicProtocolError(
            "ARN Zenodo metadata is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ArnIntrinsicProtocolError("ARN Zenodo metadata root drifted")
    record_metadata = metadata.get("metadata")
    files = metadata.get("files")
    if (
        metadata.get("doi") != binding.doi
        or metadata.get("conceptdoi") != binding.concept_doi
        or metadata.get("revision") != binding.revision
        or not isinstance(record_metadata, Mapping)
        or record_metadata.get("doi") != binding.doi
        or record_metadata.get("license") != {"id": binding.license_id}
        or not isinstance(files, list)
        or len(files) != 1
        or not isinstance(files[0], Mapping)
    ):
        raise ArnIntrinsicProtocolError(
            "ARN Zenodo DOI or exact license binding drifted"
        )
    file_entry = files[0]
    if (
        file_entry.get("key") != binding.dataset_filename
        or file_entry.get("size") != binding.dataset_size
        or file_entry.get("checksum") != f"md5:{binding.dataset_md5}"
    ):
        raise ArnIntrinsicProtocolError(
            "ARN Zenodo file binding drifted"
        )

    body: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "verified": True,
        "dataset_sha256": binding.dataset_sha256,
        "dataset_size": binding.dataset_size,
        "metadata_sha256": binding.metadata_sha256,
        "doi": binding.doi,
        "concept_doi": binding.concept_doi,
        "zenodo_revision": binding.revision,
        "license_id": binding.license_id,
        "source_qualification_report_sha256": (
            SOURCE_QUALIFICATION_REPORT_SHA256
        ),
        "header_sha256": hashlib.sha256(binding.header_bytes).hexdigest(),
        "dataset_rows_decoded": 0,
        "item_content_emitted": False,
    }
    body["self_hash"] = _content_hash(body)
    return body


def verify_official_source(
    dataset_path: Path, metadata_path: Path
) -> dict[str, Any]:
    """Verify the exact official source, DOI, and license fail-closed."""

    return _verify_source_files(
        dataset_path, metadata_path, OFFICIAL_SOURCE_BINDING
    )


def _validate_linkage_secret(linkage_secret: bytes) -> None:
    if (
        not isinstance(linkage_secret, bytes)
        or len(linkage_secret) < 32
        or len(set(linkage_secret)) < 8
    ):
        raise ArnIntrinsicProtocolError(
            "pre-source private linkage HMAC secret is invalid"
        )


def _public_split_bucket(proverb: str) -> int:
    """Compute the public split internally without returning its digest."""

    if unicodedata.unidata_version != FROZEN_UNIDATA_VERSION:
        raise ArnIntrinsicProtocolError(
            "Unicode database version drifted from the frozen splitter"
        )
    if not isinstance(proverb, str) or not proverb:
        raise ArnIntrinsicProtocolError(
            "splitter requires a non-empty proverb string"
        )
    normalized = unicodedata.normalize("NFKC", proverb)
    digest = hashlib.sha256(
        SPLIT_SALT + b"\0" + normalized.encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % 5


def split_proverb(
    proverb: str, *, linkage_secret: bytes
) -> SplitAssignment:
    """Assign a group using public mod5 and private HMAC linkage identity."""

    _validate_linkage_secret(linkage_secret)
    bucket = _public_split_bucket(proverb)
    normalized = unicodedata.normalize("NFKC", proverb)
    private_group_id = hmac.new(
        linkage_secret,
        b"group\0" + normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    hash_partition = (
        "calibration" if bucket == CALIBRATION_BUCKET else "measurement"
    )
    return SplitAssignment(
        private_group_id=private_group_id,
        bucket=bucket,
        hash_partition=hash_partition,
        effective_partition=hash_partition,
        measurement_eligible=hash_partition == "measurement",
        exclusion_codes=(),
    )


def opaque_item_id(source_id: str, *, linkage_secret: bytes) -> str:
    _validate_linkage_secret(linkage_secret)
    if (
        not isinstance(source_id, str)
        or not source_id
        or not source_id.isascii()
        or not source_id.isdecimal()
        or (len(source_id) > 1 and source_id.startswith("0"))
    ):
        raise ArnIntrinsicProtocolError(
            "source id is not a canonical positive decimal string"
        )
    numeric = int(source_id)
    if numeric <= 0:
        raise ArnIntrinsicProtocolError(
            "source id is not a canonical positive decimal string"
        )
    return hmac.new(
        linkage_secret,
        b"item\0" + source_id.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def _validate_adapted_row(row: AdaptedArnRow) -> None:
    for value in (
        row.proverb,
        row.query_narrative,
        row.first_choice,
        row.second_choice,
    ):
        if not isinstance(value, str) or not value:
            raise ArnIntrinsicProtocolError(
                "adapted row contains an empty textual field"
            )
    if row.gold_choice not in CHOICE_IDS:
        raise ArnIntrinsicProtocolError(
            "adapted row gold choice is not canonical"
        )
    if row.analogy_level not in ANALOGY_LEVELS:
        raise ArnIntrinsicProtocolError(
            "adapted row analogy level is not canonical"
        )
    if row.distractor_similarity not in DISTRACTOR_SIMILARITIES:
        raise ArnIntrinsicProtocolError(
            "adapted row distractor similarity is not canonical"
        )


def _validate_source_verification_receipt(
    source_verification: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        source_verification,
        {
            "schema",
            "verified",
            "dataset_sha256",
            "dataset_size",
            "metadata_sha256",
            "doi",
            "concept_doi",
            "zenodo_revision",
            "license_id",
            "source_qualification_report_sha256",
            "header_sha256",
            "dataset_rows_decoded",
            "item_content_emitted",
            "self_hash",
        },
        label="official source verification receipt",
    )
    body = dict(source_verification)
    claimed = body.pop("self_hash")
    if (
        source_verification["schema"] != SOURCE_SCHEMA
        or source_verification["verified"] is not True
        or source_verification["dataset_sha256"]
        != OFFICIAL_DATASET_SHA256
        or source_verification["dataset_size"] != OFFICIAL_DATASET_SIZE
        or source_verification["metadata_sha256"]
        != OFFICIAL_METADATA_SHA256
        or source_verification["doi"] != OFFICIAL_DOI
        or source_verification["concept_doi"] != OFFICIAL_CONCEPT_DOI
        or source_verification["zenodo_revision"]
        != OFFICIAL_ZENODO_REVISION
        or source_verification["license_id"] != OFFICIAL_LICENSE_ID
        or source_verification["source_qualification_report_sha256"]
        != SOURCE_QUALIFICATION_REPORT_SHA256
        or source_verification["header_sha256"]
        != hashlib.sha256(OFFICIAL_HEADER_BYTES).hexdigest()
        or source_verification["dataset_rows_decoded"] != 0
        or source_verification["item_content_emitted"] is not False
        or not _is_sha256(claimed)
        or _content_hash(body) != claimed
    ):
        raise ArnIntrinsicProtocolError(
            "official source verification receipt drifted"
        )


def _build_private_packs(
    rows: Sequence[AdaptedArnRow],
    *,
    source_sha256: str,
    linkage_secret: bytes,
    lineage: str,
    schemas: tuple[str, str, str],
    source_verification_self_hash: str | None,
    adapter_qualification_self_hash: str | None,
    quarantine_source_id: str | None,
) -> PrivatePackBundle:
    _validate_linkage_secret(linkage_secret)
    if not _is_sha256(source_sha256):
        raise ArnIntrinsicProtocolError("source pack SHA256 is invalid")
    if not rows:
        raise ArnIntrinsicProtocolError("adapted pack is empty")
    if lineage not in {
        "synthetic_qualification_fixture",
        "official_arn_measurement",
    }:
        raise ArnIntrinsicProtocolError("private pack lineage is invalid")
    if lineage == "synthetic_qualification_fixture":
        if (
            source_sha256 == OFFICIAL_DATASET_SHA256
            or source_verification_self_hash is not None
            or adapter_qualification_self_hash is not None
            or quarantine_source_id is not None
        ):
            raise ArnIntrinsicProtocolError(
                "synthetic fixture cannot impersonate official lineage"
            )
    else:
        if (
            source_sha256 != OFFICIAL_DATASET_SHA256
            or not _is_sha256(source_verification_self_hash)
            or not _is_sha256(adapter_qualification_self_hash)
            or quarantine_source_id != IMPLEMENTATION_EXPOSURE_SOURCE_ID
        ):
            raise ArnIntrinsicProtocolError(
                "official private pack lineage is incomplete"
            )

    validated_rows: list[AdaptedArnRow] = []
    source_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, AdaptedArnRow):
            raise ArnIntrinsicProtocolError("adapted row type drifted")
        _validate_adapted_row(row)
        opaque_item_id(row.source_id, linkage_secret=linkage_secret)
        if row.source_id in source_ids:
            raise ArnIntrinsicProtocolError(
                "adapted source id is duplicated"
            )
        source_ids.add(row.source_id)
        validated_rows.append(row)

    quarantine_group_id: str | None = None
    quarantine_bucket: int | None = None
    if quarantine_source_id is not None:
        anchors = [
            row for row in validated_rows
            if row.source_id == quarantine_source_id
        ]
        if len(anchors) != 1:
            raise ArnIntrinsicProtocolError(
                "implementation exposure anchor is not unique"
            )
        anchor = split_proverb(
            anchors[0].proverb, linkage_secret=linkage_secret
        )
        quarantine_group_id = anchor.private_group_id
        quarantine_bucket = anchor.bucket
        if quarantine_bucket != IMPLEMENTATION_EXPOSURE_BUCKET:
            raise ArnIntrinsicProtocolError(
                "implementation exposure bucket commitment drifted"
            )

    predictor_rows: list[dict[str, Any]] = []
    linkage_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    group_partitions: dict[str, tuple[int, str]] = {}
    split_counts: defaultdict[str, int] = defaultdict(int)
    bucket_counts: defaultdict[str, int] = defaultdict(int)
    linkage_key_commitment = hashlib.sha256(
        b"linkage-key-commitment\0" + linkage_secret
    ).hexdigest()

    for row in validated_rows:
        item_id = opaque_item_id(
            row.source_id, linkage_secret=linkage_secret
        )
        assignment = split_proverb(
            row.proverb, linkage_secret=linkage_secret
        )
        prior = group_partitions.setdefault(
            assignment.private_group_id,
            (assignment.bucket, assignment.hash_partition),
        )
        if prior != (assignment.bucket, assignment.hash_partition):
            raise ArnIntrinsicProtocolError(
                "whole-proverb group crossed a hash partition"
            )
        is_quarantined = (
            quarantine_group_id is not None
            and assignment.private_group_id == quarantine_group_id
        )
        effective_partition = (
            "implementation_exposure_quarantine"
            if is_quarantined
            else assignment.effective_partition
        )
        measurement_eligible = (
            assignment.measurement_eligible and not is_quarantined
        )
        exclusions = (
            ["IMPLEMENTATION_EXPOSURE"] if is_quarantined else []
        )
        bucket_counts[str(assignment.bucket)] += 1
        split_counts[effective_partition] += 1
        linkage_rows.append(
            {
                "opaque_item_id": item_id,
                "private_group_id": assignment.private_group_id,
                "bucket": assignment.bucket,
                "hash_partition": assignment.hash_partition,
                "effective_partition": effective_partition,
                "measurement_eligible": measurement_eligible,
                "exclusion_codes": exclusions,
            }
        )
        if not measurement_eligible:
            continue
        predictor_rows.append(
            {
                "opaque_item_id": item_id,
                "query_narrative": row.query_narrative,
                "first_choice": row.first_choice,
                "second_choice": row.second_choice,
            }
        )
        label_rows.append(
            {
                "opaque_item_id": item_id,
                "gold_choice": row.gold_choice,
                "analogy_level": row.analogy_level,
                "distractor_similarity": row.distractor_similarity,
            }
        )

    predictor_rows.sort(key=lambda value: value["opaque_item_id"])
    linkage_rows.sort(key=lambda value: value["opaque_item_id"])
    label_rows.sort(key=lambda value: value["opaque_item_id"])
    common = {
        "lineage": lineage,
        "source_sha256": source_sha256,
        "source_verification_self_hash": source_verification_self_hash,
        "adapter_qualification_self_hash": (
            adapter_qualification_self_hash
        ),
    }
    predictor_pack: dict[str, Any] = {
        "schema": schemas[0],
        **common,
        "column_contract": list(COLUMN_ACCESS_MATRIX["arms"]),
        "rows": predictor_rows,
    }
    linkage_pack: dict[str, Any] = {
        "schema": schemas[1],
        **common,
        "column_contract": list(COLUMN_ACCESS_MATRIX["splitter"]),
        "linkage_key_commitment": linkage_key_commitment,
        "public_split_digest_emitted": False,
        "rows": linkage_rows,
    }
    label_pack: dict[str, Any] = {
        "schema": schemas[2],
        **common,
        "column_contract": list(COLUMN_ACCESS_MATRIX["scorer_only"]),
        "rows": label_rows,
    }
    commitments = {
        "predictor_pack_sha256": _content_hash(predictor_pack),
        "linkage_pack_sha256": _content_hash(linkage_pack),
        "label_pack_sha256": _content_hash(label_pack),
    }
    split_aggregates = {
        "source_row_count": len(rows),
        "private_group_count": len(group_partitions),
        "bucket_counts": {
            key: bucket_counts.get(key, 0)
            for key in ("0", "1", "2", "3", "4")
        },
        "effective_partition_counts": {
            "calibration": split_counts.get("calibration", 0),
            "measurement": split_counts.get("measurement", 0),
            "implementation_exposure_quarantine": split_counts.get(
                "implementation_exposure_quarantine", 0
            ),
        },
        "measurement_item_count": len(predictor_rows),
        "quarantine_group_present": quarantine_group_id is not None,
        "quarantine_original_bucket": quarantine_bucket,
        "public_group_digest_emitted": False,
        "whole_proverb_group_cross_partition_count": 0,
        "rebalanced": False,
        "fallback_replacement": False,
    }
    return PrivatePackBundle(
        lineage=lineage,
        predictor_pack=predictor_pack,
        linkage_pack=linkage_pack,
        label_pack=label_pack,
        pack_commitments=commitments,
        safe_split_aggregates=split_aggregates,
    )


def _build_private_packs_from_adapted_fixtures(
    rows: Sequence[AdaptedArnRow],
    *,
    source_sha256: str,
    linkage_secret: bytes,
) -> PrivatePackBundle:
    """Build synthetic qualification packs that cannot become formal."""

    return _build_private_packs(
        rows,
        source_sha256=source_sha256,
        linkage_secret=linkage_secret,
        lineage="synthetic_qualification_fixture",
        schemas=(
            SYNTHETIC_PREDICTOR_PACK_SCHEMA,
            SYNTHETIC_LINKAGE_PACK_SCHEMA,
            SYNTHETIC_LABEL_PACK_SCHEMA,
        ),
        source_verification_self_hash=None,
        adapter_qualification_self_hash=None,
        quarantine_source_id=None,
    )


def run_official_adapter_once(
    *,
    dataset_path: Path,
    metadata_path: Path,
    source_verification: Mapping[str, Any],
    adapter_qualification: ValidatedImplementationQualification,
) -> ValidatedAdapterInvocation:
    """Run a validated adapter only on the reverified exact official source."""

    actual_source = verify_official_source(dataset_path, metadata_path)
    _validate_source_verification_receipt(source_verification)
    if actual_source["self_hash"] != source_verification["self_hash"]:
        raise ArnIntrinsicProtocolError(
            "official source verification changed before adapter invocation"
        )
    validated_adapter = _validated_qualification(
        adapter_qualification, component_id="raw_narrative_adapter"
    )
    adapter = build_raw_narrative_adapter()
    if (
        adapter.implementation_sha256
        != validated_adapter.implementation_file_sha256
        or adapter.qualification_receipt_sha256
        != validated_adapter.receipt["self_hash"]
    ):
        raise ArnIntrinsicProtocolError(
            "runtime adapter does not match its validated closure"
        )
    rows = tuple(adapter.adapt(dataset_path))
    invocation: dict[str, Any] = {
        "schema": ADAPTER_INVOCATION_RECEIPT_SCHEMA,
        "status": "EXACT_SOURCE_ADAPTER_INVOKED_ONCE",
        "source_verification_self_hash": source_verification["self_hash"],
        "source_sha256": OFFICIAL_DATASET_SHA256,
        "adapter_qualification_self_hash": validated_adapter.receipt[
            "self_hash"
        ],
        "adapter_implementation_file_sha256": (
            validated_adapter.implementation_file_sha256
        ),
        "adapted_row_count": len(rows),
        "adapted_output_commitment": _content_hash(
            [row.__dict__ for row in rows]
        ),
        "item_content_emitted": False,
    }
    invocation["self_hash"] = _content_hash(invocation)
    return ValidatedAdapterInvocation(
        rows=rows,
        receipt=invocation,
        adapter_qualification=validated_adapter,
        _validation_token=_ADAPTER_INVOCATION_VALIDATION_TOKEN,
    )


def build_official_private_packs(
    invocation: ValidatedAdapterInvocation,
    *,
    linkage_secret: bytes,
) -> PrivatePackBundle:
    """Bind the single validated source→adapter invocation into three packs."""

    if (
        not isinstance(invocation, ValidatedAdapterInvocation)
        or invocation._validation_token
        is not _ADAPTER_INVOCATION_VALIDATION_TOKEN
    ):
        raise ArnIntrinsicProtocolError(
            "official packs require a validated adapter invocation"
        )
    rows = invocation.rows
    invocation_body = dict(invocation.receipt)
    invocation_claimed = invocation_body.pop("self_hash", None)
    validated_adapter = _validated_qualification(
        invocation.adapter_qualification,
        component_id="raw_narrative_adapter",
    )
    if (
        invocation.receipt.get("schema")
        != ADAPTER_INVOCATION_RECEIPT_SCHEMA
        or invocation.receipt.get("status")
        != "EXACT_SOURCE_ADAPTER_INVOKED_ONCE"
        or invocation.receipt.get("source_sha256")
        != OFFICIAL_DATASET_SHA256
        or invocation.receipt.get("adapter_qualification_self_hash")
        != validated_adapter.receipt["self_hash"]
        or invocation.receipt.get("adapted_row_count") != len(rows)
        or invocation.receipt.get("adapted_output_commitment")
        != _content_hash([row.__dict__ for row in rows])
        or not _is_sha256(invocation_claimed)
        or _content_hash(invocation_body) != invocation_claimed
    ):
        raise ArnIntrinsicProtocolError(
            "validated adapter invocation drifted"
        )
    expected_ids = {
        str(value)
        for value in range(OFFICIAL_ID_MINIMUM, OFFICIAL_ID_MAXIMUM + 1)
        if value not in OFFICIAL_MISSING_IDS
    }
    observed_ids = {row.source_id for row in rows}
    if len(rows) != OFFICIAL_ROW_COUNT or observed_ids != expected_ids:
        raise ArnIntrinsicProtocolError(
            "official adapter output ID topology drifted"
        )
    observed_cells: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        _validate_adapted_row(row)
        observed_cells[
            f"{row.analogy_level}_{row.distractor_similarity}"
        ] += 1
    if dict(observed_cells) != OFFICIAL_CELL_COUNTS:
        raise ArnIntrinsicProtocolError(
            "official adapter output four-cell totals drifted"
        )
    bundle = _build_private_packs(
        rows,
        source_sha256=OFFICIAL_DATASET_SHA256,
        linkage_secret=linkage_secret,
        lineage="official_arn_measurement",
        schemas=(
            OFFICIAL_PREDICTOR_PACK_SCHEMA,
            OFFICIAL_LINKAGE_PACK_SCHEMA,
            OFFICIAL_LABEL_PACK_SCHEMA,
        ),
        source_verification_self_hash=invocation.receipt[
            "source_verification_self_hash"
        ],
        adapter_qualification_self_hash=validated_adapter.receipt[
            "self_hash"
        ],
        quarantine_source_id=IMPLEMENTATION_EXPOSURE_SOURCE_ID,
    )
    output: dict[str, Any] = {
        "schema": ADAPTER_OUTPUT_RECEIPT_SCHEMA,
        "status": "OFFICIAL_ADAPTER_OUTPUT_TOPOLOGY_VERIFIED",
        "adapter_invocation_self_hash": invocation.receipt["self_hash"],
        "source_verification_self_hash": invocation.receipt[
            "source_verification_self_hash"
        ],
        "adapter_qualification_self_hash": validated_adapter.receipt[
            "self_hash"
        ],
        "source_row_count": OFFICIAL_ROW_COUNT,
        "id_minimum": OFFICIAL_ID_MINIMUM,
        "id_maximum": OFFICIAL_ID_MAXIMUM,
        "missing_ids": list(OFFICIAL_MISSING_IDS),
        "four_cell_counts": dict(OFFICIAL_CELL_COUNTS),
        "pack_commitments": dict(bundle.pack_commitments),
        "implementation_exposure_quarantine_applied": True,
        "quarantine_original_bucket": IMPLEMENTATION_EXPOSURE_BUCKET,
        "public_group_digest_emitted": False,
        "item_content_emitted": False,
    }
    output["self_hash"] = _content_hash(output)
    return PrivatePackBundle(
        lineage=bundle.lineage,
        predictor_pack=bundle.predictor_pack,
        linkage_pack=bundle.linkage_pack,
        label_pack=bundle.label_pack,
        pack_commitments=bundle.pack_commitments,
        safe_split_aggregates=bundle.safe_split_aggregates,
        adapter_output_receipt=output,
    )


def _prediction_row(
    *,
    opaque_item_id_value: str,
    disposition: str,
    selected_choice: str | None,
    error_code: str | None,
) -> dict[str, Any]:
    if not _is_sha256(opaque_item_id_value):
        raise ArnIntrinsicProtocolError(
            "prediction opaque item id is invalid"
        )
    if disposition not in DISPOSITIONS:
        raise ArnIntrinsicProtocolError(
            "prediction disposition is invalid"
        )
    if disposition == "ANSWER":
        if selected_choice not in CHOICE_IDS or error_code is not None:
            raise ArnIntrinsicProtocolError(
                "ANSWER prediction schema is invalid"
            )
    elif disposition == "ABSTAIN":
        if selected_choice is not None or error_code is not None:
            raise ArnIntrinsicProtocolError(
                "ABSTAIN prediction schema is invalid"
            )
    elif selected_choice is not None or error_code not in ERROR_CODES:
        raise ArnIntrinsicProtocolError(
            "ERROR prediction schema is invalid"
        )
    return {
        "opaque_item_id": opaque_item_id_value,
        "disposition": disposition,
        "selected_choice": selected_choice,
        "error_code": error_code,
    }


def make_prediction_pack(
    *,
    arm_id: str,
    arm_implementation_sha256: str,
    arm_qualification_receipt_sha256: str,
    protocol_contract_sha256: str,
    predictor_pack_sha256: str,
    linkage_pack_sha256: str,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct and validate a strict, explanation-free prediction pack."""

    if arm_id not in ARM_IDS:
        raise ArnIntrinsicProtocolError("prediction arm id is invalid")
    for value in (
        arm_implementation_sha256,
        arm_qualification_receipt_sha256,
        protocol_contract_sha256,
        predictor_pack_sha256,
        linkage_pack_sha256,
    ):
        if not _is_sha256(value):
            raise ArnIntrinsicProtocolError(
                "prediction pack commitment is invalid"
            )
    normalized: list[dict[str, Any]] = []
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ArnIntrinsicProtocolError(
                "prediction row is not an object"
            )
        _require_exact_keys(
            prediction,
            {
                "opaque_item_id",
                "disposition",
                "selected_choice",
                "error_code",
            },
            label="prediction row",
        )
        normalized.append(
            _prediction_row(
                opaque_item_id_value=prediction["opaque_item_id"],
                disposition=prediction["disposition"],
                selected_choice=prediction["selected_choice"],
                error_code=prediction["error_code"],
            )
        )
    normalized.sort(key=lambda value: value["opaque_item_id"])
    if len({row["opaque_item_id"] for row in normalized}) != len(normalized):
        raise ArnIntrinsicProtocolError(
            "prediction pack contains a duplicate item"
        )
    body: dict[str, Any] = {
        "schema": PREDICTION_PACK_SCHEMA,
        "arm_id": arm_id,
        "arm_implementation_sha256": arm_implementation_sha256,
        "arm_qualification_receipt_sha256": (
            arm_qualification_receipt_sha256
        ),
        "protocol_contract_sha256": protocol_contract_sha256,
        "predictor_pack_sha256": predictor_pack_sha256,
        "linkage_pack_sha256": linkage_pack_sha256,
        "predictions": normalized,
    }
    body["self_hash"] = _content_hash(body)
    return body


def _validate_prediction_pack(
    pack: Mapping[str, Any],
    *,
    expected_arm_id: str,
    expected_protocol_contract_sha256: str,
    expected_predictor_pack_sha256: str,
    expected_linkage_pack_sha256: str,
    expected_item_ids: set[str],
) -> None:
    _require_exact_keys(
        pack,
        {
            "schema",
            "arm_id",
            "arm_implementation_sha256",
            "arm_qualification_receipt_sha256",
            "protocol_contract_sha256",
            "predictor_pack_sha256",
            "linkage_pack_sha256",
            "predictions",
            "self_hash",
        },
        label="prediction pack",
    )
    if (
        pack["schema"] != PREDICTION_PACK_SCHEMA
        or pack["arm_id"] != expected_arm_id
        or pack["protocol_contract_sha256"]
        != expected_protocol_contract_sha256
        or pack["predictor_pack_sha256"]
        != expected_predictor_pack_sha256
        or pack["linkage_pack_sha256"]
        != expected_linkage_pack_sha256
        or not _is_sha256(pack["arm_implementation_sha256"])
        or not _is_sha256(pack["arm_qualification_receipt_sha256"])
    ):
        raise ArnIntrinsicProtocolError(
            "prediction pack frozen binding drifted"
        )
    body = dict(pack)
    claimed = body.pop("self_hash")
    if not _is_sha256(claimed) or _content_hash(body) != claimed:
        raise ArnIntrinsicProtocolError(
            "prediction pack self hash drifted"
        )
    predictions = pack["predictions"]
    if not isinstance(predictions, list):
        raise ArnIntrinsicProtocolError(
            "prediction pack rows are not an array"
        )
    observed: set[str] = set()
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ArnIntrinsicProtocolError(
                "prediction row is not an object"
            )
        _require_exact_keys(
            prediction,
            {
                "opaque_item_id",
                "disposition",
                "selected_choice",
                "error_code",
            },
            label="prediction row",
        )
        normalized = _prediction_row(
            opaque_item_id_value=prediction["opaque_item_id"],
            disposition=prediction["disposition"],
            selected_choice=prediction["selected_choice"],
            error_code=prediction["error_code"],
        )
        if normalized["opaque_item_id"] in observed:
            raise ArnIntrinsicProtocolError(
                "prediction pack contains a duplicate item"
            )
        observed.add(normalized["opaque_item_id"])
    if observed != expected_item_ids:
        raise ArnIntrinsicProtocolError(
            "prediction pack does not cover the exact common input"
        )


def _measurement_item_ids(
    predictor_pack: Mapping[str, Any],
    linkage_pack: Mapping[str, Any],
) -> set[str]:
    _require_exact_keys(
        predictor_pack,
        {
            "schema",
            "lineage",
            "source_sha256",
            "source_verification_self_hash",
            "adapter_qualification_self_hash",
            "column_contract",
            "rows",
        },
        label="predictor pack",
    )
    _require_exact_keys(
        linkage_pack,
        {
            "schema",
            "lineage",
            "source_sha256",
            "source_verification_self_hash",
            "adapter_qualification_self_hash",
            "column_contract",
            "linkage_key_commitment",
            "public_split_digest_emitted",
            "rows",
        },
        label="linkage pack",
    )
    lineage = predictor_pack["lineage"]
    schemas = {
        "synthetic_qualification_fixture": (
            SYNTHETIC_PREDICTOR_PACK_SCHEMA,
            SYNTHETIC_LINKAGE_PACK_SCHEMA,
        ),
        "official_arn_measurement": (
            OFFICIAL_PREDICTOR_PACK_SCHEMA,
            OFFICIAL_LINKAGE_PACK_SCHEMA,
        ),
    }
    if (
        lineage not in schemas
        or predictor_pack["schema"] != schemas[lineage][0]
        or linkage_pack["schema"] != schemas[lineage][1]
        or linkage_pack["lineage"] != lineage
        or predictor_pack["source_sha256"] != linkage_pack["source_sha256"]
        or predictor_pack["source_verification_self_hash"]
        != linkage_pack["source_verification_self_hash"]
        or predictor_pack["adapter_qualification_self_hash"]
        != linkage_pack["adapter_qualification_self_hash"]
        or predictor_pack["column_contract"]
        != list(COLUMN_ACCESS_MATRIX["arms"])
        or linkage_pack["column_contract"]
        != list(COLUMN_ACCESS_MATRIX["splitter"])
        or not _is_sha256(linkage_pack["linkage_key_commitment"])
        or linkage_pack["public_split_digest_emitted"] is not False
        or not isinstance(predictor_pack["rows"], list)
        or not isinstance(linkage_pack["rows"], list)
    ):
        raise ArnIntrinsicProtocolError(
            "private predictor/linkage pack binding drifted"
        )
    if lineage == "official_arn_measurement":
        if (
            predictor_pack["source_sha256"] != OFFICIAL_DATASET_SHA256
            or not _is_sha256(
                predictor_pack["source_verification_self_hash"]
            )
            or not _is_sha256(
                predictor_pack["adapter_qualification_self_hash"]
            )
        ):
            raise ArnIntrinsicProtocolError(
                "official predictor/linkage closure drifted"
            )
    elif (
        predictor_pack["source_sha256"] == OFFICIAL_DATASET_SHA256
        or predictor_pack["source_verification_self_hash"] is not None
        or predictor_pack["adapter_qualification_self_hash"] is not None
    ):
        raise ArnIntrinsicProtocolError(
            "synthetic predictor/linkage impersonated official lineage"
        )
    predictor_ids: set[str] = set()
    for row in predictor_pack["rows"]:
        if not isinstance(row, Mapping):
            raise ArnIntrinsicProtocolError("predictor row is not an object")
        _require_exact_keys(
            row,
            {
                "opaque_item_id",
                "query_narrative",
                "first_choice",
                "second_choice",
            },
            label="predictor row",
        )
        if (
            not _is_sha256(row["opaque_item_id"])
            or not all(
                isinstance(row[field], str) and row[field]
                for field in (
                    "query_narrative",
                    "first_choice",
                    "second_choice",
                )
            )
            or row["opaque_item_id"] in predictor_ids
        ):
            raise ArnIntrinsicProtocolError(
                "predictor row value drifted"
            )
        predictor_ids.add(row["opaque_item_id"])

    eligible_ids: set[str] = set()
    seen_linkage_ids: set[str] = set()
    group_assignments: dict[str, tuple[int, str]] = {}
    quarantine_group_ids: set[str] = set()
    for row in linkage_pack["rows"]:
        if not isinstance(row, Mapping):
            raise ArnIntrinsicProtocolError("linkage row is not an object")
        _require_exact_keys(
            row,
            {
                "opaque_item_id",
                "private_group_id",
                "bucket",
                "hash_partition",
                "effective_partition",
                "measurement_eligible",
                "exclusion_codes",
            },
            label="linkage row",
        )
        item_id = row["opaque_item_id"]
        group_id = row["private_group_id"]
        bucket = row["bucket"]
        hash_partition = row["hash_partition"]
        if (
            not _is_sha256(item_id)
            or not _is_sha256(group_id)
            or type(bucket) is not int
            or bucket not in range(5)
            or hash_partition
            != ("calibration" if bucket == 0 else "measurement")
            or item_id in seen_linkage_ids
        ):
            raise ArnIntrinsicProtocolError(
                "linkage row value drifted"
            )
        seen_linkage_ids.add(item_id)
        prior = group_assignments.setdefault(
            group_id, (bucket, hash_partition)
        )
        if prior != (bucket, hash_partition):
            raise ArnIntrinsicProtocolError(
                "whole-proverb group crossed a hash partition"
            )
        if row["measurement_eligible"] is True:
            if (
                hash_partition != "measurement"
                or row["effective_partition"] != "measurement"
                or row["exclusion_codes"] != []
            ):
                raise ArnIntrinsicProtocolError(
                    "eligible linkage row drifted"
                )
            eligible_ids.add(item_id)
        elif row["exclusion_codes"] == ["IMPLEMENTATION_EXPOSURE"]:
            quarantine_group_ids.add(group_id)
            if (
                bucket != IMPLEMENTATION_EXPOSURE_BUCKET
                or row["effective_partition"]
                != "implementation_exposure_quarantine"
                or row["exclusion_codes"] != ["IMPLEMENTATION_EXPOSURE"]
            ):
                raise ArnIntrinsicProtocolError(
                    "implementation quarantine linkage drifted"
                )
        elif (
            bucket != 0
            or row["effective_partition"] != "calibration"
            or row["exclusion_codes"] != []
        ):
            raise ArnIntrinsicProtocolError(
                "non-measurement linkage row drifted"
            )
    if len(quarantine_group_ids) > 1:
        raise ArnIntrinsicProtocolError(
            "implementation quarantine crossed private proverb groups"
        )
    if lineage == "official_arn_measurement" and len(
        quarantine_group_ids
    ) != 1:
        raise ArnIntrinsicProtocolError(
            "official linkage pack lacks the exposure quarantine"
        )
    if lineage == "synthetic_qualification_fixture" and quarantine_group_ids:
        raise ArnIntrinsicProtocolError(
            "synthetic fixture contains an official exposure quarantine"
        )
    if predictor_ids != eligible_ids:
        raise ArnIntrinsicProtocolError(
            "predictor pack is not the exact eligible measurement set"
        )
    return predictor_ids


def _build_action_seal(
    *,
    protocol_contract_sha256: str,
    predictor_pack: Mapping[str, Any],
    linkage_pack: Mapping[str, Any],
    label_pack_sha256: str,
    prediction_packs: Mapping[str, Mapping[str, Any]],
    lifecycle: str,
    ready_freeze_manifest_self_hash: str | None,
    adapter_output_receipt_self_hash: str | None,
) -> dict[str, Any]:
    if (
        not _is_sha256(protocol_contract_sha256)
        or not _is_sha256(label_pack_sha256)
    ):
        raise ArnIntrinsicProtocolError(
            "action seal contract hash is invalid"
        )
    if set(prediction_packs) != set(ARM_IDS):
        raise ArnIntrinsicProtocolError(
            "all four frozen arms are required at the barrier"
        )
    item_ids = _measurement_item_ids(predictor_pack, linkage_pack)
    predictor_hash = _content_hash(predictor_pack)
    linkage_hash = _content_hash(linkage_pack)
    arm_hashes: dict[str, str] = {}
    arm_implementations: dict[str, str] = {}
    for arm_id in ARM_IDS:
        pack = prediction_packs[arm_id]
        _validate_prediction_pack(
            pack,
            expected_arm_id=arm_id,
            expected_protocol_contract_sha256=protocol_contract_sha256,
            expected_predictor_pack_sha256=predictor_hash,
            expected_linkage_pack_sha256=linkage_hash,
            expected_item_ids=item_ids,
        )
        arm_hashes[arm_id] = pack["self_hash"]
        arm_implementations[arm_id] = pack["arm_implementation_sha256"]
    if lifecycle == "synthetic_qualification_only":
        schema = QUALIFICATION_ACTION_SEAL_SCHEMA
        status = "SYNTHETIC_ACTIONS_SEALED_LABELS_UNOPENED"
        if (
            predictor_pack["lineage"] != "synthetic_qualification_fixture"
            or ready_freeze_manifest_self_hash is not None
            or adapter_output_receipt_self_hash is not None
        ):
            raise ArnIntrinsicProtocolError(
                "qualification seal lineage drifted"
            )
        formal_terminal_authorized = False
    elif lifecycle == "formal_official_measurement":
        schema = FORMAL_ACTION_SEAL_SCHEMA
        status = "FORMAL_ACTIONS_SEALED_LABELS_UNOPENED"
        if (
            predictor_pack["lineage"] != "official_arn_measurement"
            or not _is_sha256(ready_freeze_manifest_self_hash)
            or not _is_sha256(adapter_output_receipt_self_hash)
        ):
            raise ArnIntrinsicProtocolError("formal seal lineage drifted")
        formal_terminal_authorized = True
    else:
        raise ArnIntrinsicProtocolError("action seal lifecycle is invalid")
    body: dict[str, Any] = {
        "schema": schema,
        "status": status,
        "lifecycle": lifecycle,
        "ready_freeze_manifest_self_hash": (
            ready_freeze_manifest_self_hash
        ),
        "adapter_output_receipt_self_hash": (
            adapter_output_receipt_self_hash
        ),
        "protocol_contract_sha256": protocol_contract_sha256,
        "predictor_pack_sha256": predictor_hash,
        "linkage_pack_sha256": linkage_hash,
        "label_pack_sha256": label_pack_sha256,
        "common_measurement_item_count": len(item_ids),
        "common_measurement_item_set_commitment": _content_hash(
            sorted(item_ids)
        ),
        "arm_prediction_pack_sha256s": arm_hashes,
        "arm_implementation_sha256s": arm_implementations,
        "all_four_arms_present": True,
        "labels_opened": False,
        "formal_terminal_authorized": formal_terminal_authorized,
    }
    body["self_hash"] = _content_hash(body)
    return body


def build_qualification_action_seal(
    *,
    protocol_contract_sha256: str,
    predictor_pack: Mapping[str, Any],
    linkage_pack: Mapping[str, Any],
    label_pack_sha256: str,
    prediction_packs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create only a synthetic qualification action barrier."""

    return _build_action_seal(
        protocol_contract_sha256=protocol_contract_sha256,
        predictor_pack=predictor_pack,
        linkage_pack=linkage_pack,
        label_pack_sha256=label_pack_sha256,
        prediction_packs=prediction_packs,
        lifecycle="synthetic_qualification_only",
        ready_freeze_manifest_self_hash=None,
        adapter_output_receipt_self_hash=None,
    )


def build_all_arm_action_seal(
    *,
    ready_freeze: ValidatedReadyFreeze,
    adapter_output_receipt: Mapping[str, Any],
    predictor_pack: Mapping[str, Any],
    linkage_pack: Mapping[str, Any],
    label_pack_sha256: str,
    prediction_packs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a formal seal only from a validated ready freeze manifest."""

    ready_freeze_manifest = _validated_ready_freeze(ready_freeze)
    _validate_adapter_output_receipt(
        adapter_output_receipt,
        expected_pack_commitments={
            "predictor_pack_sha256": _content_hash(predictor_pack),
            "linkage_pack_sha256": _content_hash(linkage_pack),
            "label_pack_sha256": label_pack_sha256,
        },
    )
    if (
        ready_freeze_manifest["adapter_output_receipt_self_hash"]
        != adapter_output_receipt["self_hash"]
        or ready_freeze_manifest["pack_commitments"]
        != adapter_output_receipt["pack_commitments"]
    ):
        raise ArnIntrinsicProtocolError(
            "formal source-to-adapter-to-pack freeze binding drifted"
        )
    for arm_id in ARM_IDS:
        binding = ready_freeze_manifest["arm_closure_bindings"][arm_id]
        pack = prediction_packs.get(arm_id)
        if (
            not isinstance(pack, Mapping)
            or pack.get("arm_implementation_sha256")
            != binding["implementation_file_sha256"]
            or pack.get("arm_qualification_receipt_sha256")
            != binding["qualification_receipt_self_hash"]
        ):
            raise ArnIntrinsicProtocolError(
                "formal arm prediction closure drifted"
            )
    return _build_action_seal(
        protocol_contract_sha256=ready_freeze_manifest[
            "protocol_contract_sha256"
        ],
        predictor_pack=predictor_pack,
        linkage_pack=linkage_pack,
        label_pack_sha256=label_pack_sha256,
        prediction_packs=prediction_packs,
        lifecycle="formal_official_measurement",
        ready_freeze_manifest_self_hash=ready_freeze_manifest["self_hash"],
        adapter_output_receipt_self_hash=adapter_output_receipt["self_hash"],
    )


def _ensure_private_directory(
    path: Path, *, expected_uid: int | None = None
) -> None:
    uid = os.getuid() if expected_uid is None else expected_uid
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArnIntrinsicProtocolError(
            "private protocol directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ArnIntrinsicProtocolError(
            "private protocol directory owner or mode drifted"
        )


def _fsync_private_directory(
    path: Path, *, expected_uid: int | None = None
) -> None:
    _ensure_private_directory(path, expected_uid=expected_uid)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArnIntrinsicProtocolError(
                "private protocol directory changed during fsync"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_uid: int | None = None,
) -> str:
    raw = _canonical_bytes(value)
    uid = os.getuid() if expected_uid is None else expected_uid
    _ensure_private_directory(path.parent, expected_uid=uid)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != uid
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ArnIntrinsicProtocolError(
                "exclusive protocol file topology drifted"
            )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_private_directory(path.parent, expected_uid=uid)
    return hashlib.sha256(raw).hexdigest()


def _read_private_bytes_single_fd(
    path: Path,
    *,
    label: str,
    expected_uid: int | None = None,
    expected_mode: int = 0o600,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    uid = os.getuid() if expected_uid is None else expected_uid
    if expected_sha256 is not None and not _is_sha256(expected_sha256):
        raise ArnIntrinsicProtocolError(
            f"{label} expected SHA256 is invalid"
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ArnIntrinsicProtocolError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != uid
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise ArnIntrinsicProtocolError(
            f"{label} owner, mode, or topology drifted"
        )
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_uid != uid
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != expected_mode
        ):
            raise ArnIntrinsicProtocolError(
                f"{label} changed while opening"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or after.st_uid != uid
            or after.st_nlink != 1
            or stat.S_IMODE(after.st_mode) != expected_mode
        ):
            raise ArnIntrinsicProtocolError(
                f"{label} changed while reading"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    file_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and file_hash != expected_sha256:
        raise ArnIntrinsicProtocolError(f"{label} file SHA256 drifted")
    return raw, file_hash


def persist_action_seal_once(
    path: Path, action_seal: Mapping[str, Any]
) -> str:
    """Persist the complete all-arm barrier exactly once."""

    _validate_action_seal(action_seal)
    return _write_exclusive(path, action_seal)


def _validate_action_seal(action_seal: Mapping[str, Any]) -> None:
    _require_exact_keys(
        action_seal,
        {
            "schema",
            "status",
            "lifecycle",
            "ready_freeze_manifest_self_hash",
            "adapter_output_receipt_self_hash",
            "protocol_contract_sha256",
            "predictor_pack_sha256",
            "linkage_pack_sha256",
            "label_pack_sha256",
            "common_measurement_item_count",
            "common_measurement_item_set_commitment",
            "arm_prediction_pack_sha256s",
            "arm_implementation_sha256s",
            "all_four_arms_present",
            "labels_opened",
            "formal_terminal_authorized",
            "self_hash",
        },
        label="action seal",
    )
    body = dict(action_seal)
    claimed = body.pop("self_hash")
    arm_prediction_hashes = action_seal.get(
        "arm_prediction_pack_sha256s"
    )
    arm_implementation_hashes = action_seal.get(
        "arm_implementation_sha256s"
    )
    lifecycle = action_seal["lifecycle"]
    if lifecycle == "synthetic_qualification_only":
        lineage_valid = (
            action_seal["schema"] == QUALIFICATION_ACTION_SEAL_SCHEMA
            and action_seal["status"]
            == "SYNTHETIC_ACTIONS_SEALED_LABELS_UNOPENED"
            and action_seal["ready_freeze_manifest_self_hash"] is None
            and action_seal["adapter_output_receipt_self_hash"] is None
            and action_seal["formal_terminal_authorized"] is False
        )
    elif lifecycle == "formal_official_measurement":
        lineage_valid = (
            action_seal["schema"] == FORMAL_ACTION_SEAL_SCHEMA
            and action_seal["status"]
            == "FORMAL_ACTIONS_SEALED_LABELS_UNOPENED"
            and _is_sha256(
                action_seal["ready_freeze_manifest_self_hash"]
            )
            and _is_sha256(
                action_seal["adapter_output_receipt_self_hash"]
            )
            and action_seal["formal_terminal_authorized"] is True
        )
    else:
        lineage_valid = False
    if (
        not lineage_valid
        or action_seal["all_four_arms_present"] is not True
        or action_seal["labels_opened"] is not False
        or not _is_sha256(action_seal["protocol_contract_sha256"])
        or not _is_sha256(action_seal["predictor_pack_sha256"])
        or not _is_sha256(action_seal["linkage_pack_sha256"])
        or not _is_sha256(action_seal["label_pack_sha256"])
        or type(action_seal["common_measurement_item_count"]) is not int
        or action_seal["common_measurement_item_count"] < 0
        or not _is_sha256(
            action_seal["common_measurement_item_set_commitment"]
        )
        or not isinstance(arm_prediction_hashes, Mapping)
        or set(arm_prediction_hashes) != set(ARM_IDS)
        or not all(
            _is_sha256(value) for value in arm_prediction_hashes.values()
        )
        or not isinstance(arm_implementation_hashes, Mapping)
        or set(arm_implementation_hashes) != set(ARM_IDS)
        or not all(
            _is_sha256(value)
            for value in arm_implementation_hashes.values()
        )
        or not _is_sha256(claimed)
        or _content_hash(body) != claimed
    ):
        raise ArnIntrinsicProtocolError("action seal drifted")


def _load_canonical_mapping_single_fd(
    path: Path,
    *,
    label: str,
    expected_uid: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    raw, file_hash = _read_private_bytes_single_fd(
        path,
        label=label,
        expected_uid=expected_uid,
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArnIntrinsicProtocolError(
            f"{label} is unavailable"
        ) from exc
    if not isinstance(value, Mapping) or _canonical_bytes(value) != raw:
        raise ArnIntrinsicProtocolError(
            f"{label} is not canonical"
        )
    return value, file_hash


def _validate_label_pack(
    label_pack: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    expected_item_ids: set[str],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    _require_exact_keys(
        label_pack,
        {
            "schema",
            "lineage",
            "source_sha256",
            "source_verification_self_hash",
            "adapter_qualification_self_hash",
            "column_contract",
            "rows",
        },
        label="label pack",
    )
    expected_schema = {
        "synthetic_qualification_fixture": SYNTHETIC_LABEL_PACK_SCHEMA,
        "official_arn_measurement": OFFICIAL_LABEL_PACK_SCHEMA,
    }.get(label_pack["lineage"])
    if (
        expected_schema is None
        or label_pack["schema"] != expected_schema
        or label_pack["source_sha256"] != expected_source_sha256
        or label_pack["column_contract"]
        != list(COLUMN_ACCESS_MATRIX["scorer_only"])
        or not isinstance(label_pack["rows"], list)
    ):
        raise ArnIntrinsicProtocolError("label pack binding drifted")
    labels: dict[str, Mapping[str, Any]] = {}
    for row in label_pack["rows"]:
        if not isinstance(row, Mapping):
            raise ArnIntrinsicProtocolError("label row is not an object")
        _require_exact_keys(
            row,
            {
                "opaque_item_id",
                "gold_choice",
                "analogy_level",
                "distractor_similarity",
            },
            label="label row",
        )
        item_id = row["opaque_item_id"]
        if (
            not _is_sha256(item_id)
            or item_id in labels
            or row["gold_choice"] not in CHOICE_IDS
            or row["analogy_level"] not in ANALOGY_LEVELS
            or row["distractor_similarity"]
            not in DISTRACTOR_SIMILARITIES
        ):
            raise ArnIntrinsicProtocolError("label row value drifted")
        labels[item_id] = row
    if set(labels) != expected_item_ids:
        raise ArnIntrinsicProtocolError(
            "label pack does not match the sealed measurement set"
        )
    return labels, _content_hash(label_pack)


def _cluster_summary(
    observations: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    total = len(observations)
    correct = sum(value for _, value in observations)
    if total == 0:
        return {
            "correct": 0,
            "total": 0,
            "accuracy": None,
            "proverb_cluster_count": 0,
            "cluster_robust_standard_error": None,
            "normal_95_interval": None,
        }
    mean = correct / total
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for group_id, value in observations:
        by_group[group_id].append(value)
    group_count = len(by_group)
    if group_count < 2:
        standard_error: float | None = None
        interval: list[float] | None = None
    else:
        residual_squares = 0.0
        for values in by_group.values():
            residual = sum(value - mean for value in values)
            residual_squares += residual * residual
        variance = (
            group_count
            / (group_count - 1)
            * residual_squares
            / (total * total)
        )
        standard_error = math.sqrt(max(0.0, variance))
        critical = 1.959963984540054
        interval = [
            max(0.0, mean - critical * standard_error),
            min(1.0, mean + critical * standard_error),
        ]
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(mean, 12),
        "proverb_cluster_count": group_count,
        "cluster_robust_standard_error": (
            None if standard_error is None else round(standard_error, 12)
        ),
        "normal_95_interval": (
            None
            if interval is None
            else [round(value, 12) for value in interval]
        ),
    }


def _cluster_difference_summary(
    observations: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    total = len(observations)
    difference_sum = sum(value for _, value in observations)
    if total == 0:
        return {
            "difference_sum": 0,
            "total": 0,
            "mean_difference": None,
            "proverb_cluster_count": 0,
            "cluster_robust_standard_error": None,
            "normal_95_interval": None,
        }
    mean = difference_sum / total
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for group_id, value in observations:
        by_group[group_id].append(value)
    group_count = len(by_group)
    if group_count < 2:
        standard_error: float | None = None
        interval: list[float] | None = None
    else:
        residual_squares = sum(
            sum(value - mean for value in values) ** 2
            for values in by_group.values()
        )
        variance = (
            group_count
            / (group_count - 1)
            * residual_squares
            / (total * total)
        )
        standard_error = math.sqrt(max(0.0, variance))
        critical = 1.959963984540054
        interval = [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ]
    return {
        "difference_sum": difference_sum,
        "total": total,
        "mean_difference": round(mean, 12),
        "proverb_cluster_count": group_count,
        "cluster_robust_standard_error": (
            None if standard_error is None else round(standard_error, 12)
        ),
        "normal_95_interval": (
            None
            if interval is None
            else [round(value, 12) for value in interval]
        ),
    }


def _score_aggregates(
    *,
    action_seal: Mapping[str, Any],
    prediction_packs: Mapping[str, Mapping[str, Any]],
    linkage_pack: Mapping[str, Any],
    label_pack: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = linkage_pack["lineage"]
    predictor_schema = {
        "synthetic_qualification_fixture": (
            SYNTHETIC_PREDICTOR_PACK_SCHEMA
        ),
        "official_arn_measurement": OFFICIAL_PREDICTOR_PACK_SCHEMA,
    }.get(lineage)
    if predictor_schema is None:
        raise ArnIntrinsicProtocolError("scorer linkage lineage drifted")
    item_ids = _measurement_item_ids(
        {
            "schema": predictor_schema,
            "lineage": lineage,
            "source_sha256": linkage_pack["source_sha256"],
            "source_verification_self_hash": linkage_pack[
                "source_verification_self_hash"
            ],
            "adapter_qualification_self_hash": linkage_pack[
                "adapter_qualification_self_hash"
            ],
            "column_contract": list(COLUMN_ACCESS_MATRIX["arms"]),
            "rows": [
                {
                    "opaque_item_id": row["opaque_item_id"],
                    "query_narrative": "sealed",
                    "first_choice": "sealed",
                    "second_choice": "sealed",
                }
                for row in linkage_pack["rows"]
                if row["measurement_eligible"] is True
            ],
        },
        linkage_pack,
    )
    labels, label_hash = _validate_label_pack(
        label_pack,
        expected_source_sha256=linkage_pack["source_sha256"],
        expected_item_ids=item_ids,
    )
    if label_hash != action_seal["label_pack_sha256"]:
        raise ArnIntrinsicProtocolError(
            "opened label pack commitment drifted"
        )
    if (
        label_pack["lineage"] != lineage
        or label_pack["source_verification_self_hash"]
        != linkage_pack["source_verification_self_hash"]
        or label_pack["adapter_qualification_self_hash"]
        != linkage_pack["adapter_qualification_self_hash"]
    ):
        raise ArnIntrinsicProtocolError(
            "label pack source-to-adapter lineage drifted"
        )
    group_by_item = {
        row["opaque_item_id"]: row["private_group_id"]
        for row in linkage_pack["rows"]
        if row["measurement_eligible"] is True
    }
    arm_aggregates: dict[str, Any] = {}
    correctness_by_arm: dict[str, dict[str, int]] = {}
    for arm_id in ARM_IDS:
        pack = prediction_packs[arm_id]
        predictions = {
            row["opaque_item_id"]: row for row in pack["predictions"]
        }
        overall: list[tuple[str, int]] = []
        cells: dict[tuple[str, str], list[tuple[str, int]]] = {
            (level, similarity): []
            for level in ANALOGY_LEVELS
            for similarity in DISTRACTOR_SIMILARITIES
        }
        disposition_counts = {
            "ANSWER": 0,
            "ABSTAIN": 0,
            "ERROR": 0,
        }
        correctness_by_arm[arm_id] = {}
        for item_id in sorted(item_ids):
            prediction = predictions[item_id]
            label = labels[item_id]
            disposition_counts[prediction["disposition"]] += 1
            correct = int(
                prediction["disposition"] == "ANSWER"
                and prediction["selected_choice"] == label["gold_choice"]
            )
            correctness_by_arm[arm_id][item_id] = correct
            observation = (group_by_item[item_id], correct)
            overall.append(observation)
            cells[
                (
                    label["analogy_level"],
                    label["distractor_similarity"],
                )
            ].append(observation)
        arm_aggregates[arm_id] = {
            "overall": _cluster_summary(overall),
            "cells": {
                f"{level}_{similarity}": _cluster_summary(
                    cells[(level, similarity)]
                )
                for level in ANALOGY_LEVELS
                for similarity in DISTRACTOR_SIMILARITIES
            },
            "disposition_counts": disposition_counts,
            "abstain_counted_wrong": True,
            "error_counted_wrong": True,
        }
    paired_differences: dict[str, Any] = {}
    for comparator in (
        "semantic_only",
        "legacy_keyword",
        "flat_label_no_verifier",
    ):
        overall_differences: list[tuple[str, int]] = []
        cell_differences: dict[
            tuple[str, str], list[tuple[str, int]]
        ] = {
            (level, similarity): []
            for level in ANALOGY_LEVELS
            for similarity in DISTRACTOR_SIMILARITIES
        }
        for item_id in sorted(item_ids):
            label = labels[item_id]
            difference = (
                correctness_by_arm["full_gscl"][item_id]
                - correctness_by_arm[comparator][item_id]
            )
            observation = (group_by_item[item_id], difference)
            overall_differences.append(observation)
            cell_differences[
                (
                    label["analogy_level"],
                    label["distractor_similarity"],
                )
            ].append(observation)
        paired_differences[f"full_gscl_minus_{comparator}"] = {
            "overall": _cluster_difference_summary(
                overall_differences
            ),
            "cells": {
                f"{level}_{similarity}": (
                    _cluster_difference_summary(
                        cell_differences[(level, similarity)]
                    )
                )
                for level in ANALOGY_LEVELS
                for similarity in DISTRACTOR_SIMILARITIES
            },
            "same_item_pairing": True,
            "same_private_proverb_cluster_pairing": True,
            "effect_gate": False,
        }
    return {
        "arm_aggregates": arm_aggregates,
        "paired_differences": paired_differences,
    }


def _write_failure_terminal(
    *,
    path: Path,
    lifecycle: str,
    action_seal_self_hash: str,
    failure_code: str,
) -> None:
    failure: dict[str, Any] = {
        "schema": f"{VERSION}.immutable_failure_terminal.v1",
        "status": "FAILED_AFTER_LABEL_OPEN_ONE_SHOT_CLAIM",
        "lifecycle": lifecycle,
        "action_seal_self_hash": action_seal_self_hash,
        "failure_code": failure_code,
        "retry_or_replay_allowed": False,
        "item_content_emitted": False,
    }
    failure["self_hash"] = _content_hash(failure)
    try:
        _write_exclusive(path, failure)
    except (OSError, ArnIntrinsicProtocolError, FileExistsError):
        # The first immutable failure terminal, if any, remains authoritative.
        pass


def _open_labels_and_score_once(
    *,
    state_root: Path,
    action_seal_path: Path,
    expected_action_seal_file_sha256: str,
    prediction_packs: Mapping[str, Mapping[str, Any]],
    linkage_pack: Mapping[str, Any],
    label_loader: Callable[[], Mapping[str, Any]],
    expected_lifecycle: str,
    ready_freeze_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _is_sha256(expected_action_seal_file_sha256):
        raise ArnIntrinsicProtocolError(
            "expected action seal file SHA256 is invalid"
        )
    _ensure_private_directory(state_root)
    if expected_lifecycle == "synthetic_qualification_only":
        label_open_marker = (
            state_root / "qualification_labels_open.one_shot.safe.json"
        )
        terminal_path = (
            state_root / "qualification_aggregate_score.safe.json"
        )
        failure_path = (
            state_root / "qualification_failure_terminal.safe.json"
        )
        receipt_schema = QUALIFICATION_SCORE_RECEIPT_SCHEMA
        receipt_status = "SYNTHETIC_QUALIFICATION_SCORING_COMPLETE"
        if ready_freeze_manifest is not None:
            raise ArnIntrinsicProtocolError(
                "synthetic qualification cannot bind a formal freeze"
            )
    elif expected_lifecycle == "formal_official_measurement":
        label_open_marker = (
            state_root / "formal_labels_open.one_shot.safe.json"
        )
        terminal_path = state_root / "formal_aggregate_score.safe.json"
        failure_path = state_root / "formal_failure_terminal.safe.json"
        receipt_schema = FORMAL_SCORE_RECEIPT_SCHEMA
        receipt_status = "FORMAL_OFFLINE_AGGREGATE_SCORING_COMPLETE"
        if ready_freeze_manifest is None:
            raise ArnIntrinsicProtocolError(
                "formal scoring requires the validated ready freeze"
            )
        _validate_ready_freeze_manifest(ready_freeze_manifest)
    else:
        raise ArnIntrinsicProtocolError("scoring lifecycle is invalid")
    if label_open_marker.exists() or terminal_path.exists() or failure_path.exists():
        raise ArnIntrinsicProtocolError(
            "measurement labels have already been opened"
        )
    action_seal, actual_seal_file_hash = (
        _load_canonical_mapping_single_fd(
            action_seal_path,
            label="all-arm action seal",
            expected_sha256=expected_action_seal_file_sha256,
        )
    )
    _validate_action_seal(action_seal)
    if (
        actual_seal_file_hash != expected_action_seal_file_sha256
        or action_seal["lifecycle"] != expected_lifecycle
    ):
        raise ArnIntrinsicProtocolError(
            "all-arm action seal file commitment drifted"
        )
    if ready_freeze_manifest is not None and (
        action_seal["ready_freeze_manifest_self_hash"]
        != ready_freeze_manifest["self_hash"]
        or action_seal["protocol_contract_sha256"]
        != ready_freeze_manifest["protocol_contract_sha256"]
    ):
        raise ArnIntrinsicProtocolError(
            "formal action seal ready-freeze binding drifted"
        )
    if _content_hash(linkage_pack) != action_seal["linkage_pack_sha256"]:
        raise ArnIntrinsicProtocolError(
            "linkage pack changed after the all-arm seal"
        )
    item_ids = {
        row["opaque_item_id"]
        for row in linkage_pack.get("rows", [])
        if isinstance(row, Mapping)
        and row.get("measurement_eligible") is True
    }
    if (
        action_seal["common_measurement_item_count"] != len(item_ids)
        or action_seal["common_measurement_item_set_commitment"]
        != _content_hash(sorted(item_ids))
    ):
        raise ArnIntrinsicProtocolError(
            "sealed common measurement item set drifted"
        )
    for arm_id in ARM_IDS:
        if arm_id not in prediction_packs:
            raise ArnIntrinsicProtocolError(
                "all four prediction packs are required for scoring"
            )
        _validate_prediction_pack(
            prediction_packs[arm_id],
            expected_arm_id=arm_id,
            expected_protocol_contract_sha256=action_seal[
                "protocol_contract_sha256"
            ],
            expected_predictor_pack_sha256=action_seal[
                "predictor_pack_sha256"
            ],
            expected_linkage_pack_sha256=action_seal[
                "linkage_pack_sha256"
            ],
            expected_item_ids=item_ids,
        )
        if (
            prediction_packs[arm_id]["self_hash"]
            != action_seal["arm_prediction_pack_sha256s"][arm_id]
        ):
            raise ArnIntrinsicProtocolError(
                "prediction pack changed after the all-arm seal"
            )

    marker: dict[str, Any] = {
        "schema": f"{VERSION}.label_open_one_shot.v1",
        "status": "CLAIMED_AFTER_ALL_ARM_ACTION_SEAL",
        "action_seal_self_hash": action_seal["self_hash"],
        "label_pack_sha256": action_seal["label_pack_sha256"],
    }
    marker["self_hash"] = _content_hash(marker)
    try:
        _write_exclusive(label_open_marker, marker)
    except FileExistsError as exc:
        raise ArnIntrinsicProtocolError(
            "measurement labels have already been opened"
        ) from exc

    try:
        label_pack = label_loader()
    except Exception as exc:
        _write_failure_terminal(
            path=failure_path,
            lifecycle=expected_lifecycle,
            action_seal_self_hash=action_seal["self_hash"],
            failure_code="LABEL_LOADER_EXCEPTION",
        )
        raise ArnIntrinsicProtocolError(
            "label loader failed after the one-shot claim"
        ) from exc
    try:
        if not isinstance(label_pack, Mapping):
            raise ArnIntrinsicProtocolError(
                "label loader did not return a mapping"
            )
        scoring = _score_aggregates(
            action_seal=action_seal,
            prediction_packs=prediction_packs,
            linkage_pack=linkage_pack,
            label_pack=label_pack,
        )
    except Exception:
        _write_failure_terminal(
            path=failure_path,
            lifecycle=expected_lifecycle,
            action_seal_self_hash=action_seal["self_hash"],
            failure_code="LABEL_PACK_OR_SCORER_INVALID",
        )
        raise
    body: dict[str, Any] = {
        "schema": receipt_schema,
        "status": receipt_status,
        "lifecycle": expected_lifecycle,
        "formal_terminal": (
            expected_lifecycle == "formal_official_measurement"
        ),
        "online_or_api_evaluator_used": False,
        "effect_gate": False,
        "retry_or_resample": False,
        "action_seal_self_hash": action_seal["self_hash"],
        "label_pack_sha256": action_seal["label_pack_sha256"],
        "arm_aggregates": scoring["arm_aggregates"],
        "paired_differences": scoring["paired_differences"],
        "uncertainty_method": (
            "intercept_only_cluster_robust_sandwich_by_opaque_proverb_"
            "cluster_with_G_over_G_minus_1_and_clipped_normal_95_interval"
        ),
        "abstain_and_error_counted_wrong": True,
        "per_item_content_emitted": False,
    }
    body["nested_hashes"] = {
        "arm_aggregates": _content_hash(body["arm_aggregates"]),
        "paired_differences": _content_hash(body["paired_differences"]),
        "metric_contract": _content_hash(
            {
                "uncertainty_method": body["uncertainty_method"],
                "abstain_and_error_counted_wrong": True,
                "effect_gate": False,
            }
        ),
    }
    body["self_hash"] = _content_hash(body)
    try:
        _write_exclusive(terminal_path, body)
    except Exception:
        _write_failure_terminal(
            path=failure_path,
            lifecycle=expected_lifecycle,
            action_seal_self_hash=action_seal["self_hash"],
            failure_code="TERMINAL_PERSISTENCE_FAILED",
        )
        raise
    return body


def open_labels_and_score_qualification_once(
    *,
    state_root: Path,
    action_seal_path: Path,
    expected_action_seal_file_sha256: str,
    prediction_packs: Mapping[str, Mapping[str, Any]],
    linkage_pack: Mapping[str, Any],
    label_loader: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run only synthetic qualification scoring; never a formal terminal."""

    return _open_labels_and_score_once(
        state_root=state_root,
        action_seal_path=action_seal_path,
        expected_action_seal_file_sha256=(
            expected_action_seal_file_sha256
        ),
        prediction_packs=prediction_packs,
        linkage_pack=linkage_pack,
        label_loader=label_loader,
        expected_lifecycle="synthetic_qualification_only",
        ready_freeze_manifest=None,
    )


def open_labels_and_score_once(
    *,
    state_root: Path,
    action_seal_path: Path,
    expected_action_seal_file_sha256: str,
    prediction_packs: Mapping[str, Mapping[str, Any]],
    linkage_pack: Mapping[str, Any],
    label_loader: Callable[[], Mapping[str, Any]],
    ready_freeze: ValidatedReadyFreeze,
) -> dict[str, Any]:
    """Run formal scoring only after a validated official ready freeze."""

    ready_freeze_manifest = _validated_ready_freeze(ready_freeze)
    return _open_labels_and_score_once(
        state_root=state_root,
        action_seal_path=action_seal_path,
        expected_action_seal_file_sha256=(
            expected_action_seal_file_sha256
        ),
        prediction_packs=prediction_packs,
        linkage_pack=linkage_pack,
        label_loader=label_loader,
        expected_lifecycle="formal_official_measurement",
        ready_freeze_manifest=ready_freeze_manifest,
    )


_QUALIFICATION_VALIDATION_TOKEN = object()
_MATERIALIZATION_VALIDATION_TOKEN = object()
_RUNTIME_VALIDATION_TOKEN = object()
_READY_FREEZE_VALIDATION_TOKEN = object()


@dataclass(frozen=True)
class ValidatedImplementationQualification:
    component_id: str
    receipt: Mapping[str, Any]
    receipt_path: Path
    receipt_file_sha256: str
    implementation_path: Path
    implementation_file_sha256: str
    qualification_test_path: Path
    qualification_test_file_sha256: str
    _validation_token: object


@dataclass(frozen=True)
class ValidatedRuntimeAccess:
    receipt: Mapping[str, Any]
    receipt_path: Path
    receipt_file_sha256: str
    _validation_token: object


@dataclass(frozen=True)
class ValidatedMaterialization:
    receipt: Mapping[str, Any]
    root: Path
    arm_uid: int
    custodian_uid: int
    _validation_token: object


@dataclass(frozen=True)
class ValidatedReadyFreeze:
    manifest: Mapping[str, Any]
    manifest_path: Path
    manifest_file_sha256: str
    _validation_token: object


def _read_bound_implementation_file(
    path: Path, *, expected_sha256: str, label: str
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArnIntrinsicProtocolError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ArnIntrinsicProtocolError(f"{label} topology drifted")
    return _read_exact_regular_file(
        path,
        size=metadata.st_size,
        sha256=expected_sha256,
        label=label,
    )


def _validate_implementation_qualification_mapping(
    receipt: Mapping[str, Any], *, expected_component_id: str
) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "component_id",
            "implementation_file_sha256",
            "qualification_test_file_sha256",
            "implementation_closure_sha256",
            "source_free",
            "measurement_content_accessed",
            "formal_measurement_run",
            "qualification_scope",
            "self_hash",
        },
        label="implementation qualification receipt",
    )
    body = dict(receipt)
    claimed = body.pop("self_hash")
    closure = {
        "component_id": receipt["component_id"],
        "implementation_file_sha256": receipt[
            "implementation_file_sha256"
        ],
        "qualification_test_file_sha256": receipt[
            "qualification_test_file_sha256"
        ],
        "qualification_scope": receipt["qualification_scope"],
    }
    if (
        receipt["schema"] != IMPLEMENTATION_QUALIFICATION_SCHEMA
        or receipt["status"]
        != "QUALIFIED_SOURCE_FREE_IMPLEMENTATION_CLOSURE"
        or receipt["component_id"] != expected_component_id
        or not _is_sha256(receipt["implementation_file_sha256"])
        or not _is_sha256(receipt["qualification_test_file_sha256"])
        or receipt["implementation_closure_sha256"]
        != _content_hash(closure)
        or receipt["source_free"] is not True
        or receipt["measurement_content_accessed"] is not False
        or receipt["formal_measurement_run"] is not False
        or receipt["qualification_scope"]
        != "synthetic_and_source_free_mechanics_only"
        or not _is_sha256(claimed)
        or _content_hash(body) != claimed
    ):
        raise ArnIntrinsicProtocolError(
            "implementation qualification receipt drifted"
        )


def validate_implementation_qualification_file(
    *,
    receipt_path: Path,
    expected_receipt_file_sha256: str,
    implementation_path: Path,
    qualification_test_path: Path,
    expected_component_id: str,
) -> ValidatedImplementationQualification:
    """Validate canonical qualification and both exact bound code files."""

    allowed_components = {"raw_narrative_adapter", *ARM_IDS}
    if expected_component_id not in allowed_components:
        raise ArnIntrinsicProtocolError(
            "qualification component id is not frozen"
        )
    receipt, receipt_file_hash = _load_canonical_mapping_single_fd(
        receipt_path,
        label="implementation qualification receipt",
        expected_sha256=expected_receipt_file_sha256,
    )
    _validate_implementation_qualification_mapping(
        receipt, expected_component_id=expected_component_id
    )
    _read_bound_implementation_file(
        implementation_path,
        expected_sha256=receipt["implementation_file_sha256"],
        label="qualified implementation",
    )
    _read_bound_implementation_file(
        qualification_test_path,
        expected_sha256=receipt["qualification_test_file_sha256"],
        label="qualification test",
    )
    return ValidatedImplementationQualification(
        component_id=expected_component_id,
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_file_sha256=receipt_file_hash,
        implementation_path=implementation_path,
        implementation_file_sha256=receipt[
            "implementation_file_sha256"
        ],
        qualification_test_path=qualification_test_path,
        qualification_test_file_sha256=receipt[
            "qualification_test_file_sha256"
        ],
        _validation_token=_QUALIFICATION_VALIDATION_TOKEN,
    )


def _validated_qualification(
    value: object, *, component_id: str
) -> ValidatedImplementationQualification:
    if (
        not isinstance(value, ValidatedImplementationQualification)
        or value._validation_token is not _QUALIFICATION_VALIDATION_TOKEN
        or value.component_id != component_id
    ):
        raise ArnIntrinsicProtocolError(
            "implementation closure was not validated from exact files"
        )
    _validate_implementation_qualification_mapping(
        value.receipt, expected_component_id=component_id
    )
    current_receipt, current_receipt_file_hash = (
        _load_canonical_mapping_single_fd(
            value.receipt_path,
            label="implementation qualification receipt",
            expected_sha256=value.receipt_file_sha256,
        )
    )
    if current_receipt != value.receipt:
        raise ArnIntrinsicProtocolError(
            "validated qualification receipt changed on disk"
        )
    _read_bound_implementation_file(
        value.implementation_path,
        expected_sha256=value.implementation_file_sha256,
        label="qualified implementation",
    )
    _read_bound_implementation_file(
        value.qualification_test_path,
        expected_sha256=value.qualification_test_file_sha256,
        label="qualification test",
    )
    if (
        value.receipt_file_sha256 != current_receipt_file_hash
        or value.implementation_file_sha256
        != value.receipt["implementation_file_sha256"]
        or value.qualification_test_file_sha256
        != value.receipt["qualification_test_file_sha256"]
    ):
        raise ArnIntrinsicProtocolError(
            "validated implementation closure drifted"
        )
    return value


def materialize_qualification_capabilities_once(
    *,
    root: Path,
    bundle: PrivatePackBundle,
) -> dict[str, Any]:
    """Materialize synthetic packs separately without claiming UID isolation."""

    if bundle.lineage != "synthetic_qualification_fixture":
        raise ArnIntrinsicProtocolError(
            "qualification materializer accepts synthetic bundles only"
        )
    _ensure_private_directory(root)
    arm_root = root / "arm_capability"
    custodian_root = root / "custodian_capability"
    _ensure_private_directory(arm_root)
    _ensure_private_directory(custodian_root)
    predictor_file_hash = _write_exclusive(
        arm_root / "predictor.private.json", bundle.predictor_pack
    )
    linkage_file_hash = _write_exclusive(
        custodian_root / "linkage.private.json", bundle.linkage_pack
    )
    label_file_hash = _write_exclusive(
        custodian_root / "labels.private.json", bundle.label_pack
    )
    receipt: dict[str, Any] = {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "SYNTHETIC_QUALIFICATION_MATERIALIZED",
        "lineage": bundle.lineage,
        "formal_ready": False,
        "arm_uid": os.getuid(),
        "custodian_uid": os.getuid(),
        "uid_separated": False,
        "directory_mode": "0700",
        "file_mode": "0600",
        "arm_visible_pack_classes": ["predictor"],
        "custodian_visible_pack_classes": ["linkage", "label"],
        "predictor_pack_file_sha256": predictor_file_hash,
        "linkage_pack_file_sha256": linkage_file_hash,
        "label_pack_file_sha256": label_file_hash,
        "pack_commitments": dict(bundle.pack_commitments),
    }
    receipt["self_hash"] = _content_hash(receipt)
    return receipt


def audit_formal_capability_materialization(
    *,
    root: Path,
    arm_uid: int,
    custodian_uid: int,
    pack_commitments: Mapping[str, str],
) -> ValidatedMaterialization:
    """Audit pre-materialized packs owned by distinct OS capabilities."""

    if arm_uid == custodian_uid:
        raise ArnIntrinsicProtocolError(
            "formal arm and custodian UIDs must be distinct"
        )
    arm_root = root / "arm_capability"
    custodian_root = root / "custodian_capability"
    _ensure_private_directory(arm_root, expected_uid=arm_uid)
    _ensure_private_directory(custodian_root, expected_uid=custodian_uid)
    predictor_pack, predictor_hash = _load_canonical_mapping_single_fd(
        arm_root / "predictor.private.json",
        label="formal predictor capability",
        expected_uid=arm_uid,
    )
    linkage_pack, linkage_hash = _load_canonical_mapping_single_fd(
        custodian_root / "linkage.private.json",
        label="formal linkage capability",
        expected_uid=custodian_uid,
    )
    label_pack, label_hash = _load_canonical_mapping_single_fd(
        custodian_root / "labels.private.json",
        label="formal label capability",
        expected_uid=custodian_uid,
    )
    if (
        _content_hash(predictor_pack)
        != pack_commitments.get("predictor_pack_sha256")
        or _content_hash(linkage_pack)
        != pack_commitments.get("linkage_pack_sha256")
        or _content_hash(label_pack)
        != pack_commitments.get("label_pack_sha256")
    ):
        raise ArnIntrinsicProtocolError(
            "formal materialized pack commitment drifted"
        )
    receipt: dict[str, Any] = {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "FORMAL_CAPABILITIES_MATERIALIZED_AND_AUDITED",
        "lineage": "official_arn_measurement",
        "formal_ready": True,
        "arm_uid": arm_uid,
        "custodian_uid": custodian_uid,
        "uid_separated": True,
        "directory_mode": "0700",
        "file_mode": "0600",
        "arm_visible_pack_classes": ["predictor"],
        "custodian_visible_pack_classes": ["linkage", "label"],
        "predictor_pack_file_sha256": predictor_hash,
        "linkage_pack_file_sha256": linkage_hash,
        "label_pack_file_sha256": label_hash,
        "pack_commitments": dict(pack_commitments),
    }
    receipt["self_hash"] = _content_hash(receipt)
    _validate_materialization_receipt(
        receipt,
        expected_pack_commitments=pack_commitments,
        require_formal=True,
    )
    return ValidatedMaterialization(
        receipt=receipt,
        root=root,
        arm_uid=arm_uid,
        custodian_uid=custodian_uid,
        _validation_token=_MATERIALIZATION_VALIDATION_TOKEN,
    )


def _validate_materialization_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_pack_commitments: Mapping[str, str],
    require_formal: bool,
) -> None:
    expected_keys = {
        "schema",
        "status",
        "lineage",
        "formal_ready",
        "arm_uid",
        "custodian_uid",
        "uid_separated",
        "directory_mode",
        "file_mode",
        "arm_visible_pack_classes",
        "custodian_visible_pack_classes",
        "predictor_pack_file_sha256",
        "linkage_pack_file_sha256",
        "label_pack_file_sha256",
        "pack_commitments",
        "self_hash",
    }
    _require_exact_keys(
        receipt, expected_keys, label="capability materialization receipt"
    )
    body = dict(receipt)
    claimed = body.pop("self_hash")
    formal_fields_valid = (
        receipt["status"]
        == "FORMAL_CAPABILITIES_MATERIALIZED_AND_AUDITED"
        and receipt["lineage"] == "official_arn_measurement"
        and receipt["formal_ready"] is True
        and receipt["uid_separated"] is True
        and type(receipt["arm_uid"]) is int
        and type(receipt["custodian_uid"]) is int
        and receipt["arm_uid"] != receipt["custodian_uid"]
    )
    common_valid = (
        receipt["schema"] == MATERIALIZATION_RECEIPT_SCHEMA
        and receipt["directory_mode"] == "0700"
        and receipt["file_mode"] == "0600"
        and receipt["arm_visible_pack_classes"] == ["predictor"]
        and receipt["custodian_visible_pack_classes"]
        == ["linkage", "label"]
        and receipt["pack_commitments"] == expected_pack_commitments
        and _is_sha256(receipt["predictor_pack_file_sha256"])
        and _is_sha256(receipt["linkage_pack_file_sha256"])
        and _is_sha256(receipt["label_pack_file_sha256"])
        and _is_sha256(claimed)
        and _content_hash(body) == claimed
    )
    if not common_valid or (require_formal and not formal_fields_valid):
        raise ArnIntrinsicProtocolError(
            "capability materialization receipt drifted"
        )


def validate_runtime_access_receipt_file(
    *,
    receipt_path: Path,
    expected_receipt_file_sha256: str,
    materialization: ValidatedMaterialization,
) -> ValidatedRuntimeAccess:
    """Validate observed arm denial of linkage and label capabilities."""

    materialization_receipt = _validated_materialization(materialization)
    receipt, file_hash = _load_canonical_mapping_single_fd(
        receipt_path,
        label="runtime access qualification receipt",
        expected_sha256=expected_receipt_file_sha256,
    )
    _require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "materialization_self_hash",
            "arm_uid",
            "custodian_uid",
            "arm_predictor_read",
            "arm_linkage_open_result",
            "arm_label_open_result",
            "custodian_linkage_read",
            "custodian_label_read",
            "source_free",
            "measurement_content_accessed",
            "self_hash",
        },
        label="runtime access qualification receipt",
    )
    body = dict(receipt)
    claimed = body.pop("self_hash")
    if (
        receipt["schema"] != RUNTIME_ACCESS_RECEIPT_SCHEMA
        or receipt["status"]
        != "QUALIFIED_DISTINCT_UID_CAPABILITY_DENIAL"
        or receipt["materialization_self_hash"]
        != materialization_receipt["self_hash"]
        or receipt["arm_uid"] != materialization_receipt["arm_uid"]
        or receipt["custodian_uid"]
        != materialization_receipt["custodian_uid"]
        or receipt["arm_predictor_read"] is not True
        or receipt["arm_linkage_open_result"] != "EACCES"
        or receipt["arm_label_open_result"] != "EACCES"
        or receipt["custodian_linkage_read"] is not True
        or receipt["custodian_label_read"] is not True
        or receipt["source_free"] is not True
        or receipt["measurement_content_accessed"] is not False
        or not _is_sha256(claimed)
        or _content_hash(body) != claimed
    ):
        raise ArnIntrinsicProtocolError(
            "runtime capability access receipt drifted"
        )
    return ValidatedRuntimeAccess(
        receipt=receipt,
        receipt_path=receipt_path,
        receipt_file_sha256=file_hash,
        _validation_token=_RUNTIME_VALIDATION_TOKEN,
    )


def _validate_runtime_access(
    runtime_access: ValidatedRuntimeAccess,
    *,
    materialization: ValidatedMaterialization,
) -> None:
    if (
        not isinstance(runtime_access, ValidatedRuntimeAccess)
        or runtime_access._validation_token is not _RUNTIME_VALIDATION_TOKEN
    ):
        raise ArnIntrinsicProtocolError(
            "runtime access closure was not validated from an exact file"
        )
    materialization_receipt = _validated_materialization(materialization)
    current_receipt, current_file_hash = (
        _load_canonical_mapping_single_fd(
            runtime_access.receipt_path,
            label="runtime access qualification receipt",
            expected_sha256=runtime_access.receipt_file_sha256,
        )
    )
    if (
        runtime_access.receipt["materialization_self_hash"]
        != materialization_receipt["self_hash"]
        or current_receipt != runtime_access.receipt
        or current_file_hash != runtime_access.receipt_file_sha256
        or runtime_access.receipt_file_sha256
        != hashlib.sha256(
            _canonical_bytes(runtime_access.receipt)
        ).hexdigest()
    ):
        raise ArnIntrinsicProtocolError(
            "runtime access closure was not validated from an exact file"
        )


def _validated_materialization(
    materialization: ValidatedMaterialization,
) -> Mapping[str, Any]:
    if (
        not isinstance(materialization, ValidatedMaterialization)
        or materialization._validation_token
        is not _MATERIALIZATION_VALIDATION_TOKEN
    ):
        raise ArnIntrinsicProtocolError(
            "materialization was not validated from filesystem capabilities"
        )
    receipt = materialization.receipt
    _validate_materialization_receipt(
        receipt,
        expected_pack_commitments=receipt.get("pack_commitments", {}),
        require_formal=True,
    )
    arm_root = materialization.root / "arm_capability"
    custodian_root = materialization.root / "custodian_capability"
    predictor, predictor_file_hash = _load_canonical_mapping_single_fd(
        arm_root / "predictor.private.json",
        label="formal predictor capability",
        expected_uid=materialization.arm_uid,
        expected_sha256=receipt["predictor_pack_file_sha256"],
    )
    linkage, linkage_file_hash = _load_canonical_mapping_single_fd(
        custodian_root / "linkage.private.json",
        label="formal linkage capability",
        expected_uid=materialization.custodian_uid,
        expected_sha256=receipt["linkage_pack_file_sha256"],
    )
    labels, label_file_hash = _load_canonical_mapping_single_fd(
        custodian_root / "labels.private.json",
        label="formal label capability",
        expected_uid=materialization.custodian_uid,
        expected_sha256=receipt["label_pack_file_sha256"],
    )
    if (
        predictor_file_hash != receipt["predictor_pack_file_sha256"]
        or linkage_file_hash != receipt["linkage_pack_file_sha256"]
        or label_file_hash != receipt["label_pack_file_sha256"]
        or _content_hash(predictor)
        != receipt["pack_commitments"]["predictor_pack_sha256"]
        or _content_hash(linkage)
        != receipt["pack_commitments"]["linkage_pack_sha256"]
        or _content_hash(labels)
        != receipt["pack_commitments"]["label_pack_sha256"]
    ):
        raise ArnIntrinsicProtocolError(
            "validated materialized capabilities changed on disk"
        )
    return receipt


def _validate_adapter_output_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_pack_commitments: Mapping[str, str],
) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema",
            "status",
            "adapter_invocation_self_hash",
            "source_verification_self_hash",
            "adapter_qualification_self_hash",
            "source_row_count",
            "id_minimum",
            "id_maximum",
            "missing_ids",
            "four_cell_counts",
            "pack_commitments",
            "implementation_exposure_quarantine_applied",
            "quarantine_original_bucket",
            "public_group_digest_emitted",
            "item_content_emitted",
            "self_hash",
        },
        label="official adapter output receipt",
    )
    body = dict(receipt)
    claimed = body.pop("self_hash")
    if (
        receipt["schema"] != ADAPTER_OUTPUT_RECEIPT_SCHEMA
        or receipt["status"]
        != "OFFICIAL_ADAPTER_OUTPUT_TOPOLOGY_VERIFIED"
        or not _is_sha256(receipt["adapter_invocation_self_hash"])
        or not _is_sha256(receipt["source_verification_self_hash"])
        or not _is_sha256(receipt["adapter_qualification_self_hash"])
        or receipt["source_row_count"] != OFFICIAL_ROW_COUNT
        or receipt["id_minimum"] != OFFICIAL_ID_MINIMUM
        or receipt["id_maximum"] != OFFICIAL_ID_MAXIMUM
        or receipt["missing_ids"] != list(OFFICIAL_MISSING_IDS)
        or receipt["four_cell_counts"] != OFFICIAL_CELL_COUNTS
        or receipt["pack_commitments"] != expected_pack_commitments
        or receipt["implementation_exposure_quarantine_applied"] is not True
        or receipt["quarantine_original_bucket"]
        != IMPLEMENTATION_EXPOSURE_BUCKET
        or receipt["public_group_digest_emitted"] is not False
        or receipt["item_content_emitted"] is not False
        or not _is_sha256(claimed)
        or _content_hash(body) != claimed
    ):
        raise ArnIntrinsicProtocolError(
            "official adapter output receipt drifted"
        )


def build_safe_protocol_receipt(
    *,
    source_verification: Mapping[str, Any] | None = None,
    implementation_qualifications: Mapping[
        str, ValidatedImplementationQualification
    ]
    | None = None,
    official_bundle: PrivatePackBundle | None = None,
    materialization: ValidatedMaterialization | None = None,
    runtime_access: ValidatedRuntimeAccess | None = None,
) -> dict[str, Any]:
    """Build a safe receipt; raw hashes can never establish readiness."""

    source_verified = False
    if source_verification is not None:
        _validate_source_verification_receipt(source_verification)
        source_verified = True

    required_components = ("raw_narrative_adapter", *ARM_IDS)
    validated_components: dict[
        str, ValidatedImplementationQualification
    ] = {}
    if implementation_qualifications is not None:
        if not set(implementation_qualifications).issubset(
            set(required_components)
        ):
            raise ArnIntrinsicProtocolError(
                "implementation qualification component set drifted"
            )
        for component_id, qualification in (
            implementation_qualifications.items()
        ):
            validated_components[component_id] = _validated_qualification(
                qualification, component_id=component_id
            )
    adapter_ready = "raw_narrative_adapter" in validated_components
    all_arms_ready = all(
        arm_id in validated_components for arm_id in ARM_IDS
    )

    pack_closure_ready = False
    pack_commitments: Mapping[str, str] | None = None
    adapter_output_receipt: Mapping[str, Any] | None = None
    if official_bundle is not None:
        if (
            official_bundle.lineage != "official_arn_measurement"
            or official_bundle.adapter_output_receipt is None
            or not source_verified
            or not adapter_ready
        ):
            raise ArnIntrinsicProtocolError(
                "official adapter-to-pack bundle is premature"
            )
        pack_commitments = official_bundle.pack_commitments
        adapter_output_receipt = official_bundle.adapter_output_receipt
        _validate_adapter_output_receipt(
            adapter_output_receipt,
            expected_pack_commitments=pack_commitments,
        )
        if (
            adapter_output_receipt["source_verification_self_hash"]
            != source_verification["self_hash"]
            or adapter_output_receipt[
                "adapter_qualification_self_hash"
            ]
            != validated_components[
                "raw_narrative_adapter"
            ].receipt["self_hash"]
        ):
            raise ArnIntrinsicProtocolError(
                "official source-to-adapter closure drifted"
            )
        _measurement_item_ids(
            official_bundle.predictor_pack,
            official_bundle.linkage_pack,
        )
        _validate_label_pack(
            official_bundle.label_pack,
            expected_source_sha256=OFFICIAL_DATASET_SHA256,
            expected_item_ids={
                row["opaque_item_id"]
                for row in official_bundle.predictor_pack["rows"]
            },
        )
        pack_closure_ready = True

    materialization_ready = False
    materialization_receipt: Mapping[str, Any] | None = None
    if materialization is not None:
        if pack_commitments is None:
            raise ArnIntrinsicProtocolError(
                "materialization cannot precede official pack closure"
            )
        materialization_receipt = _validated_materialization(
            materialization
        )
        if materialization_receipt["pack_commitments"] != pack_commitments:
            raise ArnIntrinsicProtocolError(
                "materialization and official pack commitments drifted"
            )
        materialization_ready = True
    runtime_access_ready = False
    if runtime_access is not None:
        if not materialization_ready:
            raise ArnIntrinsicProtocolError(
                "runtime access cannot precede materialization"
            )
        _validate_runtime_access(
            runtime_access,
            materialization=materialization,
        )
        runtime_access_ready = True

    freeze_ready = bool(
        source_verified
        and adapter_ready
        and all_arms_ready
        and pack_closure_ready
        and materialization_ready
        and runtime_access_ready
    )
    blocker_ids: list[str] = []
    if not source_verified:
        blocker_ids.append("OFFICIAL_SOURCE_NOT_VERIFIED_IN_RECEIPT")
    if not adapter_ready:
        blocker_ids.append("RAW_NARRATIVE_ADAPTER_NOT_READY")
    if not all_arms_ready:
        blocker_ids.append("FOUR_ARM_IMPLEMENTATIONS_NOT_READY")
    if not pack_closure_ready:
        blocker_ids.append("OFFICIAL_ADAPTER_TO_PACK_CLOSURE_NOT_READY")
    if not materialization_ready:
        blocker_ids.append("CAPABILITY_MATERIALIZATION_NOT_READY")
    if not runtime_access_ready:
        blocker_ids.append("RUNTIME_ACCESS_QUALIFICATION_NOT_READY")

    normalized_implementation_closure: dict[str, Any] = {}
    for component_id in required_components:
        closure = validated_components.get(component_id)
        normalized_implementation_closure[component_id] = {
            "status": "READY" if closure is not None else "NOT_READY",
            "qualification_receipt_self_hash": (
                closure.receipt["self_hash"]
                if closure is not None
                else None
            ),
            "qualification_receipt_file_sha256": (
                closure.receipt_file_sha256
                if closure is not None
                else None
            ),
            "implementation_file_sha256": (
                closure.implementation_file_sha256
                if closure is not None
                else None
            ),
            "qualification_test_file_sha256": (
                closure.qualification_test_file_sha256
                if closure is not None
                else None
            ),
            "implementation_closure_sha256": (
                closure.receipt["implementation_closure_sha256"]
                if closure is not None
                else None
            ),
        }

    sections: dict[str, Any] = {
        "source_contract": {
            "doi": OFFICIAL_DOI,
            "concept_doi": OFFICIAL_CONCEPT_DOI,
            "zenodo_revision": OFFICIAL_ZENODO_REVISION,
            "license_id": OFFICIAL_LICENSE_ID,
            "dataset_size": OFFICIAL_DATASET_SIZE,
            "dataset_sha256": OFFICIAL_DATASET_SHA256,
            "metadata_sha256": OFFICIAL_METADATA_SHA256,
            "header_sha256": hashlib.sha256(
                OFFICIAL_HEADER_BYTES
            ).hexdigest(),
            "source_qualification_report_sha256": (
                SOURCE_QUALIFICATION_REPORT_SHA256
            ),
            "row_count": OFFICIAL_ROW_COUNT,
            "id_minimum": OFFICIAL_ID_MINIMUM,
            "id_maximum": OFFICIAL_ID_MAXIMUM,
            "missing_ids": list(OFFICIAL_MISSING_IDS),
            "four_cell_counts": dict(OFFICIAL_CELL_COUNTS),
            "source_verified": source_verified,
            "dataset_rows_decoded_by_verifier": 0,
        },
        "split_contract": {
            "normalization": "NFKC",
            "unicodedata_version": FROZEN_UNIDATA_VERSION,
            "salt_sha256": hashlib.sha256(SPLIT_SALT).hexdigest(),
            "digest": "SHA256_full_digest",
            "bucket_formula": "int(full_digest_hex,16)%5",
            "calibration_bucket": CALIBRATION_BUCKET,
            "measurement_buckets": list(MEASUREMENT_BUCKETS),
            "whole_proverb_groups": True,
            "public_split_digest_emitted": False,
            "linkage_identity": "pre_source_private_HMAC_SHA256",
            "item_identity": "pre_source_private_HMAC_SHA256",
            "rebalancing": False,
            "fallback_replacement": False,
        },
        "column_access_contract": {
            lane: list(columns)
            for lane, columns in COLUMN_ACCESS_MATRIX.items()
        }
        | {
            "proverb_is_model_input": False,
            "public_salted_hash_called_opaque": False,
            "arms_receive_predictor_pack_only": True,
            "custodian_retains_linkage_and_labels": True,
        },
        "pack_contract": {
            "separate_predictor_linkage_label_packs": True,
            "official_predictor_schema": OFFICIAL_PREDICTOR_PACK_SCHEMA,
            "official_linkage_schema": OFFICIAL_LINKAGE_PACK_SCHEMA,
            "official_label_schema": OFFICIAL_LABEL_PACK_SCHEMA,
            "synthetic_lineage_cannot_formalize": True,
            "pack_commitments_required": [
                "predictor_pack_sha256",
                "linkage_pack_sha256",
                "label_pack_sha256",
            ],
            "official_pack_closure_ready": pack_closure_ready,
            "pack_commitments": (
                dict(pack_commitments)
                if pack_commitments is not None
                else None
            ),
            "adapter_output_receipt_self_hash": (
                adapter_output_receipt["self_hash"]
                if adapter_output_receipt is not None
                else None
            ),
        },
        "prediction_contract": {
            "arms": list(ARM_IDS),
            "common_input_required": True,
            "prediction_row_exact_fields": [
                "opaque_item_id",
                "disposition",
                "selected_choice",
                "error_code",
            ],
            "dispositions": list(DISPOSITIONS),
            "choice_ids": list(CHOICE_IDS),
            "explanation_field_allowed": False,
        },
        "lifecycle_contract": {
            "all_arm_action_seal_before_labels": True,
            "labels_open_once": True,
            "retry": False,
            "resample": False,
            "online_or_api_evaluation": False,
            "formal_action_seal_requires_ready_freeze_manifest": True,
            "synthetic_terminal_is_qualification_only": True,
            "failure_after_label_open_is_terminal": True,
        },
        "metric_contract": {
            "aggregates": [
                "overall",
                "far_high",
                "far_low",
                "near_high",
                "near_low",
            ],
            "abstain_counted_wrong": True,
            "error_counted_wrong": True,
            "uncertainty_unit": "opaque_whole_proverb_cluster",
            "uncertainty_method": (
                "intercept_only_cluster_robust_sandwich_G_over_G_minus_1_"
                "clipped_normal_95_interval"
            ),
            "paired_differences": [
                "full_gscl_minus_semantic_only",
                "full_gscl_minus_legacy_keyword",
                "full_gscl_minus_flat_label_no_verifier",
            ],
            "paired_difference_scope": "overall_and_four_cells",
            "effect_gate": False,
        },
        "implementation_exposure": dict(IMPLEMENTATION_EXPOSURE),
        "implementation_closure": normalized_implementation_closure,
        "capability_closure": {
            "materialization_ready": materialization_ready,
            "materialization_receipt_self_hash": (
                materialization_receipt["self_hash"]
                if materialization_receipt is not None
                else None
            ),
            "runtime_access_ready": runtime_access_ready,
            "runtime_access_receipt_self_hash": (
                runtime_access.receipt["self_hash"]
                if runtime_access is not None
                else None
            ),
            "runtime_access_receipt_file_sha256": (
                runtime_access.receipt_file_sha256
                if runtime_access is not None
                else None
            ),
        },
    }
    receipt: dict[str, Any] = {
        "schema": PROTOCOL_RECEIPT_SCHEMA,
        "status": (
            "MECHANICAL_PROTOCOL_CLOSED_IMPLEMENTATION_NOT_READY"
            if not freeze_ready
            else "IMPLEMENTATION_CLOSURE_RECORDED_NOT_MEASURED"
        ),
        "new_study": False,
        "formal_measurement": False,
        "efficacy_evidence": False,
        "effect_gate_added": False,
        "freeze_ready": freeze_ready,
        "blocker_ids": blocker_ids,
        **sections,
    }
    receipt["nested_hashes"] = {
        name: _content_hash(value) for name, value in sections.items()
    }
    receipt["protocol_contract_sha256"] = _content_hash(
        {
            "source_contract": sections["source_contract"],
            "split_contract": sections["split_contract"],
            "column_access_contract": sections["column_access_contract"],
            "pack_contract": sections["pack_contract"],
            "prediction_contract": sections["prediction_contract"],
            "lifecycle_contract": sections["lifecycle_contract"],
            "metric_contract": sections["metric_contract"],
            "implementation_exposure": sections[
                "implementation_exposure"
            ],
            "implementation_closure": sections[
                "implementation_closure"
            ],
            "capability_closure": sections["capability_closure"],
            "pack_closure": sections["pack_contract"],
        }
    )
    receipt["self_hash"] = _content_hash(receipt)
    return receipt


def build_ready_freeze_manifest(
    *,
    protocol_receipt: Mapping[str, Any],
    official_bundle: PrivatePackBundle,
) -> dict[str, Any]:
    """Freeze the validated protocol closure without opening measurement."""

    if (
        protocol_receipt.get("schema") != PROTOCOL_RECEIPT_SCHEMA
        or protocol_receipt.get("freeze_ready") is not True
        or protocol_receipt.get("blocker_ids") != []
    ):
        raise ArnIntrinsicProtocolError(
            "protocol receipt is not ready for a formal freeze"
        )
    protocol_body = dict(protocol_receipt)
    claimed = protocol_body.pop("self_hash", None)
    if not _is_sha256(claimed) or _content_hash(protocol_body) != claimed:
        raise ArnIntrinsicProtocolError(
            "ready protocol receipt self hash drifted"
        )
    if (
        official_bundle.lineage != "official_arn_measurement"
        or official_bundle.adapter_output_receipt is None
    ):
        raise ArnIntrinsicProtocolError(
            "ready freeze requires an official adapter bundle"
        )
    _validate_adapter_output_receipt(
        official_bundle.adapter_output_receipt,
        expected_pack_commitments=official_bundle.pack_commitments,
    )
    if (
        protocol_receipt["pack_contract"]["pack_commitments"]
        != official_bundle.pack_commitments
        or protocol_receipt["pack_contract"][
            "adapter_output_receipt_self_hash"
        ]
        != official_bundle.adapter_output_receipt["self_hash"]
    ):
        raise ArnIntrinsicProtocolError(
            "ready protocol and official pack closure drifted"
        )
    manifest: dict[str, Any] = {
        "schema": READY_FREEZE_MANIFEST_SCHEMA,
        "status": "READY_CLOSURE_FROZEN_MEASUREMENT_UNOPENED",
        "freeze_ready": True,
        "measurement_opened": False,
        "protocol_receipt_self_hash": protocol_receipt["self_hash"],
        "protocol_contract_sha256": protocol_receipt[
            "protocol_contract_sha256"
        ],
        "source_verification_self_hash": (
            official_bundle.adapter_output_receipt[
                "source_verification_self_hash"
            ]
        ),
        "source_qualification_report_sha256": (
            SOURCE_QUALIFICATION_REPORT_SHA256
        ),
        "implementation_closure_hash": _content_hash(
            protocol_receipt["implementation_closure"]
        ),
        "arm_closure_bindings": {
            arm_id: {
                "implementation_file_sha256": protocol_receipt[
                    "implementation_closure"
                ][arm_id]["implementation_file_sha256"],
                "qualification_receipt_self_hash": protocol_receipt[
                    "implementation_closure"
                ][arm_id]["qualification_receipt_self_hash"],
            }
            for arm_id in ARM_IDS
        },
        "capability_closure_hash": _content_hash(
            protocol_receipt["capability_closure"]
        ),
        "adapter_output_receipt_self_hash": (
            official_bundle.adapter_output_receipt["self_hash"]
        ),
        "pack_commitments": dict(official_bundle.pack_commitments),
        "formal_action_seal_required": True,
        "effect_gate": False,
    }
    manifest["self_hash"] = _content_hash(manifest)
    return manifest


def _validate_ready_freeze_manifest(
    manifest: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        manifest,
        {
            "schema",
            "status",
            "freeze_ready",
            "measurement_opened",
            "protocol_receipt_self_hash",
            "protocol_contract_sha256",
            "source_verification_self_hash",
            "source_qualification_report_sha256",
            "implementation_closure_hash",
            "arm_closure_bindings",
            "capability_closure_hash",
            "adapter_output_receipt_self_hash",
            "pack_commitments",
            "formal_action_seal_required",
            "effect_gate",
            "self_hash",
        },
        label="ready freeze manifest",
    )
    body = dict(manifest)
    claimed = body.pop("self_hash")
    pack_commitments = manifest["pack_commitments"]
    arm_closure_bindings = manifest["arm_closure_bindings"]
    if (
        manifest["schema"] != READY_FREEZE_MANIFEST_SCHEMA
        or manifest["status"]
        != "READY_CLOSURE_FROZEN_MEASUREMENT_UNOPENED"
        or manifest["freeze_ready"] is not True
        or manifest["measurement_opened"] is not False
        or manifest["source_qualification_report_sha256"]
        != SOURCE_QUALIFICATION_REPORT_SHA256
        or manifest["formal_action_seal_required"] is not True
        or manifest["effect_gate"] is not False
        or not all(
            _is_sha256(manifest[field])
            for field in (
                "protocol_receipt_self_hash",
                "protocol_contract_sha256",
                "source_verification_self_hash",
                "implementation_closure_hash",
                "capability_closure_hash",
                "adapter_output_receipt_self_hash",
            )
        )
        or not isinstance(pack_commitments, Mapping)
        or set(pack_commitments)
        != {
            "predictor_pack_sha256",
            "linkage_pack_sha256",
            "label_pack_sha256",
        }
        or not all(_is_sha256(value) for value in pack_commitments.values())
        or not isinstance(arm_closure_bindings, Mapping)
        or set(arm_closure_bindings) != set(ARM_IDS)
        or any(
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "implementation_file_sha256",
                "qualification_receipt_self_hash",
            }
            or not _is_sha256(binding["implementation_file_sha256"])
            or not _is_sha256(
                binding["qualification_receipt_self_hash"]
            )
            for binding in arm_closure_bindings.values()
        )
        or not _is_sha256(claimed)
        or _content_hash(body) != claimed
    ):
        raise ArnIntrinsicProtocolError("ready freeze manifest drifted")


def persist_ready_freeze_manifest_once(
    path: Path, manifest: Mapping[str, Any]
) -> str:
    _validate_ready_freeze_manifest(manifest)
    return _write_exclusive(path, manifest)


def validate_ready_freeze_manifest_file(
    *,
    path: Path,
    expected_file_sha256: str,
) -> ValidatedReadyFreeze:
    manifest, file_hash = _load_canonical_mapping_single_fd(
        path,
        label="ready freeze manifest",
        expected_sha256=expected_file_sha256,
    )
    _validate_ready_freeze_manifest(manifest)
    return ValidatedReadyFreeze(
        manifest=manifest,
        manifest_path=path,
        manifest_file_sha256=file_hash,
        _validation_token=_READY_FREEZE_VALIDATION_TOKEN,
    )


def _validated_ready_freeze(
    ready_freeze: ValidatedReadyFreeze,
) -> Mapping[str, Any]:
    if (
        not isinstance(ready_freeze, ValidatedReadyFreeze)
        or ready_freeze._validation_token
        is not _READY_FREEZE_VALIDATION_TOKEN
    ):
        raise ArnIntrinsicProtocolError(
            "ready freeze was not validated from an exact canonical file"
        )
    manifest, file_hash = _load_canonical_mapping_single_fd(
        ready_freeze.manifest_path,
        label="ready freeze manifest",
        expected_sha256=ready_freeze.manifest_file_sha256,
    )
    if manifest != ready_freeze.manifest or (
        file_hash != ready_freeze.manifest_file_sha256
    ):
        raise ArnIntrinsicProtocolError(
            "validated ready freeze changed on disk"
        )
    _validate_ready_freeze_manifest(manifest)
    return manifest


__all__ = [
    "ANALOGY_LEVELS",
    "ARM_IDS",
    "ArnImplementationNotReady",
    "ArnIntrinsicProtocolError",
    "AdaptedArnRow",
    "COLUMN_ACCESS_MATRIX",
    "DISTRACTOR_SIMILARITIES",
    "FROZEN_UNIDATA_VERSION",
    "FORMAL_ACTION_SEAL_SCHEMA",
    "IMPLEMENTATION_EXPOSURE_SOURCE_ID",
    "IMPLEMENTATION_QUALIFICATION_SCHEMA",
    "OFFICIAL_DATASET_SHA256",
    "OFFICIAL_CONCEPT_DOI",
    "OFFICIAL_DOI",
    "OFFICIAL_LICENSE_ID",
    "QUALIFICATION_ACTION_SEAL_SCHEMA",
    "PrivatePackBundle",
    "RawNarrativeAdapter",
    "SPLIT_SALT",
    "SourceBinding",
    "SplitAssignment",
    "ValidatedImplementationQualification",
    "ValidatedMaterialization",
    "ValidatedReadyFreeze",
    "ValidatedRuntimeAccess",
    "audit_formal_capability_materialization",
    "build_all_arm_action_seal",
    "build_arm_algorithm",
    "build_official_private_packs",
    "build_qualification_action_seal",
    "build_raw_narrative_adapter",
    "build_ready_freeze_manifest",
    "build_safe_protocol_receipt",
    "make_prediction_pack",
    "materialize_qualification_capabilities_once",
    "opaque_item_id",
    "open_labels_and_score_qualification_once",
    "open_labels_and_score_once",
    "persist_action_seal_once",
    "persist_ready_freeze_manifest_once",
    "run_official_adapter_once",
    "split_proverb",
    "validate_implementation_qualification_file",
    "validate_runtime_access_receipt_file",
    "validate_ready_freeze_manifest_file",
    "verify_official_source",
]
