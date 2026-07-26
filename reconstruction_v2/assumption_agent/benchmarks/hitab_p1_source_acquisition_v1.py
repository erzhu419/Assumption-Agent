"""Source-free HiTab P1 custody, parser, and private selection primitives.

This module never downloads a dataset and has no model, API, evaluator, or
network entry point.  It binds the frozen HiTab custody/design, verifies the
four locally acquired files against exact byte, SHA-256, and Git-blob
identities, and provides the only parser used by the study.

The parser joins the pinned official HMT representation to the exact raw matrix
header offsets.  For a requested table, exactly one matching HMT JSON and one
matching raw JSON are opened: the central directory is inspected for traversal,
links, duplicate names, special files, and size bombs, but unrelated safe
members are not subjected to a suffix or directory grammar.

Question rows are strict JSONL objects with six required semantic fields;
extra fields are ignored and row-local schema/eligibility failures are counted
with fixed reason tokens.  TEST is identity/newline-only by default.  Its first
JSON decode requires a separately sealed promotion authorization and consumes
an O_EXCL marker before parsing.

All persisted receipts are mode 0600 and contain only fixed aggregate counts,
source identities, and cryptographic commitments.  Item IDs, questions, table
IDs, table/header/cell text, coordinates, and qrels never enter a public
receipt.  The in-memory private dataclasses are intended to be handed directly
to the later trusted formal runtime.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, BinaryIO
import unicodedata
import zipfile
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from assumption_agent.benchmarks import hitab_p1_dmc1_core_v1 as dmc_core


VERSION = "hitab_p1_source_acquisition_v1"
STUDY_ID = "HITAB_P1_DMC1_HIERARCHICAL_SET_EVALUATOR_V1"
SOURCE_COMMIT = "d179602662b490249baf068a76fbe4137029126e"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUSTODY_RELATIVE = Path("manifests/hitab_p1_public_source_custody_v1.json")
DESIGN_RELATIVE = Path(
    "manifests/hitab_p1_dmc1_hierarchical_set_evaluator_design_v1.json"
)

REQUIRED_SAMPLE_FIELDS = frozenset(
    {
        "id",
        "table_id",
        "table_source",
        "question",
        "aggregation",
        "linked_cells",
    }
)
ACCEPTED_TABLE_SOURCES = frozenset({"nsf", "statcan"})

FAMILY_BY_AGGREGATION_TOKEN = {
    "average": "AGGREGATE",
    "counta": "AGGREGATE",
    "sum": "AGGREGATE",
    "diff": "COMPARATIVE",
    "div": "COMPARATIVE",
    "greater_than": "COMPARATIVE",
    "less_than": "COMPARATIVE",
    "argmax": "SUPERLATIVE",
    "argmin": "SUPERLATIVE",
    "kth-argmax": "SUPERLATIVE",
    "kth-argmin": "SUPERLATIVE",
    "max": "SUPERLATIVE",
    "min": "SUPERLATIVE",
    "pair-argmax": "SUPERLATIVE",
    "pair-argmin": "SUPERLATIVE",
    "range": "SUPERLATIVE",
    "topk-argmax": "SUPERLATIVE",
    "topk-argmin": "SUPERLATIVE",
}
FAMILIES = ("AGGREGATE", "COMPARATIVE", "SUPERLATIVE")

BLOCK_SOURCE_SPLIT = {
    "A_form": "TRAIN",
    "A_hold": "DEV",
    "M_search": "TEST",
}
BLOCK_QUOTA_PER_FAMILY = {
    "A_form": 36,
    "A_hold": 12,
    "M_search": 12,
}
INITIAL_BLOCKS = ("A_form", "A_hold")

PUBLIC_EXPOSURE_HASHES = {
    "id": frozenset(
        {"3293a07406d17c1d85dbd35f4646dc3b55fefc98433f059133666d4efecb2598"}
    ),
    "question": frozenset(
        {"eb55e636bef4d1dc7c0ebe520414f30cfe029b8faf6ca13886049547025583f8"}
    ),
    "table_id": frozenset(
        {"a73060afb61efe1b7c817645d00c342df02407f65435a64c88d251d56150ff42"}
    ),
}

SOURCE_ATTEMPT_FILENAME = "source_identity.attempt.private.json"
SOURCE_RECEIPT_FILENAME = "source_identity.receipt.safe.json"
SOURCE_FAILURE_FILENAME = "source_identity.failure.safe.json"
DOWNLOAD_ATTEMPT_FILENAME = "source_download.attempt.private.json"
DOWNLOAD_RECEIPT_FILENAME = "source_download.receipt.safe.json"
DOWNLOAD_FAILURE_FILENAME = "source_download.failure.safe.json"
INITIAL_ATTEMPT_FILENAME = "initial_selection.attempt.private.json"
INITIAL_SECRET_FILENAME = "selection_secret.private.bin"
INITIAL_RECEIPT_FILENAME = "initial_selection.receipt.safe.json"
INITIAL_FAILURE_FILENAME = "initial_selection.failure.safe.json"
COMPONENT_REGISTRY_FILENAME = "component_registry.private.json"
M_COMPONENT_REGISTRY_FILENAME = "component_registry.with_M.private.json"
BLOCK_VIEW_FILENAMES = {
    "A_form": "A_form.label_free_view.private.json",
    "A_hold": "A_hold.label_free_view.private.json",
    "M_search": "M_search.label_free_view.private.json",
}
QREL_CUSTODY_FILENAMES = {
    "A_form": "A_form.qrels.sealed.private.json",
    "A_hold": "A_hold.qrels.sealed.private.json",
    "M_search": "M_search.qrels.sealed.private.json",
}
QREL_RELEASE_MARKER_FILENAMES = {
    block: f"{block}.qrels.release.attempt.private.json"
    for block in QREL_CUSTODY_FILENAMES
}
TEST_DECODE_ATTEMPT_FILENAME = "test_first_decode.attempt.private.json"
TEST_SELECTION_RECEIPT_FILENAME = "test_selection.receipt.safe.json"
TEST_SELECTION_FAILURE_FILENAME = "test_selection.failure.safe.json"
FORMAL_CLAIM_FILENAME = "formal_controller.claim.private.json"

SELECTION_ORDER_DOMAIN = b"HITAB_P1_SELECTION_ORDER_HMAC_SHA256_V1\x00"
WORK_ID_DOMAIN = b"HITAB_P1_OPAQUE_WORK_ID_HMAC_SHA256_V1\x00"
TABLE_COMMITMENT_DOMAIN = b"HITAB_P1_TABLE_COMMITMENT_HMAC_SHA256_V1\x00"

MIN_CORPUS_UNITS = 10
MAX_CORPUS_UNITS = 256
MAX_PROOF_REQUIREMENTS = 5
MAX_IDENTIFIER_CHARACTERS = 2_048
MAX_QUESTION_CHARACTERS = 32_000
MAX_JSONL_LINE_BYTES = 16_000_000
MAX_JSONL_ROWS = 250_000
MAX_ZIP_MEMBERS = 20_000
MAX_ZIP_MEMBER_BYTES = 64_000_000
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 1_000_000_000
MAX_ZIP_COMPRESSION_RATIO = 2_000
MAX_TABLE_JSON_BYTES = 16_000_000
MAX_TREE_DEPTH = 32
MAX_TREE_NODES = 8_192
MAX_DATA_ROWS = 4_096
MAX_DATA_COLUMNS = 4_096
HTTP_TIMEOUT_SECONDS = 900
HTTP_READ_CHUNK_BYTES = 1 << 20
EXPECTED_DOWNLOAD_HOST = "raw.githubusercontent.com"

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_TABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(
    r"(?P<sign>[+-]?)"
    r"(?P<number>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
    r"(?P<percent>%?)\Z"
)
_MATRIX_COORDINATE = re.compile(
    r"\(\s*(0|[1-9][0-9]*)\s*,\s*(0|[1-9][0-9]*)\s*\)\Z"
)


class HitabP1SourceError(RuntimeError):
    """The frozen source/parser/selection contract failed closed."""


class HitabP1RowIneligible(ValueError):
    """One row is excluded under a fixed, safe aggregate reason token."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise HitabP1SourceError("row exclusion reason is invalid")
        super().__init__(reason)
        self.reason = reason


class RequestedTableNotFound(HitabP1RowIneligible):
    """An otherwise valid row references no unique HMT member."""

    def __init__(self) -> None:
        super().__init__("requested_table_not_found")


@dataclass(frozen=True)
class SourceFileContract:
    key: str
    relative_path: str
    size_bytes: int
    git_blob_sha1: str
    is_jsonl: bool
    raw_url: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not self.key
            or not isinstance(self.relative_path, str)
            or not self.relative_path
            or PurePosixPath(self.relative_path).is_absolute()
            or ".." in PurePosixPath(self.relative_path).parts
            or type(self.size_bytes) is not int
            or self.size_bytes < 1
            or _HEX40.fullmatch(self.git_blob_sha1) is None
            or type(self.is_jsonl) is not bool
            or (self.raw_url is not None and not isinstance(self.raw_url, str))
        ):
            raise HitabP1SourceError("source file contract is invalid")


FORMAL_SOURCE_CONTRACTS = {
    "TRAIN": SourceFileContract(
        "TRAIN",
        "data/train_samples.jsonl",
        5_642_769,
        "11c98debd75d82183818f82e31170cbb207aa5bc",
        True,
        "https://raw.githubusercontent.com/microsoft/HiTab/"
        + SOURCE_COMMIT
        + "/data/train_samples.jsonl",
    ),
    "DEV": SourceFileContract(
        "DEV",
        "data/dev_samples.jsonl",
        1_259_029,
        "03f1b2fb2001155d0276c95ee9f0c765d7f43513",
        True,
        "https://raw.githubusercontent.com/microsoft/HiTab/"
        + SOURCE_COMMIT
        + "/data/dev_samples.jsonl",
    ),
    "TEST": SourceFileContract(
        "TEST",
        "data/test_samples_qualitycheck.jsonl",
        1_203_596,
        "f5a9e655fd99d05adeef0ea1fba69161b5971e3f",
        True,
        "https://raw.githubusercontent.com/microsoft/HiTab/"
        + SOURCE_COMMIT
        + "/data/test_samples_qualitycheck.jsonl",
    ),
    "TABLES": SourceFileContract(
        "TABLES",
        "data/tables.zip",
        8_752_343,
        "a884ae60aa96fb6b76b3198fa135f3a384548122",
        False,
        "https://raw.githubusercontent.com/microsoft/HiTab/"
        + SOURCE_COMMIT
        + "/data/tables.zip",
    ),
}


@dataclass(frozen=True)
class VerifiedFileIdentity:
    key: str
    size_bytes: int
    sha256: str
    git_blob_sha1: str
    raw_newline_count: int | None

    def safe_payload(self) -> dict[str, object]:
        return {
            "git_blob_sha1": self.git_blob_sha1,
            "raw_newline_count": self.raw_newline_count,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class VerifiedSourceSet:
    identities: Mapping[str, VerifiedFileIdentity]
    safe_receipt: Mapping[str, Any]

    @property
    def source_identity_commitment(self) -> str:
        value = self.safe_receipt.get("source_identity_commitment")
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise HitabP1SourceError("source identity commitment is invalid")
        return value


@dataclass(frozen=True)
class DownloadedSourceSet:
    source_paths: Mapping[str, Path]
    verified_sources: VerifiedSourceSet


@dataclass(frozen=True)
class SampleCandidate:
    split: str
    item_id: str
    table_id: str
    table_source: str
    question: str
    family: str
    linked_cells: Mapping[str, Any]

    @property
    def row_commitment(self) -> str:
        # Gold-bearing linked fields are deliberately absent.
        return stable_hash(
            {
                "family": self.family,
                "item_id": normalized_text(self.item_id),
                "question": normalized_text(self.question),
                "split": self.split,
                "table_id": normalized_text(self.table_id),
            }
        )


@dataclass(frozen=True)
class SampleParseResult:
    candidates: tuple[SampleCandidate, ...]
    safe_summary: Mapping[str, Any]


@dataclass(frozen=True)
class RequestedTablePayload:
    hmt: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class HeaderNode:
    node_key: tuple[int, ...]
    name: str
    line_idx: int | None
    children: tuple["HeaderNode", ...]


@dataclass(frozen=True)
class AtomicUnit:
    ordinal: int
    row_index: int
    column_index: int
    value_surface: str
    value_type: str
    typed_literal: str
    left_header_path: tuple[str, ...]
    top_header_path: tuple[str, ...]
    left_node_path: tuple[tuple[int, ...], ...]
    top_node_path: tuple[tuple[int, ...], ...]
    serialized: str


@dataclass(frozen=True, order=True)
class TableTypedEdge:
    source_ordinal: int
    target_ordinal: int
    edge_type: str = "FORWARD_SHARED_AXIS_OR_HEADER"

    def __post_init__(self) -> None:
        if (
            type(self.source_ordinal) is not int
            or type(self.target_ordinal) is not int
            or self.source_ordinal < 0
            or self.target_ordinal <= self.source_ordinal
            or self.edge_type != "FORWARD_SHARED_AXIS_OR_HEADER"
        ):
            raise HitabP1SourceError("table typed edge is outside grammar")


@dataclass(frozen=True)
class TableView:
    title: str
    units: tuple[AtomicUnit, ...]
    top_header_rows_num: int
    left_header_columns_num: int

    def __post_init__(self) -> None:
        if not MIN_CORPUS_UNITS <= len(self.units) <= MAX_CORPUS_UNITS:
            raise HitabP1SourceError("table corpus size is outside contract")
        if tuple(unit.ordinal for unit in self.units) != tuple(range(len(self.units))):
            raise HitabP1SourceError("table unit ordering is not canonical")
        coordinates = {(unit.row_index, unit.column_index) for unit in self.units}
        if len(coordinates) != len(self.units):
            raise HitabP1SourceError("table unit coordinates are duplicated")
        if (
            type(self.top_header_rows_num) is not int
            or type(self.left_header_columns_num) is not int
            or not 1 <= self.top_header_rows_num <= MAX_DATA_ROWS
            or not 1 <= self.left_header_columns_num <= MAX_DATA_COLUMNS
        ):
            raise HitabP1SourceError("raw table header offset is outside contract")

    @property
    def corpus_commitment(self) -> str:
        return stable_hash([unit.serialized for unit in self.units])

    @property
    def matrix_coordinate_to_ordinal(self) -> Mapping[tuple[int, int], int]:
        return {
            (
                unit.row_index + self.top_header_rows_num,
                unit.column_index + self.left_header_columns_num,
            ): unit.ordinal
            for unit in self.units
        }

    @property
    def typed_edges(self) -> tuple[TableTypedEdge, ...]:
        edges: list[TableTypedEdge] = []
        for left_ordinal in range(len(self.units)):
            left = self.units[left_ordinal]
            for right_ordinal in range(left_ordinal + 1, len(self.units)):
                right = self.units[right_ordinal]
                if (
                    left.row_index == right.row_index
                    or left.column_index == right.column_index
                    or _has_shared_prefix(
                        left.left_node_path, right.left_node_path
                    )
                    or _has_shared_prefix(
                        left.top_node_path, right.top_node_path
                    )
                ):
                    edges.append(
                        TableTypedEdge(left_ordinal, right_ordinal)
                    )
        result = tuple(edges)
        if result != tuple(sorted(set(result))):
            raise HitabP1SourceError(
                "table typed edge grammar produced duplicates"
            )
        return result

ProofDNF = dmc_core.ProofDNF


@dataclass(frozen=True)
class EligibleItem:
    candidate: SampleCandidate
    table: TableView
    qrel: ProofDNF

    @property
    def component_tokens(self) -> frozenset[str]:
        return frozenset(
            {
                "table:" + normalized_text(self.candidate.table_id),
                "question:" + stable_hash(normalized_text(self.candidate.question)),
                "corpus:" + self.table.corpus_commitment,
            }
        )

    @property
    def qrel_ordinal_mapping_commitment(self) -> str:
        return self.qrel.ordinal_mapping_commitment


@dataclass(frozen=True)
class SelectedItem:
    block: str
    work_id: str
    item: EligibleItem


@dataclass(frozen=True)
class SelectionBatch:
    selected_by_block: Mapping[str, tuple[SelectedItem, ...]]
    safe_receipt: Mapping[str, Any]
    used_component_tokens: frozenset[str]
    component_registry: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class InitialSelectionRun:
    block_views: Mapping[str, BridgeBlockView]
    safe_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class MSelectionRun:
    block_view: BridgeBlockView
    safe_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class BridgeViewItem:
    """Source-free, label-free controller input; no family or qrel channel."""

    work_id: str
    question: str
    ordered_unit_strings: tuple[str, ...]
    unit_types: tuple[str, ...]
    typed_edges: tuple[TableTypedEdge, ...]
    corpus_commitment: str

    def private_payload(self) -> dict[str, object]:
        return {
            "corpus_commitment": self.corpus_commitment,
            "ordered_unit_strings": list(self.ordered_unit_strings),
            "question": self.question,
            "typed_edges": [
                {
                    "edge_type": edge.edge_type,
                    "source_ordinal": edge.source_ordinal,
                    "target_ordinal": edge.target_ordinal,
                }
                for edge in self.typed_edges
            ],
            "unit_types": list(self.unit_types),
            "work_id": self.work_id,
        }


@dataclass(frozen=True)
class BridgeBlockView:
    block: str
    items: tuple[BridgeViewItem, ...]
    view_sha256: str


@dataclass(frozen=True)
class BridgeQrelRow:
    work_id: str
    family: str
    qrel: ProofDNF
    corpus_commitment: str
    qrel_ordinal_mapping_commitment: str


@dataclass(frozen=True)
class BridgeQrelPack:
    block: str
    action_archive_sha256: str
    rows: tuple[BridgeQrelRow, ...]
    pack_sha256: str


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HitabP1SourceError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise HitabP1SourceError("self hash already exists")
    return {**dict(body), "self_sha256": stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or not hmac.compare_digest(stable_hash(body), claimed)
    ):
        raise HitabP1SourceError("self-hashed value drifted")
    return claimed


def normalized_text(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise HitabP1SourceError("text is invalid")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip().casefold()


def normalized_text_sha256(value: str) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HitabP1SourceError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise HitabP1SourceError("JSON contains a non-finite number")


def strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite,
        )
    except HitabP1SourceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HitabP1SourceError(f"{label} is not strict JSON") from exc


def git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()  # nosec B324: immutable Git object identity
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise HitabP1SourceError("durable path is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise HitabP1SourceError("private directory parent is unavailable")
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HitabP1SourceError("private directory path is unsafe")
        break
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        os.chmod(directory, 0o700)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    """Persist one canonical JSON object with O_EXCL and exact mode 0600."""

    raw = canonical_bytes(value, newline=True)
    _ensure_private_directory(path.parent)
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
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
                raise HitabP1SourceError("exclusive JSON mode drifted")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def write_bytes_exclusive(path: Path, raw: bytes) -> str:
    """Persist private opaque bytes with O_EXCL and exact mode 0600."""

    if not isinstance(raw, bytes) or not raw:
        raise HitabP1SourceError("exclusive private bytes are invalid")
    _ensure_private_directory(path.parent)
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
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
                raise HitabP1SourceError("exclusive private byte mode drifted")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def _load_manifest(
    path: Path,
    *,
    expected_schema: str,
    expected_self_sha256: str | None = None,
) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HitabP1SourceError("bound manifest is unavailable") from exc
    value = strict_json(raw, label="bound manifest")
    if not isinstance(value, Mapping):
        raise HitabP1SourceError("bound manifest is not an object")
    observed = verify_self_hash(value)
    if (
        (
            expected_self_sha256 is not None
            and (
                _HEX64.fullmatch(expected_self_sha256) is None
                or not hmac.compare_digest(observed, expected_self_sha256)
            )
        )
        or value.get("schema") != expected_schema
        or value.get("study_id") != STUDY_ID
    ):
        raise HitabP1SourceError("bound manifest drifted")
    return value


def verify_frozen_bindings(
    custody_path: Path = PROJECT_ROOT / CUSTODY_RELATIVE,
    design_path: Path = PROJECT_ROOT / DESIGN_RELATIVE,
) -> None:
    custody = _load_manifest(
        Path(custody_path),
        expected_schema="hitab_p1_public_source_custody_v1",
    )
    design = _load_manifest(
        Path(design_path),
        expected_schema="hitab_p1_dmc1_hierarchical_set_evaluator_design_v1",
    )
    custody_self_sha256 = verify_self_hash(custody)
    files = custody.get("allowed_source_files")
    if not isinstance(files, Mapping):
        raise HitabP1SourceError("custody file registry drifted")
    custody_key = {"TRAIN": "train", "DEV": "dev", "TEST": "test", "TABLES": "tables"}
    for key, contract in FORMAL_SOURCE_CONTRACTS.items():
        row = files.get(custody_key[key])
        if (
            not isinstance(row, Mapping)
            or row.get("relative_path") != contract.relative_path
            or row.get("size_bytes") != contract.size_bytes
            or row.get("git_blob_sha1") != contract.git_blob_sha1
            or row.get("raw_url") != contract.raw_url
        ):
            raise HitabP1SourceError("custody source identity drifted")
    family = design.get("family_contract")
    blocks = design.get("block_contract")
    parser = design.get("source_parser_contract")
    qrel_contract = design.get("qrel_and_utility_contract")
    archive_contract = (
        parser.get("archive") if isinstance(parser, Mapping) else None
    )
    if (
        not isinstance(design.get("source_binding"), Mapping)
        or design["source_binding"].get("custody_self_sha256")
        != custody_self_sha256
        or not isinstance(family, Mapping)
        or not isinstance(blocks, Mapping)
        or not isinstance(parser, Mapping)
        or not isinstance(qrel_contract, Mapping)
        or not isinstance(archive_contract, Mapping)
        or family.get("accepted_native_AGGR_MAP_tokens")
        != {
            "AGGREGATE": ["average", "counta", "sum"],
            "COMPARATIVE": ["diff", "div", "greater_than", "less_than"],
            "SUPERLATIVE": [
                "argmax",
                "argmin",
                "kth-argmax",
                "kth-argmin",
                "max",
                "min",
                "pair-argmax",
                "pair-argmin",
                "range",
                "topk-argmax",
                "topk-argmin",
            ],
        }
        or parser.get("extra_JSON_fields") != "allowed_and_ignored"
        or parser.get("required_sample_semantics")
        != [
            "id",
            "table_id",
            "table_source",
            "question",
            "aggregation",
            "linked_cells",
        ]
        or archive_contract.get("blanket_member_grammar_or_suffix_gate")
        is not False
        or archive_contract.get("unrequested_member_payload_open") is not False
        or archive_contract.get("requested_table_members")
        != "for_an_eligible_table_id_open_exactly_one_safe_regular_JSON_member_beneath_an_hmt_directory_and_exactly_one_beneath_a_raw_directory_each_with_matching_basename"
        or qrel_contract.get("proof_requirement")
        != "each_official_ANSWER_coordinate_is_one_singleton_atomic_unit_bucket"
    ):
        raise HitabP1SourceError("study design parser/family binding drifted")
    for block in BLOCK_SOURCE_SPLIT:
        row = blocks.get(block)
        if (
            not isinstance(row, Mapping)
            or row.get("per_family") != BLOCK_QUOTA_PER_FAMILY[block]
        ):
            raise HitabP1SourceError("study block quota drifted")
    if (
        blocks["A_form"].get("source_split") != "TRAIN"
        or blocks["A_hold"].get("source_split") != "DEV"
        or blocks["M_search"].get("source_split")
        != "official_October_2025_quality_checked_TEST"
    ):
        raise HitabP1SourceError("study block source split drifted")


def _validate_download_url(contract: SourceFileContract) -> str:
    if contract.raw_url is None:
        raise HitabP1SourceError("source download URL is not frozen")
    try:
        parsed = urlsplit(contract.raw_url)
        port = parsed.port
    except ValueError as exc:
        raise HitabP1SourceError("source download URL is invalid") from exc
    expected_path = (
        f"/microsoft/HiTab/{SOURCE_COMMIT}/{contract.relative_path}"
    )
    if (
        parsed.scheme != "https"
        or parsed.hostname != EXPECTED_DOWNLOAD_HOST
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise HitabP1SourceError("source download URL contract drifted")
    return contract.raw_url


class _SameHostHTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        try:
            parsed = urlsplit(newurl)
            port = parsed.port
        except ValueError as exc:
            raise HTTPError(
                req.full_url, code, "redirect rejected", headers, fp
            ) from exc
        if (
            parsed.scheme != "https"
            or parsed.hostname != EXPECTED_DOWNLOAD_HOST
            or port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise HTTPError(
                req.full_url, code, "redirect rejected", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_http_opener(url: str) -> BinaryIO:
    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "hitab-p1-pinned-source-acquisition/1",
        },
        method="GET",
    )
    opener = build_opener(
        ProxyHandler({}),
        _SameHostHTTPSRedirectHandler(),
    )
    return opener.open(request, timeout=HTTP_TIMEOUT_SECONDS)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise HitabP1SourceError("exclusive source part write failed")
        written += count


def _validate_download_response(response: BinaryIO, *, expected_url: str) -> None:
    status_value = getattr(response, "status", 200)
    if status_value != 200:
        raise HitabP1SourceError("source HTTP status is not 200")
    geturl = getattr(response, "geturl", None)
    final_url = geturl() if callable(geturl) else expected_url
    if final_url != expected_url:
        raise HitabP1SourceError("source HTTP final URL drifted")
    headers = getattr(response, "headers", {})
    encoding = headers.get("Content-Encoding", "identity")
    if not isinstance(encoding, str) or encoding.casefold() not in {
        "",
        "identity",
    }:
        raise HitabP1SourceError("source HTTP content encoding drifted")


def _download_one_part(
    contract: SourceFileContract,
    *,
    part_path: Path,
    opener: Any,
) -> VerifiedFileIdentity:
    url = _validate_download_url(contract)
    _ensure_private_directory(part_path.parent)
    descriptor = os.open(
        part_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()  # nosec B324: immutable Git object identity
    blob.update(f"blob {contract.size_bytes}\0".encode("ascii"))
    total = 0
    newline_count = 0
    try:
        os.fchmod(descriptor, 0o600)
        response = opener(url)
        try:
            _validate_download_response(response, expected_url=url)
            headers = getattr(response, "headers", {})
            content_length = headers.get("Content-Length")
            if content_length is not None:
                try:
                    parsed_length = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise HitabP1SourceError(
                        "source HTTP content length is invalid"
                    ) from exc
                if parsed_length != contract.size_bytes:
                    raise HitabP1SourceError(
                        "source HTTP content length drifted"
                    )
            while True:
                chunk = response.read(HTTP_READ_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise HitabP1SourceError(
                        "source HTTP body yielded non-bytes"
                    )
                total += len(chunk)
                if total > contract.size_bytes:
                    raise HitabP1SourceError(
                        "source HTTP body exceeded exact size"
                    )
                sha256.update(chunk)
                blob.update(chunk)
                if contract.is_jsonl:
                    newline_count += chunk.count(b"\n")
                _write_all(descriptor, chunk)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != contract.size_bytes
        ):
            raise HitabP1SourceError("download part metadata drifted")
    finally:
        os.close(descriptor)
    observed_blob = blob.hexdigest()
    if (
        total != contract.size_bytes
        or not hmac.compare_digest(observed_blob, contract.git_blob_sha1)
    ):
        raise HitabP1SourceError("downloaded source identity drifted")
    return VerifiedFileIdentity(
        key=contract.key,
        size_bytes=total,
        sha256=sha256.hexdigest(),
        git_blob_sha1=observed_blob,
        raw_newline_count=newline_count if contract.is_jsonl else None,
    )


def _publish_part_exclusive(part_path: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise HitabP1SourceError("source destination already exists")
    try:
        os.link(part_path, destination, follow_symlinks=False)
        _fsync_directory(destination.parent)
        os.unlink(part_path)
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise HitabP1SourceError("source part publication failed") from exc


def download_source_set_once(
    *,
    source_root: Path,
    control_root: Path,
    custody_path: Path = PROJECT_ROOT / CUSTODY_RELATIVE,
    design_path: Path = PROJECT_ROOT / DESIGN_RELATIVE,
    contracts: Mapping[str, SourceFileContract] = FORMAL_SOURCE_CONTRACTS,
    opener: Any | None = None,
) -> DownloadedSourceSet:
    """Perform exactly four concurrent pinned GETs in one consumed attempt.

    All four intents are submitted before joining any result.  Bodies remain in
    O_EXCL part files until every file passes exact size and Git-blob identity.
    Any error is terminal for the whole attempt; every future is joined, every
    part is removed, no destination is published, and no retry is attempted.
    """

    verify_frozen_bindings(custody_path, design_path)
    if tuple(contracts) != ("TRAIN", "DEV", "TEST", "TABLES"):
        raise HitabP1SourceError("four-file download order drifted")
    if any(contract.key != key for key, contract in contracts.items()):
        raise HitabP1SourceError("four-file download registry drifted")
    for contract in contracts.values():
        _validate_download_url(contract)
    source = Path(source_root)
    control = Path(control_root)
    _ensure_private_directory(source)
    marker = control / DOWNLOAD_ATTEMPT_FILENAME
    receipt_path = control / DOWNLOAD_RECEIPT_FILENAME
    failure_path = control / DOWNLOAD_FAILURE_FILENAME
    write_json_exclusive(
        marker,
        self_hashed(
            {
                "file_intent_count": 4,
                "parallel_transport_count": 4,
                "retry_resume_range_mirror_or_provider_switch_count": 0,
                "schema": "hitab_p1_source_download_attempt_v1",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        ),
    )
    source_paths: dict[str, Path] = {}
    part_paths: dict[str, Path] = {}
    for key, contract in contracts.items():
        destination = source / PurePosixPath(contract.relative_path)
        _ensure_private_directory(destination.parent)
        part = destination.with_name(f".{destination.name}.one_shot.part")
        if (
            destination.exists()
            or destination.is_symlink()
            or part.exists()
            or part.is_symlink()
        ):
            try:
                write_json_exclusive(
                    failure_path,
                    self_hashed(
                        {
                            "attempted_file_count": 0,
                            "failure_class": "preexisting_source_or_part",
                            "retry_count": 0,
                            "schema": "hitab_p1_source_download_failure_v1",
                            "status": "terminal_attempt_consumed",
                            "study_id": STUDY_ID,
                            "version": VERSION,
                        }
                    ),
                )
            finally:
                raise HitabP1SourceError(
                    "source destination or part already exists"
                )
        source_paths[key] = destination
        part_paths[key] = part
    actual_opener = _default_http_opener if opener is None else opener
    identities: dict[str, VerifiedFileIdentity] = {}
    errors = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="hitab-download") as pool:
        for key, contract in contracts.items():
            future = pool.submit(
                _download_one_part,
                contract,
                part_path=part_paths[key],
                opener=actual_opener,
            )
            futures[future] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                identities[key] = future.result()
            except Exception:
                errors += 1
    if errors:
        for part in part_paths.values():
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
        failure = self_hashed(
            {
                "attempted_file_count": 4,
                "completed_verified_part_count": len(identities),
                "failure_class": "parallel_four_file_acquisition_failure",
                "failed_file_count": errors,
                "retry_resume_range_mirror_or_provider_switch_count": 0,
                "schema": "hitab_p1_source_download_failure_v1",
                "status": "terminal_attempt_consumed_no_source_published",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        )
        write_json_exclusive(failure_path, failure)
        raise HitabP1SourceError("formal four-file acquisition failed closed")
    try:
        for key in ("TRAIN", "DEV", "TEST", "TABLES"):
            _publish_part_exclusive(part_paths[key], source_paths[key])
    except Exception:
        # Publication is inside the same one-shot terminal.  Previously
        # published exact bodies are evidence, never authorization to retry.
        failure = self_hashed(
            {
                "attempted_file_count": 4,
                "failure_class": "atomic_source_publication_failure",
                "retry_count": 0,
                "schema": "hitab_p1_source_download_failure_v1",
                "status": "terminal_attempt_consumed",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        )
        write_json_exclusive(failure_path, failure)
        raise
    identity_payload = {
        key: identities[key].safe_payload()
        for key in ("TRAIN", "DEV", "TEST", "TABLES")
    }
    receipt = self_hashed(
        {
            "file_count": 4,
            "files": identity_payload,
            "json_decode_count": 0,
            "network_attempt_count": 4,
            "parallel_transport_count": 4,
            "retry_resume_range_mirror_or_provider_switch_count": 0,
            "schema": "hitab_p1_source_download_receipt_v1",
            "source_identity_commitment": stable_hash(identity_payload),
            "status": "four_exact_sources_acquired_once",
            "study_id": STUDY_ID,
            "test_json_decode_count": 0,
            "version": VERSION,
        }
    )
    write_json_exclusive(receipt_path, receipt)
    verified = VerifiedSourceSet(identities=identities, safe_receipt=receipt)
    return DownloadedSourceSet(
        source_paths=source_paths,
        verified_sources=verified,
    )


def _stream_file_identity(
    path: Path,
    *,
    contract: SourceFileContract,
    expected_sha256: str,
    require_mode_0600: bool,
) -> VerifiedFileIdentity:
    if _HEX64.fullmatch(expected_sha256) is None:
        raise HitabP1SourceError("expected source SHA256 is invalid")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HitabP1SourceError("pinned source file is unavailable") from exc
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()  # nosec B324: immutable Git object identity
    blob.update(f"blob {contract.size_bytes}\0".encode("ascii"))
    newline_count = 0
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != contract.size_bytes
            or (
                require_mode_0600
                and stat.S_IMODE(before.st_mode) != 0o600
            )
        ):
            raise HitabP1SourceError("pinned source file metadata drifted")
        while True:
            chunk = os.read(descriptor, 4 << 20)
            if not chunk:
                break
            total += len(chunk)
            sha256.update(chunk)
            blob.update(chunk)
            if contract.is_jsonl:
                newline_count += chunk.count(b"\n")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise HitabP1SourceError("pinned source changed during identity read")
    finally:
        os.close(descriptor)
    observed_sha256 = sha256.hexdigest()
    observed_blob = blob.hexdigest()
    if (
        total != contract.size_bytes
        or not hmac.compare_digest(observed_sha256, expected_sha256)
        or not hmac.compare_digest(observed_blob, contract.git_blob_sha1)
    ):
        raise HitabP1SourceError("pinned source byte identity drifted")
    return VerifiedFileIdentity(
        key=contract.key,
        size_bytes=total,
        sha256=observed_sha256,
        git_blob_sha1=observed_blob,
        raw_newline_count=newline_count if contract.is_jsonl else None,
    )


def verify_source_set_once(
    source_paths: Mapping[str, Path],
    *,
    expected_sha256_by_key: Mapping[str, str],
    control_root: Path,
    contracts: Mapping[str, SourceFileContract] = FORMAL_SOURCE_CONTRACTS,
    require_mode_0600: bool = True,
) -> VerifiedSourceSet:
    """Consume one identity attempt and emit only a safe aggregate receipt."""

    if set(source_paths) != set(contracts) or set(expected_sha256_by_key) != set(
        contracts
    ):
        raise HitabP1SourceError("source identity key set drifted")
    if any(contract.key != key for key, contract in contracts.items()):
        raise HitabP1SourceError("source contract key drifted")
    root = Path(control_root)
    marker = root / SOURCE_ATTEMPT_FILENAME
    receipt_path = root / SOURCE_RECEIPT_FILENAME
    failure_path = root / SOURCE_FAILURE_FILENAME
    write_json_exclusive(
        marker,
        self_hashed(
            {
                "schema": "hitab_p1_source_identity_attempt_v1",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        ),
    )
    identities: dict[str, VerifiedFileIdentity] = {}
    try:
        for key in ("TRAIN", "DEV", "TEST", "TABLES"):
            if key not in contracts:
                raise HitabP1SourceError("source identity order drifted")
            identities[key] = _stream_file_identity(
                Path(source_paths[key]),
                contract=contracts[key],
                expected_sha256=expected_sha256_by_key[key],
                require_mode_0600=require_mode_0600,
            )
        identity_payload = {
            key: identities[key].safe_payload()
            for key in ("TRAIN", "DEV", "TEST", "TABLES")
        }
        body = {
            "file_count": len(identities),
            "files": identity_payload,
            "json_decode_count": 0,
            "online_or_network_call_count": 0,
            "schema": "hitab_p1_source_identity_receipt_v1",
            "source_identity_commitment": stable_hash(identity_payload),
            "status": "four_exact_identities_verified",
            "study_id": STUDY_ID,
            "test_json_decode_count": 0,
            "version": VERSION,
        }
        receipt = self_hashed(body)
        write_json_exclusive(receipt_path, receipt)
        return VerifiedSourceSet(identities=identities, safe_receipt=receipt)
    except Exception as exc:
        failure = self_hashed(
            {
                "completed_identity_count": len(identities),
                "failure_class": "source_identity_contract_failure",
                "retry_replay_or_resample_count": 0,
                "schema": "hitab_p1_source_identity_failure_v1",
                "status": "terminal_attempt_consumed",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        )
        try:
            write_json_exclusive(failure_path, failure)
        except Exception:
            pass
        if isinstance(exc, HitabP1SourceError):
            raise
        raise HitabP1SourceError("source identity verification failed closed") from exc


def family_from_aggregation(value: object) -> str:
    if not isinstance(value, list) or not value:
        raise HitabP1RowIneligible("aggregation_not_nonempty_list")
    families: set[str] = set()
    seen: set[str] = set()
    for token in value:
        if (
            not isinstance(token, str)
            or token != token.strip()
            or token != unicodedata.normalize("NFKC", token)
        ):
            raise HitabP1RowIneligible("aggregation_token_invalid")
        canonical = token
        if not canonical or canonical in seen:
            raise HitabP1RowIneligible("aggregation_token_invalid")
        seen.add(canonical)
        family = FAMILY_BY_AGGREGATION_TOKEN.get(canonical)
        if family is None:
            raise HitabP1RowIneligible("aggregation_unknown_or_none")
        families.add(family)
    if len(families) != 1:
        raise HitabP1RowIneligible("aggregation_cross_family")
    return next(iter(families))


def _safe_source_text(
    value: object, *, maximum: int, reason: str
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise HitabP1RowIneligible(reason)
    return value


def parse_sample_row(
    value: object,
    *,
    split: str,
    public_exposure_hashes: Mapping[str, frozenset[str]] = PUBLIC_EXPOSURE_HASHES,
) -> SampleCandidate:
    if split not in {"TRAIN", "DEV", "TEST"}:
        raise HitabP1SourceError("sample split is invalid")
    if not isinstance(value, Mapping):
        raise HitabP1RowIneligible("row_not_object")
    if not REQUIRED_SAMPLE_FIELDS.issubset(value):
        raise HitabP1RowIneligible("required_semantic_field_missing")
    item_id = _safe_source_text(
        value.get("id"),
        maximum=MAX_IDENTIFIER_CHARACTERS,
        reason="item_id_invalid",
    )
    table_id = _safe_source_text(
        value.get("table_id"),
        maximum=MAX_IDENTIFIER_CHARACTERS,
        reason="table_id_invalid",
    )
    question = _safe_source_text(
        value.get("question"),
        maximum=MAX_QUESTION_CHARACTERS,
        reason="question_invalid",
    )
    source = value.get("table_source")
    if (
        not isinstance(source, str)
        or source != source.strip()
        or unicodedata.normalize("NFKC", source).casefold()
        not in ACCEPTED_TABLE_SOURCES
    ):
        raise HitabP1RowIneligible("table_source_not_statcan_or_nsf")
    linked = value.get("linked_cells")
    if not isinstance(linked, Mapping):
        raise HitabP1RowIneligible("linked_cells_invalid")
    family = family_from_aggregation(value.get("aggregation"))
    exposure_values = {
        "id": item_id,
        "question": question,
        "table_id": table_id,
    }
    if set(public_exposure_hashes) != set(exposure_values):
        raise HitabP1SourceError("public exposure registry drifted")
    if any(
        normalized_text_sha256(exposure_values[field])
        in public_exposure_hashes[field]
        for field in exposure_values
    ):
        raise HitabP1RowIneligible("public_example_excluded")
    return SampleCandidate(
        split=split,
        item_id=item_id,
        table_id=table_id,
        table_source=unicodedata.normalize("NFKC", source).casefold(),
        question=question,
        family=family,
        linked_cells=dict(linked),
    )


def parse_sample_jsonl_bytes(
    raw: bytes,
    *,
    split: str,
    allow_test_decode: bool = False,
    public_exposure_hashes: Mapping[str, frozenset[str]] = PUBLIC_EXPOSURE_HASHES,
) -> SampleParseResult:
    """Strictly decode one split, except TEST which is denied by default."""

    if not isinstance(raw, bytes):
        raise HitabP1SourceError("sample JSONL input is not bytes")
    if split == "TEST" and not allow_test_decode:
        raise HitabP1SourceError(
            "TEST JSON decode requires a sealed promotion authorization"
        )
    if split not in {"TRAIN", "DEV", "TEST"}:
        raise HitabP1SourceError("sample split is invalid")
    if not raw or not raw.endswith(b"\n"):
        raise HitabP1SourceError("source JSONL must be nonempty and newline terminated")
    candidates: list[SampleCandidate] = []
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    decode_count = 0
    lines = raw.splitlines()
    if len(lines) > MAX_JSONL_ROWS:
        raise HitabP1SourceError("source JSONL row bound exceeded")
    for line in lines:
        if not line or len(line) > MAX_JSONL_LINE_BYTES:
            raise HitabP1SourceError("source JSONL line bound drifted")
        row = strict_json(line, label="sample JSONL row")
        decode_count += 1
        try:
            candidate = parse_sample_row(
                row,
                split=split,
                public_exposure_hashes=public_exposure_hashes,
            )
            if candidate.item_id in seen_ids:
                raise HitabP1RowIneligible("duplicate_item_id")
            seen_ids.add(candidate.item_id)
            candidates.append(candidate)
        except HitabP1RowIneligible as exc:
            reasons[exc.reason] += 1
    family_counts = Counter(candidate.family for candidate in candidates)
    summary = {
        "accepted_count": len(candidates),
        "accepted_count_by_family": {
            family: family_counts[family] for family in FAMILIES
        },
        "extra_fields_allowed": True,
        "json_decode_count": decode_count,
        "row_exclusion_reason_counts": dict(sorted(reasons.items())),
        "source_row_count": len(lines),
        "split": split,
    }
    return SampleParseResult(tuple(candidates), summary)


def parse_sample_jsonl_path(
    path: Path,
    *,
    split: str,
    allow_test_decode: bool = False,
    require_mode_0600: bool = True,
    public_exposure_hashes: Mapping[str, frozenset[str]] = PUBLIC_EXPOSURE_HASHES,
) -> SampleParseResult:
    raw = _read_regular_bytes(
        Path(path),
        maximum_bytes=max(contract.size_bytes for contract in FORMAL_SOURCE_CONTRACTS.values()),
        require_mode_0600=require_mode_0600,
        label="sample JSONL",
    )
    return parse_sample_jsonl_bytes(
        raw,
        split=split,
        allow_test_decode=allow_test_decode,
        public_exposure_hashes=public_exposure_hashes,
    )


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    require_mode_0600: bool,
    label: str,
) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HitabP1SourceError(f"{label} is unavailable") from exc
    chunks: list[bytes] = []
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                require_mode_0600
                and stat.S_IMODE(before.st_mode) != 0o600
            )
            or before.st_size > maximum_bytes
        ):
            raise HitabP1SourceError(f"{label} metadata drifted")
        while True:
            chunk = os.read(descriptor, min(4 << 20, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise HitabP1SourceError(f"{label} size bound exceeded")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise HitabP1SourceError(f"{label} changed during read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def test_identity_only_summary(
    verified_sources: VerifiedSourceSet,
) -> Mapping[str, Any]:
    identity = verified_sources.identities.get("TEST")
    if not isinstance(identity, VerifiedFileIdentity):
        raise HitabP1SourceError("verified TEST identity is missing")
    return {
        "json_decode_count": 0,
        "raw_newline_count": identity.raw_newline_count,
        "source_identity_commitment": verified_sources.source_identity_commitment,
        "split": "TEST",
        "status": "identity_and_newline_only_awaiting_promotion",
    }


def _zip_member_name_is_safe(name: str) -> bool:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
    ):
        return False
    raw_parts = name.split("/")
    if raw_parts[-1] == "":
        raw_parts = raw_parts[:-1]
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        return False
    if ":" in raw_parts[0]:
        return False
    path = PurePosixPath(*raw_parts)
    return not path.is_absolute() and ".." not in path.parts


def _zip_member_kind(info: zipfile.ZipInfo) -> str:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if info.is_dir():
        if file_type not in {0, stat.S_IFDIR}:
            raise HitabP1SourceError("ZIP directory member type drifted")
        return "directory"
    if file_type == stat.S_IFLNK:
        raise HitabP1SourceError("ZIP symlink is forbidden")
    if file_type not in {0, stat.S_IFREG}:
        raise HitabP1SourceError("ZIP special member is forbidden")
    return "regular"


def _validated_zip_index(
    archive: zipfile.ZipFile,
    *,
    table_id: str,
    max_member_bytes: int,
    max_total_bytes: int,
) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    if (
        not isinstance(table_id, str)
        or _SAFE_TABLE_ID.fullmatch(table_id) is None
        or table_id in {".", ".."}
    ):
        raise HitabP1RowIneligible("table_id_not_safe_archive_basename")
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ZIP_MEMBERS:
        raise HitabP1SourceError("ZIP member count is outside contract")
    seen: set[str] = set()
    normalized_seen: set[str] = set()
    hmt_matches: list[zipfile.ZipInfo] = []
    raw_matches: list[zipfile.ZipInfo] = []
    total = 0
    expected_basename = table_id + ".json"
    for info in infos:
        name = info.filename
        if not _zip_member_name_is_safe(name):
            raise HitabP1SourceError("ZIP path traversal or unsafe name detected")
        normalized = unicodedata.normalize("NFKC", name)
        if name in seen or normalized in normalized_seen:
            raise HitabP1SourceError("ZIP duplicate member name detected")
        seen.add(name)
        normalized_seen.add(normalized)
        kind = _zip_member_kind(info)
        if (
            type(info.file_size) is not int
            or info.file_size < 0
            or info.file_size > max_member_bytes
            or type(info.compress_size) is not int
            or info.compress_size < 0
        ):
            raise HitabP1SourceError("ZIP member size is outside contract")
        total += info.file_size
        if total > max_total_bytes:
            raise HitabP1SourceError("ZIP total uncompressed size is outside contract")
        if (
            info.file_size > 0
            and (
                info.compress_size == 0
                or info.file_size
                > max(info.compress_size, 1) * MAX_ZIP_COMPRESSION_RATIO
            )
        ):
            raise HitabP1SourceError("ZIP compression ratio is outside contract")
        parts = PurePosixPath(name.rstrip("/")).parts
        if (
            kind == "regular"
            and len(parts) >= 2
            and parts[-1] == expected_basename
        ):
            if parts[-2] == "hmt":
                hmt_matches.append(info)
            elif parts[-2] == "raw":
                raw_matches.append(info)
    if not hmt_matches or not raw_matches:
        raise RequestedTableNotFound()
    if len(hmt_matches) != 1 or len(raw_matches) != 1:
        raise HitabP1SourceError("requested table member pair is not unique")
    hmt_target = hmt_matches[0]
    raw_target = raw_matches[0]
    if (
        hmt_target.file_size > MAX_TABLE_JSON_BYTES
        or raw_target.file_size > MAX_TABLE_JSON_BYTES
    ):
        raise HitabP1SourceError("requested table JSON is oversized")
    return hmt_target, raw_target


def _read_requested_table_from_archive(
    archive: zipfile.ZipFile,
    table_id: str,
    *,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_total_bytes: int = MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES,
) -> RequestedTablePayload:
    hmt_target, raw_target = _validated_zip_index(
        archive,
        table_id=table_id,
        max_member_bytes=max_member_bytes,
        max_total_bytes=max_total_bytes,
    )
    # These are the only two payload-open calls in this function.
    with archive.open(hmt_target, mode="r") as handle:
        hmt_raw = handle.read(MAX_TABLE_JSON_BYTES + 1)
        if (
            len(hmt_raw) != hmt_target.file_size
            or len(hmt_raw) > MAX_TABLE_JSON_BYTES
        ):
            raise HitabP1SourceError("requested HMT payload size drifted")
    with archive.open(raw_target, mode="r") as handle:
        raw_raw = handle.read(MAX_TABLE_JSON_BYTES + 1)
        if (
            len(raw_raw) != raw_target.file_size
            or len(raw_raw) > MAX_TABLE_JSON_BYTES
        ):
            raise HitabP1SourceError("requested raw table payload size drifted")
    hmt_value = strict_json(hmt_raw, label="requested HMT table")
    raw_value = strict_json(raw_raw, label="requested raw table")
    if not isinstance(hmt_value, Mapping) or not isinstance(raw_value, Mapping):
        raise HitabP1SourceError("requested HMT table is not an object")
    return RequestedTablePayload(hmt=dict(hmt_value), raw=dict(raw_value))


def read_requested_table_from_zip(
    zip_path: Path,
    table_id: str,
    *,
    require_mode_0600: bool = True,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_total_bytes: int = MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES,
    zip_file_factory: Any = zipfile.ZipFile,
) -> RequestedTablePayload:
    """Open exactly one safe HMT and one safe raw member for ``table_id``."""

    try:
        metadata = Path(zip_path).lstat()
    except OSError as exc:
        raise HitabP1SourceError("tables ZIP is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (require_mode_0600 and stat.S_IMODE(metadata.st_mode) != 0o600)
    ):
        raise HitabP1SourceError("tables ZIP metadata drifted")
    try:
        with zip_file_factory(Path(zip_path), mode="r") as archive:
            return _read_requested_table_from_archive(
                archive,
                table_id,
                max_member_bytes=max_member_bytes,
                max_total_bytes=max_total_bytes,
            )
    except HitabP1RowIneligible:
        raise
    except HitabP1SourceError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        raise HitabP1SourceError("tables ZIP is invalid") from exc


def _read_requested_table_from_zip_bytes(
    raw: bytes,
    table_id: str,
) -> RequestedTablePayload:
    if not isinstance(raw, bytes) or not raw:
        raise HitabP1SourceError("verified tables ZIP bytes are unavailable")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), mode="r") as archive:
            return _read_requested_table_from_archive(archive, table_id)
    except HitabP1RowIneligible:
        raise
    except HitabP1SourceError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        raise HitabP1SourceError("tables ZIP is invalid") from exc


def _header_surface(value: object, *, field: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise HitabP1SourceError(f"{field} is invalid")
    result = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    if not result or len(result) > 16_000:
        raise HitabP1SourceError(f"{field} is invalid")
    return result


def _parse_header_tree(
    value: object,
    *,
    axis_size: int,
    direction: str,
) -> tuple[HeaderNode, Mapping[int, HeaderNode], Mapping[tuple[int, ...], HeaderNode]]:
    if direction not in {"left", "top"}:
        raise HitabP1SourceError("header tree direction is invalid")
    line_nodes: dict[int, HeaderNode] = {}
    all_nodes: dict[tuple[int, ...], HeaderNode] = {}
    count = 0

    def parse(
        raw_node: object,
        *,
        key: tuple[int, ...],
        depth: int,
    ) -> HeaderNode:
        nonlocal count
        count += 1
        if count > MAX_TREE_NODES or depth > MAX_TREE_DEPTH:
            raise HitabP1SourceError("header tree bound exceeded")
        if not isinstance(raw_node, Mapping):
            raise HitabP1SourceError("header tree node is not an object")
        if not {"name", "value", "line_idx"}.issubset(raw_node):
            raise HitabP1SourceError("header tree required field is missing")
        name = _header_surface(raw_node.get("name"), field="header name")
        line_idx = raw_node.get("line_idx")
        if line_idx is not None and (
            type(line_idx) is not int or not 0 <= line_idx < axis_size
        ):
            raise HitabP1SourceError("header line index is invalid")
        children_raw = raw_node.get("children_dict", [])
        if children_raw is None:
            children_raw = []
        if not isinstance(children_raw, list):
            raise HitabP1SourceError("header children_dict is invalid")
        children = tuple(
            parse(child, key=key + (index,), depth=depth + 1)
            for index, child in enumerate(children_raw)
        )
        node = HeaderNode(
            node_key=key,
            name=name,
            line_idx=line_idx,
            children=children,
        )
        if key in all_nodes:
            raise HitabP1SourceError("header node key is duplicated")
        all_nodes[key] = node
        if line_idx is not None:
            if line_idx in line_nodes:
                raise HitabP1SourceError("header line index is duplicated")
            line_nodes[line_idx] = node
        return node

    root = parse(value, key=(), depth=0)
    if set(line_nodes) != set(range(axis_size)):
        raise HitabP1SourceError("header leaves do not cover complete data axis")
    return root, line_nodes, all_nodes


def _node_path(
    node: HeaderNode,
    *,
    nodes: Mapping[tuple[int, ...], HeaderNode],
    sentinel: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    keys = tuple(node.node_key[:depth] for depth in range(len(node.node_key) + 1))
    path_nodes = tuple(nodes[key] for key in keys)
    filtered = tuple(
        row
        for index, row in enumerate(path_nodes)
        if not (index == 0 and row.name.casefold() == sentinel.casefold())
    )
    return tuple(row.node_key for row in filtered), tuple(row.name for row in filtered)


def _decimal_canonical(value: Decimal) -> str:
    if not value.is_finite():
        raise HitabP1SourceError("numeric cell is non-finite")
    if value == 0:
        return "0"
    normalized = value.normalize()
    rendered = format(normalized, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def canonical_typed_literal(value: object) -> tuple[str, str, str]:
    """Return ``(value_type, typed_literal, surface)`` deterministically."""

    if value is None:
        raise HitabP1RowIneligible("empty_data_cell")
    if isinstance(value, bool):
        surface = "true" if value else "false"
        return "BOOLEAN", "BOOLEAN:" + surface, surface
    if type(value) is int:
        rendered = str(value)
        return "NUMBER", "NUMBER:" + rendered, rendered
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HitabP1SourceError("numeric cell is non-finite")
        rendered = _decimal_canonical(Decimal(str(value)))
        return "NUMBER", "NUMBER:" + rendered, rendered
    if not isinstance(value, str) or "\x00" in value:
        raise HitabP1SourceError("data cell value schema drifted")
    surface = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    if not surface:
        raise HitabP1RowIneligible("empty_data_cell")
    if len(surface) > 32_000:
        raise HitabP1SourceError("data cell value is oversized")
    match = _NUMBER.fullmatch(surface)
    if match is not None:
        try:
            number = Decimal(
                match.group("sign") + match.group("number").replace(",", "")
            )
        except InvalidOperation as exc:
            raise HitabP1SourceError("numeric cell cannot be canonicalized") from exc
        rendered = _decimal_canonical(number)
        if match.group("percent"):
            return "PERCENT", "PERCENT:" + rendered, surface
        return "NUMBER", "NUMBER:" + rendered, surface
    return "TEXT", "TEXT:" + normalized_text(surface), surface


def _validated_raw_table(
    value: object,
    *,
    data_rows: int,
    data_columns: int,
) -> tuple[list[list[object]], int, int]:
    if not isinstance(value, Mapping):
        raise HitabP1SourceError("raw table is not an object")
    if not {
        "texts",
        "top_header_rows_num",
        "left_header_columns_num",
    }.issubset(value):
        raise HitabP1SourceError("raw table required semantic field is missing")
    texts = value.get("texts")
    top_rows = value.get("top_header_rows_num")
    left_columns = value.get("left_header_columns_num")
    if (
        type(top_rows) is not int
        or type(left_columns) is not int
        or top_rows < 1
        or left_columns < 1
        or not isinstance(texts, list)
        or len(texts) != top_rows + data_rows
        or not texts
        or not isinstance(texts[0], list)
        or len(texts[0]) != left_columns + data_columns
    ):
        raise HitabP1SourceError("raw table matrix/header offset shape drifted")
    width = len(texts[0])
    if any(not isinstance(row, list) or len(row) != width for row in texts):
        raise HitabP1SourceError("raw table texts are not rectangular")
    return texts, top_rows, left_columns


def _optional_typed_literal(
    value: object,
) -> tuple[str, str, str] | None:
    try:
        return canonical_typed_literal(value)
    except HitabP1RowIneligible as exc:
        if exc.reason == "empty_data_cell":
            return None
        raise


def parse_hmt_table(value: object, raw_table: object) -> TableView:
    if not isinstance(value, Mapping):
        raise HitabP1SourceError("HMT table is not an object")
    if not {"title", "top_root", "left_root", "data"}.issubset(value):
        raise HitabP1SourceError("HMT table required semantic field is missing")
    title = _header_surface(value.get("title"), field="table title")
    data = value.get("data")
    if (
        not isinstance(data, list)
        or not data
        or len(data) > MAX_DATA_ROWS
        or not isinstance(data[0], list)
        or not data[0]
        or len(data[0]) > MAX_DATA_COLUMNS
    ):
        raise HitabP1SourceError("HMT data matrix shape is invalid")
    width = len(data[0])
    if any(not isinstance(row, list) or len(row) != width for row in data):
        raise HitabP1SourceError("HMT data matrix is not rectangular")
    raw_texts, top_header_rows_num, left_header_columns_num = (
        _validated_raw_table(
            raw_table,
            data_rows=len(data),
            data_columns=width,
        )
    )
    _top_root, top_lines, top_nodes = _parse_header_tree(
        value.get("top_root"), axis_size=width, direction="top"
    )
    _left_root, left_lines, left_nodes = _parse_header_tree(
        value.get("left_root"), axis_size=len(data), direction="left"
    )
    top_paths = {
        index: _node_path(
            node, nodes=top_nodes, sentinel="<TOP>"
        )
        for index, node in top_lines.items()
    }
    left_paths = {
        index: _node_path(
            node, nodes=left_nodes, sentinel="<LEFT>"
        )
        for index, node in left_lines.items()
    }
    units: list[AtomicUnit] = []
    for row_index, raw_row in enumerate(data):
        for column_index, raw_cell in enumerate(raw_row):
            if not isinstance(raw_cell, Mapping) or "value" not in raw_cell:
                raise HitabP1SourceError("HMT DataNode schema drifted")
            hmt_typed = _optional_typed_literal(raw_cell.get("value"))
            raw_typed = _optional_typed_literal(
                raw_texts[row_index + top_header_rows_num][
                    column_index + left_header_columns_num
                ]
            )
            if hmt_typed is None and raw_typed is None:
                continue
            if (
                hmt_typed is None
                or raw_typed is None
                or hmt_typed[1] != raw_typed[1]
            ):
                raise HitabP1SourceError(
                    "HMT and raw data-cell typed value drifted"
                )
            value_type, typed_literal, surface = hmt_typed
            left_keys, left_surface = left_paths[row_index]
            top_keys, top_surface = top_paths[column_index]
            serialized = canonical_bytes(
                {
                    "LEFT_PATH": list(left_surface),
                    "TOP_PATH": list(top_surface),
                    "VALUE": surface,
                    "VALUE_type": value_type,
                }
            ).decode("ascii")
            units.append(
                AtomicUnit(
                    ordinal=len(units),
                    row_index=row_index,
                    column_index=column_index,
                    value_surface=surface,
                    value_type=value_type,
                    typed_literal=typed_literal,
                    left_header_path=left_surface,
                    top_header_path=top_surface,
                    left_node_path=left_keys,
                    top_node_path=top_keys,
                    serialized=serialized,
                )
            )
    if not MIN_CORPUS_UNITS <= len(units) <= MAX_CORPUS_UNITS:
        raise HitabP1RowIneligible("ordered_corpus_size_outside_10_256")
    return TableView(
        title=title,
        units=tuple(units),
        top_header_rows_num=top_header_rows_num,
        left_header_columns_num=left_header_columns_num,
    )


def _has_shared_prefix(
    left: Sequence[tuple[int, ...]], right: Sequence[tuple[int, ...]]
) -> bool:
    return any(a == b for a, b in zip(left, right))


def _parse_matrix_coordinate(value: object) -> tuple[int, int]:
    if not isinstance(value, str):
        raise HitabP1RowIneligible("quantity_answer_coordinate_invalid")
    match = _MATRIX_COORDINATE.fullmatch(value)
    if match is None:
        raise HitabP1RowIneligible("quantity_answer_coordinate_invalid")
    return int(match.group(1)), int(match.group(2))


def build_coordinate_qrel_dnf(
    candidate: SampleCandidate, table: TableView
) -> ProofDNF:
    """Build one proof whose annotated coordinates are singleton requirements.

    Each key under ``linked_cells.quantity_link['[ANSWER]']`` is an
    independently required matrix data cell.  The raw table's exact header
    row/column counts provide the fixed full-matrix-to-data-region offset.
    Literal equality never grants credit to an unannotated coordinate,
    including repeated equal values.
    """

    quantity = candidate.linked_cells.get("quantity_link")
    if not isinstance(quantity, Mapping):
        raise HitabP1RowIneligible("quantity_link_missing_or_invalid")
    answer = quantity.get("[ANSWER]")
    if not isinstance(answer, Mapping) or not answer:
        raise HitabP1RowIneligible("quantity_answer_missing_or_invalid")
    if not 1 <= len(answer) <= MAX_PROOF_REQUIREMENTS:
        raise HitabP1RowIneligible("proof_requirement_count_outside_1_5")
    coordinate_map = table.matrix_coordinate_to_ordinal
    requirements: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    seen_coordinates: set[tuple[int, int]] = set()
    for raw_coordinate, literal in answer.items():
        coordinate = _parse_matrix_coordinate(raw_coordinate)
        if coordinate in seen_coordinates:
            raise HitabP1RowIneligible("quantity_answer_coordinate_duplicated")
        seen_coordinates.add(coordinate)
        ordinal = coordinate_map.get(coordinate)
        if ordinal is None:
            raise HitabP1RowIneligible(
                "proof_coordinate_unresolved_in_data_region"
            )
        try:
            _value_type, annotated_typed_literal, _surface = (
                canonical_typed_literal(literal)
            )
        except (HitabP1RowIneligible, HitabP1SourceError) as exc:
            raise HitabP1RowIneligible(
                "quantity_answer_literal_invalid"
            ) from exc
        if table.units[ordinal].typed_literal != annotated_typed_literal:
            raise HitabP1RowIneligible(
                "annotation_raw_HMT_typed_value_mismatch"
            )
        requirements.append((coordinate, (ordinal,)))
    proof = tuple(bucket for _coordinate, bucket in sorted(requirements))
    try:
        return ProofDNF(
            alternatives=(proof,),
            corpus_commitment=table.corpus_commitment,
        )
    except dmc_core.HitabDmc1CoreError as exc:
        raise HitabP1RowIneligible("coordinate_qrel_DNF_invalid") from exc


def materialize_candidates(
    candidates: Sequence[SampleCandidate],
    *,
    tables_zip_path: Path | None = None,
    tables_zip_bytes: bytes | None = None,
    require_mode_0600: bool = True,
) -> tuple[tuple[EligibleItem, ...], Mapping[str, int]]:
    """Materialize candidate tables/qrels while retaining only safe reasons."""

    if (tables_zip_path is None) == (tables_zip_bytes is None):
        raise HitabP1SourceError(
            "exactly one tables ZIP materialization source is required"
        )
    table_cache: dict[str, TableView | HitabP1RowIneligible] = {}
    eligible: list[EligibleItem] = []
    reasons: Counter[str] = Counter()
    for candidate in candidates:
        cached = table_cache.get(candidate.table_id)
        if cached is None:
            try:
                if tables_zip_bytes is None:
                    table_payload = read_requested_table_from_zip(
                        Path(tables_zip_path),
                        candidate.table_id,
                        require_mode_0600=require_mode_0600,
                    )
                else:
                    table_payload = _read_requested_table_from_zip_bytes(
                        tables_zip_bytes,
                        candidate.table_id,
                    )
            except HitabP1RowIneligible as exc:
                cached = exc
            else:
                try:
                    cached = parse_hmt_table(
                        table_payload.hmt,
                        table_payload.raw,
                    )
                except HitabP1RowIneligible as exc:
                    cached = exc
                except HitabP1SourceError:
                    # The ZIP and JSON syntax already passed globally fatal
                    # checks.  A requested table's semantic schema drift is a
                    # fixed row/table exclusion, never a parser repair surface.
                    cached = HitabP1RowIneligible(
                        "requested_table_semantic_schema_invalid"
                    )
            table_cache[candidate.table_id] = cached
        if isinstance(cached, HitabP1RowIneligible):
            reasons[cached.reason] += 1
            continue
        try:
            qrel = build_coordinate_qrel_dnf(candidate, cached)
        except HitabP1RowIneligible as exc:
            reasons[exc.reason] += 1
            continue
        eligible.append(EligibleItem(candidate=candidate, table=cached, qrel=qrel))
    return tuple(eligible), dict(sorted(reasons.items()))


def _frame(name: bytes, value: str) -> bytes:
    if not isinstance(name, bytes) or not name or b"\x00" in name:
        raise HitabP1SourceError("HMAC frame name is invalid")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise HitabP1SourceError("HMAC frame value is invalid")
    raw = value.encode("utf-8")
    return name + b"\x00" + len(raw).to_bytes(8, "big") + raw


def _secret(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise HitabP1SourceError("selection secret must be exactly 32 bytes")
    return value


def _selection_digest(
    secret: bytes, *, block: str, family: str, row_commitment: str
) -> bytes:
    if (
        block not in BLOCK_SOURCE_SPLIT
        or family not in FAMILIES
        or _HEX64.fullmatch(row_commitment) is None
    ):
        raise HitabP1SourceError("selection HMAC namespace drifted")
    message = (
        SELECTION_ORDER_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"block", block)
        + _frame(b"family", family)
        + _frame(b"row_commitment", row_commitment)
    )
    return hmac.new(_secret(secret), message, hashlib.sha256).digest()


def _work_id(
    secret: bytes, *, block: str, family: str, row_commitment: str
) -> str:
    message = (
        WORK_ID_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"block", block)
        + _frame(b"family", family)
        + _frame(b"row_commitment", row_commitment)
    )
    return "hitab-work-v1-" + hmac.new(
        _secret(secret), message, hashlib.sha256
    ).hexdigest()


def table_commitment(secret: bytes, table_id: str) -> str:
    message = (
        TABLE_COMMITMENT_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"table_id", normalized_text(table_id))
    )
    return hmac.new(_secret(secret), message, hashlib.sha256).hexdigest()


def _transitive_component_registry(
    items: Sequence[EligibleItem],
    *,
    prior_components: Sequence[frozenset[str]],
) -> tuple[
    tuple[frozenset[str], ...],
    Mapping[str, frozenset[str]],
]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        cursor = value
        while parent[cursor] != cursor:
            parent[cursor] = parent[parent[cursor]]
            cursor = parent[cursor]
        root = cursor
        cursor = value
        while parent[cursor] != cursor:
            following = parent[cursor]
            parent[cursor] = root
            cursor = following
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    token_sets: list[frozenset[str]] = []
    for component in prior_components:
        if (
            not isinstance(component, frozenset)
            or not component
            or any(not isinstance(token, str) or not token for token in component)
        ):
            raise HitabP1SourceError("prior component registry drifted")
        token_sets.append(component)
    token_sets.extend(item.component_tokens for item in items)
    for tokens in token_sets:
        ordered = tuple(sorted(tokens))
        find(ordered[0])
        for token in ordered[1:]:
            union(ordered[0], token)
    grouped: dict[str, set[str]] = {}
    for token in parent:
        grouped.setdefault(find(token), set()).add(token)
    registry = tuple(
        sorted(
            (frozenset(tokens) for tokens in grouped.values()),
            key=lambda tokens: tuple(sorted(tokens)),
        )
    )
    by_token: dict[str, frozenset[str]] = {}
    for component in registry:
        for token in component:
            if token in by_token:
                raise HitabP1SourceError("component token appears twice")
            by_token[token] = component
    return registry, by_token


def select_fixed_quota(
    records_by_split: Mapping[str, Sequence[EligibleItem]],
    *,
    secret: bytes,
    blocks: Sequence[str] = INITIAL_BLOCKS,
    quota_per_family: Mapping[str, int] = BLOCK_QUOTA_PER_FAMILY,
    prior_component_tokens: frozenset[str] = frozenset(),
    prior_component_registry: Sequence[frozenset[str]] = (),
) -> SelectionBatch:
    """Select fixed quotas in one HMAC order with component isolation."""

    _secret(secret)
    checked_blocks = tuple(blocks)
    if (
        not checked_blocks
        or len(set(checked_blocks)) != len(checked_blocks)
        or any(block not in BLOCK_SOURCE_SPLIT for block in checked_blocks)
    ):
        raise HitabP1SourceError("selection block set drifted")
    required_splits = {BLOCK_SOURCE_SPLIT[block] for block in checked_blocks}
    if set(records_by_split) != required_splits:
        raise HitabP1SourceError("selection source split set drifted")
    if set(quota_per_family) != set(BLOCK_SOURCE_SPLIT):
        raise HitabP1SourceError("selection quota registry drifted")
    for value in quota_per_family.values():
        if type(value) is not int or value < 1:
            raise HitabP1SourceError("selection quota is invalid")
    all_item_ids: set[tuple[str, str]] = set()
    for split, rows in records_by_split.items():
        for row in rows:
            if not isinstance(row, EligibleItem) or row.candidate.split != split:
                raise HitabP1SourceError("eligible selection row drifted")
            identity = (split, row.candidate.item_id)
            if identity in all_item_ids:
                raise HitabP1SourceError("eligible item identity is duplicated")
            all_item_ids.add(identity)
    all_items = tuple(
        row
        for split in sorted(records_by_split)
        for row in records_by_split[split]
    )
    component_registry, component_by_token = _transitive_component_registry(
        all_items,
        prior_components=prior_component_registry,
    )
    used_component_ids: set[str] = set()
    for token in prior_component_tokens:
        component = component_by_token.get(token)
        if component is None:
            raise HitabP1SourceError(
                "prior used token escaped component registry"
            )
        used_component_ids.add(stable_hash(sorted(component)))
    used = set(prior_component_tokens)
    selected_by_block: dict[str, tuple[SelectedItem, ...]] = {}
    skipped_component = 0
    for block in checked_blocks:
        split = BLOCK_SOURCE_SPLIT[block]
        selected: list[SelectedItem] = []
        for family in FAMILIES:
            ordered = sorted(
                (
                    row
                    for row in records_by_split[split]
                    if row.candidate.family == family
                ),
                key=lambda row: (
                    _selection_digest(
                        secret,
                        block=block,
                        family=family,
                        row_commitment=row.candidate.row_commitment,
                    ),
                    row.candidate.row_commitment,
                ),
            )
            family_selected: list[SelectedItem] = []
            for row in ordered:
                component = component_by_token.get(next(iter(row.component_tokens)))
                if component is None or not row.component_tokens.issubset(component):
                    raise HitabP1SourceError(
                        "eligible row escaped transitive component registry"
                    )
                component_id = stable_hash(sorted(component))
                if component_id in used_component_ids:
                    skipped_component += 1
                    continue
                family_selected.append(
                    SelectedItem(
                        block=block,
                        work_id=_work_id(
                            secret,
                            block=block,
                            family=family,
                            row_commitment=row.candidate.row_commitment,
                        ),
                        item=row,
                    )
                )
                used_component_ids.add(component_id)
                used.update(component)
                if len(family_selected) == quota_per_family[block]:
                    break
            if len(family_selected) != quota_per_family[block]:
                raise HitabP1SourceError(
                    "fixed family quota capacity is insufficient"
                )
            selected.extend(family_selected)
        selected_by_block[block] = tuple(
            sorted(selected, key=lambda row: row.work_id)
        )
    work_ids = [
        row.work_id
        for block in checked_blocks
        for row in selected_by_block[block]
    ]
    if len(work_ids) != len(set(work_ids)):
        raise HitabP1SourceError("opaque work ID collision")
    counts = {
        block: {
            family: sum(
                row.item.candidate.family == family
                for row in selected_by_block[block]
            )
            for family in FAMILIES
        }
        for block in checked_blocks
    }
    safe = self_hashed(
        {
            "block_count": len(checked_blocks),
            "component_collision_skip_count": skipped_component,
            "component_registry_commitment": stable_hash(
                sorted(stable_hash(sorted(component)) for component in component_registry)
            ),
            "component_registry_count": len(component_registry),
            "per_block_family_count": counts,
            "schema": "hitab_p1_private_selection_safe_receipt_v1",
            "selected_item_count": len(work_ids),
            "selection_commitment": stable_hash(sorted(work_ids)),
            "selection_secret_commitment": hashlib.sha256(secret).hexdigest(),
            "status": "fixed_HMAC_quotas_selected",
            "study_id": STUDY_ID,
            "version": VERSION,
        }
    )
    return SelectionBatch(
        selected_by_block=selected_by_block,
        safe_receipt=safe,
        used_component_tokens=frozenset(used),
        component_registry=component_registry,
    )


def _component_registry_commitment(
    components: Sequence[frozenset[str]],
) -> str:
    return stable_hash(
        sorted(stable_hash(sorted(component)) for component in components)
    )


def _persist_component_registry(
    path: Path,
    *,
    selection: SelectionBatch,
    phase: str,
) -> str:
    if phase not in {"initial", "with_M"}:
        raise HitabP1SourceError("component registry phase drifted")
    components = tuple(selection.component_registry)
    used = tuple(sorted(selection.used_component_tokens))
    commitment = _component_registry_commitment(components)
    if (
        commitment
        != selection.safe_receipt.get("component_registry_commitment")
    ):
        raise HitabP1SourceError("component registry safe binding drifted")
    value = self_hashed(
        {
            "component_count": len(components),
            "component_registry_commitment": commitment,
            "components": [sorted(component) for component in components],
            "phase": phase,
            "schema": "hitab_p1_collision_component_registry_v1",
            "study_id": STUDY_ID,
            "used_component_token_count": len(used),
            "used_component_tokens": list(used),
            "used_component_tokens_commitment": stable_hash(list(used)),
            "version": VERSION,
        }
    )
    write_json_exclusive(path, value)
    return str(value["self_sha256"])


def _load_component_registry(
    path: Path,
    *,
    expected_commitment: str,
    expected_phase: str,
) -> tuple[tuple[frozenset[str], ...], frozenset[str]]:
    raw = _read_regular_bytes(
        Path(path),
        maximum_bytes=64_000_000,
        require_mode_0600=True,
        label="collision component registry",
    )
    value = strict_json(raw, label="collision component registry")
    if (
        not isinstance(value, Mapping)
        or raw != canonical_bytes(value, newline=True)
        or set(value) != {
            "component_count",
            "component_registry_commitment",
            "components",
            "phase",
            "schema",
            "self_sha256",
            "study_id",
            "used_component_token_count",
            "used_component_tokens",
            "used_component_tokens_commitment",
            "version",
        }
        or value.get("schema")
        != "hitab_p1_collision_component_registry_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("version") != VERSION
        or value.get("phase") != expected_phase
    ):
        raise HitabP1SourceError("collision component registry envelope drifted")
    verify_self_hash(value)
    raw_components = value.get("components")
    raw_used = value.get("used_component_tokens")
    if not isinstance(raw_components, list) or not isinstance(raw_used, list):
        raise HitabP1SourceError("collision component registry body drifted")
    components: list[frozenset[str]] = []
    for raw_component in raw_components:
        if (
            not isinstance(raw_component, list)
            or not raw_component
            or raw_component != sorted(set(raw_component))
            or any(not isinstance(token, str) or not token for token in raw_component)
        ):
            raise HitabP1SourceError("collision component is not canonical")
        components.append(frozenset(raw_component))
    registry = tuple(
        sorted(components, key=lambda component: tuple(sorted(component)))
    )
    used = tuple(raw_used)
    if (
        len(registry) != len(set(registry))
        or len({token for component in registry for token in component})
        != sum(len(component) for component in registry)
        or value.get("component_count") != len(registry)
        or _component_registry_commitment(registry) != expected_commitment
        or value.get("component_registry_commitment") != expected_commitment
        or used != tuple(sorted(set(used)))
        or any(not isinstance(token, str) or not token for token in used)
        or value.get("used_component_token_count") != len(used)
        or value.get("used_component_tokens_commitment")
        != stable_hash(list(used))
        or not set(used).issubset(
            {token for component in registry for token in component}
        )
    ):
        raise HitabP1SourceError("collision component registry binding drifted")
    return registry, frozenset(used)


def _read_bound_source_payload(
    path: Path,
    *,
    identity: VerifiedFileIdentity,
    require_mode_0600: bool,
    label: str,
) -> bytes:
    raw = _read_regular_bytes(
        path,
        maximum_bytes=identity.size_bytes,
        require_mode_0600=require_mode_0600,
        label=label,
    )
    if (
        len(raw) != identity.size_bytes
        or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(), identity.sha256
        )
        or not hmac.compare_digest(
            git_blob_sha1(raw), identity.git_blob_sha1
        )
    ):
        raise HitabP1SourceError(f"{label} no longer matches source receipt")
    return raw


def run_initial_selection_once(
    *,
    source_paths: Mapping[str, Path],
    verified_sources: VerifiedSourceSet,
    control_root: Path,
    quota_per_family: Mapping[str, int] = BLOCK_QUOTA_PER_FAMILY,
    require_mode_0600: bool = True,
    public_exposure_hashes: Mapping[str, frozenset[str]] = PUBLIC_EXPOSURE_HASHES,
) -> InitialSelectionRun:
    """Consume one secret and select/materialize A_form and A_hold once.

    TRAIN and DEV are decoded only after the attempt marker and 32-byte secret
    are durably created.  TEST is not opened; its already verified raw newline
    count is merely copied into the safe receipt with decode count zero.
    """

    if set(source_paths) != {"TRAIN", "DEV", "TEST", "TABLES"}:
        raise HitabP1SourceError("initial source path registry drifted")
    if set(verified_sources.identities) != {"TRAIN", "DEV", "TEST", "TABLES"}:
        raise HitabP1SourceError("initial verified identity registry drifted")
    root = Path(control_root)
    marker = root / INITIAL_ATTEMPT_FILENAME
    secret_path = root / INITIAL_SECRET_FILENAME
    receipt_path = root / INITIAL_RECEIPT_FILENAME
    failure_path = root / INITIAL_FAILURE_FILENAME
    write_json_exclusive(
        marker,
        self_hashed(
            {
                "schema": "hitab_p1_initial_selection_attempt_v1",
                "source_identity_commitment": (
                    verified_sources.source_identity_commitment
                ),
                "study_id": STUDY_ID,
                "test_json_decode_count": 0,
                "version": VERSION,
            }
        ),
    )
    secret = os.urandom(32)
    if len(secret) != 32:
        raise HitabP1SourceError("OS random selection secret length drifted")
    write_bytes_exclusive(secret_path, secret)
    stage = "read_bound_TRAIN_DEV"
    try:
        train_raw = _read_bound_source_payload(
            Path(source_paths["TRAIN"]),
            identity=verified_sources.identities["TRAIN"],
            require_mode_0600=require_mode_0600,
            label="pinned TRAIN",
        )
        dev_raw = _read_bound_source_payload(
            Path(source_paths["DEV"]),
            identity=verified_sources.identities["DEV"],
            require_mode_0600=require_mode_0600,
            label="pinned DEV",
        )
        # The tables ZIP is not decoded here, but its exact identity is
        # rechecked before requested-member materialization.
        tables_raw = _read_bound_source_payload(
            Path(source_paths["TABLES"]),
            identity=verified_sources.identities["TABLES"],
            require_mode_0600=require_mode_0600,
            label="pinned TABLES",
        )
        stage = "parse_TRAIN_DEV"
        train = parse_sample_jsonl_bytes(
            train_raw,
            split="TRAIN",
            public_exposure_hashes=public_exposure_hashes,
        )
        dev = parse_sample_jsonl_bytes(
            dev_raw,
            split="DEV",
            public_exposure_hashes=public_exposure_hashes,
        )
        stage = "materialize_requested_tables_and_qrels"
        train_eligible, train_materialization_reasons = materialize_candidates(
            train.candidates,
            tables_zip_bytes=tables_raw,
        )
        dev_eligible, dev_materialization_reasons = materialize_candidates(
            dev.candidates,
            tables_zip_bytes=tables_raw,
        )
        stage = "fixed_HMAC_selection"
        selection = select_fixed_quota(
            {"TRAIN": train_eligible, "DEV": dev_eligible},
            secret=secret,
            blocks=INITIAL_BLOCKS,
            quota_per_family=quota_per_family,
        )
        stage = "persist_label_free_views_and_selected_qrel_custody"
        block_views: dict[str, BridgeBlockView] = {}
        block_view_custody_commitments: dict[str, str] = {}
        qrel_custody_commitments: dict[str, str] = {}
        for block in INITIAL_BLOCKS:
            view, qrel_rows = _split_private_materialization(
                selection,
                block=block,
            )
            block_views[block] = view
            block_view_custody_commitments[block] = _persist_block_view(
                root / BLOCK_VIEW_FILENAMES[block],
                view,
            )
            qrel_custody_commitments[block] = _persist_qrel_custody(
                root / QREL_CUSTODY_FILENAMES[block],
                block=block,
                block_view_sha256=view.view_sha256,
                rows=qrel_rows,
            )
        component_registry_self_sha256 = _persist_component_registry(
            root / COMPONENT_REGISTRY_FILENAME,
            selection=selection,
            phase="initial",
        )
        receipt = self_hashed(
            {
                "A_hold_corpus_commitment": stable_hash(
                    sorted(
                        row.item.table.corpus_commitment
                        for row in selection.selected_by_block["A_hold"]
                    )
                ),
                "A_hold_qrel_ordinal_mapping_commitment": stable_hash(
                    sorted(
                        row.item.qrel_ordinal_mapping_commitment
                        for row in selection.selected_by_block["A_hold"]
                    )
                ),
                "A_form_corpus_commitment": stable_hash(
                    sorted(
                        row.item.table.corpus_commitment
                        for row in selection.selected_by_block["A_form"]
                    )
                ),
                "A_form_qrel_ordinal_mapping_commitment": stable_hash(
                    sorted(
                        row.item.qrel_ordinal_mapping_commitment
                        for row in selection.selected_by_block["A_form"]
                    )
                ),
                "block_view_sha256": {
                    block: block_views[block].view_sha256
                    for block in INITIAL_BLOCKS
                },
                "block_view_custody_self_sha256": (
                    block_view_custody_commitments
                ),
                "component_registry_commitment": selection.safe_receipt[
                    "component_registry_commitment"
                ],
                "component_registry_count": selection.safe_receipt[
                    "component_registry_count"
                ],
                "component_registry_self_sha256": (
                    component_registry_self_sha256
                ),
                "materialization_exclusion_reason_counts": {
                    "DEV": dev_materialization_reasons,
                    "TRAIN": train_materialization_reasons,
                },
                "safe_parse_summaries": {
                    "DEV": dev.safe_summary,
                    "TRAIN": train.safe_summary,
                },
                "schema": "hitab_p1_initial_selection_safe_receipt_v1",
                "selected_qrel_custody_self_sha256": (
                    qrel_custody_commitments
                ),
                "selection_commitment": selection.safe_receipt[
                    "selection_commitment"
                ],
                "selection_secret_commitment": hashlib.sha256(secret).hexdigest(),
                "source_identity_commitment": (
                    verified_sources.source_identity_commitment
                ),
                "status": "A_form_and_A_hold_selected_once",
                "study_id": STUDY_ID,
                "test_identity_only": test_identity_only_summary(
                    verified_sources
                ),
                "test_json_decode_count": 0,
                "version": VERSION,
            }
        )
        write_json_exclusive(receipt_path, receipt)
        return InitialSelectionRun(
            block_views=dict(block_views),
            safe_receipt=receipt,
        )
    except Exception as exc:
        failure = self_hashed(
            {
                "failure_stage": stage,
                "retry_replay_resample_parser_family_quota_or_gate_change_count": 0,
                "schema": "hitab_p1_initial_selection_failure_v1",
                "status": "terminal_attempt_and_secret_consumed",
                "study_id": STUDY_ID,
                "test_json_decode_count": 0,
                "version": VERSION,
            }
        )
        try:
            write_json_exclusive(failure_path, failure)
        except Exception:
            pass
        if isinstance(exc, HitabP1SourceError):
            raise
        raise HitabP1SourceError("initial formal selection failed closed") from exc


def _split_private_materialization(
    selection: SelectionBatch,
    *,
    block: str,
) -> tuple[BridgeBlockView, tuple[BridgeQrelRow, ...]]:
    """Privately split one block; callers must persist, not return, qrels."""

    if block not in BLOCK_SOURCE_SPLIT:
        raise HitabP1SourceError("materialization block is invalid")
    selected = selection.selected_by_block.get(block)
    if not isinstance(selected, tuple) or not selected:
        raise HitabP1SourceError("selected materialization block is unavailable")
    view_items: list[BridgeViewItem] = []
    qrel_rows: list[BridgeQrelRow] = []
    for row in selected:
        if not isinstance(row, SelectedItem) or row.block != block:
            raise HitabP1SourceError("selected materialization row drifted")
        table = row.item.table
        corpus_commitment = table.corpus_commitment
        view_items.append(
            BridgeViewItem(
                work_id=row.work_id,
                question=row.item.candidate.question,
                ordered_unit_strings=tuple(
                    unit.serialized for unit in table.units
                ),
                unit_types=tuple(unit.value_type for unit in table.units),
                typed_edges=table.typed_edges,
                corpus_commitment=corpus_commitment,
            )
        )
        qrel_rows.append(
            BridgeQrelRow(
                work_id=row.work_id,
                family=row.item.candidate.family,
                qrel=row.item.qrel,
                corpus_commitment=corpus_commitment,
                qrel_ordinal_mapping_commitment=(
                    row.item.qrel_ordinal_mapping_commitment
                ),
            )
        )
    view_items_tuple = tuple(sorted(view_items, key=lambda row: row.work_id))
    qrel_rows_tuple = tuple(sorted(qrel_rows, key=lambda row: row.work_id))
    view_payload = {
        "block": block,
        "items": [row.private_payload() for row in view_items_tuple],
    }
    return (
        BridgeBlockView(
            block=block,
            items=view_items_tuple,
            view_sha256=stable_hash(view_payload),
        ),
        qrel_rows_tuple,
    )


def _persist_block_view(path: Path, value: BridgeBlockView) -> str:
    if (
        value.block not in BLOCK_VIEW_FILENAMES
        or Path(path).name != BLOCK_VIEW_FILENAMES[value.block]
        or not value.items
    ):
        raise HitabP1SourceError("label-free block view custody drifted")
    payload = {
        "block": value.block,
        "items": [row.private_payload() for row in value.items],
    }
    if value.view_sha256 != stable_hash(payload):
        raise HitabP1SourceError("label-free block view hash drifted")
    custody = self_hashed(
        {
            "block": value.block,
            "item_count": len(value.items),
            "items": payload["items"],
            "schema": "hitab_p1_label_free_block_view_custody_v1",
            "study_id": STUDY_ID,
            "version": VERSION,
            "view_sha256": value.view_sha256,
        }
    )
    write_json_exclusive(path, custody)
    return str(custody["self_sha256"])


def _load_block_view(
    path: Path,
    *,
    expected_block: str,
) -> BridgeBlockView:
    if (
        expected_block not in BLOCK_VIEW_FILENAMES
        or Path(path).name != BLOCK_VIEW_FILENAMES[expected_block]
    ):
        raise HitabP1SourceError("label-free block view path drifted")
    raw = _read_regular_bytes(
        Path(path),
        maximum_bytes=256_000_000,
        require_mode_0600=True,
        label="label-free block view custody",
    )
    value = strict_json(raw, label="label-free block view custody")
    if (
        not isinstance(value, Mapping)
        or raw != canonical_bytes(value, newline=True)
        or set(value) != {
            "block",
            "item_count",
            "items",
            "schema",
            "self_sha256",
            "study_id",
            "version",
            "view_sha256",
        }
        or value.get("schema")
        != "hitab_p1_label_free_block_view_custody_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("version") != VERSION
        or value.get("block") != expected_block
    ):
        raise HitabP1SourceError("label-free block view envelope drifted")
    verify_self_hash(value)
    raw_items = value.get("items")
    view_sha256 = value.get("view_sha256")
    if (
        not isinstance(raw_items, list)
        or not raw_items
        or value.get("item_count") != len(raw_items)
        or not isinstance(view_sha256, str)
        or _HEX64.fullmatch(view_sha256) is None
    ):
        raise HitabP1SourceError("label-free block view header drifted")
    items: list[BridgeViewItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {
            "corpus_commitment",
            "ordered_unit_strings",
            "question",
            "typed_edges",
            "unit_types",
            "work_id",
        }:
            raise HitabP1SourceError("label-free block view item drifted")
        work_id = raw_item.get("work_id")
        question = raw_item.get("question")
        ordered_units = raw_item.get("ordered_unit_strings")
        unit_types = raw_item.get("unit_types")
        raw_edges = raw_item.get("typed_edges")
        corpus = raw_item.get("corpus_commitment")
        if (
            not isinstance(work_id, str)
            or re.fullmatch(r"hitab-work-v1-[0-9a-f]{64}", work_id) is None
            or not isinstance(question, str)
            or not question
            or not isinstance(ordered_units, list)
            or not MIN_CORPUS_UNITS <= len(ordered_units) <= MAX_CORPUS_UNITS
            or any(not isinstance(row, str) or not row for row in ordered_units)
            or not isinstance(unit_types, list)
            or len(unit_types) != len(ordered_units)
            or any(not isinstance(row, str) or not row for row in unit_types)
            or not isinstance(raw_edges, list)
            or not isinstance(corpus, str)
            or _HEX64.fullmatch(corpus) is None
            or corpus != stable_hash(ordered_units)
        ):
            raise HitabP1SourceError("label-free block view item binding drifted")
        edges: list[TableTypedEdge] = []
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, Mapping) or set(raw_edge) != {
                "edge_type",
                "source_ordinal",
                "target_ordinal",
            }:
                raise HitabP1SourceError(
                    "label-free block view edge drifted"
                )
            edge = TableTypedEdge(
                source_ordinal=raw_edge.get("source_ordinal"),  # type: ignore[arg-type]
                target_ordinal=raw_edge.get("target_ordinal"),  # type: ignore[arg-type]
                edge_type=raw_edge.get("edge_type"),  # type: ignore[arg-type]
            )
            if edge.target_ordinal >= len(ordered_units):
                raise HitabP1SourceError(
                    "label-free block view edge escaped corpus"
                )
            edges.append(edge)
        checked_edges = tuple(edges)
        if checked_edges != tuple(sorted(set(checked_edges))):
            raise HitabP1SourceError(
                "label-free block view edges are not canonical"
            )
        items.append(
            BridgeViewItem(
                work_id=work_id,
                question=question,
                ordered_unit_strings=tuple(ordered_units),
                unit_types=tuple(unit_types),
                typed_edges=checked_edges,
                corpus_commitment=corpus,
            )
        )
    checked_items = tuple(items)
    if (
        checked_items
        != tuple(sorted(checked_items, key=lambda row: row.work_id))
        or len({row.work_id for row in checked_items}) != len(checked_items)
    ):
        raise HitabP1SourceError(
            "label-free block view item order drifted"
        )
    payload = {
        "block": expected_block,
        "items": [row.private_payload() for row in checked_items],
    }
    if stable_hash(payload) != view_sha256:
        raise HitabP1SourceError("label-free block view commitment drifted")
    return BridgeBlockView(
        block=expected_block,
        items=checked_items,
        view_sha256=view_sha256,
    )


def _qrel_row_payload(row: BridgeQrelRow) -> dict[str, object]:
    return {
        "corpus_commitment": row.corpus_commitment,
        "family": row.family,
        "proof": row.qrel.payload(),
        "qrel_ordinal_mapping_commitment": (
            row.qrel_ordinal_mapping_commitment
        ),
        "work_id": row.work_id,
    }


def _persist_qrel_custody(
    path: Path,
    *,
    block: str,
    block_view_sha256: str,
    rows: Sequence[BridgeQrelRow],
) -> str:
    if (
        block not in QREL_CUSTODY_FILENAMES
        or _HEX64.fullmatch(block_view_sha256) is None
    ):
        raise HitabP1SourceError("qrel custody block binding is invalid")
    checked = tuple(rows)
    if (
        not checked
        or checked != tuple(sorted(checked, key=lambda row: row.work_id))
        or len({row.work_id for row in checked}) != len(checked)
    ):
        raise HitabP1SourceError("qrel custody rows are not canonical")
    value = self_hashed(
        {
            "block": block,
            "block_view_sha256": block_view_sha256,
            "row_count": len(checked),
            "rows": [_qrel_row_payload(row) for row in checked],
            "schema": "hitab_p1_selected_qrel_custody_v1",
            "study_id": STUDY_ID,
            "version": VERSION,
        }
    )
    write_json_exclusive(path, value)
    return str(value["self_sha256"])


def _proof_from_private_payload(value: object) -> ProofDNF:
    if not isinstance(value, Mapping) or set(value) != {
        "alternatives",
        "corpus_commitment",
        "ordinal_mapping_commitment",
    }:
        raise HitabP1SourceError("private qrel proof payload drifted")
    raw_alternatives = value.get("alternatives")
    if not isinstance(raw_alternatives, list):
        raise HitabP1SourceError("private qrel alternatives drifted")
    try:
        proof = ProofDNF(
            alternatives=tuple(
                tuple(
                    tuple(bucket)
                    for bucket in alternative
                )
                for alternative in raw_alternatives
            ),
            corpus_commitment=str(value.get("corpus_commitment")),
        )
    except (TypeError, dmc_core.HitabDmc1CoreError) as exc:
        raise HitabP1SourceError("private qrel proof is invalid") from exc
    if (
        value.get("ordinal_mapping_commitment")
        != proof.ordinal_mapping_commitment
        or proof.payload() != dict(value)
    ):
        raise HitabP1SourceError("private qrel proof commitment drifted")
    return proof


def _load_qrel_custody(
    path: Path,
    *,
    expected_block: str,
) -> tuple[str, tuple[BridgeQrelRow, ...]]:
    raw = _read_regular_bytes(
        Path(path),
        maximum_bytes=64_000_000,
        require_mode_0600=True,
        label="selected qrel custody",
    )
    value = strict_json(raw, label="selected qrel custody")
    if (
        not isinstance(value, Mapping)
        or raw != canonical_bytes(value, newline=True)
        or value.get("schema") != "hitab_p1_selected_qrel_custody_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("version") != VERSION
        or value.get("block") != expected_block
        or set(value) != {
            "block",
            "block_view_sha256",
            "row_count",
            "rows",
            "schema",
            "self_sha256",
            "study_id",
            "version",
        }
    ):
        raise HitabP1SourceError("selected qrel custody envelope drifted")
    verify_self_hash(value)
    view_sha256 = value.get("block_view_sha256")
    raw_rows = value.get("rows")
    if (
        not isinstance(view_sha256, str)
        or _HEX64.fullmatch(view_sha256) is None
        or not isinstance(raw_rows, list)
        or value.get("row_count") != len(raw_rows)
        or not raw_rows
    ):
        raise HitabP1SourceError("selected qrel custody header drifted")
    rows: list[BridgeQrelRow] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "corpus_commitment",
            "family",
            "proof",
            "qrel_ordinal_mapping_commitment",
            "work_id",
        }:
            raise HitabP1SourceError("selected qrel custody row drifted")
        proof = _proof_from_private_payload(raw_row.get("proof"))
        work_id = raw_row.get("work_id")
        family = raw_row.get("family")
        corpus_commitment = raw_row.get("corpus_commitment")
        ordinal_commitment = raw_row.get(
            "qrel_ordinal_mapping_commitment"
        )
        if (
            not isinstance(work_id, str)
            or re.fullmatch(r"hitab-work-v1-[0-9a-f]{64}", work_id) is None
            or family not in FAMILIES
            or not isinstance(corpus_commitment, str)
            or _HEX64.fullmatch(corpus_commitment) is None
            or proof.corpus_commitment != corpus_commitment
            or ordinal_commitment != proof.ordinal_mapping_commitment
        ):
            raise HitabP1SourceError("selected qrel custody binding drifted")
        rows.append(
            BridgeQrelRow(
                work_id=work_id,
                family=str(family),
                qrel=proof,
                corpus_commitment=corpus_commitment,
                qrel_ordinal_mapping_commitment=proof.ordinal_mapping_commitment,
            )
        )
    checked = tuple(rows)
    if (
        checked != tuple(sorted(checked, key=lambda row: row.work_id))
        or len({row.work_id for row in checked}) != len(checked)
    ):
        raise HitabP1SourceError("selected qrel custody order drifted")
    return view_sha256, checked


def _verify_sealed_action_archive_file(
    path: Path,
    sealed_action_archive: Mapping[str, Any],
) -> str:
    if not isinstance(sealed_action_archive, Mapping):
        raise HitabP1SourceError("sealed action archive is not an object")
    action_sha256 = verify_self_hash(sealed_action_archive)
    expected = canonical_bytes(sealed_action_archive)
    raw = _read_regular_bytes(
        Path(path),
        maximum_bytes=len(expected),
        require_mode_0600=False,
        label="sealed action archive",
    )
    try:
        mode = stat.S_IMODE(Path(path).stat(follow_symlinks=False).st_mode)
    except OSError as exc:
        raise HitabP1SourceError("sealed action archive metadata unavailable") from exc
    if mode != 0o400 or raw != expected:
        raise HitabP1SourceError("sealed action archive file drifted")
    return action_sha256


def _validate_serialized_aform_registry(
    registry: Mapping[str, Any],
    *,
    expected_corpus_commitment: str,
    expected_unit_count: int,
) -> None:
    expected_fields = {
        "a_form_hmac_state_cap",
        "a_form_v0_state_cap",
        "corpus_commitment",
        "exploration_key_commitment",
        "feature_names",
        "ridge_lambda",
        "schema",
        "self_sha256",
        "states",
        "study_id",
        "target",
        "target_scale",
        "top_k",
        "unit_count",
        "v0_weights",
        "view_sha256",
    }
    states = registry.get("states")
    if (
        set(registry) != expected_fields
        or registry.get("schema")
        != "hitab_p1_dmc1_core_v1_sealed_a_form_registry_v1"
        or registry.get("study_id") != STUDY_ID
        or registry.get("corpus_commitment")
        != expected_corpus_commitment
        or registry.get("unit_count") != expected_unit_count
        or registry.get("top_k") != dmc_core.TOP_K
        or registry.get("a_form_hmac_state_cap")
        != dmc_core.A_FORM_HMAC_STATE_CAP
        or registry.get("a_form_v0_state_cap")
        != dmc_core.A_FORM_V0_STATE_CAP
        or registry.get("feature_names") != list(dmc_core.FEATURE_NAMES)
        or registry.get("v0_weights") != list(dmc_core.V0_WEIGHTS)
        or registry.get("target_scale") != dmc_core.TARGET_SCALE
        or registry.get("target")
        != "60_times_exact_DNF_set_utility_marginal"
        or registry.get("ridge_lambda") != {
            "denominator": dmc_core.RIDGE_LAMBDA.denominator,
            "numerator": dmc_core.RIDGE_LAMBDA.numerator,
        }
        or not isinstance(registry.get("view_sha256"), str)
        or _HEX64.fullmatch(str(registry.get("view_sha256"))) is None
        or not isinstance(
            registry.get("exploration_key_commitment"), str
        )
        or _HEX64.fullmatch(
            str(registry.get("exploration_key_commitment"))
        )
        is None
        or not isinstance(states, list)
        or not states
        or verify_self_hash(registry) != registry.get("self_sha256")
    ):
        raise HitabP1SourceError("A_form registry header is incomplete")
    ordered_state_keys: list[tuple[int, tuple[int, ...]]] = []
    class_counts: dict[int, Counter[str]] = {
        depth: Counter() for depth in range(dmc_core.TOP_K)
    }
    for state in states:
        if not isinstance(state, Mapping) or set(state) != {
            "actions",
            "depth",
            "selected_ordinals",
            "state_class",
            "state_sha256",
            "v0_value",
        }:
            raise HitabP1SourceError("A_form sealed state drifted")
        depth = state.get("depth")
        selected = state.get("selected_ordinals")
        state_class = state.get("state_class")
        actions = state.get("actions")
        if (
            type(depth) is not int
            or not 0 <= depth < dmc_core.TOP_K
            or not isinstance(selected, list)
            or any(
                type(ordinal) is not int
                or not 0 <= ordinal < expected_unit_count
                for ordinal in selected
            )
            or selected != sorted(set(selected))
            or len(selected) != depth
            or state_class not in dmc_core.STATE_CLASSES
            or type(state.get("v0_value")) is not int
            or not isinstance(actions, list)
        ):
            raise HitabP1SourceError("A_form sealed state header drifted")
        selected_tuple = tuple(selected)
        expected_state_sha256 = stable_hash(
            {
                "depth": depth,
                "schema": "hitab_p1_dmc1_core_v1_state_identity_v1",
                "selected_ordinals": selected,
            }
        )
        if state.get("state_sha256") != expected_state_sha256:
            raise HitabP1SourceError("A_form sealed state hash drifted")
        expected_candidates = [
            ordinal
            for ordinal in range(expected_unit_count)
            if ordinal not in selected_tuple
        ]
        observed_candidates: list[int] = []
        for action in actions:
            if not isinstance(action, Mapping) or set(action) != {
                "candidate_ordinal",
                "phi",
            }:
                raise HitabP1SourceError("A_form sealed action drifted")
            candidate = action.get("candidate_ordinal")
            phi = action.get("phi")
            if type(candidate) is not int or not isinstance(phi, list):
                raise HitabP1SourceError("A_form sealed action is incomplete")
            try:
                dmc_core.FeatureVector(tuple(phi))
            except (TypeError, dmc_core.HitabDmc1CoreError) as exc:
                raise HitabP1SourceError(
                    "A_form sealed action feature drifted"
                ) from exc
            observed_candidates.append(candidate)
        if observed_candidates != expected_candidates:
            raise HitabP1SourceError(
                "A_form state does not archive every remaining action"
            )
        ordered_state_keys.append((depth, selected_tuple))
        class_counts[depth][str(state_class)] += 1
    if (
        ordered_state_keys != sorted(ordered_state_keys)
        or len(set(ordered_state_keys)) != len(ordered_state_keys)
        or {depth for depth, _selected in ordered_state_keys}
        != set(range(dmc_core.TOP_K))
        or ordered_state_keys[0] != (0, ())
        or class_counts[0] != Counter({dmc_core.TOP_V0: 1})
        or any(
            not 1
            <= class_counts[depth][dmc_core.TOP_V0]
            <= dmc_core.A_FORM_V0_STATE_CAP
            or class_counts[depth][dmc_core.HMAC_EXPLORATION]
            > dmc_core.A_FORM_HMAC_STATE_CAP
            for depth in range(1, dmc_core.TOP_K)
        )
    ):
        raise HitabP1SourceError("A_form registry state coverage drifted")


def _validate_gpu0_cache_release_receipt(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "model_offload_or_reload",
        "physical_gpu",
        "schema",
        "self_sha256",
        "study_id",
        "torch_cuda_empty_cache_called",
    }:
        raise HitabP1SourceError(
            "GPU0 cache release receipt is incomplete"
        )
    if (
        value.get("schema")
        != "hitab_p1_gpu0_unused_cuda_cache_release_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("physical_gpu") != 0
        or value.get("torch_cuda_empty_cache_called") is not True
        or value.get("model_offload_or_reload") is not False
    ):
        raise HitabP1SourceError("GPU0 cache release receipt drifted")
    verify_self_hash(value)


def _validate_action_archive_contract(
    *,
    block: str,
    block_view: BridgeBlockView,
    sealed_action_archive: Mapping[str, Any],
) -> str:
    action_sha256 = verify_self_hash(sealed_action_archive)
    expected_schema = (
        "hitab_p1_formal_controller_v1_A_form_label_free_action_archive_v1"
        if block == "A_form"
        else (
            f"hitab_p1_formal_controller_v1_{block}_"
            "four_arm_action_archive_v1"
        )
    )
    expected_top_level_fields = (
        {
            "block",
            "block_view_sha256",
            "item_count",
            "records",
            "registry_stage_complete",
            "schema",
            "self_sha256",
            "study_id",
        }
        if block == "A_form"
        else {
            "block",
            "block_view_sha256",
            "e1_model_sha256",
            "four_arm_corpus_commitment_exact",
            "gpu0_unused_cuda_cache_release_receipt",
            "hipporag_queue_joined_before_archive",
            "item_count",
            "records",
            "schema",
            "self_sha256",
            "study_id",
        }
    )
    records = sealed_action_archive.get("records")
    if (
        set(sealed_action_archive) != expected_top_level_fields
        or sealed_action_archive.get("schema") != expected_schema
        or sealed_action_archive.get("study_id") != STUDY_ID
        or sealed_action_archive.get("block") != block
        or sealed_action_archive.get("block_view_sha256")
        != block_view.view_sha256
        or not isinstance(records, list)
        or sealed_action_archive.get("item_count") != len(records)
        or len(records) != len(block_view.items)
    ):
        raise HitabP1SourceError("sealed action archive header drifted")
    expected_bindings = {
        row.work_id: row.corpus_commitment for row in block_view.items
    }
    view_by_work_id = {
        row.work_id: row for row in block_view.items
    }
    expected_hippo_gpu = {
        row.work_id: index % 2
        for index, row in enumerate(block_view.items)
    }
    observed_bindings: dict[str, str] = {}
    if block == "A_form":
        if sealed_action_archive.get("registry_stage_complete") is not True:
            raise HitabP1SourceError("A_form registry stage is incomplete")
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "corpus_commitment",
                "registry",
                "tensor_sha256",
                "work_id",
            }:
                raise HitabP1SourceError("A_form action record drifted")
            work_id = record.get("work_id")
            corpus = record.get("corpus_commitment")
            registry = record.get("registry")
            tensor = record.get("tensor_sha256")
            if (
                not isinstance(work_id, str)
                or not isinstance(corpus, str)
                or not isinstance(registry, Mapping)
                or not isinstance(tensor, str)
                or _HEX64.fullmatch(tensor) is None
                or work_id not in view_by_work_id
            ):
                raise HitabP1SourceError("A_form registry record is incomplete")
            _validate_serialized_aform_registry(
                registry,
                expected_corpus_commitment=corpus,
                expected_unit_count=len(
                    view_by_work_id[work_id].ordered_unit_strings
                ),
            )
            if work_id in observed_bindings:
                raise HitabP1SourceError("A_form work ID is duplicated")
            observed_bindings[work_id] = corpus
    else:
        _validate_gpu0_cache_release_receipt(
            sealed_action_archive.get(
                "gpu0_unused_cuda_cache_release_receipt"
            )
        )
        if (
            sealed_action_archive.get("four_arm_corpus_commitment_exact")
            is not True
            or sealed_action_archive.get(
                "hipporag_queue_joined_before_archive"
            )
            is not True
            or not isinstance(
                sealed_action_archive.get("e1_model_sha256"), str
            )
            or _HEX64.fullmatch(
                str(sealed_action_archive.get("e1_model_sha256"))
            )
            is None
        ):
            raise HitabP1SourceError("four-arm action stage is incomplete")
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "arms",
                "tensor_sha256",
                "work_id",
            }:
                raise HitabP1SourceError("four-arm action record drifted")
            work_id = record.get("work_id")
            tensor = record.get("tensor_sha256")
            arms = record.get("arms")
            if (
                not isinstance(work_id, str)
                or not isinstance(tensor, str)
                or _HEX64.fullmatch(tensor) is None
                or not isinstance(arms, Mapping)
                or set(arms) != {"RAW", "HippoRAG", "E0", "E1"}
                or work_id not in view_by_work_id
            ):
                raise HitabP1SourceError("four-arm action record is incomplete")
            unit_count = len(
                view_by_work_id[work_id].ordered_unit_strings
            )
            corpora: set[str] = set()
            for arm_name, arm in arms.items():
                if not isinstance(arm, Mapping):
                    raise HitabP1SourceError("four-arm payload drifted")
                corpus = arm.get("corpus_commitment")
                top5 = arm.get("top5_ordinals")
                if (
                    not isinstance(corpus, str)
                    or _HEX64.fullmatch(corpus) is None
                    or not isinstance(top5, list)
                    or len(top5) != 5
                    or any(
                        type(row) is not int
                        or not 0 <= row < unit_count
                        for row in top5
                    )
                    or len(set(top5)) != 5
                ):
                    raise HitabP1SourceError("four-arm output is incomplete")
                corpora.add(corpus)
                if arm_name == "HippoRAG":
                    for field in (
                        "complete_rank_sha256",
                        "input_sha256",
                        "output_sha256",
                    ):
                        value = arm.get(field)
                        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                            raise HitabP1SourceError(
                                "HippoRAG action receipt is incomplete"
                            )
                    if (
                        type(arm.get("physical_gpu")) is not int
                        or arm.get("physical_gpu")
                        != expected_hippo_gpu[work_id]
                    ):
                        raise HitabP1SourceError(
                            "HippoRAG physical lane receipt drifted"
                        )
            if len(corpora) != 1:
                raise HitabP1SourceError(
                    "four-arm per-item corpus binding drifted"
                )
            if work_id in observed_bindings:
                raise HitabP1SourceError("four-arm work ID is duplicated")
            observed_bindings[work_id] = next(iter(corpora))
    if observed_bindings != expected_bindings:
        raise HitabP1SourceError(
            "sealed action work set or corpus binding drifted"
        )
    return action_sha256


def release_qrels_after_action_seal(
    *,
    block: str,
    qrel_custody_path: Path,
    sealed_action_archive: Mapping[str, Any],
) -> BridgeQrelPack:
    """Release one selected qrel pack only after its complete action seal."""

    if (
        block not in QREL_CUSTODY_FILENAMES
        or Path(qrel_custody_path).name != QREL_CUSTODY_FILENAMES[block]
    ):
        raise HitabP1SourceError("qrel release custody path drifted")
    verify_self_hash(sealed_action_archive)
    block_view = _load_block_view(
        Path(qrel_custody_path).parent / BLOCK_VIEW_FILENAMES[block],
        expected_block=block,
    )
    action_sha256 = _validate_action_archive_contract(
        block=block,
        block_view=block_view,
        sealed_action_archive=sealed_action_archive,
    )
    marker_path = (
        Path(qrel_custody_path).parent
        / QREL_RELEASE_MARKER_FILENAMES[block]
    )
    write_json_exclusive(
        marker_path,
        self_hashed(
            {
                "action_archive_sha256": action_sha256,
                "block": block,
                "schema": "hitab_p1_qrel_release_attempt_v1",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        ),
    )
    view_sha256, rows = _load_qrel_custody(
        qrel_custody_path,
        expected_block=block,
    )
    view_bindings = {
        row.work_id: row.corpus_commitment for row in block_view.items
    }
    qrel_bindings = {
        row.work_id: row.corpus_commitment for row in rows
    }
    if (
        view_sha256 != block_view.view_sha256
        or qrel_bindings != view_bindings
    ):
        raise HitabP1SourceError(
            "late qrel custody escaped the sealed label-free work set"
        )
    payload = {
        "action_archive_sha256": action_sha256,
        "block": block,
        "rows": [_qrel_row_payload(row) for row in rows],
    }
    return BridgeQrelPack(
        block=block,
        action_archive_sha256=action_sha256,
        rows=rows,
        pack_sha256=stable_hash(payload),
    )


def validate_promotion_authorization(
    authorization: Mapping[str, Any],
    *,
    source_identity_commitment: str,
    initial_selection_commitment: str,
) -> str:
    if (
        not isinstance(authorization, Mapping)
        or _HEX64.fullmatch(source_identity_commitment) is None
        or _HEX64.fullmatch(initial_selection_commitment) is None
    ):
        raise HitabP1SourceError("promotion authorization binding is invalid")
    observed = verify_self_hash(authorization)
    if (
        authorization.get("schema")
        != "hitab_p1_test_first_decode_authorization_v1"
        or authorization.get("study_id") != STUDY_ID
        or authorization.get("status") != "A_hold_E1_promoted"
        or authorization.get("comparison") != "E1_minus_E0"
        or authorization.get("aggregate_exact_utility_net_strictly_positive")
        is not True
        or authorization.get(
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth"
        )
        is not True
        or authorization.get("source_identity_commitment")
        != source_identity_commitment
        or authorization.get("initial_selection_commitment")
        != initial_selection_commitment
    ):
        raise HitabP1SourceError("promotion authorization did not authorize TEST")
    return observed


def _validate_initial_run_binding(
    initial_run: InitialSelectionRun,
    verified_sources: VerifiedSourceSet,
) -> None:
    if (
        not isinstance(initial_run, InitialSelectionRun)
        or set(initial_run.block_views) != set(INITIAL_BLOCKS)
        or set(verified_sources.identities)
        != {"TRAIN", "DEV", "TEST", "TABLES"}
    ):
        raise HitabP1SourceError("initial selection boundary drifted")
    verify_self_hash(initial_run.safe_receipt)
    view_hashes = initial_run.safe_receipt.get("block_view_sha256")
    if (
        initial_run.safe_receipt.get("source_identity_commitment")
        != verified_sources.source_identity_commitment
        or not isinstance(view_hashes, Mapping)
        or set(view_hashes) != set(INITIAL_BLOCKS)
        or any(
            view_hashes.get(block)
            != initial_run.block_views[block].view_sha256
            for block in INITIAL_BLOCKS
        )
    ):
        raise HitabP1SourceError("initial selection safe binding drifted")


def decode_and_select_test_once(
    *,
    test_path: Path,
    tables_zip_path: Path,
    verified_sources: VerifiedSourceSet,
    initial_run: InitialSelectionRun,
    authorization: Mapping[str, Any],
    control_root: Path,
    quota_per_family: Mapping[str, int] = BLOCK_QUOTA_PER_FAMILY,
    require_mode_0600: bool = True,
    public_exposure_hashes: Mapping[str, frozenset[str]] = PUBLIC_EXPOSURE_HASHES,
) -> MSelectionRun:
    """Consume TEST's first-decode marker, decode once, and select M once."""

    _validate_initial_run_binding(initial_run, verified_sources)
    initial_commitment = initial_run.safe_receipt.get("selection_commitment")
    secret_commitment = initial_run.safe_receipt.get(
        "selection_secret_commitment"
    )
    component_commitment = initial_run.safe_receipt.get(
        "component_registry_commitment"
    )
    if (
        not isinstance(initial_commitment, str)
        or _HEX64.fullmatch(initial_commitment) is None
        or not isinstance(secret_commitment, str)
        or _HEX64.fullmatch(secret_commitment) is None
        or not isinstance(component_commitment, str)
        or _HEX64.fullmatch(component_commitment) is None
    ):
        raise HitabP1SourceError("initial selection commitment is missing")
    authorization_sha256 = validate_promotion_authorization(
        authorization,
        source_identity_commitment=verified_sources.source_identity_commitment,
        initial_selection_commitment=initial_commitment,
    )
    root = Path(control_root)
    write_json_exclusive(
        root / TEST_DECODE_ATTEMPT_FILENAME,
        self_hashed(
            {
                "authorization_self_sha256": authorization_sha256,
                "schema": "hitab_p1_test_first_decode_attempt_v1",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        ),
    )
    stage = "load_persisted_initial_secret"
    try:
        secret = _read_regular_bytes(
            root / INITIAL_SECRET_FILENAME,
            maximum_bytes=32,
            require_mode_0600=True,
            label="persisted initial selection secret",
        )
        if (
            len(secret) != 32
            or not hmac.compare_digest(
                hashlib.sha256(secret).hexdigest(),
                secret_commitment,
            )
        ):
            raise HitabP1SourceError(
                "persisted initial selection secret commitment drifted"
            )
        stage = "load_initial_collision_component_registry"
        prior_registry, prior_used = _load_component_registry(
            root / COMPONENT_REGISTRY_FILENAME,
            expected_commitment=component_commitment,
            expected_phase="initial",
        )
        stage = "revalidate_bound_TEST_and_TABLES_before_decode"
        test_raw = _read_bound_source_payload(
            Path(test_path),
            identity=verified_sources.identities["TEST"],
            require_mode_0600=require_mode_0600,
            label="pinned TEST",
        )
        tables_raw = _read_bound_source_payload(
            Path(tables_zip_path),
            identity=verified_sources.identities["TABLES"],
            require_mode_0600=require_mode_0600,
            label="pinned TABLES",
        )
        stage = "first_TEST_JSON_decode"
        parsed = parse_sample_jsonl_bytes(
            test_raw,
            split="TEST",
            allow_test_decode=True,
            public_exposure_hashes=public_exposure_hashes,
        )
        stage = "materialize_requested_M_tables_and_qrels"
        eligible, materialization_reasons = materialize_candidates(
            parsed.candidates,
            tables_zip_bytes=tables_raw,
        )
        stage = "fixed_HMAC_M_selection"
        selected = select_fixed_quota(
            {"TEST": eligible},
            secret=secret,
            blocks=("M_search",),
            quota_per_family=quota_per_family,
            prior_component_tokens=prior_used,
            prior_component_registry=prior_registry,
        )
        stage = "persist_M_label_free_view_and_selected_qrel_custody"
        view, qrel_rows = _split_private_materialization(
            selected,
            block="M_search",
        )
        block_view_custody_self_sha256 = _persist_block_view(
            root / BLOCK_VIEW_FILENAMES["M_search"],
            view,
        )
        qrel_custody_self_sha256 = _persist_qrel_custody(
            root / QREL_CUSTODY_FILENAMES["M_search"],
            block="M_search",
            block_view_sha256=view.view_sha256,
            rows=qrel_rows,
        )
        component_registry_self_sha256 = _persist_component_registry(
            root / M_COMPONENT_REGISTRY_FILENAME,
            selection=selected,
            phase="with_M",
        )
        receipt = self_hashed(
            {
                "authorization_self_sha256": authorization_sha256,
                "M_block_view_sha256": view.view_sha256,
                "M_block_view_custody_self_sha256": (
                    block_view_custody_self_sha256
                ),
                "M_component_registry_commitment": selected.safe_receipt[
                    "component_registry_commitment"
                ],
                "M_component_registry_self_sha256": (
                    component_registry_self_sha256
                ),
                "M_corpus_commitment": stable_hash(
                    sorted(
                        row.item.table.corpus_commitment
                        for row in selected.selected_by_block["M_search"]
                    )
                ),
                "M_qrel_custody_self_sha256": (
                    qrel_custody_self_sha256
                ),
                "M_qrel_ordinal_mapping_commitment": stable_hash(
                    sorted(
                        row.item.qrel_ordinal_mapping_commitment
                        for row in selected.selected_by_block["M_search"]
                    )
                ),
                "materialization_exclusion_reason_counts": (
                    materialization_reasons
                ),
                "M_selection_commitment": selected.safe_receipt[
                    "selection_commitment"
                ],
                "safe_parse_summary": parsed.safe_summary,
                "schema": "hitab_p1_test_selection_safe_receipt_v1",
                "status": "TEST_decoded_once_and_M_selected_once",
                "study_id": STUDY_ID,
                "test_json_decode_count": parsed.safe_summary[
                    "json_decode_count"
                ],
                "version": VERSION,
            }
        )
        write_json_exclusive(
            root / TEST_SELECTION_RECEIPT_FILENAME,
            receipt,
        )
        return MSelectionRun(block_view=view, safe_receipt=receipt)
    except Exception as exc:
        failure = self_hashed(
            {
                "failure_stage": stage,
                "retry_replay_resample_parser_family_quota_or_gate_change_count": 0,
                "schema": "hitab_p1_test_selection_failure_v1",
                "status": "terminal_TEST_decode_attempt_consumed",
                "study_id": STUDY_ID,
                "version": VERSION,
            }
        )
        try:
            write_json_exclusive(
                root / TEST_SELECTION_FAILURE_FILENAME,
                failure,
            )
        except Exception:
            pass
        if isinstance(exc, HitabP1SourceError):
            raise
        raise HitabP1SourceError("TEST selection failed closed") from exc


def _bridge_block_to_formal(
    value: BridgeBlockView,
) -> Any:
    """Convert the private bridge type into the source-free controller type."""

    from assumption_agent.benchmarks import (
        hitab_p1_formal_controller_v1 as formal,
    )
    from assumption_agent.benchmarks import hitab_p1_runtime_v1 as runtime

    items = []
    for row in value.items:
        runtime_item = runtime.RuntimeItem(
            question=row.question,
            ordered_unit_strings=row.ordered_unit_strings,
            corpus_commitment=row.corpus_commitment,
            unit_types=row.unit_types,
            typed_edges=tuple(
                dmc_core.TypedEdge(
                    source_ordinal=edge.source_ordinal,
                    target_ordinal=edge.target_ordinal,
                    edge_type=edge.edge_type,
                )
                for edge in row.typed_edges
            ),
        )
        items.append(
            formal.FormalItemView(
                work_id=row.work_id,
                runtime_item=runtime_item,
            )
        )
    result = formal.BlockView.create(value.block, items)
    if result.view_sha256 != value.view_sha256:
        raise HitabP1SourceError(
            "source-free formal block adapter changed the view"
        )
    return result


def _bridge_qrels_to_formal(
    value: BridgeQrelPack,
) -> Any:
    from assumption_agent.benchmarks import (
        hitab_p1_formal_controller_v1 as formal,
    )

    rows = tuple(
        formal.QrelRow(
            work_id=row.work_id,
            family=row.family,
            proof=row.qrel,
            corpus_commitment=row.corpus_commitment,
            qrel_ordinal_mapping_commitment=(
                row.qrel_ordinal_mapping_commitment
            ),
        )
        for row in value.rows
    )
    result = formal.QrelPack.create(
        block=value.block,
        action_archive_sha256=value.action_archive_sha256,
        rows=rows,
    )
    if result.pack_sha256 != value.pack_sha256:
        raise HitabP1SourceError(
            "late qrel formal adapter changed the pack"
        )
    return result


class ProductionFormalAcquisitionBoundary:
    """One-shot production adapter implementing the formal controller boundary."""

    def __init__(
        self,
        *,
        source_paths: Mapping[str, Path],
        verified_sources: VerifiedSourceSet,
        control_root: Path,
        formal_work_root: Path,
        initial_run: InitialSelectionRun,
        quota_per_family: Mapping[str, int] = BLOCK_QUOTA_PER_FAMILY,
        require_mode_0600: bool = True,
        public_exposure_hashes: Mapping[
            str, frozenset[str]
        ] = PUBLIC_EXPOSURE_HASHES,
    ) -> None:
        if (
            set(source_paths) != {"TRAIN", "DEV", "TEST", "TABLES"}
        ):
            raise HitabP1SourceError(
                "production acquisition boundary inputs drifted"
            )
        _validate_initial_run_binding(initial_run, verified_sources)
        self._source_paths = {
            key: Path(value) for key, value in source_paths.items()
        }
        self._verified_sources = verified_sources
        self._control_root = Path(control_root)
        self._formal_work_root = Path(formal_work_root)
        self._initial_run = initial_run
        self._quota_per_family = dict(quota_per_family)
        self._require_mode_0600 = require_mode_0600
        self._public_exposure_hashes = dict(public_exposure_hashes)
        self._claimed = False
        self._loaded_blocks: set[str] = set()
        self._m_run: MSelectionRun | None = None

    def claim_formal_attempt(self, formal_marker_sha256: str) -> Any:
        from assumption_agent.benchmarks import (
            hitab_p1_formal_controller_v1 as formal,
        )

        if self._claimed or _HEX64.fullmatch(formal_marker_sha256) is None:
            raise HitabP1SourceError("formal acquisition claim is invalid")
        marker_path = self._formal_work_root / formal.FORMAL_MARKER_FILENAME
        raw = _read_regular_bytes(
            marker_path,
            maximum_bytes=1_000_000,
            require_mode_0600=False,
            label="formal controller marker",
        )
        marker = strict_json(raw, label="formal controller marker")
        try:
            marker_mode = stat.S_IMODE(
                marker_path.stat(follow_symlinks=False).st_mode
            )
        except OSError as exc:
            raise HitabP1SourceError(
                "formal controller marker metadata unavailable"
            ) from exc
        if (
            not isinstance(marker, Mapping)
            or raw != canonical_bytes(marker)
            or marker_mode != 0o400
            or marker.get("schema")
            != "hitab_p1_formal_controller_v1_one_shot_marker_v1"
            or marker.get("study_id") != STUDY_ID
            or verify_self_hash(marker) != formal_marker_sha256
        ):
            raise HitabP1SourceError("formal controller marker drifted")
        selection_commitment = self._initial_run.safe_receipt.get(
            "selection_commitment"
        )
        if (
            not isinstance(selection_commitment, str)
            or _HEX64.fullmatch(selection_commitment) is None
        ):
            raise HitabP1SourceError(
                "initial selection commitment is invalid"
            )
        claim = formal.AcquisitionClaim.create(
            source_identity_commitment=(
                self._verified_sources.source_identity_commitment
            ),
            initial_selection_commitment=selection_commitment,
        )
        write_json_exclusive(
            self._control_root / FORMAL_CLAIM_FILENAME,
            self_hashed(
                {
                    "acquisition_claim_sha256": claim.claim_sha256,
                    "formal_marker_sha256": formal_marker_sha256,
                    "schema": "hitab_p1_formal_acquisition_claim_v1",
                    "study_id": STUDY_ID,
                    "version": VERSION,
                }
            ),
        )
        self._claimed = True
        return claim

    def load_label_free_block(
        self,
        block: str,
        authorization: Mapping[str, object] | None = None,
    ) -> Any:
        from assumption_agent.benchmarks import (
            hitab_p1_formal_controller_v1 as formal,
        )

        if not self._claimed or block in self._loaded_blocks:
            raise HitabP1SourceError(
                "label-free block load is unauthorized or replayed"
            )
        if block in INITIAL_BLOCKS:
            if authorization is not None:
                raise HitabP1SourceError(
                    "initial block unexpectedly received promotion authority"
                )
            view = _load_block_view(
                self._control_root / BLOCK_VIEW_FILENAMES[block],
                expected_block=block,
            )
            expected = self._initial_run.block_views[block]
            if view.view_sha256 != expected.view_sha256:
                raise HitabP1SourceError(
                    "persisted initial label-free view drifted"
                )
        elif block == "M_search":
            if not isinstance(authorization, Mapping):
                raise HitabP1SourceError(
                    "M_search requires promotion authorization"
                )
            promotion_path = (
                self._formal_work_root
                / formal.PROMOTION_AUTHORIZATION_FILENAME
            )
            _verify_sealed_action_archive_file(
                promotion_path,
                authorization,
            )
            self._m_run = decode_and_select_test_once(
                test_path=self._source_paths["TEST"],
                tables_zip_path=self._source_paths["TABLES"],
                verified_sources=self._verified_sources,
                initial_run=self._initial_run,
                authorization=authorization,
                control_root=self._control_root,
                quota_per_family=self._quota_per_family,
                require_mode_0600=self._require_mode_0600,
                public_exposure_hashes=self._public_exposure_hashes,
            )
            view = _load_block_view(
                self._control_root / BLOCK_VIEW_FILENAMES["M_search"],
                expected_block="M_search",
            )
            if view.view_sha256 != self._m_run.block_view.view_sha256:
                raise HitabP1SourceError(
                    "persisted M_search label-free view drifted"
                )
        else:
            raise HitabP1SourceError("label-free block name drifted")
        result = _bridge_block_to_formal(view)
        self._loaded_blocks.add(block)
        return result

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> Any:
        if (
            not self._claimed
            or block not in self._loaded_blocks
            or block not in QREL_CUSTODY_FILENAMES
            or Path(custody_path)
            != self._formal_work_root / f"{block}.actions.private.json"
        ):
            raise HitabP1SourceError(
                "qrel release action custody binding drifted"
            )
        action_sha256 = _verify_sealed_action_archive_file(
            Path(custody_path),
            sealed_action_archive,
        )
        pack = release_qrels_after_action_seal(
            block=block,
            qrel_custody_path=(
                self._control_root / QREL_CUSTODY_FILENAMES[block]
            ),
            sealed_action_archive=sealed_action_archive,
        )
        if pack.action_archive_sha256 != action_sha256:
            raise HitabP1SourceError(
                "released qrels changed the action archive binding"
            )
        return _bridge_qrels_to_formal(pack)


def _load_frozen_download_receipt(
    path: Path,
    *,
    expected_self_sha256: str,
    expected_source_identity_commitment: str,
) -> Mapping[str, Any]:
    if (
        Path(path).name != DOWNLOAD_RECEIPT_FILENAME
        or _HEX64.fullmatch(expected_self_sha256) is None
        or _HEX64.fullmatch(expected_source_identity_commitment) is None
    ):
        raise HitabP1SourceError(
            "frozen source download receipt binding is invalid"
        )
    raw = _read_regular_bytes(
        Path(path),
        maximum_bytes=64_000_000,
        require_mode_0600=True,
        label="frozen source download receipt",
    )
    value = strict_json(raw, label="frozen source download receipt")
    expected_fields = {
        "file_count",
        "files",
        "json_decode_count",
        "network_attempt_count",
        "parallel_transport_count",
        "retry_resume_range_mirror_or_provider_switch_count",
        "schema",
        "self_sha256",
        "source_identity_commitment",
        "status",
        "study_id",
        "test_json_decode_count",
        "version",
    }
    if (
        not isinstance(value, Mapping)
        or raw != canonical_bytes(value, newline=True)
        or set(value) != expected_fields
        or value.get("schema") != "hitab_p1_source_download_receipt_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("version") != VERSION
        or value.get("status") != "four_exact_sources_acquired_once"
        or value.get("file_count") != 4
        or value.get("network_attempt_count") != 4
        or value.get("parallel_transport_count") != 4
        or value.get("json_decode_count") != 0
        or value.get("test_json_decode_count") != 0
        or value.get(
            "retry_resume_range_mirror_or_provider_switch_count"
        )
        != 0
        or verify_self_hash(value) != expected_self_sha256
        or value.get("source_identity_commitment")
        != expected_source_identity_commitment
    ):
        raise HitabP1SourceError(
            "frozen source download receipt policy drifted"
        )
    files = value.get("files")
    if (
        not isinstance(files, Mapping)
        or set(files) != {"TRAIN", "DEV", "TEST", "TABLES"}
        or stable_hash(files) != expected_source_identity_commitment
    ):
        raise HitabP1SourceError(
            "frozen source download identity registry drifted"
        )
    for key in ("TRAIN", "DEV", "TEST", "TABLES"):
        row = files.get(key)
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "git_blob_sha1",
                "raw_newline_count",
                "sha256",
                "size_bytes",
            }
            or type(row.get("size_bytes")) is not int
            or int(row.get("size_bytes")) < 1
            or not isinstance(row.get("sha256"), str)
            or _HEX64.fullmatch(str(row.get("sha256"))) is None
            or not isinstance(row.get("git_blob_sha1"), str)
            or _HEX40.fullmatch(str(row.get("git_blob_sha1"))) is None
            or (
                key == "TABLES"
                and row.get("raw_newline_count") is not None
            )
            or (
                key != "TABLES"
                and (
                    type(row.get("raw_newline_count")) is not int
                    or int(row.get("raw_newline_count")) < 1
                )
            )
        ):
            raise HitabP1SourceError(
                f"frozen source download row {key} drifted"
            )
    return value


def build_production_boundary_from_execution(
    execution: object,
) -> ProductionFormalAcquisitionBoundary:
    """Build the sole frozen production boundary without downloading or TEST decode."""

    from replication_runtime.hitab_p1_formal_v1 import runner

    if not isinstance(execution, runner.FrozenExecution):
        raise HitabP1SourceError(
            "production execution object type drifted"
        )
    if (
        execution.acquisition_factory_module
        != "assumption_agent.benchmarks.hitab_p1_source_acquisition_v1"
        or execution.acquisition_factory_attribute
        != "build_production_boundary_from_execution"
        or execution.acquisition_factory_file_label
        != "hitab_source_acquisition"
        or set(execution.source_paths)
        != {"TRAIN", "DEV", "TEST", "TABLES"}
        or set(execution.source_sha256s)
        != {"TRAIN", "DEV", "TEST", "TABLES"}
    ):
        raise HitabP1SourceError(
            "production execution acquisition binding drifted"
        )
    implementation_files = execution.implementation.files
    implementation_hashes = execution.implementation.file_sha256s
    required_files = {
        "hitab_source_acquisition",
        "hitab_source_custody",
        "hitab_study_design",
    }
    if (
        not isinstance(implementation_files, Mapping)
        or not required_files.issubset(implementation_files)
        or not isinstance(implementation_hashes, Mapping)
        or "hitab_source_acquisition" not in implementation_hashes
    ):
        raise HitabP1SourceError(
            "production implementation source closure drifted"
        )
    module_path = Path(implementation_files["hitab_source_acquisition"])
    current_path = Path(__file__).resolve()
    if (
        module_path.is_symlink()
        or not module_path.is_file()
        or module_path.resolve() != current_path
    ):
        raise HitabP1SourceError(
            "production source acquisition module path drifted"
        )
    module_raw = _read_regular_bytes(
        module_path,
        maximum_bytes=module_path.stat().st_size,
        require_mode_0600=False,
        label="frozen source acquisition module",
    )
    if hashlib.sha256(module_raw).hexdigest() != implementation_hashes.get(
        "hitab_source_acquisition"
    ):
        raise HitabP1SourceError(
            "production source acquisition module hash drifted"
        )
    verify_frozen_bindings(
        custody_path=Path(implementation_files["hitab_source_custody"]),
        design_path=Path(implementation_files["hitab_study_design"]),
    )
    receipt_path = Path(execution.source_receipt_path)
    control_root = receipt_path.parent
    formal_root = Path(execution.formal_work_root)
    if (
        not receipt_path.is_absolute()
        or control_root.is_symlink()
        or not control_root.is_dir()
        or control_root.resolve() != control_root
        or stat.S_IMODE(control_root.stat().st_mode) != 0o700
        or not formal_root.is_absolute()
        or formal_root.is_symlink()
        or not formal_root.is_dir()
        or formal_root.resolve() != formal_root
        or stat.S_IMODE(formal_root.stat().st_mode) != 0o700
        or control_root == formal_root
        or control_root in formal_root.parents
        or formal_root in control_root.parents
    ):
        raise HitabP1SourceError(
            "production control or formal root drifted"
        )
    download_receipt = _load_frozen_download_receipt(
        receipt_path,
        expected_self_sha256=execution.source_receipt_self_sha256,
        expected_source_identity_commitment=(
            execution.source_identity_commitment
        ),
    )
    receipt_files = download_receipt["files"]
    for key in ("TRAIN", "DEV", "TEST", "TABLES"):
        if (
            not isinstance(execution.source_paths[key], Path)
            or not execution.source_paths[key].is_absolute()
            or execution.source_paths[key].is_symlink()
            or not execution.source_paths[key].is_file()
            or receipt_files[key].get("sha256")
            != execution.source_sha256s[key]
        ):
            raise HitabP1SourceError(
                f"production source path {key} drifted"
            )
    verified = verify_source_set_once(
        execution.source_paths,
        expected_sha256_by_key=execution.source_sha256s,
        control_root=control_root,
    )
    if (
        verified.source_identity_commitment
        != execution.source_identity_commitment
        or {
            key: verified.identities[key].safe_payload()
            for key in ("TRAIN", "DEV", "TEST", "TABLES")
        }
        != dict(receipt_files)
    ):
        raise HitabP1SourceError(
            "production reverified source receipt drifted"
        )
    initial = run_initial_selection_once(
        source_paths=execution.source_paths,
        verified_sources=verified,
        control_root=control_root,
    )
    if (
        initial.safe_receipt.get("test_json_decode_count") != 0
        or (
            initial.safe_receipt.get("test_identity_only", {})
            .get("json_decode_count")
            != 0
        )
        or (control_root / TEST_DECODE_ATTEMPT_FILENAME).exists()
    ):
        raise HitabP1SourceError(
            "production factory decoded TEST before promotion"
        )
    return ProductionFormalAcquisitionBoundary(
        source_paths=execution.source_paths,
        verified_sources=verified,
        control_root=control_root,
        formal_work_root=formal_root,
        initial_run=initial,
    )


__all__ = [
    "ACCEPTED_TABLE_SOURCES",
    "BLOCK_QUOTA_PER_FAMILY",
    "BLOCK_SOURCE_SPLIT",
    "BridgeBlockView",
    "BridgeQrelPack",
    "BridgeQrelRow",
    "BridgeViewItem",
    "DownloadedSourceSet",
    "EligibleItem",
    "FAMILIES",
    "FAMILY_BY_AGGREGATION_TOKEN",
    "FORMAL_SOURCE_CONTRACTS",
    "HitabP1RowIneligible",
    "HitabP1SourceError",
    "InitialSelectionRun",
    "MSelectionRun",
    "ProofDNF",
    "ProductionFormalAcquisitionBoundary",
    "PUBLIC_EXPOSURE_HASHES",
    "RequestedTableNotFound",
    "SampleCandidate",
    "SelectionBatch",
    "SourceFileContract",
    "TableTypedEdge",
    "TableView",
    "VerifiedSourceSet",
    "build_production_boundary_from_execution",
    "build_coordinate_qrel_dnf",
    "canonical_typed_literal",
    "decode_and_select_test_once",
    "download_source_set_once",
    "family_from_aggregation",
    "git_blob_sha1",
    "materialize_candidates",
    "normalized_text_sha256",
    "parse_hmt_table",
    "parse_sample_jsonl_bytes",
    "parse_sample_jsonl_path",
    "parse_sample_row",
    "read_requested_table_from_zip",
    "release_qrels_after_action_seal",
    "run_initial_selection_once",
    "select_fixed_quota",
    "self_hashed",
    "stable_hash",
    "table_commitment",
    "test_identity_only_summary",
    "validate_promotion_authorization",
    "verify_frozen_bindings",
    "verify_self_hash",
    "verify_source_set_once",
    "write_bytes_exclusive",
    "write_json_exclusive",
]
