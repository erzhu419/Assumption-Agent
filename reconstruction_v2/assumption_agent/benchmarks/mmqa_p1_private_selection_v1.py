"""Trusted one-shot MMQA P1 selection and late-gold custody.

The adapter is the sole component allowed to join the four pinned MultiModalQA
files.  It has no model, network, evaluator, retrieval, or scoring surface.
Before decoding any gzip member it consumes one selection attempt, creates and
commits one 32-byte secret, and verifies all four compressed streams against
their frozen size, Git-blob SHA-1, and an externally frozen SHA-256
qualification receipt.

Action items deliberately expose only an opaque work ID, the question, local
ROW/TEXT ordinals with ID-free serialized content, and exact reciprocal
structural edges.  Source identities, exact question type, family, metadata,
answers, and support annotations remain in a mode-0600 trusted ledger.  Gold
ordinal packs are separate mode-0600 files.  A block gold pack can be opened
once, only through an authorization bound to immutable action archives;
``F_search`` is rejected before any custody file is inspected.

The generic ``acquire_once`` entry point exists for synthetic tests.  The
formal wrapper fixes the official paths and byte contracts and disables secret
injection.  Neither route retries, resamples, changes quotas, or falls back.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import hmac
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable
import unicodedata
from urllib.parse import unquote, urlsplit


VERSION = "mmqa_p1_private_selection_v1"
STUDY_ID = "MMQA_P1_LOCAL_PROOF_E5_V1"
FORMAL_CONTROLLER_VERSION = "mmqa_p1_formal_controller_v1"
ACTION_INTEGRATION_VERSION = "mmqa_p1_action_integration_v1"

SOURCE_CUSTODY_SELF_SHA256 = (
    "e82cb94e54a3020d1f2e41f47ed4141d19b448db985479551b1d933b43bf15f5"
)
STUDY_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)

FAMILY_BY_EXACT_TYPE = {
    "Compose(TextQ,TableQ)": "FIRST_TABLE_THEN_TEXT",
    "Compose(TableQ,TextQ)": "FIRST_TEXT_THEN_TABLE",
    "Intersect(TableQ,TextQ)": "TABLE_TEXT_INTERSECTION",
}
FAMILIES = tuple(FAMILY_BY_EXACT_TYPE.values())
EXACT_TYPE_BY_FAMILY = {family: exact for exact, family in FAMILY_BY_EXACT_TYPE.items()}

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
DEV_BLOCK_ORDER = ("F_search", "A_hold", "M_search")
BLOCK_SOURCE_SPLIT = {
    "A_form": "TRAIN",
    "F_search": "DEV",
    "A_hold": "DEV",
    "M_search": "DEV",
}
BLOCK_QUOTA_PER_FAMILY = {
    "A_form": 40,
    "F_search": 5,
    "A_hold": 15,
    "M_search": 15,
}
BLOCK_ITEM_COUNTS = {
    block: len(FAMILIES) * quota
    for block, quota in BLOCK_QUOTA_PER_FAMILY.items()
}
SELECTED_TOTAL = sum(BLOCK_ITEM_COUNTS.values())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT_RELATIVE = Path("artifacts/mmqa_p1_official_source_v1")
CUSTODY_RELATIVE = Path("manifests/mmqa_p1_source_custody_v1.json")
DESIGN_RELATIVE = Path("manifests/mmqa_p1_local_proof_e5_study_design_v1.json")
QUALIFICATION_RELATIVE = Path("manifests/mmqa_p1_source_qualification_result_v1.json")
OUTPUT_RELATIVE = Path("artifacts/mmqa_p1_private_selection_v1")

ATTEMPT_MARKER_FILENAME = "selection.one_shot.private.json"
SECRET_FILENAME = "selection_secret.private.bin"
COMMITMENT_FILENAME = "selection_commitment.public.json"
PUBLIC_RECEIPT_FILENAME = "selection_receipt.public.json"
PRIVATE_LEDGER_FILENAME = "selection_source_mapping.trusted.private.json"
FAILURE_FILENAME = "selection.terminal_failure.public.json"
ACTION_PACK_FILENAMES = {
    block: f"{block}.action.label_free.private.json" for block in BLOCK_ORDER
}
GOLD_PACK_FILENAMES = {
    block: f"{block}.gold.sealed.private.json" for block in BLOCK_ORDER
}
GOLD_OPEN_MARKER_FILENAMES = {
    block: f"{block}.gold.opened.private.json" for block in BLOCK_ORDER
}

ORDER_HMAC_DOMAIN = b"MMQA_P1_PRIVATE_ITEM_ORDER_HMAC_SHA256_V1\x00"
COMPONENT_HMAC_DOMAIN = b"MMQA_P1_PRIVATE_COMPONENT_ORDER_HMAC_SHA256_V1\x00"
BLOCK_ORDER_HMAC_DOMAIN = b"MMQA_P1_PRIVATE_BLOCK_ORDER_HMAC_SHA256_V1\x00"
WORK_ID_HMAC_DOMAIN = b"MMQA_P1_OPAQUE_WORK_ID_HMAC_SHA256_V1\x00"

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"mmqa-work-v1-[0-9a-f]{64}\Z")
_WHITESPACE = re.compile(r"\s+")
_FORMAL_CAPABILITY = object()


class MmqaP1PrivateSelectionError(RuntimeError):
    """The fixed source, private selection, or late-gold contract failed."""


@dataclass(frozen=True)
class SourceFileContract:
    file_name: str
    size_bytes: int
    git_blob_sha1: str
    maximum_uncompressed_bytes: int
    maximum_records: int
    maximum_line_bytes: int = 16_000_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.file_name, str)
            or not self.file_name
            or Path(self.file_name).name != self.file_name
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or _HEX40.fullmatch(self.git_blob_sha1) is None
            or type(self.maximum_uncompressed_bytes) is not int
            or self.maximum_uncompressed_bytes <= 0
            or type(self.maximum_records) is not int
            or self.maximum_records <= 0
            or type(self.maximum_line_bytes) is not int
            or self.maximum_line_bytes <= 0
        ):
            raise MmqaP1PrivateSelectionError("source file contract is invalid")


@dataclass(frozen=True)
class SelectionContract:
    files: Mapping[str, SourceFileContract]
    expected_train_rows: int
    expected_dev_rows: int
    maximum_identifier_characters: int = 2_048
    maximum_candidate_texts: int = 48
    maximum_gold_rows: int = 4
    maximum_gold_texts: int = 4

    def __post_init__(self) -> None:
        required = {
            "MMQA_train.jsonl.gz",
            "MMQA_dev.jsonl.gz",
            "MMQA_tables.jsonl.gz",
            "MMQA_texts.jsonl.gz",
        }
        if (
            not isinstance(self.files, Mapping)
            or set(self.files) != required
            or any(
                not isinstance(value, SourceFileContract) or value.file_name != name
                for name, value in self.files.items()
            )
            or type(self.expected_train_rows) is not int
            or self.expected_train_rows < BLOCK_ITEM_COUNTS["A_form"]
            or type(self.expected_dev_rows) is not int
            or self.expected_dev_rows
            < sum(BLOCK_ITEM_COUNTS[block] for block in DEV_BLOCK_ORDER)
        ):
            raise MmqaP1PrivateSelectionError("selection contract is invalid")
        for value in (
            self.maximum_identifier_characters,
            self.maximum_candidate_texts,
            self.maximum_gold_rows,
            self.maximum_gold_texts,
        ):
            if type(value) is not int or value < 1:
                raise MmqaP1PrivateSelectionError("selection limit is invalid")


FORMAL_CONTRACT = SelectionContract(
    files={
        "MMQA_train.jsonl.gz": SourceFileContract(
            "MMQA_train.jsonl.gz",
            11_698_210,
            "a6f55fedf35225a217defa3777338f66716304a2",
            750_000_000,
            100_000,
        ),
        "MMQA_dev.jsonl.gz": SourceFileContract(
            "MMQA_dev.jsonl.gz",
            1_310_976,
            "7b268187629fe10e2f7678b039baf49c50b29e80",
            100_000_000,
            25_000,
        ),
        "MMQA_tables.jsonl.gz": SourceFileContract(
            "MMQA_tables.jsonl.gz",
            10_344_191,
            "c2a8c4add0f12c60cdedd91ab193483bfe0ffa6f",
            2_000_000_000,
            100_000,
        ),
        "MMQA_texts.jsonl.gz": SourceFileContract(
            "MMQA_texts.jsonl.gz",
            45_851_194,
            "debfcc4389f2ddd84647f8b6a2bde3ef41431343",
            2_000_000_000,
            500_000,
        ),
    },
    expected_train_rows=23_817,
    expected_dev_rows=2_441,
)


@dataclass(frozen=True)
class QualificationBinding:
    self_sha256: str
    source_sha256_by_file: Mapping[str, str]
    train_rows: int
    dev_rows: int
    table_records: int
    table_rows: int
    text_records: int


@dataclass(frozen=True)
class QuestionCandidate:
    split: str
    family: str
    exact_type: str
    qid: str
    question: str
    table_id: str
    text_ids: tuple[str, ...]
    gold_text_ids: frozenset[str]
    answer_row_indices: frozenset[int]

    @property
    def component_resources(self) -> frozenset[str]:
        return frozenset(
            ("table:" + self.table_id,)
            + tuple("text:" + value for value in self.text_ids)
        )


@dataclass(frozen=True)
class TextMeta:
    exact_link_keys: frozenset[str]


@dataclass(frozen=True)
class TableMeta:
    row_exact_link_keys: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class EligibleItem:
    source: QuestionCandidate
    gold_row_indices: tuple[int, ...]
    gold_text_ids: tuple[str, ...]
    exact_gold_pairs: tuple[tuple[int, str], ...]

    @property
    def family(self) -> str:
        return self.source.family

    @property
    def qid(self) -> str:
        return self.source.qid

    @property
    def component_resources(self) -> frozenset[str]:
        return self.source.component_resources


@dataclass(frozen=True)
class CellContent:
    text: str
    link_titles: tuple[str, ...]


@dataclass(frozen=True)
class TableContent:
    title: str
    table_name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[CellContent, ...], ...]


@dataclass(frozen=True)
class TextContent:
    title: str
    text: str


@dataclass(frozen=True)
class Component:
    member_indices: tuple[int, ...]
    counts: tuple[int, ...]
    private_order_digest: bytes


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1PrivateSelectionError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    if field in body:
        raise MmqaP1PrivateSelectionError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    if not isinstance(value, Mapping):
        raise MmqaP1PrivateSelectionError("self-hashed value is not an object")
    body = dict(value)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or _HEX64.fullmatch(claimed) is None:
        raise MmqaP1PrivateSelectionError("self-hash is missing or invalid")
    if not hmac.compare_digest(stable_hash(body), claimed):
        raise MmqaP1PrivateSelectionError("self-hash mismatch")
    return claimed


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MmqaP1PrivateSelectionError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise MmqaP1PrivateSelectionError("JSON contains a non-finite number")


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite,
        )
    except MmqaP1PrivateSelectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MmqaP1PrivateSelectionError(f"{label} JSON is invalid") from exc


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()  # nosec B324: immutable Git object identity
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _safe_identifier(value: object, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise MmqaP1PrivateSelectionError("source identifier schema drifted")
    return value


def _require_text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MmqaP1PrivateSelectionError("source text schema drifted")
    if not allow_empty and not value.strip():
        raise MmqaP1PrivateSelectionError("source text schema drifted")
    return value


def _canonical_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("_", " ")
    return _WHITESPACE.sub(" ", normalized).strip().casefold()


def _canonical_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise MmqaP1PrivateSelectionError("source URL schema drifted") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise MmqaP1PrivateSelectionError("source URL schema drifted")
    host = parsed.netloc.casefold()
    path = unquote(parsed.path)
    if host.endswith("wikipedia.org") and path.startswith("/wiki/"):
        return "wikipedia:" + _canonical_title(path[len("/wiki/") :])
    return "url:" + host + path.rstrip("/")


def _document_exact_link_keys(title: str, url: str) -> frozenset[str]:
    return frozenset(
        {
            "title:" + _canonical_title(title),
            "url:" + _canonical_url(url),
        }
    )


def _link_projection(value: Mapping[str, Any]) -> tuple[frozenset[str], str | None]:
    keys: set[str] = set()
    title: str | None = None
    if "wiki_title" in value:
        title = _require_text(value.get("wiki_title"))
        keys.add("title:" + _canonical_title(title))
    if "url" in value:
        url = _require_text(value.get("url"))
        keys.add("url:" + _canonical_url(url))
    if not keys:
        raise MmqaP1PrivateSelectionError("table link has no exact-link field")
    return frozenset(keys), title


def _frame(name: bytes, value: str) -> bytes:
    if not isinstance(name, bytes) or not name or b"\x00" in name:
        raise MmqaP1PrivateSelectionError("HMAC frame name is invalid")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MmqaP1PrivateSelectionError("HMAC frame value is invalid")
    raw = value.encode("utf-8")
    return name + b"\x00" + len(raw).to_bytes(8, "big") + raw


def _secret(secret: bytes) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise MmqaP1PrivateSelectionError("selection secret must be exactly 32 bytes")
    return secret


def selection_hmac_message(*, split: str, family: str, qid: str) -> bytes:
    if split not in {"TRAIN", "DEV"} or family not in FAMILIES:
        raise MmqaP1PrivateSelectionError("selection HMAC namespace is invalid")
    return (
        ORDER_HMAC_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"split", split)
        + _frame(b"family", family)
        + _frame(b"qid", qid)
    )


def selection_hmac_digest(
    secret: bytes, *, split: str, family: str, qid: str
) -> bytes:
    return hmac.new(
        _secret(secret),
        selection_hmac_message(split=split, family=family, qid=qid),
        hashlib.sha256,
    ).digest()


def opaque_work_id(
    secret: bytes, *, block: str, split: str, family: str, qid: str
) -> str:
    if block not in BLOCK_ORDER or BLOCK_SOURCE_SPLIT[block] != split:
        raise MmqaP1PrivateSelectionError("work ID namespace is invalid")
    message = (
        WORK_ID_HMAC_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"block", block)
        + _frame(b"split", split)
        + _frame(b"family", family)
        + _frame(b"qid", qid)
    )
    return "mmqa-work-v1-" + hmac.new(
        _secret(secret), message, hashlib.sha256
    ).hexdigest()


def _component_hmac_digest(secret: bytes, resources: Iterable[str]) -> bytes:
    ordered = tuple(sorted(resources))
    if not ordered or len(set(ordered)) != len(ordered):
        raise MmqaP1PrivateSelectionError("component resource identity is invalid")
    message = COMPONENT_HMAC_DOMAIN + _frame(b"study", STUDY_ID)
    for resource in ordered:
        message += _frame(b"resource", resource)
    return hmac.new(_secret(secret), message, hashlib.sha256).digest()


def _block_order_digest(secret: bytes, *, block: str, item: EligibleItem) -> bytes:
    message = (
        BLOCK_ORDER_HMAC_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"block", block)
        + _frame(b"family", item.family)
        + _frame(b"qid", item.qid)
    )
    return hmac.new(_secret(secret), message, hashlib.sha256).digest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise MmqaP1PrivateSelectionError("durable directory is unavailable") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MmqaP1PrivateSelectionError("durable path is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise MmqaP1PrivateSelectionError("directory parent is unavailable")
            missing.append(cursor)
            cursor = cursor.parent
            continue
        except OSError as exc:
            raise MmqaP1PrivateSelectionError("directory cannot be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MmqaP1PrivateSelectionError("directory path is unsafe")
        break
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
            os.chmod(directory, 0o700)
        except OSError as exc:
            raise MmqaP1PrivateSelectionError("directory cannot be created") from exc
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _create_one_shot_root(path: Path) -> None:
    _ensure_durable_directory(path.parent)
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
    except FileExistsError as exc:
        raise MmqaP1PrivateSelectionError(
            "selection root already exists; replay is forbidden"
        ) from exc
    except OSError as exc:
        raise MmqaP1PrivateSelectionError("selection root cannot be created") from exc
    _fsync_directory(path)
    _fsync_directory(path.parent)


def _read_stable_regular_bytes(
    path: Path,
    *,
    label: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise MmqaP1PrivateSelectionError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MmqaP1PrivateSelectionError(f"{label} is not a regular file")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise MmqaP1PrivateSelectionError(f"{label} mode drifted")
        if expected_size is not None and before.st_size != expected_size:
            raise MmqaP1PrivateSelectionError(f"{label} size drifted")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            chunks.append(chunk)
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
            raise MmqaP1PrivateSelectionError(f"{label} changed during read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if expected_size is not None and len(raw) != expected_size:
        raise MmqaP1PrivateSelectionError(f"{label} size drifted")
    if expected_sha256 is not None and not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_sha256
    ):
        raise MmqaP1PrivateSelectionError(f"{label} SHA256 drifted")
    return raw


def _write_exclusive_bytes(path: Path, raw: bytes, *, mode: int) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise MmqaP1PrivateSelectionError("exclusive payload is not bytes")
    _ensure_durable_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise MmqaP1PrivateSelectionError("exclusive output is unavailable") from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != mode
            ):
                raise MmqaP1PrivateSelectionError("exclusive output mode drifted")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode_octal": f"{mode:04o}",
    }


def _atomic_write_json(path: Path, value: object, *, mode: int) -> dict[str, Any]:
    raw = _canonical_bytes(value, newline=True)
    staging = path.with_name(f".{path.name}.part")
    if path.exists() or path.is_symlink() or staging.exists() or staging.is_symlink():
        raise MmqaP1PrivateSelectionError("atomic output already exists")
    _write_exclusive_bytes(staging, raw, mode=mode)
    try:
        os.link(staging, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        os.unlink(staging)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise MmqaP1PrivateSelectionError("atomic publication failed") from exc
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode_octal": f"{mode:04o}",
    }


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_stable_regular_bytes(path, label=label)
    value = _strict_json(raw, label=label)
    if not isinstance(value, Mapping):
        raise MmqaP1PrivateSelectionError(f"{label} is not an object")
    return dict(value), raw


def _verify_manifest(
    path: Path, *, expected_self_sha256: str, expected_schema: str
) -> dict[str, Any]:
    value, _raw = _load_json_object(path, label="bound manifest")
    observed = verify_self_hash(value)
    if (
        not hmac.compare_digest(observed, expected_self_sha256)
        or value.get("schema") != expected_schema
        or value.get("study_id") != STUDY_ID
    ):
        raise MmqaP1PrivateSelectionError("bound manifest drifted")
    return value


def verify_study_bindings(
    custody_path: Path = PROJECT_ROOT / CUSTODY_RELATIVE,
    design_path: Path = PROJECT_ROOT / DESIGN_RELATIVE,
) -> None:
    custody = _verify_manifest(
        Path(custody_path),
        expected_self_sha256=SOURCE_CUSTODY_SELF_SHA256,
        expected_schema="mmqa_p1_source_custody_v1",
    )
    design = _verify_manifest(
        Path(design_path),
        expected_self_sha256=STUDY_DESIGN_SELF_SHA256,
        expected_schema="mmqa_p1_local_proof_e5_study_design_v1",
    )
    pinned = custody.get("pinned_source")
    files = pinned.get("files") if isinstance(pinned, Mapping) else None
    if not isinstance(files, Mapping) or set(files) != set(FORMAL_CONTRACT.files):
        raise MmqaP1PrivateSelectionError("custody source file set drifted")
    for name, contract in FORMAL_CONTRACT.files.items():
        row = files.get(name)
        if (
            not isinstance(row, Mapping)
            or row.get("size_bytes") != contract.size_bytes
            or row.get("git_blob_sha1") != contract.git_blob_sha1
        ):
            raise MmqaP1PrivateSelectionError("custody source identity drifted")
    blocks = design.get("blocks")
    population = design.get("exact_support_population")
    execution = design.get("no_gate_loop_and_execution")
    if (
        design.get("source_custody_self_sha256") != SOURCE_CUSTODY_SELF_SHA256
        or not isinstance(blocks, Mapping)
        or not isinstance(population, Mapping)
        or not isinstance(execution, Mapping)
        or population.get("families")
        != {
            "FIRST_TABLE_THEN_TEXT": "official metadata.type exactly Compose(TextQ,TableQ)",
            "FIRST_TEXT_THEN_TABLE": "official metadata.type exactly Compose(TableQ,TextQ)",
            "TABLE_TEXT_INTERSECTION": "official metadata.type exactly Intersect(TableQ,TextQ)",
        }
        or execution.get("online_evaluation") is not False
    ):
        raise MmqaP1PrivateSelectionError("study design binding drifted")
    for block in BLOCK_ORDER:
        row = blocks.get(block)
        if (
            not isinstance(row, Mapping)
            or row.get("per_family") != BLOCK_QUOTA_PER_FAMILY[block]
            or row.get("source_split") != BLOCK_SOURCE_SPLIT[block]
        ):
            raise MmqaP1PrivateSelectionError("study block contract drifted")
    if (
        blocks.get("query_disjoint_within_split") is not True
        or blocks.get("shared_table_or_text_connected_component_disjoint_across_DEV_blocks")
        is not True
    ):
        raise MmqaP1PrivateSelectionError("study disjointness contract drifted")


def load_qualification_binding(
    path: Path,
    *,
    expected_self_sha256: str,
    contract: SelectionContract,
) -> QualificationBinding:
    value, raw = _load_json_object(Path(path), label="qualification SHA256 receipt")
    if raw != _canonical_bytes(value, newline=True):
        raise MmqaP1PrivateSelectionError("qualification receipt is noncanonical")
    observed = verify_self_hash(value)
    binding = value.get("binding_self_sha256")
    source_identity = value.get("source_identity")
    train = value.get("TRAIN")
    dev = value.get("DEV")
    aggregates = value.get("schema_aggregates")
    support = value.get("support_contract")
    if (
        _HEX64.fullmatch(expected_self_sha256) is None
        or not hmac.compare_digest(observed, expected_self_sha256)
        or value.get("schema") != "mmqa_p1_source_qualification_v1_result_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "qualified_aggregate_only"
        or value.get("qualified") is not True
        or value.get("model_action_embedding_reranking_or_score_count") != 0
        or value.get("online_evaluator_call_count") != 0
        or value.get("source_item_query_document_answer_support_or_identifier_output_count")
        != 0
        or not isinstance(binding, Mapping)
        or binding.get("source_custody") != SOURCE_CUSTODY_SELF_SHA256
        or binding.get("study_design") != STUDY_DESIGN_SELF_SHA256
        or not isinstance(source_identity, Mapping)
        or set(source_identity) != set(contract.files)
        or not isinstance(train, Mapping)
        or not isinstance(dev, Mapping)
        or not isinstance(aggregates, Mapping)
        or not isinstance(support, Mapping)
        or value.get("exact_type_family_count") != len(FAMILIES)
        or support.get("answer_table_index_rows_union_exact_linked_rows") is not True
        or support.get("gold_row_bounds_inclusive")
        != [1, contract.maximum_gold_rows]
        or support.get("gold_text_bounds_inclusive")
        != [1, contract.maximum_gold_texts]
        or support.get("requires_exact_gold_row_text_pair") is not True
        or support.get("identifier_or_content_output_count") != 0
        or support.get("support_parts") != ["table", "text"]
    ):
        raise MmqaP1PrivateSelectionError("qualification receipt drifted")
    train_eligible = train.get("eligible_count_by_family")
    dev_eligible = dev.get("eligible_count_by_family")
    dev_capacity = dev.get("component_disjoint_capacity")
    if (
        train.get("required_per_family") != BLOCK_QUOTA_PER_FAMILY["A_form"]
        or not isinstance(train_eligible, Mapping)
        or set(train_eligible) != set(FAMILIES)
        or any(
            type(train_eligible[family]) is not int
            or train_eligible[family] < BLOCK_QUOTA_PER_FAMILY["A_form"]
            for family in FAMILIES
        )
        or dev.get("required_total_per_family")
        != sum(BLOCK_QUOTA_PER_FAMILY[block] for block in DEV_BLOCK_ORDER)
        or not isinstance(dev_eligible, Mapping)
        or set(dev_eligible) != set(FAMILIES)
        or any(
            type(dev_eligible[family]) is not int
            or dev_eligible[family]
            < sum(BLOCK_QUOTA_PER_FAMILY[block] for block in DEV_BLOCK_ORDER)
            for family in FAMILIES
        )
        or not isinstance(dev_capacity, Mapping)
        or dev_capacity.get("qualified") is not True
    ):
        raise MmqaP1PrivateSelectionError("qualification capacity drifted")
    sha256_by_file: dict[str, str] = {}
    for name, file_contract in contract.files.items():
        row = source_identity.get(name)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"git_blob_sha1", "sha256", "size_bytes"}
            or row.get("git_blob_sha1") != file_contract.git_blob_sha1
            or row.get("size_bytes") != file_contract.size_bytes
            or not isinstance(row.get("sha256"), str)
            or _HEX64.fullmatch(str(row.get("sha256"))) is None
        ):
            raise MmqaP1PrivateSelectionError("qualification source identity drifted")
        sha256_by_file[name] = str(row["sha256"])
    numeric = (
        train.get("question_record_count"),
        dev.get("question_record_count"),
        aggregates.get("table_record_count"),
        aggregates.get("table_row_count"),
        aggregates.get("text_record_count"),
    )
    if any(type(row) is not int or row < 1 for row in numeric):
        raise MmqaP1PrivateSelectionError("qualification aggregate drifted")
    if numeric[0] != contract.expected_train_rows or numeric[1] != contract.expected_dev_rows:
        raise MmqaP1PrivateSelectionError("qualification split count drifted")
    return QualificationBinding(
        self_sha256=observed,
        source_sha256_by_file=sha256_by_file,
        train_rows=int(numeric[0]),
        dev_rows=int(numeric[1]),
        table_records=int(numeric[2]),
        table_rows=int(numeric[3]),
        text_records=int(numeric[4]),
    )


def _read_verified_sources(
    source_paths: Mapping[str, Path],
    *,
    contract: SelectionContract,
    expected_sha256_by_file: Mapping[str, str],
    require_mode_0600: bool,
) -> dict[str, bytes]:
    if set(source_paths) != set(contract.files) or set(expected_sha256_by_file) != set(
        contract.files
    ):
        raise MmqaP1PrivateSelectionError("source file set drifted")
    result: dict[str, bytes] = {}
    for name in sorted(contract.files):
        file_contract = contract.files[name]
        expected_sha256 = expected_sha256_by_file[name]
        raw = _read_stable_regular_bytes(
            Path(source_paths[name]),
            label=f"pinned source {name}",
            expected_size=file_contract.size_bytes,
            expected_sha256=expected_sha256,
            expected_mode=0o600 if require_mode_0600 else None,
        )
        if not hmac.compare_digest(_git_blob_sha1(raw), file_contract.git_blob_sha1):
            raise MmqaP1PrivateSelectionError("pinned source Git-blob identity drifted")
        result[name] = raw
    return result


def _iter_gzip_jsonl(raw: bytes, contract: SourceFileContract) -> Iterable[Mapping[str, Any]]:
    total = 0
    count = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
            while True:
                line = handle.readline(contract.maximum_line_bytes + 1)
                if not line:
                    break
                total += len(line)
                count += 1
                if (
                    len(line) > contract.maximum_line_bytes
                    or total > contract.maximum_uncompressed_bytes
                    or count > contract.maximum_records
                ):
                    raise MmqaP1PrivateSelectionError("bounded gzip contract exceeded")
                if not line.strip():
                    raise MmqaP1PrivateSelectionError("source JSONL has a blank record")
                value = _strict_json(line.strip(), label="source JSONL record")
                if not isinstance(value, Mapping):
                    raise MmqaP1PrivateSelectionError("source record is not an object")
                yield value
    except MmqaP1PrivateSelectionError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise MmqaP1PrivateSelectionError("source gzip stream is invalid") from exc
    if count == 0:
        raise MmqaP1PrivateSelectionError("source JSONL is empty")


def _optional_identifier_list(value: object, maximum: int) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    try:
        result = tuple(_safe_identifier(item, maximum) for item in value)
    except MmqaP1PrivateSelectionError:
        return None
    if len(set(result)) != len(result):
        return None
    return result


def _question_candidate(
    row: Mapping[str, Any],
    *,
    split: str,
    contract: SelectionContract,
) -> QuestionCandidate | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise MmqaP1PrivateSelectionError("question metadata schema drifted")
    exact_type = metadata.get("type")
    if not isinstance(exact_type, str):
        raise MmqaP1PrivateSelectionError("question type schema drifted")
    family = FAMILY_BY_EXACT_TYPE.get(exact_type)
    if family is None:
        return None
    modalities = metadata.get("modalities")
    if (
        not isinstance(modalities, list)
        or len(modalities) != 2
        or any(not isinstance(value, str) for value in modalities)
        or set(modalities) != {"table", "text"}
    ):
        return None
    try:
        table_id = _safe_identifier(
            metadata.get("table_id"), contract.maximum_identifier_characters
        )
    except MmqaP1PrivateSelectionError:
        return None
    text_ids = _optional_identifier_list(
        metadata.get("text_doc_ids"), contract.maximum_identifier_characters
    )
    if not text_ids or len(text_ids) > contract.maximum_candidate_texts:
        return None
    supporting = row.get("supporting_context")
    if not isinstance(supporting, list) or not supporting:
        return None
    support_pairs: set[tuple[str, str]] = set()
    gold_table_ids: set[str] = set()
    gold_text_ids: set[str] = set()
    for support in supporting:
        if not isinstance(support, Mapping):
            return None
        try:
            doc_id = _safe_identifier(
                support.get("doc_id"), contract.maximum_identifier_characters
            )
        except MmqaP1PrivateSelectionError:
            return None
        part = support.get("doc_part")
        if part not in {"table", "text"} or (part, doc_id) in support_pairs:
            return None
        support_pairs.add((part, doc_id))
        if part == "table":
            gold_table_ids.add(doc_id)
        else:
            gold_text_ids.add(doc_id)
    if (
        gold_table_ids != {table_id}
        or not 1 <= len(gold_text_ids) <= contract.maximum_gold_texts
        or not gold_text_ids.issubset(text_ids)
        or not (set(text_ids) - gold_text_ids)
    ):
        return None
    answers = row.get("answers")
    if not isinstance(answers, list) or not answers:
        return None
    answer_rows: set[int] = set()
    for answer in answers:
        if not isinstance(answer, Mapping):
            return None
        indices = answer.get("table_indices", [])
        if indices is None:
            indices = []
        if not isinstance(indices, list):
            return None
        for index in indices:
            if (
                not isinstance(index, list)
                or len(index) != 2
                or any(type(value) is not int for value in index)
                or index[0] < 0
                or index[1] < 0
            ):
                return None
            answer_rows.add(index[0])
    return QuestionCandidate(
        split=split,
        family=family,
        exact_type=exact_type,
        qid=_safe_identifier(row.get("qid"), contract.maximum_identifier_characters),
        question=_require_text(row.get("question")),
        table_id=table_id,
        text_ids=text_ids,
        gold_text_ids=frozenset(gold_text_ids),
        answer_row_indices=frozenset(answer_rows),
    )


def _load_question_split(
    raw: bytes,
    source_contract: SourceFileContract,
    *,
    split: str,
    contract: SelectionContract,
    global_qids: set[str],
) -> tuple[list[QuestionCandidate], int]:
    candidates: list[QuestionCandidate] = []
    count = 0
    for row in _iter_gzip_jsonl(raw, source_contract):
        count += 1
        if not {"qid", "question", "answers", "metadata", "supporting_context"}.issubset(row):
            raise MmqaP1PrivateSelectionError("question required fields are missing")
        qid = _safe_identifier(row.get("qid"), contract.maximum_identifier_characters)
        _require_text(row.get("question"))
        if qid in global_qids:
            raise MmqaP1PrivateSelectionError("question ID is duplicated across splits")
        global_qids.add(qid)
        candidate = _question_candidate(row, split=split, contract=contract)
        if candidate is not None:
            candidates.append(candidate)
    return candidates, count


def _load_text_metadata(
    raw: bytes,
    source_contract: SourceFileContract,
    *,
    referenced_ids: frozenset[str],
    contract: SelectionContract,
) -> tuple[dict[str, TextMeta], int]:
    seen: set[str] = set()
    retained: dict[str, TextMeta] = {}
    for row in _iter_gzip_jsonl(raw, source_contract):
        if not {"id", "title", "url", "text"}.issubset(row):
            raise MmqaP1PrivateSelectionError("text required fields are missing")
        identifier = _safe_identifier(row.get("id"), contract.maximum_identifier_characters)
        if identifier in seen:
            raise MmqaP1PrivateSelectionError("duplicate text identifier")
        seen.add(identifier)
        title = _require_text(row.get("title"))
        url = _require_text(row.get("url"))
        _require_text(row.get("text"))
        keys = _document_exact_link_keys(title, url)
        if identifier in referenced_ids:
            retained[identifier] = TextMeta(keys)
    return retained, len(seen)


def _parse_table_row(
    source_row: object, *, retain_content: bool
) -> tuple[frozenset[str], tuple[CellContent, ...] | None]:
    if not isinstance(source_row, list) or not source_row:
        raise MmqaP1PrivateSelectionError("table row schema drifted")
    row_keys: set[str] = set()
    cells: list[CellContent] = []
    for cell in source_row:
        if not isinstance(cell, Mapping):
            raise MmqaP1PrivateSelectionError("table cell schema drifted")
        text = _require_text(cell.get("text"), allow_empty=True)
        links = cell.get("links")
        if not isinstance(links, list):
            raise MmqaP1PrivateSelectionError("table links schema drifted")
        titles: list[str] = []
        for link in links:
            if not isinstance(link, Mapping):
                raise MmqaP1PrivateSelectionError("table link schema drifted")
            keys, title = _link_projection(link)
            row_keys.update(keys)
            if title is not None:
                titles.append(title)
        if retain_content:
            cells.append(CellContent(text=text, link_titles=tuple(titles)))
    return frozenset(row_keys), tuple(cells) if retain_content else None


def _load_table_metadata(
    raw: bytes,
    source_contract: SourceFileContract,
    *,
    referenced_ids: frozenset[str],
    contract: SelectionContract,
) -> tuple[dict[str, TableMeta], int, int]:
    seen: set[str] = set()
    retained: dict[str, TableMeta] = {}
    total_rows = 0
    for row in _iter_gzip_jsonl(raw, source_contract):
        if not {"id", "title", "url", "table"}.issubset(row):
            raise MmqaP1PrivateSelectionError("table required fields are missing")
        identifier = _safe_identifier(row.get("id"), contract.maximum_identifier_characters)
        if identifier in seen:
            raise MmqaP1PrivateSelectionError("duplicate table identifier")
        seen.add(identifier)
        _require_text(row.get("title"))
        _require_text(row.get("url"))
        table = row.get("table")
        if not isinstance(table, Mapping):
            raise MmqaP1PrivateSelectionError("table object schema drifted")
        table_rows = table.get("table_rows")
        header = table.get("header")
        _require_text(table.get("table_name"), allow_empty=True)
        if (
            not isinstance(table_rows, list)
            or not table_rows
            or not isinstance(header, list)
            or not header
        ):
            raise MmqaP1PrivateSelectionError("table schema drifted")
        for column in header:
            if not isinstance(column, Mapping):
                raise MmqaP1PrivateSelectionError("table header schema drifted")
            _require_text(column.get("column_name"), allow_empty=True)
        retain = identifier in referenced_ids
        keys: list[frozenset[str]] = []
        for source_row in table_rows:
            row_keys, _content = _parse_table_row(source_row, retain_content=False)
            if retain:
                keys.append(row_keys)
        if retain:
            retained[identifier] = TableMeta(tuple(keys))
        total_rows += len(table_rows)
    return retained, len(seen), total_rows


def form_eligible_items(
    candidates: Sequence[QuestionCandidate],
    *,
    texts: Mapping[str, TextMeta],
    tables: Mapping[str, TableMeta],
    contract: SelectionContract,
) -> tuple[EligibleItem, ...]:
    result: list[EligibleItem] = []
    for candidate in candidates:
        table = tables.get(candidate.table_id)
        if table is None or any(identifier not in texts for identifier in candidate.text_ids):
            continue
        if any(index >= len(table.row_exact_link_keys) for index in candidate.answer_row_indices):
            continue
        gold_text_keys: set[str] = set()
        for identifier in candidate.gold_text_ids:
            gold_text_keys.update(texts[identifier].exact_link_keys)
        linked_rows = {
            index
            for index, row_keys in enumerate(table.row_exact_link_keys)
            if row_keys.intersection(gold_text_keys)
        }
        gold_rows = set(candidate.answer_row_indices).union(linked_rows)
        if not 1 <= len(gold_rows) <= contract.maximum_gold_rows:
            continue
        ordered_gold_texts = tuple(
            identifier
            for identifier in candidate.text_ids
            if identifier in candidate.gold_text_ids
        )
        if not 1 <= len(ordered_gold_texts) <= contract.maximum_gold_texts:
            continue
        pairs = tuple(
            (row_index, text_id)
            for row_index in sorted(gold_rows)
            for text_id in ordered_gold_texts
            if table.row_exact_link_keys[row_index].intersection(
                texts[text_id].exact_link_keys
            )
        )
        if not pairs:
            continue
        result.append(
            EligibleItem(
                source=candidate,
                gold_row_indices=tuple(sorted(gold_rows)),
                gold_text_ids=ordered_gold_texts,
                exact_gold_pairs=pairs,
            )
        )
    return tuple(result)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def _components(items: Sequence[EligibleItem], *, secret: bytes) -> tuple[Component, ...]:
    disjoint = _DisjointSet(len(items))
    owner: dict[str, int] = {}
    for index, item in enumerate(items):
        for resource in item.component_resources:
            previous = owner.setdefault(resource, index)
            disjoint.union(index, previous)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        grouped[disjoint.find(index)].append(index)
    family_index = {family: index for index, family in enumerate(FAMILIES)}
    result: list[Component] = []
    for members in grouped.values():
        counts = [0] * len(FAMILIES)
        resources: set[str] = set()
        for index in members:
            counts[family_index[items[index].family]] += 1
            resources.update(items[index].component_resources)
        result.append(
            Component(
                member_indices=tuple(sorted(members)),
                counts=tuple(counts),
                private_order_digest=_component_hmac_digest(secret, resources),
            )
        )
    return tuple(sorted(result, key=lambda row: row.private_order_digest))


def _allocate_dev_components(
    items: Sequence[EligibleItem], *, secret: bytes
) -> tuple[dict[str, tuple[EligibleItem, ...]], dict[str, tuple[int, ...]]]:
    components = _components(items, secret=secret)
    strategies = ("balanced", "scarce", "compact", "coverage")

    def attempt(order: Sequence[str], strategy: str) -> dict[str, tuple[int, ...]] | None:
        remaining = set(range(len(components)))
        assigned: dict[str, tuple[int, ...]] = {}
        for position, block in enumerate(order):
            quota = BLOCK_QUOTA_PER_FAMILY[block]
            capacity = [0] * len(FAMILIES)
            chosen: list[int] = []
            future_required = sum(
                BLOCK_QUOTA_PER_FAMILY[name] for name in order[position + 1 :]
            )
            while any(value < quota for value in capacity):
                remaining_totals = [
                    sum(components[index].counts[family] for index in remaining)
                    for family in range(len(FAMILIES))
                ]
                candidates: list[tuple[tuple[float, ...], bytes, int]] = []
                for index in remaining:
                    vector = components[index].counts
                    gains = tuple(
                        min(max(quota - capacity[family], 0), vector[family])
                        for family in range(len(FAMILIES))
                    )
                    if not any(gains):
                        continue
                    if any(
                        remaining_totals[family] - vector[family] < future_required
                        for family in range(len(FAMILIES))
                    ):
                        continue
                    gain_total = sum(gains)
                    overshoot = sum(
                        max(capacity[family] + vector[family] - quota, 0)
                        for family in range(len(FAMILIES))
                    )
                    scarce_gain = sum(
                        gains[family] / max(remaining_totals[family], 1)
                        for family in range(len(FAMILIES))
                    )
                    covered = sum(value > 0 for value in gains)
                    if strategy == "balanced":
                        primary = (float(covered), float(gain_total), scarce_gain)
                    elif strategy == "scarce":
                        primary = (scarce_gain, float(covered), float(gain_total))
                    elif strategy == "compact":
                        primary = (-float(overshoot), float(gain_total), float(covered))
                    else:
                        primary = (float(gain_total), float(covered), -float(overshoot))
                    score = (
                        *primary,
                        -float(overshoot),
                        -float(sum(vector)),
                    )
                    candidates.append((score, components[index].private_order_digest, index))
                if not candidates:
                    return None
                best_score = max(row[0] for row in candidates)
                selected = min(
                    (row for row in candidates if row[0] == best_score),
                    key=lambda row: (row[1], row[2]),
                )[2]
                chosen.append(selected)
                remaining.remove(selected)
                vector = components[selected].counts
                capacity = [
                    min(quota, capacity[family] + vector[family])
                    for family in range(len(FAMILIES))
                ]
            assigned[block] = tuple(chosen)
        return assigned

    allocation: dict[str, tuple[int, ...]] | None = None
    for order in itertools.permutations(DEV_BLOCK_ORDER):
        for strategy in strategies:
            allocation = attempt(order, strategy)
            if allocation is not None:
                break
        if allocation is not None:
            break
    if allocation is None:
        raise MmqaP1PrivateSelectionError(
            "DEV component-disjoint exact quota allocation failed"
        )
    selected: dict[str, tuple[EligibleItem, ...]] = {}
    for block in DEV_BLOCK_ORDER:
        pool = [
            items[item_index]
            for component_index in allocation[block]
            for item_index in components[component_index].member_indices
        ]
        rows: list[EligibleItem] = []
        for family in FAMILIES:
            family_rows = sorted(
                (item for item in pool if item.family == family),
                key=lambda item: (
                    selection_hmac_digest(
                        secret,
                        split="DEV",
                        family=item.family,
                        qid=item.qid,
                    ),
                    item.qid.encode("utf-8"),
                ),
            )[: BLOCK_QUOTA_PER_FAMILY[block]]
            if len(family_rows) != BLOCK_QUOTA_PER_FAMILY[block]:
                raise MmqaP1PrivateSelectionError("DEV family quota is incomplete")
            rows.extend(family_rows)
        selected[block] = tuple(
            sorted(
                rows,
                key=lambda item: (
                    _block_order_digest(secret, block=block, item=item),
                    item.qid.encode("utf-8"),
                ),
            )
        )
    used_components = {
        block: tuple(sorted(allocation[block])) for block in DEV_BLOCK_ORDER
    }
    if any(
        set(used_components[left]).intersection(used_components[right])
        for left_index, left in enumerate(DEV_BLOCK_ORDER)
        for right in DEV_BLOCK_ORDER[left_index + 1 :]
    ):
        raise MmqaP1PrivateSelectionError("DEV component leakage across blocks")
    return selected, used_components


def select_private_blocks(
    records_by_split: Mapping[str, Sequence[EligibleItem]], *, secret: bytes
) -> tuple[dict[str, tuple[EligibleItem, ...]], dict[str, tuple[int, ...]]]:
    _secret(secret)
    if set(records_by_split) != {"TRAIN", "DEV"}:
        raise MmqaP1PrivateSelectionError("selection split set drifted")
    for split, rows in records_by_split.items():
        if any(
            not isinstance(row, EligibleItem) or row.source.split != split for row in rows
        ):
            raise MmqaP1PrivateSelectionError("eligible selection row drifted")
        identities = [row.qid for row in rows]
        if len(identities) != len(set(identities)):
            raise MmqaP1PrivateSelectionError("eligible question IDs are duplicated")
    train: list[EligibleItem] = []
    for family in FAMILIES:
        ordered = sorted(
            (row for row in records_by_split["TRAIN"] if row.family == family),
            key=lambda item: (
                selection_hmac_digest(
                    secret, split="TRAIN", family=family, qid=item.qid
                ),
                item.qid.encode("utf-8"),
            ),
        )
        chosen = ordered[: BLOCK_QUOTA_PER_FAMILY["A_form"]]
        if len(chosen) != BLOCK_QUOTA_PER_FAMILY["A_form"]:
            raise MmqaP1PrivateSelectionError("TRAIN family quota is incomplete")
        train.extend(chosen)
    selected: dict[str, tuple[EligibleItem, ...]] = {
        "A_form": tuple(
            sorted(
                train,
                key=lambda item: (
                    _block_order_digest(secret, block="A_form", item=item),
                    item.qid.encode("utf-8"),
                ),
            )
        )
    }
    dev, component_indices = _allocate_dev_components(
        records_by_split["DEV"], secret=secret
    )
    selected.update(dev)
    identities = [
        (item.source.split, item.qid)
        for block in BLOCK_ORDER
        for item in selected[block]
    ]
    if (
        len(identities) != SELECTED_TOTAL
        or len(identities) != len(set(identities))
        or any(len(selected[block]) != BLOCK_ITEM_COUNTS[block] for block in BLOCK_ORDER)
    ):
        raise MmqaP1PrivateSelectionError("one-shot selected block shape drifted")
    return selected, component_indices


def assign_a_form_oof_folds(
    items: Sequence[EligibleItem], *, secret: bytes
) -> tuple[dict[str, int], tuple[int, ...]]:
    """Assign complete A_form resource components to five private HMAC folds.

    Components are ordered by the same one-shot secret in a distinct domain.
    Each component is placed in the currently smallest fold (fold ordinal is
    the deterministic tie-break).  Thus no table/text connected component can
    cross folds, and no label, score, or model output affects the assignment.
    """

    rows = tuple(items)
    if len(rows) != BLOCK_ITEM_COUNTS["A_form"]:
        raise MmqaP1PrivateSelectionError("A_form OOF population size drifted")
    components = _components(rows, secret=secret)
    if len(components) < 5:
        raise MmqaP1PrivateSelectionError(
            "A_form has fewer than five resource-disjoint OOF components"
        )
    fold_sizes = [0] * 5
    assignment: dict[str, int] = {}
    resources_by_fold: list[set[str]] = [set() for _ in range(5)]
    for component in components:
        fold = min(range(5), key=lambda value: (fold_sizes[value], value))
        for member_index in component.member_indices:
            item = rows[member_index]
            if item.qid in assignment:
                raise MmqaP1PrivateSelectionError("A_form OOF identity collision")
            assignment[item.qid] = fold
            resources_by_fold[fold].update(item.component_resources)
        fold_sizes[fold] += len(component.member_indices)
    if set(assignment) != {item.qid for item in rows} or any(size <= 0 for size in fold_sizes):
        raise MmqaP1PrivateSelectionError("A_form OOF fold assignment is incomplete")
    if any(
        resources_by_fold[left].intersection(resources_by_fold[right])
        for left in range(5)
        for right in range(left + 1, 5)
    ):
        raise MmqaP1PrivateSelectionError("A_form component crossed OOF folds")
    return assignment, tuple(fold_sizes)


def _load_selected_text_content(
    raw: bytes,
    source_contract: SourceFileContract,
    *,
    selected_ids: frozenset[str],
    contract: SelectionContract,
) -> dict[str, TextContent]:
    result: dict[str, TextContent] = {}
    for row in _iter_gzip_jsonl(raw, source_contract):
        identifier = _safe_identifier(row.get("id"), contract.maximum_identifier_characters)
        if identifier in selected_ids:
            if identifier in result:
                raise MmqaP1PrivateSelectionError("selected text is duplicated")
            result[identifier] = TextContent(
                title=_require_text(row.get("title")),
                text=_require_text(row.get("text")),
            )
    if set(result) != set(selected_ids):
        raise MmqaP1PrivateSelectionError("selected text content is unresolved")
    return result


def _load_selected_table_content(
    raw: bytes,
    source_contract: SourceFileContract,
    *,
    selected_ids: frozenset[str],
    contract: SelectionContract,
) -> dict[str, TableContent]:
    result: dict[str, TableContent] = {}
    for row in _iter_gzip_jsonl(raw, source_contract):
        identifier = _safe_identifier(row.get("id"), contract.maximum_identifier_characters)
        if identifier not in selected_ids:
            continue
        if identifier in result:
            raise MmqaP1PrivateSelectionError("selected table is duplicated")
        table = row.get("table")
        if not isinstance(table, Mapping):
            raise MmqaP1PrivateSelectionError("selected table schema drifted")
        header = table.get("header")
        table_rows = table.get("table_rows")
        if not isinstance(header, list) or not isinstance(table_rows, list):
            raise MmqaP1PrivateSelectionError("selected table shape drifted")
        headers = tuple(
            _require_text(column.get("column_name"), allow_empty=True)
            for column in header
            if isinstance(column, Mapping)
        )
        if len(headers) != len(header):
            raise MmqaP1PrivateSelectionError("selected header schema drifted")
        rows: list[tuple[CellContent, ...]] = []
        for source_row in table_rows:
            _keys, cells = _parse_table_row(source_row, retain_content=True)
            if cells is None:
                raise MmqaP1PrivateSelectionError("selected row content was lost")
            rows.append(cells)
        result[identifier] = TableContent(
            title=_require_text(row.get("title")),
            table_name=_require_text(table.get("table_name"), allow_empty=True),
            headers=headers,
            rows=tuple(rows),
        )
    if set(result) != set(selected_ids):
        raise MmqaP1PrivateSelectionError("selected table content is unresolved")
    return result


def _serialized_row(table: TableContent, row_index: int) -> str:
    if not 0 <= row_index < len(table.rows):
        raise MmqaP1PrivateSelectionError("row serialization index drifted")
    value = {
        "cells": [
            {"link_titles": list(cell.link_titles), "text": cell.text}
            for cell in table.rows[row_index]
        ],
        "headers": list(table.headers),
        "table_name": table.table_name,
        "table_title": table.title,
    }
    return _canonical_bytes(value).decode("ascii")


def _serialized_text(text: TextContent) -> str:
    return _canonical_bytes({"text": text.text, "title": text.title}).decode("ascii")


def build_private_packs(
    selected: Mapping[str, Sequence[EligibleItem]],
    *,
    secret: bytes,
    table_meta: Mapping[str, TableMeta],
    text_meta: Mapping[str, TextMeta],
    table_content: Mapping[str, TableContent],
    text_content: Mapping[str, TextContent],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    _secret(secret)
    oof_assignment, oof_fold_sizes = assign_a_form_oof_folds(
        tuple(selected.get("A_form", ())), secret=secret
    )
    actions: dict[str, dict[str, Any]] = {}
    golds: dict[str, dict[str, Any]] = {}
    ledger_items: list[dict[str, Any]] = []
    all_work_ids: set[str] = set()
    for block in BLOCK_ORDER:
        rows = tuple(selected.get(block, ()))
        if len(rows) != BLOCK_ITEM_COUNTS[block]:
            raise MmqaP1PrivateSelectionError("private block quota drifted")
        action_items: list[dict[str, Any]] = []
        gold_items: list[dict[str, Any]] = []
        counts = Counter(item.family for item in rows)
        if counts != Counter(
            {family: BLOCK_QUOTA_PER_FAMILY[block] for family in FAMILIES}
        ):
            raise MmqaP1PrivateSelectionError("private family quota drifted")
        for block_ordinal, item in enumerate(rows):
            source = item.source
            work_id = opaque_work_id(
                secret,
                block=block,
                split=source.split,
                family=source.family,
                qid=source.qid,
            )
            if work_id in all_work_ids:
                raise MmqaP1PrivateSelectionError("opaque work ID collision")
            all_work_ids.add(work_id)
            table = table_content.get(source.table_id)
            table_links = table_meta.get(source.table_id)
            if table is None or table_links is None or len(table.rows) != len(
                table_links.row_exact_link_keys
            ):
                raise MmqaP1PrivateSelectionError("selected table projection drifted")
            nodes: list[dict[str, Any]] = [
                {
                    "ordinal": row_index,
                    "node_type": "ROW",
                    "content": _serialized_row(table, row_index),
                }
                for row_index in range(len(table.rows))
            ]
            text_ordinal: dict[str, int] = {}
            for position, text_id in enumerate(source.text_ids):
                ordinal = len(table.rows) + position
                text_ordinal[text_id] = ordinal
                content = text_content.get(text_id)
                if content is None:
                    raise MmqaP1PrivateSelectionError("selected text projection drifted")
                nodes.append(
                    {
                        "ordinal": ordinal,
                        "node_type": "TEXT",
                        "content": _serialized_text(content),
                    }
                )
            edges: list[dict[str, Any]] = []
            for row_index, row_keys in enumerate(table_links.row_exact_link_keys):
                for text_id in source.text_ids:
                    meta = text_meta.get(text_id)
                    if meta is None:
                        raise MmqaP1PrivateSelectionError("selected text metadata drifted")
                    if row_keys.intersection(meta.exact_link_keys):
                        target = text_ordinal[text_id]
                        edges.extend(
                            (
                                {
                                    "source_ordinal": row_index,
                                    "target_ordinal": target,
                                    "edge_type": "ROW_TO_TEXT",
                                },
                                {
                                    "source_ordinal": target,
                                    "target_ordinal": row_index,
                                    "edge_type": "TEXT_TO_ROW",
                                },
                            )
                        )
            edges.sort(
                key=lambda row: (
                    row["source_ordinal"],
                    row["target_ordinal"],
                    row["edge_type"],
                )
            )
            if not edges or len({tuple(row.values()) for row in edges}) != len(edges):
                raise MmqaP1PrivateSelectionError("exact structural edge projection drifted")
            action_items.append(
                {
                    "work_id": work_id,
                    "question": source.question,
                    "nodes": nodes,
                    "edges": edges,
                }
            )
            gold_rows = list(item.gold_row_indices)
            gold_texts = [text_ordinal[value] for value in item.gold_text_ids]
            gold_pairs = [
                {
                    "row_ordinal": row_index,
                    "text_ordinal": text_ordinal[text_id],
                }
                for row_index, text_id in item.exact_gold_pairs
            ]
            if (
                not 1 <= len(gold_rows) <= 4
                or not 1 <= len(gold_texts) <= 4
                or not gold_pairs
            ):
                raise MmqaP1PrivateSelectionError("gold ordinal projection drifted")
            gold_item: dict[str, Any] = {
                "work_id": work_id,
                "gold_row_ordinals": gold_rows,
                "gold_text_ordinals": gold_texts,
                "exact_gold_pairs": gold_pairs,
            }
            if block == "A_form":
                fold = oof_assignment.get(source.qid)
                if type(fold) is not int or not 0 <= fold < 5:
                    raise MmqaP1PrivateSelectionError("A_form OOF fold was lost")
                gold_item["oof_fold"] = fold
            elif block in {"A_hold", "M_search"}:
                # This public three-value enum is a late scoring stratum only.
                # It is absent from action packs and is forbidden from E5
                # features, fitting, policy formation, or model input.
                gold_item["evaluation_family"] = source.family
            gold_items.append(gold_item)
            ledger_items.append(
                {
                    "block": block,
                    "block_ordinal": block_ordinal,
                    "work_id": work_id,
                    "source_split": source.split,
                    "source_family": source.family,
                    "source_exact_type": source.exact_type,
                    "source_qid": source.qid,
                    "source_table_id": source.table_id,
                    "source_text_doc_ids": list(source.text_ids),
                }
            )
        action_body = {
            "schema": f"{VERSION}_label_free_action_pack_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "block": block,
            "item_count": len(action_items),
            "item_exact_fields": ["work_id", "question", "nodes", "edges"],
            "source_identifier_family_exact_type_answer_support_or_metadata_included": False,
            "items": action_items,
        }
        action_pack = self_hashed(action_body, "action_pack_sha256")
        gold_body = {
            "schema": f"{VERSION}_sealed_gold_pack_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "block": block,
            "item_count": len(gold_items),
            "action_pack_sha256": action_pack["action_pack_sha256"],
            "source_identifier_exact_type_answer_or_support_included": False,
            "evaluation_family_included_as_late_only_scoring_stratum": block
            in {"A_hold", "M_search"},
            "evaluation_family_forbidden_from_action_E5_features_fit_or_policy": True,
            "component_atomic_HMAC_oof_fold_included": block == "A_form",
            "items": gold_items,
        }
        actions[block] = action_pack
        golds[block] = self_hashed(gold_body, "gold_pack_sha256")
    ledger_body = {
        "schema": f"{VERSION}_trusted_source_mapping_ledger_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "sole_private_source_identity_family_and_block_mapping",
        "item_count": len(ledger_items),
        "items": ledger_items,
        "action_pack_sha256_by_block": {
            block: actions[block]["action_pack_sha256"] for block in BLOCK_ORDER
        },
        "gold_pack_sha256_by_block": {
            block: golds[block]["gold_pack_sha256"] for block in BLOCK_ORDER
        },
    }
    a_form_commitment_rows = [
        {"work_id": row["work_id"], "oof_fold": row["oof_fold"]}
        for row in golds["A_form"]["items"]
    ]
    oof_public = {
        "fold_count": 5,
        "fold_sizes": {str(index): size for index, size in enumerate(oof_fold_sizes)},
        "assignment_commitment_sha256": stable_hash(a_form_commitment_rows),
        "component_atomic": True,
        "secret_HMAC_ordered_deterministic_balancing": True,
    }
    return actions, golds, self_hashed(ledger_body, "ledger_sha256"), oof_public


def _persist_private_packs(
    root: Path,
    *,
    actions: Mapping[str, Mapping[str, Any]],
    golds: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for block in BLOCK_ORDER:
        action = actions[block]
        gold = golds[block]
        action_file = _atomic_write_json(
            root / ACTION_PACK_FILENAMES[block], action, mode=0o600
        )
        gold_file = _atomic_write_json(
            root / GOLD_PACK_FILENAMES[block], gold, mode=0o600
        )
        bindings[block] = {
            "action": {
                **action_file,
                "relative_path": ACTION_PACK_FILENAMES[block],
                "semantic_sha256": action["action_pack_sha256"],
            },
            "gold": {
                **gold_file,
                "relative_path": GOLD_PACK_FILENAMES[block],
                "semantic_sha256": gold["gold_pack_sha256"],
            },
        }
    ledger_file = _atomic_write_json(
        root / PRIVATE_LEDGER_FILENAME, ledger, mode=0o600
    )
    ledger_binding = {
        **ledger_file,
        "relative_path": PRIVATE_LEDGER_FILENAME,
        "semantic_sha256": ledger["ledger_sha256"],
    }
    return bindings, ledger_binding


def _assert_public_safe(value: object) -> None:
    forbidden_keys = {
        "qid",
        "question",
        "nodes",
        "edges",
        "content",
        "work_id",
        "metadata",
        "answers",
        "supporting_context",
        "table_id",
        "text_doc_ids",
        "gold_row_ordinals",
        "gold_text_ordinals",
        "exact_gold_pairs",
        "source_qid",
        "source_table_id",
        "source_text_doc_ids",
    }

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if forbidden_keys.intersection(node):
                raise MmqaP1PrivateSelectionError("public receipt leaked an item field")
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
        elif isinstance(node, str) and _WORK_ID.fullmatch(node):
            raise MmqaP1PrivateSelectionError("public receipt leaked a work ID")

    visit(value)


def _terminal_failure(
    root: Path, *, stage: str, exc: BaseException, commitment: str | None
) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "terminal_no_retry_replay_resample_or_secret_rotation",
        "failure_stage": stage,
        "exception_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "selection_secret_commitment_sha256": commitment,
        "source_item_identifier_content_answer_or_support_published": False,
    }
    value = self_hashed(body)
    try:
        _assert_public_safe(value)
        _atomic_write_json(root / FAILURE_FILENAME, value, mode=0o644)
    except BaseException:
        pass


def acquire_once(
    *,
    source_paths: Mapping[str, Path],
    qualification_path: Path,
    expected_qualification_self_sha256: str,
    output_root: Path,
    contract: SelectionContract,
    random_bytes: Callable[[int], bytes] | None = None,
    require_source_mode_0600: bool = True,
    custody_path: Path = PROJECT_ROOT / CUSTODY_RELATIVE,
    design_path: Path = PROJECT_ROOT / DESIGN_RELATIVE,
    _formal_capability: object | None = None,
) -> dict[str, Any]:
    """Consume one selection and create ID-free action plus sealed-gold packs."""

    if contract == FORMAL_CONTRACT:
        if (
            _formal_capability is not _FORMAL_CAPABILITY
            or random_bytes is not None
            or require_source_mode_0600 is not True
        ):
            raise MmqaP1PrivateSelectionError(
                "official source is restricted to the fixed formal wrapper"
            )
    elif _formal_capability is not None:
        raise MmqaP1PrivateSelectionError("formal capability used on synthetic source")

    verify_study_bindings(custody_path, design_path)
    qualification = load_qualification_binding(
        qualification_path,
        expected_self_sha256=expected_qualification_self_sha256,
        contract=contract,
    )
    root = Path(output_root)
    _create_one_shot_root(root)
    stage = "write_one_shot_attempt_marker"
    commitment: str | None = None
    try:
        attempt_body = {
            "schema": f"{VERSION}_one_shot_marker_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "consumed_before_secret_creation_or_source_parse",
            "source_custody_self_sha256": SOURCE_CUSTODY_SELF_SHA256,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "qualification_self_sha256": qualification.self_sha256,
            "source_file_count": 4,
            "retry_replay_resample_or_secret_rotation_authorized": False,
        }
        attempt = self_hashed(attempt_body)
        _atomic_write_json(root / ATTEMPT_MARKER_FILENAME, attempt, mode=0o600)

        stage = "create_exactly_one_selection_secret"
        generator = os.urandom if random_bytes is None else random_bytes
        secret = generator(32)
        _secret(secret)
        _write_exclusive_bytes(root / SECRET_FILENAME, secret, mode=0o600)
        commitment = hashlib.sha256(secret).hexdigest()

        stage = "publish_preparse_selection_commitment"
        commitment_body = {
            "schema": f"{VERSION}_selection_commitment_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "one_32_byte_secret_committed_before_four_source_parses",
            "attempt_marker_sha256": attempt["self_sha256"],
            "selection_secret_commitment_sha256": commitment,
            "selection_secret_size_bytes": 32,
            "os_random_call_count": 1,
            "os_random_requested_bytes": 32,
            "source_file_count": 4,
            "source_identifier_content_answer_or_support_published": False,
        }
        commitment_value = self_hashed(commitment_body)
        _assert_public_safe(commitment_value)
        commitment_file = _atomic_write_json(
            root / COMMITMENT_FILENAME, commitment_value, mode=0o644
        )

        stage = "verify_all_four_compressed_source_identities"
        compressed = _read_verified_sources(
            source_paths,
            contract=contract,
            expected_sha256_by_file=qualification.source_sha256_by_file,
            require_mode_0600=require_source_mode_0600,
        )

        stage = "parse_question_splits_privately"
        global_qids: set[str] = set()
        train_candidates, train_count = _load_question_split(
            compressed["MMQA_train.jsonl.gz"],
            contract.files["MMQA_train.jsonl.gz"],
            split="TRAIN",
            contract=contract,
            global_qids=global_qids,
        )
        dev_candidates, dev_count = _load_question_split(
            compressed["MMQA_dev.jsonl.gz"],
            contract.files["MMQA_dev.jsonl.gz"],
            split="DEV",
            contract=contract,
            global_qids=global_qids,
        )
        if train_count != qualification.train_rows or dev_count != qualification.dev_rows:
            raise MmqaP1PrivateSelectionError("question split count drifted")
        candidates = tuple(train_candidates + dev_candidates)
        referenced_texts = frozenset(
            identifier for row in candidates for identifier in row.text_ids
        )
        referenced_tables = frozenset(row.table_id for row in candidates)

        stage = "parse_corpora_and_exact_structural_links_privately"
        text_meta, text_count = _load_text_metadata(
            compressed["MMQA_texts.jsonl.gz"],
            contract.files["MMQA_texts.jsonl.gz"],
            referenced_ids=referenced_texts,
            contract=contract,
        )
        table_meta, table_count, table_row_count = _load_table_metadata(
            compressed["MMQA_tables.jsonl.gz"],
            contract.files["MMQA_tables.jsonl.gz"],
            referenced_ids=referenced_tables,
            contract=contract,
        )
        if (
            text_count != qualification.text_records
            or table_count != qualification.table_records
            or table_row_count != qualification.table_rows
        ):
            raise MmqaP1PrivateSelectionError("corpus aggregate drifted")

        stage = "derive_exact_support_population_privately"
        eligible_train = form_eligible_items(
            train_candidates,
            texts=text_meta,
            tables=table_meta,
            contract=contract,
        )
        eligible_dev = form_eligible_items(
            dev_candidates,
            texts=text_meta,
            tables=table_meta,
            contract=contract,
        )

        stage = "fixed_HMAC_query_and_component_disjoint_selection"
        selected, component_indices = select_private_blocks(
            {"TRAIN": eligible_train, "DEV": eligible_dev}, secret=secret
        )

        stage = "load_only_selected_ID_free_action_content"
        selected_table_ids = frozenset(
            item.source.table_id
            for block in BLOCK_ORDER
            for item in selected[block]
        )
        selected_text_ids = frozenset(
            identifier
            for block in BLOCK_ORDER
            for item in selected[block]
            for identifier in item.source.text_ids
        )
        table_content = _load_selected_table_content(
            compressed["MMQA_tables.jsonl.gz"],
            contract.files["MMQA_tables.jsonl.gz"],
            selected_ids=selected_table_ids,
            contract=contract,
        )
        text_content = _load_selected_text_content(
            compressed["MMQA_texts.jsonl.gz"],
            contract.files["MMQA_texts.jsonl.gz"],
            selected_ids=selected_text_ids,
            contract=contract,
        )

        stage = "form_and_persist_separated_action_gold_and_ledger_packs"
        actions, golds, ledger, oof_public = build_private_packs(
            selected,
            secret=secret,
            table_meta=table_meta,
            text_meta=text_meta,
            table_content=table_content,
            text_content=text_content,
        )
        pack_bindings, ledger_binding = _persist_private_packs(
            root, actions=actions, golds=golds, ledger=ledger
        )

        stage = "persist_aggregate_public_receipt"
        receipt_body = {
            "schema": f"{VERSION}_public_receipt_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "private_one_shot_selection_complete",
            "binding_self_sha256": {
                "source_custody": SOURCE_CUSTODY_SELF_SHA256,
                "study_design": STUDY_DESIGN_SELF_SHA256,
                "qualification": qualification.self_sha256,
                "selection_commitment": commitment_value["self_sha256"],
            },
            "selection_commitment_file_sha256": commitment_file["file_sha256"],
            "selection_secret_commitment_sha256": commitment,
            "source_identity": {
                name: {
                    "size_bytes": contract.files[name].size_bytes,
                    "git_blob_sha1": contract.files[name].git_blob_sha1,
                    "sha256": qualification.source_sha256_by_file[name],
                }
                for name in sorted(contract.files)
            },
            "selection_contract": {
                "block_order": list(BLOCK_ORDER),
                "item_count_by_block": dict(BLOCK_ITEM_COUNTS),
                "per_exact_type_count_by_block": dict(BLOCK_QUOTA_PER_FAMILY),
                "exact_type_count": len(FAMILIES),
                "selected_total": SELECTED_TOTAL,
                "query_disjoint_within_split": True,
                "DEV_component_disjoint_across_blocks": True,
                "DEV_assigned_component_count_by_block": {
                    block: len(component_indices[block]) for block in DEV_BLOCK_ORDER
                },
                "selection_algorithm": "one_secret_HMAC_SHA256_v1",
                "A_form_five_fold_OOF": oof_public,
            },
            "private_pack_bindings": pack_bindings,
            "trusted_private_ledger_binding": ledger_binding,
            "isolation": {
                "action_item_exact_fields": ["work_id", "question", "nodes", "edges"],
                "source_identifiers_family_type_metadata_answers_and_support_absent_from_actions": True,
                "gold_packs_separate_mode_0600": True,
                "evaluation_family_only_in_late_A_hold_and_M_search_gold": True,
                "evaluation_family_forbidden_from_E5_features_fit_or_policy": True,
                "source_mapping_only_in_mode_0600_trusted_ledger": True,
                "F_search_gold_open_authorized": False,
            },
            "model_network_retrieval_evaluator_or_score_calls": 0,
            "retry_replay_resample_or_secret_rotation": 0,
            "source_item_identifier_content_answer_or_support_published": False,
        }
        receipt = self_hashed(receipt_body, "acquisition_sha256")
        _assert_public_safe(receipt)
        _atomic_write_json(root / PUBLIC_RECEIPT_FILENAME, receipt, mode=0o644)
        return receipt
    except BaseException as exc:
        _terminal_failure(root, stage=stage, exc=exc, commitment=commitment)
        if isinstance(exc, MmqaP1PrivateSelectionError):
            raise
        raise MmqaP1PrivateSelectionError("one-shot private selection failed") from exc


def run_formal_selection(
    project_root: str | Path, *, expected_qualification_self_sha256: str
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise MmqaP1PrivateSelectionError("formal project root is invalid")
    source_root = project / SOURCE_ROOT_RELATIVE
    return acquire_once(
        source_paths={name: source_root / name for name in FORMAL_CONTRACT.files},
        qualification_path=project / QUALIFICATION_RELATIVE,
        expected_qualification_self_sha256=expected_qualification_self_sha256,
        output_root=project / OUTPUT_RELATIVE,
        contract=FORMAL_CONTRACT,
        random_bytes=None,
        require_source_mode_0600=True,
        custody_path=project / CUSTODY_RELATIVE,
        design_path=project / DESIGN_RELATIVE,
        _formal_capability=_FORMAL_CAPABILITY,
    )


def _load_public_receipt(root: Path) -> dict[str, Any]:
    value, raw = _load_json_object(root / PUBLIC_RECEIPT_FILENAME, label="selection receipt")
    if raw != _canonical_bytes(value, newline=True):
        raise MmqaP1PrivateSelectionError("selection receipt is noncanonical")
    verify_self_hash(value, "acquisition_sha256")
    if (
        value.get("schema") != f"{VERSION}_public_receipt_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "private_one_shot_selection_complete"
    ):
        raise MmqaP1PrivateSelectionError("selection receipt drifted")
    _assert_public_safe(value)
    return value


def _pack_binding(
    receipt: Mapping[str, Any], *, block: str, role: str
) -> Mapping[str, Any]:
    packs = receipt.get("private_pack_bindings")
    row = packs.get(block) if isinstance(packs, Mapping) else None
    binding = row.get(role) if isinstance(row, Mapping) else None
    expected_name = (
        ACTION_PACK_FILENAMES[block] if role == "action" else GOLD_PACK_FILENAMES[block]
    )
    if (
        role not in {"action", "gold"}
        or not isinstance(binding, Mapping)
        or binding.get("relative_path") != expected_name
        or binding.get("mode_octal") != "0600"
        or _HEX64.fullmatch(str(binding.get("file_sha256"))) is None
        or _HEX64.fullmatch(str(binding.get("semantic_sha256"))) is None
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] <= 0
    ):
        raise MmqaP1PrivateSelectionError("private pack binding drifted")
    return binding


def _action_projection_binding(
    value: object,
) -> tuple[str, str, tuple[int, ...], tuple[int, ...]]:
    """Reconstruct the anonymous projection hash without importing the executor."""

    if not isinstance(value, Mapping) or set(value) != {
        "work_id",
        "question",
        "nodes",
        "edges",
    }:
        raise MmqaP1PrivateSelectionError("action archive pack item drifted")
    work_id = value.get("work_id")
    question = value.get("question")
    nodes = value.get("nodes")
    edges = value.get("edges")
    if (
        not isinstance(work_id, str)
        or _WORK_ID.fullmatch(work_id) is None
        or not isinstance(question, str)
        or not isinstance(nodes, list)
        or not nodes
        or not isinstance(edges, list)
        or not edges
    ):
        raise MmqaP1PrivateSelectionError("action archive pack item is invalid")

    rows: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    node_types: dict[int, str] = {}
    for expected_ordinal, node in enumerate(nodes):
        if not isinstance(node, Mapping) or set(node) != {
            "ordinal",
            "node_type",
            "content",
        }:
            raise MmqaP1PrivateSelectionError("action archive pack node drifted")
        ordinal = node.get("ordinal")
        node_type = node.get("node_type")
        content = node.get("content")
        if (
            type(ordinal) is not int
            or ordinal != expected_ordinal
            or node_type not in {"ROW", "TEXT"}
            or not isinstance(content, str)
        ):
            raise MmqaP1PrivateSelectionError("action archive pack node is invalid")
        node_types[ordinal] = str(node_type)
        projected = {"ordinal": ordinal, "serialized_content": content}
        (rows if node_type == "ROW" else texts).append(projected)

    directed: set[tuple[int, int, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping) or set(edge) != {
            "source_ordinal",
            "target_ordinal",
            "edge_type",
        }:
            raise MmqaP1PrivateSelectionError("action archive pack edge drifted")
        source = edge.get("source_ordinal")
        target = edge.get("target_ordinal")
        edge_type = edge.get("edge_type")
        if type(source) is not int or type(target) is not int:
            raise MmqaP1PrivateSelectionError("action archive pack edge is invalid")
        expected_type = (
            "ROW_TO_TEXT"
            if node_types.get(source) == "ROW" and node_types.get(target) == "TEXT"
            else "TEXT_TO_ROW"
            if node_types.get(source) == "TEXT" and node_types.get(target) == "ROW"
            else None
        )
        row = (source, target, str(edge_type))
        if edge_type != expected_type or row in directed:
            raise MmqaP1PrivateSelectionError("action archive pack edge is invalid")
        directed.add(row)

    links: list[dict[str, int]] = []
    for source, target, edge_type in sorted(directed):
        if edge_type != "ROW_TO_TEXT":
            continue
        if (target, source, "TEXT_TO_ROW") not in directed:
            raise MmqaP1PrivateSelectionError("action archive pack edge lacks reverse")
        links.append({"row_ordinal": source, "text_ordinal": target})
    if not links or len(directed) != 2 * len(links):
        raise MmqaP1PrivateSelectionError("action archive pack edges are incomplete")
    projection = {
        "schema": f"{ACTION_INTEGRATION_VERSION}_anonymous_work_item",
        "question": question,
        "rows": rows,
        "texts": texts,
        "exact_row_text_links": links,
    }
    return (
        work_id,
        stable_hash(projection),
        tuple(row["ordinal"] for row in rows),
        tuple(text["ordinal"] for text in texts),
    )


def _bound_action_projection_bindings(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    block: str,
) -> tuple[
    Mapping[str, Any],
    tuple[
        tuple[str, str, tuple[int, ...], tuple[int, ...]],
        ...,
    ],
]:
    binding = _pack_binding(receipt, block=block, role="action")
    value = _read_bound_pack(
        root, binding=binding, label=f"{block} action archive action pack"
    )
    expected_fields = {
        "schema",
        "version",
        "study_id",
        "block",
        "item_count",
        "item_exact_fields",
        "source_identifier_family_exact_type_answer_support_or_metadata_included",
        "items",
        "action_pack_sha256",
    }
    semantic = verify_self_hash(value, "action_pack_sha256")
    items = value.get("items")
    if (
        set(value) != expected_fields
        or value.get("schema") != f"{VERSION}_label_free_action_pack_v1"
        or value.get("version") != VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("block") != block
        or value.get("item_count") != BLOCK_ITEM_COUNTS[block]
        or value.get("item_exact_fields") != ["work_id", "question", "nodes", "edges"]
        or value.get(
            "source_identifier_family_exact_type_answer_support_or_metadata_included"
        )
        is not False
        or not isinstance(items, list)
        or len(items) != BLOCK_ITEM_COUNTS[block]
        or not hmac.compare_digest(semantic, str(binding["semantic_sha256"]))
    ):
        raise MmqaP1PrivateSelectionError("action archive action pack drifted")
    projected = tuple(_action_projection_binding(item) for item in items)
    if len({row[0] for row in projected}) != len(projected):
        raise MmqaP1PrivateSelectionError("action archive work IDs are duplicated")
    return binding, projected


def _read_canonical_private_artifact(
    path: Path, *, label: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = _read_stable_regular_bytes(Path(path), label=label, expected_mode=0o600)
    value = _strict_json(raw, label=label)
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_bytes(value, newline=True)
    ):
        raise MmqaP1PrivateSelectionError(f"{label} is noncanonical")
    return (
        dict(value),
        raw,
        {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "mode_octal": "0600",
        },
    )


def _valid_archive_ranking(
    value: object, *, closure: frozenset[int], required: bool
) -> bool:
    if value is None:
        return not required
    if not isinstance(value, Mapping) or set(value) != {
        "policy_id",
        "top5_ordinals",
        "selected_bundle_ordinals",
        "selected_bundle_energy_float64_hex",
    }:
        return False
    top5 = value.get("top5_ordinals")
    bundle = value.get("selected_bundle_ordinals")
    energy = value.get("selected_bundle_energy_float64_hex")
    if (
        not isinstance(value.get("policy_id"), str)
        or not value["policy_id"]
        or not isinstance(top5, list)
        or len(top5) != 5
        or len(set(top5)) != 5
        or any(type(ordinal) is not int for ordinal in top5)
        or not set(top5).issubset(closure)
        or (
            bundle is not None
            and (
                not isinstance(bundle, list)
                or not bundle
                or len(set(bundle)) != len(bundle)
                or any(type(ordinal) is not int for ordinal in bundle)
                or not set(bundle).issubset(closure)
            )
        )
        or (energy is not None and not isinstance(energy, str))
    ):
        return False
    if isinstance(energy, str):
        try:
            parsed = float.fromhex(energy)
        except ValueError:
            return False
        if not math.isfinite(parsed):
            return False
    return True


def _validate_action_archive_item(
    value: object,
    *,
    expected_work_id: str,
    expected_projection_sha256: str,
    expected_row_ordinals: tuple[int, ...],
    expected_text_ordinals: tuple[int, ...],
    block: str,
) -> None:
    expected_fields = {
        "work_id",
        "anonymous_projection_sha256",
        "coordinates",
        "coordinate_vector_sha256",
        "closure_ordinal_bytes_sha256",
        "action_feature_archive",
        "action_feature_archive_sha256",
        "E0",
        "E5",
        "RAW",
        "sealed_bundle_first_top5",
        "sealed_bundle_first_top5_sha256",
        "HippoRAG_top5_ordinals",
        "HippoRAG_payload_binding",
        "gold_family_type_answer_support_or_source_ID_read_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise MmqaP1PrivateSelectionError("action archive item shape drifted")
    coordinates = value.get("coordinates")
    features = value.get("action_feature_archive")
    sealed = value.get("sealed_bundle_first_top5")
    if (
        value.get("work_id") != expected_work_id
        or value.get("anonymous_projection_sha256") != expected_projection_sha256
        or not isinstance(coordinates, list)
        or not coordinates
        or value.get("coordinate_vector_sha256") != stable_hash(coordinates)
        or not isinstance(features, Mapping)
        or value.get("action_feature_archive_sha256") != stable_hash(features)
        or not isinstance(sealed, list)
        or not sealed
        or value.get("sealed_bundle_first_top5_sha256") != stable_hash(sealed)
        or value.get("gold_family_type_answer_support_or_source_ID_read_count") != 0
    ):
        raise MmqaP1PrivateSelectionError("action archive item binding drifted")

    coordinate_fields = {
        "ordinal",
        "minilm_similarity_float64_hex",
        "cross_encoder_relevance_float64_hex",
        "entity_anchor",
        "relation_anchor",
        "numeric_or_temporal_anchor",
    }
    all_ordinals = tuple(sorted((*expected_row_ordinals, *expected_text_ordinals)))
    if (
        len(coordinates) != len(all_ordinals)
        or [
            row.get("ordinal") if isinstance(row, Mapping) else None
            for row in coordinates
        ]
        != list(all_ordinals)
        or any(
            not isinstance(row, Mapping) or set(row) != coordinate_fields
            for row in coordinates
        )
    ):
        raise MmqaP1PrivateSelectionError("action archive coordinate coverage drifted")
    coordinate_scores: dict[int, tuple[float, float]] = {}
    for row in coordinates:
        try:
            minilm = float.fromhex(str(row["minilm_similarity_float64_hex"]))
            cross_encoder = float.fromhex(
                str(row["cross_encoder_relevance_float64_hex"])
            )
        except ValueError as exc:
            raise MmqaP1PrivateSelectionError(
                "action archive coordinate is invalid"
            ) from exc
        if (
            not math.isfinite(minilm)
            or not math.isfinite(cross_encoder)
            or any(
                row[field] not in {0, 1}
                for field in (
                    "entity_anchor",
                    "relation_anchor",
                    "numeric_or_temporal_anchor",
                )
            )
        ):
            raise MmqaP1PrivateSelectionError("action archive coordinate is invalid")
        coordinate_scores[int(row["ordinal"])] = (minilm, cross_encoder)
    retained_rows = tuple(
        sorted(
            expected_row_ordinals,
            key=lambda ordinal: (
                -coordinate_scores[ordinal][1],
                -coordinate_scores[ordinal][0],
                ordinal,
            ),
        )[:48]
    )
    expected_closure_ordinals = tuple(
        sorted((*retained_rows, *expected_text_ordinals))
    )

    closure = features.get("closure_ordinals")
    closure_hashes = features.get("three_arm_closure_ordinal_bytes_sha256")
    if (
        features.get("schema")
        != f"{ACTION_INTEGRATION_VERSION}_action_feature_archive"
        or features.get("study_id") != STUDY_ID
        or features.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
        or features.get("anonymous_projection_sha256")
        != expected_projection_sha256
        or not isinstance(closure, list)
        or not 5 <= len(closure) <= 96
        or closure != list(expected_closure_ordinals)
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in closure)
        or not isinstance(closure_hashes, Mapping)
        or set(closure_hashes) != {"AGENT", "RAW", "HIPPORAG"}
        or len(set(closure_hashes.values())) != 1
        or features.get("three_arm_closure_ordinals_byte_identical") is not True
        or any(
            features.get(field) != 0
            for field in (
                "gold_or_support_read_count",
                "network_call_count",
                "model_call_count",
                "source_reader_call_count",
                "retry_replay_resample_count",
                "source_metadata_id_feature_count",
                "family_or_question_type_feature_count",
            )
        )
    ):
        raise MmqaP1PrivateSelectionError("action feature archive drifted")
    closure_set = frozenset(closure)
    ordinal_sha256 = hashlib.sha256(_canonical_bytes(closure)).hexdigest()
    if (
        value.get("closure_ordinal_bytes_sha256") != ordinal_sha256
        or set(closure_hashes.values()) != {ordinal_sha256}
    ):
        raise MmqaP1PrivateSelectionError("action archive closure binding drifted")

    e5_required = block != "A_form"
    for arm, required in (("E0", True), ("E5", e5_required), ("RAW", True)):
        ranking = value.get(arm)
        if not _valid_archive_ranking(ranking, closure=closure_set, required=required):
            raise MmqaP1PrivateSelectionError("action archive ranking drifted")
        feature_ranking = features.get(arm)
        if ranking is None:
            if feature_ranking is not None:
                raise MmqaP1PrivateSelectionError("action archive ranking drifted")
        elif (
            not isinstance(feature_ranking, Mapping)
            or feature_ranking.get("policy_id") != ranking.get("policy_id")
            or feature_ranking.get("top5_ordinals") != ranking.get("top5_ordinals")
            or feature_ranking.get("selected_bundle_ordinals")
            != ranking.get("selected_bundle_ordinals")
        ):
            raise MmqaP1PrivateSelectionError("action feature ranking binding drifted")

    for row in sealed:
        if not isinstance(row, Mapping) or set(row) != {
            "bundle_ordinals",
            "top5_ordinals",
        }:
            raise MmqaP1PrivateSelectionError("sealed action candidate drifted")
        bundle = row.get("bundle_ordinals")
        top5 = row.get("top5_ordinals")
        if (
            not isinstance(bundle, list)
            or not bundle
            or len(set(bundle)) != len(bundle)
            or any(type(ordinal) is not int for ordinal in bundle)
            or not set(bundle).issubset(closure_set)
            or not isinstance(top5, list)
            or len(top5) != 5
            or len(set(top5)) != 5
            or any(type(ordinal) is not int for ordinal in top5)
            or not set(top5).issubset(closure_set)
        ):
            raise MmqaP1PrivateSelectionError("sealed action candidate is invalid")

    hippo_top5 = value.get("HippoRAG_top5_ordinals")
    hippo_binding = value.get("HippoRAG_payload_binding")
    if block == "A_hold":
        if (
            not isinstance(hippo_top5, list)
            or len(hippo_top5) != 5
            or len(set(hippo_top5)) != 5
            or any(type(ordinal) is not int for ordinal in hippo_top5)
            or not set(hippo_top5).issubset(closure_set)
            or not isinstance(hippo_binding, Mapping)
        ):
            raise MmqaP1PrivateSelectionError("A_hold HippoRAG archive is incomplete")
    elif hippo_top5 is not None or hippo_binding is not None:
        raise MmqaP1PrivateSelectionError("unexpected HippoRAG action archive")


def _validate_action_archive_artifact(
    path: Path,
    *,
    root: Path,
    receipt: Mapping[str, Any],
    block: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    action_binding, projections = _bound_action_projection_bindings(
        root, receipt=receipt, block=block
    )
    value, _raw, file_binding = _read_canonical_private_artifact(
        Path(path), label=f"{block} canonical action archive"
    )
    expected_fields = {
        "schema",
        "version",
        "study_id",
        "study_design_self_sha256",
        "selection_acquisition_sha256",
        "block",
        "status",
        "action_pack_sha256",
        "action_pack_file_sha256",
        "item_count",
        "coordinate_provider_block_batch_call_count",
        "hipporag_block_batch_call_count",
        "E5_model_sha256",
        "items",
        "gold_open_count_before_archive",
        "online_evaluator_call_count",
        "retry_replay_resample_count",
        "archive_sha256",
    }
    semantic = verify_self_hash(value, "archive_sha256")
    items = value.get("items")
    expected_hippo_calls = 1 if block == "A_hold" else 0
    expected_model = value.get("E5_model_sha256")
    if (
        _HEX64.fullmatch(expected_file_sha256) is None
        or not hmac.compare_digest(file_binding["file_sha256"], expected_file_sha256)
        or set(value) != expected_fields
        or value.get("schema")
        != f"{FORMAL_CONTROLLER_VERSION}_stage_action_archive_v1"
        or value.get("version") != FORMAL_CONTROLLER_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256") != STUDY_DESIGN_SELF_SHA256
        or value.get("selection_acquisition_sha256")
        != receipt.get("acquisition_sha256")
        or value.get("block") != block
        or value.get("status")
        != "all_label_free_actions_complete_and_sealed_before_gold"
        or value.get("action_pack_sha256") != action_binding.get("semantic_sha256")
        or value.get("action_pack_file_sha256") != action_binding.get("file_sha256")
        or value.get("item_count") != BLOCK_ITEM_COUNTS[block]
        or value.get("coordinate_provider_block_batch_call_count") != 1
        or value.get("hipporag_block_batch_call_count") != expected_hippo_calls
        or (
            expected_model is not None
            and (
                not isinstance(expected_model, str)
                or _HEX64.fullmatch(expected_model) is None
            )
        )
        or (block == "A_form") != (expected_model is None)
        or not isinstance(items, list)
        or len(items) != BLOCK_ITEM_COUNTS[block]
        or value.get("gold_open_count_before_archive") != 0
        or value.get("online_evaluator_call_count") != 0
        or value.get("retry_replay_resample_count") != 0
    ):
        raise MmqaP1PrivateSelectionError("canonical action archive drifted")
    for item, (
        work_id,
        projection_sha256,
        row_ordinals,
        text_ordinals,
    ) in zip(items, projections, strict=True):
        _validate_action_archive_item(
            item,
            expected_work_id=work_id,
            expected_projection_sha256=projection_sha256,
            expected_row_ordinals=row_ordinals,
            expected_text_ordinals=text_ordinals,
            block=block,
        )
    return {
        **file_binding,
        "semantic_sha256": semantic,
        "item_count": len(items),
    }


def _validate_gold_open_marker(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    block: str,
    expected_authorization_sha256: str,
) -> None:
    value, _raw, _binding = _read_canonical_private_artifact(
        root / GOLD_OPEN_MARKER_FILENAMES[block],
        label=f"{block} consumed gold-open marker",
    )
    expected_fields = {
        "schema",
        "version",
        "study_id",
        "status",
        "block",
        "acquisition_sha256",
        "authorization_sha256",
        "same_block_second_open_authorized",
        "self_sha256",
    }
    verify_self_hash(value)
    if (
        set(value) != expected_fields
        or value.get("schema") != f"{VERSION}_gold_open_marker_v1"
        or value.get("version") != VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("status")
        != "authorization_consumed_immediately_before_gold_open"
        or value.get("block") != block
        or value.get("acquisition_sha256") != receipt.get("acquisition_sha256")
        or value.get("authorization_sha256") != expected_authorization_sha256
        or value.get("same_block_second_open_authorized") is not False
    ):
        raise MmqaP1PrivateSelectionError("consumed gold-open marker drifted")


def _validate_a_hold_promotion_receipt(
    path: Path,
    *,
    root: Path,
    receipt: Mapping[str, Any],
    expected_promotion_sha256: str,
    action_archive_path: Path,
) -> dict[str, Any]:
    value, _raw, file_binding = _read_canonical_private_artifact(
        Path(path), label="canonical A_hold promotion receipt"
    )
    expected_fields = {
        "schema",
        "version",
        "study_id",
        "status",
        "block",
        "action_archive_sha256",
        "action_archive_file_sha256",
        "gold_authorization",
        "gold_pack_sha256",
        "item_count",
        "items",
        "promotion_E5_minus_E0",
        "promoted",
        "M_search_authorized",
        "reality_primary",
        "late_family_used_for_offline_stratified_scoring_only",
        "post_gold_action_reformation_count",
        "online_evaluator_call_count",
        "retry_replay_resample_count",
        "score_sha256",
    }
    semantic = verify_self_hash(value, "score_sha256")
    items = value.get("items")
    promotion = value.get("promotion_E5_minus_E0")
    if (
        _HEX64.fullmatch(expected_promotion_sha256) is None
        or not hmac.compare_digest(semantic, expected_promotion_sha256)
        or set(value) != expected_fields
        or value.get("schema")
        != f"{FORMAL_CONTROLLER_VERSION}_A_hold_offline_four_arm_score_v1"
        or value.get("version") != FORMAL_CONTROLLER_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "promoted_open_M_search"
        or value.get("block") != "A_hold"
        or value.get("item_count") != BLOCK_ITEM_COUNTS["A_hold"]
        or not isinstance(items, list)
        or len(items) != BLOCK_ITEM_COUNTS["A_hold"]
        or not isinstance(promotion, Mapping)
        or promotion.get("passed") is not True
        or value.get("promoted") is not True
        or value.get("M_search_authorized") is not True
        or value.get("late_family_used_for_offline_stratified_scoring_only")
        is not True
        or value.get("post_gold_action_reformation_count") != 0
        or value.get("online_evaluator_call_count") != 0
        or value.get("retry_replay_resample_count") != 0
    ):
        raise MmqaP1PrivateSelectionError("A_hold promotion receipt drifted")

    a_hold_archive = _validate_action_archive_artifact(
        Path(action_archive_path),
        root=root,
        receipt=receipt,
        block="A_hold",
        expected_file_sha256=str(value.get("action_archive_file_sha256")),
    )
    if value.get("action_archive_sha256") != a_hold_archive["semantic_sha256"]:
        raise MmqaP1PrivateSelectionError("A_hold promotion archive binding drifted")

    authorization = value.get("gold_authorization")
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "semantic_sha256",
        "file_sha256",
        "size_bytes",
        "mode_octal",
    }:
        raise MmqaP1PrivateSelectionError("A_hold promotion authorization drifted")
    authorization_path = Path(path).parent / "gold.open.authorization.private.json"
    auth_value, auth_raw, auth_file = _read_canonical_private_artifact(
        authorization_path, label="canonical A_hold gold authorization"
    )
    action = _pack_binding(receipt, block="A_hold", role="action")
    gold = _pack_binding(receipt, block="A_hold", role="gold")
    auth_semantic = _validate_authorization(
        auth_value,
        expected_authorization_sha256=str(authorization.get("semantic_sha256")),
        receipt=receipt,
        block="A_hold",
        action=action,
        gold=gold,
    )
    if (
        authorization.get("file_sha256") != auth_file["file_sha256"]
        or authorization.get("size_bytes") != len(auth_raw)
        or authorization.get("mode_octal") != "0600"
        or value.get("gold_pack_sha256") != gold.get("semantic_sha256")
    ):
        raise MmqaP1PrivateSelectionError("A_hold promotion gold binding drifted")
    _validate_gold_open_marker(
        root,
        receipt=receipt,
        block="A_hold",
        expected_authorization_sha256=auth_semantic,
    )

    archive_ids = [
        row.get("work_id")
        for row in _read_canonical_private_artifact(
            Path(action_archive_path), label="canonical A_hold action archive reread"
        )[0]["items"]
    ]
    score_ids = [
        row.get("work_id") if isinstance(row, Mapping) else None for row in items
    ]
    if score_ids != archive_ids:
        raise MmqaP1PrivateSelectionError("A_hold promotion item coverage drifted")
    return {**file_binding, "semantic_sha256": semantic}


def write_block_gold_open_authorization(
    path: Path,
    *,
    output_root: Path,
    block: str,
    action_archive_sha256s: Sequence[str],
    action_archive_paths: Sequence[Path],
    promotion_sha256: str | None = None,
    promotion_receipt_path: Path | None = None,
    promotion_action_archive_path: Path | None = None,
) -> dict[str, Any]:
    if block == "F_search":
        raise MmqaP1PrivateSelectionError("F_search gold is permanently sealed")
    if block not in BLOCK_ORDER:
        raise MmqaP1PrivateSelectionError("gold-open block is invalid")
    archives = tuple(action_archive_sha256s)
    archive_paths = tuple(Path(value) for value in action_archive_paths)
    if (
        len(archives) != 1
        or len(archive_paths) != 1
        or len(set(archives)) != len(archives)
        or any(not isinstance(value, str) or _HEX64.fullmatch(value) is None for value in archives)
    ):
        raise MmqaP1PrivateSelectionError("action archive bindings are invalid")
    archives = tuple(sorted(archives))
    if block == "M_search":
        if not isinstance(promotion_sha256, str) or _HEX64.fullmatch(promotion_sha256) is None:
            raise MmqaP1PrivateSelectionError("M_search requires A_hold promotion")
        if promotion_receipt_path is None or promotion_action_archive_path is None:
            raise MmqaP1PrivateSelectionError(
                "M_search requires canonical A_hold promotion artifacts"
            )
    elif (
        promotion_sha256 is not None
        or promotion_receipt_path is not None
        or promotion_action_archive_path is not None
    ):
        raise MmqaP1PrivateSelectionError("unexpected promotion binding")
    root = Path(output_root)
    receipt = _load_public_receipt(root)
    action = _pack_binding(receipt, block=block, role="action")
    gold = _pack_binding(receipt, block=block, role="gold")
    archive_paths = tuple(Path(os.path.abspath(value)) for value in archive_paths)
    archive_binding = _validate_action_archive_artifact(
        archive_paths[0],
        root=root,
        receipt=receipt,
        block=block,
        expected_file_sha256=archives[0],
    )
    if block == "M_search":
        promotion_binding = _validate_a_hold_promotion_receipt(
            Path(promotion_receipt_path),
            root=root,
            receipt=receipt,
            expected_promotion_sha256=str(promotion_sha256),
            action_archive_path=Path(promotion_action_archive_path),
        )
    else:
        promotion_binding = None
    body = {
        "schema": f"{VERSION}_block_gold_open_authorization_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "gold_open_authorized_after_immutable_action_archives",
        "block": block,
        "acquisition_sha256": receipt["acquisition_sha256"],
        "action_pack_sha256": action["semantic_sha256"],
        "gold_pack_sha256": gold["semantic_sha256"],
        "action_archive_sha256s": list(archives),
        "action_archive_paths": [str(value) for value in archive_paths],
        "action_archive_semantic_sha256s": [
            archive_binding["semantic_sha256"]
        ],
        "action_archives_complete_and_immutable": True,
        "A_hold_promotion_sha256": promotion_sha256,
        "A_hold_promotion_file_sha256": (
            None
            if promotion_binding is None
            else promotion_binding["file_sha256"]
        ),
        "A_hold_promotion_receipt_path": (
            None
            if promotion_receipt_path is None
            else str(Path(os.path.abspath(promotion_receipt_path)))
        ),
        "A_hold_promotion_action_archive_path": (
            None
            if promotion_action_archive_path is None
            else str(Path(os.path.abspath(promotion_action_archive_path)))
        ),
        "same_block_replay_authorized": False,
    }
    value = self_hashed(body, "authorization_sha256")
    _atomic_write_json(Path(path), value, mode=0o600)
    return value


def _read_bound_pack(
    root: Path, *, binding: Mapping[str, Any], label: str
) -> dict[str, Any]:
    raw = _read_stable_regular_bytes(
        root / str(binding["relative_path"]),
        label=label,
        expected_size=int(binding["size_bytes"]),
        expected_sha256=str(binding["file_sha256"]),
        expected_mode=0o600,
    )
    value = _strict_json(raw, label=label)
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value, newline=True):
        raise MmqaP1PrivateSelectionError(f"{label} is noncanonical")
    return dict(value)


def _validate_authorization(
    value: Mapping[str, Any],
    *,
    expected_authorization_sha256: str,
    receipt: Mapping[str, Any],
    block: str,
    action: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> str:
    observed = verify_self_hash(value, "authorization_sha256")
    archives = value.get("action_archive_sha256s")
    archive_paths = value.get("action_archive_paths")
    semantic_archives = value.get("action_archive_semantic_sha256s")
    promotion = value.get("A_hold_promotion_sha256")
    promotion_file = value.get("A_hold_promotion_file_sha256")
    promotion_path = value.get("A_hold_promotion_receipt_path")
    promotion_archive_path = value.get("A_hold_promotion_action_archive_path")
    expected_fields = {
        "schema",
        "version",
        "study_id",
        "status",
        "block",
        "acquisition_sha256",
        "action_pack_sha256",
        "gold_pack_sha256",
        "action_archive_sha256s",
        "action_archive_paths",
        "action_archive_semantic_sha256s",
        "action_archives_complete_and_immutable",
        "A_hold_promotion_sha256",
        "A_hold_promotion_file_sha256",
        "A_hold_promotion_receipt_path",
        "A_hold_promotion_action_archive_path",
        "same_block_replay_authorized",
        "authorization_sha256",
    }
    if (
        _HEX64.fullmatch(expected_authorization_sha256) is None
        or not hmac.compare_digest(observed, expected_authorization_sha256)
        or set(value) != expected_fields
        or value.get("schema") != f"{VERSION}_block_gold_open_authorization_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("status") != "gold_open_authorized_after_immutable_action_archives"
        or value.get("block") != block
        or value.get("acquisition_sha256") != receipt.get("acquisition_sha256")
        or value.get("action_pack_sha256") != action.get("semantic_sha256")
        or value.get("gold_pack_sha256") != gold.get("semantic_sha256")
        or not isinstance(archives, list)
        or not archives
        or archives != sorted(set(archives))
        or any(not isinstance(row, str) or _HEX64.fullmatch(row) is None for row in archives)
        or not isinstance(archive_paths, list)
        or len(archive_paths) != 1
        or any(
            not isinstance(row, str) or not Path(row).is_absolute()
            for row in archive_paths
        )
        or not isinstance(semantic_archives, list)
        or len(semantic_archives) != 1
        or any(
            not isinstance(row, str) or _HEX64.fullmatch(row) is None
            for row in semantic_archives
        )
        or value.get("action_archives_complete_and_immutable") is not True
        or value.get("same_block_replay_authorized") is not False
    ):
        raise MmqaP1PrivateSelectionError("gold-open authorization drifted")
    if block == "M_search":
        if (
            not isinstance(promotion, str)
            or _HEX64.fullmatch(promotion) is None
            or not isinstance(promotion_file, str)
            or _HEX64.fullmatch(promotion_file) is None
            or not isinstance(promotion_path, str)
            or not Path(promotion_path).is_absolute()
            or not isinstance(promotion_archive_path, str)
            or not Path(promotion_archive_path).is_absolute()
        ):
            raise MmqaP1PrivateSelectionError("M_search promotion drifted")
    elif (
        promotion is not None
        or promotion_file is not None
        or promotion_path is not None
        or promotion_archive_path is not None
    ):
        raise MmqaP1PrivateSelectionError("unexpected promotion authorization")
    return observed


def _validate_gold_pack(
    value: Mapping[str, Any], *, block: str, expected_action_sha256: str
) -> str:
    observed = verify_self_hash(value, "gold_pack_sha256")
    items = value.get("items")
    expected_pack_fields = {
        "schema",
        "version",
        "study_id",
        "block",
        "item_count",
        "action_pack_sha256",
        "source_identifier_exact_type_answer_or_support_included",
        "evaluation_family_included_as_late_only_scoring_stratum",
        "evaluation_family_forbidden_from_action_E5_features_fit_or_policy",
        "component_atomic_HMAC_oof_fold_included",
        "items",
        "gold_pack_sha256",
    }
    if (
        set(value) != expected_pack_fields
        or
        value.get("schema") != f"{VERSION}_sealed_gold_pack_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("block") != block
        or value.get("item_count") != BLOCK_ITEM_COUNTS[block]
        or value.get("action_pack_sha256") != expected_action_sha256
        or value.get("source_identifier_exact_type_answer_or_support_included")
        is not False
        or value.get("evaluation_family_included_as_late_only_scoring_stratum")
        is not (block in {"A_hold", "M_search"})
        or value.get("evaluation_family_forbidden_from_action_E5_features_fit_or_policy")
        is not True
        or value.get("component_atomic_HMAC_oof_fold_included")
        is not (block == "A_form")
        or not isinstance(items, list)
        or len(items) != BLOCK_ITEM_COUNTS[block]
    ):
        raise MmqaP1PrivateSelectionError("sealed gold pack drifted")
    for item in items:
        expected_fields = {
            "work_id",
            "gold_row_ordinals",
            "gold_text_ordinals",
            "exact_gold_pairs",
        }
        if block == "A_form":
            expected_fields.add("oof_fold")
        elif block in {"A_hold", "M_search"}:
            expected_fields.add("evaluation_family")
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise MmqaP1PrivateSelectionError("sealed gold item shape drifted")
        rows = item.get("gold_row_ordinals")
        texts = item.get("gold_text_ordinals")
        pairs = item.get("exact_gold_pairs")
        if (
            _WORK_ID.fullmatch(str(item.get("work_id"))) is None
            or not isinstance(rows, list)
            or not 1 <= len(rows) <= 4
            or rows != sorted(set(rows))
            or any(type(value) is not int or value < 0 for value in rows)
            or not isinstance(texts, list)
            or not 1 <= len(texts) <= 4
            or texts != sorted(set(texts))
            or any(type(value) is not int or value < 0 for value in texts)
            or not isinstance(pairs, list)
            or not pairs
        ):
            raise MmqaP1PrivateSelectionError("sealed gold ordinal drifted")
        expected_pairs: set[tuple[int, int]] = set()
        for pair in pairs:
            if (
                not isinstance(pair, Mapping)
                or set(pair) != {"row_ordinal", "text_ordinal"}
                or type(pair.get("row_ordinal")) is not int
                or type(pair.get("text_ordinal")) is not int
                or pair["row_ordinal"] not in rows
                or pair["text_ordinal"] not in texts
            ):
                raise MmqaP1PrivateSelectionError("sealed gold pair drifted")
            expected_pairs.add((pair["row_ordinal"], pair["text_ordinal"]))
        if len(expected_pairs) != len(pairs):
            raise MmqaP1PrivateSelectionError("sealed gold pair is duplicated")
        if block == "A_form":
            fold = item.get("oof_fold")
            if type(fold) is not int or not 0 <= fold < 5:
                raise MmqaP1PrivateSelectionError("sealed OOF fold drifted")
        elif block in {"A_hold", "M_search"} and item.get(
            "evaluation_family"
        ) not in FAMILIES:
            raise MmqaP1PrivateSelectionError("sealed evaluation stratum drifted")
    if block == "A_form":
        observed_folds = {int(item["oof_fold"]) for item in items}
        if observed_folds != set(range(5)):
            raise MmqaP1PrivateSelectionError("sealed OOF folds are incomplete")
    return observed


def open_block_gold(
    *,
    output_root: Path,
    block: str,
    authorization_path: Path,
    expected_authorization_sha256: str,
) -> dict[str, Any]:
    """Consume an archive-bound capability before touching one gold pack."""

    if block == "F_search":
        raise MmqaP1PrivateSelectionError("F_search gold is permanently sealed")
    if block not in BLOCK_ORDER:
        raise MmqaP1PrivateSelectionError("gold-open block is invalid")
    root = Path(output_root)
    receipt = _load_public_receipt(root)
    action = _pack_binding(receipt, block=block, role="action")
    gold = _pack_binding(receipt, block=block, role="gold")
    authorization_raw = _read_stable_regular_bytes(
        Path(authorization_path),
        label=f"{block} gold-open authorization",
        expected_mode=0o600,
    )
    authorization_value = _strict_json(
        authorization_raw, label=f"{block} gold-open authorization"
    )
    if (
        not isinstance(authorization_value, Mapping)
        or authorization_raw != _canonical_bytes(authorization_value, newline=True)
    ):
        raise MmqaP1PrivateSelectionError("gold-open authorization is noncanonical")
    authorization_sha256 = _validate_authorization(
        authorization_value,
        expected_authorization_sha256=expected_authorization_sha256,
        receipt=receipt,
        block=block,
        action=action,
        gold=gold,
    )
    archive_binding = _validate_action_archive_artifact(
        Path(authorization_value["action_archive_paths"][0]),
        root=root,
        receipt=receipt,
        block=block,
        expected_file_sha256=str(
            authorization_value["action_archive_sha256s"][0]
        ),
    )
    if authorization_value["action_archive_semantic_sha256s"] != [
        archive_binding["semantic_sha256"]
    ]:
        raise MmqaP1PrivateSelectionError(
            "authorized action archive semantic binding drifted"
        )
    if block == "M_search":
        promotion_binding = _validate_a_hold_promotion_receipt(
            Path(str(authorization_value["A_hold_promotion_receipt_path"])),
            root=root,
            receipt=receipt,
            expected_promotion_sha256=str(
                authorization_value["A_hold_promotion_sha256"]
            ),
            action_archive_path=Path(
                str(
                    authorization_value[
                        "A_hold_promotion_action_archive_path"
                    ]
                )
            ),
        )
        if not hmac.compare_digest(
            promotion_binding["file_sha256"],
            str(authorization_value["A_hold_promotion_file_sha256"]),
        ):
            raise MmqaP1PrivateSelectionError(
                "authorized A_hold promotion file binding drifted"
            )
    marker_body = {
        "schema": f"{VERSION}_gold_open_marker_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "authorization_consumed_immediately_before_gold_open",
        "block": block,
        "acquisition_sha256": receipt["acquisition_sha256"],
        "authorization_sha256": authorization_sha256,
        "same_block_second_open_authorized": False,
    }
    marker = self_hashed(marker_body)
    _atomic_write_json(root / GOLD_OPEN_MARKER_FILENAMES[block], marker, mode=0o600)
    pack = _read_bound_pack(root, binding=gold, label=f"{block} sealed gold pack")
    observed = _validate_gold_pack(
        pack, block=block, expected_action_sha256=str(action["semantic_sha256"])
    )
    if not hmac.compare_digest(observed, str(gold["semantic_sha256"])):
        raise MmqaP1PrivateSelectionError("authorized gold commitment drifted")
    return pack


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-acquire", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--qualification-self-sha256")
    arguments = parser.parse_args(argv)
    if (
        not arguments.formal_acquire
        or arguments.project is None
        or not isinstance(arguments.qualification_self_sha256, str)
        or _HEX64.fullmatch(arguments.qualification_self_sha256) is None
    ):
        parser.error(
            "formal execution requires --formal-acquire, --project, and a frozen "
            "--qualification-self-sha256"
        )
    run_formal_selection(
        arguments.project,
        expected_qualification_self_sha256=arguments.qualification_self_sha256,
    )
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
