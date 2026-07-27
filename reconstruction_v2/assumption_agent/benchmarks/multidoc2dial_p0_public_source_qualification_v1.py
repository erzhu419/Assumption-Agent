"""One-shot, public, non-scoring MultiDoc2Dial source qualification.

The caller supplies the single pinned official ZIP as a local regular file.
Before the ZIP central directory is opened, the complete file size and
SHA-256 are verified.  Exactly three public JSON members are then opened once:
documents, TRAIN dialogues, and VALIDATION dialogues.  The TEST member is
required by topology but its payload is never opened.

Qualification validates the public schema, dialogue history construction,
source-native input dialogue acts, and exact ``(doc_id, id_sp)`` grounding.
It writes a mode-0600 private eligibility manifest containing only opaque
identities, public domains/families, and query/dialogue hashes.  The returned
safe receipt contains aggregates and commitments only.  No secret, cohort,
action, evaluator, score, item text, identifier, or qrel value is produced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, BinaryIO
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assumption_agent.benchmarks import (  # noqa: E402
    multidoc2dial_p1_typed_core_v1 as core,
)


VERSION = "multidoc2dial_p0_public_source_qualification_v1"
STUDY_ID = "MULTIDOC2DIAL_P0_PUBLIC_SCHEMA_CAPACITY_V1"
ELIGIBILITY_RULE_VERSION = (
    "multidoc2dial_p0_input_da_history_query_group_exclusion_v2"
)
QUERY_GROUP_CONTRACT = {
    "group_field": "normalized_query_sha256",
    "maximum_selected_items_per_group": 1,
    "cross_split_overlap_policy": "exclude_every_row_before_capacity_check",
}

OFFICIAL_REPOSITORY = "https://github.com/doc2dial/multidoc2dial"
OFFICIAL_COMMIT = "6b7565989ad14858ee2c7498b605dc6d32ffe9e0"
ARCHIVE_FILENAME = "multidoc2dial.zip"
ARCHIVE_GIT_BLOB_SHA1 = "9d8dd4a24cb60ce90bb5f14730fdd1d3ca191672"
ARCHIVE_SIZE_BYTES = 6_868_509
ARCHIVE_SHA256 = (
    "f0c034c249663d7b3cb08b19cf2cc2c3d101372485be982621d4711931a1ce00"
)
TYPED_CORE_SHA256 = (
    "4ef887d64829c21f2dd5e6b344d59f78eebf8bef4b7624bb02de498a30660308"
)

ARCHIVE_ROOT = "multidoc2dial"
DOCUMENT_MEMBER = f"{ARCHIVE_ROOT}/multidoc2dial_doc.json"
TRAIN_MEMBER = f"{ARCHIVE_ROOT}/multidoc2dial_dial_train.json"
VALIDATION_MEMBER = (
    f"{ARCHIVE_ROOT}/multidoc2dial_dial_validation.json"
)
TEST_MEMBER = f"{ARCHIVE_ROOT}/multidoc2dial_dial_test.json"
REGULAR_MEMBER_WHITELIST = frozenset(
    {DOCUMENT_MEMBER, TRAIN_MEMBER, VALIDATION_MEMBER, TEST_MEMBER}
)
OPENED_MEMBER_WHITELIST = frozenset(
    {DOCUMENT_MEMBER, TRAIN_MEMBER, VALIDATION_MEMBER}
)

DOMAINS = ("dmv", "ssa", "studentaid", "va")
DOMAIN_SET = frozenset(DOMAINS)

CONDITION_QUERY = "CONDITION_QUERY"
SOLUTION_QUERY = "SOLUTION_QUERY"
POLAR_CLARIFICATION = "POLAR_CLARIFICATION"
FAMILIES = (
    CONDITION_QUERY,
    SOLUTION_QUERY,
    POLAR_CLARIFICATION,
)
FAMILY_SET = frozenset(FAMILIES)
INPUT_DA_TO_FAMILY = {
    "query_condition": CONDITION_QUERY,
    "query_solution": SOLUTION_QUERY,
    "response_positive": POLAR_CLARIFICATION,
    "response_negative": POLAR_CLARIFICATION,
}
EXPECTED_DIALOGUE_ACTS = frozenset(
    {
        "query_condition",
        "respond_solution",
        "query_solution",
        "response_positive",
        "response_negative",
        "respond_no_solution",
        "respond_solution_positive",
        "respond_solution_negative",
    }
)
MINIMUM_ELIGIBLE_FAMILY_COUNTS = {
    "TRAIN": {family: 48 for family in FAMILIES},
    "VALIDATION": {family: 24 for family in FAMILIES},
}
MINIMUM_ELIGIBLE_DOMAIN_FAMILY_COUNTS = {
    "TRAIN": {family: 12 for family in FAMILIES},
    "VALIDATION": {family: 8 for family in FAMILIES},
}
# SHA-256(raw UTF-8 dial_id) values for dialogues whose identifiers were
# already exposed while qualifying public custody.  Matching dialogues remain
# schema-validated but contribute no eligible row.
EXCLUDED_DIALOGUE_ID_SHA256 = frozenset(
    {
        "05c22c7bad20bcec2d96cc4593696a8c1d6663a777929224e9d832f008f44eeb",
        "0e93ac6e74d75eba966d31b5737ecac9189494bf54e6c711120ac809c64cd980",
        "2ad7ea806551eb64117b5c0f226e4f8f1b0eccab180532b677014dd51373446d",
        "58731f5636e7986919a110f04fdeb27094a2ed81a3771fad24aa88d76c1b7cf8",
        "6121f9879717160df66b7500d8ca60fab9dd83f3eb3744b28d891d477f66774f",
        "6cefe3571636668867b3ffe4a09f8e43df966708bbd17c93dadc8e5923fe86d6",
        "80e9914c470744e2b8b4a67357f064f9dd84395ab6e83785f08431cf2537160f",
        "8b90e0f678bd0be7308f5e96d2fbd06174225a34abc18d84eeb49de5bd9a30d7",
        "8e2a98f5fa509367389d5f712257a9b5754510b94fe39df5ae97c53becb2b27f",
        "970af506e6b60139681c588cbf710365a012db2fb4492b64c7987a3a2363bb98",
        "af4aa1816aaf6ae7efb257f65f7b39f6684d6ff5c8923318419249a606e09364",
        "b9405da4a31316bc8c071b878c135a96254b5754e88a19ad4c7c88839f96d5a8",
        "c95fea12d1d8eea93494df429f6327d3d958d2c669a6d3832b78993717e49385",
        "da5d701a58ca49513736eb8b61854d2d84c777682d62d8628c8440c9adcc7596",
        "f5355af74a3925b74bfdeb83100eb7db65d07cf5e4ce926dc71564272c9b867f",
    }
)

READ_CHUNK_BYTES = 1 << 20
MAX_TOTAL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_MEMBER_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000
MAX_JSON_DEPTH = 64
MAX_CONTAINER_ITEMS = 2_000_000
MAX_DOCUMENTS = 100_000
MAX_PASSAGES = 1_000_000
MAX_DIALOGUES_PER_SPLIT = 100_000
MAX_TURNS_PER_DIALOGUE = 1_024
MAX_IDENTIFIER_CHARACTERS = 1_024
MAX_DOCUMENT_CHARACTERS = 5_000_000
MAX_PASSAGE_CHARACTERS = 1_000_000
MAX_PATH_DEPTH = core.MAX_PATH_DEPTH
MAX_UTTERANCE_CHARACTERS = 100_000
MAX_REFERENCE_COUNT_PER_TURN = 1_024

DOCUMENT_REQUIRED_KEYS = frozenset(
    {"doc_id", "title", "doc_text", "spans"}
)
SPAN_REQUIRED_KEYS = frozenset(
    {
        "id_sp",
        "start_sp",
        "end_sp",
        "text_sp",
        "id_sec",
        "start_sec",
        "end_sec",
        "text_sec",
        "title",
        "parent_titles",
    }
)
PARENT_TITLE_KEYS = frozenset({"id_sp", "text", "level"})
DIALOGUE_REQUIRED_KEYS = frozenset({"dial_id", "turns"})
TURN_REQUIRED_KEYS = frozenset(
    {"turn_id", "role", "da", "utterance", "references"}
)
REFERENCE_REQUIRED_KEYS = frozenset({"doc_id", "id_sp"})

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MultiDoc2DialP0QualificationError(RuntimeError):
    """The frozen source or one-shot qualification contract failed closed."""


@dataclass(frozen=True)
class ArchiveContract:
    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.filename, str)
            or PurePosixPath(self.filename).name != self.filename
            or type(self.size_bytes) is not int
            or self.size_bytes < 1
            or not isinstance(self.sha256, str)
            or _HEX64.fullmatch(self.sha256) is None
        ):
            raise MultiDoc2DialP0QualificationError(
                "archive contract is invalid"
            )


OFFICIAL_ARCHIVE = ArchiveContract(
    ARCHIVE_FILENAME,
    ARCHIVE_SIZE_BYTES,
    ARCHIVE_SHA256,
)


@dataclass(frozen=True)
class _ArchiveSnapshot:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class _DocumentObservation:
    receipt: Mapping[str, Any]
    passage_keys: frozenset[tuple[str, str, str]]
    member_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _DialogueObservation:
    receipt: Mapping[str, Any]
    private_rows: tuple[Mapping[str, str], ...]
    dialogue_ids: frozenset[str]
    dialogue_hashes: frozenset[str]
    query_hashes: frozenset[str]
    item_ids: frozenset[str]
    observed_dialogue_acts: frozenset[str]
    query_preimage_fingerprints: Mapping[str, tuple[int, str]]
    qrel_cardinality_by_item: Mapping[str, int]
    member_receipt: Mapping[str, Any]


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
        raise MultiDoc2DialP0QualificationError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise MultiDoc2DialP0QualificationError(
            "body already contains a self hash"
        )
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unsupported"


def _unknown_key_bucket(key: str, value: object) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ":" + _json_type(
        value
    )


def _text(
    value: object,
    *,
    field: str,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > maximum
        or (not allow_empty and not value.strip())
    ):
        raise MultiDoc2DialP0QualificationError(f"{field} is invalid")
    return value


def _identifier(value: object, *, field: str) -> str:
    result = _text(
        value,
        field=field,
        maximum=MAX_IDENTIFIER_CHARACTERS,
    )
    if (
        result != result.strip()
        or any(ord(character) < 32 for character in result)
    ):
        raise MultiDoc2DialP0QualificationError(
            f"{field} is not canonical"
        )
    return result


def _offset(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise MultiDoc2DialP0QualificationError(
            f"{field} is not a nonnegative integer"
        )
    return value


def _turn_identifier(value: object) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    return _identifier(value, field="turn_id")


def _safe_zip_path(name: object) -> PurePosixPath:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
    ):
        raise MultiDoc2DialP0QualificationError(
            "ZIP member path is unsafe"
        )
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MultiDoc2DialP0QualificationError(
            "ZIP member path traverses the archive"
        )
    return path


def _archive_snapshot(path: Path) -> _ArchiveSnapshot:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "archive is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise MultiDoc2DialP0QualificationError(
            "archive is not a single regular file"
        )
    return _ArchiveSnapshot(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _hash_bound_archive(
    path: Path,
    contract: ArchiveContract,
) -> tuple[str, _ArchiveSnapshot]:
    before = _archive_snapshot(path)
    if before.size != contract.size_bytes:
        raise MultiDoc2DialP0QualificationError(
            "archive byte size drifted"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    observed_size = 0
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) != (before.device, before.inode, before.size):
                raise MultiDoc2DialP0QualificationError(
                    "archive changed during open"
                )
            while True:
                raw = os.read(descriptor, READ_CHUNK_BYTES)
                if not raw:
                    break
                digest.update(raw)
                observed_size += len(raw)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "archive identity read failed"
        ) from exc
    after = _archive_snapshot(path)
    if (
        after != before
        or observed_size != contract.size_bytes
        or digest.hexdigest() != contract.sha256
    ):
        raise MultiDoc2DialP0QualificationError(
            "archive byte identity drifted"
        )
    return digest.hexdigest(), before


def _git_blob_sha1_from_file(path: Path, *, size: int) -> str:
    header = f"blob {size}\0".encode("ascii")
    digest = hashlib.sha1()  # noqa: S324 - pinned Git object identity.
    digest.update(header)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            while True:
                raw = os.read(descriptor, READ_CHUNK_BYTES)
                if not raw:
                    break
                digest.update(raw)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "archive Git-blob identity read failed"
        ) from exc
    return digest.hexdigest()


def _verify_typed_core_identity() -> Mapping[str, str]:
    path = Path(core.__file__).absolute()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "frozen typed core is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise MultiDoc2DialP0QualificationError(
            "frozen typed core is not a single regular file"
        )
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
                raise MultiDoc2DialP0QualificationError(
                    "frozen typed core changed during open"
                )
            while True:
                raw = os.read(descriptor, READ_CHUNK_BYTES)
                if not raw:
                    break
                digest.update(raw)
                size += len(raw)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "frozen typed core identity read failed"
        ) from exc
    observed = digest.hexdigest()
    if (
        size != metadata.st_size
        or observed != TYPED_CORE_SHA256
        or core.VERSION != "multidoc2dial_p1_typed_core_v1"
    ):
        raise MultiDoc2DialP0QualificationError(
            "frozen typed core identity drifted"
        )
    return {
        "version": core.VERSION,
        "sha256": observed,
    }


def _validate_topology(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, zipfile.ZipInfo], Mapping[str, Any]]:
    regular: dict[str, zipfile.ZipInfo] = {}
    directory_names: set[str] = set()
    casefold_names: set[str] = set()
    total_uncompressed = 0
    total_compressed = 0
    for info in archive.infolist():
        path = _safe_zip_path(info.filename.rstrip("/"))
        canonical_name = path.as_posix() + ("/" if info.is_dir() else "")
        folded = canonical_name.casefold()
        if folded in casefold_names:
            raise MultiDoc2DialP0QualificationError(
                "ZIP contains duplicate or case-colliding names"
            )
        casefold_names.add(folded)
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if info.flag_bits & 0x1:
            raise MultiDoc2DialP0QualificationError(
                "ZIP contains an encrypted member"
            )
        if info.is_dir():
            if file_type not in {0, stat.S_IFDIR}:
                raise MultiDoc2DialP0QualificationError(
                    "ZIP directory type drifted"
                )
            directory_names.add(path.as_posix() + "/")
            continue
        if file_type not in {0, stat.S_IFREG}:
            raise MultiDoc2DialP0QualificationError(
                "ZIP contains a link or special member"
            )
        if info.filename not in REGULAR_MEMBER_WHITELIST:
            raise MultiDoc2DialP0QualificationError(
                "ZIP regular-member whitelist drifted"
            )
        if info.filename in regular:
            raise MultiDoc2DialP0QualificationError(
                "ZIP target member is duplicated"
            )
        if (
            type(info.file_size) is not int
            or type(info.compress_size) is not int
            or info.file_size <= 0
            or info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES
            or info.compress_size <= 0
            or (
                info.file_size
                > max(1, info.compress_size) * MAX_COMPRESSION_RATIO
            )
            or info.compress_type
            not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
        ):
            raise MultiDoc2DialP0QualificationError(
                "ZIP member size or compression is unsafe"
            )
        regular[info.filename] = info
        total_uncompressed += info.file_size
        total_compressed += info.compress_size
    if set(regular) != REGULAR_MEMBER_WHITELIST:
        raise MultiDoc2DialP0QualificationError(
            "ZIP does not contain the exact four-member whitelist"
        )
    allowed_directories = {
        prefix
        for member in REGULAR_MEMBER_WHITELIST
        for prefix in (
            "/".join(member.split("/")[:index]) + "/"
            for index in range(1, len(member.split("/")))
        )
    }
    if not directory_names <= allowed_directories:
        raise MultiDoc2DialP0QualificationError(
            "ZIP contains a non-whitelisted directory"
        )
    if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise MultiDoc2DialP0QualificationError(
            "ZIP total uncompressed size is unsafe"
        )
    topology = {
        "central_directory_entry_count": len(archive.infolist()),
        "directory_entry_count": len(directory_names),
        "regular_member_count": len(regular),
        "regular_member_paths": sorted(regular),
        "total_compressed_bytes": total_compressed,
        "total_uncompressed_bytes": total_uncompressed,
        "test_payload_open_count": 0,
    }
    return regular, topology


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        raw = self._source.read(size)
        if not isinstance(raw, bytes):
            raise MultiDoc2DialP0QualificationError(
                "ZIP member reader returned non-bytes"
            )
        self._digest.update(raw)
        self.size += len(raw)
        return raw

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _ijson_basic_parse(source: BinaryIO) -> Iterator[tuple[str, Any]]:
    try:
        import ijson
    except ImportError as exc:
        raise MultiDoc2DialP0QualificationError(
            "frozen ijson runtime is unavailable"
        ) from exc
    try:
        yield from ijson.basic_parse(source, use_float=False)
    except BaseException as exc:
        if isinstance(exc, MultiDoc2DialP0QualificationError):
            raise
        raise MultiDoc2DialP0QualificationError(
            "streaming JSON parse failed"
        ) from exc


def _next_event(
    events: Iterator[tuple[str, Any]],
    *,
    context: str,
) -> tuple[str, Any]:
    try:
        return next(events)
    except StopIteration as exc:
        raise MultiDoc2DialP0QualificationError(
            f"{context} ended unexpectedly"
        ) from exc


def _read_stream_value(
    events: Iterator[tuple[str, Any]],
    first: tuple[str, Any],
    *,
    depth: int,
) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise MultiDoc2DialP0QualificationError(
            "JSON nesting exceeds the frozen bound"
        )
    event, value = first
    if event == "start_map":
        result: dict[str, Any] = {}
        while True:
            child_event, child_value = _next_event(
                events,
                context="JSON object",
            )
            if child_event == "end_map":
                return result
            if (
                child_event != "map_key"
                or not isinstance(child_value, str)
                or child_value in result
            ):
                raise MultiDoc2DialP0QualificationError(
                    "JSON object contains a duplicate or invalid key"
                )
            if len(result) >= MAX_CONTAINER_ITEMS:
                raise MultiDoc2DialP0QualificationError(
                    "JSON object exceeds the frozen item bound"
                )
            result[child_value] = _read_stream_value(
                events,
                _next_event(events, context="JSON object value"),
                depth=depth + 1,
            )
    if event == "start_array":
        result_list: list[Any] = []
        while True:
            child = _next_event(events, context="JSON array")
            if child[0] == "end_array":
                return result_list
            if len(result_list) >= MAX_CONTAINER_ITEMS:
                raise MultiDoc2DialP0QualificationError(
                    "JSON array exceeds the frozen item bound"
                )
            result_list.append(
                _read_stream_value(events, child, depth=depth + 1)
            )
    if event == "string":
        if not isinstance(value, str):
            raise MultiDoc2DialP0QualificationError(
                "JSON string event is invalid"
            )
        return value
    if event == "number":
        if type(value) is not int:
            raise MultiDoc2DialP0QualificationError(
                "JSON contains a non-integer number"
            )
        return value
    if event == "boolean":
        if type(value) is not bool:
            raise MultiDoc2DialP0QualificationError(
                "JSON boolean event is invalid"
            )
        return value
    if event == "null":
        return None
    raise MultiDoc2DialP0QualificationError(
        "JSON contains an unexpected structural event"
    )


def _iter_root_domains(
    source: BinaryIO,
    *,
    root_key: str,
) -> Iterator[tuple[str, Any]]:
    events = iter(_ijson_basic_parse(source))
    if _next_event(events, context="JSON root") != ("start_map", None):
        raise MultiDoc2DialP0QualificationError(
            "JSON root is not an object"
        )
    if _next_event(events, context="JSON root key") != (
        "map_key",
        root_key,
    ):
        raise MultiDoc2DialP0QualificationError(
            "JSON root key drifted"
        )
    if _next_event(events, context="domain registry") != (
        "start_map",
        None,
    ):
        raise MultiDoc2DialP0QualificationError(
            "domain registry is not an object"
        )
    observed_domains: set[str] = set()
    while True:
        event, value = _next_event(events, context="domain registry")
        if event == "end_map":
            break
        if (
            event != "map_key"
            or not isinstance(value, str)
            or value in observed_domains
        ):
            raise MultiDoc2DialP0QualificationError(
                "domain registry key is duplicated or invalid"
            )
        observed_domains.add(value)
        yield value, _read_stream_value(
            events,
            _next_event(events, context="domain value"),
            depth=1,
        )
    if observed_domains != DOMAIN_SET:
        raise MultiDoc2DialP0QualificationError(
            "source does not contain the exact four-domain registry"
        )
    if _next_event(events, context="JSON root") != ("end_map", None):
        raise MultiDoc2DialP0QualificationError(
            "JSON root contains an extra field"
        )
    try:
        next(events)
    except StopIteration:
        return
    raise MultiDoc2DialP0QualificationError(
        "JSON contains trailing values"
    )


def _member_receipt(
    reader: _HashingReader,
    info: zipfile.ZipInfo,
) -> Mapping[str, Any]:
    if reader.size != info.file_size:
        raise MultiDoc2DialP0QualificationError(
            "JSON parser did not consume the complete member"
        )
    return {
        "content_sha256": reader.sha256,
        "crc32": f"{info.CRC:08x}",
        "size_bytes": reader.size,
    }


def _observe_documents(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> _DocumentObservation:
    passage_keys: set[tuple[str, str, str]] = set()
    global_document_keys: set[tuple[str, str]] = set()
    per_domain: dict[str, Any] = {}
    unknown_document_keys: Counter[str] = Counter()
    unknown_span_keys: Counter[str] = Counter()
    passage_length_histogram: Counter[str] = Counter()
    parent_path_depth_histogram: Counter[int] = Counter()
    global_section_count = 0
    with archive.open(info, "r") as member:
        reader = _HashingReader(member)
        for domain, documents in _iter_root_domains(
            reader,
            root_key="doc_data",
        ):
            if not isinstance(documents, dict) or not documents:
                raise MultiDoc2DialP0QualificationError(
                    "domain document registry is not a nonempty object"
                )
            document_count = 0
            passage_count = 0
            section_count = 0
            for document_key, document in documents.items():
                if len(global_document_keys) >= MAX_DOCUMENTS:
                    raise MultiDoc2DialP0QualificationError(
                        "document count exceeds the frozen bound"
                    )
                source_document_id = _identifier(
                    document_key,
                    field="document map key",
                )
                if not isinstance(document, dict) or not (
                    DOCUMENT_REQUIRED_KEYS <= set(document)
                ):
                    raise MultiDoc2DialP0QualificationError(
                        "document schema drifted"
                    )
                document_id = _identifier(
                    document.get("doc_id"),
                    field="doc_id",
                )
                if document_id != source_document_id:
                    raise MultiDoc2DialP0QualificationError(
                        "document map key and doc_id disagree"
                    )
                document_identity = (domain, document_id)
                if document_identity in global_document_keys:
                    raise MultiDoc2DialP0QualificationError(
                        "document identity is duplicated"
                    )
                global_document_keys.add(document_identity)
                _text(
                    document.get("title"),
                    field="document title",
                    maximum=MAX_DOCUMENT_CHARACTERS,
                )
                document_text = _text(
                    document.get("doc_text"),
                    field="doc_text",
                    maximum=MAX_DOCUMENT_CHARACTERS,
                )
                spans = document.get("spans")
                if not isinstance(spans, dict) or not spans:
                    raise MultiDoc2DialP0QualificationError(
                        "document spans are not a nonempty object"
                    )
                section_registry: dict[str, tuple[object, ...]] = {}
                for span_key, span in spans.items():
                    if len(passage_keys) >= MAX_PASSAGES:
                        raise MultiDoc2DialP0QualificationError(
                            "passage count exceeds the frozen bound"
                        )
                    source_span_id = _identifier(
                        span_key,
                        field="span map key",
                    )
                    if not isinstance(span, dict) or not (
                        SPAN_REQUIRED_KEYS <= set(span)
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "span schema drifted"
                        )
                    span_id = _identifier(
                        span.get("id_sp"),
                        field="id_sp",
                    )
                    if span_id != source_span_id:
                        raise MultiDoc2DialP0QualificationError(
                            "span map key and id_sp disagree"
                        )
                    start = _offset(
                        span.get("start_sp"),
                        field="start_sp",
                    )
                    end = _offset(
                        span.get("end_sp"),
                        field="end_sp",
                    )
                    passage_text = _text(
                        span.get("text_sp"),
                        field="text_sp",
                        maximum=MAX_PASSAGE_CHARACTERS,
                    )
                    section_id = _identifier(
                        span.get("id_sec"),
                        field="id_sec",
                    )
                    section_start = _offset(
                        span.get("start_sec"),
                        field="start_sec",
                    )
                    section_end = _offset(
                        span.get("end_sec"),
                        field="end_sec",
                    )
                    section_text = _text(
                        span.get("text_sec"),
                        field="text_sec",
                        maximum=MAX_DOCUMENT_CHARACTERS,
                    )
                    section_title = _text(
                        span.get("title"),
                        field="span title",
                        maximum=MAX_PASSAGE_CHARACTERS,
                        allow_empty=True,
                    )
                    parent_titles = span.get("parent_titles")
                    if (
                        not isinstance(parent_titles, list)
                        or len(parent_titles) > MAX_PATH_DEPTH
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "parent_titles is not a bounded array"
                        )
                    validated_parent_titles: list[
                        tuple[str, str, str]
                    ] = []
                    parent_span_ids: set[str] = set()
                    for parent in parent_titles:
                        if (
                            not isinstance(parent, dict)
                            or set(parent) != PARENT_TITLE_KEYS
                        ):
                            raise MultiDoc2DialP0QualificationError(
                                "parent_titles object schema drifted"
                            )
                        parent_span_id = _identifier(
                            parent.get("id_sp"),
                            field="parent title id_sp",
                        )
                        if parent_span_id in parent_span_ids:
                            raise MultiDoc2DialP0QualificationError(
                                "parent_titles contains a duplicate id_sp"
                            )
                        parent_span_ids.add(parent_span_id)
                        parent_text = _text(
                            parent.get("text"),
                            field="parent title text",
                            maximum=MAX_PASSAGE_CHARACTERS,
                            allow_empty=True,
                        )
                        parent_level = _identifier(
                            parent.get("level"),
                            field="parent title level",
                        )
                        validated_parent_titles.append(
                            (
                                parent_span_id,
                                parent_text,
                                parent_level,
                            )
                        )
                    if (
                        not 0 <= start < end <= len(document_text)
                        or document_text[start:end] != passage_text
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "passage offsets do not exactly map doc_text"
                        )
                    if (
                        not 0
                        <= section_start
                        < section_end
                        <= len(document_text)
                        or document_text[section_start:section_end]
                        != section_text
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "section offsets do not exactly map doc_text"
                        )
                    if not (
                        section_start <= start < end <= section_end
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "passage interval is outside its section"
                        )
                    section_signature: tuple[object, ...] = (
                        section_start,
                        section_end,
                        section_text,
                        section_title,
                        tuple(validated_parent_titles),
                    )
                    existing_section = section_registry.get(section_id)
                    if (
                        existing_section is not None
                        and existing_section != section_signature
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "section coordinates are inconsistent"
                        )
                    if existing_section is None:
                        section_registry[section_id] = section_signature
                        section_count += 1
                        global_section_count += 1
                    passage_identity = (domain, document_id, span_id)
                    if passage_identity in passage_keys:
                        raise MultiDoc2DialP0QualificationError(
                            "passage identity is duplicated"
                        )
                    passage_keys.add(passage_identity)
                    passage_count += 1
                    length = len(passage_text)
                    if length < 128:
                        bucket = "0_127"
                    elif length < 512:
                        bucket = "128_511"
                    elif length < 2_048:
                        bucket = "512_2047"
                    else:
                        bucket = "2048_plus"
                    passage_length_histogram[bucket] += 1
                    parent_path_depth_histogram[
                        len(validated_parent_titles)
                    ] += 1
                    for key in set(span) - SPAN_REQUIRED_KEYS:
                        unknown_span_keys[
                            _unknown_key_bucket(key, span[key])
                        ] += 1
                document_count += 1
                for key in set(document) - DOCUMENT_REQUIRED_KEYS:
                    unknown_document_keys[
                        _unknown_key_bucket(key, document[key])
                    ] += 1
            per_domain[domain] = {
                "document_count": document_count,
                "passage_count": passage_count,
                "section_count": section_count,
            }
        receipt = _member_receipt(reader, info)
    return _DocumentObservation(
        receipt={
            "domain": {key: per_domain[key] for key in sorted(per_domain)},
            "document_count": len(global_document_keys),
            "passage_count": len(passage_keys),
            "section_count": global_section_count,
            "passage_length_histogram": _counter(
                passage_length_histogram
            ),
            "parent_path_depth_histogram": _counter(
                parent_path_depth_histogram
            ),
            "unknown_document_key_buckets": _counter(
                unknown_document_keys
            ),
            "unknown_span_key_buckets": _counter(unknown_span_keys),
        },
        passage_keys=frozenset(passage_keys),
        member_receipt=receipt,
    )


def _dialogue_hash(domain: str, dialogue_id: str) -> str:
    return stable_hash(
        {"domain": domain, "opaque_dialogue_id": dialogue_id}
    )


def _opaque_item_id(
    *,
    split: str,
    domain: str,
    dialogue_id: str,
    input_turn_id: str,
    response_turn_id: str,
) -> str:
    return stable_hash(
        {
            "split": split,
            "domain": domain,
            "opaque_dialogue_id": dialogue_id,
            "opaque_input_turn_id": input_turn_id,
            "opaque_response_turn_id": response_turn_id,
        }
    )


def _reference_pairs(
    references: object,
    *,
    domain: str,
    passage_keys: frozenset[tuple[str, str, str]],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(references, list):
        raise MultiDoc2DialP0QualificationError(
            "turn references are not an array"
        )
    if len(references) > MAX_REFERENCE_COUNT_PER_TURN:
        raise MultiDoc2DialP0QualificationError(
            "turn reference count exceeds the frozen bound"
        )
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for reference in references:
        if not isinstance(reference, dict) or not (
            REFERENCE_REQUIRED_KEYS <= set(reference)
        ):
            raise MultiDoc2DialP0QualificationError(
                "reference schema drifted"
            )
        document_id = _identifier(
            reference.get("doc_id"),
            field="reference doc_id",
        )
        span_id = _identifier(
            reference.get("id_sp"),
            field="reference id_sp",
        )
        if "label" in reference:
            _text(
                reference.get("label"),
                field="reference label",
                maximum=MAX_IDENTIFIER_CHARACTERS,
                allow_empty=True,
            )
        pair = (document_id, span_id)
        if pair in seen:
            raise MultiDoc2DialP0QualificationError(
                "turn contains a duplicate reference"
            )
        if (domain, document_id, span_id) not in passage_keys:
            raise MultiDoc2DialP0QualificationError(
                "reference does not exactly map a domain passage"
            )
        seen.add(pair)
        pairs.append(pair)
    return tuple(pairs)


def _observe_dialogues(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    split: str,
    passage_keys: frozenset[tuple[str, str, str]],
) -> _DialogueObservation:
    dialogue_ids: set[str] = set()
    dialogue_hashes: set[str] = set()
    query_hashes: set[str] = set()
    query_preimage_fingerprints: dict[str, tuple[int, str]] = {}
    item_ids: set[str] = set()
    observed_dialogue_acts: set[str] = set()
    private_rows: list[Mapping[str, str]] = []
    qrel_cardinality_by_item: dict[str, int] = {}
    domain_receipts: dict[str, Any] = {}
    split_family_counts: Counter[str] = Counter()
    qrel_cardinality: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    dialogue_act_counts: Counter[str] = Counter()
    unknown_dialogue_keys: Counter[str] = Counter()
    unknown_turn_keys: Counter[str] = Counter()
    unknown_reference_keys: Counter[str] = Counter()
    total_dialogues = 0
    total_turns = 0
    excluded_dialogue_count = 0
    excluded_eligible_row_count = 0
    excluded_family_counts: Counter[str] = Counter()
    with archive.open(info, "r") as member:
        reader = _HashingReader(member)
        for domain, dialogues in _iter_root_domains(
            reader,
            root_key="dial_data",
        ):
            if not isinstance(dialogues, list) or not dialogues:
                raise MultiDoc2DialP0QualificationError(
                    "domain dialogue registry is not a nonempty array"
                )
            domain_dialogue_count = 0
            domain_turn_count = 0
            domain_family_counts: Counter[str] = Counter()
            domain_excluded_dialogues = 0
            domain_excluded_family_counts: Counter[str] = Counter()
            for dialogue in dialogues:
                if total_dialogues >= MAX_DIALOGUES_PER_SPLIT:
                    raise MultiDoc2DialP0QualificationError(
                        "dialogue count exceeds the frozen bound"
                    )
                if not isinstance(dialogue, dict) or not (
                    DIALOGUE_REQUIRED_KEYS <= set(dialogue)
                ):
                    raise MultiDoc2DialP0QualificationError(
                        "dialogue schema drifted"
                    )
                dialogue_id = _identifier(
                    dialogue.get("dial_id"),
                    field="dial_id",
                )
                if dialogue_id in dialogue_ids:
                    raise MultiDoc2DialP0QualificationError(
                        "dial_id is duplicated within a split"
                    )
                dialogue_ids.add(dialogue_id)
                dialogue_is_excluded = (
                    hashlib.sha256(dialogue_id.encode("utf-8")).hexdigest()
                    in EXCLUDED_DIALOGUE_ID_SHA256
                )
                if dialogue_is_excluded:
                    excluded_dialogue_count += 1
                    domain_excluded_dialogues += 1
                group_hash = _dialogue_hash(domain, dialogue_id)
                if group_hash in dialogue_hashes:
                    raise MultiDoc2DialP0QualificationError(
                        "dialogue grouping hash collided"
                    )
                dialogue_hashes.add(group_hash)
                turns = dialogue.get("turns")
                if (
                    not isinstance(turns, list)
                    or not turns
                    or len(turns) > MAX_TURNS_PER_DIALOGUE
                ):
                    raise MultiDoc2DialP0QualificationError(
                        "dialogue turns are outside the frozen bound"
                    )
                validated_turns: list[
                    tuple[str, str, str, str, tuple[tuple[str, str], ...]]
                ] = []
                turn_ids: set[str] = set()
                for index, turn in enumerate(turns):
                    if not isinstance(turn, dict) or not (
                        TURN_REQUIRED_KEYS <= set(turn)
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "turn schema drifted"
                        )
                    turn_id = _turn_identifier(turn.get("turn_id"))
                    if turn_id in turn_ids:
                        raise MultiDoc2DialP0QualificationError(
                            "turn_id is duplicated within a dialogue"
                        )
                    turn_ids.add(turn_id)
                    role = _identifier(turn.get("role"), field="role")
                    if role not in {"user", "agent"}:
                        raise MultiDoc2DialP0QualificationError(
                            "turn role is outside the public registry"
                        )
                    if index == 0 and role != "user":
                        raise MultiDoc2DialP0QualificationError(
                            "dialogue does not begin with a user turn"
                        )
                    dialogue_act = _identifier(
                        turn.get("da"),
                        field="dialogue act",
                    )
                    if dialogue_act not in EXPECTED_DIALOGUE_ACTS:
                        raise MultiDoc2DialP0QualificationError(
                            "dialogue act is outside the public registry"
                        )
                    observed_dialogue_acts.add(dialogue_act)
                    role_counts[role] += 1
                    dialogue_act_counts[dialogue_act] += 1
                    utterance = _text(
                        turn.get("utterance"),
                        field="utterance",
                        maximum=MAX_UTTERANCE_CHARACTERS,
                    )
                    references = _reference_pairs(
                        turn.get("references"),
                        domain=domain,
                        passage_keys=passage_keys,
                    )
                    validated_turns.append(
                        (
                            turn_id,
                            role,
                            dialogue_act,
                            utterance,
                            references,
                        )
                    )
                    for key in set(turn) - TURN_REQUIRED_KEYS:
                        unknown_turn_keys[
                            _unknown_key_bucket(key, turn[key])
                        ] += 1
                    for reference in turn["references"]:
                        for key in set(reference) - REFERENCE_REQUIRED_KEYS:
                            unknown_reference_keys[
                                _unknown_key_bucket(key, reference[key])
                            ] += 1
                history: list[core.DialogueTurn] = []
                for index, (
                    turn_id,
                    role,
                    dialogue_act,
                    utterance,
                    _references,
                ) in enumerate(validated_turns):
                    history.append(core.DialogueTurn(role=role, text=utterance))
                    if role != "user" or index + 1 >= len(validated_turns):
                        continue
                    (
                        response_turn_id,
                        response_role,
                        response_da,
                        _response_text,
                        response_references,
                    ) = validated_turns[index + 1]
                    structurally_eligible = (
                        response_role == "agent"
                        and response_da != "respond_no_solution"
                        and bool(response_references)
                    )
                    if not structurally_eligible:
                        continue
                    family = INPUT_DA_TO_FAMILY.get(dialogue_act)
                    if family is None:
                        raise MultiDoc2DialP0QualificationError(
                            "an unregistered user dialogue act entered "
                            "eligibility"
                        )
                    try:
                        normalized_payload = core.normalized_query_payload(
                            tuple(history)
                        )
                        normalized_bytes = core.canonical_bytes(
                            normalized_payload
                        )
                        query_hash = hashlib.sha256(
                            normalized_bytes
                        ).hexdigest()
                        if query_hash != core.normalized_query_sha256(
                            tuple(history)
                        ):
                            raise MultiDoc2DialP0QualificationError(
                                "typed query hash implementation drifted"
                            )
                    except core.MultiDoc2DialP1TypedCoreError as exc:
                        raise MultiDoc2DialP0QualificationError(
                            "history query violates the frozen typed core"
                        ) from exc
                    query_fingerprint = (
                        len(normalized_bytes),
                        hashlib.sha512(normalized_bytes).hexdigest(),
                    )
                    prior_fingerprint = query_preimage_fingerprints.get(
                        query_hash
                    )
                    if (
                        prior_fingerprint is not None
                        and prior_fingerprint != query_fingerprint
                    ):
                        raise MultiDoc2DialP0QualificationError(
                            "normalized query SHA-256 digest collided"
                        )
                    query_preimage_fingerprints[
                        query_hash
                    ] = query_fingerprint
                    item_id = _opaque_item_id(
                        split=split,
                        domain=domain,
                        dialogue_id=dialogue_id,
                        input_turn_id=turn_id,
                        response_turn_id=response_turn_id,
                    )
                    if item_id in item_ids:
                        raise MultiDoc2DialP0QualificationError(
                            "opaque item identity collides"
                        )
                    item_ids.add(item_id)
                    if dialogue_is_excluded:
                        excluded_eligible_row_count += 1
                        excluded_family_counts[family] += 1
                        domain_excluded_family_counts[family] += 1
                        continue
                    query_hashes.add(query_hash)
                    private_rows.append(
                        {
                            "opaque_item_id": item_id,
                            "domain": domain,
                            "family": family,
                            "normalized_query_sha256": query_hash,
                            "dialogue_sha256": group_hash,
                        }
                    )
                    split_family_counts[family] += 1
                    domain_family_counts[family] += 1
                    qrel_cardinality[len(response_references)] += 1
                    qrel_cardinality_by_item[item_id] = len(
                        response_references
                    )
                total_dialogues += 1
                total_turns += len(validated_turns)
                domain_dialogue_count += 1
                domain_turn_count += len(validated_turns)
                for key in set(dialogue) - DIALOGUE_REQUIRED_KEYS:
                    unknown_dialogue_keys[
                        _unknown_key_bucket(key, dialogue[key])
                    ] += 1
            domain_receipts[domain] = {
                "dialogue_count": domain_dialogue_count,
                "turn_count": domain_turn_count,
                "eligible_family_count": {
                    family: domain_family_counts[family]
                    for family in FAMILIES
                },
                "custody_excluded_dialogue_count": (
                    domain_excluded_dialogues
                ),
                "custody_excluded_eligible_family_count": {
                    family: domain_excluded_family_counts[family]
                    for family in FAMILIES
                },
            }
        member_receipt = _member_receipt(reader, info)
    duplicate_query_counts = Counter(
        row["normalized_query_sha256"] for row in private_rows
    )
    duplicate_groups = {
        query_hash: count
        for query_hash, count in duplicate_query_counts.items()
        if count > 1
    }
    return _DialogueObservation(
        receipt={
            "dialogue_count": total_dialogues,
            "turn_count": total_turns,
            "domain": {
                key: domain_receipts[key]
                for key in sorted(domain_receipts)
            },
            "role_count": _counter(role_counts),
            "dialogue_act_count": _counter(dialogue_act_counts),
            "eligible_family_count": {
                family: split_family_counts[family]
                for family in FAMILIES
            },
            "qrel_cardinality_histogram": _counter(qrel_cardinality),
            "normalized_query_grouping": {
                "group_count": len(duplicate_query_counts),
                "duplicate_group_count": len(duplicate_groups),
                "duplicate_row_count": sum(
                    duplicate_groups.values()
                ),
                "excess_duplicate_row_count": sum(
                    count - 1 for count in duplicate_groups.values()
                ),
                "maximum_selected_items_per_group": 1,
            },
            "custody_exclusion": {
                "excluded_dialogue_count": excluded_dialogue_count,
                "excluded_eligible_row_count": (
                    excluded_eligible_row_count
                ),
                "excluded_eligible_family_count": {
                    family: excluded_family_counts[family]
                    for family in FAMILIES
                },
            },
            "collision_count": {
                "dial_id": 0,
                "dialogue_group_hash": 0,
                "normalized_query_digest": 0,
                "opaque_item_id": 0,
                "reference_pair_within_turn": 0,
            },
            "unknown_dialogue_key_buckets": _counter(
                unknown_dialogue_keys
            ),
            "unknown_turn_key_buckets": _counter(unknown_turn_keys),
            "unknown_reference_key_buckets": _counter(
                unknown_reference_keys
            ),
        },
        private_rows=tuple(
            sorted(
                private_rows,
                key=lambda row: (
                    row["family"],
                    row["domain"],
                    row["opaque_item_id"],
                ),
            )
        ),
        dialogue_ids=frozenset(dialogue_ids),
        dialogue_hashes=frozenset(dialogue_hashes),
        query_hashes=frozenset(query_hashes),
        item_ids=frozenset(item_ids),
        observed_dialogue_acts=frozenset(observed_dialogue_acts),
        query_preimage_fingerprints=dict(query_preimage_fingerprints),
        qrel_cardinality_by_item=dict(qrel_cardinality_by_item),
        member_receipt=member_receipt,
    )


def _require_fresh_file(path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise MultiDoc2DialP0QualificationError(
            f"{label} output is not fresh"
        )
    parent = path.absolute().parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or not os.access(parent, os.W_OK | os.X_OK)
    ):
        raise MultiDoc2DialP0QualificationError(
            f"{label} output parent is unsafe"
        )


def write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    raw = canonical_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise MultiDoc2DialP0QualificationError(
                        "exclusive JSON write stalled"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "exclusive JSON write failed"
        ) from exc


def _verify_private_manifest(
    path: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "private eligibility manifest disappeared"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise MultiDoc2DialP0QualificationError(
            "private eligibility manifest metadata drifted"
        )
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            while True:
                raw = os.read(descriptor, READ_CHUNK_BYTES)
                if not raw:
                    break
                digest.update(raw)
                size += len(raw)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise MultiDoc2DialP0QualificationError(
            "private eligibility manifest verification failed"
        ) from exc
    if size != metadata.st_size:
        raise MultiDoc2DialP0QualificationError(
            "private eligibility manifest size drifted"
        )
    return {
        "file_sha256": digest.hexdigest(),
        "self_sha256": manifest["self_sha256"],
        "size_bytes": size,
        "row_count": {
            split: len(manifest["eligible_rows_by_split"][split])
            for split in ("TRAIN", "VALIDATION")
        },
    }


def _eligible_counts(
    rows: Sequence[Mapping[str, str]],
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    family_counts: Counter[str] = Counter()
    domain_counts = {domain: Counter() for domain in DOMAINS}
    for row in rows:
        family = row["family"]
        domain = row["domain"]
        if family not in FAMILY_SET or domain not in DOMAIN_SET:
            raise MultiDoc2DialP0QualificationError(
                "private eligibility row registry drifted"
            )
        family_counts[family] += 1
        domain_counts[domain][family] += 1
    return family_counts, domain_counts


def _validate_final_capacity(
    *,
    split: str,
    rows: Sequence[Mapping[str, str]],
) -> None:
    family_counts, domain_counts = _eligible_counts(rows)
    minimums = MINIMUM_ELIGIBLE_FAMILY_COUNTS[split]
    deficient = {
        family: {
            "observed": family_counts[family],
            "required": minimums[family],
        }
        for family in FAMILIES
        if family_counts[family] < minimums[family]
    }
    if deficient:
        raise MultiDoc2DialP0QualificationError(
            "eligible source-native family capacity is insufficient: "
            + canonical_bytes(deficient).decode("ascii")
        )
    domain_minimums = MINIMUM_ELIGIBLE_DOMAIN_FAMILY_COUNTS[split]
    domain_deficient = {
        domain: {
            family: {
                "observed": domain_counts[domain][family],
                "required": domain_minimums[family],
            }
            for family in FAMILIES
            if domain_counts[domain][family] < domain_minimums[family]
        }
        for domain in DOMAINS
    }
    domain_deficient = {
        domain: value
        for domain, value in domain_deficient.items()
        if value
    }
    if domain_deficient:
        raise MultiDoc2DialP0QualificationError(
            "eligible domain-by-family capacity is insufficient: "
            + canonical_bytes(domain_deficient).decode("ascii")
        )


def _query_group_aggregate(
    rows: Sequence[Mapping[str, str]],
) -> Mapping[str, int]:
    counts = Counter(row["normalized_query_sha256"] for row in rows)
    duplicate_counts = [count for count in counts.values() if count > 1]
    return {
        "group_count": len(counts),
        "duplicate_group_count": len(duplicate_counts),
        "duplicate_row_count": sum(duplicate_counts),
        "excess_duplicate_row_count": sum(
            count - 1 for count in duplicate_counts
        ),
        "maximum_selected_items_per_group": 1,
    }


def _final_dialogue_receipt(
    observation: _DialogueObservation,
    rows: Sequence[Mapping[str, str]],
) -> Mapping[str, Any]:
    family_counts, domain_counts = _eligible_counts(rows)
    result = dict(observation.receipt)
    result["pre_cross_split_eligible_row_count"] = len(
        observation.private_rows
    )
    result["eligible_row_count"] = len(rows)
    result["eligible_family_count"] = {
        family: family_counts[family] for family in FAMILIES
    }
    result["normalized_query_grouping"] = _query_group_aggregate(rows)
    qrel_cardinality = Counter(
        observation.qrel_cardinality_by_item[row["opaque_item_id"]]
        for row in rows
    )
    result["qrel_cardinality_histogram"] = _counter(qrel_cardinality)
    domain_receipts: dict[str, Any] = {}
    original_domains = observation.receipt["domain"]
    if not isinstance(original_domains, Mapping):
        raise MultiDoc2DialP0QualificationError(
            "dialogue aggregate domain registry drifted"
        )
    for domain in DOMAINS:
        domain_receipt = dict(original_domains[domain])
        domain_receipt["eligible_family_count"] = {
            family: domain_counts[domain][family]
            for family in FAMILIES
        }
        domain_receipts[domain] = domain_receipt
    result["domain"] = domain_receipts
    return result


def _cross_split_exclusion_aggregate(
    *,
    rows: Sequence[Mapping[str, str]],
    overlap_hashes: frozenset[str],
) -> Mapping[str, Any]:
    excluded = [
        row
        for row in rows
        if row["normalized_query_sha256"] in overlap_hashes
    ]
    family_counts, domain_counts = _eligible_counts(excluded)
    return {
        "excluded_row_count": len(excluded),
        "excluded_family_count": {
            family: family_counts[family] for family in FAMILIES
        },
        "excluded_domain_family_count": {
            domain: {
                family: domain_counts[domain][family]
                for family in FAMILIES
            }
            for domain in DOMAINS
        },
    }


def qualify_archive(
    *,
    archive_path: Path,
    eligibility_manifest_path: Path,
    archive_contract: ArchiveContract = OFFICIAL_ARCHIVE,
) -> dict[str, Any]:
    """Qualify one local ZIP and persist only the private eligibility rows."""

    archive_path = archive_path.absolute()
    eligibility_manifest_path = eligibility_manifest_path.absolute()
    _require_fresh_file(
        eligibility_manifest_path,
        label="private eligibility manifest",
    )
    typed_core_binding = _verify_typed_core_identity()
    custody_exclusion_binding = {
        "definition": "sha256_of_exact_validated_dial_id_utf8_bytes",
        "count": len(EXCLUDED_DIALOGUE_ID_SHA256),
        "set_sha256": stable_hash(
            sorted(EXCLUDED_DIALOGUE_ID_SHA256)
        ),
    }
    archive_sha256, original_snapshot = _hash_bound_archive(
        archive_path,
        archive_contract,
    )
    git_blob_sha1 = _git_blob_sha1_from_file(
        archive_path,
        size=archive_contract.size_bytes,
    )
    if (
        archive_contract is OFFICIAL_ARCHIVE
        and git_blob_sha1 != ARCHIVE_GIT_BLOB_SHA1
    ):
        raise MultiDoc2DialP0QualificationError(
            "official archive Git-blob identity drifted"
        )
    try:
        archive = zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise MultiDoc2DialP0QualificationError(
            "archive is not a valid ZIP"
        ) from exc
    try:
        regular, topology = _validate_topology(archive)
        documents = _observe_documents(
            archive,
            regular[DOCUMENT_MEMBER],
        )
        train = _observe_dialogues(
            archive,
            regular[TRAIN_MEMBER],
            split="TRAIN",
            passage_keys=documents.passage_keys,
        )
        validation = _observe_dialogues(
            archive,
            regular[VALIDATION_MEMBER],
            split="VALIDATION",
            passage_keys=documents.passage_keys,
        )
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, MultiDoc2DialP0QualificationError):
            raise
        raise MultiDoc2DialP0QualificationError(
            "ZIP qualification stream failed"
        ) from exc
    finally:
        archive.close()
    if _archive_snapshot(archive_path) != original_snapshot:
        raise MultiDoc2DialP0QualificationError(
            "archive changed during qualification"
        )
    if train.dialogue_ids & validation.dialogue_ids:
        raise MultiDoc2DialP0QualificationError(
            "TRAIN and VALIDATION dialogue IDs overlap"
        )
    if train.dialogue_hashes & validation.dialogue_hashes:
        raise MultiDoc2DialP0QualificationError(
            "TRAIN and VALIDATION dialogue groups overlap"
        )
    if train.item_ids & validation.item_ids:
        raise MultiDoc2DialP0QualificationError(
            "TRAIN and VALIDATION opaque item IDs overlap"
        )
    query_overlap_hashes = frozenset(
        train.query_hashes & validation.query_hashes
    )
    for query_hash in query_overlap_hashes:
        if (
            train.query_preimage_fingerprints[query_hash]
            != validation.query_preimage_fingerprints[query_hash]
        ):
            raise MultiDoc2DialP0QualificationError(
                "normalized query SHA-256 digest collided across splits"
            )
    train_rows = tuple(
        row
        for row in train.private_rows
        if row["normalized_query_sha256"] not in query_overlap_hashes
    )
    validation_rows = tuple(
        row
        for row in validation.private_rows
        if row["normalized_query_sha256"] not in query_overlap_hashes
    )
    if {
        row["normalized_query_sha256"] for row in train_rows
    } & {
        row["normalized_query_sha256"] for row in validation_rows
    }:
        raise MultiDoc2DialP0QualificationError(
            "cross-split normalized-query exclusion failed"
        )
    _validate_final_capacity(split="TRAIN", rows=train_rows)
    _validate_final_capacity(
        split="VALIDATION",
        rows=validation_rows,
    )
    train_receipt = _final_dialogue_receipt(train, train_rows)
    validation_receipt = _final_dialogue_receipt(
        validation,
        validation_rows,
    )
    cross_split_exclusion = {
        "overlap_group_count": len(query_overlap_hashes),
        "TRAIN": _cross_split_exclusion_aggregate(
            rows=train.private_rows,
            overlap_hashes=query_overlap_hashes,
        ),
        "VALIDATION": _cross_split_exclusion_aggregate(
            rows=validation.private_rows,
            overlap_hashes=query_overlap_hashes,
        ),
        "post_exclusion_overlap_group_count": 0,
    }
    observed_acts = (
        train.observed_dialogue_acts | validation.observed_dialogue_acts
    )
    if observed_acts != EXPECTED_DIALOGUE_ACTS:
        raise MultiDoc2DialP0QualificationError(
            "combined dialogue-act registry drifted"
        )
    member_receipts = {
        DOCUMENT_MEMBER: documents.member_receipt,
        TRAIN_MEMBER: train.member_receipt,
        VALIDATION_MEMBER: validation.member_receipt,
        TEST_MEMBER: {
            "central_directory_crc32": (
                f"{regular[TEST_MEMBER].CRC:08x}"
            ),
            "central_directory_size_bytes": (
                regular[TEST_MEMBER].file_size
            ),
            "payload_open_count": 0,
        },
    }
    private_manifest = self_hashed(
        {
            "version": VERSION,
            "study_id": STUDY_ID,
            "eligibility_rule_version": ELIGIBILITY_RULE_VERSION,
            "typed_core_binding": typed_core_binding,
            "custody_exclusion_binding": custody_exclusion_binding,
            "query_group_contract": QUERY_GROUP_CONTRACT,
            "source": {
                "repository": OFFICIAL_REPOSITORY,
                "commit": OFFICIAL_COMMIT,
                "archive_git_blob_sha1": git_blob_sha1,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_contract.size_bytes,
                "member_content_binding": {
                    member: member_receipts[member]
                    for member in sorted(member_receipts)
                },
            },
            "eligible_rows_by_split": {
                "TRAIN": list(train_rows),
                "VALIDATION": list(validation_rows),
            },
        }
    )
    write_json_exclusive(
        eligibility_manifest_path,
        private_manifest,
    )
    manifest_binding = _verify_private_manifest(
        eligibility_manifest_path,
        private_manifest,
    )
    return self_hashed(
        {
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": (
                "qualified_public_non_scoring_schema_grounding_and_"
                "source_native_family_capacity"
            ),
            "source": {
                "repository": OFFICIAL_REPOSITORY,
                "commit": OFFICIAL_COMMIT,
                "archive_filename": archive_contract.filename,
                "archive_git_blob_sha1": git_blob_sha1,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_contract.size_bytes,
            },
            "typed_core_binding": typed_core_binding,
            "custody_exclusion_binding": custody_exclusion_binding,
            "query_group_contract": QUERY_GROUP_CONTRACT,
            "archive_topology": topology,
            "member_receipts": member_receipts,
            "document_aggregate": documents.receipt,
            "dialogue_aggregate": {
                "TRAIN": train_receipt,
                "VALIDATION": validation_receipt,
            },
            "source_native_family_registry": {
                "families": list(FAMILIES),
                "input_da_to_family": dict(
                    sorted(INPUT_DA_TO_FAMILY.items())
                ),
                "observed_dialogue_act_set": sorted(observed_acts),
                "observed_equals_frozen_registry": True,
                "minimum_eligible_family_counts": (
                    MINIMUM_ELIGIBLE_FAMILY_COUNTS
                ),
                "minimum_eligible_domain_family_counts": (
                    MINIMUM_ELIGIBLE_DOMAIN_FAMILY_COUNTS
                ),
            },
            "split_disjointness": {
                "dial_id_overlap_count": 0,
                "dialogue_group_overlap_count": 0,
                "normalized_query_overlap_count": 0,
                "normalized_query_pre_exclusion_overlap_group_count": (
                    len(query_overlap_hashes)
                ),
                "opaque_item_id_overlap_count": 0,
            },
            "cross_split_normalized_query_exclusion": (
                cross_split_exclusion
            ),
            "private_eligibility_manifest_binding": manifest_binding,
            "access_boundary": {
                "source_payload_member_open_count": 3,
                "document_payload_member_open_count": 1,
                "train_payload_member_open_count": 1,
                "validation_payload_member_open_count": 1,
                "test_payload_member_open_count": 0,
                "source_full_extraction_count": 0,
                "secret_or_cohort_assignment_count": 0,
                "action_model_evaluator_or_score_count": 0,
                "individual_identifier_text_or_qrel_value_output_count": 0,
                "online_or_API_evaluation_count": 0,
            },
        }
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--private-eligibility-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--safe-receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    safe_receipt_path = args.safe_receipt.absolute()
    _require_fresh_file(safe_receipt_path, label="safe receipt")
    if safe_receipt_path == args.private_eligibility_manifest.absolute():
        raise MultiDoc2DialP0QualificationError(
            "safe and private outputs must be different files"
        )
    receipt = qualify_archive(
        archive_path=args.archive,
        eligibility_manifest_path=args.private_eligibility_manifest,
    )
    write_json_exclusive(safe_receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
