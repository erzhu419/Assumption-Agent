"""One-shot, public, non-scoring DSTC9 TRAIN/VALIDATION qualification.

The deterministic local USTAR bundle is byte-bound before its archive
topology is opened.  Exactly eight regular members are permitted.  FAQ,
LICENSE, and NOTICE are opened once for identity only; knowledge and the four
aligned TRAIN/VALIDATION JSON arrays are each opened and decoded once.  There
is no TEST member.

All target and non-target log histories enter a label-free prefix trie.
Eligible target rows inherit a unique maximal-dialogue leaf group; ambiguous
prefixes, frozen public examples, groups crossing official splits, and
normalized-query groups crossing official splits are excluded before the
frozen per-family unique-dialogue-group capacity check.

The private manifest contains opaque commitments only.  Safe success and
failure receipts contain aggregates, stable stages/error codes, and payload
open counts, never source identifiers, text, entity/doc/qrel values, row
hashes, actions, evaluator outcomes, or scores.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Any, BinaryIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assumption_agent.benchmarks import (  # noqa: E402
    dstc9_p1_typed_core_v1 as core,
)


VERSION = "dstc9_p0_public_source_qualification_v1"
STUDY_ID = "DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1"
ELIGIBILITY_RULE_VERSION = (
    "dstc9_p0_prefix_leaf_public_cross_split_exclusion_v1"
)

OFFICIAL_REPOSITORY = (
    "https://github.com/alexa/alexa-with-dstc9-track1-dataset"
)
OFFICIAL_COMMIT = "7ebb4c767b64826c1ac0f8bae409c0fab9cc0ae4"
BUNDLE_FILENAME = "dstc9_train_val.bundle.tar"
BUNDLE_SIZE_BYTES = 116_961_280
BUNDLE_SHA256 = (
    "6c3efa690a0829a97836dbb55bf9069b581a39e278a9601cb80a5338d21ffb83"
)

TYPED_CORE_SHA256 = (
    "a8290586595922e074e0a1aff52fd0d3eee396d0f1d366ccfc8407a5db65aa32"
)

FAQ_MEMBER = "FAQ.md"
LICENSE_MEMBER = "LICENSE"
NOTICE_MEMBER = "NOTICE"
KNOWLEDGE_MEMBER = "data/knowledge.json"
TRAIN_LABELS_MEMBER = "data/train/labels.json"
TRAIN_LOGS_MEMBER = "data/train/logs.json"
VALIDATION_LABELS_MEMBER = "data/val/labels.json"
VALIDATION_LOGS_MEMBER = "data/val/logs.json"

IDENTITY_ONLY_MEMBERS = (
    FAQ_MEMBER,
    LICENSE_MEMBER,
    NOTICE_MEMBER,
)
JSON_MEMBERS = (
    KNOWLEDGE_MEMBER,
    TRAIN_LABELS_MEMBER,
    TRAIN_LOGS_MEMBER,
    VALIDATION_LABELS_MEMBER,
    VALIDATION_LOGS_MEMBER,
)
MEMBER_PATHS = IDENTITY_ONLY_MEMBERS + JSON_MEMBERS

FAMILIES = ("hotel", "restaurant", "taxi", "train")
FAMILY_SET = frozenset(FAMILIES)
SPLITS = ("TRAIN", "VALIDATION")

PUBLIC_EXAMPLE_UTTERANCE_SHA256 = frozenset(
    {
        "0f6293543cf19d8bc19467915c1b70d8ffa8d9f5e0cdb555ec000ddcaa5b75cf",
        "2d51c5bd5adf270152a5710d3bee1948219aab4af4b877db5a53fa3d9204e892",
        "2eb4769a3799a6239737bcf82981fb0109ce8d7fdba4218eaa5a19052051eef2",
        "6953385338d451be8c3c38810717170e1c14b99a3773635674887b8814058518",
        "805600d5cc6ceff1b32a074700c775afbb207f9c46aad2174db73ccd1a025cfa",
        "e18a0e57736950ed5d00a2ef690b5258a773a9b854afcc295a62606e3997227f",
    }
)

EXPECTED_KNOWLEDGE_SNIPPETS = 2_900
EXPECTED_SPLIT_ROWS = {"TRAIN": 71_348, "VALIDATION": 9_663}
MINIMUM_UNIQUE_DIALOGUE_GROUPS = {
    "TRAIN": {family: 40 for family in FAMILIES},
    "VALIDATION": {family: 24 for family in FAMILIES},
}

READ_CHUNK_BYTES = 1 << 20
MAX_JSON_DEPTH = 64
MAX_CONTAINER_ITEMS = 2_000_000
MAX_IDENTIFIER_CHARACTERS = 1_024
MAX_TEXT_CHARACTERS = 1_000_000
MAX_TURNS_PER_LOG = 512
MAX_TOTAL_TRACE_CHARACTERS = 4_000_000
MAX_LOG_ROWS = 200_000
MAX_KNOWLEDGE_SNIPPETS = 100_000

KNOWLEDGE_ENTITY_KEYS = frozenset({"name", "docs"})
KNOWLEDGE_DOC_KEYS = frozenset({"title", "body"})
TURN_KEYS = frozenset({"speaker", "text"})
TARGET_TRUE_LABEL_KEYS = frozenset({"target", "knowledge", "response"})
TARGET_FALSE_LABEL_KEYS = frozenset({"target"})
KNOWLEDGE_REFERENCE_KEYS = frozenset(
    {"domain", "entity_id", "doc_id"}
)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class Dstc9P0QualificationError(RuntimeError):
    """A stable, non-leaking one-shot qualification failure."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.stage: str | None = None
        self.payload_open_counts: Mapping[str, int] | None = None


@dataclass(frozen=True)
class MemberContract:
    path: str
    size_bytes: int
    sha256: str
    git_blob_sha1: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or type(self.size_bytes) is not int
            or self.size_bytes < 1
            or _HEX64.fullmatch(self.sha256) is None
            or _HEX40.fullmatch(self.git_blob_sha1) is None
        ):
            raise Dstc9P0QualificationError(
                "invalid_contract",
                "member contract is invalid",
            )


@dataclass(frozen=True)
class QualificationContract:
    bundle_filename: str
    bundle_size_bytes: int
    bundle_sha256: str
    members: tuple[MemberContract, ...]
    expected_knowledge_snippets: int
    expected_split_rows: Mapping[str, int]
    minimum_unique_dialogue_groups: Mapping[str, Mapping[str, int]]
    public_example_utterance_sha256: frozenset[str]

    def __post_init__(self) -> None:
        if (
            PurePosixPath(self.bundle_filename).name
            != self.bundle_filename
            or type(self.bundle_size_bytes) is not int
            or self.bundle_size_bytes < 1
            or _HEX64.fullmatch(self.bundle_sha256) is None
            or tuple(member.path for member in self.members)
            != tuple(sorted(MEMBER_PATHS))
            or len({member.path for member in self.members})
            != len(MEMBER_PATHS)
            or type(self.expected_knowledge_snippets) is not int
            or not 1
            <= self.expected_knowledge_snippets
            <= MAX_KNOWLEDGE_SNIPPETS
            or set(self.expected_split_rows) != set(SPLITS)
            or set(self.minimum_unique_dialogue_groups) != set(SPLITS)
            or any(
                type(self.expected_split_rows[split]) is not int
                or not 1
                <= self.expected_split_rows[split]
                <= MAX_LOG_ROWS
                for split in SPLITS
            )
            or any(
                set(self.minimum_unique_dialogue_groups[split])
                != FAMILY_SET
                or any(
                    type(
                        self.minimum_unique_dialogue_groups[split][family]
                    )
                    is not int
                    or self.minimum_unique_dialogue_groups[split][family] < 1
                    for family in FAMILIES
                )
                for split in SPLITS
            )
            or any(
                _HEX64.fullmatch(value) is None
                for value in self.public_example_utterance_sha256
            )
        ):
            raise Dstc9P0QualificationError(
                "invalid_contract",
                "qualification contract is invalid",
            )

    @property
    def member_map(self) -> Mapping[str, MemberContract]:
        return {member.path: member for member in self.members}


OFFICIAL_MEMBER_CONTRACTS = (
    MemberContract(
        FAQ_MEMBER,
        4_760,
        "6c3ada2e9beb509eb8658c0a2d5893398a958908a4d121686c8417ed3dba9399",
        "f11b39bdf17345a4ebc580d20201b6397b2bfbbf",
    ),
    MemberContract(
        LICENSE_MEMBER,
        10_142,
        "09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b",
        "67db8588217f266eb561f75fae738656325deac9",
    ),
    MemberContract(
        NOTICE_MEMBER,
        67,
        "d4290ed64c2edd0fce1d84e3f9dfb2881240fe534def76b8cd29ed6af683e287",
        "616fc5889451895dbf9768e6787c8308c33bef22",
    ),
    MemberContract(
        KNOWLEDGE_MEMBER,
        471_645,
        "c8490242c23101c4e7c3e3482acd1d6dbf26c788f62c0c87fcaf622ee5360372",
        "74bc7b13ae25a85f44773285adbe033250b730a5",
    ),
    MemberContract(
        TRAIN_LABELS_MEMBER,
        6_496_347,
        "615eac39a48f9068a30a92eff03092e301b20c43121cb5198acf8b3d67557d4e",
        "23edc6fa8a76c4afe606830b175a39807917d828",
    ),
    MemberContract(
        TRAIN_LOGS_MEMBER,
        96_396_446,
        "39a87aafdc70e4adde0fc2e9b6e4caa3ab6ce8668bd7d17ea73b4534d7fa41d3",
        "f5c20a995f1c94bb5a3646b9f6aee0dcc85f5107",
    ),
    MemberContract(
        VALIDATION_LABELS_MEMBER,
        895_901,
        "6da9429d56c37ac2b0dc2b97d4bfaeb42df8596f13503719c68a76d6a86ae598",
        "e85135ee946aad59662365bd879d2861803c5c27",
    ),
    MemberContract(
        VALIDATION_LOGS_MEMBER,
        12_674_417,
        "eb31b274ba389d1ae85cccd6e0cafaf72fb320699d72be9f17efd49f04872dd7",
        "5d7957e4d879eff5311dee65fbe19ba66f8388dc",
    ),
)

OFFICIAL_CONTRACT = QualificationContract(
    bundle_filename=BUNDLE_FILENAME,
    bundle_size_bytes=BUNDLE_SIZE_BYTES,
    bundle_sha256=BUNDLE_SHA256,
    members=tuple(sorted(OFFICIAL_MEMBER_CONTRACTS, key=lambda value: value.path)),
    expected_knowledge_snippets=EXPECTED_KNOWLEDGE_SNIPPETS,
    expected_split_rows=EXPECTED_SPLIT_ROWS,
    minimum_unique_dialogue_groups=MINIMUM_UNIQUE_DIALOGUE_GROUPS,
    public_example_utterance_sha256=PUBLIC_EXAMPLE_UTTERANCE_SHA256,
)


@dataclass
class _Audit:
    stage: str = "preflight"
    payload_open_counts: dict[str, int] = field(
        default_factory=lambda: {
            "FAQ_identity": 0,
            "LICENSE_identity": 0,
            "NOTICE_identity": 0,
            "knowledge_JSON": 0,
            "TRAIN_labels_JSON": 0,
            "TRAIN_logs_JSON": 0,
            "VALIDATION_labels_JSON": 0,
            "VALIDATION_logs_JSON": 0,
        }
    )

    def opened(self, member: str) -> None:
        key = {
            FAQ_MEMBER: "FAQ_identity",
            LICENSE_MEMBER: "LICENSE_identity",
            NOTICE_MEMBER: "NOTICE_identity",
            KNOWLEDGE_MEMBER: "knowledge_JSON",
            TRAIN_LABELS_MEMBER: "TRAIN_labels_JSON",
            TRAIN_LOGS_MEMBER: "TRAIN_logs_JSON",
            VALIDATION_LABELS_MEMBER: "VALIDATION_labels_JSON",
            VALIDATION_LOGS_MEMBER: "VALIDATION_logs_JSON",
        }[member]
        self.payload_open_counts[key] += 1
        if self.payload_open_counts[key] != 1:
            raise Dstc9P0QualificationError(
                "payload_reopen",
                "source payload member was opened more than once",
            )


@dataclass(frozen=True)
class _Snapshot:
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass
class _TrieNode:
    parent: int | None
    edge: tuple[str, str] | None
    public_example_on_path: bool
    children: dict[tuple[str, str], int] = field(default_factory=dict)
    terminal_count: int = 0
    unique_leaf: int | None = None
    ambiguous: bool = False


@dataclass(frozen=True)
class _LogRow:
    split: str
    ordinal: int
    node_index: int
    normalized_query_sha256: str


@dataclass(frozen=True)
class _Candidate:
    split: str
    opaque_item_id: str
    family: str
    normalized_query_sha256: str
    node_index: int


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
        raise Dstc9P0QualificationError(
            "canonical_json_failure",
            "value is not canonical JSON",
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise Dstc9P0QualificationError(
            "self_hash_duplicate",
            "self hash already exists",
        )
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _text(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > MAX_TEXT_CHARACTERS
        or (not allow_empty and not value.strip())
    ):
        raise Dstc9P0QualificationError(
            "schema_text_invalid",
            f"{field_name} is invalid",
        )
    return value


def _identifier(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > MAX_IDENTIFIER_CHARACTERS
        or any(ord(character) < 32 for character in value)
    ):
        raise Dstc9P0QualificationError(
            "schema_identifier_invalid",
            f"{field_name} is invalid",
        )
    return value


def _reference_identifier(value: object, *, field_name: str) -> str:
    """Normalize the public label's string-or-nonnegative-integer ID."""

    if type(value) is int and 0 <= value <= 1_000_000_000:
        return str(value)
    return _identifier(value, field_name=field_name)


def normalize_turn_text(value: str) -> str:
    try:
        return core.normalize_text(
            value,
            field="turn text",
            maximum_length=core.MAX_TURN_CHARACTERS,
        )
    except core.Dstc9P1TypedCoreError as exc:
        raise Dstc9P0QualificationError(
            "typed_turn_contract",
            "turn violates the frozen typed-core contract",
        ) from exc


def public_utterance_sha256(value: str) -> str:
    normalized = normalize_turn_text(value).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalized_trace_payload(
    trace: Sequence[tuple[str, str]],
) -> Mapping[str, Any]:
    if not trace:
        raise Dstc9P0QualificationError(
            "empty_trace",
            "normalized trace is empty",
        )
    return {
        "turns": [
            {"speaker": speaker, "text": text}
            for speaker, text in trace
        ]
    }


def _snapshot(path: Path) -> _Snapshot:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Dstc9P0QualificationError(
            "bundle_unavailable",
            "bundle is unavailable",
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise Dstc9P0QualificationError(
            "bundle_metadata_invalid",
            "bundle is not a single mode-0600 regular file",
        )
    return _Snapshot(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _verify_bundle_identity(
    path: Path,
    contract: QualificationContract,
) -> _Snapshot:
    before = _snapshot(path)
    if before.size != contract.bundle_size_bytes:
        raise Dstc9P0QualificationError(
            "bundle_size_mismatch",
            "bundle size drifted",
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
            ) != (before.device, before.inode, before.size):
                raise Dstc9P0QualificationError(
                    "bundle_changed_during_open",
                    "bundle changed during open",
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
        raise Dstc9P0QualificationError(
            "bundle_identity_read_failed",
            "bundle identity read failed",
        ) from exc
    if (
        _snapshot(path) != before
        or size != contract.bundle_size_bytes
        or digest.hexdigest() != contract.bundle_sha256
    ):
        raise Dstc9P0QualificationError(
            "bundle_sha256_mismatch",
            "bundle byte identity drifted",
        )
    return before


def _octal_header_value(
    raw: bytes,
    *,
    field_name: str,
) -> int:
    stripped = raw.rstrip(b"\0 ").lstrip(b" ")
    if not stripped or any(value not in b"01234567" for value in stripped):
        raise Dstc9P0QualificationError(
            "ustar_numeric_invalid",
            f"USTAR {field_name} is not canonical octal",
        )
    return int(stripped, 8)


def _header_path(header: bytes) -> str:
    name = header[0:100].split(b"\0", 1)[0]
    prefix = header[345:500].split(b"\0", 1)[0]
    try:
        parts = [
            value.decode("utf-8")
            for value in (prefix, name)
            if value
        ]
    except UnicodeDecodeError as exc:
        raise Dstc9P0QualificationError(
            "ustar_path_encoding",
            "USTAR path is not UTF-8",
        ) from exc
    return "/".join(parts)


def _validate_ustar_topology(
    bundle_path: Path,
    archive: tarfile.TarFile,
    contract: QualificationContract,
) -> tuple[dict[str, tarfile.TarInfo], Mapping[str, Any]]:
    members = archive.getmembers()
    paths = [member.name for member in members]
    if (
        len(members) != 8
        or paths != sorted(MEMBER_PATHS)
        or set(paths) != set(MEMBER_PATHS)
    ):
        raise Dstc9P0QualificationError(
            "archive_topology_mismatch",
            "archive member topology drifted",
        )
    by_path = contract.member_map
    result: dict[str, tarfile.TarInfo] = {}
    expected_offset = 0
    try:
        with bundle_path.open("rb") as raw_archive:
            for member in members:
                expected = by_path[member.name]
                if (
                    not member.isreg()
                    or member.type != tarfile.REGTYPE
                    or member.mode != 0o600
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or member.size != expected.size_bytes
                    or member.pax_headers
                    or member.offset != expected_offset
                    or member.offset_data != member.offset + 512
                ):
                    raise Dstc9P0QualificationError(
                        "archive_member_metadata_mismatch",
                        "archive member metadata drifted",
                    )
                raw_archive.seek(member.offset)
                header = raw_archive.read(512)
                if (
                    len(header) != 512
                    or header[257:263] != b"ustar\x00"
                    or header[263:265] != b"00"
                    or header[156:157] != b"0"
                    or _header_path(header) != member.name
                    or _octal_header_value(
                        header[100:108],
                        field_name="mode",
                    )
                    != 0o600
                    or _octal_header_value(
                        header[108:116],
                        field_name="uid",
                    )
                    != 0
                    or _octal_header_value(
                        header[116:124],
                        field_name="gid",
                    )
                    != 0
                    or _octal_header_value(
                        header[124:136],
                        field_name="size",
                    )
                    != expected.size_bytes
                    or _octal_header_value(
                        header[136:148],
                        field_name="mtime",
                    )
                    != 0
                ):
                    raise Dstc9P0QualificationError(
                        "ustar_header_mismatch",
                        "archive is not the frozen USTAR encoding",
                    )
                expected_offset = member.offset_data + (
                    (member.size + 511) // 512
                ) * 512
                result[member.name] = member
            raw_archive.seek(expected_offset)
            trailer = raw_archive.read()
    except OSError as exc:
        raise Dstc9P0QualificationError(
            "ustar_topology_read_failed",
            "USTAR topology read failed",
        ) from exc
    if (
        len(trailer) < 1_024
        or len(trailer) % 512 != 0
        or any(trailer)
    ):
        raise Dstc9P0QualificationError(
            "ustar_trailer_invalid",
            "USTAR trailer is not exact zero padding",
        )
    return result, {
        "regular_member_count": 8,
        "directory_link_or_special_member_count": 0,
        "uid_gid_zero_member_count": 8,
        "mode_0600_member_count": 8,
        "mtime_zero_member_count": 8,
        "ustar_header_count": 8,
        "test_member_count": 0,
    }


class _HashingReader:
    def __init__(self, source: BinaryIO, *, expected_size: int) -> None:
        self._source = source
        self._sha256 = hashlib.sha256()
        self._git_sha1 = hashlib.sha1()  # noqa: S324 - Git identity.
        self._git_sha1.update(
            f"blob {expected_size}\0".encode("ascii")
        )
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        raw = self._source.read(size)
        if not isinstance(raw, bytes):
            raise Dstc9P0QualificationError(
                "member_reader_invalid",
                "member reader returned non-bytes",
            )
        self._sha256.update(raw)
        self._git_sha1.update(raw)
        self.size += len(raw)
        return raw

    @property
    def sha256(self) -> str:
        return self._sha256.hexdigest()

    @property
    def git_blob_sha1(self) -> str:
        return self._git_sha1.hexdigest()


def _verify_member_reader(
    reader: _HashingReader,
    contract: MemberContract,
) -> Mapping[str, Any]:
    if (
        reader.size != contract.size_bytes
        or reader.sha256 != contract.sha256
        or reader.git_blob_sha1 != contract.git_blob_sha1
    ):
        raise Dstc9P0QualificationError(
            "member_identity_mismatch",
            "member byte identity drifted",
        )
    return {
        "size_bytes": reader.size,
        "sha256": reader.sha256,
        "git_blob_sha1": reader.git_blob_sha1,
        "payload_open_count": 1,
    }


def _open_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    contract: MemberContract,
    audit: _Audit,
) -> tuple[BinaryIO, _HashingReader]:
    audit.opened(member.name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise Dstc9P0QualificationError(
            "member_open_failed",
            "member payload could not be opened",
        )
    reader = _HashingReader(
        extracted,
        expected_size=contract.size_bytes,
    )
    return extracted, reader


def _read_identity_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    contract: MemberContract,
    audit: _Audit,
) -> Mapping[str, Any]:
    extracted, reader = _open_member(archive, member, contract, audit)
    try:
        while reader.read(READ_CHUNK_BYTES):
            pass
    finally:
        extracted.close()
    return _verify_member_reader(reader, contract)


def _no_duplicate_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise Dstc9P0QualificationError(
                "json_duplicate_key",
                "JSON contains a duplicate or invalid object key",
            )
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise Dstc9P0QualificationError(
        "json_nonfinite_number",
        "JSON contains a non-finite number",
    )


def _strict_json_bytes(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Dstc9P0QualificationError(
            "json_utf8_invalid",
            "JSON is not strict UTF-8",
        ) from exc
    if text.startswith("\ufeff"):
        raise Dstc9P0QualificationError(
            "json_bom_forbidden",
            "JSON has a forbidden BOM",
        )
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise Dstc9P0QualificationError(
            "json_decode_failed",
            "JSON decoding failed",
        ) from exc


def _ijson_basic_parse(source: BinaryIO) -> Iterator[tuple[str, Any]]:
    try:
        import ijson
    except ImportError as exc:
        raise Dstc9P0QualificationError(
            "ijson_unavailable",
            "frozen ijson runtime is unavailable",
        ) from exc
    try:
        yield from ijson.basic_parse(source, use_float=False)
    except BaseException as exc:
        if isinstance(exc, Dstc9P0QualificationError):
            raise
        raise Dstc9P0QualificationError(
            "json_stream_failed",
            "streaming JSON parse failed",
        ) from exc


def _next_event(
    events: Iterator[tuple[str, Any]],
) -> tuple[str, Any]:
    try:
        return next(events)
    except StopIteration as exc:
        raise Dstc9P0QualificationError(
            "json_truncated",
            "JSON ended unexpectedly",
        ) from exc


def _read_stream_value(
    events: Iterator[tuple[str, Any]],
    first: tuple[str, Any],
    *,
    depth: int,
) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise Dstc9P0QualificationError(
            "json_depth_exceeded",
            "JSON nesting exceeds the frozen bound",
        )
    event, value = first
    if event == "start_map":
        result: dict[str, Any] = {}
        while True:
            child_event, child_value = _next_event(events)
            if child_event == "end_map":
                return result
            if (
                child_event != "map_key"
                or not isinstance(child_value, str)
                or child_value in result
            ):
                raise Dstc9P0QualificationError(
                    "json_duplicate_key",
                    "JSON object key is duplicated or invalid",
                )
            if len(result) >= MAX_CONTAINER_ITEMS:
                raise Dstc9P0QualificationError(
                    "json_container_exceeded",
                    "JSON object exceeds the frozen bound",
                )
            result[child_value] = _read_stream_value(
                events,
                _next_event(events),
                depth=depth + 1,
            )
    if event == "start_array":
        result_list: list[Any] = []
        while True:
            child = _next_event(events)
            if child[0] == "end_array":
                return result_list
            if len(result_list) >= MAX_CONTAINER_ITEMS:
                raise Dstc9P0QualificationError(
                    "json_container_exceeded",
                    "JSON array exceeds the frozen bound",
                )
            result_list.append(
                _read_stream_value(events, child, depth=depth + 1)
            )
    if event == "string" and isinstance(value, str):
        return value
    if event == "boolean" and type(value) is bool:
        return value
    if event == "number" and type(value) is int:
        return value
    if event == "null":
        return None
    raise Dstc9P0QualificationError(
        "json_event_invalid",
        "JSON contains an unsupported value",
    )


def _iter_top_array(source: BinaryIO) -> Iterator[Any]:
    events = iter(_ijson_basic_parse(source))
    if _next_event(events) != ("start_array", None):
        raise Dstc9P0QualificationError(
            "json_root_invalid",
            "JSON root is not an array",
        )
    while True:
        event = _next_event(events)
        if event[0] == "end_array":
            break
        yield _read_stream_value(events, event, depth=1)
    try:
        next(events)
    except StopIteration:
        return
    raise Dstc9P0QualificationError(
        "json_trailing_value",
        "JSON contains trailing values",
    )


def _read_knowledge(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    contract: MemberContract,
    qualification_contract: QualificationContract,
    audit: _Audit,
) -> tuple[
    frozenset[tuple[str, str, str]],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    extracted, reader = _open_member(archive, member, contract, audit)
    parts: list[bytes] = []
    try:
        while True:
            raw = reader.read(READ_CHUNK_BYTES)
            if not raw:
                break
            parts.append(raw)
    finally:
        extracted.close()
    member_receipt = _verify_member_reader(reader, contract)
    root = _strict_json_bytes(b"".join(parts))
    if not isinstance(root, dict) or set(root) != FAMILY_SET:
        raise Dstc9P0QualificationError(
            "knowledge_domain_registry",
            "knowledge root is not the exact four-domain registry",
        )
    triples: set[tuple[str, str, str]] = set()
    domain_counts: dict[str, int] = {}
    entity_name_types: Counter[str] = Counter()
    for domain in FAMILIES:
        entities = root[domain]
        if not isinstance(entities, dict) or not entities:
            raise Dstc9P0QualificationError(
                "knowledge_entity_registry",
                "knowledge entity registry is invalid",
            )
        domain_count = 0
        for entity_key, entity in entities.items():
            entity_id = _identifier(
                entity_key,
                field_name="knowledge entity key",
            )
            if (
                not isinstance(entity, dict)
                or set(entity) != KNOWLEDGE_ENTITY_KEYS
            ):
                raise Dstc9P0QualificationError(
                    "knowledge_entity_schema",
                    "knowledge entity schema drifted",
                )
            name = entity["name"]
            if name is None:
                entity_name_types["null"] += 1
            else:
                _text(name, field_name="knowledge entity name")
                entity_name_types["string"] += 1
            documents = entity["docs"]
            if not isinstance(documents, dict) or not documents:
                raise Dstc9P0QualificationError(
                    "knowledge_doc_registry",
                    "knowledge document registry is invalid",
                )
            for document_key, document in documents.items():
                document_id = _identifier(
                    document_key,
                    field_name="knowledge document key",
                )
                if (
                    not isinstance(document, dict)
                    or set(document) != KNOWLEDGE_DOC_KEYS
                ):
                    raise Dstc9P0QualificationError(
                        "knowledge_doc_schema",
                        "knowledge document schema drifted",
                    )
                title = _text(
                    document["title"],
                    field_name="knowledge title",
                )
                body = _text(
                    document["body"],
                    field_name="knowledge body",
                )
                try:
                    core.KnowledgeSnippet(
                        ordinal=len(triples),
                        entity_name=name,
                        title=title,
                        body=body,
                    )
                except core.Dstc9P1TypedCoreError as exc:
                    raise Dstc9P0QualificationError(
                        "typed_snippet_contract",
                        "knowledge snippet violates the frozen typed-core "
                        "contract",
                    ) from exc
                triple = (domain, entity_id, document_id)
                if triple in triples:
                    raise Dstc9P0QualificationError(
                        "knowledge_triple_duplicate",
                        "knowledge triple is duplicated",
                    )
                triples.add(triple)
                domain_count += 1
        domain_counts[domain] = domain_count
    if len(triples) != qualification_contract.expected_knowledge_snippets:
        raise Dstc9P0QualificationError(
            "knowledge_snippet_count",
            "knowledge snippet count drifted",
        )
    return (
        frozenset(triples),
        {
            "snippet_count": len(triples),
            "domain_snippet_count": {
                domain: domain_counts[domain] for domain in FAMILIES
            },
            "entity_name_type_count": _counter(entity_name_types),
        },
        member_receipt,
    )


def _insert_trace(
    nodes: list[_TrieNode],
    trace: Sequence[tuple[str, str]],
    *,
    public_hashes: frozenset[str],
) -> int:
    node_index = 0
    for edge in trace:
        node = nodes[node_index]
        child_index = node.children.get(edge)
        if child_index is None:
            child_index = len(nodes)
            node.children[edge] = child_index
            nodes.append(
                _TrieNode(
                    parent=node_index,
                    edge=edge,
                    public_example_on_path=(
                        node.public_example_on_path
                        or public_utterance_sha256(edge[1])
                        in public_hashes
                    ),
                )
            )
        node_index = child_index
    nodes[node_index].terminal_count += 1
    return node_index


def _read_logs(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    member_contract: MemberContract,
    qualification_contract: QualificationContract,
    audit: _Audit,
    *,
    split: str,
    nodes: list[_TrieNode],
    query_fingerprints: dict[str, tuple[int, str]],
) -> tuple[list[_LogRow], Mapping[str, Any], Mapping[str, Any]]:
    extracted, reader = _open_member(
        archive,
        member,
        member_contract,
        audit,
    )
    rows: list[_LogRow] = []
    speaker_counts: Counter[str] = Counter()
    turn_count = 0
    try:
        for ordinal, raw_log in enumerate(_iter_top_array(reader)):
            if ordinal >= MAX_LOG_ROWS:
                raise Dstc9P0QualificationError(
                    "log_row_bound",
                    "log row count exceeds the frozen bound",
                )
            if (
                not isinstance(raw_log, list)
                or not 1 <= len(raw_log) <= MAX_TURNS_PER_LOG
            ):
                raise Dstc9P0QualificationError(
                    "log_schema",
                    "log is not a bounded nonempty turn array",
                )
            trace: list[tuple[str, str]] = []
            typed_history: list[core.DialogueTurn] = []
            total_characters = 0
            for raw_turn in raw_log:
                if (
                    not isinstance(raw_turn, dict)
                    or set(raw_turn) != TURN_KEYS
                ):
                    raise Dstc9P0QualificationError(
                        "turn_schema",
                        "turn schema drifted",
                    )
                speaker = raw_turn["speaker"]
                if speaker not in {"U", "S"}:
                    raise Dstc9P0QualificationError(
                        "speaker_registry",
                        "speaker is outside the public U/S registry",
                    )
                try:
                    typed_turn = core.DialogueTurn(
                        speaker=speaker,
                        text=raw_turn["text"],
                    )
                except core.Dstc9P1TypedCoreError as exc:
                    raise Dstc9P0QualificationError(
                        "typed_turn_contract",
                        "turn violates the frozen typed-core contract",
                    ) from exc
                normalized = typed_turn.text
                total_characters += len(normalized)
                if total_characters > MAX_TOTAL_TRACE_CHARACTERS:
                    raise Dstc9P0QualificationError(
                        "trace_character_bound",
                        "normalized trace exceeds the frozen bound",
                    )
                trace.append((speaker, normalized))
                typed_history.append(typed_turn)
                speaker_counts[speaker] += 1
            if trace[-1][0] != "U":
                raise Dstc9P0QualificationError(
                    "final_turn_not_user",
                    "log does not end in a user turn",
                )
            try:
                typed_payload = core.normalized_query_payload(
                    tuple(typed_history)
                )
                payload_bytes = core.canonical_bytes(typed_payload)
                query_sha256 = core.normalized_query_sha256(
                    tuple(typed_history)
                )
            except core.Dstc9P1TypedCoreError as exc:
                raise Dstc9P0QualificationError(
                    "typed_history_contract",
                    "history violates the frozen typed-core contract",
                ) from exc
            if hashlib.sha256(payload_bytes).hexdigest() != query_sha256:
                raise Dstc9P0QualificationError(
                    "typed_query_hash_drift",
                    "typed-core query hash implementation drifted",
                )
            fingerprint = (
                len(payload_bytes),
                hashlib.sha512(payload_bytes).hexdigest(),
            )
            previous = query_fingerprints.get(query_sha256)
            if previous is not None and previous != fingerprint:
                raise Dstc9P0QualificationError(
                    "query_digest_collision",
                    "normalized query digest collided",
                )
            query_fingerprints[query_sha256] = fingerprint
            node_index = _insert_trace(
                nodes,
                trace,
                public_hashes=(
                    qualification_contract.public_example_utterance_sha256
                ),
            )
            rows.append(
                _LogRow(
                    split,
                    ordinal,
                    node_index,
                    query_sha256,
                )
            )
            turn_count += len(trace)
    finally:
        extracted.close()
    member_receipt = _verify_member_reader(reader, member_contract)
    expected = qualification_contract.expected_split_rows[split]
    if len(rows) != expected:
        raise Dstc9P0QualificationError(
            "log_row_count",
            "log row count drifted",
        )
    return rows, {
        "row_count": len(rows),
        "turn_count": turn_count,
        "speaker_count": _counter(speaker_counts),
        "final_user_turn_count": len(rows),
    }, member_receipt


def _read_labels(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    member_contract: MemberContract,
    qualification_contract: QualificationContract,
    audit: _Audit,
    *,
    split: str,
    logs: Sequence[_LogRow],
    knowledge_triples: frozenset[tuple[str, str, str]],
) -> tuple[list[_Candidate], Mapping[str, Any], Mapping[str, Any]]:
    extracted, reader = _open_member(
        archive,
        member,
        member_contract,
        audit,
    )
    candidates: list[_Candidate] = []
    target_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    item_ids: set[str] = set()
    observed_rows = 0
    try:
        for ordinal, label in enumerate(_iter_top_array(reader)):
            if ordinal >= len(logs):
                raise Dstc9P0QualificationError(
                    "label_log_alignment",
                    "labels contain more rows than logs",
                )
            if not isinstance(label, dict) or type(label.get("target")) is not bool:
                raise Dstc9P0QualificationError(
                    "label_schema",
                    "label target schema drifted",
                )
            target = label["target"]
            target_counts[str(target).lower()] += 1
            if target:
                if set(label) != TARGET_TRUE_LABEL_KEYS:
                    raise Dstc9P0QualificationError(
                        "target_true_schema",
                        "target=true label schema drifted",
                    )
                references = label["knowledge"]
                if not isinstance(references, list) or len(references) != 1:
                    raise Dstc9P0QualificationError(
                        "target_knowledge_cardinality",
                        "target=true knowledge is not a singleton",
                    )
                reference = references[0]
                if (
                    not isinstance(reference, dict)
                    or set(reference) != KNOWLEDGE_REFERENCE_KEYS
                ):
                    raise Dstc9P0QualificationError(
                        "knowledge_reference_schema",
                        "knowledge reference schema drifted",
                    )
                domain = _identifier(
                    reference["domain"],
                    field_name="knowledge reference domain",
                )
                entity_id = _reference_identifier(
                    reference["entity_id"],
                    field_name="knowledge reference entity_id",
                )
                document_id = _reference_identifier(
                    reference["doc_id"],
                    field_name="knowledge reference doc_id",
                )
                if domain not in FAMILY_SET:
                    raise Dstc9P0QualificationError(
                        "label_family_registry",
                        "knowledge reference domain is outside the registry",
                    )
                if (
                    domain,
                    entity_id,
                    document_id,
                ) not in knowledge_triples:
                    raise Dstc9P0QualificationError(
                        "knowledge_reference_unresolved",
                        "knowledge reference does not resolve exactly",
                    )
                _text(
                    label["response"],
                    field_name="target response",
                )
                item_id = stable_hash(
                    {"split": split, "source_ordinal": ordinal}
                )
                if item_id in item_ids:
                    raise Dstc9P0QualificationError(
                        "opaque_item_digest_collision",
                        "opaque item digest collided",
                    )
                item_ids.add(item_id)
                candidates.append(
                    _Candidate(
                        split=split,
                        opaque_item_id=item_id,
                        family=domain,
                        normalized_query_sha256=(
                            logs[ordinal].normalized_query_sha256
                        ),
                        node_index=logs[ordinal].node_index,
                    )
                )
                family_counts[domain] += 1
            elif set(label) != TARGET_FALSE_LABEL_KEYS:
                raise Dstc9P0QualificationError(
                    "target_false_schema",
                    "target=false label contains knowledge or response",
                )
            observed_rows += 1
    finally:
        extracted.close()
    member_receipt = _verify_member_reader(reader, member_contract)
    expected = qualification_contract.expected_split_rows[split]
    if observed_rows != expected or observed_rows != len(logs):
        raise Dstc9P0QualificationError(
            "label_log_alignment",
            "labels and logs are not exactly aligned",
        )
    return candidates, {
        "row_count": observed_rows,
        "target_count": _counter(target_counts),
        "target_true_family_count": {
            family: family_counts[family] for family in FAMILIES
        },
        "singleton_resolved_reference_count": len(candidates),
    }, member_receipt


def _finalize_trie(
    nodes: list[_TrieNode],
) -> tuple[Mapping[int, str], Mapping[str, bool], Mapping[str, tuple[int, str]]]:
    for node_index in range(len(nodes) - 1, -1, -1):
        node = nodes[node_index]
        if not node.children:
            if node.terminal_count < 1:
                raise Dstc9P0QualificationError(
                    "trie_leaf_without_terminal",
                    "prefix trie leaf is not terminal",
                )
            node.unique_leaf = node_index
            continue
        leaves: set[int] = set()
        for child_index in node.children.values():
            child = nodes[child_index]
            if child.ambiguous:
                node.ambiguous = True
                break
            if child.unique_leaf is not None:
                leaves.add(child.unique_leaf)
            if len(leaves) > 1:
                node.ambiguous = True
                break
        if not node.ambiguous and len(leaves) == 1:
            node.unique_leaf = next(iter(leaves))
        else:
            node.ambiguous = True
    leaf_group_hash: dict[int, str] = {}
    group_public: dict[str, bool] = {}
    fingerprints: dict[str, tuple[int, str]] = {}
    for node_index, node in enumerate(nodes):
        if node.unique_leaf != node_index or node.children:
            continue
        trace: list[tuple[str, str]] = []
        cursor = node_index
        while nodes[cursor].parent is not None:
            edge = nodes[cursor].edge
            if edge is None:
                raise Dstc9P0QualificationError(
                    "trie_parent_edge_missing",
                    "prefix trie edge disappeared",
                )
            trace.append(edge)
            cursor = nodes[cursor].parent  # type: ignore[assignment]
        trace.reverse()
        raw = canonical_bytes(normalized_trace_payload(trace))
        group_hash = hashlib.sha256(raw).hexdigest()
        fingerprint = (len(raw), hashlib.sha512(raw).hexdigest())
        previous = fingerprints.get(group_hash)
        if previous is not None and previous != fingerprint:
            raise Dstc9P0QualificationError(
                "dialogue_group_digest_collision",
                "dialogue group digest collided",
            )
        fingerprints[group_hash] = fingerprint
        leaf_group_hash[node_index] = group_hash
        group_public[group_hash] = node.public_example_on_path
    return leaf_group_hash, group_public, fingerprints


def _query_group_aggregate(
    candidates: Sequence[_Candidate],
) -> Mapping[str, int]:
    counts = Counter(
        candidate.normalized_query_sha256 for candidate in candidates
    )
    duplicates = [count for count in counts.values() if count > 1]
    return {
        "group_count": len(counts),
        "duplicate_group_count": len(duplicates),
        "duplicate_row_count": sum(duplicates),
        "excess_duplicate_row_count": sum(
            count - 1 for count in duplicates
        ),
        "maximum_selected_items_per_group": 1,
    }


def _family_group_counts(
    candidates: Sequence[_Candidate],
    group_by_item: Mapping[str, str],
) -> Mapping[str, int]:
    groups: dict[str, set[str]] = {
        family: set() for family in FAMILIES
    }
    for candidate in candidates:
        groups[candidate.family].add(
            group_by_item[candidate.opaque_item_id]
        )
    return {family: len(groups[family]) for family in FAMILIES}


def _write_json_exclusive(
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
                    raise Dstc9P0QualificationError(
                        "exclusive_write_stalled",
                        "exclusive JSON write stalled",
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise Dstc9P0QualificationError(
            "exclusive_write_failed",
            "exclusive JSON write failed",
        ) from exc


def _require_fresh_output(path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise Dstc9P0QualificationError(
            "output_not_fresh",
            f"{label} is not fresh",
        )
    parent = path.absolute().parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or not os.access(parent, os.W_OK | os.X_OK)
    ):
        raise Dstc9P0QualificationError(
            "output_parent_invalid",
            f"{label} parent is invalid",
        )


def _typed_core_binding(value: str) -> Mapping[str, str]:
    if _HEX64.fullmatch(value) is None or value == "0" * 64:
        raise Dstc9P0QualificationError(
            "typed_core_binding_pending",
            "typed core SHA-256 is not frozen",
        )
    path = Path(core.__file__).absolute()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Dstc9P0QualificationError(
            "typed_core_unavailable",
            "typed core source is unavailable",
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise Dstc9P0QualificationError(
            "typed_core_metadata",
            "typed core source is not a single regular file",
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
                raise Dstc9P0QualificationError(
                    "typed_core_changed_during_open",
                    "typed core changed during open",
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
        raise Dstc9P0QualificationError(
            "typed_core_identity_read",
            "typed core identity read failed",
        ) from exc
    observed = digest.hexdigest()
    if (
        size != metadata.st_size
        or observed != value
        or core.VERSION != "dstc9_p1_typed_core_v1"
        or core.STUDY_ID != STUDY_ID
    ):
        raise Dstc9P0QualificationError(
            "typed_core_identity_mismatch",
            "typed core version, study, or SHA-256 drifted",
        )
    return {
        "version": core.VERSION,
        "study_id": core.STUDY_ID,
        "sha256": observed,
    }


def _bind_failure(
    exc: Dstc9P0QualificationError,
    audit: _Audit,
) -> Dstc9P0QualificationError:
    if exc.stage is None:
        exc.stage = audit.stage
    if exc.payload_open_counts is None:
        exc.payload_open_counts = dict(audit.payload_open_counts)
    return exc


def qualify_bundle(
    *,
    bundle_path: Path,
    eligibility_manifest_path: Path,
    qualification_contract: QualificationContract = OFFICIAL_CONTRACT,
    typed_core_sha256: str = TYPED_CORE_SHA256,
) -> dict[str, Any]:
    """Run one source qualification and write one private manifest."""

    audit = _Audit()
    try:
        bundle_path = bundle_path.absolute()
        eligibility_manifest_path = eligibility_manifest_path.absolute()
        _require_fresh_output(
            eligibility_manifest_path,
            label="private eligibility manifest",
        )
        typed_binding = _typed_core_binding(typed_core_sha256)
        public_exclusion_binding = {
            "definition": (
                "sha256_of_nfkc_whitespace_folded_stripped_casefolded_"
                "utterance_utf8"
            ),
            "count": len(
                qualification_contract.public_example_utterance_sha256
            ),
            "set_sha256": stable_hash(
                sorted(
                    qualification_contract.public_example_utterance_sha256
                )
            ),
        }

        audit.stage = "bundle_identity"
        original_snapshot = _verify_bundle_identity(
            bundle_path,
            qualification_contract,
        )
        audit.stage = "archive_topology"
        try:
            archive = tarfile.open(bundle_path, mode="r:")
        except (OSError, tarfile.TarError) as exc:
            raise Dstc9P0QualificationError(
                "archive_open_failed",
                "bundle is not a readable uncompressed TAR",
            ) from exc
        try:
            members, topology = _validate_ustar_topology(
                bundle_path,
                archive,
                qualification_contract,
            )
            member_contracts = qualification_contract.member_map
            member_receipts: dict[str, Any] = {}
            for identity_member in IDENTITY_ONLY_MEMBERS:
                audit.stage = "identity_member"
                member_receipts[identity_member] = _read_identity_member(
                    archive,
                    members[identity_member],
                    member_contracts[identity_member],
                    audit,
                )

            audit.stage = "knowledge_JSON"
            (
                knowledge_triples,
                knowledge_aggregate,
                member_receipts[KNOWLEDGE_MEMBER],
            ) = _read_knowledge(
                archive,
                members[KNOWLEDGE_MEMBER],
                member_contracts[KNOWLEDGE_MEMBER],
                qualification_contract,
                audit,
            )

            nodes = [
                _TrieNode(
                    parent=None,
                    edge=None,
                    public_example_on_path=False,
                )
            ]
            query_fingerprints: dict[str, tuple[int, str]] = {}
            audit.stage = "TRAIN_logs_JSON"
            (
                train_logs,
                train_log_aggregate,
                member_receipts[TRAIN_LOGS_MEMBER],
            ) = _read_logs(
                archive,
                members[TRAIN_LOGS_MEMBER],
                member_contracts[TRAIN_LOGS_MEMBER],
                qualification_contract,
                audit,
                split="TRAIN",
                nodes=nodes,
                query_fingerprints=query_fingerprints,
            )
            audit.stage = "TRAIN_labels_JSON"
            (
                train_candidates,
                train_label_aggregate,
                member_receipts[TRAIN_LABELS_MEMBER],
            ) = _read_labels(
                archive,
                members[TRAIN_LABELS_MEMBER],
                member_contracts[TRAIN_LABELS_MEMBER],
                qualification_contract,
                audit,
                split="TRAIN",
                logs=train_logs,
                knowledge_triples=knowledge_triples,
            )
            audit.stage = "VALIDATION_logs_JSON"
            (
                validation_logs,
                validation_log_aggregate,
                member_receipts[VALIDATION_LOGS_MEMBER],
            ) = _read_logs(
                archive,
                members[VALIDATION_LOGS_MEMBER],
                member_contracts[VALIDATION_LOGS_MEMBER],
                qualification_contract,
                audit,
                split="VALIDATION",
                nodes=nodes,
                query_fingerprints=query_fingerprints,
            )
            audit.stage = "VALIDATION_labels_JSON"
            (
                validation_candidates,
                validation_label_aggregate,
                member_receipts[VALIDATION_LABELS_MEMBER],
            ) = _read_labels(
                archive,
                members[VALIDATION_LABELS_MEMBER],
                member_contracts[VALIDATION_LABELS_MEMBER],
                qualification_contract,
                audit,
                split="VALIDATION",
                logs=validation_logs,
                knowledge_triples=knowledge_triples,
            )
        finally:
            archive.close()
        if _snapshot(bundle_path) != original_snapshot:
            raise Dstc9P0QualificationError(
                "bundle_changed_during_qualification",
                "bundle changed during qualification",
            )

        audit.stage = "prefix_trie"
        leaf_hashes, public_groups, _group_fingerprints = _finalize_trie(
            nodes
        )
        all_logs = train_logs + validation_logs
        group_splits: dict[str, set[str]] = defaultdict(set)
        ambiguous_rows: Counter[str] = Counter()
        for log in all_logs:
            node = nodes[log.node_index]
            if node.ambiguous or node.unique_leaf is None:
                ambiguous_rows[log.split] += 1
                continue
            group_hash = leaf_hashes[node.unique_leaf]
            group_splits[group_hash].add(log.split)
        cross_block_groups = frozenset(
            group
            for group, splits in group_splits.items()
            if len(splits) > 1
        )
        public_group_set = frozenset(
            group for group, value in public_groups.items() if value
        )

        audit.stage = "eligibility_exclusion"
        candidates = train_candidates + validation_candidates
        if len(
            {candidate.opaque_item_id for candidate in candidates}
        ) != len(candidates):
            raise Dstc9P0QualificationError(
                "opaque_item_digest_collision",
                "opaque item digest collided across splits",
            )
        group_by_item: dict[str, str] = {}
        exclusion_reason: dict[str, str] = {}
        for candidate in candidates:
            node = nodes[candidate.node_index]
            if node.ambiguous or node.unique_leaf is None:
                exclusion_reason[candidate.opaque_item_id] = (
                    "ambiguous_prefix"
                )
                continue
            group_hash = leaf_hashes[node.unique_leaf]
            group_by_item[candidate.opaque_item_id] = group_hash
            if group_hash in public_group_set:
                exclusion_reason[candidate.opaque_item_id] = (
                    "public_example_group"
                )
        pre_query_candidates = [
            candidate
            for candidate in candidates
            if candidate.opaque_item_id not in exclusion_reason
        ]
        query_splits: dict[str, set[str]] = defaultdict(set)
        for candidate in pre_query_candidates:
            query_splits[candidate.normalized_query_sha256].add(
                candidate.split
            )
        cross_split_queries = frozenset(
            query
            for query, splits in query_splits.items()
            if len(splits) > 1
        )
        for candidate in pre_query_candidates:
            if (
                candidate.normalized_query_sha256
                in cross_split_queries
            ):
                exclusion_reason[candidate.opaque_item_id] = (
                    "cross_split_query_group"
                )
        for candidate in pre_query_candidates:
            if (
                candidate.opaque_item_id not in exclusion_reason
                and group_by_item[candidate.opaque_item_id]
                in cross_block_groups
            ):
                exclusion_reason[candidate.opaque_item_id] = (
                    "cross_block_dialogue_group"
                )
        final_candidates = [
            candidate
            for candidate in candidates
            if candidate.opaque_item_id not in exclusion_reason
        ]
        final_by_split = {
            split: [
                candidate
                for candidate in final_candidates
                if candidate.split == split
            ]
            for split in SPLITS
        }
        final_group_hashes = {
            split: {
                group_by_item[candidate.opaque_item_id]
                for candidate in final_by_split[split]
            }
            for split in SPLITS
        }
        if final_group_hashes["TRAIN"] & final_group_hashes["VALIDATION"]:
            raise Dstc9P0QualificationError(
                "post_exclusion_group_overlap",
                "dialogue groups remain across blocks",
            )
        final_query_hashes = {
            split: {
                candidate.normalized_query_sha256
                for candidate in final_by_split[split]
            }
            for split in SPLITS
        }
        if final_query_hashes["TRAIN"] & final_query_hashes["VALIDATION"]:
            raise Dstc9P0QualificationError(
                "post_exclusion_query_overlap",
                "normalized query groups remain across splits",
            )

        family_group_counts = {
            split: _family_group_counts(
                final_by_split[split],
                group_by_item,
            )
            for split in SPLITS
        }
        deficient = {
            split: {
                family: {
                    "observed": family_group_counts[split][family],
                    "required": (
                        qualification_contract
                        .minimum_unique_dialogue_groups[split][family]
                    ),
                }
                for family in FAMILIES
                if family_group_counts[split][family]
                < qualification_contract.minimum_unique_dialogue_groups[
                    split
                ][family]
            }
            for split in SPLITS
        }
        deficient = {
            split: values for split, values in deficient.items() if values
        }
        if deficient:
            raise Dstc9P0QualificationError(
                "post_exclusion_capacity",
                "post-exclusion unique dialogue-group capacity failed",
            )

        exclusion_counts: dict[str, Counter[str]] = {
            split: Counter() for split in SPLITS
        }
        exclusion_family_counts: dict[
            str, dict[str, Counter[str]]
        ] = {
            split: {
                reason: Counter()
                for reason in (
                    "ambiguous_prefix",
                    "public_example_group",
                    "cross_block_dialogue_group",
                    "cross_split_query_group",
                )
            }
            for split in SPLITS
        }
        for candidate in candidates:
            reason = exclusion_reason.get(candidate.opaque_item_id)
            if reason is not None:
                exclusion_counts[candidate.split][reason] += 1
                exclusion_family_counts[candidate.split][reason][
                    candidate.family
                ] += 1

        private_rows = {
            split: sorted(
                [
                    {
                        "opaque_item_id": candidate.opaque_item_id,
                        "domain": candidate.family,
                        "family": candidate.family,
                        "normalized_query_sha256": (
                            candidate.normalized_query_sha256
                        ),
                        "dialogue_group_sha256": (
                            group_by_item[candidate.opaque_item_id]
                        ),
                    }
                    for candidate in final_by_split[split]
                ],
                key=lambda row: (
                    row["family"],
                    row["dialogue_group_sha256"],
                    row["opaque_item_id"],
                ),
            )
            for split in SPLITS
        }
        private_manifest = self_hashed(
            {
                "version": VERSION,
                "study_id": STUDY_ID,
                "eligibility_rule_version": ELIGIBILITY_RULE_VERSION,
                "typed_core_binding": typed_binding,
                "source_binding": {
                    "repository": OFFICIAL_REPOSITORY,
                    "commit": OFFICIAL_COMMIT,
                    "bundle_sha256": qualification_contract.bundle_sha256,
                    "bundle_size_bytes": (
                        qualification_contract.bundle_size_bytes
                    ),
                    "member_identity": {
                        member: member_receipts[member]
                        for member in sorted(member_receipts)
                    },
                },
                "query_group_contract": {
                    "group_field": "normalized_query_sha256",
                    "maximum_selected_items_per_group": 1,
                    "cross_split_policy": "exclude_all_rows",
                },
                "eligible_rows_by_split": private_rows,
            }
        )
        audit.stage = "private_manifest"
        _write_json_exclusive(
            eligibility_manifest_path,
            private_manifest,
        )
        metadata = eligibility_manifest_path.lstat()
        if (
            eligibility_manifest_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise Dstc9P0QualificationError(
                "private_manifest_metadata",
                "private manifest metadata drifted",
            )
        private_file_sha256 = hashlib.sha256(
            eligibility_manifest_path.read_bytes()
        ).hexdigest()

        audit.stage = "success"
        return self_hashed(
            {
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": (
                    "qualified_public_non_scoring_schema_prefix_group_"
                    "and_capacity"
                ),
                "typed_core_binding": typed_binding,
                "source": {
                    "repository": OFFICIAL_REPOSITORY,
                    "commit": OFFICIAL_COMMIT,
                    "bundle_filename": (
                        qualification_contract.bundle_filename
                    ),
                    "bundle_size_bytes": (
                        qualification_contract.bundle_size_bytes
                    ),
                    "bundle_sha256": qualification_contract.bundle_sha256,
                },
                "archive_topology": topology,
                "member_receipts": {
                    member: member_receipts[member]
                    for member in sorted(member_receipts)
                },
                "knowledge_aggregate": knowledge_aggregate,
                "split_source_aggregate": {
                    "TRAIN": {
                        "logs": train_log_aggregate,
                        "labels": train_label_aggregate,
                    },
                    "VALIDATION": {
                        "logs": validation_log_aggregate,
                        "labels": validation_label_aggregate,
                    },
                },
                "prefix_trie_aggregate": {
                    "node_count": len(nodes),
                    "maximal_leaf_count": len(leaf_hashes),
                    "ambiguous_log_row_count": {
                        split: ambiguous_rows[split] for split in SPLITS
                    },
                    "public_example_group_count": len(public_group_set),
                    "cross_block_dialogue_group_count": len(
                        cross_block_groups
                    ),
                },
                "public_example_exclusion_binding": (
                    public_exclusion_binding
                ),
                "eligibility_exclusion_aggregate": {
                    split: {
                        "reason_row_count": {
                            reason: exclusion_counts[split][reason]
                            for reason in (
                                "ambiguous_prefix",
                                "public_example_group",
                                "cross_block_dialogue_group",
                                "cross_split_query_group",
                            )
                        },
                        "reason_family_row_count": {
                            reason: {
                                family: (
                                    exclusion_family_counts[split][reason][
                                        family
                                    ]
                                )
                                for family in FAMILIES
                            }
                            for reason in (
                                "ambiguous_prefix",
                                "public_example_group",
                                "cross_block_dialogue_group",
                                "cross_split_query_group",
                            )
                        },
                    }
                    for split in SPLITS
                },
                "cross_split_query_aggregate": {
                    "pre_exclusion_overlap_group_count": len(
                        cross_split_queries
                    ),
                    "post_exclusion_overlap_group_count": 0,
                },
                "final_eligible_aggregate": {
                    split: {
                        "row_count": len(final_by_split[split]),
                        "family_unique_dialogue_group_count": (
                            family_group_counts[split]
                        ),
                        "normalized_query_grouping": (
                            _query_group_aggregate(
                                final_by_split[split]
                            )
                        ),
                    }
                    for split in SPLITS
                },
                "private_manifest_binding": {
                    "file_sha256": private_file_sha256,
                    "self_sha256": private_manifest["self_sha256"],
                    "size_bytes": metadata.st_size,
                    "row_count": {
                        split: len(private_rows[split])
                        for split in SPLITS
                    },
                },
                "access_boundary": {
                    "payload_open_counts": dict(
                        audit.payload_open_counts
                    ),
                    "payload_member_reopen_count": 0,
                    "test_member_count": 0,
                    "bundle_full_extraction_count": 0,
                    "action_model_evaluator_score_or_secret_count": 0,
                    "individual_identifier_text_entity_doc_qrel_or_row_"
                    "hash_output_count": 0,
                    "online_or_API_evaluation_count": 0,
                },
            }
        )
    except Dstc9P0QualificationError as exc:
        raise _bind_failure(exc, audit)
    except Exception as exc:
        internal = Dstc9P0QualificationError(
            "unexpected_internal_error",
            "unexpected internal qualification failure",
        )
        raise _bind_failure(internal, audit) from exc


def failure_terminal(
    exc: BaseException,
    *,
    typed_core_sha256: str,
) -> dict[str, Any]:
    if isinstance(exc, Dstc9P0QualificationError):
        error_code = exc.error_code
        stage = exc.stage or "unknown"
        counts = dict(exc.payload_open_counts or {})
    else:
        error_code = "unexpected_internal_error"
        stage = "unknown"
        counts = {}
    return self_hashed(
        {
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "terminal_p0_failed_no_retry",
            "error_code": error_code,
            "stage": stage,
            "typed_core_sha256_declared": (
                typed_core_sha256
                if _HEX64.fullmatch(typed_core_sha256)
                else "invalid"
            ),
            "payload_open_counts": counts,
            "access_boundary": {
                "safe_failure_terminal_count": 1,
                "source_value_output_count": 0,
                "retry_replay_resample_count": 0,
                "action_model_evaluator_score_or_secret_count": 0,
                "online_or_API_evaluation_count": 0,
            },
        }
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--private-eligibility-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--safe-terminal", type=Path, required=True)
    parser.add_argument(
        "--typed-core-sha256",
        default=TYPED_CORE_SHA256,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    qualification_contract: QualificationContract = OFFICIAL_CONTRACT,
) -> int:
    args = _parse_args(argv)
    safe_path = args.safe_terminal.absolute()
    try:
        _require_fresh_output(safe_path, label="safe terminal")
    except Dstc9P0QualificationError:
        raise
    try:
        receipt = qualify_bundle(
            bundle_path=args.bundle,
            eligibility_manifest_path=(
                args.private_eligibility_manifest
            ),
            qualification_contract=qualification_contract,
            typed_core_sha256=args.typed_core_sha256,
        )
    except Exception as exc:
        _write_json_exclusive(
            safe_path,
            failure_terminal(
                exc,
                typed_core_sha256=args.typed_core_sha256,
            ),
        )
        return 2
    _write_json_exclusive(safe_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
