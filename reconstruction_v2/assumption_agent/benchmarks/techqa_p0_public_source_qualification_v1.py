"""One-shot public, non-scoring TechQA source qualification.

The pinned 2.96 GB archive is never unpacked.  Qualification first verifies
the complete archive byte identity, then performs two bounded-memory tar
streams: one captures only the two small QA JSON members, and one incrementally
parses only ``training_dev_technotes.json``.  Those exact three whitelisted
member byte streams are persisted once into a private qualified-source
directory; the corpus parser and file copy consume the same second-pass bytes,
so no third gzip pass or whole archive extraction exists.  Corpus text is
retained only while its current JSON object is validated; the persistent
in-memory identity set is limited to candidate/gold document IDs referenced by
answerable QA.

The safe receipt contains aggregate schema, family-capacity, and byte
commitments only.  It contains no question/document identifier or text, answer
span, cohort, secret, action, qrel, evaluator, score, API, or online
evaluation result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
from typing import Any, BinaryIO

if __package__ in {None, ""}:
    # Direct-script execution avoids the earlier ``python -m`` package-entry
    # failure while binding the exact project root that contains this file.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assumption_agent.benchmarks import techqa_p1_formal_v1 as formal


VERSION = "techqa_p0_public_source_qualification_v1"
STUDY_ID = "TECHQA_P0_PUBLIC_SCHEMA_CAPACITY_V1"
ELIGIBILITY_RULE_VERSION = "techqa_p0_post_corpus_eligibility_v1"

IBM_REPOSITORY = "https://github.com/IBM/techqa.git"
IBM_COMMIT = "f0cf8ce11c6ef778c6bc064ee6c1d9b3eca76faf"
IBM_TREE = "c31c7f945067b630c78e93484ca1a6dd5102a71b"
IBM_LICENSE_PATH = "LICENSE.md"
IBM_LICENSE_GIT_BLOB_SHA1 = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
IBM_LICENSE_SHA256 = (
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
)
IBM_LICENSE_SPDX = "Apache-2.0"

HF_REPOSITORY_ID = "PrimeQA/TechQA"
HF_REVISION = "60437bc79ab217679682217598a3693cab78365b"
ARCHIVE_FILENAME = "TechQA.tar.gz"
ARCHIVE_SIZE_BYTES = 2_959_973_525
ARCHIVE_SHA256 = (
    "6b094ef9a69718f727ce8d7e15c4d961e51032cefaa952e0d6af9d176d7ba118"
)
ARCHIVE_LFS_OID_SHA256 = ARCHIVE_SHA256
ARCHIVE_POINTER_GIT_BLOB_SHA1 = "33c689e2e1422393c6c67b3227e589641165d13f"
ARCHIVE_XET_HASH = (
    "8d92897927306ceb075859db42eba62860a00cebe8022a2fdf9910eebcdf0fef"
)

TRAIN_QA_BASENAME = "training_Q_A.json"
DEV_QA_BASENAME = "dev_Q_A.json"
CORPUS_BASENAME = "training_dev_technotes.json"
TARGET_BASENAMES = frozenset(
    {TRAIN_QA_BASENAME, DEV_QA_BASENAME, CORPUS_BASENAME}
)

INFORMATION = formal.INFORMATION
PROCEDURE = formal.PROCEDURE
TROUBLESHOOT = formal.TROUBLESHOOT
FAMILIES = formal.FAMILY_IDS
TROUBLESHOOT_INDICATORS = formal.TROUBLESHOOT_INDICATORS
PROCEDURE_INDICATORS = formal.PROCEDURE_INDICATORS
SOURCE_MINIMUM_FAMILY_COUNTS = formal.SOURCE_MINIMUM_FAMILY_COUNTS
MINIMUM_ANSWERABLE_FAMILY_COUNTS = {
    split: {family: minimum for family in FAMILIES}
    for split, minimum in SOURCE_MINIMUM_FAMILY_COUNTS.items()
}

EXPECTED_DOC_ID_COUNT = 50
MAX_RETAINED_CANDIDATE_OR_GOLD_IDS = 30_500
MAX_QUERY_MEMBER_BYTES = 256 * 1024 * 1024
MAX_CORPUS_MEMBER_BYTES = 64 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024 * 1024
MAX_JSON_VALUE_CHARACTERS = 32 * 1024 * 1024
READ_CHUNK_BYTES = 1 << 20

PINNED_HF_RUNTIME = {
    "python": "3.12.3",
    "huggingface_hub": "1.11.0",
    "hf_xet": "1.4.3",
    "click": "8.3.3",
    "ijson": "3.5.1",
}
PINNED_HF_RUNTIME_ROOT = (
    "/home/erzhu419/techqa_p1_20260727/source_runtime_v1/venv"
)
PINNED_HF_RUNTIME_MANIFEST_SHA256 = (
    "d80843442b9840ffca8bbfacffd3263bb23aad9dc38150b7d432958d249ef745"
)
PINNED_HF_RUNTIME_REGULAR_PATH_LIST_SHA256 = (
    "141dfbeff0b3d8ad1aacd06043a71a0e3aa42e65503da284e36f50084515fa45"
)
PINNED_SYSTEM_PYTHON_RESOLVED = "/usr/bin/python3.12"
PINNED_SYSTEM_PYTHON_SHA256 = (
    "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
)
PINNED_HF_RUNTIME_SYMLINKS = {
    "bin/python": "python3",
    "bin/python3": "/usr/bin/python3",
    "bin/python3.12": "python3",
    "lib64": "lib",
}

QUERY_REQUIRED_KEYS = frozenset(
    {
        "QUESTION_ID",
        "QUESTION_TITLE",
        "QUESTION_TEXT",
        "ANSWERABLE",
        "DOC_IDS",
    }
)
ANSWERABLE_REQUIRED_KEYS = frozenset(
    {"DOCUMENT", "START_OFFSET", "END_OFFSET"}
)
DOCUMENT_REQUIRED_KEYS = frozenset({"_id", "title", "text"})

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_VERSION = re.compile(r"[0-9]+(?:\.[0-9A-Za-z+_.-]+)*\Z")


class TechqaP0QualificationError(RuntimeError):
    """The frozen source or one-shot P0 contract failed closed."""


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
            raise TechqaP0QualificationError("archive contract is invalid")


OFFICIAL_ARCHIVE = ArchiveContract(
    ARCHIVE_FILENAME,
    ARCHIVE_SIZE_BYTES,
    ARCHIVE_SHA256,
)


@dataclass(frozen=True)
class _QueryNeed:
    split: str
    question_id: str
    family: str | None
    normalized_query_sha256: str | None
    candidate_document_ids: tuple[str, ...]
    gold_document_id: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class _QueryObservation:
    receipt: Mapping[str, Any]
    needs: tuple[_QueryNeed, ...]
    referenced_document_ids: frozenset[str]


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
        raise TechqaP0QualificationError(
            "aggregate value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise TechqaP0QualificationError("self hash already exists")
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


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


def _counter(value: Counter[Any]) -> dict[str, int]:
    return {str(key): value[key] for key in sorted(value, key=str)}


def _unknown_key_bucket(key: str, value: object) -> str:
    return (
        hashlib.sha256(key.encode("utf-8")).hexdigest()
        + ":"
        + _json_type(value)
    )


def _required_text(
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
        raise TechqaP0QualificationError(f"{field} is invalid")
    return value


def _identifier(value: object, *, field: str) -> str:
    text = _required_text(value, field=field, maximum=512)
    if text != text.strip() or any(ord(character) < 32 for character in text):
        raise TechqaP0QualificationError(f"{field} is not canonical")
    return text


def _pool_identifier(value: object) -> str:
    """Match the official processor's one permitted DOC_IDS normalization."""

    if not isinstance(value, str):
        raise TechqaP0QualificationError("DOC_IDS member is not text")
    return _identifier(value.strip(), field="trimmed DOC_IDS member")


def operational_family(question_title: str, question_text: str) -> str:
    """Call the single frozen classifier; never duplicate its semantics."""

    try:
        return formal.operational_family(question_title, question_text)
    except (
        formal.TechqaP1FormalError,
        formal.core.TechqaP1TypedCoreError,
    ) as exc:
        raise TechqaP0QualificationError(
            "frozen operational family input drifted"
        ) from exc


def _no_duplicate_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise TechqaP0QualificationError(
                "JSON contains a duplicate or non-string object key"
            )
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise TechqaP0QualificationError("JSON contains a non-finite number")


_STRICT_DECODER = json.JSONDecoder(
    object_pairs_hook=_no_duplicate_object,
    parse_constant=_reject_constant,
)


def _strict_json_bytes(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TechqaP0QualificationError("member is not strict UTF-8") from exc
    if text.startswith("\ufeff"):
        raise TechqaP0QualificationError("member has a forbidden UTF-8 BOM")
    try:
        value = _STRICT_DECODER.decode(text)
    except json.JSONDecodeError as exc:
        raise TechqaP0QualificationError("member is not strict JSON") from exc
    return value


class _HashingReader:
    """Commit and optionally persist every byte consumed by the JSON parser."""

    def __init__(
        self,
        source: BinaryIO,
        *,
        copy_descriptor: int | None = None,
    ) -> None:
        self._source = source
        self._copy_descriptor = copy_descriptor
        self._hash = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        raw = self._source.read(size)
        if not isinstance(raw, bytes):
            raise TechqaP0QualificationError("member reader returned non-bytes")
        if self._copy_descriptor is not None:
            offset = 0
            while offset < len(raw):
                try:
                    written = os.write(
                        self._copy_descriptor,
                        raw[offset:],
                    )
                except OSError as exc:
                    raise TechqaP0QualificationError(
                        "qualified corpus byte persistence failed"
                    ) from exc
                if written <= 0:
                    raise TechqaP0QualificationError(
                        "qualified corpus byte persistence stalled"
                    )
                offset += written
        self._hash.update(raw)
        self.size += len(raw)
        return raw

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()


def _ijson_kvitems(source: BinaryIO) -> Iterator[tuple[str, Any]]:
    """Use the deployment-pinned ijson 3.5.1 corpus stream."""

    try:
        import ijson
    except ImportError as exc:
        raise TechqaP0QualificationError(
            "pinned ijson runtime is unavailable"
        ) from exc
    if getattr(ijson, "__version__", None) != PINNED_HF_RUNTIME["ijson"]:
        raise TechqaP0QualificationError("pinned ijson version drifted")
    try:
        yield from ijson.kvitems(source, "")
    except BaseException as exc:
        if isinstance(exc, TechqaP0QualificationError):
            raise
        raise TechqaP0QualificationError(
            "pinned ijson corpus stream failed"
        ) from exc


def _safe_archive_path(name: object) -> PurePosixPath:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
    ):
        raise TechqaP0QualificationError("tar member path is unsafe")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise TechqaP0QualificationError("tar member path traverses archive")
    return path


def _validate_tar_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = _safe_archive_path(member.name)
    if (
        member.issym()
        or member.islnk()
        or member.ischr()
        or member.isblk()
        or member.isfifo()
        or member.type == tarfile.GNUTYPE_SPARSE
        or (not member.isdir() and not member.isreg())
    ):
        raise TechqaP0QualificationError(
            "tar contains a link, sparse, or special member"
        )
    if (
        type(member.size) is not int
        or member.size < 0
        or member.size > MAX_ARCHIVE_MEMBER_BYTES
    ):
        raise TechqaP0QualificationError("tar member size is unsafe")
    return path


def _bound_archive(path: Path, contract: ArchiveContract) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TechqaP0QualificationError("archive is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != contract.size_bytes
    ):
        raise TechqaP0QualificationError("archive metadata drifted")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise TechqaP0QualificationError(
                    "archive changed during open"
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
        raise TechqaP0QualificationError("archive read failed") from exc
    after = path.lstat()
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        or size != contract.size_bytes
        or digest.hexdigest() != contract.sha256
    ):
        raise TechqaP0QualificationError("archive byte identity drifted")
    return digest.hexdigest()


def _open_archive(path: Path) -> tuple[BinaryIO, tarfile.TarFile]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    handle: BinaryIO | None = None
    try:
        descriptor = os.open(path, flags)
        handle = os.fdopen(descriptor, "rb")
        archive = tarfile.open(fileobj=handle, mode="r|gz")
    except (OSError, tarfile.TarError) as exc:
        if handle is not None:
            handle.close()
        raise TechqaP0QualificationError(
            "archive cannot be opened as streaming gzip tar"
        ) from exc
    return handle, archive


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum: int,
) -> bytes:
    if not member.isreg() or not 0 < member.size <= maximum:
        raise TechqaP0QualificationError(
            "target tar member size or type drifted"
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise TechqaP0QualificationError("target tar member disappeared")
    parts: list[bytes] = []
    total = 0
    while True:
        raw = extracted.read(READ_CHUNK_BYTES)
        if not raw:
            break
        total += len(raw)
        if total > maximum:
            raise TechqaP0QualificationError(
                "target tar member exceeded its bound"
            )
        parts.append(raw)
    if total != member.size:
        raise TechqaP0QualificationError("target tar member was truncated")
    return b"".join(parts)


def _fresh_private_directory(path: Path) -> Path:
    root = path.absolute()
    if root.exists() or root.is_symlink():
        raise TechqaP0QualificationError(
            "qualified source directory is not fresh"
        )
    try:
        root.mkdir(mode=0o700)
    except OSError as exc:
        raise TechqaP0QualificationError(
            "qualified source directory cannot be created"
        ) from exc
    if (
        root.is_symlink()
        or not root.is_dir()
        or stat.S_IMODE(root.stat().st_mode) != 0o700
    ):
        raise TechqaP0QualificationError(
            "qualified source directory mode drifted"
        )
    return root


def _open_exclusive_private_file(path: Path) -> int:
    if path.exists() or path.is_symlink():
        raise TechqaP0QualificationError(
            "qualified source member already exists"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor
    except OSError as exc:
        raise TechqaP0QualificationError(
            "qualified source member cannot be created"
        ) from exc


def _persist_exact_member(path: Path, raw: bytes) -> None:
    descriptor = _open_exclusive_private_file(path)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise TechqaP0QualificationError(
                    "qualified source member persistence stalled"
                )
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise TechqaP0QualificationError(
            "qualified source member persistence failed"
        ) from exc
    finally:
        os.close(descriptor)


def _verify_private_member(
    path: Path,
    *,
    size_bytes: int,
    sha256: str,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TechqaP0QualificationError(
            "qualified source member is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size != size_bytes
        or _hash_file(path) != sha256
    ):
        raise TechqaP0QualificationError(
            "qualified source member byte identity drifted"
        )


def _verify_private_source_directory(
    root: Path,
    *,
    member_receipts: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        root.is_symlink()
        or not root.is_dir()
        or stat.S_IMODE(root.stat().st_mode) != 0o700
    ):
        raise TechqaP0QualificationError(
            "qualified source directory drifted"
        )
    try:
        observed = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise TechqaP0QualificationError(
            "qualified source directory cannot be enumerated"
        ) from exc
    if observed != TARGET_BASENAMES:
        raise TechqaP0QualificationError(
            "qualified source directory is not the exact whitelist"
        )
    for basename in sorted(TARGET_BASENAMES):
        receipt = member_receipts[basename]
        _verify_private_member(
            root / basename,
            size_bytes=receipt["size_bytes"],
            sha256=receipt["content_sha256"],
        )


def _collect_query_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], dict[str, int], int]:
    query_members: dict[str, bytes] = {}
    target_sizes: dict[str, int] = {}
    target_seen: set[str] = set()
    regular_count = 0
    handle, archive = _open_archive(archive_path)
    try:
        for member in archive:
            path = _validate_tar_member(member)
            if member.isreg():
                regular_count += 1
            basename = path.name
            if basename not in TARGET_BASENAMES:
                continue
            if basename in target_seen:
                raise TechqaP0QualificationError(
                    "tar target basename is duplicated"
                )
            if not member.isreg():
                raise TechqaP0QualificationError(
                    "tar target is not a regular file"
                )
            target_seen.add(basename)
            target_sizes[basename] = member.size
            if basename != CORPUS_BASENAME:
                query_members[basename] = _read_member(
                    archive,
                    member,
                    maximum=MAX_QUERY_MEMBER_BYTES,
                )
    except (OSError, tarfile.TarError) as exc:
        raise TechqaP0QualificationError(
            "tar topology stream failed"
        ) from exc
    finally:
        archive.close()
        handle.close()
    if target_seen != TARGET_BASENAMES:
        raise TechqaP0QualificationError(
            "tar does not contain exactly the three whitelisted basenames"
        )
    if target_sizes[CORPUS_BASENAME] > MAX_CORPUS_MEMBER_BYTES:
        raise TechqaP0QualificationError(
            "corpus member exceeds the frozen bound"
        )
    return query_members, target_sizes, regular_count


def _offset(value: object, *, field: str) -> int:
    if type(value) is int:
        result = value
    elif (
        isinstance(value, str)
        and value
        and value == value.strip()
        and re.fullmatch(r"[0-9]+", value) is not None
    ):
        result = int(value)
    else:
        raise TechqaP0QualificationError(f"{field} is not an offset")
    if result < 0 or result > 100_000_000:
        raise TechqaP0QualificationError(f"{field} is outside its bound")
    return result


def _observe_queries(
    query_members: Mapping[str, bytes],
    *,
    minimum_family_counts: Mapping[str, Mapping[str, int]],
) -> _QueryObservation:
    needs: list[_QueryNeed] = []
    referenced_ids: set[str] = set()
    split_receipts: dict[str, Any] = {}
    all_query_ids: set[str] = set()
    for split, basename in (
        ("TRAIN", TRAIN_QA_BASENAME),
        ("DEV", DEV_QA_BASENAME),
    ):
        rows = _strict_json_bytes(query_members[basename])
        if not isinstance(rows, list) or not rows:
            raise TechqaP0QualificationError(
                f"{split} QA root is not a nonempty array"
            )
        answerable_counts: Counter[str] = Counter()
        pre_corpus_family_counts: Counter[str] = Counter()
        query_action_ineligible_counts: Counter[str] = Counter()
        doc_id_cardinality: Counter[int] = Counter()
        unknown_keys: Counter[str] = Counter()
        row_keysets: Counter[str] = Counter()
        split_query_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise TechqaP0QualificationError("QA row is not an object")
            if not QUERY_REQUIRED_KEYS <= set(row):
                raise TechqaP0QualificationError(
                    "QA row is missing an official required field"
                )
            query_id = _identifier(
                row.get("QUESTION_ID"),
                field="QUESTION_ID",
            )
            if query_id in split_query_ids or query_id in all_query_ids:
                raise TechqaP0QualificationError(
                    "QUESTION_ID is duplicated within or across splits"
                )
            split_query_ids.add(query_id)
            all_query_ids.add(query_id)
            title = _required_text(
                row.get("QUESTION_TITLE"),
                field="QUESTION_TITLE",
                maximum=20_000,
            )
            text = _required_text(
                row.get("QUESTION_TEXT"),
                field="QUESTION_TEXT",
                maximum=100_000,
                allow_empty=True,
            )
            try:
                serialized_query = formal.core.serialize_query_text(
                    title,
                    text,
                )
                family: str | None = operational_family(title, text)
                normalized_query_sha256: str | None = hashlib.sha256(
                    formal.core.normalize_text(
                        serialized_query,
                        field="P0 normalized query",
                    ).encode("utf-8")
                ).hexdigest()
            except (
                TechqaP0QualificationError,
                formal.core.TechqaP1TypedCoreError,
            ):
                family = None
                normalized_query_sha256 = None
                query_action_ineligible_counts[
                    "shared_query_character_or_lexical_bound"
                ] += 1
            answerable = row.get("ANSWERABLE")
            if answerable not in {"Y", "N"}:
                raise TechqaP0QualificationError(
                    "ANSWERABLE is outside the official Y/N registry"
                )
            answerable_counts[str(answerable)] += 1
            raw_pool = row.get("DOC_IDS")
            if not isinstance(raw_pool, list):
                raise TechqaP0QualificationError("DOC_IDS is not an array")
            pool = tuple(
                _pool_identifier(value)
                for value in raw_pool
            )
            if (
                len(pool) != EXPECTED_DOC_ID_COUNT
                or len(set(pool)) != EXPECTED_DOC_ID_COUNT
            ):
                raise TechqaP0QualificationError(
                    "DOC_IDS is not an unordered unique set of exactly 50"
                )
            doc_id_cardinality[len(pool)] += 1
            allowed = set(QUERY_REQUIRED_KEYS)
            if answerable == "Y":
                if not ANSWERABLE_REQUIRED_KEYS <= set(row):
                    raise TechqaP0QualificationError(
                        "answerable QA is missing gold span fields"
                    )
                gold = _identifier(
                    row.get("DOCUMENT"),
                    field="DOCUMENT",
                )
                if gold not in set(pool):
                    raise TechqaP0QualificationError(
                        "gold DOCUMENT is absent from DOC_IDS"
                    )
                start = _offset(row.get("START_OFFSET"), field="START_OFFSET")
                end = _offset(row.get("END_OFFSET"), field="END_OFFSET")
                if end <= start:
                    raise TechqaP0QualificationError(
                        "gold answer span is empty or reversed"
                    )
                needs.append(
                    _QueryNeed(
                        split,
                        query_id,
                        family,
                        normalized_query_sha256,
                        pool,
                        gold,
                        start,
                        end,
                    )
                )
                referenced_ids.update(pool)
                if family is not None:
                    pre_corpus_family_counts[family] += 1
                allowed.update(ANSWERABLE_REQUIRED_KEYS)
            for key, value in row.items():
                if not isinstance(key, str):
                    raise TechqaP0QualificationError(
                        "QA object key is not a string"
                    )
                if key not in allowed:
                    unknown_keys[_unknown_key_bucket(key, value)] += 1
            row_keysets[stable_hash(sorted(row))] += 1
        expected = minimum_family_counts[split]
        if set(expected) != set(FAMILIES) or any(
            type(expected[family]) is not int or expected[family] < 1
            for family in FAMILIES
        ):
            raise TechqaP0QualificationError(
                "minimum family quota contract drifted"
            )
        split_receipts[split] = {
            "answerability_count": _counter(answerable_counts),
            "action_compatible_query_pre_corpus_family_count": _counter(
                pre_corpus_family_counts
            ),
            "answerable_family_minimum": dict(expected),
            "doc_ids_cardinality_histogram": _counter(doc_id_cardinality),
            "query_count": len(rows),
            "query_action_ineligible_count": _counter(
                query_action_ineligible_counts
            ),
            "row_keyset_sha256_histogram": _counter(row_keysets),
            "unknown_key_sha256_and_type_count": _counter(unknown_keys),
        }
    if len(referenced_ids) > MAX_RETAINED_CANDIDATE_OR_GOLD_IDS:
        raise TechqaP0QualificationError(
            "answerable candidate/gold ID retention cap exceeded"
        )
    receipt = {
        "answerable_query_count": len(needs),
        "candidate_or_gold_unique_document_id_count": len(referenced_ids),
        "candidate_or_gold_unique_document_id_retention_cap": (
            MAX_RETAINED_CANDIDATE_OR_GOLD_IDS
        ),
        "split_receipts": split_receipts,
    }
    return _QueryObservation(
        receipt=receipt,
        needs=tuple(needs),
        referenced_document_ids=frozenset(referenced_ids),
    )


def _observe_corpus(
    archive_path: Path,
    *,
    observation: _QueryObservation,
    minimum_family_counts: Mapping[str, Mapping[str, int]],
    qualified_corpus_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    int,
    dict[str, list[dict[str, str]]],
]:
    missing_ids = set(observation.referenced_document_ids)
    gold_needs: defaultdict[str, list[_QueryNeed]] = defaultdict(list)
    for need in observation.needs:
        gold_needs[need.gold_document_id].append(need)
    incompatible_ids: set[str] = set()
    validated_need_count = 0
    referenced_document_count = 0
    document_count = 0
    unknown_keys: Counter[str] = Counter()
    document_keysets: Counter[str] = Counter()
    referenced_serialized_sha256s: Counter[str] = Counter()
    seen_referenced: set[str] = set()
    corpus_sha256: str | None = None
    corpus_size: int | None = None
    corpus_seen = False
    handle, archive = _open_archive(archive_path)
    try:
        for member in archive:
            path = _validate_tar_member(member)
            if path.name != CORPUS_BASENAME:
                continue
            if corpus_seen:
                raise TechqaP0QualificationError(
                    "corpus target basename is duplicated"
                )
            corpus_seen = True
            if (
                not member.isreg()
                or not 0 < member.size <= MAX_CORPUS_MEMBER_BYTES
            ):
                raise TechqaP0QualificationError(
                    "corpus target size or type drifted"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise TechqaP0QualificationError(
                    "corpus target disappeared"
                )
            copy_descriptor = _open_exclusive_private_file(
                qualified_corpus_path
            )
            hashing = _HashingReader(
                extracted,
                copy_descriptor=copy_descriptor,
            )
            try:
                for root_document_id, raw_document in _ijson_kvitems(hashing):
                    document_count += 1
                    document_id = _identifier(
                        root_document_id,
                        field="corpus root document ID",
                    )
                    if not isinstance(raw_document, dict):
                        raise TechqaP0QualificationError(
                            "corpus document is not an object"
                        )
                    if not DOCUMENT_REQUIRED_KEYS <= set(raw_document):
                        raise TechqaP0QualificationError(
                            "corpus document is missing an official required field"
                        )
                    embedded_id = _identifier(
                        raw_document.get("_id"),
                        field="document _id",
                    )
                    if embedded_id != document_id:
                        raise TechqaP0QualificationError(
                            "corpus root ID and document _id disagree"
                        )
                    title = _required_text(
                        raw_document.get("title"),
                        field="document title",
                        maximum=MAX_JSON_VALUE_CHARACTERS,
                    )
                    text = _required_text(
                        raw_document.get("text"),
                        field="document text",
                        maximum=MAX_JSON_VALUE_CHARACTERS,
                    )
                    # ``_required_text`` already proves both public fields are
                    # nonempty; neither value survives the current iteration.
                    for key, value in raw_document.items():
                        if not isinstance(key, str):
                            raise TechqaP0QualificationError(
                                "document object key is not a string"
                            )
                        if key not in DOCUMENT_REQUIRED_KEYS:
                            unknown_keys[_unknown_key_bucket(key, value)] += 1
                    document_keysets[stable_hash(sorted(raw_document))] += 1
                    if document_id not in observation.referenced_document_ids:
                        continue
                    if document_id in seen_referenced:
                        raise TechqaP0QualificationError(
                            "referenced corpus document ID is duplicated"
                        )
                    seen_referenced.add(document_id)
                    missing_ids.discard(document_id)
                    referenced_document_count += 1
                    referenced_serialized_sha256s[
                        hashlib.sha256(
                            (title + "\n\n" + text).encode("utf-8")
                        ).hexdigest()
                    ] += 1
                    try:
                        formal.core.Document(
                            ordinal=0,
                            title=title,
                            text=text,
                        )
                    except formal.core.TechqaP1TypedCoreError:
                        # This is an item-eligibility fact, not a source-wide
                        # schema failure.  Gold-span validation still runs.
                        incompatible_ids.add(document_id)
                    for need in gold_needs.get(document_id, ()):
                        if (
                            need.end_offset > len(text)
                            or not text[
                                need.start_offset : need.end_offset
                            ].strip()
                        ):
                            raise TechqaP0QualificationError(
                                "gold answer span is outside or empty in "
                                "document text"
                            )
                        validated_need_count += 1
                corpus_sha256 = hashing.sha256
                corpus_size = hashing.size
            finally:
                try:
                    os.fsync(copy_descriptor)
                finally:
                    os.close(copy_descriptor)
            if corpus_size != member.size:
                raise TechqaP0QualificationError(
                    "corpus member was not consumed exactly"
                )
            _verify_private_member(
                qualified_corpus_path,
                size_bytes=corpus_size,
                sha256=corpus_sha256,
            )
    except (OSError, tarfile.TarError) as exc:
        raise TechqaP0QualificationError(
            "corpus tar stream failed"
        ) from exc
    finally:
        archive.close()
        handle.close()
    if (
        not corpus_seen
        or corpus_sha256 is None
        or corpus_size is None
        or validated_need_count != len(observation.needs)
    ):
        raise TechqaP0QualificationError(
            "gold corpus coverage or span validation is incomplete"
        )
    if any(need.gold_document_id in missing_ids for need in observation.needs):
        raise TechqaP0QualificationError(
            "answerable gold document is absent from the corpus"
        )
    duplicate_classes = sum(
        count > 1 for count in referenced_serialized_sha256s.values()
    )
    duplicate_ids = sum(
        count
        for count in referenced_serialized_sha256s.values()
        if count > 1
    )
    if duplicate_classes or duplicate_ids:
        raise TechqaP0QualificationError(
            "candidate document IDs do not map one-to-one to official bytes"
        )

    otherwise_eligible: list[_QueryNeed] = []
    split_hashes: defaultdict[str, set[str]] = defaultdict(set)
    split_ineligible: dict[str, Counter[str]] = {
        split: Counter() for split in ("TRAIN", "DEV")
    }
    for need in observation.needs:
        if need.family is None or need.normalized_query_sha256 is None:
            split_ineligible[need.split][
                "shared_query_character_or_lexical_bound"
            ] += 1
            continue
        candidate_ids = set(need.candidate_document_ids)
        if candidate_ids & missing_ids:
            split_ineligible[need.split][
                "candidate_document_missing_from_corpus"
            ] += 1
            continue
        if candidate_ids & incompatible_ids:
            split_ineligible[need.split][
                "candidate_document_shared_character_or_lexical_bound"
            ] += 1
            continue
        otherwise_eligible.append(need)
        split_hashes[need.split].add(need.normalized_query_sha256)

    cross_split_hashes = (
        split_hashes["TRAIN"] & split_hashes["DEV"]
    )
    eligible_by_split_family: dict[
        str, defaultdict[str, list[_QueryNeed]]
    ] = {
        split: defaultdict(list) for split in ("TRAIN", "DEV")
    }
    for need in otherwise_eligible:
        assert need.family is not None
        assert need.normalized_query_sha256 is not None
        if need.normalized_query_sha256 in cross_split_hashes:
            split_ineligible[need.split][
                "normalized_query_bytes_overlap_across_splits"
            ] += 1
            continue
        eligible_by_split_family[need.split][need.family].append(need)

    post_corpus_receipts: dict[str, Any] = {}
    private_eligible_rows: dict[str, list[dict[str, str]]] = {}
    for split in ("TRAIN", "DEV"):
        expected = minimum_family_counts.get(split)
        if (
            not isinstance(expected, Mapping)
            or set(expected) != set(FAMILIES)
            or any(
                type(expected[family]) is not int
                or expected[family] < 1
                for family in FAMILIES
            )
        ):
            raise TechqaP0QualificationError(
                "minimum family quota contract drifted"
            )
        unique_counts: dict[str, int] = {}
        duplicate_class_count = 0
        duplicate_row_count = 0
        eligible_row_count = 0
        for family in FAMILIES:
            rows = eligible_by_split_family[split][family]
            eligible_row_count += len(rows)
            query_hash_counts = Counter(
                need.normalized_query_sha256 for need in rows
            )
            unique_counts[family] = len(query_hash_counts)
            duplicate_class_count += sum(
                count > 1 for count in query_hash_counts.values()
            )
            duplicate_row_count += sum(
                count
                for count in query_hash_counts.values()
                if count > 1
            )
        post_corpus_receipts[split] = {
            "eligible_answerable_row_count": eligible_row_count,
            "eligible_unique_normalized_query_family_count": unique_counts,
            "ineligible_answerable_row_reason_count": _counter(
                split_ineligible[split]
            ),
            "normalized_query_duplicate_equivalence_class_count": (
                duplicate_class_count
            ),
            "normalized_query_duplicate_row_count": duplicate_row_count,
        }
        private_eligible_rows[split] = sorted(
            (
                {
                    "family": need.family,
                    "normalized_query_sha256": (
                        need.normalized_query_sha256
                    ),
                    "question_id": need.question_id,
                }
                for family in FAMILIES
                for need in eligible_by_split_family[split][family]
            ),
            key=lambda row: row["question_id"],
        )
        if any(
            unique_counts[family] < expected[family]
            for family in FAMILIES
        ):
            raise TechqaP0QualificationError(
                f"{split} cannot satisfy frozen unique-query family capacity "
                "after corpus qualification"
            )

    receipt = {
        "answerable_gold_span_validation_count": validated_need_count,
        "candidate_document_action_incompatible_count": len(
            incompatible_ids
        ),
        "candidate_document_missing_count": len(missing_ids),
        "cross_split_normalized_query_overlap_equivalence_class_count": (
            len(cross_split_hashes)
        ),
        "document_count": document_count,
        "document_keyset_sha256_histogram": _counter(document_keysets),
        "referenced_document_count": referenced_document_count,
        "referenced_document_duplicate_count": 0,
        "referenced_serialized_document_duplicate_equivalence_class_count": (
            duplicate_classes
        ),
        "referenced_serialized_document_duplicate_id_count": duplicate_ids,
        "serialized_document_separator": "title_utf8_then_two_LF_then_text_utf8",
        "split_post_corpus_eligibility": post_corpus_receipts,
        "unknown_key_sha256_and_type_count": _counter(unknown_keys),
    }
    return (
        receipt,
        corpus_sha256,
        corpus_size,
        private_eligible_rows,
    )


def qualify_archive(
    *,
    archive_path: Path,
    qualified_source_root: Path,
    eligibility_manifest_path: Path,
    archive_contract: ArchiveContract = OFFICIAL_ARCHIVE,
    minimum_family_counts: Mapping[
        str, Mapping[str, int]
    ] = MINIMUM_ANSWERABLE_FAMILY_COUNTS,
) -> dict[str, Any]:
    """Qualify one already-downloaded pinned archive without extraction."""

    archive_path = archive_path.absolute()
    qualified_source_root = qualified_source_root.absolute()
    eligibility_manifest_path = eligibility_manifest_path.absolute()
    if (
        qualified_source_root.name != "qualified_source"
        or eligibility_manifest_path
        != qualified_source_root.parent / "eligibility.private.json"
        or eligibility_manifest_path.exists()
        or eligibility_manifest_path.is_symlink()
    ):
        raise TechqaP0QualificationError(
            "private qualified-source output topology drifted"
        )
    archive_sha256 = _bound_archive(archive_path, archive_contract)
    query_members, target_sizes, regular_member_count = (
        _collect_query_members(archive_path)
    )
    qualified_root = _fresh_private_directory(qualified_source_root)
    for basename in (TRAIN_QA_BASENAME, DEV_QA_BASENAME):
        _persist_exact_member(
            qualified_root / basename,
            query_members[basename],
        )
    query_observation = _observe_queries(
        query_members,
        minimum_family_counts=minimum_family_counts,
    )
    (
        corpus_receipt,
        corpus_sha256,
        corpus_size,
        private_eligible_rows,
    ) = _observe_corpus(
        archive_path,
        observation=query_observation,
        minimum_family_counts=minimum_family_counts,
        qualified_corpus_path=qualified_root / CORPUS_BASENAME,
    )
    member_receipts = {
        TRAIN_QA_BASENAME: {
            "content_sha256": hashlib.sha256(
                query_members[TRAIN_QA_BASENAME]
            ).hexdigest(),
            "size_bytes": target_sizes[TRAIN_QA_BASENAME],
        },
        DEV_QA_BASENAME: {
            "content_sha256": hashlib.sha256(
                query_members[DEV_QA_BASENAME]
            ).hexdigest(),
            "size_bytes": target_sizes[DEV_QA_BASENAME],
        },
        CORPUS_BASENAME: {
            "content_sha256": corpus_sha256,
            "size_bytes": corpus_size,
        },
    }
    _verify_private_source_directory(
        qualified_root,
        member_receipts=member_receipts,
    )
    eligibility_manifest = self_hashed(
        {
            "cohort_HMAC_action_qrel_evaluator_or_score_count": 0,
            "eligibility_rule_version": ELIGIBILITY_RULE_VERSION,
            "eligible_answerable_rows_by_split": private_eligible_rows,
            "eligible_row_count_by_split": {
                split: len(private_eligible_rows[split])
                for split in ("TRAIN", "DEV")
            },
            "schema": f"{VERSION}_private_eligibility_manifest_v1",
            "source_member_content_sha256": {
                basename: member_receipts[basename]["content_sha256"]
                for basename in sorted(TARGET_BASENAMES)
            },
            "study_id": STUDY_ID,
        }
    )
    eligibility_manifest_file_sha256 = _exclusive_json(
        eligibility_manifest_path,
        eligibility_manifest,
    )
    body = {
        "access_boundary": {
            "action_model_qrel_evaluator_or_score_count": 0,
            "cohort_assignment_or_secret_count": 0,
            "individual_query_document_or_span_value_output_count": 0,
            "online_or_API_evaluation_count": 0,
            "source_archive_full_extraction_count": 0,
            "source_archive_whitelisted_member_extraction_count": 3,
        },
        "archive": {
            "filename": archive_contract.filename,
            "full_file_sha256": archive_sha256,
            "regular_member_count": regular_member_count,
            "size_bytes": archive_contract.size_bytes,
            "streaming_semantic_pass_count": 2,
            "target_members": member_receipts,
            "target_whitelist_by_basename": sorted(TARGET_BASENAMES),
        },
        "qualified_source_persistence": {
            "exact_private_regular_file_count": 3,
            "full_archive_or_nonwhitelisted_member_persistence_count": 0,
            "mode": "0600",
            "member_byte_identity_verified_against_receipt_count": 3,
        },
        "private_eligibility_manifest_binding": {
            "eligible_row_count_by_split": (
                eligibility_manifest["eligible_row_count_by_split"]
            ),
            "file_sha256": eligibility_manifest_file_sha256,
            "self_sha256": eligibility_manifest["self_sha256"],
        },
        "classifier": {
            "families": list(FAMILIES),
            "procedure_indicators": list(PROCEDURE_INDICATORS),
            "selection_or_aggregate_only": True,
            "troubleshoot_indicators": list(TROUBLESHOOT_INDICATORS),
            "troubleshoot_priority": True,
            "word_boundary_semantics": "unicode_word_boundary_exact_phrase_v1",
        },
        "corpus_aggregate": corpus_receipt,
        "official_code_binding": {
            "commit": IBM_COMMIT,
            "repository": IBM_REPOSITORY,
            "tree": IBM_TREE,
        },
        "query_aggregate": dict(query_observation.receipt),
        "schema": f"{VERSION}_safe_aggregate_receipt",
        "status": "qualified_public_non_scoring_schema_and_family_capacity",
        "study_id": STUDY_ID,
    }
    return self_hashed(body)


def _exclusive_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> str:
    raw = canonical_bytes(value, newline=True)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise TechqaP0QualificationError(
            "one-shot artifact already exists"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise TechqaP0QualificationError(
            "one-shot artifact write failed"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != mode
        or path.read_bytes() != raw
    ):
        raise TechqaP0QualificationError(
            "one-shot artifact verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(raw)
    return digest.hexdigest()


def _verify_runtime_tree(
    runtime_root: Path,
    manifest_path: Path,
) -> None:
    try:
        lines = manifest_path.read_text("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TechqaP0QualificationError(
            "runtime file manifest cannot be parsed"
        ) from exc
    declared: dict[str, str] = {}
    root_prefix = str(runtime_root) + os.sep
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (/[^\x00\r\n]+)", line)
        if match is None:
            raise TechqaP0QualificationError(
                "runtime file manifest row drifted"
            )
        digest, raw_path = match.groups()
        path = Path(raw_path)
        if (
            raw_path in declared
            or not raw_path.startswith(root_prefix)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise TechqaP0QualificationError(
                "runtime manifest path escaped or drifted"
            )
        declared[raw_path] = digest
    observed_files: list[str] = []
    observed_links: dict[str, str] = {}
    for current, directories, filenames in os.walk(
        runtime_root,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in tuple(directories) + tuple(filenames):
            path = current_path / name
            relative = path.relative_to(runtime_root).as_posix()
            if path.is_symlink():
                observed_links[relative] = os.readlink(path)
            elif path.is_file():
                observed_files.append(str(path))
    if set(observed_files) != set(declared):
        raise TechqaP0QualificationError(
            "runtime regular-file set differs from its manifest"
        )
    # The frozen path-list digest was recorded from the declared manifest
    # order.  That order is already covered by the byte-exact manifest hash;
    # re-sorting with Python's code-point order would silently substitute a
    # different collation for the one used to create the frozen manifest.
    declared_paths = list(declared)
    path_list_raw = (
        "".join(path + "\n" for path in declared_paths).encode("utf-8")
    )
    if (
        hashlib.sha256(path_list_raw).hexdigest()
        != PINNED_HF_RUNTIME_REGULAR_PATH_LIST_SHA256
    ):
        raise TechqaP0QualificationError(
            "runtime regular path-list identity drifted"
        )
    if observed_links != PINNED_HF_RUNTIME_SYMLINKS:
        raise TechqaP0QualificationError("runtime symlink topology drifted")
    for raw_path in declared_paths:
        if _hash_file(Path(raw_path)) != declared[raw_path]:
            raise TechqaP0QualificationError(
                "runtime regular-file byte identity drifted"
            )
    python = runtime_root / "bin" / "python"
    try:
        resolved_python = python.resolve(strict=True)
    except OSError as exc:
        raise TechqaP0QualificationError(
            "runtime Python symlink chain is unavailable"
        ) from exc
    if (
        str(resolved_python) != PINNED_SYSTEM_PYTHON_RESOLVED
        or _hash_file(resolved_python) != PINNED_SYSTEM_PYTHON_SHA256
    ):
        raise TechqaP0QualificationError(
            "runtime resolved Python identity drifted"
        )


def _runtime_versions(
    runtime_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> Mapping[str, str]:
    if (
        not runtime_root.is_absolute()
        or runtime_root.is_symlink()
        or not runtime_root.is_dir()
        or not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or stat.S_IMODE(manifest_path.stat().st_mode) != 0o600
        or _HEX64.fullmatch(manifest_sha256) is None
        or _hash_file(manifest_path) != manifest_sha256
    ):
        raise TechqaP0QualificationError(
            "pinned Hugging Face runtime manifest drifted"
        )
    _verify_runtime_tree(runtime_root, manifest_path)
    python = runtime_root / "bin" / "python"
    hf = runtime_root / "bin" / "hf"
    if not python.exists() or not hf.is_file() or not os.access(hf, os.X_OK):
        raise TechqaP0QualificationError(
            "pinned Hugging Face runtime executables are unavailable"
        )
    expression = (
        "import importlib.metadata as m,json,platform;"
        "print(json.dumps({"
        "'python':platform.python_version(),"
        "'huggingface_hub':m.version('huggingface_hub'),"
        "'hf_xet':m.version('hf_xet'),"
        "'click':m.version('click'),"
        "'ijson':m.version('ijson')"
        "},sort_keys=True,separators=(',',':')))"
    )
    environment = {
        "HOME": str(runtime_root.parent / "source_free_runtime_check_home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{runtime_root / 'bin'}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", "-c", expression],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        versions = json.loads(completed.stdout.decode("ascii"))
    except (
        OSError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise TechqaP0QualificationError(
            "pinned Hugging Face runtime cannot be attested"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(versions, dict)
        or versions != PINNED_HF_RUNTIME
    ):
        raise TechqaP0QualificationError(
            "pinned Hugging Face runtime version drifted"
        )
    return {str(key): str(value) for key, value in versions.items()}


DownloadRunner = Callable[
    [Sequence[str], Mapping[str, str], Path],
    subprocess.CompletedProcess[bytes],
]


def _run_download(
    command: Sequence[str],
    environment: Mapping[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=None,
        )
    except OSError as exc:
        raise TechqaP0QualificationError(
            "single Hugging Face download invocation failed to launch"
        ) from exc


def acquire_and_qualify(
    *,
    work_root: Path,
    hf_runtime_root: Path,
    hf_runtime_manifest: Path,
    hf_runtime_manifest_sha256: str,
    download_runner: DownloadRunner = _run_download,
) -> dict[str, Any]:
    """Claim, download exactly once, then run the frozen local qualifier."""

    root = work_root.absolute()
    # This is a source-free, cache-free read-only attestation.  A failure
    # leaves the one-shot work root completely absent.
    versions = _runtime_versions(
        hf_runtime_root.absolute(),
        hf_runtime_manifest.absolute(),
        hf_runtime_manifest_sha256,
    )
    if root.exists() or root.is_symlink():
        raise TechqaP0QualificationError("P0 work root is not fresh")
    root.mkdir(parents=True, mode=0o700)
    if root.is_symlink() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise TechqaP0QualificationError("P0 work root mode drifted")
    stage = "attempt_marker"
    marker = self_hashed(
        {
            "archive_filename": ARCHIVE_FILENAME,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            "hf_repository_id": HF_REPOSITORY_ID,
            "hf_revision": HF_REVISION,
            "hf_runtime_manifest_sha256": hf_runtime_manifest_sha256,
            "hf_runtime_versions": dict(versions),
            "mirror_provider_revision_or_file_switch_authorized": False,
            "network_call_count_before_marker": 0,
            "qualification_or_source_body_access_count_before_marker": 0,
            "retry_resume_or_second_invocation_authorized": False,
            "schema": f"{VERSION}_one_shot_attempt_marker",
            "study_id": STUDY_ID,
        }
    )
    marker_file_sha = _exclusive_json(root / "attempt.marker.json", marker)
    source_root = root / "source"
    hf_home = root / "hf_home"
    private_home = root / "home"
    for directory in (source_root, hf_home, private_home):
        directory.mkdir(mode=0o700)
    hf_executable = hf_runtime_root.absolute() / "bin" / "hf"
    command = (
        str(hf_executable),
        "download",
        HF_REPOSITORY_ID,
        ARCHIVE_FILENAME,
        "--repo-type",
        "dataset",
        "--revision",
        HF_REVISION,
        "--local-dir",
        str(source_root),
    )
    environment = {
        "HF_ENDPOINT": "https://huggingface.co",
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_XET": "0",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HOME": str(private_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{hf_runtime_root.absolute() / 'bin'}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "XDG_CACHE_HOME": str(hf_home),
    }
    stage = "single_pinned_hf_download"
    try:
        completed = download_runner(command, environment, root)
        if not isinstance(completed, subprocess.CompletedProcess):
            raise TechqaP0QualificationError(
                "download runner contract drifted"
            )
        download_receipt = self_hashed(
            {
                "command_argv_sha256": stable_hash(list(command)),
                "invocation_count": 1,
                "provider": "huggingface.co",
                "returncode": completed.returncode,
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "schema": f"{VERSION}_private_download_receipt",
                "study_id": STUDY_ID,
            }
        )
        _exclusive_json(root / "download.private.json", download_receipt)
        if completed.returncode != 0:
            raise TechqaP0QualificationError(
                "single pinned Hugging Face download failed"
            )
        archive = source_root / ARCHIVE_FILENAME
        stage = "archive_identity_and_local_qualification"
        receipt = qualify_archive(
            archive_path=archive,
            qualified_source_root=root / "qualified_source",
            eligibility_manifest_path=root / "eligibility.private.json",
        )
        result_file_sha = _exclusive_json(
            root / "qualification.result.json",
            receipt,
        )
        terminal = self_hashed(
            {
                "attempt_marker_file_sha256": marker_file_sha,
                "formal_P1_capability_consumed": False,
                "online_or_API_evaluation_count": 0,
                "qualification_result_file_sha256": result_file_sha,
                "qualification_result_self_sha256": receipt["self_sha256"],
                "retry_resume_mirror_provider_revision_or_file_switch_count": 0,
                "schema": f"{VERSION}_safe_terminal",
                "status": "qualified_public_non_scoring_source",
                "study_id": STUDY_ID,
            }
        )
        _exclusive_json(root / "p0_terminal.json", terminal)
        return terminal
    except BaseException as exc:
        terminal_path = root / "p0_terminal.json"
        if not terminal_path.exists() and not terminal_path.is_symlink():
            failure = self_hashed(
                {
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "exception_type_sha256": hashlib.sha256(
                        type(exc).__qualname__.encode("utf-8")
                    ).hexdigest(),
                    "formal_P1_capability_consumed": False,
                    "mirror_provider_revision_or_file_switch_authorized": False,
                    "online_evaluation_fallback_authorized": False,
                    "retry_resume_or_second_invocation_authorized": False,
                    "schema": f"{VERSION}_safe_failure_terminal",
                    "stage": stage,
                    "status": "implementation_source_or_infrastructure_invalid",
                    "study_id": STUDY_ID,
                }
            )
            _exclusive_json(terminal_path, failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify = subparsers.add_parser("qualify-local")
    qualify.add_argument("--archive", required=True, type=Path)
    qualify.add_argument(
        "--qualified-source-root",
        required=True,
        type=Path,
    )
    qualify.add_argument(
        "--eligibility-manifest",
        required=True,
        type=Path,
    )
    qualify.add_argument("--output", required=True, type=Path)
    acquire = subparsers.add_parser("acquire-and-qualify")
    acquire.add_argument("--work-root", required=True, type=Path)
    acquire.add_argument(
        "--hf-runtime-root",
        required=True,
        type=Path,
    )
    acquire.add_argument(
        "--hf-runtime-manifest",
        required=True,
        type=Path,
    )
    acquire.add_argument(
        "--hf-runtime-manifest-sha256",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "qualify-local":
        receipt = qualify_archive(
            archive_path=arguments.archive,
            qualified_source_root=arguments.qualified_source_root,
            eligibility_manifest_path=arguments.eligibility_manifest,
        )
        _exclusive_json(arguments.output.absolute(), receipt)
        safe = {
            "schema": receipt["schema"],
            "self_sha256": receipt["self_sha256"],
            "status": receipt["status"],
        }
    else:
        terminal = acquire_and_qualify(
            work_root=arguments.work_root,
            hf_runtime_root=arguments.hf_runtime_root,
            hf_runtime_manifest=arguments.hf_runtime_manifest,
            hf_runtime_manifest_sha256=(
                arguments.hf_runtime_manifest_sha256
            ),
        )
        safe = {
            "schema": terminal["schema"],
            "self_sha256": terminal["self_sha256"],
            "status": terminal["status"],
        }
    print(canonical_bytes(safe).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_FILENAME",
    "ARCHIVE_LFS_OID_SHA256",
    "ARCHIVE_POINTER_GIT_BLOB_SHA1",
    "ARCHIVE_SHA256",
    "ARCHIVE_SIZE_BYTES",
    "ArchiveContract",
    "CORPUS_BASENAME",
    "DEV_QA_BASENAME",
    "ELIGIBILITY_RULE_VERSION",
    "FAMILIES",
    "HF_REPOSITORY_ID",
    "HF_REVISION",
    "IBM_COMMIT",
    "IBM_LICENSE_GIT_BLOB_SHA1",
    "IBM_LICENSE_SHA256",
    "IBM_LICENSE_SPDX",
    "IBM_REPOSITORY",
    "IBM_TREE",
    "INFORMATION",
    "MINIMUM_ANSWERABLE_FAMILY_COUNTS",
    "OFFICIAL_ARCHIVE",
    "PROCEDURE",
    "PROCEDURE_INDICATORS",
    "PINNED_HF_RUNTIME",
    "PINNED_HF_RUNTIME_MANIFEST_SHA256",
    "STUDY_ID",
    "TRAIN_QA_BASENAME",
    "TROUBLESHOOT",
    "TROUBLESHOOT_INDICATORS",
    "TechqaP0QualificationError",
    "VERSION",
    "acquire_and_qualify",
    "canonical_bytes",
    "main",
    "operational_family",
    "qualify_archive",
    "self_hashed",
    "stable_hash",
]
