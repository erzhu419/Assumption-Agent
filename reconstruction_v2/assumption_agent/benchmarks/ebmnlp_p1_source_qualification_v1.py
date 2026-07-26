"""Private, one-shot source qualification for the frozen EBM-NLP P1 study.

The module deliberately separates three kinds of access:

* archive-header qualification lists member names and validates the exact
  document/annotation topology without opening annotation payloads;
* acquisition reads document ``.tokens``/``.text`` payloads, checks source
  identity, creates the HMAC assignment, and extracts only selected documents
  into a newly-created private directory;
* stage label opening reads only the labels for one authorized block.  Labels
  for ``F_search`` are unconditionally inaccessible and ``M_search`` labels
  additionally require an affirmative promotion capability.

No ``tarfile.extract`` operation is used.  Every persisted path is constructed
from a validated numeric PMID, directories are mode 0700, and files are mode
0600.  A failed acquisition or authorized label-open attempt consumes its
exclusive marker; this module contains no retry, reserve, replacement, or
resampling path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
from types import MappingProxyType
from typing import Any


VERSION = "ebmnlp_p1_source_qualification_v1"
STUDY_ID = "EBMNLP_P1_TYPED_PICO_SET_EVALUATOR_V1"

ARCHIVE_ROOT = "ebm_nlp_2_00"
ROLE_ORDER = ("participants", "interventions", "outcomes")
OFFICIAL_SPLITS = ("TRAIN", "TEST")
BLOCK_ORDER = ("G_form", "A_form", "F_search", "A_hold", "M_search")
BLOCK_SPLIT = {
    "G_form": "TRAIN",
    "A_form": "TRAIN",
    "F_search": "TRAIN",
    "A_hold": "TEST",
    "M_search": "TEST",
}

_PMID_PATTERN = r"(?:[1-9][0-9]{0,15})"
_DOCUMENT_RE = re.compile(
    rf"{re.escape(ARCHIVE_ROOT)}/documents/(?P<pmid>{_PMID_PATTERN})"
    r"\.(?P<kind>tokens|text)\Z"
)
_ANCILLARY_DOCUMENT_RE = re.compile(
    rf"{re.escape(ARCHIVE_ROOT)}/documents/{_PMID_PATTERN}\.pos\Z"
)
_LABEL_RE = re.compile(
    rf"{re.escape(ARCHIVE_ROOT)}/annotations/aggregated/starting_spans/"
    rf"(?P<role>{'|'.join(ROLE_ORDER)})/"
    rf"(?P<location>train|test/gold)/(?P<pmid>{_PMID_PATTERN})\.ann\Z"
)
_LABEL_TOKEN_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ASCII_LABEL_WHITESPACE = frozenset(" \t\r\n\v\f")
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")


class EbmNlpP1SourceQualificationError(RuntimeError):
    """The frozen archive, private state, or stage capability failed closed."""


@dataclass(frozen=True)
class BlockCounts:
    """Prospectively frozen abstract counts for all five disjoint blocks."""

    G_form: int
    A_form: int
    F_search: int
    A_hold: int
    M_search: int

    def as_dict(self) -> dict[str, int]:
        return {block: int(getattr(self, block)) for block in BLOCK_ORDER}


@dataclass(frozen=True)
class QualificationContract:
    """Exact source and capacity contract.

    ``require_private_archive`` is true for formal execution.  Synthetic tests
    can still satisfy it by setting their fixture to mode 0600.
    """

    archive_sha256: str
    archive_size_bytes: int
    total_public_abstract_count: int
    train_abstract_count: int
    test_abstract_count: int
    blocks: BlockCounts
    study_id: str = STUDY_ID
    require_private_archive: bool = True
    maximum_archive_member_count: int = 100_000
    maximum_total_declared_member_bytes: int = 4 * 1024 * 1024 * 1024
    maximum_document_member_bytes: int = 16 * 1024 * 1024
    maximum_label_member_bytes: int = 16 * 1024 * 1024
    maximum_ignored_regular_member_bytes: int = 512 * 1024 * 1024
    maximum_tokens_per_document: int = 1_000_000


FORMAL_CONTRACT = QualificationContract(
    archive_sha256=(
        "b7357503911ba9f708d04e24c1ab3fe9e0a79833910e53e2472ed21214a44e3f"
    ),
    archive_size_bytes=16_022_194,
    total_public_abstract_count=4_993,
    train_abstract_count=4_793,
    test_abstract_count=200,
    blocks=BlockCounts(
        G_form=1_024,
        A_form=256,
        F_search=64,
        A_hold=64,
        M_search=64,
    ),
)


@dataclass(frozen=True)
class DocumentMembers:
    tokens: str
    text: str


@dataclass(frozen=True)
class HeaderInventory:
    """Private archive topology; no annotation payload has been opened."""

    archive_sha256: str
    archive_size_bytes: int
    documents: Mapping[str, DocumentMembers]
    labels: Mapping[str, Mapping[str, Mapping[str, str]]]
    pmid_split: Mapping[str, str]
    regular_member_count: int
    directory_member_count: int
    ignored_regular_member_count: int


@dataclass(frozen=True)
class DocumentRecord:
    pmid: str
    official_split: str
    token_count: int
    tokens_sha256: str
    text_sha256: str


@dataclass(frozen=True)
class PrivateAssignment:
    blocks: Mapping[str, tuple[str, ...]]
    assignment_sha256: str

    def pmids(self, block: str) -> tuple[str, ...]:
        if block not in BLOCK_ORDER:
            raise EbmNlpP1SourceQualificationError("unknown block")
        return self.blocks[block]


@dataclass(frozen=True)
class AcquisitionResult:
    """Private in-process handle returned by the unique acquisition attempt."""

    archive_path: Path
    private_root: Path
    contract: QualificationContract
    inventory: HeaderInventory
    documents: Mapping[str, DocumentRecord]
    assignment: PrivateAssignment
    receipt_path: Path


@dataclass(frozen=True)
class LabelOpenAuthorization:
    """Trusted-controller capability consumed by a stage-specific label open."""

    stage: str
    source_sha256: str
    assignment_sha256: str
    prerequisites_sealed: bool
    promotion_authorized: bool = False


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EbmNlpP1SourceQualificationError(
            "private receipt is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_contract(contract: QualificationContract) -> None:
    if (
        not isinstance(contract.archive_sha256, str)
        or _HEX64_RE.fullmatch(contract.archive_sha256) is None
        or not isinstance(contract.archive_size_bytes, int)
        or isinstance(contract.archive_size_bytes, bool)
        or contract.archive_size_bytes <= 0
        or not isinstance(contract.study_id, str)
        or not contract.study_id
        or "\x00" in contract.study_id
    ):
        raise EbmNlpP1SourceQualificationError("source contract identity drifted")
    counts = (
        contract.total_public_abstract_count,
        contract.train_abstract_count,
        contract.test_abstract_count,
        contract.maximum_archive_member_count,
        contract.maximum_total_declared_member_bytes,
        contract.maximum_document_member_bytes,
        contract.maximum_label_member_bytes,
        contract.maximum_ignored_regular_member_bytes,
        contract.maximum_tokens_per_document,
        *contract.blocks.as_dict().values(),
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts
    ):
        raise EbmNlpP1SourceQualificationError("source contract count drifted")
    if (
        contract.total_public_abstract_count
        != contract.train_abstract_count + contract.test_abstract_count
        or contract.blocks.G_form
        + contract.blocks.A_form
        + contract.blocks.F_search
        > contract.train_abstract_count
        or contract.blocks.A_hold + contract.blocks.M_search
        > contract.test_abstract_count
    ):
        raise EbmNlpP1SourceQualificationError(
            "source contract capacity is internally inconsistent"
        )


def _regular_private_file_bytes(
    path: Path, contract: QualificationContract
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EbmNlpP1SourceQualificationError(
            "frozen archive is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise EbmNlpP1SourceQualificationError(
                "frozen archive is not a single regular file"
            )
        if contract.require_private_archive and stat.S_IMODE(before.st_mode) & 0o077:
            raise EbmNlpP1SourceQualificationError(
                "frozen archive is not private mode"
            )
        if before.st_size != contract.archive_size_bytes:
            raise EbmNlpP1SourceQualificationError(
                "frozen archive size drifted"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise EbmNlpP1SourceQualificationError(
                    "frozen archive ended early"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise EbmNlpP1SourceQualificationError(
                "frozen archive grew during read"
            )
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
            raise EbmNlpP1SourceQualificationError(
                "frozen archive changed during read"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), contract.archive_sha256):
        raise EbmNlpP1SourceQualificationError(
            "frozen archive SHA256 identity drifted"
        )
    return raw


def _safe_member_name(member: tarfile.TarInfo) -> tuple[str, bool]:
    name = member.name
    is_directory = member.type == tarfile.DIRTYPE
    if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}:
        raise EbmNlpP1SourceQualificationError(
            "archive contains a link or non-file member"
        )
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
    ):
        raise EbmNlpP1SourceQualificationError("archive member path is unsafe")
    canonical = name[:-1] if is_directory and name.endswith("/") else name
    if not canonical or canonical.endswith("/"):
        raise EbmNlpP1SourceQualificationError("archive member path is not canonical")
    parts = canonical.split("/")
    if (
        any(part in {"", ".", ".."} for part in parts)
        or "/".join(parts) != canonical
        or parts[0] != ARCHIVE_ROOT
    ):
        raise EbmNlpP1SourceQualificationError(
            "archive member escapes the exact frozen root"
        )
    if not is_directory and name.endswith("/"):
        raise EbmNlpP1SourceQualificationError(
            "regular archive member has a directory path"
        )
    return canonical, is_directory


def _scan_members(
    bundle: tarfile.TarFile,
    contract: QualificationContract,
) -> tuple[HeaderInventory, dict[str, tarfile.TarInfo]]:
    seen: set[str] = set()
    documents: dict[str, dict[str, str]] = {}
    labels: dict[str, dict[str, dict[str, str]]] = {
        split: {role: {} for role in ROLE_ORDER} for split in OFFICIAL_SPLITS
    }
    member_infos: dict[str, tarfile.TarInfo] = {}
    regular_count = 0
    directory_count = 0
    ignored_count = 0
    total_declared_bytes = 0

    try:
        iterator = iter(bundle)
        for ordinal, member in enumerate(iterator, start=1):
            if ordinal > contract.maximum_archive_member_count:
                raise EbmNlpP1SourceQualificationError(
                    "archive member-count bound exceeded"
                )
            canonical, is_directory = _safe_member_name(member)
            if canonical in seen:
                raise EbmNlpP1SourceQualificationError(
                    "archive contains a duplicate member path"
                )
            seen.add(canonical)
            if is_directory:
                if member.size != 0:
                    raise EbmNlpP1SourceQualificationError(
                        "archive directory declares payload bytes"
                    )
                directory_count += 1
                continue
            if (
                not isinstance(member.size, int)
                or isinstance(member.size, bool)
                or member.size < 0
            ):
                raise EbmNlpP1SourceQualificationError(
                    "archive member size is invalid"
                )
            regular_count += 1
            total_declared_bytes += member.size
            if total_declared_bytes > contract.maximum_total_declared_member_bytes:
                raise EbmNlpP1SourceQualificationError(
                    "archive declared-byte bound exceeded"
                )

            document_match = _DOCUMENT_RE.fullmatch(canonical)
            if document_match is not None:
                if member.size > contract.maximum_document_member_bytes:
                    raise EbmNlpP1SourceQualificationError(
                        "document member byte bound exceeded"
                    )
                pmid = document_match.group("pmid")
                kind = document_match.group("kind")
                row = documents.setdefault(pmid, {})
                if kind in row:
                    raise EbmNlpP1SourceQualificationError(
                        "document member identity is duplicated"
                    )
                row[kind] = canonical
                member_infos[canonical] = member
                continue

            label_match = _LABEL_RE.fullmatch(canonical)
            if label_match is not None:
                if member.size > contract.maximum_label_member_bytes:
                    raise EbmNlpP1SourceQualificationError(
                        "annotation member byte bound exceeded"
                    )
                split = (
                    "TRAIN"
                    if label_match.group("location") == "train"
                    else "TEST"
                )
                role = label_match.group("role")
                pmid = label_match.group("pmid")
                if pmid in labels[split][role]:
                    raise EbmNlpP1SourceQualificationError(
                        "annotation member identity is duplicated"
                    )
                labels[split][role][pmid] = canonical
                member_infos[canonical] = member
                continue

            if _ANCILLARY_DOCUMENT_RE.fullmatch(canonical) is not None:
                if member.size > contract.maximum_ignored_regular_member_bytes:
                    raise EbmNlpP1SourceQualificationError(
                        "ignored POS member byte bound exceeded"
                    )
                ignored_count += 1
                continue
            if canonical == f"{ARCHIVE_ROOT}/documents" or canonical.startswith(
                f"{ARCHIVE_ROOT}/documents/"
            ):
                raise EbmNlpP1SourceQualificationError(
                    "document member does not match the exact frozen path pattern"
                )
            label_namespace = (
                f"{ARCHIVE_ROOT}/annotations/aggregated/starting_spans"
            )
            if canonical == label_namespace or canonical.startswith(
                label_namespace + "/"
            ):
                raise EbmNlpP1SourceQualificationError(
                    "annotation member does not match the exact frozen path pattern"
                )
            if member.size > contract.maximum_ignored_regular_member_bytes:
                raise EbmNlpP1SourceQualificationError(
                    "ignored regular member byte bound exceeded"
                )
            ignored_count += 1
    except (tarfile.TarError, OSError) as exc:
        raise EbmNlpP1SourceQualificationError(
            "archive headers cannot be qualified"
        ) from exc

    document_pairs: dict[str, DocumentMembers] = {}
    for pmid, row in documents.items():
        if set(row) != {"tokens", "text"}:
            raise EbmNlpP1SourceQualificationError(
                "document tokens/text pair is incomplete"
            )
        document_pairs[pmid] = DocumentMembers(
            tokens=row["tokens"], text=row["text"]
        )

    split_sets: dict[str, set[str]] = {}
    for split in OFFICIAL_SPLITS:
        role_sets = [set(labels[split][role]) for role in ROLE_ORDER]
        if not role_sets or any(values != role_sets[0] for values in role_sets[1:]):
            raise EbmNlpP1SourceQualificationError(
                "three role annotation PMID sets differ within a split"
            )
        split_sets[split] = role_sets[0]
    if split_sets["TRAIN"] & split_sets["TEST"]:
        raise EbmNlpP1SourceQualificationError(
            "a PMID occurs in both official splits"
        )
    all_label_pmids = split_sets["TRAIN"] | split_sets["TEST"]
    if set(document_pairs) != all_label_pmids:
        raise EbmNlpP1SourceQualificationError(
            "document and annotation PMID identity sets differ"
        )
    if (
        len(document_pairs) != contract.total_public_abstract_count
        or len(split_sets["TRAIN"]) != contract.train_abstract_count
        or len(split_sets["TEST"]) != contract.test_abstract_count
    ):
        raise EbmNlpP1SourceQualificationError(
            "exact public split/count contract failed"
        )

    pmid_split = {
        **{pmid: "TRAIN" for pmid in split_sets["TRAIN"]},
        **{pmid: "TEST" for pmid in split_sets["TEST"]},
    }
    immutable_labels = MappingProxyType(
        {
            split: MappingProxyType(
                {
                    role: MappingProxyType(dict(labels[split][role]))
                    for role in ROLE_ORDER
                }
            )
            for split in OFFICIAL_SPLITS
        }
    )
    inventory = HeaderInventory(
        archive_sha256=contract.archive_sha256,
        archive_size_bytes=contract.archive_size_bytes,
        documents=MappingProxyType(dict(document_pairs)),
        labels=immutable_labels,
        pmid_split=MappingProxyType(pmid_split),
        regular_member_count=regular_count,
        directory_member_count=directory_count,
        ignored_regular_member_count=ignored_count,
    )
    return inventory, member_infos


def _open_bundle(raw: bytes) -> tarfile.TarFile:
    try:
        return tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise EbmNlpP1SourceQualificationError(
            "frozen source is not the expected gzip tar archive"
        ) from exc


def qualify_archive_headers(
    archive_path: Path,
    contract: QualificationContract = FORMAL_CONTRACT,
) -> HeaderInventory:
    """Validate identity and complete member topology without payload access."""

    _validate_contract(contract)
    raw = _regular_private_file_bytes(Path(archive_path), contract)
    with _open_bundle(raw) as bundle:
        inventory, _member_infos = _scan_members(bundle, contract)
    return inventory


def _read_member(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum_bytes: int,
) -> bytes:
    if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
        raise EbmNlpP1SourceQualificationError(
            "authorized payload is not a regular member"
        )
    if member.size > maximum_bytes:
        raise EbmNlpP1SourceQualificationError(
            "authorized payload exceeds its frozen byte bound"
        )
    try:
        handle = bundle.extractfile(member)
        if handle is None:
            raise EbmNlpP1SourceQualificationError(
                "authorized member has no readable payload"
            )
        with handle:
            raw = handle.read(member.size + 1)
    except (tarfile.TarError, OSError) as exc:
        raise EbmNlpP1SourceQualificationError(
            "authorized member payload cannot be read"
        ) from exc
    if len(raw) != member.size:
        raise EbmNlpP1SourceQualificationError(
            "authorized member payload size drifted"
        )
    return raw


def parse_document_tokens(
    raw: bytes, *, maximum_tokens: int = 1_000_000
) -> tuple[str, ...]:
    """Parse the official whitespace-token file without normalization."""

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EbmNlpP1SourceQualificationError(
            "document tokens are not strict UTF-8"
        ) from exc
    if "\x00" in text:
        raise EbmNlpP1SourceQualificationError("document tokens contain NUL")
    tokens = tuple(text.split())
    if not tokens or len(tokens) > maximum_tokens:
        raise EbmNlpP1SourceQualificationError(
            "document token count is outside the frozen bound"
        )
    return tokens


def _validate_document_text(raw: bytes) -> None:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EbmNlpP1SourceQualificationError(
            "document text is not strict UTF-8"
        ) from exc
    if not text.strip() or "\x00" in text:
        raise EbmNlpP1SourceQualificationError(
            "document text is empty or contains NUL"
        )


def parse_label_payload(raw: bytes, *, expected_token_count: int) -> tuple[int, ...]:
    """Parse one starting-span annotation using its documented integer domain."""

    if (
        not isinstance(expected_token_count, int)
        or isinstance(expected_token_count, bool)
        or expected_token_count <= 0
    ):
        raise EbmNlpP1SourceQualificationError(
            "expected label length is invalid"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EbmNlpP1SourceQualificationError(
            "annotation labels are not strict UTF-8"
        ) from exc
    if any(
        not ("0" <= character <= "9")
        and character not in _ASCII_LABEL_WHITESPACE
        for character in text
    ):
        raise EbmNlpP1SourceQualificationError(
            "annotation contains a non-integer token"
        )
    pieces = text.split()
    if len(pieces) != expected_token_count:
        raise EbmNlpP1SourceQualificationError(
            "annotation length does not match document tokens"
        )
    if any(_LABEL_TOKEN_RE.fullmatch(piece) is None for piece in pieces):
        raise EbmNlpP1SourceQualificationError(
            "annotation contains a nonnegative-integer schema violation"
        )
    return tuple(int(piece) for piece in pieces)


def _private_mkdir(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise EbmNlpP1SourceQualificationError(
            "private acquisition directory cannot be created exactly once"
        ) from exc


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EbmNlpP1SourceQualificationError(
            "private one-shot file is already consumed or unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, _canonical_bytes(value) + b"\n")


def _ordered_pmids(
    pmids: Sequence[str],
    *,
    official_split: str,
    secret: bytes,
    study_id: str,
) -> tuple[str, ...]:
    scored: list[tuple[bytes, bytes, str]] = []
    for pmid in pmids:
        if re.fullmatch(_PMID_PATTERN, pmid) is None:
            raise EbmNlpP1SourceQualificationError(
                "eligible PMID is not canonical ASCII"
            )
        pmid_bytes = pmid.encode("ascii")
        message = (
            study_id.encode("utf-8")
            + b"\x00"
            + official_split.encode("ascii")
            + b"\x00"
            + pmid_bytes
        )
        scored.append(
            (hmac.new(secret, message, hashlib.sha256).digest(), pmid_bytes, pmid)
        )
    scored.sort()
    return tuple(row[2] for row in scored)


def assign_blocks(
    eligible_by_split: Mapping[str, Sequence[str]],
    *,
    secret: bytes,
    contract: QualificationContract,
) -> PrivateAssignment:
    """Create all disjoint blocks using only split names and canonical PMIDs."""

    _validate_contract(contract)
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EbmNlpP1SourceQualificationError(
            "selection secret must be exactly 32 bytes"
        )
    if set(eligible_by_split) != set(OFFICIAL_SPLITS):
        raise EbmNlpP1SourceQualificationError(
            "eligible split registry drifted"
        )
    normalized: dict[str, tuple[str, ...]] = {}
    for split in OFFICIAL_SPLITS:
        values = tuple(eligible_by_split[split])
        if len(values) != len(set(values)):
            raise EbmNlpP1SourceQualificationError(
                "eligible PMID is duplicated within a split"
            )
        normalized[split] = values
    if set(normalized["TRAIN"]) & set(normalized["TEST"]):
        raise EbmNlpP1SourceQualificationError(
            "eligible PMID occurs in both splits"
        )

    train = _ordered_pmids(
        normalized["TRAIN"],
        official_split="TRAIN",
        secret=secret,
        study_id=contract.study_id,
    )
    test = _ordered_pmids(
        normalized["TEST"],
        official_split="TEST",
        secret=secret,
        study_id=contract.study_id,
    )
    train_required = (
        contract.blocks.G_form
        + contract.blocks.A_form
        + contract.blocks.F_search
    )
    test_required = contract.blocks.A_hold + contract.blocks.M_search
    if len(train) < train_required or len(test) < test_required:
        raise EbmNlpP1SourceQualificationError(
            "eligible source capacity is insufficient without replacement"
        )

    g_end = contract.blocks.G_form
    a_end = g_end + contract.blocks.A_form
    f_end = a_end + contract.blocks.F_search
    ah_end = contract.blocks.A_hold
    m_end = ah_end + contract.blocks.M_search
    blocks = {
        "G_form": train[:g_end],
        "A_form": train[g_end:a_end],
        "F_search": train[a_end:f_end],
        "A_hold": test[:ah_end],
        "M_search": test[ah_end:m_end],
    }
    flattened = [pmid for block in BLOCK_ORDER for pmid in blocks[block]]
    if len(flattened) != len(set(flattened)):
        raise EbmNlpP1SourceQualificationError(
            "HMAC block assignment is not disjoint"
        )
    body = {
        "schema": "ebmnlp_p1_private_assignment_v1",
        "source_sha256": contract.archive_sha256,
        "study_id": contract.study_id,
        "hmac_message": "UTF8_study_id_NUL_official_split_NUL_PMID",
        "blocks": {block: list(blocks[block]) for block in BLOCK_ORDER},
    }
    assignment_hash = _stable_hash(body)
    return PrivateAssignment(
        blocks=MappingProxyType(
            {block: tuple(blocks[block]) for block in BLOCK_ORDER}
        ),
        assignment_sha256=assignment_hash,
    )


def _document_payloads(
    bundle: tarfile.TarFile,
    inventory: HeaderInventory,
    member_infos: Mapping[str, tarfile.TarInfo],
    contract: QualificationContract,
) -> tuple[dict[str, DocumentRecord], dict[str, tuple[bytes, bytes]]]:
    records: dict[str, DocumentRecord] = {}
    payloads: dict[str, tuple[bytes, bytes]] = {}
    digest_pair_owner: dict[tuple[str, str], str] = {}
    for pmid in sorted(inventory.documents, key=lambda value: value.encode("ascii")):
        members = inventory.documents[pmid]
        tokens_raw = _read_member(
            bundle,
            member_infos[members.tokens],
            maximum_bytes=contract.maximum_document_member_bytes,
        )
        text_raw = _read_member(
            bundle,
            member_infos[members.text],
            maximum_bytes=contract.maximum_document_member_bytes,
        )
        tokens = parse_document_tokens(
            tokens_raw, maximum_tokens=contract.maximum_tokens_per_document
        )
        _validate_document_text(text_raw)
        tokens_sha256 = hashlib.sha256(tokens_raw).hexdigest()
        text_sha256 = hashlib.sha256(text_raw).hexdigest()
        digest_pair = (tokens_sha256, text_sha256)
        previous = digest_pair_owner.get(digest_pair)
        if previous is not None and previous != pmid:
            raise EbmNlpP1SourceQualificationError(
                "distinct PMIDs share an exact tokens/text digest pair"
            )
        digest_pair_owner[digest_pair] = pmid
        records[pmid] = DocumentRecord(
            pmid=pmid,
            official_split=inventory.pmid_split[pmid],
            token_count=len(tokens),
            tokens_sha256=tokens_sha256,
            text_sha256=text_sha256,
        )
        payloads[pmid] = (tokens_raw, text_raw)
    return records, payloads


def acquire_once(
    *,
    archive_path: Path,
    private_root: Path,
    contract: QualificationContract = FORMAL_CONTRACT,
    secret_factory: Callable[[int], bytes] = os.urandom,
) -> AcquisitionResult:
    """Consume one private acquisition attempt and seal all block assignments.

    The output root must not already exist.  It is intentionally left in place
    after any exception so the source epoch cannot be replayed through this
    entry point.
    """

    _validate_contract(contract)
    archive_path = Path(archive_path)
    private_root = Path(private_root)
    parent = private_root.parent
    try:
        parent_mode = parent.lstat().st_mode
    except OSError as exc:
        raise EbmNlpP1SourceQualificationError(
            "private acquisition parent is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
        raise EbmNlpP1SourceQualificationError(
            "private acquisition parent is unsafe"
        )
    _private_mkdir(private_root)
    marker_path = private_root / "acquisition.attempt_consumed.json"
    _write_private_json(
        marker_path,
        {
            "schema": "ebmnlp_p1_acquisition_attempt_consumed_v1",
            "status": "consumed_before_archive_open_no_retry_or_rescue",
            "study_id": contract.study_id,
        },
    )

    raw = _regular_private_file_bytes(archive_path, contract)
    with _open_bundle(raw) as bundle:
        inventory, member_infos = _scan_members(bundle, contract)
        records, document_payloads = _document_payloads(
            bundle, inventory, member_infos, contract
        )
    try:
        secret = secret_factory(32)
    except Exception as exc:
        raise EbmNlpP1SourceQualificationError(
            "fresh selection secret generation failed"
        ) from exc
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EbmNlpP1SourceQualificationError(
            "fresh selection secret generation drifted"
        )
    assignment = assign_blocks(
        {
            split: tuple(
                pmid
                for pmid, source_split in inventory.pmid_split.items()
                if source_split == split
            )
            for split in OFFICIAL_SPLITS
        },
        secret=secret,
        contract=contract,
    )
    _write_exclusive(private_root / "selection_secret.private.bin", secret)
    assignment_body = {
        "schema": "ebmnlp_p1_private_assignment_v1",
        "source_sha256": contract.archive_sha256,
        "study_id": contract.study_id,
        "hmac_message": "UTF8_study_id_NUL_official_split_NUL_PMID",
        "blocks": {
            block: list(assignment.blocks[block]) for block in BLOCK_ORDER
        },
    }
    _write_private_json(
        private_root / "assignment.private.json",
        {**assignment_body, "assignment_sha256": assignment.assignment_sha256},
    )

    documents_root = private_root / "documents"
    _private_mkdir(documents_root)
    selected_pmids = {
        pmid for block in BLOCK_ORDER for pmid in assignment.blocks[block]
    }
    for pmid in sorted(selected_pmids, key=lambda value: value.encode("ascii")):
        tokens_raw, text_raw = document_payloads[pmid]
        _write_exclusive(documents_root / f"{pmid}.tokens", tokens_raw)
        _write_exclusive(documents_root / f"{pmid}.text", text_raw)

    receipt_body = {
        "schema": "ebmnlp_p1_private_acquisition_receipt_v1",
        "status": "qualified_assigned_documents_extracted_labels_unopened",
        "study_id": contract.study_id,
        "source_sha256": contract.archive_sha256,
        "assignment_sha256": assignment.assignment_sha256,
        "public_abstract_count": len(records),
        "official_split_abstract_counts": {
            split: sum(
                value == split for value in inventory.pmid_split.values()
            )
            for split in OFFICIAL_SPLITS
        },
        "block_counts": {
            block: len(assignment.blocks[block]) for block in BLOCK_ORDER
        },
        "document_payload_open_count": len(records) * 2,
        "annotation_payload_open_count": 0,
        "cross_split_PMID_overlap_count": 0,
        "tokens_text_digest_pair_collision_count": 0,
        "selection_inputs": "study_id_official_split_PMID_only",
        "formal_retry_resample_replacement_or_rescue_count": 0,
    }
    receipt_path = private_root / "acquisition.receipt.json"
    _write_private_json(
        receipt_path,
        {**receipt_body, "receipt_sha256": _stable_hash(receipt_body)},
    )
    return AcquisitionResult(
        archive_path=archive_path,
        private_root=private_root,
        contract=contract,
        inventory=inventory,
        documents=MappingProxyType(records),
        assignment=assignment,
        receipt_path=receipt_path,
    )


def _validate_authorization(
    result: AcquisitionResult,
    stage: str,
    authorization: LabelOpenAuthorization,
) -> None:
    if stage == "F_search":
        raise EbmNlpP1SourceQualificationError(
            "F_search annotation payload is permanently inaccessible"
        )
    if stage not in BLOCK_ORDER:
        raise EbmNlpP1SourceQualificationError("unknown label stage")
    if (
        not isinstance(authorization, LabelOpenAuthorization)
        or authorization.stage != stage
        or not hmac.compare_digest(
            authorization.source_sha256, result.inventory.archive_sha256
        )
        or not hmac.compare_digest(
            authorization.assignment_sha256,
            result.assignment.assignment_sha256,
        )
        or authorization.prerequisites_sealed is not True
    ):
        raise EbmNlpP1SourceQualificationError(
            "stage label capability is absent or drifted"
        )
    if stage == "M_search" and authorization.promotion_authorized is not True:
        raise EbmNlpP1SourceQualificationError(
            "M_search labels require valid A_hold promotion"
        )


def _verify_extracted_token_count(
    result: AcquisitionResult, pmid: str
) -> int:
    path = result.private_root / "documents" / f"{pmid}.tokens"
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise EbmNlpP1SourceQualificationError(
            "private extracted token file is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or hashlib.sha256(raw).hexdigest()
        != result.documents[pmid].tokens_sha256
    ):
        raise EbmNlpP1SourceQualificationError(
            "private extracted token file identity drifted"
        )
    tokens = parse_document_tokens(
        raw, maximum_tokens=result.contract.maximum_tokens_per_document
    )
    if len(tokens) != result.documents[pmid].token_count:
        raise EbmNlpP1SourceQualificationError(
            "private extracted token count drifted"
        )
    return len(tokens)


def open_labels_for_stage(
    result: AcquisitionResult,
    *,
    stage: str,
    authorization: LabelOpenAuthorization,
) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    """Open exactly one authorized block's three-role labels, once.

    The returned mapping is private in-memory data.  No labels or PMIDs are
    written to the aggregate completion receipt.
    """

    _validate_authorization(result, stage, authorization)
    marker_directory = result.private_root / "label_open_markers"
    if not marker_directory.exists():
        _private_mkdir(marker_directory)
    else:
        try:
            metadata = marker_directory.lstat()
        except OSError as exc:
            raise EbmNlpP1SourceQualificationError(
                "label marker directory is unavailable"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise EbmNlpP1SourceQualificationError(
                "label marker directory is unsafe"
            )
    marker = marker_directory / f"{stage}.attempt_consumed.json"
    _write_private_json(
        marker,
        {
            "schema": "ebmnlp_p1_label_open_attempt_v1",
            "stage": stage,
            "status": "consumed_before_annotation_payload_open_no_retry",
            "source_sha256": result.inventory.archive_sha256,
            "assignment_sha256": result.assignment.assignment_sha256,
        },
    )

    raw = _regular_private_file_bytes(result.archive_path, result.contract)
    with _open_bundle(raw) as bundle:
        inventory, member_infos = _scan_members(bundle, result.contract)
        if inventory.pmid_split != result.inventory.pmid_split:
            raise EbmNlpP1SourceQualificationError(
                "archive topology drifted after acquisition"
            )
        split = BLOCK_SPLIT[stage]
        private_labels: dict[str, Mapping[str, tuple[int, ...]]] = {}
        for pmid in result.assignment.pmids(stage):
            expected_token_count = _verify_extracted_token_count(result, pmid)
            role_labels: dict[str, tuple[int, ...]] = {}
            for role in ROLE_ORDER:
                member_name = inventory.labels[split][role][pmid]
                label_raw = _read_member(
                    bundle,
                    member_infos[member_name],
                    maximum_bytes=result.contract.maximum_label_member_bytes,
                )
                role_labels[role] = parse_label_payload(
                    label_raw, expected_token_count=expected_token_count
                )
            private_labels[pmid] = MappingProxyType(role_labels)

    completion = marker_directory / f"{stage}.complete.json"
    _write_private_json(
        completion,
        {
            "schema": "ebmnlp_p1_label_open_complete_v1",
            "stage": stage,
            "status": "authorized_labels_opened_once",
            "abstract_count": len(private_labels),
            "role_count": len(ROLE_ORDER),
            "label_member_open_count": len(private_labels) * len(ROLE_ORDER),
            "PMID_or_label_value_output_count": 0,
        },
    )
    return MappingProxyType(private_labels)


def open_f_search_labels(*_args: object, **_kwargs: object) -> None:
    """An explicit fail-closed API makes accidental F gold access impossible."""

    raise EbmNlpP1SourceQualificationError(
        "F_search annotation payload is permanently inaccessible"
    )


__all__ = [
    "AcquisitionResult",
    "BLOCK_ORDER",
    "BLOCK_SPLIT",
    "BlockCounts",
    "DocumentRecord",
    "EbmNlpP1SourceQualificationError",
    "FORMAL_CONTRACT",
    "HeaderInventory",
    "LabelOpenAuthorization",
    "PrivateAssignment",
    "QualificationContract",
    "ROLE_ORDER",
    "STUDY_ID",
    "acquire_once",
    "assign_blocks",
    "open_f_search_labels",
    "open_labels_for_stage",
    "parse_document_tokens",
    "parse_label_payload",
    "qualify_archive_headers",
]
