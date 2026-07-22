"""One-shot aggregate-only qualification for the pinned FanOutQA P1 source.

The formal call consumes its marker before opening either source.  It verifies
the exact release-bound DEV object, audits the official revision-cache tar as
a stream without extracting article content, and emits aggregate schema,
coverage, structural-family, and page-disjoint-capacity facts only.  It never
opens TEST, runs a model, scores an action, or invokes an online evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import unicodedata
from urllib.parse import urlsplit


VERSION = "fanoutqa_p1_source_qualification_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = Path(__file__).resolve()
TEST_PATH = PROJECT_ROOT / "tests/test_fanoutqa_p1_source_qualification_v1.py"
SOURCE_ROOT = PROJECT_ROOT / "artifacts/fanoutqa_p1_official_source_v1"
DEV_PATH = SOURCE_ROOT / "fanout-final-dev.json"
CACHE_PATH = SOURCE_ROOT / "wikicache.tar.gz"
CUSTODY_PATH = PROJECT_ROOT / "manifests/fanoutqa_p1_source_custody_v1.json"
DESIGN_PATH = (
    PROJECT_ROOT / "manifests/fanoutqa_p1_typed_fanout_e3_study_design_v1.json"
)
DOWNLOAD_RECEIPT_PATH = (
    PROJECT_ROOT / "manifests/fanoutqa_p1_source_download_receipt_v1.json"
)
FREEZE_PATH = (
    PROJECT_ROOT / "manifests/fanoutqa_p1_source_qualification_freeze_v1.json"
)
MARKER_PATH = (
    PROJECT_ROOT
    / "artifacts/fanoutqa_p1_source_qualification_v1/qualification.one_shot_marker.json"
)
FAILURE_PATH = (
    PROJECT_ROOT
    / "artifacts/fanoutqa_p1_source_qualification_v1/qualification.terminal_failure.json"
)
RESULT_PATH = PROJECT_ROOT / "manifests/fanoutqa_p1_source_qualification_result_v1.json"

OFFICIAL_RELEASE = "ccf127bd0b1e1091e98ffb9aff7dc694eaf58d54"
OFFICIAL_TREE = "2b6a01b63fda51ac237daded5347fc460105fac5"
DEV_GIT_BLOB_SHA1 = "76ad1feb689b754bfe4e5e24d3ea371b647efa67"
DEV_SHA256 = "359300b029c6891567816f351bf8786e9b018d7af8a1a44b7da9ba5ef4651288"
DEV_SIZE_BYTES = 1177174
CACHE_SIZE_BYTES = 1538812319
DEV_COUNT = 310
EXPECTED_CUSTODY_SELF_SHA256 = (
    "d0674510f876912ea097750513180db667c1c64b7995d414a0132dcb2896e3b4"
)
EXPECTED_DESIGN_SELF_SHA256 = (
    "1586a3898bbce54d428c5a91635598824fd39a5eae7fa75246adb302e4083e7a"
)
EXAMPLE_QUESTION_DENY_SHA256 = frozenset(
    {"bc7a89c9bf662eef176ae1f93f2e017637a7f366205db0d7c342c32e236edb9d"}
)

FAMILIES = ("HIERARCHICAL", "DEPENDENCY_FLAT", "PARALLEL_FLAT")
BLOCK_QUOTAS = {"A_form": 8, "F_search": 4, "A_hold": 6, "M_search": 6}
REQUIRED_PER_FAMILY = sum(BLOCK_QUOTAS.values())
MIN_EVIDENCE_PAGES = 3
MAX_EVIDENCE_PAGES = 10
MAX_CACHE_FILES = 1_000_000
MAX_CACHE_UNCOMPRESSED_BYTES = 20_000_000_000
PUBLIC_CAPACITY_DOMAIN = "fanoutqa-p1-public-page-disjoint-capacity-v1"

ALLOWED_CATEGORIES = frozenset(
    {
        "Architecture",
        "Astronomy",
        "Business",
        "Culture",
        "Demographics",
        "Economics",
        "Education",
        "Film Studies",
        "Finance",
        "Geography",
        "History",
        "International Relations",
        "Japanese Culture",
        "Law",
        "Linguistics",
        "Literature",
        "Music",
        "Other",
        "Physics",
        "Politics",
        "Sports",
        "Statistics",
        "Technology",
        "Television",
        "Video Games",
    }
)

TOP_KEYS = frozenset({"id", "question", "decomposition", "answer", "categories"})
SUB_KEYS = frozenset(
    {"id", "question", "decomposition", "answer", "depends_on", "evidence"}
)
EVIDENCE_KEYS = frozenset({"pageid", "revid", "title", "url"})
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CACHE_MEMBER = re.compile(r"(?:\./)?wikicache/([1-9][0-9]*)-dated\.md\Z")


class FanOutQaP1SourceQualificationError(RuntimeError):
    """The fixed source or aggregate-only contract failed closed."""


@dataclass(frozen=True)
class QualificationContract:
    dev_count: int
    dev_size_bytes: int
    dev_git_blob_sha1: str
    dev_sha256: str
    cache_size_bytes: int
    required_per_family: int
    min_evidence_pages: int = MIN_EVIDENCE_PAGES
    max_evidence_pages: int = MAX_EVIDENCE_PAGES
    max_cache_files: int = MAX_CACHE_FILES
    max_cache_uncompressed_bytes: int = MAX_CACHE_UNCOMPRESSED_BYTES


FORMAL_CONTRACT = QualificationContract(
    dev_count=DEV_COUNT,
    dev_size_bytes=DEV_SIZE_BYTES,
    dev_git_blob_sha1=DEV_GIT_BLOB_SHA1,
    dev_sha256=DEV_SHA256,
    cache_size_bytes=CACHE_SIZE_BYTES,
    required_per_family=REQUIRED_PER_FAMILY,
)


@dataclass(frozen=True)
class Candidate:
    question_sha256: str
    family: str
    pageids: frozenset[int]


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
        raise FanOutQaP1SourceQualificationError(
            "qualification value is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FanOutQaP1SourceQualificationError(
            "one-shot qualification path is already consumed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        if path.is_symlink() or path.read_bytes() != raw:
            raise FanOutQaP1SourceQualificationError(
                "qualification receipt reopen verification failed"
            )
    except OSError as exc:
        raise FanOutQaP1SourceQualificationError(
            "qualification receipt reopen verification failed"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _load_canonical_self_hashed(path: Path, field: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 2 * 1024 * 1024
        ):
            raise FanOutQaP1SourceQualificationError(f"{field} is unavailable")
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except FanOutQaP1SourceQualificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FanOutQaP1SourceQualificationError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise FanOutQaP1SourceQualificationError(f"{field} is invalid")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or _semantic_hash(body) != declared
    ):
        raise FanOutQaP1SourceQualificationError(f"{field} self hash drifted")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FanOutQaP1SourceQualificationError(
            "frozen implementation file is unreadable"
        ) from exc
    return digest.hexdigest()


def _normalized_question_sha256(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not normalized:
        raise FanOutQaP1SourceQualificationError("question schema drifted")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_text(value: object, *, maximum: int = 200_000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise FanOutQaP1SourceQualificationError("text schema drifted")
    return value


def _validate_answer(value: object) -> None:
    primitive = isinstance(value, (bool, int, float, str)) and not isinstance(
        value, complex
    )
    if primitive:
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            raise FanOutQaP1SourceQualificationError("answer schema drifted")
        if isinstance(value, str) and "\x00" in value:
            raise FanOutQaP1SourceQualificationError("answer schema drifted")
        return
    if isinstance(value, list):
        if not value:
            raise FanOutQaP1SourceQualificationError("answer schema drifted")
        for item in value:
            if isinstance(item, (list, dict)):
                raise FanOutQaP1SourceQualificationError("answer schema drifted")
            _validate_answer(item)
        return
    if isinstance(value, dict):
        if not value:
            raise FanOutQaP1SourceQualificationError("answer schema drifted")
        for key, item in value.items():
            _safe_text(key, maximum=10_000)
            if isinstance(item, (list, dict)):
                raise FanOutQaP1SourceQualificationError("answer schema drifted")
            _validate_answer(item)
        return
    raise FanOutQaP1SourceQualificationError("answer schema drifted")


def _validate_evidence(value: object) -> tuple[int, int]:
    if not isinstance(value, dict) or frozenset(value) != EVIDENCE_KEYS:
        raise FanOutQaP1SourceQualificationError("evidence schema drifted")
    pageid = value["pageid"]
    revid = value["revid"]
    if (
        isinstance(pageid, bool)
        or not isinstance(pageid, int)
        or pageid <= 0
        or isinstance(revid, bool)
        or not isinstance(revid, int)
        or revid <= 0
    ):
        raise FanOutQaP1SourceQualificationError("evidence identity drifted")
    _safe_text(value["title"], maximum=10_000)
    url = _safe_text(value["url"], maximum=20_000)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise FanOutQaP1SourceQualificationError(
            "evidence URL schema drifted"
        ) from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "en.wikipedia.org"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or (not parsed.path and not parsed.query)
        or any(character.isspace() for character in url)
    ):
        raise FanOutQaP1SourceQualificationError("evidence URL schema drifted")
    return pageid, revid


def _acyclic(node_ids: set[str], edges: set[tuple[str, str]]) -> None:
    outgoing = {node_id: set() for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for left, right in edges:
        if left == right or left not in node_ids or right not in node_ids:
            raise FanOutQaP1SourceQualificationError("decomposition graph drifted")
        if right not in outgoing[left]:
            outgoing[left].add(right)
            indegree[right] += 1
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for right in sorted(outgoing[node_id]):
            indegree[right] -= 1
            if indegree[right] == 0:
                ready.append(right)
                ready.sort()
    if visited != len(node_ids):
        raise FanOutQaP1SourceQualificationError("decomposition graph is cyclic")


def _parse_item(
    value: object,
) -> tuple[str, str, frozenset[int], int, int, int, str]:
    if not isinstance(value, dict) or frozenset(value) != TOP_KEYS:
        raise FanOutQaP1SourceQualificationError("DEV item schema drifted")
    item_id = _safe_text(value["id"], maximum=10_000)
    question = _safe_text(value["question"], maximum=100_000)
    question_sha256 = _normalized_question_sha256(question)
    _validate_answer(value["answer"])
    categories = value["categories"]
    if (
        not isinstance(categories, list)
        or not categories
        or any(
            not isinstance(category, str) or category not in ALLOWED_CATEGORIES
            for category in categories
        )
        or len(categories) != len(set(categories))
    ):
        raise FanOutQaP1SourceQualificationError("category schema drifted")
    decomposition = value["decomposition"]
    if not isinstance(decomposition, list) or not decomposition:
        raise FanOutQaP1SourceQualificationError("decomposition schema drifted")

    node_ids: set[str] = set()
    dependency_refs: list[tuple[str, str]] = []
    parent_edges: set[tuple[str, str]] = set()
    evidence_revisions: dict[int, set[int]] = {}
    maximum_depth = 0
    dependency_edge_count = 0

    def walk(rows: object, *, depth: int, parent: str | None) -> None:
        nonlocal maximum_depth, dependency_edge_count
        if not isinstance(rows, list) or not rows or depth > 32:
            raise FanOutQaP1SourceQualificationError("decomposition schema drifted")
        maximum_depth = max(maximum_depth, depth)
        for row in rows:
            if not isinstance(row, dict) or frozenset(row) != SUB_KEYS:
                raise FanOutQaP1SourceQualificationError(
                    "subquestion schema drifted"
                )
            node_id = _safe_text(row["id"], maximum=10_000)
            if node_id in node_ids:
                raise FanOutQaP1SourceQualificationError(
                    "subquestion identity drifted"
                )
            node_ids.add(node_id)
            if parent is not None:
                parent_edges.add((parent, node_id))
            _safe_text(row["question"], maximum=100_000)
            _validate_answer(row["answer"])
            depends_on = row["depends_on"]
            if (
                not isinstance(depends_on, list)
                or any(not isinstance(ref, str) or not ref for ref in depends_on)
                or len(depends_on) != len(set(depends_on))
            ):
                raise FanOutQaP1SourceQualificationError(
                    "dependency schema drifted"
                )
            dependency_edge_count += len(depends_on)
            dependency_refs.extend((ref, node_id) for ref in depends_on)
            children = row["decomposition"]
            evidence = row["evidence"]
            if evidence is None:
                if not isinstance(children, list) or not children:
                    raise FanOutQaP1SourceQualificationError(
                        "decomposition evidence totality drifted"
                    )
                walk(children, depth=depth + 1, parent=node_id)
            else:
                if not isinstance(children, list) or children:
                    raise FanOutQaP1SourceQualificationError(
                        "leaf evidence totality drifted"
                    )
                pageid, revid = _validate_evidence(evidence)
                evidence_revisions.setdefault(pageid, set()).add(revid)

    walk(decomposition, depth=1, parent=None)
    all_edges = set(parent_edges)
    for left, right in dependency_refs:
        if left not in node_ids:
            raise FanOutQaP1SourceQualificationError(
                "dependency reference closure drifted"
            )
        all_edges.add((left, right))
    _acyclic(node_ids, all_edges)
    if not evidence_revisions:
        raise FanOutQaP1SourceQualificationError("evidence closure drifted")
    conflicting_revision_count = sum(
        1 for revisions in evidence_revisions.values() if len(revisions) != 1
    )
    family = (
        "HIERARCHICAL"
        if maximum_depth >= 2
        else "DEPENDENCY_FLAT"
        if dependency_edge_count > 0
        else "PARALLEL_FLAT"
    )
    return (
        item_id,
        question_sha256,
        frozenset(evidence_revisions),
        conflicting_revision_count,
        maximum_depth,
        dependency_edge_count,
        family,
    )


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _bound_regular_bytes(path: Path, expected_size: int) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        path_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_dev != path_before.st_dev
            or before.st_ino != path_before.st_ino
            or before.st_size != expected_size
        ):
            raise FanOutQaP1SourceQualificationError(
                "source is not one bound regular file"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_dev != path_after.st_dev
            or after.st_ino != path_after.st_ino
        ):
            raise FanOutQaP1SourceQualificationError(
                "source changed during qualification"
            )
        return b"".join(chunks), before
    except FanOutQaP1SourceQualificationError:
        raise
    except OSError as exc:
        raise FanOutQaP1SourceQualificationError(
            "source descriptor validation failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class _DigestReader(io.RawIOBase):
    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        block = self.raw.read(size)
        if block:
            self.digest.update(block)
            self.byte_count += len(block)
        return block

    def readinto(self, target: bytearray) -> int:
        block = self.read(len(target))
        target[: len(block)] = block
        return len(block)


def _audit_cache_tar(
    path: Path, contract: QualificationContract
) -> dict[str, object]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        path_before = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != contract.cache_size_bytes
            or before.st_dev != path_before.st_dev
            or before.st_ino != path_before.st_ino
        ):
            raise FanOutQaP1SourceQualificationError(
                "cache archive is not one bound regular file"
            )
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as binary:
            digest_reader = _DigestReader(binary)
            buffered = io.BufferedReader(digest_reader, buffer_size=1024 * 1024)
            pageids: set[int] = set()
            file_count = 0
            directory_count = 0
            total_uncompressed = 0
            minimum_file_size: int | None = None
            maximum_file_size = 0
            try:
                with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
                    for member in archive:
                        name = member.name
                        pure = PurePosixPath(name)
                        if (
                            pure.is_absolute()
                            or not pure.parts
                            or ".." in pure.parts
                            or "\x00" in name
                        ):
                            raise FanOutQaP1SourceQualificationError(
                                "cache archive member path drifted"
                            )
                        if member.isdir():
                            if name.rstrip("/") not in {".", "./wikicache", "wikicache"}:
                                raise FanOutQaP1SourceQualificationError(
                                    "cache archive directory drifted"
                                )
                            directory_count += 1
                            continue
                        if not member.isfile() or member.issym() or member.islnk():
                            raise FanOutQaP1SourceQualificationError(
                                "cache archive contains a non-regular member"
                            )
                        match = _CACHE_MEMBER.fullmatch(name)
                        if match is None:
                            raise FanOutQaP1SourceQualificationError(
                                "cache archive file grammar drifted"
                            )
                        pageid = int(match.group(1))
                        if pageid in pageids:
                            raise FanOutQaP1SourceQualificationError(
                                "cache archive page identity is duplicated"
                            )
                        pageids.add(pageid)
                        file_count += 1
                        total_uncompressed += member.size
                        minimum_file_size = (
                            member.size
                            if minimum_file_size is None
                            else min(minimum_file_size, member.size)
                        )
                        maximum_file_size = max(maximum_file_size, member.size)
                        if (
                            file_count > contract.max_cache_files
                            or total_uncompressed
                            > contract.max_cache_uncompressed_bytes
                        ):
                            raise FanOutQaP1SourceQualificationError(
                                "cache archive aggregate bound exceeded"
                            )
            except (tarfile.TarError, EOFError, OSError) as exc:
                raise FanOutQaP1SourceQualificationError(
                    "cache archive stream is invalid"
                ) from exc
            while buffered.read(1024 * 1024):
                pass
            buffered.close()
            cache_sha256 = digest_reader.digest.hexdigest()
            compressed_bytes = digest_reader.byte_count
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_dev != path_after.st_dev
            or after.st_ino != path_after.st_ino
            or compressed_bytes != before.st_size
        ):
            raise FanOutQaP1SourceQualificationError(
                "cache archive changed during qualification"
            )
        if not pageids or minimum_file_size is None:
            raise FanOutQaP1SourceQualificationError("cache archive is empty")
        return {
            "cache_sha256": cache_sha256,
            "compressed_size_bytes": compressed_bytes,
            "directory_count": directory_count,
            "file_count": file_count,
            "maximum_file_size": maximum_file_size,
            "minimum_file_size": minimum_file_size,
            "pageids": pageids,
            "total_uncompressed_bytes": total_uncompressed,
        }
    except FanOutQaP1SourceQualificationError:
        raise
    except OSError as exc:
        raise FanOutQaP1SourceQualificationError(
            "cache archive descriptor validation failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _public_order(candidate: Candidate, round_index: int) -> bytes:
    return hashlib.sha256(
        (
            PUBLIC_CAPACITY_DOMAIN
            + "\0"
            + str(round_index)
            + "\0"
            + candidate.family
            + "\0"
            + candidate.question_sha256
        ).encode("ascii")
    ).digest()


def _page_disjoint_capacity(
    candidates: Sequence[Candidate], required: int
) -> tuple[dict[str, int], dict[str, int]]:
    selected_pages: set[int] = set()
    selected_questions: set[str] = set()
    counts = {family: 0 for family in FAMILIES}
    collision_skips = {family: 0 for family in FAMILIES}
    by_family = {
        family: [candidate for candidate in candidates if candidate.family == family]
        for family in FAMILIES
    }
    for round_index in range(required):
        for family in FAMILIES:
            ordered = sorted(
                by_family[family], key=lambda row: _public_order(row, round_index)
            )
            chosen: Candidate | None = None
            for candidate in ordered:
                if candidate.question_sha256 in selected_questions:
                    continue
                if not selected_pages.isdisjoint(candidate.pageids):
                    collision_skips[family] += 1
                    continue
                chosen = candidate
                break
            if chosen is not None:
                selected_questions.add(chosen.question_sha256)
                selected_pages.update(chosen.pageids)
                counts[family] += 1
    return counts, collision_skips


def analyze_sources(
    dev_path: Path,
    cache_path: Path,
    *,
    contract: QualificationContract = FORMAL_CONTRACT,
    deny_question_sha256: Iterable[str] = EXAMPLE_QUESTION_DENY_SHA256,
) -> dict[str, object]:
    """Analyze fixed sources and return only safe aggregate facts."""

    deny = frozenset(deny_question_sha256)
    if any(not isinstance(value, str) or _HEX64.fullmatch(value) is None for value in deny):
        raise FanOutQaP1SourceQualificationError("question denylist drifted")
    raw, _ = _bound_regular_bytes(dev_path, contract.dev_size_bytes)
    dev_sha256 = hashlib.sha256(raw).hexdigest()
    dev_git_blob_sha1 = _git_blob_sha1(raw)
    if (
        dev_sha256 != contract.dev_sha256
        or dev_git_blob_sha1 != contract.dev_git_blob_sha1
    ):
        raise FanOutQaP1SourceQualificationError("DEV source identity drifted")
    try:
        source = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FanOutQaP1SourceQualificationError("DEV JSON is invalid") from exc
    if not isinstance(source, list) or len(source) != contract.dev_count:
        raise FanOutQaP1SourceQualificationError("DEV row count drifted")

    item_ids: set[str] = set()
    question_hashes: set[str] = set()
    parsed: list[tuple[str, str, frozenset[int], int, int, int, str]] = []
    family_total = {family: 0 for family in FAMILIES}
    category_counts = {category: 0 for category in sorted(ALLOWED_CATEGORIES)}
    depth_histogram: dict[str, int] = {}
    dependency_histogram: dict[str, int] = {}
    evidence_count_histogram: dict[str, int] = {}
    denied_example_match_count = 0
    conflicting_revision_item_count = 0

    for value in source:
        row = _parse_item(value)
        item_id, question_hash, pageids, conflict_count, depth, dependencies, family = row
        if item_id in item_ids or question_hash in question_hashes:
            raise FanOutQaP1SourceQualificationError("DEV identity closure drifted")
        item_ids.add(item_id)
        question_hashes.add(question_hash)
        parsed.append(row)
        family_total[family] += 1
        for category in value["categories"]:
            category_counts[category] += 1
        depth_histogram[str(depth)] = depth_histogram.get(str(depth), 0) + 1
        dependency_histogram[str(dependencies)] = (
            dependency_histogram.get(str(dependencies), 0) + 1
        )
        evidence_count_histogram[str(len(pageids))] = (
            evidence_count_histogram.get(str(len(pageids)), 0) + 1
        )
        if question_hash in deny:
            denied_example_match_count += 1
        if conflict_count:
            conflicting_revision_item_count += 1

    cache = _audit_cache_tar(cache_path, contract)
    cache_pageids = cache.pop("pageids")
    if not isinstance(cache_pageids, set):
        raise FanOutQaP1SourceQualificationError("cache inventory drifted")

    ineligible = {
        "cache_missing_required_page": 0,
        "conflicting_page_revision": 0,
        "evidence_page_count_outside_3_through_10": 0,
        "paper_example_question_denylist": 0,
    }
    eligible_family_counts = {family: 0 for family in FAMILIES}
    eligible: list[Candidate] = []
    distinct_dev_evidence_pages: set[int] = set()
    cache_covered_dev_evidence_pages: set[int] = set()
    for _, question_hash, pageids, conflict_count, _, _, family in parsed:
        distinct_dev_evidence_pages.update(pageids)
        cache_covered_dev_evidence_pages.update(pageids & cache_pageids)
        if question_hash in deny:
            ineligible["paper_example_question_denylist"] += 1
            continue
        if conflict_count:
            ineligible["conflicting_page_revision"] += 1
            continue
        if not contract.min_evidence_pages <= len(pageids) <= contract.max_evidence_pages:
            ineligible["evidence_page_count_outside_3_through_10"] += 1
            continue
        if not pageids.issubset(cache_pageids):
            ineligible["cache_missing_required_page"] += 1
            continue
        eligible.append(Candidate(question_hash, family, pageids))
        eligible_family_counts[family] += 1

    disjoint_counts, collision_skips = _page_disjoint_capacity(
        eligible, contract.required_per_family
    )
    qualified = all(
        disjoint_counts[family] == contract.required_per_family
        for family in FAMILIES
    )
    return {
        "allowed_categories": sorted(ALLOWED_CATEGORIES),
        "cache_aggregate": cache,
        "cache_covered_distinct_DEV_evidence_page_count": len(
            cache_covered_dev_evidence_pages
        ),
        "category_counts": category_counts,
        "conflicting_revision_item_count": conflicting_revision_item_count,
        "dependency_edge_count_histogram": dict(
            sorted(dependency_histogram.items(), key=lambda row: int(row[0]))
        ),
        "denied_paper_example_match_count": denied_example_match_count,
        "depth_histogram": dict(
            sorted(depth_histogram.items(), key=lambda row: int(row[0]))
        ),
        "DEV_git_blob_sha1": dev_git_blob_sha1,
        "DEV_row_count": len(source),
        "DEV_sha256": dev_sha256,
        "DEV_size_bytes": len(raw),
        "distinct_DEV_evidence_page_count": len(distinct_dev_evidence_pages),
        "eligible_family_counts": eligible_family_counts,
        "evidence_page_count_histogram": dict(
            sorted(evidence_count_histogram.items(), key=lambda row: int(row[0]))
        ),
        "family_total_counts": family_total,
        "ineligible_reason_counts": ineligible,
        "page_disjoint_capacity_collision_skips": collision_skips,
        "page_disjoint_capacity_counts": disjoint_counts,
        "qualified": qualified,
        "required_page_disjoint_capacity_per_family": contract.required_per_family,
        "schema_contract": {
            "evidence_keys": sorted(EVIDENCE_KEYS),
            "subquestion_keys": sorted(SUB_KEYS),
            "top_level_keys": sorted(TOP_KEYS),
        },
    }


def _validate_public_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    custody = _load_canonical_self_hashed(CUSTODY_PATH, "source custody")
    design = _load_canonical_self_hashed(DESIGN_PATH, "study design")
    download = _load_canonical_self_hashed(
        DOWNLOAD_RECEIPT_PATH, "source download receipt"
    )
    freeze = _load_canonical_self_hashed(FREEZE_PATH, "qualification freeze")
    expected_bindings = {
        "qualifier": (
            QUALIFIER_PATH,
            "assumption_agent/benchmarks/fanoutqa_p1_source_qualification_v1.py",
        ),
        "tests": (TEST_PATH, "tests/test_fanoutqa_p1_source_qualification_v1.py"),
        "source_custody": (
            CUSTODY_PATH,
            "manifests/fanoutqa_p1_source_custody_v1.json",
        ),
        "study_design": (
            DESIGN_PATH,
            "manifests/fanoutqa_p1_typed_fanout_e3_study_design_v1.json",
        ),
    }
    bindings = freeze.get("file_bindings")
    implementation_commit = freeze.get("implementation_commit")
    cache_download = download.get("cache")
    dev_download = download.get("DEV")
    if (
        custody.get("schema") != "fanoutqa_p1_source_custody_v1"
        or custody.get("self_sha256") != EXPECTED_CUSTODY_SELF_SHA256
        or custody.get("official_release", {}).get("commit") != OFFICIAL_RELEASE
        or custody.get("official_release", {}).get("tree") != OFFICIAL_TREE
        or custody.get("custody_boundary", {}).get(
            "strict_source_bytes_before_freeze"
        )
        is not False
        or custody.get("formal_scope", {}).get("test_parse_or_use") is not False
        or design.get("schema")
        != "fanoutqa_p1_typed_fanout_e3_study_design_v1"
        or design.get("self_sha256") != EXPECTED_DESIGN_SELF_SHA256
        or design.get("study_id") != "FANOUTQA_P1_TYPED_FANOUT_E3_V1"
        or design.get("source_binding", {}).get("test_open") is not False
        or download.get("schema") != "fanoutqa_p1_source_download_receipt_v1"
        or download.get("status")
        != "downloaded_exact_pinned_DEV_and_cache_without_semantic_parse"
        or not isinstance(dev_download, dict)
        or dev_download.get("size_bytes") != DEV_SIZE_BYTES
        or dev_download.get("git_blob_sha1") != DEV_GIT_BLOB_SHA1
        or dev_download.get("sha256") != DEV_SHA256
        or not isinstance(cache_download, dict)
        or cache_download.get("size_bytes") != CACHE_SIZE_BYTES
        or not isinstance(cache_download.get("sha256"), str)
        or _HEX64.fullmatch(str(cache_download.get("sha256"))) is None
        or cache_download.get("etag") != "f55692e0e4dc9adb045243d093ced30a-15"
        or cache_download.get("last_modified")
        != "Tue, 13 Feb 2024 20:57:56 GMT"
        or download.get("semantic_source_parse_during_download") is not False
        or download.get("TEST_downloaded_or_opened") is not False
        or freeze.get("schema")
        != "fanoutqa_p1_source_qualification_freeze_v1"
        or freeze.get("status")
        != "frozen_before_formal_persisted_source_download_and_any_dataset_JSON_parse"
        or not isinstance(implementation_commit, str)
        or _HEX40.fullmatch(implementation_commit) is None
        or implementation_commit == "0" * 40
        or freeze.get("formal_DEV_file_present_at_freeze") is not False
        or freeze.get("formal_cache_file_present_at_freeze") is not False
        or freeze.get("formal_qualification_attempt_count_at_freeze") != 0
        or freeze.get("model_action_or_score_count_at_freeze") != 0
        or not isinstance(bindings, dict)
        or set(bindings) != set(expected_bindings)
    ):
        raise FanOutQaP1SourceQualificationError("public contract drifted")
    for role, (path, relative) in expected_bindings.items():
        row = bindings.get(role)
        if (
            not isinstance(row, dict)
            or set(row) != {"relative_path", "sha256"}
            or row.get("relative_path") != relative
            or not isinstance(row.get("sha256"), str)
            or _HEX64.fullmatch(str(row.get("sha256"))) is None
            or _file_sha256(path) != row.get("sha256")
        ):
            raise FanOutQaP1SourceQualificationError(
                "qualification freeze implementation binding drifted"
            )
    return custody, design, download, freeze


def _terminal_failure(stage: str, exc: BaseException) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "status": "terminal_FanOutQA_P1_source_route_no_retry",
        "failure_stage": stage,
        "failure_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "source_item_question_answer_evidence_URL_or_id_output_count": 0,
        "model_action_or_score_count": 0,
        "external_network_calls": 0,
        "online_evaluator_or_API_calls": 0,
        "retry_replay_resample_or_contract_revision": 0,
    }
    try:
        _write_exclusive(FAILURE_PATH, {**body, "self_sha256": _semantic_hash(body)})
    except BaseException:
        pass


def run_source_qualification() -> dict[str, object]:
    """Consume the sole fixed FanOutQA P1 aggregate-only qualification."""

    if any(path.exists() or path.is_symlink() for path in (MARKER_PATH, FAILURE_PATH, RESULT_PATH)):
        raise FanOutQaP1SourceQualificationError(
            "FanOutQA P1 source qualification path is already consumed"
        )
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker_v1",
        "status": "started_before_contract_validation_or_dataset_JSON_parse",
        "official_release": OFFICIAL_RELEASE,
        "official_tree": OFFICIAL_TREE,
        "DEV_git_blob_sha1": DEV_GIT_BLOB_SHA1,
        "DEV_size_bytes": DEV_SIZE_BYTES,
        "cache_size_bytes": CACHE_SIZE_BYTES,
        "test_open_authorized": False,
        "source_item_question_answer_evidence_URL_or_id_output_count": 0,
        "model_action_or_score_count": 0,
        "retry_replay_resample_or_contract_revision": 0,
    }
    _write_exclusive(MARKER_PATH, {**marker_body, "self_sha256": _semantic_hash(marker_body)})
    stage = "frozen_public_contracts"
    try:
        custody, design, download, freeze = _validate_public_contracts()
        stage = "aggregate_DEV_schema_cache_and_structural_capacity"
        aggregate = analyze_sources(DEV_PATH, CACHE_PATH)
        if (
            aggregate.get("cache_aggregate", {}).get("cache_sha256")
            != download["cache"]["sha256"]
        ):
            raise FanOutQaP1SourceQualificationError(
                "cache archive download receipt drifted"
            )
        qualified = aggregate["qualified"] is True
        body: dict[str, object] = {
            "schema": f"{VERSION}_result_v1",
            "status": (
                "qualified_aggregate_source_cache_and_structural_capacity"
                if qualified
                else "terminal_FanOutQA_P1_structural_capacity_failed"
            ),
            "qualified": qualified,
            "official_release": OFFICIAL_RELEASE,
            "official_tree": OFFICIAL_TREE,
            "source_custody_self_sha256": custody["self_sha256"],
            "study_design_self_sha256": design["self_sha256"],
            "source_download_receipt_self_sha256": download["self_sha256"],
            "qualification_freeze_self_sha256": freeze["self_sha256"],
            "aggregate": aggregate,
            "test_opened": False,
            "cache_article_content_extracted_or_parsed": False,
            "source_item_question_answer_evidence_URL_or_id_output_count": 0,
            "model_action_or_score_count": 0,
            "external_network_calls": 0,
            "online_evaluator_or_API_calls": 0,
            "retry_replay_resample_or_contract_revision": 0,
            "claim_boundary": "derived_closed_cache_corpus_not_official_full_Wikipedia_open_book",
        }
        result = {**body, "self_sha256": _semantic_hash(body)}
        _write_exclusive(RESULT_PATH, result)
        if not qualified:
            raise FanOutQaP1SourceQualificationError(
                "FanOutQA P1 aggregate capacity failed"
            )
        return result
    except BaseException as exc:
        if not RESULT_PATH.exists():
            _terminal_failure(stage, exc)
        raise


def main() -> int:
    run_source_qualification()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_QUOTAS",
    "Candidate",
    "FanOutQaP1SourceQualificationError",
    "FAMILIES",
    "QualificationContract",
    "analyze_sources",
    "run_source_qualification",
]
