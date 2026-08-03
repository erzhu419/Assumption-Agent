"""Git-object replay for the frozen Phase-3A Gate-17 parent audit.

This module is deliberately an evidence generator and replayer, not an
auditor-signing implementation.  It reads Git objects through NUL-delimited
plumbing commands, constructs the already-frozen formal rows, and exposes the
public fields that a purpose-4 actor may attest after an independent replay.
It never checks out a commit, mutates a worktree, generates a key, or signs an
object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    LEGACY_PARENT_SOURCE_IDS,
    build_formal_object,
    candidate_content_root,
    candidate_record_tree_root,
    encode_formal_object,
    git_sha1_commit_id,
    id_digest_v1,
    validate_parent_absence_audit_bundle_v1,
)


PARENT_DSL_VERSION: Final = "hegel-old-dsl-v1.0.0"
PARENT_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.2"
TOUCHED_PATH_RULE_ID: Final = "git-any-parent-result-or-deletion-blob-v1"
PATH_ALIAS_RULE_ID: Final = "repo-path-sha256-raw-bytes-v1"
PATH_NAME_PREDICATE_PROFILE_ID: Final = (
    "legacy-parent-formal-artifact-path-name-absence-v1"
)
CONTENT_PREDICATE_PROFILE_ID: Final = (
    "legacy-parent-formal-artifact-unique-blob-content-absence-v1"
)
PUBLIC_RECEIPT_SCHEMA_ID: Final = "hegel-parent-absence-audit-receipt/1"
DEFAULT_GIT_EXECUTABLE: Final = Path("/usr/bin/git")

FAIL_GIT_AUDIT: Final = "FAIL_PARENT_AUDIT_GIT_PLUMBING"
FAIL_GIT_OBJECT_FORMAT: Final = "FAIL_PARENT_AUDIT_GIT_OBJECT_FORMAT"
FAIL_GIT_SHALLOW: Final = "FAIL_PARENT_AUDIT_SHALLOW_REPOSITORY"
FAIL_GIT_REACHABILITY: Final = "FAIL_PARENT_AUDIT_REACHABILITY"
FAIL_GIT_COMMIT_OBJECT: Final = "FAIL_PARENT_AUDIT_COMMIT_OBJECT"
FAIL_GIT_DIFF_RECORD: Final = "FAIL_PARENT_AUDIT_DIFF_RECORD"
FAIL_GIT_NON_BLOB: Final = "FAIL_PARENT_AUDIT_NON_BLOB_ENTRY"
FAIL_GIT_PATH_ENCODING: Final = "FAIL_PARENT_AUDIT_PATH_NOT_UTF8"
FAIL_GIT_BLOB_IDENTITY: Final = "FAIL_PARENT_AUDIT_GIT_BLOB_IDENTITY"
FAIL_CONTENT_PREDICATE_MATCH: Final = "FAIL_PARENT_AUDIT_CONTENT_PREDICATE_MATCH"
FAIL_LEGACY_SOURCE_CONTENT: Final = "FAIL_PARENT_AUDIT_LEGACY_SOURCE_CONTENT"
FAIL_REPLAY_MISMATCH: Final = "FAIL_PARENT_AUDIT_REPLAY_MISMATCH"


class ParentAbsenceAuditError(RuntimeError):
    """Stable fail-closed error raised by the Gate-17 Git audit."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> None:
    raise ParentAbsenceAuditError(code, detail)


@dataclass(frozen=True)
class _BlobRef:
    path: bytes
    mode: int
    digest: bytes


@dataclass(frozen=True)
class ParentAbsenceAuditEvidence:
    """Complete in-memory preimage for Gate 17, without a signature."""

    top_level_path_rows: tuple[Mapping[str, object], ...]
    history_rows: tuple[Mapping[str, object], ...]
    touched_path_rows_by_history_row: tuple[
        tuple[Mapping[str, object], ...], ...
    ]
    legacy_source_rows: tuple[Mapping[str, object], ...]
    audit_bundle_fields: Mapping[str, object]
    audit_bundle_root: bytes
    attestation_static_fields: Mapping[str, object]
    path_name_receipt: Mapping[str, object]


# These are diagnostic path-name predicates, not a replacement for the formal
# row replay.  Input paths are ASCII-folded and '-' / '.' are normalized to
# '_' before byte-substring matching.  The deliberately broad aliases make a
# path spelling variation visible rather than silently escaping the audit.
_PATH_NAME_PREDICATES: Final = MappingProxyType(
    {
        "typed_or_parent_binding_manifest": (
            b"typed_binding_manifest",
            b"parent_binding_manifest",
        ),
        "split_seed_commitment_or_allocation": (
            b"split_seed_commitment",
            b"split_seed_allocation",
            b"split_assignment_manifest",
            b"split_allocation_manifest",
        ),
        "hidden_access_ledger": (b"hidden_access_ledger",),
    }
)

# Exact byte signatures of a realized formal artifact.  These are searched in
# every unique blob referenced by the 7,945-row formal union, including blobs
# stored under generic filenames and binary blobs.  Natural-language phrases
# are intentionally excluded: a discussion of a future object is not itself a
# machine-readable object.  A hit fails closed for external review.
_CONTENT_ABSENCE_PREDICATES: Final = MappingProxyType(
    {
        "typed_or_parent_binding_manifest": (
            b"hegel-typed-binding-manifest/",
            b"typed_binding_manifest_root",
            b"parent_binding_manifest_root",
        ),
        "split_seed_commitment_or_allocation": (
            b"hegel-split-seed-commitment-manifest/",
            b"split_seed_commitment_digest",
            b"split_seed_commitment_manifest_root",
            b"hegel-split-assignment-row/",
            b"split_assignment_tree_root",
            b"split_allocation_manifest_root",
        ),
        "hidden_access_ledger": (
            b"hegel-hidden-access-ledger-record/",
            b"hidden_access_ledger_genesis_root",
            b"hidden_access_ledger_head_root",
        ),
    }
)

_STRUCTURED_PATH_SUFFIXES: Final = (
    b".bib",
    b".cbor",
    b".cfg",
    b".csv",
    b".html",
    b".ini",
    b".js",
    b".json",
    b".jsonl",
    b".md",
    b".py",
    b".rs",
    b".sh",
    b".toml",
    b".ts",
    b".tsv",
    b".txt",
    b".xml",
    b".yaml",
    b".yml",
)


def _git_environment() -> dict[str, str]:
    return {
        "LC_ALL": "C",
        "LANG": "C",
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
    }


def _resolve_git_executable(git_executable: str | Path) -> Path:
    try:
        executable = Path(git_executable).resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        _fail(
            FAIL_GIT_AUDIT,
            f"Git executable cannot be resolved: {type(error).__name__}",
        )
    if not executable.is_absolute() or not executable.is_file() or not os.access(
        executable, os.X_OK
    ):
        _fail(FAIL_GIT_AUDIT, "Git executable is not an executable regular file")
    return executable


def _git_command(
    git_executable: str | Path,
    repository: Path,
    arguments: Sequence[str],
) -> tuple[Path, list[str]]:
    executable = _resolve_git_executable(git_executable)
    try:
        safe_repository = repository.resolve(strict=True)
    except OSError as error:
        _fail(
            FAIL_GIT_AUDIT,
            f"Git repository cannot be resolved: {type(error).__name__}",
        )
    return safe_repository, [
        str(executable),
        "-c",
        "core.quotePath=false",
        "-c",
        f"safe.directory={safe_repository}",
        *arguments,
    ]


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> bytes:
    safe_repository, command = _git_command(
        git_executable, repository, arguments
    )
    try:
        completed = subprocess.run(
            command,
            cwd=safe_repository,
            env=_git_environment(),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        _fail(
            FAIL_GIT_AUDIT,
            f"Git command could not start: {type(error).__name__}",
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "backslashreplace").strip()
        _fail(FAIL_GIT_AUDIT, f"{' '.join(command)} failed: {detail}")
    return completed.stdout


def _resolve_repository(
    repository: str | Path,
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> Path:
    candidate = Path(repository).resolve()
    raw_root = _run_git(
        candidate,
        ("rev-parse", "--show-toplevel"),
        git_executable=git_executable,
    ).rstrip(b"\n")
    try:
        root = Path(os.fsdecode(raw_root)).resolve()
    except (UnicodeError, ValueError) as error:
        _fail(FAIL_GIT_AUDIT, f"repository root cannot be represented: {error}")
    object_format = _run_git(
        root,
        ("rev-parse", "--show-object-format"),
        git_executable=git_executable,
    ).strip()
    if object_format != b"sha1":
        _fail(FAIL_GIT_OBJECT_FORMAT, f"expected sha1, found {object_format!r}")
    shallow = _run_git(
        root,
        ("rev-parse", "--is-shallow-repository"),
        git_executable=git_executable,
    ).strip()
    if shallow != b"false":
        _fail(FAIL_GIT_SHALLOW, "reachable-history replay requires a complete repository")
    return root


def _batch_cat_file(
    repository: Path,
    object_digests: Sequence[bytes],
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> dict[bytes, tuple[bytes, bytes]]:
    """Return object type and payload for exact raw SHA-1 object IDs."""

    if not object_digests:
        return {}
    payload = b"".join(digest.hex().encode("ascii") + b"\n" for digest in object_digests)
    output = _run_git(
        repository,
        ("cat-file", "--batch"),
        input_bytes=payload,
        git_executable=git_executable,
    )
    offset = 0
    result: dict[bytes, tuple[bytes, bytes]] = {}
    for requested in object_digests:
        newline = output.find(b"\n", offset)
        if newline < 0:
            _fail(FAIL_GIT_AUDIT, "truncated git cat-file --batch header")
        header = output[offset:newline].split(b" ")
        offset = newline + 1
        if len(header) != 3 or header[1] == b"missing":
            _fail(FAIL_GIT_AUDIT, f"Git object {requested.hex()} is missing")
        try:
            returned = bytes.fromhex(header[0].decode("ascii"))
            size = int(header[2])
        except (UnicodeError, ValueError) as error:
            _fail(FAIL_GIT_AUDIT, f"malformed cat-file header: {error}")
        if returned != requested or len(returned) != 20 or size < 0:
            _fail(FAIL_GIT_AUDIT, "cat-file returned a different or malformed object ID")
        end = offset + size
        if end >= len(output) or output[end : end + 1] != b"\n":
            _fail(FAIL_GIT_AUDIT, "truncated git cat-file --batch payload")
        result[requested] = (header[1], output[offset:end])
        offset = end + 1
    if offset != len(output):
        _fail(FAIL_GIT_AUDIT, "cat-file emitted trailing bytes")
    return result


def _batch_blob_sizes(
    repository: Path,
    blob_digests: Sequence[bytes],
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> dict[bytes, int]:
    if not blob_digests:
        return {}
    ordered = tuple(sorted(set(blob_digests)))
    payload = b"".join(digest.hex().encode("ascii") + b"\n" for digest in ordered)
    output = _run_git(
        repository,
        ("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"),
        input_bytes=payload,
        git_executable=git_executable,
    )
    lines = output.splitlines()
    if len(lines) != len(ordered):
        _fail(FAIL_GIT_AUDIT, "cat-file --batch-check row count differs")
    sizes: dict[bytes, int] = {}
    for expected, line in zip(ordered, lines, strict=True):
        parts = line.split(b" ")
        if len(parts) != 3:
            _fail(FAIL_GIT_AUDIT, "malformed cat-file --batch-check row")
        try:
            returned = bytes.fromhex(parts[0].decode("ascii"))
            size = int(parts[2])
        except (UnicodeError, ValueError) as error:
            _fail(FAIL_GIT_AUDIT, f"malformed blob metadata: {error}")
        if returned != expected or parts[1] != b"blob" or size < 0:
            _fail(
                FAIL_GIT_NON_BLOB,
                f"audited entry {expected.hex()} is not an available Git blob",
            )
        sizes[expected] = size
    return sizes


def _blob_inventory_digest(sizes: Mapping[bytes, int]) -> str:
    hasher = hashlib.sha256()
    for digest in sorted(sizes):
        hasher.update(digest)
        hasher.update(sizes[digest].to_bytes(8, "big"))
    return hasher.hexdigest()


def _digest_set_sha256(digests: Sequence[bytes] | set[bytes]) -> str:
    return hashlib.sha256(b"".join(sorted(set(digests)))).hexdigest()


def _all_content_signatures() -> tuple[bytes, ...]:
    signatures: list[bytes] = []
    for group in _CONTENT_ABSENCE_PREDICATES.values():
        signatures.extend(group)
    signatures.extend(source_id.encode("ascii") for source_id in LEGACY_PARENT_SOURCE_IDS)
    if len(signatures) != len(set(signatures)):
        _fail(FAIL_REPLAY_MISMATCH, "content predicate signatures are not unique")
    return tuple(signatures)


def _scan_streamed_blob(
    stream: object,
    *,
    digest: bytes,
    size: int,
    signatures: Sequence[bytes],
) -> Mapping[bytes, int]:
    # ``stream`` is a BufferedReader at runtime.  Keeping the annotation broad
    # avoids importing a private IO implementation type.
    read = getattr(stream, "read")
    remaining = size
    object_hasher = hashlib.sha1(
        b"blob " + str(size).encode("ascii") + b"\x00"
    )
    tails = {signature: b"" for signature in signatures}
    counts = {signature: 0 for signature in signatures}
    while remaining:
        chunk = read(min(1 << 20, remaining))
        if not chunk:
            _fail(FAIL_GIT_AUDIT, f"truncated blob payload for {digest.hex()}")
        remaining -= len(chunk)
        object_hasher.update(chunk)
        for signature in signatures:
            tail = tails[signature]
            window = tail + chunk
            start = 0
            while True:
                match = window.find(signature, start)
                if match < 0:
                    break
                # Matches contained entirely in the previous tail were already
                # counted.  Everything ending in this chunk is new.
                if match + len(signature) > len(tail):
                    counts[signature] += 1
                start = match + 1
            keep = len(signature) - 1
            tails[signature] = window[-keep:] if keep else b""
    if object_hasher.digest() != digest:
        _fail(
            FAIL_GIT_BLOB_IDENTITY,
            f"recomputed Git blob object ID differs for {digest.hex()}",
        )
    return MappingProxyType(counts)


def _scan_unique_blob_contents(
    repository: Path,
    sizes: Mapping[bytes, int],
    top_level_rows: Sequence[Mapping[str, object]],
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> Mapping[str, object]:
    """Stream every bound unique blob, proving object identity and predicates."""

    signatures = _all_content_signatures()
    occurrence_totals = {signature: 0 for signature in signatures}
    matching_blobs = {signature: set() for signature in signatures}
    safe_repository, command = _git_command(
        git_executable,
        repository,
        ("cat-file", "--batch"),
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=safe_repository,
            env=_git_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        _fail(
            FAIL_GIT_AUDIT,
            f"streaming Git command could not start: {type(error).__name__}",
        )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        for expected in sorted(sizes):
            process.stdin.write(expected.hex().encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n").split(b" ")
            if len(header) != 3:
                _fail(FAIL_GIT_AUDIT, f"malformed streaming cat-file header {header!r}")
            try:
                returned = bytes.fromhex(header[0].decode("ascii"))
                returned_size = int(header[2])
            except (UnicodeError, ValueError) as error:
                _fail(FAIL_GIT_AUDIT, f"malformed streaming blob metadata: {error}")
            if (
                returned != expected
                or header[1] != b"blob"
                or returned_size != sizes[expected]
            ):
                _fail(
                    FAIL_GIT_BLOB_IDENTITY,
                    f"Git type/size differs for bound blob {expected.hex()}",
                )
            counts = _scan_streamed_blob(
                process.stdout,
                digest=expected,
                size=returned_size,
                signatures=signatures,
            )
            if process.stdout.read(1) != b"\n":
                _fail(FAIL_GIT_AUDIT, "streaming cat-file blob lacks its delimiter")
            for signature, count in counts.items():
                occurrence_totals[signature] += count
                if count:
                    matching_blobs[signature].add(expected)
        process.stdin.close()
        if process.stdout.read(1) != b"":
            _fail(FAIL_GIT_AUDIT, "streaming cat-file emitted trailing bytes")
        return_code = process.wait()
        stderr = process.stderr.read()
        if return_code != 0 or stderr:
            _fail(
                FAIL_GIT_AUDIT,
                "streaming cat-file failed: "
                + stderr.decode("utf-8", "backslashreplace").strip(),
            )
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise

    row_digests = [row["git_blob_digest"] for row in top_level_rows]
    if any(type(digest) is not bytes for digest in row_digests):
        _fail(FAIL_REPLAY_MISMATCH, "audited path row has a non-byte blob digest")
    structured_digests = {
        row["git_blob_digest"]
        for row in top_level_rows
        if row["raw_repository_path_utf8_bytes"].lower().endswith(  # type: ignore[union-attr]
            _STRUCTURED_PATH_SUFFIXES
        )
    }

    absence_results: list[dict[str, object]] = []
    for predicate_id, predicate_signatures in _CONTENT_ABSENCE_PREDICATES.items():
        group_blobs: set[bytes] = set()
        occurrence_count = 0
        for signature in predicate_signatures:
            occurrence_count += occurrence_totals[signature]
            group_blobs.update(matching_blobs[signature])
        matching_row_count = sum(digest in group_blobs for digest in row_digests)
        absence_results.append(
            {
                "predicate_id": predicate_id,
                "exact_signatures_ascii": [
                    signature.decode("ascii") for signature in predicate_signatures
                ],
                "match_occurrence_count": occurrence_count,
                "matching_unique_blob_count": len(group_blobs),
                "matching_path_blob_row_count": matching_row_count,
                "matching_blob_digest_set_sha256": _digest_set_sha256(group_blobs),
                "absent": occurrence_count == 0,
            }
        )
    if not all(result["absent"] for result in absence_results):
        matches = [
            result["predicate_id"] for result in absence_results if not result["absent"]
        ]
        _fail(
            FAIL_CONTENT_PREDICATE_MATCH,
            f"formal-artifact content signatures were found: {matches}",
        )

    legacy_results: list[dict[str, object]] = []
    for source_id in LEGACY_PARENT_SOURCE_IDS:
        signature = source_id.encode("ascii")
        source_blobs = matching_blobs[signature]
        occurrence_count = occurrence_totals[signature]
        legacy_results.append(
            {
                "legacy_parent_payload_source_id": source_id,
                "match_occurrence_count": occurrence_count,
                "matching_unique_blob_count": len(source_blobs),
                "matching_path_blob_row_count": sum(
                    digest in source_blobs for digest in row_digests
                ),
                "matching_blob_digest_set_sha256": _digest_set_sha256(source_blobs),
                "present": occurrence_count > 0,
            }
        )
    if not all(result["present"] for result in legacy_results):
        missing = [
            result["legacy_parent_payload_source_id"]
            for result in legacy_results
            if not result["present"]
        ]
        _fail(
            FAIL_LEGACY_SOURCE_CONTENT,
            f"frozen legacy source IDs are absent from bound blobs: {missing}",
        )

    return MappingProxyType(
        {
            "content_predicate_profile_id": CONTENT_PREDICATE_PROFILE_ID,
            "inspected_path_blob_row_count": len(top_level_rows),
            "inspected_unique_blob_count": len(sizes),
            "inspected_total_byte_length": sum(sizes.values()),
            "inspected_blob_inventory_sha256": _blob_inventory_digest(sizes),
            "git_blob_object_id_and_size_verified": True,
            "structured_candidate_unique_blob_count": len(structured_digests),
            "unscannable_relevant_structured_blob_count": 0,
            "content_absence_predicates": absence_results,
            "legacy_source_presence": legacy_results,
            "all_content_absence_predicates_absent": True,
            "all_legacy_sources_present": True,
        }
    )


def _parse_commit_parents(commit_digest: bytes, payload: bytes) -> tuple[bytes, ...]:
    parents: list[bytes] = []
    saw_tree = False
    for line in payload.split(b"\n"):
        if line == b"":
            break
        if line.startswith(b"tree "):
            if saw_tree:
                _fail(FAIL_GIT_COMMIT_OBJECT, "commit contains duplicate tree headers")
            saw_tree = True
        elif line.startswith(b"parent "):
            raw = line[7:]
            try:
                parent = bytes.fromhex(raw.decode("ascii"))
            except (UnicodeError, ValueError) as error:
                _fail(FAIL_GIT_COMMIT_OBJECT, f"malformed parent header: {error}")
            if len(parent) != 20:
                _fail(FAIL_GIT_COMMIT_OBJECT, "parent header is not SHA-1")
            parents.append(parent)
    if not saw_tree:
        _fail(FAIL_GIT_COMMIT_OBJECT, f"commit {commit_digest.hex()} has no tree")
    if len(parents) != len(set(parents)):
        _fail(FAIL_GIT_COMMIT_OBJECT, "commit repeats a parent")
    return tuple(parents)


def _reachable_commit_graph(
    repository: Path,
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> tuple[tuple[bytes, ...], Mapping[bytes, tuple[bytes, ...]], Mapping[bytes, int]]:
    parent_hex = AUDITED_PARENT_COMMIT_SHA1.hex()
    lines = _run_git(
        repository,
        ("rev-list", parent_hex),
        git_executable=git_executable,
    ).splitlines()
    try:
        commits = tuple(bytes.fromhex(line.decode("ascii")) for line in lines)
    except (UnicodeError, ValueError) as error:
        _fail(FAIL_GIT_REACHABILITY, f"rev-list emitted a malformed commit ID: {error}")
    if not commits or commits[0] != AUDITED_PARENT_COMMIT_SHA1:
        _fail(FAIL_GIT_REACHABILITY, "frozen audited parent is not the rev-list head")
    if len(commits) != len(set(commits)) or any(len(item) != 20 for item in commits):
        _fail(FAIL_GIT_REACHABILITY, "rev-list contains duplicate or non-SHA1 IDs")

    objects = _batch_cat_file(
        repository,
        commits,
        git_executable=git_executable,
    )
    parent_map: dict[bytes, tuple[bytes, ...]] = {}
    for commit in commits:
        object_type, payload = objects[commit]
        if object_type != b"commit":
            _fail(FAIL_GIT_COMMIT_OBJECT, f"{commit.hex()} is not a commit object")
        parent_map[commit] = _parse_commit_parents(commit, payload)
    commit_set = set(commits)
    for commit, parents in parent_map.items():
        missing = [parent.hex() for parent in parents if parent not in commit_set]
        if missing:
            _fail(
                FAIL_GIT_REACHABILITY,
                f"reachable parents of {commit.hex()} are absent: {missing}",
            )

    child_map: dict[bytes, list[bytes]] = {commit: [] for commit in commits}
    remaining_parent_count: dict[bytes, int] = {}
    for commit, parents in parent_map.items():
        remaining_parent_count[commit] = len(parents)
        for parent in parents:
            child_map[parent].append(commit)
    ready = [commit for commit, count in remaining_parent_count.items() if count == 0]
    heapq.heapify(ready)
    generation: dict[bytes, int] = {}
    while ready:
        commit = heapq.heappop(ready)
        parents = parent_map[commit]
        generation[commit] = 0 if not parents else 1 + max(generation[p] for p in parents)
        for child in child_map[commit]:
            remaining_parent_count[child] -= 1
            if remaining_parent_count[child] == 0:
                heapq.heappush(ready, child)
    if len(generation) != len(commits):
        _fail(FAIL_GIT_REACHABILITY, "reachable history is cyclic or incomplete")
    ordered = tuple(sorted(commits, key=lambda item: (generation[item], item)))
    return ordered, MappingProxyType(parent_map), MappingProxyType(generation)


def _validate_raw_path(raw_path: bytes) -> None:
    if not raw_path or b"\x00" in raw_path or raw_path.startswith(b"/"):
        _fail(FAIL_GIT_PATH_ENCODING, f"invalid repository path bytes {raw_path!r}")
    try:
        raw_path.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        _fail(FAIL_GIT_PATH_ENCODING, f"repository path is not UTF-8: {error}")


def _root_blob_refs(
    repository: Path,
    commit: bytes,
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> tuple[_BlobRef, ...]:
    output = _run_git(
        repository,
        ("ls-tree", "-r", "-z", "--full-tree", commit.hex()),
        git_executable=git_executable,
    )
    refs: list[_BlobRef] = []
    for record in output.split(b"\x00"):
        if not record:
            continue
        try:
            header, path = record.split(b"\t", 1)
            mode_raw, object_type, digest_raw = header.split(b" ", 2)
            mode = int(mode_raw, 8)
            digest = bytes.fromhex(digest_raw.decode("ascii"))
        except (ValueError, UnicodeError) as error:
            _fail(FAIL_GIT_AUDIT, f"malformed ls-tree record: {error}")
        _validate_raw_path(path)
        if object_type != b"blob" or len(digest) != 20:
            _fail(
                FAIL_GIT_NON_BLOB,
                f"root commit entry {path!r} is {object_type!r}, not a SHA-1 blob",
            )
        refs.append(_BlobRef(path=path, mode=mode, digest=digest))
    return tuple(sorted(set(refs), key=lambda ref: (ref.path, ref.digest, ref.mode)))


def _parse_parent_diff(
    repository: Path,
    parent: bytes,
    commit: bytes,
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> tuple[tuple[_BlobRef, ...], tuple[_BlobRef, ...]]:
    output = _run_git(
        repository,
        (
            "diff-tree",
            "-r",
            "--raw",
            "--no-renames",
            "--no-commit-id",
            "--no-ext-diff",
            "--no-textconv",
            "--abbrev=40",
            "-z",
            parent.hex(),
            commit.hex(),
        ),
        git_executable=git_executable,
    )
    parts = output.split(b"\x00")
    if parts and parts[-1] == b"":
        parts.pop()
    if len(parts) % 2 != 0:
        _fail(FAIL_GIT_DIFF_RECORD, "raw diff does not contain metadata/path pairs")
    resulting: list[_BlobRef] = []
    deleted: list[_BlobRef] = []
    zero = b"0" * 40
    for index in range(0, len(parts), 2):
        metadata, path = parts[index], parts[index + 1]
        fields = metadata.split(b" ")
        if len(fields) != 5 or not fields[0].startswith(b":"):
            _fail(FAIL_GIT_DIFF_RECORD, f"malformed raw diff metadata {metadata!r}")
        old_mode_raw = fields[0][1:]
        new_mode_raw, old_digest_raw, new_digest_raw, status = fields[1:]
        if not status or status[:1] not in {b"A", b"D", b"M", b"T"}:
            _fail(FAIL_GIT_DIFF_RECORD, f"unexpected --no-renames status {status!r}")
        _validate_raw_path(path)
        try:
            old_mode = int(old_mode_raw, 8)
            new_mode = int(new_mode_raw, 8)
        except ValueError as error:
            _fail(FAIL_GIT_DIFF_RECORD, f"invalid file mode: {error}")
        if new_digest_raw != zero:
            try:
                digest = bytes.fromhex(new_digest_raw.decode("ascii"))
            except (ValueError, UnicodeError) as error:
                _fail(FAIL_GIT_DIFF_RECORD, f"invalid resulting object ID: {error}")
            resulting.append(_BlobRef(path=path, mode=new_mode, digest=digest))
        elif old_digest_raw != zero:
            try:
                digest = bytes.fromhex(old_digest_raw.decode("ascii"))
            except (ValueError, UnicodeError) as error:
                _fail(FAIL_GIT_DIFF_RECORD, f"invalid deleted object ID: {error}")
            deleted.append(_BlobRef(path=path, mode=old_mode, digest=digest))
        else:
            _fail(FAIL_GIT_DIFF_RECORD, "diff record has neither old nor new object")
    return tuple(resulting), tuple(deleted)


def _commit_touched_refs(
    repository: Path,
    commit: bytes,
    parents: tuple[bytes, ...],
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> tuple[_BlobRef, ...]:
    if not parents:
        return _root_blob_refs(
            repository,
            commit,
            git_executable=git_executable,
        )
    resulting_by_path: dict[bytes, _BlobRef] = {}
    deleted_by_record: dict[tuple[bytes, bytes, int], _BlobRef] = {}
    for parent in parents:  # commit-object parent order is intentionally preserved
        resulting, deleted = _parse_parent_diff(
            repository,
            parent,
            commit,
            git_executable=git_executable,
        )
        for ref in resulting:
            previous = resulting_by_path.get(ref.path)
            if previous is not None and previous != ref:
                _fail(
                    FAIL_GIT_DIFF_RECORD,
                    f"parents disagree on resulting tree entry for {ref.path!r}",
                )
            resulting_by_path[ref.path] = ref
        for ref in deleted:
            deleted_by_record[(ref.path, ref.digest, ref.mode)] = ref

    # A resulting commit blob takes precedence.  Parent blobs are retained only
    # when the resulting path is absent (a deletion against one or more parents).
    refs = list(resulting_by_path.values())
    refs.extend(
        ref
        for ref in deleted_by_record.values()
        if ref.path not in resulting_by_path
    )
    return tuple(sorted(refs, key=lambda ref: (ref.path, ref.digest, ref.mode)))


def _path_alias_digest(raw_path: bytes) -> bytes:
    alias = "repo-path-sha256:" + hashlib.sha256(raw_path).hexdigest()
    return id_digest_v1(alias)


def _row_for_blob(ref: _BlobRef, sizes: Mapping[bytes, int]) -> dict[str, object]:
    return {
        "repository_path_alias_id_digest": _path_alias_digest(ref.path),
        "raw_repository_path_utf8_bytes": ref.path,
        "git_object_algorithm_id": 1,
        "git_blob_digest": ref.digest,
        "file_mode": ref.mode,
        "byte_length": sizes[ref.digest],
    }


def _formal_path_order(row: Mapping[str, object]) -> tuple[object, object, object]:
    return (
        row["raw_repository_path_utf8_bytes"],
        row["repository_path_alias_id_digest"],
        row["git_blob_digest"],
    )


def _legacy_source_rows() -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for role_id, source_id in enumerate(LEGACY_PARENT_SOURCE_IDS, start=1):
        rows.append(
            {
                "target_role_id": role_id,
                "legacy_parent_payload_source_id_digest": id_digest_v1(source_id),
                "diagnostic_namespace_id": role_id,
                "diagnostic_digest": bytes.fromhex(source_id.rsplit("_", 1)[-1]),
                "source_repository_commit_id": git_sha1_commit_id(
                    AUDITED_PARENT_COMMIT_SHA1
                ),
            }
        )
    return tuple(rows)


def _path_search_key(path: bytes) -> bytes:
    folded = bytes(byte + 32 if 65 <= byte <= 90 else byte for byte in path)
    return folded.replace(b"-", b"_").replace(b".", b"_")


def _path_name_receipt(
    top_level_rows: Sequence[Mapping[str, object]],
    *,
    audit_bundle_root: bytes,
    audited_path_tree_root: bytes,
    content_blob_audit: Mapping[str, object],
) -> Mapping[str, object]:
    predicates: list[dict[str, object]] = []
    for predicate_id, terms in _PATH_NAME_PREDICATES.items():
        matched_rows = [
            row
            for row in top_level_rows
            if any(term in _path_search_key(row["raw_repository_path_utf8_bytes"]) for term in terms)  # type: ignore[arg-type]
        ]
        unique_paths = {
            row["raw_repository_path_utf8_bytes"] for row in matched_rows
        }
        predicates.append(
            {
                "predicate_id": predicate_id,
                "normalized_substrings_ascii": [term.decode("ascii") for term in terms],
                "matched_unique_path_count": len(unique_paths),
                "matched_path_blob_row_count": len(matched_rows),
                "matched_path_blob_tree_root": candidate_record_tree_root(
                    "AuditedPathBlobRecordV1", matched_rows
                ).hex(),
                "absent": len(matched_rows) == 0,
            }
        )
    body: dict[str, object] = {
        "schema_id": PUBLIC_RECEIPT_SCHEMA_ID,
        "audited_parent_commit_sha1": AUDITED_PARENT_COMMIT_SHA1.hex(),
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "touched_path_rule_id": TOUCHED_PATH_RULE_ID,
        "path_alias_rule_id": PATH_ALIAS_RULE_ID,
        "path_name_predicate_profile_id": PATH_NAME_PREDICATE_PROFILE_ID,
        "audited_path_tree_root": audited_path_tree_root.hex(),
        "audit_bundle_root": audit_bundle_root.hex(),
        "predicates": predicates,
        "all_predicates_absent": all(item["absent"] for item in predicates),
        "content_blob_audit": dict(content_blob_audit),
        "authority_claim": False,
        "purpose_4_signature_present": False,
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    body["diagnostic_receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return MappingProxyType(body)


def generate_parent_absence_audit_v1(
    repository: str | Path,
    *,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> ParentAbsenceAuditEvidence:
    """Generate the complete unsigned Gate-17 evidence from Git objects."""

    executable = _resolve_git_executable(git_executable)
    repo = _resolve_repository(repository, git_executable=executable)
    ordered_commits, parent_map, generations = _reachable_commit_graph(
        repo,
        git_executable=executable,
    )

    touched_refs_by_commit: dict[bytes, tuple[_BlobRef, ...]] = {}
    all_blob_digests: list[bytes] = []
    for commit in ordered_commits:
        refs = _commit_touched_refs(
            repo,
            commit,
            parent_map[commit],
            git_executable=executable,
        )
        touched_refs_by_commit[commit] = refs
        all_blob_digests.extend(ref.digest for ref in refs)
    sizes = _batch_blob_sizes(
        repo,
        all_blob_digests,
        git_executable=executable,
    )

    history_rows: list[Mapping[str, object]] = []
    touched_rows_by_history: list[tuple[Mapping[str, object], ...]] = []
    union_by_cbor: dict[bytes, Mapping[str, object]] = {}
    for commit in ordered_commits:
        rows = tuple(
            sorted(
                (_row_for_blob(ref, sizes) for ref in touched_refs_by_commit[commit]),
                key=_formal_path_order,
            )
        )
        touched_root = candidate_record_tree_root("AuditedPathBlobRecordV1", rows)
        history_rows.append(
            {
                "commit_generation": generations[commit],
                "repository_commit_id": git_sha1_commit_id(commit),
                "ordered_parent_commit_ids": tuple(
                    git_sha1_commit_id(parent) for parent in parent_map[commit]
                ),
                "touched_path_set_root": touched_root,
            }
        )
        touched_rows_by_history.append(rows)
        for row in rows:
            union_by_cbor.setdefault(
                encode_formal_object("AuditedPathBlobRecordV1", row), row
            )

    top_level_rows = tuple(
        sorted(union_by_cbor.values(), key=_formal_path_order)
    )
    content_blob_audit = _scan_unique_blob_contents(
        repo,
        sizes,
        top_level_rows,
        git_executable=executable,
    )
    legacy_rows = _legacy_source_rows()
    path_root = candidate_record_tree_root("AuditedPathBlobRecordV1", top_level_rows)
    history_root = candidate_record_tree_root("AuditedHistoryRowV1", history_rows)
    legacy_root = candidate_record_tree_root("LegacyParentSourceRowV1", legacy_rows)
    audit_bundle_fields: Mapping[str, object] = MappingProxyType(
        {
            "audited_parent_repository_commit_id": git_sha1_commit_id(
                AUDITED_PARENT_COMMIT_SHA1
            ),
            "audited_path_tree_root": path_root,
            "audited_history_tree_root": history_root,
            "legacy_source_tree_root": legacy_root,
            "audited_path_count": len(top_level_rows),
            "audited_history_row_count": len(history_rows),
            "legacy_source_count": len(legacy_rows),
        }
    )
    audit_bundle_root = validate_parent_absence_audit_bundle_v1(
        top_level_rows,
        history_rows,
        touched_rows_by_history,
        legacy_rows,
        audit_bundle_fields,
    )
    attestation_static_fields: Mapping[str, object] = MappingProxyType(
        {
            "parent_dsl_version_digest": id_digest_v1(PARENT_DSL_VERSION),
            "parent_freeze_version_digest": id_digest_v1(PARENT_FREEZE_VERSION),
            "parent_repository_commit_id": git_sha1_commit_id(
                AUDITED_PARENT_COMMIT_SHA1
            ),
            "audit_bundle_root": audit_bundle_root,
            "absence_reason_bitmask": 0b1111,
        }
    )
    return ParentAbsenceAuditEvidence(
        top_level_path_rows=top_level_rows,
        history_rows=tuple(history_rows),
        touched_path_rows_by_history_row=tuple(touched_rows_by_history),
        legacy_source_rows=legacy_rows,
        audit_bundle_fields=audit_bundle_fields,
        audit_bundle_root=audit_bundle_root,
        attestation_static_fields=attestation_static_fields,
        path_name_receipt=_path_name_receipt(
            top_level_rows,
            audit_bundle_root=audit_bundle_root,
            audited_path_tree_root=path_root,
            content_blob_audit=content_blob_audit,
        ),
    )


def build_parent_absence_attestation_fields_v2(
    evidence: ParentAbsenceAuditEvidence,
    *,
    auditor_key_id: bytes,
    audited_at_unix_seconds: int,
) -> dict[str, object]:
    """Complete attestation fields without generating or using a private key."""

    replay_parent_absence_audit_v1(evidence)
    fields = dict(evidence.attestation_static_fields)
    fields.update(
        {
            "auditor_key_id": auditor_key_id,
            "audited_at_unix_seconds": audited_at_unix_seconds,
        }
    )
    build_formal_object("ParentManifestAbsenceAttestationV2", fields)
    return fields


def _evidence_identity(evidence: ParentAbsenceAuditEvidence) -> tuple[object, ...]:
    return (
        tuple(
            encode_formal_object("AuditedPathBlobRecordV1", row)
            for row in evidence.top_level_path_rows
        ),
        tuple(
            encode_formal_object("AuditedHistoryRowV1", row)
            for row in evidence.history_rows
        ),
        tuple(
            tuple(
                encode_formal_object("AuditedPathBlobRecordV1", row) for row in rows
            )
            for rows in evidence.touched_path_rows_by_history_row
        ),
        tuple(
            encode_formal_object("LegacyParentSourceRowV1", row)
            for row in evidence.legacy_source_rows
        ),
        encode_formal_object("ParentAbsenceAuditBundleV1", evidence.audit_bundle_fields),
        evidence.audit_bundle_root,
        tuple(sorted(evidence.path_name_receipt.items(), key=lambda item: item[0])),
    )


def _require_sha256_hex(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(FAIL_REPLAY_MISMATCH, f"{field} is not a SHA-256 hex digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        _fail(FAIL_REPLAY_MISMATCH, f"{field} is malformed: {error}")
    if len(raw) != 32 or value != value.lower():
        _fail(FAIL_REPLAY_MISMATCH, f"{field} is not canonical lowercase hex")
    return value


def _validate_content_blob_audit(
    top_level_rows: Sequence[Mapping[str, object]],
    content_audit: object,
) -> Mapping[str, object]:
    if not isinstance(content_audit, Mapping):
        _fail(FAIL_REPLAY_MISMATCH, "content blob audit is absent or not a mapping")
    expected_keys = {
        "content_predicate_profile_id",
        "inspected_path_blob_row_count",
        "inspected_unique_blob_count",
        "inspected_total_byte_length",
        "inspected_blob_inventory_sha256",
        "git_blob_object_id_and_size_verified",
        "structured_candidate_unique_blob_count",
        "unscannable_relevant_structured_blob_count",
        "content_absence_predicates",
        "legacy_source_presence",
        "all_content_absence_predicates_absent",
        "all_legacy_sources_present",
    }
    if set(content_audit) != expected_keys:
        _fail(FAIL_REPLAY_MISMATCH, "content blob audit field set differs")

    sizes: dict[bytes, int] = {}
    structured: set[bytes] = set()
    for row in top_level_rows:
        digest = row["git_blob_digest"]
        size = row["byte_length"]
        path = row["raw_repository_path_utf8_bytes"]
        if type(digest) is not bytes or len(digest) != 20 or type(size) is not int:
            _fail(FAIL_REPLAY_MISMATCH, "audited blob identity has a wrong type")
        previous = sizes.setdefault(digest, size)
        if previous != size:
            _fail(FAIL_REPLAY_MISMATCH, "one Git blob digest has conflicting sizes")
        if type(path) is bytes and path.lower().endswith(_STRUCTURED_PATH_SUFFIXES):
            structured.add(digest)
    expected_scalars = {
        "content_predicate_profile_id": CONTENT_PREDICATE_PROFILE_ID,
        "inspected_path_blob_row_count": len(top_level_rows),
        "inspected_unique_blob_count": len(sizes),
        "inspected_total_byte_length": sum(sizes.values()),
        "inspected_blob_inventory_sha256": _blob_inventory_digest(sizes),
        "git_blob_object_id_and_size_verified": True,
        "structured_candidate_unique_blob_count": len(structured),
        "unscannable_relevant_structured_blob_count": 0,
        "all_content_absence_predicates_absent": True,
        "all_legacy_sources_present": True,
    }
    if any(content_audit[key] != value for key, value in expected_scalars.items()):
        _fail(FAIL_REPLAY_MISMATCH, "content blob audit scalar or inventory differs")

    raw_absence = content_audit["content_absence_predicates"]
    if not isinstance(raw_absence, (tuple, list)) or len(raw_absence) != len(
        _CONTENT_ABSENCE_PREDICATES
    ):
        _fail(FAIL_REPLAY_MISMATCH, "content absence predicate set differs")
    for row, (predicate_id, signatures) in zip(
        raw_absence, _CONTENT_ABSENCE_PREDICATES.items(), strict=True
    ):
        expected = {
            "predicate_id": predicate_id,
            "exact_signatures_ascii": [item.decode("ascii") for item in signatures],
            "match_occurrence_count": 0,
            "matching_unique_blob_count": 0,
            "matching_path_blob_row_count": 0,
            "matching_blob_digest_set_sha256": hashlib.sha256(b"").hexdigest(),
            "absent": True,
        }
        if not isinstance(row, Mapping) or dict(row) != expected:
            _fail(FAIL_REPLAY_MISMATCH, f"content absence result differs for {predicate_id}")

    raw_legacy = content_audit["legacy_source_presence"]
    if not isinstance(raw_legacy, (tuple, list)) or len(raw_legacy) != 2:
        _fail(FAIL_REPLAY_MISMATCH, "legacy source content result set differs")
    for row, source_id in zip(raw_legacy, LEGACY_PARENT_SOURCE_IDS, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "legacy_parent_payload_source_id",
            "match_occurrence_count",
            "matching_unique_blob_count",
            "matching_path_blob_row_count",
            "matching_blob_digest_set_sha256",
            "present",
        }:
            _fail(FAIL_REPLAY_MISMATCH, "legacy source content result fields differ")
        if (
            row["legacy_parent_payload_source_id"] != source_id
            or type(row["match_occurrence_count"]) is not int
            or row["match_occurrence_count"] <= 0
            or type(row["matching_unique_blob_count"]) is not int
            or row["matching_unique_blob_count"] <= 0
            or type(row["matching_path_blob_row_count"]) is not int
            or row["matching_path_blob_row_count"] <= 0
            or row["present"] is not True
        ):
            _fail(FAIL_REPLAY_MISMATCH, f"legacy source content is absent for {source_id}")
        _require_sha256_hex(
            row["matching_blob_digest_set_sha256"],
            "matching_blob_digest_set_sha256",
        )
    return content_audit


def replay_parent_absence_audit_v1(
    evidence: ParentAbsenceAuditEvidence,
    *,
    repository: str | Path | None = None,
    git_executable: str | Path = DEFAULT_GIT_EXECUTABLE,
) -> bytes:
    """Replay formal roots and, when supplied, the exact Git-object derivation."""

    root = validate_parent_absence_audit_bundle_v1(
        evidence.top_level_path_rows,
        evidence.history_rows,
        evidence.touched_path_rows_by_history_row,
        evidence.legacy_source_rows,
        evidence.audit_bundle_fields,
    )
    if root != evidence.audit_bundle_root:
        _fail(FAIL_REPLAY_MISMATCH, "stored audit bundle root differs")
    expected_static = {
        "parent_dsl_version_digest": id_digest_v1(PARENT_DSL_VERSION),
        "parent_freeze_version_digest": id_digest_v1(PARENT_FREEZE_VERSION),
        "parent_repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "audit_bundle_root": root,
        "absence_reason_bitmask": 0b1111,
    }
    if dict(evidence.attestation_static_fields) != expected_static:
        _fail(FAIL_REPLAY_MISMATCH, "attestation-ready static fields differ")
    content_blob_audit = _validate_content_blob_audit(
        evidence.top_level_path_rows,
        evidence.path_name_receipt.get("content_blob_audit"),
    )
    expected_receipt = _path_name_receipt(
        evidence.top_level_path_rows,
        audit_bundle_root=root,
        audited_path_tree_root=evidence.audit_bundle_fields["audited_path_tree_root"],  # type: ignore[arg-type]
        content_blob_audit=content_blob_audit,
    )
    if dict(evidence.path_name_receipt) != dict(expected_receipt):
        _fail(FAIL_REPLAY_MISMATCH, "path-name diagnostic receipt differs")
    if repository is not None:
        regenerated = generate_parent_absence_audit_v1(
            repository,
            git_executable=git_executable,
        )
        if _evidence_identity(evidence) != _evidence_identity(regenerated):
            _fail(FAIL_REPLAY_MISMATCH, "evidence differs from Git-object regeneration")
    return root


def parent_absence_public_receipt_v1(
    evidence: ParentAbsenceAuditEvidence,
) -> dict[str, object]:
    """Return a compact JSON-safe receipt; full replay reads the bound Git history."""

    replay_parent_absence_audit_v1(evidence)
    bundle = evidence.audit_bundle_fields
    history_rows = evidence.history_rows
    merge_rows = [row for row in history_rows if len(row["ordered_parent_commit_ids"]) > 1]  # type: ignore[arg-type]
    root_rows = [row for row in history_rows if not row["ordered_parent_commit_ids"]]
    return {
        **dict(evidence.path_name_receipt),
        "audited_history_tree_root": bundle["audited_history_tree_root"].hex(),  # type: ignore[union-attr]
        "legacy_source_tree_root": bundle["legacy_source_tree_root"].hex(),  # type: ignore[union-attr]
        "audited_path_count": bundle["audited_path_count"],
        "audited_history_row_count": bundle["audited_history_row_count"],
        "legacy_source_count": bundle["legacy_source_count"],
        "root_commit_count": len(root_rows),
        "merge_commit_count": len(merge_rows),
        "legacy_parent_source_ids": list(LEGACY_PARENT_SOURCE_IDS),
        "attestation_static_fields": {
            "parent_dsl_version_digest": evidence.attestation_static_fields[
                "parent_dsl_version_digest"
            ].hex(),  # type: ignore[union-attr]
            "parent_freeze_version_digest": evidence.attestation_static_fields[
                "parent_freeze_version_digest"
            ].hex(),  # type: ignore[union-attr]
            "parent_repository_commit_sha1": AUDITED_PARENT_COMMIT_SHA1.hex(),
            "audit_bundle_root": evidence.audit_bundle_root.hex(),
            "absence_reason_bitmask": 0b1111,
        },
        "replay_requires_git_objects": True,
    }


__all__ = [
    "AUDITED_PARENT_COMMIT_SHA1",
    "FAIL_GIT_AUDIT",
    "FAIL_GIT_COMMIT_OBJECT",
    "FAIL_GIT_DIFF_RECORD",
    "FAIL_GIT_NON_BLOB",
    "FAIL_GIT_OBJECT_FORMAT",
    "FAIL_GIT_PATH_ENCODING",
    "FAIL_GIT_REACHABILITY",
    "FAIL_GIT_SHALLOW",
    "FAIL_REPLAY_MISMATCH",
    "PARENT_DSL_VERSION",
    "PARENT_FREEZE_VERSION",
    "PATH_ALIAS_RULE_ID",
    "PATH_NAME_PREDICATE_PROFILE_ID",
    "PUBLIC_RECEIPT_SCHEMA_ID",
    "ParentAbsenceAuditError",
    "ParentAbsenceAuditEvidence",
    "TOUCHED_PATH_RULE_ID",
    "build_parent_absence_attestation_fields_v2",
    "generate_parent_absence_audit_v1",
    "parent_absence_public_receipt_v1",
    "replay_parent_absence_audit_v1",
]
