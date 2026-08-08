"""Owner-accepted local two-commit admission for the formal M3 runtime.

The protocol deliberately makes a narrower claim than an external signature:

* Commit C contains the complete runtime and this validator;
* Commit D is the single child of C and adds only one canonical approval blob;
* the approved runtime paths are identical in C, D, the stage-zero index and
  the live working tree; and
* the returned receipt says ``LOCAL_TWO_COMMIT_ADMISSION`` and nothing more.

The artifact builder is usable while HEAD is Commit C and the approval path is
still absent.  It returns bytes but never writes them.  The live validator is
usable only after those exact bytes have been committed as Commit D.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_m3_runtime_source_preflight_v1 import (
    DEFAULT_M3_RUNTIME_SOURCE_PATHS,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
GIT_EXECUTABLE: Final = Path("/usr/bin/git")

PUBLICATION_COMMIT_B: Final = "78d5c77994ad9088c082c32a948b5a2b40407966"
BASIS_COMMIT_A: Final = "0af65964235390ce2bebefea7379eaa9c50eda24"
FORMAL_RUN_ID_HEX: Final = "e4af9f57c38fb298462ec628c4ed8a03"
EXECUTION_MANIFEST_ROOT_HEX: Final = (
    "fd84e901e2259943ebf981eeaee8d6dd807c6ca82ae0f89315c57a4808659453"
)
CANONICAL_ATTEMPT_ID: Final = "attempt-1"

ADMISSION_MODULE_REPOSITORY_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_local_admission_v1.py"
)
APPROVAL_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m3_runtime/"
    "local_two_commit_admission_v1.json"
)
DIRECT_ENTRYPOINT_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_start_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_formal_execution_entrypoint_v1.py",
)
M3_RUNTIME_SOURCE_PATHS: tuple[str, ...] = tuple(
    sorted(
        {*DEFAULT_M3_RUNTIME_SOURCE_PATHS, ADMISSION_MODULE_REPOSITORY_PATH},
        key=lambda value: value.encode("utf-8"),
    )
)

ARTIFACT_SCHEMA: Final = "hegel-phase3-m3-local-two-commit-admission/1"
ARTIFACT_KIND: Final = "M3_LOCAL_TWO_COMMIT_ADMISSION"
CLAIM_LEVEL: Final = "LOCAL_TWO_COMMIT_ADMISSION"
POLICY_ID: Final = "LOCAL_TWO_COMMIT_ADMISSION_V1"
APPROVED_SCOPE: Final = "CANONICAL_ENUMERATION_ONLY"
APPROVAL_STATEMENT: Final = (
    "OWNER_ACCEPTED_RUNTIME_C_ADMITTED_FOR_CANONICAL_M3_RUN_ONLY"
)
MANIFEST_SCHEMA: Final = "hegel-m3-runtime-source-manifest/1"
RECEIPT_SCHEMA: Final = (
    "hegel-phase3-m3-local-two-commit-admission-receipt/1"
)

SOURCE_SET_DOMAIN: Final = b"HEGEL/M3_RUNTIME_SOURCE_SET/V1"
PATH_SET_DOMAIN: Final = b"HEGEL/M3/RUNTIME_SOURCE_PATH_SET/V1"
ARTIFACT_HASH_DOMAIN: Final = b"HEGEL/M3/LOCAL_TWO_COMMIT_ADMISSION/V1"
RECEIPT_HASH_DOMAIN: Final = b"HEGEL/M3/LOCAL_TWO_COMMIT_RECEIPT/V1"

MAX_RUNTIME_SOURCE_FILES: Final = 128
MAX_RUNTIME_SOURCE_FILE_BYTES: Final = 32 * 1024 * 1024
MAX_APPROVAL_ARTIFACT_BYTES: Final = 4 * 1024 * 1024

FAIL_REPOSITORY: Final = "FAIL_M3_RUNTIME_ADMISSION_REPOSITORY"
FAIL_HEAD: Final = "FAIL_M3_RUNTIME_ADMISSION_HEAD"
FAIL_TOPOLOGY: Final = "FAIL_M3_RUNTIME_ADMISSION_TOPOLOGY"
FAIL_DIFF: Final = "FAIL_M3_RUNTIME_ADMISSION_DIFF"
FAIL_APPROVAL_BLOB: Final = "FAIL_M3_RUNTIME_ADMISSION_APPROVAL_BLOB"
FAIL_APPROVAL_CANONICAL: Final = "FAIL_M3_RUNTIME_ADMISSION_APPROVAL_CANONICAL"
FAIL_ARTIFACT_BINDING: Final = "FAIL_M3_RUNTIME_ADMISSION_ARTIFACT_BINDING"
FAIL_SOURCE_IDENTITY: Final = "FAIL_M3_RUNTIME_ADMISSION_SOURCE_IDENTITY"
FAIL_INDEX: Final = "FAIL_M3_RUNTIME_ADMISSION_INDEX"
FAIL_WORKTREE: Final = "FAIL_M3_RUNTIME_ADMISSION_WORKTREE"
FAIL_SYMLINK: Final = "FAIL_M3_RUNTIME_ADMISSION_SYMLINK"
FAIL_RACE: Final = "FAIL_M3_RUNTIME_ADMISSION_RACE"
FAIL_RECEIPT: Final = "FAIL_M3_RUNTIME_ADMISSION_RECEIPT"

_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}")
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")


class M3LocalAdmissionError(RuntimeError):
    """Stable fail-closed error raised by the local admission boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3LocalAdmissionError(code, detail)


def _freeze_json_v1(value: object) -> object:
    """Recursively freeze JSON-shaped evidence returned as a capability."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_v1(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json_v1(item) for item in value)
    return value


def _json_array_tuple_v1(value: object) -> tuple[object, ...] | None:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return None


@dataclass(frozen=True, slots=True)
class BuiltLocalAdmissionArtifactV1:
    runtime_commit_c: str
    fields: Mapping[str, object]
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class LocalTwoCommitAdmissionResultV1:
    runtime_commit_c: str
    approval_commit_d: str
    artifact_fields: Mapping[str, object]
    manifest_fields: Mapping[str, object]
    receipt_fields: Mapping[str, object]


class _Pairs(tuple):
    pass


def canonical_json_v1(value: object) -> bytes:
    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(child) for child in item]
        return item

    try:
        return (
            json.dumps(
                plain(value),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        _fail(FAIL_APPROVAL_CANONICAL, f"value is not canonicalizable: {exc}")


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> _Pairs:
        keys = [key for key, _value in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate object key")
        return _Pairs(pairs)

    def reject_number(token: str) -> NoReturn:
        raise ValueError(f"non-integer JSON number {token}")

    try:
        decoded = json.loads(
            payload.decode("ascii", "strict"),
            object_pairs_hook=pairs_hook,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        _fail(FAIL_APPROVAL_CANONICAL, f"approval is not strict JSON: {type(exc).__name__}")

    def plain(value: object) -> object:
        if isinstance(value, _Pairs):
            return {key: plain(item) for key, item in value}
        if type(value) is list:
            return [plain(item) for item in value]
        if value is None or type(value) in {bool, int, str}:
            return value
        _fail(FAIL_APPROVAL_CANONICAL, "approval contains an unsupported JSON value")

    result = plain(decoded)
    if type(result) is not dict or canonical_json_v1(result) != payload:
        _fail(FAIL_APPROVAL_CANONICAL, "approval is not canonical JSON object bytes")
    return result


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_v1(value)).hexdigest()


def _git_environment_v1() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_PAGER": "cat",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    code: str,
) -> bytes:
    try:
        executable = GIT_EXECUTABLE.lstat()
    except OSError as exc:
        _fail(code, f"formal Git executable is unavailable: {type(exc).__name__}")
    if (
        not arguments
        or any(
            type(value) is not str or not value or "\x00" in value
            for value in arguments
        )
        or not stat.S_ISREG(executable.st_mode)
        or executable.st_uid != 0
        or stat.S_IMODE(executable.st_mode) != 0o755
        or GIT_EXECUTABLE.resolve(strict=True) != GIT_EXECUTABLE
    ):
        _fail(code, "formal Git executable or argument vector differs")
    try:
        completed = subprocess.run(
            [
                GIT_EXECUTABLE.as_posix(),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "core.pager=cat",
                *arguments,
            ],
            cwd=repository,
            env=_git_environment_v1(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(code, f"Git read failed: {type(exc).__name__}")
    if completed.returncode != 0 or completed.stderr:
        _fail(code, "Git rejected an admission identity query")
    return completed.stdout


def _require_repository(repository_root: Path) -> Path:
    if not repository_root.is_absolute():
        _fail(FAIL_REPOSITORY, "repository root must be absolute")
    requested = Path(os.path.abspath(os.fspath(repository_root)))
    try:
        lexical = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_REPOSITORY, f"repository root is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(lexical.st_mode)
        or not stat.S_ISDIR(lexical.st_mode)
        or requested != resolved
    ):
        _fail(FAIL_REPOSITORY, "repository root must be one real direct directory")
    top = _git(resolved, ("rev-parse", "--show-toplevel"), code=FAIL_REPOSITORY)
    try:
        observed = Path(top.decode("utf-8", "strict").strip())
    except UnicodeDecodeError:
        _fail(FAIL_REPOSITORY, "Git top-level path is not UTF-8")
    if observed != resolved:
        _fail(FAIL_REPOSITORY, "repository root is not the exact Git top level")
    return resolved


def _resolve_commit(repository: Path, revision: str, *, code: str) -> str:
    if type(revision) is not str or not revision or "\x00" in revision:
        _fail(code, "commit revision is malformed")
    payload = _git(
        repository,
        ("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"),
        code=code,
    )
    try:
        commit = payload.decode("ascii", "strict").strip()
    except UnicodeDecodeError:
        _fail(code, "resolved commit is not ASCII")
    if _SHA1_RE.fullmatch(commit) is None:
        _fail(code, "revision did not resolve to one full SHA-1 commit")
    return commit


def _commit_tree(repository: Path, commit: str) -> str:
    payload = _git(
        repository,
        ("show", "-s", "--format=%T", commit),
        code=FAIL_TOPOLOGY,
    )
    try:
        value = payload.decode("ascii", "strict").strip()
    except UnicodeDecodeError:
        _fail(FAIL_TOPOLOGY, "commit tree ID is not ASCII")
    if _SHA1_RE.fullmatch(value) is None:
        _fail(FAIL_TOPOLOGY, "commit tree ID is malformed")
    return value


def _commit_parents(repository: Path, commit: str) -> tuple[str, ...]:
    payload = _git(
        repository,
        ("show", "-s", "--format=%P", commit),
        code=FAIL_TOPOLOGY,
    )
    try:
        values = tuple(payload.decode("ascii", "strict").strip().split())
    except UnicodeDecodeError:
        _fail(FAIL_TOPOLOGY, "commit parent IDs are not ASCII")
    if any(_SHA1_RE.fullmatch(value) is None for value in values):
        _fail(FAIL_TOPOLOGY, "commit parent list is malformed")
    return values


def _normalize_paths(runtime_paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(runtime_paths, (str, bytes)):
        _fail(FAIL_SOURCE_IDENTITY, "runtime source paths must be a sequence")
    values = tuple(runtime_paths)
    if not values or len(values) > MAX_RUNTIME_SOURCE_FILES:
        _fail(FAIL_SOURCE_IDENTITY, "runtime source path count is outside the bound")
    normalized: list[str] = []
    for raw in values:
        if type(raw) is not str or not raw or "\x00" in raw or "\\" in raw:
            _fail(FAIL_SOURCE_IDENTITY, "runtime source path is malformed")
        path = PurePosixPath(raw)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            _fail(FAIL_SOURCE_IDENTITY, "runtime source path is not canonical relative text")
        canonical = path.as_posix()
        if canonical != raw:
            _fail(FAIL_SOURCE_IDENTITY, "runtime source path is not canonical POSIX text")
        normalized.append(canonical)
    ordered = tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))
    if len(ordered) != len(set(ordered)):
        _fail(FAIL_SOURCE_IDENTITY, "runtime source path set contains a duplicate")
    if ADMISSION_MODULE_REPOSITORY_PATH not in ordered or any(
        path not in ordered for path in DIRECT_ENTRYPOINT_PATHS
    ):
        _fail(FAIL_SOURCE_IDENTITY, "runtime closure omits admission or direct entrypoint code")
    return ordered


def _tree_entry(
    repository: Path,
    commit: str,
    repository_path: str,
    *,
    missing_ok: bool = False,
) -> tuple[str, str] | None:
    payload = _git(
        repository,
        ("ls-tree", "-z", "--full-tree", commit, "--", repository_path),
        code=FAIL_SOURCE_IDENTITY,
    )
    rows = tuple(row for row in payload.split(b"\x00") if row)
    if missing_ok and not rows:
        return None
    if len(rows) != 1:
        _fail(FAIL_SOURCE_IDENTITY, f"tree path is absent or ambiguous: {repository_path}")
    try:
        metadata, raw_path = rows[0].split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii", "strict").split(" ")
        observed_path = raw_path.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_SOURCE_IDENTITY, f"tree row is malformed: {repository_path}")
    if (
        observed_path != repository_path
        or kind != "blob"
        or mode not in {"100644", "100755"}
        or _SHA1_RE.fullmatch(object_id) is None
    ):
        _fail(FAIL_SOURCE_IDENTITY, f"tree identity differs: {repository_path}")
    return mode, object_id


def _index_entry(repository: Path, repository_path: str) -> tuple[str, str] | None:
    payload = _git(
        repository,
        ("ls-files", "--stage", "-z", "--", repository_path),
        code=FAIL_INDEX,
    )
    rows = tuple(row for row in payload.split(b"\x00") if row)
    if not rows:
        return None
    if len(rows) != 1:
        _fail(FAIL_INDEX, f"index path is absent or unmerged: {repository_path}")
    try:
        metadata, raw_path = rows[0].split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii", "strict").split(" ")
        observed_path = raw_path.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_INDEX, f"index row is malformed: {repository_path}")
    if (
        observed_path != repository_path
        or stage != "0"
        or mode not in {"100644", "100755"}
        or _SHA1_RE.fullmatch(object_id) is None
    ):
        _fail(FAIL_INDEX, f"index identity differs: {repository_path}")
    return mode, object_id


def _blob_payload(repository: Path, object_id: str, *, maximum_bytes: int) -> bytes:
    payload = _git(
        repository,
        ("cat-file", "blob", object_id),
        code=FAIL_SOURCE_IDENTITY,
    )
    if not payload or len(payload) > maximum_bytes:
        _fail(FAIL_SOURCE_IDENTITY, "Git blob size is outside the bound")
    preimage = b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload
    if hashlib.sha1(preimage).hexdigest() != object_id:
        _fail(FAIL_SOURCE_IDENTITY, "Git blob bytes differ from their object ID")
    return payload


def _read_worktree_file(
    repository: Path,
    repository_path: str,
    *,
    expected_mode: str,
    maximum_bytes: int,
) -> bytes:
    components = PurePosixPath(repository_path).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptors: list[int] = []
    directory_links: list[tuple[int, str, int]] = []
    try:
        current = os.open(repository, directory_flags)
        descriptors.append(current)
        root_before = os.fstat(current)
        root_lexical_before = repository.lstat()
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or (root_before.st_dev, root_before.st_ino)
            != (root_lexical_before.st_dev, root_lexical_before.st_ino)
        ):
            _fail(FAIL_RACE, "repository root changed before stable runtime read")
        for component in components[:-1]:
            parent = current
            try:
                current = os.open(component, directory_flags, dir_fd=current)
            except OSError as exc:
                code = FAIL_SYMLINK if exc.errno == 40 else FAIL_WORKTREE
                _fail(code, f"cannot open runtime ancestor safely: {repository_path}")
            descriptors.append(current)
            directory_links.append((parent, component, current))
        try:
            descriptor = os.open(
                components[-1],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
        except OSError as exc:
            code = FAIL_SYMLINK if exc.errno == 40 else FAIL_WORKTREE
            _fail(code, f"cannot open runtime file safely: {repository_path}")
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        exact_mode = 0o755 if expected_mode == "100755" else 0o644
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != exact_mode
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            _fail(FAIL_WORKTREE, f"runtime file metadata differs: {repository_path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_WORKTREE, f"runtime file read was short: {repository_path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_WORKTREE, f"runtime file grew during read: {repository_path}")
        after = os.fstat(descriptor)
        namespace = os.stat(
            components[-1],
            dir_fd=current,
            follow_symlinks=False,
        )
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or (namespace.st_dev, namespace.st_ino) != (after.st_dev, after.st_ino)
        ):
            _fail(FAIL_RACE, f"runtime file changed while read: {repository_path}")
        for parent, component, child in directory_links:
            named = os.stat(component, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                _fail(FAIL_RACE, f"runtime ancestor changed while read: {repository_path}")
        root_after = os.fstat(descriptors[0])
        root_lexical_after = repository.lstat()
        if (
            (root_after.st_dev, root_after.st_ino)
            != (root_before.st_dev, root_before.st_ino)
            or (root_lexical_after.st_dev, root_lexical_after.st_ino)
            != (root_after.st_dev, root_after.st_ino)
        ):
            _fail(FAIL_RACE, "repository root changed during stable runtime read")
        return b"".join(chunks)
    except M3LocalAdmissionError:
        raise
    except OSError as exc:
        _fail(FAIL_WORKTREE, f"stable runtime read failed: {type(exc).__name__}")
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _source_manifest_for_commit(
    repository: Path,
    commit: str,
    paths: Sequence[str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for repository_path in paths:
        entry = _tree_entry(repository, commit, repository_path)
        assert entry is not None
        mode, object_id = entry
        payload = _blob_payload(
            repository, object_id, maximum_bytes=MAX_RUNTIME_SOURCE_FILE_BYTES
        )
        rows.append(
            {
                "repository_path": repository_path,
                "git_mode": mode,
                "checkout_mode_octal": "0755" if mode == "100755" else "0644",
                "git_blob_sha1": object_id,
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    source_set_sha256 = hashlib.sha256(
        SOURCE_SET_DOMAIN + b"\x00" + canonical_json_v1(rows)
    ).hexdigest()
    return {
        "schema": MANIFEST_SCHEMA,
        "expected_runtime_commit": commit,
        "expected_runtime_tree": _commit_tree(repository, commit),
        "runtime_source_file_count": len(rows),
        "runtime_source_files": rows,
        "runtime_source_set_sha256": source_set_sha256,
    }


def _path_set_sha256(paths: Sequence[str]) -> str:
    return hashlib.sha256(
        PATH_SET_DOMAIN + b"\x00" + canonical_json_v1(list(paths))
    ).hexdigest()


def _expected_artifact_fields(
    repository: Path,
    runtime_commit_c: str,
    paths: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = _source_manifest_for_commit(repository, runtime_commit_c, paths)
    body: dict[str, object] = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "claim_level": CLAIM_LEVEL,
        "policy_id": POLICY_ID,
        "approval_repository_path": APPROVAL_REPOSITORY_PATH,
        "publication_commit_b": PUBLICATION_COMMIT_B,
        "basis_commit_a": BASIS_COMMIT_A,
        "runtime_parent_commit_b": PUBLICATION_COMMIT_B,
        "runtime_commit_c": runtime_commit_c,
        "runtime_tree_c": manifest["expected_runtime_tree"],
        "runtime_source_manifest_schema": MANIFEST_SCHEMA,
        "runtime_source_path_count": len(paths),
        "runtime_source_paths": list(paths),
        "runtime_source_path_set_sha256": _path_set_sha256(paths),
        "runtime_source_set_sha256": manifest["runtime_source_set_sha256"],
        "runtime_source_manifest_sha256": hashlib.sha256(
            canonical_json_v1(manifest)
        ).hexdigest(),
        "formal_run_id_hex": FORMAL_RUN_ID_HEX,
        "execution_manifest_root_hex": EXECUTION_MANIFEST_ROOT_HEX,
        "canonical_attempt_id": CANONICAL_ATTEMPT_ID,
        "approved_action_ids": [
            "phase3-m3-start",
            "phase3-m3-formal-execution",
        ],
        "approved_scope": APPROVED_SCOPE,
        "direct_entrypoint_paths": list(DIRECT_ENTRYPOINT_PATHS),
        "formal_m3_start_allowed": True,
        "formal_m3_formal_execution_allowed": True,
        "role_evaluation_allowed": False,
        "active_promotion_allowed": False,
        "network_fetch_allowed": False,
        "docker_pull_allowed": False,
        "external_actor_attestation": False,
        "external_signatures": [],
        "approval_statement": APPROVAL_STATEMENT,
    }
    fields = dict(body)
    fields["approval_artifact_sha256"] = _domain_hash(ARTIFACT_HASH_DOMAIN, body)
    return fields, manifest


def _require_frozen_topology(repository: Path, runtime_commit_c: str) -> None:
    if _commit_parents(repository, runtime_commit_c) != (PUBLICATION_COMMIT_B,):
        _fail(FAIL_TOPOLOGY, "Commit C is not the sole child of Commit B")
    if _commit_parents(repository, PUBLICATION_COMMIT_B) != (BASIS_COMMIT_A,):
        _fail(FAIL_TOPOLOGY, "Commit B is not the sole child of Commit A")


def _verify_live_paths(
    repository: Path,
    *,
    runtime_commit_c: str,
    approval_commit_d: str,
    paths: Sequence[str],
) -> tuple[tuple[object, ...], ...]:
    snapshot: list[tuple[object, ...]] = []
    for repository_path in paths:
        c_entry = _tree_entry(repository, runtime_commit_c, repository_path)
        d_entry = _tree_entry(repository, approval_commit_d, repository_path)
        if c_entry is None or d_entry is None or c_entry != d_entry:
            _fail(FAIL_SOURCE_IDENTITY, f"runtime path differs between C and D: {repository_path}")
        index_entry = _index_entry(repository, repository_path)
        if index_entry != d_entry:
            _fail(FAIL_INDEX, f"runtime index differs from D: {repository_path}")
        mode, object_id = d_entry
        committed = _blob_payload(
            repository, object_id, maximum_bytes=MAX_RUNTIME_SOURCE_FILE_BYTES
        )
        working = _read_worktree_file(
            repository,
            repository_path,
            expected_mode=mode,
            maximum_bytes=MAX_RUNTIME_SOURCE_FILE_BYTES,
        )
        if working != committed:
            _fail(FAIL_WORKTREE, f"runtime worktree bytes differ from D: {repository_path}")
        snapshot.append(
            (
                repository_path,
                mode,
                object_id,
                len(committed),
                hashlib.sha256(committed).hexdigest(),
            )
        )
    return tuple(snapshot)


def _verify_approval_live_path(
    repository: Path,
    *,
    runtime_commit_c: str,
    approval_commit_d: str,
) -> tuple[bytes, str, str]:
    if _tree_entry(
        repository,
        runtime_commit_c,
        APPROVAL_REPOSITORY_PATH,
        missing_ok=True,
    ) is not None:
        _fail(FAIL_DIFF, "approval artifact already exists in Commit C")
    d_entry = _tree_entry(repository, approval_commit_d, APPROVAL_REPOSITORY_PATH)
    if d_entry is None or d_entry[0] != "100644":
        _fail(FAIL_APPROVAL_BLOB, "Commit D approval is not one mode-100644 blob")
    if _index_entry(repository, APPROVAL_REPOSITORY_PATH) != d_entry:
        _fail(FAIL_INDEX, "approval artifact index differs from Commit D")
    payload = _blob_payload(
        repository, d_entry[1], maximum_bytes=MAX_APPROVAL_ARTIFACT_BYTES
    )
    working = _read_worktree_file(
        repository,
        APPROVAL_REPOSITORY_PATH,
        expected_mode=d_entry[0],
        maximum_bytes=MAX_APPROVAL_ARTIFACT_BYTES,
    )
    if working != payload:
        _fail(FAIL_WORKTREE, "approval artifact worktree bytes differ from Commit D")
    return payload, d_entry[1], hashlib.sha256(payload).hexdigest()


def _require_clean_status(repository: Path, paths: Sequence[str]) -> None:
    payload = _git(
        repository,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *paths,
        ),
        code=FAIL_WORKTREE,
    )
    if payload:
        _fail(FAIL_WORKTREE, "admission path set has staged, unstaged, or untracked changes")


def _approval_path_absent_before_d(repository: Path, runtime_commit_c: str) -> None:
    if _tree_entry(
        repository,
        runtime_commit_c,
        APPROVAL_REPOSITORY_PATH,
        missing_ok=True,
    ) is not None or _index_entry(repository, APPROVAL_REPOSITORY_PATH) is not None:
        _fail(FAIL_DIFF, "approval artifact is already tracked before Commit D")
    approval_path = repository / APPROVAL_REPOSITORY_PATH
    try:
        approval_path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        _fail(FAIL_WORKTREE, f"approval path cannot be inspected: {type(exc).__name__}")
    _fail(FAIL_DIFF, "approval artifact worktree path already exists before Commit D")


def build_local_admission_artifact_v1(
    runtime_revision: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> BuiltLocalAdmissionArtifactV1:
    """Build, but never persist, Commit D's approval bytes while at Commit C."""

    repository = _require_repository(repository_root)
    paths = _normalize_paths(M3_RUNTIME_SOURCE_PATHS)
    runtime_commit_c = _resolve_commit(repository, runtime_revision, code=FAIL_HEAD)
    head_before = _resolve_commit(repository, "HEAD", code=FAIL_HEAD)
    if head_before != runtime_commit_c:
        _fail(FAIL_HEAD, "artifact builder requires HEAD to equal Commit C")
    _require_frozen_topology(repository, runtime_commit_c)
    _approval_path_absent_before_d(repository, runtime_commit_c)
    first = _verify_live_paths(
        repository,
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=runtime_commit_c,
        paths=paths,
    )
    _require_clean_status(repository, (*paths, APPROVAL_REPOSITORY_PATH))
    fields, _manifest = _expected_artifact_fields(repository, runtime_commit_c, paths)
    second = _verify_live_paths(
        repository,
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=runtime_commit_c,
        paths=paths,
    )
    head_after = _resolve_commit(repository, "HEAD", code=FAIL_RACE)
    if first != second or head_after != head_before:
        _fail(FAIL_RACE, "Commit C or runtime path snapshot changed during build")
    _require_clean_status(repository, (*paths, APPROVAL_REPOSITORY_PATH))
    payload = canonical_json_v1(fields)
    frozen_fields = _freeze_json_v1(fields)
    assert isinstance(frozen_fields, Mapping)
    return BuiltLocalAdmissionArtifactV1(
        runtime_commit_c=runtime_commit_c,
        fields=frozen_fields,
        canonical_bytes=payload,
    )


def _validate_embedded_artifact_and_manifest_v1(
    artifact: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    artifact_fields = {
        "schema",
        "artifact_kind",
        "claim_level",
        "policy_id",
        "approval_repository_path",
        "publication_commit_b",
        "basis_commit_a",
        "runtime_parent_commit_b",
        "runtime_commit_c",
        "runtime_tree_c",
        "runtime_source_manifest_schema",
        "runtime_source_path_count",
        "runtime_source_paths",
        "runtime_source_path_set_sha256",
        "runtime_source_set_sha256",
        "runtime_source_manifest_sha256",
        "formal_run_id_hex",
        "execution_manifest_root_hex",
        "canonical_attempt_id",
        "approved_action_ids",
        "approved_scope",
        "direct_entrypoint_paths",
        "formal_m3_start_allowed",
        "formal_m3_formal_execution_allowed",
        "role_evaluation_allowed",
        "active_promotion_allowed",
        "network_fetch_allowed",
        "docker_pull_allowed",
        "external_actor_attestation",
        "external_signatures",
        "approval_statement",
        "approval_artifact_sha256",
    }
    manifest_fields = {
        "schema",
        "expected_runtime_commit",
        "expected_runtime_tree",
        "runtime_source_file_count",
        "runtime_source_files",
        "runtime_source_set_sha256",
    }
    row_fields = {
        "repository_path",
        "git_mode",
        "checkout_mode_octal",
        "git_blob_sha1",
        "byte_length",
        "sha256",
    }
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != artifact_fields
        or not isinstance(manifest, Mapping)
        or set(manifest) != manifest_fields
    ):
        _fail(FAIL_ARTIFACT_BINDING, "embedded artifact or manifest field set differs")
    paths = _normalize_paths(M3_RUNTIME_SOURCE_PATHS)
    body = dict(artifact)
    claimed = body.pop("approval_artifact_sha256", None)
    runtime_commit_c = artifact.get("runtime_commit_c")
    runtime_tree_c = artifact.get("runtime_tree_c")
    rows = manifest.get("runtime_source_files")
    if (
        artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("artifact_kind") != ARTIFACT_KIND
        or artifact.get("claim_level") != CLAIM_LEVEL
        or artifact.get("policy_id") != POLICY_ID
        or artifact.get("approval_repository_path") != APPROVAL_REPOSITORY_PATH
        or artifact.get("publication_commit_b") != PUBLICATION_COMMIT_B
        or artifact.get("basis_commit_a") != BASIS_COMMIT_A
        or artifact.get("runtime_parent_commit_b") != PUBLICATION_COMMIT_B
        or type(runtime_commit_c) is not str
        or _SHA1_RE.fullmatch(runtime_commit_c) is None
        or type(runtime_tree_c) is not str
        or _SHA1_RE.fullmatch(runtime_tree_c) is None
        or artifact.get("runtime_source_manifest_schema") != MANIFEST_SCHEMA
        or artifact.get("runtime_source_path_count") != len(paths)
        or _json_array_tuple_v1(artifact.get("runtime_source_paths")) != paths
        or artifact.get("runtime_source_path_set_sha256") != _path_set_sha256(paths)
        or artifact.get("formal_run_id_hex") != FORMAL_RUN_ID_HEX
        or artifact.get("execution_manifest_root_hex") != EXECUTION_MANIFEST_ROOT_HEX
        or artifact.get("canonical_attempt_id") != CANONICAL_ATTEMPT_ID
        or _json_array_tuple_v1(artifact.get("approved_action_ids"))
        != ("phase3-m3-start", "phase3-m3-formal-execution")
        or artifact.get("approved_scope") != APPROVED_SCOPE
        or _json_array_tuple_v1(artifact.get("direct_entrypoint_paths"))
        != DIRECT_ENTRYPOINT_PATHS
        or artifact.get("formal_m3_start_allowed") is not True
        or artifact.get("formal_m3_formal_execution_allowed") is not True
        or any(
            artifact.get(field) is not False
            for field in (
                "role_evaluation_allowed",
                "active_promotion_allowed",
                "network_fetch_allowed",
                "docker_pull_allowed",
                "external_actor_attestation",
            )
        )
        or _json_array_tuple_v1(artifact.get("external_signatures")) != ()
        or artifact.get("approval_statement") != APPROVAL_STATEMENT
        or type(claimed) is not str
        or _SHA256_RE.fullmatch(claimed) is None
        or claimed != _domain_hash(ARTIFACT_HASH_DOMAIN, body)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("expected_runtime_commit") != runtime_commit_c
        or manifest.get("expected_runtime_tree") != runtime_tree_c
        or manifest.get("runtime_source_file_count") != len(paths)
        or not isinstance(rows, (tuple, list))
        or len(rows) != len(paths)
    ):
        _fail(FAIL_ARTIFACT_BINDING, "embedded artifact static binding differs")
    normalized_rows: list[dict[str, object]] = []
    for expected_path, row in zip(paths, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != row_fields:
            _fail(FAIL_ARTIFACT_BINDING, "runtime source manifest row field set differs")
        mode = row.get("git_mode")
        if (
            row.get("repository_path") != expected_path
            or mode not in {"100644", "100755"}
            or row.get("checkout_mode_octal")
            != ("0755" if mode == "100755" else "0644")
            or type(row.get("git_blob_sha1")) is not str
            or _SHA1_RE.fullmatch(row["git_blob_sha1"]) is None
            or type(row.get("byte_length")) is not int
            or row["byte_length"] < 1
            or row["byte_length"] > MAX_RUNTIME_SOURCE_FILE_BYTES
            or type(row.get("sha256")) is not str
            or _SHA256_RE.fullmatch(row["sha256"]) is None
        ):
            _fail(FAIL_ARTIFACT_BINDING, "runtime source manifest row identity differs")
        normalized_rows.append(dict(row))
    source_set_sha256 = hashlib.sha256(
        SOURCE_SET_DOMAIN + b"\x00" + canonical_json_v1(normalized_rows)
    ).hexdigest()
    manifest_sha256 = hashlib.sha256(canonical_json_v1(manifest)).hexdigest()
    if (
        manifest.get("runtime_source_set_sha256") != source_set_sha256
        or artifact.get("runtime_source_set_sha256") != source_set_sha256
        or artifact.get("runtime_source_manifest_sha256") != manifest_sha256
    ):
        _fail(FAIL_ARTIFACT_BINDING, "embedded runtime source digest binding differs")


def _validate_artifact_fields(
    fields: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> None:
    if not isinstance(fields, Mapping) or set(fields) != set(expected):
        _fail(FAIL_ARTIFACT_BINDING, "approval artifact field set differs")
    body = dict(fields)
    claimed = body.pop("approval_artifact_sha256", None)
    if (
        type(claimed) is not str
        or _SHA256_RE.fullmatch(claimed) is None
        or claimed != _domain_hash(ARTIFACT_HASH_DOMAIN, body)
    ):
        _fail(FAIL_ARTIFACT_BINDING, "approval artifact self-hash differs")
    if dict(fields) != dict(expected):
        _fail(FAIL_ARTIFACT_BINDING, "approval artifact does not bind exact Commit C inputs")


def validate_local_admission_receipt_v1(
    receipt: Mapping[str, object],
    *,
    artifact_fields: Mapping[str, object],
    manifest_fields: Mapping[str, object],
) -> None:
    _validate_embedded_artifact_and_manifest_v1(
        artifact_fields,
        manifest_fields,
    )
    expected_fields = {
        "schema",
        "artifact_kind",
        "claim_level",
        "policy_id",
        "runtime_commit_c",
        "runtime_tree_c",
        "approval_commit_d",
        "approval_tree_d",
        "head_commit",
        "approval_repository_path",
        "approval_blob_sha1",
        "approval_artifact_file_byte_length",
        "approval_artifact_file_sha256",
        "approval_artifact_claim_sha256",
        "runtime_source_path_count",
        "runtime_source_path_set_sha256",
        "runtime_source_set_sha256",
        "runtime_source_manifest_sha256",
        "formal_run_id_hex",
        "execution_manifest_root_hex",
        "canonical_attempt_id",
        "commit_d_single_parent_c",
        "commit_d_adds_only_approval_artifact",
        "runtime_paths_equal_c_d_index_worktree",
        "approval_equal_d_index_worktree",
        "path_scoped_status_clean",
        "symlink_free",
        "exact_file_modes",
        "external_actor_attestation",
        "external_signatures",
        "network_fetch_allowed",
        "docker_pull_allowed",
        "docker_invoked",
        "state_changed",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        _fail(FAIL_RECEIPT, "local admission receipt field set differs")
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    exact_true = (
        "commit_d_single_parent_c",
        "commit_d_adds_only_approval_artifact",
        "runtime_paths_equal_c_d_index_worktree",
        "approval_equal_d_index_worktree",
        "path_scoped_status_clean",
        "symlink_free",
        "exact_file_modes",
    )
    exact_false = (
        "external_actor_attestation",
        "network_fetch_allowed",
        "docker_pull_allowed",
        "docker_invoked",
        "state_changed",
    )
    manifest_digest = hashlib.sha256(canonical_json_v1(manifest_fields)).hexdigest()
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != ARTIFACT_KIND
        or receipt.get("claim_level") != CLAIM_LEVEL
        or receipt.get("policy_id") != POLICY_ID
        or any(receipt.get(field) is not True for field in exact_true)
        or any(receipt.get(field) is not False for field in exact_false)
        or _json_array_tuple_v1(receipt.get("external_signatures")) != ()
        or receipt.get("runtime_commit_c") != artifact_fields.get("runtime_commit_c")
        or receipt.get("runtime_tree_c") != artifact_fields.get("runtime_tree_c")
        or receipt.get("approval_repository_path") != APPROVAL_REPOSITORY_PATH
        or receipt.get("approval_artifact_claim_sha256")
        != artifact_fields.get("approval_artifact_sha256")
        or receipt.get("runtime_source_path_count")
        != artifact_fields.get("runtime_source_path_count")
        or receipt.get("runtime_source_path_set_sha256")
        != artifact_fields.get("runtime_source_path_set_sha256")
        or receipt.get("runtime_source_set_sha256")
        != manifest_fields.get("runtime_source_set_sha256")
        or receipt.get("runtime_source_manifest_sha256") != manifest_digest
        or receipt.get("formal_run_id_hex") != FORMAL_RUN_ID_HEX
        or receipt.get("execution_manifest_root_hex") != EXECUTION_MANIFEST_ROOT_HEX
        or receipt.get("canonical_attempt_id") != CANONICAL_ATTEMPT_ID
        or receipt.get("head_commit") != receipt.get("approval_commit_d")
        or type(receipt.get("approval_commit_d")) is not str
        or _SHA1_RE.fullmatch(receipt["approval_commit_d"]) is None
        or type(receipt.get("approval_tree_d")) is not str
        or _SHA1_RE.fullmatch(receipt["approval_tree_d"]) is None
        or type(receipt.get("approval_blob_sha1")) is not str
        or _SHA1_RE.fullmatch(receipt["approval_blob_sha1"]) is None
        or type(receipt.get("approval_artifact_file_byte_length")) is not int
        or receipt["approval_artifact_file_byte_length"] < 1
        or type(receipt.get("approval_artifact_file_sha256")) is not str
        or _SHA256_RE.fullmatch(receipt["approval_artifact_file_sha256"]) is None
        or type(claimed) is not str
        or _SHA256_RE.fullmatch(claimed) is None
        or claimed != _domain_hash(RECEIPT_HASH_DOMAIN, body)
    ):
        _fail(FAIL_RECEIPT, "local admission receipt identity differs")


def validate_live_local_admission_v1(
    approval_revision: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> LocalTwoCommitAdmissionResultV1:
    """Validate explicit Commit D and return its local-only admission receipt."""

    repository = _require_repository(repository_root)
    paths = _normalize_paths(M3_RUNTIME_SOURCE_PATHS)
    if type(approval_revision) is not str or _SHA1_RE.fullmatch(approval_revision) is None:
        _fail(FAIL_HEAD, "live admission requires explicit full Commit D")
    approval_commit_d = _resolve_commit(repository, approval_revision, code=FAIL_HEAD)
    head_before = _resolve_commit(repository, "HEAD", code=FAIL_HEAD)
    if approval_commit_d != approval_revision or head_before != approval_commit_d:
        _fail(FAIL_HEAD, "explicit Commit D does not equal checked-out HEAD")
    parents = _commit_parents(repository, approval_commit_d)
    if len(parents) != 1:
        _fail(FAIL_TOPOLOGY, "Commit D must have exactly one parent")
    runtime_commit_c = parents[0]
    _require_frozen_topology(repository, runtime_commit_c)
    expected_diff = b"A\x00" + APPROVAL_REPOSITORY_PATH.encode("utf-8") + b"\x00"
    diff_before = _git(
        repository,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-ext-diff",
            "-r",
            "-z",
            "--no-renames",
            runtime_commit_c,
            approval_commit_d,
        ),
        code=FAIL_DIFF,
    )
    if diff_before != expected_diff:
        _fail(FAIL_DIFF, "Commit D must add only the fixed approval artifact")

    expected_artifact, manifest = _expected_artifact_fields(
        repository, runtime_commit_c, paths
    )
    runtime_snapshot_before = _verify_live_paths(
        repository,
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=approval_commit_d,
        paths=paths,
    )
    approval_payload, approval_blob, approval_file_sha256 = _verify_approval_live_path(
        repository,
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=approval_commit_d,
    )
    artifact = _strict_json_object(approval_payload)
    _validate_artifact_fields(artifact, expected=expected_artifact)
    _require_clean_status(repository, (*paths, APPROVAL_REPOSITORY_PATH))

    runtime_snapshot_after = _verify_live_paths(
        repository,
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=approval_commit_d,
        paths=paths,
    )
    approval_payload_after, approval_blob_after, approval_file_sha256_after = (
        _verify_approval_live_path(
            repository,
            runtime_commit_c=runtime_commit_c,
            approval_commit_d=approval_commit_d,
        )
    )
    head_after = _resolve_commit(repository, "HEAD", code=FAIL_RACE)
    parents_after = _commit_parents(repository, approval_commit_d)
    diff_after = _git(
        repository,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-ext-diff",
            "-r",
            "-z",
            "--no-renames",
            runtime_commit_c,
            approval_commit_d,
        ),
        code=FAIL_RACE,
    )
    _require_clean_status(repository, (*paths, APPROVAL_REPOSITORY_PATH))
    if (
        head_after != head_before
        or parents_after != parents
        or diff_after != diff_before
        or runtime_snapshot_after != runtime_snapshot_before
        or approval_payload_after != approval_payload
        or approval_blob_after != approval_blob
        or approval_file_sha256_after != approval_file_sha256
    ):
        _fail(FAIL_RACE, "admission identity changed during live validation")

    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "claim_level": CLAIM_LEVEL,
        "policy_id": POLICY_ID,
        "runtime_commit_c": runtime_commit_c,
        "runtime_tree_c": manifest["expected_runtime_tree"],
        "approval_commit_d": approval_commit_d,
        "approval_tree_d": _commit_tree(repository, approval_commit_d),
        "head_commit": approval_commit_d,
        "approval_repository_path": APPROVAL_REPOSITORY_PATH,
        "approval_blob_sha1": approval_blob,
        "approval_artifact_file_byte_length": len(approval_payload),
        "approval_artifact_file_sha256": approval_file_sha256,
        "approval_artifact_claim_sha256": artifact["approval_artifact_sha256"],
        "runtime_source_path_count": len(paths),
        "runtime_source_path_set_sha256": artifact["runtime_source_path_set_sha256"],
        "runtime_source_set_sha256": manifest["runtime_source_set_sha256"],
        "runtime_source_manifest_sha256": hashlib.sha256(
            canonical_json_v1(manifest)
        ).hexdigest(),
        "formal_run_id_hex": FORMAL_RUN_ID_HEX,
        "execution_manifest_root_hex": EXECUTION_MANIFEST_ROOT_HEX,
        "canonical_attempt_id": CANONICAL_ATTEMPT_ID,
        "commit_d_single_parent_c": True,
        "commit_d_adds_only_approval_artifact": True,
        "runtime_paths_equal_c_d_index_worktree": True,
        "approval_equal_d_index_worktree": True,
        "path_scoped_status_clean": True,
        "symlink_free": True,
        "exact_file_modes": True,
        "external_actor_attestation": False,
        "external_signatures": [],
        "network_fetch_allowed": False,
        "docker_pull_allowed": False,
        "docker_invoked": False,
        "state_changed": False,
    }
    receipt["receipt_sha256"] = _domain_hash(RECEIPT_HASH_DOMAIN, receipt)
    validate_local_admission_receipt_v1(
        receipt,
        artifact_fields=artifact,
        manifest_fields=manifest,
    )
    frozen_artifact = _freeze_json_v1(artifact)
    frozen_manifest = _freeze_json_v1(manifest)
    frozen_receipt = _freeze_json_v1(receipt)
    assert isinstance(frozen_artifact, Mapping)
    assert isinstance(frozen_manifest, Mapping)
    assert isinstance(frozen_receipt, Mapping)
    return LocalTwoCommitAdmissionResultV1(
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=approval_commit_d,
        artifact_fields=frozen_artifact,
        manifest_fields=frozen_manifest,
        receipt_fields=frozen_receipt,
    )


__all__ = [
    "ADMISSION_MODULE_REPOSITORY_PATH",
    "APPROVAL_REPOSITORY_PATH",
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA",
    "BuiltLocalAdmissionArtifactV1",
    "CLAIM_LEVEL",
    "LocalTwoCommitAdmissionResultV1",
    "M3LocalAdmissionError",
    "M3_RUNTIME_SOURCE_PATHS",
    "POLICY_ID",
    "RECEIPT_SCHEMA",
    "build_local_admission_artifact_v1",
    "canonical_json_v1",
    "validate_live_local_admission_v1",
    "validate_local_admission_receipt_v1",
]
