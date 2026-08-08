"""Read-only source-identity preflight for the formal M3 runtime.

This module binds the Python orchestration code that may later cross the
``phase3-m3-start`` boundary to one already-existing local Git commit.  It
does not write state, contact Docker, execute an enumerator, or create a
formal M3 root.

The check is intentionally stricter than an ordinary ``git status``:

* the requested revision and ``HEAD`` must resolve to the same full commit;
* every allowlisted path must be one regular blob in that commit;
* the stage-zero index entry must have the same blob ID and Git mode;
* the working-tree inode must be caller-owned, symlink-free, stably readable,
  byte-identical to the commit blob, and have the exact checkout mode; and
* path-scoped porcelain status must be empty.

The returned manifest and receipt contain only deterministic, public source
identity.  They are diagnostic preflight evidence, not an authority-bearing
certificate or permission to start M3.
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


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
GIT_EXECUTABLE: Final = Path("/usr/bin/git")

MANIFEST_SCHEMA: Final = "hegel-m3-runtime-source-manifest/1"
RECEIPT_SCHEMA: Final = "hegel-m3-runtime-source-preflight/1"
ARTIFACT_KIND: Final = "M3_RUNTIME_SOURCE_IDENTITY_PREFLIGHT"
CLAIM_LEVEL: Final = "RUNTIME_SOURCE_IDENTITY_PREFLIGHT_ONLY"
SOURCE_SET_DOMAIN: Final = b"HEGEL/M3_RUNTIME_SOURCE_SET/V1"

DEFAULT_M3_RUNTIME_SOURCE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/_vendor/__init__.py",
    "Hegel Machine/src/hegel_machine/_vendor/tomli/LICENSE",
    "Hegel Machine/src/hegel_machine/_vendor/tomli/__init__.py",
    "Hegel Machine/src/hegel_machine/_vendor/tomli/_parser.py",
    "Hegel Machine/src/hegel_machine/_vendor/tomli/_re.py",
    "Hegel Machine/src/hegel_machine/_vendor/tomli/_types.py",
    "Hegel Machine/src/hegel_machine/__init__.py",
    "Hegel Machine/src/hegel_machine/hashing.py",
    "Hegel Machine/src/hegel_machine/laws.py",
    "Hegel Machine/src/hegel_machine/milestones.py",
    "Hegel Machine/src/hegel_machine/phase2b_adapter.py",
    "Hegel Machine/src/hegel_machine/phase2b_freeze_v1.py",
    "Hegel Machine/src/hegel_machine/phase2b_selector.py",
    "Hegel Machine/src/hegel_machine/phase2b_wire.py",
    "Hegel Machine/src/hegel_machine/phase3_container_actor_runtime_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_capacity_witness_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_certificate_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_closure_preflight.py",
    "Hegel Machine/src/hegel_machine/phase3_dsl_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_local_runtime_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_actor_protocol_qualification_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_bridge_dag_binary_qualification_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_bridge_dag_node_builder_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_bridge_full_dag_replay_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_commit_b_publication_audit_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_container_ceremony_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_errata_qualification_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_errata_vectors_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_external_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_formal_container_executor_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_formal_static_basis_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_parent_absence_audit_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_purpose4_detached_audit_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_purpose4_keybearing_detached_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_replay_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_rows_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_secret_absence_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_split_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dual_enumeration_supervisor_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_formal_execution_cli_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_formal_execution_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_formal_execution_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_implementation_qualification_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_local_admission_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_offline_docker_runner_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_runtime_source_preflight_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_start_cli_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_start_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_start_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink1_registry_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink1_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink1_publication_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink1_replay_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_strict_replay_v1.py",
    "Hegel Machine/src/hegel_machine/recognition.py",
    "Hegel Machine/src/hegel_machine/schema.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
)

MAX_RUNTIME_SOURCE_FILES: Final = 64
MAX_RUNTIME_SOURCE_FILE_BYTES: Final = 32 * 1024 * 1024

FAIL_REPOSITORY = "FAIL_M3_RUNTIME_SOURCE_REPOSITORY"
FAIL_COMMIT = "FAIL_M3_RUNTIME_SOURCE_COMMIT"
FAIL_PATH_SET = "FAIL_M3_RUNTIME_SOURCE_PATH_SET"
FAIL_GIT_TREE = "FAIL_M3_RUNTIME_SOURCE_GIT_TREE"
FAIL_INDEX = "FAIL_M3_RUNTIME_SOURCE_INDEX"
FAIL_DIRTY = "FAIL_M3_RUNTIME_SOURCE_DIRTY"
FAIL_WORKTREE = "FAIL_M3_RUNTIME_SOURCE_WORKTREE"
FAIL_SYMLINK = "FAIL_M3_RUNTIME_SOURCE_SYMLINK"
FAIL_MODE = "FAIL_M3_RUNTIME_SOURCE_MODE"
FAIL_BYTES = "FAIL_M3_RUNTIME_SOURCE_BYTES"
FAIL_RECEIPT = "FAIL_M3_RUNTIME_SOURCE_RECEIPT"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")


class M3RuntimeSourcePreflightError(RuntimeError):
    """Stable fail-closed runtime-source preflight error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3RuntimeSourcePreflightError(code, detail)


@dataclass(frozen=True, slots=True)
class RuntimeSourcePreflightResultV1:
    """Deterministic public result of one successful read-only preflight."""

    expected_runtime_commit: str
    manifest_fields: Mapping[str, object]
    receipt_fields: Mapping[str, object]


def canonical_json_v1(value: object) -> bytes:
    """Return the unique JSON framing used by this diagnostic artifact."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        _fail(FAIL_RECEIPT, f"runtime-source JSON is not canonicalizable: {exc}")


def _git_environment_v1() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git(repository: Path, arguments: Sequence[str], *, code: str) -> bytes:
    if (
        not arguments
        or any(type(value) is not str or not value or "\x00" in value for value in arguments)
        or not GIT_EXECUTABLE.is_file()
        or GIT_EXECUTABLE.is_symlink()
        or GIT_EXECUTABLE.resolve(strict=True) != GIT_EXECUTABLE
    ):
        _fail(code, "formal Git executable or argument vector differs")
    try:
        completed = subprocess.run(
            [GIT_EXECUTABLE.as_posix(), *arguments],
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
        _fail(code, "Git rejected the runtime-source identity query")
    return completed.stdout


def _require_repository(repository_root: Path) -> Path:
    if not repository_root.is_absolute():
        _fail(FAIL_REPOSITORY, "repository root must be absolute")
    requested = Path(os.path.abspath(os.fspath(repository_root)))
    try:
        metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_REPOSITORY, f"repository root is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or requested != resolved
    ):
        _fail(FAIL_REPOSITORY, "repository root must be a real, direct directory")
    top_level = _git(
        resolved,
        ("rev-parse", "--show-toplevel"),
        code=FAIL_REPOSITORY,
    )
    try:
        observed = Path(top_level.decode("utf-8", "strict").strip())
    except UnicodeDecodeError:
        _fail(FAIL_REPOSITORY, "Git top-level path is not UTF-8")
    if observed != resolved:
        _fail(FAIL_REPOSITORY, "repository root is not the exact Git top level")
    return resolved


def resolve_runtime_commit_v1(repository_root: Path, revision: str) -> str:
    """Resolve one already-existing local revision to a full SHA-1 commit."""

    repository = _require_repository(repository_root)
    if type(revision) is not str or not revision or "\x00" in revision:
        _fail(FAIL_COMMIT, "expected runtime revision is malformed")
    payload = _git(
        repository,
        ("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"),
        code=FAIL_COMMIT,
    )
    try:
        commit = payload.decode("ascii", "strict").strip()
    except UnicodeDecodeError:
        _fail(FAIL_COMMIT, "resolved runtime commit is not ASCII")
    if _SHA1_RE.fullmatch(commit) is None:
        _fail(FAIL_COMMIT, "runtime revision did not resolve to one full SHA-1 commit")
    return commit


def _normalize_paths(runtime_paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(runtime_paths, (str, bytes)):
        _fail(FAIL_PATH_SET, "runtime source paths must be a sequence of paths")
    values = tuple(runtime_paths)
    if not values or len(values) > MAX_RUNTIME_SOURCE_FILES:
        _fail(FAIL_PATH_SET, "runtime source path count is outside the frozen bound")
    normalized: list[str] = []
    for raw in values:
        if type(raw) is not str or not raw or "\x00" in raw or "\\" in raw:
            _fail(FAIL_PATH_SET, "runtime source path is malformed")
        path = PurePosixPath(raw)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            _fail(FAIL_PATH_SET, "runtime source path must be canonical and relative")
        canonical = path.as_posix()
        if canonical != raw:
            _fail(FAIL_PATH_SET, "runtime source path is not canonical POSIX text")
        normalized.append(canonical)
    if len(normalized) != len(set(normalized)):
        _fail(FAIL_PATH_SET, "runtime source path set contains a duplicate")
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))


def _parse_tree_entry(payload: bytes, *, repository_path: str) -> tuple[str, str]:
    rows = tuple(row for row in payload.split(b"\x00") if row)
    if len(rows) != 1:
        _fail(FAIL_GIT_TREE, f"runtime source is absent or ambiguous: {repository_path}")
    try:
        metadata, raw_path = rows[0].split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii", "strict").split(" ")
        observed_path = raw_path.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_GIT_TREE, f"runtime source tree row is malformed: {repository_path}")
    if (
        observed_path != repository_path
        or kind != "blob"
        or mode not in {"100644", "100755"}
        or _SHA1_RE.fullmatch(object_id) is None
    ):
        _fail(FAIL_GIT_TREE, f"runtime source Git identity differs: {repository_path}")
    return mode, object_id


def _parse_index_entry(payload: bytes, *, repository_path: str) -> tuple[str, str]:
    rows = tuple(row for row in payload.split(b"\x00") if row)
    if len(rows) != 1:
        _fail(FAIL_INDEX, f"runtime source index row is absent or unmerged: {repository_path}")
    try:
        metadata, raw_path = rows[0].split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii", "strict").split(" ")
        observed_path = raw_path.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_INDEX, f"runtime source index row is malformed: {repository_path}")
    if (
        observed_path != repository_path
        or stage != "0"
        or mode not in {"100644", "100755"}
        or _SHA1_RE.fullmatch(object_id) is None
    ):
        _fail(FAIL_INDEX, f"runtime source index identity differs: {repository_path}")
    return mode, object_id


def _open_component(
    parent_descriptor: int,
    component: str,
    *,
    directory: bool,
    repository_path: str,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        return os.open(component, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        code = FAIL_SYMLINK if exc.errno in {40} else FAIL_WORKTREE
        _fail(code, f"runtime source path cannot be opened safely: {repository_path}")


def _read_worktree_file(
    repository: Path,
    repository_path: str,
    *,
    expected_mode: str,
) -> tuple[bytes, int]:
    components = PurePosixPath(repository_path).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptors: list[int] = []
    try:
        root_descriptor = os.open(repository, directory_flags)
        descriptors.append(root_descriptor)
        current = root_descriptor
        for component in components[:-1]:
            current = _open_component(
                current,
                component,
                directory=True,
                repository_path=repository_path,
            )
            descriptors.append(current)
        file_descriptor = _open_component(
            current,
            components[-1],
            directory=False,
            repository_path=repository_path,
        )
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        exact_posix_mode = 0o755 if expected_mode == "100755" else 0o644
        if not stat.S_ISREG(before.st_mode):
            _fail(FAIL_WORKTREE, f"runtime source is not a regular file: {repository_path}")
        if before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != exact_posix_mode:
            _fail(FAIL_MODE, f"runtime source owner or mode differs: {repository_path}")
        if before.st_size < 1 or before.st_size > MAX_RUNTIME_SOURCE_FILE_BYTES:
            _fail(FAIL_WORKTREE, f"runtime source size is outside the bound: {repository_path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(file_descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(FAIL_WORKTREE, f"runtime source read was short: {repository_path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_descriptor, 1):
            _fail(FAIL_WORKTREE, f"runtime source grew during read: {repository_path}")
        after = os.fstat(file_descriptor)
        stable = (
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
        observed_after = (
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
        if stable != observed_after:
            _fail(FAIL_WORKTREE, f"runtime source changed during read: {repository_path}")
        namespace = os.stat(
            components[-1],
            dir_fd=current,
            follow_symlinks=False,
        )
        if (namespace.st_dev, namespace.st_ino) != (after.st_dev, after.st_ino):
            _fail(FAIL_WORKTREE, f"runtime source namespace changed: {repository_path}")
        return b"".join(chunks), exact_posix_mode
    except M3RuntimeSourcePreflightError:
        raise
    except OSError as exc:
        _fail(FAIL_WORKTREE, f"runtime source stable read failed: {type(exc).__name__}")
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _blob_sha1(payload: bytes) -> str:
    preimage = b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload
    return hashlib.sha1(preimage).hexdigest()


def _manifest_and_receipt(
    *,
    commit: str,
    commit_tree: str,
    rows: Sequence[Mapping[str, object]],
) -> RuntimeSourcePreflightResultV1:
    row_values = [dict(row) for row in rows]
    source_set_sha256 = hashlib.sha256(
        SOURCE_SET_DOMAIN + b"\x00" + canonical_json_v1(row_values)
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "expected_runtime_commit": commit,
        "expected_runtime_tree": commit_tree,
        "runtime_source_file_count": len(row_values),
        "runtime_source_files": row_values,
        "runtime_source_set_sha256": source_set_sha256,
    }
    manifest_sha256 = hashlib.sha256(canonical_json_v1(manifest)).hexdigest()
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "claim_level": CLAIM_LEVEL,
        "expected_runtime_commit": commit,
        "expected_runtime_tree": commit_tree,
        "head_commit": commit,
        "runtime_source_file_count": len(row_values),
        "runtime_source_set_sha256": source_set_sha256,
        "manifest_sha256": manifest_sha256,
        "git_index_matches_commit": True,
        "working_tree_matches_commit": True,
        "path_scoped_status_clean": True,
        "symlink_free": True,
        "exact_file_modes": True,
        "docker_invoked": False,
        "state_changed": False,
        "formal_m3_output_generated": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json_v1(receipt)).hexdigest()
    validate_runtime_source_preflight_v1(manifest, receipt)
    return RuntimeSourcePreflightResultV1(
        expected_runtime_commit=commit,
        manifest_fields=MappingProxyType(manifest),
        receipt_fields=MappingProxyType(receipt),
    )


def build_runtime_source_preflight_v1(
    expected_runtime_revision: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    runtime_paths: Sequence[str] = DEFAULT_M3_RUNTIME_SOURCE_PATHS,
) -> RuntimeSourcePreflightResultV1:
    """Verify one checked-out runtime source set against one local commit.

    The function is read-only with respect to Git, the working tree, M3 state,
    and Docker.  A successful result is deterministic for a given commit and
    path set; it deliberately contains no wall-clock timestamp or host path.
    """

    repository = _require_repository(repository_root)
    paths = _normalize_paths(runtime_paths)
    commit = resolve_runtime_commit_v1(repository, expected_runtime_revision)
    head = resolve_runtime_commit_v1(repository, "HEAD")
    if head != commit:
        _fail(FAIL_COMMIT, "checked-out HEAD is not the expected runtime commit")
    tree_payload = _git(
        repository,
        ("show", "-s", "--format=%T", commit),
        code=FAIL_COMMIT,
    )
    try:
        commit_tree = tree_payload.decode("ascii", "strict").strip()
    except UnicodeDecodeError:
        _fail(FAIL_COMMIT, "runtime commit tree ID is not ASCII")
    if _SHA1_RE.fullmatch(commit_tree) is None:
        _fail(FAIL_COMMIT, "runtime commit tree ID is not a full SHA-1")

    rows: list[Mapping[str, object]] = []
    for repository_path in paths:
        tree_mode, tree_object = _parse_tree_entry(
            _git(
                repository,
                ("ls-tree", "-z", "--full-tree", commit, "--", repository_path),
                code=FAIL_GIT_TREE,
            ),
            repository_path=repository_path,
        )
        index_mode, index_object = _parse_index_entry(
            _git(
                repository,
                ("ls-files", "--stage", "-z", "--", repository_path),
                code=FAIL_INDEX,
            ),
            repository_path=repository_path,
        )
        if (index_mode, index_object) != (tree_mode, tree_object):
            _fail(FAIL_INDEX, f"runtime source index differs from commit: {repository_path}")
        committed_payload = _git(
            repository,
            ("cat-file", "blob", tree_object),
            code=FAIL_GIT_TREE,
        )
        if not committed_payload or _blob_sha1(committed_payload) != tree_object:
            _fail(FAIL_GIT_TREE, f"runtime source commit blob differs: {repository_path}")
        working_payload, checkout_mode = _read_worktree_file(
            repository,
            repository_path,
            expected_mode=tree_mode,
        )
        if working_payload != committed_payload:
            _fail(FAIL_BYTES, f"runtime source bytes differ from commit: {repository_path}")
        rows.append(
            MappingProxyType(
                {
                    "repository_path": repository_path,
                    "git_mode": tree_mode,
                    "checkout_mode_octal": f"{checkout_mode:04o}",
                    "git_blob_sha1": tree_object,
                    "byte_length": len(committed_payload),
                    "sha256": hashlib.sha256(committed_payload).hexdigest(),
                }
            )
        )

    status = _git(
        repository,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *paths),
        code=FAIL_DIRTY,
    )
    if status:
        _fail(FAIL_DIRTY, "runtime source path set has staged, unstaged, or untracked changes")
    return _manifest_and_receipt(commit=commit, commit_tree=commit_tree, rows=rows)


def validate_runtime_source_preflight_v1(
    manifest: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    """Validate the deterministic framing and cross-links of one result."""

    manifest_fields = {
        "schema",
        "expected_runtime_commit",
        "expected_runtime_tree",
        "runtime_source_file_count",
        "runtime_source_files",
        "runtime_source_set_sha256",
    }
    receipt_fields = {
        "schema",
        "artifact_kind",
        "claim_level",
        "expected_runtime_commit",
        "expected_runtime_tree",
        "head_commit",
        "runtime_source_file_count",
        "runtime_source_set_sha256",
        "manifest_sha256",
        "git_index_matches_commit",
        "working_tree_matches_commit",
        "path_scoped_status_clean",
        "symlink_free",
        "exact_file_modes",
        "docker_invoked",
        "state_changed",
        "formal_m3_output_generated",
        "receipt_sha256",
    }
    if set(manifest) != manifest_fields or set(receipt) != receipt_fields:
        _fail(FAIL_RECEIPT, "runtime-source manifest or receipt field set differs")
    rows = manifest.get("runtime_source_files")
    if type(rows) is not list or not rows:
        _fail(FAIL_RECEIPT, "runtime-source manifest rows are absent")
    row_fields = {
        "repository_path",
        "git_mode",
        "checkout_mode_octal",
        "git_blob_sha1",
        "byte_length",
        "sha256",
    }
    if any(type(row) is not dict or set(row) != row_fields for row in rows):
        _fail(FAIL_RECEIPT, "runtime-source manifest row shape differs")
    if any(type(row["repository_path"]) is not str for row in rows):
        _fail(FAIL_RECEIPT, "runtime-source manifest path type differs")
    paths = [row["repository_path"] for row in rows]
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")) or len(paths) != len(set(paths)):
        _fail(FAIL_RECEIPT, "runtime-source manifest row order differs")
    if any(
        type(row["repository_path"]) is not str
        or row["git_mode"] not in {"100644", "100755"}
        or row["checkout_mode_octal"]
        != ("0755" if row["git_mode"] == "100755" else "0644")
        or type(row["git_blob_sha1"]) is not str
        or _SHA1_RE.fullmatch(row["git_blob_sha1"]) is None
        or type(row["byte_length"]) is not int
        or not 1 <= row["byte_length"] <= MAX_RUNTIME_SOURCE_FILE_BYTES
        or type(row["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
        for row in rows
    ):
        _fail(FAIL_RECEIPT, "runtime-source manifest row value differs")
    commit = manifest.get("expected_runtime_commit")
    tree = manifest.get("expected_runtime_tree")
    source_set = manifest.get("runtime_source_set_sha256")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or type(commit) is not str
        or _SHA1_RE.fullmatch(commit) is None
        or type(tree) is not str
        or _SHA1_RE.fullmatch(tree) is None
        or manifest.get("runtime_source_file_count") != len(rows)
        or type(source_set) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_set) is None
    ):
        _fail(FAIL_RECEIPT, "runtime-source manifest identity differs")
    expected_source_set = hashlib.sha256(
        SOURCE_SET_DOMAIN + b"\x00" + canonical_json_v1(rows)
    ).hexdigest()
    if source_set != expected_source_set:
        _fail(FAIL_RECEIPT, "runtime-source set digest differs")

    receipt_body = dict(receipt)
    claimed_receipt_hash = receipt_body.pop("receipt_sha256", None)
    required_true = (
        "git_index_matches_commit",
        "working_tree_matches_commit",
        "path_scoped_status_clean",
        "symlink_free",
        "exact_file_modes",
    )
    required_false = (
        "docker_invoked",
        "state_changed",
        "formal_m3_output_generated",
    )
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != ARTIFACT_KIND
        or receipt.get("claim_level") != CLAIM_LEVEL
        or receipt.get("expected_runtime_commit") != commit
        or receipt.get("expected_runtime_tree") != tree
        or receipt.get("head_commit") != commit
        or receipt.get("runtime_source_file_count") != len(rows)
        or receipt.get("runtime_source_set_sha256") != source_set
        or receipt.get("manifest_sha256")
        != hashlib.sha256(canonical_json_v1(dict(manifest))).hexdigest()
        or any(receipt.get(name) is not True for name in required_true)
        or any(receipt.get(name) is not False for name in required_false)
        or type(claimed_receipt_hash) is not str
        or claimed_receipt_hash != hashlib.sha256(canonical_json_v1(receipt_body)).hexdigest()
    ):
        _fail(FAIL_RECEIPT, "runtime-source receipt identity or cross-link differs")


__all__ = [
    "ARTIFACT_KIND",
    "CLAIM_LEVEL",
    "DEFAULT_M3_RUNTIME_SOURCE_PATHS",
    "FAIL_BYTES",
    "FAIL_COMMIT",
    "FAIL_DIRTY",
    "FAIL_GIT_TREE",
    "FAIL_INDEX",
    "FAIL_MODE",
    "FAIL_PATH_SET",
    "FAIL_RECEIPT",
    "FAIL_REPOSITORY",
    "FAIL_SYMLINK",
    "FAIL_WORKTREE",
    "MANIFEST_SCHEMA",
    "M3RuntimeSourcePreflightError",
    "RECEIPT_SCHEMA",
    "RuntimeSourcePreflightResultV1",
    "build_runtime_source_preflight_v1",
    "canonical_json_v1",
    "resolve_runtime_commit_v1",
    "validate_runtime_source_preflight_v1",
]
