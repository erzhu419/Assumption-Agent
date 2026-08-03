"""Offline, staged-byte audit for the Phase-3A publication Commit B.

The audit has two deliberately separate phases:

* ``run_commit_b_publication_actor_audit_v1`` snapshots the exact Git-index
  bytes of the allowlisted A-to-B candidate, excluding this audit's own
  receipt path, and asks a no-key purpose-4 Docker actor to lint those bytes.
* ``finalize_staged_commit_b_publication_v1`` runs after that canonical receipt
  has itself been staged.  It replays the original candidate from the index,
  validates the staged receipt, and checks that the final staged path set is
  exactly ``candidate + receipt``.

Both phases are diagnostic publication controls.  They create no formal gate,
signature, seed, key, marker, formal root, or M3 state transition.  The Docker
actor is an owner-accepted technically isolated role, not an independent
person or organization.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Final, Mapping, NoReturn, Sequence
import zlib

from .phase3_container_actor_runtime_v1 import TECHNICAL_ACTOR_DISCLOSURE_V1
from .phase3_local_runtime_v1 import (
    DEFAULT_DOCKER_EXECUTABLE,
    DEFAULT_LINUX_LOCAL_RUNTIME_PARENT,
    LOCAL_DOCKER_HOST,
    LinuxLocalTemporaryDirectoryV1,
    Phase3LocalRuntimeError,
    build_local_docker_daemon_identity_receipt_v1,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
)
from .phase3_m25_external_v1 import (
    ExternalGenesisPreflightError,
    assert_public_payload_contains_no_secret_fields,
    validate_commit_b_changed_paths,
)
from .phase3_m25_secret_absence_v1 import (
    FORBIDDEN_EXACT_BASENAMES,
    FORBIDDEN_EXTENSIONS,
    FORBIDDEN_JSON_SECRET_KEYS,
    PRIVATE_KEY_MAGIC_HEADERS,
    _private_key_magic_hit as _secret_private_key_magic_hit_v1,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase3_container_actor_profile_v1.json"
PROFILE_REPOSITORY_PATH: Final = (
    "Hegel Machine/config/phase3_container_actor_profile_v1.json"
)
SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
WORKER_PATH: Final = PROJECT_ROOT / "tools/phase3_m25_commit_b_publication_audit_worker_v1.py"
CLI_PATH: Final = PROJECT_ROOT / "tools/phase3_m25_commit_b_publication_audit_v1.py"
PROBE_PATH: Final = PROJECT_ROOT / "tools/phase3_container_actor_probe_v1.py"

SCHEMA: Final = "hegel-phase3-m25-commit-b-publication-audit/1"
MANIFEST_SCHEMA: Final = "hegel-phase3-m25-commit-b-index-manifest/1"
REQUEST_SCHEMA: Final = "hegel-phase3-m25-commit-b-publication-audit-request/1"
RECEIPT_SCHEMA: Final = "hegel-phase3-m25-commit-b-publication-audit-receipt/1"
FINAL_STATUS_SCHEMA: Final = "hegel-phase3-m25-commit-b-publication-final-host-audit/1"
RUNTIME_SCHEMA: Final = "hegel-phase3-m25-commit-b-audit-runtime/1"
POLICY_ID: Final = "hegel-phase3-m25-commit-b-publication-policy-v1"
INVENTORY_DOMAIN: Final = b"HEGEL/PHASE3/M25/COMMIT_B/INDEX_INVENTORY/V1\x00"
RUNTIME_DOMAIN: Final = b"HEGEL/PHASE3/M25/COMMIT_B/AUDIT_RUNTIME/V1\x00"

ALLOWED_PUBLIC_PREFIXES: Final = (
    "Hegel Machine/artifacts/phase3_m25_external",
    "Hegel Machine/docs/phase3_m25_external_status.md",
)
EXECUTABLE_PREFIXES: Final = (
    "Hegel Machine/src",
    "Hegel Machine/rust",
    "Hegel Machine/tests",
    "Hegel Machine/tools",
)
AUDIT_RECEIPT_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/"
    "phase3_m25_commit_b_publication_audit_receipt_v1.json"
)
FORMAL_PUBLIC_PARENT_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/formal_genesis_v2"
)
FORMAL_EVIDENCE_REPOSITORY_PATH: Final = (
    FORMAL_PUBLIC_PARENT_REPOSITORY_PATH
    + "/phase3_m25_formal_gate_evidence_v1.json"
)
FORMAL_PROMOTION_REPOSITORY_PATH: Final = (
    FORMAL_PUBLIC_PARENT_REPOSITORY_PATH
    + "/phase3_m25_gate_promotion_v1.json"
)
FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH: Final = (
    FORMAL_PROMOTION_REPOSITORY_PATH + ".publication-receipt.json"
)
ACTOR_QUALIFICATION_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_actor_qualification_v1.json"
)
ERRATA_QUALIFICATION_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_errata_qualification_v1.json"
)
M3_IMPLEMENTATION_QUALIFICATION_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m3_implementation_qualification_v1.json"
)
BRIDGE_QUALIFICATION_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/"
    "phase3_m25_bridge_dag_rust_binary_qualification_v1.json"
)
LIVE_PROTOCOL_QUALIFICATION_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/"
    "phase3_m25_live_actor_protocol_qualification_v1.json"
)
PRE_GENESIS_EXECUTION_STATUS_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/"
    "phase3_m25_pre_genesis_execution_status_v1.json"
)
PRE_GENESIS_READINESS_REPOSITORY_PATH: Final = (
    "Hegel Machine/artifacts/phase3_m25_external/phase3_m25_pre_genesis_readiness_v1.json"
)
EXTERNAL_STATUS_REPOSITORY_PATH: Final = (
    "Hegel Machine/docs/phase3_m25_external_status.md"
)
FORMAL_PUBLIC_PATHS: Final = (
    FORMAL_EVIDENCE_REPOSITORY_PATH,
    FORMAL_PROMOTION_REPOSITORY_PATH,
    FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH,
)
CANONICAL_JSON_REQUIRED_PATHS: Final = frozenset(
    (*FORMAL_PUBLIC_PATHS, AUDIT_RECEIPT_REPOSITORY_PATH)
)
PUBLICATION_ROLE_REGISTRY: Final = {
    ACTOR_QUALIFICATION_REPOSITORY_PATH: "ACTOR_ELIGIBILITY",
    ERRATA_QUALIFICATION_REPOSITORY_PATH: "ERRATA_QUALIFICATION",
    M3_IMPLEMENTATION_QUALIFICATION_REPOSITORY_PATH: "M3_IMPLEMENTATION_QUALIFICATION",
    BRIDGE_QUALIFICATION_REPOSITORY_PATH: "BRIDGE_BINARY_QUALIFICATION",
    LIVE_PROTOCOL_QUALIFICATION_REPOSITORY_PATH: "LIVE_ACTOR_PROTOCOL_QUALIFICATION",
    PRE_GENESIS_EXECUTION_STATUS_REPOSITORY_PATH: "PRE_GENESIS_EXECUTION_STATUS",
    PRE_GENESIS_READINESS_REPOSITORY_PATH: "PRE_GENESIS_READINESS",
    FORMAL_EVIDENCE_REPOSITORY_PATH: "FORMAL_GATE_EVIDENCE",
    FORMAL_PROMOTION_REPOSITORY_PATH: "FORMAL_GATE_PROMOTION",
    FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH: "FORMAL_TRANSACTION_PUBLICATION_RECEIPT",
    EXTERNAL_STATUS_REPOSITORY_PATH: "EXTERNAL_STATUS_DOCUMENT",
    AUDIT_RECEIPT_REPOSITORY_PATH: "COMMIT_B_PREPARE_AUDIT_RECEIPT",
}
if len(PUBLICATION_ROLE_REGISTRY) != len(set(PUBLICATION_ROLE_REGISTRY.values())):
    raise RuntimeError("Commit-B publication role registry must be one-to-one")

# The live-protocol qualification source set originally combines every direct
# ``src/hegel_machine/*.py`` file with this frozen non-package registry.  A
# post-commit verifier must discover the package members from the selected
# Commit-A tree, not from the verifier checkout's ``Path.glob`` result.
LIVE_PROTOCOL_FIXED_NONPACKAGE_SOURCE_PATHS_V1: Final = (
    "Hegel Machine/config/phase3_container_actor_profile_v1.json",
    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
    "Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json",
    "Hegel Machine/tools/phase3_m25_formal_actor_worker_v1.py",
    "Hegel Machine/tools/phase3_m25_python_bridge_actor_worker_v1.py",
    "Hegel Machine/tools/phase3_m25_parent_auditor_actor_worker_v1.py",
    "Hegel Machine/tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py",
    "Hegel Machine/tools/phase3_m25_formal_rust_actor_worker_v1.sh",
    "Hegel Machine/tools/phase3_split_partition_calculator_fd3_v1.py",
    "Hegel Machine/tools/phase3_split_partition_calculator_fd3_v1.rs",
    "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
    "Hegel Machine/tools/phase3_container_actor_probe_v1.rs",
    "Hegel Machine/tools/phase3_m25_actor_operation_probe_v1.py",
    "Hegel Machine/tools/phase3_m25_bridge_dag_binary_qualification_v1.py",
    "Hegel Machine/tools/phase3_m25_seed_custody_verifier_v1.py",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.toml",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.lock",
    "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
    "Hegel Machine/rust/formal_bridge_m25/src/main.rs",
    "Hegel Machine/rust/m25_bridge_dag_replay/Cargo.toml",
    "Hegel Machine/rust/m25_bridge_dag_replay/Cargo.lock",
    "Hegel Machine/rust/m25_bridge_dag_replay/src/lib.rs",
    "Hegel Machine/rust/m25_bridge_dag_replay/src/main.rs",
    "Hegel Machine/tools/phase3_m25_actor_protocol_qualification_v1.py",
    "Hegel Machine/tools/phase3_m25_protocol_qualification_finalize_worker_v1.sh",
    "Hegel Machine/tests/test_phase3_m25_actor_protocol_qualification_v1.py",
    (
        "Hegel Machine/docs/"
        "Hegel_Machine_Phase3A_M25_Live_Actor_Protocol_Qualification_v1.md"
    ),
)
LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1: Final = "Hegel Machine/src/hegel_machine"
LIVE_PROTOCOL_REQUIRED_PACKAGE_PATHS_V1: Final = frozenset(
    {
        LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1
        + "/phase3_m25_actor_protocol_qualification_v1.py",
        LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1
        + "/phase3_m25_formal_container_executor_v1.py",
    }
)

PYTHON_IMAGE_KEY: Final = "policy_auditor"
MAX_FILE_BYTES: Final = 32 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 128 * 1024 * 1024
MAX_ACTOR_OUTPUT_BYTES: Final = 4 * 1024 * 1024
_LOWER_SHA1 = re.compile(r"[0-9a-f]{40}")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFIXED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_RAW_PATH_PATTERN = re.compile(
    rb"(?i)(?:/home/[a-z0-9._-]+/|/users/[a-z0-9._ -]+/|"
    rb"/mnt/[a-z]/users/[a-z0-9._ -]+/|[a-z]:\\\\users\\\\|"
    rb"\\\\\\\\wsl(?:\.localhost)?\\\\)"
)

FAIL_GIT_INDEX = "FAIL_COMMIT_B_AUDIT_GIT_INDEX"
FAIL_PATH_POLICY = "FAIL_COMMIT_B_AUDIT_PATH_POLICY"
FAIL_FILE_POLICY = "FAIL_COMMIT_B_AUDIT_FILE_POLICY"
FAIL_JSON_POLICY = "FAIL_COMMIT_B_AUDIT_JSON_POLICY"
FAIL_SECRET_POLICY = "FAIL_COMMIT_B_AUDIT_SECRET_POLICY"
FAIL_RUNTIME_BASIS = "FAIL_COMMIT_B_AUDIT_RUNTIME_BASIS"
FAIL_ACTOR_POLICY = "FAIL_COMMIT_B_AUDIT_ACTOR_POLICY"
FAIL_ACTOR_RESPONSE = "FAIL_COMMIT_B_AUDIT_ACTOR_RESPONSE"
FAIL_FORMAL_REPLAY = "FAIL_COMMIT_B_AUDIT_FORMAL_REPLAY"
FAIL_FINAL_STAGED_SET = "FAIL_COMMIT_B_AUDIT_FINAL_STAGED_SET"
FAIL_LOCAL_RUNTIME = "FAIL_COMMIT_B_AUDIT_LOCAL_RUNTIME"


class CommitBPublicationAuditError(RuntimeError):
    """Stable fail-closed error at the Commit-B publication boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise CommitBPublicationAuditError(code, detail)


def canonical_json_v1(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _require_commit(value: str) -> str:
    if type(value) is not str or _LOWER_SHA1.fullmatch(value) is None:
        _fail(FAIL_GIT_INDEX, "basis commit must be a lowercase Git SHA-1")
    return value


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 120,
) -> bytes:
    if not arguments or any(
        type(item) is not str or not item or "\0" in item for item in arguments
    ):
        _fail(FAIL_GIT_INDEX, "Git argument vector is malformed")
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repository,
            env=_git_environment(),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_GIT_INDEX, f"Git read could not complete: {type(exc).__name__}")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "backslashreplace")[-600:].strip()
        _fail(FAIL_GIT_INDEX, f"Git read exited {completed.returncode}: {detail}")
    return completed.stdout


def _raw_commit_tree_and_parent_v1(
    repository: Path, commit_sha1: str
) -> tuple[str, str]:
    """Parse identity-bearing headers from the raw commit object itself.

    Revision walks may honor local graft metadata.  Publication verification
    therefore derives both the tree and sole parent only from ``cat-file``'s
    raw commit bytes.
    """

    commit = _require_commit(commit_sha1)
    raw = _git(repository, ("cat-file", "commit", commit))
    if b"\0" in raw or b"\r" in raw or b"\n\n" not in raw:
        _fail(FAIL_FINAL_STAGED_SET, "publication commit object framing is malformed")
    header, _message = raw.split(b"\n\n", 1)
    rows: list[tuple[bytes, bytes]] = []
    for line in header.split(b"\n"):
        if line.startswith(b" "):
            if not rows:
                _fail(FAIL_FINAL_STAGED_SET, "publication commit has orphan continuation")
            continue
        try:
            key, value = line.split(b" ", 1)
        except ValueError:
            _fail(FAIL_FINAL_STAGED_SET, "publication commit header is malformed")
        if re.fullmatch(rb"[a-z][a-z0-9-]*", key) is None or not value:
            _fail(FAIL_FINAL_STAGED_SET, "publication commit header key/value is malformed")
        rows.append((key, value))
    trees = [value for key, value in rows if key == b"tree"]
    parents = [value for key, value in rows if key == b"parent"]
    if len(trees) != 1 or len(parents) != 1:
        _fail(FAIL_FINAL_STAGED_SET, "publication commit must encode one tree and one parent")
    try:
        tree = trees[0].decode("ascii", "strict")
        parent = parents[0].decode("ascii", "strict")
    except UnicodeDecodeError:
        _fail(FAIL_FINAL_STAGED_SET, "publication commit tree/parent is not ASCII")
    if _LOWER_SHA1.fullmatch(tree) is None or _LOWER_SHA1.fullmatch(parent) is None:
        _fail(FAIL_FINAL_STAGED_SET, "publication commit tree/parent identity is malformed")
    _git(repository, ("cat-file", "-e", f"{tree}^{{tree}}"))
    return tree, parent


def _basis_blob_v1(repository: Path, basis_commit: str, path: str) -> bytes:
    """Read one regular blob from the supplied repository's basis tree."""

    commit = _require_commit(basis_commit)
    normalized = _normalize_repository_path(path.encode("utf-8"))
    raw = _git(repository, ("ls-tree", "-z", commit, "--", normalized))
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        _fail(FAIL_FORMAL_REPLAY, f"basis blob {normalized!r} is absent or ambiguous")
    try:
        metadata, raw_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii", "strict").split(" ")
        observed_path = raw_path.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_FORMAL_REPLAY, f"basis blob {normalized!r} tree entry is malformed")
    if (
        observed_path != normalized
        or mode not in {"100644", "100755"}
        or object_type != "blob"
        or _LOWER_SHA1.fullmatch(object_id) is None
    ):
        _fail(FAIL_FORMAL_REPLAY, f"basis blob {normalized!r} is not a regular Git blob")
    return _git(repository, ("cat-file", "blob", object_id))


def _prefixed_sha256_v1(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_prefixed_sha256_v1(value: object, *, label: str) -> str:
    if type(value) is not str or _PREFIXED_SHA256.fullmatch(value) is None:
        _fail(FAIL_FORMAL_REPLAY, f"{label} is not an exact prefixed SHA-256")
    return value


def _repository(repository: Path) -> Path:
    try:
        requested = Path(os.path.abspath(os.fspath(repository)))
        metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_GIT_INDEX, f"repository cannot be resolved: {exc}")
    if requested != resolved or stat.S_ISLNK(metadata.st_mode) or not resolved.is_dir():
        _fail(FAIL_GIT_INDEX, "repository must be a real absolute directory")
    top = _git(resolved, ("rev-parse", "--show-toplevel")).decode("utf-8", "strict").strip()
    if Path(top).resolve(strict=True) != resolved:
        _fail(FAIL_GIT_INDEX, "repository is not the Git toplevel")
    return resolved


def _normalize_repository_path(raw: bytes) -> str:
    try:
        value = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        _fail(FAIL_PATH_POLICY, "Commit-B paths must be valid UTF-8")
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail(FAIL_PATH_POLICY, "Commit-B path is not canonical repository-relative UTF-8")
    return value


def _raw_staged_rows(repository: Path, basis_commit: str) -> tuple[tuple[str, str], ...]:
    output = _git(
        repository,
        (
            "diff",
            "--cached",
            "--raw",
            "-z",
            "--no-renames",
            "--no-abbrev",
            basis_commit,
            "--",
        ),
    )
    tokens = output.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    if len(tokens) % 2:
        _fail(FAIL_GIT_INDEX, "Git staged diff framing is malformed")
    rows: list[tuple[str, str]] = []
    for offset in range(0, len(tokens), 2):
        header, raw_path = tokens[offset : offset + 2]
        fields = header.split(b" ")
        if len(fields) != 5 or not fields[0].startswith(b":"):
            _fail(FAIL_GIT_INDEX, "Git staged raw row is malformed")
        status_raw = fields[4]
        try:
            status_text = status_raw.decode("ascii", "strict")
            new_mode = fields[1].decode("ascii", "strict")
        except UnicodeDecodeError:
            _fail(FAIL_GIT_INDEX, "Git staged metadata is not ASCII")
        if status_text not in {"A", "M"}:
            _fail(FAIL_PATH_POLICY, f"Commit-B status {status_text!r} is forbidden")
        path = _normalize_repository_path(raw_path)
        rows.append((path, new_mode))
    if len({path for path, _mode in rows}) != len(rows):
        _fail(FAIL_GIT_INDEX, "Git staged diff contains duplicate paths")
    return tuple(sorted(rows))


def _raw_commit_rows(
    repository: Path, basis_commit: str, publication_commit: str
) -> tuple[tuple[str, str], ...]:
    """Return exact A-to-B tree changes with rename detection disabled."""

    output = _git(
        repository,
        (
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--no-renames",
            "--no-abbrev",
            basis_commit,
            publication_commit,
            "--",
        ),
    )
    tokens = output.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    if len(tokens) % 2:
        _fail(FAIL_GIT_INDEX, "Git committed diff framing is malformed")
    rows: list[tuple[str, str]] = []
    for offset in range(0, len(tokens), 2):
        header, raw_path = tokens[offset : offset + 2]
        fields = header.split(b" ")
        if len(fields) != 5 or not fields[0].startswith(b":"):
            _fail(FAIL_GIT_INDEX, "Git committed raw row is malformed")
        try:
            new_mode = fields[1].decode("ascii", "strict")
            status_text = fields[4].decode("ascii", "strict")
        except UnicodeDecodeError:
            _fail(FAIL_GIT_INDEX, "Git committed metadata is not ASCII")
        if status_text not in {"A", "M"}:
            _fail(FAIL_PATH_POLICY, f"Commit-B status {status_text!r} is forbidden")
        rows.append((_normalize_repository_path(raw_path), new_mode))
    if len({path for path, _mode in rows}) != len(rows):
        _fail(FAIL_GIT_INDEX, "Git committed diff contains duplicate paths")
    return tuple(sorted(rows))


def _index_entry(repository: Path, path: str) -> tuple[str, str, bytes]:
    output = _git(repository, ("ls-files", "--stage", "-z", "--", path))
    records = [record for record in output.split(b"\0") if record]
    if len(records) != 1:
        _fail(FAIL_GIT_INDEX, f"index path {path!r} is absent or unmerged")
    try:
        metadata, raw_path = records[0].split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii", "strict").split(" ")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_GIT_INDEX, f"index entry {path!r} is malformed")
    if raw_path.decode("utf-8", "strict") != path or stage != "0":
        _fail(FAIL_GIT_INDEX, f"index entry {path!r} has a nonzero stage")
    if mode != "100644" or _LOWER_SHA1.fullmatch(object_id) is None:
        _fail(FAIL_FILE_POLICY, f"index entry {path!r} is executable, symlinked, or non-blob")
    payload = _git(repository, ("cat-file", "blob", object_id))
    if len(payload) > MAX_FILE_BYTES:
        _fail(FAIL_FILE_POLICY, f"index entry {path!r} exceeds the public size bound")
    return mode, object_id, payload


def _open_anchored_parent_v1(repository: Path, path: str) -> tuple[int, str]:
    """Open a repository-relative parent without following any symlink component."""

    normalized = _normalize_repository_path(path.encode("utf-8"))
    parts = PurePosixPath(normalized).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        current = os.open(repository, directory_flags)
    except OSError as exc:
        _fail(FAIL_FILE_POLICY, f"repository anchor for {path!r} is unavailable: {exc}")
    try:
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            _fail(FAIL_FILE_POLICY, "repository anchor is not a directory")
        for component in parts[:-1]:
            try:
                following = os.open(component, directory_flags, dir_fd=current)
            except OSError as exc:
                _fail(
                    FAIL_FILE_POLICY,
                    f"audited worktree parent for {path!r} is unavailable: {exc}",
                )
            os.close(current)
            current = following
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _worktree_regular_bytes(repository: Path, path: str) -> bytes:
    parent_descriptor, basename = _open_anchored_parent_v1(repository, path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        try:
            descriptor = os.open(basename, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            _fail(FAIL_FILE_POLICY, f"audited worktree path {path!r} is unavailable: {exc}")
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o111:
                _fail(FAIL_FILE_POLICY, f"audited worktree path {path!r} is not non-executable regular")
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(descriptor, min(1 << 20, MAX_FILE_BYTES + 1 - total))
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if total > MAX_FILE_BYTES:
                    _fail(FAIL_FILE_POLICY, f"audited worktree path {path!r} is oversized")
            after = os.fstat(descriptor)
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                _fail(FAIL_FILE_POLICY, f"audited worktree path {path!r} changed while read")
        finally:
            os.close(descriptor)

        replay_parent, replay_basename = _open_anchored_parent_v1(repository, path)
        try:
            if (
                os.fstat(parent_descriptor).st_dev,
                os.fstat(parent_descriptor).st_ino,
            ) != (os.fstat(replay_parent).st_dev, os.fstat(replay_parent).st_ino):
                _fail(FAIL_FILE_POLICY, f"audited worktree parent for {path!r} changed while read")
            try:
                replay = os.open(replay_basename, flags, dir_fd=replay_parent)
            except OSError as exc:
                _fail(FAIL_FILE_POLICY, f"audited worktree path {path!r} changed while read: {exc}")
            try:
                replay_metadata = os.fstat(replay)
                if (replay_metadata.st_dev, replay_metadata.st_ino) != (
                    before.st_dev,
                    before.st_ino,
                ):
                    _fail(FAIL_FILE_POLICY, f"audited worktree path {path!r} changed while read")
            finally:
                os.close(replay)
        finally:
            os.close(replay_parent)
        return b"".join(chunks)
    finally:
        os.close(parent_descriptor)


def _read_repo_external_private_file_v1(repository: Path, path: Path) -> bytes:
    """Read a canonical owner-only external file through held nofollow dirfds."""

    raw_path = os.fspath(path)
    if (
        not os.path.isabs(raw_path)
        or os.path.normpath(raw_path) != raw_path
        or os.path.abspath(raw_path) != raw_path
    ):
        _fail(FAIL_FINAL_STAGED_SET, "repo-external receipt path is not canonical absolute")
    requested = Path(raw_path)
    try:
        requested.relative_to(repository)
    except ValueError:
        pass
    else:
        _fail(FAIL_FINAL_STAGED_SET, "finalize receipt must remain outside the repository")
    relative = "/".join(requested.parts[1:])
    parent, basename = _open_anchored_parent_v1(Path(requested.anchor), relative)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        try:
            descriptor = os.open(basename, flags, dir_fd=parent)
        except OSError as exc:
            _fail(FAIL_FINAL_STAGED_SET, f"repo-external finalize receipt cannot be opened: {exc}")
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.geteuid()
                or before.st_size > MAX_ACTOR_OUTPUT_BYTES
            ):
                _fail(FAIL_FINAL_STAGED_SET, "finalize receipt owner/mode/type/size policy differs")
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(
                    descriptor, min(1 << 20, MAX_ACTOR_OUTPUT_BYTES + 1 - total)
                )
                if not block:
                    break
                chunks.append(block)
                total += len(block)
                if total > MAX_ACTOR_OUTPUT_BYTES:
                    _fail(FAIL_FINAL_STAGED_SET, "finalize receipt exceeds size policy")
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                _fail(FAIL_FINAL_STAGED_SET, "finalize receipt changed while read")
        finally:
            os.close(descriptor)
        replay_parent, replay_basename = _open_anchored_parent_v1(
            Path(requested.anchor), relative
        )
        try:
            parent_metadata = os.fstat(parent)
            replay_parent_metadata = os.fstat(replay_parent)
            if (parent_metadata.st_dev, parent_metadata.st_ino) != (
                replay_parent_metadata.st_dev,
                replay_parent_metadata.st_ino,
            ):
                _fail(FAIL_FINAL_STAGED_SET, "finalize receipt parent changed while read")
            replay = os.open(replay_basename, flags, dir_fd=replay_parent)
            try:
                replay_metadata = os.fstat(replay)
                if (replay_metadata.st_dev, replay_metadata.st_ino) != (
                    before.st_dev,
                    before.st_ino,
                ):
                    _fail(FAIL_FINAL_STAGED_SET, "finalize receipt identity changed while read")
            finally:
                os.close(replay)
        finally:
            os.close(replay_parent)
        return b"".join(chunks)
    finally:
        os.close(parent)


class _Pairs(tuple):
    pass


def _strict_json(payload: bytes, *, path: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> _Pairs:
        keys = [key for key, _value in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON object key")
        return _Pairs(pairs)

    try:
        value = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonstandard JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        _fail(FAIL_JSON_POLICY, f"public JSON {path!r} is invalid: {exc}")
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, _Pairs):
        return {key: _plain_json(child) for key, child in value}
    if isinstance(value, list):
        return [_plain_json(child) for child in value]
    return value


def _normalize_json_key(key: str) -> str:
    with_breaks = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", with_breaks.casefold()).strip("_")


def _forbidden_json_key(value: object) -> str | None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, _Pairs):
            for key, child in current:
                normalized = _normalize_json_key(key)
                if normalized in FORBIDDEN_JSON_SECRET_KEYS:
                    return normalized
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return None


def _private_key_magic_hit_v1(payload: bytes) -> str | None:
    for header_id, header in sorted(PRIVATE_KEY_MAGIC_HEADERS.items()):
        if _secret_private_key_magic_hit_v1(payload, header_id, header):
            return header_id
    return None


def _path_filename_token(path: str) -> str | None:
    basename = PurePosixPath(path).name.casefold()
    if basename == ".env" or basename.startswith(".env."):
        return "ENV_FILENAME_RULE"
    if basename in FORBIDDEN_EXACT_BASENAMES:
        return "EXACT_BASENAME"
    if PurePosixPath(basename).suffix.casefold() in FORBIDDEN_EXTENSIONS:
        return "FORBIDDEN_EXTENSION"
    return None


def _public_file_lint_v1(
    path: str,
    payload: bytes,
    *,
    raw_path_tokens: Sequence[bytes],
) -> None:
    if _path_filename_token(path) is not None:
        _fail(FAIL_SECRET_POLICY, f"public path {path!r} has a secret-bearing filename")
    if not (
        path.endswith(".json")
        or path == "Hegel Machine/docs/phase3_m25_external_status.md"
    ):
        _fail(FAIL_FILE_POLICY, f"public path {path!r} has a non-public file type")
    header = _private_key_magic_hit_v1(payload)
    if header is not None:
        _fail(
            FAIL_SECRET_POLICY,
            f"public path {path!r} contains private-key magic or a complete block",
        )
    if _RAW_PATH_PATTERN.search(payload) is not None or any(
        token and token in payload for token in raw_path_tokens
    ):
        _fail(FAIL_SECRET_POLICY, f"public path {path!r} contains a raw host/author path")
    if path.endswith(".json"):
        decoded = _strict_json(payload, path=path)
        forbidden = _forbidden_json_key(decoded)
        if forbidden is not None:
            _fail(FAIL_SECRET_POLICY, f"public JSON {path!r} has forbidden field {forbidden!r}")
        plain = _plain_json(decoded)
        try:
            assert_public_payload_contains_no_secret_fields(plain)
        except ExternalGenesisPreflightError as exc:
            _fail(FAIL_SECRET_POLICY, f"public JSON {path!r} failed public-field policy: {exc.code}")
        if path in CANONICAL_JSON_REQUIRED_PATHS and canonical_json_v1(plain) != payload:
            _fail(FAIL_JSON_POLICY, f"public JSON {path!r} is not canonical ASCII JSON")


def _inventory_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256(INVENTORY_DOMAIN)
    for row in rows:
        digest.update(canonical_json_v1(dict(row)))
    return digest.hexdigest()


def _required_role_paths(*, exclude_receipt: bool) -> tuple[str, ...]:
    return tuple(
        sorted(
            path
            for path in PUBLICATION_ROLE_REGISTRY
            if not (exclude_receipt and path == AUDIT_RECEIPT_REPOSITORY_PATH)
        )
    )


def _validate_exact_role_set(paths: Sequence[str], *, exclude_receipt: bool) -> None:
    observed = tuple(sorted(paths))
    required = _required_role_paths(exclude_receipt=exclude_receipt)
    unknown = sorted(set(observed) - set(PUBLICATION_ROLE_REGISTRY))
    missing = sorted(set(required) - set(observed))
    duplicates = sorted(path for path in set(observed) if observed.count(path) != 1)
    forbidden_receipt = (
        exclude_receipt and AUDIT_RECEIPT_REPOSITORY_PATH in observed
    )
    if unknown or missing or duplicates or forbidden_receipt or observed != required:
        _fail(
            FAIL_PATH_POLICY,
            "Commit-B exact path-to-role registry/cardinality differs "
            f"(unknown={unknown}, missing={missing}, duplicates={duplicates})",
        )


def build_staged_candidate_manifest_v1(
    repository: Path,
    *,
    basis_commit: str,
    exclude_receipt: bool = True,
    permit_staged_receipt_for_replay: bool = False,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Read exact staged A-to-B bytes and return a public manifest plus bytes."""

    root = _repository(repository)
    commit = _require_commit(basis_commit)
    _git(root, ("cat-file", "-e", f"{commit}^{{commit}}"))
    head = _git(root, ("rev-parse", "HEAD")).decode("ascii", "strict").strip()
    if head != commit:
        _fail(FAIL_GIT_INDEX, "publication audit requires HEAD to equal the basis commit")
    raw_rows = _raw_staged_rows(root, commit)
    all_paths = tuple(path for path, _mode in raw_rows)
    receipt_present = AUDIT_RECEIPT_REPOSITORY_PATH in all_paths
    if exclude_receipt and receipt_present and not permit_staged_receipt_for_replay:
        _fail(FAIL_PATH_POLICY, "audit receipt self-output must not be staged during actor audit")
    candidate_rows = (
        tuple(
            (path, mode)
            for path, mode in raw_rows
            if path != AUDIT_RECEIPT_REPOSITORY_PATH
        )
        if exclude_receipt
        else raw_rows
    )
    if not candidate_rows:
        _fail(FAIL_PATH_POLICY, "Commit-B staged candidate is empty")
    _validate_exact_role_set(
        tuple(path for path, _mode in candidate_rows),
        exclude_receipt=exclude_receipt,
    )
    try:
        validate_commit_b_changed_paths(
            tuple(path for path, _mode in candidate_rows),
            allowed_public_prefixes=ALLOWED_PUBLIC_PREFIXES,
            executable_prefixes=EXECUTABLE_PREFIXES,
        )
    except ExternalGenesisPreflightError as exc:
        _fail(FAIL_PATH_POLICY, f"Commit-B allowlist rejected staged path: {exc.code}")
    raw_tokens = tuple(
        value.encode("utf-8")
        for value in {root.as_posix(), PROJECT_ROOT.as_posix(), Path.home().as_posix()}
        if len(value) >= 4
    )
    files: dict[str, bytes] = {}
    rows: list[dict[str, object]] = []
    total = 0
    for path, diff_mode in candidate_rows:
        mode, object_id, payload = _index_entry(root, path)
        if diff_mode != mode:
            _fail(FAIL_GIT_INDEX, f"staged diff/index mode differs for {path!r}")
        worktree = _worktree_regular_bytes(root, path)
        if worktree != payload:
            _fail(FAIL_FILE_POLICY, f"audited path {path!r} has unstaged byte drift")
        _public_file_lint_v1(path, payload, raw_path_tokens=raw_tokens)
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            _fail(FAIL_FILE_POLICY, "Commit-B candidate exceeds total public size bound")
        files[path] = payload
        rows.append(
            {
                "path": path,
                "role_id": PUBLICATION_ROLE_REGISTRY[path],
                "git_mode": mode,
                "index_blob_sha1": object_id,
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "policy_id": POLICY_ID,
        "audit_phase": (
            "PREPARE_EXCLUDING_RECEIPT"
            if exclude_receipt
            else "FINALIZE_INCLUDING_RECEIPT"
        ),
        "basis_commit_sha1": commit,
        "changed_path_scope": "EXACT_GIT_INDEX_DIFF_FROM_BASIS_COMMIT",
        "allowed_public_prefixes": list(ALLOWED_PUBLIC_PREFIXES),
        "executable_prefixes": list(EXECUTABLE_PREFIXES),
        "excluded_self_output_repository_path": AUDIT_RECEIPT_REPOSITORY_PATH,
        "excluded_self_output_present_in_candidate": not exclude_receipt,
        "path_role_registry": [
            {
                "path": path,
                "role_id": PUBLICATION_ROLE_REGISTRY[path],
                "required_cardinality": 1,
            }
            for path in _required_role_paths(exclude_receipt=exclude_receipt)
        ],
        "role_cardinalities": {
            PUBLICATION_ROLE_REGISTRY[path]: 1
            for path in _required_role_paths(exclude_receipt=exclude_receipt)
        },
        "candidate_files": rows,
        "candidate_file_count": len(rows),
        "candidate_total_byte_length": total,
        "candidate_inventory_sha256": _inventory_sha256(rows),
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_v1(manifest)).hexdigest()
    return manifest, files


def _manifest_body(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    claimed = body.pop("manifest_sha256", None)
    if type(claimed) is not str or hashlib.sha256(canonical_json_v1(body)).hexdigest() != claimed:
        _fail(FAIL_ACTOR_RESPONSE, "candidate manifest self-hash differs")
    return {**body, "manifest_sha256": claimed}


def _receipt_excluded_manifest_projection_v1(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Project a validated finalize manifest back to its prepare inventory."""

    manifest = _manifest_body(value)
    candidate_rows = manifest.get("candidate_files")
    if type(candidate_rows) is not list or any(type(row) is not dict for row in candidate_rows):
        _fail(FAIL_FINAL_STAGED_SET, "final actor manifest candidate rows are malformed")
    receipt_rows = [
        row for row in candidate_rows
        if row.get("path") == AUDIT_RECEIPT_REPOSITORY_PATH
    ]
    if len(receipt_rows) != 1:
        _fail(FAIL_FINAL_STAGED_SET, "final actor manifest has no unique receipt row")
    rows = [
        dict(row) for row in candidate_rows
        if row.get("path") != AUDIT_RECEIPT_REPOSITORY_PATH
    ]
    projected = dict(manifest)
    projected.pop("manifest_sha256", None)
    projected.update(
        {
            "audit_phase": "PREPARE_EXCLUDING_RECEIPT",
            "excluded_self_output_present_in_candidate": False,
            "path_role_registry": [
                {
                    "path": path,
                    "role_id": PUBLICATION_ROLE_REGISTRY[path],
                    "required_cardinality": 1,
                }
                for path in _required_role_paths(exclude_receipt=True)
            ],
            "role_cardinalities": {
                PUBLICATION_ROLE_REGISTRY[path]: 1
                for path in _required_role_paths(exclude_receipt=True)
            },
            "candidate_files": rows,
            "candidate_file_count": len(rows),
            "candidate_total_byte_length": sum(
                int(row.get("byte_length", -1)) for row in rows
            ),
            "candidate_inventory_sha256": _inventory_sha256(rows),
        }
    )
    projected["manifest_sha256"] = hashlib.sha256(
        canonical_json_v1(projected)
    ).hexdigest()
    return projected


def _write_candidate_snapshot(root: Path, manifest: Mapping[str, object], files: Mapping[str, bytes]) -> Path:
    candidate = root / "candidate"
    file_root = candidate / "files"
    file_root.mkdir(parents=True, mode=0o700)
    for path, payload in files.items():
        destination = file_root.joinpath(*PurePosixPath(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if os.write(descriptor, payload) != len(payload):
                _fail(FAIL_FILE_POLICY, "short candidate snapshot write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        destination.chmod(0o444)
    manifest_path = candidate / "manifest.json"
    manifest_path.write_bytes(canonical_json_v1(dict(manifest)))
    manifest_path.chmod(0o444)
    for directory in sorted(
        (item for item in candidate.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    candidate.chmod(0o555)
    return candidate


def _basis_file(
    repository: Path, basis_commit: str, source: Path, runtime_path: str
) -> tuple[dict[str, object], bytes]:
    try:
        relative = source.resolve(strict=True).relative_to(repository).as_posix()
    except (OSError, ValueError) as exc:
        _fail(FAIL_RUNTIME_BASIS, f"runtime source is outside repository: {exc}")
    tree = _git(repository, ("ls-tree", basis_commit, "--", relative)).rstrip(b"\n")
    try:
        metadata, tree_path = tree.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
    except (ValueError, UnicodeDecodeError):
        _fail(FAIL_RUNTIME_BASIS, f"runtime source {relative!r} is absent from basis")
    if tree_path.decode("utf-8", "strict") != relative or mode not in {"100644", "100755"} or object_type != "blob":
        _fail(FAIL_RUNTIME_BASIS, f"runtime source {relative!r} has invalid tree identity")
    committed = _git(repository, ("cat-file", "blob", object_id))
    if source.read_bytes() != committed:
        _fail(FAIL_RUNTIME_BASIS, f"runtime source {relative!r} differs from basis commit")
    return (
        {
            "runtime_path": runtime_path,
            "repository_path": relative,
            "basis_tree_mode": mode,
            "basis_tree_blob_sha1": object_id,
            "byte_length": len(committed),
            "sha256": hashlib.sha256(committed).hexdigest(),
        },
        committed,
    )


def _copy_actor_runtime(root: Path, *, repository: Path, basis_commit: str) -> dict[str, object]:
    runtime = root / "runtime"
    runtime.mkdir(mode=0o700)
    specs = (
        (Path(__file__), "control/phase3_m25_commit_b_publication_audit_v1.py"),
        (CLI_PATH, "control/phase3_m25_commit_b_publication_audit_cli_v1.py"),
        (WORKER_PATH, "worker.py"),
        (PROBE_PATH, "probe.py"),
        (PROFILE_PATH, "control/phase3_container_actor_profile_v1.json"),
        (SECCOMP_PATH, "control/phase3_internal_actor_seccomp_v1.json"),
    )
    rows: list[dict[str, object]] = []
    for source, destination_text in specs:
        row, payload = _basis_file(repository, basis_commit, source, destination_text)
        destination = runtime.joinpath(*PurePosixPath(destination_text).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(0o444)
        rows.append(row)
    rows.sort(key=lambda row: str(row["runtime_path"]))
    digest = hashlib.sha256(RUNTIME_DOMAIN)
    for row in rows:
        digest.update(canonical_json_v1(row))
    for directory in sorted(
        (item for item in runtime.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    runtime.chmod(0o555)
    return {
        "schema": RUNTIME_SCHEMA,
        "basis_commit_sha1": basis_commit,
        "files": rows,
        "file_count": len(rows),
        "runtime_inventory_sha256": digest.hexdigest(),
    }


def _load_image_ref(repository: Path, basis_commit: str) -> str:
    """Load the purpose-4 image only from the caller's Commit-A profile blob."""

    try:
        payload = _basis_blob_v1(repository, basis_commit, PROFILE_REPOSITORY_PATH)
        profile = _plain_json(_strict_json(payload, path=PROFILE_REPOSITORY_PATH))
        if type(profile) is not dict or type(profile.get("images")) is not dict:
            raise TypeError("profile/images is not an object")
        value = profile["images"][PYTHON_IMAGE_KEY]
    except (CommitBPublicationAuditError, KeyError, TypeError) as exc:
        _fail(FAIL_ACTOR_POLICY, f"actor profile image is unavailable: {exc}")
    if type(value) is not str or re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", value) is None:
        _fail(FAIL_ACTOR_POLICY, "purpose-4 Python image is not digest pinned")
    return value


def build_actor_request_v1(
    *,
    manifest: Mapping[str, object],
    runtime_inventory: Mapping[str, object],
    basis_commit: str,
    actor_image_ref: str,
    raw_path_tokens: Sequence[str],
) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "purpose_id": 4,
        "basis_commit_sha1": _require_commit(basis_commit),
        "actor_image_ref": actor_image_ref,
        "audit_phase": manifest.get("audit_phase"),
        "candidate_manifest": dict(manifest),
        "runtime_inventory": dict(runtime_inventory),
        "private_forbidden_raw_path_tokens": list(raw_path_tokens),
        "signature_generation_requested": False,
        "key_seed_or_marker_access_requested": False,
        "formal_gate_or_m3_transition_requested": False,
    }
    request["request_sha256"] = hashlib.sha256(canonical_json_v1(request)).hexdigest()
    return request


def publication_actor_container_command_v1(
    *, candidate: Path, runtime: Path, request_path: Path, image_ref: str
) -> tuple[str, ...]:
    seccomp = runtime / "control/phase3_internal_actor_seccomp_v1.json"
    return (
        DEFAULT_DOCKER_EXECUTABLE.as_posix(),
        f"--host={LOCAL_DOCKER_HOST}",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        f"--security-opt=seccomp={seccomp.resolve(strict=True)}",
        "--user=65534:65534",
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--ulimit=nofile=64:64",
        "--ipc=private",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
        f"--mount=type=bind,src={candidate.resolve()},dst=/candidate,readonly,bind-propagation=rprivate",
        f"--mount=type=bind,src={runtime.resolve()},dst=/runtime,readonly,bind-propagation=rprivate",
        f"--mount=type=bind,src={request_path.resolve()},dst=/request.json,readonly,bind-propagation=rprivate",
        "--entrypoint=/usr/bin/env",
        image_ref,
        "-i",
        "LC_ALL=C",
        "LANG=C",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "HEGEL_ACTOR_PROFILE_ID=hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_PURPOSE_ID=4",
        f"HEGEL_ACTOR_IMAGE_REF={image_ref}",
        "/usr/local/bin/python3",
        "-I",
        "-B",
        "/runtime/worker.py",
        "/candidate",
        "/request.json",
    )


def _validate_live_isolation_v1(value: object, *, actor_image_ref: str) -> None:
    if not isinstance(value, Mapping):
        _fail(FAIL_ACTOR_POLICY, "actor live-isolation receipt is absent")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if type(claimed) is not str or hashlib.sha256(canonical_json_v1(body)).hexdigest() != claimed:
        _fail(FAIL_ACTOR_POLICY, "actor live-isolation receipt hash differs")
    checks = body.get("required_checks")
    if (
        body.get("schema") != "hegel-phase3-m25-commit-b-purpose4-live-isolation/1"
        or body.get("purpose_id") != 4
        or body.get("actor_image_ref") != actor_image_ref
        or body.get("uid") != 65534
        or body.get("gid") != 65534
        or not isinstance(checks, Mapping)
        or set(checks) != {
            "nonroot_exact",
            "capability_sets_zero",
            "no_new_privileges",
            "seccomp_filter",
            "network_loopback_only",
            "six_syscalls_blocked_eperm",
            "immutable_mounts_read_only",
            "tmp_private_writable",
            "environment_exact",
            "inherited_fds_exact",
            "cgroup_limits_exact",
        }
        or any(value is not True for value in checks.values())
        or body.get("all_required_checks_passed") is not True
    ):
        _fail(FAIL_ACTOR_POLICY, "actor live-isolation checks differ")


def validate_actor_receipt_v1(
    receipt: Mapping[str, object],
    *,
    expected_manifest: Mapping[str, object],
    expected_request_sha256: str | None,
    actor_image_ref: str,
) -> dict[str, object]:
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if type(claimed) is not str or hashlib.sha256(canonical_json_v1(body)).hexdigest() != claimed:
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 publication receipt hash differs")
    exact = {
        "schema",
        "artifact_kind",
        "policy_id",
        "purpose_id",
        "audit_phase",
        "basis_commit_sha1",
        "actor_image_ref",
        "request_sha256",
        "candidate_manifest",
        "actor_recomputed_inventory_sha256",
        "runtime_inventory_sha256",
        "private_forbidden_raw_path_token_sha256s",
        "isolation_live_receipt",
        "required_checks",
        "all_required_checks_passed",
        "authority_disclosure",
        "authority_boundary",
    }
    checks = body.get("required_checks")
    boundary = body.get("authority_boundary")
    manifest = body.get("candidate_manifest")
    if (
        set(body) != exact
        or body.get("schema") != RECEIPT_SCHEMA
        or body.get("artifact_kind") != "DIAGNOSTIC_PUBLICATION_CONTROL"
        or body.get("policy_id") != POLICY_ID
        or body.get("purpose_id") != 4
        or body.get("audit_phase") != expected_manifest.get("audit_phase")
        or body.get("basis_commit_sha1") != expected_manifest.get("basis_commit_sha1")
        or body.get("actor_image_ref") != actor_image_ref
        or (expected_request_sha256 is not None and body.get("request_sha256") != expected_request_sha256)
        or manifest != dict(expected_manifest)
        or body.get("actor_recomputed_inventory_sha256")
        != expected_manifest.get("candidate_inventory_sha256")
        or not isinstance(checks, Mapping)
        or set(checks) != {
            "exact_manifest_and_file_set",
            "path_mode_size_sha256_bound",
            "nonallowlisted_and_executable_paths_absent",
            "json_strict_duplicate_free_and_required_bit_exact",
            "forbidden_secret_field_names_absent",
            "private_key_magic_and_complete_blocks_absent",
            "raw_author_or_host_paths_absent",
            "receipt_scope_exact_for_audit_phase",
            "no_key_seed_signature_marker_or_formal_action",
        }
        or any(value is not True for value in checks.values())
        or body.get("all_required_checks_passed") is not True
        or body.get("authority_disclosure") != dict(TECHNICAL_ACTOR_DISCLOSURE_V1)
        or boundary != {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        }
    ):
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 publication receipt fields differ")
    _validate_live_isolation_v1(body["isolation_live_receipt"], actor_image_ref=actor_image_ref)
    return {**body, "receipt_sha256": claimed}


def _json_transport_v1(value: object) -> object:
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_transport_v1(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_transport_v1(child) for child in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    _fail(FAIL_FORMAL_REPLAY, f"unsupported public transport type {type(value).__name__}")


def render_external_status_v1(
    *, basis_commit: str, files: Mapping[str, bytes]
) -> bytes:
    """Render the one deterministic, pre-audit external status document."""

    digest_roles = (
        ("actor_qualification", ACTOR_QUALIFICATION_REPOSITORY_PATH),
        ("errata_qualification", ERRATA_QUALIFICATION_REPOSITORY_PATH),
        ("m3_implementation_qualification", M3_IMPLEMENTATION_QUALIFICATION_REPOSITORY_PATH),
        ("bridge_qualification", BRIDGE_QUALIFICATION_REPOSITORY_PATH),
        ("live_protocol_qualification", LIVE_PROTOCOL_QUALIFICATION_REPOSITORY_PATH),
        ("pre_genesis_execution_status", PRE_GENESIS_EXECUTION_STATUS_REPOSITORY_PATH),
        ("pre_genesis_readiness", PRE_GENESIS_READINESS_REPOSITORY_PATH),
        ("formal_gate_evidence", FORMAL_EVIDENCE_REPOSITORY_PATH),
        ("formal_gate_promotion", FORMAL_PROMOTION_REPOSITORY_PATH),
        ("formal_transaction_receipt", FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH),
    )
    if any(path not in files for _role, path in digest_roles):
        _fail(FAIL_FORMAL_REPLAY, "external status renderer lacks a required role payload")
    lines = [
        "# Phase-3A M2.5 external genesis status",
        "",
        f"- basis_commit_sha1: `{_require_commit(basis_commit)}`",
        "- formal_gates: `24/24`",
        "- child_state: `NOT_RUN`",
        "- m3_run_started: `false`",
        "- publication_state: `COMMIT_B_CANDIDATE`",
        "- audit_receipt_scope: `EXCLUDED_SELF_OUTPUT_UNTIL_PREPARE`",
    ]
    lines.extend(
        f"- {role}_sha256: `{hashlib.sha256(files[path]).hexdigest()}`"
        for role, path in digest_roles
    )
    lines.extend(
        [
            "- formal_gate_delta_from_publication_audit: `0`",
            "- phase3_m3_start_required_separately: `true`",
            "- authority_effect: `NONE`",
            "",
        ]
    )
    return "\n".join(lines).encode("ascii")


def build_external_status_from_worktree_v1(
    *,
    repository: Path = REPOSITORY_ROOT,
    basis_commit: str,
) -> bytes:
    """Read the ten explicit public inputs and render the unique status bytes.

    This pre-staging helper deliberately reads worktree inputs.  The later
    ``prepare`` phase independently re-reads and binds the exact Git-index
    blobs, including the generated status document.
    """

    root = _repository(repository)
    commit = _require_commit(basis_commit)
    _git(root, ("cat-file", "-e", f"{commit}^{{commit}}"))
    head = _git(root, ("rev-parse", "HEAD")).decode("ascii", "strict").strip()
    if head != commit:
        _fail(FAIL_GIT_INDEX, "status rendering requires HEAD to equal the basis commit")
    input_paths = tuple(
        sorted(
            set(PUBLICATION_ROLE_REGISTRY)
            - {EXTERNAL_STATUS_REPOSITORY_PATH, AUDIT_RECEIPT_REPOSITORY_PATH}
        )
    )
    if len(input_paths) != 10:
        raise RuntimeError("external status input registry must contain exactly ten roles")
    raw_tokens = tuple(
        value.encode("utf-8")
        for value in {root.as_posix(), PROJECT_ROOT.as_posix(), Path.home().as_posix()}
        if len(value) >= 4
    )
    files: dict[str, bytes] = {}
    total = 0
    for path in input_paths:
        payload = _worktree_regular_bytes(root, path)
        _public_file_lint_v1(path, payload, raw_path_tokens=raw_tokens)
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            _fail(FAIL_FILE_POLICY, "external status inputs exceed total public size bound")
        files[path] = payload
    return render_external_status_v1(basis_commit=commit, files=files)


def _decode_public_json_object(files: Mapping[str, bytes], path: str) -> dict[str, object]:
    value = _plain_json(_strict_json(files[path], path=path))
    if type(value) is not dict:
        _fail(FAIL_FORMAL_REPLAY, f"publication role {path!r} is not a JSON object")
    return value


def _validate_pre_genesis_readiness_v1(
    readiness: Mapping[str, object], execution_status: Mapping[str, object], *, basis_commit: str
) -> None:
    readiness_keys = {
        "schema", "basis_commit", "ready_for_explicit_execute", "blockers",
        "formal_gates_before", "formal_gates_after", "child_state",
        "m3_run_started", "qualification_side_effects_performed",
        "qualification_network_mode",
        "qualification_persistent_rust_binary_verified_or_written",
        "qualification_non_authoritative_roots_computed",
        "ceremony_actor_key_seed_marker_side_effects_performed",
        "formal_authority_or_gate_effect",
        "static_replay_roots_are_execution_bindings",
    }
    common = {
        "basis_commit": basis_commit,
        "ready_for_explicit_execute": True,
        "blockers": [],
        "formal_gates_before": 14,
        "formal_gates_after": 14,
        "child_state": "NOT_RUN",
        "m3_run_started": False,
        "qualification_side_effects_performed": True,
        "qualification_network_mode": "none",
        "qualification_persistent_rust_binary_verified_or_written": True,
        "qualification_non_authoritative_roots_computed": True,
        "ceremony_actor_key_seed_marker_side_effects_performed": False,
        "formal_authority_or_gate_effect": "NONE",
        "static_replay_roots_are_execution_bindings": False,
    }
    if (
        set(readiness) != readiness_keys
        or readiness.get("schema") != "hegel-phase3-m25-formal-container-readiness/2"
        or any(readiness.get(key) != value for key, value in common.items())
    ):
        _fail(FAIL_FORMAL_REPLAY, "pre-genesis readiness role differs")
    execution_keys = readiness_keys | {
        "ceremony_execution_enabled_for_basis",
        "external_genesis_executed",
        "blocking_prerequisites",
    }
    if (
        set(execution_status) != execution_keys
        or execution_status.get("schema") != "hegel-phase3-m25-execution-status/2"
        or any(execution_status.get(key) != value for key, value in common.items())
        or execution_status.get("ceremony_execution_enabled_for_basis") is not True
        or execution_status.get("external_genesis_executed") is not False
        or execution_status.get("blocking_prerequisites") != []
    ):
        _fail(FAIL_FORMAL_REPLAY, "pre-genesis execution-status role differs")


def _exact_public_mapping_v1(
    value: object, keys: set[str], *, label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        _fail(FAIL_FORMAL_REPLAY, f"{label} field set differs")
    return dict(value)


def _live_protocol_source_paths_v1(
    repository: Path, basis_commit: str
) -> tuple[str, ...]:
    """Rebuild the frozen live source-path set from the supplied basis tree.

    The original qualification deliberately included every direct
    ``src/hegel_machine/*.py`` package member.  Enumerating that portion from
    the verifier's checkout would make an untracked or later-added module
    change the meaning of an older Commit-A.  Only ``git ls-tree`` over the
    caller-selected repository and basis is authoritative here.
    """

    commit = _require_commit(basis_commit)
    directory = LIVE_PROTOCOL_PACKAGE_DIRECTORY_V1
    output = _git(
        repository,
        ("ls-tree", "-r", "-z", "--full-tree", commit, "--", directory),
    )
    package_paths: list[str] = []
    for record in output.split(b"\x00"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode(
                "ascii", "strict"
            ).split(" ")
        except (ValueError, UnicodeDecodeError):
            _fail(FAIL_FORMAL_REPLAY, "live package basis-tree row is malformed")
        path = _normalize_repository_path(raw_path)
        pure = PurePosixPath(path)
        if pure.parent.as_posix() != directory or pure.suffix != ".py":
            continue
        if (
            mode not in {"100644", "100755"}
            or object_type != "blob"
            or _LOWER_SHA1.fullmatch(object_id) is None
        ):
            _fail(
                FAIL_FORMAL_REPLAY,
                f"live package basis member {path!r} is not a regular Git blob",
            )
        package_paths.append(path)
    if (
        len(package_paths) != len(set(package_paths))
        or not LIVE_PROTOCOL_REQUIRED_PACKAGE_PATHS_V1.issubset(package_paths)
    ):
        _fail(
            FAIL_FORMAL_REPLAY,
            "live package basis-tree path set is duplicate or incomplete",
        )
    paths = set(LIVE_PROTOCOL_FIXED_NONPACKAGE_SOURCE_PATHS_V1)
    paths.update(package_paths)
    return tuple(sorted(paths))


def _commit_source_set_digest_v1(
    repository: Path,
    basis_commit: str,
    paths: Sequence[str],
    *,
    domain: bytes,
) -> tuple[str, int]:
    normalized = tuple(sorted(set(paths)))
    if len(normalized) != len(paths):
        _fail(FAIL_FORMAL_REPLAY, "commit-only source path set contains duplicates")
    digest = hashlib.sha256(domain)
    for path in normalized:
        payload = _basis_blob_v1(repository, basis_commit, path)
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(hashlib.sha256(payload).digest())
    return "sha256:" + digest.hexdigest(), len(normalized)


def _committed_toolchain_policy_v1(
    repository: Path, basis_commit: str
) -> tuple[str, bytes, dict[str, object]]:
    """Load and validate the frozen Rust policy from the selected basis tree."""

    from . import phase3_m25_errata_qualification_v1 as errata_module

    path = "Hegel Machine/config/phase3_m25_approved_local_rust_toolchain_v1.json"
    blob = _basis_blob_v1(repository, basis_commit, path)
    value = _plain_json(_strict_json(blob, path=path))
    policy = _exact_public_mapping_v1(
        value,
        {
            "schema_version", "authority_boundary", "image_ref",
            "oci_manifest_digest", "image_id", "operating_system", "architecture",
            "cargo_binary_path", "cargo_binary_sha256",
            "cargo_version_stdout_sha256", "cargo_version", "rustc_binary_path",
            "rustc_binary_sha256", "rustc_version",
            "rustc_verbose_version_stdout_sha256", "runtime_environment_sha256",
            "build_environment_sha256", "runtime_seccomp_sha256",
            "build_seccomp_sha256", "cargo_lock_sha256",
            "cargo_lock_registry_package_count", "dependency_snapshot_domain",
            "dependency_snapshot_root", "dependency_snapshot_file_count",
            "host_cargo_cache_mounted_into_container", "required_docker_flags",
        },
        label="committed Rust toolchain policy",
    )
    if (
        policy["schema_version"]
        != "hegel-phase3-m25-approved-local-rust-oci-toolchain/2"
        or policy["authority_boundary"]
        != "LOCAL_DETERMINISTIC_BUILD_POLICY_NOT_EXTERNAL_ATTESTATION"
        or policy["image_ref"] != errata_module.RUST_IMAGE_REF
        or policy["oci_manifest_digest"]
        != errata_module.RUST_IMAGE_REF.rsplit("@", 1)[1]
        or policy["operating_system"] != "linux"
        or policy["architecture"] != "amd64"
        or policy["cargo_binary_path"] != errata_module.RUST_CARGO_PATH
        or policy["rustc_binary_path"] != errata_module.RUSTC_PATH
        or policy["dependency_snapshot_domain"] != errata_module.CARGO_SNAPSHOT_DOMAIN
        or policy["host_cargo_cache_mounted_into_container"] is not False
        or policy["required_docker_flags"]
        != [
            "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
        ]
    ):
        _fail(FAIL_FORMAL_REPLAY, "committed Rust toolchain policy identity differs")
    for key in (
        "oci_manifest_digest", "image_id", "cargo_binary_sha256",
        "cargo_version_stdout_sha256", "rustc_binary_sha256",
        "rustc_verbose_version_stdout_sha256", "runtime_environment_sha256",
        "build_environment_sha256", "runtime_seccomp_sha256",
        "build_seccomp_sha256", "cargo_lock_sha256", "dependency_snapshot_root",
    ):
        _require_prefixed_sha256_v1(policy[key], label=f"Rust policy {key}")
    for key in ("cargo_version", "rustc_version"):
        if type(policy[key]) is not str or not policy[key] or "\n" in policy[key]:
            _fail(FAIL_FORMAL_REPLAY, f"committed Rust policy {key} differs")
    for key in (
        "cargo_lock_registry_package_count", "dependency_snapshot_file_count",
    ):
        if type(policy[key]) is not int or policy[key] < 1:
            _fail(FAIL_FORMAL_REPLAY, f"committed Rust policy {key} differs")
    environment_bindings = {
        "runtime_environment_sha256": errata_module.RUST_RUNTIME_ENVIRONMENT,
        "build_environment_sha256": errata_module.RUST_BUILD_ENVIRONMENT,
    }
    for field, environment in environment_bindings.items():
        payload = json.dumps(
            dict(environment), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        if policy[field] != _prefixed_sha256_v1(payload):
            _fail(FAIL_FORMAL_REPLAY, f"committed Rust policy {field} differs")
    file_bindings = {
        "runtime_seccomp_sha256":
            "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
        "build_seccomp_sha256":
            "Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json",
        "cargo_lock_sha256": "Hegel Machine/rust/formal_bridge_m25/Cargo.lock",
    }
    if any(
        policy[field]
        != _prefixed_sha256_v1(_basis_blob_v1(repository, basis_commit, file_path))
        for field, file_path in file_bindings.items()
    ):
        _fail(FAIL_FORMAL_REPLAY, "committed Rust policy file binding differs")
    return path, blob, policy


def _validate_actor_report_public_only_v1(
    report: Mapping[str, object], *, repository: Path, basis_commit: str
) -> dict[str, object]:
    """Replay actor evidence against only report bytes and Commit-A blobs."""

    from . import phase3_container_actor_runtime_v1 as actor_module

    profile_blob = _basis_blob_v1(
        repository, basis_commit, PROFILE_REPOSITORY_PATH
    )
    profile_value = _plain_json(
        _strict_json(profile_blob, path=PROFILE_REPOSITORY_PATH)
    )
    if type(profile_value) is not dict:
        _fail(FAIL_FORMAL_REPLAY, "committed actor profile is not an object")
    validated = actor_module.validate_qualification_report(
        report, profile_override=profile_value
    )
    bindings = _exact_public_mapping_v1(
        validated.get("input_bindings"),
        {"profile", "seccomp", "python_probe", "rust_probe", "supervisor_runtime"},
        label="actor qualification input bindings",
    )
    expected_paths = {
        "profile": PROFILE_REPOSITORY_PATH,
        "seccomp": "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
        "python_probe": "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
        "rust_probe": "Hegel Machine/tools/phase3_container_actor_probe_v1.rs",
        "supervisor_runtime": (
            "Hegel Machine/src/hegel_machine/phase3_container_actor_runtime_v1.py"
        ),
    }
    for role, path in expected_paths.items():
        payload = _basis_blob_v1(repository, basis_commit, path)
        git_blob_sha1 = hashlib.sha1(
            b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload
        ).hexdigest()
        expected_binding = {
            "repository_path": path,
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob_sha1": git_blob_sha1,
            "basis_tree_blob_sha1_or_null": git_blob_sha1,
            "basis_commit_matches": True,
        }
        if bindings[role] != expected_binding:
            _fail(
                FAIL_FORMAL_REPLAY,
                f"actor qualification {role} binding differs from basis blob",
            )
    if (
        validated.get("basis_commit") != basis_commit
        or validated.get("basis_commit_contains_all_inputs") is not True
    ):
        _fail(FAIL_FORMAL_REPLAY, "actor qualification basis binding differs")
    return validated


def _validate_archived_secret_absence_receipt_v1(
    receipt: object, *, repository: Path, basis_commit: str
) -> None:
    from . import phase3_m25_secret_absence_v1 as secret_module

    value = _exact_public_mapping_v1(
        receipt,
        {
            "artifact", "schema_version", "artifact_kind", "machine_freeze_id",
            "status", "pass", "audited_commit_id", "scope", "policy", "counts",
            "zero_findings", "findings", "immediate_second_replay_equal",
            "authority_boundary", "claim_boundary", "diagnostic_report_id",
        },
        label="archived repository secret-absence receipt",
    )
    counts = _exact_public_mapping_v1(
        value["counts"],
        {
            "ancestor_commit_count", "path_state_commit_count",
            "tree_entry_observation_count", "unique_blob_count",
            "unique_blob_path_association_count", "unique_blob_bytes_scanned",
            "json_blob_count", "synthetic_vector_path_exemption_count",
            "unsupported_non_blob_entry_count", "finding_count",
            "offending_unique_blob_count",
        },
        label="archived secret-absence counts",
    )
    if (
        value["artifact"] != secret_module.ARTIFACT_NAME
        or value["schema_version"] != secret_module.REPORT_SCHEMA
        or value["artifact_kind"] != secret_module.ARTIFACT_KIND
        or value["machine_freeze_id"] != secret_module.MACHINE_FREEZE_ID
        or value["status"] != secret_module.PASS_STATUS
        or value["pass"] is not True
        or value["audited_commit_id"] != basis_commit
        or value["scope"] != {
            "repository_relative_prefix": secret_module.REPOSITORY_SCOPE_PREFIX,
            "history_scope": "SPECIFIED_COMMIT_AND_ALL_ANCESTORS",
            "object_scope": "UNIQUE_BLOBS_FROM_ALL_SUBTREE_PATH_STATES",
            "working_tree_consulted": False,
            "blob_content_disclosed_in_receipt": False,
        }
        or value["policy"] != secret_module._policy_payload()
        or any(type(count) is not int or count < 0 for count in counts.values())
        or counts["ancestor_commit_count"] < 1
        or counts["finding_count"] != 0
        or counts["offending_unique_blob_count"] != 0
        or value["zero_findings"] is not True
        or value["findings"] != []
        or value["immediate_second_replay_equal"] is not True
        or value["authority_boundary"] != {
            "diagnostic_only": True,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_gate_delta": 0,
            "child_state_effect": "NONE",
            "universal_secret_detection_claim": False,
        }
        or value["claim_boundary"] != secret_module.CLAIM_BOUNDARY
    ):
        _fail(FAIL_FORMAL_REPLAY, "archived secret-absence receipt differs")
    body = dict(value)
    claimed = body.pop("diagnostic_report_id")
    if claimed != secret_module.stable_hash(
        body, prefix="phase3_m25_secret_absence_"
    ):
        _fail(FAIL_FORMAL_REPLAY, "archived secret-absence receipt self-ID differs")
    try:
        replayed = (
            secret_module.repository_genesis_secret_absence_report_for_repository_v1(
                repository,
                repository / "Hegel Machine",
                basis_commit,
            )
        )
    except Exception as exc:
        _fail(
            FAIL_FORMAL_REPLAY,
            f"supplied-repository secret-absence replay failed: {type(exc).__name__}",
        )
    if canonical_json_v1(value) != canonical_json_v1(replayed):
        _fail(
            FAIL_FORMAL_REPLAY,
            "archived secret-absence receipt differs from supplied-repository history",
        )


def _validate_errata_report_public_only_v1(
    report: Mapping[str, object], *, repository: Path, basis_commit: str
) -> dict[str, object]:
    """Replay archived errata evidence from the supplied Commit-A tree."""

    from . import phase3_m25_errata_qualification_v1 as errata_module

    policy_path, policy_blob, policy = _committed_toolchain_policy_v1(
        repository, basis_commit
    )
    try:
        validated = errata_module.validate_archived_errata_qualification_report_v1(
            report,
            toolchain_policy=policy,
            approved_toolchain_policy_sha256=_prefixed_sha256_v1(policy_blob),
        )
    except Exception as exc:
        _fail(FAIL_FORMAL_REPLAY, f"archived errata envelope differs: {type(exc).__name__}")
    if validated.get("implementation_basis_commit") != basis_commit:
        _fail(FAIL_FORMAL_REPLAY, "archived errata report binds another basis commit")
    expected_sources = {
        relative: _prefixed_sha256_v1(
            _basis_blob_v1(repository, basis_commit, f"Hegel Machine/{relative}")
        )
        for relative in errata_module.SOURCE_PATHS
    }
    if validated.get("source_bindings") != expected_sources:
        _fail(FAIL_FORMAL_REPLAY, "archived errata source bindings differ")
    document_paths = {
        "BASE_AMENDMENT": (
            "Hegel Machine/docs/"
            "Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md"
        ),
        "ERRATA_RESOLUTION": (
            "Hegel Machine/docs/"
            "Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md"
        ),
        "IMPLEMENTATION_CLOSURE_ADDENDUM": (
            "Hegel Machine/docs/"
            "Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md"
        ),
    }
    expected_documents = {
        role: _prefixed_sha256_v1(_basis_blob_v1(repository, basis_commit, path))
        for role, path in document_paths.items()
    }
    if validated.get("normative_document_bindings") != expected_documents:
        _fail(FAIL_FORMAL_REPLAY, "archived errata document bindings differ")
    golden_path = "Hegel Machine/golden_vectors/phase3_m25_errata_wire_v1.json"
    golden_blob = _basis_blob_v1(repository, basis_commit, golden_path)
    if validated.get("golden_fixture_sha256") != _prefixed_sha256_v1(golden_blob):
        _fail(FAIL_FORMAL_REPLAY, "archived errata golden binding differs")
    golden_value = _plain_json(_strict_json(golden_blob, path=golden_path))
    if type(golden_value) is not dict or "report" not in golden_value:
        _fail(FAIL_FORMAL_REPLAY, "committed errata golden fixture differs")
    try:
        golden_report = errata_module._validate_vector_report(
            golden_value["report"], "committed golden report"
        )
        python_report = errata_module._validate_vector_report(
            validated.get("python_report"), "archived Python report"
        )
        rust_report = errata_module._validate_vector_report(
            validated.get("rust_report"), "archived Rust report"
        )
    except Exception as exc:
        _fail(FAIL_FORMAL_REPLAY, f"archived errata vector report differs: {type(exc).__name__}")
    if not (
        errata_module._json_type_strict_equal(python_report, golden_report)
        and errata_module._json_type_strict_equal(rust_report, golden_report)
    ):
        _fail(FAIL_FORMAL_REPLAY, "archived errata dual reports differ from golden")
    _validate_archived_secret_absence_receipt_v1(
        validated.get("repository_secret_absence_receipt"),
        repository=repository,
        basis_commit=basis_commit,
    )
    if policy_path not in {
        f"Hegel Machine/{relative}" for relative in errata_module.SOURCE_PATHS
    }:
        _fail(FAIL_FORMAL_REPLAY, "errata source set omits committed toolchain policy")
    return validated


def _validate_bridge_report_public_only_v1(
    report: Mapping[str, object], *, repository: Path, basis_commit: str
) -> str:
    """Validate the bridge report using only its bytes and Commit-A blobs."""

    from . import phase3_m25_bridge_dag_binary_qualification_v1 as bridge_module

    value = _exact_public_mapping_v1(
        dict(report),
        {
            "artifact", "schema_version", "artifact_kind", "status", "claim_level",
            "implementation_basis_commit", "source", "dependency", "toolchain",
            "container", "build", "replay_tests", "authority_boundary",
            "diagnostic_report_sha256",
        },
        label="bridge qualification report",
    )
    if (
        value["artifact"] != "phase3_m25_bridge_dag_rust_binary_qualification_v1"
        or value["schema_version"] != bridge_module.SCHEMA_VERSION
        or value["artifact_kind"] != bridge_module.ARTIFACT_KIND
        or value["status"] != bridge_module.STATUS
        or value["claim_level"] != bridge_module.CLAIM_LEVEL
        or value["authority_boundary"] != bridge_module.AUTHORITY_BOUNDARY
        or value["implementation_basis_commit"] != basis_commit
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge qualification identity/authority differs")
    report_hash = _require_prefixed_sha256_v1(
        value["diagnostic_report_sha256"], label="bridge report self-hash"
    )
    body = dict(value)
    body.pop("diagnostic_report_sha256")
    expected_report_hash = _prefixed_sha256_v1(
        bridge_module.REPORT_SHA256_PREFIX + canonical_json_v1(body)
    )
    if report_hash != expected_report_hash:
        _fail(FAIL_FORMAL_REPLAY, "bridge qualification self-hash differs")

    source = _exact_public_mapping_v1(
        value["source"],
        {
            "archive_domain", "basis_commit", "git_archive_exact",
            "worktree_bytes_equal_commit", "snapshot_read_only",
            "snapshot_manifest_sha256", "bindings",
        },
        label="bridge source evidence",
    )
    bindings = _exact_public_mapping_v1(
        source["bindings"], set(bridge_module.QUALIFICATION_SOURCE_PATHS),
        label="bridge source bindings",
    )
    expected_bindings = {
        path: _prefixed_sha256_v1(_basis_blob_v1(repository, basis_commit, path))
        for path in bridge_module.QUALIFICATION_SOURCE_PATHS
    }
    if (
        source["archive_domain"] != bridge_module.SOURCE_ARCHIVE_DOMAIN
        or source["basis_commit"] != basis_commit
        or source["git_archive_exact"] is not True
        or source["worktree_bytes_equal_commit"] is not True
        or source["snapshot_read_only"] is not True
        or bindings != expected_bindings
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge source evidence differs from basis blobs")
    _require_prefixed_sha256_v1(
        source["snapshot_manifest_sha256"], label="bridge source snapshot"
    )

    dependency = _exact_public_mapping_v1(
        value["dependency"],
        {
            "cargo_lock_repository_path", "cargo_lock_sha256", "snapshot_domain",
            "snapshot_root", "snapshot_file_count", "registry_package_count",
            "vendor_manifest_sha256", "locked_archive_checksums_verified",
            "host_cargo_cache_mounted_into_container",
        },
        label="bridge dependency evidence",
    )
    cargo_lock_path = f"{bridge_module.CRATE_REPOSITORY_PATH}/Cargo.lock"
    if (
        dependency["cargo_lock_repository_path"] != cargo_lock_path
        or dependency["cargo_lock_sha256"] != expected_bindings[cargo_lock_path]
        or dependency["snapshot_domain"] != bridge_module.CARGO_SNAPSHOT_DOMAIN
        or dependency["locked_archive_checksums_verified"] is not True
        or dependency["host_cargo_cache_mounted_into_container"] is not False
        or type(dependency["snapshot_file_count"]) is not int
        or type(dependency["registry_package_count"]) is not int
        or dependency["snapshot_file_count"] < 1
        or dependency["registry_package_count"] < 1
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge dependency evidence differs")
    for key in ("snapshot_root", "vendor_manifest_sha256"):
        _require_prefixed_sha256_v1(dependency[key], label=f"bridge dependency {key}")

    policy_path = bridge_module.APPROVED_TOOLCHAIN_POLICY_REPOSITORY_PATH
    policy_blob = _basis_blob_v1(repository, basis_commit, policy_path)
    policy_value = _plain_json(_strict_json(policy_blob, path=policy_path))
    policy = _exact_public_mapping_v1(
        policy_value,
        {
            "schema_version", "authority_boundary", "image_ref",
            "oci_manifest_digest", "image_id", "operating_system", "architecture",
            "cargo_binary_path", "cargo_binary_sha256",
            "cargo_version_stdout_sha256", "cargo_version", "rustc_binary_path",
            "rustc_binary_sha256", "rustc_version",
            "rustc_verbose_version_stdout_sha256", "runtime_environment_sha256",
            "build_environment_sha256", "runtime_seccomp_sha256",
            "build_seccomp_sha256", "cargo_lock_sha256",
            "cargo_lock_registry_package_count", "dependency_snapshot_domain",
            "dependency_snapshot_root", "dependency_snapshot_file_count",
            "host_cargo_cache_mounted_into_container", "required_docker_flags",
        },
        label="committed bridge toolchain policy",
    )
    if (
        policy["schema_version"]
        != "hegel-phase3-m25-approved-local-rust-oci-toolchain/2"
        or policy["authority_boundary"]
        != "LOCAL_DETERMINISTIC_BUILD_POLICY_NOT_EXTERNAL_ATTESTATION"
        or policy["image_ref"] != bridge_module.RUST_IMAGE_REF
        or policy["oci_manifest_digest"]
        != bridge_module.RUST_IMAGE_REF.rsplit("@", 1)[1]
        or policy["operating_system"] != "linux"
        or policy["architecture"] != "amd64"
        or policy["cargo_binary_path"] != bridge_module.RUST_CARGO_PATH
        or type(policy["rustc_binary_path"]) is not str
        or not policy["rustc_binary_path"]
        or policy["dependency_snapshot_domain"] != bridge_module.CARGO_SNAPSHOT_DOMAIN
        or policy["host_cargo_cache_mounted_into_container"] is not False
        or policy["required_docker_flags"]
        != [
            "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
        ]
    ):
        _fail(FAIL_FORMAL_REPLAY, "committed bridge toolchain policy identity differs")
    for key in (
        "oci_manifest_digest", "image_id", "cargo_binary_sha256",
        "cargo_version_stdout_sha256", "rustc_binary_sha256",
        "rustc_verbose_version_stdout_sha256", "runtime_environment_sha256",
        "build_environment_sha256", "runtime_seccomp_sha256",
        "build_seccomp_sha256", "cargo_lock_sha256", "dependency_snapshot_root",
    ):
        _require_prefixed_sha256_v1(policy[key], label=f"bridge policy {key}")
    for key in ("cargo_version", "rustc_version"):
        if type(policy[key]) is not str or not policy[key] or "\n" in policy[key]:
            _fail(FAIL_FORMAL_REPLAY, f"bridge policy {key} differs")
    for key in (
        "cargo_lock_registry_package_count", "dependency_snapshot_file_count",
    ):
        if type(policy[key]) is not int or policy[key] < 1:
            _fail(FAIL_FORMAL_REPLAY, f"bridge policy {key} differs")
    seccomp_bindings = {
        "runtime_seccomp_sha256":
            "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
        "build_seccomp_sha256":
            "Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json",
    }
    if any(
        policy[field] != expected_bindings[path]
        for field, path in seccomp_bindings.items()
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge policy seccomp binding differs from basis blobs")

    if (
        dependency["snapshot_root"] != policy["dependency_snapshot_root"]
        or dependency["snapshot_file_count"]
        != policy["dependency_snapshot_file_count"]
        or dependency["registry_package_count"]
        != policy["cargo_lock_registry_package_count"]
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge dependency evidence differs from policy")

    toolchain = _exact_public_mapping_v1(
        value["toolchain"],
        {"approved_policy_repository_path", "approved_policy_sha256", "receipt"},
        label="bridge toolchain evidence",
    )
    receipt = _exact_public_mapping_v1(
        toolchain["receipt"],
        {
            "image_ref", "image_id", "oci_manifest_digest", "operating_system",
            "architecture", "cargo_binary_path", "cargo_binary_sha256",
            "cargo_version", "cargo_version_stdout_sha256", "rustc_binary_path",
            "rustc_binary_sha256", "rustc_version",
            "rustc_verbose_version_stdout_sha256", "runtime_environment_sha256",
            "build_environment_sha256", "runtime_seccomp_sha256",
            "build_seccomp_sha256", "image_config_environment_ignored",
            "pull_policy", "network_mode", "toolchain_receipt_is_external_attestation",
        },
        label="bridge toolchain receipt",
    )
    expected_receipt = {
        key: policy[key]
        for key in (
            "image_ref", "image_id", "oci_manifest_digest", "operating_system",
            "architecture", "cargo_binary_path", "cargo_binary_sha256",
            "cargo_version", "cargo_version_stdout_sha256", "rustc_binary_path",
            "rustc_binary_sha256", "rustc_version",
            "rustc_verbose_version_stdout_sha256", "runtime_environment_sha256",
            "build_environment_sha256", "runtime_seccomp_sha256",
            "build_seccomp_sha256",
        )
    }
    expected_receipt.update(
        {
            "image_config_environment_ignored": True,
            "pull_policy": "never",
            "network_mode": "none",
            "toolchain_receipt_is_external_attestation": False,
        }
    )
    if (
        toolchain["approved_policy_repository_path"] != policy_path
        or toolchain["approved_policy_sha256"] != _prefixed_sha256_v1(policy_blob)
        or receipt != expected_receipt
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge toolchain basis binding differs")

    container = _exact_public_mapping_v1(
        value["container"],
        {
            "docker_executable", "docker_host", "control_plane_binding",
            "daemon_identity_receipt", "daemon_receipt_binding", "image_ref",
            "pull_policy", "network_mode", "read_only_root",
            "inherited_environment_allowed", "runtime_docker_policy_id",
            "build_docker_policy_id", "runtime_seccomp_sha256",
            "build_seccomp_sha256",
        },
        label="bridge container evidence",
    )
    daemon = _exact_public_mapping_v1(
        container["daemon_identity_receipt"],
        set(container["daemon_identity_receipt"])
        if type(container["daemon_identity_receipt"]) is dict else set(),
        label="bridge Docker daemon receipt",
    )
    try:
        daemon_binding = local_docker_daemon_receipt_binding_v1(daemon).hex()
    except Phase3LocalRuntimeError:
        _fail(FAIL_FORMAL_REPLAY, "bridge Docker daemon receipt differs")
    if (
        container["docker_executable"] != "/usr/bin/docker"
        or container["docker_host"] != "unix:///var/run/docker.sock"
        or container["image_ref"] != bridge_module.RUST_IMAGE_REF
        or container["pull_policy"] != "never"
        or container["network_mode"] != "none"
        or container["read_only_root"] is not True
        or container["inherited_environment_allowed"] is not False
        or container["runtime_docker_policy_id"]
        != bridge_module.RUNTIME_DOCKER_POLICY_ID
        or container["build_docker_policy_id"] != bridge_module.BUILD_DOCKER_POLICY_ID
        or container["runtime_seccomp_sha256"] != policy["runtime_seccomp_sha256"]
        or container["build_seccomp_sha256"] != policy["build_seccomp_sha256"]
        or container["daemon_receipt_binding"] != daemon_binding
        or container["control_plane_binding"] != daemon.get("control_plane_binding")
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge container evidence differs")

    build = _exact_public_mapping_v1(
        value["build"],
        {
            "release_profile", "cargo_locked", "cargo_offline", "fresh_linux_local_target",
            "source_mount_read_only", "vendor_mount_read_only", "test_command",
            "build_command", "fresh_binary_sha256", "persisted_binary",
        },
        label="bridge build evidence",
    )
    fresh_digest = _require_prefixed_sha256_v1(
        build["fresh_binary_sha256"], label="bridge fresh binary"
    )
    if (
        any(
            build[field] is not True
            for field in (
                "release_profile", "cargo_locked", "cargo_offline",
                "fresh_linux_local_target", "source_mount_read_only",
                "vendor_mount_read_only",
            )
        )
        or build["test_command"] != list(bridge_module.TEST_COMMAND)
        or build["build_command"] != list(bridge_module.BUILD_COMMAND)
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge build evidence differs")
    persisted = _exact_public_mapping_v1(
        build["persisted_binary"],
        {"repository_path", "sha256", "mode_octal", "atomic_replace", "is_symlink"},
        label="bridge persisted-binary declaration",
    )
    if persisted != {
        "repository_path": bridge_module.PERSISTED_BINARY_REPOSITORY_PATH,
        "sha256": fresh_digest,
        "mode_octal": "0755",
        "atomic_replace": True,
        "is_symlink": False,
    }:
        _fail(FAIL_FORMAL_REPLAY, "bridge persisted-binary declaration differs")

    replays = _exact_public_mapping_v1(
        value["replay_tests"],
        {
            "fixture_repository_path", "fixture_sha256", "package_sha256",
            "contains_private_key", "contains_signature", "contains_seed", "tests",
            "all_passed",
        },
        label="bridge replay evidence",
    )
    fixture_path = bridge_module.GOLDEN_FIXTURE_REPOSITORY_PATH
    fixture_blob = _basis_blob_v1(repository, basis_commit, fixture_path)
    fixture_value = _plain_json(_strict_json(fixture_blob, path=fixture_path))
    fixture = _exact_public_mapping_v1(
        fixture_value,
        {
            "artifact_kind", "authority", "compression",
            "contains_formal_commitment", "contains_private_key", "contains_seed",
            "contains_signature", "expected", "package_sha256",
            "package_uncompressed_size", "package_zlib_base64", "purpose_id",
            "schema_version",
        },
        label="committed bridge replay fixture",
    )
    encoded_package = fixture["package_zlib_base64"]
    if type(encoded_package) is not str:
        _fail(FAIL_FORMAL_REPLAY, "committed bridge replay fixture payload differs")
    try:
        package = zlib.decompress(base64.b64decode(encoded_package, validate=True))
    except (ValueError, zlib.error):
        _fail(FAIL_FORMAL_REPLAY, "committed bridge replay fixture cannot be decoded")
    if (
        fixture["schema_version"]
        != "hegel-m25-bridge-dag-purpose1-replay-fixture/1"
        or fixture["artifact_kind"] != "SYNTHETIC_PUBLIC_NON_AUTHORITATIVE"
        or fixture["authority"] is not False
        or fixture["purpose_id"] != 1
        or fixture["compression"] != "zlib-level-9"
        or any(
            fixture[field] is not False
            for field in (
                "contains_formal_commitment", "contains_private_key",
                "contains_seed", "contains_signature",
            )
        )
        or type(fixture["package_uncompressed_size"]) is not int
        or fixture["package_uncompressed_size"] != len(package)
        or fixture["package_sha256"] != _prefixed_sha256_v1(package)
    ):
        _fail(FAIL_FORMAL_REPLAY, "committed bridge replay fixture binding differs")
    if (
        replays["fixture_repository_path"] != fixture_path
        or replays["fixture_sha256"] != _prefixed_sha256_v1(fixture_blob)
        or replays["package_sha256"] != fixture["package_sha256"]
        or replays["contains_private_key"] is not False
        or replays["contains_signature"] is not False
        or replays["contains_seed"] is not False
        or replays["all_passed"] is not True
        or type(replays["tests"]) is not list
    ):
        _fail(FAIL_FORMAL_REPLAY, "bridge replay evidence/basis fixture differs")
    tests = replays["tests"]
    expected_ids = [
        "FRESH_PUBLIC_PURPOSE1_REPLAY_PASS",
        "PUBLIC_PREIMAGE_SUBSTITUTION_REJECTED",
        "PUBLIC_NODE_OMISSION_REJECTED",
        "AUTHORITATIVE_FLAG_WITHOUT_RUNTIME_OPT_IN_REJECTED",
        "PERSISTED_PUBLIC_PURPOSE1_REPLAY_PASS",
    ]
    if [row.get("test_id") if type(row) is dict else None for row in tests] != expected_ids:
        _fail(FAIL_FORMAL_REPLAY, "bridge replay test identity/order differs")
    expected_errors = [
        None, bridge_module.FAIL_ROOT_BINDING, bridge_module.FAIL_NODE_SET,
        bridge_module.FAIL_PACKAGE_AUTHORITY, None,
    ]
    for index, (row_value, error) in enumerate(
        zip(tests, expected_errors, strict=True)
    ):
        row = _exact_public_mapping_v1(
            row_value,
            {
                "test_id", "expected_returncode", "observed_returncode",
                "expected_error_code_or_null", "stdout_sha256", "stderr_sha256",
            },
            label=f"bridge replay test {index}",
        )
        expected_returncode = 0 if error is None else 1
        if (
            row["expected_error_code_or_null"] != error
            or row["expected_returncode"] != expected_returncode
            or row["observed_returncode"] != expected_returncode
        ):
            _fail(FAIL_FORMAL_REPLAY, f"bridge replay test {index} differs")
        _require_prefixed_sha256_v1(
            row["stdout_sha256"], label=f"bridge replay test {index} stdout"
        )
        _require_prefixed_sha256_v1(
            row["stderr_sha256"], label=f"bridge replay test {index} stderr"
        )
    if tests[0]["stdout_sha256"] != tests[-1]["stdout_sha256"]:
        _fail(FAIL_FORMAL_REPLAY, "bridge fresh/persisted replay receipt differs")
    return fresh_digest


def _validate_live_report_public_only_v1(
    report: Mapping[str, object],
    *,
    repository: Path,
    basis_commit: str,
    m3_receipt: Mapping[str, object],
    bridge_report: Mapping[str, object],
    bridge_binary_digest: str,
) -> object:
    """Deep-replay live evidence while replacing all local-file checks."""

    from . import phase3_m25_actor_protocol_qualification_v1 as live_module

    validated = live_module.validate_actor_protocol_qualification_report_v1(
        report,
        expected_basis_commit=basis_commit,
        verify_commit_sources=False,
        verify_local_implementation_bindings=False,
    )
    source_digest, source_count = _commit_source_set_digest_v1(
        repository,
        basis_commit,
        _live_protocol_source_paths_v1(repository, basis_commit),
        domain=live_module.SOURCE_SET_HASH_DOMAIN,
    )
    bindings = _exact_public_mapping_v1(
        report.get("implementation_bindings"),
        {
            "formal_rust_replay_binary_sha256", "bridge_rust_replay_binary_sha256",
            "bridge_rust_qualification_report_sha256",
            "m3_implementation_qualification_receipt_sha256",
            "m3_implementation_qualification_receipt",
        },
        label="live implementation bindings",
    )
    rust_receipt = m3_receipt.get("rust")
    if type(rust_receipt) is not dict:
        _fail(FAIL_FORMAL_REPLAY, "M3 Rust qualification receipt is absent")
    rust_binary = rust_receipt.get("binary_digest")
    if type(rust_binary) is not str or _LOWER_SHA256.fullmatch(rust_binary) is None:
        _fail(FAIL_FORMAL_REPLAY, "M3 Rust binary digest is malformed")
    if (
        report.get("commit_a_source_set_sha256") != source_digest
        or report.get("commit_a_source_file_count") != source_count
        or bindings["formal_rust_replay_binary_sha256"] != f"sha256:{rust_binary}"
        or bindings["bridge_rust_replay_binary_sha256"] != bridge_binary_digest
        or bindings["bridge_rust_qualification_report_sha256"]
        != bridge_report.get("diagnostic_report_sha256")
        or bindings["m3_implementation_qualification_receipt"] != dict(m3_receipt)
        or bindings["m3_implementation_qualification_receipt_sha256"]
        != _prefixed_sha256_v1(canonical_json_v1(dict(m3_receipt)))
    ):
        _fail(FAIL_FORMAL_REPLAY, "live report commit-only source/implementation binding differs")
    return validated


def _validate_role_specific_public_payloads_v1(
    files: Mapping[str, bytes],
    *,
    repository: Path,
    basis_commit: str,
    typed_inputs: object,
    commit_only: bool,
) -> dict[str, object]:
    if type(commit_only) is not bool:
        _fail(FAIL_FORMAL_REPLAY, "role replay scope is not a boolean")
    required = set(_required_role_paths(exclude_receipt=True))
    if not required.issubset(files):
        _fail(FAIL_FORMAL_REPLAY, "role-specific replay lacks a required publication payload")
    actor = _decode_public_json_object(files, ACTOR_QUALIFICATION_REPOSITORY_PATH)
    errata = _decode_public_json_object(files, ERRATA_QUALIFICATION_REPOSITORY_PATH)
    m3_receipt = _decode_public_json_object(
        files, M3_IMPLEMENTATION_QUALIFICATION_REPOSITORY_PATH
    )
    bridge = _decode_public_json_object(files, BRIDGE_QUALIFICATION_REPOSITORY_PATH)
    live = _decode_public_json_object(files, LIVE_PROTOCOL_QUALIFICATION_REPOSITORY_PATH)
    execution_status = _decode_public_json_object(
        files, PRE_GENESIS_EXECUTION_STATUS_REPOSITORY_PATH
    )
    readiness = _decode_public_json_object(files, PRE_GENESIS_READINESS_REPOSITORY_PATH)
    try:
        from .phase3_container_actor_runtime_v1 import validate_qualification_report
        from .phase3_m3_implementation_qualification_v1 import (
            load_committed_dual_golden_v1,
            validate_qualification_receipt_v1,
        )
        from .phase3_m25_actor_protocol_qualification_v1 import (
            validate_actor_protocol_qualification_report_v1,
        )
        from .phase3_m25_bridge_dag_binary_qualification_v1 import (
            validate_rust_bridge_dag_binary_qualification_report_v1,
        )

        validated_actor = (
            _validate_actor_report_public_only_v1(
                actor, repository=repository, basis_commit=basis_commit
            )
            if commit_only
            else validate_qualification_report(actor)
        )
        validated_errata = (
            _validate_errata_report_public_only_v1(
                errata, repository=repository, basis_commit=basis_commit
            )
            if commit_only
            else errata
        )
        golden, _preimage, _root = load_committed_dual_golden_v1(
            repository, basis_commit
        )
        validate_qualification_receipt_v1(
            m3_receipt, golden=golden, basis_commit=basis_commit
        )
        if commit_only:
            bridge_digest = _validate_bridge_report_public_only_v1(
                bridge, repository=repository, basis_commit=basis_commit
            )
            validated_live = _validate_live_report_public_only_v1(
                live,
                repository=repository,
                basis_commit=basis_commit,
                m3_receipt=m3_receipt,
                bridge_report=bridge,
                bridge_binary_digest=bridge_digest,
            )
        else:
            validate_rust_bridge_dag_binary_qualification_report_v1(
                bridge, expected_basis_commit=basis_commit
            )
            validated_live = validate_actor_protocol_qualification_report_v1(
                live, expected_basis_commit=basis_commit
            )
    except Exception as exc:
        _fail(FAIL_FORMAL_REPLAY, f"standalone publication role replay failed: {type(exc).__name__}")
    actor_from_evidence = _json_transport_v1(
        getattr(typed_inputs, "actor_qualification_report", None)
    )
    errata_from_evidence = _json_transport_v1(
        getattr(typed_inputs, "errata_qualification_report", None)
    )
    live_m3 = live.get("implementation_bindings")
    if (
        validated_actor.get("basis_commit") != basis_commit
        or actor != actor_from_evidence
        or errata != errata_from_evidence
        or validated_errata != errata
        or not isinstance(live_m3, Mapping)
        or live_m3.get("m3_implementation_qualification_receipt") != m3_receipt
        or dict(validated_live.report) != live
    ):
        _fail(FAIL_FORMAL_REPLAY, "standalone/embedded publication role binding differs")
    _validate_pre_genesis_readiness_v1(
        readiness, execution_status, basis_commit=basis_commit
    )
    if files[EXTERNAL_STATUS_REPOSITORY_PATH] != render_external_status_v1(
        basis_commit=basis_commit, files=files
    ):
        _fail(FAIL_FORMAL_REPLAY, "external status Markdown differs from exact renderer")
    return {
        "role_specific_payload_count": 11,
        "standalone_actor_equals_formal_evidence": True,
        "standalone_errata_equals_formal_evidence": True,
        "standalone_m3_equals_live_protocol_binding": True,
        "bridge_and_live_protocol_strict_replay_passed": True,
        "pre_genesis_status_and_readiness_bound": True,
        "external_status_exact_renderer_equal": True,
    }


def _host_strict_replay_formal_public_payloads_v1(
    files: Mapping[str, bytes],
    *,
    basis_commit: str,
    require_formal_payloads: bool,
    repository: Path = REPOSITORY_ROOT,
    commit_only: bool = False,
) -> dict[str, object]:
    present = tuple(path for path in FORMAL_PUBLIC_PATHS if path in files)
    if present and set(present) != set(FORMAL_PUBLIC_PATHS):
        _fail(FAIL_FORMAL_REPLAY, "formal public payload set is incomplete")
    if require_formal_payloads and set(present) != set(FORMAL_PUBLIC_PATHS):
        _fail(FAIL_FORMAL_REPLAY, "formal evidence/promotion/transaction receipt are required")
    if not present:
        return {
            "formal_public_payloads_present": False,
            "formal_public_payload_count": 0,
            "formal_gate_replay_performed": False,
        }
    decoded: dict[str, dict[str, object]] = {}
    for path in FORMAL_PUBLIC_PATHS:
        value = _plain_json(_strict_json(files[path], path=path))
        if type(value) is not dict:
            _fail(FAIL_FORMAL_REPLAY, f"formal public payload {path!r} is not an object")
        decoded[path] = value
    try:
        from .phase3_m25_formal_container_executor_v1 import (
            PUBLICATION_RECEIPT_SCHEMA,
            load_gate_evidence_inputs_v1,
            replay_public_gate_evidence_v1,
        )
        from .phase3_m25_container_ceremony_v1 import (
            _evaluate_gates_15_24_with_prevalidated_report_basis_v1,
            promote_gate_evidence_v1,
        )

        typed_inputs = load_gate_evidence_inputs_v1(
            decoded[FORMAL_EVIDENCE_REPOSITORY_PATH]
        )
        if commit_only:
            prevalidated_actor = _validate_actor_report_public_only_v1(
                typed_inputs.actor_qualification_report,
                repository=repository,
                basis_commit=basis_commit,
            )
            prevalidated_errata = _validate_errata_report_public_only_v1(
                typed_inputs.errata_qualification_report,
                repository=repository,
                basis_commit=basis_commit,
            )
            replayed = promote_gate_evidence_v1(
                _evaluate_gates_15_24_with_prevalidated_report_basis_v1(
                    typed_inputs,
                    actor_report=prevalidated_actor,
                    errata_report=prevalidated_errata,
                )
            )
        else:
            replayed = replay_public_gate_evidence_v1(
                decoded[FORMAL_EVIDENCE_REPOSITORY_PATH]
            )
    except Exception as exc:
        _fail(FAIL_FORMAL_REPLAY, f"formal gate evidence replay failed: {type(exc).__name__}")
    if canonical_json_v1(replayed) != files[FORMAL_PROMOTION_REPOSITORY_PATH]:
        _fail(FAIL_FORMAL_REPLAY, "formal promotion differs from host strict replay")
    promotion = decoded[FORMAL_PROMOTION_REPOSITORY_PATH]
    transaction = decoded[FORMAL_TRANSACTION_RECEIPT_REPOSITORY_PATH]
    run_id = typed_inputs.execution_candidate_fields.get("run_id")
    ledger_id = typed_inputs.ledger_genesis_fields.get("ledger_id")
    seed_verification_digest = transaction.get(
        "seed_custody_verification_receipt_sha256_or_null"
    )
    expected_receipt_keys = {
        "schema",
        "basis_commit",
        "run_id_hex",
        "ledger_id_hex",
        "public_evidence_sha256",
        "public_promotion_sha256",
        "seed_custody_verification_receipt_sha256_or_null",
        "prospective_public_replay_passed",
        "marker_was_complete_during_staging",
        "actor_cleanup_required_before_publication",
        "authority_disclosure",
        "contains_private_key",
        "contains_raw_split_seed",
        "contains_split_assignment_rows",
    }
    if (
        promotion.get("basis_commit") != basis_commit
        or promotion.get("child_state") != "NOT_RUN"
        or promotion.get("m3_run_started") is not False
        or promotion.get("m3_entry_qualified") is not True
        or promotion.get("phase3_m3_start_required_separately") is not True
        or set(transaction) != expected_receipt_keys
        or transaction.get("schema") != PUBLICATION_RECEIPT_SCHEMA
        or transaction.get("basis_commit") != basis_commit
        or type(run_id) is not bytes
        or len(run_id) != 16
        or type(ledger_id) is not bytes
        or len(ledger_id) != 16
        or run_id == ledger_id
        or transaction.get("run_id_hex") != run_id.hex()
        or transaction.get("ledger_id_hex") != ledger_id.hex()
        or transaction.get("public_evidence_sha256")
        != hashlib.sha256(files[FORMAL_EVIDENCE_REPOSITORY_PATH]).hexdigest()
        or transaction.get("public_promotion_sha256")
        != hashlib.sha256(files[FORMAL_PROMOTION_REPOSITORY_PATH]).hexdigest()
        or transaction.get("prospective_public_replay_passed") is not True
        or transaction.get("marker_was_complete_during_staging") is not False
        or transaction.get("actor_cleanup_required_before_publication") is not True
        or type(seed_verification_digest) is not str
        or _LOWER_SHA256.fullmatch(seed_verification_digest) is None
        or transaction.get("authority_disclosure") != dict(TECHNICAL_ACTOR_DISCLOSURE_V1)
        or any(
            transaction.get(name) is not False
            for name in (
                "contains_private_key",
                "contains_raw_split_seed",
                "contains_split_assignment_rows",
            )
        )
    ):
        _fail(FAIL_FORMAL_REPLAY, "formal public receipt/NOT_RUN authority boundary differs")
    role_replay = _validate_role_specific_public_payloads_v1(
        files,
        repository=repository,
        basis_commit=basis_commit,
        typed_inputs=typed_inputs,
        commit_only=commit_only,
    )
    return {
        "formal_public_payloads_present": True,
        "formal_public_payload_count": 3,
        "formal_gate_replay_performed": True,
        "formal_gate_replay_basis_commit": basis_commit,
        "formal_promotion_sha256": hashlib.sha256(
            files[FORMAL_PROMOTION_REPOSITORY_PATH]
        ).hexdigest(),
        "child_state": "NOT_RUN",
        "m3_run_started": False,
        "role_specific_replay": role_replay,
    }


@dataclass(frozen=True, slots=True)
class CommitBActorAuditResultV1:
    receipt: Mapping[str, object]
    canonical_receipt_bytes: bytes
    manifest: Mapping[str, object]
    host_formal_replay: Mapping[str, object]


def run_commit_b_publication_actor_audit_v1(
    *,
    repository: Path = REPOSITORY_ROOT,
    basis_commit: str,
    finalize_index: bool = False,
    permit_staged_receipt_for_prepare_replay: bool = False,
) -> CommitBActorAuditResultV1:
    """Run no-key purpose-4 audit over exact staged bytes, then host-replay."""

    root = _repository(repository)
    manifest, files = build_staged_candidate_manifest_v1(
        root,
        basis_commit=basis_commit,
        exclude_receipt=not finalize_index,
        permit_staged_receipt_for_replay=permit_staged_receipt_for_prepare_replay,
    )
    image_ref = _load_image_ref(root, basis_commit)
    try:
        temporary_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-commit-b-publication-audit-",
            repository_root=root,
            parent=DEFAULT_LINUX_LOCAL_RUNTIME_PARENT,
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_LOCAL_RUNTIME, f"{exc.code}: {exc.detail}")
    with temporary_owner as temporary:
        private_root = Path(temporary)
        try:
            control = prepare_local_docker_control_plane_v1(
                private_root, repository_root=root
            )
            version = subprocess.run(
                control.command("version", "--format", "{{json .}}"),
                env=dict(control.environment), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60,
            )
            info = subprocess.run(
                control.command("info", "--format", "{{json .}}"),
                env=dict(control.environment), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60,
            )
            if version.returncode or info.returncode:
                raise ValueError("local Docker identity query failed")
            daemon = build_local_docker_daemon_identity_receipt_v1(
                control,
                version_payload=json.loads(version.stdout),
                info_payload=json.loads(info.stdout),
                repository_root=root,
            )
            local_docker_daemon_receipt_binding_v1(daemon)
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError, Phase3LocalRuntimeError) as exc:
            _fail(FAIL_LOCAL_RUNTIME, f"local Docker control plane failed: {type(exc).__name__}")
        candidate = _write_candidate_snapshot(private_root, manifest, files)
        runtime_inventory = _copy_actor_runtime(
            private_root, repository=root, basis_commit=basis_commit
        )
        runtime = private_root / "runtime"
        private_tokens = sorted(
            {
                root.as_posix(),
                PROJECT_ROOT.as_posix(),
                Path.home().as_posix(),
            }
        )
        request = build_actor_request_v1(
            manifest=manifest,
            runtime_inventory=runtime_inventory,
            basis_commit=basis_commit,
            actor_image_ref=image_ref,
            raw_path_tokens=private_tokens,
        )
        request_path = private_root / "request.json"
        request_path.write_bytes(canonical_json_v1(request))
        request_path.chmod(0o444)
        command = publication_actor_container_command_v1(
            candidate=candidate,
            runtime=runtime,
            request_path=request_path,
            image_ref=image_ref,
        )
        try:
            completed = subprocess.run(
                command,
                env=dict(control.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=900,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _fail(FAIL_ACTOR_POLICY, f"purpose-4 actor could not complete: {type(exc).__name__}")
        if completed.returncode != 0 or completed.stderr:
            detail = completed.stderr.decode("ascii", "replace")[-300:].strip()
            _fail(FAIL_ACTOR_POLICY, f"purpose-4 actor exited {completed.returncode}: {detail}")
        if len(completed.stdout) > MAX_ACTOR_OUTPUT_BYTES:
            _fail(FAIL_ACTOR_RESPONSE, "purpose-4 actor output is oversized")
        try:
            response = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(FAIL_ACTOR_RESPONSE, f"purpose-4 actor response is invalid JSON: {exc}")
        if canonical_json_v1(response) != completed.stdout:
            _fail(FAIL_ACTOR_RESPONSE, "purpose-4 actor response is not canonical JSON")
        validated = validate_actor_receipt_v1(
            response,
            expected_manifest=manifest,
            expected_request_sha256=str(request["request_sha256"]),
            actor_image_ref=image_ref,
        )
    replay_manifest, replay_files = build_staged_candidate_manifest_v1(
        root,
        basis_commit=basis_commit,
        exclude_receipt=not finalize_index,
        permit_staged_receipt_for_replay=permit_staged_receipt_for_prepare_replay,
    )
    if replay_manifest != manifest or replay_files != files:
        _fail(FAIL_GIT_INDEX, "Git index/worktree changed across purpose-4 actor audit")
    formal = _host_strict_replay_formal_public_payloads_v1(
        replay_files,
        basis_commit=basis_commit,
        require_formal_payloads=True,
        repository=root,
        commit_only=False,
    )
    return CommitBActorAuditResultV1(
        receipt=validated,
        canonical_receipt_bytes=canonical_json_v1(validated),
        manifest=manifest,
        host_formal_replay=formal,
    )


def _read_index_path(repository: Path, path: str) -> bytes:
    _mode, _object_id, payload = _index_entry(repository, path)
    return payload


def finalize_staged_commit_b_publication_v1(
    *, repository: Path = REPOSITORY_ROOT, basis_commit: str
) -> dict[str, object]:
    """Host-only final replay after the actor receipt has itself been staged."""

    root = _repository(repository)
    raw_rows = _raw_staged_rows(root, _require_commit(basis_commit))
    all_paths = tuple(path for path, _mode in raw_rows)
    if all_paths.count(AUDIT_RECEIPT_REPOSITORY_PATH) != 1:
        _fail(FAIL_FINAL_STAGED_SET, "final staged set must contain exactly one audit receipt")
    prepare_manifest, _prepare_files = build_staged_candidate_manifest_v1(
        root,
        basis_commit=basis_commit,
        exclude_receipt=True,
        permit_staged_receipt_for_replay=True,
    )
    receipt_bytes = _read_index_path(root, AUDIT_RECEIPT_REPOSITORY_PATH)
    if _worktree_regular_bytes(root, AUDIT_RECEIPT_REPOSITORY_PATH) != receipt_bytes:
        _fail(FAIL_FINAL_STAGED_SET, "staged audit receipt has unstaged drift")
    receipt_value = _plain_json(
        _strict_json(receipt_bytes, path=AUDIT_RECEIPT_REPOSITORY_PATH)
    )
    if type(receipt_value) is not dict or canonical_json_v1(receipt_value) != receipt_bytes:
        _fail(FAIL_FINAL_STAGED_SET, "staged audit receipt is not canonical JSON")
    fresh_prepare = run_commit_b_publication_actor_audit_v1(
        repository=root,
        basis_commit=basis_commit,
        finalize_index=False,
        permit_staged_receipt_for_prepare_replay=True,
    )
    if fresh_prepare.manifest != prepare_manifest or fresh_prepare.canonical_receipt_bytes != receipt_bytes:
        _fail(
            FAIL_FINAL_STAGED_SET,
            "staged prepare receipt differs from a fresh receipt-excluded actor replay",
        )
    validated = dict(fresh_prepare.receipt)
    final_actor = run_commit_b_publication_actor_audit_v1(
        repository=root,
        basis_commit=basis_commit,
        finalize_index=True,
    )
    manifest = final_actor.manifest
    if _receipt_excluded_manifest_projection_v1(manifest) != fresh_prepare.manifest:
        _fail(
            FAIL_FINAL_STAGED_SET,
            "final actor receipt-excluded inventory differs from fresh prepare replay",
        )
    final_rows = manifest.get("candidate_files")
    assert isinstance(final_rows, list)
    final_receipt_row = next(
        row for row in final_rows
        if isinstance(row, Mapping)
        and row.get("path") == AUDIT_RECEIPT_REPOSITORY_PATH
    )
    if (
        final_receipt_row.get("byte_length") != len(receipt_bytes)
        or final_receipt_row.get("sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
    ):
        _fail(FAIL_FINAL_STAGED_SET, "final actor receipt row differs from staged receipt")
    expected_final = tuple(
        sorted(
            [str(row["path"]) for row in manifest["candidate_files"]]  # type: ignore[index]
        )
    )
    if tuple(sorted(all_paths)) != expected_final:
        _fail(FAIL_FINAL_STAGED_SET, "final staged path set differs from candidate + receipt")
    _public_file_lint_v1(
        AUDIT_RECEIPT_REPOSITORY_PATH,
        receipt_bytes,
        raw_path_tokens=tuple(
            value.encode("utf-8")
            for value in {root.as_posix(), PROJECT_ROOT.as_posix(), Path.home().as_posix()}
        ),
    )
    formal = dict(final_actor.host_formal_replay)
    status: dict[str, object] = {
        "schema": FINAL_STATUS_SCHEMA,
        "status": "PASS_EXACT_STAGED_COMMIT_B_PUBLICATION",
        "basis_commit_sha1": basis_commit,
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "actor_receipt_sha256": validated["receipt_sha256"],
        "finalize_actor_receipt_sha256": final_actor.receipt["receipt_sha256"],
        "finalize_actor_receipt": dict(final_actor.receipt),
        "staged_audit_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "final_staged_path_count": len(all_paths),
        "formal_host_replay": formal,
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
            "commit_created_or_pushed": False,
        },
    }
    status["status_sha256"] = hashlib.sha256(canonical_json_v1(status)).hexdigest()
    return status


def verify_commit_b_publication_commit_v1(
    *,
    repository: Path = REPOSITORY_ROOT,
    basis_commit: str,
    publication_commit: str,
    finalize_receipt_path: Path,
) -> dict[str, object]:
    """Verify the committed B parent/tree without consulting worktree bytes."""

    root = _repository(repository)
    basis = _require_commit(basis_commit)
    commit_b = _require_commit(publication_commit)
    expected_actor_image_ref = _load_image_ref(root, basis)
    publication_tree, raw_parent = _raw_commit_tree_and_parent_v1(root, commit_b)
    if raw_parent != basis:
        _fail(FAIL_FINAL_STAGED_SET, "publication commit must have the basis commit as its sole parent")
    commit_rows = _raw_commit_rows(root, basis, publication_tree)
    paths = tuple(path for path, _mode in commit_rows)
    if not paths or AUDIT_RECEIPT_REPOSITORY_PATH not in paths:
        _fail(FAIL_FINAL_STAGED_SET, "publication commit lacks changed paths or audit receipt")
    _validate_exact_role_set(paths, exclude_receipt=False)
    try:
        validate_commit_b_changed_paths(
            paths,
            allowed_public_prefixes=ALLOWED_PUBLIC_PREFIXES,
            executable_prefixes=EXECUTABLE_PREFIXES,
        )
    except ExternalGenesisPreflightError as exc:
        _fail(FAIL_PATH_POLICY, f"publication commit path policy failed: {exc.code}")
    files: dict[str, bytes] = {}
    all_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    raw_tokens = tuple(
        value.encode("utf-8")
        for value in {root.as_posix(), PROJECT_ROOT.as_posix(), Path.home().as_posix()}
    )
    for path, diff_mode in commit_rows:
        tree = _git(root, ("ls-tree", publication_tree, "--", path)).rstrip(b"\n")
        try:
            metadata, tree_path = tree.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
        except (ValueError, UnicodeDecodeError):
            _fail(FAIL_FINAL_STAGED_SET, f"publication tree entry {path!r} is malformed")
        if (
            tree_path.decode("utf-8", "strict") != path
            or mode != "100644"
            or diff_mode != mode
            or object_type != "blob"
            or _LOWER_SHA1.fullmatch(object_id) is None
        ):
            _fail(FAIL_FILE_POLICY, f"publication tree entry {path!r} is not mode-100644 blob")
        payload = _git(root, ("cat-file", "blob", object_id))
        _public_file_lint_v1(path, payload, raw_path_tokens=raw_tokens)
        files[path] = payload
        row = {
            "path": path,
            "role_id": PUBLICATION_ROLE_REGISTRY[path],
            "git_mode": mode,
            "index_blob_sha1": object_id,
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        all_rows.append(row)
        if path != AUDIT_RECEIPT_REPOSITORY_PATH:
            rows.append(row)
    rows.sort(key=lambda row: str(row["path"]))
    total = sum(int(row["byte_length"]) for row in rows)
    expected_manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "policy_id": POLICY_ID,
        "audit_phase": "PREPARE_EXCLUDING_RECEIPT",
        "basis_commit_sha1": basis,
        "changed_path_scope": "EXACT_GIT_INDEX_DIFF_FROM_BASIS_COMMIT",
        "allowed_public_prefixes": list(ALLOWED_PUBLIC_PREFIXES),
        "executable_prefixes": list(EXECUTABLE_PREFIXES),
        "excluded_self_output_repository_path": AUDIT_RECEIPT_REPOSITORY_PATH,
        "excluded_self_output_present_in_candidate": False,
        "path_role_registry": [
            {
                "path": path,
                "role_id": PUBLICATION_ROLE_REGISTRY[path],
                "required_cardinality": 1,
            }
            for path in _required_role_paths(exclude_receipt=True)
        ],
        "role_cardinalities": {
            PUBLICATION_ROLE_REGISTRY[path]: 1
            for path in _required_role_paths(exclude_receipt=True)
        },
        "candidate_files": rows,
        "candidate_file_count": len(rows),
        "candidate_total_byte_length": total,
        "candidate_inventory_sha256": _inventory_sha256(rows),
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        },
    }
    expected_manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_v1(expected_manifest)
    ).hexdigest()
    receipt_raw = files[AUDIT_RECEIPT_REPOSITORY_PATH]
    receipt = _plain_json(_strict_json(receipt_raw, path=AUDIT_RECEIPT_REPOSITORY_PATH))
    if type(receipt) is not dict or canonical_json_v1(receipt) != receipt_raw:
        _fail(FAIL_FINAL_STAGED_SET, "committed publication receipt is not canonical JSON")
    validated = validate_actor_receipt_v1(
        receipt,
        expected_manifest=expected_manifest,
        expected_request_sha256=None,
        actor_image_ref=expected_actor_image_ref,
    )
    formal = _host_strict_replay_formal_public_payloads_v1(
        files,
        basis_commit=basis,
        require_formal_payloads=True,
        repository=root,
        commit_only=True,
    )
    all_rows.sort(key=lambda row: str(row["path"]))
    final_manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "policy_id": POLICY_ID,
        "audit_phase": "FINALIZE_INCLUDING_RECEIPT",
        "basis_commit_sha1": basis,
        "changed_path_scope": "EXACT_GIT_INDEX_DIFF_FROM_BASIS_COMMIT",
        "allowed_public_prefixes": list(ALLOWED_PUBLIC_PREFIXES),
        "executable_prefixes": list(EXECUTABLE_PREFIXES),
        "excluded_self_output_repository_path": AUDIT_RECEIPT_REPOSITORY_PATH,
        "excluded_self_output_present_in_candidate": True,
        "path_role_registry": [
            {
                "path": path,
                "role_id": PUBLICATION_ROLE_REGISTRY[path],
                "required_cardinality": 1,
            }
            for path in _required_role_paths(exclude_receipt=False)
        ],
        "role_cardinalities": {
            PUBLICATION_ROLE_REGISTRY[path]: 1
            for path in _required_role_paths(exclude_receipt=False)
        },
        "candidate_files": all_rows,
        "candidate_file_count": len(all_rows),
        "candidate_total_byte_length": sum(
            int(row["byte_length"]) for row in all_rows
        ),
        "candidate_inventory_sha256": _inventory_sha256(all_rows),
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
        },
    }
    final_manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_v1(final_manifest)
    ).hexdigest()
    final_raw = _read_repo_external_private_file_v1(root, finalize_receipt_path)
    final_value = _plain_json(_strict_json(final_raw, path="<repo-external-finalize-receipt>"))
    if type(final_value) is not dict or canonical_json_v1(final_value) != final_raw:
        _fail(FAIL_FINAL_STAGED_SET, "repo-external finalize receipt is not canonical JSON")
    final_body = dict(final_value)
    final_hash = final_body.pop("status_sha256", None)
    final_actor_value = final_value.get("finalize_actor_receipt")
    final_exact_keys = {
        "schema", "status", "basis_commit_sha1", "candidate_manifest_sha256",
        "actor_receipt_sha256", "finalize_actor_receipt_sha256",
        "finalize_actor_receipt", "staged_audit_receipt_sha256",
        "final_staged_path_count", "formal_host_replay", "authority_boundary",
        "status_sha256",
    }
    if (
        set(final_value) != final_exact_keys
        or final_value.get("schema") != FINAL_STATUS_SCHEMA
        or final_value.get("status") != "PASS_EXACT_STAGED_COMMIT_B_PUBLICATION"
        or final_value.get("basis_commit_sha1") != basis
        or type(final_hash) is not str
        or hashlib.sha256(canonical_json_v1(final_body)).hexdigest() != final_hash
        or not isinstance(final_actor_value, Mapping)
    ):
        _fail(FAIL_FINAL_STAGED_SET, "repo-external finalize status fields/self-hash differ")
    final_actor = validate_actor_receipt_v1(
        final_actor_value,
        expected_manifest=final_manifest,
        expected_request_sha256=None,
        actor_image_ref=expected_actor_image_ref,
    )
    if (
        final_value.get("candidate_manifest_sha256") != final_manifest["manifest_sha256"]
        or final_value.get("actor_receipt_sha256") != validated["receipt_sha256"]
        or final_value.get("finalize_actor_receipt_sha256") != final_actor["receipt_sha256"]
        or final_value.get("staged_audit_receipt_sha256") != hashlib.sha256(receipt_raw).hexdigest()
        or final_value.get("final_staged_path_count") != len(paths)
        or final_value.get("formal_host_replay") != formal
        or final_value.get("authority_boundary") != {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "creates_seed_key_signature_marker_or_formal_root": False,
            "m3_start_or_state_transition": False,
            "commit_created_or_pushed": False,
        }
    ):
        _fail(FAIL_FINAL_STAGED_SET, "finalize receipt differs from committed parent/tree replay")
    status: dict[str, object] = {
        "schema": "hegel-phase3-m25-commit-b-publication-commit-verification/1",
        "status": "PASS_COMMIT_B_PARENT_AND_TREE",
        "basis_commit_sha1": basis,
        "publication_commit_sha1": commit_b,
        "sole_parent_verified": True,
        "changed_path_count": len(paths),
        "audit_receipt_sha256": validated["receipt_sha256"],
        "finalize_receipt_sha256": hashlib.sha256(final_raw).hexdigest(),
        "finalize_actor_receipt_sha256": final_actor["receipt_sha256"],
        "finalize_tree_inventory_equal": True,
        "formal_host_replay": formal,
        "authority_boundary": {
            "diagnostic_publication_control_only": True,
            "formal_gate_delta": 0,
            "m3_start_or_state_transition": False,
            "commit_or_push_mutation_performed": False,
        },
    }
    status["status_sha256"] = hashlib.sha256(canonical_json_v1(status)).hexdigest()
    return status


__all__ = [
    "ALLOWED_PUBLIC_PREFIXES",
    "AUDIT_RECEIPT_REPOSITORY_PATH",
    "EXTERNAL_STATUS_REPOSITORY_PATH",
    "CommitBActorAuditResultV1",
    "CommitBPublicationAuditError",
    "EXECUTABLE_PREFIXES",
    "FAIL_ACTOR_POLICY",
    "FAIL_ACTOR_RESPONSE",
    "FAIL_FILE_POLICY",
    "FAIL_FINAL_STAGED_SET",
    "FAIL_FORMAL_REPLAY",
    "FAIL_GIT_INDEX",
    "FAIL_JSON_POLICY",
    "FAIL_PATH_POLICY",
    "FAIL_SECRET_POLICY",
    "FINAL_STATUS_SCHEMA",
    "FORMAL_PUBLIC_PATHS",
    "MANIFEST_SCHEMA",
    "POLICY_ID",
    "RECEIPT_SCHEMA",
    "build_actor_request_v1",
    "build_external_status_from_worktree_v1",
    "build_staged_candidate_manifest_v1",
    "canonical_json_v1",
    "finalize_staged_commit_b_publication_v1",
    "publication_actor_container_command_v1",
    "run_commit_b_publication_actor_audit_v1",
    "validate_actor_receipt_v1",
    "verify_commit_b_publication_commit_v1",
]
