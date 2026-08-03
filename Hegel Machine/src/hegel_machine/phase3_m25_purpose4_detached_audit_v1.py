"""Detached, offline purpose-4 replay for the frozen Gate-17 parent audit.

The host-side adapter in this module creates a self-contained Git object
snapshot containing *only* objects reachable from the frozen audited parent.
The snapshot has no remote, alternate object directory, shallow boundary,
promisor pack, graft, or replacement ref.  A purpose-4 container then runs the
existing :mod:`phase3_m25_parent_absence_audit_v1` generator against that
read-only snapshot.  It receives no host-generated audit rows.

This module never creates a key, seed, signature, marker, or formal gate.  Its
only actor output is a compact public audit receipt and the exact public
purpose-4 signing request that a later ceremony step may authorize.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Final, Mapping, NoReturn, Sequence

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

from .phase3_m25_parent_absence_audit_v1 import (
    CONTENT_PREDICATE_PROFILE_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    PATH_ALIAS_RULE_ID,
    PATH_NAME_PREDICATE_PROFILE_ID,
    PUBLIC_RECEIPT_SCHEMA_ID,
    TOUCHED_PATH_RULE_ID,
)
from .phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    LEGACY_PARENT_SOURCE_IDS,
    OBJECT_TAGS,
    candidate_content_root,
    decode_formal_object,
    encode_formal_object,
    external_signature_preimage_v1,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase3_container_actor_profile_v1.json"
SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
WORKER_PATH: Final = PROJECT_ROOT / "tools/phase3_m25_purpose4_detached_audit_worker_v1.py"
PARENT_MODULE_PATH: Final = (
    PROJECT_ROOT / "src/hegel_machine/phase3_m25_parent_absence_audit_v1.py"
)
WIRE_MODULE_PATH: Final = PROJECT_ROOT / "src/hegel_machine/phase3_m25_wire_v1.py"
CBOR_MODULE_PATH: Final = PROJECT_ROOT / "src/hegel_machine/strict_cbor_v1.py"
PROBE_PATH: Final = PROJECT_ROOT / "tools/phase3_container_actor_probe_v1.py"

SNAPSHOT_SCHEMA: Final = "hegel-gate17-detached-git-snapshot/1"
ACTOR_REQUEST_SCHEMA: Final = "hegel-gate17-purpose4-detached-request/1"
ACTOR_RESPONSE_SCHEMA: Final = "hegel-gate17-purpose4-detached-response/1"
AUDITED_REF: Final = "refs/hegel/audited-parent"
OBJECT_INVENTORY_DOMAIN: Final = (
    b"HEGEL/GATE17/DETACHED_SNAPSHOT_OBJECT_INVENTORY/V1\x00"
)
FILE_INVENTORY_DOMAIN: Final = (
    b"HEGEL/GATE17/DETACHED_SNAPSHOT_FILE_INVENTORY/V1\x00"
)
RUNTIME_INVENTORY_DOMAIN: Final = (
    b"HEGEL/GATE17/PURPOSE4_RUNTIME_INVENTORY/V1\x00"
)
RUNTIME_SOURCE_BINDING_DOMAIN: Final = (
    b"HEGEL/GATE17/PURPOSE4_RUNTIME_SOURCE_BINDINGS/V1\x00"
)
PYTHON_IMAGE_KEY: Final = "policy_auditor"
MAX_ACTOR_OUTPUT_BYTES: Final = 2 * 1024 * 1024
DEFAULT_TEMPORARY_PARENT: Final = DEFAULT_LINUX_LOCAL_RUNTIME_PARENT
LIVE_PROBE_SYSCALL_IDS: Final = (
    "socket(AF_INET, SOCK_STREAM)",
    "socket(AF_INET6, SOCK_STREAM)",
    "mount",
    "ptrace(PTRACE_TRACEME)",
    "bpf(BPF_MAP_CREATE)",
    "perf_event_open",
)

FAIL_SNAPSHOT_BUILD: Final = "FAIL_GATE17_DETACHED_SNAPSHOT_BUILD"
FAIL_SNAPSHOT_POLICY: Final = "FAIL_GATE17_DETACHED_SNAPSHOT_POLICY"
FAIL_SNAPSHOT_INVENTORY: Final = "FAIL_GATE17_DETACHED_SNAPSHOT_INVENTORY"
FAIL_GIT_RUNTIME_BINDING: Final = "FAIL_GATE17_GIT_RUNTIME_BINDING"
FAIL_RUNTIME_BASIS: Final = "FAIL_GATE17_PURPOSE4_RUNTIME_BASIS"
FAIL_ACTOR_POLICY: Final = "FAIL_GATE17_PURPOSE4_ACTOR_POLICY"
FAIL_ACTOR_RESPONSE: Final = "FAIL_GATE17_PURPOSE4_ACTOR_RESPONSE"
FAIL_RECEIPT_INCOMPLETE: Final = "FAIL_GATE17_PURPOSE4_RECEIPT_INCOMPLETE"
FAIL_LOCAL_RUNTIME: Final = "FAIL_GATE17_LOCAL_RUNTIME"

_LOWER_SHA1 = re.compile(r"[0-9a-f]{40}")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


class Purpose4DetachedAuditError(RuntimeError):
    """Stable fail-closed error at the detached purpose-4 boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Purpose4DetachedAuditError(code, detail)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_environment(git_executable: Path) -> dict[str, str]:
    environment = {
        "LC_ALL": "C",
        "LANG": "C",
        "PATH": str(git_executable.parent) + ":/usr/bin:/bin",
        "HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_SSH_COMMAND": "false",
    }
    return environment


def _run_git(
    git_executable: Path,
    repository: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 300,
) -> bytes:
    try:
        safe_repository = repository.resolve(strict=True)
    except OSError as exc:
        _fail(
            FAIL_SNAPSHOT_BUILD,
            f"Git repository cannot be resolved: {type(exc).__name__}",
        )
    try:
        result = subprocess.run(
            [
                str(git_executable),
                "-c",
                "core.quotePath=false",
                "-c",
                f"safe.directory={safe_repository}",
                *arguments,
            ],
            cwd=safe_repository,
            env=_git_environment(git_executable),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_SNAPSHOT_BUILD, f"Git command could not complete: {type(exc).__name__}")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "backslashreplace")[-1000:].strip()
        _fail(
            FAIL_SNAPSHOT_BUILD,
            f"Git command {arguments[0]!r} exited {result.returncode}: {detail}",
        )
    return result.stdout


def _resolve_git_executable(path: Path | None = None) -> Path:
    candidate = path
    if candidate is None:
        candidate = Path("/usr/bin/git")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_GIT_RUNTIME_BINDING, f"Git executable cannot be resolved: {exc}")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        _fail(FAIL_GIT_RUNTIME_BINDING, "Git executable is not an executable regular file")
    return resolved


def _git_version(git_executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(git_executable), "--version"],
            env=_git_environment(git_executable),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_GIT_RUNTIME_BINDING, f"Git version query failed: {type(exc).__name__}")
    try:
        value = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError:
        _fail(FAIL_GIT_RUNTIME_BINDING, "Git version is not ASCII")
    if result.returncode != 0 or result.stderr or not value.startswith("git version "):
        _fail(FAIL_GIT_RUNTIME_BINDING, "Git version query is not canonical")
    return value


def git_runtime_binding_v1(git_executable: Path | None = None) -> dict[str, object]:
    """Bind the exact executable used to build and replay the snapshot."""

    executable = _resolve_git_executable(git_executable)
    return {
        "container_path": "/runtime/bin/git",
        "byte_length": executable.stat().st_size,
        "sha256": _sha256_file(executable),
        "version": _git_version(executable),
    }


def _require_parent(value: bytes, *, frozen: bool) -> bytes:
    if type(value) is not bytes or len(value) != 20:
        _fail(FAIL_SNAPSHOT_POLICY, "audited parent must be exactly 20 bytes")
    if frozen and value != AUDITED_PARENT_COMMIT_SHA1:
        _fail(FAIL_SNAPSHOT_POLICY, "formal purpose-4 replay requires the frozen parent")
    return value


def _require_basis_commit(value: str) -> str:
    if type(value) is not str or _LOWER_SHA1.fullmatch(value) is None:
        _fail(FAIL_RUNTIME_BASIS, "basis commit must be a lowercase Git SHA-1")
    return value


def _snapshot_git_dir(snapshot: Path) -> Path:
    root = snapshot.resolve()
    git_dir = root / ".git"
    if set(path.name for path in root.iterdir()) != {".git"} or not git_dir.is_dir():
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot worktree must contain only .git")
    return git_dir


def _reject_external_object_dependencies(git_dir: Path) -> None:
    forbidden_files = (
        git_dir / "shallow",
        git_dir / "objects/info/alternates",
        git_dir / "objects/info/http-alternates",
        git_dir / "info/grafts",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden_files):
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot has a shallow/alternate/graft dependency")
    if (git_dir / "refs/replace").exists():
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot contains replacement refs")
    if any(git_dir.glob("objects/pack/*.promisor")):
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot contains a promisor pack")
    config = (git_dir / "config").read_text(encoding="utf-8", errors="strict").lower()
    forbidden_config_tokens = (
        "promisor",
        "partialclone",
        "alternaterefscommand",
        "[include]",
        "[includeif ",
        "[remote \"",
        "insteadof",
        "url =",
        "pushurl =",
    )
    if any(token in config for token in forbidden_config_tokens):
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot config names an external or promisor dependency")


def _reachable_object_rows(
    snapshot: Path, git_executable: Path, parent: bytes
) -> tuple[tuple[bytes, str, int], ...]:
    parent_hex = parent.hex()
    reachable_raw = _run_git(
        git_executable,
        snapshot,
        ("rev-list", "--objects", "--no-object-names", parent_hex),
    )
    try:
        reachable = tuple(
            sorted({bytes.fromhex(line.decode("ascii")) for line in reachable_raw.splitlines()})
        )
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(FAIL_SNAPSHOT_INVENTORY, f"reachable object list is malformed: {exc}")
    if not reachable or any(len(digest) != 20 for digest in reachable):
        _fail(FAIL_SNAPSHOT_INVENTORY, "reachable object set is empty or malformed")

    all_raw = _run_git(
        git_executable,
        snapshot,
        ("cat-file", "--batch-all-objects", "--batch-check=%(objectname)"),
    )
    try:
        all_objects = tuple(
            sorted({bytes.fromhex(line.decode("ascii")) for line in all_raw.splitlines()})
        )
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(FAIL_SNAPSHOT_INVENTORY, f"snapshot object list is malformed: {exc}")
    if all_objects != reachable:
        _fail(FAIL_SNAPSHOT_INVENTORY, "snapshot contains missing or non-parent-reachable objects")

    request = b"".join(digest.hex().encode("ascii") + b"\n" for digest in reachable)
    metadata = _run_git(
        git_executable,
        snapshot,
        ("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"),
        input_bytes=request,
    ).splitlines()
    if len(metadata) != len(reachable):
        _fail(FAIL_SNAPSHOT_INVENTORY, "object metadata row count differs")
    rows: list[tuple[bytes, str, int]] = []
    for expected, line in zip(reachable, metadata, strict=True):
        parts = line.split(b" ")
        if len(parts) != 3:
            _fail(FAIL_SNAPSHOT_INVENTORY, "object metadata is malformed")
        try:
            returned = bytes.fromhex(parts[0].decode("ascii"))
            kind = parts[1].decode("ascii")
            size = int(parts[2])
        except (UnicodeDecodeError, ValueError) as exc:
            _fail(FAIL_SNAPSHOT_INVENTORY, f"object metadata is malformed: {exc}")
        if returned != expected or kind not in {"blob", "tree", "commit", "tag"} or size < 0:
            _fail(FAIL_SNAPSHOT_INVENTORY, "object identity/type/size differs")
        rows.append((expected, kind, size))
    return tuple(rows)


def _object_inventory(rows: Sequence[tuple[bytes, str, int]]) -> dict[str, object]:
    digest = hashlib.sha256(OBJECT_INVENTORY_DOMAIN)
    counts = {kind: 0 for kind in ("blob", "tree", "commit", "tag")}
    total = 0
    for object_id, kind, size in rows:
        encoded_kind = kind.encode("ascii")
        digest.update(object_id)
        digest.update(len(encoded_kind).to_bytes(1, "big"))
        digest.update(encoded_kind)
        digest.update(size.to_bytes(8, "big"))
        counts[kind] += 1
        total += size
    return {
        "reachable_object_count": len(rows),
        "reachable_object_counts_by_type": counts,
        "reachable_inflated_byte_length": total,
        "reachable_object_inventory_sha256": digest.hexdigest(),
    }


def _snapshot_file_inventory(git_dir: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    digest = hashlib.sha256(FILE_INVENTORY_DOMAIN)
    for path in sorted(git_dir.rglob("*"), key=lambda item: item.relative_to(git_dir).as_posix()):
        if path.is_symlink():
            _fail(FAIL_SNAPSHOT_POLICY, "snapshot must not contain symlinks")
        if path.is_dir():
            continue
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(FAIL_SNAPSHOT_POLICY, "snapshot must contain only directories and regular files")
        relative = path.relative_to(git_dir).as_posix()
        raw_path = relative.encode("utf-8")
        file_digest = _sha256_file(path)
        row = {
            "path": relative,
            "byte_length": metadata.st_size,
            "sha256": file_digest,
        }
        rows.append(row)
        digest.update(len(raw_path).to_bytes(4, "big"))
        digest.update(raw_path)
        digest.update(metadata.st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest))
    return rows, digest.hexdigest()


def _manifest_body(
    snapshot: Path,
    *,
    git_executable: Path,
    parent: bytes,
    basis_commit: str,
) -> dict[str, object]:
    basis_commit = _require_basis_commit(basis_commit)
    git_dir = _snapshot_git_dir(snapshot)
    _reject_external_object_dependencies(git_dir)
    if _run_git(git_executable, snapshot, ("rev-parse", "--show-object-format")).strip() != b"sha1":
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot object format is not sha1")
    if _run_git(git_executable, snapshot, ("rev-parse", "--is-shallow-repository")).strip() != b"false":
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot is shallow")
    if _run_git(
        git_executable,
        snapshot,
        ("for-each-ref", "--format=%(refname)", "refs/replace"),
    ).strip():
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot contains a packed replacement ref")
    expected = parent.hex().encode("ascii")
    if _run_git(git_executable, snapshot, ("rev-parse", f"{AUDITED_REF}^{{commit}}")).strip() != expected:
        _fail(FAIL_SNAPSHOT_POLICY, "audited parent ref differs")
    if _run_git(git_executable, snapshot, ("rev-parse", "HEAD^{commit}")).strip() != expected:
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot HEAD differs from audited parent")
    _run_git(git_executable, snapshot, ("fsck", "--full", "--strict", "--no-reflogs", parent.hex()), timeout=900)
    objects = _reachable_object_rows(snapshot, git_executable, parent)
    files, file_digest = _snapshot_file_inventory(git_dir)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "basis_commit_sha1": basis_commit,
        "audited_parent_commit_sha1": parent.hex(),
        "audited_parent_ref": AUDITED_REF,
        "head_is_audited_parent": True,
        "object_format": "sha1",
        "shallow_repository": False,
        "alternate_object_directories_present": False,
        "promisor_or_partial_clone_present": False,
        "grafts_or_replace_refs_present": False,
        "remote_configuration_present": False,
        "worktree_payload_entry_count": 0,
        "git_runtime_binding": git_runtime_binding_v1(git_executable),
        **_object_inventory(objects),
        "snapshot_file_count": len(files),
        "snapshot_file_inventory": files,
        "snapshot_file_inventory_sha256": file_digest,
        "container_mount_read_only_required": True,
    }


def validate_detached_parent_snapshot_v1(
    snapshot: Path,
    manifest: Mapping[str, object],
    *,
    git_executable: Path | None = None,
    require_frozen_parent: bool = True,
    expected_basis_commit: str | None = None,
) -> dict[str, object]:
    """Recompute every snapshot binding and reject hidden object dependencies."""

    executable = _resolve_git_executable(git_executable)
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        _fail(FAIL_SNAPSHOT_INVENTORY, "snapshot manifest schema differs")
    parent_hex = manifest.get("audited_parent_commit_sha1")
    if type(parent_hex) is not str or _LOWER_SHA1.fullmatch(parent_hex) is None:
        _fail(FAIL_SNAPSHOT_INVENTORY, "snapshot parent ID is malformed")
    parent = _require_parent(bytes.fromhex(parent_hex), frozen=require_frozen_parent)
    basis_commit = _require_basis_commit(str(manifest.get("basis_commit_sha1")))
    if expected_basis_commit is not None and basis_commit != _require_basis_commit(
        expected_basis_commit
    ):
        _fail(FAIL_RUNTIME_BASIS, "snapshot basis commit differs from request")
    supplied = dict(manifest)
    claimed = supplied.pop("manifest_sha256", None)
    if type(claimed) is not str or _LOWER_SHA256.fullmatch(claimed) is None:
        _fail(FAIL_SNAPSHOT_INVENTORY, "snapshot manifest digest is malformed")
    if hashlib.sha256(_canonical_json(supplied)).hexdigest() != claimed:
        _fail(FAIL_SNAPSHOT_INVENTORY, "snapshot manifest self-digest differs")
    rebuilt = _manifest_body(
        snapshot.resolve(),
        git_executable=executable,
        parent=parent,
        basis_commit=basis_commit,
    )
    if rebuilt != supplied:
        _fail(FAIL_SNAPSHOT_INVENTORY, "detached snapshot differs from its exact inventory")
    return {**rebuilt, "manifest_sha256": claimed}


def _remove_init_noise(git_dir: Path) -> None:
    hooks = git_dir / "hooks"
    if hooks.exists():
        shutil.rmtree(hooks)
    for path in (git_dir / "description", git_dir / "info/exclude"):
        if path.exists():
            path.unlink()
    info = git_dir / "info"
    if info.exists() and not any(info.iterdir()):
        info.rmdir()
    logs = git_dir / "logs"
    if logs.exists():
        shutil.rmtree(logs)


def _set_snapshot_read_only(root: Path, read_only: bool) -> None:
    directory_mode = 0o555 if read_only else 0o700
    file_mode = 0o444 if read_only else 0o600
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=not read_only)
    if not read_only:
        root.chmod(directory_mode)
    for path in paths:
        if path.is_symlink():
            _fail(FAIL_SNAPSHOT_POLICY, "snapshot contains a symlink")
        path.chmod(directory_mode if path.is_dir() else file_mode)
    if read_only:
        root.chmod(directory_mode)


@dataclass(slots=True)
class DetachedParentSnapshotV1(AbstractContextManager["DetachedParentSnapshotV1"]):
    root: Path
    manifest: Mapping[str, object]
    git_executable: Path
    _temporary: LinuxLocalTemporaryDirectoryV1 | None

    def close(self) -> None:
        if self._temporary is not None:
            _set_snapshot_read_only(self.root, False)
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "DetachedParentSnapshotV1":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _prepare_snapshot(
    repository: Path,
    *,
    parent: bytes,
    basis_commit: str,
    git_executable: Path,
    temporary_parent: Path | None,
    require_frozen_parent: bool,
) -> DetachedParentSnapshotV1:
    parent = _require_parent(parent, frozen=require_frozen_parent)
    basis_commit = _require_basis_commit(basis_commit)
    source = repository.resolve(strict=True)
    resolved_basis = _run_git(
        git_executable, source, ("rev-parse", f"{basis_commit}^{{commit}}")
    ).strip()
    if resolved_basis != basis_commit.encode("ascii"):
        _fail(FAIL_RUNTIME_BASIS, "source repository resolves another basis commit")
    temporary_root = (
        DEFAULT_TEMPORARY_PARENT if temporary_parent is None else temporary_parent
    ).resolve(strict=True)
    try:
        temporary_root.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        pass
    else:
        _fail(FAIL_SNAPSHOT_POLICY, "snapshot temporary parent must be outside the repository")
    try:
        temporary = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-gate17-detached-",
            repository_root=REPOSITORY_ROOT,
            parent=temporary_root,
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_LOCAL_RUNTIME, f"{exc.code}: {exc.detail}")
    root = Path(temporary.name) / "snapshot"
    root.mkdir(mode=0o700)
    try:
        _run_git(git_executable, root, ("init", "--quiet", "--initial-branch=unused"))
        git_dir = root / ".git"
        _run_git(git_executable, root, ("config", "core.logAllRefUpdates", "false"))
        _run_git(git_executable, root, ("config", "gc.auto", "0"))
        resolved = _run_git(
            git_executable,
            source,
            ("rev-parse", f"{parent.hex()}^{{commit}}"),
        ).strip()
        if resolved != parent.hex().encode("ascii"):
            _fail(FAIL_SNAPSHOT_BUILD, "source repository resolves another audited parent")
        pack_prefix = git_dir / "objects/pack/pack"
        _run_git(
            git_executable,
            source,
            (
                "pack-objects",
                "--revs",
                "--delta-base-offset",
                str(pack_prefix),
            ),
            input_bytes=parent.hex().encode("ascii") + b"\n",
            timeout=1800,
        )
        _run_git(
            git_executable,
            root,
            ("update-ref", AUDITED_REF, parent.hex()),
        )
        (git_dir / "HEAD").write_bytes(f"ref: {AUDITED_REF}\n".encode("ascii"))
        _remove_init_noise(git_dir)
        body = _manifest_body(
            root,
            git_executable=git_executable,
            parent=parent,
            basis_commit=basis_commit,
        )
        manifest = {
            **body,
            "manifest_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
        }
        validate_detached_parent_snapshot_v1(
            root,
            manifest,
            git_executable=git_executable,
            require_frozen_parent=require_frozen_parent,
            expected_basis_commit=basis_commit,
        )
        _set_snapshot_read_only(root, True)
        return DetachedParentSnapshotV1(root, manifest, git_executable, temporary)
    except BaseException:
        try:
            _set_snapshot_read_only(Path(temporary.name), False)
        except Exception:
            pass
        temporary.cleanup()
        raise


def prepare_detached_parent_snapshot_v1(
    repository: Path = REPOSITORY_ROOT,
    *,
    basis_commit: str,
    git_executable: Path | None = None,
    temporary_parent: Path | None = None,
) -> DetachedParentSnapshotV1:
    """Create a temporary, self-contained snapshot for the frozen parent."""

    return _prepare_snapshot(
        repository,
        parent=AUDITED_PARENT_COMMIT_SHA1,
        basis_commit=basis_commit,
        git_executable=_resolve_git_executable(git_executable),
        temporary_parent=temporary_parent,
        require_frozen_parent=True,
    )


def _runtime_inventory(runtime_root: Path) -> dict[str, object]:
    files = sorted(path for path in runtime_root.rglob("*") if path.is_file())
    digest = hashlib.sha256(RUNTIME_INVENTORY_DOMAIN)
    rows: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(runtime_root).as_posix()
        raw = relative.encode("utf-8")
        file_digest = _sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "byte_length": size, "sha256": file_digest})
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest))
    return {
        "files": rows,
        "file_count": len(rows),
        "inventory_sha256": digest.hexdigest(),
    }


def _git_blob_sha1(payload: bytes) -> str:
    preimage = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(preimage).hexdigest()


def _committed_source_binding_v1(
    source: Path,
    runtime_path: str,
    *,
    basis_commit: str,
    git_executable: Path,
    repository: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Prove that one actor source is the exact blob in ``basis_commit``."""

    basis_commit = _require_basis_commit(basis_commit)
    repository = repository.resolve(strict=True)
    source = source.resolve(strict=True)
    try:
        relative = source.relative_to(repository).as_posix()
    except ValueError:
        _fail(FAIL_RUNTIME_BASIS, "runtime source is outside the basis repository")
    raw_tree = _run_git(
        git_executable,
        repository,
        ("ls-tree", "-z", basis_commit, "--", relative),
    )
    rows = [row for row in raw_tree.split(b"\0") if row]
    if len(rows) != 1:
        _fail(FAIL_RUNTIME_BASIS, f"basis commit has no unique blob for {relative}")
    header, separator, raw_path = rows[0].partition(b"\t")
    try:
        mode, object_type, object_id = header.decode("ascii").split(" ")
        tree_path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        _fail(FAIL_RUNTIME_BASIS, f"basis tree row is malformed for {relative}")
    if (
        not separator
        or mode not in {"100644", "100755"}
        or object_type != "blob"
        or _LOWER_SHA1.fullmatch(object_id) is None
        or tree_path != relative
    ):
        _fail(FAIL_RUNTIME_BASIS, f"basis tree row differs for {relative}")
    committed = _run_git(
        git_executable, repository, ("cat-file", "blob", object_id)
    )
    worktree = source.read_bytes()
    worktree_blob = _git_blob_sha1(worktree)
    if committed != worktree or object_id != worktree_blob:
        _fail(
            FAIL_RUNTIME_BASIS,
            f"uncommitted runtime source bytes are forbidden: {relative}",
        )
    return {
        "runtime_path": runtime_path,
        "repository_path": relative,
        "basis_tree_mode": mode,
        "basis_tree_blob_sha1": object_id,
        "byte_length": len(worktree),
        "sha256": hashlib.sha256(worktree).hexdigest(),
    }


def _runtime_source_bindings_v1(
    source_specs: Sequence[tuple[Path, str]],
    *,
    basis_commit: str,
    git_executable: Path,
    repository: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    basis_commit = _require_basis_commit(basis_commit)
    rows = [
        _committed_source_binding_v1(
            source,
            runtime_path,
            basis_commit=basis_commit,
            git_executable=git_executable,
            repository=repository,
        )
        for source, runtime_path in source_specs
    ]
    if len({str(row["runtime_path"]) for row in rows}) != len(rows):
        _fail(FAIL_RUNTIME_BASIS, "runtime source destinations are not unique")
    body: dict[str, object] = {
        "schema": "hegel-gate17-purpose4-runtime-source-bindings/1",
        "basis_commit_sha1": basis_commit,
        "committed_source_files": rows,
        "external_git_dependency": git_runtime_binding_v1(git_executable),
    }
    body["binding_sha256"] = hashlib.sha256(
        RUNTIME_SOURCE_BINDING_DOMAIN + _canonical_json(body)
    ).hexdigest()
    return body


def _actor_source_specs() -> tuple[tuple[Path, str], ...]:
    return (
        (Path(__file__), f"hegel_machine/{Path(__file__).name}"),
        (PARENT_MODULE_PATH, f"hegel_machine/{PARENT_MODULE_PATH.name}"),
        (WIRE_MODULE_PATH, f"hegel_machine/{WIRE_MODULE_PATH.name}"),
        (CBOR_MODULE_PATH, f"hegel_machine/{CBOR_MODULE_PATH.name}"),
        (WORKER_PATH, "worker.py"),
        (PROBE_PATH, "probe.py"),
        (
            PROFILE_PATH,
            "control/phase3_container_actor_profile_v1.json",
        ),
        (
            SECCOMP_PATH,
            "control/phase3_internal_actor_seccomp_v1.json",
        ),
    )


def _copy_actor_runtime(
    runtime_root: Path,
    git_executable: Path,
    *,
    basis_commit: str,
) -> dict[str, object]:
    package = runtime_root / "hegel_machine"
    binary = runtime_root / "bin/git"
    package.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    source_specs = _actor_source_specs()
    source_bindings = _runtime_source_bindings_v1(
        source_specs,
        basis_commit=basis_commit,
        git_executable=git_executable,
    )
    for source, runtime_path in source_specs:
        destination = runtime_root / runtime_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(git_executable, binary)
    for path in runtime_root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else (0o555 if path == binary else 0o444))
    runtime_root.chmod(0o555)
    return {
        "runtime_inventory": _runtime_inventory(runtime_root),
        "runtime_source_bindings": source_bindings,
    }


def _require_image_ref(value: str) -> str:
    if type(value) is not str or re.fullmatch(
        r"[^@\s]+@sha256:[0-9a-f]{64}", value
    ) is None:
        _fail(FAIL_ACTOR_POLICY, "purpose-4 image is not digest pinned")
    return value


def _validate_runtime_source_bindings_v1(
    value: Mapping[str, object],
    *,
    basis_commit: str,
    runtime_inventory: Mapping[str, object],
    git_binding: Mapping[str, object],
) -> dict[str, object]:
    basis_commit = _require_basis_commit(basis_commit)
    body = dict(value)
    claimed = body.pop("binding_sha256", None)
    if type(claimed) is not str or _LOWER_SHA256.fullmatch(claimed) is None:
        _fail(FAIL_RUNTIME_BASIS, "runtime source-binding digest is malformed")
    if hashlib.sha256(
        RUNTIME_SOURCE_BINDING_DOMAIN + _canonical_json(body)
    ).hexdigest() != claimed:
        _fail(FAIL_RUNTIME_BASIS, "runtime source-binding digest differs")
    if set(body) != {
        "schema",
        "basis_commit_sha1",
        "committed_source_files",
        "external_git_dependency",
    }:
        _fail(FAIL_RUNTIME_BASIS, "runtime source-binding field set differs")
    if (
        body["schema"] != "hegel-gate17-purpose4-runtime-source-bindings/1"
        or body["basis_commit_sha1"] != basis_commit
        or body["external_git_dependency"] != dict(git_binding)
    ):
        _fail(FAIL_RUNTIME_BASIS, "runtime source basis or Git dependency differs")
    inventory_rows = runtime_inventory.get("files")
    source_rows = body["committed_source_files"]
    if type(inventory_rows) is not list or type(source_rows) is not list or not source_rows:
        _fail(FAIL_RUNTIME_BASIS, "runtime or source inventory is absent")
    inventory_by_path = {
        row.get("path"): row
        for row in inventory_rows
        if isinstance(row, Mapping) and type(row.get("path")) is str
    }
    exact_row_keys = {
        "runtime_path",
        "repository_path",
        "basis_tree_mode",
        "basis_tree_blob_sha1",
        "byte_length",
        "sha256",
    }
    seen: set[str] = set()
    for row in source_rows:
        if not isinstance(row, Mapping) or set(row) != exact_row_keys:
            _fail(FAIL_RUNTIME_BASIS, "committed runtime source row is malformed")
        runtime_path = row["runtime_path"]
        if (
            type(runtime_path) is not str
            or not runtime_path
            or runtime_path.startswith("/")
            or ".." in Path(runtime_path).parts
            or runtime_path in seen
            or type(row["repository_path"]) is not str
            or str(row["repository_path"]).startswith("/")
            or row["basis_tree_mode"] not in {"100644", "100755"}
            or type(row["basis_tree_blob_sha1"]) is not str
            or _LOWER_SHA1.fullmatch(str(row["basis_tree_blob_sha1"])) is None
            or type(row["byte_length"]) is not int
            or row["byte_length"] < 0
            or type(row["sha256"]) is not str
            or _LOWER_SHA256.fullmatch(str(row["sha256"])) is None
        ):
            _fail(FAIL_RUNTIME_BASIS, "committed runtime source row differs")
        seen.add(runtime_path)
        inventory_row = inventory_by_path.get(runtime_path)
        if not isinstance(inventory_row, Mapping) or (
            inventory_row.get("byte_length") != row["byte_length"]
            or inventory_row.get("sha256") != row["sha256"]
        ):
            _fail(FAIL_RUNTIME_BASIS, "runtime bytes differ from committed source binding")
    git_row = inventory_by_path.get("bin/git")
    if not isinstance(git_row, Mapping) or (
        git_row.get("byte_length") != git_binding.get("byte_length")
        or git_row.get("sha256") != git_binding.get("sha256")
    ):
        _fail(FAIL_GIT_RUNTIME_BINDING, "runtime Git differs from its external binding")
    return {**body, "binding_sha256": claimed}


def build_purpose4_actor_request_v1(
    *,
    snapshot_manifest: Mapping[str, object],
    runtime_inventory: Mapping[str, object],
    runtime_source_bindings: Mapping[str, object],
    basis_commit: str,
    actor_image_ref: str,
    auditor_key_id: bytes,
    audited_at_unix_seconds: int,
) -> dict[str, object]:
    basis_commit = _require_basis_commit(basis_commit)
    actor_image_ref = _require_image_ref(actor_image_ref)
    if type(auditor_key_id) is not bytes or len(auditor_key_id) != 16:
        _fail(FAIL_ACTOR_POLICY, "auditor key ID must be 16 bytes")
    if type(audited_at_unix_seconds) is not int or audited_at_unix_seconds < 0:
        _fail(FAIL_ACTOR_POLICY, "audit timestamp must be a nonnegative integer")
    if snapshot_manifest.get("basis_commit_sha1") != basis_commit:
        _fail(FAIL_RUNTIME_BASIS, "request and snapshot basis commits differ")
    git_binding = snapshot_manifest.get("git_runtime_binding")
    if not isinstance(git_binding, Mapping):
        _fail(FAIL_GIT_RUNTIME_BINDING, "snapshot Git binding is absent")
    validated_sources = _validate_runtime_source_bindings_v1(
        runtime_source_bindings,
        basis_commit=basis_commit,
        runtime_inventory=runtime_inventory,
        git_binding=git_binding,
    )
    request = {
        "schema": ACTOR_REQUEST_SCHEMA,
        "purpose_id": 4,
        "basis_commit_sha1": basis_commit,
        "actor_image_ref": actor_image_ref,
        "snapshot_manifest": dict(snapshot_manifest),
        "runtime_inventory": dict(runtime_inventory),
        "runtime_source_bindings": validated_sources,
        "auditor_key_id_hex": auditor_key_id.hex(),
        "audited_at_unix_seconds": audited_at_unix_seconds,
        "signature_generation_requested": False,
        "key_seed_or_marker_access_requested": False,
    }
    request["request_sha256"] = hashlib.sha256(_canonical_json(request)).hexdigest()
    return request


def _load_profile_image() -> str:
    try:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        image = profile["images"][PYTHON_IMAGE_KEY]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _fail(FAIL_ACTOR_POLICY, f"purpose-4 image profile is invalid: {exc}")
    return _require_image_ref(image)


def purpose4_container_command_v1(
    *,
    snapshot: Path,
    runtime: Path,
    request_path: Path,
    seccomp_path: Path,
    image_ref: str,
) -> tuple[str, ...]:
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
        f"--security-opt=seccomp={seccomp_path.resolve(strict=True)}",
        "--user=65534:65534",
        "--pids-limit=64",
        "--memory=512m",
        "--memory-swap=512m",
        "--ulimit=nofile=64:64",
        "--ipc=private",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
        f"--mount=type=bind,src={snapshot.resolve()},dst=/snapshot,readonly,bind-propagation=rprivate",
        f"--mount=type=bind,src={runtime.resolve()},dst=/runtime,readonly,bind-propagation=rprivate",
        f"--mount=type=bind,src={request_path.resolve()},dst=/request.json,readonly,bind-propagation=rprivate",
        "--entrypoint=/usr/bin/env",
        image_ref,
        "-i",
        "LC_ALL=C",
        "LANG=C",
        "PATH=/runtime/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "HEGEL_ACTOR_PROFILE_ID=hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_PURPOSE_ID=4",
        f"HEGEL_ACTOR_IMAGE_REF={image_ref}",
        "/usr/local/bin/python3",
        "-I",
        "-B",
        "/runtime/worker.py",
        "/snapshot",
        "/request.json",
    )


def _validate_complete_receipt(receipt: object, audit_root_hex: str) -> Mapping[str, object]:
    if not isinstance(receipt, Mapping):
        _fail(FAIL_RECEIPT_INCOMPLETE, "parent audit receipt is absent")
    required = {
        "schema_id": PUBLIC_RECEIPT_SCHEMA_ID,
        "audited_parent_commit_sha1": AUDITED_PARENT_COMMIT_SHA1.hex(),
        "audit_bundle_root": audit_root_hex,
        "all_predicates_absent": True,
        "authority_claim": False,
        "purpose_4_signature_present": False,
        "replay_requires_git_objects": True,
    }
    if any(receipt.get(name) != value for name, value in required.items()):
        _fail(FAIL_RECEIPT_INCOMPLETE, "parent path/audit receipt summary differs")
    predicates = receipt.get("predicates")
    expected_predicates = (
        (
            "typed_or_parent_binding_manifest",
            ["typed_binding_manifest", "parent_binding_manifest"],
        ),
        (
            "split_seed_commitment_or_allocation",
            [
                "split_seed_commitment",
                "split_seed_allocation",
                "split_assignment_manifest",
                "split_allocation_manifest",
            ],
        ),
        ("hidden_access_ledger", ["hidden_access_ledger"]),
    )
    empty_root = hashlib.sha256(b"").hexdigest()
    if not isinstance(predicates, list) or len(predicates) != 3:
        _fail(FAIL_RECEIPT_INCOMPLETE, "path absence predicates are incomplete")
    for row, (predicate_id, terms) in zip(predicates, expected_predicates, strict=True):
        if not isinstance(row, Mapping) or dict(row) != {
            "predicate_id": predicate_id,
            "normalized_substrings_ascii": terms,
            "matched_unique_path_count": 0,
            "matched_path_blob_row_count": 0,
            "matched_path_blob_tree_root": empty_root,
            "absent": True,
        }:
            _fail(FAIL_RECEIPT_INCOMPLETE, "a path absence predicate differs")
    content = receipt.get("content_blob_audit")
    if not isinstance(content, Mapping) or any(
        content.get(name) is not value
        for name, value in {
            "git_blob_object_id_and_size_verified": True,
            "all_content_absence_predicates_absent": True,
            "all_legacy_sources_present": True,
        }.items()
    ) or content.get("unscannable_relevant_structured_blob_count") != 0:
        _fail(FAIL_RECEIPT_INCOMPLETE, "content-level absence receipt is incomplete")
    if (
        content.get("content_predicate_profile_id") != CONTENT_PREDICATE_PROFILE_ID
        or content.get("inspected_path_blob_row_count") != receipt.get("audited_path_count")
        or type(content.get("inspected_unique_blob_count")) is not int
        or content["inspected_unique_blob_count"] <= 0
        or type(content.get("inspected_total_byte_length")) is not int
        or content["inspected_total_byte_length"] <= 0
        or type(content.get("structured_candidate_unique_blob_count")) is not int
        or content["structured_candidate_unique_blob_count"] <= 0
        or type(content.get("inspected_blob_inventory_sha256")) is not str
        or _LOWER_SHA256.fullmatch(content["inspected_blob_inventory_sha256"]) is None
    ):
        _fail(FAIL_RECEIPT_INCOMPLETE, "content inventory summary differs")
    content_absence = content.get("content_absence_predicates")
    if not isinstance(content_absence, list) or len(content_absence) != 3:
        _fail(FAIL_RECEIPT_INCOMPLETE, "content absence predicate set differs")
    for row, (predicate_id, _terms) in zip(
        content_absence, expected_predicates, strict=True
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("predicate_id") != predicate_id
            or row.get("match_occurrence_count") != 0
            or row.get("matching_unique_blob_count") != 0
            or row.get("matching_path_blob_row_count") != 0
            or row.get("matching_blob_digest_set_sha256") != empty_root
            or row.get("absent") is not True
        ):
            _fail(FAIL_RECEIPT_INCOMPLETE, "a content absence predicate differs")
    legacy = content.get("legacy_source_presence")
    if not isinstance(legacy, list) or len(legacy) != 2:
        _fail(FAIL_RECEIPT_INCOMPLETE, "legacy source presence set differs")
    for row, source_id in zip(legacy, LEGACY_PARENT_SOURCE_IDS, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("legacy_parent_payload_source_id") != source_id
            or type(row.get("match_occurrence_count")) is not int
            or row["match_occurrence_count"] <= 0
            or type(row.get("matching_unique_blob_count")) is not int
            or row["matching_unique_blob_count"] <= 0
            or type(row.get("matching_path_blob_row_count")) is not int
            or row["matching_path_blob_row_count"] <= 0
            or row.get("present") is not True
            or type(row.get("matching_blob_digest_set_sha256")) is not str
            or _LOWER_SHA256.fullmatch(row["matching_blob_digest_set_sha256"]) is None
        ):
            _fail(FAIL_RECEIPT_INCOMPLETE, "legacy source content evidence differs")
    diagnostic_body = {
        name: receipt[name]
        for name in (
            "schema_id",
            "audited_parent_commit_sha1",
            "parent_dsl_version",
            "parent_freeze_version",
            "touched_path_rule_id",
            "path_alias_rule_id",
            "path_name_predicate_profile_id",
            "audited_path_tree_root",
            "audit_bundle_root",
            "predicates",
            "all_predicates_absent",
            "content_blob_audit",
            "authority_claim",
            "purpose_4_signature_present",
        )
    }
    expected_static = {
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "touched_path_rule_id": TOUCHED_PATH_RULE_ID,
        "path_alias_rule_id": PATH_ALIAS_RULE_ID,
        "path_name_predicate_profile_id": PATH_NAME_PREDICATE_PROFILE_ID,
    }
    if any(receipt.get(name) != value for name, value in expected_static.items()):
        _fail(FAIL_RECEIPT_INCOMPLETE, "path receipt profile binding differs")
    diagnostic_digest = hashlib.sha256(
        json.dumps(
            diagnostic_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    if receipt.get("diagnostic_receipt_sha256") != diagnostic_digest:
        _fail(FAIL_RECEIPT_INCOMPLETE, "path/content diagnostic receipt digest differs")
    return receipt


def _validate_purpose4_live_probe_receipt_v1(
    value: object, *, actor_image_ref: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(FAIL_ACTOR_POLICY, "purpose-4 live-probe receipt is absent")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if type(claimed) is not str or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed:
        _fail(FAIL_ACTOR_POLICY, "purpose-4 live-probe receipt digest differs")
    exact_keys = {
        "schema",
        "profile_id",
        "purpose_id",
        "implementation",
        "actor_image_ref",
        "identity",
        "proc_status",
        "namespaces",
        "network_interfaces",
        "syscall_probes",
        "filesystem_probes",
        "environment",
        "open_fds",
        "cgroup_limits",
        "required_checks",
        "all_required_checks_passed",
    }
    if set(body) != exact_keys:
        _fail(FAIL_ACTOR_POLICY, "purpose-4 live-probe field set differs")
    identity = body["identity"]
    status = body["proc_status"]
    namespaces = body["namespaces"]
    syscall_rows = body["syscall_probes"]
    filesystem = body["filesystem_probes"]
    environment = body["environment"]
    cgroup = body["cgroup_limits"]
    checks = body["required_checks"]
    if (
        body["schema"] != "hegel-gate17-purpose4-live-probe/1"
        or body["profile_id"]
        != "hegel-owner-accepted-container-technical-actors-v1"
        or body["purpose_id"] != 4
        or body["implementation"] != "python-ctypes-v1"
        or body["actor_image_ref"] != actor_image_ref
        or not isinstance(identity, Mapping)
        or identity.get("uid") != 65534
        or identity.get("gid") != 65534
        or type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or not isinstance(status, Mapping)
        or status.get("NoNewPrivs") != 1
        or status.get("Seccomp") != 2
        or any(
            status.get(name) != "0000000000000000"
            for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        )
        or body["network_interfaces"] != ["lo"]
        or body["open_fds"] != [0, 1, 2]
        or body["all_required_checks_passed"] is not True
    ):
        _fail(FAIL_ACTOR_POLICY, "purpose-4 live runtime identity differs")
    if not isinstance(namespaces, Mapping) or set(namespaces) != {
        "pid",
        "mnt",
        "net",
        "ipc",
        "uts",
    } or not all(
        re.fullmatch(r"[a-z]+:\[[0-9]+\]", str(item))
        for item in namespaces.values()
    ):
        _fail(FAIL_ACTOR_POLICY, "purpose-4 namespace live probe differs")
    if type(syscall_rows) is not list or [
        row.get("probe_id") for row in syscall_rows if isinstance(row, Mapping)
    ] != list(LIVE_PROBE_SYSCALL_IDS) or any(
        not isinstance(row, Mapping)
        or row.get("return_value") != -1
        or row.get("errno") != 1
        for row in syscall_rows
    ):
        _fail(FAIL_ACTOR_POLICY, "purpose-4 seccomp syscall probes differ")
    if not isinstance(filesystem, Mapping) or any(
        not isinstance(filesystem.get(name), Mapping)
        or filesystem[name].get("denied") is not True  # type: ignore[union-attr]
        or filesystem[name].get("errno") not in {1, 13, 30}  # type: ignore[union-attr]
        for name in ("root_write", "snapshot_write", "runtime_write", "request_write")
    ) or filesystem.get("tmp_write") != {"denied": False, "errno": 0} or filesystem.get(
        "forbidden_paths_present"
    ) != [] or filesystem.get("cross_purpose_paths_present") != []:
        _fail(FAIL_ACTOR_POLICY, "purpose-4 filesystem live probes differ")
    expected_environment = {
        "HEGEL_ACTOR_IMAGE_REF": actor_image_ref,
        "HEGEL_ACTOR_PROFILE_ID": "hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_PURPOSE_ID": "4",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/runtime/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    if environment != expected_environment or cgroup != {
        "memory_max": str(512 * 1024 * 1024),
        "memory_swap_max": "0",
        "pids_max": "64",
    } or not isinstance(checks, Mapping) or not checks or not all(
        item is True for item in checks.values()
    ):
        _fail(FAIL_ACTOR_POLICY, "purpose-4 environment/resource probes differ")
    return {**body, "receipt_sha256": claimed}


def validate_purpose4_actor_response_v1(
    response: object,
    *,
    request: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(response, Mapping):
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 response is not an object")
    body = dict(response)
    claimed = body.pop("response_sha256", None)
    if type(claimed) is not str or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed:
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 response digest differs")
    exact_keys = {
        "schema",
        "purpose_id",
        "basis_commit_sha1",
        "actor_image_ref",
        "request_sha256",
        "snapshot_manifest_sha256",
        "runtime_inventory_sha256",
        "runtime_source_binding_sha256",
        "git_runtime_binding",
        "isolation_live_probe_receipt",
        "parent_absence_public_receipt",
        "attestation_cbor_hex",
        "attestation_root_hex",
        "signature_preimage_hex",
        "signer_purpose_id",
        "signer_key_epoch",
        "signature_present",
        "private_key_seed_marker_accessed",
        "network_access_performed",
    }
    if set(body) != exact_keys:
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 response field set differs")
    if (
        body["schema"] != ACTOR_RESPONSE_SCHEMA
        or body["purpose_id"] != 4
        or body["basis_commit_sha1"] != request.get("basis_commit_sha1")
        or body["actor_image_ref"] != request.get("actor_image_ref")
        or body["request_sha256"] != request.get("request_sha256")
        or body["snapshot_manifest_sha256"]
        != request["snapshot_manifest"]["manifest_sha256"]  # type: ignore[index]
        or body["runtime_inventory_sha256"]
        != request["runtime_inventory"]["inventory_sha256"]  # type: ignore[index]
        or body["runtime_source_binding_sha256"]
        != request["runtime_source_bindings"]["binding_sha256"]  # type: ignore[index]
        or body["signer_purpose_id"] != 4
        or body["signer_key_epoch"] != 0
        or body["signature_present"] is not False
        or body["private_key_seed_marker_accessed"] is not False
        or body["network_access_performed"] is not False
    ):
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 response policy fields differ")
    if body["git_runtime_binding"] != request["snapshot_manifest"]["git_runtime_binding"]:  # type: ignore[index]
        _fail(FAIL_GIT_RUNTIME_BINDING, "actor used another Git runtime")
    _validate_purpose4_live_probe_receipt_v1(
        body["isolation_live_probe_receipt"],
        actor_image_ref=str(body["actor_image_ref"]),
    )
    try:
        cbor = bytes.fromhex(body["attestation_cbor_hex"])
        claimed_root = bytes.fromhex(body["attestation_root_hex"])
        signature_preimage = bytes.fromhex(body["signature_preimage_hex"])
    except (TypeError, ValueError):
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 formal bytes are malformed")
    decoded = decode_formal_object(
        cbor, expected_name="ParentManifestAbsenceAttestationV2"
    )
    if encode_formal_object("ParentManifestAbsenceAttestationV2", decoded.fields) != cbor:
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 attestation CBOR is not canonical")
    rebuilt_root = candidate_content_root(
        "ParentManifestAbsenceAttestationV2", decoded.fields
    )
    if rebuilt_root != claimed_root:
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 attestation root differs")
    if (
        decoded.fields["auditor_key_id"]
        != bytes.fromhex(str(request["auditor_key_id_hex"]))
        or decoded.fields["audited_at_unix_seconds"]
        != request["audited_at_unix_seconds"]
    ):
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 attestation actor/timestamp differs")
    expected_preimage = external_signature_preimage_v1(
        OBJECT_TAGS["ParentManifestAbsenceAttestationV2"], rebuilt_root, 4, 0
    )
    if signature_preimage != expected_preimage:
        _fail(FAIL_ACTOR_RESPONSE, "purpose-4 signature preimage differs")
    _validate_complete_receipt(
        body["parent_absence_public_receipt"],
        decoded.fields["audit_bundle_root"].hex(),  # type: ignore[union-attr]
    )
    return {**body, "response_sha256": claimed}


def run_purpose4_detached_audit_v1(
    snapshot: DetachedParentSnapshotV1,
    *,
    auditor_key_id: bytes,
    audited_at_unix_seconds: int,
) -> dict[str, object]:
    """Run the no-key purpose-4 generator in an offline isolated container."""

    basis_commit = _require_basis_commit(
        str(snapshot.manifest.get("basis_commit_sha1"))
    )
    validate_detached_parent_snapshot_v1(
        snapshot.root,
        snapshot.manifest,
        git_executable=snapshot.git_executable,
        expected_basis_commit=basis_commit,
    )
    image_ref = _load_profile_image()
    try:
        temporary_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-gate17-purpose4-runtime-",
            repository_root=REPOSITORY_ROOT,
            parent=DEFAULT_TEMPORARY_PARENT,
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_LOCAL_RUNTIME, f"{exc.code}: {exc.detail}")
    with temporary_owner as temporary:
        root = Path(temporary)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                root,
                repository_root=REPOSITORY_ROOT,
            )
            version_result = subprocess.run(
                control_plane.command("version", "--format", "{{json .}}"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
                env=dict(control_plane.environment),
            )
            info_result = subprocess.run(
                control_plane.command("info", "--format", "{{json .}}"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
                env=dict(control_plane.environment),
            )
            if version_result.returncode != 0 or info_result.returncode != 0:
                raise ValueError("local Docker daemon identity query failed")
            daemon_receipt = build_local_docker_daemon_identity_receipt_v1(
                control_plane,
                version_payload=json.loads(version_result.stdout),
                info_payload=json.loads(info_result.stdout),
                repository_root=REPOSITORY_ROOT,
            )
            local_docker_daemon_receipt_binding_v1(daemon_receipt)
        except (
            OSError,
            subprocess.TimeoutExpired,
            ValueError,
            json.JSONDecodeError,
            Phase3LocalRuntimeError,
        ) as exc:
            _fail(FAIL_LOCAL_RUNTIME, f"purpose-4 Docker control plane failed: {exc}")
        runtime = root / "runtime"
        request_path = root / "request.json"
        runtime.mkdir(mode=0o700)
        try:
            runtime_bundle = _copy_actor_runtime(
                runtime,
                snapshot.git_executable,
                basis_commit=basis_commit,
            )
            request = build_purpose4_actor_request_v1(
                snapshot_manifest=snapshot.manifest,
                runtime_inventory=runtime_bundle["runtime_inventory"],  # type: ignore[arg-type]
                runtime_source_bindings=runtime_bundle["runtime_source_bindings"],  # type: ignore[arg-type]
                basis_commit=basis_commit,
                actor_image_ref=image_ref,
                auditor_key_id=auditor_key_id,
                audited_at_unix_seconds=audited_at_unix_seconds,
            )
            request_path.write_bytes(_canonical_json(request))
            request_path.chmod(0o444)
            command = purpose4_container_command_v1(
                snapshot=snapshot.root,
                runtime=runtime,
                request_path=request_path,
                seccomp_path=(
                    runtime / "control/phase3_internal_actor_seccomp_v1.json"
                ),
                image_ref=image_ref,
            )
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=1800,
                    env=dict(control_plane.environment),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                _fail(FAIL_ACTOR_POLICY, f"purpose-4 container could not complete: {type(exc).__name__}")
            if completed.returncode != 0 or completed.stderr:
                detail = completed.stderr.decode("ascii", "replace")[-256:].strip()
                _fail(
                    FAIL_ACTOR_POLICY,
                    f"purpose-4 container exited {completed.returncode}: {detail}",
                )
            if len(completed.stdout) > MAX_ACTOR_OUTPUT_BYTES:
                _fail(FAIL_ACTOR_RESPONSE, "purpose-4 public response is oversized")
            try:
                response = json.loads(completed.stdout)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _fail(FAIL_ACTOR_RESPONSE, f"purpose-4 response is invalid JSON: {exc}")
            return validate_purpose4_actor_response_v1(response, request=request)
        finally:
            if request_path.exists():
                request_path.chmod(0o600)
            if runtime.exists():
                _set_snapshot_read_only(runtime, False)


__all__ = [
    "ACTOR_REQUEST_SCHEMA",
    "ACTOR_RESPONSE_SCHEMA",
    "AUDITED_REF",
    "DetachedParentSnapshotV1",
    "FAIL_ACTOR_POLICY",
    "FAIL_ACTOR_RESPONSE",
    "FAIL_GIT_RUNTIME_BINDING",
    "FAIL_RECEIPT_INCOMPLETE",
    "FAIL_RUNTIME_BASIS",
    "FAIL_SNAPSHOT_BUILD",
    "FAIL_SNAPSHOT_INVENTORY",
    "FAIL_SNAPSHOT_POLICY",
    "Purpose4DetachedAuditError",
    "SNAPSHOT_SCHEMA",
    "build_purpose4_actor_request_v1",
    "git_runtime_binding_v1",
    "prepare_detached_parent_snapshot_v1",
    "purpose4_container_command_v1",
    "run_purpose4_detached_audit_v1",
    "validate_detached_parent_snapshot_v1",
    "validate_purpose4_actor_response_v1",
]
