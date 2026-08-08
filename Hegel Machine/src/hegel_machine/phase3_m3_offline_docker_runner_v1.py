"""Offline Docker runner for the formal M3 dual enumerator supervisor.

The runner extracts the Python and Rust source closures from frozen Commit A,
checks the already-qualified Rust executable and Python interpreter identities,
and executes both target-free enumerators with digest-pinned local images.
It never permits a registry pull or a container network.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from . import phase3_m3_implementation_qualification_v1 as _qualification
from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    Phase3LocalRuntimeError,
    prepare_local_docker_control_plane_v1,
)
from .phase3_m25_formal_static_basis_v1 import (
    CONTAINER_SECCOMP_PATH,
    _git_blob,
    _source_file_rows,
)
from .phase3_m25_wire_v1 import candidate_record_tree_root
from .phase3_m3_dual_enumeration_supervisor_v1 import (
    COMMIT_A,
    FROZEN_IMPLEMENTATIONS,
    EnumerationInvocationV1,
    EnumerationRunResultV1,
)


FAIL_PREFLIGHT: Final = "FAIL_M3_FORMAL_RUNNER_PREFLIGHT"
FAIL_INVOCATION: Final = "FAIL_M3_FORMAL_RUNNER_INVOCATION"
FAIL_EXECUTION: Final = "FAIL_M3_FORMAL_RUNNER_EXECUTION"
FAIL_STABILITY: Final = "FAIL_M3_FORMAL_RUNNER_INPUT_STABILITY"
FAIL_RESUME: Final = "FAIL_M3_FORMAL_RUNNER_RESUME"
FAIL_TERMINALIZE: Final = "FAIL_M3_FORMAL_RUNNER_UNSAFE_TERMINALIZATION"
MAX_ENUMERATION_SECONDS: Final = 1_800
ATTEMPT_INTENT_SCHEMA: Final = "hegel-m3-offline-runner-attempt-intent/1"
START_MARKER_SCHEMA: Final = "hegel-m3-enumerator-process-start/1"
COMPLETION_MARKER_SCHEMA: Final = "hegel-m3-enumerator-process-completion/1"
JOURNAL_COMPLETION_SCHEMA: Final = "hegel-m3-enumerator-journal-completion/1"
FAILURE_OBSERVATION_SCHEMA: Final = "hegel-m3-enumerator-failure-observation/1"
PROBE_START_SCHEMA: Final = "hegel-m3-python-runtime-probe-start/1"
PROBE_COMPLETION_SCHEMA: Final = "hegel-m3-python-runtime-probe-completion/1"
PROBE_FAILURE_SCHEMA: Final = "hegel-m3-python-runtime-probe-failure/1"
INVOCATION_DIGEST_DOMAIN: Final = b"HEGEL/M3/ENUMERATION_INVOCATION/V1"
RESTART_POLICY: Final = "no"
FAILURE_CLEANUP_POLICY: Final = "OBSERVE_STOP_KILL_NEVER_REMOVE_V1"
PYTHON_PROBE_MAXIMUM_SECONDS: Final = 60


class M3OfflineDockerRunnerError(RuntimeError):
    """Stable fail-closed error for the isolated formal runner."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3OfflineDockerRunnerError(code, detail)


def _require_private_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        _fail(FAIL_PREFLIGHT, f"{label} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_PREFLIGHT, f"{label} is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != path
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        _fail(FAIL_PREFLIGHT, f"{label} must be a real caller-owned mode-0700 directory")
    return resolved


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _exclusive_write(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    code: str = FAIL_EXECUTION,
    allow_identical_existing: bool = False,
) -> str:
    """Durably create one file below a private real directory.

    The file is removed again if any write, metadata, or fsync step fails.  All
    name operations are relative to the already-open parent directory so a
    final-component symlink or parent-path replacement cannot redirect them.
    """

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        _fail(code, "exclusive-write path must be an absolute ordinary child")
    parent = path.parent
    try:
        parent_lexical = parent.lstat()
        parent_resolved = parent.resolve(strict=True)
    except OSError as exc:
        _fail(code, f"exclusive-write parent is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(parent_lexical.st_mode)
        or not stat.S_ISDIR(parent_lexical.st_mode)
        or parent_resolved != parent
        or parent_lexical.st_uid != os.geteuid()
        or stat.S_IMODE(parent_lexical.st_mode) != 0o700
    ):
        _fail(code, "exclusive-write parent must be real caller-owned mode 0700")

    directory_descriptor: int | None = None
    descriptor: int | None = None
    created = False
    complete = False
    try:
        directory_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened_parent = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(opened_parent.st_mode)
            or (opened_parent.st_dev, opened_parent.st_ino)
            != (parent_lexical.st_dev, parent_lexical.st_ino)
        ):
            _fail(code, "exclusive-write parent identity changed")
        try:
            descriptor = os.open(
                path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                mode,
                dir_fd=directory_descriptor,
            )
            created = True
        except FileExistsError:
            if not allow_identical_existing:
                _fail(code, f"exclusive evidence already exists: {path.name}")
            directory_entry = os.stat(
                path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            existing, existing_metadata = _stable_regular_read(
                path, maximum_bytes=max(len(payload), 1), code=code
            )
            lexical_parent_after = parent.lstat()
            if (
                existing != payload
                or existing_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(existing_metadata.st_mode) != mode
                or (existing_metadata.st_dev, existing_metadata.st_ino)
                != (directory_entry.st_dev, directory_entry.st_ino)
                or (lexical_parent_after.st_dev, lexical_parent_after.st_ino)
                != (opened_parent.st_dev, opened_parent.st_ino)
            ):
                _fail(code, f"existing exclusive evidence differs: {path.name}")
            return "EXISTING_IDENTICAL"
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(code, "short execution-evidence write")
            view = view[written:]
        os.fsync(descriptor)
        written_metadata = os.fstat(descriptor)
        named_metadata = os.stat(
            path.name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        lexical_parent_after = parent.lstat()
        lexical_file_after = path.lstat()
        if (
            not stat.S_ISREG(written_metadata.st_mode)
            or stat.S_IMODE(written_metadata.st_mode) != mode
            or written_metadata.st_size != len(payload)
            or (written_metadata.st_dev, written_metadata.st_ino)
            != (named_metadata.st_dev, named_metadata.st_ino)
            or (written_metadata.st_dev, written_metadata.st_ino)
            != (lexical_file_after.st_dev, lexical_file_after.st_ino)
            or (opened_parent.st_dev, opened_parent.st_ino)
            != (lexical_parent_after.st_dev, lexical_parent_after.st_ino)
        ):
            _fail(code, "exclusive evidence identity differs after write")
        os.fsync(directory_descriptor)
        complete = True
        return "CREATED_NEW"
    except M3OfflineDockerRunnerError:
        raise
    except OSError as exc:
        _fail(code, f"cannot publish execution evidence: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and not complete and directory_descriptor is not None:
            try:
                os.unlink(path.name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except OSError:
                # The primary operation already failed.  A surviving partial
                # file remains fail-closed because every later create is O_EXCL.
                pass
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _stable_regular_read(
    path: Path, *, maximum_bytes: int, code: str
) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        lexical = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (lexical.st_dev, lexical.st_ino) != (before.st_dev, before.st_ino)
            or before.st_size > maximum_bytes
        ):
            _fail(code, f"bounded regular-file identity differs: {path.name}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(code, f"file ended early: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(code, f"file grew while read: {path.name}")
        after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_size,
            named_after.st_mode,
            named_after.st_mtime_ns,
            named_after.st_ctime_ns,
        ):
            _fail(code, f"file changed while read: {path.name}")
        return b"".join(chunks), before
    except M3OfflineDockerRunnerError:
        raise
    except OSError as exc:
        _fail(code, f"stable read failed for {path.name}: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _tree_digest(root: Path) -> bytes:
    records: list[bytes] = []
    try:
        root_before = root.lstat()
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        _fail(FAIL_STABILITY, f"immutable input root is unavailable: {type(exc).__name__}")
    if (
        stat.S_ISLNK(root_before.st_mode)
        or not stat.S_ISDIR(root_before.st_mode)
        or root_resolved != root
        or root_before.st_uid != os.geteuid()
        or stat.S_IMODE(root_before.st_mode) & 0o022
    ):
        _fail(FAIL_STABILITY, "immutable input root metadata differs")
    records.append(
        b"D"
        + (0).to_bytes(4, "big")
        + stat.S_IMODE(root_before.st_mode).to_bytes(4, "big")
    )
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        metadata = path.lstat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
            _fail(FAIL_STABILITY, "immutable input tree metadata is unsafe")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if stat.S_ISDIR(metadata.st_mode):
            records.append(
                b"D"
                + len(relative).to_bytes(4, "big")
                + relative
                + stat.S_IMODE(metadata.st_mode).to_bytes(4, "big")
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail(FAIL_STABILITY, "immutable input tree contains a non-regular file")
        payload, stable_metadata = _stable_regular_read(
            path, maximum_bytes=128 * 1024 * 1024, code=FAIL_STABILITY
        )
        records.append(
            b"F"
            + len(relative).to_bytes(4, "big")
            + relative
            + stat.S_IMODE(stable_metadata.st_mode).to_bytes(4, "big")
            + len(payload).to_bytes(8, "big")
            + hashlib.sha256(payload).digest()
        )
    root_after = root.lstat()
    if (
        root_before.st_dev,
        root_before.st_ino,
        root_before.st_uid,
        root_before.st_mode,
        root_before.st_mtime_ns,
        root_before.st_ctime_ns,
    ) != (
        root_after.st_dev,
        root_after.st_ino,
        root_after.st_uid,
        root_after.st_mode,
        root_after.st_mtime_ns,
        root_after.st_ctime_ns,
    ):
        _fail(FAIL_STABILITY, "immutable input root changed while hashed")
    return hashlib.sha256(b"HEGEL/M3/RUNTIME_INPUT_TREE/V1\x00" + b"".join(records)).digest()


def _verify_snapshot_bytes(
    root: Path,
    blobs: Mapping[str, bytes],
    *,
    strip_prefix: str | None = None,
) -> None:
    expected: dict[str, bytes] = {}
    for repository_path, payload in blobs.items():
        relative = repository_path
        if strip_prefix is not None:
            if not relative.startswith(strip_prefix):
                _fail(FAIL_STABILITY, "snapshot source escaped its frozen prefix")
            relative = relative[len(strip_prefix) :]
        expected[relative] = payload
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != set(expected):
        _fail(FAIL_STABILITY, "persisted immutable snapshot file set differs")
    for relative, payload in expected.items():
        path = root / relative
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or _stable_regular_read(
                path, maximum_bytes=128 * 1024 * 1024, code=FAIL_STABILITY
            )[0]
            != payload
        ):
            _fail(FAIL_STABILITY, f"persisted immutable snapshot differs: {relative}")


def _ensure_private_child_directory(
    parent: Path, name: str, *, code: str
) -> Path:
    if not parent.is_absolute() or not name or "/" in name or name in {".", ".."}:
        _fail(code, "private child-directory identity is malformed")
    _require_private_directory(parent, label="private child-directory parent")
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
            created = True
        except FileExistsError:
            pass
        child_descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptor,
        )
        try:
            metadata = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                _fail(code, "private child directory metadata differs")
        finally:
            os.close(child_descriptor)
        if created:
            os.fsync(descriptor)
    except M3OfflineDockerRunnerError:
        raise
    except OSError as exc:
        _fail(code, f"private child directory failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    child = parent / name
    if child.resolve(strict=True) != child:
        _fail(code, "private child directory traverses an alias")
    return child


def _private_directory_identity(
    path: Path, *, code: str, label: str
) -> tuple[int, int, int, int]:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _fail(code, f"{label} is unavailable: {type(exc).__name__}")
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != path
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        _fail(code, f"{label} must be a real caller-owned mode-0700 directory")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_mode,
    )


def _create_private_child_directory(
    parent: Path, name: str, *, code: str
) -> tuple[Path, tuple[int, int, int, int]]:
    """Durably create an exact private child below an already-private parent."""

    if not name or "/" in name or name in {".", ".."}:
        _fail(code, "private output child name is malformed")
    parent_identity = _private_directory_identity(
        parent, code=code, label="private output parent"
    )
    parent_descriptor: int | None = None
    child_descriptor: int | None = None
    created = False
    complete = False
    try:
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened_parent = os.fstat(parent_descriptor)
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
            opened_parent.st_uid,
            opened_parent.st_mode,
        ) != parent_identity:
            _fail(code, "private output parent identity changed")
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        created = True
        child_descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        child_metadata = os.fstat(child_descriptor)
        if (
            not stat.S_ISDIR(child_metadata.st_mode)
            or child_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(child_metadata.st_mode) != 0o700
        ):
            _fail(code, "private output child metadata differs")
        os.fsync(parent_descriptor)
        identity = (
            child_metadata.st_dev,
            child_metadata.st_ino,
            child_metadata.st_uid,
            child_metadata.st_mode,
        )
        complete = True
    except FileExistsError:
        _fail(code, "private output child already exists")
    except M3OfflineDockerRunnerError:
        raise
    except OSError as exc:
        _fail(code, f"private output child creation failed: {type(exc).__name__}")
    finally:
        if child_descriptor is not None:
            os.close(child_descriptor)
        if created and not complete and parent_descriptor is not None:
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    child = parent / name
    if _private_directory_identity(
        child, code=code, label="private output child"
    ) != identity:
        _fail(code, "private output child identity changed after creation")
    return child, identity


def _read_canonical_json_object(
    path: Path,
    *,
    expected_fields: frozenset[str],
    label: str,
    code: str,
) -> tuple[Mapping[str, object], bytes]:
    payload, _metadata = _stable_regular_read(
        path,
        maximum_bytes=1_048_576,
        code=code,
    )
    value = _qualification._strict_json_load(payload, label=label, code=code)
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _fail(code, f"{label} fields differ")
    if _canonical_json_bytes(value) != payload:
        _fail(code, f"{label} is not canonical JSON")
    return MappingProxyType(dict(value)), payload


def _snapshot_verified_executable(
    source: Path,
    destination: Path,
    *,
    expected_digest: bytes,
    expected_size: int,
) -> tuple[Path, bytes]:
    """Copy a verified executable into the attempt-owned immutable tree."""

    try:
        source_metadata = source.lstat()
        source_resolved = source.resolve(strict=True)
        payload, stable_source_metadata = _stable_regular_read(
            source,
            maximum_bytes=max(expected_size, 1),
            code=FAIL_PREFLIGHT,
        )
    except M3OfflineDockerRunnerError:
        raise
    except OSError as exc:
        _fail(FAIL_PREFLIGHT, f"qualified executable is unavailable: {type(exc).__name__}")
    if (
        type(expected_digest) is not bytes
        or len(expected_digest) != 32
        or type(expected_size) is not int
        or expected_size < 1
        or stat.S_ISLNK(source_metadata.st_mode)
        or not stat.S_ISREG(source_metadata.st_mode)
        or source_resolved != source
        or (source_metadata.st_dev, source_metadata.st_ino)
        != (stable_source_metadata.st_dev, stable_source_metadata.st_ino)
        or stat.S_IMODE(source_metadata.st_mode) != 0o555
        or len(payload) != expected_size
        or hashlib.sha256(payload).digest() != expected_digest
    ):
        _fail(FAIL_PREFLIGHT, "qualified executable identity differs")
    if destination.exists() or destination.is_symlink():
        copied, copied_metadata = _stable_regular_read(
            destination,
            maximum_bytes=expected_size,
            code=FAIL_STABILITY,
        )
        if (
            copied != payload
            or stat.S_IMODE(copied_metadata.st_mode) != 0o555
            or hashlib.sha256(copied).digest() != expected_digest
        ):
            _fail(FAIL_STABILITY, "persisted executable snapshot differs")
    else:
        _exclusive_write(
            destination,
            payload,
            mode=0o555,
            code=FAIL_PREFLIGHT,
        )
    return destination, payload


def _invocation_digest_v1(
    invocation: EnumerationInvocationV1, *, attempt_root: Path
) -> str:
    try:
        relative_output = invocation.output_parent.relative_to(attempt_root).as_posix()
    except ValueError:
        _fail(FAIL_INVOCATION, "enumeration output escapes the formal attempt")
    document: dict[str, object] = {
        "implementation": invocation.implementation,
        "implementation_id": invocation.implementation_id,
        "basis_commit": invocation.basis_commit,
        "source_root": invocation.source_root.hex(),
        "binary_digest": invocation.binary_digest.hex(),
        "image_ref": invocation.image_ref,
        "implementation_binding_root": invocation.implementation_binding_root.hex(),
        "bound_executable_locator": invocation.bound_executable_locator,
        "child_dsl_spec_root": invocation.child_dsl_spec_root.hex(),
        "operator_semantics_root": invocation.operator_semantics_root.hex(),
        "identifier_registry_root": invocation.identifier_registry_root.hex(),
        "canonical_program_budget": invocation.canonical_program_budget,
        "raw_operator_application_cap": invocation.raw_operator_application_cap,
        "pull_policy": invocation.pull_policy,
        "network_mode": invocation.network_mode,
        "restart_policy": RESTART_POLICY,
        "failure_cleanup_policy": FAILURE_CLEANUP_POLICY,
        "relative_output_path": relative_output,
    }
    return hashlib.sha256(
        INVOCATION_DIGEST_DOMAIN + b"\x00" + _canonical_json_bytes(document)
    ).hexdigest()


class OfflineDockerEnumerationRunnerV1:
    """Context-managed callable used concurrently by the formal supervisor."""

    def __init__(
        self,
        *,
        repository_root: Path,
        attempt_root: Path,
        implementation_qualification_receipt: Mapping[str, object],
    ) -> None:
        self.repository_root = repository_root.resolve(strict=True)
        self.attempt_root = _require_private_directory(
            attempt_root, label="formal attempt root"
        )
        attempt_metadata = self.attempt_root.lstat()
        self._attempt_directory_identity = (
            attempt_metadata.st_dev,
            attempt_metadata.st_ino,
            attempt_metadata.st_uid,
            attempt_metadata.st_mode,
        )
        self.qualification_receipt = MappingProxyType(
            dict(implementation_qualification_receipt)
        )
        self._runtime_owner: LinuxLocalTemporaryDirectoryV1 | None = None
        self._control_plane = None
        self._python_snapshot: Path | None = None
        self._rust_snapshot: Path | None = None
        self._rust_binary: Path | None = None
        self._seccomp_snapshot: Path | None = None
        self._inputs_root: Path | None = None
        self._input_tree_digests: Mapping[str, bytes] | None = None
        self._container_names: Mapping[str, str] | None = None
        self._probe_container_name: str | None = None
        self._journal_root: Path | None = None
        self._journal_directory_identity: tuple[int, int, int, int] | None = None
        self._attempt_intent_sha256: str | None = None
        self.preflight_receipt: Mapping[str, object] | None = None

    def _assert_attempt_root_stable(self) -> None:
        try:
            metadata = self.attempt_root.lstat()
            resolved = self.attempt_root.resolve(strict=True)
        except OSError as exc:
            _fail(FAIL_STABILITY, f"formal attempt root disappeared: {type(exc).__name__}")
        observed = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_mode,
        )
        if (
            resolved != self.attempt_root
            or not stat.S_ISDIR(metadata.st_mode)
            or observed != self._attempt_directory_identity
        ):
            _fail(FAIL_STABILITY, "formal attempt root identity changed")

    def _python_probe_command_v1(self) -> list[str]:
        if (
            self._control_plane is None
            or self._seccomp_snapshot is None
            or self._probe_container_name is None
        ):
            _fail(FAIL_PREFLIGHT, "named Python probe identity is incomplete")
        script = (
            "import hashlib,json,os,sys;"
            "p=os.path.realpath(sys.executable);"
            "b=open(p,'rb').read();"
            "print(json.dumps({'binary_path':p,'binary_sha256':"
            "hashlib.sha256(b).hexdigest(),'version':sys.version},"
            "sort_keys=True,separators=(',',':')))"
        )
        environment = _qualification.PYTHON_RUNTIME_ENVIRONMENT
        exact_environment = tuple(
            f"{key}={environment[key]}" for key in sorted(environment)
        )
        return self._control_plane.command(
            "run",
            f"--name={self._probe_container_name}",
            "--pull=never",
            "--network=none",
            f"--restart={RESTART_POLICY}",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._seccomp_snapshot}",
            "--user=65534:65534",
            "--pids-limit=64",
            "--memory=512m",
            "--memory-swap=512m",
            "--ulimit=nofile=128:128",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,"
            "uid=65534,gid=65534,mode=0700",
            "--entrypoint=/usr/bin/env",
            FROZEN_IMPLEMENTATIONS["python"].image_ref,
            "-i",
            *exact_environment,
            "python3",
            "-c",
            script,
        )

    @staticmethod
    def _parse_python_probe_stdout_v1(
        stdout: bytes,
    ) -> tuple[str, bytes, bytes]:
        try:
            value = _qualification._parse_single_json(
                stdout,
                label="named Python interpreter probe",
            )
        except Exception as exc:
            _fail(FAIL_PREFLIGHT, f"Python probe output is invalid: {exc}")
        if set(value) != {"binary_path", "binary_sha256", "version"}:
            _fail(FAIL_PREFLIGHT, "Python probe output fields differ")
        path = value["binary_path"]
        digest_text = value["binary_sha256"]
        version = value["version"]
        if (
            type(path) is not str
            or not path.startswith("/usr/local/bin/python")
            or type(digest_text) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest_text) is None
            or type(version) is not str
            or _canonical_json_bytes(value) != stdout
        ):
            _fail(FAIL_PREFLIGHT, "Python probe identity differs")
        return (
            path,
            bytes.fromhex(digest_text),
            hashlib.sha256(version.encode("utf-8")).digest(),
        )

    def _python_probe_paths_v1(self) -> tuple[Path, Path, Path]:
        if self._journal_root is None:
            _fail(FAIL_PREFLIGHT, "Python probe journal root is unavailable")
        return (
            self._journal_root / "python-probe-started.json",
            self._journal_root / "python-probe-completed.json",
            self.attempt_root / "python-runtime-probe-stdout.json",
        )

    @staticmethod
    def _observation_proves_clean_probe_exit_v1(
        observation: Mapping[str, object],
    ) -> bool:
        return (
            observation.get("presence") == "PRESENT"
            and observation.get("running_or_null") is False
            and observation.get("restarting_or_null") is False
            and observation.get("status_or_null") == "exited"
            and observation.get("exit_code_or_null") == 0
            and observation.get("oom_killed_or_null") is False
            and observation.get("docker_error_sha256_or_null")
            == hashlib.sha256(b"").hexdigest()
        )

    def _terminalize_probe_after_failure_v1(
        self, *, cause: BaseException
    ) -> None:
        if (
            self._probe_container_name is None
            or self._attempt_intent_sha256 is None
            or self._journal_root is None
        ):
            _fail(FAIL_TERMINALIZE, "Python probe terminalization identity is unavailable")
        (
            actions,
            observations,
            _final,
            safe,
            terminalization_status,
        ) = self._quiesce_named_container(
            container_name=self._probe_container_name,
            observation_label="python-probe",
        )
        cause_code = getattr(cause, "code", type(cause).__name__)
        if type(cause_code) is not str or not cause_code or "\x00" in cause_code:
            cause_code = "UNCLASSIFIED_PYTHON_PROBE_FAILURE"
        observation_id = time.time_ns()
        record: dict[str, object] = {
            "schema": PROBE_FAILURE_SCHEMA,
            "container_name": self._probe_container_name,
            "attempt_intent_sha256": self._attempt_intent_sha256,
            "cause_code": cause_code,
            "cause_detail_sha256": hashlib.sha256(
                str(cause).encode("utf-8", "replace")
            ).hexdigest(),
            "cleanup_policy": FAILURE_CLEANUP_POLICY,
            "container_removal_attempted": False,
            "actions": actions,
            "observations": observations,
            "terminalization_status": terminalization_status,
            "safe_to_terminalize_execution": safe,
            "recorded_at_unix_seconds": int(time.time()),
            "observation_id_unix_nanoseconds": observation_id,
        }
        payload = _canonical_json_bytes(record)
        _exclusive_write(
            self._journal_root / f"python-probe-failure-{observation_id:020d}.json",
            payload,
            code=FAIL_TERMINALIZE,
        )
        if not safe:
            _fail(
                FAIL_TERMINALIZE,
                "named Python probe is not proven non-running after stop/kill; "
                f"failure_journal_sha256={hashlib.sha256(payload).hexdigest()}",
            )

    def _run_or_resume_named_python_probe_v1(
        self,
    ) -> tuple[str, bytes, bytes, bytes]:
        try:
            return self._run_or_resume_named_python_probe_unprotected_v1()
        except BaseException as exc:
            try:
                start_path, completion_path, stdout_path = self._python_probe_paths_v1()
                probe_may_exist = any(
                    path.exists() or path.is_symlink()
                    for path in (start_path, completion_path, stdout_path)
                )
                if probe_may_exist:
                    self._terminalize_probe_after_failure_v1(cause=exc)
            except M3OfflineDockerRunnerError as terminalization_error:
                if terminalization_error.code == FAIL_TERMINALIZE:
                    raise terminalization_error from exc
                _fail(
                    FAIL_TERMINALIZE,
                    "Python probe cleanup failed before non-running state was "
                    f"proven: {terminalization_error.code}",
                )
            except BaseException as terminalization_error:
                _fail(
                    FAIL_TERMINALIZE,
                    "Python probe cleanup raised before non-running state was "
                    f"proven: {type(terminalization_error).__name__}",
                )
            raise

    def _run_or_resume_named_python_probe_unprotected_v1(
        self,
    ) -> tuple[str, bytes, bytes, bytes]:
        if (
            self._control_plane is None
            or self._probe_container_name is None
            or self._attempt_intent_sha256 is None
        ):
            _fail(FAIL_PREFLIGHT, "named Python probe preflight is incomplete")
        start_path, completion_path, stdout_path = self._python_probe_paths_v1()
        start_exists = start_path.exists() or start_path.is_symlink()
        completion_exists = completion_path.exists() or completion_path.is_symlink()
        stdout_exists = stdout_path.exists() or stdout_path.is_symlink()
        if start_exists or completion_exists or stdout_exists:
            if not (start_exists and completion_exists and stdout_exists):
                cause = M3OfflineDockerRunnerError(
                    FAIL_PREFLIGHT,
                    "incomplete named Python probe cannot be rerun",
                )
                raise cause
            start_marker, _start_payload = _read_canonical_json_object(
                start_path,
                expected_fields=frozenset(
                    {
                        "schema",
                        "container_name",
                        "attempt_intent_sha256",
                        "image_ref",
                        "started_at_unix_seconds",
                    }
                ),
                label="Python probe start",
                code=FAIL_PREFLIGHT,
            )
            completion, _completion_payload = _read_canonical_json_object(
                completion_path,
                expected_fields=frozenset(
                    {
                        "schema",
                        "container_name",
                        "attempt_intent_sha256",
                        "image_ref",
                        "binary_path",
                        "binary_sha256",
                        "version_sha256",
                        "stdout_sha256",
                        "started_at_unix_seconds",
                        "finished_at_unix_seconds",
                        "docker_started_at",
                        "docker_finished_at",
                    }
                ),
                label="Python probe completion",
                code=FAIL_PREFLIGHT,
            )
            stdout, stdout_metadata = _stable_regular_read(
                stdout_path,
                maximum_bytes=1_048_576,
                code=FAIL_PREFLIGHT,
            )
            path, digest, version_digest = self._parse_python_probe_stdout_v1(stdout)
            replay_actions: list[dict[str, object]] = []
            observation = self._observe_named_container_for_terminalization(
                container_name=self._probe_container_name,
                observation_label="python-probe-replay",
                phase="completion-replay",
                action_log=replay_actions,
            )
            started = completion.get("started_at_unix_seconds")
            finished = completion.get("finished_at_unix_seconds")
            if (
                start_marker.get("schema") != PROBE_START_SCHEMA
                or completion.get("schema") != PROBE_COMPLETION_SCHEMA
                or start_marker.get("container_name") != self._probe_container_name
                or completion.get("container_name") != self._probe_container_name
                or start_marker.get("attempt_intent_sha256")
                != self._attempt_intent_sha256
                or completion.get("attempt_intent_sha256")
                != self._attempt_intent_sha256
                or start_marker.get("image_ref")
                != FROZEN_IMPLEMENTATIONS["python"].image_ref
                or completion.get("image_ref")
                != FROZEN_IMPLEMENTATIONS["python"].image_ref
                or completion.get("binary_path") != path
                or completion.get("binary_sha256") != digest.hex()
                or completion.get("version_sha256") != version_digest.hex()
                or completion.get("stdout_sha256")
                != hashlib.sha256(stdout).hexdigest()
                or type(started) is not int
                or type(finished) is not int
                or start_marker.get("started_at_unix_seconds") != started
                or finished < started
                or completion.get("docker_started_at")
                != observation.get("started_at_or_null")
                or completion.get("docker_finished_at")
                != observation.get("finished_at_or_null")
                or stdout_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(stdout_metadata.st_mode) != 0o600
                or not self._observation_proves_clean_probe_exit_v1(observation)
            ):
                cause = M3OfflineDockerRunnerError(
                    FAIL_PREFLIGHT,
                    "persisted named Python probe identity differs",
                )
                raise cause
            return path, digest, version_digest, stdout

        prestart_actions: list[dict[str, object]] = []
        first_absence = self._observe_named_container_for_terminalization(
            container_name=self._probe_container_name,
            observation_label="python-probe-prestart",
            phase="absence-check",
            action_log=prestart_actions,
        )
        second_absence = self._observe_named_container_for_terminalization(
            container_name=self._probe_container_name,
            observation_label="python-probe-prestart",
            phase="absence-confirmation",
            action_log=prestart_actions,
        )
        if (
            first_absence.get("presence") != "ABSENT"
            or second_absence.get("presence") != "ABSENT"
        ):
            _fail(
                FAIL_TERMINALIZE,
                "named Python probe container name is not proven absent",
            )
        started = int(time.time())
        start_marker: dict[str, object] = {
            "schema": PROBE_START_SCHEMA,
            "container_name": self._probe_container_name,
            "attempt_intent_sha256": self._attempt_intent_sha256,
            "image_ref": FROZEN_IMPLEMENTATIONS["python"].image_ref,
            "started_at_unix_seconds": started,
        }
        _exclusive_write(
            start_path,
            _canonical_json_bytes(start_marker),
            code=FAIL_PREFLIGHT,
        )
        completed = _qualification._run(
            self._python_probe_command_v1(),
            code=FAIL_PREFLIGHT,
            timeout=PYTHON_PROBE_MAXIMUM_SECONDS,
            environment=self._control_plane.environment,
        )
        finished = int(time.time())
        if completed.stderr != b"":
            _fail(FAIL_PREFLIGHT, "named Python probe emitted stderr")
        path, digest, version_digest = self._parse_python_probe_stdout_v1(
            completed.stdout
        )
        observation_actions: list[dict[str, object]] = []
        observation = self._observe_named_container_for_terminalization(
            container_name=self._probe_container_name,
            observation_label="python-probe-completion",
            phase="clean-exit",
            action_log=observation_actions,
        )
        if not self._observation_proves_clean_probe_exit_v1(observation):
            _fail(FAIL_PREFLIGHT, "named Python probe lacks an exact clean exit")
        _exclusive_write(
            stdout_path,
            completed.stdout,
            code=FAIL_PREFLIGHT,
        )
        completion: dict[str, object] = {
            "schema": PROBE_COMPLETION_SCHEMA,
            "container_name": self._probe_container_name,
            "attempt_intent_sha256": self._attempt_intent_sha256,
            "image_ref": FROZEN_IMPLEMENTATIONS["python"].image_ref,
            "binary_path": path,
            "binary_sha256": digest.hex(),
            "version_sha256": version_digest.hex(),
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "started_at_unix_seconds": started,
            "finished_at_unix_seconds": finished,
            "docker_started_at": observation["started_at_or_null"],
            "docker_finished_at": observation["finished_at_or_null"],
        }
        _exclusive_write(
            completion_path,
            _canonical_json_bytes(completion),
            code=FAIL_PREFLIGHT,
        )
        return path, digest, version_digest, completed.stdout

    def __enter__(self) -> "OfflineDockerEnumerationRunnerV1":
        self._assert_attempt_root_stable()
        try:
            python_blobs = _qualification.validate_python_source_closure_v1(
                self.repository_root, COMMIT_A
            )
            rust_blobs = _qualification.validate_rust_source_closure_v1(
                self.repository_root, COMMIT_A
            )
            python_source_root = candidate_record_tree_root(
                "SourceFileRecordV1",
                _source_file_rows(
                    self.repository_root,
                    COMMIT_A,
                    _qualification.PYTHON_SOURCE_PATHS,
                ),
            )
            rust_source_root = candidate_record_tree_root(
                "SourceFileRecordV1",
                _source_file_rows(
                    self.repository_root,
                    COMMIT_A,
                    _qualification.RUST_SOURCE_PATHS,
                ),
            )
        except Exception as exc:
            _fail(FAIL_PREFLIGHT, f"Commit-A source closure replay failed: {exc}")
        if (
            python_source_root != FROZEN_IMPLEMENTATIONS["python"].source_root
            or rust_source_root != FROZEN_IMPLEMENTATIONS["rust"].source_root
        ):
            _fail(FAIL_PREFLIGHT, "Commit-A source roots differ from the frozen bindings")

        inputs = self.attempt_root / "immutable-inputs"
        try:
            package = inputs / "python-package/hegel_machine"
            rust_snapshot = inputs / "rust-source"
            if inputs.exists():
                inputs_metadata = inputs.lstat()
                if (
                    stat.S_ISLNK(inputs_metadata.st_mode)
                    or not stat.S_ISDIR(inputs_metadata.st_mode)
                    or inputs.resolve(strict=True) != inputs
                    or inputs_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(inputs_metadata.st_mode) != 0o700
                ):
                    _fail(FAIL_STABILITY, "persisted immutable input root differs")
                _verify_snapshot_bytes(
                    package,
                    python_blobs,
                    strip_prefix="Hegel Machine/src/hegel_machine/",
                )
                _verify_snapshot_bytes(rust_snapshot, rust_blobs)
            else:
                inputs.mkdir(mode=0o700, exist_ok=False)
                package.parent.mkdir(mode=0o755)
                _qualification._write_snapshot(
                    package,
                    python_blobs,
                    strip_prefix="Hegel Machine/src/hegel_machine/",
                )
                _qualification._write_snapshot(rust_snapshot, rust_blobs)
        except Exception as exc:
            _fail(FAIL_PREFLIGHT, f"immutable input materialization failed: {exc}")
        self._python_snapshot = package.parent
        self._rust_snapshot = rust_snapshot

        qualified_rust_binary = (
            self.repository_root
            / "Hegel Machine/rust/m3_closure_enumerator/target/m3_qualification"
            / COMMIT_A
            / "hegel-m3-closure-enumerator"
        )
        rust_binary, binary_payload = _snapshot_verified_executable(
            qualified_rust_binary,
            inputs / "rust-enumerator",
            expected_digest=FROZEN_IMPLEMENTATIONS["rust"].binary_digest,
            expected_size=710_496,
        )
        self._rust_binary = rust_binary

        try:
            committed_seccomp = _git_blob(
                self.repository_root, COMMIT_A, CONTAINER_SECCOMP_PATH
            )
            seccomp_snapshot = inputs / "runtime-seccomp-v1.json"
            if seccomp_snapshot.exists():
                metadata = seccomp_snapshot.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o444
                    or _stable_regular_read(
                        seccomp_snapshot,
                        maximum_bytes=1_048_576,
                        code=FAIL_STABILITY,
                    )[0]
                    != committed_seccomp
                ):
                    _fail(FAIL_STABILITY, "persisted runtime seccomp differs")
            else:
                _exclusive_write(seccomp_snapshot, committed_seccomp, mode=0o444)
        except Exception as exc:
            _fail(FAIL_PREFLIGHT, f"runtime seccomp snapshot failed: {exc}")
        self._seccomp_snapshot = seccomp_snapshot
        self._inputs_root = inputs

        try:
            runtime_owner = LinuxLocalTemporaryDirectoryV1(
                prefix="hegel-m3-formal-runtime-",
                repository_root=self.repository_root,
            )
            runtime_root = Path(runtime_owner.__enter__())
            control = prepare_local_docker_control_plane_v1(
                runtime_root, repository_root=self.repository_root
            )
            _daemon_receipt, daemon_binding = (
                _qualification._qualify_local_docker_control_plane_v1(
                    control, repository_root=self.repository_root
                )
            )
        except (Exception, Phase3LocalRuntimeError) as exc:
            if "runtime_owner" in locals():
                runtime_owner.__exit__(type(exc), exc, exc.__traceback__)
            _fail(FAIL_PREFLIGHT, f"local offline Docker boundary failed: {exc}")
        claimed_daemon = self.qualification_receipt.get(
            "local_docker_daemon_receipt_binding"
        )
        if (
            type(claimed_daemon) is not str
            or re.fullmatch(r"[0-9a-f]{64}", claimed_daemon) is None
            or daemon_binding.hex() != claimed_daemon
        ):
            runtime_owner.__exit__(None, None, None)
            _fail(FAIL_PREFLIGHT, "local Docker daemon identity changed after qualification")
        self._control_plane = control
        try:
            tree_digests = {
                "python": _tree_digest(self._python_snapshot),
                "rust": _tree_digest(self._rust_snapshot),
                "all_inputs": _tree_digest(inputs),
            }
            self._input_tree_digests = MappingProxyType(tree_digests)
            attempt_token = hashlib.sha256(
                self.attempt_root.as_posix().encode("utf-8")
            ).hexdigest()[:16]
            self._container_names = MappingProxyType(
                {
                    name: f"hegel-m3-{attempt_token}-{name}"
                    for name in ("python", "rust")
                }
            )
            self._probe_container_name = (
                f"hegel-m3-{attempt_token}-python-probe"
            )
            qualification_root = self.qualification_receipt.get("receipt_root")
            if (
                type(qualification_root) is not str
                or re.fullmatch(r"[0-9a-f]{64}", qualification_root) is None
            ):
                _fail(FAIL_PREFLIGHT, "implementation qualification root is malformed")
            journal = _ensure_private_child_directory(
                self.attempt_root, "runner-journal", code=FAIL_PREFLIGHT
            )
            self._journal_root = journal
            self._journal_directory_identity = _private_directory_identity(
                journal,
                code=FAIL_PREFLIGHT,
                label="runner journal root",
            )
            intent: dict[str, object] = {
                "schema": ATTEMPT_INTENT_SCHEMA,
                "basis_commit": COMMIT_A,
                "attempt_root_path_sha256": hashlib.sha256(
                    self.attempt_root.as_posix().encode("utf-8")
                ).hexdigest(),
                "implementation_qualification_receipt_root": qualification_root,
                "python_implementation_binding_root": FROZEN_IMPLEMENTATIONS[
                    "python"
                ].implementation_binding_root.hex(),
                "rust_implementation_binding_root": FROZEN_IMPLEMENTATIONS[
                    "rust"
                ].implementation_binding_root.hex(),
                "all_immutable_inputs_sha256": tree_digests["all_inputs"].hex(),
                "enumeration_output_relative_path": "formal-enumeration",
                "pull_policy": "never",
                "network_mode": "none",
                "restart_policy": RESTART_POLICY,
                "failure_cleanup_policy": FAILURE_CLEANUP_POLICY,
                "maximum_enumeration_seconds": MAX_ENUMERATION_SECONDS,
                "container_names": dict(self._container_names),
                "python_probe_container_name": self._probe_container_name,
                "python_probe_maximum_seconds": PYTHON_PROBE_MAXIMUM_SECONDS,
                "python_probe_auto_remove": False,
            }
            intent_payload = _canonical_json_bytes(intent)
            _exclusive_write(
                self.attempt_root / "runner-attempt-intent.json",
                intent_payload,
                code=FAIL_PREFLIGHT,
                allow_identical_existing=True,
            )
            self._attempt_intent_sha256 = hashlib.sha256(intent_payload).hexdigest()
            (
                _python_path,
                python_digest,
                _python_version_digest,
                python_probe_stdout,
            ) = self._run_or_resume_named_python_probe_v1()
            probe_start_path, probe_completion_path, _probe_stdout_path = (
                self._python_probe_paths_v1()
            )
            probe_start_payload, _probe_start_metadata = _stable_regular_read(
                probe_start_path,
                maximum_bytes=1_048_576,
                code=FAIL_PREFLIGHT,
            )
            probe_completion_payload, _probe_completion_metadata = (
                _stable_regular_read(
                    probe_completion_path,
                    maximum_bytes=1_048_576,
                    code=FAIL_PREFLIGHT,
                )
            )
            if python_digest != FROZEN_IMPLEMENTATIONS["python"].binary_digest:
                cause = M3OfflineDockerRunnerError(
                    FAIL_PREFLIGHT,
                    "Python interpreter digest differs from qualification",
                )
                self._terminalize_probe_after_failure_v1(cause=cause)
                raise cause
            self.preflight_receipt = MappingProxyType(
                {
                    "basis_commit": COMMIT_A,
                    "python_source_root": python_source_root.hex(),
                    "rust_source_root": rust_source_root.hex(),
                    "python_input_tree_sha256": tree_digests["python"].hex(),
                    "rust_input_tree_sha256": tree_digests["rust"].hex(),
                    "all_immutable_inputs_sha256": tree_digests["all_inputs"].hex(),
                    "rust_binary_sha256": hashlib.sha256(binary_payload).hexdigest(),
                    "runtime_seccomp_sha256": hashlib.sha256(
                        committed_seccomp
                    ).hexdigest(),
                    "python_probe_stdout_sha256": hashlib.sha256(
                        python_probe_stdout
                    ).hexdigest(),
                    "python_probe_start_sha256": hashlib.sha256(
                        probe_start_payload
                    ).hexdigest(),
                    "python_probe_completion_sha256": hashlib.sha256(
                        probe_completion_payload
                    ).hexdigest(),
                    "python_probe_container_name": self._probe_container_name,
                    "docker_daemon_receipt_binding": daemon_binding.hex(),
                    "pull_policy": "never",
                    "network_mode": "none",
                    "maximum_enumeration_seconds": MAX_ENUMERATION_SECONDS,
                    "container_names": dict(self._container_names),
                    "attempt_intent_sha256": self._attempt_intent_sha256,
                }
            )
            _exclusive_write(
                self.attempt_root / "runner-preflight.json",
                _canonical_json_bytes(self.preflight_receipt),
                code=FAIL_PREFLIGHT,
                allow_identical_existing=True,
            )
            self._assert_attempt_root_stable()
        except BaseException as exc:
            try:
                runtime_owner.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
            self._control_plane = None
            raise
        self._runtime_owner = runtime_owner
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._runtime_owner is not None:
            self._runtime_owner.__exit__(exc_type, exc, traceback)
            self._runtime_owner = None
        return False

    def _validate_invocation(self, invocation: EnumerationInvocationV1) -> None:
        self._assert_attempt_root_stable()
        frozen = FROZEN_IMPLEMENTATIONS.get(invocation.implementation)
        if frozen is None:
            _fail(FAIL_INVOCATION, "unknown implementation")
        exact = (
            invocation.implementation_id == frozen.implementation_id,
            invocation.basis_commit == COMMIT_A,
            invocation.source_root == frozen.source_root,
            invocation.binary_digest == frozen.binary_digest,
            invocation.image_ref == frozen.image_ref,
            invocation.implementation_binding_root
            == frozen.implementation_binding_root,
            invocation.bound_executable_locator == frozen.bound_executable_locator,
            invocation.pull_policy == "never",
            invocation.network_mode == "none",
            invocation.canonical_program_budget == 50_000,
            invocation.raw_operator_application_cap == 5_000_000,
            invocation.output_parent.is_absolute(),
            invocation.output_parent
            == self.attempt_root
            / "formal-enumeration"
            / invocation.implementation,
        )
        if not all(exact):
            _fail(FAIL_INVOCATION, f"{invocation.implementation} invocation differs")
        try:
            parent_metadata = invocation.output_parent.parent.lstat()
            parent_resolved = invocation.output_parent.parent.resolve(strict=True)
        except OSError as exc:
            _fail(FAIL_INVOCATION, f"enumeration output parent is unavailable: {exc}")
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_resolved != invocation.output_parent.parent
            or parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        ):
            _fail(FAIL_INVOCATION, "enumeration output parent identity differs")

    def _docker_command(
        self,
        invocation: EnumerationInvocationV1,
        *,
        options: tuple[str, ...],
        command: tuple[str, ...],
        environment: Mapping[str, str],
    ) -> list[str]:
        if (
            self._control_plane is None
            or self._seccomp_snapshot is None
            or self._container_names is None
        ):
            _fail(FAIL_PREFLIGHT, "Docker runner preflight is incomplete")
        user = f"{os.getuid()}:{os.getgid()}"
        exact_environment = tuple(
            f"{key}={environment[key]}" for key in sorted(environment)
        )
        return self._control_plane.command(
            "run",
            f"--name={self._container_names[invocation.implementation]}",
            "--pull=never",
            "--network=none",
            f"--restart={RESTART_POLICY}",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._seccomp_snapshot}",
            f"--user={user}",
            "--pids-limit=64",
            "--memory=512m",
            "--memory-swap=512m",
            "--ulimit=nofile=128:128",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,"
            f"uid={os.getuid()},gid={os.getgid()},mode=0700",
            *options,
            "--entrypoint=/usr/bin/env",
            invocation.image_ref,
            "-i",
            *exact_environment,
            *command,
        )

    def _journal_paths(self, implementation: str) -> tuple[Path, Path]:
        if (
            self._journal_root is None
            or self._journal_directory_identity is None
            or self._container_names is None
            or implementation not in self._container_names
        ):
            _fail(FAIL_PREFLIGHT, "runner journal identity is unavailable")
        if _private_directory_identity(
            self._journal_root,
            code=FAIL_STABILITY,
            label="runner journal root",
        ) != self._journal_directory_identity:
            _fail(FAIL_STABILITY, "runner journal root identity changed")
        return (
            self._journal_root / f"{implementation}-started.json",
            self._journal_root / f"{implementation}-completed.json",
        )

    def _claim_execution_start(
        self,
        invocation: EnumerationInvocationV1,
        *,
        started_at_unix_seconds: int,
    ) -> str:
        if (
            self._attempt_intent_sha256 is None
            or self._container_names is None
            or type(started_at_unix_seconds) is not int
            or started_at_unix_seconds < 0
        ):
            _fail(FAIL_PREFLIGHT, "runner start-journal state is incomplete")
        invocation_sha256 = _invocation_digest_v1(
            invocation, attempt_root=self.attempt_root
        )
        start_path, _completion_path = self._journal_paths(
            invocation.implementation
        )
        start_marker: dict[str, object] = {
            "schema": START_MARKER_SCHEMA,
            "implementation": invocation.implementation,
            "implementation_id": invocation.implementation_id,
            "container_name": self._container_names[invocation.implementation],
            "invocation_sha256": invocation_sha256,
            "attempt_intent_sha256": self._attempt_intent_sha256,
            "started_at_unix_seconds": started_at_unix_seconds,
        }
        _exclusive_write(
            start_path,
            _canonical_json_bytes(start_marker),
            code=FAIL_RESUME,
        )
        return invocation_sha256

    def _publish_completion_journal(
        self,
        invocation: EnumerationInvocationV1,
        *,
        invocation_sha256: str,
        started_at_unix_seconds: int,
        finished_at_unix_seconds: int,
        process_completion_payload: bytes,
    ) -> None:
        if self._attempt_intent_sha256 is None or self._container_names is None:
            _fail(FAIL_PREFLIGHT, "runner completion-journal state is incomplete")
        _start_path, completion_path = self._journal_paths(
            invocation.implementation
        )
        journal_marker: dict[str, object] = {
            "schema": JOURNAL_COMPLETION_SCHEMA,
            "implementation": invocation.implementation,
            "implementation_id": invocation.implementation_id,
            "container_name": self._container_names[invocation.implementation],
            "invocation_sha256": invocation_sha256,
            "attempt_intent_sha256": self._attempt_intent_sha256,
            "started_at_unix_seconds": started_at_unix_seconds,
            "finished_at_unix_seconds": finished_at_unix_seconds,
            "process_completion_sha256": hashlib.sha256(
                process_completion_payload
            ).hexdigest(),
        }
        _exclusive_write(
            completion_path,
            _canonical_json_bytes(journal_marker),
            code=FAIL_EXECUTION,
        )

    def _terminalization_control_call(
        self,
        *arguments: str,
        label: str,
        action_log: list[dict[str, object]],
    ) -> object | None:
        if self._control_plane is None:
            detail = "Docker control plane is unavailable"
            action_log.append(
                {
                    "action": label,
                    "succeeded": False,
                    "stdout_sha256_or_null": None,
                    "stderr_sha256_or_null": None,
                    "failure_detail_sha256_or_null": hashlib.sha256(
                        detail.encode("utf-8")
                    ).hexdigest(),
                }
            )
            return None
        try:
            completed = _qualification._run(
                self._control_plane.command(*arguments),
                code=FAIL_TERMINALIZE,
                timeout=60,
                environment=self._control_plane.environment,
            )
        except Exception as exc:
            action_log.append(
                {
                    "action": label,
                    "succeeded": False,
                    "stdout_sha256_or_null": None,
                    "stderr_sha256_or_null": None,
                    "failure_detail_sha256_or_null": hashlib.sha256(
                        str(exc).encode("utf-8", "replace")
                    ).hexdigest(),
                }
            )
            return None
        action_log.append(
            {
                "action": label,
                "succeeded": True,
                "stdout_sha256_or_null": hashlib.sha256(
                    completed.stdout
                ).hexdigest(),
                "stderr_sha256_or_null": hashlib.sha256(
                    completed.stderr
                ).hexdigest(),
                "failure_detail_sha256_or_null": None,
            }
        )
        return completed

    def _observe_named_container_for_terminalization(
        self,
        *,
        container_name: str,
        observation_label: str,
        phase: str,
        action_log: list[dict[str, object]],
    ) -> dict[str, object]:
        if (
            type(container_name) is not str
            or not container_name
            or container_name.startswith("-")
            or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", container_name) is None
        ):
            return {
                "phase": phase,
                "presence": "UNAVAILABLE",
                "running_or_null": None,
                "restarting_or_null": None,
                "reason_sha256_or_null": hashlib.sha256(
                    b"container name is malformed"
                ).hexdigest(),
            }
        listing = self._terminalization_control_call(
            "container",
            "ls",
            "--all",
            f"--filter=name=^/{container_name}$",
            "--format={{json .Names}}",
            label=f"{observation_label}/{phase}/list-exact-name",
            action_log=action_log,
        )
        if listing is None:
            return {
                "phase": phase,
                "presence": "UNAVAILABLE",
                "running_or_null": None,
                "restarting_or_null": None,
                "reason_sha256_or_null": hashlib.sha256(
                    b"exact-name listing unavailable"
                ).hexdigest(),
            }
        expected_listing = (
            json.dumps(container_name, ensure_ascii=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
        if listing.stdout == b"":
            return {
                "phase": phase,
                "presence": "ABSENT",
                "running_or_null": False,
                "restarting_or_null": False,
                "reason_sha256_or_null": None,
            }
        if listing.stdout != expected_listing:
            return {
                "phase": phase,
                "presence": "UNAVAILABLE",
                "running_or_null": None,
                "restarting_or_null": None,
                "reason_sha256_or_null": hashlib.sha256(
                    b"exact-name listing identity differs"
                ).hexdigest(),
            }
        inspected = self._terminalization_control_call(
            "container",
            "inspect",
            "--format={{json .State}}",
            container_name,
            label=f"{observation_label}/{phase}/inspect-state",
            action_log=action_log,
        )
        if inspected is None:
            return {
                "phase": phase,
                "presence": "UNAVAILABLE",
                "running_or_null": None,
                "restarting_or_null": None,
                "reason_sha256_or_null": hashlib.sha256(
                    b"container state unavailable"
                ).hexdigest(),
            }
        try:
            state = _qualification._parse_single_json(
                inspected.stdout,
                label=f"{observation_label} terminalization state",
            )
        except Exception as exc:
            return {
                "phase": phase,
                "presence": "UNAVAILABLE",
                "running_or_null": None,
                "restarting_or_null": None,
                "reason_sha256_or_null": hashlib.sha256(
                    str(exc).encode("utf-8", "replace")
                ).hexdigest(),
            }
        status = state.get("Status")
        running = state.get("Running")
        restarting = state.get("Restarting")
        exit_code = state.get("ExitCode")
        oom_killed = state.get("OOMKilled")
        docker_error = state.get("Error")
        started_at = state.get("StartedAt")
        finished_at = state.get("FinishedAt")
        if (
            type(status) is not str
            or type(running) is not bool
            or type(restarting) is not bool
            or type(exit_code) is not int
            or type(oom_killed) is not bool
            or type(docker_error) is not str
            or type(started_at) is not str
            or type(finished_at) is not str
        ):
            return {
                "phase": phase,
                "presence": "UNAVAILABLE",
                "running_or_null": None,
                "restarting_or_null": None,
                "reason_sha256_or_null": hashlib.sha256(
                    b"container state fields differ"
                ).hexdigest(),
            }
        return {
            "phase": phase,
            "presence": "PRESENT",
            "running_or_null": running,
            "restarting_or_null": restarting,
            "status_or_null": status,
            "exit_code_or_null": exit_code,
            "oom_killed_or_null": oom_killed,
            "docker_error_sha256_or_null": hashlib.sha256(
                docker_error.encode("utf-8")
            ).hexdigest(),
            "started_at_or_null": started_at,
            "finished_at_or_null": finished_at,
            "reason_sha256_or_null": None,
        }

    @staticmethod
    def _observation_proves_not_running(observation: Mapping[str, object]) -> bool:
        return observation.get("presence") == "ABSENT" or (
            observation.get("presence") == "PRESENT"
            and observation.get("running_or_null") is False
            and observation.get("restarting_or_null") is False
        )

    def _quiesce_named_container(
        self,
        *,
        container_name: str,
        observation_label: str,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, object],
        bool,
        str,
    ]:
        action_log: list[dict[str, object]] = []
        observations: list[dict[str, object]] = []
        initial = self._observe_named_container_for_terminalization(
            container_name=container_name,
            observation_label=observation_label,
            phase="initial",
            action_log=action_log,
        )
        observations.append(initial)
        final = initial
        if initial.get("presence") == "ABSENT":
            confirmation = self._observe_named_container_for_terminalization(
                container_name=container_name,
                observation_label=observation_label,
                phase="absence-confirmation",
                action_log=action_log,
            )
            observations.append(confirmation)
            final = confirmation
        if not self._observation_proves_not_running(final):
            self._terminalization_control_call(
                "container",
                "stop",
                "--time=10",
                container_name,
                label=f"{observation_label}/stop-fixed-name",
                action_log=action_log,
            )
            after_stop = self._observe_named_container_for_terminalization(
                container_name=container_name,
                observation_label=observation_label,
                phase="after-stop",
                action_log=action_log,
            )
            observations.append(after_stop)
            final = after_stop
            if not self._observation_proves_not_running(after_stop):
                self._terminalization_control_call(
                    "container",
                    "kill",
                    container_name,
                    label=f"{observation_label}/kill-fixed-name",
                    action_log=action_log,
                )
                after_kill = self._observe_named_container_for_terminalization(
                    container_name=container_name,
                    observation_label=observation_label,
                    phase="after-kill",
                    action_log=action_log,
                )
                observations.append(after_kill)
                final = after_kill
        safe = self._observation_proves_not_running(final)
        if final.get("presence") == "ABSENT":
            status = "SAFE_CONTAINER_ABSENT"
        elif safe:
            status = "SAFE_CONTAINER_NOT_RUNNING"
        elif final.get("presence") == "PRESENT":
            status = "UNSAFE_CONTAINER_STILL_RUNNING"
        else:
            status = "UNSAFE_FINAL_STATE_UNAVAILABLE"
        return action_log, observations, final, safe, status

    def _terminalize_after_failure(
        self,
        invocation: EnumerationInvocationV1,
        *,
        cause: BaseException,
    ) -> None:
        if self._container_names is None or self._attempt_intent_sha256 is None:
            _fail(FAIL_TERMINALIZE, "terminalization identity is unavailable")
        start_path, _completion_path = self._journal_paths(
            invocation.implementation
        )
        execution_may_exist = (
            start_path.exists()
            or start_path.is_symlink()
            or invocation.output_parent.exists()
            or invocation.output_parent.is_symlink()
        )
        if not execution_may_exist:
            return

        container_name = self._container_names[invocation.implementation]
        (
            action_log,
            observations,
            _final,
            safe,
            terminalization_status,
        ) = self._quiesce_named_container(
            container_name=container_name,
            observation_label=invocation.implementation,
        )
        cause_code = getattr(cause, "code", type(cause).__name__)
        if type(cause_code) is not str or not cause_code or "\x00" in cause_code:
            cause_code = "UNCLASSIFIED_RUNNER_FAILURE"
        recorded_at = int(time.time())
        observation_id = time.time_ns()
        failure_record: dict[str, object] = {
            "schema": FAILURE_OBSERVATION_SCHEMA,
            "implementation": invocation.implementation,
            "implementation_id": invocation.implementation_id,
            "container_name": self._container_names[invocation.implementation],
            "invocation_sha256": _invocation_digest_v1(
                invocation,
                attempt_root=self.attempt_root,
            ),
            "attempt_intent_sha256": self._attempt_intent_sha256,
            "cause_code": cause_code,
            "cause_detail_sha256": hashlib.sha256(
                str(cause).encode("utf-8", "replace")
            ).hexdigest(),
            "cleanup_policy": FAILURE_CLEANUP_POLICY,
            "container_removal_attempted": False,
            "actions": action_log,
            "observations": observations,
            "terminalization_status": terminalization_status,
            "safe_to_terminalize_execution": safe,
            "recorded_at_unix_seconds": recorded_at,
            "observation_id_unix_nanoseconds": observation_id,
        }
        failure_payload = _canonical_json_bytes(failure_record)
        if self._journal_root is None:
            _fail(FAIL_TERMINALIZE, "failure journal root is unavailable")
        _exclusive_write(
            self._journal_root
            / (
                f"{invocation.implementation}-failure-"
                f"{observation_id:020d}.json"
            ),
            failure_payload,
            code=FAIL_TERMINALIZE,
        )
        if not safe:
            _fail(
                FAIL_TERMINALIZE,
                "named container is not proven non-running after stop/kill; "
                f"failure_journal_sha256={hashlib.sha256(failure_payload).hexdigest()}",
            )

    def _resume_completed_result(
        self, invocation: EnumerationInvocationV1
    ) -> EnumerationRunResultV1 | None:
        self._assert_attempt_root_stable()
        if self._attempt_intent_sha256 is None or self._container_names is None:
            _fail(FAIL_PREFLIGHT, "runner resume identity is unavailable")
        start_path, journal_completion_path = self._journal_paths(
            invocation.implementation
        )
        output_exists = (
            invocation.output_parent.exists() or invocation.output_parent.is_symlink()
        )
        start_exists = start_path.exists() or start_path.is_symlink()
        journal_completion_exists = (
            journal_completion_path.exists()
            or journal_completion_path.is_symlink()
        )
        if not output_exists and not start_exists and not journal_completion_exists:
            return None
        if not (output_exists and start_exists and journal_completion_exists):
            _fail(
                FAIL_RESUME,
                "incomplete persisted execution cannot be rerun",
            )
        try:
            _private_directory_identity(
                invocation.output_parent,
                code=FAIL_RESUME,
                label="persisted implementation output",
            )
            start_marker, _start_payload = _read_canonical_json_object(
                start_path,
                expected_fields=frozenset(
                    {
                        "schema",
                        "implementation",
                        "implementation_id",
                        "container_name",
                        "invocation_sha256",
                        "attempt_intent_sha256",
                        "started_at_unix_seconds",
                    }
                ),
                label=f"{invocation.implementation} execution start",
                code=FAIL_RESUME,
            )
            journal_completion, _journal_completion_payload = (
                _read_canonical_json_object(
                    journal_completion_path,
                    expected_fields=frozenset(
                        {
                            "schema",
                            "implementation",
                            "implementation_id",
                            "container_name",
                            "invocation_sha256",
                            "attempt_intent_sha256",
                            "started_at_unix_seconds",
                            "finished_at_unix_seconds",
                            "process_completion_sha256",
                        }
                    ),
                    label=f"{invocation.implementation} journal completion",
                    code=FAIL_RESUME,
                )
            )
            marker_path = invocation.output_parent / "process-completion.json"
            marker, marker_payload = _read_canonical_json_object(
                marker_path,
                expected_fields=frozenset(
                    {
                        "schema",
                        "implementation",
                        "implementation_id",
                        "container_name",
                        "invocation_sha256",
                        "attempt_intent_sha256",
                        "started_at_unix_seconds",
                        "finished_at_unix_seconds",
                        "process_exit_code",
                        "stdout_sha256",
                        "stderr_sha256",
                        "pull_policy",
                        "network_mode",
                        "docker_started_at",
                        "docker_finished_at",
                        "docker_oom_killed",
                        "docker_error",
                    }
                ),
                label=f"{invocation.implementation} process completion",
                code=FAIL_RESUME,
            )
            stdout, _stdout_metadata = _stable_regular_read(
                invocation.output_parent / "execution-stdout.json",
                maximum_bytes=1_048_576,
                code=FAIL_RESUME,
            )
            stderr, _stderr_metadata = _stable_regular_read(
                invocation.output_parent / "execution-stderr.bin",
                maximum_bytes=1_048_576,
                code=FAIL_RESUME,
            )
        except M3OfflineDockerRunnerError:
            raise
        except Exception as exc:
            _fail(FAIL_RESUME, f"persisted completion replay failed: {exc}")
        invocation_sha256 = _invocation_digest_v1(
            invocation, attempt_root=self.attempt_root
        )
        started = marker.get("started_at_unix_seconds")
        finished = marker.get("finished_at_unix_seconds")
        common_identity = {
            "implementation": invocation.implementation,
            "implementation_id": invocation.implementation_id,
            "container_name": self._container_names[invocation.implementation],
            "invocation_sha256": invocation_sha256,
            "attempt_intent_sha256": self._attempt_intent_sha256,
        }
        if (
            start_marker.get("schema") != START_MARKER_SCHEMA
            or journal_completion.get("schema") != JOURNAL_COMPLETION_SCHEMA
            or marker.get("schema") != COMPLETION_MARKER_SCHEMA
            or any(
                document.get(field) != value
                for document in (start_marker, journal_completion, marker)
                for field, value in common_identity.items()
            )
            or type(started) is not int
            or type(finished) is not int
            or type(start_marker.get("started_at_unix_seconds")) is not int
            or type(journal_completion.get("started_at_unix_seconds")) is not int
            or type(journal_completion.get("finished_at_unix_seconds")) is not int
            or type(marker.get("process_exit_code")) is not int
            or started < 0
            or finished < started
            or start_marker.get("started_at_unix_seconds") != started
            or journal_completion.get("started_at_unix_seconds") != started
            or journal_completion.get("finished_at_unix_seconds") != finished
            or journal_completion.get("process_completion_sha256")
            != hashlib.sha256(marker_payload).hexdigest()
            or marker.get("process_exit_code") != 0
            or marker.get("pull_policy") != "never"
            or marker.get("network_mode") != "none"
            or marker.get("docker_oom_killed") is not False
            or marker.get("docker_error") != ""
            or marker.get("stdout_sha256") != hashlib.sha256(stdout).hexdigest()
            or marker.get("stderr_sha256") != hashlib.sha256(stderr).hexdigest()
            or stderr != b""
        ):
            _fail(FAIL_RESUME, "persisted journal or process identity differs")
        try:
            container_state = self._inspect_container_state(invocation)
        except Exception as exc:
            _fail(
                FAIL_RESUME,
                f"persisted container completion cannot be replayed: {exc}",
            )
        if (
            marker.get("docker_started_at") != container_state["StartedAt"]
            or marker.get("docker_finished_at") != container_state["FinishedAt"]
        ):
            _fail(FAIL_RESUME, "persisted container completion identity differs")
        try:
            report = _qualification._parse_single_json(
                stdout,
                label=f"persisted formal {invocation.implementation} report",
            )
        except Exception as exc:
            _fail(FAIL_RESUME, f"persisted enumerator report cannot be replayed: {exc}")
        return EnumerationRunResultV1(
            invocation=invocation,
            report=MappingProxyType(dict(report)),
            started_at_unix_seconds=started,
            finished_at_unix_seconds=finished,
            process_exit_code=0,
        )

    def _inspect_container_state(
        self, invocation: EnumerationInvocationV1
    ) -> Mapping[str, object]:
        if self._control_plane is None or self._container_names is None:
            _fail(FAIL_PREFLIGHT, "Docker control plane is unavailable")
        try:
            completed = _qualification._run(
                self._control_plane.command(
                    "inspect",
                    "--format={{json .State}}",
                    self._container_names[invocation.implementation],
                ),
                code=FAIL_EXECUTION,
                timeout=60,
                environment=self._control_plane.environment,
            )
            state = _qualification._parse_single_json(
                completed.stdout,
                label=f"{invocation.implementation} container state",
            )
        except Exception as exc:
            _fail(FAIL_EXECUTION, f"container completion state is unavailable: {exc}")
        if (
            state.get("Status") != "exited"
            or state.get("Running") is not False
            or state.get("Restarting") is not False
            or state.get("ExitCode") != 0
            or state.get("OOMKilled") is not False
            or state.get("Error") != ""
            or type(state.get("StartedAt")) is not str
            or type(state.get("FinishedAt")) is not str
        ):
            _fail(FAIL_EXECUTION, "container did not reach an exact clean exit")
        return state

    def __call__(
        self, invocation: EnumerationInvocationV1
    ) -> EnumerationRunResultV1:
        self._validate_invocation(invocation)
        if (
            self._control_plane is None
            or self._python_snapshot is None
            or self._rust_binary is None
            or self._container_names is None
        ):
            _fail(FAIL_PREFLIGHT, "runner context is not active")
        try:
            return self._execute_or_resume_v1(invocation)
        except BaseException as exc:
            try:
                self._terminalize_after_failure(invocation, cause=exc)
            except M3OfflineDockerRunnerError as terminalization_error:
                if terminalization_error.code == FAIL_TERMINALIZE:
                    raise terminalization_error from exc
                _fail(
                    FAIL_TERMINALIZE,
                    "failure cleanup boundary failed before non-running state "
                    f"was proven: {terminalization_error.code}",
                )
            except BaseException as terminalization_error:
                _fail(
                    FAIL_TERMINALIZE,
                    "failure cleanup raised before non-running state was proven: "
                    f"{type(terminalization_error).__name__}",
                )
            raise

    def _execute_or_resume_v1(
        self, invocation: EnumerationInvocationV1
    ) -> EnumerationRunResultV1:
        self.verify_inputs_stable_v1()
        resumed = self._resume_completed_result(invocation)
        if resumed is not None:
            return resumed
        roots = (
            invocation.child_dsl_spec_root,
            invocation.operator_semantics_root,
            invocation.identifier_registry_root,
        )
        started = int(time.time())
        if invocation.implementation == "python":
            options = (
                "-v",
                f"{self._python_snapshot}:/input:ro",
                "-v",
                f"{invocation.output_parent}:/output:rw",
                "-w",
                "/input/hegel_machine",
            )
            command = (
                "python3",
                "/input/hegel_machine/phase3_m3_isolated_entrypoint_v1.py",
                "--enumerate-prefix",
                "--child-dsl-spec-root",
                roots[0].hex(),
                "--operator-semantics-root",
                roots[1].hex(),
                "--identifier-registry-root",
                roots[2].hex(),
                "--output-directory",
                "/output/archive",
            )
            environment = _qualification.PYTHON_RUNTIME_ENVIRONMENT
        else:
            options = (
                "-v",
                f"{self._rust_binary}:/input/enumerator:ro",
                "-v",
                f"{invocation.output_parent}:/output:rw",
            )
            command = (
                "/input/enumerator",
                "--enumerate-prefix",
                "--child-dsl-spec-root",
                roots[0].hex(),
                "--operator-semantics-root",
                roots[1].hex(),
                "--identifier-registry-root",
                roots[2].hex(),
                "--output-directory",
                "/output/archive",
            )
            environment = _qualification.RUST_RUNTIME_ENVIRONMENT
        try:
            invocation_sha256 = self._claim_execution_start(
                invocation,
                started_at_unix_seconds=started,
            )
            output_path, output_identity = _create_private_child_directory(
                invocation.output_parent.parent,
                invocation.output_parent.name,
                code=FAIL_EXECUTION,
            )
            if output_path != invocation.output_parent:
                _fail(FAIL_EXECUTION, "created enumeration output path differs")
            if _private_directory_identity(
                invocation.output_parent,
                code=FAIL_EXECUTION,
                label="enumeration output directory",
            ) != output_identity:
                _fail(FAIL_EXECUTION, "enumeration output identity changed before run")
            completed = _qualification._run(
                self._docker_command(
                    invocation,
                    options=options,
                    command=command,
                    environment=environment,
                ),
                code=FAIL_EXECUTION,
                timeout=MAX_ENUMERATION_SECONDS,
                environment=self._control_plane.environment,
            )
            finished = int(time.time())
            if _private_directory_identity(
                invocation.output_parent,
                code=FAIL_EXECUTION,
                label="enumeration output directory",
            ) != output_identity:
                _fail(FAIL_EXECUTION, "enumeration output identity changed during run")
            if completed.stderr != b"":
                _fail(
                    FAIL_EXECUTION,
                    f"{invocation.implementation} successful run emitted stderr",
                )
            report = _qualification._parse_single_json(
                completed.stdout,
                label=f"formal {invocation.implementation} enumerator report",
            )
            container_state = self._inspect_container_state(invocation)
            _exclusive_write(
                invocation.output_parent / "execution-stdout.json",
                completed.stdout,
            )
            _exclusive_write(
                invocation.output_parent / "execution-stderr.bin",
                completed.stderr,
            )
            completion = {
                "schema": COMPLETION_MARKER_SCHEMA,
                "implementation": invocation.implementation,
                "implementation_id": invocation.implementation_id,
                "container_name": self._container_names[invocation.implementation],
                "invocation_sha256": invocation_sha256,
                "attempt_intent_sha256": self._attempt_intent_sha256,
                "started_at_unix_seconds": started,
                "finished_at_unix_seconds": finished,
                "process_exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "pull_policy": "never",
                "network_mode": "none",
                "docker_started_at": container_state["StartedAt"],
                "docker_finished_at": container_state["FinishedAt"],
                "docker_oom_killed": container_state["OOMKilled"],
                "docker_error": container_state["Error"],
            }
            completion_payload = _canonical_json_bytes(completion)
            _exclusive_write(
                invocation.output_parent / "process-completion.json",
                completion_payload,
            )
            self._publish_completion_journal(
                invocation,
                invocation_sha256=invocation_sha256,
                started_at_unix_seconds=started,
                finished_at_unix_seconds=finished,
                process_completion_payload=completion_payload,
            )
        except M3OfflineDockerRunnerError:
            raise
        except Exception as exc:
            _fail(
                FAIL_EXECUTION,
                f"{invocation.implementation} offline execution failed: {exc}",
            )
        self.verify_inputs_stable_v1()
        return EnumerationRunResultV1(
            invocation=invocation,
            report=MappingProxyType(dict(report)),
            started_at_unix_seconds=started,
            finished_at_unix_seconds=finished,
            process_exit_code=completed.returncode,
        )

    def verify_inputs_stable_v1(self) -> None:
        self._assert_attempt_root_stable()
        if (
            self._python_snapshot is None
            or self._rust_snapshot is None
            or self._rust_binary is None
            or self._inputs_root is None
            or self._input_tree_digests is None
            or self._journal_root is None
            or self._journal_directory_identity is None
            or self._attempt_intent_sha256 is None
            or self._probe_container_name is None
            or self.preflight_receipt is None
        ):
            _fail(FAIL_STABILITY, "runner input state is incomplete")
        rust_binary_payload, rust_binary_metadata = _stable_regular_read(
            self._rust_binary,
            maximum_bytes=16 * 1024 * 1024,
            code=FAIL_STABILITY,
        )
        intent_payload, intent_metadata = _stable_regular_read(
            self.attempt_root / "runner-attempt-intent.json",
            maximum_bytes=1_048_576,
            code=FAIL_STABILITY,
        )
        preflight_payload, preflight_metadata = _stable_regular_read(
            self.attempt_root / "runner-preflight.json",
            maximum_bytes=1_048_576,
            code=FAIL_STABILITY,
        )
        (
            _probe_path,
            probe_binary_digest,
            _probe_version_digest,
            probe_stdout,
        ) = self._run_or_resume_named_python_probe_v1()
        probe_start_path, probe_completion_path, _probe_stdout_path = (
            self._python_probe_paths_v1()
        )
        probe_start_payload, probe_start_metadata = _stable_regular_read(
            probe_start_path,
            maximum_bytes=1_048_576,
            code=FAIL_STABILITY,
        )
        probe_completion_payload, probe_completion_metadata = _stable_regular_read(
            probe_completion_path,
            maximum_bytes=1_048_576,
            code=FAIL_STABILITY,
        )
        if _private_directory_identity(
            self._journal_root,
            code=FAIL_STABILITY,
            label="runner journal root",
        ) != self._journal_directory_identity:
            _fail(FAIL_STABILITY, "runner journal root identity changed")
        if (
            _tree_digest(self._python_snapshot) != self._input_tree_digests["python"]
            or _tree_digest(self._rust_snapshot) != self._input_tree_digests["rust"]
            or _tree_digest(self._inputs_root)
            != self._input_tree_digests["all_inputs"]
            or hashlib.sha256(rust_binary_payload).digest()
            != FROZEN_IMPLEMENTATIONS["rust"].binary_digest
            or stat.S_IMODE(rust_binary_metadata.st_mode) != 0o555
            or hashlib.sha256(intent_payload).hexdigest()
            != self._attempt_intent_sha256
            or preflight_payload != _canonical_json_bytes(self.preflight_receipt)
            or intent_metadata.st_uid != os.geteuid()
            or preflight_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(intent_metadata.st_mode) != 0o600
            or stat.S_IMODE(preflight_metadata.st_mode) != 0o600
            or probe_binary_digest
            != FROZEN_IMPLEMENTATIONS["python"].binary_digest
            or self.preflight_receipt.get("python_probe_container_name")
            != self._probe_container_name
            or self.preflight_receipt.get("python_probe_stdout_sha256")
            != hashlib.sha256(probe_stdout).hexdigest()
            or self.preflight_receipt.get("python_probe_start_sha256")
            != hashlib.sha256(probe_start_payload).hexdigest()
            or self.preflight_receipt.get("python_probe_completion_sha256")
            != hashlib.sha256(probe_completion_payload).hexdigest()
            or probe_start_metadata.st_uid != os.geteuid()
            or probe_completion_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(probe_start_metadata.st_mode) != 0o600
            or stat.S_IMODE(probe_completion_metadata.st_mode) != 0o600
        ):
            _fail(FAIL_STABILITY, "formal runner inputs changed during execution")


__all__ = [
    "FAIL_EXECUTION",
    "FAIL_INVOCATION",
    "FAIL_PREFLIGHT",
    "FAIL_RESUME",
    "FAIL_STABILITY",
    "FAIL_TERMINALIZE",
    "M3OfflineDockerRunnerError",
    "MAX_ENUMERATION_SECONDS",
    "OfflineDockerEnumerationRunnerV1",
]
