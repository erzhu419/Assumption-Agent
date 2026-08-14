"""Fail-closed V3 local Docker executor for the Phase-2B V2 recognizer.

This module turns :class:`~hegel_machine.phase2b_runner.OciRecognizerRunSpec`
into one bounded local Docker-container lifecycle.  It pins every host path component with
``openat``/``O_NOFOLLOW``, verifies the local image and seccomp binding, uses a
sanitized Docker control plane, bounds every client pipe, extracts the one
prediction archive from a private container tmpfs while a fixed supervisor is
still alive, removes the container, and only then independently replays and
publishes the archive on the host.

The image boundary is independently hard-gated to the final V3 image digest,
its exact seven-field raw config, V2 actor-profile labels, owner-accepted
authority-requirement identifier, and 28-entry recognizer source root.  The
resulting receipt is deliberately a *local mechanics observation*.  It is
not an external runtime attestation, a signed recognizer-operation receipt, an
actual-960 qualification, a score, effect evidence, or Phase-2B/C1 exit
evidence.  Those claims require the separate owner-accepted actor ceremony.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn, Sequence

from .hashing import stable_hash
from .phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
)
from .phase2b_recognizer_input_archive_v2 import (
    ARCHIVE_HEADER_BYTES_V2,
    MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
    DecodedRecognizerInputArchiveV2,
    RecognizerInputArchiveDispositionV2,
    decode_public_recognizer_input_archive_v2,
)
from .phase2b_recognizer_prediction_archive_v2 import (
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    DecodedRecognizerPredictionArchiveV2,
    PredictionArchiveDispositionV2,
    decode_public_recognizer_prediction_archive_v2,
)
from .phase2b_recognizer_runtime_cli_v2 import (
    EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
    MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
    RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
    RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
    RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2,
    decode_execution_freeze_manifest_json_v2,
)
from .phase2b_runner import (
    FROZEN_RECOGNIZER_ENTRYPOINT,
    OciRecognizerRunSpec,
    RECOGNIZER_RUNTIME_UID_GID,
    RuntimeLimits,
    TOTAL_RECOGNIZER_CASE_COUNT,
)
from .phase3_local_runtime_v1 import (
    DEFAULT_DOCKER_EXECUTABLE,
    DEFAULT_DOCKER_SOCKET,
    LOCAL_DOCKER_HOST,
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    prepare_local_docker_control_plane_v1,
    validate_linux_local_host_path_v1,
)


LOCAL_EXECUTION_RECEIPT_SCHEMA_V3: Final = (
    "hegel-machine-phase2b-recognizer-local-execution-receipt/3"
)
LOCAL_EXECUTION_COMPLETE_REASON_V3: Final = (
    "phase2b_recognizer_v3_local_docker_mechanics_complete_not_authority"
)
LOCAL_EXECUTION_DISPOSITION_V3: Final = (
    "LOCAL_RECOGNIZER_V3_MECHANICS_COMPLETE_NOT_AUTHORITY"
)

FAIL_PREFLIGHT: Final = "FAIL_PHASE2B_RECOGNIZER_HOST_PREFLIGHT"
FAIL_HOST_PATH: Final = "FAIL_PHASE2B_RECOGNIZER_HOST_PATH"
FAIL_INPUT_BINDING: Final = "FAIL_PHASE2B_RECOGNIZER_INPUT_BINDING"
FAIL_SECCOMP_BINDING: Final = "FAIL_PHASE2B_RECOGNIZER_SECCOMP_BINDING"
FAIL_DOCKER_CONTROL_PLANE: Final = (
    "FAIL_PHASE2B_RECOGNIZER_DOCKER_CONTROL_PLANE"
)
FAIL_IMAGE_BINDING: Final = "FAIL_PHASE2B_RECOGNIZER_IMAGE_BINDING"
FAIL_DOCKER_EXECUTION: Final = "FAIL_PHASE2B_RECOGNIZER_DOCKER_EXECUTION"
FAIL_DOCKER_TIMEOUT: Final = "FAIL_PHASE2B_RECOGNIZER_DOCKER_TIMEOUT"
FAIL_DOCKER_CAPTURE: Final = "FAIL_PHASE2B_RECOGNIZER_DOCKER_CAPTURE"
FAIL_OUTPUT_BINDING: Final = "FAIL_PHASE2B_RECOGNIZER_OUTPUT_BINDING"

IMAGE_INSPECT_TIMEOUT_SECONDS_V3: Final = 60
RECOGNIZER_TIMEOUT_SECONDS_V3: Final = 3_600
MAXIMUM_IMAGE_INSPECT_STDOUT_BYTES_V3: Final = 1_048_576
MAXIMUM_DOCKER_STDOUT_BYTES_V3: Final = 65_536
MAXIMUM_DOCKER_STDERR_BYTES_V3: Final = 8_192
MAXIMUM_DOCKER_RESULT_WIRE_BYTES_V3: Final = (
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2 + 65_536 + 64
)
MAXIMUM_SECCOMP_BYTES_V3: Final = 65_536
MAXIMUM_DOCKER_EXECUTABLE_BYTES_V3: Final = 134_217_728
MAXIMUM_PATH_BYTES_V3: Final = 4_096
MAXIMUM_PATH_COMPONENTS_V3: Final = 64
MAXIMUM_NAMESPACE_NAMES_V3: Final = 8
_READ_CHUNK_BYTES_V3: Final = 1 << 20
_EXACT_PATH_TYPE_V3: Final = type(Path("/"))
RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3: Final = (
    "sha256:d27247e9b4b6fcbeca38aea81359b29a855c05f7a7a6e9f2c2276d360160054f"
)
RECOGNIZER_SOURCE_ROOT_SHA256_V3: Final = (
    "d71f96926156b78bbd42d016b04c40aaa3e0abc5dad2510adc05c23e8dd54f54"
)
OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
_IMAGE_TITLE_V3: Final = "Hegel Machine Phase2B V3 recognizer runtime"
_IMAGE_DESCRIPTION_V3: Final = (
    "Non-authoritative local Docker recognizer mechanics; no actual-960 "
    "execution, scoring, gate, or C1 evidence"
)
_IMAGE_BASE_DIGEST_V3: Final = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
_IMAGE_CONFIG_FIELDS_V3: Final = frozenset(
    {"ArgsEscaped", "User", "Env", "Entrypoint", "Cmd", "WorkingDir", "Labels"}
)
_IMAGE_ARGS_ESCAPED_V3: Final = True
_IMAGE_ENVIRONMENT_V3: Final = (
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG=C.UTF-8",
    "GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D",
    "PYTHON_VERSION=3.11.15",
    "PYTHON_SHA256=272179ddd9a2e41a0fc8e42e33dfbdca0b3711aa5abf372d3f2d51543d09b625",
    "HOME=/nonexistent",
    "PYTHONHASHSEED=0",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=/opt/hegel/src",
)
_CONTAINER_ENVIRONMENT_V3: Final = (
    "PYTHONHASHSEED=0",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=/opt/hegel/src",
    "HOME=/nonexistent",
    "NO_PROXY=*",
    *_IMAGE_ENVIRONMENT_V3[:5],
)
_STAGED_INPUT_DIRECTORY_V3: Final = "recognizer-input"
_STAGED_SECCOMP_FILENAME_V3: Final = "recognizer-seccomp.json"
_RUNTIME_RECEIPT_FILENAME_V3: Final = ".recognizer-runtime-receipt-v3.json"
_RUNTIME_RECEIPT_PENDING_FILENAME_V3: Final = (
    ".recognizer-runtime-receipt-v3.json.next"
)
_CONTAINER_SUPERVISOR_ENTRYPOINT_V3: Final = "/usr/local/bin/python3"
_CONTAINER_RESULT_WIRE_MAGIC_V3: Final = b"HEGEL-PHASE2B-RESULT-V3\x00"
_CONTAINER_SUPERVISOR_CODE_V3: Final = "\n".join(
    (
        "import contextlib, io, os, sys, time",
        "from hegel_machine.phase2b_recognizer_runtime_cli_v2 import main",
        "captured_stdout = io.StringIO()",
        "captured_stderr = io.StringIO()",
        "with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):",
        "    exit_code = main()",
        "if exit_code != 0:",
        "    sys.stderr.write(captured_stderr.getvalue())",
        "    sys.stderr.flush()",
        "    raise SystemExit(exit_code)",
        "if captured_stderr.getvalue() != '':",
        "    raise SystemExit(96)",
        "receipt = captured_stdout.getvalue().encode('ascii')",
        "if not receipt.endswith(b'\\n') or receipt.count(b'\\n') != 1 or len(receipt) > 65536:",
        "    raise SystemExit(97)",
        "directory = os.open('/phase2b/output', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)",
        "try:",
        f"    if sorted(os.listdir(directory)) != ['{RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}']:",
        "        raise SystemExit(98)",
        f"    pending = os.open('{_RUNTIME_RECEIPT_PENDING_FILENAME_V3}', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)",
        "    try:",
        "        offset = 0",
        "        while offset < len(receipt):",
        "            written = os.write(pending, receipt[offset:])",
        "            if written <= 0:",
        "                raise OSError('short receipt write')",
        "            offset += written",
        "        os.fchmod(pending, 0o400)",
        "        os.fsync(pending)",
        "    finally:",
        "        os.close(pending)",
        f"    os.link('{_RUNTIME_RECEIPT_PENDING_FILENAME_V3}', '{_RUNTIME_RECEIPT_FILENAME_V3}', src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)",
        "    os.fsync(directory)",
        f"    os.unlink('{_RUNTIME_RECEIPT_PENDING_FILENAME_V3}', dir_fd=directory)",
        "    os.fsync(directory)",
        "finally:",
        "    os.close(directory)",
        "while True:",
        "    time.sleep(3600)",
    )
)
_CONTAINER_RESULT_EXTRACTOR_CODE_V3: Final = "\n".join(
    (
        "import os, stat, struct, sys, time",
        f"receipt_name = '{_RUNTIME_RECEIPT_FILENAME_V3}'",
        f"receipt_pending_name = '{_RUNTIME_RECEIPT_PENDING_FILENAME_V3}'",
        f"prediction_name = '{RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}'",
        f"prediction_pending_name = '.{RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}.pending'",
        "allowed_names = {receipt_name, receipt_pending_name, prediction_name, prediction_pending_name}",
        "while True:",
        "    directory = os.open('/phase2b/output', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)",
        "    names = sorted(os.listdir(directory))",
        "    if not set(names) <= allowed_names:",
        "        raise SystemExit(90)",
        "    if names == [receipt_name, prediction_name]:",
        "        break",
        "    os.close(directory)",
        "    time.sleep(0.05)",
        "def read_exact(name, minimum, maximum, exact_mode):",
        "    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)",
        "    try:",
        "        before = os.fstat(descriptor)",
        "        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != exact_mode or before.st_uid != 65534 or before.st_gid != 65534 or before.st_nlink != 1 or not minimum <= before.st_size <= maximum:",
        "            raise SystemExit(92)",
        "        chunks = []",
        "        total = 0",
        "        while total <= maximum:",
        "            chunk = os.read(descriptor, min(1048576, maximum + 1 - total))",
        "            if not chunk:",
        "                break",
        "            chunks.append(chunk)",
        "            total += len(chunk)",
        "        after = os.fstat(descriptor)",
        "        fingerprint = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid, row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)",
        "        payload = b''.join(chunks)",
        "        if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:",
        "            raise SystemExit(93)",
        "        return payload",
        "    finally:",
        "        os.close(descriptor)",
        "receipt = read_exact(receipt_name, 2, 65536, 0o400)",
        f"prediction = read_exact(prediction_name, {PREDICTION_ARCHIVE_HEADER_BYTES_V2}, {MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2}, 0o444)",
        "if sorted(os.listdir(directory)) != [receipt_name, prediction_name]:",
        "    raise SystemExit(94)",
        "os.close(directory)",
        f"sys.stdout.buffer.write({_CONTAINER_RESULT_WIRE_MAGIC_V3!r} + struct.pack('>II', len(receipt), len(prediction)) + receipt + prediction)",
        "sys.stdout.buffer.flush()",
    )
)
_OUTPUT_PENDING_FILENAME_V3: Final = (
    ".phase2b-recognizer-v3-prediction-archive-v2.pending"
)
_CONTAINER_ID_V3 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME_V3 = re.compile(r"hegel-phase2b-recognizer-v3-[0-9a-f]{24}")
_RECEIPT_ISSUE_TOKEN_V3 = object()
_RECEIPT_FIXED_CLAIMS_V3: Final = (
    "docker_invoked",
    "local_mechanics_complete",
    "external_runtime_attestation_issued",
    "runtime_attestation_authority_verified",
    "actor_profile_authority_verified",
    "authority_requirement_satisfied",
    "recognizer_operation_signature_verified",
    "actual_input_archive_supplied",
    "actual_960_recognizer_executed",
    "actual_960_case_run_verified",
    "fresh_role_keys_generated",
    "key_generation_performed",
    "custodian_ceremony_executed",
    "signed_execution_registry_materialized",
    "signed_monotonic_time_materialized",
    "store_receipt_materialized",
    "prediction_commit_before_reveal_verified",
    "reveal_authority_verified",
    "scoring_performed",
    "prediction_scored",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
    "malicious_host_root_resistance",
    "malicious_docker_daemon_resistance",
)
_RUNTIME_RECEIPT_FIELDS_V3: Final = frozenset(
    {
        "schema_version",
        "disposition",
        "reason",
        "input_archive_id",
        "input_archive_sha256",
        "execution_freeze_manifest_id",
        "execution_freeze_manifest_sha256",
        "prediction_archive_id",
        "prediction_archive_sha256",
        "prediction_case_count",
        "prediction_archive_bytes",
        "output_filename",
        "receipt_id",
    }
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST_REF = re.compile(r"sha256:[0-9a-f]{64}")


class Phase2BRecognizerHostExecutorErrorV3(RuntimeError):
    """Stable fail-closed host-executor error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BRecognizerHostExecutorErrorV3(code, detail)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        _fail(FAIL_PREFLIGHT, f"{label} must be lowercase SHA-256")


def _required_open_flag(name: str, code: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail(code, f"host kernel lacks required {name}")
    return value


def _directory_flags(code: str) -> int:
    return (
        os.O_RDONLY
        | _required_open_flag("O_DIRECTORY", code)
        | _required_open_flag("O_NOFOLLOW", code)
        | _required_open_flag("O_CLOEXEC", code)
        | _required_open_flag("O_NONBLOCK", code)
    )


def _regular_flags(code: str) -> int:
    return (
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW", code)
        | _required_open_flag("O_CLOEXEC", code)
        | _required_open_flag("O_NONBLOCK", code)
    )


def _exact_absolute_path(value: str, label: str, code: str) -> Path:
    if (
        type(value) is not str
        or not value
        or any(
            character == ","
            or ord(character) < 0x20
            or ord(character) == 0x7F
            for character in value
        )
    ):
        _fail(code, f"{label} must be exact nonempty text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        _fail(code, f"{label} is not UTF-8 encodable: {type(exc).__name__}")
    pure = PurePosixPath(value)
    if (
        len(encoded) > MAXIMUM_PATH_BYTES_V3
        or not pure.is_absolute()
        or value == "/"
        or str(pure) != value
        or os.path.abspath(value) != value
        or os.path.normpath(value) != value
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
        or len(pure.parts) - 1 > MAXIMUM_PATH_COMPONENTS_V3
    ):
        _fail(code, f"{label} is not an exact specific absolute path")
    return Path(value)


@dataclass(frozen=True, slots=True)
class _PathComponentIdentityV3:
    component: str
    device: int
    inode: int
    mode: int
    owner_uid: int
    owner_gid: int

    @property
    def inode_key(self) -> tuple[int, int]:
        return (self.device, self.inode)


def _directory_identity(
    component: str,
    metadata: os.stat_result,
    *,
    code: str,
) -> _PathComponentIdentityV3:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(code, f"host path component is not a directory: {component}")
    return _PathComponentIdentityV3(
        component=component,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        owner_uid=metadata.st_uid,
        owner_gid=metadata.st_gid,
    )


@dataclass(slots=True)
class _PinnedDirectoryV3:
    path: Path
    descriptors: tuple[int, ...]
    identities: tuple[_PathComponentIdentityV3, ...]

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    @property
    def final_identity(self) -> _PathComponentIdentityV3:
        return self.identities[-1]

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_componentwise_directory_v3(
    path: Path,
    *,
    label: str,
    code: str = FAIL_HOST_PATH,
) -> _PinnedDirectoryV3:
    exact = _exact_absolute_path(path.as_posix(), label, code)
    descriptors: list[int] = []
    identities: list[_PathComponentIdentityV3] = []
    try:
        root_descriptor = os.open("/", _directory_flags(code))
        descriptors.append(root_descriptor)
        identities.append(
            _directory_identity("/", os.fstat(root_descriptor), code=code)
        )
        current_descriptor = root_descriptor
        for component in exact.parts[1:]:
            descriptor = os.open(
                component,
                _directory_flags(code),
                dir_fd=current_descriptor,
            )
            descriptors.append(descriptor)
            identities.append(
                _directory_identity(component, os.fstat(descriptor), code=code)
            )
            current_descriptor = descriptor
    except Phase2BRecognizerHostExecutorErrorV3:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as exc:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        _fail(code, f"{label} componentwise secure-open failed: {type(exc).__name__}")
    return _PinnedDirectoryV3(
        path=exact,
        descriptors=tuple(descriptors),
        identities=tuple(identities),
    )


def _recheck_pinned_directory_v3(
    pinned: _PinnedDirectoryV3,
    *,
    label: str,
    code: str = FAIL_HOST_PATH,
) -> None:
    rebound = _open_componentwise_directory_v3(
        pinned.path,
        label=f"{label} recheck",
        code=code,
    )
    try:
        if rebound.identities != pinned.identities:
            _fail(code, f"{label} component identity drift")
        if _directory_identity(
            pinned.path.name,
            os.fstat(pinned.descriptor),
            code=code,
        ) != pinned.final_identity:
            _fail(code, f"{label} held-directory identity drift")
    finally:
        rebound.close()


def _require_pairwise_nonoverlap_v3(
    directories: Mapping[str, _PinnedDirectoryV3],
) -> None:
    rows = tuple(sorted(directories.items()))
    for index, (left_name, left) in enumerate(rows):
        left_ancestors = {item.inode_key for item in left.identities}
        for right_name, right in rows[index + 1 :]:
            right_ancestors = {item.inode_key for item in right.identities}
            if (
                left.final_identity.inode_key in right_ancestors
                or right.final_identity.inode_key in left_ancestors
            ):
                _fail(
                    FAIL_HOST_PATH,
                    f"{left_name} and {right_name} directories alias or overlap",
                )


def _directory_names_v3(
    pinned: _PinnedDirectoryV3,
    *,
    label: str,
    code: str,
) -> frozenset[str]:
    scan_descriptor: int | None = None
    try:
        scan_descriptor = os.open(
            ".",
            _directory_flags(code),
            dir_fd=pinned.descriptor,
        )
        names: list[str] = []
        with os.scandir(scan_descriptor) as entries:
            scan_descriptor = None  # scandir owns the duplicated descriptor.
            for entry in entries:
                if len(names) >= MAXIMUM_NAMESPACE_NAMES_V3:
                    _fail(code, f"{label} namespace exceeds its bounded cap")
                if type(entry.name) is not str or not entry.name:
                    _fail(code, f"{label} namespace contains an invalid name")
                names.append(entry.name)
        if len(names) != len(set(names)):
            _fail(code, f"{label} namespace repeats a name")
        return frozenset(names)
    except Phase2BRecognizerHostExecutorErrorV3:
        raise
    except OSError as exc:
        _fail(code, f"{label} namespace scan failed: {type(exc).__name__}")
    finally:
        if scan_descriptor is not None:
            os.close(scan_descriptor)


@dataclass(slots=True)
class _PinnedRegularFileV3:
    base_directory: _PinnedDirectoryV3
    relative_path: PurePosixPath
    intermediate_descriptors: tuple[int, ...]
    descriptor: int
    metadata_fingerprint: tuple[int, ...]
    payload: bytes
    sha256: str
    single_link_required: bool

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass
        for descriptor in reversed(self.intermediate_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _regular_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_descriptor_v3(
    descriptor: int,
    *,
    maximum_bytes: int,
    minimum_bytes: int,
    code: str,
    label: str,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size < minimum_bytes
        or before.st_size > maximum_bytes
    ):
        _fail(code, f"{label} is not a bounded regular file")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES_V3, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
    except OSError as exc:
        _fail(code, f"{label} bounded read failed: {type(exc).__name__}")
    after = os.fstat(descriptor)
    if (
        len(payload) != before.st_size
        or not minimum_bytes <= len(payload) <= maximum_bytes
        or _regular_fingerprint(before) != _regular_fingerprint(after)
    ):
        _fail(code, f"{label} changed during bounded read")
    return payload, before


def _open_relative_regular_file_v3(
    base: _PinnedDirectoryV3,
    relative: str,
    *,
    maximum_bytes: int,
    minimum_bytes: int,
    single_link_required: bool,
    label: str,
    code: str,
) -> _PinnedRegularFileV3:
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or len(relative_path.parts) > MAXIMUM_PATH_COMPONENTS_V3
    ):
        _fail(code, f"{label} relative path escapes its pinned root")
    intermediate: list[int] = []
    descriptor: int | None = None
    current = base.descriptor
    try:
        for component in relative_path.parts[:-1]:
            child = os.open(component, _directory_flags(code), dir_fd=current)
            _directory_identity(component, os.fstat(child), code=code)
            intermediate.append(child)
            current = child
        name = relative_path.parts[-1]
        descriptor = os.open(name, _regular_flags(code), dir_fd=current)
        payload, metadata = _read_regular_descriptor_v3(
            descriptor,
            maximum_bytes=maximum_bytes,
            minimum_bytes=minimum_bytes,
            code=code,
            label=label,
        )
        named = os.stat(name, dir_fd=current, follow_symlinks=False)
        if (
            _regular_fingerprint(metadata) != _regular_fingerprint(named)
            or (single_link_required and metadata.st_nlink != 1)
        ):
            _fail(code, f"{label} named inode or hardlink policy differs")
    except Phase2BRecognizerHostExecutorErrorV3:
        if descriptor is not None:
            os.close(descriptor)
        for child in reversed(intermediate):
            os.close(child)
        raise
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for child in reversed(intermediate):
            try:
                os.close(child)
            except OSError:
                pass
        _fail(code, f"{label} secure-open failed: {type(exc).__name__}")
    return _PinnedRegularFileV3(
        base_directory=base,
        relative_path=relative_path,
        intermediate_descriptors=tuple(intermediate),
        descriptor=descriptor,
        metadata_fingerprint=_regular_fingerprint(metadata),
        payload=payload,
        sha256=_sha256(payload),
        single_link_required=single_link_required,
    )


def _recheck_regular_file_v3(
    pinned: _PinnedRegularFileV3,
    *,
    maximum_bytes: int,
    minimum_bytes: int,
    label: str,
    code: str,
) -> None:
    rebound = _open_relative_regular_file_v3(
        pinned.base_directory,
        pinned.relative_path.as_posix(),
        maximum_bytes=maximum_bytes,
        minimum_bytes=minimum_bytes,
        single_link_required=pinned.single_link_required,
        label=f"{label} recheck",
        code=code,
    )
    try:
        if (
            rebound.metadata_fingerprint != pinned.metadata_fingerprint
            or rebound.sha256 != pinned.sha256
            or rebound.payload != pinned.payload
            or _regular_fingerprint(os.fstat(pinned.descriptor))
            != pinned.metadata_fingerprint
        ):
            _fail(code, f"{label} identity or content drift")
    finally:
        rebound.close()


def _open_absolute_regular_file_v3(
    path: Path,
    *,
    maximum_bytes: int,
    minimum_bytes: int,
    single_link_required: bool,
    label: str,
    code: str,
) -> tuple[_PinnedDirectoryV3, _PinnedRegularFileV3]:
    exact = _exact_absolute_path(path.as_posix(), label, code)
    parent = _open_componentwise_directory_v3(
        exact.parent,
        label=f"{label} parent",
        code=code,
    )
    try:
        regular = _open_relative_regular_file_v3(
            parent,
            exact.name,
            maximum_bytes=maximum_bytes,
            minimum_bytes=minimum_bytes,
            single_link_required=single_link_required,
            label=label,
            code=code,
        )
    except BaseException:
        parent.close()
        raise
    return parent, regular


def _write_all_v3(descriptor: int, payload: bytes, *, code: str, label: str) -> None:
    view = memoryview(payload)
    offset = 0
    try:
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                _fail(code, f"{label} short write")
            offset += written
    except OSError as exc:
        _fail(code, f"{label} write failed: {type(exc).__name__}")


def _create_private_child_directory_v3(
    base: _PinnedDirectoryV3,
    name: str,
    *,
    label: str,
    code: str,
) -> _PinnedDirectoryV3:
    if (
        type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
        or "," in name
        or any(ord(character) < 0x20 for character in name)
    ):
        _fail(code, f"{label} child name is not exact")
    descriptor: int | None = None
    try:
        os.mkdir(name, 0o700, dir_fd=base.descriptor)
        descriptor = os.open(name, _directory_flags(code), dir_fd=base.descriptor)
        os.fchmod(descriptor, 0o700)
        os.fsync(descriptor)
        os.fsync(base.descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(code, f"{label} private directory creation failed: {type(exc).__name__}")
    if descriptor is not None:
        os.close(descriptor)
    return _open_componentwise_directory_v3(
        base.path / name,
        label=label,
        code=code,
    )


def _write_new_regular_file_v3(
    base: _PinnedDirectoryV3,
    name: str,
    payload: bytes,
    *,
    final_mode: int,
    maximum_bytes: int,
    minimum_bytes: int,
    label: str,
    code: str,
) -> _PinnedRegularFileV3:
    if (
        type(payload) is not bytes
        or not minimum_bytes <= len(payload) <= maximum_bytes
        or type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
        or "," in name
        or final_mode not in {0o400, 0o440, 0o444}
    ):
        _fail(code, f"{label} staged-file arguments drift")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _required_open_flag("O_NOFOLLOW", code)
            | _required_open_flag("O_CLOEXEC", code),
            0o600,
            dir_fd=base.descriptor,
        )
        _write_all_v3(descriptor, payload, code=code, label=label)
        os.fsync(descriptor)
        os.fchmod(descriptor, final_mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(base.descriptor)
    except Phase2BRecognizerHostExecutorErrorV3:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _fail(code, f"{label} durable staging failed: {type(exc).__name__}")
    return _open_relative_regular_file_v3(
        base,
        name,
        maximum_bytes=maximum_bytes,
        minimum_bytes=minimum_bytes,
        single_link_required=True,
        label=label,
        code=code,
    )


def _stage_and_repin_readonly_input_v3(
    runtime_directory: _PinnedDirectoryV3,
    *,
    input_payload: bytes,
    manifest_payload: bytes,
) -> tuple[_PinnedDirectoryV3, _PinnedRegularFileV3, _PinnedRegularFileV3]:
    """Create staged inputs provisionally, seal to 0555, then pin final state."""

    with ExitStack() as provisional:
        provisional_directory = _create_private_child_directory_v3(
            runtime_directory,
            _STAGED_INPUT_DIRECTORY_V3,
            label="private staged recognizer input",
            code=FAIL_INPUT_BINDING,
        )
        provisional.callback(provisional_directory.close)
        provisional_input = _write_new_regular_file_v3(
            provisional_directory,
            RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
            input_payload,
            final_mode=0o444,
            maximum_bytes=MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
            minimum_bytes=ARCHIVE_HEADER_BYTES_V2,
            label="staged recognizer input archive",
            code=FAIL_INPUT_BINDING,
        )
        provisional.callback(provisional_input.close)
        provisional_manifest = _write_new_regular_file_v3(
            provisional_directory,
            EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
            manifest_payload,
            final_mode=0o444,
            maximum_bytes=MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
            minimum_bytes=2,
            label="staged execution freeze manifest",
            code=FAIL_INPUT_BINDING,
        )
        provisional.callback(provisional_manifest.close)
        os.fchmod(provisional_directory.descriptor, 0o555)
        os.fsync(provisional_directory.descriptor)
        os.fsync(runtime_directory.descriptor)

    final_directory: _PinnedDirectoryV3 | None = None
    final_input: _PinnedRegularFileV3 | None = None
    final_manifest: _PinnedRegularFileV3 | None = None
    try:
        final_directory = _open_componentwise_directory_v3(
            runtime_directory.path / _STAGED_INPUT_DIRECTORY_V3,
            label="sealed staged recognizer input",
            code=FAIL_INPUT_BINDING,
        )
        final_input = _open_relative_regular_file_v3(
            final_directory,
            RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
            maximum_bytes=MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
            minimum_bytes=ARCHIVE_HEADER_BYTES_V2,
            single_link_required=True,
            label="sealed staged recognizer input archive",
            code=FAIL_INPUT_BINDING,
        )
        final_manifest = _open_relative_regular_file_v3(
            final_directory,
            EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
            maximum_bytes=MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
            minimum_bytes=2,
            single_link_required=True,
            label="sealed staged execution freeze manifest",
            code=FAIL_INPUT_BINDING,
        )
        if (
            stat.S_IMODE(final_directory.final_identity.mode) != 0o555
            or stat.S_IMODE(final_input.metadata_fingerprint[2]) != 0o444
            or stat.S_IMODE(final_manifest.metadata_fingerprint[2]) != 0o444
            or final_input.payload != input_payload
            or final_manifest.payload != manifest_payload
            or _directory_names_v3(
                final_directory,
                label="sealed staged recognizer input",
                code=FAIL_INPUT_BINDING,
            )
            != frozenset(
                {
                    RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                    EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
                }
            )
        ):
            _fail(FAIL_INPUT_BINDING, "sealed staged input final binding drift")
        _recheck_pinned_directory_v3(
            final_directory,
            label="sealed staged recognizer input",
            code=FAIL_INPUT_BINDING,
        )
        return final_directory, final_input, final_manifest
    except BaseException:
        if final_manifest is not None:
            final_manifest.close()
        if final_input is not None:
            final_input.close()
        if final_directory is not None:
            final_directory.close()
        raise


def _unique_json_object_v3(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("JSON object repeats a key")
        result[key] = value
    return result


def _reject_json_constant_v3(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json_v3(payload: bytes, *, label: str, code: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object_v3,
            parse_constant=_reject_json_constant_v3,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        _fail(code, f"{label} is not strict duplicate-free JSON: {type(exc).__name__}")


def _canonical_ascii_json_line_v3(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _extract_container_result_wire_v3(payload: bytes) -> tuple[bytes, bytes]:
    """Decode the bounded supervisor receipt/prediction wire from ``docker exec``."""

    header_size = len(_CONTAINER_RESULT_WIRE_MAGIC_V3) + 8
    if (
        type(payload) is not bytes
        or not header_size + 2 + PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= len(payload)
        <= MAXIMUM_DOCKER_RESULT_WIRE_BYTES_V3
        or not payload.startswith(_CONTAINER_RESULT_WIRE_MAGIC_V3)
    ):
        _fail(FAIL_OUTPUT_BINDING, "container result wire byte bound/magic drift")
    offset = len(_CONTAINER_RESULT_WIRE_MAGIC_V3)
    receipt_size = int.from_bytes(payload[offset : offset + 4], "big")
    prediction_size = int.from_bytes(payload[offset + 4 : offset + 8], "big")
    offset += 8
    if (
        not 2 <= receipt_size <= 65_536
        or not PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= prediction_size
        <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
        or len(payload) != offset + receipt_size + prediction_size
    ):
        _fail(FAIL_OUTPUT_BINDING, "container result wire length projection drift")
    receipt = payload[offset : offset + receipt_size]
    prediction = payload[offset + receipt_size :]
    if not receipt.endswith(b"\n") or receipt.count(b"\n") != 1:
        _fail(FAIL_OUTPUT_BINDING, "container result wire receipt framing drift")
    return receipt, prediction


@dataclass(frozen=True, slots=True)
class _BoundedCompletedProcessV3:
    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    started_monotonic_ns: int
    finished_monotonic_ns: int


def _kill_process_group_v3(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _kill_and_reap_process_v3(
    process: subprocess.Popen[bytes],
    *,
    code: str,
) -> None:
    if process.poll() is None:
        _kill_process_group_v3(process)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _fail(code, f"Docker client process could not be reaped: {type(exc).__name__}")


def _run_bounded_command_v3(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: Path,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    timeout_code: str,
    execution_code: str,
    forced_cleanup: Callable[[], None] | None = None,
) -> _BoundedCompletedProcessV3:
    if (
        type(command) not in {tuple, list}
        or not command
        or any(type(item) is not str or not item or "\x00" in item for item in command)
        or command[0] != DEFAULT_DOCKER_EXECUTABLE.as_posix()
        or tuple(command)[1:2] != (f"--host={LOCAL_DOCKER_HOST}",)
    ):
        _fail(execution_code, "Docker command vector is not exact and absolute")
    if (
        type(timeout_seconds) is not int
        or timeout_seconds < 1
        or type(maximum_stdout_bytes) is not int
        or maximum_stdout_bytes < 1
        or type(maximum_stderr_bytes) is not int
        or maximum_stderr_bytes < 1
    ):
        _fail(execution_code, "bounded process limits are malformed")
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow_label: str | None = None
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    cleanup_completed = False

    def cleanup_once() -> None:
        nonlocal cleanup_completed
        if forced_cleanup is not None and not cleanup_completed:
            forced_cleanup()
            cleanup_completed = True

    try:
        process = subprocess.Popen(
            list(command),
            cwd=working_directory.as_posix(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            _fail(execution_code, "Docker process pipes are unavailable")
        streams = {
            process.stdout.fileno(): ("stdout", process.stdout, stdout_buffer, maximum_stdout_bytes),
            process.stderr.fileno(): ("stderr", process.stderr, stderr_buffer, maximum_stderr_bytes),
        }
        for descriptor, row in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(row[1], selectors.EVENT_READ, data=descriptor)
        open_descriptors = set(streams)
        while open_descriptors:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_and_reap_process_v3(process, code=execution_code)
                cleanup_once()
                _fail(timeout_code, f"Docker command exceeded exact {timeout_seconds}s timeout")
            events = selector.select(timeout=min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = tuple(
                    (selectors.SelectorKey(
                        fileobj=streams[item][1],
                        fd=item,
                        events=selectors.EVENT_READ,
                        data=item,
                    ), selectors.EVENT_READ)
                    for item in tuple(open_descriptors)
                )
            for key, _mask in events:
                descriptor = key.data
                if descriptor not in open_descriptors:
                    continue
                name, stream, buffer, cap = streams[descriptor]
                try:
                    chunk = os.read(descriptor, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    open_descriptors.remove(descriptor)
                    continue
                remaining_capacity = cap - len(buffer)
                if len(chunk) > remaining_capacity:
                    if remaining_capacity > 0:
                        buffer.extend(chunk[:remaining_capacity])
                    overflow_label = name
                    _kill_and_reap_process_v3(process, code=execution_code)
                    break
                buffer.extend(chunk)
            if overflow_label is not None:
                cleanup_once()
                _fail(
                    FAIL_DOCKER_CAPTURE,
                    f"Docker {overflow_label} exceeded its exact byte cap",
                )
        try:
            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_and_reap_process_v3(process, code=execution_code)
            cleanup_once()
            _fail(timeout_code, f"Docker command exceeded exact {timeout_seconds}s timeout")
        finished = time.monotonic_ns()
        return _BoundedCompletedProcessV3(
            command=tuple(command),
            returncode=returncode,
            stdout=bytes(stdout_buffer),
            stderr=bytes(stderr_buffer),
            started_monotonic_ns=started,
            finished_monotonic_ns=finished,
        )
    except Phase2BRecognizerHostExecutorErrorV3:
        if process is not None:
            _kill_and_reap_process_v3(process, code=execution_code)
        cleanup_once()
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        if process is not None:
            _kill_and_reap_process_v3(process, code=execution_code)
        cleanup_once()
        _fail(execution_code, f"Docker command failed locally: {type(exc).__name__}")
    except BaseException:
        try:
            if process is not None:
                _kill_and_reap_process_v3(process, code=execution_code)
        except Exception:
            pass
        try:
            cleanup_once()
        except Exception:
            pass
        raise
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _validate_control_plane_v3(
    control_plane: LocalDockerControlPlaneV1,
    *,
    runtime_root: Path,
) -> None:
    if type(control_plane) is not LocalDockerControlPlaneV1:
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker control plane DTO differs")
    expected_keys = {
        "DOCKER_CONFIG",
        "DOCKER_HOST",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
    }
    if type(control_plane.environment) is not MappingProxyType:
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker control-plane environment is mutable")
    environment = dict(control_plane.environment)
    expected_environment = {
        "DOCKER_CONFIG": (runtime_root / "docker-config").as_posix(),
        "DOCKER_HOST": LOCAL_DOCKER_HOST,
        "HOME": (runtime_root / "docker-home").as_posix(),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        expected_socket = DEFAULT_DOCKER_SOCKET.resolve(strict=True)
    except OSError as exc:
        _fail(
            FAIL_DOCKER_CONTROL_PLANE,
            f"exact Docker Unix socket is unavailable: {type(exc).__name__}",
        )
    if (
        control_plane.executable != DEFAULT_DOCKER_EXECUTABLE
        or control_plane.socket_path != expected_socket
        or control_plane.config_directory != runtime_root / "docker-config"
        or set(environment) != expected_keys
        or environment != expected_environment
        or type(control_plane.binding) is not MappingProxyType
        or frozenset(control_plane.binding) != frozenset(
            {
                "schema",
                "docker_executable",
                "docker_executable_sha256",
                "docker_socket_requested",
                "docker_socket_resolved",
                "docker_host",
                "docker_config_sha256",
                "environment_keys",
                "proxy_environment_keys",
                "network_endpoint_kind",
                "runtime_mount_id",
                "runtime_filesystem_type",
            }
        )
        or control_plane.binding.get("docker_executable")
        != DEFAULT_DOCKER_EXECUTABLE.as_posix()
        or control_plane.binding.get("docker_socket_requested")
        != DEFAULT_DOCKER_SOCKET.as_posix()
        or control_plane.binding.get("docker_socket_resolved")
        != expected_socket.as_posix()
        or control_plane.binding.get("docker_host") != LOCAL_DOCKER_HOST
        or control_plane.binding.get("environment_keys") != sorted(expected_keys)
        or control_plane.binding.get("proxy_environment_keys") != []
        or control_plane.binding.get("network_endpoint_kind")
        != "LOCAL_UNIX_SOCKET"
    ):
        _fail(
            FAIL_DOCKER_CONTROL_PLANE,
            "Docker executable/socket/config/environment binding drift",
        )
    try:
        socket_metadata = DEFAULT_DOCKER_SOCKET.lstat()
    except OSError as exc:
        _fail(
            FAIL_DOCKER_CONTROL_PLANE,
            f"exact Docker Unix socket is unavailable: {type(exc).__name__}",
        )
    if not stat.S_ISSOCK(socket_metadata.st_mode):
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker endpoint is not a Unix socket")


def _validate_image_inspect_v3(
    payload: bytes,
    *,
    spec: OciRecognizerRunSpec,
) -> tuple[str, str]:
    value = _decode_json_v3(payload, label="Docker image inspection", code=FAIL_IMAGE_BINDING)
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(FAIL_IMAGE_BINDING, "Docker image inspection shape differs")
    row = value[0]
    config = row.get("Config")
    rootfs = row.get("RootFS")
    if type(config) is not dict or type(rootfs) is not dict:
        _fail(FAIL_IMAGE_BINDING, "Docker image config/rootfs is absent")
    image_id = row.get("Id")
    if (
        image_id != spec.image_digest
        or image_id != RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3
        or spec.image_digest != RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3
        or type(image_id) is not str
        or _DIGEST_REF.fullmatch(image_id) is None
        or row.get("Os") != "linux"
        or row.get("Architecture") != "amd64"
        or frozenset(config) != _IMAGE_CONFIG_FIELDS_V3
        or config.get("ArgsEscaped") is not _IMAGE_ARGS_ESCAPED_V3
        or config.get("User") != RECOGNIZER_RUNTIME_UID_GID
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Entrypoint") != [FROZEN_RECOGNIZER_ENTRYPOINT[0]]
        or config.get("Cmd") != list(FROZEN_RECOGNIZER_ENTRYPOINT[1:])
        or rootfs.get("Type") != "layers"
        or type(rootfs.get("Layers")) is not list
        or not rootfs["Layers"]
        or any(
            type(item) is not str or _DIGEST_REF.fullmatch(item) is None
            for item in rootfs["Layers"]
        )
    ):
        _fail(FAIL_IMAGE_BINDING, "Docker image identity/platform/config binding drift")
    environment = config.get("Env")
    if type(environment) is not list or tuple(environment) != _IMAGE_ENVIRONMENT_V3:
        _fail(FAIL_IMAGE_BINDING, "Docker image exact environment binding drift")
    labels = config.get("Labels")
    if type(labels) is not dict or any(
        type(key) is not str or type(item) is not str
        for key, item in labels.items()
    ):
        _fail(FAIL_IMAGE_BINDING, "Docker image label shape differs")
    required_labels = {
        "org.opencontainers.image.title": _IMAGE_TITLE_V3,
        "org.opencontainers.image.description": _IMAGE_DESCRIPTION_V3,
        "org.opencontainers.image.base.digest": _IMAGE_BASE_DIGEST_V3,
        "org.hegel-machine.phase2b.actor-profile.content-id": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
        ),
        "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        "org.hegel-machine.phase2b.owner-accepted-authority-requirement.id": (
            OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
        ),
        "org.hegel-machine.phase2b.recognizer-source-root.sha256": (
            RECOGNIZER_SOURCE_ROOT_SHA256_V3
        ),
    }
    if labels != required_labels:
        _fail(FAIL_IMAGE_BINDING, "Docker image frozen label binding drift")
    # Docker's image Id is the digest of the full config blob.  The canonical
    # projection hash is retained as convenient local replay evidence; the Id
    # comparison above is the exact full-config content binding.
    return image_id, stable_hash(config)


def _validate_runtime_receipt_v3(
    payload: bytes,
    *,
    input_archive: DecodedRecognizerInputArchiveV2,
    input_archive_sha256: str,
    execution_manifest_id: str,
    execution_manifest_sha256: str,
    prediction_archive: DecodedRecognizerPredictionArchiveV2,
    prediction_archive_sha256: str,
    prediction_archive_bytes: int,
) -> str:
    value = _decode_json_v3(payload, label="recognizer runtime receipt", code=FAIL_OUTPUT_BINDING)
    if type(value) is not dict or frozenset(value) != _RUNTIME_RECEIPT_FIELDS_V3:
        _fail(FAIL_OUTPUT_BINDING, "recognizer runtime receipt fields differ")
    if _canonical_ascii_json_line_v3(value) != payload:
        _fail(FAIL_OUTPUT_BINDING, "recognizer runtime receipt is not canonical one-line JSON")
    receipt_without_id = dict(value)
    receipt_id = receipt_without_id.pop("receipt_id")
    expected_receipt_id = stable_hash(
        receipt_without_id,
        prefix="phase2b_recognizer_runtime_receipt_v2_",
    )
    expected = {
        "schema_version": RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2,
        "disposition": "COMPLETE",
        "reason": "phase2b_recognizer_runtime_v2_complete",
        "input_archive_id": input_archive.archive_id,
        "input_archive_sha256": input_archive_sha256,
        "execution_freeze_manifest_id": execution_manifest_id,
        "execution_freeze_manifest_sha256": execution_manifest_sha256,
        "prediction_archive_id": prediction_archive.archive_id,
        "prediction_archive_sha256": prediction_archive_sha256,
        "prediction_case_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "prediction_archive_bytes": prediction_archive_bytes,
        "output_filename": RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
    }
    if receipt_without_id != expected or receipt_id != expected_receipt_id:
        _fail(FAIL_OUTPUT_BINDING, "recognizer runtime receipt/output cross-binding drift")
    return expected_receipt_id


def _container_name_v3(runtime_root: Path) -> str:
    value = "hegel-phase2b-recognizer-v3-" + _sha256(
        (runtime_root.as_posix() + "\x00" + str(os.getpid())).encode("utf-8")
    )[:24]
    if _CONTAINER_NAME_V3.fullmatch(value) is None:
        _fail(FAIL_DOCKER_EXECUTION, "private container name derivation drift")
    return value


def _docker_create_command_v3(
    control_plane: LocalDockerControlPlaneV1,
    *,
    spec: OciRecognizerRunSpec,
    container_name: str,
    staged_input_directory: Path,
    staged_seccomp_path: Path,
) -> tuple[str, ...]:
    if (
        _CONTAINER_NAME_V3.fullmatch(container_name) is None
        or type(staged_input_directory) is not _EXACT_PATH_TYPE_V3
        or type(staged_seccomp_path) is not _EXACT_PATH_TYPE_V3
    ):
        _fail(FAIL_PREFLIGHT, "Docker create binding arguments drift")
    limits = spec.limits
    command = tuple(
        control_plane.command(
            "container",
            "create",
            f"--name={container_name}",
            "--platform=linux/amd64",
            "--no-healthcheck",
            "--pull=never",
            "--network=none",
            "--ipc=none",
            "--read-only",
            f"--user={RECOGNIZER_RUNTIME_UID_GID}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--security-opt=seccomp={staged_seccomp_path.as_posix()}",
            f"--entrypoint={_CONTAINER_SUPERVISOR_ENTRYPOINT_V3}",
            "--workdir=/opt/hegel",
            "--restart=no",
            "--stop-timeout=1",
            f"--cpus={limits.cpus}",
            f"--memory={limits.memory}",
            f"--pids-limit={limits.pids}",
            f"--memory-swap={limits.memory}",
            "--ulimit=nofile=256:256",
            (
                "--ulimit=fsize="
                f"{limits.maximum_output_bytes}:{limits.maximum_output_bytes}"
            ),
            "--env=PYTHONHASHSEED=0",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=PYTHONPATH=/opt/hegel/src",
            "--env=HOME=/nonexistent",
            "--env=NO_PROXY=*",
            (
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,"
                f"size={limits.tmpfs_bytes},mode=0700,uid=65534,gid=65534"
            ),
            (
                "--mount=type=bind,src="
                f"{staged_input_directory.as_posix()},"
                "dst=/phase2b/input,readonly"
            ),
            (
                "--tmpfs=/phase2b/output:rw,noexec,nosuid,nodev,"
                f"size={limits.maximum_output_bytes},mode=0700,uid=65534,gid=65534"
            ),
            spec.image_digest,
            "-c",
            _CONTAINER_SUPERVISOR_CODE_V3,
            *spec.entrypoint[1:],
        )
    )
    if any(
        type(item) is not str
        or not item
        or "\x00" in item
        or (item.startswith("--mount=") and item.count(",") != 3)
        for item in command
    ):
        _fail(FAIL_PREFLIGHT, "Docker create vector token drift")
    return command


def _parse_created_container_id_v3(payload: bytes) -> str:
    try:
        value = payload.decode("ascii")
    except UnicodeError as exc:
        _fail(FAIL_DOCKER_EXECUTION, f"Docker create ID is not ASCII: {type(exc).__name__}")
    if not value.endswith("\n") or _CONTAINER_ID_V3.fullmatch(value[:-1]) is None:
        _fail(FAIL_DOCKER_EXECUTION, "Docker create did not return one exact full ID")
    return value[:-1]


def _validate_created_container_inspect_v3(
    payload: bytes,
    *,
    container_id: str,
    container_name: str,
    spec: OciRecognizerRunSpec,
    staged_input_directory: Path,
    expected_seccomp: Mapping[str, object],
) -> None:
    value = _decode_json_v3(
        payload,
        label="created recognizer container inspection",
        code=FAIL_DOCKER_EXECUTION,
    )
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(FAIL_DOCKER_EXECUTION, "created container inspection shape drift")
    row = value[0]
    state = row.get("State")
    config = row.get("Config")
    host = row.get("HostConfig")
    mounts = row.get("Mounts")
    network = row.get("NetworkSettings")
    if (
        any(type(item) is not dict for item in (state, config, host, network))
        or type(mounts) is not list
    ):
        _fail(FAIL_DOCKER_EXECUTION, "created container inspection sections are absent")
    security_opt = host.get("SecurityOpt")
    tmpfs = host.get("Tmpfs")
    ulimits = host.get("Ulimits")
    expected_input_mount = {
        "Type": "bind",
        "Source": staged_input_directory.as_posix(),
        "Destination": "/phase2b/input",
        "Mode": "",
        "RW": False,
        "Propagation": "rprivate",
    }
    if type(security_opt) is not list or len(security_opt) != 2:
        _fail(FAIL_DOCKER_EXECUTION, "created container security options drift")
    seccomp_options = tuple(
        item for item in security_opt if type(item) is str and item.startswith("seccomp=")
    )
    if len(seccomp_options) != 1:
        _fail(FAIL_DOCKER_EXECUTION, "created container seccomp option is absent")
    observed_seccomp = _decode_json_v3(
        seccomp_options[0][len("seccomp=") :].encode("utf-8"),
        label="created container embedded seccomp",
        code=FAIL_DOCKER_EXECUTION,
    )
    expected_tmpfs = {
        "/tmp": (
            "rw,noexec,nosuid,nodev,size=268435456,"
            "mode=0700,uid=65534,gid=65534"
        ),
        "/phase2b/output": (
            "rw,noexec,nosuid,nodev,size=536870912,"
            "mode=0700,uid=65534,gid=65534"
        ),
    }
    expected_ulimits = {
        ("fsize", spec.limits.maximum_output_bytes, spec.limits.maximum_output_bytes),
        ("nofile", 256, 256),
    }
    observed_ulimits = (
        {
            (item.get("Name"), item.get("Hard"), item.get("Soft"))
            for item in ulimits
        }
        if type(ulimits) is list and all(type(item) is dict for item in ulimits)
        else set()
    )
    if (
        row.get("Id") != container_id
        or row.get("Name") != "/" + container_name
        or row.get("Image") != spec.image_digest
        or row.get("Path") != _CONTAINER_SUPERVISOR_ENTRYPOINT_V3
        or row.get("Args")
        != ["-c", _CONTAINER_SUPERVISOR_CODE_V3, *spec.entrypoint[1:]]
        or state.get("Status") != "created"
        or state.get("Running") is not False
        or config.get("Image") != spec.image_digest
        or config.get("User") != RECOGNIZER_RUNTIME_UID_GID
        or config.get("Env") != list(_CONTAINER_ENVIRONMENT_V3)
        or config.get("Entrypoint") != [_CONTAINER_SUPERVISOR_ENTRYPOINT_V3]
        or config.get("Cmd")
        != ["-c", _CONTAINER_SUPERVISOR_CODE_V3, *spec.entrypoint[1:]]
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Healthcheck") != {"Test": ["NONE"]}
        or config.get("Volumes") is not None
        or host.get("AutoRemove") is not False
        or host.get("ReadonlyRootfs") is not True
        or host.get("NetworkMode") != "none"
        or host.get("IpcMode") != "none"
        or host.get("CapDrop") != ["ALL"]
        or "no-new-privileges:true" not in security_opt
        or observed_seccomp != dict(expected_seccomp)
        or tmpfs != expected_tmpfs
        or observed_ulimits != expected_ulimits
        or host.get("Memory") != 2_147_483_648
        or host.get("MemorySwap") != 2_147_483_648
        or host.get("NanoCpus") != 2_000_000_000
        or host.get("PidsLimit") != 128
        or host.get("Privileged") is not False
        or host.get("CgroupnsMode") != "private"
        or host.get("PidMode") != ""
        or host.get("UsernsMode") != ""
        or host.get("Runtime") != "runc"
        or mounts != [expected_input_mount]
        or network.get("Ports") not in ({}, None)
    ):
        _fail(FAIL_DOCKER_EXECUTION, "created container isolation binding drift")


def _force_remove_container_v3(
    *,
    control_plane: LocalDockerControlPlaneV1,
    environment: Mapping[str, str],
    working_directory: Path,
    container_id: str,
    allow_already_absent: bool = False,
) -> None:
    if (
        _CONTAINER_ID_V3.fullmatch(container_id) is None
        and _CONTAINER_NAME_V3.fullmatch(container_id) is None
    ):
        _fail(FAIL_DOCKER_EXECUTION, "container cleanup ID drift")
    if type(allow_already_absent) is not bool:
        _fail(FAIL_DOCKER_EXECUTION, "container cleanup absence policy drift")
    removed = _run_bounded_command_v3(
        tuple(control_plane.command("container", "rm", "--force", container_id)),
        environment=environment,
        working_directory=working_directory,
        timeout_seconds=60,
        maximum_stdout_bytes=256,
        maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
        timeout_code=FAIL_DOCKER_TIMEOUT,
        execution_code=FAIL_DOCKER_EXECUTION,
    )
    inspected = _run_bounded_command_v3(
        tuple(control_plane.command("container", "inspect", container_id)),
        environment=environment,
        working_directory=working_directory,
        timeout_seconds=30,
        maximum_stdout_bytes=1_024,
        maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
        timeout_code=FAIL_DOCKER_TIMEOUT,
        execution_code=FAIL_DOCKER_EXECUTION,
    )
    absent_stderr = (
        f"Error response from daemon: No such container: {container_id}\n"
    ).encode("ascii")
    removed_existing = (
        removed.returncode == 0
        and removed.stdout == (container_id + "\n").encode("ascii")
        and removed.stderr == b""
    )
    removed_absent = (
        removed.returncode == 0
        and removed.stdout == b""
        and removed.stderr == absent_stderr
    )
    inspect_proves_absent = (
        inspected.returncode == 1
        and inspected.stdout == b"[]\n"
        and inspected.stderr == absent_stderr
    )
    if (
        not inspect_proves_absent
        or not (removed_existing or (allow_already_absent and removed_absent))
    ):
        _fail(
            FAIL_DOCKER_EXECUTION,
            "recognizer container removal/absence proof drift",
        )


def _crossbind_prediction_to_input_v3(
    prediction: DecodedRecognizerPredictionArchiveV2,
    *,
    input_archive: DecodedRecognizerInputArchiveV2,
    input_archive_sha256: str,
    execution_manifest_id: str,
    protocol_id: str,
) -> None:
    context = prediction.context
    if (
        context.input_archive_id != input_archive.archive_id
        or context.input_archive_sha256 != input_archive_sha256
        or context.execution_freeze_manifest_id != execution_manifest_id
        or context.protocol_id != protocol_id
        or context.batch_id != input_archive.batch_id
        or context.batch_policy_id != input_archive.batch_policy_id
        or tuple(prediction.input_row_ids) != tuple(input_archive.row_ids)
        or tuple(item.input_row_id for item in prediction.records)
        != tuple(input_archive.row_ids)
    ):
        _fail(FAIL_OUTPUT_BINDING, "prediction archive does not bind this input/freeze/order")


def _cleanup_exact_output_inodes_v3(
    output_directory: _PinnedDirectoryV3,
    bindings: Sequence[tuple[str, tuple[int, int]]],
) -> bool:
    """Remove only the named output inodes created by this invocation."""

    cleanup_complete = True
    for name, expected_identity in reversed(tuple(bindings)):
        try:
            metadata = os.stat(
                name,
                dir_fd=output_directory.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        except BaseException:
            cleanup_complete = False
            continue
        if (metadata.st_dev, metadata.st_ino) != expected_identity:
            cleanup_complete = False
            continue
        try:
            os.unlink(name, dir_fd=output_directory.descriptor)
        except BaseException:
            cleanup_complete = False
    try:
        os.fsync(output_directory.descriptor)
    except BaseException:
        cleanup_complete = False
    try:
        if _directory_names_v3(
            output_directory,
            label="host prediction output after failure cleanup",
            code=FAIL_OUTPUT_BINDING,
        ):
            cleanup_complete = False
    except BaseException:
        cleanup_complete = False
    return cleanup_complete


def _publish_prediction_output_v3(
    output_directory: _PinnedDirectoryV3,
    payload: bytes,
) -> _PinnedRegularFileV3:
    if type(payload) is not bytes or not (
        PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= len(payload)
        <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
    ):
        _fail(FAIL_OUTPUT_BINDING, "prediction publication payload bound drift")
    if _directory_names_v3(
        output_directory,
        label="host prediction output before publication",
        code=FAIL_OUTPUT_BINDING,
    ):
        _fail(FAIL_OUTPUT_BINDING, "host prediction output is not empty before publication")
    descriptor: int | None = None
    opened: _PinnedRegularFileV3 | None = None
    created: list[tuple[str, tuple[int, int]]] = []
    try:
        descriptor = os.open(
            _OUTPUT_PENDING_FILENAME_V3,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _required_open_flag("O_NOFOLLOW", FAIL_OUTPUT_BINDING)
            | _required_open_flag("O_CLOEXEC", FAIL_OUTPUT_BINDING),
            0o600,
            dir_fd=output_directory.descriptor,
        )
        metadata = os.fstat(descriptor)
        created_identity = (metadata.st_dev, metadata.st_ino)
        created.append((_OUTPUT_PENDING_FILENAME_V3, created_identity))
        _write_all_v3(
            descriptor,
            payload,
            code=FAIL_OUTPUT_BINDING,
            label="host prediction output",
        )
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.fsync(output_directory.descriptor)
        os.link(
            _OUTPUT_PENDING_FILENAME_V3,
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
            src_dir_fd=output_directory.descriptor,
            dst_dir_fd=output_directory.descriptor,
            follow_symlinks=False,
        )
        created.append(
            (RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2, created_identity)
        )
        linked_metadata = os.stat(
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
            dir_fd=output_directory.descriptor,
            follow_symlinks=False,
        )
        if (linked_metadata.st_dev, linked_metadata.st_ino) != created_identity:
            _fail(FAIL_OUTPUT_BINDING, "host prediction hardlink inode binding drift")
        os.fsync(output_directory.descriptor)
        os.unlink(_OUTPUT_PENDING_FILENAME_V3, dir_fd=output_directory.descriptor)
        os.fsync(output_directory.descriptor)
        os.close(descriptor)
        descriptor = None
        if _directory_names_v3(
            output_directory,
            label="host prediction output after publication",
            code=FAIL_OUTPUT_BINDING,
        ) != frozenset({RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2}):
            _fail(FAIL_OUTPUT_BINDING, "host prediction output publication namespace drift")
        opened = _open_relative_regular_file_v3(
            output_directory,
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
            maximum_bytes=MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
            minimum_bytes=PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            single_link_required=True,
            label="host-published recognizer prediction archive",
            code=FAIL_OUTPUT_BINDING,
        )
        if opened.payload != payload:
            _fail(FAIL_OUTPUT_BINDING, "host-published prediction bytes drift")
        return opened
    except BaseException as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                pass
        if opened is not None:
            opened.close()
        if not _cleanup_exact_output_inodes_v3(output_directory, created):
            _fail(
                FAIL_OUTPUT_BINDING,
                "prediction publication failed and exact output cleanup was incomplete",
            )
        if isinstance(exc, OSError):
            _fail(
                FAIL_OUTPUT_BINDING,
                f"host prediction publication failed: {type(exc).__name__}",
            )
        raise


@dataclass(frozen=True, slots=True, init=False)
class Phase2BRecognizerLocalExecutionReceiptV3:
    """Privately issued success receipt for local mechanics only."""

    schema_version: str
    disposition: str
    reason: str
    run_spec_id: str
    freeze_manifest_id: str
    protocol_id: str
    image_digest: str
    observed_image_id: str
    recognizer_source_root_sha256: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    owner_accepted_authority_requirement_id: str
    image_args_escaped: bool
    image_inspect_sha256: str
    image_config_projection_sha256: str
    docker_control_plane_binding_sha256: str
    docker_executable_sha256: str
    seccomp_profile_sha256: str
    host_path_binding_sha256: str
    input_manifest_sha256: str
    input_archive_id: str
    input_archive_sha256: str
    execution_freeze_manifest_sha256: str
    command_sha256: str
    container_supervisor_sha256: str
    container_result_extractor_sha256: str
    process_exit_code: int
    timed_out: bool
    started_monotonic_ns: int
    finished_monotonic_ns: int
    stdout_bytes: int
    stdout_sha256: str
    stderr_bytes: int
    stderr_sha256: str
    runtime_receipt_id: str
    prediction_archive_id: str
    prediction_archive_sha256: str
    prediction_archive_bytes: int
    prediction_case_count: int
    output_file_identity_sha256: str
    docker_invoked: bool = field(init=False, default=True)
    local_mechanics_complete: bool = field(init=False, default=True)
    external_runtime_attestation_issued: bool = field(init=False, default=False)
    runtime_attestation_authority_verified: bool = field(init=False, default=False)
    actor_profile_authority_verified: bool = field(init=False, default=False)
    authority_requirement_satisfied: bool = field(init=False, default=False)
    recognizer_operation_signature_verified: bool = field(init=False, default=False)
    actual_input_archive_supplied: bool = field(init=False, default=False)
    actual_960_recognizer_executed: bool = field(init=False, default=False)
    actual_960_case_run_verified: bool = field(init=False, default=False)
    fresh_role_keys_generated: bool = field(init=False, default=False)
    key_generation_performed: bool = field(init=False, default=False)
    custodian_ceremony_executed: bool = field(init=False, default=False)
    signed_execution_registry_materialized: bool = field(init=False, default=False)
    signed_monotonic_time_materialized: bool = field(init=False, default=False)
    store_receipt_materialized: bool = field(init=False, default=False)
    prediction_commit_before_reveal_verified: bool = field(init=False, default=False)
    reveal_authority_verified: bool = field(init=False, default=False)
    scoring_performed: bool = field(init=False, default=False)
    prediction_scored: bool = field(init=False, default=False)
    formal_gate_evaluation_performed: bool = field(init=False, default=False)
    effect_evidence: bool = field(init=False, default=False)
    formal_c1_report_verified: bool = field(init=False, default=False)
    c1_exit_evidence: bool = field(init=False, default=False)
    malicious_host_root_resistance: bool = field(init=False, default=False)
    malicious_docker_daemon_resistance: bool = field(init=False, default=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("local recognizer execution receipts are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        **values: object,
    ) -> "Phase2BRecognizerLocalExecutionReceiptV3":
        if token is not _RECEIPT_ISSUE_TOKEN_V3:
            raise TypeError("local recognizer receipt issuer token drift")
        # Dataclass fields are declared init=False at the class level because
        # public construction is forbidden.  The mechanics fields precede the
        # fixed boolean claim boundary in declaration order.
        value_fields = tuple(
            item.name
            for item in fields(cls)
            if item.name not in _RECEIPT_FIXED_CLAIMS_V3
        )
        if frozenset(values) != frozenset(value_fields):
            raise TypeError("local recognizer receipt issuance field manifest drift")
        value = object.__new__(cls)
        for name in value_fields:
            object.__setattr__(value, name, values[name])
        object.__setattr__(value, "docker_invoked", True)
        object.__setattr__(value, "local_mechanics_complete", True)
        for name in _RECEIPT_FIXED_CLAIMS_V3[2:]:
            object.__setattr__(value, name, False)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not Phase2BRecognizerLocalExecutionReceiptV3:
            raise TypeError("local recognizer receipt exact type drift")
        if (
            self.schema_version != LOCAL_EXECUTION_RECEIPT_SCHEMA_V3
            or self.disposition != LOCAL_EXECUTION_DISPOSITION_V3
            or self.reason != LOCAL_EXECUTION_COMPLETE_REASON_V3
            or not self.run_spec_id
            or not self.freeze_manifest_id
            or not self.protocol_id
            or self.image_digest != self.observed_image_id
            or self.image_digest != RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3
            or _DIGEST_REF.fullmatch(self.image_digest) is None
            or self.recognizer_source_root_sha256
            != RECOGNIZER_SOURCE_ROOT_SHA256_V3
            or self.actor_profile_content_id
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
            or self.actor_profile_canonical_sha256
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
            or self.owner_accepted_authority_requirement_id
            != OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
            or self.image_args_escaped is not True
            or type(self.process_exit_code) is not int
            or self.process_exit_code != 0
            or self.timed_out is not False
            or type(self.started_monotonic_ns) is not int
            or type(self.finished_monotonic_ns) is not int
            or self.finished_monotonic_ns < self.started_monotonic_ns
            or type(self.stdout_bytes) is not int
            or not 1 <= self.stdout_bytes <= MAXIMUM_DOCKER_STDOUT_BYTES_V3
            or self.stderr_bytes != 0
            or self.stderr_sha256 != _sha256(b"")
            or self.prediction_case_count != TOTAL_RECOGNIZER_CASE_COUNT
            or not (
                PREDICTION_ARCHIVE_HEADER_BYTES_V2
                <= self.prediction_archive_bytes
                <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
            )
        ):
            raise ValueError("local recognizer execution receipt invariant drift")
        prefixed_ids = (
            (self.run_spec_id, "phase2b_oci_run_spec_"),
            (self.freeze_manifest_id, "phase2b_execution_freeze_"),
            (self.protocol_id, "phase2b_protocol_"),
            (self.runtime_receipt_id, "phase2b_recognizer_runtime_receipt_v2_"),
            (self.input_archive_id, "phase2b_recognizer_input_archive_v2_"),
            (
                self.prediction_archive_id,
                "phase2b_recognizer_prediction_archive_v2_",
            ),
        )
        if any(
            type(value) is not str
            or not value.startswith(prefix)
            or _LOWER_SHA256.fullmatch(value[len(prefix) :]) is None
            for value, prefix in prefixed_ids
        ):
            raise ValueError("local recognizer receipt content-address identity drift")
        for name in (
            "image_inspect_sha256",
            "image_config_projection_sha256",
            "recognizer_source_root_sha256",
            "actor_profile_canonical_sha256",
            "docker_control_plane_binding_sha256",
            "docker_executable_sha256",
            "seccomp_profile_sha256",
            "host_path_binding_sha256",
            "input_manifest_sha256",
            "input_archive_sha256",
            "execution_freeze_manifest_sha256",
            "command_sha256",
            "container_supervisor_sha256",
            "container_result_extractor_sha256",
            "stdout_sha256",
            "prediction_archive_sha256",
            "output_file_identity_sha256",
        ):
            if _LOWER_SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"local receipt {name} is not SHA-256")
        if (
            self.docker_invoked is not True
            or self.local_mechanics_complete is not True
            or any(
                getattr(self, name) is not False
                for name in _RECEIPT_FIXED_CLAIMS_V3[2:]
            )
        ):
            raise ValueError("local recognizer receipt claim-boundary drift")

    @property
    def receipt_id(self) -> str:
        return stable_hash(
            self,
            prefix="phase2b_recognizer_local_execution_receipt_v3_",
        )

    @property
    def claim_boundary(self) -> Mapping[str, bool]:
        return MappingProxyType(
            {name: False for name in _RECEIPT_FIXED_CLAIMS_V3[2:]}
        )


def _issue_receipt_after_publication_v3(
    output_directory: _PinnedDirectoryV3,
    prediction_file: _PinnedRegularFileV3,
    **values: object,
) -> Phase2BRecognizerLocalExecutionReceiptV3:
    """Issue the receipt or durably remove only the exact published file."""

    try:
        return Phase2BRecognizerLocalExecutionReceiptV3._issue(
            _RECEIPT_ISSUE_TOKEN_V3,
            **values,
        )
    except BaseException:
        expected_fingerprint = prediction_file.metadata_fingerprint
        expected_identity = (expected_fingerprint[0], expected_fingerprint[1])
        prediction_file.close()
        cleanup_complete = True
        try:
            metadata = os.stat(
                RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
                dir_fd=output_directory.descriptor,
                follow_symlinks=False,
            )
            if _regular_fingerprint(metadata) != expected_fingerprint:
                cleanup_complete = False
        except BaseException:
            cleanup_complete = False
        if cleanup_complete and not _cleanup_exact_output_inodes_v3(
            output_directory,
            ((RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2, expected_identity),),
        ):
            cleanup_complete = False
        if not cleanup_complete:
            _fail(
                FAIL_OUTPUT_BINDING,
                "receipt issuance failed and exact prediction cleanup was incomplete",
            )
        raise


def execute_phase2b_recognizer_docker_v3(
    spec: OciRecognizerRunSpec,
    *,
    runtime_parent: Path = Path("/tmp"),
) -> Phase2BRecognizerLocalExecutionReceiptV3:
    """Execute one exact local recognizer run and issue a non-authority receipt.

    All host material is pinned and decoded before the first Docker subprocess.
    The image-inspection subprocess is local-only; after it succeeds every
    binding is rechecked before the recognizer process is started.  Any failure
    raises a stable error and no success receipt is returned.
    """

    if type(spec) is not OciRecognizerRunSpec:
        _fail(FAIL_PREFLIGHT, "executor requires exact OciRecognizerRunSpec")
    try:
        supplied_values = {
            item.name: getattr(spec, item.name) for item in fields(OciRecognizerRunSpec)
        }
        supplied_limits = supplied_values["limits"]
        if type(supplied_limits) is not RuntimeLimits:
            _fail(FAIL_PREFLIGHT, "executor requires exact RuntimeLimits")
        limit_values = {
            item.name: getattr(supplied_limits, item.name) for item in fields(RuntimeLimits)
        }
    except Phase2BRecognizerHostExecutorErrorV3:
        raise
    except Exception as exc:
        _fail(FAIL_PREFLIGHT, f"run specification snapshot rejected: {type(exc).__name__}")
    string_fields = (
        "freeze_manifest_id",
        "protocol_id",
        "image_digest",
        "input_host_directory",
        "output_host_directory",
        "repository_host_directory",
        "seccomp_profile_host_path",
        "seccomp_profile_sha256",
        "input_manifest_sha256",
    )
    if (
        any(type(supplied_values[name]) is not str for name in string_fields)
        or type(supplied_values["expected_case_count"]) is not int
        or type(supplied_values["entrypoint"]) is not tuple
        or any(type(item) is not str for item in supplied_values["entrypoint"])
    ):
        _fail(FAIL_PREFLIGHT, "run specification primitive type manifest drift")
    try:
        if getattr(spec, "limits") is not supplied_limits or any(
            getattr(spec, item.name) != supplied_values[item.name]
            for item in fields(OciRecognizerRunSpec)
            if item.name != "limits"
        ) or any(
            getattr(supplied_limits, item.name) != limit_values[item.name]
            for item in fields(RuntimeLimits)
        ):
            _fail(FAIL_PREFLIGHT, "run specification changed during snapshot")
    except Phase2BRecognizerHostExecutorErrorV3:
        raise
    except Exception as exc:
        _fail(FAIL_PREFLIGHT, f"run specification recheck rejected: {type(exc).__name__}")
    frozen_limits = RuntimeLimits()
    if tuple(limit_values[item.name] for item in fields(RuntimeLimits)) != tuple(
        getattr(frozen_limits, item.name) for item in fields(RuntimeLimits)
    ) or any(
        type(limit_values[item.name]) is not type(getattr(frozen_limits, item.name))
        for item in fields(RuntimeLimits)
    ):
        _fail(FAIL_PREFLIGHT, "recognizer runtime limits differ from the frozen values")
    try:
        spec = OciRecognizerRunSpec(
            **{
                name: value
                for name, value in supplied_values.items()
                if name != "limits"
            },
            limits=RuntimeLimits(**limit_values),
        )
    except (TypeError, ValueError) as exc:
        _fail(FAIL_PREFLIGHT, f"run specification rejected: {type(exc).__name__}")
    if spec.image_digest != RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3:
        _fail(
            FAIL_IMAGE_BINDING,
            "run specification does not bind the final V3 recognizer image digest",
        )
    if spec.entrypoint != FROZEN_RECOGNIZER_ENTRYPOINT:
        _fail(FAIL_PREFLIGHT, "run specification default entrypoint/Cmd binding drift")
    if type(runtime_parent) is not _EXACT_PATH_TYPE_V3:
        _fail(FAIL_PREFLIGHT, "runtime parent must be exact Path")
    runtime_parent = _exact_absolute_path(
        runtime_parent.as_posix(),
        "recognizer private runtime parent",
        FAIL_HOST_PATH,
    )

    input_path = _exact_absolute_path(
        spec.input_host_directory, "recognizer input directory", FAIL_HOST_PATH
    )
    output_path = _exact_absolute_path(
        spec.output_host_directory, "recognizer output directory", FAIL_HOST_PATH
    )
    repository_path = _exact_absolute_path(
        spec.repository_host_directory, "repository directory", FAIL_HOST_PATH
    )
    try:
        input_mount = validate_linux_local_host_path_v1(
            input_path, repository_root=repository_path
        )
        output_mount = validate_linux_local_host_path_v1(
            output_path, repository_root=repository_path
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_HOST_PATH, f"input/output Linux-local validation rejected: {exc.code}")

    with ExitStack() as stack:
        runtime_parent_directory = _open_componentwise_directory_v3(
            runtime_parent,
            label="recognizer private runtime parent",
            code=FAIL_HOST_PATH,
        )
        stack.callback(runtime_parent_directory.close)
        input_directory = _open_componentwise_directory_v3(
            input_path, label="recognizer input directory"
        )
        stack.callback(input_directory.close)
        output_directory = _open_componentwise_directory_v3(
            output_path, label="recognizer output directory"
        )
        stack.callback(output_directory.close)
        repository_directory = _open_componentwise_directory_v3(
            repository_path, label="repository directory"
        )
        stack.callback(repository_directory.close)
        directories = {
            "input": input_directory,
            "output": output_directory,
            "repository": repository_directory,
        }
        _require_pairwise_nonoverlap_v3(directories)
        runtime_parent_ancestors = {
            item.inode_key for item in runtime_parent_directory.identities
        }
        if any(
            directory.final_identity.inode_key in runtime_parent_ancestors
            for directory in directories.values()
        ):
            _fail(
                FAIL_HOST_PATH,
                "runtime parent equals or is nested inside an input/output/repository root",
            )
        expected_input_names = frozenset(
            {
                RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
            }
        )
        if _directory_names_v3(
            input_directory, label="recognizer input", code=FAIL_INPUT_BINDING
        ) != expected_input_names:
            _fail(FAIL_INPUT_BINDING, "recognizer input namespace is not exact-two")
        if _directory_names_v3(
            output_directory, label="recognizer output", code=FAIL_OUTPUT_BINDING
        ):
            _fail(FAIL_OUTPUT_BINDING, "recognizer output directory is not initially empty")

        input_file = _open_relative_regular_file_v3(
            input_directory,
            RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
            maximum_bytes=MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
            minimum_bytes=ARCHIVE_HEADER_BYTES_V2,
            single_link_required=True,
            label="recognizer input archive",
            code=FAIL_INPUT_BINDING,
        )
        stack.callback(input_file.close)
        manifest_file = _open_relative_regular_file_v3(
            input_directory,
            EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
            maximum_bytes=MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
            minimum_bytes=2,
            single_link_required=True,
            label="execution freeze manifest",
            code=FAIL_INPUT_BINDING,
        )
        stack.callback(manifest_file.close)
        seccomp_relative = spec.seccomp_profile_host_path[
            len(spec.repository_host_directory) + 1 :
        ]
        seccomp_file = _open_relative_regular_file_v3(
            repository_directory,
            seccomp_relative,
            maximum_bytes=MAXIMUM_SECCOMP_BYTES_V3,
            minimum_bytes=2,
            single_link_required=True,
            label="recognizer seccomp profile",
            code=FAIL_SECCOMP_BINDING,
        )
        stack.callback(seccomp_file.close)
        if (
            seccomp_file.sha256 != spec.seccomp_profile_sha256
            or seccomp_file.sha256
            != "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
        ):
            _fail(FAIL_SECCOMP_BINDING, "recognizer seccomp digest differs")
        seccomp_value = _decode_json_v3(
            seccomp_file.payload,
            label="recognizer seccomp profile",
            code=FAIL_SECCOMP_BINDING,
        )
        if (
            type(seccomp_value) is not dict
            or seccomp_value.get("defaultAction") != "SCMP_ACT_ALLOW"
            or type(seccomp_value.get("syscalls")) is not list
            or not seccomp_value["syscalls"]
        ):
            _fail(FAIL_SECCOMP_BINDING, "recognizer seccomp semantic shape differs")
        try:
            execution_manifest = decode_execution_freeze_manifest_json_v2(
                manifest_file.payload
            )
            input_archive = decode_public_recognizer_input_archive_v2(input_file.payload)
        except Exception as exc:
            _fail(FAIL_INPUT_BINDING, f"recognizer input decode failed: {type(exc).__name__}")
        if (
            execution_manifest.manifest_id != spec.freeze_manifest_id
            or execution_manifest.protocol_id != spec.protocol_id
            or execution_manifest.recognizer_image_digest != spec.image_digest
            or spec.input_manifest_sha256 != input_file.sha256
            or type(input_archive) is not DecodedRecognizerInputArchiveV2
            or input_archive.disposition is not RecognizerInputArchiveDispositionV2.COMPLETE
            or input_archive.archive != input_file.payload
            or type(input_archive.rows) is not tuple
            or len(input_archive.rows) != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            _fail(FAIL_INPUT_BINDING, "recognizer input/freeze/run-spec binding drift")

        docker_parent, docker_executable = _open_absolute_regular_file_v3(
            DEFAULT_DOCKER_EXECUTABLE,
            maximum_bytes=MAXIMUM_DOCKER_EXECUTABLE_BYTES_V3,
            minimum_bytes=1,
            single_link_required=False,
            label="exact Docker executable",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(docker_parent.close)
        stack.callback(docker_executable.close)
        _recheck_pinned_directory_v3(
            runtime_parent_directory,
            label="private runtime parent before creation",
            code=FAIL_HOST_PATH,
        )
        try:
            runtime_owner = LinuxLocalTemporaryDirectoryV1(
                prefix="hegel-phase2b-recognizer-v3-",
                repository_root=repository_path,
                parent=runtime_parent,
            )
            runtime_root = Path(stack.enter_context(runtime_owner))
            runtime_directory = _open_componentwise_directory_v3(
                runtime_root,
                label="private recognizer runtime directory",
                code=FAIL_DOCKER_CONTROL_PLANE,
            )
            stack.callback(runtime_directory.close)
            _recheck_pinned_directory_v3(
                runtime_parent_directory,
                label="private runtime parent after creation",
                code=FAIL_HOST_PATH,
            )
            if runtime_directory.identities[
                : len(runtime_parent_directory.identities)
            ] != runtime_parent_directory.identities:
                _fail(
                    FAIL_HOST_PATH,
                    "private runtime directory escaped its pinned parent ancestry",
                )
            runtime_directories = {
                **directories,
                "runtime": runtime_directory,
            }
            _require_pairwise_nonoverlap_v3(runtime_directories)
            (
                staged_input_directory,
                staged_input_file,
                staged_manifest_file,
            ) = _stage_and_repin_readonly_input_v3(
                runtime_directory,
                input_payload=input_file.payload,
                manifest_payload=manifest_file.payload,
            )
            stack.callback(staged_input_directory.close)
            stack.callback(staged_input_file.close)
            stack.callback(staged_manifest_file.close)
            staged_seccomp_file = _write_new_regular_file_v3(
                runtime_directory,
                _STAGED_SECCOMP_FILENAME_V3,
                seccomp_file.payload,
                final_mode=0o444,
                maximum_bytes=MAXIMUM_SECCOMP_BYTES_V3,
                minimum_bytes=2,
                label="staged recognizer seccomp profile",
                code=FAIL_SECCOMP_BINDING,
            )
            stack.callback(staged_seccomp_file.close)
            control_plane = prepare_local_docker_control_plane_v1(
                runtime_root, repository_root=repository_path
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_DOCKER_CONTROL_PLANE, f"local Docker boundary rejected: {exc.code}")
        except OSError as exc:
            _fail(FAIL_DOCKER_CONTROL_PLANE, f"private runtime staging failed: {type(exc).__name__}")
        _validate_control_plane_v3(control_plane, runtime_root=runtime_root)
        if control_plane.binding.get("docker_executable_sha256") != docker_executable.sha256:
            _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker executable digest binding drift")
        config_directory = _open_componentwise_directory_v3(
            control_plane.config_directory,
            label="Docker client config directory",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(config_directory.close)
        config_file = _open_relative_regular_file_v3(
            config_directory,
            "config.json",
            maximum_bytes=16,
            minimum_bytes=3,
            single_link_required=True,
            label="empty Docker client config",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(config_file.close)
        if config_file.payload != b"{}\n":
            _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker client config is not exact-empty")

        path_binding_sha256 = stable_hash(
            {
                "original": {
                    name: directory.identities
                    for name, directory in sorted(directories.items())
                },
                "runtime_parent": runtime_parent_directory.identities,
                "runtime": runtime_directory.identities,
                "staged_input": staged_input_directory.identities,
                "input_mount": input_mount,
                "output_mount": output_mount,
            }
        )
        control_plane_binding_sha256 = stable_hash(dict(control_plane.binding))
        image_command = tuple(control_plane.command("image", "inspect", spec.image_digest))
        image_inspect = _run_bounded_command_v3(
            image_command,
            environment=control_plane.environment,
            working_directory=runtime_root,
            timeout_seconds=IMAGE_INSPECT_TIMEOUT_SECONDS_V3,
            maximum_stdout_bytes=MAXIMUM_IMAGE_INSPECT_STDOUT_BYTES_V3,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
            timeout_code=FAIL_DOCKER_TIMEOUT,
            execution_code=FAIL_IMAGE_BINDING,
        )
        if image_inspect.returncode != 0 or image_inspect.stderr != b"":
            detail = image_inspect.stderr.decode("utf-8", "replace")[-256:].strip()
            _fail(FAIL_IMAGE_BINDING, f"local pinned image inspection failed: {detail}")
        observed_image_id, image_config_sha256 = _validate_image_inspect_v3(
            image_inspect.stdout, spec=spec
        )

        recheck_files = (
            (
                input_file,
                MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
                ARCHIVE_HEADER_BYTES_V2,
                "recognizer input archive",
                FAIL_INPUT_BINDING,
            ),
            (
                manifest_file,
                MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
                2,
                "execution freeze manifest",
                FAIL_INPUT_BINDING,
            ),
            (seccomp_file, MAXIMUM_SECCOMP_BYTES_V3, 2, "recognizer seccomp profile", FAIL_SECCOMP_BINDING),
            (staged_input_file, MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2, ARCHIVE_HEADER_BYTES_V2, "staged recognizer input", FAIL_INPUT_BINDING),
            (staged_manifest_file, MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2, 2, "staged execution manifest", FAIL_INPUT_BINDING),
            (staged_seccomp_file, MAXIMUM_SECCOMP_BYTES_V3, 2, "staged seccomp", FAIL_SECCOMP_BINDING),
            (docker_executable, MAXIMUM_DOCKER_EXECUTABLE_BYTES_V3, 1, "exact Docker executable", FAIL_DOCKER_CONTROL_PLANE),
            (config_file, 16, 3, "empty Docker client config", FAIL_DOCKER_CONTROL_PLANE),
        )
        for name, directory in directories.items():
            _recheck_pinned_directory_v3(directory, label=name)
        _recheck_pinned_directory_v3(
            runtime_parent_directory,
            label="private runtime parent",
            code=FAIL_HOST_PATH,
        )
        _recheck_pinned_directory_v3(runtime_directory, label="private runtime")
        _recheck_pinned_directory_v3(staged_input_directory, label="staged input")
        _recheck_pinned_directory_v3(config_directory, label="Docker config")
        _require_pairwise_nonoverlap_v3(runtime_directories)
        for pinned, maximum, minimum, label, code in recheck_files:
            _recheck_regular_file_v3(
                pinned,
                maximum_bytes=maximum,
                minimum_bytes=minimum,
                label=label,
                code=code,
            )
        if _directory_names_v3(
            output_directory,
            label="host output before container creation",
            code=FAIL_OUTPUT_BINDING,
        ):
            _fail(FAIL_OUTPUT_BINDING, "host output changed before container creation")

        container_name = _container_name_v3(runtime_root)
        create_command = _docker_create_command_v3(
            control_plane,
            spec=spec,
            container_name=container_name,
            staged_input_directory=staged_input_directory.path,
            staged_seccomp_path=runtime_root / _STAGED_SECCOMP_FILENAME_V3,
        )
        named_container_removed = False

        def remove_named_container() -> None:
            nonlocal named_container_removed
            if not named_container_removed:
                _force_remove_container_v3(
                    control_plane=control_plane,
                    environment=control_plane.environment,
                    working_directory=runtime_root,
                    container_id=container_name,
                    allow_already_absent=True,
                )
                named_container_removed = True

        created = _run_bounded_command_v3(
            create_command,
            environment=control_plane.environment,
            working_directory=runtime_root,
            timeout_seconds=60,
            maximum_stdout_bytes=256,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
            timeout_code=FAIL_DOCKER_TIMEOUT,
            execution_code=FAIL_DOCKER_EXECUTION,
            forced_cleanup=remove_named_container,
        )
        if created.returncode != 0 or created.stderr != b"":
            remove_named_container()
            detail = created.stderr.decode("utf-8", "replace")[-256:].strip()
            _fail(FAIL_DOCKER_EXECUTION, f"recognizer container creation failed: {detail}")
        try:
            container_id = _parse_created_container_id_v3(created.stdout)
        except BaseException:
            remove_named_container()
            raise

        removed = False

        def remove_container() -> None:
            nonlocal removed
            if not removed:
                _force_remove_container_v3(
                    control_plane=control_plane,
                    environment=control_plane.environment,
                    working_directory=runtime_root,
                    container_id=container_id,
                )
                removed = True

        try:
            container_inspect_command = tuple(
                control_plane.command("container", "inspect", container_id)
            )
            container_inspect = _run_bounded_command_v3(
                container_inspect_command,
                environment=control_plane.environment,
                working_directory=runtime_root,
                timeout_seconds=30,
                maximum_stdout_bytes=MAXIMUM_IMAGE_INSPECT_STDOUT_BYTES_V3,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
                timeout_code=FAIL_DOCKER_TIMEOUT,
                execution_code=FAIL_DOCKER_EXECUTION,
                forced_cleanup=remove_container,
            )
            if container_inspect.returncode != 0 or container_inspect.stderr != b"":
                _fail(FAIL_DOCKER_EXECUTION, "created recognizer container inspect failed")
            _validate_created_container_inspect_v3(
                container_inspect.stdout,
                container_id=container_id,
                container_name=container_name,
                spec=spec,
                staged_input_directory=staged_input_directory.path,
                expected_seccomp=seccomp_value,
            )
            _recheck_pinned_directory_v3(
                runtime_parent_directory,
                label="private runtime parent after container create",
                code=FAIL_HOST_PATH,
            )
            _recheck_pinned_directory_v3(
                runtime_directory,
                label="private runtime after container create",
                code=FAIL_DOCKER_CONTROL_PLANE,
            )
            _require_pairwise_nonoverlap_v3(runtime_directories)
            for pinned, maximum, minimum, label, code in recheck_files:
                _recheck_regular_file_v3(
                    pinned,
                    maximum_bytes=maximum,
                    minimum_bytes=minimum,
                    label=f"{label} after container create",
                    code=code,
                )
            _recheck_pinned_directory_v3(
                staged_input_directory,
                label="staged input after container create",
                code=FAIL_INPUT_BINDING,
            )
            start_command = tuple(control_plane.command("start", container_id))
            extractor_command = tuple(
                control_plane.command(
                    "container",
                    "exec",
                    f"--user={RECOGNIZER_RUNTIME_UID_GID}",
                    container_id,
                    _CONTAINER_SUPERVISOR_ENTRYPOINT_V3,
                    "-c",
                    _CONTAINER_RESULT_EXTRACTOR_CODE_V3,
                )
            )
            recognizer_started_monotonic_ns = time.monotonic_ns()
            recognizer_deadline_monotonic_ns = (
                recognizer_started_monotonic_ns
                + RECOGNIZER_TIMEOUT_SECONDS_V3 * 1_000_000_000
            )
            container_started = _run_bounded_command_v3(
                start_command,
                environment=control_plane.environment,
                working_directory=runtime_root,
                timeout_seconds=60,
                maximum_stdout_bytes=256,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
                timeout_code=FAIL_DOCKER_TIMEOUT,
                execution_code=FAIL_DOCKER_EXECUTION,
                forced_cleanup=remove_container,
            )
            if (
                container_started.returncode != 0
                or container_started.stderr != b""
                or container_started.stdout != (container_id + "\n").encode("ascii")
            ):
                detail = container_started.stderr.decode("utf-8", "replace")[-256:].strip()
                _fail(
                    FAIL_DOCKER_EXECUTION,
                    "recognizer container start failed "
                    f"({container_started.returncode}): {detail}",
                )
            remaining_nanoseconds = (
                recognizer_deadline_monotonic_ns - time.monotonic_ns()
            )
            if remaining_nanoseconds <= 0:
                _fail(
                    FAIL_DOCKER_TIMEOUT,
                    "recognizer deadline elapsed before result extraction",
                )
            remaining_seconds = max(
                1,
                (remaining_nanoseconds + 999_999_999) // 1_000_000_000,
            )
            runtime_observation = _run_bounded_command_v3(
                extractor_command,
                environment=control_plane.environment,
                working_directory=runtime_root,
                timeout_seconds=remaining_seconds,
                maximum_stdout_bytes=MAXIMUM_DOCKER_RESULT_WIRE_BYTES_V3,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
                timeout_code=FAIL_DOCKER_TIMEOUT,
                execution_code=FAIL_OUTPUT_BINDING,
                forced_cleanup=remove_container,
            )
            if runtime_observation.returncode != 0 or runtime_observation.stderr != b"":
                detail = runtime_observation.stderr.decode("utf-8", "replace")[-256:].strip()
                _fail(
                    FAIL_OUTPUT_BINDING,
                    "bounded recognizer result extraction failed "
                    f"({runtime_observation.returncode}): {detail}",
                )
            if (
                runtime_observation.finished_monotonic_ns
                > recognizer_deadline_monotonic_ns
            ):
                _fail(
                    FAIL_DOCKER_TIMEOUT,
                    "recognizer result arrived after the one absolute deadline",
                )
            runtime_receipt_payload, prediction_payload = (
                _extract_container_result_wire_v3(runtime_observation.stdout)
            )
        finally:
            remove_container()

        for name, directory in directories.items():
            _recheck_pinned_directory_v3(directory, label=f"{name} after container")
        _recheck_pinned_directory_v3(
            runtime_parent_directory,
            label="private runtime parent after container",
            code=FAIL_HOST_PATH,
        )
        _recheck_pinned_directory_v3(runtime_directory, label="private runtime after container")
        _require_pairwise_nonoverlap_v3(runtime_directories)
        _recheck_pinned_directory_v3(staged_input_directory, label="staged input after container")
        _recheck_pinned_directory_v3(config_directory, label="Docker config after container")
        for pinned, maximum, minimum, label, code in recheck_files:
            _recheck_regular_file_v3(
                pinned,
                maximum_bytes=maximum,
                minimum_bytes=minimum,
                label=f"{label} after container",
                code=code,
            )
        if _directory_names_v3(
            output_directory,
            label="host output after container",
            code=FAIL_OUTPUT_BINDING,
        ):
            _fail(FAIL_OUTPUT_BINDING, "host output changed during container execution")
        try:
            prediction_archive = decode_public_recognizer_prediction_archive_v2(
                prediction_payload
            )
        except Exception as exc:
            _fail(FAIL_OUTPUT_BINDING, f"prediction archive decode failed: {type(exc).__name__}")
        if (
            type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2
            or prediction_archive.disposition is not PredictionArchiveDispositionV2.COMPLETE
            or prediction_archive.archive != prediction_payload
            or type(prediction_archive.records) is not tuple
            or len(prediction_archive.records) != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            _fail(FAIL_OUTPUT_BINDING, "prediction archive replay/count drift")
        _crossbind_prediction_to_input_v3(
            prediction_archive,
            input_archive=input_archive,
            input_archive_sha256=input_file.sha256,
            execution_manifest_id=execution_manifest.manifest_id,
            protocol_id=execution_manifest.protocol_id,
        )
        prediction_sha256 = _sha256(prediction_payload)
        runtime_receipt_id = _validate_runtime_receipt_v3(
            runtime_receipt_payload,
            input_archive=input_archive,
            input_archive_sha256=input_file.sha256,
            execution_manifest_id=execution_manifest.manifest_id,
            execution_manifest_sha256=manifest_file.sha256,
            prediction_archive=prediction_archive,
            prediction_archive_sha256=prediction_sha256,
            prediction_archive_bytes=len(prediction_payload),
        )
        prediction_file = _publish_prediction_output_v3(
            output_directory, prediction_payload
        )
        stack.callback(prediction_file.close)
        if prediction_file.payload != prediction_payload or prediction_file.sha256 != prediction_sha256:
            _fail(FAIL_OUTPUT_BINDING, "host-published prediction bytes drift")
        _recheck_pinned_directory_v3(
            output_directory, label="host output after publication", code=FAIL_OUTPUT_BINDING
        )
        _recheck_regular_file_v3(
            prediction_file,
            maximum_bytes=MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
            minimum_bytes=PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            label="host-published prediction archive",
            code=FAIL_OUTPUT_BINDING,
        )

        return _issue_receipt_after_publication_v3(
            output_directory,
            prediction_file,
            schema_version=LOCAL_EXECUTION_RECEIPT_SCHEMA_V3,
            disposition=LOCAL_EXECUTION_DISPOSITION_V3,
            reason=LOCAL_EXECUTION_COMPLETE_REASON_V3,
            run_spec_id=spec.run_spec_id,
            freeze_manifest_id=spec.freeze_manifest_id,
            protocol_id=spec.protocol_id,
            image_digest=spec.image_digest,
            observed_image_id=observed_image_id,
            recognizer_source_root_sha256=RECOGNIZER_SOURCE_ROOT_SHA256_V3,
            actor_profile_content_id=(
                PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
            ),
            actor_profile_canonical_sha256=(
                PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
            ),
            owner_accepted_authority_requirement_id=(
                OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
            ),
            image_args_escaped=True,
            image_inspect_sha256=_sha256(image_inspect.stdout),
            image_config_projection_sha256=image_config_sha256,
            docker_control_plane_binding_sha256=control_plane_binding_sha256,
            docker_executable_sha256=docker_executable.sha256,
            seccomp_profile_sha256=seccomp_file.sha256,
            host_path_binding_sha256=path_binding_sha256,
            input_manifest_sha256=spec.input_manifest_sha256,
            input_archive_id=input_archive.archive_id,
            input_archive_sha256=input_file.sha256,
            execution_freeze_manifest_sha256=manifest_file.sha256,
            command_sha256=stable_hash(
                {
                    "image_inspect": image_command,
                    "create": create_command,
                    "created_container_inspect": container_inspect_command,
                    "start": start_command,
                    "extract_result": extractor_command,
                    "remove": tuple(
                        control_plane.command(
                            "container", "rm", "--force", container_id
                        )
                    ),
                    "verify_removed": tuple(
                        control_plane.command(
                            "container", "inspect", container_id
                        )
                    ),
                }
            ),
            container_supervisor_sha256=_sha256(
                (
                    _CONTAINER_SUPERVISOR_ENTRYPOINT_V3
                    + "\x00"
                    + _CONTAINER_SUPERVISOR_CODE_V3
                ).encode("utf-8")
            ),
            container_result_extractor_sha256=_sha256(
                (
                    _CONTAINER_SUPERVISOR_ENTRYPOINT_V3
                    + "\x00"
                    + _CONTAINER_RESULT_EXTRACTOR_CODE_V3
                ).encode("utf-8")
            ),
            process_exit_code=0,
            timed_out=False,
            started_monotonic_ns=recognizer_started_monotonic_ns,
            finished_monotonic_ns=runtime_observation.finished_monotonic_ns,
            stdout_bytes=len(runtime_receipt_payload),
            stdout_sha256=_sha256(runtime_receipt_payload),
            stderr_bytes=0,
            stderr_sha256=_sha256(b""),
            runtime_receipt_id=runtime_receipt_id,
            prediction_archive_id=prediction_archive.archive_id,
            prediction_archive_sha256=prediction_sha256,
            prediction_archive_bytes=len(prediction_payload),
            prediction_case_count=len(prediction_archive.records),
            output_file_identity_sha256=stable_hash(
                prediction_file.metadata_fingerprint
            ),
        )


__all__ = (
    "FAIL_DOCKER_CAPTURE",
    "FAIL_DOCKER_CONTROL_PLANE",
    "FAIL_DOCKER_EXECUTION",
    "FAIL_DOCKER_TIMEOUT",
    "FAIL_HOST_PATH",
    "FAIL_IMAGE_BINDING",
    "FAIL_INPUT_BINDING",
    "FAIL_OUTPUT_BINDING",
    "FAIL_PREFLIGHT",
    "FAIL_SECCOMP_BINDING",
    "IMAGE_INSPECT_TIMEOUT_SECONDS_V3",
    "LOCAL_EXECUTION_COMPLETE_REASON_V3",
    "LOCAL_EXECUTION_DISPOSITION_V3",
    "LOCAL_EXECUTION_RECEIPT_SCHEMA_V3",
    "MAXIMUM_DOCKER_STDERR_BYTES_V3",
    "MAXIMUM_DOCKER_STDOUT_BYTES_V3",
    "OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3",
    "Phase2BRecognizerHostExecutorErrorV3",
    "Phase2BRecognizerLocalExecutionReceiptV3",
    "RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3",
    "RECOGNIZER_SOURCE_ROOT_SHA256_V3",
    "RECOGNIZER_TIMEOUT_SECONDS_V3",
    "execute_phase2b_recognizer_docker_v3",
)
