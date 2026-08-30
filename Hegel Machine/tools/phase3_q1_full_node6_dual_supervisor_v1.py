#!/usr/bin/env python3
"""One-process supervisor for the full-node6 dual capacity diagnostic.

The outer campaign starts this file once after authorization consumption.  It
starts the Python and Rust endpoint containers concurrently, waits for both to
exit, then starts a separately isolated trusted-host *wire replay* container.
The outer process never imports the replay implementation or either quotient
engine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Callable, Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]


SUPERVISOR_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-dual-supervisor/1"
)
RESOURCE_EVIDENCE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-container-resource-evidence/1"
)
TRAJECTORY_EVENT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-supervisor-trajectory-event/1"
)
TRAJECTORY_TERMINAL_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-supervisor-trajectory-terminal/1"
)
ACTION_ID: Final = "full-node6-capacity-preflight-v1"
SUCCESS_STATUS: Final = "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
GUARD_STATUS: Final = "PREFLIGHT_CAPACITY_GUARD_HIT"
DOCKER: Final = "/usr/bin/docker"
PYTHON_IMAGE: Final = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE: Final = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
PYTHON_ENDPOINT_RELATIVE_PATH: Final = (
    "tools/phase3_q1_full_node6_capacity_entrypoint_v1.py"
)
PYTHON_ENDPOINT_MODULE_NAMES: Final = (
    "phase3_m3_bounded_enumerator_shrink2_v1",
    "phase3_m3_bounded_enumerator_shrink3_v1",
    "phase3_m3_bounded_enumerator_shrink4_v1",
    "phase3_m3_bounded_enumerator_shrink5_v1",
    "phase3_m3_bounded_enumerator_shrink6_v1",
    "phase3_m3_bounded_enumerator_v1",
    "phase3_m3_dsl_core_v1",
    "phase3_m3_record_wire_v1",
    "phase3_m3_shrink1_core_v1",
    "phase3_m3_shrink2_core_v1",
    "phase3_m3_shrink3_core_v1",
    "phase3_m3_shrink4_core_v1",
    "phase3_m3_shrink5_core_v1",
    "phase3_m3_shrink6_core_v1",
    "phase3_q0_evaluator_v1",
    "phase3_q0_input_adapter_v1",
    "phase3_q1_archive_projection_v1",
    "phase3_q1_capacity_preflight_v1",
    "phase3_q1_external_sort_profile_v1",
    "phase3_q1_formal_archive_contract_v1",
    "phase3_q1_full_node6_capacity_preflight_v1",
    "phase3_q1_full_node6_projection_v1",
    "phase3_q1_partition_snapshot_v1",
    "phase3_q1_quotient_contract_v1",
    "phase3_q1_universe_v1",
    "strict_ast_shrink1_v1",
    "strict_ast_shrink2_v1",
    "strict_ast_shrink3_v1",
    "strict_ast_shrink4_v1",
    "strict_ast_shrink5_v1",
    "strict_ast_shrink6_v1",
    "strict_ast_v1",
    "strict_cbor_v1",
)
PYTHON_ENDPOINT_SOURCE_PATHS: Final = (
    PYTHON_ENDPOINT_RELATIVE_PATH,
    *(f"src/hegel_machine/{name}.py" for name in PYTHON_ENDPOINT_MODULE_NAMES),
)
RUST_BINARY_CONTAINER_PATH: Final = "/actor/hegel-q1-full-capacity-preflight-engine"
HOST_REPLAY_RELATIVE_PATH: Final = (
    "src/hegel_machine/phase3_q1_full_node6_host_replay_v1.py"
)
RUST_MANIFEST_RELATIVE_PATH: Final = (
    "rust/q1_full_capacity_preflight_engine/Cargo.toml"
)
RUST_ENDPOINT_SOURCE_PATHS: Final = (
    "rust/q1_full_capacity_preflight_engine/Cargo.lock",
    RUST_MANIFEST_RELATIVE_PATH,
    "rust/q1_full_capacity_preflight_engine/src/frozen_v1.rs",
    "rust/q1_full_capacity_preflight_engine/src/lib.rs",
    "rust/q1_full_capacity_preflight_engine/src/main.rs",
    "rust/q1_archive_projection_oracle/Cargo.toml",
    "rust/q1_archive_projection_oracle/src/lib.rs",
    "rust/strict_canonicalizer/Cargo.toml",
    "rust/strict_canonicalizer/src/lib.rs",
    "rust/strict_canonicalizer_shrink1/Cargo.toml",
    "rust/strict_canonicalizer_shrink1/src/lib.rs",
    "rust/strict_canonicalizer_shrink2/Cargo.toml",
    "rust/strict_canonicalizer_shrink2/src/lib.rs",
    "rust/strict_canonicalizer_shrink3/Cargo.toml",
    "rust/strict_canonicalizer_shrink3/src/lib.rs",
    "rust/strict_canonicalizer_shrink4/Cargo.toml",
    "rust/strict_canonicalizer_shrink4/src/lib.rs",
    "rust/strict_canonicalizer_shrink5/Cargo.toml",
    "rust/strict_canonicalizer_shrink5/src/lib.rs",
    "rust/strict_canonicalizer_shrink6/Cargo.toml",
    "rust/strict_canonicalizer_shrink6/src/lib.rs",
)
HOST_REPLAY_SOURCE_PATHS: Final = (HOST_REPLAY_RELATIVE_PATH,)
EXECUTION_CLOSURE_ROW_COUNT: Final = 55
CARGO_CACHE_SOURCE: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache"
)
TECHNICAL_CONTRACT_RELATIVE_PATH: Final = (
    "config/phase3_q1_full_node6_capacity_preflight_v1.json"
)
MEMORY_BYTES: Final = 15_032_385_536
PROJECTED_HOST_REPLAY_OUTPUT_BYTES: Final = 1_048_576
PROJECTED_PEAK_HOST_REPLAY_SCRATCH_BYTES: Final = 0
CONTAINER_USER: Final = "65534:65534"
PIDS_LIMIT: Final = 128
NOFILE_ULIMIT: Final = "256:256"
TMPFS_SPEC: Final = "/tmp:rw,noexec,nosuid,nodev,size=1g,mode=1777"
SECCOMP_RELATIVE_PATH: Final = "config/phase3_internal_actor_seccomp_v1.json"
SECCOMP_SHA256: Final = (
    "sha256:d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
)
RUST_BUILD_TIMEOUT_SECONDS: Final = 1800
PER_SIGNATURE_WALL_TIME_SECONDS: Final = 172800
ORDERED_SIGNATURE_COUNT: Final = 2
BOUNDED_PROJECTION_WALL_TIME_SECONDS: Final = 172800
ENDPOINT_PAIR_WALL_TIME_SECONDS: Final = (
    PER_SIGNATURE_WALL_TIME_SECONDS * ORDERED_SIGNATURE_COUNT
    + BOUNDED_PROJECTION_WALL_TIME_SECONDS
)
HOST_REPLAY_WALL_TIME_SECONDS: Final = 172800
OTHER_PREPARATION_AND_PUBLICATION_SECONDS: Final = 600
ENDPOINT_OUTPUT_CEILING_BYTES: Final = 34_359_738_368
ENDPOINT_SCRATCH_CEILING_BYTES: Final = 51_539_607_552
ENDPOINT_MEMORY_CEILING_BYTES: Final = MEMORY_BYTES
ENDPOINT_WALL_CEILING_SECONDS: Final = 172_800
HOST_REPLAY_OUTPUT_CEILING_BYTES: Final = 34_359_738_368
HOST_REPLAY_SCRATCH_CEILING_BYTES: Final = 51_539_607_552
HOST_REPLAY_MEMORY_CEILING_BYTES: Final = MEMORY_BYTES
HOST_REPLAY_WALL_CEILING_SECONDS: Final = 172_800
INFEASIBLE_STATUS: Final = "PREFLIGHT_RESOURCE_ENVELOPE_INFEASIBLE"
OUTER_SUPERVISOR_WALL_TIME_SECONDS: Final = (
    RUST_BUILD_TIMEOUT_SECONDS
    + ENDPOINT_PAIR_WALL_TIME_SECONDS
    + HOST_REPLAY_WALL_TIME_SECONDS
    + OTHER_PREPARATION_AND_PUBLICATION_SECONDS
)
ACTOR_READY_FILENAME: Final = "actor-ready-v1"
ACTOR_RETURN_CODE_FILENAME: Final = "actor-return-code-v1"
SUPERVISOR_RELEASE_FILENAME: Final = "supervisor-release-v1"
_ACTOR_HOLD_SCRIPT: Final = (
    'status=0\n'
    '"$@" || status=$?\n'
    'umask 022\n'
    'printf "%s\\n" "$status" > /scratch/' + ACTOR_RETURN_CODE_FILENAME + '\n'
    ': > /scratch/' + ACTOR_READY_FILENAME + '\n'
    'while [ ! -e /scratch/' + SUPERVISOR_RELEASE_FILENAME + ' ]; do sleep 1; done\n'
    'exit "$status"\n'
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class FullNode6SupervisorError(RuntimeError):
    """Stable terminal supervisor failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FullNode6SupervisorError(code, detail)


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_WIRE", f"canonical JSON: {error}")


def _canonical_object(payload: bytes, label: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 16 * 1024 * 1024:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_WIRE", f"{label} size/type differs")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_WIRE", f"{label}: {error}")
    if type(value) is not dict or _canonical(value) != payload:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_WIRE", f"{label} is not canonical")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_EVIDENCE", f"directory fsync: {error}")


@dataclass(frozen=True, slots=True)
class SupervisorRunSpecV1:
    capsule_digest: str
    source_commit: str
    python_snapshot: Path
    rust_snapshot: Path
    host_replay_snapshot: Path
    rust_binary: Path
    admission_json: Path
    seccomp_profile: Path
    run_root: Path
    trajectory_root: Path

    def __post_init__(self) -> None:
        if (
            type(self.capsule_digest) is not str
            or _DIGEST.fullmatch(self.capsule_digest) is None
            or type(self.source_commit) is not str
            or _COMMIT.fullmatch(self.source_commit) is None
        ):
            _fail("REJECT_FULL_NODE6_SUPERVISOR_SPEC", "execution binding differs")
        for name in (
            "python_snapshot",
            "rust_snapshot",
            "host_replay_snapshot",
            "rust_binary",
            "admission_json",
            "seccomp_profile",
            "run_root",
            "trajectory_root",
        ):
            if not isinstance(getattr(self, name), Path):
                _fail("REJECT_FULL_NODE6_SUPERVISOR_SPEC", f"{name} must be Path")


@dataclass(frozen=True, slots=True)
class EndpointProcessResultV1:
    actor_id: str
    return_code: int
    stdout: bytes
    stderr: bytes
    resource_evidence: dict[str, object]


HostReplay = Callable[
    [Sequence[str], Path],
    EndpointProcessResultV1,
]


@dataclass(slots=True)
class SupervisorTrajectoryV1:
    """Append-only phase trail owned by one supervisor run root."""

    run_root: Path
    trajectory_root: Path
    capsule_digest: str
    source_commit: str
    started_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    event_count: int = 0
    closed: bool = False

    @property
    def event_path(self) -> Path:
        return self.trajectory_root / "supervisor-trajectory-v1.jsonl"

    @property
    def terminal_path(self) -> Path:
        return self.trajectory_root / "supervisor-trajectory-terminal-v1.json"

    @classmethod
    def create(
        cls,
        run_root: Path,
        trajectory_root: Path,
        capsule_digest: str,
        source_commit: str,
    ) -> "SupervisorTrajectoryV1":
        if trajectory_root == run_root or trajectory_root.exists():
            _fail(
                "REJECT_FULL_NODE6_SUPERVISOR_SPEC",
                "trajectory root must be absent and distinct from run root",
            )
        try:
            trajectory_root.mkdir(mode=0o700)
            _fsync_directory(trajectory_root.parent)
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_TRAJECTORY", f"root: {error}")
        trajectory = cls(run_root, trajectory_root, capsule_digest, source_commit)
        try:
            descriptor = os.open(
                trajectory.event_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
            _fsync_directory(trajectory_root)
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_TRAJECTORY", f"create: {error}")
        trajectory.emit("SUPERVISOR", "STARTED")
        return trajectory

    def emit(
        self,
        phase: str,
        status: str,
        *,
        detail: str | None = None,
        command_artifact: str | None = None,
        return_code: int | None = None,
        decision: str | None = None,
    ) -> None:
        if self.closed:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_TRAJECTORY", "trajectory is closed")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", phase) or status not in (
            "STARTED",
            "COMPLETED",
            "FAILED",
        ):
            _fail("FAIL_FULL_NODE6_SUPERVISOR_TRAJECTORY", "event registry differs")
        self.event_count += 1
        event = {
            "capsule_digest": self.capsule_digest,
            "command_artifact_path_or_null": (
                os.fspath(self.run_root / command_artifact)
                if command_artifact is not None
                else None
            ),
            "decision_or_null": decision,
            "detail_or_null": detail,
            "elapsed_monotonic_ns": time.monotonic_ns()
            - self.started_monotonic_ns,
            "event_ordinal": self.event_count,
            "phase": phase,
            "return_code_or_null": return_code,
            "schema_version": TRAJECTORY_EVENT_SCHEMA_VERSION,
            "source_commit": self.source_commit,
            "status": status,
        }
        try:
            with self.event_path.open("ab", buffering=0) as output:
                output.write(_canonical(event))
                os.fsync(output.fileno())
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_TRAJECTORY", f"append: {error}")

    def finish(
        self,
        terminal_status: str,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> dict[str, object]:
        status = "COMPLETED" if error_code is None else "FAILED"
        self.emit(
            "TERMINAL",
            status,
            detail=error_detail,
            decision=terminal_status,
        )
        summary = {
            "capsule_digest": self.capsule_digest,
            "error_code_or_null": error_code,
            "error_detail_or_null": error_detail,
            "event_count": self.event_count,
            "event_log_path": os.fspath(self.event_path),
            "run_root": os.fspath(self.run_root),
            "schema_version": TRAJECTORY_TERMINAL_SCHEMA_VERSION,
            "source_commit": self.source_commit,
            "terminal_status": terminal_status,
        }
        _publish_run_file(self.terminal_path, _canonical(summary))
        try:
            self.event_path.chmod(0o400)
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_TRAJECTORY", f"close: {error}")
        self.closed = True
        return summary


def _require_readonly_snapshot(path: Path, label: str) -> None:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label}: {error}")
    if (
        path.is_symlink()
        or not stat.S_ISDIR(status.st_mode)
        or resolved != path
        or status.st_mode & 0o222
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label} snapshot differs")
    # The supervisor never asks the host filesystem to make a snapshot
    # writable; the Docker bind itself is also explicitly read-only.


def _require_snapshot_path_registry(
    root: Path,
    expected_paths: Sequence[str],
    label: str,
) -> None:
    observed: set[str] = set()
    try:
        for path in root.rglob("*"):
            status = path.lstat()
            if path.is_symlink():
                _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label} symlink differs")
            if stat.S_ISDIR(status.st_mode):
                if status.st_mode & 0o222 or status.st_mode & 0o005 != 0o005:
                    _fail(
                        "FAIL_FULL_NODE6_SUPERVISOR_SOURCE",
                        f"{label} directory mode differs",
                    )
                continue
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_mode & 0o222
                or status.st_mode & 0o004 == 0
            ):
                _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label} file differs")
            observed.add(path.relative_to(root).as_posix())
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label}: {error}")
    if observed != set(expected_paths):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label} file registry differs")


def _require_distinct_snapshots(spec: SupervisorRunSpecV1) -> None:
    _require_readonly_snapshot(spec.python_snapshot, "Python")
    _require_readonly_snapshot(spec.rust_snapshot, "Rust")
    _require_readonly_snapshot(spec.host_replay_snapshot, "host replay")
    _require_snapshot_path_registry(
        spec.python_snapshot,
        PYTHON_ENDPOINT_SOURCE_PATHS,
        "Python",
    )
    _require_snapshot_path_registry(
        spec.rust_snapshot,
        RUST_ENDPOINT_SOURCE_PATHS,
        "Rust",
    )
    _require_snapshot_path_registry(
        spec.host_replay_snapshot,
        HOST_REPLAY_SOURCE_PATHS,
        "host replay",
    )
    if len(
        {
            spec.python_snapshot.resolve(),
            spec.rust_snapshot.resolve(),
            spec.host_replay_snapshot.resolve(),
        }
    ) != 3:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", "source snapshots are not distinct")
    try:
        binary_status = spec.rust_binary.lstat()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"Rust binary unavailable: {error}")
    if spec.rust_binary.is_symlink() or not stat.S_ISREG(binary_status.st_mode):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "Rust binary differs")
    if binary_status.st_mode & 0o005 != 0o005:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "Rust binary is not actor executable")
    try:
        admission_status = spec.admission_json.lstat()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"admission JSON unavailable: {error}")
    if spec.admission_json.is_symlink() or not stat.S_ISREG(admission_status.st_mode):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "admission JSON differs")
    if admission_status.st_mode & 0o004 == 0:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "admission JSON is not actor-readable")
    try:
        seccomp_status = spec.seccomp_profile.lstat()
        seccomp_payload = spec.seccomp_profile.read_bytes()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"seccomp profile: {error}")
    if (
        spec.seccomp_profile.is_symlink()
        or not stat.S_ISREG(seccomp_status.st_mode)
        or "sha256:" + sha256(seccomp_payload).hexdigest() != SECCOMP_SHA256
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "seccomp profile differs")


def _common_docker_args(
    *,
    name: str,
    cidfile: Path,
    cpuset: str,
    image: str,
    snapshot: Path,
    output: Path,
    scratch: Path,
    seccomp_profile: Path,
    output_destination: str = "/output",
    output_readonly: bool = False,
    additional_mounts: Sequence[tuple[Path, str, bool]] = (),
) -> list[str]:
    output_mount = f"type=bind,src={output},dst={output_destination}"
    if output_readonly:
        output_mount += ",readonly"
    command = [
        DOCKER,
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        name,
        "--user",
        CONTAINER_USER,
        "--cidfile",
        str(cidfile),
        "--network",
        "none",
        "--ipc",
        "private",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--security-opt",
        f"seccomp={seccomp_profile}",
        "--cpuset-cpus",
        cpuset,
        "--memory",
        str(MEMORY_BYTES),
        "--memory-swap",
        str(MEMORY_BYTES),
        "--pids-limit",
        str(PIDS_LIMIT),
        "--ulimit",
        f"nofile={NOFILE_ULIMIT}",
        "--tmpfs",
        TMPFS_SPEC,
        "--env",
        "HOME=/tmp",
        "--env",
        "LANG=C.UTF-8",
        "--env",
        "LC_ALL=C.UTF-8",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "TZ=UTC",
        "--mount",
        f"type=bind,src={snapshot},dst=/snapshot,readonly",
        "--mount",
        output_mount,
        "--mount",
        f"type=bind,src={scratch},dst=/scratch",
    ]
    for source, destination, readonly in additional_mounts:
        mount = f"type=bind,src={source},dst={destination}"
        if readonly:
            mount += ",readonly"
        command.extend(("--mount", mount))
    command.extend(("--workdir", "/snapshot", image))
    return command


def _append_actor_hold_v1(command: list[str], actor_argv: Sequence[str]) -> None:
    command.extend(
        (
            "/bin/sh",
            "-c",
            _ACTOR_HOLD_SCRIPT,
            "hegel-full-node6-actor-hold-v1",
            *actor_argv,
        )
    )


def endpoint_commands_v1(
    spec: SupervisorRunSpecV1,
    *,
    python_output: Path,
    rust_output: Path,
    python_scratch: Path,
    rust_scratch: Path,
    python_cidfile: Path,
    rust_cidfile: Path,
) -> tuple[list[str], list[str]]:
    """Return the two fixed offline container commands."""

    suffix = spec.capsule_digest[-12:]
    python = _common_docker_args(
        name=f"hegel-q1-node6-python-{suffix}",
        cidfile=python_cidfile,
        cpuset="0-11",
        image=PYTHON_IMAGE,
        snapshot=spec.python_snapshot,
        output=python_output,
        scratch=python_scratch,
        seccomp_profile=spec.seccomp_profile,
    )
    python_image_index = python.index(PYTHON_IMAGE)
    python[python_image_index:python_image_index] = [
        "--mount",
        f"type=bind,src={spec.admission_json},dst=/admission/admission.json,readonly",
    ]
    python_actor = [
        "/usr/local/bin/python3",
        "-I",
        "-S",
        "-B",
        f"/snapshot/{PYTHON_ENDPOINT_RELATIVE_PATH}",
        "--action",
        ACTION_ID,
        "--admission-json",
        "/admission/admission.json",
        "--capsule-digest",
        spec.capsule_digest,
        "--source-commit",
        spec.source_commit,
        "--output-dir",
        "/output",
        "--scratch-dir",
        "/scratch",
    ]
    _append_actor_hold_v1(python, python_actor)
    rust = _common_docker_args(
        name=f"hegel-q1-node6-rust-{suffix}",
        cidfile=rust_cidfile,
        cpuset="12-23",
        image=RUST_IMAGE,
        snapshot=spec.rust_snapshot,
        output=rust_output,
        scratch=rust_scratch,
        seccomp_profile=spec.seccomp_profile,
    )
    rust_image_index = rust.index(RUST_IMAGE)
    rust[rust_image_index:rust_image_index] = [
        "--mount",
        f"type=bind,src={spec.rust_binary},dst={RUST_BINARY_CONTAINER_PATH},readonly",
    ]
    rust_actor = [
        RUST_BINARY_CONTAINER_PATH,
        "--action",
        ACTION_ID,
        "--capsule-digest",
        spec.capsule_digest,
        "--source-commit",
        spec.source_commit,
        "--output-dir",
        "/output",
        "--scratch-dir",
        "/scratch",
    ]
    _append_actor_hold_v1(rust, rust_actor)
    return python, rust


def host_replay_command_v1(
    spec: SupervisorRunSpecV1,
    *,
    python_output: Path,
    rust_output: Path,
    host_scratch: Path,
    host_cidfile: Path,
) -> list[str]:
    """Return the pinned isolated target-blind host replay command."""

    command = _common_docker_args(
        name=f"hegel-q1-node6-host-replay-{spec.capsule_digest[-12:]}",
        cidfile=host_cidfile,
        cpuset="0-11",
        image=PYTHON_IMAGE,
        snapshot=spec.host_replay_snapshot,
        output=python_output,
        scratch=host_scratch,
        seccomp_profile=spec.seccomp_profile,
        output_destination="/python-output",
        output_readonly=True,
        additional_mounts=((rust_output, "/rust-output", True),),
    )
    actor_argv = [
        "/usr/local/bin/python3",
        "-I",
        "-S",
        "-B",
        f"/snapshot/{HOST_REPLAY_RELATIVE_PATH}",
        "--run",
        "--python-output",
        "/python-output",
        "--rust-output",
        "/rust-output",
        "--capsule-digest",
        spec.capsule_digest,
        "--source-commit",
        spec.source_commit,
        "--scratch-dir",
        "/scratch",
    ]
    _append_actor_hold_v1(command, actor_argv)
    return command


def _run_capture(
    command: Sequence[str],
    label: str,
    *,
    timeout_seconds: int = 30,
    evidence_root: Path | None = None,
) -> bytes:
    if evidence_root is not None:
        _publish_run_file(
            evidence_root / "rust-build-command-v1.json",
            _canonical(
                {
                    "command": list(command),
                    "network_allowed": False,
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-rust-build-command/1"
                    ),
                    "timeout_seconds": timeout_seconds,
                }
            ),
        )
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        if evidence_root is not None:
            _publish_run_file(
                evidence_root / "rust-build.stdout",
                error.stdout if type(error.stdout) is bytes else b"",
            )
            _publish_run_file(
                evidence_root / "rust-build.stderr",
                error.stderr if type(error.stderr) is bytes else b"",
            )
            _publish_run_file(
                evidence_root / "rust-build-return-v1.json",
                _canonical(
                    {
                        "return_code": None,
                        "schema_version": (
                            "hegel-phase3a-q1-full-node6-rust-build-return/1"
                        ),
                        "timed_out": True,
                    }
                ),
            )
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", f"{label}: {error}")
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", f"{label}: {error}")
    if evidence_root is not None:
        _publish_run_file(evidence_root / "rust-build.stdout", completed.stdout)
        _publish_run_file(evidence_root / "rust-build.stderr", completed.stderr)
        _publish_run_file(
            evidence_root / "rust-build-return-v1.json",
            _canonical(
                {
                    "return_code": completed.returncode,
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-rust-build-return/1"
                    ),
                    "timed_out": False,
                }
            ),
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-1000:]
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", f"{label}: {detail}")
    return completed.stdout


def _require_local_image(image: str) -> None:
    payload = _run_capture([DOCKER, "image", "inspect", image], f"image {image}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", f"image inspect JSON: {error}")
    if type(value) is not list or len(value) != 1:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", f"image {image} is not local")


def _wait_cid(cidfile: Path, process: subprocess.Popen[bytes]) -> str:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            value = cidfile.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            value = ""
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"cidfile: {error}")
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value
        if process.poll() is not None:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "endpoint exited before cid capture")
        time.sleep(0.02)
    _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "container id capture timed out")


def _read_cgroup_evidence(pid: int) -> dict[str, object]:
    try:
        proc_payload = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"proc cgroup: {error}")
    match = re.fullmatch(r"0::(/[^\r\n]*)\n", proc_payload)
    if match is None:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "host is not exposing cgroup v2")
    relative = match.group(1).lstrip("/")
    root = Path("/sys/fs/cgroup") / relative
    rows: dict[str, str] = {}
    for name in (
        "cpu.stat",
        "memory.current",
        "memory.events",
        "memory.max",
        "memory.peak",
        "pids.current",
        "pids.max",
        "pids.peak",
    ):
        try:
            rows[name] = (root / name).read_text(encoding="ascii")
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"{name}: {error}")
    return {"cgroup_path": "/" + relative, "rows": rows}


def _capture_resource_evidence(
    container_id: str,
    actor_id: str,
    expected_seccomp_profile: Path,
) -> dict[str, object]:
    payload = _run_capture([DOCKER, "inspect", container_id], f"inspect {actor_id}")
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"inspect JSON: {error}")
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "container inspect cardinality differs")
    row = rows[0]
    state = row.get("State")
    host_config = row.get("HostConfig")
    config = row.get("Config")
    mounts = row.get("Mounts")
    if (
        type(state) is not dict
        or type(host_config) is not dict
        or type(config) is not dict
        or type(mounts) is not list
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "container inspect shape differs")
    pid = state.get("Pid")
    expected_cpuset = "12-23" if actor_id == "RUST_ENDPOINT" else "0-11"
    expected_image = RUST_IMAGE if actor_id == "RUST_ENDPOINT" else PYTHON_IMAGE
    if actor_id == "TRUSTED_HOST_REPLAY":
        expected_mounts = {
            "/python-output": False,
            "/rust-output": False,
            "/scratch": True,
            "/snapshot": False,
        }
    else:
        expected_mounts = (
        {
            "/admission/admission.json": False,
            "/output": True,
            "/scratch": True,
            "/snapshot": False,
        }
        if actor_id == "PYTHON_ENDPOINT"
        else {
            RUST_BINARY_CONTAINER_PATH: False,
            "/output": True,
            "/scratch": True,
            "/snapshot": False,
        }
        )
    observed_mounts = {
        item.get("Destination"): item.get("RW")
        for item in mounts
        if type(item) is dict and item.get("Type") == "bind"
    }
    if (
        type(pid) is not int
        or pid < 1
        or host_config.get("NetworkMode") != "none"
        or host_config.get("IpcMode") != "private"
        or host_config.get("ReadonlyRootfs") is not True
        or host_config.get("CpusetCpus") != expected_cpuset
        or host_config.get("Memory") != MEMORY_BYTES
        or host_config.get("MemorySwap") != MEMORY_BYTES
        or host_config.get("PidsLimit") != PIDS_LIMIT
        or host_config.get("Ulimits")
        != [{"Name": "nofile", "Hard": 256, "Soft": 256}]
        or host_config.get("Tmpfs")
        != {"/tmp": TMPFS_SPEC.split(":", 1)[1]}
        or host_config.get("CapDrop") != ["ALL"]
        or "no-new-privileges" not in host_config.get("SecurityOpt", [])
        or f"seccomp={expected_seccomp_profile}" not in host_config.get(
            "SecurityOpt", []
        )
        or config.get("Image") != expected_image
        or config.get("User") != CONTAINER_USER
        or observed_mounts != expected_mounts
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"{actor_id} isolation differs")
    cgroup = _read_cgroup_evidence(pid)
    cgroup_rows = cgroup.get("rows")
    memory_peak_text = (
        cgroup_rows.get("memory.peak", "").strip()
        if type(cgroup_rows) is dict
        else ""
    )
    if not memory_peak_text.isascii() or not memory_peak_text.isdigit():
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "memory.peak differs")
    return {
        "actor_id": actor_id,
        "cgroup": cgroup,
        "cgroup_capture_phase": "AFTER_ACTOR_OUTPUT_BEFORE_CONTAINER_RELEASE",
        "container_id": container_id,
        "cpuset_cpus": expected_cpuset,
        "memory_bytes": MEMORY_BYTES,
        "memory_peak_bytes": int(memory_peak_text),
        "memory_swap_bytes": MEMORY_BYTES,
        "network_mode": "none",
        "ipc_mode": "private",
        "nofile_ulimit": NOFILE_ULIMIT,
        "pids_limit": PIDS_LIMIT,
        "read_only_rootfs": True,
        "runtime_seccomp_sha256": SECCOMP_SHA256,
        "schema_version": RESOURCE_EVIDENCE_SCHEMA_VERSION,
        "tmpfs": TMPFS_SPEC,
        "user": CONTAINER_USER,
    }


def _container_name_from_command(command: Sequence[str]) -> str:
    positions = [index for index, value in enumerate(command) if value == "--name"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", "container name registry differs")
    value = command[positions[0] + 1]
    if type(value) is not str or not value:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", "container name differs")
    return value


def _seccomp_profile_from_command(command: Sequence[str]) -> Path:
    values = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--security-opt"
    ]
    matches = [value for value in values if value.startswith("seccomp=")]
    if len(matches) != 1:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", "seccomp registry differs")
    return Path(matches[0].split("=", 1)[1])


def _bind_source_from_command(command: Sequence[str], destination: str) -> Path:
    matches: list[Path] = []
    for index, value in enumerate(command[:-1]):
        if value != "--mount":
            continue
        fields = command[index + 1].split(",")
        registry = {
            field.split("=", 1)[0]: field.split("=", 1)[1]
            for field in fields
            if "=" in field
        }
        if registry.get("type") == "bind" and registry.get("dst") == destination:
            source = registry.get("src")
            if source is not None:
                matches.append(Path(source))
    if len(matches) != 1:
        _fail(
            "FAIL_FULL_NODE6_SUPERVISOR_DOCKER",
            f"mount registry differs for {destination}",
        )
    return matches[0]


def _actor_ready_now_v1(scratch: Path) -> int | None:
    ready = scratch / ACTOR_READY_FILENAME
    return_code = scratch / ACTOR_RETURN_CODE_FILENAME
    if not ready.exists() and not return_code.exists():
        return None
    try:
        ready_status = ready.lstat()
        return_status = return_code.lstat()
        payload = return_code.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"actor ready: {error}")
    if (
        ready.is_symlink()
        or return_code.is_symlink()
        or not stat.S_ISREG(ready_status.st_mode)
        or not stat.S_ISREG(return_status.st_mode)
        or ready_status.st_size != 0
        or re.fullmatch(r"-?[0-9]+\n", payload) is None
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "actor ready wire differs")
    try:
        for path in (return_code, ready):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(scratch)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"actor ready fsync: {error}")
    return int(payload)


def _publish_actor_release_v1(scratch: Path) -> int:
    release_started_ns = time.monotonic_ns()
    _publish_run_file(scratch / SUPERVISOR_RELEASE_FILENAME, b"release\n")
    return release_started_ns


def _add_interval_evidence_v1(
    evidence: dict[str, object],
    *,
    start_ns: int,
    actor_exit_ns: int,
    cgroup_capture_ns: int,
    release_ns: int,
    container_exit_ns: int,
    actor_return_code: int,
) -> dict[str, object]:
    if not (
        start_ns <= actor_exit_ns <= cgroup_capture_ns <= release_ns <= container_exit_ns
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "wall evidence order differs")
    return {
        **evidence,
        "actor_command_return_code": actor_return_code,
        "actor_completed_wall_time_ns": actor_exit_ns - start_ns,
        "actor_exit_after_output_monotonic_ns": actor_exit_ns,
        "cgroup_capture_monotonic_ns": cgroup_capture_ns,
        "container_exit_monotonic_ns": container_exit_ns,
        "container_hold_release_monotonic_ns": release_ns,
        "measurement_interval_definition": (
            "HOST_MONOTONIC_IMMEDIATELY_BEFORE_DOCKER_POPEN_TO_DOCKER_"
            "CONTAINER_EXIT_INCLUDING_IMPORT_MEASUREMENT_ENCODER_OUTPUT_"
            "CGROUP_CAPTURE_AND_HOLD_RELEASE"
        ),
        "measurement_interval_start_monotonic_ns": start_ns,
        "measurement_wall_time_ns": container_exit_ns - start_ns,
        "measurement_wall_time_seconds": (
            container_exit_ns - start_ns
        ) / 1_000_000_000,
    }


def _force_remove_endpoint_containers_v1(targets: Sequence[str]) -> list[str]:
    failures: list[str] = []
    for target in dict.fromkeys(targets):
        try:
            completed = subprocess.run(
                [DOCKER, "rm", "-f", target],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{target}:{type(error).__name__}:{error}")
            continue
        stderr = completed.stderr.decode("utf-8", "replace")
        if completed.returncode != 0 and "No such container" not in stderr:
            failures.append(f"{target}:return_code={completed.returncode}:{stderr[-500:]}")
    return failures


def _stop_endpoint_cli_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _launch_pair_actual_v1(
    python_command: Sequence[str],
    rust_command: Sequence[str],
    python_cidfile: Path,
    rust_cidfile: Path,
    *,
    wall_time_seconds: float = ENDPOINT_PAIR_WALL_TIME_SECONDS,
) -> tuple[EndpointProcessResultV1, EndpointProcessResultV1]:
    """Start both containers and capture completed-workload cgroup peaks."""

    processes: list[subprocess.Popen[bytes]] = []
    python_name = _container_name_from_command(python_command)
    rust_name = _container_name_from_command(rust_command)
    python_seccomp = _seccomp_profile_from_command(python_command)
    rust_seccomp = _seccomp_profile_from_command(rust_command)
    python_scratch = _bind_source_from_command(python_command, "/scratch")
    rust_scratch = _bind_source_from_command(rust_command, "/scratch")
    if python_seccomp != rust_seccomp or python_scratch == rust_scratch:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_DOCKER", "endpoint isolation registry differs")
    python_cid: str | None = None
    rust_cid: str | None = None
    containers_removed = False
    pair_start = time.monotonic()
    deadline = pair_start + wall_time_seconds
    communication: dict[str, tuple[bytes, bytes]] = {}
    communication_errors: dict[str, BaseException] = {}
    communication_end_ns: dict[str, int] = {}

    def communicate(actor: str, process: subprocess.Popen[bytes]) -> None:
        try:
            communication[actor] = process.communicate()
        except BaseException as error:
            communication_errors[actor] = error
        finally:
            communication_end_ns[actor] = time.monotonic_ns()

    def terminate_as_timeout() -> NoReturn:
        nonlocal containers_removed
        cleanup_failures = _force_remove_endpoint_containers_v1(
            [python_cid or python_name, rust_cid or rust_name]
        )
        containers_removed = True
        _stop_endpoint_cli_processes(processes)
        detail = f"endpoint pair exceeded {wall_time_seconds}s"
        if cleanup_failures:
            detail += "; cleanup_failures=" + "|".join(cleanup_failures)
        _fail("FULL_NODE6_ENDPOINT_PAIR_TIMEOUT", detail)

    try:
        python_start_ns = time.monotonic_ns()
        python = subprocess.Popen(
            list(python_command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(python)
        rust_start_ns = time.monotonic_ns()
        rust = subprocess.Popen(
            list(rust_command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(rust)
        threads = [
            threading.Thread(
                target=communicate,
                args=("python", python),
                name="hegel-q1-node6-python-communicate",
                daemon=True,
            ),
            threading.Thread(
                target=communicate,
                args=("rust", rust),
                name="hegel-q1-node6-rust-communicate",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        python_cid = _wait_cid(python_cidfile, python)
        rust_cid = _wait_cid(rust_cidfile, rust)
        if python_cid == rust_cid:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "endpoint cgroup identity differs")

        pending = {
            "python": (python_scratch, python),
            "rust": (rust_scratch, rust),
        }
        actor_codes: dict[str, int] = {}
        actor_exit_ns: dict[str, int] = {}
        while pending:
            for actor, (scratch, process) in tuple(pending.items()):
                code = _actor_ready_now_v1(scratch)
                if code is not None:
                    actor_codes[actor] = code
                    actor_exit_ns[actor] = time.monotonic_ns()
                    del pending[actor]
                elif process.poll() is not None:
                    _fail(
                        "FAIL_FULL_NODE6_SUPERVISOR_ENDPOINT_IO",
                        f"{actor} container exited without completed-workload receipt",
                    )
            if pending:
                if time.monotonic() >= deadline:
                    terminate_as_timeout()
                time.sleep(0.02)

        python_evidence = _capture_resource_evidence(
            python_cid,
            "PYTHON_ENDPOINT",
            python_seccomp,
        )
        python_capture_ns = time.monotonic_ns()
        rust_evidence = _capture_resource_evidence(
            rust_cid,
            "RUST_ENDPOINT",
            rust_seccomp,
        )
        rust_capture_ns = time.monotonic_ns()
        python_release_ns = _publish_actor_release_v1(python_scratch)
        rust_release_ns = _publish_actor_release_v1(rust_scratch)

        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            terminate_as_timeout()
        if communication_errors:
            details = "|".join(
                f"{actor}:{type(error).__name__}:{error}"
                for actor, error in sorted(communication_errors.items())
            )
            _fail("FAIL_FULL_NODE6_SUPERVISOR_ENDPOINT_IO", details)
        python_stdout, python_stderr = communication["python"]
        rust_stdout, rust_stderr = communication["rust"]
        if (
            python.returncode != actor_codes["python"]
            or rust.returncode != actor_codes["rust"]
        ):
            _fail("FAIL_FULL_NODE6_SUPERVISOR_ENDPOINT_IO", "return code replay differs")
        python_evidence = _add_interval_evidence_v1(
            python_evidence,
            start_ns=python_start_ns,
            actor_exit_ns=actor_exit_ns["python"],
            cgroup_capture_ns=python_capture_ns,
            release_ns=python_release_ns,
            container_exit_ns=communication_end_ns["python"],
            actor_return_code=actor_codes["python"],
        )
        rust_evidence = _add_interval_evidence_v1(
            rust_evidence,
            start_ns=rust_start_ns,
            actor_exit_ns=actor_exit_ns["rust"],
            cgroup_capture_ns=rust_capture_ns,
            release_ns=rust_release_ns,
            container_exit_ns=communication_end_ns["rust"],
            actor_return_code=actor_codes["rust"],
        )
    except BaseException:
        if not containers_removed:
            _force_remove_endpoint_containers_v1(
                [python_cid or python_name, rust_cid or rust_name]
            )
        _stop_endpoint_cli_processes(processes)
        raise
    return (
        EndpointProcessResultV1(
            "PYTHON_ENDPOINT",
            python.returncode,
            python_stdout,
            python_stderr,
            python_evidence,
        ),
        EndpointProcessResultV1(
            "RUST_ENDPOINT",
            rust.returncode,
            rust_stdout,
            rust_stderr,
            rust_evidence,
        ),
    )


def _launch_host_replay_actual_v1(
    command: Sequence[str],
    cidfile: Path,
    *,
    wall_time_seconds: float = HOST_REPLAY_WALL_TIME_SECONDS,
) -> EndpointProcessResultV1:
    """Run one isolated replay and capture its completed-workload cgroup peak."""

    name = _container_name_from_command(command)
    seccomp = _seccomp_profile_from_command(command)
    scratch = _bind_source_from_command(command, "/scratch")
    process: subprocess.Popen[bytes] | None = None
    cid: str | None = None
    removed = False
    deadline = time.monotonic() + wall_time_seconds
    communication: tuple[bytes, bytes] | None = None
    communication_error: BaseException | None = None
    container_exit_ns: int | None = None

    def communicate() -> None:
        nonlocal communication, communication_error, container_exit_ns
        try:
            assert process is not None
            communication = process.communicate()
        except BaseException as error:
            communication_error = error
        finally:
            container_exit_ns = time.monotonic_ns()

    try:
        start_ns = time.monotonic_ns()
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        thread = threading.Thread(
            target=communicate,
            name="hegel-q1-node6-host-replay-communicate",
            daemon=True,
        )
        thread.start()
        cid = _wait_cid(cidfile, process)
        code: int | None = None
        while code is None:
            code = _actor_ready_now_v1(scratch)
            if code is None and process.poll() is not None:
                _fail(
                    "FAIL_FULL_NODE6_SUPERVISOR_HOST_REPLAY_IO",
                    "host replay exited without completed-workload receipt",
                )
            if code is None:
                if time.monotonic() >= deadline:
                    failures = _force_remove_endpoint_containers_v1([cid or name])
                    removed = True
                    _stop_endpoint_cli_processes([process])
                    detail = f"host replay exceeded {wall_time_seconds}s"
                    if failures:
                        detail += "; cleanup_failures=" + "|".join(failures)
                    _fail("FULL_NODE6_HOST_REPLAY_TIMEOUT", detail)
                time.sleep(0.02)
        actor_exit_ns = time.monotonic_ns()
        evidence = _capture_resource_evidence(cid, "TRUSTED_HOST_REPLAY", seccomp)
        capture_ns = time.monotonic_ns()
        release_ns = _publish_actor_release_v1(scratch)
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            failures = _force_remove_endpoint_containers_v1([cid or name])
            removed = True
            _stop_endpoint_cli_processes([process])
            detail = f"host replay exceeded {wall_time_seconds}s"
            if failures:
                detail += "; cleanup_failures=" + "|".join(failures)
            _fail("FULL_NODE6_HOST_REPLAY_TIMEOUT", detail)
        if communication_error is not None or communication is None:
            _fail(
                "FAIL_FULL_NODE6_SUPERVISOR_HOST_REPLAY_IO",
                f"{type(communication_error).__name__}:{communication_error}",
            )
        if process.returncode != code or container_exit_ns is None:
            _fail(
                "FAIL_FULL_NODE6_SUPERVISOR_HOST_REPLAY_IO",
                "return code replay differs",
            )
        evidence = _add_interval_evidence_v1(
            evidence,
            start_ns=start_ns,
            actor_exit_ns=actor_exit_ns,
            cgroup_capture_ns=capture_ns,
            release_ns=release_ns,
            container_exit_ns=container_exit_ns,
            actor_return_code=code,
        )
        stdout, stderr = communication
        return EndpointProcessResultV1(
            "TRUSTED_HOST_REPLAY",
            process.returncode,
            stdout,
            stderr,
            evidence,
        )
    except BaseException:
        if not removed:
            _force_remove_endpoint_containers_v1([cid or name])
        if process is not None:
            _stop_endpoint_cli_processes([process])
        raise


LaunchPair = Callable[
    [Sequence[str], Sequence[str], Path, Path],
    tuple[EndpointProcessResultV1, EndpointProcessResultV1],
]


def _publish_run_file(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(path.parent)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_EVIDENCE", f"{path.name}: {error}")


def _prepare_scratch_roots_v1(run_root: Path, roots: Sequence[Path]) -> None:
    created: list[Path] = []
    try:
        for root in roots:
            if root.parent != run_root or root.exists():
                _fail(
                    "FAIL_FULL_NODE6_SUPERVISOR_SCRATCH",
                    f"scratch root differs: {root}",
                )
            root.mkdir(mode=0o700)
            root.chmod(0o777)
            created.append(root)
        _fsync_directory(run_root)
    except BaseException:
        for root in created:
            if root.exists() and not root.is_symlink():
                shutil.rmtree(root)
        _fsync_directory(run_root)
        raise


def _cleanup_scratch_roots_v1(
    run_root: Path,
    roots: Sequence[Path],
    *,
    receipt_name: str = "scratch-cleanup-v1.json",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for root in roots:
        if root.parent != run_root:
            failures.append(f"outside_run_root:{root}")
            continue
        try:
            if root.is_symlink():
                raise OSError("scratch root became a symlink")
            if root.exists():
                shutil.rmtree(root)
            absent = not root.exists() and not root.is_symlink()
        except OSError as error:
            absent = False
            failures.append(f"{root.name}:{type(error).__name__}:{error}")
        rows.append(
            {
                "host_path": os.fspath(root),
                "root_absent_after_cleanup": absent,
            }
        )
        if not absent and not any(value.startswith(f"{root.name}:") for value in failures):
            failures.append(f"{root.name}:absence proof failed")
    _fsync_directory(run_root)
    receipt = {
        "failure_rows": failures,
        "rows": rows,
        "schema_version": "hegel-phase3a-q1-full-node6-scratch-cleanup/1",
        "terminal_absence_proved": not failures and all(
            row["root_absent_after_cleanup"] is True for row in rows
        ),
    }
    _publish_run_file(run_root / receipt_name, _canonical(receipt))
    if failures:
        _fail(
            "FAIL_FULL_NODE6_SUPERVISOR_SCRATCH_CLEANUP",
            "|".join(failures),
        )
    return receipt


def _prepare_endpoint_output_roots_v1(
    run_root: Path,
    roots: Sequence[Path],
) -> None:
    for root in roots:
        if root.parent != run_root or root.exists():
            _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", "output root differs")
        root.mkdir(mode=0o700)
        root.chmod(0o777)
    _fsync_directory(run_root)


def _set_endpoint_output_access_v1(
    roots: Sequence[Path],
    *,
    mode: int,
) -> None:
    for root in roots:
        try:
            status = root.lstat()
            children = tuple(root.iterdir())
        except OSError as error:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", f"output root: {error}")
        if root.is_symlink() or not stat.S_ISDIR(status.st_mode):
            _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", "output root identity differs")
        for child in children:
            try:
                child_status = child.lstat()
            except OSError as error:
                _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", f"output child: {error}")
            if child.is_symlink() or not stat.S_ISREG(child_status.st_mode):
                _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", "output child differs")
            if mode == 0o555 and child_status.st_mode & 0o004 == 0:
                _fail(
                    "FAIL_FULL_NODE6_SUPERVISOR_ISOLATION",
                    "output child is not replay-readable",
                )
        root.chmod(mode)
        if stat.S_IMODE(root.lstat().st_mode) != mode:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_ISOLATION", "output mode differs")
        _fsync_directory(root.parent)


def _validate_host_replay_result_v1(
    payload: bytes,
    *,
    capsule_digest: str,
    source_commit: str,
) -> dict[str, object]:
    if len(payload) > PROJECTED_HOST_REPLAY_OUTPUT_BYTES:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_HOST_REPLAY_WIRE", "output bound exceeded")
    value = _canonical_object(payload, "host replay stdout")
    endpoint_rows = value.get("endpoint_rows")
    endpoint_output_accounting_valid = (
        type(endpoint_rows) is list
        and len(endpoint_rows) == 2
        and [row.get("actor_id") for row in endpoint_rows if type(row) is dict]
        == ["PYTHON_ENDPOINT", "RUST_ENDPOINT"]
        and all(
            type(row.get("actual_diagnostic_output_bytes")) is int
            and row["actual_diagnostic_output_bytes"] > 0
            and type(row.get("actual_preflight_peak_scratch_bytes")) is int
            and row["actual_preflight_peak_scratch_bytes"] >= 0
            for row in endpoint_rows
            if type(row) is dict
        )
    )
    terminal_status = value.get("terminal_status")
    formal_resource_projection = value.get("formal_resource_projection")
    formal_resource_valid = (
        type(formal_resource_projection) is dict
        and (
            (
                terminal_status == SUCCESS_STATUS
                and type(
                    formal_resource_projection.get(
                        "projected_formal_total_output_bytes"
                    )
                )
                is int
                and formal_resource_projection[
                    "projected_formal_total_output_bytes"
                ]
                > 0
                and type(
                    formal_resource_projection.get(
                        "projected_peak_formal_scratch_bytes"
                    )
                )
                is int
                and formal_resource_projection[
                    "projected_peak_formal_scratch_bytes"
                ]
                >= 0
            )
            or (
                terminal_status == GUARD_STATUS
                and formal_resource_projection
                == {
                    "projected_formal_total_output_bytes": None,
                    "projected_peak_formal_scratch_bytes": None,
                }
            )
        )
    )
    if (
        value.get("schema_version")
        != "hegel-phase3a-q1-full-node6-host-replay/1"
        or value.get("action_id") != ACTION_ID
        or value.get("capsule_digest") != capsule_digest
        or value.get("source_commit") != source_commit
        or value.get("diagnostic_only") is not True
        or value.get("q1_state") != "NOT_RUN"
        or value.get("q1_gate_mask") != 0
        or value.get("q1_gate_count") != 0
        or value.get("q1_output_slots") != [None] * 8
        or value.get("q1_formal_roots") is not None
        or value.get("q1_receipt") is not None
        or value.get("projected_host_replay_output_bytes")
        != PROJECTED_HOST_REPLAY_OUTPUT_BYTES
        or value.get("projected_peak_host_replay_scratch_bytes")
        != PROJECTED_PEAK_HOST_REPLAY_SCRATCH_BYTES
        or not endpoint_output_accounting_valid
        or not formal_resource_valid
        or (
            terminal_status == SUCCESS_STATUS
            and value.get("registered_resource_guard_id") is not None
        )
        or (
            terminal_status == GUARD_STATUS
            and value.get("registered_resource_guard_id") not in range(1, 9)
        )
        or value.get("terminal_status") not in (SUCCESS_STATUS, GUARD_STATUS)
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_HOST_REPLAY_WIRE", "boundary differs")
    return value


def _pow2ceil_v1(value: int) -> int:
    if type(value) is not int or value < 0:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "pow2ceil input differs")
    return 1 << (max(1, value) - 1).bit_length()


def _hour_rounded_wall_guard_seconds_v1(elapsed_ns: int) -> int:
    if type(elapsed_ns) is not int or elapsed_ns < 0:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "wall input differs")
    hour_ns = 3_600 * 1_000_000_000
    return 3_600 * ((3 * elapsed_ns + hour_ns - 1) // hour_ns)


def _adjudicate_resources_v1(
    replay: Mapping[str, object],
    python: EndpointProcessResultV1,
    rust: EndpointProcessResultV1,
    host_process: EndpointProcessResultV1,
) -> dict[str, object]:
    endpoint_rows = replay.get("endpoint_rows")
    if type(endpoint_rows) is not list or len(endpoint_rows) != 2:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "endpoint resource rows differ")
    evidence_rows = [python.resource_evidence, rust.resource_evidence]
    for expected_actor, evidence in zip(
        ("PYTHON_ENDPOINT", "RUST_ENDPOINT"),
        evidence_rows,
        strict=True,
    ):
        if evidence.get("actor_id") != expected_actor:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "endpoint evidence actor differs")
        for field in ("memory_peak_bytes", "measurement_wall_time_ns"):
            if type(evidence.get(field)) is not int or evidence[field] < 0:
                _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"{field} differs")
    host_evidence = host_process.resource_evidence
    if host_evidence.get("actor_id") != "TRUSTED_HOST_REPLAY":
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "host evidence actor differs")
    for field in ("memory_peak_bytes", "measurement_wall_time_ns"):
        if type(host_evidence.get(field)) is not int or host_evidence[field] < 0:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", f"host {field} differs")
    all_evidence = [*evidence_rows, host_evidence]
    container_ids = [row.get("container_id") for row in all_evidence]
    cgroup_paths = [
        row.get("cgroup", {}).get("cgroup_path")
        if type(row.get("cgroup")) is dict
        else None
        for row in all_evidence
    ]
    if (
        any(type(value) is not str or not value for value in container_ids)
        or len(set(container_ids)) != 3
        or any(type(value) is not str or not value.startswith("/") for value in cgroup_paths)
        or len(set(cgroup_paths)) != 3
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "fresh cgroup identity differs")

    direct_inputs = []
    for endpoint_row, evidence in zip(endpoint_rows, evidence_rows, strict=True):
        if type(endpoint_row) is not dict:
            _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "endpoint accounting differs")
        direct_inputs.append(
            {
                "actor_id": endpoint_row["actor_id"],
                "actual_diagnostic_output_bytes": endpoint_row[
                    "actual_diagnostic_output_bytes"
                ],
                "actual_preflight_peak_scratch_bytes": endpoint_row[
                    "actual_preflight_peak_scratch_bytes"
                ],
                "actual_peak_rss_bytes": evidence["memory_peak_bytes"],
                "actual_wall_time_ns": evidence["measurement_wall_time_ns"],
            }
        )
    input_terminal = replay.get("terminal_status")
    registered_guard_id = replay.get("registered_resource_guard_id")
    if input_terminal == GUARD_STATUS:
        return {
            "derived_envelope": {"evaluated": False, "reason": "REGISTERED_GUARD_1_TO_8"},
            "direct_endpoint_resources": direct_inputs,
            "first_resource_guard_id": registered_guard_id,
            "first_resource_guard_name": "REGISTERED_ENDPOINT_GUARD",
            "host_replay_passed": True,
            "input_terminal_status": input_terminal,
            "q1_state": "NOT_RUN",
            "schema_version": "hegel-phase3a-q1-full-node6-resource-adjudication/1",
            "terminal_status": GUARD_STATUS,
        }

    violated_direct: list[int] = []
    if any(row["actual_diagnostic_output_bytes"] > ENDPOINT_OUTPUT_CEILING_BYTES for row in direct_inputs):
        violated_direct.append(9)
    if any(row["actual_preflight_peak_scratch_bytes"] > ENDPOINT_SCRATCH_CEILING_BYTES for row in direct_inputs):
        violated_direct.append(10)
    if any(row["actual_peak_rss_bytes"] > ENDPOINT_MEMORY_CEILING_BYTES for row in direct_inputs):
        violated_direct.append(11)
    if any(
        row["actual_wall_time_ns"]
        > ENDPOINT_WALL_CEILING_SECONDS * 1_000_000_000
        for row in direct_inputs
    ):
        violated_direct.append(12)
    guard_names = {
        9: "OUTPUT_BYTES",
        10: "SCRATCH_BYTES",
        11: "RESIDENT_MEMORY",
        12: "WALL_TIME",
    }
    if violated_direct:
        first_guard = min(violated_direct)
        return {
            "derived_envelope": {"evaluated": False, "reason": "DIRECT_GUARD_9_TO_12"},
            "direct_endpoint_resources": direct_inputs,
            "first_resource_guard_id": first_guard,
            "first_resource_guard_name": guard_names[first_guard],
            "host_replay_passed": True,
            "input_terminal_status": input_terminal,
            "q1_state": "NOT_RUN",
            "schema_version": "hegel-phase3a-q1-full-node6-resource-adjudication/1",
            "terminal_status": GUARD_STATUS,
            "violated_direct_resource_guard_ids": violated_direct,
        }

    formal = replay.get("formal_resource_projection")
    if type(formal) is not dict:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "formal resource projection differs")
    projected_formal_output = formal.get("projected_formal_total_output_bytes")
    projected_formal_scratch = formal.get("projected_peak_formal_scratch_bytes")
    if (
        type(projected_formal_output) is not int
        or projected_formal_output < 0
        or type(projected_formal_scratch) is not int
        or projected_formal_scratch < 0
    ):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "formal resource values differ")
    maximum_endpoint_rss = max(row["actual_peak_rss_bytes"] for row in direct_inputs)
    slower_endpoint_wall = max(row["actual_wall_time_ns"] for row in direct_inputs)
    projected_host_output = replay.get("projected_host_replay_output_bytes")
    projected_host_scratch = replay.get("projected_peak_host_replay_scratch_bytes")
    if type(projected_host_output) is not int or type(projected_host_scratch) is not int:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_RESOURCE", "host projection values differ")
    derived = {
        "endpoint_memory_guard_bytes": _pow2ceil_v1(2 * maximum_endpoint_rss),
        "endpoint_output_guard_bytes": _pow2ceil_v1(2 * projected_formal_output),
        "endpoint_scratch_guard_bytes": _pow2ceil_v1(2 * projected_formal_scratch),
        "endpoint_wall_guard_seconds": _hour_rounded_wall_guard_seconds_v1(
            slower_endpoint_wall
        ),
        "evaluated": True,
        "host_memory_guard_bytes": _pow2ceil_v1(
            2 * host_evidence["memory_peak_bytes"]
        ),
        "host_output_guard_bytes": _pow2ceil_v1(2 * projected_host_output),
        "host_scratch_guard_bytes": _pow2ceil_v1(2 * projected_host_scratch),
        "host_wall_guard_seconds": _hour_rounded_wall_guard_seconds_v1(
            host_evidence["measurement_wall_time_ns"]
        ),
    }
    ceilings = {
        "endpoint_memory_guard_bytes": ENDPOINT_MEMORY_CEILING_BYTES,
        "endpoint_output_guard_bytes": ENDPOINT_OUTPUT_CEILING_BYTES,
        "endpoint_scratch_guard_bytes": ENDPOINT_SCRATCH_CEILING_BYTES,
        "endpoint_wall_guard_seconds": ENDPOINT_WALL_CEILING_SECONDS,
        "host_memory_guard_bytes": HOST_REPLAY_MEMORY_CEILING_BYTES,
        "host_output_guard_bytes": HOST_REPLAY_OUTPUT_CEILING_BYTES,
        "host_scratch_guard_bytes": HOST_REPLAY_SCRATCH_CEILING_BYTES,
        "host_wall_guard_seconds": HOST_REPLAY_WALL_CEILING_SECONDS,
    }
    exceeded = [
        field
        for field, ceiling in ceilings.items()
        if derived[field] > ceiling
    ]
    terminal_status = INFEASIBLE_STATUS if exceeded else SUCCESS_STATUS
    return {
        "derived_envelope": derived,
        "derived_hard_ceilings": ceilings,
        "direct_endpoint_resources": direct_inputs,
        "first_resource_guard_id": None,
        "first_resource_guard_name": None,
        "host_replay_actual_resources": {
            "actual_peak_rss_bytes": host_evidence["memory_peak_bytes"],
            "actual_wall_time_ns": host_evidence["measurement_wall_time_ns"],
        },
        "host_replay_passed": True,
        "infeasible_derived_fields": exceeded,
        "input_terminal_status": input_terminal,
        "q1_state": "NOT_RUN",
        "schema_version": "hegel-phase3a-q1-full-node6-resource-adjudication/1",
        "terminal_status": terminal_status,
    }


def run_supervisor_v1(
    spec: SupervisorRunSpecV1,
    *,
    launch_pair: LaunchPair = _launch_pair_actual_v1,
    launch_host_replay: HostReplay = _launch_host_replay_actual_v1,
    check_local_images: bool = True,
    trajectory: SupervisorTrajectoryV1 | None = None,
) -> dict[str, object]:
    """Run one dual measurement and host replay, with no retry path."""

    _require_distinct_snapshots(spec)
    if check_local_images:
        _require_local_image(PYTHON_IMAGE)
        _require_local_image(RUST_IMAGE)
    if spec.run_root.exists():
        allowed = {
            "admission.json",
            "cargo-home",
            "host-replay-snapshot",
            "python-snapshot",
            "rust-build-command-v1.json",
            "rust-build-return-v1.json",
            "rust-build.stderr",
            "rust-build.stdout",
            "rust-build",
            "rust-snapshot",
            "seccomp-profile-v1.json",
            "source-closure-receipt-v1.json",
        }
        try:
            observed = {path.name for path in spec.run_root.iterdir()}
        except OSError as error:
            _fail("REJECT_FULL_NODE6_SUPERVISOR_SPEC", f"run root: {error}")
        if not observed.issubset(allowed):
            _fail("REJECT_FULL_NODE6_SUPERVISOR_SPEC", "run root contains unknown state")
    else:
        spec.run_root.mkdir(mode=0o700, parents=True)
    owns_trajectory = trajectory is None
    if trajectory is None:
        trajectory = SupervisorTrajectoryV1.create(
            spec.run_root,
            spec.trajectory_root,
            spec.capsule_digest,
            spec.source_commit,
        )
    active_phase = "ENDPOINT_COMMAND_PREPARATION"
    trajectory.emit(active_phase, "STARTED")
    try:
        python_output = spec.run_root / "python-output"
        rust_output = spec.run_root / "rust-output"
        python_scratch = spec.run_root / "python-scratch"
        rust_scratch = spec.run_root / "rust-scratch"
        scratch_roots = (python_scratch, rust_scratch)
        output_roots = (python_output, rust_output)
        _prepare_endpoint_output_roots_v1(spec.run_root, output_roots)
        python_cidfile = spec.run_root / "python.cid"
        rust_cidfile = spec.run_root / "rust.cid"
        host_cidfile = spec.run_root / "host-replay.cid"
        host_scratch = spec.run_root / "host-replay-scratch"
        python_command, rust_command = endpoint_commands_v1(
            spec,
            python_output=python_output,
            rust_output=rust_output,
            python_scratch=python_scratch,
            rust_scratch=rust_scratch,
            python_cidfile=python_cidfile,
            rust_cidfile=rust_cidfile,
        )
        host_command = host_replay_command_v1(
            spec,
            python_output=python_output,
            rust_output=rust_output,
            host_scratch=host_scratch,
            host_cidfile=host_cidfile,
        )
        _publish_run_file(
            spec.run_root / "execution-plan-v1.json",
            _canonical(
                {
                    "network_allowed": False,
                    "endpoint_pair_wall_time_seconds": (
                        ENDPOINT_PAIR_WALL_TIME_SECONDS
                    ),
                    "host_replay_command": host_command,
                    "host_replay_source_manifest": [HOST_REPLAY_RELATIVE_PATH],
                    "host_replay_wall_time_seconds": HOST_REPLAY_WALL_TIME_SECONDS,
                    "isolation": {
                        "ipc": "private",
                        "nofile_ulimit": NOFILE_ULIMIT,
                        "pids_limit": PIDS_LIMIT,
                        "runtime_seccomp_sha256": SECCOMP_SHA256,
                        "tmpfs": TMPFS_SPEC,
                        "user": CONTAINER_USER,
                    },
                    "python_command": python_command,
                    "python_rust_concurrent": True,
                    "rust_command": rust_command,
                    "scratch": {
                        "cleanup_receipt_relative_path": "scratch-cleanup-v1.json",
                        "container_path": "/scratch",
                        "python_host_path": os.fspath(python_scratch),
                        "roots_absent_after_endpoint_exit_required": True,
                        "rust_host_path": os.fspath(rust_scratch),
                        "host_replay_host_path": os.fspath(host_scratch),
                    },
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-supervisor-execution-plan/1"
                    ),
                    "trusted_host_replay_after_both_exit": True,
                }
            ),
        )
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="execution-plan-v1.json",
        )

        active_phase = "SCRATCH_PREPARATION"
        trajectory.emit(active_phase, "STARTED")
        _prepare_scratch_roots_v1(spec.run_root, scratch_roots)
        trajectory.emit(
            active_phase,
            "COMPLETED",
            decision="DISTINCT_EMPTY_WRITABLE_ROOTS",
        )

        active_phase = "ENDPOINT_PAIR"
        trajectory.emit(
            active_phase,
            "STARTED",
            command_artifact="execution-plan-v1.json",
        )
        try:
            python, rust = launch_pair(
                python_command,
                rust_command,
                python_cidfile,
                rust_cidfile,
            )
            _publish_run_file(
                spec.run_root / "endpoint-resource-evidence-v1.json",
                _canonical(
                    {
                        "rows": [python.resource_evidence, rust.resource_evidence],
                        "schema_version": RESOURCE_EVIDENCE_SCHEMA_VERSION,
                    }
                ),
            )
        finally:
            active_phase = "SCRATCH_CLEANUP"
            trajectory.emit(active_phase, "STARTED")
            scratch_cleanup = _cleanup_scratch_roots_v1(
                spec.run_root,
                scratch_roots,
            )
            trajectory.emit(
                active_phase,
                "COMPLETED",
                command_artifact="scratch-cleanup-v1.json",
                decision="BOTH_SCRATCH_ROOTS_ABSENT",
            )
            active_phase = "ENDPOINT_PAIR"
        for actor, result in (("python", python), ("rust", rust)):
            _publish_run_file(spec.run_root / f"{actor}.stdout", result.stdout)
            _publish_run_file(spec.run_root / f"{actor}.stderr", result.stderr)
        trajectory.emit(
            active_phase,
            "COMPLETED",
            detail=(
                f"python_return_code={python.return_code};"
                f"rust_return_code={rust.return_code}"
            ),
            return_code=(
                0 if python.return_code == 0 and rust.return_code == 0 else 1
            ),
            decision="BOTH_ENDPOINTS_EXITED",
            command_artifact="endpoint-resource-evidence-v1.json",
        )
        if (
            python.actor_id != "PYTHON_ENDPOINT"
            or rust.actor_id != "RUST_ENDPOINT"
            or python.return_code != 0
            or rust.return_code != 0
        ):
            _fail("FULL_NODE6_ENDPOINT_FAILED", "an endpoint did not exit cleanly")

        active_phase = "ENDPOINT_OUTPUT_SEAL"
        trajectory.emit(active_phase, "STARTED")
        _set_endpoint_output_access_v1(output_roots, mode=0o555)
        trajectory.emit(
            active_phase,
            "COMPLETED",
            decision="DISTINCT_ENDPOINT_OUTPUTS_READ_ONLY_FOR_REPLAY",
        )

        active_phase = "TRUSTED_HOST_REPLAY"
        trajectory.emit(
            active_phase,
            "STARTED",
            command_artifact="execution-plan-v1.json",
        )
        _prepare_scratch_roots_v1(spec.run_root, (host_scratch,))
        try:
            host_process = launch_host_replay(host_command, host_cidfile)
            host_process.resource_evidence.update(
                {
                    "actual_captured_stdout_bytes": len(host_process.stdout),
                    "projected_host_replay_output_bytes": (
                        PROJECTED_HOST_REPLAY_OUTPUT_BYTES
                    ),
                    "projected_peak_host_replay_scratch_bytes": (
                        PROJECTED_PEAK_HOST_REPLAY_SCRATCH_BYTES
                    ),
                }
            )
            _publish_run_file(
                spec.run_root / "host-replay-resource-evidence-v1.json",
                _canonical(
                    {
                        "rows": [host_process.resource_evidence],
                        "schema_version": RESOURCE_EVIDENCE_SCHEMA_VERSION,
                    }
                ),
            )
        finally:
            host_scratch_cleanup = _cleanup_scratch_roots_v1(
                spec.run_root,
                (host_scratch,),
                receipt_name="host-replay-scratch-cleanup-v1.json",
            )
            _set_endpoint_output_access_v1(output_roots, mode=0o500)
        _publish_run_file(spec.run_root / "host-replay.stdout", host_process.stdout)
        _publish_run_file(spec.run_root / "host-replay.stderr", host_process.stderr)
        if (
            host_process.actor_id != "TRUSTED_HOST_REPLAY"
            or host_process.return_code != 0
        ):
            _fail("FULL_NODE6_HOST_REPLAY_FAILED", "trusted replay did not exit cleanly")
        replay = _validate_host_replay_result_v1(
            host_process.stdout,
            capsule_digest=spec.capsule_digest,
            source_commit=spec.source_commit,
        )
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="host-replay.stdout",
            return_code=host_process.return_code,
            decision=str(replay["terminal_status"]),
        )
        active_phase = "RESOURCE_ADJUDICATION"
        trajectory.emit(active_phase, "STARTED")
        resource_adjudication = _adjudicate_resources_v1(
            replay,
            python,
            rust,
            host_process,
        )
        _publish_run_file(
            spec.run_root / "resource-adjudication-v1.json",
            _canonical(resource_adjudication),
        )
        adjudicated_terminal = str(resource_adjudication["terminal_status"])
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="resource-adjudication-v1.json",
            decision=adjudicated_terminal,
        )
        active_phase = "SUPERVISOR_RESULT"
        trajectory.emit(active_phase, "STARTED")
        result = {
            "action_id": ACTION_ID,
            "capsule_digest": spec.capsule_digest,
            "diagnostic_only": True,
            "host_replay": replay,
            "host_replay_passed": True,
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_output_slots": [None] * 8,
            "q1_state": "NOT_RUN",
            "resource_evidence": [
                python.resource_evidence,
                rust.resource_evidence,
                host_process.resource_evidence,
            ],
            "resource_accounting": {
                "endpoint_actual_diagnostic_output_bytes": [
                    {
                        "actor_id": row["actor_id"],
                        "actual_diagnostic_output_bytes": row[
                            "actual_diagnostic_output_bytes"
                        ],
                    }
                    for row in replay["endpoint_rows"]
                ],
                "projected_host_replay_output_bytes": replay[
                    "projected_host_replay_output_bytes"
                ],
                "projected_peak_host_replay_scratch_bytes": replay[
                    "projected_peak_host_replay_scratch_bytes"
                ],
            },
            "resource_adjudication": resource_adjudication,
            "host_replay_scratch_cleanup": host_scratch_cleanup,
            "resource_evidence_artifacts": {
                "adjudication": os.fspath(
                    spec.run_root / "resource-adjudication-v1.json"
                ),
                "endpoints": os.fspath(
                    spec.run_root / "endpoint-resource-evidence-v1.json"
                ),
                "trusted_host_replay": os.fspath(
                    spec.run_root / "host-replay-resource-evidence-v1.json"
                ),
            },
            "scratch_cleanup": scratch_cleanup,
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "source_commit": spec.source_commit,
            "terminal_status": adjudicated_terminal,
            "trajectory": {
                "event_log_path": os.fspath(trajectory.event_path),
                "terminal_path": os.fspath(trajectory.terminal_path),
            },
        }
        _publish_run_file(
            spec.run_root / "supervisor-result-v1.json",
            _canonical(result),
        )
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="supervisor-result-v1.json",
            decision=adjudicated_terminal,
        )
        if owns_trajectory:
            trajectory.finish(str(result["terminal_status"]))
        return result
    except FullNode6SupervisorError as error:
        if "output_roots" in locals():
            existing_output_roots = tuple(
                root for root in output_roots if root.exists() and not root.is_symlink()
            )
            if existing_output_roots:
                _set_endpoint_output_access_v1(existing_output_roots, mode=0o500)
        if not trajectory.closed:
            trajectory.emit(active_phase, "FAILED", detail=str(error))
            if owns_trajectory:
                trajectory.finish(
                    "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME",
                    error_code=error.code,
                    error_detail=error.detail,
                )
        raise


def _git_output(project_root: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", os.fspath(project_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"{label}: {error}")
    if completed.returncode != 0:
        _fail(
            "FAIL_FULL_NODE6_SUPERVISOR_SOURCE",
            f"{label}: {completed.stderr.decode('utf-8', 'replace')[-1000:]}",
        )
    return completed.stdout


def _replay_clean_commit(project_root: Path, source_commit: str) -> None:
    try:
        resolved = project_root.resolve(strict=True)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"project root: {error}")
    if resolved != project_root or project_root.is_symlink():
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "project root differs")
    head = _git_output(project_root, ["rev-parse", "HEAD"], "HEAD").decode("ascii").strip()
    status = _git_output(
        project_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "worktree status",
    )
    if head != source_commit or status:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "source is not the exact clean commit")


def _materialize_snapshot(
    project_root: Path,
    source_commit: str,
    target: Path,
    source_paths: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Materialize an exact commit-blob allowlist, never a repository archive."""

    if target.exists() or not source_paths or len(set(source_paths)) != len(source_paths):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"snapshot registry: {target.name}")
    target.mkdir(mode=0o700)
    rows: list[dict[str, object]] = []
    try:
        for relative in source_paths:
            pure = Path(relative)
            if (
                type(relative) is not str
                or not relative
                or pure.is_absolute()
                or ".." in pure.parts
                or pure.as_posix() != relative
            ):
                _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "snapshot path differs")
            payload = _git_output(
                project_root,
                ["show", f"{source_commit}:{relative}"],
                f"blob {relative}",
            )
            destination = target / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _publish_run_file(destination, payload)
            rows.append(
                {
                    "byte_count": len(payload),
                    "relative_path": relative,
                    "sha256": sha256(payload).hexdigest(),
                }
            )
    except BaseException:
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        raise
    return tuple(rows)


def _materialize_host_replay_snapshot(
    project_root: Path,
    source_commit: str,
    target: Path,
) -> tuple[dict[str, object], ...]:
    """Materialize only the target-blind replay program into its snapshot."""

    return _materialize_snapshot(
        project_root,
        source_commit,
        target,
        HOST_REPLAY_SOURCE_PATHS,
    )


def _verify_snapshot_manifest_v1(
    root: Path,
    expected_rows: Sequence[Mapping[str, object]],
) -> None:
    expected = {
        row.get("relative_path"): row
        for row in expected_rows
        if type(row) is dict
    }
    if len(expected) != len(expected_rows) or any(type(path) is not str for path in expected):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "snapshot manifest differs")
    observed: dict[str, dict[str, object]] = {}
    try:
        for path in root.rglob("*"):
            status = path.lstat()
            if path.is_symlink():
                _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "snapshot symlink differs")
            if stat.S_ISDIR(status.st_mode):
                continue
            if not stat.S_ISREG(status.st_mode):
                _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "snapshot special file differs")
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            observed[relative] = {
                "byte_count": len(payload),
                "relative_path": relative,
                "sha256": sha256(payload).hexdigest(),
            }
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"snapshot replay: {error}")
    if observed != expected:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "snapshot file closure differs")


def _make_tree_readonly(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False):
        directory_path = Path(directory)
        for name in files:
            path = directory_path / name
            mode = path.stat().st_mode
            path.chmod(0o555 if mode & 0o111 else 0o444)
        for name in names:
            (directory_path / name).chmod(0o555)
        directory_path.chmod(0o555)


def _load_technical_contract(project_root: Path) -> dict[str, object]:
    path = project_root / TECHNICAL_CONTRACT_RELATIVE_PATH
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"technical contract: {error}")
    if type(value) is not dict:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "technical contract differs")
    return value


def _validate_contract_pair_deadline(contract: Mapping[str, object]) -> None:
    envelope = contract.get("resource_envelope")
    if type(envelope) is not dict:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "resource envelope differs")
    for actor in ("python_endpoint", "rust_endpoint"):
        row = envelope.get(actor)
        if (
            type(row) is not dict
            or row.get("wall_time_seconds") != PER_SIGNATURE_WALL_TIME_SECONDS
        ):
            _fail(
                "FAIL_FULL_NODE6_SUPERVISOR_SOURCE",
                f"{actor} wall-time differs from pair deadline",
            )
    host_replay = envelope.get("trusted_host_replay")
    measurement = contract.get("measurement_policy")
    execution = contract.get("execution_policy")
    if (
        type(measurement) is not dict
        or measurement.get("signatures_run_sequentially_within_endpoint") is not True
        or measurement.get("signature_order") != [1, 2]
        or measurement.get("per_signature_wall_time_seconds")
        != PER_SIGNATURE_WALL_TIME_SECONDS
        or measurement.get("ordered_signature_count") != ORDERED_SIGNATURE_COUNT
        or measurement.get("bounded_projection_wall_time_seconds")
        != BOUNDED_PROJECTION_WALL_TIME_SECONDS
        or measurement.get("endpoint_pair_process_deadline_seconds")
        != ENDPOINT_PAIR_WALL_TIME_SECONDS
        or measurement.get("trusted_host_replay_process_deadline_seconds")
        != HOST_REPLAY_WALL_TIME_SECONDS
        or type(host_replay) is not dict
        or host_replay.get("wall_time_seconds") != HOST_REPLAY_WALL_TIME_SECONDS
        or type(execution) is not dict
        or execution.get("outer_supervisor_wall_time_seconds")
        != OUTER_SUPERVISOR_WALL_TIME_SECONDS
    ):
        _fail(
            "FAIL_FULL_NODE6_SUPERVISOR_SOURCE",
            "measurement/outer wall-time binding differs",
        )


def _validate_contract_execution_source_closure(
    contract: Mapping[str, object],
) -> None:
    expected = {
        "python_endpoint_relative_paths_in_exact_order": list(
            PYTHON_ENDPOINT_SOURCE_PATHS
        ),
        "python_endpoint_row_count": len(PYTHON_ENDPOINT_SOURCE_PATHS),
        "rust_release_build_relative_paths_in_exact_order": list(
            RUST_ENDPOINT_SOURCE_PATHS
        ),
        "rust_release_build_row_count": len(RUST_ENDPOINT_SOURCE_PATHS),
        "endpoint_execution_union_row_count": EXECUTION_CLOSURE_ROW_COUNT,
        "host_replay_relative_paths_in_exact_order": list(
            HOST_REPLAY_SOURCE_PATHS
        ),
        "host_replay_row_count": len(HOST_REPLAY_SOURCE_PATHS),
        "target_split_role_files_mounted": False,
    }
    if contract.get("execution_source_closure") != expected:
        _fail(
            "FAIL_FULL_NODE6_SUPERVISOR_SOURCE",
            "execution source closure binding differs",
        )


def _write_admission(
    project_root: Path,
    path: Path,
    capsule_digest: str,
    source_commit: str,
) -> None:
    contract = _load_technical_contract(project_root)
    _validate_contract_pair_deadline(contract)
    _validate_contract_execution_source_closure(contract)
    predecessor = contract.get("predecessor")
    binding = contract.get("semantic_and_projection_binding")
    if type(predecessor) is not dict or type(binding) is not dict:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "technical binding differs")
    value = {
        "action_id": ACTION_ID,
        "capsule_digest": capsule_digest,
        "diagnostic_only": True,
        "predecessor_artifact_sha256": predecessor.get("artifact_sha256"),
        "predecessor_capsule_digest": predecessor.get("capsule_digest"),
        "predecessor_terminal_receipt_digest": predecessor.get(
            "terminal_ledger_record_digest"
        ),
        "projection_profile_root": binding.get("projection_profile_root"),
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "schema_version": "hegel-phase3a-q1-full-node6-capacity-admission/1",
        "source_commit": source_commit,
    }
    if path.exists():
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "admission already exists")
    _publish_run_file(path, _canonical(value))
    path.chmod(0o444)


def _prepare_seccomp_profile(project_root: Path, path: Path) -> None:
    source = project_root / SECCOMP_RELATIVE_PATH
    try:
        payload = source.read_bytes()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", f"seccomp source: {error}")
    if "sha256:" + sha256(payload).hexdigest() != SECCOMP_SHA256:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "seccomp source digest differs")
    _publish_run_file(path, payload)
    path.chmod(0o444)


def _build_rust_binary(
    rust_snapshot: Path,
    build_root: Path,
    cargo_home: Path,
) -> Path:
    if build_root.exists() or cargo_home.exists():
        _fail("FAIL_FULL_NODE6_SUPERVISOR_BUILD", "Rust build state already exists")
    build_root.mkdir(mode=0o700)
    try:
        shutil.copytree(CARGO_CACHE_SOURCE, cargo_home)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_BUILD", f"Cargo cache: {error}")
    command = [
        DOCKER,
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpuset-cpus",
        "12-23",
        "--memory",
        str(MEMORY_BYTES),
        "--memory-swap",
        str(MEMORY_BYTES),
        "--pids-limit",
        str(PIDS_LIMIT),
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=1073741824",
        "--mount",
        f"type=bind,src={rust_snapshot},dst=/snapshot,readonly",
        "--mount",
        f"type=bind,src={build_root},dst=/build",
        "--mount",
        f"type=bind,src={cargo_home},dst=/cargo",
        "--env",
        "CARGO_HOME=/cargo",
        "--env",
        "CARGO_NET_OFFLINE=true",
        "--workdir",
        "/snapshot/rust/q1_full_capacity_preflight_engine",
        RUST_IMAGE,
        "/usr/local/cargo/bin/cargo",
        "build",
        "--release",
        "--offline",
        "--locked",
        "--target-dir",
        "/build/target",
    ]
    _run_capture(
        command,
        "offline Rust release build",
        timeout_seconds=RUST_BUILD_TIMEOUT_SECONDS,
        evidence_root=build_root.parent,
    )
    binary = build_root / "target/release/hegel-q1-full-capacity-preflight-engine"
    try:
        status = binary.lstat()
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_BUILD", f"Rust binary: {error}")
    if binary.is_symlink() or not stat.S_ISREG(status.st_mode) or not os.access(binary, os.X_OK):
        _fail("FAIL_FULL_NODE6_SUPERVISOR_BUILD", "Rust binary output differs")
    return binary


def _publish_artifact(path: Path, value: Mapping[str, object]) -> tuple[str, int]:
    payload = _canonical(dict(value))
    try:
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(path.parent)
    except OSError as error:
        _fail("FAIL_FULL_NODE6_SUPERVISOR_ARTIFACT", str(error))
    return "sha256:" + sha256(payload).hexdigest(), len(payload)


def run_public_supervisor_v1(
    *,
    project_root: Path,
    source_commit: str,
    capsule_digest: str,
    artifact: Path,
    run_root: Path,
    trajectory_root: Path,
) -> dict[str, object]:
    """Implement the exact outer-campaign supervisor invocation."""

    if run_root.exists() or trajectory_root.exists() or artifact.exists():
        _fail(
            "REJECT_FULL_NODE6_SUPERVISOR_SPEC",
            "run root, trajectory root, or artifact already exists",
        )
    run_root.mkdir(mode=0o700, parents=True)
    trajectory = SupervisorTrajectoryV1.create(
        run_root,
        trajectory_root,
        capsule_digest,
        source_commit,
    )
    active_phase = "SOURCE_REPLAY"
    try:
        trajectory.emit(active_phase, "STARTED")
        _replay_clean_commit(project_root, source_commit)
        trajectory.emit(active_phase, "COMPLETED", decision="EXACT_CLEAN_COMMIT")

        active_phase = "LOCAL_IMAGE_ADMISSION"
        trajectory.emit(active_phase, "STARTED")
        _require_local_image(PYTHON_IMAGE)
        _require_local_image(RUST_IMAGE)
        trajectory.emit(active_phase, "COMPLETED", decision="BOTH_IMAGES_LOCAL")

        python_snapshot = run_root / "python-snapshot"
        rust_snapshot = run_root / "rust-snapshot"
        host_replay_snapshot = run_root / "host-replay-snapshot"
        active_phase = "SOURCE_SNAPSHOTS"
        trajectory.emit(active_phase, "STARTED")
        python_source_rows = _materialize_snapshot(
            project_root,
            source_commit,
            python_snapshot,
            PYTHON_ENDPOINT_SOURCE_PATHS,
        )
        rust_source_rows = _materialize_snapshot(
            project_root,
            source_commit,
            rust_snapshot,
            RUST_ENDPOINT_SOURCE_PATHS,
        )
        host_source_rows = _materialize_host_replay_snapshot(
            project_root,
            source_commit,
            host_replay_snapshot,
        )
        if (
            len(python_source_rows) + len(rust_source_rows)
            != EXECUTION_CLOSURE_ROW_COUNT
            or len(host_source_rows) != 1
        ):
            _fail("FAIL_FULL_NODE6_SUPERVISOR_SOURCE", "source closure cardinality differs")
        _publish_run_file(
            run_root / "source-closure-receipt-v1.json",
            _canonical(
                {
                    "execution_closure_row_count": EXECUTION_CLOSURE_ROW_COUNT,
                    "host_replay": list(host_source_rows),
                    "host_replay_row_count": 1,
                    "python_endpoint": list(python_source_rows),
                    "python_endpoint_row_count": len(PYTHON_ENDPOINT_SOURCE_PATHS),
                    "rust_release_build": list(rust_source_rows),
                    "rust_release_build_row_count": len(RUST_ENDPOINT_SOURCE_PATHS),
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-source-closure-receipt/1"
                    ),
                    "source_commit": source_commit,
                    "target_blind_exact_manifests": True,
                }
            ),
        )
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="source-closure-receipt-v1.json",
            decision="THREE_TARGET_BLIND_EXACT_COMMIT_BLOB_MANIFESTS",
        )

        active_phase = "SECCOMP_PROFILE_PUBLICATION"
        trajectory.emit(active_phase, "STARTED")
        seccomp_profile = run_root / "seccomp-profile-v1.json"
        _prepare_seccomp_profile(project_root, seccomp_profile)
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="seccomp-profile-v1.json",
            decision=SECCOMP_SHA256,
        )

        active_phase = "ADMISSION_PUBLICATION"
        trajectory.emit(active_phase, "STARTED")
        admission = run_root / "admission.json"
        _write_admission(project_root, admission, capsule_digest, source_commit)
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="admission.json",
        )

        active_phase = "RUST_OFFLINE_BUILD"
        trajectory.emit(
            active_phase,
            "STARTED",
            command_artifact="rust-build-command-v1.json",
        )
        rust_binary = _build_rust_binary(
            rust_snapshot,
            run_root / "rust-build",
            run_root / "cargo-home",
        )
        trajectory.emit(
            active_phase,
            "COMPLETED",
            command_artifact="rust-build-return-v1.json",
            return_code=0,
        )

        active_phase = "SNAPSHOT_READONLY"
        trajectory.emit(active_phase, "STARTED")
        _make_tree_readonly(python_snapshot)
        _make_tree_readonly(rust_snapshot)
        _make_tree_readonly(host_replay_snapshot)
        _verify_snapshot_manifest_v1(python_snapshot, python_source_rows)
        _verify_snapshot_manifest_v1(rust_snapshot, rust_source_rows)
        _verify_snapshot_manifest_v1(host_replay_snapshot, host_source_rows)
        rust_binary.chmod(0o555)
        trajectory.emit(active_phase, "COMPLETED")

        active_phase = "MEASUREMENT_AND_REPLAY"
        trajectory.emit(active_phase, "STARTED")
        result = run_supervisor_v1(
            SupervisorRunSpecV1(
                capsule_digest=capsule_digest,
                source_commit=source_commit,
                python_snapshot=python_snapshot,
                rust_snapshot=rust_snapshot,
                host_replay_snapshot=host_replay_snapshot,
                rust_binary=rust_binary,
                admission_json=admission,
                seccomp_profile=seccomp_profile,
                run_root=run_root,
                trajectory_root=trajectory_root,
            ),
            check_local_images=False,
            trajectory=trajectory,
        )
        trajectory.emit(
            active_phase,
            "COMPLETED",
            decision=str(result["terminal_status"]),
        )
        result = {
            **result,
            "preparation": {
                "measurement_interval_includes_preparation": False,
                "preparation_completed_before_endpoint_launch": True,
                "rust_build": "OFFLINE_LOCKED_COMPLETED_BEFORE_MEASUREMENT",
                "source_closure_receipt": os.fspath(
                    run_root / "source-closure-receipt-v1.json"
                ),
                "source_snapshots": (
                    "PYTHON_34_RUST_RELEASE_21_HOST_REPLAY_1_DISTINCT_"
                    "READ_ONLY_EXACT_COMMIT_BLOB_MANIFESTS"
                ),
            },
            "trajectory": {
                "event_log_path": os.fspath(trajectory.event_path),
                "terminal_path": os.fspath(trajectory.terminal_path),
            },
        }

        active_phase = "ARTIFACT_PUBLICATION"
        trajectory.emit(active_phase, "STARTED")
        artifact_sha256, artifact_bytes = _publish_artifact(artifact, result)
        trajectory.emit(
            active_phase,
            "COMPLETED",
            detail=f"artifact_bytes={artifact_bytes}",
            decision="ARTIFACT_PUBLISHED_NO_REPLACE",
        )
        terminal = trajectory.finish(str(result["terminal_status"]))
        return {
            "artifact_bytes": artifact_bytes,
            "artifact_path": os.fspath(artifact),
            "artifact_sha256": artifact_sha256,
            "capsule_digest": capsule_digest,
            "host_replay_passed": result["host_replay_passed"],
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_output_slots": [None] * 8,
            "q1_state": "NOT_RUN",
            "schema_version": "hegel-phase3a-q1-full-node6-supervisor-return/1",
            "source_commit": source_commit,
            "terminal_status": result["terminal_status"],
            "trajectory_event_count": terminal["event_count"],
            "trajectory_event_log_path": os.fspath(trajectory.event_path),
            "trajectory_terminal_path": os.fspath(trajectory.terminal_path),
        }
    except FullNode6SupervisorError as error:
        if not trajectory.closed:
            trajectory.emit(active_phase, "FAILED", detail=str(error))
            trajectory.finish(
                "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME",
                error_code=error.code,
                error_detail=error.detail,
            )
        raise
    except Exception as error:
        code = "FAIL_FULL_NODE6_SUPERVISOR_UNEXPECTED"
        detail = f"{type(error).__name__}: {error}"
        if not trajectory.closed:
            trajectory.emit(active_phase, "FAILED", detail=detail)
            trajectory.finish(
                "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME",
                error_code=code,
                error_detail=detail,
            )
        raise FullNode6SupervisorError(code, detail) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--capsule-digest", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--python-snapshot", type=Path)
    parser.add_argument("--rust-snapshot", type=Path)
    parser.add_argument("--host-replay-snapshot", type=Path)
    parser.add_argument("--rust-binary", type=Path)
    parser.add_argument("--admission-json", type=Path)
    parser.add_argument("--seccomp-profile", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    return parser


def _terminal_exit_code_v1(terminal_status: object) -> int:
    return 3 if terminal_status == INFEASIBLE_STATUS else 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.run:
        _fail("REJECT_FULL_NODE6_SUPERVISOR_MODE", "--run is required")
    public = (arguments.project_root, arguments.artifact)
    internal = (
        arguments.python_snapshot,
        arguments.rust_snapshot,
        arguments.host_replay_snapshot,
        arguments.rust_binary,
        arguments.admission_json,
        arguments.seccomp_profile,
    )
    if all(value is not None for value in public) and all(
        value is None for value in internal
    ):
        result = run_public_supervisor_v1(
            project_root=arguments.project_root,
            source_commit=arguments.source_commit,
            capsule_digest=arguments.capsule_digest,
            artifact=arguments.artifact,
            run_root=arguments.run_root,
            trajectory_root=arguments.trajectory_root,
        )
        sys.stdout.buffer.write(_canonical(result))
        return _terminal_exit_code_v1(result.get("terminal_status"))
    if any(value is not None for value in public) or any(
        value is None for value in internal
    ):
        _fail("REJECT_FULL_NODE6_SUPERVISOR_MODE", "supervisor argument registry differs")
    spec = SupervisorRunSpecV1(
        capsule_digest=arguments.capsule_digest,
        source_commit=arguments.source_commit,
        python_snapshot=arguments.python_snapshot,
        rust_snapshot=arguments.rust_snapshot,
        host_replay_snapshot=arguments.host_replay_snapshot,
        rust_binary=arguments.rust_binary,
        admission_json=arguments.admission_json,
        seccomp_profile=arguments.seccomp_profile,
        run_root=arguments.run_root,
        trajectory_root=arguments.trajectory_root,
    )
    result = run_supervisor_v1(spec)
    sys.stdout.buffer.write(_canonical(result))
    return _terminal_exit_code_v1(result.get("terminal_status"))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FullNode6SupervisorError as error:
        sys.stdout.buffer.write(
            _canonical(
                {
                    "error_code": error.code,
                    "error_detail": error.detail,
                    "q1_gate_count": 0,
                    "q1_gate_mask": 0,
                    "q1_output_slots": [None] * 8,
                    "q1_state": "NOT_RUN",
                    "schema_version": (
                        "hegel-phase3a-q1-full-node6-supervisor-error/1"
                    ),
                    "terminal_status": (
                        "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME"
                    ),
                }
            )
        )
        sys.stderr.write(str(error) + "\n")
        raise SystemExit(1)
