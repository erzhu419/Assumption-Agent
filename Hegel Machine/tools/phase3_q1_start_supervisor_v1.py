#!/usr/bin/env python3
"""Durable, target-blind Q1 formal-materialization start supervisor.

The supervisor launches exactly three isolated actors in the fixed order
Python -> Rust -> host replay.  A durable intent precedes every actor Popen.
Completed actors are replayed from their output/checkpoint and are never run a
second time.  Infrastructure failures before the next intent are resumable
PENDING; anything uncertain after an intent is terminal no-resume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import importlib
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
import types
from typing import Callable, Final, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Final = PROJECT_ROOT / "src" / "hegel_machine"
RUN_SPEC_SCHEMA: Final = "hegel-phase3a-q1-start-supervisor-run-spec/1"
INTENT_SCHEMA: Final = "hegel-phase3a-q1-start-actor-intent/1"
CHECKPOINT_SCHEMA: Final = "hegel-phase3a-q1-start-clean-checkpoint/1"
RUNTIME_EVIDENCE_SCHEMA: Final = "hegel-phase3a-q1-start-runtime-evidence/1"
GATE18_INPUT_SCHEMA: Final = "hegel-phase3a-q1-start-gate18-runtime-inputs/1"
OUTCOME_SCHEMA: Final = "hegel-phase3a-q1-start-supervisor-outcome/1"
PENDING_STATUS: Final = "Q1_START_PENDING_BEFORE_NEXT_ACTOR_INTENT"
TERMINAL_STATUS: Final = "Q1_START_FAILED_TERMINAL_NO_RESUME"
SUCCESS_STATUS: Final = "Q1_START_FORMAL_ACTORS_COMPLETE_GATES_15_18_INPUTS_READY"

MATERIALIZER_ACTION: Final = "materialize-full-node6-formal-archive-v1"
MATERIALIZER_STATUS: Final = "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN"
MATERIALIZER_SCHEMA: Final = "hegel-phase3a-q1-full-node6-formal-archive-materializer/1"
HOST_ACTION: Final = "replay-full-node6-formal-archive-v1"
HOST_STATUS: Final = "DUAL_FORMAL_ARCHIVE_EXACT_AGREEMENT"
HOST_SCHEMA: Final = "hegel-q1-formal-host-replay-result/1"
PRESTART_STATUS: Final = "Q1_PRESTART_14_OF_20_APPLIED_Q1_NOT_RUN"

DOCKER: Final = "/usr/bin/docker"
CONTAINER_USER: Final = "65534:65534"
PIDS_LIMIT: Final = 128
NOFILE_LIMIT: Final = 256
TMPFS_SPEC: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777"
PYTHON_ENTRYPOINT: Final = "tools/phase3_q1_full_node6_formal_materializer_entrypoint_v1.py"
HOST_ENTRYPOINT: Final = "tools/phase3_q1_formal_host_replay_v1.py"
RUST_CONTAINER_BINARY: Final = "/actor/hegel-q1-formal-archive-materializer"
ACTOR_RETURN_PREFIX: Final = ".q1-start-actor-return-"
ACTOR_RETURN_SUFFIX: Final = "-v1"
ACTOR_RELEASE: Final = ".q1-start-supervisor-release-v1"

ENDPOINT_MEMORY_BYTES: Final = 2_147_483_648
ENDPOINT_OUTPUT_BYTES: Final = 16_777_216
ENDPOINT_SCRATCH_BYTES: Final = 4_194_304
ENDPOINT_WALL_SECONDS: Final = 3_600
HOST_MEMORY_BYTES: Final = 67_108_864
HOST_OUTPUT_BYTES: Final = 2_097_152
HOST_SCRATCH_BYTES: Final = 1
HOST_WALL_SECONDS: Final = 3_600

STAGES: Final = ("python", "rust", "host")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PINNED_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_CPUSET = re.compile(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*\Z")

_HOLD_SCRIPT: Final = (
    'status=0\n'
    '"$@" || status=$?\n'
    'umask 022\n'
    ': > "/scratch/' + ACTOR_RETURN_PREFIX + '${status}' + ACTOR_RETURN_SUFFIX + '"\n'
    'while [ ! -e /scratch/' + ACTOR_RELEASE + ' ]; do sleep 1; done\n'
    'exit "$status"\n'
)

_SOURCE_CLOSURE_DOMAIN: Final = b"HEGEL/Q1/ACTOR_SOURCE_CLOSURE/V1\0"
_FORMAL_FILE_CLOSURE_DOMAIN: Final = b"HEGEL/Q1/FORMAL_FILE_CLOSURE/V1\0"
_OUTPUT_TREE_CLOSURE_DOMAIN: Final = b"HEGEL/Q1/OUTPUT_TREE_CLOSURE/V1\0"
_MOUNT_REGISTRY_DOMAIN: Final = b"HEGEL/Q1/MOUNT_REGISTRY/V1\0"
_SCRATCH_ABSENCE_DOMAIN: Final = b"HEGEL/Q1/SCRATCH_ABSENCE/V1\0"
_GUARD_NAMES: Final = (
    "RAW_OPERATOR_APPLICATIONS",
    "BEHAVIOR_CLASSES",
    "VISIBLE_FRONTIER_TOTAL",
    "VISIBLE_FRONTIER_PER_CLASS",
    "CONTINUATION_BANK_TOTAL",
    "CONTINUATION_BANK_PER_CLASS",
    "WORK_QUEUE_POINTS",
    "SATURATION_ROUNDS",
)
_FORMAL_GUARD_LIMITS: Final = {
    1: (32_768, 1_024, 4_096, 32, 8_192, 4_096, 16_384, 7),
    2: (32_768, 256, 1_024, 256, 8_192, 4_096, 16_384, 7),
}
_FORMAL_LIMIT_FIELDS: Final = (
    "input_signature_id",
    "maximum_ast_depth",
    "maximum_ast_node_count",
    "maximum_raw_operator_applications",
    "maximum_behavior_classes",
    "maximum_visible_frontier_points",
    "maximum_visible_frontier_points_per_class",
    "maximum_continuation_bank_points",
    "maximum_continuation_bank_points_per_class",
    "maximum_work_queue_points",
    "maximum_saturation_rounds",
    "maximum_wall_time_seconds",
)
_FORMAL_LIMIT_ROWS: Final = tuple(
    dict(
        zip(
            _FORMAL_LIMIT_FIELDS,
            (signature_id, 3, 6, *limits, 3_600),
            strict=True,
        )
    )
    for signature_id, limits in _FORMAL_GUARD_LIMITS.items()
)
_ENDPOINT_RESULT_KEYS: Final = {
    "action_id", "actor_id", "actual_formal_peak_scratch_bytes",
    "actual_formal_terminal_live_scratch_bytes", "capsule_digest",
    "closure_bundle_root", "closure_semantics_version", "diagnostic_only",
    "dsl_version", "formal_archive_materialized", "formal_archive_payload_bytes",
    "formal_capacity_limits_by_signature", "formal_output_bytes_limit",
    "formal_scratch_bytes_limit", "full_v16_structural_limits_applied",
    "implementation_id", "maximum_ast_depth", "maximum_ast_node_count",
    "partitions", "prestart_admission_receipt_root", "q1_gate_count",
    "q1_gate_mask", "q1_gate_total", "q1_output_slots", "q1_state",
    "role_evaluation_performed", "run_genesis_root", "schema_version",
    "source_commit", "split_accessed", "status", "target_truth_accessed",
}


class Q1StartSupervisorError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class PendingCondition(Q1StartSupervisorError):
    """A non-core condition detected before the next durable intent."""


class CompletedCheckpointBindingPending(PendingCondition):
    """A temporary live-binding failure after exact checkpoint replay."""


class TerminalCondition(Q1StartSupervisorError):
    """A consumed/uncertain actor state that must never be resumed."""


def _reject(code: str, detail: str) -> NoReturn:
    raise Q1StartSupervisorError(code, detail)


def _terminal(code: str, detail: str) -> NoReturn:
    raise TerminalCondition(code, detail)


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
        _reject("REJECT_Q1_START_CANONICAL_JSON", str(error))


def _strict_object(payload: bytes, label: str, *, maximum: int = 16 * 1024 * 1024) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _reject("REJECT_Q1_START_JSON", f"{label} byte bound/type differs")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite token {value}")

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _reject("REJECT_Q1_START_JSON", f"{label}: {error}")
    if type(value) is not dict or _canonical(value) != payload:
        _reject("REJECT_Q1_START_JSON", f"{label} is not one strict canonical object")
    return value


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _reject("REJECT_Q1_START_SCHEMA", f"{label} key closure differs")


def _path(value: object, label: str) -> Path:
    if type(value) is not str or not value.startswith("/") or "," in value:
        _reject("REJECT_Q1_START_SPEC", f"{label} must be an absolute mount-safe path")
    candidate = Path(value)
    if os.path.normpath(value) != value:
        _reject("REJECT_Q1_START_SPEC", f"{label} is not canonical")
    return candidate


def _uint(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _reject("REJECT_Q1_START_SPEC", f"{label} must be an unsigned integer")
    return value


def _cpuset_members(value: str) -> frozenset[int]:
    if _CPUSET.fullmatch(value) is None:
        _reject("REJECT_Q1_START_SPEC", "cpuset syntax differs")
    members: set[int] = set()
    previous = -1
    for item in value.split(","):
        if "-" in item:
            left_text, right_text = item.split("-", 1)
            left, right = int(left_text), int(right_text)
            if left >= right:
                _reject("REJECT_Q1_START_SPEC", "cpuset range differs")
            values = range(left, right + 1)
        else:
            values = (int(item),)
        for member in values:
            if member <= previous or member in members:
                _reject("REJECT_Q1_START_SPEC", "cpuset is not strictly ordered")
            members.add(member)
            previous = member
    return frozenset(members)


@dataclass(frozen=True, slots=True)
class RuntimeLimitV1:
    cpuset_cpus: str
    memory_bytes: int
    output_bytes: int
    scratch_bytes: int
    stdout_bytes: int
    wall_time_seconds: int

    def object(self) -> dict[str, object]:
        return {
            "cpuset_cpus": self.cpuset_cpus,
            "memory_bytes": self.memory_bytes,
            "output_bytes": self.output_bytes,
            "scratch_bytes": self.scratch_bytes,
            "stdout_bytes": self.stdout_bytes,
            "wall_time_seconds": self.wall_time_seconds,
        }


@dataclass(frozen=True, slots=True)
class SourceManifestRowV1:
    relative_path: str
    byte_count: int
    raw_sha256: str

    def object(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "raw_sha256": self.raw_sha256,
        }


@dataclass(frozen=True, slots=True)
class Q1StartRunSpecV1:
    capsule_digest: str
    prestart_capsule_digest: str
    source_commit: str
    run_genesis_root: str
    prestart_admission_receipt_root: str
    prestart_artifact: Path
    python_image: str
    rust_image: str
    rust_binary_raw_sha256: str
    seccomp_profile_raw_sha256: str
    python_admission_raw_sha256: str
    actor_implementation_roots: Mapping[str, str]
    python_source_root: Path
    rust_source_root: Path
    host_source_root: Path
    rust_binary: Path
    python_admission_json: Path
    seccomp_profile: Path
    run_root: Path
    checkpoint_root: Path
    python_output: Path
    python_scratch: Path
    rust_output: Path
    rust_scratch: Path
    host_output: Path
    host_scratch: Path
    runtime_limits: Mapping[str, RuntimeLimitV1]
    actor_source_manifests: Mapping[str, tuple[SourceManifestRowV1, ...]]

    def binding(self) -> dict[str, object]:
        return {
            "capsule_digest": self.capsule_digest,
            "prestart_admission_receipt_root": self.prestart_admission_receipt_root,
            "run_genesis_root": self.run_genesis_root,
            "source_commit": self.source_commit,
        }

    def output_for(self, stage: str) -> Path:
        return getattr(self, f"{stage}_output")

    def scratch_for(self, stage: str) -> Path:
        return getattr(self, f"{stage}_scratch")

    def source_for(self, stage: str) -> Path:
        return self.host_source_root if stage == "host" else getattr(self, f"{stage}_source_root")

    def image_for(self, stage: str) -> str:
        return self.rust_image if stage == "rust" else self.python_image


@dataclass(frozen=True, slots=True)
class ActorProcessResultV1:
    return_code: int | None
    stdout: bytes
    stderr: bytes
    outcome_known: bool
    runtime_evidence: Mapping[str, object]


RunLauncher = Callable[[str, Sequence[str], Q1StartRunSpecV1], ActorProcessResultV1]
StagePreflight = Callable[[str, Q1StartRunSpecV1], None]
EndpointReplayer = Callable[[str, Path, Q1StartRunSpecV1], Mapping[str, object]]


_RUN_SPEC_KEYS = {
    "schema_version",
    "capsule_digest",
    "prestart_capsule_digest",
    "source_commit",
    "run_genesis_root",
    "prestart_admission_receipt_root",
    "prestart_artifact",
    "python_image",
    "rust_image",
    "rust_binary_raw_sha256",
    "seccomp_profile_raw_sha256",
    "python_admission_raw_sha256",
    "actor_implementation_roots",
    "python_source_root",
    "rust_source_root",
    "host_source_root",
    "rust_binary",
    "python_admission_json",
    "seccomp_profile",
    "run_root",
    "checkpoint_root",
    "python_output",
    "python_scratch",
    "rust_output",
    "rust_scratch",
    "host_output",
    "host_scratch",
    "runtime_limits",
    "actor_source_manifests",
}
_LIMIT_KEYS = {
    "cpuset_cpus",
    "memory_bytes",
    "output_bytes",
    "scratch_bytes",
    "stdout_bytes",
    "wall_time_seconds",
}
_SOURCE_MANIFEST_ROW_KEYS = {"relative_path", "byte_count", "raw_sha256"}


def _load_source_manifest(value: object, stage: str) -> tuple[SourceManifestRowV1, ...]:
    if type(value) is not list or not value:
        _reject("REJECT_Q1_START_SPEC", f"{stage} source manifest must be nonempty")
    rows: list[SourceManifestRowV1] = []
    previous = ""
    for index, item in enumerate(value):
        if type(item) is not dict:
            _reject("REJECT_Q1_START_SPEC", f"{stage} source row {index} is not an object")
        _keys(item, _SOURCE_MANIFEST_ROW_KEYS, f"{stage} source row {index}")
        relative = item["relative_path"]
        raw = item["raw_sha256"]
        if (
            type(relative) is not str
            or not relative
            or relative.startswith("/")
            or Path(relative).as_posix() != relative
            or any(part in ("", ".", "..") for part in relative.split("/"))
            or relative <= previous
        ):
            _reject("REJECT_Q1_START_SPEC", f"{stage} source row order/path differs")
        if type(raw) is not str or _DIGEST.fullmatch(raw) is None:
            _reject("REJECT_Q1_START_SPEC", f"{stage} source row digest differs")
        rows.append(SourceManifestRowV1(relative, _uint(item["byte_count"], "source byte_count"), raw))
        previous = relative
    return tuple(rows)


def _load_limit(value: object, stage: str) -> RuntimeLimitV1:
    if type(value) is not dict:
        _reject("REJECT_Q1_START_SPEC", f"{stage} runtime limit is not an object")
    _keys(value, _LIMIT_KEYS, f"{stage} runtime limit")
    cpuset = value["cpuset_cpus"]
    if type(cpuset) is not str:
        _reject("REJECT_Q1_START_SPEC", f"{stage} cpuset type differs")
    _cpuset_members(cpuset)
    limit = RuntimeLimitV1(
        cpuset,
        _uint(value["memory_bytes"], f"{stage}.memory_bytes"),
        _uint(value["output_bytes"], f"{stage}.output_bytes"),
        _uint(value["scratch_bytes"], f"{stage}.scratch_bytes"),
        _uint(value["stdout_bytes"], f"{stage}.stdout_bytes"),
        _uint(value["wall_time_seconds"], f"{stage}.wall_time_seconds"),
    )
    expected = (
        (ENDPOINT_MEMORY_BYTES, ENDPOINT_OUTPUT_BYTES, ENDPOINT_SCRATCH_BYTES, ENDPOINT_OUTPUT_BYTES, ENDPOINT_WALL_SECONDS)
        if stage != "host"
        else (HOST_MEMORY_BYTES, HOST_OUTPUT_BYTES, HOST_SCRATCH_BYTES, HOST_OUTPUT_BYTES, HOST_WALL_SECONDS)
    )
    if (
        limit.memory_bytes,
        limit.output_bytes,
        limit.scratch_bytes,
        limit.stdout_bytes,
        limit.wall_time_seconds,
    ) != expected:
        _reject("REJECT_Q1_START_SPEC", f"{stage} frozen resource envelope differs")
    return limit


def load_run_spec_v1(payload: bytes) -> Q1StartRunSpecV1:
    value = _strict_object(payload, "run spec", maximum=1_048_576)
    _keys(value, _RUN_SPEC_KEYS, "run spec")
    if value["schema_version"] != RUN_SPEC_SCHEMA:
        _reject("REJECT_Q1_START_SPEC", "run spec version differs")
    for name in (
        "capsule_digest",
        "prestart_capsule_digest",
        "run_genesis_root",
        "prestart_admission_receipt_root",
        "rust_binary_raw_sha256",
        "seccomp_profile_raw_sha256",
        "python_admission_raw_sha256",
    ):
        if type(value[name]) is not str or _DIGEST.fullmatch(value[name]) is None:
            _reject("REJECT_Q1_START_SPEC", f"{name} differs")
    if value["capsule_digest"] == value["prestart_capsule_digest"]:
        _reject("REJECT_Q1_START_SPEC", "Q1 start must use a capsule distinct from prestart")
    if type(value["source_commit"]) is not str or _COMMIT.fullmatch(value["source_commit"]) is None:
        _reject("REJECT_Q1_START_SPEC", "source_commit differs")
    for name in ("python_image", "rust_image"):
        if type(value[name]) is not str or _PINNED_IMAGE.fullmatch(value[name]) is None:
            _reject("REJECT_Q1_START_SPEC", f"{name} is not digest-pinned")
    limits_value = value["runtime_limits"]
    if type(limits_value) is not dict or set(limits_value) != set(STAGES):
        _reject("REJECT_Q1_START_SPEC", "runtime limit actor registry differs")
    limits = {stage: _load_limit(limits_value[stage], stage) for stage in STAGES}
    manifests_value = value["actor_source_manifests"]
    if type(manifests_value) is not dict or set(manifests_value) != set(STAGES):
        _reject("REJECT_Q1_START_SPEC", "actor source manifest registry differs")
    manifests = {
        stage: _load_source_manifest(manifests_value[stage], stage)
        for stage in STAGES
    }
    implementation_roots = value["actor_implementation_roots"]
    if (
        type(implementation_roots) is not dict
        or set(implementation_roots) != set(STAGES)
        or any(
            type(root) is not str or _DIGEST.fullmatch(root) is None
            for root in implementation_roots.values()
        )
    ):
        _reject("REJECT_Q1_START_SPEC", "actor implementation root registry differs")
    paths = {
        name: _path(value[name], name)
        for name in (
            "prestart_artifact",
            "python_source_root",
            "rust_source_root",
            "host_source_root",
            "rust_binary",
            "python_admission_json",
            "seccomp_profile",
            "run_root",
            "checkpoint_root",
            "python_output",
            "python_scratch",
            "rust_output",
            "rust_scratch",
            "host_output",
            "host_scratch",
        )
    }
    child_names = (
        "checkpoint_root",
        "python_source_root",
        "rust_source_root",
        "host_source_root",
        "python_output",
        "python_scratch",
        "rust_output",
        "rust_scratch",
        "host_output",
        "host_scratch",
    )
    if any(paths[name].parent != paths["run_root"] for name in child_names):
        _reject("REJECT_Q1_START_SPEC", "run-owned paths must be direct run-root children")
    if len({paths[name] for name in child_names}) != len(child_names):
        _reject("REJECT_Q1_START_SPEC", "run-owned paths overlap")
    return Q1StartRunSpecV1(
        value["capsule_digest"],
        value["prestart_capsule_digest"],
        value["source_commit"],
        value["run_genesis_root"],
        value["prestart_admission_receipt_root"],
        paths["prestart_artifact"],
        value["python_image"],
        value["rust_image"],
        value["rust_binary_raw_sha256"],
        value["seccomp_profile_raw_sha256"],
        value["python_admission_raw_sha256"],
        dict(implementation_roots),
        paths["python_source_root"],
        paths["rust_source_root"],
        paths["host_source_root"],
        paths["rust_binary"],
        paths["python_admission_json"],
        paths["seccomp_profile"],
        paths["run_root"],
        paths["checkpoint_root"],
        paths["python_output"],
        paths["python_scratch"],
        paths["rust_output"],
        paths["rust_scratch"],
        paths["host_output"],
        paths["host_scratch"],
        limits,
        manifests,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical(dict(value))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    _fsync_directory(path.parent)


def _read_canonical_file(path: Path, label: str, *, maximum: int = 16 * 1024 * 1024) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        _reject("REJECT_Q1_START_EVIDENCE", f"{label}: {error}")
    return _strict_object(payload, label, maximum=maximum)


def _tagged_raw_sha256(path: Path) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    with path.open("rb", buffering=0) as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return "sha256:" + digest.hexdigest(), byte_count


def _object_root(domain: bytes, value: object) -> str:
    digest = sha256()
    digest.update(domain)
    digest.update(_canonical(value))
    return "sha256:" + digest.hexdigest()


def _replay_bound_regular(path: Path, expected_raw_sha256: str, label: str) -> dict[str, object]:
    _existing_regular(path, label)
    try:
        raw, byte_count = _tagged_raw_sha256(path)
    except OSError as error:
        raise PendingCondition("Q1_START_BOUND_FILE_PENDING", f"{label}: {error}") from error
    if raw != expected_raw_sha256:
        raise PendingCondition("Q1_START_BOUND_FILE_PENDING", f"{label} bytes differ")
    return {"path": str(path), "byte_count": byte_count, "raw_sha256": raw}


def _replay_source_manifest(stage: str, spec: Q1StartRunSpecV1) -> dict[str, object]:
    root = spec.source_for(stage)
    _existing_directory(root, f"{stage} source root")
    expected = spec.actor_source_manifests[stage]
    expected_paths = tuple(row.relative_path for row in expected)
    observed_paths: list[str] = []
    try:
        for path in root.rglob("*"):
            status = path.lstat()
            if path.is_symlink() or not (
                stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)
            ):
                raise PendingCondition(
                    "Q1_START_SOURCE_MANIFEST_PENDING",
                    f"{stage} source contains unsupported entry {path}",
                )
            if stat.S_ISREG(status.st_mode):
                observed_paths.append(path.relative_to(root).as_posix())
    except OSError as error:
        raise PendingCondition("Q1_START_SOURCE_MANIFEST_PENDING", f"{stage}: {error}") from error
    observed_paths.sort()
    if tuple(observed_paths) != expected_paths:
        raise PendingCondition(
            "Q1_START_SOURCE_MANIFEST_PENDING",
            f"{stage} source file registry differs",
        )
    replayed: list[dict[str, object]] = []
    for row in expected:
        path = root / row.relative_path
        try:
            status = path.lstat()
            raw, byte_count = _tagged_raw_sha256(path)
        except OSError as error:
            raise PendingCondition(
                "Q1_START_SOURCE_MANIFEST_PENDING",
                f"{stage}:{row.relative_path}:{error}",
            ) from error
        if (
            path.is_symlink()
            or not stat.S_ISREG(status.st_mode)
            or byte_count != row.byte_count
            or raw != row.raw_sha256
        ):
            raise PendingCondition(
                "Q1_START_SOURCE_MANIFEST_PENDING",
                f"{stage}:{row.relative_path}:byte replay differs",
            )
        replayed.append(row.object())
    return {
        "source_root": str(root),
        "file_count": len(replayed),
        "byte_count": sum(row.byte_count for row in expected),
        "source_closure_root": _object_root(_SOURCE_CLOSURE_DOMAIN, replayed),
    }


def _replay_stage_bindings(stage: str, spec: Q1StartRunSpecV1) -> dict[str, object]:
    evidence: dict[str, object] = {
        "actor_implementation_root": spec.actor_implementation_roots[stage],
        "actor_source_manifest": _replay_source_manifest(stage, spec),
        "seccomp_profile": _replay_bound_regular(
            spec.seccomp_profile,
            spec.seccomp_profile_raw_sha256,
            "seccomp profile",
        ),
        "rust_binary": None,
        "python_admission": None,
    }
    if stage == "python":
        evidence["python_admission"] = _replay_bound_regular(
            spec.python_admission_json,
            spec.python_admission_raw_sha256,
            "Python admission JSON",
        )
    if stage == "rust":
        evidence["rust_binary"] = _replay_bound_regular(
            spec.rust_binary,
            spec.rust_binary_raw_sha256,
            "Rust formal binary",
        )
    evidence["mount_registry_root"] = _object_root(
        _MOUNT_REGISTRY_DOMAIN,
        _expected_mounts(spec, stage),
    )
    return evidence


def _relaxed_json_object(payload: bytes, label: str) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite token {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _reject("REJECT_Q1_START_JSON", f"{label}: {error}")
    if type(value) is not dict:
        _reject("REJECT_Q1_START_JSON", f"{label} must be one object")
    return value


def _validate_prestart(spec: Q1StartRunSpecV1) -> None:
    try:
        payload = spec.prestart_artifact.read_bytes()
    except OSError as error:
        raise PendingCondition("Q1_START_PREDECESSOR_PENDING", str(error)) from error
    value = _strict_object(payload, "prestart authoritative amendment")
    expected = {
        "capsule_digest": spec.prestart_capsule_digest,
        "source_commit": spec.source_commit,
        "run_genesis_root": spec.run_genesis_root,
        "prestart_admission_receipt_root": spec.prestart_admission_receipt_root,
        "status": PRESTART_STATUS,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q1_execution_started": False,
        "future_q1_start_requires_distinct_authorization": True,
        "actor_implementation_roots": dict(spec.actor_implementation_roots),
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        _reject("REJECT_Q1_START_PRESTART_REPLAY", "authoritative 14-of-20 binding differs")


def _existing_regular(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PendingCondition("Q1_START_PATH_PENDING", f"{label}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(status.st_mode) or resolved != path:
        _reject("REJECT_Q1_START_PATH", f"{label} identity differs")
    if executable and status.st_mode & 0o111 == 0:
        raise PendingCondition("Q1_START_PATH_PENDING", f"{label} is not executable")


def _existing_directory(path: Path, label: str) -> None:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PendingCondition("Q1_START_PATH_PENDING", f"{label}: {error}") from error
    if path.is_symlink() or not stat.S_ISDIR(status.st_mode) or resolved != path:
        _reject("REJECT_Q1_START_PATH", f"{label} identity differs")


def _local_image(image: str) -> None:
    try:
        completed = subprocess.run(
            [DOCKER, "image", "inspect", image],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PendingCondition("Q1_START_LOCAL_IMAGE_PENDING", str(error)) from error
    if completed.returncode != 0:
        raise PendingCondition(
            "Q1_START_LOCAL_IMAGE_PENDING",
            completed.stderr.decode("utf-8", "replace")[:512],
        )


def default_preflight_v1(stage: str, spec: Q1StartRunSpecV1) -> None:
    """Only non-measurement checks; every failure occurs before stage intent."""

    _existing_directory(spec.run_root, "run_root")
    _existing_directory(spec.source_for(stage), f"{stage} source root")
    _existing_regular(spec.seccomp_profile, "seccomp profile")
    _relaxed_json_object(spec.seccomp_profile.read_bytes(), "seccomp profile")
    if stage == "python":
        _existing_regular(spec.python_admission_json, "Python admission JSON")
        admission = _read_canonical_file(spec.python_admission_json, "Python admission JSON")
        for name, expected in spec.binding().items():
            if admission.get(name) != expected:
                _reject("REJECT_Q1_START_ADMISSION", f"{name} differs")
        if (
            admission.get("action_id") != MATERIALIZER_ACTION
            or admission.get("q1_state") != "NOT_RUN"
            or admission.get("q1_gate_count") != 14
            or admission.get("q1_gate_total") != 20
            or admission.get("q1_gate_mask") != 16_383
            or admission.get("q1_output_slots") != [None] * 8
            or admission.get("diagnostic_only") is not False
            or admission.get("target_truth_accessed") is not False
            or admission.get("split_accessed") is not False
            or admission.get("role_evaluation_performed") is not False
        ):
            _reject("REJECT_Q1_START_ADMISSION", "formal 14-of-20 boundary differs")
    if stage == "rust":
        _existing_regular(spec.rust_binary, "Rust formal binary", executable=True)
    try:
        effective = frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        raise PendingCondition("Q1_START_CPUSET_PENDING", str(error)) from error
    required = _cpuset_members(spec.runtime_limits[stage].cpuset_cpus)
    if not required.issubset(effective):
        raise PendingCondition("Q1_START_CPUSET_PENDING", f"{stage} cpuset is not effective")
    limit = spec.runtime_limits[stage]
    if shutil.disk_usage(spec.run_root).free < limit.output_bytes + limit.scratch_bytes + 8 * 1024 * 1024:
        raise PendingCondition("Q1_START_DISK_PENDING", f"{stage} run-root capacity unavailable")
    _local_image(spec.image_for(stage))


def _mount(source: Path, destination: str, *, readonly: bool) -> str:
    value = f"type=bind,src={source},dst={destination}"
    return value + (",readonly" if readonly else "")


def _common_command(spec: Q1StartRunSpecV1, stage: str) -> list[str]:
    limit = spec.runtime_limits[stage]
    suffix = spec.capsule_digest[-12:]
    command = [
        DOCKER,
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        f"hegel-q1-start-{stage}-{suffix}",
        "--user",
        CONTAINER_USER,
        "--cidfile",
        str(spec.checkpoint_root / f"{stage}.cid"),
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
        f"seccomp={spec.seccomp_profile}",
        "--cpuset-cpus",
        limit.cpuset_cpus,
        "--memory",
        str(limit.memory_bytes),
        "--memory-swap",
        str(limit.memory_bytes),
        "--pids-limit",
        str(PIDS_LIMIT),
        "--ulimit",
        f"nofile={NOFILE_LIMIT}:{NOFILE_LIMIT}",
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
        _mount(spec.source_for(stage), "/source", readonly=True),
        "--mount",
        _mount(spec.output_for(stage), "/output", readonly=False),
        "--mount",
        _mount(spec.scratch_for(stage), "/scratch", readonly=False),
    ]
    if stage == "python":
        command.extend(("--mount", _mount(spec.python_admission_json, "/admission/admission.json", readonly=True)))
    elif stage == "rust":
        command.extend(("--mount", _mount(spec.rust_binary, RUST_CONTAINER_BINARY, readonly=True)))
    else:
        command.extend(
            (
                "--mount",
                _mount(spec.python_output, "/python-output", readonly=True),
                "--mount",
                _mount(spec.rust_output, "/rust-output", readonly=True),
            )
        )
    command.extend(("--workdir", "/source", spec.image_for(stage)))
    return command


def _actor_argv(spec: Q1StartRunSpecV1, stage: str) -> list[str]:
    binding = [
        "--capsule-digest",
        spec.capsule_digest,
        "--source-commit",
        spec.source_commit,
        "--run-genesis-root",
        spec.run_genesis_root,
        "--prestart-admission-receipt-root",
        spec.prestart_admission_receipt_root,
    ]
    if stage == "python":
        return [
            "/usr/local/bin/python3",
            "-I",
            "-S",
            "-B",
            f"/source/{PYTHON_ENTRYPOINT}",
            "--action",
            MATERIALIZER_ACTION,
            "--admission-json",
            "/admission/admission.json",
            *binding,
            "--output-dir",
            "/output",
            "--scratch-dir",
            "/scratch",
        ]
    if stage == "rust":
        return [
            RUST_CONTAINER_BINARY,
            "--action",
            MATERIALIZER_ACTION,
            *binding,
            "--output-dir",
            "/output",
            "--scratch-dir",
            "/scratch",
        ]
    return [
        "/usr/local/bin/python3",
        "-I",
        "-S",
        "-B",
        f"/source/{HOST_ENTRYPOINT}",
        "--python-root",
        "/python-output",
        "--rust-root",
        "/rust-output",
        "--action",
        HOST_ACTION,
        *binding,
    ]


def docker_commands_v1(spec: Q1StartRunSpecV1) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for stage in STAGES:
        command = _common_command(spec, stage)
        command.extend(("/bin/sh", "-c", _HOLD_SCRIPT, "hegel-q1-start-held-actor-v1", *_actor_argv(spec, stage)))
        result[stage] = tuple(command)
    return result


def _expected_mounts(spec: Q1StartRunSpecV1, stage: str) -> list[dict[str, object]]:
    rows = [
        {"destination": "/output", "read_write": True, "source": str(spec.output_for(stage))},
        {"destination": "/scratch", "read_write": True, "source": str(spec.scratch_for(stage))},
        {"destination": "/source", "read_write": False, "source": str(spec.source_for(stage))},
    ]
    if stage == "python":
        rows.append({"destination": "/admission/admission.json", "read_write": False, "source": str(spec.python_admission_json)})
    elif stage == "rust":
        rows.append({"destination": RUST_CONTAINER_BINARY, "read_write": False, "source": str(spec.rust_binary)})
    else:
        rows.extend(
            (
                {"destination": "/python-output", "read_write": False, "source": str(spec.python_output)},
                {"destination": "/rust-output", "read_write": False, "source": str(spec.rust_output)},
            )
        )
    return sorted(rows, key=lambda row: str(row["destination"]))


def _docker_inspect(
    container_id: str,
    spec: Q1StartRunSpecV1,
    stage: str,
    command: Sequence[str],
    actor_deadline: float,
) -> tuple[dict[str, object], int]:
    remaining = actor_deadline - time.monotonic()
    if remaining <= 0:
        _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "actor wall-time exhausted before inspect")
    try:
        completed = subprocess.run(
            [DOCKER, "inspect", container_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(30, remaining),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"docker inspect {stage}: {error}")
    if completed.returncode != 0:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"docker inspect {stage} failed")
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"docker inspect JSON: {error}")
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", "docker inspect cardinality differs")
    row = rows[0]
    state, host, config, mounts = row.get("State"), row.get("HostConfig"), row.get("Config"), row.get("Mounts")
    if not all(type(value) is dict for value in (state, host, config)) or type(mounts) is not list:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", "docker inspect shape differs")
    assert isinstance(state, dict) and isinstance(host, dict) and isinstance(config, dict) and isinstance(mounts, list)
    observed_mounts = sorted(
        (
            {
                "destination": item.get("Destination"),
                "read_write": item.get("RW"),
                "source": item.get("Source"),
            }
            for item in mounts
            if type(item) is dict and item.get("Type") == "bind"
        ),
        key=lambda item: str(item["destination"]),
    )
    limit = spec.runtime_limits[stage]
    required = {
        "NetworkMode": "none",
        "IpcMode": "private",
        "ReadonlyRootfs": True,
        "CpusetCpus": limit.cpuset_cpus,
        "Memory": limit.memory_bytes,
        "MemorySwap": limit.memory_bytes,
        "PidsLimit": PIDS_LIMIT,
        "CapDrop": ["ALL"],
        "Ulimits": [{"Name": "nofile", "Hard": NOFILE_LIMIT, "Soft": NOFILE_LIMIT}],
        "Tmpfs": {"/tmp": TMPFS_SPEC.split(":", 1)[1]},
        "AutoRemove": True,
    }
    mismatches = [name for name, expected in required.items() if host.get(name) != expected]
    if config.get("Image") != spec.image_for(stage):
        mismatches.append("Config.Image")
    if config.get("User") != CONTAINER_USER:
        mismatches.append("Config.User")
    if config.get("WorkingDir") != "/source":
        mismatches.append("Config.WorkingDir")
    image_index = list(command).index(spec.image_for(stage))
    if config.get("Cmd") != list(command)[image_index + 1 :]:
        mismatches.append("Config.Cmd")
    environment = config.get("Env")
    required_environment = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
    }
    if type(environment) is not list:
        mismatches.append("Config.Env")
    else:
        observed_environment: dict[str, list[str]] = {}
        for item in environment:
            if type(item) is str and "=" in item:
                name, item_value = item.split("=", 1)
                observed_environment.setdefault(name, []).append(item_value)
        if any(observed_environment.get(name) != [expected] for name, expected in required_environment.items()):
            mismatches.append("Config.Env")
    if observed_mounts != _expected_mounts(spec, stage):
        mismatches.append("Mounts")
    security = host.get("SecurityOpt")
    seccomp_rows = (
        [item for item in security if type(item) is str and item.startswith("seccomp=")]
        if type(security) is list
        else []
    )
    if type(security) is not list or "no-new-privileges" not in security or len(seccomp_rows) != 1:
        mismatches.append("SecurityOpt")
    else:
        try:
            expected_seccomp = _relaxed_json_object(
                spec.seccomp_profile.read_bytes(),
                "expected seccomp profile",
            )
            observed_seccomp = _relaxed_json_object(
                seccomp_rows[0].split("=", 1)[1].encode("utf-8"),
                "observed seccomp profile",
            )
        except (OSError, Q1StartSupervisorError):
            mismatches.append("SecurityOpt.seccomp")
        else:
            if expected_seccomp != observed_seccomp:
                mismatches.append("SecurityOpt.seccomp")
    if mismatches:
        _terminal("Q1_START_RUNTIME_ISOLATION_MISMATCH", ",".join(mismatches))
    pid = state.get("Pid")
    if type(pid) is not int or pid < 1:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", "container PID differs")
    try:
        cgroup_line = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except OSError as error:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"cgroup binding: {error}")
    match = re.fullmatch(r"0::(/[^\r\n]*)\n", cgroup_line)
    if match is None:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", "cgroup v2 binding differs")
    cgroup_path = "/" + match.group(1).lstrip("/")
    try:
        memory_peak_text = (Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "memory.peak").read_text(encoding="ascii").strip()
    except OSError as error:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"memory.peak: {error}")
    if not memory_peak_text.isdigit():
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", "memory.peak syntax differs")
    return ({"cgroup_path": cgroup_path, "mounts": observed_mounts}, int(memory_peak_text))


def _wait_for_actor_receipt(
    process: subprocess.Popen[bytes],
    scratch: Path,
    deadline: float,
) -> tuple[int, Path]:
    while time.monotonic() < deadline:
        try:
            entries = tuple(scratch.iterdir())
        except OSError as error:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", f"held receipt scan: {error}")
        receipts = [
            path
            for path in entries
            if path.name.startswith(ACTOR_RETURN_PREFIX)
            and path.name.endswith(ACTOR_RETURN_SUFFIX)
        ]
        if receipts:
            if len(entries) != 1 or len(receipts) != 1:
                _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "held receipt registry differs")
            returned = receipts[0]
            code_text = returned.name[
                len(ACTOR_RETURN_PREFIX) : -len(ACTOR_RETURN_SUFFIX)
            ]
            try:
                status = returned.lstat()
            except OSError as error:
                _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", f"held receipt: {error}")
            if (
                not code_text.isascii()
                or not code_text.isdigit()
                or (len(code_text) > 1 and code_text.startswith("0"))
                or int(code_text) > 255
                or not stat.S_ISREG(status.st_mode)
                or status.st_size != 0
            ):
                _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "held return-code receipt differs")
            return int(code_text), returned
        if process.poll() is not None:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "container exited before held receipt")
        time.sleep(0.02)
    _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "actor wall-time exhausted")


def _read_cid(path: Path, process: subprocess.Popen[bytes], actor_deadline: float) -> str:
    cid_deadline = min(time.monotonic() + 30, actor_deadline)
    while time.monotonic() < cid_deadline:
        try:
            value = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            value = ""
        except OSError as error:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", f"cidfile: {error}")
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value
        if process.poll() is not None:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "container exited before cidfile")
        time.sleep(0.02)
    _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "cidfile timed out")


def actual_launcher_v1(stage: str, command: Sequence[str], spec: Q1StartRunSpecV1) -> ActorProcessResultV1:
    """Run one held container and capture isolation plus cgroup memory peak."""

    scratch = spec.scratch_for(stage)
    cidfile = spec.checkpoint_root / f"{stage}.cid"
    started = time.monotonic_ns()
    actor_deadline = started / 1_000_000_000 + spec.runtime_limits[stage].wall_time_seconds
    process: subprocess.Popen[bytes] | None = None
    container_id: str | None = None
    communication: list[object] = []

    def communicate() -> None:
        assert process is not None
        try:
            communication.append(process.communicate())
        except BaseException as error:  # captured as unknown, never retried
            communication.append(error)

    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        thread = threading.Thread(target=communicate, name=f"q1-start-{stage}-pipes", daemon=True)
        thread.start()
        container_id = _read_cid(cidfile, process, actor_deadline)
        actor_code, return_receipt = _wait_for_actor_receipt(
            process,
            scratch,
            actor_deadline,
        )
        actor_returned = time.monotonic_ns()
        inspect_evidence, memory_peak = _docker_inspect(
            container_id,
            spec,
            stage,
            command,
            actor_deadline,
        )
        if time.monotonic() >= actor_deadline:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "actor wall-time exhausted before release")
        release = scratch / ACTOR_RELEASE
        descriptor = os.open(release, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        _fsync_directory(scratch)
        thread.join(max(0.0, actor_deadline - time.monotonic()))
        if thread.is_alive() or not communication or isinstance(communication[0], BaseException):
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "container pipe completion differs")
        assert process.returncode is not None
        if process.returncode != actor_code:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "actor/container return code differs")
        stdout, stderr = communication[0]
        assert isinstance(stdout, bytes) and isinstance(stderr, bytes)
        exited = time.monotonic_ns()
        if exited - started > spec.runtime_limits[stage].wall_time_seconds * 1_000_000_000:
            _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "actor wall-time exhausted before exit")
        return_receipt.unlink()
        (scratch / ACTOR_RELEASE).unlink()
        _fsync_directory(scratch)
        try:
            cidfile.unlink()
            _fsync_directory(cidfile.parent)
        except FileNotFoundError:
            pass
        evidence = {
            "schema_version": RUNTIME_EVIDENCE_SCHEMA,
            "stage": stage,
            "container_id": container_id,
            "image": spec.image_for(stage),
            "cpuset_cpus": spec.runtime_limits[stage].cpuset_cpus,
            "memory_limit_bytes": spec.runtime_limits[stage].memory_bytes,
            "memory_peak_bytes": memory_peak,
            "network_mode": "none",
            "ipc_mode": "private",
            "read_only_rootfs": True,
            "mounts": inspect_evidence["mounts"],
            "cgroup_path": inspect_evidence["cgroup_path"],
            "command": list(command),
            "environment_overrides": {
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TZ": "UTC",
            },
            "cap_drop": ["ALL"],
            "nofile_limit": NOFILE_LIMIT,
            "tmpfs": TMPFS_SPEC,
            "seccomp_profile_raw_sha256": spec.seccomp_profile_raw_sha256,
            "measurement_interval_start_monotonic_ns": started,
            "actor_return_monotonic_ns": actor_returned,
            "container_exit_monotonic_ns": exited,
            "measurement_wall_time_ns": exited - started,
            "wall_time_limit_ns": spec.runtime_limits[stage].wall_time_seconds * 1_000_000_000,
            "maximum_concurrent_actor_containers": 1,
            "actor_terminal_scratch_payload_bytes": 0,
            "supervisor_control_peak_payload_bytes": 0,
        }
        return ActorProcessResultV1(actor_code, stdout, stderr, True, evidence)
    except BaseException:
        if container_id is not None:
            subprocess.run([DOCKER, "rm", "-f", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        elif process is not None and process.poll() is None:
            process.terminate()
        raise


def _tree_bytes(root: Path, maximum: int) -> int:
    total = 0
    for path in root.rglob("*"):
        status = path.lstat()
        if path.is_symlink() or not (stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)):
            _terminal("Q1_START_OUTPUT_TREE_INVALID", f"unsupported output entry {path}")
        if stat.S_ISREG(status.st_mode):
            total += status.st_size
            if total > maximum:
                _terminal("Q1_START_OUTPUT_LIMIT_EXCEEDED", str(root))
    return total


def _ordered_regular_paths(root: Path) -> tuple[str, ...]:
    rows: list[str] = []
    for path in root.rglob("*"):
        status = path.lstat()
        if path.is_symlink() or not (
            stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)
        ):
            _terminal("Q1_START_OUTPUT_TREE_INVALID", f"unsupported output entry {path}")
        if stat.S_ISREG(status.st_mode):
            rows.append(path.relative_to(root).as_posix())
    return tuple(sorted(rows))


def _file_closure_root(root: Path, relative_paths: Sequence[str], domain: bytes) -> str:
    digest = sha256()
    digest.update(domain)
    previous = ""
    for relative in relative_paths:
        if type(relative) is not str or not relative or relative <= previous:
            _terminal("Q1_START_OUTPUT_TREE_INVALID", "file closure path order differs")
        path = root / relative
        try:
            status = path.lstat()
        except OSError as error:
            _terminal("Q1_START_OUTPUT_TREE_INVALID", f"{relative}: {error}")
        if path.is_symlink() or not stat.S_ISREG(status.st_mode):
            _terminal("Q1_START_OUTPUT_TREE_INVALID", f"{relative} is not regular")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(status.st_size.to_bytes(8, "big"))
        try:
            with path.open("rb", buffering=0) as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError as error:
            _terminal("Q1_START_OUTPUT_TREE_INVALID", f"{relative}: {error}")
        previous = relative
    return "sha256:" + digest.hexdigest()


def _output_tree_closure_root(root: Path) -> str:
    return _file_closure_root(root, _ordered_regular_paths(root), _OUTPUT_TREE_CLOSURE_DOMAIN)


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        status = path.lstat()
        path.chmod(0o444 if stat.S_ISREG(status.st_mode) else 0o555)
    root.chmod(0o555)
    _fsync_directory(root.parent)


def _load_formal_replay_core_v1(spec: Q1StartRunSpecV1):
    package_root = spec.host_source_root / "src" / "hegel_machine"
    namespace = "_hegel_q1_snapshot_" + spec.capsule_digest.removeprefix("sha256:")
    package = sys.modules.get(namespace)
    expected_path = [str(package_root)]
    if package is not None and getattr(package, "__path__", None) != expected_path:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                sys.modules.pop(name, None)
        package = None
    if package is None:
        if not package_root.is_dir():
            _terminal(
                "Q1_START_ENDPOINT_REPLAY_UNAVAILABLE",
                "materialized host package source root differs",
            )
        package = types.ModuleType(namespace)
        package.__package__ = namespace
        package.__path__ = expected_path  # type: ignore[attr-defined]
        sys.modules[namespace] = package
    try:
        return importlib.import_module(
            namespace + ".phase3_q1_formal_host_replay_v1"
        )
    except (ImportError, OSError) as error:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                sys.modules.pop(name, None)
        _terminal("Q1_START_ENDPOINT_REPLAY_UNAVAILABLE", str(error))


def _guard_rows(input_signature_id: int, fixed_point: object) -> list[dict[str, object]]:
    limits = _FORMAL_GUARD_LIMITS[input_signature_id]
    observed = (
        getattr(fixed_point, "raw_application_count"),
        getattr(fixed_point, "class_count"),
        getattr(fixed_point, "frontier_point_count"),
        getattr(fixed_point, "maximum_frontier_points_per_class"),
        getattr(fixed_point, "bank_point_count"),
        getattr(fixed_point, "maximum_bank_points_per_class"),
        limits[6],
        limits[7],
    )
    rows: list[dict[str, object]] = []
    for index, (name, value, limit) in enumerate(
        zip(_GUARD_NAMES, observed, limits, strict=True),
        1,
    ):
        if type(value) is not int or value < 0 or value > limit:
            _terminal(
                "Q1_START_FORMAL_GUARD_EVIDENCE_MISMATCH",
                f"signature {input_signature_id} guard {index}",
            )
        rows.append(
            {
                "guard_id": index,
                "guard_name": name,
                "evidence_kind": (
                    "REPLAYED_FIXED_POINT_COUNT"
                    if index <= 6
                    else "FROZEN_LIMIT_CONSERVATIVE_UPPER_BOUND"
                ),
                "observed_or_conservative_bound": value,
                "frozen_limit": limit,
                "guard_hit": False,
            }
        )
    return rows


def default_endpoint_replayer_v1(stage: str, output: Path, spec: Q1StartRunSpecV1) -> Mapping[str, object]:
    core = _load_formal_replay_core_v1(spec)
    replay = core.replay_endpoint_formal_archive_v1(output)
    if (
        replay.endpoint_kind != stage
        or replay.capsule_digest != spec.capsule_digest
        or replay.source_commit != spec.source_commit
        or replay.run_genesis_root != spec.run_genesis_root
        or replay.prestart_admission_receipt_root != spec.prestart_admission_receipt_root
    ):
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} identity differs")
    relative_paths = tuple(replay.formal_relative_paths)
    if relative_paths != tuple(sorted(relative_paths)):
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} formal path order differs")
    signature_rows = [
        {
            "input_signature_id": item.input_signature_id,
            "fixed_point_record_root": "sha256:" + item.fixed_point.record_root.hex(),
            "same_run_record_set_root": "sha256:" + item.same_run_record_set_root.hex(),
            "signature_manifest_root": "sha256:" + item.signature_manifest_root.hex(),
            "saturation_state_root": "sha256:" + item.saturation_state_root.hex(),
            "guard_evidence_rows": _guard_rows(item.input_signature_id, item.fixed_point),
        }
        for item in replay.signatures
    ]
    return {
        "actor_id": replay.endpoint_kind,
        "archive_payload_bytes": replay.archive_payload_bytes,
        "closure_bundle_root": "sha256:" + replay.closure_bundle_root.hex(),
        "formal_relative_paths": list(relative_paths),
        "formal_file_closure_root": _file_closure_root(
            output,
            relative_paths,
            _FORMAL_FILE_CLOSURE_DOMAIN,
        ),
        "signature_rows": signature_rows,
    }


def _validate_runtime_evidence(stage: str, evidence: Mapping[str, object], spec: Q1StartRunSpecV1) -> dict[str, object]:
    value = dict(evidence)
    expected_keys = {
        "schema_version",
        "stage",
        "container_id",
        "image",
        "cpuset_cpus",
        "memory_limit_bytes",
        "memory_peak_bytes",
        "network_mode",
        "ipc_mode",
        "read_only_rootfs",
        "mounts",
        "cgroup_path",
        "measurement_interval_start_monotonic_ns",
        "actor_return_monotonic_ns",
        "container_exit_monotonic_ns",
        "measurement_wall_time_ns",
        "wall_time_limit_ns",
        "maximum_concurrent_actor_containers",
        "actor_terminal_scratch_payload_bytes",
        "supervisor_control_peak_payload_bytes",
        "command",
        "environment_overrides",
        "cap_drop",
        "nofile_limit",
        "tmpfs",
        "seccomp_profile_raw_sha256",
    }
    if set(value) != expected_keys:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"{stage} runtime evidence closure differs")
    limit = spec.runtime_limits[stage]
    fixed = {
        "schema_version": RUNTIME_EVIDENCE_SCHEMA,
        "stage": stage,
        "image": spec.image_for(stage),
        "cpuset_cpus": limit.cpuset_cpus,
        "memory_limit_bytes": limit.memory_bytes,
        "network_mode": "none",
        "ipc_mode": "private",
        "read_only_rootfs": True,
        "mounts": _expected_mounts(spec, stage),
        "maximum_concurrent_actor_containers": 1,
        "actor_terminal_scratch_payload_bytes": 0,
        "supervisor_control_peak_payload_bytes": 0,
        "command": list(docker_commands_v1(spec)[stage]),
        "environment_overrides": {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",
        },
        "cap_drop": ["ALL"],
        "nofile_limit": NOFILE_LIMIT,
        "tmpfs": TMPFS_SPEC,
        "seccomp_profile_raw_sha256": spec.seccomp_profile_raw_sha256,
        "wall_time_limit_ns": limit.wall_time_seconds * 1_000_000_000,
    }
    if any(value.get(name) != expected for name, expected in fixed.items()):
        _terminal("Q1_START_RUNTIME_ISOLATION_MISMATCH", f"{stage} fixed evidence differs")
    if type(value["container_id"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["container_id"]) is None:
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"{stage} container identity differs")
    if type(value["cgroup_path"]) is not str or not value["cgroup_path"].startswith("/"):
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"{stage} cgroup path differs")
    peak = value["memory_peak_bytes"]
    if type(peak) is not int or peak < 0 or peak > limit.memory_bytes:
        _terminal("Q1_START_MEMORY_LIMIT_EXCEEDED", stage)
    times = [
        value["measurement_interval_start_monotonic_ns"],
        value["actor_return_monotonic_ns"],
        value["container_exit_monotonic_ns"],
    ]
    if any(type(item) is not int or item < 0 for item in times) or times != sorted(times):
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"{stage} interval differs")
    if (
        type(value["measurement_wall_time_ns"]) is not int
        or value["measurement_wall_time_ns"] != times[2] - times[0]
        or value["measurement_wall_time_ns"] > value["wall_time_limit_ns"]
    ):
        _terminal("Q1_START_RUNTIME_EVIDENCE_UNKNOWN", f"{stage} wall-time differs")
    return value


_ENDPOINT_REPLAY_KEYS = {
    "actor_id",
    "archive_payload_bytes",
    "closure_bundle_root",
    "formal_relative_paths",
    "formal_file_closure_root",
    "signature_rows",
}
_SIGNATURE_REPLAY_KEYS = {
    "input_signature_id",
    "fixed_point_record_root",
    "same_run_record_set_root",
    "signature_manifest_root",
    "saturation_state_root",
    "guard_evidence_rows",
}
_GUARD_EVIDENCE_KEYS = {
    "guard_id",
    "guard_name",
    "evidence_kind",
    "observed_or_conservative_bound",
    "frozen_limit",
    "guard_hit",
}


def _validate_endpoint_replay_summary(stage: str, replay: Mapping[str, object]) -> dict[str, object]:
    value = dict(replay)
    if set(value) != _ENDPOINT_REPLAY_KEYS or value.get("actor_id") != stage:
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} replay closure differs")
    for name in ("closure_bundle_root", "formal_file_closure_root"):
        if type(value[name]) is not str or _DIGEST.fullmatch(value[name]) is None:
            _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} {name} differs")
    if type(value["archive_payload_bytes"]) is not int or value["archive_payload_bytes"] <= 0:
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} payload count differs")
    paths = value["formal_relative_paths"]
    if (
        type(paths) is not list
        or not paths
        or any(type(path) is not str or not path for path in paths)
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
    ):
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} formal paths differ")
    signatures = value["signature_rows"]
    if type(signatures) is not list or len(signatures) != 2:
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} signatures differ")
    for signature_id, row in enumerate(signatures, 1):
        if type(row) is not dict or set(row) != _SIGNATURE_REPLAY_KEYS:
            _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} signature closure differs")
        if row["input_signature_id"] != signature_id:
            _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} signature order differs")
        for name in (
            "fixed_point_record_root",
            "same_run_record_set_root",
            "signature_manifest_root",
            "saturation_state_root",
        ):
            if type(row[name]) is not str or _DIGEST.fullmatch(row[name]) is None:
                _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} signature root differs")
        guards = row["guard_evidence_rows"]
        limits = _FORMAL_GUARD_LIMITS[signature_id]
        if type(guards) is not list or len(guards) != 8:
            _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} guard rows differ")
        for guard_id, guard in enumerate(guards, 1):
            if type(guard) is not dict or set(guard) != _GUARD_EVIDENCE_KEYS:
                _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} guard closure differs")
            observed = guard["observed_or_conservative_bound"]
            if (
                guard["guard_id"] != guard_id
                or guard["guard_name"] != _GUARD_NAMES[guard_id - 1]
                or guard["frozen_limit"] != limits[guard_id - 1]
                or guard["guard_hit"] is not False
                or type(observed) is not int
                or observed < 0
                or observed > limits[guard_id - 1]
                or guard["evidence_kind"]
                != (
                    "REPLAYED_FIXED_POINT_COUNT"
                    if guard_id <= 6
                    else "FROZEN_LIMIT_CONSERVATIVE_UPPER_BOUND"
                )
                or (guard_id >= 7 and observed != limits[guard_id - 1])
            ):
                _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage} guard evidence differs")
    return value


def _validate_endpoint(
    stage: str,
    process: ActorProcessResultV1,
    spec: Q1StartRunSpecV1,
    endpoint_replayer: EndpointReplayer,
) -> tuple[dict[str, object], dict[str, object], int]:
    limit = spec.runtime_limits[stage]
    if process.outcome_known is not True or type(process.return_code) is not int:
        _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", stage)
    if process.return_code != 0:
        _terminal("Q1_START_ACTOR_NONZERO", f"{stage}:{process.return_code}")
    if process.stderr != b"":
        _terminal("Q1_START_ACTOR_STDERR", stage)
    result = _strict_object(process.stdout, f"{stage} stdout", maximum=limit.stdout_bytes)
    expected_actor = stage
    expected_impl = f"hegel-{stage}-q1-formal-archive-materializer-v1"
    if (
        set(result) != _ENDPOINT_RESULT_KEYS
        or result.get("schema_version") != MATERIALIZER_SCHEMA
        or result.get("actor_id") != expected_actor
        or result.get("implementation_id") != expected_impl
        or result.get("action_id") != MATERIALIZER_ACTION
        or result.get("status") != MATERIALIZER_STATUS
        or any(result.get(name) != expected for name, expected in spec.binding().items())
        or result.get("q1_state") != "NOT_RUN"
        or result.get("q1_gate_count") != 14
        or result.get("q1_gate_total") != 20
        or result.get("q1_gate_mask") != 16_383
        or result.get("q1_output_slots") != [None] * 8
        or result.get("diagnostic_only") is not False
        or result.get("formal_archive_materialized") is not True
        or result.get("target_truth_accessed") is not False
        or result.get("split_accessed") is not False
        or result.get("role_evaluation_performed") is not False
        or result.get("formal_capacity_limits_by_signature") != list(_FORMAL_LIMIT_ROWS)
        or result.get("formal_output_bytes_limit") != ENDPOINT_OUTPUT_BYTES
        or result.get("formal_scratch_bytes_limit") != ENDPOINT_SCRATCH_BYTES
        or result.get("actual_formal_terminal_live_scratch_bytes") != 0
        or type(result.get("actual_formal_peak_scratch_bytes")) is not int
        or not 0 <= result["actual_formal_peak_scratch_bytes"] <= ENDPOINT_SCRATCH_BYTES
        or result.get("dsl_version") != "hegel-old-dsl-v1.6.0"
        or result.get("closure_semantics_version") != "hegel-quotient-closure-v1.0.1"
        or result.get("full_v16_structural_limits_applied") is not True
        or result.get("maximum_ast_depth") != 3
        or result.get("maximum_ast_node_count") != 6
        or type(result.get("partitions")) is not list
        or len(result["partitions"]) != 2
        or type(result.get("formal_archive_payload_bytes")) is not int
        or result["formal_archive_payload_bytes"] <= 0
        or type(result.get("closure_bundle_root")) is not str
        or _DIGEST.fullmatch(result["closure_bundle_root"]) is None
    ):
        _terminal("Q1_START_ENDPOINT_RESULT_MISMATCH", stage)
    endpoint_path = spec.output_for(stage) / "endpoint-result.json"
    try:
        persisted = endpoint_path.read_bytes()
    except OSError as error:
        _terminal("Q1_START_ENDPOINT_TREE_INVALID", f"{stage}: {error}")
    if persisted != process.stdout:
        _terminal("Q1_START_ENDPOINT_RESULT_MISMATCH", f"{stage} stdout/tree differ")
    tree_bytes = _tree_bytes(spec.output_for(stage), limit.output_bytes)
    try:
        replay = _validate_endpoint_replay_summary(
            stage,
            endpoint_replayer(stage, spec.output_for(stage), spec),
        )
    except TerminalCondition:
        raise
    except BaseException as error:
        _terminal("Q1_START_ENDPOINT_REPLAY_MISMATCH", f"{stage}:{type(error).__name__}:{error}")
    runtime = _validate_runtime_evidence(stage, process.runtime_evidence, spec)
    return result, {"endpoint_replay": replay, "runtime_evidence": runtime}, tree_bytes


_HOST_RESULT_KEYS = {
    "schema_version", "actor_id", "implementation_id", "action_id", "status",
    "capsule_digest", "source_commit", "run_genesis_root", "prestart_admission_receipt_root",
    "q1_state", "q1_gate_count", "q1_gate_total", "q1_gate_mask", "q1_output_slots",
    "formal_capacity_limits_by_signature", "formal_output_bytes_limit", "formal_scratch_bytes_limit",
    "semantic_binding_root", "projection_profile_root", "closure_bundle_root",
    "archive_payload_bytes_per_endpoint", "formal_file_count_per_endpoint", "signatures",
    "python_rust_formal_bytes_exact", "python_rust_replayed_roots_exact",
    "target_truth_accessed", "split_accessed", "role_evaluation_performed",
    "target_split_role_absence_attested", "q1_completion_receipt_issued", "state_published",
}
_HOST_SIGNATURE_KEYS = {
    "input_signature_id",
    "universe_root",
    "fixed_point_record_root",
    "signature_manifest_root",
    "saturation_state_root",
    "record_count_by_stream",
}


def _validate_host_result_object(
    result: Mapping[str, object],
    spec: Q1StartRunSpecV1,
) -> dict[str, object]:
    result = dict(result)
    if set(result) != _HOST_RESULT_KEYS:
        _terminal("Q1_START_HOST_RESULT_MISMATCH", "host result key closure differs")
    expected = {
        "schema_version": HOST_SCHEMA,
        "actor_id": "host",
        "implementation_id": "hegel-host-q1-formal-archive-replay-v1",
        "action_id": HOST_ACTION,
        "status": HOST_STATUS,
        **spec.binding(),
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "python_rust_formal_bytes_exact": True,
        "python_rust_replayed_roots_exact": True,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "target_split_role_absence_attested": True,
        "q1_completion_receipt_issued": False,
        "state_published": False,
        "formal_capacity_limits_by_signature": list(_FORMAL_LIMIT_ROWS),
        "formal_output_bytes_limit": ENDPOINT_OUTPUT_BYTES,
        "formal_scratch_bytes_limit": ENDPOINT_SCRATCH_BYTES,
    }
    if any(result.get(name) != value for name, value in expected.items()):
        _terminal("Q1_START_HOST_RESULT_MISMATCH", "host result boundary differs")
    signatures = result.get("signatures")
    if (
        type(signatures) is not list
        or len(signatures) != 2
        or type(result.get("archive_payload_bytes_per_endpoint")) is not int
        or result["archive_payload_bytes_per_endpoint"] <= 0
        or type(result.get("formal_file_count_per_endpoint")) is not int
        or result["formal_file_count_per_endpoint"] <= 0
    ):
        _terminal("Q1_START_HOST_RESULT_MISMATCH", "host replay closure differs")
    assert isinstance(signatures, list)
    for signature_id, row in enumerate(signatures, 1):
        if (
            type(row) is not dict
            or set(row) != _HOST_SIGNATURE_KEYS
            or row.get("input_signature_id") != signature_id
            or any(
                type(row.get(name)) is not str
                or _DIGEST.fullmatch(row[name]) is None
                for name in (
                    "universe_root",
                    "fixed_point_record_root",
                    "signature_manifest_root",
                    "saturation_state_root",
                )
            )
            or type(row.get("record_count_by_stream")) is not list
            or len(row["record_count_by_stream"]) != 4
            or any(type(count) is not int or count <= 0 for count in row["record_count_by_stream"])
        ):
            _terminal("Q1_START_HOST_RESULT_MISMATCH", "host signature replay differs")
    return result


def _validate_host(process: ActorProcessResultV1, spec: Q1StartRunSpecV1) -> tuple[dict[str, object], dict[str, object], int]:
    limit = spec.runtime_limits["host"]
    if process.outcome_known is not True or type(process.return_code) is not int:
        _terminal("Q1_START_ACTOR_OUTCOME_UNKNOWN", "host")
    if process.return_code != 0:
        _terminal("Q1_START_ACTOR_NONZERO", f"host:{process.return_code}")
    if process.stderr != b"":
        _terminal("Q1_START_ACTOR_STDERR", "host")
    result = _validate_host_result_object(
        _strict_object(process.stdout, "host stdout", maximum=limit.stdout_bytes),
        spec,
    )
    result_path = spec.host_output / "host-result.json"
    _publish_exclusive(result_path, result)
    tree_bytes = _tree_bytes(spec.host_output, limit.output_bytes)
    runtime = _validate_runtime_evidence("host", process.runtime_evidence, spec)
    return result, {"host_replay_validated": True, "runtime_evidence": runtime}, tree_bytes


def _stage_paths(spec: Q1StartRunSpecV1, stage: str) -> tuple[Path, Path]:
    return (
        spec.checkpoint_root / f"{STAGES.index(stage) + 1:02d}-{stage}-intent.json",
        spec.checkpoint_root / f"{STAGES.index(stage) + 1:02d}-{stage}-complete.json",
    )


_INTENT_KEYS = {
    "schema_version", "stage", "ordinal", "capsule_digest", "source_commit",
    "run_genesis_root", "prestart_admission_receipt_root", "command",
    "output_path", "scratch_path", "status", "maximum_actor_popen_count",
}
_CHECKPOINT_KEYS = {
    "schema_version", "stage", "ordinal", "capsule_digest", "source_commit",
    "run_genesis_root", "prestart_admission_receipt_root", "command",
    "intent_path", "output_path", "scratch_path", "output_tree_bytes",
    "output_tree_closure_root", "scratch_absence_root",
    "scratch_absent_after_cleanup", "actor_popen_count", "result",
    "validation", "status",
}
_TERMINAL_KEYS = {
    "schema_version", "status", "capsule_digest", "source_commit",
    "run_genesis_root", "prestart_admission_receipt_root", "active_stage",
    "completed_stages", "error_code", "error_detail", "resume_allowed",
    "q1_completion_receipt_issued", "state_published",
}


def _intent_object(spec: Q1StartRunSpecV1, stage: str, command: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": INTENT_SCHEMA,
        "stage": stage,
        "ordinal": STAGES.index(stage) + 1,
        **spec.binding(),
        "command": list(command),
        "output_path": str(spec.output_for(stage)),
        "scratch_path": str(spec.scratch_for(stage)),
        "status": "DURABLE_INTENT_FSYNCED_BEFORE_ACTOR_POPEN",
        "maximum_actor_popen_count": 1,
    }


def _prepare_stage_roots(spec: Q1StartRunSpecV1, stage: str) -> None:
    created: list[Path] = []
    try:
        cidfile = spec.checkpoint_root / f"{stage}.cid"
        if cidfile.exists() or cidfile.is_symlink():
            raise PendingCondition(
                "Q1_START_STAGE_ROOT_PENDING",
                f"unexpected pre-intent cidfile {cidfile}",
            )
        for root in (spec.output_for(stage), spec.scratch_for(stage)):
            if root.exists() or root.is_symlink():
                raise PendingCondition("Q1_START_STAGE_ROOT_PENDING", f"unexpected pre-intent root {root}")
            root.mkdir(mode=0o777)
            created.append(root)
            root.chmod(0o777)
        _fsync_directory(spec.run_root)
    except PendingCondition:
        for root in reversed(created):
            shutil.rmtree(root, ignore_errors=True)
        _fsync_directory(spec.run_root)
        raise
    except OSError as error:
        failures: list[str] = []
        for root in reversed(created):
            try:
                if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
                    failures.append(f"{root.name}:created root is not clean")
                else:
                    root.rmdir()
            except OSError as cleanup_error:
                failures.append(f"{root.name}:{cleanup_error}")
        try:
            _fsync_directory(spec.run_root)
        except OSError as cleanup_error:
            failures.append(f"run-root-fsync:{cleanup_error}")
        intent_path, checkpoint_path = _stage_paths(spec, stage)
        cidfile = spec.checkpoint_root / f"{stage}.cid"
        for path in (
            spec.output_for(stage),
            spec.scratch_for(stage),
            cidfile,
            intent_path,
            checkpoint_path,
        ):
            if path.exists() or path.is_symlink():
                failures.append(f"{path.name}:absence not proved")
        if failures:
            _terminal("Q1_START_PREINTENT_CLEANUP_UNPROVED", "|".join(failures))
        raise PendingCondition(
            "Q1_START_STAGE_ROOT_PREPARATION_PENDING",
            f"{stage}: {error}",
        ) from error


def _cleanup_preintent_stage_roots(spec: Q1StartRunSpecV1, stage: str) -> None:
    failures: list[str] = []
    for root in (spec.scratch_for(stage), spec.output_for(stage)):
        try:
            if root.is_dir() and not root.is_symlink() and not any(root.iterdir()):
                root.rmdir()
        except OSError as error:
            failures.append(f"{root.name}:{error}")
    _fsync_directory(spec.run_root)
    intent_path, checkpoint_path = _stage_paths(spec, stage)
    cidfile = spec.checkpoint_root / f"{stage}.cid"
    for root in (
        spec.scratch_for(stage),
        spec.output_for(stage),
        cidfile,
        intent_path,
        checkpoint_path,
    ):
        if root.exists() or root.is_symlink():
            failures.append(f"{root.name}:absence not proved")
    if failures:
        _terminal("Q1_START_PREINTENT_CLEANUP_UNPROVED", "|".join(failures))


def _scratch_clean_and_remove(spec: Q1StartRunSpecV1, stage: str) -> None:
    scratch = spec.scratch_for(stage)
    try:
        if scratch.is_symlink() or not scratch.is_dir() or any(scratch.iterdir()):
            _terminal("Q1_START_SCRATCH_NOT_CLEAN", stage)
        scratch.rmdir()
        _fsync_directory(spec.run_root)
    except TerminalCondition:
        raise
    except OSError as error:
        _terminal("Q1_START_SCRATCH_NOT_CLEAN", f"{stage}:{error}")


def _checkpoint_object(
    spec: Q1StartRunSpecV1,
    stage: str,
    command: Sequence[str],
    result: Mapping[str, object],
    validation: Mapping[str, object],
    output_tree_bytes: int,
) -> dict[str, object]:
    intent_path, _ = _stage_paths(spec, stage)
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": stage,
        "ordinal": STAGES.index(stage) + 1,
        **spec.binding(),
        "command": list(command),
        "intent_path": str(intent_path),
        "output_path": str(spec.output_for(stage)),
        "scratch_path": str(spec.scratch_for(stage)),
        "output_tree_bytes": output_tree_bytes,
        "output_tree_closure_root": _output_tree_closure_root(spec.output_for(stage)),
        "scratch_absence_root": _object_root(
            _SCRATCH_ABSENCE_DOMAIN,
            {"path": str(spec.scratch_for(stage)), "absent": True},
        ),
        "scratch_absent_after_cleanup": True,
        "actor_popen_count": 1,
        "result": dict(result),
        "validation": dict(validation),
        "status": "CLEAN_CHECKPOINT_COMPLETE_READONLY_REPLAY_ONLY",
    }


def _replay_checkpoint(
    spec: Q1StartRunSpecV1,
    stage: str,
    command: Sequence[str],
    endpoint_replayer: EndpointReplayer,
) -> dict[str, object]:
    intent_path, checkpoint_path = _stage_paths(spec, stage)
    intent = _read_canonical_file(intent_path, f"{stage} intent")
    if set(intent) != _INTENT_KEYS or intent != _intent_object(spec, stage, command):
        _terminal("Q1_START_INTENT_REPLAY_MISMATCH", stage)
    checkpoint = _read_canonical_file(checkpoint_path, f"{stage} checkpoint")
    if (
        set(checkpoint) != _CHECKPOINT_KEYS
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("stage") != stage
        or checkpoint.get("ordinal") != STAGES.index(stage) + 1
        or checkpoint.get("command") != list(command)
        or checkpoint.get("output_path") != str(spec.output_for(stage))
        or checkpoint.get("scratch_path") != str(spec.scratch_for(stage))
        or checkpoint.get("intent_path") != str(intent_path)
        or checkpoint.get("scratch_absent_after_cleanup") is not True
        or checkpoint.get("actor_popen_count") != 1
        or checkpoint.get("status") != "CLEAN_CHECKPOINT_COMPLETE_READONLY_REPLAY_ONLY"
        or any(checkpoint.get(name) != value for name, value in spec.binding().items())
        or spec.scratch_for(stage).exists()
    ):
        _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", stage)
    result = checkpoint.get("result")
    validation = checkpoint.get("validation")
    if type(result) is not dict or type(validation) is not dict:
        _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} payload differs")
    if stage != "host":
        try:
            persisted = (spec.output_for(stage) / "endpoint-result.json").read_bytes()
        except OSError as error:
            _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage}:{error}")
        if persisted != _canonical(result):
            _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} result differs")
        replay = _validate_endpoint_replay_summary(
            stage,
            endpoint_replayer(stage, spec.output_for(stage), spec),
        )
        if validation.get("endpoint_replay") != replay:
            _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} replay differs")
    else:
        try:
            persisted = (spec.host_output / "host-result.json").read_bytes()
        except OSError as error:
            _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"host:{error}")
        if persisted != _canonical(result) or validation.get("host_replay_validated") is not True:
            _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", "host result differs")
    if checkpoint.get("output_tree_bytes") != _tree_bytes(spec.output_for(stage), spec.runtime_limits[stage].output_bytes):
        _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} output accounting differs")
    if checkpoint.get("output_tree_closure_root") != _output_tree_closure_root(spec.output_for(stage)):
        _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} output closure differs")
    if checkpoint.get("scratch_absence_root") != _object_root(
        _SCRATCH_ABSENCE_DOMAIN,
        {"path": str(spec.scratch_for(stage)), "absent": True},
    ):
        _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} scratch absence differs")
    runtime = validation.get("runtime_evidence")
    if type(runtime) is not dict:
        _terminal("Q1_START_CHECKPOINT_REPLAY_MISMATCH", f"{stage} runtime evidence differs")
    _validate_runtime_evidence(stage, runtime, spec)
    # Validate the durable checkpoint and its outputs before replaying mutable
    # local source/runtime availability.  Therefore a PendingCondition below
    # means this actor is already exactly complete and must never be relaunched.
    try:
        current_bindings = _replay_stage_bindings(stage, spec)
    except PendingCondition as error:
        raise CompletedCheckpointBindingPending(error.code, error.detail) from error
    if validation.get("stage_binding_replay") != current_bindings:
        raise PendingCondition(
            "Q1_START_CHECKPOINT_BINDING_PENDING",
            f"{stage} source/runtime binding replay differs",
        )
    return checkpoint


def _terminal_outcome(spec: Q1StartRunSpecV1, stage: str, completed: Sequence[str], error: Q1StartSupervisorError) -> dict[str, object]:
    return {
        "schema_version": OUTCOME_SCHEMA,
        "status": TERMINAL_STATUS,
        **spec.binding(),
        "active_stage": stage,
        "completed_stages": list(completed),
        "error_code": error.code,
        "error_detail": error.detail,
        "resume_allowed": False,
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }


def _publish_terminal(spec: Q1StartRunSpecV1, outcome: Mapping[str, object]) -> None:
    if set(outcome) != _TERMINAL_KEYS:
        _reject("REJECT_Q1_START_TERMINAL_WIRE", "terminal key closure differs")
    path = spec.checkpoint_root / "terminal-outcome.json"
    if path.exists():
        if _read_canonical_file(path, "terminal outcome") != dict(outcome):
            _reject("REJECT_Q1_START_TERMINAL_REPLAY", "terminal outcome differs")
        return
    _publish_exclusive(path, outcome)


def _replay_terminal(spec: Q1StartRunSpecV1, path: Path) -> dict[str, object]:
    outcome = _read_canonical_file(path, "terminal outcome")
    if (
        set(outcome) != _TERMINAL_KEYS
        or outcome.get("schema_version") != OUTCOME_SCHEMA
        or outcome.get("status") != TERMINAL_STATUS
        or outcome.get("resume_allowed") is not False
        or outcome.get("q1_completion_receipt_issued") is not False
        or outcome.get("state_published") is not False
        or any(outcome.get(name) != value for name, value in spec.binding().items())
    ):
        _reject("REJECT_Q1_START_TERMINAL_REPLAY", "terminal outcome differs")
    return outcome


def _success_outcome(spec: Q1StartRunSpecV1, checkpoints: Sequence[Mapping[str, object]]) -> dict[str, object]:
    host_result = dict(checkpoints[2]["result"])
    python_replay = dict(checkpoints[0]["validation"]["endpoint_replay"])
    rust_replay = dict(checkpoints[1]["validation"]["endpoint_replay"])
    for name in (
        "archive_payload_bytes",
        "closure_bundle_root",
        "formal_relative_paths",
        "formal_file_closure_root",
        "signature_rows",
    ):
        if python_replay[name] != rust_replay[name]:
            _terminal("Q1_START_DUAL_FORMAL_CLOSURE_MISMATCH", name)
    gate18 = {
        "schema_version": GATE18_INPUT_SCHEMA,
        **spec.binding(),
        "actor_order": list(STAGES),
        "maximum_concurrent_actor_containers": 1,
        "maximum_actor_popen_count_by_stage": {stage: 1 for stage in STAGES},
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "stage_rows": [
            {
                "stage": stage,
                "ordinal": index + 1,
                "intent_path": str(_stage_paths(spec, stage)[0]),
                "checkpoint_path": str(_stage_paths(spec, stage)[1]),
                "output_path": str(spec.output_for(stage)),
                "scratch_path": str(spec.scratch_for(stage)),
                "scratch_absent_after_cleanup": True,
                "output_tree_bytes": checkpoints[index]["output_tree_bytes"],
                "source_manifest_replay": checkpoints[index]["validation"]["stage_binding_replay"]["actor_source_manifest"],
                "runtime_binding_replay": {
                    "actor_implementation_root": checkpoints[index]["validation"]["stage_binding_replay"]["actor_implementation_root"],
                    "seccomp_profile": checkpoints[index]["validation"]["stage_binding_replay"]["seccomp_profile"],
                    "rust_binary": checkpoints[index]["validation"]["stage_binding_replay"]["rust_binary"],
                    "python_admission": checkpoints[index]["validation"]["stage_binding_replay"]["python_admission"],
                },
                "mount_registry_root": checkpoints[index]["validation"]["stage_binding_replay"]["mount_registry_root"],
                "output_tree_closure_root": checkpoints[index]["output_tree_closure_root"],
                "scratch_absence_root": checkpoints[index]["scratch_absence_root"],
                "runtime_limit": spec.runtime_limits[stage].object(),
                "runtime_evidence": checkpoints[index]["validation"]["runtime_evidence"],
                "actor_reported_formal_peak_scratch_bytes": (
                    checkpoints[index]["result"].get("actual_formal_peak_scratch_bytes")
                    if stage != "host"
                    else None
                ),
                "actor_reported_formal_terminal_live_scratch_bytes": (
                    checkpoints[index]["result"].get("actual_formal_terminal_live_scratch_bytes")
                    if stage != "host"
                    else None
                ),
                "writable_mount_destinations": ["/output", "/scratch"],
            }
            for index, stage in enumerate(STAGES)
        ],
    }
    return {
        "schema_version": OUTCOME_SCHEMA,
        "status": SUCCESS_STATUS,
        **spec.binding(),
        "completed_stages": list(STAGES),
        "checkpoint_paths": [str(_stage_paths(spec, stage)[1]) for stage in STAGES],
        "host_result": host_result,
        "gate15_candidate": {
            "both_signature_fixed_points_complete_no_guard_hit": True,
            "no_guard_proof_basis": "EXACT_ENDPOINT_SUCCESS_AND_FROZEN_ACTOR_SOURCE",
            "formal_capacity_limits_by_signature": host_result["formal_capacity_limits_by_signature"],
            "actor_implementation_roots": dict(spec.actor_implementation_roots),
            "actor_source_closure_roots": {
                stage: checkpoints[index]["validation"]["stage_binding_replay"]["actor_source_manifest"]["source_closure_root"]
                for index, stage in enumerate(STAGES)
            },
            "python_signature_replay": python_replay["signature_rows"],
            "rust_signature_replay": rust_replay["signature_rows"],
        },
        "gate16_candidate": {
            "python_rust_formal_bytes_exact": host_result["python_rust_formal_bytes_exact"],
            "python_rust_replayed_roots_exact": host_result["python_rust_replayed_roots_exact"],
            "formal_file_closure_root": python_replay["formal_file_closure_root"],
            "formal_relative_paths": python_replay["formal_relative_paths"],
            "signature_state_preimages": python_replay["signature_rows"],
        },
        "gate17_candidate": {"host_replay_status": host_result["status"]},
        "gate18_runtime_receipt_inputs": gate18,
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }


def run_supervisor_v1(
    spec: Q1StartRunSpecV1,
    *,
    launcher: RunLauncher = actual_launcher_v1,
    preflight: StagePreflight = default_preflight_v1,
    endpoint_replayer: EndpointReplayer = default_endpoint_replayer_v1,
) -> dict[str, object]:
    _existing_directory(spec.run_root, "run_root")
    try:
        if not spec.checkpoint_root.exists():
            spec.checkpoint_root.mkdir(mode=0o700)
            _fsync_directory(spec.run_root)
        _existing_directory(spec.checkpoint_root, "checkpoint_root")
    except PendingCondition:
        raise
    except OSError as error:
        raise PendingCondition("Q1_START_CHECKPOINT_ROOT_PENDING", str(error)) from error
    terminal_path = spec.checkpoint_root / "terminal-outcome.json"
    if terminal_path.exists():
        return _replay_terminal(spec, terminal_path)
    commands = docker_commands_v1(spec)
    checkpoints: list[dict[str, object]] = []
    completed: list[str] = []
    actor_popen_count_this_entry = 0
    for stage in STAGES:
        stage_index = STAGES.index(stage)
        intent_path, checkpoint_path = _stage_paths(spec, stage)
        if checkpoint_path.exists():
            if not intent_path.exists():
                error = TerminalCondition("Q1_START_CHECKPOINT_WITHOUT_INTENT", stage)
                outcome = _terminal_outcome(spec, stage, completed, error)
                _publish_terminal(spec, outcome)
                return outcome
            try:
                checkpoint = _replay_checkpoint(spec, stage, commands[stage], endpoint_replayer)
            except CompletedCheckpointBindingPending as pending:
                if stage == "host":
                    error = TerminalCondition(
                        "Q1_START_POST_HOST_CHECKPOINT_BINDING_UNAVAILABLE",
                        pending.detail,
                    )
                    outcome = _terminal_outcome(spec, stage, [*completed, stage], error)
                    _publish_terminal(spec, outcome)
                    return outcome
                next_stage = STAGES[stage_index + 1]
                next_intent, next_checkpoint = _stage_paths(spec, next_stage)
                if (
                    next_intent.exists()
                    or next_intent.is_symlink()
                    or next_checkpoint.exists()
                    or next_checkpoint.is_symlink()
                ):
                    error = TerminalCondition(
                        "Q1_START_SUCCESSOR_STATE_PRESENT_DURING_CHECKPOINT_PENDING",
                        next_stage,
                    )
                    outcome = _terminal_outcome(spec, next_stage, [*completed, stage], error)
                    _publish_terminal(spec, outcome)
                    return outcome
                return {
                    "schema_version": OUTCOME_SCHEMA,
                    "status": PENDING_STATUS,
                    **spec.binding(),
                    "next_stage": next_stage,
                    "completed_stages": [*completed, stage],
                    "pending_code": pending.code,
                    "pending_detail": pending.detail,
                    "next_actor_intent_absent": True,
                    "actor_popen_count_this_entry": actor_popen_count_this_entry,
                    "resume_allowed": True,
                }
            except PendingCondition as pending:
                # A PendingCondition before exact durable-checkpoint replay is
                # not a clean boundary and cannot be represented as resumable.
                error = TerminalCondition(
                    "Q1_START_CHECKPOINT_REPLAY_NOT_CLEAN",
                    f"{pending.code}: {pending.detail}",
                )
                outcome = _terminal_outcome(spec, stage, completed, error)
                _publish_terminal(spec, outcome)
                return outcome
            except Q1StartSupervisorError as error:
                outcome = _terminal_outcome(spec, stage, completed, error)
                _publish_terminal(spec, outcome)
                return outcome
            checkpoints.append(checkpoint)
            completed.append(stage)
            continue
        if intent_path.exists():
            error = TerminalCondition("Q1_START_INTENT_WITHOUT_CLEAN_CHECKPOINT", stage)
            outcome = _terminal_outcome(spec, stage, completed, error)
            _publish_terminal(spec, outcome)
            return outcome
        if any(_stage_paths(spec, later)[0].exists() or _stage_paths(spec, later)[1].exists() for later in STAGES[STAGES.index(stage) + 1 :]):
            error = TerminalCondition("Q1_START_STAGE_ORDER_MISMATCH", stage)
            outcome = _terminal_outcome(spec, stage, completed, error)
            _publish_terminal(spec, outcome)
            return outcome
        prepared = False
        try:
            _validate_prestart(spec)
            if stage_index > 0:
                predecessor_stage = STAGES[stage_index - 1]
                predecessor = _replay_checkpoint(
                    spec,
                    predecessor_stage,
                    commands[predecessor_stage],
                    endpoint_replayer,
                )
                if not checkpoints or predecessor != checkpoints[-1]:
                    _terminal("Q1_START_PREDECESSOR_REPLAY_MISMATCH", stage)
            stage_bindings = _replay_stage_bindings(stage, spec)
            preflight(stage, spec)
            _prepare_stage_roots(spec, stage)
            prepared = True
            # Replay immediately before the durable intent/Popen boundary too;
            # injected preflight hooks cannot bypass source/runtime closure.
            _validate_prestart(spec)
            if stage_index > 0:
                predecessor_stage = STAGES[stage_index - 1]
                predecessor = _replay_checkpoint(
                    spec,
                    predecessor_stage,
                    commands[predecessor_stage],
                    endpoint_replayer,
                )
                if predecessor != checkpoints[-1]:
                    _terminal("Q1_START_PREDECESSOR_REPLAY_MISMATCH", stage)
            stage_bindings = _replay_stage_bindings(stage, spec)
        except PendingCondition as pending:
            if prepared and not intent_path.exists():
                try:
                    _cleanup_preintent_stage_roots(spec, stage)
                except Q1StartSupervisorError as cleanup_error:
                    outcome = _terminal_outcome(spec, stage, completed, cleanup_error)
                    _publish_terminal(spec, outcome)
                    return outcome
            return {
                "schema_version": OUTCOME_SCHEMA,
                "status": PENDING_STATUS,
                **spec.binding(),
                "next_stage": stage,
                "completed_stages": completed,
                "pending_code": pending.code,
                "pending_detail": pending.detail,
                "next_actor_intent_absent": not intent_path.exists(),
                "actor_popen_count_this_entry": actor_popen_count_this_entry,
                "resume_allowed": True,
            }
        except Q1StartSupervisorError as error:
            outcome = _terminal_outcome(spec, stage, completed, error)
            _publish_terminal(spec, outcome)
            return outcome
        intent = _intent_object(spec, stage, commands[stage])
        try:
            _publish_exclusive(intent_path, intent)
        except OSError as error:
            if intent_path.exists() or intent_path.is_symlink():
                outcome = _terminal_outcome(
                    spec,
                    stage,
                    completed,
                    TerminalCondition(
                        "Q1_START_INTENT_DURABILITY_UNKNOWN",
                        f"{stage}: {error}",
                    ),
                )
                _publish_terminal(spec, outcome)
                return outcome
            try:
                _cleanup_preintent_stage_roots(spec, stage)
            except Q1StartSupervisorError as cleanup_error:
                outcome = _terminal_outcome(spec, stage, completed, cleanup_error)
                _publish_terminal(spec, outcome)
                return outcome
            return {
                "schema_version": OUTCOME_SCHEMA,
                "status": PENDING_STATUS,
                **spec.binding(),
                "next_stage": stage,
                "completed_stages": completed,
                "pending_code": "Q1_START_INTENT_PUBLICATION_PENDING",
                "pending_detail": f"{stage}: {error}",
                "next_actor_intent_absent": True,
                "actor_popen_count_this_entry": actor_popen_count_this_entry,
                "resume_allowed": True,
            }
        try:
            if _read_canonical_file(intent_path, f"{stage} intent") != intent:
                _terminal("Q1_START_INTENT_DURABILITY_UNKNOWN", stage)
            actor_popen_count_this_entry += 1
            process = launcher(stage, commands[stage], spec)
            if stage == "host":
                result, validation, tree_bytes = _validate_host(process, spec)
            else:
                result, validation, tree_bytes = _validate_endpoint(stage, process, spec, endpoint_replayer)
            validation = {
                **validation,
                "stage_binding_replay": stage_bindings,
            }
            _scratch_clean_and_remove(spec, stage)
            _freeze_tree(spec.output_for(stage))
            checkpoint = _checkpoint_object(spec, stage, commands[stage], result, validation, tree_bytes)
            _publish_exclusive(checkpoint_path, checkpoint)
            if _read_canonical_file(checkpoint_path, f"{stage} checkpoint") != checkpoint:
                _terminal("Q1_START_CHECKPOINT_DURABILITY_UNKNOWN", stage)
        except BaseException as caught:
            error = caught if isinstance(caught, Q1StartSupervisorError) else TerminalCondition(
                "Q1_START_POST_INTENT_EXCEPTION",
                f"{type(caught).__name__}: {caught}",
            )
            outcome = _terminal_outcome(spec, stage, completed, error)
            _publish_terminal(spec, outcome)
            return outcome
        checkpoints.append(checkpoint)
        completed.append(stage)
    try:
        outcome = _success_outcome(spec, checkpoints)
    except Q1StartSupervisorError as error:
        outcome = _terminal_outcome(spec, "host", completed, error)
        _publish_terminal(spec, outcome)
        return outcome
    outcome_path = spec.checkpoint_root / "supervisor-outcome.json"
    if outcome_path.exists():
        if _read_canonical_file(outcome_path, "supervisor outcome") != outcome:
            _reject("REJECT_Q1_START_OUTCOME_REPLAY", "success outcome differs")
    else:
        _publish_exclusive(outcome_path, outcome)
    return outcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    spec: Q1StartRunSpecV1 | None = None
    try:
        spec = load_run_spec_v1(arguments.run_spec.read_bytes())
        outcome = run_supervisor_v1(spec)
    except PendingCondition as pending:
        if spec is None:
            # load_run_spec_v1 is validation-only and cannot emit PENDING.  Keep
            # this branch terminal if that invariant ever changes, because no
            # exact campaign binding would be available for a resumable wire.
            outcome = {
                "schema_version": OUTCOME_SCHEMA,
                "status": TERMINAL_STATUS,
                "error_code": "REJECT_Q1_START_UNBOUND_PENDING",
                "error_detail": f"{pending.code}: {pending.detail}",
                "resume_allowed": False,
                "q1_completion_receipt_issued": False,
                "state_published": False,
            }
        else:
            python_intent, python_checkpoint = _stage_paths(spec, "python")
            if (
                python_intent.exists()
                or python_intent.is_symlink()
                or python_checkpoint.exists()
                or python_checkpoint.is_symlink()
            ):
                outcome = _terminal_outcome(
                    spec,
                    "python",
                    [],
                    TerminalCondition(
                        "Q1_START_TOP_LEVEL_PENDING_NOT_CLEAN",
                        f"{pending.code}: {pending.detail}",
                    ),
                )
            else:
                outcome = {
                    "schema_version": OUTCOME_SCHEMA,
                    "status": PENDING_STATUS,
                    **spec.binding(),
                    "next_stage": "python",
                    "completed_stages": [],
                    "pending_code": pending.code,
                    "pending_detail": pending.detail,
                    "next_actor_intent_absent": True,
                    "actor_popen_count_this_entry": 0,
                    "resume_allowed": True,
                }
    except Q1StartSupervisorError as error:
        outcome = {
            "schema_version": OUTCOME_SCHEMA,
            "status": TERMINAL_STATUS,
            "error_code": error.code,
            "error_detail": error.detail,
            "resume_allowed": False,
            "q1_completion_receipt_issued": False,
            "state_published": False,
        }
    except OSError as error:
        outcome = {
            "schema_version": OUTCOME_SCHEMA,
            "status": TERMINAL_STATUS,
            "error_code": "REJECT_Q1_START_RUN_SPEC_IO",
            "error_detail": str(error),
            "resume_allowed": False,
            "q1_completion_receipt_issued": False,
            "state_published": False,
        }
    sys.stdout.buffer.write(_canonical(outcome))
    if outcome.get("status") == PENDING_STATUS:
        return 75
    return 0 if outcome.get("status") == SUCCESS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ActorProcessResultV1",
    "PendingCondition",
    "Q1StartRunSpecV1",
    "Q1StartSupervisorError",
    "RuntimeLimitV1",
    "SourceManifestRowV1",
    "actual_launcher_v1",
    "default_endpoint_replayer_v1",
    "default_preflight_v1",
    "docker_commands_v1",
    "load_run_spec_v1",
    "main",
    "run_supervisor_v1",
]
