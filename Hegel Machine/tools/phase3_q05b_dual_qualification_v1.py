#!/usr/bin/env python3
"""Offline three-actor supervisor for Phase-3A Q0.5b.

``--dry-run`` remains zero-authority and read-only.  ``--run`` is implemented
but admits only one supervisor-owned attempt after every frozen Commit-A
precondition is replayed from a completely clean, explicitly requested full
40-hex commit.  No receipt or artifact exists merely because the entrypoint is
implemented.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha1, sha256
import importlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import types
from typing import Callable, Final, Iterable, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE_PATH: Final = "config/phase3_q05b_dual_isolation_v1.json"
CONFIG_PATH: Final = PROJECT_ROOT / CONFIG_RELATIVE_PATH
ACTUAL_ARTIFACT_RELATIVE_PATH: Final = (
    "artifacts/phase3_q05b_dual_qualification_v1.json"
)
ACTUAL_TEMPORARY_PARENT: Final = Path("/tmp")
PLAN_SCHEMA_VERSION: Final = "hegel-phase3a-q05b-dual-qualification-plan/1"
ERROR_SCHEMA_VERSION: Final = "hegel-phase3a-q05b-dual-qualification-error/1"
PROFILE_ID: Final = "hegel-phase3a-q05b-three-actor-offline-qualification-v1"
STATUS_DRY_RUN: Final = "DRY_RUN_VALIDATED_NOT_EXECUTED"
STDOUT_SET_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-sealed-actor-stdout-set/1"
)
RESOURCE_TRANSCRIPT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-live-container-resource-transcript/1"
)
HELD_ACTOR_WRAPPER_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-held-actor-wrapper/1"
)
HOST_CONTROL_STDOUT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-host-semantic-control-envelope/1"
)
HOST_CONTROL_STDOUT_STATUS: Final = (
    "HOST_SEMANTIC_WITNESS_EMITTED_NOT_RECEIPT"
)
HOST_SEMANTIC_WITNESS_RELATIVE_PATH: Final = "host-semantic-witness.json"
HOST_STAGED_SIDECAR_ROOT: Final = "sidecars"
HELD_ACTOR_WRAPPER_SCRIPT: Final = (
    "set -eu; umask 077; "
    "test \"$#\" -gt 0; "
    "for f in actor.stdout exit-code done release; do test ! -e \"/control/$f\"; done; "
    "set +e; \"$@\" > /control/actor.stdout; actor_status=$?; set -e; "
    "cat /control/actor.stdout; "
    "printf '%s\\n' \"$actor_status\" > /control/exit-code; "
    "printf 'ACTOR_COMPLETE_HELD\\n' > /control/done; "
    "while test ! -f /control/release; do sleep 1; done; "
    "exit \"$actor_status\""
)
HELD_DONE_BYTES: Final = b"ACTOR_COMPLETE_HELD\n"
HELD_SUCCESS_EXIT_BYTES: Final = b"0\n"
HELD_RELEASE_BYTES: Final = b"HOST_FINAL_SAMPLE_SEALED\n"
ACTUAL_ACTOR_LIVE_MOUNT_SOURCE_REPLAY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ACTOR_LIVE_MOUNT_SOURCE_REPLAY/V1\x00"
)

DOCKER_EXECUTABLE: Final = "/usr/bin/docker"
DOCKER_HOST: Final = "unix:///var/run/docker.sock"
DOCKER_EXECUTION_NAMESPACE_LABEL: Final = (
    "org.hegel.q05b.execution_namespace"
)
DOCKER_EXECUTION_SLOT_LABEL: Final = "org.hegel.q05b.slot"
DOCKER_EXECUTION_SOURCE_COMMIT_LABEL: Final = (
    "org.hegel.q05b.source_commit"
)
DOCKER_EXECUTION_RESERVED_LABEL_KEYS: Final = (
    DOCKER_EXECUTION_NAMESPACE_LABEL,
    DOCKER_EXECUTION_SLOT_LABEL,
    DOCKER_EXECUTION_SOURCE_COMMIT_LABEL,
)
DOCKER_PRECREATE_ABSENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/PRECREATE_ABSENCE/V1\x00"
)
DOCKER_OWNERSHIP_LABEL_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/OWNERSHIP_LABELS/V1\x00"
)
DOCKER_OWNED_INSPECT_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/OWNED_INSPECT/V1\x00"
)
DOCKER_EXECUTION_SLOT_REGISTRY: Final = (
    (1, "RUST_TEST"),
    (2, "RUST_RELEASE"),
    (3, "PYTHON_ENDPOINT"),
    (4, "RUST_ENDPOINT"),
    (5, "TRUSTED_HOST_REPLAY"),
)
PYTHON_IMAGE: Final = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
RUST_IMAGE: Final = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
PINNED_IMAGE_BASE_LABELS: Final = {
    PYTHON_IMAGE: {},
    RUST_IMAGE: {
        "org.opencontainers.image.source": "https://github.com/rust-lang/docker-rust"
    },
}
PYTHON_RUNTIME_ENVIRONMENT: Final = {
    "GPG_KEY": "A035C8C19219BA821ECEA86B64E628F8D684696D",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHON_SHA256": "272179ddd9a2e41a0fc8e42e33dfbdca0b3711aa5abf372d3f2d51543d09b625",
    "PYTHON_VERSION": "3.11.15",
    "TZ": "UTC",
}
RUST_RUNTIME_ENVIRONMENT: Final = {
    "CARGO_HOME": "/usr/local/cargo",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "RUSTUP_HOME": "/usr/local/rustup",
    "RUST_VERSION": "1.88.0",
    "TZ": "UTC",
}
RUNTIME_SECCOMP_RELATIVE_PATH: Final = (
    "config/phase3_internal_actor_seccomp_v1.json"
)
BUILD_SECCOMP_RELATIVE_PATH: Final = (
    "config/phase3_m3_offline_build_seccomp_v1.json"
)
RUNTIME_SECCOMP_SHA256: Final = (
    "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
)
BUILD_SECCOMP_SHA256: Final = (
    "41ebe406e2ea5c30572db12566e8c52b2986d23e8b1d6523dff00f5f7fce01fe"
)

ROLE_ROWS: Final = (
    (1, "PYTHON_ENDPOINT", "0-11", "ENDPOINTS_PARALLEL", "14g", "14g", 128, "256:256"),
    (2, "RUST_ENDPOINT", "12-23", "ENDPOINTS_PARALLEL", "14g", "14g", 128, "256:256"),
    (3, "TRUSTED_HOST_REPLAY", "0-11", "AFTER_BOTH_ENDPOINTS_EXIT", "14g", "14g", 128, "256:256"),
)
EXECUTION_PROTOCOL: Final = (
    (1, "VERIFY_CLEAN_FULL40_COMMIT_AND_PINNED_LOCAL_IMAGES"),
    (2, "MATERIALIZE_DISTINCT_READ_ONLY_GIT_BLOB_SNAPSHOTS"),
    (3, "OFFLINE_RUST_TEST_AND_RELEASE_BUILD"),
    (4, "RUN_PYTHON_AND_PREBUILT_RUST_ENDPOINTS_IN_PARALLEL"),
    (5, "CAPTURE_LIVE_ENDPOINT_RESOURCES_THEN_WAIT_AND_SEAL_OUTPUTS_AND_STDOUT"),
    (6, "RUN_TRUSTED_HOST_REPLAY_AND_CAPTURE_LIVE_HOST_RESOURCES"),
    (7, "REVERIFY_COMMIT_IMAGES_BINARY_SNAPSHOTS_AND_SIDECAR_IDENTITIES"),
    (8, "BUILD_AND_STRICT_REPLAY_19_ROW_CANDIDATE_IN_MEMORY"),
    (9, "ADD_PREDICATE20_WHILE_Q1_AUTHORITY_REMAINS_CLOSED"),
    (10, "ATOMICALLY_PUBLISH_ONE_CANONICAL_ARTIFACT_WITH_FSYNC_AND_NOREPLACE"),
)
BLOCKED_PREDICATES: Final = ()
PENDING_ACTUAL_EVIDENCE_PREDICATE_IDS: Final = tuple(range(1, 21))
PYTHON_SOURCE_ALLOWLIST: Final = (
    "config/phase3_q05b_node3_dual_projection_qualification_v1.json",
    "config/phase3_q1_archive_projection_freeze_v1.json",
    "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md",
    "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink2_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink3_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink4_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink5_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink6_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "src/hegel_machine/phase3_m3_record_wire_v1.py",
    "src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink4_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink5_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "src/hegel_machine/phase3_q05b_wire_qualification_contract_v1.py",
    "src/hegel_machine/phase3_q0_evaluator_v1.py",
    "src/hegel_machine/phase3_q0_input_adapter_v1.py",
    "src/hegel_machine/phase3_q0_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q1_archive_projection_v1.py",
    "src/hegel_machine/phase3_q1_capacity_preflight_v1.py",
    "src/hegel_machine/phase3_q1_external_sort_profile_v1.py",
    "src/hegel_machine/phase3_q1_formal_archive_contract_v1.py",
    "src/hegel_machine/phase3_q1_partition_snapshot_v1.py",
    "src/hegel_machine/phase3_q1_qualification_wire_v1.py",
    "src/hegel_machine/phase3_q1_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q1_semantic_coverage_v1.py",
    "src/hegel_machine/phase3_q1_universe_v1.py",
    "src/hegel_machine/strict_ast_shrink1_v1.py",
    "src/hegel_machine/strict_ast_shrink2_v1.py",
    "src/hegel_machine/strict_ast_shrink3_v1.py",
    "src/hegel_machine/strict_ast_shrink4_v1.py",
    "src/hegel_machine/strict_ast_shrink5_v1.py",
    "src/hegel_machine/strict_ast_shrink6_v1.py",
    "src/hegel_machine/strict_ast_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
    "tools/phase3_q1_python_projection_entrypoint_v1.py",
)
RUST_SOURCE_ALLOWLIST: Final = (
    "config/phase3_q05b_node3_dual_projection_qualification_v1.json",
    "config/phase3_q1_archive_projection_freeze_v1.json",
    "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md",
    "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md",
    "rust/q1_archive_projection_oracle/Cargo.lock",
    "rust/q1_archive_projection_oracle/Cargo.toml",
    "rust/q1_archive_projection_oracle/src/lib.rs",
    "rust/q1_archive_projection_oracle/src/main.rs",
    "rust/q1_archive_projection_oracle/tests/cli.rs",
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
    "src/hegel_machine/phase3_q05b_wire_qualification_contract_v1.py",
    "src/hegel_machine/phase3_q1_archive_projection_v1.py",
    "src/hegel_machine/phase3_q1_external_sort_profile_v1.py",
    "src/hegel_machine/phase3_q1_qualification_wire_v1.py",
)
HOST_SOURCE_ALLOWLIST: Final = tuple(
    sorted(
        set(PYTHON_SOURCE_ALLOWLIST)
        | {
            "config/phase3_internal_actor_seccomp_v1.json",
            "config/phase3_m3_offline_build_seccomp_v1.json",
            "config/phase3_q05b_dual_isolation_v1.json",
            "src/hegel_machine/phase3_q05b_actual_admission_v1.py",
            "src/hegel_machine/phase3_q05b_actual_artifact_v1.py",
            "src/hegel_machine/phase3_q05b_host_replay_v1.py",
            "src/hegel_machine/phase3_q05b_negative_vectors_v1.py",
            "tools/phase3_q05b_dual_qualification_v1.py",
        }
    )
)
ACTOR_SOURCE_ALLOWLISTS: Final = {
    "PYTHON_ENDPOINT": PYTHON_SOURCE_ALLOWLIST,
    "RUST_ENDPOINT": RUST_SOURCE_ALLOWLIST,
    "TRUSTED_HOST_REPLAY": HOST_SOURCE_ALLOWLIST,
}

FAIL_CONFIG = "FAIL_Q05B_DUAL_CONFIG"
FAIL_SOURCE = "FAIL_Q05B_DUAL_SOURCE"
FAIL_POLICY = "FAIL_Q05B_DUAL_POLICY"
FAIL_ACTUAL_NOT_IMPLEMENTED = "FAIL_Q05B_ACTUAL_NOT_IMPLEMENTED"
FAIL_ACTUAL_ADMISSION = "FAIL_Q05B_ACTUAL_ADMISSION"
FAIL_ARTIFACT = "FAIL_Q05B_DUAL_ARTIFACT"


class Q05BDualSupervisorError(RuntimeError):
    """Stable fail-closed dual-supervisor error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        artifact_written: bool = False,
    ) -> None:
        if type(artifact_written) is not bool:
            raise TypeError("artifact_written must be one exact bool")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.artifact_written = artifact_written


def _fail(
    code: str,
    detail: str,
    *,
    artifact_written: bool = False,
) -> NoReturn:
    raise Q05BDualSupervisorError(
        code,
        detail,
        artifact_written=artifact_written,
    )


def _canonical_json_bytes(value: object) -> bytes:
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
    except (TypeError, ValueError) as error:
        _fail(FAIL_POLICY, f"cannot encode canonical JSON: {error}")


@contextmanager
def _docker_ownership_signal_guard_v1():
    """Defer catchable process signals across ownership handoff/cleanup."""

    if not hasattr(signal, "pthread_sigmask"):
        _fail(FAIL_POLICY, "pthread signal masking is unavailable")
    blocked = set(signal.valid_signals())
    blocked.discard(signal.SIGKILL)
    blocked.discard(signal.SIGSTOP)
    try:
        prior = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    except (OSError, ValueError) as error:
        _fail(FAIL_POLICY, f"cannot block ownership handoff signals: {error}")
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, prior)


def _strict_json_object_v1(payload: bytes, name: str) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as error:
        _fail(FAIL_CONFIG, f"{name} is not strict JSON: {error}")
    if type(value) is not dict:
        _fail(FAIL_CONFIG, f"{name} must be one JSON object")
    return value


def _exact_dict(value: object, keys: Iterable[str], name: str) -> dict[str, object]:
    expected = set(keys)
    if type(value) is not dict or set(value) != expected:
        _fail(FAIL_CONFIG, f"{name} fields differ")
    return value


def _require_type_exact_v1(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected):
        _fail(FAIL_CONFIG, f"{name} type differs")
    if type(expected) is dict:
        assert type(value) is dict
        if set(value) != set(expected):
            _fail(FAIL_CONFIG, f"{name} fields differ")
        for key in expected:
            _require_type_exact_v1(value[key], expected[key], f"{name}.{key}")
        return
    if type(expected) in (list, tuple):
        if len(value) != len(expected):  # type: ignore[arg-type]
            _fail(FAIL_CONFIG, f"{name} length differs")
        for index, (item, expected_item) in enumerate(
            zip(value, expected, strict=True)  # type: ignore[arg-type]
        ):
            _require_type_exact_v1(item, expected_item, f"{name}[{index}]")
        return
    if value != expected:
        _fail(FAIL_CONFIG, f"{name} value differs")


def _sha256_file(path: Path, label: str) -> str:
    try:
        status_before = path.lstat()
        payload = path.read_bytes()
        status_after = path.lstat()
    except OSError as error:
        _fail(FAIL_CONFIG, f"cannot read {label}: {error}")
    before = (
        status_before.st_dev,
        status_before.st_ino,
        status_before.st_mode,
        status_before.st_size,
        status_before.st_mtime_ns,
        status_before.st_ctime_ns,
    )
    after = (
        status_after.st_dev,
        status_after.st_ino,
        status_after.st_mode,
        status_after.st_size,
        status_after.st_mtime_ns,
        status_after.st_ctime_ns,
    )
    if (
        path.is_symlink()
        or not stat.S_ISREG(status_before.st_mode)
        or before != after
        or len(payload) != status_before.st_size
    ):
        _fail(FAIL_CONFIG, f"{label} changed or is not a regular file")
    return sha256(payload).hexdigest()


def _strict_json_value_v1(payload: bytes, name: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")
            ),
            parse_float=lambda token: (_ for _ in ()).throw(
                ValueError(f"finite JSON float {token!r}")
            ),
        )
    except (UnicodeError, ValueError) as error:
        _fail(FAIL_POLICY, f"{name} is not strict JSON: {error}")


def _parse_decimal_line_v1(payload: bytes, name: str) -> int:
    if type(payload) is not bytes or re.fullmatch(rb"(?:0|[1-9][0-9]*)\n", payload) is None:
        _fail(FAIL_POLICY, f"{name} is not one canonical decimal line")
    return int(payload[:-1])


def _read_held_control_file_v1(
    directory: int,
    name: str,
    expected_mode: int,
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_size > 1024 * 1024
        ):
            _fail(FAIL_POLICY, f"held control file identity differs: {name}")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            total += len(block)
            blocks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or total != before.st_size:
            _fail(FAIL_POLICY, f"held control file changed: {name}")
        return b"".join(blocks), after
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_held_actor_stdout_v1(payload: bytes, actor_id: str) -> None:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > 1024 * 1024
        or payload.count(b"\n") != 1
        or not payload.endswith(b"\n")
    ):
        _fail(FAIL_POLICY, "held actor stdout framing differs")
    value = _strict_json_object_v1(payload, "held actor stdout")
    expected_implementation = {
        "PYTHON_ENDPOINT": "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
        "RUST_ENDPOINT": "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1",
        "TRUSTED_HOST_REPLAY": "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1",
    }.get(actor_id)
    if actor_id == "TRUSTED_HOST_REPLAY":
        expected_fields = {
            "action_id",
            "actor_id",
            "file_count",
            "final_isolation_root",
            "implementation_id",
            "loaded_module_root",
            "loaded_module_rows",
            "q1_formal_roots",
            "q1_gate_count",
            "q1_gate_mask",
            "q1_output_slots",
            "q1_state",
            "qualification_receipt",
            "runtime_identity_sha256",
            "loaded_module_root",
            "schema_version",
            "semantic_replay_root",
            "source_identity_sha256",
            "status",
            "witness_length",
            "witness_relative_path",
            "witness_root",
            "witness_sha256",
        }
        root_fields = (
            "runtime_identity_sha256",
            "semantic_replay_root",
            "source_identity_sha256",
            "witness_root",
            "witness_sha256",
        )
        if (
            set(value) != expected_fields
            or expected_implementation is None
            or value.get("actor_id") != actor_id
            or value.get("implementation_id") != expected_implementation
            or value.get("schema_version") != HOST_CONTROL_STDOUT_SCHEMA_VERSION
            or value.get("status") != HOST_CONTROL_STDOUT_STATUS
            or value.get("action_id") != "trusted-host-semantic-replay-v1"
            or type(value.get("file_count")) is not int
            or value["file_count"] != 6
            or type(value.get("witness_length")) is not int
            or value["witness_length"] < 1
            or value.get("witness_relative_path")
            != HOST_SEMANTIC_WITNESS_RELATIVE_PATH
            or value.get("q1_state") != "NOT_RUN"
            or type(value.get("q1_gate_count")) is not int
            or value["q1_gate_count"] != 0
            or type(value.get("q1_gate_mask")) is not int
            or value["q1_gate_mask"] != 0
            or value.get("q1_formal_roots") is not None
            or value.get("q1_output_slots") != [None] * 8
            or value.get("qualification_receipt") is not None
            or value.get("final_isolation_root") is not None
            or type(value.get("loaded_module_rows")) is not list
            or not value["loaded_module_rows"]
            or any(
                type(row) is not list
                or len(row) != 3
                or type(row[0]) is not str
                or (row[1] is not None and type(row[1]) is not str)
                or (row[2] is not None and type(row[2]) is not str)
                for row in value["loaded_module_rows"]
            )
            or value.get("loaded_module_root")
            != sha256(
                b"HEGEL/Q05B/HOST/LOADED_MODULE_CLOSURE/V1\x00"
                + _canonical_json_bytes(value["loaded_module_rows"])
            ).hexdigest()
            or any(
                type(value.get(field)) is not str
                or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None
                for field in root_fields
            )
            or _canonical_json_bytes(value) != payload
        ):
            _fail(FAIL_POLICY, "held host control stdout differs")
        return
    if (
        expected_implementation is None
        or value.get("actor_id") != actor_id
        or value.get("implementation_id") != expected_implementation
        or value.get("schema_version") != "hegel-q05b-actor-envelope/1"
        or value.get("status") != "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED"
        or value.get("action_id") != "bounded-node3-golden-v1"
        or type(value.get("file_count")) is not int
        or value["file_count"] != 5
        or value.get("q1_state") != "NOT_RUN"
        or type(value.get("q1_gate_count")) is not int
        or value["q1_gate_count"] != 0
        or type(value.get("q1_gate_mask")) is not int
        or value["q1_gate_mask"] != 0
        or value.get("q1_formal_roots") is not None
        or value.get("q1_output_slots") != [None] * 8
        or _canonical_json_bytes(value) != payload
    ):
        _fail(FAIL_POLICY, "held actor stdout authority/identity differs")


def _held_control_identity_v1(
    control_root: Path,
    names: tuple[str, ...],
    expected_mode: int,
    actor_id: str,
) -> dict[str, object]:
    root_before = control_root.lstat()
    if control_root.is_symlink() or not stat.S_ISDIR(root_before.st_mode):
        _fail(FAIL_POLICY, "held control root differs")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory = os.open(control_root, flags)
    try:
        anchored = os.fstat(directory)
        if (
            (anchored.st_dev, anchored.st_ino)
            != (root_before.st_dev, root_before.st_ino)
            or tuple(sorted(os.listdir(directory))) != names
        ):
            _fail(FAIL_POLICY, "held control root/file set changed")
        rows: list[list[object]] = []
        payloads: dict[str, bytes] = {}
        for name in names:
            payload, status = _read_held_control_file_v1(
                directory,
                name,
                expected_mode,
            )
            payloads[name] = payload
            rows.append(
                [
                    name,
                    status.st_dev,
                    status.st_ino,
                    status.st_nlink,
                    status.st_uid,
                    status.st_gid,
                    stat.S_IMODE(status.st_mode),
                    status.st_size,
                    status.st_mtime_ns,
                    status.st_ctime_ns,
                    sha256(payload).hexdigest(),
                ]
            )
        if payloads.get("exit-code") != HELD_SUCCESS_EXIT_BYTES:
            _fail(FAIL_POLICY, "held actor exit-code differs")
        if payloads.get("done") != HELD_DONE_BYTES:
            _fail(FAIL_POLICY, "held actor done marker differs")
        if "release" in payloads and payloads["release"] != HELD_RELEASE_BYTES:
            _fail(FAIL_POLICY, "held actor release marker differs")
        _validate_held_actor_stdout_v1(payloads["actor.stdout"], actor_id)
        value = {
            "schema_version": "hegel-phase3a-q05b-held-control-evidence/1",
            "actor_id": actor_id,
            "control_root_path": control_root.resolve(strict=True).as_posix(),
            "root_device": anchored.st_dev,
            "root_inode": anchored.st_ino,
            "root_nlink": anchored.st_nlink,
            "root_mode": stat.S_IMODE(anchored.st_mode),
            "file_rows": rows,
            "actor_stdout_hex": payloads["actor.stdout"].hex(),
        }
        value["manifest_sha256"] = sha256(_canonical_json_bytes(value)).hexdigest()
        return value
    finally:
        os.close(directory)


def seal_held_actor_completion_v1(
    control_root: Path,
    actor_id: str,
) -> dict[str, object]:
    if stat.S_IMODE(control_root.lstat().st_mode) != 0o700:
        _fail(FAIL_POLICY, "held control initial directory mode differs")
    preliminary = _held_control_identity_v1(
        control_root,
        ("actor.stdout", "done", "exit-code"),
        0o600,
        actor_id,
    )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(control_root, flags)
    try:
        for name in ("actor.stdout", "done", "exit-code"):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
            )
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory)
    finally:
        os.close(directory)
    sealed = _held_control_identity_v1(
        control_root,
        ("actor.stdout", "done", "exit-code"),
        0o444,
        actor_id,
    )
    if preliminary["actor_stdout_hex"] != sealed["actor_stdout_hex"]:
        _fail(FAIL_POLICY, "held actor stdout changed while sealing")
    return sealed


def held_final_resource_sample_v1(
    control_root: Path,
    completion_evidence: Mapping[str, object],
    role_id: int,
    container_id: str,
    expected_container_name: str,
    mount_registry: "SealedActorMountRegistryV1",
    inspect_before_payload: bytes,
    inspect_after_reader: Callable[[], bytes],
    *,
    seccomp_evidence: Mapping[str, object],
    docker_execution_principal: Mapping[str, object] | None = None,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, object]:
    """Freshly collect the final sample only after sealed child completion."""

    if (
        not isinstance(control_root, Path)
        or type(completion_evidence) is not dict
        or type(completion_evidence.get("actor_id")) is not str
        or type(completion_evidence.get("manifest_sha256")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", completion_evidence["manifest_sha256"]
        )
        is None
    ):
        _fail(FAIL_POLICY, "held final resource sample inputs differ")
    if _held_control_identity_v1(
        control_root,
        ("actor.stdout", "done", "exit-code"),
        0o444,
        completion_evidence["actor_id"],
    ) != completion_evidence:
        _fail(FAIL_POLICY, "held completion changed before final sampling")
    sample_start_ns = time.monotonic_ns()
    result = collect_bound_live_resource_transcript_v1(
        role_id,
        container_id,
        expected_container_name,
        mount_registry,
        inspect_before_payload,
        inspect_after_reader,
        seccomp_evidence=seccomp_evidence,
        docker_execution_principal=docker_execution_principal,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    if result.get("anchored_collection") is not True:
        _fail(FAIL_POLICY, "held final resource sample was not anchored")
    sample_finish_ns = time.monotonic_ns()
    result["sample_ordinal"] = 1
    result["sample_monotonic_ns"] = sample_start_ns
    result["sample_duration_ns"] = sample_finish_ns - sample_start_ns
    result["actor_child_complete_held"] = True
    result["completion_manifest_sha256"] = completion_evidence["manifest_sha256"]
    result["fresh_after_done_collection"] = True
    return result


def release_held_actor_v1(
    control_root: Path,
    completion_evidence: Mapping[str, object],
    held_final_sample: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(completion_evidence) is not dict
        or type(held_final_sample) is not dict
        or held_final_sample.get("actor_child_complete_held") is not True
        or held_final_sample.get("fresh_after_done_collection") is not True
        or held_final_sample.get("anchored_collection") is not True
        or held_final_sample.get("completion_manifest_sha256")
        != completion_evidence.get("manifest_sha256")
    ):
        _fail(FAIL_POLICY, "held actor release ordering evidence differs")
    actor_id = completion_evidence.get("actor_id")
    if type(actor_id) is not str:
        _fail(FAIL_POLICY, "held actor completion identity differs")
    if _held_control_identity_v1(
        control_root,
        ("actor.stdout", "done", "exit-code"),
        0o444,
        actor_id,
    ) != completion_evidence:
        _fail(FAIL_POLICY, "held control changed before release")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(control_root, flags)
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open("release", file_flags, 0o400, dir_fd=directory)
        try:
            os.write(descriptor, HELD_RELEASE_BYTES)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fchmod(directory, 0o555)
        os.fsync(directory)
    finally:
        os.close(directory)
    return _held_control_identity_v1(
        control_root,
        ("actor.stdout", "done", "exit-code", "release"),
        0o444,
        actor_id,
    )


def validate_held_actor_exit_v1(
    control_root: Path,
    release_evidence: Mapping[str, object],
    docker_stdout: bytes,
    docker_exit_code: int,
) -> None:
    if (
        type(release_evidence) is not dict
        or type(release_evidence.get("actor_id")) is not str
        or type(docker_stdout) is not bytes
        or type(docker_exit_code) is not int
        or docker_exit_code != 0
    ):
        _fail(FAIL_POLICY, "held actor final exit evidence differs")
    replay = _held_control_identity_v1(
        control_root,
        ("actor.stdout", "done", "exit-code", "release"),
        0o444,
        release_evidence["actor_id"],
    )
    if replay != release_evidence or bytes.fromhex(replay["actor_stdout_hex"]) != docker_stdout:
        _fail(FAIL_POLICY, "held actor forwarded stdout differs")


@dataclass(frozen=True, slots=True)
class SealedActorMountRegistryV1:
    """Exact bind-source registry mechanically extracted from one Docker command."""

    role_id: int
    command_sha256: str
    mount_rows: tuple[tuple[str, str, bool], ...]
    container_argv: tuple[str, ...]
    security_options: tuple[str, str]
    environment_rows: tuple[tuple[str, str], ...]
    working_directory: str
    registry_sha256: str

    def __post_init__(self) -> None:
        if type(self.role_id) is not int or self.role_id not in (1, 2, 3):
            _fail(FAIL_POLICY, "sealed mount registry role differs")
        for value in (self.command_sha256, self.registry_sha256):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                _fail(FAIL_POLICY, "sealed mount registry hash differs")
        expected_destinations = {
            1: ("/control", "/output", "/snapshot"),
            2: (
                "/control",
                "/output",
                "/runtime/hegel-q1-archive-projection-oracle",
            ),
            3: (
                "/control",
                "/inputs/python",
                "/inputs/rust",
                "/inputs/stdout/manifest.json",
                "/inputs/stdout/python.stdout",
                "/inputs/stdout/rust.stdout",
                "/snapshot",
                "/staging",
            ),
        }[self.role_id]
        if tuple(row[0] for row in self.mount_rows) != expected_destinations:
            _fail(FAIL_POLICY, "sealed mount destination registry differs")
        for destination, source, writable in self.mount_rows:
            if (
                type(destination) is not str
                or type(source) is not str
                or not source.startswith("/")
                or ".." in PurePosixPath(source).parts
                or "docker.sock" in source
                or type(writable) is not bool
            ):
                _fail(FAIL_POLICY, "sealed mount row differs")
        expected_environment = (
            PYTHON_RUNTIME_ENVIRONMENT
            if self.role_id in (1, 3)
            else RUST_RUNTIME_ENVIRONMENT
        )
        if (
            type(self.container_argv) is not tuple
            or not self.container_argv
            or any(type(item) is not str or not item for item in self.container_argv)
            or type(self.security_options) is not tuple
            or len(self.security_options) != 2
            or self.security_options[0] != "no-new-privileges"
            or not self.security_options[1].startswith("seccomp=/")
            or self.environment_rows != tuple(sorted(expected_environment.items()))
            or type(self.working_directory) is not str
            or self.working_directory != ("/snapshot" if self.role_id in (1, 3) else "")
        ):
            _fail(FAIL_POLICY, "sealed actor command policy registry differs")
        body = {
            "command_sha256": self.command_sha256,
            "container_argv": list(self.container_argv),
            "environment_rows": [list(row) for row in self.environment_rows],
            "mount_rows": [list(row) for row in self.mount_rows],
            "role_id": self.role_id,
            "schema_version": "hegel-phase3a-q05b-sealed-command-mount-registry/1",
            "security_options": list(self.security_options),
            "working_directory": self.working_directory,
        }
        if sha256(_canonical_json_bytes(body)).hexdigest() != self.registry_sha256:
            _fail(FAIL_POLICY, "sealed mount registry root differs")

    @property
    def expected_sources(self) -> dict[str, str]:
        return {destination: source for destination, source, _ in self.mount_rows}


def sealed_actor_mount_registry_v1(
    role_id: int,
    command: Sequence[str],
) -> SealedActorMountRegistryV1:
    """Derive expected bind sources from the exact command admitted by outer."""

    if (
        type(role_id) is not int
        or role_id not in (1, 2, 3)
        or type(command) not in (tuple, list)
        or any(type(item) is not str or not item for item in command)
    ):
        _fail(FAIL_POLICY, "sealed command mount registry input differs")
    observed: dict[str, tuple[str, bool]] = {}
    for index, item in enumerate(command):
        if item != "--mount":
            continue
        if index + 1 >= len(command):
            _fail(FAIL_POLICY, "Docker command mount option is truncated")
        specification = command[index + 1]
        match = re.fullmatch(
            r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?",
            specification,
        )
        if match is None:
            _fail(FAIL_POLICY, "Docker command bind specification differs")
        source, destination, read_only = match.groups()
        if destination in observed:
            _fail(FAIL_POLICY, "Docker command repeats a bind destination")
        observed[destination] = (source, read_only is None)
    rows = tuple(
        (destination, observed[destination][0], observed[destination][1])
        for destination in sorted(observed)
    )
    expected_image = PYTHON_IMAGE if role_id in (1, 3) else RUST_IMAGE
    image_indexes = tuple(
        index for index, item in enumerate(command) if item == expected_image
    )
    if len(image_indexes) != 1 or image_indexes[0] == len(command) - 1:
        _fail(FAIL_POLICY, "Docker command image/payload boundary differs")
    container_argv = tuple(command[image_indexes[0] + 1 :])
    security_options = tuple(
        item.removeprefix("--security-opt=")
        for item in command
        if item.startswith("--security-opt=")
    )
    work_indexes = tuple(index for index, item in enumerate(command) if item == "-w")
    if len(work_indexes) > 1 or (
        work_indexes and work_indexes[0] + 1 >= len(command)
    ):
        _fail(FAIL_POLICY, "Docker command working directory differs")
    working_directory = command[work_indexes[0] + 1] if work_indexes else ""
    environment_rows = tuple(
        sorted(
            (
                PYTHON_RUNTIME_ENVIRONMENT
                if role_id in (1, 3)
                else RUST_RUNTIME_ENVIRONMENT
            ).items()
        )
    )
    command_sha256 = sha256(
        _canonical_json_bytes(list(command))
    ).hexdigest()
    body = {
        "command_sha256": command_sha256,
        "container_argv": list(container_argv),
        "environment_rows": [list(row) for row in environment_rows],
        "mount_rows": [list(row) for row in rows],
        "role_id": role_id,
        "schema_version": "hegel-phase3a-q05b-sealed-command-mount-registry/1",
        "security_options": list(security_options),
        "working_directory": working_directory,
    }
    return SealedActorMountRegistryV1(
        role_id,
        command_sha256,
        rows,
        container_argv,
        security_options,
        environment_rows,
        working_directory,
        sha256(_canonical_json_bytes(body)).hexdigest(),
    )


def actor_mount_registry_object_v1(
    registry: SealedActorMountRegistryV1,
) -> dict[str, object]:
    if type(registry) is not SealedActorMountRegistryV1:
        _fail(FAIL_ACTUAL_ADMISSION, "actor mount registry object input differs")
    return {
        "schema_version": "hegel-phase3a-q05b-sealed-command-mount-registry/1",
        "role_id": registry.role_id,
        "command_sha256": registry.command_sha256,
        "mount_rows": [list(row) for row in registry.mount_rows],
        "container_argv": list(registry.container_argv),
        "security_options": list(registry.security_options),
        "environment_rows": [list(row) for row in registry.environment_rows],
        "working_directory": registry.working_directory,
        "registry_sha256": registry.registry_sha256,
    }


@dataclass(slots=True)
class HeldActorMountSourceV1:
    destination: str
    source: Path
    descriptor: int
    source_row: dict[str, object]
    expected_payload_sha256: str | None
    pre_stat_signature: tuple[int, ...]
    close_state: str = "HELD"


@dataclass(slots=True)
class HeldActorMountBindingV1:
    role_id: int
    actor_id: str
    exact_command: tuple[str, ...]
    registry: SealedActorMountRegistryV1
    binding: dict[str, object]
    sources: tuple[HeldActorMountSourceV1, ...]
    seccomp: HeldActorMountSourceV1
    closed: bool = False


def _read_held_mount_file_v1(descriptor: int, maximum: int) -> bytes:
    if (
        type(descriptor) is not int
        or type(maximum) is not int
        or maximum < 1
        or (fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE)
        != os.O_RDONLY
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "held mount file descriptor differs")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > maximum:
                _fail(FAIL_ACTUAL_ADMISSION, "held mount file exceeds bound")
        return b"".join(blocks)
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)


def _held_mount_source_replay_v1(
    source: HeldActorMountSourceV1,
    *,
    after_start: bool = False,
) -> dict[str, object]:
    if type(source) is not HeldActorMountSourceV1 or type(after_start) is not bool:
        _fail(FAIL_ACTUAL_ADMISSION, "held mount source replay input differs")
    row = source.source_row
    try:
        held = os.fstat(source.descriptor)
        path = source.source.lstat()
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"held mount source replay failed: {error}")
    source_type = row["source_type"]
    expected_type = stat.S_ISDIR if source_type == "DIRECTORY" else stat.S_ISREG
    if (
        not expected_type(held.st_mode)
        or not expected_type(path.st_mode)
        or (held.st_dev, held.st_ino) != (row["source_device"], row["source_inode"])
        or (path.st_dev, path.st_ino) != (held.st_dev, held.st_ino)
        or held.st_nlink != path.st_nlink
        or (
            not (
                after_start
                and row["writable"] is True
                and source_type == "DIRECTORY"
            )
            and held.st_nlink != row["source_nlink"]
        )
        or (
            source_type == "DIRECTORY" and held.st_nlink < 2
        )
        or held.st_uid != row["source_uid"]
        or held.st_gid != row["source_gid"]
        or path.st_uid != row["source_uid"]
        or path.st_gid != row["source_gid"]
        or stat.S_IMODE(held.st_mode) != row["source_mode"]
        or stat.S_IMODE(path.st_mode) != row["source_mode"]
        or (fcntl.fcntl(source.descriptor, fcntl.F_GETFL) & os.O_ACCMODE)
        != os.O_RDONLY
        or (
            after_start
            and row["writable"] is False
            and (
                held.st_dev,
                held.st_ino,
                held.st_mode,
                held.st_nlink,
                held.st_uid,
                held.st_gid,
                held.st_size,
                held.st_mtime_ns,
                held.st_ctime_ns,
            )
            != source.pre_stat_signature
        )
        or (
            after_start
            and row["writable"] is True
            and (held.st_uid, held.st_gid)
            != (
                source.pre_stat_signature[4],
                source.pre_stat_signature[5],
            )
        )
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "held mount source identity changed")
    payload_sha256: str | None = None
    if source_type == "REGULAR_FILE":
        payload = _read_held_mount_file_v1(
            source.descriptor,
            max(1, int(row["source_size"])),
        )
        payload_sha256 = sha256(payload).hexdigest()
        if (
            len(payload) != row["source_size"]
            or payload_sha256 != source.expected_payload_sha256
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "held mount file payload changed")
    return {
        "destination": source.destination,
        "source": source.source.as_posix(),
        "source_device": held.st_dev,
        "source_inode": held.st_ino,
        "source_nlink": held.st_nlink,
        "source_uid": held.st_uid,
        "source_gid": held.st_gid,
        "source_mode": stat.S_IMODE(held.st_mode),
        "source_type": source_type,
        "payload_sha256": payload_sha256,
        "path_matches_held_descriptor": True,
        "held_descriptor_read_only": True,
    }


def _open_held_actor_mount_source_v1(
    role_id: int,
    destination: str,
    source_path: Path,
    writable: bool,
    source_type: str,
    source_mode: int,
    authority_kind: str,
    authority_label: str,
    authority_evidence: Mapping[str, object],
    *,
    expected_identity: tuple[
        int, int, int, int | None, int | None, int, int | None
    ],
    expected_payload_sha256: str | None = None,
    require_empty_directory: bool = False,
    ownership_sink: Callable[[HeldActorMountSourceV1], None],
) -> HeldActorMountSourceV1:
    if (
        type(role_id) is not int
        or role_id not in (1, 2, 3)
        or type(destination) is not str
        or not destination
        or not isinstance(source_path, Path)
        or not source_path.is_absolute()
        or type(writable) is not bool
        or source_type not in ("DIRECTORY", "REGULAR_FILE")
        or type(source_mode) is not int
        or type(expected_identity) is not tuple
        or len(expected_identity) != 7
        or type(require_empty_directory) is not bool
        or not callable(ownership_sink)
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "held mount source input differs")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if source_type == "DIRECTORY":
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        path_before = source_path.lstat()
        descriptor = os.open(source_path, flags)
        held = os.fstat(descriptor)
        path_after = source_path.lstat()
        (
            expected_dev,
            expected_ino,
            expected_nlink,
            expected_uid,
            expected_gid,
            expected_mode,
            expected_size,
        ) = expected_identity
        expected_type = stat.S_ISDIR if source_type == "DIRECTORY" else stat.S_ISREG
        if (
            source_path.is_symlink()
            or source_path.resolve(strict=True) != source_path
            or not expected_type(path_before.st_mode)
            or not expected_type(held.st_mode)
            or not expected_type(path_after.st_mode)
            or (held.st_dev, held.st_ino)
            != (expected_dev, expected_ino)
            or (path_before.st_dev, path_before.st_ino)
            != (held.st_dev, held.st_ino)
            or (path_after.st_dev, path_after.st_ino)
            != (held.st_dev, held.st_ino)
            or held.st_nlink != expected_nlink
            or path_after.st_nlink != expected_nlink
            or (expected_uid is not None and held.st_uid != expected_uid)
            or (expected_gid is not None and held.st_gid != expected_gid)
            or stat.S_IMODE(held.st_mode) != expected_mode
            or expected_mode != source_mode
            or (
                source_type == "REGULAR_FILE"
                and (held.st_size != expected_size or held.st_nlink != 1)
            )
            or (
                source_type == "DIRECTORY"
                and (held.st_nlink < 2 or expected_size is not None)
            )
            or (fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE)
            != os.O_RDONLY
            or (
                require_empty_directory
                and (source_type != "DIRECTORY" or os.listdir(descriptor))
            )
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "held mount source admission differs")
        source_size = held.st_size if source_type == "REGULAR_FILE" else None
        row = _ACTUAL_ADMISSION.build_actor_mount_source_row_v1(
            role_id,
            destination,
            source_path.as_posix(),
            writable,
            source_type,
            held.st_dev,
            held.st_ino,
            held.st_nlink,
            held.st_uid,
            held.st_gid,
            stat.S_IMODE(held.st_mode),
            source_size,
            authority_kind,
            authority_label,
            authority_evidence,
        )
        value = HeldActorMountSourceV1(
            destination,
            source_path,
            descriptor,
            row,
            expected_payload_sha256,
            (
                held.st_dev,
                held.st_ino,
                held.st_mode,
                held.st_nlink,
                held.st_uid,
                held.st_gid,
                held.st_size,
                held.st_mtime_ns,
                held.st_ctime_ns,
            ),
        )
        _held_mount_source_replay_v1(value)
        with _docker_ownership_signal_guard_v1():
            ownership_sink(value)
            descriptor = None
        return value
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"cannot anchor actor mount source: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _close_held_mount_sources_best_effort_v1(
    sources: Sequence[HeldActorMountSourceV1],
) -> tuple[str, ...]:
    errors: list[str] = []
    for source in sources:
        if type(source) is not HeldActorMountSourceV1:
            errors.append("mount source close type differs")
            continue
        descriptor = source.descriptor
        if descriptor == -1:
            if source.close_state == "CLOSED":
                continue
            if source.close_state == "UNCERTAIN_CLOSE":
                errors.append(
                    f"uncertain close already recorded: {source.destination}"
                )
                continue
            errors.append(
                f"mount source descriptor/state differs: {source.destination}"
            )
            continue
        if (
            type(descriptor) is not int
            or descriptor < 0
            or source.close_state != "HELD"
        ):
            errors.append(f"invalid descriptor: {descriptor!r}")
            continue
        # Transfer the numeric descriptor out of the object before the single
        # close attempt.  A successful close may make the same integer
        # immediately reusable by another thread; probing or retrying it could
        # therefore close an unrelated file.
        source.descriptor = -1
        source.close_state = "CLOSE_ATTEMPTED"
        try:
            os.close(descriptor)
        except OSError as error:
            source.close_state = "UNCERTAIN_CLOSE"
            errors.append(
                f"descriptor {descriptor} uncertain close: {error}"
            )
        else:
            source.close_state = "CLOSED"
    return tuple(errors)


def close_held_actor_mount_binding_v1(binding: HeldActorMountBindingV1) -> None:
    if type(binding) is not HeldActorMountBindingV1:
        _fail(FAIL_ACTUAL_ADMISSION, "held actor mount close input differs")
    if binding.closed:
        return
    errors = _close_held_mount_sources_best_effort_v1(
        (*binding.sources, binding.seccomp)
    )
    binding.closed = all(
        source.descriptor == -1 and source.close_state == "CLOSED"
        for source in (*binding.sources, binding.seccomp)
    )
    if errors:
        _fail(FAIL_ACTUAL_ADMISSION, "held actor mount close failed: " + errors[0])
    if not binding.closed:
        _fail(FAIL_ACTUAL_ADMISSION, "held actor mount close left a live descriptor")


def _close_actor_mount_bindings_best_effort_v1(
    bindings: Sequence[HeldActorMountBindingV1],
) -> tuple[str, ...]:
    errors: list[str] = []
    for binding in bindings:
        try:
            close_held_actor_mount_binding_v1(binding)
        except BaseException as error:
            actor_id = (
                binding.actor_id
                if type(binding) is HeldActorMountBindingV1
                else "UNKNOWN_ACTOR"
            )
            errors.append(f"{actor_id}:{type(error).__name__}:{error}")
    return tuple(errors)


def replay_held_actor_mount_binding_after_start_v1(
    binding: HeldActorMountBindingV1,
    actor: "HeldActorProcessV1",
) -> dict[str, object]:
    if (
        type(binding) is not HeldActorMountBindingV1
        or binding.closed
        or type(actor) is not HeldActorProcessV1
        or actor.role_id != binding.role_id
        or actor.actor_id != binding.actor_id
        or actor.command != binding.exact_command
        or actor.mount_registry != binding.registry
        or actor.seccomp_evidence is None
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "started actor mount authority differs")
    expected_seccomp_authority = binding.seccomp.source_row["authority_root"]
    if _ACTUAL_ADMISSION.actor_mount_authority_root_v1(
        "RUNTIME_SECCOMP_POLICY",
        f"{binding.actor_id}/@seccomp",
        actor.seccomp_evidence,
    ) != expected_seccomp_authority:
        _fail(FAIL_ACTUAL_ADMISSION, "started actor seccomp authority differs")
    source_replays = [
        _held_mount_source_replay_v1(source, after_start=True)
        for source in binding.sources
    ]
    seccomp_replay = _held_mount_source_replay_v1(
        binding.seccomp, after_start=True
    )
    try:
        return _ACTUAL_ADMISSION.build_actor_mount_launch_replay_v1(
            binding.binding,
            source_replays,
            seccomp_replay,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def strict_replay_actor_completion_mount_sources_v1(
    completion: Mapping[str, object],
    mount_binding: Mapping[str, object],
    launch_replay: Mapping[str, object],
) -> dict[str, object]:
    """Replay every live/post-exit Docker Mount.Source against one binding."""

    try:
        binding = _ACTUAL_ADMISSION.validate_actor_mount_binding_v1(
            mount_binding
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    if (
        type(completion) is not dict
        or type(launch_replay) is not dict
        or completion.get("actor_id") != binding["actor_id"]
        or completion.get("command_sha256")
        != binding["command_mount_registry"]["command_sha256"]
        or completion.get("mount_registry_sha256")
        != binding["command_mount_registry"]["registry_sha256"]
        or launch_replay.get("role_id") != binding["role_id"]
        or launch_replay.get("actor_id") != binding["actor_id"]
        or launch_replay.get("mount_binding_root")
        != binding["mount_binding_root"]
        or type(launch_replay.get("launch_replay_root")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", launch_replay["launch_replay_root"]
        )
        is None
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "actor completion mount identity differs")
    final_resource = completion.get("final_resource_transcript")
    if (
        type(final_resource) is not dict
        or type(final_resource.get("live_sample_objects")) is not list
        or not final_resource["live_sample_objects"]
        or type(final_resource.get("post_exit_inspect_hex")) is not str
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "actor completion mount preimages differ")
    expected_rows = binding["command_mount_registry"]["mount_rows"]
    expected = {
        row[0]: {"Source": row[1], "RW": row[2], "Type": "bind"}
        for row in expected_rows
    }
    payload_rows: list[list[object]] = []

    def replay_payload(payload_hex: object, label: str) -> None:
        if (
            type(payload_hex) is not str
            or len(payload_hex) % 2
            or re.fullmatch(r"[0-9a-f]+", payload_hex) is None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "actor live mount inspect hex differs")
        payload = bytes.fromhex(payload_hex)
        document = _strict_json_value_v1(payload, label)
        if (
            type(document) is not list
            or len(document) != 1
            or type(document[0]) is not dict
            or type(document[0].get("Mounts")) is not list
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "actor live mount inspect shape differs")
        observed: dict[str, dict[str, object]] = {}
        for row in document[0]["Mounts"]:
            if (
                type(row) is not dict
                or type(row.get("Destination")) is not str
                or type(row.get("Source")) is not str
                or type(row.get("RW")) is not bool
                or row.get("Type") != "bind"
                or row["Destination"] in observed
            ):
                _fail(FAIL_ACTUAL_ADMISSION, "actor live mount inspect row differs")
            observed[row["Destination"]] = {
                "Source": row["Source"],
                "RW": row["RW"],
                "Type": row["Type"],
            }
        if observed != expected:
            _fail(FAIL_ACTUAL_ADMISSION, "actor live Mount.Source registry differs")
        payload_rows.append([label, len(payload), sha256(payload).hexdigest()])

    for ordinal, sample in enumerate(
        final_resource["live_sample_objects"], start=1
    ):
        if (
            type(sample) is not dict
            or sample.get("mount_registry_sha256")
            != binding["command_mount_registry"]["registry_sha256"]
            or sample.get("mount_command_sha256")
            != binding["command_mount_registry"]["command_sha256"]
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "actor live mount sample root differs")
        replay_payload(sample.get("inspect_payload_hex"), f"LIVE_{ordinal}_BEFORE")
        replay_payload(
            sample.get("inspect_after_payload_hex"),
            f"LIVE_{ordinal}_AFTER",
        )
    replay_payload(final_resource["post_exit_inspect_hex"], "POST_EXIT")
    body: dict[str, object] = {
        "schema_version": (
            "hegel-phase3a-q05b-actor-live-mount-source-replay/1"
        ),
        "role_id": binding["role_id"],
        "actor_id": binding["actor_id"],
        "mount_binding_root": binding["mount_binding_root"],
        "mount_launch_replay_root": launch_replay["launch_replay_root"],
        "inspect_payload_rows": payload_rows,
        "all_live_and_post_exit_sources_exact": True,
    }
    value = dict(body)
    value["live_mount_replay_root"] = sha256(
        ACTUAL_ACTOR_LIVE_MOUNT_SOURCE_REPLAY_ROOT_DOMAIN
        + binding["role_id"].to_bytes(1, "big")
        + _canonical_json_bytes(body)
    ).hexdigest()
    return value


def live_resource_transcript_v1(
    role_id: int,
    container_id: str,
    expected_container_name: str,
    mount_registry: SealedActorMountRegistryV1,
    inspect_payload: bytes,
    cgroup_payloads: Mapping[str, bytes],
    proc_cgroup_payload: bytes,
    cgroup_path: str,
    cgroup_directory_identity: tuple[int, int],
    proc_limits_payload: bytes,
    *,
    seccomp_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Strictly replay observations captured while one container is alive.

    The caller must capture Docker inspect, cgroup-v2 counters, and
    ``/proc/<init-pid>/limits`` before waiting for or removing the container.
    This function binds their exact raw bytes and rejects policy drift.
    """

    if type(role_id) is not int or role_id not in (1, 2, 3):
        _fail(FAIL_POLICY, "resource transcript role id differs")
    if type(mount_registry) is not SealedActorMountRegistryV1 or mount_registry.role_id != role_id:
        _fail(FAIL_POLICY, "resource transcript mount registry differs")
    if type(container_id) is not str or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        _fail(FAIL_POLICY, "resource transcript container id differs")
    expected_role = ROLE_ROWS[role_id - 1]
    inspected = _strict_json_value_v1(inspect_payload, "Docker inspect transcript")
    if type(inspected) is not list or len(inspected) != 1 or type(inspected[0]) is not dict:
        _fail(FAIL_POLICY, "Docker inspect transcript shape differs")
    document = inspected[0]
    state = document.get("State")
    host_config = document.get("HostConfig")
    container_config = document.get("Config")
    mounts = document.get("Mounts")
    if (
        type(state) is not dict
        or type(host_config) is not dict
        or type(container_config) is not dict
        or type(mounts) is not list
    ):
        _fail(FAIL_POLICY, "Docker inspect state/host config differs")
    expected_image = PYTHON_IMAGE if role_id in (1, 3) else RUST_IMAGE
    security_options = host_config.get("SecurityOpt")
    validate_docker_inspect_seccomp_semantics_v1(
        security_options,
        mount_registry.security_options,
        seccomp_evidence,
        RUNTIME_SECCOMP_RELATIVE_PATH,
    )
    environment = container_config.get("Env")
    observed_environment: dict[str, str] = {}
    if type(environment) is not list:
        _fail(FAIL_POLICY, "live Docker environment type differs")
    for item in environment:
        if type(item) is not str or "=" not in item:
            _fail(FAIL_POLICY, "live Docker environment row differs")
        key, value = item.split("=", 1)
        if not key or key in observed_environment:
            _fail(FAIL_POLICY, "live Docker environment contains a duplicate")
        observed_environment[key] = value
    if (
        document.get("Id") != container_id
        or type(expected_container_name) is not str
        or document.get("Name") != f"/{expected_container_name}"
        or state.get("Running") is not True
        or state.get("OOMKilled") is not False
        or type(state.get("Pid")) is not int
        or state["Pid"] <= 0
        or container_config.get("Image") != expected_image
        or container_config.get("User") != f"{os.getuid()}:{os.getgid()}"
        or container_config.get("Entrypoint") is not None
        or container_config.get("Cmd") != list(mount_registry.container_argv)
        or container_config.get("WorkingDir") != mount_registry.working_directory
        or tuple(sorted(observed_environment.items()))
        != mount_registry.environment_rows
        or host_config.get("AutoRemove") is not False
        or host_config.get("NetworkMode") != "none"
        or host_config.get("ReadonlyRootfs") is not True
        or host_config.get("CapDrop") != ["ALL"]
        or host_config.get("IpcMode") != "none"
        or host_config.get("PidMode") != ""
        or host_config.get("CgroupnsMode") != "private"
        or host_config.get("UsernsMode") != ""
        or host_config.get("Privileged") is not False
        or host_config.get("Devices") != []
        or host_config.get("DeviceRequests") is not None
        or host_config.get("CpusetCpus") != expected_role[2]
        or type(host_config.get("Memory")) is not int
        or host_config["Memory"] != 14 * 1024 * 1024 * 1024
        or type(host_config.get("MemorySwap")) is not int
        or host_config["MemorySwap"] != 14 * 1024 * 1024 * 1024
        or type(host_config.get("PidsLimit")) is not int
        or host_config["PidsLimit"] != 128
        or host_config.get("Tmpfs")
        != {"/tmp": "rw,noexec,nosuid,nodev,size=2g,mode=1777"}
    ):
        _fail(FAIL_POLICY, "live Docker inspect policy differs")
    expected_mounts = {
        1: {
            "/snapshot": False,
            "/output": True,
            "/control": True,
        },
        2: {
            "/runtime/hegel-q1-archive-projection-oracle": False,
            "/output": True,
            "/control": True,
        },
        3: {
            "/snapshot": False,
            "/inputs/python": False,
            "/inputs/rust": False,
            "/inputs/stdout/python.stdout": False,
            "/inputs/stdout/rust.stdout": False,
            "/inputs/stdout/manifest.json": False,
            "/control": True,
            "/staging": True,
        },
    }[role_id]
    expected_mount_sources = mount_registry.expected_sources
    if set(expected_mount_sources) != set(expected_mounts):
        _fail(FAIL_POLICY, "expected mount source registry differs")
    observed_mounts: dict[str, bool] = {}
    observed_sources: dict[str, str] = {}
    for row in mounts:
        if type(row) is not dict:
            _fail(FAIL_POLICY, "Docker mount row type differs")
        destination = row.get("Destination")
        source = row.get("Source")
        writable = row.get("RW")
        if (
            type(destination) is not str
            or type(source) is not str
            or not source.startswith("/")
            or "docker.sock" in source
            or type(writable) is not bool
            or row.get("Type") != "bind"
            or destination in observed_mounts
        ):
            _fail(FAIL_POLICY, "Docker bind mount row differs")
        observed_mounts[destination] = writable
        observed_sources[destination] = source
    if observed_mounts != expected_mounts or observed_sources != expected_mount_sources:
        _fail(FAIL_POLICY, "Docker bind mount registry differs")
    ulimits = host_config.get("Ulimits")
    if type(ulimits) is not list or len(ulimits) != 1 or type(ulimits[0]) is not dict:
        _fail(FAIL_POLICY, "Docker nofile inspect row differs")
    if ulimits[0] != {"Name": "nofile", "Hard": 256, "Soft": 256}:
        _fail(FAIL_POLICY, "Docker nofile limit differs")
    expected_cgroup_names = {
        "memory.current",
        "memory.peak",
        "memory.events",
        "pids.current",
        "pids.peak",
    }
    if type(cgroup_payloads) is not dict or set(cgroup_payloads) != expected_cgroup_names:
        _fail(FAIL_POLICY, "cgroup transcript file set differs")
    if (
        type(proc_cgroup_payload) is not bytes
        or type(cgroup_path) is not str
        or type(cgroup_directory_identity) is not tuple
        or len(cgroup_directory_identity) != 2
        or any(type(item) is not int or item < 1 for item in cgroup_directory_identity)
    ):
        _fail(FAIL_POLICY, "proc/cgroup identity types differ")
    cgroup_match = re.fullmatch(rb"0::(/[^\r\n]*)\n", proc_cgroup_payload)
    if cgroup_match is None:
        _fail(FAIL_POLICY, "proc cgroup v2 row differs")
    observed_cgroup_path = cgroup_match.group(1).decode("ascii", "strict")
    if (
        observed_cgroup_path != cgroup_path
        or ".." in PurePosixPath(cgroup_path).parts
        or container_id not in cgroup_path
    ):
        _fail(FAIL_POLICY, "cgroup path is not bound to the container id")
    memory_current = _parse_decimal_line_v1(
        cgroup_payloads["memory.current"], "memory.current"
    )
    memory_peak = _parse_decimal_line_v1(
        cgroup_payloads["memory.peak"], "memory.peak"
    )
    pids_current = _parse_decimal_line_v1(
        cgroup_payloads["pids.current"], "pids.current"
    )
    pids_peak = _parse_decimal_line_v1(cgroup_payloads["pids.peak"], "pids.peak")
    events: dict[str, int] = {}
    for line in cgroup_payloads["memory.events"].splitlines():
        match = re.fullmatch(rb"([a-z_]+) (0|[1-9][0-9]*)", line)
        if match is None:
            _fail(FAIL_POLICY, "memory.events row differs")
        key = match.group(1).decode("ascii")
        if key in events:
            _fail(FAIL_POLICY, "memory.events contains a duplicate")
        events[key] = int(match.group(2))
    if not {"low", "high", "max", "oom", "oom_kill", "oom_group_kill"} <= set(events):
        _fail(FAIL_POLICY, "memory.events required rows are absent")
    if any(events[name] != 0 for name in ("oom", "oom_kill", "oom_group_kill")):
        _fail(FAIL_POLICY, "container reported an OOM event")
    if (
        memory_current > 14 * 1024 * 1024 * 1024
        or memory_peak < memory_current
        or memory_peak > 14 * 1024 * 1024 * 1024
        or pids_current > 128
        or pids_peak < pids_current
        or pids_peak > 128
    ):
        _fail(FAIL_POLICY, "cgroup resource high-water mark differs")
    if type(proc_limits_payload) is not bytes:
        _fail(FAIL_POLICY, "proc limits transcript type differs")
    nofile_rows = tuple(
        line
        for line in proc_limits_payload.decode("ascii", "strict").splitlines()
        if line.startswith("Max open files")
    )
    if len(nofile_rows) != 1:
        _fail(FAIL_POLICY, "proc limits nofile row differs")
    nofile_match = re.fullmatch(
        r"Max open files[ ]+([0-9]+)[ ]+([0-9]+)[ ]+files[ ]*",
        nofile_rows[0],
    )
    if nofile_match is None:
        _fail(FAIL_POLICY, "proc limits nofile format differs")
    if nofile_match.groups() != ("256", "256"):
        _fail(FAIL_POLICY, "proc limits nofile values differ")
    cgroup_digest = sha256()
    path_bytes = cgroup_path.encode("ascii")
    cgroup_digest.update(len(path_bytes).to_bytes(4, "big"))
    cgroup_digest.update(path_bytes)
    cgroup_digest.update(cgroup_directory_identity[0].to_bytes(8, "big"))
    cgroup_digest.update(cgroup_directory_identity[1].to_bytes(8, "big"))
    for name in sorted(cgroup_payloads):
        name_bytes = name.encode("ascii")
        payload = cgroup_payloads[name]
        cgroup_digest.update(len(name_bytes).to_bytes(4, "big"))
        cgroup_digest.update(name_bytes)
        cgroup_digest.update(len(payload).to_bytes(8, "big"))
        cgroup_digest.update(payload)
    return {
        "schema_version": RESOURCE_TRANSCRIPT_SCHEMA_VERSION,
        "container_id": container_id,
        "role_id": role_id,
        "captured_while_running": True,
        "cpuset_cpus": expected_role[2],
        "memory_limit_bytes": 14 * 1024 * 1024 * 1024,
        "memory_swap_limit_bytes": 14 * 1024 * 1024 * 1024,
        "pids_limit": 128,
        "nofile_soft": 256,
        "nofile_hard": 256,
        "oom_killed": False,
        "memory_current_bytes": memory_current,
        "memory_peak_bytes": memory_peak,
        "pids_current": pids_current,
        "pids_peak": pids_peak,
        "memory_events": [[name, events[name]] for name in sorted(events)],
        "cgroup_path": cgroup_path,
        "cgroup_directory_device": cgroup_directory_identity[0],
        "cgroup_directory_inode": cgroup_directory_identity[1],
        "inspect_sha256": sha256(inspect_payload).hexdigest(),
        "inspect_payload_hex": inspect_payload.hex(),
        "proc_cgroup_sha256": sha256(proc_cgroup_payload).hexdigest(),
        "proc_cgroup_payload_hex": proc_cgroup_payload.hex(),
        "cgroup_sha256": cgroup_digest.hexdigest(),
        "cgroup_payload_rows": [
            [name, cgroup_payloads[name].hex()]
            for name in sorted(cgroup_payloads)
        ],
        "proc_limits_sha256": sha256(proc_limits_payload).hexdigest(),
        "proc_limits_payload_hex": proc_limits_payload.hex(),
        "mount_registry_sha256": mount_registry.registry_sha256,
        "mount_command_sha256": mount_registry.command_sha256,
    }


def _read_pseudofile_at_v1(directory: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(FAIL_POLICY, f"anchored pseudofile differs: {name}")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            total += len(block)
            if total > 1024 * 1024:
                _fail(FAIL_POLICY, f"anchored pseudofile exceeds bound: {name}")
            blocks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(FAIL_POLICY, f"anchored pseudofile changed: {name}")
        return b"".join(blocks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_directory_chain_v1(root: int, parts: tuple[str, ...]) -> int:
    current = os.dup(root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        for part in parts:
            if part in ("", ".", "..") or "/" in part:
                _fail(FAIL_POLICY, "anchored directory component differs")
            following = os.open(part, flags, dir_fd=current)
            if not stat.S_ISDIR(os.fstat(following).st_mode):
                os.close(following)
                _fail(FAIL_POLICY, "anchored path component is not a directory")
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def collect_bound_live_resource_transcript_v1(
    role_id: int,
    container_id: str,
    expected_container_name: str,
    mount_registry: SealedActorMountRegistryV1,
    inspect_before_payload: bytes,
    inspect_after_reader: Callable[[], bytes],
    *,
    seccomp_evidence: Mapping[str, object],
    docker_execution_principal: Mapping[str, object] | None = None,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, object]:
    """Collect proc/cgroup bytes through anchored descriptors from inspect PID."""

    if not callable(inspect_after_reader):
        _fail(FAIL_POLICY, "post-sample Docker inspect reader differs")
    if docker_execution_principal is not None:
        before_ownership = _validate_owned_docker_inspect_payload_v1(
            inspect_before_payload,
            docker_execution_principal,
        )
        if before_ownership["container_id"] != container_id:
            _fail(FAIL_POLICY, "pre-sample Docker ownership ID differs")
    before = _strict_json_value_v1(inspect_before_payload, "pre-sample Docker inspect")
    if (
        type(before) is not list
        or len(before) != 1
        or type(before[0]) is not dict
        or type(before[0].get("State")) is not dict
    ):
        _fail(FAIL_POLICY, "pre-sample Docker inspect shape differs")
    pid = before[0]["State"].get("Pid")
    if (
        before[0].get("Id") != container_id
        or type(pid) is not int
        or pid < 1
        or before[0]["State"].get("Running") is not True
    ):
        _fail(FAIL_POLICY, "pre-sample Docker inspect PID differs")
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    proc_descriptor = os.open(proc_root, root_flags)
    pid_descriptor: int | None = None
    cgroup_root_descriptor: int | None = None
    cgroup_descriptor: int | None = None
    try:
        pid_descriptor = os.open(str(pid), root_flags, dir_fd=proc_descriptor)
        pid_before = os.fstat(pid_descriptor)
        proc_cgroup_payload = _read_pseudofile_at_v1(pid_descriptor, "cgroup")
        proc_limits_payload = _read_pseudofile_at_v1(pid_descriptor, "limits")
        match = re.fullmatch(rb"0::(/[^\r\n]*)\n", proc_cgroup_payload)
        if match is None:
            _fail(FAIL_POLICY, "anchored proc cgroup v2 row differs")
        cgroup_path = match.group(1).decode("ascii", "strict")
        if container_id not in cgroup_path or ".." in PurePosixPath(cgroup_path).parts:
            _fail(FAIL_POLICY, "anchored proc cgroup path lacks container identity")
        cgroup_root_descriptor = os.open(cgroup_root, root_flags)
        cgroup_parts = tuple(PurePosixPath(cgroup_path).parts[1:])
        cgroup_descriptor = _open_directory_chain_v1(
            cgroup_root_descriptor,
            cgroup_parts,
        )
        cgroup_before = os.fstat(cgroup_descriptor)
        cgroup_payloads = {
            name: _read_pseudofile_at_v1(cgroup_descriptor, name)
            for name in (
                "memory.current",
                "memory.peak",
                "memory.events",
                "pids.current",
                "pids.peak",
            )
        }
        pid_after = os.fstat(pid_descriptor)
        cgroup_after = os.fstat(cgroup_descriptor)
        if (
            (pid_before.st_dev, pid_before.st_ino)
            != (pid_after.st_dev, pid_after.st_ino)
            or (cgroup_before.st_dev, cgroup_before.st_ino)
            != (cgroup_after.st_dev, cgroup_after.st_ino)
        ):
            _fail(FAIL_POLICY, "proc or cgroup directory changed during sampling")
        reopened_pid = os.open(str(pid), root_flags, dir_fd=proc_descriptor)
        try:
            reopened_pid_status = os.fstat(reopened_pid)
        finally:
            os.close(reopened_pid)
        reopened_cgroup = _open_directory_chain_v1(
            cgroup_root_descriptor,
            cgroup_parts,
        )
        try:
            reopened_cgroup_status = os.fstat(reopened_cgroup)
        finally:
            os.close(reopened_cgroup)
        if (
            (reopened_pid_status.st_dev, reopened_pid_status.st_ino)
            != (pid_before.st_dev, pid_before.st_ino)
            or (reopened_cgroup_status.st_dev, reopened_cgroup_status.st_ino)
            != (cgroup_before.st_dev, cgroup_before.st_ino)
        ):
            _fail(FAIL_POLICY, "proc/cgroup path identity changed during sampling")
        # This callback is intentionally invoked only after every anchored
        # proc/cgroup read and reopen check.  A caller cannot pre-capture the
        # nominal "after" inspect and present it as a later observation.
        inspect_after_payload = inspect_after_reader()
        if type(inspect_after_payload) is not bytes:
            _fail(FAIL_POLICY, "post-sample Docker inspect payload differs")
        if docker_execution_principal is not None:
            after_ownership = _validate_owned_docker_inspect_payload_v1(
                inspect_after_payload,
                docker_execution_principal,
            )
            if after_ownership["container_id"] != container_id:
                _fail(FAIL_POLICY, "post-sample Docker ownership ID differs")
        after = _strict_json_value_v1(
            inspect_after_payload,
            "post-sample Docker inspect",
        )
        if (
            type(after) is not list
            or len(after) != 1
            or type(after[0]) is not dict
            or type(after[0].get("State")) is not dict
            or after[0].get("Id") != container_id
            or after[0]["State"].get("Pid") != pid
            or after[0]["State"].get("Running") is not True
        ):
            _fail(FAIL_POLICY, "Docker inspect PID changed after sampling")
        result = live_resource_transcript_v1(
            role_id,
            container_id,
            expected_container_name,
            mount_registry,
            inspect_before_payload,
            cgroup_payloads,
            proc_cgroup_payload,
            cgroup_path,
            (cgroup_before.st_dev, cgroup_before.st_ino),
            proc_limits_payload,
            seccomp_evidence=seccomp_evidence,
        )
        after_replay = live_resource_transcript_v1(
            role_id,
            container_id,
            expected_container_name,
            mount_registry,
            inspect_after_payload,
            cgroup_payloads,
            proc_cgroup_payload,
            cgroup_path,
            (cgroup_before.st_dev, cgroup_before.st_ino),
            proc_limits_payload,
            seccomp_evidence=seccomp_evidence,
        )
        inspect_only_fields = {"inspect_sha256", "inspect_payload_hex"}
        if {
            key: value for key, value in result.items() if key not in inspect_only_fields
        } != {
            key: value
            for key, value in after_replay.items()
            if key not in inspect_only_fields
        }:
            _fail(FAIL_POLICY, "Docker inspect policy changed during sampling")
        result["inspect_after_sha256"] = sha256(inspect_after_payload).hexdigest()
        result["inspect_after_payload_hex"] = inspect_after_payload.hex()
        result["proc_pid_directory_device"] = pid_before.st_dev
        result["proc_pid_directory_inode"] = pid_before.st_ino
        result["anchored_collection"] = True
        return result
    finally:
        if cgroup_descriptor is not None:
            os.close(cgroup_descriptor)
        if cgroup_root_descriptor is not None:
            os.close(cgroup_root_descriptor)
        if pid_descriptor is not None:
            os.close(pid_descriptor)
        os.close(proc_descriptor)


def _validate_resource_sample_raw_preimages_v1(
    sample: Mapping[str, object],
) -> None:
    """Recompute every raw observation digest kept for artifact replay."""

    if type(sample) is not dict:
        _fail(FAIL_POLICY, "resource sample raw preimage type differs")
    decoded: dict[str, bytes] = {}
    for payload_field, digest_field in (
        ("inspect_payload_hex", "inspect_sha256"),
        ("inspect_after_payload_hex", "inspect_after_sha256"),
        ("proc_cgroup_payload_hex", "proc_cgroup_sha256"),
        ("proc_limits_payload_hex", "proc_limits_sha256"),
    ):
        payload_hex = sample.get(payload_field)
        digest_hex = sample.get(digest_field)
        if (
            type(payload_hex) is not str
            or len(payload_hex) % 2
            or re.fullmatch(r"[0-9a-f]*", payload_hex) is None
            or type(digest_hex) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest_hex) is None
        ):
            _fail(FAIL_POLICY, "resource sample raw preimage encoding differs")
        payload = bytes.fromhex(payload_hex)
        if sha256(payload).hexdigest() != digest_hex:
            _fail(FAIL_POLICY, "resource sample raw preimage digest differs")
        decoded[payload_field] = payload
    rows = sample.get("cgroup_payload_rows")
    expected_names = (
        "memory.current",
        "memory.events",
        "memory.peak",
        "pids.current",
        "pids.peak",
    )
    if (
        type(rows) is not list
        or len(rows) != len(expected_names)
        or type(sample.get("cgroup_path")) is not str
        or type(sample.get("cgroup_directory_device")) is not int
        or type(sample.get("cgroup_directory_inode")) is not int
    ):
        _fail(FAIL_POLICY, "resource sample cgroup raw preimage differs")
    cgroup_digest = sha256()
    path_bytes = sample["cgroup_path"].encode("ascii", "strict")
    cgroup_digest.update(len(path_bytes).to_bytes(4, "big"))
    cgroup_digest.update(path_bytes)
    cgroup_digest.update(sample["cgroup_directory_device"].to_bytes(8, "big"))
    cgroup_digest.update(sample["cgroup_directory_inode"].to_bytes(8, "big"))
    for row, expected_name in zip(rows, expected_names, strict=True):
        if (
            type(row) is not list
            or len(row) != 2
            or row[0] != expected_name
            or type(row[1]) is not str
            or len(row[1]) % 2
            or re.fullmatch(r"[0-9a-f]*", row[1]) is None
        ):
            _fail(FAIL_POLICY, "resource sample cgroup raw row differs")
        payload = bytes.fromhex(row[1])
        name_bytes = expected_name.encode("ascii")
        cgroup_digest.update(len(name_bytes).to_bytes(4, "big"))
        cgroup_digest.update(name_bytes)
        cgroup_digest.update(len(payload).to_bytes(8, "big"))
        cgroup_digest.update(payload)
    if cgroup_digest.hexdigest() != sample.get("cgroup_sha256"):
        _fail(FAIL_POLICY, "resource sample cgroup raw digest differs")


def final_resource_transcript_v1(
    live_samples: tuple[Mapping[str, object], ...],
    post_exit_inspect_payload: bytes,
    *,
    command_security_options: Sequence[str],
    seccomp_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Close continuous live sampling with a successful post-exit inspect."""

    if type(live_samples) is not tuple or not live_samples:
        _fail(FAIL_POLICY, "live resource sample sequence differs")
    first = live_samples[0]
    if type(first) is not dict:
        _fail(FAIL_POLICY, "live resource sample type differs")
    container_id = first.get("container_id")
    role_id = first.get("role_id")
    previous_memory_peak = -1
    previous_pids_peak = -1
    previous_sample_finish_ns: int | None = None
    maximum_inter_sample_gap_ns = 0
    for ordinal, sample in enumerate(live_samples, start=1):
        if (
            type(sample) is not dict
            or sample.get("schema_version") != RESOURCE_TRANSCRIPT_SCHEMA_VERSION
            or sample.get("container_id") != container_id
            or sample.get("role_id") != role_id
            or sample.get("captured_while_running") is not True
            or type(sample.get("memory_peak_bytes")) is not int
            or type(sample.get("pids_peak")) is not int
            or sample["memory_peak_bytes"] < previous_memory_peak
            or sample["pids_peak"] < previous_pids_peak
            or type(sample.get("sample_ordinal")) is not int
            or sample["sample_ordinal"] != ordinal
            or type(sample.get("sample_monotonic_ns")) is not int
            or sample["sample_monotonic_ns"] < 0
            or type(sample.get("sample_duration_ns")) is not int
            or sample["sample_duration_ns"] < 0
            or type(sample.get("inspect_payload_hex")) is not str
            or type(sample.get("inspect_after_payload_hex")) is not str
            or type(sample.get("proc_cgroup_payload_hex")) is not str
            or type(sample.get("proc_limits_payload_hex")) is not str
            or type(sample.get("cgroup_payload_rows")) is not list
        ):
            _fail(FAIL_POLICY, "live resource sample continuity differs")
        _validate_resource_sample_raw_preimages_v1(sample)
        if previous_sample_finish_ns is not None:
            gap = sample["sample_monotonic_ns"] - previous_sample_finish_ns
            if gap < 0 or gap > 250_000_000:
                _fail(FAIL_POLICY, "live resource sampling exceeded frozen gap")
            maximum_inter_sample_gap_ns = max(maximum_inter_sample_gap_ns, gap)
        previous_sample_finish_ns = (
            sample["sample_monotonic_ns"] + sample["sample_duration_ns"]
        )
        previous_memory_peak = sample["memory_peak_bytes"]
        previous_pids_peak = sample["pids_peak"]
    if (
        live_samples[-1].get("actor_child_complete_held") is not True
        or live_samples[-1].get("fresh_after_done_collection") is not True
        or live_samples[-1].get("anchored_collection") is not True
        or type(live_samples[-1].get("completion_manifest_sha256")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}",
            live_samples[-1]["completion_manifest_sha256"],
        )
        is None
    ):
        _fail(FAIL_POLICY, "final sample was not captured while actor child was held")
    inspected = _strict_json_value_v1(
        post_exit_inspect_payload,
        "post-exit Docker inspect transcript",
    )
    if type(inspected) is not list or len(inspected) != 1 or type(inspected[0]) is not dict:
        _fail(FAIL_POLICY, "post-exit Docker inspect shape differs")
    document = inspected[0]
    state = document.get("State")
    host_config = document.get("HostConfig")
    if (
        type(state) is not dict
        or type(host_config) is not dict
        or document.get("Id") != container_id
        or state.get("Running") is not False
        or state.get("OOMKilled") is not False
        or type(state.get("ExitCode")) is not int
        or state["ExitCode"] != 0
        or host_config.get("AutoRemove") is not False
    ):
        _fail(FAIL_POLICY, "post-exit Docker state differs")
    validate_docker_inspect_seccomp_semantics_v1(
        host_config.get("SecurityOpt"),
        command_security_options,
        seccomp_evidence,
        RUNTIME_SECCOMP_RELATIVE_PATH,
    )
    sample_rows = [
        [
            index,
            sample["inspect_sha256"],
            sample["proc_cgroup_sha256"],
            sample["cgroup_sha256"],
            sample["proc_limits_sha256"],
            sample["memory_peak_bytes"],
            sample["pids_peak"],
            sample["memory_events"],
            sample.get("actor_child_complete_held", False),
            sample.get("completion_manifest_sha256"),
            sample["sample_monotonic_ns"],
            sample["sample_duration_ns"],
        ]
        for index, sample in enumerate(live_samples, start=1)
    ]
    value = {
        "schema_version": (
            "hegel-phase3a-q05b-final-container-resource-transcript/1"
        ),
        "container_id": container_id,
        "role_id": role_id,
        "sampling_interval_milliseconds": 250,
        "continuous_sampling_through_child_completion": True,
        "fresh_held_final_before_release": True,
        "post_release_wrapper_only_exits": True,
        "post_exit_zero_and_no_oom": True,
        "peak_scope": "CHILD_PLUS_WRAPPER_THROUGH_HELD_FINAL_SAMPLE",
        "actor_exit_code": 0,
        "oom_killed": False,
        "sample_count": len(sample_rows),
        "sample_rows": sample_rows,
        "maximum_inter_sample_gap_ns": maximum_inter_sample_gap_ns,
        "live_sample_objects": [dict(sample) for sample in live_samples],
        "final_memory_peak_bytes": previous_memory_peak,
        "final_pids_peak": previous_pids_peak,
        "post_exit_inspect_sha256": sha256(post_exit_inspect_payload).hexdigest(),
        "post_exit_inspect_hex": post_exit_inspect_payload.hex(),
        "explicit_remove_admitted_after_this_transcript": True,
    }
    value["transcript_sha256"] = sha256(_canonical_json_bytes(value)).hexdigest()
    return value


def docker_explicit_remove_command_v1(container_id: str) -> list[str]:
    if type(container_id) is not str or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        _fail(FAIL_POLICY, "container removal id differs")
    return [
        DOCKER_EXECUTABLE,
        f"--host={DOCKER_HOST}",
        "rm",
        container_id,
    ]


def _validate_isolation_config_value_v1(
    value: object,
    *,
    engineering_status: str,
    actual_preconditions: Mapping[str, object],
    project_root: Path | None,
) -> dict[str, object]:
    """Validate every claim-critical config field from one parsed value.

    ``project_root=None`` performs the same byte-level policy validation while
    deferring only the two filesystem seccomp replays.  Commit-A admission uses
    that pure mode; the ordinary loader additionally replays both local files.
    """

    if (
        type(value) is not dict
        or type(engineering_status) is not str
        or type(actual_preconditions) is not dict
        or (project_root is not None and not isinstance(project_root, Path))
    ):
        _fail(FAIL_CONFIG, "dual isolation config validator inputs differ")
    expected_top = {
        "schema_version",
        "profile_id",
        "claim_scope",
        "engineering_status",
        "images",
        "docker",
        "seccomp",
        "resource_roles",
        "mount_policy",
        "stdout_capture_policy",
        "held_actor_protocol",
        "live_resource_evidence_policy",
        "runtime_command_inspect_policy",
        "source_snapshot_policy",
        "source_allowlist_policy",
        "rust_build_policy",
        "actor_commands",
        "execution_protocol",
        "qualification_receipt_protocol",
        "artifact_layout",
        "dry_run_authority",
        "actual_preconditions",
    }
    if set(value) != expected_top:
        _fail(FAIL_CONFIG, "dual isolation config top-level fields differ")
    if (
        value["schema_version"] != "hegel-phase3a-q05b-dual-isolation/1"
        or value["profile_id"] != PROFILE_ID
        or value["claim_scope"] != "Q05B_TARGET_BLIND_QUALIFICATION_ONLY"
        or value["engineering_status"] != engineering_status
    ):
        _fail(FAIL_CONFIG, "dual isolation profile identity differs")
    images = _exact_dict(
        value["images"],
        {"python_endpoint", "rust_build", "rust_runtime", "trusted_host"},
        "images",
    )
    _require_type_exact_v1(images, {
        "python_endpoint": PYTHON_IMAGE,
        "rust_build": RUST_IMAGE,
        "rust_runtime": RUST_IMAGE,
        "trusted_host": PYTHON_IMAGE,
    }, "images")
    docker = _exact_dict(
        value["docker"],
        {
            "executable",
            "host",
            "ownership_namespace_domain_ascii",
            "ownership_namespace_derivation",
            "attempt_nonce_length_bytes",
            "execution_slot_rows",
            "unique_container_name_template",
            "reserved_label_keys",
            "command_label_policy",
            "config_label_policy",
            "python_pinned_image_base_label_rows",
            "rust_pinned_image_base_label_rows",
            "initial_name_absence_sample_count",
            "precreate_name_absence_sample_count",
            "container_name_usage",
            "destructive_target",
            "foreign_name_collision_policy",
            "unknown_daemon_or_ownership_state_policy",
            "remove_by_container_name_forbidden",
            "docker_inventory_baseline_scope",
            "pull_policy",
            "network",
            "root_filesystem_read_only",
            "cap_drop",
            "no_new_privileges",
            "ipc",
            "cgroup_namespace",
            "pids_limit",
            "nofile_ulimit",
            "memory",
            "memory_swap",
            "runtime_tmpfs",
            "build_tmpfs",
            "run_as_caller_uid_gid",
            "docker_socket_mounted_into_actor",
            "auto_remove",
            "explicit_remove_after_post_exit_inspect",
        },
        "docker",
    )
    _require_type_exact_v1(docker, {
        "executable": DOCKER_EXECUTABLE,
        "host": DOCKER_HOST,
        "ownership_namespace_domain_ascii": (
            "HEGEL/Q05B/DOCKER/OWNERSHIP_NAMESPACE/V1"
        ),
        "ownership_namespace_derivation": (
            "SHA256(DOMAIN_NUL_ATTEMPT_NONCE_32_SOURCE_COMMIT_ASCII_40)"
        ),
        "attempt_nonce_length_bytes": 32,
        "execution_slot_rows": [
            [1, "RUST_TEST", "rust-test"],
            [2, "RUST_RELEASE", "rust-release"],
            [3, "PYTHON_ENDPOINT", "python"],
            [4, "RUST_ENDPOINT", "rust"],
            [5, "TRUSTED_HOST_REPLAY", "host"],
        ],
        "unique_container_name_template": (
            "hegel-q05b-{FULL64_EXECUTION_NAMESPACE}-{SLOT_SUFFIX}"
        ),
        "reserved_label_keys": list(DOCKER_EXECUTION_RESERVED_LABEL_KEYS),
        "command_label_policy": (
            "EXACTLY_THREE_RESERVED_LABELS_IN_FROZEN_ORDER"
        ),
        "config_label_policy": (
            "PINNED_IMAGE_BASE_LABELS_UNION_EXACT_THREE_RESERVED_LABELS"
        ),
        "python_pinned_image_base_label_rows": [],
        "rust_pinned_image_base_label_rows": [
            [
                "org.opencontainers.image.source",
                "https://github.com/rust-lang/docker-rust",
            ]
        ],
        "initial_name_absence_sample_count": 2,
        "precreate_name_absence_sample_count": 2,
        "container_name_usage": "READ_ONLY_DISCOVERY_ONLY",
        "destructive_target": (
            "OWNERSHIP_VALIDATED_64_LOWERHEX_CONTAINER_ID_ONLY"
        ),
        "foreign_name_collision_policy": "ZERO_MUTATION_FAIL_CLOSED",
        "unknown_daemon_or_ownership_state_policy": "ZERO_MUTATION_FAIL_CLOSED",
        "remove_by_container_name_forbidden": True,
        "docker_inventory_baseline_scope": (
            "RUN_AUDIT_ONLY_NOT_ADMISSION_OR_CLAIM_EVIDENCE"
        ),
        "pull_policy": "never",
        "network": "none",
        "root_filesystem_read_only": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "ipc": "none",
        "cgroup_namespace": "private",
        "pids_limit": 128,
        "nofile_ulimit": "256:256",
        "memory": "14g",
        "memory_swap": "14g",
        "runtime_tmpfs": "/tmp:rw,noexec,nosuid,nodev,size=2g,mode=1777",
        "build_tmpfs": "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777",
        "run_as_caller_uid_gid": True,
        "docker_socket_mounted_into_actor": False,
        "auto_remove": False,
        "explicit_remove_after_post_exit_inspect": True,
    }, "docker")
    seccomp = _exact_dict(
        value["seccomp"],
        {
            "runtime_profile",
            "runtime_profile_sha256",
            "build_profile",
            "build_profile_sha256",
            "default_seccomp_forbidden",
        },
        "seccomp",
    )
    _require_type_exact_v1(seccomp, {
        "runtime_profile": RUNTIME_SECCOMP_RELATIVE_PATH,
        "runtime_profile_sha256": RUNTIME_SECCOMP_SHA256,
        "build_profile": BUILD_SECCOMP_RELATIVE_PATH,
        "build_profile_sha256": BUILD_SECCOMP_SHA256,
        "default_seccomp_forbidden": True,
    }, "seccomp")
    if project_root is not None:
        if (
            _sha256_file(
                project_root / RUNTIME_SECCOMP_RELATIVE_PATH,
                "runtime seccomp",
            )
            != RUNTIME_SECCOMP_SHA256
            or _sha256_file(
                project_root / BUILD_SECCOMP_RELATIVE_PATH,
                "build seccomp",
            )
            != BUILD_SECCOMP_SHA256
        ):
            _fail(FAIL_CONFIG, "committed seccomp bytes differ")
    _require_type_exact_v1(
        value["resource_roles"],
        [list(row) for row in ROLE_ROWS],
        "resource_roles",
    )
    _require_type_exact_v1(
        value["execution_protocol"],
        [list(row) for row in EXECUTION_PROTOCOL],
        "execution_protocol",
    )
    authority = _exact_dict(
        value["dry_run_authority"],
        {
            "qualification_predicate_count",
            "qualification_predicate_mask",
            "qualification_predicate_total",
            "q1_state",
            "q1_gate_count",
            "q1_gate_mask",
            "q1_gate_total",
            "q1_formal_output_roots",
            "q1_receipt",
            "q2_state",
            "m3_formal_roots",
            "formal_fixed_point_claimed",
            "outside_certificate_issued",
            "active_transition_allowed",
            "artifact_written",
        },
        "dry_run_authority",
    )
    _require_type_exact_v1(authority, {
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_gate_total": 20,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "formal_fixed_point_claimed": False,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
        "artifact_written": False,
    }, "dry_run_authority")
    receipt = _exact_dict(
        value["qualification_receipt_protocol"],
        {
            "candidate_tag",
            "candidate_count",
            "candidate_mask",
            "final_tag",
            "final_count",
            "final_mask",
            "candidate_must_be_strictly_decoded_before_final",
            "predicate20_requires_candidate_root_and_closed_q1_authority",
            "dry_run_candidate_receipt",
            "dry_run_final_receipt",
        },
        "qualification_receipt_protocol",
    )
    _require_type_exact_v1(
        receipt,
        {
            "candidate_tag": 14853,
            "candidate_count": 19,
            "candidate_mask": 0x7FFFF,
            "final_tag": 14854,
            "final_count": 20,
            "final_mask": 0xFFFFF,
            "candidate_must_be_strictly_decoded_before_final": True,
            "predicate20_requires_candidate_root_and_closed_q1_authority": True,
            "dry_run_candidate_receipt": None,
            "dry_run_final_receipt": None,
        },
        "qualification_receipt_protocol",
    )
    preconditions = _exact_dict(
        value["actual_preconditions"],
        set(actual_preconditions),
        "actual_preconditions",
    )
    _require_type_exact_v1(
        preconditions,
        actual_preconditions,
        "actual_preconditions",
    )
    _validate_policy_sections_v1(value)
    return value


def load_isolation_config_v1(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    path = project_root / CONFIG_RELATIVE_PATH
    try:
        payload = path.read_bytes()
    except OSError as error:
        _fail(FAIL_CONFIG, f"cannot read isolation config: {error}")
    value = _strict_json_object_v1(payload, "dual isolation config")
    return _validate_isolation_config_value_v1(
        value,
        engineering_status=COMMIT_A_ACTUAL_ENGINEERING_STATUS,
        actual_preconditions=COMMIT_A_ACTUAL_PRECONDITIONS_V1,
        project_root=project_root,
    )


def _validate_policy_sections_v1(config: Mapping[str, object]) -> None:
    mounts = _exact_dict(
        config["mount_policy"],
        {
            "source_snapshots",
            "python_output",
            "rust_output",
            "endpoint_output_exchange",
            "host_artifact_staging",
            "cargo_cache",
            "prebuilt_rust_binary",
            "docker_socket",
            "target_truth_or_split_sources",
        },
        "mount_policy",
    )
    _require_type_exact_v1(
        mounts,
        {
            "source_snapshots": "DISTINCT_GIT_BLOB_MATERIALIZED_READ_ONLY",
            "python_output": "PYTHON_WRITABLE_ONLY_THEN_HOST_READ_ONLY",
            "rust_output": "RUST_WRITABLE_ONLY_THEN_HOST_READ_ONLY",
            "endpoint_output_exchange": False,
            "host_artifact_staging": "HOST_ONLY_WRITABLE_ACTUAL_MODE_ONLY",
            "cargo_cache": "RUST_BUILD_READ_ONLY_OFFLINE_IDENTITY",
            "prebuilt_rust_binary": "RUNTIME_READ_ONLY",
            "docker_socket": "NEVER_MOUNTED_IN_ANY_ACTOR",
            "target_truth_or_split_sources": "ABSENT",
        },
        "mount_policy",
    )
    stdout_policy = _exact_dict(
        config["stdout_capture_policy"],
        {
            "schema_version",
            "endpoint_stdout_maximum_bytes",
            "stdout_file_mode",
            "manifest_file_mode",
            "python_stdout_relative_path",
            "rust_stdout_relative_path",
            "manifest_relative_path",
            "host_mount_rows",
            "separate_from_five_sidecar_trees",
            "manifest_binds_actor_id_length_sha256",
            "host_strict_actor_envelope_replay",
        },
        "stdout_capture_policy",
    )
    _require_type_exact_v1(
        stdout_policy,
        {
            "schema_version": STDOUT_SET_SCHEMA_VERSION,
            "endpoint_stdout_maximum_bytes": 1024 * 1024,
            "stdout_file_mode": "0444",
            "manifest_file_mode": "0444",
            "python_stdout_relative_path": "stdout/python.stdout",
            "rust_stdout_relative_path": "stdout/rust.stdout",
            "manifest_relative_path": "stdout/manifest.json",
            "host_mount_rows": [
                [1, "PYTHON_ENDPOINT_STDOUT", "/inputs/stdout/python.stdout"],
                [2, "RUST_ENDPOINT_STDOUT", "/inputs/stdout/rust.stdout"],
                [3, "STDOUT_SET_MANIFEST", "/inputs/stdout/manifest.json"],
            ],
            "separate_from_five_sidecar_trees": True,
            "manifest_binds_actor_id_length_sha256": True,
            "host_strict_actor_envelope_replay": True,
        },
        "stdout_capture_policy",
    )
    held = _exact_dict(
        config["held_actor_protocol"],
        {
            "schema_version",
            "wrapper_script_exact",
            "wrapper_script_sha256",
            "control_mount_destination",
            "control_directory_initial_mode",
            "wrapper_initial_file_mode",
            "sealed_control_file_mode",
            "actor_stdout_relative_path",
            "exit_code_relative_path",
            "done_relative_path",
            "release_relative_path",
            "done_exact_bytes",
            "success_exit_code_exact_bytes",
            "release_exact_bytes",
            "actor_stdout_maximum_bytes",
            "wrapper_holds_container_and_cgroup_after_child_exit",
            "final_cgroup_sample_only_after_done_is_sealed",
            "release_only_after_final_sample_binds_control_manifest",
            "wrapper_forwards_actor_stdout_byte_exact_before_done",
            "wrapper_exits_with_actor_exit_code",
            "wrapper_resource_overhead_can_only_conservatively_raise_peak",
            "post_exit_inspect_then_explicit_remove",
        },
        "held_actor_protocol",
    )
    _require_type_exact_v1(
        held,
        {
            "schema_version": HELD_ACTOR_WRAPPER_SCHEMA_VERSION,
            "wrapper_script_exact": HELD_ACTOR_WRAPPER_SCRIPT,
            "wrapper_script_sha256": sha256(
                HELD_ACTOR_WRAPPER_SCRIPT.encode("utf-8")
            ).hexdigest(),
            "control_mount_destination": "/control",
            "control_directory_initial_mode": "0700_EMPTY",
            "wrapper_initial_file_mode": "0600",
            "sealed_control_file_mode": "0444",
            "actor_stdout_relative_path": "actor.stdout",
            "exit_code_relative_path": "exit-code",
            "done_relative_path": "done",
            "release_relative_path": "release",
            "done_exact_bytes": "ACTOR_COMPLETE_HELD\\n",
            "success_exit_code_exact_bytes": "0\\n",
            "release_exact_bytes": "HOST_FINAL_SAMPLE_SEALED\\n",
            "actor_stdout_maximum_bytes": 1024 * 1024,
            "wrapper_holds_container_and_cgroup_after_child_exit": True,
            "final_cgroup_sample_only_after_done_is_sealed": True,
            "release_only_after_final_sample_binds_control_manifest": True,
            "wrapper_forwards_actor_stdout_byte_exact_before_done": True,
            "wrapper_exits_with_actor_exit_code": True,
            "wrapper_resource_overhead_can_only_conservatively_raise_peak": True,
            "post_exit_inspect_then_explicit_remove": True,
        },
        "held_actor_protocol",
    )
    resource_policy = _exact_dict(
        config["live_resource_evidence_policy"],
        {
            "schema_version",
            "cidfile_required",
            "cidfile_parent_mode",
            "cidfile_initial_mode",
            "cidfile_mode_after_capture",
            "cidfile_regular_nlink1_and_nofollow_required",
            "cidfile_parent_and_entry_identity_stable_required",
            "docker_inspect_while_running_required",
            "cgroup_capture_while_running_required",
            "proc_cgroup_binding_required",
            "proc_limits_capture_while_running_required",
            "required_inspect_fields",
            "required_cgroup_fields",
            "required_proc_limits",
            "required_transcript_fields",
            "sampling_interval_milliseconds",
            "sampling_interval_semantics",
            "one_independent_sampling_worker_per_live_actor",
            "collection_duration_is_recorded_and_not_counted_as_unsampled_gap",
            "continuous_sampling_until_child_completion",
            "no_sampling_claim_after_held_final_release",
            "peak_scope",
            "final_live_sample_required",
            "post_exit_inspect_requires_exit_zero_and_no_oom",
            "endpoint_transcripts_captured_before_wait",
            "host_transcript_captured_before_wait",
            "auto_remove_forbidden",
            "explicit_remove_only_after_final_transcript_sealed",
            "stdout_drain_maximum_bytes",
            "stderr_drain_maximum_bytes",
            "pipe_drains_continue_after_overflow_without_retaining_extra_bytes",
            "pipe_drain_overflow_or_thread_error_fails_qualification",
            "failure_cleanup_requires_abort_release_cli_stop_wait_and_owned_id_force_remove",
            "failure_cleanup_force_remove_target_policy",
            "foreign_or_unknown_name_cleanup_mutation_forbidden",
        },
        "live_resource_evidence_policy",
    )
    _require_type_exact_v1(
        resource_policy,
        {
            "schema_version": RESOURCE_TRANSCRIPT_SCHEMA_VERSION,
            "cidfile_required": True,
            "cidfile_parent_mode": "0700",
            "cidfile_initial_mode": "0600",
            "cidfile_mode_after_capture": "0444",
            "cidfile_regular_nlink1_and_nofollow_required": True,
            "cidfile_parent_and_entry_identity_stable_required": True,
            "docker_inspect_while_running_required": True,
            "cgroup_capture_while_running_required": True,
            "proc_cgroup_binding_required": True,
            "proc_limits_capture_while_running_required": True,
            "required_inspect_fields": [
                "Id",
                "Name",
                "State.Running",
                "State.OOMKilled",
                "State.Pid",
                "Config.Image",
                "Config.Labels",
                "Config.User",
                "Config.Env",
                "Config.Entrypoint",
                "Config.Cmd",
                "Config.WorkingDir",
                "HostConfig.AutoRemove",
                "HostConfig.NetworkMode",
                "HostConfig.ReadonlyRootfs",
                "HostConfig.CapDrop",
                "HostConfig.SecurityOpt",
                "HostConfig.IpcMode",
                "HostConfig.PidMode",
                "HostConfig.CgroupnsMode",
                "HostConfig.UsernsMode",
                "HostConfig.Privileged",
                "HostConfig.Devices",
                "HostConfig.DeviceRequests",
                "HostConfig.CpusetCpus",
                "HostConfig.Memory",
                "HostConfig.MemorySwap",
                "HostConfig.PidsLimit",
                "HostConfig.Ulimits",
                "HostConfig.Tmpfs",
                "Mounts",
            ],
            "required_cgroup_fields": [
                "memory.current",
                "memory.peak",
                "memory.events",
                "pids.current",
                "pids.peak",
            ],
            "required_proc_limits": ["Max open files"],
            "required_transcript_fields": [
                "container_id",
                "role_id",
                "captured_while_running",
                "cpuset_cpus",
                "memory_limit_bytes",
                "memory_swap_limit_bytes",
                "pids_limit",
                "nofile_soft",
                "nofile_hard",
                "oom_killed",
                "memory_current_bytes",
                "memory_peak_bytes",
                "pids_current",
                "pids_peak",
                "memory_events",
                "cgroup_path",
                "cgroup_directory_device",
                "cgroup_directory_inode",
                "inspect_sha256",
                "inspect_payload_hex",
                "inspect_after_sha256",
                "inspect_after_payload_hex",
                "proc_cgroup_sha256",
                "proc_cgroup_payload_hex",
                "cgroup_sha256",
                "cgroup_payload_rows",
                "proc_limits_sha256",
                "proc_limits_payload_hex",
            ],
            "sampling_interval_milliseconds": 250,
            "sampling_interval_semantics": (
                "MAXIMUM_UNSAMPLED_GAP_FROM_PREVIOUS_COLLECTION_FINISH_TO_"
                "NEXT_COLLECTION_START"
            ),
            "one_independent_sampling_worker_per_live_actor": True,
            "collection_duration_is_recorded_and_not_counted_as_unsampled_gap": True,
            "continuous_sampling_until_child_completion": True,
            "no_sampling_claim_after_held_final_release": True,
            "peak_scope": "CHILD_PLUS_WRAPPER_THROUGH_HELD_FINAL_SAMPLE",
            "final_live_sample_required": True,
            "post_exit_inspect_requires_exit_zero_and_no_oom": True,
            "endpoint_transcripts_captured_before_wait": True,
            "host_transcript_captured_before_wait": True,
            "auto_remove_forbidden": True,
            "explicit_remove_only_after_final_transcript_sealed": True,
            "stdout_drain_maximum_bytes": 1024 * 1024,
            "stderr_drain_maximum_bytes": 16 * 1024 * 1024,
            "pipe_drains_continue_after_overflow_without_retaining_extra_bytes": True,
            "pipe_drain_overflow_or_thread_error_fails_qualification": True,
            "failure_cleanup_requires_abort_release_cli_stop_wait_and_owned_id_force_remove": True,
            "failure_cleanup_force_remove_target_policy": (
                "OWNERSHIP_VALIDATED_64_LOWERHEX_CONTAINER_ID_ONLY"
            ),
            "foreign_or_unknown_name_cleanup_mutation_forbidden": True,
        },
        "live_resource_evidence_policy",
    )
    runtime_inspect = _exact_dict(
        config["runtime_command_inspect_policy"],
        {
            "schema_version",
            "container_argv_derivation",
            "held_prefix",
            "environment_rows",
            "working_directory_rows",
            "config_exact_rows",
            "host_config_exact_rows",
            "command_labels",
            "config_labels",
            "missing_extra_or_mismatched_config_labels_forbidden",
            "extra_environment_keys_forbidden",
            "extra_or_reordered_container_argv_forbidden",
        },
        "runtime_command_inspect_policy",
    )
    _require_type_exact_v1(
        runtime_inspect,
        {
            "schema_version": "hegel-phase3a-q05b-runtime-command-inspect-policy/1",
            "container_argv_derivation": (
                "HELD_PREFIX_PLUS_ACTOR_COMMANDS_WITH_HOST_IDENTITY_"
                "PLACEHOLDERS_SUBSTITUTED"
            ),
            "held_prefix": [
                "/bin/sh",
                "-ceu",
                "HELD_ACTOR_WRAPPER_SCRIPT_EXACT",
                "hegel-q05b-held-actor",
            ],
            "environment_rows": [
                [
                    role_id,
                    actor_id,
                    [list(row) for row in sorted(environment.items())],
                ]
                for role_id, actor_id, environment in (
                    (1, "PYTHON_ENDPOINT", PYTHON_RUNTIME_ENVIRONMENT),
                    (2, "RUST_ENDPOINT", RUST_RUNTIME_ENVIRONMENT),
                    (3, "TRUSTED_HOST_REPLAY", PYTHON_RUNTIME_ENVIRONMENT),
                )
            ],
            "working_directory_rows": [
                [1, "PYTHON_ENDPOINT", "/snapshot"],
                [2, "RUST_ENDPOINT", ""],
                [3, "TRUSTED_HOST_REPLAY", "/snapshot"],
            ],
            "config_exact_rows": [
                ["Entrypoint", None],
                ["Cmd", "EXACT_GENERATED_CONTAINER_ARGV"],
                ["Env", "EXACT_ROLE_ENVIRONMENT_ROWS"],
                ["WorkingDir", "EXACT_ROLE_WORKING_DIRECTORY"],
            ],
            "host_config_exact_rows": [
                ["AutoRemove", False],
                ["NetworkMode", "none"],
                ["ReadonlyRootfs", True],
                ["CapDrop", ["ALL"]],
                ["SecurityOpt", "EXACT_NO_NEW_PRIVILEGES_AND_SEALED_SECCOMP"],
                ["IpcMode", "none"],
                ["PidMode", ""],
                ["CgroupnsMode", "private"],
                ["UsernsMode", ""],
                ["Privileged", False],
                ["Devices", []],
                ["DeviceRequests", None],
            ],
            "command_labels": (
                "EXACT_THREE_RESERVED_LABEL_ROWS_FROM_DOCKER_EXECUTION_AUTHORITY"
            ),
            "config_labels": (
                "PINNED_IMAGE_BASE_LABELS_UNION_EXACT_THREE_RESERVED_LABEL_ROWS"
            ),
            "missing_extra_or_mismatched_config_labels_forbidden": True,
            "extra_environment_keys_forbidden": True,
            "extra_or_reordered_container_argv_forbidden": True,
        },
        "runtime_command_inspect_policy",
    )
    snapshot = _exact_dict(
        config["source_snapshot_policy"],
        {
            "commit_wire",
            "head_must_equal_requested_commit",
            "worktree_and_index_must_be_completely_clean",
            "materialization",
            "worktree_copy_forbidden",
            "symlink_gitlink_and_nonblob_rows_forbidden",
            "snapshot_files_mode",
            "snapshot_directories_mode",
            "snapshot_mount",
            "anchored_dirfd_nofollow_only",
            "symlink_fifo_socket_device_forbidden",
            "fsync_each_file_and_directory",
            "parent_identity_pre_and_post_required",
            "pre_and_post_manifest_replay",
            "pre_and_post_commit_revalidation",
        },
        "source_snapshot_policy",
    )
    _require_type_exact_v1(
        snapshot,
        {
            "commit_wire": "FULL_LOWERCASE_GIT_SHA1_40_HEX",
            "head_must_equal_requested_commit": True,
            "worktree_and_index_must_be_completely_clean": True,
            "materialization": "GIT_LS_TREE_AND_CAT_FILE_BLOB_ONLY",
            "worktree_copy_forbidden": True,
            "symlink_gitlink_and_nonblob_rows_forbidden": True,
            "snapshot_files_mode": "0444_OR_0555_FROM_COMMIT_MODE",
            "snapshot_directories_mode": "0555",
            "snapshot_mount": "READ_ONLY",
            "anchored_dirfd_nofollow_only": True,
            "symlink_fifo_socket_device_forbidden": True,
            "fsync_each_file_and_directory": True,
            "parent_identity_pre_and_post_required": True,
            "pre_and_post_manifest_replay": True,
            "pre_and_post_commit_revalidation": True,
        },
        "source_snapshot_policy",
    )
    allowlists = _exact_dict(
        config["source_allowlist_policy"],
        {
            "schema_version",
            "actor_rows",
            "path_registry_hash_preimage",
            "project_subdirectory_prefix_from_git_show_prefix",
            "git_queries_use_prefix_plus_project_relative_path",
            "snapshot_manifest_paths_remain_project_relative",
            "source_identity_digest_preimage",
            "rust_compile_env_uses_27_path_source_identity_digest",
            "json_manifest_digest_must_not_replace_source_identity_digest",
            "receipt_binds_commit_rows_and_source_identity_digest",
        },
        "source_allowlist_policy",
    )
    allowlist_rows = [
        [
            ordinal,
            actor_id,
            len(ACTOR_SOURCE_ALLOWLISTS[actor_id]),
            sha256(
                _canonical_json_bytes(list(ACTOR_SOURCE_ALLOWLISTS[actor_id]))
            ).hexdigest(),
        ]
        for ordinal, actor_id in enumerate(
            ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"),
            start=1,
        )
    ]
    _require_type_exact_v1(
        allowlists,
        {
            "schema_version": "hegel-phase3a-q05b-actor-source-allowlists/1",
            "actor_rows": allowlist_rows,
            "path_registry_hash_preimage": (
                "CANONICAL_JSON_ORDERED_PROJECT_RELATIVE_PATH_ARRAY_PLUS_LF"
            ),
            "project_subdirectory_prefix_from_git_show_prefix": True,
            "git_queries_use_prefix_plus_project_relative_path": True,
            "snapshot_manifest_paths_remain_project_relative": True,
            "source_identity_digest_preimage": (
                "SHA256_SUM_U32BE_PATHLEN_PATH_U64BE_BLOBLEN_BLOB"
            ),
            "rust_compile_env_uses_27_path_source_identity_digest": True,
            "json_manifest_digest_must_not_replace_source_identity_digest": True,
            "receipt_binds_commit_rows_and_source_identity_digest": True,
        },
        "source_allowlist_policy",
    )
    rust = _exact_dict(
        config["rust_build_policy"],
        {
            "network",
            "pull_policy",
            "cargo_home",
            "direct_external_cache_mount_forbidden",
            "lock_checksum_verified_crate_archives_required",
            "safe_regular_member_only_unpack_required",
            "minimal_registry_index_and_config_bound",
            "sealed_cargo_home_file_manifest_bound",
            "sealed_cargo_home_mount",
            "cargo_target_output",
            "actual_offline_build_policy_test_required",
            "commands",
            "compile_time_source_identity_env",
            "compile_time_source_identity_value",
            "runtime_uses_prebuilt_binary_only",
            "runtime_source_mount_forbidden",
            "runtime_cargo_invocation_forbidden",
            "runtime_rehashes_proc_self_exe",
            "runtime_reports_embedded_source_identity",
            "binary_sha256_replayed_before_and_after_runtime",
        },
        "rust_build_policy",
    )
    _require_type_exact_v1(
        rust,
        {
            "network": "none",
            "pull_policy": "never",
            "cargo_home": (
                "COMMIT_LOCK_DERIVED_PREUNPACKED_SEALED_READ_ONLY_HOME"
            ),
            "direct_external_cache_mount_forbidden": True,
            "lock_checksum_verified_crate_archives_required": True,
            "safe_regular_member_only_unpack_required": True,
            "minimal_registry_index_and_config_bound": True,
            "sealed_cargo_home_file_manifest_bound": True,
            "sealed_cargo_home_mount": "READ_ONLY",
            "cargo_target_output": "CONTROLLED_WRITABLE_EPHEMERAL",
            "actual_offline_build_policy_test_required": True,
            "commands": [
                ["cargo", "test", "--locked", "--offline", "--all-targets"],
                [
                    "cargo",
                    "build",
                    "--locked",
                    "--offline",
                    "--release",
                    "--bin",
                    "hegel-q1-archive-projection-oracle",
                ],
            ],
            "compile_time_source_identity_env": (
                "HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256"
            ),
            "compile_time_source_identity_value": (
                "COMMIT_A_RUST_EXACT_ALLOWLIST_RAW_FRAMED_SOURCE_IDENTITY_SHA256_HEX"
            ),
            "runtime_uses_prebuilt_binary_only": True,
            "runtime_source_mount_forbidden": True,
            "runtime_cargo_invocation_forbidden": True,
            "runtime_rehashes_proc_self_exe": True,
            "runtime_reports_embedded_source_identity": True,
            "binary_sha256_replayed_before_and_after_runtime": True,
        },
        "rust_build_policy",
    )
    commands = _exact_dict(
        config["actor_commands"],
        {"python", "rust", "trusted_host"},
        "actor_commands",
    )
    _require_type_exact_v1(
        commands,
        {
            "python": [
                "/usr/local/bin/python3",
                "-I",
                "-S",
                "-B",
                "/snapshot/tools/phase3_q1_python_projection_entrypoint_v1.py",
                "--action",
                "bounded-node3-golden-v1",
                "--output-dir",
                "/output",
            ],
            "rust": [
                "/runtime/hegel-q1-archive-projection-oracle",
                "--action",
                "bounded-node3-golden-v1",
                "--output-dir",
                "/output",
            ],
            "trusted_host": [
                "/usr/local/bin/python3",
                "-I",
                "-S",
                "-B",
                "/snapshot/tools/phase3_q05b_dual_qualification_v1.py",
                "--internal-host-replay",
                "--python-output",
                "/inputs/python",
                "--rust-output",
                "/inputs/rust",
                "--python-stdout",
                "/inputs/stdout/python.stdout",
                "--rust-stdout",
                "/inputs/stdout/rust.stdout",
                "--stdout-manifest",
                "/inputs/stdout/manifest.json",
                "--staging-output",
                "/staging",
                "--host-source-identity-root-hex",
                "COMMIT_A_HOST_SOURCE_IDENTITY_ROOT_HEX",
                "--host-runtime-identity-root-hex",
                "PINNED_HOST_RUNTIME_IDENTITY_ROOT_HEX",
            ],
        },
        "actor_commands",
    )
    artifact = _exact_dict(
        config["artifact_layout"],
        {
            "relative_path",
            "format",
            "mode",
            "sidecar_payload_encoding",
            "sidecar_rows",
            "required_evidence",
            "artifact_set_root_domain",
            "atomic_publication",
        },
        "artifact_layout",
    )
    _require_type_exact_v1(
        artifact,
        {
            "relative_path": "artifacts/phase3_q05b_dual_qualification_v1.json",
            "format": "ONE_CANONICAL_JSON_OBJECT_PLUS_LF",
            "mode": "0444",
            "sidecar_payload_encoding": "LOWERCASE_HEX_OF_EXACT_CBOR_BYTES",
            "sidecar_rows": [
                [0, "preimages/000-full-v16-leaf-manifest-v1.cbor", 292, 70244],
                [1, "preimages/001-odd-node3-partition-evidence-v1.cbor", 292, 1244549],
                [2, "preimages/002-sink-node3-partition-evidence-v1.cbor", 292, 1078063],
                [3, "neutral/q05b-node3-sidecar-manifest-v1.cbor", 292, 552],
                [4, "neutral/q05b-node3-golden-manifest-v1.cbor", 292, 4134],
            ],
            "required_evidence": [
                "python_stdout_hex_and_sha256",
                "rust_stdout_hex_and_sha256",
                "sealed_stdout_set_manifest_hex_length_sha256",
                "five_sidecar_cbor_hex_length_sha256_content_root",
                "python_rust_host_byte_equality_evidence",
                "host_strict_replay_and_shadow_assembler",
                "isolation_source_runtime_resource_evidence",
                "three_live_container_resource_transcripts",
                "three_actor_commit_blob_rows_and_source_identity_digests",
                "sealed_cargo_home_file_manifest_and_lock_checksums",
                "candidate_receipt_cbor_hex_and_root",
                "final_receipt_cbor_hex_and_root",
            ],
            "artifact_set_root_domain": "HEGEL/Q05B/QUALIFICATION/ARTIFACT_SET/V1",
            "atomic_publication": (
                "DIRFD_NOFOLLOW_FSYNC_LINK_NOREPLACE_UNLINK_FSYNC"
            ),
        },
        "artifact_layout",
    )


def _git(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(FAIL_SOURCE, f"Git inspection failed: {error}")
    return completed.stdout.strip()


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(FAIL_SOURCE, f"Git blob inspection failed: {error}")
    return completed.stdout


def git_project_prefix_v1(project_root: Path) -> str:
    prefix = _git(project_root, "rev-parse", "--show-prefix")
    if prefix == "":
        return ""
    if (
        prefix.startswith("/")
        or not prefix.endswith("/")
        or ".." in Path(prefix).parts
        or Path(prefix).as_posix() != prefix.rstrip("/")
    ):
        _fail(FAIL_SOURCE, "Git project subdirectory prefix differs")
    return prefix


def git_repository_root_v1(project_root: Path) -> Path:
    raw = _git(project_root, "rev-parse", "--show-toplevel")
    repository = Path(raw)
    if not repository.is_absolute():
        _fail(FAIL_SOURCE, "Git repository root is not absolute")
    try:
        value = repository.lstat()
    except OSError as error:
        _fail(FAIL_SOURCE, f"Git repository root is unavailable: {error}")
    if repository.is_symlink() or not stat.S_ISDIR(value.st_mode):
        _fail(FAIL_SOURCE, "Git repository root must be a nonsymlink directory")
    return repository


def _git_tree_path_v1(project_root: Path, relative_path: str) -> str:
    return git_project_prefix_v1(project_root) + relative_path


@dataclass(frozen=True, slots=True)
class GitBlobRowV1:
    path: str
    mode: int
    object_id: str
    size: int
    sha256_hex: str

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or self.path.startswith("/")
            or ".." in Path(self.path).parts
            or Path(self.path).as_posix() != self.path
            or type(self.mode) is not int
            or self.mode not in (0o100644, 0o100755)
            or re.fullmatch(r"[0-9a-f]{40}", self.object_id) is None
            or type(self.size) is not int
            or self.size < 0
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256_hex) is None
        ):
            _fail(FAIL_SOURCE, "Git blob manifest row differs")


def git_blob_row_v1(project_root: Path, commit: str, relative_path: str) -> GitBlobRowV1:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _fail(FAIL_SOURCE, "Git blob commit must be full lowercase 40-hex")
    if (
        type(relative_path) is not str
        or relative_path.startswith("/")
        or ".." in Path(relative_path).parts
        or Path(relative_path).as_posix() != relative_path
    ):
        _fail(FAIL_SOURCE, "Git blob relative path differs")
    repository_root = git_repository_root_v1(project_root)
    if _git(repository_root, "cat-file", "-t", commit) != "commit":
        _fail(FAIL_SOURCE, "Git source identity is not a commit")
    tree_path = _git_tree_path_v1(project_root, relative_path)
    listing = _git(repository_root, "ls-tree", commit, "--", tree_path)
    match = re.fullmatch(r"(100644|100755) blob ([0-9a-f]{40})\t(.+)", listing)
    if match is None or match.group(3) != tree_path:
        _fail(FAIL_SOURCE, "Git tree row is absent, symlink, gitlink, or nonblob")
    payload = _git_bytes(repository_root, "cat-file", "blob", f"{commit}:{tree_path}")
    return GitBlobRowV1(
        relative_path,
        0o100755 if match.group(1) == "100755" else 0o100644,
        match.group(2),
        len(payload),
        sha256(payload).hexdigest(),
    )


def git_blob_manifest_v1(
    project_root: Path,
    commit: str,
    relative_paths: Sequence[str],
) -> tuple[GitBlobRowV1, ...]:
    if type(relative_paths) not in (tuple, list) or not relative_paths:
        _fail(FAIL_SOURCE, "Git blob allowlist must be a nonempty ordered sequence")
    if any(type(path) is not str for path in relative_paths):
        _fail(FAIL_SOURCE, "Git blob allowlist path type differs")
    ordered = tuple(relative_paths)
    if ordered != tuple(sorted(ordered)) or len(set(ordered)) != len(ordered):
        _fail(FAIL_SOURCE, "Git blob allowlist must be sorted and unique")
    return tuple(git_blob_row_v1(project_root, commit, path) for path in ordered)


def git_blob_manifest_sha256_v1(rows: tuple[GitBlobRowV1, ...]) -> str:
    if type(rows) is not tuple or not rows:
        _fail(FAIL_SOURCE, "Git blob manifest rows differ")
    payload = _canonical_json_bytes(
        [
            [row.path, row.mode, row.object_id, row.size, row.sha256_hex]
            for row in rows
        ]
    )
    return sha256(payload).hexdigest()


def git_source_identity_digest_v1(
    project_root: Path,
    commit: str,
    rows: tuple[GitBlobRowV1, ...],
) -> str:
    """Compute the actor algorithm, never the JSON manifest digest."""

    if type(rows) is not tuple or not rows:
        _fail(FAIL_SOURCE, "source identity rows differ")
    repository_root = git_repository_root_v1(project_root)
    digest = sha256()
    for expected in rows:
        if git_blob_row_v1(project_root, commit, expected.path) != expected:
            _fail(FAIL_SOURCE, f"source identity Git row changed: {expected.path}")
        payload = _git_bytes(
            repository_root,
            "cat-file",
            "blob",
            f"{commit}:{_git_tree_path_v1(project_root, expected.path)}",
        )
        path_bytes = expected.path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(4, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def actor_source_evidence_v1(
    project_root: Path,
    commit: str,
    actor_id: str,
) -> dict[str, object]:
    paths = ACTOR_SOURCE_ALLOWLISTS.get(actor_id)
    if paths is None:
        _fail(FAIL_SOURCE, "actor source allowlist identity differs")
    rows = git_blob_manifest_v1(project_root, commit, paths)
    repository_root = git_repository_root_v1(project_root)
    blob_preimages: list[list[object]] = []
    framed = sha256()
    for row in rows:
        payload = _git_bytes(
            repository_root,
            "cat-file",
            "blob",
            f"{commit}:{_git_tree_path_v1(project_root, row.path)}",
        )
        path_bytes = row.path.encode("utf-8")
        framed.update(len(path_bytes).to_bytes(4, "big"))
        framed.update(path_bytes)
        framed.update(len(payload).to_bytes(8, "big"))
        framed.update(payload)
        blob_preimages.append(
            [
                row.path,
                row.mode,
                row.object_id,
                row.size,
                row.sha256_hex,
                payload.hex(),
            ]
        )
    path_registry_sha256 = sha256(_canonical_json_bytes(list(paths))).hexdigest()
    source_identity = git_source_identity_digest_v1(
        project_root,
        commit,
        rows,
    )
    if framed.hexdigest() != source_identity:
        _fail(FAIL_SOURCE, "actor raw blob preimages differ from source identity")
    return {
        "schema_version": "hegel-phase3a-q05b-actor-source-evidence/1",
        "actor_id": actor_id,
        "commit": commit,
        "project_git_prefix": git_project_prefix_v1(project_root),
        "path_registry_sha256": path_registry_sha256,
        "source_identity_sha256": source_identity,
        "rows": [
            [row.path, row.mode, row.object_id, row.size, row.sha256_hex]
            for row in rows
        ],
        "blob_preimage_rows": blob_preimages,
    }


def _parse_raw_git_tree_v1(payload: bytes) -> dict[str, tuple[str, str]]:
    if type(payload) is not bytes or not payload:
        _fail(FAIL_SOURCE, "raw Git tree payload differs")
    offset = 0
    rows: dict[str, tuple[str, str]] = {}
    while offset < len(payload):
        space = payload.find(b" ", offset)
        nul = payload.find(b"\x00", space + 1) if space >= 0 else -1
        if space <= offset or nul <= space + 1 or nul + 21 > len(payload):
            _fail(FAIL_SOURCE, "raw Git tree framing differs")
        mode_bytes = payload[offset:space]
        name_bytes = payload[space + 1 : nul]
        object_id = payload[nul + 1 : nul + 21].hex()
        try:
            mode = mode_bytes.decode("ascii", "strict")
            name = name_bytes.decode("utf-8", "strict")
        except UnicodeError as error:
            _fail(FAIL_SOURCE, f"raw Git tree text differs: {error}")
        if (
            re.fullmatch(r"(?:100644|100755|40000)", mode) is None
            or name in ("", ".", "..")
            or "/" in name
            or name in rows
            or re.fullmatch(r"[0-9a-f]{40}", object_id) is None
        ):
            _fail(FAIL_SOURCE, "raw Git tree row differs")
        rows[name] = (mode, object_id)
        offset = nul + 21
    return rows


def git_source_object_closure_evidence_v1(
    project_root: Path,
    commit: str,
    relative_paths: Sequence[str],
) -> dict[str, object]:
    """Bind raw commit plus every de-duplicated tree object to allowlist blobs."""

    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or type(relative_paths) not in (tuple, list)
        or not relative_paths
        or tuple(relative_paths) != tuple(sorted(set(relative_paths)))
    ):
        _fail(FAIL_SOURCE, "Git source object closure inputs differ")
    repository_root = git_repository_root_v1(project_root)
    commit_payload = _git_bytes(repository_root, "cat-file", "commit", commit)
    if sha1(b"commit " + str(len(commit_payload)).encode("ascii") + b"\x00" + commit_payload).hexdigest() != commit:
        _fail(FAIL_SOURCE, "raw Git commit payload does not reproduce object id")
    tree_match = re.match(rb"tree ([0-9a-f]{40})\n", commit_payload)
    if tree_match is None:
        _fail(FAIL_SOURCE, "raw Git commit lacks canonical root tree row")
    root_tree_id = tree_match.group(1).decode("ascii")
    prefix_with_slash = git_project_prefix_v1(project_root)
    project_tree_prefix = prefix_with_slash[:-1] if prefix_with_slash else ""
    if (
        project_tree_prefix.endswith("/")
        or project_tree_prefix.startswith("/")
        or ".." in PurePosixPath(project_tree_prefix).parts
    ):
        _fail(FAIL_SOURCE, "normalized project tree prefix differs")

    tree_payloads: dict[str, bytes] = {}

    def tree_rows(object_id: str) -> dict[str, tuple[str, str]]:
        payload = _git_bytes(repository_root, "cat-file", "tree", object_id)
        expected = sha1(
            b"tree " + str(len(payload)).encode("ascii") + b"\x00" + payload
        ).hexdigest()
        if expected != object_id:
            _fail(FAIL_SOURCE, "raw Git tree payload does not reproduce object id")
        prior = tree_payloads.get(object_id)
        if prior is not None and prior != payload:
            _fail(FAIL_SOURCE, "raw Git tree object id collision")
        tree_payloads[object_id] = payload
        return _parse_raw_git_tree_v1(payload)

    project_tree_id = root_tree_id
    for component in PurePosixPath(project_tree_prefix).parts if project_tree_prefix else ():
        row = tree_rows(project_tree_id).get(component)
        if row is None or row[0] != "40000":
            _fail(FAIL_SOURCE, "project prefix tree component differs")
        project_tree_id = row[1]
    # The project tree itself is always part of the closure, including when
    # the project is the repository root.
    tree_rows(project_tree_id)
    blob_rows = {
        row.path: row
        for row in git_blob_manifest_v1(project_root, commit, relative_paths)
    }
    for relative in relative_paths:
        parts = PurePosixPath(relative).parts
        current_tree_id = project_tree_id
        for index, component in enumerate(parts):
            row = tree_rows(current_tree_id).get(component)
            if row is None:
                _fail(FAIL_SOURCE, f"allowlist path absent from raw tree: {relative}")
            if index < len(parts) - 1:
                if row[0] != "40000":
                    _fail(FAIL_SOURCE, f"allowlist parent is not a tree: {relative}")
                current_tree_id = row[1]
            else:
                expected_blob = blob_rows[relative]
                if row[0] not in ("100644", "100755") or row[1] != expected_blob.object_id:
                    _fail(FAIL_SOURCE, f"allowlist blob/raw-tree binding differs: {relative}")
    tree_object_rows = [
        [object_id, tree_payloads[object_id].hex()]
        for object_id in sorted(tree_payloads)
    ]
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-git-source-object-closure/1",
        "commit": commit,
        "commit_payload_hex": commit_payload.hex(),
        "commit_payload_sha256": sha256(commit_payload).hexdigest(),
        "root_tree_object_id": root_tree_id,
        "project_tree_prefix": project_tree_prefix,
        "project_tree_object_id": project_tree_id,
        "allowlist_union": list(relative_paths),
        "tree_object_rows": tree_object_rows,
    }
    value["closure_sha256"] = sha256(_canonical_json_bytes(value)).hexdigest()
    return value


def _open_snapshot_directory_v1(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    current = os.dup(root_descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        for part in parts:
            if part in ("", ".", "..") or "/" in part:
                _fail(FAIL_SOURCE, "snapshot directory component differs")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
            following = os.open(part, directory_flags, dir_fd=current)
            value = os.fstat(following)
            if not stat.S_ISDIR(value.st_mode):
                os.close(following)
                _fail(FAIL_SOURCE, "snapshot path component is not a directory")
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _snapshot_file_paths_v1(root_descriptor: int) -> tuple[str, ...]:
    observed: list[str] = []

    def visit(directory: int, prefix: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory)):
            if name in ("", ".", "..") or "/" in name:
                _fail(FAIL_SOURCE, "snapshot entry name differs")
            value = os.stat(name, dir_fd=directory, follow_symlinks=False)
            relative = prefix + (name,)
            if stat.S_ISDIR(value.st_mode):
                child = _open_snapshot_directory_v1(directory, (name,), create=False)
                try:
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(value.st_mode):
                observed.append("/".join(relative))
            else:
                _fail(FAIL_SOURCE, "snapshot contains a symlink or special file")

    visit(root_descriptor, ())
    return tuple(sorted(observed))


def _read_snapshot_file_v1(
    root_descriptor: int,
    relative_path: str,
) -> tuple[bytes, os.stat_result]:
    parts = Path(relative_path).parts
    parent = _open_snapshot_directory_v1(
        root_descriptor,
        tuple(parts[:-1]),
        create=False,
    )
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(parts[-1], flags, dir_fd=parent)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(FAIL_SOURCE, "snapshot file is not regular or has another hard link")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            blocks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or total != before.st_size:
            _fail(FAIL_SOURCE, "snapshot file changed during anchored replay")
        return b"".join(blocks), after
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def sealed_snapshot_identity_v1(
    root: Path,
    relative_paths: Sequence[str],
) -> dict[str, object]:
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or type(relative_paths) not in (tuple, list)
        or not relative_paths
        or tuple(relative_paths) != tuple(sorted(relative_paths))
    ):
        _fail(FAIL_SOURCE, "sealed snapshot identity inputs differ")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        root_status = os.fstat(descriptor)
        root_path_status = root.lstat()
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or stat.S_IMODE(root_status.st_mode) != 0o555
            or (root_status.st_dev, root_status.st_ino)
            != (root_path_status.st_dev, root_path_status.st_ino)
            or _snapshot_file_paths_v1(descriptor) != tuple(relative_paths)
        ):
            _fail(FAIL_SOURCE, "sealed snapshot root identity differs")
        rows: list[list[object]] = []
        for relative in relative_paths:
            payload, value = _read_snapshot_file_v1(descriptor, relative)
            rows.append(
                [
                    relative,
                    value.st_dev,
                    value.st_ino,
                    value.st_nlink,
                    value.st_uid,
                    value.st_gid,
                    stat.S_IMODE(value.st_mode),
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                    sha256(payload).hexdigest(),
                ]
            )
        value = {
            "schema_version": "hegel-phase3a-q05b-sealed-snapshot-identity/1",
            "root_device": root_status.st_dev,
            "root_inode": root_status.st_ino,
            "root_mode": stat.S_IMODE(root_status.st_mode),
            "file_rows": rows,
        }
        value["manifest_sha256"] = sha256(_canonical_json_bytes(value)).hexdigest()
        return value
    finally:
        os.close(descriptor)


def replay_sealed_snapshot_identity_v1(
    root: Path,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    if type(evidence) is not dict or type(evidence.get("file_rows")) is not list:
        _fail(FAIL_SOURCE, "sealed snapshot evidence shape differs")
    paths = tuple(row[0] for row in evidence["file_rows"] if type(row) is list)
    if len(paths) != len(evidence["file_rows"]):
        _fail(FAIL_SOURCE, "sealed snapshot evidence rows differ")
    replay = sealed_snapshot_identity_v1(root, paths)
    if replay != evidence:
        _fail(FAIL_SOURCE, "sealed snapshot changed across actor execution")
    return replay


def materialize_git_blob_snapshot_v1(
    project_root: Path,
    commit: str,
    rows: tuple[GitBlobRowV1, ...],
    destination: Path,
) -> None:
    if (
        not isinstance(destination, Path)
        or not destination.is_absolute()
        or destination.name in ("", ".", "..")
        or type(rows) is not tuple
        or not rows
    ):
        _fail(FAIL_SOURCE, "Git blob snapshot destination or manifest differs")
    if tuple(row.path for row in rows) != tuple(sorted(row.path for row in rows)):
        _fail(FAIL_SOURCE, "Git blob snapshot manifest order differs")
    head_before = _git(project_root, "rev-parse", "--verify", "HEAD")
    repository_root = git_repository_root_v1(project_root)
    replay_before = git_blob_manifest_v1(
        project_root,
        commit,
        tuple(row.path for row in rows),
    )
    if replay_before != rows:
        _fail(FAIL_SOURCE, "Git blob pre-materialization manifest differs")
    parent = destination.parent
    try:
        parent_status = parent.lstat()
    except OSError as error:
        _fail(FAIL_SOURCE, f"snapshot parent is unavailable: {error}")
    if parent.is_symlink() or not stat.S_ISDIR(parent_status.st_mode):
        _fail(FAIL_SOURCE, "snapshot parent must be one nonsymlink directory")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, directory_flags)
    root_descriptor: int | None = None
    try:
        anchored_parent = os.fstat(parent_descriptor)
        if (
            anchored_parent.st_dev != parent_status.st_dev
            or anchored_parent.st_ino != parent_status.st_ino
        ):
            _fail(FAIL_SOURCE, "snapshot parent changed before anchored open")
        os.mkdir(destination.name, 0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        root_descriptor = os.open(
            destination.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        for expected in rows:
            replay = git_blob_row_v1(project_root, commit, expected.path)
            if replay != expected:
                _fail(FAIL_SOURCE, f"Git blob row changed: {expected.path}")
            payload = _git_bytes(
                repository_root,
                "cat-file",
                "blob",
                f"{commit}:{_git_tree_path_v1(project_root, expected.path)}",
            )
            if (
                len(payload) != expected.size
                or sha256(payload).hexdigest() != expected.sha256_hex
            ):
                _fail(FAIL_SOURCE, f"Git blob payload changed: {expected.path}")
            parts = Path(expected.path).parts
            output_parent = _open_snapshot_directory_v1(
                root_descriptor,
                tuple(parts[:-1]),
                create=True,
            )
            descriptor: int | None = None
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(parts[-1], flags, 0o400, dir_fd=output_parent)
                offset = 0
                while offset < len(payload):
                    offset += os.write(descriptor, payload[offset:])
                os.fsync(descriptor)
                os.fchmod(
                    descriptor,
                    0o555 if expected.mode == 0o100755 else 0o444,
                )
                os.fsync(descriptor)
                os.fsync(output_parent)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                os.close(output_parent)
        directories = {
            tuple(Path(row.path).parts[:depth])
            for row in rows
            for depth in range(1, len(Path(row.path).parts))
        }
        for parts in sorted(directories, key=lambda item: (-len(item), item)):
            descriptor = _open_snapshot_directory_v1(
                root_descriptor,
                parts,
                create=False,
            )
            try:
                os.fchmod(descriptor, 0o555)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fchmod(root_descriptor, 0o555)
        os.fsync(root_descriptor)
        observed = _snapshot_file_paths_v1(root_descriptor)
        if observed != tuple(row.path for row in rows):
            _fail(FAIL_SOURCE, "materialized Git blob snapshot file set differs")
        for expected in rows:
            payload, value = _read_snapshot_file_v1(root_descriptor, expected.path)
            expected_mode = 0o555 if expected.mode == 0o100755 else 0o444
            if (
                stat.S_IMODE(value.st_mode) != expected_mode
                or len(payload) != expected.size
                or sha256(payload).hexdigest() != expected.sha256_hex
            ):
                _fail(FAIL_SOURCE, f"materialized Git blob differs: {expected.path}")
        replay_after = git_blob_manifest_v1(
            project_root,
            commit,
            tuple(row.path for row in rows),
        )
        if replay_after != rows or _git(project_root, "rev-parse", "--verify", "HEAD") != head_before:
            _fail(FAIL_SOURCE, "Git blob post-materialization replay differs")
        anchored_root = os.fstat(root_descriptor)
        destination_after = destination.lstat()
        if (destination_after.st_dev, destination_after.st_ino) != (
            anchored_root.st_dev,
            anchored_root.st_ino,
        ):
            _fail(FAIL_SOURCE, "snapshot root path changed during materialization")
        os.fsync(parent_descriptor)
        parent_after = parent.lstat()
        if (parent_after.st_dev, parent_after.st_ino) != (
            anchored_parent.st_dev,
            anchored_parent.st_ino,
        ):
            _fail(FAIL_SOURCE, "snapshot parent path changed during materialization")
    except FileExistsError:
        _fail(FAIL_SOURCE, "Git blob snapshot destination already exists")
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)


def materialize_actor_git_blob_snapshot_v1(
    project_root: Path,
    commit: str,
    actor_id: str,
    destination: Path,
) -> dict[str, object]:
    """Materialize one exact actor allowlist and return receipt-ready evidence."""

    before = actor_source_evidence_v1(project_root, commit, actor_id)
    rows = git_blob_manifest_v1(
        project_root,
        commit,
        ACTOR_SOURCE_ALLOWLISTS[actor_id],
    )
    materialize_git_blob_snapshot_v1(project_root, commit, rows, destination)
    after = actor_source_evidence_v1(project_root, commit, actor_id)
    if before != after:
        _fail(FAIL_SOURCE, "actor source evidence changed during snapshot seal")
    result = dict(before)
    result["sealed_snapshot_identity"] = sealed_snapshot_identity_v1(
        destination,
        ACTOR_SOURCE_ALLOWLISTS[actor_id],
    )
    return result


_LOCK_PACKAGE_BLOCK = re.compile(
    r"(?ms)^\[\[package\]\]\n(.*?)(?=^\[\[package\]\]\n|\Z)"
)


def _lock_text_field_v1(block: str, field: str) -> str | None:
    match = re.search(rf'(?m)^{re.escape(field)} = "([^"]+)"$', block)
    return None if match is None else match.group(1)


def locked_registry_packages_v1(
    lock_payload: bytes,
) -> tuple[tuple[str, str, str], ...]:
    if type(lock_payload) is not bytes:
        _fail(FAIL_SOURCE, "Cargo.lock payload type differs")
    try:
        lock_text = lock_payload.decode("utf-8", "strict")
    except UnicodeError as error:
        _fail(FAIL_SOURCE, f"Cargo.lock is not UTF-8: {error}")
    rows: list[tuple[str, str, str]] = []
    for block in _LOCK_PACKAGE_BLOCK.findall(lock_text):
        source = _lock_text_field_v1(block, "source")
        if source is None:
            continue
        name = _lock_text_field_v1(block, "name")
        version = _lock_text_field_v1(block, "version")
        checksum = _lock_text_field_v1(block, "checksum")
        if (
            name is None
            or version is None
            or checksum is None
            or source != "registry+https://github.com/rust-lang/crates.io-index"
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            _fail(FAIL_SOURCE, "Cargo.lock registry package identity differs")
        rows.append((name, version, checksum))
    ordered = tuple(sorted(rows))
    if not ordered or len(set(ordered)) != len(ordered):
        _fail(FAIL_SOURCE, "Cargo.lock registry package set differs")
    return ordered


def _read_external_regular_v1(path: Path, name: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 256 * 1024 * 1024:
            _fail(FAIL_SOURCE, f"{name} is not one bounded regular file")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            blocks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or total != before.st_size:
            _fail(FAIL_SOURCE, f"{name} changed during replay")
        return b"".join(blocks)
    except OSError as error:
        _fail(FAIL_SOURCE, f"cannot read {name}: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class SealedCargoFileV1:
    path: str
    mode: int
    size: int
    sha256_hex: str

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or self.path.startswith("/")
            or ".." in PurePosixPath(self.path).parts
            or PurePosixPath(self.path).as_posix() != self.path
            or type(self.mode) is not int
            or self.mode not in (0o100644, 0o100755)
            or type(self.size) is not int
            or self.size < 0
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256_hex) is None
        ):
            _fail(FAIL_SOURCE, "sealed Cargo file row differs")


def validate_registry_index_entry_v1(
    payload: bytes,
    locked_name: str,
    locked_version: str,
    locked_checksum: str,
) -> None:
    if (
        type(payload) is not bytes
        or len(payload) < 8
        or payload[0] != 3
        or type(locked_name) is not str
        or type(locked_version) is not str
        or re.fullmatch(r"[0-9a-f]{64}", locked_checksum) is None
    ):
        _fail(FAIL_SOURCE, "registry index cache header/input differs")
    fields = payload[5:].split(b"\x00")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) < 3 or (len(fields) - 1) % 2 != 0:
        _fail(FAIL_SOURCE, "registry index cache framing differs")
    try:
        metadata = fields[0].decode("ascii", "strict")
    except UnicodeError as error:
        _fail(FAIL_SOURCE, f"registry index metadata differs: {error}")
    if not metadata.startswith("etag: "):
        _fail(FAIL_SOURCE, "registry index metadata lacks etag")
    seen: set[str] = set()
    locked_matches = 0
    for offset in range(1, len(fields), 2):
        try:
            version = fields[offset].decode("ascii", "strict")
        except UnicodeError as error:
            _fail(FAIL_SOURCE, f"registry index version differs: {error}")
        if version in seen:
            _fail(FAIL_SOURCE, "registry index contains a duplicate version")
        seen.add(version)
        document = _strict_json_value_v1(
            fields[offset + 1],
            f"registry index {locked_name} {version}",
        )
        if type(document) is not dict:
            _fail(FAIL_SOURCE, "registry index record is not an object")
        checksum = document.get("cksum")
        if (
            document.get("name") != locked_name
            or document.get("vers") != version
            or type(checksum) is not str
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            _fail(FAIL_SOURCE, "registry index record identity differs")
        if version == locked_version:
            locked_matches += 1
            if checksum != locked_checksum:
                _fail(FAIL_SOURCE, "registry index checksum differs from Cargo.lock")
    if locked_matches != 1:
        _fail(FAIL_SOURCE, "locked registry version is absent or duplicated")


def sealed_cargo_home_material_v1(
    lock_payload: bytes,
    external_cache: Path,
) -> tuple[tuple[SealedCargoFileV1, bytes], ...]:
    """Derive a complete pre-unpacked Cargo home from checksum-bound crates."""

    dependencies = locked_registry_packages_v1(lock_payload)
    material: dict[str, tuple[SealedCargoFileV1, bytes]] = {}

    def add(path: str, payload: bytes, mode: int = 0o100644) -> None:
        row = SealedCargoFileV1(path, mode, len(payload), sha256(payload).hexdigest())
        prior = material.get(path)
        if prior is not None and prior != (row, payload):
            _fail(FAIL_SOURCE, f"sealed Cargo path collision: {path}")
        material[path] = (row, payload)

    registry_ids: set[str] = set()
    for name, version, checksum in dependencies:
        stem = f"{name}-{version}"
        archives = sorted((external_cache / "registry/cache").glob(f"*/{stem}.crate"))
        if len(archives) != 1:
            _fail(FAIL_SOURCE, f"locked crate archive is not unique: {stem}")
        archive = archives[0]
        registry_id = archive.parent.name
        if re.fullmatch(r"[A-Za-z0-9._-]+", registry_id) is None:
            _fail(FAIL_SOURCE, "Cargo registry identity differs")
        registry_ids.add(registry_id)
        archive_payload = _read_external_regular_v1(archive, f"crate archive {stem}")
        if sha256(archive_payload).hexdigest() != checksum:
            _fail(FAIL_SOURCE, f"crate checksum differs from Cargo.lock: {stem}")
        add(f"registry/cache/{registry_id}/{stem}.crate", archive_payload)
        seen: set[str] = set()
        total_unpacked = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as bundle:
                for member in sorted(bundle.getmembers(), key=lambda item: item.name):
                    pure = PurePosixPath(member.name)
                    if (
                        pure.is_absolute()
                        or not pure.parts
                        or pure.parts[0] != stem
                        or any(part in ("", ".", "..") for part in pure.parts)
                    ):
                        _fail(FAIL_SOURCE, f"unsafe crate archive path: {member.name}")
                    if member.isdir():
                        continue
                    if not member.isfile() or len(pure.parts) < 2:
                        _fail(FAIL_SOURCE, f"crate member is not regular: {member.name}")
                    relative = PurePosixPath(*pure.parts[1:]).as_posix()
                    if relative in seen or member.size < 0 or member.size > 64 * 1024 * 1024:
                        _fail(FAIL_SOURCE, f"crate member identity differs: {member.name}")
                    stream = bundle.extractfile(member)
                    if stream is None:
                        _fail(FAIL_SOURCE, f"cannot extract crate member: {member.name}")
                    payload = stream.read()
                    total_unpacked += len(payload)
                    if len(payload) != member.size or total_unpacked > 512 * 1024 * 1024:
                        _fail(FAIL_SOURCE, f"crate member size differs: {member.name}")
                    seen.add(relative)
                    add(
                        f"registry/src/{registry_id}/{stem}/{relative}",
                        payload,
                        0o100755 if member.mode & 0o100 else 0o100644,
                    )
        except (OSError, tarfile.TarError) as error:
            _fail(FAIL_SOURCE, f"cannot replay crate archive {stem}: {error}")
        if not seen:
            _fail(FAIL_SOURCE, f"crate archive is empty: {stem}")
        add(f"registry/src/{registry_id}/{stem}/.cargo-ok", b'{"v":1}')
        index_root = external_cache / "registry/index" / registry_id
        candidates = sorted(
            path
            for path in (index_root / ".cache").rglob(name)
            if path.name == name
        )
        if len(candidates) != 1:
            _fail(FAIL_SOURCE, f"registry index entry is not unique: {name}")
        index_payload = _read_external_regular_v1(
            candidates[0], f"registry index entry {name}"
        )
        validate_registry_index_entry_v1(
            index_payload,
            name,
            version,
            checksum,
        )
        relative_index = candidates[0].relative_to(index_root).as_posix()
        add(f"registry/index/{registry_id}/{relative_index}", index_payload)
    if len(registry_ids) != 1:
        _fail(FAIL_SOURCE, "Cargo.lock spans multiple registry identities")
    registry_id = next(iter(registry_ids))
    config_path = external_cache / "registry/index" / registry_id / "config.json"
    add(
        f"registry/index/{registry_id}/config.json",
        _read_external_regular_v1(config_path, "registry config.json"),
    )
    add(".package-cache", b"")
    add(".package-cache-mutate", b"")
    return tuple(material[path] for path in sorted(material))


def materialize_sealed_cargo_home_v1(
    lock_payload: bytes,
    external_cache: Path,
    destination: Path,
) -> dict[str, object]:
    material = sealed_cargo_home_material_v1(lock_payload, external_cache)
    if not isinstance(destination, Path) or not destination.is_absolute():
        _fail(FAIL_SOURCE, "sealed Cargo destination differs")
    parent = destination.parent
    parent_status = parent.lstat()
    if parent.is_symlink() or not stat.S_ISDIR(parent_status.st_mode):
        _fail(FAIL_SOURCE, "sealed Cargo parent differs")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, flags)
    root_descriptor: int | None = None
    try:
        anchored = os.fstat(parent_descriptor)
        if (anchored.st_dev, anchored.st_ino) != (
            parent_status.st_dev,
            parent_status.st_ino,
        ):
            _fail(FAIL_SOURCE, "sealed Cargo parent changed")
        os.mkdir(destination.name, 0o700, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        parent_after = parent.lstat()
        if (parent_after.st_dev, parent_after.st_ino) != (
            anchored.st_dev,
            anchored.st_ino,
        ):
            _fail(FAIL_SOURCE, "sealed Cargo parent path changed")
        root_descriptor = os.open(destination.name, flags, dir_fd=parent_descriptor)
        for row, payload in material:
            parts = PurePosixPath(row.path).parts
            output_parent = _open_snapshot_directory_v1(
                root_descriptor, tuple(parts[:-1]), create=True
            )
            descriptor: int | None = None
            try:
                file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    file_flags |= os.O_NOFOLLOW
                descriptor = os.open(parts[-1], file_flags, 0o400, dir_fd=output_parent)
                offset = 0
                while offset < len(payload):
                    offset += os.write(descriptor, payload[offset:])
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o555 if row.mode == 0o100755 else 0o444)
                os.fsync(descriptor)
                os.fsync(output_parent)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                os.close(output_parent)
        directories = {
            tuple(PurePosixPath(row.path).parts[:depth])
            for row, _ in material
            for depth in range(1, len(PurePosixPath(row.path).parts))
        }
        for parts in sorted(directories, key=lambda item: (-len(item), item)):
            directory = _open_snapshot_directory_v1(
                root_descriptor, parts, create=False
            )
            try:
                os.fchmod(directory, 0o555)
                os.fsync(directory)
            finally:
                os.close(directory)
        os.fchmod(root_descriptor, 0o555)
        os.fsync(root_descriptor)
        if _snapshot_file_paths_v1(root_descriptor) != tuple(row.path for row, _ in material):
            _fail(FAIL_SOURCE, "sealed Cargo file set differs")
        for row, _ in material:
            payload, status = _read_snapshot_file_v1(root_descriptor, row.path)
            expected_mode = 0o555 if row.mode == 0o100755 else 0o444
            if (
                stat.S_IMODE(status.st_mode) != expected_mode
                or len(payload) != row.size
                or sha256(payload).hexdigest() != row.sha256_hex
            ):
                _fail(FAIL_SOURCE, f"sealed Cargo file differs: {row.path}")
        anchored_root = os.fstat(root_descriptor)
        destination_after = destination.lstat()
        if (destination_after.st_dev, destination_after.st_ino) != (
            anchored_root.st_dev,
            anchored_root.st_ino,
        ):
            _fail(FAIL_SOURCE, "sealed Cargo root path changed")
        os.fsync(parent_descriptor)
        parent_final = parent.lstat()
        if (parent_final.st_dev, parent_final.st_ino) != (
            anchored.st_dev,
            anchored.st_ino,
        ):
            _fail(FAIL_SOURCE, "sealed Cargo parent path changed after materialization")
    except FileExistsError:
        _fail(FAIL_SOURCE, "sealed Cargo destination already exists")
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)
    rows = [[row.path, row.mode, row.size, row.sha256_hex] for row, _ in material]
    preimage_rows = [
        [row.path, row.mode, payload.hex()]
        for row, payload in material
    ]
    locked_packages = locked_registry_packages_v1(lock_payload)
    return {
        "schema_version": "hegel-phase3a-q05b-sealed-cargo-home/1",
        "locked_registry_package_count": len(locked_packages),
        "locked_packages": [list(row) for row in locked_packages],
        "file_count": len(rows),
        "file_rows": rows,
        "file_preimage_rows": preimage_rows,
        "manifest_sha256": sha256(_canonical_json_bytes(rows)).hexdigest(),
        "sealed_snapshot_identity": sealed_snapshot_identity_v1(
            destination,
            tuple(row.path for row, _ in material),
        ),
        "root_mode": "0555",
        "file_modes": "0444_OR_0555",
        "cargo_home_mount": "READ_ONLY_PREUNPACKED",
    }


@dataclass(frozen=True, slots=True)
class GitSourceStatusV1:
    head: str
    clean: bool
    porcelain_line_count: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", self.head) is None:
            _fail(FAIL_SOURCE, "HEAD is not a full lowercase 40-hex commit")
        if type(self.clean) is not bool or type(self.porcelain_line_count) is not int:
            _fail(FAIL_SOURCE, "Git status types differ")


def git_source_status_v1(project_root: Path) -> GitSourceStatusV1:
    head = _git(project_root, "rev-parse", "--verify", "HEAD")
    porcelain = _git(
        project_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    lines = tuple(line for line in porcelain.splitlines() if line)
    return GitSourceStatusV1(head, not lines, len(lines))


def verify_actual_source_commit_v1(project_root: Path, requested: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", requested) is None:
        _fail(FAIL_SOURCE, "actual requires one full lowercase 40-hex commit")
    status = git_source_status_v1(project_root)
    if requested != status.head or not status.clean:
        _fail(FAIL_SOURCE, "actual requires requested HEAD and a completely clean tree")
    return status.head


def git_source_admission_transcript_v1(
    project_root: Path,
    requested: str,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Capture the two target-bound raw Git commands used at admission."""

    if (
        not isinstance(project_root, Path)
        or not project_root.is_absolute()
        or re.fullmatch(r"[0-9a-f]{40}", requested) is None
        or not callable(runner)
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Git admission transcript inputs differ")
    commands = (
        (
            "VERIFY_HEAD",
            ["git", "-C", project_root.as_posix(), "rev-parse", "--verify", "HEAD"],
        ),
        (
            "VERIFY_CLEAN_STATUS_Z",
            [
                "git", "-C", project_root.as_posix(), "status",
                "--porcelain=v1", "--untracked-files=all", "-z",
            ],
        ),
    )
    rows: list[dict[str, object]] = []
    for ordinal, (purpose, argv) in enumerate(commands, start=1):
        try:
            completed = runner(
                argv,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except BaseException as error:
            _fail(
                FAIL_ACTUAL_ADMISSION,
                f"Git admission command failed to execute: {type(error).__name__}:{error}",
            )
        returncode = getattr(completed, "returncode", None)
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
        if (
            type(returncode) is not int
            or type(stdout) is not bytes
            or type(stderr) is not bytes
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "Git admission command result differs")
        rows.append(
            {
                "ordinal": ordinal,
                "purpose": purpose,
                "argv": argv,
                "returncode": returncode,
                "stdout_hex": stdout.hex(),
                "stderr_hex": stderr.hex(),
                "stdout_sha256": sha256(stdout).hexdigest(),
                "stderr_sha256": sha256(stderr).hexdigest(),
            }
        )
    body: dict[str, object] = {
        "schema_version": ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION,
        "project_root": project_root.as_posix(),
        "requested_source_commit": requested,
        "command_rows": rows,
    }
    value = dict(body)
    value["transcript_root"] = sha256(
        ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN + _canonical_json_bytes(body)
    ).hexdigest()
    try:
        return _ACTUAL_ADMISSION.validate_git_source_transcript_v1(
            value, requested
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _docker_common_v1(
    *,
    image: str,
    name: str,
    cpuset: str,
    seccomp_host_path: Path | str,
    cidfile_host_path: Path | str,
    docker_slot_row: Mapping[str, object],
    build: bool = False,
) -> list[str]:
    tmpfs = (
        "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777"
        if build
        else "/tmp:rw,noexec,nosuid,nodev,size=2g,mode=1777"
    )
    label_tokens: list[str] = []
    if (
        type(docker_slot_row) is not dict
        or set(docker_slot_row)
            != {
                "slot_id",
                "slot",
                "container_name",
                "labels",
                "expected_container_labels",
            }
        or docker_slot_row.get("container_name") != name
        or type(docker_slot_row.get("labels")) is not list
        or len(docker_slot_row["labels"]) != 3
    ):
        _fail(FAIL_POLICY, "Docker execution slot row differs")
    expected_keys = DOCKER_EXECUTION_RESERVED_LABEL_KEYS
    for ordinal, label in enumerate(docker_slot_row["labels"]):
        if (
            type(label) is not list
            or len(label) != 2
            or label[0] != expected_keys[ordinal]
            or type(label[1]) is not str
            or not label[1]
        ):
            _fail(FAIL_POLICY, "Docker execution slot label row differs")
        label_tokens.append(f"--label={label[0]}={label[1]}")
    return [
        DOCKER_EXECUTABLE,
        f"--host={DOCKER_HOST}",
        "run",
        "--name",
        name,
        *label_tokens,
        f"--cidfile={cidfile_host_path}",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--security-opt=seccomp={seccomp_host_path}",
        "--ipc=none",
        "--cgroupns=private",
        "--pids-limit=128",
        "--ulimit=nofile=256:256",
        "--memory=14g",
        "--memory-swap=14g",
        f"--cpuset-cpus={cpuset}",
        f"--tmpfs={tmpfs}",
        f"--user={os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        "LANG=C.UTF-8",
        "-e",
        "LC_ALL=C.UTF-8",
        "-e",
        "TZ=UTC",
        image,
    ]


def _docker_execution_slot_row_v1(
    authority: Mapping[str, object],
    slot: str,
) -> dict[str, object]:
    """Select one already-pure-validated authority row without weakening it."""

    if (
        type(authority) is not dict
        or type(slot) is not str
        or slot not in {row[1] for row in DOCKER_EXECUTION_SLOT_REGISTRY}
        or authority.get("container_name_usage") != "READ_ONLY_DISCOVERY_ONLY"
        or authority.get("destructive_target")
        != "OWNERSHIP_VALIDATED_64_LOWERHEX_CONTAINER_ID_ONLY"
        or authority.get("reserved_label_keys")
        != list(DOCKER_EXECUTION_RESERVED_LABEL_KEYS)
        or type(authority.get("manifest_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", authority["manifest_sha256"]) is None
        or type(authority.get("ordered_slot_rows")) is not list
        or len(authority["ordered_slot_rows"]) != len(DOCKER_EXECUTION_SLOT_REGISTRY)
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority surface differs")
    matches = [
        row
        for row in authority["ordered_slot_rows"]
        if type(row) is dict and row.get("slot") == slot
    ]
    if len(matches) != 1:
        _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority slot differs")
    row = matches[0]
    expected_slot_id = dict((name, slot_id) for slot_id, name in DOCKER_EXECUTION_SLOT_REGISTRY)[slot]
    labels = row.get("labels")
    if (
        set(row)
        != {
            "slot_id",
            "slot",
            "container_name",
            "labels",
            "expected_container_labels",
        }
        or row.get("slot_id") != expected_slot_id
        or type(row.get("container_name")) is not str
        or re.fullmatch(
            r"hegel-q05b-[0-9a-f]{64}-(?:rust-test|rust-release|python|rust|host)",
            row["container_name"],
        )
        is None
        or type(labels) is not list
        or len(labels) != 3
        or [label[0] if type(label) is list and len(label) == 2 else None for label in labels]
        != list(DOCKER_EXECUTION_RESERVED_LABEL_KEYS)
        or any(
            type(label) is not list
            or len(label) != 2
            or type(label[1]) is not str
            or not label[1]
            for label in labels
        )
        or type(row.get("expected_container_labels")) is not list
        or any(
            type(label) is not list
            or len(label) != 2
            or type(label[0]) is not str
            or type(label[1]) is not str
            for label in row["expected_container_labels"]
        )
        or row["expected_container_labels"]
        != sorted(row["expected_container_labels"], key=lambda item: item[0])
        or len({item[0] for item in row["expected_container_labels"]})
        != len(row["expected_container_labels"])
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority slot row differs")
    return dict(row)


def _docker_execution_principal_v1(
    command: Sequence[str],
    authority: Mapping[str, object],
    slot: str,
) -> dict[str, object]:
    """Bind one exact Docker command to its admitted unique-name principal."""

    if type(command) not in (tuple, list) or any(
        type(item) is not str or not item for item in command
    ):
        _fail(FAIL_POLICY, "Docker execution principal command differs")
    exact = tuple(command)
    row = _docker_execution_slot_row_v1(authority, slot)
    name_indexes = tuple(index for index, item in enumerate(exact) if item == "--name")
    if len(name_indexes) != 1 or name_indexes[0] + 1 >= len(exact):
        _fail(FAIL_POLICY, "Docker execution principal name differs")
    name_index = name_indexes[0]
    expected_label_tokens = tuple(
        f"--label={key}={value}" for key, value in row["labels"]
    )
    observed_label_tokens = tuple(
        item for item in exact if item == "--label" or item.startswith("--label=")
    )
    if (
        exact[name_index + 1] != row["container_name"]
        or exact[name_index + 2 : name_index + 5] != expected_label_tokens
        or observed_label_tokens != expected_label_tokens
    ):
        _fail(FAIL_POLICY, "Docker execution principal labels differ")
    expected_image = (
        RUST_IMAGE
        if slot in ("RUST_TEST", "RUST_RELEASE", "RUST_ENDPOINT")
        else PYTHON_IMAGE
    )
    image_indexes = tuple(
        index for index, item in enumerate(exact) if item == expected_image
    )
    if len(image_indexes) != 1:
        _fail(FAIL_POLICY, "Docker execution principal image differs")
    labels = [list(label) for label in row["labels"]]
    return {
        "authority_manifest_sha256": authority["manifest_sha256"],
        "slot_id": row["slot_id"],
        "slot": row["slot"],
        "container_name": row["container_name"],
        "labels": labels,
        "expected_container_labels": [
            list(label) for label in row["expected_container_labels"]
        ],
        "ownership_label_root": sha256(
            DOCKER_OWNERSHIP_LABEL_ROOT_DOMAIN + _canonical_json_bytes(labels)
        ).hexdigest(),
        "image": expected_image,
        "container_argv": list(exact[image_indexes[0] + 1 :]),
        "command_sha256": sha256(_canonical_json_bytes(list(exact))).hexdigest(),
    }


def _mount(source: Path | str, target: str, *, read_only: bool) -> str:
    source_text = str(source)
    if "docker.sock" in source_text:
        _fail(FAIL_POLICY, "Docker socket may not be mounted into an actor")
    suffix = ",readonly" if read_only else ""
    return f"type=bind,src={source_text},dst={target}{suffix}"


def _held_actor_payload_v1(payload_command: Sequence[str]) -> list[str]:
    if (
        type(payload_command) not in (tuple, list)
        or not payload_command
        or any(type(item) is not str or not item for item in payload_command)
    ):
        _fail(FAIL_POLICY, "held actor payload command differs")
    return [
        "/bin/sh",
        "-ceu",
        HELD_ACTOR_WRAPPER_SCRIPT,
        "hegel-q05b-held-actor",
        *payload_command,
    ]


def python_endpoint_command_v1(
    snapshot: Path,
    output: Path,
    control: Path,
    runtime_seccomp: Path,
    *,
    docker_slot_row: Mapping[str, object],
    cidfile: Path | str = "/sealed/control/python.cid",
) -> list[str]:
    name = docker_slot_row.get("container_name")
    if type(name) is not str:
        _fail(FAIL_POLICY, "Python Docker slot container name differs")
    command = _docker_common_v1(
        image=PYTHON_IMAGE,
        name=name,
        cpuset="0-11",
        seccomp_host_path=runtime_seccomp,
        cidfile_host_path=cidfile,
        docker_slot_row=docker_slot_row,
    )
    image = command.pop()
    command.extend(
        [
            "--mount",
            _mount(snapshot, "/snapshot", read_only=True),
            "--mount",
            _mount(output, "/output", read_only=False),
            "--mount",
            _mount(control, "/control", read_only=False),
            "-w",
            "/snapshot",
            image,
            *_held_actor_payload_v1(
                [
                    "/usr/local/bin/python3",
                    "-I",
                    "-S",
                    "-B",
                    "/snapshot/tools/phase3_q1_python_projection_entrypoint_v1.py",
                    "--action",
                    "bounded-node3-golden-v1",
                    "--output-dir",
                    "/output",
                ]
            ),
        ]
    )
    return command


def rust_runtime_command_v1(
    binary: Path,
    output: Path,
    control: Path,
    runtime_seccomp: Path,
    *,
    docker_slot_row: Mapping[str, object],
    cidfile: Path | str = "/sealed/control/rust.cid",
) -> list[str]:
    name = docker_slot_row.get("container_name")
    if type(name) is not str:
        _fail(FAIL_POLICY, "Rust Docker slot container name differs")
    command = _docker_common_v1(
        image=RUST_IMAGE,
        name=name,
        cpuset="12-23",
        seccomp_host_path=runtime_seccomp,
        cidfile_host_path=cidfile,
        docker_slot_row=docker_slot_row,
    )
    image = command.pop()
    command.extend(
        [
            "--mount",
            _mount(binary, "/runtime/hegel-q1-archive-projection-oracle", read_only=True),
            "--mount",
            _mount(output, "/output", read_only=False),
            "--mount",
            _mount(control, "/control", read_only=False),
            image,
            *_held_actor_payload_v1(
                [
                    "/runtime/hegel-q1-archive-projection-oracle",
                    "--action",
                    "bounded-node3-golden-v1",
                    "--output-dir",
                    "/output",
                ]
            ),
        ]
    )
    if "cargo" in command[command.index(image) + 1 :]:
        _fail(FAIL_POLICY, "Rust runtime command must use only the prebuilt binary")
    return command


def trusted_host_command_v1(
    snapshot: Path,
    python_output: Path,
    rust_output: Path,
    python_stdout: Path,
    rust_stdout: Path,
    stdout_manifest: Path,
    control: Path,
    staging: Path,
    runtime_seccomp: Path,
    *,
    host_source_identity_root_hex: str = "0" * 64,
    host_runtime_identity_root_hex: str = "0" * 64,
    docker_slot_row: Mapping[str, object],
    cidfile: Path | str = "/sealed/control/host.cid",
) -> list[str]:
    for identity in (
        host_source_identity_root_hex,
        host_runtime_identity_root_hex,
    ):
        if type(identity) is not str or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            _fail(FAIL_SOURCE, "trusted-host command identity root differs")
    name = docker_slot_row.get("container_name")
    if type(name) is not str:
        _fail(FAIL_POLICY, "host Docker slot container name differs")
    command = _docker_common_v1(
        image=PYTHON_IMAGE,
        name=name,
        cpuset="0-11",
        seccomp_host_path=runtime_seccomp,
        cidfile_host_path=cidfile,
        docker_slot_row=docker_slot_row,
    )
    image = command.pop()
    command.extend(
        [
            "--mount",
            _mount(snapshot, "/snapshot", read_only=True),
            "--mount",
            _mount(python_output, "/inputs/python", read_only=True),
            "--mount",
            _mount(rust_output, "/inputs/rust", read_only=True),
            "--mount",
            _mount(python_stdout, "/inputs/stdout/python.stdout", read_only=True),
            "--mount",
            _mount(rust_stdout, "/inputs/stdout/rust.stdout", read_only=True),
            "--mount",
            _mount(stdout_manifest, "/inputs/stdout/manifest.json", read_only=True),
            "--mount",
            _mount(control, "/control", read_only=False),
            "--mount",
            _mount(staging, "/staging", read_only=False),
            "-w",
            "/snapshot",
            image,
            *_held_actor_payload_v1(
                [
                    "/usr/local/bin/python3",
                    "-I",
                    "-S",
                    "-B",
                    "/snapshot/tools/phase3_q05b_dual_qualification_v1.py",
                    "--internal-host-replay",
                    "--python-output",
                    "/inputs/python",
                    "--rust-output",
                    "/inputs/rust",
                    "--python-stdout",
                    "/inputs/stdout/python.stdout",
                    "--rust-stdout",
                    "/inputs/stdout/rust.stdout",
                    "--stdout-manifest",
                    "/inputs/stdout/manifest.json",
                    "--staging-output",
                    "/staging",
                    "--host-source-identity-root-hex",
                    host_source_identity_root_hex,
                    "--host-runtime-identity-root-hex",
                    host_runtime_identity_root_hex,
                ]
            ),
        ]
    )
    return command


def rust_build_commands_v1(
    snapshot: Path,
    cargo_home: Path,
    target_output: Path,
    source_identity_sha256: str = "0" * 64,
    cidfile: Path | str = "/sealed/control/rust-build.cid",
    *,
    build_seccomp: Path,
    docker_slot_row: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    if re.fullmatch(r"[0-9a-f]{64}", source_identity_sha256) is None:
        _fail(FAIL_SOURCE, "Rust compile-time source identity must be 64 lowercase hex")
    container_name = docker_slot_row.get("container_name")
    if type(container_name) is not str:
        _fail(FAIL_POLICY, "build Docker slot container name differs")
    common = _docker_common_v1(
        image=RUST_IMAGE,
        name=container_name,
        cpuset="12-23",
        seccomp_host_path=build_seccomp,
        cidfile_host_path=cidfile,
        docker_slot_row=docker_slot_row,
        build=True,
    )
    image = common.pop()
    prefix = common + [
        "--mount",
        _mount(snapshot, "/snapshot", read_only=True),
        "--mount",
        _mount(cargo_home, "/cargo-home", read_only=True),
        "--mount",
        _mount(target_output, "/target-output", read_only=False),
        "-e",
        "CARGO_HOME=/cargo-home",
        "-e",
        "CARGO_NET_OFFLINE=true",
        "-e",
        "CARGO_TARGET_DIR=/target-output",
        "-e",
        f"HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256={source_identity_sha256}",
        "-w",
        "/snapshot/rust/q1_archive_projection_oracle",
        image,
    ]
    test = prefix + ["cargo", "test", "--locked", "--offline", "--all-targets"]
    build = prefix + [
        "cargo",
        "build",
        "--locked",
        "--offline",
        "--release",
        "--bin",
        "hegel-q1-archive-projection-oracle",
    ]
    return test, build


def _exact_generated_actor_command_v1(
    role_id: int,
    container_name: str,
    cidfile: Path,
    registry: SealedActorMountRegistryV1,
    docker_slot_row: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    sources = registry.expected_sources
    seccomp = Path(registry.security_options[1].removeprefix("seccomp="))
    if role_id == 1:
        expected = python_endpoint_command_v1(
            Path(sources["/snapshot"]),
            Path(sources["/output"]),
            Path(sources["/control"]),
            seccomp,
            cidfile=cidfile,
            docker_slot_row=docker_slot_row,
        )
    elif role_id == 2:
        expected = rust_runtime_command_v1(
            Path(sources["/runtime/hegel-q1-archive-projection-oracle"]),
            Path(sources["/output"]),
            Path(sources["/control"]),
            seccomp,
            cidfile=cidfile,
            docker_slot_row=docker_slot_row,
        )
    elif role_id == 3:
        argv = registry.container_argv
        source_flag = "--host-source-identity-root-hex"
        runtime_flag = "--host-runtime-identity-root-hex"
        if (
            argv.count(source_flag) != 1
            or argv.count(runtime_flag) != 1
            or argv.index(source_flag) + 1 >= len(argv)
            or argv.index(runtime_flag) + 1 >= len(argv)
        ):
            _fail(FAIL_POLICY, "trusted-host command identity arguments differ")
        source_root = argv[argv.index(source_flag) + 1]
        runtime_root = argv[argv.index(runtime_flag) + 1]
        if (
            re.fullmatch(r"[0-9a-f]{64}", source_root) is None
            or re.fullmatch(r"[0-9a-f]{64}", runtime_root) is None
        ):
            _fail(FAIL_POLICY, "trusted-host command identity roots differ")
        expected = trusted_host_command_v1(
            Path(sources["/snapshot"]),
            Path(sources["/inputs/python"]),
            Path(sources["/inputs/rust"]),
            Path(sources["/inputs/stdout/python.stdout"]),
            Path(sources["/inputs/stdout/rust.stdout"]),
            Path(sources["/inputs/stdout/manifest.json"]),
            Path(sources["/control"]),
            Path(sources["/staging"]),
            seccomp,
            host_source_identity_root_hex=source_root,
            host_runtime_identity_root_hex=runtime_root,
            cidfile=cidfile,
            docker_slot_row=docker_slot_row,
        )
    else:
        _fail(FAIL_POLICY, "generated actor command role differs")
    return tuple(expected)


@dataclass(slots=True)
class HeldActorProcessV1:
    role_id: int
    actor_id: str
    container_name: str
    command: tuple[str, ...]
    cidfile: Path
    control_root: Path
    mount_registry: SealedActorMountRegistryV1
    process: subprocess.Popen[bytes]
    stdout_drain: "BoundedPipeDrainV1"
    stderr_drain: "BoundedPipeDrainV1"
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    sample_thread: threading.Thread | None
    sample_rows: list[dict[str, object]]
    sample_errors: list[str]
    sample_complete: threading.Event
    child_done_observed: threading.Event
    sample_stop: threading.Event
    sample_lock: threading.Lock
    container_id: str | None
    cid_parent_identity: tuple[int, int, int, int]
    cidfile_evidence: dict[str, object] | None
    cleanup_errors: list[str]
    seccomp_evidence: dict[str, object] | None = None
    docker_execution_authority_manifest_sha256: str | None = None
    docker_execution_slot_row: dict[str, object] | None = None
    ownership_label_root: str | None = None
    precreate_absence_evidence: dict[str, object] | None = None
    docker_execution_principal: dict[str, object] | None = None
    failure_cleanup_attempted: bool = False
    failure_cleanup_complete: bool = False


@dataclass(slots=True)
class BoundedPipeDrainV1:
    maximum: int
    payload: bytearray
    total: int
    overflow: bool
    digest: object
    errors: list[str]


def _unstarted_pipe_workers_v1(
    process: subprocess.Popen[bytes],
    stdout_maximum: int,
    stderr_maximum: int,
    *,
    stdout_name: str | None = None,
    stderr_name: str | None = None,
) -> tuple[
    BoundedPipeDrainV1,
    BoundedPipeDrainV1,
    threading.Thread,
    threading.Thread,
]:
    if process.stdout is None or process.stderr is None:
        _fail(FAIL_POLICY, "Docker CLI pipes are absent after Popen")
    stdout_drain = BoundedPipeDrainV1(
        stdout_maximum, bytearray(), 0, False, sha256(), []
    )
    stderr_drain = BoundedPipeDrainV1(
        stderr_maximum, bytearray(), 0, False, sha256(), []
    )
    stdout_thread = threading.Thread(
        target=_drain_pipe_v1,
        args=(process.stdout, stdout_drain),
        name=stdout_name,
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_pipe_v1,
        args=(process.stderr, stderr_drain),
        name=stderr_name,
        daemon=True,
    )
    return stdout_drain, stderr_drain, stdout_thread, stderr_thread


def _held_actor_after_popen_v1(
    role_id: int,
    actor_id: str,
    container_name: str,
    exact_command: tuple[str, ...],
    cidfile: Path,
    control_root: Path,
    mount_registry: SealedActorMountRegistryV1,
    process: subprocess.Popen[bytes],
    cid_parent_status: os.stat_result,
    seccomp_evidence: dict[str, object],
    principal: dict[str, object],
    docker_slot_row: dict[str, object],
    precreate_absence: dict[str, object],
) -> HeldActorProcessV1:
    stdout_drain, stderr_drain, stdout_thread, stderr_thread = (
        _unstarted_pipe_workers_v1(
            process,
            1024 * 1024,
            16 * 1024 * 1024,
            stdout_name=f"q05b-{role_id}-stdout-drain",
            stderr_name=f"q05b-{role_id}-stderr-drain",
        )
    )
    return HeldActorProcessV1(
        role_id,
        actor_id,
        container_name,
        exact_command,
        cidfile,
        control_root,
        mount_registry,
        process,
        stdout_drain,
        stderr_drain,
        stdout_thread,
        stderr_thread,
        None,
        [],
        [],
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Lock(),
        None,
        (
            cid_parent_status.st_dev,
            cid_parent_status.st_ino,
            stat.S_IMODE(cid_parent_status.st_mode),
            cid_parent_status.st_nlink,
        ),
        None,
        [],
        seccomp_evidence,
        principal["authority_manifest_sha256"],
        docker_slot_row,
        principal["ownership_label_root"],
        precreate_absence,
        principal,
    )


def _drain_pipe_v1(pipe, state: BoundedPipeDrainV1) -> None:
    try:
        while True:
            block = pipe.read(64 * 1024)
            if not block:
                return
            state.total += len(block)
            state.digest.update(block)
            remaining = state.maximum + 1 - len(state.payload)
            if remaining > 0:
                state.payload.extend(block[:remaining])
            if state.total > state.maximum:
                state.overflow = True
    except BaseException as error:
        state.errors.append(f"{type(error).__name__}:{error}")


def start_held_actor_process_v1(
    role_id: int,
    actor_id: str,
    container_name: str,
    command: Sequence[str],
    cidfile: Path,
    control_root: Path,
    *,
    docker_execution_authority: Mapping[str, object],
    docker_slot: str,
    inspect_reader: Callable[[str], bytes] | None = None,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    cid_timeout_seconds: float = 60.0,
    command_runner: Callable[..., object] = subprocess.run,
    ownership_sink: Callable[[HeldActorProcessV1], None] | None = None,
) -> HeldActorProcessV1:
    """Start one held actor while continuously draining both Docker pipes."""

    if (
        type(role_id) is not int
        or role_id not in (1, 2, 3)
        or actor_id != ROLE_ROWS[role_id - 1][1]
        or type(container_name) is not str
        or not container_name
        or type(command) not in (tuple, list)
        or not isinstance(cidfile, Path)
        or not cidfile.is_absolute()
        or cidfile.exists()
        or cidfile.parent.is_symlink()
        or not cidfile.parent.is_dir()
        or not isinstance(control_root, Path)
        or not control_root.is_absolute()
        or control_root.is_symlink()
        or not control_root.is_dir()
        or stat.S_IMODE(control_root.stat().st_mode) != 0o700
        or any(control_root.iterdir())
        or not callable(command_runner)
        or (ownership_sink is not None and not callable(ownership_sink))
    ):
        _fail(FAIL_POLICY, "held actor start inputs differ")
    cid_parent_status = cidfile.parent.lstat()
    if (
        not stat.S_ISDIR(cid_parent_status.st_mode)
        or stat.S_IMODE(cid_parent_status.st_mode) != 0o700
        or cid_parent_status.st_nlink < 2
    ):
        _fail(FAIL_POLICY, "held actor cid parent identity differs")
    exact_command = tuple(command)
    principal = _docker_execution_principal_v1(
        exact_command,
        docker_execution_authority,
        docker_slot,
    )
    if (
        f"--cidfile={cidfile}" not in exact_command
        or ("--name", container_name)
        not in tuple(zip(exact_command, exact_command[1:]))
        or HELD_ACTOR_WRAPPER_SCRIPT not in exact_command
        or principal["slot"] != actor_id
        or principal["container_name"] != container_name
    ):
        _fail(FAIL_POLICY, "held actor command/control binding differs")
    mount_registry = sealed_actor_mount_registry_v1(role_id, exact_command)
    if exact_command != _exact_generated_actor_command_v1(
        role_id,
        container_name,
        cidfile,
        mount_registry,
        _docker_execution_slot_row_v1(
            docker_execution_authority,
            docker_slot,
        ),
    ):
        _fail(FAIL_POLICY, "held actor command differs from exact generator")
    seccomp_path = Path(
        mount_registry.security_options[1].removeprefix("seccomp=")
    )
    seccomp_evidence = sealed_policy_file_evidence_v1(
        seccomp_path,
        RUNTIME_SECCOMP_RELATIVE_PATH,
    )
    precreate_absence = docker_precreate_absence_evidence_v1(
        docker_execution_authority,
        docker_slot,
        command_runner,
    )
    process: subprocess.Popen[bytes] | None = None
    actor: HeldActorProcessV1 | None = None
    try:
        # The outer exception frame exists before Popen and the signal guard
        # spans Popen's return plus construction of the first process-owning
        # actor object.  Thus neither CALL->STORE nor guard restoration can
        # expose a daemon/process without selecting a cleanup state below.
        with _docker_ownership_signal_guard_v1():
            process = subprocess.Popen(
                list(exact_command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                umask=0o077,
            )
            actor = _held_actor_after_popen_v1(
                role_id,
                actor_id,
                container_name,
                exact_command,
                cidfile,
                control_root,
                mount_registry,
                process,
                cid_parent_status,
                seccomp_evidence,
                principal,
                _docker_execution_slot_row_v1(
                    docker_execution_authority,
                    docker_slot,
                ),
                precreate_absence,
            )
        stdout_thread = actor.stdout_thread
        stderr_thread = actor.stderr_thread
        stdout_thread.start()
        stderr_thread.start()
        start_held_actor_resource_sampler_v1(
            actor,
            inspect_reader=inspect_reader,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
            cid_timeout_seconds=cid_timeout_seconds,
        )
        if ownership_sink is not None:
            ownership_sink(actor)
        return actor
    except BaseException as error:
        if type(actor) is HeldActorProcessV1:
            _raise_after_actor_cleanup_v1(
                (actor,),
                command_runner,
                Q05BDualSupervisorError(
                    FAIL_POLICY,
                    f"held actor worker startup failed: {error}",
                ),
                "held actor worker startup failed",
            )
        if process is not None:
            _raise_after_unbound_post_popen_cleanup_v1(
                process,
                principal,
                command_runner,
                error,
                "held actor post-Popen setup failed",
            )
        original = (
            Q05BDualSupervisorError(
                FAIL_POLICY,
                f"cannot start held actor: {error}",
            )
            if isinstance(error, OSError)
            else error
        )
        _raise_after_popen_call_failure_cleanup_v1(
            principal,
            command_runner,
            original,
            "held actor Popen failed after precreate authority",
        )
        # All branches above are NoReturn; this keeps static narrowing exact.
        raise AssertionError("unreachable held actor cleanup branch")


def _observe_unsealed_cidfile_v1(
    cidfile: Path,
    parent_identity: tuple[int, int, int, int],
    prior_file_identity: tuple[int, int] | None = None,
) -> tuple[bool, tuple[int, int] | None]:
    """Observe one Docker cidfile without sealing an incomplete write.

    Docker creates the cidfile before its container-id payload is necessarily
    visible.  Absence, an empty file, and a canonical lower-hex prefix are
    therefore pending states.  A complete result is returned only after one
    anchored, no-follow read has the pinned Docker 29.1.3 stable 64-byte
    lowercase-hex identity (with no line terminator).
    """

    if (
        not isinstance(cidfile, Path)
        or not cidfile.is_absolute()
        or cidfile.name in ("", ".", "..")
        or type(parent_identity) is not tuple
        or len(parent_identity) != 4
        or any(type(item) is not int or item < 0 for item in parent_identity)
        or (
            prior_file_identity is not None
            and (
                type(prior_file_identity) is not tuple
                or len(prior_file_identity) != 2
                or any(
                    type(item) is not int or item < 0
                    for item in prior_file_identity
                )
            )
        )
    ):
        _fail(FAIL_POLICY, "Docker cid readiness input differs")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent = os.open(cidfile.parent, directory_flags)
    except OSError as error:
        _fail(FAIL_POLICY, f"Docker cid parent cannot be anchored: {error}")
    descriptor: int | None = None
    try:
        parent_status = os.fstat(parent)
        observed_parent = (
            parent_status.st_dev,
            parent_status.st_ino,
            stat.S_IMODE(parent_status.st_mode),
            parent_status.st_nlink,
        )
        if not stat.S_ISDIR(parent_status.st_mode) or observed_parent != parent_identity:
            _fail(FAIL_POLICY, "Docker cid parent changed while waiting")
        try:
            descriptor = os.open(
                cidfile.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
        except FileNotFoundError:
            if prior_file_identity is not None:
                _fail(FAIL_POLICY, "Docker cidfile disappeared while waiting")
            return False, None
        except OSError as error:
            _fail(FAIL_POLICY, f"Docker cidfile cannot be anchored: {error}")
        before = os.fstat(descriptor)
        file_identity = (before.st_dev, before.st_ino)
        if prior_file_identity is not None and file_identity != prior_file_identity:
            _fail(FAIL_POLICY, "Docker cidfile was replaced while waiting")
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or before.st_size > 64
        ):
            _fail(FAIL_POLICY, "Docker cidfile pending identity differs")
        payload = os.read(descriptor, 65)
        overflow = os.read(descriptor, 1)
        after = os.fstat(descriptor)
        try:
            entry = os.stat(cidfile.name, dir_fd=parent, follow_symlinks=False)
        except OSError as error:
            _fail(FAIL_POLICY, f"Docker cidfile entry changed while waiting: {error}")
        held_signature = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        entry_signature = (
            entry.st_dev,
            entry.st_ino,
            entry.st_mode,
            entry.st_nlink,
            entry.st_uid,
            entry.st_gid,
            entry.st_size,
            entry.st_mtime_ns,
            entry.st_ctime_ns,
        )
        if (after.st_dev, after.st_ino) != (entry.st_dev, entry.st_ino):
            _fail(FAIL_POLICY, "Docker cidfile entry was replaced while waiting")
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_uid != os.getuid()
            or after.st_gid != os.getgid()
            or after.st_size > 64
        ):
            _fail(FAIL_POLICY, "Docker cidfile pending identity changed")
        try:
            parent_after = cidfile.parent.lstat()
        except OSError as error:
            _fail(FAIL_POLICY, f"Docker cid parent changed while waiting: {error}")
        if (
            not stat.S_ISDIR(parent_after.st_mode)
            or (
                parent_after.st_dev,
                parent_after.st_ino,
                stat.S_IMODE(parent_after.st_mode),
                parent_after.st_nlink,
            )
            != parent_identity
        ):
            _fail(FAIL_POLICY, "Docker cid parent path changed while waiting")
        before_signature = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            before_signature != held_signature
            or held_signature != entry_signature
            or len(payload) != after.st_size
        ):
            return False, file_identity
        if overflow or len(payload) > 64:
            _fail(FAIL_POLICY, "Docker cidfile is oversized")
        if re.fullmatch(rb"[0-9a-f]{64}", payload) is not None:
            return True, file_identity
        if len(payload) < 64 and all(byte in b"0123456789abcdef" for byte in payload):
            return False, file_identity
        _fail(FAIL_POLICY, "Docker cidfile payload is not a canonical prefix")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _seal_cidfile_v1(
    cidfile: Path,
    parent_identity: tuple[int, int, int, int],
    expected_file_identity: tuple[int, int] | None = None,
) -> tuple[str, dict[str, object]]:
    """Anchor one Docker cidfile, require 0600/nlink1, then seal it 0444."""

    if (
        not isinstance(cidfile, Path)
        or not cidfile.is_absolute()
        or cidfile.name in ("", ".", "..")
        or type(parent_identity) is not tuple
        or len(parent_identity) != 4
        or any(type(item) is not int or item < 0 for item in parent_identity)
        or (
            expected_file_identity is not None
            and (
                type(expected_file_identity) is not tuple
                or len(expected_file_identity) != 2
                or any(
                    type(item) is not int or item < 0
                    for item in expected_file_identity
                )
            )
        )
    ):
        _fail(FAIL_POLICY, "held actor cid seal input differs")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent = os.open(cidfile.parent, flags)
    except OSError as error:
        _fail(FAIL_POLICY, f"held actor cid parent cannot be anchored: {error}")
    descriptor: int | None = None
    try:
        parent_status = os.fstat(parent)
        observed_parent = (
            parent_status.st_dev,
            parent_status.st_ino,
            stat.S_IMODE(parent_status.st_mode),
            parent_status.st_nlink,
        )
        if (
            not stat.S_ISDIR(parent_status.st_mode)
            or observed_parent != parent_identity
        ):
            _fail(FAIL_POLICY, "held actor cid parent changed")
        descriptor = os.open(
            cidfile.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        payload = os.read(descriptor, 128)
        if os.read(descriptor, 1):
            _fail(FAIL_POLICY, "held actor cidfile is oversized")
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or before.st_size != 64
            or re.fullmatch(rb"[0-9a-f]{64}", payload) is None
            or (
                expected_file_identity is not None
                and (before.st_dev, before.st_ino) != expected_file_identity
            )
        ):
            _fail(FAIL_POLICY, "held actor cidfile identity differs")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
        ) or stat.S_IMODE(after.st_mode) != 0o444:
            _fail(FAIL_POLICY, "held actor cidfile changed while sealing")
        entry = os.stat(cidfile.name, dir_fd=parent, follow_symlinks=False)
        if (
            entry.st_dev,
            entry.st_ino,
            entry.st_mode,
            entry.st_nlink,
            entry.st_uid,
            entry.st_gid,
            entry.st_size,
            entry.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail(FAIL_POLICY, "held actor cid directory entry changed")
        os.fsync(parent)
        parent_after = os.fstat(parent)
        if (
            parent_after.st_dev,
            parent_after.st_ino,
            stat.S_IMODE(parent_after.st_mode),
            parent_after.st_nlink,
        ) != parent_identity:
            _fail(FAIL_POLICY, "held actor cid parent changed while sealing")
    except OSError as error:
        _fail(FAIL_POLICY, f"held actor cidfile cannot be sealed: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
    container_id = payload.decode("ascii")
    evidence: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-sealed-docker-cidfile/1",
        "cidfile_path": cidfile.resolve(strict=True).as_posix(),
        "relative_name": cidfile.name,
        "parent_device": parent_identity[0],
        "parent_inode": parent_identity[1],
        "parent_mode": parent_identity[2],
        "parent_nlink": parent_identity[3],
        "file_device": after.st_dev,
        "file_inode": after.st_ino,
        "file_mode": stat.S_IMODE(after.st_mode),
        "file_nlink": after.st_nlink,
        "file_uid": after.st_uid,
        "file_gid": after.st_gid,
        "file_size": after.st_size,
        "payload_hex": payload.hex(),
        "payload_sha256": sha256(payload).hexdigest(),
        "container_id": container_id,
    }
    evidence["manifest_sha256"] = sha256(_canonical_json_bytes(evidence)).hexdigest()
    return container_id, evidence


def _wait_for_cidfile_v1(
    actor: HeldActorProcessV1,
    timeout_seconds: float,
) -> str:
    if (
        type(actor) is not HeldActorProcessV1
        or type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        _fail(FAIL_POLICY, "held actor cid wait input differs")
    deadline = time.monotonic() + float(timeout_seconds)
    file_identity: tuple[int, int] | None = None
    while time.monotonic() < deadline:
        if actor.process.poll() is not None:
            _fail(FAIL_POLICY, "held actor exited before cidfile")
        ready, file_identity = _observe_unsealed_cidfile_v1(
            actor.cidfile,
            actor.cid_parent_identity,
            file_identity,
        )
        if ready:
            break
        time.sleep(0.05)
    else:
        _fail(FAIL_POLICY, "held actor cidfile timed out")
    assert file_identity is not None
    container_id, evidence = _seal_cidfile_v1(
        actor.cidfile,
        actor.cid_parent_identity,
        file_identity,
    )
    actor.container_id = container_id
    actor.cidfile_evidence = evidence
    return container_id


def _sample_actor_resources_v1(
    actor: HeldActorProcessV1,
    inspect_reader: Callable[[str], bytes],
    proc_root: Path,
    cgroup_root: Path,
    cid_timeout_seconds: float,
) -> None:
    try:
        container_id = _wait_for_cidfile_v1(actor, cid_timeout_seconds)
        prior_finish_ns: int | None = None
        ordinal = 0
        while not actor.sample_stop.is_set():
            if actor.process.poll() is not None:
                _fail(FAIL_POLICY, "held actor exited before host release")
            start_ns = time.monotonic_ns()
            if (
                prior_finish_ns is not None
                and start_ns - prior_finish_ns > 250_000_000
            ):
                _fail(FAIL_POLICY, "held actor sampling gap exceeded 250ms")
            inspect_before = inspect_reader(container_id)
            sample = collect_bound_live_resource_transcript_v1(
                actor.role_id,
                container_id,
                actor.container_name,
                actor.mount_registry,
                inspect_before,
                lambda: inspect_reader(container_id),
                seccomp_evidence=actor.seccomp_evidence,
                docker_execution_principal=actor.docker_execution_principal,
                proc_root=proc_root,
                cgroup_root=cgroup_root,
            )
            finish_ns = time.monotonic_ns()
            ordinal += 1
            sample["sample_ordinal"] = ordinal
            sample["sample_monotonic_ns"] = start_ns
            sample["sample_duration_ns"] = finish_ns - start_ns
            with actor.sample_lock:
                actor.sample_rows.append(sample)
            prior_finish_ns = finish_ns
            if os.path.lexists(actor.control_root / "done"):
                actor.child_done_observed.set()
            # The frozen 250 ms value is the maximum unsampled gap, not a
            # bound on Docker-inspect/proc/cgroup collection duration.  Event
            # waiting makes a close request interrupt this 200 ms cadence.
            if actor.sample_stop.wait(0.2):
                return
    except BaseException as error:
        actor.sample_errors.append(f"{type(error).__name__}:{error}")
    finally:
        actor.sample_complete.set()


def start_held_actor_resource_sampler_v1(
    actor: HeldActorProcessV1,
    *,
    inspect_reader: Callable[[str], bytes] | None = None,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    cid_timeout_seconds: float = 60.0,
) -> None:
    if type(actor) is not HeldActorProcessV1 or actor.sample_thread is not None:
        _fail(FAIL_POLICY, "held actor sampler start differs")
    if inspect_reader is None:
        inspect_reader = _docker_inspect_payload_v1
    if not callable(inspect_reader):
        _fail(FAIL_POLICY, "held actor sampler inspect reader differs")
    actor.sample_thread = threading.Thread(
        target=_sample_actor_resources_v1,
        args=(actor, inspect_reader, proc_root, cgroup_root, cid_timeout_seconds),
        name=f"q05b-{actor.role_id}-resource-sampler",
        daemon=True,
    )
    actor.sample_thread.start()


def _docker_inspect_payload_v1(container_id: str) -> bytes:
    completed = subprocess.run(
        [DOCKER_EXECUTABLE, f"--host={DOCKER_HOST}", "inspect", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or not completed.stdout or completed.stderr:
        _fail(FAIL_POLICY, "Docker inspect failed during held lifecycle")
    return completed.stdout


def _wait_for_held_child_done_v1(
    actor: HeldActorProcessV1,
    timeout_seconds: float,
) -> str:
    if (
        type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        _fail(FAIL_POLICY, "held actor child timeout differs")
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if actor.sample_errors:
            _fail(FAIL_POLICY, f"held actor sampler failed: {actor.sample_errors[0]}")
        if actor.process.poll() is not None:
            _fail(FAIL_POLICY, "held actor exited before host release")
        if actor.child_done_observed.wait(0.05):
            if (
                type(actor.container_id) is not str
                or re.fullmatch(r"[0-9a-f]{64}", actor.container_id) is None
                or type(actor.cidfile_evidence) is not dict
            ):
                _fail(FAIL_POLICY, "held actor done preceded sealed cid identity")
            return actor.container_id
        if actor.sample_complete.is_set():
            # The sampler publishes its error immediately before setting the
            # completion event.  It can therefore finish in the interval
            # between the loop's first error check and this completion check.
            # Replay the error slot after observing completion so the stable
            # failure preserves the underlying sampler cause rather than
            # replacing it with the generic premature-stop diagnosis.
            if actor.sample_errors:
                _fail(
                    FAIL_POLICY,
                    f"held actor sampler failed: {actor.sample_errors[0]}",
                )
            _fail(FAIL_POLICY, "held actor sampler stopped before child done")
    # The sampler can publish an error after the iteration's leading check
    # while its completion event is not yet visible.  Preserve that exact
    # cause when the same iteration consumes the remaining deadline instead
    # of replacing it with the generic child timeout.
    if actor.sample_errors:
        _fail(FAIL_POLICY, f"held actor sampler failed: {actor.sample_errors[0]}")
    _fail(FAIL_POLICY, "held actor child done timed out")


def _run_docker_control_v1(
    command: Sequence[str],
    runner: Callable[..., object],
) -> object:
    if type(command) not in (list, tuple) or not callable(runner):
        _fail(FAIL_POLICY, "Docker control command input differs")
    return runner(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _docker_absence_evidence_v1(
    container_identity: str,
    runner: Callable[..., object],
) -> dict[str, object]:
    if (
        type(container_identity) is not str
        or (
            re.fullmatch(r"[0-9a-f]{64}", container_identity) is None
            and re.fullmatch(
                r"[a-z0-9][a-z0-9_.-]{0,127}", container_identity
            )
            is None
        )
    ):
        _fail(FAIL_POLICY, "Docker absence identity differs")
    result = _run_docker_control_v1(
        [
            DOCKER_EXECUTABLE,
            f"--host={DOCKER_HOST}",
            "inspect",
            container_identity,
        ],
        runner,
    )
    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    authoritative_not_found = {
        (
            b"",
            f"Error: No such object: {container_identity}\n".encode("ascii"),
        ),
        (
            b"",
            f"Error: No such container: {container_identity}\n".encode("ascii"),
        ),
        (
            b"",
            (
                "Error response from daemon: No such container: "
                f"{container_identity}\n"
            ).encode("ascii"),
        ),
        (
            b"[]\n",
            f"error: no such object: {container_identity}\n".encode("ascii"),
        ),
    }
    if (
        type(returncode) is not int
        or returncode != 1
        or type(stdout) is not bytes
        or type(stderr) is not bytes
        or (stdout, stderr) not in authoritative_not_found
    ):
        _fail(
            FAIL_POLICY,
            "Docker inspect did not return authoritative not-found evidence",
        )
    return {
        "schema_version": "hegel-phase3a-q05b-docker-authoritative-absence/1",
        "container_identity": container_identity,
        "inspect_exit_code": returncode,
        "inspect_stdout_hex": stdout.hex(),
        "inspect_stdout_sha256": sha256(stdout).hexdigest(),
        "inspect_stderr_hex": stderr.hex(),
        "inspect_stderr_sha256": sha256(stderr).hexdigest(),
    }


def docker_precreate_absence_evidence_v1(
    authority: Mapping[str, object],
    slot: str,
    runner: Callable[..., object],
) -> dict[str, object]:
    """Collect two fresh target-bound name-absence samples immediately pre-Popen."""

    row = _docker_execution_slot_row_v1(authority, slot)
    name = row["container_name"]
    first = _docker_absence_evidence_v1(name, runner)
    second = _docker_absence_evidence_v1(name, runner)
    try:
        value = _ACTUAL_ADMISSION.build_docker_precreate_absence_v1(
            authority,
            row["slot_id"],
            first,
            second,
        )
        return _ACTUAL_ADMISSION.validate_docker_precreate_absence_v1(
            value,
            authority,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _validate_owned_docker_inspect_payload_v1(
    payload: bytes,
    principal: Mapping[str, object],
) -> dict[str, object]:
    """Require ID, unique name, reserved labels, image, and exact Cmd together."""

    if type(payload) is not bytes or not payload or type(principal) is not dict:
        _fail(FAIL_POLICY, "owned Docker inspect input differs")
    value = _strict_json_value_v1(payload, "owned Docker inspect")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(FAIL_POLICY, "owned Docker inspect shape differs")
    document = value[0]
    config = document.get("Config")
    container_id = document.get("Id")
    labels = config.get("Labels") if type(config) is dict else None
    expected_labels = dict(principal.get("expected_container_labels", []))
    if (
        type(container_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or document.get("Name") != f"/{principal.get('container_name')}"
        or type(config) is not dict
        or config.get("Image") != principal.get("image")
        or config.get("Cmd") != principal.get("container_argv")
        or labels != expected_labels
    ):
        _fail(FAIL_POLICY, "Docker inspect ownership principal differs")
    body: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-docker-owned-inspect/1",
        "docker_execution_authority_manifest_sha256": principal[
            "authority_manifest_sha256"
        ],
        "slot_id": principal["slot_id"],
        "slot": principal["slot"],
        "container_id": container_id,
        "container_name": principal["container_name"],
        "ownership_label_root": principal["ownership_label_root"],
        "image": principal["image"],
        "command_sha256": principal["command_sha256"],
        "inspect_hex": payload.hex(),
        "inspect_sha256": sha256(payload).hexdigest(),
    }
    result = dict(body)
    result["ownership_inspect_root"] = sha256(
        DOCKER_OWNED_INSPECT_ROOT_DOMAIN + _canonical_json_bytes(body)
    ).hexdigest()
    return result


def _read_owned_container_by_name_v1(
    principal: Mapping[str, object],
    runner: Callable[..., object],
) -> tuple[str, dict[str, object]]:
    """Read-only unique-name discovery; foreign or ambiguous state fails closed."""

    name = principal.get("container_name")
    if type(name) is not str:
        _fail(FAIL_POLICY, "Docker name discovery principal differs")
    result = _run_docker_control_v1(
        [DOCKER_EXECUTABLE, f"--host={DOCKER_HOST}", "inspect", name],
        runner,
    )
    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    if returncode == 0 and type(stdout) is bytes and stdout and stderr == b"":
        evidence = _validate_owned_docker_inspect_payload_v1(stdout, principal)
        return evidence["container_id"], evidence
    if returncode == 1 and type(stdout) is bytes and type(stderr) is bytes:
        authoritative_not_found = {
            (
                b"",
                f"Error: No such object: {name}\n".encode("ascii"),
            ),
            (
                b"",
                f"Error: No such container: {name}\n".encode("ascii"),
            ),
            (
                b"",
                (
                    "Error response from daemon: No such container: "
                    f"{name}\n"
                ).encode("ascii"),
            ),
            (
                b"[]\n",
                f"error: no such object: {name}\n".encode("ascii"),
            ),
        }
        if (stdout, stderr) in authoritative_not_found:
            return "", {
                "schema_version": (
                    "hegel-phase3a-q05b-docker-authoritative-absence/1"
                ),
                "container_identity": name,
                "inspect_exit_code": 1,
                "inspect_stdout_hex": stdout.hex(),
                "inspect_stdout_sha256": sha256(stdout).hexdigest(),
                "inspect_stderr_hex": stderr.hex(),
                "inspect_stderr_sha256": sha256(stderr).hexdigest(),
            }
    _fail(FAIL_POLICY, "Docker unique-name discovery was not authoritative")


def _write_abort_release_v1(control_root: Path) -> None:
    """Best-effort nonqualifying release used only on a failed lifecycle."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(control_root, flags)
    descriptor: int | None = None
    try:
        if "release" in os.listdir(directory):
            return
        descriptor = os.open(
            "release",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=directory,
        )
        payload = b"ABORT_NO_QUALIFICATION\n"
        if os.write(descriptor, payload) != len(payload):
            raise OSError("short abort-release write")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.fsync(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _docker_remove_and_quiet_absence_v1(
    container_id: str | None,
    principal: Mapping[str, object],
    runner: Callable[..., object],
    errors: list[str],
    *,
    quiet_poll_count: int = 4,
    maximum_poll_count: int = 20,
) -> None:
    """Discover by name read-only, but destructively target only an owned CID."""

    if (
        (
            container_id is not None
            and (
                type(container_id) is not str
                or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
            )
        )
        or type(principal) is not dict
        or type(principal.get("container_name")) is not str
        or type(quiet_poll_count) is not int
        or quiet_poll_count < 2
        or type(maximum_poll_count) is not int
        or maximum_poll_count < quiet_poll_count + 1
    ):
        errors.append("docker cleanup principal/quiet-window registry differs")
        return

    def inspect_owned_id(target: str) -> bool:
        completed = _run_docker_control_v1(
            [DOCKER_EXECUTABLE, f"--host={DOCKER_HOST}", "inspect", target],
            runner,
        )
        returncode = getattr(completed, "returncode", None)
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
        if returncode == 0 and type(stdout) is bytes and stdout and stderr == b"":
            evidence = _validate_owned_docker_inspect_payload_v1(stdout, principal)
            if evidence["container_id"] != target:
                _fail(FAIL_POLICY, "Docker cleanup ID inspect differs")
            return True
        if returncode == 1 and type(stdout) is bytes and type(stderr) is bytes:
            authoritative = {
                (b"", f"Error: No such object: {target}\n".encode("ascii")),
                (b"", f"Error: No such container: {target}\n".encode("ascii")),
                (
                    b"",
                    (
                        "Error response from daemon: No such container: "
                        f"{target}\n"
                    ).encode("ascii"),
                ),
                (b"[]\n", f"error: no such object: {target}\n".encode("ascii")),
            }
            if (stdout, stderr) in authoritative:
                return False
        _fail(FAIL_POLICY, "Docker cleanup ID inspect was not authoritative")

    def remove_owned_id(target: str) -> None:
        if not inspect_owned_id(target):
            return
        removal = _run_docker_control_v1(
            [DOCKER_EXECUTABLE, f"--host={DOCKER_HOST}", "rm", "-f", target],
            runner,
        )
        if getattr(removal, "returncode", None) != 0:
            _fail(FAIL_POLICY, "Docker owned-ID cleanup removal failed")
        _docker_absence_evidence_v1(target, runner)

    try:
        if container_id is not None:
            # Do not inspect the name after CID deletion: a same-name ABA
            # replacement is outside this attempt and must never be touched.
            remove_owned_id(container_id)
            return
        quiet = 0
        for _poll in range(maximum_poll_count):
            time.sleep(0.05)
            discovered_id, _evidence = _read_owned_container_by_name_v1(
                principal,
                runner,
            )
            if discovered_id:
                remove_owned_id(discovered_id)
                return
            quiet += 1
        # Docker exposes no frozen upper bound from CLI fork/exec to daemon
        # create.  Repeated name absence after Popen began therefore cannot
        # prove that a late container will not appear.  Only discovery of an
        # exact owned principal followed by CID-only removal closes this path.
        errors.append(
            "docker-owned-container-id-unresolved:potential-late-create"
        )
    except BaseException as error:
        errors.append(
            "docker-owned-id-cleanup:"
            f"{type(error).__name__}:{error}"
        )


def _cleanup_unbound_post_popen_body_v1(
    process: subprocess.Popen[bytes],
    principal: Mapping[str, object],
    runner: Callable[..., object],
) -> tuple[str, ...]:
    """Close the Popen-success/setup-failure gap without trusting a cidfile."""

    errors: list[str] = []
    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=2.0)
    except BaseException:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except BaseException as error:
            errors.append(f"docker-cli:{type(error).__name__}:{error}")
    _docker_remove_and_quiet_absence_v1(
        None,
        principal,
        runner,
        errors,
    )
    return tuple(errors)


def _cleanup_unbound_post_popen_v1(
    process: subprocess.Popen[bytes],
    principal: Mapping[str, object],
    runner: Callable[..., object],
) -> tuple[str, ...]:
    with _docker_ownership_signal_guard_v1():
        return _cleanup_unbound_post_popen_body_v1(
            process,
            principal,
            runner,
        )


def _raise_after_unbound_post_popen_cleanup_v1(
    process: subprocess.Popen[bytes],
    principal: Mapping[str, object],
    runner: Callable[..., object],
    original: BaseException,
    context: str,
) -> NoReturn:
    errors = _cleanup_unbound_post_popen_v1(process, principal, runner)
    if errors:
        _fail(
            FAIL_POLICY,
            f"{context}; cleanup closure failed: {'; '.join(errors)}; "
            f"original={type(original).__name__}:{original}",
        )
    raise original


def _raise_after_popen_call_failure_cleanup_v1(
    principal: Mapping[str, object],
    runner: Callable[..., object],
    original: BaseException,
    context: str,
) -> NoReturn:
    """Fail closed when Popen may have created a daemon object before raising."""

    errors: list[str] = []
    with _docker_ownership_signal_guard_v1():
        _docker_remove_and_quiet_absence_v1(
            None,
            principal,
            runner,
            errors,
        )
    if errors:
        _fail(
            FAIL_POLICY,
            f"{context}; cleanup closure failed: {'; '.join(errors)}; "
            f"original={type(original).__name__}:{original}",
        )
    raise original


def _abort_held_actor_cleanup_body_v1(
    actor: HeldActorProcessV1,
    runner: Callable[..., object],
) -> tuple[str, ...]:
    """Stop sampling, force-remove, and return every cleanup defect."""

    # These fields are observable diagnostics, never cleanup authority.  Both
    # actor_starter and actor_group_closer are injected callbacks and can
    # mutate every HeldActorProcessV1 field.  Cleanup therefore replays the
    # process and ownership principal on every invocation; owned-ID removal is
    # idempotent and a missing/foreign identity remains fail closed.
    actor.failure_cleanup_attempted = True
    actor.failure_cleanup_complete = False

    try:
        actor.sample_stop.set()
    except BaseException as error:
        actor.cleanup_errors.append(
            f"resource-sampler-stop:{type(error).__name__}:{error}"
        )
    sample_thread = actor.sample_thread
    if sample_thread is not None:
        try:
            # Thread.start() can fail after the object has been assigned to
            # actor.sample_thread.  Joining such an unstarted Thread raises
            # RuntimeError and must never prevent CLI stop/wait or owned-CID
            # cleanup below.
            if sample_thread.ident is not None:
                sample_thread.join(timeout=2.0)
                if sample_thread.is_alive():
                    actor.cleanup_errors.append("resource sampler did not stop")
        except BaseException as error:
            actor.cleanup_errors.append(
                f"resource-sampler-join:{type(error).__name__}:{error}"
            )
    try:
        if actor.control_root.is_dir() and os.path.lexists(actor.control_root / "done"):
            _write_abort_release_v1(actor.control_root)
    except BaseException as error:
        actor.cleanup_errors.append(f"abort-release:{type(error).__name__}:{error}")
    try:
        if actor.process.poll() is None:
            actor.process.terminate()
        actor.process.wait(timeout=2.0)
    except BaseException:
        try:
            actor.process.kill()
            actor.process.wait(timeout=2.0)
        except BaseException as error:
            actor.cleanup_errors.append(f"docker-cli:{type(error).__name__}:{error}")
    for label, thread in (
        ("stdout", actor.stdout_thread),
        ("stderr", actor.stderr_thread),
    ):
        try:
            if thread.ident is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    actor.cleanup_errors.append(
                        f"{label} pipe drain did not stop during cleanup"
                    )
        except BaseException as error:
            actor.cleanup_errors.append(
                f"{label}-pipe-join:{type(error).__name__}:{error}"
            )
    # A failed readiness observer may have rejected a path replacement.  Never
    # reacquire that cidfile and never use a bare name as a destructive target.
    if type(actor.docker_execution_principal) is not dict:
        actor.cleanup_errors.append(
            "docker ownership principal absent; zero destructive cleanup attempted"
        )
    else:
        _docker_remove_and_quiet_absence_v1(
            actor.container_id,
            actor.docker_execution_principal,
            runner,
            actor.cleanup_errors,
        )
    actor.failure_cleanup_complete = True
    return tuple(actor.cleanup_errors)


def _abort_held_actor_cleanup_v1(
    actor: HeldActorProcessV1,
    runner: Callable[..., object],
) -> tuple[str, ...]:
    with _docker_ownership_signal_guard_v1():
        return _abort_held_actor_cleanup_body_v1(actor, runner)


def _cleanup_actor_set_v1(
    actors: Sequence[HeldActorProcessV1],
    runner: Callable[..., object],
) -> tuple[str, ...]:
    """Attempt cleanup for every actor and preserve actor-qualified defects."""

    errors: list[str] = []
    for actor in actors:
        if type(actor) is not HeldActorProcessV1:
            errors.append("UNKNOWN_ACTOR:cleanup target type differs")
            continue
        try:
            actor_errors = _abort_held_actor_cleanup_v1(actor, runner)
        except BaseException as error:
            errors.append(
                f"{actor.actor_id}:cleanup raised:{type(error).__name__}:{error}"
            )
            continue
        errors.extend(f"{actor.actor_id}:{row}" for row in actor_errors)
    return tuple(errors)


def _raise_after_actor_cleanup_v1(
    actors: Sequence[HeldActorProcessV1],
    runner: Callable[..., object],
    original: BaseException,
    context: str,
) -> NoReturn:
    """Never hide a residual container behind the original lifecycle error."""

    cleanup_errors = _cleanup_actor_set_v1(actors, runner)
    if cleanup_errors:
        detail = "; ".join(cleanup_errors)
        _fail(
            FAIL_POLICY,
            f"{context}; cleanup closure failed: {detail}; "
            f"original={type(original).__name__}:{original}",
        )
    raise original


def close_held_actor_process_v1(
    actor: HeldActorProcessV1,
    *,
    child_timeout_seconds: float,
    exit_timeout_seconds: float = 60.0,
    inspect_reader: Callable[[str], bytes] = _docker_inspect_payload_v1,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    command_runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Close done->fresh-final-sample->release->exit->inspect->rm exactly."""

    if (
        type(actor) is not HeldActorProcessV1
        or not callable(inspect_reader)
        or not callable(command_runner)
        or type(exit_timeout_seconds) not in (int, float)
        or isinstance(exit_timeout_seconds, bool)
        or exit_timeout_seconds <= 0
    ):
        _fail(FAIL_POLICY, "held actor close input differs")
    try:
        container_id = _wait_for_held_child_done_v1(actor, child_timeout_seconds)
        completion = seal_held_actor_completion_v1(actor.control_root, actor.actor_id)

        # Stop the independent cadence only after done is sealed.  Event.wait
        # wakes the sampler immediately, so the subsequent fresh collection is
        # still within the <=250 ms unsampled-gap contract.
        actor.sample_stop.set()
        if actor.sample_thread is None:
            _fail(FAIL_POLICY, "held actor sampler was not started")
        actor.sample_thread.join(timeout=10.0)
        if actor.sample_thread.is_alive() or not actor.sample_complete.is_set():
            _fail(FAIL_POLICY, "held actor sampler did not stop for final sample")
        if actor.sample_errors:
            _fail(FAIL_POLICY, f"held actor sampler failed: {actor.sample_errors[0]}")
        with actor.sample_lock:
            continuous_samples = tuple(dict(row) for row in actor.sample_rows)
        if not continuous_samples:
            _fail(FAIL_POLICY, "held actor has no live resource sample")

        inspect_before = inspect_reader(container_id)
        if type(actor.docker_execution_principal) is not dict:
            _fail(FAIL_POLICY, "held actor Docker ownership principal is absent")
        live_ownership = _validate_owned_docker_inspect_payload_v1(
            inspect_before,
            actor.docker_execution_principal,
        )
        held_final = held_final_resource_sample_v1(
            actor.control_root,
            completion,
            actor.role_id,
            container_id,
            actor.container_name,
            actor.mount_registry,
            inspect_before,
            lambda: inspect_reader(container_id),
            seccomp_evidence=actor.seccomp_evidence,
            docker_execution_principal=actor.docker_execution_principal,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
        )
        held_final["sample_ordinal"] = len(continuous_samples) + 1
        release = release_held_actor_v1(actor.control_root, completion, held_final)
        try:
            exit_code = actor.process.wait(timeout=exit_timeout_seconds)
        except subprocess.TimeoutExpired:
            _fail(FAIL_POLICY, "held actor did not exit after release")
        actor.stdout_thread.join(timeout=10.0)
        actor.stderr_thread.join(timeout=10.0)
        if actor.stdout_thread.is_alive() or actor.stderr_thread.is_alive():
            _fail(FAIL_POLICY, "held actor pipe drain did not close")
        for name, state in (
            ("stdout", actor.stdout_drain),
            ("stderr", actor.stderr_drain),
        ):
            if state.errors:
                _fail(FAIL_POLICY, f"held actor {name} drain failed: {state.errors[0]}")
            if state.overflow or len(state.payload) != state.total:
                _fail(FAIL_POLICY, f"held actor {name} exceeded bounded drain")
        stdout = bytes(actor.stdout_drain.payload)
        stderr = bytes(actor.stderr_drain.payload)
        if stderr:
            _fail(FAIL_POLICY, "held actor emitted stderr")
        validate_held_actor_exit_v1(
            actor.control_root,
            release,
            stdout,
            exit_code,
        )
        post_exit_inspect = inspect_reader(container_id)
        post_ownership = _validate_owned_docker_inspect_payload_v1(
            post_exit_inspect,
            actor.docker_execution_principal,
        )
        final_resource = final_resource_transcript_v1(
            continuous_samples + (held_final,),
            post_exit_inspect,
            command_security_options=actor.mount_registry.security_options,
            seccomp_evidence=actor.seccomp_evidence,
        )
        explicit_remove_command = docker_explicit_remove_command_v1(container_id)
        removal = _run_docker_control_v1(
            explicit_remove_command,
            command_runner,
        )
        if type(getattr(removal, "returncode", None)) is not int or removal.returncode != 0:
            _fail(FAIL_POLICY, "explicit Docker removal failed")
        absence = _docker_absence_evidence_v1(container_id, command_runner)
        if type(actor.seccomp_evidence) is not dict:
            _fail(FAIL_POLICY, "held actor seccomp evidence is absent")
        seccomp_path = actor.seccomp_evidence.get("absolute_path")
        if (
            type(seccomp_path) is not str
            or sealed_policy_file_evidence_v1(
                Path(seccomp_path),
                RUNTIME_SECCOMP_RELATIVE_PATH,
            )
            != actor.seccomp_evidence
        ):
            _fail(FAIL_POLICY, "held actor seccomp identity changed")
        return {
            "schema_version": "hegel-phase3a-q05b-held-actor-complete-evidence/1",
            "actor_id": actor.actor_id,
            "container_id": container_id,
            "docker_execution_authority_manifest_sha256": (
                actor.docker_execution_authority_manifest_sha256
            ),
            "docker_execution_slot_row": actor.docker_execution_slot_row,
            "ownership_label_root": actor.ownership_label_root,
            "precreate_absence_evidence": actor.precreate_absence_evidence,
            "command_sha256": actor.mount_registry.command_sha256,
            "mount_registry_sha256": actor.mount_registry.registry_sha256,
            "cidfile_evidence": actor.cidfile_evidence,
            "seccomp_evidence": actor.seccomp_evidence,
            "control_root_path": completion["control_root_path"],
            "control_root_nlink": completion["root_nlink"],
            "completion_evidence": completion,
            "continuous_sample_count": len(continuous_samples),
            "held_final_resource": held_final,
            "live_ownership_inspect_evidence": live_ownership,
            "release_evidence": release,
            "post_exit_inspect_hex": post_exit_inspect.hex(),
            "post_exit_inspect_sha256": sha256(post_exit_inspect).hexdigest(),
            "post_ownership_inspect_evidence": post_ownership,
            "final_resource_transcript": final_resource,
            "stdout_hex": stdout.hex(),
            "stdout_sha256": actor.stdout_drain.digest.hexdigest(),
            "stdout_length": actor.stdout_drain.total,
            "stderr_sha256": actor.stderr_drain.digest.hexdigest(),
            "stderr_length": actor.stderr_drain.total,
            "explicit_remove_exit_code": removal.returncode,
            "explicit_remove_command": explicit_remove_command,
            "cleanup_target_kind": "OWNERSHIP_VALIDATED_CONTAINER_ID",
            "container_name_was_never_a_destructive_target": True,
            "docker_absence_evidence": absence,
        }
    except BaseException as error:
        _raise_after_actor_cleanup_v1(
            (actor,),
            command_runner,
            error,
            f"held actor lifecycle failed for {actor.actor_id}",
        )


def strict_replay_docker_completion_ownership_v1(
    completion: Mapping[str, object],
    authority: Mapping[str, object],
    slot: str,
    command: Sequence[str],
) -> dict[str, object]:
    """Reject an injected closer result before any stdout or sidecar is trusted."""

    principal = _docker_execution_principal_v1(command, authority, slot)
    slot_row = _docker_execution_slot_row_v1(authority, slot)
    if type(completion) is not dict:
        _fail(FAIL_ACTUAL_ADMISSION, "Docker completion ownership shape differs")
    container_id = completion.get("container_id")
    cidfile = completion.get("cidfile_evidence")
    if (
        completion.get("actor_id") != slot
        or type(container_id) is not str
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or completion.get("docker_execution_authority_manifest_sha256")
        != authority.get("manifest_sha256")
        or completion.get("docker_execution_slot_row") != slot_row
        or completion.get("ownership_label_root")
        != principal["ownership_label_root"]
        or completion.get("command_sha256") != principal["command_sha256"]
        or type(cidfile) is not dict
        or cidfile.get("container_id") != container_id
        or cidfile.get("payload_hex") != container_id.encode("ascii").hex()
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Docker completion causal identity differs")
    try:
        precreate = _ACTUAL_ADMISSION.validate_docker_precreate_absence_v1(
            completion.get("precreate_absence_evidence"),
            authority,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    if precreate.get("slot") != slot:
        _fail(FAIL_ACTUAL_ADMISSION, "Docker completion precreate slot differs")

    owned_rows: dict[str, dict[str, object]] = {}
    for key in (
        "live_ownership_inspect_evidence",
        "post_ownership_inspect_evidence",
    ):
        observed = completion.get(key)
        if type(observed) is not dict or type(observed.get("inspect_hex")) is not str:
            _fail(FAIL_ACTUAL_ADMISSION, "Docker completion owned inspect differs")
        try:
            replay = _validate_owned_docker_inspect_payload_v1(
                bytes.fromhex(observed["inspect_hex"]),
                principal,
            )
        except ValueError as error:
            _fail(FAIL_ACTUAL_ADMISSION, f"Docker owned inspect hex differs: {error}")
        if replay != observed or replay.get("container_id") != container_id:
            _fail(FAIL_ACTUAL_ADMISSION, "Docker completion owned inspect replay differs")
        owned_rows[key] = replay
    post = owned_rows["post_ownership_inspect_evidence"]
    if (
        completion.get("post_exit_inspect_hex") != post["inspect_hex"]
        or completion.get("post_exit_inspect_sha256") != post["inspect_sha256"]
        or completion.get("explicit_remove_exit_code") != 0
        or completion.get("explicit_remove_command")
        != docker_explicit_remove_command_v1(container_id)
        or completion.get("cleanup_target_kind")
        != "OWNERSHIP_VALIDATED_CONTAINER_ID"
        or completion.get("container_name_was_never_a_destructive_target") is not True
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Docker completion removal closure differs")
    try:
        absence = _ACTUAL_ADMISSION.validate_docker_authoritative_absence_v1(
            completion.get("docker_absence_evidence"),
            container_id,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    return {
        "slot": slot,
        "container_id": container_id,
        "precreate_absence_root": precreate["precreate_absence_root"],
        "live_ownership_inspect_root": owned_rows[
            "live_ownership_inspect_evidence"
        ]["ownership_inspect_root"],
        "post_ownership_inspect_root": post["ownership_inspect_root"],
        "post_remove_absence_sha256": sha256(
            _canonical_json_bytes(absence)
        ).hexdigest(),
    }


def dry_run_plan_v1(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    config = load_isolation_config_v1(project_root)
    source = git_source_status_v1(project_root)
    placeholder = Path("/sealed")
    placeholder_nonce = b"\x00" * 32
    try:
        placeholder_slot_rows = {
            row["slot"]: row
            for row in _ACTUAL_ADMISSION.docker_execution_slot_rows_v1(
                source.head,
                placeholder_nonce,
            )
        }
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    commands = {
        "rust_offline_test": rust_build_commands_v1(
            placeholder / "rust-source",
            placeholder / "cargo-home",
            placeholder / "target-output",
            build_seccomp=(
                placeholder / "host-source" / BUILD_SECCOMP_RELATIVE_PATH
            ),
            docker_slot_row=placeholder_slot_rows["RUST_TEST"],
        )[0],
        "rust_offline_release_build": rust_build_commands_v1(
            placeholder / "rust-source",
            placeholder / "cargo-home",
            placeholder / "target-output",
            build_seccomp=(
                placeholder / "host-source" / BUILD_SECCOMP_RELATIVE_PATH
            ),
            docker_slot_row=placeholder_slot_rows["RUST_RELEASE"],
        )[1],
        "python_endpoint": python_endpoint_command_v1(
            placeholder / "python-source",
            placeholder / "python-output",
            placeholder / "python-control",
            placeholder / "host-source" / RUNTIME_SECCOMP_RELATIVE_PATH,
            docker_slot_row=placeholder_slot_rows["PYTHON_ENDPOINT"],
        ),
        "rust_endpoint": rust_runtime_command_v1(
            placeholder / "prebuilt-runtime",
            placeholder / "rust-output",
            placeholder / "rust-control",
            placeholder / "host-source" / RUNTIME_SECCOMP_RELATIVE_PATH,
            docker_slot_row=placeholder_slot_rows["RUST_ENDPOINT"],
        ),
        "trusted_host_after_endpoints": trusted_host_command_v1(
            placeholder / "host-source",
            placeholder / "python-output",
            placeholder / "rust-output",
            placeholder / "stdout" / "python.stdout",
            placeholder / "stdout" / "rust.stdout",
            placeholder / "stdout" / "manifest.json",
            placeholder / "host-control",
            placeholder / "host-staging",
            placeholder / "host-source" / RUNTIME_SECCOMP_RELATIVE_PATH,
            docker_slot_row=placeholder_slot_rows["TRUSTED_HOST_REPLAY"],
        ),
    }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": STATUS_DRY_RUN,
        "execution": "DRY_RUN",
        "profile_id": config["profile_id"],
        "config_sha256": _sha256_file(
            project_root / CONFIG_RELATIVE_PATH,
            "dual isolation config",
        ),
        "source": {
            "head": source.head,
            "clean": source.clean,
            "porcelain_line_count": source.porcelain_line_count,
            "actual_requires_clean_full40_requested_head": True,
            "git_blob_materialization": [
                "git ls-tree COMMIT -- PATH",
                "git cat-file blob COMMIT:PATH",
            ],
            "worktree_copy_forbidden": True,
        },
        "docker_execution_placeholder": {
            "status": "NON_AUTHORITATIVE_NOT_EXECUTED",
            "placeholder_nonce_sha256": sha256(placeholder_nonce).hexdigest(),
            "source_commit": source.head,
            "ordered_slot_rows": [
                placeholder_slot_rows[slot]
                for _slot_id, slot in DOCKER_EXECUTION_SLOT_REGISTRY
            ],
            "initial_name_absence_collected": False,
            "precreate_name_absence_collected": False,
        },
        "images": config["images"],
        "resource_roles": config["resource_roles"],
        "stdout_capture_policy": config["stdout_capture_policy"],
        "live_resource_evidence_policy": config["live_resource_evidence_policy"],
        "source_allowlist_policy": config["source_allowlist_policy"],
        "commands": commands,
        "execution_protocol": config["execution_protocol"],
        "artifact_layout": config["artifact_layout"],
        "actual_blockers": [],
        "actual_implementation_blockers": [],
        "pending_actual_evidence_predicate_ids": list(
            PENDING_ACTUAL_EVIDENCE_PREDICATE_IDS
        ),
        "parallel_endpoint_stage": ["PYTHON_ENDPOINT", "RUST_ENDPOINT"],
        "trusted_host_starts_only_after_both_endpoint_exit": True,
        "host_replay_scope": {
            "five_sidecars_strict_replay": False,
            "materialized_counting_external_sort_ledger_replay": False,
            "python_rust_host_neutral_bytes_equal": False,
            "shadow_assembler_predicate12_closed": False,
            "pending_actual_execution": True,
        },
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "qualification_candidate_receipt": None,
        "qualification_final_receipt": None,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_gate_total": 20,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "formal_fixed_point_claimed": False,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
        "receipt_created": False,
        "artifact_path": None,
        "artifact_written": False,
        "actual_entrypoint_implemented": True,
        "actual_execution_status": "NOT_EXECUTED_AT_COMMIT_A",
        "execution_admission_policy": (
            "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION"
        ),
        "actual_admitted": False,
    }


@dataclass
class AnchoredPublishedArtifactV1:
    """Held ownership proof for one atomically published final inode."""

    parent_descriptor: int
    final_descriptor: int
    parent_device: int
    parent_inode: int
    file_device: int
    file_inode: int
    file_mode: int
    file_nlink: int
    payload_length: int
    payload_sha256: str
    closed: bool = False


ACTUAL_FINAL_DELIVERY_IDENTITY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-final-delivery-identity/2"
)
ACTUAL_FINAL_DELIVERY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/FINAL_DELIVERY/V1\x00"
)


def atomic_publish_canonical_artifact_v1(
    path: Path,
    value: object,
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> AnchoredPublishedArtifactV1:
    """Publish canonical JSON atomically without replacing an existing path.

    This primitive is provided for later actual-mode review.  Dry-run never
    calls it.
    """

    payload = _canonical_json_bytes(value)
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or (fault_hook is not None and not callable(fault_hook))
    ):
        _fail(FAIL_ARTIFACT, "artifact path must be absolute")
    parent = path.parent
    if path.name in ("", ".", "..") or "/" in path.name:
        _fail(FAIL_ARTIFACT, "artifact basename differs")
    try:
        parent_status = parent.lstat()
    except OSError as error:
        _fail(FAIL_ARTIFACT, f"artifact parent is unavailable: {error}")
    if parent.is_symlink() or not stat.S_ISDIR(parent_status.st_mode):
        _fail(FAIL_ARTIFACT, "artifact parent must be a nonsymlink directory")
    temporary_name = f".{path.name}.q05b-tmp-{os.getpid()}"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    final_descriptor: int | None = None
    directory: int | None = None
    linked = False
    publication_complete = False
    temporary_present = False
    temporary_identity: tuple[int, int] | None = None
    linked_final_identity: tuple[int, int] | None = None
    published_handle: AnchoredPublishedArtifactV1 | None = None
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory = os.open(parent, directory_flags)
        anchored_parent = os.fstat(directory)
        if (
            anchored_parent.st_dev != parent_status.st_dev
            or anchored_parent.st_ino != parent_status.st_ino
        ):
            _fail(FAIL_ARTIFACT, "artifact parent changed before anchored open")
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory)
        temporary_present = True
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        temporary_status = os.fstat(descriptor)
        temporary_identity = (temporary_status.st_dev, temporary_status.st_ino)
        if (
            not stat.S_ISREG(temporary_status.st_mode)
            or temporary_status.st_nlink != 1
            or stat.S_IMODE(temporary_status.st_mode) != 0o444
            or temporary_status.st_size != len(payload)
        ):
            _fail(FAIL_ARTIFACT, "temporary artifact identity differs")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            _fail(FAIL_ARTIFACT, "temporary artifact bytes differ")
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        linked = True
        final_descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        final_status = os.fstat(final_descriptor)
        final_entry = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        linked_final_identity = (final_status.st_dev, final_status.st_ino)
        if (
            not stat.S_ISREG(final_status.st_mode)
            or stat.S_IMODE(final_status.st_mode) != 0o444
            or final_status.st_size != len(payload)
            or final_status.st_nlink != 2
            or linked_final_identity != temporary_identity
            or (
                final_entry.st_dev,
                final_entry.st_ino,
                final_entry.st_mode,
                final_entry.st_nlink,
                final_entry.st_size,
            )
            != (
                final_status.st_dev,
                final_status.st_ino,
                final_status.st_mode,
                final_status.st_nlink,
                final_status.st_size,
            )
        ):
            _fail(FAIL_ARTIFACT, "published artifact identity differs")
        if os.read(final_descriptor, len(payload) + 1) != payload:
            _fail(FAIL_ARTIFACT, "published artifact bytes differ")
        if fault_hook is not None:
            fault_hook("AFTER_FINAL_LINK_VALIDATED_BEFORE_TEMP_UNLINK")
        os.fsync(directory)
        os.unlink(temporary_name, dir_fd=directory)
        temporary_present = False
        os.fsync(directory)
        temporary_after_unlink = os.fstat(descriptor)
        final_after_unlink = os.fstat(final_descriptor)
        final_entry_after_unlink = os.stat(
            path.name,
            dir_fd=directory,
            follow_symlinks=False,
        )
        if (
            (temporary_after_unlink.st_dev, temporary_after_unlink.st_ino)
            != temporary_identity
            or temporary_after_unlink.st_nlink != 1
            or (final_after_unlink.st_dev, final_after_unlink.st_ino)
            != temporary_identity
            or final_after_unlink.st_nlink != 1
            or (final_entry_after_unlink.st_dev, final_entry_after_unlink.st_ino)
            != temporary_identity
            or final_entry_after_unlink.st_nlink != 1
            or stat.S_IMODE(final_after_unlink.st_mode) != 0o444
            or final_after_unlink.st_size != len(payload)
        ):
            _fail(FAIL_ARTIFACT, "published artifact unlink closure differs")
        os.lseek(final_descriptor, 0, os.SEEK_SET)
        if os.read(final_descriptor, len(payload) + 1) != payload:
            _fail(FAIL_ARTIFACT, "published artifact bytes changed after unlink")
        parent_after = parent.lstat()
        if (
            parent_after.st_dev != anchored_parent.st_dev
            or parent_after.st_ino != anchored_parent.st_ino
        ):
            _fail(FAIL_ARTIFACT, "artifact parent path changed during publication")
        published_handle = AnchoredPublishedArtifactV1(
            parent_descriptor=directory,
            final_descriptor=final_descriptor,
            parent_device=anchored_parent.st_dev,
            parent_inode=anchored_parent.st_ino,
            file_device=final_after_unlink.st_dev,
            file_inode=final_after_unlink.st_ino,
            file_mode=stat.S_IMODE(final_after_unlink.st_mode),
            file_nlink=final_after_unlink.st_nlink,
            payload_length=len(payload),
            payload_sha256=sha256(payload).hexdigest(),
        )
        publication_complete = True
    except FileExistsError:
        _fail(FAIL_ARTIFACT, "artifact or temporary path already exists")
    except OSError as error:
        _fail(FAIL_ARTIFACT, f"atomic artifact publication failed: {error}")
    finally:
        rollback_errors: list[str] = []
        if directory is not None and linked and not publication_complete:
            try:
                current = os.stat(
                    path.name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
                rollback_identity = linked_final_identity or temporary_identity
                if rollback_identity is None or (
                    current.st_dev,
                    current.st_ino,
                ) != rollback_identity:
                    raise OSError("published final inode changed before rollback")
                os.unlink(path.name, dir_fd=directory)
                linked = False
                os.fsync(directory)
            except OSError as cleanup_error:
                rollback_errors.append(
                    f"failed publication could not be rolled back: {cleanup_error}"
                )
        if directory is not None and temporary_present:
            try:
                temporary_entry = os.stat(
                    temporary_name,
                    dir_fd=directory,
                    follow_symlinks=False,
                )
                allowed_identities = {
                    identity
                    for identity in (temporary_identity, linked_final_identity)
                    if identity is not None
                }
                if (
                    temporary_entry.st_dev,
                    temporary_entry.st_ino,
                ) not in allowed_identities:
                    raise OSError("temporary artifact inode changed before rollback")
                os.unlink(temporary_name, dir_fd=directory)
                os.fsync(directory)
            except OSError as cleanup_error:
                rollback_errors.append(
                    f"failed temporary artifact could not be rolled back: {cleanup_error}"
                )
        if final_descriptor is not None and not publication_complete:
            os.close(final_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None and not publication_complete:
            os.close(directory)
        if rollback_errors:
            _fail(FAIL_ARTIFACT, "; ".join(rollback_errors))
    if published_handle is None:
        _fail(FAIL_ARTIFACT, "atomic publication produced no ownership handle")
    return published_handle


def read_anchored_published_artifact_v1(
    handle: AnchoredPublishedArtifactV1,
    path: Path,
    expected_payload: bytes,
) -> bytes:
    """Replay final bytes only through the publisher-owned held descriptor."""

    if (
        type(handle) is not AnchoredPublishedArtifactV1
        or handle.closed
        or not isinstance(path, Path)
        or not path.is_absolute()
        or path.name in ("", ".", "..")
        or type(expected_payload) is not bytes
        or not expected_payload.endswith(b"\n")
        or type(handle.parent_descriptor) is not int
        or type(handle.final_descriptor) is not int
        or handle.parent_descriptor < 0
        or handle.final_descriptor < 0
        or any(
            type(getattr(handle, name)) is not int
            for name in (
                "parent_device",
                "parent_inode",
                "file_device",
                "file_inode",
                "file_mode",
                "file_nlink",
                "payload_length",
            )
        )
        or type(handle.payload_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", handle.payload_sha256) is None
    ):
        _fail(FAIL_ARTIFACT, "published artifact ownership handle differs")
    try:
        parent_before = os.fstat(handle.parent_descriptor)
        final_before = os.fstat(handle.final_descriptor)
        parent_path = path.parent.lstat()
        final_entry = os.stat(
            path.name,
            dir_fd=handle.parent_descriptor,
            follow_symlinks=False,
        )
        parent_access = (
            fcntl.fcntl(handle.parent_descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        )
        final_access = (
            fcntl.fcntl(handle.final_descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        )
        os.lseek(handle.final_descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(handle.final_descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > len(expected_payload):
                _fail(FAIL_ARTIFACT, "owned artifact exceeds expected length")
            blocks.append(block)
        payload = b"".join(blocks)
        final_after = os.fstat(handle.final_descriptor)
        parent_after = os.fstat(handle.parent_descriptor)
        final_entry_after = os.stat(
            path.name,
            dir_fd=handle.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        _fail(FAIL_ARTIFACT, f"published artifact ownership replay failed: {error}")
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or parent_access != os.O_RDONLY
        or (parent_before.st_dev, parent_before.st_ino)
        != (handle.parent_device, handle.parent_inode)
        or (parent_path.st_dev, parent_path.st_ino)
        != (handle.parent_device, handle.parent_inode)
        or not stat.S_ISREG(final_before.st_mode)
        or final_access != os.O_RDONLY
        or (final_before.st_dev, final_before.st_ino)
        != (handle.file_device, handle.file_inode)
        or stat.S_IMODE(final_before.st_mode) != handle.file_mode
        or final_before.st_nlink != handle.file_nlink
        or final_before.st_size != handle.payload_length
        or handle.file_mode != 0o444
        or handle.file_nlink != 1
        or handle.payload_length != len(expected_payload)
        or handle.payload_sha256 != sha256(expected_payload).hexdigest()
        or payload != expected_payload
        or (
            final_entry.st_dev,
            final_entry.st_ino,
            final_entry.st_mode,
            final_entry.st_nlink,
            final_entry.st_size,
        )
        != (
            final_before.st_dev,
            final_before.st_ino,
            final_before.st_mode,
            final_before.st_nlink,
            final_before.st_size,
        )
        or (
            final_before.st_dev,
            final_before.st_ino,
            final_before.st_mode,
            final_before.st_nlink,
            final_before.st_size,
            final_before.st_mtime_ns,
            final_before.st_ctime_ns,
        )
        != (
            final_after.st_dev,
            final_after.st_ino,
            final_after.st_mode,
            final_after.st_nlink,
            final_after.st_size,
            final_after.st_mtime_ns,
            final_after.st_ctime_ns,
        )
        or (final_entry_after.st_dev, final_entry_after.st_ino)
        != (handle.file_device, handle.file_inode)
        or (parent_after.st_dev, parent_after.st_ino)
        != (handle.parent_device, handle.parent_inode)
    ):
        _fail(FAIL_ARTIFACT, "published artifact ownership authority differs")
    return payload


def rollback_anchored_published_artifact_v1(
    handle: AnchoredPublishedArtifactV1,
    path: Path,
    expected_payload: bytes,
) -> None:
    """Unlink only the exact final inode owned by the publisher handle."""

    read_anchored_published_artifact_v1(handle, path, expected_payload)
    try:
        os.unlink(path.name, dir_fd=handle.parent_descriptor)
        os.fsync(handle.parent_descriptor)
        unlinked = os.fstat(handle.final_descriptor)
        if (
            (unlinked.st_dev, unlinked.st_ino)
            != (handle.file_device, handle.file_inode)
            or unlinked.st_nlink != 0
        ):
            _fail(FAIL_ARTIFACT, "owned artifact rollback link count differs")
        try:
            os.stat(
                path.name,
                dir_fd=handle.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            _fail(FAIL_ARTIFACT, "owned artifact remains after rollback")
        parent_path = path.parent.lstat()
        if (parent_path.st_dev, parent_path.st_ino) != (
            handle.parent_device,
            handle.parent_inode,
        ):
            _fail(FAIL_ARTIFACT, "owned artifact rollback parent changed")
    except OSError as error:
        _fail(FAIL_ARTIFACT, f"owned artifact rollback failed: {error}")


def _artifact_path_residual_conservative_v1(path: Path) -> bool:
    """Only one explicit FileNotFound result proves pathname absence."""

    if not isinstance(path, Path) or not path.is_absolute():
        return True
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _owned_artifact_residual_conservative_v1(
    handle: AnchoredPublishedArtifactV1,
    path: Path,
) -> bool:
    """Return true unless both anchored and pathname views prove absence."""

    if (
        type(handle) is not AnchoredPublishedArtifactV1
        or not isinstance(path, Path)
        or not path.is_absolute()
    ):
        return True
    try:
        os.stat(
            path.name,
            dir_fd=handle.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError:
        return True
    else:
        return True
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def close_anchored_published_artifact_v1(
    handle: AnchoredPublishedArtifactV1,
) -> None:
    if type(handle) is not AnchoredPublishedArtifactV1:
        _fail(FAIL_ARTIFACT, "published artifact close handle differs")
    if handle.closed:
        return
    errors: list[str] = []
    for descriptor in (handle.final_descriptor, handle.parent_descriptor):
        try:
            os.close(descriptor)
        except OSError as error:
            errors.append(str(error))
    handle.closed = True
    if errors:
        _fail(FAIL_ARTIFACT, "published artifact handle close failed: " + "; ".join(errors))


def read_published_canonical_artifact_v1(
    path: Path,
    expected_payload: bytes,
) -> bytes:
    """Anchor and replay the published 0444/nlink1 artifact without follows."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.name in ("", ".", "..")
        or type(expected_payload) is not bytes
        or not expected_payload.endswith(b"\n")
        or len(expected_payload) > 1024 * 1024 * 1024
    ):
        _fail(FAIL_ARTIFACT, "published artifact replay input differs")
    parent_before = path.parent.lstat()
    if path.parent.is_symlink() or not stat.S_ISDIR(parent_before.st_mode):
        _fail(FAIL_ARTIFACT, "published artifact parent differs")
    directory = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        anchored_parent = os.fstat(directory)
        if (anchored_parent.st_dev, anchored_parent.st_ino) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ):
            _fail(FAIL_ARTIFACT, "published artifact parent changed")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
        before = os.fstat(descriptor)
        entry = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size != len(expected_payload)
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
            )
            != (
                entry.st_dev,
                entry.st_ino,
                entry.st_mode,
                entry.st_nlink,
                entry.st_size,
            )
        ):
            _fail(FAIL_ARTIFACT, "published artifact file identity differs")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > len(expected_payload):
                _fail(FAIL_ARTIFACT, "published artifact exceeds expected length")
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
        parent_after = os.fstat(directory)
        if (
            payload != expected_payload
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (parent_after.st_dev, parent_after.st_ino)
            != (anchored_parent.st_dev, anchored_parent.st_ino)
        ):
            _fail(FAIL_ARTIFACT, "published artifact changed during replay")
        return payload
    except OSError as error:
        _fail(FAIL_ARTIFACT, f"published artifact anchored replay failed: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


HOST_STAGED_SIDECAR_PATHS: Final = (
    "preimages/000-full-v16-leaf-manifest-v1.cbor",
    "preimages/001-odd-node3-partition-evidence-v1.cbor",
    "preimages/002-sink-node3-partition-evidence-v1.cbor",
    "neutral/q05b-node3-sidecar-manifest-v1.cbor",
    "neutral/q05b-node3-golden-manifest-v1.cbor",
)


def _load_host_replay_module_v1(snapshot_root: Path = PROJECT_ROOT):
    """Load only the host replay closure without executing package __init__."""

    package_name = "hegel_machine"
    package_root = snapshot_root / "src/hegel_machine"
    if not package_root.is_dir():
        _fail(FAIL_SOURCE, "trusted-host package root is unavailable")
    existing = sys.modules.get(package_name)
    if existing is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        package.__file__ = None
        package.__spec__ = None
        sys.modules[package_name] = package
    else:
        paths = getattr(existing, "__path__", None)
        if paths is None or str(package_root) not in tuple(paths):
            _fail(FAIL_SOURCE, "trusted-host package bootstrap identity differs")
    return importlib.import_module(
        "hegel_machine.phase3_q05b_host_replay_v1"
    )


def _load_actual_admission_module_v1(snapshot_root: Path = PROJECT_ROOT):
    """Load the pure admission contract without executing package __init__."""

    package_name = "hegel_machine"
    package_root = snapshot_root / "src/hegel_machine"
    if not package_root.is_dir():
        _fail(FAIL_SOURCE, "actual-admission package root is unavailable")
    existing = sys.modules.get(package_name)
    if existing is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        package.__file__ = None
        package.__spec__ = None
        sys.modules[package_name] = package
    else:
        paths = getattr(existing, "__path__", None)
        if paths is None or str(package_root) not in tuple(paths):
            _fail(FAIL_SOURCE, "actual-admission package bootstrap identity differs")
    return importlib.import_module(
        "hegel_machine.phase3_q05b_actual_admission_v1"
    )


def _read_sealed_regular_file_v1(
    path: Path,
    maximum: int,
    name: str,
    *,
    expected_mode: int = 0o444,
) -> bytes:
    if (
        not isinstance(path, Path)
        or type(maximum) is not int
        or maximum < 1
        or type(expected_mode) is not int
        or expected_mode not in (0o444, 0o555)
    ):
        _fail(FAIL_POLICY, f"{name} sealed read input differs")
    before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        anchored = os.fstat(descriptor)
        if (
            not stat.S_ISREG(anchored.st_mode)
            or anchored.st_nlink != 1
            or stat.S_IMODE(anchored.st_mode) != expected_mode
            or anchored.st_size < 1
            or anchored.st_size > maximum
            or (anchored.st_dev, anchored.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            _fail(FAIL_POLICY, f"{name} sealed file identity differs")
        blocks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not block:
                break
            blocks.append(block)
            size += len(block)
            if size > maximum:
                _fail(FAIL_POLICY, f"{name} exceeds maximum size")
        after = os.fstat(descriptor)
        if (
            anchored.st_dev,
            anchored.st_ino,
            anchored.st_mode,
            anchored.st_nlink,
            anchored.st_size,
            anchored.st_mtime_ns,
            anchored.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(FAIL_POLICY, f"{name} changed while reading")
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _sealed_policy_file_evidence_and_payload_v1(
    path: Path,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], bytes]:
    """Bind one policy file and retain bytes from the same anchored read.

    Runtime/build seccomp files are mounted from the independently materialized
    HOST snapshot.  The command string alone is not evidence of which inode or
    Commit-A blob Docker consumed, so every run carries this anchored identity.
    """

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(relative_path) is not str
        or not relative_path
        or relative_path.startswith("/")
        or ".." in PurePosixPath(relative_path).parts
        or (
            expected_sha256 is not None
            and (
                type(expected_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            )
        )
    ):
        _fail(FAIL_SOURCE, "sealed policy file input differs")
    before = path.lstat()
    payload = _read_sealed_regular_file_v1(
        path,
        4 * 1024 * 1024,
        f"sealed policy file {relative_path}",
    )
    after = path.lstat()
    digest = sha256(payload).hexdigest()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        _fail(FAIL_SOURCE, "sealed policy file identity differs")
    value: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-sealed-policy-file/1",
        "absolute_path": path.as_posix(),
        "snapshot_relative_path": relative_path,
        "file_device": before.st_dev,
        "file_inode": before.st_ino,
        "file_nlink": before.st_nlink,
        "file_uid": before.st_uid,
        "file_gid": before.st_gid,
        "file_mode": stat.S_IMODE(before.st_mode),
        "file_size": before.st_size,
        "file_mtime_ns": before.st_mtime_ns,
        "file_ctime_ns": before.st_ctime_ns,
        "payload_sha256": digest,
    }
    value["manifest_sha256"] = sha256(_canonical_json_bytes(value)).hexdigest()
    return value, payload


def sealed_policy_file_evidence_v1(
    path: Path,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Bind one policy file to its absolute sealed path and raw bytes."""

    evidence, _payload = _sealed_policy_file_evidence_and_payload_v1(
        path,
        relative_path,
        expected_sha256=expected_sha256,
    )
    return evidence


def _require_json_semantic_type_exact_v1(
    observed: object,
    expected: object,
    name: str,
) -> None:
    """Compare decoded JSON without Python's bool/int or int/float aliases."""

    if type(observed) is not type(expected):
        _fail(FAIL_POLICY, f"{name} type differs")
    if type(expected) is dict:
        assert type(observed) is dict
        if set(observed) != set(expected):
            _fail(FAIL_POLICY, f"{name} fields differ")
        for key in expected:
            _require_json_semantic_type_exact_v1(
                observed[key],
                expected[key],
                f"{name}.{key}",
            )
        return
    if type(expected) is list:
        assert type(observed) is list
        if len(observed) != len(expected):
            _fail(FAIL_POLICY, f"{name} length differs")
        for index, (item, expected_item) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _require_json_semantic_type_exact_v1(
                item,
                expected_item,
                f"{name}[{index}]",
            )
        return
    if observed != expected:
        _fail(FAIL_POLICY, f"{name} value differs")


def validate_docker_inspect_seccomp_semantics_v1(
    observed_security_options: object,
    command_security_options: Sequence[str],
    sealed_policy_evidence: Mapping[str, object],
    expected_relative_path: str,
) -> dict[str, object]:
    """Cross Docker 29's inline seccomp JSON to the sealed command policy.

    The command authority remains the literal absolute policy path.  Docker
    inspect is accepted only in its daemon-normalized two-item form, whose
    second item is inline strict JSON semantically and type-exactly equal to
    bytes read from the same anchored inode represented by the sealed evidence.
    """

    if (
        type(command_security_options) not in (tuple, list)
        or len(command_security_options) != 2
        or any(type(item) is not str for item in command_security_options)
        or command_security_options[0] != "no-new-privileges"
        or not command_security_options[1].startswith("seccomp=")
        or expected_relative_path
        not in (RUNTIME_SECCOMP_RELATIVE_PATH, BUILD_SECCOMP_RELATIVE_PATH)
        or type(sealed_policy_evidence) is not dict
    ):
        _fail(FAIL_POLICY, "Docker command seccomp registry differs")
    command_policy_text = command_security_options[1].removeprefix("seccomp=")
    command_policy_path = Path(command_policy_text)
    if (
        not command_policy_path.is_absolute()
        or command_policy_text.startswith("{")
        or command_policy_text != command_policy_path.as_posix()
        or sealed_policy_evidence.get("absolute_path") != command_policy_text
        or sealed_policy_evidence.get("snapshot_relative_path")
        != expected_relative_path
    ):
        _fail(FAIL_POLICY, "Docker command seccomp path binding differs")
    replay, sealed_payload = _sealed_policy_file_evidence_and_payload_v1(
        command_policy_path,
        expected_relative_path,
    )
    if replay != sealed_policy_evidence:
        _fail(FAIL_POLICY, "Docker seccomp sealed evidence changed")
    if (
        type(observed_security_options) is not list
        or len(observed_security_options) != 2
        or any(type(item) is not str for item in observed_security_options)
        or observed_security_options[0] != "no-new-privileges"
        or not observed_security_options[1].startswith("seccomp=")
    ):
        _fail(FAIL_POLICY, "Docker inspect SecurityOpt differs")
    inline_text = observed_security_options[1].removeprefix("seccomp=")
    try:
        inline_payload = inline_text.encode("ascii", "strict")
    except UnicodeError as error:
        _fail(FAIL_POLICY, f"Docker inspect seccomp JSON is not ASCII: {error}")
    stripped = inline_payload.strip()
    if not stripped.startswith(b"{") or not stripped.endswith(b"}"):
        _fail(FAIL_POLICY, "Docker inspect seccomp must be inline JSON")
    sealed_value = _strict_json_value_v1(
        sealed_payload,
        "sealed seccomp policy",
    )
    observed_value = _strict_json_value_v1(
        inline_payload,
        "Docker inspect inline seccomp policy",
    )
    if type(sealed_value) is not dict or type(observed_value) is not dict:
        _fail(FAIL_POLICY, "Docker seccomp policy must be one JSON object")
    _require_json_semantic_type_exact_v1(
        observed_value,
        sealed_value,
        "Docker inspect seccomp semantics",
    )
    return replay


SEALED_TREE_IDENTITY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-sealed-tree-identity/1"
)


def seal_directory_tree_read_only_v1(root: Path) -> None:
    """Seal an actor-produced regular-file tree after its strict replay."""

    if not isinstance(root, Path) or not root.is_absolute():
        _fail(FAIL_POLICY, "sealed tree root input differs")
    before = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(before.st_mode):
        _fail(FAIL_POLICY, "sealed tree root must be a nonsymlink directory")
    entries = tuple(root.rglob("*"))
    for path in entries:
        value = path.lstat()
        if path.is_symlink() or not (
            stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)
        ):
            _fail(FAIL_POLICY, "sealed tree contains a symlink or special file")
        if stat.S_ISREG(value.st_mode) and (
            value.st_nlink != 1 or stat.S_IMODE(value.st_mode) != 0o444
        ):
            _fail(FAIL_POLICY, "sealed tree file mode/link differs")
    for path in sorted(
        (path for path in entries if path.is_dir()),
        key=lambda item: (-len(item.parts), item.as_posix()),
    ):
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        anchored = os.fstat(descriptor)
        if (anchored.st_dev, anchored.st_ino) != (before.st_dev, before.st_ino):
            _fail(FAIL_POLICY, "sealed tree root changed before seal")
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sealed_tree_identity_v1(
    root: Path,
    expected_files: Sequence[str],
    *,
    maximum_file_bytes: int = 64 * 1024 * 1024,
    expected_file_modes: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Replay one exact sealed tree through one anchored root descriptor."""

    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or type(expected_files) not in (tuple, list)
        or tuple(expected_files) != tuple(sorted(set(expected_files)))
        or type(maximum_file_bytes) is not int
        or maximum_file_bytes < 1
        or (
            expected_file_modes is not None
            and (
                type(expected_file_modes) is not dict
                or set(expected_file_modes) != set(expected_files)
                or any(
                    type(mode) is not int or mode not in (0o444, 0o555)
                    for mode in expected_file_modes.values()
                )
            )
        )
    ):
        _fail(FAIL_POLICY, "sealed tree identity inputs differ")
    expected_directories = tuple(
        sorted(
            {
                PurePosixPath(*PurePosixPath(relative).parts[:depth]).as_posix()
                for relative in expected_files
                for depth in range(1, len(PurePosixPath(relative).parts))
            }
        )
    )
    try:
        root_before = root.lstat()
        resolved_before = root.resolve(strict=True)
    except OSError as error:
        _fail(FAIL_POLICY, f"sealed tree root is unavailable: {error}")
    if root != resolved_before or root.is_symlink():
        _fail(FAIL_POLICY, "sealed tree root path differs")
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )

    def stat_tuple(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def anchored_graph(
        directory: int,
        prefix: tuple[str, ...] = (),
    ) -> tuple[dict[str, tuple[int, ...]], tuple[str, ...], tuple[str, ...]]:
        snapshots: dict[str, tuple[int, ...]] = {}
        directories: list[str] = []
        files: list[str] = []
        for name in sorted(os.listdir(directory)):
            if type(name) is not str or name in ("", ".", "..") or "/" in name:
                _fail(FAIL_POLICY, "sealed tree entry name differs")
            entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
            relative_parts = prefix + (name,)
            relative = PurePosixPath(*relative_parts).as_posix()
            snapshots[relative] = stat_tuple(entry)
            if stat.S_ISDIR(entry.st_mode):
                if stat.S_IMODE(entry.st_mode) != 0o555 or entry.st_nlink < 2:
                    _fail(FAIL_POLICY, "sealed tree directory identity differs")
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory,
                )
                try:
                    anchored_child = os.fstat(child)
                    if stat_tuple(anchored_child) != stat_tuple(entry):
                        _fail(FAIL_POLICY, "sealed tree directory entry changed")
                    child_snapshots, child_directories, child_files = anchored_graph(
                        child, relative_parts
                    )
                finally:
                    os.close(child)
                directories.append(relative)
                directories.extend(child_directories)
                files.extend(child_files)
                snapshots.update(child_snapshots)
            elif stat.S_ISREG(entry.st_mode):
                files.append(relative)
            else:
                _fail(FAIL_POLICY, "sealed tree contains a symlink or special file")
        return snapshots, tuple(directories), tuple(files)

    try:
        anchored_root = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(anchored_root.st_mode)
            or stat.S_IMODE(anchored_root.st_mode) != 0o555
            or anchored_root.st_nlink < 2
            or (anchored_root.st_dev, anchored_root.st_ino)
            != (root_before.st_dev, root_before.st_ino)
        ):
            _fail(FAIL_POLICY, "sealed tree root identity differs")
        snapshots, observed_directories, observed_files = anchored_graph(descriptor)
        if (
            tuple(sorted(observed_files)) != tuple(expected_files)
            or tuple(sorted(observed_directories)) != expected_directories
        ):
            _fail(FAIL_POLICY, "sealed tree file/directory registry differs")
        directory_rows: list[list[object]] = []
        for relative in expected_directories:
            value = snapshots[relative]
            directory_rows.append(
                [
                    relative,
                    value[0],
                    value[1],
                    value[3],
                    value[4],
                    value[5],
                    stat.S_IMODE(value[2]),
                    value[7],
                    value[8],
                ]
            )
        file_rows: list[list[object]] = []
        for relative in expected_files:
            expected_mode = (
                0o444
                if expected_file_modes is None
                else expected_file_modes[relative]
            )
            payload, value = _read_snapshot_file_v1(descriptor, relative)
            if (
                len(payload) > maximum_file_bytes
                or stat_tuple(value) != snapshots[relative]
                or stat.S_IMODE(value.st_mode) != expected_mode
            ):
                _fail(FAIL_POLICY, "sealed tree file identity differs")
            file_rows.append(
                [
                    relative,
                    value.st_dev,
                    value.st_ino,
                    value.st_nlink,
                    value.st_uid,
                    value.st_gid,
                    stat.S_IMODE(value.st_mode),
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                    sha256(payload).hexdigest(),
                ]
            )
        after_snapshots, after_directories, after_files = anchored_graph(descriptor)
        anchored_after = os.fstat(descriptor)
        root_after = root.lstat()
        resolved_after = root.resolve(strict=True)
        if (
            snapshots != after_snapshots
            or observed_directories != after_directories
            or observed_files != after_files
            or stat_tuple(anchored_root) != stat_tuple(anchored_after)
            or (root_after.st_dev, root_after.st_ino)
            != (anchored_root.st_dev, anchored_root.st_ino)
            or root != resolved_after
        ):
            _fail(FAIL_POLICY, "sealed tree changed during identity replay")
    except OSError as error:
        _fail(FAIL_POLICY, f"sealed tree anchored replay failed: {error}")
    finally:
        os.close(descriptor)
    body: dict[str, object] = {
        "schema_version": SEALED_TREE_IDENTITY_SCHEMA_VERSION,
        "root_path": root.as_posix(),
        "root_device": anchored_root.st_dev,
        "root_inode": anchored_root.st_ino,
        "root_nlink": anchored_root.st_nlink,
        "root_mode": stat.S_IMODE(anchored_root.st_mode),
        "directory_rows": directory_rows,
        "file_rows": file_rows,
    }
    body["manifest_sha256"] = sha256(_canonical_json_bytes(body)).hexdigest()
    return body


def _write_exclusive_sealed_file_v1(
    directory: int,
    name: str,
    payload: bytes,
) -> os.stat_result:
    if (
        type(name) is not str
        or not name
        or "/" in name
        or name in (".", "..")
        or type(payload) is not bytes
        or not payload
    ):
        _fail(FAIL_ARTIFACT, "host staging file input differs")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
            _fail(FAIL_ARTIFACT, "host staging file identity differs")
        return result
    finally:
        os.close(descriptor)


def write_host_semantic_staging_v1(
    replay: object,
    staging_root: Path,
    host_module: object,
    negative_corpus_cbor: bytes,
    negative_corpus_root: bytes,
    negative_category_roots: tuple[tuple[int, bytes], ...],
) -> tuple[bytes, dict[str, object]]:
    """Seal five replayed sidecars plus one separate semantic witness."""

    root_before = staging_root.lstat()
    if (
        staging_root.is_symlink()
        or not stat.S_ISDIR(root_before.st_mode)
        or stat.S_IMODE(root_before.st_mode) != 0o700
        or tuple(staging_root.iterdir())
    ):
        _fail(FAIL_ARTIFACT, "host staging root must be an empty 0700 directory")
    try:
        witness = host_module.host_semantic_witness_bytes_v1(
            replay,
            negative_corpus_cbor,
            negative_corpus_root,
            negative_category_roots,
        )
        host_module.decode_host_semantic_witness_v1(
            witness,
            replay,
            negative_corpus_cbor,
            negative_corpus_root,
            negative_category_roots,
        )
    except AttributeError as error:
        _fail(FAIL_ARTIFACT, f"host witness module API differs: {error}")
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_descriptor = os.open(staging_root, root_flags)
    sidecars_descriptor: int | None = None
    directory_descriptors: dict[str, int] = {}
    try:
        anchored = os.fstat(root_descriptor)
        if (anchored.st_dev, anchored.st_ino) != (
            root_before.st_dev,
            root_before.st_ino,
        ):
            _fail(FAIL_ARTIFACT, "host staging root changed before write")
        os.mkdir(HOST_STAGED_SIDECAR_ROOT, 0o700, dir_fd=root_descriptor)
        sidecars_descriptor = os.open(
            HOST_STAGED_SIDECAR_ROOT,
            root_flags,
            dir_fd=root_descriptor,
        )
        for name in ("neutral", "preimages"):
            os.mkdir(name, 0o700, dir_fd=sidecars_descriptor)
            directory_descriptors[name] = os.open(
                name,
                root_flags,
                dir_fd=sidecars_descriptor,
            )
        payloads = replay.python.payloads
        if type(payloads) is not tuple or len(payloads) != 5:
            _fail(FAIL_ARTIFACT, "host replay sidecar payload registry differs")
        rows: list[list[object]] = []
        for relative, payload in zip(HOST_STAGED_SIDECAR_PATHS, payloads, strict=True):
            parent, name = relative.split("/", 1)
            status = _write_exclusive_sealed_file_v1(
                directory_descriptors[parent],
                name,
                payload,
            )
            rows.append(
                [relative, status.st_size, sha256(payload).hexdigest(), 0o444]
            )
        witness_status = _write_exclusive_sealed_file_v1(
            root_descriptor,
            HOST_SEMANTIC_WITNESS_RELATIVE_PATH,
            witness,
        )
        rows.append(
            [
                HOST_SEMANTIC_WITNESS_RELATIVE_PATH,
                witness_status.st_size,
                sha256(witness).hexdigest(),
                0o444,
            ]
        )
        for descriptor in directory_descriptors.values():
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        os.fchmod(sidecars_descriptor, 0o555)
        os.fsync(sidecars_descriptor)
        os.fchmod(root_descriptor, 0o555)
        os.fsync(root_descriptor)
        evidence = {
            "schema_version": "hegel-phase3a-q05b-host-semantic-staging/1",
            "file_count": 6,
            "file_rows": rows,
            "semantic_replay_root": replay.dual_replay_root.hex(),
            "witness_root": host_module.decode_host_semantic_witness_v1(witness)[
                "witness_root"
            ],
        }
        evidence["staging_manifest_sha256"] = sha256(
            _canonical_json_bytes(evidence)
        ).hexdigest()
        return witness, evidence
    finally:
        for descriptor in directory_descriptors.values():
            os.close(descriptor)
        if sidecars_descriptor is not None:
            os.close(sidecars_descriptor)
        os.close(root_descriptor)


def host_control_stdout_bytes_v1(
    replay: object,
    witness: bytes,
    host_source_identity_root: bytes,
    host_runtime_identity_root: bytes,
    host_module: object,
    loaded_module_rows: tuple[tuple[str, str | None, str | None], ...],
) -> bytes:
    decoded = host_module.decode_host_semantic_witness_v1(witness)
    value = {
        "action_id": "trusted-host-semantic-replay-v1",
        "actor_id": "TRUSTED_HOST_REPLAY",
        "file_count": 6,
        "final_isolation_root": None,
        "implementation_id": "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1",
        "loaded_module_rows": [list(row) for row in loaded_module_rows],
        "loaded_module_root": sha256(
            b"HEGEL/Q05B/HOST/LOADED_MODULE_CLOSURE/V1\x00"
            + _canonical_json_bytes([list(row) for row in loaded_module_rows])
        ).hexdigest(),
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "qualification_receipt": None,
        "runtime_identity_sha256": host_runtime_identity_root.hex(),
        "schema_version": HOST_CONTROL_STDOUT_SCHEMA_VERSION,
        "semantic_replay_root": replay.dual_replay_root.hex(),
        "source_identity_sha256": host_source_identity_root.hex(),
        "status": HOST_CONTROL_STDOUT_STATUS,
        "witness_length": len(witness),
        "witness_relative_path": HOST_SEMANTIC_WITNESS_RELATIVE_PATH,
        "witness_root": decoded["witness_root"],
        "witness_sha256": sha256(witness).hexdigest(),
    }
    payload = _canonical_json_bytes(value)
    _validate_held_actor_stdout_v1(payload, "TRUSTED_HOST_REPLAY")
    return payload


def host_source_identity_digest_v1(snapshot_root: Path) -> str:
    """Replay the host allowlist with the actor raw-framing algorithm."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(snapshot_root, flags)
    try:
        observed_paths = _snapshot_file_paths_v1(descriptor)
        if observed_paths != HOST_SOURCE_ALLOWLIST:
            _fail(FAIL_SOURCE, "trusted-host snapshot allowlist differs")
        digest = sha256()
        for relative in HOST_SOURCE_ALLOWLIST:
            payload, _status = _read_snapshot_file_v1(descriptor, relative)
            path_bytes = relative.encode("utf-8")
            digest.update(len(path_bytes).to_bytes(4, "big"))
            digest.update(path_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def python_runtime_identity_digest_v1() -> str:
    """Use the same pinned Python runtime identity preimage as the endpoint."""

    executable = Path(sys.executable).resolve(strict=True)
    digest = sha256(b"HEGEL/Q05B/PYTHON_RUNTIME_IDENTITY/V1\x00")
    path_bytes = executable.as_posix().encode("utf-8")
    digest.update(len(path_bytes).to_bytes(4, "big"))
    digest.update(path_bytes)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(executable, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(FAIL_SOURCE, "Python runtime executable is not regular")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(FAIL_SOURCE, "Python runtime executable changed while hashing")
    finally:
        os.close(descriptor)
    version = sys.version.encode("utf-8")
    digest.update(len(version).to_bytes(4, "big"))
    digest.update(version)
    return digest.hexdigest()


def host_loaded_module_rows_v1(
    snapshot_root: Path,
) -> tuple[tuple[str, str | None, str | None], ...]:
    """Bind the exact target-blind project-module closure loaded by the child."""

    forbidden = {
        "hegel_machine.__init__",
        "hegel_machine.phase3_dsl_v1",
        "hegel_machine.phase3_m25_rows_v1",
        "hegel_machine.phase3_m25_split_v1",
        "hegel_machine.phase3_m25_formal_static_basis_v1",
    }
    rows: list[tuple[str, str | None, str | None]] = []
    root = snapshot_root.resolve(strict=True)
    for module_name in sorted(
        name
        for name in sys.modules
        if name == "hegel_machine" or name.startswith("hegel_machine.")
    ):
        if module_name in forbidden:
            _fail(FAIL_SOURCE, f"forbidden trusted-host module loaded: {module_name}")
        module = sys.modules[module_name]
        source = getattr(module, "__file__", None)
        if module_name == "hegel_machine":
            if source is not None or getattr(module, "__spec__", None) is not None:
                _fail(FAIL_SOURCE, "trusted-host empty package acquired initializer identity")
            rows.append((module_name, None, None))
            continue
        if type(source) is not str:
            _fail(FAIL_SOURCE, f"trusted-host module lacks source path: {module_name}")
        source_path = Path(source).resolve(strict=True)
        try:
            relative = source_path.relative_to(root).as_posix()
        except ValueError:
            _fail(FAIL_SOURCE, f"trusted-host module escaped snapshot: {module_name}")
        if relative not in HOST_SOURCE_ALLOWLIST or relative.endswith("/__init__.py"):
            _fail(FAIL_SOURCE, f"trusted-host module is outside allowlist: {module_name}")
        payload = source_path.read_bytes()
        rows.append((module_name, relative, sha256(payload).hexdigest()))
    if not rows or rows[0] != ("hegel_machine", None, None):
        _fail(FAIL_SOURCE, "trusted-host empty package row is absent")
    return tuple(rows)


def internal_host_replay_v1(
    python_output: Path,
    rust_output: Path,
    python_stdout_path: Path,
    rust_stdout_path: Path,
    stdout_manifest_path: Path,
    staging_output: Path,
    host_source_identity_root_hex: str,
    host_runtime_identity_root_hex: str,
    *,
    snapshot_root: Path = PROJECT_ROOT,
) -> bytes:
    """Container-only semantic replay; it never claims final isolation."""

    for name, value in (
        ("host source identity", host_source_identity_root_hex),
        ("host runtime identity", host_runtime_identity_root_hex),
    ):
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            _fail(FAIL_SOURCE, f"{name} differs")
    if host_source_identity_digest_v1(snapshot_root) != host_source_identity_root_hex:
        _fail(FAIL_SOURCE, "trusted-host child source identity differs from outer")
    if python_runtime_identity_digest_v1() != host_runtime_identity_root_hex:
        _fail(FAIL_SOURCE, "trusted-host child runtime identity differs from outer")
    host_module = _load_host_replay_module_v1(snapshot_root)
    python_stdout = _read_sealed_regular_file_v1(
        python_stdout_path, 1024 * 1024, "Python stdout"
    )
    rust_stdout = _read_sealed_regular_file_v1(
        rust_stdout_path, 1024 * 1024, "Rust stdout"
    )
    stdout_manifest = _read_sealed_regular_file_v1(
        stdout_manifest_path, 1024 * 1024, "stdout manifest"
    )
    replay = host_module.dual_actor_host_replay_v1(
        python_stdout,
        python_output,
        rust_stdout,
        rust_output,
        stdout_manifest,
        bytes.fromhex(host_source_identity_root_hex),
        bytes.fromhex(host_runtime_identity_root_hex),
    )
    negative_module = importlib.import_module(
        "hegel_machine.phase3_q05b_negative_vectors_v1"
    )
    strict_cbor_module = importlib.import_module("hegel_machine.strict_cbor_v1")
    negative_corpus = negative_module.run_q05b_negative_vector_corpus_v1()
    negative_corpus_cbor = strict_cbor_module.canonical_cbor_encode(
        negative_corpus.canonical_object()
    )
    witness, _staging_evidence = write_host_semantic_staging_v1(
        replay,
        staging_output,
        host_module,
        negative_corpus_cbor,
        negative_corpus.corpus_root,
        negative_corpus.category_roots,
    )
    loaded_module_rows = host_loaded_module_rows_v1(snapshot_root)
    if (
        host_source_identity_digest_v1(snapshot_root)
        != host_source_identity_root_hex
        or python_runtime_identity_digest_v1()
        != host_runtime_identity_root_hex
    ):
        _fail(FAIL_SOURCE, "trusted-host source/runtime identity changed during replay")
    return host_control_stdout_bytes_v1(
        replay,
        witness,
        bytes.fromhex(host_source_identity_root_hex),
        bytes.fromhex(host_runtime_identity_root_hex),
        host_module,
        loaded_module_rows,
    )


def local_pinned_image_evidence_v1(
    image: str,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    if image not in (PYTHON_IMAGE, RUST_IMAGE) or not callable(runner):
        _fail(FAIL_POLICY, "pinned image evidence input differs")
    completed = runner(
        [DOCKER_EXECUTABLE, f"--host={DOCKER_HOST}", "image", "inspect", image],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    returncode = getattr(completed, "returncode", None)
    stdout = getattr(completed, "stdout", None)
    stderr = getattr(completed, "stderr", None)
    if (
        type(returncode) is not int
        or returncode != 0
        or type(stdout) is not bytes
        or not stdout
        or type(stderr) is not bytes
        or stderr
    ):
        _fail(FAIL_POLICY, "pinned local image inspect failed")
    value = _strict_json_value_v1(stdout, "pinned image inspect")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(FAIL_POLICY, "pinned image inspect shape differs")
    document = value[0]
    repo_digests = document.get("RepoDigests")
    config = document.get("Config")
    if (
        type(document.get("Id")) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", document["Id"]) is None
        or type(repo_digests) is not list
        or image not in repo_digests
        or any(type(item) is not str for item in repo_digests)
        or document.get("Os") != "linux"
        or type(document.get("Architecture")) is not str
        or type(config) is not dict
        or type(config.get("Env")) is not list
        or any(type(item) is not str for item in config["Env"])
    ):
        _fail(FAIL_POLICY, "pinned image identity differs")
    evidence: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-pinned-local-image-evidence/1",
        "requested_reference": image,
        "image_id": document["Id"],
        "repo_digests": repo_digests,
        "os": document["Os"],
        "architecture": document["Architecture"],
        "raw_inspect_hex": stdout.hex(),
        "raw_inspect_sha256": sha256(stdout).hexdigest(),
    }
    evidence["evidence_sha256"] = sha256(_canonical_json_bytes(evidence)).hexdigest()
    return evidence


def cross_docker_authority_to_pinned_image_labels_v1(
    authority: Mapping[str, object],
    image_evidence: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Close authority base-label claims against both raw pinned-image inspects."""

    if (
        type(authority) is not dict
        or type(image_evidence) is not dict
        or set(image_evidence) != {"python", "rust"}
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Docker image-label cross input differs")
    observed_base: dict[str, dict[str, str]] = {}
    for key, image in (("python", PYTHON_IMAGE), ("rust", RUST_IMAGE)):
        evidence = image_evidence[key]
        if (
            type(evidence) is not dict
            or evidence.get("requested_reference") != image
            or type(evidence.get("raw_inspect_hex")) is not str
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "Docker pinned image-label evidence differs")
        try:
            raw = bytes.fromhex(evidence["raw_inspect_hex"])
        except ValueError as error:
            _fail(FAIL_ACTUAL_ADMISSION, f"Docker image inspect hex differs: {error}")
        if sha256(raw).hexdigest() != evidence.get("raw_inspect_sha256"):
            _fail(FAIL_ACTUAL_ADMISSION, "Docker image inspect digest differs")
        value = _strict_json_value_v1(raw, "Docker pinned image label inspect")
        if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
            _fail(FAIL_ACTUAL_ADMISSION, "Docker pinned image label inspect differs")
        config = value[0].get("Config")
        labels = config.get("Labels") if type(config) is dict else None
        if labels is None:
            labels = {}
        if (
            type(labels) is not dict
            or any(type(name) is not str or type(item) is not str for name, item in labels.items())
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "Docker pinned image base labels differ")
        observed_base[key] = dict(labels)
    for slot_id, slot in DOCKER_EXECUTION_SLOT_REGISTRY:
        row = _docker_execution_slot_row_v1(authority, slot)
        base_labels = {
            name: value
            for name, value in row["expected_container_labels"]
            if name not in DOCKER_EXECUTION_RESERVED_LABEL_KEYS
        }
        image_key = (
            "rust"
            if slot in {"RUST_TEST", "RUST_RELEASE", "RUST_ENDPOINT"}
            else "python"
        )
        if row["slot_id"] != slot_id or base_labels != observed_base[image_key]:
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "Docker authority/pinned image base-label cross differs",
            )
    return {
        "python_base_labels": observed_base["python"],
        "rust_base_labels": observed_base["rust"],
        "docker_execution_authority_manifest_sha256": authority[
            "manifest_sha256"
        ],
    }


def sealed_snapshot_path_evidence_v1(
    root: Path,
    relative_paths: Sequence[str],
) -> dict[str, object]:
    legacy_identity = sealed_snapshot_identity_v1(root, relative_paths)
    modes = {
        row[0]: row[6]
        for row in legacy_identity["file_rows"]
    }
    identity = sealed_tree_identity_v1(
        root,
        tuple(relative_paths),
        expected_file_modes=modes,
    )
    if identity["root_nlink"] < 2:
        _fail(FAIL_SOURCE, "sealed snapshot root link identity differs")
    return identity


def seal_actor_sidecar_tree_v1(
    root: Path,
    host_module: object,
) -> tuple[tuple[bytes, ...], dict[str, object]]:
    try:
        before = host_module.read_exact_sidecar_tree_v1(root)
    except AttributeError as error:
        _fail(FAIL_POLICY, f"host sidecar replay API differs: {error}")
    seal_directory_tree_read_only_v1(root)
    after = host_module.read_exact_sidecar_tree_v1(root)
    if before != after:
        _fail(FAIL_POLICY, "actor sidecar bytes changed while sealing")
    identity = sealed_tree_identity_v1(
        root,
        tuple(sorted(HOST_STAGED_SIDECAR_PATHS)),
    )
    return after, identity


def seal_endpoint_stdout_set_v1(
    root: Path,
    python_stdout: bytes,
    rust_stdout: bytes,
    host_module: object,
    *,
    precreated_empty: bool = False,
) -> tuple[Path, Path, Path, bytes, dict[str, object]]:
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or type(python_stdout) is not bytes
        or type(rust_stdout) is not bytes
        or type(precreated_empty) is not bool
    ):
        _fail(FAIL_POLICY, "stdout set seal input differs")
    if root.exists():
        root_status = root.lstat()
        if (
            not precreated_empty
            or root.is_symlink()
            or not stat.S_ISDIR(root_status.st_mode)
            or stat.S_IMODE(root_status.st_mode) != 0o700
            or tuple(root.iterdir())
        ):
            _fail(FAIL_POLICY, "precreated stdout root differs")
    else:
        if precreated_empty:
            _fail(FAIL_POLICY, "precreated stdout root is absent")
        root.mkdir(mode=0o700)
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        manifest = host_module.sealed_actor_stdout_manifest_bytes_v1(
            python_stdout,
            rust_stdout,
        )
        for name, payload in (
            ("python.stdout", python_stdout),
            ("rust.stdout", rust_stdout),
            ("manifest.json", manifest),
        ):
            _write_exclusive_sealed_file_v1(descriptor, name, payload)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    host_module.validate_sealed_actor_stdout_set_v1(
        python_stdout,
        rust_stdout,
        manifest,
    )
    identity = sealed_tree_identity_v1(
        root,
        ("manifest.json", "python.stdout", "rust.stdout"),
        maximum_file_bytes=1024 * 1024,
    )
    return (
        root / "python.stdout",
        root / "rust.stdout",
        root / "manifest.json",
        manifest,
        identity,
    )


def detach_cargo_release_binary_v1(
    source: Path,
    destination: Path,
) -> dict[str, object]:
    """Detach Cargo's hard-linked release output into one private inode.

    Cargo normally hard-links ``target/release/NAME`` to the corresponding
    file below ``target/release/deps``.  The runtime sealer deliberately
    rejects that shared inode.  This boundary replays the Cargo path through
    anchored directory/file descriptors, copies its stable bytes into one
    fresh supervisor-owned directory, and leaves a mode-0755, nlink-1 file for
    :func:`seal_prebuilt_binary_v1` to freeze to mode 0555.  A failed copy
    never deletes or renames the destination entry: its state is deliberately
    left to the actual run's one-shot cleanup of the owned private work root.
    """

    if (
        not isinstance(source, Path)
        or not source.is_absolute()
        or source.name in ("", ".", "..")
        or not isinstance(destination, Path)
        or not destination.is_absolute()
        or destination.name in ("", ".", "..")
        or source == destination
        or source.parent == destination.parent
        or os.path.lexists(destination)
    ):
        _fail(FAIL_SOURCE, "Cargo binary detach input differs")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_parent_descriptor: int | None = None
    source_descriptor: int | None = None
    destination_parent_descriptor: int | None = None
    destination_descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    complete = False
    failure: BaseException | None = None
    result: dict[str, object] | None = None

    def parent_identity(value: os.stat_result) -> dict[str, int]:
        return {
            "device": value.st_dev,
            "inode": value.st_ino,
            "nlink": value.st_nlink,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": stat.S_IMODE(value.st_mode),
        }

    def file_identity(value: os.stat_result) -> dict[str, int]:
        return {
            "device": value.st_dev,
            "inode": value.st_ino,
            "nlink": value.st_nlink,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": stat.S_IMODE(value.st_mode),
            "size": value.st_size,
            "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns,
        }

    def read_descriptor(
        descriptor: int,
        maximum: int,
        label: str,
    ) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(
                descriptor,
                min(1024 * 1024, maximum + 1 - total),
            )
            if not block:
                break
            total += len(block)
            if total > maximum:
                _fail(FAIL_SOURCE, f"{label} exceeds binary size bound")
            blocks.append(block)
        return b"".join(blocks)

    try:
        source_parent_before = source.parent.lstat()
        destination_parent_before = destination.parent.lstat()
        if (
            source.parent.is_symlink()
            or not stat.S_ISDIR(source_parent_before.st_mode)
            or source.parent.resolve(strict=True) != source.parent
            or destination.parent.is_symlink()
            or not stat.S_ISDIR(destination_parent_before.st_mode)
            or destination.parent.resolve(strict=True) != destination.parent
            or stat.S_IMODE(destination_parent_before.st_mode) != 0o700
            or destination_parent_before.st_uid != os.getuid()
            or destination_parent_before.st_gid != os.getgid()
            or tuple(destination.parent.iterdir())
        ):
            _fail(FAIL_SOURCE, "Cargo binary detach parent identity differs")
        source_parent_descriptor = os.open(source.parent, directory_flags)
        destination_parent_descriptor = os.open(
            destination.parent,
            directory_flags,
        )
        anchored_source_parent = os.fstat(source_parent_descriptor)
        anchored_destination_parent = os.fstat(destination_parent_descriptor)
        if (
            parent_identity(anchored_source_parent)
            != parent_identity(source_parent_before)
            or parent_identity(anchored_destination_parent)
            != parent_identity(destination_parent_before)
            or os.listdir(destination_parent_descriptor)
        ):
            _fail(FAIL_SOURCE, "Cargo binary detach parent changed before open")

        source_descriptor = os.open(
            source.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_parent_descriptor,
        )
        source_before = os.fstat(source_descriptor)
        source_entry_before = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_nlink < 1
            or stat.S_IMODE(source_before.st_mode) != 0o755
            or not 0 < source_before.st_size <= 256 * 1024 * 1024
            or file_identity(source_before) != file_identity(source_entry_before)
            or source.resolve(strict=True) != source
        ):
            _fail(FAIL_SOURCE, "Cargo release binary identity differs")
        payload = read_descriptor(
            source_descriptor,
            256 * 1024 * 1024,
            "Cargo release binary",
        )
        if len(payload) != source_before.st_size:
            _fail(FAIL_SOURCE, "Cargo release binary size changed")

        destination_descriptor = os.open(
            destination.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o700,
            dir_fd=destination_parent_descriptor,
        )
        destination_created = os.fstat(destination_descriptor)
        created_identity = (
            destination_created.st_dev,
            destination_created.st_ino,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(destination_descriptor, payload[offset:])
            if type(written) is not int or written <= 0:
                _fail(FAIL_SOURCE, "detached binary write made no progress")
            offset += written
        os.fsync(destination_descriptor)
        os.fchmod(destination_descriptor, 0o755)
        os.fsync(destination_descriptor)
        detached_payload = read_descriptor(
            destination_descriptor,
            256 * 1024 * 1024,
            "detached runtime binary",
        )
        detached = os.fstat(destination_descriptor)
        detached_entry = os.stat(
            destination.name,
            dir_fd=destination_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            detached_payload != payload
            or not stat.S_ISREG(detached.st_mode)
            or detached.st_nlink != 1
            or stat.S_IMODE(detached.st_mode) != 0o755
            or detached.st_size != len(payload)
            or file_identity(detached) != file_identity(detached_entry)
            or (detached.st_dev, detached.st_ino) != created_identity
        ):
            _fail(FAIL_SOURCE, "detached runtime binary identity differs")

        source_after = os.fstat(source_descriptor)
        source_payload_after = read_descriptor(
            source_descriptor,
            256 * 1024 * 1024,
            "Cargo release binary replay",
        )
        source_final = os.fstat(source_descriptor)
        source_entry_after = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        source_parent_after = os.fstat(source_parent_descriptor)
        source_parent_path_after = source.parent.lstat()
        if (
            file_identity(source_before) != file_identity(source_after)
            or file_identity(source_after) != file_identity(source_final)
            or file_identity(source_after) != file_identity(source_entry_after)
            or source_payload_after != payload
            or parent_identity(anchored_source_parent)
            != parent_identity(source_parent_after)
            or parent_identity(source_parent_after)
            != parent_identity(source_parent_path_after)
            or not stat.S_ISDIR(source_parent_path_after.st_mode)
            or source.parent.resolve(strict=True) != source.parent
            or source.resolve(strict=True) != source
        ):
            _fail(FAIL_SOURCE, "Cargo release binary changed during detach")

        os.fsync(destination_parent_descriptor)
        detached_entry_after_fsync = os.stat(
            destination.name,
            dir_fd=destination_parent_descriptor,
            follow_symlinks=False,
        )
        destination_parent_after = os.fstat(destination_parent_descriptor)
        destination_parent_path_after = destination.parent.lstat()
        if (
            parent_identity(anchored_destination_parent)
            != parent_identity(destination_parent_after)
            or parent_identity(destination_parent_after)
            != parent_identity(destination_parent_path_after)
            or not stat.S_ISDIR(destination_parent_path_after.st_mode)
            or file_identity(detached)
            != file_identity(detached_entry_after_fsync)
            or destination.parent.resolve(strict=True) != destination.parent
            or destination.resolve(strict=True) != destination
        ):
            _fail(FAIL_SOURCE, "detached binary parent/path changed")

        payload_sha256 = sha256(payload).hexdigest()
        body: dict[str, object] = {
            "schema_version": (
                "hegel-phase3a-q05b-detached-cargo-release-binary/1"
            ),
            "source_path": source.as_posix(),
            "detached_path": destination.as_posix(),
            "source_parent_before": parent_identity(
                anchored_source_parent
            ),
            "source_parent_after": parent_identity(source_parent_after),
            "source_fd_before": file_identity(source_before),
            "source_fd_after": file_identity(source_final),
            "source_path_before": file_identity(source_entry_before),
            "source_path_after": file_identity(source_entry_after),
            "source_sha256_before": payload_sha256,
            "source_sha256_after": sha256(source_payload_after).hexdigest(),
            "detached_parent_before": parent_identity(
                anchored_destination_parent
            ),
            "detached_parent_after": parent_identity(
                destination_parent_after
            ),
            "detached_fd": file_identity(detached),
            "detached_path_identity": file_identity(
                detached_entry_after_fsync
            ),
            "detached_sha256": sha256(detached_payload).hexdigest(),
            "source_and_detached_bytes_equal": True,
        }
        body["manifest_sha256"] = sha256(
            _canonical_json_bytes(body)
        ).hexdigest()
        result = body
        complete = True
    except BaseException as error:
        failure = error
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)
        if source_parent_descriptor is not None:
            os.close(source_parent_descriptor)
        if destination_parent_descriptor is not None:
            os.close(destination_parent_descriptor)

    if failure is not None:
        if destination_descriptor is not None:
            failure = Q05BDualSupervisorError(
                FAIL_SOURCE,
                "Cargo binary detach failed; named destination state is "
                "unresolved; deferred outer-owned-root cleanup required; "
                f"original={type(failure).__name__}:{failure}",
            )
        if isinstance(failure, Q05BDualSupervisorError):
            raise failure
        _fail(
            FAIL_SOURCE,
            f"Cargo binary detach failed: {type(failure).__name__}:{failure}",
        )
    if not complete or result is None:
        _fail(FAIL_SOURCE, "Cargo binary detach produced no evidence")
    return result


def validate_detached_binary_binding_v1(
    detach_evidence: Mapping[str, object],
    sealed_binary_evidence: Mapping[str, object],
    source: Path,
    destination: Path,
) -> dict[str, object]:
    """Replay detach evidence and bind it to the existing sealed binary."""

    if (
        type(detach_evidence) is not dict
        or type(sealed_binary_evidence) is not dict
        or not isinstance(source, Path)
        or not source.is_absolute()
        or not isinstance(destination, Path)
        or not destination.is_absolute()
    ):
        _fail(FAIL_SOURCE, "detached binary binding input differs")
    expected_keys = {
        "schema_version",
        "source_path",
        "detached_path",
        "source_parent_before",
        "source_parent_after",
        "source_fd_before",
        "source_fd_after",
        "source_path_before",
        "source_path_after",
        "source_sha256_before",
        "source_sha256_after",
        "detached_parent_before",
        "detached_parent_after",
        "detached_fd",
        "detached_path_identity",
        "detached_sha256",
        "source_and_detached_bytes_equal",
        "manifest_sha256",
    }
    if set(detach_evidence) != expected_keys:
        _fail(FAIL_SOURCE, "detached binary evidence fields differ")
    body = dict(detach_evidence)
    manifest = body.pop("manifest_sha256")
    parent_keys = {"device", "inode", "nlink", "uid", "gid", "mode"}
    file_keys = parent_keys | {"size", "mtime_ns", "ctime_ns"}
    parent_rows = (
        detach_evidence["source_parent_before"],
        detach_evidence["source_parent_after"],
        detach_evidence["detached_parent_before"],
        detach_evidence["detached_parent_after"],
    )
    file_rows = (
        detach_evidence["source_fd_before"],
        detach_evidence["source_fd_after"],
        detach_evidence["source_path_before"],
        detach_evidence["source_path_after"],
        detach_evidence["detached_fd"],
        detach_evidence["detached_path_identity"],
    )
    source_row = detach_evidence["source_fd_after"]
    detached_row = detach_evidence["detached_fd"]
    if (
        detach_evidence["schema_version"]
        != "hegel-phase3a-q05b-detached-cargo-release-binary/1"
        or type(manifest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", manifest) is None
        or manifest != sha256(_canonical_json_bytes(body)).hexdigest()
        or detach_evidence["source_path"] != source.as_posix()
        or detach_evidence["detached_path"] != destination.as_posix()
        or any(
            type(row) is not dict
            or set(row) != parent_keys
            or any(type(value) is not int for value in row.values())
            for row in parent_rows
        )
        or any(
            type(row) is not dict
            or set(row) != file_keys
            or any(type(value) is not int for value in row.values())
            for row in file_rows
        )
        or detach_evidence["source_parent_before"]
        != detach_evidence["source_parent_after"]
        or detach_evidence["detached_parent_before"]
        != detach_evidence["detached_parent_after"]
        or detach_evidence["source_fd_before"]
        != detach_evidence["source_fd_after"]
        or detach_evidence["source_fd_after"]
        != detach_evidence["source_path_before"]
        or detach_evidence["source_path_before"]
        != detach_evidence["source_path_after"]
        or detach_evidence["detached_fd"]
        != detach_evidence["detached_path_identity"]
        or type(detach_evidence["source_sha256_before"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", detach_evidence["source_sha256_before"]
        )
        is None
        or detach_evidence["source_sha256_before"]
        != detach_evidence["source_sha256_after"]
        or detach_evidence["source_sha256_after"]
        != detach_evidence["detached_sha256"]
        or source_row.get("size") != detached_row.get("size")
        or source_row.get("mode") != 0o755
        or type(source_row.get("nlink")) is not int
        or source_row["nlink"] < 1
        or detached_row.get("nlink") != 1
        or detached_row.get("mode") != 0o755
        or detach_evidence["source_and_detached_bytes_equal"] is not True
        or sealed_binary_evidence.get("binary_path") != destination.as_posix()
        or sealed_binary_evidence.get("device") != detached_row.get("device")
        or sealed_binary_evidence.get("inode") != detached_row.get("inode")
        or sealed_binary_evidence.get("nlink") != detached_row.get("nlink")
        or sealed_binary_evidence.get("uid") != detached_row.get("uid")
        or sealed_binary_evidence.get("gid") != detached_row.get("gid")
        or sealed_binary_evidence.get("size") != detached_row.get("size")
        or sealed_binary_evidence.get("mtime_ns") != detached_row.get("mtime_ns")
        or sealed_binary_evidence.get("sha256")
        != detach_evidence["detached_sha256"]
        or sealed_binary_evidence.get("mode") != 0o555
        or type(sealed_binary_evidence.get("ctime_ns")) is not int
        or sealed_binary_evidence["ctime_ns"] < detached_row.get("ctime_ns")
    ):
        _fail(FAIL_SOURCE, "detached/sealed binary binding differs")

    source_parent_descriptor = os.open(
        source.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    source_descriptor: int | None = None
    try:
        source_parent_before = os.fstat(source_parent_descriptor)
        source_descriptor = os.open(
            source.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_parent_descriptor,
        )
        source_before = os.fstat(source_descriptor)
        source_path_before = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        payload = bytearray()
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > 256 * 1024 * 1024:
                _fail(FAIL_SOURCE, "Cargo release source replay exceeds bound")
        source_after = os.fstat(source_descriptor)
        source_path_after = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        source_parent_after = os.fstat(source_parent_descriptor)
        source_parent_path_after = source.parent.lstat()
        source_entry_path_after = source.lstat()
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(source_parent_descriptor)

    def live_parent_row(value: os.stat_result) -> dict[str, int]:
        return {
            "device": value.st_dev,
            "inode": value.st_ino,
            "nlink": value.st_nlink,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": stat.S_IMODE(value.st_mode),
        }

    def live_file_row(value: os.stat_result) -> dict[str, int]:
        return {
            **live_parent_row(value),
            "size": value.st_size,
            "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns,
        }

    if (
        live_parent_row(source_parent_before)
        != detach_evidence["source_parent_after"]
        or live_parent_row(source_parent_after)
        != detach_evidence["source_parent_after"]
        or live_parent_row(source_parent_path_after)
        != detach_evidence["source_parent_after"]
        or live_file_row(source_before) != source_row
        or live_file_row(source_after) != source_row
        or live_file_row(source_path_before) != source_row
        or live_file_row(source_path_after) != source_row
        or live_file_row(source_entry_path_after) != source_row
        or sha256(bytes(payload)).hexdigest()
        != detach_evidence["source_sha256_after"]
        or not stat.S_ISDIR(source_parent_path_after.st_mode)
        or source.parent.resolve(strict=True) != source.parent
        or source.resolve(strict=True) != source
        or replay_sealed_prebuilt_binary_v1(
            destination, sealed_binary_evidence
        )
        != sealed_binary_evidence
    ):
        _fail(FAIL_SOURCE, "Cargo release source changed after detach")
    return dict(detach_evidence)


def seal_prebuilt_binary_v1(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_absolute():
        _fail(FAIL_SOURCE, "prebuilt binary path differs")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= 256 * 1024 * 1024
        ):
            _fail(FAIL_SOURCE, "prebuilt binary identity differs")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        if len(payload) != before.st_size:
            _fail(FAIL_SOURCE, "prebuilt binary size changed")
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
        ) or stat.S_IMODE(after.st_mode) != 0o555:
            _fail(FAIL_SOURCE, "prebuilt binary changed while sealing")
    finally:
        os.close(descriptor)
    evidence: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-sealed-prebuilt-rust-binary/1",
        "binary_path": path.resolve(strict=True).as_posix(),
        "device": after.st_dev,
        "inode": after.st_ino,
        "nlink": after.st_nlink,
        "uid": after.st_uid,
        "gid": after.st_gid,
        "mode": stat.S_IMODE(after.st_mode),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "sha256": sha256(payload).hexdigest(),
        "payload_hex": payload.hex(),
    }
    evidence["manifest_sha256"] = sha256(_canonical_json_bytes(evidence)).hexdigest()
    return evidence


def replay_sealed_prebuilt_binary_v1(
    path: Path,
    expected: Mapping[str, object],
) -> dict[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o555
            or before.st_nlink != 1
            or not 0 < before.st_size <= 256 * 1024 * 1024
        ):
            _fail(FAIL_SOURCE, "sealed prebuilt binary replay identity differs")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(payload) != before.st_size:
            _fail(FAIL_SOURCE, "sealed prebuilt binary changed during replay")
    finally:
        os.close(descriptor)
    replay: dict[str, object] = {
        "schema_version": "hegel-phase3a-q05b-sealed-prebuilt-rust-binary/1",
        "binary_path": path.resolve(strict=True).as_posix(),
        "device": after.st_dev,
        "inode": after.st_ino,
        "nlink": after.st_nlink,
        "uid": after.st_uid,
        "gid": after.st_gid,
        "mode": stat.S_IMODE(after.st_mode),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "sha256": sha256(payload).hexdigest(),
        "payload_hex": payload.hex(),
    }
    replay["manifest_sha256"] = sha256(_canonical_json_bytes(replay)).hexdigest()
    if replay != expected:
        _fail(FAIL_SOURCE, "sealed prebuilt binary changed after runtime")
    return replay


def _private_empty_directory_v1(parent: Path, name: str) -> Path:
    if (
        not isinstance(parent, Path)
        or not parent.is_absolute()
        or type(name) is not str
        or not name
        or "/" in name
        or name in (".", "..")
    ):
        _fail(FAIL_POLICY, "private directory input differs")
    parent_status = parent.lstat()
    if parent.is_symlink() or not stat.S_ISDIR(parent_status.st_mode):
        _fail(FAIL_POLICY, "private directory parent differs")
    descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        anchored = os.fstat(descriptor)
        if (anchored.st_dev, anchored.st_ino) != (
            parent_status.st_dev,
            parent_status.st_ino,
        ):
            _fail(FAIL_POLICY, "private directory parent changed")
        os.mkdir(name, 0o700, dir_fd=descriptor)
        os.fsync(descriptor)
        child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(child.st_mode) or stat.S_IMODE(child.st_mode) != 0o700:
            _fail(FAIL_POLICY, "private directory identity differs")
    except FileExistsError:
        _fail(FAIL_POLICY, "private directory already exists")
    finally:
        os.close(descriptor)
    return parent / name


def _bounded_process_state_v1(maximum: int) -> BoundedPipeDrainV1:
    return BoundedPipeDrainV1(maximum, bytearray(), 0, False, sha256(), [])


def _cleanup_offline_rust_failure_v1(
    process: subprocess.Popen[bytes],
    stdout_thread: threading.Thread,
    stderr_thread: threading.Thread,
    container_id: str | None,
    principal: Mapping[str, object],
    command_runner: Callable[..., object],
) -> tuple[str, ...]:
    """Keep CLI stop, pipe joins, and owned-CID cleanup signal-atomic."""

    with _docker_ownership_signal_guard_v1():
        errors: list[str] = []
        try:
            if process.poll() is None:
                process.kill()
        except BaseException as error:
            errors.append(f"docker-cli-kill:{type(error).__name__}:{error}")
        try:
            process.wait(timeout=2.0)
        except BaseException as error:
            errors.append(f"docker-cli-wait:{type(error).__name__}:{error}")
        for label, thread in (
            ("stdout", stdout_thread),
            ("stderr", stderr_thread),
        ):
            try:
                if thread.ident is not None:
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        errors.append(f"{label} pipe drain did not stop")
            except BaseException as error:
                errors.append(
                    f"{label}-pipe-join:{type(error).__name__}:{error}"
                )
        _docker_remove_and_quiet_absence_v1(
            container_id,
            principal,
            command_runner,
            errors,
        )
        return tuple(errors)


def run_offline_rust_build_container_v1(
    command: Sequence[str],
    cidfile: Path,
    *,
    docker_execution_authority: Mapping[str, object],
    docker_slot: str,
    timeout_seconds: float = 30 * 60,
    inspect_reader: Callable[[str], bytes] = _docker_inspect_payload_v1,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Run one exact offline Rust test/build container and retain raw evidence."""

    if (
        type(command) not in (tuple, list)
        or not command
        or not isinstance(cidfile, Path)
        or not cidfile.is_absolute()
        or cidfile.exists()
        or type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or not callable(inspect_reader)
        or not callable(command_runner)
    ):
        _fail(FAIL_POLICY, "offline Rust container input differs")
    exact = tuple(command)
    principal = _docker_execution_principal_v1(
        exact,
        docker_execution_authority,
        docker_slot,
    )
    if (
        exact[:3] != (DOCKER_EXECUTABLE, f"--host={DOCKER_HOST}", "run")
        or f"--cidfile={cidfile}" not in exact
        or "--network=none" not in exact
        or "--pull=never" not in exact
        or "CARGO_NET_OFFLINE=true" not in exact
        or "--cgroupns=private" not in exact
        or "--rm" in exact
    ):
        _fail(FAIL_POLICY, "offline Rust container command differs")
    command_security_options = tuple(
        item.removeprefix("--security-opt=")
        for item in exact
        if item.startswith("--security-opt=")
    )
    if (
        len(command_security_options) != 2
        or command_security_options[0] != "no-new-privileges"
        or not command_security_options[1].startswith("seccomp=")
    ):
        _fail(FAIL_POLICY, "offline Rust seccomp command binding differs")
    seccomp_path = Path(
        command_security_options[1].removeprefix("seccomp=")
    )
    seccomp_evidence = sealed_policy_file_evidence_v1(
        seccomp_path,
        BUILD_SECCOMP_RELATIVE_PATH,
    )
    name_index = exact.index("--name") if "--name" in exact else -1
    if name_index < 0 or name_index + 1 >= len(exact):
        _fail(FAIL_POLICY, "offline Rust container name differs")
    container_name = exact[name_index + 1]
    if container_name != principal["container_name"]:
        _fail(FAIL_POLICY, "offline Rust ownership name differs")
    parent = cidfile.parent.lstat()
    parent_identity = (
        parent.st_dev,
        parent.st_ino,
        stat.S_IMODE(parent.st_mode),
        parent.st_nlink,
    )
    precreate_absence = docker_precreate_absence_evidence_v1(
        docker_execution_authority,
        docker_slot,
        command_runner,
    )
    process: subprocess.Popen[bytes] | None = None
    stdout_state: BoundedPipeDrainV1 | None = None
    stderr_state: BoundedPipeDrainV1 | None = None
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    container_id: str | None = None
    complete = False
    cleanup_errors: list[str] = []
    try:
        # Establish the outer finally before Popen.  Signals stay blocked from
        # Popen's return through the complete worker-tuple handoff; when the
        # guard restores the prior mask, this same try/finally is already the
        # active exception frame.
        with _docker_ownership_signal_guard_v1():
            process = subprocess.Popen(
                list(exact),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                umask=0o077,
            )
            stdout_state, stderr_state, stdout_thread, stderr_thread = (
                _unstarted_pipe_workers_v1(
                    process,
                    16 * 1024 * 1024,
                    64 * 1024 * 1024,
                )
            )
        stdout_thread.start()
        stderr_thread.start()
        deadline = time.monotonic() + min(float(timeout_seconds), 60.0)
        cidfile_identity: tuple[int, int] | None = None
        cidfile_ready = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _fail(FAIL_POLICY, "offline Rust container exited before cidfile")
            cidfile_ready, cidfile_identity = _observe_unsealed_cidfile_v1(
                cidfile,
                parent_identity,
                cidfile_identity,
            )
            if cidfile_ready:
                break
            time.sleep(0.05)
        if not cidfile_ready:
            _fail(FAIL_POLICY, "offline Rust cidfile timed out")
        assert cidfile_identity is not None
        container_id, cid_evidence = _seal_cidfile_v1(
            cidfile,
            parent_identity,
            cidfile_identity,
        )
        live_inspect = inspect_reader(container_id)
        live_ownership = _validate_owned_docker_inspect_payload_v1(
            live_inspect,
            principal,
        )
        live_value = _strict_json_value_v1(live_inspect, "offline Rust live inspect")
        if (
            type(live_value) is not list
            or len(live_value) != 1
            or type(live_value[0]) is not dict
            or live_value[0].get("Id") != container_id
            or type(live_value[0].get("State")) is not dict
            or live_value[0]["State"].get("Running") is not True
        ):
            _fail(FAIL_POLICY, "offline Rust live inspect state differs")
        validate_docker_inspect_seccomp_semantics_v1(
            live_value[0].get("HostConfig", {}).get("SecurityOpt")
            if type(live_value[0].get("HostConfig")) is dict
            else None,
            command_security_options,
            seccomp_evidence,
            BUILD_SECCOMP_RELATIVE_PATH,
        )
        try:
            exit_code = process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            _fail(FAIL_POLICY, "offline Rust container timed out")
        stdout_thread.join(timeout=10.0)
        stderr_thread.join(timeout=10.0)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            _fail(FAIL_POLICY, "offline Rust pipe drains did not stop")
        for label, state in (("stdout", stdout_state), ("stderr", stderr_state)):
            if state.errors or state.overflow or len(state.payload) != state.total:
                _fail(FAIL_POLICY, f"offline Rust {label} transcript exceeded bound")
        if exit_code != 0:
            _fail(FAIL_POLICY, "offline Rust command failed")
        post_inspect = inspect_reader(container_id)
        post_ownership = _validate_owned_docker_inspect_payload_v1(
            post_inspect,
            principal,
        )
        post_value = _strict_json_value_v1(post_inspect, "offline Rust post inspect")
        if (
            type(post_value) is not list
            or len(post_value) != 1
            or type(post_value[0]) is not dict
            or type(post_value[0].get("State")) is not dict
            or post_value[0].get("Id") != container_id
            or post_value[0]["State"].get("Running") is not False
            or post_value[0]["State"].get("OOMKilled") is not False
            or post_value[0]["State"].get("ExitCode") != 0
            or type(post_value[0].get("HostConfig")) is not dict
            or post_value[0]["HostConfig"].get("AutoRemove") is not False
        ):
            _fail(FAIL_POLICY, "offline Rust post inspect state differs")
        validate_docker_inspect_seccomp_semantics_v1(
            post_value[0]["HostConfig"].get("SecurityOpt"),
            command_security_options,
            seccomp_evidence,
            BUILD_SECCOMP_RELATIVE_PATH,
        )
        explicit_remove_command = docker_explicit_remove_command_v1(container_id)
        removal = _run_docker_control_v1(
            explicit_remove_command,
            command_runner,
        )
        if getattr(removal, "returncode", None) != 0:
            _fail(FAIL_POLICY, "offline Rust container removal failed")
        absence = _docker_absence_evidence_v1(container_id, command_runner)
        complete = True
        evidence: dict[str, object] = {
            "schema_version": "hegel-phase3a-q05b-offline-rust-container-run/1",
            "command": list(exact),
            "command_sha256": sha256(_canonical_json_bytes(list(exact))).hexdigest(),
            "docker_execution_authority_manifest_sha256": principal[
                "authority_manifest_sha256"
            ],
            "docker_execution_slot_row": _docker_execution_slot_row_v1(
                docker_execution_authority,
                docker_slot,
            ),
            "ownership_label_root": principal["ownership_label_root"],
            "precreate_absence_evidence": precreate_absence,
            "cidfile_evidence": cid_evidence,
            "seccomp_evidence": seccomp_evidence,
            "live_inspect_hex": live_inspect.hex(),
            "live_inspect_sha256": sha256(live_inspect).hexdigest(),
            "live_ownership_inspect_evidence": live_ownership,
            "post_inspect_hex": post_inspect.hex(),
            "post_inspect_sha256": sha256(post_inspect).hexdigest(),
            "post_ownership_inspect_evidence": post_ownership,
            "stdout_hex": bytes(stdout_state.payload).hex(),
            "stdout_sha256": stdout_state.digest.hexdigest(),
            "stdout_length": stdout_state.total,
            "stderr_hex": bytes(stderr_state.payload).hex(),
            "stderr_sha256": stderr_state.digest.hexdigest(),
            "stderr_length": stderr_state.total,
            "exit_code": exit_code,
            "explicit_remove_command": explicit_remove_command,
            "cleanup_target_kind": "OWNERSHIP_VALIDATED_CONTAINER_ID",
            "container_name_was_never_a_destructive_target": True,
            "docker_absence_evidence": absence,
        }
        evidence["evidence_sha256"] = sha256(_canonical_json_bytes(evidence)).hexdigest()
        return evidence
    finally:
        if not complete:
            original_error = sys.exc_info()[1]
            if process is None:
                # Popen may have reached the daemon before raising or before
                # its result could be stored.  Only exact-owned discovery can
                # close this state; repeated name absence stays unresolved.
                with _docker_ownership_signal_guard_v1():
                    _docker_remove_and_quiet_absence_v1(
                        None,
                        principal,
                        command_runner,
                        cleanup_errors,
                    )
            elif stdout_thread is None or stderr_thread is None:
                cleanup_errors.extend(
                    _cleanup_unbound_post_popen_v1(
                        process,
                        principal,
                        command_runner,
                    )
                )
            else:
                cleanup_errors.extend(
                    _cleanup_offline_rust_failure_v1(
                        process,
                        stdout_thread,
                        stderr_thread,
                        container_id,
                        principal,
                        command_runner,
                    )
                )
            if cleanup_errors:
                original_detail = (
                    ""
                    if original_error is None
                    else (
                        "; original="
                        f"{type(original_error).__name__}:{original_error}"
                    )
                )
                _fail(
                    FAIL_POLICY,
                    "offline Rust failure cleanup closure failed: "
                    + "; ".join(cleanup_errors)
                    + original_detail,
                )


_ACTUAL_ADMISSION = _load_actual_admission_module_v1()

ACTUAL_ADMISSION_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_SCHEMA_VERSION
ACTUAL_PRECONDITION_BUNDLE_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_PRECONDITION_BUNDLE_SCHEMA_VERSION
ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION
ACTUAL_ADMISSION_DECISION_ID: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_DECISION_ID
ACTUAL_ADMISSION_DECISION_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_DECISION_ROOT_DOMAIN
ACTUAL_ADMISSION_ATTEMPT_ID_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_ATTEMPT_ID_DOMAIN
ACTUAL_PRECONDITION_EVIDENCE_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_PRECONDITION_EVIDENCE_ROOT_DOMAIN
ACTUAL_PRECONDITION_BUNDLE_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_PRECONDITION_BUNDLE_ROOT_DOMAIN
ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN
ACTUAL_PRECONDITION_REGISTRY_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_PRECONDITION_REGISTRY_ROOT_DOMAIN
ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN
ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION
ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_SCHEMA_VERSION
ACTUAL_FRESH_RUNTIME_EVIDENCE_OBJECT_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_EVIDENCE_OBJECT_ROOT_DOMAIN
ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_ROOT_DOMAIN
ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES: Final = _ACTUAL_ADMISSION.ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES
ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES
ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION
ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN
ACTUAL_FRESH_RUNTIME_CHECKPOINT_MAX_BYTES: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_CHECKPOINT_MAX_BYTES
ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY: Final = _ACTUAL_ADMISSION.ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY
ACTUAL_ACTOR_MOUNT_BINDING_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_ACTOR_MOUNT_BINDING_SCHEMA_VERSION
ACTUAL_ACTOR_MOUNT_SOURCE_SCHEMA_VERSION: Final = _ACTUAL_ADMISSION.ACTUAL_ACTOR_MOUNT_SOURCE_SCHEMA_VERSION
ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY: Final = _ACTUAL_ADMISSION.ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY
ACTUAL_RUNTIME_PRECONDITION_REGISTRY: Final = _ACTUAL_ADMISSION.ACTUAL_RUNTIME_PRECONDITION_REGISTRY
ACTUAL_PRECONDITION_PREIMAGE_SCHEMAS: Final = _ACTUAL_ADMISSION.ACTUAL_PRECONDITION_PREIMAGE_SCHEMAS
COMMIT_A_ACTUAL_ENGINEERING_STATUS: Final = _ACTUAL_ADMISSION.COMMIT_A_ACTUAL_ENGINEERING_STATUS
COMMIT_A_ACTUAL_PRECONDITIONS_V1: Final = _ACTUAL_ADMISSION.COMMIT_A_ACTUAL_PRECONDITIONS_V1
ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY
ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY: Final = _ACTUAL_ADMISSION.ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY


def actual_precondition_registry_root_v1() -> str:
    return _ACTUAL_ADMISSION.actual_precondition_registry_root_v1()


def build_fresh_runtime_evidence_set_v1(
    source_commit: str,
    image_rows: Sequence[Mapping[str, object]],
    actor_rows: Sequence[Mapping[str, object]],
    cargo_material_evidence: Mapping[str, object],
    cargo_snapshot_evidence: Mapping[str, object],
    cargo_tree_evidence: Mapping[str, object],
    seccomp_rows: Sequence[Mapping[str, object]],
    binary_evidence: Mapping[str, object],
) -> dict[str, object]:
    try:
        return _ACTUAL_ADMISSION.build_fresh_runtime_evidence_set_v1(
            source_commit,
            image_rows,
            actor_rows,
            cargo_material_evidence,
            cargo_snapshot_evidence,
            cargo_tree_evidence,
            seccomp_rows,
            binary_evidence,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def build_fresh_runtime_checkpoint_v1(
    source_commit: str,
    artifact_path: Path,
    checkpoint_id: int,
    attempt_id: str,
    boundary_root: str,
    issue_record_root: str,
    consumed_marker_root: str,
    issue_fresh_runtime_evidence: Mapping[str, object],
    observed_fresh_runtime_evidence: Mapping[str, object],
    artifact_absence_evidence: Mapping[str, object],
    mount_binding_rows: Sequence[Mapping[str, object]],
    dynamic_authority_set: Mapping[str, object] | None = None,
    stage_5_evidence: Mapping[str, object] | None = None,
    *,
    stage_5_issue_record: Mapping[str, object] | None = None,
    stage_5_consumed_marker_evidence: Mapping[str, object] | None = None,
    stage_5_checkpoint_1: Mapping[str, object] | None = None,
    stage_5_mount_launch_replay_rows: (
        Sequence[Mapping[str, object]] | None
    ) = None,
) -> dict[str, object]:
    try:
        value = _ACTUAL_ADMISSION.build_fresh_runtime_checkpoint_v1(
            source_commit,
            artifact_path.as_posix(),
            checkpoint_id,
            attempt_id,
            boundary_root,
            issue_record_root,
            consumed_marker_root,
            issue_fresh_runtime_evidence,
            observed_fresh_runtime_evidence,
            artifact_absence_evidence,
            mount_binding_rows,
            dynamic_authority_set,
            stage_5_evidence,
            stage_5_issue_record=stage_5_issue_record,
            stage_5_consumed_marker_evidence=(
                stage_5_consumed_marker_evidence
            ),
            stage_5_checkpoint_1=stage_5_checkpoint_1,
            stage_5_mount_launch_replay_rows=(
                stage_5_mount_launch_replay_rows
            ),
        )
        return _ACTUAL_ADMISSION.decode_fresh_runtime_checkpoint_v1(
            _canonical_json_bytes(value),
            source_commit,
            artifact_path.as_posix(),
            checkpoint_id,
            attempt_id,
            boundary_root,
            issue_record_root,
            consumed_marker_root,
            issue_fresh_runtime_evidence,
            artifact_absence_evidence,
            mount_binding_rows,
            dynamic_authority_set,
            stage_5_evidence,
            stage_5_issue_record=stage_5_issue_record,
            stage_5_consumed_marker_evidence=(
                stage_5_consumed_marker_evidence
            ),
            stage_5_checkpoint_1=stage_5_checkpoint_1,
            stage_5_mount_launch_replay_rows=(
                stage_5_mount_launch_replay_rows
            ),
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _validate_commit_a_actual_config_bytes_v1(
    commit_a_config_bytes: bytes,
) -> dict[str, object]:
    if (
        type(commit_a_config_bytes) is not bytes
        or not commit_a_config_bytes
        or len(commit_a_config_bytes) > 4 * 1024 * 1024
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "Commit-A config bytes differ")
    try:
        value = _strict_json_object_v1(
            commit_a_config_bytes,
            "Commit-A actual admission config",
        )
        return _validate_isolation_config_value_v1(
            value,
            engineering_status=COMMIT_A_ACTUAL_ENGINEERING_STATUS,
            actual_preconditions=COMMIT_A_ACTUAL_PRECONDITIONS_V1,
            project_root=None,
        )
    except Q05BDualSupervisorError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    raise AssertionError("unreachable")


def _admission_sha256_v1(value: object) -> str:
    try:
        return sha256(_canonical_json_bytes(value)).hexdigest()
    except (TypeError, ValueError) as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"admission preimage is not canonical JSON: {error}")


def _validate_prior_stage_rows_v1(
    rows: object,
    source_commit: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if type(rows) is not tuple or len(rows) != 3:
        _fail(FAIL_ACTUAL_ADMISSION, "prior stage evidence registry differs")
    result: list[dict[str, object]] = []
    for expected_id, raw in enumerate(rows, start=1):
        if type(raw) is not dict:
            _fail(FAIL_ACTUAL_ADMISSION, "prior stage evidence row differs")
        try:
            result.append(
                validate_actual_stage_evidence_v1(
                    raw,
                    expected_id,
                    ACTUAL_ORCHESTRATION_STAGE_REGISTRY[expected_id - 1][1],
                    source_commit,
                )
            )
        except Q05BDualSupervisorError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)
        if (
            raw.get("qualification_count") != 0
            or raw.get("qualification_mask") != 0
            or raw.get("candidate_receipt_hex") is not None
            or raw.get("final_receipt_hex") is not None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "prior stage authority is not closed")
    return result[0], result[1], result[2]


def actual_work_root_identity_v1(
    work_root: Path,
    layout: Mapping[str, object],
) -> dict[str, object]:
    if (
        not isinstance(work_root, Path)
        or not work_root.is_absolute()
        or type(layout) is not dict
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "admission work-root input differs")
    try:
        before = work_root.lstat()
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"admission work root is unavailable: {error}")
    if work_root.is_symlink() or not stat.S_ISDIR(before.st_mode):
        _fail(FAIL_ACTUAL_ADMISSION, "admission work root is not an anchored directory")
    value = {
        "schema_version": "hegel-phase3a-q05b-admission-work-root-identity/1",
        "absolute_path": work_root.as_posix(),
        "device": before.st_dev,
        "inode": before.st_ino,
        "nlink": before.st_nlink,
        "mode": stat.S_IMODE(before.st_mode),
        "layout_sha256": _admission_sha256_v1(layout),
    }
    after = work_root.lstat()
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        _fail(FAIL_ACTUAL_ADMISSION, "admission work root changed")
    return value


def _validate_work_root_identity_v1(value: object) -> dict[str, object]:
    try:
        return _ACTUAL_ADMISSION.validate_work_root_identity_v1(value)
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def actual_artifact_absence_evidence_v1(artifact_path: Path) -> dict[str, object]:
    if not isinstance(artifact_path, Path) or not artifact_path.is_absolute():
        _fail(FAIL_ACTUAL_ADMISSION, "admission artifact path differs")
    parent = artifact_path.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(parent, flags)
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"artifact parent cannot be anchored: {error}")
    try:
        before = os.fstat(descriptor)
        try:
            os.stat(artifact_path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail(FAIL_ACTUAL_ADMISSION, "artifact target exists at admission")
        after = os.fstat(descriptor)
        path_after = parent.lstat()
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
            or not stat.S_ISDIR(after.st_mode)
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "artifact parent identity changed")
        return {
            "schema_version": "hegel-phase3a-q05b-admission-artifact-absence/1",
            "artifact_path": artifact_path.as_posix(),
            "parent_path": parent.as_posix(),
            "parent_device": after.st_dev,
            "parent_inode": after.st_ino,
            "parent_nlink": after.st_nlink,
            "parent_mode": stat.S_IMODE(after.st_mode),
            "target_absent": True,
            "nofollow_dirfd_checked": True,
        }
    finally:
        os.close(descriptor)


def _validate_artifact_absence_evidence_v1(
    value: object,
    artifact_path: Path,
) -> dict[str, object]:
    try:
        return _ACTUAL_ADMISSION.validate_artifact_absence_evidence_v1(
            value, artifact_path.as_posix()
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _stage_root_rows_v1(
    stages: tuple[dict[str, object], dict[str, object], dict[str, object]],
) -> list[list[object]]:
    return [
        [index, stage["stage_evidence_root"]]
        for index, stage in enumerate(stages, start=1)
    ]


def _fresh_actor_source_identity_v1(
    source_evidence: Mapping[str, object],
    snapshot_evidence: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(source_evidence) is not dict
        or type(snapshot_evidence) is not dict
        or type(source_evidence.get("rows")) is not list
        or type(snapshot_evidence.get("file_rows")) is not list
        or len(source_evidence["rows"]) != len(snapshot_evidence["file_rows"])
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "fresh actor source identity input differs")
    snapshot_registry = [
        [row[0], row[6], row[7], row[10]]
        for row in snapshot_evidence["file_rows"]
        if type(row) is list and len(row) == 11
    ]
    if len(snapshot_registry) != len(snapshot_evidence["file_rows"]):
        _fail(FAIL_ACTUAL_ADMISSION, "fresh actor snapshot registry differs")
    return {
        "schema_version": "hegel-phase3a-q05b-fresh-actor-source-identity/1",
        "actor_id": source_evidence.get("actor_id"),
        "source_commit": source_evidence.get("commit"),
        "project_git_prefix": source_evidence.get("project_git_prefix"),
        "path_registry_sha256": source_evidence.get("path_registry_sha256"),
        "source_identity_sha256": source_evidence.get("source_identity_sha256"),
        "blob_count": len(source_evidence["rows"]),
        "snapshot_file_registry_sha256": sha256(
            _canonical_json_bytes(snapshot_registry)
        ).hexdigest(),
        "stage_1_source_evidence_sha256": _admission_sha256_v1(
            source_evidence
        ),
    }


def _fresh_cargo_material_identity_v1(
    cargo_evidence: Mapping[str, object],
) -> dict[str, object]:
    if (
        type(cargo_evidence) is not dict
        or type(cargo_evidence.get("locked_packages")) is not list
        or type(cargo_evidence.get("file_rows")) is not list
        or type(cargo_evidence.get("sealed_snapshot_identity")) is not dict
        or type(cargo_evidence.get("sealed_tree_identity")) is not dict
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "fresh Cargo material identity input differs")
    snapshot = cargo_evidence["sealed_snapshot_identity"]
    tree = cargo_evidence["sealed_tree_identity"]
    return {
        "schema_version": "hegel-phase3a-q05b-fresh-cargo-material-identity/1",
        "root_path": cargo_evidence.get("root_path"),
        "root_nlink": cargo_evidence.get("root_nlink"),
        "file_count": cargo_evidence.get("file_count"),
        "locked_registry_package_count": cargo_evidence.get(
            "locked_registry_package_count"
        ),
        "locked_packages_sha256": _admission_sha256_v1(
            cargo_evidence["locked_packages"]
        ),
        "file_registry_sha256": _admission_sha256_v1(
            cargo_evidence["file_rows"]
        ),
        "material_manifest_sha256": cargo_evidence.get("manifest_sha256"),
        "sealed_snapshot_manifest_sha256": snapshot.get("manifest_sha256"),
        "sealed_tree_manifest_sha256": tree.get("manifest_sha256"),
        "stage_2_cargo_evidence_sha256": _admission_sha256_v1(cargo_evidence),
    }


def _fresh_binary_identity_v1(
    binary_evidence: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "binary_path", "device", "inode", "nlink", "uid", "gid", "mode",
        "size", "mtime_ns", "ctime_ns", "sha256", "manifest_sha256",
    )
    if type(binary_evidence) is not dict or any(
        field not in binary_evidence for field in fields
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "fresh binary identity input differs")
    return {
        "schema_version": (
            "hegel-phase3a-q05b-fresh-prebuilt-rust-binary-identity/1"
        ),
        "binary_path": binary_evidence["binary_path"],
        "device": binary_evidence["device"],
        "inode": binary_evidence["inode"],
        "nlink": binary_evidence["nlink"],
        "uid": binary_evidence["uid"],
        "gid": binary_evidence["gid"],
        "mode": binary_evidence["mode"],
        "size": binary_evidence["size"],
        "mtime_ns": binary_evidence["mtime_ns"],
        "ctime_ns": binary_evidence["ctime_ns"],
        "sha256": binary_evidence["sha256"],
        "sealed_binary_manifest_sha256": binary_evidence["manifest_sha256"],
        "stage_3_binary_evidence_sha256": _admission_sha256_v1(
            binary_evidence
        ),
    }


def collect_fresh_runtime_evidence_set_v1(
    project_root: Path,
    source_commit: str,
    paths: Mapping[str, Path],
    source_evidence: Mapping[str, object],
    snapshot_evidence: Mapping[str, object],
    image_evidence: Mapping[str, object],
    cargo_evidence: Mapping[str, object],
    seccomp_evidence: Mapping[str, object],
    binary_evidence: Mapping[str, object],
    *,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Freshly replay every pre-launch source/image/runtime object in rows 5--8."""

    actor_ids = ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY")
    required_paths = {
        "python_snapshot", "rust_snapshot", "host_snapshot", "cargo_home", "binary"
    }
    if (
        not isinstance(project_root, Path)
        or not project_root.is_absolute()
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or type(paths) is not dict
        or not required_paths.issubset(paths)
        or any(not isinstance(paths[name], Path) for name in required_paths)
        or type(source_evidence) is not dict
        or set(source_evidence) != set(actor_ids)
        or type(snapshot_evidence) is not dict
        or set(snapshot_evidence) != set(actor_ids)
        or type(image_evidence) is not dict
        or set(image_evidence) != {"python", "rust"}
        or type(cargo_evidence) is not dict
        or type(seccomp_evidence) is not dict
        or set(seccomp_evidence) != {"runtime", "build"}
        or type(binary_evidence) is not dict
        or not callable(command_runner)
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "fresh runtime evidence collector inputs differ")

    current_images = {
        "python": local_pinned_image_evidence_v1(
            PYTHON_IMAGE, runner=command_runner
        ),
        "rust": local_pinned_image_evidence_v1(
            RUST_IMAGE, runner=command_runner
        ),
    }
    if current_images != image_evidence:
        _fail(FAIL_ACTUAL_ADMISSION, "pinned image changed before admission")
    image_rows: list[dict[str, object]] = []
    for label, reference in (("python", PYTHON_IMAGE), ("rust", RUST_IMAGE)):
        evidence = current_images[label]
        image_rows.append(
            {
                "label": label,
                "reference": reference,
                "evidence": evidence,
                "evidence_root": _ACTUAL_ADMISSION.fresh_runtime_evidence_object_root_v1(
                    "PINNED_IMAGE", label, evidence
                ),
            }
        )

    snapshot_keys = {
        "PYTHON_ENDPOINT": "python_snapshot",
        "RUST_ENDPOINT": "rust_snapshot",
        "TRUSTED_HOST_REPLAY": "host_snapshot",
    }
    actor_rows: list[dict[str, object]] = []
    for actor_id in actor_ids:
        current_source = actor_source_evidence_v1(
            project_root, source_commit, actor_id
        )
        current_snapshot = sealed_snapshot_path_evidence_v1(
            paths[snapshot_keys[actor_id]], ACTOR_SOURCE_ALLOWLISTS[actor_id]
        )
        if (
            current_source != source_evidence[actor_id]
            or current_snapshot != snapshot_evidence[actor_id]
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                f"actor source/snapshot changed before admission: {actor_id}",
            )
        source_identity = _fresh_actor_source_identity_v1(
            current_source, current_snapshot
        )
        actor_rows.append(
            {
                "actor_id": actor_id,
                "source_identity": source_identity,
                "source_identity_root": (
                    _ACTUAL_ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SOURCE", actor_id, source_identity
                    )
                ),
                "snapshot_evidence": current_snapshot,
                "snapshot_evidence_root": (
                    _ACTUAL_ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "ACTOR_SNAPSHOT", actor_id, current_snapshot
                    )
                ),
            }
        )

    cargo_rows = cargo_evidence.get("file_rows")
    if type(cargo_rows) is not list or not cargo_rows:
        _fail(FAIL_ACTUAL_ADMISSION, "sealed Cargo file registry differs")
    cargo_paths = tuple(
        row[0]
        for row in cargo_rows
        if type(row) is list and len(row) == 4 and type(row[0]) is str
    )
    if len(cargo_paths) != len(cargo_rows):
        _fail(FAIL_ACTUAL_ADMISSION, "sealed Cargo file row differs")
    cargo_snapshot = sealed_snapshot_identity_v1(paths["cargo_home"], cargo_paths)
    cargo_modes = {
        row[0]: 0o555 if row[1] == 0o100755 else 0o444
        for row in cargo_rows
    }
    cargo_tree = sealed_tree_identity_v1(
        paths["cargo_home"], cargo_paths, expected_file_modes=cargo_modes
    )
    if (
        cargo_snapshot != cargo_evidence.get("sealed_snapshot_identity")
        or cargo_tree != cargo_evidence.get("sealed_tree_identity")
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "sealed Cargo changed before admission")

    host_source = source_evidence["TRUSTED_HOST_REPLAY"]
    if type(host_source) is not dict:
        _fail(FAIL_ACTUAL_ADMISSION, "trusted-host source evidence differs")
    host_blob_sha = {
        row[0]: row[4]
        for row in host_source.get("blob_preimage_rows", [])
        if type(row) is list and len(row) == 6
    }
    seccomp_rows: list[dict[str, object]] = []
    for label, relative in (
        ("runtime", RUNTIME_SECCOMP_RELATIVE_PATH),
        ("build", BUILD_SECCOMP_RELATIVE_PATH),
    ):
        expected_payload_sha = host_blob_sha.get(relative)
        if type(expected_payload_sha) is not str:
            _fail(
                FAIL_ACTUAL_ADMISSION,
                f"trusted-host source omits sealed {label} seccomp",
            )
        current = sealed_policy_file_evidence_v1(
            paths["host_snapshot"] / relative,
            relative,
            expected_sha256=expected_payload_sha,
        )
        if current != seccomp_evidence[label]:
            _fail(
                FAIL_ACTUAL_ADMISSION,
                f"sealed {label} seccomp changed before admission",
            )
        seccomp_rows.append(
            {
                "label": label,
                "relative_path": relative,
                "evidence": current,
                "evidence_root": (
                    _ACTUAL_ADMISSION.fresh_runtime_evidence_object_root_v1(
                        "SECCOMP_POLICY", label, current
                    )
                ),
            }
        )

    current_binary = replay_sealed_prebuilt_binary_v1(
        paths["binary"], binary_evidence
    )
    cargo_identity = _fresh_cargo_material_identity_v1(cargo_evidence)
    binary_identity = _fresh_binary_identity_v1(current_binary)
    return build_fresh_runtime_evidence_set_v1(
        source_commit,
        image_rows,
        actor_rows,
        cargo_identity,
        cargo_snapshot,
        cargo_tree,
        seccomp_rows,
        binary_identity,
    )


def _actual_precondition_preimages_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: Path,
    stages: tuple[dict[str, object], dict[str, object], dict[str, object]],
    fresh_status: GitSourceStatusV1,
    git_source_transcript: Mapping[str, object],
    artifact_absence: Mapping[str, object],
    fresh_runtime_evidence: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    config = _validate_commit_a_actual_config_bytes_v1(commit_a_config_bytes)
    stage1, stage2, stage3 = stages
    evidence1 = stage1["evidence"]
    evidence2 = stage2["evidence"]
    evidence3 = stage3["evidence"]
    if any(type(value) is not dict for value in (evidence1, evidence2, evidence3)):
        _fail(FAIL_ACTUAL_ADMISSION, "prior stage evidence payload differs")
    config_hex = evidence1.get("config_hex")
    if (
        type(config_hex) is not str
        or config_hex != commit_a_config_bytes.hex()
        or evidence1.get("config_sha256") != sha256(commit_a_config_bytes).hexdigest()
        or evidence1.get("fixed_artifact_path") != artifact_path.as_posix()
        or evidence1.get("q1_authority") != config["dry_run_authority"]
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "stage 1 does not bind Commit-A admission inputs")
    if (
        type(fresh_status) is not GitSourceStatusV1
        or fresh_status.head != source_commit
        or fresh_status.clean is not True
        or fresh_status.porcelain_line_count != 0
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "fresh Commit-A source status differs")
    try:
        _ACTUAL_ADMISSION.validate_git_source_transcript_v1(
            git_source_transcript, source_commit
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    _validate_artifact_absence_evidence_v1(artifact_absence, artifact_path)
    roots = _stage_root_rows_v1(stages)
    source_evidence = evidence1.get("source_evidence")
    snapshot_evidence = evidence2.get("snapshot_evidence")
    image_evidence = evidence1.get("image_evidence")
    if (
        type(source_evidence) is not dict
        or type(snapshot_evidence) is not dict
        or type(image_evidence) is not dict
        or set(source_evidence) != {"PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"}
        or set(snapshot_evidence) != set(source_evidence)
        or set(image_evidence) != {"python", "rust"}
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "source/snapshot/image stage registry differs")
    host_source = source_evidence["TRUSTED_HOST_REPLAY"]
    if type(host_source) is not dict:
        _fail(FAIL_ACTUAL_ADMISSION, "trusted-host source evidence differs")
    config_blob_rows = [
        row
        for row in host_source.get("blob_preimage_rows", [])
        if type(row) is list and row and row[0] == CONFIG_RELATIVE_PATH
    ]
    if (
        len(config_blob_rows) != 1
        or len(config_blob_rows[0]) != 6
        or type(config_blob_rows[0][3]) is not int
        or config_blob_rows[0][3] != len(commit_a_config_bytes)
        or config_blob_rows[0][4] != sha256(commit_a_config_bytes).hexdigest()
        or config_blob_rows[0][5] != commit_a_config_bytes.hex()
    ):
        _fail(
            FAIL_ACTUAL_ADMISSION,
            "Commit-A Git config blob differs from runtime config bytes",
        )
    try:
        fresh_set = _ACTUAL_ADMISSION.validate_fresh_runtime_evidence_set_v1(
            fresh_runtime_evidence, source_commit
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    fresh_images = fresh_set["image_rows"]
    fresh_actors = fresh_set["actor_rows"]
    fresh_cargo = fresh_set["cargo"]
    fresh_seccomp = fresh_set["seccomp_rows"]
    fresh_binary = fresh_set["binary"]
    if (
        [row["evidence"] for row in fresh_images]
        != [image_evidence["python"], image_evidence["rust"]]
        or [row["snapshot_evidence"] for row in fresh_actors]
        != [
            snapshot_evidence[actor_id]
            for actor_id in (
                "PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"
            )
        ]
        or any(
            row["source_identity"]["stage_1_source_evidence_sha256"]
            != _admission_sha256_v1(source_evidence[row["actor_id"]])
            for row in fresh_actors
        )
        or fresh_cargo["material_identity"]["stage_2_cargo_evidence_sha256"]
        != _admission_sha256_v1(evidence2.get("cargo_evidence"))
        or [row["evidence"] for row in fresh_seccomp]
        != [
            evidence2.get("seccomp_evidence", {}).get("runtime"),
            evidence2.get("seccomp_evidence", {}).get("build"),
        ]
        or fresh_binary["identity"]["stage_3_binary_evidence_sha256"]
        != _admission_sha256_v1(evidence3.get("binary"))
    ):
        _fail(
            FAIL_ACTUAL_ADMISSION,
            "fresh runtime evidence differs from stages 1 through 3",
        )
    offline_identity = {
        "schema_version": "hegel-phase3a-q05b-fresh-offline-build-identity/1",
        "stage_3_root": roots[2][1],
        "rust_test_transcript_sha256": _admission_sha256_v1(
            evidence3.get("rust_test")
        ),
        "rust_release_build_transcript_sha256": _admission_sha256_v1(
            evidence3.get("rust_release_build")
        ),
        "rust_snapshot_manifest_sha256": fresh_actors[1][
            "snapshot_evidence"
        ]["manifest_sha256"],
        "cargo_snapshot_manifest_sha256": fresh_cargo[
            "snapshot_evidence"
        ]["manifest_sha256"],
        "cargo_tree_manifest_sha256": fresh_cargo["tree_evidence"][
            "manifest_sha256"
        ],
        "binary_manifest_sha256": fresh_binary["identity"][
            "sealed_binary_manifest_sha256"
        ],
        "stage_3_evidence_sha256": _admission_sha256_v1(evidence3),
    }
    try:
        command_mount_resource_policy_root = (
            _ACTUAL_ADMISSION.command_mount_resource_policy_root_v1(
                commit_a_config_bytes
            )
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    return (
        {
            "stage_1_root": roots[0][1],
            "requested_source_commit": source_commit,
            "fresh_head_commit": fresh_status.head,
            "clean": True,
            "porcelain_line_count": 0,
            "git_source_transcript": dict(git_source_transcript),
        },
        {
            "stage_1_root": roots[0][1],
            "config_relative_path": CONFIG_RELATIVE_PATH,
            "commit_a_config_hex": commit_a_config_bytes.hex(),
            "runtime_loaded_config_hex": config_hex,
            "config_length": len(commit_a_config_bytes),
            "config_sha256": sha256(commit_a_config_bytes).hexdigest(),
        },
        {
            "stage_1_root": roots[0][1],
            "engineering_status": COMMIT_A_ACTUAL_ENGINEERING_STATUS,
            "actual_preconditions": config["actual_preconditions"],
            "entrypoint": "run_actual_v1",
            "entrypoint_implemented": True,
            "conditional_single_attempt_policy": "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION",
        },
        {
            "stage_1_root": roots[0][1],
            "stage_3_root": roots[2][1],
            "artifact_absence_evidence": dict(artifact_absence),
        },
        {
            "stage_1_root": roots[0][1],
            "image_rows": fresh_images,
            "fresh_runtime_evidence_root": fresh_set[
                "fresh_runtime_evidence_root"
            ],
        },
        {
            "stage_1_root": roots[0][1],
            "stage_2_root": roots[1][1],
            "actor_rows": fresh_actors,
            "fresh_runtime_evidence_root": fresh_set[
                "fresh_runtime_evidence_root"
            ],
        },
        {
            "stage_2_root": roots[1][1],
            "stage_3_root": roots[2][1],
            "cargo_lock_sha256": evidence2.get("cargo_lock_sha256"),
            "cargo_material_identity": fresh_cargo["material_identity"],
            "cargo_material_identity_root": fresh_cargo[
                "material_identity_root"
            ],
            "cargo_snapshot_evidence": fresh_cargo["snapshot_evidence"],
            "cargo_snapshot_evidence_root": fresh_cargo[
                "snapshot_evidence_root"
            ],
            "cargo_tree_evidence": fresh_cargo["tree_evidence"],
            "cargo_tree_evidence_root": fresh_cargo["tree_evidence_root"],
            "offline_build_identity": offline_identity,
            "offline_build_identity_root": (
                _ACTUAL_ADMISSION.fresh_runtime_evidence_object_root_v1(
                    "OFFLINE_BUILD_TRANSCRIPT", "rust", offline_identity
                )
            ),
            "fresh_runtime_evidence_root": fresh_set[
                "fresh_runtime_evidence_root"
            ],
        },
        {
            "stage_2_root": roots[1][1],
            "stage_3_root": roots[2][1],
            "seccomp_rows": fresh_seccomp,
            "binary_identity": fresh_binary["identity"],
            "binary_identity_root": fresh_binary["identity_root"],
            "fresh_runtime_evidence_root": fresh_set[
                "fresh_runtime_evidence_root"
            ],
        },
        {
            "stage_1_root": roots[0][1],
            "planned_command_registry_sha256": _admission_sha256_v1(evidence1.get("planned_commands")),
            "command_mount_resource_policy_sha256": command_mount_resource_policy_root,
            "prelaunch_policy_bound": True,
        },
        {
            "stage_1_root": roots[0][1],
            "qualification_authority": dict(ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY),
            "closed_q1_authority": {**ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY, "formal_output_roots": [None] * 8},
        },
        {
            "prior_stage_root_rows": roots,
            "policy_name": "FRESH_SOURCE_IMAGE_RUNTIME_SNAPSHOT_REPLAY_BEFORE_PREDICATE19",
            "policy_bound_at_admission": True,
            "fulfilled_at_admission": False,
        },
        {
            "stage_1_root": roots[0][1],
            "artifact_path": artifact_path.as_posix(),
            "policy_name": "DIRFD_NOFOLLOW_FSYNC_LINK_NOREPLACE_UNLINK_FSYNC",
            "policy_bound_at_admission": True,
            "fulfilled_at_admission": False,
        },
    )


def build_actual_precondition_bundle_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: Path,
    work_root_identity: Mapping[str, object],
    prior_stage_evidence_rows: tuple[dict[str, object], ...],
    fresh_status: GitSourceStatusV1,
    git_source_transcript: Mapping[str, object],
    artifact_absence: Mapping[str, object],
    fresh_runtime_evidence: Mapping[str, object],
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        _fail(FAIL_ACTUAL_ADMISSION, "admission source commit differs")
    if not isinstance(artifact_path, Path) or not artifact_path.is_absolute():
        _fail(FAIL_ACTUAL_ADMISSION, "admission artifact path differs")
    work = _validate_work_root_identity_v1(work_root_identity)
    stages = _validate_prior_stage_rows_v1(prior_stage_evidence_rows, source_commit)
    preimages = _actual_precondition_preimages_v1(
        source_commit,
        commit_a_config_bytes,
        artifact_path,
        stages,
        fresh_status,
        git_source_transcript,
        artifact_absence,
        fresh_runtime_evidence,
    )
    try:
        return _ACTUAL_ADMISSION.build_actual_precondition_bundle_v1(
            source_commit,
            commit_a_config_bytes,
            artifact_path.as_posix(),
            work,
            _stage_root_rows_v1(stages),
            preimages,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def decode_actual_precondition_bundle_v1(
    payload: bytes,
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: Path,
    work_root_identity: Mapping[str, object],
    prior_stage_evidence_rows: tuple[dict[str, object], ...],
    fresh_status: GitSourceStatusV1,
    git_source_transcript: Mapping[str, object],
    artifact_absence: Mapping[str, object],
    fresh_runtime_evidence: Mapping[str, object],
) -> dict[str, object]:
    expected = build_actual_precondition_bundle_v1(
        source_commit,
        commit_a_config_bytes,
        artifact_path,
        work_root_identity,
        prior_stage_evidence_rows,
        fresh_status,
        git_source_transcript,
        artifact_absence,
        fresh_runtime_evidence,
    )
    try:
        return _ACTUAL_ADMISSION.decode_actual_precondition_bundle_v1(
            payload,
            source_commit,
            commit_a_config_bytes,
            artifact_path.as_posix(),
            expected["work_root_identity"],
            expected["prior_stage_root_rows"],
            [row["preimage"] for row in expected["ordered_precondition_rows"]],
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def build_actual_admission_decision_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: Path,
    attempt_nonce: bytes,
    precondition_bundle: Mapping[str, object],
) -> dict[str, object]:
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(artifact_path, Path)
        or not artifact_path.is_absolute()
        or type(attempt_nonce) is not bytes
        or len(attempt_nonce) != 32
        or type(precondition_bundle) is not dict
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "admission decision inputs differ")
    _validate_commit_a_actual_config_bytes_v1(commit_a_config_bytes)
    try:
        return _ACTUAL_ADMISSION.build_actual_admission_decision_v1(
            source_commit,
            commit_a_config_bytes,
            artifact_path.as_posix(),
            attempt_nonce,
            precondition_bundle,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def decode_actual_admission_decision_v1(
    payload: bytes,
    commit_a_config_bytes: bytes,
    expected_source_commit: str,
    expected_artifact_path: Path,
    expected_precondition_bundle: Mapping[str, object],
) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 16 * 1024 * 1024:
        _fail(FAIL_ACTUAL_ADMISSION, "admission decoder inputs differ")
    try:
        return _ACTUAL_ADMISSION.decode_actual_admission_decision_v1(
            payload,
            commit_a_config_bytes,
            expected_source_commit,
            expected_artifact_path.as_posix(),
            expected_precondition_bundle,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def canonical_actual_admission_decision_bytes_v1(
    value: object,
    commit_a_config_bytes: bytes,
    expected_source_commit: str,
    expected_artifact_path: Path,
    expected_precondition_bundle: Mapping[str, object],
) -> bytes:
    try:
        return _ACTUAL_ADMISSION.canonical_actual_admission_decision_bytes_v1(
            value,
            commit_a_config_bytes,
            expected_source_commit,
            expected_artifact_path.as_posix(),
            expected_precondition_bundle,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def build_stage3_to4_admission_boundary_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: Path,
    precondition_bundle: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    try:
        return _ACTUAL_ADMISSION.build_stage3_to4_admission_boundary_v1(
            source_commit,
            commit_a_config_bytes,
            artifact_path.as_posix(),
            precondition_bundle,
            decision,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def decode_stage3_to4_admission_boundary_v1(
    payload: bytes,
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: Path,
    work_root_identity: Mapping[str, object],
    prior_stage_evidence_rows: tuple[dict[str, object], ...],
    fresh_status: GitSourceStatusV1,
    git_source_transcript: Mapping[str, object],
    artifact_absence: Mapping[str, object],
    fresh_runtime_evidence: Mapping[str, object],
) -> dict[str, object]:
    try:
        value = _strict_json_object_v1(payload, "stage3-to4 admission boundary")
        bundle_payload = bytes.fromhex(value["precondition_bundle_hex"])
        decision_payload = bytes.fromhex(value["decision_hex"])
    except (Q05BDualSupervisorError, KeyError, TypeError, ValueError) as error:
        detail = getattr(error, "detail", "admission boundary payload encoding differs")
        _fail(FAIL_ACTUAL_ADMISSION, detail)
    bundle = decode_actual_precondition_bundle_v1(
        bundle_payload,
        source_commit,
        commit_a_config_bytes,
        artifact_path,
        work_root_identity,
        prior_stage_evidence_rows,
        fresh_status,
        git_source_transcript,
        artifact_absence,
        fresh_runtime_evidence,
    )
    decision = decode_actual_admission_decision_v1(
        decision_payload, commit_a_config_bytes, source_commit, artifact_path, bundle
    )
    try:
        return _ACTUAL_ADMISSION.decode_stage3_to4_admission_boundary_v1(
            payload,
            source_commit,
            commit_a_config_bytes,
            artifact_path.as_posix(),
            bundle,
            decision,
        )
    except (KeyError, TypeError, ValueError, _ACTUAL_ADMISSION.Q05BActualAdmissionError) as error:
        detail = getattr(error, "detail", "admission boundary payload encoding differs")
        _fail(FAIL_ACTUAL_ADMISSION, detail)


ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION
)
ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION
)
ACTUAL_ADMISSION_CONSUMED_MARKER_SCHEMA_VERSION: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_CONSUMED_MARKER_SCHEMA_VERSION
)
ACTUAL_ADMISSION_SPENDING_INTENT_SCHEMA_VERSION: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_SPENDING_INTENT_SCHEMA_VERSION
)
ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_SCHEMA_VERSION: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_SCHEMA_VERSION
)
ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN
)
ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN
)
ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN
)
ACTUAL_ADMISSION_SPENDING_INTENT_ROOT_DOMAIN: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_SPENDING_INTENT_ROOT_DOMAIN
)
ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
)
ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE: Final = (
    _ACTUAL_ADMISSION.ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE
)


def _admission_marker_names_v1(attempt_id: str) -> tuple[str, str, str, str]:
    try:
        return _ACTUAL_ADMISSION.actual_admission_marker_names_v1(attempt_id)
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _open_anchored_admission_work_root_v1(
    work_root: Path,
    expected_identity: Mapping[str, object],
) -> int:
    work = _validate_work_root_identity_v1(expected_identity)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(work_root, flags)
        anchored = os.fstat(descriptor)
        path_status = work_root.lstat()
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"admission work-root anchor failed: {error}")
    if (
        not stat.S_ISDIR(anchored.st_mode)
        or stat.S_IMODE(anchored.st_mode) != 0o700
        or (anchored.st_dev, anchored.st_ino)
        != (work["device"], work["inode"])
        or anchored.st_nlink != work["nlink"]
        or (path_status.st_dev, path_status.st_ino)
        != (anchored.st_dev, anchored.st_ino)
        or path_status.st_nlink != work["nlink"]
    ):
        os.close(descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, "admission work-root anchor identity differs")
    return descriptor


def _read_marker_at_v1(
    root_descriptor: int,
    name: str,
    maximum: int = 16 * 1024 * 1024,
) -> tuple[bytes, os.stat_result]:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"admission marker replay failed: {error}")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size < 1
            or before.st_size > maximum
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission marker file identity differs")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > maximum:
                _fail(FAIL_ACTUAL_ADMISSION, "admission marker exceeds bound")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission marker changed during replay")
        return b"".join(blocks), after
    finally:
        os.close(descriptor)


def issue_actual_admission_marker_v1(
    work_root: Path,
    work_root_identity: Mapping[str, object],
    boundary: Mapping[str, object],
) -> tuple[dict[str, object], int, int]:
    if (
        type(boundary) is not dict
        or type(boundary.get("attempt_id")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", boundary["attempt_id"]) is None
        or type(boundary.get("boundary_root")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", boundary["boundary_root"]) is None
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "admission issued boundary differs")
    payload = _canonical_json_bytes(boundary)
    issued_name, spending_name, consumed_name, failed_name = (
        _admission_marker_names_v1(boundary["attempt_id"])
    )
    root_descriptor = _open_anchored_admission_work_root_v1(
        work_root, work_root_identity
    )
    marker_writer: int | None = None
    marker_descriptor: int | None = None
    try:
        for name in (issued_name, spending_name, consumed_name, failed_name):
            try:
                os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                _fail(FAIL_ACTUAL_ADMISSION, "admission marker already exists")
        marker_writer = os.open(
            issued_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(marker_writer, payload[offset:])
        os.fsync(marker_writer)
        os.fchmod(marker_writer, 0o444)
        os.fsync(marker_writer)
        issued_status = os.fstat(marker_writer)
        if (
            not stat.S_ISREG(issued_status.st_mode)
            or issued_status.st_nlink != 1
            or stat.S_IMODE(issued_status.st_mode) != 0o444
            or issued_status.st_size != len(payload)
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "issued admission marker identity differs")
        os.fsync(root_descriptor)
        marker_descriptor = os.open(
            issued_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        held_status = os.fstat(marker_descriptor)
        if (held_status.st_dev, held_status.st_ino) != (
            issued_status.st_dev,
            issued_status.st_ino,
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "issued held descriptor identity differs")
        os.close(marker_writer)
        marker_writer = None
    except BaseException:
        if marker_writer is not None:
            os.close(marker_writer)
        if marker_descriptor is not None:
            os.close(marker_descriptor)
        os.close(root_descriptor)
        raise
    assert marker_writer is None and marker_descriptor is not None
    try:
        replayed, replay_status = _read_held_admission_marker_v1(
            marker_descriptor,
            1,
        )
        issued_path_status = os.stat(
            issued_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except BaseException:
        os.close(marker_descriptor)
        os.close(root_descriptor)
        raise
    if (
        replayed != payload
        or (replay_status.st_dev, replay_status.st_ino)
        != (issued_status.st_dev, issued_status.st_ino)
        or (issued_path_status.st_dev, issued_path_status.st_ino)
        != (replay_status.st_dev, replay_status.st_ino)
    ):
        os.close(marker_descriptor)
        os.close(root_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, "issued admission marker replay differs")
    anchored = os.fstat(root_descriptor)
    try:
        marker = _ACTUAL_ADMISSION.build_actual_admission_issued_marker_evidence_v1(
            boundary["attempt_id"],
            boundary["boundary_root"],
            payload,
            file_device=replay_status.st_dev,
            file_inode=replay_status.st_ino,
            file_nlink=replay_status.st_nlink,
            file_mode=stat.S_IMODE(replay_status.st_mode),
            work_root_device=anchored.st_dev,
            work_root_inode=anchored.st_ino,
            work_root_mode=stat.S_IMODE(anchored.st_mode),
        )
        record = _ACTUAL_ADMISSION.build_actual_admission_issue_record_v1(
            boundary, marker
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        os.close(marker_descriptor)
        os.close(root_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    return record, root_descriptor, marker_descriptor


def validate_actual_admission_issue_record_v1(
    value: object,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        return _ACTUAL_ADMISSION.validate_actual_admission_issue_record_v1(value)
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def validate_actual_admission_issued_marker_evidence_v1(
    value: object,
    boundary_payload: bytes,
) -> dict[str, object]:
    try:
        return _ACTUAL_ADMISSION.validate_actual_admission_issued_marker_evidence_v1(
            value, boundary_payload
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def validate_actual_admission_spending_intent_v1(
    value: object,
    issue_record: Mapping[str, object],
) -> dict[str, object]:
    try:
        return _ACTUAL_ADMISSION.validate_actual_admission_spending_intent_v1(
            value, issue_record
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def consume_actual_admission_marker_v1(
    root_descriptor: int,
    issued_descriptor: int,
    issue_record: Mapping[str, object],
) -> tuple[dict[str, object], int, int]:
    record, boundary = validate_actual_admission_issue_record_v1(issue_record)
    marker = record["issued_marker_evidence"]
    issued_name = marker["issued_relative_path"]
    spending_name = marker["spending_relative_path"]
    consumed_name = marker["consumed_relative_path"]
    failed_name = marker["failed_relative_path"]
    payload = bytes.fromhex(record["pure_boundary_hex"])
    try:
        spending = _ACTUAL_ADMISSION.build_actual_admission_spending_intent_v1(
            record
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)
    spending_payload = _canonical_json_bytes(spending)
    spending_writer: int | None = None
    spending_descriptor: int | None = None
    try:
        spending_writer = os.open(
            spending_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        offset = 0
        while offset < len(spending_payload):
            written = os.write(spending_writer, spending_payload[offset:])
            if written <= 0:
                _fail(FAIL_ACTUAL_ADMISSION, "spending intent write made no progress")
            offset += written
        os.fsync(spending_writer)
        os.fchmod(spending_writer, 0o444)
        os.fsync(spending_writer)
        spending_status = os.fstat(spending_writer)
        if (
            not stat.S_ISREG(spending_status.st_mode)
            or spending_status.st_nlink != 1
            or stat.S_IMODE(spending_status.st_mode) != 0o444
            or spending_status.st_size != len(spending_payload)
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "spending intent identity differs")
        os.fsync(root_descriptor)
        spending_descriptor = os.open(
            spending_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
        held_spending_status = os.fstat(spending_descriptor)
        if (held_spending_status.st_dev, held_spending_status.st_ino) != (
            spending_status.st_dev,
            spending_status.st_ino,
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "spending held descriptor identity differs")
        os.close(spending_writer)
        spending_writer = None
    except BaseException as error:
        if spending_writer is not None:
            os.close(spending_writer)
        if spending_descriptor is not None:
            os.close(spending_descriptor)
        if isinstance(error, OSError):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                f"admission spending intent failed: {error}",
            )
        raise
    assert spending_writer is None and spending_descriptor is not None
    try:
        replayed_spending, replayed_spending_status = (
            _read_held_admission_marker_v1(
                spending_descriptor,
                1,
            )
        )
        spending_path_status = os.stat(
            spending_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except BaseException:
        os.close(spending_descriptor)
        raise
    if (
        replayed_spending != spending_payload
        or (replayed_spending_status.st_dev, replayed_spending_status.st_ino)
        != (spending_status.st_dev, spending_status.st_ino)
        or (spending_path_status.st_dev, spending_path_status.st_ino)
        != (replayed_spending_status.st_dev, replayed_spending_status.st_ino)
    ):
        os.close(spending_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, "spending intent replay differs")
    try:
        replayed, issued_status = _read_held_admission_marker_v1(
            issued_descriptor,
            1,
        )
        issued_path_status = os.stat(
            issued_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except BaseException:
        os.close(spending_descriptor)
        raise
    if (
        replayed != payload
        or issued_status.st_dev != marker["file_device"]
        or issued_status.st_ino != marker["file_inode"]
        or issued_status.st_nlink != 1
        or (issued_path_status.st_dev, issued_path_status.st_ino)
        != (issued_status.st_dev, issued_status.st_ino)
    ):
        os.close(spending_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, "issued marker changed before consume")
    try:
        os.link(
            issued_name,
            consumed_name,
            src_dir_fd=root_descriptor,
            dst_dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        os.fsync(root_descriptor)
        linked = os.stat(
            consumed_name, dir_fd=root_descriptor, follow_symlinks=False
        )
        if (
            (linked.st_dev, linked.st_ino)
            != (issued_status.st_dev, issued_status.st_ino)
            or linked.st_nlink != 2
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "consumed marker hardlink identity differs")
    except OSError as error:
        os.close(spending_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, f"admission marker consume failed: {error}")
    try:
        issued_payload, issued_final_status = _read_marker_at_v1(
            root_descriptor, issued_name
        )
        consumed_payload, consumed_status = _read_marker_at_v1(
            root_descriptor, consumed_name
        )
    except BaseException:
        os.close(spending_descriptor)
        raise
    anchored = os.fstat(root_descriptor)
    if (
        issued_payload != payload
        or consumed_payload != payload
        or (issued_final_status.st_dev, issued_final_status.st_ino)
        != (issued_status.st_dev, issued_status.st_ino)
        or (consumed_status.st_dev, consumed_status.st_ino)
        != (issued_status.st_dev, issued_status.st_ino)
        or issued_final_status.st_nlink != 2
        or consumed_status.st_nlink != 2
    ):
        os.close(spending_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, "consumed admission marker replay differs")
    try:
        consumed_descriptor = os.open(
            consumed_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_descriptor,
        )
    except OSError as error:
        os.close(spending_descriptor)
        _fail(FAIL_ACTUAL_ADMISSION, f"held admission marker open failed: {error}")
    try:
        validated = (
            _ACTUAL_ADMISSION.build_actual_admission_consumed_marker_evidence_v1(
                record,
                spending,
                spending_file_device=replayed_spending_status.st_dev,
                spending_file_inode=replayed_spending_status.st_ino,
                spending_file_nlink=replayed_spending_status.st_nlink,
                spending_file_mode=stat.S_IMODE(
                    replayed_spending_status.st_mode
                ),
                file_device=consumed_status.st_dev,
                file_inode=consumed_status.st_ino,
                file_nlink=consumed_status.st_nlink,
                file_mode=stat.S_IMODE(consumed_status.st_mode),
                work_root_device=anchored.st_dev,
                work_root_inode=anchored.st_ino,
                work_root_mode=stat.S_IMODE(anchored.st_mode),
            )
        )
    except BaseException:
        os.close(spending_descriptor)
        os.close(consumed_descriptor)
        raise
    return validated, spending_descriptor, consumed_descriptor


def validate_actual_admission_consumed_marker_evidence_v1(
    value: object,
    issue_record: Mapping[str, object],
) -> dict[str, object]:
    try:
        return (
            _ACTUAL_ADMISSION.validate_actual_admission_consumed_marker_evidence_v1(
                value, issue_record
            )
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _revalidate_anchored_admission_work_root_v1(
    root_descriptor: int,
    work_root: Path,
    expected_identity: Mapping[str, object],
) -> dict[str, object]:
    work = _validate_work_root_identity_v1(expected_identity)
    try:
        anchored = os.fstat(root_descriptor)
        observed = work_root.lstat()
    except OSError as error:
        _fail(
            FAIL_ACTUAL_ADMISSION,
            f"admission work-root revalidation failed: {error}",
        )
    if (
        not stat.S_ISDIR(anchored.st_mode)
        or stat.S_IMODE(anchored.st_mode) != 0o700
        or (anchored.st_dev, anchored.st_ino) != (work["device"], work["inode"])
        or anchored.st_nlink != work["nlink"]
        or (observed.st_dev, observed.st_ino)
        != (anchored.st_dev, anchored.st_ino)
        or observed.st_nlink != work["nlink"]
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "admission work-root path changed after issue")
    return {
        "schema_version": "hegel-phase3a-q05b-admission-work-root-replay/1",
        "absolute_path": work_root.as_posix(),
        "device": anchored.st_dev,
        "inode": anchored.st_ino,
        "nlink": anchored.st_nlink,
        "mode": stat.S_IMODE(anchored.st_mode),
        "path_matches_anchored_descriptor": True,
    }


def _read_held_admission_marker_v1(
    descriptor: int,
    expected_nlink: int,
    maximum: int = 16 * 1024 * 1024,
) -> tuple[bytes, os.stat_result]:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(expected_nlink) is not int
        or expected_nlink not in (1, 2)
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "held admission marker descriptor differs")
    try:
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > maximum:
                _fail(FAIL_ACTUAL_ADMISSION, "held admission marker exceeds bound")
        after = os.fstat(descriptor)
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"held admission marker replay failed: {error}")
    if (
        not stat.S_ISREG(before.st_mode)
        or access_mode != os.O_RDONLY
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != expected_nlink
        or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "held admission marker identity differs")
    return b"".join(blocks), after


def replay_live_actual_admission_markers_v1(
    root_descriptor: int,
    issued_descriptor: int,
    spending_descriptor: int,
    consumed_descriptor: int,
    work_root: Path,
    work_root_identity: Mapping[str, object],
    issue_record: Mapping[str, object],
    consumed_evidence: Mapping[str, object],
    checkpoint: str,
) -> dict[str, object]:
    if type(checkpoint) is not str or re.fullmatch(r"[A-Z0-9_]+", checkpoint) is None:
        _fail(FAIL_ACTUAL_ADMISSION, "admission live checkpoint differs")
    record, boundary = validate_actual_admission_issue_record_v1(issue_record)
    consumed = validate_actual_admission_consumed_marker_evidence_v1(
        consumed_evidence,
        record,
    )
    work = _validate_work_root_identity_v1(work_root_identity)
    marker = record["issued_marker_evidence"]
    payload = bytes.fromhex(record["pure_boundary_hex"])
    issued_payload, issued_status = _read_held_admission_marker_v1(
        issued_descriptor,
        2,
    )
    spending_payload, spending_status = _read_held_admission_marker_v1(
        spending_descriptor,
        1,
    )
    consumed_payload, consumed_status = _read_held_admission_marker_v1(
        consumed_descriptor,
        2,
    )
    try:
        root_status = os.fstat(root_descriptor)
        path_status = work_root.lstat()
        issued_path_status = os.stat(
            marker["issued_relative_path"],
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        spending_path_status = os.stat(
            marker["spending_relative_path"],
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        consumed_path_status = os.stat(
            marker["consumed_relative_path"],
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        try:
            os.stat(
                marker["failed_relative_path"],
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            _fail(FAIL_ACTUAL_ADMISSION, "failed admission marker unexpectedly exists")
    except OSError as error:
        _fail(FAIL_ACTUAL_ADMISSION, f"live admission marker path replay failed: {error}")
    expected_inode = (consumed["file_device"], consumed["file_inode"])
    if (
        issued_payload != payload
        or consumed_payload != payload
        or spending_payload.hex() != consumed["spending_intent_hex"]
        or (issued_status.st_dev, issued_status.st_ino) != expected_inode
        or (consumed_status.st_dev, consumed_status.st_ino) != expected_inode
        or (issued_path_status.st_dev, issued_path_status.st_ino) != expected_inode
        or (consumed_path_status.st_dev, consumed_path_status.st_ino)
        != expected_inode
        or issued_status.st_nlink != 2
        or consumed_status.st_nlink != 2
        or spending_status.st_dev != consumed["spending_file_device"]
        or spending_status.st_ino != consumed["spending_file_inode"]
        or spending_status.st_nlink != 1
        or stat.S_IMODE(spending_status.st_mode) != 0o444
        or (spending_path_status.st_dev, spending_path_status.st_ino)
        != (spending_status.st_dev, spending_status.st_ino)
        or not stat.S_ISDIR(root_status.st_mode)
        or stat.S_IMODE(root_status.st_mode) != 0o700
        or (root_status.st_dev, root_status.st_ino)
        != (work["device"], work["inode"])
        or root_status.st_nlink != work["nlink"]
        or (path_status.st_dev, path_status.st_ino)
        != (root_status.st_dev, root_status.st_ino)
        or stat.S_IMODE(path_status.st_mode) != 0o700
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "live admission marker authority differs")
    try:
        return _ACTUAL_ADMISSION.build_actual_admission_live_marker_replay_v1(
            checkpoint,
            record,
            consumed,
            work_root_device=root_status.st_dev,
            work_root_inode=root_status.st_ino,
            work_root_nlink=root_status.st_nlink,
            work_root_mode=stat.S_IMODE(root_status.st_mode),
            issued_file_device=issued_status.st_dev,
            issued_file_inode=issued_status.st_ino,
            issued_file_nlink=issued_status.st_nlink,
            consumed_file_device=consumed_status.st_dev,
            consumed_file_inode=consumed_status.st_ino,
            consumed_file_nlink=consumed_status.st_nlink,
            spending_file_device=spending_status.st_dev,
            spending_file_inode=spending_status.st_ino,
            spending_file_nlink=spending_status.st_nlink,
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def validate_actual_admission_live_marker_replay_surface_v1(
    value: object,
    expected_checkpoint: str,
) -> dict[str, object]:
    try:
        return (
            _ACTUAL_ADMISSION.validate_actual_admission_live_marker_replay_surface_v1(
                value, expected_checkpoint
            )
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


def _same_live_admission_authority_v1(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    ignored = {"checkpoint", "live_marker_replay_root"}
    return {
        key: value for key, value in left.items() if key not in ignored
    } == {
        key: value for key, value in right.items() if key not in ignored
    }


class ActualAdmissionAttemptLatchV1:
    """One-process atomic single-consumption latch for one issued attempt."""

    def __init__(self, boundary: Mapping[str, object]) -> None:
        if (
            type(boundary) is not dict
            or type(boundary.get("attempt_id")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", boundary["attempt_id"]) is None
            or type(boundary.get("boundary_root")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", boundary["boundary_root"]) is None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission latch boundary differs")
        self._attempt_id = boundary["attempt_id"]
        self._boundary_root = boundary["boundary_root"]
        self._lock = threading.Lock()
        self._consumed = False

    def consume_once(self, boundary: Mapping[str, object]) -> str:
        if (
            type(boundary) is not dict
            or boundary.get("attempt_id") != self._attempt_id
            or boundary.get("boundary_root") != self._boundary_root
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission consume boundary differs")
        with self._lock:
            if self._consumed:
                _fail(FAIL_ACTUAL_ADMISSION, "admission attempt was already consumed")
            self._consumed = True
            return self._attempt_id


def validate_stage3_to4_admission_boundary_surface_v1(
    value: object,
    source_commit: str,
    artifact_path: Path,
    prior_stage_evidence_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Kernel-level exact surface check; the issuer must also fully replay it."""

    stages = _validate_prior_stage_rows_v1(prior_stage_evidence_rows, source_commit)
    try:
        return _ACTUAL_ADMISSION.validate_stage3_to4_admission_boundary_surface_v1(
            value,
            source_commit,
            artifact_path.as_posix(),
            _stage_root_rows_v1(stages),
        )
    except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
        _fail(FAIL_ACTUAL_ADMISSION, error.detail)


ACTUAL_ORCHESTRATION_STAGE_REGISTRY: Final = (
    (1, "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY"),
    (2, "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO"),
    (3, "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY"),
    (4, "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL"),
    (5, "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES"),
    (6, "RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY"),
    (7, "CLOSE_HOST_RESOURCE_ISOLATION_NEGATIVE_VECTORS_AND_TOCTOU"),
    (8, "BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE"),
    (9, "ADD_PREDICATE20_AND_BUILD_CANONICAL_ARTIFACT_IN_MEMORY"),
    (10, "STRICTLY_REPLAY_COMPLETE_ARTIFACT_BEFORE_PUBLICATION"),
)
ACTUAL_STAGE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actual-orchestration-stage-evidence/1"
)


def close_held_actor_group_v1(
    actors: Sequence[HeldActorProcessV1],
    *,
    child_timeout_seconds: float,
    inspect_reader: Callable[[str], bytes] = _docker_inspect_payload_v1,
    command_runner: Callable[..., object] = subprocess.run,
) -> tuple[dict[str, object], ...]:
    """Close held actors synchronously while not-yet-closed samplers remain live."""

    if (
        type(actors) not in (tuple, list)
        or not actors
        or any(type(actor) is not HeldActorProcessV1 for actor in actors)
        or len({actor.actor_id for actor in actors}) != len(actors)
        or type(child_timeout_seconds) not in (int, float)
        or isinstance(child_timeout_seconds, bool)
        or child_timeout_seconds <= 0
    ):
        _fail(FAIL_POLICY, "held actor group input differs")
    results: list[dict[str, object]] = []
    for index, actor in enumerate(actors):
        try:
            value = close_held_actor_process_v1(
                actor,
                child_timeout_seconds=child_timeout_seconds,
                inspect_reader=inspect_reader,
                command_runner=command_runner,
            )
        except BaseException as error:
            cleanup_errors = _cleanup_actor_set_v1(
                actors[index + 1 :],
                command_runner,
            )
            if cleanup_errors:
                _fail(
                    FAIL_POLICY,
                    f"held actor group close failed for {actor.actor_id}; "
                    "remaining actor cleanup closure failed: "
                    + "; ".join(cleanup_errors)
                    + f"; original={type(error).__name__}:{error}",
                )
            raise
        if type(value) is not dict or value.get("actor_id") != actor.actor_id:
            cleanup_errors = _cleanup_actor_set_v1(
                actors[index + 1 :],
                command_runner,
            )
            detail = f"held actor group result differs for {actor.actor_id}"
            if cleanup_errors:
                detail += "; remaining cleanup failed: " + "; ".join(
                    cleanup_errors
                )
            _fail(FAIL_POLICY, detail)
        results.append(value)
    ordered = tuple(results)
    if len(ordered) != len(actors):
        _fail(FAIL_POLICY, "held actor group result registry differs")
    return ordered


def actual_stage_evidence_v1(
    stage_id: int,
    stage_name: str,
    source_commit: str,
    evidence: object,
    *,
    qualification_count: int,
    qualification_mask: int,
    candidate_receipt_hex: str | None,
    final_receipt_hex: str | None,
) -> dict[str, object]:
    """Create one exact in-memory orchestration transition commitment."""

    if (
        type(stage_id) is not int
        or not 1 <= stage_id <= len(ACTUAL_ORCHESTRATION_STAGE_REGISTRY)
        or ACTUAL_ORCHESTRATION_STAGE_REGISTRY[stage_id - 1]
        != (stage_id, stage_name)
        or type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or type(qualification_count) is not int
        or type(qualification_mask) is not int
    ):
        _fail(FAIL_POLICY, "actual orchestration stage identity differs")
    if stage_id <= 7:
        expected_authority = (0, 0, None, None)
    elif stage_id == 8:
        expected_authority = (19, 0x7FFFF, candidate_receipt_hex, None)
    else:
        expected_authority = (
            20,
            0xFFFFF,
            candidate_receipt_hex,
            final_receipt_hex,
        )
    if (
        (qualification_count, qualification_mask, candidate_receipt_hex, final_receipt_hex)
        != expected_authority
        or (candidate_receipt_hex is not None and re.fullmatch(r"[0-9a-f]+", candidate_receipt_hex) is None)
        or (final_receipt_hex is not None and re.fullmatch(r"[0-9a-f]+", final_receipt_hex) is None)
    ):
        _fail(FAIL_POLICY, "actual orchestration receipt transition differs")
    value = {
        "candidate_receipt_hex": candidate_receipt_hex,
        "evidence": evidence,
        "final_receipt_hex": final_receipt_hex,
        "q1_authority": {
            "certificate_active": False,
            "formal_output_roots": [None] * 8,
            "gate_count": 0,
            "gate_mask": 0,
            "state": "NOT_RUN",
        },
        "qualification_count": qualification_count,
        "qualification_mask": qualification_mask,
        "schema_version": ACTUAL_STAGE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "status": "STAGE_COMPLETE_IN_MEMORY_NOT_PUBLISHED",
    }
    value["stage_evidence_root"] = sha256(
        b"HEGEL/Q05B/ACTUAL/STAGE_EVIDENCE/V1\x00"
        + stage_id.to_bytes(2, "big")
        + _canonical_json_bytes(value)
    ).hexdigest()
    return value


def validate_actual_stage_evidence_v1(
    value: object,
    stage_id: int,
    stage_name: str,
    source_commit: str,
) -> dict[str, object]:
    if type(value) is not dict:
        _fail(FAIL_POLICY, "actual orchestration stage result is not an object")
    expected_keys = {
        "candidate_receipt_hex",
        "evidence",
        "final_receipt_hex",
        "q1_authority",
        "qualification_count",
        "qualification_mask",
        "schema_version",
        "source_commit",
        "stage_evidence_root",
        "stage_id",
        "stage_name",
        "status",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != ACTUAL_STAGE_SCHEMA_VERSION
        or value.get("source_commit") != source_commit
        or type(value.get("stage_id")) is not int
        or value["stage_id"] != stage_id
        or value.get("stage_name") != stage_name
        or value.get("status") != "STAGE_COMPLETE_IN_MEMORY_NOT_PUBLISHED"
        or _canonical_json_bytes(value.get("q1_authority"))
        != _canonical_json_bytes(
            {
                "certificate_active": False,
                "formal_output_roots": [None] * 8,
                "gate_count": 0,
                "gate_mask": 0,
                "state": "NOT_RUN",
            }
        )
    ):
        _fail(FAIL_POLICY, "actual orchestration stage wire differs")
    replay = actual_stage_evidence_v1(
        stage_id,
        stage_name,
        source_commit,
        value["evidence"],
        qualification_count=value["qualification_count"],
        qualification_mask=value["qualification_mask"],
        candidate_receipt_hex=value["candidate_receipt_hex"],
        final_receipt_hex=value["final_receipt_hex"],
    )
    if replay != value:
        _fail(FAIL_POLICY, "actual orchestration stage root differs")
    if stage_id >= 8:
        try:
            host_module = _load_host_replay_module_v1(PROJECT_ROOT)
            wire_module = importlib.import_module(
                "hegel_machine.phase3_q1_qualification_wire_v1"
            )
            candidate_payload = bytes.fromhex(value["candidate_receipt_hex"])
            candidate = wire_module.decode_qualification_candidate_receipt_v1(
                candidate_payload
            )
        except (ValueError, AttributeError, Exception) as error:
            _fail(FAIL_POLICY, f"actual candidate receipt strict replay failed: {error}")
        if candidate.source_commit.hex() != source_commit:
            _fail(FAIL_POLICY, "actual candidate source commit differs")
        if stage_id >= 9:
            try:
                final_payload = bytes.fromhex(value["final_receipt_hex"])
                final = wire_module.decode_qualification_receipt_v1(final_payload)
            except (ValueError, AttributeError, Exception) as error:
                _fail(FAIL_POLICY, f"actual final receipt strict replay failed: {error}")
            if final.candidate_receipt.canonical_bytes != candidate_payload:
                _fail(FAIL_POLICY, "actual final/candidate receipt binding differs")
    return value


def _close_backend_active_mount_slots_body_v1(
    backend: object,
    slot_indexes: Sequence[int],
) -> tuple[str, ...]:
    """Close each fixed binding in place and detach only a closed slot."""

    slots = getattr(backend, "active_mount_binding_slots", None)
    if type(slots) is not list or len(slots) != 3:
        return ("active_mount_binding_slots:registry differs",)
    errors: list[str] = []
    for index in slot_indexes:
        if type(index) is not int or not 0 <= index < 3:
            errors.append("active_mount_binding_slots:index differs")
            continue
        binding = slots[index]
        if binding is None:
            continue
        if type(binding) is not HeldActorMountBindingV1:
            errors.append(f"active_mount_binding_slots[{index}]:type differs")
            continue
        try:
            close_held_actor_mount_binding_v1(binding)
        except BaseException as error:
            errors.append(
                f"{binding.actor_id}:{type(error).__name__}:{error}"
            )
        else:
            slots[index] = None
    return tuple(errors)


def _close_backend_active_mount_slots_v1(
    backend: object,
    slot_indexes: Sequence[int],
) -> tuple[str, ...]:
    with _docker_ownership_signal_guard_v1():
        return _close_backend_active_mount_slots_body_v1(
            backend,
            slot_indexes,
        )


def _cleanup_backend_active_actor_slots_body_v1(
    backend: object,
    slot_indexes: Sequence[int],
    runner: Callable[..., object],
) -> tuple[str, ...]:
    """Close fixed actor slots in place and detach only completed cleanup."""

    slots = getattr(backend, "active_actor_slots", None)
    if type(slots) is not list or len(slots) != 3:
        return ("active_actor_slots:registry differs",)
    errors: list[str] = []
    seen: set[int] = set()
    for index in slot_indexes:
        if type(index) is not int or not 0 <= index < 3:
            errors.append("active_actor_slots:index differs")
            continue
        actor = slots[index]
        if actor is None:
            continue
        if type(actor) is not HeldActorProcessV1:
            errors.append(f"active_actor_slots[{index}]:type differs")
            continue
        identity = id(actor)
        if identity in seen:
            errors.append(f"active_actor_slots[{index}]:actor aliases prior slot")
            continue
        seen.add(identity)
        try:
            actor_errors = _abort_held_actor_cleanup_v1(actor, runner)
        except BaseException as error:
            errors.append(
                f"{actor.actor_id}:cleanup raised:{type(error).__name__}:{error}"
            )
            continue
        if not actor_errors:
            slots[index] = None
        errors.extend(f"{actor.actor_id}:{row}" for row in actor_errors)
    return tuple(errors)


def _cleanup_backend_active_actor_slots_v1(
    backend: object,
    slot_indexes: Sequence[int],
    runner: Callable[..., object],
) -> tuple[str, ...]:
    with _docker_ownership_signal_guard_v1():
        return _cleanup_backend_active_actor_slots_body_v1(
            backend,
            slot_indexes,
            runner,
        )


class ConcreteQ05BActualBackendV1:
    """Concrete offline backend; construction does not execute any actor."""

    def __init__(
        self,
        project_root: Path,
        source_commit: str,
        artifact_path: Path,
        cargo_cache_source: Path,
        work_root: Path,
        *,
        inspect_reader: Callable[[str], bytes] = _docker_inspect_payload_v1,
        command_runner: Callable[..., object] = subprocess.run,
        actor_starter: Callable[..., HeldActorProcessV1] = start_held_actor_process_v1,
        actor_group_closer: Callable[..., tuple[dict[str, object], ...]] = (
            close_held_actor_group_v1
        ),
        admission_nonce_source: Callable[[int], bytes] = os.urandom,
    ) -> None:
        if (
            not isinstance(project_root, Path)
            or not project_root.is_absolute()
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
            or not isinstance(artifact_path, Path)
            or not artifact_path.is_absolute()
            or not isinstance(cargo_cache_source, Path)
            or not cargo_cache_source.is_absolute()
            or not isinstance(work_root, Path)
            or not work_root.is_absolute()
            or not callable(inspect_reader)
            or not callable(command_runner)
            or not callable(actor_starter)
            or not callable(actor_group_closer)
            or not callable(admission_nonce_source)
        ):
            _fail(FAIL_POLICY, "concrete actual backend inputs differ")
        root_status = work_root.lstat()
        if (
            work_root.is_symlink()
            or not stat.S_ISDIR(root_status.st_mode)
            or stat.S_IMODE(root_status.st_mode) != 0o700
            or tuple(work_root.iterdir())
        ):
            _fail(FAIL_POLICY, "actual work root must be one empty 0700 directory")
        expected_artifact = project_root / ACTUAL_ARTIFACT_RELATIVE_PATH
        if artifact_path != expected_artifact:
            _fail(FAIL_ARTIFACT, "actual artifact path is not the frozen path")
        self.project_root = project_root
        self.source_commit = source_commit
        self.artifact_path = artifact_path
        self.cargo_cache_source = cargo_cache_source
        self.work_root = work_root
        self.inspect_reader = inspect_reader
        self.command_runner = command_runner
        self.actor_starter = actor_starter
        self.actor_group_closer = actor_group_closer
        admission_nonce = admission_nonce_source(32)
        if type(admission_nonce) is not bytes or len(admission_nonce) != 32:
            _fail(FAIL_ACTUAL_ADMISSION, "admission nonce source differs")
        self.admission_nonce = admission_nonce
        try:
            self.docker_execution_slot_rows = (
                _ACTUAL_ADMISSION.docker_execution_slot_rows_v1(
                    source_commit,
                    admission_nonce,
                )
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)
        self.docker_execution_authority: dict[str, object] | None = None
        self.admission_git_transcript_collector = git_source_admission_transcript_v1
        self.completed_stage = 0
        self.paths: dict[str, Path] = {}
        self.source_evidence: dict[str, dict[str, object]] = {}
        self.source_object_closure: dict[str, object] | None = None
        self.image_evidence: dict[str, dict[str, object]] = {}
        self.snapshot_evidence: dict[str, dict[str, object]] = {}
        self.cargo_evidence: dict[str, object] | None = None
        self.cargo_lock_payload: bytes | None = None
        self.build_evidence: dict[str, object] | None = None
        self.binary_detach_evidence: dict[str, object] | None = None
        self.binary_evidence: dict[str, object] | None = None
        self.seccomp_evidence: dict[str, dict[str, object]] = {}
        # These fixed, preallocated slots are the ownership handoff boundary
        # for live actors and held mount descriptors.  A returned object is
        # stored here before any list append, tuple construction, validation,
        # or evidence allocation can fail.
        self.active_actor_slots: list[HeldActorProcessV1 | None] = [
            None,
            None,
            None,
        ]
        self.active_mount_binding_slots: list[HeldActorMountBindingV1 | None] = [
            None,
            None,
            None,
        ]
        self.endpoint_actors: tuple[HeldActorProcessV1, HeldActorProcessV1] | None = None
        self.host_actor: HeldActorProcessV1 | None = None
        self.endpoint_complete: tuple[dict[str, object], dict[str, object]] | None = None
        self.endpoint_payloads: tuple[tuple[bytes, ...], tuple[bytes, ...]] | None = None
        self.endpoint_tree_identities: tuple[dict[str, object], dict[str, object]] | None = None
        self.stdout_paths: tuple[Path, Path, Path] | None = None
        self.endpoint_stdout: tuple[bytes, bytes] | None = None
        self.sidecar_canonical_rows: list[dict[str, object]] | None = None
        self.stdout_manifest: bytes | None = None
        self.stdout_tree_identity: dict[str, object] | None = None
        self.host_complete: dict[str, object] | None = None
        self.host_command: list[str] | None = None
        self.host_staging_identity: dict[str, object] | None = None
        self.host_witness: bytes | None = None
        self.host_source_identity_root: bytes | None = None
        self.host_runtime_identity_root: bytes | None = None
        self.outer_replay: object | None = None
        self.actual_evidence_sections: dict[str, object] | None = None
        self.candidate_replay: dict[str, object] | None = None
        self.actual_artifact_value: dict[str, object] | None = None
        self.actual_artifact_replay: dict[str, object] | None = None
        self.admission_precondition_bundle: dict[str, object] | None = None
        self.admission_decision: dict[str, object] | None = None
        self.admission_boundary: dict[str, object] | None = None
        self.admission_fresh_status: GitSourceStatusV1 | None = None
        self.admission_git_source_transcript: dict[str, object] | None = None
        self.admission_consume_git_source_transcript: dict[str, object] | None = None
        self.admission_artifact_absence: dict[str, object] | None = None
        self.admission_fresh_runtime_evidence: dict[str, object] | None = None
        self.admission_fresh_runtime_checkpoints: dict[int, dict[str, object]] = {}
        self.dynamic_mount_authority_set: dict[str, object] | None = None
        self.stage_evidence_rows: dict[int, dict[str, object]] = {}
        self.actor_mount_bindings: dict[int, dict[str, object]] = {}
        self.actor_mount_launch_replays: dict[int, dict[str, object]] = {}
        self.admission_consume_artifact_absence: dict[str, object] | None = None
        self.admission_work_root_identity: dict[str, object] | None = None
        self.admission_issue_record: dict[str, object] | None = None
        self.admission_work_root_descriptor: int | None = None
        self.admission_issued_marker_descriptor: int | None = None
        self.admission_spending_marker_descriptor: int | None = None
        self.admission_consumed_marker_descriptor: int | None = None
        self.admission_consumed_marker_evidence: dict[str, object] | None = None
        self.admission_work_root_replay: dict[str, object] | None = None
        self.admission_live_marker_replays: dict[str, dict[str, object]] = {}
        self.admission_latch: ActualAdmissionAttemptLatchV1 | None = None
        self.admission_consumed = False

    def issue_stage3_to4_admission_boundary_v1(
        self,
        context: Mapping[str, object],
    ) -> dict[str, object]:
        """Issue one boundary only after stage 3 is fixed and before launch."""

        if (
            self.completed_stage != 3
            or type(context) is not dict
            or self.admission_boundary is not None
            or self.admission_issue_record is not None
            or self.admission_work_root_descriptor is not None
            or self.admission_issued_marker_descriptor is not None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission boundary issue order differs")
        stage_rows = tuple(context.get(f"stage_{index:02d}") for index in range(1, 4))
        if any(type(row) is not dict for row in stage_rows):
            _fail(FAIL_ACTUAL_ADMISSION, "admission prior stages are incomplete")
        typed_stages = tuple(stage_rows)
        stage1_evidence = typed_stages[0].get("evidence")
        if type(stage1_evidence) is not dict:
            _fail(FAIL_ACTUAL_ADMISSION, "admission stage 1 evidence differs")
        config_hex = stage1_evidence.get("config_hex")
        if (
            type(config_hex) is not str
            or not config_hex
            or len(config_hex) % 2
            or re.fullmatch(r"[0-9a-f]+", config_hex) is None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "Commit-A config stage bytes differ")
        commit_a_config_bytes = bytes.fromhex(config_hex)
        git_transcript = self.admission_git_transcript_collector(
            self.project_root, self.source_commit
        )
        fresh_status = GitSourceStatusV1(self.source_commit, True, 0)
        artifact_absence = actual_artifact_absence_evidence_v1(self.artifact_path)
        layout = stage1_evidence.get("layout")
        if type(layout) is not dict:
            _fail(FAIL_ACTUAL_ADMISSION, "admission work layout differs")
        work_identity = actual_work_root_identity_v1(self.work_root, layout)
        if self.cargo_evidence is None or self.binary_evidence is None:
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "fresh runtime replay preceded sealed Cargo/binary evidence",
            )
        fresh_runtime = collect_fresh_runtime_evidence_set_v1(
            self.project_root,
            self.source_commit,
            self.paths,
            self.source_evidence,
            self.snapshot_evidence,
            self.image_evidence,
            self.cargo_evidence,
            self.seccomp_evidence,
            self.binary_evidence,
            command_runner=self.command_runner,
        )
        bundle = build_actual_precondition_bundle_v1(
            self.source_commit,
            commit_a_config_bytes,
            self.artifact_path,
            work_identity,
            typed_stages,
            fresh_status,
            git_transcript,
            artifact_absence,
            fresh_runtime,
        )
        docker_authority = stage1_evidence.get("docker_execution_authority")
        if (
            self.docker_execution_authority is None
            or docker_authority != self.docker_execution_authority
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "Stage-1 Docker execution authority differs at issue",
            )
        try:
            _ACTUAL_ADMISSION.validate_docker_execution_authority_v1(
                docker_authority,
                self.source_commit,
                self.admission_nonce,
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)
        decision = build_actual_admission_decision_v1(
            self.source_commit,
            commit_a_config_bytes,
            self.artifact_path,
            self.admission_nonce,
            bundle,
        )
        try:
            _ACTUAL_ADMISSION.cross_docker_execution_authority_to_admission_decision_v1(
                docker_authority,
                decision,
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)
        boundary = build_stage3_to4_admission_boundary_v1(
            self.source_commit,
            commit_a_config_bytes,
            self.artifact_path,
            bundle,
            decision,
        )
        replay = decode_stage3_to4_admission_boundary_v1(
            _canonical_json_bytes(boundary),
            self.source_commit,
            commit_a_config_bytes,
            self.artifact_path,
            work_identity,
            typed_stages,
            fresh_status,
            git_transcript,
            artifact_absence,
            fresh_runtime,
        )
        validate_stage3_to4_admission_boundary_surface_v1(
            replay,
            self.source_commit,
            self.artifact_path,
            typed_stages,
        )
        issue_record, root_descriptor, issued_descriptor = (
            issue_actual_admission_marker_v1(
            self.work_root,
            work_identity,
            replay,
            )
        )
        try:
            validated_record, marker_boundary = (
                validate_actual_admission_issue_record_v1(issue_record)
            )
        except BaseException:
            os.close(issued_descriptor)
            os.close(root_descriptor)
            raise
        if _canonical_json_bytes(marker_boundary) != _canonical_json_bytes(replay):
            os.close(issued_descriptor)
            os.close(root_descriptor)
            _fail(FAIL_ACTUAL_ADMISSION, "issued marker boundary differs")
        self.admission_precondition_bundle = bundle
        self.admission_decision = decision
        self.admission_boundary = replay
        self.admission_fresh_status = fresh_status
        self.admission_git_source_transcript = git_transcript
        self.admission_artifact_absence = artifact_absence
        self.admission_fresh_runtime_evidence = fresh_runtime
        self.admission_work_root_identity = work_identity
        self.admission_issue_record = validated_record
        self.admission_work_root_descriptor = root_descriptor
        self.admission_issued_marker_descriptor = issued_descriptor
        return validated_record

    def strict_replay_stage3_to4_admission_boundary_v1(
        self,
        issue_record: object,
        context: Mapping[str, object],
    ) -> dict[str, object]:
        if (
            type(issue_record) is not dict
            or type(context) is not dict
            or self.admission_boundary is None
            or self.admission_issue_record is None
            or self.admission_precondition_bundle is None
            or self.admission_decision is None
            or self.admission_fresh_status is None
            or self.admission_git_source_transcript is None
            or self.admission_artifact_absence is None
            or self.admission_fresh_runtime_evidence is None
            or self.admission_work_root_identity is None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission boundary replay state differs")
        stages = tuple(context.get(f"stage_{index:02d}") for index in range(1, 4))
        if any(type(row) is not dict for row in stages):
            _fail(FAIL_ACTUAL_ADMISSION, "admission boundary replay stages differ")
        record, marker_boundary = validate_actual_admission_issue_record_v1(issue_record)
        if _canonical_json_bytes(record) != _canonical_json_bytes(
            self.admission_issue_record
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission issued/replayed record differs")
        config_bytes = bytes.fromhex(stages[0]["evidence"]["config_hex"])
        replay = decode_stage3_to4_admission_boundary_v1(
            _canonical_json_bytes(marker_boundary),
            self.source_commit,
            config_bytes,
            self.artifact_path,
            self.admission_work_root_identity,
            tuple(stages),
            self.admission_fresh_status,
            self.admission_git_source_transcript,
            self.admission_artifact_absence,
            self.admission_fresh_runtime_evidence,
        )
        if _canonical_json_bytes(replay) != _canonical_json_bytes(self.admission_boundary):
            _fail(FAIL_ACTUAL_ADMISSION, "admission issued/replayed boundary differs")
        return record

    def _consume_stage3_to4_admission_boundary_v1(
        self,
        context: Mapping[str, object],
    ) -> str:
        if (
            self.admission_boundary is None
            or self.admission_issue_record is None
            or self.admission_work_root_descriptor is None
            or self.admission_issued_marker_descriptor is None
            or self.admission_artifact_absence is None
            or self.admission_work_root_identity is None
            or self.admission_consumed
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admission attempt is absent or consumed")
        # Persistent spend is the first Stage-4 transition.  Caller context,
        # Git, artifact, and path replays happen only after the nonce cannot be
        # reused by this backend or by a restarted empty-work-root admission.
        self.admission_consumed = True
        consumed, spending_descriptor, consumed_descriptor = (
            consume_actual_admission_marker_v1(
            self.admission_work_root_descriptor,
            self.admission_issued_marker_descriptor,
            self.admission_issue_record,
            )
        )
        self.admission_consumed_marker_evidence = consumed
        self.admission_spending_marker_descriptor = spending_descriptor
        self.admission_consumed_marker_descriptor = consumed_descriptor
        work_root_replay = _revalidate_anchored_admission_work_root_v1(
            self.admission_work_root_descriptor,
            self.work_root,
            self.admission_work_root_identity,
        )
        self.admission_work_root_replay = work_root_replay
        self.admission_live_marker_replays["CONSUME_BEFORE_PREFLIGHT"] = (
            replay_live_actual_admission_markers_v1(
                self.admission_work_root_descriptor,
                self.admission_issued_marker_descriptor,
                self.admission_spending_marker_descriptor,
                self.admission_consumed_marker_descriptor,
                self.work_root,
                self.admission_work_root_identity,
                self.admission_issue_record,
                self.admission_consumed_marker_evidence,
                "CONSUME_BEFORE_PREFLIGHT",
            )
        )
        if type(context) is not dict:
            _fail(FAIL_ACTUAL_ADMISSION, "stage 4 admission context differs")
        issue_record = context.get("stage3_to4_admission_issue_record")
        if (
            type(issue_record) is not dict
            or _canonical_json_bytes(issue_record)
            != _canonical_json_bytes(self.admission_issue_record)
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "stage 4 admission issue record differs")
        fresh_git = self.admission_git_transcript_collector(
            self.project_root,
            self.source_commit,
        )
        self.admission_consume_git_source_transcript = fresh_git
        return self.admission_boundary["attempt_id"]

    def strict_replay_actual_admission_live_authority_v1(
        self,
        checkpoint: str,
    ) -> dict[str, object]:
        if (
            not self.admission_consumed
            or self.admission_issue_record is None
            or self.admission_consumed_marker_evidence is None
            or self.admission_work_root_identity is None
            or self.admission_work_root_descriptor is None
            or self.admission_issued_marker_descriptor is None
            or self.admission_spending_marker_descriptor is None
            or self.admission_consumed_marker_descriptor is None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "live admission authority is incomplete")
        replay = replay_live_actual_admission_markers_v1(
            self.admission_work_root_descriptor,
            self.admission_issued_marker_descriptor,
            self.admission_spending_marker_descriptor,
            self.admission_consumed_marker_descriptor,
            self.work_root,
            self.admission_work_root_identity,
            self.admission_issue_record,
            self.admission_consumed_marker_evidence,
            checkpoint,
        )
        self.admission_live_marker_replays[checkpoint] = replay
        return replay

    def _collect_fresh_runtime_checkpoint_v1(
        self,
        checkpoint_id: int,
        mount_binding_rows: Sequence[Mapping[str, object]],
        dynamic_authority_set: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if (
            type(checkpoint_id) is not int
            or checkpoint_id != len(self.admission_fresh_runtime_checkpoints) + 1
            or self.admission_boundary is None
            or self.admission_issue_record is None
            or self.admission_consumed_marker_evidence is None
            or self.admission_fresh_runtime_evidence is None
            or self.cargo_evidence is None
            or self.binary_evidence is None
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "fresh runtime checkpoint order/state differs",
            )
        if checkpoint_id == 1:
            if dynamic_authority_set is not None:
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "endpoint checkpoint carried dynamic host authority",
                )
        elif checkpoint_id == 2:
            if (
                type(dynamic_authority_set) is not dict
                or type(self.stage_evidence_rows.get(5)) is not dict
            ):
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "host checkpoint dynamic authority is absent",
                )
        else:
            checkpoint_two = self.admission_fresh_runtime_checkpoints.get(2)
            if (
                type(dynamic_authority_set) is not dict
                or type(checkpoint_two) is not dict
                or _canonical_json_bytes(dynamic_authority_set)
                != _canonical_json_bytes(
                    checkpoint_two.get("dynamic_authority_set")
                )
            ):
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "stage7 dynamic authority differs from checkpoint2 bytes",
                )
        observed = collect_fresh_runtime_evidence_set_v1(
            self.project_root,
            self.source_commit,
            self.paths,
            self.source_evidence,
            self.snapshot_evidence,
            self.image_evidence,
            self.cargo_evidence,
            self.seccomp_evidence,
            self.binary_evidence,
            command_runner=self.command_runner,
        )
        absence = actual_artifact_absence_evidence_v1(self.artifact_path)
        if (
            self.admission_artifact_absence is None
            or absence != self.admission_artifact_absence
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "artifact absence changed at fresh runtime checkpoint",
            )
        if checkpoint_id == 1:
            if self.admission_consume_artifact_absence is not None:
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "consume artifact absence was already recorded",
                )
            # Preserve the consume-time observation before the byte-equality
            # replay below.  A replay failure has already spent the attempt,
            # and must not erase the exact absence observation that preceded
            # the rejected launch.
            self.admission_consume_artifact_absence = dict(absence)
        elif (
            self.admission_consume_artifact_absence is None
            or absence != self.admission_consume_artifact_absence
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "consume artifact absence differs at later checkpoint",
            )
        checkpoint = build_fresh_runtime_checkpoint_v1(
            self.source_commit,
            self.artifact_path,
            checkpoint_id,
            self.admission_boundary["attempt_id"],
            self.admission_boundary["boundary_root"],
            self.admission_issue_record["issue_record_root"],
            self.admission_consumed_marker_evidence["consumed_marker_root"],
            self.admission_fresh_runtime_evidence,
            observed,
            absence,
            mount_binding_rows,
            dynamic_authority_set,
            None if checkpoint_id == 1 else self.stage_evidence_rows[5],
            stage_5_issue_record=(
                None if checkpoint_id == 1 else self.admission_issue_record
            ),
            stage_5_consumed_marker_evidence=(
                None
                if checkpoint_id == 1
                else self.admission_consumed_marker_evidence
            ),
            stage_5_checkpoint_1=(
                None
                if checkpoint_id == 1
                else self.admission_fresh_runtime_checkpoints[1]
            ),
            stage_5_mount_launch_replay_rows=(
                None
                if checkpoint_id == 1
                else [
                    self.actor_mount_launch_replays[role_id]
                    for role_id in (1, 2)
                ]
            ),
        )
        self.admission_fresh_runtime_checkpoints[checkpoint_id] = checkpoint
        return checkpoint

    def _prepare_actor_mount_binding_v1(
        self,
        role_id: int,
        actor_id: str,
        command: Sequence[str],
        *,
        ownership_slot_index: int | None = None,
    ) -> HeldActorMountBindingV1:
        """Anchor exact argv sources against admission/stage evidence."""

        if (
            type(role_id) is not int
            or role_id not in (1, 2, 3)
            or actor_id != ROLE_ROWS[role_id - 1][1]
            or self.admission_fresh_runtime_evidence is None
            or (
                ownership_slot_index is not None
                and (
                    type(ownership_slot_index) is not int
                    or ownership_slot_index != role_id - 1
                    or type(self.active_mount_binding_slots) is not list
                    or len(self.active_mount_binding_slots) != 3
                    or self.active_mount_binding_slots[ownership_slot_index]
                    is not None
                )
            )
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "actor mount binding state differs")
        try:
            fresh = _ACTUAL_ADMISSION.validate_fresh_runtime_evidence_set_v1(
                self.admission_fresh_runtime_evidence,
                self.source_commit,
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)
        registry = sealed_actor_mount_registry_v1(role_id, command)
        if registry.role_id != role_id:
            _fail(FAIL_ACTUAL_ADMISSION, "actor mount registry role differs")
        actors = {row["actor_id"]: row for row in fresh["actor_rows"]}
        runtime_seccomp = next(
            (
                row
                for row in fresh["seccomp_rows"]
                if row["label"] == "runtime"
            ),
            None,
        )
        if (
            type(runtime_seccomp) is not dict
            or runtime_seccomp["evidence"] != self.seccomp_evidence.get("runtime")
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "admitted runtime seccomp differs")

        def mutable_directory_spec(
            destination: str,
            path: Path,
        ) -> tuple[
            str,
            str,
            dict[str, object],
            tuple[int, int, int, int, int, int, None],
            str | None,
            bool,
        ]:
            try:
                value = path.lstat()
            except OSError as error:
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    f"prelaunch writable mount is unavailable: {error}",
                )
            evidence = (
                _ACTUAL_ADMISSION.build_prelaunch_writable_directory_evidence_v1(
                    role_id,
                    destination,
                    path.as_posix(),
                    value.st_dev,
                    value.st_ino,
                    value.st_nlink,
                    value.st_uid,
                    value.st_gid,
                    stat.S_IMODE(value.st_mode),
                )
            )
            return (
                "PRELAUNCH_WRITABLE_DIRECTORY",
                f"{actor_id}{destination}",
                evidence,
                (
                    value.st_dev,
                    value.st_ino,
                    value.st_nlink,
                    value.st_uid,
                    value.st_gid,
                    stat.S_IMODE(value.st_mode),
                    None,
                ),
                None,
                True,
            )

        def tree_spec(
            kind: str,
            label: str,
            evidence: Mapping[str, object],
        ) -> tuple[
            str,
            str,
            Mapping[str, object],
            tuple[int, int, int, None, None, int, None],
            None,
            bool,
        ]:
            if (
                type(evidence) is not dict
                or type(evidence.get("root_device")) is not int
                or type(evidence.get("root_inode")) is not int
                or type(evidence.get("root_nlink")) is not int
                or type(evidence.get("root_mode")) is not int
            ):
                _fail(FAIL_ACTUAL_ADMISSION, "sealed mount tree identity differs")
            return (
                kind,
                label,
                evidence,
                (
                    evidence["root_device"],
                    evidence["root_inode"],
                    evidence["root_nlink"],
                    None,
                    None,
                    evidence["root_mode"],
                    None,
                ),
                None,
                False,
            )

        def file_spec(
            kind: str,
            label: str,
            evidence: Mapping[str, object],
            identity: tuple[int, int, int, int, int, int, int, str],
        ) -> tuple[
            str,
            str,
            Mapping[str, object],
            tuple[int, int, int, int, int, int, int],
            str,
            bool,
        ]:
            return (
                kind,
                label,
                evidence,
                identity[:7],
                identity[7],
                False,
            )

        expected_paths: dict[str, Path]
        specs: dict[str, tuple[object, ...]] = {}
        if role_id == 1:
            expected_paths = {
                "/control": self.paths["python_control"],
                "/output": self.paths["python_output"],
                "/snapshot": self.paths["python_snapshot"],
            }
            specs["/control"] = mutable_directory_spec(
                "/control", expected_paths["/control"]
            )
            specs["/output"] = mutable_directory_spec(
                "/output", expected_paths["/output"]
            )
            specs["/snapshot"] = tree_spec(
                "FRESH_ACTOR_SNAPSHOT",
                "PYTHON_ENDPOINT/snapshot",
                actors["PYTHON_ENDPOINT"]["snapshot_evidence"],
            )
        elif role_id == 2:
            expected_paths = {
                "/control": self.paths["rust_control"],
                "/output": self.paths["rust_output"],
                "/runtime/hegel-q1-archive-projection-oracle": self.paths[
                    "binary"
                ],
            }
            specs["/control"] = mutable_directory_spec(
                "/control", expected_paths["/control"]
            )
            specs["/output"] = mutable_directory_spec(
                "/output", expected_paths["/output"]
            )
            binary = fresh["binary"]["identity"]
            specs["/runtime/hegel-q1-archive-projection-oracle"] = file_spec(
                "FRESH_PREBUILT_RUST_BINARY",
                "RUST_ENDPOINT/runtime",
                binary,
                (
                    binary["device"],
                    binary["inode"],
                    binary["nlink"],
                    binary["uid"],
                    binary["gid"],
                    binary["mode"],
                    binary["size"],
                    binary["sha256"],
                ),
            )
        else:
            if (
                self.endpoint_tree_identities is None
                or self.stdout_paths is None
                or self.stdout_tree_identity is None
            ):
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "trusted-host sealed input identities are absent",
                )
            expected_paths = {
                "/control": self.paths["host_control"],
                "/inputs/python": self.paths["python_output"],
                "/inputs/rust": self.paths["rust_output"],
                "/inputs/stdout/python.stdout": self.stdout_paths[0],
                "/inputs/stdout/rust.stdout": self.stdout_paths[1],
                "/inputs/stdout/manifest.json": self.stdout_paths[2],
                "/snapshot": self.paths["host_snapshot"],
                "/staging": self.paths["host_staging"],
            }
            specs["/control"] = mutable_directory_spec(
                "/control", expected_paths["/control"]
            )
            specs["/staging"] = mutable_directory_spec(
                "/staging", expected_paths["/staging"]
            )
            specs["/snapshot"] = tree_spec(
                "FRESH_ACTOR_SNAPSHOT",
                "TRUSTED_HOST_REPLAY/snapshot",
                actors["TRUSTED_HOST_REPLAY"]["snapshot_evidence"],
            )
            specs["/inputs/python"] = tree_spec(
                "SEALED_ENDPOINT_TREE",
                "PYTHON_ENDPOINT/output",
                self.endpoint_tree_identities[0],
            )
            specs["/inputs/rust"] = tree_spec(
                "SEALED_ENDPOINT_TREE",
                "RUST_ENDPOINT/output",
                self.endpoint_tree_identities[1],
            )
            stdout_files = {
                row[0]: row for row in self.stdout_tree_identity["file_rows"]
            }
            for destination, relative in (
                ("/inputs/stdout/python.stdout", "python.stdout"),
                ("/inputs/stdout/rust.stdout", "rust.stdout"),
                ("/inputs/stdout/manifest.json", "manifest.json"),
            ):
                row = stdout_files.get(relative)
                if type(row) is not list or len(row) != 11:
                    _fail(
                        FAIL_ACTUAL_ADMISSION,
                        "sealed stdout mount file identity differs",
                    )
                authority = {
                    "schema_version": (
                        "hegel-phase3a-q05b-sealed-stdout-mount-file/1"
                    ),
                    "tree_manifest_sha256": self.stdout_tree_identity[
                        "manifest_sha256"
                    ],
                    "relative_path": relative,
                    "file_row": row,
                }
                specs[destination] = file_spec(
                    "SEALED_STDOUT_FILE",
                    f"TRUSTED_HOST_REPLAY{destination}",
                    authority,
                    (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[10],
                    ),
                )
        if set(expected_paths) != set(registry.expected_sources):
            _fail(FAIL_ACTUAL_ADMISSION, "actor mount destination registry differs")
        if any(
            registry.expected_sources[destination]
            != expected_paths[destination].as_posix()
            for destination in expected_paths
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "actor mount Source differs from sealed path registry",
            )

        expected_role_rows = ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[role_id - 1][2]
        held_source_slots: list[HeldActorMountSourceV1 | None] = [
            None
        ] * (len(expected_role_rows) + 1)

        def hold_source(index: int, source: HeldActorMountSourceV1) -> None:
            if held_source_slots[index] is not None:
                _fail(FAIL_POLICY, "held mount ownership slot differs")
            held_source_slots[index] = source

        try:
            for index, (
                destination,
                writable,
                source_type,
                source_mode,
            ) in enumerate(expected_role_rows):
                spec = specs[destination]
                source = _open_held_actor_mount_source_v1(
                    role_id,
                    destination,
                    expected_paths[destination],
                    writable,
                    source_type,
                    source_mode,
                    spec[0],
                    spec[1],
                    spec[2],
                    expected_identity=spec[3],
                    expected_payload_sha256=spec[4],
                    require_empty_directory=spec[5],
                    ownership_sink=(
                        lambda value, slot=index: hold_source(slot, value)
                    ),
                )
                if held_source_slots[index] is not source:
                    _fail(FAIL_POLICY, "held mount ownership handoff differs")
            seccomp_evidence = runtime_seccomp["evidence"]
            seccomp_path = Path(seccomp_evidence["absolute_path"])
            if (
                registry.security_options[1]
                != f"seccomp={seccomp_path.as_posix()}"
            ):
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "actor seccomp Source differs from admitted policy",
                )
            seccomp_index = len(expected_role_rows)
            held_seccomp = _open_held_actor_mount_source_v1(
                role_id,
                "@seccomp",
                seccomp_path,
                False,
                "REGULAR_FILE",
                0o444,
                "RUNTIME_SECCOMP_POLICY",
                f"{actor_id}/@seccomp",
                seccomp_evidence,
                expected_identity=(
                    seccomp_evidence["file_device"],
                    seccomp_evidence["file_inode"],
                    seccomp_evidence["file_nlink"],
                    seccomp_evidence["file_uid"],
                    seccomp_evidence["file_gid"],
                    seccomp_evidence["file_mode"],
                    seccomp_evidence["file_size"],
                ),
                expected_payload_sha256=seccomp_evidence["payload_sha256"],
                ownership_sink=(
                    lambda value: hold_source(seccomp_index, value)
                ),
            )
            if held_source_slots[seccomp_index] is not held_seccomp:
                _fail(FAIL_POLICY, "held seccomp ownership handoff differs")
            held_sources = tuple(
                source
                for source in held_source_slots[:-1]
                if type(source) is HeldActorMountSourceV1
            )
            if len(held_sources) != len(expected_role_rows):
                _fail(FAIL_POLICY, "held mount ownership registry incomplete")
            binding_value = _ACTUAL_ADMISSION.build_actor_mount_binding_v1(
                tuple(command),
                actor_mount_registry_object_v1(registry),
                [source.source_row for source in held_sources],
                held_seccomp.source_row,
            )
            binding = HeldActorMountBindingV1(
                role_id,
                actor_id,
                tuple(command),
                registry,
                binding_value,
                tuple(held_sources),
                held_seccomp,
            )
            if ownership_slot_index is not None:
                self.active_mount_binding_slots[ownership_slot_index] = binding
            return binding
        except BaseException as original:
            close_errors = _close_held_mount_sources_best_effort_v1(
                tuple(
                    source
                    for source in held_source_slots
                    if type(source) is HeldActorMountSourceV1
                )
            )
            if close_errors:
                _fail(
                    FAIL_POLICY,
                    "partial actor mount preparation close failed: "
                    + "; ".join(close_errors)
                    + f"; original={type(original).__name__}:{original}",
                )
            raise

    def _launch_prepared_actor_mount_binding_v1(
        self,
        binding: HeldActorMountBindingV1,
        docker_slot: str,
        cidfile: Path,
        control_root: Path,
    ) -> HeldActorProcessV1:
        actor: HeldActorProcessV1 | None = None
        failure: BaseException | None = None
        unbound_cleanup_errors: tuple[str, ...] = ()
        if (
            type(binding) is not HeldActorMountBindingV1
            or binding.role_id not in (1, 2, 3)
            or self.docker_execution_authority is None
            or type(self.active_actor_slots) is not list
            or len(self.active_actor_slots) != 3
            or self.active_actor_slots[binding.role_id - 1] is not None
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority is absent")
        ownership_slot_index = binding.role_id - 1
        docker_slot_row = _docker_execution_slot_row_v1(
            self.docker_execution_authority,
            docker_slot,
        )
        container_name = docker_slot_row["container_name"]
        expected_principal = _docker_execution_principal_v1(
            binding.exact_command,
            self.docker_execution_authority,
            docker_slot,
        )

        def ownership_sink(candidate: HeldActorProcessV1) -> None:
            if type(candidate) is not HeldActorProcessV1:
                _fail(FAIL_POLICY, "actor ownership handoff type differs")
            current = self.active_actor_slots[ownership_slot_index]
            if current is not None and current is not candidate:
                _fail(FAIL_POLICY, "actor ownership handoff slot differs")
            self.active_actor_slots[ownership_slot_index] = candidate

        try:
            with _docker_ownership_signal_guard_v1():
                actor = self.actor_starter(
                    binding.role_id,
                    binding.actor_id,
                    container_name,
                    binding.exact_command,
                    cidfile,
                    control_root,
                    docker_execution_authority=self.docker_execution_authority,
                    docker_slot=docker_slot,
                    inspect_reader=self.inspect_reader,
                    command_runner=self.command_runner,
                    ownership_sink=ownership_sink,
                )
                if self.active_actor_slots[ownership_slot_index] is None:
                    ownership_sink(actor)
                elif self.active_actor_slots[ownership_slot_index] is not actor:
                    _fail(FAIL_POLICY, "actor ownership handoff result differs")
            if (
                type(actor) is not HeldActorProcessV1
                or actor.container_name != container_name
                or actor.cidfile != cidfile
                or actor.control_root != control_root
                or actor.docker_execution_authority_manifest_sha256
                != self.docker_execution_authority["manifest_sha256"]
                or actor.docker_execution_slot_row != docker_slot_row
                or type(actor.precreate_absence_evidence) is not dict
                or actor.ownership_label_root
                != expected_principal["ownership_label_root"]
                or actor.docker_execution_principal != expected_principal
                or actor.failure_cleanup_attempted is not False
                or actor.failure_cleanup_complete is not False
            ):
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "actor starter returned a different mount principal",
                )
            try:
                precreate_replay = (
                    _ACTUAL_ADMISSION.validate_docker_precreate_absence_v1(
                        actor.precreate_absence_evidence,
                        self.docker_execution_authority,
                    )
                )
            except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
                _fail(FAIL_ACTUAL_ADMISSION, error.detail)
            if precreate_replay != actor.precreate_absence_evidence:
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "actor starter Docker precreate replay differs",
                )
            launch_replay = replay_held_actor_mount_binding_after_start_v1(
                binding, actor
            )
            self.actor_mount_bindings[binding.role_id] = binding.binding
            self.actor_mount_launch_replays[binding.role_id] = launch_replay
        except BaseException as error:
            failure = error
            owned = self.active_actor_slots[ownership_slot_index]
            if actor is None and type(owned) is HeldActorProcessV1:
                actor = owned
            elif type(owned) is not HeldActorProcessV1 and type(actor) is not HeldActorProcessV1:
                # An injected starter is allowed to raise before returning and
                # may have ignored ownership_sink after causing a Docker daemon
                # side effect.  Names remain read-only discovery targets; only
                # an exact owned principal can yield a CID for destructive
                # cleanup.  Persistent absence is deliberately unresolved.
                unbound_errors: list[str] = []
                with _docker_ownership_signal_guard_v1():
                    _docker_remove_and_quiet_absence_v1(
                        None,
                        expected_principal,
                        self.command_runner,
                        unbound_errors,
                    )
                unbound_cleanup_errors = tuple(unbound_errors)
        close_error: BaseException | None = None
        try:
            close_held_actor_mount_binding_v1(binding)
        except BaseException as error:
            close_error = error
        if close_error is not None:
            failure = Q05BDualSupervisorError(
                FAIL_POLICY,
                f"{binding.actor_id} mount authority close failed: "
                f"{type(close_error).__name__}:{close_error}; "
                + (
                    "launch replay otherwise passed"
                    if failure is None
                    else f"original={type(failure).__name__}:{failure}"
                ),
            )
        if failure is not None:
            if type(self.active_actor_slots[ownership_slot_index]) is HeldActorProcessV1:
                cleanup_errors = _cleanup_backend_active_actor_slots_v1(
                    self,
                    (ownership_slot_index,),
                    self.command_runner,
                )
                if cleanup_errors:
                    _fail(
                        FAIL_POLICY,
                        f"{binding.actor_id} mount launch replay cleanup failed: "
                        + "; ".join(cleanup_errors)
                        + f"; original={type(failure).__name__}:{failure}",
                    )
            elif type(actor) is HeldActorProcessV1:
                _raise_after_actor_cleanup_v1(
                    (actor,), self.command_runner, failure,
                    f"{binding.actor_id} mount launch replay failed",
                )
            elif unbound_cleanup_errors:
                _fail(
                    FAIL_POLICY,
                    f"{binding.actor_id} unbound starter cleanup closure failed: "
                    + "; ".join(unbound_cleanup_errors)
                    + f"; original={type(failure).__name__}:{failure}",
                )
            raise failure
        if (
            type(actor) is not HeldActorProcessV1
            or self.active_actor_slots[ownership_slot_index] is not actor
        ):
            _fail(FAIL_POLICY, "actor mount launch returned no exact actor")
        return actor

    def _stage(
        self,
        stage_id: int,
        evidence: object,
        *,
        candidate_receipt_hex: str | None = None,
        final_receipt_hex: str | None = None,
    ) -> dict[str, object]:
        if stage_id != self.completed_stage + 1 or not 1 <= stage_id <= 10:
            _fail(FAIL_POLICY, "concrete backend stage order differs")
        if stage_id <= 7:
            qualification_count = 0
            qualification_mask = 0
            if candidate_receipt_hex is not None or final_receipt_hex is not None:
                _fail(FAIL_POLICY, "pre-candidate stage carried receipt bytes")
        elif stage_id == 8:
            qualification_count = 19
            qualification_mask = 0x7FFFF
            if candidate_receipt_hex is None or final_receipt_hex is not None:
                _fail(FAIL_POLICY, "candidate stage receipt transition differs")
        else:
            qualification_count = 20
            qualification_mask = 0xFFFFF
            if candidate_receipt_hex is None or final_receipt_hex is None:
                _fail(FAIL_POLICY, "final stage receipt transition differs")
        if stage_id >= 4:
            if (
                not self.admission_consumed
                or self.admission_boundary is None
                or self.admission_issue_record is None
                or self.admission_consumed_marker_evidence is None
                or self.admission_work_root_replay is None
                or self.admission_consume_git_source_transcript is None
                or self.admission_consume_artifact_absence is None
                or 1 not in self.admission_fresh_runtime_checkpoints
                or self.admission_work_root_descriptor is None
                or self.admission_issued_marker_descriptor is None
                or self.admission_spending_marker_descriptor is None
                or self.admission_consumed_marker_descriptor is None
                or type(evidence) is not dict
            ):
                _fail(FAIL_ACTUAL_ADMISSION, "post-admission stage lacks consumed attempt")
            live_replay = self.strict_replay_actual_admission_live_authority_v1(
                f"STAGE_{stage_id:02d}_BEFORE_EVIDENCE",
            )
            evidence = dict(evidence)
            evidence["actual_admission_attempt_id"] = self.admission_boundary[
                "attempt_id"
            ]
            evidence["actual_admission_boundary_root"] = self.admission_boundary[
                "boundary_root"
            ]
            evidence["actual_admission_issue_record_root"] = (
                self.admission_issue_record["issue_record_root"]
            )
            evidence["actual_admission_consumed_marker_evidence"] = dict(
                self.admission_consumed_marker_evidence
            )
            evidence["actual_admission_work_root_replay"] = dict(
                self.admission_work_root_replay
            )
            evidence["actual_admission_consume_git_source_transcript"] = dict(
                self.admission_consume_git_source_transcript
            )
            evidence["actual_admission_consume_artifact_absence"] = dict(
                self.admission_consume_artifact_absence
            )
            evidence["actual_admission_fresh_checkpoint_root_rows"] = [
                [
                    checkpoint_id,
                    ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[
                        checkpoint_id - 1
                    ][1],
                    checkpoint["checkpoint_root"],
                ]
                for checkpoint_id, checkpoint in sorted(
                    self.admission_fresh_runtime_checkpoints.items()
                )
            ]
            expected_mount_roles = (1, 2) if stage_id <= 5 else (1, 2, 3)
            if (
                tuple(sorted(self.actor_mount_bindings)) != expected_mount_roles
                or tuple(sorted(self.actor_mount_launch_replays))
                != expected_mount_roles
            ):
                _fail(
                    FAIL_ACTUAL_ADMISSION,
                    "post-admission actor mount root registry differs",
                )
            evidence["actual_actor_mount_binding_root_rows"] = [
                [
                    role_id,
                    ROLE_ROWS[role_id - 1][1],
                    self.actor_mount_bindings[role_id]["mount_binding_root"],
                ]
                for role_id in expected_mount_roles
            ]
            evidence["actual_actor_mount_launch_root_rows"] = [
                [
                    role_id,
                    ROLE_ROWS[role_id - 1][1],
                    self.actor_mount_launch_replays[role_id][
                        "launch_replay_root"
                    ],
                ]
                for role_id in expected_mount_roles
            ]
            evidence["actual_admission_live_marker_replay"] = live_replay
        stage_name = ACTUAL_ORCHESTRATION_STAGE_REGISTRY[stage_id - 1][1]
        value = actual_stage_evidence_v1(
            stage_id,
            stage_name,
            self.source_commit,
            evidence,
            qualification_count=qualification_count,
            qualification_mask=qualification_mask,
            candidate_receipt_hex=candidate_receipt_hex,
            final_receipt_hex=final_receipt_hex,
        )
        self.completed_stage = stage_id
        self.stage_evidence_rows[stage_id] = value
        return value

    def _create_layout_v1(self) -> dict[str, str]:
        snapshots = _private_empty_directory_v1(self.work_root, "snapshots")
        self.paths["python_snapshot"] = snapshots / "python"
        self.paths["rust_snapshot"] = snapshots / "rust"
        self.paths["host_snapshot"] = snapshots / "host"
        self.paths["cargo_home"] = self.work_root / "cargo-home"
        self.paths["target_output"] = _private_empty_directory_v1(
            self.work_root, "target-output"
        )
        self.paths["cargo_release_binary"] = (
            self.paths["target_output"]
            / "release/hegel-q1-archive-projection-oracle"
        )
        self.paths["runtime_binary_parent"] = (
            self.paths["target_output"] / "runtime-binary"
        )
        for role in ("python", "rust", "host"):
            self.paths[f"{role}_output"] = (
                _private_empty_directory_v1(self.work_root, f"{role}-output")
                if role != "host"
                else self.work_root / "host-output-unused"
            )
            self.paths[f"{role}_control"] = _private_empty_directory_v1(
                self.work_root, f"{role}-control"
            )
            self.paths[f"{role}_cid_parent"] = _private_empty_directory_v1(
                self.work_root, f"{role}-cid"
            )
            self.paths[f"{role}_cidfile"] = (
                self.paths[f"{role}_cid_parent"] / f"{role}.cid"
            )
        self.paths["host_staging"] = _private_empty_directory_v1(
            self.work_root, "host-staging"
        )
        self.paths["build_cid_parent"] = _private_empty_directory_v1(
            self.work_root, "build-cid"
        )
        self.paths["build_test_cidfile"] = self.paths["build_cid_parent"] / "test.cid"
        self.paths["build_release_cidfile"] = (
            self.paths["build_cid_parent"] / "release.cid"
        )
        self.paths["stdout_root"] = _private_empty_directory_v1(
            self.work_root, "stdout"
        )
        self.paths["binary"] = (
            self.paths["runtime_binary_parent"]
            / "hegel-q1-archive-projection-oracle"
        )
        return {name: path.as_posix() for name, path in sorted(self.paths.items())}

    def stage_01_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        verify_actual_source_commit_v1(self.project_root, self.source_commit)
        config = load_isolation_config_v1(self.project_root)
        if self.artifact_path.exists():
            _fail(FAIL_ARTIFACT, "frozen actual artifact already exists")
        cargo_status = self.cargo_cache_source.lstat()
        if (
            self.cargo_cache_source.is_symlink()
            or not stat.S_ISDIR(cargo_status.st_mode)
        ):
            _fail(FAIL_SOURCE, "external Cargo cache source differs")
        initial_name_absence_rows: list[dict[str, object]] = []
        try:
            for slot_row in self.docker_execution_slot_rows:
                name = slot_row["container_name"]
                first = _docker_absence_evidence_v1(name, self.command_runner)
                second = _docker_absence_evidence_v1(name, self.command_runner)
                initial_name_absence_rows.append(
                    _ACTUAL_ADMISSION.build_docker_initial_name_absence_row_v1(
                        self.source_commit,
                        self.admission_nonce,
                        slot_row["slot_id"],
                        first,
                        second,
                    )
                )
            authority = _ACTUAL_ADMISSION.build_docker_execution_authority_v1(
                self.source_commit,
                self.admission_nonce,
                initial_name_absence_rows,
            )
            self.docker_execution_authority = (
                _ACTUAL_ADMISSION.validate_docker_execution_authority_v1(
                    authority,
                    self.source_commit,
                    self.admission_nonce,
                )
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)
        layout = self._create_layout_v1()
        for actor_id in ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"):
            self.source_evidence[actor_id] = actor_source_evidence_v1(
                self.project_root,
                self.source_commit,
                actor_id,
            )
        allowlist_union = tuple(
            sorted(
                set(PYTHON_SOURCE_ALLOWLIST)
                | set(RUST_SOURCE_ALLOWLIST)
                | set(HOST_SOURCE_ALLOWLIST)
            )
        )
        self.source_object_closure = git_source_object_closure_evidence_v1(
            self.project_root,
            self.source_commit,
            allowlist_union,
        )
        self.image_evidence = {
            "python": local_pinned_image_evidence_v1(
                PYTHON_IMAGE,
                runner=self.command_runner,
            ),
            "rust": local_pinned_image_evidence_v1(
                RUST_IMAGE,
                runner=self.command_runner,
            ),
        }
        cross_docker_authority_to_pinned_image_labels_v1(
            self.docker_execution_authority,
            self.image_evidence,
        )
        host_seccomp = self.paths["host_snapshot"] / RUNTIME_SECCOMP_RELATIVE_PATH
        assert self.docker_execution_authority is not None
        python_slot = _docker_execution_slot_row_v1(
            self.docker_execution_authority,
            "PYTHON_ENDPOINT",
        )
        rust_slot = _docker_execution_slot_row_v1(
            self.docker_execution_authority,
            "RUST_ENDPOINT",
        )
        host_slot = _docker_execution_slot_row_v1(
            self.docker_execution_authority,
            "TRUSTED_HOST_REPLAY",
        )
        rust_test_slot = _docker_execution_slot_row_v1(
            self.docker_execution_authority,
            "RUST_TEST",
        )
        rust_release_slot = _docker_execution_slot_row_v1(
            self.docker_execution_authority,
            "RUST_RELEASE",
        )
        python_command = python_endpoint_command_v1(
            self.paths["python_snapshot"],
            self.paths["python_output"],
            self.paths["python_control"],
            host_seccomp,
            cidfile=self.paths["python_cidfile"],
            docker_slot_row=python_slot,
        )
        rust_command = rust_runtime_command_v1(
            self.paths["binary"],
            self.paths["rust_output"],
            self.paths["rust_control"],
            host_seccomp,
            cidfile=self.paths["rust_cidfile"],
            docker_slot_row=rust_slot,
        )
        host_template = trusted_host_command_v1(
            self.paths["host_snapshot"],
            self.paths["python_output"],
            self.paths["rust_output"],
            self.paths["stdout_root"] / "python.stdout",
            self.paths["stdout_root"] / "rust.stdout",
            self.paths["stdout_root"] / "manifest.json",
            self.paths["host_control"],
            self.paths["host_staging"],
            host_seccomp,
            cidfile=self.paths["host_cidfile"],
            docker_slot_row=host_slot,
        )
        rust_source = self.source_evidence["RUST_ENDPOINT"][
            "source_identity_sha256"
        ]
        test_commands = rust_build_commands_v1(
            self.paths["rust_snapshot"],
            self.paths["cargo_home"],
            self.paths["target_output"],
            rust_source,
            self.paths["build_test_cidfile"],
            build_seccomp=self.paths["host_snapshot"] / BUILD_SECCOMP_RELATIVE_PATH,
            docker_slot_row=rust_test_slot,
        )
        release_commands = rust_build_commands_v1(
            self.paths["rust_snapshot"],
            self.paths["cargo_home"],
            self.paths["target_output"],
            rust_source,
            self.paths["build_release_cidfile"],
            build_seccomp=self.paths["host_snapshot"] / BUILD_SECCOMP_RELATIVE_PATH,
            docker_slot_row=rust_release_slot,
        )
        self.planned_commands = {
            "python": python_command,
            "rust": rust_command,
            "host_template": host_template,
            "rust_test": test_commands[0],
            "rust_release": release_commands[1],
        }
        config_payload = (self.project_root / CONFIG_RELATIVE_PATH).read_bytes()
        evidence = {
            "config_hex": config_payload.hex(),
            "config_sha256": sha256(config_payload).hexdigest(),
            "fixed_artifact_path": self.artifact_path.as_posix(),
            "layout": layout,
            "cargo_cache_source": self.cargo_cache_source.resolve(strict=True).as_posix(),
            "cargo_cache_root_identity": [
                cargo_status.st_dev,
                cargo_status.st_ino,
                cargo_status.st_nlink,
                stat.S_IMODE(cargo_status.st_mode),
            ],
            "source_evidence": self.source_evidence,
            "source_object_closure": self.source_object_closure,
            "image_evidence": self.image_evidence,
            "planned_commands": self.planned_commands,
            "q1_authority": config["dry_run_authority"],
            "docker_execution_authority": self.docker_execution_authority,
        }
        return self._stage(1, evidence)

    def stage_02_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        for actor_id, key in (
            ("PYTHON_ENDPOINT", "python_snapshot"),
            ("RUST_ENDPOINT", "rust_snapshot"),
            ("TRUSTED_HOST_REPLAY", "host_snapshot"),
        ):
            materialized = materialize_actor_git_blob_snapshot_v1(
                self.project_root,
                self.source_commit,
                actor_id,
                self.paths[key],
            )
            if materialized != self.source_evidence[actor_id] | {
                "sealed_snapshot_identity": materialized["sealed_snapshot_identity"]
            }:
                # This expression deliberately checks all source fields while
                # allowing only the newly-created snapshot identity field.
                for field, value in self.source_evidence[actor_id].items():
                    if materialized.get(field) != value:
                        _fail(FAIL_SOURCE, f"snapshot source evidence differs: {actor_id}")
            self.snapshot_evidence[actor_id] = sealed_snapshot_path_evidence_v1(
                self.paths[key],
                ACTOR_SOURCE_ALLOWLISTS[actor_id],
            )
        lock_payload = _read_sealed_regular_file_v1(
            self.paths["rust_snapshot"] / "rust/q1_archive_projection_oracle/Cargo.lock",
            4 * 1024 * 1024,
            "Commit-A Cargo.lock",
        )
        self.cargo_lock_payload = lock_payload
        self.cargo_evidence = materialize_sealed_cargo_home_v1(
            lock_payload,
            self.cargo_cache_source,
            self.paths["cargo_home"],
        )
        self.cargo_evidence["root_path"] = self.paths["cargo_home"].as_posix()
        self.cargo_evidence["root_nlink"] = self.paths["cargo_home"].lstat().st_nlink
        cargo_modes = {
            row[0]: 0o555 if row[1] == 0o100755 else 0o444
            for row in self.cargo_evidence["file_rows"]
        }
        self.cargo_evidence["sealed_tree_identity"] = sealed_tree_identity_v1(
            self.paths["cargo_home"],
            tuple(row[0] for row in self.cargo_evidence["file_rows"]),
            expected_file_modes=cargo_modes,
        )
        host_blob_sha256 = {
            row[0]: row[4]
            for row in self.source_evidence["TRUSTED_HOST_REPLAY"][
                "blob_preimage_rows"
            ]
        }
        self.seccomp_evidence = {
            "runtime": sealed_policy_file_evidence_v1(
                self.paths["host_snapshot"] / RUNTIME_SECCOMP_RELATIVE_PATH,
                RUNTIME_SECCOMP_RELATIVE_PATH,
                expected_sha256=host_blob_sha256[RUNTIME_SECCOMP_RELATIVE_PATH],
            ),
            "build": sealed_policy_file_evidence_v1(
                self.paths["host_snapshot"] / BUILD_SECCOMP_RELATIVE_PATH,
                BUILD_SECCOMP_RELATIVE_PATH,
                expected_sha256=host_blob_sha256[BUILD_SECCOMP_RELATIVE_PATH],
            ),
        }
        evidence = {
            "snapshot_evidence": self.snapshot_evidence,
            "cargo_lock_hex": lock_payload.hex(),
            "cargo_lock_sha256": sha256(lock_payload).hexdigest(),
            "cargo_evidence": self.cargo_evidence,
            "seccomp_evidence": self.seccomp_evidence,
        }
        return self._stage(2, evidence)

    def stage_03_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        if self.cargo_evidence is None:
            _fail(FAIL_POLICY, "Cargo evidence is absent before Rust build")
        test_run = run_offline_rust_build_container_v1(
            self.planned_commands["rust_test"],
            self.paths["build_test_cidfile"],
            docker_execution_authority=self.docker_execution_authority,
            docker_slot="RUST_TEST",
            inspect_reader=self.inspect_reader,
            command_runner=self.command_runner,
        )
        release_run = run_offline_rust_build_container_v1(
            self.planned_commands["rust_release"],
            self.paths["build_release_cidfile"],
            docker_execution_authority=self.docker_execution_authority,
            docker_slot="RUST_RELEASE",
            inspect_reader=self.inspect_reader,
            command_runner=self.command_runner,
        )
        if (
            test_run.get("seccomp_evidence") != self.seccomp_evidence.get("build")
            or release_run.get("seccomp_evidence")
            != self.seccomp_evidence.get("build")
        ):
            _fail(FAIL_SOURCE, "offline build seccomp identity differs")
        runtime_binary_parent = _private_empty_directory_v1(
            self.paths["target_output"], "runtime-binary"
        )
        if runtime_binary_parent != self.paths["runtime_binary_parent"]:
            _fail(FAIL_SOURCE, "runtime binary parent path differs")
        self.binary_detach_evidence = detach_cargo_release_binary_v1(
            self.paths["cargo_release_binary"],
            self.paths["binary"],
        )
        self.binary_evidence = seal_prebuilt_binary_v1(self.paths["binary"])
        validate_detached_binary_binding_v1(
            self.binary_detach_evidence,
            self.binary_evidence,
            self.paths["cargo_release_binary"],
            self.paths["binary"],
        )
        rust_snapshot_replay = sealed_snapshot_path_evidence_v1(
            self.paths["rust_snapshot"],
            RUST_SOURCE_ALLOWLIST,
        )
        if rust_snapshot_replay != self.snapshot_evidence["RUST_ENDPOINT"]:
            _fail(FAIL_SOURCE, "Rust snapshot changed during offline build")
        cargo_paths = tuple(row[0] for row in self.cargo_evidence["file_rows"])
        cargo_replay = sealed_snapshot_identity_v1(
            self.paths["cargo_home"],
            cargo_paths,
        )
        if cargo_replay != self.cargo_evidence["sealed_snapshot_identity"]:
            _fail(FAIL_SOURCE, "sealed Cargo home changed during offline build")
        cargo_modes = {
            row[0]: 0o555 if row[1] == 0o100755 else 0o444
            for row in self.cargo_evidence["file_rows"]
        }
        cargo_tree_replay = sealed_tree_identity_v1(
            self.paths["cargo_home"],
            cargo_paths,
            expected_file_modes=cargo_modes,
        )
        if cargo_tree_replay != self.cargo_evidence["sealed_tree_identity"]:
            _fail(FAIL_SOURCE, "sealed Cargo tree identity changed during build")
        self.build_evidence = {
            "rust_test": test_run,
            "rust_release_build": release_run,
            "binary_detach": self.binary_detach_evidence,
            "binary": self.binary_evidence,
            "rust_snapshot_post_build": rust_snapshot_replay,
            "cargo_snapshot_post_build": cargo_replay,
            "cargo_tree_post_build": cargo_tree_replay,
        }
        return self._stage(3, self.build_evidence)

    def stage_04_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        """Launch both endpoint containers before either held lifecycle closes."""

        attempt_id = self._consume_stage3_to4_admission_boundary_v1(_context)
        if self.binary_evidence is None or self.completed_stage != 3:
            _fail(FAIL_POLICY, "endpoint launch preceded sealed Rust binary")
        runtime_seccomp = self.seccomp_evidence.get("runtime")
        if type(runtime_seccomp) is not dict:
            _fail(FAIL_SOURCE, "runtime seccomp evidence is absent")
        if self.docker_execution_authority is None:
            _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority is absent")
        python_command = python_endpoint_command_v1(
            self.paths["python_snapshot"],
            self.paths["python_output"],
            self.paths["python_control"],
            Path(runtime_seccomp["absolute_path"]),
            cidfile=self.paths["python_cidfile"],
            docker_slot_row=_docker_execution_slot_row_v1(
                self.docker_execution_authority,
                "PYTHON_ENDPOINT",
            ),
        )
        rust_command = rust_runtime_command_v1(
            self.paths["binary"],
            self.paths["rust_output"],
            self.paths["rust_control"],
            Path(runtime_seccomp["absolute_path"]),
            cidfile=self.paths["rust_cidfile"],
            docker_slot_row=_docker_execution_slot_row_v1(
                self.docker_execution_authority,
                "RUST_ENDPOINT",
            ),
        )
        if (
            python_command != self.planned_commands["python"]
            or rust_command != self.planned_commands["rust"]
        ):
            _fail(FAIL_POLICY, "endpoint command changed after source seal")
        if (
            type(self.active_mount_binding_slots) is not list
            or len(self.active_mount_binding_slots) != 3
            or any(self.active_mount_binding_slots[index] is not None for index in (0, 1))
            or type(self.active_actor_slots) is not list
            or len(self.active_actor_slots) != 3
            or any(self.active_actor_slots[index] is not None for index in (0, 1))
        ):
            _fail(FAIL_POLICY, "endpoint active ownership registry differs")
        try:
            python_binding = self._prepare_actor_mount_binding_v1(
                1,
                "PYTHON_ENDPOINT",
                python_command,
                ownership_slot_index=0,
            )
            rust_binding = self._prepare_actor_mount_binding_v1(
                2,
                "RUST_ENDPOINT",
                rust_command,
                ownership_slot_index=1,
            )
            if (
                type(python_binding) is not HeldActorMountBindingV1
                or type(rust_binding) is not HeldActorMountBindingV1
            ):
                _fail(FAIL_POLICY, "prepared endpoint mount registry differs")
            self._collect_fresh_runtime_checkpoint_v1(
                1,
                [python_binding.binding, rust_binding.binding],
            )
        except BaseException as original:
            close_errors = _close_backend_active_mount_slots_v1(self, (0, 1))
            if close_errors:
                _fail(
                    FAIL_POLICY,
                    "prepared endpoint mount authority close failed: "
                    + "; ".join(close_errors)
                    + f"; original={type(original).__name__}:{original}",
                )
            raise
        try:
            python_actor = self._launch_prepared_actor_mount_binding_v1(
                python_binding,
                "PYTHON_ENDPOINT",
                self.paths["python_cidfile"],
                self.paths["python_control"],
            )
            self.active_mount_binding_slots[0] = None
            rust_actor = self._launch_prepared_actor_mount_binding_v1(
                rust_binding,
                "RUST_ENDPOINT",
                self.paths["rust_cidfile"],
                self.paths["rust_control"],
            )
            self.active_mount_binding_slots[1] = None
            if (
                type(python_actor) is not HeldActorProcessV1
                or type(rust_actor) is not HeldActorProcessV1
                or (python_actor.actor_id, rust_actor.actor_id)
                != ("PYTHON_ENDPOINT", "RUST_ENDPOINT")
            ):
                _fail(FAIL_POLICY, "endpoint starter returned a non-exact actor set")
            for actor in (python_actor, rust_actor):
                if actor.seccomp_evidence != runtime_seccomp:
                    _fail(FAIL_SOURCE, "endpoint runtime seccomp evidence differs")
            self.endpoint_actors = (python_actor, rust_actor)
        except BaseException as error:
            close_errors = _close_backend_active_mount_slots_v1(self, (0, 1))
            if close_errors:
                error = Q05BDualSupervisorError(
                    FAIL_POLICY,
                    "parallel endpoint mount authority close failed: "
                    + "; ".join(close_errors)
                    + f"; original={type(error).__name__}:{error}",
                )
            cleanup_errors = _cleanup_backend_active_actor_slots_v1(
                self,
                (0, 1),
                self.command_runner,
            )
            if cleanup_errors:
                _fail(
                    FAIL_POLICY,
                    "parallel endpoint startup cleanup failed: "
                    + "; ".join(cleanup_errors)
                    + f"; original={type(error).__name__}:{error}",
                )
            raise error
        assert self.endpoint_actors is not None
        evidence = {
            "actor_rows": [
                {
                    "actor_id": actor.actor_id,
                    "command": list(actor.command),
                    "command_sha256": actor.mount_registry.command_sha256,
                    "mount_registry_sha256": actor.mount_registry.registry_sha256,
                    "mount_binding_root": self.actor_mount_bindings[
                        actor.role_id
                    ]["mount_binding_root"],
                    "mount_launch_replay_root": self.actor_mount_launch_replays[
                        actor.role_id
                    ]["launch_replay_root"],
                    "runtime_seccomp_evidence": actor.seccomp_evidence,
                    "docker_execution_authority_manifest_sha256": (
                        actor.docker_execution_authority_manifest_sha256
                    ),
                    "docker_execution_slot_row": actor.docker_execution_slot_row,
                    "ownership_label_root": actor.ownership_label_root,
                    "precreate_absence_evidence": actor.precreate_absence_evidence,
                    "sampling_worker_started": (
                        actor.sample_thread is not None
                        and actor.sample_thread.is_alive()
                    ),
                }
                for actor in self.endpoint_actors
            ],
            "both_endpoints_started_before_close": True,
            "consumed_admission_attempt_id": attempt_id,
            "qualification_receipt": None,
        }
        if any(row["sampling_worker_started"] is not True for row in evidence["actor_rows"]):
            _raise_after_actor_cleanup_v1(
                self.endpoint_actors,
                self.command_runner,
                Q05BDualSupervisorError(
                    FAIL_POLICY,
                    "endpoint resource sampler is not live",
                ),
                "endpoint resource sampler is not live",
            )
        return self._stage(4, evidence)

    def stage_05_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        """Close already-concurrent endpoints in frozen order.

        The not-yet-closed actor remains held with its independent sampler live.
        """

        if self.endpoint_actors is None or self.completed_stage != 4:
            _fail(FAIL_POLICY, "endpoint close preceded dual held launch")
        completed = self.actor_group_closer(
            self.endpoint_actors,
            child_timeout_seconds=30 * 60,
            inspect_reader=self.inspect_reader,
            command_runner=self.command_runner,
        )
        if (
            type(completed) is not tuple
            or len(completed) != 2
            or tuple(row.get("actor_id") for row in completed)
            != ("PYTHON_ENDPOINT", "RUST_ENDPOINT")
        ):
            _fail(FAIL_POLICY, "endpoint completion registry differs")
        if self.docker_execution_authority is None:
            _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority is absent")
        for row, slot, command_key in zip(
            completed,
            ("PYTHON_ENDPOINT", "RUST_ENDPOINT"),
            ("python", "rust"),
            strict=True,
        ):
            strict_replay_docker_completion_ownership_v1(
                row,
                self.docker_execution_authority,
                slot,
                self.planned_commands[command_key],
            )
        # The closer result is trusted only after both Docker ownership
        # transcripts replay.  Until this point the fixed active slots retain
        # the live actors for outer failure cleanup.
        self.active_actor_slots[0] = None
        self.active_actor_slots[1] = None
        python_stdout = bytes.fromhex(completed[0]["stdout_hex"])
        rust_stdout = bytes.fromhex(completed[1]["stdout_hex"])
        host_module = _load_host_replay_module_v1(self.project_root)
        python_payloads, python_tree = seal_actor_sidecar_tree_v1(
            self.paths["python_output"], host_module
        )
        rust_payloads, rust_tree = seal_actor_sidecar_tree_v1(
            self.paths["rust_output"], host_module
        )
        if python_payloads != rust_payloads:
            _fail(FAIL_POLICY, "Python/Rust neutral sidecar bytes differ")
        python_replay = host_module.replay_actor_sidecars_v1(
            "PYTHON_ENDPOINT", python_stdout, self.paths["python_output"]
        )
        rust_replay = host_module.replay_actor_sidecars_v1(
            "RUST_ENDPOINT", rust_stdout, self.paths["rust_output"]
        )
        if (
            python_replay.payloads != python_payloads
            or rust_replay.payloads != rust_payloads
            or python_replay.shadow_assembler.root
            != rust_replay.shadow_assembler.root
        ):
            _fail(FAIL_POLICY, "sealed endpoint replay differs")
        roots = (
            python_replay.leaf_manifest.manifest_root,
            python_replay.partitions[0].evidence_root,
            python_replay.partitions[1].evidence_root,
            python_replay.sidecar_manifest.manifest_root,
            python_replay.golden_manifest.manifest_root,
        )
        wire_module = importlib.import_module(
            "hegel_machine.phase3_q1_qualification_wire_v1"
        )
        self.sidecar_canonical_rows = [
            {
                "path": path.decode("ascii"),
                "mode": 0o444,
                "length": len(payload),
                "raw_sha256": sha256(payload).hexdigest(),
                "content_root": root.hex(),
                "cbor_hex": payload.hex(),
            }
            for path, payload, root in zip(
                wire_module.ORDERED_OUTPUT_RELATIVE_PATHS,
                python_payloads,
                roots,
                strict=True,
            )
        ]
        (
            python_stdout_path,
            rust_stdout_path,
            stdout_manifest_path,
            stdout_manifest,
            stdout_tree,
        ) = seal_endpoint_stdout_set_v1(
            self.paths["stdout_root"],
            python_stdout,
            rust_stdout,
            host_module,
            precreated_empty=True,
        )
        self.endpoint_complete = (completed[0], completed[1])
        self.endpoint_payloads = (python_payloads, rust_payloads)
        self.endpoint_tree_identities = (python_tree, rust_tree)
        self.endpoint_stdout = (python_stdout, rust_stdout)
        self.stdout_paths = (
            python_stdout_path,
            rust_stdout_path,
            stdout_manifest_path,
        )
        self.stdout_manifest = stdout_manifest
        self.stdout_tree_identity = stdout_tree
        evidence = {
            "actor_completion_rows": list(completed),
            "five_sidecars": {
                "canonical_rows": self.sidecar_canonical_rows,
                "python_output_tree": python_tree,
                "rust_output_tree": rust_tree,
            },
            "endpoint_stdout_set": {
                "python_stdout_hex": python_stdout.hex(),
                "rust_stdout_hex": rust_stdout.hex(),
                "manifest_hex": stdout_manifest.hex(),
                "sealed_stdout_tree": stdout_tree,
            },
            "strict_endpoint_replay_roots": [
                python_replay.host_replay_root.hex(),
                rust_replay.host_replay_root.hex(),
            ],
            "qualification_receipt": None,
        }
        stage_5 = self._stage(5, evidence)
        try:
            return _ACTUAL_ADMISSION.validate_actual_stage_5_evidence_v1(
                stage_5,
                self.source_commit,
                issue_record=self.admission_issue_record,
                consumed_marker_evidence=(
                    self.admission_consumed_marker_evidence
                ),
                checkpoint_1=self.admission_fresh_runtime_checkpoints[1],
                mount_launch_replay_rows=[
                    self.actor_mount_launch_replays[role_id]
                    for role_id in (1, 2)
                ],
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)

    def _stage_5_dynamic_mount_authority_set_v1(self) -> dict[str, object]:
        """Replay the exact concrete Stage-5 row into host mount authority."""

        if (
            self.endpoint_tree_identities is None
            or self.stdout_tree_identity is None
            or type(self.stage_evidence_rows.get(5)) is not dict
        ):
            _fail(
                FAIL_ACTUAL_ADMISSION,
                "trusted-host dynamic mount authority inputs are absent",
            )
        try:
            return _ACTUAL_ADMISSION.build_dynamic_mount_authority_set_v1(
                self.source_commit,
                self.stage_evidence_rows[5],
                self.endpoint_tree_identities[0],
                self.endpoint_tree_identities[1],
                self.stdout_tree_identity,
                issue_record=self.admission_issue_record,
                consumed_marker_evidence=(
                    self.admission_consumed_marker_evidence
                ),
                checkpoint_1=self.admission_fresh_runtime_checkpoints[1],
                mount_launch_replay_rows=[
                    self.actor_mount_launch_replays[role_id]
                    for role_id in (1, 2)
                ],
            )
        except _ACTUAL_ADMISSION.Q05BActualAdmissionError as error:
            _fail(FAIL_ACTUAL_ADMISSION, error.detail)

    def stage_06_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        """Run the third held actor and seal its semantic-only witness tree."""

        if (
            self.completed_stage != 5
            or self.endpoint_stdout is None
            or self.stdout_paths is None
            or self.stdout_manifest is None
        ):
            _fail(FAIL_POLICY, "trusted-host replay preceded sealed endpoints")
        wire_module = importlib.import_module(
            "hegel_machine.phase3_q1_qualification_wire_v1"
        )
        python_envelope = wire_module.validate_actor_stdout_envelope_v1(
            self.endpoint_stdout[0]
        )
        self.host_source_identity_root = bytes.fromhex(
            self.source_evidence["TRUSTED_HOST_REPLAY"][
                "source_identity_sha256"
            ]
        )
        self.host_runtime_identity_root = bytes.fromhex(
            python_envelope["runtime_identity_sha256"]
        )
        runtime_seccomp = self.seccomp_evidence["runtime"]
        if self.docker_execution_authority is None:
            _fail(FAIL_ACTUAL_ADMISSION, "Docker execution authority is absent")
        host_command = trusted_host_command_v1(
            self.paths["host_snapshot"],
            self.paths["python_output"],
            self.paths["rust_output"],
            self.stdout_paths[0],
            self.stdout_paths[1],
            self.stdout_paths[2],
            self.paths["host_control"],
            self.paths["host_staging"],
            Path(runtime_seccomp["absolute_path"]),
            host_source_identity_root_hex=self.host_source_identity_root.hex(),
            host_runtime_identity_root_hex=self.host_runtime_identity_root.hex(),
            cidfile=self.paths["host_cidfile"],
            docker_slot_row=_docker_execution_slot_row_v1(
                self.docker_execution_authority,
                "TRUSTED_HOST_REPLAY",
            ),
        )
        self.host_command = host_command
        dynamic_authority_set = self._stage_5_dynamic_mount_authority_set_v1()
        self.dynamic_mount_authority_set = dynamic_authority_set
        if (
            self.active_mount_binding_slots[2] is not None
            or self.active_actor_slots[2] is not None
            or self.host_actor is not None
        ):
            _fail(FAIL_POLICY, "trusted-host active ownership registry differs")
        try:
            binding = self._prepare_actor_mount_binding_v1(
                3,
                "TRUSTED_HOST_REPLAY",
                host_command,
                ownership_slot_index=2,
            )
            self._collect_fresh_runtime_checkpoint_v1(
                2,
                [binding.binding],
                dynamic_authority_set,
            )
            actor = self._launch_prepared_actor_mount_binding_v1(
                binding,
                "TRUSTED_HOST_REPLAY",
                self.paths["host_cidfile"],
                self.paths["host_control"],
            )
            self.host_actor = actor
            self.active_mount_binding_slots[2] = None
            if (
                type(actor) is not HeldActorProcessV1
                or actor.seccomp_evidence != runtime_seccomp
            ):
                _fail(FAIL_POLICY, "trusted-host starter evidence differs")
            completed = self.actor_group_closer(
                (actor,),
                child_timeout_seconds=30 * 60,
                inspect_reader=self.inspect_reader,
                command_runner=self.command_runner,
            )
            if (
                type(completed) is not tuple
                or len(completed) != 1
                or completed[0].get("actor_id") != "TRUSTED_HOST_REPLAY"
            ):
                _fail(FAIL_POLICY, "trusted-host completion registry differs")
            strict_replay_docker_completion_ownership_v1(
                completed[0],
                self.docker_execution_authority,
                "TRUSTED_HOST_REPLAY",
                host_command,
            )
        except BaseException as original:
            close_errors = _close_backend_active_mount_slots_v1(self, (2,))
            cleanup_errors = _cleanup_backend_active_actor_slots_v1(
                self,
                (2,),
                self.command_runner,
            )
            if not cleanup_errors:
                self.host_actor = None
            all_errors = close_errors + cleanup_errors
            if all_errors:
                _fail(
                    FAIL_POLICY,
                    "trusted-host active ownership cleanup failed: "
                    + "; ".join(all_errors)
                    + f"; original={type(original).__name__}:{original}",
                )
            raise
        self.active_actor_slots[2] = None
        self.host_actor = None
        self.host_complete = completed[0]
        staged_files = tuple(
            sorted(
                tuple(f"sidecars/{path}" for path in HOST_STAGED_SIDECAR_PATHS)
                + (HOST_SEMANTIC_WITNESS_RELATIVE_PATH,)
            )
        )
        self.host_staging_identity = sealed_tree_identity_v1(
            self.paths["host_staging"], staged_files
        )
        self.host_witness = _read_sealed_regular_file_v1(
            self.paths["host_staging"] / HOST_SEMANTIC_WITNESS_RELATIVE_PATH,
            16 * 1024 * 1024,
            "trusted-host semantic witness",
        )
        host_module = _load_host_replay_module_v1(self.project_root)
        witness_value = host_module.decode_host_semantic_witness_v1(
            self.host_witness
        )
        evidence = {
            "host_actor_completion": self.host_complete,
            "host_command": host_command,
            "mount_binding_root": self.actor_mount_bindings[3][
                "mount_binding_root"
            ],
            "mount_launch_replay_root": self.actor_mount_launch_replays[3][
                "launch_replay_root"
            ],
            "host_source_identity_root": self.host_source_identity_root.hex(),
            "host_runtime_identity_root": self.host_runtime_identity_root.hex(),
            "staging_tree_identity": self.host_staging_identity,
            "witness_hex": self.host_witness.hex(),
            "witness_root": witness_value["witness_root"],
            "pending_predicate_ids": witness_value["pending_predicate_ids"],
            "qualification_receipt": None,
        }
        return self._stage(6, evidence)

    def stage_07_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        """Outer issuer independently replays all semantic/preimage material."""

        if (
            self.completed_stage != 6
            or self.endpoint_stdout is None
            or self.endpoint_complete is None
            or self.endpoint_payloads is None
            or self.endpoint_tree_identities is None
            or self.stdout_manifest is None
            or self.stdout_tree_identity is None
            or self.host_complete is None
            or self.host_staging_identity is None
            or self.host_witness is None
            or self.host_source_identity_root is None
            or self.host_runtime_identity_root is None
            or self.dynamic_mount_authority_set is None
        ):
            _fail(FAIL_POLICY, "outer replay inputs are incomplete")
        if tuple(sorted(self.actor_mount_bindings)) != (1, 2, 3):
            _fail(FAIL_ACTUAL_ADMISSION, "actor mount binding registry is incomplete")
        self._collect_fresh_runtime_checkpoint_v1(
            3,
            [self.actor_mount_bindings[role_id] for role_id in (1, 2, 3)],
            self.dynamic_mount_authority_set,
        )
        live_mount_replays = [
            strict_replay_actor_completion_mount_sources_v1(
                completion,
                self.actor_mount_bindings[role_id],
                self.actor_mount_launch_replays[role_id],
            )
            for role_id, completion in zip(
                (1, 2, 3),
                (
                    self.endpoint_complete[0],
                    self.endpoint_complete[1],
                    self.host_complete,
                ),
                strict=True,
            )
        ]
        verify_actual_source_commit_v1(self.project_root, self.source_commit)
        host_module = _load_host_replay_module_v1(self.project_root)
        replay = host_module.dual_actor_host_replay_v1(
            self.endpoint_stdout[0],
            self.paths["python_output"],
            self.endpoint_stdout[1],
            self.paths["rust_output"],
            self.stdout_manifest,
            self.host_source_identity_root,
            self.host_runtime_identity_root,
        )
        negative_module = importlib.import_module(
            "hegel_machine.phase3_q05b_negative_vectors_v1"
        )
        strict_cbor_module = importlib.import_module("hegel_machine.strict_cbor_v1")
        negative_corpus = negative_module.run_q05b_negative_vector_corpus_v1()
        negative_cbor = strict_cbor_module.canonical_cbor_encode(
            negative_corpus.canonical_object()
        )
        witness_value = host_module.decode_host_semantic_witness_v1(
            self.host_witness,
            replay,
            negative_cbor,
            negative_corpus.corpus_root,
            negative_corpus.category_roots,
        )
        staged_payloads = host_module.read_exact_sidecar_tree_v1(
            self.paths["host_staging"] / HOST_STAGED_SIDECAR_ROOT
        )
        if (
            staged_payloads != self.endpoint_payloads[0]
            or self.endpoint_payloads[0] != self.endpoint_payloads[1]
        ):
            _fail(FAIL_POLICY, "trusted-host staged sidecar bytes differ")
        for actor_id in ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"):
            if actor_source_evidence_v1(
                self.project_root, self.source_commit, actor_id
            ) != self.source_evidence[actor_id]:
                _fail(FAIL_SOURCE, f"actor Git source changed: {actor_id}")
            snapshot = self.paths[
                {
                    "PYTHON_ENDPOINT": "python_snapshot",
                    "RUST_ENDPOINT": "rust_snapshot",
                    "TRUSTED_HOST_REPLAY": "host_snapshot",
                }[actor_id]
            ]
            if sealed_snapshot_path_evidence_v1(
                snapshot, ACTOR_SOURCE_ALLOWLISTS[actor_id]
            ) != self.snapshot_evidence[actor_id]:
                _fail(FAIL_SOURCE, f"actor snapshot changed: {actor_id}")
        if (
            self.source_object_closure is None
            or git_source_object_closure_evidence_v1(
                self.project_root,
                self.source_commit,
                tuple(
                    sorted(
                        set(PYTHON_SOURCE_ALLOWLIST)
                        | set(RUST_SOURCE_ALLOWLIST)
                        | set(HOST_SOURCE_ALLOWLIST)
                    )
                ),
            )
            != self.source_object_closure
        ):
            _fail(FAIL_SOURCE, "raw Git object closure changed")
        for label, relative in (
            ("runtime", RUNTIME_SECCOMP_RELATIVE_PATH),
            ("build", BUILD_SECCOMP_RELATIVE_PATH),
        ):
            prior = self.seccomp_evidence[label]
            if sealed_policy_file_evidence_v1(
                Path(prior["absolute_path"]),
                relative,
                expected_sha256=prior["payload_sha256"],
            ) != prior:
                _fail(FAIL_SOURCE, f"sealed {label} seccomp changed")
        if (
            self.binary_detach_evidence is None
            or self.binary_evidence is None
            or validate_detached_binary_binding_v1(
                self.binary_detach_evidence,
                self.binary_evidence,
                self.paths["cargo_release_binary"],
                self.paths["binary"],
            )
            != self.binary_detach_evidence
            or replay_sealed_prebuilt_binary_v1(
                self.paths["binary"], self.binary_evidence
            )
            != self.binary_evidence
        ):
            _fail(FAIL_SOURCE, "sealed Rust runtime changed")
        if self.cargo_evidence is None:
            _fail(FAIL_SOURCE, "sealed Cargo evidence is absent")
        cargo_paths = tuple(row[0] for row in self.cargo_evidence["file_rows"])
        if sealed_snapshot_identity_v1(
            self.paths["cargo_home"], cargo_paths
        ) != self.cargo_evidence["sealed_snapshot_identity"]:
            _fail(FAIL_SOURCE, "sealed Cargo home changed after actor replay")
        cargo_modes = {
            row[0]: 0o555 if row[1] == 0o100755 else 0o444
            for row in self.cargo_evidence["file_rows"]
        }
        if sealed_tree_identity_v1(
            self.paths["cargo_home"],
            cargo_paths,
            expected_file_modes=cargo_modes,
        ) != self.cargo_evidence["sealed_tree_identity"]:
            _fail(FAIL_SOURCE, "sealed Cargo tree changed after actor replay")
        for label, image in (("python", PYTHON_IMAGE), ("rust", RUST_IMAGE)):
            if local_pinned_image_evidence_v1(
                image, runner=self.command_runner
            ) != self.image_evidence[label]:
                _fail(FAIL_SOURCE, f"pinned {label} image changed")
        if (
            sealed_tree_identity_v1(
                self.paths["python_output"],
                tuple(sorted(HOST_STAGED_SIDECAR_PATHS)),
            )
            != self.endpoint_tree_identities[0]
            or sealed_tree_identity_v1(
                self.paths["rust_output"],
                tuple(sorted(HOST_STAGED_SIDECAR_PATHS)),
            )
            != self.endpoint_tree_identities[1]
            or sealed_tree_identity_v1(
                self.paths["stdout_root"],
                ("manifest.json", "python.stdout", "rust.stdout"),
                maximum_file_bytes=1024 * 1024,
            )
            != self.stdout_tree_identity
        ):
            _fail(FAIL_SOURCE, "sealed endpoint/stdout trees changed")
        staged_files = tuple(
            sorted(
                tuple(f"sidecars/{path}" for path in HOST_STAGED_SIDECAR_PATHS)
                + (HOST_SEMANTIC_WITNESS_RELATIVE_PATH,)
            )
        )
        if sealed_tree_identity_v1(
            self.paths["host_staging"], staged_files
        ) != self.host_staging_identity:
            _fail(FAIL_SOURCE, "sealed trusted-host staging changed")
        python_scratch_roots = [
            [root.hex() for root in partition.scratch_ledger_roots]
            for partition in replay.python.partition_replays
        ]
        rust_scratch_roots = [
            [root.hex() for root in partition.scratch_ledger_roots]
            for partition in replay.rust.partition_replays
        ]
        host_scratch_roots = witness_value.get("host_scratch_partition_roots")
        if (
            len(python_scratch_roots) != 2
            or len(rust_scratch_roots) != 2
            or type(host_scratch_roots) is not list
            or len(host_scratch_roots) != 2
            or any(
                type(partition) is not list or len(partition) != 4
                for partition in host_scratch_roots
            )
        ):
            _fail(FAIL_POLICY, "three-actor scratch ledger registry differs")
        scratch_rows: list[dict[str, object]] = []
        for actor_id, scratch_roots, producer_replay_root in zip(
            (
                "PYTHON_ENDPOINT",
                "RUST_ENDPOINT",
                "TRUSTED_HOST_REPLAY",
            ),
            (python_scratch_roots, rust_scratch_roots, host_scratch_roots),
            (
                replay.python.host_replay_root.hex(),
                replay.rust.host_replay_root.hex(),
                witness_value["host_scratch_evidence_root"],
            ),
            strict=True,
        ):
            preimage = {
                "actor_id": actor_id,
                "partition_scratch_ledger_roots": scratch_roots,
                "producer_replay_root": producer_replay_root,
            }
            scratch_rows.append(
                {
                    **preimage,
                    "scratch_root": sha256(
                        b"HEGEL/Q05B/ACTUAL/SCRATCH_ACTOR/V1\x00"
                        + _canonical_json_bytes(preimage)
                    ).hexdigest(),
                }
            )
        host_control = _strict_json_value_v1(
            bytes.fromhex(self.host_complete["stdout_hex"]),
            "trusted-host control stdout",
        )
        negative_roots = dict(negative_corpus.category_roots)
        evidence = {
            "semantic_replay_root": replay.dual_replay_root.hex(),
            "predicate11_semantic_component_root": (
                replay.predicate11_semantic_component_root.hex()
            ),
            "semantic_predicate_rows": [
                [predicate_id, root.hex()]
                for predicate_id, root in replay.predicate_evidence_rows
            ],
            "shadow_assembler_root": replay.shadow_assembler_root.hex(),
            "host_witness_root": witness_value["witness_root"],
            "host_loaded_module_rows": host_control["loaded_module_rows"],
            "host_loaded_module_root": host_control["loaded_module_root"],
            "negative_corpus": {
                "canonical_cbor_hex": negative_cbor.hex(),
                "corpus_root": negative_corpus.corpus_root.hex(),
                "category13_root": negative_roots[13].hex(),
                "category18_root": negative_roots[18].hex(),
            },
            "scratch_rows": scratch_rows,
            "three_actor_final_resource_rows": [
                self.endpoint_complete[0]["final_resource_transcript"],
                self.endpoint_complete[1]["final_resource_transcript"],
                self.host_complete["final_resource_transcript"],
            ],
            "three_actor_control_rows": [
                self.endpoint_complete[0],
                self.endpoint_complete[1],
                self.host_complete,
            ],
            "three_actor_live_mount_replays": live_mount_replays,
            "pending_predicate_ids": list(witness_value["pending_predicate_ids"]),
            "q1_authority": {
                "certificate_active": False,
                "formal_output_roots": [None] * 8,
                "gate_count": 0,
                "gate_mask": 0,
                "state": "NOT_RUN",
            },
            "qualification_receipt": None,
        }
        verify_actual_source_commit_v1(self.project_root, self.source_commit)
        self.outer_replay = replay
        self.outer_stage07_evidence = evidence
        return self._stage(7, evidence)

    def _artifact_actor_source_row_v1(self, actor_id: str) -> dict[str, object]:
        source = self.source_evidence[actor_id]
        blob_rows = [list(row[:5]) for row in source["blob_preimage_rows"]]
        return {
            "actor_id": actor_id,
            "allowlist_count": len(blob_rows),
            "blob_rows": blob_rows,
            "commit": self.source_commit,
            "git_blob_manifest_sha256": sha256(
                _canonical_json_bytes(blob_rows)
            ).hexdigest(),
            "path_registry_sha256": source["path_registry_sha256"],
            "source_identity_sha256": source["source_identity_sha256"],
        }

    def _artifact_source_wire_profile_v1(self) -> dict[str, object]:
        if (
            self.outer_replay is None
            or self.source_object_closure is None
            or self.sidecar_canonical_rows is None
        ):
            _fail(FAIL_ARTIFACT, "artifact source profile inputs are incomplete")
        payload_by_path: dict[str, list[object]] = {}
        for actor_id in (
            "PYTHON_ENDPOINT",
            "RUST_ENDPOINT",
            "TRUSTED_HOST_REPLAY",
        ):
            for row in self.source_evidence[actor_id]["blob_preimage_rows"]:
                candidate = [row[0], row[1], row[2], row[5]]
                prior = payload_by_path.get(row[0])
                if prior is not None and prior != candidate:
                    _fail(FAIL_SOURCE, "actor Git blob preimages disagree")
                payload_by_path[row[0]] = candidate
        wire_module = importlib.import_module(
            "hegel_machine.phase3_q1_qualification_wire_v1"
        )
        leaf = self.outer_replay.python.leaf_manifest
        golden = self.outer_replay.python.golden_manifest
        return {
            "source_commit": self.source_commit,
            "source_commit_raw20_hex": self.source_commit,
            "full_leaf_manifest_root": leaf.manifest_root.hex(),
            "q0_receipt_root": (
                wire_module.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION.hex()
            ),
            "q1_projection_profile_root": golden.q1_projection_profile_root.hex(),
            "q1_semantic_binding_root": golden.q1_semantic_binding_root.hex(),
            "qualification_predicate_registry_root": (
                wire_module.QUALIFICATION_PREDICATE_REGISTRY_ROOT.hex()
            ),
            "qualification_tag_registry_root": (
                wire_module.QUALIFICATION_TAG_REGISTRY_ROOT.hex()
            ),
            "qualification_wire_profile_root": (
                wire_module.qualification_wire_profile_root_v1().hex()
            ),
            "actor_source_path_rows": [
                [actor_id, list(ACTOR_SOURCE_ALLOWLISTS[actor_id])]
                for actor_id in (
                    "PYTHON_ENDPOINT",
                    "RUST_ENDPOINT",
                    "TRUSTED_HOST_REPLAY",
                )
            ],
            "git_blob_payload_table": [
                payload_by_path[path] for path in sorted(payload_by_path)
            ],
            "git_commit_object_hex": self.source_object_closure[
                "commit_payload_hex"
            ],
            "git_tree_object_rows": self.source_object_closure[
                "tree_object_rows"
            ],
            "project_tree_prefix": self.source_object_closure[
                "project_tree_prefix"
            ],
            "external_commit_replay": {
                "commit": self.source_commit,
                "head_clean_before": True,
                "head_clean_after": True,
                "tree_oid": self.source_object_closure["root_tree_object_id"],
            },
            "pinned_image_rows": [
                ["python", self.image_evidence["python"]],
                ["rust", self.image_evidence["rust"]],
            ],
        }

    def _artifact_actor_rows_v1(self) -> list[dict[str, object]]:
        if (
            self.endpoint_complete is None
            or self.host_complete is None
            or self.host_command is None
            or self.endpoint_stdout is None
            or self.binary_evidence is None
        ):
            _fail(FAIL_ARTIFACT, "artifact actor inputs are incomplete")
        wire_module = importlib.import_module(
            "hegel_machine.phase3_q1_qualification_wire_v1"
        )
        python_envelope = wire_module.validate_actor_stdout_envelope_v1(
            self.endpoint_stdout[0]
        )
        rust_envelope = wire_module.validate_actor_stdout_envelope_v1(
            self.endpoint_stdout[1]
        )
        rows: list[dict[str, object]] = []
        for actor_id, command, control, runtime_identity in (
            (
                "PYTHON_ENDPOINT",
                self.planned_commands["python"],
                self.endpoint_complete[0],
                python_envelope["runtime_identity_sha256"],
            ),
            (
                "RUST_ENDPOINT",
                self.planned_commands["rust"],
                self.endpoint_complete[1],
                rust_envelope["runtime_identity_sha256"],
            ),
            (
                "TRUSTED_HOST_REPLAY",
                self.host_command,
                self.host_complete,
                self.host_runtime_identity_root.hex(),
            ),
        ):
            rows.append(
                {
                    "actor_id": actor_id,
                    "command": list(command),
                    "control_evidence": control,
                    "runtime_identity_sha256": runtime_identity,
                    "snapshot_identity": self.snapshot_evidence[actor_id],
                    "source_evidence": self._artifact_actor_source_row_v1(
                        actor_id
                    ),
                }
            )
        return rows

    def _artifact_cargo_build_binary_v1(self) -> dict[str, object]:
        if (
            self.cargo_evidence is None
            or self.cargo_lock_payload is None
            or self.build_evidence is None
            or self.binary_detach_evidence is None
            or self.binary_evidence is None
        ):
            _fail(FAIL_ARTIFACT, "artifact Cargo inputs are incomplete")
        if (
            self.build_evidence.get("binary_detach")
            != self.binary_detach_evidence
            or validate_detached_binary_binding_v1(
                self.binary_detach_evidence,
                self.binary_evidence,
                self.paths["cargo_release_binary"],
                self.paths["binary"],
            )
            != self.binary_detach_evidence
        ):
            _fail(FAIL_ARTIFACT, "artifact binary detach binding differs")
        binary = bytes.fromhex(self.binary_evidence["payload_hex"])
        runtime_identity = sha256(
            b"HEGEL/Q05B/RUST_RUNTIME_IDENTITY/V1\x00"
            + len(binary).to_bytes(8, "big")
            + binary
        ).hexdigest()
        binary_identity = {
            "path": self.binary_evidence["binary_path"],
            "device": self.binary_evidence["device"],
            "inode": self.binary_evidence["inode"],
            "nlink": self.binary_evidence["nlink"],
            "uid": self.binary_evidence["uid"],
            "gid": self.binary_evidence["gid"],
            "mode": self.binary_evidence["mode"],
            "size": self.binary_evidence["size"],
            "mtime_ns": self.binary_evidence["mtime_ns"],
            "ctime_ns": self.binary_evidence["ctime_ns"],
            "sha256": self.binary_evidence["sha256"],
        }
        return {
            "schema_version": "hegel-phase3a-q05b-cargo-build-binary-evidence/1",
            "lock_hex": self.cargo_lock_payload.hex(),
            "locked_packages": self.cargo_evidence["locked_packages"],
            "sealed_cargo_files": self.cargo_evidence["file_preimage_rows"],
            "sealed_cargo_manifest_sha256": self.cargo_evidence[
                "manifest_sha256"
            ],
            "sealed_cargo_tree": self.cargo_evidence["sealed_tree_identity"],
            "cargo_snapshot_post_build": self.build_evidence[
                "cargo_tree_post_build"
            ],
            "rust_snapshot_post_build": self.build_evidence[
                "rust_snapshot_post_build"
            ],
            "rust_test": self.build_evidence["rust_test"],
            "rust_release_build": self.build_evidence["rust_release_build"],
            "binary_detach_evidence": self.binary_detach_evidence,
            "rust_image_inspect_hex": self.image_evidence["rust"][
                "raw_inspect_hex"
            ],
            "rust_image_inspect_sha256": self.image_evidence["rust"][
                "raw_inspect_sha256"
            ],
            "target_output_root_path": self.paths["target_output"].as_posix(),
            "binary_path": self.binary_evidence["binary_path"],
            "binary_hex": binary.hex(),
            "binary_sha256": self.binary_evidence["sha256"],
            "binary_runtime_identity_sha256": runtime_identity,
            "binary_file_identity": binary_identity,
        }

    def actual_artifact_evidence_sections_v1(self) -> dict[str, object]:
        """Map production evidence into the pure artifact replayer schema."""

        if (
            self.completed_stage != 7
            or self.sidecar_canonical_rows is None
            or self.endpoint_tree_identities is None
            or self.endpoint_stdout is None
            or self.stdout_manifest is None
            or self.stdout_tree_identity is None
            or self.host_staging_identity is None
            or self.host_witness is None
            or self.endpoint_complete is None
            or self.host_complete is None
            or self.outer_replay is None
            or self.admission_issue_record is None
            or self.admission_consumed_marker_evidence is None
            or self.admission_work_root_replay is None
            or self.admission_consume_git_source_transcript is None
            or self.admission_consume_artifact_absence is None
            or tuple(sorted(self.admission_fresh_runtime_checkpoints))
            != (1, 2, 3)
            or any(
                stage_id not in self.stage_evidence_rows
                for stage_id in (1, 2, 3, 5)
            )
        ):
            _fail(FAIL_ARTIFACT, "artifact section assembly preceded stage 7")
        artifact_module = importlib.import_module(
            "hegel_machine.phase3_q05b_actual_artifact_v1"
        )
        source = self._artifact_source_wire_profile_v1()
        five_sidecars = {
            "canonical_rows": self.sidecar_canonical_rows,
            "python_output_tree": self.endpoint_tree_identities[0],
            "rust_output_tree": self.endpoint_tree_identities[1],
        }
        endpoint = {
            "python_stdout_hex": self.endpoint_stdout[0].hex(),
            "rust_stdout_hex": self.endpoint_stdout[1].hex(),
            "manifest_hex": self.stdout_manifest.hex(),
            "sealed_stdout_tree": self.stdout_tree_identity,
        }
        config_path = "config/phase3_q05b_dual_isolation_v1.json"
        config_rows = [
            row for row in source["git_blob_payload_table"]
            if row[0] == config_path
        ]
        if len(config_rows) != 1:
            _fail(FAIL_ARTIFACT, "artifact admission config blob differs")
        config_row = config_rows[0]
        pre_artifact_live = self.strict_replay_actual_admission_live_authority_v1(
            "PRE_ARTIFACT_ASSEMBLY"
        )
        actual_admission = (
            artifact_module.build_actual_admission_artifact_evidence_v1(
                source_commit=self.source_commit,
                artifact_path=self.artifact_path.as_posix(),
                commit_a_config_bytes=bytes.fromhex(config_row[3]),
                commit_a_config_git_blob_oid=config_row[2],
                prior_stage_evidence_rows=[
                    self.stage_evidence_rows[stage_id]
                    for stage_id in (1, 2, 3)
                ],
                issue_record=self.admission_issue_record,
                consumed_marker_evidence=(
                    self.admission_consumed_marker_evidence
                ),
                consume_work_root_replay=self.admission_work_root_replay,
                consume_git_source_transcript=(
                    self.admission_consume_git_source_transcript
                ),
                consume_artifact_absence_evidence=(
                    self.admission_consume_artifact_absence
                ),
                fresh_runtime_checkpoint_rows=[
                    self.admission_fresh_runtime_checkpoints[checkpoint_id]
                    for checkpoint_id in (1, 2, 3)
                ],
                pre_artifact_live_marker_replay=pre_artifact_live,
                anti_replay_scope=ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE,
                stage_5_evidence=self.stage_evidence_rows[5],
                stage_5_actor_completion_rows=[
                    self.endpoint_complete[0],
                    self.endpoint_complete[1],
                ],
                stage_5_strict_endpoint_replay_roots=[
                    self.outer_replay.python.host_replay_root.hex(),
                    self.outer_replay.rust.host_replay_root.hex(),
                ],
                stage_5_live_marker_replay=self.stage_evidence_rows[5][
                    "evidence"
                ]["actual_admission_live_marker_replay"],
                stage_5_mount_launch_replay_rows=[
                    self.actor_mount_launch_replays[role_id]
                    for role_id in (1, 2)
                ],
                five_sidecars=five_sidecars,
                endpoint_stdout_set=endpoint,
            )
        )
        actors = self._artifact_actor_rows_v1()
        resources = [
            self.endpoint_complete[0]["final_resource_transcript"],
            self.endpoint_complete[1]["final_resource_transcript"],
            self.host_complete["final_resource_transcript"],
        ]
        host_control_payload = bytes.fromhex(self.host_complete["stdout_hex"])
        host_control = _strict_json_value_v1(
            host_control_payload, "trusted-host artifact control"
        )
        decoded_witness = _load_host_replay_module_v1(
            self.project_root
        ).decode_host_semantic_witness_v1(self.host_witness)
        staged_rows = [
            [
                row["path"],
                row["length"],
                row["raw_sha256"],
                row["mode"],
            ]
            for row in self.sidecar_canonical_rows
        ]
        host_binding = {
            "host_actor_row": actors[2],
            "host_control_sha256": sha256(host_control_payload).hexdigest(),
            "host_final_resource": resources[2],
            "loaded_module_root": host_control["loaded_module_root"],
            "semantic_replay_root": self.outer_replay.dual_replay_root.hex(),
            "witness_root": decoded_witness["witness_root"],
        }
        host_stage = {
            "host_control_stdout_hex": host_control_payload.hex(),
            "host_execution_binding_preimage": host_binding,
            "loaded_module_root": host_control["loaded_module_root"],
            "loaded_module_rows": host_control["loaded_module_rows"],
            "staged_sidecar_rows": staged_rows,
            "staging_tree": self.host_staging_identity,
            "witness_hex": self.host_witness.hex(),
            "witness_root": decoded_witness["witness_root"],
        }
        negative = self.outer_stage07_evidence["negative_corpus"]
        scratch = self.outer_stage07_evidence["scratch_rows"]
        cargo = self._artifact_cargo_build_binary_v1()
        resource_preimage = {"final_resource_rows": resources}
        isolation_preimage = {
            "actual_admission": actual_admission,
            "actor_rows": actors,
            "cargo_build_binary": cargo,
            "endpoint_stdout_set": endpoint,
            "final_resource_rows": resources,
            "five_sidecars": five_sidecars,
            "host_stage": host_stage,
            "negative_corpus": negative,
            "scratch_rows": scratch,
            "source_wire_profile": source,
        }
        bundle_preimage = {
            "actual_admission_evidence_root": actual_admission[
                "actual_admission_evidence_root"
            ],
            "five_sidecars": five_sidecars,
            "host_witness_root": decoded_witness["witness_root"],
            "scratch_rows": scratch,
            "semantic_replay_root": self.outer_replay.dual_replay_root.hex(),
        }
        semantic = {
            "bundle_preimage": bundle_preimage,
            "host_execution_binding_preimage": host_binding,
            "isolation_preimage": isolation_preimage,
            "resource_preimage": resource_preimage,
            "semantic_component_root": (
                self.outer_replay.predicate11_semantic_component_root.hex()
            ),
        }
        return {
            "source_wire_profile": source,
            "five_sidecars": five_sidecars,
            "endpoint_stdout_set": endpoint,
            "host_stage": host_stage,
            "actor_rows": actors,
            "cargo_build_binary": cargo,
            "final_resource_rows": resources,
            "negative_corpus": negative,
            "scratch_rows": scratch,
            "actual_admission": actual_admission,
            "semantic_execution": semantic,
        }

    def stage_08_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        """Strictly replay every embedded preimage and freeze only rows 1--19."""

        if self.completed_stage != 7:
            _fail(FAIL_POLICY, "candidate replay preceded stage 7 closure")
        try:
            artifact_module = importlib.import_module(
                "hegel_machine.phase3_q05b_actual_artifact_v1"
            )
            sections = self.actual_artifact_evidence_sections_v1()
            candidate = artifact_module.replay_actual_evidence_1_19_v1(
                sections
            )
        except Exception as error:
            _fail(FAIL_ARTIFACT, f"candidate evidence strict replay failed: {error}")
        try:
            candidate = artifact_module.validate_stage8_candidate_registry_v1(
                candidate,
                sections["actual_admission"][
                    "actual_admission_evidence_root"
                ],
            )
        except Exception as error:
            _fail(
                FAIL_ARTIFACT,
                f"candidate Stage-8 adapter validation failed: {error}",
            )
        self.actual_evidence_sections = sections
        self.candidate_replay = candidate
        evidence = {
            "actual_admission_evidence_root": candidate[
                "actual_admission_evidence_root"
            ],
            "artifact_section_names": list(artifact_module.SECTION_NAMES),
            "bundle_evidence_root": candidate["bundle_evidence_root"],
            "candidate_receipt_root": candidate["candidate_receipt_root"],
            "closed_q1_authority": candidate["closed_q1_authority"],
            "host_execution_binding_root": candidate[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": candidate["isolation_evidence_root"],
            "ordered_predicate_rows": candidate["ordered_predicate_rows"],
            "resource_evidence_root": candidate["resource_evidence_root"],
            "strict_evidence_replay_equal": True,
        }
        return self._stage(
            8,
            evidence,
            candidate_receipt_hex=candidate["candidate_receipt_cbor_hex"],
        )

    def stage_09_v1(self, _context: Mapping[str, object]) -> dict[str, object]:
        """Add predicate 20 and build the complete artifact only in memory."""

        if (
            self.completed_stage != 8
            or self.actual_evidence_sections is None
            or self.candidate_replay is None
        ):
            _fail(FAIL_POLICY, "final artifact build preceded candidate replay")
        try:
            artifact_module = importlib.import_module(
                "hegel_machine.phase3_q05b_actual_artifact_v1"
            )
            artifact = artifact_module.build_actual_artifact_v1(
                self.actual_evidence_sections
            )
        except Exception as error:
            _fail(FAIL_ARTIFACT, f"final artifact build failed: {error}")
        if type(artifact) is not dict or type(artifact.get("derived")) is not dict:
            _fail(FAIL_ARTIFACT, "final artifact derived registry differs")
        try:
            derived = artifact_module.validate_stage9_derived_registry_v1(
                artifact["derived"],
                self.candidate_replay,
                self.actual_evidence_sections["actual_admission"][
                    "actual_admission_evidence_root"
                ],
            )
        except Exception as error:
            _fail(
                FAIL_ARTIFACT,
                f"final Stage-9 adapter validation failed: {error}",
            )
        self.actual_artifact_value = artifact
        evidence = {
            "actual_admission_evidence_root": derived[
                "actual_admission_evidence_root"
            ],
            "artifact_set_root": derived["artifact_set_root"],
            "bundle_evidence_root": derived["bundle_evidence_root"],
            "candidate_receipt_root": derived["candidate_receipt_root"],
            "closed_q1_authority": derived["closed_q1_authority"],
            "final_receipt_root": derived["final_receipt_root"],
            "host_execution_binding_root": derived[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": derived["isolation_evidence_root"],
            "ordered_predicate_rows": derived["ordered_predicate_rows"],
            "predicate20_added_after_candidate_replay": True,
            "qualification_count": 20,
            "qualification_mask": 0xFFFFF,
            "resource_evidence_root": derived["resource_evidence_root"],
            "strict_derived_cross_equal": True,
        }
        return self._stage(
            9,
            evidence,
            candidate_receipt_hex=derived["candidate_receipt_cbor_hex"],
            final_receipt_hex=derived["final_receipt_cbor_hex"],
        )

    def canonical_artifact_value_v1(
        self,
        context: Mapping[str, object],
    ) -> dict[str, object]:
        if (
            self.completed_stage not in (9, 10)
            or self.actual_artifact_value is None
            or type(context) is not dict
            or type(context.get("stage_09")) is not dict
        ):
            _fail(FAIL_ARTIFACT, "canonical artifact request preceded stage 9")
        return _strict_json_value_v1(
            _canonical_json_bytes(self.actual_artifact_value),
            "in-memory actual artifact",
        )

    def strict_replay_artifact_value_v1(
        self,
        value: object,
        context: Mapping[str, object],
    ) -> dict[str, object]:
        if (
            self.completed_stage not in (9, 10)
            or self.actual_artifact_value is None
            or value != self.actual_artifact_value
            or type(context) is not dict
            or type(context.get("stage_09")) is not dict
        ):
            _fail(FAIL_ARTIFACT, "artifact strict replay request differs")
        try:
            artifact_module = importlib.import_module(
                "hegel_machine.phase3_q05b_actual_artifact_v1"
            )
            replayed = artifact_module.decode_and_replay_actual_artifact_v1(
                _canonical_json_bytes(value)
            )
        except Exception as error:
            _fail(FAIL_ARTIFACT, f"complete artifact strict replay failed: {error}")
        if replayed != value:
            _fail(FAIL_ARTIFACT, "complete artifact strict replay differs")
        self.actual_artifact_replay = replayed
        return replayed

    def stage_10_v1(self, context: Mapping[str, object]) -> dict[str, object]:
        """Bind the replayed artifact; publication remains outside this stage."""

        if (
            self.completed_stage != 9
            or self.actual_artifact_value is None
            or self.actual_artifact_replay != self.actual_artifact_value
            or self.actual_evidence_sections is None
            or self.candidate_replay is None
            or type(context) is not dict
            or type(context.get("stage_09")) is not dict
            or context.get("strict_replayed_artifact")
            != self.actual_artifact_value
        ):
            _fail(FAIL_ARTIFACT, "stage 10 preceded complete artifact replay")
        try:
            artifact_module = importlib.import_module(
                "hegel_machine.phase3_q05b_actual_artifact_v1"
            )
            payload = artifact_module.canonical_actual_artifact_bytes_v1(
                self.actual_artifact_value
            )
            derived = artifact_module.validate_stage9_derived_registry_v1(
                self.actual_artifact_value["derived"],
                self.candidate_replay,
                self.actual_evidence_sections["actual_admission"][
                    "actual_admission_evidence_root"
                ],
            )
        except Exception as error:
            _fail(FAIL_ARTIFACT, f"Stage-10 artifact/derived replay failed: {error}")
        digest = sha256(payload).hexdigest()
        if context.get("strict_replayed_artifact_sha256") != digest:
            _fail(FAIL_ARTIFACT, "stage 10 canonical artifact digest differs")
        stage9 = context["stage_09"]
        stage9_evidence = stage9.get("evidence")
        expected_stage9_evidence = {
            "actual_admission_evidence_root": derived[
                "actual_admission_evidence_root"
            ],
            "artifact_set_root": derived["artifact_set_root"],
            "bundle_evidence_root": derived["bundle_evidence_root"],
            "candidate_receipt_root": derived["candidate_receipt_root"],
            "closed_q1_authority": derived["closed_q1_authority"],
            "final_receipt_root": derived["final_receipt_root"],
            "host_execution_binding_root": derived[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": derived["isolation_evidence_root"],
            "ordered_predicate_rows": derived["ordered_predicate_rows"],
            "predicate20_added_after_candidate_replay": True,
            "qualification_count": 20,
            "qualification_mask": 0xFFFFF,
            "resource_evidence_root": derived["resource_evidence_root"],
            "strict_derived_cross_equal": True,
        }
        if (
            type(stage9_evidence) is not dict
            or any(
                stage9_evidence.get(name) != expected
                for name, expected in expected_stage9_evidence.items()
            )
            or stage9.get("candidate_receipt_hex")
            != derived["candidate_receipt_cbor_hex"]
            or stage9.get("final_receipt_hex")
            != derived["final_receipt_cbor_hex"]
            or type(stage9.get("qualification_count")) is not int
            or stage9["qualification_count"] != 20
            or type(stage9.get("qualification_mask")) is not int
            or stage9["qualification_mask"] != 0xFFFFF
        ):
            _fail(FAIL_ARTIFACT, "Stage-10 differs from Stage-9 adapter")
        evidence = {
            "actual_admission_evidence_root": derived[
                "actual_admission_evidence_root"
            ],
            "artifact_length": len(payload),
            "artifact_set_root": derived["artifact_set_root"],
            "artifact_sha256": digest,
            "bundle_evidence_root": derived["bundle_evidence_root"],
            "candidate_receipt_hex": derived["candidate_receipt_cbor_hex"],
            "candidate_receipt_root": derived["candidate_receipt_root"],
            "closed_q1_authority": derived["closed_q1_authority"],
            "final_receipt_hex": derived["final_receipt_cbor_hex"],
            "final_receipt_root": derived["final_receipt_root"],
            "host_execution_binding_root": derived[
                "host_execution_binding_root"
            ],
            "isolation_evidence_root": derived["isolation_evidence_root"],
            "ordered_predicate_rows": derived["ordered_predicate_rows"],
            "qualification_count": 20,
            "qualification_mask": 0xFFFFF,
            "resource_evidence_root": derived["resource_evidence_root"],
            "strict_replay_equal": True,
        }
        return self._stage(
            10,
            evidence,
            candidate_receipt_hex=derived["candidate_receipt_cbor_hex"],
            final_receipt_hex=derived["final_receipt_cbor_hex"],
        )


def _validate_actual_stage8_to9_adapter_v1(
    context: Mapping[str, object],
    artifact_value: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], bytes]:
    """Independently replay the artifact and close Stage 8/9 projections."""

    if type(context) is not dict:
        _fail(FAIL_ARTIFACT, "Stage8/9 adapter context differs")
    try:
        artifact_module = importlib.import_module(
            "hegel_machine.phase3_q05b_actual_artifact_v1"
        )
        payload = artifact_module.canonical_actual_artifact_bytes_v1(
            artifact_value
        )
        artifact = artifact_module.decode_and_replay_actual_artifact_v1(payload)
        if artifact != artifact_value:
            _fail(FAIL_ARTIFACT, "independent artifact replay differs")
        sections = artifact["sections"]
        admission_root = sections["actual_admission"][
            "actual_admission_evidence_root"
        ]
        candidate = artifact_module.replay_actual_evidence_1_19_v1(sections)
        candidate = artifact_module.validate_stage8_candidate_registry_v1(
            candidate, admission_root
        )
        derived = artifact_module.validate_stage9_derived_registry_v1(
            artifact["derived"], candidate, admission_root
        )
    except Q05BDualSupervisorError:
        raise
    except Exception as error:
        _fail(FAIL_ARTIFACT, f"independent Stage8/9 replay failed: {error}")

    stage8 = context.get("stage_08")
    stage9 = context.get("stage_09")
    if type(stage8) is not dict or type(stage9) is not dict:
        _fail(FAIL_ARTIFACT, "Stage8/9 evidence rows are absent")
    stage8_evidence = stage8.get("evidence")
    stage8_projection = {
        "actual_admission_evidence_root": candidate[
            "actual_admission_evidence_root"
        ],
        "artifact_section_names": list(artifact_module.SECTION_NAMES),
        "bundle_evidence_root": candidate["bundle_evidence_root"],
        "candidate_receipt_root": candidate["candidate_receipt_root"],
        "closed_q1_authority": candidate["closed_q1_authority"],
        "host_execution_binding_root": candidate[
            "host_execution_binding_root"
        ],
        "isolation_evidence_root": candidate["isolation_evidence_root"],
        "ordered_predicate_rows": candidate["ordered_predicate_rows"],
        "resource_evidence_root": candidate["resource_evidence_root"],
        "strict_evidence_replay_equal": True,
    }
    if (
        type(stage8_evidence) is not dict
        or any(
            _canonical_json_bytes(stage8_evidence.get(name))
            != _canonical_json_bytes(expected)
            for name, expected in stage8_projection.items()
        )
        or type(stage8.get("qualification_count")) is not int
        or stage8["qualification_count"] != 19
        or type(stage8.get("qualification_mask")) is not int
        or stage8["qualification_mask"] != 0x7FFFF
        or stage8.get("candidate_receipt_hex")
        != candidate["candidate_receipt_cbor_hex"]
        or stage8.get("final_receipt_hex") is not None
    ):
        _fail(FAIL_ARTIFACT, "Stage-8 row differs from strict candidate replay")

    stage9_evidence = stage9.get("evidence")
    stage9_projection = {
        "actual_admission_evidence_root": derived[
            "actual_admission_evidence_root"
        ],
        "artifact_set_root": derived["artifact_set_root"],
        "bundle_evidence_root": derived["bundle_evidence_root"],
        "candidate_receipt_root": derived["candidate_receipt_root"],
        "closed_q1_authority": derived["closed_q1_authority"],
        "final_receipt_root": derived["final_receipt_root"],
        "host_execution_binding_root": derived[
            "host_execution_binding_root"
        ],
        "isolation_evidence_root": derived["isolation_evidence_root"],
        "ordered_predicate_rows": derived["ordered_predicate_rows"],
        "predicate20_added_after_candidate_replay": True,
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "resource_evidence_root": derived["resource_evidence_root"],
        "strict_derived_cross_equal": True,
    }
    if (
        type(stage9_evidence) is not dict
        or any(
            _canonical_json_bytes(stage9_evidence.get(name))
            != _canonical_json_bytes(expected)
            for name, expected in stage9_projection.items()
        )
        or type(stage9.get("qualification_count")) is not int
        or stage9["qualification_count"] != 20
        or type(stage9.get("qualification_mask")) is not int
        or stage9["qualification_mask"] != 0xFFFFF
        or stage9.get("candidate_receipt_hex")
        != derived["candidate_receipt_cbor_hex"]
        or stage9.get("final_receipt_hex")
        != derived["final_receipt_cbor_hex"]
    ):
        _fail(FAIL_ARTIFACT, "Stage-9 row differs from strict derived replay")
    return artifact, candidate, derived, payload


def _validate_actual_stage10_adapter_v1(
    stage10: object,
    stage9: Mapping[str, object],
    derived: Mapping[str, object],
    artifact_payload: bytes,
) -> dict[str, object]:
    if type(stage10) is not dict or type(stage9) is not dict:
        _fail(FAIL_ARTIFACT, "Stage-10 adapter rows differ")
    try:
        wire_module = importlib.import_module(
            "hegel_machine.phase3_q1_qualification_wire_v1"
        )
        candidate_hex = stage10.get("candidate_receipt_hex")
        final_hex = stage10.get("final_receipt_hex")
        if (
            type(candidate_hex) is not str
            or not candidate_hex
            or len(candidate_hex) % 2
            or re.fullmatch(r"[0-9a-f]+", candidate_hex) is None
            or type(final_hex) is not str
            or not final_hex
            or len(final_hex) % 2
            or re.fullmatch(r"[0-9a-f]+", final_hex) is None
        ):
            _fail(FAIL_ARTIFACT, "Stage-10 receipt hex differs")
        candidate_payload = bytes.fromhex(candidate_hex)
        final_payload = bytes.fromhex(final_hex)
        decoded_candidate = wire_module.decode_qualification_candidate_receipt_v1(
            candidate_payload
        )
        decoded_final = wire_module.decode_qualification_receipt_v1(final_payload)
    except Q05BDualSupervisorError:
        raise
    except Exception as error:
        _fail(FAIL_ARTIFACT, f"Stage-10 receipt strict replay failed: {error}")
    if (
        decoded_candidate.canonical_bytes != candidate_payload
        or decoded_final.canonical_bytes != final_payload
        or decoded_final.candidate_receipt.canonical_bytes != candidate_payload
        or decoded_candidate.receipt_root.hex()
        != derived.get("candidate_receipt_root")
        or decoded_final.receipt_root.hex() != derived.get("final_receipt_root")
    ):
        _fail(FAIL_ARTIFACT, "Stage-10 receipt/root binding differs")
    stage9_evidence = stage9.get("evidence")
    expected_stage9 = {
        "actual_admission_evidence_root": derived[
            "actual_admission_evidence_root"
        ],
        "artifact_set_root": derived["artifact_set_root"],
        "bundle_evidence_root": derived["bundle_evidence_root"],
        "candidate_receipt_root": derived["candidate_receipt_root"],
        "closed_q1_authority": derived["closed_q1_authority"],
        "final_receipt_root": derived["final_receipt_root"],
        "host_execution_binding_root": derived[
            "host_execution_binding_root"
        ],
        "isolation_evidence_root": derived["isolation_evidence_root"],
        "ordered_predicate_rows": derived["ordered_predicate_rows"],
        "predicate20_added_after_candidate_replay": True,
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "resource_evidence_root": derived["resource_evidence_root"],
        "strict_derived_cross_equal": True,
    }
    if (
        type(stage9_evidence) is not dict
        or any(
            _canonical_json_bytes(stage9_evidence.get(name))
            != _canonical_json_bytes(value)
            for name, value in expected_stage9.items()
        )
    ):
        _fail(FAIL_ARTIFACT, "Stage-10 differs from Stage-9 derived registry")
    evidence = stage10.get("evidence")
    expected = {
        "actual_admission_evidence_root": derived[
            "actual_admission_evidence_root"
        ],
        "artifact_length": len(artifact_payload),
        "artifact_set_root": derived["artifact_set_root"],
        "artifact_sha256": sha256(artifact_payload).hexdigest(),
        "bundle_evidence_root": derived["bundle_evidence_root"],
        "candidate_receipt_hex": derived["candidate_receipt_cbor_hex"],
        "candidate_receipt_root": derived["candidate_receipt_root"],
        "closed_q1_authority": derived["closed_q1_authority"],
        "final_receipt_hex": derived["final_receipt_cbor_hex"],
        "final_receipt_root": derived["final_receipt_root"],
        "host_execution_binding_root": derived[
            "host_execution_binding_root"
        ],
        "isolation_evidence_root": derived["isolation_evidence_root"],
        "ordered_predicate_rows": derived["ordered_predicate_rows"],
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "resource_evidence_root": derived["resource_evidence_root"],
        "strict_replay_equal": True,
    }
    if (
        type(evidence) is not dict
        or any(
            _canonical_json_bytes(evidence.get(name))
            != _canonical_json_bytes(value)
            for name, value in expected.items()
        )
        or type(stage10.get("qualification_count")) is not int
        or stage10["qualification_count"] != 20
        or type(stage10.get("qualification_mask")) is not int
        or stage10["qualification_mask"] != 0xFFFFF
        or stage10.get("candidate_receipt_hex")
        != derived["candidate_receipt_cbor_hex"]
        or stage10.get("final_receipt_hex")
        != derived["final_receipt_cbor_hex"]
        or stage9.get("candidate_receipt_hex")
        != stage10.get("candidate_receipt_hex")
        or stage9.get("final_receipt_hex") != stage10.get("final_receipt_hex")
    ):
        _fail(FAIL_ARTIFACT, "Stage-10 row differs from artifact/Stage-9 replay")
    return stage10


def build_actual_final_delivery_identity_v1(
    *,
    source_commit: str,
    artifact_path: Path,
    ordered_stage_root_rows: Sequence[Sequence[object]],
    actual_admission_section: Mapping[str, object],
    stage_10_live_marker_replay: Mapping[str, object],
    prepublication_live_marker_replay: Mapping[str, object],
    postpublication_live_marker_replay: Mapping[str, object],
    published_handle: AnchoredPublishedArtifactV1,
    final_delivery_bytes: bytes,
    first_anchored_replay_bytes: bytes,
    artifact_set_root: str,
    candidate_receipt_root: str,
    final_receipt_root: str,
) -> tuple[dict[str, object], str]:
    """Bind the complete post-artifact causal handoff without artifact cycles."""

    if (
        type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(artifact_path, Path)
        or not artifact_path.is_absolute()
        or type(ordered_stage_root_rows) not in (tuple, list)
        or len(ordered_stage_root_rows) != 10
        or type(actual_admission_section) is not dict
        or type(published_handle) is not AnchoredPublishedArtifactV1
        or type(final_delivery_bytes) is not bytes
        or type(first_anchored_replay_bytes) is not bytes
    ):
        _fail(FAIL_ARTIFACT, "final delivery identity input differs")
    stage_rows: list[list[object]] = []
    for expected, row in zip(
        ACTUAL_ORCHESTRATION_STAGE_REGISTRY,
        ordered_stage_root_rows,
        strict=True,
    ):
        if (
            type(row) not in (tuple, list)
            or len(row) != 3
            or type(row[0]) is not int
            or (row[0], row[1]) != expected
            or type(row[1]) is not str
            or type(row[2]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", row[2]) is None
        ):
            _fail(FAIL_ARTIFACT, "final delivery ordered stage row differs")
        stage_rows.append([row[0], row[1], row[2]])
    admission_root = actual_admission_section.get(
        "actual_admission_evidence_root"
    )
    pre_artifact_surface = actual_admission_section.get(
        "pre_artifact_live_marker_replay"
    )
    root_registry = actual_admission_section.get("root_registry")
    if (
        type(admission_root) is not str
        or re.fullmatch(r"[0-9a-f]{64}", admission_root) is None
        or type(root_registry) is not dict
    ):
        _fail(FAIL_ARTIFACT, "final delivery admission section differs")
    pre_artifact_live = validate_actual_admission_live_marker_replay_surface_v1(
        pre_artifact_surface,
        "PRE_ARTIFACT_ASSEMBLY",
    )
    stage10_live = validate_actual_admission_live_marker_replay_surface_v1(
        stage_10_live_marker_replay,
        "STAGE_10_BEFORE_EVIDENCE",
    )
    prepublication_live = validate_actual_admission_live_marker_replay_surface_v1(
        prepublication_live_marker_replay,
        "PREPUBLICATION_AFTER_STAGE10",
    )
    postpublication_live = validate_actual_admission_live_marker_replay_surface_v1(
        postpublication_live_marker_replay,
        "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY",
    )
    live_surfaces = (
        pre_artifact_live,
        stage10_live,
        prepublication_live,
        postpublication_live,
    )
    if (
        root_registry.get("pre_artifact_live_marker_replay_root")
        != pre_artifact_live["live_marker_replay_root"]
        or any(
            not _same_live_admission_authority_v1(live_surfaces[0], row)
            for row in live_surfaces[1:]
        )
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "final delivery live authority differs")
    for name, root in (
        ("artifact_set_root", artifact_set_root),
        ("candidate_receipt_root", candidate_receipt_root),
        ("final_receipt_root", final_receipt_root),
    ):
        if type(root) is not str or re.fullmatch(r"[0-9a-f]{64}", root) is None:
            _fail(FAIL_ARTIFACT, f"final delivery {name} differs")
    if (
        published_handle.payload_length != len(final_delivery_bytes)
        or published_handle.payload_sha256
        != sha256(final_delivery_bytes).hexdigest()
        or final_delivery_bytes != first_anchored_replay_bytes
    ):
        _fail(FAIL_ARTIFACT, "final delivery anchored payload differs")
    identity: dict[str, object] = {
        "schema_version": ACTUAL_FINAL_DELIVERY_IDENTITY_SCHEMA_VERSION,
        "handoff_point": (
            "FINAL_ANCHORED_REPLAY_AFTER_POSTPUBLICATION_LIVE_"
            "BEFORE_HANDLE_CLOSE_AND_RETURN"
        ),
        "source_commit": source_commit,
        "artifact_path": artifact_path.as_posix(),
        "ordered_stage_root_rows": stage_rows,
        "actual_admission_evidence_root": admission_root,
        "pre_artifact_live_marker_replay_root": pre_artifact_live[
            "live_marker_replay_root"
        ],
        "stage_10_live_marker_replay_root": stage10_live[
            "live_marker_replay_root"
        ],
        "prepublication_live_marker_replay_root": prepublication_live[
            "live_marker_replay_root"
        ],
        "postpublication_live_marker_replay_root": postpublication_live[
            "live_marker_replay_root"
        ],
        "artifact_set_root": artifact_set_root,
        "candidate_receipt_root": candidate_receipt_root,
        "final_receipt_root": final_receipt_root,
        "publisher_parent_device": published_handle.parent_device,
        "publisher_parent_inode": published_handle.parent_inode,
        "publisher_file_device": published_handle.file_device,
        "publisher_file_inode": published_handle.file_inode,
        "publisher_file_mode": published_handle.file_mode,
        "publisher_file_nlink": published_handle.file_nlink,
        "payload_length": len(final_delivery_bytes),
        "payload_sha256": sha256(final_delivery_bytes).hexdigest(),
        "first_anchored_replay_sha256": sha256(
            first_anchored_replay_bytes
        ).hexdigest(),
        "final_path_matches_publisher_owned_inode": True,
        "same_uid_mutation_after_handoff_outside_claim": True,
    }
    root = sha256(
        ACTUAL_FINAL_DELIVERY_ROOT_DOMAIN + _canonical_json_bytes(identity)
    ).hexdigest()
    return identity, root

def orchestrate_actual_with_backend_v1(
    source_commit: str,
    artifact_path: Path,
    backend: object,
    *,
    publisher: Callable[
        [Path, object], AnchoredPublishedArtifactV1
    ] = atomic_publish_canonical_artifact_v1,
    pre_handoff_cleanup: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Fail-closed kernel for the implemented conditional actual path.

    The backend owns concrete Docker/process operations.  This kernel freezes
    their causal order and prevents candidate/final receipt bytes or an
    artifact from existing before the corresponding strict-replay stages.
    The public ``--run`` action uses this kernel, but Stage 4 is admitted only
    after Stages 1--3 establish and strictly replay the single-use boundary.
    Entering this function alone creates no predicate pass, receipt, artifact,
    or Q1 authority transition.
    """

    if (
        type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(artifact_path, Path)
        or not artifact_path.is_absolute()
        or artifact_path.exists()
        or not callable(publisher)
        or (pre_handoff_cleanup is not None and not callable(pre_handoff_cleanup))
    ):
        _fail(FAIL_POLICY, "actual orchestration kernel inputs differ")
    context: dict[str, object] = {
        "source_commit": source_commit,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
    }

    def context_snapshot() -> dict[str, object]:
        return _strict_json_value_v1(
            _canonical_json_bytes(context),
            "actual orchestration context snapshot",
        )

    def execute_stage(stage_id: int, stage_name: str) -> dict[str, object]:
        method_name = f"stage_{stage_id:02d}_v1"
        method = getattr(backend, method_name, None)
        if not callable(method):
            _fail(FAIL_ACTUAL_NOT_IMPLEMENTED, f"actual backend lacks {method_name}")
        result = method(context_snapshot())
        validated = validate_actual_stage_evidence_v1(
            result,
            stage_id,
            stage_name,
            source_commit,
        )
        frozen = _strict_json_value_v1(
            _canonical_json_bytes(validated),
            f"actual Stage-{stage_id} frozen evidence",
        )
        validate_actual_stage_evidence_v1(
            frozen,
            stage_id,
            stage_name,
            source_commit,
        )
        context[f"stage_{stage_id:02d}"] = frozen
        if artifact_path.exists():
            _fail(FAIL_ARTIFACT, "backend published an artifact before stage 10")
        return validated

    for stage_id, stage_name in ACTUAL_ORCHESTRATION_STAGE_REGISTRY[:3]:
        execute_stage(stage_id, stage_name)
    issuer = getattr(backend, "issue_stage3_to4_admission_boundary_v1", None)
    boundary_replayer = getattr(
        backend,
        "strict_replay_stage3_to4_admission_boundary_v1",
        None,
    )
    if not callable(issuer) or not callable(boundary_replayer):
        _fail(
            FAIL_ACTUAL_NOT_IMPLEMENTED,
            "actual backend lacks stage3-to4 admission boundary APIs",
        )
    issued_record = issuer(context_snapshot())
    validated_record, issued_boundary = validate_actual_admission_issue_record_v1(
        issued_record
    )
    stage_rows = tuple(context[f"stage_{index:02d}"] for index in range(1, 4))
    validate_stage3_to4_admission_boundary_surface_v1(
        issued_boundary,
        source_commit,
        artifact_path,
        stage_rows,
    )
    replayed_record = boundary_replayer(
        _strict_json_value_v1(
            _canonical_json_bytes(validated_record),
            "actual admission issue record snapshot",
        ),
        context_snapshot(),
    )
    if (
        type(replayed_record) is not dict
        or _canonical_json_bytes(replayed_record)
        != _canonical_json_bytes(validated_record)
    ):
        _fail(FAIL_ACTUAL_ADMISSION, "issued admission record replay differs")
    context["stage3_to4_admission_issue_record"] = replayed_record
    for stage_id, stage_name in ACTUAL_ORCHESTRATION_STAGE_REGISTRY[3:8]:
        execute_stage(stage_id, stage_name)
    execute_stage(*ACTUAL_ORCHESTRATION_STAGE_REGISTRY[8])
    artifact_value_method = getattr(backend, "canonical_artifact_value_v1", None)
    replay_method = getattr(backend, "strict_replay_artifact_value_v1", None)
    if not callable(artifact_value_method) or not callable(replay_method):
        _fail(FAIL_ACTUAL_NOT_IMPLEMENTED, "actual backend lacks artifact replay API")
    artifact_value = artifact_value_method(context_snapshot())
    replayed_value = replay_method(
        _strict_json_value_v1(
            _canonical_json_bytes(artifact_value),
            "backend artifact replay input",
        ),
        context_snapshot(),
    )
    if replayed_value != artifact_value:
        _fail(FAIL_ARTIFACT, "actual backend artifact strict replay differs")
    (
        validated_artifact,
        _candidate_registry,
        derived_registry,
        artifact_bytes,
    ) = _validate_actual_stage8_to9_adapter_v1(context, artifact_value)
    context["strict_replayed_artifact"] = validated_artifact
    context["strict_replayed_artifact_sha256"] = sha256(artifact_bytes).hexdigest()
    execute_stage(*ACTUAL_ORCHESTRATION_STAGE_REGISTRY[9])
    final_stage = context["stage_10"]
    _validate_actual_stage10_adapter_v1(
        final_stage,
        context["stage_09"],
        derived_registry,
        artifact_bytes,
    )
    live_authority_replayer = getattr(
        backend,
        "strict_replay_actual_admission_live_authority_v1",
        None,
    )
    if not callable(live_authority_replayer):
        _fail(
            FAIL_ACTUAL_NOT_IMPLEMENTED,
            "actual backend lacks live admission authority replay",
        )
    stage10_live = validate_actual_admission_live_marker_replay_surface_v1(
        final_stage["evidence"].get("actual_admission_live_marker_replay"),
        "STAGE_10_BEFORE_EVIDENCE",
    )
    prepublication_live = validate_actual_admission_live_marker_replay_surface_v1(
        live_authority_replayer("PREPUBLICATION_AFTER_STAGE10"),
        "PREPUBLICATION_AFTER_STAGE10",
    )
    if not _same_live_admission_authority_v1(stage10_live, prepublication_live):
        _fail(FAIL_ACTUAL_ADMISSION, "prepublication admission authority changed")
    published_handle: AnchoredPublishedArtifactV1 | None = None
    try:
        returned_handle = publisher(artifact_path, validated_artifact)
    except BaseException as publisher_error:
        try:
            artifact_path.lstat()
        except FileNotFoundError:
            raise publisher_error
        except OSError as residual_error:
            _fail(
                FAIL_ARTIFACT,
                "publisher failed without ownership handle and residual status "
                f"is unreadable: publisher={type(publisher_error).__name__}:"
                f"{publisher_error}; residual={residual_error}",
                artifact_written=True,
            )
        _fail(
            FAIL_ARTIFACT,
            "publisher failed without ownership handle and left an unowned "
            f"artifact residual: {type(publisher_error).__name__}:{publisher_error}",
            artifact_written=True,
        )
    if type(returned_handle) is not AnchoredPublishedArtifactV1:
        _fail(
            FAIL_ARTIFACT,
            "publisher returned no exact ownership handle",
            artifact_written=_artifact_path_residual_conservative_v1(
                artifact_path
            ),
        )
    published_handle = returned_handle
    final_handoff_complete = False
    publisher_handle_close_status = "NOT_ATTEMPTED"
    publication_failure: BaseException | None = None
    rollback_proved_absent = False
    try:
        published_bytes = read_anchored_published_artifact_v1(
            published_handle,
            artifact_path,
            artifact_bytes,
        )
        published_value = _strict_json_value_v1(
            published_bytes,
            "published actual artifact",
        )
        (
            published_replay,
            _published_candidate,
            published_derived,
            published_payload,
        ) = _validate_actual_stage8_to9_adapter_v1(context, published_value)
        if (
            published_replay != validated_artifact
            or published_derived != derived_registry
            or published_payload != artifact_bytes
        ):
            _fail(FAIL_ARTIFACT, "published actual artifact strict replay differs")
        postpublication_live = (
            validate_actual_admission_live_marker_replay_surface_v1(
                live_authority_replayer(
                    "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY"
                ),
                "POSTPUBLICATION_AFTER_ANCHORED_ARTIFACT_REPLAY",
            )
        )
        if not _same_live_admission_authority_v1(
            prepublication_live,
            postpublication_live,
        ):
            _fail(FAIL_ACTUAL_ADMISSION, "postpublication admission authority changed")
        descriptor_errors = _detach_and_close_actual_admission_descriptors_v1(
            backend
        )
        if descriptor_errors:
            _fail(
                FAIL_POLICY,
                "actual admission descriptor release failed before owned "
                "publication handoff: " + "; ".join(descriptor_errors),
            )
        if pre_handoff_cleanup is not None:
            pre_handoff_cleanup()
        final_delivery_bytes = read_anchored_published_artifact_v1(
            published_handle,
            artifact_path,
            artifact_bytes,
        )
        if final_delivery_bytes != published_bytes:
            _fail(FAIL_ARTIFACT, "final anchored delivery bytes differ")
        ordered_stage_root_rows: list[list[object]] = []
        for stage_id, stage_name in ACTUAL_ORCHESTRATION_STAGE_REGISTRY:
            stage = context.get(f"stage_{stage_id:02d}")
            replayed_stage = validate_actual_stage_evidence_v1(
                _strict_json_value_v1(
                    _canonical_json_bytes(stage),
                    f"final delivery Stage-{stage_id} replay",
                ),
                stage_id,
                stage_name,
                source_commit,
            )
            if _canonical_json_bytes(replayed_stage) != _canonical_json_bytes(stage):
                _fail(FAIL_ARTIFACT, "final delivery stage snapshot differs")
            ordered_stage_root_rows.append(
                [stage_id, stage_name, replayed_stage["stage_evidence_root"]]
            )
        final_delivery_identity, final_delivery_root = (
            build_actual_final_delivery_identity_v1(
                source_commit=source_commit,
                artifact_path=artifact_path,
                ordered_stage_root_rows=ordered_stage_root_rows,
                actual_admission_section=validated_artifact["sections"][
                    "actual_admission"
                ],
                stage_10_live_marker_replay=final_stage["evidence"][
                    "actual_admission_live_marker_replay"
                ],
                prepublication_live_marker_replay=prepublication_live,
                postpublication_live_marker_replay=postpublication_live,
                published_handle=published_handle,
                final_delivery_bytes=final_delivery_bytes,
                first_anchored_replay_bytes=published_bytes,
                artifact_set_root=derived_registry["artifact_set_root"],
                candidate_receipt_root=derived_registry[
                    "candidate_receipt_root"
                ],
                final_receipt_root=derived_registry["final_receipt_root"],
            )
        )
        final_handoff_complete = True
    except BaseException as error:
        try:
            rollback_anchored_published_artifact_v1(
                published_handle,
                artifact_path,
                artifact_bytes,
            )
        except BaseException as rollback_error:
            publication_failure = Q05BDualSupervisorError(
                FAIL_ARTIFACT,
                "post-publication replay and same-inode rollback failed: "
                f"replay={type(error).__name__}:{error}; "
                f"rollback={type(rollback_error).__name__}:{rollback_error}",
                artifact_written=_owned_artifact_residual_conservative_v1(
                    published_handle,
                    artifact_path,
                ),
            )
        else:
            rollback_proved_absent = True
            publication_failure = error
    try:
        close_anchored_published_artifact_v1(published_handle)
    except BaseException as close_error:
        if final_handoff_complete:
            publisher_handle_close_status = "ERROR_AFTER_FINAL_HANDOFF"
        else:
            if isinstance(publication_failure, Q05BDualSupervisorError):
                artifact_written = publication_failure.artifact_written
            elif rollback_proved_absent:
                artifact_written = False
            else:
                artifact_written = _owned_artifact_residual_conservative_v1(
                    published_handle,
                    artifact_path,
                )
            publication_failure = Q05BDualSupervisorError(
                FAIL_ARTIFACT,
                "publication failure preserved across handle-close failure: "
                f"failure={type(publication_failure).__name__}:"
                f"{publication_failure}; handle_close="
                f"{type(close_error).__name__}:{close_error}",
                artifact_written=artifact_written,
            )
    else:
        publisher_handle_close_status = "CLOSED"
    if publication_failure is not None:
        raise publication_failure
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256(final_delivery_bytes).hexdigest(),
        "artifact_set_root": final_stage["evidence"]["artifact_set_root"],
        "candidate_receipt_root": final_stage["evidence"][
            "candidate_receipt_root"
        ],
        "final_receipt_root": final_stage["evidence"]["final_receipt_root"],
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "actual_admission_prepublication_live_replay_root": (
            prepublication_live["live_marker_replay_root"]
        ),
        "actual_admission_pre_artifact_live_replay_root": (
            validated_artifact["sections"]["actual_admission"][
                "pre_artifact_live_marker_replay"
            ]["live_marker_replay_root"]
        ),
        "actual_admission_stage10_live_replay_root": (
            stage10_live["live_marker_replay_root"]
        ),
        "actual_admission_postpublication_live_replay_root": (
            postpublication_live["live_marker_replay_root"]
        ),
        "actual_final_delivery_identity": final_delivery_identity,
        "actual_final_delivery_root": final_delivery_root,
        "publisher_handle_close_status": publisher_handle_close_status,
        "stage_roots": [
            context[f"stage_{stage_id:02d}"]["stage_evidence_root"]
            for stage_id, _stage_name in ACTUAL_ORCHESTRATION_STAGE_REGISTRY
        ],
    }


def run_actual_v1(
    project_root: Path,
    source_commit: str,
    artifact: Path,
    cargo_cache_source: Path,
) -> dict[str, object]:
    """Execute one supervisor-owned conditional attempt.

    This entrypoint accepts no caller-selected work root, nonce, or admission
    boundary.  Those values are created inside one private temporary directory
    and remain owned by the concrete backend and orchestration kernel.
    """

    if (
        not isinstance(project_root, Path)
        or not project_root.is_absolute()
        or type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(artifact, Path)
        or not artifact.is_absolute()
        or not isinstance(cargo_cache_source, Path)
        or not cargo_cache_source.is_absolute()
    ):
        _fail(FAIL_SOURCE, "actual entrypoint inputs differ")
    try:
        project_status = project_root.lstat()
    except OSError as error:
        _fail(FAIL_SOURCE, f"actual project root is unreadable: {error}")
    if project_root.is_symlink() or not stat.S_ISDIR(project_status.st_mode):
        _fail(FAIL_SOURCE, "actual project root is not one ordinary directory")
    expected_artifact = project_root / ACTUAL_ARTIFACT_RELATIVE_PATH
    if artifact != expected_artifact:
        _fail(FAIL_ARTIFACT, "actual artifact path is not the frozen path")
    if os.path.lexists(artifact):
        _fail(FAIL_ARTIFACT, "actual artifact target already exists")
    try:
        cargo_status = cargo_cache_source.lstat()
    except OSError as error:
        _fail(FAIL_SOURCE, f"Cargo cache source is unreadable: {error}")
    if cargo_cache_source.is_symlink() or not stat.S_ISDIR(cargo_status.st_mode):
        _fail(FAIL_SOURCE, "Cargo cache source is not one non-symlink directory")
    verify_actual_source_commit_v1(project_root, source_commit)

    try:
        temporary_parent_status = ACTUAL_TEMPORARY_PARENT.lstat()
    except OSError as error:
        _fail(FAIL_POLICY, f"trusted temporary parent is unreadable: {error}")
    if (
        ACTUAL_TEMPORARY_PARENT.is_symlink()
        or not stat.S_ISDIR(temporary_parent_status.st_mode)
        or stat.S_IMODE(temporary_parent_status.st_mode) != 0o1777
    ):
        _fail(
            FAIL_POLICY,
            "trusted temporary parent is not the frozen non-symlink 01777 directory",
        )

    temporary_manager = tempfile.TemporaryDirectory(
        prefix="hegel-q05b-actual-",
        dir=ACTUAL_TEMPORARY_PARENT,
    )
    cleanup_attempted = False

    def cleanup_temporary_root_once_v1() -> None:
        nonlocal cleanup_attempted
        if cleanup_attempted:
            _fail(FAIL_POLICY, "actual temporary root cleanup was attempted twice")
        cleanup_attempted = True
        temporary_manager.cleanup()

    backend: ConcreteQ05BActualBackendV1 | None = None
    try:
        raw_work_root = temporary_manager.name
        work_root = Path(raw_work_root)
        work_status = work_root.lstat()
        if (
            work_root.is_symlink()
            or not stat.S_ISDIR(work_status.st_mode)
            or stat.S_IMODE(work_status.st_mode) != 0o700
            or work_status.st_nlink != 2
            or work_status.st_uid != os.geteuid()
            or tuple(work_root.iterdir())
        ):
            _fail(
                FAIL_POLICY,
                "supervisor temporary work root is not one private empty 0700 directory",
            )
        backend = ConcreteQ05BActualBackendV1(
            project_root,
            source_commit,
            artifact,
            cargo_cache_source,
            work_root,
        )
        result = orchestrate_actual_with_backend_v1(
            source_commit,
            artifact,
            backend,
            pre_handoff_cleanup=cleanup_temporary_root_once_v1,
        )
    except BaseException as original:
        cleanup_errors: list[str] = []
        if backend is not None:
            try:
                cleanup_errors.extend(
                    _cleanup_failed_actual_backend_v1(backend)
                )
            except BaseException as error:
                cleanup_errors.append(
                    f"backend-cleanup:{type(error).__name__}:{error}"
                )
        if not cleanup_attempted:
            try:
                cleanup_temporary_root_once_v1()
            except BaseException as error:
                cleanup_errors.append(
                    f"temporary-root:{type(error).__name__}:{error}"
                )
        if cleanup_errors:
            if isinstance(original, Q05BDualSupervisorError):
                artifact_written = original.artifact_written
            else:
                artifact_written = _artifact_path_residual_conservative_v1(
                    artifact
                )
            _fail(
                FAIL_POLICY,
                "actual attempt failure cleanup did not close: "
                + "; ".join(cleanup_errors)
                + f"; original={type(original).__name__}:{original}",
                artifact_written=artifact_written,
            )
        raise
    return result


_ACTUAL_ADMISSION_DESCRIPTOR_FIELDS: Final = (
    "admission_work_root_descriptor",
    "admission_issued_marker_descriptor",
    "admission_spending_marker_descriptor",
    "admission_consumed_marker_descriptor",
)


def _detach_and_close_actual_admission_descriptors_v1(
    backend: object,
) -> tuple[str, ...]:
    """Detach all four held descriptors before one best-effort close each."""

    captured: list[tuple[str, object]] = []
    for field in _ACTUAL_ADMISSION_DESCRIPTOR_FIELDS:
        captured.append((field, getattr(backend, field, None)))
    for field, _descriptor in captured:
        try:
            setattr(backend, field, None)
        except BaseException:
            # Concrete backends always expose mutable fields.  Continue so any
            # already captured descriptors are still closed exactly once.
            pass
    errors: list[str] = []
    seen: set[int] = set()
    for field, descriptor in captured:
        if descriptor is None:
            continue
        if type(descriptor) is not int or descriptor < 0:
            errors.append(f"{field}:descriptor type differs")
            continue
        if descriptor in seen:
            errors.append(f"{field}:descriptor aliases another held field")
            continue
        seen.add(descriptor)
        try:
            os.close(descriptor)
        except BaseException as error:
            errors.append(f"{field}:{type(error).__name__}:{error}")
    return tuple(errors)


def _cleanup_failed_actual_backend_body_v1(backend: object) -> tuple[str, ...]:
    """Close fixed ownership registries, then release admission descriptors."""

    errors: list[str] = []
    try:
        mount_errors = _close_backend_active_mount_slots_v1(
            backend,
            (0, 1, 2),
        )
    except BaseException as error:
        errors.append(
            "active-mount-sweep:cleanup raised:"
            f"{type(error).__name__}:{error}"
        )
    else:
        errors.extend(mount_errors)
    runner = getattr(backend, "command_runner", subprocess.run)
    if not callable(runner):
        errors.append("active actors:cleanup runner differs")
        runner = None

    def cleanup_one(actor: object, label: str) -> bool:
        if actor is None:
            return True
        if type(actor) is not HeldActorProcessV1:
            errors.append(f"{label}:actor type differs")
            return False
        if runner is None:
            return False
        try:
            actor_errors = _abort_held_actor_cleanup_v1(actor, runner)
        except BaseException as error:
            errors.append(
                f"{label}:cleanup raised:{type(error).__name__}:{error}"
            )
            return False
        for row in actor_errors:
            errors.append(f"{label}:{row}")
        return not actor_errors

    active = getattr(backend, "active_actor_slots", None)
    if type(active) is not list or len(active) != 3:
        errors.append("active_actor_slots:registry differs")
        active_0 = active_1 = active_2 = None
        active_ok_0 = active_ok_1 = active_ok_2 = False
    else:
        active_0 = active[0]
        active_1 = active[1]
        active_2 = active[2]
        active_ok_0 = cleanup_one(active_0, "active_actor_slots[0]")
        if active_ok_0:
            active[0] = None
        active_ok_1 = cleanup_one(active_1, "active_actor_slots[1]")
        if active_ok_1:
            active[1] = None
        active_ok_2 = cleanup_one(active_2, "active_actor_slots[2]")
        if active_ok_2:
            active[2] = None
    actors = getattr(backend, "endpoint_actors", None)
    endpoints_ok = True
    if actors is not None:
        if type(actors) is not tuple or len(actors) != 2:
            errors.append("endpoint_actors:registry type differs")
            endpoints_ok = False
        else:
            endpoint_0 = actors[0]
            endpoint_1 = actors[1]
            endpoint_ok_0 = (
                active_ok_0
                if endpoint_0 is active_0
                else (
                    active_ok_1
                    if endpoint_0 is active_1
                    else (
                        active_ok_2
                        if endpoint_0 is active_2
                        else cleanup_one(endpoint_0, "endpoint_actors[0]")
                    )
                )
            )
            endpoint_ok_1 = (
                active_ok_0
                if endpoint_1 is active_0
                else (
                    active_ok_1
                    if endpoint_1 is active_1
                    else (
                        active_ok_2
                        if endpoint_1 is active_2
                        else (
                            endpoint_ok_0
                            if endpoint_1 is endpoint_0
                            else cleanup_one(endpoint_1, "endpoint_actors[1]")
                        )
                    )
                )
            )
            endpoints_ok = endpoint_ok_0 and endpoint_ok_1
    host_actor = getattr(backend, "host_actor", None)
    host_ok = (
        active_ok_0
        if host_actor is active_0
        else (
            active_ok_1
            if host_actor is active_1
            else (
                active_ok_2
                if host_actor is active_2
                else cleanup_one(host_actor, "host_actor")
            )
        )
    )
    try:
        if endpoints_ok:
            setattr(backend, "endpoint_actors", None)
        if host_ok:
            setattr(backend, "host_actor", None)
    except BaseException as error:
        errors.append(f"actor-registry-detach:{type(error).__name__}:{error}")
    try:
        descriptor_errors = _detach_and_close_actual_admission_descriptors_v1(
            backend
        )
    except BaseException as error:
        errors.append(
            "actual-admission-descriptor-sweep:cleanup raised:"
            f"{type(error).__name__}:{error}"
        )
    else:
        errors.extend(descriptor_errors)
    return tuple(errors)


def _cleanup_failed_actual_backend_v1(backend: object) -> tuple[str, ...]:
    with _docker_ownership_signal_guard_v1():
        return _cleanup_failed_actual_backend_body_v1(backend)


class _CanonicalArgumentParser(argparse.ArgumentParser):
    """Turn argument rejection into the supervisor's canonical error wire."""

    def error(self, message: str) -> NoReturn:
        _fail(FAIL_POLICY, f"command-line arguments differ: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--internal-host-replay", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--cargo-cache-source", type=Path)
    parser.add_argument("--python-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rust-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--python-stdout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rust-stdout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stdout-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--staging-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--host-source-identity-root-hex", help=argparse.SUPPRESS)
    parser.add_argument("--host-runtime-identity-root-hex", help=argparse.SUPPRESS)
    return parser


def _error_object(error: Q05BDualSupervisorError) -> dict[str, object]:
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "status": error.code,
        "detail": error.detail,
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "qualification_candidate_receipt": None,
        "qualification_final_receipt": None,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "receipt_created": False,
        "artifact_written": error.artifact_written,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        options = _parser().parse_args(arguments)
        project_root = (
            options.project_root
            if options.run
            else options.project_root.resolve(strict=True)
        )
        if options.dry_run:
            if any(
                value is not None
                for value in (
                    options.source_commit,
                    options.artifact,
                    options.cargo_cache_source,
                    options.python_output,
                    options.rust_output,
                    options.python_stdout,
                    options.rust_stdout,
                    options.stdout_manifest,
                    options.staging_output,
                    options.host_source_identity_root_hex,
                    options.host_runtime_identity_root_hex,
                )
            ):
                _fail(FAIL_POLICY, "dry-run admits no source/output/artifact arguments")
            value = dry_run_plan_v1(project_root)
        elif options.run:
            internal_values = (
                options.python_output,
                options.rust_output,
                options.python_stdout,
                options.rust_stdout,
                options.stdout_manifest,
                options.staging_output,
                options.host_source_identity_root_hex,
                options.host_runtime_identity_root_hex,
            )
            if (
                options.source_commit is None
                or options.artifact is None
                or options.cargo_cache_source is None
                or any(value is not None for value in internal_values)
            ):
                _fail(
                    FAIL_SOURCE,
                    "--run requires only --source-commit, --artifact, and "
                    "--cargo-cache-source",
                )
            value = run_actual_v1(
                project_root,
                options.source_commit,
                options.artifact,
                options.cargo_cache_source,
            )
        else:
            required = (
                options.python_output,
                options.rust_output,
                options.python_stdout,
                options.rust_stdout,
                options.stdout_manifest,
                options.staging_output,
                options.host_source_identity_root_hex,
                options.host_runtime_identity_root_hex,
            )
            if (
                options.source_commit is not None
                or options.artifact is not None
                or options.cargo_cache_source is not None
                or any(item is None for item in required)
            ):
                _fail(FAIL_POLICY, "internal trusted-host arguments differ")
            value = internal_host_replay_v1(
                options.python_output.resolve(strict=True),
                options.rust_output.resolve(strict=True),
                options.python_stdout.resolve(strict=True),
                options.rust_stdout.resolve(strict=True),
                options.stdout_manifest.resolve(strict=True),
                options.staging_output.resolve(strict=True),
                options.host_source_identity_root_hex,
                options.host_runtime_identity_root_hex,
                snapshot_root=project_root,
            )
            sys.stdout.buffer.write(value)
            return 0
    except (OSError, Q05BDualSupervisorError) as error:
        if isinstance(error, Q05BDualSupervisorError):
            failure = error
        else:
            failure = Q05BDualSupervisorError(FAIL_SOURCE, str(error))
        sys.stdout.buffer.write(_canonical_json_bytes(_error_object(failure)))
        return 1
    sys.stdout.buffer.write(_canonical_json_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
