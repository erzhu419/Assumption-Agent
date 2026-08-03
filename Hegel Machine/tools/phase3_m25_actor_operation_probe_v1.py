#!/usr/bin/env python3
"""In-process, operation-bound isolation probe for Python M2.5 actors.

Each purpose worker imports and runs this function before *every* operation.
The receipt is therefore produced by the same process that subsequently
performs key generation, replay, signing, seed handling, or marker promotion.
The supervisor independently replays every field before trusting outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping

import phase3_container_actor_probe_v1 as base_probe


SCHEMA = "hegel-phase3-m25-operation-bound-live-probe/1"
OUTPUT = Path("/output")
BASE_ENV_KEYS = {
    "HEGEL_ACTOR_IMAGE_REF",
    "HEGEL_ACTOR_PROFILE_ID",
    "HEGEL_BASIS_COMMIT",
    "HEGEL_DAEMON_RECEIPT_SHA256",
    "HEGEL_HOST_REPOSITORY_PATH_SHA256",
    "HEGEL_PROFILE_SHA256",
    "HEGEL_PURPOSE_ID",
    "HEGEL_RUN_ID",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
}
PRIVATE_ENV_KEYS = {"HEGEL_HOST_REPOSITORY_PATH"}
PROBE_ONLY_ENV_KEYS = {
    "HEGEL_OPERATION_ID",
    "HEGEL_OPERATION_NONCE",
    "HEGEL_OPERATION_REQUEST_SHA256",
    "HEGEL_OPERATION_SEQUENCE",
    "HEGEL_PROBE_INPUT_WRITE_PATH",
}
CAPABILITY_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
OPERATION_ALLOWLIST = {
    1: {
        "qualify-only",
        "keygen",
        "keygen-resume",
        "purpose1-authorized-sign",
        "bridge-replay-sign-python",
        "seed-split-real",
        "seed-split-resume",
        "seed-split-synthetic",
        "complete-marker",
    },
    2: {"qualify-only", "keygen", "keygen-resume", "bridge-replay-sign-python"},
    4: {"qualify-only", "keygen", "keygen-resume", "purpose4-parent-sign"},
}
PURPOSE1_CUSTODY_WRITE_OPERATIONS = {
    "seed-split-real",
    "seed-split-resume",
    "seed-split-synthetic",
    "complete-marker",
}


class OperationProbeFailure(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _write_probe(path: Path) -> dict[str, object]:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        return {"succeeded": False, "errno": int(exc.errno or 0)}
    try:
        os.write(descriptor, b"probe\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.unlink()
    return {"succeeded": True, "errno": 0}


def _pid_fds(pid: int) -> list[int]:
    root = Path(f"/proc/{pid}/fd")
    result: list[int] = []
    for entry in root.iterdir():
        try:
            os.readlink(entry)
        except FileNotFoundError:
            continue
        if entry.name.isdigit():
            result.append(int(entry.name))
    return sorted(result)


def _pid_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        key, separator, value = item.partition(b"=")
        if not separator:
            raise OperationProbeFailure
        decoded_key = key.decode("ascii", "strict")
        if decoded_key in result:
            raise OperationProbeFailure
        result[decoded_key] = value.decode("utf-8", "strict")
    return dict(sorted(result.items()))


def _cgroup_limits() -> dict[str, str]:
    return {
        "memory_max": Path("/sys/fs/cgroup/memory.max").read_text(encoding="ascii").strip(),
        "memory_swap_max": Path("/sys/fs/cgroup/memory.swap.max").read_text(encoding="ascii").strip(),
        "pids_max": Path("/sys/fs/cgroup/pids.max").read_text(encoding="ascii").strip(),
    }


def _mount_destinations() -> list[str]:
    destinations: list[str] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines():
        left, separator, _right = line.partition(" - ")
        fields = left.split()
        if not separator or len(fields) < 6:
            raise OperationProbeFailure
        destination = re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            fields[4],
        )
        if destination in {"/custody", "/input", "/output", "/state", "/tmp"}:
            destinations.append(destination)
    return sorted(destinations)


def _redact_private_environment(
    environment: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    raw_path = environment.get("HEGEL_HOST_REPOSITORY_PATH", "")
    claimed_digest = environment.get("HEGEL_HOST_REPOSITORY_PATH_SHA256", "")
    if (
        not raw_path
        or not Path(raw_path).is_absolute()
        or re.fullmatch(r"[0-9a-f]{64}", claimed_digest) is None
        or hashlib.sha256(raw_path.encode("utf-8")).hexdigest() != claimed_digest
    ):
        raise OperationProbeFailure
    report_environment = dict(environment)
    report_environment.pop("HEGEL_HOST_REPOSITORY_PATH", None)
    return raw_path, dict(sorted(report_environment.items()))


def _validate_environment(
    operation: str,
) -> tuple[int, dict[str, str], dict[str, str], str]:
    launch_environment = dict(sorted(os.environ.items()))
    if set(launch_environment) != BASE_ENV_KEYS | PROBE_ONLY_ENV_KEYS | PRIVATE_ENV_KEYS:
        raise OperationProbeFailure
    host_repository_path, environment = _redact_private_environment(
        launch_environment
    )
    # This is a process-bound isolation claim, not merely a receipt redaction.
    # Remove the probe-only raw path from the actual worker environment before
    # this function performs any further checks and before the caller can
    # generate keys, sign, replay, or handle the split seed.
    if os.environ.pop("HEGEL_HOST_REPOSITORY_PATH", None) != host_repository_path:
        raise OperationProbeFailure
    try:
        purpose = int(environment["HEGEL_PURPOSE_ID"])
    except ValueError as exc:
        raise OperationProbeFailure from exc
    if (
        purpose not in OPERATION_ALLOWLIST
        or operation not in OPERATION_ALLOWLIST[purpose]
        or environment["HEGEL_OPERATION_ID"] != operation
        or environment["HEGEL_ACTOR_PROFILE_ID"]
        != "hegel-owner-accepted-container-technical-actors-v1"
        or environment["HEGEL_PROBE_INPUT_WRITE_PATH"]
        != "/input/.hegel-write-probe"
        or environment["LANG"] != "C"
        or environment["LC_ALL"] != "C.UTF-8"
        or environment["PATH"] != "/usr/local/bin:/usr/bin:/bin"
        or environment["PYTHONDONTWRITEBYTECODE"] != "1"
        or environment["PYTHONHASHSEED"] != "0"
        or re.fullmatch(r"[0-9a-f]{40}", environment["HEGEL_BASIS_COMMIT"])
        is None
        or re.fullmatch(r"[0-9a-f]{64}", environment["HEGEL_PROFILE_SHA256"])
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            environment["HEGEL_HOST_REPOSITORY_PATH_SHA256"],
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", environment["HEGEL_DAEMON_RECEIPT_SHA256"]
        )
        is None
        or re.fullmatch(r"[0-9a-f]{32}", environment["HEGEL_RUN_ID"])
        is None
        or re.fullmatch(r"[0-9a-f]{32}", environment["HEGEL_OPERATION_NONCE"])
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", environment["HEGEL_OPERATION_REQUEST_SHA256"]
        )
        is None
        or re.fullmatch(r"[1-9][0-9]*", environment["HEGEL_OPERATION_SEQUENCE"])
        is None
        or re.fullmatch(
            r"[^@\s]+@sha256:[0-9a-f]{64}", environment["HEGEL_ACTOR_IMAGE_REF"]
        )
        is None
    ):
        raise OperationProbeFailure
    base_environment = {
        key: value for key, value in environment.items() if key in BASE_ENV_KEYS
    }
    return purpose, environment, base_environment, host_repository_path


def qualify_operation_v1(operation: str) -> Mapping[str, object]:
    """Write and return the exact receipt before the caller continues."""

    purpose, environment, base_environment, host_repository_path = (
        _validate_environment(operation)
    )
    receipt_path = OUTPUT / f"operation-probe-{operation}.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise OperationProbeFailure
    status = base_probe._proc_status()
    syscall_rows = base_probe._syscall_rows()
    input_path = Path(environment["HEGEL_PROBE_INPUT_WRITE_PATH"])
    filesystem: dict[str, object] = {
        "root_write": base_probe._write_denial("/hegel-formal-root-write-probe"),
        "input_write": base_probe._write_denial(str(input_path)),
        "output_write": _write_probe(OUTPUT / ".hegel-operation-write-probe"),
        "state_write": _write_probe(Path("/state/.hegel-operation-write-probe")),
        "custody_present": Path("/custody").exists(),
        "forbidden_paths_present": [
            path for path in base_probe.FORBIDDEN_PATHS if Path(path).exists()
        ]
        + (
            ["HEGEL_HOST_REPOSITORY_PATH"]
            if Path(host_repository_path).exists()
            else []
        ),
        "cross_purpose_paths_present": [
            path for path in base_probe.CROSS_PURPOSE_PATHS if Path(path).exists()
        ],
        "mount_destinations": _mount_destinations(),
    }
    custody_write_required = (
        purpose == 1 and operation in PURPOSE1_CUSTODY_WRITE_OPERATIONS
    )
    if custody_write_required:
        filesystem["custody_write_or_null"] = _write_probe(
            Path("/custody/.hegel-operation-write-probe")
        )
    else:
        filesystem["custody_write_or_null"] = None
    cgroup = _cgroup_limits()
    blocked = all(
        isinstance(row, Mapping)
        and row.get("return_value") == -1
        and row.get("errno") == 1
        for row in syscall_rows
    )
    expected_mounts = ["/input", "/output", "/state", "/tmp"]
    if purpose == 1:
        expected_mounts.insert(0, "/custody")
    if custody_write_required:
        custody_scope_exact = (
            filesystem["custody_present"] is True
            and filesystem["custody_write_or_null"]
            == {"succeeded": True, "errno": 0}
        )
    elif purpose == 1:
        custody_scope_exact = (
            filesystem["custody_present"] is True
            and filesystem["custody_write_or_null"] is None
        )
    else:
        custody_scope_exact = (
            filesystem["custody_present"] is False
            and filesystem["custody_write_or_null"] is None
        )
    checks = {
        "same_worker_process": os.getpid() > 1,
        "identity_nonroot": os.getuid() == 65534 and os.getgid() == 65534,
        "capabilities_zero": all(
            status.get(name) == "0000000000000000" for name in CAPABILITY_FIELDS
        ),
        "no_new_privileges": status.get("NoNewPrivs") == 1,
        "seccomp_mode": status.get("Seccomp") == 2,
        "network_loopback_only": sorted(os.listdir("/sys/class/net")) == ["lo"],
        "blocked_syscalls_eperm": blocked and len(syscall_rows) == 6,
        "root_input_read_only": all(
            isinstance(filesystem[name], Mapping)
            and filesystem[name].get("denied") is True
            and filesystem[name].get("errno") in {1, 13, 30}
            for name in ("root_write", "input_write")
        ),
        "output_state_writable": all(
            filesystem[name] == {"succeeded": True, "errno": 0}
            for name in ("output_write", "state_write")
        ),
        "custody_scope_exact": custody_scope_exact,
        "forbidden_paths_absent": not filesystem["forbidden_paths_present"],
        "cross_purpose_paths_absent": not filesystem["cross_purpose_paths_present"],
        "mount_destinations_exact": filesystem["mount_destinations"]
        == sorted(expected_mounts),
        "operation_environment_exact": set(environment)
        == BASE_ENV_KEYS | PROBE_ONLY_ENV_KEYS,
        "pid1_environment_exact": _pid_environment(1) == base_environment,
        "worker_fds_exact": base_probe._open_fds() == [0, 1, 2],
        "pid1_fds_exact": _pid_fds(1) == [0, 1, 2],
        "memory_limit_exact": cgroup["memory_max"] == str(512 * 1024 * 1024),
        "memory_swap_zero": cgroup["memory_swap_max"] == "0",
        "pids_limit_exact": cgroup["pids_max"] == "64",
    }
    if not all(checks.values()):
        raise OperationProbeFailure
    body: dict[str, object] = {
        "schema": SCHEMA,
        "implementation": "python-ctypes-in-process-v1",
        "operation_id": operation,
        "operation_sequence": int(environment["HEGEL_OPERATION_SEQUENCE"]),
        "operation_nonce_hex": environment["HEGEL_OPERATION_NONCE"],
        "operation_request_sha256": environment[
            "HEGEL_OPERATION_REQUEST_SHA256"
        ],
        "purpose_id": purpose,
        "identity": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
        },
        "proc_status": status,
        "namespaces": base_probe._namespace_rows(),
        "network_interfaces": sorted(os.listdir("/sys/class/net")),
        "syscall_probes": syscall_rows,
        "filesystem_probes": filesystem,
        "operation_environment": environment,
        "pid1_environment": base_environment,
        "worker_open_fds": [0, 1, 2],
        "pid1_open_fds": [0, 1, 2],
        "cgroup_limits": cgroup,
        "required_checks": checks,
        "all_required_checks_passed": True,
    }
    body["receipt_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    payload = _canonical_json(body)
    descriptor = os.open(
        receipt_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        os.fchmod(descriptor, 0o644)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OperationProbeFailure
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(OUTPUT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return body


__all__ = ["OperationProbeFailure", "qualify_operation_v1"]
