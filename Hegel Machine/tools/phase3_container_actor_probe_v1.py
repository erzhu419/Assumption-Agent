#!/usr/bin/env python3
"""Live in-container probes for the Phase-3 technical-actor profile.

The supervisor mounts this file read-only and invokes it under ``env -i``.
It deliberately uses only the Python standard library so the pinned image is
the complete runtime dependency.  The output is one compact JSON line.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time


SCHEMA = "hegel-container-actor-live-probe/1"
IMPLEMENTATION = "python-ctypes-v1"
STATUS_FIELDS = (
    "CapInh",
    "CapPrm",
    "CapEff",
    "CapBnd",
    "CapAmb",
    "NoNewPrivs",
    "Seccomp",
)
NAMESPACE_KINDS = ("pid", "mnt", "net", "ipc", "uts")
FORBIDDEN_PATHS = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/workspace",
    "/repo",
    "/mnt/c",
)
CROSS_PURPOSE_PATHS = (
    "/purpose-1",
    "/purpose-2",
    "/purpose-3",
    "/purpose-4",
    "/actor-1",
    "/actor-2",
    "/actor-3",
    "/actor-4",
)

# Linux x86_64 syscall numbers.  The pinned images and profile are amd64; the
# supervisor rejects any other image architecture before launching this file.
SYS_BPF = 321
SYS_PERF_EVENT_OPEN = 298


def _errno_row(call) -> dict[str, int]:
    ctypes.set_errno(0)
    result = int(call())
    observed_errno = ctypes.get_errno() if result == -1 else 0
    return {"return_value": result, "errno": observed_errno}


def _syscall_rows() -> list[dict[str, object]]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.socket.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    libc.socket.restype = ctypes.c_int
    libc.mount.argtypes = (
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    )
    libc.mount.restype = ctypes.c_int
    libc.ptrace.argtypes = (
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    libc.ptrace.restype = ctypes.c_long
    libc.syscall.restype = ctypes.c_long

    def ptrace_traceme_in_child() -> int:
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            ctypes.set_errno(0)
            value = int(libc.ptrace(0, None, None, None))
            child_errno = ctypes.get_errno() if value == -1 else 0
            os.write(write_fd, struct.pack("=qi", value, child_errno))
            os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        payload = b""
        while len(payload) < 12:
            chunk = os.read(read_fd, 12 - len(payload))
            if not chunk:
                break
            payload += chunk
        os.close(read_fd)
        os.waitpid(child_pid, 0)
        value, child_errno = struct.unpack("=qi", payload)
        ctypes.set_errno(child_errno)
        return int(value)

    probes = (
        (
            "socket(AF_INET, SOCK_STREAM)",
            lambda: libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0),
        ),
        (
            "socket(AF_INET6, SOCK_STREAM)",
            lambda: libc.socket(socket.AF_INET6, socket.SOCK_STREAM, 0),
        ),
        (
            "mount",
            lambda: libc.mount(
                None, b"/tmp/hegel-mount-probe", None, 0, None
            ),
        ),
        (
            "ptrace(PTRACE_TRACEME)",
            ptrace_traceme_in_child,
        ),
        (
            "bpf(BPF_MAP_CREATE)",
            lambda: libc.syscall(SYS_BPF, 0, None, 0),
        ),
        (
            "perf_event_open",
            lambda: libc.syscall(
                SYS_PERF_EVENT_OPEN,
                ctypes.create_string_buffer(128),
                0,
                -1,
                -1,
                0,
            ),
        ),
    )
    rows: list[dict[str, object]] = []
    for probe_id, call in probes:
        row: dict[str, object] = {"probe_id": probe_id}
        row.update(_errno_row(call))
        # A negative-control socket may succeed.  Close it immediately.
        if probe_id.startswith("socket(") and row["return_value"] >= 0:
            os.close(int(row["return_value"]))
        rows.append(row)
    return rows


def _proc_status() -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in STATUS_FIELDS:
            raw = value.strip()
            parsed[key] = int(raw) if key in {"NoNewPrivs", "Seccomp"} else raw
    return parsed


def _namespace_rows() -> dict[str, str]:
    return {
        kind: os.readlink(f"/proc/self/ns/{kind}")
        for kind in NAMESPACE_KINDS
    }


def _open_fds() -> list[int]:
    # ``listdir`` may momentarily expose its own directory FD.  It is closed
    # before this loop; only still-resolvable inherited descriptors are kept.
    names = os.listdir("/proc/self/fd")
    result: list[int] = []
    for name in names:
        try:
            os.readlink(f"/proc/self/fd/{name}")
        except FileNotFoundError:
            continue
        if name.isdigit():
            result.append(int(name))
    return sorted(result)


def _write_denial(path: str) -> dict[str, object]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        return {"denied": True, "errno": int(exc.errno or 0)}
    else:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        return {"denied": False, "errno": 0}


def main() -> int:
    purpose_text = os.environ.get("HEGEL_PURPOSE_ID", "")
    profile_id = os.environ.get("HEGEL_ACTOR_PROFILE_ID", "")
    input_probe_path = os.environ.get(
        "HEGEL_PROBE_INPUT_WRITE_PATH", "/actor_input/profile.json"
    )
    linger_seconds = os.environ.get("HEGEL_PROBE_LINGER_SECONDS", "0")
    try:
        purpose_id = int(purpose_text)
    except ValueError:
        purpose_id = -1

    forbidden_present = [path for path in FORBIDDEN_PATHS if Path(path).exists()]
    host_repository_path = os.environ.get("HEGEL_HOST_REPOSITORY_PATH", "")
    claimed_digest = os.environ.get("HEGEL_HOST_REPOSITORY_PATH_SHA256", "")
    if host_repository_path and hashlib.sha256(
        host_repository_path.encode("utf-8")
    ).hexdigest() != claimed_digest:
        return 70
    if host_repository_path and Path(host_repository_path).exists():
        # Never disclose the host path itself.  The supervisor binds its digest
        # in the report-safe environment and this label only records failure.
        forbidden_present.append("HEGEL_HOST_REPOSITORY_PATH")
    # The raw path has now served its sole filesystem-probe purpose.  Remove it
    # from the real process environment before syscall probes, serialization,
    # lingering, or any possible descendant process.
    os.environ.pop("HEGEL_HOST_REPOSITORY_PATH", None)
    cross_present = [path for path in CROSS_PURPOSE_PATHS if Path(path).exists()]
    report_environment = dict(sorted(os.environ.items()))
    report = {
        "schema": SCHEMA,
        "implementation": IMPLEMENTATION,
        "profile_id": profile_id,
        "purpose_id": purpose_id,
        "identity": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "pid": os.getpid(),
        },
        "proc_status": _proc_status(),
        "namespaces": _namespace_rows(),
        "network_interfaces": sorted(os.listdir("/sys/class/net")),
        "syscall_probes": _syscall_rows(),
        "filesystem_probes": {
            "root_write": _write_denial("/hegel-container-root-write-probe"),
            "input_write": _write_denial(input_probe_path),
            "forbidden_paths_present": forbidden_present,
            "cross_purpose_paths_present": cross_present,
        },
        "environment": report_environment,
        "open_fds": _open_fds(),
    }
    encoded = json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    print(encoded, flush=True)
    # Keep all four workers alive concurrently so the supervisor can prove
    # that namespace identities are distinct rather than sequentially reused.
    time.sleep(int(linger_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
