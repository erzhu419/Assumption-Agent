"""Purpose-separated, non-authoritative Phase-3 M3 shadow ceremony.

This module exists so engineering work can continue without misrepresenting
locally orchestrated processes as the independent external actors required by
the formal M2.5 freeze.  It provides real process, directory, pipe, key, and
signature separation on one host, but every output is permanently labelled
``INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE``.

The hard authority boundary is part of the validated report schema:

* the formal gate delta is zero;
* the formal state remains ``NOT_RUN``;
* no formal root or external-actor evidence is issued;
* an internal shadow-ready result cannot invoke the formal M3 transition.

The 32-byte split seed is created only inside the custodian child process.  It
is sent to two calculator children over anonymous pipes, is never returned to
the orchestrator, and is never serialized to a marker, ledger, report, or
signature envelope.  Ed25519 private keys likewise remain in the four role
processes and disappear when those processes exit.

This is a POSIX research control, not an operating-system security boundary.
All children still share one kernel/user identity and one root orchestrator;
the explicit non-authoritative label therefore cannot be relaxed by callers.
"""

from __future__ import annotations

import hashlib
import ctypes
import ctypes.util
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn, Sequence

from .strict_cbor_v1 import (
    StrictCborError,
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
)

try:  # Optional by project policy; runtime entry fails closed when absent.
    import cryptography as _cryptography
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by monkeypatch below.
    _cryptography = None
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-runtime/1"
SNAPSHOT_SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-input-snapshot/1"
CALCULATOR_SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-calculators/1"
ENVELOPE_SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-envelope/1"
STATE_SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-state/1"
ADMISSION_SCHEMA_VERSION: Final = "hegel-phase3-m3-shadow-admission/1"

AUTHORITY_CLASS: Final = "INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE"
EVIDENCE_CLASS: Final = "DIAGNOSTIC_INTEGRITY_ONLY_NOT_SHADOW_WIRE_OBJECT"
SEED_COMMITMENT_PREFIX: Final = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1"
FORMAL_STATE: Final = "NOT_RUN"
INTERNAL_STATE_BEFORE: Final = "INTERNAL_NOT_RUN"
INTERNAL_STATE_AFTER: Final = "INTERNAL_SHADOW_READY"

_MODULE_PATH: Final = Path(__file__).resolve()
if _is_isolated_worker := str(_MODULE_PATH).startswith("/worker-src/"):
    PROJECT_ROOT: Final = Path("/worker-src")
    REPOSITORY_ROOT: Final = Path("/live-repository-not-mounted")
else:
    PROJECT_ROOT = _MODULE_PATH.parents[2]
    REPOSITORY_ROOT = PROJECT_ROOT.parent
VIRTUAL_STATE_ROOT: Final = Path("/private-state")

PURPOSE_ROLES: Final = MappingProxyType(
    {
        1: "CUSTODIAN",
        2: "PYTHON_ATTESTER",
        3: "RUST_ATTESTER",
        4: "AUDITOR",
    }
)
PURPOSE_IDS: Final = tuple(PURPOSE_ROLES)

SANITIZED_ENVIRONMENT: Final = MappingProxyType(
    {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
)

ISOLATION_LEVEL: Final = "BWRAP_USER_PID_NET_IPC_UTS_SECCOMP_V1"
NAMESPACE_KINDS: Final = ("mnt", "user", "pid", "net", "ipc", "uts")
SECCOMP_DENIED_SYSCALLS: Final = (
    "socket",
    "socketpair",
    "connect",
    "accept",
    "accept4",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "bpf",
    "keyctl",
    "add_key",
    "request_key",
    "mount",
    "umount2",
    "pivot_root",
    "setns",
    "unshare",
    "kexec_load",
    "init_module",
    "finit_module",
    "delete_module",
    "open_by_handle_at",
    "perf_event_open",
)
ATTACK_SYSCALL_PROBES: Final = (
    ("SOCKET_AF_INET_STREAM", "socket", (socket.AF_INET, socket.SOCK_STREAM, 0)),
    ("SOCKET_AF_INET6_STREAM", "socket", (socket.AF_INET6, socket.SOCK_STREAM, 0)),
    ("MOUNT", "mount", (0, 0, 0, 0, 0)),
    ("PTRACE_TRACEME", "ptrace", (0, 0, 0, 0)),
    ("BPF_MAP_CREATE", "bpf", (0, 0, 0)),
    ("PERF_EVENT_OPEN", "perf_event_open", (0, 0, -1, -1, 0)),
)
CAPABILITY_STATUS_FIELDS: Final = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")

SNAPSHOT_DIAGNOSTIC_ID: Final = b"hegel-shadow-runtime-diagnostic-snapshot/1"
CALCULATOR_DIAGNOSTIC_ID: Final = b"hegel-shadow-runtime-diagnostic-calculators/1"
ENVELOPE_DIAGNOSTIC_ID: Final = b"hegel-shadow-runtime-diagnostic-envelope-payload/1"
PRIVATE_MARKER_MAGIC: Final = b"hegel-shadow-runtime-private-marker/1"
PRIVATE_LEDGER_MAGIC: Final = b"hegel-shadow-runtime-private-ledger/1"
PUBLIC_OUTPUT_DIAGNOSTIC_ID: Final = b"hegel-shadow-runtime-diagnostic-public-output/1"
CALCULATOR_ENDPOINT_RESPONSE_SCHEMA_ID: Final = (
    b"hegel-phase3-split-calculator-fd3-response/1"
)
CALCULATOR_ENDPOINT_PATHS: Final = MappingProxyType(
    {
        "python": Path("/calculator-endpoints/python"),
        "rust": Path("/calculator-endpoints/rust"),
    }
)
CALCULATOR_ENDPOINT_IDS: Final = MappingProxyType(
    {"python": "PYTHON_FD3_ENDPOINT_V1", "rust": "RUST_FD3_ENDPOINT_V1"}
)
CALCULATOR_QUALIFICATION_SEED: Final = bytes(range(32))
CALCULATOR_QUALIFICATION_COMMITMENT: Final = bytes.fromhex(
    "3126668b3227a5e6ab711bcaa66f9d573a7e8bf8b1d1c6cabbb07a96ccf566ba"
)

SNAPSHOT_HASH_DOMAIN: Final = "HEGEL/DIAGNOSTIC/SHADOW_RUNTIME/INPUT_SNAPSHOT/V1"
CALCULATOR_HASH_DOMAIN: Final = "HEGEL/DIAGNOSTIC/SHADOW_RUNTIME/CALCULATORS/V1"
SIGNATURE_DOMAIN: Final = b"HEGEL/DIAGNOSTIC/SHADOW_RUNTIME/ED25519_ENVELOPE/V1"
KEY_ID_DOMAIN: Final = b"HEGEL/DIAGNOSTIC/SHADOW_RUNTIME/ED25519_KEY_ID/V1"

MARKER_FILE_NAME: Final = "shadow_seed_genesis.marker.cbor"
LEDGER_FILE_NAME: Final = "shadow_seed_genesis.ledger.bin"
MAX_SOURCE_BYTES: Final = 64 * 1024 * 1024
MAX_IPC_BYTES: Final = 2 * 1024 * 1024
MAX_STATE_BYTES: Final = 64 * 1024

_LABEL_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)

FAIL_SHADOW_CRYPTO_BACKEND_UNAVAILABLE: Final = (
    "FAIL_SHADOW_CRYPTO_BACKEND_UNAVAILABLE"
)
FAIL_SHADOW_PLATFORM_UNSUPPORTED: Final = "FAIL_SHADOW_PLATFORM_UNSUPPORTED"
FAIL_SHADOW_BWRAP_UNAVAILABLE: Final = "FAIL_SHADOW_BWRAP_UNAVAILABLE"
FAIL_SHADOW_BWRAP_ISOLATION: Final = "FAIL_SHADOW_BWRAP_ISOLATION"
FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE: Final = (
    "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"
)
FAIL_SHADOW_SECCOMP_UNAVAILABLE: Final = "FAIL_SHADOW_SECCOMP_UNAVAILABLE"
FAIL_SHADOW_SECCOMP_PROBE: Final = "FAIL_SHADOW_SECCOMP_PROBE"
REJECT_SHADOW_FIELD_SET: Final = "REJECT_SHADOW_FIELD_SET"
REJECT_SHADOW_FIELD_TYPE: Final = "REJECT_SHADOW_FIELD_TYPE"
FAIL_SHADOW_STATE_PATH_INVALID: Final = "FAIL_SHADOW_STATE_PATH_INVALID"
FAIL_SHADOW_STATE_INSIDE_REPOSITORY: Final = (
    "FAIL_SHADOW_STATE_INSIDE_REPOSITORY"
)
FAIL_SHADOW_STATE_PERMISSIONS: Final = "FAIL_SHADOW_STATE_PERMISSIONS"
FAIL_SHADOW_PRIVATE_FILE_PERMISSIONS: Final = (
    "FAIL_SHADOW_PRIVATE_FILE_PERMISSIONS"
)
FAIL_SHADOW_STATE_ALREADY_COMPLETE: Final = "FAIL_SHADOW_STATE_ALREADY_COMPLETE"
FAIL_SHADOW_STATE_PENDING_RECOVERY_REQUIRED: Final = (
    "FAIL_SHADOW_STATE_PENDING_RECOVERY_REQUIRED"
)
FAIL_SHADOW_MARKER_TAMPERED: Final = "FAIL_SHADOW_MARKER_TAMPERED"
FAIL_SHADOW_LEDGER_TAMPERED: Final = "FAIL_SHADOW_LEDGER_TAMPERED"
FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID: Final = (
    "FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID"
)
FAIL_SHADOW_SNAPSHOT_MUTATED: Final = "FAIL_SHADOW_SNAPSHOT_MUTATED"
FAIL_SHADOW_CALCULATOR_PROCESS: Final = "FAIL_SHADOW_CALCULATOR_PROCESS"
FAIL_SHADOW_CALCULATOR_DISAGREEMENT: Final = (
    "FAIL_SHADOW_CALCULATOR_DISAGREEMENT"
)
FAIL_SHADOW_ROLE_PROCESS: Final = "FAIL_SHADOW_ROLE_PROCESS"
FAIL_SHADOW_IPC_PROTOCOL: Final = "FAIL_SHADOW_IPC_PROTOCOL"
FAIL_SHADOW_SIGNATURE_INVALID: Final = "FAIL_SHADOW_SIGNATURE_INVALID"
FAIL_SHADOW_PURPOSE_OR_KEY_COLLISION: Final = (
    "FAIL_SHADOW_PURPOSE_OR_KEY_COLLISION"
)
FAIL_SHADOW_AUTHORITY_ESCALATION: Final = "FAIL_SHADOW_AUTHORITY_ESCALATION"
FAIL_SHADOW_REPORT_INVALID: Final = "FAIL_SHADOW_REPORT_INVALID"


class ShadowRuntimeError(RuntimeError):
    """Stable fail-closed error emitted by the internal shadow runtime."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ShadowRuntimeError(code, detail)


def _require_exact_fields(
    value: object,
    fields: Sequence[str],
    *,
    context: str,
    code: str = REJECT_SHADOW_FIELD_SET,
) -> dict[str, object]:
    if type(value) is not dict:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} must be an object")
    assert isinstance(value, dict)
    if set(value) != set(fields):
        _fail(code, f"{context} has a non-exact field set")
    return value


def _require_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} must be an exact boolean")
    return value


def _require_int(value: object, *, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} must be an exact integer")
    return value


def _require_text(value: object, *, context: str) -> str:
    if type(value) is not str:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} must be text")
    return value


def _json_type_strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_strict_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _decode_hex(value: object, length: int, *, context: str) -> bytes:
    text = _require_text(value, context=context)
    if len(text) != length * 2 or text.lower() != text:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} must be lowercase exact hex")
    try:
        decoded = bytes.fromhex(text)
    except ValueError:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} is not hex")
    if len(decoded) != length:
        _fail(REJECT_SHADOW_FIELD_TYPE, f"{context} has the wrong length")
    return decoded


def _require_crypto_backend() -> None:
    if (
        _Ed25519PrivateKey is None
        or _Ed25519PublicKey is None
        or _serialization is None
        or _InvalidSignature is None
        or _cryptography is None
    ):
        _fail(
            FAIL_SHADOW_CRYPTO_BACKEND_UNAVAILABLE,
            "the Ed25519 backend is unavailable",
        )


def _require_posix_runtime() -> None:
    if os.name != "posix" or not hasattr(os, "fork"):
        _fail(
            FAIL_SHADOW_PLATFORM_UNSUPPORTED,
            "the shadow isolation runtime requires POSIX fork and anonymous pipes",
        )


def _namespace_links() -> dict[str, str]:
    result: dict[str, str] = {}
    for kind in NAMESPACE_KINDS:
        try:
            result[kind] = os.readlink(f"/proc/self/ns/{kind}")
        except OSError:
            _fail(
                FAIL_SHADOW_BWRAP_ISOLATION,
                f"cannot inspect the {kind} namespace",
            )
    return result


def _require_bwrap() -> tuple[Path, str, str]:
    executable = shutil.which("bwrap")
    if executable is None:
        _fail(FAIL_SHADOW_BWRAP_UNAVAILABLE, "bubblewrap is not installed")
    path = Path(executable).resolve()
    try:
        info = path.stat()
        payload = path.read_bytes()
    except OSError:
        _fail(FAIL_SHADOW_BWRAP_UNAVAILABLE, "bubblewrap cannot be inspected")
    if not stat.S_ISREG(info.st_mode):
        _fail(FAIL_SHADOW_BWRAP_UNAVAILABLE, "bubblewrap is not a regular file")
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(SANITIZED_ENVIRONMENT),
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        _fail(FAIL_SHADOW_BWRAP_UNAVAILABLE, "bubblewrap version probe failed")
    try:
        version = completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError:
        _fail(FAIL_SHADOW_BWRAP_UNAVAILABLE, "bubblewrap version is not ASCII")
    if completed.returncode != 0 or not version.startswith("bubblewrap "):
        _fail(FAIL_SHADOW_BWRAP_UNAVAILABLE, "unexpected bubblewrap version output")
    return path, version, hashlib.sha256(payload).hexdigest()


def _read_proc_security_status() -> tuple[int, int, dict[str, str]]:
    try:
        lines = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        _fail(FAIL_SHADOW_SECCOMP_PROBE, "cannot read /proc/self/status")
    values: dict[str, int] = {}
    capabilities: dict[str, str] = {}
    for line in lines:
        if line.startswith("Seccomp:"):
            values["Seccomp"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("NoNewPrivs:"):
            values["NoNewPrivs"] = int(line.split(":", 1)[1].strip())
        else:
            for field in CAPABILITY_STATUS_FIELDS:
                if line.startswith(field + ":"):
                    capabilities[field] = line.split(":", 1)[1].strip().lower()
    if (
        set(values) != {"Seccomp", "NoNewPrivs"}
        or set(capabilities) != set(CAPABILITY_STATUS_FIELDS)
    ):
        _fail(FAIL_SHADOW_SECCOMP_PROBE, "kernel security status fields are absent")
    if any(re.fullmatch(r"[0-9a-f]{16}", value) is None for value in capabilities.values()):
        _fail(FAIL_SHADOW_SECCOMP_PROBE, "capability status syntax differs")
    return values["Seccomp"], values["NoNewPrivs"], capabilities


def _install_seccomp_filter() -> None:
    library_name = "libseccomp.so.2"
    try:
        library = ctypes.CDLL(library_name, use_errno=True)
    except OSError:
        _fail(FAIL_SHADOW_SECCOMP_UNAVAILABLE, "libseccomp.so.2 is unavailable")

    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_release.restype = None

    scmp_act_allow = 0x7FFF0000
    scmp_act_errno_eperm = 0x00050000 | errno.EPERM
    context = library.seccomp_init(scmp_act_allow)
    if not context:
        _fail(FAIL_SHADOW_SECCOMP_UNAVAILABLE, "seccomp_init returned null")
    try:
        for syscall_name in SECCOMP_DENIED_SYSCALLS:
            number = library.seccomp_syscall_resolve_name(syscall_name.encode("ascii"))
            if number < 0:
                _fail(
                    FAIL_SHADOW_SECCOMP_UNAVAILABLE,
                    f"kernel architecture lacks required syscall {syscall_name}",
                )
            result = library.seccomp_rule_add(
                context, scmp_act_errno_eperm, number, 0
            )
            if result != 0:
                _fail(
                    FAIL_SHADOW_SECCOMP_UNAVAILABLE,
                    f"failed to add seccomp rule for {syscall_name}",
                )

        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_no_new_privs = 38
        if libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
            _fail(FAIL_SHADOW_SECCOMP_UNAVAILABLE, "PR_SET_NO_NEW_PRIVS failed")
        if library.seccomp_load(context) != 0:
            _fail(FAIL_SHADOW_SECCOMP_UNAVAILABLE, "seccomp_load failed")
    finally:
        library.seccomp_release(context)


def _probe_tmpfs_mount() -> bool:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    for line in lines:
        left, separator, right = line.partition(" - ")
        fields = left.split()
        if separator and len(fields) >= 5 and fields[4] == "/tmp":
            return right.split()[0] == "tmpfs"
    return False


def _probe_repository_read_only(ceremony_id: bytes) -> bool:
    for root in (Path("/worker-src"), Path("/basis")):
        probe = root / f".hegel-shadow-ro-probe-{ceremony_id.hex()}"
        if probe.exists():
            _fail(FAIL_SHADOW_BWRAP_ISOLATION, "read-only probe path already exists")
        try:
            fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as error:
            if error.errno not in {errno.EROFS, errno.EACCES, errno.EPERM}:
                return False
        else:
            os.close(fd)
            try:
                probe.unlink()
            except OSError:
                pass
            return False
    return True


def _resolve_seccomp_syscall(syscall_name: str) -> int:
    library_name = "libseccomp.so.2"
    try:
        library = ctypes.CDLL(library_name, use_errno=True)
    except OSError:
        _fail(FAIL_SHADOW_SECCOMP_UNAVAILABLE, "libseccomp probe backend unavailable")
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    number = library.seccomp_syscall_resolve_name(syscall_name.encode("ascii"))
    if number < 0:
        _fail(
            FAIL_SHADOW_SECCOMP_UNAVAILABLE,
            f"cannot resolve attack probe syscall {syscall_name}",
        )
    return number


def _attack_syscall_errno_rows() -> list[dict[str, str]]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rows: list[dict[str, str]] = []
    for attack_id, syscall_name, arguments in ATTACK_SYSCALL_PROBES:
        number = _resolve_seccomp_syscall(syscall_name)
        ctypes.set_errno(0)
        result = libc.syscall(number, *arguments)
        observed_errno = ctypes.get_errno()
        if result != -1 or observed_errno != errno.EPERM:
            _fail(
                FAIL_SHADOW_SECCOMP_PROBE,
                f"attack probe {attack_id} was not denied with EPERM",
            )
        rows.append({"attack_id": attack_id, "errno": "EPERM"})
    return rows


def _network_interfaces() -> list[str]:
    try:
        lines = Path("/proc/net/dev").read_text(encoding="ascii").splitlines()[2:]
    except (OSError, UnicodeDecodeError):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "cannot inspect network interfaces")
    interfaces = sorted(line.split(":", 1)[0].strip() for line in lines if ":" in line)
    if interfaces != ["lo"]:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "network namespace exposes non-loopback interface")
    return interfaces


def _lower_loopback_and_irreversibly_drop_bootstrap_capabilities() -> None:
    """Prepare the net namespace before the untrusted role worker starts.

    Bubblewrap 0.6.x deliberately raises ``lo`` in a newly-created network
    namespace.  The frozen shadow profile requires loopback to be present but
    down.  The launcher therefore gives this tiny pre-exec bootstrap only
    ``CAP_NET_ADMIN`` and ``CAP_SETPCAP``.  It lowers ``lo``, removes every
    capability from the bounding/ambient/inheritable/permitted/effective sets,
    sets no-new-privileges, verifies the result, and only then execs the role
    worker.  The role worker never executes request-controlled code while a
    capability is present.
    """

    if _open_fd_numbers() != [0, 1, 2]:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "bootstrap inherited unexpected file descriptors",
        )
    try:
        control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ifreq = struct.pack("256s", b"lo")
        flags_payload = fcntl.ioctl(control.fileno(), 0x8913, ifreq)
        flags = struct.unpack_from("H", flags_payload, 16)[0]
        lowered = bytearray(flags_payload)
        struct.pack_into("H", lowered, 16, flags & ~0x1)
        fcntl.ioctl(control.fileno(), 0x8914, bytes(lowered))
        observed = fcntl.ioctl(control.fileno(), 0x8913, ifreq)
        if struct.unpack_from("H", observed, 16)[0] & 0x1:
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "bootstrap could not lower loopback",
            )
    except OSError:
        _bootstrap_seccomp, _bootstrap_nnp, bootstrap_capabilities = (
            _read_proc_security_status()
        )
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "bootstrap loopback setup failed with capabilities "
            + json.dumps(bootstrap_capabilities, sort_keys=True),
        )
    finally:
        try:
            control.close()
        except (OSError, UnboundLocalError):
            pass

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    pr_capbset_drop = 24
    pr_set_no_new_privs = 38
    pr_cap_ambient = 47
    pr_cap_ambient_clear_all = 4

    if prctl(pr_cap_ambient, pr_cap_ambient_clear_all, 0, 0, 0) != 0:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "bootstrap could not clear ambient capabilities",
        )
    # Linux currently defines far fewer than 64 capability numbers. EINVAL for
    # an unknown future slot is expected; every known slot must be droppable.
    for capability in range(64):
        ctypes.set_errno(0)
        result = prctl(pr_capbset_drop, capability, 0, 0, 0)
        observed_errno = ctypes.get_errno()
        if result != 0 and observed_errno != errno.EINVAL:
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "bootstrap could not empty the capability bounding set",
            )

    class _CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class _CapData(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    header = _CapHeader(0x20080522, 0)
    data = (_CapData * 2)()
    capset = libc.capset
    capset.argtypes = [ctypes.POINTER(_CapHeader), ctypes.POINTER(_CapData)]
    capset.restype = ctypes.c_int
    if capset(ctypes.byref(header), data) != 0:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "bootstrap could not clear process capabilities",
        )
    if prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "bootstrap could not set no-new-privileges",
        )
    _seccomp_mode, no_new_privs, capabilities = _read_proc_security_status()
    if no_new_privs != 1 or any(
        value != "0000000000000000" for value in capabilities.values()
    ):
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "bootstrap capability erasure did not verify",
        )


def _internal_shadow_bootstrap() -> NoReturn:
    _lower_loopback_and_irreversibly_drop_bootstrap_capabilities()
    executable = str(Path(sys.executable).resolve())
    os.execve(
        executable,
        [
            executable,
            "-m",
            "hegel_machine.phase3_m3_shadow_runtime_v1",
            "--internal-shadow-worker",
        ],
        dict(os.environ),
    )
    raise AssertionError("os.execve unexpectedly returned")


def _security_evidence(
    *,
    orchestrator_namespaces: Mapping[str, str],
    ceremony_id: bytes,
    forbidden_host_paths: Mapping[str, str],
) -> dict[str, object]:
    inherited_fds = _open_fd_numbers()
    if inherited_fds != [0, 1, 2]:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "unexpected inherited file descriptor")
    try:
        interface_probe_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "cannot open pre-filter interface probe")
    _install_seccomp_filter()
    try:
        ifreq = struct.pack("256s", b"lo")
        flags_payload = fcntl.ioctl(interface_probe_socket.fileno(), 0x8913, ifreq)
        loopback_is_up = bool(struct.unpack_from("H", flags_payload, 16)[0] & 0x1)
    except OSError:
        interface_probe_socket.close()
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "cannot inspect loopback flags")
    interface_probe_socket.close()
    if loopback_is_up:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "loopback interface is not down")
    seccomp_mode, no_new_privs, capabilities = _read_proc_security_status()
    if (
        seccomp_mode != 2
        or no_new_privs != 1
        or any(value != "0000000000000000" for value in capabilities.values())
    ):
        _fail(
            FAIL_SHADOW_SECCOMP_PROBE,
            "seccomp filter or no-new-privileges live probe failed",
        )
    attack_rows = _attack_syscall_errno_rows()

    namespaces = _namespace_links()
    if set(orchestrator_namespaces) != set(NAMESPACE_KINDS):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "orchestrator namespace set differs")
    separated = {
        kind: namespaces[kind] != orchestrator_namespaces[kind]
        for kind in NAMESPACE_KINDS
    }
    if not all(separated.values()):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "required namespace was not unshared")
    if not _probe_tmpfs_mount():
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "/tmp is not an isolated tmpfs")
    if not _probe_repository_read_only(ceremony_id):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "repository mount is writable")
    required_forbidden = {
        "live_repository",
        "host_home",
        "windows_mnt_c",
        "docker_socket",
    }
    if set(forbidden_host_paths) != required_forbidden:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "forbidden host path registry differs")
    host_path_absence = {
        name: not Path(path).exists() for name, path in forbidden_host_paths.items()
    }
    if not all(host_path_absence.values()):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "forbidden host path is visible")
    cross_purpose_absence = {
        f"purpose_{purpose_id}": not Path(f"/purpose-{purpose_id}").exists()
        for purpose_id in PURPOSE_IDS
    }
    if not all(cross_purpose_absence.values()):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "cross-purpose path is visible")
    core_soft, core_hard = resource.getrlimit(resource.RLIMIT_CORE)
    if (core_soft, core_hard) != (0, 0):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "core dumps are not disabled")
    umask_probe = Path("/work/.umask-probe")
    fd = os.open(umask_probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    os.close(fd)
    umask_mode = stat.S_IMODE(umask_probe.stat().st_mode)
    umask_probe.unlink()
    if umask_mode != 0o600:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "umask 0077 live probe failed")
    return {
        "isolation_level": ISOLATION_LEVEL,
        "namespace_links": namespaces,
        "namespace_unshared_from_orchestrator": separated,
        "seccomp_mode": seccomp_mode,
        "no_new_privs": no_new_privs,
        "capability_status_hex": capabilities,
        "seccomp_forbidden_syscalls": list(SECCOMP_DENIED_SYSCALLS),
        "seccomp_forbidden_syscall_count": len(SECCOMP_DENIED_SYSCALLS),
        "attack_syscall_errno_rows": attack_rows,
        "attack_syscall_probe_count": len(attack_rows),
        "repository_mount_read_only_live_probe": True,
        "tmp_mount_type": "tmpfs",
        "network_interfaces": _network_interfaces(),
        "loopback_interface_up": False,
        "host_path_absence": host_path_absence,
        "cross_purpose_path_absence": cross_purpose_absence,
        "basis_snapshot_visible": Path("/basis").is_dir(),
        "worker_source_snapshot_visible": Path("/worker-src").is_dir(),
        "purpose_private_tmpfs": True,
        "purpose_private_home": Path("/home").is_dir(),
        "umask_0077_live_probe": True,
        "core_dump_disabled": True,
        "public_launch_request_channel": "STDIN_PUBLIC_JSON_NO_SECRET",
        "inherited_fd_numbers": inherited_fds,
        "unexpected_inherited_fd_count": 0,
        "public_evidence_output_fd": 5,
        "landlock_status": "UNAVAILABLE_NONBLOCKING_GAP_DISCLOSED",
        "transient_capability_probe_incident_count": 0,
        "network_fetch_allowed": False,
    }


def _security_evidence_digest(evidence: Mapping[str, object]) -> bytes:
    return hashlib.sha256(
        b"HEGEL/INTERNAL_SHADOW/SECURITY_EVIDENCE/V1\x00"
        + _json_bytes(evidence)
    ).digest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_state_directory(path: Path) -> Path:
    if not path.is_absolute():
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "state directory must be absolute")
    try:
        raw_stat = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "state directory does not exist")
    if stat.S_ISLNK(raw_stat.st_mode) or not stat.S_ISDIR(raw_stat.st_mode):
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "state path must be a real directory")
    if _is_inside(resolved, REPOSITORY_ROOT.resolve()):
        _fail(
            FAIL_SHADOW_STATE_INSIDE_REPOSITORY,
            "private shadow state may not be stored in the repository",
        )
    if raw_stat.st_uid != os.geteuid() or stat.S_IMODE(raw_stat.st_mode) != 0o700:
        _fail(
            FAIL_SHADOW_STATE_PERMISSIONS,
            "state directory must be owned by the current user with mode 0700",
        )
    return resolved


def create_shadow_state_directory(path: Path | str) -> Path:
    """Create or validate one external 0700 state directory.

    The directory must be outside the Git repository.  A completed or pending
    marker is never removed or overwritten by this function.
    """

    candidate = Path(path)
    if not candidate.is_absolute():
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "state directory must be absolute")
    resolved_guess = candidate.resolve(strict=False)
    if _is_inside(resolved_guess, REPOSITORY_ROOT.resolve()):
        _fail(
            FAIL_SHADOW_STATE_INSIDE_REPOSITORY,
            "private shadow state may not be stored in the repository",
        )
    try:
        candidate.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        return _validate_state_directory(candidate)
    except OSError:
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "state directory could not be created")
    return _validate_state_directory(candidate)


def _read_file_nofollow(path: Path, maximum: int, *, code: str) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW)
    except OSError:
        _fail(code, f"unable to open {path.name} without following links")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            _fail(code, f"{path.name} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(code, f"{path.name} changed during reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            _fail(code, f"{path.name} grew during reading")
        after = os.fstat(fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail(code, f"{path.name} changed during reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_all(fd: int, payload: bytes | bytearray | memoryview) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "short write to private pipe or file")
        offset += written


def _write_new_private_file(path: Path, payload: bytes) -> None:
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
        )
    except OSError:
        _fail(
            FAIL_SHADOW_PRIVATE_FILE_PERMISSIONS,
            f"private file {path.name} could not be exclusively created",
        )
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    if _mode(path) != 0o600:
        _fail(
            FAIL_SHADOW_PRIVATE_FILE_PERMISSIONS,
            f"private file {path.name} is not mode 0600",
        )


def _create_worker_source_snapshot(destination: Path) -> dict[str, object]:
    """Create a minimal read-only package; never bind the live repository."""

    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    package = destination / "hegel_machine"
    package.mkdir(mode=0o700)
    sources = {
        "__init__.py": b'"""Isolated shadow worker package."""\n',
        "phase3_m3_shadow_runtime_v1.py": Path(__file__).read_bytes(),
        "strict_cbor_v1.py": (Path(__file__).with_name("strict_cbor_v1.py")).read_bytes(),
    }
    entries: list[dict[str, object]] = []
    for name in sorted(sources):
        payload = sources[name]
        target = package / name
        _write_new_private_file(target, payload)
        os.chmod(target, 0o400)
        entries.append(
            {
                "path": f"hegel_machine/{name}",
                "size_bytes": len(payload),
                "sha256_hex": hashlib.sha256(payload).hexdigest(),
            }
        )
    os.chmod(package, 0o500)
    os.chmod(destination, 0o500)
    digest = hashlib.sha256(
        b"HEGEL/DIAGNOSTIC/SHADOW_RUNTIME/WORKER_SOURCE_SNAPSHOT/V1\x00"
        + _json_bytes(entries)
    ).hexdigest()
    return {
        "evidence_class": EVIDENCE_CLASS,
        "entry_count": len(entries),
        "entries": entries,
        "snapshot_sha256_hex": digest,
        "directory_mode_octal": "0500",
        "live_repository_bound": False,
    }


def _validate_worker_source_snapshot(
    destination: Path, report: Mapping[str, object]
) -> None:
    if _mode(destination) != 0o500:
        _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker source directory mode changed")
    package = destination / "hegel_machine"
    if _mode(package) != 0o500:
        _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker package mode changed")
    entries = report.get("entries")
    if type(entries) is not list or len(entries) != report.get("entry_count"):
        _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker source manifest differs")
    expected_names: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "size_bytes", "sha256_hex"}:
            _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker source entry differs")
        relative = _require_text(entry["path"], context="worker source path")
        target = destination / relative
        expected_names.add(Path(relative).name)
        if _mode(target) != 0o400:
            _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker source is not read-only")
        payload = _read_file_nofollow(
            target, MAX_SOURCE_BYTES, code=FAIL_SHADOW_SNAPSHOT_MUTATED
        )
        if (
            len(payload) != entry["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != entry["sha256_hex"]
        ):
            _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker source bytes changed")
    if {item.name for item in package.iterdir()} != expected_names:
        _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "worker source file set changed")


def _resolve_calculator_endpoint_bindings(
    python_calculator_path: Path | str,
    rust_calculator_path: Path | str,
) -> tuple[dict[str, Path], dict[str, str]]:
    bindings: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for endpoint_id, raw_path in (
        ("python", python_calculator_path),
        ("rust", rust_calculator_path),
    ):
        path = Path(raw_path)
        if not path.is_absolute():
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                f"{endpoint_id} calculator path must be explicit and absolute",
            )
        try:
            info = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError:
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                f"{endpoint_id} calculator endpoint is absent",
            )
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                f"{endpoint_id} calculator endpoint is not a real regular file",
            )
        if endpoint_id == "rust" and not os.access(resolved, os.X_OK):
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "Rust calculator endpoint is not executable",
            )
        payload = _read_file_nofollow(
            resolved,
            MAX_SOURCE_BYTES,
            code=FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
        )
        bindings[endpoint_id] = resolved
        digests[endpoint_id] = hashlib.sha256(payload).hexdigest()
    if bindings["python"] == bindings["rust"] or digests["python"] == digests["rust"]:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "Python and Rust calculator endpoints are not implementation-distinct",
        )
    return bindings, digests


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _snapshot_value(entries: Sequence[dict[str, object]]) -> tuple[object, ...]:
    rows: list[tuple[object, ...]] = []
    for entry in entries:
        rows.append(
            (
                str(entry["label"]).encode("ascii"),
                int(entry["size_bytes"]),
                bytes.fromhex(str(entry["sha256_hex"])),
            )
        )
    return (SNAPSHOT_DIAGNOSTIC_ID, 1, tuple(rows))


def _validate_snapshot_directory(
    snapshot_dir: Path,
    report: Mapping[str, object],
) -> None:
    if _mode(snapshot_dir) != 0o500:
        _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "snapshot directory is not mode 0500")
    expected_names: set[str] = set()
    entries = report["entries"]
    if type(entries) is not list:
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot entries must be a list")
    for raw_entry in entries:
        entry = _require_exact_fields(
            raw_entry,
            ("label", "size_bytes", "sha256_hex", "snapshot_mode_octal"),
            context="snapshot entry",
            code=FAIL_SHADOW_REPORT_INVALID,
        )
        label = _require_text(entry["label"], context="snapshot label")
        expected_names.add(label)
        target = snapshot_dir / label
        try:
            info = target.lstat()
        except OSError:
            _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "snapshot entry disappeared")
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o400:
            _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "snapshot entry is not read-only")
        payload = _read_file_nofollow(
            target,
            MAX_SOURCE_BYTES,
            code=FAIL_SHADOW_SNAPSHOT_MUTATED,
        )
        if (
            info.st_size != entry["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != entry["sha256_hex"]
            or entry["snapshot_mode_octal"] != "0400"
        ):
            _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "snapshot content or metadata changed")
    actual_names = {item.name for item in snapshot_dir.iterdir()}
    if actual_names != expected_names:
        _fail(FAIL_SHADOW_SNAPSHOT_MUTATED, "snapshot entry set changed")


def create_readonly_input_snapshot(
    inputs: Mapping[str, Path | str],
    destination: Path | str,
) -> dict[str, object]:
    """Copy labelled regular files into an exact 0500/0400 snapshot.

    This public hook runs before any seed exists.  Sources are opened with
    ``O_NOFOLLOW`` where supported and are checked before/after reading.
    """

    if type(inputs) is not dict or not inputs:
        _fail(
            FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID,
            "snapshot inputs must be a nonempty exact dictionary",
        )
    destination_path = Path(destination)
    try:
        destination_path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError:
        _fail(
            FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID,
            "snapshot destination could not be exclusively created",
        )
    os.chmod(destination_path, 0o700)
    entries: list[dict[str, object]] = []
    try:
        for label in sorted(inputs):
            if type(label) is not str or _LABEL_RE.fullmatch(label) is None:
                _fail(
                    FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID,
                    "snapshot labels must be bounded safe ASCII names",
                )
            source = Path(inputs[label])
            try:
                raw_info = source.lstat()
            except OSError:
                _fail(
                    FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID,
                    f"snapshot source {label} does not exist",
                )
            if stat.S_ISLNK(raw_info.st_mode) or not stat.S_ISREG(raw_info.st_mode):
                _fail(
                    FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID,
                    f"snapshot source {label} must be a real regular file",
                )
            payload = _read_file_nofollow(
                source,
                MAX_SOURCE_BYTES,
                code=FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID,
            )
            target = destination_path / label
            _write_new_private_file(target, payload)
            os.chmod(target, 0o400)
            entries.append(
                {
                    "label": label,
                    "size_bytes": len(payload),
                    "sha256_hex": hashlib.sha256(payload).hexdigest(),
                    "snapshot_mode_octal": "0400",
                }
            )
        value = _snapshot_value(entries)
        encoded = canonical_cbor_encode(value)
        digest = content_hash(SNAPSHOT_HASH_DOMAIN, value)
        report: dict[str, object] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "evidence_class": EVIDENCE_CLASS,
            "entry_count": len(entries),
            "entries": entries,
            "manifest_cbor_hex": encoded.hex(),
            "manifest_sha256_hex": digest.hex(),
            "directory_mode_octal": "0500",
        }
        os.chmod(destination_path, 0o500)
        _validate_snapshot_directory(destination_path, report)
        validate_readonly_input_snapshot_report(report)
        return report
    except BaseException:
        try:
            os.chmod(destination_path, 0o700)
            for item in destination_path.iterdir():
                try:
                    os.chmod(item, 0o600)
                except OSError:
                    pass
            shutil.rmtree(destination_path)
        except OSError:
            pass
        raise


def validate_readonly_input_snapshot_report(report: object) -> None:
    """Validate the strict public identity of a read-only snapshot."""

    value = _require_exact_fields(
        report,
        (
            "schema_version",
            "evidence_class",
            "entry_count",
            "entries",
            "manifest_cbor_hex",
            "manifest_sha256_hex",
            "directory_mode_octal",
        ),
        context="snapshot report",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    if (
        value["schema_version"] != SNAPSHOT_SCHEMA_VERSION
        or value["evidence_class"] != EVIDENCE_CLASS
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "unknown snapshot schema")
    if value["directory_mode_octal"] != "0500":
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot directory mode is not frozen")
    entries = value["entries"]
    if type(entries) is not list or not entries:
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot entries must be nonempty")
    labels: list[str] = []
    for raw_entry in entries:
        entry = _require_exact_fields(
            raw_entry,
            ("label", "size_bytes", "sha256_hex", "snapshot_mode_octal"),
            context="snapshot entry",
            code=FAIL_SHADOW_REPORT_INVALID,
        )
        label = _require_text(entry["label"], context="snapshot label")
        if _LABEL_RE.fullmatch(label) is None:
            _fail(FAIL_SHADOW_REPORT_INVALID, "invalid snapshot label")
        labels.append(label)
        _require_int(entry["size_bytes"], context="snapshot size")
        _decode_hex(entry["sha256_hex"], 32, context="snapshot entry digest")
        if entry["snapshot_mode_octal"] != "0400":
            _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot file mode is not frozen")
    if labels != sorted(labels) or len(labels) != len(set(labels)):
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot labels are not unique and sorted")
    if value["entry_count"] != len(entries) or type(value["entry_count"]) is not int:
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot entry count differs")
    expected_value = _snapshot_value(entries)
    encoded = _decode_hex(
        value["manifest_cbor_hex"],
        len(canonical_cbor_encode(expected_value)),
        context="snapshot manifest CBOR",
    )
    try:
        decoded = canonical_cbor_decode(encoded)
    except StrictCborError:
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot manifest CBOR is not canonical")
    if decoded != expected_value:
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot manifest bytes differ")
    digest = _decode_hex(
        value["manifest_sha256_hex"], 32, context="snapshot manifest digest"
    )
    if digest != content_hash(SNAPSHOT_HASH_DOMAIN, expected_value):
        _fail(FAIL_SHADOW_REPORT_INVALID, "snapshot manifest digest differs")


def _marker_value(
    ceremony_id: bytes,
    snapshot_digest: bytes,
    status_id: int,
    commitment: bytes | None,
) -> tuple[object, ...]:
    return (
        PRIVATE_MARKER_MAGIC,
        1,
        ceremony_id,
        snapshot_digest,
        status_id,
        commitment,
    )


def _ledger_value(
    sequence: int,
    ceremony_id: bytes,
    snapshot_digest: bytes,
    status_id: int,
    commitment: bytes | None,
) -> tuple[object, ...]:
    return (
        PRIVATE_LEDGER_MAGIC,
        1,
        sequence,
        ceremony_id,
        snapshot_digest,
        status_id,
        commitment,
    )


def _ledger_frame(value: tuple[object, ...]) -> bytes:
    payload = canonical_cbor_encode(value)
    return len(payload).to_bytes(4, "big") + payload


def _decode_marker(payload: bytes) -> tuple[object, ...]:
    try:
        value = canonical_cbor_decode(payload)
    except StrictCborError:
        _fail(FAIL_SHADOW_MARKER_TAMPERED, "marker is not strict canonical CBOR")
    if (
        type(value) is not tuple
        or len(value) != 6
        or value[0:2] != (PRIVATE_MARKER_MAGIC, 1)
        or type(value[2]) is not bytes
        or len(value[2]) != 16
        or type(value[3]) is not bytes
        or len(value[3]) != 32
        or type(value[4]) is not int
        or value[4] not in {1, 2}
        or (
            (value[4] == 1 and value[5] is not None)
            or (value[4] == 2 and (type(value[5]) is not bytes or len(value[5]) != 32))
        )
    ):
        _fail(FAIL_SHADOW_MARKER_TAMPERED, "marker fields are invalid")
    return value


def _decode_ledger(payload: bytes) -> tuple[tuple[object, ...], ...]:
    values: list[tuple[object, ...]] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 4:
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger frame length is truncated")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        if length == 0 or length > MAX_STATE_BYTES or offset + length > len(payload):
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger frame length is invalid")
        raw = payload[offset : offset + length]
        offset += length
        try:
            value = canonical_cbor_decode(raw)
        except StrictCborError:
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger entry is not canonical CBOR")
        if type(value) is not tuple or len(value) != 7:
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger entry shape is invalid")
        values.append(value)
    if len(values) not in {1, 2}:
        _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger must have one or two entries")
    first = values[0]
    expected_first = _ledger_value(1, first[3], first[4], 1, None)
    if first != expected_first:
        _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger PENDING entry is invalid")
    if len(values) == 2:
        second = values[1]
        if (
            type(second[6]) is not bytes
            or len(second[6]) != 32
            or second != _ledger_value(2, first[3], first[4], 2, second[6])
        ):
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger COMPLETE entry is invalid")
    return tuple(values)


def _assert_private_state_absent(state_dir: Path) -> None:
    marker = state_dir / MARKER_FILE_NAME
    ledger = state_dir / LEDGER_FILE_NAME
    if not marker.exists() and not ledger.exists():
        return
    if not marker.exists() or not ledger.exists():
        _fail(FAIL_SHADOW_MARKER_TAMPERED, "marker and ledger presence differs")
    report = inspect_shadow_state(state_dir)
    if report["status"] == "PENDING":
        _fail(
            FAIL_SHADOW_STATE_PENDING_RECOVERY_REQUIRED,
            "a PENDING seed genesis may not be redrawn",
        )
    _fail(
        FAIL_SHADOW_STATE_ALREADY_COMPLETE,
        "this state directory already contains a completed seed genesis",
    )


def _create_pending_state(
    state_dir: Path,
    ceremony_id: bytes,
    snapshot_digest: bytes,
) -> None:
    _assert_private_state_absent(state_dir)
    marker_value = _marker_value(ceremony_id, snapshot_digest, 1, None)
    ledger_value = _ledger_value(1, ceremony_id, snapshot_digest, 1, None)
    _write_new_private_file(
        state_dir / MARKER_FILE_NAME, canonical_cbor_encode(marker_value)
    )
    try:
        _write_new_private_file(state_dir / LEDGER_FILE_NAME, _ledger_frame(ledger_value))
    except BaseException:
        # A marker without its ledger is deliberately not rolled back; recovery
        # must be explicit and may not generate another seed.
        raise
    _fsync_directory(state_dir)


def _complete_private_state(
    state_dir: Path,
    ceremony_id: bytes,
    snapshot_digest: bytes,
    commitment: bytes,
) -> dict[str, object]:
    pending = inspect_shadow_state(state_dir)
    if (
        pending["status"] != "PENDING"
        or pending["ceremony_id_hex"] != ceremony_id.hex()
        or pending["snapshot_manifest_sha256_hex"] != snapshot_digest.hex()
    ):
        _fail(FAIL_SHADOW_MARKER_TAMPERED, "PENDING state identity differs")

    ledger_path = state_dir / LEDGER_FILE_NAME
    ledger_payload = _read_file_nofollow(
        ledger_path, MAX_STATE_BYTES, code=FAIL_SHADOW_LEDGER_TAMPERED
    )
    values = _decode_ledger(ledger_payload)
    if len(values) != 1:
        _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger is not at PENDING")
    complete_entry = _ledger_value(2, ceremony_id, snapshot_digest, 2, commitment)
    try:
        fd = os.open(ledger_path, os.O_WRONLY | os.O_APPEND | _NOFOLLOW)
    except OSError:
        _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger cannot be opened for append")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "ledger permissions changed")
        _write_all(fd, _ledger_frame(complete_entry))
        os.fsync(fd)
    finally:
        os.close(fd)

    marker_path = state_dir / MARKER_FILE_NAME
    complete_marker = canonical_cbor_encode(
        _marker_value(ceremony_id, snapshot_digest, 2, commitment)
    )
    temporary = state_dir / (MARKER_FILE_NAME + ".complete.tmp")
    _write_new_private_file(temporary, complete_marker)
    try:
        os.replace(temporary, marker_path)
    except OSError:
        _fail(FAIL_SHADOW_MARKER_TAMPERED, "marker atomic replacement failed")
    _fsync_directory(state_dir)
    return inspect_shadow_state(state_dir)


def inspect_shadow_state(state_directory: Path | str) -> dict[str, object]:
    """Return a secret-free receipt for a PENDING or COMPLETE private state."""

    state_dir = _validate_state_directory(Path(state_directory))
    marker_path = state_dir / MARKER_FILE_NAME
    ledger_path = state_dir / LEDGER_FILE_NAME
    try:
        marker_mode = _mode(marker_path)
        ledger_mode = _mode(ledger_path)
    except OSError:
        _fail(FAIL_SHADOW_MARKER_TAMPERED, "marker or ledger is absent")
    if marker_mode != 0o600 or ledger_mode != 0o600:
        _fail(
            FAIL_SHADOW_PRIVATE_FILE_PERMISSIONS,
            "marker and ledger must both be mode 0600",
        )
    marker_payload = _read_file_nofollow(
        marker_path, MAX_STATE_BYTES, code=FAIL_SHADOW_MARKER_TAMPERED
    )
    ledger_payload = _read_file_nofollow(
        ledger_path, MAX_STATE_BYTES, code=FAIL_SHADOW_LEDGER_TAMPERED
    )
    marker = _decode_marker(marker_payload)
    ledger = _decode_ledger(ledger_payload)
    status_id = marker[4]
    if (
        ledger[-1][2] != status_id
        or ledger[-1][3] != marker[2]
        or ledger[-1][4] != marker[3]
        or ledger[-1][5] != marker[4]
        or ledger[-1][6] != marker[5]
        or len(ledger) != status_id
    ):
        _fail(FAIL_SHADOW_LEDGER_TAMPERED, "marker and ledger heads differ")
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "status": "PENDING" if status_id == 1 else "COMPLETE",
        "ceremony_id_hex": marker[2].hex(),
        "snapshot_manifest_sha256_hex": marker[3].hex(),
        "seed_commitment_sha256_hex_or_null": (
            None if marker[5] is None else marker[5].hex()
        ),
        "ledger_entry_count": len(ledger),
        "marker_sha256_hex": hashlib.sha256(marker_payload).hexdigest(),
        "ledger_sha256_hex": hashlib.sha256(ledger_payload).hexdigest(),
        "marker_mode_octal": "0600",
        "ledger_mode_octal": "0600",
        "contains_raw_seed": False,
        "contains_private_key": False,
    }


def _sanitize_child(role_dir: Path) -> tuple[list[str], str]:
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(SANITIZED_ENVIRONMENT)
    os.chdir(role_dir)
    return sorted(os.environ), f"{_mode(role_dir):04o}"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _public_output_frame(value: object) -> bytes:
    diagnostic_payload = canonical_cbor_encode(
        (PUBLIC_OUTPUT_DIAGNOSTIC_ID, 1, _json_bytes(value))
    )
    return len(diagnostic_payload).to_bytes(8, "big") + diagnostic_payload


def _decode_public_output_frame(frame: bytes) -> object:
    if len(frame) < 8:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "public FD5 frame is truncated")
    length = int.from_bytes(frame[:8], "big")
    if length == 0 or length > MAX_IPC_BYTES or len(frame) != 8 + length:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "public FD5 frame length differs")
    try:
        decoded = canonical_cbor_decode(frame[8:])
    except StrictCborError:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "public FD5 payload is not canonical CBOR")
    if (
        type(decoded) is not tuple
        or len(decoded) != 3
        or decoded[0:2] != (PUBLIC_OUTPUT_DIAGNOSTIC_ID, 1)
        or type(decoded[2]) is not bytes
    ):
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "public FD5 payload shape differs")
    try:
        value = json.loads(decoded[2].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "public FD5 diagnostic JSON differs")
    _lint_public_output(value)
    return value


def _lint_public_output(value: object) -> None:
    forbidden_keys = {
        "raw_seed",
        "raw_seed_hex",
        "master_seed",
        "master_seed_hex",
        "private_key",
        "private_key_hex",
        "ed25519_private_key",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str or key.lower() in forbidden_keys:
                _fail(
                    FAIL_SHADOW_IPC_PROTOCOL,
                    "public output contains a secret field",
                )
            _lint_public_output(item)
    elif isinstance(value, list):
        for item in value:
            _lint_public_output(item)
    elif isinstance(value, str):
        upper = value.upper()
        if "BEGIN PRIVATE KEY" in upper or "BEGIN OPENSSH PRIVATE KEY" in upper:
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "public output contains a private-key header")


def _open_fd_numbers(limit: int = 64) -> list[int]:
    result: list[int] = []
    for fd in range(limit):
        try:
            fcntl.fcntl(fd, fcntl.F_GETFD)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
        else:
            result.append(fd)
    return result


def _read_pipe_to_eof(fd: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(65536, MAX_IPC_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_IPC_BYTES:
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "child response exceeds IPC bound")
    return b"".join(chunks)


def _fork_json_worker(
    worker: Callable[[], dict[str, object]],
) -> tuple[dict[str, object], int]:
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        _fail(FAIL_SHADOW_ROLE_PROCESS, "could not fork role process")
    if pid == 0:  # pragma: no branch - child path is observed via IPC.
        os.close(read_fd)
        try:
            result = {"ok": True, "result": worker()}
        except ShadowRuntimeError as error:
            result = {"ok": False, "code": error.code, "detail": error.detail}
        except BaseException as error:  # Never serialize exception values/secrets.
            result = {
                "ok": False,
                "code": FAIL_SHADOW_ROLE_PROCESS,
                "detail": f"child raised {type(error).__name__}",
            }
        try:
            _write_all(write_fd, _json_bytes(result))
        except BaseException:
            pass
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    try:
        payload = _read_pipe_to_eof(read_fd)
    finally:
        os.close(read_fd)
    waited_pid, status = os.waitpid(pid, 0)
    if waited_pid != pid or status != 0:
        _fail(FAIL_SHADOW_ROLE_PROCESS, "role child exited abnormally")
    try:
        response = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "child response is not strict JSON")
    if type(response) is not dict or type(response.get("ok")) is not bool:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "child response envelope is malformed")
    if response["ok"] is False:
        if set(response) != {"ok", "code", "detail"}:
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "child error field set is malformed")
        code = _require_text(response["code"], context="child error code")
        detail = _require_text(response["detail"], context="child error detail")
        _fail(code, detail)
    if set(response) != {"ok", "result"} or type(response["result"]) is not dict:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "child success field set is malformed")
    return response["result"], pid


def _calculator_child_fd_setup() -> None:
    for fd in (0, 1, 2):
        try:
            os.close(fd)
        except OSError:
            pass


def _decode_calculator_endpoint_frame(frame: bytes) -> bytes:
    if len(frame) < 8:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "calculator FD5 frame is truncated")
    length = int.from_bytes(frame[:8], "big")
    payload = frame[8:]
    if length != len(payload) or length == 0 or length > 256:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "calculator FD5 length differs")
    try:
        decoded = canonical_cbor_decode(payload)
    except StrictCborError:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "calculator FD5 CBOR is invalid")
    if (
        canonical_cbor_encode(decoded) != payload
        or type(decoded) is not tuple
        or len(decoded) != 3
        or decoded[0:2] != (1, CALCULATOR_ENDPOINT_RESPONSE_SCHEMA_ID)
        or type(decoded[2]) is not bytes
        or len(decoded[2]) != 32
    ):
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "calculator FD5 schema differs")
    return decoded[2]


def _run_external_calculator_endpoint(
    *, endpoint_id: str, secret: bytearray, role_dir: Path
) -> dict[str, object]:
    if endpoint_id not in CALCULATOR_ENDPOINT_PATHS:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "calculator endpoint ID differs")
    endpoint_path = CALCULATOR_ENDPOINT_PATHS[endpoint_id]
    if not endpoint_path.is_file():
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            f"bound {endpoint_id} calculator endpoint is absent",
        )
    command = (
        [str(Path(sys.executable).resolve()), str(endpoint_path)]
        if endpoint_id == "python"
        else [str(endpoint_path)]
    )
    if _open_fd_numbers() != [0, 1, 2]:
        _fail(
            FAIL_SHADOW_CALCULATOR_PROCESS,
            "custodian FD table is not clean before calculator launch",
        )
    seed_read_fd, seed_write_fd = os.pipe()
    response_read_fd, response_write_fd = os.pipe()
    if (seed_read_fd, seed_write_fd, response_read_fd, response_write_fd) != (
        3,
        4,
        5,
        6,
    ):
        for fd in (seed_read_fd, seed_write_fd, response_read_fd, response_write_fd):
            os.close(fd)
        _fail(
            FAIL_SHADOW_CALCULATOR_PROCESS,
            "calculator contract FDs could not be allocated exactly",
        )
    response_parent_fd = fcntl.fcntl(
        response_read_fd, fcntl.F_DUPFD_CLOEXEC, 10
    )
    os.close(response_read_fd)
    os.dup2(response_write_fd, 5, inheritable=True)
    os.close(response_write_fd)
    response_write_fd = 5
    os.set_inheritable(seed_read_fd, True)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(seed_read_fd, response_write_fd),
            preexec_fn=_calculator_child_fd_setup,
            cwd=role_dir,
            env=dict(SANITIZED_ENVIRONMENT),
        )
    except OSError:
        for fd in (
            seed_read_fd,
            seed_write_fd,
            response_parent_fd,
            response_write_fd,
        ):
            try:
                os.close(fd)
            except OSError:
                pass
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "could not launch calculator endpoint")
    os.close(seed_read_fd)
    os.close(response_write_fd)
    try:
        _write_all(seed_write_fd, secret)
    finally:
        os.close(seed_write_fd)
    try:
        payload = _read_pipe_to_eof(response_parent_fd)
    finally:
        os.close(response_parent_fd)
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "calculator endpoint timed out")
    if process.returncode != 0 or stdout or stderr:
        _fail(
            FAIL_SHADOW_CALCULATOR_PROCESS,
            f"{endpoint_id} calculator endpoint failed closed "
            f"(rc={process.returncode},fd5_bytes={len(payload)},"
            f"stdout_bytes={len(stdout)},stderr_bytes={len(stderr)})",
        )
    digest = _decode_calculator_endpoint_frame(payload)
    return {
        "calculator_id": CALCULATOR_ENDPOINT_IDS[endpoint_id],
        "process_id": process.pid,
        "seed_commitment_sha256_hex": digest.hex(),
        "environment_keys": sorted(SANITIZED_ENVIRONMENT),
        "working_directory_mode_octal": f"{_mode(role_dir):04o}",
        "secret_input_fd": 3,
        "public_output_fd": 5,
        "unexpected_inherited_fd_count": 0,
        "seccomp_mode": 2,
    }


def _qualify_bound_calculator_endpoints(role_dir: Path) -> dict[str, object]:
    directories = {
        endpoint_id: role_dir / f"qualification_{endpoint_id}"
        for endpoint_id in ("python", "rust")
    }
    for directory in directories.values():
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    secret = bytearray(CALCULATOR_QUALIFICATION_SEED)
    try:
        workers = [
            _run_external_calculator_endpoint(
                endpoint_id=endpoint_id,
                secret=secret,
                role_dir=directories[endpoint_id],
            )
            for endpoint_id in ("python", "rust")
        ]
    finally:
        for index in range(len(secret)):
            secret[index] = 0
    commitments = {
        str(worker["seed_commitment_sha256_hex"]) for worker in workers
    }
    if commitments != {CALCULATOR_QUALIFICATION_COMMITMENT.hex()}:
        _fail(
            FAIL_SHADOW_CALCULATOR_DISAGREEMENT,
            "Python/Rust FD3 qualification vector differs",
        )
    endpoint_digests = {
        endpoint_id: hashlib.sha256(
            CALCULATOR_ENDPOINT_PATHS[endpoint_id].read_bytes()
        ).hexdigest()
        for endpoint_id in ("python", "rust")
    }
    return {
        "evidence_class": EVIDENCE_CLASS,
        "status": "DUAL_PYTHON_RUST_FD3_ENDPOINTS_BIT_EXACT_PASS",
        "response_schema_id_hex": CALCULATOR_ENDPOINT_RESPONSE_SCHEMA_ID.hex(),
        "known_commitment_sha256_hex": CALCULATOR_QUALIFICATION_COMMITMENT.hex(),
        "endpoint_sha256_hex": endpoint_digests,
        "workers": workers,
        "contains_raw_seed": False,
    }


def _calculator_receipt_value(
    ceremony_id: bytes,
    snapshot_digest: bytes,
    workers: Sequence[Mapping[str, object]],
) -> tuple[object, ...]:
    return (
        CALCULATOR_DIAGNOSTIC_ID,
        1,
        ceremony_id,
        snapshot_digest,
        tuple(
            (
                str(worker["calculator_id"]).encode("ascii"),
                int(worker["process_id"]),
                bytes.fromhex(str(worker["seed_commitment_sha256_hex"])),
                tuple(str(item).encode("ascii") for item in worker["environment_keys"]),
                str(worker["working_directory_mode_octal"]).encode("ascii"),
                int(worker["secret_input_fd"]),
                int(worker["public_output_fd"]),
                int(worker["unexpected_inherited_fd_count"]),
                int(worker["seccomp_mode"]),
            )
            for worker in workers
        ),
    )


def _build_calculator_agreement(
    ceremony_id: bytes,
    snapshot_digest: bytes,
    workers: Sequence[dict[str, object]],
) -> dict[str, object]:
    if len(workers) != 2:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "exactly two calculators are required")
    digests = [worker.get("seed_commitment_sha256_hex") for worker in workers]
    if len(set(digests)) != 1:
        _fail(
            FAIL_SHADOW_CALCULATOR_DISAGREEMENT,
            "the two anonymous-pipe calculator results differ",
        )
    value = _calculator_receipt_value(ceremony_id, snapshot_digest, workers)
    encoded = canonical_cbor_encode(value)
    return {
        "schema_version": CALCULATOR_SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "contract_id": "M25_SPLIT_SEED_COMMITMENT_V1",
        "seed_transport": "ANONYMOUS_PIPE_FD_ONLY",
        "workers": list(workers),
        "agreement": True,
        "seed_commitment_sha256_hex": digests[0],
        "receipt_cbor_hex": encoded.hex(),
        "receipt_sha256_hex": content_hash(CALCULATOR_HASH_DOMAIN, value).hex(),
    }


def _validate_calculator_agreement(report: object) -> dict[str, object]:
    value = _require_exact_fields(
        report,
        (
            "schema_version",
            "evidence_class",
            "contract_id",
            "seed_transport",
            "workers",
            "agreement",
            "seed_commitment_sha256_hex",
            "receipt_cbor_hex",
            "receipt_sha256_hex",
        ),
        context="calculator agreement",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    if (
        value["schema_version"] != CALCULATOR_SCHEMA_VERSION
        or value["evidence_class"] != EVIDENCE_CLASS
        or value["contract_id"] != "M25_SPLIT_SEED_COMMITMENT_V1"
        or value["seed_transport"] != "ANONYMOUS_PIPE_FD_ONLY"
        or _require_bool(value["agreement"], context="calculator agreement") is not True
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator fixed metadata differs")
    workers = value["workers"]
    if type(workers) is not list or len(workers) != 2:
        _fail(FAIL_SHADOW_REPORT_INVALID, "exactly two calculator workers are required")
    expected_ids = ("PYTHON_FD3_ENDPOINT_V1", "RUST_FD3_ENDPOINT_V1")
    pids: list[int] = []
    digests: list[bytes] = []
    for index, raw_worker in enumerate(workers):
        worker = _require_exact_fields(
            raw_worker,
            (
                "calculator_id",
                "process_id",
                "seed_commitment_sha256_hex",
                "environment_keys",
                "working_directory_mode_octal",
                "secret_input_fd",
                "public_output_fd",
                "unexpected_inherited_fd_count",
                "seccomp_mode",
            ),
            context="calculator worker",
            code=FAIL_SHADOW_REPORT_INVALID,
        )
        if worker["calculator_id"] != expected_ids[index]:
            _fail(FAIL_SHADOW_REPORT_INVALID, "calculator ordering differs")
        pids.append(_require_int(worker["process_id"], context="calculator PID", minimum=1))
        digests.append(
            _decode_hex(
                worker["seed_commitment_sha256_hex"],
                32,
                context="calculator digest",
            )
        )
        if worker["environment_keys"] != sorted(SANITIZED_ENVIRONMENT):
            _fail(FAIL_SHADOW_REPORT_INVALID, "calculator environment was not sanitized")
        if worker["working_directory_mode_octal"] != "0700":
            _fail(FAIL_SHADOW_REPORT_INVALID, "calculator directory was not mode 0700")
        if (
            worker["secret_input_fd"] != 3
            or type(worker["secret_input_fd"]) is not int
            or worker["public_output_fd"] != 5
            or type(worker["public_output_fd"]) is not int
            or worker["unexpected_inherited_fd_count"] != 0
            or type(worker["unexpected_inherited_fd_count"]) is not int
            or worker["seccomp_mode"] != 2
            or type(worker["seccomp_mode"]) is not int
        ):
            _fail(
                FAIL_SHADOW_REPORT_INVALID,
                "calculator FD3/seccomp evidence differs",
            )
    if len(set(pids)) != 2 or len(set(digests)) != 1:
        _fail(FAIL_SHADOW_CALCULATOR_DISAGREEMENT, "calculator isolation/agreement differs")
    commitment = _decode_hex(
        value["seed_commitment_sha256_hex"], 32, context="seed commitment"
    )
    if commitment != digests[0]:
        _fail(FAIL_SHADOW_REPORT_INVALID, "agreement commitment differs")
    # The ceremony/snapshot fields are recovered from and bound by the exact CBOR.
    receipt_hex = _require_text(value["receipt_cbor_hex"], context="calculator CBOR")
    try:
        receipt_bytes = bytes.fromhex(receipt_hex)
        decoded = canonical_cbor_decode(receipt_bytes)
    except (ValueError, StrictCborError):
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator receipt CBOR is invalid")
    if canonical_cbor_encode(decoded).hex() != receipt_hex:
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator receipt CBOR is not exact")
    if type(decoded) is not tuple or len(decoded) != 5 or decoded[0:2] != (
        CALCULATOR_DIAGNOSTIC_ID,
        1,
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator receipt shape differs")
    if decoded[4] != _calculator_receipt_value(decoded[2], decoded[3], workers)[4]:
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator receipt worker rows differ")
    digest = _decode_hex(
        value["receipt_sha256_hex"], 32, context="calculator receipt digest"
    )
    if digest != content_hash(CALCULATOR_HASH_DOMAIN, decoded):
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator receipt digest differs")
    return value


def _envelope_payload(
    *,
    purpose_id: int,
    ceremony_id: bytes,
    snapshot_digest: bytes,
    calculator_receipt_digest: bytes,
    seed_commitment: bytes,
    marker_digest: bytes,
    ledger_digest: bytes,
    worker_instance_id: bytes,
    security_evidence_digest: bytes,
    process_id: int,
) -> tuple[object, ...]:
    return (
        ENVELOPE_DIAGNOSTIC_ID,
        1,
        AUTHORITY_CLASS.encode("ascii"),
        purpose_id,
        PURPOSE_ROLES[purpose_id].encode("ascii"),
        ceremony_id,
        snapshot_digest,
        calculator_receipt_digest,
        seed_commitment,
        marker_digest,
        ledger_digest,
        worker_instance_id,
        security_evidence_digest,
        process_id,
    )


def _signature_preimage(purpose_id: int, payload: bytes) -> bytes:
    return SIGNATURE_DOMAIN + b"\x00" + purpose_id.to_bytes(2, "big") + payload


def _build_envelope(
    *,
    private_key: object,
    purpose_id: int,
    ceremony_id: bytes,
    snapshot_digest: bytes,
    calculator_receipt_digest: bytes,
    seed_commitment: bytes,
    marker_digest: bytes,
    ledger_digest: bytes,
    worker_instance_id: bytes,
    security_evidence: Mapping[str, object],
    process_id: int,
) -> dict[str, object]:
    _require_crypto_backend()
    assert _serialization is not None
    payload_value = _envelope_payload(
        purpose_id=purpose_id,
        ceremony_id=ceremony_id,
        snapshot_digest=snapshot_digest,
        calculator_receipt_digest=calculator_receipt_digest,
        seed_commitment=seed_commitment,
        marker_digest=marker_digest,
        ledger_digest=ledger_digest,
        worker_instance_id=worker_instance_id,
        security_evidence_digest=_security_evidence_digest(security_evidence),
        process_id=process_id,
    )
    payload = canonical_cbor_encode(payload_value)
    public_key = private_key.public_key().public_bytes(
        encoding=_serialization.Encoding.Raw,
        format=_serialization.PublicFormat.Raw,
    )
    key_id = hashlib.sha256(KEY_ID_DOMAIN + b"\x00" + public_key).digest()[:16]
    signature = private_key.sign(_signature_preimage(purpose_id, payload))
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "authority_class": AUTHORITY_CLASS,
        "purpose_id": purpose_id,
        "role": PURPOSE_ROLES[purpose_id],
        "key_id_hex": key_id.hex(),
        "public_key_hex": public_key.hex(),
        "payload_cbor_hex": payload.hex(),
        "payload_sha256_hex": hashlib.sha256(payload).hexdigest(),
        "signature_domain": SIGNATURE_DOMAIN.decode("ascii"),
        "signature_hex": signature.hex(),
    }


def verify_shadow_envelope(envelope: object) -> dict[str, object]:
    """Replay one public purpose-specific Ed25519 envelope."""

    _require_crypto_backend()
    value = _require_exact_fields(
        envelope,
        (
            "schema_version",
            "evidence_class",
            "authority_class",
            "purpose_id",
            "role",
            "key_id_hex",
            "public_key_hex",
            "payload_cbor_hex",
            "payload_sha256_hex",
            "signature_domain",
            "signature_hex",
        ),
        context="shadow envelope",
        code=FAIL_SHADOW_SIGNATURE_INVALID,
    )
    purpose_id = _require_int(value["purpose_id"], context="purpose ID", minimum=1)
    if (
        value["schema_version"] != ENVELOPE_SCHEMA_VERSION
        or value["evidence_class"] != EVIDENCE_CLASS
        or value["authority_class"] != AUTHORITY_CLASS
        or purpose_id not in PURPOSE_ROLES
        or value["role"] != PURPOSE_ROLES[purpose_id]
        or value["signature_domain"] != SIGNATURE_DOMAIN.decode("ascii")
    ):
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "envelope fixed metadata differs")
    public_key = _decode_hex(value["public_key_hex"], 32, context="public key")
    key_id = _decode_hex(value["key_id_hex"], 16, context="key ID")
    expected_key_id = hashlib.sha256(KEY_ID_DOMAIN + b"\x00" + public_key).digest()[:16]
    if key_id != expected_key_id:
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "key ID differs from public key")
    payload_hex = _require_text(value["payload_cbor_hex"], context="envelope payload")
    try:
        payload = bytes.fromhex(payload_hex)
        decoded = canonical_cbor_decode(payload)
    except (ValueError, StrictCborError):
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "payload is not exact canonical CBOR")
    if canonical_cbor_encode(decoded).hex() != payload_hex:
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "payload hex is not canonical lowercase")
    if type(decoded) is not tuple or len(decoded) != 14:
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "payload shape differs")
    if (
        decoded[0:5]
        != (
            ENVELOPE_DIAGNOSTIC_ID,
            1,
            AUTHORITY_CLASS.encode("ascii"),
            purpose_id,
            PURPOSE_ROLES[purpose_id].encode("ascii"),
        )
        or any(type(decoded[index]) is not bytes for index in range(5, 13))
        or tuple(len(decoded[index]) for index in range(5, 13))
        != (16, 32, 32, 32, 32, 32, 16, 32)
        or type(decoded[13]) is not int
        or decoded[13] < 1
    ):
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "payload field types differ")
    if _decode_hex(
        value["payload_sha256_hex"], 32, context="payload digest"
    ) != hashlib.sha256(payload).digest():
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "payload digest differs")
    signature = _decode_hex(value["signature_hex"], 64, context="signature")
    assert _Ed25519PublicKey is not None
    try:
        _Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, _signature_preimage(purpose_id, payload)
        )
    except (ValueError, _InvalidSignature):
        _fail(FAIL_SHADOW_SIGNATURE_INVALID, "Ed25519 signature verification failed")
    return {
        "purpose_id": purpose_id,
        "role": PURPOSE_ROLES[purpose_id],
        "key_id_hex": key_id.hex(),
        "ceremony_id_hex": decoded[5].hex(),
        "snapshot_manifest_sha256_hex": decoded[6].hex(),
        "calculator_receipt_sha256_hex": decoded[7].hex(),
        "seed_commitment_sha256_hex": decoded[8].hex(),
        "marker_sha256_hex": decoded[9].hex(),
        "ledger_sha256_hex": decoded[10].hex(),
        "worker_instance_id_hex": decoded[11].hex(),
        "security_evidence_sha256_hex": decoded[12].hex(),
        "process_id": decoded[13],
    }


def _new_private_key() -> object:
    _require_crypto_backend()
    assert _Ed25519PrivateKey is not None
    return _Ed25519PrivateKey.generate()


def _generate_split_seed() -> bytearray:
    try:
        return bytearray(os.getrandom(32))
    except (AttributeError, OSError):
        try:
            return bytearray(os.urandom(32))
        except OSError:
            _fail(FAIL_SHADOW_ROLE_PROCESS, "CSPRNG seed generation failed")


def _random_id16() -> bytes:
    try:
        value = os.getrandom(16)
    except (AttributeError, OSError):
        value = os.urandom(16)
    if value == bytes(16):
        _fail(FAIL_SHADOW_ROLE_PROCESS, "CSPRNG returned an all-zero identifier")
    return value


def _role_process_record(
    role: str,
    purpose_id: int,
    environment_keys: list[str],
    cwd_mode: str,
    security_evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "role": role,
        "purpose_id": purpose_id,
        "process_id": os.getpid(),
        "environment_keys": environment_keys,
        "working_directory_mode_octal": cwd_mode,
        "security_evidence": dict(security_evidence),
    }


def _custodian_worker(
    *,
    state_dir: Path,
    role_dir: Path,
    calculator_dirs: tuple[Path, Path],
    ceremony_id: bytes,
    snapshot_digest: bytes,
    environment_keys: list[str],
    cwd_mode: str,
    security_evidence: Mapping[str, object],
) -> dict[str, object]:
    private_key = _new_private_key()
    worker_instance_id = _random_id16()
    _create_pending_state(state_dir, ceremony_id, snapshot_digest)
    secret = _generate_split_seed()
    if len(secret) != 32:
        _fail(FAIL_SHADOW_ROLE_PROCESS, "CSPRNG returned a non-32-byte seed")
    try:
        workers = [
            _run_external_calculator_endpoint(
                endpoint_id="python",
                secret=secret,
                role_dir=calculator_dirs[0],
            ),
            _run_external_calculator_endpoint(
                endpoint_id="rust",
                secret=secret,
                role_dir=calculator_dirs[1],
            ),
        ]
    finally:
        for index in range(len(secret)):
            secret[index] = 0
    calculator_agreement = _build_calculator_agreement(
        ceremony_id, snapshot_digest, workers
    )
    commitment = bytes.fromhex(
        str(calculator_agreement["seed_commitment_sha256_hex"])
    )
    state_evidence = _complete_private_state(
        state_dir, ceremony_id, snapshot_digest, commitment
    )
    envelope = _build_envelope(
        private_key=private_key,
        purpose_id=1,
        ceremony_id=ceremony_id,
        snapshot_digest=snapshot_digest,
        calculator_receipt_digest=bytes.fromhex(
            str(calculator_agreement["receipt_sha256_hex"])
        ),
        seed_commitment=commitment,
        marker_digest=bytes.fromhex(str(state_evidence["marker_sha256_hex"])),
        ledger_digest=bytes.fromhex(str(state_evidence["ledger_sha256_hex"])),
        worker_instance_id=worker_instance_id,
        security_evidence=security_evidence,
        process_id=os.getpid(),
    )
    return {
        "calculator_agreement": calculator_agreement,
        "state_evidence": state_evidence,
        "envelope": envelope,
        "process": _role_process_record(
            "CUSTODIAN", 1, environment_keys, cwd_mode, security_evidence
        ),
    }


def _attester_worker(
    *,
    purpose_id: int,
    role_dir: Path,
    ceremony_id: bytes,
    snapshot_digest: bytes,
    calculator_receipt_digest: bytes,
    seed_commitment: bytes,
    marker_digest: bytes,
    ledger_digest: bytes,
    environment_keys: list[str],
    cwd_mode: str,
    security_evidence: Mapping[str, object],
) -> dict[str, object]:
    private_key = _new_private_key()
    worker_instance_id = _random_id16()
    envelope = _build_envelope(
        private_key=private_key,
        purpose_id=purpose_id,
        ceremony_id=ceremony_id,
        snapshot_digest=snapshot_digest,
        calculator_receipt_digest=calculator_receipt_digest,
        seed_commitment=seed_commitment,
        marker_digest=marker_digest,
        ledger_digest=ledger_digest,
        worker_instance_id=worker_instance_id,
        security_evidence=security_evidence,
        process_id=os.getpid(),
    )
    return {
        "envelope": envelope,
        "process": _role_process_record(
            PURPOSE_ROLES[purpose_id],
            purpose_id,
            environment_keys,
            cwd_mode,
            security_evidence,
        ),
    }


def _validate_state_evidence(value: object) -> dict[str, object]:
    report = _require_exact_fields(
        value,
        (
            "schema_version",
            "evidence_class",
            "status",
            "ceremony_id_hex",
            "snapshot_manifest_sha256_hex",
            "seed_commitment_sha256_hex_or_null",
            "ledger_entry_count",
            "marker_sha256_hex",
            "ledger_sha256_hex",
            "marker_mode_octal",
            "ledger_mode_octal",
            "contains_raw_seed",
            "contains_private_key",
        ),
        context="state evidence",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    if (
        report["schema_version"] != STATE_SCHEMA_VERSION
        or report["evidence_class"] != EVIDENCE_CLASS
        or report["status"] != "COMPLETE"
        or report["ledger_entry_count"] != 2
        or type(report["ledger_entry_count"]) is not int
        or report["marker_mode_octal"] != "0600"
        or report["ledger_mode_octal"] != "0600"
        or _require_bool(report["contains_raw_seed"], context="raw seed flag") is not False
        or _require_bool(report["contains_private_key"], context="private key flag") is not False
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "private state evidence differs")
    _decode_hex(report["ceremony_id_hex"], 16, context="state ceremony ID")
    for field in (
        "snapshot_manifest_sha256_hex",
        "seed_commitment_sha256_hex_or_null",
        "marker_sha256_hex",
        "ledger_sha256_hex",
    ):
        _decode_hex(report[field], 32, context=f"state {field}")
    return report


def _validate_security_evidence(value: object) -> dict[str, object]:
    evidence = _require_exact_fields(
        value,
        (
            "isolation_level",
            "namespace_links",
            "namespace_unshared_from_orchestrator",
            "seccomp_mode",
            "no_new_privs",
            "capability_status_hex",
            "seccomp_forbidden_syscalls",
            "seccomp_forbidden_syscall_count",
            "attack_syscall_errno_rows",
            "attack_syscall_probe_count",
            "repository_mount_read_only_live_probe",
            "tmp_mount_type",
            "network_interfaces",
            "loopback_interface_up",
            "host_path_absence",
            "cross_purpose_path_absence",
            "basis_snapshot_visible",
            "worker_source_snapshot_visible",
            "purpose_private_tmpfs",
            "purpose_private_home",
            "umask_0077_live_probe",
            "core_dump_disabled",
            "public_launch_request_channel",
            "inherited_fd_numbers",
            "unexpected_inherited_fd_count",
            "public_evidence_output_fd",
            "landlock_status",
            "transient_capability_probe_incident_count",
            "network_fetch_allowed",
        ),
        context="security evidence",
        code=FAIL_SHADOW_BWRAP_ISOLATION,
    )
    if (
        evidence["isolation_level"] != ISOLATION_LEVEL
        or evidence["seccomp_mode"] != 2
        or type(evidence["seccomp_mode"]) is not int
        or evidence["no_new_privs"] != 1
        or type(evidence["no_new_privs"]) is not int
        or evidence["capability_status_hex"]
        != {field: "0000000000000000" for field in CAPABILITY_STATUS_FIELDS}
        or evidence["seccomp_forbidden_syscalls"] != list(SECCOMP_DENIED_SYSCALLS)
        or evidence["seccomp_forbidden_syscall_count"] != len(SECCOMP_DENIED_SYSCALLS)
        or type(evidence["seccomp_forbidden_syscall_count"]) is not int
        or evidence["attack_syscall_errno_rows"]
        != [
            {"attack_id": attack_id, "errno": "EPERM"}
            for attack_id, _syscall_name, _arguments in ATTACK_SYSCALL_PROBES
        ]
        or evidence["attack_syscall_probe_count"] != len(ATTACK_SYSCALL_PROBES)
        or type(evidence["attack_syscall_probe_count"]) is not int
        or _require_bool(
            evidence["repository_mount_read_only_live_probe"],
            context="repository read-only probe",
        )
        is not True
        or evidence["tmp_mount_type"] != "tmpfs"
        or evidence["network_interfaces"] != ["lo"]
        or _require_bool(
            evidence["loopback_interface_up"], context="loopback interface up"
        )
        is not False
        or evidence["host_path_absence"]
        != {
            "live_repository": True,
            "host_home": True,
            "windows_mnt_c": True,
            "docker_socket": True,
        }
        or evidence["cross_purpose_path_absence"]
        != {f"purpose_{purpose_id}": True for purpose_id in PURPOSE_IDS}
        or _require_bool(evidence["basis_snapshot_visible"], context="basis visible")
        is not True
        or _require_bool(
            evidence["worker_source_snapshot_visible"], context="worker source visible"
        )
        is not True
        or _require_bool(
            evidence["purpose_private_tmpfs"], context="private tmpfs"
        )
        is not True
        or _require_bool(
            evidence["purpose_private_home"], context="private home"
        )
        is not True
        or _require_bool(evidence["umask_0077_live_probe"], context="umask probe")
        is not True
        or _require_bool(evidence["core_dump_disabled"], context="core dump disabled")
        is not True
        or evidence["public_launch_request_channel"]
        != "STDIN_PUBLIC_JSON_NO_SECRET"
        or evidence["inherited_fd_numbers"] != [0, 1, 2]
        or evidence["unexpected_inherited_fd_count"] != 0
        or type(evidence["unexpected_inherited_fd_count"]) is not int
        or evidence["public_evidence_output_fd"] != 5
        or type(evidence["public_evidence_output_fd"]) is not int
        or evidence["landlock_status"]
        != "UNAVAILABLE_NONBLOCKING_GAP_DISCLOSED"
        or evidence["transient_capability_probe_incident_count"] != 0
        or type(evidence["transient_capability_probe_incident_count"]) is not int
        or _require_bool(
            evidence["network_fetch_allowed"], context="network fetch allowed"
        )
        is not False
    ):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "required security evidence differs")
    links = _require_exact_fields(
        evidence["namespace_links"],
        NAMESPACE_KINDS,
        context="namespace links",
        code=FAIL_SHADOW_BWRAP_ISOLATION,
    )
    separated = _require_exact_fields(
        evidence["namespace_unshared_from_orchestrator"],
        NAMESPACE_KINDS,
        context="namespace separation",
        code=FAIL_SHADOW_BWRAP_ISOLATION,
    )
    for kind in NAMESPACE_KINDS:
        link = _require_text(links[kind], context=f"{kind} namespace link")
        if not link.startswith(f"{kind}:[") or not link.endswith("]"):
            _fail(FAIL_SHADOW_BWRAP_ISOLATION, "namespace link syntax differs")
        if _require_bool(separated[kind], context=f"{kind} separation") is not True:
            _fail(FAIL_SHADOW_BWRAP_ISOLATION, "namespace was not separated")
    return evidence


def _validate_process_record(record: object, purpose_id: int) -> dict[str, object]:
    value = _require_exact_fields(
        record,
        (
            "role",
            "purpose_id",
            "process_id",
            "host_process_id",
            "environment_keys",
            "working_directory_mode_octal",
            "security_evidence",
        ),
        context="role process",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    if (
        value["role"] != PURPOSE_ROLES[purpose_id]
        or value["purpose_id"] != purpose_id
        or type(value["purpose_id"]) is not int
        or value["environment_keys"] != sorted(SANITIZED_ENVIRONMENT)
        or value["working_directory_mode_octal"] != "0700"
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "role process metadata differs")
    _require_int(value["process_id"], context="role PID", minimum=1)
    _require_int(value["host_process_id"], context="host role PID", minimum=1)
    _validate_security_evidence(value["security_evidence"])
    return value


def validate_shadow_runtime_report(report: object) -> None:
    """Strictly replay signatures and the no-authority boundary."""

    value = _require_exact_fields(
        report,
        (
            "schema_version",
            "evidence_class",
            "authority_class",
            "ceremony_id_hex",
            "input_snapshot",
            "calculator_agreement",
            "state_evidence",
            "envelopes",
            "fresh_admission_probes",
            "process_isolation",
            "authority_boundary",
            "shadow_status",
        ),
        context="shadow runtime report",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["evidence_class"] != EVIDENCE_CLASS
        or value["authority_class"] != AUTHORITY_CLASS
        or value["shadow_status"] != "INTERNAL_SHADOW_CEREMONY_COMPLETE"
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "shadow report fixed metadata differs")
    ceremony_id = _decode_hex(value["ceremony_id_hex"], 16, context="ceremony ID")
    validate_shadow_admission_report(value["fresh_admission_probes"])
    admission = value["fresh_admission_probes"]
    assert isinstance(admission, dict)
    if (
        admission["ceremony_id_hex"] != ceremony_id.hex()
        or admission["basis_commit_id_or_null"] is not None
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "fresh admission identity differs")
    validate_readonly_input_snapshot_report(value["input_snapshot"])
    snapshot = value["input_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot_digest = _decode_hex(
        snapshot["manifest_sha256_hex"], 32, context="snapshot digest"
    )
    if admission["snapshot_manifest_sha256_hex"] != snapshot_digest.hex():
        _fail(FAIL_SHADOW_REPORT_INVALID, "fresh admission snapshot differs")
    calculator = _validate_calculator_agreement(value["calculator_agreement"])
    calculator_cbor = bytes.fromhex(str(calculator["receipt_cbor_hex"]))
    decoded_calculator = canonical_cbor_decode(calculator_cbor)
    if decoded_calculator[2] != ceremony_id or decoded_calculator[3] != snapshot_digest:
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator receipt identity differs")
    calculator_digest = _decode_hex(
        calculator["receipt_sha256_hex"], 32, context="calculator receipt digest"
    )
    commitment = _decode_hex(
        calculator["seed_commitment_sha256_hex"], 32, context="seed commitment"
    )
    state_evidence = _validate_state_evidence(value["state_evidence"])
    if (
        state_evidence["ceremony_id_hex"] != ceremony_id.hex()
        or state_evidence["snapshot_manifest_sha256_hex"] != snapshot_digest.hex()
        or state_evidence["seed_commitment_sha256_hex_or_null"] != commitment.hex()
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "state evidence identity differs")
    marker_digest = _decode_hex(
        state_evidence["marker_sha256_hex"], 32, context="marker digest"
    )
    ledger_digest = _decode_hex(
        state_evidence["ledger_sha256_hex"], 32, context="ledger digest"
    )

    envelopes = value["envelopes"]
    if type(envelopes) is not list or len(envelopes) != 4:
        _fail(FAIL_SHADOW_REPORT_INVALID, "exactly four envelopes are required")
    verified = [verify_shadow_envelope(envelope) for envelope in envelopes]
    if [item["purpose_id"] for item in verified] != list(PURPOSE_IDS):
        _fail(FAIL_SHADOW_REPORT_INVALID, "envelope purpose ordering differs")
    expected_bindings = {
        "ceremony_id_hex": ceremony_id.hex(),
        "snapshot_manifest_sha256_hex": snapshot_digest.hex(),
        "calculator_receipt_sha256_hex": calculator_digest.hex(),
        "seed_commitment_sha256_hex": commitment.hex(),
        "marker_sha256_hex": marker_digest.hex(),
        "ledger_sha256_hex": ledger_digest.hex(),
    }
    for envelope in verified:
        if any(envelope[field] != expected for field, expected in expected_bindings.items()):
            _fail(FAIL_SHADOW_SIGNATURE_INVALID, "envelope binds different evidence")
    key_ids = [item["key_id_hex"] for item in verified]
    worker_ids = [item["worker_instance_id_hex"] for item in verified]
    role_pids = [int(item["process_id"]) for item in verified]
    if len(set(key_ids)) != 4 or len(set(worker_ids)) != 4:
        _fail(
            FAIL_SHADOW_PURPOSE_OR_KEY_COLLISION,
            "purpose keys, instances, and processes must all be distinct",
        )

    isolation = _require_exact_fields(
        value["process_isolation"],
        (
            "orchestrator_process_id",
            "role_processes",
            "calculator_process_ids",
            "every_role_namespace_unshared_from_orchestrator",
            "calculator_process_ids_distinct",
            "purpose_ids",
            "purpose_key_ids_distinct",
            "role_directory_mode_octal",
            "snapshot_directory_mode_octal",
            "sanitized_environment_keys",
            "seed_transport",
            "required_security",
        ),
        context="process isolation",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    orchestrator_pid = _require_int(
        isolation["orchestrator_process_id"], context="orchestrator PID", minimum=1
    )
    role_processes = isolation["role_processes"]
    if type(role_processes) is not list or len(role_processes) != 4:
        _fail(FAIL_SHADOW_REPORT_INVALID, "role process list differs")
    process_records = [
        _validate_process_record(record, purpose_id)
        for record, purpose_id in zip(role_processes, PURPOSE_IDS, strict=True)
    ]
    if len({int(item["host_process_id"]) for item in process_records}) != 4:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "host role process IDs collide")
    for kind in NAMESPACE_KINDS:
        if (
            len(
                {
                    str(item["security_evidence"]["namespace_links"][kind])
                    for item in process_records
                }
            )
            != 4
        ):
            _fail(
                FAIL_SHADOW_BWRAP_ISOLATION,
                f"role {kind} namespace identities collide",
            )
    recorded_role_pids = [int(item["process_id"]) for item in process_records]
    security_digests = [
        _security_evidence_digest(item["security_evidence"]).hex()
        for item in process_records
    ]
    if security_digests != [
        str(item["security_evidence_sha256_hex"]) for item in verified
    ]:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "signed security evidence differs")
    calculator_pids_raw = isolation["calculator_process_ids"]
    if type(calculator_pids_raw) is not list or len(calculator_pids_raw) != 2:
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator PID list differs")
    calculator_pids = [
        _require_int(item, context="calculator PID", minimum=1)
        for item in calculator_pids_raw
    ]
    calculator_worker_pids = [
        int(item["process_id"]) for item in calculator["workers"]
    ]
    if (
        recorded_role_pids != role_pids
        or calculator_pids != calculator_worker_pids
        or _require_bool(
            isolation["every_role_namespace_unshared_from_orchestrator"],
            context="role namespace separation flag",
        )
        is not True
        or _require_bool(
            isolation["calculator_process_ids_distinct"],
            context="calculator PID distinct flag",
        )
        is not True
        or isolation["purpose_ids"] != list(PURPOSE_IDS)
        or _require_bool(
            isolation["purpose_key_ids_distinct"], context="key distinct flag"
        )
        is not True
        or isolation["role_directory_mode_octal"] != "0700"
        or isolation["snapshot_directory_mode_octal"] != "0500"
        or isolation["sanitized_environment_keys"] != sorted(SANITIZED_ENVIRONMENT)
        or isolation["seed_transport"] != "ANONYMOUS_PIPE_FD_ONLY"
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "process isolation evidence differs")
    if len(set(calculator_pids)) != 2:
        _fail(FAIL_SHADOW_REPORT_INVALID, "calculator process IDs collide")
    if orchestrator_pid < 1:
        _fail(FAIL_SHADOW_REPORT_INVALID, "orchestrator PID is invalid")
    required_security = _require_exact_fields(
        isolation["required_security"],
        (
            "isolation_level",
            "bwrap_required",
            "bwrap_version",
            "bwrap_executable_sha256_hex",
            "unshared_namespaces",
            "seccomp_required",
            "seccomp_forbidden_syscalls",
            "seccomp_forbidden_syscall_count",
            "repository_read_only_required",
            "tmpfs_tmp_required",
            "anonymous_secret_fd",
            "network_fetch_allowed",
        ),
        context="required security",
        code=FAIL_SHADOW_BWRAP_ISOLATION,
    )
    if (
        required_security["isolation_level"] != ISOLATION_LEVEL
        or _require_bool(required_security["bwrap_required"], context="bwrap required")
        is not True
        or not _require_text(
            required_security["bwrap_version"], context="bwrap version"
        ).startswith("bubblewrap ")
        or len(
            _decode_hex(
                required_security["bwrap_executable_sha256_hex"],
                32,
                context="bwrap executable digest",
            )
        )
        != 32
        or required_security["unshared_namespaces"] != list(NAMESPACE_KINDS)
        or _require_bool(required_security["seccomp_required"], context="seccomp required")
        is not True
        or required_security["seccomp_forbidden_syscalls"]
        != list(SECCOMP_DENIED_SYSCALLS)
        or required_security["seccomp_forbidden_syscall_count"]
        != len(SECCOMP_DENIED_SYSCALLS)
        or type(required_security["seccomp_forbidden_syscall_count"]) is not int
        or _require_bool(
            required_security["repository_read_only_required"],
            context="repository read-only required",
        )
        is not True
        or _require_bool(required_security["tmpfs_tmp_required"], context="tmpfs required")
        is not True
        or required_security["anonymous_secret_fd"] != 3
        or type(required_security["anonymous_secret_fd"]) is not int
        or _require_bool(
            required_security["network_fetch_allowed"],
            context="runtime network fetch allowed",
        )
        is not False
    ):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "required security contract differs")

    boundary = _require_exact_fields(
        value["authority_boundary"],
        (
            "formal_gate_delta",
            "formal_gates_before",
            "formal_gates_after",
            "formal_gates_total",
            "formal_roots_issued",
            "external_actor_evidence",
            "formal_state_before",
            "formal_state_after",
            "formal_transition",
            "formal_m3_start_allowed",
            "technical_actor_eligible",
            "formal_technical_actor_evidence_issued",
            "internal_state_before",
            "internal_state_after",
            "internal_shadow_start_allowed",
            "report_alone_authorizes_formal_execution",
        ),
        context="authority boundary",
        code=FAIL_SHADOW_AUTHORITY_ESCALATION,
    )
    expected_boundary = {
        "formal_gates_before": 14,
        "formal_gates_after": 14,
        "formal_gates_total": 24,
        "formal_gate_delta": 0,
        "formal_roots_issued": False,
        "external_actor_evidence": False,
        "formal_state_before": FORMAL_STATE,
        "formal_state_after": FORMAL_STATE,
        "formal_transition": None,
        "formal_m3_start_allowed": False,
        "technical_actor_eligible": False,
        "formal_technical_actor_evidence_issued": False,
        "internal_state_before": INTERNAL_STATE_BEFORE,
        "internal_state_after": INTERNAL_STATE_AFTER,
        "internal_shadow_start_allowed": True,
        "report_alone_authorizes_formal_execution": False,
    }
    if not _json_type_strict_equal(boundary, expected_boundary):
        _fail(FAIL_SHADOW_AUTHORITY_ESCALATION, "formal authority boundary changed")


def _bwrap_worker_command(
    *,
    bwrap_path: Path,
    basis_snapshot_dir: Path,
    worker_source_dir: Path,
    private_state_dir: Path | None,
    calculator_endpoints: Mapping[str, Path] | None = None,
) -> list[str]:
    if _cryptography is None:
        _fail(FAIL_SHADOW_CRYPTO_BACKEND_UNAVAILABLE, "cryptography is unavailable")
    cryptography_source = Path(_cryptography.__file__).resolve().parent
    for source, context in (
        (basis_snapshot_dir, "basis snapshot"),
        (worker_source_dir, "worker source snapshot"),
        (cryptography_source, "cryptography dependency"),
    ):
        if not source.is_dir():
            _fail(FAIL_SHADOW_BWRAP_ISOLATION, f"{context} is absent")
    if private_state_dir is not None and not private_state_dir.is_dir():
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "private state directory is absent")
    endpoint_mounts: list[str] = []
    if calculator_endpoints is not None:
        if set(calculator_endpoints) != {"python", "rust"}:
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "calculator endpoint registry differs",
            )
        endpoint_mounts.extend(["--dir", "/calculator-endpoints"])
        for endpoint_id in ("python", "rust"):
            source = calculator_endpoints[endpoint_id]
            if not source.is_file() or source.is_symlink():
                _fail(
                    FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                    f"{endpoint_id} calculator endpoint is not a regular file",
                )
            endpoint_mounts.extend(
                [
                    "--ro-bind",
                    str(source),
                    f"/calculator-endpoints/{endpoint_id}",
                ]
            )
    private_state_mount = (
        []
        if private_state_dir is None
        else ["--bind", str(private_state_dir), str(VIRTUAL_STATE_ROOT)]
    )
    command = [
        str(bwrap_path),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--uid",
        "0",
        "--gid",
        "0",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--hostname",
        "hegel-shadow",
        "--cap-add",
        "CAP_NET_ADMIN",
        "--cap-add",
        "CAP_SETPCAP",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--ro-bind",
        str(basis_snapshot_dir),
        "/basis",
        "--ro-bind",
        str(worker_source_dir),
        "/worker-src",
        "--dir",
        "/worker-deps",
        "--ro-bind",
        str(cryptography_source),
        "/worker-deps/cryptography",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/work",
        "--tmpfs",
        "/run",
        *private_state_mount,
        *endpoint_mounts,
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--clearenv",
        "--setenv",
        "LANG",
        SANITIZED_ENVIRONMENT["LANG"],
        "--setenv",
        "LC_ALL",
        SANITIZED_ENVIRONMENT["LC_ALL"],
        "--setenv",
        "PATH",
        SANITIZED_ENVIRONMENT["PATH"],
        "--setenv",
        "TZ",
        SANITIZED_ENVIRONMENT["TZ"],
        "--setenv",
        "PYTHONPATH",
        "/worker-src:/worker-deps",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--chdir",
        "/work",
        "--",
        str(Path(sys.executable).resolve()),
        "-m",
        "hegel_machine.phase3_m3_shadow_runtime_v1",
        "--internal-shadow-bootstrap",
    ]
    return command


def _start_bwrap_json_worker(
    *,
    bwrap_path: Path,
    basis_snapshot_dir: Path,
    worker_source_dir: Path,
    private_state_dir: Path | None,
    calculator_endpoints: Mapping[str, Path] | None = None,
) -> subprocess.Popen[bytes]:
    command = _bwrap_worker_command(
        bwrap_path=bwrap_path,
        basis_snapshot_dir=basis_snapshot_dir,
        worker_source_dir=worker_source_dir,
        private_state_dir=private_state_dir,
        calculator_endpoints=calculator_endpoints,
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            env={},
        )
    except OSError:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "could not launch bubblewrap")
    return process


def _finish_bwrap_json_worker(
    process: subprocess.Popen[bytes],
    request: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    host_pid = process.pid
    try:
        stdout, stderr = process.communicate(_json_bytes(dict(request)), timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "bubblewrap worker timed out")
    if process.returncode != 0:
        try:
            error_type = stderr.decode("ascii", errors="ignore")[:3000].strip()
        except BaseException:
            error_type = ""
        _fail(
            FAIL_SHADOW_BWRAP_ISOLATION,
            "bubblewrap worker failed" + (f" ({error_type})" if error_type else ""),
        )
    if stderr:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "successful worker wrote to stderr")
    response = _decode_public_output_frame(stdout)
    if type(response) is not dict or type(response.get("ok")) is not bool:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "bubblewrap response envelope differs")
    if response["ok"] is False:
        if set(response) != {"ok", "code", "detail"}:
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "bubblewrap error field set differs")
        _fail(
            _require_text(response["code"], context="bubblewrap error code"),
            _require_text(response["detail"], context="bubblewrap error detail"),
        )
    if set(response) != {"ok", "result"} or type(response["result"]) is not dict:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "bubblewrap success field set differs")
    return response["result"], host_pid


def _terminate_bwrap_workers(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def _host_namespace_links(pid: int) -> dict[str, str]:
    links: dict[str, str] = {}
    for kind in NAMESPACE_KINDS:
        try:
            links[kind] = os.readlink(f"/proc/{pid}/ns/{kind}")
        except OSError:
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "could not inspect a live worker namespace",
            )
    return links


def _single_descendant_leaf_pid(supervisor_pid: int) -> int:
    """Resolve bwrap supervisor -> namespace-init -> actual role worker."""

    current = supervisor_pid
    for _depth in range(8):
        children_path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children = [int(item) for item in children_path.read_text().split()]
        except (OSError, ValueError):
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "could not inspect the bwrap descendant chain",
            )
        if not children:
            if current == supervisor_pid:
                _fail(
                    FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                    "bwrap role worker descendant is absent",
                )
            return current
        if len(children) != 1:
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "bwrap descendant chain is not single-purpose",
            )
        current = children[0]
    _fail(
        FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
        "bwrap descendant chain exceeds the frozen depth bound",
    )


def _wait_for_distinct_live_worker_namespaces(
    processes: Sequence[subprocess.Popen[bytes]],
    orchestrator_namespaces: Mapping[str, str],
) -> dict[int, tuple[int, dict[str, str]]]:
    """Require four simultaneously-live, pairwise-distinct namespace sets."""

    if len(processes) != 4 or len({process.pid for process in processes}) != 4:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "exactly four distinct live worker processes are required",
        )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            _fail(
                FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                "worker exited before simultaneous namespace qualification",
            )
        try:
            observed = {
                process.pid: (
                    worker_pid := _single_descendant_leaf_pid(process.pid),
                    _host_namespace_links(worker_pid),
                )
                for process in processes
            }
        except ShadowRuntimeError:
            time.sleep(0.01)
            continue
        if all(
            all(
                links[kind] != orchestrator_namespaces[kind]
                for kind in NAMESPACE_KINDS
            )
            for _worker_pid, links in observed.values()
        ) and all(
            len({links[kind] for _worker_pid, links in observed.values()}) == 4
            for kind in NAMESPACE_KINDS
        ) and len({worker_pid for worker_pid, _links in observed.values()}) == 4:
            return observed
        time.sleep(0.01)
    _fail(
        FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
        "four pairwise-distinct live namespace sets did not materialize",
    )


def _bind_host_process_evidence(
    result: Mapping[str, object],
    *,
    host_pid: int,
    host_namespace_links: Mapping[str, str],
) -> None:
    process = result.get("process")
    if type(process) is not dict:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "worker process evidence is absent")
    security = process.get("security_evidence")
    if type(security) is not dict or security.get("namespace_links") != dict(
        host_namespace_links
    ):
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "worker namespace self-report differs from host observation",
        )
    process["host_process_id"] = host_pid


def _bwrap_json_worker(
    *,
    bwrap_path: Path,
    basis_snapshot_dir: Path,
    worker_source_dir: Path,
    private_state_dir: Path | None,
    request: Mapping[str, object],
    calculator_endpoints: Mapping[str, Path] | None = None,
) -> tuple[dict[str, object], int]:
    """Synchronous compatibility wrapper for one isolated worker."""

    process = _start_bwrap_json_worker(
        bwrap_path=bwrap_path,
        basis_snapshot_dir=basis_snapshot_dir,
        worker_source_dir=worker_source_dir,
        private_state_dir=private_state_dir,
        calculator_endpoints=calculator_endpoints,
    )
    return _finish_bwrap_json_worker(process, request)


def _request_path(value: object, *, context: str) -> Path:
    text = _require_text(value, context=context)
    path = Path(text)
    if not path.is_absolute():
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, f"{context} must be absolute")
    resolved = path.resolve(strict=False)
    if not (
        resolved == Path("/work")
        or _is_inside(resolved, Path("/work"))
        or resolved == VIRTUAL_STATE_ROOT
        or _is_inside(resolved, VIRTUAL_STATE_ROOT)
    ):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, f"{context} escapes private mounts")
    return path


def _internal_worker_request(request: object) -> dict[str, object]:
    base = _require_exact_fields(
        request,
        (
            "worker_kind",
            "purpose_id",
            "role_dir",
            "ceremony_id_hex",
            "orchestrator_namespaces",
            "forbidden_host_paths",
            "payload",
        ),
        context="internal worker request",
        code=FAIL_SHADOW_IPC_PROTOCOL,
    )
    kind = _require_text(base["worker_kind"], context="worker kind")
    if kind not in {"probe", "custodian", "attester"}:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "unknown worker kind")
    purpose_id = _require_int(base["purpose_id"], context="worker purpose", minimum=1)
    if purpose_id not in PURPOSE_IDS:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "unknown worker purpose")
    if (kind == "custodian" and purpose_id != 1) or (
        kind == "attester" and purpose_id == 1
    ):
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "worker kind/purpose mismatch")
    _request_path(base["role_dir"], context="worker role directory")
    _decode_hex(base["ceremony_id_hex"], 16, context="worker ceremony ID")
    namespaces = _require_exact_fields(
        base["orchestrator_namespaces"],
        NAMESPACE_KINDS,
        context="orchestrator namespaces",
        code=FAIL_SHADOW_IPC_PROTOCOL,
    )
    for item in namespaces.values():
        _require_text(item, context="orchestrator namespace")
    forbidden = _require_exact_fields(
        base["forbidden_host_paths"],
        ("live_repository", "host_home", "windows_mnt_c", "docker_socket"),
        context="forbidden host paths",
        code=FAIL_SHADOW_IPC_PROTOCOL,
    )
    for item in forbidden.values():
        _require_text(item, context="forbidden host path")
    if type(base["payload"]) is not dict:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "worker payload must be an object")
    return base


def _internal_shadow_worker() -> dict[str, object]:
    raw = sys.stdin.buffer.read(MAX_IPC_BYTES + 1)
    if len(raw) > MAX_IPC_BYTES:
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "worker request exceeds IPC bound")
    try:
        request = _internal_worker_request(json.loads(raw.decode("ascii")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(FAIL_SHADOW_IPC_PROTOCOL, "worker request is not strict JSON")
    kind = str(request["worker_kind"])
    purpose_id = int(request["purpose_id"])
    role_dir = _request_path(request["role_dir"], context="worker role directory")
    ceremony_id = bytes.fromhex(str(request["ceremony_id_hex"]))
    for private_dir in (Path("/work"), Path("/home"), Path("/tmp")):
        os.chmod(private_dir, 0o700)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    environment_keys, cwd_mode = _sanitize_child(role_dir)
    security = _security_evidence(
        orchestrator_namespaces=request["orchestrator_namespaces"],
        ceremony_id=ceremony_id,
        forbidden_host_paths=request["forbidden_host_paths"],
    )
    payload = request["payload"]
    assert isinstance(payload, dict)
    if kind == "probe":
        expected_payload = (
            {"qualify_calculator_endpoints": True} if purpose_id == 1 else {}
        )
        if not _json_type_strict_equal(payload, expected_payload):
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "probe payload differs")
        calculator_qualification = (
            _qualify_bound_calculator_endpoints(role_dir)
            if purpose_id == 1
            else None
        )
        return {
            "purpose_id": purpose_id,
            "role": PURPOSE_ROLES[purpose_id],
            "calculator_qualification_or_null": calculator_qualification,
            "process": _role_process_record(
                PURPOSE_ROLES[purpose_id],
                purpose_id,
                environment_keys,
                cwd_mode,
                security,
            ),
        }
    if kind == "custodian":
        values = _require_exact_fields(
            payload,
            ("state_dir", "calculator_dirs", "snapshot_digest_hex"),
            context="custodian payload",
            code=FAIL_SHADOW_IPC_PROTOCOL,
        )
        state_dir = _request_path(values["state_dir"], context="custodian state")
        calculator_dirs_raw = values["calculator_dirs"]
        if type(calculator_dirs_raw) is not list or len(calculator_dirs_raw) != 2:
            _fail(FAIL_SHADOW_IPC_PROTOCOL, "calculator directory list differs")
        calculator_dirs = tuple(
            _request_path(item, context="calculator directory")
            for item in calculator_dirs_raw
        )
        for calculator_dir in calculator_dirs:
            calculator_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        snapshot_digest = _decode_hex(
            values["snapshot_digest_hex"], 32, context="snapshot digest"
        )
        return _custodian_worker(
            state_dir=state_dir,
            role_dir=role_dir,
            calculator_dirs=(calculator_dirs[0], calculator_dirs[1]),
            ceremony_id=ceremony_id,
            snapshot_digest=snapshot_digest,
            environment_keys=environment_keys,
            cwd_mode=cwd_mode,
            security_evidence=security,
        )

    values = _require_exact_fields(
        payload,
        (
            "snapshot_digest_hex",
            "calculator_receipt_digest_hex",
            "seed_commitment_hex",
            "marker_digest_hex",
            "ledger_digest_hex",
        ),
        context="attester payload",
        code=FAIL_SHADOW_IPC_PROTOCOL,
    )
    return _attester_worker(
        purpose_id=purpose_id,
        role_dir=role_dir,
        ceremony_id=ceremony_id,
        snapshot_digest=_decode_hex(
            values["snapshot_digest_hex"], 32, context="snapshot digest"
        ),
        calculator_receipt_digest=_decode_hex(
            values["calculator_receipt_digest_hex"],
            32,
            context="calculator receipt digest",
        ),
        seed_commitment=_decode_hex(
            values["seed_commitment_hex"], 32, context="seed commitment"
        ),
        marker_digest=_decode_hex(
            values["marker_digest_hex"], 32, context="marker digest"
        ),
        ledger_digest=_decode_hex(
            values["ledger_digest_hex"], 32, context="ledger digest"
        ),
        environment_keys=environment_keys,
        cwd_mode=cwd_mode,
        security_evidence=security,
    )


def _internal_worker_cli() -> int:
    try:
        result = {"ok": True, "result": _internal_shadow_worker()}
    except ShadowRuntimeError as error:
        result = {"ok": False, "code": error.code, "detail": error.detail}
    except BaseException as error:
        result = {
            "ok": False,
            "code": FAIL_SHADOW_ROLE_PROCESS,
            "detail": f"worker raised {type(error).__name__}",
        }
    os.dup2(sys.stdout.fileno(), 5, inheritable=True)
    os.close(sys.stdout.fileno())
    sys.stdout = None
    _write_all(5, _public_output_frame(result))
    os.close(5)
    return 0


def _prepare_role_directories(runtime_dir: Path) -> dict[int | str, Path]:
    paths: dict[int | str, Path] = {}
    for purpose_id, role in PURPOSE_ROLES.items():
        path = runtime_dir / role.lower()
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        paths[purpose_id] = path
    custodian = paths[1]
    assert isinstance(custodian, Path)
    for name in ("calculator_python", "calculator_incremental"):
        path = custodian / name
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        paths[name] = path
    return paths


def _cleanup_runtime_directory(runtime_dir: Path) -> None:
    # Only the exact directory created by this invocation is ever removed.
    if not runtime_dir.name.startswith("shadow-runtime-"):
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "refusing broad runtime cleanup")
    for root, directories, files in os.walk(runtime_dir, topdown=False):
        for name in files:
            try:
                os.chmod(Path(root) / name, 0o600)
            except OSError:
                pass
        for name in directories:
            try:
                os.chmod(Path(root) / name, 0o700)
            except OSError:
                pass
    try:
        os.chmod(runtime_dir, 0o700)
        shutil.rmtree(runtime_dir)
    except OSError:
        _fail(FAIL_SHADOW_STATE_PATH_INVALID, "isolated runtime cleanup failed")


def _force_remove_private_tree(path: Path) -> None:
    if not path.is_dir():
        return
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.chmod(Path(root) / name, 0o600)
            except OSError:
                pass
        for name in directories:
            try:
                os.chmod(Path(root) / name, 0o700)
            except OSError:
                pass
    os.chmod(path, 0o700)
    shutil.rmtree(path)


def _admission_receipt_digest(
    *,
    ceremony_id: bytes,
    snapshot_digest: bytes,
    purpose_id: int,
    process: Mapping[str, object],
) -> str:
    value = {
        "authority_class": AUTHORITY_CLASS,
        "ceremony_id_hex": ceremony_id.hex(),
        "snapshot_manifest_sha256_hex": snapshot_digest.hex(),
        "purpose_id": purpose_id,
        "role": PURPOSE_ROLES[purpose_id],
        "process": dict(process),
    }
    return hashlib.sha256(
        b"HEGEL/INTERNAL_SHADOW/ADMISSION_PROBE_RECEIPT/V1\x00"
        + _json_bytes(value)
    ).hexdigest()


def _validate_calculator_endpoint_qualification(value: object) -> dict[str, object]:
    qualification = _require_exact_fields(
        value,
        (
            "evidence_class",
            "status",
            "response_schema_id_hex",
            "known_commitment_sha256_hex",
            "endpoint_sha256_hex",
            "workers",
            "contains_raw_seed",
        ),
        context="calculator endpoint qualification",
        code=FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
    )
    if (
        qualification["evidence_class"] != EVIDENCE_CLASS
        or qualification["status"]
        != "DUAL_PYTHON_RUST_FD3_ENDPOINTS_BIT_EXACT_PASS"
        or qualification["response_schema_id_hex"]
        != CALCULATOR_ENDPOINT_RESPONSE_SCHEMA_ID.hex()
        or qualification["known_commitment_sha256_hex"]
        != CALCULATOR_QUALIFICATION_COMMITMENT.hex()
        or _require_bool(
            qualification["contains_raw_seed"], context="qualification raw seed"
        )
        is not False
    ):
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "calculator endpoint qualification metadata differs",
        )
    endpoint_digests = _require_exact_fields(
        qualification["endpoint_sha256_hex"],
        ("python", "rust"),
        context="calculator endpoint digests",
        code=FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
    )
    for digest in endpoint_digests.values():
        _decode_hex(digest, 32, context="calculator endpoint digest")
    if endpoint_digests["python"] == endpoint_digests["rust"]:
        _fail(
            FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
            "calculator endpoint digests collide",
        )
    workers = qualification["workers"]
    if type(workers) is not list or len(workers) != 2:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "qualification worker count differs")
    for expected_id, worker in zip(
        CALCULATOR_ENDPOINT_IDS.values(), workers, strict=True
    ):
        row = _require_exact_fields(
            worker,
            (
                "calculator_id",
                "process_id",
                "seed_commitment_sha256_hex",
                "environment_keys",
                "working_directory_mode_octal",
                "secret_input_fd",
                "public_output_fd",
                "unexpected_inherited_fd_count",
                "seccomp_mode",
            ),
            context="qualification calculator worker",
            code=FAIL_SHADOW_CALCULATOR_PROCESS,
        )
        if (
            row["calculator_id"] != expected_id
            or row["seed_commitment_sha256_hex"]
            != CALCULATOR_QUALIFICATION_COMMITMENT.hex()
            or row["environment_keys"] != sorted(SANITIZED_ENVIRONMENT)
            or row["working_directory_mode_octal"] != "0700"
            or row["secret_input_fd"] != 3
            or row["public_output_fd"] != 5
            or row["unexpected_inherited_fd_count"] != 0
            or row["seccomp_mode"] != 2
        ):
            _fail(
                FAIL_SHADOW_CALCULATOR_PROCESS,
                "qualification calculator evidence differs",
            )
        _require_int(row["process_id"], context="qualification calculator PID", minimum=1)
    if len({int(worker["process_id"]) for worker in workers}) != 2:
        _fail(FAIL_SHADOW_CALCULATOR_PROCESS, "qualification calculator PIDs collide")
    return qualification


def validate_shadow_admission_report(report: object) -> None:
    """Validate four no-key/no-seed/no-marker strong-isolation probes."""

    value = _require_exact_fields(
        report,
        (
            "schema_version",
            "evidence_class",
            "authority_class",
            "basis_commit_id_or_null",
            "ceremony_id_hex",
            "snapshot_manifest_sha256_hex",
            "calculator_endpoint_qualification",
            "purpose_probe_receipts",
            "isolation_plan_inputs",
            "side_effects",
            "authority_boundary",
            "admission_status",
        ),
        context="shadow admission report",
        code=FAIL_SHADOW_REPORT_INVALID,
    )
    if (
        value["schema_version"] != ADMISSION_SCHEMA_VERSION
        or value["evidence_class"] != EVIDENCE_CLASS
        or value["authority_class"] != AUTHORITY_CLASS
        or value["admission_status"] != "INTERNAL_SHADOW_ADMISSION_PASS"
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "admission fixed metadata differs")
    basis_commit = value["basis_commit_id_or_null"]
    if basis_commit is not None and (
        type(basis_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", basis_commit) is None
    ):
        _fail(FAIL_SHADOW_REPORT_INVALID, "basis commit ID must be SHA-1 hex or null")
    ceremony_id = _decode_hex(value["ceremony_id_hex"], 16, context="admission ceremony ID")
    snapshot_digest = _decode_hex(
        value["snapshot_manifest_sha256_hex"],
        32,
        context="admission snapshot digest",
    )
    calculator_qualification = _validate_calculator_endpoint_qualification(
        value["calculator_endpoint_qualification"]
    )
    receipts = value["purpose_probe_receipts"]
    if type(receipts) is not list or len(receipts) != 4:
        _fail(FAIL_SHADOW_REPORT_INVALID, "admission requires four probe receipts")
    for raw_receipt, purpose_id in zip(receipts, PURPOSE_IDS, strict=True):
        receipt = _require_exact_fields(
            raw_receipt,
            ("purpose_id", "role", "process", "receipt_sha256_hex"),
            context="admission probe receipt",
            code=FAIL_SHADOW_REPORT_INVALID,
        )
        if (
            receipt["purpose_id"] != purpose_id
            or type(receipt["purpose_id"]) is not int
            or receipt["role"] != PURPOSE_ROLES[purpose_id]
        ):
            _fail(FAIL_SHADOW_REPORT_INVALID, "admission purpose ordering differs")
        process = _validate_process_record(receipt["process"], purpose_id)
        expected_digest = _admission_receipt_digest(
            ceremony_id=ceremony_id,
            snapshot_digest=snapshot_digest,
            purpose_id=purpose_id,
            process=process,
        )
        if receipt["receipt_sha256_hex"] != expected_digest:
            _fail(FAIL_SHADOW_REPORT_INVALID, "admission receipt digest differs")
    processes = [receipt["process"] for receipt in receipts]
    if len({int(item["host_process_id"]) for item in processes}) != 4:
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "admission host PIDs collide")
    for kind in NAMESPACE_KINDS:
        if (
            len(
                {
                    str(item["security_evidence"]["namespace_links"][kind])
                    for item in processes
                }
            )
            != 4
        ):
            _fail(
                FAIL_SHADOW_BWRAP_ISOLATION,
                f"admission {kind} namespace identities collide",
            )

    plan = _require_exact_fields(
        value["isolation_plan_inputs"],
        (
            "isolation_level",
            "purpose_ids",
            "bwrap_version",
            "bwrap_executable_sha256_hex",
            "unshared_namespaces",
            "seccomp_forbidden_syscalls",
            "repository_mount_mode",
            "private_state_directory_mode",
            "private_state_file_mode",
            "input_snapshot_directory_mode",
            "input_snapshot_file_mode",
            "secret_transport_fd",
            "calculator_endpoint_sha256_hex",
            "fresh_start_reprobe_required",
            "network_fetch_allowed",
        ),
        context="admission isolation plan",
        code=FAIL_SHADOW_BWRAP_ISOLATION,
    )
    if (
        plan["isolation_level"] != ISOLATION_LEVEL
        or plan["purpose_ids"] != list(PURPOSE_IDS)
        or not _require_text(plan["bwrap_version"], context="bwrap version").startswith(
            "bubblewrap "
        )
        or len(
            _decode_hex(
                plan["bwrap_executable_sha256_hex"],
                32,
                context="bwrap executable digest",
            )
        )
        != 32
        or plan["unshared_namespaces"] != list(NAMESPACE_KINDS)
        or plan["seccomp_forbidden_syscalls"] != list(SECCOMP_DENIED_SYSCALLS)
        or plan["repository_mount_mode"] != "READ_ONLY"
        or plan["private_state_directory_mode"] != "0700"
        or plan["private_state_file_mode"] != "0600"
        or plan["input_snapshot_directory_mode"] != "0500"
        or plan["input_snapshot_file_mode"] != "0400"
        or plan["secret_transport_fd"] != 3
        or type(plan["secret_transport_fd"]) is not int
        or plan["calculator_endpoint_sha256_hex"]
        != calculator_qualification["endpoint_sha256_hex"]
        or _require_bool(
            plan["fresh_start_reprobe_required"], context="fresh reprobe required"
        )
        is not True
        or _require_bool(
            plan["network_fetch_allowed"], context="admission network fetch allowed"
        )
        is not False
    ):
        _fail(FAIL_SHADOW_BWRAP_ISOLATION, "admission isolation plan differs")
    effects = _require_exact_fields(
        value["side_effects"],
        ("key_generated", "seed_generated", "marker_written", "formal_root_issued"),
        context="admission side effects",
        code=FAIL_SHADOW_AUTHORITY_ESCALATION,
    )
    if any(_require_bool(item, context="admission side effect") for item in effects.values()):
        _fail(FAIL_SHADOW_AUTHORITY_ESCALATION, "admission performed a side effect")
    expected_boundary = {
        "formal_gates_before": 14,
        "formal_gates_after": 14,
        "formal_gates_total": 24,
        "formal_gate_delta": 0,
        "formal_state_before": FORMAL_STATE,
        "formal_state_after": FORMAL_STATE,
        "formal_transition": None,
        "formal_m3_start_allowed": False,
        "technical_actor_eligible": False,
        "formal_technical_actor_evidence_issued": False,
        "internal_ceremony_start_requires_explicit_action": True,
        "report_alone_authorizes_formal_execution": False,
    }
    boundary = _require_exact_fields(
        value["authority_boundary"],
        tuple(expected_boundary),
        context="admission authority boundary",
        code=FAIL_SHADOW_AUTHORITY_ESCALATION,
    )
    if not _json_type_strict_equal(boundary, expected_boundary):
        _fail(FAIL_SHADOW_AUTHORITY_ESCALATION, "admission authority changed")


def run_shadow_admission_probes(
    *,
    basis_snapshot: Mapping[str, object],
    basis_snapshot_directory: Path | str,
    ceremony_id: bytes,
    python_calculator_path: Path | str,
    rust_calculator_path: Path | str,
    basis_commit_id: str | None = None,
) -> dict[str, object]:
    """Run four bwrap+seccomp probes without creating keys, seed, or marker."""

    _require_posix_runtime()
    validate_readonly_input_snapshot_report(basis_snapshot)
    if type(ceremony_id) is not bytes or len(ceremony_id) != 16 or ceremony_id == bytes(16):
        _fail(REJECT_SHADOW_FIELD_TYPE, "ceremony_id must be nonzero 16-byte bytes")
    if basis_commit_id is not None and (
        type(basis_commit_id) is not str
        or re.fullmatch(r"[0-9a-f]{40}", basis_commit_id) is None
    ):
        _fail(REJECT_SHADOW_FIELD_TYPE, "basis_commit_id must be SHA-1 hex or null")
    bwrap_path, bwrap_version, bwrap_digest = _require_bwrap()
    calculator_endpoints, calculator_endpoint_digests = (
        _resolve_calculator_endpoint_bindings(
            python_calculator_path, rust_calculator_path
        )
    )
    orchestrator_namespaces = _namespace_links()
    snapshot_digest = bytes.fromhex(str(basis_snapshot["manifest_sha256_hex"]))
    basis_dir = Path(basis_snapshot_directory).resolve(strict=True)
    _validate_snapshot_directory(basis_dir, basis_snapshot)
    admission_dir = Path(
        tempfile.mkdtemp(prefix="hegel-shadow-admission-", dir="/tmp")
    )
    os.chmod(admission_dir, 0o700)
    try:
        worker_source_dir = admission_dir / "worker_source"
        worker_source_report = _create_worker_source_snapshot(worker_source_dir)
        processes = [
            _start_bwrap_json_worker(
                bwrap_path=bwrap_path,
                basis_snapshot_dir=basis_dir,
                worker_source_dir=worker_source_dir,
                private_state_dir=None,
                calculator_endpoints=(
                    calculator_endpoints if purpose_id == 1 else None
                ),
            )
            for purpose_id in PURPOSE_IDS
        ]
        live_namespaces = _wait_for_distinct_live_worker_namespaces(
            processes, orchestrator_namespaces
        )
        receipts: list[dict[str, object]] = []
        calculator_qualification: object = None
        try:
            for purpose_id, process in zip(PURPOSE_IDS, processes, strict=True):
                result, host_pid = _finish_bwrap_json_worker(
                    process,
                    {
                    "worker_kind": "probe",
                    "purpose_id": purpose_id,
                    "role_dir": "/work",
                    "ceremony_id_hex": ceremony_id.hex(),
                    "orchestrator_namespaces": orchestrator_namespaces,
                    "forbidden_host_paths": {
                        "live_repository": str(REPOSITORY_ROOT),
                        "host_home": str(Path.home()),
                        "windows_mnt_c": "/mnt/c",
                        "docker_socket": "/var/run/docker.sock",
                    },
                    "payload": (
                        {"qualify_calculator_endpoints": True}
                        if purpose_id == 1
                        else {}
                    ),
                    },
                )
                probe = _require_exact_fields(
                    result,
                    (
                        "purpose_id",
                        "role",
                        "calculator_qualification_or_null",
                        "process",
                    ),
                    context="admission worker result",
                    code=FAIL_SHADOW_IPC_PROTOCOL,
                )
                worker_host_pid, worker_host_namespaces = live_namespaces[host_pid]
                _bind_host_process_evidence(
                    probe,
                    host_pid=worker_host_pid,
                    host_namespace_links=worker_host_namespaces,
                )
                process_record = _validate_process_record(
                    probe["process"], purpose_id
                )
                receipts.append(
                    {
                        "purpose_id": purpose_id,
                        "role": PURPOSE_ROLES[purpose_id],
                        "process": process_record,
                        "receipt_sha256_hex": _admission_receipt_digest(
                            ceremony_id=ceremony_id,
                            snapshot_digest=snapshot_digest,
                            purpose_id=purpose_id,
                            process=process_record,
                        ),
                    }
                )
                if purpose_id == 1:
                    calculator_qualification = probe[
                        "calculator_qualification_or_null"
                    ]
                elif probe["calculator_qualification_or_null"] is not None:
                    _fail(
                        FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE,
                        "non-custodian probe accessed calculator endpoints",
                    )
        finally:
            _terminate_bwrap_workers(processes)
        if any(
            (admission_dir / name).exists()
            for name in (MARKER_FILE_NAME, LEDGER_FILE_NAME)
        ):
            _fail(FAIL_SHADOW_AUTHORITY_ESCALATION, "admission wrote private state")
        _validate_worker_source_snapshot(worker_source_dir, worker_source_report)
        report: dict[str, object] = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "evidence_class": EVIDENCE_CLASS,
            "authority_class": AUTHORITY_CLASS,
            "basis_commit_id_or_null": basis_commit_id,
            "ceremony_id_hex": ceremony_id.hex(),
            "snapshot_manifest_sha256_hex": snapshot_digest.hex(),
            "calculator_endpoint_qualification": calculator_qualification,
            "purpose_probe_receipts": receipts,
            "isolation_plan_inputs": {
                "isolation_level": ISOLATION_LEVEL,
                "purpose_ids": list(PURPOSE_IDS),
                "bwrap_version": bwrap_version,
                "bwrap_executable_sha256_hex": bwrap_digest,
                "unshared_namespaces": list(NAMESPACE_KINDS),
                "seccomp_forbidden_syscalls": list(SECCOMP_DENIED_SYSCALLS),
                "repository_mount_mode": "READ_ONLY",
                "private_state_directory_mode": "0700",
                "private_state_file_mode": "0600",
                "input_snapshot_directory_mode": "0500",
                "input_snapshot_file_mode": "0400",
                "secret_transport_fd": 3,
                "calculator_endpoint_sha256_hex": calculator_endpoint_digests,
                "fresh_start_reprobe_required": True,
                "network_fetch_allowed": False,
            },
            "side_effects": {
                "key_generated": False,
                "seed_generated": False,
                "marker_written": False,
                "formal_root_issued": False,
            },
            "authority_boundary": {
                "formal_gates_before": 14,
                "formal_gates_after": 14,
                "formal_gates_total": 24,
                "formal_gate_delta": 0,
                "formal_state_before": FORMAL_STATE,
                "formal_state_after": FORMAL_STATE,
                "formal_transition": None,
                "formal_m3_start_allowed": False,
                "technical_actor_eligible": False,
                "formal_technical_actor_evidence_issued": False,
                "internal_ceremony_start_requires_explicit_action": True,
                "report_alone_authorizes_formal_execution": False,
            },
            "admission_status": "INTERNAL_SHADOW_ADMISSION_PASS",
        }
        validate_shadow_admission_report(report)
        return report
    finally:
        if admission_dir.exists():
            _force_remove_private_tree(admission_dir)


def probe_shadow_admission(
    *,
    basis_commit_id: str,
    shadow_run_id: bytes,
    input_files: Mapping[str, Path | str],
    python_calculator_path: Path | str,
    rust_calculator_path: Path | str,
) -> dict[str, object]:
    """Convenience admission API that builds only an ephemeral /tmp snapshot."""

    temporary = Path(tempfile.mkdtemp(prefix="hegel-shadow-basis-", dir="/tmp"))
    os.chmod(temporary, 0o700)
    try:
        snapshot = create_readonly_input_snapshot(
            dict(input_files), temporary / "input_snapshot"
        )
        return run_shadow_admission_probes(
            basis_snapshot=snapshot,
            basis_snapshot_directory=temporary / "input_snapshot",
            ceremony_id=shadow_run_id,
            python_calculator_path=python_calculator_path,
            rust_calculator_path=rust_calculator_path,
            basis_commit_id=basis_commit_id,
        )
    finally:
        if temporary.exists():
            _force_remove_private_tree(temporary)


def run_internal_shadow_ceremony(
    *,
    state_directory: Path | str,
    input_files: Mapping[str, Path | str],
    python_calculator_path: Path | str,
    rust_calculator_path: Path | str,
    ceremony_id: bytes | None = None,
    snapshot_hook: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run a purpose-separated internal ceremony without formal authority.

    ``state_directory`` is persistent, external to the repository, and one
    shot.  A failure after PENDING never redraws a seed.  ``snapshot_hook`` is
    an optional pre-seed audit callback; the snapshot is fully revalidated
    after the callback and again after all role processes finish.
    """

    _require_posix_runtime()
    _require_crypto_backend()
    bwrap_path, bwrap_version, bwrap_digest = _require_bwrap()
    calculator_endpoints, _calculator_endpoint_digests = (
        _resolve_calculator_endpoint_bindings(
            python_calculator_path, rust_calculator_path
        )
    )
    orchestrator_namespaces = _namespace_links()
    state_dir = _validate_state_directory(Path(state_directory))
    _assert_private_state_absent(state_dir)
    if ceremony_id is None:
        ceremony_id = _random_id16()
    if type(ceremony_id) is not bytes or len(ceremony_id) != 16 or ceremony_id == bytes(16):
        _fail(REJECT_SHADOW_FIELD_TYPE, "ceremony_id must be nonzero 16-byte bytes")
    if snapshot_hook is not None and not callable(snapshot_hook):
        _fail(REJECT_SHADOW_FIELD_TYPE, "snapshot_hook must be callable")

    runtime_dir = Path(tempfile.mkdtemp(prefix="shadow-runtime-", dir="/tmp"))
    os.chmod(runtime_dir, 0o700)
    report: dict[str, object] | None = None
    try:
        snapshot_dir = runtime_dir / "input_snapshot"
        snapshot = create_readonly_input_snapshot(dict(input_files), snapshot_dir)
        if snapshot_hook is not None:
            snapshot_hook(snapshot_dir, snapshot)
        _validate_snapshot_directory(snapshot_dir, snapshot)
        snapshot_digest = bytes.fromhex(str(snapshot["manifest_sha256_hex"]))
        fresh_admission = run_shadow_admission_probes(
            basis_snapshot=snapshot,
            basis_snapshot_directory=snapshot_dir,
            ceremony_id=ceremony_id,
            python_calculator_path=python_calculator_path,
            rust_calculator_path=rust_calculator_path,
            basis_commit_id=None,
        )

        worker_source_dir = runtime_dir / "worker_source"
        worker_source_report = _create_worker_source_snapshot(worker_source_dir)
        common_request = {
            "ceremony_id_hex": ceremony_id.hex(),
            "orchestrator_namespaces": orchestrator_namespaces,
            "forbidden_host_paths": {
                "live_repository": str(REPOSITORY_ROOT),
                "host_home": str(Path.home()),
                "windows_mnt_c": "/mnt/c",
                "docker_socket": "/var/run/docker.sock",
            },
        }
        role_workers = [
            _start_bwrap_json_worker(
                bwrap_path=bwrap_path,
                basis_snapshot_dir=snapshot_dir,
                worker_source_dir=worker_source_dir,
                private_state_dir=state_dir if purpose_id == 1 else None,
                calculator_endpoints=(
                    calculator_endpoints if purpose_id == 1 else None
                ),
            )
            for purpose_id in PURPOSE_IDS
        ]
        live_namespaces = _wait_for_distinct_live_worker_namespaces(
            role_workers, orchestrator_namespaces
        )
        try:
            custodian_result, custodian_supervisor_pid = _finish_bwrap_json_worker(
                role_workers[0],
                {
                **common_request,
                "worker_kind": "custodian",
                "purpose_id": 1,
                "role_dir": "/work",
                "payload": {
                    "state_dir": "/private-state",
                    "calculator_dirs": [
                        "/work/calculator_python",
                        "/work/calculator_rust",
                    ],
                    "snapshot_digest_hex": snapshot_digest.hex(),
                },
                },
            )
            custodian = _require_exact_fields(
                custodian_result,
                ("calculator_agreement", "state_evidence", "envelope", "process"),
                context="custodian result",
                code=FAIL_SHADOW_IPC_PROTOCOL,
            )
            custodian_host_pid, custodian_namespaces = live_namespaces[
                custodian_supervisor_pid
            ]
            _bind_host_process_evidence(
                custodian,
                host_pid=custodian_host_pid,
                host_namespace_links=custodian_namespaces,
            )
            calculator = _validate_calculator_agreement(
                custodian["calculator_agreement"]
            )
            state_evidence = _validate_state_evidence(custodian["state_evidence"])
            commitment = bytes.fromhex(str(calculator["seed_commitment_sha256_hex"]))
            calculator_digest = bytes.fromhex(str(calculator["receipt_sha256_hex"]))
            marker_digest = bytes.fromhex(str(state_evidence["marker_sha256_hex"]))
            ledger_digest = bytes.fromhex(str(state_evidence["ledger_sha256_hex"]))

            envelopes = [custodian["envelope"]]
            process_records = [custodian["process"]]
            for purpose_id, role_worker in zip(
                (2, 3, 4), role_workers[1:], strict=True
            ):
                attester_result, attester_supervisor_pid = (
                    _finish_bwrap_json_worker(
                        role_worker,
                        {
                    **common_request,
                    "worker_kind": "attester",
                    "purpose_id": purpose_id,
                    "role_dir": "/work",
                    "payload": {
                        "snapshot_digest_hex": snapshot_digest.hex(),
                        "calculator_receipt_digest_hex": calculator_digest.hex(),
                        "seed_commitment_hex": commitment.hex(),
                        "marker_digest_hex": marker_digest.hex(),
                        "ledger_digest_hex": ledger_digest.hex(),
                    },
                        },
                    )
                )
                attester = _require_exact_fields(
                    attester_result,
                    ("envelope", "process"),
                    context="attester result",
                    code=FAIL_SHADOW_IPC_PROTOCOL,
                )
                attester_host_pid, attester_namespaces = live_namespaces[
                    attester_supervisor_pid
                ]
                _bind_host_process_evidence(
                    attester,
                    host_pid=attester_host_pid,
                    host_namespace_links=attester_namespaces,
                )
                envelopes.append(attester["envelope"])
                process_records.append(attester["process"])
        finally:
            _terminate_bwrap_workers(role_workers)

        _validate_snapshot_directory(snapshot_dir, snapshot)
        _validate_worker_source_snapshot(worker_source_dir, worker_source_report)
        # Re-read the persistent state after every role has exited.
        if inspect_shadow_state(state_dir) != state_evidence:
            _fail(FAIL_SHADOW_LEDGER_TAMPERED, "private state changed after completion")
        calculator_pids = [int(item["process_id"]) for item in calculator["workers"]]
        report = {
            "schema_version": SCHEMA_VERSION,
            "evidence_class": EVIDENCE_CLASS,
            "authority_class": AUTHORITY_CLASS,
            "ceremony_id_hex": ceremony_id.hex(),
            "input_snapshot": snapshot,
            "calculator_agreement": calculator,
            "state_evidence": state_evidence,
            "envelopes": envelopes,
            "fresh_admission_probes": fresh_admission,
            "process_isolation": {
                "orchestrator_process_id": os.getpid(),
                "role_processes": process_records,
                "calculator_process_ids": calculator_pids,
                "every_role_namespace_unshared_from_orchestrator": True,
                "calculator_process_ids_distinct": len(set(calculator_pids)) == 2,
                "purpose_ids": list(PURPOSE_IDS),
                "purpose_key_ids_distinct": (
                    len({str(item["key_id_hex"]) for item in envelopes}) == 4
                ),
                "role_directory_mode_octal": "0700",
                "snapshot_directory_mode_octal": "0500",
                "sanitized_environment_keys": sorted(SANITIZED_ENVIRONMENT),
                "seed_transport": "ANONYMOUS_PIPE_FD_ONLY",
                "required_security": {
                    "isolation_level": ISOLATION_LEVEL,
                    "bwrap_required": True,
                    "bwrap_version": bwrap_version,
                    "bwrap_executable_sha256_hex": bwrap_digest,
                    "unshared_namespaces": list(NAMESPACE_KINDS),
                    "seccomp_required": True,
                    "seccomp_forbidden_syscalls": list(SECCOMP_DENIED_SYSCALLS),
                    "seccomp_forbidden_syscall_count": len(SECCOMP_DENIED_SYSCALLS),
                    "repository_read_only_required": True,
                    "tmpfs_tmp_required": True,
                    "anonymous_secret_fd": 3,
                    "network_fetch_allowed": False,
                },
            },
            "authority_boundary": {
                "formal_gates_before": 14,
                "formal_gates_after": 14,
                "formal_gates_total": 24,
                "formal_gate_delta": 0,
                "formal_roots_issued": False,
                "external_actor_evidence": False,
                "formal_state_before": FORMAL_STATE,
                "formal_state_after": FORMAL_STATE,
                "formal_transition": None,
                "formal_m3_start_allowed": False,
                "technical_actor_eligible": False,
                "formal_technical_actor_evidence_issued": False,
                "internal_state_before": INTERNAL_STATE_BEFORE,
                "internal_state_after": INTERNAL_STATE_AFTER,
                "internal_shadow_start_allowed": True,
                "report_alone_authorizes_formal_execution": False,
            },
            "shadow_status": "INTERNAL_SHADOW_CEREMONY_COMPLETE",
        }
        validate_shadow_runtime_report(report)
        return report
    finally:
        _cleanup_runtime_directory(runtime_dir)


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "AUTHORITY_CLASS",
    "CALCULATOR_SCHEMA_VERSION",
    "ENVELOPE_SCHEMA_VERSION",
    "EVIDENCE_CLASS",
    "FAIL_SHADOW_AUTHORITY_ESCALATION",
    "FAIL_SHADOW_BWRAP_ISOLATION",
    "FAIL_SHADOW_BWRAP_UNAVAILABLE",
    "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
    "FAIL_SHADOW_CALCULATOR_DISAGREEMENT",
    "FAIL_SHADOW_CALCULATOR_PROCESS",
    "FAIL_SHADOW_CRYPTO_BACKEND_UNAVAILABLE",
    "FAIL_SHADOW_IPC_PROTOCOL",
    "FAIL_SHADOW_LEDGER_TAMPERED",
    "FAIL_SHADOW_MARKER_TAMPERED",
    "FAIL_SHADOW_PLATFORM_UNSUPPORTED",
    "FAIL_SHADOW_PRIVATE_FILE_PERMISSIONS",
    "FAIL_SHADOW_PURPOSE_OR_KEY_COLLISION",
    "FAIL_SHADOW_REPORT_INVALID",
    "FAIL_SHADOW_ROLE_PROCESS",
    "FAIL_SHADOW_SECCOMP_PROBE",
    "FAIL_SHADOW_SECCOMP_UNAVAILABLE",
    "FAIL_SHADOW_SIGNATURE_INVALID",
    "FAIL_SHADOW_SNAPSHOT_MUTATED",
    "FAIL_SHADOW_SNAPSHOT_SOURCE_INVALID",
    "FAIL_SHADOW_STATE_ALREADY_COMPLETE",
    "FAIL_SHADOW_STATE_INSIDE_REPOSITORY",
    "FAIL_SHADOW_STATE_PATH_INVALID",
    "FAIL_SHADOW_STATE_PENDING_RECOVERY_REQUIRED",
    "FAIL_SHADOW_STATE_PERMISSIONS",
    "FORMAL_STATE",
    "INTERNAL_STATE_AFTER",
    "INTERNAL_STATE_BEFORE",
    "ISOLATION_LEVEL",
    "NAMESPACE_KINDS",
    "PURPOSE_IDS",
    "PURPOSE_ROLES",
    "REJECT_SHADOW_FIELD_SET",
    "REJECT_SHADOW_FIELD_TYPE",
    "SANITIZED_ENVIRONMENT",
    "SECCOMP_DENIED_SYSCALLS",
    "SCHEMA_VERSION",
    "SNAPSHOT_SCHEMA_VERSION",
    "ShadowRuntimeError",
    "create_readonly_input_snapshot",
    "create_shadow_state_directory",
    "inspect_shadow_state",
    "probe_shadow_admission",
    "run_internal_shadow_ceremony",
    "run_shadow_admission_probes",
    "validate_readonly_input_snapshot_report",
    "validate_shadow_admission_report",
    "validate_shadow_runtime_report",
    "verify_shadow_envelope",
]


if __name__ == "__main__":  # Internal bubblewrap worker entry only.
    if sys.argv[1:] == ["--internal-shadow-bootstrap"]:
        _internal_shadow_bootstrap()
    if sys.argv[1:] != ["--internal-shadow-worker"]:
        raise SystemExit(2)
    raise SystemExit(_internal_worker_cli())
