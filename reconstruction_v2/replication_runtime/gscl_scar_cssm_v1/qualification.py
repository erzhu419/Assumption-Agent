"""Source-free exact runtime qualification for the SCAR CSSM workers.

This harness is deliberately not an effect study.  It builds a deterministic
public synthetic action pack in memory, launches the same two worker CLI
processes under the same per-child Landlock policy, waits for both runtime
receipts, publishes the cross-shard release, and stops at a durable records
barrier.  It never creates or opens a SCAR label pack or HMAC secret and never
imports or calls the scorer.

The synthetic pack contains two executable items (one per shard) and 389
intentional pre-model normalized-collision items.  Its 391-item topology is a
compatibility requirement of the frozen worker validator, not evidence about
the SCAR source or any task effect.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from replication_runtime.gscl_scar_cssm_v1 import worker


VERSION = "gscl_scar_cssm_qualification_v1"
SAFE_RECEIPT_SCHEMA = f"{VERSION}.safe_receipt.v1"
SYNTHETIC_ACTION_SCHEMA = "gscl_scar_cssm_source_v1.action_pack.v1"
SYNTHETIC_ITEM_COUNT = 391
EXECUTABLE_ITEM_ORDINALS = frozenset({0, 1})
EXPECTED_PREMODEL_FAILURE_COUNT = SYNTHETIC_ITEM_COUNT - len(
    EXECUTABLE_ITEM_ORDINALS
)
PUBLIC_COMPATIBILITY_SOURCE_SHA256 = (
    "12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369"
)
PUBLIC_COMPATIBILITY_SOURCE_SIZE_BYTES = 1_393_355
ARM_IDS = (
    "semantic_only",
    "flat_structural",
    "full_no_composition",
    "full_with_length2_composition",
    "full_with_length2_composition_target_color_shuffle",
)
VARIANT_NAMES = ("base", "system_swap")
SHARD_COUNT = 2
MAXIMUM_RECEIPT_BYTES = 4 * 1024 * 1024
MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ITEM_TOKEN = re.compile(r"scar-item-v1-[0-9a-f]{64}\Z")
_SLOT_TOKEN = re.compile(r"scar-slot-v1-[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}\Z")

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14
_LL_BASE = (
    _LL_EXECUTE
    | _LL_WRITE_FILE
    | _LL_READ_FILE
    | _LL_READ_DIR
    | _LL_REMOVE_DIR
    | _LL_REMOVE_FILE
    | _LL_MAKE_CHAR
    | _LL_MAKE_DIR
    | _LL_MAKE_REG
    | _LL_MAKE_SOCK
    | _LL_MAKE_FIFO
    | _LL_MAKE_BLOCK
    | _LL_MAKE_SYM
)


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class ScarCssmQualificationError(RuntimeError):
    """Stable source-free qualification error."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...

    def wait(self) -> int: ...


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    status: str
    reason_codes: tuple[str, ...]
    host_mem_available_bytes: int | None = None
    selected_gpu_free_mib: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    study_id: str
    root: Path
    project_root: Path
    python_executable: Path
    qwen_model_root: Path
    qwen_manifest_path: Path
    minilm_model_root: Path
    minilm_manifest_path: Path
    nvidia_smi_path: Path
    gpu_uuids: tuple[str, str]
    execution_freeze_sha256: str
    lock_path: Path
    minimum_gpu_free_mib: int = 1
    minimum_host_available_bytes: int = 1
    runtime_barrier_timeout_seconds: int = 3_600


@dataclass(frozen=True, slots=True)
class QualificationPaths:
    root: Path
    inputs: Path
    control: Path
    release_directory: Path
    work: Path
    runtime: Path
    action_pack: Path
    sandbox_receipt: Path
    forbidden_label_canary: Path
    action_release: Path
    safe_receipt: Path

    @classmethod
    def for_root(cls, root: Path) -> "QualificationPaths":
        return cls(
            root=root,
            inputs=root / "synthetic_input",
            control=root / "control",
            release_directory=root / "release",
            work=root / "work",
            runtime=root / "runtime",
            action_pack=root / "synthetic_input/action.synthetic.private.json",
            sandbox_receipt=root / "synthetic_input/sandbox.safe.json",
            forbidden_label_canary=(
                root / "control/forbidden_label_canary.synthetic.private"
            ),
            action_release=root / "release/two_shard_action_release.safe.json",
            safe_receipt=root / "control/qualification.safe.json",
        )


@dataclass(frozen=True, slots=True)
class QualificationDependencies:
    filesystem_type: Callable[[Path], str]
    resource_probe: Callable[[QualificationConfig], AdmissionDecision]
    popen_factory: Callable[..., ProcessHandle]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_CANONICAL_JSON_INVALID"
        ) from exc


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(body)
    return {**normalized, "self_sha256": _content_hash(normalized)}


def _synthetic_token(kind: str, *parts: object) -> str:
    prefix = "scar-item-v1-" if kind == "item" else "scar-slot-v1-"
    payload = ":".join((VERSION, kind, *(str(part) for part in parts)))
    return prefix + hashlib.sha256(payload.encode("ascii")).hexdigest()


def _public_story(ordinal: int, side: str) -> str:
    return " ".join(
        f"Qualification{ordinal:03d}{side}{index:03d}" for index in range(44)
    ) + "."


def _synthetic_side(
    ordinal: int, side: str, *, normalized_collision: bool
) -> dict[str, Any]:
    story = _public_story(ordinal, side)
    surfaces = (
        ("K", "K")
        if normalized_collision and side == "left"
        else (
            f"Qualification{ordinal:03d}{side}000",
            f"Qualification{ordinal:03d}{side}043",
        )
    )
    return {
        "background": story,
        "slots": [
            {
                "opaque_slot_id": _synthetic_token(
                    "slot", ordinal, side, slot_index
                ),
                "surface": surface,
            }
            for slot_index, surface in enumerate(surfaces)
        ],
        "system": f"public synthetic system {ordinal:03d} {side}",
    }


def build_public_synthetic_action_pack_v1(study_id: str) -> dict[str, Any]:
    """Build the deterministic public 391-item worker-compatibility pack."""

    if not isinstance(study_id, str) or _STUDY_ID.fullmatch(study_id) is None:
        raise ScarCssmQualificationError("QUALIFICATION_STUDY_ID_INVALID")
    items: list[dict[str, Any]] = []
    for ordinal in range(SYNTHETIC_ITEM_COUNT):
        collision = ordinal not in EXECUTABLE_ITEM_ORDINALS
        left = _synthetic_side(
            ordinal, "left", normalized_collision=collision
        )
        right = _synthetic_side(
            ordinal, "right", normalized_collision=False
        )
        items.append(
            {
                "item_token": _synthetic_token("item", ordinal),
                "variants": {
                    "base": {"left": left, "right": right},
                    "system_swap": {"left": right, "right": left},
                },
            }
        )
    core = {
        "items": items,
        "schema": SYNTHETIC_ACTION_SCHEMA,
        "slot_collection_semantics": "unordered",
        "source_sha256": PUBLIC_COMPATIBILITY_SOURCE_SHA256,
        "source_size_bytes": PUBLIC_COMPATIBILITY_SOURCE_SIZE_BYTES,
        "study_id": study_id,
        "variant_names": list(VARIANT_NAMES),
    }
    action_commitment = _content_hash(core)
    body = {
        **core,
        "action_commitment_sha256": action_commitment,
        "cross_binding_hmac_sha256": _content_hash(
            {
                "qualification_only": True,
                "secret_used": False,
                "study_id": study_id,
            }
        ),
        "label_commitment_sha256": _content_hash(
            {
                "label_pack_created": False,
                "qualification_only": True,
                "study_id": study_id,
            }
        ),
    }
    return {**body, "self_sha256": _content_hash(body)}


def _sandbox_receipt(study_id: str) -> dict[str, Any]:
    return _self_hashed(
        {
            "action_external_network_denied": True,
            "action_label_path_denied": True,
            "ip_address_deny": "any",
            "qualification_only": True,
            "restrict_address_families": "AF_UNIX",
            "schema": worker.SANDBOX_RECEIPT_SCHEMA,
            "status": "frozen",
            "study_id": study_id,
        }
    )


def _publish_once(path: Path, raw: bytes) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ScarCssmQualificationError("QUALIFICATION_OUTPUT_ALREADY_EXISTS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, maximum_bytes: int) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_INPUT_UNAVAILABLE"
        ) from exc
    if (
        not path.is_absolute()
        or not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        raise ScarCssmQualificationError("QUALIFICATION_INPUT_INVALID")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_INPUT_UNAVAILABLE"
        ) from exc
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        identity
        != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        or identity
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or len(raw) > maximum_bytes
    ):
        raise ScarCssmQualificationError("QUALIFICATION_INPUT_CHANGED")
    return raw, hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, *, issue_id: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ScarCssmQualificationError(issue_id) from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise ScarCssmQualificationError(issue_id)
    return value


def _load_self_hashed(path: Path, *, maximum_bytes: int) -> tuple[dict[str, Any], str]:
    raw, file_hash = _read_regular(path, maximum_bytes=maximum_bytes)
    value = _strict_json(raw, issue_id="QUALIFICATION_RECEIPT_INVALID")
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if not isinstance(claimed, str) or claimed != _content_hash(body):
        raise ScarCssmQualificationError("QUALIFICATION_RECEIPT_INVALID")
    return value, file_hash


def _validate_config(config: QualificationConfig) -> None:
    paths = (
        config.root,
        config.project_root,
        config.python_executable,
        config.qwen_model_root,
        config.qwen_manifest_path,
        config.minilm_model_root,
        config.minilm_manifest_path,
        config.nvidia_smi_path,
        config.lock_path,
    )
    if (
        not isinstance(config.study_id, str)
        or _STUDY_ID.fullmatch(config.study_id) is None
        or any(not isinstance(path, Path) or not path.is_absolute() for path in paths)
        or config.root == Path(config.root.anchor)
        or len(set(config.gpu_uuids)) != SHARD_COUNT
        or any(_GPU_UUID.fullmatch(value) is None for value in config.gpu_uuids)
        or _SHA256.fullmatch(config.execution_freeze_sha256) is None
        or type(config.minimum_gpu_free_mib) is not int
        or config.minimum_gpu_free_mib < 0
        or type(config.minimum_host_available_bytes) is not int
        or config.minimum_host_available_bytes < 0
        or type(config.runtime_barrier_timeout_seconds) is not int
        or config.runtime_barrier_timeout_seconds < 1
    ):
        raise ScarCssmQualificationError("QUALIFICATION_CONFIG_INVALID")
    root = config.root.resolve(strict=False)
    for readable_root in (
        config.project_root,
        config.qwen_model_root,
        config.minilm_model_root,
        config.python_executable.resolve(strict=True).parent.parent,
    ):
        try:
            root.relative_to(readable_root.resolve(strict=True))
        except ValueError:
            continue
        raise ScarCssmQualificationError(
            "QUALIFICATION_SANDBOX_TOPOLOGY_INVALID"
        )


def _mount_fstype(path: Path) -> str:
    candidate = path
    while not candidate.exists():
        if candidate == candidate.parent:
            raise ScarCssmQualificationError("QUALIFICATION_MOUNT_UNAVAILABLE")
        candidate = candidate.parent
    resolved = candidate.resolve(strict=True)
    best: tuple[int, str] | None = None
    try:
        rows = Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_MOUNT_UNAVAILABLE"
        ) from exc
    for row in rows:
        left, separator, right = row.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount = Path(left_fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        candidate_value = (len(mount.parts), right_fields[0])
        if best is None or candidate_value[0] > best[0]:
            best = candidate_value
    if best is None:
        raise ScarCssmQualificationError("QUALIFICATION_MOUNT_UNAVAILABLE")
    return best[1]


def _run_nvidia_smi(path: Path, arguments: Sequence[str]) -> list[str]:
    try:
        completed = subprocess.run(
            [str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_RESOURCE_PROBE_FAILED"
        ) from exc
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _default_resource_probe(config: QualificationConfig) -> AdmissionDecision:
    try:
        mem_available = None
        for row in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if row.startswith("MemAvailable:"):
                fields = row.split()
                if len(fields) == 3 and fields[1].isdigit() and fields[2] == "kB":
                    mem_available = int(fields[1]) * 1024
                    break
        if mem_available is None:
            raise ValueError("MemAvailable absent")
        gpu_rows = _run_nvidia_smi(
            config.nvidia_smi_path,
            ("--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"),
        )
        gpu_free: dict[str, int] = {}
        for row in gpu_rows:
            fields = [value.strip() for value in row.split(",")]
            if len(fields) != 2 or not fields[1].isdigit():
                raise ValueError("GPU row invalid")
            gpu_free[fields[0]] = int(fields[1])
        process_rows = _run_nvidia_smi(
            config.nvidia_smi_path,
            ("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"),
        )
        busy = {
            row.split(",", 1)[0].strip()
            for row in process_rows
            if "," in row
        }
        selected = tuple(gpu_free[value] for value in config.gpu_uuids)
    except (KeyError, OSError, ValueError) as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_RESOURCE_PROBE_FAILED"
        ) from exc
    reasons: list[str] = []
    if mem_available < config.minimum_host_available_bytes:
        reasons.append("HOST_MEMORY_BELOW_POLICY")
    if any(value in busy for value in config.gpu_uuids):
        reasons.append("SELECTED_GPU_HAS_COMPUTE_PROCESS")
    if any(value < config.minimum_gpu_free_mib for value in selected):
        reasons.append("SELECTED_GPU_FREE_MEMORY_BELOW_POLICY")
    return AdmissionDecision(
        status=("DEFERRED_SHARED_RESOURCE" if reasons else "ADMITTED_SHARED_RESOURCE"),
        reason_codes=tuple(reasons),
        host_mem_available_bytes=mem_available,
        selected_gpu_free_mib=selected,
    )


DEFAULT_DEPENDENCIES = QualificationDependencies(
    filesystem_type=_mount_fstype,
    resource_probe=_default_resource_probe,
    popen_factory=subprocess.Popen,
    monotonic=time.monotonic,
    sleep=time.sleep,
)


def _try_lock(path: Path) -> int | None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _release_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _landlock_abi_version() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if result < 1:
        raise ScarCssmQualificationError("QUALIFICATION_LANDLOCK_UNAVAILABLE")
    return result


def _landlock_rights(abi: int) -> int:
    rights = _LL_BASE
    if abi >= 2:
        rights |= _LL_REFER
    if abi >= 3:
        rights |= _LL_TRUNCATE
    return rights


def _python_runtime_read_roots(executable: Path) -> tuple[Path, ...]:
    """Return the venv and its declared base runtime without widening to /var."""

    try:
        environment_root = executable.resolve(strict=True).parent.parent
        pyvenv = environment_root / "pyvenv.cfg"
        roots = [environment_root]
        if pyvenv.exists():
            metadata = pyvenv.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_size > 16 * 1024
            ):
                raise ValueError("invalid pyvenv.cfg metadata")
            home_values = [
                line.partition("=")[2].strip()
                for line in pyvenv.read_text(
                    encoding="utf-8", errors="strict"
                ).splitlines()
                if line.partition("=")[0].strip().lower() == "home"
            ]
            if len(home_values) != 1:
                raise ValueError("pyvenv.cfg home is not unique")
            home = Path(home_values[0])
            if not home.is_absolute():
                raise ValueError("pyvenv.cfg home is not absolute")
            home = home.resolve(strict=True)
            base_root = home.parent if home.name == "bin" else home
            if not base_root.is_dir():
                raise ValueError("pyvenv.cfg base runtime is not a directory")
            roots.append(base_root)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ScarCssmQualificationError(
            "QUALIFICATION_PYTHON_RUNTIME_TOPOLOGY_INVALID"
        ) from exc
    return tuple(dict.fromkeys(roots))


def _apply_landlock(
    *,
    read_paths: Sequence[Path],
    write_paths: Sequence[Path],
    device_paths: Sequence[Path],
) -> None:
    abi = _landlock_abi_version()
    handled = _landlock_rights(abi)
    libc = ctypes.CDLL(None, use_errno=True)
    attribute = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(attribute),
            ctypes.sizeof(attribute),
            ctypes.c_uint(0),
        )
    )
    if ruleset_fd < 0:
        raise ScarCssmQualificationError("QUALIFICATION_LANDLOCK_CREATE_FAILED")

    def add(path: Path, rights: int) -> None:
        flags = os.O_PATH | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ScarCssmQualificationError(
                "QUALIFICATION_LANDLOCK_ALLOWLIST_UNAVAILABLE"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            allowed = rights
            if not stat.S_ISDIR(metadata.st_mode):
                allowed &= ~(
                    _LL_READ_DIR
                    | _LL_REMOVE_DIR
                    | _LL_REMOVE_FILE
                    | _LL_MAKE_CHAR
                    | _LL_MAKE_DIR
                    | _LL_MAKE_REG
                    | _LL_MAKE_SOCK
                    | _LL_MAKE_FIFO
                    | _LL_MAKE_BLOCK
                    | _LL_MAKE_SYM
                    | _LL_REFER
                )
            rule = _PathBeneathAttr(
                allowed_access=allowed,
                parent_fd=descriptor,
                reserved=0,
            )
            if (
                libc.syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(rule),
                    ctypes.c_uint(0),
                )
                != 0
            ):
                raise ScarCssmQualificationError(
                    "QUALIFICATION_LANDLOCK_RULE_FAILED"
                )
        finally:
            os.close(descriptor)

    try:
        read_rights = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR
        for path in dict.fromkeys(read_paths):
            add(path, read_rights)
        for path in dict.fromkeys(write_paths):
            add(path, handled)
        for path in dict.fromkeys(device_paths):
            add(path, _LL_READ_FILE | _LL_WRITE_FILE)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise ScarCssmQualificationError(
                "QUALIFICATION_NO_NEW_PRIVS_FAILED"
            )
        if (
            libc.syscall(
                _SYS_LANDLOCK_RESTRICT_SELF,
                ruleset_fd,
                ctypes.c_uint(0),
            )
            != 0
        ):
            raise ScarCssmQualificationError(
                "QUALIFICATION_LANDLOCK_RESTRICT_FAILED"
            )
    finally:
        os.close(ruleset_fd)


def _landlock_paths(
    config: QualificationConfig,
    paths: QualificationPaths,
    shard: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    system = (
        Path("/usr"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/lib"),
        Path("/lib64"),
        Path("/dev/null"),
        Path("/dev/urandom"),
    )
    python_runtime_roots = _python_runtime_read_roots(
        config.python_executable
    )
    read_paths = tuple(
        path
        for path in (
            *system,
            config.project_root,
            config.qwen_model_root,
            config.minilm_model_root,
            *python_runtime_roots,
            paths.action_pack,
            paths.sandbox_receipt,
            config.qwen_manifest_path,
            config.minilm_manifest_path,
            paths.release_directory,
        )
        if path.exists()
    )
    write_paths = (paths.work, paths.runtime / f"shard{shard}")
    device_paths = tuple(
        path
        for path in (
            Path("/dev/nvidia0"),
            Path("/dev/nvidia1"),
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia-uvm"),
            Path("/dev/nvidia-uvm-tools"),
            Path("/dev/nvidia-modeset"),
        )
        if path.exists()
    )
    if paths.forbidden_label_canary in read_paths:
        raise ScarCssmQualificationError(
            "QUALIFICATION_SANDBOX_TOPOLOGY_INVALID"
        )
    return read_paths, write_paths, device_paths


def _apply_child_landlock(
    config: QualificationConfig,
    paths: QualificationPaths,
    shard: int,
) -> None:
    read_paths, write_paths, devices = _landlock_paths(config, paths, shard)
    _apply_landlock(
        read_paths=read_paths,
        write_paths=write_paths,
        device_paths=devices,
    )


def _child_environment(
    config: QualificationConfig,
    paths: QualificationPaths,
    shard: int,
) -> dict[str, str]:
    private = paths.runtime / f"shard{shard}"
    home = private / "home"
    temporary = private / "tmp"
    cache = private / "cache"
    for path in (private, home, temporary, cache):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": config.gpu_uuids[shard],
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NO_PROXY": "*",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(config.project_root),
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TZ": "UTC",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _worker_argv(
    config: QualificationConfig,
    paths: QualificationPaths,
    *,
    shard: int,
    action_file_sha256: str,
    action_commitment_sha256: str,
    implementation_closure_sha256: str,
    sandbox_file_sha256: str,
) -> list[str]:
    return [
        str(config.python_executable),
        "-m",
        "replication_runtime.gscl_scar_cssm_v1.worker",
        "--action-pack",
        str(paths.action_pack),
        "--output-root",
        str(paths.work),
        "--study-id",
        config.study_id,
        "--shard-index",
        str(shard),
        "--qwen-model-root",
        str(config.qwen_model_root),
        "--qwen-manifest",
        str(config.qwen_manifest_path),
        "--minilm-model-root",
        str(config.minilm_model_root),
        "--minilm-manifest",
        str(config.minilm_manifest_path),
        "--sandbox-receipt",
        str(paths.sandbox_receipt),
        "--action-release",
        str(paths.action_release),
        "--forbidden-label-probe",
        str(paths.forbidden_label_canary),
        "--expected-action-file-sha256",
        action_file_sha256,
        "--expected-action-commitment-sha256",
        action_commitment_sha256,
        "--expected-implementation-closure-sha256",
        implementation_closure_sha256,
        "--expected-sandbox-receipt-sha256",
        sandbox_file_sha256,
        "--expected-execution-freeze-sha256",
        config.execution_freeze_sha256,
        "--expected-gpu-uuid",
        config.gpu_uuids[shard],
        "--expected-peer-gpu-uuid",
        config.gpu_uuids[1 - shard],
    ]


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    return descriptor


def _validate_runtime_receipt(
    value: Mapping[str, Any],
    *,
    config: QualificationConfig,
    shard: int,
    implementation_closure_sha256: str,
    sandbox_receipt: Mapping[str, Any],
    sandbox_file_sha256: str,
) -> None:
    expected_keys = {
        "execution",
        "execution_freeze_sha256",
        "forbidden_label_negative_canary",
        "gpu",
        "implementation_closure",
        "minilm",
        "network_negative_canary",
        "process_sandbox",
        "qwen",
        "sandbox_freeze",
        "sandbox_freeze_file",
        "schema",
        "self_sha256",
        "shard_count",
        "shard_index",
        "status",
        "study_id",
        "version",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value["schema"] != worker.RUNTIME_RECEIPT_SCHEMA
        or value["version"] != worker.VERSION
        or value["status"] != "qualified_before_action_pack_open"
        or value["study_id"] != config.study_id
        or value["shard_count"] != SHARD_COUNT
        or value["shard_index"] != shard
        or value["execution_freeze_sha256"]
        != config.execution_freeze_sha256
        or value["implementation_closure"].get("self_sha256")
        != implementation_closure_sha256
        or value["sandbox_freeze"] != sandbox_receipt
        or value["sandbox_freeze_file"].get("sha256")
        != sandbox_file_sha256
    ):
        raise ScarCssmQualificationError(
            "QUALIFICATION_RUNTIME_RECEIPT_INVALID"
        )
    process = value["process_sandbox"]
    network = value["network_negative_canary"]
    label = value["forbidden_label_negative_canary"]
    gpu = value["gpu"]
    execution = value["execution"]
    if (
        type(process) is not dict
        or set(process)
        != {"no_new_privileges", "seccomp_filter_count", "seccomp_mode"}
        or process["no_new_privileges"] is not True
        or type(network) is not dict
        or set(network)
        != {"AF_INET", "AF_INET6", "external_connect_attempt_count"}
        or network["external_connect_attempt_count"] != 0
        or any(
            type(network[name]) is not dict
            or set(network[name]) != {"creation_denied", "errno"}
            or network[name]["creation_denied"] is not True
            or type(network[name]["errno"]) is not int
            or network[name]["errno"] <= 0
            for name in ("AF_INET", "AF_INET6")
        )
        or type(label) is not dict
        or set(label) != {"errno", "open_denied", "read_count"}
        or label["open_denied"] is not True
        or label["read_count"] != 0
        or type(label["errno"]) is not int
        or label["errno"] not in {errno.EACCES, errno.EPERM}
        or type(gpu) is not dict
        or gpu.get("physical_uuid") != config.gpu_uuids[shard]
        or gpu.get("cuda_visible_devices") != config.gpu_uuids[shard]
        or gpu.get("visible_device_count") != 1
        or gpu.get("logical_current_device") != 0
        or gpu.get("parameter_devices") != ["cuda:0"]
        or type(execution) is not dict
        or execution.get("hf_hub_offline") != "1"
        or execution.get("transformers_offline") != "1"
        or execution.get("tokenizers_parallelism") not in {"false", "False"}
        or execution.get("cuda_runtime_available") is not True
        or execution.get("deterministic_algorithms") is not True
        or execution.get("matmul_tf32") is not False
        or execution.get("cudnn_tf32") is not False
        or execution.get("cudnn_benchmark") is not False
        or type(value["qwen"]) is not dict
        or not isinstance(value["qwen"].get("runtime_commitment"), str)
        or _SHA256.fullmatch(value["qwen"]["runtime_commitment"]) is None
        or type(value["minilm"]) is not dict
        or not isinstance(value["minilm"].get("encoder_binding_sha256"), str)
        or _SHA256.fullmatch(value["minilm"]["encoder_binding_sha256"])
        is None
    ):
        raise ScarCssmQualificationError(
            "QUALIFICATION_RUNTIME_RECEIPT_INVALID"
        )


def _wait_for_runtime_barrier(
    *,
    processes: Mapping[int, ProcessHandle],
    config: QualificationConfig,
    paths: QualificationPaths,
    dependencies: QualificationDependencies,
    implementation_closure_sha256: str,
    sandbox_receipt: Mapping[str, Any],
    sandbox_file_sha256: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    deadline = dependencies.monotonic() + config.runtime_barrier_timeout_seconds
    values: dict[int, dict[str, Any]] = {}
    file_hashes: dict[int, str] = {}
    while set(values) != set(range(SHARD_COUNT)):
        for shard in range(SHARD_COUNT):
            if shard in values:
                continue
            process = processes[shard]
            status = process.poll()
            runtime_path = paths.work / f"shard{shard}.runtime.safe.json"
            sentinel_path = paths.work / f"shard{shard}.attempt.sentinel"
            if runtime_path.exists() and sentinel_path.exists():
                try:
                    value, file_hash = _load_self_hashed(
                        runtime_path, maximum_bytes=MAXIMUM_RECEIPT_BYTES
                    )
                    sentinel_raw, _sentinel_hash = _read_regular(
                        sentinel_path, maximum_bytes=MAXIMUM_RECEIPT_BYTES
                    )
                    sentinel = _strict_json(
                        sentinel_raw,
                        issue_id="QUALIFICATION_SENTINEL_INVALID",
                    )
                except ScarCssmQualificationError:
                    if status is None and dependencies.monotonic() < deadline:
                        continue
                    raise
                expected_sentinel = {
                    "expected_action_commitment_sha256": (
                        build_public_synthetic_action_pack_v1(config.study_id)[
                            "action_commitment_sha256"
                        ]
                    ),
                    "expected_action_file_sha256": hashlib.sha256(
                        _canonical_bytes(
                            build_public_synthetic_action_pack_v1(config.study_id)
                        )
                    ).hexdigest(),
                    "expected_execution_freeze_sha256": (
                        config.execution_freeze_sha256
                    ),
                    "runtime_receipt_sha256": file_hash,
                    "shard_count": SHARD_COUNT,
                    "shard_index": shard,
                    "study_id": config.study_id,
                    "version": worker.VERSION,
                }
                if sentinel != expected_sentinel:
                    raise ScarCssmQualificationError(
                        "QUALIFICATION_SENTINEL_INVALID"
                    )
                _validate_runtime_receipt(
                    value,
                    config=config,
                    shard=shard,
                    implementation_closure_sha256=(
                        implementation_closure_sha256
                    ),
                    sandbox_receipt=sandbox_receipt,
                    sandbox_file_sha256=sandbox_file_sha256,
                )
                values[shard] = value
                file_hashes[shard] = file_hash
            elif status is not None:
                raise ScarCssmQualificationError(
                    "QUALIFICATION_WORKER_EXITED_BEFORE_RUNTIME_BARRIER"
                )
        if set(values) == set(range(SHARD_COUNT)):
            break
        if dependencies.monotonic() >= deadline:
            raise ScarCssmQualificationError(
                "QUALIFICATION_RUNTIME_BARRIER_TIMEOUT"
            )
        dependencies.sleep(0.1)
    if (
        values[0]["qwen"]["runtime_commitment"]
        != values[1]["qwen"]["runtime_commitment"]
        or values[0]["minilm"]["encoder_binding_sha256"]
        != values[1]["minilm"]["encoder_binding_sha256"]
    ):
        raise ScarCssmQualificationError(
            "QUALIFICATION_CROSS_SHARD_RUNTIME_DRIFT"
        )
    return values, file_hashes


def _action_release(
    *,
    config: QualificationConfig,
    action_file_sha256: str,
    action_commitment_sha256: str,
    runtime_file_hashes: Mapping[int, str],
) -> dict[str, Any]:
    return _self_hashed(
        {
            "action_commitment_sha256": action_commitment_sha256,
            "action_file_sha256": action_file_sha256,
            "execution_freeze_sha256": config.execution_freeze_sha256,
            "gpu_uuid_by_shard": {
                str(shard): config.gpu_uuids[shard]
                for shard in range(SHARD_COUNT)
            },
            "runtime_receipt_file_sha256_by_shard": {
                str(shard): runtime_file_hashes[shard]
                for shard in range(SHARD_COUNT)
            },
            "schema": worker.ACTION_RELEASE_SCHEMA,
            "shard_count": SHARD_COUNT,
            "status": "release_both_shards_to_action_pack",
            "study_id": config.study_id,
        }
    )


def _validate_prediction_static(value: Any, *, item_token: str) -> None:
    if type(value) is not dict or set(value) != {
        "diagnostics",
        "execution",
        "item_token",
        "proposal_pools",
        "variants",
    }:
        raise ScarCssmQualificationError("QUALIFICATION_RECORD_SCHEMA_INVALID")
    if value["item_token"] != item_token:
        raise ScarCssmQualificationError("QUALIFICATION_RECORD_SCHEMA_INVALID")
    execution = value["execution"]
    if type(execution) is not dict or set(execution) != {
        "document_call_count",
        "error_code",
        "structural_status",
    }:
        raise ScarCssmQualificationError("QUALIFICATION_RECORD_SCHEMA_INVALID")
    for key in ("variants", "proposal_pools", "diagnostics"):
        if type(value[key]) is not dict or set(value[key]) != set(
            VARIANT_NAMES
        ):
            raise ScarCssmQualificationError(
                "QUALIFICATION_RECORD_SCHEMA_INVALID"
            )
    for variant_name in VARIANT_NAMES:
        variant = value["variants"][variant_name]
        pools = value["proposal_pools"][variant_name]
        if (
            type(variant) is not dict
            or set(variant) != {"arms"}
            or type(variant["arms"]) is not dict
            or set(variant["arms"]) != set(ARM_IDS)
            or type(pools) is not dict
            or set(pools) != {"semantic_kbest", "structure_kbest"}
            or type(pools["semantic_kbest"]) is not list
            or type(pools["structure_kbest"]) is not list
            or type(value["diagnostics"][variant_name]) is not dict
        ):
            raise ScarCssmQualificationError(
                "QUALIFICATION_RECORD_SCHEMA_INVALID"
            )
        for arm in variant["arms"].values():
            if type(arm) is not dict or set(arm) != {
                "disposition",
                "error_code",
                "pairs",
            }:
                raise ScarCssmQualificationError(
                    "QUALIFICATION_RECORD_SCHEMA_INVALID"
                )


def _validate_evidence_static(value: Any) -> None:
    if type(value) is not dict or set(value) != {
        "availability",
        "error_code",
        "semantic_matrix",
        "sides",
        "variants",
    }:
        raise ScarCssmQualificationError("QUALIFICATION_RECORD_SCHEMA_INVALID")
    if value["availability"] not in {"COMPLETE", "PREMODEL_TYPED_FAILURE"}:
        raise ScarCssmQualificationError("QUALIFICATION_RECORD_SCHEMA_INVALID")


def _load_records(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> list[dict[str, Any]]:
    raw, file_hash = _read_regular(
        path, maximum_bytes=SYNTHETIC_ITEM_COUNT * MAXIMUM_RECORD_BYTES
    )
    if (
        file_hash != expected_sha256
        or len(raw) != expected_size
        or (raw and not raw.endswith(b"\n"))
    ):
        raise ScarCssmQualificationError("QUALIFICATION_RECORD_FILE_INVALID")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line or len(line) > MAXIMUM_RECORD_BYTES:
            raise ScarCssmQualificationError(
                "QUALIFICATION_RECORD_FILE_INVALID"
            )
        value = _strict_json(
            line, issue_id="QUALIFICATION_RECORD_SCHEMA_INVALID"
        )
        if set(value) != {
            "evidence",
            "item_token",
            "ordinal_within_shard",
            "prediction",
            "self_sha256",
        }:
            raise ScarCssmQualificationError(
                "QUALIFICATION_RECORD_SCHEMA_INVALID"
            )
        body = dict(value)
        claimed = body.pop("self_sha256")
        if claimed != _content_hash(body):
            raise ScarCssmQualificationError(
                "QUALIFICATION_RECORD_SCHEMA_INVALID"
            )
        _validate_prediction_static(
            value["prediction"], item_token=value["item_token"]
        )
        _validate_evidence_static(value["evidence"])
        rows.append(value)
    return rows


def _validate_terminal(
    value: Mapping[str, Any],
    *,
    config: QualificationConfig,
    shard: int,
    action_file_sha256: str,
    action_commitment_sha256: str,
    release: Mapping[str, Any],
    release_file_sha256: str,
    runtime: Mapping[str, Any],
    runtime_file_sha256: str,
) -> None:
    expected_keys = {
        "action_commitment_sha256",
        "action_pack_file_receipt",
        "action_release_file_receipt",
        "action_release_self_sha256",
        "arm_ids",
        "document_call_count",
        "encoder_binding_sha256",
        "external_network_call_count",
        "formal_label_pack_access_count",
        "formal_scorer_access_count",
        "item_count",
        "mechanism_resource_totals",
        "output_root_receipt",
        "private_records_file_sha256",
        "private_records_file_size_bytes",
        "runtime_receipt_file_sha256",
        "runtime_receipt_self_sha256",
        "schema",
        "self_sha256",
        "shard_count",
        "shard_index",
        "status",
        "structural_error_code_counts",
        "structural_typed_failure_count",
        "study_id",
        "variant_names",
        "version",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value["schema"] != worker.SHARD_TERMINAL_SCHEMA
        or value["version"] != worker.VERSION
        or value["status"] != "complete"
        or value["study_id"] != config.study_id
        or value["shard_count"] != SHARD_COUNT
        or value["shard_index"] != shard
        or value["arm_ids"] != list(ARM_IDS)
        or value["variant_names"] != list(VARIANT_NAMES)
        or value["action_commitment_sha256"]
        != action_commitment_sha256
        or value["action_pack_file_receipt"].get("sha256")
        != action_file_sha256
        or value["action_release_file_receipt"].get("sha256")
        != release_file_sha256
        or value["action_release_self_sha256"] != release["self_sha256"]
        or value["runtime_receipt_file_sha256"] != runtime_file_sha256
        or value["runtime_receipt_self_sha256"] != runtime["self_sha256"]
        or value["encoder_binding_sha256"]
        != runtime["minilm"]["encoder_binding_sha256"]
        or value["external_network_call_count"] != 0
        or value["formal_label_pack_access_count"] != 0
        or value["formal_scorer_access_count"] != 0
    ):
        raise ScarCssmQualificationError(
            "QUALIFICATION_SHARD_TERMINAL_INVALID"
        )


def _verify_records_barrier(
    *,
    config: QualificationConfig,
    paths: QualificationPaths,
    action_pack: Mapping[str, Any],
    action_file_sha256: str,
    release: Mapping[str, Any],
    release_file_sha256: str,
    runtimes: Mapping[int, Mapping[str, Any]],
    runtime_file_hashes: Mapping[int, str],
) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    terminal_file_hashes: dict[str, str] = {}
    record_file_hashes: dict[str, str] = {}
    terminal_values: dict[int, dict[str, Any]] = {}
    resource_totals: dict[str, int] = {}
    total_document_calls = 0
    total_typed_failures = 0
    for shard in range(SHARD_COUNT):
        terminal, terminal_file_hash = _load_self_hashed(
            paths.work / f"shard{shard}.terminal.safe.json",
            maximum_bytes=MAXIMUM_RECEIPT_BYTES,
        )
        _validate_terminal(
            terminal,
            config=config,
            shard=shard,
            action_file_sha256=action_file_sha256,
            action_commitment_sha256=action_pack[
                "action_commitment_sha256"
            ],
            release=release,
            release_file_sha256=release_file_sha256,
            runtime=runtimes[shard],
            runtime_file_sha256=runtime_file_hashes[shard],
        )
        records = _load_records(
            paths.work / f"shard{shard}.records.private.jsonl",
            expected_sha256=terminal["private_records_file_sha256"],
            expected_size=terminal["private_records_file_size_bytes"],
        )
        expected_items = action_pack["items"][shard::SHARD_COUNT]
        if (
            terminal["item_count"] != len(expected_items)
            or len(records) != len(expected_items)
            or [row["item_token"] for row in records]
            != [row["item_token"] for row in expected_items]
            or [row["ordinal_within_shard"] for row in records]
            != list(range(len(records)))
        ):
            raise ScarCssmQualificationError(
                "QUALIFICATION_RECORD_RECOMPOSITION_INVALID"
            )
        shard_document_calls = 0
        shard_failures = 0
        shard_error_counts: dict[str, int] = {}
        shard_resources: dict[str, int] = {}
        for local_ordinal, record in enumerate(records):
            global_ordinal = shard + local_ordinal * SHARD_COUNT
            executable = global_ordinal in EXECUTABLE_ITEM_ORDINALS
            prediction = record["prediction"]
            evidence = record["evidence"]
            expected_execution = (
                {
                    "document_call_count": 2,
                    "error_code": None,
                    "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
                }
                if executable
                else {
                    "document_call_count": 0,
                    "error_code": "SLOT_BINDER_TYPED_FAILURE",
                    "structural_status": "TYPED_FAILURE",
                }
            )
            expected_availability = (
                "COMPLETE" if executable else "PREMODEL_TYPED_FAILURE"
            )
            if (
                prediction["execution"] != expected_execution
                or evidence["availability"] != expected_availability
            ):
                raise ScarCssmQualificationError(
                    "QUALIFICATION_RECORD_EXECUTION_INVALID"
                )
            shard_document_calls += expected_execution["document_call_count"]
            shard_failures += int(not executable)
            if expected_execution["error_code"] is not None:
                code = expected_execution["error_code"]
                shard_error_counts[code] = shard_error_counts.get(code, 0) + 1
            counts = worker._mechanism_resource_counts(evidence)  # noqa: SLF001
            for key, count in counts.items():
                shard_resources[key] = shard_resources.get(key, 0) + count
                resource_totals[key] = resource_totals.get(key, 0) + count
        if (
            terminal["document_call_count"] != shard_document_calls
            or terminal["structural_typed_failure_count"] != shard_failures
            or terminal["structural_error_code_counts"] != shard_error_counts
            or terminal["mechanism_resource_totals"] != shard_resources
        ):
            raise ScarCssmQualificationError(
                "QUALIFICATION_SHARD_ACCOUNTING_INVALID"
            )
        total_document_calls += shard_document_calls
        total_typed_failures += shard_failures
        terminal_file_hashes[str(shard)] = terminal_file_hash
        record_file_hashes[str(shard)] = terminal[
            "private_records_file_sha256"
        ]
        terminal_values[shard] = terminal
        all_records.extend(records)
    if (
        len(all_records) != SYNTHETIC_ITEM_COUNT
        or total_document_calls != 2 * len(EXECUTABLE_ITEM_ORDINALS)
        or total_typed_failures != EXPECTED_PREMODEL_FAILURE_COUNT
        or len({row["item_token"] for row in all_records})
        != SYNTHETIC_ITEM_COUNT
    ):
        raise ScarCssmQualificationError(
            "QUALIFICATION_RECORD_RECOMPOSITION_INVALID"
        )
    return {
        "document_call_count": total_document_calls,
        "mechanism_resource_totals": dict(sorted(resource_totals.items())),
        "private_records_file_sha256_by_shard": record_file_hashes,
        "records_recomposition_commitment_sha256": _content_hash(
            sorted(all_records, key=lambda row: row["item_token"])
        ),
        "shard_terminal_file_sha256_by_shard": terminal_file_hashes,
        "structural_typed_failure_count": total_typed_failures,
    }


def _safe_value(value: Any) -> None:
    forbidden_keys = {
        "background",
        "evidence",
        "gold_pairs",
        "item_token",
        "opaque_slot_id",
        "pairs",
        "prediction",
        "surface",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_keys:
                raise ScarCssmQualificationError(
                    "QUALIFICATION_SAFE_RECEIPT_LEAKAGE"
                )
            _safe_value(child)
    elif isinstance(value, list):
        for child in value:
            _safe_value(child)
    elif isinstance(value, str) and (
        "scar-item-v1-" in value or "scar-slot-v1-" in value
    ):
        raise ScarCssmQualificationError(
            "QUALIFICATION_SAFE_RECEIPT_LEAKAGE"
        )


def _deferred_receipt(
    config: QualificationConfig, decision: AdmissionDecision
) -> dict[str, Any]:
    body = {
        "effect_study_attempt_count": 0,
        "formal_label_pack_access_count": 0,
        "formal_scar_source_access_count": 0,
        "formal_scorer_access_count": 0,
        "hmac_secret_access_count": 0,
        "qualification_only": True,
        "resource_admission": {
            "host_mem_available_bytes": decision.host_mem_available_bytes,
            "reason_codes": list(decision.reason_codes),
            "selected_gpu_free_mib": list(decision.selected_gpu_free_mib),
            "status": decision.status,
        },
        "schema": SAFE_RECEIPT_SCHEMA,
        "status": decision.status,
        "study_id": config.study_id,
    }
    _safe_value(body)
    return _self_hashed(body)


def run_qualification_once(
    config: QualificationConfig,
    *,
    dependencies: QualificationDependencies = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    """Execute one fresh source-free qualification root through records seal."""

    _validate_config(config)
    if config.root.exists() or config.root.is_symlink():
        raise ScarCssmQualificationError("QUALIFICATION_ROOT_NOT_FRESH")
    if dependencies.filesystem_type(config.root) != "ext4":
        raise ScarCssmQualificationError("QUALIFICATION_ROOT_NOT_EXT4")
    lock = _try_lock(config.lock_path)
    if lock is None:
        return _deferred_receipt(
            config,
            AdmissionDecision(
                status="DEFERRED_SHARED_RESOURCE",
                reason_codes=("QUALIFICATION_LOCK_OCCUPIED",),
            ),
        )
    try:
        decision = dependencies.resource_probe(config)
        if decision.status != "ADMITTED_SHARED_RESOURCE":
            return _deferred_receipt(config, decision)
        paths = QualificationPaths.for_root(config.root)
        for path in (
            paths.root,
            paths.inputs,
            paths.control,
            paths.release_directory,
            paths.work,
            paths.runtime,
        ):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(path, 0o700)
        action_pack = build_public_synthetic_action_pack_v1(config.study_id)
        action_raw = _canonical_bytes(action_pack)
        action_file_sha256 = _publish_once(paths.action_pack, action_raw)
        sandbox_receipt = _sandbox_receipt(config.study_id)
        sandbox_file_sha256 = _publish_once(
            paths.sandbox_receipt, _canonical_bytes(sandbox_receipt)
        )
        _publish_once(
            paths.forbidden_label_canary,
            b"public synthetic denied-path canary; contains no labels\n",
        )
        implementation_closure = dict(worker._implementation_closure())  # noqa: SLF001
        implementation_hash = implementation_closure["self_sha256"]
        if not isinstance(implementation_hash, str) or _SHA256.fullmatch(
            implementation_hash
        ) is None:
            raise ScarCssmQualificationError(
                "QUALIFICATION_IMPLEMENTATION_CLOSURE_INVALID"
            )

        processes: dict[int, ProcessHandle] = {}
        for shard in range(SHARD_COUNT):
            stdout = _open_log(
                paths.control / f"shard{shard}.stdout.private.log"
            )
            stderr = _open_log(
                paths.control / f"shard{shard}.stderr.private.log"
            )
            try:
                processes[shard] = dependencies.popen_factory(
                    _worker_argv(
                        config,
                        paths,
                        shard=shard,
                        action_file_sha256=action_file_sha256,
                        action_commitment_sha256=action_pack[
                            "action_commitment_sha256"
                        ],
                        implementation_closure_sha256=implementation_hash,
                        sandbox_file_sha256=sandbox_file_sha256,
                    ),
                    cwd=config.project_root,
                    env=_child_environment(config, paths, shard),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    close_fds=True,
                    preexec_fn=lambda shard=shard: _apply_child_landlock(
                        config, paths, shard
                    ),
                )
            finally:
                os.close(stdout)
                os.close(stderr)
        if set(processes) != set(range(SHARD_COUNT)):
            raise ScarCssmQualificationError(
                "QUALIFICATION_SHARD_SUBMISSION_INCOMPLETE"
            )
        runtimes, runtime_file_hashes = _wait_for_runtime_barrier(
            processes=processes,
            config=config,
            paths=paths,
            dependencies=dependencies,
            implementation_closure_sha256=implementation_hash,
            sandbox_receipt=sandbox_receipt,
            sandbox_file_sha256=sandbox_file_sha256,
        )
        release = _action_release(
            config=config,
            action_file_sha256=action_file_sha256,
            action_commitment_sha256=action_pack[
                "action_commitment_sha256"
            ],
            runtime_file_hashes=runtime_file_hashes,
        )
        release_file_sha256 = _publish_once(
            paths.action_release, _canonical_bytes(release)
        )
        statuses = {shard: process.wait() for shard, process in processes.items()}
        if statuses != {0: 0, 1: 0}:
            raise ScarCssmQualificationError("QUALIFICATION_ACTION_SHARD_FAILED")
        barrier = _verify_records_barrier(
            config=config,
            paths=paths,
            action_pack=action_pack,
            action_file_sha256=action_file_sha256,
            release=release,
            release_file_sha256=release_file_sha256,
            runtimes=runtimes,
            runtime_file_hashes=runtime_file_hashes,
        )
        body = {
            "action_pack_file_sha256": action_file_sha256,
            "action_pack_self_sha256": action_pack["self_sha256"],
            "cross_shard_release_file_sha256": release_file_sha256,
            "cross_shard_release_self_sha256": release["self_sha256"],
            "effect_study_attempt_count": 0,
            "executable_synthetic_item_count": len(EXECUTABLE_ITEM_ORDINALS),
            "formal_label_pack_access_count": 0,
            "formal_scar_source_access_count": 0,
            "formal_scorer_access_count": 0,
            "gpu_uuid_by_shard": {
                str(shard): config.gpu_uuids[shard]
                for shard in range(SHARD_COUNT)
            },
            "hmac_secret_access_count": 0,
            "implementation_closure_sha256": implementation_hash,
            "network_family_negative_canary_passed_by_shard": {
                str(shard): True for shard in range(SHARD_COUNT)
            },
            "forbidden_label_negative_canary_passed_by_shard": {
                str(shard): True for shard in range(SHARD_COUNT)
            },
            "premodel_synthetic_failure_count": (
                EXPECTED_PREMODEL_FAILURE_COUNT
            ),
            "qualification_only": True,
            "records_barrier": barrier,
            "runtime_receipt_file_sha256_by_shard": {
                str(shard): runtime_file_hashes[shard]
                for shard in range(SHARD_COUNT)
            },
            "schema": SAFE_RECEIPT_SCHEMA,
            "shard_count": SHARD_COUNT,
            "status": "QUALIFIED_SOURCE_FREE_EXACT_RUNTIME",
            "study_id": config.study_id,
            "synthetic_action_item_count": SYNTHETIC_ITEM_COUNT,
            "worker_cli_module": (
                "replication_runtime.gscl_scar_cssm_v1.worker"
            ),
            "worker_version": worker.VERSION,
        }
        _safe_value(body)
        receipt = _self_hashed(body)
        _publish_once(paths.safe_receipt, _canonical_bytes(receipt))
        return receipt
    finally:
        _release_lock(lock)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source-free SCAR CSSM exact runtime qualification"
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--qwen-model-root", required=True, type=Path)
    parser.add_argument("--qwen-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model-root", required=True, type=Path)
    parser.add_argument("--minilm-manifest", required=True, type=Path)
    parser.add_argument("--nvidia-smi", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True, action="append")
    parser.add_argument("--execution-freeze-sha256", required=True)
    parser.add_argument("--lock-path", required=True, type=Path)
    parser.add_argument("--minimum-gpu-free-mib", type=int, default=1)
    parser.add_argument("--minimum-host-available-bytes", type=int, default=1)
    parser.add_argument("--runtime-barrier-timeout-seconds", type=int, default=3_600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if len(arguments.gpu_uuid) != SHARD_COUNT:
        raise ScarCssmQualificationError("QUALIFICATION_CONFIG_INVALID")
    config = QualificationConfig(
        study_id=arguments.study_id,
        root=arguments.root,
        project_root=arguments.project_root,
        python_executable=arguments.python_executable,
        qwen_model_root=arguments.qwen_model_root,
        qwen_manifest_path=arguments.qwen_manifest,
        minilm_model_root=arguments.minilm_model_root,
        minilm_manifest_path=arguments.minilm_manifest,
        nvidia_smi_path=arguments.nvidia_smi,
        gpu_uuids=(arguments.gpu_uuid[0], arguments.gpu_uuid[1]),
        execution_freeze_sha256=arguments.execution_freeze_sha256,
        lock_path=arguments.lock_path,
        minimum_gpu_free_mib=arguments.minimum_gpu_free_mib,
        minimum_host_available_bytes=arguments.minimum_host_available_bytes,
        runtime_barrier_timeout_seconds=(
            arguments.runtime_barrier_timeout_seconds
        ),
    )
    result = run_qualification_once(config)
    print(_canonical_bytes(result).decode("ascii"))
    return 0 if result["status"] == "QUALIFIED_SOURCE_FREE_EXACT_RUNTIME" else 75


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AdmissionDecision",
    "QualificationConfig",
    "QualificationDependencies",
    "SAFE_RECEIPT_SCHEMA",
    "ScarCssmQualificationError",
    "VERSION",
    "build_public_synthetic_action_pack_v1",
    "run_qualification_once",
]
