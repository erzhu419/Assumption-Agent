"""One-shot formal controller for the SCAR CSSM intrinsic measurement.

The controller has a deliberately narrow lifecycle:

* shared-node admission happens before an attempt or formal input is read;
* two label-blind action shards are submitted before either is waited on;
* the complete action closure is durably sealed before label/secret access;
* the frozen offline scorer is invoked exactly once, in memory, afterwards;
* every mutable formal artifact lives below one verified ext4 root; and
* an attempt sentinel makes every post-attempt outcome terminal.

The action worker and scorer are separate frozen components.  In particular,
this module does not import the scorer until after the durable action barrier.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import unicodedata


VERSION = "gscl_scar_cssm_controller_v1"
CONFIG_SCHEMA = f"{VERSION}.config.v1"
ATTEMPT_SCHEMA = f"{VERSION}.attempt.v1"
INTENT_SCHEMA = f"{VERSION}.two_shard_intent.v1"
BARRIER_SCHEMA = f"{VERSION}.action_barrier.safe.v1"
PRIVATE_TERMINAL_SCHEMA = f"{VERSION}.private_terminal.v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}.safe_terminal.v1"
SHARD_TERMINAL_SCHEMA = (
    "gscl_scar_cssm_worker_v1.shard.safe_terminal.v1"
)
PREDICTION_PACK_SCHEMA = (
    "gscl_scar_cssm_score_v1.prediction_pack.v1"
)
SHARD_COUNT = 2
OFFICIAL_ACTION_ITEM_COUNT = 391
OFFICIAL_PRIMARY_ITEM_COUNT = 362
OFFICIAL_AMBIGUOUS_ITEM_COUNT = 29
EX_TEMPFAIL = 75
EX_SOFTWARE = 70

ARM_IDS = (
    "semantic_only",
    "flat_structural",
    "full_no_composition",
    "full_with_length2_composition",
    "full_with_length2_composition_target_color_shuffle",
)
VARIANT_NAMES = ("base", "system_swap")

MAX_ACTION_PACK_BYTES = 16 * 1024 * 1024
MAX_LABEL_PACK_BYTES = 16 * 1024 * 1024
MAX_SHARD_ITEMS_BYTES = 512 * 1024 * 1024
# Keep the controller's reader limit identical to the frozen worker writer
# limit.  A smaller aggregate-side limit would turn an otherwise valid shard
# into a post-action infrastructure failure.
MAX_ITEM_LINE_BYTES = 16 * 1024 * 1024
MAX_SCORE_RESULT_BYTES = 512 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STUDY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ITEM_TOKEN = re.compile(r"scar-item-v1-[0-9a-f]{64}\Z")

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

_BINDING_KEYS = frozenset(
    {
        "action_implementation_file_sha256",
        "action_pack_commitment_sha256",
        "action_pack_file_sha256",
        "controller_file_sha256",
        "encoder_binding_sha256",
        "execution_freeze_sha256",
        "implementation_closure_sha256",
        "label_pack_file_sha256",
        "minilm_manifest_file_sha256",
        "python_executable_file_sha256",
        "qwen_canary_self_sha256",
        "qwen_manifest_file_sha256",
        "qwen_runtime_commitment",
        "scorer_implementation_file_sha256",
        "sandbox_receipt_file_sha256",
        "sandbox_receipt_self_sha256",
        "secret_file_sha256",
        "source_implementation_file_sha256",
        "worker_file_sha256",
    }
)


class ScarCssmControllerError(RuntimeError):
    """Stable fail-closed controller error."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


class ScarCssmControllerAlreadyConsumed(ScarCssmControllerError):
    """The one-shot mutable root already contains attempt evidence."""


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self) -> int: ...


@dataclass(frozen=True, slots=True)
class MeasurementPolicy:
    action_item_count: int = OFFICIAL_ACTION_ITEM_COUNT
    primary_item_count: int = OFFICIAL_PRIMARY_ITEM_COUNT
    ambiguous_item_count: int = OFFICIAL_AMBIGUOUS_ITEM_COUNT

    def validate(self) -> None:
        values = (
            self.action_item_count,
            self.primary_item_count,
            self.ambiguous_item_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ScarCssmControllerError("CONTROLLER_POLICY_INVALID")
        if self.primary_item_count + self.ambiguous_item_count != self.action_item_count:
            raise ScarCssmControllerError("CONTROLLER_POLICY_INVALID")


OFFICIAL_POLICY = MeasurementPolicy()


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    status: str
    reason_codes: tuple[str, ...]
    host_mem_available_bytes: int | None = None
    selected_gpu_free_mib: tuple[int, ...] = ()

    def safe_payload(self) -> dict[str, Any]:
        return {
            "host_mem_available_bytes": self.host_mem_available_bytes,
            "reason_codes": list(self.reason_codes),
            "selected_gpu_free_mib": list(self.selected_gpu_free_mib),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class FormalConfig:
    study_id: str
    mutable_root: Path
    project_root: Path
    action_pack_path: Path
    label_pack_path: Path
    secret_path: Path
    python_executable: Path
    qwen_model_root: Path
    qwen_manifest_path: Path
    minilm_model_root: Path
    minilm_manifest_path: Path
    sandbox_receipt_path: Path
    nvidia_smi_path: Path
    gpu_uuids: tuple[str, str]
    lock_path: Path
    minimum_gpu_free_mib: int
    minimum_host_available_bytes: int
    bindings: Mapping[str, str]
    self_sha256: str

    def body(self) -> dict[str, Any]:
        return {
            "action_pack_path": str(self.action_pack_path),
            "bindings": dict(sorted(self.bindings.items())),
            "gpu_uuids": list(self.gpu_uuids),
            "label_pack_path": str(self.label_pack_path),
            "lock_path": str(self.lock_path),
            "minilm_manifest_path": str(self.minilm_manifest_path),
            "minilm_model_root": str(self.minilm_model_root),
            "minimum_gpu_free_mib": self.minimum_gpu_free_mib,
            "minimum_host_available_bytes": self.minimum_host_available_bytes,
            "mutable_root": str(self.mutable_root),
            "nvidia_smi_path": str(self.nvidia_smi_path),
            "project_root": str(self.project_root),
            "python_executable": str(self.python_executable),
            "qwen_manifest_path": str(self.qwen_manifest_path),
            "qwen_model_root": str(self.qwen_model_root),
            "sandbox_receipt_path": str(self.sandbox_receipt_path),
            "schema": CONFIG_SCHEMA,
            "secret_path": str(self.secret_path),
            "study_id": self.study_id,
        }


@dataclass(frozen=True, slots=True)
class ControllerPaths:
    root: Path
    control: Path
    work: Path
    runtime: Path
    attempt: Path
    intent: Path
    action_release: Path
    barrier: Path
    prediction: Path
    private_result: Path
    safe_aggregate: Path
    private_terminal: Path
    safe_terminal: Path

    @classmethod
    def for_root(cls, root: Path) -> "ControllerPaths":
        control = root / "control"
        return cls(
            root=root,
            control=control,
            work=root / "work",
            runtime=root / "runtime",
            attempt=control / "formal_attempt.sentinel",
            intent=control / "two_shard_intent.safe.json",
            action_release=control / "two_shard_action_release.safe.json",
            barrier=control / "action_barrier.safe.json",
            prediction=control / "prediction_pack.private.json",
            private_result=control / "score.private.json",
            safe_aggregate=control / "score.safe.json",
            private_terminal=control / "formal_terminal.private.json",
            safe_terminal=control / "formal_terminal.safe.json",
        )


@dataclass(frozen=True, slots=True)
class ControllerDependencies:
    filesystem_type: Callable[[Path], str]
    resource_probe: Callable[[FormalConfig], AdmissionDecision]
    popen_factory: Callable[..., ProcessHandle]
    validate_action_pack: Callable[[Mapping[str, Any], str], None]
    score_once: Callable[..., Any]


@dataclass(slots=True)
class _RunState:
    stage: str = "pre_attempt"
    action_child_launch_count: int = 0
    action_release_count: int = 0
    action_barrier_count: int = 0
    label_pack_access_count: int = 0
    secret_access_count: int = 0
    offline_scorer_call_count: int = 0


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
        raise ScarCssmControllerError("CONTROLLER_CANONICAL_JSON_INVALID") from exc


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "self_sha256": _content_hash(value)}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ScarCssmControllerError(f"{field}_INVALID") from exc
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise ScarCssmControllerError(f"{field}_NOT_CANONICAL")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, value: Mapping[str, Any] | bytes) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ScarCssmControllerError("CONTROLLER_OUTPUT_ALREADY_EXISTS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(raw).hexdigest()


def _read_regular_once(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
    expected_sha256: str | None,
    required_mode: int | None,
) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ScarCssmControllerError(f"{field}_UNAVAILABLE") from exc
    if (
        not path.is_absolute()
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > maximum_bytes
        or before.st_uid != os.getuid()
        or (required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode)
    ):
        raise ScarCssmControllerError(f"{field}_METADATA_INVALID")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if (
                (opened.st_dev, opened.st_ino, opened.st_size)
                != (before.st_dev, before.st_ino, before.st_size)
            ):
                raise ScarCssmControllerError(f"{field}_CHANGED")
            raw = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ScarCssmControllerError(f"{field}_UNAVAILABLE") from exc
    if (
        len(raw) != before.st_size
        or len(raw) > maximum_bytes
        or (after.st_dev, after.st_ino, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (
            expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256
        )
    ):
        raise ScarCssmControllerError(f"{field}_BINDING_INVALID")
    return raw


def _load_canonical_json_file(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
    expected_sha256: str,
    required_mode: int | None = 0o600,
) -> dict[str, Any]:
    return _strict_json(
        _read_regular_once(
            path,
            field=field,
            maximum_bytes=maximum_bytes,
            expected_sha256=expected_sha256,
            required_mode=required_mode,
        ),
        field=field,
    )


def _validate_config(config: FormalConfig) -> None:
    paths = (
        config.mutable_root,
        config.project_root,
        config.action_pack_path,
        config.label_pack_path,
        config.secret_path,
        config.python_executable,
        config.qwen_model_root,
        config.qwen_manifest_path,
        config.minilm_model_root,
        config.minilm_manifest_path,
        config.sandbox_receipt_path,
        config.nvidia_smi_path,
        config.lock_path,
    )
    if (
        not isinstance(config.study_id, str)
        or _STUDY_ID.fullmatch(config.study_id) is None
        or any(not isinstance(path, Path) or not path.is_absolute() for path in paths)
        or config.mutable_root == Path(config.mutable_root.anchor)
        or len(set(config.gpu_uuids)) != SHARD_COUNT
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", value) is None
            for value in config.gpu_uuids
        )
        or type(config.minimum_gpu_free_mib) is not int
        or config.minimum_gpu_free_mib < 0
        or type(config.minimum_host_available_bytes) is not int
        or config.minimum_host_available_bytes < 0
        or set(config.bindings) != _BINDING_KEYS
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in config.bindings.values()
        )
        or _SHA256.fullmatch(config.self_sha256) is None
        or _content_hash(config.body()) != config.self_sha256
    ):
        raise ScarCssmControllerError("CONTROLLER_CONFIG_INVALID")


def load_config(path: Path) -> FormalConfig:
    raw = _read_regular_once(
        path,
        field="CONTROLLER_CONFIG",
        maximum_bytes=1 << 20,
        expected_sha256=None,
        required_mode=0o600,
    )
    value = _strict_json(raw, field="CONTROLLER_CONFIG")
    expected_keys = {
        "action_pack_path",
        "bindings",
        "gpu_uuids",
        "label_pack_path",
        "lock_path",
        "minilm_manifest_path",
        "minilm_model_root",
        "minimum_gpu_free_mib",
        "minimum_host_available_bytes",
        "mutable_root",
        "nvidia_smi_path",
        "project_root",
        "python_executable",
        "qwen_manifest_path",
        "qwen_model_root",
        "sandbox_receipt_path",
        "schema",
        "secret_path",
        "self_sha256",
        "study_id",
    }
    if set(value) != expected_keys or value.get("schema") != CONFIG_SCHEMA:
        raise ScarCssmControllerError("CONTROLLER_CONFIG_INVALID")
    gpu_uuids = value["gpu_uuids"]
    if type(gpu_uuids) is not list or len(gpu_uuids) != SHARD_COUNT:
        raise ScarCssmControllerError("CONTROLLER_CONFIG_INVALID")
    if type(value["bindings"]) is not dict:
        raise ScarCssmControllerError("CONTROLLER_CONFIG_INVALID")
    config = FormalConfig(
        study_id=value["study_id"],
        mutable_root=Path(value["mutable_root"]),
        project_root=Path(value["project_root"]),
        action_pack_path=Path(value["action_pack_path"]),
        label_pack_path=Path(value["label_pack_path"]),
        secret_path=Path(value["secret_path"]),
        python_executable=Path(value["python_executable"]),
        qwen_model_root=Path(value["qwen_model_root"]),
        qwen_manifest_path=Path(value["qwen_manifest_path"]),
        minilm_model_root=Path(value["minilm_model_root"]),
        minilm_manifest_path=Path(value["minilm_manifest_path"]),
        sandbox_receipt_path=Path(value["sandbox_receipt_path"]),
        nvidia_smi_path=Path(value["nvidia_smi_path"]),
        gpu_uuids=(str(gpu_uuids[0]), str(gpu_uuids[1])),
        lock_path=Path(value["lock_path"]),
        minimum_gpu_free_mib=value["minimum_gpu_free_mib"],
        minimum_host_available_bytes=value["minimum_host_available_bytes"],
        bindings=dict(value["bindings"]),
        self_sha256=value["self_sha256"],
    )
    _validate_config(config)
    return config


def _mount_fstype(path: Path) -> str:
    candidate = path
    while not candidate.exists():
        if candidate == candidate.parent:
            raise ScarCssmControllerError("CONTROLLER_MOUNT_UNAVAILABLE")
        candidate = candidate.parent
    resolved = candidate.resolve(strict=True)
    best: tuple[int, str] | None = None
    try:
        rows = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ScarCssmControllerError("CONTROLLER_MOUNT_UNAVAILABLE") from exc
    for row in rows:
        left, separator, right = row.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        trailing = right.split()
        if len(fields) < 5 or not trailing:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        choice = (len(mount.parts), trailing[0])
        if best is None or choice[0] > best[0]:
            best = choice
    if best is None:
        raise ScarCssmControllerError("CONTROLLER_MOUNT_UNAVAILABLE")
    return best[1]


def _verify_all_runtime_paths_ext4(
    config: FormalConfig, filesystem_type: Callable[[Path], str]
) -> None:
    required = (
        config.mutable_root,
        config.project_root,
        config.action_pack_path,
        config.label_pack_path,
        config.secret_path,
        config.python_executable,
        config.qwen_model_root,
        config.qwen_manifest_path,
        config.minilm_model_root,
        config.minilm_manifest_path,
        config.sandbox_receipt_path,
    )
    if any(filesystem_type(path) != "ext4" for path in required):
        raise ScarCssmControllerError("CONTROLLER_RUNTIME_PATH_NOT_EXT4")


def _parse_mem_available() -> int:
    try:
        rows = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ScarCssmControllerError("CONTROLLER_RESOURCE_PROBE_FAILED") from exc
    for row in rows:
        if row.startswith("MemAvailable:"):
            fields = row.split()
            if len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit():
                return int(fields[1]) * 1024
    raise ScarCssmControllerError("CONTROLLER_RESOURCE_PROBE_FAILED")


def _run_nvidia_smi(path: Path, arguments: Sequence[str]) -> list[str]:
    try:
        completed = subprocess.run(
            [str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScarCssmControllerError("CONTROLLER_RESOURCE_PROBE_FAILED") from exc
    return [row.strip() for row in completed.stdout.splitlines() if row.strip()]


def _resource_probe(config: FormalConfig) -> AdmissionDecision:
    host_available = _parse_mem_available()
    gpu_rows = _run_nvidia_smi(
        config.nvidia_smi_path,
        ("--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"),
    )
    registry: dict[str, tuple[str, int]] = {}
    for row in gpu_rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            raise ScarCssmControllerError("CONTROLLER_RESOURCE_PROBE_FAILED")
        registry[fields[1]] = (fields[0], int(fields[2]))
    if any(gpu_uuid not in registry for gpu_uuid in config.gpu_uuids):
        raise ScarCssmControllerError("CONTROLLER_RESOURCE_PROBE_FAILED")
    process_uuids = set(
        _run_nvidia_smi(
            config.nvidia_smi_path,
            ("--query-compute-apps=gpu_uuid", "--format=csv,noheader"),
        )
    )
    selected = tuple(registry[gpu_uuid] for gpu_uuid in config.gpu_uuids)
    reasons: list[str] = []
    if any(gpu_uuid in process_uuids for gpu_uuid in config.gpu_uuids):
        reasons.append("SELECTED_GPU_HAS_COMPUTE_PROCESS")
    if any(free < config.minimum_gpu_free_mib for _index, free in selected):
        reasons.append("SELECTED_GPU_FREE_MEMORY_BELOW_POLICY")
    if host_available < config.minimum_host_available_bytes:
        reasons.append("HOST_MEMORY_BELOW_POLICY")
    return AdmissionDecision(
        status="DEFERRED_SHARED_RESOURCE" if reasons else "ADMITTED_SHARED_RESOURCE",
        reason_codes=tuple(reasons),
        host_mem_available_bytes=host_available,
        selected_gpu_free_mib=tuple(free for _index, free in selected),
    )


def _validate_action_pack_production(value: Mapping[str, Any], study_id: str) -> None:
    from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as source

    source.validate_scar_cssm_action_pack_v1(value, study_id=study_id)


def _score_once_production(
    action_pack: Mapping[str, Any],
    label_pack: Mapping[str, Any],
    prediction_pack: Mapping[str, Any],
    *,
    secret: bytes,
    study_id: str,
) -> Any:
    # Importing the scorer is itself delayed until the caller has published
    # the action barrier and opened the late capabilities.
    from assumption_agent.benchmarks import gscl_scar_cssm_score_v1 as scorer

    return scorer.score_scar_cssm_predictions_v1(
        action_pack,
        label_pack,
        prediction_pack,
        secret=secret,
        study_id=study_id,
    )


DEFAULT_DEPENDENCIES = ControllerDependencies(
    filesystem_type=_mount_fstype,
    resource_probe=_resource_probe,
    popen_factory=subprocess.Popen,
    validate_action_pack=_validate_action_pack_production,
    score_once=_score_once_production,
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
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _release_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _implementation_paths(config: FormalConfig) -> dict[str, Path]:
    return {
        "action_implementation_file_sha256": (
            config.project_root / "assumption_agent/benchmarks/gscl_scar_cssm_action_v1.py"
        ),
        "controller_file_sha256": Path(__file__).resolve(strict=True),
        "scorer_implementation_file_sha256": (
            config.project_root / "assumption_agent/benchmarks/gscl_scar_cssm_score_v1.py"
        ),
        "source_implementation_file_sha256": (
            config.project_root / "assumption_agent/benchmarks/gscl_scar_cssm_source_v1.py"
        ),
        "worker_file_sha256": (
            config.project_root / "replication_runtime/gscl_scar_cssm_v1/worker.py"
        ),
        "python_executable_file_sha256": config.python_executable.resolve(strict=True),
        "qwen_manifest_file_sha256": config.qwen_manifest_path,
        "minilm_manifest_file_sha256": config.minilm_manifest_path,
        "sandbox_receipt_file_sha256": config.sandbox_receipt_path,
    }


def _verify_implementation_bindings(config: FormalConfig) -> None:
    for key, path in _implementation_paths(config).items():
        try:
            metadata = path.lstat()
            raw = path.read_bytes()
        except OSError as exc:
            raise ScarCssmControllerError("CONTROLLER_IMPLEMENTATION_UNAVAILABLE") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ScarCssmControllerError("CONTROLLER_IMPLEMENTATION_INVALID")
        if hashlib.sha256(raw).hexdigest() != config.bindings[key]:
            raise ScarCssmControllerError("CONTROLLER_IMPLEMENTATION_BINDING_DRIFT")


def _load_sandbox_receipt(config: FormalConfig) -> dict[str, Any]:
    value = _load_canonical_json_file(
        config.sandbox_receipt_path,
        field="CONTROLLER_SANDBOX_RECEIPT",
        maximum_bytes=1 << 20,
        expected_sha256=config.bindings["sandbox_receipt_file_sha256"],
    )
    claimed = value.get("self_sha256")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if (
        value.get("schema") != "gscl_scar_cssm_sandbox_freeze_v1"
        or value.get("study_id") != config.study_id
        or value.get("status") != "frozen"
        or value.get("ip_address_deny") != "any"
        or value.get("restrict_address_families") != "AF_UNIX"
        or value.get("action_label_path_denied") is not True
        or value.get("action_external_network_denied") is not True
        or claimed != _content_hash(body)
        or claimed != config.bindings["sandbox_receipt_self_sha256"]
    ):
        raise ScarCssmControllerError("CONTROLLER_SANDBOX_RECEIPT_INVALID")
    return value


def _validate_action_pack_shape(
    action_pack: Mapping[str, Any], *, config: FormalConfig, policy: MeasurementPolicy
) -> tuple[list[Mapping[str, Any]], tuple[str, ...]]:
    if (
        type(action_pack) is not dict
        or action_pack.get("study_id") != config.study_id
        or not isinstance(action_pack.get("action_commitment_sha256"), str)
        or _SHA256.fullmatch(action_pack["action_commitment_sha256"]) is None
        or action_pack["action_commitment_sha256"]
        != config.bindings["action_pack_commitment_sha256"]
        or action_pack.get("variant_names") != list(VARIANT_NAMES)
        or type(action_pack.get("items")) is not list
        or len(action_pack["items"]) != policy.action_item_count
    ):
        raise ScarCssmControllerError("CONTROLLER_ACTION_PACK_SHAPE_INVALID")
    tokens: list[str] = []
    for item in action_pack["items"]:
        if (
            type(item) is not dict
            or set(item) != {"item_token", "variants"}
            or not isinstance(item.get("item_token"), str)
            or _ITEM_TOKEN.fullmatch(item["item_token"]) is None
            or type(item.get("variants")) is not dict
            or tuple(item["variants"]) != VARIANT_NAMES
        ):
            raise ScarCssmControllerError("CONTROLLER_ACTION_ITEM_INVALID")
        tokens.append(item["item_token"])
    if len(set(tokens)) != len(tokens):
        raise ScarCssmControllerError("CONTROLLER_ACTION_ITEM_DUPLICATE")
    return action_pack["items"], tuple(tokens)


def _collision_tokens(items: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    collisions: set[str] = set()
    for item in items:
        try:
            base = item["variants"]["base"]
            sides = (base["left"], base["right"])
            normalized_sides = [
                [
                    unicodedata.normalize("NFKC", slot["surface"]).casefold()
                    for slot in side["slots"]
                ]
                for side in sides
            ]
        except (KeyError, TypeError) as exc:
            raise ScarCssmControllerError("CONTROLLER_ACTION_ITEM_INVALID") from exc
        if any(len(values) != len(set(values)) for values in normalized_sides):
            collisions.add(item["item_token"])
    return frozenset(collisions)


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
        raise ScarCssmControllerError("CONTROLLER_LANDLOCK_UNAVAILABLE")
    return result


def _landlock_rights(abi: int) -> int:
    rights = _LL_BASE
    if abi >= 2:
        rights |= _LL_REFER
    if abi >= 3:
        rights |= _LL_TRUNCATE
    return rights


def _apply_landlock(
    *,
    read_paths: Sequence[Path],
    write_paths: Sequence[Path],
    device_paths: Sequence[Path],
) -> None:
    abi = _landlock_abi_version()
    handled = _landlock_rights(abi)
    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attribute = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(ruleset_attribute),
            ctypes.sizeof(ruleset_attribute),
            ctypes.c_uint(0),
        )
    )
    if ruleset_fd < 0:
        raise ScarCssmControllerError("CONTROLLER_LANDLOCK_CREATE_FAILED")

    def add(path: Path, rights: int) -> None:
        flags = os.O_PATH | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ScarCssmControllerError(
                "CONTROLLER_LANDLOCK_ALLOWLIST_UNAVAILABLE"
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
            attribute = _PathBeneathAttr(
                allowed_access=allowed,
                parent_fd=descriptor,
                reserved=0,
            )
            result = int(
                libc.syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(attribute),
                    ctypes.c_uint(0),
                )
            )
            if result != 0:
                raise ScarCssmControllerError("CONTROLLER_LANDLOCK_RULE_FAILED")
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
            raise ScarCssmControllerError("CONTROLLER_NO_NEW_PRIVS_FAILED")
        if (
            libc.syscall(
                _SYS_LANDLOCK_RESTRICT_SELF,
                ruleset_fd,
                ctypes.c_uint(0),
            )
            != 0
        ):
            raise ScarCssmControllerError("CONTROLLER_LANDLOCK_RESTRICT_FAILED")
    finally:
        os.close(ruleset_fd)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(directory.resolve(strict=True))
    except ValueError:
        return False
    return True


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
        raise ScarCssmControllerError(
            "CONTROLLER_PYTHON_RUNTIME_TOPOLOGY_INVALID"
        ) from exc
    return tuple(dict.fromkeys(roots))


def _action_landlock_paths(
    config: FormalConfig, paths: ControllerPaths, shard: int
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    system_candidates = (
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
    directory_reads = (
        config.project_root,
        config.qwen_model_root,
        config.minilm_model_root,
        *python_runtime_roots,
        paths.control,
    )
    for forbidden in (config.label_pack_path, config.secret_path):
        if any(_is_within(forbidden, directory) for directory in directory_reads):
            raise ScarCssmControllerError(
                "CONTROLLER_ACTION_LABEL_SANDBOX_TOPOLOGY_INVALID"
            )
    read_paths = tuple(
        path
        for path in (
            *system_candidates,
            *directory_reads,
            config.action_pack_path,
            config.qwen_manifest_path,
            config.minilm_manifest_path,
            config.sandbox_receipt_path,
        )
        if path.exists()
    )
    write_paths = (paths.work, paths.runtime / f"shard{shard}")
    devices = tuple(
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
    return read_paths, write_paths, devices


def _apply_action_landlock(
    config: FormalConfig, paths: ControllerPaths, shard: int
) -> None:
    read_paths, write_paths, device_paths = _action_landlock_paths(
        config, paths, shard
    )
    _apply_landlock(
        read_paths=read_paths,
        write_paths=write_paths,
        device_paths=device_paths,
    )


def _child_environment(config: FormalConfig, paths: ControllerPaths, shard: int) -> dict[str, str]:
    private_home = paths.runtime / f"shard{shard}" / "home"
    temporary = paths.runtime / f"shard{shard}" / "tmp"
    cache = paths.runtime / f"shard{shard}" / "cache"
    for path in (private_home, temporary, cache):
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": config.gpu_uuids[shard],
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(cache / "huggingface"),
        "HOME": str(private_home),
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


def _worker_argv(config: FormalConfig, paths: ControllerPaths, shard: int) -> list[str]:
    return [
        str(config.python_executable),
        "-m",
        "replication_runtime.gscl_scar_cssm_v1.worker",
        "--action-pack",
        str(config.action_pack_path),
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
        str(config.sandbox_receipt_path),
        "--action-release",
        str(paths.action_release),
        "--forbidden-label-probe",
        str(config.label_pack_path),
        "--expected-action-file-sha256",
        config.bindings["action_pack_file_sha256"],
        "--expected-action-commitment-sha256",
        config.bindings["action_pack_commitment_sha256"],
        "--expected-implementation-closure-sha256",
        config.bindings["implementation_closure_sha256"],
        "--expected-sandbox-receipt-sha256",
        config.bindings["sandbox_receipt_file_sha256"],
        "--expected-execution-freeze-sha256",
        config.bindings["execution_freeze_sha256"],
        "--expected-gpu-uuid",
        config.gpu_uuids[shard],
        "--expected-peer-gpu-uuid",
        config.gpu_uuids[1 - shard],
    ]


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    return descriptor


def _launch_shards(
    config: FormalConfig,
    paths: ControllerPaths,
    *,
    popen_factory: Callable[..., ProcessHandle],
    state: _RunState,
) -> dict[int, ProcessHandle]:
    processes: dict[int, ProcessHandle] = {}
    launch_error: Exception | None = None
    for shard in range(SHARD_COUNT):
        stdout = _open_log(paths.control / f"shard{shard}.stdout.private.log")
        stderr = _open_log(paths.control / f"shard{shard}.stderr.private.log")
        try:
            processes[shard] = popen_factory(
                _worker_argv(config, paths, shard),
                cwd=config.project_root,
                env=_child_environment(config, paths, shard),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                preexec_fn=lambda shard=shard: _apply_action_landlock(
                    config, paths, shard
                ),
            )
            state.action_child_launch_count += 1
        except Exception as exc:
            launch_error = exc
            break
        finally:
            os.close(stdout)
            os.close(stderr)
    if launch_error is not None or set(processes) != set(range(SHARD_COUNT)):
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait()
        raise ScarCssmControllerError("CONTROLLER_SHARD_SUBMISSION_INCOMPLETE") from launch_error
    return processes


def _load_worker_terminal(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_regular_once(
        path,
        field="CONTROLLER_SHARD_TERMINAL",
        maximum_bytes=1 << 20,
        expected_sha256=None,
        required_mode=0o600,
    )
    value = _strict_json(raw, field="CONTROLLER_SHARD_TERMINAL")
    claimed = value.get("self_sha256")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if not isinstance(claimed, str) or _content_hash(body) != claimed:
        raise ScarCssmControllerError("CONTROLLER_SHARD_TERMINAL_SELF_INVALID")
    return value, hashlib.sha256(raw).hexdigest()


def _load_runtime_receipt(
    path: Path, *, config: FormalConfig, shard: int
) -> tuple[dict[str, Any], str]:
    raw = _read_regular_once(
        path,
        field="CONTROLLER_RUNTIME_RECEIPT",
        maximum_bytes=32 * 1024 * 1024,
        expected_sha256=None,
        required_mode=0o600,
    )
    value = _strict_json(raw, field="CONTROLLER_RUNTIME_RECEIPT")
    claimed = value.get("self_sha256")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
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
        set(value) != expected_keys
        or claimed != _content_hash(body)
        or value["schema"] != "gscl_scar_cssm_worker_v1.runtime.safe_receipt.v1"
        or value["status"] != "qualified_before_action_pack_open"
        or value["study_id"] != config.study_id
        or value["version"] != "gscl_scar_cssm_worker_v1"
        or value["shard_count"] != SHARD_COUNT
        or value["shard_index"] != shard
        or value["execution_freeze_sha256"]
        != config.bindings["execution_freeze_sha256"]
    ):
        raise ScarCssmControllerError("CONTROLLER_RUNTIME_RECEIPT_INVALID")
    gpu = value["gpu"]
    qwen = value["qwen"]
    minilm = value["minilm"]
    implementation = value["implementation_closure"]
    sandbox = value["sandbox_freeze"]
    sandbox_file = value["sandbox_freeze_file"]
    process_sandbox = value["process_sandbox"]
    network = value["network_negative_canary"]
    label_canary = value["forbidden_label_negative_canary"]
    execution = value["execution"]
    if (
        type(gpu) is not dict
        or gpu.get("physical_uuid") != config.gpu_uuids[shard]
        or gpu.get("cuda_visible_devices") != config.gpu_uuids[shard]
        or gpu.get("visible_device_count") != 1
        or gpu.get("logical_current_device") != 0
        or gpu.get("parameter_devices") != ["cuda:0"]
        or type(qwen) is not dict
        or qwen.get("runtime_commitment")
        != config.bindings["qwen_runtime_commitment"]
        or type(qwen.get("qualification_canary")) is not dict
        or qwen["qualification_canary"].get("self_sha256")
        != config.bindings["qwen_canary_self_sha256"]
        or type(minilm) is not dict
        or minilm.get("encoder_binding_sha256")
        != config.bindings["encoder_binding_sha256"]
        or type(implementation) is not dict
        or implementation.get("self_sha256")
        != config.bindings["implementation_closure_sha256"]
        or type(sandbox) is not dict
        or sandbox.get("self_sha256")
        != config.bindings["sandbox_receipt_self_sha256"]
        or type(sandbox_file) is not dict
        or sandbox_file.get("sha256")
        != config.bindings["sandbox_receipt_file_sha256"]
        or type(process_sandbox) is not dict
        or process_sandbox.get("no_new_privileges") is not True
        or type(network) is not dict
        or not all(
            type(network.get(family)) is dict
            and network[family].get("creation_denied") is True
            for family in ("AF_INET", "AF_INET6")
        )
        or network.get("external_connect_attempt_count") != 0
        or type(label_canary) is not dict
        or label_canary.get("open_denied") is not True
        or label_canary.get("read_count") != 0
        or type(execution) is not dict
        or execution.get("hf_hub_offline") != "1"
        or execution.get("transformers_offline") != "1"
        or execution.get("deterministic_algorithms") is not True
        or execution.get("matmul_tf32") is not False
        or execution.get("cudnn_tf32") is not False
        or execution.get("cudnn_benchmark") is not False
        or type(execution.get("python")) is not dict
        or execution["python"].get("executable_sha256")
        != config.bindings["python_executable_file_sha256"]
    ):
        raise ScarCssmControllerError("CONTROLLER_RUNTIME_BINDING_INVALID")
    return value, hashlib.sha256(raw).hexdigest()


def _wait_for_runtime_barrier(
    config: FormalConfig,
    paths: ControllerPaths,
    processes: Mapping[int, ProcessHandle],
    *,
    timeout_seconds: float = 3_600.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    deadline = time.monotonic() + timeout_seconds
    sentinel_paths = {
        shard: paths.work / f"shard{shard}.attempt.sentinel"
        for shard in range(SHARD_COUNT)
    }
    while not all(path.exists() and not path.is_symlink() for path in sentinel_paths.values()):
        exited: dict[int, int] = {}
        for shard, process in processes.items():
            returncode = process.poll()
            if returncode is not None:
                exited[shard] = returncode
        if exited:
            raise ScarCssmControllerError("CONTROLLER_SHARD_FAILED_BEFORE_RELEASE")
        if time.monotonic() >= deadline:
            raise ScarCssmControllerError("CONTROLLER_RUNTIME_BARRIER_TIMEOUT")
        time.sleep(0.25)

    runtime_receipts: dict[str, Any] = {}
    runtime_file_hashes: dict[str, str] = {}
    for shard in range(SHARD_COUNT):
        runtime_path = paths.work / f"shard{shard}.runtime.safe.json"
        runtime, runtime_file_sha256 = _load_runtime_receipt(
            runtime_path, config=config, shard=shard
        )
        sentinel_raw = _read_regular_once(
            sentinel_paths[shard],
            field="CONTROLLER_SHARD_SENTINEL",
            maximum_bytes=1 << 20,
            expected_sha256=None,
            required_mode=0o600,
        )
        sentinel = _strict_json(sentinel_raw, field="CONTROLLER_SHARD_SENTINEL")
        if sentinel != {
            "expected_action_commitment_sha256": config.bindings[
                "action_pack_commitment_sha256"
            ],
            "expected_action_file_sha256": config.bindings[
                "action_pack_file_sha256"
            ],
            "expected_execution_freeze_sha256": config.bindings[
                "execution_freeze_sha256"
            ],
            "runtime_receipt_sha256": runtime_file_sha256,
            "shard_count": SHARD_COUNT,
            "shard_index": shard,
            "study_id": config.study_id,
            "version": "gscl_scar_cssm_worker_v1",
        }:
            raise ScarCssmControllerError("CONTROLLER_SHARD_SENTINEL_INVALID")
        if processes[shard].poll() is not None:
            raise ScarCssmControllerError("CONTROLLER_SHARD_FAILED_BEFORE_RELEASE")
        runtime_receipts[str(shard)] = runtime
        runtime_file_hashes[str(shard)] = runtime_file_sha256
    if (
        len(set(config.gpu_uuids)) != SHARD_COUNT
        or {runtime_receipts[str(shard)]["gpu"]["physical_uuid"] for shard in range(SHARD_COUNT)}
        != set(config.gpu_uuids)
    ):
        raise ScarCssmControllerError("CONTROLLER_TWO_GPU_RUNTIME_INVALID")
    return runtime_receipts, runtime_file_hashes


def _publish_action_release(
    config: FormalConfig,
    paths: ControllerPaths,
    *,
    runtime_file_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    release = _self_hashed(
        {
            "action_commitment_sha256": config.bindings[
                "action_pack_commitment_sha256"
            ],
            "action_file_sha256": config.bindings["action_pack_file_sha256"],
            "execution_freeze_sha256": config.bindings["execution_freeze_sha256"],
            "gpu_uuid_by_shard": {
                str(shard): config.gpu_uuids[shard] for shard in range(SHARD_COUNT)
            },
            "runtime_receipt_file_sha256_by_shard": dict(runtime_file_hashes),
            "schema": "gscl_scar_cssm_worker_v1.two_shard_action_release.v1",
            "shard_count": SHARD_COUNT,
            "status": "release_both_shards_to_action_pack",
            "study_id": config.study_id,
        }
    )
    return release, _publish_once(paths.action_release, release)


def _load_shard_items(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    raw = _read_regular_once(
        path,
        field="CONTROLLER_SHARD_ITEMS",
        maximum_bytes=MAX_SHARD_ITEMS_BYTES,
        expected_sha256=expected_sha256,
        required_mode=0o600,
    )
    if raw and not raw.endswith(b"\n"):
        raise ScarCssmControllerError("CONTROLLER_SHARD_ITEMS_NOT_CANONICAL")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line or len(line) + 1 > MAX_ITEM_LINE_BYTES:
            raise ScarCssmControllerError("CONTROLLER_SHARD_ITEM_INVALID")
        rows.append(_strict_json(line, field="CONTROLLER_SHARD_ITEM"))
    return rows


def _validate_execution_row(row: Mapping[str, Any], *, collision: bool) -> tuple[int, int]:
    if type(row) is not dict or set(row) != {
        "diagnostics",
        "execution",
        "item_token",
        "private_mechanism_receipts",
        "proposal_pools",
        "variants",
    }:
        raise ScarCssmControllerError("CONTROLLER_ACTION_ROW_INVALID")
    execution = row["execution"]
    if type(execution) is not dict or set(execution) != {
        "document_call_count",
        "error_code",
        "structural_status",
    }:
        raise ScarCssmControllerError("CONTROLLER_ACTION_EXECUTION_INVALID")
    expected = (
        {
            "document_call_count": 0,
            "error_code": "SLOT_BINDER_TYPED_FAILURE",
            "structural_status": "TYPED_FAILURE",
        }
        if collision
        else {
            "document_call_count": 2,
            "error_code": None,
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
        }
    )
    if (
        execution != expected
        or type(execution["document_call_count"]) is not int
        or isinstance(execution["document_call_count"], bool)
    ):
        raise ScarCssmControllerError("CONTROLLER_UNEXPECTED_TYPED_OR_RUNTIME_FAILURE")
    return expected["document_call_count"], int(collision)


def _restore_scorer_wire_order(
    prediction: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild order-sensitive scorer mappings after canonical JSON transport.

    Worker records are canonical JSON, so every object is serialized with
    lexicographically sorted keys.  The frozen scorer deliberately requires
    semantic order for variant and arm mappings.  JSON decoding therefore
    cannot be passed through verbatim even when its values and key sets are
    otherwise exact.
    """

    variants = prediction.get("variants")
    pools = prediction.get("proposal_pools")
    diagnostics = prediction.get("diagnostics")
    if (
        type(variants) is not dict
        or set(variants) != set(VARIANT_NAMES)
        or type(pools) is not dict
        or set(pools) != set(VARIANT_NAMES)
        or type(diagnostics) is not dict
        or set(diagnostics) != set(VARIANT_NAMES)
    ):
        raise ScarCssmControllerError("CONTROLLER_SCORER_WIRE_ORDER_INVALID")

    ordered_variants: dict[str, Any] = {}
    ordered_pools: dict[str, Any] = {}
    ordered_diagnostics: dict[str, Any] = {}
    diagnostic_keys = {
        "arms",
        "left_binder",
        "left_graph_receipt_sha256",
        "mapping_receipt_sha256_by_arm",
        "right_binder",
        "right_graph_receipt_sha256",
        "structural_diagnostics_available",
        "target_color_shuffle_effective",
    }
    for variant_name in VARIANT_NAMES:
        variant = variants[variant_name]
        pool = pools[variant_name]
        diagnostic = diagnostics[variant_name]
        if (
            type(variant) is not dict
            or set(variant) != {"arms"}
            or type(variant["arms"]) is not dict
            or set(variant["arms"]) != set(ARM_IDS)
            or type(pool) is not dict
            or set(pool) != {"semantic_kbest", "structure_kbest"}
            or type(diagnostic) is not dict
            or set(diagnostic) != diagnostic_keys
            or type(diagnostic["mapping_receipt_sha256_by_arm"]) is not dict
            or set(diagnostic["mapping_receipt_sha256_by_arm"]) != set(ARM_IDS)
            or type(diagnostic["arms"]) is not dict
            or set(diagnostic["arms"]) != set(ARM_IDS)
        ):
            raise ScarCssmControllerError("CONTROLLER_SCORER_WIRE_ORDER_INVALID")
        ordered_variants[variant_name] = {
            "arms": {arm_id: variant["arms"][arm_id] for arm_id in ARM_IDS}
        }
        ordered_pools[variant_name] = {
            "semantic_kbest": pool["semantic_kbest"],
            "structure_kbest": pool["structure_kbest"],
        }
        ordered_diagnostics[variant_name] = {
            "structural_diagnostics_available": diagnostic[
                "structural_diagnostics_available"
            ],
            "target_color_shuffle_effective": diagnostic[
                "target_color_shuffle_effective"
            ],
            "left_binder": diagnostic["left_binder"],
            "right_binder": diagnostic["right_binder"],
            "left_graph_receipt_sha256": diagnostic[
                "left_graph_receipt_sha256"
            ],
            "right_graph_receipt_sha256": diagnostic[
                "right_graph_receipt_sha256"
            ],
            "mapping_receipt_sha256_by_arm": {
                arm_id: diagnostic["mapping_receipt_sha256_by_arm"][arm_id]
                for arm_id in ARM_IDS
            },
            "arms": {
                arm_id: diagnostic["arms"][arm_id] for arm_id in ARM_IDS
            },
        }

    private_variants = evidence.get("variants")
    private_sides = evidence.get("sides")
    if (
        type(evidence) is not dict
        or set(evidence)
        != {"availability", "error_code", "semantic_matrix", "sides", "variants"}
        or type(private_sides) is not dict
        or set(private_sides) != {"left", "right"}
        or type(private_variants) is not dict
        or set(private_variants) != set(VARIANT_NAMES)
    ):
        raise ScarCssmControllerError("CONTROLLER_SCORER_WIRE_ORDER_INVALID")
    ordered_evidence = {
        "availability": evidence["availability"],
        "error_code": evidence["error_code"],
        "semantic_matrix": evidence["semantic_matrix"],
        "sides": {side: private_sides[side] for side in ("left", "right")},
        "variants": {
            variant_name: private_variants[variant_name]
            for variant_name in VARIANT_NAMES
        },
    }
    return {
        "item_token": prediction["item_token"],
        "variants": ordered_variants,
        "proposal_pools": ordered_pools,
        "execution": prediction["execution"],
        "diagnostics": ordered_diagnostics,
        "private_mechanism_receipts": ordered_evidence,
    }


def _verify_shard_closure(
    config: FormalConfig,
    paths: ControllerPaths,
    *,
    action_items: Sequence[Mapping[str, Any]],
    action_commitment: str,
    collision_tokens: frozenset[str],
    policy: MeasurementPolicy,
    runtime_receipts: Mapping[str, Mapping[str, Any]],
    runtime_file_hashes: Mapping[str, str],
    action_release: Mapping[str, Any],
    action_release_file_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_files = {
        f"shard{shard}.{suffix}"
        for shard in range(SHARD_COUNT)
        for suffix in (
            "attempt.sentinel",
            "records.private.jsonl",
            "runtime.safe.json",
            "terminal.safe.json",
        )
    }
    observed_files = {path.name for path in paths.work.iterdir()}
    if observed_files != expected_files or any(path.is_symlink() for path in paths.work.iterdir()):
        raise ScarCssmControllerError("CONTROLLER_SHARD_OUTPUT_SET_INVALID")

    all_rows: list[dict[str, Any]] = []
    terminal_hashes: dict[str, str] = {}
    private_hashes: dict[str, str] = {}
    total_calls = total_failures = 0
    for shard in range(SHARD_COUNT):
        terminal_path = paths.work / f"shard{shard}.terminal.safe.json"
        terminal, terminal_file_sha256 = _load_worker_terminal(terminal_path)
        expected_terminal_keys = {
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
        expected_partition = [
            item for ordinal, item in enumerate(action_items) if ordinal % SHARD_COUNT == shard
        ]
        if (
            set(terminal) != expected_terminal_keys
            or terminal["schema"] != SHARD_TERMINAL_SCHEMA
            or terminal["status"] != "complete"
            or terminal["version"] != "gscl_scar_cssm_worker_v1"
            or terminal["study_id"] != config.study_id
            or terminal["shard_count"] != SHARD_COUNT
            or terminal["shard_index"] != shard
            or terminal["item_count"] != len(expected_partition)
            or terminal["arm_ids"] != list(ARM_IDS)
            or terminal["variant_names"] != list(VARIANT_NAMES)
            or terminal["action_commitment_sha256"]
            != action_commitment
            or terminal["external_network_call_count"] != 0
            or terminal["formal_label_pack_access_count"] != 0
            or terminal["formal_scorer_access_count"] != 0
            or terminal["encoder_binding_sha256"]
            != config.bindings["encoder_binding_sha256"]
            or terminal["runtime_receipt_file_sha256"]
            != runtime_file_hashes[str(shard)]
            or terminal["runtime_receipt_self_sha256"]
            != runtime_receipts[str(shard)]["self_sha256"]
            or terminal["action_release_self_sha256"]
            != action_release["self_sha256"]
            or type(terminal["action_pack_file_receipt"]) is not dict
            or terminal["action_pack_file_receipt"].get("sha256")
            != config.bindings["action_pack_file_sha256"]
            or terminal["action_pack_file_receipt"].get("mode_octal") != "0600"
            or type(terminal["action_release_file_receipt"]) is not dict
            or terminal["action_release_file_receipt"].get("sha256")
            != action_release_file_sha256
            or terminal["action_release_file_receipt"].get("mode_octal") != "0600"
            or type(terminal["output_root_receipt"]) is not dict
            or terminal["output_root_receipt"].get("filesystem_type") != "ext4"
            or terminal["output_root_receipt"].get("mode_octal") != "0700"
            or type(terminal["mechanism_resource_totals"]) is not dict
            or any(
                type(value) is not int or isinstance(value, bool) or value < 0
                for value in terminal["mechanism_resource_totals"].values()
            )
        ):
            raise ScarCssmControllerError("CONTROLLER_SHARD_TERMINAL_INVALID")
        private_path = paths.work / f"shard{shard}.records.private.jsonl"
        records = _load_shard_items(
            private_path, expected_sha256=terminal["private_records_file_sha256"]
        )
        expected_tokens = [item["item_token"] for item in expected_partition]
        if (
            terminal["private_records_file_size_bytes"] != private_path.stat().st_size
            or len(records) != len(expected_partition)
        ):
            raise ScarCssmControllerError("CONTROLLER_SHARD_RECORD_FILE_INVALID")
        rows: list[dict[str, Any]] = []
        for ordinal, record in enumerate(records):
            if type(record) is not dict or set(record) != {
                "evidence",
                "item_token",
                "ordinal_within_shard",
                "prediction",
                "self_sha256",
            }:
                raise ScarCssmControllerError("CONTROLLER_SHARD_RECORD_INVALID")
            record_body = {
                key: child for key, child in record.items() if key != "self_sha256"
            }
            prediction = record["prediction"]
            evidence = record["evidence"]
            if (
                record["self_sha256"] != _content_hash(record_body)
                or record["ordinal_within_shard"] != ordinal
                or record["item_token"] != expected_tokens[ordinal]
                or type(prediction) is not dict
                or prediction.get("item_token") != record["item_token"]
                or type(evidence) is not dict
                or evidence.get("availability")
                != (
                    "PREMODEL_TYPED_FAILURE"
                    if record["item_token"] in collision_tokens
                    else "COMPLETE"
                )
            ):
                raise ScarCssmControllerError("CONTROLLER_SHARD_RECORD_INVALID")
            if "private_mechanism_receipts" in prediction:
                raise ScarCssmControllerError("CONTROLLER_SHARD_RECORD_INVALID")
            rows.append(_restore_scorer_wire_order(prediction, evidence))
        if [row.get("item_token") for row in rows] != expected_tokens:
            raise ScarCssmControllerError("CONTROLLER_SHARD_PARTITION_INVALID")
        shard_calls = shard_failures = 0
        for row in rows:
            calls, failures = _validate_execution_row(
                row, collision=row["item_token"] in collision_tokens
            )
            shard_calls += calls
            shard_failures += failures
        if (
            type(terminal["document_call_count"]) is not int
            or isinstance(terminal["document_call_count"], bool)
            or type(terminal["structural_typed_failure_count"]) is not int
            or isinstance(terminal["structural_typed_failure_count"], bool)
            or terminal["document_call_count"] != shard_calls
            or terminal["structural_typed_failure_count"] != shard_failures
            or terminal["structural_error_code_counts"]
            != (
                {"SLOT_BINDER_TYPED_FAILURE": shard_failures}
                if shard_failures
                else {}
            )
        ):
            raise ScarCssmControllerError("CONTROLLER_SHARD_ACCOUNTING_INVALID")
        total_calls += shard_calls
        total_failures += shard_failures
        all_rows.extend(rows)
        terminal_hashes[str(shard)] = terminal_file_sha256
        private_hashes[str(shard)] = terminal["private_records_file_sha256"]

    if (
        len(all_rows) != policy.action_item_count
        or total_failures != policy.ambiguous_item_count
        or total_calls != 2 * policy.primary_item_count
        or {row["item_token"] for row in all_rows}
        != {item["item_token"] for item in action_items}
    ):
        raise ScarCssmControllerError("CONTROLLER_SHARD_CLOSURE_INVALID")
    return all_rows, {
        "document_call_count": total_calls,
        "private_records_file_sha256_by_shard": private_hashes,
        "runtime_binding": {
            "encoder_binding_sha256": config.bindings["encoder_binding_sha256"],
            "execution_freeze_sha256": config.bindings["execution_freeze_sha256"],
            "gpu_uuid_by_shard": {
                str(shard): config.gpu_uuids[shard] for shard in range(SHARD_COUNT)
            },
            "implementation_closure_sha256": config.bindings[
                "implementation_closure_sha256"
            ],
            "qwen_canary_self_sha256": config.bindings[
                "qwen_canary_self_sha256"
            ],
            "qwen_runtime_commitment": config.bindings["qwen_runtime_commitment"],
            "runtime_receipt_file_sha256_by_shard": dict(runtime_file_hashes),
        },
        "shard_terminal_file_sha256_by_shard": terminal_hashes,
        "structural_typed_failure_count": total_failures,
    }


def _prediction_pack(
    *,
    action_commitment: str,
    rows: Sequence[Mapping[str, Any]],
    study_id: str,
) -> dict[str, Any]:
    body = {
        "arm_ids": list(ARM_IDS),
        "items": sorted((dict(row) for row in rows), key=lambda row: row["item_token"]),
        "schema": PREDICTION_PACK_SCHEMA,
        "source_action_commitment_sha256": action_commitment,
        "study_id": study_id,
        "variant_names": list(VARIANT_NAMES),
    }
    return _self_hashed(body)


def _safe_value(value: Any) -> None:
    forbidden = {"item_token", "opaque_slot_id", "surface", "gold_pairs", "per_item"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                raise ScarCssmControllerError("CONTROLLER_SAFE_OUTPUT_LEAKAGE")
            _safe_value(child)
    elif isinstance(value, list):
        for child in value:
            _safe_value(child)
    elif isinstance(value, str) and ("scar-item-v1-" in value or "scar-slot-v1-" in value):
        raise ScarCssmControllerError("CONTROLLER_SAFE_OUTPUT_LEAKAGE")


def _verify_result_object(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ScarCssmControllerError(f"{field}_INVALID")
    claimed = value.get("self_sha256")
    body = {key: child for key, child in value.items() if key != "self_sha256"}
    if not isinstance(claimed, str) or _content_hash(body) != claimed:
        raise ScarCssmControllerError(f"{field}_SELF_INVALID")
    return value


def _failure_hash(exc: BaseException) -> str:
    return hashlib.sha256(
        f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
    ).hexdigest()


def _failure_terminal(config: FormalConfig, state: _RunState, exc: BaseException) -> dict[str, Any]:
    issue = getattr(exc, "issue_id", type(exc).__name__)
    return _self_hashed(
        {
            "action_barrier_count": state.action_barrier_count,
            "action_child_launch_count": state.action_child_launch_count,
            "action_release_count": state.action_release_count,
            "api_or_online_evaluator_call_count": 0,
            "config_self_sha256": config.self_sha256,
            "failure_issue_sha256": hashlib.sha256(str(issue).encode("utf-8")).hexdigest(),
            "failure_stage": state.stage,
            "failure_type_sha256": _failure_hash(exc),
            "label_pack_access_count": state.label_pack_access_count,
            "offline_scorer_call_count": state.offline_scorer_call_count,
            "replay_retry_resample_or_fallback_count": 0,
            "schema": SAFE_TERMINAL_SCHEMA,
            "secret_access_count": state.secret_access_count,
            "status": "failed_after_formal_attempt",
            "study_id": config.study_id,
        }
    )


def _pre_attempt_receipt(config: FormalConfig, decision: AdmissionDecision) -> dict[str, Any]:
    return _self_hashed(
        {
            "api_or_online_evaluator_call_count": 0,
            "config_self_sha256": config.self_sha256,
            "effect_study_attempt_count": 0,
            "formal_input_access_count": 0,
            "resource_admission": decision.safe_payload(),
            "schema": f"{VERSION}.pre_attempt_resource_receipt.v1",
            "status": decision.status,
            "study_id": config.study_id,
        }
    )


def _assert_fresh(paths: ControllerPaths) -> None:
    consumed = (
        paths.attempt,
        paths.intent,
        paths.action_release,
        paths.barrier,
        paths.prediction,
        paths.private_result,
        paths.safe_aggregate,
        paths.private_terminal,
        paths.safe_terminal,
        paths.work,
        paths.runtime,
    )
    if any(path.exists() or path.is_symlink() for path in consumed):
        raise ScarCssmControllerAlreadyConsumed("CONTROLLER_FORMAL_ROOT_CONSUMED")


def run_formal_once(
    config: FormalConfig,
    *,
    dependencies: ControllerDependencies = DEFAULT_DEPENDENCIES,
    policy: MeasurementPolicy = OFFICIAL_POLICY,
) -> Mapping[str, Any]:
    """Run or terminally fail one admitted formal attempt; never retry it."""

    _validate_config(config)
    policy.validate()
    paths = ControllerPaths.for_root(config.mutable_root)
    _assert_fresh(paths)
    _verify_all_runtime_paths_ext4(config, dependencies.filesystem_type)

    lock = _try_lock(config.lock_path)
    if lock is None:
        return _pre_attempt_receipt(
            config,
            AdmissionDecision(
                status="DEFERRED_SHARED_RESOURCE",
                reason_codes=("FORMAL_LOCK_OCCUPIED",),
            ),
        )
    try:
        decision = dependencies.resource_probe(config)
        if decision.status == "DEFERRED_SHARED_RESOURCE":
            return _pre_attempt_receipt(config, decision)
        if decision.status != "ADMITTED_SHARED_RESOURCE":
            return _pre_attempt_receipt(config, decision)

        # The nearest existing ancestor was already proven ext4.  Claim the
        # attempt immediately after creating its control directory; all later
        # directory/runtime failures are therefore terminal, never deferrals.
        paths.control.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(paths.root, 0o700)
        os.chmod(paths.control, 0o700)
        attempt = _self_hashed(
            {
                "action_barrier_count": 0,
                "action_release_count": 0,
                "api_or_online_evaluator_call_count": 0,
                "config_self_sha256": config.self_sha256,
                "label_pack_access_count": 0,
                "offline_scorer_call_count": 0,
                "replay_retry_resample_or_fallback_count": 0,
                "resource_admission": decision.safe_payload(),
                "schema": ATTEMPT_SCHEMA,
                "secret_access_count": 0,
                "status": "formal_attempt_claimed_once",
                "study_id": config.study_id,
            }
        )
        attempt_file_sha256 = _publish_once(paths.attempt, attempt)
        state = _RunState(stage="verify_frozen_inputs_and_implementation")
        processes: dict[int, ProcessHandle] = {}
        try:
            paths.work.mkdir(mode=0o700, exist_ok=False)
            paths.runtime.mkdir(mode=0o700, exist_ok=False)
            _verify_implementation_bindings(config)
            sandbox_receipt = _load_sandbox_receipt(config)
            action_pack = _load_canonical_json_file(
                config.action_pack_path,
                field="CONTROLLER_ACTION_PACK",
                maximum_bytes=MAX_ACTION_PACK_BYTES,
                expected_sha256=config.bindings["action_pack_file_sha256"],
            )
            dependencies.validate_action_pack(action_pack, config.study_id)
            action_items, action_tokens = _validate_action_pack_shape(
                action_pack, config=config, policy=policy
            )
            collision_tokens = _collision_tokens(action_items)
            if len(collision_tokens) != policy.ambiguous_item_count:
                raise ScarCssmControllerError("CONTROLLER_AMBIGUOUS_COHORT_DRIFT")

            state.stage = "freeze_two_shard_intent"
            intent = _self_hashed(
                {
                    "action_commitment_sha256": action_pack["action_commitment_sha256"],
                    "all_shards_submitted_before_wait": True,
                    "arm_ids": list(ARM_IDS),
                    "action_pack_release_before_both_runtime_receipts": False,
                    "gpu_uuid_by_shard": {
                        str(shard): config.gpu_uuids[shard]
                        for shard in range(SHARD_COUNT)
                    },
                    "label_or_secret_available_to_action_children": False,
                    "sandbox_receipt_self_sha256": sandbox_receipt["self_sha256"],
                    "schema": INTENT_SCHEMA,
                    "shard_count": SHARD_COUNT,
                    "status": "two_shard_action_intent_frozen",
                    "study_id": config.study_id,
                    "variant_names": list(VARIANT_NAMES),
                }
            )
            intent_file_sha256 = _publish_once(paths.intent, intent)

            state.stage = "launch_two_action_shards_concurrently"
            processes = _launch_shards(
                config,
                paths,
                popen_factory=dependencies.popen_factory,
                state=state,
            )
            state.stage = "validate_two_runtime_receipts_before_action_release"
            runtime_receipts, runtime_file_hashes = _wait_for_runtime_barrier(
                config, paths, processes
            )
            state.stage = "release_both_shards_to_action_pack"
            action_release, action_release_file_sha256 = _publish_action_release(
                config,
                paths,
                runtime_file_hashes=runtime_file_hashes,
            )
            state.action_release_count = 1
            statuses = {
                shard: process.wait() for shard, process in processes.items()
            }
            if statuses != {0: 0, 1: 0}:
                raise ScarCssmControllerError("CONTROLLER_ACTION_SHARD_FAILED")

            state.stage = "validate_and_seal_action_closure"
            rows, closure = _verify_shard_closure(
                config,
                paths,
                action_items=action_items,
                action_commitment=action_pack["action_commitment_sha256"],
                collision_tokens=collision_tokens,
                policy=policy,
                runtime_receipts=runtime_receipts,
                runtime_file_hashes=runtime_file_hashes,
                action_release=action_release,
                action_release_file_sha256=action_release_file_sha256,
            )
            if sorted(row["item_token"] for row in rows) != sorted(action_tokens):
                raise ScarCssmControllerError("CONTROLLER_ACTION_TOKEN_CLOSURE_INVALID")
            prediction = _prediction_pack(
                action_commitment=action_pack["action_commitment_sha256"],
                rows=rows,
                study_id=config.study_id,
            )
            prediction_file_sha256 = _publish_once(paths.prediction, prediction)
            barrier = _self_hashed(
                {
                    "action_commitment_sha256": action_pack["action_commitment_sha256"],
                    "action_release_file_sha256": action_release_file_sha256,
                    "action_release_self_sha256": action_release["self_sha256"],
                    "action_item_count": policy.action_item_count,
                    "all_action_items_durable": True,
                    "ambiguous_expected_typed_failure_count": policy.ambiguous_item_count,
                    "closure": closure,
                    "intent_file_sha256": intent_file_sha256,
                    "intent_self_sha256": intent["self_sha256"],
                    "item_token_set_sha256": _content_hash(sorted(action_tokens)),
                    "label_pack_access_count_before_barrier": 0,
                    "offline_scorer_call_count_before_barrier": 0,
                    "prediction_pack_file_sha256": prediction_file_sha256,
                    "prediction_pack_self_sha256": prediction["self_sha256"],
                    "schema": BARRIER_SCHEMA,
                    "secret_access_count_before_barrier": 0,
                    "shard_count": SHARD_COUNT,
                    "status": "complete_action_closure_sealed",
                    "study_id": config.study_id,
                }
            )
            barrier_file_sha256 = _publish_once(paths.barrier, barrier)
            state.action_barrier_count = 1

            state.stage = "open_late_label_and_secret_capabilities"
            state.label_pack_access_count = 1
            label_raw = _read_regular_once(
                config.label_pack_path,
                field="CONTROLLER_LABEL_PACK",
                maximum_bytes=MAX_LABEL_PACK_BYTES,
                expected_sha256=config.bindings["label_pack_file_sha256"],
                required_mode=0o600,
            )
            label_pack = _strict_json(label_raw, field="CONTROLLER_LABEL_PACK")
            state.secret_access_count = 1
            secret = _read_regular_once(
                config.secret_path,
                field="CONTROLLER_SECRET",
                maximum_bytes=32,
                expected_sha256=config.bindings["secret_file_sha256"],
                required_mode=0o600,
            )
            if len(secret) != 32:
                raise ScarCssmControllerError("CONTROLLER_SECRET_INVALID")

            state.stage = "invoke_frozen_offline_scorer_once"
            state.offline_scorer_call_count = 1
            scored = dependencies.score_once(
                action_pack,
                label_pack,
                prediction,
                secret=secret,
                study_id=config.study_id,
            )
            private_result = _verify_result_object(
                getattr(scored, "private_result", None),
                field="CONTROLLER_PRIVATE_SCORE",
            )
            safe_aggregate = _verify_result_object(
                getattr(scored, "safe_aggregate", None),
                field="CONTROLLER_SAFE_SCORE",
            )
            _safe_value(safe_aggregate)
            private_result_file_sha256 = _publish_once(paths.private_result, private_result)
            safe_aggregate_file_sha256 = _publish_once(paths.safe_aggregate, safe_aggregate)

            state.stage = "publish_private_and_safe_terminals"
            private_terminal = _self_hashed(
                {
                    "action_barrier_file_sha256": barrier_file_sha256,
                    "action_barrier_self_sha256": barrier["self_sha256"],
                    "action_pack_file_sha256": config.bindings["action_pack_file_sha256"],
                    "attempt_file_sha256": attempt_file_sha256,
                    "attempt_self_sha256": attempt["self_sha256"],
                    "label_pack_file_sha256": config.bindings["label_pack_file_sha256"],
                    "prediction_pack_file_sha256": prediction_file_sha256,
                    "prediction_pack_self_sha256": prediction["self_sha256"],
                    "private_score_file_sha256": private_result_file_sha256,
                    "private_score_self_sha256": private_result["self_sha256"],
                    "schema": PRIVATE_TERMINAL_SCHEMA,
                    "secret_file_sha256": config.bindings["secret_file_sha256"],
                    "status": "completed_protocol_valid",
                    "study_id": config.study_id,
                }
            )
            private_terminal_file_sha256 = _publish_once(
                paths.private_terminal, private_terminal
            )
            terminal = _self_hashed(
                {
                    "action_barrier_count": state.action_barrier_count,
                    "action_barrier_file_sha256": barrier_file_sha256,
                    "action_barrier_self_sha256": barrier["self_sha256"],
                    "action_child_launch_count": state.action_child_launch_count,
                    "action_release_count": state.action_release_count,
                    "action_release_file_sha256": action_release_file_sha256,
                    "action_release_self_sha256": action_release["self_sha256"],
                    "aggregate_only_safe_receipt": True,
                    "api_or_online_evaluator_call_count": 0,
                    "attempt_file_sha256": attempt_file_sha256,
                    "attempt_self_sha256": attempt["self_sha256"],
                    "config_self_sha256": config.self_sha256,
                    "label_pack_access_count": state.label_pack_access_count,
                    "label_pack_opened_only_after_action_barrier": True,
                    "offline_scorer_call_count": state.offline_scorer_call_count,
                    "private_terminal_file_sha256": private_terminal_file_sha256,
                    "private_terminal_self_sha256": private_terminal["self_sha256"],
                    "replay_retry_resample_or_fallback_count": 0,
                    "runtime_binding": closure["runtime_binding"],
                    "safe_aggregate_file_sha256": safe_aggregate_file_sha256,
                    "safe_aggregate_self_sha256": safe_aggregate["self_sha256"],
                    "schema": SAFE_TERMINAL_SCHEMA,
                    "secret_access_count": state.secret_access_count,
                    "secret_opened_only_after_action_barrier": True,
                    "shard_count": SHARD_COUNT,
                    "status": "completed_protocol_valid",
                    "study_id": config.study_id,
                }
            )
            _safe_value(terminal)
            _publish_once(paths.safe_terminal, terminal)
            return terminal
        except Exception as exc:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                process.wait()
            failure = _failure_terminal(config, state, exc)
            if not paths.safe_terminal.exists() and not paths.safe_terminal.is_symlink():
                _publish_once(paths.safe_terminal, failure)
            return failure
    finally:
        _release_lock(lock)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SCAR CSSM formal once")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_formal_once(load_config(arguments.config))
    print(_canonical_bytes(result).decode("ascii"))
    if result.get("status") == "completed_protocol_valid":
        return 0
    if result.get("status") == "DEFERRED_SHARED_RESOURCE":
        return EX_TEMPFAIL
    if result.get("effect_study_attempt_count") == 0:
        return EX_SOFTWARE
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AdmissionDecision",
    "CONFIG_SCHEMA",
    "ControllerDependencies",
    "EX_SOFTWARE",
    "EX_TEMPFAIL",
    "FormalConfig",
    "MeasurementPolicy",
    "OFFICIAL_POLICY",
    "SAFE_TERMINAL_SCHEMA",
    "ScarCssmControllerAlreadyConsumed",
    "ScarCssmControllerError",
    "VERSION",
    "load_config",
    "run_formal_once",
]
