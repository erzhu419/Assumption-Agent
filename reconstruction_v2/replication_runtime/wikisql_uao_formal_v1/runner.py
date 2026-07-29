"""Crash-closed one-shot outer controller for the WikiSQL UAO P4 study.

The controller has no retry, replay, resampling, model, provider, or online
evaluation surface.  It verifies one content-addressed configuration and all
frozen source/code/runtime/model/hardware/service bindings, installs an outer
Landlock policy, and then claims one exclusive attempt.

The source compiler is called exactly once.  Agent (physical GPU 1), RAW
(CPU), and candidate-restricted official HippoRAG (physical GPU 0) are then
submitted before any one of them is awaited.  Every action child receives a
second, narrower Landlock policy in which the A_hold label pack is not
reachable.  Only after three common action packs are validated, fsynced, and
sealed by an action-barrier receipt is the independent offline scorer allowed
to read the minimal A_hold label pack.  Private item/action/score material
never enters the outer safe terminal.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from assumption_agent.benchmarks import (
    wikisql_uao_reality_v1 as reality,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as source_compiler,
)


VERSION = "wikisql_uao_formal_outer_v1"
STUDY_ID = "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1"
CONFIG_SCHEMA = f"{VERSION}_content_addressed_config_v1"
ATTEMPT_SCHEMA = f"{VERSION}_attempt_v1"
LIVE_SCHEMA = f"{VERSION}_live_receipt_v1"
INTENT_SCHEMA = f"{VERSION}_action_intent_receipt_v1"
BARRIER_SCHEMA = f"{VERSION}_action_barrier_receipt_v1"
TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
FAILURE_SCHEMA = f"{VERSION}_safe_failure_v1"

FORMAL_ROOT = Path("/home/erzhu419/wikisql_uao_p4_20260729/formal_v1")
UNIT_NAME = "wikisql-uao-p4-formal-v1.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
DESIGN_RELATIVE_PATH = Path("manifests/wikisql_uao_p4_study_design_v1.json")
SERVICE_RELATIVE_PATH = Path("manifests/wikisql-uao-p4-formal-v1.service")
SOURCE_RELATIVE_PATH = Path("source/data.tar.bz2")
FORMAL_ITEM_COUNT = 72

ACTION_MODULE = "assumption_agent.benchmarks.wikisql_uao_action_runtime_v1"
OFFICIAL_MODULE = "replication_runtime.wikisql_uao_official_v1.worker"
SCORER_MODULE = "assumption_agent.benchmarks.wikisql_uao_scorer_v1"

REQUIRED_FILE_BINDINGS = frozenset(
    {
        "design",
        "nvidia_smi_executable",
        "official_python_executable",
        "python_executable",
        "service_unit",
        "source_archive",
        "systemctl_executable",
    }
)
REQUIRED_TREE_BINDINGS = frozenset(
    {
        "babel_dependency_tree",
        "code_tree",
        "encoder_model_tree",
        "hippo_llm_model_tree",
        "official_base_dependency_tree",
        "official_hipporag_tree",
        "official_overlay_dependency_tree",
        "official_python_dependency_tree",
        "python_dependency_tree",
        "python_runtime_tree",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "bindings",
        "design_self_sha256",
        "encoder_model_semantic_sha256",
        "formal_root",
        "gpu_uuids",
        "schema",
        "self_sha256",
        "study_id",
        "unit_name",
    }
)
_BINDINGS_KEYS = frozenset({"files", "trees"})
_FILE_BINDING_KEYS = frozenset(
    {"mode_octal", "path", "sha256", "size_bytes"}
)
_TREE_BINDING_KEYS = frozenset(
    {"file_count", "path", "sha256", "total_bytes"}
)
_GPU_KEYS = frozenset({"0", "1"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]{8,}\Z")
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}\Z")
_FORBIDDEN_SAFE_KEYS = frozenset(
    {
        "action",
        "answer",
        "condition_value",
        "document",
        "gold_row_ids",
        "item",
        "items",
        "opaque_item_id",
        "physical_rows",
        "qrel",
        "question",
        "row",
        "rows",
        "source_line_number",
        "source_table_id",
        "sql",
        "table_header",
        "top5_row_ids",
    }
)
_REQUIRED_SERVICE_LINES = frozenset(
    {
        "Type=oneshot",
        "UMask=0077",
        "Restart=no",
        "RestrictAddressFamilies=AF_UNIX",
        "IPAddressDeny=any",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "KillMode=control-group",
        "TimeoutStartSec=infinity",
    }
)
_FORBIDDEN_SERVICE_PREFIXES = (
    "AmbientCapabilities=",
    "BindPaths=",
    "BindReadOnlyPaths=",
    "CapabilityBoundingSet=",
    "InaccessiblePaths=",
    "MountFlags=",
    "PrivateDevices=",
    "PrivateNetwork=",
    "ProtectHome=",
    "ProtectSystem=",
    "ReadOnlyPaths=",
    "ReadWritePaths=",
    "RootDirectory=",
    "TemporaryFileSystem=",
)

# x86_64 Linux Landlock syscalls and filesystem access bits.
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


class WikiSQLUAOFormalError(RuntimeError):
    """A frozen binding, isolation boundary, or one-shot stage drifted."""


class _DuplicateKey(ValueError):
    pass


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WikiSQLUAOFormalError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value, newline=False)).hexdigest()


def _self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise WikiSQLUAOFormalError("self hash already exists")
    return {**value, "self_sha256": semantic_sha256(value)}


def _pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise WikiSQLUAOFormalError(f"forbidden JSON constant {value}")


def _load_canonical_json(
    path: Path,
    *,
    mode: int,
    field: str,
) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateKey,
    ) as exc:
        raise WikiSQLUAOFormalError(f"{field} is unreadable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
        or not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
    ):
        raise WikiSQLUAOFormalError(
            f"{field} metadata or canonical bytes drifted"
        )
    return value


def _exact_dict(
    value: object,
    keys: frozenset[str],
    field: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise WikiSQLUAOFormalError(f"{field} shape drifted")
    return value


def _absolute(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise WikiSQLUAOFormalError(f"{field} path drifted")
    path = Path(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise WikiSQLUAOFormalError(f"{field} path drifted")
    return path


def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise WikiSQLUAOFormalError(f"{field} SHA-256 drifted")
    return value


def _nonnegative(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise WikiSQLUAOFormalError(f"{field} count drifted")
    return value


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise WikiSQLUAOFormalError("bound file cannot be hashed") from exc
    return digest.hexdigest(), size


def tree_identity(root: Path) -> tuple[str, int, int]:
    """Hash one direct tree, including modes and contained symlink targets."""

    try:
        metadata = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise WikiSQLUAOFormalError("bound tree is unavailable") from exc
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise WikiSQLUAOFormalError("bound tree root drifted")
    entries: list[dict[str, object]] = []
    file_count = 0
    total_bytes = 0
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        try:
            child = path.lstat()
        except OSError as exc:
            raise WikiSQLUAOFormalError(
                "bound tree entry is unavailable"
            ) from exc
        mode = f"{stat.S_IMODE(child.st_mode):04o}"
        if stat.S_ISDIR(child.st_mode):
            entries.append(
                {"kind": "directory", "mode_octal": mode, "path": relative}
            )
            continue
        if stat.S_ISLNK(child.st_mode):
            try:
                target = os.readlink(path)
                resolved = path.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, ValueError) as exc:
                raise WikiSQLUAOFormalError(
                    "bound tree symlink escapes or dangles"
                ) from exc
            entries.append(
                {
                    "kind": "symlink",
                    "mode_octal": mode,
                    "path": relative,
                    "target": target,
                }
            )
            continue
        if not stat.S_ISREG(child.st_mode):
            raise WikiSQLUAOFormalError(
                "bound tree contains a special file"
            )
        digest, size = _file_sha256(path)
        file_count += 1
        total_bytes += size
        entries.append(
            {
                "kind": "file",
                "mode_octal": mode,
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return semantic_sha256(entries), file_count, total_bytes


@dataclass(frozen=True, slots=True)
class FileBinding:
    path: Path
    sha256: str
    size_bytes: int
    mode: int

    @classmethod
    def parse(cls, value: object, field: str) -> "FileBinding":
        row = _exact_dict(value, _FILE_BINDING_KEYS, field)
        mode = row["mode_octal"]
        if (
            not isinstance(mode, str)
            or re.fullmatch(r"0[0-7]{3}", mode) is None
        ):
            raise WikiSQLUAOFormalError(f"{field} mode drifted")
        return cls(
            path=_absolute(row["path"], field),
            sha256=_hex64(row["sha256"], field),
            size_bytes=_nonnegative(row["size_bytes"], field),
            mode=int(mode, 8),
        )

    def payload(self) -> dict[str, object]:
        return {
            "mode_octal": f"{self.mode:04o}",
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def verify(self, field: str) -> None:
        try:
            metadata = self.path.lstat()
        except OSError as exc:
            raise WikiSQLUAOFormalError(f"{field} is unavailable") from exc
        observed, size = _file_sha256(self.path)
        if (
            self.path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != self.mode
            or metadata.st_size != self.size_bytes
            or size != self.size_bytes
            or observed != self.sha256
        ):
            raise WikiSQLUAOFormalError(f"{field} binding drifted")


@dataclass(frozen=True, slots=True)
class TreeBinding:
    path: Path
    sha256: str
    file_count: int
    total_bytes: int

    @classmethod
    def parse(cls, value: object, field: str) -> "TreeBinding":
        row = _exact_dict(value, _TREE_BINDING_KEYS, field)
        return cls(
            path=_absolute(row["path"], field),
            sha256=_hex64(row["sha256"], field),
            file_count=_nonnegative(row["file_count"], field),
            total_bytes=_nonnegative(row["total_bytes"], field),
        )

    def payload(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "path": str(self.path),
            "sha256": self.sha256,
            "total_bytes": self.total_bytes,
        }

    def verify(self, field: str) -> None:
        identity = tree_identity(self.path)
        if identity != (self.sha256, self.file_count, self.total_bytes):
            raise WikiSQLUAOFormalError(f"{field} binding drifted")


@dataclass(frozen=True, slots=True)
class FormalConfig:
    path: Path
    formal_root: Path
    design_self_sha256: str
    encoder_model_semantic_sha256: str
    files: Mapping[str, FileBinding]
    trees: Mapping[str, TreeBinding]
    gpu_uuids: Mapping[str, str]
    self_sha256: str

    def file(self, name: str) -> FileBinding:
        try:
            return self.files[name]
        except KeyError as exc:
            raise WikiSQLUAOFormalError("file binding is absent") from exc

    def tree(self, name: str) -> TreeBinding:
        try:
            return self.trees[name]
        except KeyError as exc:
            raise WikiSQLUAOFormalError("tree binding is absent") from exc


def load_config(path: Path) -> FormalConfig:
    expected_path = FORMAL_ROOT / "control/formal_config.json"
    if path != expected_path:
        raise WikiSQLUAOFormalError("formal config path drifted")
    value = _load_canonical_json(path, mode=0o600, field="formal config")
    _exact_dict(value, _CONFIG_KEYS, "formal config")
    supplied_self = _hex64(value["self_sha256"], "formal config")
    base = {key: child for key, child in value.items() if key != "self_sha256"}
    if semantic_sha256(base) != supplied_self:
        raise WikiSQLUAOFormalError("formal config self hash drifted")
    if (
        value["schema"] != CONFIG_SCHEMA
        or value["study_id"] != STUDY_ID
        or value["unit_name"] != UNIT_NAME
        or _absolute(value["formal_root"], "formal root") != FORMAL_ROOT
    ):
        raise WikiSQLUAOFormalError("formal config identity drifted")
    bindings = _exact_dict(value["bindings"], _BINDINGS_KEYS, "bindings")
    raw_files = bindings["files"]
    raw_trees = bindings["trees"]
    if (
        type(raw_files) is not dict
        or set(raw_files) != REQUIRED_FILE_BINDINGS
        or type(raw_trees) is not dict
        or set(raw_trees) != REQUIRED_TREE_BINDINGS
    ):
        raise WikiSQLUAOFormalError("binding registry drifted")
    files = {
        name: FileBinding.parse(raw_files[name], f"file binding {name}")
        for name in sorted(REQUIRED_FILE_BINDINGS)
    }
    trees = {
        name: TreeBinding.parse(raw_trees[name], f"tree binding {name}")
        for name in sorted(REQUIRED_TREE_BINDINGS)
    }
    gpu = _exact_dict(value["gpu_uuids"], _GPU_KEYS, "GPU UUIDs")
    if any(
        not isinstance(gpu[index], str)
        or _GPU_UUID.fullmatch(gpu[index]) is None
        for index in ("0", "1")
    ) or gpu["0"] == gpu["1"]:
        raise WikiSQLUAOFormalError("GPU UUID binding drifted")
    design_self = _hex64(value["design_self_sha256"], "design self")
    encoder_model_semantic = _hex64(
        value["encoder_model_semantic_sha256"],
        "encoder model semantic tree",
    )
    config = FormalConfig(
        path=path,
        formal_root=FORMAL_ROOT,
        design_self_sha256=design_self,
        encoder_model_semantic_sha256=encoder_model_semantic,
        files=files,
        trees=trees,
        gpu_uuids={"0": gpu["0"], "1": gpu["1"]},  # type: ignore[dict-item]
        self_sha256=supplied_self,
    )
    code = config.tree("code_tree").path
    if (
        code != FORMAL_ROOT / "reconstruction_v2"
        or config.file("design").path != code / DESIGN_RELATIVE_PATH
        or config.file("service_unit").path != code / SERVICE_RELATIVE_PATH
        or config.file("source_archive").path
        != FORMAL_ROOT / SOURCE_RELATIVE_PATH
    ):
        raise WikiSQLUAOFormalError("fixed formal layout drifted")
    protected_label = (
        FORMAL_ROOT / "work/compiled_source/private/A_hold.labels.json"
    )
    for name, binding in config.trees.items():
        try:
            protected_label.relative_to(binding.path)
        except ValueError:
            continue
        raise WikiSQLUAOFormalError(
            f"tree binding {name} would expose A_hold labels"
        )
    return config


@dataclass(frozen=True, slots=True)
class FormalPaths:
    root: Path
    control: Path
    work: Path
    compiled: Path
    attempt: Path
    live: Path
    intent: Path
    barrier: Path
    terminal: Path
    agent_root: Path
    raw_root: Path
    hippo_root: Path
    scorer_root: Path
    agent_action: Path
    raw_action: Path
    hippo_action: Path
    agent_policy: Path
    agent_receipt: Path
    hippo_receipt: Path
    scorer_labels: Path
    score_private: Path
    score_safe: Path
    score_terminal: Path

    @classmethod
    def for_root(cls, root: Path) -> "FormalPaths":
        control = root / "control"
        work = root / "work"
        agent = work / "agent"
        raw = work / "raw"
        hippo = work / "hippo"
        scorer = work / "scorer"
        return cls(
            root=root,
            control=control,
            work=work,
            compiled=work / "compiled_source",
            attempt=control / "formal_attempt.json",
            live=control / "outer_live.safe.json",
            intent=control / "action_intents.safe.json",
            barrier=control / "action_barrier.safe.json",
            terminal=control / "outer_terminal.safe.json",
            agent_root=agent,
            raw_root=raw,
            hippo_root=hippo,
            scorer_root=scorer,
            agent_action=agent / "A_hold.Agent.actions.json",
            raw_action=raw / "A_hold.RAW.actions.json",
            hippo_action=hippo / "A_hold.HippoRAG.actions.json",
            agent_policy=agent / "compiled_policy.private.json",
            agent_receipt=agent / "agent.safe.json",
            hippo_receipt=hippo / "hipporag.safe.json",
            scorer_labels=scorer / "A_hold.minimal.labels.json",
            score_private=scorer / "three_arm.private.json",
            score_safe=scorer / "score.safe.json",
            score_terminal=scorer / "scorer_terminal.safe.json",
        )

    @property
    def a_form_view(self) -> Path:
        return self.compiled / "private/A_form.action_views.json"

    @property
    def a_form_labels(self) -> Path:
        return self.compiled / "private/A_form.labels.json"

    @property
    def a_hold_view(self) -> Path:
        return self.compiled / "private/A_hold.action_views.json"

    @property
    def a_hold_labels(self) -> Path:
        return self.compiled / "private/A_hold.labels.json"

    @property
    def provenance(self) -> Path:
        return self.compiled / "private/controller_only.provenance.json"

    @property
    def compiler_receipt(self) -> Path:
        return self.compiled / "safe/source_compiler_receipt.json"


def _write_once(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int = 0o600,
) -> str:
    raw = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, mode)
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise WikiSQLUAOFormalError(
            "exclusive formal artifact write failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(descriptor)
    except OSError as exc:
        raise WikiSQLUAOFormalError("directory durability failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _durable_file_sha256(path: Path, field: str) -> str:
    try:
        metadata = path.lstat()
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise WikiSQLUAOFormalError(f"{field} is not durable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise WikiSQLUAOFormalError(f"{field} metadata drifted")
    _fsync_directory(path.parent)
    return _file_sha256(path)[0]


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def landlock_abi_version() -> int:
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
        raise WikiSQLUAOFormalError("Landlock ABI is unavailable")
    return result


def _landlock_rights(abi: int) -> int:
    rights = _LL_BASE
    if abi >= 2:
        rights |= _LL_REFER
    if abi >= 3:
        rights |= _LL_TRUNCATE
    return rights


def apply_landlock(
    *,
    read_paths: Sequence[Path],
    write_paths: Sequence[Path],
    device_paths: Sequence[Path] = (),
) -> None:
    """Restrict the calling process; failure is terminal and has no fallback."""

    abi = landlock_abi_version()
    handled = _landlock_rights(abi)
    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(ruleset_attr),
            ctypes.sizeof(ruleset_attr),
            ctypes.c_uint(0),
        )
    )
    if ruleset_fd < 0:
        raise WikiSQLUAOFormalError("Landlock ruleset creation failed")

    def add(path: Path, rights: int) -> None:
        flags = os.O_PATH | os.O_CLOEXEC
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise WikiSQLUAOFormalError(
                "Landlock allowlist path is unavailable"
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
                raise WikiSQLUAOFormalError(
                    "Landlock path rule creation failed"
                )
        finally:
            os.close(descriptor)

    try:
        read_rights = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR
        for path in dict.fromkeys(Path(row) for row in read_paths):
            add(path, read_rights)
        for path in dict.fromkeys(Path(row) for row in write_paths):
            add(path, handled)
        for path in dict.fromkeys(Path(row) for row in device_paths):
            add(path, _LL_READ_FILE | _LL_WRITE_FILE)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise WikiSQLUAOFormalError("no_new_privs could not be set")
        if (
            libc.syscall(
                _SYS_LANDLOCK_RESTRICT_SELF,
                ruleset_fd,
                ctypes.c_uint(0),
            )
            != 0
        ):
            raise WikiSQLUAOFormalError("Landlock restriction failed")
    finally:
        os.close(ruleset_fd)


@dataclass(frozen=True, slots=True)
class ServiceAttestation:
    nrestarts: int
    invocation_id: str
    active_state: str
    sub_state: str
    fragment_path: Path
    drop_in_paths: str
    restart: str
    exec_start: str
    service_type: str
    timeout_start_usec: str
    cpu_quota_per_sec_usec: str
    memory_max: str
    tasks_max: str
    ip_address_deny: str
    umask: str
    private_tmp: str
    no_new_privileges: str
    restrict_address_families: str
    kill_mode: str


@dataclass(frozen=True, slots=True)
class GPUAttestation:
    uuids: Mapping[str, str]
    compute_process_count: int


def _systemctl_attestation(
    config: FormalConfig,
    *,
    unit_name: str = UNIT_NAME,
    installed_unit_path: Path = INSTALLED_UNIT_PATH,
) -> ServiceAttestation:
    executable = config.file("systemctl_executable").path
    runtime_root = Path(f"/run/user/{os.getuid()}")
    try:
        runtime_metadata = runtime_root.lstat()
    except OSError as exc:
        raise WikiSQLUAOFormalError(
            "user runtime directory is unavailable"
        ) from exc
    if (
        runtime_root.is_symlink()
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or runtime_metadata.st_uid != os.getuid()
    ):
        raise WikiSQLUAOFormalError(
            "user runtime directory metadata drifted"
        )
    argv = [
        str(executable),
        "--user",
        "show",
        unit_name,
        "--no-pager",
        "--property=NRestarts",
        "--property=InvocationID",
        "--property=ActiveState",
        "--property=SubState",
        "--property=FragmentPath",
        "--property=DropInPaths",
        "--property=Restart",
        "--property=ExecStart",
        "--property=Type",
        "--property=TimeoutStartUSec",
        "--property=CPUQuotaPerSecUSec",
        "--property=MemoryMax",
        "--property=TasksMax",
        "--property=IPAddressDeny",
        "--property=UMask",
        "--property=PrivateTmp",
        "--property=NoNewPrivileges",
        "--property=RestrictAddressFamilies",
        "--property=KillMode",
    ]
    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env={
                "DBUS_SESSION_BUS_ADDRESS": (
                    f"unix:path={runtime_root / 'bus'}"
                ),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "XDG_RUNTIME_DIR": str(runtime_root),
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WikiSQLUAOFormalError(
            "live user-service attestation failed"
        ) from exc
    rows: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator != "=" or key in rows:
            raise WikiSQLUAOFormalError(
                "live user-service attestation shape drifted"
            )
        rows[key] = value
    if set(rows) != {
        "CPUQuotaPerSecUSec",
        "DropInPaths",
        "ExecStart",
        "NRestarts",
        "InvocationID",
        "ActiveState",
        "SubState",
        "FragmentPath",
        "IPAddressDeny",
        "KillMode",
        "MemoryMax",
        "NoNewPrivileges",
        "PrivateTmp",
        "Restart",
        "RestrictAddressFamilies",
        "TasksMax",
        "TimeoutStartUSec",
        "Type",
        "UMask",
    }:
        raise WikiSQLUAOFormalError(
            "live user-service attestation fields drifted"
        )
    try:
        nrestarts = int(rows["NRestarts"])
    except ValueError as exc:
        raise WikiSQLUAOFormalError("NRestarts is not an integer") from exc
    fragment_path = _absolute(rows["FragmentPath"], "service fragment")
    if fragment_path != installed_unit_path:
        raise WikiSQLUAOFormalError(
            "installed user-service fragment path drifted"
        )
    return ServiceAttestation(
        nrestarts=nrestarts,
        invocation_id=rows["InvocationID"],
        active_state=rows["ActiveState"],
        sub_state=rows["SubState"],
        fragment_path=fragment_path,
        drop_in_paths=rows["DropInPaths"],
        restart=rows["Restart"],
        exec_start=rows["ExecStart"],
        service_type=rows["Type"],
        timeout_start_usec=rows["TimeoutStartUSec"],
        cpu_quota_per_sec_usec=rows["CPUQuotaPerSecUSec"],
        memory_max=rows["MemoryMax"],
        tasks_max=rows["TasksMax"],
        ip_address_deny=rows["IPAddressDeny"],
        umask=rows["UMask"],
        private_tmp=rows["PrivateTmp"],
        no_new_privileges=rows["NoNewPrivileges"],
        restrict_address_families=rows["RestrictAddressFamilies"],
        kill_mode=rows["KillMode"],
    )


def _gpu_attestation(config: FormalConfig) -> GPUAttestation:
    executable = config.file("nvidia_smi_executable").path
    try:
        identity = subprocess.run(
            [
                str(executable),
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
        )
        processes = subprocess.run(
            [
                str(executable),
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WikiSQLUAOFormalError("GPU attestation failed") from exc
    rows: dict[str, str] = {}
    for line in identity.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) != 2
            or fields[0] in rows
            or fields[0] not in {"0", "1"}
            or _GPU_UUID.fullmatch(fields[1]) is None
        ):
            raise WikiSQLUAOFormalError("GPU identity output drifted")
        rows[fields[0]] = fields[1]
    process_rows = [
        line
        for line in processes.stdout.splitlines()
        if line.strip() and "No running processes found" not in line
    ]
    return GPUAttestation(
        uuids=rows,
        compute_process_count=len(process_rows),
    )


def _verify_service_profile(raw: bytes, config: FormalConfig) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WikiSQLUAOFormalError("service unit is not UTF-8") from exc
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    required = {
        *_REQUIRED_SERVICE_LINES,
        f"WorkingDirectory={FORMAL_ROOT}/reconstruction_v2",
    }
    if not required.issubset(lines):
        raise WikiSQLUAOFormalError(
            "minimal UAO v3 service directives drifted"
        )
    if any(
        line.startswith(prefix)
        for line in lines
        for prefix in _FORBIDDEN_SERVICE_PREFIXES
    ):
        raise WikiSQLUAOFormalError(
            "capability or mount hardening is forbidden for this service"
        )
    exec_lines = [line for line in lines if line.startswith("ExecStart=")]
    if (
        len(exec_lines) != 1
        or str(FORMAL_ROOT / "control/formal_config.json")
        not in exec_lines[0]
        or str(config.file("python_executable").path) not in exec_lines[0]
        or str(config.tree("code_tree").path) not in exec_lines[0]
        or str(config.tree("python_dependency_tree").path)
        not in exec_lines[0]
        or str(config.tree("babel_dependency_tree").path)
        not in exec_lines[0]
        or "replication_runtime.wikisql_uao_formal_v1.runner"
        not in exec_lines[0]
    ):
        raise WikiSQLUAOFormalError("service ExecStart drifted")


def _verify_effective_service_profile(
    service: ServiceAttestation,
    unit_raw: bytes,
) -> None:
    try:
        unit_text = unit_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WikiSQLUAOFormalError("service unit is not UTF-8") from exc
    exec_lines = [
        line.strip()
        for line in unit_text.splitlines()
        if line.strip().startswith("ExecStart=")
    ]
    if len(exec_lines) != 1:
        raise WikiSQLUAOFormalError("service ExecStart shape drifted")
    expected_argv = exec_lines[0].removeprefix("ExecStart=")
    prefix = "{ path=/usr/bin/env ; argv[]="
    delimiter = " ; ignore_errors=no ; "
    if (
        not service.exec_start.startswith(prefix)
        or service.exec_start.count(prefix) != 1
        or service.exec_start.count(delimiter) != 1
        or not service.exec_start.endswith(" }")
    ):
        raise WikiSQLUAOFormalError(
            "effective service ExecStart shape drifted"
        )
    effective_argv = service.exec_start[
        len(prefix) : service.exec_start.index(delimiter)
    ]
    if effective_argv != expected_argv:
        raise WikiSQLUAOFormalError(
            "effective service ExecStart drifted"
        )
    if (
        service.drop_in_paths != ""
        or service.restart != "no"
        or service.service_type != "oneshot"
        or service.timeout_start_usec != "infinity"
        or service.cpu_quota_per_sec_usec != "7s"
        or service.memory_max != "42949672960"
        or service.tasks_max != "128"
        or set(service.ip_address_deny.split())
        != {"::/0", "0.0.0.0/0"}
        or service.umask != "0077"
        or service.private_tmp != "yes"
        or service.no_new_privileges != "yes"
        or service.restrict_address_families != "AF_UNIX"
        or service.kill_mode != "control-group"
    ):
        raise WikiSQLUAOFormalError(
            "effective user-service properties drifted"
        )


def _verify_design(config: FormalConfig) -> None:
    binding = config.file("design")
    value = _load_canonical_json(
        binding.path,
        mode=binding.mode,
        field="study design",
    )
    supplied = _hex64(value.get("self_sha256"), "study design")
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if (
        value.get("schema") != "wikisql_uao_p4_study_design_v1"
        or value.get("study_id") != STUDY_ID
        or supplied != config.design_self_sha256
        or semantic_sha256(body) != supplied
    ):
        raise WikiSQLUAOFormalError("study design binding drifted")


def _verify_bindings(
    config: FormalConfig,
    paths: FormalPaths,
    service_probe: Callable[[FormalConfig], ServiceAttestation],
    gpu_probe: Callable[[FormalConfig], GPUAttestation],
    abi_probe: Callable[[], int],
) -> tuple[ServiceAttestation, GPUAttestation, int]:
    # The source path and its declared commitment are already fixed by the
    # canonical config, but source payload bytes must not be read before the
    # exclusive attempt exists.  Verify every non-source binding here and the
    # archive itself immediately after the durable attempt/live receipts.
    for name in sorted(config.files):
        if name == "source_archive":
            continue
        config.file(name).verify(f"file binding {name}")
    for name in sorted(config.trees):
        config.tree(name).verify(f"tree binding {name}")
    if (
        action_runtime.directory_tree_sha256(
            config.tree("encoder_model_tree").path
        )
        != config.encoder_model_semantic_sha256
    ):
        raise WikiSQLUAOFormalError(
            "encoder model semantic tree binding drifted"
        )
    _verify_design(config)
    service_raw = config.file("service_unit").path.read_bytes()
    _verify_service_profile(service_raw, config)
    service = service_probe(config)
    if (
        service.nrestarts != 0
        or _INVOCATION_ID.fullmatch(service.invocation_id) is None
        or service.active_state not in {"activating", "active"}
        or service.sub_state not in {"start", "running"}
    ):
        raise WikiSQLUAOFormalError("live service state drifted")
    _verify_effective_service_profile(service, service_raw)
    try:
        fragment_raw = service.fragment_path.read_bytes()
    except OSError as exc:
        raise WikiSQLUAOFormalError(
            "installed service fragment cannot be read"
        ) from exc
    if (
        hashlib.sha256(fragment_raw).hexdigest()
        != config.file("service_unit").sha256
    ):
        raise WikiSQLUAOFormalError("installed service unit drifted")
    gpu = gpu_probe(config)
    if (
        dict(gpu.uuids) != dict(config.gpu_uuids)
        or gpu.compute_process_count != 0
    ):
        raise WikiSQLUAOFormalError("GPU identity or availability drifted")
    abi = abi_probe()
    if type(abi) is not int or abi < 3:
        raise WikiSQLUAOFormalError("Landlock ABI is below the frozen minimum")
    for directory in (paths.root, paths.control, paths.work):
        try:
            metadata = directory.lstat()
        except OSError as exc:
            raise WikiSQLUAOFormalError(
                "formal root layout is unavailable"
            ) from exc
        if (
            directory.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise WikiSQLUAOFormalError("formal root layout drifted")
    occupied = (
        paths.attempt,
        paths.live,
        paths.intent,
        paths.barrier,
        paths.terminal,
        paths.compiled,
        paths.agent_root,
        paths.raw_root,
        paths.hippo_root,
        paths.scorer_root,
    )
    if any(path.exists() or path.is_symlink() for path in occupied):
        raise WikiSQLUAOFormalError(
            "formal output or attempt evidence already exists"
        )
    return service, gpu, abi


def _existing_system_read_paths() -> tuple[Path, ...]:
    candidates = (
        Path("/usr"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev/null"),
        Path("/dev/urandom"),
    )
    return tuple(path for path in candidates if path.exists())


def _gpu_device_paths(index: str) -> tuple[Path, ...]:
    candidates = (
        Path(f"/dev/nvidia{index}"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
        Path("/dev/nvidia-uvm-tools"),
        Path("/dev/nvidia-modeset"),
    )
    return tuple(path for path in candidates if path.exists())


def _outer_landlock(config: FormalConfig, paths: FormalPaths) -> None:
    read_paths = [
        *_existing_system_read_paths(),
        *(binding.path for binding in config.files.values()),
        *(binding.path for binding in config.trees.values()),
    ]
    apply_landlock(
        read_paths=read_paths,
        write_paths=(paths.root, Path("/tmp")),
        # Landlock layers can only remove rights.  The parent must therefore
        # admit both frozen physical GPUs before Agent/HippoRAG children narrow
        # their own device views to GPU1 and GPU0 respectively.
        device_paths=(
            *_gpu_device_paths("0"),
            *_gpu_device_paths("1"),
        ),
    )


@dataclass(frozen=True, slots=True)
class SourceArtifacts:
    output_file_sha256: Mapping[str, str]
    compiler_receipt_self_sha256: str
    a_hold_view_self_sha256: str
    a_hold_label_pack_self_sha256: str


SourceCompile = Callable[
    [FormalConfig, FormalPaths],
    Mapping[str, str],
]


def _compile_source_production(
    config: FormalConfig,
    paths: FormalPaths,
) -> Mapping[str, str]:
    bundle = source_compiler.compile_archive(
        config.file("source_archive").path,
        expected_archive_sha256=config.file("source_archive").sha256,
        config=source_compiler.CompilerConfig.production(),
    )
    return source_compiler.write_compilation(paths.compiled, bundle)


def _verify_source_outputs(
    config: FormalConfig,
    paths: FormalPaths,
    output_hashes: Mapping[str, str],
) -> SourceArtifacts:
    expected = {
        "private/selection_secret.bin",
        "private/A_form.action_views.json",
        "private/A_form.labels.json",
        "private/A_hold.action_views.json",
        "private/A_hold.labels.json",
        "private/controller_only.provenance.json",
        "safe/source_compiler_receipt.json",
    }
    if (
        type(output_hashes) is not dict
        or set(output_hashes) != expected
        or any(
            _HEX64.fullmatch(value) is None
            for value in output_hashes.values()
            if isinstance(value, str)
        )
        or any(not isinstance(value, str) for value in output_hashes.values())
    ):
        raise WikiSQLUAOFormalError("source compiler output registry drifted")
    for relative in expected:
        path = paths.compiled / relative
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise WikiSQLUAOFormalError(
                "source compiler output is absent"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
        ):
            raise WikiSQLUAOFormalError(
                "source compiler output metadata drifted"
            )
        # Do not reopen A_hold labels before the action barrier.  Their file
        # commitment came directly from the unique compiler invocation.
        if relative != "private/A_hold.labels.json":
            observed = _durable_file_sha256(
                path, f"source compiler output {relative}"
            )
            if observed != output_hashes[relative]:
                raise WikiSQLUAOFormalError(
                    "source compiler output hash drifted"
                )
    receipt = _load_canonical_json(
        paths.compiler_receipt,
        mode=0o600,
        field="source compiler safe receipt",
    )
    receipt_self = _hex64(
        receipt.get("self_sha256"), "source compiler receipt"
    )
    body = {
        key: value
        for key, value in receipt.items()
        if key != "self_sha256"
    }
    if (
        semantic_sha256(body) != receipt_self
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status")
        != "compiled_source_and_sealed_private_packs"
        or receipt.get("selected_item_count") != 264
        or receipt.get("authorized_member_open_count")
        != len(source_compiler.REQUIRED_MEMBERS)
        or receipt.get("train_test_table_overlap_count") != 0
        or receipt.get("selected_sqlite_consistency_assert_count") != 264
        or receipt.get("source_archive_sha256")
        != config.file("source_archive").sha256
        or receipt.get("source_archive_git_blob_sha1")
        != source_compiler.PRODUCTION_ARCHIVE_GIT_BLOB_SHA1
        or type(receipt.get("sqlite_rowid_eligible_count")) is not int
        or receipt["sqlite_rowid_eligible_count"] < 264
        or receipt.get("babel_runtime_version")
        != source_compiler.PRODUCTION_BABEL_VERSION
        or receipt.get("babel_required_production_version")
        != source_compiler.PRODUCTION_BABEL_VERSION
        or receipt.get("babel_locale") != source_compiler.BABEL_LOCALE
    ):
        raise WikiSQLUAOFormalError("source compiler receipt drifted")
    expected_eligibility_contract = {
        "condition_count": 1,
        "condition_operator_indices": [0, 1, 2],
        "table_physical_row_count_minimum": reality.MIN_TABLE_ROWS,
        "table_physical_row_count_maximum": reality.MAX_TABLE_ROWS,
        "column_count_minimum": 1,
        "column_count_maximum": source_compiler.MAX_COLUMNS,
        "question_character_count_maximum": (
            source_compiler.MAX_QUESTION_CHARACTERS
        ),
        "header_or_cell_character_count_maximum": (
            source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
        ),
        "canonical_serialized_row_character_count_maximum": (
            reality.MAX_SERIALIZED_ROW_CHARACTERS
        ),
        "canonical_serialized_rows_must_round_trip": True,
        "canonical_serialized_rows_must_be_unique": True,
        "sqlite_schema_rowid_order_and_normalized_cells_must_match_json_before_gold_derivation": True,
        "sqlite_gold_row_count_minimum": reality.MIN_GOLD_ROWS,
        "sqlite_gold_row_count_maximum": reality.MAX_GOLD_ROWS,
        "sqlite_gold_authoritative_before_HMAC": True,
    }
    if receipt.get("eligibility_contract") != expected_eligibility_contract:
        raise WikiSQLUAOFormalError(
            "source compiler eligibility contract drifted"
        )
    commitments = receipt.get("pack_commitments")
    if not isinstance(commitments, Mapping):
        raise WikiSQLUAOFormalError(
            "source compiler pack commitments drifted"
        )
    try:
        a_hold_label_self = _hex64(
            commitments["A_hold_label"]["self_sha256"],  # type: ignore[index]
            "A_hold label pack",
        )
    except (KeyError, TypeError) as exc:
        raise WikiSQLUAOFormalError(
            "A_hold label commitment is absent"
        ) from exc
    a_form_view = _load_canonical_json(
        paths.a_form_view, mode=0o600, field="A_form view pack"
    )
    a_form_labels = _load_canonical_json(
        paths.a_form_labels, mode=0o600, field="A_form label pack"
    )
    a_hold_view = _load_canonical_json(
        paths.a_hold_view, mode=0o600, field="A_hold view pack"
    )
    try:
        action_runtime.require_formal_agent_counts(
            a_form_view_pack=a_form_view,
            a_form_label_pack=a_form_labels,
            a_hold_view_pack=a_hold_view,
        )
        action_runtime.decode_view_pack(
            a_hold_view,
            expected_block="A_hold",
            expected_count=FORMAL_ITEM_COUNT,
        )
    except action_runtime.WikiSQLUAOActionRuntimeError as exc:
        raise WikiSQLUAOFormalError(
            "source compiler action boundary drifted"
        ) from exc
    a_hold_view_self = _hex64(
        a_hold_view.get("self_sha256"), "A_hold action view pack"
    )
    return SourceArtifacts(
        output_file_sha256=dict(output_hashes),
        compiler_receipt_self_sha256=receipt_self,
        a_hold_view_self_sha256=a_hold_view_self,
        a_hold_label_pack_self_sha256=a_hold_label_self,
    )


def _project_minimal_labels_production(
    paths: FormalPaths,
    source: SourceArtifacts,
) -> str:
    """Open A_hold labels only after the barrier and emit scorer language."""

    source_pack = _load_canonical_json(
        paths.a_hold_labels,
        mode=0o600,
        field="post-barrier A_hold source label pack",
    )
    if (
        source_pack.get("schema")
        != f"{source_compiler.VERSION}_private_label_pack_v1"
        or source_pack.get("study_id") != STUDY_ID
        or source_pack.get("block") != "A_hold"
        or source_pack.get("item_count") != FORMAL_ITEM_COUNT
        or source_pack.get("release_policy")
        != "after_all_A_hold_three_arm_actions_are_sealed"
        or source_pack.get("self_sha256")
        != source.a_hold_label_pack_self_sha256
    ):
        raise WikiSQLUAOFormalError("post-barrier A_hold labels drifted")
    body = {
        key: value
        for key, value in source_pack.items()
        if key != "self_sha256"
    }
    if semantic_sha256(body) != source_pack["self_sha256"]:
        raise WikiSQLUAOFormalError(
            "post-barrier A_hold label self hash drifted"
        )
    raw_items = source_pack.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != FORMAL_ITEM_COUNT:
        raise WikiSQLUAOFormalError("A_hold label item count drifted")
    expected_item_fields = source_compiler.LABEL_VIEW_FIELDS
    if any(
        type(item) is not dict or set(item) != expected_item_fields
        for item in raw_items
    ):
        raise WikiSQLUAOFormalError("A_hold minimal item fields drifted")
    scorer = importlib.import_module(
        "assumption_agent.benchmarks.wikisql_uao_scorer_v1"
    )
    try:
        projected = scorer.build_minimal_label_pack(
            action_view_pack_sha256=source.a_hold_view_self_sha256,
            items=raw_items,
        )
    except Exception as exc:
        raise WikiSQLUAOFormalError(
            "post-barrier minimal label projection failed"
        ) from exc
    return _write_once(paths.scorer_labels, projected, mode=0o600)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    read_paths: tuple[Path, ...]
    write_paths: tuple[Path, ...]
    device_paths: tuple[Path, ...] = ()


CommandBuilder = Callable[
    [FormalConfig, FormalPaths, SourceArtifacts],
    Mapping[str, CommandSpec],
]
LabelProjector = Callable[[FormalPaths, SourceArtifacts], str]


def _lane_environment(
    config: FormalConfig,
    root: Path,
    *,
    cuda_visible_devices: str,
) -> dict[str, str]:
    module_roots = (
        config.tree("code_tree").path,
        config.tree("python_dependency_tree").path,
        config.tree("babel_dependency_tree").path,
    )
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in module_roots),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _python_prefix(config: FormalConfig, module: str) -> tuple[str, ...]:
    return (
        str(config.file("python_executable").path),
        "-S",
        "-B",
        "-s",
        "-m",
        module,
    )


def _production_action_commands(
    config: FormalConfig,
    paths: FormalPaths,
    source: SourceArtifacts,
) -> Mapping[str, CommandSpec]:
    del source
    shared = (
        *_existing_system_read_paths(),
        config.file("python_executable").path,
        config.tree("python_runtime_tree").path,
        config.tree("python_dependency_tree").path,
        config.tree("babel_dependency_tree").path,
        config.tree("code_tree").path,
    )
    agent_argv = (
        *_python_prefix(config, ACTION_MODULE),
        "agent",
        "--a-form-view",
        str(paths.a_form_view),
        "--a-form-labels",
        str(paths.a_form_labels),
        "--a-hold-view",
        str(paths.a_hold_view),
        "--action-output",
        str(paths.agent_action),
        "--policy-output",
        str(paths.agent_policy),
        "--receipt-output",
        str(paths.agent_receipt),
        "--encoder-model",
        str(config.tree("encoder_model_tree").path),
        "--encoder-model-sha256",
        config.encoder_model_semantic_sha256,
        "--device",
        "cuda:0",
    )
    raw_argv = (
        *_python_prefix(config, ACTION_MODULE),
        "raw",
        "--view",
        str(paths.a_hold_view),
        "--action-output",
        str(paths.raw_action),
    )
    hippo_argv = (
        str(config.file("official_python_executable").path),
        "-S",
        "-B",
        "-s",
        "-m",
        OFFICIAL_MODULE,
        "--input",
        str(paths.a_hold_view),
        "--action-output",
        str(paths.hippo_action),
        "--safe-receipt-output",
        str(paths.hippo_receipt),
        "--index-parent",
        str(paths.hippo_root / "indexes"),
        "--llm-model",
        str(config.tree("hippo_llm_model_tree").path),
        "--embedding-model",
        str(config.tree("encoder_model_tree").path),
    )
    hippo_environment = _lane_environment(
        config, paths.hippo_root, cuda_visible_devices="0"
    )
    hippo_environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(config.tree("code_tree").path),
            str(config.tree("official_python_dependency_tree").path),
            str(config.tree("babel_dependency_tree").path),
            str(config.tree("official_hipporag_tree").path),
            str(config.tree("official_overlay_dependency_tree").path),
            str(config.tree("official_base_dependency_tree").path),
        )
    )
    result = {
        "Agent": CommandSpec(
            name="Agent",
            argv=agent_argv,
            cwd=paths.agent_root,
            environment=_lane_environment(
                config, paths.agent_root, cuda_visible_devices="1"
            ),
            read_paths=(
                *shared,
                config.tree("encoder_model_tree").path,
                paths.a_form_view,
                paths.a_form_labels,
                paths.a_hold_view,
            ),
            write_paths=(paths.agent_root,),
            device_paths=_gpu_device_paths("1"),
        ),
        "RAW": CommandSpec(
            name="RAW",
            argv=raw_argv,
            cwd=paths.raw_root,
            environment=_lane_environment(
                config, paths.raw_root, cuda_visible_devices=""
            ),
            read_paths=(*shared, paths.a_hold_view),
            write_paths=(paths.raw_root,),
        ),
        "HippoRAG": CommandSpec(
            name="HippoRAG",
            argv=hippo_argv,
            cwd=paths.hippo_root,
            environment=hippo_environment,
            read_paths=(
                *_existing_system_read_paths(),
                config.file("official_python_executable").path,
                config.tree("python_runtime_tree").path,
                config.tree("code_tree").path,
                config.tree("official_python_dependency_tree").path,
                config.tree("babel_dependency_tree").path,
                config.tree("official_hipporag_tree").path,
                config.tree("official_overlay_dependency_tree").path,
                config.tree("official_base_dependency_tree").path,
                config.tree("encoder_model_tree").path,
                config.tree("hippo_llm_model_tree").path,
                paths.a_hold_view,
            ),
            write_paths=(paths.hippo_root,),
            device_paths=_gpu_device_paths("0"),
        ),
    }
    _validate_action_command_isolation(result, paths)
    return result


def _production_scorer_command(
    config: FormalConfig,
    paths: FormalPaths,
    source: SourceArtifacts,
) -> CommandSpec:
    del source
    argv = (
        *_python_prefix(config, SCORER_MODULE),
        "--action-view-pack",
        str(paths.a_hold_view),
        "--minimal-label-pack",
        str(paths.scorer_labels),
        "--agent-action-pack",
        str(paths.agent_action),
        "--raw-action-pack",
        str(paths.raw_action),
        "--hipporag-action-pack",
        str(paths.hippo_action),
        "--private-score-output",
        str(paths.score_private),
        "--safe-receipt-output",
        str(paths.score_safe),
        "--terminal-output",
        str(paths.score_terminal),
    )
    shared = (
        *_existing_system_read_paths(),
        config.file("python_executable").path,
        config.tree("python_runtime_tree").path,
        config.tree("python_dependency_tree").path,
        config.tree("babel_dependency_tree").path,
        config.tree("code_tree").path,
    )
    return CommandSpec(
        name="scorer",
        argv=argv,
        cwd=paths.scorer_root,
        environment=_lane_environment(
            config, paths.scorer_root, cuda_visible_devices=""
        ),
        read_paths=(
            *shared,
            paths.a_hold_view,
            paths.scorer_labels,
            paths.agent_action,
            paths.raw_action,
            paths.hippo_action,
            paths.barrier,
        ),
        write_paths=(paths.scorer_root,),
    )


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _validate_action_command_isolation(
    commands: Mapping[str, CommandSpec],
    paths: FormalPaths,
) -> None:
    if set(commands) != {"Agent", "RAW", "HippoRAG"}:
        raise WikiSQLUAOFormalError("three-arm command registry drifted")
    for name, command in commands.items():
        if command.name != name or not command.argv:
            raise WikiSQLUAOFormalError("action command identity drifted")
        serialized = "\0".join(
            (
                *command.argv,
                *(
                    f"{key}={value}"
                    for key, value in sorted(command.environment.items())
                ),
            )
        )
        if str(paths.a_hold_labels) in serialized:
            raise WikiSQLUAOFormalError(
                "A_hold labels leaked into an action command"
            )
        if any(
            _path_contains(path, paths.a_hold_labels)
            for path in command.read_paths
        ):
            raise WikiSQLUAOFormalError(
                "A_hold labels are reachable in an action Landlock profile"
            )
    if (
        commands["Agent"].environment.get("CUDA_VISIBLE_DEVICES") != "1"
        or commands["HippoRAG"].environment.get("CUDA_VISIBLE_DEVICES") != "0"
        or commands["RAW"].environment.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise WikiSQLUAOFormalError("physical GPU lane assignment drifted")


def _prepare_lane_root(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        (path / "home").mkdir(mode=0o700)
        (path / "tmp").mkdir(mode=0o700)
    except OSError as exc:
        raise WikiSQLUAOFormalError("fresh lane root cannot be created") from exc


def _command_commitment(command: CommandSpec) -> str:
    return semantic_sha256(
        {
            "argv": list(command.argv),
            "cwd": str(command.cwd),
            "environment": dict(sorted(command.environment.items())),
            "name": command.name,
            "read_paths": sorted(str(path) for path in command.read_paths),
            "write_paths": sorted(str(path) for path in command.write_paths),
            "device_paths": sorted(str(path) for path in command.device_paths),
        }
    )


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
    except OSError as exc:
        raise WikiSQLUAOFormalError("lane log cannot be created") from exc
    return descriptor


def _launch_one(
    command: CommandSpec,
    *,
    child_landlock: Callable[..., None],
) -> subprocess.Popen[bytes]:
    stdout_path = command.cwd / "stdout.log"
    stderr_path = command.cwd / "stderr.log"
    stdout = _open_log(stdout_path)
    stderr = _open_log(stderr_path)

    def isolate() -> None:
        child_landlock(
            read_paths=command.read_paths,
            write_paths=command.write_paths,
            device_paths=command.device_paths,
        )

    try:
        process = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env=dict(command.environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            preexec_fn=isolate,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WikiSQLUAOFormalError(
            f"{command.name} child could not be launched"
        ) from exc
    finally:
        os.close(stdout)
        os.close(stderr)
    return process


def _launch_actions_concurrently(
    commands: Mapping[str, CommandSpec],
    *,
    child_landlock: Callable[..., None],
    on_launch: Callable[[], None],
) -> Mapping[str, int]:
    processes: dict[str, subprocess.Popen[bytes]] = {}
    launch_error: Exception | None = None
    for name in ("Agent", "RAW", "HippoRAG"):
        try:
            processes[name] = _launch_one(
                commands[name], child_landlock=child_landlock
            )
            on_launch()
        except Exception as exc:  # preserve already-launched children
            launch_error = exc
            break
    statuses: dict[str, int] = {}
    for name, process in processes.items():
        statuses[name] = process.wait()
    if launch_error is not None:
        raise WikiSQLUAOFormalError(
            "three-arm concurrent submission was incomplete"
        ) from launch_error
    if set(processes) != {"Agent", "RAW", "HippoRAG"}:
        raise WikiSQLUAOFormalError(
            "three-arm concurrent submission was incomplete"
        )
    return statuses


@dataclass(frozen=True, slots=True)
class ActionArtifacts:
    file_sha256: Mapping[str, str]
    pack_self_sha256: Mapping[str, str]
    item_id_set_sha256: str


def _verify_common_actions(
    paths: FormalPaths,
    source: SourceArtifacts,
) -> ActionArtifacts:
    action_paths = {
        "Agent": paths.agent_action,
        "RAW": paths.raw_action,
        "HippoRAG": paths.hippo_action,
    }
    file_hashes: dict[str, str] = {}
    pack_hashes: dict[str, str] = {}
    identifiers: dict[str, tuple[str, ...]] = {}
    for arm, path in action_paths.items():
        file_hashes[arm] = _durable_file_sha256(
            path, f"{arm} common action pack"
        )
        value = _load_canonical_json(
            path,
            mode=0o600,
            field=f"{arm} common action pack",
        )
        try:
            rows = action_runtime.decode_action_pack(
                value,
                expected_block="A_hold",
                expected_arm=arm,
                expected_action_view_pack_sha256=(
                    source.a_hold_view_self_sha256
                ),
            )
        except action_runtime.WikiSQLUAOActionRuntimeError as exc:
            raise WikiSQLUAOFormalError(
                f"{arm} common action pack drifted"
            ) from exc
        if len(rows) != FORMAL_ITEM_COUNT:
            raise WikiSQLUAOFormalError(
                f"{arm} common action count drifted"
            )
        pack_hashes[arm] = _hex64(
            value.get("self_sha256"), f"{arm} action pack"
        )
        identifiers[arm] = tuple(
            str(row["opaque_item_id"]) for row in rows
        )
    if not (
        identifiers["Agent"]
        == identifiers["RAW"]
        == identifiers["HippoRAG"]
    ):
        raise WikiSQLUAOFormalError(
            "three common action packs do not share item IDs"
        )
    for path, field in (
        (paths.agent_policy, "compiled Agent policy"),
        (paths.agent_receipt, "Agent safe receipt"),
        (paths.hippo_receipt, "HippoRAG safe receipt"),
    ):
        _durable_file_sha256(path, field)
    return ActionArtifacts(
        file_sha256=file_hashes,
        pack_self_sha256=pack_hashes,
        item_id_set_sha256=semantic_sha256(list(identifiers["Agent"])),
    )


def _safe_recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_safe_recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_safe_recursive_keys(child))
    return keys


@dataclass(frozen=True, slots=True)
class ScorerArtifacts:
    terminal: Mapping[str, object]
    terminal_file_sha256: str
    safe_receipt_file_sha256: str
    private_score_file_sha256: str


def _verify_scorer_outputs(paths: FormalPaths) -> ScorerArtifacts:
    private_file = _durable_file_sha256(
        paths.score_private, "private score pack"
    )
    safe_file = _durable_file_sha256(
        paths.score_safe, "safe aggregate score receipt"
    )
    terminal_file = _durable_file_sha256(
        paths.score_terminal, "scorer safe terminal"
    )
    terminal = _load_canonical_json(
        paths.score_terminal,
        mode=0o600,
        field="scorer safe terminal",
    )
    expected_keys = {
        "block",
        "primary_passed",
        "private_score_file_sha256",
        "private_score_pack_sha256",
        "safe_aggregate_file_sha256",
        "safe_aggregate_receipt_sha256",
        "schema",
        "self_sha256",
        "status",
        "study_id",
    }
    supplied = _hex64(terminal.get("self_sha256"), "scorer terminal")
    terminal_private_self = _hex64(
        terminal.get("private_score_pack_sha256"),
        "scorer terminal private score pack",
    )
    terminal_safe_self = _hex64(
        terminal.get("safe_aggregate_receipt_sha256"),
        "scorer terminal safe aggregate receipt",
    )
    body = {
        key: child
        for key, child in terminal.items()
        if key != "self_sha256"
    }
    if (
        set(terminal) != expected_keys
        or semantic_sha256(body) != supplied
        or terminal.get("schema")
        != "wikisql_uao_scorer_v1_safe_terminal_v1"
        or terminal.get("study_id") != STUDY_ID
        or terminal.get("block") != "A_hold"
        or terminal.get("status") != "completed"
        or type(terminal.get("primary_passed")) is not bool
        or terminal.get("private_score_file_sha256") != private_file
        or terminal.get("safe_aggregate_file_sha256") != safe_file
        or _safe_recursive_keys(terminal) & _FORBIDDEN_SAFE_KEYS
    ):
        raise WikiSQLUAOFormalError("scorer safe terminal drifted")
    safe_receipt = _load_canonical_json(
        paths.score_safe,
        mode=0o600,
        field="scorer safe aggregate receipt",
    )
    safe_self = _hex64(
        safe_receipt.get("self_sha256"), "safe aggregate score receipt"
    )
    safe_body = {
        key: child
        for key, child in safe_receipt.items()
        if key != "self_sha256"
    }
    safe_keys = frozenset(
        {
            "Agent_vs_HippoRAG",
            "Agent_vs_RAW",
            "alpha_denominator",
            "alpha_numerator",
            "block",
            "family_counts",
            "input_commitments",
            "item_count",
            "offline_aggregate_primary_call_count",
            "online_evaluation_count",
            "primary_passed",
            "private_score_pack_sha256",
            "schema",
            "self_sha256",
            "status",
            "study_id",
        }
    )
    commitment_keys = frozenset(
        {
            "Agent_action_pack_sha256",
            "HippoRAG_action_pack_sha256",
            "RAW_action_pack_sha256",
            "action_view_pack_sha256",
            "minimal_label_pack_sha256",
        }
    )
    comparison_keys = frozenset(
        {
            "baseline",
            "exact_p_denominator",
            "exact_p_numerator",
            "family_net_u",
            "nonzero_pair_count",
            "observed_net_u",
            "passed",
        }
    )
    family_keys = frozenset(reality.FAMILY_ORDER)
    family_counts = _exact_dict(
        safe_receipt.get("family_counts"),
        family_keys,
        "safe aggregate family counts",
    )
    input_commitments = _exact_dict(
        safe_receipt.get("input_commitments"),
        commitment_keys,
        "safe aggregate input commitments",
    )

    def validate_comparison(key: str, baseline: str) -> bool:
        comparison = _exact_dict(
            safe_receipt.get(key),
            comparison_keys,
            f"safe aggregate {key}",
        )
        family_net = _exact_dict(
            comparison.get("family_net_u"),
            family_keys,
            f"safe aggregate {key} family net",
        )
        observed = comparison.get("observed_net_u")
        nonzero = comparison.get("nonzero_pair_count")
        numerator = comparison.get("exact_p_numerator")
        denominator = comparison.get("exact_p_denominator")
        if (
            comparison.get("baseline") != baseline
            or type(observed) is not int
            or not -FORMAL_ITEM_COUNT * 6
            <= observed
            <= FORMAL_ITEM_COUNT * 6
            or any(
                type(family_net[family]) is not int
                or not -24 * 6 <= family_net[family] <= 24 * 6
                for family in reality.FAMILY_ORDER
            )
            or sum(family_net.values()) != observed
            or type(nonzero) is not int
            or not 0 <= nonzero <= FORMAL_ITEM_COUNT
            or type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
            or not 0 <= numerator <= denominator
            or (1 << nonzero) % denominator != 0
            or type(comparison.get("passed")) is not bool
        ):
            return False
        expected_pass = (
            observed > 0
            and 10 * numerator <= denominator
            and all(
                family_net[family] > 0
                for family in reality.FAMILY_ORDER
            )
        )
        return comparison["passed"] is expected_pass

    safe_primary = safe_receipt.get("primary_passed")
    raw_valid = validate_comparison("Agent_vs_RAW", "raw")
    hippo_valid = validate_comparison(
        "Agent_vs_HippoRAG", "hipporag"
    )
    if (
        set(safe_receipt) != safe_keys
        or semantic_sha256(safe_body) != safe_self
        or safe_receipt.get("schema")
        != "wikisql_uao_scorer_v1_safe_aggregate_receipt_v1"
        or safe_receipt.get("study_id") != STUDY_ID
        or safe_receipt.get("block") != "A_hold"
        or safe_receipt.get("item_count") != FORMAL_ITEM_COUNT
        or safe_receipt.get("alpha_numerator") != 1
        or safe_receipt.get("alpha_denominator") != 10
        or safe_receipt.get("offline_aggregate_primary_call_count") != 1
        or safe_receipt.get("online_evaluation_count") != 0
        or type(safe_primary) is not bool
        or safe_primary != terminal.get("primary_passed")
        or safe_receipt.get("status")
        != (
            "PASS_REALITY_PRIMARY"
            if safe_primary
            else "FAIL_REALITY_PRIMARY"
        )
        or safe_receipt.get("private_score_pack_sha256")
        != terminal_private_self
        or terminal_safe_self != safe_self
        or any(
            family_counts[family] != 24
            for family in reality.FAMILY_ORDER
        )
        or any(
            _HEX64.fullmatch(input_commitments[key]) is None
            if isinstance(input_commitments[key], str)
            else True
            for key in commitment_keys
        )
        or not raw_valid
        or not hippo_valid
        or safe_primary
        is not (
            safe_receipt["Agent_vs_RAW"]["passed"]  # type: ignore[index]
            and safe_receipt["Agent_vs_HippoRAG"]["passed"]  # type: ignore[index]
        )
        or _safe_recursive_keys(safe_receipt) & _FORBIDDEN_SAFE_KEYS
    ):
        raise WikiSQLUAOFormalError(
            "scorer safe aggregate receipt drifted"
        )
    return ScorerArtifacts(
        terminal=terminal,
        terminal_file_sha256=terminal_file,
        safe_receipt_file_sha256=safe_file,
        private_score_file_sha256=private_file,
    )


@dataclass(slots=True)
class RunState:
    stage: str = "load_config"
    attempt_claimed: bool = False
    source_compiler_invocation_count: int = 0
    action_child_launch_count: int = 0
    action_barrier_count: int = 0
    a_hold_label_projection_count: int = 0
    scorer_launch_count: int = 0


@dataclass(frozen=True, slots=True)
class Dependencies:
    service_probe: Callable[[FormalConfig], ServiceAttestation]
    gpu_probe: Callable[[FormalConfig], GPUAttestation]
    abi_probe: Callable[[], int]
    outer_landlock: Callable[[FormalConfig, FormalPaths], None]
    child_landlock: Callable[..., None]
    source_compile: SourceCompile
    action_commands: CommandBuilder
    label_projector: LabelProjector
    scorer_command: Callable[
        [FormalConfig, FormalPaths, SourceArtifacts],
        CommandSpec,
    ]


PRODUCTION_DEPENDENCIES = Dependencies(
    service_probe=_systemctl_attestation,
    gpu_probe=_gpu_attestation,
    abi_probe=landlock_abi_version,
    outer_landlock=_outer_landlock,
    child_landlock=apply_landlock,
    source_compile=_compile_source_production,
    action_commands=_production_action_commands,
    label_projector=_project_minimal_labels_production,
    scorer_command=_production_scorer_command,
)


def _failure_terminal(
    *,
    config: FormalConfig | None,
    state: RunState,
    error: BaseException,
) -> Mapping[str, object]:
    paths = FormalPaths.for_root(FORMAL_ROOT)
    if paths.terminal.exists() or paths.terminal.is_symlink():
        raise WikiSQLUAOFormalError(
            "formal terminal already exists; retry is forbidden"
        ) from error
    fingerprint = hashlib.sha256(
        f"{type(error).__name__}:{error}".encode("utf-8", errors="replace")
    ).hexdigest()
    terminal = _self_hashed(
        {
            "API_or_online_evaluation_count": 0,
            "a_hold_label_projection_count": (
                state.a_hold_label_projection_count
            ),
            "action_barrier_count": state.action_barrier_count,
            "action_child_launch_count": state.action_child_launch_count,
            "aggregate_only_public_receipt": True,
            "attempt_claimed": state.attempt_claimed,
            "config_self_sha256": (
                config.self_sha256 if config is not None else None
            ),
            "failure_fingerprint_sha256": fingerprint,
            "retry_replay_resample_or_fallback_count": 0,
            "schema": FAILURE_SCHEMA,
            "scorer_launch_count": state.scorer_launch_count,
            "source_compiler_invocation_count": (
                state.source_compiler_invocation_count
            ),
            "stage": state.stage,
            "status": "formal_failed_no_retry_efficacy_unknown",
            "study_id": STUDY_ID,
        }
    )
    _write_once(paths.terminal, terminal, mode=0o600)
    return terminal


def _claim_attempt(
    config: FormalConfig,
    paths: FormalPaths,
    service: ServiceAttestation,
    gpu: GPUAttestation,
    abi: int,
) -> tuple[Mapping[str, object], str]:
    config_file_sha256 = _file_sha256(config.path)[0]
    attempt = _self_hashed(
        {
            "config_file_sha256": config_file_sha256,
            "config_self_sha256": config.self_sha256,
            "design_self_sha256": config.design_self_sha256,
            "gpu_UUID_binding_sha256": semantic_sha256(
                dict(sorted(gpu.uuids.items()))
            ),
            "invocation_id_sha256": hashlib.sha256(
                service.invocation_id.encode("ascii")
            ).hexdigest(),
            "landlock_abi": abi,
            "nrestarts": service.nrestarts,
            "schema": ATTEMPT_SCHEMA,
            "source_compiler_invocation_limit": 1,
            "status": "claimed_once",
            "study_id": STUDY_ID,
        }
    )
    file_sha = _write_once(paths.attempt, attempt, mode=0o600)
    return attempt, file_sha


def _write_live(
    config: FormalConfig,
    paths: FormalPaths,
    service: ServiceAttestation,
    attempt: Mapping[str, object],
    attempt_file_sha256: str,
) -> tuple[Mapping[str, object], str]:
    live = _self_hashed(
        {
            "attempt_file_sha256": attempt_file_sha256,
            "attempt_self_sha256": attempt["self_sha256"],
            "config_self_sha256": config.self_sha256,
            "invocation_id_sha256": hashlib.sha256(
                service.invocation_id.encode("ascii")
            ).hexdigest(),
            "nrestarts": service.nrestarts,
            "schema": LIVE_SCHEMA,
            "status": "live_unique_attempt",
            "study_id": STUDY_ID,
        }
    )
    return live, _write_once(paths.live, live, mode=0o600)


def _validate_scorer_command(command: CommandSpec, paths: FormalPaths) -> None:
    if command.name != "scorer" or not command.argv:
        raise WikiSQLUAOFormalError("scorer command identity drifted")
    if str(paths.a_hold_labels) in "\0".join(command.argv):
        raise WikiSQLUAOFormalError(
            "source A_hold label pack leaked into scorer command"
        )
    if not any(
        path.resolve(strict=False) == paths.scorer_labels.resolve(strict=False)
        for path in command.read_paths
    ):
        raise WikiSQLUAOFormalError(
            "scorer minimal label pack is outside its Landlock profile"
        )
    if any(
        _path_contains(path, paths.a_hold_labels)
        for path in command.read_paths
    ):
        raise WikiSQLUAOFormalError(
            "source A_hold labels are reachable by scorer"
        )
    if command.environment.get("CUDA_VISIBLE_DEVICES") != "":
        raise WikiSQLUAOFormalError("offline scorer unexpectedly sees a GPU")


def _run_with_dependencies(
    config_path: Path,
    dependencies: Dependencies,
) -> Mapping[str, object]:
    paths = FormalPaths.for_root(FORMAL_ROOT)
    state = RunState()
    config: FormalConfig | None = None
    if paths.terminal.exists() or paths.terminal.is_symlink():
        raise WikiSQLUAOFormalError(
            "formal terminal already exists; retry is forbidden"
        )
    if paths.attempt.exists() or paths.attempt.is_symlink():
        raise WikiSQLUAOFormalError(
            "formal attempt already exists; retry is forbidden"
        )
    try:
        state.stage = "load_content_addressed_config"
        config = load_config(config_path)

        state.stage = "verify_pre_attempt_bindings"
        service, gpu, abi = _verify_bindings(
            config,
            paths,
            dependencies.service_probe,
            dependencies.gpu_probe,
            dependencies.abi_probe,
        )

        state.stage = "install_outer_landlock"
        dependencies.outer_landlock(config, paths)

        state.stage = "claim_unique_attempt"
        attempt, attempt_file_sha = _claim_attempt(
            config, paths, service, gpu, abi
        )
        state.attempt_claimed = True

        state.stage = "write_outer_live_receipt"
        live, live_file_sha = _write_live(
            config, paths, service, attempt, attempt_file_sha
        )

        state.stage = "verify_formal_source_binding_after_attempt"
        config.file("source_archive").verify(
            "file binding source_archive"
        )

        state.stage = "compile_source_once"
        state.source_compiler_invocation_count += 1
        output_hashes = dependencies.source_compile(config, paths)
        source = _verify_source_outputs(config, paths, output_hashes)

        state.stage = "prepare_fresh_lane_roots"
        for lane_root in (
            paths.agent_root,
            paths.raw_root,
            paths.hippo_root,
            paths.scorer_root,
        ):
            _prepare_lane_root(lane_root)

        state.stage = "freeze_three_action_intents"
        commands = dependencies.action_commands(config, paths, source)
        _validate_action_command_isolation(commands, paths)
        intent = _self_hashed(
            {
                "A_hold_label_available_to_action_children": False,
                "all_logical_actions_submitted_before_wait": True,
                "command_commitments": {
                    name: _command_commitment(commands[name])
                    for name in ("Agent", "RAW", "HippoRAG")
                },
                "physical_GPU_assignments": {
                    "Agent": "GPU1",
                    "HippoRAG": "GPU0",
                    "RAW": "CPU",
                },
                "schema": INTENT_SCHEMA,
                "status": "three_action_intents_frozen",
                "study_id": STUDY_ID,
            }
        )
        intent_file_sha = _write_once(paths.intent, intent, mode=0o600)

        state.stage = "launch_three_actions_concurrently"

        def launched() -> None:
            state.action_child_launch_count += 1

        statuses = _launch_actions_concurrently(
            commands,
            child_landlock=dependencies.child_landlock,
            on_launch=launched,
        )
        if statuses != {"Agent": 0, "RAW": 0, "HippoRAG": 0}:
            raise WikiSQLUAOFormalError(
                "one or more action children failed; retry is forbidden"
            )

        state.stage = "validate_and_durably_seal_common_actions"
        actions = _verify_common_actions(paths, source)
        barrier = _self_hashed(
            {
                (
                    "A_hold_label_release_to_action_or_scorer_count_"
                    "before_barrier"
                ): 0,
                "action_file_sha256": dict(actions.file_sha256),
                "action_item_id_set_sha256": actions.item_id_set_sha256,
                "action_pack_self_sha256": dict(
                    actions.pack_self_sha256
                ),
                "all_three_actions_durable": True,
                "intent_file_sha256": intent_file_sha,
                "intent_self_sha256": intent["self_sha256"],
                "schema": BARRIER_SCHEMA,
                "status": "three_common_action_packs_sealed",
                "study_id": STUDY_ID,
            }
        )
        barrier_file_sha = _write_once(
            paths.barrier, barrier, mode=0o600
        )
        state.action_barrier_count = 1

        state.stage = "post_barrier_project_minimal_A_hold_labels"
        projected_label_file_sha = dependencies.label_projector(paths, source)
        _hex64(projected_label_file_sha, "projected label file")
        state.a_hold_label_projection_count = 1

        state.stage = "launch_independent_offline_scorer"
        scorer_command = dependencies.scorer_command(config, paths, source)
        _validate_scorer_command(scorer_command, paths)
        scorer_process = _launch_one(
            scorer_command,
            child_landlock=dependencies.child_landlock,
        )
        state.scorer_launch_count = 1
        scorer_status = scorer_process.wait()
        if scorer_status != 0:
            raise WikiSQLUAOFormalError(
                "offline scorer failed; retry is forbidden"
            )

        state.stage = "validate_scorer_and_write_safe_terminal"
        scorer = _verify_scorer_outputs(paths)
        terminal = _self_hashed(
            {
                "API_or_online_evaluation_count": 0,
                "a_hold_label_opened_only_after_action_barrier": True,
                "a_hold_minimal_label_file_sha256": (
                    projected_label_file_sha
                ),
                "action_barrier_file_sha256": barrier_file_sha,
                "action_barrier_self_sha256": barrier["self_sha256"],
                "action_child_launch_count": (
                    state.action_child_launch_count
                ),
                "aggregate_only_public_receipt": True,
                "attempt_file_sha256": attempt_file_sha,
                "attempt_self_sha256": attempt["self_sha256"],
                "config_self_sha256": config.self_sha256,
                "live_file_sha256": live_file_sha,
                "live_self_sha256": live["self_sha256"],
                "nrestarts": service.nrestarts,
                "primary_passed": scorer.terminal["primary_passed"],
                "retry_replay_resample_or_fallback_count": 0,
                "schema": TERMINAL_SCHEMA,
                "scorer_safe_aggregate_file_sha256": (
                    scorer.safe_receipt_file_sha256
                ),
                "scorer_safe_terminal_file_sha256": (
                    scorer.terminal_file_sha256
                ),
                "scorer_safe_terminal_self_sha256": scorer.terminal[
                    "self_sha256"
                ],
                "source_compiler_invocation_count": (
                    state.source_compiler_invocation_count
                ),
                "source_compiler_receipt_self_sha256": (
                    source.compiler_receipt_self_sha256
                ),
                "status": "completed_protocol_valid",
                "study_id": STUDY_ID,
                "three_common_action_pack_self_sha256": dict(
                    actions.pack_self_sha256
                ),
            }
        )
        if _safe_recursive_keys(terminal) & _FORBIDDEN_SAFE_KEYS:
            raise WikiSQLUAOFormalError(
                "outer terminal contains private fields"
            )
        _write_once(paths.terminal, terminal, mode=0o600)
        return terminal
    except Exception as error:
        return _failure_terminal(
            config=config,
            state=state,
            error=error,
        )


def run_formal_production(config_path: Path) -> Mapping[str, object]:
    return _run_with_dependencies(config_path, PRODUCTION_DEPENDENCIES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    terminal = run_formal_production(arguments.config)
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if terminal.get("status") == "completed_protocol_valid" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTION_MODULE",
    "BARRIER_SCHEMA",
    "CONFIG_SCHEMA",
    "CommandSpec",
    "Dependencies",
    "FAILURE_SCHEMA",
    "FORMAL_ROOT",
    "FileBinding",
    "FormalConfig",
    "FormalPaths",
    "GPUAttestation",
    "OFFICIAL_MODULE",
    "PRODUCTION_DEPENDENCIES",
    "SCORER_MODULE",
    "STUDY_ID",
    "ServiceAttestation",
    "SourceArtifacts",
    "TERMINAL_SCHEMA",
    "TreeBinding",
    "UNIT_NAME",
    "VERSION",
    "WikiSQLUAOFormalError",
    "_run_with_dependencies",
    "apply_landlock",
    "canonical_json_bytes",
    "landlock_abi_version",
    "load_config",
    "main",
    "run_formal_production",
    "semantic_sha256",
    "tree_identity",
]
