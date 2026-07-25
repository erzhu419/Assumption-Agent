"""Production, no-network MMQA P1 formal-action runtime.

This module is the concrete process/runtime layer below the remote outer
lifecycle and above :mod:`mmqa_p1_formal_controller_v1`.  It is intended to
run as a transient user-systemd child whose effective address-family policy
allows only ``AF_UNIX``.  It live-verifies the implementation and execution
freezes, the already-completed local synthetic runtime receipt, and the fresh
official-HippoRAG receipt exactly once before forming any formal action.

For each controller block, one private anonymous block is written and exactly
two fixed CLI workers are launched concurrently: MiniLM on ``cuda:0`` and the
cross encoder on ``cuda:1``.  Their mode-0600 coordinate archives are strictly
aligned before deterministic surface anchors are added.  A_hold is passed once
to the frozen four-worker CPU official-HippoRAG block, whose private terminal
archive is written, reread, and rebound before ordinals are returned.

The runtime never reads formal benchmark source files, gold packs, the trusted
source ledger, API credentials, or the parent environment.  It has no retry,
OOM resize, provider switch, online evaluator, or Ruoli fallback.  The final
outer wrapper contains only hashes, aggregate counts/status, and the
already-safe controller outcomes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import errno
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Protocol

from . import mmqa_p1_action_integration_v1 as integration
from . import mmqa_p1_block_coordinate_worker_v1 as coordinate_worker
from . import mmqa_p1_formal_controller_v1 as formal_controller
from . import mmqa_p1_local_action_executor_v1 as action_executor
from . import mmqa_p1_local_runtime_preflight_v1 as local_preflight
from . import mmqa_p1_official_hipporag_block_v1 as official_hippo
from . import mmqa_p1_remote_outer_lifecycle_v1 as remote_outer


VERSION = "mmqa_p1_formal_action_runtime_v1"
STUDY_ID = formal_controller.STUDY_ID
STUDY_DESIGN_SELF_SHA256 = formal_controller.STUDY_DESIGN_SELF_SHA256

RUNTIME_ROOT_RELATIVE = Path("artifacts/mmqa_p1_formal_action_runtime_v1")
PRIVATE_ROOT_DIRECTORY = "private"
CONTROLLER_ROOT_DIRECTORY = "controller"
ATTEMPT_FILENAME = "formal_action_runtime.one_shot.private.json"
FAILURE_FILENAME = "formal_action_runtime.terminal_failure.private.json"
WRAPPER_TERMINAL_FILENAME = "formal_action_runtime.safe_terminal.private.json"
WRAPPER_TERMINAL_RELATIVE = (
    RUNTIME_ROOT_RELATIVE / WRAPPER_TERMINAL_FILENAME
)
WRAPPER_SCHEMA = f"{VERSION}_safe_terminal_v1"
WRAPPER_STATUS = "formal_action_lifecycle_complete_controller_terminal_bound"

ANONYMOUS_BLOCK_FILENAME = "anonymous.block.private.json"
MINILM_ARCHIVE_FILENAME = "minilm.coordinates.private.json"
CROSS_ENCODER_ARCHIVE_FILENAME = "cross_encoder.coordinates.private.json"
HIPPORAG_TERMINAL_FILENAME = "official_hipporag.terminal.private.json"

COORDINATE_PROCESS_COUNT_PER_BLOCK = 2
COORDINATE_WORKER_TIMEOUT_SECONDS = 4 * 60 * 60
REQUIRED_ADDRESS_FAMILY_DENIAL_ERRNO = 97

SYSTEMD_PARENT_NETWORK_CONTRACT = (
    remote_outer.OUTER_NETWORK_ISOLATION_CONTRACT
)
SYSTEMD_EXECUTION_POLICY = remote_outer.EXECUTION_POLICY

RUNTIME_REQUIRED_IMPLEMENTATION_RELATIVES = frozenset(
    {
        "assumption_agent/benchmarks/mmqa_p1_action_integration_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_block_coordinate_worker_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_formal_action_runtime_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_formal_controller_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_local_action_executor_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_local_runtime_preflight_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_official_hipporag_block_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_private_selection_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_typed_proof_e5_core_v1.py",
        "tests/test_mmqa_p1_block_coordinate_worker_v1.py",
        "tests/test_mmqa_p1_formal_action_runtime_v1.py",
        "tests/test_mmqa_p1_formal_controller_v1.py",
        "tests/test_mmqa_p1_official_hipporag_block_v1.py",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_MAP_KEY = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_ALLOWED_CONTROLLER_STATUSES = frozenset(
    {
        "lifecycle_complete_promoted_M_scored",
        "lifecycle_complete_valid_nonpromotion_M_sealed",
    }
)
_FORBIDDEN_ENV_FRAGMENTS = (
    "API_KEY",
    "OPENAI",
    "RUOLI",
    "BEARER",
    "CREDENTIAL",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "PROXY",
)
EXECUTION_FREEZE_ARGUMENT_MARKER = (
    "__MMQA_P1_EXECUTION_FREEZE_SELF_SHA256__"
)
LOCAL_PREFLIGHT_ARGUMENT_MARKER = (
    "__MMQA_P1_LOCAL_PREFLIGHT_SELF_SHA256__"
)
SELECTION_ACQUISITION_ARGUMENT_MARKER = (
    "__MMQA_P1_SELECTION_ACQUISITION_SHA256__"
)
_DYNAMIC_ARGUMENT_MARKERS = {
    "--execution-freeze-self-sha256": (
        EXECUTION_FREEZE_ARGUMENT_MARKER
    ),
    "--local-preflight-self-sha256": LOCAL_PREFLIGHT_ARGUMENT_MARKER,
    "--selection-acquisition-sha256": (
        SELECTION_ACQUISITION_ARGUMENT_MARKER
    ),
}
_OFFICIAL_RUNTIME_ARGUMENT_FIELDS = {
    "--official-runtime-python": "runtime_python",
    "--official-pyvenv-cfg": "pyvenv_cfg",
    "--official-overlay-root": "overlay_root",
    "--official-hipporag-source-root": "hipporag_source_root",
    "--official-p16-site-root": "p16_site_root",
    "--official-local-llm-model": "local_llm_model",
    "--official-local-embedding-model": "local_embedding_model",
}
_OFFICIAL_RECEIPT_SHA_ARGUMENT = "--official-preflight-receipt-sha256"


class MmqaP1FormalActionRuntimeError(RuntimeError):
    """The concrete one-shot runtime failed closed."""


class ProcessRunner(Protocol):
    def __call__(self, command: Sequence[str], **kwargs: object) -> object: ...


class CoordinateExecutorFactory(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1FormalActionRuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256 = _semantic_hash(
    SYSTEMD_PARENT_NETWORK_CONTRACT
)


def _self_hashed(
    body: Mapping[str, Any], field: str = "self_sha256"
) -> dict[str, Any]:
    if field in body:
        raise MmqaP1FormalActionRuntimeError("self-hash field already exists")
    return {**dict(body), field: _semantic_hash(body)}


def _verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or _HEX64.fullmatch(claimed) is None:
        raise MmqaP1FormalActionRuntimeError("self-hash is absent or invalid")
    if not hmac.compare_digest(_semantic_hash(body), claimed):
        raise MmqaP1FormalActionRuntimeError("self-hash drifted")
    return claimed


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MmqaP1FormalActionRuntimeError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _absolute_lexical(path: str | Path, field: str) -> Path:
    value = Path(path).expanduser().absolute()
    text = str(value)
    if (
        not text.startswith("/")
        or text == "/"
        or text.endswith("/")
        or "\x00" in text
        or "//" in text
        or "/./" in text
        or "/../" in text
    ):
        raise MmqaP1FormalActionRuntimeError(
            f"{field} must be one normalized absolute lexical path"
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MmqaP1FormalActionRuntimeError(
                "runtime durable path is not a directory"
            )
        os.fsync(descriptor)
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "runtime directory cannot be synchronized"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _create_private_directory_once(path: Path) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise MmqaP1FormalActionRuntimeError(
            "private directory parent is unsafe"
        )
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "private directory exists or cannot be created"
        ) from exc
    _fsync_directory(path)
    _fsync_directory(path.parent)


def _write_once(path: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _canonical_bytes(value)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise MmqaP1FormalActionRuntimeError("runtime output parent is unsafe")
    staging = path.with_name(f".{path.name}.part")
    if (
        path.exists()
        or path.is_symlink()
        or staging.exists()
        or staging.is_symlink()
    ):
        raise MmqaP1FormalActionRuntimeError("runtime output already exists")
    descriptor = -1
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        os.unlink(staging)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "runtime output cannot be sealed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise MmqaP1FormalActionRuntimeError(
            "runtime output type or mode drifted"
        )
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode_octal": "0600",
    }


def _read_canonical_file(
    path: Path,
    *,
    label: str,
    required_mode: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    descriptor = -1
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                required_mode is not None
                and stat.S_IMODE(before.st_mode) != required_mode
            )
        ):
            raise MmqaP1FormalActionRuntimeError(
                f"{label} is not a sealed regular file"
            )
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MmqaP1FormalActionRuntimeError(
                f"{label} changed while read"
            )
    except FileNotFoundError as exc:
        raise MmqaP1FormalActionRuntimeError(f"{label} is unavailable") from exc
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(f"{label} cannot be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MmqaP1FormalActionRuntimeError(
            f"{label} is invalid JSON"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise MmqaP1FormalActionRuntimeError(f"{label} is not canonical")
    return value, raw


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _regular_file_sha256(path: Path) -> str:
    descriptor = -1
    digest = hashlib.sha256()
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise MmqaP1FormalActionRuntimeError(
                "bound runtime file is unavailable"
            )
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        opened = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise MmqaP1FormalActionRuntimeError(
                "bound runtime file changed while hashed"
            )
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "bound runtime file cannot be hashed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _validated_string_map(
    value: Mapping[str, str], field: str
) -> Mapping[str, str]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(
            not isinstance(key, str)
            or _SAFE_MAP_KEY.fullmatch(key) is None
            or not isinstance(item, str)
            or not item
            or "\x00" in item
            for key, item in value.items()
        )
    ):
        raise MmqaP1FormalActionRuntimeError(f"{field} registry drifted")
    return MappingProxyType(dict(sorted(value.items())))


@dataclass(frozen=True)
class FormalActionRuntimeConfig:
    project_root: Path
    execution_freeze_self_sha256: str
    implementation_freeze_self_sha256: str
    local_preflight_receipt: Path
    local_preflight_self_sha256: str
    typed_python: Path
    typed_python_resolved_sha256: str
    minilm_model: Path
    minilm_required_tree_sha256: str
    cross_encoder_model: Path
    cross_encoder_required_tree_sha256: str
    nvidia_smi: Path
    systemd_run: Path
    systemd_run_resolved_sha256: str
    systemd_isolation_disposition_sha256: str
    runtime_module_sha256: str
    official_preflight_receipt: Path
    official_preflight_receipt_sha256: str
    official_runtime_paths: official_hippo.FreshComparatorRuntimePaths
    official_expected_package_versions: Mapping[str, str]
    official_expected_module_import_roots: Mapping[str, str]
    selection_acquisition_sha256: str
    controller_arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "execution_freeze_self_sha256",
            "implementation_freeze_self_sha256",
            "local_preflight_self_sha256",
            "typed_python_resolved_sha256",
            "minilm_required_tree_sha256",
            "cross_encoder_required_tree_sha256",
            "systemd_run_resolved_sha256",
            "systemd_isolation_disposition_sha256",
            "runtime_module_sha256",
            "official_preflight_receipt_sha256",
            "selection_acquisition_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.minilm_required_tree_sha256
            != coordinate_worker.ROLE_REQUIRED_TREE_SHA256[
                coordinate_worker.ROLE_MINILM
            ]
            or self.cross_encoder_required_tree_sha256
            != coordinate_worker.ROLE_REQUIRED_TREE_SHA256[
                coordinate_worker.ROLE_CROSS_ENCODER
            ]
        ):
            raise MmqaP1FormalActionRuntimeError(
                "local model required-tree freeze drifted"
            )
        for field in (
            "project_root",
            "local_preflight_receipt",
            "typed_python",
            "minilm_model",
            "cross_encoder_model",
            "nvidia_smi",
            "systemd_run",
            "official_preflight_receipt",
        ):
            object.__setattr__(
                self,
                field,
                _absolute_lexical(getattr(self, field), field),
            )
        if not isinstance(
            self.official_runtime_paths,
            official_hippo.FreshComparatorRuntimePaths,
        ):
            raise MmqaP1FormalActionRuntimeError(
                "official comparator path binding drifted"
            )
        if (
            self.systemd_isolation_disposition_sha256
            != _semantic_hash(
                remote_outer._transient_unit_contract(  # noqa: SLF001
                    self.project_root
                )
            )
        ):
            raise MmqaP1FormalActionRuntimeError(
                "systemd isolation disposition freeze drifted"
            )
        object.__setattr__(
            self,
            "official_expected_package_versions",
            _validated_string_map(
                self.official_expected_package_versions,
                "official package version",
            ),
        )
        object.__setattr__(
            self,
            "official_expected_module_import_roots",
            _validated_string_map(
                self.official_expected_module_import_roots,
                "official module import-root",
            ),
        )
        arguments = tuple(self.controller_arguments)
        if any(
            not isinstance(value, str) or "\x00" in value
            for value in arguments
        ):
            raise MmqaP1FormalActionRuntimeError(
                "controller argument registry drifted"
            )
        object.__setattr__(self, "controller_arguments", arguments)
        controller_argument_template(self)


def controller_argument_template(
    config: FormalActionRuntimeConfig,
) -> tuple[str, ...]:
    """Restore the three post-freeze values to their unique freeze markers."""

    if not isinstance(config, FormalActionRuntimeConfig):
        raise MmqaP1FormalActionRuntimeError(
            "controller argument template config drifted"
        )
    arguments = list(config.controller_arguments)
    actual_values = {
        "--execution-freeze-self-sha256": (
            config.execution_freeze_self_sha256
        ),
        "--local-preflight-self-sha256": (
            config.local_preflight_self_sha256
        ),
        "--selection-acquisition-sha256": (
            config.selection_acquisition_sha256
        ),
    }
    for marker in _DYNAMIC_ARGUMENT_MARKERS.values():
        if any(marker in value for value in arguments):
            raise MmqaP1FormalActionRuntimeError(
                "controller argument marker appears in actual arguments"
            )
    for flag, field in _OFFICIAL_RUNTIME_ARGUMENT_FIELDS.items():
        positions = [
            index for index, value in enumerate(arguments) if value == flag
        ]
        expected = str(getattr(config.official_runtime_paths, field))
        if (
            len(positions) != 1
            or positions[0] + 1 >= len(arguments)
            or arguments[positions[0] + 1] != expected
        ):
            raise MmqaP1FormalActionRuntimeError(
                f"{flag} must bind the frozen official runtime path once"
            )
    receipt_sha_positions = [
        index
        for index, value in enumerate(arguments)
        if value == _OFFICIAL_RECEIPT_SHA_ARGUMENT
    ]
    if (
        len(receipt_sha_positions) != 1
        or receipt_sha_positions[0] + 1 >= len(arguments)
        or arguments[receipt_sha_positions[0] + 1]
        != config.official_preflight_receipt_sha256
    ):
        raise MmqaP1FormalActionRuntimeError(
            f"{_OFFICIAL_RECEIPT_SHA_ARGUMENT} must bind the frozen "
            "official receipt hash once"
        )
    for flag, marker in _DYNAMIC_ARGUMENT_MARKERS.items():
        positions = [
            index for index, value in enumerate(arguments) if value == flag
        ]
        if (
            len(positions) != 1
            or positions[0] + 1 >= len(arguments)
            or arguments[positions[0] + 1] != actual_values[flag]
        ):
            raise MmqaP1FormalActionRuntimeError(
                f"{flag} must occur once in split flag/value form"
            )
        arguments[positions[0] + 1] = marker
    if any(
        arguments.count(marker) != 1
        for marker in _DYNAMIC_ARGUMENT_MARKERS.values()
    ):
        raise MmqaP1FormalActionRuntimeError(
            "controller argument template marker cardinality drifted"
        )
    return tuple(arguments)


def _formal_action_transient_unit_contract(
    config: FormalActionRuntimeConfig,
) -> Mapping[str, object]:
    return remote_outer._transient_unit_contract(  # noqa: SLF001
        config.project_root
    )


def _official_runtime_contract(
    config: FormalActionRuntimeConfig,
) -> Mapping[str, object]:
    paths = config.official_runtime_paths
    return {
        "expected_module_import_roots": dict(
            sorted(config.official_expected_module_import_roots.items())
        ),
        "expected_package_versions": dict(
            sorted(config.official_expected_package_versions.items())
        ),
        "path_binding": paths.path_binding(),
        "paths": {
            field: getattr(paths, field)
            for field in _OFFICIAL_RUNTIME_ARGUMENT_FIELDS.values()
        },
    }


def _official_preflight_receipt_contract(
    config: FormalActionRuntimeConfig,
) -> Mapping[str, object]:
    try:
        relative = config.official_preflight_receipt.relative_to(
            config.project_root
        )
    except ValueError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "official preflight receipt is outside the project"
        ) from exc
    if not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise MmqaP1FormalActionRuntimeError(
            "official preflight receipt path is unsafe"
        )
    return {
        "relative_path": relative.as_posix(),
        "schema": official_hippo.FRESH_PREFLIGHT_SCHEMA,
        "self_hash_field": "receipt_sha256",
        "self_sha256": config.official_preflight_receipt_sha256,
        "status": (
            "passed_public_synthetic_candidate_only_fresh_runtime"
        ),
    }


def _project_relative(project: Path, path: Path, field: str) -> Path:
    try:
        resolved_project = project.resolve(strict=True)
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(resolved_project)
    except (OSError, ValueError) as exc:
        raise MmqaP1FormalActionRuntimeError(
            f"{field} is outside the project"
        ) from exc
    if path != resolved or any(part in {"", ".", ".."} for part in relative.parts):
        raise MmqaP1FormalActionRuntimeError(
            f"{field} traverses a symlink or unsafe path"
        )
    return relative


def verify_execution_and_implementation_freezes(
    config: FormalActionRuntimeConfig,
) -> Mapping[str, Any]:
    """Live-verify both frozen manifests and every listed implementation file."""

    if not isinstance(config, FormalActionRuntimeConfig):
        raise MmqaP1FormalActionRuntimeError("freeze config type drifted")
    try:
        project = config.project_root.resolve(strict=True)
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "formal project root is unavailable"
        ) from exc
    if config.project_root != project or not project.is_dir():
        raise MmqaP1FormalActionRuntimeError(
            "formal project root is symlinked or invalid"
        )
    module_path = Path(__file__).absolute()
    if _regular_file_sha256(module_path) != config.runtime_module_sha256:
        raise MmqaP1FormalActionRuntimeError(
            "formal action runtime file identity drifted"
        )
    module_relative = _project_relative(
        project, module_path, "formal action runtime module"
    )
    execution_spec = remote_outer.ReceiptSpec(
        name="execution_freeze",
        relative_path=remote_outer.EXECUTION_FREEZE_RELATIVE,
        expected_schema="mmqa_p1_execution_freeze_v1",
        expected_status="frozen_before_outer_one_shot",
        expected_self_sha256=config.execution_freeze_self_sha256,
    )
    implementation_spec = remote_outer.ReceiptSpec(
        name="implementation_freeze",
        relative_path=remote_outer.IMPLEMENTATION_FREEZE_RELATIVE,
        expected_schema="mmqa_p1_implementation_freeze_v1",
        expected_status="frozen_before_execution_freeze",
        expected_self_sha256=config.implementation_freeze_self_sha256,
    )
    try:
        implementation, implementation_binding = (
            remote_outer._load_receipt_value_and_binding(  # noqa: SLF001
                project, implementation_spec
            )
        )
        execution, execution_binding = (
            remote_outer._load_receipt_value_and_binding(  # noqa: SLF001
                project, execution_spec
            )
        )
        official_adapter = execution.get(
            "official_hipporag_adapter_relative_path"
        )
        if not isinstance(official_adapter, str):
            raise MmqaP1FormalActionRuntimeError(
                "execution freeze official adapter is absent"
            )
        inventory = remote_outer._verify_frozen_inventory(  # noqa: SLF001
            project,
            implementation,
            required_relatives=(
                module_relative.as_posix(),
                official_adapter,
                *sorted(RUNTIME_REQUIRED_IMPLEMENTATION_RELATIVES),
            ),
        )
        if official_adapter not in inventory:
            raise MmqaP1FormalActionRuntimeError(
                "official adapter is not implementation-frozen"
            )
        runtime_rows = execution.get("runtime_path_bindings")
        controller_contract = execution.get(
            "formal_controller_receipt_contract"
        )
        official_contract = execution.get(
            "official_hipporag_preflight_receipt_contract"
        )
        expected_runtime_rows = {
            "systemd_run": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        config.systemd_run
                    )
                ),
                "resolved_file_sha256": (
                    config.systemd_run_resolved_sha256
                ),
            },
            "controller_executable": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        config.typed_python
                    )
                ),
                "resolved_file_sha256": (
                    config.typed_python_resolved_sha256
                ),
            },
            "controller_module": {
                "file_sha256": config.runtime_module_sha256,
                "project_relative_path": module_relative.as_posix(),
            },
            "env_executable": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        remote_outer.ENV_PATH
                    )
                ),
                "resolved_file_sha256": _regular_file_sha256(
                    remote_outer.ENV_PATH.resolve(strict=True)
                ),
            },
            "cross_encoder_model": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        config.cross_encoder_model
                    )
                )
            },
            "minilm_model": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        config.minilm_model
                    )
                )
            },
            "nvidia_smi": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        config.nvidia_smi
                    )
                ),
                "resolved_file_sha256": _regular_file_sha256(
                    config.nvidia_smi.resolve(strict=True)
                ),
            },
            "typed_python": {
                "lexical_path_sha256": (
                    remote_outer._lexical_path_sha256(  # noqa: SLF001
                        config.typed_python
                    )
                ),
                "resolved_file_sha256": (
                    config.typed_python_resolved_sha256
                ),
            },
        }
        if (
            not isinstance(runtime_rows, Mapping)
            or set(runtime_rows) != set(expected_runtime_rows)
            or any(
                runtime_rows.get(name) != row
                for name, row in expected_runtime_rows.items()
            )
            or _regular_file_sha256(
                config.systemd_run.resolve(strict=True)
            )
            != config.systemd_run_resolved_sha256
            or execution.get("schema") != "mmqa_p1_execution_freeze_v1"
            or execution.get("status") != "frozen_before_outer_one_shot"
            or execution.get("study_id") != STUDY_ID
            or execution.get("study_design_self_sha256")
            != STUDY_DESIGN_SELF_SHA256
            or execution.get("source_custody_self_sha256")
            != remote_outer.EXPECTED_CUSTODY_SELF_SHA256
            or execution.get("download_authorization_self_sha256")
            != remote_outer.EXPECTED_AUTHORIZATION_SELF_SHA256
            or execution.get(
                "preexecution_runtime_disposition_self_sha256"
            )
            != remote_outer.EXPECTED_RUNTIME_DISPOSITION_SELF_SHA256
            or execution.get("stage_order")
            != list(remote_outer.STAGE_ORDER)
            or execution.get("implementation_freeze_self_sha256")
            != config.implementation_freeze_self_sha256
            or execution.get("controller_argument_template_sha256")
            != _semantic_hash(list(controller_argument_template(config)))
            or execution.get("execution_policy") != SYSTEMD_EXECUTION_POLICY
            or execution.get("outer_network_isolation_contract")
            != SYSTEMD_PARENT_NETWORK_CONTRACT
            or execution.get("formal_action_transient_unit_contract")
            != _formal_action_transient_unit_contract(config)
            or execution.get("formal_child_environment_sha256")
            != _semantic_hash(
                remote_outer._formal_child_environment(  # noqa: SLF001
                    project
                )
            )
            or execution.get(
                "source_acquisition_child_environment_sha256"
            )
            != _semantic_hash(
                remote_outer._source_acquisition_child_environment(  # noqa: SLF001
                    project
                )
            )
            or execution.get(
                "source_acquisition_transient_unit_contract"
            )
            != remote_outer._source_acquisition_transient_unit_contract(  # noqa: SLF001
                project
            )
            or execution.get("systemd_client_environment_sha256")
            != _semantic_hash(
                remote_outer._systemd_client_environment()  # noqa: SLF001
            )
            or execution.get("official_hipporag_runtime_contract")
            != _official_runtime_contract(config)
            or controller_contract
            != {
                "relative_path": WRAPPER_TERMINAL_RELATIVE.as_posix(),
                "schema": WRAPPER_SCHEMA,
                "status": WRAPPER_STATUS,
            }
            or official_contract
            != _official_preflight_receipt_contract(config)
        ):
            raise MmqaP1FormalActionRuntimeError(
                "systemd execution freeze contract drifted"
            )
    except (
        remote_outer.MMQAP1RemoteOuterLifecycleError,
        MmqaP1FormalActionRuntimeError,
    ) as exc:
        raise MmqaP1FormalActionRuntimeError(
            "implementation/execution freeze live verification failed"
        ) from exc
    return {
        "status": "implementation_and_execution_freezes_live_verified_once",
        "implementation_freeze_self_sha256": (
            config.implementation_freeze_self_sha256
        ),
        "implementation_freeze_file_sha256": (
            implementation_binding.file_sha256
        ),
        "execution_freeze_self_sha256": config.execution_freeze_self_sha256,
        "execution_freeze_file_sha256": execution_binding.file_sha256,
        "implementation_inventory_count": len(inventory),
        "runtime_module_sha256": config.runtime_module_sha256,
        "controller_argument_template_sha256": _semantic_hash(
            list(controller_argument_template(config))
        ),
        "formal_action_transient_unit_contract_sha256": _semantic_hash(
            _formal_action_transient_unit_contract(config)
        ),
        "outer_network_isolation_contract_sha256": (
            SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256
        ),
        "official_hipporag_runtime_contract_sha256": _semantic_hash(
            _official_runtime_contract(config)
        ),
        "verification_count": 1,
    }


def validate_local_preflight_once(
    config: FormalActionRuntimeConfig,
    *,
    isolation_inspector: Callable[
        [], object
    ] = local_preflight.production_address_family_isolation_probe,
) -> Mapping[str, Any]:
    """Validate the pinned local receipt and re-hash both local model assets."""

    expected_path = (
        config.project_root / remote_outer.PREFLIGHT_RECEIPT_RELATIVE
    ).absolute()
    if config.local_preflight_receipt != expected_path:
        raise MmqaP1FormalActionRuntimeError(
            "local preflight receipt escaped its frozen project path"
        )
    value, raw = _read_canonical_file(
        config.local_preflight_receipt,
        label="local runtime preflight receipt",
        required_mode=0o600,
    )
    observed = _verify_self_hash(value, "self_sha256")
    assets = value.get("asset_bindings")
    runtime = value.get("runtime_binding")
    boundary = value.get("claim_boundary")
    concurrency = value.get("concurrency")
    receipt_isolation = value.get("address_family_isolation_probe")
    runtime_identity = (
        runtime.get("typed_runtime_identity_sha256")
        if isinstance(runtime, Mapping)
        else None
    )
    if (
        not hmac.compare_digest(observed, config.local_preflight_self_sha256)
        or value.get("schema") != local_preflight.RECEIPT_SCHEMA
        or value.get("status")
        != "passed_public_synthetic_non_scoring_runtime_action_preflight"
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or not isinstance(receipt_isolation, Mapping)
        or receipt_isolation.get("address_family_isolation_contract")
        != local_preflight.ADDRESS_FAMILY_ISOLATION_CONTRACT
        or value.get("address_family_isolation_probe_sha256")
        != _semantic_hash(receipt_isolation)
        or not isinstance(assets, Mapping)
        or assets.get("minilm_required_tree_sha256")
        != config.minilm_required_tree_sha256
        or assets.get("cross_encoder_required_tree_sha256")
        != config.cross_encoder_required_tree_sha256
        or not isinstance(runtime, Mapping)
        or not isinstance(runtime_identity, str)
        or _HEX64.fullmatch(runtime_identity) is None
        or runtime.get("typed_runtime_lexical_path_sha256")
        != hashlib.sha256(
            os.fsencode(str(config.typed_python))
        ).hexdigest()
        or not isinstance(boundary, Mapping)
        or any(
            boundary.get(key) != 0
            for key in (
                "api_or_provider_call_count",
                "formal_HippoRAG_call_count",
                "formal_MMQA_source_or_row_access_count",
                "label_or_score_access_count",
                "online_evaluator_call_count",
                "retry_replay_or_resample_count",
            )
        )
        or concurrency
        != {
            "cross_encoder_physical_gpu_1_process_cap": 1,
            "minilm_physical_gpu_0_process_cap": 1,
            "model_process_co_residency": False,
            "typed_gpu_process_cap": 2,
        }
    ):
        raise MmqaP1FormalActionRuntimeError(
            "local runtime preflight receipt drifted"
        )
    if not callable(isolation_inspector):
        raise MmqaP1FormalActionRuntimeError(
            "local address-family isolation inspector drifted"
        )
    try:
        current_isolation = isolation_inspector()
    except Exception as exc:
        raise MmqaP1FormalActionRuntimeError(
            "local address-family isolation revalidation failed"
        ) from exc
    if (
        not isinstance(current_isolation, Mapping)
        or dict(current_isolation) != dict(receipt_isolation)
    ):
        raise MmqaP1FormalActionRuntimeError(
            "local address-family isolation changed after preflight"
        )
    try:
        typed = local_preflight._verify_typed_python(  # noqa: SLF001
            config.typed_python
        )
        minilm = local_preflight._verify_minilm_asset(  # noqa: SLF001
            config.minilm_model
        )
        cross_encoder = local_preflight._verify_ce_asset(  # noqa: SLF001
            config.cross_encoder_model
        )
    except local_preflight.MmqaP1LocalRuntimePreflightError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "local runtime or model live binding drifted"
        ) from exc
    if (
        typed.get("executable_resolved_file_sha256")
        != config.typed_python_resolved_sha256
        or minilm.get("required_tree_sha256")
        != config.minilm_required_tree_sha256
        or cross_encoder.get("required_tree_sha256")
        != config.cross_encoder_required_tree_sha256
    ):
        raise MmqaP1FormalActionRuntimeError(
            "live local runtime binding differs from the preflight"
        )
    return {
        "status": "local_preflight_and_assets_live_verified_once",
        "receipt_self_sha256": observed,
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "typed_runtime_identity_sha256": runtime_identity,
        "minilm_required_tree_sha256": (
            config.minilm_required_tree_sha256
        ),
        "cross_encoder_required_tree_sha256": (
            config.cross_encoder_required_tree_sha256
        ),
        "address_family_isolation_probe_sha256": _semantic_hash(
            receipt_isolation
        ),
        "address_family_isolation_revalidation_count": 1,
        "verification_count": 1,
    }


def validate_official_preflight_once(
    config: FormalActionRuntimeConfig,
) -> official_hippo.FreshComparatorRuntimeBinding:
    """Validate the receipt once with full live filesystem revalidation."""

    value, _raw = _read_canonical_file(
        config.official_preflight_receipt,
        label="fresh official HippoRAG preflight receipt",
        required_mode=0o600,
    )
    observed = _verify_self_hash(value, "receipt_sha256")
    if not hmac.compare_digest(
        observed, config.official_preflight_receipt_sha256
    ):
        raise MmqaP1FormalActionRuntimeError(
            "fresh official preflight receipt expected hash drifted"
        )
    if (
        value.get("expected_package_versions")
        != dict(config.official_expected_package_versions)
        or value.get("expected_module_import_roots")
        != dict(config.official_expected_module_import_roots)
    ):
        raise MmqaP1FormalActionRuntimeError(
            "fresh official expected version/import binding drifted"
        )
    try:
        binding = official_hippo.validate_fresh_preflight_receipt(
            value,
            paths=config.official_runtime_paths,
            filesystem_inspector=(
                official_hippo.production_filesystem_inspector
            ),
        )
    except official_hippo.MmqaP1OfficialHippoRAGBlockError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "fresh official receipt live verification failed"
        ) from exc
    if binding.receipt_sha256 != value.get("receipt_sha256"):
        raise MmqaP1FormalActionRuntimeError(
            "fresh official runtime binding identity drifted"
        )
    return binding


def verify_systemd_isolation_once(
    config: FormalActionRuntimeConfig,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> Mapping[str, Any]:
    """Prove AF denial, process hardening, and both frozen GPUs at startup."""

    if (
        not isinstance(config, FormalActionRuntimeConfig)
        or not callable(process_runner)
    ):
        raise MmqaP1FormalActionRuntimeError(
            "systemd isolation probe config drifted"
        )

    denied: dict[str, str] = {}
    for family, label in (
        (socket.AF_INET, "AF_INET"),
        (socket.AF_INET6, "AF_INET6"),
    ):
        candidate: socket.socket | None = None
        try:
            candidate = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            if (
                errno.EAFNOSUPPORT
                != REQUIRED_ADDRESS_FAMILY_DENIAL_ERRNO
                or exc.errno != REQUIRED_ADDRESS_FAMILY_DENIAL_ERRNO
            ):
                raise MmqaP1FormalActionRuntimeError(
                    f"{label} failed for a reason other than the frozen "
                    "RestrictAddressFamilies denial"
                ) from exc
            denied[label] = f"{type(exc).__name__}:{exc.errno}"
        else:
            candidate.close()
            raise MmqaP1FormalActionRuntimeError(
                f"{label} socket creation succeeded; systemd isolation is absent"
            )
        finally:
            if candidate is not None:
                candidate.close()
    if set(denied) != {"AF_INET", "AF_INET6"}:
        raise MmqaP1FormalActionRuntimeError(
            "systemd INET denial probe is incomplete"
        )
    try:
        status_rows = Path("/proc/self/status").read_text(
            encoding="ascii"
        ).splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MmqaP1FormalActionRuntimeError(
            "systemd process status cannot be inspected"
        ) from exc
    status = {
        key.strip(): value.strip()
        for row in status_rows
        if ":" in row
        for key, value in (row.split(":", 1),)
        if key in {"NoNewPrivs", "Umask"}
    }
    if status != {"NoNewPrivs": "1", "Umask": "0077"}:
        raise MmqaP1FormalActionRuntimeError(
            "systemd NoNewPrivs or UMask contract is absent"
        )
    try:
        completed = process_runner(
            [
                str(config.nvidia_smi),
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            check=False,
            cwd=config.project_root,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": f"{config.nvidia_smi.parent}:/usr/bin:/bin",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except Exception as exc:
        raise MmqaP1FormalActionRuntimeError(
            "frozen GPU visibility probe failed"
        ) from exc
    stdout = getattr(completed, "stdout", b"")
    if (
        getattr(completed, "returncode", None) != 0
        or not isinstance(stdout, bytes)
    ):
        raise MmqaP1FormalActionRuntimeError(
            "frozen GPU visibility probe returned no terminal"
        )
    try:
        gpu_indices = tuple(
            int(row.strip())
            for row in stdout.decode("ascii").splitlines()
            if row.strip()
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise MmqaP1FormalActionRuntimeError(
            "frozen GPU visibility terminal drifted"
        ) from exc
    if gpu_indices != (0, 1):
        raise MmqaP1FormalActionRuntimeError(
            "physical GPU 0/1 visibility contract drifted"
        )
    return {
        "status": (
            "AF_INET_AF_INET6_denied_NoNewPrivs_and_UMask_verified"
        ),
        "probe_count": 6,
        "denied_family_count": 2,
        "denial_errno_binding_sha256": _semantic_hash(denied),
        "NoNewPrivs_verified": True,
        "UMask_0077_verified": True,
        "process_status_binding_sha256": _semantic_hash(status),
        "gpu_0_visible": True,
        "gpu_1_visible": True,
        "gpu_visibility_binding_sha256": _semantic_hash(gpu_indices),
        "outer_network_isolation_contract": (
            SYSTEMD_PARENT_NETWORK_CONTRACT
        ),
        "outer_network_isolation_contract_sha256": (
            SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256
        ),
        "formal_action_transient_unit_contract_sha256": (
            config.systemd_isolation_disposition_sha256
        ),
        "verification_count": 1,
    }


def _safe_coordinate_environment(
    *,
    config: FormalActionRuntimeConfig,
    role_root: Path,
) -> dict[str, str]:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(role_root / "hf"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(role_root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": f"{config.typed_python.parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(config.project_root),
        "TEMP": str(role_root / "tmp"),
        "TMP": str(role_root / "tmp"),
        "TMPDIR": str(role_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if any(
        fragment in key.upper()
        for key in environment
        for fragment in _FORBIDDEN_ENV_FRAGMENTS
    ):
        raise MmqaP1FormalActionRuntimeError(
            "coordinate environment contains a forbidden credential surface"
        )
    return environment


class FormalCoordinateProvider:
    """Concrete two-CLI block provider consumed by the generic controller."""

    def __init__(
        self,
        *,
        config: FormalActionRuntimeConfig,
        private_root: Path,
        local_runtime_identity_sha256: str,
        process_runner: ProcessRunner = subprocess.run,
        executor_factory: CoordinateExecutorFactory = ThreadPoolExecutor,
    ) -> None:
        if (
            not isinstance(config, FormalActionRuntimeConfig)
            or not callable(process_runner)
            or not callable(executor_factory)
        ):
            raise MmqaP1FormalActionRuntimeError(
                "coordinate provider construction drifted"
            )
        self.config = config
        self.private_root = private_root
        self.local_runtime_identity_sha256 = _sha256(
            local_runtime_identity_sha256,
            "derived local runtime identity",
        )
        self.process_runner = process_runner
        self.executor_factory = executor_factory
        self._consumed: set[str] = set()
        self._safe_blocks: dict[str, Mapping[str, Any]] = {}

    def _launch_role(
        self,
        *,
        block: str,
        role: str,
        input_path: Path,
        output_path: Path,
        role_root: Path,
    ) -> object:
        if role not in coordinate_worker.ROLES:
            raise MmqaP1FormalActionRuntimeError(
                "coordinate worker role drifted"
            )
        model = (
            self.config.minilm_model
            if role == coordinate_worker.ROLE_MINILM
            else self.config.cross_encoder_model
        )
        required_tree = (
            self.config.minilm_required_tree_sha256
            if role == coordinate_worker.ROLE_MINILM
            else self.config.cross_encoder_required_tree_sha256
        )
        command = [
            str(self.config.typed_python),
            "-B",
            "-m",
            (
                "assumption_agent.benchmarks."
                "mmqa_p1_block_coordinate_worker_v1"
            ),
            "--role",
            role,
            "--input-block",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(model),
            "--required-tree-sha256",
            required_tree,
            "--local-runtime-identity-sha256",
            self.local_runtime_identity_sha256,
        ]
        try:
            result = self.process_runner(
                command,
                check=False,
                cwd=self.config.project_root,
                env=_safe_coordinate_environment(
                    config=self.config, role_root=role_root
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=COORDINATE_WORKER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise MmqaP1FormalActionRuntimeError(
                f"{block} {role} coordinate CLI failed; no retry permitted"
            ) from exc
        returncode = getattr(result, "returncode", None)
        if type(returncode) is not int or returncode != 0:
            stdout = getattr(result, "stdout", b"")
            stderr = getattr(result, "stderr", b"")
            stdout_bytes = stdout if isinstance(stdout, bytes) else b""
            stderr_bytes = stderr if isinstance(stderr, bytes) else b""
            raise MmqaP1FormalActionRuntimeError(
                f"{block} {role} coordinate CLI failed; "
                f"returncode={returncode}; "
                f"stdout_sha256={hashlib.sha256(stdout_bytes).hexdigest()}; "
                f"stderr_sha256={hashlib.sha256(stderr_bytes).hexdigest()}; "
                "no retry permitted"
            )
        return result

    def __call__(
        self,
        *,
        block: str,
        items: Mapping[str, integration.AnonymousWorkItem],
    ) -> Mapping[str, Sequence[integration.UnitCoordinates]]:
        if (
            block not in formal_controller.BLOCK_ORDER
            or block in self._consumed
            or not isinstance(items, Mapping)
            or not items
        ):
            raise MmqaP1FormalActionRuntimeError(
                "coordinate block is invalid, empty, or replayed"
            )
        self._consumed.add(block)
        block_items = tuple(
            coordinate_worker.AnonymousBlockItem(work_id, work_item)
            for work_id, work_item in items.items()
        )
        stage_root = self.private_root / "coordinates" / block
        if not (stage_root.parent.exists() and stage_root.parent.is_dir()):
            if stage_root.parent.parent != self.private_root:
                raise MmqaP1FormalActionRuntimeError(
                    "coordinate stage escaped its private root"
                )
            _create_private_directory_once(stage_root.parent)
        _create_private_directory_once(stage_root)
        input_path = stage_root / ANONYMOUS_BLOCK_FILENAME
        minilm_path = stage_root / MINILM_ARCHIVE_FILENAME
        cross_encoder_path = stage_root / CROSS_ENCODER_ARCHIVE_FILENAME
        try:
            anonymous_file_sha256 = (
                coordinate_worker.write_private_anonymous_block(
                    input_path, block_items
                )
            )
        except coordinate_worker.MmqaP1BlockCoordinateWorkerError as exc:
            raise MmqaP1FormalActionRuntimeError(
                f"{block} anonymous block seal failed"
            ) from exc
        role_outputs = {
            coordinate_worker.ROLE_MINILM: minilm_path,
            coordinate_worker.ROLE_CROSS_ENCODER: cross_encoder_path,
        }
        role_roots: dict[str, Path] = {}
        for role in coordinate_worker.ROLES:
            role_root = stage_root / role.lower()
            _create_private_directory_once(role_root)
            for name in ("home", "hf", "tmp"):
                _create_private_directory_once(role_root / name)
            role_roots[role] = role_root
        futures: dict[str, Future[object]] = {}
        try:
            pool_value = self.executor_factory(
                max_workers=COORDINATE_PROCESS_COUNT_PER_BLOCK,
                thread_name_prefix=f"mmqa-p1-coordinates-{block}",
            )
            with pool_value as pool:  # type: ignore[attr-defined]
                for role in coordinate_worker.ROLES:
                    futures[role] = pool.submit(  # type: ignore[attr-defined]
                        self._launch_role,
                        block=block,
                        role=role,
                        input_path=input_path,
                        output_path=role_outputs[role],
                        role_root=role_roots[role],
                    )
                for role in coordinate_worker.ROLES:
                    futures[role].result()
        except MmqaP1FormalActionRuntimeError:
            for future in futures.values():
                future.cancel()
            raise
        except Exception as exc:
            for future in futures.values():
                future.cancel()
            raise MmqaP1FormalActionRuntimeError(
                f"{block} coordinate process pair failed; no retry permitted"
            ) from exc
        if len(futures) != COORDINATE_PROCESS_COUNT_PER_BLOCK:
            raise MmqaP1FormalActionRuntimeError(
                "coordinate process count drifted"
            )
        try:
            minilm = coordinate_worker.load_coordinate_archive(minilm_path)
            cross_encoder = coordinate_worker.load_coordinate_archive(
                cross_encoder_path
            )
            coordinate_worker.validate_coordinate_archive_for_block(
                minilm,
                block_items,
                expected_role=coordinate_worker.ROLE_MINILM,
            )
            coordinate_worker.validate_coordinate_archive_for_block(
                cross_encoder,
                block_items,
                expected_role=coordinate_worker.ROLE_CROSS_ENCODER,
            )
        except coordinate_worker.MmqaP1BlockCoordinateWorkerError as exc:
            raise MmqaP1FormalActionRuntimeError(
                f"{block} coordinate archive validation failed"
            ) from exc
        expected_path_hash = {
            coordinate_worker.ROLE_MINILM: hashlib.sha256(
                str(self.config.minilm_model).encode("utf-8")
            ).hexdigest(),
            coordinate_worker.ROLE_CROSS_ENCODER: hashlib.sha256(
                str(self.config.cross_encoder_model).encode("utf-8")
            ).hexdigest(),
        }
        for archive in (minilm, cross_encoder):
            if (
                archive.model_path_sha256
                != expected_path_hash[archive.role]
                or archive.local_runtime_identity_sha256
                != self.local_runtime_identity_sha256
            ):
                raise MmqaP1FormalActionRuntimeError(
                    f"{block} coordinate model/runtime binding drifted"
                )
        mini_by_key = {
            (row.work_id, row.ordinal): row.coordinate
            for row in minilm.rows
        }
        ce_by_key = {
            (row.work_id, row.ordinal): row.coordinate
            for row in cross_encoder.rows
        }
        result: dict[str, tuple[integration.UnitCoordinates, ...]] = {}
        for item in block_items:
            coordinates = []
            for unit in item.work_item.units:
                key = (item.work_id, unit.ordinal)
                try:
                    anchor = action_executor.deterministic_anchor_flags(
                        item.work_item.question, unit.serialized_content
                    )
                    mini_value = mini_by_key[key]
                    ce_value = ce_by_key[key]
                except (
                    KeyError,
                    action_executor.MmqaP1LocalActionExecutorError,
                ) as exc:
                    raise MmqaP1FormalActionRuntimeError(
                        f"{block} coordinate/anchor merge drifted"
                    ) from exc
                coordinates.append(
                    integration.UnitCoordinates(
                        ordinal=unit.ordinal,
                        minilm_similarity=mini_value,
                        cross_encoder_relevance=ce_value,
                        entity_anchor=anchor[0],
                        relation_anchor=anchor[1],
                        numeric_or_temporal_anchor=anchor[2],
                    )
                )
            result[item.work_id] = tuple(coordinates)
        self._safe_blocks[block] = {
            "item_count": len(block_items),
            "unit_count": len(minilm.rows),
            "anonymous_block_file_sha256": anonymous_file_sha256,
            "anonymous_block_semantic_sha256": minilm.anonymous_block_sha256,
            "minilm_archive_sha256": minilm.archive_sha256,
            "minilm_archive_file_sha256": _regular_file_sha256(minilm_path),
            "cross_encoder_archive_sha256": cross_encoder.archive_sha256,
            "cross_encoder_archive_file_sha256": _regular_file_sha256(
                cross_encoder_path
            ),
            "coordinate_CLI_process_count": 2,
            "concurrent_process_cap": 2,
            "dynamic_batch_resize_count": 0,
            "retry_replay_resample_count": 0,
            "network_or_API_call_count": 0,
            "anchor_parser_version": action_executor.ANCHOR_PARSER_VERSION,
        }
        return result

    def safe_summary(self) -> Mapping[str, Any]:
        return {
            "status": "all_consumed_coordinate_blocks_sealed",
            "block_count": len(self._safe_blocks),
            "blocks": {
                block: dict(self._safe_blocks[block])
                for block in formal_controller.BLOCK_ORDER
                if block in self._safe_blocks
            },
            "total_coordinate_CLI_process_count": (
                COORDINATE_PROCESS_COUNT_PER_BLOCK
                * len(self._safe_blocks)
            ),
            "retry_replay_resample_count": 0,
        }


class FormalHippoExecutor:
    """One 45-item official CPU block inheriting the systemd AF_UNIX policy."""

    def __init__(
        self,
        *,
        runtime_binding: official_hippo.FreshComparatorRuntimeBinding,
        private_root: Path,
        block_runner: Callable[..., object] = (
            official_hippo.run_ahold_official_hipporag_block
        ),
    ) -> None:
        if (
            not isinstance(
                runtime_binding,
                official_hippo.FreshComparatorRuntimeBinding,
            )
            or not callable(block_runner)
        ):
            raise MmqaP1FormalActionRuntimeError(
                "official Hippo executor construction drifted"
            )
        self.runtime_binding = runtime_binding
        self.private_root = private_root
        self.block_runner = block_runner
        self._consumed = False
        self._safe: Mapping[str, Any] | None = None

    def __call__(
        self,
        *,
        block: str,
        payloads: Mapping[
            str, action_executor.CandidateRestrictedHippoRAGPayload
        ],
    ) -> Mapping[str, Sequence[int]]:
        if (
            block != "A_hold"
            or self._consumed
            or not isinstance(payloads, Mapping)
        ):
            raise MmqaP1FormalActionRuntimeError(
                "official A_hold block is invalid or replayed"
            )
        self._consumed = True
        items = tuple(
            official_hippo.AHoldHippoItem(work_id, payload)
            for work_id, payload in payloads.items()
        )
        try:
            official_hippo.validate_ahold_items(items)
            archive_value = self.block_runner(
                items,
                runtime_binding=self.runtime_binding,
                stage_parent=self.private_root / "official_A_hold",
            )
        except (
            official_hippo.MmqaP1OfficialHippoRAGBlockError,
            MmqaP1FormalActionRuntimeError,
        ) as exc:
            raise MmqaP1FormalActionRuntimeError(
                "official A_hold block failed; no retry permitted"
            ) from exc
        if not isinstance(
            archive_value, official_hippo.OfficialTerminalArchive
        ):
            raise MmqaP1FormalActionRuntimeError(
                "official A_hold block returned a malformed archive"
            )
        stage_root = self.private_root / "official_A_hold"
        terminal_path = stage_root / HIPPORAG_TERMINAL_FILENAME
        try:
            official_hippo.validate_terminal_archive_for_items(
                archive_value, items
            )
            file_sha256 = official_hippo.write_private_terminal_archive(
                terminal_path, archive_value
            )
            archive = official_hippo.load_private_terminal_archive(
                terminal_path
            )
            official_hippo.validate_terminal_archive_for_items(
                archive, items
            )
        except official_hippo.MmqaP1OfficialHippoRAGBlockError as exc:
            raise MmqaP1FormalActionRuntimeError(
                "official terminal archive seal/reload failed"
            ) from exc
        if archive.runtime_binding_sha256 != self.runtime_binding.binding_sha256:
            raise MmqaP1FormalActionRuntimeError(
                "official terminal runtime binding drifted"
            )
        if (
            archive.address_family_isolation_probe_sha256
            != self.runtime_binding.address_family_isolation_probe_sha256
        ):
            raise MmqaP1FormalActionRuntimeError(
                "official terminal isolation binding drifted"
            )
        self._safe = {
            "status": "official_A_hold_terminal_sealed_and_reread",
            "item_count": len(archive.rows),
            "max_workers": official_hippo.MAX_WORKERS,
            "execution_device_disposition_sha256": _semantic_hash(
                official_hippo.EXECUTION_DEVICE_DISPOSITION
            ),
            "outer_network_isolation_contract_sha256": (
                SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256
            ),
            "official_address_family_contract_sha256": _semantic_hash(
                official_hippo.ADDRESS_FAMILY_ISOLATION_CONTRACT
            ),
            "official_address_family_probe_sha256": (
                archive.address_family_isolation_probe_sha256
            ),
            "runtime_binding_sha256": archive.runtime_binding_sha256,
            "A_hold_input_sha256": archive.A_hold_input_sha256,
            "archive_sha256": archive.archive_sha256,
            "archive_file_sha256": file_sha256,
            "item_launcher_call_count": len(archive.rows),
            "fresh_isolated_index_count": len(archive.rows),
            "parent_address_family_restriction_inherited_count": len(
                archive.rows
            ),
            "bwrap_call_count": 0,
            "retry_replay_resample_count": 0,
            "online_evaluator_call_count": 0,
        }
        return {
            row.work_id: row.top5_source_ordinals for row in archive.rows
        }

    def safe_summary(self) -> Mapping[str, Any]:
        if self._safe is None:
            raise MmqaP1FormalActionRuntimeError(
                "official A_hold terminal was never consumed"
            )
        return dict(self._safe)


def _validate_controller_terminal(
    value: Mapping[str, Any],
    *,
    controller_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MmqaP1FormalActionRuntimeError(
            "generic controller terminal is not a mapping"
        )
    checked = dict(value)
    observed = formal_controller.verify_self_hash(
        checked, "final_sha256"
    )
    path = controller_root / formal_controller.FINAL_RECEIPT_FILENAME
    persisted, raw = _read_canonical_file(
        path,
        label="generic controller final terminal",
        required_mode=0o600,
    )
    if (
        persisted != checked
        or checked.get("schema")
        != f"{formal_controller.VERSION}_hash_safe_final_receipt_v1"
        or checked.get("version") != formal_controller.VERSION
        or checked.get("study_id") != STUDY_ID
        or checked.get("status") not in _ALLOWED_CONTROLLER_STATUSES
        or persisted.get("final_sha256") != observed
    ):
        raise MmqaP1FormalActionRuntimeError(
            "generic controller final terminal drifted"
        )
    a_hold = checked.get("A_hold")
    m_search = checked.get("M_search")
    if (
        not isinstance(a_hold, Mapping)
        or type(a_hold.get("promoted")) is not bool
        or type(a_hold.get("reality_primary_passed")) is not bool
        or not isinstance(m_search, Mapping)
        or type(m_search.get("authorized")) is not bool
        or type(m_search.get("gold_opened")) is not bool
        or type(m_search.get("L5_passed")) is not bool
    ):
        raise MmqaP1FormalActionRuntimeError(
            "generic controller safe outcome shape drifted"
        )
    return {
        "status": str(checked["status"]),
        "final_sha256": observed,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "promotion_passed": bool(a_hold["promoted"]),
        "reality_primary_passed": bool(
            a_hold["reality_primary_passed"]
        ),
        "M_search_authorized": bool(m_search["authorized"]),
        "M_search_gold_opened": bool(m_search["gold_opened"]),
        "L5_passed": bool(m_search["L5_passed"]),
    }


def _assert_safe_wrapper(value: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "work_id",
        "question",
        "items",
        "rows",
        "nodes",
        "edges",
        "coordinates",
        "gold",
        "gold_row_ordinals",
        "gold_text_ordinals",
        "exact_gold_pairs",
        "answer",
        "answers",
        "support",
        "supporting_context",
        "qid",
        "family",
        "evaluation_family",
        "model",
        "coefficients",
    }
    work_id = re.compile(r"mmqa-work-v1-[0-9a-f]{64}\Z")

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if forbidden_keys.intersection(node):
                raise MmqaP1FormalActionRuntimeError(
                    "safe wrapper contains a private field"
                )
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
        elif isinstance(node, str) and work_id.fullmatch(node):
            raise MmqaP1FormalActionRuntimeError(
                "safe wrapper contains a private work ID"
            )

    visit(value)


def _write_runtime_failure(
    *,
    root: Path,
    phase: str,
    exc: BaseException,
) -> None:
    path = root / FAILURE_FILENAME
    if path.exists() or path.is_symlink():
        return
    message = f"{type(exc).__module__}.{type(exc).__qualname__}:{exc}"
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "terminal_consumed_no_retry_replay_or_provider_switch",
        "failed_phase": phase,
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "exception_message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
        "retry_replay_resample_or_provider_switch_authorized": False,
        "online_evaluator_or_API_call_count": 0,
    }
    _write_once(path, _self_hashed(body, "failure_sha256"))


FreezeVerifier = Callable[[FormalActionRuntimeConfig], Mapping[str, Any]]
LocalPreflightVerifier = Callable[
    [FormalActionRuntimeConfig], Mapping[str, Any]
]
OfficialPreflightVerifier = Callable[
    [FormalActionRuntimeConfig],
    official_hippo.FreshComparatorRuntimeBinding,
]
IsolationVerifier = Callable[
    [FormalActionRuntimeConfig], Mapping[str, Any]
]
ControllerRunner = Callable[..., Mapping[str, Any]]


def run_formal_action_runtime(
    config: FormalActionRuntimeConfig,
    *,
    freeze_verifier: FreezeVerifier = (
        verify_execution_and_implementation_freezes
    ),
    local_preflight_verifier: LocalPreflightVerifier = (
        validate_local_preflight_once
    ),
    official_preflight_verifier: OfficialPreflightVerifier = (
        validate_official_preflight_once
    ),
    isolation_verifier: IsolationVerifier = verify_systemd_isolation_once,
    coordinate_process_runner: ProcessRunner = subprocess.run,
    coordinate_executor_factory: CoordinateExecutorFactory = (
        ThreadPoolExecutor
    ),
    official_block_runner: Callable[..., object] = (
        official_hippo.run_ahold_official_hipporag_block
    ),
    controller_runner: ControllerRunner = formal_controller.run_lifecycle,
) -> dict[str, Any]:
    """Consume the only concrete formal-action runtime root."""

    if not isinstance(config, FormalActionRuntimeConfig):
        raise MmqaP1FormalActionRuntimeError("runtime config type drifted")
    try:
        project = config.project_root.resolve(strict=True)
    except OSError as exc:
        raise MmqaP1FormalActionRuntimeError(
            "formal project root is unavailable"
        ) from exc
    if project != config.project_root or not project.is_dir():
        raise MmqaP1FormalActionRuntimeError(
            "formal project root is invalid or symlinked"
        )
    root = project / RUNTIME_ROOT_RELATIVE
    phase = "create_one_shot_runtime_root"
    root_created = False
    try:
        _create_private_directory_once(root)
        root_created = True
        phase = "seal_runtime_attempt"
        attempt = _self_hashed(
            {
                "schema": f"{VERSION}_one_shot_attempt_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "runtime_consumed_before_live_verification",
                "execution_freeze_self_sha256": (
                    config.execution_freeze_self_sha256
                ),
                "implementation_freeze_self_sha256": (
                    config.implementation_freeze_self_sha256
                ),
                "selection_acquisition_sha256": (
                    config.selection_acquisition_sha256
                ),
                "outer_network_isolation_contract_sha256": (
                    SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256
                ),
                "formal_action_transient_unit_contract_sha256": (
                    config.systemd_isolation_disposition_sha256
                ),
                "retry_replay_resample_or_provider_switch_authorized": False,
                "online_evaluator_or_API_authorized": False,
            },
            "attempt_sha256",
        )
        attempt_binding = _write_once(root / ATTEMPT_FILENAME, attempt)

        phase = "live_verify_implementation_and_execution_freezes_once"
        freezes = dict(freeze_verifier(config))
        if freezes.get("verification_count") != 1:
            raise MmqaP1FormalActionRuntimeError(
                "freeze verifier call contract drifted"
            )

        phase = "active_systemd_isolation_probe"
        isolation = dict(isolation_verifier(config))
        if (
            isolation.get("outer_network_isolation_contract_sha256")
            != SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256
            or isolation.get(
                "formal_action_transient_unit_contract_sha256"
            )
            != config.systemd_isolation_disposition_sha256
            or isolation.get("verification_count") != 1
        ):
            raise MmqaP1FormalActionRuntimeError(
                "systemd isolation verifier returned a malformed binding"
            )

        phase = "live_verify_local_preflight_and_assets_once"
        local_binding = dict(local_preflight_verifier(config))
        local_runtime_identity = local_binding.get(
            "typed_runtime_identity_sha256"
        )
        if (
            local_binding.get("verification_count") != 1
            or not isinstance(local_runtime_identity, str)
            or _HEX64.fullmatch(local_runtime_identity) is None
        ):
            raise MmqaP1FormalActionRuntimeError(
                "local preflight verifier call contract drifted"
            )

        phase = "live_verify_fresh_official_preflight_once"
        official_binding = official_preflight_verifier(config)
        if not isinstance(
            official_binding,
            official_hippo.FreshComparatorRuntimeBinding,
        ):
            raise MmqaP1FormalActionRuntimeError(
                "official preflight verifier returned no runtime capability"
            )

        phase = "create_private_runtime_surfaces"
        private_root = root / PRIVATE_ROOT_DIRECTORY
        _create_private_directory_once(private_root)
        coordinate_provider = FormalCoordinateProvider(
            config=config,
            private_root=private_root,
            local_runtime_identity_sha256=local_runtime_identity,
            process_runner=coordinate_process_runner,
            executor_factory=coordinate_executor_factory,
        )
        hippo_executor = FormalHippoExecutor(
            runtime_binding=official_binding,
            private_root=private_root,
            block_runner=official_block_runner,
        )

        phase = "run_generic_formal_lifecycle"
        controller_root = root / CONTROLLER_ROOT_DIRECTORY
        controller_terminal = controller_runner(
            selection_root=(
                project / remote_outer.SELECTION_ROOT_RELATIVE
            ),
            control_root=controller_root,
            expected_selection_acquisition_sha256=(
                config.selection_acquisition_sha256
            ),
            coordinate_provider=coordinate_provider,
            hippo_executor=hippo_executor,
        )
        phase = "validate_generic_controller_safe_terminal"
        controller_safe = _validate_controller_terminal(
            controller_terminal, controller_root=controller_root
        )
        coordinate_safe = dict(coordinate_provider.safe_summary())
        hippo_safe = dict(hippo_executor.safe_summary())

        phase = "seal_safe_outer_wrapper_terminal"
        body = {
            "schema": WRAPPER_SCHEMA,
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": WRAPPER_STATUS,
            "attempt": {
                "semantic_sha256": attempt["attempt_sha256"],
                **dict(attempt_binding),
            },
            "systemd_parent_isolation": isolation,
            "freeze_verification": freezes,
            "local_runtime_preflight": local_binding,
            "official_runtime": {
                "receipt_sha256": official_binding.receipt_sha256,
                "filesystem_binding_sha256": (
                    official_binding.filesystem_binding_sha256
                ),
                "runtime_probe_sha256": (
                    official_binding.runtime_probe_sha256
                ),
                "address_family_isolation_probe_sha256": (
                    official_binding.address_family_isolation_probe_sha256
                ),
                "binding_sha256": official_binding.binding_sha256,
                "live_verification_count": 1,
            },
            "coordinate_runtime": coordinate_safe,
            "official_A_hold": hippo_safe,
            "controller_terminal": controller_safe,
            "outcomes": {
                "promotion_passed": controller_safe["promotion_passed"],
                "reality_primary_passed": (
                    controller_safe["reality_primary_passed"]
                ),
                "M_search_authorized": (
                    controller_safe["M_search_authorized"]
                ),
                "M_search_gold_opened": (
                    controller_safe["M_search_gold_opened"]
                ),
                "L5_passed": controller_safe["L5_passed"],
            },
            "claim_boundary": {
                "runtime_layer_formal_benchmark_source_file_read_count": 0,
                "runtime_layer_gold_pack_read_count": 0,
                "runtime_layer_trusted_ledger_read_count": 0,
                "API_or_Ruoli_environment_read_or_forward_count": 0,
                "online_evaluator_call_count": 0,
                "retry_replay_resample_OOM_resize_or_provider_switch_count": 0,
                "nested_bwrap_launch_count": 0,
                "all_runtime_private_files_mode_0600": True,
            },
        }
        wrapper = _self_hashed(body, "wrapper_sha256")
        _assert_safe_wrapper(wrapper)
        _write_once(root / WRAPPER_TERMINAL_FILENAME, wrapper)
        return wrapper
    except Exception as exc:
        if root_created:
            try:
                _write_runtime_failure(root=root, phase=phase, exc=exc)
            except Exception:
                pass
        if isinstance(exc, MmqaP1FormalActionRuntimeError):
            raise
        raise MmqaP1FormalActionRuntimeError(
            f"formal action runtime failed during {phase}"
        ) from exc


def _key_value_rows(values: Sequence[str], field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if not isinstance(raw, str) or raw.count("=") != 1:
            raise MmqaP1FormalActionRuntimeError(
                f"{field} must use NAME=VALUE"
            )
        key, value = raw.split("=", 1)
        if (
            _SAFE_MAP_KEY.fullmatch(key) is None
            or not value
            or "\x00" in value
            or key in result
        ):
            raise MmqaP1FormalActionRuntimeError(
                f"{field} registry drifted"
            )
        result[key] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--execution-freeze-self-sha256", required=True
    )
    parser.add_argument(
        "--implementation-freeze-self-sha256", required=True
    )
    parser.add_argument(
        "--local-preflight-receipt", required=True, type=Path
    )
    parser.add_argument(
        "--local-preflight-self-sha256", required=True
    )
    parser.add_argument("--typed-python", required=True, type=Path)
    parser.add_argument(
        "--typed-python-resolved-sha256", required=True
    )
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument(
        "--minilm-required-tree-sha256", required=True
    )
    parser.add_argument(
        "--cross-encoder-model", required=True, type=Path
    )
    parser.add_argument(
        "--cross-encoder-required-tree-sha256", required=True
    )
    parser.add_argument(
        "--nvidia-smi",
        type=Path,
        default=Path("/usr/bin/nvidia-smi"),
    )
    parser.add_argument(
        "--systemd-run",
        type=Path,
        default=Path("/usr/bin/systemd-run"),
    )
    parser.add_argument(
        "--systemd-run-resolved-sha256", required=True
    )
    parser.add_argument(
        "--systemd-isolation-disposition-sha256", required=True
    )
    parser.add_argument("--runtime-module-sha256", required=True)
    parser.add_argument(
        "--official-preflight-receipt", required=True, type=Path
    )
    parser.add_argument(
        "--official-preflight-receipt-sha256", required=True
    )
    parser.add_argument(
        "--official-runtime-python", required=True
    )
    parser.add_argument("--official-pyvenv-cfg", required=True)
    parser.add_argument("--official-overlay-root", required=True)
    parser.add_argument(
        "--official-hipporag-source-root", required=True
    )
    parser.add_argument("--official-p16-site-root", required=True)
    parser.add_argument("--official-local-llm-model", required=True)
    parser.add_argument(
        "--official-local-embedding-model", required=True
    )
    parser.add_argument(
        "--official-package-version",
        action="append",
        default=[],
        help="frozen NAME=VERSION; repeat for every receipt entry",
    )
    parser.add_argument(
        "--official-module-import-root",
        action="append",
        default=[],
        help="frozen MODULE=ROOT_LABEL; repeat for every receipt entry",
    )
    parser.add_argument(
        "--selection-acquisition-sha256", required=True
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    values = list(argv if argv is not None else os.sys.argv[1:])
    arguments = _parser().parse_args(values)
    config = FormalActionRuntimeConfig(
        project_root=arguments.project,
        execution_freeze_self_sha256=(
            arguments.execution_freeze_self_sha256
        ),
        implementation_freeze_self_sha256=(
            arguments.implementation_freeze_self_sha256
        ),
        local_preflight_receipt=arguments.local_preflight_receipt,
        local_preflight_self_sha256=(
            arguments.local_preflight_self_sha256
        ),
        typed_python=arguments.typed_python,
        typed_python_resolved_sha256=(
            arguments.typed_python_resolved_sha256
        ),
        minilm_model=arguments.minilm_model,
        minilm_required_tree_sha256=(
            arguments.minilm_required_tree_sha256
        ),
        cross_encoder_model=arguments.cross_encoder_model,
        cross_encoder_required_tree_sha256=(
            arguments.cross_encoder_required_tree_sha256
        ),
        nvidia_smi=arguments.nvidia_smi,
        systemd_run=arguments.systemd_run,
        systemd_run_resolved_sha256=(
            arguments.systemd_run_resolved_sha256
        ),
        systemd_isolation_disposition_sha256=(
            arguments.systemd_isolation_disposition_sha256
        ),
        runtime_module_sha256=arguments.runtime_module_sha256,
        official_preflight_receipt=(
            arguments.official_preflight_receipt
        ),
        official_preflight_receipt_sha256=(
            arguments.official_preflight_receipt_sha256
        ),
        official_runtime_paths=official_hippo.FreshComparatorRuntimePaths(
            runtime_python=arguments.official_runtime_python,
            pyvenv_cfg=arguments.official_pyvenv_cfg,
            overlay_root=arguments.official_overlay_root,
            hipporag_source_root=(
                arguments.official_hipporag_source_root
            ),
            p16_site_root=arguments.official_p16_site_root,
            local_llm_model=arguments.official_local_llm_model,
            local_embedding_model=(
                arguments.official_local_embedding_model
            ),
        ),
        official_expected_package_versions=_key_value_rows(
            arguments.official_package_version,
            "official package version",
        ),
        official_expected_module_import_roots=_key_value_rows(
            arguments.official_module_import_root,
            "official module import-root",
        ),
        selection_acquisition_sha256=(
            arguments.selection_acquisition_sha256
        ),
        controller_arguments=tuple(values),
    )
    run_formal_action_runtime(config)
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION",
    "STUDY_ID",
    "STUDY_DESIGN_SELF_SHA256",
    "RUNTIME_ROOT_RELATIVE",
    "WRAPPER_TERMINAL_RELATIVE",
    "WRAPPER_SCHEMA",
    "WRAPPER_STATUS",
    "SYSTEMD_PARENT_NETWORK_CONTRACT",
    "SYSTEMD_PARENT_NETWORK_CONTRACT_SHA256",
    "SYSTEMD_EXECUTION_POLICY",
    "REQUIRED_ADDRESS_FAMILY_DENIAL_ERRNO",
    "EXECUTION_FREEZE_ARGUMENT_MARKER",
    "LOCAL_PREFLIGHT_ARGUMENT_MARKER",
    "SELECTION_ACQUISITION_ARGUMENT_MARKER",
    "MmqaP1FormalActionRuntimeError",
    "FormalActionRuntimeConfig",
    "FormalCoordinateProvider",
    "FormalHippoExecutor",
    "controller_argument_template",
    "verify_execution_and_implementation_freezes",
    "validate_local_preflight_once",
    "validate_official_preflight_once",
    "verify_systemd_isolation_once",
    "run_formal_action_runtime",
    "main",
]
