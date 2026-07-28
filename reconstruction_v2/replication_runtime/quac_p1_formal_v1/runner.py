"""Strict production outer closure for the frozen QuAC P1 formal study.

The public CLI accepts exactly one canonical configuration path.  It exposes
no source object, model, provider, evaluator, secret, cohort, or injected
callable surface.  A fixed global attempt marker is claimed once, after which
every failure is terminal.  Before either formal source file is opened, the
outer closure verifies the project, core, effect design, source-free canary,
execution freeze, service unit, and complete local runtime bindings.  Each
formal source is then opened through one direct file descriptor exactly once,
hash-checked, and strictly decoded.  Only then is the existing scientific core
invoked; its internal 32-byte selection secret is therefore created after the
entire frozen execution boundary has passed.

The module never downloads data, calls a network or API, evaluates online, or
changes a candidate.  Private source objects are passed only in memory to the
core.  This outer layer writes only one aggregate safe success or failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    quac_p1_formal_controller_v1 as controller,
)
from assumption_agent.benchmarks import (
    quac_p1_formal_runner_v1 as core_runner,
)
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime
from replication_runtime import quac_p1_source_free_canary_v1 as canary


VERSION = "quac_p1_formal_outer_v1"
STUDY_ID = "QUAC_P1_RJMC_DIALOGUE_EVIDENCE_L5_V1"
CONFIG_SCHEMA = f"{VERSION}_config_v1"
GLOBAL_ATTEMPT_SCHEMA = f"{VERSION}_global_attempt_v1"
OUTER_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
OUTER_FAILURE_SCHEMA = f"{VERSION}_safe_failure_v1"
EXPECTED_DESIGN_SELF_SHA256 = (
    "def417300b3c25f127517eef1cdd61760757762f08cc5a9b9877b261036dace2"
)
EXPECTED_TRAIN_SHA256 = (
    "ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a"
)
EXPECTED_DEV_SHA256 = (
    "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378"
)
EXPECTED_TRAIN_SIZE_BYTES = 68114819
EXPECTED_DEV_SIZE_BYTES = 8929167
INCIDENT_SCHEMA = "quac_p1_postqualification_hash_only_custody_incident_v1"
INCIDENT_STATUS = (
    "transparent_hash_only_custody_incident_formal_epoch_unconsumed_"
    "continuation_allowed_only_with_binding"
)
EXPECTED_INCIDENT_SELF_SHA256 = (
    "ca219ae09314064f1126549f8092f56bf10f7e96f12131bcd26c04cf2d416494"
)
EXPECTED_INCIDENT_FILE_SHA256 = (
    "70494886f0f8caa81c03788aa2e928ef935b50169f77770ec260d67a68f34678"
)
IMPLEMENTATION_FREEZE_SCHEMA = "quac_p1_formal_implementation_freeze_v1"
IMPLEMENTATION_FREEZE_STATUS = (
    "frozen_implementation_before_formal_attempt"
)
EXECUTION_FREEZE_SCHEMA = "quac_p1_formal_execution_freeze_v1"
EXECUTION_FREEZE_STATUS = "frozen_execution_before_formal_attempt"
INCIDENT_RELATIVE_PATH = (
    "manifests/quac_p1_postqualification_hash_only_custody_incident_v1.json"
)
FORMAL_CORE_RELATIVE_PATH = (
    "assumption_agent/benchmarks/quac_p1_formal_runner_v1.py"
)
OUTER_RUNNER_RELATIVE_PATH = (
    "replication_runtime/quac_p1_formal_v1/runner.py"
)
OUTER_TEST_RELATIVE_PATH = "tests/test_quac_p1_formal_outer_v1.py"
FORMAL_UNIT_RELATIVE_PATH = "manifests/quac_p1_formal_v1.service"
PRODUCTION_MODULE = "replication_runtime.quac_p1_formal_v1.runner"
FORMAL_CONFIG_FILENAME = "formal_config.json"
IMPLEMENTATION_FREEZE_FILENAME = "implementation_freeze.json"
EXECUTION_FREEZE_FILENAME = "execution_freeze.json"
GLOBAL_ATTEMPT_FILENAME = "global_attempt.json"
OUTER_SAFE_TERMINAL_FILENAME = "outer_terminal.safe.json"
CORE_WORK_ROOT_NAME = "formal_study"
SYSTEMCTL_PATH = Path("/usr/bin/systemctl")
ENV_PATH = Path("/usr/bin/env")
INSTALLED_USER_UNIT_DIRECTORY = Path(
    "/home/erzhu419/.config/systemd/user"
)
SERVICE_CPU_QUOTA = "700%"
SERVICE_MEMORY_MAX = "42949672960"
SERVICE_TASKS_MAX = "128"
SERVICE_TIMEOUT_START_SEC = "infinity"
PRIOR_HASH_ONLY_OPERATION_COUNT = 1
PRIOR_HASH_ONLY_MEMBER_READ_COUNT = 2
REQUIRED_IMPLEMENTATION_RELATIVE_PATHS = (
    "assumption_agent/__init__.py",
    "assumption_agent/models.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/quac_p1_action_adapter_v1.py",
    "assumption_agent/benchmarks/quac_p1_formal_acquisition_v1.py",
    "assumption_agent/benchmarks/quac_p1_source_qualification_v1.py",
    "assumption_agent/benchmarks/quac_p1_formal_controller_v1.py",
    FORMAL_CORE_RELATIVE_PATH,
    "assumption_agent/benchmarks/quac_p1_runtime_v1.py",
    "replication_runtime/__init__.py",
    "replication_runtime/maud_extraction_p2_official_v1/__init__.py",
    "replication_runtime/maud_extraction_p2_official_v1/worker.py",
    "replication_runtime/quac_p1_official_v1/__init__.py",
    "replication_runtime/quac_p1_official_v1/contract.py",
    "replication_runtime/quac_p1_official_v1/worker.py",
    "replication_runtime/quac_p1_formal_v1/__init__.py",
    OUTER_RUNNER_RELATIVE_PATH,
    "replication_runtime/quac_p1_source_free_canary_v1.py",
    OUTER_TEST_RELATIVE_PATH,
)
REQUIRED_RECEIPT_RELATIVE_PATHS = (
    "manifests/red_queen_poststop_rjmc_architecture_decision_v1.json",
    "manifests/quac_p1_source_custody_v1.json",
    "manifests/quac_p1_source_qualification_freeze_v1.json",
    "manifests/quac_p1_source_qualification_result_v1.json",
    "manifests/quac_p1_source_transport_receipt_v1.json",
    "manifests/quac_rjmc_source_free_qualification_freeze_v1.json",
    "manifests/quac_rjmc_source_free_qualification_result_v1.json",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_KEYS = frozenset(
    {
        "canary_binding",
        "config_path",
        "control_root",
        "core_binding",
        "design_binding",
        "execution_freeze_binding",
        "global_attempt_marker_path",
        "implementation_freeze_binding",
        "incident_binding",
        "outer_safe_terminal_path",
        "project_binding",
        "runtime_bindings",
        "schema",
        "self_sha256",
        "service_unit_binding",
        "source_bindings",
        "work_root",
    }
)
_TREE_KEYS = frozenset(
    {"file_count", "path", "total_bytes", "tree_sha256"}
)
_JSON_BINDING_KEYS = frozenset(
    {"file_sha256", "path", "schema", "self_sha256"}
)
_CORE_KEYS = frozenset({"file_sha256", "relative_path"})
_SOURCE_KEYS = frozenset(
    {"mode_octal", "path", "sha256", "size_bytes"}
)
_SOURCE_BINDINGS_KEYS = frozenset({"dev", "train"})
_CANARY_KEYS = frozenset(
    {
        "config_file_sha256",
        "config_path",
        "config_self_sha256",
        "safe_terminal_file_sha256",
        "safe_terminal_path",
        "safe_terminal_self_sha256",
    }
)
_SERVICE_KEYS = frozenset(
    {
        "env_executable_file_sha256",
        "file_sha256",
        "installed_path",
        "path",
        "systemctl_executable_file_sha256",
        "systemctl_executable_path",
        "unit_name",
    }
)
_CANARY_SAFE_KEYS = frozenset(
    {
        "API_or_online_evaluation_call_count",
        "aggregate_only_public_receipt",
        "asset_freeze_self_sha256",
        "canary_attempt_file_sha256",
        "config_self_sha256",
        "effect_execution_design_self_sha256",
        "formal_source_access_count",
        "max_concurrent_physical_model_lanes",
        "minilm_encode_call_count",
        "official_index_call_count",
        "official_retrieve_call_count",
        "parallel_submission_barrier_passed",
        "project_binding_sha256",
        "retry_replay_resample_or_fallback_count",
        "runtime_binding_sha256",
        "runtime_safe_terminal_self_sha256",
        "runtime_verification_token_sha256",
        "schema",
        "self_sha256",
        "source_path_loader_label_qrel_answer_input_count",
        "status",
        "study_id",
        "synthetic_document_count",
        "synthetic_query_count",
    }
)
_IMPLEMENTATION_FREEZE_KEYS = frozenset(
    {
        "API_or_online_evaluation_authorized",
        "architecture_and_source_receipt_bindings",
        "code_and_test_file_sha256",
        "core_binding",
        "custody_incident_binding",
        "deployment_archive_binding",
        "deployment_archive_excludes_formal_sources",
        "design_binding",
        "formal_attempt_limit",
        "project_binding",
        "schema",
        "self_sha256",
        "source_bindings",
        "status",
        "study_id",
    }
)
_EXECUTION_FREEZE_KEYS = frozenset(
    {
        "API_or_online_evaluation_authorized",
        "canary_binding",
        "control_layout",
        "formal_attempt_limit",
        "formal_source_loader_access_count_at_freeze",
        "fresh_state_assertions",
        "implementation_freeze_binding",
        "live_service_attestation_required",
        "preformal_semantic_source_decode_count",
        "prior_postqualification_hash_only_member_read_count",
        "prior_postqualification_hash_only_operation_count",
        "runtime_binding_sha256",
        "runtime_bindings",
        "schema",
        "self_sha256",
        "service_unit_binding",
        "source_bindings",
        "status",
        "study_id",
    }
)
_ARCHIVE_BINDING_KEYS = frozenset(
    {"mode_octal", "path", "sha256", "size_bytes"}
)
_FORBIDDEN_SAFE_KEYS = frozenset(
    {
        "answer",
        "current_qrel",
        "document",
        "item_id",
        "paired_delta_sha256",
        "previous_qrel",
        "private_item_score_sha256",
        "query_id",
        "question",
        "rows",
        "text",
        "unit_id",
    }
)


class QuacP1FormalOuterError(RuntimeError):
    """The immutable formal config, pre-source closure, or run drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1FormalOuterError(
            "outer value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)[:-1]).hexdigest()


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise QuacP1FormalOuterError("self hash already exists")
    return {**body, "self_sha256": stable_hash(body)}


def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1FormalOuterError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise QuacP1FormalOuterError(f"{field} must be positive")
    return value


def _exact_dict(
    value: object,
    keys: frozenset[str],
    field: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise QuacP1FormalOuterError(f"{field} shape drifted")
    return value


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise QuacP1FormalOuterError(f"{field} path drifted")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
    ):
        raise QuacP1FormalOuterError(f"{field} path drifted")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise QuacP1FormalOuterError(
            "bound non-source file cannot be read"
        ) from exc
    return digest.hexdigest()


def _direct_regular(
    path: Path,
    *,
    mode: int,
    field: str,
) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            f"{field} is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise QuacP1FormalOuterError(
            f"{field} direct-file metadata drifted"
        )
    return metadata


def _write_once(
    path: Path,
    value: Mapping[str, object],
) -> str:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o400)
        metadata = path.lstat()
        observed = path.read_bytes()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            "outer artifact cannot be created exactly once"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or observed != raw
    ):
        raise QuacP1FormalOuterError(
            "outer artifact verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourceFileBinding:
    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _absolute_path(str(self.path), "formal source")
        _positive_int(self.size_bytes, "formal source size")
        _hex64(self.sha256, "formal source")

    def payload(self) -> dict[str, object]:
        return {
            "mode_octal": "0600",
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def parse(cls, value: object, field: str) -> "SourceFileBinding":
        row = _exact_dict(value, _SOURCE_KEYS, field)
        if row["mode_octal"] != "0600":
            raise QuacP1FormalOuterError(
                f"{field} mode binding drifted"
            )
        return cls(
            path=_absolute_path(row["path"], field),
            size_bytes=_positive_int(row["size_bytes"], field),
            sha256=_hex64(row["sha256"], field),
        )


@dataclass(frozen=True)
class JsonFileBinding:
    path: Path
    file_sha256: str
    self_sha256: str
    schema: str

    def __post_init__(self) -> None:
        _absolute_path(str(self.path), "bound JSON")
        _hex64(self.file_sha256, "bound JSON file")
        _hex64(self.self_sha256, "bound JSON self")
        if not isinstance(self.schema, str) or not self.schema:
            raise QuacP1FormalOuterError(
                "bound JSON schema drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "path": str(self.path),
            "schema": self.schema,
            "self_sha256": self.self_sha256,
        }

    @classmethod
    def parse(cls, value: object, field: str) -> "JsonFileBinding":
        row = _exact_dict(value, _JSON_BINDING_KEYS, field)
        if not isinstance(row["schema"], str):
            raise QuacP1FormalOuterError(
                f"{field} schema drifted"
            )
        return cls(
            path=_absolute_path(row["path"], field),
            file_sha256=_hex64(row["file_sha256"], field),
            self_sha256=_hex64(row["self_sha256"], field),
            schema=row["schema"],
        )


@dataclass(frozen=True)
class CanaryBinding:
    config_path: Path
    config_file_sha256: str
    config_self_sha256: str
    safe_terminal_path: Path
    safe_terminal_file_sha256: str
    safe_terminal_self_sha256: str

    def __post_init__(self) -> None:
        _absolute_path(str(self.config_path), "canary config")
        _absolute_path(str(self.safe_terminal_path), "canary terminal")
        for value, field in (
            (self.config_file_sha256, "canary config file"),
            (self.config_self_sha256, "canary config self"),
            (self.safe_terminal_file_sha256, "canary terminal file"),
            (self.safe_terminal_self_sha256, "canary terminal self"),
        ):
            _hex64(value, field)

    def payload(self) -> dict[str, object]:
        return {
            "config_file_sha256": self.config_file_sha256,
            "config_path": str(self.config_path),
            "config_self_sha256": self.config_self_sha256,
            "safe_terminal_file_sha256": (
                self.safe_terminal_file_sha256
            ),
            "safe_terminal_path": str(self.safe_terminal_path),
            "safe_terminal_self_sha256": (
                self.safe_terminal_self_sha256
            ),
        }

    @classmethod
    def parse(cls, value: object) -> "CanaryBinding":
        row = _exact_dict(value, _CANARY_KEYS, "canary binding")
        return cls(
            config_path=_absolute_path(
                row["config_path"],
                "canary config",
            ),
            config_file_sha256=_hex64(
                row["config_file_sha256"],
                "canary config file",
            ),
            config_self_sha256=_hex64(
                row["config_self_sha256"],
                "canary config self",
            ),
            safe_terminal_path=_absolute_path(
                row["safe_terminal_path"],
                "canary terminal",
            ),
            safe_terminal_file_sha256=_hex64(
                row["safe_terminal_file_sha256"],
                "canary terminal file",
            ),
            safe_terminal_self_sha256=_hex64(
                row["safe_terminal_self_sha256"],
                "canary terminal self",
            ),
        )


@dataclass(frozen=True)
class ServiceUnitBinding:
    path: Path
    installed_path: Path
    file_sha256: str
    unit_name: str
    env_executable_file_sha256: str
    systemctl_executable_path: Path
    systemctl_executable_file_sha256: str

    def __post_init__(self) -> None:
        _absolute_path(str(self.path), "service unit")
        _absolute_path(
            str(self.installed_path),
            "installed service unit",
        )
        _hex64(self.file_sha256, "service unit file")
        _absolute_path(
            str(self.systemctl_executable_path),
            "systemctl executable",
        )
        _hex64(
            self.env_executable_file_sha256,
            "env executable file",
        )
        _hex64(
            self.systemctl_executable_file_sha256,
            "systemctl executable file",
        )
        if (
            not isinstance(self.unit_name, str)
            or not self.unit_name.endswith(".service")
            or self.path.name != self.unit_name
            or self.path
            != _PROJECT_ROOT / FORMAL_UNIT_RELATIVE_PATH
            or self.installed_path.name != self.unit_name
            or self.installed_path.parent
            != INSTALLED_USER_UNIT_DIRECTORY
            or self.installed_path == self.path
            or self.systemctl_executable_path != SYSTEMCTL_PATH
        ):
            raise QuacP1FormalOuterError(
                "service unit name drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "env_executable_file_sha256": (
                self.env_executable_file_sha256
            ),
            "file_sha256": self.file_sha256,
            "installed_path": str(self.installed_path),
            "path": str(self.path),
            "systemctl_executable_file_sha256": (
                self.systemctl_executable_file_sha256
            ),
            "systemctl_executable_path": str(
                self.systemctl_executable_path
            ),
            "unit_name": self.unit_name,
        }

    @classmethod
    def parse(cls, value: object) -> "ServiceUnitBinding":
        row = _exact_dict(value, _SERVICE_KEYS, "service unit binding")
        if not isinstance(row["unit_name"], str):
            raise QuacP1FormalOuterError(
                "service unit name drifted"
            )
        return cls(
            path=_absolute_path(row["path"], "service unit"),
            installed_path=_absolute_path(
                row["installed_path"],
                "installed service unit",
            ),
            file_sha256=_hex64(
                row["file_sha256"],
                "service unit",
            ),
            unit_name=row["unit_name"],
            env_executable_file_sha256=_hex64(
                row["env_executable_file_sha256"],
                "env executable file",
            ),
            systemctl_executable_path=_absolute_path(
                row["systemctl_executable_path"],
                "systemctl executable",
            ),
            systemctl_executable_file_sha256=_hex64(
                row["systemctl_executable_file_sha256"],
                "systemctl executable file",
            ),
        )


@dataclass(frozen=True)
class CoreBinding:
    file_sha256: str

    def __post_init__(self) -> None:
        _hex64(self.file_sha256, "formal core file")

    def payload(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "relative_path": FORMAL_CORE_RELATIVE_PATH,
        }

    @classmethod
    def parse(cls, value: object) -> "CoreBinding":
        row = _exact_dict(value, _CORE_KEYS, "core binding")
        if row["relative_path"] != FORMAL_CORE_RELATIVE_PATH:
            raise QuacP1FormalOuterError(
                "formal core relative path drifted"
            )
        return cls(
            _hex64(row["file_sha256"], "formal core file")
        )


def _parse_project_binding(
    value: object,
) -> runtime.FrozenTreeBinding:
    row = _exact_dict(value, _TREE_KEYS, "project binding")
    binding = runtime.FrozenTreeBinding(
        path=str(_absolute_path(row["path"], "project root")),
        tree_sha256=_hex64(
            row["tree_sha256"],
            "project tree",
        ),
        file_count=_positive_int(
            row["file_count"],
            "project file count",
        ),
        total_bytes=_positive_int(
            row["total_bytes"],
            "project total bytes",
        ),
    )
    return binding


def _project_payload(
    binding: runtime.FrozenTreeBinding,
) -> dict[str, object]:
    return {
        "file_count": binding.file_count,
        "path": binding.path,
        "total_bytes": binding.total_bytes,
        "tree_sha256": binding.tree_sha256,
    }


@dataclass(frozen=True)
class SourceFreeFormalConfig:
    config_path: Path
    control_root: Path
    work_root: Path
    global_attempt_marker_path: Path
    outer_safe_terminal_path: Path
    train: SourceFileBinding
    dev: SourceFileBinding
    design_binding: JsonFileBinding
    project_binding: runtime.FrozenTreeBinding
    core_binding: CoreBinding
    incident_binding: JsonFileBinding
    canary_binding: CanaryBinding
    runtime_bindings: runtime.RuntimeBindings
    implementation_freeze_binding: JsonFileBinding
    execution_freeze_binding: JsonFileBinding
    service_unit_binding: ServiceUnitBinding
    self_sha256: str

    def __post_init__(self) -> None:
        for path, field in (
            (self.config_path, "config"),
            (self.control_root, "control root"),
            (self.work_root, "work root"),
            (self.global_attempt_marker_path, "global attempt"),
            (self.outer_safe_terminal_path, "outer terminal"),
        ):
            _absolute_path(str(path), field)
        if (
            self.train.sha256 != EXPECTED_TRAIN_SHA256
            or self.dev.sha256 != EXPECTED_DEV_SHA256
            or self.train.size_bytes != EXPECTED_TRAIN_SIZE_BYTES
            or self.dev.size_bytes != EXPECTED_DEV_SIZE_BYTES
            or self.design_binding.self_sha256
            != EXPECTED_DESIGN_SELF_SHA256
            or self.project_binding.path != str(_PROJECT_ROOT)
            or self.incident_binding.path
            != _PROJECT_ROOT / INCIDENT_RELATIVE_PATH
            or self.incident_binding.file_sha256
            != EXPECTED_INCIDENT_FILE_SHA256
            or self.incident_binding.self_sha256
            != EXPECTED_INCIDENT_SELF_SHA256
            or self.incident_binding.schema != INCIDENT_SCHEMA
            or self.implementation_freeze_binding.schema
            != IMPLEMENTATION_FREEZE_SCHEMA
            or self.execution_freeze_binding.schema
            != EXECUTION_FREEZE_SCHEMA
        ):
            raise QuacP1FormalOuterError(
                "frozen study identity drifted"
            )
        if (
            self.config_path
            != self.control_root / FORMAL_CONFIG_FILENAME
            or self.implementation_freeze_binding.path
            != self.control_root / IMPLEMENTATION_FREEZE_FILENAME
            or self.execution_freeze_binding.path
            != self.control_root / EXECUTION_FREEZE_FILENAME
            or self.global_attempt_marker_path
            != self.control_root / GLOBAL_ATTEMPT_FILENAME
            or self.outer_safe_terminal_path
            != self.control_root / OUTER_SAFE_TERMINAL_FILENAME
            or self.work_root
            != self.control_root / CORE_WORK_ROOT_NAME
            or self.control_root == _PROJECT_ROOT
            or self.control_root.is_relative_to(_PROJECT_ROOT)
            or _PROJECT_ROOT.is_relative_to(self.control_root)
        ):
            raise QuacP1FormalOuterError(
                "fixed formal control layout drifted"
            )
        paths = (
            self.config_path,
            self.control_root,
            self.work_root,
            self.global_attempt_marker_path,
            self.outer_safe_terminal_path,
            self.train.path,
            self.dev.path,
            self.design_binding.path,
            self.incident_binding.path,
            self.canary_binding.config_path,
            self.canary_binding.safe_terminal_path,
            self.implementation_freeze_binding.path,
            self.execution_freeze_binding.path,
            self.service_unit_binding.path,
        )
        if len(set(paths)) != len(paths):
            raise QuacP1FormalOuterError(
                "formal path identities collide"
            )
        for source in (self.train.path, self.dev.path):
            if (
                source.is_relative_to(_PROJECT_ROOT)
                or source.is_relative_to(self.control_root)
            ):
                raise QuacP1FormalOuterError(
                    "formal source cannot be inside a mutable formal tree"
                )
        _hex64(self.self_sha256, "formal config self")
        if stable_hash(self.body()) != self.self_sha256:
            raise QuacP1FormalOuterError(
                "formal config self hash drifted"
            )

    def body(self) -> dict[str, object]:
        return {
            "canary_binding": self.canary_binding.payload(),
            "config_path": str(self.config_path),
            "control_root": str(self.control_root),
            "core_binding": self.core_binding.payload(),
            "design_binding": self.design_binding.payload(),
            "execution_freeze_binding": (
                self.execution_freeze_binding.payload()
            ),
            "global_attempt_marker_path": str(
                self.global_attempt_marker_path
            ),
            "implementation_freeze_binding": (
                self.implementation_freeze_binding.payload()
            ),
            "incident_binding": self.incident_binding.payload(),
            "outer_safe_terminal_path": str(
                self.outer_safe_terminal_path
            ),
            "project_binding": _project_payload(
                self.project_binding
            ),
            "runtime_bindings": canary.runtime_bindings_payload(
                self.runtime_bindings
            ),
            "schema": CONFIG_SCHEMA,
            "service_unit_binding": (
                self.service_unit_binding.payload()
            ),
            "source_bindings": {
                "dev": self.dev.payload(),
                "train": self.train.payload(),
            },
            "work_root": str(self.work_root),
        }

    def payload(self) -> dict[str, object]:
        return {**self.body(), "self_sha256": self.self_sha256}


def parse_config(value: object) -> SourceFreeFormalConfig:
    row = _exact_dict(value, _CONFIG_KEYS, "formal config")
    if row["schema"] != CONFIG_SCHEMA:
        raise QuacP1FormalOuterError(
            "formal config schema drifted"
        )
    body = {
        key: item
        for key, item in row.items()
        if key != "self_sha256"
    }
    supplied_self = _hex64(
        row["self_sha256"],
        "formal config self",
    )
    if stable_hash(body) != supplied_self:
        raise QuacP1FormalOuterError(
            "formal config self hash drifted"
        )
    source_rows = _exact_dict(
        row["source_bindings"],
        _SOURCE_BINDINGS_KEYS,
        "source bindings",
    )
    return SourceFreeFormalConfig(
        config_path=_absolute_path(row["config_path"], "config"),
        control_root=_absolute_path(
            row["control_root"],
            "control root",
        ),
        work_root=_absolute_path(row["work_root"], "work root"),
        global_attempt_marker_path=_absolute_path(
            row["global_attempt_marker_path"],
            "global attempt",
        ),
        outer_safe_terminal_path=_absolute_path(
            row["outer_safe_terminal_path"],
            "outer terminal",
        ),
        train=SourceFileBinding.parse(
            source_rows["train"],
            "TRAIN source",
        ),
        dev=SourceFileBinding.parse(
            source_rows["dev"],
            "DEV source",
        ),
        design_binding=JsonFileBinding.parse(
            row["design_binding"],
            "design binding",
        ),
        project_binding=_parse_project_binding(
            row["project_binding"]
        ),
        core_binding=CoreBinding.parse(row["core_binding"]),
        incident_binding=JsonFileBinding.parse(
            row["incident_binding"],
            "custody incident binding",
        ),
        canary_binding=CanaryBinding.parse(row["canary_binding"]),
        runtime_bindings=canary.parse_runtime_bindings(
            row["runtime_bindings"]
        ),
        implementation_freeze_binding=JsonFileBinding.parse(
            row["implementation_freeze_binding"],
            "implementation freeze binding",
        ),
        execution_freeze_binding=JsonFileBinding.parse(
            row["execution_freeze_binding"],
            "execution freeze binding",
        ),
        service_unit_binding=ServiceUnitBinding.parse(
            row["service_unit_binding"]
        ),
        self_sha256=supplied_self,
    )


def load_config(path: Path) -> SourceFreeFormalConfig:
    path = _absolute_path(str(path), "formal config")
    metadata = _direct_regular(
        path,
        mode=0o400,
        field="formal config",
    )
    if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
        raise QuacP1FormalOuterError(
            "formal config ownership drifted"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise QuacP1FormalOuterError(
            "formal config cannot be decoded"
        ) from exc
    if type(value) is not dict or raw != canonical_bytes(value):
        raise QuacP1FormalOuterError(
            "formal config is not exact canonical JSON"
        )
    config = parse_config(value)
    if config.config_path != path:
        raise QuacP1FormalOuterError(
            "formal config path binding drifted"
        )
    return config


def _load_bound_json(
    binding: JsonFileBinding,
    *,
    field: str,
    canonical_required: bool,
) -> dict[str, object]:
    metadata = _direct_regular(
        binding.path,
        mode=0o400,
        field=field,
    )
    if metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
        raise QuacP1FormalOuterError(
            f"{field} ownership drifted"
        )
    try:
        raw = binding.path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise QuacP1FormalOuterError(
            f"{field} cannot be decoded"
        ) from exc
    if (
        not isinstance(value, dict)
        or hashlib.sha256(raw).hexdigest() != binding.file_sha256
        or value.get("schema") != binding.schema
        or value.get("self_sha256") != binding.self_sha256
        or (
            canonical_required
            and raw != canonical_bytes(value)
        )
    ):
        raise QuacP1FormalOuterError(
            f"{field} binding drifted"
        )
    body = {
        key: item
        for key, item in value.items()
        if key != "self_sha256"
    }
    if stable_hash(body) != binding.self_sha256:
        raise QuacP1FormalOuterError(
            f"{field} self hash drifted"
        )
    return value


def _contains_scalar(value: object, target: object) -> bool:
    if value == target:
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_scalar(item, target)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_scalar(item, target) for item in value)
    return False


@dataclass(frozen=True)
class PreSourceReceipt:
    project_tree_sha256: str
    core_file_sha256: str
    design_self_sha256: str
    incident_self_sha256: str
    implementation_freeze_self_sha256: str
    execution_freeze_self_sha256: str
    service_unit_file_sha256: str
    canary_terminal_self_sha256: str
    runtime_binding_sha256: str


@dataclass(frozen=True)
class LiveServiceReceipt:
    attestation_sha256: str
    invocation_id_sha256: str
    main_pid: int
    restart_count: int

    def __post_init__(self) -> None:
        _hex64(self.attestation_sha256, "live service attestation")
        _hex64(self.invocation_id_sha256, "live service invocation")
        if (
            type(self.main_pid) is not int
            or self.main_pid != os.getpid()
            or type(self.restart_count) is not int
            or self.restart_count != 0
        ):
            raise QuacP1FormalOuterError(
                "live service receipt drifted"
            )


def _source_bindings_payload(
    config: SourceFreeFormalConfig,
) -> dict[str, object]:
    return {
        "dev": config.dev.payload(),
        "train": config.train.payload(),
    }


def _verify_control_layout(
    config: SourceFreeFormalConfig,
) -> None:
    try:
        root = config.control_root.lstat()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            "formal control root is unavailable"
        ) from exc
    if (
        config.control_root.is_symlink()
        or not stat.S_ISDIR(root.st_mode)
        or stat.S_IMODE(root.st_mode) != 0o700
        or root.st_uid != os.getuid()
    ):
        raise QuacP1FormalOuterError(
            "formal control root metadata drifted"
        )
    for name in ("cache", "home", "tmp"):
        path = config.control_root / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise QuacP1FormalOuterError(
                "closed service directory is unavailable"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise QuacP1FormalOuterError(
                "closed service directory metadata drifted"
            )


def _verify_incident(
    config: SourceFreeFormalConfig,
) -> dict[str, object]:
    value = _load_bound_json(
        config.incident_binding,
        field="hash-only custody incident",
        canonical_required=False,
    )
    incident = value.get("incident")
    exposure = value.get("exposure_boundary")
    validity = value.get("validity_assessment")
    continuation = value.get("continuation_disposition")
    activity = value.get("activity_counts_at_incident")
    if (
        value.get("status") != INCIDENT_STATUS
        or value.get("study_id") != STUDY_ID
        or not isinstance(incident, Mapping)
        or incident.get("postqualification_hash_only_operation_count")
        != PRIOR_HASH_ONLY_OPERATION_COUNT
        or incident.get("postqualification_hash_only_member_read_count")
        != PRIOR_HASH_ONLY_MEMBER_READ_COUNT
        or incident.get("train_observed_sha256")
        != EXPECTED_TRAIN_SHA256
        or incident.get("dev_observed_sha256") != EXPECTED_DEV_SHA256
        or not isinstance(exposure, Mapping)
        or exposure.get("semantic_field_extracted") is not False
        or exposure.get("structured_content_parsed") is not False
        or exposure.get("raw_source_bytes_exposed_to_model_or_user")
        is not False
        or not isinstance(validity, Mapping)
        or validity.get("formal_epoch_consumed") is not False
        or validity.get("semantic_holdout_contamination_observed")
        is not False
        or validity.get(
            "study_remains_interpretable_with_explicit_incident_binding"
        )
        is not True
        or not isinstance(continuation, Mapping)
        or continuation.get("formal_execution_must_bind_this_incident")
        is not True
        or continuation.get(
            "no_further_preformal_source_body_hash_parse_or_decode_authorized"
        )
        is not True
        or not isinstance(activity, Mapping)
        or activity.get("formal_source_decode") != 0
        or activity.get("model") != 0
        or activity.get("score") != 0
        or activity.get("selection") != 0
    ):
        raise QuacP1FormalOuterError(
            "hash-only custody incident semantics drifted"
        )
    return value


def _verify_archive_binding(value: object) -> None:
    row = _exact_dict(
        value,
        _ARCHIVE_BINDING_KEYS,
        "deployment archive binding",
    )
    if row["mode_octal"] != "0400":
        raise QuacP1FormalOuterError(
            "deployment archive mode binding drifted"
        )
    path = _absolute_path(row["path"], "deployment archive")
    size = _positive_int(row["size_bytes"], "deployment archive size")
    expected = _hex64(row["sha256"], "deployment archive")
    metadata = _direct_regular(
        path,
        mode=0o400,
        field="deployment archive",
    )
    if metadata.st_size != size or _sha256_file(path) != expected:
        raise QuacP1FormalOuterError(
            "deployment archive binding drifted"
        )


def _verify_implementation_freeze(
    config: SourceFreeFormalConfig,
) -> dict[str, object]:
    value = _load_bound_json(
        config.implementation_freeze_binding,
        field="implementation freeze",
        canonical_required=True,
    )
    _exact_dict(
        value,
        _IMPLEMENTATION_FREEZE_KEYS,
        "implementation freeze",
    )
    if (
        value.get("status") != IMPLEMENTATION_FREEZE_STATUS
        or value.get("study_id") != STUDY_ID
        or value.get("formal_attempt_limit") != 1
        or value.get("API_or_online_evaluation_authorized") is not False
        or value.get("deployment_archive_excludes_formal_sources")
        is not True
        or value.get("project_binding")
        != _project_payload(config.project_binding)
        or value.get("core_binding") != config.core_binding.payload()
        or value.get("design_binding")
        != config.design_binding.payload()
        or value.get("custody_incident_binding")
        != config.incident_binding.payload()
        or value.get("source_bindings")
        != _source_bindings_payload(config)
    ):
        raise QuacP1FormalOuterError(
            "implementation freeze semantics drifted"
        )
    _verify_archive_binding(value["deployment_archive_binding"])
    code_rows = value.get("code_and_test_file_sha256")
    if (
        type(code_rows) is not dict
        or set(code_rows) != set(REQUIRED_IMPLEMENTATION_RELATIVE_PATHS)
    ):
        raise QuacP1FormalOuterError(
            "implementation code/test closure drifted"
        )
    for relative in REQUIRED_IMPLEMENTATION_RELATIVE_PATHS:
        expected = _hex64(
            code_rows[relative],
            f"implementation file {relative}",
        )
        path = _PROJECT_ROOT / relative
        _direct_regular(
            path,
            mode=0o400,
            field=f"implementation file {relative}",
        )
        if _sha256_file(path) != expected:
            raise QuacP1FormalOuterError(
                "implementation code/test file drifted"
            )
    receipt_rows = value.get(
        "architecture_and_source_receipt_bindings"
    )
    if (
        type(receipt_rows) is not dict
        or set(receipt_rows) != set(REQUIRED_RECEIPT_RELATIVE_PATHS)
    ):
        raise QuacP1FormalOuterError(
            "architecture/source receipt closure drifted"
        )
    for relative in REQUIRED_RECEIPT_RELATIVE_PATHS:
        binding = JsonFileBinding.parse(
            receipt_rows[relative],
            f"receipt {relative}",
        )
        if binding.path != _PROJECT_ROOT / relative:
            raise QuacP1FormalOuterError(
                "architecture/source receipt path drifted"
            )
        _load_bound_json(
            binding,
            field=f"receipt {relative}",
            canonical_required=False,
        )
    return value


def _verify_execution_freeze(
    config: SourceFreeFormalConfig,
    *,
    runtime_binding_sha256: str,
) -> dict[str, object]:
    value = _load_bound_json(
        config.execution_freeze_binding,
        field="execution freeze",
        canonical_required=True,
    )
    _exact_dict(
        value,
        _EXECUTION_FREEZE_KEYS,
        "execution freeze",
    )
    expected_control = {
        "config_path": str(config.config_path),
        "control_root": str(config.control_root),
        "global_attempt_marker_path": str(
            config.global_attempt_marker_path
        ),
        "outer_safe_terminal_path": str(
            config.outer_safe_terminal_path
        ),
        "work_root": str(config.work_root),
    }
    expected_freshness = {
        "global_attempt_marker_absent": True,
        "outer_safe_terminal_absent": True,
        "work_root_absent": True,
    }
    if (
        value.get("status") != EXECUTION_FREEZE_STATUS
        or value.get("study_id") != STUDY_ID
        or value.get("formal_attempt_limit") != 1
        or value.get("API_or_online_evaluation_authorized") is not False
        or value.get("implementation_freeze_binding")
        != config.implementation_freeze_binding.payload()
        or value.get("service_unit_binding")
        != config.service_unit_binding.payload()
        or value.get("canary_binding")
        != config.canary_binding.payload()
        or value.get("runtime_bindings")
        != canary.runtime_bindings_payload(config.runtime_bindings)
        or value.get("runtime_binding_sha256")
        != runtime_binding_sha256
        or value.get("control_layout") != expected_control
        or value.get("source_bindings")
        != _source_bindings_payload(config)
        or value.get("fresh_state_assertions") != expected_freshness
        or value.get("live_service_attestation_required") is not True
        or value.get("formal_source_loader_access_count_at_freeze") != 0
        or value.get("preformal_semantic_source_decode_count") != 0
        or value.get(
            "prior_postqualification_hash_only_operation_count"
        )
        != PRIOR_HASH_ONLY_OPERATION_COUNT
        or value.get(
            "prior_postqualification_hash_only_member_read_count"
        )
        != PRIOR_HASH_ONLY_MEMBER_READ_COUNT
    ):
        raise QuacP1FormalOuterError(
            "execution freeze semantics drifted"
        )
    return value


def _expected_service_environment(
    config: SourceFreeFormalConfig,
) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(config.control_root / "cache"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(config.control_root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(
            (
                str(_PROJECT_ROOT),
                config.runtime_bindings.gpu0_python.import_tree.path,
            )
        ),
        "TEMP": str(config.control_root / "tmp"),
        "TMP": str(config.control_root / "tmp"),
        "TMPDIR": str(config.control_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _expected_service_argv(
    config: SourceFreeFormalConfig,
) -> list[str]:
    environment = _expected_service_environment(config)
    return [
        str(ENV_PATH),
        "-i",
        *[
            f"{key}={value}"
            for key, value in sorted(environment.items())
        ],
        config.runtime_bindings.gpu0_python.executable.path,
        "-S",
        "-B",
        "-s",
        "-m",
        PRODUCTION_MODULE,
        "--config",
        str(config.config_path),
    ]


def _expected_service_directives(
    _config: SourceFreeFormalConfig,
) -> dict[str, str]:
    return {
        "CPUQuota": SERVICE_CPU_QUOTA,
        "IPAddressDeny": "any",
        "KillMode": "control-group",
        "MemoryMax": SERVICE_MEMORY_MAX,
        "NoNewPrivileges": "yes",
        "PrivateTmp": "yes",
        "Restart": "no",
        "RestrictAddressFamilies": "AF_UNIX",
        "TasksMax": SERVICE_TASKS_MAX,
        "TimeoutStartSec": SERVICE_TIMEOUT_START_SEC,
        "Type": "oneshot",
        "UMask": "0077",
        "WorkingDirectory": str(_PROJECT_ROOT),
    }


def _verify_service_unit(
    config: SourceFreeFormalConfig,
) -> None:
    binding = config.service_unit_binding
    _direct_regular(binding.path, mode=0o400, field="service unit")
    _direct_regular(
        ENV_PATH,
        mode=0o755,
        field="env executable",
    )
    _direct_regular(
        binding.systemctl_executable_path,
        mode=0o755,
        field="systemctl executable",
    )
    if (
        _sha256_file(ENV_PATH)
        != binding.env_executable_file_sha256
        or _sha256_file(binding.systemctl_executable_path)
        != binding.systemctl_executable_file_sha256
    ):
        raise QuacP1FormalOuterError(
            "service executable binding drifted"
        )
    try:
        raw = binding.path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise QuacP1FormalOuterError(
            "service unit cannot be read"
        ) from exc
    lines = text.splitlines()
    exec_lines = [
        row.removeprefix("ExecStart=")
        for row in lines
        if row.startswith("ExecStart=")
    ]
    try:
        argv = shlex.split(exec_lines[0])
    except (ValueError, IndexError) as exc:
        raise QuacP1FormalOuterError(
            "service ExecStart cannot be parsed"
        ) from exc
    expected_argv = _expected_service_argv(config)
    exact_directives = _expected_service_directives(config)
    observed_directives: dict[str, list[str]] = {}
    for line in lines:
        if "=" not in line or line.startswith("ExecStart="):
            continue
        key, value = line.split("=", 1)
        observed_directives.setdefault(key, []).append(value)
    installed = binding.installed_path
    try:
        installed_metadata = installed.lstat()
        installed_target = installed.readlink()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            "installed service unit binding is unavailable"
        ) from exc
    if not installed_target.is_absolute():
        installed_target = installed.parent / installed_target
    if (
        hashlib.sha256(raw).hexdigest() != binding.file_sha256
        or len(exec_lines) != 1
        or argv != expected_argv
        or any(
            observed_directives.get(key) != [value]
            for key, value in exact_directives.items()
        )
        or not stat.S_ISLNK(installed_metadata.st_mode)
        or installed_target != binding.path
        or any(
            marker in text.upper()
            for marker in ("RUOLI", "API_KEY", "OPENAI_API")
        )
        or any(
            line.startswith(("Environment=", "EnvironmentFile="))
            for line in lines
        )
    ):
        raise QuacP1FormalOuterError(
            "service unit semantic binding drifted"
        )


def _verify_live_service_attestation(
    config: SourceFreeFormalConfig,
) -> LiveServiceReceipt:
    runtime_root = Path(f"/run/user/{os.getuid()}")
    try:
        runtime_metadata = runtime_root.lstat()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            "user runtime directory is unavailable"
        ) from exc
    if (
        runtime_root.is_symlink()
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or runtime_metadata.st_uid != os.getuid()
    ):
        raise QuacP1FormalOuterError(
            "user runtime directory metadata drifted"
        )
    properties = (
        "ActiveState",
        "ExecMainPID",
        "FragmentPath",
        "InvocationID",
        "MainPID",
        "NRestarts",
        "Restart",
        "SubState",
        "Type",
    )
    command = [
        str(config.service_unit_binding.systemctl_executable_path),
        "--user",
        "show",
        config.service_unit_binding.unit_name,
        f"--property={','.join(properties)}",
        "--no-pager",
    ]
    environment = {
        "DBUS_SESSION_BUS_ADDRESS": (
            f"unix:path={runtime_root / 'bus'}"
        ),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(runtime_root),
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QuacP1FormalOuterError(
            "live systemd attestation could not run"
        ) from exc
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > 4096
    ):
        raise QuacP1FormalOuterError(
            "live systemd attestation command failed"
        )
    try:
        text = completed.stdout.decode("ascii")
        rows = {}
        for line in text.splitlines():
            key, value = line.split("=", 1)
            if key in rows:
                raise ValueError("duplicate systemd property")
            rows[key] = value
    except (UnicodeDecodeError, ValueError) as exc:
        raise QuacP1FormalOuterError(
            "live systemd attestation output drifted"
        ) from exc
    invocation = rows.get("InvocationID")
    expected_pid = str(os.getpid())
    fragment_value = rows.get("FragmentPath")
    try:
        fragment_path = (
            Path(fragment_value)
            if isinstance(fragment_value, str)
            else Path()
        )
        resolved_fragment = fragment_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise QuacP1FormalOuterError(
            "live systemd fragment path cannot be resolved"
        ) from exc
    if (
        set(rows) != set(properties)
        or rows.get("ActiveState") != "activating"
        or rows.get("SubState") != "start"
        or rows.get("Type") != "oneshot"
        or rows.get("MainPID") != expected_pid
        or rows.get("ExecMainPID") != expected_pid
        or rows.get("NRestarts") != "0"
        or rows.get("Restart") != "no"
        or fragment_path
        not in {
            config.service_unit_binding.installed_path,
            config.service_unit_binding.path,
        }
        or resolved_fragment != config.service_unit_binding.path
        or not isinstance(invocation, str)
        or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
    ):
        raise QuacP1FormalOuterError(
            "live systemd service identity drifted"
        )
    return LiveServiceReceipt(
        attestation_sha256=stable_hash(rows),
        invocation_id_sha256=hashlib.sha256(
            invocation.encode("ascii")
        ).hexdigest(),
        main_pid=os.getpid(),
        restart_count=0,
    )


def _verify_canary(
    config: SourceFreeFormalConfig,
) -> str:
    binding = config.canary_binding
    _direct_regular(
        binding.config_path,
        mode=0o400,
        field="canary config",
    )
    if (
        _sha256_file(binding.config_path)
        != binding.config_file_sha256
    ):
        raise QuacP1FormalOuterError(
            "canary config file binding drifted"
        )
    canary_config = canary.load_config(binding.config_path)
    if (
        canary_config.self_sha256 != binding.config_self_sha256
        or canary.runtime_bindings_payload(
            canary_config.runtime_bindings
        )
        != canary.runtime_bindings_payload(config.runtime_bindings)
    ):
        raise QuacP1FormalOuterError(
            "canary config semantic binding drifted"
        )
    canary_config.project_binding.verify()
    canary_config.asset_freeze_binding.verify(
        config.runtime_bindings
    )
    _direct_regular(
        binding.safe_terminal_path,
        mode=0o400,
        field="canary safe terminal",
    )
    try:
        raw = binding.safe_terminal_path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise QuacP1FormalOuterError(
            "canary safe terminal cannot be decoded"
        ) from exc
    body = {
        key: item
        for key, item in value.items()
        if key != "self_sha256"
    } if isinstance(value, dict) else {}
    expected_runtime_binding = runtime.stable_hash(
        config.runtime_bindings.semantic_payload()
    )
    if (
        type(value) is not dict
        or set(value) != _CANARY_SAFE_KEYS
        or raw != canary.canonical_bytes(value)
        or hashlib.sha256(raw).hexdigest()
        != binding.safe_terminal_file_sha256
        or value.get("self_sha256")
        != binding.safe_terminal_self_sha256
        or canary.stable_hash(body) != value.get("self_sha256")
        or value.get("schema") != canary.SAFE_TERMINAL_SCHEMA
        or value.get("status")
        != "passed_source_free_two_lane_single_index_canary"
        or value.get("study_id") != STUDY_ID
        or value.get("config_self_sha256")
        != binding.config_self_sha256
        or value.get("asset_freeze_self_sha256")
        != canary_config.asset_freeze_binding.self_sha256
        or value.get("effect_execution_design_self_sha256")
        != EXPECTED_DESIGN_SELF_SHA256
        or value.get("runtime_binding_sha256")
        != expected_runtime_binding
        or value.get("formal_source_access_count") != 0
        or value.get("API_or_online_evaluation_call_count") != 0
        or value.get("retry_replay_resample_or_fallback_count") != 0
        or value.get("source_path_loader_label_qrel_answer_input_count")
        != 0
    ):
        raise QuacP1FormalOuterError(
            "canary safe terminal binding drifted"
        )
    return binding.safe_terminal_self_sha256


def _verify_pre_source_bindings(
    config: SourceFreeFormalConfig,
) -> PreSourceReceipt:
    _verify_control_layout(config)
    if config.work_root.exists() or config.work_root.is_symlink():
        raise QuacP1FormalOuterError(
            "formal work root is not fresh"
        )
    if (
        config.outer_safe_terminal_path.exists()
        or config.outer_safe_terminal_path.is_symlink()
    ):
        raise QuacP1FormalOuterError(
            "outer terminal already exists"
        )
    try:
        project_metadata = _PROJECT_ROOT.lstat()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            "formal project root is unavailable"
        ) from exc
    if (
        _PROJECT_ROOT.is_symlink()
        or not stat.S_ISDIR(project_metadata.st_mode)
        or stat.S_IMODE(project_metadata.st_mode) != 0o500
        or project_metadata.st_uid != os.getuid()
    ):
        raise QuacP1FormalOuterError(
            "formal project root metadata drifted"
        )
    config.project_binding.verify()
    core_path = _PROJECT_ROOT / FORMAL_CORE_RELATIVE_PATH
    _direct_regular(core_path, mode=0o400, field="formal core")
    if _sha256_file(core_path) != config.core_binding.file_sha256:
        raise QuacP1FormalOuterError(
            "formal core binding mismatched"
        )
    design = _load_bound_json(
        config.design_binding,
        field="effect design",
        canonical_required=False,
    )
    if (
        design.get("self_sha256")
        != EXPECTED_DESIGN_SELF_SHA256
    ):
        raise QuacP1FormalOuterError(
            "effect design identity drifted"
        )
    _verify_incident(config)
    implementation_freeze = _verify_implementation_freeze(config)
    _verify_service_unit(config)
    canary_self = _verify_canary(config)
    runtime_binding_sha256 = runtime.stable_hash(
        config.runtime_bindings.semantic_payload()
    )
    execution_freeze = _verify_execution_freeze(
        config,
        runtime_binding_sha256=runtime_binding_sha256,
    )
    if (
        implementation_freeze.get("self_sha256")
        != config.implementation_freeze_binding.self_sha256
        or execution_freeze.get("self_sha256")
        != config.execution_freeze_binding.self_sha256
    ):
        raise QuacP1FormalOuterError(
            "formal freeze self binding drifted"
        )
    source_metadata = []
    for name, binding in (
        ("TRAIN", config.train),
        ("DEV", config.dev),
    ):
        metadata = _direct_regular(
            binding.path,
            mode=0o600,
            field=f"{name} source",
        )
        if (
            metadata.st_size != binding.size_bytes
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise QuacP1FormalOuterError(
                f"{name} source metadata binding drifted"
            )
        source_metadata.append(metadata)
    if (
        config.train.path == config.dev.path
        or (
            source_metadata[0].st_dev,
            source_metadata[0].st_ino,
        )
        == (
            source_metadata[1].st_dev,
            source_metadata[1].st_ino,
        )
    ):
        raise QuacP1FormalOuterError(
            "TRAIN and DEV source identities collide"
        )
    return PreSourceReceipt(
        project_tree_sha256=config.project_binding.tree_sha256,
        core_file_sha256=config.core_binding.file_sha256,
        design_self_sha256=config.design_binding.self_sha256,
        incident_self_sha256=config.incident_binding.self_sha256,
        implementation_freeze_self_sha256=(
            config.implementation_freeze_binding.self_sha256
        ),
        execution_freeze_self_sha256=(
            config.execution_freeze_binding.self_sha256
        ),
        service_unit_file_sha256=(
            config.service_unit_binding.file_sha256
        ),
        canary_terminal_self_sha256=canary_self,
        runtime_binding_sha256=runtime_binding_sha256,
    )


class _DuplicateJsonKey(ValueError):
    pass


def _strict_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_source_once(
    binding: SourceFileBinding,
    *,
    field: str,
    open_counts: dict[str, int],
) -> object:
    if not hasattr(os, "O_NOFOLLOW"):
        raise QuacP1FormalOuterError(
            "formal source read requires O_NOFOLLOW"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(binding.path, flags)
        open_counts[field] += 1
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != binding.size_bytes
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
        ):
            raise QuacP1FormalOuterError(
                f"{field} source descriptor binding drifted"
            )
        chunks: list[bytes] = []
        remaining = binding.size_bytes
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise QuacP1FormalOuterError(
                    f"{field} source ended early"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            raise QuacP1FormalOuterError(
                f"{field} source grew during its sole read"
            )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QuacP1FormalOuterError(
            f"{field} source cannot be opened exactly once"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        pathname_after = binding.path.lstat()
    except OSError as exc:
        raise QuacP1FormalOuterError(
            f"{field} source pathname disappeared after its sole read"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    )
    pathname_identity = (
        pathname_after.st_dev,
        pathname_after.st_ino,
        pathname_after.st_mode,
        pathname_after.st_uid,
        pathname_after.st_nlink,
        pathname_after.st_size,
        pathname_after.st_mtime_ns,
    )
    if (
        binding.path.is_symlink()
        or before_identity != after_identity
        or before_identity != pathname_identity
    ):
        raise QuacP1FormalOuterError(
            f"{field} source changed during its sole read"
        )
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != binding.sha256:
        raise QuacP1FormalOuterError(
            f"{field} source SHA-256 mismatched"
        )
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKey,
        ValueError,
    ) as exc:
        raise QuacP1FormalOuterError(
            f"{field} source strict decode failed"
        ) from exc


def _build_production_dependencies(
    config: SourceFreeFormalConfig,
    verified_bindings: object,
) -> tuple[object, object]:
    if not isinstance(
        verified_bindings,
        runtime.VerifiedRuntimeBindings,
    ):
        raise QuacP1FormalOuterError(
            "production requires a real verified runtime token"
        )
    encoder = runtime.LocalMiniLMGpu0Encoder(
        Path(config.runtime_bindings.minilm_asset.path)
    )
    official_lane = runtime.LocalOfficialGpu1Lane()
    try:
        executor = core_runner.BoundRuntimeExecutor(
            bindings=config.runtime_bindings,
            verified_bindings=verified_bindings,
            encoder=encoder,
            official_lane=official_lane,
            action_adapter=runtime.FrozenActionAdapter(),
        )
    except TypeError as exc:
        raise QuacP1FormalOuterError(
            "formal core executor has not adopted the verified-binding "
            "runtime interface"
        ) from exc
    return executor, core_runner.FrozenScientificOps()


def _safe_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        result = set(value)
        for item in value.values():
            result.update(_safe_keys(item))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_safe_keys(item))
        return result
    return set()


def _validate_core_terminal(
    result: object,
    *,
    config: SourceFreeFormalConfig,
) -> tuple[Mapping[str, object], str]:
    if not isinstance(result, core_runner.FormalRunResult):
        raise QuacP1FormalOuterError(
            "formal core result type drifted"
        )
    expected_path = config.work_root / core_runner.TERMINAL_FILENAME
    if result.terminal_path != expected_path:
        raise QuacP1FormalOuterError(
            "formal core terminal path drifted"
        )
    _direct_regular(
        expected_path,
        mode=0o400,
        field="formal core terminal",
    )
    try:
        raw = expected_path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise QuacP1FormalOuterError(
            "formal core terminal cannot be decoded"
        ) from exc
    if (
        type(value) is not dict
        or raw != core_runner.canonical_bytes(value)
        or dict(result.terminal) != value
        or value.get("schema") != core_runner.SAFE_TERMINAL_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("effect_design_self_sha256")
        != EXPECTED_DESIGN_SELF_SHA256
        or value.get("status")
        not in {
            "VALID_COMPLETE_PROMOTED_M_MEASURED",
            "VALID_NONPROMOTION_M_UNOPENED",
        }
        or value.get("API_or_online_evaluation_call_count") != 0
        or value.get("retry_replay_resample_repair_or_fallback_count")
        != 0
        or value.get("secret_generation_count") != 1
        or _safe_keys(value).intersection(_FORBIDDEN_SAFE_KEYS)
    ):
        raise QuacP1FormalOuterError(
            "formal core safe terminal semantics drifted"
        )
    supplied_self = _hex64(
        value.get("self_sha256"),
        "formal core terminal self",
    )
    body = {
        key: item
        for key, item in value.items()
        if key != "self_sha256"
    }
    if core_runner.stable_hash(body) != supplied_self:
        raise QuacP1FormalOuterError(
            "formal core terminal self hash drifted"
        )
    inner = value.get("inner_scientific_terminal")
    if (
        not isinstance(inner, Mapping)
        or inner.get("schema")
        != f"{controller.VERSION}_safe_terminal_v1"
        or inner.get("status") != value.get("status")
        or inner.get("study_id") != STUDY_ID
        or inner.get("execution_design_self_sha256")
        != EXPECTED_DESIGN_SELF_SHA256
        or inner.get("online_or_API_evaluation_count") != 0
        or inner.get("retry_replay_resample_repair_count") != 0
    ):
        raise QuacP1FormalOuterError(
            "inner scientific terminal semantics drifted"
        )
    inner_body = {
        key: item
        for key, item in inner.items()
        if key != "terminal_self_sha256"
    }
    if (
        inner.get("terminal_self_sha256")
        != controller.stable_hash(inner_body)
    ):
        raise QuacP1FormalOuterError(
            "inner scientific terminal self hash drifted"
        )
    return value, hashlib.sha256(raw).hexdigest()


def _claim_global_attempt(
    config: SourceFreeFormalConfig,
) -> str:
    if (
        config.global_attempt_marker_path.exists()
        or config.global_attempt_marker_path.is_symlink()
    ):
        raise QuacP1FormalOuterError(
            "global formal attempt already exists; retry is forbidden"
        )
    if (
        config.outer_safe_terminal_path.exists()
        or config.outer_safe_terminal_path.is_symlink()
    ):
        raise QuacP1FormalOuterError(
            "outer formal terminal already exists; retry is forbidden"
        )
    marker = _self_hashed(
        {
            "API_or_online_evaluation_authorized": False,
            "config_self_sha256": config.self_sha256,
            "formal_source_access_count_at_claim": 0,
            "incident_self_sha256": EXPECTED_INCIDENT_SELF_SHA256,
            "preformal_semantic_source_decode_count": 0,
            "prior_postqualification_hash_only_member_read_count": (
                PRIOR_HASH_ONLY_MEMBER_READ_COUNT
            ),
            "prior_postqualification_hash_only_operation_count": (
                PRIOR_HASH_ONLY_OPERATION_COUNT
            ),
            "retry_replay_resample_repair_or_fallback_authorized": False,
            "schema": GLOBAL_ATTEMPT_SCHEMA,
            "study_id": STUDY_ID,
            "work_root": str(config.work_root),
        }
    )
    return _write_once(config.global_attempt_marker_path, marker)


def _write_failure(
    *,
    config: SourceFreeFormalConfig,
    marker_file_sha256: str,
    stage: str,
    source_open_counts: Mapping[str, int],
    live_service_attestation_count: int,
    runtime_verification_count: int,
    core_invocation_count: int,
) -> None:
    if (
        config.outer_safe_terminal_path.exists()
        or config.outer_safe_terminal_path.is_symlink()
    ):
        return
    value = _self_hashed(
        {
            "API_or_online_evaluation_call_count": 0,
            "aggregate_only_public_receipt": True,
            "config_self_sha256": config.self_sha256,
            "core_invocation_count": core_invocation_count,
            "failure_code": "formal_outer_stage_failed",
            "failure_stage": stage,
            "formal_source_fd_open_counts": dict(
                sorted(source_open_counts.items())
            ),
            "global_attempt_marker_file_sha256": marker_file_sha256,
            "live_service_attestation_count": (
                live_service_attestation_count
            ),
            "preformal_semantic_source_decode_count": 0,
            "prior_postqualification_hash_only_member_read_count": (
                PRIOR_HASH_ONLY_MEMBER_READ_COUNT
            ),
            "prior_postqualification_hash_only_operation_count": (
                PRIOR_HASH_ONLY_OPERATION_COUNT
            ),
            "private_source_item_query_document_qrel_action_score_or_secret_values_in_this_safe_receipt": False,
            "retry_replay_resample_repair_or_fallback_count": 0,
            "runtime_full_tree_verification_count": (
                runtime_verification_count
            ),
            "schema": OUTER_FAILURE_SCHEMA,
            "status": "IMPLEMENTATION_OR_INFRASTRUCTURE_INVALID_NO_RETRY",
            "study_id": STUDY_ID,
        }
    )
    try:
        _write_once(config.outer_safe_terminal_path, value)
    except QuacP1FormalOuterError:
        return


def _run_with_dependencies(
    config: SourceFreeFormalConfig,
    *,
    pre_source_verifier: Callable[
        [SourceFreeFormalConfig], PreSourceReceipt
    ],
    live_service_attestor: Callable[
        [SourceFreeFormalConfig], LiveServiceReceipt
    ],
    runtime_verifier: Callable[..., object],
    dependency_builder: Callable[
        [SourceFreeFormalConfig, object], tuple[object, object]
    ],
    source_reader: Callable[..., object],
    core_callable: Callable[..., object],
) -> Mapping[str, object]:
    """Internal injectable test seam; production CLI never exposes it."""

    marker_file_sha256 = _claim_global_attempt(config)
    source_open_counts = {"DEV": 0, "TRAIN": 0}
    runtime_verification_count = 0
    live_service_attestation_count = 0
    core_invocation_count = 0
    stage = "verify_all_pre_source_frozen_bindings"
    try:
        pre_source = pre_source_verifier(config)
        if not isinstance(pre_source, PreSourceReceipt):
            raise QuacP1FormalOuterError(
                "pre-source receipt type drifted"
            )
        stage = "attest_live_one_shot_systemd_service"
        live_service_attestation_count += 1
        live_service = live_service_attestor(config)
        if (
            not isinstance(live_service, LiveServiceReceipt)
            or live_service_attestation_count != 1
        ):
            raise QuacP1FormalOuterError(
                "live service attestation receipt drifted"
            )
        stage = "verify_runtime_bindings_once_before_source"
        runtime_verification_count += 1
        verified_bindings = runtime_verifier(
            config.runtime_bindings,
            source_access_count=0,
        )
        if runtime_verification_count != 1:
            raise QuacP1FormalOuterError(
                "runtime binding verification count drifted"
            )
        executor, scientific_ops = dependency_builder(
            config,
            verified_bindings,
        )
        stage = "read_and_strict_decode_TRAIN_once"
        train_holder = [
            source_reader(
                config.train,
                field="TRAIN",
                open_counts=source_open_counts,
            )
        ]
        stage = "read_and_strict_decode_DEV_once"
        dev_holder = [
            source_reader(
                config.dev,
                field="DEV",
                open_counts=source_open_counts,
            )
        ]
        if source_open_counts != {"DEV": 1, "TRAIN": 1}:
            raise QuacP1FormalOuterError(
                "formal source read-once counts drifted"
            )
        stage = "invoke_exact_formal_core_after_all_freezes"
        core_invocation_count += 1
        core_result = core_callable(
            train_obj=train_holder.pop(),
            dev_obj=dev_holder.pop(),
            work_root=config.work_root,
            block_executor=executor,
            scientific_ops=scientific_ops,
        )
        core_terminal, core_file_sha256 = _validate_core_terminal(
            core_result,
            config=config,
        )
        stage = "write_aggregate_outer_terminal"
        terminal = _self_hashed(
            {
                "API_or_online_evaluation_call_count": 0,
                "aggregate_only_public_receipt": True,
                "canary_safe_terminal_self_sha256": (
                    pre_source.canary_terminal_self_sha256
                ),
                "config_self_sha256": config.self_sha256,
                "core_invocation_count": core_invocation_count,
                "core_safe_terminal_file_sha256": core_file_sha256,
                "core_safe_terminal_self_sha256": core_terminal[
                    "self_sha256"
                ],
                "effect_execution_design_self_sha256": (
                    pre_source.design_self_sha256
                ),
                "execution_freeze_self_sha256": (
                    pre_source.execution_freeze_self_sha256
                ),
                "formal_source_fd_open_counts": dict(
                    sorted(source_open_counts.items())
                ),
                "global_attempt_marker_file_sha256": (
                    marker_file_sha256
                ),
                "inner_valid_status": core_terminal["status"],
                "implementation_freeze_self_sha256": (
                    pre_source.implementation_freeze_self_sha256
                ),
                "incident_self_sha256": (
                    pre_source.incident_self_sha256
                ),
                "live_service_attestation_count": (
                    live_service_attestation_count
                ),
                "live_service_attestation_sha256": (
                    live_service.attestation_sha256
                ),
                "live_service_invocation_id_sha256": (
                    live_service.invocation_id_sha256
                ),
                "preformal_semantic_source_decode_count": 0,
                "prior_postqualification_hash_only_member_read_count": (
                    PRIOR_HASH_ONLY_MEMBER_READ_COUNT
                ),
                "prior_postqualification_hash_only_operation_count": (
                    PRIOR_HASH_ONLY_OPERATION_COUNT
                ),
                "private_source_item_query_document_qrel_action_score_or_secret_values_in_this_safe_receipt": False,
                "project_tree_sha256": (
                    pre_source.project_tree_sha256
                ),
                "retry_replay_resample_repair_or_fallback_count": 0,
                "runtime_binding_sha256": (
                    pre_source.runtime_binding_sha256
                ),
                "runtime_full_tree_verification_count": (
                    runtime_verification_count
                ),
                "schema": OUTER_TERMINAL_SCHEMA,
                "secret_generation_count": 1,
                "service_unit_file_sha256": (
                    pre_source.service_unit_file_sha256
                ),
                "source_file_sha256": {
                    "DEV": config.dev.sha256,
                    "TRAIN": config.train.sha256,
                },
                "status": core_terminal["status"],
                "study_id": STUDY_ID,
            }
        )
        _write_once(config.outer_safe_terminal_path, terminal)
        return terminal
    except BaseException:
        _write_failure(
            config=config,
            marker_file_sha256=marker_file_sha256,
            stage=stage,
            source_open_counts=source_open_counts,
            live_service_attestation_count=(
                live_service_attestation_count
            ),
            runtime_verification_count=runtime_verification_count,
            core_invocation_count=core_invocation_count,
        )
        raise


def run_formal_production(
    config: SourceFreeFormalConfig,
) -> Mapping[str, object]:
    """Run the non-injectable production closure exactly once."""

    if not isinstance(config, SourceFreeFormalConfig):
        raise QuacP1FormalOuterError(
            "SourceFreeFormalConfig is required"
        )
    return _run_with_dependencies(
        config,
        pre_source_verifier=_verify_pre_source_bindings,
        live_service_attestor=_verify_live_service_attestation,
        runtime_verifier=runtime.verify_runtime_bindings_once,
        dependency_builder=_build_production_dependencies,
        source_reader=_read_source_once,
        core_callable=core_runner.run_formal_once,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=VERSION,
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        run_formal_production(config)
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "EXPECTED_DESIGN_SELF_SHA256",
    "EXPECTED_DEV_SIZE_BYTES",
    "EXPECTED_DEV_SHA256",
    "EXPECTED_TRAIN_SIZE_BYTES",
    "EXPECTED_TRAIN_SHA256",
    "GLOBAL_ATTEMPT_SCHEMA",
    "OUTER_FAILURE_SCHEMA",
    "OUTER_TERMINAL_SCHEMA",
    "QuacP1FormalOuterError",
    "SourceFreeFormalConfig",
    "canonical_bytes",
    "load_config",
    "main",
    "parse_config",
    "run_formal_production",
    "stable_hash",
]
